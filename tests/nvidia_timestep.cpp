/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <new>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <meep.hpp>

#include "backend/backend.hpp"
#include "backend/descriptors.hpp"
#include "backend/diagnostics.hpp"
#include "backend/halo_plan.hpp"
#include "backend/initialization_plan.hpp"
#include "backend/lifecycle.hpp"
#include "backend/material_recipe.hpp"
#include "backend/nvidia/nvidia_backend.hpp"
#include "backend/nvidia/nvidia_graph.hpp"
#include "backend/nvidia/nvidia_initialization.hpp"
#include "backend/nvidia/nvidia_materials.hpp"
#include "backend/nvidia/nvidia_polarization.hpp"
#include "backend/nvidia/runtime.hpp"
#include "backend/precision.hpp"
#include "backend/prepare.hpp"
#include "backend/step_plan.hpp"
#include "backend/storage_plan.hpp"
#include "material_data.hpp"
#include "meep_internals.hpp"
#include "meepgeom.hpp"

using namespace meep;

/* The host-fallback steady-state gate needs to observe ordinary C++ heap
   allocation, not only CUDA device/pinned allocation accounting. Keep this
   disabled outside the narrow warmed-up advance window below. */
static std::atomic<bool> count_heap_allocations(false);
static std::atomic<size_t> heap_allocation_calls(0);

void *operator new(std::size_t bytes) {
  void *result = malloc(bytes ? bytes : 1);
  if (!result) throw std::bad_alloc();
  if (count_heap_allocations.load(std::memory_order_relaxed)) {
    heap_allocation_calls.fetch_add(1, std::memory_order_relaxed);
  }
  return result;
}

void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *pointer) noexcept { free(pointer); }
void operator delete[](void *pointer) noexcept { free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { free(pointer); }
void operator delete[](void *pointer, std::size_t) noexcept { free(pointer); }

static void require(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message);
    meep::abort("nvidia_timestep failed");
  }
}

static void require_selected_graph_mode(fields &f) {
  if (!getenv("MEEP_NVIDIA_GRAPH_ASSERT")) return;
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(f.backend);
  const NvidiaGraphStatistics stats =
      backend ? backend->graph_statistics_for_testing() : NvidiaGraphStatistics();
  const bool expected = getenv("MEEP_NVIDIA_GRAPH_EXPECT_ENABLED") != NULL;
  require(backend && stats.valid && stats.enabled == expected,
          "NVIDIA focused fixture did not retain the requested graph mode");
}

#if MEEP_SINGLE
static uint32_t float_bits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}
#endif

static void test_polarization_coefficient_rounding() {
  const realnum dt = realnum(1.0 / 12.0);
  const nvidia::polarization_coefficients omega =
      nvidia::derive_polarization_coefficients(0.28, 0.09, double(dt), false);
  const realnum omega2pi = 2 * pi * 0.28;
  const realnum expected_omega0dtsqr = omega2pi * omega2pi * dt * dt;
  require(omega.omega0dtsqr == double(expected_omega0dtsqr) &&
              omega.omega0dtsqr_denom == double(expected_omega0dtsqr),
          "NVIDIA polarization omega coefficient did not preserve host realnum rounding");

  const nvidia::polarization_coefficients damping =
      nvidia::derive_polarization_coefficients(1.17, 0.035, double(dt), false);
  const realnum g2pi = 0.035 * 2 * pi;
  const realnum expected_gamma1inv = 1 / (1 + g2pi * dt / 2);
  const realnum expected_gamma1 = 1 - g2pi * dt / 2;
  require(damping.gamma1inv == double(expected_gamma1inv) &&
              damping.gamma1 == double(expected_gamma1),
          "NVIDIA polarization damping coefficients did not preserve host realnum rounding");

  const realnum current = realnum(0.37), previous = realnum(-0.19), forcing = realnum(0.23);
  const realnum expected_next =
      expected_gamma1inv *
      (current * (2 - realnum(damping.omega0dtsqr_denom)) - expected_gamma1 * previous +
       realnum(damping.omega0dtsqr) * forcing);
  const realnum launch_next =
      realnum(damping.gamma1inv) *
      (current * (2 - realnum(damping.omega0dtsqr_denom)) -
       realnum(damping.gamma1) * previous + realnum(damping.omega0dtsqr) * forcing);
  require(memcmp(&expected_next, &launch_next, sizeof(realnum)) == 0,
          "NVIDIA polarization launch coefficients changed the host recurrence");

#if MEEP_SINGLE
  require(float_bits(realnum(omega.omega0dtsqr)) == 0x3cb013c8u,
          "native-single omega coefficient regressed by an ULP");
  require(float_bits(realnum(damping.gamma1inv)) == 0x3f7dacf2u,
          "native-single damping coefficient regressed by an ULP");
#endif

  const double tensor[3][3] = {{0.0, 0.31000000238418579, 0.23000000417232513},
                               {-0.31000000238418579, 0.0, 0.17000000178813934},
                               {-0.23000000417232513, -0.17000000178813934, 0.0}};
  const direction order[3] = {Z, X, Y};
  const nvidia::gyrotropic_coefficients gyro_lorentz =
      nvidia::derive_gyrotropic_coefficients(0.73, 0.06, 0.19, tensor, double(dt),
                                             GYROTROPIC_LORENTZIAN, order);
  const realnum gyro_omega = 2 * pi * realnum(0.73) * dt;
  const realnum gyro_gamma = 2 * pi * realnum(0.06) * dt;
  const realnum gyro_a = gyro_omega * gyro_omega;
  const realnum gyro_pt = pi * dt;
  const realnum gyro_gd = 1 + gyro_gamma / 2;
  const realnum gyro_gx = gyro_pt * realnum(tensor[Y][Z]);
  const realnum gyro_gy = gyro_pt * realnum(tensor[Z][X]);
  const realnum gyro_gz = gyro_pt * realnum(tensor[X][Y]);
  const realnum gyro_invdet =
      1.0 / gyro_gd /
      (gyro_gd * gyro_gd + gyro_gx * gyro_gx + gyro_gy * gyro_gy + gyro_gz * gyro_gz);
  require(gyro_lorentz.omega0dtsqr == double(gyro_a) &&
              gyro_lorentz.gamma1 == double(realnum(1 - gyro_gamma / 2)) &&
              gyro_lorentz.pt == double(gyro_pt) &&
              gyro_lorentz.inverse[0][0] ==
                  double(gyro_invdet * (gyro_gd * gyro_gd + gyro_gz * gyro_gz)),
          "NVIDIA gyrotropic Lorentzian coefficients changed host realnum rounding");

  const nvidia::gyrotropic_coefficients gyro_saturated =
      nvidia::derive_gyrotropic_coefficients(0.73, 0.06, 0.19, tensor, double(dt),
                                             GYROTROPIC_SATURATED, order);
  const realnum sat_alpha = realnum(0.19), sat_gd = 0.5;
  const realnum sat_gx = -0.5 * sat_alpha * realnum(tensor[Y][Z]);
  const realnum sat_gy = -0.5 * sat_alpha * realnum(tensor[Z][X]);
  const realnum sat_gz = -0.5 * sat_alpha * realnum(tensor[X][Y]);
  const realnum sat_invdet =
      1.0 / sat_gd /
      (sat_gd * sat_gd + sat_gx * sat_gx + sat_gy * sat_gy + sat_gz * sat_gz);
  require(gyro_saturated.alpha == double(sat_alpha) &&
              gyro_saturated.dt2pi == double(realnum(2 * pi * dt)) &&
              gyro_saturated.inverse[0][0] ==
                  double(sat_invdet * (sat_gd * sat_gd + sat_gz * sat_gz)),
          "NVIDIA saturated gyrotropic coefficients changed host realnum rounding");
#if MEEP_SINGLE
  require(float_bits(realnum(gyro_lorentz.omega0dtsqr)) == 0x3e159a9au &&
              float_bits(realnum(gyro_lorentz.inverse[0][0])) == 0x3f7aafefu,
          "native-single gyrotropic Lorentzian coefficient bits regressed");
  require(float_bits(realnum(gyro_saturated.dt2pi)) == 0x3f060a92u &&
              float_bits(realnum(gyro_saturated.inverse[0][0])) == 0x3fff3fb5u,
          "native-single saturated gyrotropic coefficient bits regressed");
#endif
}

static void test_polarization_storage_key_encoding() {
  const uint64_t electric = polarization_storage_aux(E_stuff, 7, 1025);
  const uint64_t magnetic = polarization_storage_aux(H_stuff, 7, 1025);
  const uint64_t later = polarization_storage_aux(E_stuff, 8, 1);
  require(electric != magnetic && electric != later && magnetic != later,
          "polarization storage keys collide across field families or state ordinals");
  require(polarization_storage_field_type(electric) == E_stuff &&
              polarization_storage_field_type(magnetic) == H_stuff &&
              polarization_storage_state_index(electric) == 7 &&
              polarization_storage_state_index(magnetic) == 7 &&
              polarization_storage_state_index(later) == 8 &&
              polarization_storage_layout_ordinal(electric) == 1025 &&
              polarization_storage_layout_ordinal(magnetic) == 1025 &&
              polarization_storage_layout_ordinal(later) == 1,
          "polarization storage key decoding lost field, state, or 32-bit layout ordinal");
}

static double isotropic_eps(const vec &p) { return p.x() < 0.0 ? 2.0 : 3.0; }
static double uniform_conductivity(const vec &) { return 0.17; }
static double phase_target_conductivity(const vec &) { return 0.29; }
static double unit_value(const vec &) { return 1.0; }
static double zero_value(const vec &) { return 0.0; }
static double chi2_value(const vec &) { return 0.03125; }
static double chi3_value(const vec &) { return 0.0625; }

class linear_anisotropic_material : public material_function {
public:
  explicit linear_anisotropic_material(bool full, bool magnetic = false)
      : full_(full), magnetic_(magnetic) {}

  bool has_mu() override { return magnetic_; }

  void eff_chi1inv_row(component c, double row[3], const volume &, double, int) override {
    row[0] = row[1] = row[2] = 0.0;
    const int d = component_index(c);
    row[d] = 0.5 + 0.05 * d;
    if (full_) {
      static const double offdiagonal[3][3] = {
          {0.0, 0.03, -0.02}, {0.03, 0.0, 0.04}, {-0.02, 0.04, 0.0}};
      for (int other = 0; other < 3; ++other)
        if (other != d) row[other] = offdiagonal[d][other];
    }
    else {
      if (d == X) row[Z] = 0.04;
      if (d == Z) row[X] = 0.04;
    }
  }

private:
  bool full_;
  bool magnetic_;
};

class dispersion_sigma_material : public material_function {
public:
  explicit dispersion_sigma_material(bool anisotropic) : anisotropic_(anisotropic) {}

  void sigma_row(component c, double row[3], const vec &) override {
    row[0] = row[1] = row[2] = 0.0;
    const int d = component_index(c);
    row[d] = 0.08 + 0.01 * d;
    if (!anisotropic_) return;
    static const double offdiagonal[3][3] = {
        {0.0, 0.004, -0.003}, {0.004, 0.0, 0.005}, {-0.003, 0.005, 0.0}};
    for (int other = 0; other < 3; ++other)
      if (other != d) row[other] = offdiagonal[d][other];
  }

private:
  bool anisotropic_;
};

class inherited_noisy_lorentzian : public noisy_lorentzian_susceptibility {
public:
  inherited_noisy_lorentzian(realnum amplitude, realnum omega, realnum gamma)
      : noisy_lorentzian_susceptibility(amplitude, omega, gamma) {}
  susceptibility *clone() const override { return new inherited_noisy_lorentzian(*this); }
};

class inherited_lorentzian : public lorentzian_susceptibility {
public:
  inherited_lorentzian(realnum omega, realnum gamma, bool drude = false)
      : lorentzian_susceptibility(omega, gamma, drude) {}
  susceptibility *clone() const override { return new inherited_lorentzian(*this); }
};

class opaque_inherited_lorentzian : public lorentzian_susceptibility {
public:
  opaque_inherited_lorentzian(realnum omega, realnum gamma)
      : lorentzian_susceptibility(omega, gamma) {}
  susceptibility *clone() const override { return new opaque_inherited_lorentzian(*this); }
  bool internal_layout(std::vector<InternalArrayLayout> &out, const grid_volume &,
                       void *) const override {
    out.clear();
    return false;
  }
};

struct host_callback_trace {
  std::vector<char> events;
};

class counted_inherited_lorentzian : public lorentzian_susceptibility {
public:
  counted_inherited_lorentzian(realnum omega, realnum gamma, host_callback_trace *trace)
      : lorentzian_susceptibility(omega, gamma), trace_(trace) {}
  susceptibility *clone() const override { return new counted_inherited_lorentzian(*this); }
  void update_P(realnum *W[NUM_FIELD_COMPONENTS][2],
                realnum *W_prev[NUM_FIELD_COMPONENTS][2], realnum dt, const grid_volume &gv,
                void *data) const override {
    trace_->events.push_back('U');
    lorentzian_susceptibility::update_P(W, W_prev, dt, gv, data);
  }
  void subtract_P(field_type ft, realnum *f_minus_p[NUM_FIELD_COMPONENTS][2],
                  void *data) const override {
    trace_->events.push_back('S');
    lorentzian_susceptibility::subtract_P(ft, f_minus_p, data);
  }

private:
  host_callback_trace *trace_;
};

class inherited_multilevel : public multilevel_susceptibility {
public:
  inherited_multilevel(int levels, int transitions, const realnum *Gamma, const realnum *N0,
                       const realnum *alpha, const realnum *omega, const realnum *gamma,
                       const realnum *sigmat)
      : multilevel_susceptibility(levels, transitions, Gamma, N0, alpha, omega, gamma,
                                  sigmat) {}
  susceptibility *clone() const override { return new inherited_multilevel(*this); }
};

static realnum initial_value(size_t array, size_t element) {
  return realnum(0.02 * sin(double(17 * array + element)) +
                 0.01 * cos(double(3 * array + 5 * element)));
}

static void round_real_arrays(CpuArrayCatalog &catalog) {
  for (size_t i = 0; i < catalog.size(); ++i) {
    const ArraySpec &spec = catalog.spec(ArrayId{uint32_t(i)});
    if (is_valid(spec.alias_of) || spec.element_type != ElementType::realnum_value) continue;
    realnum *values = catalog.resolve<realnum>(spec.id);
    for (size_t j = 0; j < spec.elements; ++j)
      values[j] = realnum(float(values[j]));
  }
}

static void initialize_fields(fields &cpu, fields &gpu, bool f32, double scale = 1.0) {
  require(cpu.array_catalog->size() == gpu.array_catalog->size(),
          "CPU and NVIDIA catalogs have different sizes");
  for (size_t i = 0; i < cpu.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &cpu_spec = cpu.array_catalog->spec(id);
    const ArraySpec &gpu_spec = gpu.array_catalog->spec(id);
    require(cpu.array_catalog->key(id) == gpu.array_catalog->key(id),
            "CPU and NVIDIA catalog keys differ");
    require(cpu_spec.elements == gpu_spec.elements && cpu_spec.alias_of == gpu_spec.alias_of,
            "CPU and NVIDIA catalog layouts differ");
    if (is_valid(cpu_spec.alias_of) ||
        (cpu_spec.role != array_role::field && cpu_spec.role != array_role::polarization) ||
        cpu_spec.element_type != ElementType::realnum_value)
      continue;

    std::vector<realnum> values(cpu_spec.elements);
    for (size_t j = 0; j < values.size(); ++j) {
      values[j] = realnum(scale * initial_value(i, j));
      if (f32) values[j] = realnum(float(values[j]));
    }
    memcpy(cpu.array_catalog->resolve_untyped(id), values.data(), values.size() * sizeof(realnum));
    gpu.backend->write(ArrayRef{id, 0, values.size()}, values.data(),
                       values.size() * sizeof(realnum));
  }
}

static void initialize_multilevel_fields(fields &cpu, fields &gpu, bool f32,
                                         double scale = 1.0) {
  require(cpu.array_catalog->size() == gpu.array_catalog->size(),
          "CPU and NVIDIA multilevel catalogs have different sizes");
  for (size_t i = 0; i < cpu.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &cpu_spec = cpu.array_catalog->spec(id);
    const StorageKey &key = cpu.array_catalog->key(id);
    const bool gamma_inv =
        key.kind == int(array_kind::polarization_internal) &&
        polarization_storage_layout_ordinal(key.aux) == 0;
    if (is_valid(cpu_spec.alias_of) || gamma_inv ||
        (cpu_spec.role != array_role::field && cpu_spec.role != array_role::polarization) ||
        cpu_spec.element_type != ElementType::realnum_value)
      continue;
    std::vector<realnum> values(cpu_spec.elements);
    for (size_t j = 0; j < values.size(); ++j) {
      values[j] = realnum(scale * initial_value(i, j));
      if (f32) values[j] = realnum(float(values[j]));
    }
    memcpy(cpu.array_catalog->resolve_untyped(id), values.data(), values.size() * sizeof(realnum));
    gpu.backend->write(ArrayRef{id, 0, values.size()}, values.data(),
                       values.size() * sizeof(realnum));
  }
}

static void compare_fields(fields &cpu, fields &gpu, double tolerance) {
  for (size_t i = 0; i < cpu.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = cpu.array_catalog->spec(id);
    if (is_valid(spec.alias_of) ||
        (spec.role != array_role::field && spec.role != array_role::polarization) ||
        spec.element_type != ElementType::realnum_value)
      continue;
    const realnum *expected = cpu.array_catalog->resolve<realnum>(id);
    std::vector<realnum> observed(spec.elements);
    gpu.backend->read(ArrayRef{id, 0, spec.elements}, observed.data(),
                      observed.size() * sizeof(realnum));
    for (size_t j = 0; j < observed.size(); ++j) {
      const double error = fabs(double(observed[j]) - double(expected[j]));
      const double scale = 1.0 + fabs(double(expected[j]));
      if (std::isnan(double(expected[j])) || std::isnan(double(observed[j])) ||
          ((!std::isfinite(double(expected[j])) || !std::isfinite(double(observed[j]))) &&
           double(expected[j]) != double(observed[j])) ||
          (std::isfinite(double(expected[j])) && std::isfinite(double(observed[j])) &&
           error > tolerance * scale)) {
        const StorageKey &key = cpu.array_catalog->key(id);
        fprintf(stderr,
                "array %zu (%s,c=%d,cmp=%d,aux=%llu) element %zu differs: cpu=%.17g "
                "nvidia=%.17g error=%.3g tol=%.3g\n",
                i, array_kind_name(array_kind(key.kind)), key.component_, key.cmp,
                static_cast<unsigned long long>(key.aux), j,
                double(expected[j]), double(observed[j]), error, tolerance * scale);
        meep::abort("NVIDIA timestep differs from CPU");
      }
    }
  }
}

static bool is_magnetic_backup_kind(int kind) {
  return kind == int(array_kind::f_backup) || kind == int(array_kind::f_u_backup) ||
         kind == int(array_kind::f_w_backup) || kind == int(array_kind::f_cond_backup) ||
         kind == int(array_kind::f_bfast_backup);
}

/* CPU magnetic synchronization may retain lazy legacy backup allocations that
   a resident backend deliberately never catalogs.  Compare the physical live
   state by semantic StorageKey rather than coincidental ArrayId numbering. */
static void compare_live_fields_by_key(fields &cpu, fields &gpu, double tolerance) {
  for (size_t i = 0; i < cpu.array_catalog->size(); ++i) {
    const ArrayId cpu_id{uint32_t(i)};
    const ArraySpec &cpu_spec = cpu.array_catalog->spec(cpu_id);
    const StorageKey &key = cpu.array_catalog->key(cpu_id);
    if (is_valid(cpu_spec.alias_of) || is_magnetic_backup_kind(key.kind) ||
        (cpu_spec.role != array_role::field && cpu_spec.role != array_role::polarization) ||
        cpu_spec.element_type != ElementType::realnum_value)
      continue;
    const ArrayId gpu_id = gpu.array_catalog->find(key);
    require(is_valid(gpu_id), "NVIDIA live catalog is missing a CPU field key");
    const ArraySpec &gpu_spec = gpu.array_catalog->spec(gpu_id);
    require(gpu_spec.elements == cpu_spec.elements,
            "NVIDIA live catalog field extent differs from CPU");
    const realnum *expected = cpu.array_catalog->resolve<realnum>(cpu_id);
    std::vector<realnum> observed(cpu_spec.elements);
    gpu.backend->read(ArrayRef{gpu_id, 0, gpu_spec.elements}, observed.data(),
                      observed.size() * sizeof(realnum));
    for (size_t j = 0; j < observed.size(); ++j) {
      const double error = fabs(double(observed[j]) - double(expected[j]));
      const double scale = 1.0 + fabs(double(expected[j]));
      if (std::isnan(double(expected[j])) || std::isnan(double(observed[j])) ||
          ((!std::isfinite(double(expected[j])) || !std::isfinite(double(observed[j]))) &&
           double(expected[j]) != double(observed[j])) ||
          (std::isfinite(double(expected[j])) && std::isfinite(double(observed[j])) &&
           error > tolerance * scale)) {
        fprintf(stderr,
                "live array (%s,c=%d,cmp=%d,aux=%llu) element %zu differs: cpu=%.17g "
                "nvidia=%.17g error=%.3g tol=%.3g\n",
                array_kind_name(array_kind(key.kind)), key.component_, key.cmp,
                static_cast<unsigned long long>(key.aux), j,
                double(expected[j]), double(observed[j]), error, tolerance * scale);
        meep::abort("NVIDIA live timestep state differs from CPU");
      }
    }
  }
}

static void initialize_live_fields_by_key(fields &cpu, fields &gpu, bool f32,
                                          double scale) {
  for (size_t i = 0; i < cpu.array_catalog->size(); ++i) {
    const ArrayId cpu_id{uint32_t(i)};
    const ArraySpec &cpu_spec = cpu.array_catalog->spec(cpu_id);
    const StorageKey &key = cpu.array_catalog->key(cpu_id);
    if (is_valid(cpu_spec.alias_of) || is_magnetic_backup_kind(key.kind) ||
        (cpu_spec.role != array_role::field && cpu_spec.role != array_role::polarization) ||
        cpu_spec.element_type != ElementType::realnum_value)
      continue;
    const ArrayId gpu_id = gpu.array_catalog->find(key);
    require(is_valid(gpu_id), "NVIDIA live catalog is missing an initialized CPU field key");
    const ArraySpec &gpu_spec = gpu.array_catalog->spec(gpu_id);
    require(gpu_spec.elements == cpu_spec.elements,
            "NVIDIA initialized field extent differs from CPU");
    std::vector<realnum> values(cpu_spec.elements);
    for (size_t j = 0; j < values.size(); ++j) {
      values[j] = realnum(scale * initial_value(i, j));
      if (f32) values[j] = realnum(float(values[j]));
    }
    memcpy(cpu.array_catalog->resolve_untyped(cpu_id), values.data(),
           values.size() * sizeof(realnum));
    gpu.backend->write(ArrayRef{gpu_id, 0, values.size()}, values.data(),
                       values.size() * sizeof(realnum));
  }
}

static void compare_material_rows(fields &cpu, fields &gpu, double tolerance) {
  for (size_t i = 0; i < cpu.array_catalog->size(); ++i) {
    const ArrayId cpu_id{uint32_t(i)};
    const ArraySpec &cpu_spec = cpu.array_catalog->spec(cpu_id);
    if (is_valid(cpu_spec.alias_of) || cpu_spec.role != array_role::material ||
        cpu_spec.element_type != ElementType::realnum_value)
      continue;
    const StorageKey &key = cpu.array_catalog->key(cpu_id);
    const array_kind kind = array_kind(key.kind);
    if (kind != array_kind::chi1inv && kind != array_kind::conductivity &&
        kind != array_kind::condinv)
      continue;
    const ArrayId gpu_id = gpu.array_catalog->find(key);
    require(is_valid(gpu_id), "NVIDIA catalog is missing a current material row");
    const ArraySpec &gpu_spec = gpu.array_catalog->spec(gpu_id);
    require(gpu_spec.elements == cpu_spec.elements,
            "NVIDIA current material row has the wrong extent");
    const realnum *expected = cpu.array_catalog->resolve<realnum>(cpu_id);
    std::vector<realnum> observed(cpu_spec.elements);
    gpu.backend->read(ArrayRef{gpu_id, 0, gpu_spec.elements}, observed.data(),
                      observed.size() * sizeof(realnum));
    for (size_t j = 0; j < observed.size(); ++j) {
      const double error = fabs(double(observed[j]) - double(expected[j]));
      const double scale = 1.0 + fabs(double(expected[j]));
      if (std::isnan(double(expected[j])) || std::isnan(double(observed[j])) ||
          ((!std::isfinite(double(expected[j])) || !std::isfinite(double(observed[j]))) &&
           double(expected[j]) != double(observed[j])) ||
          (std::isfinite(double(expected[j])) && std::isfinite(double(observed[j])) &&
           error > tolerance * scale)) {
        fprintf(stderr,
                "material row (%s,c=%d,aux=%llu) element %zu differs: cpu=%.17g "
                "nvidia=%.17g error=%.3g tol=%.3g\n",
                array_kind_name(kind), key.component_,
                static_cast<unsigned long long>(key.aux), j, double(expected[j]),
                double(observed[j]), error, tolerance * scale);
        meep::abort("NVIDIA material phase differs from CPU");
      }
    }
  }
}

static void compare_all_initialized_material_rows(fields &cpu, fields &gpu,
                                                  double tolerance) {
  for (size_t i = 0; i < cpu.array_catalog->size(); ++i) {
    const ArrayId cpu_id{uint32_t(i)};
    const ArraySpec &cpu_spec = cpu.array_catalog->spec(cpu_id);
    if (is_valid(cpu_spec.alias_of) || cpu_spec.role != array_role::material ||
        cpu_spec.element_type != ElementType::realnum_value)
      continue;
    const StorageKey &key = cpu.array_catalog->key(cpu_id);
    const ArrayId gpu_id = gpu.array_catalog->find(key);
    require(is_valid(gpu_id), "NVIDIA native initialization omitted a retained material row");
    const ArraySpec &gpu_spec = gpu.array_catalog->spec(gpu_id);
    require(gpu_spec.elements == cpu_spec.elements,
            "NVIDIA native initialized material extent differs from CPU");
    const realnum *expected = cpu.array_catalog->resolve<realnum>(cpu_id);
    std::vector<realnum> observed(cpu_spec.elements);
    gpu.backend->read(ArrayRef{gpu_id, 0, gpu_spec.elements}, observed.data(),
                      observed.size() * sizeof(realnum));
    for (size_t j = 0; j < observed.size(); ++j) {
      const double error = fabs(double(observed[j]) - double(expected[j]));
      const double scale = 1.0 + fabs(double(expected[j]));
      if (std::isnan(double(expected[j])) || std::isnan(double(observed[j])) ||
          ((!std::isfinite(double(expected[j])) || !std::isfinite(double(observed[j]))) &&
           double(expected[j]) != double(observed[j])) ||
          (std::isfinite(double(expected[j])) && std::isfinite(double(observed[j])) &&
           error > tolerance * scale)) {
        fprintf(stderr,
                "native material row (%s,c=%d,cmp=%d,aux=%llu) element %zu differs: "
                "cpu=%.17g nvidia=%.17g\n",
                array_kind_name(array_kind(key.kind)), key.component_, key.cmp,
                static_cast<unsigned long long>(key.aux), j, double(expected[j]),
                double(observed[j]));
        meep::abort("NVIDIA native material initialization differs from CPU");
      }
    }
  }
}

static double max_field_difference(fields &cpu, fields &gpu, size_t &array, size_t &element,
                                   double &absolute) {
  double maximum = 0.0;
  absolute = 0.0;
  array = element = 0;
  for (size_t i = 0; i < cpu.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = cpu.array_catalog->spec(id);
    if (is_valid(spec.alias_of) ||
        (spec.role != array_role::field && spec.role != array_role::polarization) ||
        spec.element_type != ElementType::realnum_value)
      continue;
    const realnum *expected = cpu.array_catalog->resolve<realnum>(id);
    std::vector<realnum> observed(spec.elements);
    gpu.backend->read(ArrayRef{id, 0, spec.elements}, observed.data(),
                      observed.size() * sizeof(realnum));
    for (size_t j = 0; j < observed.size(); ++j) {
      const double error = fabs(double(observed[j]) - double(expected[j]));
      const double relative = error / (1.0 + fabs(double(expected[j])));
      if (relative > maximum) {
        maximum = relative;
        absolute = error;
        array = i;
        element = j;
      }
    }
  }
  return maximum;
}

static void require_polarization_plan(fields &f, field_type ft, bool drude,
                                      bool anisotropic, unsigned int required_halo_phases,
                                      size_t minimum_states) {
  require(f.descriptors && f.array_catalog && f.halos,
          "dispersion test has no prepared backend-neutral state");
  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  size_t states = 0, updates = 0, subtractions = 0;
  bool saw_negative_stride = false;
  bool saw_requested_model = false;
  for (size_t i = 0; i < f.descriptors->polarizations.size(); ++i) {
    const PolarizationDescriptor &d = f.descriptors->polarizations[i];
    if (d.ft != ft || d.kind != SusceptibilityKind::lorentzian) continue;
    saw_requested_model |= d.lorentzian.drude == drude;
    states += d.lorentzian_states.size();
  }
  for (size_t i = 0; i < plan.polarization_updates.size(); ++i) {
    const PolarizationUpdate &update = plan.polarization_updates[i];
    if (is_electric(update.region.c) != (ft == E_stuff)) continue;
    ++updates;
    const unsigned offdiagonals =
        (update.region.variant_key & polarization_two_offdiagonals)
            ? 2
            : unsigned((update.region.variant_key & polarization_one_offdiagonal) != 0);
    if (anisotropic)
      require(offdiagonals > 0, "anisotropic dispersion row lost its cross operands");
    saw_negative_stride |= update.primary_stride < 0 || update.cross_stride1 < 0 ||
                           update.cross_stride2 < 0;
  }
  for (size_t i = 0; i < plan.polarization_subtractions.size(); ++i)
    if (is_electric(plan.polarization_subtractions[i].c) == (ft == E_stuff)) ++subtractions;
  require(saw_requested_model, "requested Lorentz/Drude descriptor was not prepared");
  require(states >= minimum_states && updates >= minimum_states && subtractions >= minimum_states,
          "dispersion plan lacks recurrence or subtraction rows");
  if (ft == H_stuff)
    require(saw_negative_stride, "magnetic dispersion row lacks negative strides");

  unsigned int observed = 0;
  for (size_t i = 0; i < f.halos->plans.size(); ++i) {
    const HaloPlan &halo = f.halos->plans[i];
    if (halo.ft == (ft == E_stuff ? PE_stuff : PH_stuff) && halo.same_rank &&
        halo.block_elements)
      observed |= 1u << unsigned(halo.phase);
  }
  require((observed & required_halo_phases) == required_halo_phases,
          "required same-rank polarization halo transform was not prepared");
}

static void require_gyrotropic_plan(fields &f, field_type ft, gyrotropy_model model,
                                    unsigned int required_w_halo_phases) {
  require(f.descriptors && f.array_catalog && f.halos,
          "gyrotropic test has no prepared backend-neutral state");
  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  size_t states = 0, updates = 0, subtractions = 0, p_halo_elements = 0;
  bool saw_model = false, saw_negative_stride = false;
  for (const PolarizationDescriptor &descriptor : f.descriptors->polarizations) {
    if (descriptor.ft != ft || descriptor.kind != SusceptibilityKind::gyrotropic) continue;
    saw_model |= descriptor.gyrotropic.model == model;
    states += descriptor.gyrotropic_states.size();
    for (const GyrotropicStateArrays &state : descriptor.gyrotropic_states)
      for (int dd = 0; dd < 3; ++dd)
        require(is_valid(state.p[dd]) && is_valid(state.p_prev[dd]) &&
                    state.p[dd] != state.p_prev[dd],
                "gyrotropic descriptor lacks six distinct state arrays");
  }
  for (const PolarizationUpdate &update : plan.polarization_updates) {
    if (update.kind != PolarizationUpdateKind::gyrotropic ||
        is_electric(update.region.c) != (ft == E_stuff))
      continue;
    ++updates;
    require(update.gyro_model == model && update.region.variant_key == 0,
            "gyrotropic update lost its model or gained Lorentzian variants");
    require(is_valid(update.p) && is_valid(update.p_prev) && is_valid(update.p_cross1) &&
                is_valid(update.p_prev_cross1) && is_valid(update.p_cross2) &&
                is_valid(update.p_prev_cross2) && is_valid(update.primary_w) &&
                is_valid(update.diagonal_sigma),
            "gyrotropic update lacks a required operand");
    saw_negative_stride |= update.primary_stride < 0 && update.cross_stride1 < 0 &&
                           update.cross_stride2 < 0;
  }
  for (const PolarizationSubtraction &subtraction : plan.polarization_subtractions)
    if (is_electric(subtraction.c) == (ft == E_stuff)) ++subtractions;
  require(saw_model && states && updates >= states && subtractions >= states,
          "gyrotropic plan lacks recurrence or ordered subtraction rows");
  if (ft == H_stuff)
    require(saw_negative_stride, "magnetic gyrotropic plan lacks negative strides");

  unsigned int observed_w = 0;
  for (const HaloPlan &halo : f.halos->plans) {
    if ((halo.ft == WE_stuff || halo.ft == WH_stuff) && halo.same_rank && halo.block_elements)
      observed_w |= 1u << unsigned(halo.phase);
    if (halo.ft == PE_stuff || halo.ft == PH_stuff) p_halo_elements += halo.block_elements;
  }
  require((observed_w & required_w_halo_phases) == required_w_halo_phases,
          "required gyrotropic W halo transform was not prepared");
  require(p_halo_elements == 0, "gyrotropic state incorrectly created PE/PH P halos");
}

static void require_halo_phases(const fields &f, unsigned int required) {
  unsigned int observed = 0;
  for (size_t i = 0; i < f.halos->plans.size(); ++i) {
    const HaloPlan &plan = f.halos->plans[i];
    if (plan.same_rank && plan.block_elements) observed |= 1u << unsigned(plan.phase);
  }
  require((observed & required) == required, "required same-rank halo transform was not prepared");
}

static void require_physics_variants(const StepPlan &plan, unsigned int required_curl_combinations,
                                     unsigned int required_constitutive_combinations) {
  unsigned int observed_curl_combinations = 0;
  unsigned int observed_constitutive_combinations = 0;
  for (size_t i = 0; i < plan.db_updates.size(); ++i) {
    const uint32_t variants = plan.db_updates[i].region.variant_key;
    const unsigned int combination = unsigned((variants & curl_has_pml) != 0) |
                                     (unsigned((variants & curl_has_pml_aux) != 0) << 1) |
                                     (unsigned((variants & curl_has_conductivity) != 0) << 2);
    observed_curl_combinations |= 1u << combination;
  }
  for (size_t i = 0; i < plan.eh_updates.size(); ++i) {
    const uint32_t variants = plan.eh_updates[i].region.variant_key;
    const unsigned int offdiagonals =
        (variants & constitutive_two_offdiagonals)
            ? 2
            : unsigned((variants & constitutive_one_offdiagonal) != 0);
    const unsigned int combination =
        offdiagonals + (unsigned((variants & constitutive_has_pml) != 0) * 3);
    observed_constitutive_combinations |= 1u << combination;
  }
  require((observed_curl_combinations & required_curl_combinations) == required_curl_combinations,
          "required NVIDIA curl PML/conductivity combination was not prepared");
  require((observed_constitutive_combinations & required_constitutive_combinations) ==
              required_constitutive_combinations,
          "required NVIDIA constitutive PML/anisotropy combination was not prepared");
}

static void set_uniform_conductivity(structure &s) {
  s.set_conductivity(Bx, uniform_conductivity);
  s.set_conductivity(By, uniform_conductivity);
  s.set_conductivity(Bz, uniform_conductivity);
  s.set_conductivity(Dx, uniform_conductivity);
  s.set_conductivity(Dy, uniform_conductivity);
  s.set_conductivity(Dz, uniform_conductivity);
}

static void run_beta_case(const char *name, precision_policy_kind policy, bool real_fields,
                          double beta, bool use_pml, bool conductivity, int chunks) {
  const grid_volume gv = vol2d(3.0, 2.0, 8.0);
  const boundary_region boundaries =
      use_pml ? pml(0.4, X) + pml(0.4, Y) : no_pml();
  structure cpu_structure(gv, isotropic_eps, boundaries, identity(), chunks);
  structure gpu_structure(gv, isotropic_eps, boundaries, identity(), chunks);
  if (conductivity) {
    set_uniform_conductivity(cpu_structure);
    set_uniform_conductivity(gpu_structure);
  }

  fields cpu(&cpu_structure, 0, beta);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&gpu_structure, options, 0, beta);
  if (real_fields) {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  gaussian_src_time cpu_time(0.31, 0.14), gpu_time(0.31, 0.14);
  const std::complex<double> amplitude =
      real_fields ? std::complex<double>(0.37, 0.0) : std::complex<double>(0.37, -0.23);
  cpu.add_point_source(Ez, cpu_time, vec(0.73, 0.83), amplitude);
  gpu.add_point_source(Ez, gpu_time, vec(0.73, 0.83), amplitude);

  cpu.advance(1);
  cpu.t = 0;
  gpu.init_backend();
  const StepPlan prepared = build_step_plan(gpu, StepProgram::ordinary);
  require(prepared.beta == beta && !prepared.beta_updates.empty(),
          "NVIDIA beta plan has no coordinate updates");
  bool saw_b = false, saw_d = false, saw_x = false, saw_y = false;
  bool saw_cmp0 = false, saw_cmp1 = false, saw_main = false, saw_aux = false, saw_cond = false;
  for (const BetaUpdate &update : prepared.beta_updates) {
    const StorageKey &target = gpu.array_catalog->key(update.target);
    const StorageKey &source = gpu.array_catalog->key(update.source);
    saw_b |= is_B(component(target.component_));
    saw_d |= is_D(component(target.component_));
    saw_x |= component_direction(component(target.component_)) == X;
    saw_y |= component_direction(component(target.component_)) == Y;
    saw_cmp0 |= target.cmp == 0;
    saw_cmp1 |= target.cmp == 1;
    saw_main |= (update.region.variant_key & beta_has_pml) != 0;
    saw_aux |= (update.region.variant_key & beta_has_pml_aux) != 0;
    saw_cond |= (update.region.variant_key & beta_has_conductivity) != 0;
    require(real_fields ? source.cmp == target.cmp : source.cmp == 1 - target.cmp,
            "NVIDIA beta source has incorrect real/complex cmp routing");
#if MEEP_SINGLE
    require(float_bits(realnum(update.betadt)) ==
                (update.betadt < 0 ? 0xbd88b8dcu : 0x3d88b8dcu),
            "native-single beta coefficient bits regressed");
#endif
  }
  require(saw_b && saw_d && saw_x && saw_y && saw_cmp0,
          "NVIDIA beta plan lacks B/D, X/Y, or cmp0 coverage");
  require(real_fields ? !saw_cmp1 : saw_cmp1,
          "NVIDIA beta plan has incorrect real/complex cmp coverage");
  require(saw_main == use_pml && saw_aux == use_pml && saw_cond == conductivity,
          "NVIDIA beta plan has incorrect PML/conductivity variants");

  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed);
  const bool reduced_precision = sizeof(realnum) == sizeof(float) || narrowed;
  const double tolerance = reduced_precision ? 3e-5 : 3e-13;
  const int checkpoints[] = {1, 2, 100};
  int previous = 0;
  for (size_t i = 0; i < sizeof(checkpoints) / sizeof(checkpoints[0]); ++i) {
    const int delta = checkpoints[i] - previous;
    cpu.advance(delta);
    gpu.advance(delta);
    require(cpu.t == gpu.t, "NVIDIA beta timestep did not advance host time");
    compare_fields(cpu, gpu, tolerance);
    previous = checkpoints[i];
  }
  master_printf("nvidia_timestep: beta-%s/%s PASS\n", name, precision_policy_name(policy));
}

static void run_bfast_case(const char *name, const grid_volume &gv,
                           precision_policy_kind policy, bool real_fields,
                           const std::vector<double> &scaled_k, bool use_pml,
                           bool conductivity, int chunks, const vec *bloch = NULL,
                           double beta = 0.0) {
  const boundary_region boundaries =
      use_pml ? pml(0.35, X) + pml(0.35, Y) : no_pml();
  structure cpu_structure(gv, isotropic_eps, boundaries, identity(), chunks);
  structure gpu_structure(gv, isotropic_eps, boundaries, identity(), chunks);
  if (conductivity) {
    set_uniform_conductivity(cpu_structure);
    set_uniform_conductivity(gpu_structure);
  }

  fields cpu(&cpu_structure, 0, beta, true, 0, 0, scaled_k);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(&gpu_structure, options, 0, beta, true, 0, 0, scaled_k);
  if (real_fields) {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  else if (bloch) {
    cpu.use_bloch(*bloch);
    gpu.use_bloch(*bloch);
  }
  const component source_component = gv.dim == D1 ? Ex : Ez;
  gaussian_src_time cpu_time(0.31, 0.14), gpu_time(0.31, 0.14);
  const std::complex<double> amplitude =
      real_fields ? std::complex<double>(0.37, 0.0) : std::complex<double>(0.37, -0.23);
  cpu.add_point_source(source_component, cpu_time, gv.center(), amplitude);
  gpu.add_point_source(source_component, gpu_time, gv.center(), amplitude);

  cpu.advance(1);
  cpu.t = 0;
  gpu.init_backend();
  const StepPlan prepared = build_step_plan(gpu, StepProgram::ordinary);
  require(prepared.bfast_scaled_k == scaled_k && !prepared.bfast_updates.empty(),
          "NVIDIA BFAST plan has no coordinate updates");
  bool saw_one_source = false, saw_two_sources = false, saw_zero_pair = false;
  bool saw_main = false, saw_aux = false, saw_cond = false;
  size_t paired = 0;
  for (const CurlUpdate &curl : prepared.db_updates) {
    require((curl.region.variant_key & curl_has_bfast) != 0 &&
                curl.bfast_update_index < prepared.bfast_updates.size(),
            "NVIDIA BFAST plan has an unpaired curl row");
    const BfastUpdate &update = prepared.bfast_updates[curl.bfast_update_index];
    ++paired;
    saw_one_source |= is_valid(update.source1) != is_valid(update.source2);
    saw_two_sources |= is_valid(update.source1) && is_valid(update.source2);
    saw_zero_pair |= update.k1 == 0.0 && update.k2 == 0.0;
    saw_main |= (update.region.variant_key & bfast_has_pml) != 0;
    saw_aux |= (update.region.variant_key & bfast_has_pml_aux) != 0;
    saw_cond |= (update.region.variant_key & bfast_has_conductivity) != 0;
    const StorageKey &target = gpu.array_catalog->key(update.target);
    const StorageKey &state = gpu.array_catalog->key(update.f_bfast);
    require(state.chunk == target.chunk && state.component_ == target.component_ &&
                state.cmp == target.cmp && state.kind == int(array_kind::f_bfast),
            "NVIDIA BFAST persistent-state identity is incorrect");
#if MEEP_SINGLE
    const uint32_t k1_bits = float_bits(realnum(update.k1));
    const uint32_t k2_bits = float_bits(realnum(update.k2));
    const bool known_k1 = k1_bits == 0 || k1_bits == 0x80000000u ||
                          k1_bits == 0x3e2e147bu ||
                          k1_bits == 0xbe2e147bu || k1_bits == 0x3de147aeu ||
                          k1_bits == 0xbde147aeu || k1_bits == 0x3d8f5c29u ||
                          k1_bits == 0xbd8f5c29u;
    const bool known_k2 = k2_bits == 0 || k2_bits == 0x80000000u ||
                          k2_bits == 0x3e2e147bu ||
                          k2_bits == 0xbe2e147bu || k2_bits == 0x3de147aeu ||
                          k2_bits == 0xbde147aeu || k2_bits == 0x3d8f5c29u ||
                          k2_bits == 0xbd8f5c29u;
    require(known_k1 && known_k2, "native-single BFAST coefficient bits regressed");
#endif
  }
  require(paired == prepared.bfast_updates.size() && paired == prepared.db_updates.size(),
          "NVIDIA BFAST plan pairing is not one-to-one");
  if (gv.dim == D1) require(saw_one_source, "D1 BFAST plan has no one-source row");
  if (gv.dim == D3) require(saw_two_sources, "D3 BFAST plan has no two-source row");
  if (scaled_k[1] == 0.0 || scaled_k[2] == 0.0)
    require(saw_zero_pair, "BFAST plan did not retain a zero-k row while enabled");
  require(saw_main == use_pml && saw_aux == use_pml && saw_cond == conductivity,
          "NVIDIA BFAST plan has incorrect PML/conductivity variants");
  require((beta != 0.0) == !prepared.beta_updates.empty(),
          "NVIDIA beta+BFAST composition has incorrect beta tail");

  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed);
  const bool reduced_precision = sizeof(realnum) == sizeof(float) || narrowed;
  const double tolerance = reduced_precision ? 4e-5 : 4e-13;
  const int checkpoints[] = {1, 2, 100};
  int previous = 0;
  for (size_t i = 0; i < sizeof(checkpoints) / sizeof(checkpoints[0]); ++i) {
    const int delta = checkpoints[i] - previous;
    cpu.advance(delta);
    gpu.advance(delta);
    require(cpu.t == gpu.t, "NVIDIA BFAST timestep did not advance host time");
    compare_fields(cpu, gpu, tolerance);
    previous = checkpoints[i];
  }
  master_printf("nvidia_timestep: bfast-%s/%s PASS\n", name,
                precision_policy_name(policy));
}

static void require_cylindrical_components(fields &f) {
  f.require_component(Er);
  f.require_component(Ep);
  f.require_component(Ez);
  f.require_component(Hr);
  f.require_component(Hp);
  f.require_component(Hz);
}

static void run_cylindrical_case(const char *name, precision_policy_kind policy, double m,
                                 bool real_fields, bool use_pml, bool conductivity,
                                 bool zero_near_origin, bool annular, bool use_bfast,
                                 double courant = 0.5) {
  grid_volume gv = volcyl(2.5, 3.0, 6.0);
  if (annular) gv.shift_origin(R, 8);
  const boundary_region boundaries = use_pml ? pml(0.35) : no_pml();
  linear_anisotropic_material cpu_material(true), gpu_material(true);
  structure cpu_structure(gv, cpu_material, boundaries, identity(), 2, courant);
  structure gpu_structure(gv, gpu_material, boundaries, identity(), 2, courant);
  if (conductivity) {
    set_uniform_conductivity(cpu_structure);
    set_uniform_conductivity(gpu_structure);
  }
  const std::vector<double> scaled_k =
      use_bfast ? std::vector<double>{0.17, -0.11, 0.07} : std::vector<double>{0, 0, 0};
  fields cpu(&cpu_structure, m, 0, zero_near_origin, 64, 64, scaled_k);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(&gpu_structure, options, m, 0, zero_near_origin, 64, 64, scaled_k);
  if (real_fields) {
    require(m == 0.0, "nonzero cylindrical m cannot use real fields");
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  require_cylindrical_components(cpu);
  require_cylindrical_components(gpu);
  cpu.advance(1);
  cpu.t = 0;
  gpu.init_backend();

  const StepPlan prepared = build_step_plan(gpu, StepProgram::ordinary);
  require(prepared.cylindrical_m == m && !prepared.cylindrical_radial_prefixes.empty(),
          "NVIDIA cylindrical plan lacks its coordinate fingerprint or radial prefixes");
  require((m != 0.0) == !prepared.cylindrical_m_updates.empty(),
          "NVIDIA cylindrical plan has incorrect m/r descriptor presence");
  const bool has_origin = !annular;
  require(has_origin == !prepared.cylindrical_origin_actions.empty(),
          "NVIDIA cylindrical plan has incorrect origin-action presence");
  require(use_bfast == !prepared.bfast_updates.empty(),
          "NVIDIA cylindrical plan has incorrect BFAST descriptor presence");
  bool saw_axis_replay = false, saw_prefix_pair = false, saw_m_r = false, saw_m_z = false;
  for (const CurlUpdate &curl : prepared.db_updates)
    if (component_direction(curl.region.c) == Z) {
      require(curl.radial_prefix_index < prepared.cylindrical_radial_prefixes.size(),
              "NVIDIA cylindrical Z curl is not paired with a radial prefix");
      saw_prefix_pair = true;
    }
  for (const CylindricalMOverRUpdate &update : prepared.cylindrical_m_updates) {
    saw_m_r |= component_direction(update.region.c) == R;
    saw_m_z |= component_direction(update.region.c) == Z;
#if MEEP_SINGLE
    const uint32_t magnitude = float_bits(realnum(fabs(update.numerator)));
    if (fabs(m) == 0.5)
      require(magnitude == 0x3f000000u,
              "native-single half-m cylindrical coefficient bits regressed");
    else if (fabs(m) == 1.0)
      require(magnitude == 0x3f800000u,
              "native-single unit-m cylindrical coefficient bits regressed");
    else if (fabs(m) == 3.0)
      require(magnitude == (courant == 0.25 ? 0x3fc00000u : 0x40400000u),
              "native-single high-m cylindrical coefficient bits regressed");
#endif
  }
  for (const CylindricalAxisUpdate &update : prepared.cylindrical_axis_updates) {
#if MEEP_SINGLE
    require(float_bits(realnum(update.dt)) == 0x3daaaaabu,
            "native-single cylindrical axis dt bits regressed");
    if (update.kind == CylindricalAxisKind::m0_dz)
      require(float_bits(realnum(fabs(update.scale))) == 0x40000000u,
              "native-single m=0 cylindrical axis scale bits regressed");
    else
      require(float_bits(realnum(fabs(update.scale))) == 0x3f000000u &&
                  (float_bits(realnum(fabs(update.source2_multiplier))) == 0x3f800000u ||
                   float_bits(realnum(fabs(update.source2_multiplier))) == 0x40000000u),
              "native-single |m|=1 cylindrical axis coefficient bits regressed");
#endif
  }
  for (const ConstitutiveUpdate &update : prepared.eh_updates)
    saw_axis_replay |= (update.region.variant_key & constitutive_axis_override) != 0;
  require(saw_prefix_pair && (m == 0.0 || (saw_m_r && saw_m_z)),
          "NVIDIA cylindrical plan lacks prefix or R/Z m-tail coverage");
  require(has_origin == saw_axis_replay,
          "NVIDIA cylindrical plan has incorrect constitutive axis replay presence");

  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed, 0.2);
  const bool reduced_precision = sizeof(realnum) == sizeof(float) || narrowed;
  const double tolerance = reduced_precision ? 8e-5 : 8e-13;
  const int checkpoints[] = {1, 2, 100};
  int previous = 0;
  for (size_t i = 0; i < sizeof(checkpoints) / sizeof(checkpoints[0]); ++i) {
    const int delta = checkpoints[i] - previous;
    cpu.advance(delta);
    gpu.advance(delta);
    require(cpu.t == gpu.t, "NVIDIA cylindrical timestep did not advance host time");
    if (getenv("MEEP_NVIDIA_CYLINDRICAL_TRACE"))
    {
      size_t array = 0, element = 0;
      double absolute = 0.0;
      const double maximum = max_field_difference(cpu, gpu, array, element, absolute);
      const StorageKey &key = cpu.array_catalog->key(ArrayId{uint32_t(array)});
      master_printf("nvidia_timestep: cylindrical-%s/%s checkpoint %d max-relative %.9g "
                    "absolute %.9g array %zu (%s,c=%d,cmp=%d) element %zu\n",
                    name, precision_policy_name(policy), checkpoints[i], maximum, absolute, array,
                    array_kind_name(array_kind(key.kind)), key.component_, key.cmp, element);
    }
    compare_fields(cpu, gpu, tolerance);
    previous = checkpoints[i];
  }
  master_printf("nvidia_timestep: cylindrical-%s/%s PASS\n", name,
                precision_policy_name(policy));
}

static void test_nvidia_cylindrical_change_m(precision_policy_kind policy) {
  const grid_volume gv = volcyl(2.5, 3.0, 6.0);
  linear_anisotropic_material cpu_material(true), gpu_material(true);
  structure cpu_structure(gv, cpu_material, no_pml(), identity(), 2);
  structure gpu_structure(gv, gpu_material, no_pml(), identity(), 2);
  fields cpu(&cpu_structure, +1.0);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(&gpu_structure, options, +1.0);
  require_cylindrical_components(cpu);
  require_cylindrical_components(gpu);
  cpu.advance(1);
  cpu.t = 0;
  gpu.init_backend();
  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed, 0.2);
  cpu.advance(2);
  gpu.advance(2);
  compare_fields(cpu, gpu, (sizeof(realnum) == sizeof(float) || narrowed) ? 8e-5 : 8e-13);

  struct surviving_snapshot {
    int chunk;
    component c;
    std::vector<realnum> values;
  };
  std::vector<surviving_snapshot> snapshots;
  for (int chunk = 0; chunk < gpu.num_chunks; ++chunk)
    FOR_COMPONENTS(c) {
      const ArrayId id = gpu.array_catalog->find(
          StorageKey{chunk, int(array_kind::f), int(c), 0, 0});
      if (!is_valid(id)) continue;
      surviving_snapshot snapshot;
      snapshot.chunk = chunk;
      snapshot.c = c;
      snapshot.values.resize(gpu.storage_plan->arrays[id.value].elements);
      gpu.backend->read(ArrayRef{id, 0, snapshot.values.size()}, snapshot.values.data(),
                        snapshot.values.size() * sizeof(realnum));
      snapshots.push_back(snapshot);
      realnum *host_mirror = gpu.chunks[chunk]->f[c][0];
      std::fill(host_mirror, host_mirror + snapshots.back().values.size(), realnum(-91.25));
    }

  cpu.change_m(0.0);
  gpu.change_m(0.0);
  require(!gpu.backend_state && !gpu.executable && gpu.is_real,
          "NVIDIA +1-to-0 change_m did not retire resident complex state");
  for (int chunk = 0; chunk < gpu.num_chunks; ++chunk)
    require(!gpu.chunks[chunk]->f[Er][1] && !gpu.chunks[chunk]->f[Ep][1] &&
                !gpu.chunks[chunk]->f[Ez][1] && !gpu.chunks[chunk]->f[Hr][1] &&
                !gpu.chunks[chunk]->f[Hp][1] && !gpu.chunks[chunk]->f[Hz][1],
            "NVIDIA +1-to-0 change_m retained imaginary field arrays");
  for (const surviving_snapshot &snapshot : snapshots) {
    const realnum *observed = gpu.chunks[snapshot.chunk]->f[snapshot.c][0];
    require(observed &&
                memcmp(snapshot.values.data(), observed,
                       snapshot.values.size() * sizeof(realnum)) == 0,
            "NVIDIA change_m did not migrate a surviving real field exactly");
  }

  cpu.advance(1);
  gpu.advance(1);
  require(gpu.backend_state && gpu.executable && gpu.is_real,
          "NVIDIA +1-to-0 change_m did not rebuild real resident state");
  compare_fields(cpu, gpu, (sizeof(realnum) == sizeof(float) || narrowed) ? 8e-5 : 8e-13);
  master_printf("nvidia_timestep: cylindrical-change-m/%s PASS\n",
                precision_policy_name(policy));
}

static void require_source_plan(fields &f, bool conductivity, bool integrated,
                                bool expect_cross_copy, bool expect_nonlinear = false) {
  require(f.descriptors, "source test has no prepared descriptor set");
  const SourcePlan &sources = f.descriptors->sources;
  require(sources.source_times.size() == 1 && sources.scalars.size() == 1,
          "point source did not prepare one scalar slot");
  require(sources.source_times[0].source_time_id == 0 && sources.source_times[0].scalar_slot == 0,
          "point source scalar mapping is not canonical");
  require(!sources.sources.empty(), "point source did not prepare a spatial descriptor");
  for (size_t i = 0; i < sources.sources.size(); ++i) {
    const SourceDescriptor &source = sources.sources[i];
    require(source.integrated == integrated && source.source_time_id == 0 &&
                !source.indices.empty() &&
                source.indices.size() == source.complex_amplitudes.size(),
            "point-source descriptor shape is invalid");
    require(is_valid(source.destination), "point source has no destination");
    require(is_valid(source.integrated_destination) == integrated,
            "point-source integrated destination does not match its mode");
    require(is_valid(source.condinv) == conductivity,
            "point-source conductivity descriptor does not match the material");
  }

  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  size_t evaluations = 0, source_spans = 0;
  bool have_nonlinear_primary_source = false;
  bool have_nonlinear_cross_source = false;
  for (size_t i = 0; i < plan.operations.size(); ++i) {
    const Operation &op = plan.operations[i];
    if (op.kind == OpKind::evaluate_source_scalars) {
      require(op.descriptor_index == 0 && op.descriptor_count == 1,
              "source evaluation does not cover the scalar descriptor");
      ++evaluations;
    }
    if (!integrated && op.kind == OpKind::apply_sources) {
      if (op.ft == D_stuff) {
        require(op.source_descriptor_index == 0 &&
                    op.source_descriptor_count == sources.sources.size(),
                "D source operation does not cover the point descriptor");
        source_spans += op.source_descriptor_count;
      }
      else
        require(op.source_descriptor_count == 0,
                "point source was attached to the wrong field-type operation");
    }
    if (integrated && op.kind == OpKind::update_eh && op.ft == E_stuff) {
      require(op.source_descriptor_index == 0 &&
                  op.source_descriptor_count == sources.sources.size(),
              "integrated source was not attached to the E update");
      source_spans += op.source_descriptor_count;
      bool have_primary_copy = false, have_cross_copy = false;
      for (size_t j = op.descriptor_index;
           j < size_t(op.descriptor_index) + op.descriptor_count; ++j) {
        const ConstitutiveUpdate &update = plan.eh_updates[j];
        have_primary_copy |= is_valid(update.primary) && update.primary != update.base_primary;
        have_cross_copy |=
            (is_valid(update.cross1) && update.cross1 != update.base_cross1) ||
            (is_valid(update.cross2) && update.cross2 != update.base_cross2);
        if (expect_nonlinear &&
            (update.region.variant_key & constitutive_has_nonlinearity) != 0) {
          for (size_t k = 0; k < sources.sources.size(); ++k) {
            const SourceDescriptor &source = sources.sources[k];
            const ArrayId destinations[] = {source.integrated_destination,
                                            source.integrated_destination_imag};
            for (size_t d = 0; d < 2; ++d) {
              if (!is_valid(destinations[d])) continue;
              have_nonlinear_primary_source |= update.primary == destinations[d];
              have_nonlinear_cross_source |= update.cross1 == destinations[d] ||
                                             update.cross2 == destinations[d];
            }
          }
        }
      }
      require(have_primary_copy, "integrated source has no prepared primary copy");
      require(have_cross_copy == expect_cross_copy,
              "integrated source cross-copy coverage differs from expectation");
    }
  }
  require(evaluations == 4, "ordinary source plan did not schedule four scalar evaluations");
  require(source_spans == sources.sources.size(), "point-source descriptor span was not unique");
  if (expect_nonlinear) {
    require(have_nonlinear_primary_source,
            "integrated source does not alter a nonlinear primary input");
    require(have_nonlinear_cross_source,
            "integrated source does not alter a nonlinear centered-cross input");
  }
}

static void compare_source_scalars(const fields &cpu, const fields &gpu) {
  require(cpu.descriptors && gpu.descriptors, "source scalar comparison has no descriptors");
  const std::vector<SourceDescriptor> &cpu_sources = cpu.descriptors->sources.sources;
  const std::vector<SourceDescriptor> &gpu_sources = gpu.descriptors->sources.sources;
  require(cpu_sources.size() == gpu_sources.size(),
          "CPU and NVIDIA source descriptor counts differ");
  for (size_t i = 0; i < cpu_sources.size(); ++i) {
    require(cpu_sources[i].source_time_id == gpu_sources[i].source_time_id &&
                cpu_sources[i].indices == gpu_sources[i].indices &&
                cpu_sources[i].complex_amplitudes == gpu_sources[i].complex_amplitudes,
            "CPU and NVIDIA source descriptor order differs");
  }
  const std::vector<SourceStepScalar> &expected = cpu.descriptors->sources.scalars;
  const std::vector<SourceStepScalar> &observed = gpu.descriptors->sources.scalars;
  require(expected.size() == observed.size(), "CPU and NVIDIA source scalar counts differ");
  for (size_t i = 0; i < expected.size(); ++i) {
    require(expected[i].current == observed[i].current,
            "NVIDIA current scalar differs from host evaluation");
    require(expected[i].dipole == observed[i].dipole,
            "NVIDIA dipole scalar differs from host evaluation");
  }
}

static std::complex<double> plane_source_amplitude(const vec &point) {
  const double phase = 0.7 * point.x() - 0.4 * point.y();
  return std::complex<double>(std::cos(phase), std::sin(phase));
}

static void run_source_case(const char *name, precision_policy_kind policy, bool real_fields,
                            bool conductivity, bool integrated = false,
                            material_function *material = NULL,
                            const boundary_region *boundaries = NULL, bool continuous = false,
                            bool plane_source = false, bool nonlinear = false) {
  const bool anisotropic = material != NULL;
  const grid_volume gv = anisotropic ? vol3d(2.0, 2.0, 2.0, 5.0) : vol2d(2.0, 2.0, 8.0);
  const boundary_region no_boundaries = no_pml();
  const boundary_region &boundary = boundaries ? *boundaries : no_boundaries;
  std::unique_ptr<structure> cpu_structure, gpu_structure;
  if (material) {
    cpu_structure.reset(new structure(gv, *material, boundary, identity(), 1));
    gpu_structure.reset(new structure(gv, *material, boundary, identity(), 1));
  }
  else {
    cpu_structure.reset(new structure(gv, isotropic_eps, boundary, identity(), 1));
    gpu_structure.reset(new structure(gv, isotropic_eps, boundary, identity(), 1));
  }
  if (conductivity) {
    cpu_structure->set_conductivity(Dz, uniform_conductivity);
    gpu_structure->set_conductivity(Dz, uniform_conductivity);
  }
  if (nonlinear) {
    cpu_structure->set_chi3(chi3_value);
    gpu_structure->set_chi3(chi3_value);
  }

  fields cpu(cpu_structure.get());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(gpu_structure.get(), options);
  if (real_fields) {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  std::unique_ptr<src_time> cpu_time, gpu_time;
  if (continuous) {
    cpu_time.reset(new continuous_src_time(std::complex<double>(0.31, -0.02), 0.2, 0.0, 4.0));
    gpu_time.reset(new continuous_src_time(std::complex<double>(0.31, -0.02), 0.2, 0.0, 4.0));
  }
  else {
    cpu_time.reset(new gaussian_src_time(0.31, 0.14));
    gpu_time.reset(new gaussian_src_time(0.31, 0.14));
  }
  cpu_time->is_integrated = integrated;
  gpu_time->is_integrated = integrated;
  const std::complex<double> amplitude =
      real_fields ? std::complex<double>(0.37, 0.0) : std::complex<double>(0.37, -0.23);
  const vec location = anisotropic ? vec(0.73, 0.83, 0.41) : vec(0.73, 0.83);
  if (plane_source) {
    const volume plane(vec(0.25, 0.83), vec(1.75, 0.83));
    cpu.add_volume_source(Ez, *cpu_time, plane, plane_source_amplitude, amplitude);
    gpu.add_volume_source(Ez, *gpu_time, plane, plane_source_amplitude, amplitude);
  }
  else {
    cpu.add_point_source(Ez, *cpu_time, location, amplitude);
    gpu.add_point_source(Ez, *gpu_time, location, amplitude);
  }

  /* Prepare the CPU catalog without retaining the preparation step's fields. */
  cpu.advance(1);
  cpu.t = 0;
  gpu.init_backend();
  require_source_plan(cpu, conductivity, integrated, anisotropic && integrated, nonlinear);
  require_source_plan(gpu, conductivity, integrated, anisotropic && integrated, nonlinear);

  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed);
  const bool reduced_precision = sizeof(realnum) == sizeof(float) || narrowed;
  const double tolerance = reduced_precision ? 2e-5 : 2e-13;
  const int checkpoints[] = {1, 2, 8};
  int previous = 0;
  for (size_t i = 0; i < sizeof(checkpoints) / sizeof(checkpoints[0]); ++i) {
    const int delta = checkpoints[i] - previous;
    cpu.advance(delta);
    gpu.advance(delta);
    require(cpu.t == gpu.t, "NVIDIA source timestep did not advance host time");
    compare_fields(cpu, gpu, tolerance);
    compare_source_scalars(cpu, gpu);
    previous = checkpoints[i];
  }
  master_printf("nvidia_timestep: %s/%s PASS\n", name, precision_policy_name(policy));
}

struct custom_source_trace {
  std::vector<double> times;
};

static std::complex<double> custom_source_value(double time, void *data) {
  custom_source_trace *trace = static_cast<custom_source_trace *>(data);
  if (trace) trace->times.push_back(time);
  return std::complex<double>(std::sin(0.7 * time), -0.5 * std::cos(0.3 * time));
}

static void run_custom_source_case(const char *name, precision_policy_kind policy,
                                   bool integrated) {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  if (getenv("MEEP_NVIDIA_GRAPH_ASSERT")) {
    options.strict = false;
    options.fallback = fallback_policy::warn;
  }
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();

  custom_source_trace cpu_trace, gpu_trace;
  custom_src_time cpu_time(custom_source_value, &cpu_trace);
  custom_src_time gpu_time(custom_source_value, &gpu_trace);
  cpu_time.is_integrated = integrated;
  gpu_time.is_integrated = integrated;
  const vec location(0.73, 0.83);
  cpu.add_point_source(Ez, cpu_time, location, std::complex<double>(0.37, 0.0));
  gpu.add_point_source(Ez, gpu_time, location, std::complex<double>(0.37, 0.0));

  /* Prepare both live source objects through the same first step, then reset
     field/time state so callback ordering can be compared from an equal cache. */
  cpu.advance(1);
  gpu.advance(1);
  require_selected_graph_mode(gpu);
  cpu.t = gpu.t = 0;
  cpu_trace.times.clear();
  gpu_trace.times.clear();
  require_source_plan(cpu, false, integrated, false);
  require_source_plan(gpu, false, integrated, false);

  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed);
  const bool reduced_precision = sizeof(realnum) == sizeof(float) || narrowed;
  const double tolerance = reduced_precision ? 2e-5 : 2e-13;
  const int checkpoints[] = {1, 2, 8};
  int previous = 0;
  for (size_t i = 0; i < sizeof(checkpoints) / sizeof(checkpoints[0]); ++i) {
    const int delta = checkpoints[i] - previous;
    cpu.advance(delta);
    gpu.advance(delta);
    compare_fields(cpu, gpu, tolerance);
    compare_source_scalars(cpu, gpu);
    require(cpu_trace.times == gpu_trace.times,
            "NVIDIA custom-source callback count/order/time differs");
    previous = checkpoints[i];
  }
  require(!cpu_trace.times.empty(), "custom-source callback was not evaluated");
  master_printf("nvidia_timestep: %s/%s PASS\n", name, precision_policy_name(policy));
}

static void run_case(const char *name, const grid_volume &gv, precision_policy_kind policy,
                     bool real_fields, const boundary_region &boundaries,
                     const symmetry &symmetries, int chunks, const vec *bloch,
                     unsigned int required_halo_phases, unsigned int required_curl_combinations,
                     unsigned int required_constitutive_combinations, bool conductivity,
                     bool check_lifecycle, material_function *material = NULL,
                     bool chi2 = false, bool chi3 = false,
                     component nonlinear_component = NO_COMPONENT) {
  std::unique_ptr<structure> cpu_structure, gpu_structure;
  if (material) {
    cpu_structure.reset(new structure(gv, *material, boundaries, symmetries, chunks));
    gpu_structure.reset(new structure(gv, *material, boundaries, symmetries, chunks));
  }
  else {
    cpu_structure.reset(new structure(gv, isotropic_eps, boundaries, symmetries, chunks));
    gpu_structure.reset(new structure(gv, isotropic_eps, boundaries, symmetries, chunks));
  }
  if (conductivity) {
    set_uniform_conductivity(*cpu_structure);
    set_uniform_conductivity(*gpu_structure);
  }
  if (chi2) {
    if (nonlinear_component == NO_COMPONENT) {
      cpu_structure->set_chi2(chi2_value);
      gpu_structure->set_chi2(chi2_value);
    }
    else {
      simple_material_function nonlinear(chi2_value);
      cpu_structure->set_chi2(nonlinear_component, nonlinear);
      gpu_structure->set_chi2(nonlinear_component, nonlinear);
    }
  }
  if (chi3) {
    if (nonlinear_component == NO_COMPONENT) {
      cpu_structure->set_chi3(chi3_value);
      gpu_structure->set_chi3(chi3_value);
    }
    else {
      simple_material_function nonlinear(chi3_value);
      cpu_structure->set_chi3(nonlinear_component, nonlinear);
      gpu_structure->set_chi3(nonlinear_component, nonlinear);
    }
  }

  fields cpu(cpu_structure.get());
  if (real_fields)
    cpu.use_real_fields();
  else if (bloch)
    cpu.use_bloch(*bloch);
  cpu.require_component(Ez);
  if (nonlinear_component != NO_COMPONENT) cpu.require_component(nonlinear_component);
  /* Preserve a like-for-like prepared CPU reference. */
  cpu.advance(1);
  cpu.t = 0;

  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(gpu_structure.get(), options);
  if (real_fields)
    gpu.use_real_fields();
  else if (bloch)
    gpu.use_bloch(*bloch);
  gpu.require_component(Ez);
  if (nonlinear_component != NO_COMPONENT) gpu.require_component(nonlinear_component);
  gpu.init_backend();
  require_halo_phases(gpu, required_halo_phases);
  const StepPlan prepared = build_step_plan(gpu, StepProgram::ordinary);
  require_physics_variants(prepared, required_curl_combinations,
                           required_constitutive_combinations);
  bool found_nonlinearity = false;
  bool found_requested_nonlinearity = false;
  bool found_negative_centered_crosses = false;
  for (size_t i = 0; i < prepared.eh_updates.size(); ++i) {
    const ConstitutiveUpdate &update = prepared.eh_updates[i];
    const bool nonlinear =
        (update.region.variant_key & constitutive_has_nonlinearity) != 0;
    found_nonlinearity |= nonlinear;
    if (nonlinear && nonlinear_component != NO_COMPONENT &&
        gpu.array_catalog->key(update.target).component_ == int(nonlinear_component)) {
      found_requested_nonlinearity = true;
      found_negative_centered_crosses |=
          is_valid(update.cross1) && is_valid(update.cross2) && update.primary_stride < 0 &&
          update.cross1_stride < 0 && update.cross2_stride < 0;
    }
  }
  require(found_nonlinearity == (chi2 || chi3),
          "NVIDIA nonlinear constitutive descriptor coverage differs");
  if (nonlinear_component != NO_COMPONENT) {
    require(found_requested_nonlinearity,
            "NVIDIA plan has no nonlinear descriptor for the requested component");
    require(found_negative_centered_crosses,
            "magnetic nonlinear descriptor lacks negative centered-cross strides");
  }

  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed);

  const bool reduced_precision = sizeof(realnum) == sizeof(float) || narrowed;
  const double tolerance = reduced_precision ? 2e-5 : 2e-13;
  const int checkpoints[] = {1, 2, 100};
  int previous = 0;
  for (size_t i = 0; i < sizeof(checkpoints) / sizeof(checkpoints[0]); ++i) {
    const int delta = checkpoints[i] - previous;
    cpu.advance(delta);
    gpu.advance(delta);
    require(cpu.t == gpu.t, "NVIDIA timestep did not advance host time");
    compare_fields(cpu, gpu, tolerance);
    previous = checkpoints[i];
  }

  if (check_lifecycle) {
    invalidate(gpu, MutationKind::field_layout, "nvidia_timestep migration test");
    BackendState *const live_state = gpu.backend_state;
    Executable *const live_executable = gpu.executable;
    const nvidia::memory_accounting before_failure = nvidia::current_memory_accounting();
    const nvidia::testing::failure_point failure_points[] = {
        nvidia::testing::failure_point::device_to_host_copy,
        nvidia::testing::failure_point::state_rebuild_sync};
    for (nvidia::testing::failure_point point : failure_points) {
      nvidia::testing::fail_next(point);
      bool rejected = false;
      try {
        gpu.init_backend();
      }
      catch (const std::exception &) { rejected = true; }
      nvidia::testing::clear_failure();
      const nvidia::memory_accounting after_failure = nvidia::current_memory_accounting();
      require(rejected && gpu.backend_state == live_state && gpu.executable == live_executable &&
                  !gpu.backend->is_poisoned() &&
                  before_failure.device_bytes_current == after_failure.device_bytes_current &&
                  before_failure.pinned_bytes_current == after_failure.pinned_bytes_current,
              "failed NVIDIA authority migration changed or poisoned the live epoch");
    }
    gpu.init_backend();
    compare_fields(cpu, gpu, tolerance);

    cpu.advance(1);
    gpu.advance(1);
    require(cpu.t == gpu.t, "NVIDIA timestep lost time across a storage rebuild");
    compare_fields(cpu, gpu, tolerance);
    require(gpu.initialization_plan && gpu.backend_state,
            "NVIDIA lifecycle test has no prepared state");
    bool rejected_stale_refresh = false;
    try {
      gpu.backend->initialize(*gpu.initialization_plan, *gpu.backend_state);
    }
    catch (const std::runtime_error &error) {
      rejected_stale_refresh =
          std::string(error.what()).find("stale host mirror") != std::string::npos;
    }
    require(rejected_stale_refresh,
            "NVIDIA initialization accepted a stale host mirror after stepping");
  }
  master_printf("nvidia_timestep: %s/%s PASS\n", name, precision_policy_name(policy));
}

static void run_legacy_flux_case(const char *name, precision_policy_kind policy,
                                 bool complex_fields) {
  const grid_volume gv = vol2d(3.0, 2.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(&gpu_structure, options);
  if (complex_fields) {
    const vec bloch(0.17, 0.11);
    cpu.use_bloch(bloch);
    gpu.use_bloch(bloch);
  }
  else {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  const component components[] = {Ex, Ey, Ez, Hx, Hy, Hz};
  for (size_t i = 0; i < sizeof(components) / sizeof(components[0]); ++i) {
    cpu.require_component(components[i]);
    gpu.require_component(components[i]);
  }
  cpu.advance(1);
  cpu.t = 0;
  flux_vol *cpu_x = cpu.add_flux_vol(X, volume(vec(0.0, -0.75), vec(0.0, 0.75)));
  flux_vol *gpu_x = gpu.add_flux_vol(X, volume(vec(0.0, -0.75), vec(0.0, 0.75)));
  flux_vol *cpu_y = cpu.add_flux_vol(Y, volume(vec(-1.0, 0.0), vec(1.0, 0.0)));
  flux_vol *gpu_y = gpu.add_flux_vol(Y, volume(vec(-1.0, 0.0), vec(1.0, 0.0)));
  gpu.init_backend();
  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed, 0.41);

  const double tolerance = (sizeof(realnum) == sizeof(float) || narrowed) ? 2e-5 : 2e-12;
  for (int step = 0; step < 2; ++step) {
    cpu.advance(1);
    nvidia::testing::reset_transfer_accounting();
    const nvidia::memory_accounting before = nvidia::current_memory_accounting();
    gpu.advance(1);
    const nvidia::memory_accounting after = nvidia::current_memory_accounting();
    const nvidia::testing::transfer_accounting transfers =
        nvidia::testing::current_transfer_accounting();
    if (!step) {
      const StepPlan plan = build_step_plan(gpu, StepProgram::ordinary);
      require(plan.legacy_flux_updates.size() == 2 && !plan.legacy_flux_terms.empty(),
              "NVIDIA legacy flux plan lacks ordered monitor terms after refresh");
    }
    require(std::fabs(cpu_x->flux() - gpu_x->flux()) <=
                    tolerance * (1.0 + std::fabs(cpu_x->flux())) &&
                std::fabs(cpu_y->flux() - gpu_y->flux()) <=
                    tolerance * (1.0 + std::fabs(cpu_y->flux())),
            "NVIDIA legacy flux scalar differs from CPU");
    require(transfers.device_to_host_calls == 1 &&
                transfers.device_to_host_bytes == 2 * sizeof(double),
            "NVIDIA legacy flux did not publish with one final scalar transfer");
    if (step)
      require(before.device_bytes_current == after.device_bytes_current &&
                  before.pinned_bytes_current == after.pinned_bytes_current,
              "NVIDIA legacy flux allocated storage during steady execution");
  }
  master_printf("nvidia_timestep: flux-%s/%s PASS\n", name,
                precision_policy_name(policy));
}

static void run_legacy_flux_symmetry_case(precision_policy_kind policy) {
  const grid_volume gv = vol2d(3.0, 2.0, 8.0);
  const symmetry sym = -mirror(Y, gv);
  structure cpu_structure(gv, isotropic_eps, no_pml(), sym, 2);
  structure gpu_structure(gv, isotropic_eps, no_pml(), sym, 2);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(&gpu_structure, options);
  const vec bloch(0.17, 0.0);
  cpu.use_bloch(bloch);
  gpu.use_bloch(bloch);
  const component components[] = {Ex, Ey, Ez, Hx, Hy, Hz};
  for (size_t i = 0; i < sizeof(components) / sizeof(components[0]); ++i) {
    cpu.require_component(components[i]);
    gpu.require_component(components[i]);
  }
  cpu.advance(1);
  cpu.t = 0;
  flux_vol *cpu_flux =
      cpu.add_flux_vol(X, volume(vec(0.0, -0.9), vec(0.0, 0.9)));
  flux_vol *gpu_flux =
      gpu.add_flux_vol(X, volume(vec(0.0, -0.9), vec(0.0, 0.9)));
  gpu.init_backend();
  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed, 0.29);
  cpu.advance(1);
  gpu.advance(1);
  const double tolerance = (sizeof(realnum) == sizeof(float) || narrowed) ? 8e-5 : 3e-12;
  require(std::fabs(cpu_flux->flux() - gpu_flux->flux()) <=
              tolerance * (1.0 + std::fabs(cpu_flux->flux())),
          "NVIDIA symmetry-transformed legacy flux differs from CPU");
  const StepPlan plan = build_step_plan(gpu, StepProgram::ordinary);
  bool transformed = false;
  for (const LegacyFluxTerm &term : plan.legacy_flux_terms)
    transformed |= term.symmetry_index != 0 || term.lattice_shift != ivec(0);
  require(transformed, "legacy flux symmetry fixture produced no transformed term");
  master_printf("nvidia_timestep: flux-symmetry/%s PASS\n",
                precision_policy_name(policy));
}

static void run_legacy_flux_material_case(precision_policy_kind policy, bool anisotropic,
                                          bool conductivity) {
  const grid_volume gv = vol3d(2.0, 2.0, 2.0, 5.0);
  const boundary_region boundaries = pml(0.4, X) + pml(0.4, Y);
  linear_anisotropic_material material(true);
  std::unique_ptr<structure> cpu_structure, gpu_structure;
  if (anisotropic) {
    cpu_structure.reset(new structure(gv, material, boundaries, identity(), 2));
    gpu_structure.reset(new structure(gv, material, boundaries, identity(), 2));
  }
  else {
    cpu_structure.reset(new structure(gv, isotropic_eps, boundaries, identity(), 2));
    gpu_structure.reset(new structure(gv, isotropic_eps, boundaries, identity(), 2));
  }
  if (conductivity) {
    set_uniform_conductivity(*cpu_structure);
    set_uniform_conductivity(*gpu_structure);
  }
  fields cpu(cpu_structure.get());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(gpu_structure.get(), options);
  if (anisotropic) {
    const vec bloch(0.11, 0.07, 0.05);
    cpu.use_bloch(bloch);
    gpu.use_bloch(bloch);
  }
  else {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  cpu.require_component(Ez);
  gpu.require_component(Ez);
  gaussian_src_time cpu_source(0.31, 0.14), gpu_source(0.31, 0.14);
  const std::complex<double> amplitude =
      anisotropic ? std::complex<double>(0.23, -0.11) : std::complex<double>(0.23, 0.0);
  cpu.add_point_source(Ez, cpu_source, vec(0.21, -0.17, 0.13), amplitude);
  gpu.add_point_source(Ez, gpu_source, vec(0.21, -0.17, 0.13), amplitude);
  cpu.advance(1);
  cpu.t = 0;
  flux_vol *cpu_flux = cpu.add_flux_vol(
      Z, volume(vec(-0.75, -0.75, 0.0), vec(0.75, 0.75, 0.0)));
  flux_vol *gpu_flux = gpu.add_flux_vol(
      Z, volume(vec(-0.75, -0.75, 0.0), vec(0.75, 0.75, 0.0)));
  gpu.init_backend();
  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_live_fields_by_key(cpu, gpu, narrowed, 0.013);
  cpu.advance(1);
  gpu.advance(1);
  const double tolerance = (sizeof(realnum) == sizeof(float) || narrowed) ? 1e-4 : 4e-12;
  require(std::fabs(cpu_flux->flux() - gpu_flux->flux()) <=
              tolerance * (1.0 + std::fabs(cpu_flux->flux())),
          "NVIDIA PML/conductive material legacy flux differs from CPU");
  const StepPlan plan = build_step_plan(gpu, StepProgram::ordinary);
  bool saw_split_h = false, saw_conductivity = false, saw_anisotropy = false;
  for (const CurlUpdate &update : plan.db_updates) {
    saw_split_h |= (update.region.variant_key & curl_has_pml_aux) != 0;
    saw_conductivity |= (update.region.variant_key & curl_has_conductivity) != 0;
  }
  for (const ConstitutiveUpdate &update : plan.eh_updates)
    saw_anisotropy |= (update.region.variant_key & constitutive_one_offdiagonal) != 0 ||
                      (update.region.variant_key & constitutive_two_offdiagonals) != 0;
  require(saw_split_h && saw_conductivity == conductivity && saw_anisotropy == anisotropic,
          "legacy flux material fixture lacks its requested PML/material composition");
  master_printf("nvidia_timestep: flux-material-%s/%s PASS\n",
                anisotropic ? "anisotropic-pml" : "pml-conductive",
                precision_policy_name(policy));
}

static void test_legacy_flux_missing_components() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  flux_vol *flux = gpu.add_flux_vol(Z, volume(vec(-0.5, 0.0), vec(0.5, 0.0)));
  gpu.init_backend();
  gpu.advance(1);
  require(flux->flux() == 0.0,
          "NVIDIA legacy flux did not treat absent field components as zero");
  master_printf("nvidia_timestep: flux missing-components PASS\n");
}

static void run_cylindrical_flux_case(precision_policy_kind policy) {
  grid_volume gv = volcyl(2.5, 3.0, 6.0);
  gv.shift_origin(R, 8);
  structure cpu_structure(gv, isotropic_eps, pml(0.35), identity(), 2);
  structure gpu_structure(gv, isotropic_eps, pml(0.35), identity(), 2);
  fields cpu(&cpu_structure, 1.0);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(&gpu_structure, options, 1.0);
  require_cylindrical_components(cpu);
  require_cylindrical_components(gpu);
  cpu.advance(1);
  cpu.t = 0;
  flux_vol *cpu_r =
      cpu.add_flux_vol(R, volume(veccyl(1.25, -1.0), veccyl(1.25, 1.0)));
  flux_vol *gpu_r =
      gpu.add_flux_vol(R, volume(veccyl(1.25, -1.0), veccyl(1.25, 1.0)));
  flux_vol *cpu_p =
      cpu.add_flux_vol(P, volume(veccyl(0.5, -1.0), veccyl(2.0, 1.0)));
  flux_vol *gpu_p =
      gpu.add_flux_vol(P, volume(veccyl(0.5, -1.0), veccyl(2.0, 1.0)));
  flux_vol *cpu_z = cpu.add_flux_vol(Z, volume(veccyl(0.5, 0.0), veccyl(2.0, 0.0)));
  flux_vol *gpu_z = gpu.add_flux_vol(Z, volume(veccyl(0.5, 0.0), veccyl(2.0, 0.0)));
  gpu.init_backend();
  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed, 0.17);
  cpu.advance(1);
  gpu.advance(1);
  const double tolerance = (sizeof(realnum) == sizeof(float) || narrowed) ? 8e-5 : 3e-12;
  require(std::fabs(cpu_r->flux() - gpu_r->flux()) <=
                  tolerance * (1.0 + std::fabs(cpu_r->flux())) &&
              std::fabs(cpu_p->flux() - gpu_p->flux()) <=
                  tolerance * (1.0 + std::fabs(cpu_p->flux())) &&
              std::fabs(cpu_z->flux() - gpu_z->flux()) <=
                  tolerance * (1.0 + std::fabs(cpu_z->flux())),
          "NVIDIA cylindrical R/P/Z legacy flux differs from CPU");
  const StepPlan plan = build_step_plan(gpu, StepProgram::ordinary);
  bool saw_radial_volume = false;
  for (const LegacyFluxTerm &term : plan.legacy_flux_terms)
    saw_radial_volume |= term.dV1 != 0.0;
  require(saw_radial_volume,
          "NVIDIA cylindrical legacy flux did not exercise radial volume weighting");
  master_printf("nvidia_timestep: flux-cylindrical-rpz/%s PASS\n",
                precision_policy_name(policy));
}

static void test_legacy_flux_add_remove() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  const component components[] = {Ex, Ey, Ez, Hx, Hy, Hz};
  for (size_t i = 0; i < sizeof(components) / sizeof(components[0]); ++i) {
    cpu.require_component(components[i]);
    gpu.require_component(components[i]);
  }
  cpu.advance(1);
  cpu.t = 0;
  gpu.init_backend();
  initialize_fields(cpu, gpu, false, 0.23);
  cpu.advance(1);
  gpu.advance(1);
  BackendState *const live_state = gpu.backend_state;
  const double tolerance = sizeof(realnum) == sizeof(float) ? 8e-5 : 2e-12;

  flux_vol *cpu_first = cpu.add_flux_vol(Z, volume(vec(-0.75, 0.0), vec(0.75, 0.0)));
  flux_vol *gpu_first = gpu.add_flux_vol(Z, volume(vec(-0.75, 0.0), vec(0.75, 0.0)));
  Executable *old_executable = gpu.executable;
  cpu.advance(1);
  gpu.advance(1);
  require(gpu.backend_state == live_state && gpu.executable != old_executable &&
              std::fabs(cpu_first->flux() - gpu_first->flux()) <=
                  tolerance * (1.0 + std::fabs(cpu_first->flux())),
          "NVIDIA live legacy flux addition did not recompile and publish");

  flux_vol *cpu_second = cpu.add_flux_vol(X, volume(vec(0.0, -0.75), vec(0.0, 0.75)));
  flux_vol *gpu_second = gpu.add_flux_vol(X, volume(vec(0.0, -0.75), vec(0.0, 0.75)));
  old_executable = gpu.executable;
  cpu.advance(1);
  gpu.advance(1);
  require(gpu.backend_state == live_state && gpu.executable != old_executable &&
              std::fabs(cpu_first->flux() - gpu_first->flux()) <=
                  tolerance * (1.0 + std::fabs(cpu_first->flux())) &&
              std::fabs(cpu_second->flux() - gpu_second->flux()) <=
                  tolerance * (1.0 + std::fabs(cpu_second->flux())),
          "NVIDIA second legacy flux addition changed list ordering or publication");

  old_executable = gpu.executable;
  cpu.remove_fluxes();
  gpu.remove_fluxes();
  nvidia::testing::reset_transfer_accounting();
  cpu.advance(1);
  gpu.advance(1);
  const nvidia::testing::transfer_accounting transfers =
      nvidia::testing::current_transfer_accounting();
  require(gpu.backend_state == live_state && gpu.executable != old_executable &&
              transfers.device_to_host_calls == 0,
          "NVIDIA legacy flux removal did not replace code without scalar transfer");

  flux_vol *cpu_again =
      cpu.add_flux_vol(Y, volume(vec(-0.75, 0.0), vec(0.75, 0.0)));
  flux_vol *gpu_again =
      gpu.add_flux_vol(Y, volume(vec(-0.75, 0.0), vec(0.75, 0.0)));
  old_executable = gpu.executable;
  cpu.advance(1);
  gpu.advance(1);
  require(gpu.backend_state == live_state && gpu.executable != old_executable &&
              std::fabs(cpu_again->flux() - gpu_again->flux()) <=
                  tolerance * (1.0 + std::fabs(cpu_again->flux())) &&
              std::fabs(cpu_again->flux()) > std::numeric_limits<realnum>::epsilon() &&
              std::fabs(gpu_again->flux()) > std::numeric_limits<realnum>::epsilon(),
          "NVIDIA legacy flux add-after-remove did not recompile or publish");
  master_printf("nvidia_timestep: flux add/remove/add PASS\n");
}

static void test_legacy_flux_postlaunch_poison() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  execution_options options;
  options.backend = backend_kind::nvidia;
  if (getenv("MEEP_NVIDIA_GRAPH_ASSERT")) {
    options.strict = false;
    options.fallback = fallback_policy::warn;
  }
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ex);
  gpu.require_component(Ey);
  gpu.require_component(Hx);
  gpu.require_component(Hy);
  flux_vol *flux = gpu.add_flux_vol(Z, volume(vec(-0.75, 0.0), vec(0.75, 0.0)));
  gpu.init_backend();
  require_selected_graph_mode(gpu);
  gpu.advance(1);
  const double published = flux->flux();

  nvidia::testing::fail_next(nvidia::testing::failure_point::device_to_host_copy);
  bool failed = false;
  try {
    gpu.advance(1);
  }
  catch (const std::runtime_error &error) {
    failed = std::string(error.what()).find("device-to-host") != std::string::npos;
  }
  nvidia::testing::clear_failure();
  require(failed && gpu.backend->is_poisoned() && flux->flux() == published,
          "failed NVIDIA legacy flux publication did not poison or changed public scalar");
  bool rejected = false;
  try {
    gpu.advance(1);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, "poisoned NVIDIA legacy flux backend accepted a later advance");
  master_printf("nvidia_timestep: flux postlaunch poison PASS\n");
}

static void run_magnetic_sync_case(const char *name, precision_policy_kind policy,
                                   bool real_fields, bool use_bfast) {
  const grid_volume gv = vol2d(3.0, 2.0, 8.0);
  const boundary_region boundaries = pml(0.4, X) + pml(0.4, Y);
  linear_anisotropic_material material(true, true);
  structure cpu_structure(gv, material, boundaries, identity(), 2);
  structure gpu_structure(gv, material, boundaries, identity(), 2);
  set_uniform_conductivity(cpu_structure);
  set_uniform_conductivity(gpu_structure);
  const std::vector<double> scaled_k =
      use_bfast ? std::vector<double>{0.17, -0.11, 0.07} : std::vector<double>(3, 0.0);

  fields cpu(&cpu_structure, 0, 0, true, 0, 0, scaled_k);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(&gpu_structure, options, 0, 0, true, 0, 0, scaled_k);
  if (real_fields) {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  else {
    const vec bloch(0.17, 0.11);
    cpu.use_bloch(bloch);
    gpu.use_bloch(bloch);
  }
  gaussian_src_time source(0.31, 0.12);
  cpu.add_point_source(Ez, source, vec(0.11, 0.13));
  gpu.add_point_source(Ez, source, vec(0.11, 0.13));
  custom_source_trace cpu_b_trace, gpu_b_trace, cpu_h_trace, gpu_h_trace;
  custom_src_time cpu_b_source(custom_source_value, &cpu_b_trace);
  custom_src_time gpu_b_source(custom_source_value, &gpu_b_trace);
  custom_src_time cpu_h_source(custom_source_value, &cpu_h_trace);
  custom_src_time gpu_h_source(custom_source_value, &gpu_h_trace);
  cpu_b_source.is_integrated = gpu_b_source.is_integrated = false;
  cpu_h_source.is_integrated = gpu_h_source.is_integrated = true;
  cpu.add_point_source(Hz, cpu_b_source, vec(-0.31, 0.27),
                       std::complex<double>(0.19, real_fields ? 0.0 : -0.07));
  gpu.add_point_source(Hz, gpu_b_source, vec(-0.31, 0.27),
                       std::complex<double>(0.19, real_fields ? 0.0 : -0.07));
  cpu.add_point_source(Hy, cpu_h_source, vec(0.29, -0.23),
                       std::complex<double>(-0.13, real_fields ? 0.0 : 0.09));
  gpu.add_point_source(Hy, gpu_h_source, vec(0.29, -0.23),
                       std::complex<double>(-0.13, real_fields ? 0.0 : 0.09));
  cpu.require_component(Ez);
  gpu.require_component(Ez);

  cpu.advance(1);
  cpu.t = 0;
  gpu.init_backend();
  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed);
  cpu_b_trace.times.clear();
  gpu_b_trace.times.clear();
  cpu_h_trace.times.clear();
  gpu_h_trace.times.clear();
  const double tolerance = (sizeof(realnum) == sizeof(float) || narrowed) ? 8e-5 : 8e-13;

  cpu.advance(2);
  gpu.advance(2);
  compare_fields(cpu, gpu, tolerance);
  require(cpu_b_trace.times == gpu_b_trace.times && cpu_h_trace.times == gpu_h_trace.times,
          "NVIDIA magnetic ordinary source times differ from CPU");
  cpu_b_trace.times.clear();
  gpu_b_trace.times.clear();
  cpu_h_trace.times.clear();
  gpu_h_trace.times.clear();

  cpu.synchronize_magnetic_fields();
  gpu.synchronize_magnetic_fields();
  compare_fields(cpu, gpu, tolerance);
  require(cpu_b_trace.times == gpu_b_trace.times && cpu_h_trace.times == gpu_h_trace.times &&
              !cpu_b_trace.times.empty() && !cpu_h_trace.times.empty(),
          "NVIDIA direct magnetic synchronization source times differ from CPU");
  cpu.synchronize_magnetic_fields();
  gpu.synchronize_magnetic_fields();
  cpu.restore_magnetic_fields();
  gpu.restore_magnetic_fields();
  compare_fields(cpu, gpu, tolerance);
  cpu.restore_magnetic_fields();
  gpu.restore_magnetic_fields();
  compare_fields(cpu, gpu, tolerance);

  cpu.synchronize_magnetic_fields();
  gpu.synchronize_magnetic_fields();
  cpu_b_trace.times.clear();
  gpu_b_trace.times.clear();
  cpu_h_trace.times.clear();
  gpu_h_trace.times.clear();
  cpu.advance(2);
  gpu.advance(2);
  compare_fields(cpu, gpu, tolerance);
  require(cpu_b_trace.times == gpu_b_trace.times && cpu_h_trace.times == gpu_h_trace.times,
          "NVIDIA trailing magnetic resynchronization source times differ from CPU");
  cpu.restore_magnetic_fields();
  gpu.restore_magnetic_fields();
  compare_fields(cpu, gpu, tolerance);

  master_printf("nvidia_timestep: magnetic-%s/%s PASS\n", name,
                precision_policy_name(policy));
}

static void test_magnetic_pre_step_and_recompile() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  gaussian_src_time initial(0.29, 0.11);
  initial.is_integrated = false;
  cpu.add_point_source(Hz, initial, vec(0.17, -0.19), 0.23);
  gpu.add_point_source(Hz, initial, vec(0.17, -0.19), 0.23);

  /* This is intentionally the first backend operation: it proves the public
     synchronization path prepares storage and compiles an ordinary plan. */
  cpu.synchronize_magnetic_fields();
  gpu.synchronize_magnetic_fields();
  require(gpu.backend_state && gpu.executable,
          "NVIDIA pre-step magnetic synchronization did not prepare resident execution");
  compare_live_fields_by_key(cpu, gpu, sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-13);

  /* A non-integrated source changes only source descriptors/executable shape.
     The live snapshot remains valid because its ordered storage layout did not
     change, and restore must accept the replacement executable. */
  gaussian_src_time added(0.37, 0.09);
  added.is_integrated = false;
  cpu.add_point_source(Hz, added, vec(-0.21, 0.25), -0.17);
  gpu.add_point_source(Hz, added, vec(-0.21, 0.25), -0.17);
  require(is_dirty(gpu, dirty_executable),
          "NVIDIA source-only mutation did not invalidate the executable");
  Executable *old_executable = gpu.executable;
  cpu.restore_magnetic_fields();
  gpu.restore_magnetic_fields();
  require(gpu.executable == old_executable && is_dirty(gpu, dirty_executable),
          "NVIDIA restore did not retain the snapshot-compatible executable");
  compare_live_fields_by_key(cpu, gpu, sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-13);
  cpu.advance(2);
  gpu.advance(2);
  require(gpu.executable && gpu.executable != old_executable &&
              !is_dirty(gpu, dirty_executable),
          "NVIDIA source-only mutation did not recompile on subsequent advance");
  compare_live_fields_by_key(cpu, gpu, sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-13);

  cpu.synchronize_magnetic_fields();
  gpu.synchronize_magnetic_fields();
  gaussian_src_time trailing(0.41, 0.08);
  trailing.is_integrated = false;
  cpu.add_point_source(Hz, trailing, vec(0.23, 0.21), 0.11);
  gpu.add_point_source(Hz, trailing, vec(0.23, 0.21), 0.11);
  old_executable = gpu.executable;
  cpu.advance(1);
  gpu.advance(1);
  require(gpu.executable && gpu.executable != old_executable &&
              !is_dirty(gpu, dirty_executable),
          "NVIDIA live-snapshot source mutation did not recompile for advance");
  compare_live_fields_by_key(cpu, gpu, sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-13);
  cpu.restore_magnetic_fields();
  gpu.restore_magnetic_fields();
  compare_live_fields_by_key(cpu, gpu, sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-13);
  master_printf("nvidia_timestep: magnetic pre-step/recompile PASS\n");
}

static void test_magnetic_historical_host_backups() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure reference_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure migrating_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  fields reference(&reference_structure);
  fields migrating(&migrating_structure);
  reference.use_real_fields();
  migrating.use_real_fields();
  gaussian_src_time source(0.31, 0.12);
  reference.add_point_source(Ez, source, vec(0.11, 0.13));
  migrating.add_point_source(Ez, source, vec(0.11, 0.13));
  reference.advance(2);
  migrating.advance(2);
  migrating.synchronize_magnetic_fields();
  migrating.restore_magnetic_fields();
  bool saw_backup = false;
  for (int chunk = 0; chunk < migrating.num_chunks; ++chunk)
    if (migrating.chunks[chunk]->is_mine()) DOCMP2 FOR_COMPONENTS(c)
      saw_backup = saw_backup || migrating.chunks[chunk]->f_backup[c][cmp] ||
                   migrating.chunks[chunk]->f_u_backup[c][cmp] ||
                   migrating.chunks[chunk]->f_w_backup[c][cmp] ||
                   migrating.chunks[chunk]->f_cond_backup[c][cmp] ||
                   migrating.chunks[chunk]->f_bfast_backup[c][cmp];
  require(saw_backup, "CPU magnetic synchronization did not retain historical host backups");

  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  migrating.select_backend(options);
  migrating.init_backend();
  reference.advance(2);
  migrating.advance(2);
  compare_live_fields_by_key(reference, migrating,
                             sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-13);
  reference.synchronize_magnetic_fields();
  migrating.synchronize_magnetic_fields();
  compare_live_fields_by_key(reference, migrating,
                             sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-13);
  reference.restore_magnetic_fields();
  migrating.restore_magnetic_fields();
  compare_live_fields_by_key(reference, migrating,
                             sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-13);
  master_printf("nvidia_timestep: magnetic historical host backups PASS\n");
}

static void require_material_targets_host_only(fields &gpu) {
  require(gpu.array_catalog != NULL, "NVIDIA material phase has no current catalog");
  for (int chunk = 0; chunk < gpu.num_chunks; ++chunk) {
    if (!gpu.chunks[chunk]->is_mine() || !gpu.chunks[chunk]->new_s) continue;
    const structure_chunk &target = *gpu.chunks[chunk]->new_s;
    FOR_COMPONENTS(c) for (int d = 0; d < 5; ++d) {
      require(!target.chi1inv[c][d] ||
                  !gpu.array_catalog->contains_address(target.chi1inv[c][d]),
              "NVIDIA catalog contains a target chi1inv row");
      require(!target.conductivity[c][d] ||
                  !gpu.array_catalog->contains_address(target.conductivity[c][d]),
              "NVIDIA catalog contains a target conductivity row");
      require(!target.condinv[c][d] ||
                  !gpu.array_catalog->contains_address(target.condinv[c][d]),
              "NVIDIA catalog contains a target condinv row");
    }
  }
}

static bool same_material_initialization_statistics(
    const NvidiaMaterialInitializationStatistics &a,
    const NvidiaMaterialInitializationStatistics &b);

static void run_material_phase_case(const char *name, precision_policy_kind policy,
                                    bool real_fields, bool current_conductivity) {
  const grid_volume gv = vol2d(3.0, 2.0, 8.0);
  const boundary_region boundaries = pml(0.4, X) + pml(0.4, Y);
  linear_anisotropic_material current_material(false);
  linear_anisotropic_material target_material(true);
  structure cpu_structure(gv, current_material, boundaries, identity(), 2);
  structure gpu_structure(gv, current_material, boundaries, identity(), 2);
  structure cpu_target(gv, target_material, boundaries, identity(), 2);
  structure gpu_target(gv, target_material, boundaries, identity(), 2);
  if (current_conductivity) {
    set_uniform_conductivity(cpu_structure);
    set_uniform_conductivity(gpu_structure);
  }
  else {
    cpu_target.set_conductivity(Dz, phase_target_conductivity);
    gpu_target.set_conductivity(Dz, phase_target_conductivity);
  }

  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&gpu_structure, options);
  if (real_fields) {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  else {
    const vec bloch(0.17, -0.11);
    cpu.use_bloch(bloch);
    gpu.use_bloch(bloch);
  }
  cpu.require_component(Ez);
  gpu.require_component(Ez);
  cpu.advance(1);
  cpu.t = 0;
  gpu.init_backend();
  require_selected_graph_mode(gpu);
  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_live_fields_by_key(cpu, gpu, narrowed, 0.35);

  require(cpu.phase_in_material(&cpu_target, 4.0 * cpu.dt) == 4 &&
              gpu.phase_in_material(&gpu_target, 4.0 * gpu.dt) == 4,
          "material phase setup returned the wrong countdown");
  gpu.init_backend();
  require_selected_graph_mode(gpu);
  gpu.synchronize_magnetic_fields();
  gpu.restore_magnetic_fields();
  require_material_targets_host_only(gpu);
  const StepPlan plan = build_step_plan(gpu, StepProgram::ordinary);
  require(!plan.material_refresh_arrays.empty(),
          "NVIDIA material phase plan has no refresh rows");
  size_t expected_bytes = 0;
  for (const MaterialRefreshArray &row : plan.material_refresh_arrays)
    expected_bytes +=
        row.elements * storage_element_bytes(ElementType::realnum_value,
                                             policy_for(policy).material);

  const double tolerance = (sizeof(realnum) == sizeof(float) || narrowed) ? 8e-5 : 8e-13;
  for (int step = 1; step <= 5; ++step) {
    cpu.advance(1);
    nvidia::testing::reset_transfer_accounting();
    backend_reset_material_phase_preflight_counts_for_testing();
    gpu.advance(1);
    const nvidia::testing::transfer_accounting transfers =
        nvidia::testing::current_transfer_accounting();
    const bool active = step <= 4;
    require(transfers.host_to_device_calls ==
                (active ? plan.material_refresh_arrays.size() : 0),
            "NVIDIA material phase issued the wrong number of refresh uploads");
    require(transfers.host_to_device_bytes == (active ? expected_bytes : 0),
            "NVIDIA material phase uploaded the wrong byte count");
    require((active && backend_material_phase_collective_count_for_testing() != 0 &&
             backend_material_phase_scan_count_for_testing() != 0) ||
                (!active && backend_material_phase_collective_count_for_testing() == 0 &&
                 backend_material_phase_scan_count_for_testing() == 0),
            "NVIDIA material phase preflight did not enter or leave its cached collective path");
    compare_fields(cpu, gpu, tolerance);
    compare_material_rows(cpu, gpu, tolerance);
  }
  NvidiaBackend *const backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend && gpu.backend_state && !gpu.backend_state->material_phase_active &&
              !gpu.executable->material_phase_active,
          "completed NVIDIA material phase retained its compiled active flag");
  const NvidiaMaterialInitializationStatistics initialization_before =
      backend->material_initialization_statistics_for_testing();
  nvidia::testing::reset_material_transfer_accounting();
  backend_reset_material_phase_preflight_counts_for_testing();
  cpu.advance(100);
  gpu.advance(100);
  const nvidia::testing::material_transfer_accounting material_transfers =
      nvidia::testing::current_material_transfer_accounting();
  require(material_transfers.compact_calls == 0 && material_transfers.compact_bytes == 0 &&
              material_transfers.dense_output_calls == 0 &&
              material_transfers.dense_output_bytes == 0 &&
              material_transfers.tiled_output_calls == 0 &&
              material_transfers.tiled_output_bytes == 0 &&
              backend_material_phase_collective_count_for_testing() == 0 &&
              backend_material_phase_scan_count_for_testing() == 0 &&
              same_material_initialization_statistics(
                  initialization_before,
                  backend->material_initialization_statistics_for_testing()),
          "completed material phase repeated material work during 100 clean steps");
  compare_fields(cpu, gpu, tolerance);
  master_printf("nvidia_timestep: material-%s/%s PASS\n", name,
                precision_policy_name(policy));
}

static void install_native_homogeneous_material(structure &s,
                                                meep_geom::absorber_list absorbers = 0) {
  using namespace meep_geom;
  material_type material = new material_data();
  material->which_subclass = material_data::MEDIUM;
  material->medium = medium_struct(2.75);
  material->medium.epsilon_diag = make_vector3(2.75, 3.125, 3.5);
  material->medium.epsilon_offdiag.x.re = 0.11;
  material->medium.epsilon_offdiag.y.re = -0.07;
  material->medium.epsilon_offdiag.z.re = 0.05;
  material->medium.mu_diag = make_vector3(1.25, 1.375, 1.5);
  material->medium.mu_offdiag.x.re = -0.03;
  material->medium.mu_offdiag.y.re = 0.02;
  material->medium.mu_offdiag.z.re = 0.04;
  material->medium.E_chi2_diag = make_vector3(0.13, -0.17, 0.19);
  material->medium.E_chi3_diag = make_vector3(0.021, 0.023, -0.027);
  material->medium.H_chi2_diag = make_vector3(-0.09, 0.07, 0.05);
  material->medium.H_chi3_diag = make_vector3(0.031, -0.029, 0.037);
  material->medium.D_conductivity_diag = make_vector3(0.015, 0.025, 0.035);
  material->medium.B_conductivity_diag = make_vector3(0.012, 0.022, 0.032);
  meep_geom::susceptibility electric = meep_geom::susceptibility();
  electric.frequency = 0.47;
  electric.gamma = 0.061;
  electric.sigma_diag = make_vector3(0.8, 0.7, 0.6);
  electric.sigma_offdiag = make_vector3(0.04, -0.03, 0.02);
  material->medium.E_susceptibilities.push_back(electric);
  meep_geom::susceptibility electric_duplicate = electric;
  electric_duplicate.sigma_diag = make_vector3(8.0, 7.0, 6.0);
  electric_duplicate.sigma_offdiag = make_vector3(0.4, -0.3, 0.2);
  material->medium.E_susceptibilities.push_back(electric_duplicate);
  meep_geom::susceptibility electric_second = meep_geom::susceptibility();
  electric_second.frequency = 0.73;
  electric_second.gamma = 0.083;
  electric_second.sigma_diag = make_vector3(0.31, 0.29, 0.23);
  electric_second.sigma_offdiag = make_vector3(-0.017, 0.019, -0.013);
  material->medium.E_susceptibilities.push_back(electric_second);
  meep_geom::susceptibility magnetic = meep_geom::susceptibility();
  magnetic.frequency = 0.39;
  magnetic.gamma = 0.047;
  magnetic.drude = true;
  magnetic.sigma_diag = make_vector3(0.5, 0.4, 0.3);
  magnetic.sigma_offdiag = make_vector3(-0.02, 0.015, 0.01);
  material->medium.H_susceptibilities.push_back(magnetic);
  meep_geom::susceptibility magnetic_duplicate = magnetic;
  magnetic_duplicate.sigma_diag = make_vector3(5.0, 4.0, 3.0);
  magnetic_duplicate.sigma_offdiag = make_vector3(-0.2, 0.15, 0.1);
  material->medium.H_susceptibilities.push_back(magnetic_duplicate);
  geometric_object_list empty_geometry = {0, NULL};
  set_materials_from_geometry(&s, empty_geometry, make_vector3(), true, 1e-5, 256, false,
                              material, absorbers);
  material_free(material);
}

static double native_cubic_pml_profile(double u, void *) { return u * u * u; }

static void require_native_ir_preflight_rejected(NvidiaBackend &backend,
                                                 const InitializationPlan &canonical_plan,
                                                 MaterialIR mutated, const char *message) {
  require(canonical_plan.materials.size() == 1,
          "native IR mutation fixture has no canonical material recipe");
  refresh_material_ir_signatures_for_testing(mutated);
  const MaterialRecipe &canonical = canonical_plan.materials[0];
  bool rejected = false;
  try {
    MaterialRecipeInput input;
    input.disposition = canonical.disposition();
    input.description = canonical.description();
    input.eps_averaging = canonical.eps_averaging();
    input.subpixel_tol = canonical.subpixel_tol();
    input.subpixel_maxeval = canonical.subpixel_maxeval();
    input.host_callback_id = canonical.host_callback_id();
    input.from_host_callback = canonical.from_host_callback();
    input.support_reason_bits = canonical.support_reason_bits();
    input.rows = canonical.rows();
    input.dense_fallback_rows = canonical.dense_fallback_rows();
    input.callback_tiles = canonical.callback_tiles();
    input.callback_owners = canonical.callback_owners();
    input.topology = canonical.topology();
    input.ir.reset(new MaterialIR(mutated));
    InitializationPlan plan = canonical_plan;
    plan.materials.clear();
    plan.materials.push_back(MaterialRecipe(input));
    backend.preflight_initialization(plan);
  }
  catch (const std::invalid_argument &error) {
    rejected = std::string(error.what()).find("canonical owned snapshot") != std::string::npos;
  }
  require(rejected, message);
}

static void test_native_absorber_initialization(precision_policy_kind precision) {
  using namespace meep_geom;
  const grid_volume gv = vol1d(2.0, 8.0);
  const boundary_region custom_pml =
      boundary_region(boundary_region::PML, 0.25, 1e-15, 1.4,
                      native_cubic_pml_profile, NULL, 1.0 / 4.0, 1.0 / 5.0, Z, Low) +
      boundary_region(boundary_region::PML, 0.25, 1e-15, 1.4,
                      native_cubic_pml_profile, NULL, 1.0 / 4.0, 1.0 / 5.0, Z, High);
  structure cpu_structure(gv, isotropic_eps, custom_pml, identity(), 1);
  structure gpu_structure(gv, isotropic_eps, custom_pml, identity(), 1);
  absorber_list absorbers = create_absorber_list();
  add_absorbing_layer(absorbers, 0.5, Z, ALL_SIDES, 1e-9, 1.0);
  install_native_homogeneous_material(cpu_structure, absorbers);
  install_native_homogeneous_material(gpu_structure, absorbers);
  destroy_absorber_list(absorbers);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  FOR_E_AND_H(c) if (gv.has_field(c)) {
      cpu.require_component(c);
      gpu.require_component(c);
    }
  cpu.init_backend();
  nvidia::testing::reset_transfer_accounting();
  nvidia::testing::reset_material_transfer_accounting();
  gpu.init_backend();
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend != NULL, "absorber material fixture did not select NVIDIA");
  if (count_processors() == 2)
    require(backend->device_ordinal_for_testing() == my_rank(),
            "two-rank native material test did not select one rank per GPU");
  const NvidiaMaterialInitializationStatistics statistics =
      backend->material_initialization_statistics_for_testing();
  const nvidia::testing::transfer_accounting initialization_transfers =
      nvidia::testing::current_transfer_accounting();
  const nvidia::testing::material_transfer_accounting material_transfers =
      nvidia::testing::current_material_transfer_accounting();
  const MaterialIR *ir = material_ir_for(gpu);
  require(ir && !ir->absorbers.empty() && !ir->pml_axes.empty(),
          "absorber/PML native fixture lost its owned profiles");
  require(gpu.initialization_plan != NULL,
          "absorber/PML native fixture lost its initialization plan");
  MaterialIR mutated = *ir;
  mutated.materials[mutated.default_material].parameters[0] += 0.125;
  mutated.materials[mutated.default_material].comparison_medium[0] += 0.125;
  require_native_ir_preflight_rejected(*backend, *gpu.initialization_plan, mutated,
                                       "re-signed medium mutation passed native preflight");
  mutated = *ir;
  mutated.absorbers[0].samples[0] += 0.125;
  require_native_ir_preflight_rejected(*backend, *gpu.initialization_plan, mutated,
                                       "re-signed absorber mutation passed native preflight");
  mutated = *ir;
  mutated.cell[0] += 0.125;
  require_native_ir_preflight_rejected(*backend, *gpu.initialization_plan, mutated,
                                       "re-signed cell mutation passed native preflight");
  for (const MaterialIRPmlAxis &axis : ir->pml_axes)
    if (axis.profile_active)
      require(!axis.analytic_quadratic && axis.mean_stretch == 1.4 &&
                  axis.profile_integral == 1.0 / 4.0 &&
                  axis.profile_integral_u == 1.0 / 5.0,
              "custom sampled PML provenance was not preserved");
  size_t rows = 0, pairs = 0, points = 0, absorber_bytes = 0, pml_bytes = 0;
  for (const MaterialIRPml &absorber : ir->absorbers)
    absorber_bytes += absorber.samples.size() * sizeof(double);
  for (const MaterialIRPmlAxis &axis : ir->pml_axes)
    if (axis.profile_active) pml_bytes += axis.profile_samples.size() * sizeof(double);
  for (const MaterialIRTopologyRow &row : ir->topology) {
    const array_kind kind = array_kind(row.key.kind);
    if (kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
        kind == array_kind::pml_siginv)
      continue;
    ++rows;
    const component c = component(row.key.component_);
    if (kind == array_kind::conductivity && (is_D(c) || is_B(c)) &&
        direction(row.key.aux) == component_direction(c)) {
      ++pairs;
      const MaterialIRChunk &chunk = ir->chunks[size_t(row.key.chunk)];
      points += chunk.loop_count[c];
    }
  }
  const size_t header_bytes = ir->absorbers.size() * sizeof(nvidia::material_absorber_header);
  const size_t expected_compact = header_bytes + absorber_bytes + pml_bytes;
  const size_t candidate_attempts = 1 + gpu.classification_reentries;
  require(statistics.valid && statistics.device_native &&
              statistics.owned_ir_bytes >= statistics.compact_input_host_to_device_bytes &&
              statistics.dense_oracle_bytes == 0 &&
              statistics.dense_output_host_to_device_calls == 0 &&
              statistics.dense_output_host_to_device_bytes == 0 &&
              statistics.compact_input_host_to_device_calls == 1 &&
              statistics.compact_input_host_to_device_bytes == expected_compact &&
              statistics.decoded_parameter_bytes == header_bytes &&
              statistics.absorber_profile_bytes == absorber_bytes &&
              statistics.pml_profile_bytes == pml_bytes &&
              statistics.constant_fill_kernel_launches == 2 * rows - 2 * pairs &&
              statistics.conductivity_kernel_launches == pairs &&
              statistics.pointwise_kernel_launches == 2 * rows - pairs &&
              statistics.pml_kernel_launches == ir->pml_axes.size() &&
              statistics.absorber_points_evaluated == points &&
              statistics.synchronizations == 1,
          "absorber/PML native initialization accounting is not exact");
  require(material_transfers.dense_output_calls == 0 &&
              material_transfers.dense_output_bytes == 0 &&
              material_transfers.compact_calls == candidate_attempts &&
              material_transfers.compact_bytes == candidate_attempts * expected_compact &&
              initialization_transfers.host_to_device_calls >= material_transfers.compact_calls &&
              initialization_transfers.host_to_device_bytes >= material_transfers.compact_bytes,
          "global/tagged transfer accounting found a dense absorber material upload");
  const double tolerance =
      (sizeof(realnum) == sizeof(float) || precision != precision_policy_kind::native) ? 8e-6
                                                                                       : 2e-13;
  compare_all_initialized_material_rows(cpu, gpu, tolerance);
  master_printf("nvidia_timestep: native-absorber/%s PASS\n", precision_policy_name(precision));
}

static void test_native_material_initialization_retry() {
  using namespace meep_geom;
  const nvidia::testing::failure_point failures[] = {
      nvidia::testing::failure_point::material_compact_allocate,
      nvidia::testing::failure_point::material_ir_upload,
      nvidia::testing::failure_point::material_pointwise_launch,
      nvidia::testing::failure_point::material_pml_launch,
      nvidia::testing::failure_point::material_initialization_sync};
  for (size_t failure_index = 0;
       failure_index < sizeof(failures) / sizeof(failures[0]); ++failure_index) {
    const nvidia::testing::failure_point failure = failures[failure_index];
    const grid_volume gv = vol1d(2.0, 8.0);
    structure s(gv, isotropic_eps, pml(0.25), identity(), 1);
    absorber_list absorbers = create_absorber_list();
    add_absorbing_layer(absorbers, 0.5, Z, ALL_SIDES, 1e-9, 1.0);
    install_native_homogeneous_material(s, absorbers);
    destroy_absorber_list(absorbers);
    execution_options options;
    options.backend = backend_kind::nvidia;
    fields f(&s, options);
    f.use_real_fields();
    f.require_component(Ex);
    const nvidia::memory_accounting before = nvidia::current_memory_accounting();
    nvidia::testing::fail_next(failure);
    bool rejected = false;
    try { f.init_backend(); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
    const nvidia::memory_accounting after = nvidia::current_memory_accounting();
    require(rejected && f.backend_state == NULL && f.executable == NULL &&
                before.device_bytes_current == after.device_bytes_current &&
                before.pinned_bytes_current == after.pinned_bytes_current,
            "failed cold native material initialization published or leaked a candidate");
    f.init_backend();
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(f.backend);
    require(backend && f.backend_state && f.executable &&
                backend->material_initialization_statistics_for_testing().valid,
            "cold native material initialization failure was not retryable");
  }

  const grid_volume gv = vol1d(2.0, 8.0);
  structure s(gv, isotropic_eps, pml(0.25), identity(), 1);
  absorber_list absorbers = create_absorber_list();
  add_absorbing_layer(absorbers, 0.5, Z, ALL_SIDES, 1e-9, 1.0);
  install_native_homogeneous_material(s, absorbers);
  destroy_absorber_list(absorbers);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields f(&s, options);
  f.use_real_fields();
  f.require_component(Ex);
  f.init_backend();
  BackendState *const live_state = f.backend_state;
  Executable *const live_executable = f.executable;
  invalidate(f, MutationKind::material_values,
             "native material initialization replacement retry");
  nvidia::testing::fail_next(nvidia::testing::failure_point::material_pml_launch);
  bool rejected = false;
  try { f.init_backend(); }
  catch (const std::exception &) { rejected = true; }
  nvidia::testing::clear_failure();
  require(rejected, "injected native material replacement failure was ignored");
  require(f.backend_state == live_state,
          "failed native material replacement changed the live state");
  require(f.executable == live_executable,
          "failed native material replacement changed the live executable");
  require(!f.backend->is_poisoned(),
          "failed native material replacement poisoned the live backend");
  f.init_backend();
  require(f.backend_state != live_state && f.executable != live_executable,
          "native material replacement retry did not commit a new epoch");
  master_printf("nvidia_timestep: native-material rollback/retry PASS\n");
}

static void install_native_file_material(structure &s, bool periodicity = false,
                                         double sample_delta = 0.0, int sample_axis = 2) {
  using namespace meep_geom;
  material_type file_material = new material_data();
  file_material->which_subclass = material_data::MATERIAL_FILE;
  file_material->medium = medium_struct(7.0);
  file_material->medium.mu_diag = make_vector3(3.0, 4.0, 5.0);
  file_material->medium.D_conductivity_diag = make_vector3(0.2, 0.3, 0.4);
  file_material->medium.E_chi2_diag = make_vector3(0.1, 0.2, 0.3);
  file_material->epsilon_dims[0] = file_material->epsilon_dims[1] =
      file_material->epsilon_dims[2] = 1;
  require(sample_axis >= 0 && sample_axis < 3, "FILE sample axis is invalid");
  file_material->epsilon_dims[sample_axis] = 2;
  file_material->epsilon_data = new double[2]{1.5 + sample_delta, 2.5 - 0.5 * sample_delta};
  geometric_object_list empty_geometry = {0, NULL};
  set_materials_from_geometry(&s, empty_geometry, make_vector3(), false, 1e-5, 64, periodicity,
                              file_material);
  material_free(file_material);
}

static void install_native_grid_material(structure &s, meep_geom::absorber_list absorbers,
                                         bool endpoint_conductivity = true,
                                         bool periodicity = false,
                                         bool dispersion = true, double weight_delta = 0.0,
                                         double projection_offset = 0.0,
                                         double damping_delta = 0.0,
                                         double endpoint_delta = 0.0,
                                         int sample_axis = 2) {
  using namespace meep_geom;
  material_type grid = make_material_grid(false, 3.5, 0.4, -0.125 + damping_delta);
  grid->material_grid_kinds = material_data::U_MIN;
  require(sample_axis >= 0 && sample_axis < 3, "MaterialGrid sample axis is invalid");
  grid->grid_size = sample_axis == 0 ? make_vector3(2, 1, 1)
                    : sample_axis == 1 ? make_vector3(1, 2, 1)
                                       : make_vector3(1, 1, 2);
  grid->weights = new double[2]{-0.25 + weight_delta, 1.25 - 0.5 * weight_delta};
  grid->medium_1 = medium_struct(2.0);
  grid->medium_2 = medium_struct(5.0);
  grid->medium_1.epsilon_diag = make_vector3(2.0, 3.0, 4.0);
  grid->medium_2.epsilon_diag =
      make_vector3(5.0 + endpoint_delta, 6.0 - 0.5 * endpoint_delta,
                   7.0 + 0.25 * endpoint_delta);
  grid->medium_1.epsilon_offdiag.x.re = 0.1;
  grid->medium_1.epsilon_offdiag.y.re = -0.05;
  grid->medium_1.epsilon_offdiag.z.re = 0.08;
  grid->medium_2.epsilon_offdiag.x.re = -0.07;
  grid->medium_2.epsilon_offdiag.y.re = 0.04;
  grid->medium_2.epsilon_offdiag.z.re = -0.03;
  if (endpoint_conductivity) {
    grid->medium_1.D_conductivity_diag = make_vector3(0.2, 0.3, 0.4);
    grid->medium_2.D_conductivity_diag = make_vector3(0.7, 0.8, 0.9);
  }
  if (dispersion) {
    meep_geom::susceptibility shared = meep_geom::susceptibility();
    shared.frequency = 0.61;
    shared.gamma = 0.07;
    shared.sigma_diag = make_vector3(1.0, 0.5, 0.25);
    meep_geom::susceptibility duplicate = shared;
    duplicate.sigma_diag = make_vector3(9.0, 8.0, 7.0);
    meep_geom::susceptibility distinct = meep_geom::susceptibility();
    distinct.frequency = shared.frequency;
    distinct.gamma = shared.gamma;
    distinct.bias = make_vector3(0.0, 0.0, 0.125);
    distinct.sigma_diag = make_vector3(0.2, 0.2, 0.2);
    grid->medium_1.E_susceptibilities.push_back(shared);
    grid->medium_2.E_susceptibilities.push_back(duplicate);
    grid->medium_2.E_susceptibilities.push_back(distinct);
    grid->medium.E_susceptibilities = grid->medium_1.E_susceptibilities;
    grid->medium.E_susceptibilities.insert(grid->medium.E_susceptibilities.end(),
                                           grid->medium_2.E_susceptibilities.begin(),
                                           grid->medium_2.E_susceptibilities.end());
  }
  material_type endpoint_1 = new material_data();
  material_type endpoint_2 = new material_data();
  endpoint_1->medium = grid->medium_1;
  endpoint_2->medium = grid->medium_2;
  material_type extras_data[2] = {endpoint_1, endpoint_2};
  material_type_list extras;
  extras.items = extras_data;
  extras.num_items = 2;
  geometric_object_list empty_geometry = {0, NULL};
  geom_epsilon *geps = make_geom_epsilon(&s, &empty_geometry, make_vector3(), periodicity,
                                         grid, extras);
  geps->u_p = projection_offset;
  set_materials_from_geom_epsilon(&s, geps, false, 1e-5, 64, absorbers);
  delete geps;
  material_free(endpoint_1);
  material_free(endpoint_2);
  material_free(grid);
}

static bool same_material_initialization_statistics(
    const NvidiaMaterialInitializationStatistics &a,
    const NvidiaMaterialInitializationStatistics &b) {
#define SAME_STAT(field) a.field == b.field
  return SAME_STAT(compact_input_host_to_device_calls) &&
         SAME_STAT(compact_input_host_to_device_bytes) && SAME_STAT(owned_ir_bytes) &&
         SAME_STAT(dense_oracle_bytes) && SAME_STAT(dense_output_host_to_device_calls) &&
         SAME_STAT(dense_output_host_to_device_bytes) &&
         SAME_STAT(tiled_output_host_to_device_calls) &&
         SAME_STAT(tiled_output_host_to_device_bytes) && SAME_STAT(logical_output_bytes) &&
         SAME_STAT(callback_scratch_bytes) && SAME_STAT(upload_descriptor_bytes) &&
         SAME_STAT(classification_status_bytes) && SAME_STAT(classification_result_bytes) &&
         SAME_STAT(decoded_parameter_bytes) &&
         SAME_STAT(absorber_profile_bytes) && SAME_STAT(pml_profile_bytes) &&
         SAME_STAT(file_sample_bytes) && SAME_STAT(grid_weight_bytes) &&
         SAME_STAT(geometry_object_bytes) && SAME_STAT(geometry_image_bytes) &&
         SAME_STAT(geometry_value_bytes) && SAME_STAT(geometry_analytic_bytes) &&
         SAME_STAT(geometry_patch_bytes) &&
         SAME_STAT(constant_fill_kernel_launches) && SAME_STAT(conductivity_kernel_launches) &&
         SAME_STAT(file_table_kernel_launches) && SAME_STAT(grid_table_kernel_launches) &&
         SAME_STAT(geometry_bulk_kernel_launches) &&
         SAME_STAT(geometry_analytic_kernel_launches) &&
         SAME_STAT(geometry_patch_kernel_launches) &&
         SAME_STAT(pointwise_kernel_launches) && SAME_STAT(pml_kernel_launches) &&
         SAME_STAT(absorber_points_evaluated) && SAME_STAT(file_points_evaluated) &&
         SAME_STAT(grid_points_evaluated) && SAME_STAT(geometry_bulk_points) &&
         SAME_STAT(geometry_analytic_points) && SAME_STAT(geometry_patch_points) &&
         SAME_STAT(synchronizations) &&
         SAME_STAT(device_native) && SAME_STAT(valid);
#undef SAME_STAT
}

static void install_material_oracle_from(fields &source, fields &destination) {
  require(source.array_catalog && destination.array_catalog,
          "semantic material replacement requires prepared CPU catalogs");
  for (size_t i = 0; i < source.array_catalog->size(); ++i) {
    const ArrayId source_id{uint32_t(i)};
    const ArraySpec &source_spec = source.array_catalog->spec(source_id);
    if (source_spec.role != array_role::material || is_valid(source_spec.alias_of) ||
        source_spec.classification_elided)
      continue;
    const StorageKey &key = source.array_catalog->key(source_id);
    const ArrayId destination_id = destination.array_catalog->find(key);
    require(is_valid(destination_id),
            "semantic material replacement changed retained storage topology");
    const ArraySpec &destination_spec = destination.array_catalog->spec(destination_id);
    require(destination_spec.element_type == source_spec.element_type &&
                destination_spec.elements == source_spec.elements,
            "semantic material replacement changed retained storage shape");
    memcpy(destination.array_catalog->resolve_untyped(destination_id),
           source.array_catalog->resolve_untyped(source_id),
           source_spec.elements * host_element_bytes(source_spec.element_type));
  }
  destination.material_ir = source.material_ir;
}

static void test_native_table_materials(precision_policy_kind precision, bool real_fields) {
  using namespace meep_geom;
  for (int table_kind = 0; table_kind < 2; ++table_kind) {
    const grid_volume gv = table_kind ? vol3d(1.0, 1.0, 2.0, 4.0) : vol1d(2.0, 8.0);
    structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
    structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
    absorber_list absorbers = create_absorber_list();
    if (table_kind) add_absorbing_layer(absorbers, 0.5, Z, ALL_SIDES, 1e-9, 1.0);
    if (table_kind) {
      install_native_grid_material(cpu_structure, absorbers);
      install_native_grid_material(gpu_structure, absorbers);
    }
    else {
      install_native_file_material(cpu_structure);
      install_native_file_material(gpu_structure);
    }
    destroy_absorber_list(absorbers);
    fields cpu(&cpu_structure);
    execution_options options;
    options.backend = backend_kind::nvidia;
    options.precision = precision;
    options.strict = false;
    options.fallback = fallback_policy::warn;
    fields gpu(&gpu_structure, options);
    if (real_fields) {
      cpu.use_real_fields();
      gpu.use_real_fields();
    }
    else {
      const vec k = table_kind ? vec(0.125, 0.0625, -0.03125) : vec(0.125);
      cpu.use_bloch(k);
      gpu.use_bloch(k);
    }
    FOR_E_AND_H(c) if (gv.has_field(c)) {
        cpu.require_component(c);
        gpu.require_component(c);
      }
    cpu.init_backend();
    nvidia::testing::reset_material_transfer_accounting();
    gpu.init_backend();
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
    const NvidiaMaterialInitializationStatistics statistics =
        backend ? backend->material_initialization_statistics_for_testing()
                : NvidiaMaterialInitializationStatistics();
    const nvidia::testing::material_transfer_accounting transfers =
        nvidia::testing::current_material_transfer_accounting();
    const MaterialIR *table_ir = material_ir_for(gpu);
    require(table_ir, "table material fixture lost its owned IR");
    const auto append_packed = [](size_t &offset, size_t bytes, size_t alignment) {
      offset += (alignment - offset % alignment) % alignment;
      offset += bytes;
    };
    size_t expected_compact = 0;
    append_packed(expected_compact, sizeof(nvidia::material_table_header),
                  alignof(nvidia::material_table_header));
    if (table_kind) {
      append_packed(expected_compact, sizeof(nvidia::material_medium_header),
                    alignof(nvidia::material_medium_header));
      append_packed(expected_compact, sizeof(nvidia::material_susceptibility_record),
                    alignof(nvidia::material_susceptibility_record));
      append_packed(expected_compact, sizeof(nvidia::material_medium_header),
                    alignof(nvidia::material_medium_header));
      append_packed(expected_compact, 2 * sizeof(nvidia::material_susceptibility_record),
                    alignof(nvidia::material_susceptibility_record));
    }
    const size_t table_sample_bytes = 2 * sizeof(double);
    append_packed(expected_compact, table_sample_bytes, alignof(double));
    size_t absorber_bytes = 0, pml_bytes = 0;
    if (!table_ir->absorbers.empty()) {
      append_packed(expected_compact,
                    table_ir->absorbers.size() * sizeof(nvidia::material_absorber_header),
                    alignof(nvidia::material_absorber_header));
      for (const MaterialIRPml &absorber : table_ir->absorbers) {
        const size_t bytes = absorber.samples.size() * sizeof(double);
        append_packed(expected_compact, bytes, alignof(double));
        absorber_bytes += bytes;
      }
    }
    for (const MaterialIRPmlAxis &axis : table_ir->pml_axes)
      if (axis.profile_active) {
        const size_t bytes = axis.profile_samples.size() * sizeof(double);
        append_packed(expected_compact, bytes, alignof(double));
        pml_bytes += bytes;
      }
    const size_t expected_decoded =
        expected_compact - table_sample_bytes - absorber_bytes - pml_bytes;
    const size_t candidate_attempts = 1 + gpu.classification_reentries;
    require(backend && statistics.valid && statistics.device_native &&
                statistics.synchronizations == 1 &&
                statistics.dense_output_host_to_device_calls == 0 &&
                statistics.dense_output_host_to_device_bytes == 0 &&
                statistics.compact_input_host_to_device_calls == 1 &&
                statistics.compact_input_host_to_device_bytes == expected_compact &&
                statistics.decoded_parameter_bytes == expected_decoded &&
                statistics.absorber_profile_bytes == absorber_bytes &&
                statistics.pml_profile_bytes == pml_bytes &&
                transfers.dense_output_calls == 0 && transfers.dense_output_bytes == 0 &&
                transfers.compact_calls == candidate_attempts &&
                transfers.compact_bytes == candidate_attempts * expected_compact,
            "table material did not preserve native initialization accounting");
    if (table_kind)
      require(statistics.grid_table_kernel_launches > 0 &&
                  statistics.file_table_kernel_launches == 0 &&
                  statistics.grid_weight_bytes == 2 * sizeof(double),
              "MaterialGrid table execution/accounting is absent");
    else
      require(statistics.file_table_kernel_launches > 0 &&
                  statistics.grid_table_kernel_launches == 0 &&
                  statistics.file_sample_bytes == 2 * sizeof(double),
              "FILE table execution/accounting is absent");
    if (!table_kind) {
      bool saw_electric_diagonal = false;
      for (size_t i = 0; i < gpu.array_catalog->size(); ++i) {
        const StorageKey &key = gpu.array_catalog->key(ArrayId{uint32_t(i)});
        if (gpu.array_catalog->find(key) != ArrayId{uint32_t(i)}) continue;
        if (key.kind != int(array_kind::chi1inv) &&
            key.kind != int(array_kind::conductivity) && key.kind != int(array_kind::condinv) &&
            key.kind != int(array_kind::chi2) && key.kind != int(array_kind::chi3) &&
            key.kind != int(array_kind::sigma))
          continue;
        const component c = component(key.component_);
        const bool electric_diagonal =
            key.kind == int(array_kind::chi1inv) && is_electric(c) &&
            direction(key.aux) == component_direction(c);
        require(electric_diagonal,
                "FILE retained a non-electric or non-diagonal material property row");
        saw_electric_diagonal = true;
      }
      require(saw_electric_diagonal &&
                  gpu_structure.get_chi1inv(Hx, X, gv.center(), 0, false) ==
                      std::complex<double>(1.0, 0.0),
              "FILE did not retain electric-only chi1 semantics with identity H");
    }
    if (table_kind) {
      const MaterialIR *ir = material_ir_for(gpu);
      require(ir, "MaterialGrid fixture lost its owned IR");
      size_t retained_sigma = 0;
      std::set<uint32_t> identities;
      std::map<int, std::vector<uint32_t> > retained_identity_order;
      for (const MaterialIRSusceptibility &sus : ir->susceptibilities)
        if (sus.field_type == E_stuff) identities.insert(sus.identity);
      for (size_t i = 0; i < gpu.array_catalog->size(); ++i) {
        const ArrayId id{uint32_t(i)};
        const StorageKey &key = gpu.array_catalog->key(id);
        if (key.kind != int(array_kind::sigma)) continue;
        const ArraySpec &spec = gpu.array_catalog->spec(id);
        if (spec.classification_elided) continue;
        const ArrayId canonical = gpu.array_catalog->find(key);
        require(is_valid(canonical),
                "retained MaterialGrid sigma key has no canonical catalog identity");
        if (canonical != id) continue;
        const field_type ft = field_type(key.aux % uint64_t(NUM_FIELD_TYPES));
        const uint32_t identity = uint32_t(key.aux / uint64_t(NUM_FIELD_TYPES));
        const MaterialIRSusceptibility *owned_identity = NULL;
        for (const MaterialIRSusceptibility &sus : ir->susceptibilities)
          if (sus.field_type == ft && sus.identity == identity) {
            owned_identity = &sus;
            break;
          }
        require(ft == E_stuff && owned_identity && key.cmp >= 0 && key.cmp < 5,
                "retained MaterialGrid sigma row has stale identity/state metadata");
        std::vector<uint32_t> &component_order = retained_identity_order[key.component_];
        if (component_order.empty() || component_order.back() != identity)
          component_order.push_back(identity);
        std::vector<realnum> observed(spec.elements);
        gpu.backend->read(ArrayRef{id, 0, spec.elements}, observed.data(),
                          observed.size() * sizeof(realnum));
        for (realnum value : observed)
          require(std::isfinite(double(value)),
                  "retained MaterialGrid sigma row contains a non-finite value");
        ++retained_sigma;
      }
      bool ordered = !retained_identity_order.empty();
      for (const std::pair<const int, std::vector<uint32_t> > &entry : retained_identity_order)
        ordered = ordered && entry.second.size() == 2 && entry.second[0] == 0 &&
                  entry.second[1] == 1;
      require(ir && identities.size() == 2 && *identities.begin() == 0 &&
                  *identities.rbegin() == 1 && retained_sigma > 0 && ordered,
              "MaterialGrid recurrence identities or retained sigma rows are absent");
    }
    compare_all_initialized_material_rows(
        cpu, gpu, sizeof(realnum) == sizeof(float) || precision != precision_policy_kind::native
                      ? 8e-6
                      : 2e-13);
    gpu.init_backend();
    const NvidiaMaterialInitializationStatistics reentered =
        backend->material_initialization_statistics_for_testing();
    require(reentered.file_table_kernel_launches == statistics.file_table_kernel_launches &&
                reentered.grid_table_kernel_launches == statistics.grid_table_kernel_launches &&
                reentered.compact_input_host_to_device_bytes ==
                    statistics.compact_input_host_to_device_bytes &&
                reentered.synchronizations == statistics.synchronizations,
            "clean table-material re-entry repeated initialization work");
    initialize_live_fields_by_key(cpu, gpu, precision != precision_policy_kind::native, 0.11);
    cpu.advance(2);
    gpu.advance(2);
    compare_live_fields_by_key(
        cpu, gpu, (sizeof(realnum) == sizeof(float) || precision != precision_policy_kind::native)
                      ? 2e-4
                      : 5e-12);
  }
  master_printf("nvidia_timestep: native FILE/MaterialGrid tables %s/%s PASS\n",
                real_fields ? "real" : "complex", precision_policy_name(precision));
}

static void test_native_table_dimension_matrix(precision_policy_kind precision) {
  using namespace meep_geom;
  struct DimensionCase {
    const char *name;
    grid_volume volume;
    int sample_axis;
  };
  const DimensionCase cases[] = {{"d1", vol1d(2.0, 8.0), 2},
                                 {"d2", vol2d(2.0, 1.5, 6.0), 0},
                                 {"d3", vol3d(2.0, 1.5, 1.25, 6.0), 1},
                                 {"cyl", volcyl(2.0, 1.5, 6.0), 0}};
  for (const DimensionCase &dimension_case : cases)
    for (int grid_kind = 0; grid_kind < 2; ++grid_kind) {
      structure cpu_structure(dimension_case.volume, isotropic_eps, no_pml(), identity(), 1);
      structure gpu_structure(dimension_case.volume, isotropic_eps, no_pml(), identity(), 1);
      absorber_list absorbers = create_absorber_list();
      if (grid_kind) {
        install_native_grid_material(cpu_structure, absorbers, true, false, false, 0.0, 0.0,
                                     0.0, 0.0, dimension_case.sample_axis);
        install_native_grid_material(gpu_structure, absorbers, true, false, false, 0.0, 0.0,
                                     0.0, 0.0, dimension_case.sample_axis);
      }
      else {
        install_native_file_material(cpu_structure, false, 0.0, dimension_case.sample_axis);
        install_native_file_material(gpu_structure, false, 0.0, dimension_case.sample_axis);
      }
      destroy_absorber_list(absorbers);
      fields cpu(&cpu_structure);
      execution_options options;
      options.backend = backend_kind::nvidia;
      options.precision = precision;
      fields gpu(&gpu_structure, options);
      cpu.use_real_fields();
      gpu.use_real_fields();
      FOR_E_AND_H(c) if (dimension_case.volume.has_field(c)) {
          cpu.require_component(c);
          gpu.require_component(c);
        }
      cpu.init_backend();
      gpu.init_backend();
      NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
      const MaterialIR *ir = material_ir_for(gpu);
      const NvidiaMaterialInitializationStatistics statistics =
          backend ? backend->material_initialization_statistics_for_testing()
                  : NvidiaMaterialInitializationStatistics();
      const size_t launches = grid_kind ? statistics.grid_table_kernel_launches
                                        : statistics.file_table_kernel_launches;
      const size_t points = grid_kind ? statistics.grid_points_evaluated
                                      : statistics.file_points_evaluated;
      const bool varying_table = ir && ir->default_material < ir->materials.size() &&
                                 ir->materials[ir->default_material].samples.size() >= 2 &&
                                 ir->materials[ir->default_material].samples.front() !=
                                     ir->materials[ir->default_material].samples.back();
      require(ir && ir->dimensions == int(dimension_case.volume.dim) && backend &&
                  statistics.valid && statistics.device_native && launches > 0 && points > 1 &&
                  statistics.dense_output_host_to_device_calls == 0 &&
                  varying_table,
              "dimension table fixture did not execute a nonvacuous native table launch");
      compare_all_initialized_material_rows(
          cpu, gpu, sizeof(realnum) == sizeof(float) || precision != precision_policy_kind::native
                        ? 8e-6
                        : 2e-13);
      master_printf("nvidia_timestep: native-%s-%s-dimension/%s PASS\n",
                    grid_kind ? "grid" : "file", dimension_case.name,
                    precision_policy_name(precision));
    }
}

static void test_native_table_periodicity_invariance() {
  using namespace meep_geom;
  for (int table_kind = 0; table_kind < 2; ++table_kind) {
    const grid_volume gv = table_kind ? vol3d(1.0, 1.0, 2.0, 4.0) : vol1d(2.0, 8.0);
    structure nonperiodic(gv, isotropic_eps, no_pml(), identity(), 1);
    structure periodic(gv, isotropic_eps, no_pml(), identity(), 1);
    absorber_list absorbers = create_absorber_list();
    if (table_kind) {
      install_native_grid_material(nonperiodic, absorbers, true, false);
      install_native_grid_material(periodic, absorbers, true, true);
    }
    else {
      install_native_file_material(nonperiodic, false);
      install_native_file_material(periodic, true);
    }
    destroy_absorber_list(absorbers);
    fields cpu(&nonperiodic);
    execution_options options;
    options.backend = backend_kind::nvidia;
    fields gpu(&periodic, options);
    cpu.use_real_fields();
    gpu.use_real_fields();
    FOR_E_AND_H(c) if (gv.has_field(c)) {
        cpu.require_component(c);
        gpu.require_component(c);
      }
    cpu.init_backend();
    gpu.init_backend();
    const MaterialIR *cpu_ir = material_ir_for(cpu);
    const MaterialIR *gpu_ir = material_ir_for(gpu);
    require(cpu_ir && gpu_ir && !cpu_ir->ensure_periodicity && gpu_ir->ensure_periodicity &&
                cpu_ir->signature != gpu_ir->signature,
            "table periodicity fixture did not preserve its semantic flag");
    compare_all_initialized_material_rows(cpu, gpu,
                                          sizeof(realnum) == sizeof(float) ? 8e-6 : 2e-13);
  }
  master_printf("nvidia_timestep: native table periodicity invariance PASS\n");
}

static void test_native_table_retry() {
  using namespace meep_geom;
  const nvidia::testing::failure_point failures[2] = {
      nvidia::testing::failure_point::material_file_launch,
      nvidia::testing::failure_point::material_grid_launch};
  for (int table_kind = 0; table_kind < 2; ++table_kind) {
    const grid_volume gv = vol1d(2.0, 8.0);
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    absorber_list absorbers = create_absorber_list();
    if (table_kind) {
      add_absorbing_layer(absorbers, 0.5, Z, ALL_SIDES, 1e-9, 1.0);
      install_native_grid_material(s, absorbers);
    }
    else install_native_file_material(s);
    destroy_absorber_list(absorbers);
    execution_options options;
    options.backend = backend_kind::nvidia;
    fields gpu(&s, options);
    gpu.use_real_fields();
    gpu.require_component(Ex);
    const nvidia::memory_accounting before = nvidia::current_memory_accounting();
    nvidia::testing::fail_next(failures[table_kind]);
    bool rejected = false;
    try { gpu.init_backend(); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
    const nvidia::memory_accounting after = nvidia::current_memory_accounting();
    require(rejected && !gpu.backend_state && !gpu.executable &&
                before.device_bytes_current == after.device_bytes_current &&
                before.pinned_bytes_current == after.pinned_bytes_current,
            "failed table initialization published or leaked a candidate");
    gpu.init_backend();
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
    require(backend && gpu.backend_state && gpu.executable &&
                backend->material_initialization_statistics_for_testing().valid,
            "table initialization failure was not retryable");
  }
  master_printf("nvidia_timestep: native FILE/MaterialGrid retry PASS\n");
}

enum class NativeTableMutation {
  file_sample,
  grid_weight,
  grid_projection,
  grid_damping,
  grid_endpoint
};

static bool native_table_mutation_is_grid(NativeTableMutation mutation) {
  return mutation != NativeTableMutation::file_sample;
}

static void install_native_table_mutation(structure &s, NativeTableMutation mutation,
                                          bool mutated) {
  using namespace meep_geom;
  if (!native_table_mutation_is_grid(mutation)) {
    install_native_file_material(s, false, mutated ? 0.375 : 0.0);
    return;
  }
  absorber_list absorbers = create_absorber_list();
  install_native_grid_material(
      s, absorbers, true, false, true,
      mutated && mutation == NativeTableMutation::grid_weight ? 0.1875 : 0.0,
      mutated && mutation == NativeTableMutation::grid_projection ? 0.125 : 0.0,
      mutated && mutation == NativeTableMutation::grid_damping ? 0.0625 : 0.0,
      mutated && mutation == NativeTableMutation::grid_endpoint ? 0.3125 : 0.0);
  destroy_absorber_list(absorbers);
}

static void test_native_table_semantic_replacement() {
  const NativeTableMutation mutations[] = {
      NativeTableMutation::file_sample, NativeTableMutation::grid_weight,
      NativeTableMutation::grid_projection, NativeTableMutation::grid_damping,
      NativeTableMutation::grid_endpoint};
  for (NativeTableMutation mutation : mutations) {
    const grid_volume gv = vol1d(2.0, 8.0);
    structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
    install_native_table_mutation(gpu_structure, mutation, false);
    execution_options options;
    options.backend = backend_kind::nvidia;
    fields gpu(&gpu_structure, options);
    gpu.use_real_fields();
    FOR_E_AND_H(c) if (gv.has_field(c)) gpu.require_component(c);
    gpu.init_backend();
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
    require(backend, "semantic table replacement fixture did not select NVIDIA");
    BackendState *const old_state = gpu.backend_state;
    Executable *const old_executable = gpu.executable;
    const std::shared_ptr<const void> old_ir = gpu.material_ir;
    const uint64_t old_signature = material_ir_for(gpu)->signature;
    InitializationPlan *const old_initialization = gpu.initialization_plan;
    const uint64_t old_recipe_signature =
        old_initialization->materials[0].signature();
    const NvidiaMaterialInitializationStatistics old_statistics =
        backend->material_initialization_statistics_for_testing();

    structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
    install_native_table_mutation(cpu_structure, mutation, true);
    fields cpu(&cpu_structure);
    cpu.use_real_fields();
    FOR_E_AND_H(c) if (gv.has_field(c)) cpu.require_component(c);
    cpu.init_backend();
    install_material_oracle_from(cpu, gpu);
    invalidate(gpu, MutationKind::material_values,
               "native FILE/MaterialGrid semantic replacement");
    nvidia::testing::fail_next(native_table_mutation_is_grid(mutation)
                                   ? nvidia::testing::failure_point::material_grid_launch
                                   : nvidia::testing::failure_point::material_file_launch);
    bool rejected = false;
    try { gpu.init_backend(); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
    require(rejected && gpu.backend_state == old_state && gpu.executable == old_executable &&
                gpu.material_ir != old_ir && material_ir_for(gpu)->signature != old_signature &&
                gpu.initialization_plan == old_initialization &&
                gpu.initialization_plan->materials[0].signature() == old_recipe_signature &&
                same_material_initialization_statistics(
                    backend->material_initialization_statistics_for_testing(), old_statistics) &&
                !gpu.backend->is_poisoned(),
            "failed semantic table replacement changed the installed epoch or statistics");

    gpu.init_backend();
    require(gpu.backend_state != old_state && gpu.executable != old_executable &&
                gpu.initialization_plan != old_initialization &&
                gpu.initialization_plan->materials[0].signature() != old_recipe_signature,
            "semantic table replacement retry did not publish the mutated epoch");
    BackendState *const replacement_state = gpu.backend_state;
    Executable *const replacement_executable = gpu.executable;
    const NvidiaMaterialInitializationStatistics replacement_statistics =
        backend->material_initialization_statistics_for_testing();
    nvidia::testing::reset_transfer_accounting();
    nvidia::testing::reset_material_transfer_accounting();
    gpu.init_backend();
    const nvidia::testing::transfer_accounting clean_transfers =
        nvidia::testing::current_transfer_accounting();
    const nvidia::testing::material_transfer_accounting clean_material =
        nvidia::testing::current_material_transfer_accounting();
    require(gpu.backend_state == replacement_state && gpu.executable == replacement_executable &&
                same_material_initialization_statistics(
                    backend->material_initialization_statistics_for_testing(),
                    replacement_statistics) &&
                clean_transfers.host_to_device_calls == 0 &&
                clean_transfers.device_to_host_calls == 0 && clean_material.compact_calls == 0 &&
                clean_material.dense_output_calls == 0,
            "clean semantic table re-entry repeated initialization work");

    compare_all_initialized_material_rows(cpu, gpu,
                                          sizeof(realnum) == sizeof(float) ? 8e-6 : 2e-13);
    initialize_live_fields_by_key(cpu, gpu, sizeof(realnum) == sizeof(float), 0.013);
    int completed = 0;
    const int checkpoints[] = {1, 2, 100};
    for (int checkpoint : checkpoints) {
      cpu.advance(checkpoint - completed);
      gpu.advance(checkpoint - completed);
      completed = checkpoint;
      compare_live_fields_by_key(cpu, gpu,
                                 sizeof(realnum) == sizeof(float) ? 4e-4 : 2e-10);
    }
  }
  master_printf("nvidia_timestep: native table semantic replacement PASS\n");
}

static void test_native_table_mpi() {
  using namespace meep_geom;
  const int chunks = std::max(1, count_processors());
  for (int table_kind = 0; table_kind < 2; ++table_kind) {
    const grid_volume gv = table_kind ? vol3d(1.0, 1.0, 2.0, 4.0) : vol1d(2.0, 8.0);
    structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), chunks);
    structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), chunks);
    absorber_list absorbers = create_absorber_list();
    if (table_kind) {
      install_native_grid_material(cpu_structure, absorbers);
      install_native_grid_material(gpu_structure, absorbers);
    }
    else {
      install_native_file_material(cpu_structure);
      install_native_file_material(gpu_structure);
    }
    destroy_absorber_list(absorbers);
    fields cpu(&cpu_structure);
    execution_options options;
    options.backend = backend_kind::nvidia;
    fields gpu(&gpu_structure, options);
    cpu.use_real_fields();
    gpu.use_real_fields();
    FOR_E_AND_H(c) if (gv.has_field(c)) {
        cpu.require_component(c);
        gpu.require_component(c);
      }
    cpu.init_backend();
    gpu.init_backend();
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
    const NvidiaMaterialInitializationStatistics statistics =
        backend ? backend->material_initialization_statistics_for_testing()
                : NvidiaMaterialInitializationStatistics();
    const bool local_work = table_kind ? statistics.grid_table_kernel_launches > 0
                                       : statistics.file_table_kernel_launches > 0;
    require(backend && statistics.valid && statistics.device_native &&
                or_to_all(local_work) && statistics.dense_output_host_to_device_calls == 0,
            "MPI table initialization did not preserve native split ownership");
    compare_all_initialized_material_rows(cpu, gpu,
                                          sizeof(realnum) == sizeof(float) ? 8e-6 : 2e-13);
    initialize_live_fields_by_key(cpu, gpu, sizeof(realnum) == sizeof(float), 0.017);
    cpu.advance(1);
    gpu.advance(1);
    compare_live_fields_by_key(cpu, gpu,
                               sizeof(realnum) == sizeof(float) ? 4e-4 : 2e-10);
    cpu.advance(1);
    gpu.advance(1);
    compare_live_fields_by_key(cpu, gpu,
                               sizeof(realnum) == sizeof(float) ? 4e-4 : 2e-10);

    structure failure_structure(gv, isotropic_eps, no_pml(), identity(), chunks);
    absorbers = create_absorber_list();
    if (table_kind) install_native_grid_material(failure_structure, absorbers);
    else install_native_file_material(failure_structure);
    destroy_absorber_list(absorbers);
    fields failure(&failure_structure, options);
    failure.use_real_fields();
    failure.require_component(Ex);
    if (my_rank() == 0)
      nvidia::testing::fail_next(table_kind
                                     ? nvidia::testing::failure_point::material_grid_launch
                                     : nvidia::testing::failure_point::material_file_launch);
    bool rejected = false;
    try { failure.init_backend(); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
    require(and_to_all(rejected) && !failure.backend_state && !failure.executable,
            "rank-asymmetric table failure published a partial MPI epoch");
    failure.init_backend();
    require(failure.backend_state && failure.executable,
            "rank-asymmetric table failure was not retryable");
  }
  master_printf("nvidia_timestep: native table MPI/split/asymmetric failure PASS\n");
}

static void test_native_table_preupload_rejection() {
  using namespace meep_geom;
  struct MutationCase {
    nvidia::testing::failure_point point;
    bool grid;
  };
  const MutationCase mutations[] = {
      {nvidia::testing::failure_point::material_table_semantic_mutation, false},
      {nvidia::testing::failure_point::material_table_far_coordinate_mutation, false},
      {nvidia::testing::failure_point::material_table_component_mutation, false},
      {nvidia::testing::failure_point::material_table_tensor_mutation, false},
      {nvidia::testing::failure_point::material_table_loop_mutation, false},
      {nvidia::testing::failure_point::material_table_shift_mutation, false},
      {nvidia::testing::failure_point::material_table_destination_mutation, false},
      {nvidia::testing::failure_point::material_table_source_mutation, true},
      {nvidia::testing::failure_point::material_table_header_mutation, false}};
  for (const MutationCase &mutation : mutations) {
    const grid_volume gv = vol1d(2.0, 8.0);
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    absorber_list absorbers = create_absorber_list();
    if (mutation.grid) install_native_grid_material(s, absorbers);
    else install_native_file_material(s);
    destroy_absorber_list(absorbers);
    execution_options options;
    options.backend = backend_kind::nvidia;
    fields gpu(&s, options);
    gpu.use_real_fields();
    gpu.require_component(Ex);
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
    require(backend, "table descriptor mutation fixture did not select NVIDIA");
    const nvidia::memory_accounting before = nvidia::current_memory_accounting();
    nvidia::testing::reset_transfer_accounting();
    nvidia::testing::reset_material_transfer_accounting();
    nvidia::testing::fail_next(mutation.point);
    bool rejected = false;
    try { gpu.init_backend(); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
    const nvidia::memory_accounting after = nvidia::current_memory_accounting();
    const nvidia::testing::transfer_accounting transfers =
        nvidia::testing::current_transfer_accounting();
    const nvidia::testing::material_transfer_accounting material_transfers =
        nvidia::testing::current_material_transfer_accounting();
    const NvidiaMaterialInitializationStatistics statistics =
        backend->material_initialization_statistics_for_testing();
    require(rejected && !gpu.backend_state && !gpu.executable &&
                before.device_bytes_current == after.device_bytes_current &&
                before.pinned_bytes_current == after.pinned_bytes_current &&
                transfers.host_to_device_calls == 0 && transfers.host_to_device_bytes == 0 &&
                material_transfers.compact_calls == 0 &&
                material_transfers.compact_bytes == 0 &&
                statistics.pointwise_kernel_launches == 0 &&
                statistics.synchronizations == 0 && !statistics.valid,
            "malformed table descriptor crossed the pre-upload transaction boundary");
  }
  master_printf("nvidia_timestep: native table pre-upload rejection PASS\n");
}

static void test_native_table_schema_preflight() {
  using namespace meep_geom;
  for (int malformed_case = 0; malformed_case < 4; ++malformed_case) {
    const grid_volume gv = vol1d(2.0, 8.0);
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    absorber_list absorbers = create_absorber_list();
    if (malformed_case < 2)
      install_native_file_material(s);
    else
      install_native_grid_material(s, absorbers, false);
    destroy_absorber_list(absorbers);
    MaterialIR *ir =
        const_cast<MaterialIR *>(static_cast<const MaterialIR *>(s.material_ir.get()));
    require(ir && ir->default_material < ir->materials.size(),
            "table schema fixture lost its owned IR");
    MaterialIRMaterial &material = ir->materials[ir->default_material];
    if (malformed_case == 0)
      material.parameters[material.parameters.size() - 3] = 0;
    else if (malformed_case == 1)
      material.samples.clear();
    else if (malformed_case == 2)
      material.parameters[0] = 0;
    else
      material.parameters[0] = double(std::numeric_limits<int>::max()) + 1.0;
    refresh_material_ir_signatures_for_testing(*ir);

    execution_options options;
    options.backend = backend_kind::nvidia;
    fields gpu(&s, options);
    gpu.use_real_fields();
    gpu.require_component(Ex);
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
    require(backend, "table schema fixture did not select NVIDIA");
    const nvidia::memory_accounting before = nvidia::current_memory_accounting();
    nvidia::testing::reset_transfer_accounting();
    nvidia::testing::reset_material_transfer_accounting();
    bool rejected = false;
    try { gpu.init_backend(); }
    catch (const std::exception &) { rejected = true; }
    const nvidia::memory_accounting after = nvidia::current_memory_accounting();
    const nvidia::testing::transfer_accounting transfers =
        nvidia::testing::current_transfer_accounting();
    const nvidia::testing::material_transfer_accounting material_transfers =
        nvidia::testing::current_material_transfer_accounting();
    const NvidiaMaterialInitializationStatistics statistics =
        backend->material_initialization_statistics_for_testing();
    require(and_to_all(rejected) && !gpu.backend_state && !gpu.executable &&
                before.device_bytes_current == after.device_bytes_current &&
                before.pinned_bytes_current == after.pinned_bytes_current &&
                transfers.host_to_device_calls == 0 && transfers.host_to_device_bytes == 0 &&
                material_transfers.compact_calls == 0 &&
                material_transfers.compact_bytes == 0 &&
                statistics.pointwise_kernel_launches == 0 &&
                statistics.synchronizations == 0 && !statistics.valid,
            "malformed table schema crossed native preflight");
  }
  master_printf("nvidia_timestep: native table schema preflight PASS\n");
}

static void test_native_grid_damping_only() {
  using namespace meep_geom;
  const grid_volume gv = vol1d(2.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  absorber_list absorbers = create_absorber_list();
  install_native_grid_material(cpu_structure, absorbers, false);
  install_native_grid_material(gpu_structure, absorbers, false);
  destroy_absorber_list(absorbers);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  cpu.require_component(Ex);
  gpu.require_component(Ex);
  cpu.advance(1);
  gpu.advance(1);
  const MaterialIR *ir = material_ir_for(gpu);
  require(ir, "damping-only MaterialGrid lost its owned IR");
  bool topology_conductivity = false, topology_condinv = false;
  for (const MaterialIRTopologyRow &row : ir->topology) {
    const component c = component(row.key.component_);
    if (!is_D(c) || direction(row.key.aux) != component_direction(c)) continue;
    topology_conductivity = topology_conductivity ||
                            row.key.kind == int(array_kind::conductivity);
    topology_condinv = topology_condinv || row.key.kind == int(array_kind::condinv);
  }
  require(ir && topology_conductivity && topology_condinv,
          "damping-only MaterialGrid omitted D conductivity topology");
  compare_all_initialized_material_rows(cpu, gpu,
                                        sizeof(realnum) == sizeof(float) ? 8e-6 : 2e-13);
  initialize_live_fields_by_key(cpu, gpu, false, 0.07);
  cpu.advance(1);
  gpu.advance(1);
  compare_live_fields_by_key(cpu, gpu, sizeof(realnum) == sizeof(float) ? 2e-4 : 5e-12);
  master_printf("nvidia_timestep: native MaterialGrid damping-only PASS\n");
}

static void install_native_chi_only_material(structure &s, bool chi2) {
  using namespace meep_geom;
  material_type material = new material_data();
  material->which_subclass = material_data::MEDIUM;
  material->medium = medium_struct(2.0);
  if (chi2) {
    material->medium.E_chi2_diag = make_vector3(0.17, -0.13, 0.11);
    material->medium.H_chi2_diag = make_vector3(-0.07, 0.09, 0.05);
  }
  else {
    material->medium.E_chi3_diag = make_vector3(0.017, -0.013, 0.011);
    material->medium.H_chi3_diag = make_vector3(-0.007, 0.009, 0.005);
  }
  geometric_object_list empty_geometry = {0, NULL};
  set_materials_from_geometry(&s, empty_geometry, make_vector3(), true, 1e-5, 128, false,
                              material);
  material_free(material);
}

static void test_native_dimension_and_chi_pair(const char *name, const grid_volume &gv,
                                               bool chi2) {
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  install_native_chi_only_material(cpu_structure, chi2);
  install_native_chi_only_material(gpu_structure, chi2);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  FOR_E_AND_H(c) if (gv.has_field(c)) {
      cpu.require_component(c);
      gpu.require_component(c);
    }
  cpu.init_backend();
  gpu.init_backend();
  const MaterialIR *ir = material_ir_for(gpu);
  require(ir && ir->dimensions == int(gv.dim),
          "native dimension fixture captured the wrong dimensional enum");
  size_t chi2_rows = 0, chi3_rows = 0;
  for (const MaterialIRTopologyRow &row : ir->topology) {
    chi2_rows += row.key.kind == int(array_kind::chi2);
    chi3_rows += row.key.kind == int(array_kind::chi3);
  }
  require(chi2_rows > 0 && chi2_rows == chi3_rows,
          "chi-only native fixture did not publish paired chi2/chi3 rows");
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  const NvidiaMaterialInitializationStatistics statistics =
      backend ? backend->material_initialization_statistics_for_testing()
              : NvidiaMaterialInitializationStatistics();
  require(gpu.initialization_plan && gpu.initialization_plan->materials.size() == 1 &&
              gpu.initialization_plan->materials[0].disposition() ==
                  MaterialRecipeDisposition::device_native &&
              backend && statistics.valid && statistics.device_native &&
              statistics.pointwise_kernel_launches > 0,
          "dimension/chi fixture did not execute device-native NVIDIA initialization");
  compare_all_initialized_material_rows(cpu, gpu,
                                        sizeof(realnum) == sizeof(float) ? 8e-6 : 2e-13);
  master_printf("nvidia_timestep: native-%s-%s-only PASS\n", name,
                chi2 ? "chi2" : "chi3");
}

static void test_native_perfect_metal_signed_zero() {
  using namespace meep_geom;
  const grid_volume gv = vol3d(1.0, 1.0, 1.0, 4.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  material_type perfect = new material_data();
  perfect->which_subclass = material_data::PERFECT_METAL;
  geometric_object_list empty_geometry = {0, NULL};
  set_materials_from_geometry(&cpu_structure, empty_geometry, make_vector3(), true, 1e-5, 64,
                              false, perfect);
  set_materials_from_geometry(&gpu_structure, empty_geometry, make_vector3(), true, 1e-5, 64,
                              false, perfect);
  material_free(perfect);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  FOR_E_AND_H(c) if (gv.has_field(c)) {
      cpu.require_component(c);
      gpu.require_component(c);
    }
  cpu.init_backend();
  gpu.init_backend();
  const MaterialIR *ir = material_ir_for(gpu);
  bool saw_electric = false, saw_magnetic = false;
  require(ir && ir->materials[ir->default_material].kind == material_data::PERFECT_METAL,
          "perfect-metal native fixture lost its owned material tag");
  for (const MaterialIRTopologyRow &row : ir->topology) {
    if (row.key.kind != int(array_kind::chi1inv) || row.key.component_ < 0 ||
        component_direction(component(row.key.component_)) != direction(row.key.aux))
      continue;
    const bool electric = is_electric(component(row.key.component_));
    const bool magnetic = is_magnetic(component(row.key.component_));
    if (magnetic) {
      const std::complex<double> value = gpu_structure.get_chi1inv(
          component(row.key.component_), direction(row.key.aux), gv.center(), 0, false);
      require(value == std::complex<double>(1.0, 0.0),
              "perfect-metal magnetic diagonal is not exact positive one");
      saw_magnetic = true;
    }
    const ArrayId id = gpu.array_catalog->find(row.key);
    if (!is_valid(id)) continue;
    std::vector<realnum> observed(row.elements);
    gpu.backend->read(ArrayRef{id, 0, row.elements}, observed.data(),
                      observed.size() * sizeof(realnum));
    for (realnum value : observed) {
      if (electric)
        require(value == realnum(0) && std::signbit(value),
                "perfect-metal electric diagonal is not exact negative zero");
      if (magnetic)
        require(value == realnum(1),
                "retained perfect-metal magnetic diagonal is not exact positive one");
    }
    saw_electric = saw_electric || electric;
    saw_magnetic = saw_magnetic || magnetic;
  }
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(saw_electric && saw_magnetic && backend &&
              backend->material_initialization_statistics_for_testing()
                      .constant_fill_kernel_launches > 0,
          "perfect-metal diagonal topology or native execution is absent");
  master_printf("nvidia_timestep: native-perfect-metal signed-zero PASS\n");
}

static void test_native_dimension_pml(const char *name, const grid_volume &gv) {
  const bool cylindrical = gv.dim == Dcyl;
  const boundary_region boundaries =
      cylindrical ? pml(0.35, R, High) + pml(0.35, Z, Low)
                  : pml(0.35, X, Low) + pml(0.35, Y, High);
  structure cpu_structure(gv, isotropic_eps, boundaries, identity(), 1);
  structure gpu_structure(gv, isotropic_eps, boundaries, identity(), 1);
  install_native_chi_only_material(cpu_structure, true);
  install_native_chi_only_material(gpu_structure, true);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  FOR_E_AND_H(c) if (gv.has_field(c)) {
      cpu.require_component(c);
      gpu.require_component(c);
    }
  cpu.init_backend();
  gpu.init_backend();
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  const MaterialIR *ir = material_ir_for(gpu);
  require(ir && !ir->pml_axes.empty() && backend && gpu.initialization_plan &&
              gpu.initialization_plan->materials.size() == 1 &&
              gpu.initialization_plan->materials[0].disposition() ==
                  MaterialRecipeDisposition::device_native,
          "dimension/PML fixture did not select device-native NVIDIA initialization");
  bool first_direction = false, second_direction = false, nonzero_profile = false;
  for (const MaterialIRPmlAxis &axis : ir->pml_axes) {
    first_direction = first_direction ||
                      axis.direction == int(cylindrical ? R : X);
    second_direction = second_direction ||
                       axis.direction == int(cylindrical ? Z : Y);
    for (double sample : axis.profile_samples)
      nonzero_profile = nonzero_profile || sample != 0.0;
  }
  const NvidiaMaterialInitializationStatistics statistics =
      backend->material_initialization_statistics_for_testing();
  require(first_direction && second_direction && nonzero_profile && statistics.valid &&
              statistics.device_native && statistics.pml_kernel_launches == ir->pml_axes.size() &&
              statistics.pml_kernel_launches > 0,
          "dimension/PML fixture did not execute nonvacuous native PML launches");
  compare_all_initialized_material_rows(cpu, gpu,
                                        sizeof(realnum) == sizeof(float) ? 8e-6 : 2e-13);
  master_printf("nvidia_timestep: native-%s-pml PASS\n", name);
}

static void test_native_table_dimension_pml(const char *name, const grid_volume &gv) {
  using namespace meep_geom;
  const bool cylindrical = gv.dim == Dcyl;
  const boundary_region boundaries =
      cylindrical ? pml(0.35, R, High) + pml(0.35, Z, Low)
                  : pml(0.35, X, Low) + pml(0.35, Y, High);
  structure cpu_structure(gv, isotropic_eps, boundaries, identity(), 1);
  structure gpu_structure(gv, isotropic_eps, boundaries, identity(), 1);
  absorber_list absorbers = create_absorber_list();
  add_absorbing_layer(absorbers, 0.35, cylindrical ? R : X,
                      cylindrical ? High : ALL_SIDES, 1e-9, 1.0);
  install_native_grid_material(cpu_structure, absorbers, true, false, false);
  install_native_grid_material(gpu_structure, absorbers, true, false, false);
  destroy_absorber_list(absorbers);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  FOR_E_AND_H(c) if (gv.has_field(c)) {
      cpu.require_component(c);
      gpu.require_component(c);
    }
  cpu.init_backend();
  gpu.init_backend();
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  const MaterialIR *ir = material_ir_for(gpu);
  const NvidiaMaterialInitializationStatistics statistics =
      backend ? backend->material_initialization_statistics_for_testing()
              : NvidiaMaterialInitializationStatistics();
  require(ir && ir->dimensions == int(gv.dim) && backend && statistics.valid &&
              statistics.device_native && statistics.grid_table_kernel_launches > 0 &&
              statistics.pml_kernel_launches > 0 &&
              statistics.absorber_points_evaluated > 0 &&
              statistics.dense_output_host_to_device_calls == 0,
          "table dimension/PML fixture did not execute native table/PML initialization");
  compare_all_initialized_material_rows(cpu, gpu,
                                        sizeof(realnum) == sizeof(float) ? 8e-6 : 2e-13);
  master_printf("nvidia_timestep: native-table-%s-pml PASS\n", name);
}

static void test_native_pml_rounding_association() {
  const double resolution = 1.3;
  const double thickness = 0.19230769230769229;
  require(int(thickness * (2 * resolution) + 0.5) == 1,
          "PML rounding fixture no longer exercises the requested half-cell case");
  const grid_volume gv = vol1d(2.0, resolution);
  const boundary_region boundaries = pml(thickness, Z, High);
  structure cpu_structure(gv, isotropic_eps, boundaries, identity(), 1);
  structure gpu_structure(gv, isotropic_eps, boundaries, identity(), 1);
  install_native_homogeneous_material(cpu_structure);
  install_native_homogeneous_material(gpu_structure);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  FOR_E_AND_H(c) if (gv.has_field(c)) {
      cpu.require_component(c);
      gpu.require_component(c);
    }
  cpu.init_backend();
  gpu.init_backend();
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  const MaterialIR *ir = material_ir_for(gpu);
  const NvidiaMaterialInitializationStatistics statistics =
      backend ? backend->material_initialization_statistics_for_testing()
              : NvidiaMaterialInitializationStatistics();
  require(ir && !ir->pml_axes.empty() && backend && statistics.valid &&
              statistics.device_native && statistics.pml_kernel_launches == ir->pml_axes.size() &&
              statistics.pml_kernel_launches > 0,
          "half-cell PML rounding fixture did not execute native PML initialization");
  compare_all_initialized_material_rows(cpu, gpu,
                                        sizeof(realnum) == sizeof(float) ? 8e-6 : 2e-13);
  master_printf("nvidia_timestep: native-pml-rounding-association PASS\n");
}

static void test_native_material_initialization(precision_policy_kind precision,
                                                bool real_fields) {
  const grid_volume gv = vol3d(1.25, 1.0, 0.75, 6.0);
  structure cpu_structure(gv, isotropic_eps, pml(0.2), identity(), 2);
  structure gpu_structure(gv, isotropic_eps, pml(0.2), identity(), 2);
  install_native_homogeneous_material(cpu_structure);
  install_native_homogeneous_material(gpu_structure);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&gpu_structure, options);
  if (real_fields) {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  else {
    const vec bloch(0.09, -0.07, 0.05);
    cpu.use_bloch(bloch);
    gpu.use_bloch(bloch);
  }
  for (component c : {Ex, Ey, Ez, Hx, Hy, Hz}) {
    cpu.require_component(c);
    gpu.require_component(c);
  }
  cpu.init_backend();
  nvidia::testing::reset_transfer_accounting();
  nvidia::testing::reset_material_transfer_accounting();
  gpu.init_backend();
  require(gpu.initialization_plan && gpu.initialization_plan->materials.size() == 1 &&
              gpu.initialization_plan->materials[0].disposition() ==
                  MaterialRecipeDisposition::device_native,
          "homogeneous owned IR did not take the device-native route");
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend != NULL, "native material fixture did not select the NVIDIA backend");
  const NvidiaMaterialInitializationStatistics statistics =
      backend->material_initialization_statistics_for_testing();
  const nvidia::testing::transfer_accounting initialization_transfers =
      nvidia::testing::current_transfer_accounting();
  const nvidia::testing::material_transfer_accounting material_transfers =
      nvidia::testing::current_material_transfer_accounting();
  const MaterialIR *ir = material_ir_for(gpu);
  require(ir != NULL, "native material fixture lost its owned IR");
  std::vector<double> electric_state_frequencies;
  std::set<uint32_t> electric_state_ids;
  for (const MaterialIRSusceptibility &sus : ir->susceptibilities)
    if (sus.field_type == E_stuff) {
      electric_state_ids.insert(sus.identity);
      require(sus.parameters.size() > 9,
              "native electric susceptibility payload is incomplete");
      electric_state_frequencies.push_back(sus.parameters[9]);
    }
  require(electric_state_frequencies.size() == 2 && electric_state_ids.size() == 2 &&
              *electric_state_ids.begin() == 0 && *electric_state_ids.rbegin() == 1 &&
              electric_state_frequencies[0] == 0.47 && electric_state_frequencies[1] == 0.73,
          "native susceptibility state identities differ from CPU prepend ordering");
  size_t expected_rows = 0, expected_pml = ir->pml_axes.size(), expected_compact = 0;
  size_t logical_status_bytes = 0, physical_material_elements = 0;
  for (const MaterialIRTopologyRow &row : ir->topology) {
    const array_kind kind = array_kind(row.key.kind);
    physical_material_elements += row.elements;
    size_t logical = 0;
    if (kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
        kind == array_kind::pml_siginv) {
      for (const MaterialIRPmlAxis &axis : ir->pml_axes)
        if (axis.chunk == row.key.chunk && uint64_t(axis.direction) == row.key.aux)
          logical = axis.elements;
    }
    else
      for (const MaterialIRDestination &destination : ir->destinations)
        if (destination.key == row.key) logical = size_t(destination.point_count);
    require(logical && logical <= row.elements,
            "native material fixture has an invalid logical classification span");
    logical_status_bytes += logical;
    if (kind != array_kind::pml_sig && kind != array_kind::pml_kap &&
        kind != array_kind::pml_siginv)
      ++expected_rows;
  }
  for (const MaterialIRPmlAxis &axis : ir->pml_axes) {
    const size_t samples = axis.profile_active ? axis.profile_samples.size() : 0;
    require(samples <= std::numeric_limits<size_t>::max() / sizeof(double) &&
                samples * sizeof(double) <=
                    std::numeric_limits<size_t>::max() - expected_compact,
            "native material compact-input expectation overflowed");
    expected_compact += samples * sizeof(double);
  }
  const size_t candidate_attempts = 1 + gpu.classification_reentries;
  require(statistics.valid && statistics.device_native &&
              statistics.owned_ir_bytes >= statistics.compact_input_host_to_device_bytes &&
              statistics.dense_oracle_bytes == 0 &&
              statistics.dense_output_host_to_device_calls == 0 &&
              statistics.dense_output_host_to_device_bytes == 0 &&
              statistics.constant_fill_kernel_launches == 2 * expected_rows &&
              statistics.conductivity_kernel_launches == 0 &&
              statistics.file_table_kernel_launches == 0 &&
              statistics.grid_table_kernel_launches == 0 &&
              statistics.pointwise_kernel_launches == 2 * expected_rows &&
              statistics.pml_kernel_launches == expected_pml &&
              statistics.compact_input_host_to_device_calls == (expected_compact ? 1 : 0) &&
              statistics.compact_input_host_to_device_bytes == expected_compact &&
              statistics.decoded_parameter_bytes == 0 &&
              statistics.absorber_profile_bytes == 0 &&
              statistics.pml_profile_bytes == expected_compact &&
              statistics.file_sample_bytes == 0 && statistics.grid_weight_bytes == 0 &&
              statistics.absorber_points_evaluated == 0 &&
              statistics.file_points_evaluated == 0 &&
              statistics.grid_points_evaluated == 0 &&
              statistics.classification_status_bytes == logical_status_bytes &&
              logical_status_bytes <= physical_material_elements &&
              statistics.synchronizations == 1,
          "native material initialization accounting is not exact");
  require(material_transfers.dense_output_calls == 0 &&
              material_transfers.dense_output_bytes == 0 &&
              material_transfers.compact_calls ==
                  candidate_attempts * size_t(expected_compact != 0) &&
              material_transfers.compact_bytes == candidate_attempts * expected_compact &&
              initialization_transfers.host_to_device_calls >= material_transfers.compact_calls &&
              initialization_transfers.host_to_device_bytes >= material_transfers.compact_bytes,
          "global/tagged transfer accounting found a dense native material upload");
  const double tolerance =
      (sizeof(realnum) == sizeof(float) || precision != precision_policy_kind::native) ? 8e-6
                                                                                       : 2e-13;
  compare_all_initialized_material_rows(cpu, gpu, tolerance);
  initialize_live_fields_by_key(cpu, gpu, precision != precision_policy_kind::native, 0.19);
  nvidia::testing::reset_transfer_accounting();
  cpu.advance(2);
  gpu.advance(2);
  const nvidia::testing::transfer_accounting steady =
      nvidia::testing::current_transfer_accounting();
  require(steady.host_to_device_calls == 0 && steady.host_to_device_bytes == 0 &&
              steady.device_to_host_calls == 0 && steady.device_to_host_bytes == 0,
          "steady timestep performed material or field transfers");
  /* Native material classification appends and may tombstone provisional
     rows independently of the CPU catalog, so physical ArrayId numbering is
     not a cross-backend identity. */
  compare_live_fields_by_key(cpu, gpu, tolerance * 20);
  const NvidiaMaterialInitializationStatistics after =
      backend->material_initialization_statistics_for_testing();
  require(after.pointwise_kernel_launches == statistics.pointwise_kernel_launches &&
              after.pml_kernel_launches == statistics.pml_kernel_launches &&
              after.compact_input_host_to_device_bytes ==
                  statistics.compact_input_host_to_device_bytes,
          "steady stepping repeated native material initialization work");
  master_printf("nvidia_timestep: native-material-%s/%s PASS\n",
                real_fields ? "real" : "complex", precision_policy_name(precision));
}

static void pr54_borrowed_material(vector3 point, void *data, meep_geom::medium_struct *medium) {
  const double base = *static_cast<const double *>(data);
  *medium = meep_geom::medium_struct(base + 0.03125 * point.x);
  medium->mu_diag = meep_geom::make_vector3(1.5 + 0.015625 * point.y,
                                            1.625 - 0.0078125 * point.x, 1.75);
}

static double pr54_legacy_epsilon(const vec &) { return 2.375; }

static bool retained_material_key(const StoragePlan &plan, const StorageKey &key) {
  for (size_t i = 0; i < plan.keys.size(); ++i)
    if (plan.keys[i] == key)
      return plan.arrays[i].role == array_role::material &&
             !plan.arrays[i].classification_elided;
  return false;
}

static bool same_material_fallback_statistics(const MaterialFallbackStatistics &a,
                                              const MaterialFallbackStatistics &b) {
  return a.warnings == b.warnings && a.dense_rows == b.dense_rows &&
         a.dense_bytes == b.dense_bytes && a.interface_points == b.interface_points &&
         a.callback_tiles == b.callback_tiles && a.callback_points == b.callback_points &&
         a.callback_calls == b.callback_calls &&
         a.classification_launches == b.classification_launches &&
         a.classification_device_to_host_calls == b.classification_device_to_host_calls &&
         a.classification_device_to_host_bytes == b.classification_device_to_host_bytes;
}

static void check_pr54_route_timesteps(fields &cpu, fields &gpu) {
  NvidiaBackend *const backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend && gpu.backend_state, "PR5.4 route has no committed NVIDIA epoch");
  const NvidiaMaterialInitializationStatistics initialization_before =
      backend->material_initialization_statistics_for_testing();
  const MaterialFallbackStatistics fallback_before =
      gpu.backend_state->material_fallback_statistics;
  nvidia::testing::reset_material_transfer_accounting();
  backend_reset_material_phase_preflight_counts_for_testing();
  initialize_live_fields_by_key(cpu, gpu, true, 0.071);
  int completed = 0;
  const int checkpoints[] = {1, 2, 100};
  for (int checkpoint : checkpoints) {
    cpu.advance(checkpoint - completed);
    gpu.advance(checkpoint - completed);
    completed = checkpoint;
    compare_live_fields_by_key(cpu, gpu, 4e-4);
  }
  const NvidiaMaterialInitializationStatistics initialization_after =
      backend->material_initialization_statistics_for_testing();
  const nvidia::testing::material_transfer_accounting material_transfers =
      nvidia::testing::current_material_transfer_accounting();
  require(same_material_initialization_statistics(initialization_before,
                                                  initialization_after) &&
              same_material_fallback_statistics(
                  fallback_before, gpu.backend_state->material_fallback_statistics) &&
              material_transfers.compact_calls == 0 && material_transfers.compact_bytes == 0 &&
              material_transfers.dense_output_calls == 0 &&
              material_transfers.dense_output_bytes == 0 &&
              material_transfers.tiled_output_calls == 0 &&
              material_transfers.tiled_output_bytes == 0 &&
              backend_material_phase_collective_count_for_testing() == 0 &&
              backend_material_phase_scan_count_for_testing() == 0,
          "100 clean route steps repeated material scans, collectives, transfers, kernels, "
          "callbacks, synchronization, or capacity work");
}

static void test_pr54_classification_groups() {
  using namespace meep_geom;
  const grid_volume gv = vol3d(1.0, 1.0, 1.0, 5.0);
  geometric_object_list geometry = {0, NULL};
  execution_options options;
  options.backend = backend_kind::nvidia;

  material_type sigma_material = new material_data();
  sigma_material->which_subclass = material_data::MEDIUM;
  sigma_material->medium = medium_struct(2.0);
  meep_geom::susceptibility sigma = meep_geom::susceptibility();
  sigma.frequency = 0.4;
  sigma.gamma = 0.03;
  sigma.sigma_diag = make_vector3(0.0, 0.0, 0.0);
  sigma.sigma_offdiag = make_vector3(0.2, -0.15, 0.1);
  sigma_material->medium.E_susceptibilities.push_back(sigma);
  structure sigma_cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure sigma_gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  set_materials_from_geometry(&sigma_cpu_structure, geometry, make_vector3(), true, 1e-5, 64,
                              false, sigma_material);
  set_materials_from_geometry(&sigma_gpu_structure, geometry, make_vector3(), true, 1e-5, 64,
                              false, sigma_material);
  material_free(sigma_material);
  fields sigma_cpu(&sigma_cpu_structure);
  fields sigma_gpu(&sigma_gpu_structure, options);
  sigma_cpu.use_real_fields();
  sigma_gpu.use_real_fields();
  for (component c : {Ex, Ey, Ez}) {
    sigma_cpu.require_component(c);
    sigma_gpu.require_component(c);
  }
  sigma_cpu.init_backend();
  sigma_gpu.init_backend();
  require(sigma_gpu.storage_plan, "sigma classification fixture has no storage plan");
  bool saw_offdiagonal_sigma = false;
  for (size_t i = 0; i < sigma_gpu.storage_plan->keys.size(); ++i) {
    const StorageKey &key = sigma_gpu.storage_plan->keys[i];
    if (key.kind != int(array_kind::sigma) ||
        sigma_gpu.storage_plan->arrays[i].classification_elided || key.component_ < 0)
      continue;
    const direction diagonal = component_direction(component(key.component_));
    if (key.cmp == int(diagonal)) continue;
    saw_offdiagonal_sigma = true;
    require(retained_material_key(
                *sigma_gpu.storage_plan,
                StorageKey{key.chunk, int(array_kind::sigma), key.component_, int(diagonal),
                           key.aux}),
            "retained sigma offdiagonal lost its same-state diagonal sigma");
  }
  require(saw_offdiagonal_sigma,
          "zero-diagonal sigma fixture retained no offdiagonal susceptibility row");
  compare_all_initialized_material_rows(sigma_cpu, sigma_gpu,
                                        sizeof(realnum) == sizeof(float) ? 8e-6 : 2e-13);

  material_type nonlinear_material = new material_data();
  nonlinear_material->which_subclass = material_data::MEDIUM;
  nonlinear_material->medium = medium_struct(1.0);
  nonlinear_material->medium.E_chi2_diag = make_vector3(0.2, 0.0, 0.0);
  structure nonlinear_cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure nonlinear_gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  set_materials_from_geometry(&nonlinear_cpu_structure, geometry, make_vector3(), true, 1e-5,
                              64, false, nonlinear_material);
  set_materials_from_geometry(&nonlinear_gpu_structure, geometry, make_vector3(), true, 1e-5,
                              64, false, nonlinear_material);
  material_free(nonlinear_material);
  fields nonlinear_cpu(&nonlinear_cpu_structure);
  fields nonlinear_gpu(&nonlinear_gpu_structure, options);
  nonlinear_cpu.use_real_fields();
  nonlinear_gpu.use_real_fields();
  nonlinear_cpu.require_component(Ex);
  nonlinear_gpu.require_component(Ex);
  nonlinear_cpu.init_backend();
  nonlinear_gpu.init_backend();
  require(nonlinear_gpu.storage_plan,
          "nonlinear classification fixture has no storage plan");
  bool saw_nonlinear = false;
  for (size_t i = 0; i < nonlinear_gpu.storage_plan->keys.size(); ++i) {
    const StorageKey &key = nonlinear_gpu.storage_plan->keys[i];
    if ((key.kind != int(array_kind::chi2) && key.kind != int(array_kind::chi3)) ||
        nonlinear_gpu.storage_plan->arrays[i].classification_elided)
      continue;
    saw_nonlinear = true;
    const direction diagonal = component_direction(component(key.component_));
    require(retained_material_key(
                *nonlinear_gpu.storage_plan,
                StorageKey{key.chunk, int(array_kind::chi2), key.component_, -1, 0}) &&
                retained_material_key(
                    *nonlinear_gpu.storage_plan,
                    StorageKey{key.chunk, int(array_kind::chi3), key.component_, -1, 0}) &&
                retained_material_key(
                    *nonlinear_gpu.storage_plan,
                    StorageKey{key.chunk, int(array_kind::chi1inv), key.component_, -1,
                               uint64_t(diagonal)}),
            "nonlinear row lost its chi2/chi3/diagonal-chi1 classification group");
  }
  require(saw_nonlinear, "epsilon=1 nonlinear fixture retained no nonlinear row");
  compare_all_initialized_material_rows(nonlinear_cpu, nonlinear_gpu,
                                        sizeof(realnum) == sizeof(float) ? 8e-6 : 2e-13);
  master_printf("nvidia_timestep: PR5.4 classification groups PASS\n");
}

static void test_pr54_callback_routes() {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.75, 1.25, 16.0);
  geometric_object_list geometry = {0, NULL};
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::mixed;
  options.strict = false;
  options.fallback = fallback_policy::warn;

  std::shared_ptr<const OwnedMaterialCallback> owner(new OwnedMaterialCallback(
      UINT64_C(0x707235342d74696c), UINT64_C(0x707235342d736967),
      owned_material_callback_tiled_capabilities,
      [](vector3 point, medium_struct &medium) {
        medium = medium_struct(2.25 + 0.0625 * point.x - 0.03125 * point.y);
      }));
  material_type tiled_material = make_owned_user_material_for_backend(owner, false);
  structure tiled_cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure tiled_gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  set_materials_from_geometry(&tiled_cpu_structure, geometry, make_vector3(), false, 1e-5, 64,
                              false, tiled_material);
  set_materials_from_geometry(&tiled_gpu_structure, geometry, make_vector3(), false, 1e-5, 64,
                              false, tiled_material);
  material_free(tiled_material);
  fields tiled_cpu(&tiled_cpu_structure);
  fields tiled_gpu(&tiled_gpu_structure, options);
  tiled_cpu.use_real_fields();
  tiled_gpu.use_real_fields();
  tiled_cpu.require_component(Ez);
  tiled_gpu.require_component(Ez);
  tiled_cpu.init_backend();
  nvidia::testing::reset_transfer_accounting();
  nvidia::testing::reset_material_transfer_accounting();
  nvidia::testing::fail_next(
      nvidia::testing::failure_point::material_callback_prepare);
  bool callback_prepare_rejected = false;
  try { tiled_gpu.init_backend(); }
  catch (const std::runtime_error &error) {
    callback_prepare_rejected =
        std::string(error.what()).find("callback preparation failure") != std::string::npos;
  }
  const nvidia::testing::transfer_accounting failed_callback_transfers =
      nvidia::testing::current_transfer_accounting();
  require(callback_prepare_rejected && !tiled_gpu.backend_state && !tiled_gpu.executable &&
              failed_callback_transfers.host_to_device_calls == 0 &&
              failed_callback_transfers.device_to_host_calls == 0,
          "owned callback prepare failure crossed the CUDA or publication boundary");
  nvidia::testing::reset_material_transfer_accounting();
  tiled_gpu.init_backend();
  require(tiled_gpu.initialization_plan &&
              tiled_gpu.initialization_plan->materials.size() == 1 &&
              tiled_gpu.initialization_plan->materials[0].disposition() ==
                  MaterialRecipeDisposition::tiled_callback,
          "owned callback did not select the NVIDIA tiled route");
  compare_all_initialized_material_rows(tiled_cpu, tiled_gpu, 8e-6);
  NvidiaBackend *const tiled_backend = dynamic_cast<NvidiaBackend *>(tiled_gpu.backend);
  const NvidiaMaterialInitializationStatistics tiled_statistics =
      tiled_backend ? tiled_backend->material_initialization_statistics_for_testing()
                    : NvidiaMaterialInitializationStatistics();
  const nvidia::testing::material_transfer_accounting tiled_transfers =
      nvidia::testing::current_material_transfer_accounting();
  bool short_tile = false;
  uint64_t tiled_points = 0;
  for (const MaterialCallbackTile &tile :
       tiled_gpu.initialization_plan->materials[0].callback_tiles()) {
    require(tile.count && tile.count <= 256,
            "owned callback emitted a tile outside the bounded contract");
    short_tile = short_tile || tile.count < 256;
    tiled_points += tile.count;
  }
  require(tiled_gpu.backend_state && tiled_backend && tiled_statistics.valid &&
              tiled_gpu.backend_state->material_fallback_statistics.callback_calls ==
                  tiled_gpu.backend_state->material_fallback_statistics.callback_points &&
              tiled_points ==
                  tiled_gpu.backend_state->material_fallback_statistics.callback_points &&
              tiled_gpu.backend_state->material_fallback_statistics.callback_tiles > 1 &&
              short_tile && tiled_statistics.callback_scratch_bytes == 256 * sizeof(realnum) &&
              tiled_statistics.logical_output_bytes > tiled_statistics.callback_scratch_bytes &&
              tiled_statistics.dense_output_host_to_device_calls == 0 &&
              tiled_statistics.dense_output_host_to_device_bytes == 0 &&
              tiled_statistics.tiled_output_host_to_device_calls > 0 &&
              tiled_statistics.tiled_output_host_to_device_bytes > 0 &&
              tiled_statistics.tiled_output_host_to_device_calls ==
                  tiled_transfers.tiled_output_calls &&
              tiled_statistics.tiled_output_host_to_device_bytes ==
                  tiled_transfers.tiled_output_bytes &&
              tiled_transfers.dense_output_calls == 0 &&
              tiled_transfers.dense_output_bytes == 0 &&
              tiled_statistics.classification_status_bytes > 0 &&
              tiled_statistics.classification_result_bytes > 0 &&
              tiled_statistics.synchronizations == 1,
          "owned tiled callback execution/staging/accounting is not exact");
  check_pr54_route_timesteps(tiled_cpu, tiled_gpu);

  double borrowed_base = 2.5;
  material_type host_material =
      make_user_material(pr54_borrowed_material, &borrowed_base, false);
  structure host_cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure host_gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  set_materials_from_geometry(&host_cpu_structure, geometry, make_vector3(), false, 1e-5, 64,
                              false, host_material);
  set_materials_from_geometry(&host_gpu_structure, geometry, make_vector3(), false, 1e-5, 64,
                              false, host_material);
  material_free(host_material);
  fields host_cpu(&host_cpu_structure);
  fields host_gpu(&host_gpu_structure, options);
  host_cpu.use_real_fields();
  host_gpu.use_real_fields();
  host_cpu.require_component(Ez);
  host_gpu.require_component(Ez);
  host_cpu.init_backend();
  nvidia::testing::reset_material_transfer_accounting();
  host_gpu.init_backend();
  require(host_gpu.initialization_plan && host_gpu.initialization_plan->materials.size() == 1 &&
              host_gpu.initialization_plan->materials[0].disposition() ==
                  MaterialRecipeDisposition::host_reference,
          "borrowed callback did not select the NVIDIA host-reference route");
  compare_all_initialized_material_rows(host_cpu, host_gpu, 8e-6);
  NvidiaBackend *const host_backend = dynamic_cast<NvidiaBackend *>(host_gpu.backend);
  const NvidiaMaterialInitializationStatistics host_statistics =
      host_backend ? host_backend->material_initialization_statistics_for_testing()
                   : NvidiaMaterialInitializationStatistics();
  const nvidia::testing::material_transfer_accounting host_transfers =
      nvidia::testing::current_material_transfer_accounting();
  require(host_gpu.backend_state && host_backend && host_statistics.valid &&
              host_gpu.backend_state->material_fallback_statistics.callback_calls == 0 &&
              host_gpu.backend_state->material_fallback_statistics.dense_rows > 0 &&
              host_statistics.callback_scratch_bytes == 0 &&
              host_statistics.dense_output_host_to_device_calls > 0 &&
              host_statistics.dense_output_host_to_device_bytes > 0 &&
              host_statistics.tiled_output_host_to_device_calls == 0 &&
              host_statistics.tiled_output_host_to_device_bytes == 0 &&
              host_statistics.dense_output_host_to_device_calls ==
                  host_transfers.dense_output_calls &&
              host_statistics.dense_output_host_to_device_bytes ==
                  host_transfers.dense_output_bytes &&
              host_transfers.tiled_output_calls == 0 &&
              host_transfers.tiled_output_bytes == 0 &&
              host_statistics.synchronizations == 1,
          "host-reference route re-evaluated a callback or lost dense accounting");
  check_pr54_route_timesteps(host_cpu, host_gpu);

  fields *recursive_fields = NULL;
  std::shared_ptr<const OwnedMaterialCallback> recursive_owner(new OwnedMaterialCallback(
      UINT64_C(0x707235342d726563), UINT64_C(0x707235342d726573),
      owned_material_callback_tiled_capabilities,
      [&recursive_fields](vector3, medium_struct &medium) {
        if (recursive_fields) recursive_fields->init_backend();
        medium = medium_struct(2.0);
      }));
  material_type recursive_material =
      make_owned_user_material_for_backend(recursive_owner, false);
  structure recursive_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  set_materials_from_geometry(&recursive_structure, geometry, make_vector3(), false, 1e-5, 64,
                              false, recursive_material);
  material_free(recursive_material);
  fields recursive_gpu(&recursive_structure, options);
  recursive_fields = &recursive_gpu;
  recursive_gpu.use_real_fields();
  recursive_gpu.require_component(Ez);
  nvidia::testing::reset_transfer_accounting();
  bool recursion_rejected = false;
  try { recursive_gpu.init_backend(); }
  catch (const std::runtime_error &error) {
    recursion_rejected =
        std::string(error.what()).find("recursive owned tiled material callback") !=
        std::string::npos;
  }
  const nvidia::testing::transfer_accounting recursive_transfers =
      nvidia::testing::current_transfer_accounting();
  require(recursion_rejected && !recursive_gpu.backend_state && !recursive_gpu.executable &&
              recursive_transfers.host_to_device_calls == 0 &&
              recursive_transfers.device_to_host_calls == 0,
          "recursive callback crossed the CUDA or publication boundary");

  simple_material_function legacy_epsilon(pr54_legacy_epsilon);
  structure legacy_cpu_structure(gv, legacy_epsilon, no_pml(), identity(), 1);
  structure legacy_gpu_structure(gv, legacy_epsilon, no_pml(), identity(), 1);
  fields legacy_cpu(&legacy_cpu_structure);
  fields legacy_gpu(&legacy_gpu_structure, options);
  legacy_cpu.use_real_fields();
  legacy_gpu.use_real_fields();
  legacy_cpu.require_component(Ez);
  legacy_gpu.require_component(Ez);
  legacy_cpu.init_backend();
  legacy_gpu.init_backend();
  require(!material_ir_for(legacy_gpu) && legacy_gpu.initialization_plan &&
              legacy_gpu.initialization_plan->materials.size() == 1 &&
              legacy_gpu.initialization_plan->materials[0].disposition() ==
                  MaterialRecipeDisposition::host_reference,
          "legacy no-IR material did not select host-reference");
  compare_all_initialized_material_rows(legacy_cpu, legacy_gpu, 8e-6);
  check_pr54_route_timesteps(legacy_cpu, legacy_gpu);
  master_printf("nvidia_timestep: PR5.4 tiled/host routes PASS\n");
}

static void install_pr54_tiled_material(structure &s, size_t &callback_calls);

static void run_pr54_route_policy_case(precision_policy_kind policy, bool magnetic,
                                       bool complex_fields, bool tiled) {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.25, 1.0, 10.0);
  geometric_object_list geometry = {0, NULL};
  size_t callback_calls = 0;
  material_type material = NULL;
  double borrowed_base = 2.375;
  if (tiled) {
    std::shared_ptr<const OwnedMaterialCallback> owner(new OwnedMaterialCallback(
        UINT64_C(0x707235342d6d6174) + uint64_t(magnetic),
        UINT64_C(0x707235342d706f6c) + uint64_t(complex_fields),
        owned_material_callback_tiled_capabilities,
        [&callback_calls](vector3 point, medium_struct &medium) {
          ++callback_calls;
          medium = medium_struct(2.25 + 0.03125 * point.x);
          medium.mu_diag = make_vector3(1.5 + 0.015625 * point.y,
                                        1.625 - 0.0078125 * point.x, 1.75);
        }));
    material = make_owned_user_material_for_backend(owner, false);
  }
  else
    material = make_user_material(pr54_borrowed_material, &borrowed_base, false);

  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  set_materials_from_geometry(&cpu_structure, geometry, make_vector3(), false, 1e-5, 64,
                              false, material);
  set_materials_from_geometry(&gpu_structure, geometry, make_vector3(), false, 1e-5, 64,
                              false, material);
  material_free(material);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&gpu_structure, options);
  if (complex_fields) {
    const vec bloch(0.11, -0.07);
    cpu.use_bloch(bloch);
    gpu.use_bloch(bloch);
  }
  else {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  const component c = magnetic ? Hz : Ez;
  cpu.require_component(c);
  gpu.require_component(c);
  gaussian_src_time cpu_source(0.31, 0.14), gpu_source(0.31, 0.14);
  const std::complex<double> amplitude =
      complex_fields ? std::complex<double>(0.17, -0.09) : std::complex<double>(0.17, 0.0);
  cpu.add_point_source(c, cpu_source, vec(0.17, -0.13), amplitude);
  gpu.add_point_source(c, gpu_source, vec(0.17, -0.13), amplitude);
  cpu.init_backend();
  gpu.init_backend();
  const MaterialRecipeDisposition expected = tiled
                                                  ? MaterialRecipeDisposition::tiled_callback
                                                  : MaterialRecipeDisposition::host_reference;
  require(gpu.backend_state && gpu.initialization_plan &&
              gpu.initialization_plan->materials.size() == 1 &&
              gpu.initialization_plan->materials[0].disposition() == expected &&
              gpu.backend_state->material_route == expected &&
              (!tiled || callback_calls > 0),
          "PR5.4 route/policy fixture selected the wrong material disposition");
  compare_all_initialized_material_rows(
      cpu, gpu, sizeof(realnum) == sizeof(float) || policy != precision_policy_kind::native
                    ? 8e-5
                    : 2e-12);
  check_pr54_route_timesteps(cpu, gpu);
  master_printf("nvidia_timestep: PR5.4 %s/%s/%s/%s PASS\n",
                tiled ? "tiled" : "host-reference", magnetic ? "H" : "E",
                complex_fields ? "complex" : "real", precision_policy_name(policy));
}

static void run_pr54_strict_rejection_case(precision_policy_kind policy) {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.25, 1.0, 10.0);
  size_t callback_calls = 0;
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  install_pr54_tiled_material(s, callback_calls);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  options.strict = true;
  options.fallback = fallback_policy::error;
  fields f(&s, options);
  f.use_real_fields();
  f.require_component(Ez);
  const size_t callbacks_before = callback_calls;
  const nvidia::memory_accounting memory_before = nvidia::current_memory_accounting();
  nvidia::testing::reset_transfer_accounting();
  nvidia::testing::reset_material_transfer_accounting();
  bool rejected = false;
  try { f.advance(1); }
  catch (const std::exception &) { rejected = true; }
  const nvidia::memory_accounting memory_after = nvidia::current_memory_accounting();
  const nvidia::testing::transfer_accounting transfers =
      nvidia::testing::current_transfer_accounting();
  const nvidia::testing::material_transfer_accounting material =
      nvidia::testing::current_material_transfer_accounting();
  const bool clean = rejected && !f.backend_state && !f.executable &&
                     callback_calls == callbacks_before &&
                     f.backend->material_fallback_warning_count() == 0 &&
                     memory_before.device_bytes_current == memory_after.device_bytes_current &&
                     memory_before.pinned_bytes_current == memory_after.pinned_bytes_current &&
                     transfers.host_to_device_calls == 0 && transfers.device_to_host_calls == 0 &&
                     material.compact_calls == 0 && material.dense_output_calls == 0 &&
                     material.tiled_output_calls == 0;
  if (!clean)
    fprintf(stderr,
            "PR5.4 strict %s rejected=%d state=%d executable=%d callbacks=%zu/%zu "
            "warnings=%llu device=%zu/%zu pinned=%zu/%zu h2d=%zu d2h=%zu material=%zu/%zu/%zu\n",
            precision_policy_name(policy), int(rejected), int(f.backend_state != NULL),
            int(f.executable != NULL), callbacks_before, callback_calls,
            (unsigned long long)f.backend->material_fallback_warning_count(),
            memory_before.device_bytes_current, memory_after.device_bytes_current,
            memory_before.pinned_bytes_current, memory_after.pinned_bytes_current,
            transfers.host_to_device_calls, transfers.device_to_host_calls,
            material.compact_calls, material.dense_output_calls, material.tiled_output_calls);
  require(clean,
          "strict material fallback crossed the callback/allocation/transfer/publication boundary");
  master_printf("nvidia_timestep: PR5.4 strict/%s PASS\n", precision_policy_name(policy));
}

static void test_pr54_route_policy_matrix() {
  const precision_policy_kind policies[] = {
      precision_policy_kind::native, precision_policy_kind::mixed,
      precision_policy_kind::f32};
  for (precision_policy_kind policy : policies) {
    run_pr54_strict_rejection_case(policy);
    for (bool magnetic : {false, true})
      for (bool complex_fields : {false, true})
        for (bool tiled : {false, true})
          run_pr54_route_policy_case(policy, magnetic, complex_fields, tiled);
  }
  master_printf("nvidia_timestep: PR5.4 route/policy matrix PASS\n");
}

static void install_geometry_block_material(structure &s) {
  using namespace meep_geom;
  material_type medium = make_dielectric(5.0);
  medium->medium.epsilon_diag = make_vector3(5.0, 4.0, 3.0);
  medium->medium.epsilon_offdiag.x.re = 0.07;
  medium->medium.epsilon_offdiag.y.re = -0.05;
  medium->medium.epsilon_offdiag.z.re = 0.03;
  medium->medium.mu_diag = make_vector3(1.7, 1.5, 1.3);
  medium->medium.mu_offdiag.x.re = -0.04;
  medium->medium.mu_offdiag.y.re = 0.025;
  medium->medium.mu_offdiag.z.re = 0.015;
  medium->medium.D_conductivity_diag = make_vector3(0.02, 0.03, 0.04);
  medium->medium.B_conductivity_diag = make_vector3(0.012, 0.022, 0.032);
  medium->medium.E_chi2_diag = make_vector3(0.11, -0.07, 0.05);
  medium->medium.E_chi3_diag = make_vector3(0.013, 0.017, -0.019);
  medium->medium.H_chi2_diag = make_vector3(-0.09, 0.08, 0.06);
  medium->medium.H_chi3_diag = make_vector3(0.009, -0.008, 0.007);
  meep_geom::susceptibility electric = meep_geom::susceptibility();
  electric.frequency = 0.43;
  electric.gamma = 0.037;
  electric.sigma_diag = make_vector3(0.6, 0.5, 0.4);
  electric.sigma_offdiag = make_vector3(0.03, -0.02, 0.01);
  medium->medium.E_susceptibilities.push_back(electric);
  meep_geom::susceptibility magnetic = meep_geom::susceptibility();
  magnetic.frequency = 0.31;
  magnetic.gamma = 0.027;
  magnetic.sigma_diag = make_vector3(0.35, 0.3, 0.25);
  magnetic.sigma_offdiag = make_vector3(-0.018, 0.014, 0.009);
  medium->medium.H_susceptibilities.push_back(magnetic);
  geometric_object object =
      make_block(medium, make_vector3(0.03, -0.02, 0.0), make_vector3(1, 0, 0),
                 make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                 make_vector3(0.63, 0.61, 1.0));
  geometric_object_list geometry = {1, &object};
  material_type extras_data[1] = {medium};
  material_type_list extras;
  extras.num_items = 1;
  extras.items = extras_data;
  set_materials_from_geometry(&s, geometry, make_vector3(), true, 1e-5, 256, false,
                              vacuum, 0, extras);
  geometric_object_destroy(object);
  material_free(medium);
}

static void test_native_material_geometry(precision_policy_kind precision) {
  const grid_volume gv = vol2d(1.0, 1.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  install_geometry_block_material(cpu_structure);
  install_geometry_block_material(gpu_structure);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  FOR_E_AND_H(c) if (gv.has_field(c)) {
      cpu.require_component(c);
      gpu.require_component(c);
    }
  cpu.init_backend();
  nvidia::testing::reset_transfer_accounting();
  nvidia::testing::reset_material_transfer_accounting();
  gpu.init_backend();
  const MaterialIR *ir = material_ir_for(gpu);
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  const NvidiaMaterialInitializationStatistics statistics =
      backend ? backend->material_initialization_statistics_for_testing()
              : NvidiaMaterialInitializationStatistics();
  size_t bulk_points = 0;
  if (ir)
    for (const MaterialIRBulkSpan &span : ir->bulk_spans) bulk_points += span.count;
  require(ir && backend && ir->requires_hybrid && !ir->bulk_spans.empty() &&
              !ir->analytic_interfaces.empty() && !ir->hybrid_patches.empty() &&
              gpu.initialization_plan && gpu.initialization_plan->materials.size() == 1 &&
              gpu.initialization_plan->materials[0].disposition() ==
                  MaterialRecipeDisposition::hybrid_interface,
          "geometry block fixture did not retain the native/hybrid partition");
  require(statistics.valid && statistics.device_native &&
              statistics.geometry_bulk_points == bulk_points &&
              statistics.geometry_analytic_points == ir->analytic_interfaces.size() &&
              statistics.geometry_patch_points == ir->hybrid_patches.size() &&
              statistics.geometry_bulk_kernel_launches > 0 &&
              statistics.geometry_analytic_kernel_launches > 0 &&
              statistics.geometry_patch_kernel_launches > 0 &&
              statistics.geometry_patch_bytes ==
                  ir->hybrid_patches.size() * sizeof(nvidia::geometry_patch_record) &&
              statistics.dense_output_host_to_device_calls == 0 &&
              statistics.dense_output_host_to_device_bytes == 0 &&
              statistics.synchronizations == 1,
          "geometry block initialization accounting is inconsistent");
  compare_all_initialized_material_rows(
      cpu, gpu, sizeof(realnum) == sizeof(float) || precision != precision_policy_kind::native
                    ? 8e-5
                    : 5e-12);
  master_printf("nvidia_timestep: material-geometry/%s PASS\n",
                precision_policy_name(precision));
}

enum driven_geometry_route { driven_bulk, driven_analytic, driven_patch };

static void run_noisy_replay_case(precision_policy_kind policy);
static void test_finite_diagnostics(precision_policy_kind policy);

static void install_driven_geometry(structure &s, driven_geometry_route route) {
  using namespace meep_geom;
  material_type medium = make_dielectric(4.0);
  meep_geom::susceptibility oscillator = meep_geom::susceptibility();
  oscillator.frequency = 0.47;
  oscillator.gamma = 0.035;
  oscillator.sigma_diag = make_vector3(0.35, 0.25, 0.15);
  medium->medium.E_susceptibilities.push_back(oscillator);
  geometric_object object =
      route == driven_patch
          ? make_sphere(medium, make_vector3(0.03, -0.02), 0.37)
          : make_block(medium, make_vector3(0.03, -0.02), make_vector3(1, 0, 0),
                       make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                       make_vector3(0.63, 0.61, 1.0));
  geometric_object_list geometry = {1, &object};
  set_materials_from_geometry(&s, geometry, make_vector3(), route != driven_bulk, 1e-5, 256,
                              false, vacuum);
  geometric_object_destroy(object);
  material_free(medium);
}

static void compare_geometry_dft(fields &cpu, fields &gpu, const dft_fields &cpu_monitor,
                                 const dft_fields &gpu_monitor, component c,
                                 double tolerance) {
  int cpu_rank = 0, gpu_rank = 0;
  size_t cpu_dims[3] = {0, 0, 0}, gpu_dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> expected(
      cpu.get_dft_array(cpu_monitor, c, 0, &cpu_rank, cpu_dims));
  std::unique_ptr<std::complex<realnum>[]> observed(
      gpu.get_dft_array(gpu_monitor, c, 0, &gpu_rank, gpu_dims));
  require(expected && observed && cpu_rank == gpu_rank,
          "driven geometry DFT metadata differs");
  size_t count = 1;
  for (int axis = 0; axis < cpu_rank; ++axis) {
    require(cpu_dims[axis] == gpu_dims[axis], "driven geometry DFT dimensions differ");
    count *= cpu_dims[axis];
  }
  for (size_t i = 0; i < count; ++i)
    require(std::abs(expected[i] - observed[i]) <=
                tolerance * (1.0 + std::abs(expected[i])),
            "driven geometry DFT values differ");
}

static void test_driven_material_geometry(
    driven_geometry_route route,
    precision_policy_kind precision = precision_policy_kind::native) {
  const char *name = route == driven_bulk ? "bulk" : route == driven_analytic ? "analytic"
                                                                           : "patch";
  const grid_volume gv = vol2d(1.0, 1.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  install_driven_geometry(cpu_structure, route);
  install_driven_geometry(gpu_structure, route);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  for (component c : {Ex, Ey, Ez, Hx, Hy, Hz}) {
    cpu.require_component(c);
    gpu.require_component(c);
  }
  gaussian_src_time cpu_time(0.31, 0.14), gpu_time(0.31, 0.14);
  cpu.add_point_source(Ez, cpu_time, vec(-0.31, 0.27), std::complex<double>(0.37, 0.0));
  gpu.add_point_source(Ez, gpu_time, vec(-0.31, 0.27), std::complex<double>(0.37, 0.0));
  component monitor_component = Ez;
  dft_fields cpu_monitor =
      cpu.add_dft_fields(&monitor_component, 1, cpu.v, 0.31, 0.31, 1);
  dft_fields gpu_monitor =
      gpu.add_dft_fields(&monitor_component, 1, gpu.v, 0.31, 0.31, 1);
  flux_vol *cpu_flux =
      cpu.add_flux_vol(X, volume(vec(0.0, -0.45), vec(0.0, 0.45)));
  flux_vol *gpu_flux =
      gpu.add_flux_vol(X, volume(vec(0.0, -0.45), vec(0.0, 0.45)));
  cpu.init_backend();
  gpu.init_backend();
  NvidiaBackend *graph_backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  const bool assert_graph = getenv("MEEP_NVIDIA_GRAPH_ASSERT") != NULL;
  const bool expect_graph = getenv("MEEP_NVIDIA_GRAPH_EXPECT_ENABLED") != NULL;
  if (assert_graph) {
    const NvidiaGraphStatistics stats =
        graph_backend ? graph_backend->graph_statistics_for_testing() : NvidiaGraphStatistics();
    require(graph_backend && stats.valid && stats.enabled == expect_graph,
            "NVIDIA graph mode selection differs from expectation");
    require(!expect_graph ||
                (stats.segment_count > 0 && stats.capture_count == stats.segment_count &&
                 stats.instantiate_count == stats.segment_count),
            "NVIDIA graph capture/instantiate accounting is incomplete");
  }
  const MaterialIR *ir = material_ir_for(gpu);
  require(ir && !ir->bulk_spans.empty(), "driven geometry fixture has no bulk work");
  require((route == driven_bulk && ir->analytic_interfaces.empty() &&
           ir->hybrid_patches.empty()) ||
              (route == driven_analytic && !ir->analytic_interfaces.empty()) ||
              (route == driven_patch && !ir->hybrid_patches.empty()),
          "driven geometry fixture did not exercise its requested route");
  size_t polarization_arrays = 0;
  for (size_t i = 0; i < gpu.array_catalog->size(); ++i)
    polarization_arrays += gpu.array_catalog->spec(ArrayId{uint32_t(i)}).role ==
                           array_role::polarization;
  const StepPlan plan = build_step_plan(gpu, StepProgram::ordinary);
  bool polarization = false, dft = false, flux_half = false, flux_full = false;
  for (const Operation &operation : plan.operations) {
    polarization = polarization || operation.kind == OpKind::update_polarization;
    dft = dft || operation.kind == OpKind::update_dft;
    flux_half = flux_half || operation.kind == OpKind::update_flux_half;
    flux_full = flux_full || operation.kind == OpKind::update_flux;
  }
  require(polarization_arrays && polarization && dft && flux_half && flux_full,
          "driven geometry fixture omitted polarization/DFT/flux state");
  const double tolerance =
      sizeof(realnum) == sizeof(float) || precision != precision_policy_kind::native ? 2e-4
                                                                                    : 2e-11;
  int completed = 0;
  const int checkpoints[] = {1, 2, 100};
  for (int checkpoint : checkpoints) {
    cpu.advance(checkpoint - completed);
    gpu.advance(checkpoint - completed);
    completed = checkpoint;
    compare_live_fields_by_key(cpu, gpu, tolerance);
    require(std::fabs(cpu_flux->flux() - gpu_flux->flux()) <=
                tolerance * (1.0 + std::fabs(cpu_flux->flux())),
            "driven geometry flux differs");
    compare_geometry_dft(cpu, gpu, cpu_monitor, gpu_monitor, monitor_component, tolerance);
  }
  if (assert_graph) {
    const NvidiaGraphStatistics stats = graph_backend->graph_statistics_for_testing();
    require((expect_graph && stats.scalar_write_count == size_t(completed) &&
             stats.launch_count == size_t(completed) * stats.segment_count &&
             stats.boundary_count > 0) ||
                (!expect_graph && stats.scalar_write_count == 0 && stats.launch_count == 0),
            "NVIDIA graph steady-state accounting differs from the compiled schedule");
  }
  master_printf("nvidia_timestep: material-geometry-driven-%s PASS\n", name);
}

static void test_graph_required_compile_failure() {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "required", 1);
  const grid_volume gv = vol2d(1.0, 1.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  nvidia::testing::fail_next(nvidia::testing::failure_point::graph_instantiate);
  bool rejected = false;
  try { gpu.init_backend(); }
  catch (const std::exception &error) {
    rejected = std::string(error.what()).find("graph instantiate") != std::string::npos;
  }
  nvidia::testing::clear_failure();
  require(rejected && gpu.t == 0,
          "required NVIDIA graph capture failure was not rejected before dispatch");
}

static void test_graph_teardown_lifetime() {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "required", 1);
  nvidia::testing::reset_graph_accounting();
  const nvidia::memory_accounting memory_before = nvidia::current_memory_accounting();
  size_t instantiated = 0;
  {
    const grid_volume gv = vol2d(1.0, 1.0, 6.0);
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    execution_options options;
    options.backend = backend_kind::nvidia;
    options.strict = false;
    options.fallback = fallback_policy::warn;
    fields gpu(&s, options);
    gpu.use_real_fields();
    gpu.require_component(Ez);
    gpu.init_backend();
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
    const NvidiaGraphStatistics stats =
        backend ? backend->graph_statistics_for_testing() : NvidiaGraphStatistics();
    require(backend && stats.enabled && stats.segment_count > 0 &&
                stats.instantiate_count == stats.segment_count,
            "NVIDIA graph teardown fixture did not own executable graphs");
    instantiated = stats.instantiate_count;
  }
  const nvidia::graph_accounting graph_after = nvidia::testing::current_graph_accounting();
  const nvidia::memory_accounting memory_after = nvidia::current_memory_accounting();
  require(graph_after.executable_destroys == instantiated &&
              graph_after.graph_destroys == instantiated,
          "NVIDIA executable teardown did not destroy every captured graph definition first");
  require(memory_after.device_bytes_current == memory_before.device_bytes_current &&
              memory_after.pinned_bytes_current == memory_before.pinned_bytes_current,
          "NVIDIA graph teardown outlived or leaked a captured allocation");
}

static void test_graph_collective_reconciliation() {
  using nvidia::testing::graph_collective_probe;
  const int rank = my_rank();
  const int ranks = count_processors();
  const int asymmetric_rank = ranks - 1;
  const graph_collective_probe supported = {
      "auto", true, true, true, true, true, true, true, true, true, true};

  const auto expect_rejected = [&](graph_collective_probe probe, const char *message) {
    bool rejected = false;
    try {
      (void)nvidia::testing::reconcile_graph_execution_for_testing(probe);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    require(and_to_all(rejected), message);
  };
  const auto expect_enabled = [&](graph_collective_probe probe, bool enabled,
                                  const char *message) {
    bool selected = false;
    bool completed = false;
    try {
      selected = nvidia::testing::reconcile_graph_execution_for_testing(probe).use_graph;
      completed = true;
    }
    catch (...) {}
    const bool all_completed = and_to_all(completed);
    const int minimum_selected = min_to_all(selected ? 1 : 0);
    const int maximum_selected = max_to_all(selected ? 1 : 0);
    require(all_completed && minimum_selected == maximum_selected && selected == enabled,
            message);
  };

  graph_collective_probe probe = supported;
  probe.mode = rank == 0 ? "invalid" : "auto";
  expect_rejected(probe, "rank-asymmetric graph-mode parse failure was not collective");

  probe = supported;
  probe.mode = rank == 0 ? "required" : "auto";
  if (ranks > 1)
    expect_rejected(probe, "rank-asymmetric graph-mode mismatch was not collective");
  else
    expect_enabled(probe, true, "single-rank required graph mode was not selected");

  probe = supported;
  probe.lowering_valid = rank != asymmetric_rank;
  expect_rejected(probe, "rank-asymmetric graph lowering failure was not collective");

  probe = supported;
  probe.validation_valid = rank != asymmetric_rank;
  expect_rejected(probe, "rank-asymmetric graph validation failure was not collective");

  probe = supported;
  probe.runtime_capture_supported = rank != asymmetric_rank;
  expect_enabled(probe, false, "automatic graph support fallback was not collective");

  probe = supported;
  probe.allocation_valid = rank != asymmetric_rank;
  expect_enabled(probe, false, "automatic graph allocation fallback was not collective");

  probe = supported;
  probe.capture_valid = rank != asymmetric_rank;
  expect_enabled(probe, false, "automatic graph capture fallback was not collective");

  probe = supported;
  probe.instantiate_valid = rank != asymmetric_rank;
  expect_enabled(probe, false, "automatic graph instantiate fallback was not collective");

  probe = supported;
  probe.capture_valid = false;
  probe.graph_destroy_valid = rank != asymmetric_rank;
  expect_rejected(probe,
                  "rank-asymmetric graph-definition cleanup failure was not collective");

  probe = supported;
  probe.instantiate_valid = false;
  probe.graph_exec_destroy_valid = rank != asymmetric_rank;
  expect_rejected(probe,
                  "rank-asymmetric graph-executable cleanup failure was not collective");

  probe = supported;
  probe.instantiate_valid = false;
  probe.graph_device_restore_valid = rank != asymmetric_rank;
  expect_rejected(probe,
                  "rank-asymmetric graph cleanup device-restoration failure was not collective");

  probe = supported;
  probe.mode = "required";
  probe.program_graphable = rank != asymmetric_rank;
  expect_rejected(probe, "required graph support failure was not collective");

  probe = supported;
  probe.mode = "required";
  probe.allocation_valid = rank != asymmetric_rank;
  expect_rejected(probe, "required graph allocation failure was not collective");

  probe = supported;
  probe.mode = "required";
  probe.capture_valid = rank != asymmetric_rank;
  expect_rejected(probe, "required graph capture failure was not collective");

  probe = supported;
  probe.mode = "required";
  probe.instantiate_valid = rank != asymmetric_rank;
  expect_rejected(probe, "required graph instantiate failure was not collective");

  expect_enabled(supported, true, "collectively supported automatic graph mode was not enabled");
  master_printf("nvidia_timestep: graph collective reconciliation PASS (%d ranks)\n", ranks);
}

static void test_graph_auto_capture_fallback() {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "auto", 1);
  const grid_volume gv = vol2d(1.0, 1.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  nvidia::testing::fail_next(nvidia::testing::failure_point::graph_instantiate);
  gpu.init_backend();
  nvidia::testing::clear_failure();
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  const NvidiaGraphStatistics stats =
      backend ? backend->graph_statistics_for_testing() : NvidiaGraphStatistics();
  require(backend && stats.valid && !stats.enabled,
          "automatic NVIDIA graph capture failure did not select whole-program eager");
  gpu.advance(1);
  require(gpu.t == 1 && !gpu.backend->is_poisoned(),
          "automatic NVIDIA graph fallback was not dispatchable");
}

static void test_graph_launch_failure_poison() {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "required", 1);
  const grid_volume gv = vol2d(1.0, 1.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  gpu.init_backend();
  nvidia::testing::fail_next(nvidia::testing::failure_point::graph_launch);
  bool rejected = false;
  try { gpu.advance(1); }
  catch (const std::exception &error) {
    rejected = std::string(error.what()).find("graph launch") != std::string::npos;
  }
  nvidia::testing::clear_failure();
  require(rejected && gpu.t == 0 && gpu.backend->is_poisoned(),
          "NVIDIA graph launch failure did not poison before host time publication");
}

static void test_graph_execution_integration() {
  setenv("MEEP_NVIDIA_GRAPH_ASSERT", "1", 1);
  unsetenv("MEEP_NVIDIA_GRAPH_EXPECT_ENABLED");
  setenv("MEEP_NVIDIA_GRAPH_MODE", "eager", 1);
  test_driven_material_geometry(driven_bulk);
  setenv("MEEP_NVIDIA_GRAPH_MODE", "required", 1);
  setenv("MEEP_NVIDIA_GRAPH_EXPECT_ENABLED", "1", 1);
  test_driven_material_geometry(driven_bulk);
  test_driven_material_geometry(driven_bulk, precision_policy_kind::f32);
  run_material_phase_case("graph-material-phase", precision_policy_kind::native, true, false);
  run_custom_source_case("graph-custom-source", precision_policy_kind::native, false);
  run_noisy_replay_case(precision_policy_kind::native);
  test_finite_diagnostics(precision_policy_kind::native);
  test_legacy_flux_postlaunch_poison();
  set_finite_check_mode(FiniteCheckMode::off);
  unsetenv("MEEP_NVIDIA_GRAPH_ASSERT");
  unsetenv("MEEP_NVIDIA_GRAPH_EXPECT_ENABLED");
  test_graph_auto_capture_fallback();
  test_graph_required_compile_failure();
  test_graph_launch_failure_poison();
  test_graph_teardown_lifetime();
  unsetenv("MEEP_NVIDIA_GRAPH_MODE");
  master_printf("nvidia_timestep: graph execution integration PASS\n");
}

static void test_native_mesh_geometry() {
  using namespace meep_geom;
  const grid_volume gv = vol3d(3.0, 3.0, 3.0, 4.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  material_type cpu_medium = make_dielectric(6.0);
  material_type gpu_medium = make_dielectric(6.0);
  const vector3 vertices[8] = {
      make_vector3(-1, -1, -1), make_vector3(1, -1, -1),
      make_vector3(1, 1, -1), make_vector3(-1, 1, -1),
      make_vector3(-1, -1, 1), make_vector3(1, -1, 1),
      make_vector3(1, 1, 1), make_vector3(-1, 1, 1)};
  const int triangles[36] = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
                             0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
                             2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7};
  geometric_object cpu_object = make_mesh(cpu_medium, vertices, 8, triangles, 12);
  geometric_object gpu_object = make_mesh(gpu_medium, vertices, 8, triangles, 12);
  geometric_object_list cpu_geometry = {1, &cpu_object};
  geometric_object_list gpu_geometry = {1, &gpu_object};
  set_materials_from_geometry(&cpu_structure, cpu_geometry, make_vector3(), false, 1e-5, 128,
                              false, vacuum);
  set_materials_from_geometry(&gpu_structure, gpu_geometry, make_vector3(), false, 1e-5, 128,
                              false, vacuum);
  geometric_object_destroy(cpu_object);
  geometric_object_destroy(gpu_object);
  material_free(cpu_medium);
  material_free(gpu_medium);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  for (component c : {Ex, Ey, Ez, Hx, Hy, Hz}) {
    cpu.require_component(c);
    gpu.require_component(c);
  }
  cpu.init_backend();
  gpu.init_backend();
  const MaterialIR *ir = material_ir_for(gpu);
  const NvidiaBackend *backend = dynamic_cast<const NvidiaBackend *>(gpu.backend);
  const NvidiaMaterialInitializationStatistics statistics =
      backend ? backend->material_initialization_statistics_for_testing()
              : NvidiaMaterialInitializationStatistics();
  require(ir && ir->objects.size() == 1 && ir->objects[0].bvh_count > 1 &&
              !ir->bulk_spans.empty() && ir->analytic_interfaces.empty() &&
              ir->hybrid_patches.empty() && statistics.geometry_object_bytes > 0,
          "mesh geometry fixture did not execute the packed BVH bulk path");
  compare_all_initialized_material_rows(cpu, gpu, 5e-12);
  master_printf("nvidia_timestep: material-geometry-mesh PASS\n");
}

enum geometry_shape_case {
  geometry_sphere,
  geometry_ellipsoid,
  geometry_cylinder,
  geometry_cone,
  geometry_wedge,
  geometry_negative_wedge,
  geometry_transformed_block,
  geometry_prism
};

static geometric_object make_geometry_shape(meep_geom::material_type material,
                                            geometry_shape_case shape) {
  using namespace meep_geom;
  switch (shape) {
    case geometry_sphere: return make_sphere(material, make_vector3(), 0.8);
    case geometry_ellipsoid:
      return make_ellipsoid(material, make_vector3(), make_vector3(1, 0, 0),
                            make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                            make_vector3(1.6, 1.0, 1.4));
    case geometry_cylinder:
      return make_cylinder(material, make_vector3(), 0.7, 1.5, make_vector3(0, 0, 1));
    case geometry_cone:
      return make_cone(material, make_vector3(), 0.8, 1.5, make_vector3(0, 0, 1), 0.35);
    case geometry_wedge:
      return make_wedge(material, make_vector3(), 0.8, 1.5, make_vector3(0, 0, 1),
                        0.5 * M_PI, make_vector3(1, 0, 0));
    case geometry_negative_wedge:
      return make_wedge(material, make_vector3(), 0.8, 1.5, make_vector3(0, 0, 1),
                        -0.5 * M_PI, make_vector3(1, 0, 0));
    case geometry_transformed_block:
      return make_block(material, make_vector3(0.25, -0.5, 0.125), make_vector3(0, 1, 0),
                        make_vector3(-1, 0, 0), make_vector3(0, 0, 1),
                        make_vector3(1.4, 0.8, 1.2));
    case geometry_prism: {
      const vector3 square[4] = {make_vector3(-0.7, -0.7), make_vector3(0.7, -0.7),
                                 make_vector3(0.7, 0.7), make_vector3(-0.7, 0.7)};
      return make_prism_with_center(material, make_vector3(), square, 4, 1.4,
                                    make_vector3(0, 0, 1));
    }
  }
  throw std::logic_error("unknown geometry shape fixture");
}

static void test_native_geometry_shapes() {
  using namespace meep_geom;
  const geometry_shape_case shapes[] = {
      geometry_sphere, geometry_ellipsoid, geometry_cylinder, geometry_cone,
      geometry_wedge, geometry_negative_wedge, geometry_transformed_block,
      geometry_prism};
  const char *names[] = {"sphere", "ellipsoid", "cylinder", "cone", "wedge",
                         "negative-wedge", "transformed-block", "prism"};
  for (size_t shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); ++shape) {
    const grid_volume gv = vol3d(3.0, 3.0, 3.0, 4.0);
    structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
    structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
    meep_geom::material_type cpu_medium = meep_geom::make_dielectric(3.25);
    meep_geom::material_type gpu_medium = meep_geom::make_dielectric(3.25);
    geometric_object cpu_object = make_geometry_shape(cpu_medium, shapes[shape]);
    geometric_object gpu_object = make_geometry_shape(gpu_medium, shapes[shape]);
    geometric_object_list cpu_geometry = {1, &cpu_object};
    geometric_object_list gpu_geometry = {1, &gpu_object};
    set_materials_from_geometry(&cpu_structure, cpu_geometry, make_vector3(), false, 1e-5, 128,
                                false, vacuum);
    set_materials_from_geometry(&gpu_structure, gpu_geometry, make_vector3(), false, 1e-5, 128,
                                false, vacuum);
    geometric_object_destroy(cpu_object);
    geometric_object_destroy(gpu_object);
    material_free(cpu_medium);
    material_free(gpu_medium);
    fields cpu(&cpu_structure);
    execution_options options;
    options.backend = backend_kind::nvidia;
    fields gpu(&gpu_structure, options);
    cpu.use_real_fields();
    gpu.use_real_fields();
    for (component c : {Ex, Ey, Ez, Hx, Hy, Hz}) {
      cpu.require_component(c);
      gpu.require_component(c);
    }
    cpu.init_backend();
    gpu.init_backend();
    compare_all_initialized_material_rows(cpu, gpu, 5e-12);
    master_printf("nvidia_timestep: material-geometry-%s PASS\n", names[shape]);
  }
}

static meep_geom::material_type make_geometry_grid_material(bool averaging, int reducer) {
  using namespace meep_geom;
  material_type grid = new material_data();
  grid->which_subclass = material_data::MATERIAL_GRID;
  grid->do_averaging = averaging;
  grid->material_grid_kinds = static_cast<decltype(grid->material_grid_kinds)>(reducer);
  grid->grid_size = make_vector3(2, 2, 1);
  grid->weights = new double[4]{0.15, 0.35, 0.65, 0.85};
  grid->medium_1 = medium_struct(2.0);
  grid->medium_2 = medium_struct(6.0);
  grid->medium_1.D_conductivity_diag = make_vector3(0.01, 0.02, 0.03);
  grid->medium_2.D_conductivity_diag = make_vector3(0.05, 0.06, 0.07);
  meep_geom::susceptibility first = meep_geom::susceptibility();
  first.frequency = 0.53;
  first.gamma = 0.041;
  first.sigma_diag = make_vector3(0.2, 0.3, 0.4);
  first.sigma_offdiag = make_vector3(0.015, -0.012, 0.009);
  meep_geom::susceptibility second = first;
  second.sigma_diag = make_vector3(0.7, 0.8, 0.9);
  second.sigma_offdiag = make_vector3(-0.025, 0.021, -0.017);
  grid->medium_1.E_susceptibilities.push_back(first);
  grid->medium_2.E_susceptibilities.push_back(second);
  grid->medium.E_susceptibilities.push_back(first);
  grid->medium.E_susceptibilities.push_back(second);
  grid->beta = 1.7;
  grid->eta = 0.43;
  grid->damping = 0.09;
  return grid;
}

static void test_native_geometry_grid(bool object_grid, bool averaging) {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.0, 1.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  material_type cpu_grid = make_geometry_grid_material(averaging, material_data::U_MEAN);
  material_type gpu_grid = make_geometry_grid_material(averaging, material_data::U_MEAN);
  geometric_object cpu_object = {}, gpu_object = {};
  geometric_object_list cpu_geometry = {0, NULL}, gpu_geometry = {0, NULL};
  if (object_grid) {
    cpu_object = make_block(cpu_grid, make_vector3(), make_vector3(1, 0, 0),
                            make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                            make_vector3(0.72, 0.68, 1.0));
    gpu_object = make_block(gpu_grid, make_vector3(), make_vector3(1, 0, 0),
                            make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                            make_vector3(0.72, 0.68, 1.0));
    cpu_geometry = geometric_object_list{1, &cpu_object};
    gpu_geometry = geometric_object_list{1, &gpu_object};
  }
  set_materials_from_geometry(&cpu_structure, cpu_geometry, make_vector3(), averaging, 1e-5, 256,
                              false, object_grid ? vacuum : cpu_grid);
  set_materials_from_geometry(&gpu_structure, gpu_geometry, make_vector3(), averaging, 1e-5, 256,
                              false, object_grid ? vacuum : gpu_grid);
  if (object_grid) {
    geometric_object_destroy(cpu_object);
    geometric_object_destroy(gpu_object);
  }
  material_free(cpu_grid);
  material_free(gpu_grid);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  for (component c : {Ex, Ey, Ez}) {
    cpu.require_component(c);
    gpu.require_component(c);
  }
  cpu.init_backend();
  gpu.init_backend();
  const MaterialIR *ir = material_ir_for(gpu);
  const NvidiaBackend *backend = dynamic_cast<const NvidiaBackend *>(gpu.backend);
  const NvidiaMaterialInitializationStatistics statistics =
      backend ? backend->material_initialization_statistics_for_testing()
              : NvidiaMaterialInitializationStatistics();
  require(ir && backend && statistics.valid && statistics.device_native &&
              statistics.grid_weight_bytes == 4 * sizeof(double) &&
              statistics.dense_output_host_to_device_calls == 0 &&
              (averaging ? !ir->hybrid_patches.empty() : ir->hybrid_patches.empty()),
          "geometry MaterialGrid did not take the expected compact route");
  compare_all_initialized_material_rows(cpu, gpu, 5e-11);
  master_printf("nvidia_timestep: material-geometry-%s-grid-%s PASS\n",
                object_grid ? "object" : "default", averaging ? "hybrid" : "bulk");
}

static meep_geom::material_type make_geometry_file_material() {
  using namespace meep_geom;
  material_type file = new material_data();
  file->which_subclass = material_data::MATERIAL_FILE;
  file->medium = medium_struct(1.0);
  file->epsilon_dims[0] = 2;
  file->epsilon_dims[1] = 2;
  file->epsilon_dims[2] = 1;
  file->epsilon_data = new double[4]{2.0, 3.0, 5.0, 7.0};
  return file;
}

static void test_native_object_file_geometry() {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.0, 1.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  material_type cpu_file = make_geometry_file_material();
  material_type gpu_file = make_geometry_file_material();
  geometric_object cpu_object = make_sphere(cpu_file, make_vector3(), 0.42);
  geometric_object gpu_object = make_sphere(gpu_file, make_vector3(), 0.42);
  geometric_object_list cpu_geometry = {1, &cpu_object};
  geometric_object_list gpu_geometry = {1, &gpu_object};
  set_materials_from_geometry(&cpu_structure, cpu_geometry, make_vector3(), false, 1e-5, 128,
                              false, vacuum);
  set_materials_from_geometry(&gpu_structure, gpu_geometry, make_vector3(), false, 1e-5, 128,
                              false, vacuum);
  geometric_object_destroy(cpu_object);
  geometric_object_destroy(gpu_object);
  material_free(cpu_file);
  material_free(gpu_file);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  for (component c : {Ex, Ey, Ez, Hx, Hy, Hz}) {
    cpu.require_component(c);
    gpu.require_component(c);
  }
  cpu.init_backend();
  gpu.init_backend();
  const NvidiaBackend *backend = dynamic_cast<const NvidiaBackend *>(gpu.backend);
  const NvidiaMaterialInitializationStatistics statistics =
      backend ? backend->material_initialization_statistics_for_testing()
              : NvidiaMaterialInitializationStatistics();
  require(backend && statistics.valid && statistics.file_sample_bytes == 4 * sizeof(double) &&
              statistics.dense_output_host_to_device_calls == 0,
          "object FILE material did not execute the compact geometry path");
  compare_all_initialized_material_rows(cpu, gpu, 5e-12);
  master_printf("nvidia_timestep: material-geometry-object-file PASS\n");
}

static void test_native_geometry_grid_absorber() {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.5, 1.25, 8.0);
  structure cpu_structure(gv, isotropic_eps, pml(0.25), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, pml(0.25), identity(), 1);
  material_type cpu_grid = make_geometry_grid_material(false, material_data::U_DEFAULT);
  material_type gpu_grid = make_geometry_grid_material(false, material_data::U_DEFAULT);
  geometric_object cpu_object =
      make_block(cpu_grid, make_vector3(), make_vector3(1, 0, 0), make_vector3(0, 1, 0),
                 make_vector3(0, 0, 1), make_vector3(0.9, 0.8, 1.0));
  geometric_object gpu_object =
      make_block(gpu_grid, make_vector3(), make_vector3(1, 0, 0), make_vector3(0, 1, 0),
                 make_vector3(0, 0, 1), make_vector3(0.9, 0.8, 1.0));
  geometric_object_list cpu_geometry = {1, &cpu_object};
  geometric_object_list gpu_geometry = {1, &gpu_object};
  absorber_list absorbers = create_absorber_list();
  add_absorbing_layer(absorbers, 0.5, ALL_DIRECTIONS, ALL_SIDES, 1e-9, 1.0);
  set_materials_from_geometry(&cpu_structure, cpu_geometry, make_vector3(), false, 1e-5, 128,
                              false, vacuum, absorbers);
  set_materials_from_geometry(&gpu_structure, gpu_geometry, make_vector3(), false, 1e-5, 128,
                              false, vacuum, absorbers);
  destroy_absorber_list(absorbers);
  geometric_object_destroy(cpu_object);
  geometric_object_destroy(gpu_object);
  material_free(cpu_grid);
  material_free(gpu_grid);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  for (component c : {Ex, Ey, Ez}) {
    cpu.require_component(c);
    gpu.require_component(c);
  }
  cpu.init_backend();
  gpu.init_backend();
  const NvidiaBackend *backend = dynamic_cast<const NvidiaBackend *>(gpu.backend);
  const NvidiaMaterialInitializationStatistics statistics =
      backend ? backend->material_initialization_statistics_for_testing()
              : NvidiaMaterialInitializationStatistics();
  require(backend && statistics.valid && statistics.absorber_profile_bytes > 0 &&
              statistics.pml_profile_bytes > 0 && statistics.geometry_bulk_points > 0 &&
              statistics.pml_kernel_launches > 0,
          "geometry MaterialGrid absorber/PML path was not executed");
  compare_all_initialized_material_rows(cpu, gpu, 8e-11);
  master_printf("nvidia_timestep: material-geometry-grid-absorber PASS\n");
}

static meep_geom::material_type make_geometry_constant_grid(double weight, int reducer) {
  using namespace meep_geom;
  material_type grid = new material_data();
  grid->which_subclass = material_data::MATERIAL_GRID;
  grid->do_averaging = false;
  grid->material_grid_kinds = static_cast<decltype(grid->material_grid_kinds)>(reducer);
  grid->grid_size = make_vector3(1, 1, 1);
  grid->weights = new double[1]{weight};
  grid->medium_1 = medium_struct(2.0);
  grid->medium_2 = medium_struct(6.0);
  grid->medium = grid->medium_1;
  grid->beta = 0.0;
  grid->eta = 0.5;
  grid->damping = 0.0;
  return grid;
}

static void test_native_geometry_grid_reducers() {
  using namespace meep_geom;
  const int reducers[4] = {material_data::U_DEFAULT, material_data::U_MIN,
                           material_data::U_PROD, material_data::U_MEAN};
  for (int reducer : reducers) {
    const grid_volume gv = vol2d(1.0, 1.0, 8.0);
    structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
    structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
    material_type cpu_materials[3] = {make_geometry_constant_grid(0.2, reducer),
                                      make_geometry_constant_grid(0.8, reducer),
                                      make_geometry_constant_grid(0.4, reducer)};
    material_type gpu_materials[3] = {make_geometry_constant_grid(0.2, reducer),
                                      make_geometry_constant_grid(0.8, reducer),
                                      make_geometry_constant_grid(0.4, reducer)};
    geometric_object cpu_objects[2] = {
        make_block(cpu_materials[0], make_vector3(), make_vector3(1, 0, 0),
                   make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                   make_vector3(0.85, 0.85, 1.0)),
        make_block(cpu_materials[1], make_vector3(), make_vector3(1, 0, 0),
                   make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                   make_vector3(0.55, 0.55, 1.0))};
    geometric_object gpu_objects[2] = {
        make_block(gpu_materials[0], make_vector3(), make_vector3(1, 0, 0),
                   make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                   make_vector3(0.85, 0.85, 1.0)),
        make_block(gpu_materials[1], make_vector3(), make_vector3(1, 0, 0),
                   make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                   make_vector3(0.55, 0.55, 1.0))};
    geometric_object_list cpu_geometry = {2, cpu_objects};
    geometric_object_list gpu_geometry = {2, gpu_objects};
    set_materials_from_geometry(&cpu_structure, cpu_geometry, make_vector3(), false, 1e-5, 128,
                                false, cpu_materials[2]);
    set_materials_from_geometry(&gpu_structure, gpu_geometry, make_vector3(), false, 1e-5, 128,
                                false, gpu_materials[2]);
    for (int i = 1; i >= 0; --i) {
      geometric_object_destroy(cpu_objects[i]);
      geometric_object_destroy(gpu_objects[i]);
    }
    for (int i = 0; i < 3; ++i) {
      material_free(cpu_materials[i]);
      material_free(gpu_materials[i]);
    }
    fields cpu(&cpu_structure);
    execution_options options;
    options.backend = backend_kind::nvidia;
    fields gpu(&gpu_structure, options);
    cpu.use_real_fields();
    gpu.use_real_fields();
    for (component c : {Ex, Ey, Ez}) {
      cpu.require_component(c);
      gpu.require_component(c);
    }
    cpu.init_backend();
    gpu.init_backend();
    compare_all_initialized_material_rows(cpu, gpu, 5e-12);
    master_printf("nvidia_timestep: material-geometry-grid-reducer-%d PASS\n", reducer);
  }
}

static void test_native_geometry_retry() {
  const nvidia::testing::failure_point failures[] = {
      nvidia::testing::failure_point::material_geometry_descriptor_mutation,
      nvidia::testing::failure_point::material_geometry_compact_mutation,
      nvidia::testing::failure_point::material_compact_allocate,
      nvidia::testing::failure_point::material_ir_upload,
      nvidia::testing::failure_point::material_geometry_bulk_launch,
      nvidia::testing::failure_point::material_geometry_analytic_launch,
      nvidia::testing::failure_point::material_geometry_patch_launch,
      nvidia::testing::failure_point::material_initialization_sync};
  for (size_t failure_index = 0;
       failure_index < sizeof(failures) / sizeof(failures[0]); ++failure_index) {
    const nvidia::testing::failure_point failure = failures[failure_index];
    structure s(vol2d(1.0, 1.0, 8.0), isotropic_eps, no_pml(), identity(), 1);
    install_geometry_block_material(s);
    execution_options options;
    options.backend = backend_kind::nvidia;
    fields f(&s, options);
    f.use_real_fields();
    for (component c : {Ex, Ey, Ez, Hx, Hy, Hz}) f.require_component(c);
    const nvidia::memory_accounting before = nvidia::current_memory_accounting();
    nvidia::testing::fail_next(failure);
    bool rejected = false;
    try { f.init_backend(); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
    const nvidia::memory_accounting after = nvidia::current_memory_accounting();
    require(rejected && f.backend_state == NULL && f.executable == NULL &&
                before.device_bytes_current == after.device_bytes_current &&
                before.pinned_bytes_current == after.pinned_bytes_current,
            "failed cold geometry initialization published or leaked a candidate");
    f.init_backend();
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(f.backend);
    require(backend && f.backend_state && f.executable &&
                backend->material_initialization_statistics_for_testing().valid,
            "cold geometry initialization failure was not retryable");
  }

  structure s(vol2d(1.0, 1.0, 8.0), isotropic_eps, no_pml(), identity(), 1);
  install_geometry_block_material(s);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields f(&s, options);
  f.use_real_fields();
  for (component c : {Ex, Ey, Ez, Hx, Hy, Hz}) f.require_component(c);
  f.init_backend();
  BackendState *const live_state = f.backend_state;
  Executable *const live_executable = f.executable;
  invalidate(f, MutationKind::material_values, "geometry initialization replacement retry");
  nvidia::testing::fail_next(nvidia::testing::failure_point::material_geometry_patch_launch);
  bool rejected = false;
  try { f.init_backend(); }
  catch (const std::exception &) { rejected = true; }
  nvidia::testing::clear_failure();
  require(rejected && f.backend_state == live_state && f.executable == live_executable &&
              !f.backend->is_poisoned(),
          "failed geometry replacement did not preserve the live epoch");
  f.init_backend();
  require(f.backend_state != live_state && f.executable != live_executable,
          "geometry replacement retry did not publish a new epoch");
  master_printf("nvidia_timestep: material-geometry rollback/retry PASS\n");
}

struct Pr54MaterialEpochSummary {
  BackendState *state;
  Executable *executable;
  StoragePlan *storage;
  InitializationPlan *initialization;
  StepPlan *step;
  uint32_t dirty;
  uint64_t prepared_classification_hash;
  MaterialRecipeDisposition global_route;
  MaterialRecipeDisposition local_route;
  uint64_t support_reasons;
  uint64_t recipe_signature;
  uint64_t classification_hash;
  bool local_presence;
  bool global_presence;
  bool presence_validated;
  bool policy_pending;
  MaterialFallbackStatistics fallback;
  NvidiaMaterialInitializationStatistics initialization_statistics;
  uint64_t warning_count;
  bool warning_emitted;

  explicit Pr54MaterialEpochSummary(fields &f)
      : state(f.backend_state), executable(f.executable), storage(f.storage_plan),
        initialization(f.initialization_plan), step(f.step_plans[0]), dirty(f.dirty_mask),
        prepared_classification_hash(f.prepared_classification_hash),
        global_route(state ? state->material_route : MaterialRecipeDisposition::device_native),
        local_route(state ? state->material_local_route : MaterialRecipeDisposition::device_native),
        support_reasons(state ? state->material_support_reasons : 0),
        recipe_signature(state ? state->material_recipe_signature : 0),
        classification_hash(state ? state->material_classification_hash : 0),
        local_presence(state && state->material_fallback_local_presence),
        global_presence(state && state->material_fallback_global_presence),
        presence_validated(state && state->material_fallback_presence_validated),
        policy_pending(state && state->material_fallback_policy_pending),
        fallback(state ? state->material_fallback_statistics : MaterialFallbackStatistics()),
        initialization_statistics(),
        warning_count(f.backend ? f.backend->material_fallback_warning_count() : 0),
        warning_emitted(f.backend && f.backend->material_fallback_warning_emitted()) {
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(f.backend);
    if (backend)
      initialization_statistics = backend->material_initialization_statistics_for_testing();
  }

  bool matches(fields &f) const {
    const Pr54MaterialEpochSummary other(f);
    return state == other.state && executable == other.executable && storage == other.storage &&
           initialization == other.initialization && step == other.step && dirty == other.dirty &&
           prepared_classification_hash == other.prepared_classification_hash &&
           global_route == other.global_route && local_route == other.local_route &&
           support_reasons == other.support_reasons &&
           recipe_signature == other.recipe_signature &&
           classification_hash == other.classification_hash &&
           local_presence == other.local_presence && global_presence == other.global_presence &&
           presence_validated == other.presence_validated &&
           policy_pending == other.policy_pending &&
           same_material_fallback_statistics(fallback, other.fallback) &&
           same_material_initialization_statistics(initialization_statistics,
                                                   other.initialization_statistics) &&
           warning_count == other.warning_count && warning_emitted == other.warning_emitted;
  }
};

static void install_pr54_tiled_material(structure &s, size_t &callback_calls) {
  using namespace meep_geom;
  std::shared_ptr<const OwnedMaterialCallback> owner(new OwnedMaterialCallback(
      UINT64_C(0x707235342d666169), UINT64_C(0x707235342d726f6c),
      owned_material_callback_tiled_capabilities,
      [&callback_calls](vector3 point, medium_struct &medium) {
        ++callback_calls;
        medium = medium_struct(2.125 + 0.03125 * point.x - 0.015625 * point.y);
      }));
  material_type material = make_owned_user_material_for_backend(owner, false);
  geometric_object_list empty = {0, NULL};
  set_materials_from_geometry(&s, empty, make_vector3(), false, 1e-5, 64, false, material);
  material_free(material);
}

static void test_pr54_post_callback_failure_matrix() {
  const nvidia::testing::failure_point failures[] = {
      nvidia::testing::failure_point::material_callback_dispatch,
      nvidia::testing::failure_point::material_tiled_upload,
      nvidia::testing::failure_point::material_initialization_sync,
      nvidia::testing::failure_point::material_classification_d2h,
      nvidia::testing::failure_point::material_classification_result_mutation,
      nvidia::testing::failure_point::material_finalize,
      nvidia::testing::failure_point::material_compile};
  const grid_volume gv = vol2d(1.5, 1.25, 12.0);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  options.strict = false;
  options.fallback = fallback_policy::warn;

  for (size_t failure_index = 0;
       failure_index < sizeof(failures) / sizeof(failures[0]); ++failure_index) {
    const nvidia::testing::failure_point failure = failures[failure_index];
    size_t callback_calls = 0;
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    install_pr54_tiled_material(s, callback_calls);
    fields f(&s, options);
    f.use_real_fields();
    f.require_component(Ez);
    const nvidia::memory_accounting before = nvidia::current_memory_accounting();
    nvidia::testing::fail_next(failure);
    bool rejected = false;
    try { f.init_backend(); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
    const nvidia::memory_accounting after = nvidia::current_memory_accounting();
    const bool clean = rejected && callback_calls > 0 && !f.backend_state && !f.executable &&
                       before.device_bytes_current == after.device_bytes_current &&
                       before.pinned_bytes_current == after.pinned_bytes_current &&
                       f.backend->material_fallback_warning_count() == 0;
    if (!clean)
      fprintf(stderr,
              "PR5.4 cold failure %zu rejected=%d callbacks=%zu state=%d executable=%d "
              "device=%zu/%zu pinned=%zu/%zu warnings=%llu\n",
              failure_index, int(rejected), callback_calls, int(f.backend_state != NULL),
              int(f.executable != NULL), before.device_bytes_current,
              after.device_bytes_current, before.pinned_bytes_current,
              after.pinned_bytes_current,
              (unsigned long long)f.backend->material_fallback_warning_count());
    require(clean,
            "cold post-callback material failure published or leaked a candidate");
    f.init_backend();
    require(f.backend_state && f.executable &&
                f.backend_state->material_route == MaterialRecipeDisposition::tiled_callback &&
                f.backend->material_fallback_warning_count() == 1,
            "cold post-callback material failure was not retryable");
  }

  for (nvidia::testing::failure_point failure : failures) {
    size_t callback_calls = 0;
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    install_pr54_tiled_material(s, callback_calls);
    fields f(&s, options);
    f.use_real_fields();
    f.require_component(Ez);
    f.init_backend();
    invalidate(f, MutationKind::material_values, "PR5.4 warm post-callback failure");
    const Pr54MaterialEpochSummary entry(f);
    const size_t callbacks_before = callback_calls;
    nvidia::testing::fail_next(failure);
    bool rejected = false;
    try { f.init_backend(); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
    require(rejected && callback_calls > callbacks_before && entry.matches(f) &&
                !f.backend->is_poisoned(),
            "warm post-callback material failure changed the committed epoch");
    BackendState *const previous_state = f.backend_state;
    f.init_backend();
    require(f.backend_state && f.backend_state != previous_state && f.executable &&
                f.backend_state->material_route == MaterialRecipeDisposition::tiled_callback &&
                f.backend->material_fallback_warning_count() == 1,
            "warm post-callback material failure was not retryable");
  }
  master_printf("nvidia_timestep: PR5.4 post-callback rollback/retry PASS\n");
}

static void test_native_geometry_tensor_preflight() {
  using namespace meep_geom;
  const auto singularize = [](std::vector<double> &values, size_t base, bool magnetic) {
    const size_t diagonal = base + (magnetic ? 9 : 0);
    const size_t offdiagonal = base + (magnetic ? 12 : 3);
    values[diagonal] = values[diagonal + 1] = values[diagonal + 2] = 1.0;
    for (size_t i = 0; i < 6; ++i) values[offdiagonal + i] = 0.0;
    values[offdiagonal] = 1.0;
  };
  const auto reject = [&](bool grid) {
    structure s(vol2d(1.0, 1.0, 8.0), isotropic_eps, no_pml(), identity(), 1);
    if (grid) {
      material_type material = make_geometry_grid_material(false, material_data::U_DEFAULT);
      geometric_object object =
          make_block(material, make_vector3(), make_vector3(1, 0, 0), make_vector3(0, 1, 0),
                     make_vector3(0, 0, 1), make_vector3(0.75, 0.75, 1.0));
      geometric_object_list geometry = {1, &object};
      set_materials_from_geometry(&s, geometry, make_vector3(), false, 1e-5, 128, false,
                                  vacuum);
      geometric_object_destroy(object);
      material_free(material);
    }
    else
      install_geometry_block_material(s);
    std::shared_ptr<MaterialIR> malformed(
        new MaterialIR(*static_cast<const MaterialIR *>(s.material_ir.get())));
    bool changed = false;
    for (MaterialIRMaterial &material : malformed->materials) {
      if (!grid && material.kind == material_data::MEDIUM && !material.parameters.empty() &&
          material.comparison_medium.size() >= 18) {
        singularize(material.parameters, 0, false);
        singularize(material.comparison_medium, 0, false);
        changed = true;
        break;
      }
      if (grid && material.kind == material_data::MATERIAL_GRID &&
          material.parameters.size() >= 21) {
        singularize(material.parameters, 3, false);
        changed = true;
        break;
      }
    }
    require(changed, "singular geometry preflight fixture found no target material");
    refresh_material_ir_signatures_for_testing(*malformed);
    s.material_ir = std::static_pointer_cast<const void>(malformed);
    execution_options options;
    options.backend = backend_kind::nvidia;
    fields f(&s, options);
    f.use_real_fields();
    for (component c : {Ex, Ey, Ez}) f.require_component(c);
    const nvidia::memory_accounting before = nvidia::current_memory_accounting();
    nvidia::testing::reset_transfer_accounting();
    bool rejected = false;
    try { f.init_backend(); }
    catch (const std::exception &error) {
      rejected = std::string(error.what()).find("tensor is singular") != std::string::npos;
    }
    const nvidia::memory_accounting after = nvidia::current_memory_accounting();
    const nvidia::testing::transfer_accounting transfers =
        nvidia::testing::current_transfer_accounting();
    require(rejected && f.backend_state == NULL && f.executable == NULL &&
                before.device_bytes_current == after.device_bytes_current &&
                before.pinned_bytes_current == after.pinned_bytes_current &&
                transfers.host_to_device_calls == 0 && transfers.device_to_host_calls == 0 &&
                transfers.device_to_device_calls == 0,
            grid ? "singular MaterialGrid tensor crossed preflight"
                 : "singular object medium tensor crossed preflight");
  };
  reject(false);
  reject(true);
  master_printf("nvidia_timestep: material-geometry tensor preflight PASS\n");
}

static void install_geometry_admission_stress_material(structure &s) {
  using namespace meep_geom;
  const int material_count = 18;
  std::vector<material_type> materials;
  std::vector<geometric_object> objects;
  materials.reserve(material_count);
  objects.reserve(material_count);
  for (int i = 0; i < material_count; ++i) {
    material_type medium = make_dielectric(2.0 + 0.07 * i);
    medium->medium.epsilon_diag =
        make_vector3(2.0 + 0.07 * i, 2.3 + 0.05 * i, 2.6 + 0.03 * i);
    medium->medium.mu_diag =
        make_vector3(1.1 + 0.01 * i, 1.2 + 0.01 * i, 1.3 + 0.01 * i);
    medium->medium.D_conductivity_diag =
        make_vector3(0.001 * (i + 1), 0.0015 * (i + 1), 0.002 * (i + 1));
    medium->medium.B_conductivity_diag =
        make_vector3(0.0007 * (i + 1), 0.0009 * (i + 1), 0.0011 * (i + 1));
    medium->medium.E_chi2_diag = make_vector3(0.01, -0.008, 0.006);
    medium->medium.E_chi3_diag = make_vector3(0.002, 0.003, -0.001);
    medium->medium.H_chi2_diag = make_vector3(-0.007, 0.005, 0.004);
    medium->medium.H_chi3_diag = make_vector3(0.001, -0.0015, 0.0025);
    materials.push_back(medium);
    const int x = i % 6, y = i / 6;
    objects.push_back(make_block(
        medium, make_vector3(-1.25 + 0.5 * x, -0.5 + 0.5 * y, 0.0),
        make_vector3(1, 0, 0), make_vector3(0, 1, 0), make_vector3(0, 0, 1),
        make_vector3(0.375, 0.375, 1.0)));
  }
  geometric_object_list geometry = {int(objects.size()), objects.data()};
  material_type_list extras;
  extras.items = materials.data();
  extras.num_items = int(materials.size());
  set_materials_from_geometry(&s, geometry, make_vector3(), false, 1e-5, 256, false,
                              vacuum, 0, extras);
  for (int i = material_count - 1; i >= 0; --i) geometric_object_destroy(objects[size_t(i)]);
  for (material_type material : materials) material_free(material);
}

static size_t parse_initialization_peak(const std::string &message) {
  const std::string marker = "NVIDIA initialization peak requires ";
  const size_t begin = message.find(marker);
  if (begin == std::string::npos) return 0;
  char *end = NULL;
  const unsigned long long value =
      std::strtoull(message.c_str() + begin + marker.size(), &end, 10);
  if (!end || end == message.c_str() + begin + marker.size() ||
      value > std::numeric_limits<size_t>::max())
    return 0;
  return size_t(value);
}

static void test_native_geometry_memory_admission() {
  structure s(vol2d(3.0, 2.0, 8.0), isotropic_eps, pml(0.2), identity(), 4);
  install_geometry_admission_stress_material(s);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields f(&s, options);
  f.use_real_fields();
  for (component c : {Ex, Ey, Ez, Hx, Hy, Hz}) f.require_component(c);
  const nvidia::memory_accounting before = nvidia::current_memory_accounting();
  nvidia::testing::reset_transfer_accounting();
  nvidia::testing::set_initialization_memory_budget_for_testing(1);
  bool rejected = false;
  size_t exact_peak = 0;
  try { f.init_backend(); }
  catch (const std::exception &error) {
    exact_peak = parse_initialization_peak(error.what());
    rejected = exact_peak != 0;
  }
  nvidia::testing::set_initialization_memory_budget_for_testing(
      std::numeric_limits<size_t>::max());
  const nvidia::memory_accounting after = nvidia::current_memory_accounting();
  const nvidia::testing::transfer_accounting transfers =
      nvidia::testing::current_transfer_accounting();
  require(rejected && f.backend_state == NULL && f.executable == NULL &&
              before.device_bytes_current == after.device_bytes_current &&
              before.pinned_bytes_current == after.pinned_bytes_current &&
              transfers.host_to_device_calls == 0 && transfers.device_to_host_calls == 0 &&
              transfers.device_to_device_calls == 0,
          "combined geometry memory admission allocated, transferred, or published");
  f.init_backend();
  require(f.backend_state && f.executable,
          "geometry memory-admission rejection was not retryable");
  const NvidiaBackend *backend = dynamic_cast<const NvidiaBackend *>(f.backend);
  const MaterialIR *ir = material_ir_for(f);
  require(backend && ir, "geometry admission retry lost its backend or owned IR");
  const size_t actual_compact =
      backend->material_initialization_statistics_for_testing()
          .compact_input_host_to_device_bytes;
  require(f.initialization_plan && f.initialization_plan->materials.size() == 1,
          "geometry admission retry lost its initialization recipe");
  const uint64_t logical_u64 =
      classify_material_support(f.initialization_plan->materials[0]).compact_input_bytes;
  require(logical_u64 <= uint64_t(std::numeric_limits<size_t>::max()) &&
              actual_compact > size_t(logical_u64),
          "geometry admission stress did not exceed the old logical IR estimate");
  const nvidia::memory_accounting retry_memory = nvidia::current_memory_accounting();
  require(retry_memory.pinned_bytes_current >= after.pinned_bytes_current &&
              exact_peak > retry_memory.pinned_bytes_current - after.pinned_bytes_current +
                               actual_compact,
          "geometry admission stress cannot reconstruct the old peak estimate");
  const size_t retry_pinned =
      retry_memory.pinned_bytes_current - after.pinned_bytes_current;
  const size_t old_peak = exact_peak - retry_pinned - actual_compact + size_t(logical_u64);
  require(old_peak < exact_peak - 1,
          "geometry admission stress left no threshold between old and exact peaks");
  const size_t threshold = old_peak + (exact_peak - old_peak) / 2;
  master_printf(
      "nvidia_timestep: material-geometry admission old=%zu exact=%zu threshold=%zu "
      "logical=%zu packed=%zu staging=%zu\n",
      old_peak, exact_peak, threshold, size_t(logical_u64), actual_compact, retry_pinned);

  structure threshold_structure(vol2d(3.0, 2.0, 8.0), isotropic_eps, pml(0.2), identity(), 4);
  install_geometry_admission_stress_material(threshold_structure);
  fields threshold_fields(&threshold_structure, options);
  threshold_fields.use_real_fields();
  for (component c : {Ex, Ey, Ez, Hx, Hy, Hz}) threshold_fields.require_component(c);
  const nvidia::memory_accounting threshold_before = nvidia::current_memory_accounting();
  nvidia::testing::reset_transfer_accounting();
  nvidia::testing::set_initialization_memory_budget_for_testing(threshold);
  rejected = false;
  try { threshold_fields.init_backend(); }
  catch (const std::exception &error) {
    rejected = parse_initialization_peak(error.what()) == exact_peak;
  }
  nvidia::testing::set_initialization_memory_budget_for_testing(
      std::numeric_limits<size_t>::max());
  const nvidia::memory_accounting threshold_after = nvidia::current_memory_accounting();
  const nvidia::testing::transfer_accounting threshold_transfers =
      nvidia::testing::current_transfer_accounting();
  require(rejected && threshold_fields.backend_state == NULL &&
              threshold_fields.executable == NULL &&
              threshold_before.device_bytes_current == threshold_after.device_bytes_current &&
              threshold_before.pinned_bytes_current == threshold_after.pinned_bytes_current &&
              threshold_transfers.host_to_device_calls == 0 &&
              threshold_transfers.device_to_host_calls == 0 &&
              threshold_transfers.device_to_device_calls == 0,
          "between-estimates geometry budget allocated, transferred, launched, or published");
  threshold_fields.init_backend();
  require(threshold_fields.backend_state && threshold_fields.executable,
          "between-estimates geometry budget rejection was not retryable");
  master_printf("nvidia_timestep: material-geometry memory admission PASS\n");
}

static void test_native_geometry_periodic() {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.0, 1.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  material_type cpu_medium = make_dielectric(4.25);
  material_type gpu_medium = make_dielectric(4.25);
  geometric_object cpu_object =
      make_sphere(cpu_medium, make_vector3(0.43, 0.0, 0.0), 0.31);
  geometric_object gpu_object =
      make_sphere(gpu_medium, make_vector3(0.43, 0.0, 0.0), 0.31);
  geometric_object_list cpu_geometry = {1, &cpu_object};
  geometric_object_list gpu_geometry = {1, &gpu_object};
  set_materials_from_geometry(&cpu_structure, cpu_geometry, make_vector3(), false, 1e-5, 128,
                              true, vacuum);
  set_materials_from_geometry(&gpu_structure, gpu_geometry, make_vector3(), false, 1e-5, 128,
                              true, vacuum);
  geometric_object_destroy(cpu_object);
  geometric_object_destroy(gpu_object);
  material_free(cpu_medium);
  material_free(gpu_medium);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  for (component c : {Ex, Ey, Ez}) {
    cpu.require_component(c);
    gpu.require_component(c);
  }
  cpu.init_backend();
  gpu.init_backend();
  const MaterialIR *ir = material_ir_for(gpu);
  require(ir && ir->ensure_periodicity && ir->images.size() > ir->objects.size() &&
              ir->active_images.size() > ir->objects.size(),
          "periodic geometry fixture did not retain ordered periodic images");
  for (size_t i = 1; i < ir->active_images.size(); ++i)
    require(ir->images[ir->active_images[i - 1]].ordinal <
                ir->images[ir->active_images[i]].ordinal,
            "periodic geometry active image order changed during packing");
  compare_all_initialized_material_rows(cpu, gpu, 5e-12);
  master_printf("nvidia_timestep: material-geometry-periodic PASS\n");
}

static void test_native_geometry_skew_lattice() {
  using namespace meep_geom;
  geom_initialize();
  dimensions = 3;
  geometry_center = make_vector3();
  ensure_periodicity = false;
  set_default_material(vacuum);
  geometry_lattice.size = make_vector3(3, 3, 3);
  geometry_lattice.basis_size = make_vector3(1.0, 1.0, 1.0);
  geometry_lattice.basis1 = make_vector3(1, 0, 0);
  geometry_lattice.basis2 = make_vector3(0.5, sqrt(3.0) * 0.5, 0);
  geometry_lattice.basis3 = make_vector3(0, 0, 1);
  geom_fix_lattice();
  const grid_volume gv = vol3d(3.0, 3.0, 3.0, 4.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  material_type cpu_medium = make_dielectric(5.5);
  material_type gpu_medium = make_dielectric(5.5);
  geometric_object cpu_object = make_sphere(cpu_medium, make_vector3(), 0.83);
  geometric_object gpu_object = make_sphere(gpu_medium, make_vector3(), 0.83);
  geometric_object_list cpu_geometry = {1, &cpu_object};
  geometric_object_list gpu_geometry = {1, &gpu_object};
  material_type_list no_extra;
  no_extra.num_items = 0;
  no_extra.items = NULL;
  geom_epsilon cpu_geps(cpu_geometry, no_extra, cpu_structure.gv.pad().surroundings());
  geom_epsilon gpu_geps(gpu_geometry, no_extra, gpu_structure.gv.pad().surroundings());
  set_materials_from_geom_epsilon(&cpu_structure, &cpu_geps, false, 1e-5, 128);
  set_materials_from_geom_epsilon(&gpu_structure, &gpu_geps, false, 1e-5, 128);
  geometric_object_destroy(cpu_object);
  geometric_object_destroy(gpu_object);
  material_free(cpu_medium);
  material_free(gpu_medium);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  for (component c : {Ex, Ey, Ez}) {
    cpu.require_component(c);
    gpu.require_component(c);
  }
  cpu.init_backend();
  gpu.init_backend();
  const MaterialIR *ir = material_ir_for(gpu);
  require(ir && ir->lattice_metric[3] != 0.0,
          "skew geometry fixture lost its nonorthogonal metric");
  compare_all_initialized_material_rows(cpu, gpu, 5e-12);
  geom_initialize();
  master_printf("nvidia_timestep: material-geometry-skew-lattice PASS\n");
}

static void test_native_geometry_dimensions() {
  using namespace meep_geom;
  struct DimensionCase {
    const char *name;
    grid_volume volume;
  };
  const DimensionCase cases[] = {{"d1", vol1d(1.5, 8.0)},
                                 {"cyl", volcyl(1.5, 1.25, 6.0)}};
  for (const DimensionCase &test : cases) {
    structure cpu_structure(test.volume, isotropic_eps, no_pml(), identity(), 1);
    structure gpu_structure(test.volume, isotropic_eps, no_pml(), identity(), 1);
    material_type cpu_medium = make_dielectric(3.75);
    material_type gpu_medium = make_dielectric(3.75);
    geometric_object cpu_object = make_sphere(cpu_medium, make_vector3(), 0.42);
    geometric_object gpu_object = make_sphere(gpu_medium, make_vector3(), 0.42);
    geometric_object_list cpu_geometry = {1, &cpu_object};
    geometric_object_list gpu_geometry = {1, &gpu_object};
    set_materials_from_geometry(&cpu_structure, cpu_geometry, make_vector3(), false, 1e-5, 128,
                                false, vacuum);
    set_materials_from_geometry(&gpu_structure, gpu_geometry, make_vector3(), false, 1e-5, 128,
                                false, vacuum);
    geometric_object_destroy(cpu_object);
    geometric_object_destroy(gpu_object);
    material_free(cpu_medium);
    material_free(gpu_medium);
    fields cpu(&cpu_structure);
    execution_options options;
    options.backend = backend_kind::nvidia;
    fields gpu(&gpu_structure, options);
    cpu.use_real_fields();
    gpu.use_real_fields();
    FOR_E_AND_H(c) if (test.volume.has_field(c)) {
        cpu.require_component(c);
        gpu.require_component(c);
      }
    cpu.init_backend();
    gpu.init_backend();
    const MaterialIR *ir = material_ir_for(gpu);
    require(ir && ir->dimensions == int(test.volume.dim),
            "geometry dimension fixture captured the wrong dimensional enum");
    compare_all_initialized_material_rows(cpu, gpu, 5e-12);
    master_printf("nvidia_timestep: material-geometry-%s PASS\n", test.name);
  }
}

static void test_native_geometry_field_growth() {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.0, 1.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  material_type cpu_medium = make_dielectric(4.5);
  material_type gpu_medium = make_dielectric(4.5);
  geometric_object cpu_object =
      make_block(cpu_medium, make_vector3(), make_vector3(1, 0, 0),
                 make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                 make_vector3(0.63, 0.61, 1.0));
  geometric_object gpu_object =
      make_block(gpu_medium, make_vector3(), make_vector3(1, 0, 0),
                 make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                 make_vector3(0.63, 0.61, 1.0));
  geometric_object_list cpu_geometry = {1, &cpu_object};
  geometric_object_list gpu_geometry = {1, &gpu_object};
  set_materials_from_geometry(&cpu_structure, cpu_geometry, make_vector3(), false, 1e-5, 128,
                              false, vacuum);
  set_materials_from_geometry(&gpu_structure, gpu_geometry, make_vector3(), false, 1e-5, 128,
                              false, vacuum);
  geometric_object_destroy(cpu_object);
  geometric_object_destroy(gpu_object);
  material_free(cpu_medium);
  material_free(gpu_medium);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  cpu.require_component(Ez);
  gpu.require_component(Ez);
  cpu.init_backend();
  gpu.init_backend();
  const size_t first_catalog = gpu.array_catalog->size();
  const uint64_t first_layout_generation =
      gpu.mutation_generation[size_t(MutationKind::field_layout)];
  cpu.require_component(Ex);
  gpu.require_component(Ex);
  cpu.advance(1);
  gpu.advance(1);
  const NvidiaBackend *backend = dynamic_cast<const NvidiaBackend *>(gpu.backend);
  const NvidiaMaterialInitializationStatistics statistics =
      backend ? backend->material_initialization_statistics_for_testing()
              : NvidiaMaterialInitializationStatistics();
  require(backend && statistics.valid && gpu.array_catalog->size() > first_catalog &&
              gpu.mutation_generation[size_t(MutationKind::field_layout)] >
                  first_layout_generation &&
              statistics.geometry_bulk_kernel_launches > 0,
          "geometry field growth did not rebind and replace the resident epoch");
  compare_all_initialized_material_rows(cpu, gpu, 5e-12);
  master_printf("nvidia_timestep: material-geometry-field-growth PASS\n");
}

static void test_native_geometry_mpi() {
  const int ranks = count_processors();
  require(ranks == 2 || ranks == 4,
          "geometry MPI validation requires exactly two or four ranks");
  const grid_volume gv = vol2d(2.0, 1.0, 8.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), ranks);
  install_geometry_block_material(s);
  const MaterialIR *ir = static_cast<const MaterialIR *>(s.material_ir.get());
  const bool local_patch_empty = ir && ir->hybrid_patches.empty();
  const bool local_patch_nonempty = ir && !ir->hybrid_patches.empty();
  const size_t signature_low = ir ? uint32_t(ir->signature) : 0;
  const size_t signature_high = ir ? uint32_t(ir->signature >> 32) : 0;
  const bool same_semantic_signature =
      ir && sum_to_all(signature_low) == signature_low * size_t(ranks) &&
      sum_to_all(signature_high) == signature_high * size_t(ranks);
  require(and_to_all(same_semantic_signature) && or_to_all(local_patch_empty) &&
              or_to_all(local_patch_nonempty),
          "geometry MPI split did not preserve a global hybrid route with asymmetric patches");
  master_printf("nvidia_timestep: material-geometry collective partition np%d PASS\n", ranks);
}

static void test_native_geometry_cuda_mpi_initialization() {
  const int ranks = count_processors();
  require(ranks == 2,
          "CUDA geometry initialization validation requires exactly two ranks");
  backend_set_initialization_only_for_testing(true);
  const grid_volume gv = vol2d(2.0, 1.0, 8.0);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;

  {
    structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), ranks);
    install_geometry_block_material(gpu_structure);
    fields gpu(&gpu_structure, options);
    gpu.use_real_fields();
    for (component c : {Ex, Ey, Ez, Hx, Hy, Hz}) gpu.require_component(c);
    nvidia::testing::reset_transfer_accounting();
    nvidia::testing::reset_material_transfer_accounting();
    gpu.init_backend();
    const NvidiaBackend *backend = dynamic_cast<const NvidiaBackend *>(gpu.backend);
    const nvidia::testing::transfer_accounting transfers =
        nvidia::testing::current_transfer_accounting();
    const nvidia::testing::material_transfer_accounting material =
        nvidia::testing::current_material_transfer_accounting();
    require(backend && backend->device_ordinal_for_testing() == my_global_rank() &&
                !gpu.backend_state && !gpu.executable && material.compact_calls != 0 &&
                material.dense_output_calls == 0 && transfers.host_to_device_calls != 0,
            "CUDA MPI geometry initialization did not map one rank per GPU");
  }

  {
    structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
    install_geometry_block_material(gpu_structure);
    fields gpu(&gpu_structure, options);
    gpu.use_real_fields();
    for (component c : {Ex, Ey, Ez}) gpu.require_component(c);
    nvidia::testing::reset_material_transfer_accounting();
    gpu.init_backend();
    const NvidiaBackend *backend = dynamic_cast<const NvidiaBackend *>(gpu.backend);
    const nvidia::testing::material_transfer_accounting material =
        nvidia::testing::current_material_transfer_accounting();
    const bool local_work = material.compact_calls != 0;
    require(backend && !gpu.backend_state && !gpu.executable &&
                or_to_all(local_work) && !and_to_all(local_work),
            "CUDA MPI geometry initialization did not preserve an idle rank");
  }

  {
    using namespace meep_geom;
    std::shared_ptr<const OwnedMaterialCallback> owner(new OwnedMaterialCallback(
        UINT64_C(0x707235342d6d7069), UINT64_C(0x707235342d6d7073),
        owned_material_callback_tiled_capabilities,
        [](vector3 point, medium_struct &medium) {
          medium = medium_struct(2.125 + 0.03125 * point.x);
        }));
    material_type callback = make_owned_user_material_for_backend(owner, false);
    structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
    geometric_object_list empty = {0, NULL};
    set_materials_from_geometry(&gpu_structure, empty, make_vector3(), false, 1e-5, 64,
                                false, callback);
    material_free(callback);
    fields gpu(&gpu_structure, options);
    gpu.use_real_fields();
    gpu.require_component(Ez);
    nvidia::testing::reset_material_transfer_accounting();
    gpu.init_backend();
    const nvidia::testing::material_transfer_accounting material =
        nvidia::testing::current_material_transfer_accounting();
    const bool tiled_work = material.tiled_output_calls != 0;
    require(sum_to_all(int(tiled_work)) == 1 && material.compact_calls == 0 &&
                material.dense_output_calls == 0 &&
                !gpu.backend_state && !gpu.executable,
            "CUDA MPI tiled/native owner-idle route or classification diverged");
  }

  {
    structure failure_structure(gv, isotropic_eps, no_pml(), identity(), ranks);
    install_geometry_block_material(failure_structure);
    fields failure(&failure_structure, options);
    failure.use_real_fields();
    for (component c : {Ex, Ey, Ez}) failure.require_component(c);
    if (my_rank() == 0)
      nvidia::testing::fail_next(
          nvidia::testing::failure_point::material_geometry_bulk_launch);
    bool rejected = false;
    try { failure.init_backend(); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
    require(and_to_all(rejected) && !failure.backend_state && !failure.executable,
            "rank-asymmetric CUDA geometry failure published a partial MPI epoch");
    failure.init_backend();
    require(!failure.backend_state && !failure.executable &&
                nvidia::testing::current_material_transfer_accounting().compact_calls != 0,
            "rank-asymmetric CUDA geometry failure was not retryable");
  }

  {
    const nvidia::testing::failure_point failures[] = {
        nvidia::testing::failure_point::material_initialization_sync,
        nvidia::testing::failure_point::material_classification_d2h,
        nvidia::testing::failure_point::material_classification_result_mutation,
        nvidia::testing::failure_point::material_finalize};
    for (int target = 0; target < ranks; ++target)
      for (size_t failure_index = 0;
           failure_index < sizeof(failures) / sizeof(failures[0]); ++failure_index) {
        const nvidia::testing::failure_point point = failures[failure_index];
        structure failure_structure(gv, isotropic_eps, no_pml(), identity(), ranks);
        install_geometry_block_material(failure_structure);
        fields failure(&failure_structure, options);
        failure.use_real_fields();
        for (component c : {Ex, Ey, Ez}) failure.require_component(c);
        if (my_rank() == target) nvidia::testing::fail_next(point);
        bool rejected = false;
        try { failure.init_backend(); }
        catch (const std::exception &) { rejected = true; }
        nvidia::testing::clear_failure();
        const bool clean = and_to_all(rejected) && !failure.backend_state &&
                           !failure.executable &&
                           failure.backend->material_fallback_warning_count() == 0;
        if (!clean)
          fprintf(stderr,
                  "rank-asymmetric CUDA failure %zu target=%d local-rank=%d rejected=%d "
                  "state=%d executable=%d warnings=%llu\n",
                  failure_index, target, my_rank(), int(rejected),
                  int(failure.backend_state != NULL), int(failure.executable != NULL),
                  (unsigned long long)failure.backend->material_fallback_warning_count());
        require(clean,
                "rank-asymmetric CUDA classification/finalize/compile failure published state");
        failure.init_backend();
        require(!failure.backend_state && !failure.executable,
                "rank-asymmetric CUDA classification/finalize/compile failure was not retryable");
      }
  }

  {
    const nvidia::testing::failure_point failures[] = {
        nvidia::testing::failure_point::material_callback_dispatch,
        nvidia::testing::failure_point::material_tiled_upload};
    for (int target = 0; target < ranks; ++target)
      for (nvidia::testing::failure_point point : failures) {
        size_t callback_calls = 0;
        structure failure_structure(gv, isotropic_eps, no_pml(), identity(), ranks);
        install_pr54_tiled_material(failure_structure, callback_calls);
        fields failure(&failure_structure, options);
        failure.use_real_fields();
        failure.require_component(Ez);
        if (my_rank() == target) nvidia::testing::fail_next(point);
        bool rejected = false;
        try { failure.init_backend(); }
        catch (const std::exception &) { rejected = true; }
        nvidia::testing::clear_failure();
        require(and_to_all(rejected) && callback_calls > 0 && !failure.backend_state &&
                    !failure.executable &&
                    failure.backend->material_fallback_warning_count() == 0,
                "rank-asymmetric CUDA tiled post-callback failure published state");
        failure.init_backend();
        require(!failure.backend_state && !failure.executable,
                "rank-asymmetric CUDA tiled post-callback failure was not retryable");
      }
  }

  divide_parallel_processes(ranks);
  require(count_processors() == 1 && my_rank() == 0,
          "CUDA geometry split communicator did not contain one local rank");
  {
    structure split_structure(gv, isotropic_eps, no_pml(), identity(), 1);
    install_geometry_block_material(split_structure);
    execution_options split_options = options;
    split_options.device_id = my_global_rank();
    fields split(&split_structure, split_options);
    split.use_real_fields();
    for (component c : {Ex, Ey, Ez}) split.require_component(c);
    split.init_backend();
    const NvidiaBackend *backend = dynamic_cast<const NvidiaBackend *>(split.backend);
    require(backend && backend->device_ordinal_for_testing() == my_global_rank() &&
                !split.backend_state && !split.executable,
            "CUDA geometry split communicator selected the wrong GPU or failed init");
  }
  end_divide_parallel();
  backend_set_initialization_only_for_testing(false);
  master_printf("nvidia_timestep: material-geometry CUDA MPI init/rollback PASS\n");
}

static void test_material_copy_after_compile_rejected() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  linear_anisotropic_material current_material(false), target_material(true);
  structure current(gv, current_material, no_pml(), identity(), 1);
  structure target(gv, target_material, no_pml(), identity(), 1);
  target.set_conductivity(Dz, phase_target_conductivity);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&current, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  gpu.phase_in_material(&target, 2.0 * gpu.dt);
  gpu.init_backend();
  gpu.synchronize_magnetic_fields();
  gpu.restore_magnetic_fields();
  const int countdown = gpu.phasein_time;
  structure_chunk *const current_chunk = gpu.chunks[0]->s;
  const realnum current_value = current_chunk->conductivity[Dz][Z][0];
  {
    fields copy(gpu);
    bool rejected = false;
    try {
      gpu.advance(1);
    }
    catch (const std::logic_error &error) {
      rejected = std::string(error.what()).find("current storage") != std::string::npos;
    }
    require(rejected, "NVIDIA material phase accepted shared current storage after compile");
    require(gpu.phasein_time == countdown && gpu.chunks[0]->s == current_chunk &&
                gpu.chunks[0]->s->conductivity[Dz][Z][0] == current_value,
            "rejected NVIDIA material phase changed countdown or current coefficients");
  }
  gpu.advance(1);
  require(gpu.phasein_time == countdown - 1,
          "NVIDIA material phase did not recover after the copied fields was destroyed");
  master_printf("nvidia_timestep: material copy-after-compile rejection PASS\n");
}

static void test_material_cpu_setup_to_nvidia() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  linear_anisotropic_material current_material(false), target_material(true);
  structure reference_structure(gv, current_material, no_pml(), identity(), 1);
  structure migrating_structure(gv, current_material, no_pml(), identity(), 1);
  structure reference_target(gv, target_material, no_pml(), identity(), 1);
  structure migrating_target(gv, target_material, no_pml(), identity(), 1);
  reference_target.set_conductivity(Dz, phase_target_conductivity);
  migrating_target.set_conductivity(Dz, phase_target_conductivity);
  fields reference(&reference_structure);
  fields migrating(&migrating_structure);
  reference.use_real_fields();
  migrating.use_real_fields();
  gaussian_src_time source(0.31, 0.12);
  reference.add_point_source(Ez, source, vec(0.11, 0.13));
  migrating.add_point_source(Ez, source, vec(0.11, 0.13));
  require(reference.phase_in_material(&reference_target, 3.0 * reference.dt) == 3 &&
              migrating.phase_in_material(&migrating_target, 3.0 * migrating.dt) == 3,
          "CPU material setup did not retain its countdown before NVIDIA selection");
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  migrating.select_backend(options);
  migrating.init_backend();
  for (int step = 0; step < 4; ++step) {
    reference.advance(1);
    migrating.advance(1);
    compare_fields(reference, migrating,
                   sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-13);
    compare_material_rows(reference, migrating,
                          sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-13);
  }
  require_material_targets_host_only(migrating);
  master_printf("nvidia_timestep: material CPU-setup-to-NVIDIA PASS\n");
}

static void test_material_recipe_owned_upload(precision_policy_kind precision) {
  const grid_volume gv = vol2d(2.4, 2.0, 8.0);
  structure s(gv, isotropic_eps, pml(0.35, X), identity(), 2);
  s.set_conductivity(Dz, phase_target_conductivity);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision;
  fields f(&s, options);
  f.use_real_fields();
  f.require_component(Ez);
  nvidia::testing::reset_transfer_accounting();
  f.init_backend();
  require(f.initialization_plan && f.initialization_plan->materials.size() == 1,
          "NVIDIA material recipe fixture has no frozen recipe");
  const MaterialRecipe frozen = f.initialization_plan->materials[0];
  validate_material_recipe(frozen);
  require(!frozen.rows().empty(), "NVIDIA material recipe fixture has no material rows");
  StoragePlan expected_upload = *f.storage_plan;
  mark_material_storage_provisional(frozen, expected_upload);
  apply_precision_policy(expected_upload, policy_for(f.options.precision));
  size_t expected_calls = 0, expected_bytes = 0;
  for (const ArraySpec &spec : expected_upload.arrays)
    if (!is_valid(spec.alias_of)) {
      ++expected_calls;
      const size_t bytes = storage_bytes(spec);
      require(bytes <= std::numeric_limits<size_t>::max() - expected_bytes,
              "NVIDIA material upload byte expectation overflowed");
      expected_bytes += bytes;
    }
  const nvidia::testing::transfer_accounting initial_transfers =
      nvidia::testing::current_transfer_accounting();
  /* Cold state construction may upload backend-private coordinate metadata in
     addition to the initialization plan. The direct replay below isolates and
     checks the recipe's exact call/byte count. */
  require(initial_transfers.host_to_device_calls >= expected_calls &&
              initial_transfers.host_to_device_bytes >= expected_bytes &&
              initial_transfers.device_to_host_calls == 0 &&
              initial_transfers.device_to_host_bytes == 0,
          "NVIDIA material initialization omitted recipe storage or downloaded host data");

  struct MutatedRow {
    ArrayId id;
    std::vector<unsigned char> original;
  };
  std::vector<MutatedRow> mutated;
  for (const MaterialRecipeRow &row : frozen.rows()) {
    const ArrayId id = f.array_catalog->find(row.key);
    require(is_valid(id), "NVIDIA frozen material row has no catalog identity");
    unsigned char *live =
        static_cast<unsigned char *>(f.array_catalog->resolve_untyped(id));
    require(live && !row.values.empty(), "NVIDIA frozen material row has no owned payload");
    MutatedRow saved{id, std::vector<unsigned char>(live, live + row.values.size())};
    mutated.push_back(saved);
    live[0] ^= 0x5a;
  }

  /* Direct reinitialization is intentionally the final operation on this
     state. It proves every native material/PML row comes from the immutable
     recipe even though the compatibility catalog was changed afterward. */
  nvidia::testing::reset_transfer_accounting();
  f.backend->initialize(*f.initialization_plan, *f.backend_state);
  const nvidia::testing::transfer_accounting replay_transfers =
      nvidia::testing::current_transfer_accounting();
  require(replay_transfers.host_to_device_calls == expected_calls &&
              replay_transfers.host_to_device_bytes == expected_bytes &&
              replay_transfers.device_to_host_calls == 0 &&
              replay_transfers.device_to_host_bytes == 0,
          "NVIDIA frozen material replay transfer accounting differs from exact storage bytes");
  for (size_t i = 0; i < frozen.rows().size(); ++i) {
    const MaterialRecipeRow &row = frozen.rows()[i];
    std::vector<unsigned char> observed(row.values.size());
    f.backend->read(ArrayRef{mutated[i].id, 0, row.elements}, observed.data(), observed.size());
    std::vector<unsigned char> expected = row.values;
    if (precision != precision_policy_kind::native)
      for (size_t element = 0; element < row.elements; ++element) {
        realnum value = 0;
        memcpy(&value, expected.data() + element * sizeof(realnum), sizeof(value));
        value = realnum(float(value));
        memcpy(expected.data() + element * sizeof(realnum), &value, sizeof(value));
      }
    require(observed == expected,
            "NVIDIA material initialization borrowed a mutated CPU coefficient row");
    memcpy(f.array_catalog->resolve_untyped(mutated[i].id), mutated[i].original.data(),
           mutated[i].original.size());
  }
  master_printf("nvidia_timestep: material frozen-recipe upload PASS\n");
}

static void test_material_value_refresh_preserves_host_edit(precision_policy_kind precision) {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, isotropic_eps, pml(0.25, X), identity(), 2);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision;
  fields f(&s, options);
  f.use_real_fields();
  f.require_component(Ez);
  f.init_backend();
  f.advance(1);

  ArrayId target = invalid_array();
  realnum *host = NULL;
  StorageKey key;
  for (size_t i = 0; i < f.storage_plan->arrays.size(); ++i) {
    const ArraySpec &spec = f.storage_plan->arrays[i];
    if (spec.role != array_role::material || is_valid(spec.alias_of) || !spec.elements) continue;
    target = spec.id;
    key = f.storage_plan->keys[i];
    host = f.array_catalog->resolve<realnum>(target);
    if (host) break;
  }
  require(is_valid(target) && host, "NVIDIA material refresh fixture has no mutable host row");
  const realnum sentinel = host[0] + realnum(0.1875);
  host[0] = sentinel;
  invalidate(f, MutationKind::material_values, "NVIDIA material host-value sentinel");
  f.init_backend();

  const MaterialRecipe &recipe = f.initialization_plan->materials[0];
  bool recipe_saw_sentinel = false;
  for (const MaterialRecipeRow &row : recipe.rows())
    if (row.key == key && row.values.size() >= sizeof(realnum)) {
      realnum captured = 0;
      memcpy(&captured, row.values.data(), sizeof(captured));
      recipe_saw_sentinel = captured == sentinel;
    }
  std::vector<realnum> observed(1);
  f.backend->read(ArrayRef{target, 0, 1}, observed.data(), sizeof(realnum));
  const realnum device_sentinel =
      precision == precision_policy_kind::native ? sentinel : realnum(float(sentinel));
  require(recipe_saw_sentinel && observed[0] == device_sentinel,
          "NVIDIA authority migration overwrote a new host material coefficient");
  master_printf("nvidia_timestep: material host-value refresh PASS\n");
}

static void test_material_collective_preflight() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  linear_anisotropic_material current_material(false), target_material(true);
  structure current(gv, current_material, no_pml(), identity(), 1);
  structure target(gv, target_material, no_pml(), identity(), 1);
  target.set_conductivity(Dz, phase_target_conductivity);
  fields f(&current);
  f.use_real_fields();
  f.require_component(Ez);
  std::unique_ptr<PreparedMaterialPhaseStorage> prepared =
      prepare_material_phase_storage(f, target);
  prepared->commit();
  f.phase_in_material(&target, 3.0 * f.dt);
  const uint64_t signature = compute_material_phase_target_signature(f);
  const int countdown = f.phasein_time;
  realnum *current_row = NULL;
  for (int i = 0; i < f.num_chunks && !current_row; ++i)
    if (f.chunks[i]->is_mine()) current_row = f.chunks[i]->s->chi1inv[Ex][X];
  const realnum current_value = current_row ? current_row[0] : realnum(0);

  if (my_rank() == 0) {
    structure_chunk &changed = *target.chunks[0];
    changed.trivial_chi1inv[Ex][X] = !changed.trivial_chi1inv[Ex][X];
  }
  bool rejected = false;
  try {
    nvidia::validate_material_phase_state(f, signature);
  }
  catch (const std::logic_error &) {
    rejected = true;
  }
  require(and_to_all(rejected),
          "rank-asymmetric material target mutation was not rejected collectively");
  require(f.phasein_time == countdown && (!current_row || current_row[0] == current_value),
          "material target rejection changed countdown or current coefficients");
  if (my_rank() == 0) {
    structure_chunk &changed = *target.chunks[0];
    changed.trivial_chi1inv[Ex][X] = !changed.trivial_chi1inv[Ex][X];
  }
  nvidia::validate_material_phase_state(f, signature);

  if (my_rank() == 0) ++f.chunks[0]->s->refcount;
  rejected = false;
  try {
    nvidia::validate_material_phase_state(f, signature);
  }
  catch (const std::logic_error &) {
    rejected = true;
  }
  require(and_to_all(rejected),
          "rank-asymmetric shared material storage was not rejected collectively");
  if (my_rank() == 0) --f.chunks[0]->s->refcount;
  nvidia::validate_material_phase_state(f, signature);

  if (my_rank() == 0) f.phasein_time = countdown - 1;
  rejected = false;
  try {
    nvidia::validate_material_phase_state(f, signature);
  }
  catch (const std::logic_error &) {
    rejected = true;
  }
  require(and_to_all(rejected),
          "rank-asymmetric positive material countdown was not rejected collectively");
  require((my_rank() == 0 ? f.phasein_time == countdown - 1 : f.phasein_time == countdown) &&
              (!current_row || current_row[0] == current_value),
          "positive-countdown rejection changed countdown or current coefficients");
  if (my_rank() == 0) f.phasein_time = countdown;
  nvidia::validate_material_phase_state(f, signature);

  if (my_rank() == 0) f.phasein_time = 0;
  rejected = false;
  try {
    nvidia::validate_material_phase_state(f, signature);
  }
  catch (const std::logic_error &) {
    rejected = true;
  }
  require(and_to_all(rejected),
          "rank-asymmetric material countdown was not rejected collectively");
  if (my_rank() == 0) f.phasein_time = countdown;
  nvidia::validate_material_phase_state(f, signature);
  master_printf("nvidia_timestep: material collective preflight PASS\n");
}

static void require_advance_rejected(fields &f, const char *expected) {
  bool rejected = false;
  try {
    f.advance(1);
  }
  catch (const std::exception &error) {
    rejected = std::string(error.what()).find(expected) != std::string::npos;
  }
  require(rejected, "NVIDIA unsupported configuration was not rejected as expected");
}

static void expect_compile_rejected(fields &gpu, StepPlan plan, const char *expected) {
  plan.signature = compute_step_plan_signature(plan);
  Executable *unexpected = NULL;
  bool rejected = false;
  try {
    unexpected = gpu.backend->compile(plan, *gpu.backend_state);
  }
  catch (const std::runtime_error &error) {
    rejected = std::string(error.what()).find(expected) != std::string::npos;
    if (!rejected)
      fprintf(stderr, "unexpected compile rejection: %s (wanted: %s)\n", error.what(),
              expected);
  }
  delete unexpected;
  require(rejected, "malformed descriptor was not rejected as expected");
}

static void expect_stale_compile_rejected(fields &gpu, const StepPlan &plan) {
  Executable *unexpected = NULL;
  bool rejected = false;
  try {
    unexpected = gpu.backend->compile(plan, *gpu.backend_state);
  }
  catch (const std::runtime_error &error) {
    rejected = std::string(error.what()).find("stale StepPlan signature") != std::string::npos;
  }
  delete unexpected;
  require(rejected, "stale legacy flux plan signature was accepted");
}

static void test_legacy_flux_compile_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::mixed;
  fields gpu(&s, options);
  const vec bloch(0.13, -0.09);
  gpu.use_bloch(bloch);
  const component components[] = {Ex, Ey, Ez, Hx, Hy, Hz};
  for (size_t i = 0; i < sizeof(components) / sizeof(components[0]); ++i)
    gpu.require_component(components[i]);
  gpu.add_flux_vol(Z, volume(vec(-0.75, 0.0), vec(0.75, 0.0)));
  gpu.init_backend();
  gpu.advance(1);
  const StepPlan baseline = build_step_plan(gpu, StepProgram::ordinary);
  require(baseline.legacy_flux_updates.size() == 1 &&
              baseline.legacy_flux_terms.size() >= 2,
          "legacy flux rejection fixture lacks canonical descriptors");
  size_t half_marker = baseline.operations.size(), final_marker = baseline.operations.size();
  for (size_t i = 0; i < baseline.operations.size(); ++i) {
    if (baseline.operations[i].kind == OpKind::update_flux_half) half_marker = i;
    if (baseline.operations[i].kind == OpKind::update_flux) final_marker = i;
  }
  require(half_marker < baseline.operations.size() && final_marker < baseline.operations.size() &&
              !baseline.operations[half_marker].accesses.empty() &&
              !baseline.operations[final_marker].accesses.empty(),
          "legacy flux rejection fixture lacks marker accesses");

  StepPlan malformed = baseline;
  malformed.legacy_flux_terms[0].sign = -malformed.legacy_flux_terms[0].sign;
  expect_stale_compile_rejected(gpu, malformed);
  expect_compile_rejected(gpu, malformed, "legacy flux descriptors are non-canonical");

  malformed = baseline;
  ++malformed.legacy_flux_updates[0].flux_ordinal;
  expect_compile_rejected(gpu, malformed, "legacy flux descriptors are non-canonical");
  malformed = baseline;
  ++malformed.legacy_flux_updates[0].term_index;
  expect_compile_rejected(gpu, malformed, "legacy flux descriptors are non-canonical");
  malformed = baseline;
  malformed.legacy_flux_updates[0].term_count =
      uint32_t(malformed.legacy_flux_terms.size() + 1);
  expect_compile_rejected(gpu, malformed, "legacy flux descriptors are non-canonical");
  malformed = baseline;
  ++malformed.legacy_flux_terms[0].term_ordinal;
  expect_compile_rejected(gpu, malformed, "legacy flux descriptors are non-canonical");
  malformed = baseline;
  ++malformed.legacy_flux_terms[0].e_real.value;
  expect_compile_rejected(gpu, malformed, "legacy flux descriptors are non-canonical");
  malformed = baseline;
  malformed.legacy_flux_terms[0].e_offsets[0] =
      std::numeric_limits<ptrdiff_t>::max();
  expect_compile_rejected(gpu, malformed, "legacy flux descriptors are non-canonical");
  malformed = baseline;
  malformed.legacy_flux_terms[0].phase_real =
      std::numeric_limits<double>::quiet_NaN();
  expect_compile_rejected(gpu, malformed, "legacy flux descriptors are non-canonical");
  malformed = baseline;
  malformed.legacy_flux_terms[0].boundary_weights[0][0] =
      std::numeric_limits<double>::infinity();
  expect_compile_rejected(gpu, malformed, "legacy flux descriptors are non-canonical");

  malformed = baseline;
  ++malformed.operations[half_marker].legacy_flux_count;
  expect_compile_rejected(gpu, malformed, "legacy flux marker span is incomplete");
  malformed = baseline;
  malformed.operations[final_marker].accesses.pop_back();
  expect_compile_rejected(gpu, malformed, "legacy flux marker identity or accesses are stale");
  malformed = baseline;
  ++malformed.operations[half_marker].accesses[0].array.offset;
  expect_compile_rejected(gpu, malformed, "legacy flux marker identity or accesses are stale");

  Executable *valid = gpu.backend->compile(baseline, *gpu.backend_state);
  require(valid != NULL, "valid legacy flux plan did not compile after rejection checks");
  delete valid;
  master_printf("nvidia_timestep: flux compile rejections PASS\n");
}

static void test_material_compile_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  linear_anisotropic_material current_material(false), target_material(true);
  structure current(gv, current_material, pml(0.4), identity(), 1);
  structure target(gv, target_material, pml(0.4), identity(), 1);
  target.set_conductivity(Dz, phase_target_conductivity);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&current, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  gpu.phase_in_material(&target, 3.0 * gpu.dt);
  gpu.init_backend();

  const StepPlan baseline = build_step_plan(gpu, StepProgram::ordinary);
  require(!baseline.material_refresh_arrays.empty(),
          "material rejection fixture has no refresh descriptors");
  size_t phase_index = baseline.operations.size();
  size_t coefficient_index = baseline.operations.size();
  for (size_t i = 0; i < baseline.operations.size(); ++i) {
    if (baseline.operations[i].kind == OpKind::phase_material) phase_index = i;
    if (baseline.operations[i].kind == OpKind::update_material_coefficients)
      coefficient_index = i;
  }
  require(phase_index < baseline.operations.size() &&
              coefficient_index < baseline.operations.size(),
          "material rejection fixture has no refresh operations");

  StepPlan malformed = baseline;
  malformed.material_refresh_arrays[0].chunk = gpu.num_chunks;
  expect_compile_rejected(gpu, malformed, "non-canonical");
  malformed = baseline;
  malformed.material_refresh_arrays[0].c = NO_COMPONENT;
  expect_compile_rejected(gpu, malformed, "non-canonical");
  malformed = baseline;
  malformed.material_refresh_arrays[0].d = NO_DIRECTION;
  expect_compile_rejected(gpu, malformed, "non-canonical");
  malformed = baseline;
  malformed.material_refresh_arrays[0].family = MaterialRefreshFamily::conductivity;
  expect_compile_rejected(gpu, malformed, "non-canonical");
  malformed = baseline;
  ++malformed.material_refresh_arrays[0].current.value;
  expect_compile_rejected(gpu, malformed, "non-canonical");
  malformed = baseline;
  ++malformed.material_refresh_arrays[0].elements;
  expect_compile_rejected(gpu, malformed, "non-canonical");
  malformed = baseline;
  malformed.operations[phase_index].material_refresh_count =
      uint32_t(malformed.material_refresh_arrays.size() + 1);
  expect_compile_rejected(gpu, malformed, "material refresh descriptor span");
  malformed = baseline;
  require(!malformed.operations[coefficient_index].accesses.empty(),
          "material rejection fixture has no coefficient access");
  malformed.operations[coefficient_index].accesses.pop_back();
  expect_compile_rejected(gpu, malformed, "incomplete or non-canonical");

  component target_c = NO_COMPONENT;
  direction target_d = NO_DIRECTION;
  FOR_COMPONENTS(c) for (int d = 0; d < 5; ++d)
    if (target_c == NO_COMPONENT && target.chunks[0]->chi1inv[c][d]) {
      target_c = c;
      target_d = direction(d);
    }
  require(target_c != NO_COMPONENT, "material rejection fixture has no target row");
  target.chunks[0]->trivial_chi1inv[target_c][target_d] =
      !target.chunks[0]->trivial_chi1inv[target_c][target_d];
  expect_compile_rejected(gpu, baseline, "target fingerprint is stale");
  target.chunks[0]->trivial_chi1inv[target_c][target_d] =
      !target.chunks[0]->trivial_chi1inv[target_c][target_d];

  Executable *valid = gpu.backend->compile(baseline, *gpu.backend_state);
  require(valid != NULL, "valid material plan did not compile after rejection checks");
  delete valid;

  gpu.synchronize_magnetic_fields();
  gpu.restore_magnetic_fields();
  const int countdown = gpu.phasein_time;
  const realnum current_value = gpu.chunks[0]->s->chi1inv[target_c][target_d][0];
  target.chunks[0]->trivial_chi1inv[target_c][target_d] =
      !target.chunks[0]->trivial_chi1inv[target_c][target_d];
  bool rejected = false;
  try {
    gpu.advance(1);
  }
  catch (const std::logic_error &error) {
    rejected = std::string(error.what()).find("target") != std::string::npos;
  }
  require(rejected && gpu.phasein_time == countdown &&
              gpu.chunks[0]->s->chi1inv[target_c][target_d][0] == current_value &&
              !gpu.backend->is_poisoned(),
          "stale material target changed countdown/current coefficients or poisoned backend");
  target.chunks[0]->trivial_chi1inv[target_c][target_d] =
      !target.chunks[0]->trivial_chi1inv[target_c][target_d];
  gpu.advance(1);
  require(gpu.phasein_time == countdown - 1,
          "material target rejection did not permit a corrected retry");
  master_printf("nvidia_timestep: material rejection checks PASS\n");
}

static void test_magnetic_compile_rejections() {
  const grid_volume gv = vol2d(3.0, 2.0, 8.0);
  structure s(gv, isotropic_eps, pml(0.4), identity(), 2);
  set_uniform_conductivity(s);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  const std::vector<double> scaled_k{0.17, -0.11, 0.07};
  fields gpu(&s, options, 0, 0, true, 0, 0, scaled_k);
  gpu.use_bloch(vec(0.17, 0.11));
  gaussian_src_time source(0.31, 0.12);
  gpu.add_point_source(Ez, source, vec(0.11, 0.13));
  gpu.require_component(Ez);
  gpu.init_backend();
  const StepPlan baseline = build_step_plan(gpu, StepProgram::ordinary);
  require(baseline.magnetic_state_arrays.size() > 2,
          "magnetic rejection fixture has too few snapshot rows");

  StepPlan malformed = baseline;
  malformed.magnetic_state_arrays.erase(malformed.magnetic_state_arrays.begin());
  expect_compile_rejected(gpu, malformed, "magnetic snapshot descriptor count");

  malformed = baseline;
  malformed.magnetic_state_arrays.push_back(malformed.magnetic_state_arrays[0]);
  expect_compile_rejected(gpu, malformed, "magnetic snapshot descriptor count");

  malformed = baseline;
  std::swap(malformed.magnetic_state_arrays[0], malformed.magnetic_state_arrays[1]);
  expect_compile_rejected(gpu, malformed, "magnetic snapshot descriptors");

  malformed = baseline;
  malformed.magnetic_state_arrays[0].live = malformed.magnetic_state_arrays[1].live;
  expect_compile_rejected(gpu, malformed, "magnetic snapshot descriptors");

  malformed = baseline;
  ++malformed.magnetic_state_arrays[0].chunk;
  expect_compile_rejected(gpu, malformed, "magnetic snapshot descriptors");

  malformed = baseline;
  malformed.magnetic_state_arrays[0].c = Ex;
  expect_compile_rejected(gpu, malformed, "magnetic snapshot descriptors");

  malformed = baseline;
  malformed.magnetic_state_arrays[0].cmp ^= 1;
  expect_compile_rejected(gpu, malformed, "magnetic snapshot descriptors");

  malformed = baseline;
  malformed.magnetic_state_arrays[0].family = MagneticStateFamily::bfast;
  expect_compile_rejected(gpu, malformed, "magnetic snapshot descriptors");

  malformed = baseline;
  --malformed.magnetic_state_arrays[0].elements;
  expect_compile_rejected(gpu, malformed, "magnetic snapshot descriptors");

  malformed = baseline;
  malformed.magnetic_state_arrays[0].average =
      !malformed.magnetic_state_arrays[0].average;
  expect_compile_rejected(gpu, malformed, "magnetic snapshot descriptors");

  malformed = baseline;
  ++malformed.magnetic_half_step.update_b;
  expect_compile_rejected(gpu, malformed, "magnetic half-step schedule");

  malformed = baseline;
  malformed.operations[malformed.magnetic_half_step.update_b].kind = OpKind::update_eh;
  expect_compile_rejected(gpu, malformed, "BFAST descriptor is not paired");

  malformed = baseline;
  malformed.operations[malformed.magnetic_half_step.update_b].ft = D_stuff;
  expect_compile_rejected(gpu, malformed, "incomplete or non-canonical");

  malformed = baseline;
  malformed.operations[malformed.magnetic_half_step.evaluate_h_sources].source_time_offset = 1.0;
  expect_compile_rejected(gpu, malformed, "incomplete or non-canonical");

  size_t sync_index = baseline.operations.size();
  for (size_t i = 0; i < baseline.operations.size(); ++i)
    if (baseline.operations[i].kind == OpKind::synchronize_magnetic_fields) sync_index = i;
  require(sync_index < baseline.operations.size(), "magnetic rejection fixture has no marker");
  malformed = baseline;
  --malformed.operations[sync_index].magnetic_state_count;
  expect_compile_rejected(gpu, malformed, "magnetic snapshot span");

  malformed = baseline;
  ++malformed.operations[sync_index].magnetic_state_index;
  expect_compile_rejected(gpu, malformed, "magnetic snapshot span");

  malformed = baseline;
  require(!malformed.operations[sync_index].accesses.empty(),
          "magnetic rejection fixture has no marker access");
  malformed.operations[sync_index].accesses.pop_back();
  expect_compile_rejected(gpu, malformed, "incomplete or non-canonical");

  size_t restore_index = baseline.operations.size();
  for (size_t i = 0; i < baseline.operations.size(); ++i)
    if (baseline.operations[i].kind == OpKind::restore_magnetic_fields) restore_index = i;
  require(restore_index < baseline.operations.size(),
          "magnetic rejection fixture has no restore marker");
  malformed = baseline;
  --malformed.operations[restore_index].magnetic_state_count;
  expect_compile_rejected(gpu, malformed, "magnetic snapshot span");

  malformed = baseline;
  require(!malformed.operations[restore_index].accesses.empty(),
          "magnetic rejection fixture has no restore access");
  malformed.operations[restore_index].accesses.pop_back();
  expect_compile_rejected(gpu, malformed, "incomplete or non-canonical");

  Executable *valid = gpu.backend->compile(baseline, *gpu.backend_state);
  require(valid != NULL, "valid magnetic plan did not compile after rejection checks");
  delete valid;

  structure source_free_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  fields source_free(&source_free_structure, options);
  source_free.use_real_fields();
  source_free.require_component(Ez);
  source_free.init_backend();
  const StepPlan source_free_baseline = build_step_plan(source_free, StepProgram::ordinary);
  require(source_free_baseline.magnetic_half_step.evaluate_b_sources == UINT32_MAX &&
              source_free_baseline.magnetic_half_step.evaluate_h_sources == UINT32_MAX,
          "source-free magnetic schedule unexpectedly evaluates source scalars");
  malformed = source_free_baseline;
  malformed.magnetic_half_step.evaluate_b_sources = malformed.magnetic_half_step.update_b;
  expect_compile_rejected(source_free, malformed, "magnetic half-step schedule");
  malformed = source_free_baseline;
  malformed.magnetic_half_step.evaluate_h_sources = malformed.magnetic_half_step.update_h;
  expect_compile_rejected(source_free, malformed, "magnetic half-step schedule");

  master_printf("nvidia_timestep: magnetic rejection checks PASS\n");
}

static void test_nonlinear_compile_rejections() {
  const grid_volume gv = vol3d(2.0, 2.0, 2.0, 5.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  s.set_chi3(chi3_value);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  fields gpu(&s, options);
  gpu.use_bloch(vec(0.11, 0.07, 0.05));
  gpu.require_component(Ez);
  gpu.init_backend();

  const StepPlan baseline = build_step_plan(gpu, StepProgram::ordinary);
  size_t nonlinear_index = baseline.eh_updates.size();
  for (size_t i = 0; i < baseline.eh_updates.size(); ++i) {
    const ConstitutiveUpdate &update = baseline.eh_updates[i];
    if ((update.region.variant_key & constitutive_has_nonlinearity) &&
        is_valid(update.cross1)) {
      nonlinear_index = i;
      break;
    }
  }
  require(nonlinear_index < baseline.eh_updates.size(),
          "nonlinear rejection test has no centered cross-field descriptor");
  const ArrayId target = baseline.eh_updates[nonlinear_index].target;
  const size_t elements = gpu.storage_plan->arrays[target.value].elements;
  std::vector<realnum> before(elements), after(elements);
  gpu.backend->read(ArrayRef{target, 0, elements}, before.data(), before.size() * sizeof(realnum));

  StepPlan malformed = baseline;
  malformed.eh_updates[nonlinear_index].chi3 = invalid_array();
  expect_compile_rejected(gpu, malformed, "nonlinearity bit and operand arrays disagree");

  malformed = baseline;
  ConstitutiveUpdate &out_of_range = malformed.eh_updates[nonlinear_index];
  out_of_range.region.base = 0;
  out_of_range.region.counts[0] = 1;
  out_of_range.region.counts[1] = 1;
  out_of_range.region.counts[2] = 1;
  expect_compile_rejected(gpu, malformed, "nonlinear cross1 index range exceeds its array");

  gpu.backend->read(ArrayRef{target, 0, elements}, after.data(), after.size() * sizeof(realnum));
  require(before == after, "rejected nonlinear descriptors mutated device storage");
}

static void test_polarization_compile_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 2);
  dispersion_sigma_material sigma(false);
  s.add_susceptibility(sigma, E_stuff, lorentzian_susceptibility(0.73, 0.09));
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  gpu.init_backend();

  const StepPlan baseline = build_step_plan(gpu, StepProgram::ordinary);
  require(!baseline.polarization_updates.empty() && !baseline.polarization_subtractions.empty(),
          "polarization rejection test has no descriptors");
  const ArrayId p = baseline.polarization_updates[0].p;
  const size_t elements = gpu.storage_plan->arrays[p.value].elements;
  std::vector<realnum> before(elements), after(elements);
  gpu.backend->read(ArrayRef{p, 0, elements}, before.data(), before.size() * sizeof(realnum));

  StepPlan malformed = baseline;
  malformed.polarization_updates[0].p_prev = invalid_array();
  expect_compile_rejected(gpu, malformed, "incomplete state");

  malformed = baseline;
  malformed.polarization_updates[0].region.variant_key |= polarization_two_offdiagonals;
  malformed.polarization_updates[0].region.variant_key &= ~polarization_one_offdiagonal;
  expect_compile_rejected(gpu, malformed, "second off-diagonal without first");

  malformed = baseline;
  --malformed.polarization_subtractions[0].elements;
  expect_compile_rejected(gpu, malformed, "not a full-array operation");

  gpu.backend->read(ArrayRef{p, 0, elements}, after.data(), after.size() * sizeof(realnum));
  require(before == after, "rejected polarization descriptors mutated device storage");
}

static void test_gyrotropic_compile_rejections() {
  const grid_volume gv = vol3d(2.0, 2.0, 2.0, 6.0);
  linear_anisotropic_material material(false);
  dispersion_sigma_material sigma(false);
  structure s(gv, material, no_pml(), identity(), 2);
  s.add_susceptibility(sigma, E_stuff,
                       gyrotropic_susceptibility(vec(0.17, -0.23, 0.31), 0.73, 0.09));
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ex);
  gpu.require_component(Ey);
  gpu.require_component(Ez);
  gpu.init_backend();

  const StepPlan baseline = build_step_plan(gpu, StepProgram::ordinary);
  size_t index = baseline.polarization_updates.size();
  for (size_t i = 0; i < baseline.polarization_updates.size(); ++i)
    if (baseline.polarization_updates[i].kind == PolarizationUpdateKind::gyrotropic) {
      index = i;
      break;
    }
  require(index < baseline.polarization_updates.size(),
          "gyrotropic rejection test has no update descriptor");
  const ArrayId p = baseline.polarization_updates[index].p;
  const size_t elements = gpu.storage_plan->arrays[p.value].elements;
  std::vector<realnum> before(elements), after(elements);
  gpu.backend->read(ArrayRef{p, 0, elements}, before.data(), before.size() * sizeof(realnum));

  StepPlan malformed = baseline;
  malformed.polarization_updates[index].p_prev_cross2 = invalid_array();
  expect_compile_rejected(gpu, malformed, "incomplete state");
  malformed = baseline;
  malformed.polarization_updates[index].p_prev_cross2 =
      malformed.polarization_updates[index].p;
  expect_compile_rejected(gpu, malformed, "state arrays alias");
  malformed = baseline;
  malformed.polarization_updates[index].kind = static_cast<PolarizationUpdateKind>(99);
  expect_compile_rejected(gpu, malformed, "invalid update kind");
  malformed = baseline;
  malformed.polarization_updates[index].gyro_model = static_cast<gyrotropy_model>(99);
  expect_compile_rejected(gpu, malformed, "invalid model");
  malformed = baseline;
  malformed.polarization_updates[index].region.variant_key = polarization_drude;
  expect_compile_rejected(gpu, malformed, "Lorentzian variant bits");
  malformed = baseline;
  malformed.polarization_updates[index].offdiagonal_sigma1 =
      malformed.polarization_updates[index].diagonal_sigma;
  expect_compile_rejected(gpu, malformed, "anisotropic sigma");
  malformed = baseline;
  malformed.polarization_updates[index].p_cross1 =
      malformed.polarization_updates[index].primary_w;
  expect_compile_rejected(gpu, malformed, "not polarization storage");
  malformed = baseline;
  malformed.polarization_updates[index].region.base = 0;
  malformed.polarization_updates[index].region.counts[0] = 1;
  malformed.polarization_updates[index].region.counts[1] = 1;
  malformed.polarization_updates[index].region.counts[2] = 1;
  expect_compile_rejected(gpu, malformed, "gyrotropic W1 index range exceeds its array");

  gpu.backend->read(ArrayRef{p, 0, elements}, after.data(), after.size() * sizeof(realnum));
  require(before == after, "rejected gyrotropic descriptors mutated device storage");
}

static void test_beta_compile_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, isotropic_eps, pml(0.4, X) + pml(0.4, Y), identity(), 1);
  set_uniform_conductivity(s);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  fields gpu(&s, options, 0, 0.17);
  gaussian_src_time source_time(0.31, 0.14);
  gpu.add_point_source(Ez, source_time, vec(0.73, 0.83));
  gpu.init_backend();

  const StepPlan baseline = build_step_plan(gpu, StepProgram::ordinary);
  require(!baseline.beta_updates.empty(), "beta rejection test has no update descriptor");
  size_t operation_index = baseline.operations.size();
  for (size_t i = 0; i < baseline.operations.size(); ++i)
    if (baseline.operations[i].kind == OpKind::update_db &&
        baseline.operations[i].beta_descriptor_count) {
      operation_index = i;
      break;
    }
  require(operation_index < baseline.operations.size(), "beta rejection test has no owning op");

  const ArrayId target = baseline.beta_updates[0].target;
  const size_t elements = gpu.storage_plan->arrays[target.value].elements;
  std::vector<realnum> before(elements), after(elements);
  gpu.backend->read(ArrayRef{target, 0, elements}, before.data(), before.size() * sizeof(realnum));

  StepPlan malformed = baseline;
  malformed.operations[operation_index].beta_descriptor_count =
      uint32_t(malformed.beta_updates.size() + 1);
  expect_compile_rejected(gpu, malformed, "beta descriptor span is out of range");

  malformed = baseline;
  malformed.beta_updates[0].source = invalid_array();
  expect_compile_rejected(gpu, malformed, "no source field");

  malformed = baseline;
  malformed.beta_updates[0].source = malformed.beta_updates[0].target;
  expect_compile_rejected(gpu, malformed, "aliases mutable and input state");

  malformed = baseline;
  malformed.beta_updates[0].region.counts[0] = 0;
  expect_compile_rejected(gpu, malformed, "empty axis");

  malformed = baseline;
  malformed.beta_updates[0].region.base = std::numeric_limits<size_t>::max();
  expect_compile_rejected(gpu, malformed, "base exceeds ptrdiff_t");

  malformed = baseline;
  malformed.beta_updates[0].betadt = std::numeric_limits<double>::infinity();
  expect_compile_rejected(gpu, malformed, "coefficient is non-finite");

  gpu.backend->read(ArrayRef{target, 0, elements}, after.data(), after.size() * sizeof(realnum));
  require(before == after, "rejected beta descriptors mutated device storage");
}

static void test_bfast_compile_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, isotropic_eps, pml(0.4, X) + pml(0.4, Y), identity(), 1);
  set_uniform_conductivity(s);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  const std::vector<double> scaled_k{0.17, -0.11, 0.07};
  fields gpu(&s, options, 0, 0, true, 0, 0, scaled_k);
  gaussian_src_time source_time(0.31, 0.14);
  gpu.add_point_source(Ez, source_time, vec(0.73, 0.83));
  gpu.init_backend();

  const StepPlan baseline = build_step_plan(gpu, StepProgram::ordinary);
  require(!baseline.bfast_updates.empty() && baseline.db_updates.size() > 1,
          "BFAST rejection test has insufficient descriptors");
  size_t curl_index = baseline.db_updates.size();
  for (size_t i = 0; i < baseline.db_updates.size(); ++i) {
    const uint32_t index = baseline.db_updates[i].bfast_update_index;
    if (index < baseline.bfast_updates.size() &&
        baseline.bfast_updates[index].region.variant_key ==
            (bfast_has_pml | bfast_has_pml_aux | bfast_has_conductivity)) {
      curl_index = i;
      break;
    }
  }
  require(curl_index < baseline.db_updates.size(),
          "BFAST rejection test has no fully auxiliary descriptor");
  const uint32_t bfast_index = baseline.db_updates[curl_index].bfast_update_index;
  require(bfast_index < baseline.bfast_updates.size(),
          "BFAST rejection test has no paired descriptor");
  const ArrayId target = baseline.bfast_updates[bfast_index].target;
  const size_t elements = gpu.storage_plan->arrays[target.value].elements;
  std::vector<realnum> before(elements), after(elements);
  gpu.backend->read(ArrayRef{target, 0, elements}, before.data(), before.size() * sizeof(realnum));

  StepPlan malformed = baseline;
  malformed.db_updates[curl_index].region.variant_key &= ~curl_has_bfast;
  expect_compile_rejected(gpu, malformed, "bit and paired index disagree");
  malformed = baseline;
  malformed.db_updates[curl_index].bfast_update_index = UINT32_MAX;
  expect_compile_rejected(gpu, malformed, "live BFAST coordinate has an unpaired curl row");
  malformed = baseline;
  malformed.db_updates[curl_index].region.variant_key &= ~curl_has_bfast;
  malformed.db_updates[curl_index].bfast_update_index = UINT32_MAX;
  expect_compile_rejected(gpu, malformed, "live BFAST coordinate has an unpaired curl row");
  malformed = baseline;
  malformed.db_updates[curl_index].bfast_update_index = uint32_t(malformed.bfast_updates.size());
  expect_compile_rejected(gpu, malformed, "paired index is out of range");

  malformed = baseline;
  malformed.bfast_updates.push_back(malformed.bfast_updates[bfast_index]);
  expect_compile_rejected(gpu, malformed, "not paired with exactly one curl");
  malformed = baseline;
  const size_t duplicate_curl = curl_index == 0 ? 1 : 0;
  malformed.db_updates[duplicate_curl] = malformed.db_updates[curl_index];
  expect_compile_rejected(gpu, malformed, "not paired with exactly one curl");

  malformed = baseline;
  malformed.bfast_updates[bfast_index].region.begin.set_direction(
      X, malformed.bfast_updates[bfast_index].region.begin.in_direction(X) + 2);
  expect_compile_rejected(gpu, malformed, "curl and paired BFAST descriptors disagree");
  malformed = baseline;
  std::swap(malformed.bfast_updates[bfast_index].source1,
            malformed.bfast_updates[bfast_index].source2);
  expect_compile_rejected(gpu, malformed, "source identity");
  malformed = baseline;
  ++malformed.bfast_updates[bfast_index].stride1;
  expect_compile_rejected(gpu, malformed, "source stride");
  malformed = baseline;
  std::swap(malformed.bfast_updates[bfast_index].pml.sig,
            malformed.bfast_updates[bfast_index].pml.kap);
  expect_compile_rejected(gpu, malformed, "curl and paired BFAST descriptors disagree");

  malformed = baseline;
  malformed.bfast_updates[bfast_index].source1 = invalid_array();
  malformed.bfast_updates[bfast_index].source2 = invalid_array();
  expect_compile_rejected(gpu, malformed, "no source field");
  malformed = baseline;
  malformed.bfast_updates[bfast_index].f_bfast = invalid_array();
  expect_compile_rejected(gpu, malformed, "persistent-state identity");
  malformed = baseline;
  malformed.bfast_updates[bfast_index].f_bfast = malformed.bfast_updates[bfast_index].target;
  expect_compile_rejected(gpu, malformed, "persistent-state identity");
  malformed = baseline;
  malformed.bfast_updates[bfast_index].f_bfast = malformed.bfast_updates[bfast_index].source1;
  expect_compile_rejected(gpu, malformed, "persistent-state identity");

  malformed = baseline;
  malformed.bfast_updates[bfast_index].region.variant_key |= 1u << 12;
  expect_compile_rejected(gpu, malformed, "unsupported variant bit");
  malformed = baseline;
  malformed.bfast_updates[bfast_index].target_u = invalid_array();
  expect_compile_rejected(gpu, malformed, "variant bits and auxiliary arrays disagree");
  malformed = baseline;
  malformed.bfast_updates[bfast_index].k1 = std::numeric_limits<double>::infinity();
  expect_compile_rejected(gpu, malformed, "coefficient is non-finite");
  malformed = baseline;
  malformed.bfast_updates[bfast_index].k1 = -malformed.bfast_updates[bfast_index].k1;
  expect_compile_rejected(gpu, malformed, "coefficients do not match live coordinate routing");
  malformed = baseline;
  malformed.bfast_scaled_k[0] = -malformed.bfast_scaled_k[0];
  expect_compile_rejected(gpu, malformed, "coordinate fingerprint is stale");
  malformed = baseline;
  malformed.bfast_updates[bfast_index].region.base = 0;
  malformed.db_updates[curl_index].region.base = 0;
  expect_compile_rejected(gpu, malformed, "index range exceeds its array");

  gpu.backend->read(ArrayRef{target, 0, elements}, after.data(), after.size() * sizeof(realnum));
  require(before == after, "rejected BFAST descriptors mutated device storage");
}

static void test_cylindrical_compile_rejections() {
  const grid_volume gv = volcyl(2.5, 3.0, 6.0);
  linear_anisotropic_material material(true);
  structure s(gv, material, pml(0.35), identity(), 1);
  set_uniform_conductivity(s);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;
  const std::vector<double> scaled_k{0.17, -0.11, 0.07};
  fields gpu(&s, options, +1.0, 0, true, 64, 64, scaled_k);
  require_cylindrical_components(gpu);
  gpu.init_backend();

  const StepPlan baseline = build_step_plan(gpu, StepProgram::ordinary);
  require(!baseline.cylindrical_radial_prefixes.empty() &&
              !baseline.cylindrical_m_updates.empty() &&
              !baseline.cylindrical_axis_updates.empty() &&
              !baseline.cylindrical_origin_actions.empty(),
          "cylindrical rejection fixture lacks required descriptors");
  size_t update_db_op = baseline.operations.size(), z_curl = baseline.db_updates.size();
  for (size_t oi = 0; oi < baseline.operations.size(); ++oi) {
    const Operation &op = baseline.operations[oi];
    if (op.kind != OpKind::update_db || !op.descriptor_count) continue;
    if (update_db_op == baseline.operations.size()) update_db_op = oi;
    for (size_t i = op.descriptor_index; i < size_t(op.descriptor_index) + op.descriptor_count; ++i)
      if (baseline.db_updates[i].radial_prefix_index != UINT32_MAX) {
        z_curl = i;
        update_db_op = oi;
        break;
      }
    if (z_curl < baseline.db_updates.size()) break;
  }
  require(update_db_op < baseline.operations.size() && z_curl < baseline.db_updates.size(),
          "cylindrical rejection fixture lacks a paired Z curl");
  const ArrayId target = baseline.db_updates[z_curl].target;
  const size_t elements = gpu.storage_plan->arrays[target.value].elements;
  std::vector<realnum> before(elements), after(elements);
  gpu.backend->read(ArrayRef{target, 0, elements}, before.data(), before.size() * sizeof(realnum));

  StepPlan malformed = baseline;
  malformed.db_updates[z_curl].radial_prefix_index = UINT32_MAX;
  expect_compile_rejected(gpu, malformed, "lacks a valid radial-prefix descriptor");
  malformed = baseline;
  malformed.cylindrical_radial_prefixes[0].scratch =
      malformed.cylindrical_radial_prefixes[0].source;
  expect_compile_rejected(gpu, malformed, "storage identity is invalid");
  malformed = baseline;
  malformed.db_updates[z_curl].radial_prefix_index =
      uint32_t(malformed.cylindrical_radial_prefixes.size());
  expect_compile_rejected(gpu, malformed, "lacks a valid radial-prefix descriptor");
  malformed = baseline;
  malformed.cylindrical_radial_prefixes.push_back(malformed.cylindrical_radial_prefixes[0]);
  expect_compile_rejected(gpu, malformed, "not referenced exactly once");
  malformed = baseline;
  ++malformed.cylindrical_radial_prefixes[0].nr;
  expect_compile_rejected(gpu, malformed, "shape does not match its chunk");
  malformed = baseline;
  ++malformed.cylindrical_m_updates[0].raw_radial_start;
  expect_compile_rejected(gpu, malformed, "raw radial coordinate is stale");
  malformed = baseline;
  malformed.cylindrical_m_updates[0].source = malformed.cylindrical_m_updates[0].target;
  expect_compile_rejected(gpu, malformed, "target or source identity is invalid");
  malformed = baseline;
  malformed.cylindrical_m_updates[0].numerator =
      std::numeric_limits<double>::infinity();
  expect_compile_rejected(gpu, malformed, "coefficient is stale");
  malformed = baseline;
  malformed.cylindrical_m_updates[0].region.variant_key ^= cylindrical_m_has_pml_aux;
  expect_compile_rejected(gpu, malformed, "variant bits and auxiliary arrays disagree");
  malformed = baseline;
  malformed.cylindrical_axis_updates[0].region.c = Dr;
  expect_compile_rejected(gpu, malformed, "target or first-source identity is invalid");
  malformed = baseline;
  malformed.cylindrical_axis_updates[0].kind = static_cast<CylindricalAxisKind>(99);
  expect_compile_rejected(gpu, malformed, "kind is invalid");
  malformed = baseline;
  ++malformed.cylindrical_axis_updates[0].source1_neighbor_offset;
  expect_compile_rejected(gpu, malformed, "coefficient or offset is stale");
  malformed = baseline;
  malformed.cylindrical_axis_updates[0].region.variant_key ^= cylindrical_axis_has_pml_aux;
  expect_compile_rejected(gpu, malformed, "variant bits and auxiliary arrays disagree");
  malformed = baseline;
  malformed.cylindrical_axis_updates[0].target_u =
      malformed.cylindrical_axis_updates[0].target;
  expect_compile_rejected(gpu, malformed, "auxiliary storage identity is invalid");
  malformed = baseline;
  malformed.cylindrical_origin_actions.clear();
  malformed.cylindrical_axis_updates.clear();
  malformed.cylindrical_zero_slabs.clear();
  for (Operation &op : malformed.operations)
    if (op.kind == OpKind::update_db) {
      op.cylindrical_origin_action_index = 0;
      op.cylindrical_origin_action_count = 0;
    }
  expect_compile_rejected(gpu, malformed, "incomplete or non-canonical");
  malformed = baseline;
  malformed.cylindrical_origin_actions[0].kind =
      static_cast<CylindricalOriginActionKind>(99);
  expect_compile_rejected(gpu, malformed, "origin action kind is invalid");
  size_t zero_action = baseline.cylindrical_origin_actions.size();
  for (size_t i = 0; i < baseline.cylindrical_origin_actions.size(); ++i)
    if (baseline.cylindrical_origin_actions[i].kind ==
        CylindricalOriginActionKind::zero_slab) {
      zero_action = i;
      break;
    }
  require(zero_action < baseline.cylindrical_origin_actions.size(),
          "cylindrical rejection fixture lacks a zero action");
  malformed = baseline;
  malformed.cylindrical_origin_actions[zero_action].index =
      uint32_t(malformed.cylindrical_zero_slabs.size());
  expect_compile_rejected(gpu, malformed, "zero action index is out of range");
  malformed = baseline;
  ++malformed.cylindrical_zero_slabs[baseline.cylindrical_origin_actions[zero_action].index].base;
  expect_compile_rejected(gpu, malformed, "incomplete or non-canonical");
  malformed = baseline;
  malformed.cylindrical_m_updates.clear();
  for (Operation &op : malformed.operations)
    if (op.kind == OpKind::update_db) {
      op.cylindrical_m_descriptor_index = 0;
      op.cylindrical_m_descriptor_count = 0;
    }
  expect_compile_rejected(gpu, malformed, "incomplete or non-canonical");
  malformed = baseline;
  malformed.cylindrical_origin_r[0] += 1.0;
  expect_compile_rejected(gpu, malformed, "chunk fingerprint is stale");
  malformed = baseline;
  malformed.cylindrical_m = -malformed.cylindrical_m;
  expect_compile_rejected(gpu, malformed, "coordinate fingerprint is stale");
  malformed = baseline;
  malformed.cylindrical_zero_near_origin[0] ^= 1;
  expect_compile_rejected(gpu, malformed, "chunk fingerprint is stale");

  const double saved_chunk_m = gpu.chunks[0]->m;
  gpu.chunks[0]->m = -saved_chunk_m;
  expect_compile_rejected(gpu, baseline, "chunk fingerprint is stale");
  gpu.chunks[0]->m = saved_chunk_m;

  const uint32_t bfast_index = baseline.db_updates[z_curl].bfast_update_index;
  require(bfast_index < baseline.bfast_updates.size(),
          "cylindrical rejection fixture lacks transformed-Z BFAST pairing");
  malformed = baseline;
  const component target_component = malformed.db_updates[z_curl].region.c;
  const component raw_source = is_D(target_component) ? Hp : Ep;
  malformed.bfast_updates[bfast_index].source1 = gpu.array_catalog->find(
      StorageKey{malformed.db_updates[z_curl].region.chunk, int(array_kind::f), int(raw_source),
                 malformed.db_updates[z_curl].region.cmp, 0});
  expect_compile_rejected(gpu, malformed, "source identity");
  malformed = baseline;
  malformed.bfast_updates[bfast_index].k1 = -malformed.bfast_updates[bfast_index].k1;
  expect_compile_rejected(gpu, malformed, "coefficients do not match live coordinate routing");

  size_t axis_replay = baseline.eh_updates.size();
  for (size_t i = 0; i < baseline.eh_updates.size(); ++i)
    if (baseline.eh_updates[i].region.variant_key & constitutive_axis_override) {
      axis_replay = i;
      break;
    }
  require(axis_replay > 0 && axis_replay < baseline.eh_updates.size(),
          "cylindrical rejection fixture lacks constitutive axis replay");
  size_t axis_operation = baseline.operations.size();
  for (size_t i = 0; i < baseline.operations.size(); ++i) {
    const Operation &op = baseline.operations[i];
    if (op.kind == OpKind::update_eh && axis_replay >= op.descriptor_index &&
        axis_replay < size_t(op.descriptor_index) + op.descriptor_count) {
      axis_operation = i;
      break;
    }
  }
  require(axis_operation < baseline.operations.size(),
          "cylindrical rejection fixture lacks axis replay operation");
  malformed = baseline;
  malformed.eh_updates[axis_replay].region.variant_key &= ~constitutive_axis_override;
  expect_compile_rejected(gpu, malformed, "incomplete or non-canonical");
  malformed = baseline;
  malformed.operations[axis_operation].descriptor_count =
      uint32_t(size_t(malformed.operations[axis_operation].descriptor_index) +
               malformed.operations[axis_operation].descriptor_count - axis_replay);
  malformed.operations[axis_operation].descriptor_index = uint32_t(axis_replay);
  expect_compile_rejected(gpu, malformed, "lacks an adjacent ordinary row");
  malformed = baseline;
  malformed.eh_updates[axis_replay].cross1 = malformed.eh_updates[axis_replay - 1].cross1;
  expect_compile_rejected(gpu, malformed, "does not match its ordinary row");
  malformed = baseline;
  malformed.eh_updates[axis_replay].previous_w = malformed.eh_updates[axis_replay].target;
  expect_compile_rejected(gpu, malformed, "does not match its ordinary row");

  gpu.backend->read(ArrayRef{target, 0, elements}, after.data(), after.size() * sizeof(realnum));
  require(before == after, "rejected cylindrical descriptors mutated device storage");
}

static void run_dispersion_case(const char *name, precision_policy_kind policy, bool real_fields,
                                bool drude, bool anisotropic_sigma, bool magnetic,
                                bool use_pml, int chunks, const vec *bloch,
                                unsigned int required_polarization_halo_phases,
                                bool multiple_states = false, bool integrated_source = false,
                                bool nonlinear = false, bool negate_symmetry = false) {
  const grid_volume gv = vol3d(2.4, 2.0, 1.6, 6.0);
  const boundary_region boundaries = use_pml ? pml(0.35, X) + pml(0.35, Y) : no_pml();
  const symmetry sym = negate_symmetry ? -mirror(Y, gv) : identity();
  linear_anisotropic_material tensor_material(anisotropic_sigma, magnetic);
  dispersion_sigma_material sigma(anisotropic_sigma);
  std::unique_ptr<structure> cpu_structure, gpu_structure;
  if (anisotropic_sigma || magnetic) {
    cpu_structure.reset(new structure(gv, tensor_material, boundaries, sym, chunks));
    gpu_structure.reset(new structure(gv, tensor_material, boundaries, sym, chunks));
  }
  else {
    cpu_structure.reset(new structure(gv, isotropic_eps, boundaries, sym, chunks));
    gpu_structure.reset(new structure(gv, isotropic_eps, boundaries, sym, chunks));
  }

  const field_type ft = magnetic ? H_stuff : E_stuff;
  const lorentzian_susceptibility primary(drude ? 0.28 : 0.73, 0.09, drude);
  cpu_structure->add_susceptibility(sigma, ft, primary);
  gpu_structure->add_susceptibility(sigma, ft, primary);
  if (multiple_states) {
    const lorentzian_susceptibility secondary(1.17, 0.035, !drude);
    cpu_structure->add_susceptibility(sigma, ft, secondary);
    gpu_structure->add_susceptibility(sigma, ft, secondary);
  }
  if (nonlinear) {
    cpu_structure->set_chi3(chi3_value);
    gpu_structure->set_chi3(chi3_value);
  }

  fields cpu(cpu_structure.get());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(gpu_structure.get(), options);
  if (real_fields) {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  else if (bloch) {
    cpu.use_bloch(*bloch);
    gpu.use_bloch(*bloch);
  }
  if (anisotropic_sigma) {
    const component requested[] = {magnetic ? Hx : Ex, magnetic ? Hy : Ey,
                                   magnetic ? Hz : Ez};
    for (size_t i = 0; i < 3; ++i) {
      cpu.require_component(requested[i]);
      gpu.require_component(requested[i]);
    }
  }
  else {
    cpu.require_component(magnetic ? Hz : Ez);
    gpu.require_component(magnetic ? Hz : Ez);
  }

  std::unique_ptr<gaussian_src_time> cpu_source, gpu_source;
  if (integrated_source) {
    cpu_source.reset(new gaussian_src_time(0.31, 0.14));
    gpu_source.reset(new gaussian_src_time(0.31, 0.14));
    cpu_source->is_integrated = true;
    gpu_source->is_integrated = true;
    const std::complex<double> amplitude =
        real_fields ? std::complex<double>(0.21, 0.0) : std::complex<double>(0.21, -0.13);
    cpu.add_point_source(Ez, *cpu_source, vec(0.37, 0.41, 0.29), amplitude);
    gpu.add_point_source(Ez, *gpu_source, vec(0.37, 0.41, 0.29), amplitude);
  }

  cpu.advance(1);
  cpu.t = 0;
  build_storage_catalog(cpu, *cpu.array_catalog, *cpu.storage_plan);
  gpu.init_backend();
  require_polarization_plan(gpu, ft, drude, anisotropic_sigma,
                            required_polarization_halo_phases,
                            multiple_states ? size_t(2) : size_t(1));
  if (integrated_source)
    require_source_plan(gpu, false, true, anisotropic_sigma || nonlinear, nonlinear);

  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed, 0.01);
  const bool reduced_precision = sizeof(realnum) == sizeof(float) || narrowed;
  const double tolerance = reduced_precision ? 2e-5 : 2e-13;
  const int checkpoints[] = {1, 2, 100};
  int previous = 0;
  for (size_t i = 0; i < sizeof(checkpoints) / sizeof(checkpoints[0]); ++i) {
    const int delta = checkpoints[i] - previous;
    cpu.advance(delta);
    gpu.advance(delta);
    require(cpu.t == gpu.t, "NVIDIA dispersive timestep did not advance host time");
    compare_fields(cpu, gpu, tolerance);
    previous = checkpoints[i];
  }
  master_printf("nvidia_timestep: dispersion-%s/%s PASS\n", name,
                precision_policy_name(policy));
}

static execution_options host_custom_nvidia_options(
    precision_policy_kind policy = precision_policy_kind::native) {
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  return options;
}

static void run_host_custom_fallback_case(precision_policy_kind policy, bool magnetic) {
  const grid_volume gv = vol2d(2.4, 2.0, 8.0);
  const boundary_region boundary = magnetic ? no_pml() : pml(0.35, X);
  structure cpu_structure(gv, isotropic_eps, boundary, identity(), 2);
  structure gpu_structure(gv, isotropic_eps, boundary, identity(), 2);
  dispersion_sigma_material sigma(false);
  const field_type ft = magnetic ? H_stuff : E_stuff;
  host_callback_trace cpu_trace, gpu_trace;
  counted_inherited_lorentzian cpu_custom(0.73, 0.06, &cpu_trace);
  counted_inherited_lorentzian cpu_custom_second(0.91, 0.045, &cpu_trace);
  counted_inherited_lorentzian gpu_custom(0.73, 0.06, &gpu_trace);
  counted_inherited_lorentzian gpu_custom_second(0.91, 0.045, &gpu_trace);
  lorentzian_susceptibility native(1.17, 0.035);
  if (magnetic) {
    cpu_structure.add_susceptibility(sigma, ft, native);
    cpu_structure.add_susceptibility(sigma, ft, cpu_custom_second);
    cpu_structure.add_susceptibility(sigma, ft, cpu_custom);
    gpu_structure.add_susceptibility(sigma, ft, native);
    gpu_structure.add_susceptibility(sigma, ft, gpu_custom_second);
    gpu_structure.add_susceptibility(sigma, ft, gpu_custom);
  }
  else {
    cpu_structure.add_susceptibility(sigma, ft, cpu_custom);
    cpu_structure.add_susceptibility(sigma, ft, cpu_custom_second);
    cpu_structure.add_susceptibility(sigma, ft, native);
    gpu_structure.add_susceptibility(sigma, ft, gpu_custom);
    gpu_structure.add_susceptibility(sigma, ft, gpu_custom_second);
    gpu_structure.add_susceptibility(sigma, ft, native);
  }

  fields cpu(&cpu_structure);
  execution_options options = host_custom_nvidia_options(policy);
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  cpu.require_component(magnetic ? Hz : Ez);
  gpu.require_component(magnetic ? Hz : Ez);
  gaussian_src_time cpu_source(0.31, 0.14), gpu_source(0.31, 0.14);
  if (!magnetic) {
    cpu_source.is_integrated = true;
    gpu_source.is_integrated = true;
    cpu.add_point_source(Ez, cpu_source, vec(0.37, 0.41), 0.21);
    gpu.add_point_source(Ez, gpu_source, vec(0.37, 0.41), 0.21);
  }
  cpu.advance(1);
  cpu.t = 0;
  build_storage_catalog(cpu, *cpu.array_catalog, *cpu.storage_plan);
  gpu.init_backend();
  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed, 0.19);

  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend != NULL, "custom fallback fixture did not select NVIDIA");
  const StepPlan resident_plan = build_step_plan(gpu, StepProgram::ordinary);
  StoragePlan device_plan = *gpu.storage_plan;
  apply_precision_policy(device_plan, policy_for(policy));
  size_t callbacks_per_step = 0;
  size_t download_calls_per_step = 0, download_bytes_per_step = 0;
  size_t upload_calls_per_step = 0, upload_bytes_per_step = 0;
  for (const HostSegment &segment : resident_plan.host_segments)
    callbacks_per_step += segment.callback_count;
  for (const Operation &op : resident_plan.operations) {
    if (op.kind != OpKind::host_callback) continue;
    std::set<uint32_t> access_ids;
    for (const BufferAccess &access : op.accesses) {
      require(access_ids.insert(access.array.id.value).second,
              "custom fallback access union contains duplicate ArrayIds");
      require(access.mode == AccessMode::read || access.mode == AccessMode::write ||
                  access.mode == AccessMode::read_write,
              "custom fallback access union contains an invalid mode");
      const ArraySpec &spec = device_plan.arrays[access.array.id.value];
      const size_t bytes = access.array.elements *
                           storage_element_bytes(spec.element_type, spec.storage);
      if (access.mode != AccessMode::write) {
        ++download_calls_per_step;
        download_bytes_per_step += bytes;
      }
      if (access.mode != AccessMode::read) {
        ++upload_calls_per_step;
        upload_bytes_per_step += bytes;
      }
    }
  }
  require(resident_plan.host_segments.size() == 2 && callbacks_per_step != 0,
          "custom fallback fixture has an unexpected host-segment topology");
  cpu.advance(1);
  gpu.advance(1);
  compare_fields(cpu, gpu,
                 (sizeof(realnum) == sizeof(float) || narrowed) ? 8e-5 : 8e-13);
  const NvidiaHostFallbackStatistics before = backend->host_fallback_statistics_for_testing();
  const nvidia::memory_accounting memory_before = nvidia::current_memory_accounting();
  cpu_trace.events.clear();
  gpu_trace.events.clear();
  cpu_trace.events.reserve(4096);
  gpu_trace.events.reserve(4096);
  const int measured_steps = 100;
  cpu.advance(measured_steps);
  const size_t gpu_heap_before = heap_allocation_calls.load(std::memory_order_relaxed);
  count_heap_allocations.store(true, std::memory_order_relaxed);
  gpu.advance(measured_steps);
  count_heap_allocations.store(false, std::memory_order_relaxed);
  const size_t gpu_heap_after = heap_allocation_calls.load(std::memory_order_relaxed);
  compare_fields(cpu, gpu,
                 (sizeof(realnum) == sizeof(float) || narrowed) ? 8e-5 : 8e-13);
  require(!cpu_trace.events.empty() && gpu_trace.events == cpu_trace.events,
          "custom fallback changed subtract_P/update_P call count or order");
  const NvidiaHostFallbackStatistics after = backend->host_fallback_statistics_for_testing();
  require(after.segment_executions - before.segment_executions ==
              size_t(measured_steps) * resident_plan.host_segments.size(),
          "custom fallback did not execute two host segments per timestep");
  require(after.callback_resolutions - before.callback_resolutions ==
              size_t(measured_steps) * callbacks_per_step,
          "custom fallback did not re-resolve the callback identity per segment");
  const bool exact_transfers =
      after.device_to_host_calls - before.device_to_host_calls ==
                  size_t(measured_steps) * download_calls_per_step &&
              after.host_to_device_calls - before.host_to_device_calls ==
                  size_t(measured_steps) * upload_calls_per_step &&
              after.device_to_host_bytes - before.device_to_host_bytes ==
                  size_t(measured_steps) * download_bytes_per_step &&
              after.host_to_device_bytes - before.host_to_device_bytes ==
                  size_t(measured_steps) * upload_bytes_per_step &&
              after.synchronizations - before.synchronizations ==
                  size_t(measured_steps) * resident_plan.host_segments.size() &&
              after.steady_capacity_growths == before.steady_capacity_growths &&
              /* The inherited boundary communicator performs one allocation
                 per step. A rebuilt StepPlan/vector would add allocations per
                 host segment and exceed this exact warmed-up baseline. */
              gpu_heap_after - gpu_heap_before == size_t(measured_steps);
  if (!exact_transfers)
    fprintf(stderr,
            "host fallback accounting got d2h=%zu/%zu bytes=%zu/%zu h2d=%zu/%zu bytes=%zu/%zu "
            "sync=%zu/%zu growth=%zu heap=%zu/%zu\n",
            after.device_to_host_calls - before.device_to_host_calls,
            size_t(measured_steps) * download_calls_per_step,
            after.device_to_host_bytes - before.device_to_host_bytes,
            size_t(measured_steps) * download_bytes_per_step,
            after.host_to_device_calls - before.host_to_device_calls,
            size_t(measured_steps) * upload_calls_per_step,
            after.host_to_device_bytes - before.host_to_device_bytes,
            size_t(measured_steps) * upload_bytes_per_step,
            after.synchronizations - before.synchronizations,
            size_t(measured_steps) * resident_plan.host_segments.size(),
            after.steady_capacity_growths - before.steady_capacity_growths,
            gpu_heap_after - gpu_heap_before, size_t(measured_steps));
  require(exact_transfers,
          "custom fallback transfer accounting differs from the declared compact union");
  const nvidia::memory_accounting memory_after = nvidia::current_memory_accounting();
  require(memory_before.device_bytes_current == memory_after.device_bytes_current &&
              memory_before.pinned_bytes_current == memory_after.pinned_bytes_current,
          "custom fallback allocated device or pinned storage during steady execution");
  master_printf("nvidia_timestep: host-custom-%s/%s PASS\n", magnetic ? "H" : "E",
                precision_policy_name(policy));
}

static void test_host_custom_opaque_halo_ownership() {
  const grid_volume gv = vol2d(2.4, 2.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  opaque_inherited_lorentzian cpu_custom(0.73, 0.06);
  opaque_inherited_lorentzian gpu_custom(0.73, 0.06);
  cpu_structure.add_susceptibility(unit_value, E_stuff, cpu_custom);
  gpu_structure.add_susceptibility(unit_value, E_stuff, gpu_custom);

  fields cpu(&cpu_structure);
  fields gpu(&gpu_structure, host_custom_nvidia_options());
  cpu.use_real_fields();
  gpu.use_real_fields();
  cpu.require_component(Ez);
  gpu.require_component(Ez);
  gaussian_src_time cpu_source(0.31, 0.14), gpu_source(0.31, 0.14);
  cpu.add_point_source(Ez, cpu_source, vec(0.37, 0.41), 0.21);
  gpu.add_point_source(Ez, gpu_source, vec(0.37, 0.41), 0.21);
  cpu.advance(1);
  cpu.t = 0;
  build_storage_catalog(cpu, *cpu.array_catalog, *cpu.storage_plan);
  gpu.init_backend();

  size_t opaque_elements = 0;
  for (const HaloPlan &source : gpu.halos->plans) {
    if (source.ft != PE_stuff || !source.block_elements) continue;
    require(source.storage == HaloStorageDisposition::host_owned,
            "opaque custom polarization source halo is not host-owned");
    HaloPlan remapped;
    std::string why;
    require(remap_halo_plan(source, gpu.halos->arrays, gpu.halos->host_arrays,
                            *gpu.array_catalog, gpu.is_real ? 1 : 2, remapped, why),
            "opaque custom polarization halo remap failed");
    require(remapped.storage == HaloStorageDisposition::host_owned,
            "opaque custom polarization halo unexpectedly became device-canonical");
    opaque_elements += source.block_elements;
  }
  require(opaque_elements > 0, "opaque custom fixture produced no host-owned polarization halo");

  const StepPlan plan = build_step_plan(gpu, StepProgram::ordinary);
  size_t owned_halo_descriptors = 0;
  for (const HostSegment &segment : plan.host_segments)
    if (segment.phase == HostSegmentPhase::polarization_and_halo)
      owned_halo_descriptors += segment.host_halo_plan_count;
  require(owned_halo_descriptors > 0,
          "opaque custom polarization halo is not owned by a host callback segment");

  initialize_fields(cpu, gpu, false, 0.07);
  cpu.advance(1);
  gpu.advance(1);
  compare_fields(cpu, gpu, sizeof(realnum) == sizeof(float) ? 8e-5 : 1e-6);
  master_printf("nvidia_timestep: host-custom opaque halo ownership PASS\n");
}

static void test_host_custom_postdispatch_poison() {
  const nvidia::testing::failure_point failures[] = {
      nvidia::testing::failure_point::device_to_host_copy,
      nvidia::testing::failure_point::host_segment_after_download,
      nvidia::testing::failure_point::host_segment_after_callback,
      nvidia::testing::failure_point::host_to_device_copy,
      nvidia::testing::failure_point::host_segment_after_upload};
  for (nvidia::testing::failure_point point : failures) {
    const grid_volume gv = vol2d(2.0, 2.0, 6.0);
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    inherited_lorentzian custom(0.73, 0.06);
    s.add_susceptibility(unit_value, E_stuff, custom);
    execution_options options = host_custom_nvidia_options();
    fields f(&s, options);
    f.use_real_fields();
    f.require_component(Ez);
    f.advance(1);
    nvidia::testing::fail_next(point);
    bool failed = false;
    try { f.advance(1); }
    catch (const std::exception &) { failed = true; }
    nvidia::testing::clear_failure();
    require(failed && f.backend->is_poisoned(),
            "custom fallback transfer/callback failure did not poison the resident backend");
    failed = false;
    try { f.advance(1); }
    catch (const std::exception &) { failed = true; }
    require(failed, "poisoned custom fallback backend accepted another advance");
  }
  master_printf("nvidia_timestep: host-custom postdispatch poison PASS\n");
}

static void test_host_custom_compile_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  inherited_lorentzian custom(0.73, 0.06);
  s.add_susceptibility(unit_value, E_stuff, custom);
  execution_options options = host_custom_nvidia_options();
  fields f(&s, options);
  f.use_real_fields();
  f.require_component(Ez);
  f.init_backend();
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(f.backend);
  require(backend && f.backend_state, "custom rejection fixture has no NVIDIA state");
  const StepPlan canonical = build_step_plan(f, StepProgram::ordinary);
  auto rejected = [&](StepPlan candidate, const char *label) {
    candidate.signature = compute_step_plan_signature(candidate);
    bool failed = false;
    try {
      std::unique_ptr<Executable> executable(backend->compile(candidate, *f.backend_state));
    }
    catch (const std::exception &) {
      failed = true;
    }
    require(failed, label);
  };

  StepPlan missing = canonical;
  missing.host_callbacks.clear();
  missing.host_segments.clear();
  missing.operations.erase(
      std::remove_if(missing.operations.begin(), missing.operations.end(),
                     [](const Operation &op) { return op.kind == OpKind::host_callback; }),
      missing.operations.end());
  rejected(missing, "re-signed host callback deletion bypassed live presence validation");

  StepPlan changed = canonical;
  size_t marker = changed.operations.size();
  for (size_t i = 0; i < changed.operations.size(); ++i)
    if (changed.operations[i].kind == OpKind::host_callback) {
      marker = i;
      break;
    }
  require(marker < changed.operations.size() && !changed.operations[marker].accesses.empty(),
          "custom rejection fixture has no marker access");
  changed.operations[marker].accesses[0].mode =
      changed.operations[marker].accesses[0].mode == AccessMode::read
          ? AccessMode::write
          : AccessMode::read;
  rejected(changed, "re-signed host marker access mutation was accepted");

  changed = canonical;
  const size_t covered = changed.host_segments[0].operation_index;
  changed.operations[covered].source_time_offset = 0.125;
  rejected(changed, "re-signed covered host operation mutation was accepted");

  const std::vector<PolarizationDescriptor> descriptors = f.descriptors->polarizations;
  f.descriptors->polarizations.erase(
      std::remove_if(f.descriptors->polarizations.begin(), f.descriptors->polarizations.end(),
                     [](const PolarizationDescriptor &descriptor) {
                       return descriptor.kind == SusceptibilityKind::host_custom;
                     }),
      f.descriptors->polarizations.end());
  rejected(canonical, "live custom state without a custom descriptor was accepted");
  f.descriptors->polarizations = descriptors;
  f.phasein_time = 1;
  rejected(canonical, "active material phasing with host callbacks was accepted");
  f.phasein_time = 0;
  master_printf("nvidia_timestep: host-custom compile rejections PASS\n");
}

static void test_host_custom_complex_eh_composition() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  dispersion_sigma_material sigma(false);
  host_callback_trace cpu_trace, gpu_trace;
  counted_inherited_lorentzian cpu_e(0.73, 0.06, &cpu_trace);
  counted_inherited_lorentzian cpu_h(0.91, 0.045, &cpu_trace);
  counted_inherited_lorentzian gpu_e(0.73, 0.06, &gpu_trace);
  counted_inherited_lorentzian gpu_h(0.91, 0.045, &gpu_trace);
  cpu_structure.add_susceptibility(sigma, E_stuff, cpu_e);
  cpu_structure.add_susceptibility(sigma, H_stuff, cpu_h);
  gpu_structure.add_susceptibility(sigma, E_stuff, gpu_e);
  gpu_structure.add_susceptibility(sigma, H_stuff, gpu_h);
  fields cpu(&cpu_structure);
  execution_options options = host_custom_nvidia_options();
  fields gpu(&gpu_structure, options);
  cpu.use_bloch(vec(0.13, 0.07));
  gpu.use_bloch(vec(0.13, 0.07));
  cpu.require_component(Ez);
  cpu.require_component(Hz);
  gpu.require_component(Ez);
  gpu.require_component(Hz);
  gaussian_src_time cpu_e_source(0.31, 0.14), gpu_e_source(0.31, 0.14);
  gaussian_src_time cpu_h_source(0.27, 0.12), gpu_h_source(0.27, 0.12);
  cpu.add_point_source(Ez, cpu_e_source, vec(0.31, 0.37), std::complex<double>(0.2, -0.1));
  gpu.add_point_source(Ez, gpu_e_source, vec(0.31, 0.37), std::complex<double>(0.2, -0.1));
  cpu.add_point_source(Hz, cpu_h_source, vec(-0.29, 0.23), std::complex<double>(-0.1, 0.07));
  gpu.add_point_source(Hz, gpu_h_source, vec(-0.29, 0.23), std::complex<double>(-0.1, 0.07));
  component monitor_component = Ez;
  dft_fields cpu_monitor =
      cpu.add_dft_fields(&monitor_component, 1, cpu.v, 0.31, 0.31, 1);
  dft_fields gpu_monitor =
      gpu.add_dft_fields(&monitor_component, 1, gpu.v, 0.31, 0.31, 1);
  flux_vol *cpu_flux =
      cpu.add_flux_vol(X, volume(vec(0.0, -0.75), vec(0.0, 0.75)));
  flux_vol *gpu_flux =
      gpu.add_flux_vol(X, volume(vec(0.0, -0.75), vec(0.0, 0.75)));
  cpu.advance(1);
  cpu.t = 0;
  build_storage_catalog(cpu, *cpu.array_catalog, *cpu.storage_plan);
  gpu.init_backend();
  initialize_fields(cpu, gpu, false, 0.13);
  const StepPlan plan = build_step_plan(gpu, StepProgram::ordinary);
  size_t electric_segments = 0, magnetic_segments = 0;
  for (const HostSegment &segment : plan.host_segments) {
    electric_segments += segment.ft == E_stuff;
    magnetic_segments += segment.ft == H_stuff;
  }
  require(electric_segments == 2 && magnetic_segments == 2,
          "complex E+H custom fixture does not contain both host phases");
  bool dft_update = false, flux_half = false, flux_full = false;
  for (const Operation &operation : plan.operations) {
    dft_update = dft_update || operation.kind == OpKind::update_dft;
    flux_half = flux_half || operation.kind == OpKind::update_flux_half;
    flux_full = flux_full || operation.kind == OpKind::update_flux;
  }
  require(dft_update && flux_half && flux_full,
          "complex E+H custom fixture omitted DFT/flux composition operations");
  cpu_trace.events.clear();
  gpu_trace.events.clear();
  cpu.advance(100);
  gpu.advance(100);
  compare_fields(cpu, gpu, sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-13);
  require(std::abs(cpu_flux->flux() - gpu_flux->flux()) <=
              (sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-12),
          "complex E+H custom DFT/flux composition changed flux");
  int cpu_rank = 0, gpu_rank = 0;
  size_t cpu_dims[3] = {0, 0, 0}, gpu_dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> expected_dft(
      cpu.get_dft_array(cpu_monitor, monitor_component, 0, &cpu_rank, cpu_dims));
  std::unique_ptr<std::complex<realnum>[]> observed_dft(
      gpu.get_dft_array(gpu_monitor, monitor_component, 0, &gpu_rank, gpu_dims));
  require(expected_dft && observed_dft,
          "complex E+H custom DFT/flux composition returned no DFT values");
  require(cpu_rank == gpu_rank, "complex E+H custom DFT/flux composition changed DFT rank");
  size_t dft_elements = 1;
  for (int axis = 0; axis < cpu_rank; ++axis) {
    require(cpu_dims[axis] == gpu_dims[axis],
            "complex E+H custom DFT/flux composition changed DFT dimensions");
    dft_elements *= cpu_dims[axis];
  }
  require(dft_elements > 0, "complex E+H custom DFT/flux composition produced an empty DFT array");
  const double dft_tolerance = sizeof(realnum) == sizeof(float) ? 8e-5 : 8e-12;
  bool saw_dft_signal = false;
  for (size_t i = 0; i < dft_elements; ++i) {
    const double error = std::abs(expected_dft[i] - observed_dft[i]);
    require(error <= dft_tolerance * (1.0 + std::abs(expected_dft[i])),
            "complex E+H custom DFT/flux composition changed accumulated DFT values");
    saw_dft_signal =
        saw_dft_signal || std::abs(expected_dft[i]) > std::numeric_limits<realnum>::epsilon();
  }
  require(saw_dft_signal,
          "complex E+H custom DFT/flux composition did not accumulate a DFT signal");
  require(!cpu_trace.events.empty() && cpu_trace.events == gpu_trace.events,
          "complex E+H custom fallback changed callback order or count");
  master_printf("nvidia_timestep: host-custom complex E+H composition PASS\n");
}

static void test_host_custom_precision_rejection(precision_policy_kind policy) {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  inherited_lorentzian custom(0.73, 0.06);
  s.add_susceptibility(unit_value, E_stuff, custom);
  execution_options options = host_custom_nvidia_options(policy);
  fields f(&s, options);
  f.use_real_fields();
  f.require_component(Ez);
  bool rejected = false;
  try {
    f.advance(1);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected && f.t == 0,
          "non-native host-custom fallback was not rejected before stepping");
  master_printf("nvidia_timestep: host-custom-%s rejection PASS\n",
                precision_policy_name(policy));
}

static void run_gyrotropic_case(const char *name, precision_policy_kind policy, bool real_fields,
                                gyrotropy_model model, bool magnetic, bool use_pml,
                                const vec *bloch, bool negate_symmetry,
                                unsigned int required_w_halo_phases) {
  const grid_volume gv = vol3d(2.4, 2.0, 1.6, 6.0);
  const boundary_region boundaries = use_pml ? pml(0.35, X) + pml(0.35, Y) : no_pml();
  const symmetry sym = negate_symmetry ? -mirror(Y, gv) : identity();
  linear_anisotropic_material material(false, magnetic);
  dispersion_sigma_material sigma(false);
  std::unique_ptr<structure> cpu_structure(new structure(gv, material, boundaries, sym, 2));
  std::unique_ptr<structure> gpu_structure(new structure(gv, material, boundaries, sym, 2));
  const field_type ft = magnetic ? H_stuff : E_stuff;
  const gyrotropic_susceptibility gyro(vec(0.17, -0.23, 0.31), 0.73, 0.06, 0.19, model);
  cpu_structure->add_susceptibility(sigma, ft, gyro);
  gpu_structure->add_susceptibility(sigma, ft, gyro);

  fields cpu(cpu_structure.get());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(gpu_structure.get(), options);
  if (real_fields) {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  else if (bloch) {
    cpu.use_bloch(*bloch);
    gpu.use_bloch(*bloch);
  }
  const component requested[3] = {magnetic ? Hx : Ex, magnetic ? Hy : Ey,
                                  magnetic ? Hz : Ez};
  for (int i = 0; i < 3; ++i) {
    cpu.require_component(requested[i]);
    gpu.require_component(requested[i]);
  }

  cpu.advance(1);
  cpu.t = 0;
  build_storage_catalog(cpu, *cpu.array_catalog, *cpu.storage_plan);
  gpu.init_backend();
  require_gyrotropic_plan(gpu, ft, model, required_w_halo_phases);
  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed, 0.001);
  const bool reduced_precision = sizeof(realnum) == sizeof(float) || narrowed;
  const double tolerance = reduced_precision ? 2e-5 : 2e-13;
  const int checkpoints[] = {1, 2, 100};
  int previous = 0;
  for (int checkpoint : checkpoints) {
    const int delta = checkpoint - previous;
    cpu.advance(delta);
    gpu.advance(delta);
    require(cpu.t == gpu.t, "NVIDIA gyrotropic timestep did not advance host time");
    compare_fields(cpu, gpu, tolerance);
    previous = checkpoint;
  }
  master_printf("nvidia_timestep: gyrotropic-%s/%s PASS\n", name,
                precision_policy_name(policy));
}

static void add_multilevel_test_medium(structure &s, field_type ft, bool large) {
  const realnum gamma3[] = {realnum(0.02), realnum(-0.01), realnum(0.0),
                            realnum(-0.005), realnum(0.03), realnum(-0.004),
                            realnum(0.0), realnum(-0.006), realnum(0.025)};
  const realnum n03[] = {realnum(0.7), realnum(0.2), realnum(0.1)};
  const realnum alpha3[] = {realnum(-0.4), realnum(0.0), realnum(0.4),
                            realnum(-0.3), realnum(0.0), realnum(0.3)};
  const realnum omega2[] = {realnum(0.43), realnum(0.71)};
  const realnum damping2[] = {realnum(0.035), realnum(0.057)};
  const realnum sigmat2[] = {realnum(0.8), realnum(0.9), realnum(1.0), realnum(1.1),
                             realnum(1.2), realnum(0.6), realnum(0.7), realnum(0.85),
                             realnum(0.95), realnum(1.05)};
  const realnum gamma2[] = {realnum(0.02), realnum(-0.01), realnum(-0.005), realnum(0.03)};
  const realnum n02[] = {realnum(0.8), realnum(0.2)};
  const realnum alpha2[] = {realnum(-0.4), realnum(0.4)};
  const realnum omega1[] = {realnum(0.53)};
  const realnum damping1[] = {realnum(0.045)};
  const realnum sigmat1[] = {realnum(0.75), realnum(0.85), realnum(0.95), realnum(1.05),
                             realnum(1.15)};
  if (large) {
    multilevel_susceptibility susceptibility(3, 2, gamma3, n03, alpha3, omega2, damping2,
                                             sigmat2);
    s.add_susceptibility(unit_value, ft, susceptibility);
  }
  else {
    multilevel_susceptibility susceptibility(2, 1, gamma2, n02, alpha2, omega1, damping1,
                                             sigmat1);
    s.add_susceptibility(unit_value, ft, susceptibility);
  }
}

static void run_multilevel_case(const char *name, precision_policy_kind policy,
                                const grid_volume &gv, bool real_fields, bool magnetic,
                                bool large) {
  std::unique_ptr<structure> cpu_structure(
      new structure(gv, isotropic_eps, no_pml(), identity(), 2));
  std::unique_ptr<structure> gpu_structure(
      new structure(gv, isotropic_eps, no_pml(), identity(), 2));
  const field_type ft = magnetic ? H_stuff : E_stuff;
  add_multilevel_test_medium(*cpu_structure, ft, large);
  add_multilevel_test_medium(*gpu_structure, ft, large);

  fields cpu(cpu_structure.get());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(gpu_structure.get(), options);
  if (real_fields) {
    cpu.use_real_fields();
    gpu.use_real_fields();
  }
  else {
    const vec bloch = gv.dim == D3 ? vec(0.09, -0.07, 0.05) : vec(0.09, -0.07);
    cpu.use_bloch(bloch);
    gpu.use_bloch(bloch);
  }
  const component components[] = {magnetic ? Hx : Ex, magnetic ? Hy : Ey,
                                  magnetic ? Hz : Ez};
  const size_t component_count = gv.dim == D1 ? 1 : 3;
  for (size_t i = 0; i < component_count; ++i) {
    const component c = components[i];
    cpu.require_component(c);
    gpu.require_component(c);
  }
  cpu.advance(1);
  cpu.t = 0;
  build_storage_catalog(cpu, *cpu.array_catalog, *cpu.storage_plan);
  gpu.init_backend();
  const field_type halo_ft = magnetic ? PH_stuff : PE_stuff;
  size_t canonical_halo_elements = 0;
  for (const HaloPlan &source : gpu.halos->plans) {
    if (source.ft != halo_ft || !source.block_elements) continue;
    require(source.storage == HaloStorageDisposition::host_owned,
            "multilevel source polarization halo is not initially host-owned");
    HaloPlan canonical;
    std::string why;
    require(remap_halo_plan(source, gpu.halos->arrays, gpu.halos->host_arrays,
                            *gpu.array_catalog, gpu.is_real ? 1 : 2, canonical, why),
            "multilevel polarization halo did not remap");
    require(canonical.storage == HaloStorageDisposition::canonical,
            "exact multilevel polarization halo remained host-owned after remap");
    canonical_halo_elements += source.block_elements;
  }
  require(canonical_halo_elements > 0,
          "multilevel fixture produced no remappable polarization halo elements");
  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_multilevel_fields(cpu, gpu, narrowed, 0.002);
  const double tolerance = (sizeof(realnum) == sizeof(float) || narrowed) ? 8e-5 : 1e-8;
  const int checkpoints[] = {1, 2, 20};
  int previous = 0;
  for (int checkpoint : checkpoints) {
    const int delta = checkpoint - previous;
    cpu.advance(delta);
    gpu.advance(delta);
    compare_fields(cpu, gpu, tolerance);
    previous = checkpoint;
  }
  master_printf("nvidia_timestep: multilevel-%s/%s PASS\n", name,
                precision_policy_name(policy));
}

static void run_multilevel_multitile_pml_case(precision_policy_kind policy) {
  const grid_volume gv = vol2d(3.0, 3.0, 8.0);
  const boundary_region boundaries = pml(0.45, X) + pml(0.45, Y);
  linear_anisotropic_material cpu_material(true), gpu_material(true);
  std::unique_ptr<structure> cpu_structure(
      new structure(gv, cpu_material, boundaries, identity(), 1));
  std::unique_ptr<structure> gpu_structure(
      new structure(gv, gpu_material, boundaries, identity(), 1));
  add_multilevel_test_medium(*cpu_structure, E_stuff, true);
  add_multilevel_test_medium(*gpu_structure, E_stuff, true);

  fields cpu(cpu_structure.get(), 0, 0, true, 0, 4);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(gpu_structure.get(), options, 0, 0, true, 0, 4);
  cpu.use_real_fields();
  gpu.use_real_fields();
  const component components[] = {Ex, Ey, Ez};
  for (component c : components) {
    cpu.require_component(c);
    gpu.require_component(c);
  }
  cpu.advance(1);
  cpu.t = 0;
  build_storage_catalog(cpu, *cpu.array_catalog, *cpu.storage_plan);
  gpu.init_backend();

  const StepPlan prepared = build_step_plan(gpu, StepProgram::ordinary);
  std::map<std::tuple<int, int, int>, size_t> tile_counts;
  std::map<uint32_t, size_t> previous_copy_counts;
  bool multitile = false;
  for (const ConstitutiveUpdate &update : prepared.eh_updates) {
    const std::tuple<int, int, int> identity(update.region.chunk, int(update.region.c),
                                             update.region.cmp);
    multitile = multitile || ++tile_counts[identity] > 1;
    const bool copy =
        (update.region.variant_key & constitutive_copy_w_previous) != 0;
    require(copy == is_valid(update.previous_w),
            "NVIDIA multilevel multi-tile previous-W operand and copy bit disagree");
    if (copy) ++previous_copy_counts[update.previous_w.value];
  }
  bool owns_previous = false;
  for (size_t i = 0; i < gpu.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    if (gpu.array_catalog->key(id).kind != int(array_kind::f_w_prev)) continue;
    owns_previous = true;
    require(previous_copy_counts[id.value] == 1,
            "NVIDIA multilevel multi-tile plan did not copy f_w_prev exactly once");
  }
  require(multitile && owns_previous,
          "NVIDIA multilevel PML fixture did not produce multi-tile previous-W storage");

  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_multilevel_fields(cpu, gpu, narrowed, 0.002);
  const double tolerance = (sizeof(realnum) == sizeof(float) || narrowed) ? 8e-5 : 3e-8;
  const int checkpoints[] = {1, 2, 12};
  int previous = 0;
  double maximum_relative = 0.0, maximum_absolute = 0.0;
  for (int checkpoint : checkpoints) {
    const int delta = checkpoint - previous;
    cpu.advance(delta);
    gpu.advance(delta);
    size_t array = 0, element = 0;
    double absolute = 0.0;
    maximum_relative = std::max(maximum_relative,
                                max_field_difference(cpu, gpu, array, element, absolute));
    maximum_absolute = std::max(maximum_absolute, absolute);
    if (checkpoint == 1 && sizeof(realnum) == sizeof(double) && !narrowed) {
      bool checked_previous = false;
      for (size_t i = 0; i < cpu.array_catalog->size(); ++i) {
        const ArrayId id{uint32_t(i)};
        const ArraySpec &spec = cpu.array_catalog->spec(id);
        if (cpu.array_catalog->key(id).kind != int(array_kind::f_w_prev) ||
            is_valid(spec.alias_of) || spec.element_type != ElementType::realnum_value)
          continue;
        checked_previous = true;
        const realnum *expected = cpu.array_catalog->resolve<realnum>(id);
        std::vector<realnum> observed(spec.elements);
        gpu.backend->read(ArrayRef{id, 0, spec.elements}, observed.data(),
                          observed.size() * sizeof(realnum));
        for (size_t j = 0; j < observed.size(); ++j)
          require(fabs(double(observed[j]) - double(expected[j])) <=
                      1e-13 * (1.0 + fabs(double(expected[j]))),
                  "NVIDIA multilevel first-step f_w_prev copy differs from CPU");
      }
      require(checked_previous,
              "NVIDIA multilevel first-step oracle found no f_w_prev storage");
    }
    compare_fields(cpu, gpu, tolerance);
    previous = checkpoint;
  }
  master_printf("nvidia_timestep: multilevel-multitile-pml/%s PASS "
                "(max relative %.3g, absolute %.3g)\n",
                precision_policy_name(policy), maximum_relative, maximum_absolute);
}

static void test_multilevel_backend_reselection() {
  const grid_volume gv = vol2d(2.4, 2.0, 7.0);
  structure reference_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  structure migrating_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  add_multilevel_test_medium(reference_structure, E_stuff, true);
  add_multilevel_test_medium(migrating_structure, E_stuff, true);
  fields reference(&reference_structure);
  fields migrating(&migrating_structure);
  reference.use_real_fields();
  migrating.use_real_fields();
  const component components[] = {Ex, Ey, Ez};
  for (component c : components) {
    reference.require_component(c);
    migrating.require_component(c);
  }
  gaussian_src_time reference_source(0.31, 0.14), migrating_source(0.31, 0.14);
  reference.add_point_source(Ez, reference_source, vec(0.21, 0.17), 0.23);
  migrating.add_point_source(Ez, migrating_source, vec(0.21, 0.17), 0.23);
  reference.advance(2);
  migrating.advance(2);

  execution_options nvidia_options;
  nvidia_options.backend = backend_kind::nvidia;
  nvidia_options.precision = precision_policy_kind::native;
  migrating.select_backend(nvidia_options);
  migrating.init_backend();
  compare_fields(reference, migrating, 0.0);
  reference.advance(1);
  migrating.advance(1);
  compare_fields(reference, migrating, sizeof(realnum) == sizeof(float) ? 8e-5 : 3e-8);
  require(migrating.backend_state && migrating.executable && !migrating.backend->is_poisoned(),
          "multilevel CPU-to-NVIDIA selection did not publish a usable resident epoch");

  execution_options cpu_options;
  migrating.select_backend(cpu_options);
  compare_fields(reference, migrating, sizeof(realnum) == sizeof(float) ? 8e-5 : 3e-8);
  reference.advance(1);
  migrating.advance(1);
  compare_fields(reference, migrating, sizeof(realnum) == sizeof(float) ? 8e-5 : 3e-8);

  migrating.select_backend(nvidia_options);
  migrating.init_backend();
  compare_fields(reference, migrating, sizeof(realnum) == sizeof(float) ? 8e-5 : 3e-8);
  reference.advance(1);
  migrating.advance(1);
  compare_fields(reference, migrating, sizeof(realnum) == sizeof(float) ? 8e-5 : 3e-8);
  require(migrating.backend_state && migrating.executable && !migrating.backend->is_poisoned(),
          "multilevel NVIDIA re-selection did not restore a usable resident epoch");
  master_printf("nvidia_timestep: multilevel backend reselection PASS\n");
}

static void test_multilevel_gamma_inv_host_authority() {
  if (sizeof(realnum) != sizeof(double)) return;
  const precision_policy_kind policies[] = {precision_policy_kind::mixed,
                                            precision_policy_kind::f32};
  for (precision_policy_kind policy : policies) {
    const grid_volume gv = vol2d(2.4, 2.0, 7.0);
    structure s(gv, isotropic_eps, no_pml(), identity(), 2);
    add_multilevel_test_medium(s, E_stuff, true);
    fields f(&s);
    f.use_real_fields();
    f.require_component(Ex);
    f.require_component(Ey);
    f.require_component(Ez);
    f.advance(1);
    build_storage_catalog(f, *f.array_catalog, *f.storage_plan);

    std::vector<StorageKey> keys;
    std::vector<std::vector<realnum> > authoritative;
    for (size_t i = 0; i < f.array_catalog->size(); ++i) {
      const ArrayId id{uint32_t(i)};
      const ArraySpec &spec = f.array_catalog->spec(id);
      const StorageKey &key = f.array_catalog->key(id);
      if (key.kind != int(array_kind::polarization_internal) ||
          key.component_ != int(Centered) || key.cmp != -1 ||
          polarization_storage_layout_ordinal(key.aux) != 0 ||
          is_valid(spec.alias_of))
        continue;
      keys.push_back(key);
      authoritative.push_back(std::vector<realnum>(spec.elements));
      memcpy(authoritative.back().data(), f.array_catalog->resolve_untyped(id),
             spec.elements * sizeof(realnum));
    }
    require(!keys.empty(), "multilevel GammaInv authority fixture has no host matrix");

    execution_options nvidia_options;
    nvidia_options.backend = backend_kind::nvidia;
    nvidia_options.precision = policy;
    f.select_backend(nvidia_options);
    f.init_backend();
    bool observed_narrowing = false;
    for (size_t i = 0; i < keys.size(); ++i) {
      const ArrayId id = f.array_catalog->find(keys[i]);
      require(is_valid(id), "NVIDIA multilevel migration lost GammaInv storage");
      std::vector<realnum> resident(authoritative[i].size());
      f.backend->read(ArrayRef{id, 0, resident.size()}, resident.data(),
                      resident.size() * sizeof(realnum));
      observed_narrowing = observed_narrowing ||
                           memcmp(resident.data(), authoritative[i].data(),
                                  resident.size() * sizeof(realnum)) != 0;
    }
    require(observed_narrowing,
            "multilevel GammaInv authority fixture did not exercise narrowed storage");
    f.advance(1);

    execution_options cpu_options;
    f.select_backend(cpu_options);
    for (size_t i = 0; i < keys.size(); ++i) {
      const ArrayId id = f.array_catalog->find(keys[i]);
      require(is_valid(id), "multilevel CPU migration lost GammaInv storage");
      const ArraySpec &spec = f.array_catalog->spec(id);
      require(spec.elements == authoritative[i].size() &&
                  memcmp(f.array_catalog->resolve_untyped(id), authoritative[i].data(),
                         spec.elements * sizeof(realnum)) == 0,
              "narrowed NVIDIA GammaInv overwrote host-authoritative coefficients");
    }
  }
  master_printf("nvidia_timestep: multilevel GammaInv host authority PASS\n");
}

static void test_multilevel_zero_row() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  std::unique_ptr<structure> cpu_structure(
      new structure(gv, isotropic_eps, no_pml(), identity(), 1));
  std::unique_ptr<structure> gpu_structure(
      new structure(gv, isotropic_eps, no_pml(), identity(), 1));
  const realnum gamma[] = {realnum(0.02), 0, 0, realnum(0.03)};
  const realnum n0[] = {realnum(0.8), realnum(0.2)};
  const realnum alpha[] = {realnum(-0.4), realnum(0.5)};
  const realnum omega[] = {realnum(0.63)};
  const realnum damping[] = {realnum(0.04)};
  const realnum sigmat[] = {1, 1, 1, 1, 1};
  multilevel_susceptibility cpu_multilevel(2, 1, gamma, n0, alpha, omega, damping, sigmat);
  multilevel_susceptibility gpu_multilevel(2, 1, gamma, n0, alpha, omega, damping, sigmat);
  cpu_structure->add_susceptibility(zero_value, E_stuff, cpu_multilevel);
  gpu_structure->add_susceptibility(zero_value, E_stuff, gpu_multilevel);
  fields cpu(cpu_structure.get());
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(gpu_structure.get(), options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  cpu.require_component(Ez);
  gpu.require_component(Ez);
  cpu.advance(1);
  cpu.t = 0;
  build_storage_catalog(cpu, *cpu.array_catalog, *cpu.storage_plan);
  gpu.init_backend();
  initialize_multilevel_fields(cpu, gpu, false, 0.002);
  cpu.advance(2);
  gpu.advance(2);
  require(gpu.step_plans[0] && gpu.step_plans[0]->multilevel_population_updates.size() == 1 &&
              gpu.step_plans[0]->multilevel_population_terms.empty() &&
              gpu.step_plans[0]->multilevel_transition_updates.empty(),
          "NVIDIA multilevel zero-row fixture did not retain population-only evolution");
  compare_fields(cpu, gpu, sizeof(realnum) == sizeof(float) ? 8e-5 : 2e-12);
  master_printf("nvidia_timestep: multilevel-zero-row PASS\n");
}

static void test_multilevel_compile_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 2);
  add_multilevel_test_medium(s, E_stuff, true);
  lorentzian_susceptibility ordinary(0.73, 0.06, false);
  s.add_susceptibility(unit_value, E_stuff, ordinary);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ex);
  gpu.require_component(Ey);
  gpu.require_component(Ez);
  gpu.advance(1);
  require(gpu.step_plans[0] && gpu.backend_state && gpu.executable,
          "NVIDIA multilevel rejection fixture did not publish an executable");
  const StepPlan baseline = *gpu.step_plans[0];
  BackendState *const state = gpu.backend_state;
  Executable *const executable = gpu.executable;
  const nvidia::memory_accounting memory = nvidia::current_memory_accounting();
  require(!baseline.multilevel_population_updates.empty() &&
              baseline.multilevel_population_terms.size() >= 2 &&
              baseline.multilevel_transition_updates.size() >= 2 &&
              !baseline.polarization_groups.empty() &&
              !baseline.polarization_subtractions.empty(),
          "NVIDIA multilevel rejection fixture lacks canonical rows");
  size_t multilevel_group_index = baseline.polarization_groups.size();
  for (size_t i = 0; i < baseline.polarization_groups.size(); ++i)
    if (baseline.polarization_groups[i].kind == PolarizationGroupKind::multilevel) {
      multilevel_group_index = i;
      break;
    }
  require(multilevel_group_index < baseline.polarization_groups.size(),
          "NVIDIA multilevel rejection fixture has no multilevel group");

  StepPlan malformed = baseline;
  ++malformed.multilevel_population_updates[0].levels;
  expect_compile_rejected(gpu, malformed, "population rows differ");
  malformed = baseline;
  malformed.multilevel_population_updates[0].gamma_inv =
      malformed.multilevel_population_updates[0].populations;
  expect_compile_rejected(gpu, malformed, "population rows differ");
  malformed = baseline;
  std::swap(malformed.multilevel_population_terms[0],
            malformed.multilevel_population_terms[1]);
  expect_compile_rejected(gpu, malformed, "population terms differ");
  malformed = baseline;
  malformed.multilevel_transition_updates[0].p =
      malformed.multilevel_transition_updates[0].p_prev;
  expect_compile_rejected(gpu, malformed, "transition rows differ");
  malformed = baseline;
  --malformed.polarization_groups[multilevel_group_index].transition_count;
  expect_compile_rejected(gpu, malformed, "polarization groups differ");
  malformed = baseline;
  for (PolarizationUpdateGroup &group : malformed.polarization_groups)
    if (group.kind == PolarizationGroupKind::multilevel)
      group.kind = PolarizationGroupKind::recurrence;
  malformed.multilevel_population_updates.clear();
  malformed.multilevel_population_terms.clear();
  malformed.multilevel_transition_updates.clear();
  malformed.multilevel_coefficients.clear();
  expect_compile_rejected(gpu, malformed, "polarization groups differ");
  PolarizationDescriptor *installed_multilevel = NULL, *installed_lorentz = NULL;
  for (PolarizationDescriptor &descriptor : gpu.descriptors->polarizations)
    if (descriptor.kind == SusceptibilityKind::multilevel) {
      installed_multilevel = &descriptor;
    }
    else if (descriptor.kind == SusceptibilityKind::lorentzian)
      installed_lorentz = &descriptor;
  require(installed_multilevel && installed_lorentz,
          "NVIDIA multilevel rejection fixture lacks a supported replacement descriptor");
  const PolarizationDescriptor saved_multilevel = *installed_multilevel;
  *installed_multilevel = *installed_lorentz;
  expect_compile_rejected(gpu, malformed, "descriptors differ from live exact states");
  *installed_multilevel = saved_multilevel;
  malformed = baseline;
  malformed.polarization_subtractions[0].transition_index = 7;
  expect_compile_rejected(gpu, malformed, "subtraction rows differ");
  malformed = baseline;
  for (Operation &op : malformed.operations)
    if ((op.kind == OpKind::update_polarization || op.kind == OpKind::update_eh) &&
        !op.accesses.empty()) {
      op.accesses[0].mode = AccessMode::write;
      break;
    }
  expect_compile_rejected(gpu, malformed, "operation span or access set");

  const nvidia::memory_accounting after = nvidia::current_memory_accounting();
  require(gpu.backend_state == state && gpu.executable == executable &&
              !gpu.backend->is_poisoned() &&
              memory.device_bytes_current == after.device_bytes_current &&
              memory.pinned_bytes_current == after.pinned_bytes_current,
          "rejected NVIDIA multilevel compile mutated the live epoch");
  master_printf("nvidia_timestep: multilevel compile rejections PASS\n");
}

static void test_multilevel_lifecycle() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  execution_options options;
  options.backend = backend_kind::nvidia;
  structure s(gv, isotropic_eps, no_pml(), identity(), 2);
  add_multilevel_test_medium(s, E_stuff, true);
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ex);
  gpu.require_component(Ey);
  gpu.require_component(Ez);
  gpu.advance(1);
  BackendState *const state = gpu.backend_state;
  Executable *const executable = gpu.executable;
  const nvidia::memory_accounting memory_before = nvidia::current_memory_accounting();
  nvidia::testing::reset_transfer_accounting();
  gpu.advance(1);
  const nvidia::testing::transfer_accounting transfers =
      nvidia::testing::current_transfer_accounting();
  const nvidia::memory_accounting memory_after = nvidia::current_memory_accounting();
  require(gpu.backend_state == state && gpu.executable == executable &&
              transfers.host_to_device_calls == 0 && transfers.device_to_host_calls == 0 &&
              memory_before.device_bytes_current == memory_after.device_bytes_current &&
              memory_before.pinned_bytes_current == memory_after.pinned_bytes_current,
          "NVIDIA multilevel steady step allocated, transferred, or rebuilt");

  structure retry_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  add_multilevel_test_medium(retry_structure, E_stuff, true);
  fields retry(&retry_structure, options);
  retry.use_real_fields();
  retry.require_component(Ex);
  retry.require_component(Ey);
  retry.require_component(Ez);
  retry.init_backend();
  BackendState *const prepared_state = retry.backend_state;
  Executable *const prepared_executable = retry.executable;
  invalidate(retry, MutationKind::coordinate_definition,
             "NVIDIA multilevel compile allocation retry");
  nvidia::testing::fail_next(nvidia::testing::failure_point::device_allocate);
  bool failed = false;
  try { retry.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  nvidia::testing::clear_failure();
  require(failed && retry.t == 0 && retry.backend_state == prepared_state &&
              retry.executable == prepared_executable && !retry.backend->is_poisoned(),
          "NVIDIA multilevel compile allocation failure published or dispatched");
  retry.advance(1);
  require(retry.executable && retry.executable != prepared_executable && retry.t == 1,
          "NVIDIA multilevel compile allocation failure was not retryable");

  structure population_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  add_multilevel_test_medium(population_structure, E_stuff, true);
  fields population_failure(&population_structure, options);
  population_failure.use_real_fields();
  population_failure.require_component(Ez);
  population_failure.advance(1);
  nvidia::testing::fail_next(nvidia::testing::failure_point::multilevel_population);
  failed = false;
  try { population_failure.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  nvidia::testing::clear_failure();
  require(failed && population_failure.backend->is_poisoned(),
          "NVIDIA multilevel population postlaunch failure did not poison");

  structure transition_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  add_multilevel_test_medium(transition_structure, E_stuff, true);
  fields transition_failure(&transition_structure, options);
  transition_failure.use_real_fields();
  transition_failure.require_component(Ez);
  transition_failure.advance(1);
  nvidia::testing::fail_next(nvidia::testing::failure_point::multilevel_transition);
  failed = false;
  try { transition_failure.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  nvidia::testing::clear_failure();
  require(failed && transition_failure.backend->is_poisoned(),
          "NVIDIA multilevel transition postlaunch failure did not poison");
  master_printf("nvidia_timestep: multilevel lifecycle PASS\n");
}

static void test_multilevel_composition() {
  const grid_volume gv = vol3d(2.0, 1.8, 1.6, 5.0);
  const boundary_region boundaries = pml(0.35, X) + pml(0.35, Y);
  structure s(gv, isotropic_eps, boundaries, identity(), 4);
  add_multilevel_test_medium(s, E_stuff, true);
  add_multilevel_test_medium(s, H_stuff, false);
  lorentzian_susceptibility lorentz(0.73, 0.06, false);
  noisy_lorentzian_susceptibility noisy(0.015625, 0.91, 0.04, false);
  gyrotropic_susceptibility gyro(vec(0.17, -0.23, 0.31), 0.83, 0.05, 0.13,
                                 GYROTROPIC_LORENTZIAN);
  s.add_susceptibility(unit_value, E_stuff, lorentz);
  s.add_susceptibility(unit_value, E_stuff, noisy);
  s.add_susceptibility(unit_value, H_stuff, gyro);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&s, options);
  gpu.use_bloch(vec(0.09, -0.07, 0.05));
  const component components[] = {Ex, Ey, Ez, Hx, Hy, Hz};
  for (component c : components) gpu.require_component(c);
  gaussian_src_time source(0.31, 0.14);
  source.is_integrated = true;
  gpu.add_point_source(Ez, source, vec(0.31, 0.27, 0.19),
                       std::complex<double>(0.17, -0.08));
  component monitor_component = Ez;
  dft_fields monitor = gpu.add_dft_fields(&monitor_component, 1, gpu.v, 0.31, 0.31, 1);
  flux_vol *flux =
      gpu.add_flux_vol(Z, volume(vec(-0.7, -0.6, 0.0), vec(0.7, 0.6, 0.0)));
  set_random_seed(0x13572468UL);
  gpu.advance(2);
  require(gpu.step_plans[0] && gpu.executable && !gpu.backend->is_poisoned(),
          "NVIDIA multilevel composition did not publish a usable executable");
  bool multilevel_e = false, multilevel_h = false, ordinary = false, noisy_row = false,
       gyro_row = false, source_op = false, dft_op = false, flux_half = false,
       flux_full = false;
  for (const PolarizationUpdateGroup &group : gpu.step_plans[0]->polarization_groups) {
    if (group.kind == PolarizationGroupKind::multilevel) {
      multilevel_e = multilevel_e || group.ft == E_stuff;
      multilevel_h = multilevel_h || group.ft == H_stuff;
    }
  }
  for (const PolarizationUpdate &update : gpu.step_plans[0]->polarization_updates) {
    ordinary = ordinary || update.kind == PolarizationUpdateKind::lorentzian;
    noisy_row = noisy_row || update.kind == PolarizationUpdateKind::noisy_add;
    gyro_row = gyro_row || update.kind == PolarizationUpdateKind::gyrotropic;
  }
  for (const Operation &op : gpu.step_plans[0]->operations) {
    source_op = source_op || op.kind == OpKind::apply_sources;
    dft_op = dft_op || op.kind == OpKind::update_dft;
    flux_half = flux_half || op.kind == OpKind::update_flux_half;
    flux_full = flux_full || op.kind == OpKind::update_flux;
  }
  require(multilevel_e && multilevel_h && ordinary && noisy_row && gyro_row && source_op &&
              dft_op && flux_half && flux_full && std::isfinite(flux->flux()),
          "NVIDIA multilevel composition omitted a requested feature");
  master_printf("nvidia_timestep: multilevel composition PASS\n");
}

static void test_multilevel_capability_rejections() {
  execution_options options;
  options.backend = backend_kind::nvidia;
  {
    const grid_volume gv = volcyl(2.0, 2.0, 6.0);
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    add_multilevel_test_medium(s, E_stuff, false);
    fields gpu(&s, options);
    gpu.use_real_fields();
    gpu.require_component(Ez);
    require_advance_rejected(gpu, "multilevel polarization does not support cylindrical");
  }
  {
    const grid_volume gv = vol2d(2.0, 2.0, 6.0);
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    add_multilevel_test_medium(s, E_stuff, false);
    fields gpu(&s, options);
    gpu.require_component(Ez);
    gaussian_src_time source(0.31, 0.14);
    gpu.add_point_source(Ez, source, vec(0.25, 0.25), 0.2);
    bool rejected = false;
    try { (void)gpu.solve_cw(1e-4, 1, std::complex<double>(0.3, 0.0), 2); }
    catch (const std::exception &error) {
      const std::string message(error.what());
      rejected = message.find("multilevel") != std::string::npos ||
                 message.find("dispersion, polarization") != std::string::npos;
    }
    require(rejected, "NVIDIA solve_cw accepted exact multilevel polarization");
  }
  master_printf("nvidia_timestep: multilevel capability rejections PASS\n");
}

static void run_finite_diagnostic_case(const char *name, precision_policy_kind policy,
                                       FiniteCheckMode mode, bool poison, realnum value,
                                       bool expect_rejection, int expected_step) {
  set_finite_check_mode(mode);
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  if (getenv("MEEP_NVIDIA_GRAPH_ASSERT")) {
    options.strict = false;
    options.fallback = fallback_policy::warn;
  }
  fields f(&s, options);
  f.use_real_fields();
  f.require_component(Ez);
  f.init_backend();
  require_selected_graph_mode(f);

  const StepPlan prepared = build_step_plan(f, StepProgram::ordinary);
  const Operation *finite = NULL;
  for (size_t i = 0; i < prepared.operations.size(); ++i)
    if (prepared.operations[i].kind == OpKind::finite_value_check) finite = &prepared.operations[i];
  require(finite && !finite->accesses.empty(), "NVIDIA finite check has no declared spans");

  int expected_component = -1;
  if (poison) {
    expected_component = f.storage_plan->keys[finite->accesses[0].array.id.value].component_;
    for (size_t i = 0; i < finite->accesses.size(); ++i) {
      const ArrayRef ref = finite->accesses[i].array;
      std::vector<realnum> values(ref.elements, value);
      f.backend->write(ref, values.data(), values.size() * sizeof(realnum));
    }
  }

  bool rejected = false;
  try {
    f.advance(2);
  }
  catch (const std::runtime_error &error) {
    rejected =
        std::string(error.what()).find("simulation fields are NaN or Inf") != std::string::npos;
    if (!expect_rejection || !rejected) throw;
  }
  require(rejected == expect_rejection, "NVIDIA finite-value diagnostic result differs");
  require(f.t == expected_step, "NVIDIA finite-value diagnostic cadence differs");
  require(f.nonfinite_flag == unsigned(expect_rejection),
          "NVIDIA finite-value diagnostic flag differs");
  if (expect_rejection) {
    require(f.first_bad_step == expected_step, "NVIDIA first-bad step differs");
    require(f.first_bad_component == expected_component, "NVIDIA first-bad component differs");
  }
  master_printf("nvidia_timestep: finite-%s/%s PASS\n", name, precision_policy_name(policy));
}

static void test_finite_diagnostics(precision_policy_kind policy) {
  run_finite_diagnostic_case("pass", policy, FiniteCheckMode::step, false, realnum(0), false, 2);
  run_finite_diagnostic_case("nan-step", policy, FiniteCheckMode::step, true,
                             std::numeric_limits<realnum>::quiet_NaN(), true, 1);
  run_finite_diagnostic_case("inf-batch", policy, FiniteCheckMode::batch, true,
                             std::numeric_limits<realnum>::infinity(), true, 2);
  run_finite_diagnostic_case("off", policy, FiniteCheckMode::off, true,
                             std::numeric_limits<realnum>::quiet_NaN(), false, 2);
}

struct noisy_snapshot {
  std::vector<StorageKey> keys;
  std::vector<std::vector<realnum> > values;
};

static noisy_snapshot capture_noisy_state(fields &f) {
  noisy_snapshot result;
  for (size_t i = 0; i < f.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = f.array_catalog->spec(id);
    if (spec.role != array_role::polarization || is_valid(spec.alias_of) ||
        spec.element_type != ElementType::realnum_value)
      continue;
    result.keys.push_back(f.array_catalog->key(id));
    result.values.push_back(std::vector<realnum>(spec.elements));
    f.backend->read(ArrayRef{id, 0, spec.elements}, result.values.back().data(),
                    spec.elements * sizeof(realnum));
  }
  return result;
}

static bool same_noisy_snapshot(const noisy_snapshot &a, const noisy_snapshot &b) {
  if (a.keys.size() != b.keys.size() || a.values.size() != b.values.size()) return false;
  for (size_t i = 0; i < a.keys.size(); ++i)
    if (!(a.keys[i] == b.keys[i]) || a.values[i].size() != b.values[i].size() ||
        memcmp(a.values[i].data(), b.values[i].data(),
               a.values[i].size() * sizeof(realnum)) != 0)
      return false;
  return true;
}

static noisy_snapshot capture_noisy_group(fields &f, field_type ft, int state_index,
                                          component selected_component = Dielectric) {
  require(f.step_plans[0] && f.array_catalog,
          "NVIDIA noisy group capture requires an installed plan and catalog");
  noisy_snapshot result;
  std::set<uint32_t> seen;
  for (const PolarizationUpdate &update : f.step_plans[0]->polarization_updates) {
    if (update.kind != PolarizationUpdateKind::noisy_add || update.ft != ft ||
        update.state_index != state_index ||
        (selected_component != Dielectric && update.region.c != selected_component) ||
        !seen.insert(update.p.value).second)
      continue;
    const ArraySpec &spec = f.array_catalog->spec(update.p);
    StorageKey key = f.array_catalog->key(update.p);
    require(polarization_storage_field_type(key.aux) == ft &&
                polarization_storage_state_index(key.aux) == state_index,
            "NVIDIA noisy state has the wrong packed storage identity");
    key.aux = polarization_storage_layout_ordinal(key.aux);
    result.keys.push_back(key);
    result.values.push_back(std::vector<realnum>(spec.elements));
    f.backend->read(ArrayRef{update.p, 0, spec.elements}, result.values.back().data(),
                    spec.elements * sizeof(realnum));
  }
  std::vector<size_t> order(result.keys.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const StorageKey &x = result.keys[a], &y = result.keys[b];
    if (x.chunk != y.chunk) return x.chunk < y.chunk;
    if (x.kind != y.kind) return x.kind < y.kind;
    if (x.component_ != y.component_) return x.component_ < y.component_;
    if (x.cmp != y.cmp) return x.cmp < y.cmp;
    return x.aux < y.aux;
  });
  noisy_snapshot sorted;
  for (size_t i : order) {
    sorted.keys.push_back(result.keys[i]);
    sorted.values.push_back(result.values[i]);
  }
  return sorted;
}

static noisy_snapshot run_noisy_order_fixture(precision_policy_kind policy, int mode) {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 2);
  noisy_lorentzian_susceptibility target(0.03125, 0.73, 0.06, false);
  noisy_lorentzian_susceptibility inserted(0.01953125, 1.17, 0.035, true);
  s.add_susceptibility(unit_value, E_stuff, target);
  if (mode != 0) s.add_susceptibility(unit_value, E_stuff, inserted);
  if (mode == 2) {
    s.remove_susceptibilities();
    s.add_susceptibility(unit_value, E_stuff, target);
  }
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  set_random_seed(0x10293847UL);
  gpu.advance(1);
  const int target_state = mode == 1 ? 1 : 0;
  noisy_snapshot result = capture_noisy_group(gpu, E_stuff, target_state, Ez);
  require(!result.values.empty(), "NVIDIA noisy order fixture produced no target state");
  return result;
}

static void test_noisy_rebuild_lifecycle() {
  const precision_policy_kind policy = precision_policy_kind::native;
  const auto zero_resident = [](fields &f) {
    for (size_t i = 0; i < f.array_catalog->size(); ++i) {
      const ArrayId id{uint32_t(i)};
      const ArraySpec &spec = f.array_catalog->spec(id);
      if (is_valid(spec.alias_of) || spec.element_type != ElementType::realnum_value ||
          (spec.role != array_role::field && spec.role != array_role::polarization))
        continue;
      std::vector<realnum> zero(spec.elements, realnum(0));
      f.backend->write(ArrayRef{id, 0, spec.elements}, zero.data(),
                       spec.elements * sizeof(realnum));
    }
  };
  const noisy_snapshot baseline = run_noisy_order_fixture(policy, 0);
  const noisy_snapshot shifted = run_noisy_order_fixture(policy, 1);
  const noisy_snapshot restored = run_noisy_order_fixture(policy, 2);
  require(!same_noisy_snapshot(baseline, shifted),
          "NVIDIA noisy stream did not change when an earlier state changed its ordinal");
  require(same_noisy_snapshot(baseline, restored),
          "NVIDIA noisy remove/add did not restore semantic state identity");

  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  noisy_lorentzian_susceptibility noisy(0.03125, 0.73, 0.06, false);
  structure resident_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  resident_structure.add_susceptibility(unit_value, E_stuff, noisy);
  execution_options nvidia_options;
  nvidia_options.backend = backend_kind::nvidia;
  nvidia_options.precision = policy;
  fields resident(&resident_structure, nvidia_options);
  resident.use_real_fields();
  resident.require_component(Ez);
  set_random_seed(0x55667788UL);
  resident.advance(1);
  const noisy_snapshot resident_baseline = capture_noisy_group(resident, E_stuff, 0, Ez);
  const size_t initial_catalog_size = resident.array_catalog->size();
  const uint64_t initial_layout_generation =
      resident.mutation_generation[size_t(MutationKind::field_layout)];

  zero_resident(resident);
  resident.t = 0;
  resident.require_component(Ex);
  set_random_seed(0x55667788UL);
  resident.advance(1);
  require(resident.backend_state && resident.executable &&
              resident.array_catalog->size() > initial_catalog_size &&
              resident.mutation_generation[size_t(MutationKind::field_layout)] >
                  initial_layout_generation,
          "NVIDIA noisy field growth did not rebuild the resident catalog");
  require(same_noisy_snapshot(resident_baseline,
                              capture_noisy_group(resident, E_stuff, 0, Ez)),
          "NVIDIA noisy catalog replacement changed the semantic stream");

  zero_resident(resident);
  resident.t = 0;
  execution_options cpu_options;
  resident.select_backend(cpu_options);
  resident.select_backend(nvidia_options);
  set_random_seed(0x55667788UL);
  resident.advance(1);
  require(same_noisy_snapshot(resident_baseline,
                              capture_noisy_group(resident, E_stuff, 0, Ez)),
          "NVIDIA noisy backend reselection changed the semantic stream");

  structure migrating_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  migrating_structure.add_susceptibility(unit_value, E_stuff, noisy);
  fields migrating(&migrating_structure);
  migrating.use_real_fields();
  migrating.require_component(Ez);
  migrating.select_backend(nvidia_options);
  set_random_seed(0x55667788UL);
  migrating.advance(1);
  require(same_noisy_snapshot(resident_baseline,
                              capture_noisy_group(migrating, E_stuff, 0, Ez)),
          "CPU-to-NVIDIA noisy selection changed the semantic stream");
  master_printf("nvidia_timestep: noisy rebuild lifecycle PASS\n");
}

static noisy_snapshot run_noisy_seed_case(precision_policy_kind policy, unsigned long seed,
                                          bool drude, field_type ft, bool complex_fields,
                                          realnum noise_amplitude, bool exact_noisy = true) {
  const grid_volume gv = vol3d(2.0, 1.8, 1.6, 5.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 2);
  noisy_lorentzian_susceptibility noisy(noise_amplitude, 0.73, 0.06, drude);
  lorentzian_susceptibility ordinary(0.73, 0.06, drude);
  if (exact_noisy)
    s.add_susceptibility(unit_value, ft, noisy);
  else
    s.add_susceptibility(unit_value, ft, ordinary);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  if (getenv("MEEP_NVIDIA_GRAPH_ASSERT")) {
    options.strict = false;
    options.fallback = fallback_policy::warn;
  }
  fields gpu(&s, options);
  if (complex_fields)
    gpu.use_bloch(vec(0.11, 0.07, 0.05));
  else
    gpu.use_real_fields();
  gpu.require_component(ft == E_stuff ? Ez : Hz);
  set_random_seed(seed);
  gpu.init_backend();
  require_selected_graph_mode(gpu);
  gpu.advance(3);
  require(gpu.backend_state && gpu.executable &&
              gpu.backend_state->random_seed_snapshot_accepted == exact_noisy,
          "NVIDIA noisy run did not publish seed metadata");
  return capture_noisy_state(gpu);
}

static void run_noisy_replay_case(precision_policy_kind policy) {
  const noisy_snapshot first =
      run_noisy_seed_case(policy, 0x12345678UL, false, E_stuff, false, realnum(0.03125));
  const noisy_snapshot replay =
      run_noisy_seed_case(policy, 0x12345678UL, false, E_stuff, false, realnum(0.03125));
  const noisy_snapshot changed =
      run_noisy_seed_case(policy, 0x87654321UL, false, E_stuff, false, realnum(0.03125));
  const noisy_snapshot wrapped = run_noisy_seed_case(
      policy, 0x12345678UL + (1UL << 32), false, E_stuff, false, realnum(0.03125));
  const noisy_snapshot magnetic =
      run_noisy_seed_case(policy, 0x12345678UL, true, H_stuff, true, realnum(0.0625));
  const noisy_snapshot zero_noise =
      run_noisy_seed_case(policy, 0x12345678UL, false, E_stuff, false, realnum(0));
  const noisy_snapshot ordinary =
      run_noisy_seed_case(policy, 0x12345678UL, false, E_stuff, false, realnum(0), false);
  require(!first.values.empty() && same_noisy_snapshot(first, replay),
          "NVIDIA noisy fixed-seed replay is not bitwise stable");
  require(same_noisy_snapshot(first, wrapped),
          "NVIDIA noisy seed did not preserve low-32-bit equivalence");
  require(!same_noisy_snapshot(first, changed),
          "NVIDIA noisy stream did not change with semantic seed");
  require(!magnetic.values.empty(), "NVIDIA noisy Drude magnetic run produced no state");
  require(same_noisy_snapshot(zero_noise, ordinary),
          "NVIDIA finite zero-noise path differs from ordinary Lorentz recurrence");
  set_random_seed(424242);
  const int expected_mt = random_int(0, 1000000);
  (void)run_noisy_seed_case(policy, 424242, false, E_stuff, false, realnum(0.03125));
  const int observed_mt = random_int(0, 1000000);
  set_random_seed(424242);
  (void)random_seed_snapshot();
  require(observed_mt == expected_mt,
          "NVIDIA counter-noise execution consumed the legacy MT stream");
  master_printf("nvidia_timestep: noisy-replay/%s PASS\n", precision_policy_name(policy));
}

static void require_noisy_composition_plan(fields &gpu, bool expect_phase,
                                           bool expect_negate, bool expect_diagonal,
                                           bool expect_anisotropic) {
  require(gpu.step_plans[0] && gpu.descriptors && gpu.halos,
          "NVIDIA noisy composition did not publish its prepared plan");
  const StepPlan &plan = *gpu.step_plans[0];
  std::set<int> chunks;
  std::set<int> electric_states;
  std::set<int> magnetic_states;
  bool saw_diagonal = false, saw_anisotropic = false;
  bool saw_source = false, saw_dft = false, saw_flux_half = false, saw_flux = false;
  for (const PolarizationUpdate &update : plan.polarization_updates) {
    const uint32_t offdiagonal =
        update.region.variant_key &
        (polarization_one_offdiagonal | polarization_two_offdiagonals);
    if (update.kind == PolarizationUpdateKind::lorentzian) {
      saw_diagonal |= offdiagonal == 0;
      saw_anisotropic |= offdiagonal != 0;
    }
    if (update.kind != PolarizationUpdateKind::noisy_add) continue;
    chunks.insert(update.region.chunk);
    (update.ft == E_stuff ? electric_states : magnetic_states).insert(update.state_index);
  }
  for (const Operation &op : plan.operations) {
    saw_source |= op.kind == OpKind::apply_sources;
    saw_dft |= op.kind == OpKind::update_dft;
    saw_flux_half |= op.kind == OpKind::update_flux_half;
    saw_flux |= op.kind == OpKind::update_flux;
  }
  unsigned halo_phases = 0;
  for (const HaloPlan &halo : gpu.halos->plans)
    if ((halo.ft == PE_stuff || halo.ft == PH_stuff) && halo.same_rank && halo.block_elements)
      halo_phases |= 1u << unsigned(halo.phase);
  require(chunks.size() >= 2, "NVIDIA noisy composition did not span multiple chunks");
  require(electric_states.count(0) && electric_states.count(1),
          "NVIDIA noisy composition lost linked-list E-state order");
  require(magnetic_states.count(0), "NVIDIA noisy composition lost its H/Drude state");
  if (expect_diagonal)
    require(saw_diagonal, "NVIDIA noisy composition did not exercise diagonal sigma rows");
  if (expect_anisotropic)
    require(saw_anisotropic,
            "NVIDIA noisy composition did not exercise anisotropic sigma rows");
  require(saw_source && saw_dft && saw_flux_half && saw_flux,
          "NVIDIA noisy composition omitted source, DFT, or legacy-flux operations");
  require((halo_phases & (1u << CONNECT_COPY)) != 0,
          "NVIDIA noisy composition omitted COPY polarization halos");
  if (expect_phase)
    require((halo_phases & (1u << CONNECT_PHASE)) != 0,
            "NVIDIA noisy composition omitted PHASE polarization halos");
  if (expect_negate)
    require((halo_phases & (1u << CONNECT_NEGATE)) != 0,
            "NVIDIA noisy composition omitted NEGATE polarization halos");
  bool saw_pml = false;
  for (const CurlUpdate &update : plan.db_updates)
    saw_pml |= update.region.variant_key != 0;
  require(saw_pml || expect_negate,
          "NVIDIA noisy composition omitted the requested PML physics");
}

static void run_noisy_composition_case(precision_policy_kind policy) {
  const grid_volume gv = vol3d(2.4, 2.0, 1.6, 6.0);
  const boundary_region boundaries = pml(0.35, X) + pml(0.35, Y);
  linear_anisotropic_material material(true, true);
  dispersion_sigma_material anisotropic_sigma(true);
  dispersion_sigma_material diagonal_sigma(false);
  structure s(gv, material, boundaries, identity(), 4);
  noisy_lorentzian_susceptibility electric_lorentz(0.015625, 0.73, 0.06, false);
  noisy_lorentzian_susceptibility electric_drude(-0.01171875, 0.28, 0.09, true);
  noisy_lorentzian_susceptibility magnetic_drude(0.0078125, 0.41, 0.04, true);
  s.add_susceptibility(anisotropic_sigma, E_stuff, electric_lorentz);
  s.add_susceptibility(diagonal_sigma, E_stuff, electric_drude);
  s.add_susceptibility(anisotropic_sigma, H_stuff, magnetic_drude);

  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(&s, options);
  gpu.use_bloch(vec(0.11, 0.07, 0.05));
  const component components[] = {Ex, Ey, Ez, Hx, Hy, Hz};
  for (component c : components) gpu.require_component(c);
  gaussian_src_time source(0.31, 0.14);
  source.is_integrated = true;
  gpu.add_point_source(Ez, source, vec(0.37, 0.41, 0.29),
                       std::complex<double>(0.21, -0.13));
  component monitor_component = Ez;
  dft_fields monitor = gpu.add_dft_fields(&monitor_component, 1, gpu.v, 0.31, 0.31, 1);
  flux_vol *flux =
      gpu.add_flux_vol(Z, volume(vec(-0.75, -0.65, 0.0), vec(0.75, 0.65, 0.0)));
  set_random_seed(0x13579bdfUL);
  gpu.advance(2);
  require_noisy_composition_plan(gpu, true, false, false, true);
  require(std::isfinite(flux->flux()),
          "NVIDIA noisy source/DFT/flux composition published a nonfinite flux");

  structure negate_structure(gv, isotropic_eps, no_pml(), -mirror(Y, gv), 2);
  negate_structure.add_susceptibility(diagonal_sigma, E_stuff, electric_lorentz);
  negate_structure.add_susceptibility(diagonal_sigma, E_stuff, electric_drude);
  negate_structure.add_susceptibility(diagonal_sigma, H_stuff, magnetic_drude);
  fields negate(&negate_structure, options);
  negate.use_real_fields();
  for (component c : components) negate.require_component(c);
  gaussian_src_time negate_source(0.31, 0.14);
  negate.add_point_source(Ez, negate_source, vec(0.37, 0.41, 0.29), 0.21);
  component negate_monitor_component = Ez;
  dft_fields negate_monitor =
      negate.add_dft_fields(&negate_monitor_component, 1, negate.v, 0.31, 0.31, 1);
  flux_vol *negate_flux =
      negate.add_flux_vol(Z, volume(vec(-0.75, -0.65, 0.0), vec(0.75, 0.65, 0.0)));
  set_random_seed(0x2468ace0UL);
  negate.advance(2);
  require_noisy_composition_plan(negate, false, true, true, false);
  require(std::isfinite(negate_flux->flux()),
          "NVIDIA noisy NEGATE composition published a nonfinite flux");
  master_printf("nvidia_timestep: noisy-composition/%s PASS\n",
                precision_policy_name(policy));
}

static void run_noisy_sigma_diagnostic_case(precision_policy_kind policy,
                                            realnum noise_amplitude,
                                            realnum dynamic_sigma,
                                            const char *name) {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  noisy_lorentzian_susceptibility noisy(noise_amplitude, 0.73, 0.06, false);
  s.add_susceptibility(unit_value, E_stuff, noisy);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  set_finite_check_mode(FiniteCheckMode::step);
  set_random_seed(0x31415926UL);
  gpu.advance(1);
  const PolarizationUpdate *target = NULL;
  for (const PolarizationUpdate &update : gpu.step_plans[0]->polarization_updates)
    if (update.kind == PolarizationUpdateKind::noisy_add && update.region.c == Ez) {
      target = &update;
      break;
    }
  require(target && is_valid(target->diagonal_sigma),
          "NVIDIA noisy diagnostic fixture lacks its sigma row");
  const ArraySpec &sigma_spec = gpu.array_catalog->spec(target->diagonal_sigma);
  std::vector<realnum> sigma(sigma_spec.elements, dynamic_sigma);
  gpu.backend->write(ArrayRef{target->diagonal_sigma, 0, sigma.size()}, sigma.data(),
                     sigma.size() * sizeof(realnum));
  bool rejected = false;
  try { gpu.advance(2); }
  catch (const std::runtime_error &error) {
    rejected = std::string(error.what()).find("simulation fields are NaN or Inf") !=
               std::string::npos;
  }
  if (!(rejected && gpu.nonfinite_flag && gpu.first_bad_component == int(Ez)))
    fprintf(stderr,
            "noisy diagnostic %s: rejected=%d flag=%u component=%d expected=%d step=%d\n",
            name, int(rejected), gpu.nonfinite_flag, gpu.first_bad_component, int(Ez),
            gpu.first_bad_step);
  require(rejected && gpu.nonfinite_flag && gpu.first_bad_component == int(Ez),
          "NVIDIA noisy dynamic-sigma diagnostic lacked component attribution");
  set_finite_check_mode(FiniteCheckMode::off);
  master_printf("nvidia_timestep: noisy-diagnostic-%s/%s PASS\n", name,
                precision_policy_name(policy));
}

static void test_noisy_sigma_diagnostics(precision_policy_kind policy) {
  run_noisy_sigma_diagnostic_case(policy, realnum(0.03125), realnum(-1), "negative");
  run_noisy_sigma_diagnostic_case(policy, realnum(0.03125),
                                  std::numeric_limits<realnum>::quiet_NaN(), "nan");
  run_noisy_sigma_diagnostic_case(policy, realnum(0.03125),
                                  std::numeric_limits<realnum>::infinity(), "infinite");
  run_noisy_sigma_diagnostic_case(policy, realnum(0), realnum(-1), "zero-amplitude-negative");
}

struct resident_value_snapshot {
  std::vector<ArrayId> ids;
  std::vector<std::vector<realnum> > values;
};

static resident_value_snapshot capture_resident_values(fields &f) {
  resident_value_snapshot result;
  for (size_t i = 0; i < f.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = f.array_catalog->spec(id);
    if (is_valid(spec.alias_of) || spec.element_type != ElementType::realnum_value ||
        (spec.role != array_role::field && spec.role != array_role::polarization))
      continue;
    result.ids.push_back(id);
    result.values.push_back(std::vector<realnum>(spec.elements));
    f.backend->read(ArrayRef{id, 0, spec.elements}, result.values.back().data(),
                    spec.elements * sizeof(realnum));
  }
  return result;
}

static void restore_resident_values(fields &f, const resident_value_snapshot &snapshot) {
  for (size_t i = 0; i < snapshot.ids.size(); ++i)
    f.backend->write(ArrayRef{snapshot.ids[i], 0, snapshot.values[i].size()},
                     snapshot.values[i].data(), snapshot.values[i].size() * sizeof(realnum));
}

static bool same_resident_values(const resident_value_snapshot &a,
                                 const resident_value_snapshot &b) {
  if (a.ids != b.ids || a.values.size() != b.values.size()) return false;
  for (size_t i = 0; i < a.values.size(); ++i)
    if (a.values[i].size() != b.values[i].size() ||
        memcmp(a.values[i].data(), b.values[i].data(),
               a.values[i].size() * sizeof(realnum)) != 0)
      return false;
  return true;
}

static void test_noisy_lifecycle() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 2);
  noisy_lorentzian_susceptibility noisy(0.03125, 0.73, 0.06);
  s.add_susceptibility(unit_value, E_stuff, noisy);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  set_random_seed(0x12345678UL);
  gpu.init_backend();
  gpu.advance(2);
  BackendState *const state = gpu.backend_state;
  Executable *const executable = gpu.executable;
  const int replay_time = gpu.t;
  const resident_value_snapshot baseline = capture_resident_values(gpu);

  nvidia::testing::reset_transfer_accounting();
  const nvidia::memory_accounting memory_before = nvidia::current_memory_accounting();
  gpu.advance(1);
  const nvidia::memory_accounting memory_after = nvidia::current_memory_accounting();
  const nvidia::testing::transfer_accounting steady =
      nvidia::testing::current_transfer_accounting();
  const resident_value_snapshot seed_s = capture_resident_values(gpu);
  require(steady.host_to_device_calls == 0 && steady.device_to_host_calls == 0 &&
              memory_before.device_bytes_current == memory_after.device_bytes_current &&
              memory_before.pinned_bytes_current == memory_after.pinned_bytes_current,
          "NVIDIA noisy steady step allocated or transferred host data");

  restore_resident_values(gpu, baseline);
  gpu.t = replay_time;
  set_random_seed(0x87654321UL);
  nvidia::testing::reset_transfer_accounting();
  gpu.advance(1);
  const nvidia::testing::transfer_accounting changed =
      nvidia::testing::current_transfer_accounting();
  const resident_value_snapshot seed_t = capture_resident_values(gpu);
  require(changed.host_to_device_calls == 1 &&
              changed.host_to_device_bytes == sizeof(nvidia::noisy_seed_block) &&
              changed.device_to_host_calls == 0 && !same_resident_values(seed_s, seed_t),
          "NVIDIA noisy reseed transfer or output differs");

  restore_resident_values(gpu, baseline);
  gpu.t = replay_time;
  restore_random_seed();
  gpu.advance(1);
  require(same_resident_values(seed_s, capture_resident_values(gpu)) &&
              gpu.backend_state == state && gpu.executable == executable,
          "NVIDIA noisy S/T/restore replay changed state or executable identity");

  restore_resident_values(gpu, baseline);
  gpu.t = replay_time;
  set_random_seed(0x12345678UL);
  nvidia::testing::reset_transfer_accounting();
  gpu.advance(1);
  const nvidia::testing::transfer_accounting repeated =
      nvidia::testing::current_transfer_accounting();
  require(same_resident_values(seed_s, capture_resident_values(gpu)) &&
              repeated.host_to_device_calls == 1 &&
              repeated.host_to_device_bytes == sizeof(nvidia::noisy_seed_block) &&
              gpu.backend_state == state && gpu.executable == executable,
          "NVIDIA repeated semantic seed did not refresh without rebuilding");

  set_random_seed(0x31415926UL);
  const resident_value_snapshot retry_input = capture_resident_values(gpu);
  nvidia::testing::fail_next(nvidia::testing::failure_point::noisy_seed_copy);
  bool failed = false;
  const int before_failure_time = gpu.t;
  try { gpu.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  nvidia::testing::clear_failure();
  require(failed && !gpu.backend->is_poisoned() && gpu.t == before_failure_time &&
              gpu.backend_state == state && gpu.executable == executable,
          "retryable NVIDIA noisy seed copy failure changed the live epoch");
  gpu.advance(1);
  const resident_value_snapshot retry_output = capture_resident_values(gpu);
  restore_resident_values(gpu, retry_input);
  gpu.t = before_failure_time;
  set_random_seed(0x31415926UL);
  gpu.advance(1);
  require(same_resident_values(retry_output, capture_resident_values(gpu)),
          "NVIDIA noisy seed-copy retry differs from a clean same-key dispatch");

  nvidia::testing::fail_next(nvidia::testing::failure_point::noisy_add);
  failed = false;
  try { gpu.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  nvidia::testing::clear_failure();
  require(failed && gpu.backend->is_poisoned(),
          "NVIDIA noisy postlaunch failure did not poison the backend");
  require_advance_rejected(gpu, "poisoned");
  std::vector<realnum> rejected_read(1);
  bool read_rejected = false;
  try {
    gpu.backend->read(ArrayRef{ArrayId{0}, 0, 1}, rejected_read.data(), sizeof(realnum));
  }
  catch (const std::exception &) { read_rejected = true; }
  require(read_rejected, "NVIDIA noisy poisoned backend accepted a readback");
  bool reselection_rejected = false;
  try {
    execution_options cpu_options;
    gpu.select_backend(cpu_options);
  }
  catch (const std::exception &error) {
    reselection_rejected = std::string(error.what()).find("poison") != std::string::npos;
  }
  require(reselection_rejected,
          "NVIDIA noisy poisoned backend accepted migration/reselection");

  structure replacement_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  replacement_structure.add_susceptibility(unit_value, E_stuff, noisy);
  fields sync_failure(&replacement_structure, options);
  sync_failure.use_real_fields();
  sync_failure.require_component(Ez);
  set_random_seed(0x27182818UL);
  sync_failure.init_backend();
  sync_failure.advance(1);
  set_random_seed(0x16180339UL);
  nvidia::testing::fail_next(nvidia::testing::failure_point::noisy_seed_sync);
  failed = false;
  try { sync_failure.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  nvidia::testing::clear_failure();
  require(failed && sync_failure.backend->is_poisoned(),
          "NVIDIA noisy seed synchronization failure did not poison");
  structure recovered_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  recovered_structure.add_susceptibility(unit_value, E_stuff, noisy);
  fields recovered(&recovered_structure, options);
  recovered.use_real_fields();
  recovered.require_component(Ez);
  set_random_seed(0x16180339UL);
  recovered.advance(1);
  require(recovered.backend_state && recovered.executable && !recovered.backend->is_poisoned(),
          "replacement NVIDIA noisy state was not usable after seed-sync poison");

  structure cold_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  cold_structure.add_susceptibility(unit_value, E_stuff, noisy);
  fields cold(&cold_structure, options);
  cold.use_real_fields();
  cold.require_component(Ez);
  set_random_seed(0xabcdef01UL);
  const int expected_mt = random_int(0, 1000000);
  set_random_seed(0xabcdef01UL);
  nvidia::testing::fail_next(nvidia::testing::failure_point::device_allocate);
  failed = false;
  try { cold.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  nvidia::testing::clear_failure();
  require(failed && !cold.backend->is_poisoned() && cold.t == 0 && !cold.backend_state &&
              !cold.executable && random_int(0, 1000000) == expected_mt,
          "cold NVIDIA noisy allocation failure dispatched, drew MT state, or published epoch");
  cold.advance(1);
  const noisy_snapshot cold_retry = capture_noisy_group(cold, E_stuff, 0, Ez);
  structure clean_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  clean_structure.add_susceptibility(unit_value, E_stuff, noisy);
  fields clean(&clean_structure, options);
  clean.use_real_fields();
  clean.require_component(Ez);
  set_random_seed(0xabcdef01UL);
  clean.advance(1);
  require(same_noisy_snapshot(cold_retry, capture_noisy_group(clean, E_stuff, 0, Ez)),
          "cold NVIDIA noisy allocation retry differs from clean execution");

  structure compile_structure(gv, isotropic_eps, no_pml(), identity(), 2);
  compile_structure.add_susceptibility(unit_value, E_stuff, noisy);
  fields compile_failure(&compile_structure, options);
  compile_failure.use_real_fields();
  compile_failure.require_component(Ez);
  set_random_seed(0xabcdef01UL);
  compile_failure.init_backend();
  BackendState *const prepared_state = compile_failure.backend_state;
  require(prepared_state && !compile_failure.executable &&
              !prepared_state->random_seed_snapshot_accepted,
          "NVIDIA noisy compile-failure fixture was not cold at executable publication");
  nvidia::testing::fail_next(nvidia::testing::failure_point::device_allocate);
  failed = false;
  try { compile_failure.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  nvidia::testing::clear_failure();
  require(failed && !compile_failure.backend->is_poisoned() && compile_failure.t == 0 &&
              compile_failure.backend_state == prepared_state && !compile_failure.executable &&
              !prepared_state->random_seed_snapshot_accepted,
          "NVIDIA noisy executable allocation failure published or dispatched");
  compile_failure.advance(1);
  require(same_noisy_snapshot(cold_retry,
                              capture_noisy_group(compile_failure, E_stuff, 0, Ez)),
          "NVIDIA noisy executable allocation retry differs from clean execution");
  master_printf("nvidia_timestep: noisy lifecycle PASS\n");
}

static void test_noisy_compile_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 2);
  noisy_lorentzian_susceptibility noisy(0.03125, 0.73, 0.06);
  s.add_susceptibility(unit_value, E_stuff, noisy);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  set_random_seed(12345);
  gpu.advance(1);
  require(gpu.step_plans[0] && gpu.backend_state,
          "NVIDIA noisy rejection fixture did not prepare a plan");
  const StepPlan baseline = *gpu.step_plans[0];
  BackendState *const live_state = gpu.backend_state;
  Executable *const live_executable = gpu.executable;
  const nvidia::memory_accounting memory_before = nvidia::current_memory_accounting();
  const resident_value_snapshot values_before = capture_resident_values(gpu);
  size_t noisy_index = baseline.polarization_updates.size();
  for (size_t i = 0; i < baseline.polarization_updates.size(); ++i)
    if (baseline.polarization_updates[i].kind == PolarizationUpdateKind::noisy_add) {
      noisy_index = i;
      break;
    }
  require(noisy_index < baseline.polarization_updates.size(),
          "NVIDIA noisy rejection fixture has no noise row");

  StepPlan malformed = baseline;
  malformed.polarization_updates[noisy_index].dt = 0.0;
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  ++malformed.polarization_updates[noisy_index].noise_algorithm_version;
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  malformed.polarization_updates[noisy_index].region.chunk = -1;
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  malformed.polarization_updates[noisy_index].state_index = -1;
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  malformed.polarization_updates[noisy_index].region.cmp = 2;
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  malformed.polarization_updates[noisy_index].ft = H_stuff;
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  malformed.polarization_updates[noisy_index].omega_0 =
      std::numeric_limits<double>::infinity();
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  malformed.polarization_updates[noisy_index].gamma = -0.01;
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  malformed.polarization_updates[noisy_index].noise_amplitude =
      std::numeric_limits<double>::quiet_NaN();
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  malformed.polarization_updates[noisy_index].region.counts[2] = 0;
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  for (int axis = 0; axis < 3; ++axis)
    if (malformed.polarization_updates[noisy_index].region.counts[axis] > 1) {
      malformed.polarization_updates[noisy_index].region.strides[axis] = 0;
      break;
    }
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  malformed.polarization_updates[noisy_index].p =
      malformed.polarization_updates[noisy_index].diagonal_sigma;
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  for (Operation &op : malformed.operations)
    if (op.kind == OpKind::update_polarization) {
      op.guard = guard_device(7);
      break;
    }
  expect_compile_rejected(gpu, malformed, "operation span or access set");
  malformed = baseline;
  for (Operation &op : malformed.operations)
    if (op.kind == OpKind::update_polarization && !op.accesses.empty()) {
      op.accesses[0].mode = AccessMode::write;
      break;
    }
  expect_compile_rejected(gpu, malformed, "operation span or access set");
  malformed = baseline;
  malformed.polarization_updates.erase(malformed.polarization_updates.begin() + noisy_index);
  for (Operation &op : malformed.operations)
    if (op.kind == OpKind::update_polarization && op.ft == E_stuff) --op.descriptor_count;
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  malformed = baseline;
  malformed.polarization_updates.push_back(malformed.polarization_updates[noisy_index]);
  for (Operation &op : malformed.operations)
    if (op.kind == OpKind::update_polarization && op.ft == E_stuff) ++op.descriptor_count;
  expect_compile_rejected(gpu, malformed, "descriptor authority");
  if (noisy_index > 0) {
    malformed = baseline;
    std::swap(malformed.polarization_updates[noisy_index - 1],
              malformed.polarization_updates[noisy_index]);
    expect_compile_rejected(gpu, malformed, "descriptor authority");
  }

  PolarizationDescriptor *descriptor = NULL;
  for (PolarizationDescriptor &candidate : gpu.descriptors->polarizations)
    if (candidate.kind == SusceptibilityKind::noisy_lorentzian) {
      descriptor = &candidate;
      break;
    }
  require(descriptor, "NVIDIA noisy rejection fixture has no descriptor");
  descriptor->kind = SusceptibilityKind::lorentzian;
  expect_compile_rejected(gpu, baseline, "live linked-list order");
  descriptor->kind = SusceptibilityKind::noisy_lorentzian;

  const nvidia::memory_accounting memory_after = nvidia::current_memory_accounting();
  require(gpu.backend_state == live_state && gpu.executable == live_executable &&
              !gpu.backend->is_poisoned() &&
              memory_before.device_bytes_current == memory_after.device_bytes_current &&
              memory_before.pinned_bytes_current == memory_after.pinned_bytes_current &&
              same_resident_values(values_before, capture_resident_values(gpu)),
          "rejected NVIDIA noisy compile mutated the live epoch or device state");

  const int saved_t = gpu.t;
  gpu.t = -1;
  bool rejected = false;
  try { gpu.backend->advance(*gpu.executable, *gpu.backend_state, 1); }
  catch (const std::invalid_argument &) { rejected = true; }
  gpu.t = saved_t;
  require(rejected && !gpu.backend->is_poisoned(),
          "NVIDIA noisy negative timestep was accepted or poisoned predispatch");
  master_printf("nvidia_timestep: noisy compile rejections PASS\n");
}

static void test_noisy_capability_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  execution_options options;
  options.backend = backend_kind::nvidia;

  inherited_noisy_lorentzian derived(0.03125, 0.73, 0.06);
  structure derived_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  derived_structure.add_susceptibility(unit_value, E_stuff, derived);
  fields derived_fields(&derived_structure, options);
  derived_fields.use_real_fields();
  derived_fields.require_component(Ez);
  set_random_seed(12345);
  bool rejected = false;
  try { derived_fields.advance(1); }
  catch (const std::exception &) { rejected = true; }
  require(rejected &&
              (!derived_fields.backend_state ||
               !derived_fields.backend_state->random_seed_snapshot_accepted),
          "NVIDIA accepted a derived noisy host-custom susceptibility or seed metadata");

  const realnum Gamma[] = {realnum(0), realnum(0.02), realnum(0), realnum(-0.02)};
  const realnum N0[] = {realnum(1), realnum(0)};
  const realnum alpha[] = {realnum(-0.2), realnum(0.2)};
  const realnum omega[] = {realnum(0.73)};
  const realnum gamma[] = {realnum(0.06)};
  const realnum sigmat[] = {realnum(1), realnum(1), realnum(1), realnum(0), realnum(0)};
  inherited_multilevel multilevel(2, 1, Gamma, N0, alpha, omega, gamma, sigmat);
  structure multilevel_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  multilevel_structure.add_susceptibility(unit_value, E_stuff, multilevel);
  fields multilevel_fields(&multilevel_structure, options);
  multilevel_fields.use_real_fields();
  multilevel_fields.require_component(Ez);
  rejected = false;
  try { multilevel_fields.advance(1); }
  catch (const std::exception &) { rejected = true; }
  require(rejected, "NVIDIA accepted a derived multilevel host-custom susceptibility");

  noisy_lorentzian_susceptibility noisy(0.03125, 0.73, 0.06);
  structure cw_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  cw_structure.add_susceptibility(unit_value, E_stuff, noisy);
  fields cw(&cw_structure, options);
  cw.require_component(Ez);
  rejected = false;
  try { (void)cw.solve_cw(1e-4, 1, std::complex<double>(0.3, 0.0), 2); }
  catch (const std::exception &) { rejected = true; }
  require(rejected && (!cw.backend_state || !cw.backend_state->random_seed_snapshot_accepted),
          "NVIDIA solve_cw accepted noisy polarization or published seed metadata");
  master_printf("nvidia_timestep: noisy capability rejections PASS\n");
}

static void test_noisy_profile(const char *mode) {
  const bool reseed = !strcmp(mode, "reseed");
  require(reseed || !strcmp(mode, "steady"), "unknown NVIDIA noisy profile mode");
  const grid_volume gv = vol3d(2.0, 2.0, 2.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 2);
  noisy_lorentzian_susceptibility noisy(0.03125, 0.73, 0.06);
  s.add_susceptibility(unit_value, E_stuff, noisy);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&s, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  set_random_seed(12345);
  gpu.advance(2);
  if (reseed) set_random_seed(67890);
  nvidia::testing::reset_transfer_accounting();
  const nvidia::memory_accounting before = nvidia::current_memory_accounting();
  gpu.advance(8);
  const nvidia::memory_accounting after = nvidia::current_memory_accounting();
  const nvidia::testing::transfer_accounting transfers =
      nvidia::testing::current_transfer_accounting();
  require(transfers.host_to_device_calls == size_t(reseed) &&
              transfers.host_to_device_bytes ==
                  (reseed ? sizeof(nvidia::noisy_seed_block) : size_t(0)) &&
              transfers.device_to_host_calls == 0 &&
              before.device_bytes_current == after.device_bytes_current &&
              before.pinned_bytes_current == after.pinned_bytes_current,
          "NVIDIA noisy profile transfer/allocation contract differs");
  master_printf("nvidia_timestep: noisy profile %s PASS\n", mode);
}

static void test_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;

}

static void test_compile_allocation_retry() {
  const grid_volume gv = vol2d(3.0, 3.0, 8.0);
  structure s(gv, isotropic_eps, pml(0.5), identity(), 2);
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields f(&s, options);
  f.use_real_fields();
  f.require_component(Ez);
  f.advance(1);
  BackendState *const live_state = f.backend_state;
  Executable *const live_executable = f.executable;
  const nvidia::memory_accounting before = nvidia::current_memory_accounting();
  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  const auto inject = [&](nvidia::testing::failure_point point, const char *operation) {
    nvidia::testing::fail_next(point);
    bool rejected = false;
    std::string reason;
    try {
      std::unique_ptr<Executable> unexpected(f.backend->compile(plan, *f.backend_state));
    }
    catch (const std::exception &error) {
      rejected = true;
      reason = error.what();
    }
    nvidia::testing::clear_failure();
    const nvidia::memory_accounting after = nvidia::current_memory_accounting();
    require(rejected && reason.find(operation) != std::string::npos &&
                f.backend_state == live_state && f.executable == live_executable &&
                !f.backend->is_poisoned() &&
                before.device_bytes_current == after.device_bytes_current &&
                before.pinned_bytes_current == after.pinned_bytes_current,
            "failed NVIDIA executable compilation changed the live epoch");
  };
  inject(nvidia::testing::failure_point::device_allocate, "cudaMalloc");
  inject(nvidia::testing::failure_point::host_to_device_copy,
         "cudaMemcpyAsync(host-to-device)");
  f.advance(1);
  require(f.backend_state == live_state && f.executable == live_executable,
          "ordinary advance did not reuse the epoch after compile failure");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  if (getenv("MEEP_NVIDIA_MATERIAL_MPI_NATIVE_ONLY")) {
    require(count_processors() == 2,
            "native material MPI validation requires exactly two ranks");
    test_native_absorber_initialization(precision_policy_kind::native);
    master_printf("nvidia_timestep: native material MPI PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_MATERIAL_MPI_VALIDATION_ONLY")) {
    test_material_collective_preflight();
    return 0;
  }
  if (getenv("MEEP_NVIDIA_TABLE_MPI_ONLY")) {
    test_native_table_mpi();
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_MATERIAL_GEOMETRY_CUDA_MPI_ONLY")) {
    test_native_geometry_cuda_mpi_initialization();
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_MATERIAL_GEOMETRY_MPI_ONLY")) {
    test_native_geometry_mpi();
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_GRAPH_MPI_ONLY")) {
    test_graph_collective_reconciliation();
    return 0;
  }
  require(count_processors() == 1, "nvidia_timestep is a single-rank test");
  if (getenv("MEEP_NVIDIA_GRAPH_ONLY")) {
    test_graph_execution_integration();
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_REQUIRE_NATIVE_SINGLE"))
    require(sizeof(realnum) == sizeof(float),
            "native-single validation was built with double realnum");
  if (getenv("MEEP_NVIDIA_COMPILE_RETRY_ONLY")) {
    test_compile_allocation_retry();
    master_printf("nvidia_timestep: compile retry checks PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_REBUILD_RETRY_ONLY")) {
    run_case("state-rebuild-retry", vol2d(2.0, 2.0, 8.0),
             precision_policy_kind::native, true, no_pml(), identity(), 2, NULL,
             1u << CONNECT_COPY, 1u << 0, 1u << 0, false, true);
    master_printf("nvidia_timestep: rebuild retry checks PASS\n");
    return 0;
  }
  const bool gyro_only = getenv("MEEP_NVIDIA_TIMESTEP_GYRO_ONLY") != NULL;
  const bool beta_only = getenv("MEEP_NVIDIA_TIMESTEP_BETA_ONLY") != NULL;
  const bool bfast_only = getenv("MEEP_NVIDIA_TIMESTEP_BFAST_ONLY") != NULL;
  const bool cylindrical_only = getenv("MEEP_NVIDIA_TIMESTEP_CYLINDRICAL_ONLY") != NULL;
  const bool magnetic_only = getenv("MEEP_NVIDIA_TIMESTEP_MAGNETIC_ONLY") != NULL;
  const bool material_only = getenv("MEEP_NVIDIA_TIMESTEP_MATERIAL_ONLY") != NULL;
  const bool flux_only = getenv("MEEP_NVIDIA_TIMESTEP_FLUX_ONLY") != NULL;
  const bool noisy_only = getenv("MEEP_NVIDIA_TIMESTEP_NOISY_ONLY") != NULL;
  const bool multilevel_only = getenv("MEEP_NVIDIA_TIMESTEP_MULTILEVEL_ONLY") != NULL;
  const bool host_custom_only = getenv("MEEP_NVIDIA_TIMESTEP_HOST_CUSTOM_ONLY") != NULL;
  const char *cylindrical_case = getenv("MEEP_NVIDIA_CYLINDRICAL_CASE");
  test_polarization_coefficient_rounding();
  test_polarization_storage_key_encoding();
  if (getenv("MEEP_NVIDIA_COEFFICIENTS_ONLY")) {
    master_printf("nvidia_timestep: coefficient checks PASS\n");
    return 0;
  }
  set_finite_check_mode(FiniteCheckMode::off);
  if (getenv("MEEP_NVIDIA_PR54_NATIVE_SMOKE")) {
    test_native_material_initialization(precision_policy_kind::native, true);
    master_printf("nvidia_timestep: PR5.4 native smoke PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_PR54_CALLBACK_ROUTES_ONLY")) {
    test_pr54_callback_routes();
    return 0;
  }
  if (getenv("MEEP_NVIDIA_PR54_ROUTE_MATRIX_ONLY")) {
    test_pr54_route_policy_matrix();
    return 0;
  }
  if (getenv("MEEP_NVIDIA_PR54_CLASSIFICATION_ONLY")) {
    test_pr54_classification_groups();
    return 0;
  }
  if (getenv("MEEP_NVIDIA_PR54_FAILURES_ONLY")) {
    test_pr54_post_callback_failure_matrix();
    return 0;
  }
  if (getenv("MEEP_NVIDIA_BFAST_REJECTIONS_ONLY")) {
    test_bfast_compile_rejections();
    master_printf("nvidia_timestep: BFAST rejection checks PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_CYLINDRICAL_REJECTIONS_ONLY")) {
    test_cylindrical_compile_rejections();
    master_printf("nvidia_timestep: cylindrical rejection checks PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_CYLINDRICAL_CHANGE_M_ONLY")) {
    test_nvidia_cylindrical_change_m(precision_policy_kind::native);
    master_printf("nvidia_timestep: cylindrical change_m checks PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_MAGNETIC_REJECTIONS_ONLY")) {
    test_magnetic_compile_rejections();
    return 0;
  }
  if (getenv("MEEP_NVIDIA_MAGNETIC_LIFECYCLE_ONLY")) {
    test_magnetic_pre_step_and_recompile();
    test_magnetic_historical_host_backups();
    return 0;
  }
  if (getenv("MEEP_NVIDIA_MATERIAL_LIFECYCLE_ONLY")) {
    test_material_copy_after_compile_rejected();
    test_material_cpu_setup_to_nvidia();
    return 0;
  }
  if (getenv("MEEP_NVIDIA_TABLE_PREFLIGHT_ONLY")) {
    test_native_table_schema_preflight();
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_TABLE_REPLACEMENT_ONLY")) {
    test_native_table_semantic_replacement();
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_TABLE_DIMENSIONS_ONLY")) {
    test_native_table_dimension_matrix(precision_policy_kind::native);
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_MATERIAL_RECIPE_ONLY")) {
    const precision_policy_kind policies[] = {precision_policy_kind::native,
                                              precision_policy_kind::mixed,
                                              precision_policy_kind::f32};
    for (precision_policy_kind policy : policies) {
      test_material_recipe_owned_upload(policy);
      test_material_value_refresh_preserves_host_edit(policy);
    }
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_MATERIAL_NATIVE_INIT_ONLY")) {
    const precision_policy_kind policies[] = {precision_policy_kind::native,
                                              precision_policy_kind::mixed,
                                              precision_policy_kind::f32};
    for (precision_policy_kind policy : policies) {
      test_native_material_initialization(policy, true);
      test_native_material_initialization(policy, false);
      test_native_absorber_initialization(policy);
      test_native_table_materials(policy, true);
      test_native_table_materials(policy, false);
    }
    test_native_material_initialization_retry();
    test_native_table_retry();
    test_native_table_semantic_replacement();
    test_native_table_preupload_rejection();
    test_native_table_schema_preflight();
    test_native_table_periodicity_invariance();
    test_native_table_dimension_matrix(precision_policy_kind::native);
    test_native_grid_damping_only();
    test_native_dimension_and_chi_pair("d2", vol2d(1.5, 1.25, 6.0), true);
    test_native_dimension_and_chi_pair("cyl", volcyl(1.5, 1.25, 6.0), false);
    test_native_dimension_pml("d2", vol2d(1.5, 1.25, 6.0));
    test_native_dimension_pml("cyl", volcyl(1.5, 1.25, 6.0));
    test_native_table_dimension_pml("d2", vol2d(1.5, 1.25, 6.0));
    test_native_table_dimension_pml("cyl", volcyl(1.5, 1.25, 6.0));
    test_native_pml_rounding_association();
    test_native_perfect_metal_signed_zero();
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_MATERIAL_GEOMETRY_ADMISSION_ONLY")) {
    test_native_geometry_memory_admission();
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (const char *geometry_case = getenv("MEEP_NVIDIA_MATERIAL_GEOMETRY_CASE")) {
    if (!strcmp(geometry_case, "bulk")) test_driven_material_geometry(driven_bulk);
    else if (!strcmp(geometry_case, "analytic")) test_driven_material_geometry(driven_analytic);
    else if (!strcmp(geometry_case, "patch")) test_driven_material_geometry(driven_patch);
    else throw std::invalid_argument("unknown NVIDIA material geometry case");
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_MATERIAL_GEOMETRY_ONLY")) {
    test_native_material_geometry(precision_policy_kind::native);
    test_native_material_geometry(precision_policy_kind::mixed);
    test_native_material_geometry(precision_policy_kind::f32);
    test_driven_material_geometry(driven_bulk);
    test_driven_material_geometry(driven_analytic);
    test_driven_material_geometry(driven_patch);
    test_native_mesh_geometry();
    test_native_geometry_shapes();
    test_native_geometry_grid(true, false);
    test_native_geometry_grid(true, true);
    test_native_geometry_grid(false, true);
    test_native_object_file_geometry();
    test_native_geometry_grid_absorber();
    test_native_geometry_grid_reducers();
    test_native_geometry_retry();
    test_native_geometry_tensor_preflight();
    test_native_geometry_memory_admission();
    test_native_geometry_periodic();
    test_native_geometry_dimensions();
    test_native_geometry_field_growth();
    test_native_geometry_skew_lattice();
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_MATERIAL_REJECTIONS_ONLY")) {
    test_material_compile_rejections();
    return 0;
  }
  if (getenv("MEEP_NVIDIA_TIMESTEP_NOISY_LIFECYCLE_ONLY")) {
    test_noisy_lifecycle();
    test_noisy_rebuild_lifecycle();
    master_printf("nvidia_timestep: noisy lifecycle checks PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_TIMESTEP_NOISY_REJECTIONS_ONLY")) {
    test_noisy_compile_rejections();
    test_noisy_capability_rejections();
    master_printf("nvidia_timestep: noisy rejection checks PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_TIMESTEP_MULTILEVEL_LIFECYCLE_ONLY")) {
    test_multilevel_lifecycle();
    return 0;
  }
  if (getenv("MEEP_NVIDIA_TIMESTEP_MULTILEVEL_REJECTIONS_ONLY")) {
    test_multilevel_compile_rejections();
    test_multilevel_capability_rejections();
    return 0;
  }
  if (const char *mode = getenv("MEEP_NVIDIA_TIMESTEP_NOISY_PROFILE")) {
    test_noisy_profile(mode);
    return 0;
  }
  const grid_volume gv2 = vol2d(3.0, 2.0, 8.0);
  const grid_volume gv3 = vol3d(2.0, 2.0, 2.0, 5.0);
  const vec bloch2(0.17, 0.11);
  const vec bloch3(0.11, 0.07, 0.05);
  const boundary_region xy_pml = pml(0.4, X) + pml(0.4, Y);
  linear_anisotropic_material one_offdiagonal(false);
  linear_anisotropic_material two_offdiagonals(true);
  linear_anisotropic_material magnetic_two_offdiagonals(true, true);
  const precision_policy_kind policies[] = {
      precision_policy_kind::native, precision_policy_kind::mixed, precision_policy_kind::f32};
  for (size_t p = 0; p < sizeof(policies) / sizeof(policies[0]); ++p) {
    if (host_custom_only) {
      if (policies[p] == precision_policy_kind::native) {
        run_host_custom_fallback_case(policies[p], false);
        run_host_custom_fallback_case(policies[p], true);
      }
      else
        test_host_custom_precision_rejection(policies[p]);
      continue;
    }
    if (multilevel_only) {
      run_multilevel_case("d1-real-e-l2t1", policies[p], vol1d(2.4, 7.0), true, false,
                          false);
      run_multilevel_case("d2-real-e-l3t2", policies[p], vol2d(2.4, 2.0, 7.0), true, false,
                          true);
      run_multilevel_case("d3-complex-h-l3t2", policies[p],
                          vol3d(1.8, 1.6, 1.4, 5.0), false, true, true);
      run_multilevel_multitile_pml_case(policies[p]);
      continue;
    }
    if (noisy_only) {
      run_noisy_replay_case(policies[p]);
      run_noisy_composition_case(policies[p]);
      test_noisy_sigma_diagnostics(policies[p]);
      continue;
    }
    if (flux_only) {
      run_legacy_flux_case("real-two-monitor", policies[p], false);
      run_legacy_flux_case("complex-bloch-two-monitor", policies[p], true);
      run_legacy_flux_symmetry_case(policies[p]);
      run_legacy_flux_material_case(policies[p], false, true);
      run_legacy_flux_material_case(policies[p], true, false);
      run_cylindrical_flux_case(policies[p]);
      continue;
    }
    if (material_only) {
      run_material_phase_case("real-target-conductivity", policies[p], true, false);
      run_material_phase_case("complex-current-conductivity", policies[p], false, true);
      continue;
    }
    if (magnetic_only) {
      run_magnetic_sync_case("real-pml-conductive-bfast", policies[p], true, true);
      run_magnetic_sync_case("complex-pml-conductive", policies[p], false, false);
      continue;
    }
    if (cylindrical_only) {
      if (!cylindrical_case || !strcmp(cylindrical_case, "m0-real-pml-conductive"))
        run_cylindrical_case("m0-real-pml-conductive", policies[p], 0.0, true, true, true, true,
                             false, false);
      if (!cylindrical_case || !strcmp(cylindrical_case, "m0-complex"))
        run_cylindrical_case("m0-complex", policies[p], 0.0, false, false, false, true, false,
                             false);
      if (!cylindrical_case || !strcmp(cylindrical_case, "m-plus1-pml-conductive"))
        run_cylindrical_case("m-plus1-pml-conductive", policies[p], +1.0, false, true, true,
                             true, false, false);
      if (cylindrical_case && !strcmp(cylindrical_case, "m-plus1-pml-bfast"))
        run_cylindrical_case("m-plus1-pml-bfast", policies[p], +1.0, false, true, false, true,
                             false, true);
      if (!cylindrical_case || !strcmp(cylindrical_case, "m-minus1"))
        run_cylindrical_case("m-minus1", policies[p], -1.0, false, false, false, true, false,
                             false);
      if (!cylindrical_case || !strcmp(cylindrical_case, "m-half-annular-conductive"))
        run_cylindrical_case("m-half-annular-conductive", policies[p], +0.5, false, false, true,
                             true, true, false);
      if (!cylindrical_case || !strcmp(cylindrical_case, "m-plus3-default-zero"))
        run_cylindrical_case("m-plus3-default-zero", policies[p], +3.0, false, false, false,
                             true, false, false);
      if (!cylindrical_case || !strcmp(cylindrical_case, "m-minus3-accurate-bfast"))
        run_cylindrical_case("m-minus3-accurate-bfast", policies[p], -3.0, false, true, false,
                             false, false, true, 0.25);
      if (cylindrical_case && !strcmp(cylindrical_case, "m-minus3-accurate-pml"))
        run_cylindrical_case("m-minus3-accurate-pml", policies[p], -3.0, false, true, false,
                             false, false, false, 0.25);
      if (cylindrical_case && !strcmp(cylindrical_case, "m-minus3-accurate-bfast-nopml"))
        run_cylindrical_case("m-minus3-accurate-bfast-nopml", policies[p], -3.0, false, false,
                             false, false, false, true, 0.25);
      continue;
    }
    if (!beta_only && !bfast_only) {
      run_gyrotropic_case("lorentz-real-e-copy", policies[p], true, GYROTROPIC_LORENTZIAN,
                          false, false, NULL, false, 1u << CONNECT_COPY);
      run_gyrotropic_case("lorentz-complex-e-pml-phase", policies[p], false,
                          GYROTROPIC_LORENTZIAN, false, true, &bloch3, false,
                          (1u << CONNECT_COPY) | (1u << CONNECT_PHASE));
      run_gyrotropic_case("drude-real-h-negate", policies[p], true, GYROTROPIC_DRUDE, true,
                          false, NULL, true, 1u << CONNECT_NEGATE);
      run_gyrotropic_case("drude-complex-h-pml-phase", policies[p], false, GYROTROPIC_DRUDE,
                          true, true, &bloch3, false,
                          (1u << CONNECT_COPY) | (1u << CONNECT_PHASE));
      run_gyrotropic_case("saturated-real-e-copy", policies[p], true, GYROTROPIC_SATURATED,
                          false, false, NULL, false, 1u << CONNECT_COPY);
      run_gyrotropic_case("saturated-complex-h-pml-phase", policies[p], false,
                          GYROTROPIC_SATURATED, true, true, &bloch3, false,
                          (1u << CONNECT_COPY) | (1u << CONNECT_PHASE));
    }
    if (!gyro_only && !bfast_only) {
      run_beta_case("real-positive", policies[p], true, +0.17, false, false, 2);
      run_beta_case("real-negative-pml", policies[p], true, -0.17, true, false, 4);
      run_beta_case("complex-positive-conductive", policies[p], false, +0.17, false, true, 2);
      run_beta_case("complex-negative-pml-conductive", policies[p], false, -0.17, true, true, 4);
    }
    if (!gyro_only && !beta_only) {
      const std::vector<double> positive_k{0.17, 0.11, 0.07};
      const std::vector<double> negative_k{-0.17, -0.11, -0.07};
      const std::vector<double> sparse_k{0.17, 0.0, 0.0};
      run_bfast_case("d1-one-source", vol1d(3.0, 8.0), policies[p], true, positive_k,
                     false, false, 2);
      run_bfast_case("d3-two-source-copy", gv3, policies[p], true, positive_k, false,
                     false, 4);
      run_bfast_case("d3-negative-complex-phase", gv3, policies[p], false, negative_k,
                     false, false, 4, &bloch3);
      run_bfast_case("d2-pml", gv2, policies[p], true, positive_k, true, false, 2);
      run_bfast_case("d2-pml-conductive-zero-row", gv2, policies[p], false, sparse_k, true,
                     true, 4, &bloch2);
      run_bfast_case("d2-beta-composed", gv2, policies[p], false, positive_k, false,
                     false, 2, &bloch2, 0.17);
    }
    if (gyro_only || beta_only || bfast_only || magnetic_only || material_only || flux_only ||
        noisy_only || multilevel_only || host_custom_only)
      continue;
    run_dispersion_case("real-lorentz-copy", policies[p], true, false, false, false, false, 2,
                        NULL, 1u << CONNECT_COPY);
    run_dispersion_case("complex-lorentz-pml-phase", policies[p], false, false, false, false,
                        true, 4, &bloch3, (1u << CONNECT_COPY) | (1u << CONNECT_PHASE));
    run_dispersion_case("complex-drude-anisotropic-pml", policies[p], false, true, true, false,
                        true, 2, &bloch3, 1u << CONNECT_COPY);
    run_dispersion_case("complex-magnetic-lorentz-pml", policies[p], false, false, true, true,
                        true, 2, &bloch3, 1u << CONNECT_COPY);
    run_dispersion_case("multiple-integrated-source-chi3", policies[p], true, false, false,
                        false, false, 2, NULL, 1u << CONNECT_COPY, true, true, true);
    run_dispersion_case("real-lorentz-negate", policies[p], true, false, false, false, false, 2,
                        NULL, 1u << CONNECT_NEGATE, false, false, false, true);
    run_dispersion_case("complex-magnetic-lorentz-negate", policies[p], false, false, false, true,
                        false, 2, NULL, 1u << CONNECT_NEGATE, false, false, false, true);
    run_source_case("real-point-source", policies[p], true, false);
    run_source_case("complex-conductive-point-source", policies[p], false, true);
    run_source_case("complex-continuous-point-source", policies[p], false, false, false, NULL,
                    NULL, true);
    run_source_case("complex-volume-plane-source", policies[p], false, false, false, NULL, NULL,
                    false, true);
    run_source_case("real-integrated-point-source", policies[p], true, false, true);
    run_source_case("complex-integrated-anisotropic-pml", policies[p], false, false, true,
                    &two_offdiagonals, &xy_pml);
    run_source_case("complex-integrated-anisotropic-chi3-pml", policies[p], false, false, true,
                    &two_offdiagonals, &xy_pml, false, false, true);
    run_custom_source_case("custom-point-source", policies[p], false);
    run_custom_source_case("custom-integrated-point-source", policies[p], true);
    run_case("real-copy", gv2, policies[p], true, no_pml(), identity(), 4, NULL, 1u << CONNECT_COPY,
             1u << 0, 1u << 0, false, true);
    run_case("complex-phase", gv2, policies[p], false, no_pml(), identity(), 4, &bloch2,
             (1u << CONNECT_COPY) | (1u << CONNECT_PHASE), 1u << 0, 1u << 0, false, false);
    run_case("complex-negate", gv2, policies[p], false, no_pml(), -mirror(Y, gv2), 2, NULL,
             1u << CONNECT_NEGATE, 1u << 0, 1u << 0, false, false);
    run_case("real-conductivity", gv3, policies[p], true, no_pml(), identity(), 2, NULL,
             1u << CONNECT_COPY, 1u << 4, 1u << 0, true, false);
    run_case("real-pml-conductivity", gv3, policies[p], true, xy_pml, identity(), 2, NULL,
             1u << CONNECT_COPY, (1u << 5) | (1u << 6) | (1u << 7), 1u << 3, true, false);
    run_case("complex-pml", gv3, policies[p], false, xy_pml, identity(), 2, &bloch3,
             (1u << CONNECT_COPY) | (1u << CONNECT_PHASE), (1u << 1) | (1u << 2) | (1u << 3),
             1u << 3, false, false);
    run_case("real-anisotropic-2x2", gv3, policies[p], true, no_pml(), identity(), 2, NULL,
             1u << CONNECT_COPY, 1u << 0, 1u << 1, false, false, &one_offdiagonal);
    run_case("complex-anisotropic-3x3-pml", gv3, policies[p], false, xy_pml, identity(), 2, &bloch3,
             (1u << CONNECT_COPY) | (1u << CONNECT_PHASE), (1u << 1) | (1u << 2) | (1u << 3),
             1u << 5, false, false, &two_offdiagonals);
    run_case("real-diagonal-chi2", gv3, policies[p], true, no_pml(), identity(), 2, NULL,
             1u << CONNECT_COPY, 1u << 0, 1u << 0, false, false, NULL, true, false);
    run_case("complex-diagonal-chi3", gv3, policies[p], false, no_pml(), identity(), 2, &bloch3,
             (1u << CONNECT_COPY) | (1u << CONNECT_PHASE), 1u << 0, 1u << 0, false, false,
             NULL, false, true);
    run_case("complex-anisotropic-chi2-chi3-pml", gv3, policies[p], false, xy_pml, identity(), 2,
             &bloch3, (1u << CONNECT_COPY) | (1u << CONNECT_PHASE),
             (1u << 1) | (1u << 2) | (1u << 3), 1u << 5, false, false,
             &two_offdiagonals, true, true);
    run_case("complex-magnetic-anisotropic-chi2-chi3-pml", gv3, policies[p], false, xy_pml,
             identity(), 2, &bloch3, (1u << CONNECT_COPY) | (1u << CONNECT_PHASE),
             (1u << 1) | (1u << 2) | (1u << 3), 1u << 5, false, false,
             &magnetic_two_offdiagonals, true, true, Hz);
    test_finite_diagnostics(policies[p]);
  }
  if (noisy_only) {
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (host_custom_only) {
    test_host_custom_opaque_halo_ownership();
    test_host_custom_complex_eh_composition();
    test_host_custom_compile_rejections();
    test_host_custom_postdispatch_poison();
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (multilevel_only) {
    test_multilevel_zero_row();
    test_multilevel_composition();
    test_multilevel_compile_rejections();
    test_multilevel_capability_rejections();
    test_multilevel_lifecycle();
    test_multilevel_backend_reselection();
    test_multilevel_gamma_inv_host_authority();
    master_printf("nvidia_timestep: PASS\n");
    return 0;
  }
  if (magnetic_only) {
    test_magnetic_pre_step_and_recompile();
    test_magnetic_historical_host_backups();
    test_magnetic_compile_rejections();
  }
  if (material_only) {
    test_material_copy_after_compile_rejected();
    test_material_cpu_setup_to_nvidia();
    test_material_compile_rejections();
  }
  if (flux_only) {
    test_legacy_flux_missing_components();
    test_legacy_flux_add_remove();
    test_legacy_flux_postlaunch_poison();
    test_legacy_flux_compile_rejections();
  }
  set_finite_check_mode(FiniteCheckMode::off);
  if (cylindrical_only && !cylindrical_case) {
    test_nvidia_cylindrical_change_m(precision_policy_kind::native);
    test_cylindrical_compile_rejections();
  }
  else if (!material_only && !flux_only) {
    if (!beta_only && !bfast_only) test_gyrotropic_compile_rejections();
    if (!gyro_only && !bfast_only) test_beta_compile_rejections();
    if (!gyro_only && !beta_only) test_bfast_compile_rejections();
  }
  if (!gyro_only && !beta_only && !bfast_only && !cylindrical_only && !magnetic_only &&
      !material_only && !flux_only) {
    test_compile_allocation_retry();
    test_rejections();
    test_nonlinear_compile_rejections();
    test_polarization_compile_rejections();
  }
  master_printf("nvidia_timestep: PASS\n");
  return 0;
}
