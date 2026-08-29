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

#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <meep.hpp>

#include "backend/backend.hpp"
#include "backend/descriptors.hpp"
#include "backend/diagnostics.hpp"
#include "backend/halo_plan.hpp"
#include "backend/lifecycle.hpp"
#include "backend/nvidia/nvidia_backend.hpp"
#include "backend/precision.hpp"
#include "backend/step_plan.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

using namespace meep;

static void require(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message);
    meep::abort("nvidia_timestep failed");
  }
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

static double isotropic_eps(const vec &p) { return p.x() < 0.0 ? 2.0 : 3.0; }
static double uniform_conductivity(const vec &) { return 0.17; }
static double unit_value(const vec &) { return 1.0; }
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
      if (error > tolerance * scale) {
        const StorageKey &key = cpu.array_catalog->key(id);
        fprintf(stderr,
                "array %zu (%s,c=%d,cmp=%d,aux=%d) element %zu differs: cpu=%.17g "
                "nvidia=%.17g error=%.3g tol=%.3g\n",
                i, array_kind_name(array_kind(key.kind)), key.component_, key.cmp, key.aux, j,
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
      if (error > tolerance * scale) {
        fprintf(stderr,
                "live array (%s,c=%d,cmp=%d,aux=%d) element %zu differs: cpu=%.17g "
                "nvidia=%.17g error=%.3g tol=%.3g\n",
                array_kind_name(array_kind(key.kind)), key.component_, key.cmp, key.aux, j,
                double(expected[j]), double(observed[j]), error, tolerance * scale);
        meep::abort("NVIDIA live timestep state differs from CPU");
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

static void require_advance_rejected(fields &f, const char *expected) {
  bool rejected = false;
  try {
    f.advance(1);
  }
  catch (const std::runtime_error &error) {
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

static void run_finite_diagnostic_case(const char *name, precision_policy_kind policy,
                                       FiniteCheckMode mode, bool poison, realnum value,
                                       bool expect_rejection, int expected_step) {
  set_finite_check_mode(mode);
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure s(gv, isotropic_eps, no_pml(), identity(), 1);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields f(&s, options);
  f.use_real_fields();
  f.require_component(Ez);
  f.init_backend();

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

static void test_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;

  {
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    s.add_susceptibility(unit_value, E_stuff,
                         noisy_lorentzian_susceptibility(0.01, 1.1, 0.05));
    fields f(&s, options);
    f.use_real_fields();
    f.require_component(Ez);
    require_advance_rejected(f, "noisy_lorentzian");
  }
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  require(count_processors() == 1, "nvidia_timestep is a single-rank test");
  if (getenv("MEEP_NVIDIA_REQUIRE_NATIVE_SINGLE"))
    require(sizeof(realnum) == sizeof(float),
            "native-single validation was built with double realnum");
  const bool gyro_only = getenv("MEEP_NVIDIA_TIMESTEP_GYRO_ONLY") != NULL;
  const bool beta_only = getenv("MEEP_NVIDIA_TIMESTEP_BETA_ONLY") != NULL;
  const bool bfast_only = getenv("MEEP_NVIDIA_TIMESTEP_BFAST_ONLY") != NULL;
  const bool cylindrical_only = getenv("MEEP_NVIDIA_TIMESTEP_CYLINDRICAL_ONLY") != NULL;
  const bool magnetic_only = getenv("MEEP_NVIDIA_TIMESTEP_MAGNETIC_ONLY") != NULL;
  const char *cylindrical_case = getenv("MEEP_NVIDIA_CYLINDRICAL_CASE");
  test_polarization_coefficient_rounding();
  if (getenv("MEEP_NVIDIA_COEFFICIENTS_ONLY")) {
    master_printf("nvidia_timestep: coefficient checks PASS\n");
    return 0;
  }
  set_finite_check_mode(FiniteCheckMode::off);
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
    if (gyro_only || beta_only || bfast_only || magnetic_only) continue;
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
  if (magnetic_only) {
    test_magnetic_pre_step_and_recompile();
    test_magnetic_historical_host_backups();
    test_magnetic_compile_rejections();
  }
  set_finite_check_mode(FiniteCheckMode::off);
  if (cylindrical_only && !cylindrical_case) {
    test_nvidia_cylindrical_change_m(precision_policy_kind::native);
    test_cylindrical_compile_rejections();
  }
  else {
    if (!beta_only && !bfast_only) test_gyrotropic_compile_rejections();
    if (!gyro_only && !bfast_only) test_beta_compile_rejections();
    if (!gyro_only && !beta_only) test_bfast_compile_rejections();
  }
  if (!gyro_only && !beta_only && !bfast_only && !cylindrical_only && !magnetic_only) {
    test_rejections();
    test_nonlinear_compile_rejections();
    test_polarization_compile_rejections();
  }
  master_printf("nvidia_timestep: PASS\n");
  return 0;
}
