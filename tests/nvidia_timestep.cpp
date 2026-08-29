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
#include "backend/nvidia/runtime.hpp"
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
  require(count_processors() == 1, "nvidia_timestep is a single-rank test");
  if (getenv("MEEP_NVIDIA_REQUIRE_NATIVE_SINGLE"))
    require(sizeof(realnum) == sizeof(float),
            "native-single validation was built with double realnum");
  if (getenv("MEEP_NVIDIA_COMPILE_RETRY_ONLY")) {
    test_compile_allocation_retry();
    master_printf("nvidia_timestep: compile retry checks PASS\n");
    return 0;
  }
  const bool gyro_only = getenv("MEEP_NVIDIA_TIMESTEP_GYRO_ONLY") != NULL;
  test_polarization_coefficient_rounding();
  if (getenv("MEEP_NVIDIA_COEFFICIENTS_ONLY")) {
    master_printf("nvidia_timestep: coefficient checks PASS\n");
    return 0;
  }
  set_finite_check_mode(FiniteCheckMode::off);
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
    if (gyro_only) continue;
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
  set_finite_check_mode(FiniteCheckMode::off);
  test_gyrotropic_compile_rejections();
  if (!gyro_only) {
    test_compile_allocation_retry();
    test_rejections();
    test_nonlinear_compile_rejections();
    test_polarization_compile_rejections();
  }
  master_printf("nvidia_timestep: PASS\n");
  return 0;
}
