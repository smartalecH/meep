/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include <math.h>
#include <stdio.h>
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

static double isotropic_eps(const vec &p) { return p.x() < 0.0 ? 2.0 : 3.0; }
static double uniform_conductivity(const vec &) { return 0.17; }
static double unit_value(const vec &) { return 1.0; }

class linear_anisotropic_material : public material_function {
public:
  explicit linear_anisotropic_material(bool full) : full_(full) {}

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

static void initialize_fields(fields &cpu, fields &gpu, bool f32) {
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
    if (is_valid(cpu_spec.alias_of) || cpu_spec.role != array_role::field ||
        cpu_spec.element_type != ElementType::realnum_value)
      continue;

    std::vector<realnum> values(cpu_spec.elements);
    for (size_t j = 0; j < values.size(); ++j) {
      values[j] = initial_value(i, j);
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
    if (is_valid(spec.alias_of) || spec.role != array_role::field ||
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
                                bool expect_cross_copy) {
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
      }
      require(have_primary_copy, "integrated source has no prepared primary copy");
      require(have_cross_copy == expect_cross_copy,
              "integrated source cross-copy coverage differs from expectation");
    }
  }
  require(evaluations == 4, "ordinary source plan did not schedule four scalar evaluations");
  require(source_spans == sources.sources.size(), "point-source descriptor span was not unique");
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
                            bool plane_source = false) {
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
  require_source_plan(cpu, conductivity, integrated, anisotropic && integrated);
  require_source_plan(gpu, conductivity, integrated, anisotropic && integrated);

  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed);
  const double tolerance = narrowed ? 2e-5 : 2e-13;
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
  const double tolerance = narrowed ? 2e-5 : 2e-13;
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
                     bool check_lifecycle, material_function *material = NULL) {
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

  fields cpu(cpu_structure.get());
  if (real_fields)
    cpu.use_real_fields();
  else if (bloch)
    cpu.use_bloch(*bloch);
  cpu.require_component(Ez);
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
  gpu.init_backend();
  require_halo_phases(gpu, required_halo_phases);
  const StepPlan prepared = build_step_plan(gpu, StepProgram::ordinary);
  require_physics_variants(prepared, required_curl_combinations,
                           required_constitutive_combinations);

  const bool narrowed = policy != precision_policy_kind::native;
  if (narrowed) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, narrowed);

  const double tolerance = narrowed ? 2e-5 : 2e-13;
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
    s.set_chi3(unit_value);
    fields f(&s, options);
    f.use_real_fields();
    f.require_component(Ez);
    require_advance_rejected(f, "nonlinearity");
  }
  {
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    s.add_susceptibility(unit_value, E_stuff, lorentzian_susceptibility(1.1, 0.05));
    fields f(&s, options);
    f.use_real_fields();
    f.require_component(Ez);
    require_advance_rejected(f, "dispersion");
  }
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  require(count_processors() == 1, "nvidia_timestep is a single-rank test");
  set_finite_check_mode(FiniteCheckMode::off);
  const grid_volume gv2 = vol2d(3.0, 2.0, 8.0);
  const grid_volume gv3 = vol3d(2.0, 2.0, 2.0, 5.0);
  const vec bloch2(0.17, 0.11);
  const vec bloch3(0.11, 0.07, 0.05);
  const boundary_region xy_pml = pml(0.4, X) + pml(0.4, Y);
  linear_anisotropic_material one_offdiagonal(false);
  linear_anisotropic_material two_offdiagonals(true);
  const precision_policy_kind policies[] = {
      precision_policy_kind::native, precision_policy_kind::mixed, precision_policy_kind::f32};
  for (size_t p = 0; p < sizeof(policies) / sizeof(policies[0]); ++p) {
    run_source_case("real-point-source", policies[p], true, false);
    run_source_case("complex-conductive-point-source", policies[p], false, true);
    run_source_case("complex-continuous-point-source", policies[p], false, false, false, NULL,
                    NULL, true);
    run_source_case("complex-volume-plane-source", policies[p], false, false, false, NULL, NULL,
                    false, true);
    run_source_case("real-integrated-point-source", policies[p], true, false, true);
    run_source_case("complex-integrated-anisotropic-pml", policies[p], false, false, true,
                    &two_offdiagonals, &xy_pml);
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
    test_finite_diagnostics(policies[p]);
  }
  set_finite_check_mode(FiniteCheckMode::off);
  test_rejections();
  master_printf("nvidia_timestep: PASS\n");
  return 0;
}
