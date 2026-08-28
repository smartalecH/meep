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

#include <stdexcept>
#include <string>
#include <vector>

#include <meep.hpp>

#include "backend/backend.hpp"
#include "backend/diagnostics.hpp"
#include "backend/halo_plan.hpp"
#include "backend/lifecycle.hpp"
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

class offdiagonal_material : public material_function {
public:
  void eff_chi1inv_row(component c, double row[3], const volume &, double, int) override {
    row[0] = row[1] = row[2] = 0.0;
    const int d = component_index(c);
    row[d] = 0.5;
    row[(d + 1) % 3] = 0.05;
  }
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
                                     bool require_constitutive_pml) {
  unsigned int observed_curl_combinations = 0;
  bool observed_constitutive_pml = false;
  for (size_t i = 0; i < plan.db_updates.size(); ++i) {
    const uint32_t variants = plan.db_updates[i].region.variant_key;
    const unsigned int combination = unsigned((variants & curl_has_pml) != 0) |
                                     (unsigned((variants & curl_has_pml_aux) != 0) << 1) |
                                     (unsigned((variants & curl_has_conductivity) != 0) << 2);
    observed_curl_combinations |= 1u << combination;
  }
  for (size_t i = 0; i < plan.eh_updates.size(); ++i)
    observed_constitutive_pml |=
        (plan.eh_updates[i].region.variant_key & constitutive_has_pml) != 0;
  require((observed_curl_combinations & required_curl_combinations) == required_curl_combinations,
          "required NVIDIA curl PML/conductivity combination was not prepared");
  require(!require_constitutive_pml || observed_constitutive_pml,
          "required diagonal constitutive PML variant was not prepared");
}

static void set_uniform_conductivity(structure &s) {
  s.set_conductivity(Bx, uniform_conductivity);
  s.set_conductivity(By, uniform_conductivity);
  s.set_conductivity(Bz, uniform_conductivity);
  s.set_conductivity(Dx, uniform_conductivity);
  s.set_conductivity(Dy, uniform_conductivity);
  s.set_conductivity(Dz, uniform_conductivity);
}

static void run_case(const char *name, const grid_volume &gv, precision_policy_kind policy,
                     bool real_fields, const boundary_region &boundaries,
                     const symmetry &symmetries, int chunks, const vec *bloch,
                     unsigned int required_halo_phases, unsigned int required_curl_combinations,
                     bool require_constitutive_pml, bool conductivity, bool check_lifecycle) {
  structure cpu_structure(gv, isotropic_eps, boundaries, symmetries, chunks);
  structure gpu_structure(gv, isotropic_eps, boundaries, symmetries, chunks);
  if (conductivity) {
    set_uniform_conductivity(cpu_structure);
    set_uniform_conductivity(gpu_structure);
  }

  fields cpu(&cpu_structure);
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
  fields gpu(&gpu_structure, options);
  if (real_fields)
    gpu.use_real_fields();
  else if (bloch)
    gpu.use_bloch(*bloch);
  gpu.require_component(Ez);
  gpu.init_backend();
  require_halo_phases(gpu, required_halo_phases);
  const StepPlan prepared = build_step_plan(gpu, StepProgram::ordinary);
  require_physics_variants(prepared, required_curl_combinations, require_constitutive_pml);

  const bool f32 = policy == precision_policy_kind::f32;
  if (f32) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, f32);

  const double tolerance = f32 ? 2e-5 : 2e-13;
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
  master_printf("nvidia_timestep: %s/%s PASS\n", name, f32 ? "f32" : "native");
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

static void test_rejections() {
  const grid_volume gv = vol2d(2.0, 2.0, 6.0);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy_kind::native;

  {
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    fields f(&s, options);
    f.use_real_fields();
    gaussian_src_time src(0.3, 0.1);
    f.add_point_source(Ez, src, vec(0.1, 0.1));
    require_advance_rejected(f, "does not support sources");
  }
  {
    offdiagonal_material material;
    structure s(gv, material, no_pml(), identity(), 1);
    fields f(&s, options);
    f.use_real_fields();
    f.require_component(Ez);
    require_advance_rejected(f, "anisotropy");
  }
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
  {
    structure s(gv, isotropic_eps, no_pml(), identity(), 1);
    options.precision = precision_policy_kind::mixed;
    fields f(&s, options);
    f.use_real_fields();
    f.require_component(Ez);
    require_advance_rejected(f, "precision=native and precision=f32 only");
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
  const precision_policy_kind policies[] = {precision_policy_kind::native,
                                            precision_policy_kind::f32};
  for (size_t p = 0; p < sizeof(policies) / sizeof(policies[0]); ++p) {
    run_case("real-copy", gv2, policies[p], true, no_pml(), identity(), 4, NULL, 1u << CONNECT_COPY,
             1u << 0, false, false, true);
    run_case("complex-phase", gv2, policies[p], false, no_pml(), identity(), 4, &bloch2,
             (1u << CONNECT_COPY) | (1u << CONNECT_PHASE), 1u << 0, false, false, false);
    run_case("complex-negate", gv2, policies[p], false, no_pml(), -mirror(Y, gv2), 2, NULL,
             1u << CONNECT_NEGATE, 1u << 0, false, false, false);
    run_case("real-conductivity", gv3, policies[p], true, no_pml(), identity(), 2, NULL,
             1u << CONNECT_COPY, 1u << 4, false, true, false);
    run_case("real-pml-conductivity", gv3, policies[p], true, xy_pml, identity(), 2, NULL,
             1u << CONNECT_COPY, (1u << 5) | (1u << 6) | (1u << 7), true, true, false);
    run_case("complex-pml", gv3, policies[p], false, xy_pml, identity(), 2, &bloch3,
             (1u << CONNECT_COPY) | (1u << CONNECT_PHASE), (1u << 1) | (1u << 2) | (1u << 3), true,
             false, false);
  }
  test_rejections();
  master_printf("nvidia_timestep: PASS\n");
  return 0;
}
