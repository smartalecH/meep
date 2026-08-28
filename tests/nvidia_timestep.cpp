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

#include <vector>

#include <meep.hpp>

#include "backend/backend.hpp"
#include "backend/diagnostics.hpp"
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

static realnum initial_value(size_t array, size_t element) {
  return realnum(0.02 * sin(double(17 * array + element)) +
                 0.01 * cos(double(3 * array + 5 * element)));
}

static void round_real_arrays(CpuArrayCatalog &catalog) {
  for (size_t i = 0; i < catalog.size(); ++i) {
    const ArraySpec &spec = catalog.spec(ArrayId{uint32_t(i)});
    if (is_valid(spec.alias_of) || spec.element_type != ElementType::realnum_value) continue;
    realnum *values = catalog.resolve<realnum>(spec.id);
    for (size_t j = 0; j < spec.elements; ++j) values[j] = realnum(float(values[j]));
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

static void run_policy(precision_policy_kind policy) {
  const grid_volume gv = vol2d(3.0, 2.0, 8.0);
  structure cpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);
  structure gpu_structure(gv, isotropic_eps, no_pml(), identity(), 1);

  fields cpu(&cpu_structure);
  cpu.use_real_fields();
  cpu.require_component(Ez);
  /* Preserve a like-for-like prepared CPU reference. */
  cpu.advance(1);
  cpu.t = 0;

  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  fields gpu(&gpu_structure, options);
  gpu.use_real_fields();
  gpu.require_component(Ez);
  gpu.init_backend();

  const bool f32 = policy == precision_policy_kind::f32;
  if (f32) round_real_arrays(*cpu.array_catalog);
  initialize_fields(cpu, gpu, f32);

  cpu.advance(3);
  gpu.advance(3);
  require(cpu.t == gpu.t, "NVIDIA timestep did not advance host time");
  compare_fields(cpu, gpu, f32 ? 3e-6 : 2e-13);
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  require(count_processors() == 1, "nvidia_timestep is a single-rank test");
  set_finite_check_mode(FiniteCheckMode::off);
  run_policy(precision_policy_kind::native);
  run_policy(precision_policy_kind::f32);
  master_printf("nvidia_timestep: PASS\n");
  return 0;
}
