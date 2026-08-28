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
#include <vector>

#include "backend/backend.hpp"
#include "backend/lifecycle.hpp"
#include "backend/precision.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

using namespace meep;

static double eps_slab(const vec &p) { return fabs(p.y()) < 0.4 ? 12.0 : 1.0; }

static void require(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "[rank %d] FAIL: %s\n", my_rank(), message);
    meep::abort("nvidia_backend_state failed");
  }
}

static void run_policy(precision_policy_kind policy) {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5));
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.device_id = automatic_device;
  options.precision = policy;
  fields f(&s, options);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.11, 0.13));

  f.init_backend();
  const backend_capabilities capabilities = f.backend_caps();
  require(strcmp(capabilities.name, "nvidia") == 0, "NVIDIA backend was not selected");
  require(capabilities.memory_budget_bytes > 0, "device memory was not reported");
  require(f.backend_state != NULL, "resident backend state was not created");
  require(f.array_catalog != NULL && f.array_catalog->size() > 0,
          "host compatibility catalog was not prepared");

  ArrayId chosen = invalid_array();
  for (size_t i = 0; i < f.array_catalog->size(); ++i) {
    const ArraySpec &spec = f.array_catalog->spec(ArrayId{uint32_t(i)});
    if (!is_valid(spec.alias_of) && spec.role == array_role::field &&
        spec.element_type == ElementType::realnum_value && spec.elements >= 2) {
      chosen = spec.id;
      break;
    }
  }
  require(is_valid(chosen), "no canonical array was available for a transfer test");

  const bool narrowed = policy != precision_policy_kind::native;
  realnum *host_values = static_cast<realnum *>(f.array_catalog->resolve_untyped(chosen));
  realnum initial_expected[2] = {host_values[0], host_values[1]};
  if (narrowed) {
    initial_expected[0] = realnum(float(initial_expected[0]));
    initial_expected[1] = realnum(float(initial_expected[1]));
  }
  realnum observed[2] = {0, 0};
  f.backend->read(ArrayRef{chosen, 0, 2}, observed, sizeof(observed));
  require(!memcmp(initial_expected, observed, sizeof(observed)),
          "initial device upload did not apply the selected precision");

  const realnum replacement[2] = {realnum(1.234567890123), realnum(-2.345678901234)};
  realnum rounded[2] = {replacement[0], replacement[1]};
  if (narrowed) {
    rounded[0] = realnum(float(rounded[0]));
    rounded[1] = realnum(float(rounded[1]));
  }
  f.backend->write(ArrayRef{chosen, 0, 2}, replacement, sizeof(replacement));
  observed[0] = observed[1] = 0;
  f.backend->read(ArrayRef{chosen, 0, 2}, observed, sizeof(observed));
  require(!memcmp(rounded, observed, sizeof(observed)),
          "device write/read did not preserve storage precision");
  host_values = static_cast<realnum *>(f.array_catalog->resolve_untyped(chosen));
  require(!memcmp(rounded, host_values, sizeof(rounded)),
          "compatibility host mirror differs from device storage");

  invalidate(f, MutationKind::material_values);
  f.init_backend();
  observed[0] = observed[1] = 0;
  f.backend->read(ArrayRef{chosen, 0, 2}, observed, sizeof(observed));
  require(!memcmp(rounded, observed, sizeof(observed)),
          "initialization refresh lost device values");

  invalidate(f, MutationKind::field_layout);
  f.init_backend();
  observed[0] = observed[1] = 0;
  f.backend->read(ArrayRef{chosen, 0, 2}, observed, sizeof(observed));
  require(!memcmp(rounded, observed, sizeof(observed)), "storage rebuild lost device values");

  bool rejected = false;
  const ArraySpec rebuilt_spec = f.array_catalog->spec(chosen);
  try {
    f.backend->read(ArrayRef{chosen, rebuilt_spec.elements, 1}, observed, sizeof(realnum));
  }
  catch (const std::out_of_range &) { rejected = true; }
  require(rejected, "out-of-range device access was not rejected");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);

  run_policy(precision_policy_kind::native);
  run_policy(precision_policy_kind::mixed);
  run_policy(precision_policy_kind::f32);

  if (my_rank() == 0) printf("nvidia_backend_state: PASS\n");
  return 0;
}
