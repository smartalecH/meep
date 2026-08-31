/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* How arrays get their initial values, as data.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 */

#ifndef MEEP_BACKEND_INITIALIZATION_PLAN_HPP
#define MEEP_BACKEND_INITIALIZATION_PLAN_HPP

#include <string>
#include <vector>

#include "meep.hpp"
#include "backend/array_ref.hpp"
#include "backend/material_recipe.hpp"

namespace meep {

enum class InitKind { zero, constant, material_geometry, pml_profile, host_callback, host_array,
                      checkpoint };

const char *init_kind_name(InitKind k);

struct InitRegion {
  int chunk;  // -1 == every chunk owning the destination
  ivec begin; // an empty extent means the whole array
  ivec end;
  bool whole; // true when begin/end are unset

  InitRegion() : chunk(-1), begin(D1), end(D1), whole(true) {}
  InitRegion(int c, const ivec &b, const ivec &e) : chunk(c), begin(b), end(e), whole(false) {}

  bool contains(const InitRegion &other) const;
};

struct InitOperation {
  InitKind kind;
  ArrayRef destination;
  uint32_t descriptor_index;
  InitRegion region;
};

struct PmlRecipe {
  int chunk;
  int direction_;
  std::vector<double> sigma;
  std::vector<double> kappa;
  std::vector<double> sigma_inv;
};

struct HostCallbackRecipe {
  uint32_t id;
  std::string description;
};

struct InitializationPlan {
  uint64_t material_values_generation = 0;
  uint64_t material_region_generation = 0;
  std::vector<InitOperation> operations;
  std::vector<MaterialRecipe> materials;
  std::vector<PmlRecipe> pml;
  std::vector<HostCallbackRecipe> host_callbacks;

  /* Emit the subset of operations needed to refresh `region` only.
     No Phase-1 consumer: the in-place design update that would use it is
     deferred (§14). Built and unit-tested, deliberately unwired. */
  InitializationPlan restrict_to(const InitRegion &region) const;

  void clear() {
    material_values_generation = 0;
    material_region_generation = 0;
    operations.clear();
    materials.clear();
    pml.clear();
    host_callbacks.clear();
  }
};

InitializationPlan build_initialization_plan(fields &f);

} // namespace meep

#endif // MEEP_BACKEND_INITIALIZATION_PLAN_HPP
