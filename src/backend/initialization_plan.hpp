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

/* A deep-copied, backend-neutral description of the material geometry, held by
   the simulation rather than borrowed from a temporary SWIG
   geometric_object_list. */
struct MaterialRecipe {
  std::string description;
  bool eps_averaging;
  double subpixel_tol;
  int subpixel_maxeval;
  uint32_t host_callback_id; // valid when the material is an arbitrary eps_func
  bool from_host_callback;

  MaterialRecipe()
      : eps_averaging(true), subpixel_tol(1e-4), subpixel_maxeval(100000),
        host_callback_id(0xffffffffu), from_host_callback(false) {}
};

struct PmlRecipe {
  int direction_;
  double thickness;
  double strength;
  double r_asymptotic;
};

struct HostCallbackRecipe {
  uint32_t id;
  std::string description;
};

struct InitializationPlan {
  std::vector<InitOperation> operations;
  std::vector<MaterialRecipe> materials;
  std::vector<PmlRecipe> pml;
  std::vector<HostCallbackRecipe> host_callbacks;

  /* Emit the subset of operations needed to refresh `region` only.
     No Phase-1 consumer: the in-place design update that would use it is
     deferred (§14). Built and unit-tested, deliberately unwired. */
  InitializationPlan restrict_to(const InitRegion &region) const;

  void clear() {
    operations.clear();
    materials.clear();
    pml.clear();
    host_callbacks.clear();
  }
};

InitializationPlan build_initialization_plan(fields &f);

} // namespace meep

#endif // MEEP_BACKEND_INITIALIZATION_PLAN_HPP
