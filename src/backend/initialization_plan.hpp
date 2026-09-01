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

  void validate() const;
  bool empty() const;
  bool contains(const InitRegion &other) const;
  bool overlaps(const InitRegion &other) const;
  InitRegion intersection(const InitRegion &other) const;
};

struct InitPointSpan {
  uint64_t first;
  uint64_t count;
};

struct InitOperation {
  InitKind kind;
  ArrayRef destination;
  uint32_t descriptor_index;
  InitRegion region;
  /* Exact canonical point ordinals selected by a regional restriction. Empty
     means the complete destination. IDs are never renumbered. */
  std::vector<InitPointSpan> point_spans;
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

enum class RegionalSupportReason {
  supported,
  empty,
  opaque_coordinates,
  incomplete_group,
  whole_row_kernel,
  remote_dependency
};

const char *regional_support_reason_name(RegionalSupportReason reason);

struct InitializationPlan {
  uint64_t material_values_generation = 0;
  uint64_t material_region_generation = 0;
  std::vector<InitOperation> operations;
  std::vector<MaterialRecipe> materials;
  std::vector<PmlRecipe> pml;
  std::vector<HostCallbackRecipe> host_callbacks;
  /* Compact selectors into the immutable MaterialIR. They preserve original
     destination/job identities while excluding work outside a restricted
     refresh. Empty selectors on a non-regional plan mean the complete IR. */
  std::vector<uint32_t> material_destinations;
  std::vector<MaterialIRBulkSpan> material_bulk_spans;
  std::vector<uint32_t> material_analytic_interfaces;
  std::vector<uint32_t> material_hybrid_patches;
  std::vector<MaterialCallbackTile> material_callback_tiles;
  /* A same-simulation multi-rank plan may classify local regional work, but
     publishing it would require PR7 transport/reconciliation. */
  bool requires_remote_transport = false;
  bool regional = false;
  bool regional_supported = true;
  RegionalSupportReason regional_reason = RegionalSupportReason::supported;
  std::string regional_unsupported_reason;
  InitRegion requested_region;

  /* Emit the subset of operations needed to refresh `region` only. */
  InitializationPlan restrict_to(const InitRegion &region) const;

  void clear() {
    material_values_generation = 0;
    material_region_generation = 0;
    operations.clear();
    materials.clear();
    pml.clear();
    host_callbacks.clear();
    material_destinations.clear();
    material_bulk_spans.clear();
    material_analytic_interfaces.clear();
    material_hybrid_patches.clear();
    material_callback_tiles.clear();
    requires_remote_transport = false;
    regional = false;
    regional_supported = true;
    regional_reason = RegionalSupportReason::supported;
    regional_unsupported_reason.clear();
    requested_region = InitRegion();
  }
};

InitializationPlan build_initialization_plan(fields &f);

/* Backend-private regional mutation authority. The region is owned by the
   fields lifetime and consumed only after a successful material transaction. */
void invalidate_material_region(fields &f, const InitRegion &region,
                                const char *reason = NULL);
bool pending_material_region(const fields &f, InitRegion *region);
void clear_pending_material_region(const fields &f);
bool regional_replacement_preserves_unselected(const InitializationPlan &installed,
                                               const InitializationPlan &candidate,
                                               const InitializationPlan &restricted,
                                               std::string *reason);

} // namespace meep

#endif // MEEP_BACKEND_INITIALIZATION_PLAN_HPP
