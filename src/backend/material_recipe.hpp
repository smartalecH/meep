/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* Frozen, backend-neutral material initialization input.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 */

#ifndef MEEP_BACKEND_MATERIAL_RECIPE_HPP
#define MEEP_BACKEND_MATERIAL_RECIPE_HPP

#include <stdint.h>
#include <memory>
#include <string>
#include <vector>

#include "backend/storage_plan.hpp"
#include "backend/material_ir.hpp"
#include "backend/precision.hpp"

namespace meep {

struct MaterialClassification;

enum class MaterialRecipeDisposition {
  device_native,
  host_reference,
  hybrid_interface,
  tiled_callback
};

enum MaterialSupportReason : uint64_t {
  material_support_none = 0,
  material_support_object_lookup = UINT64_C(1) << 0,
  material_support_adaptive_averaging = UINT64_C(1) << 1,
  material_support_unowned_callback = UINT64_C(1) << 2,
  material_support_missing_dense_fallback = UINT64_C(1) << 3,
  material_support_no_owned_ir = UINT64_C(1) << 4
};

struct MaterialSupportDecision {
  MaterialRecipeDisposition disposition;
  uint64_t reason_bits;
  uint64_t native_points;
  uint64_t interface_points;
  uint64_t callback_points;
  uint64_t compact_input_bytes;
  uint64_t dense_fallback_bytes;
  uint64_t scratch_bytes;
};

const char *material_recipe_disposition_name(MaterialRecipeDisposition disposition);

/* One owned dense row for the host-reference disposition. Later device-native
   recipes replace `values` with compact geometry/medium IR; keeping the two
   representations distinct prevents this compatibility slice from claiming
   zero dense host coefficient construction. */
struct MaterialRecipeRow {
  StorageKey key;
  array_role role;
  ElementType element_type;
  Precision storage;
  size_t elements;
  size_t alignment;
  std::vector<unsigned char> values;

  bool operator==(const MaterialRecipeRow &other) const;
};

struct MaterialRecipeInput {
  MaterialRecipeDisposition disposition;
  std::string description;
  bool eps_averaging;
  double subpixel_tol;
  int subpixel_maxeval;
  uint32_t host_callback_id;
  bool from_host_callback;
  uint64_t support_reason_bits;
  std::vector<MaterialRecipeRow> rows;
  std::vector<MaterialIRTopologyRow> topology;
  std::shared_ptr<const MaterialIR> ir;

  MaterialRecipeInput();
};

/* Immutable after construction: all input containers are copied and only
   const accessors are exposed. The stored signature is recomputed during
   validation, so serialization/copy corruption cannot silently change output. */
class MaterialRecipe {
public:
  explicit MaterialRecipe(const MaterialRecipeInput &input);

  uint32_t version() const { return version_; }
  MaterialRecipeDisposition disposition() const { return disposition_; }
  const std::string &description() const { return description_; }
  bool eps_averaging() const { return eps_averaging_; }
  double subpixel_tol() const { return subpixel_tol_; }
  int subpixel_maxeval() const { return subpixel_maxeval_; }
  uint32_t host_callback_id() const { return host_callback_id_; }
  bool from_host_callback() const { return from_host_callback_; }
  uint64_t support_reason_bits() const { return support_reason_bits_; }
  const std::vector<MaterialRecipeRow> &rows() const { return rows_; }
  const std::vector<MaterialIRTopologyRow> &topology() const { return topology_; }
  const std::shared_ptr<const MaterialIR> &ir() const { return ir_; }
  uint64_t signature() const { return signature_; }

  bool operator==(const MaterialRecipe &other) const;
  bool operator!=(const MaterialRecipe &other) const { return !(*this == other); }

private:
  uint32_t version_;
  MaterialRecipeDisposition disposition_;
  std::string description_;
  bool eps_averaging_;
  double subpixel_tol_;
  int subpixel_maxeval_;
  uint32_t host_callback_id_;
  bool from_host_callback_;
  uint64_t support_reason_bits_;
  std::vector<MaterialRecipeRow> rows_;
  std::vector<MaterialIRTopologyRow> topology_;
  std::shared_ptr<const MaterialIR> ir_;
  uint64_t signature_;
};

MaterialRecipe build_host_reference_material_recipe(const fields &f);
MaterialSupportDecision classify_material_support(const MaterialRecipe &recipe);
void validate_material_recipe(const MaterialRecipe &recipe);
void mark_material_storage_provisional(const MaterialRecipe &recipe, StoragePlan &plan);
void resolve_material_storage(const MaterialRecipe &recipe,
                              const MaterialClassification &classification,
                              const StoragePlan &authoritative, StoragePlan &provisional,
                              const PrecisionPolicy &policy);
void resolve_material_storage(const MaterialRecipe &recipe,
                              const MaterialClassification &classification,
                              const StoragePlan &authoritative, StoragePlan &provisional);
bool has_provisional_material_storage(const StoragePlan &plan);

/* Focused failure seam for collective recipe-capture tests. */
void set_material_recipe_failure_for_testing(int rank, int mode);

} // namespace meep

#endif // MEEP_BACKEND_MATERIAL_RECIPE_HPP
