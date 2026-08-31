/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

/* Classification: the facts that are produced *by* material initialization
 * rather than supplied to it.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 *
 * set_chi1inv decides whether an array exists by reducing over the values it
 * just computed and deleting the trivial rows; add_susceptibility does the same
 * for sigma; update_eh picks the anisotropic tiled path from which chi1inv rows
 * survived; is_aniso2d can *add components* after materials are set; add_dft
 * chooses a decimation factor from has_nonlinearities(). A linear
 * allocate -> initialize -> compile lifecycle cannot express any of that.
 *
 * This is not an optimization. The CPU code discovers these facts implicitly
 * today, and the prepared representation has to reproduce them from the same
 * values rather than guess from the recipe, or the two diverge.
 */

#ifndef MEEP_BACKEND_CLASSIFICATION_HPP
#define MEEP_BACKEND_CLASSIFICATION_HPP

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

#include "meep.hpp"
#include "backend/array_ref.hpp"
#include "backend/storage_plan.hpp"

namespace meep {

typedef uint64_t component_mask;

const uint32_t material_classification_facts_version = 2;

enum MaterialVariantBits : uint32_t {
  material_variant_anisotropic = 1u << 0,
  material_variant_nonlinear = 1u << 1,
  material_variant_conductivity = 1u << 2,
  material_variant_pml = 1u << 3,
  material_variant_sigma = 1u << 4,
  material_variant_chi2 = 1u << 5,
  material_variant_chi3 = 1u << 6
};

/* Compact, pointer-free facts emitted by a material initializer.  CPU and
   device backends use the same records; the collective assembler below is the
   sole authority for normalization, group validation, and hashing. */
struct MaterialRowClassificationFact {
  StorageKey key;
  uint8_t state;
};

struct MaterialVariantClassificationFact {
  uint32_t operation;
  uint32_t region;
  int chunk;
  int field_type_;
  int component_;
  int cmp;
  int dimension;
  int begin[5];
  int end[5];
  size_t base;
  size_t counts[3];
  ptrdiff_t strides[3];
  uint32_t variant_key;
};

struct MaterialClassificationFacts {
  uint32_t version;
  std::vector<MaterialRowClassificationFact> rows;
  std::vector<MaterialVariantClassificationFact> variants;
  component_mask required_components;
  bool has_nonlinearities;
  bool aniso2d;
  /* Zero means that this rank contributes no DFT decimation constraint. */
  int min_decimation_factor;

  MaterialClassificationFacts()
      : version(material_classification_facts_version), required_components(0),
        has_nonlinearities(false), aniso2d(false), min_decimation_factor(0) {}
};

struct MaterialClassification {
  enum ProvisionalRowState : uint8_t {
    not_provisional = 0,
    retained = 1,
    elided_row = 2
  };

  std::vector<ArrayId> elided;        // provisional arrays that turned out trivial
  /* One entry per StoragePlan ArrayId. This makes classification total: an
     omitted status can never be mistaken for retention. */
  std::vector<uint8_t> provisional_row_state;
  /* Exact ordinary-StepPlan constitutive regions in canonical operation,
     chunk, and within-chunk order. */
  std::vector<MaterialVariantClassificationFact> variant_facts;
  component_mask required_components; // may grow: is_aniso2d, beta coupling
  bool has_nonlinearities;
  int min_decimation_factor;
  uint64_t hash;

  /* Per-chunk, per-field-type: does update_eh take the anisotropic tiled path?
     This is what drives gvs_eh, which used to be recomputed as a side effect of
     every fields::update_eh call. */
  std::vector<uint8_t> anisotropic_eh; // [chunk * NUM_FIELD_TYPES + ft]

  bool aniso2d;

  MaterialClassification()
      : required_components(0), has_nonlinearities(false), min_decimation_factor(1), hash(0),
        aniso2d(false) {}
};

struct PreparedSimulation {
  StoragePlan storage;
  uint64_t classification_hash;
  size_t reentry_count; // pass-1 re-entries; must never exceed 1

  PreparedSimulation() : classification_hash(0), reentry_count(0) {}
};

struct MaterialRecipe;
struct InitializationPlan;
struct StepPlan;

/* Assemble local compact facts into one rank-identical semantic result. */
MaterialClassification assemble_material_classification(
    fields &f, const StoragePlan &plan, const MaterialRecipe &recipe,
    const MaterialClassificationFacts &local_facts);

/* Recompute this rank's semantic variants after a test/backend changes row
   status facts, without inspecting coefficient payloads. */
void refresh_material_classification_variants(fields &f, const StoragePlan &plan,
                                              MaterialClassification &classification);

/* Validate every backend-produced fact before lifecycle code reads promotion
   bits or publishes elision.  On success this replaces `hash` with the common
   semantic-recipe-inclusive canonical hash. */
void validate_material_classification(fields &f, const StoragePlan &plan,
                                      const MaterialRecipe &recipe,
                                      MaterialClassification &classification);

/* Final post-resolution safety gate.  No alias, initialization action, step
   action, or access may retain a tombstoned ArrayId. */
void validate_material_classification_consumers(const StoragePlan &plan,
                                                const InitializationPlan &initialization,
                                                const StepPlan &steps,
                                                const MaterialClassification &classification);

/* Reproduce, from the values initialization actually produced, every fact in
   the table in section 9.1 of the plan.

   Collective wherever the underlying CPU test is: is_aniso2d uses or_to_all and
   add_dft uses min_to_all on the decimation factor. The resulting hash must be
   identical on every rank -- a wrong hash deadlocks under MPI rather than
   merely mis-optimizing -- so it is computed order-independently across chunks
   and then reduced. */
MaterialClassification classify(fields &f, const StoragePlan &plan);

/* Return the component layout present on any rank in the active Meep
   communicator.  This is collective: idle ranks deliberately contribute an
   empty local mask rather than interpreting that empty mask as a globally
   missing layout. */
component_mask global_field_component_presence(fields &f);

/* Apply the classification: choose update variants and publish the tiling
   decision. Returns true if it promoted the mutation (e.g. is_aniso2d added
   components), meaning pass 1 has to be re-entered exactly once. */
bool apply_classification(fields &f, const MaterialClassification &cls);

/* Execute the private pass-2 transition. Kept backend-private so tests and
   future resident preparation can exercise the exact classification boundary
   without exposing fields::classify_and_finalize through the public/SWIG API. */
void backend_classify_and_finalize(fields &f);

const char *classification_summary(const MaterialClassification &cls);

} // namespace meep

#endif // MEEP_BACKEND_CLASSIFICATION_HPP
