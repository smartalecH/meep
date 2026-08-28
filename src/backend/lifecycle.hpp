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

/* Generations and typed invalidation.
 *
 * This header is BACKEND-PRIVATE. It is not installed and must never be
 * included from meep.hpp (see global rule 5 of the Phase 1 plan): both
 * python/meep.i and scheme/meep.i include meep.hpp, so anything reachable
 * from there leaks into the SWIG surface.
 *
 * `fields` therefore stores only POD counters (see fields::dirty_mask and
 * fields::mutation_generation in meep.hpp) and all interpretation of those
 * counters lives here.
 */

#ifndef MEEP_BACKEND_LIFECYCLE_HPP
#define MEEP_BACKEND_LIFECYCLE_HPP

#include <stdint.h>

namespace meep {

class fields;

/* Why the caller says state changed. This is the *cause*, not the effect;
   invalidation_closure() maps a cause onto the set of prepared artifacts that
   the cause can invalidate. */
enum class MutationKind {
  field_values,        // zero/restart or compatible checkpoint load; refresh resident values
  source_values,       // amplitudes changed, same indices/components
  source_definition,   // add/remove/replace source
  monitor_definition,  // add/remove/change monitor region
  material_values,     // coefficients changed, same represented layout
  material_region,     // coefficients changed inside a bounded region only
  material_phase,      // phase-in state changes coefficients and the operation schedule
  material_definition, // recipe or susceptibility set changed
  field_layout,        // component, real/complex mode, or optional array changed
  boundary_topology,   // boundary, Bloch phase, symmetry, or user volume changed
  chunk_topology,      // decomposition or chunk ownership changed
  precision_policy     // storage precision changed; state must be converted or rebuilt
};

/* Keep in sync with fields::num_mutation_kinds in meep.hpp; lifecycle.cpp
   static_asserts that they agree. */
const int mutation_kind_count = 12;

typedef uint32_t DirtyMask;
enum DirtyBit : DirtyMask {
  dirty_none = 0,
  dirty_initialization = 1 << 0,
  dirty_source_plan = 1 << 1,
  dirty_monitor_plan = 1 << 2,
  dirty_storage = 1 << 3,
  dirty_regions = 1 << 4,
  dirty_halos = 1 << 5,
  dirty_executable = 1 << 6,
  dirty_classification = 1 << 7
};

/* The table in §6.4 of the plan, as code. */
DirtyMask invalidation_closure(MutationKind cause);

const char *mutation_kind_name(MutationKind cause);
const char *dirty_bit_name(DirtyBit bit);

/* Record a mutation: union its closure into f.dirty_mask and bump the
   generation counter for `cause`.

   This is a free function rather than a member of `fields` only because the
   MutationKind type may not appear in meep.hpp.

   invalidate() deliberately does NOT touch the connection bookkeeping below.
   The two are kept separate so that in PR 1 the connection counters can mirror
   the legacy flags exactly, 1:1 with the assignment sites they replace, while
   the dirty mask and generations describe the forward-looking model that
   PRs 3-7 consume. PR 2 makes the counters authoritative. */
void invalidate(fields &f, MutationKind cause, const char *site = "?");

/* Rank-local observations must be reduced before dirty bits that gate
   collective preparation are changed. */
bool invalidate_collectively(fields &f, MutationKind cause, bool locally_observed,
                             const char *site = "?");

void lifecycle_init(fields &f);

uint64_t generation(const fields &f, MutationKind cause);

/* True when any cause whose closure includes `bit` has fired since the
   corresponding artifact was last (re)built. */
bool is_dirty(const fields &f, DirtyBit bit);
void clear_dirty(fields &f, DirtyMask bits);

/* --- Chunk connectivity ---------------------------------------------------
   Shadows `fields::chunk_connections_valid`, which is deleted in PR 2. Call
   note_connections_invalidated() at exactly the sites that used to assign
   `chunk_connections_valid = false`. */
void note_connections_invalidated(fields &f);
void note_connections_built(fields &f);
bool connections_are_current(const fields &f);

/* --- Possibly-rank-local invalidation --------------------------------------
   Shadows `fields::changed_materials`. Despite the legacy name, that flag does
   not track material changes: it records that some invalidation may have been
   applied on a subset of ranks (component allocation in require_component, the
   lazy allocations in step_db/update_eh/update_pols, per-chunk material
   phasing), so connect_chunks() must run a collective and_to_all before acting
   on it. Sites that are guaranteed to run on every rank -- set_boundary,
   use_bloch, use_real_fields -- deliberately do not mark it. */
void mark_local_invalidation(fields &f);
bool needs_connection_sync(const fields &f);
void note_connection_sync_done(fields &f);

/* --- PR 1 shadow assertions -----------------------------------------------
   No-ops when NDEBUG is defined. The assertions-on CI configuration is what
   gives them value; see §4 of the plan. */
void assert_connections_shadow(const fields &f, bool legacy_valid, const char *where);
void assert_local_invalidation_shadow(const fields &f, bool legacy_flag, const char *where);

} // namespace meep

#endif // MEEP_BACKEND_LIFECYCLE_HPP
