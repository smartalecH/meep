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

#include "meep.hpp"
#include "backend/lifecycle.hpp"

namespace meep {

static_assert(fields::num_mutation_kinds == mutation_kind_count,
              "fields::num_mutation_kinds is out of sync with meep::mutation_kind_count");

DirtyMask invalidation_closure(MutationKind cause) {
  switch (cause) {
    /* Values only: storage and executable layout remain valid, but a resident
       backend must refresh the authoritative values from the host mutation. */
    case MutationKind::field_values: return dirty_initialization;

    /* Amplitudes changed but the index tables did not, so only the values that
       get pushed to the backend are stale. */
    case MutationKind::source_values: return dirty_initialization;

    /* A different set of sources means a different source plan and a different
       emitted operation list. Promotion to storage/halos happens separately,
       when preparation discovers a newly required component. */
    case MutationKind::source_definition:
      return dirty_source_plan | dirty_regions | dirty_executable;

    /* A monitor owns accumulator storage, so its definition reaches storage. */
    case MutationKind::monitor_definition:
      return dirty_monitor_plan | dirty_regions | dirty_storage | dirty_executable;

    /* Coefficients changed in place. Classification has to re-run because a
       coefficient can become (or stop being) trivial, but in the common case
       the classification hash is unchanged and the executable is reused. */
    case MutationKind::material_values: return dirty_initialization | dirty_classification;

    /* Same as material_values, but preparation may restrict the initialization
       delta to the changed region. */
    case MutationKind::material_region: return dirty_initialization | dirty_classification;

    case MutationKind::material_phase:
      return dirty_initialization | dirty_classification | dirty_executable;

    /* A different recipe or susceptibility set changes which arrays exist,
       which polarization internals need halo exchange, and the op list. */
    case MutationKind::material_definition:
      return dirty_initialization | dirty_storage | dirty_halos | dirty_classification |
             dirty_executable;

    case MutationKind::field_layout: return dirty_storage | dirty_halos | dirty_executable;

    case MutationKind::boundary_topology: return dirty_regions | dirty_halos | dirty_executable;

    case MutationKind::chunk_topology:
      return dirty_storage | dirty_regions | dirty_halos | dirty_executable;

    case MutationKind::coordinate_definition: return dirty_executable;

    case MutationKind::precision_policy:
      return dirty_storage | dirty_initialization | dirty_executable;
  }
  return dirty_none; // unreachable; keeps -Wreturn-type quiet
}

const char *mutation_kind_name(MutationKind cause) {
  switch (cause) {
    case MutationKind::field_values: return "field_values";
    case MutationKind::source_values: return "source_values";
    case MutationKind::source_definition: return "source_definition";
    case MutationKind::monitor_definition: return "monitor_definition";
    case MutationKind::material_values: return "material_values";
    case MutationKind::material_region: return "material_region";
    case MutationKind::material_phase: return "material_phase";
    case MutationKind::material_definition: return "material_definition";
    case MutationKind::field_layout: return "field_layout";
    case MutationKind::boundary_topology: return "boundary_topology";
    case MutationKind::chunk_topology: return "chunk_topology";
    case MutationKind::coordinate_definition: return "coordinate_definition";
    case MutationKind::precision_policy: return "precision_policy";
  }
  return "?";
}

const char *dirty_bit_name(DirtyBit bit) {
  switch (bit) {
    case dirty_none: return "none";
    case dirty_initialization: return "initialization";
    case dirty_source_plan: return "source_plan";
    case dirty_monitor_plan: return "monitor_plan";
    case dirty_storage: return "storage";
    case dirty_regions: return "regions";
    case dirty_halos: return "halos";
    case dirty_executable: return "executable";
    case dirty_classification: return "classification";
  }
  return "?";
}

void lifecycle_init(fields &f) {
  f.dirty_mask = dirty_none;
  for (int i = 0; i < fields::num_mutation_kinds; ++i)
    f.mutation_generation[i] = 0;
  f.connections_generation = 0;
  f.connections_built_generation = 0;
  f.local_invalidation_generation = 0;
  f.local_invalidation_synced = 0;
  f.storage_prepared_mask = 0;
  f.prepared_classification_hash = 0;
  f.classification_reentries = 0;
  f.nonfinite_flag = 0;
  f.first_bad_step = -1;
  f.first_bad_component = -1;
}

bool invalidate_collectively(fields &f, MutationKind cause, bool locally_observed,
                             const char *site) {
  const bool anywhere = or_to_all(locally_observed);
  if (anywhere) invalidate(f, cause, site);
  return anywhere;
}

void invalidate(fields &f, MutationKind cause, const char *site) {
  f.dirty_mask |= invalidation_closure(cause);
  ++f.mutation_generation[static_cast<int>(cause)];
  (void)site;
}

uint64_t generation(const fields &f, MutationKind cause) {
  return f.mutation_generation[static_cast<int>(cause)];
}

bool is_dirty(const fields &f, DirtyBit bit) { return (f.dirty_mask & bit) != 0; }

void clear_dirty(fields &f, DirtyMask bits) { f.dirty_mask &= ~bits; }

void note_connections_invalidated(fields &f) { ++f.connections_generation; }

void note_connections_built(fields &f) {
  f.connections_built_generation = f.connections_generation;
  clear_dirty(f, dirty_halos);
}

bool connections_are_current(const fields &f) {
  return f.connections_generation == f.connections_built_generation;
}

void mark_local_invalidation(fields &f) { ++f.local_invalidation_generation; }

bool needs_connection_sync(const fields &f) {
  return f.local_invalidation_generation != f.local_invalidation_synced;
}

void note_connection_sync_done(fields &f) {
  f.local_invalidation_synced = f.local_invalidation_generation;
}

#ifndef NDEBUG
void assert_connections_shadow(const fields &f, bool legacy_valid, const char *where) {
  if (connections_are_current(f) != legacy_valid)
    meep::abort("lifecycle shadow mismatch at %s: chunk_connections_valid=%d but "
                "connections_are_current()=%d (gen %llu vs built %llu)\n",
                where, int(legacy_valid), int(connections_are_current(f)),
                (unsigned long long)f.connections_generation,
                (unsigned long long)f.connections_built_generation);
}

void assert_local_invalidation_shadow(const fields &f, bool legacy_flag, const char *where) {
  if (needs_connection_sync(f) != legacy_flag)
    meep::abort("lifecycle shadow mismatch at %s: changed_materials=%d but "
                "needs_connection_sync()=%d (gen %llu vs synced %llu)\n",
                where, int(legacy_flag), int(needs_connection_sync(f)),
                (unsigned long long)f.local_invalidation_generation,
                (unsigned long long)f.local_invalidation_synced);
}
#else
void assert_connections_shadow(const fields &, bool, const char *) {}
void assert_local_invalidation_shadow(const fields &, bool, const char *) {}
#endif

} // namespace meep
