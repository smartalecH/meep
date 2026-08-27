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

/* The timestep as data.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 *
 * fields::step walked chunks and variants and invoked many small operations,
 * repeating the same plan decisions every step. This replaces the procedure
 * with a constructed plan that a backend executes.
 *
 * An operation is COARSE: an update_db operation refers to a span of
 * chunk/component regions, not one voxel. There is no backend virtual call and
 * no backend branch per voxel, and the CPU kernels are untouched.
 */

#ifndef MEEP_BACKEND_STEP_PLAN_HPP
#define MEEP_BACKEND_STEP_PLAN_HPP

#include <stdint.h>
#include <string>
#include <vector>

#include "meep.hpp"
#include "backend/array_ref.hpp"

namespace meep {

enum class OpKind {
  restore_magnetic_fields,
  phase_material,
  update_material_coefficients,
  evaluate_source_scalars,
  update_db,
  update_eh,
  update_polarization,
  apply_sources,
  zero_boundary,
  pack_halo,
  transfer_halo,  // host/MPI scheduler operation
  exchange_local, // same-rank fused gather/scatter; deferred, see section 14
  unpack_halo,
  update_flux_half,
  update_flux,
  increment_time,
  update_dft,
  synchronize_magnetic_fields,
  finite_value_check,
  reduction,
  host_callback,
  pack_state,   // solve_cw BiCGSTAB vector packing
  unpack_state,
  num_kinds
};

const char *op_kind_name(OpKind k);

enum class AccessMode { read, write, read_write };

struct BufferAccess {
  ArrayRef array;
  AccessMode mode;
};

/* The guard mechanism is recorded in the plan rather than resolved by the
   executor because it determines graph structure in Phase 2: a device_predicate
   keeps a node in the graph, a graph_variant forces a second compiled program,
   a segment_boundary forces a host round trip. The CPU executor treats them all
   the same, but the choice has to be made now -- retrofitting it means
   re-reviewing the whole plan. */
enum class GuardKind {
  always,
  static_predicate, // resolved during preparation; the op is present or absent
  device_predicate, // kernel reads StepScalars and returns immediately when false
  graph_variant,    // the host scheduler selects a separately compiled program
  segment_boundary  // the host evaluates the guard between device segments
};

struct Guard {
  GuardKind kind;
  uint32_t scalar_slot;
  uint32_t variant_index;
};

inline Guard guard_always() { return Guard{GuardKind::always, 0, 0}; }
inline Guard guard_static(bool) { return Guard{GuardKind::static_predicate, 0, 0}; }
inline Guard guard_device(uint32_t slot) { return Guard{GuardKind::device_predicate, slot, 0}; }
inline Guard guard_variant(uint32_t v) { return Guard{GuardKind::graph_variant, 0, v}; }
inline Guard guard_segment(uint32_t slot) { return Guard{GuardKind::segment_boundary, slot, 0}; }

struct Operation {
  OpKind kind;
  uint32_t descriptor_index;
  Guard guard;
  /* PR 5 keeps the coarse (field_type, chunk span) argument the existing CPU
     entry points take. PR 6 replaces it with real descriptors. */
  field_type ft;
  double source_time_offset; // for evaluate_source_scalars: 0, 0.5*dt or dt
  std::vector<BufferAccess> accesses;
};

struct UpdateRegion {
  int chunk;
  component c;
  int cmp;
  ivec begin;
  ivec end;
  /* PML, conductivity, anisotropy, precision, coordinate and stride cases.
     Does not encode a backend. */
  uint32_t variant_key;
};

/* Which timestep program a plan describes. solve_cw is a genuinely different
   program, not a variant: step_source skips non-integrated sources, update_eh
   runs with skip_w_components, and update_dfts is disabled. If step() begins
   executing an ordinary-case plan without a corresponding solve_cw plan, CW
   solves silently run the wrong program. */
enum class StepProgram { ordinary, solve_cw };

struct StepPlan {
  StepProgram program;
  std::vector<Operation> operations;
  std::vector<UpdateRegion> db_updates;
  std::vector<UpdateRegion> eh_updates;
  uint64_t signature;

  StepPlan() : program(StepProgram::ordinary), signature(0) {}
  void clear() {
    operations.clear();
    db_updates.clear();
    eh_updates.clear();
    signature = 0;
  }
};

/* A direct transcription of the order in src/step.cpp. It omits empty work and
   NEVER reorders what remains. It is not a scheduler that infers Maxwell's
   equations from array dependencies: if a reviewer cannot line this up against
   fields::step_once by eye, it is wrong. */
StepPlan build_step_plan(fields &f, StepProgram program);

/* Human-readable operation sequence, for the trace test and for debugging. */
void format_step_plan(const StepPlan &p, std::vector<std::string> &out);

} // namespace meep

#endif // MEEP_BACKEND_STEP_PLAN_HPP
