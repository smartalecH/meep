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

/* The CPU backend's timestep executor.
 *
 * Deliberately its own translation unit, and not only for tidiness. The
 * executor was first written inside src/step.cpp, next to calc_sources and
 * fields_chunk::step_source, and the bitwise harness caught the consequence:
 * restructuring the file changed inlining in its neighbours, and
 * d2_custom_source picked up a 1-ULP difference at the source point. Phase 1
 * must change no arithmetic, and the cheapest way to guarantee that of code
 * the refactor does not touch is to not recompile it in a different context.
 */

#include <stdio.h>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "backend/lifecycle.hpp"
#include "backend/step_plan.hpp"

using namespace std;

namespace meep {

/* The timing sink each boundary step used to be wrapped in, preserved exactly
   so timing output does not change. */
static time_sink boundary_timing_scope(field_type ft) {
  switch (ft) {
    case B_stuff: return BoundarySteppingB;
    case WH_stuff: return BoundarySteppingWH;
    case PH_stuff: return BoundarySteppingPH;
    case H_stuff: return BoundarySteppingH;
    case D_stuff: return BoundarySteppingD;
    case WE_stuff: return BoundarySteppingWE;
    case PE_stuff: return BoundarySteppingPE;
    default: return BoundarySteppingE;
  }
}

/* The CPU executor. Each operation dispatches to the existing implementation at
   the same granularity as today's chunk loops, so there is no dispatch
   regression and no virtual call anywhere near a voxel loop. */
void fields::execute_step_plan(const StepPlan &plan, int save_synchronized_magnetic_fields) {
  bool segment_guard = false; // phase_material's collective E/H reconciliation

  for (const Operation &op : plan.operations) {
    if (op.guard.kind == GuardKind::segment_boundary && !segment_guard) continue;

    switch (op.kind) {
      case OpKind::restore_magnetic_fields:
        if (synchronized_magnetic_fields) {
          synchronized_magnetic_fields = 1; // reset synchronization count
          restore_magnetic_fields();
        }
        break;

      case OpKind::phase_material: segment_guard = phase_material_mix(); break;

      case OpKind::update_material_coefficients:
        // update cached conductivity-inverse array, if needed
        for (int i = 0; i < num_chunks; i++)
          chunks[i]->s->update_condinv();
        break;

      case OpKind::evaluate_source_scalars:
        /* Deliberately a call into step.cpp rather than the expression inline
           here -- see the comment on fields::evaluate_source_scalars. */
        evaluate_source_scalars(op.source_time_offset);
        break;

      case OpKind::update_db: {
        auto step_timer =
            with_timing_scope(op.ft == B_stuff ? FieldUpdateB : FieldUpdateD);
        step_db(op.ft);
        break;
      }

      case OpKind::apply_sources: step_source(op.ft); break;

      case OpKind::update_eh: {
        auto step_timer =
            with_timing_scope(op.ft == H_stuff ? FieldUpdateH : FieldUpdateE);
        update_eh(op.ft);
        break;
      }

      case OpKind::update_polarization: update_pols(op.ft); break;

      case OpKind::transfer_halo: {
        auto step_timer = with_timing_scope(boundary_timing_scope(op.ft));
        step_boundaries(op.ft);
        break;
      }

      case OpKind::update_flux_half:
        if (fluxes) fluxes->update_half();
        break;

      case OpKind::update_flux:
        if (fluxes) fluxes->update();
        break;

      case OpKind::increment_time: t += 1; break;

      case OpKind::update_dft: update_dfts(); break;

      case OpKind::synchronize_magnetic_fields:
        /* finished_working() closes the Stepping scope *before* the re-synch,
           exactly as the procedural body did, so timing attribution does not
           move. */
        finished_working();
        if (save_synchronized_magnetic_fields) {
          synchronize_magnetic_fields();
          synchronized_magnetic_fields = save_synchronized_magnetic_fields;
        }
        break;

      case OpKind::finite_value_check:
        /* Owned by advance(), which knows the MEEP_FINITE_CHECK mode. Present
           in the plan so Phase 2 can lower it. */
        break;

      /* Not emitted by the CPU builder; the vocabulary exists for Phase 2. */
      case OpKind::zero_boundary:
      case OpKind::pack_halo:
      case OpKind::unpack_halo:
      case OpKind::exchange_local:
      case OpKind::reduction:
      case OpKind::host_callback:
      case OpKind::pack_state:
      case OpKind::unpack_state:
      case OpKind::num_kinds: break;
    }
  }
}


} // namespace meep
