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

#include <stdio.h>

#include "backend/step_plan.hpp"
#include "backend/halo_plan.hpp"
#include "meep_internals.hpp"

namespace meep {

const char *op_kind_name(OpKind k) {
  switch (k) {
    case OpKind::restore_magnetic_fields: return "restore_magnetic_fields";
    case OpKind::phase_material: return "phase_material";
    case OpKind::update_material_coefficients: return "update_material_coefficients";
    case OpKind::evaluate_source_scalars: return "evaluate_source_scalars";
    case OpKind::update_db: return "update_db";
    case OpKind::update_eh: return "update_eh";
    case OpKind::update_polarization: return "update_polarization";
    case OpKind::apply_sources: return "apply_sources";
    case OpKind::zero_boundary: return "zero_boundary";
    case OpKind::pack_halo: return "pack_halo";
    case OpKind::transfer_halo: return "transfer_halo";
    case OpKind::exchange_local: return "exchange_local";
    case OpKind::unpack_halo: return "unpack_halo";
    case OpKind::update_flux_half: return "update_flux_half";
    case OpKind::update_flux: return "update_flux";
    case OpKind::increment_time: return "increment_time";
    case OpKind::update_dft: return "update_dft";
    case OpKind::synchronize_magnetic_fields: return "synchronize_magnetic_fields";
    case OpKind::finite_value_check: return "finite_value_check";
    case OpKind::reduction: return "reduction";
    case OpKind::host_callback: return "host_callback";
    case OpKind::pack_state: return "pack_state";
    case OpKind::unpack_state: return "unpack_state";
    case OpKind::num_kinds: break;
  }
  return "?";
}

static const char *ft_name(field_type ft) {
  switch (ft) {
    case E_stuff: return "E";
    case H_stuff: return "H";
    case D_stuff: return "D";
    case B_stuff: return "B";
    case PE_stuff: return "PE";
    case PH_stuff: return "PH";
    case WE_stuff: return "WE";
    case WH_stuff: return "WH";
    default: return "?";
  }
}

namespace {

class StepPlanBuilder {
public:
  explicit StepPlanBuilder(StepProgram program) { plan_.program = program; }

  void add(OpKind k, field_type ft = field_type(NUM_FIELD_TYPES), Guard g = guard_always(),
           double src_offset = 0.0) {
    Operation op;
    op.kind = k;
    op.descriptor_index = 0;
    op.guard = g;
    op.ft = ft;
    op.source_time_offset = src_offset;
    plan_.operations.push_back(op);
  }

  void add_if(bool present, OpKind k, field_type ft = field_type(NUM_FIELD_TYPES),
              double src_offset = 0.0) {
    if (present) add(k, ft, guard_static(true), src_offset);
  }

  /* One semantic boundary step.
   *
   * The plan asks for this to expand into zero-metal, pack, transfer and
   * unpack. On CPU it stays fused into a single transfer_halo, for two
   * concrete reasons in step_boundaries: comms_manager's lifetime spans all
   * three communication phases (its destructor is what completes the
   * outstanding requests), and the receives are posted *before* the metal
   * zeroing, so the plan's nominal order is not the implementation's order
   * either. Splitting it would change the request-completion point and the
   * timing scopes and buys nothing on CPU.
   *
   * The four OpKinds exist in the enum because Phase 2 needs the vocabulary;
   * see the deviation note in ~/meep-phase1-pr5.md. */
  void add_boundaries(field_type ft, Guard g = guard_always()) {
    add(OpKind::transfer_halo, ft, g);
  }

  StepPlan finish() {
    uint64_t sig = 0xcbf29ce484222325ull;
    for (const Operation &op : plan_.operations) {
      sig ^= uint64_t(op.kind) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.ft) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.guard.kind) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
    }
    plan_.signature = sig;
    return plan_;
  }

private:
  StepPlan plan_;
};

} // namespace

/* Transcribed from fields::step_once. Read the two side by side.
 *
 * step_boundaries() begins with zero_metal for every owned chunk and then does
 * pack/transfer/unpack, which is why add_boundaries emits zero_boundary first.
 */
StepPlan build_step_plan(fields &f, StepProgram program) {
  StepPlanBuilder p(program);
  const bool cw = program == StepProgram::solve_cw;

  const bool has_sources = f.sources != NULL;
  const bool has_fluxes = f.fluxes != NULL;
  const bool phasing = f.is_phasing();
  /* update_dfts is disabled under solve_cw (see dft.cpp), so the operation is
     statically absent rather than guarded.

     dft_chunks are per-chunk, so a rank owning no chunk that intersects a
     monitor builds a program of a different shape from its peers. On CPU that
     is harmless -- update_dfts() is a no-op with no chunks -- and it is
     deliberately left alone.

     Reducing it with or_to_all here DEADLOCKS, and it is worth being precise
     about why, because it is the plan's section 6.4 hazard showing up for
     real. build_step_plan runs from step_plan_for(), which rebuilds when
     dirty_executable is set -- and dirty_executable can be set on a *subset*
     of ranks, because the lazy-allocation sites in step_db, update_eh and
     update_pols are rank-local by design (each returns a flag saying "I
     allocated, reconnect"). One rank rebuilds, enters the reduction, and waits
     for peers that never arrive. Observed: tests/flux hung indefinitely at
     np=2 with the reduction in place.

     The real fix is to make the *rebuild decision* collective, the way
     connect_chunks() gates on sync_chunk_connections(). That is follow-up
     work, recorded in ~/meep-phase1-pr5.md -- and it applies equally to the
     collectives PR 4 put inside classify(). */
  bool has_dfts = false;
  for (int i = 0; i < f.num_chunks && !has_dfts; ++i)
    if (f.chunks[i]->is_mine() && f.chunks[i]->dft_chunks) has_dfts = true;

  /* Magnetic re-synchronization is a graph_variant: the whole program differs
     when synchronized fields are active. */
  p.add(OpKind::restore_magnetic_fields, field_type(NUM_FIELD_TYPES), guard_variant(0));

  /* phase_material's conditional E/H reconciliation is a segment_boundary: the
     condition is an or_to_all over all chunks, so the host has to evaluate it
     between device segments. */
  if (phasing) {
    p.add(OpKind::phase_material, field_type(NUM_FIELD_TYPES), guard_static(true));
    p.add(OpKind::evaluate_source_scalars, field_type(NUM_FIELD_TYPES), guard_segment(0), 0.5);
    p.add(OpKind::update_eh, H_stuff, guard_segment(0));
    p.add_boundaries(H_stuff, guard_segment(0));
    p.add(OpKind::evaluate_source_scalars, field_type(NUM_FIELD_TYPES), guard_segment(0), 1.0);
    p.add(OpKind::update_eh, E_stuff, guard_segment(0));
    p.add_boundaries(E_stuff, guard_segment(0));
  }

  p.add(OpKind::update_material_coefficients);

  p.add_if(has_sources, OpKind::evaluate_source_scalars, field_type(NUM_FIELD_TYPES), 0.0);
  p.add(OpKind::update_db, B_stuff);
  p.add(OpKind::apply_sources, B_stuff);
  p.add_boundaries(B_stuff);

  p.add_if(has_sources, OpKind::evaluate_source_scalars, field_type(NUM_FIELD_TYPES), 0.5);
  p.add(OpKind::update_eh, H_stuff);
  p.add_boundaries(WH_stuff);
  p.add(OpKind::update_polarization, H_stuff);
  p.add_boundaries(PH_stuff);
  p.add_boundaries(H_stuff);

  p.add_if(has_fluxes, OpKind::update_flux_half);

  p.add_if(has_sources, OpKind::evaluate_source_scalars, field_type(NUM_FIELD_TYPES), 0.5);
  p.add(OpKind::update_db, D_stuff);
  p.add(OpKind::apply_sources, D_stuff);
  p.add_boundaries(D_stuff);

  p.add_if(has_sources, OpKind::evaluate_source_scalars, field_type(NUM_FIELD_TYPES), 1.0);
  p.add(OpKind::update_eh, E_stuff);
  p.add_boundaries(WE_stuff);
  p.add(OpKind::update_polarization, E_stuff);
  p.add_boundaries(PE_stuff);
  p.add_boundaries(E_stuff);

  p.add_if(has_fluxes, OpKind::update_flux);
  p.add(OpKind::increment_time);
  /* solve_cw flattens all D/B state plus f_u, f_cond, f_bfast, the f_w
     companion and (where f_w exists) the E/H array into one contiguous
     complex<realnum> vector on every BiCGSTAB matrix-vector product. The ops
     are recorded so Phase 2 can lower them; on CPU the packing stays in
     fields_to_array / array_to_fields, whose element ordering these ops must
     match exactly. */
  if (cw) {
    p.add(OpKind::unpack_state);
    p.add(OpKind::pack_state);
  }
  /* The decimation predicate is a device_predicate: the node stays in the
     graph and the kernel returns early when the step is not due. */
  if (has_dfts && !cw) p.add(OpKind::update_dft, field_type(NUM_FIELD_TYPES), guard_device(0));
  p.add(OpKind::synchronize_magnetic_fields, field_type(NUM_FIELD_TYPES), guard_variant(0));
  p.add(OpKind::finite_value_check);

  return p.finish();
}

void format_step_plan(const StepPlan &p, std::vector<std::string> &out) {
  out.clear();
  char buf[128];
  for (const Operation &op : p.operations) {
    if (op.ft == field_type(NUM_FIELD_TYPES))
      snprintf(buf, sizeof buf, "%s", op_kind_name(op.kind));
    else
      snprintf(buf, sizeof buf, "%s(%s)", op_kind_name(op.kind), ft_name(op.ft));
    out.push_back(buf);
  }
}

} // namespace meep
