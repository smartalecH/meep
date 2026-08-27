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

#include <array>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "backend/diagnostics.hpp"
#include "backend/lifecycle.hpp"
#include "backend/halo_plan.hpp"
#include "backend/prepare.hpp"
#include "backend/storage_plan.hpp"

#include "config.h"

#define RESTRICT

using namespace std;

namespace meep {

void fields::step() { advance(1); }

/* advance(n) is numerically equivalent to n consecutive step() calls: the
   per-step body below is untouched, and the only thing that moves is *when*
   the non-finite check runs (see backend/diagnostics.hpp).

   The plan also suggests hoisting the wall-time progress report and the
   am_now_working_on(Stepping) scope out of the per-step body. We deliberately
   do not: the Stepping scope currently closes *before* the magnetic re-synch
   and the finite check, so hoisting it would re-attribute those to Stepping,
   and hoisting the progress report would change its cadence for n > 1. Both
   are ruled out by the stronger requirement that timing scopes and progress
   output stay identical. The per-iteration cost of both is negligible next to
   a timestep. */
void fields::advance(int n) {
  if (n <= 0) return;
  /* Storage is realized before the loop, never inside it. From here the
     timestep only executes. */
  prepare_storage_if_stale();
  const FiniteCheckMode mode = finite_check_mode();
  for (int i = 0; i < n; ++i) {
    step_once();
    if (mode == FiniteCheckMode::step) check_finite_fields();
  }
  if (mode == FiniteCheckMode::batch) check_finite_fields();
}

/* The same center-point read fields::step has always performed. A per-voxel
   diagnostic would mean touching STEP_* (global rule 6) and would regress CPU
   performance; the device-native version is Phase 2 (decision D). */
void fields::check_finite_fields() {
  if (!std::isfinite(get_field(D_EnergyDensity, gv.center(), false))) {
    if (!nonfinite_flag) {
      nonfinite_flag = 1;
      first_bad_step = t;
      first_bad_component = int(D_EnergyDensity);
    }
    meep::abort("simulation fields are NaN or Inf");
  }
}

void fields::step_once() {
  // however many times the fields have been synched, we want to restore now
  int save_synchronized_magnetic_fields = synchronized_magnetic_fields;
  if (synchronized_magnetic_fields) {
    synchronized_magnetic_fields = 1; // reset synchronization count
    restore_magnetic_fields();
  }

  am_now_working_on(Stepping);

  if (!t) {
    last_step_output_wall_time = wall_time();
    last_step_output_t = t;
  }
  if (verbosity > 0 && wall_time() > last_step_output_wall_time + MEEP_MIN_OUTPUT_TIME) {
    master_printf("on time step %d (time=%g), %g s/step\n", t, time(),
                  (wall_time() - last_step_output_wall_time) / (t - last_step_output_t));
    if (save_synchronized_magnetic_fields)
      master_printf("  (doing expensive timestepping of synched fields)\n");
    last_step_output_wall_time = wall_time();
    last_step_output_t = t;
  }

  phase_material();

  // update cached conductivity-inverse array, if needed
  for (int i = 0; i < num_chunks; i++)
    chunks[i]->s->update_condinv();

  calc_sources(time()); // for B sources
  {
    auto step_timer = with_timing_scope(FieldUpdateB);
    step_db(B_stuff);
  }
  step_source(B_stuff);
  {
    auto step_timer = with_timing_scope(BoundarySteppingB);
    step_boundaries(B_stuff);
  }
  calc_sources(time() + 0.5 * dt); // for integrated H sources
  {
    auto step_timer = with_timing_scope(FieldUpdateH);
    update_eh(H_stuff);
  }
  {
    auto step_timer = with_timing_scope(BoundarySteppingWH);
    step_boundaries(WH_stuff);
  }
  update_pols(H_stuff);
  {
    auto step_timer = with_timing_scope(BoundarySteppingPH);
    step_boundaries(PH_stuff);
  }
  {
    auto step_timer = with_timing_scope(BoundarySteppingH);
    step_boundaries(H_stuff);
  }

  if (fluxes) fluxes->update_half();

  calc_sources(time() + 0.5 * dt); // for D sources
  {
    auto step_timer = with_timing_scope(FieldUpdateD);
    step_db(D_stuff);
  }
  step_source(D_stuff);
  {
    auto step_timer = with_timing_scope(BoundarySteppingD);
    step_boundaries(D_stuff);
  }
  calc_sources(time() + dt); // for integrated E sources
  {
    auto step_timer = with_timing_scope(FieldUpdateE);
    update_eh(E_stuff);
  }
  {
    auto step_timer = with_timing_scope(BoundarySteppingWE);
    step_boundaries(WE_stuff);
  }
  update_pols(E_stuff);
  {
    auto step_timer = with_timing_scope(BoundarySteppingPE);
    step_boundaries(PE_stuff);
  }
  {
    auto step_timer = with_timing_scope(BoundarySteppingE);
    step_boundaries(E_stuff);
  }

  if (fluxes) fluxes->update();
  t += 1;
  update_dfts();
  finished_working();

  // re-synch magnetic fields if they were previously synchronized
  if (save_synchronized_magnetic_fields) {
    synchronize_magnetic_fields();
    synchronized_magnetic_fields = save_synchronized_magnetic_fields;
  }

  changed_materials = false; // any material changes were handled in connect_chunks()
  note_connection_sync_done(*this);
  assert_local_invalidation_shadow(*this, changed_materials, "step_once end");
}

void fields::phase_material() {
  bool changed = false;
  if (is_phasing()) {
    CHUNK_OPENMP
    for (int i = 0; i < num_chunks; i++)
      if (chunks[i]->is_mine()) {
        chunks[i]->phase_material(phasein_time);
        changed = changed || chunks[i]->new_s;
      }
    phasein_time--;
    am_now_working_on(MpiAllTime);
    bool changed_mpi = or_to_all(changed);
    finished_working();
    if (changed_mpi) {
      calc_sources(time() + 0.5 * dt); // for integrated H sources
      update_eh(H_stuff);              // ensure H = 1/mu * B
      step_boundaries(H_stuff);
      calc_sources(time() + dt); // for integrated E sources
      update_eh(E_stuff);        // ensure E = 1/eps * D
      step_boundaries(E_stuff);
    }
  }
}

void fields_chunk::phase_material(int phasein_time) {
  if (new_s && phasein_time > 0) {
    changing_structure();
    s->mix_with(new_s, 1.0 / phasein_time);
  }
}

/* Walks one side of a HaloPlan in communication-block order, yielding the
   address of each real. Kept as an explicit cursor rather than materializing a
   pointer vector: this runs every timestep, and the whole point of the plan is
   that it does not carry addresses around. */
namespace {
class side_cursor {
public:
  side_cursor(const HaloArrayTable &tbl, const std::vector<SlabRef> &slabs,
              const std::vector<ElementRef> &residue, const std::vector<HaloSegment> &order)
      : tbl_(tbl), slabs_(slabs), residue_(residue), order_(order) {}

  realnum *next() {
    while (seg_ < order_.size()) {
      const HaloSegment &g = order_[seg_];
      if (g.nslabs) {
        if (k_ < g.count) {
          const SlabRef &sl = slabs_[g.first_slab + s_];
          realnum *p = tbl_.base(sl.array) + sl.base + ptrdiff_t(k_) * sl.strides[0];
          if (++s_ == g.nslabs) {
            s_ = 0;
            ++k_;
          }
          return p;
        }
      }
      else if (rk_ < g.residue) {
        const ElementRef &e = residue_[r_++];
        ++rk_;
        return tbl_.base(e.array) + e.index;
      }
      ++seg_;
      k_ = 0;
      s_ = 0;
      rk_ = 0;
    }
    return NULL;
  }

private:
  const HaloArrayTable &tbl_;
  const std::vector<SlabRef> &slabs_;
  const std::vector<ElementRef> &residue_;
  const std::vector<HaloSegment> &order_;
  size_t seg_ = 0, r_ = 0;
  uint32_t k_ = 0, s_ = 0, rk_ = 0;
};
} // namespace

void fields::unpack_halo(const HaloPlan &p, const realnum *block) {
  if (!p.block_elements) return;
  side_cursor cur(halos->arrays, p.scatter_slabs, p.scatter, p.scatter_order);
  const realnum *in = block + p.block_offset;

  switch (p.phase) {
    case CONNECT_PHASE: {
      const size_t num_transfers = p.block_elements / 2; // two realnums per complex
      for (size_t n = 0; n < num_transfers; ++n) {
        /* Reproduce the legacy expression exactly: phase * complex(re, im),
           in that association. Any reassociation loses bit-identity. */
        std::complex<realnum> temp =
            p.phase_values[n] * std::complex<realnum>(in[2 * n], in[2 * n + 1]);
        realnum *re = cur.next();
        realnum *im = cur.next();
        *re = temp.real();
        *im = temp.imag();
      }
      break;
    }
    case CONNECT_NEGATE:
      for (size_t n = 0; n < p.block_elements; ++n)
        *cur.next() = -in[n];
      break;
    case CONNECT_COPY:
      for (size_t n = 0; n < p.block_elements; ++n)
        *cur.next() = in[n];
      break;
  }
}

void fields::pack_halo(const HaloPlan &p, realnum *block) {
  if (!p.block_elements) return;
  side_cursor cur(halos->arrays, p.gather_slabs, p.gather, p.gather_order);
  realnum *out = block + p.block_offset;
  for (size_t n = 0; n < p.block_elements; ++n)
    out[n] = *cur.next();
}

void fields::process_incoming_chunk_data(field_type ft, const chunk_pair &comm_pair) {
  am_now_working_on(Boundaries);
  const realnum *block = comm_blocks[ft][chunk_pair_to_index(comm_pair)];
  /* Unpack in the order the sender packed. all_connect_phases is the contract;
     HaloPlan::sequence_index records each plan's position in it. */
  for (connect_phase ip : all_connect_phases)
    if (const HaloPlan *p = halos->find({ft, ip, comm_pair})) unpack_halo(*p, block);
  finished_working();
}

void fields::step_boundaries(field_type ft) {
  connect_chunks(); // re-connect if !chunk_connections_valid

  {
    // Initiate receive operations as early as possible.
    std::unique_ptr<comms_manager> manager = create_comms_manager();

    const auto &sequence = comms_sequence_for_field[ft];
    for (const comms_operation &op : sequence.receive_ops) {
      if (chunks[op.other_chunk_idx]->is_mine()) { continue; }
      chunk_pair comm_pair{op.other_chunk_idx, op.my_chunk_idx};
      comms_manager::receive_callback cb = [this, ft, comm_pair]() {
        process_incoming_chunk_data(ft, comm_pair);
      };
      manager->receive_real_async(comm_blocks[ft][op.pair_idx], static_cast<int>(op.transfer_size),
                                  op.other_proc_id, op.tag, cb);
    }

    // Do the metals first!
    for (int i = 0; i < num_chunks; i++)
      if (chunks[i]->is_mine()) zero_metal(ft, i);

    // Copy outgoing data into buffers while following the predefined sequence of comms operations.
    // Trigger the asynchronous send immediately once the outgoing comms buffer has been filled.
    am_now_working_on(Boundaries);

    for (const comms_operation &op : sequence.send_ops) {
      const std::pair<int, int> comm_pair{op.my_chunk_idx, op.other_chunk_idx};
      const int pair_idx = op.pair_idx;

      realnum *outgoing_comm_block = comm_blocks[ft][pair_idx];
      for (connect_phase ip : all_connect_phases)
        if (const HaloPlan *p = halos->find({ft, ip, comm_pair})) pack_halo(*p, outgoing_comm_block);
      if (chunks[op.other_chunk_idx]->is_mine()) { continue; }
      manager->send_real_async(comm_blocks[ft][pair_idx], static_cast<int>(op.transfer_size),
                               op.other_proc_id, op.tag);
    }

    // Process local transfers, which do not depend on a communication mechanism across nodes.
    for (const comms_operation &op : sequence.receive_ops) {
      if (chunks[op.other_chunk_idx]->is_mine()) {
        process_incoming_chunk_data(ft, {op.other_chunk_idx, op.my_chunk_idx});
      }
    }
    finished_working();

    am_now_working_on(MpiOneTime);
    // Let the communication manager drop out of scope to complete all outstanding requests.
    // As data is received, the installed callback handles copying the data from the comm buffer
    // back into the chunk field array.
  }
  finished_working();
}

void fields::step_source(field_type ft, bool including_integrated) {
  if (ft != D_stuff && ft != B_stuff) meep::abort("only step_source(D/B) is okay");
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) chunks[i]->step_source(ft, including_integrated);
}

void fields_chunk::step_source(field_type ft, bool including_integrated) {
  if (doing_solve_cw && !including_integrated) return;
  for (const src_vol &sv : sources[ft]) {
    component c = direction_component(first_field_component(ft), component_direction(sv.c));
    const realnum *cndinv = s->condinv[c][component_direction(sv.c)];
    if ((including_integrated || !sv.t()->is_integrated) && f[c][0] &&
        ((ft == D_stuff && is_electric(sv.c)) || (ft == B_stuff && is_magnetic(sv.c)))) {
      if (cndinv)
        for (size_t j = 0; j < sv.num_points(); j++) {
          const ptrdiff_t i = sv.index_at(j);
          const complex<double> A = sv.current(j) * dt * double(cndinv[i]);
          f[c][0][i] -= real(A);
          if (!is_real) f[c][1][i] -= imag(A);
        }
      else
        for (size_t j = 0; j < sv.num_points(); j++) {
          const complex<double> A = sv.current(j) * dt;
          const ptrdiff_t i = sv.index_at(j);
          f[c][0][i] -= real(A);
          if (!is_real) f[c][1][i] -= imag(A);
        }
    }
  }
}

void fields::calc_sources(double tim) {
  for (src_time *s = sources; s; s = s->next)
    s->update(tim, dt);
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) chunks[i]->calc_sources(tim);
}

void fields_chunk::calc_sources(double time) {
  (void)time; // unused;
}

} // namespace meep
