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
#include "backend/descriptors.hpp"
#include "backend/lifecycle.hpp"
#include "backend/halo_plan.hpp"
#include "backend/prepare.hpp"
#include "backend/step_plan.hpp"
#include "backend/backend.hpp"
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
/* Routes through the selected backend: one virtual call per batch, never per
   operation and certainly never per voxel. The CPU backend calls straight back
   into advance_cpu(). */
void fields::advance(int n) {
  if (n <= 0) return;
  init_backend();
  ensure_backend_executable();
  backend_refresh_noisy_seed(*this, *step_plans[0], "fields::advance noisy seed refresh");
  const bool collective_custom_dispatch =
      backend->requires_full_storage_preparation() &&
      backend_state->host_custom_preflight_required;
  std::string custom_prepare_error;
  bool custom_prepare_poisoned = false;
  try {
    backend_prepare_host_custom_dispatch(*this, *executable, *backend_state, n,
                                         "fields::advance custom fallback preflight");
  }
  catch (const std::exception &e) {
    custom_prepare_error = e.what();
    custom_prepare_poisoned = backend->is_poisoned();
  }
  catch (...) {
    custom_prepare_error = "unknown host custom fallback preflight failure";
    custom_prepare_poisoned = backend->is_poisoned();
  }
  if (collective_custom_dispatch) {
    size_t local_status = size_t(!custom_prepare_error.empty()) |
                          (size_t(custom_prepare_poisoned) << 1);
    size_t global_status = 0;
    backend_note_host_custom_collective_for_testing();
    bw_or_to_all(&local_status, &global_status, 1);
    if (global_status & 1) {
      backend_discard_host_custom_dispatch(*this);
      if (global_status & 2) backend->poison();
      if (custom_prepare_error.empty())
        throw std::runtime_error(
            "host custom fallback preflight failed on another MPI rank");
      throw std::runtime_error(custom_prepare_error);
    }
  }
  else if (!custom_prepare_error.empty())
    throw std::runtime_error(custom_prepare_error);
  const bool collective_noisy_dispatch =
      backend->requires_full_storage_preparation() && backend_state->noisy_preflight_required;
  const bool collective_dispatch = collective_noisy_dispatch || collective_custom_dispatch;
  if (!collective_dispatch) {
    try {
      backend->advance(*executable, *backend_state, n);
      (void)backend_finish_host_custom_dispatch(
          *this, "fields::advance custom fallback dispatch");
    }
    catch (...) {
      const bool poison = backend_abort_host_custom_dispatch(*this);
      if (backend->requires_full_storage_preparation() && poison) backend->poison();
      throw;
    }
    return;
  }

  std::string local_error;
  bool local_poison = false;
  bool local_crossed_callback = false;
  try {
    backend->advance(*executable, *backend_state, n);
    local_crossed_callback = backend_finish_host_custom_dispatch(
        *this, "fields::advance custom fallback dispatch");
  }
  catch (const std::exception &e) {
    local_poison = backend_abort_host_custom_dispatch(*this);
    local_error = e.what();
  }
  catch (...) {
    local_poison = backend_abort_host_custom_dispatch(*this);
    local_error = "unknown backend advance failure";
  }
  if (collective_noisy_dispatch) backend_note_noisy_collective_for_testing();
  if (collective_custom_dispatch) backend_note_host_custom_collective_for_testing();
  size_t local_status = size_t(!local_error.empty()) |
                        (size_t(local_poison || collective_noisy_dispatch) << 1) |
                        (size_t(local_crossed_callback) << 2);
  size_t global_status = 0;
  bw_or_to_all(&local_status, &global_status, 1);
  const bool failed = (global_status & 1) != 0;
  if (failed) {
    if (global_status & (size_t(2) | size_t(4))) backend->poison();
    if (local_error.empty())
      throw std::runtime_error("backend advance failed on another MPI rank");
    throw std::runtime_error(local_error);
  }
}

void fields::ensure_backend_executable() {
  /* A legacy flux add/remove keeps storage stable but changes collective
     region recipes and both flux-marker access sets. Stage descriptors, both
     plans, and the replacement executable as one resident transaction. */
  if (backend_try_refresh_legacy_flux(*this, "fields::advance legacy flux refresh")) {
    bool local_noisy = false, local_multilevel = has_local_exact_multilevel(*this);
    for (const PolarizationUpdate &update : step_plans[0]->polarization_updates)
      local_noisy = local_noisy || update.kind == PolarizationUpdateKind::noisy_add;
    for (const PolarizationUpdateGroup &group : step_plans[0]->polarization_groups)
      local_multilevel =
          local_multilevel || group.kind == PolarizationGroupKind::multilevel;
    size_t local_presence = size_t(local_noisy) | (size_t(local_multilevel) << 1);
    size_t global_presence = 0;
    bw_or_to_all(&local_presence, &global_presence, 1);
    backend_state->noisy_preflight_required = (global_presence & 1) != 0;
    backend_state->noisy_static_validation_required =
        backend_state->noisy_preflight_required;
    backend_state->multilevel_preflight_required = (global_presence & 2) != 0;
    backend_state->multilevel_static_validation_required = false;
    backend_state->multilevel_plan_validated = backend_state->multilevel_preflight_required;
    backend_state->multilevel_validated_plan_signature = step_plans[0]->signature;
    if (backend_state->host_custom_policy_pending) {
      backend_publish_host_custom_policy(*this, backend_state->host_custom_local_presence,
                                         backend_state->host_custom_preflight_required);
      backend_state->host_custom_policy_pending = false;
    }
    return;
  }

  /* step_plan_for clears dirty_executable, so remember whether the compiled
     backend artifact was stale before asking it to rebuild the data plan. */
  const DirtyMask entry_dirty_mask = DirtyMask(dirty_mask);
  const bool local_recompile =
      !executable || is_dirty(*this, dirty_executable) ||
      (backend->requires_full_storage_preparation() && backend_state &&
       backend_state->host_custom_policy_pending);
  const StepPlan &plan = step_plan_for(StepProgram::ordinary);
  bool recompile = local_recompile;
  bool staged_multilevel_presence = false;
  bool staged_multilevel_validation = false;
  bool staged_custom_presence = false;
  bool staged_custom_validation = false;
  if (backend->requires_full_storage_preparation()) {
    bool local_noisy = false, local_multilevel = has_local_exact_multilevel(*this);
    for (const PolarizationUpdate &update : plan.polarization_updates)
      local_noisy = local_noisy || update.kind == PolarizationUpdateKind::noisy_add;
    for (const PolarizationUpdateGroup &group : plan.polarization_groups)
      local_multilevel =
          local_multilevel || group.kind == PolarizationGroupKind::multilevel;
    const bool inspect_noisy_signature = local_noisy || backend_state->noisy_plan_validated ||
                                         backend_state->noisy_preflight_required;
    const bool inspect_multilevel_signature =
        local_multilevel || backend_state->multilevel_plan_validated ||
        backend_state->multilevel_preflight_required;
    const bool local_custom = backend_state->host_custom_presence_validated
                                  ? backend_state->host_custom_local_presence
                                  : backend->host_custom_fallback_enabled();
    const bool inspect_custom_signature =
        local_custom || backend_state->host_custom_plan_validated ||
        backend_state->host_custom_preflight_required;
    const uint64_t recomputed_signature =
        (inspect_noisy_signature || inspect_multilevel_signature || inspect_custom_signature)
            ? compute_step_plan_signature(plan)
            : plan.signature;
    const bool changed_validated_plan = backend_state->noisy_plan_validated &&
                                        (backend_state->noisy_validated_plan_signature !=
                                             plan.signature ||
                                         recomputed_signature != plan.signature);
    const bool preserve_validated_presence = backend_state->noisy_plan_validated &&
                                             backend_state->noisy_preflight_required;
    const bool require_static_validation =
        changed_validated_plan || (local_noisy && !backend_state->noisy_plan_validated);
    const bool changed_validated_multilevel_plan =
        backend_state->multilevel_plan_validated &&
        (backend_state->multilevel_validated_plan_signature != plan.signature ||
         recomputed_signature != plan.signature);
    const bool preserve_validated_multilevel_presence =
        backend_state->multilevel_plan_validated &&
        backend_state->multilevel_preflight_required;
    const bool require_multilevel_validation =
        backend_state->multilevel_static_validation_required ||
        changed_validated_multilevel_plan ||
        (local_multilevel && !backend_state->multilevel_plan_validated);
    const bool changed_validated_custom_plan =
        backend_state->host_custom_plan_validated &&
        (backend_state->host_custom_validated_plan_signature != plan.signature ||
         (inspect_custom_signature && recomputed_signature != plan.signature));
    const bool preserve_validated_custom_presence =
        backend_state->host_custom_plan_validated &&
        backend_state->host_custom_preflight_required;
    const bool require_custom_validation =
        changed_validated_custom_plan ||
        (local_custom && !backend_state->host_custom_plan_validated);
    size_t local_status = size_t(local_recompile) | (size_t(local_noisy) << 1) |
                          (size_t(preserve_validated_presence) << 2) |
                          (size_t(require_static_validation) << 3) |
                          (size_t(local_multilevel) << 4) |
                          (size_t(preserve_validated_multilevel_presence) << 5) |
                          (size_t(require_multilevel_validation) << 6) |
                          (size_t(local_custom) << 7) |
                          (size_t(preserve_validated_custom_presence) << 8) |
                          (size_t(require_custom_validation) << 9);
    size_t global_status = 0;
    bw_or_to_all(&local_status, &global_status, 1);
    recompile =
        (global_status & (size_t(1) | (size_t(1) << 6) | (size_t(1) << 9))) != 0;
    backend_state->noisy_preflight_required =
        (global_status & (size_t(2) | size_t(4))) != 0;
    backend_state->noisy_static_validation_required = (global_status & size_t(8)) != 0;
    staged_multilevel_presence =
        (global_status & ((size_t(1) << 4) | (size_t(1) << 5))) != 0;
    staged_multilevel_validation = (global_status & (size_t(1) << 6)) != 0;
    staged_custom_presence =
        (global_status & ((size_t(1) << 7) | (size_t(1) << 8))) != 0;
    staged_custom_validation = (global_status & (size_t(1) << 9)) != 0;
  }
  if (!recompile) return;

  Executable *previous = executable;
  Executable *replacement = NULL;
  std::string local_error;
  try {
    if (staged_multilevel_validation) {
      bool local_multilevel_actions = false;
      local_error = backend_validate_multilevel_plan(*this, plan, local_multilevel_actions);
      if (local_error.empty() && local_multilevel_actions != has_local_exact_multilevel(*this))
        local_error = "multilevel live state and installed action presence differ";
    }
    if (local_error.empty() && staged_custom_validation) {
      backend_validate_host_custom_plan(*this, plan, *backend_state);
    }
    if (local_error.empty()) {
      replacement = backend->compile(plan, *backend_state);
      if (!replacement) throw std::runtime_error("backend returned no executable");
    }
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown backend compilation failure";
  }
  const bool failed = backend->requires_full_storage_preparation() ? or_to_all(!local_error.empty())
                                                                   : !local_error.empty();
  if (backend->requires_full_storage_preparation() && staged_multilevel_validation)
    backend_note_multilevel_collective_for_testing();
  if (backend->requires_full_storage_preparation() && staged_custom_validation)
    backend_note_host_custom_collective_for_testing();
  if (failed) {
    delete replacement;
    executable = previous;
    dirty_mask = entry_dirty_mask;
    if (local_error.empty())
      throw std::runtime_error("backend compilation failed on another MPI rank");
    throw std::runtime_error(local_error);
  }
  executable = replacement;
  delete previous;
  if (staged_multilevel_validation) {
    backend_state->multilevel_preflight_required = staged_multilevel_presence;
    backend_state->multilevel_static_validation_required = false;
    backend_state->multilevel_plan_validated = staged_multilevel_presence;
    backend_state->multilevel_validated_plan_signature = plan.signature;
  }
  if (staged_custom_validation) {
    backend_state->host_custom_preflight_required = staged_custom_presence;
    backend_state->host_custom_plan_validated = staged_custom_presence;
    backend_state->host_custom_validated_plan_signature = plan.signature;
  }
  if (backend_state->host_custom_policy_pending) {
    backend_publish_host_custom_policy(*this, backend_state->host_custom_local_presence,
                                       backend_state->host_custom_preflight_required);
    backend_state->host_custom_policy_pending = false;
  }
}

void fields::advance_cpu(int n) {
  if (n <= 0) return;
  /* Storage is realized before the loop, never inside it. From here the
     timestep only executes. */
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

  /* solve_cw drives fields::step() with doing_solve_cw set (cw_fields.cpp:91,
     151, 243). It is a genuinely different timestep program -- step_source
     skips non-integrated sources, update_dfts is disabled -- so the program has
     to be selected here. Running the ordinary plan under solve_cw is the one
     failure mode in this stack that produces wrong physics silently rather than
     crashing. */
  /* The three times the plan's evaluate_source_scalars operations use.
   *
   * Written this way on purpose. `time() + 0.5 * dt` expands to
   * `(double)t * dt + 0.5 * dt`, and GCC may contract the t*dt multiply into
   * the add; whether it does depends on the surrounding code, and the two
   * forms differ in the last bit. That is not a stable value to build on -- it
   * flipped merely from moving the expression, which the bitwise harness
   * caught as a 1-ULP field difference at a custom source's location.
   *
   * cw_source_time is shared with the legacy and resident CW paths and is kept
   * out of line so the product rounds before the offset addition. */
  step_source_times[0] = cw_source_time(t, dt, 0.0);
  step_source_times[1] = cw_source_time(t, dt, 0.5);
  step_source_times[2] = cw_source_time(t, dt, 1.0);

  const bool cw = num_chunks && chunks[0]->is_solving_cw();
  execute_step_plan(step_plan_for(cw ? StepProgram::solve_cw : StepProgram::ordinary),
                    save_synchronized_magnetic_fields);

  changed_materials = false; // any material changes were handled in connect_chunks()
  note_connection_sync_done(*this);
  assert_local_invalidation_shadow(*this, changed_materials, "step_once end");
}

/* Build the plan for `program` if it is stale, and return it.
 *
 * A plan is rebuilt only when something it depends on changes -- dirty_executable
 * is the trigger, and PR 4's classification hash is what decides whether a
 * material change reaches it. */
const StepPlan &fields::step_plan_for(StepProgram program) {
  const int idx = program == StepProgram::solve_cw ? 1 : 0;
  if (!step_plans[idx] || is_dirty(*this, dirty_executable)) {
    if (!step_plans[idx]) step_plans[idx] = new StepPlan;
    *step_plans[idx] = build_step_plan(*this, program);
    /* Both programs are rebuilt together: solve_cw is entered and left by
       flipping doing_solve_cw, and a stale CW plan is the one failure mode in
       this stack that produces wrong physics rather than a crash. */
    const int other = 1 - idx;
    if (step_plans[other])
      *step_plans[other] =
          build_step_plan(*this, other ? StepProgram::solve_cw : StepProgram::ordinary);
    clear_dirty(*this, dirty_executable);
  }
  return *step_plans[idx];
}

/* The mixing half of what used to be fields::phase_material. The conditional
   E/H reconciliation that followed it is now a segment_boundary-guarded block
   in the plan; the guard is this function's return value, and it is collective
   (or_to_all over all chunks), so every rank has to reach it. */
bool fields::phase_material_mix() {
  bool changed = false;
  if (!is_phasing()) return false;
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
  return changed_mpi;
}

void fields::phase_material() {
  if (phase_material_mix()) {
    calc_sources(time() + 0.5 * dt); // for integrated H sources
    update_eh(H_stuff);              // ensure H = 1/mu * B
    step_boundaries(H_stuff);
    calc_sources(time() + dt); // for integrated E sources
    update_eh(E_stuff);        // ensure E = 1/eps * D
    step_boundaries(E_stuff);
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
  side_cursor(const HaloArrayTable &tbl, const HostHaloArrayTable &host_tbl,
              const std::vector<SlabRef> &slabs, const std::vector<ElementRef> &residue,
              const std::vector<HaloSegment> &order, const std::vector<HostElementRef> &host,
              HaloStorageDisposition storage)
      : tbl_(tbl), host_tbl_(host_tbl), slabs_(slabs), residue_(residue), order_(order),
        host_(host), use_host_(storage == HaloStorageDisposition::host_owned) {}

  realnum *next() {
    if (use_host_)
      return host_pos_ < host_.size() ? host_tbl_.address(host_[host_pos_++].id) : NULL;
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
  const HostHaloArrayTable &host_tbl_;
  const std::vector<SlabRef> &slabs_;
  const std::vector<ElementRef> &residue_;
  const std::vector<HaloSegment> &order_;
  const std::vector<HostElementRef> &host_;
  bool use_host_;
  size_t host_pos_ = 0;
  size_t seg_ = 0, r_ = 0;
  uint32_t k_ = 0, s_ = 0, rk_ = 0;
};
} // namespace

void fields::unpack_halo(const HaloPlan &p, const realnum *block) {
  if (!p.block_elements) return;
  if (p.storage == HaloStorageDisposition::host_owned &&
      p.host_scatter.size() != p.block_elements)
    meep::abort("opaque host scatter length differs from its communication block");
  side_cursor cur(halos->arrays, halos->host_arrays, p.scatter_slabs, p.scatter, p.scatter_order,
                  p.host_scatter, p.storage);
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
  if (p.storage == HaloStorageDisposition::host_owned && p.host_gather.size() != p.block_elements)
    meep::abort("opaque host gather length differs from its communication block");
  side_cursor cur(halos->arrays, halos->host_arrays, p.gather_slabs, p.gather, p.gather_order,
                  p.host_gather, p.storage);
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
        if (const HaloPlan *p = halos->find({ft, ip, comm_pair}))
          pack_halo(*p, outgoing_comm_block);
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

/* The source-scalar evaluation the plan schedules.
 *
 * This lives in step.cpp, textually where fields::step_once used to compute it,
 * and that placement is load-bearing rather than sentimental. `time() + 0.5*dt`
 * is `(double)t * dt + 0.5 * dt`, and GCC may contract the t*dt multiply into
 * the add. Whether it does depends on the surrounding code, and the two forms
 * differ by 1 ULP -- which reaches a custom source's Python callback and shows
 * up as a 1-ULP field difference at the source point. The bitwise harness
 * caught exactly that when this expression was evaluated inside the executor's
 * dispatch loop instead. */
void fields::evaluate_source_scalars(double offset_in_dt) {
  calc_sources(step_source_times[offset_in_dt == 0.0 ? 0 : offset_in_dt == 0.5 ? 1 : 2]);
  populate_source_scalars(*this, descriptors->sources);
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
