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

#include <assert.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string.h>
#include <vector>

#include "backend/backend.hpp"
#include "backend/prepare.hpp"
#include "backend/lifecycle.hpp"
#include "backend/storage_plan.hpp"
#include "backend/classification.hpp"
#include "backend/descriptors.hpp"
#include "meep_internals.hpp"

namespace meep {

namespace {

int material_prepare_failure_rank = -1;
int material_prepare_failure_after = -1;

bool material_phase_storage_needs_update(const structure_chunk &current,
                                         const structure_chunk &target) {
  if (current.refcount > 1 || current.condinv_stale) return true;
  FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
    if (target.chi1inv[c][d] &&
        (!current.chi1inv[c][d] ||
         (current.trivial_chi1inv[c][d] && !target.trivial_chi1inv[c][d])))
      return true;
    if (target.conductivity[c][d] && !current.conductivity[c][d]) return true;
    if (d == component_direction(c) && current.conductivity[c][d] && !current.condinv[c][d])
      return true;
  }
  return false;
}

bool material_coefficient_storage_needs_update(const structure_chunk &current) {
  FOR_COMPONENTS(c) {
    const direction d = component_direction(c);
    const bool have_conductivity = current.conductivity[c][d] != NULL;
    const bool have_inverse = current.condinv[c][d] != NULL;
    if ((have_conductivity || have_inverse) &&
        (current.condinv_stale || have_conductivity != have_inverse))
      return true;
  }
  return false;
}

void realize_material_phase_union(structure_chunk &current, const structure_chunk &target) {
  bool condinv_refresh_required = false;
  FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
    if (target.chi1inv[c][d]) {
      if (!current.chi1inv[c][d]) {
        current.chi1inv[c][d] = new realnum[current.gv.ntot()];
        const realnum initial = component_direction(c) == d ? realnum(1) : realnum(0);
        std::fill(current.chi1inv[c][d], current.chi1inv[c][d] + current.gv.ntot(), initial);
      }
      current.trivial_chi1inv[c][d] =
          current.trivial_chi1inv[c][d] && target.trivial_chi1inv[c][d];
    }
    if (target.conductivity[c][d] && !current.conductivity[c][d]) {
      current.conductivity[c][d] = new realnum[current.gv.ntot()];
      std::fill(current.conductivity[c][d],
                current.conductivity[c][d] + current.gv.ntot(), realnum(0));
      condinv_refresh_required = true;
    }
    if (d == component_direction(c) && current.conductivity[c][d] && !current.condinv[c][d])
      condinv_refresh_required = true;
  }
  if (condinv_refresh_required) current.condinv_stale = true;
  current.update_condinv();
}

} // namespace

void set_material_phase_prepare_failure_for_testing(int rank, int after_chunks) {
  material_prepare_failure_rank = rank;
  material_prepare_failure_after = after_chunks;
}

PreparedMaterialPhaseStorage::PreparedMaterialPhaseStorage(fields &f, const structure &target)
    : PreparedMaterialPhaseStorage(f, &target) {}

PreparedMaterialPhaseStorage::PreparedMaterialPhaseStorage(fields &f)
    : PreparedMaterialPhaseStorage(f, NULL) {}

PreparedMaterialPhaseStorage::PreparedMaterialPhaseStorage(fields &f, const structure *target)
    : owner_(f), chunks_(size_t(f.num_chunks)), committed_(false) {
  int staged = 0;
  for (int i = 0; i < owner_.num_chunks; ++i) {
    if (!owner_.chunks[i]->is_mine()) continue;
    const structure_chunk *target_chunk =
        target ? target->chunks[i] : owner_.chunks[i]->new_s;
    if (!target_chunk)
      throw std::logic_error("active material phase has no retained target chunk");
    if (!material_phase_storage_needs_update(*owner_.chunks[i]->s, *target_chunk)) continue;
    chunks_[size_t(i)].reset(new structure_chunk(owner_.chunks[i]->s));
    realize_material_phase_union(*chunks_[size_t(i)], *target_chunk);
    ++staged;
    if (my_rank() == material_prepare_failure_rank &&
        staged == material_prepare_failure_after)
      throw std::runtime_error("injected material phase storage preparation failure");
  }
}

PreparedMaterialPhaseStorage::~PreparedMaterialPhaseStorage() {}

void PreparedMaterialPhaseStorage::commit() {
  if (committed_) return;
  for (int i = 0; i < owner_.num_chunks; ++i) {
    if (!chunks_[size_t(i)]) continue;
    structure_chunk *old = owner_.chunks[i]->s;
    owner_.chunks[i]->s = chunks_[size_t(i)].release();
    if (old->refcount-- <= 1) delete old;
  }
  committed_ = true;
}

std::unique_ptr<PreparedMaterialPhaseStorage> prepare_material_phase_storage(
    fields &f, const structure &target) {
  return std::unique_ptr<PreparedMaterialPhaseStorage>(
      new PreparedMaterialPhaseStorage(f, target));
}

std::unique_ptr<PreparedMaterialPhaseStorage> prepare_material_phase_storage(fields &f) {
  return std::unique_ptr<PreparedMaterialPhaseStorage>(new PreparedMaterialPhaseStorage(f));
}

PreparedMaterialCoefficientStorage::PreparedMaterialCoefficientStorage(fields &f)
    : owner_(f), chunks_(size_t(f.num_chunks)), committed_(false) {
  for (int i = 0; i < owner_.num_chunks; ++i) {
    if (!owner_.chunks[i]->is_mine() ||
        !material_coefficient_storage_needs_update(*owner_.chunks[i]->s))
      continue;
    chunks_[size_t(i)].reset(new structure_chunk(owner_.chunks[i]->s));
    chunks_[size_t(i)]->condinv_stale = true;
    chunks_[size_t(i)]->update_condinv();
  }
}

PreparedMaterialCoefficientStorage::~PreparedMaterialCoefficientStorage() {}

void PreparedMaterialCoefficientStorage::commit() {
  if (committed_) return;
  for (int i = 0; i < owner_.num_chunks; ++i) {
    if (!chunks_[size_t(i)]) continue;
    structure_chunk *old = owner_.chunks[i]->s;
    owner_.chunks[i]->s = chunks_[size_t(i)].release();
    if (old->refcount-- <= 1) delete old;
  }
  committed_ = true;
}

std::unique_ptr<PreparedMaterialCoefficientStorage>
prepare_material_coefficient_storage(fields &f) {
  return std::unique_ptr<PreparedMaterialCoefficientStorage>(
      new PreparedMaterialCoefficientStorage(f));
}

/* Every condition below is copied verbatim from the lazy site it replaces.
   They are transcriptions, not reinterpretations: if one of them drifts, the
   array either stops existing (wrong physics) or starts existing when it did
   not (different dumped state, so the bitwise harness catches it). */

bool prepare_step_db(fields_chunk &fc, field_type ft, StoragePlan &) {
  const int is_real = fc.is_real; // DOCMP reads this
  bool allocated_u = false;
  const grid_volume &gv = fc.gv;
  const bool use_bfast =
      fc.bfast_scaled_k[0] || fc.bfast_scaled_k[1] || fc.bfast_scaled_k[2];

  DOCMP FOR_FT_COMPONENTS(ft, cc) {
    if (!fc.f[cc][cmp]) continue;
    const direction d_c = component_direction(cc);
    const direction dsig0 = cycle_direction(gv.dim, d_c, 1);
    const direction dsig = fc.s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
    const direction dsigu0 = cycle_direction(gv.dim, d_c, 2);
    const direction dsigu = fc.s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;

    if (dsig != NO_DIRECTION && fc.s->conductivity[cc][d_c] && !fc.f_cond[cc][cmp]) {
      fc.f_cond[cc][cmp] = new realnum[gv.ntot()];
      memset(fc.f_cond[cc][cmp], 0, sizeof(realnum) * gv.ntot());
    }
    if (dsigu != NO_DIRECTION && !fc.f_u[cc][cmp]) {
      fc.f_u[cc][cmp] = new realnum[gv.ntot()];
      memcpy(fc.f_u[cc][cmp], fc.f[cc][cmp], gv.ntot() * sizeof(realnum));
      allocated_u = true;
    }
    if (use_bfast && !fc.f_bfast[cc][cmp]) {
      fc.f_bfast[cc][cmp] = new realnum[gv.ntot()];
      memset(fc.f_bfast[cc][cmp], 0, sizeof(realnum) * gv.ntot());
    }

    /* f_rderiv_int is a scratch buffer for the cylindrical 1/r d(rF)/dr trick.
       Only the allocation hoists: it is refilled from the current fields on
       every step, so the fill stays in step_db. */
    if (gv.dim == Dcyl && d_c == Z && !fc.f_rderiv_int)
      fc.f_rderiv_int = new realnum[gv.ntot()];
  }
  return allocated_u;
}

void assert_step_db_prepared(const fields_chunk &fc, field_type ft) {
#ifndef NDEBUG
  const int is_real = fc.is_real;
  const grid_volume &gv = fc.gv;
  const bool use_bfast =
      fc.bfast_scaled_k[0] || fc.bfast_scaled_k[1] || fc.bfast_scaled_k[2];
  DOCMP FOR_FT_COMPONENTS(ft, cc) {
    if (!fc.f[cc][cmp]) continue;
    const direction d_c = component_direction(cc);
    const direction dsig0 = cycle_direction(gv.dim, d_c, 1);
    const direction dsig = fc.s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
    const direction dsigu0 = cycle_direction(gv.dim, d_c, 2);
    const direction dsigu = fc.s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;
    if (dsig != NO_DIRECTION && fc.s->conductivity[cc][d_c])
      assert(fc.f_cond[cc][cmp] && "step_db would allocate f_cond");
    if (dsigu != NO_DIRECTION) assert(fc.f_u[cc][cmp] && "step_db would allocate f_u");
    if (use_bfast) assert(fc.f_bfast[cc][cmp] && "step_db would allocate f_bfast");
    if (gv.dim == Dcyl && d_c == Z)
      assert(fc.f_rderiv_int && "step_db would allocate f_rderiv_int");
  }
#else
  (void)fc;
  (void)ft;
#endif
}

/* update_eh's `have_int_sources` and the per-polarization needs_P() query,
   factored out so preparation and the assertion cannot disagree. */
static bool needs_f_minus_p(fields_chunk &fc, field_type ft, component ec, int cmp) {
  field_type ft2 = ft == E_stuff ? D_stuff : B_stuff;
  if (!fc.f[ec][cmp]) return false;
  if (!fc.is_solving_cw()) {
    for (const src_vol &sv : fc.sources[ft2])
      if (sv.t()->is_integrated) return true;
  }
  for (polarization_state *p = fc.pol[ft]; p; p = p->next)
    if (p->s->needs_P(ec, cmp, fc.f)) return true;
  return false;
}

bool prepare_update_eh(fields_chunk &fc, field_type ft, bool skip_w_components, StoragePlan &) {
  const int is_real = fc.is_real; // DOCMP reads this
  const field_type ft2 = ft == E_stuff ? D_stuff : B_stuff;
  const grid_volume &gv = fc.gv;
  bool allocated_eh = false;

  /* f_minus_p moves in *both* directions: update_eh deletes it again when no
     source is integrated and no polarization needs P. Hoisting only the
     allocation would leave a stale array behind and change dumped state. */
  FOR_FT_COMPONENTS(ft, ec) {
    const component dc = field_type_component(ft2, ec);
    DOCMP {
      if (needs_f_minus_p(fc, ft, ec, cmp)) {
        if (!fc.f_minus_p[dc][cmp]) fc.f_minus_p[dc][cmp] = new realnum[gv.ntot()];
      }
      else if (fc.f_minus_p[dc][cmp]) {
        delete[] fc.f_minus_p[dc][cmp];
        fc.f_minus_p[dc][cmp] = 0;
      }
    }
  }

  bool have_f_minus_p = false;
  FOR_FT_COMPONENTS(ft2, dc) {
    if (fc.f_minus_p[dc][0]) {
      have_f_minus_p = true;
      break;
    }
  }

  DOCMP FOR_FT_COMPONENTS(ft, ec) {
    if (!fc.f[ec][cmp]) continue;
    const component dc = field_type_component(ft2, ec);
    const direction d_ec = component_direction(ec);
    const direction dsigw0 = d_ec;
    const direction dsigw = fc.s->sigsize[dsigw0] > 1 ? dsigw0 : NO_DIRECTION;

    // E/H split off from B/D (breaks the H == B alias)
    if (fc.f[ec][cmp] == fc.f[dc][cmp] &&
        (fc.s->chi1inv[ec][d_ec] || have_f_minus_p || dsigw != NO_DIRECTION)) {
      fc.f[ec][cmp] = new realnum[gv.ntot()];
      memcpy(fc.f[ec][cmp], fc.f[dc][cmp], gv.ntot() * sizeof(realnum));
      allocated_eh = true;
    }

    // W auxiliary field
    if (!fc.f_w[ec][cmp] && dsigw != NO_DIRECTION) {
      fc.f_w[ec][cmp] = new realnum[gv.ntot()];
      memcpy(fc.f_w[ec][cmp], fc.f[ec][cmp], gv.ntot() * sizeof(realnum));
      if (fc.needs_W_notowned(ec)) allocated_eh = true; // communication needed
    }

    /* Matches update_eh's `if (f_w[ec][cmp] && skip_w_components) continue;`,
       which sits *before* the f_w_prev allocation. Getting this wrong would
       create an f_w_prev that solve_cw never had, and fields::dump writes
       f_w_prev, so the harness would see it. */
    if (fc.f_w[ec][cmp] && skip_w_components) continue;

    if (fc.needs_W_prev(ec) && !fc.f_w_prev[ec][cmp])
      fc.f_w_prev[ec][cmp] = new realnum[gv.ntot()];
  }
  return allocated_eh;
}

void assert_update_eh_prepared(const fields_chunk &fc, field_type ft, bool skip_w_components) {
#ifndef NDEBUG
  const int is_real = fc.is_real;
  const field_type ft2 = ft == E_stuff ? D_stuff : B_stuff;
  fields_chunk &m = const_cast<fields_chunk &>(fc); // needs_f_minus_p takes non-const
  FOR_FT_COMPONENTS(ft, ec) {
    const component dc = field_type_component(ft2, ec);
    DOCMP {
      const bool want = needs_f_minus_p(m, ft, ec, cmp);
      assert(want == (fc.f_minus_p[dc][cmp] != 0) && "update_eh would (de)allocate f_minus_p");
    }
  }
  DOCMP FOR_FT_COMPONENTS(ft, ec) {
    if (!fc.f[ec][cmp]) continue;
    const component dc = field_type_component(ft2, ec);
    const direction d_ec = component_direction(ec);
    const direction dsigw = fc.s->sigsize[d_ec] > 1 ? d_ec : NO_DIRECTION;
    if (dsigw != NO_DIRECTION) assert(fc.f_w[ec][cmp] && "update_eh would allocate f_w");
    (void)dc;
    if (fc.f_w[ec][cmp] && skip_w_components) continue;
    if (m.needs_W_prev(ec)) assert(fc.f_w_prev[ec][cmp] && "update_eh would allocate f_w_prev");
  }
#else
  (void)fc;
  (void)ft;
  (void)skip_w_components;
#endif
}

bool prepare_polarizations(fields_chunk &fc, field_type ft, StoragePlan &) {
  bool allocated = false;
  for (polarization_state *p = fc.pol[ft]; p; p = p->next) {
    if (p->data) continue;
    void *data = p->s->new_internal_data(fc.f, fc.gv);
    if (!data) continue;
    p->data = data;
    try { p->s->init_internal_data(fc.f, fc.dt, fc.gv, p->data); }
    catch (...) {
      p->s->delete_internal_data(p->data);
      p->data = NULL;
      throw;
    }
    allocated = true;
  }
  return allocated;
}

/* --- the fields-level entry point --------------------------------------- */

/* Preparation is per field type, not global.
 *
 * The obvious implementation -- prepare everything the moment anything is
 * dirty -- is observably different from the lazy behavior it replaces, and the
 * bitwise harness caught it: a simulation that calls
 * synchronize_magnetic_fields() before its first step runs only step_db(B) and
 * update_eh(H), so the lazy path had created f_u for the B components alone.
 * Preparing all four field types at that point produces twice as many f_u
 * arrays, and fields::dump serializes exactly which arrays exist, so a
 * checkpoint taken there differs.
 *
 * Per-field-type granularity reproduces the old set exactly while still
 * keeping allocation out of every loop body. It also matches the shape of the
 * lazy code, which allocated per update call.
 */
void fields::prepare_storage_for(field_type ft) {
  bool reconnect = false;
  /* solve_cw runs update_eh with skip_w_components, which changes whether
     f_w_prev exists, so preparation has to know which program is next. */
  const bool skip_w = num_chunks && chunks[0]->is_solving_cw();

  for (int i = 0; i < num_chunks; ++i) {
    if (!chunks[i]->is_mine()) continue;
    fields_chunk &fc = *chunks[i];
    if (ft == B_stuff || ft == D_stuff) {
      reconnect |= prepare_step_db(fc, ft, *storage_plan);
    }
    else if (ft == E_stuff || ft == H_stuff) {
      reconnect |= prepare_update_eh(fc, ft, skip_w, *storage_plan);
      if (backend && backend->requires_full_storage_preparation())
        reconnect |= prepare_polarizations(fc, ft, *storage_plan);
    }
  }
  prepare_dfts(*this, *storage_plan);

  /* Allocating up front without triggering the equivalent reconnect leaves
     boundaries wrong -- the lazy paths returned a flag for exactly this. */
  if (reconnect) {
    invalidate(*this, MutationKind::field_layout);
    note_connections_invalidated(*this);
    mark_local_invalidation(*this);
    chunk_connections_valid = false;
    changed_materials = true;
    /* field_layout re-dirties storage; we are mid-preparation, so absorb it
       rather than looping. */
    clear_dirty(*this, dirty_storage);
  }

  /* Freeze: the catalog is rebuilt from whatever now exists, and from here the
     timestep is asserted not to allocate. */
  build_storage_catalog(*this, *array_catalog, *storage_plan);

  /* Descriptors are built on the real path, not only in tests. Nothing on CPU
     reads them, but a descriptor that cannot be built from the live objects is
     a descriptor that is wrong, and this way the bitwise harness covers their
     construction too. */
  refresh_operation_descriptors(*this, true);
}

void fields::prepare_storage() {
  prepare_storage_if_stale(B_stuff);
  prepare_storage_if_stale(D_stuff);
  prepare_storage_if_stale(H_stuff);
  prepare_storage_if_stale(E_stuff);
}

void fields::prepare_storage_if_stale(field_type ft) {
  if (is_dirty(*this, dirty_storage)) {
    storage_prepared_mask = 0; // everything has to be re-examined
    clear_dirty(*this, dirty_storage);
  }
  const uint32_t bit = uint32_t(1) << int(ft);
  if (storage_prepared_mask & bit) return;
  storage_prepared_mask |= bit;

  /* Pass 1: storage superset, initialization, relocatable metadata. */
  prepare_storage_for(ft);

  /* Pass 2: classify what initialization actually produced, then finalize.
     Both is_aniso2d and has_nonlinearities are collective, so this has to be
     entered on every rank -- which it is, because storage_prepared_mask is
     driven by dirty_storage, and every cause that sets it is either collective
     or routed through the and_to_all in connect_chunks. */
  classify_and_finalize();
}

/* The two-pass boundary. Kept separate from prepare_storage_for() so the
   re-entry bound below is explicit and assertable. */
void backend_classify_and_finalize(fields &f) {
  MaterialClassification cls = classify(f, *f.storage_plan);

  if (cls.hash != f.prepared_classification_hash) {
    if (f.prepared_classification_hash) f.dirty_mask |= dirty_executable;
    /* Component promotion is discovered from owned chunks, but recursive
       preparation contains collectives.  Every rank must either re-enter or
       skip it together, including ranks that own no affected chunk. */
    const bool promoted = or_to_all(apply_classification(f, cls));
    f.prepared_classification_hash = cls.hash;

    if (promoted) {
      /* Discovering an anisotropic coupling in 2D adds field components, which
         changes storage and halo topology and therefore re-enters pass 1. The
         re-entry is bounded to a single iteration because classification
         depends only on material values, which pass 1 does not modify. */
      ++f.classification_reentries;
      assert(f.classification_reentries <= 1 &&
             "classification re-entered pass 1 more than once");
      f.require_source_components();
      f.prepare_storage();
    }
  }
  else {
    /* Unchanged hash: the common case after a material *value* change. Reuse
       everything, including (from PR 5) the compiled executable. Still publish
       the tiling, since a fresh chunk may have been prepared. */
    apply_classification(f, cls);
  }
}

void fields::classify_and_finalize() { backend_classify_and_finalize(*this); }

void prepare_dfts(fields &f, StoragePlan &) {
  /* dft_chunk allocates in its constructor, not during stepping, so DFT
     storage never violated the no-allocation invariant. Nothing to do; the
     hook exists so PR 6's DftDescriptor has a home. */
  (void)f;
}

} // namespace meep
