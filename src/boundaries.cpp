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

#include <algorithm>
#include <map>
#include <stdlib.h>
#include <complex>

#include "meep.hpp"
#include "meep/mympi.hpp"
#include "meep_internals.hpp"
#include "backend/lifecycle.hpp"
#include "backend/halo_plan.hpp"

#define UNUSED(x) (void)x // silence compiler warnings

using namespace std;

namespace meep {

namespace {

// Creates an optimized comms_sequence from a vector of comms_operations.
// Send operations are prioritized in descending order by the amount of data that is transferred.
comms_sequence optimize_comms_operations(const std::vector<comms_operation> &operations) {
  comms_sequence ret;
  std::map<int, size_t> send_size_by_my_chunk_idx;
  std::map<int, std::vector<comms_operation> > send_ops_by_my_chunk_idx;

  for (const auto &op : operations) {
    if (op.comm_direction == Incoming) {
      ret.receive_ops.push_back(op);
      continue;
    }

    // Group send operations by source chunk and accumulate the transfer size - excluding chunk
    // pairs that reside on the same processor.
    if (op.other_proc_id != my_rank()) {
      send_size_by_my_chunk_idx[op.my_chunk_idx] += op.transfer_size;
    }
    else {
      // Make sure that op.my_chunk_idx is represented in the map.
      send_size_by_my_chunk_idx[op.my_chunk_idx] += 0;
    }
    send_ops_by_my_chunk_idx[op.my_chunk_idx].push_back(op);
  }

  // Sort in descending order to prioritize large transfers.
  std::vector<std::pair<int, size_t> > send_op_sizes(send_size_by_my_chunk_idx.begin(),
                                                     send_size_by_my_chunk_idx.end());
  std::sort(send_op_sizes.begin(), send_op_sizes.end(),
            [](const std::pair<int, size_t> &a, const std::pair<int, size_t> &b) -> bool {
              return a.second > b.second;
            });

  // Assemble send operations.
  for (const auto &size_pair : send_op_sizes) {
    int my_chunk_idx = size_pair.first;
    const auto &ops_vector = send_ops_by_my_chunk_idx[my_chunk_idx];
    ret.send_ops.insert(std::end(ret.send_ops), std::begin(ops_vector), std::end(ops_vector));
  }
  return ret;
}

} // namespace

void fields::set_boundary(boundary_side b, direction d, boundary_condition cond) {
  if (boundaries[b][d] != cond) {
    boundaries[b][d] = cond;
    // If it's periodic, the boundary condition on the opposite side must be the same.
    if (cond == Periodic || boundaries[1 - b][d] == Periodic) boundaries[1 - b][d] = cond;
    // we don't need to call sync_chunk_connections() since set_boundary()
    // should always be called on every process
    invalidate(*this, MutationKind::boundary_topology);
    note_connections_invalidated(*this);
    chunk_connections_valid = false;
  }
}

boundary_condition fields::get_boundary(boundary_side b, direction d) const {
  return boundaries[b][d];
}

void fields::use_bloch(direction d, complex<double> kk) {
  k[d] = kk;
  for (int b = 0; b < 2; b++)
    set_boundary(boundary_side(b), d, Periodic);
  // Use user_volume (full cell) size, not gv (symmetry-reduced cell) size,
  // because ilattice_vector and locate_point_in_user_volume translate by
  // the full cell period. Using gv here would give the wrong Bloch phase
  // when the cell is halved by mirror symmetries. (See issue #132.)
  if (real(kk) * user_volume.num_direction(d) == 0.5 * a) // check b.z. edge exactly
    eikna[d] = -exp(-imag(kk) * ((2 * pi / a) * user_volume.num_direction(d)));
  else {
    const complex<double> I = complex<double>(0.0, 1.0);
    eikna[d] = exp(I * kk * ((2 * pi / a) * user_volume.num_direction(d)));
  }
  coskna[d] = real(eikna[d]);
  sinkna[d] = imag(eikna[d]);
  if (is_real && kk != 0.0) // FIXME: allow real phases (c.f. CONNECT_PHASE)
    meep::abort("Can't use real fields with bloch boundary conditions!\n");
  invalidate(*this, MutationKind::boundary_topology);
  note_connections_invalidated(*this);
  chunk_connections_valid = false; // FIXME: we don't always need to invalidate
}

void fields::use_bloch(const vec &k) {
  // Note that I allow a 1D k input when in cylindrical, since in that case
  // it is unambiguous.
  if (k.dim != gv.dim && !(k.dim == D1 && gv.dim == Dcyl))
    meep::abort("Aaaack, k has wrong dimensions!\n");
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (gv.has_boundary(Low, d) && d != R) use_bloch(d, k.in_direction(d));
  }
}

ivec fields::ilattice_vector(direction d) const {
  switch (user_volume.dim) {
    case D1: return ivec(2 * user_volume.nz());
    case Dcyl: return iveccyl(0, 2 * user_volume.nz()); // Only Z direction here
    case D2:
      switch (d) {
        case X: return ivec(user_volume.nx() * 2, 0);
        case Y: return ivec(0, user_volume.ny() * 2);
        case Z: // fall-thru
        case R: // fall-thru
        case P: // fall-thru
        case NO_DIRECTION: break;
      }
    case D3:
      switch (d) {
        case X: return ivec(user_volume.nx() * 2, 0, 0);
        case Y: return ivec(0, user_volume.ny() * 2, 0);
        case Z: return ivec(0, 0, user_volume.nz() * 2);
        case R: // fall-thru
        case P: // fall-thru
        case NO_DIRECTION: break;
      }
  }
  meep::abort("Aaack in ilattice_vector.\n");
  return ivec(0);
}

vec fields::lattice_vector(direction d) const { return gv[ilattice_vector(d)]; }

void fields::disconnect_chunks() {
  note_connections_invalidated(*this);
  halos->clear();
  chunk_connections_valid = false;
  FOR_FIELD_TYPES(ft) {
    for (int i = 0; i < num_chunks * num_chunks; i++) {
      delete[] comm_blocks[ft][i];
      comm_blocks[ft][i] = 0;
    }
    comms_sequence_for_field[ft].clear();
  }
  comm_sizes.clear();
}

// this should be called by any code that might set chunk_connections_valid = false,
// with the caveat that we need to be careful that we call it on all processes
void fields::sync_chunk_connections() {
  /* make sure all processes agree on chunk_connections_valid to avoid deadlocks
     when we eventually call connect_chunks */
  am_now_working_on(MpiAllTime);
  const bool was_valid = chunk_connections_valid;
  chunk_connections_valid = and_to_all(chunk_connections_valid);
  finished_working();
  /* Another rank invalidated while we did not: mirror that into our own
     connection generation so the shadow stays exact. */
  if (was_valid && !chunk_connections_valid) note_connections_invalidated(*this);
  /* Deliberately NOT note_connection_sync_done() here: the legacy
     `changed_materials` flag stays set until the end of the timestep, so
     clearing the shadow now would desynchronize it. step_once() clears both. */
}

void fields::connect_chunks() {
  // might have invalidated connections in step_db, update_eh, or update_pols:
  assert_local_invalidation_shadow(*this, changed_materials, "connect_chunks");
  if (changed_materials) sync_chunk_connections();

  assert_connections_shadow(*this, chunk_connections_valid, "connect_chunks");
  if (!chunk_connections_valid) {
    am_now_working_on(Connecting);
    disconnect_chunks();
    find_metals();
    connect_the_chunks();
    finished_working();
    note_connections_built(*this);
    chunk_connections_valid = true;
  }
}

bool fields::on_metal_boundary(const ivec &here) {
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (user_volume.has_boundary(High, d) &&
        here.in_direction(d) == user_volume.big_corner().in_direction(d)) {
      if (boundaries[High][d] == Metallic) return true;
    }
    if (boundaries[Low][d] == Magnetic &&
        here.in_direction(d) == user_volume.little_corner().in_direction(d) + 1)
      return true;
    if (boundaries[Low][d] == Metallic &&
        here.in_direction(d) == user_volume.little_corner().in_direction(d))
      return true;
  }
  return false;
}

bool fields::locate_point_in_user_volume(ivec *there, complex<double> *phase) const {
  // Check if a translational symmetry is needed to bring the point in...
  if (!user_volume.owns(*there)) {
    LOOP_OVER_DIRECTIONS(gv.dim, d) {
      if (boundaries[High][d] == Periodic &&
          there->in_direction(d) <= user_volume.little_corner().in_direction(d)) {
        while (there->in_direction(d) <= user_volume.little_corner().in_direction(d)) {
          *there += ilattice_vector(d);
          *phase *= conj(eikna[d]);
        }
      }
      else if (boundaries[High][d] == Periodic &&
               there->in_direction(d) - ilattice_vector(d).in_direction(d) >
                   user_volume.little_corner().in_direction(d)) {
        while (there->in_direction(d) - ilattice_vector(d).in_direction(d) >
               user_volume.little_corner().in_direction(d)) {
          *there -= ilattice_vector(d);
          *phase *= eikna[d];
        }
      }
    }
  }
  return user_volume.owns(*there);
}

void fields::locate_volume_source_in_user_volume(const vec p1, const vec p2, vec newp1[8],
                                                 vec newp2[8], complex<double> kphase[8],
                                                 int &ncopies) const {
  // For periodic boundary conditions,
  // this function locates up to 8 translated copies of the initial grid_volume specified by (p1,p2)
  // First bring center of grid_volume inside
  ncopies = 1;
  newp1[0] = p1;
  newp2[0] = p2;
  kphase[0] = 1;
  vec cen = (newp1[0] + newp2[0]) * 0.5;
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (boundaries[High][d] == Periodic) {
      while (cen.in_direction(d) < gv.boundary_location(Low, d)) {
        newp1[0] += lattice_vector(d);
        newp2[0] += lattice_vector(d);
        kphase[0] *= conj(eikna[d]);
        cen = (newp1[0] + newp2[0]) * 0.5;
      }
      while (cen.in_direction(d) > gv.boundary_location(High, d)) {
        newp1[0] -= lattice_vector(d);
        newp2[0] -= lattice_vector(d);
        kphase[0] *= eikna[d];
        cen = (newp1[0] + newp2[0]) * 0.5;
      }
    }
  }

  // if grid_volume extends outside user_volume in any direction, we need to duplicate already
  // existing copies
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (boundaries[High][d] == Periodic) {
      if (newp1[0].in_direction(d) < gv.boundary_location(Low, d) ||
          newp2[0].in_direction(d) < gv.boundary_location(Low, d)) {
        for (int j = 0; j < ncopies; j++) {
          newp1[ncopies + j] = newp1[j] + lattice_vector(d);
          newp2[ncopies + j] = newp2[j] + lattice_vector(d);
          kphase[ncopies + j] = kphase[j] * conj(eikna[d]);
        }
        ncopies *= 2;
      }
      else if (newp1[0].in_direction(d) > gv.boundary_location(High, d) ||
               newp2[0].in_direction(d) > gv.boundary_location(High, d)) {
        for (int j = 0; j < ncopies; j++) {
          newp1[ncopies + j] = newp1[j] - lattice_vector(d);
          newp2[ncopies + j] = newp2[j] - lattice_vector(d);
          kphase[ncopies + j] = kphase[j] * eikna[d];
        }
        ncopies *= 2;
      }
    }
  }
}

bool fields::locate_component_point(component *c, ivec *there, complex<double> *phase) const {
  // returns true if this point and component exist in the user_volume.  If
  // that is the case, on return *c and *there store the component and
  // location of where the point actually is, and *phase determines holds
  // the phase needed to get the true field.  If the point is not located,
  // *c and *there will hold undefined values.

  // Check if nothing tricky is needed...
  *phase = 1.0;
  if (!locate_point_in_user_volume(there, phase)) return false;
  // Check if a rotation or inversion brings the point in...
  if (user_volume.owns(*there))
    for (int sn = 0; sn < S.multiplicity(); sn++) {
      const ivec here = S.transform(*there, sn);
      if (gv.owns(here)) {
        *there = here;
        *phase *= S.phase_shift(*c, sn);
        *c = direction_component(*c, S.transform(component_direction(*c), sn).d);
        return true;
      }
    }
  return false;
}

void fields::zero_metal(field_type ft, int chunk_idx) {
  const ZeroPlan &z = halos->zeros[ft][chunk_idx];
  for (const SlabRef &sl : z.slabs) {
    realnum *base = halos->arrays.base(sl.array) + sl.base;
    const ptrdiff_t stride = sl.strides[0];
    for (int k = 0; k < sl.counts[0]; ++k)
      base[ptrdiff_t(k) * stride] = 0.0;
  }
  for (const ElementRef &e : z.residue)
    halos->arrays.base(e.array)[e.index] = 0.0;
}

void fields::find_metals() {
  FOR_FIELD_TYPES(ft) { halos->zeros[ft].assign(num_chunks, ZeroPlan()); }

  std::vector<ElementRef> refs;
  std::vector<HaloSegment> order; // unused: every write here is a zero
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      const grid_volume vi = chunks[i]->gv;
      FOR_FIELD_TYPES(ft) {
        refs.clear();
        DOCMP FOR_COMPONENTS(c) {
          if (type(c) == ft && chunks[i]->f[c][cmp]) {
            const ArrayId id =
                halos->arrays.intern({i, int(array_role::field), int(c), cmp, 0},
                                     chunks[i]->f[c][cmp], size_t(vi.ntot()), array_role::field);
            LOOP_OVER_VOL_OWNED(vi, c, n) {
              if (IVEC_LOOP_AT_BOUNDARY) { // todo: just loop over boundaries
                IVEC_LOOP_ILOC(vi, here);
                if (on_metal_boundary(here)) refs.push_back(ElementRef{id, ptrdiff_t(n)});
              }
            }
          }
        }
        ZeroPlan &z = halos->zeros[ft][i];
        coalesce_into_slabs(refs, 1, z.slabs, z.residue, order);
      }
    }
}

bool fields_chunk::needs_W_notowned(component c) {
  for (susceptibility *chiP = s->chiP[type(c)]; chiP; chiP = chiP->next)
    if (chiP->needs_W_notowned(c, f)) return true;
  return false;
}

/* approximate comparisons of phases to 1 or -1.   exp(ix) * exp(-ix) is not exactly 1
   due to roundoff errors, but we want to treat it equal for the purpose of determining
   the boundary conditions, because we have optimized b.c.'s for phase +1 and -1, and
   roundoff errors on the order of 1e-13 are not significant relative to the truncation
   errors in meep anyway.  (the alternative would be to accumulate symmetry phases more
   accurately by adding exponents rather than multiplying). */
static bool phase_isclose(std::complex<double> thephase, double realphase) {
  return fabs(thephase.imag()) < 1e-13 && fabs(thephase.real() - realphase) < 1e-13;
}
static connect_phase connect_phase_from_phase(std::complex<double> thephase) {
  return phase_isclose(thephase, 1.0)    ? CONNECT_COPY
         : phase_isclose(thephase, -1.0) ? CONNECT_NEGATE
                                         : CONNECT_PHASE;
}

void fields::connect_the_chunks() {
  /* For some of the chunks, H==B, and we definitely don't need to
     send B between two such chunks.   We'll still send B when
     the recipient has H != B, since the recipient needs to get B
     from somewhere (although it could get it locally, in principle).
     When the sender has H != B, we'll skip sending B (we'll only send H)
     since we need to get the correct curl H in the E update.  This is
     a bit subtle since the non-owned B may be different from H even
     on an H==B chunk (true?), but since we don't use the non-owned B
     for anything(?) it shouldn't matter. */
  std::vector<int> B_redundant(num_chunks * 2 * 5);
  for (int i = 0; i < num_chunks; ++i)
    FOR_H_AND_B(hc, bc) {
      B_redundant[5 * (num_chunks + i) + bc - Bx] = chunks[i]->f[hc][0] == chunks[i]->f[bc][0];
    }
  am_now_working_on(MpiAllTime);
  and_to_all(B_redundant.data() + 5 * num_chunks, B_redundant.data(), 5 * num_chunks);
  finished_working();

  /* Figure out whether we need the notowned W field (== E/H in
     non-PML regions) in update_pols, e.g. if we have an anisotropic
     susceptibility.  In this case, we have an additional
     communication step where we communicate the notowned W.  Then,
     after updating the polarizations, we communicate the notowned E/H
     ... this does the E/H communication twice between non-PML regions
     and hence is somewhat wasteful, but greatly simplifies the case
     of communicating between a PML region (which has a separate W
     array) and a non-PML region (no separate W). */
  bool needs_W_notowned[NUM_FIELD_COMPONENTS];
  FOR_COMPONENTS(c) { needs_W_notowned[c] = false; }
  FOR_E_AND_H(c) {
    for (int i = 0; i < num_chunks; i++)
      needs_W_notowned[c] = needs_W_notowned[c] || chunks[i]->needs_W_notowned(c);
  }
  am_now_working_on(MpiAllTime);
  FOR_E_AND_H(c) { needs_W_notowned[c] = or_to_all(needs_W_notowned[c]); }
  finished_working();

  comm_sizes.clear();
  const size_t num_reals_per_voxel = is_real ? 1 : 2;
  for (int i = 0; i < num_chunks; i++) {
    // First count the border elements...
    const grid_volume vi = chunks[i]->gv;
    FOR_COMPONENTS(corig) {
      if (have_component(corig)) LOOP_OVER_VOL_NOTOWNED(vi, corig, n) {
          IVEC_LOOP_ILOC(vi, here);
          component c = corig;
          // We're looking at a border element...
          complex<double> thephase;
          if (locate_component_point(&c, &here, &thephase) && !on_metal_boundary(here))
            for (int j = 0; j < num_chunks; j++) {
              const std::pair<int, int> pair_j_to_i{j, i};
              if ((chunks[i]->is_mine() || chunks[j]->is_mine()) && chunks[j]->gv.owns(here) &&
                  !(is_B(corig) && is_B(c) && B_redundant[5 * i + corig - Bx] &&
                    B_redundant[5 * j + c - Bx])) {
                const connect_phase ip = connect_phase_from_phase(thephase);
                comm_sizes[{type(c), ip, pair_j_to_i}] += num_reals_per_voxel;

                if (needs_W_notowned[corig]) {
                  field_type f = is_electric(corig) ? WE_stuff : WH_stuff;
                  comm_sizes[{f, ip, pair_j_to_i}] += num_reals_per_voxel;
                }
                if (is_electric(corig) || is_magnetic(corig)) {
                  field_type f = is_electric(corig) ? PE_stuff : PH_stuff;
                  size_t ni = 0, cni = 0;
                  for (polarization_state *pi = chunks[i]->pol[type(corig)]; pi; pi = pi->next)
                    for (polarization_state *pj = chunks[j]->pol[type(c)]; pj; pj = pj->next)
                      if (*pi->s == *pj->s) {
                        if (pi->data && chunks[i]->is_mine()) {
                          ni += pi->s->num_internal_notowned_needed(corig, pi->data);
                          cni += pi->s->num_cinternal_notowned_needed(corig, pi->data);
                        }
                        else if (pj->data && chunks[j]->is_mine()) {
                          ni += pj->s->num_internal_notowned_needed(c, pj->data);
                          cni += pj->s->num_cinternal_notowned_needed(c, pj->data);
                        }
                      }
                  comm_sizes[{f, ip, pair_j_to_i}] += cni * num_reals_per_voxel;
                  comm_sizes[{f, CONNECT_COPY, pair_j_to_i}] += ni;
                }
              } // if is_mine and owns...
            }   // loop over j chunks
        }       // LOOP_OVER_VOL_NOTOWNED
    }           // FOR_COMPONENTS

    // Allocating comm blocks as we go...
    FOR_FIELD_TYPES(ft) {
      for (int j = 0; j < num_chunks; j++) {
        delete[] comm_blocks[ft][j + i * num_chunks];
        comm_blocks[ft][j + i * num_chunks] = new realnum[comm_size_tot(ft, {j, i})];
      }
    }
  } // loop over i chunks

  // Preallocate the plan element lists.
  for (const std::pair<const comms_key, size_t> &key_and_comm_size : comm_sizes) {
    const chunk_pair &pair_j_to_i = key_and_comm_size.first.pair;
    HaloPlan &p = halos->get_or_create(key_and_comm_size.first);
    if (chunks[pair_j_to_i.first]->is_mine()) p.gather.reserve(key_and_comm_size.second);
    if (chunks[pair_j_to_i.second]->is_mine()) p.scatter.reserve(key_and_comm_size.second);
  }

  // Next start setting up the connections...
  for (int i = 0; i < num_chunks; i++) {
    const grid_volume &vi = chunks[i]->gv;

    FOR_COMPONENTS(corig) {
      if (have_component(corig)) LOOP_OVER_VOL_NOTOWNED(vi, corig, n) {
          IVEC_LOOP_ILOC(vi, here);
          component c = corig;
          // We're looking at a border element...
          std::complex<double> thephase;
          if (locate_component_point(&c, &here, &thephase) && !on_metal_boundary(here)) {
            for (int j = 0; j < num_chunks; j++) {
              const std::pair<int, int> pair_j_to_i{j, i};
              const bool i_is_mine = chunks[i]->is_mine();
              const bool j_is_mine = chunks[j]->is_mine();
              if (!i_is_mine && !j_is_mine) { continue; }

              /* Every connection is recorded twice: once as the legacy host
                 pointer, and once as a relocatable (ArrayId, index) reference.
                 They are emitted from the *same* call so the two cannot drift
                 apart; the byte-identity test in tests/halo_plan.cpp asserts
                 that packing from the references reproduces the legacy comm
                 block exactly. The legacy lists are deleted below this PR's
                 switchover commit. */
              auto push_back_phase = [this, &thephase, &pair_j_to_i](field_type f) {
                halos->get_or_create({f, CONNECT_PHASE, pair_j_to_i})
                    .phase_values.push_back(
                        std::complex<realnum>(thephase.real(), thephase.imag()));
              };
              auto push_back_incoming = [this, &pair_j_to_i](field_type f, connect_phase ip,
                                                             ArrayId id, ptrdiff_t idx) {
                halos->get_or_create({f, ip, pair_j_to_i}).scatter.push_back(ElementRef{id, idx});
              };
              auto push_back_outgoing = [this, &pair_j_to_i](field_type f, connect_phase ip,
                                                             ArrayId id, ptrdiff_t idx) {
                halos->get_or_create({f, ip, pair_j_to_i}).gather.push_back(ElementRef{id, idx});
              };

              /* Interning helpers. The f_w case deliberately falls back to the
                 f key when f_w is absent, so the same storage never gets two
                 identities -- that would silently break run coalescing. */
              auto fld = [this](int ch, component cc, int cmp) {
                return halos->arrays.intern({ch, int(array_role::field), int(cc), cmp, 0},
                                            chunks[ch]->f[cc][cmp], size_t(chunks[ch]->gv.ntot()),
                                            array_role::field);
              };
              auto wfld = [this, &fld](int ch, component cc, int cmp) {
                realnum *w = chunks[ch]->f_w[cc][cmp];
                if (!w) return fld(ch, cc, cmp);
                return halos->arrays.intern({ch, int(array_role::field), int(cc), cmp, 1}, w,
                                            size_t(chunks[ch]->gv.ntot()), array_role::field);
              };
              /* Polarization internals are an opaque void* blob today, so the
                 best available identity is (blob, offset from the blob base).
                 PR 6 makes each built-in susceptibility publish its layout and
                 replaces this with a typed ArrayRef. */
              auto polbase = [this](int ch, field_type ftp, int state_idx, void *data) {
                return halos->arrays.intern(
                    {ch, int(array_role::polarization), int(ftp), -1, state_idx},
                    static_cast<realnum *>(data), 0, array_role::polarization);
              };

              if (chunks[j]->gv.owns(here) &&
                  !(is_B(corig) && is_B(c) && B_redundant[5 * i + corig - Bx] &&
                    B_redundant[5 * j + c - Bx])) {
                const connect_phase ip = connect_phase_from_phase(thephase);
                const ptrdiff_t m = chunks[j]->gv.index(c, here);

                {
                  field_type f = type(c);
                  if (i_is_mine) {
                    if (ip == CONNECT_PHASE) { push_back_phase(f); }
                    DOCMP { push_back_incoming(f, ip, fld(i, corig, cmp), n); }
                  }
                  if (j_is_mine) {
                    DOCMP { push_back_outgoing(f, ip, fld(j, c, cmp), m); }
                  }
                }

                if (needs_W_notowned[corig]) {
                  field_type f = is_electric(corig) ? WE_stuff : WH_stuff;
                  if (i_is_mine) {
                    if (ip == CONNECT_PHASE) { push_back_phase(f); }
                    DOCMP { push_back_incoming(f, ip, wfld(i, corig, cmp), n); }
                  }
                  if (j_is_mine) {
                    DOCMP { push_back_outgoing(f, ip, wfld(j, c, cmp), m); }
                  }
                }

                if (is_electric(corig) || is_magnetic(corig)) {
                  field_type f = is_electric(corig) ? PE_stuff : PH_stuff;
                  int pi_idx = -1;
                  for (polarization_state *pi = chunks[i]->pol[type(corig)]; pi; pi = pi->next) {
                    ++pi_idx;
                    int pj_idx = -1;
                    for (polarization_state *pj = chunks[j]->pol[type(c)]; pj; pj = pj->next) {
                      ++pj_idx;
                      if (*pi->s == *pj->s) {
                        polarization_state *po = NULL;
                        if (pi->data && chunks[i]->is_mine())
                          po = pi;
                        else if (pj->data && chunks[j]->is_mine())
                          po = pj;
                        if (po) {
                          const connect_phase iip = CONNECT_COPY;
                          const ArrayId in_pol =
                              i_is_mine ? polbase(i, type(corig), pi_idx, pi->data)
                                        : invalid_array();
                          const ArrayId out_pol =
                              j_is_mine ? polbase(j, type(c), pj_idx, pj->data) : invalid_array();
                          realnum *in_base = static_cast<realnum *>(pi->data);
                          realnum *out_base = static_cast<realnum *>(pj->data);
                          const size_t ni = po->s->num_internal_notowned_needed(corig, po->data);
                          for (size_t k = 0; k < ni; ++k) {
                            if (i_is_mine) {
                              push_back_incoming(
                                  f, iip, in_pol,
                                  po->s->internal_notowned_ptr(k, corig, n, pi->data) - in_base);
                            }
                            if (j_is_mine) {
                              push_back_outgoing(
                                  f, iip, out_pol,
                                  po->s->internal_notowned_ptr(k, c, m, pj->data) - out_base);
                            }
                          }
                          const size_t cni = po->s->num_cinternal_notowned_needed(corig, po->data);
                          for (size_t k = 0; k < cni; ++k) {
                            if (i_is_mine) {
                              if (ip == CONNECT_PHASE) { push_back_phase(f); }

                              DOCMP {
                                push_back_incoming(
                                    f, ip, in_pol,
                                    po->s->cinternal_notowned_ptr(k, corig, cmp, n, pi->data) -
                                        in_base);
                              }
                            }
                            if (j_is_mine) {
                              DOCMP {
                                push_back_outgoing(
                                    f, ip, out_pol,
                                    po->s->cinternal_notowned_ptr(k, c, cmp, m, pj->data) -
                                        out_base);
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                } // is_electric(corig)
              }   // if is_mine and owns...
            }     // loop over j chunks
          }       // here in user_volume
        }         // LOOP_OVER_VOL_NOTOWNED
    }             // FOR_COMPONENTS
  }               // loop over i chunks

  FOR_FIELD_TYPES(f) {

    // Calculate the sequence of sends and receives in advance.
    // Initiate receive operations as early as possible.
    std::unique_ptr<comms_manager> manager = create_comms_manager();
    std::vector<comms_operation> operations;
    std::vector<int> tagto(count_processors());

    for (int j = 0; j < num_chunks; j++) {
      for (int i = 0; i < num_chunks; i++) {
        const chunk_pair pair{j, i};
        const size_t comm_size = comm_size_tot(f, pair);
        if (!comm_size) continue;
        if (comm_size > manager->max_transfer_size()) {
          // MPI uses int for size to send/recv
          meep::abort("communications size too big for the current implementation");
        }
        const int pair_idx = j + i * num_chunks;

        if (chunks[j]->is_mine()) {
          operations.push_back(comms_operation{/*my_chunk_idx=*/j,
                                               /*other_chunk_idx=*/i,
                                               /*other_proc_id=*/chunks[i]->n_proc(),
                                               /*pair_idx=*/pair_idx,
                                               /*transfer_size=*/comm_size,
                                               /*comm_direction=*/Outgoing,
                                               /*tag=*/tagto[chunks[i]->n_proc()]++});
        }
        if (chunks[i]->is_mine()) {
          operations.push_back(comms_operation{/*my_chunk_idx=*/i,
                                               /*other_chunk_idx=*/j,
                                               /*other_proc_id=*/chunks[j]->n_proc(),
                                               /*pair_idx=*/pair_idx,
                                               /*transfer_size=*/comm_size,
                                               /*comm_direction=*/Incoming,
                                               /*tag=*/tagto[chunks[j]->n_proc()]++});
        }
      }
    }

    comms_sequence_for_field[f] = optimize_comms_operations(operations);
  }

  finalize_halo_plans();
}

/* Fill in everything about a HaloPlan that is not known until every connection
   has been pushed: its position in the communication block, its peer, and the
   slab decomposition of its gather/scatter lists. */
void fields::finalize_halo_plans() {
  const int interleave = is_real ? 1 : 2;

  FOR_FIELD_TYPES(ft) {
    for (int i = 0; i < num_chunks; i++)
      for (int j = 0; j < num_chunks; j++) {
        const chunk_pair pair{j, i};
        size_t offset = 0;
        /* The block layout contract: phases in all_connect_phases declaration
           order, which is exactly what step_boundaries packs and what
           process_incoming_chunk_data unpacks. sequence_index records it so
           the two can be checked against each other instead of merely
           agreeing by convention. */
        for (connect_phase ip : all_connect_phases) {
          const comms_key key{ft, ip, pair};
          const size_t sz = get_comm_size(key);
          HaloPlan *p = halos->find(key);
          if (!sz) {
            if (p) {
              p->block_offset = offset;
              p->block_elements = 0;
            }
            continue;
          }
          if (!p) p = &halos->get_or_create(key);
          p->block_offset = offset;
          p->block_elements = sz;
          p->peer_rank = chunks[j]->is_mine() ? chunks[i]->n_proc() : chunks[j]->n_proc();
          p->same_rank = chunks[i]->n_proc() == chunks[j]->n_proc();
          offset += sz;
        }
      }
  }

  /* Coalesce. The gather and scatter sides are folded independently: a run
     that is contiguous on the sending chunk need not be contiguous on the
     receiving one, and vice versa. */
  for (HaloPlan &p : halos->plans) {
    /* Polarization internal streams are pushed one real at a time
       (num_internal_notowned_needed), not as complex pairs, so they must not
       be de-interleaved as if they were. Fold them as a single stream. */
    const int iv = (p.ft == PE_stuff || p.ft == PH_stuff) ? 1 : interleave;
    std::vector<SlabRef> slabs;
    std::vector<ElementRef> residue;
    std::vector<HaloSegment> order;

    coalesce_into_slabs(p.gather, iv, slabs, residue, order);
    p.gather_slabs.swap(slabs);
    p.gather.swap(residue);
    p.gather_order.swap(order);

    coalesce_into_slabs(p.scatter, iv, slabs, residue, order);
    p.scatter_slabs.swap(slabs);
    p.scatter.swap(residue);
    p.scatter_order.swap(order);
  }
}

} // namespace meep
