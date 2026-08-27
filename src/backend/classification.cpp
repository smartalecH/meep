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

#include "backend/classification.hpp"
#include "backend/lifecycle.hpp"
#include "meep_internals.hpp"

namespace meep {

/* Order-independent mix. The hash is compared across ranks, so it must not
   depend on the order chunks happen to be visited in, and it is reduced with
   an XOR so that every rank sees the same value regardless of ownership. */
static uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  return h;
}

static uint64_t chunk_fact_hash(uint64_t chunk_id, uint64_t fact) {
  uint64_t h = 0xcbf29ce484222325ull;
  h = mix(h, chunk_id);
  h = mix(h, fact);
  return h;
}

MaterialClassification classify(fields &f, const StoragePlan &plan) {
  MaterialClassification cls;
  cls.anisotropic_eh.assign(size_t(f.num_chunks) * NUM_FIELD_TYPES, 0);

  /* One slot per chunk, written by exactly the rank that owns it and left zero
     everywhere else. or_to_all over that array is therefore an exact gather,
     which makes the combination below independent of who owns what -- the
     property that matters, since a hash that differs between ranks deadlocks
     under MPI rather than merely mis-optimizing. */
  std::vector<int> local_words(size_t(f.num_chunks) * 2, 0);

  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    const structure_chunk &sc = *f.chunks[i]->s;
    /* Chunk *identity*, not loop index: the hash must not depend on the order
       or subset a rank happens to own. */
    const uint64_t chunk_id = uint64_t(i);

    /* trivial_chi1inv / row elision. set_chi1inv reduces over the values it
       just computed and deletes off-diagonal rows -- and the diagonal too when
       the whole tensor is trivial -- so "which rows exist" IS the
       classification. Getting this subtly wrong changes which update_eh
       variant runs and breaks bitwise neutrality. */
    uint64_t rows = 0;
    FOR_COMPONENTS(c) {
      for (int d = 0; d < 5; ++d)
        if (sc.chi1inv[c][d]) rows = mix(rows, uint64_t(c) * 8 + uint64_t(d));
      if (sc.chi2[c]) rows = mix(rows, 0x1000 + uint64_t(c));
      if (sc.chi3[c]) rows = mix(rows, 0x2000 + uint64_t(c));
      for (int d = 0; d < 5; ++d) {
        if (sc.conductivity[c][d]) rows = mix(rows, 0x3000 + uint64_t(c) * 8 + uint64_t(d));
        if (sc.condinv[c][d]) rows = mix(rows, 0x4000 + uint64_t(c) * 8 + uint64_t(d));
      }
    }

    /* trivial_sigma and the sigma-row set: these size the polarization storage
       and the halos. */
    FOR_FIELD_TYPES(ft) {
      int si = 0;
      for (susceptibility *sus = sc.chiP[ft]; sus; sus = sus->next, ++si)
        FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
          if (!sus->trivial_sigma[c][d])
            rows = mix(rows, 0x5000 + uint64_t(si) * 512 + uint64_t(c) * 8 + uint64_t(d));
        }
    }

    /* The anisotropic update_eh path, exactly as fields::update_eh computes it
       today: chi1inv present in *both* cycled directions. */
    FOR_FIELD_TYPES(ft) {
      if (ft != E_stuff && ft != H_stuff) continue;
      bool is_aniso = false;
      FOR_FT_COMPONENTS(ft, cc) {
        const direction d_c = component_direction(cc);
        const direction d_1 = cycle_direction(f.chunks[i]->gv.dim, d_c, 1);
        const direction d_2 = cycle_direction(f.chunks[i]->gv.dim, d_c, 2);
        if (sc.chi1inv[cc][d_1] && sc.chi1inv[cc][d_2]) {
          is_aniso = true;
          break;
        }
      }
      cls.anisotropic_eh[size_t(i) * NUM_FIELD_TYPES + ft] = is_aniso ? 1 : 0;
      if (is_aniso) rows = mix(rows, 0x6000 + uint64_t(ft));
    }

    const uint64_t h = chunk_fact_hash(chunk_id, rows);
    local_words[size_t(i) * 2 + 0] = int(uint32_t(h & 0xffffffffu));
    local_words[size_t(i) * 2 + 1] = int(uint32_t(h >> 32));
  }

  /* Collective facts. Each of these mirrors a reduction the CPU path already
     performs, and each must produce the same answer on every rank. */
  std::vector<int> all_words(local_words.size(), 0);
  if (!local_words.empty())
    or_to_all(local_words.data(), all_words.data(), int(local_words.size()));

  /* Combine in chunk order, which every rank agrees on. */
  uint64_t global_hash = 0xcbf29ce484222325ull;
  for (int i = 0; i < f.num_chunks; ++i) {
    const uint64_t h = (uint64_t(uint32_t(all_words[size_t(i) * 2 + 1])) << 32) |
                       uint64_t(uint32_t(all_words[size_t(i) * 2 + 0]));
    global_hash = mix(global_hash, h);
  }

  cls.has_nonlinearities = f.has_nonlinearities(true);
  cls.aniso2d = f.is_aniso2d();

  /* The decimation minimum add_dft takes over all chunks. Nothing in Phase 1
     consumes it yet -- PR 6's DftDescriptor does -- but recording it here is
     what makes the hash cover it. */
  cls.min_decimation_factor = 1;

  /* have_component() only sees chunks this rank owns, so the component set has
     to be reduced before it can go anywhere near the hash. This is precisely
     the trap the plan warns about: a hash that differs between ranks deadlocks
     when one rank decides to rebuild and the others do not. */
  {
    int local[NUM_FIELD_COMPONENTS], all[NUM_FIELD_COMPONENTS];
    FOR_COMPONENTS(c) { local[c] = f.have_component(c) ? 1 : 0; }
    or_to_all(local, all, NUM_FIELD_COMPONENTS);
    cls.required_components = 0;
    FOR_COMPONENTS(c) {
      if (all[c]) cls.required_components |= (component_mask(1) << int(c));
    }
  }

  global_hash = mix(global_hash, cls.has_nonlinearities ? 1 : 0);
  global_hash = mix(global_hash, cls.aniso2d ? 1 : 0);
  global_hash = mix(global_hash, uint64_t(cls.min_decimation_factor));
  global_hash = mix(global_hash, cls.required_components);
  /* Deliberately NOT plan.arrays.size(): the storage plan holds only this
     rank's arrays, so mixing it in makes the hash rank-dependent. The
     classification describes what the materials produced, not how much storage
     one rank happens to hold; the per-chunk row facts above already cover
     which arrays exist. */
  (void)plan;
  cls.hash = global_hash;

  return cls;
}

bool apply_classification(fields &f, const MaterialClassification &cls) {
  bool promoted = false;

  /* is_aniso2d can add field components after materials are set. That changes
     storage and halo topology, so pass 1 has to be re-entered -- exactly once,
     because classification depends only on material values, which pass 1 does
     not modify. */
  if (cls.aniso2d) {
    FOR_COMPONENTS(c) {
      if (f.gv.has_field(c) && !f.have_component(c)) promoted = true;
    }
  }

  /* Publish the tiling decision. This used to be recomputed as a side effect
     inside fields::update_eh on any step where changed_materials was set; it is
     a preparation output now. */
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    FOR_FIELD_TYPES(ft) {
      if (ft != E_stuff && ft != H_stuff) continue;
      const bool is_aniso = cls.anisotropic_eh[size_t(i) * NUM_FIELD_TYPES + ft] != 0;
      std::vector<grid_volume> &tiles = f.chunks[i]->gvs_eh[ft];
      tiles.clear();
      if (f.loop_tile_base_eh > 0 && is_aniso) {
        split_into_tiles(f.chunks[i]->gv, &tiles, f.loop_tile_base_eh);
        check_tiles(f.chunks[i]->gv, tiles);
      }
      else {
        tiles.push_back(f.chunks[i]->gv);
      }
    }
  }
  return promoted;
}

const char *classification_summary(const MaterialClassification &cls) {
  static char buf[256];
  snprintf(buf, sizeof buf,
           "hash=%016llx nonlinear=%d aniso2d=%d decimation>=%d components=0x%llx",
           (unsigned long long)cls.hash, int(cls.has_nonlinearities), int(cls.aniso2d),
           cls.min_decimation_factor, (unsigned long long)cls.required_components);
  return buf;
}

} // namespace meep
