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

/* Boundary exchange as relocatable data.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 *
 * connect_the_chunks() used to emit one realnum* per transferred real into
 * fields_chunk::connections_in / connections_out. This replaces those pointer
 * lists with (ArrayId, index) references, coalesced into slabs wherever the
 * connection is a contiguous or constant-stride run.
 */

#ifndef MEEP_BACKEND_HALO_PLAN_HPP
#define MEEP_BACKEND_HALO_PLAN_HPP

#include <complex>
#include <string>
#include <unordered_map>
#include <vector>

#include "meep.hpp"
#include "backend/array_ref.hpp"

namespace meep {

class CpuArrayCatalog;

/* --- CPU array table -------------------------------------------------------
   The single place that knows what address an ArrayId currently resolves to.
   Plans hold IDs; this holds pointers. Rebuilt whenever connectivity is, so a
   reallocation cannot leave a plan pointing at freed storage.

   PR 3 replaces this with the real CpuArrayCatalog, which also covers material,
   DFT and source arrays; this is the subset the halo plans need. */

struct HaloArrayKey {
  int chunk;
  int role;       // array_role, as int so the key stays trivially hashable
  int component_; // meep::component, or -1
  int cmp;        // 0/1, or -1
  int aux;        // 0 = f, 1 = f_w; polarization: index of the state in the chunk

  bool operator==(const HaloArrayKey &o) const {
    return chunk == o.chunk && role == o.role && component_ == o.component_ && cmp == o.cmp &&
           aux == o.aux;
  }
};

struct HaloArrayKeyHash {
  size_t operator()(const HaloArrayKey &k) const {
    size_t h = size_t(k.chunk) * 1000003u;
    h = h * 31 + size_t(k.role);
    h = h * 31 + size_t(k.component_ + 2);
    h = h * 31 + size_t(k.cmp + 2);
    h = h * 31 + size_t(k.aux + 2);
    return h;
  }
};

class HaloArrayTable {
public:
  void clear() {
    specs_.clear();
    bases_.clear();
    index_.clear();
  }

  /* Intern a base pointer under a descriptive key. Idempotent: the same key
     always yields the same ArrayId within one connectivity generation. */
  ArrayId intern(const HaloArrayKey &key, realnum *base, size_t elements, array_role role);

  realnum *base(ArrayId id) const { return bases_[id.value]; }
  const ArraySpec &spec(ArrayId id) const { return specs_[id.value]; }
  size_t size() const { return specs_.size(); }

private:
  std::vector<ArraySpec> specs_;
  std::vector<realnum *> bases_;
  std::unordered_map<HaloArrayKey, uint32_t, HaloArrayKeyHash> index_;
};

/* --- The plan --------------------------------------------------------------
   One HaloPlan per (field_type, connect_phase, chunk_pair) -- the same key the
   legacy comms_key used. */

/* One step of the block order: either a group of `nslabs` slabs, each of
   length `count`, walked round-robin (which is how an interleaved complex run
   was originally pushed), or `residue` consecutive entries from the residue
   list. */
struct HaloSegment {
  uint32_t first_slab;
  uint32_t nslabs;
  uint32_t count;
  uint32_t residue;
};

struct HaloPlan {
  field_type ft;
  chunk_pair chunks;
  connect_phase phase;
  int peer_rank;
  int tag;
  bool same_rank; // populated now; exchange_local lowering is deferred

  /* Position within the (field_type, chunk_pair) communication block. The send
     side used to iterate all_connect_phases in declaration order while the
     receive side unpacked PHASE, then NEGATE, then COPY; nothing enforced that
     those agreed, and a divergence corrupts boundaries silently rather than
     failing. sequence_index makes the contract explicit and the byte-identity
     test asserts it. */
  uint32_t sequence_index;
  size_t block_offset;   // where this plan's reals start in the comm block
  size_t block_elements; // how many reals it contributes

  std::vector<SlabRef> gather_slabs;
  std::vector<SlabRef> scatter_slabs;

  /* Residue: symmetry images, periodic wraps, polarization internals -- the
     entries the coalescer could not fold into a slab. */
  std::vector<ElementRef> gather;
  std::vector<ElementRef> scatter;

  /* How to walk the two spans above to reproduce the communication block
     byte-for-byte. Slabs and residue interleave in general, so "all slabs then
     all residue" is not the block order and cannot be assumed. */
  std::vector<HaloSegment> gather_order;
  std::vector<HaloSegment> scatter_order;

  std::vector<std::complex<realnum> > phase_values; // populated for CONNECT_PHASE
};

/* Flattened element order for one side of a plan: exactly the order the reals
   appear in the communication block. */
void expand_gather(const HaloPlan &p, std::vector<ElementRef> &out);
void expand_scatter(const HaloPlan &p, std::vector<ElementRef> &out);

/* Fold constant-stride runs of ElementRefs on one array into SlabRefs.

   `interleave` is the number of interleaved sub-streams: complex fields push
   the real and imaginary parts of a voxel back to back into *different* arrays
   (f[c][0] and f[c][1]), so a naive stride detector sees every run broken at
   length one. A group of `interleave` slabs covering the same voxel range
   jointly spans a contiguous block of the input, which is what keeps the
   expansion order exact.

   Only 1-D runs are produced. Merging parallel rows into 2-D slabs would cut
   metadata further but complicates the ordering contract, and the acceptance
   criterion is stated in elements, not descriptors -- a 3D chunk face already
   folds to a stack of full-length rows. */
void coalesce_into_slabs(const std::vector<ElementRef> &in, int interleave,
                         std::vector<SlabRef> &slabs, std::vector<ElementRef> &residue,
                         std::vector<HaloSegment> &order);

struct CoalesceStats {
  size_t total_elements;
  size_t slab_elements;
  size_t residue_elements;
  size_t slab_count;
  double ratio() const {
    return total_elements ? double(slab_elements) / double(total_elements) : 1.0;
  }
};

CoalesceStats coalesce_stats(const std::vector<HaloPlan> &plans);

/* Translate a CPU halo plan's private array IDs into the canonical storage
   catalog namespace. The source plan remains unchanged and continues to serve
   the legacy CPU executor. */
bool remap_halo_plan(const HaloPlan &source, const HaloArrayTable &source_arrays,
                     const CpuArrayCatalog &catalog, int field_interleave, HaloPlan &destination,
                     std::string &why);

/* Metal-zero lists get the same treatment: they were one realnum* per zeroed
   point in fields_chunk::zeroes. Order is irrelevant here since every write is
   a zero, so no HaloSegment list is needed. */
struct ZeroPlan {
  std::vector<SlabRef> slabs;
  std::vector<ElementRef> residue;
};

bool remap_zero_plan(const ZeroPlan &source, const HaloArrayTable &source_arrays,
                     const CpuArrayCatalog &catalog, ZeroPlan &destination, std::string &why);

/* Everything the boundary exchange needs that used to be host addresses.
   Owned by `fields` through an opaque pointer so that none of these types
   reach meep.hpp. Rebuilt wholesale by connect_the_chunks(). */
class halo_plan_set {
public:
  HaloArrayTable arrays;
  std::vector<HaloPlan> plans;
  std::unordered_map<comms_key, uint32_t, comms_key_hash_fn> index;
  std::vector<ZeroPlan> zeros[NUM_FIELD_TYPES]; // indexed by chunk

  void clear() {
    arrays.clear();
    plans.clear();
    index.clear();
    FOR_FIELD_TYPES(ft) { zeros[ft].clear(); }
  }

  HaloPlan &get_or_create(const comms_key &k) {
    auto it = index.find(k);
    if (it != index.end()) return plans[it->second];
    index[k] = uint32_t(plans.size());
    plans.emplace_back();
    HaloPlan &p = plans.back();
    p.ft = k.ft;
    p.phase = k.phase;
    p.chunks = k.pair;
    p.peer_rank = -1;
    p.tag = 0;
    p.same_rank = false;
    p.sequence_index = uint32_t(k.phase);
    p.block_offset = 0;
    p.block_elements = 0;
    return p;
  }

  HaloPlan *find(const comms_key &k) {
    auto it = index.find(k);
    return it == index.end() ? nullptr : &plans[it->second];
  }
  const HaloPlan *find(const comms_key &k) const {
    auto it = index.find(k);
    return it == index.end() ? nullptr : &plans[it->second];
  }
};

} // namespace meep

#endif // MEEP_BACKEND_HALO_PLAN_HPP
