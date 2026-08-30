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
#include <functional>
#include <limits>
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
  int susceptibility_id;
  int internal_index;
  ptrdiff_t point_index;
  bool complex_internal;

  HaloArrayKey(int chunk_, int role_, int component__, int cmp_, int aux_,
               int susceptibility_id_ = -1, int internal_index_ = -1,
               ptrdiff_t point_index_ = -1, bool complex_internal_ = false)
      : chunk(chunk_), role(role_), component_(component__), cmp(cmp_), aux(aux_),
        susceptibility_id(susceptibility_id_), internal_index(internal_index_),
        point_index(point_index_), complex_internal(complex_internal_) {}

  bool operator==(const HaloArrayKey &o) const {
    return chunk == o.chunk && role == o.role && component_ == o.component_ && cmp == o.cmp &&
           aux == o.aux && susceptibility_id == o.susceptibility_id &&
           internal_index == o.internal_index && point_index == o.point_index &&
           complex_internal == o.complex_internal;
  }
};

struct HaloArrayKeyHash {
  size_t operator()(const HaloArrayKey &k) const {
    size_t h = std::hash<int>()(k.chunk);
    h = h * 31 + std::hash<int>()(k.role);
    h = h * 31 + std::hash<int>()(k.component_);
    h = h * 31 + std::hash<int>()(k.cmp);
    h = h * 31 + std::hash<int>()(k.aux);
    h = h * 31 + std::hash<int>()(k.susceptibility_id);
    h = h * 31 + std::hash<int>()(k.internal_index);
    h = h * 31 + std::hash<ptrdiff_t>()(k.point_index);
    return h * 31 + std::hash<bool>()(k.complex_internal);
  }
};

class HaloArrayTable {
public:
  void clear() {
    specs_.clear();
    bases_.clear();
    keys_.clear();
    index_.clear();
  }

  /* Intern a base pointer under a descriptive key. Idempotent: the same key
     always yields the same ArrayId within one connectivity generation. */
  ArrayId intern(const HaloArrayKey &key, realnum *base, size_t elements, array_role role);

  bool contains(ArrayId id) const { return id.value < bases_.size(); }
  realnum *base(ArrayId id) const {
    if (!contains(id)) meep::abort("halo array id is out of range");
    return bases_[id.value];
  }
  const ArraySpec &spec(ArrayId id) const {
    if (!contains(id)) meep::abort("halo array spec id is out of range");
    return specs_[id.value];
  }
  const HaloArrayKey &key(ArrayId id) const {
    if (!contains(id)) meep::abort("halo array key id is out of range");
    return keys_[id.value];
  }
  size_t size() const { return specs_.size(); }

private:
  std::vector<ArraySpec> specs_;
  std::vector<realnum *> bases_;
  std::vector<HaloArrayKey> keys_;
  std::unordered_map<HaloArrayKey, uint32_t, HaloArrayKeyHash> index_;
};

/* Opaque susceptibility state deliberately has no canonical device ArrayId.
   A custom susceptibility may return halo pointers into storage whose layout
   is known only to its virtual methods.  Keep the exact live host addresses in
   a separate table; source-local ArrayIds may coexist only as an address
   mirror for later all-or-nothing catalog remapping.  HostHaloId is meaningful
   only for the current connectivity generation. */
struct HostHaloId {
  uint64_t generation;
  uint32_t value;
  bool operator==(HostHaloId o) const {
    return generation == o.generation && value == o.value;
  }
  bool operator!=(HostHaloId o) const { return !(*this == o); }
};

inline HostHaloId invalid_host_halo() {
  return HostHaloId{0, std::numeric_limits<uint32_t>::max()};
}
inline bool valid(HostHaloId id) { return id != invalid_host_halo(); }

struct HostHaloKey {
  int chunk;
  int ft;
  int state_index;
  int susceptibility_id;
  int component_;
  int cmp;
  int internal_index;
  ptrdiff_t point_index;
  bool complex_internal;

  bool operator==(const HostHaloKey &o) const {
    return chunk == o.chunk && ft == o.ft && state_index == o.state_index &&
           susceptibility_id == o.susceptibility_id && component_ == o.component_ &&
           cmp == o.cmp && internal_index == o.internal_index && point_index == o.point_index &&
           complex_internal == o.complex_internal;
  }
};

struct HostHaloKeyHash {
  size_t operator()(const HostHaloKey &k) const {
    size_t h = std::hash<int>()(k.chunk);
    h = h * 31 + std::hash<int>()(k.ft);
    h = h * 31 + std::hash<int>()(k.state_index);
    h = h * 31 + std::hash<int>()(k.susceptibility_id);
    h = h * 31 + std::hash<int>()(k.component_);
    h = h * 31 + std::hash<int>()(k.cmp);
    h = h * 31 + std::hash<int>()(k.internal_index);
    h = h * 31 + std::hash<ptrdiff_t>()(k.point_index);
    return h * 31 + std::hash<bool>()(k.complex_internal);
  }
};

class HostHaloArrayTable {
public:
  HostHaloArrayTable() : generation_(1) {}
  void clear() {
    addresses_.clear();
    keys_.clear();
    index_.clear();
    if (++generation_ == 0) meep::abort("opaque host halo generation overflow");
  }

  HostHaloId intern(const HostHaloKey &key, realnum *address);
  realnum *address(HostHaloId id) const;
  const HostHaloKey &key(HostHaloId id) const;
  bool contains(HostHaloId id) const {
    return id.generation == generation_ && id.value < addresses_.size();
  }
  size_t size() const { return addresses_.size(); }
  uint64_t generation() const { return generation_; }

private:
  uint64_t generation_;
  std::vector<realnum *> addresses_;
  std::vector<HostHaloKey> keys_;
  std::unordered_map<HostHaloKey, uint32_t, HostHaloKeyHash> index_;
};

struct HostElementRef {
  HostHaloId id;
};

enum class HaloStorageDisposition { canonical, host_owned };

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
  HaloStorageDisposition storage;

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

  /* Opaque polarization internals have no canonical device identity.  PE/PH
     plans retain their exact live ordering here instead of manufacturing a
     zero-sized ArrayId.  PR3 may remap layout-publishing rows to canonical
     arrays, while these entries remain explicitly host-owned. */
  std::vector<HostElementRef> host_gather;
  std::vector<HostElementRef> host_scatter;

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
void expand_gather_addresses(const HaloPlan &p, const HaloArrayTable &arrays,
                             const HostHaloArrayTable &host_arrays,
                             std::vector<realnum *> &out);
void expand_scatter_addresses(const HaloPlan &p, const HaloArrayTable &arrays,
                              const HostHaloArrayTable &host_arrays,
                              std::vector<realnum *> &out);

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

/* Validate endpoint-specific virtual halo counts before converting them to an
   unsigned communication size.  An absent endpoint contributes no opinion;
   two live endpoints must agree exactly. */
bool resolve_host_halo_count(bool have_in, int incoming, bool have_out, int outgoing,
                             size_t &count, std::string &why);
bool checked_add_halo_elements(size_t current, size_t added, size_t &result, std::string &why);
bool checked_multiply_halo_elements(size_t count, size_t lanes, size_t &result,
                                    std::string &why);

/* All ranks call this in the same deterministic connection order.  The
   sender and receiver roots publish the independently computed block size;
   comparing the two before allocation/exchange prevents an opaque virtual
   implementation from giving MPI peers different message lengths. */
bool reconcile_host_halo_comm_size(int sender_rank, size_t sender_local,
                                   int receiver_rank, size_t receiver_local,
                                   std::string &why);

/* Translate a CPU halo plan's private array IDs into the canonical storage
   catalog namespace. PE/PH is all-or-nothing: if any element is absent from
   the catalog, the destination retains the complete ordered host mirror. The
   source plan remains unchanged and continues to serve the legacy CPU
   executor. */
bool remap_halo_plan(const HaloPlan &source, const HaloArrayTable &source_arrays,
                     const HostHaloArrayTable &source_host_arrays,
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
  HostHaloArrayTable host_arrays;
  std::vector<HaloPlan> plans;
  std::unordered_map<comms_key, uint32_t, comms_key_hash_fn> index;
  std::vector<ZeroPlan> zeros[NUM_FIELD_TYPES]; // indexed by chunk

  void clear() {
    arrays.clear();
    host_arrays.clear();
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
    p.storage = (k.ft == PE_stuff || k.ft == PH_stuff) ? HaloStorageDisposition::host_owned
                                                       : HaloStorageDisposition::canonical;
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
