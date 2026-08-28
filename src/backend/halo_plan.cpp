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

#include "backend/halo_plan.hpp"
#include "backend/storage_plan.hpp"
#include "meep/mympi.hpp"

#include <limits>

namespace meep {

ArrayId HaloArrayTable::intern(const HaloArrayKey &key, realnum *base, size_t elements,
                               array_role role) {
  auto it = index_.find(key);
  if (it != index_.end()) {
    const ArrayId id{it->second};
    if (bases_[id.value] != base || specs_[id.value].elements != elements ||
        specs_[id.value].role != role)
      meep::abort("halo array identity resolved to inconsistent storage");
    return id;
  }

  if (specs_.size() >= std::numeric_limits<uint32_t>::max())
    meep::abort("too many halo arrays");
  const uint32_t id = uint32_t(specs_.size());
  ArraySpec spec;
  spec.id = ArrayId{id};
  spec.role = role;
  spec.element_type = ElementType::realnum_value;
  spec.storage = sizeof(realnum) == sizeof(float) ? Precision::f32 : Precision::f64;
  spec.elements = elements;
  spec.alignment = alignof(realnum);
  spec.alias_of = invalid_array();
  spec.classification_provisional = false;
  specs_.push_back(spec);
  bases_.push_back(base);
  index_[key] = id;
  return ArrayId{id};
}

HostHaloId HostHaloArrayTable::intern(const HostHaloKey &key, realnum *address) {
  auto it = index_.find(key);
  if (it != index_.end()) {
    const HostHaloId id{generation_, it->second};
    if (addresses_[id.value] != address)
      meep::abort("opaque halo identity resolved to two host addresses");
    return id;
  }
  if (!address) meep::abort("opaque halo identity resolved to a null host address");
  if (addresses_.size() >= std::numeric_limits<uint32_t>::max())
    meep::abort("too many opaque host halo addresses");
  const uint32_t id = uint32_t(addresses_.size());
  addresses_.push_back(address);
  keys_.push_back(key);
  index_[key] = id;
  return HostHaloId{generation_, id};
}

realnum *HostHaloArrayTable::address(HostHaloId id) const {
  if (!contains(id)) meep::abort("opaque host halo id is stale or out of range");
  return addresses_[id.value];
}

const HostHaloKey &HostHaloArrayTable::key(HostHaloId id) const {
  if (!contains(id)) meep::abort("opaque host halo key id is stale or out of range");
  return keys_[id.value];
}

/* --- Slab coalescing ------------------------------------------------------ */

namespace {

/* Longest run starting at `i` such that every one of the `iv` interleaved
   sub-streams is a constant-index-stride run on a single array. Returns the
   run length in *voxels* (so it covers run * iv input entries), and fills
   `arrays` / `strides` for each sub-stream. */
size_t measure_run(const std::vector<ElementRef> &in, size_t i, size_t iv,
                   std::vector<ArrayId> &arrays, std::vector<ptrdiff_t> &strides) {
  const size_t n = in.size();
  if (i + 2 * iv > n) return 0; // need at least two voxels to have a stride

  arrays.assign(iv, invalid_array());
  strides.assign(iv, 0);
  for (size_t s = 0; s < iv; ++s) {
    if (in[i + s].array != in[i + iv + s].array) return 0;
    arrays[s] = in[i + s].array;
    strides[s] = in[i + iv + s].index - in[i + s].index;
  }

  size_t run = 2;
  while (i + (run + 1) * iv <= n) {
    bool ok = true;
    for (size_t s = 0; s < iv && ok; ++s) {
      const ElementRef &e = in[i + run * iv + s];
      ok = e.array == arrays[s] && e.index == in[i + s].index + ptrdiff_t(run) * strides[s];
    }
    if (!ok) break;
    ++run;
  }
  return run;
}

} // namespace

void coalesce_into_slabs(const std::vector<ElementRef> &in, int interleave_in,
                         std::vector<SlabRef> &slabs, std::vector<ElementRef> &residue,
                         std::vector<HaloSegment> &order) {
  slabs.clear();
  residue.clear();
  order.clear();
  if (in.empty()) return;

  const size_t iv = interleave_in < 1 ? 1 : size_t(interleave_in);
  /* If the list is not a whole number of interleaved voxels the stream is not
     what we think it is; fall back to pure residue rather than guess. */
  const bool aligned = (in.size() % iv) == 0;

  std::vector<ArrayId> arrays;
  std::vector<ptrdiff_t> strides;
  size_t i = 0;
  size_t pending_residue = 0;

  auto flush_residue = [&]() {
    if (!pending_residue) return;
    HaloSegment seg;
    seg.first_slab = 0;
    seg.nslabs = 0;
    seg.count = 0;
    seg.residue = uint32_t(pending_residue);
    order.push_back(seg);
    pending_residue = 0;
  };

  while (i < in.size()) {
    /* A run has to be at least three voxels long to be worth a descriptor:
       two voxels cost about the same either way. */
    const size_t run = aligned ? measure_run(in, i, iv, arrays, strides) : 0;
    if (run >= 3) {
      HaloSegment seg;
      seg.first_slab = uint32_t(slabs.size());
      seg.nslabs = uint32_t(iv);
      seg.count = uint32_t(run);
      seg.residue = 0;
      flush_residue();
      order.push_back(seg);
      for (size_t s = 0; s < iv; ++s) {
        SlabRef sl;
        sl.array = arrays[s];
        sl.base = in[i + s].index;
        sl.counts[0] = int(run);
        sl.counts[1] = 1;
        sl.counts[2] = 1;
        sl.strides[0] = strides[s];
        sl.strides[1] = 0;
        sl.strides[2] = 0;
        slabs.push_back(sl);
      }
      i += run * iv;
    }
    else {
      residue.push_back(in[i]);
      ++pending_residue;
      ++i;
    }
  }
  flush_residue();
}

static void expand_side(const std::vector<SlabRef> &slabs, const std::vector<ElementRef> &residue,
                        const std::vector<HaloSegment> &order, std::vector<ElementRef> &out) {
  out.clear();
  size_t r = 0;
  for (const HaloSegment &seg : order) {
    if (seg.nslabs) {
      for (uint32_t k = 0; k < seg.count; ++k)
        for (uint32_t s = 0; s < seg.nslabs; ++s) {
          const SlabRef &sl = slabs[seg.first_slab + s];
          out.push_back(ElementRef{sl.array, sl.base + ptrdiff_t(k) * sl.strides[0]});
        }
    }
    else {
      for (uint32_t k = 0; k < seg.residue; ++k)
        out.push_back(residue[r++]);
    }
  }
}

void expand_gather(const HaloPlan &p, std::vector<ElementRef> &out) {
  expand_side(p.gather_slabs, p.gather, p.gather_order, out);
}

void expand_scatter(const HaloPlan &p, std::vector<ElementRef> &out) {
  expand_side(p.scatter_slabs, p.scatter, p.scatter_order, out);
}

static void expand_addresses(const std::vector<SlabRef> &slabs,
                             const std::vector<ElementRef> &residue,
                             const std::vector<HaloSegment> &order,
                             const std::vector<HostElementRef> &host,
                             HaloStorageDisposition storage,
                             const HaloArrayTable &arrays, const HostHaloArrayTable &host_arrays,
                             std::vector<realnum *> &out) {
  out.clear();
  if (storage == HaloStorageDisposition::host_owned) {
    out.reserve(host.size());
    for (const HostElementRef &ref : host)
      out.push_back(host_arrays.address(ref.id));
    return;
  }

  std::vector<ElementRef> expanded;
  expand_side(slabs, residue, order, expanded);
  out.reserve(expanded.size());
  for (const ElementRef &ref : expanded)
    out.push_back(arrays.base(ref.array) + ref.index);
}

void expand_gather_addresses(const HaloPlan &p, const HaloArrayTable &arrays,
                             const HostHaloArrayTable &host_arrays,
                             std::vector<realnum *> &out) {
  expand_addresses(p.gather_slabs, p.gather, p.gather_order, p.host_gather, p.storage, arrays,
                   host_arrays, out);
}

void expand_scatter_addresses(const HaloPlan &p, const HaloArrayTable &arrays,
                              const HostHaloArrayTable &host_arrays,
                              std::vector<realnum *> &out) {
  expand_addresses(p.scatter_slabs, p.scatter, p.scatter_order, p.host_scatter, p.storage, arrays,
                   host_arrays, out);
}

CoalesceStats coalesce_stats(const std::vector<HaloPlan> &plans) {
  CoalesceStats st{0, 0, 0, 0};
  for (const HaloPlan &p : plans) {
    if (p.storage == HaloStorageDisposition::host_owned) {
      st.residue_elements += p.host_gather.size();
    }
    else {
      for (const SlabRef &s : p.gather_slabs) {
        st.slab_elements += s.elements();
        ++st.slab_count;
      }
      st.residue_elements += p.gather.size();
    }
  }
  st.total_elements = st.slab_elements + st.residue_elements;
  return st;
}

bool resolve_host_halo_count(bool have_in, int incoming, bool have_out, int outgoing,
                             size_t &count, std::string &why) {
  count = 0;
  why.clear();
  if (!have_in && !have_out) {
    why = "internal halo has no live endpoint";
    return false;
  }
  if ((have_in && incoming < 0) || (have_out && outgoing < 0)) {
    why = "susceptibility returned a negative internal halo count";
    return false;
  }
  if (have_in && have_out && incoming != outgoing) {
    why = "susceptibility endpoints disagree on internal halo count";
    return false;
  }
  count = size_t(have_in ? incoming : outgoing);
  return true;
}

bool checked_add_halo_elements(size_t current, size_t added, size_t &result, std::string &why) {
  if (added > std::numeric_limits<size_t>::max() - current) {
    why = "internal halo element count overflow";
    return false;
  }
  result = current + added;
  why.clear();
  return true;
}

bool checked_multiply_halo_elements(size_t count, size_t lanes, size_t &result,
                                    std::string &why) {
  if (count && lanes > std::numeric_limits<size_t>::max() / count) {
    why = "internal halo lane count overflow";
    return false;
  }
  result = count * lanes;
  why.clear();
  return true;
}

bool reconcile_host_halo_comm_size(int sender_rank, size_t sender_local,
                                   int receiver_rank, size_t receiver_local,
                                   std::string &why) {
  size_t sender_count = sender_local;
  size_t receiver_count = receiver_local;
  broadcast(sender_rank, &sender_count, 1);
  broadcast(receiver_rank, &receiver_count, 1);
  if (sender_count != receiver_count) {
    why = "polarization halo sender and receiver disagree on communication size";
    return false;
  }
  why.clear();
  return true;
}

namespace {

bool remap_elements(const std::vector<ElementRef> &source, const HaloArrayTable &source_arrays,
                    const CpuArrayCatalog &catalog, std::vector<ElementRef> &destination,
                    std::string &why) {
  destination.clear();
  destination.reserve(source.size());
  for (size_t i = 0; i < source.size(); ++i) {
    const ElementRef &ref = source[i];
    const realnum *address = source_arrays.base(ref.array) + ref.index;
    ArrayId id = invalid_array();
    ptrdiff_t offset = 0;
    if (!catalog.locate(address, id, offset)) {
      why = "halo element is not represented in the canonical storage catalog";
      destination.clear();
      return false;
    }
    destination.push_back(ElementRef{id, offset});
  }
  return true;
}

void expand_zero(const ZeroPlan &source, std::vector<ElementRef> &out) {
  out.clear();
  for (size_t i = 0; i < source.slabs.size(); ++i) {
    const SlabRef &slab = source.slabs[i];
    for (int i0 = 0; i0 < slab.counts[0]; ++i0)
      for (int i1 = 0; i1 < slab.counts[1]; ++i1)
        for (int i2 = 0; i2 < slab.counts[2]; ++i2)
          out.push_back(ElementRef{
              slab.array, slab.base + ptrdiff_t(i0) * slab.strides[0] +
                              ptrdiff_t(i1) * slab.strides[1] +
                              ptrdiff_t(i2) * slab.strides[2]});
  }
  out.insert(out.end(), source.residue.begin(), source.residue.end());
}

} // namespace

bool remap_halo_plan(const HaloPlan &source, const HaloArrayTable &source_arrays,
                     const CpuArrayCatalog &catalog, int field_interleave, HaloPlan &destination,
                     std::string &why) {
  why.clear();
  HaloPlan staged = source;
  std::vector<ElementRef> source_gather, source_scatter, gather, scatter;
  expand_gather(source, source_gather);
  expand_scatter(source, source_scatter);

  std::string gather_why, scatter_why;
  const bool gather_mapped =
      remap_elements(source_gather, source_arrays, catalog, gather, gather_why);
  const bool scatter_mapped =
      remap_elements(source_scatter, source_arrays, catalog, scatter, scatter_why);
  const bool polarization = source.ft == PE_stuff || source.ft == PH_stuff;

  if (!gather_mapped || !scatter_mapped) {
    if (!polarization) {
      why = !gather_mapped ? gather_why : scatter_why;
      return false;
    }
    if ((!source_gather.empty() && source.host_gather.size() != source_gather.size()) ||
        (!source_scatter.empty() && source.host_scatter.size() != source_scatter.size())) {
      why = "polarization halo lacks a complete ordered host mirror";
      return false;
    }

    staged.storage = HaloStorageDisposition::host_owned;
    staged.gather_slabs.clear();
    staged.scatter_slabs.clear();
    staged.gather.clear();
    staged.scatter.clear();
    staged.gather_order.clear();
    staged.scatter_order.clear();
    destination = staged;
    why.clear();
    return true;
  }

  const int interleave = polarization ? 1 : field_interleave;
  coalesce_into_slabs(gather, interleave, staged.gather_slabs, staged.gather,
                      staged.gather_order);
  coalesce_into_slabs(scatter, interleave, staged.scatter_slabs, staged.scatter,
                      staged.scatter_order);
  staged.host_gather.clear();
  staged.host_scatter.clear();
  staged.storage = HaloStorageDisposition::canonical;
  destination = staged;
  return true;
}

bool remap_zero_plan(const ZeroPlan &source, const HaloArrayTable &source_arrays,
                     const CpuArrayCatalog &catalog, ZeroPlan &destination, std::string &why) {
  why.clear();
  std::vector<ElementRef> expanded, remapped, unused_residue;
  std::vector<HaloSegment> unused_order;
  expand_zero(source, expanded);
  if (!remap_elements(expanded, source_arrays, catalog, remapped, why)) return false;
  coalesce_into_slabs(remapped, 1, destination.slabs, destination.residue, unused_order);
  return true;
}

} // namespace meep
