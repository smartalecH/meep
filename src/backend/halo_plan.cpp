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

namespace meep {

ArrayId HaloArrayTable::intern(const HaloArrayKey &key, realnum *base, size_t elements,
                               array_role role) {
  auto it = index_.find(key);
  if (it != index_.end()) return ArrayId{it->second};

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

CoalesceStats coalesce_stats(const std::vector<HaloPlan> &plans) {
  CoalesceStats st{0, 0, 0, 0};
  for (const HaloPlan &p : plans) {
    for (const SlabRef &s : p.gather_slabs) {
      st.slab_elements += s.elements();
      ++st.slab_count;
    }
    st.residue_elements += p.gather.size();
  }
  st.total_elements = st.slab_elements + st.residue_elements;
  return st;
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
  destination = source;

  std::vector<ElementRef> source_gather, source_scatter, gather, scatter;
  expand_gather(source, source_gather);
  expand_scatter(source, source_scatter);
  if (!remap_elements(source_gather, source_arrays, catalog, gather, why) ||
      !remap_elements(source_scatter, source_arrays, catalog, scatter, why))
    return false;

  const int interleave = source.ft == PE_stuff || source.ft == PH_stuff ? 1 : field_interleave;
  coalesce_into_slabs(gather, interleave, destination.gather_slabs, destination.gather,
                      destination.gather_order);
  coalesce_into_slabs(scatter, interleave, destination.scatter_slabs, destination.scatter,
                      destination.scatter_order);
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
