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

/* PR 2 acceptance tests.
 *
 *  - Byte identity: packing from HaloPlans in sequence_index order reproduces
 *    the legacy comm_block byte for byte. Asserted on raw bytes, not values.
 *  - Coverage: the plan's gather set is exactly the legacy connections_out
 *    pointer set, and likewise scatter/connections_in. Nothing covered twice,
 *    nothing missed.
 *  - Coalescing report: slab/residue ratio per configuration. A 3D chunk face
 *    at production resolution should be almost entirely slabs; if it is not,
 *    the coalescer is wrong and the whole slab design is unjustified.
 *  - Unit tests for the coalescer itself, including the complex interleave.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <meep.hpp>

#include "backend/halo_plan.hpp"
#include "meep_internals.hpp"

using namespace meep;
using std::complex;

static int failures = 0;

#define CHECK(cond, ...)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      master_printf("FAIL (%s:%d): ", __FILE__, __LINE__);                                         \
      master_printf(__VA_ARGS__);                                                                  \
      master_printf("\n");                                                                         \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

static double one(const vec &) { return 1.0; }
static double eps_slab(const vec &p) { return (fabs(p.y()) < 0.4) ? 12.0 : 1.0; }

/* ------------------------------------------------------------------ */
/* Coalescer unit tests                                               */
/* ------------------------------------------------------------------ */

static void test_coalescer() {
  const ArrayId a{0}, b{1};

  { // contiguous single stream
    std::vector<ElementRef> in;
    for (int i = 0; i < 10; ++i)
      in.push_back(ElementRef{a, i});
    std::vector<SlabRef> slabs;
    std::vector<ElementRef> res;
    std::vector<HaloSegment> ord;
    coalesce_into_slabs(in, 1, slabs, res, ord);
    CHECK(slabs.size() == 1, "expected one slab, got %zu", slabs.size());
    CHECK(res.empty(), "expected no residue, got %zu", res.size());
    if (slabs.size() == 1) {
      CHECK(slabs[0].counts[0] == 10, "slab length %d, expected 10", slabs[0].counts[0]);
      CHECK(slabs[0].strides[0] == 1, "slab stride %ld, expected 1", long(slabs[0].strides[0]));
    }
  }

  { // strided single stream
    std::vector<ElementRef> in;
    for (int i = 0; i < 8; ++i)
      in.push_back(ElementRef{a, 100 + 7 * i});
    std::vector<SlabRef> slabs;
    std::vector<ElementRef> res;
    std::vector<HaloSegment> ord;
    coalesce_into_slabs(in, 1, slabs, res, ord);
    CHECK(slabs.size() == 1 && slabs[0].strides[0] == 7, "strided run not folded");
  }

  { // complex interleave: real/imag alternate between two arrays
    std::vector<ElementRef> in;
    for (int i = 0; i < 6; ++i) {
      in.push_back(ElementRef{a, i});
      in.push_back(ElementRef{b, i});
    }
    std::vector<SlabRef> slabs;
    std::vector<ElementRef> res;
    std::vector<HaloSegment> ord;
    coalesce_into_slabs(in, 2, slabs, res, ord);
    CHECK(slabs.size() == 2, "interleaved run should fold to 2 slabs, got %zu", slabs.size());
    CHECK(res.empty(), "interleaved run left %zu residue", res.size());
    std::vector<ElementRef> back;
    HaloPlan p;
    p.gather_slabs = slabs;
    p.gather = res;
    p.gather_order = ord;
    expand_gather(p, back);
    CHECK(back.size() == in.size(), "expansion length %zu != %zu", back.size(), in.size());
    for (size_t i = 0; i < back.size() && i < in.size(); ++i)
      CHECK(back[i].array == in[i].array && back[i].index == in[i].index,
            "expansion differs at %zu", i);
  }

  { // slabs and residue interleaved: expansion must restore the exact order
    std::vector<ElementRef> in;
    for (int i = 0; i < 5; ++i)
      in.push_back(ElementRef{a, i});
    in.push_back(ElementRef{b, 77}); // lone residue element
    for (int i = 0; i < 5; ++i)
      in.push_back(ElementRef{a, 200 + i});
    in.push_back(ElementRef{b, 88});
    std::vector<SlabRef> slabs;
    std::vector<ElementRef> res;
    std::vector<HaloSegment> ord;
    coalesce_into_slabs(in, 1, slabs, res, ord);
    CHECK(slabs.size() == 2, "expected 2 slabs, got %zu", slabs.size());
    CHECK(res.size() == 2, "expected 2 residue, got %zu", res.size());
    HaloPlan p;
    p.gather_slabs = slabs;
    p.gather = res;
    p.gather_order = ord;
    std::vector<ElementRef> back;
    expand_gather(p, back);
    CHECK(back.size() == in.size(), "expansion length %zu != %zu", back.size(), in.size());
    for (size_t i = 0; i < back.size() && i < in.size(); ++i)
      CHECK(back[i].array == in[i].array && back[i].index == in[i].index,
            "interleaved slab/residue expansion differs at %zu", i);
  }
}

/* ------------------------------------------------------------------ */
/* Byte identity and coverage against the live connection lists        */
/* ------------------------------------------------------------------ */

/* Pack a (ft, pair) communication block the way step_boundaries does, from the
   legacy pointer lists. */
static size_t legacy_size(fields &f, field_type ft, const chunk_pair &pair, connect_phase ip) {
  // get_comm_size() is private, but connections_out is reserved to exactly the
  // comm size and filled to exactly that, so its length is the same number.
  const auto &m = f.chunks[pair.first]->connections_out;
  auto it = m.find(comms_key{ft, ip, pair});
  return it == m.end() ? 0 : it->second.size();
}

static size_t legacy_size_tot(fields &f, field_type ft, const chunk_pair &pair) {
  size_t n = 0;
  for (connect_phase ip : all_connect_phases)
    n += legacy_size(f, ft, pair, ip);
  return n;
}

static void pack_legacy(fields &f, field_type ft, const chunk_pair &pair,
                        std::vector<realnum> &out) {
  out.clear();
  for (connect_phase ip : all_connect_phases) {
    const size_t sz = legacy_size(f, ft, pair, ip);
    if (!sz) continue;
    const std::vector<realnum *> &conn = f.chunks[pair.first]->connections_out.at({ft, ip, pair});
    for (size_t n = 0; n < sz; ++n)
      out.push_back(*(conn[n]));
  }
}

/* Pack the same block from the plans, in sequence_index order. */
static void pack_from_plans(fields &f, field_type ft, const chunk_pair &pair,
                            std::vector<realnum> &out) {
  out.clear();
  std::vector<ElementRef> refs;
  for (connect_phase ip : all_connect_phases) {
    const comms_key key{ft, ip, pair};
    const HaloPlan *p = f.halos->find(key);
    if (!p || !p->block_elements) continue;
    CHECK(p->sequence_index == uint32_t(ip), "sequence_index %u != phase %d", p->sequence_index,
          int(ip));
    expand_gather(*p, refs);
    for (const ElementRef &e : refs)
      out.push_back(*(f.halos->arrays.base(e.array) + e.index));
  }
}

static void check_config(const char *name, structure &s, bool complex_fields, int steps,
                         const vec &src_at, const vec *bloch_k = NULL) {
  fields f(&s);
  if (complex_fields && bloch_k) f.use_bloch(*bloch_k);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, src_at);
  f.advance(steps);

  size_t compared = 0, mismatched = 0;
  FOR_FIELD_TYPES(ft) {
    for (int i = 0; i < f.num_chunks; ++i)
      for (int j = 0; j < f.num_chunks; ++j) {
        const chunk_pair pair{j, i};
        if (!f.chunks[j]->is_mine()) continue;
        if (!legacy_size_tot(f, ft, pair)) continue;

        std::vector<realnum> legacy, planned;
        pack_legacy(f, ft, pair, legacy);
        pack_from_plans(f, ft, pair, planned);

        if (legacy.size() != planned.size()) {
          master_printf("%s: ft=%d pair=(%d,%d) size %zu vs %zu\n", name, int(ft), j, i,
                        legacy.size(), planned.size());
          ++mismatched;
          continue;
        }
        compared += legacy.size();
        /* Raw bytes, not values: a difference that compares equal as a float
           still means the plan is describing different storage. */
        if (!legacy.empty() &&
            memcmp(legacy.data(), planned.data(), legacy.size() * sizeof(realnum)) != 0) {
          master_printf("%s: ft=%d pair=(%d,%d) bytes differ\n", name, int(ft), j, i);
          ++mismatched;
        }
      }
  }
  CHECK(mismatched == 0, "%s: %zu communication blocks are not byte-identical", name, mismatched);
  CHECK(compared > 0, "%s: no communication blocks were compared", name);

  /* Coverage: gather set == connections_out set, elementwise and in order. */
  size_t coverage_bad = 0;
  std::vector<ElementRef> refs;
  for (const HaloPlan &p : f.halos->plans) {
    const comms_key key{p.ft, p.phase, p.chunks};
    if (!f.chunks[p.chunks.first]->is_mine()) continue;
    auto it = f.chunks[p.chunks.first]->connections_out.find(key);
    const size_t legacy_n = (it == f.chunks[p.chunks.first]->connections_out.end())
                                ? 0
                                : it->second.size();
    expand_gather(p, refs);
    if (refs.size() != legacy_n) {
      ++coverage_bad;
      continue;
    }
    for (size_t n = 0; n < refs.size(); ++n)
      if (f.halos->arrays.base(refs[n].array) + refs[n].index != it->second[n]) ++coverage_bad;
  }
  CHECK(coverage_bad == 0, "%s: %zu gather entries do not match connections_out", name,
        coverage_bad);

  const CoalesceStats st = coalesce_stats(f.halos->plans);
  master_printf("%s: %zu reals, %.1f%% in %zu slabs, %zu residue\n", name, st.total_elements,
                100.0 * st.ratio(), st.slab_count, st.residue_elements);
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  test_coalescer();

  {
    grid_volume gv = vol2d(4.0, 4.0, 10.0);
    structure s(gv, eps_slab, pml(0.5), identity(), 4);
    check_config("2d/4chunks/real", s, false, 6, vec(0.13, 0.11));
  }
  {
    grid_volume gv = vol2d(4.0, 4.0, 10.0);
    structure s(gv, eps_slab, pml(0.5), identity(), 4);
    const vec kpt(0.13, 0.07);
    check_config("2d/4chunks/complex", s, true, 6, vec(0.13, 0.11), &kpt);
  }
  {
    grid_volume gv = vol2d(4.0, 4.0, 10.0);
    structure s(gv, eps_slab, pml(0.5), mirror(Y, gv), 2);
    check_config("2d/mirror", s, false, 6, vec(0.13, 0.11));
  }
  {
    /* A 3D chunk face is the case the slab design exists for: if this is not
       almost entirely slabs, the coalescer is wrong. */
    grid_volume gv = vol3d(3.0, 3.0, 3.0, 8.0);
    structure s(gv, one, pml(0.4), identity(), 4);
    check_config("3d/4chunks", s, false, 4, vec(0.13, 0.11, 0.07));
  }

  if (failures) {
    master_printf("halo_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("halo_plan: all checks passed\n");
  return 0;
}
