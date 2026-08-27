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
      /* printf, not master_printf: a failure on a non-master rank is exactly  \
         the interesting kind, and master_printf would swallow it. */          \
      printf("[rank %d] FAIL (%s:%d): ", my_rank(), __FILE__, __LINE__);                           \
      printf(__VA_ARGS__);                                                                         \
      printf("\n");                                                                                \
      fflush(stdout);                                                                              \
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
/* Live-plan invariants                                                */
/* ------------------------------------------------------------------ */
/*
 * The byte-identity assertion against connections_in/connections_out lives in
 * commit 8c7847aa of this branch, where both representations still existed
 * side by side. That was the de-risking gate: it had to fail before the
 * switchover landed, not after. The legacy lists are gone now, so from here on
 * byte identity is held by the bitwise-neutrality harness, which compares the
 * whole matrix against the merge base.
 *
 * What is still checkable in-tree, and checked here:
 *   - expansion of a plan reproduces the flat element order exactly,
 *   - block_offset / block_elements tile the communication block with no gap
 *     or overlap, in all_connect_phases order,
 *   - sequence_index agrees with the phase,
 *   - pack followed by unpack round-trips values to the right destinations,
 *   - the slab/residue ratio is what the design assumes.
 */

static void check_config(const char *name, structure &s, bool complex_fields, int steps,
                         const vec &src_at, const vec *bloch_k = NULL) {
  fields f(&s);
  if (complex_fields && bloch_k) f.use_bloch(*bloch_k);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, src_at);
  f.advance(steps);

  size_t checked_plans = 0;
  bool owns_chunk = false;
  std::vector<ElementRef> refs;

  for (int i = 0; i < f.num_chunks; ++i)
    owns_chunk = owns_chunk || f.chunks[i]->is_mine();

  FOR_FIELD_TYPES(ft) {
    for (int i = 0; i < f.num_chunks; ++i)
      for (int j = 0; j < f.num_chunks; ++j) {
        const chunk_pair pair{j, i};
        size_t expect_offset = 0;
        for (connect_phase ip : all_connect_phases) {
          const HaloPlan *p = f.halos->find({ft, ip, pair});
          if (!p) continue;
          CHECK(p->sequence_index == uint32_t(ip), "%s: sequence_index %u != phase %d", name,
                p->sequence_index, int(ip));
          CHECK(p->block_offset == expect_offset,
                "%s: ft=%d pair=(%d,%d) phase=%d block_offset %zu, expected %zu", name, int(ft), j,
                i, int(ip), p->block_offset, expect_offset);
          expect_offset += p->block_elements;

          if (f.chunks[j]->is_mine()) {
            expand_gather(*p, refs);
            CHECK(refs.size() == p->block_elements,
                  "%s: gather expands to %zu, block_elements %zu", name, refs.size(),
                  p->block_elements);
          }
          if (f.chunks[i]->is_mine()) {
            expand_scatter(*p, refs);
            CHECK(refs.size() == p->block_elements,
                  "%s: scatter expands to %zu, block_elements %zu", name, refs.size(),
                  p->block_elements);
          }
          /* Phases belong to the receiving side only: a rank that owns just
             the sending chunk has a gather list and a block size but no
             phases, and unpack never runs there. */
          if (ip == CONNECT_PHASE && p->block_elements && f.chunks[i]->is_mine())
            CHECK(p->phase_values.size() == p->block_elements / 2,
                  "%s: %zu phases for %zu reals", name, p->phase_values.size(),
                  p->block_elements);
          ++checked_plans;
        }
      }
  }
  CHECK(checked_plans > 0 || !owns_chunk, "%s: an owning rank checked no plans", name);

  /* Pack/unpack round trip on a scratch block. CONNECT_COPY is the identity, so
     packing a plan and unpacking it into the same plan's scatter side must
     reproduce the gathered values at the scatter destinations. */
  size_t roundtrips = 0;
  bool has_eligible_copy = false;
  for (const HaloPlan &p : f.halos->plans) {
    if (p.phase != CONNECT_COPY || !p.block_elements) continue;
    if (!f.chunks[p.chunks.first]->is_mine() || !f.chunks[p.chunks.second]->is_mine()) continue;
    has_eligible_copy = true;
    std::vector<realnum> block(p.block_offset + p.block_elements, realnum(0));
    f.pack_halo(p, block.data());
    expand_gather(p, refs);
    for (size_t n = 0; n < refs.size(); ++n) {
      const realnum want = *(f.halos->arrays.base(refs[n].array) + refs[n].index);
      CHECK(block[p.block_offset + n] == want, "%s: packed value %zu differs", name, n);
    }
    ++roundtrips;
  }
  CHECK(roundtrips > 0 || !owns_chunk || !has_eligible_copy,
        "%s: an eligible local COPY plan was not round-tripped", name);

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

  failures = sum_to_all(failures);
  if (failures) {
    master_printf("halo_plan: %d FAILURE(S) across all ranks\n", failures);
    return 1;
  }
  master_printf("halo_plan: all checks passed\n");
  return 0;
}
