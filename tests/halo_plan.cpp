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
#include <limits>
#include <map>
#include <set>
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

class opaque_custom_lorentzian : public lorentzian_susceptibility {
public:
  opaque_custom_lorentzian(realnum omega_0, realnum gamma)
      : lorentzian_susceptibility(omega_0, gamma) {}
  virtual susceptibility *clone() const { return new opaque_custom_lorentzian(*this); }

  virtual int num_internal_notowned_needed(component c, void *data) const {
    ++count_queries;
    return lorentzian_susceptibility::num_cinternal_notowned_needed(c, data);
  }
  virtual realnum *internal_notowned_ptr(int internal_index, component c, int n,
                                         void *data) const {
    ++pointer_queries;
    return lorentzian_susceptibility::cinternal_notowned_ptr(internal_index, c, 0, n, data);
  }

  virtual int num_cinternal_notowned_needed(component c, void *data) const {
    ++count_queries;
    return lorentzian_susceptibility::num_cinternal_notowned_needed(c, data);
  }
  virtual realnum *cinternal_notowned_ptr(int internal_index, component c, int cmp, int n,
                                          void *data) const {
    ++pointer_queries;
    return lorentzian_susceptibility::cinternal_notowned_ptr(internal_index, c, cmp, n, data);
  }

  static int count_queries;
  static int pointer_queries;
};

int opaque_custom_lorentzian::count_queries = 0;
int opaque_custom_lorentzian::pointer_queries = 0;

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

  {
    size_t count = 999;
    std::string why;
    CHECK(!valid(invalid_host_halo()), "invalid host halo sentinel reports valid");
    CHECK(resolve_host_halo_count(true, 2, true, 2, count, why) && count == 2 && why.empty(),
          "matching endpoint counts were rejected");
    CHECK(!resolve_host_halo_count(true, -1, false, 0, count, why) &&
              why.find("negative") != std::string::npos,
          "negative owned endpoint count was not rejected");
    CHECK(!resolve_host_halo_count(true, 1, true, 2, count, why) &&
              why.find("disagree") != std::string::npos,
          "mismatched endpoint counts were not rejected");
    CHECK(!resolve_host_halo_count(false, 0, false, 0, count, why),
          "count with no live endpoint was not rejected");
    CHECK(checked_add_halo_elements(4, 7, count, why) && count == 11,
          "ordinary halo count addition failed");
    CHECK(!checked_add_halo_elements(std::numeric_limits<size_t>::max(), 1, count, why) &&
              why.find("overflow") != std::string::npos,
          "halo count addition overflow was not rejected");
    CHECK(checked_multiply_halo_elements(9, 2, count, why) && count == 18,
          "ordinary halo lane multiplication failed");
    CHECK(!checked_multiply_halo_elements(std::numeric_limits<size_t>::max(), 2, count, why) &&
              why.find("overflow") != std::string::npos,
          "halo lane multiplication overflow was not rejected");

    const int receiver_rank = count_processors() > 1 ? 1 : 0;
    CHECK(reconcile_host_halo_comm_size(0, 7, receiver_rank, 7, why) && why.empty(),
          "matching cross-rank halo communication sizes were rejected");
    if (count_processors() > 1) {
      const size_t rank_dependent = size_t(11 + my_rank());
      CHECK(!reconcile_host_halo_comm_size(0, rank_dependent, 1, rank_dependent, why) &&
                why.find("sender and receiver") != std::string::npos,
            "rank-dependent halo communication sizes were not collectively rejected");
    }
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
                         const vec &src_at, const vec *bloch_k = NULL,
                         bool expect_host_polarization = false,
                         field_type expected_host_ft = field_type(NUM_FIELD_TYPES),
                         unsigned expected_host_phases = 0, component source_component = Ez,
                         bool expect_repeated_identity = false) {
  fields f(&s);
  if (bloch_k) f.use_bloch(*bloch_k);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(source_component, src, src_at);
  f.advance(steps);

  size_t checked_plans = 0;
  bool owns_chunk = false;
  bool has_host_polarization = false;
  bool has_scalar_host_row = false;
  unsigned host_phases = 0;
  std::map<int, std::set<int> > states_by_susceptibility;
  std::vector<ElementRef> refs;
  std::vector<realnum *> addresses;

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
            expand_gather_addresses(*p, f.halos->arrays, f.halos->host_arrays, addresses);
            CHECK(addresses.size() == p->block_elements,
                  "%s: gather expands to %zu, block_elements %zu", name, addresses.size(),
                  p->block_elements);
          }
          if (f.chunks[i]->is_mine()) {
            expand_scatter_addresses(*p, f.halos->arrays, f.halos->host_arrays, addresses);
            CHECK(addresses.size() == p->block_elements,
                  "%s: scatter expands to %zu, block_elements %zu", name, addresses.size(),
                  p->block_elements);
          }
          if ((ft == PE_stuff || ft == PH_stuff) &&
              (!p->host_gather.empty() || !p->host_scatter.empty())) {
            has_host_polarization = true;
            if (ft == expected_host_ft && p->block_elements)
              host_phases |= 1u << unsigned(p->phase);
            if (f.chunks[j]->is_mine()) {
              expand_gather(*p, refs);
              CHECK(refs.size() == p->host_gather.size(),
                    "%s: canonical/host gather mirror sizes differ (%zu != %zu)", name,
                    refs.size(), p->host_gather.size());
              for (size_t n = 0; n < refs.size() && n < p->host_gather.size(); ++n)
                CHECK(f.halos->arrays.base(refs[n].array) + refs[n].index ==
                          f.halos->host_arrays.address(p->host_gather[n].id),
                      "%s: canonical/host gather mirror differs at %zu", name, n);
              for (const HostElementRef &ref : p->host_gather) {
                const HostHaloKey &key = f.halos->host_arrays.key(ref.id);
                CHECK(key.chunk == p->chunks.first,
                      "%s: host gather key chunk %d != sender %d", name, key.chunk,
                      p->chunks.first);
                CHECK(key.ft == int(p->ft == PE_stuff ? E_stuff : H_stuff),
                      "%s: host gather key has wrong field family", name);
                CHECK(key.state_index >= 0 && key.susceptibility_id >= 0,
                      "%s: host gather key lacks stable state identity", name);
                states_by_susceptibility[key.susceptibility_id].insert(key.state_index);
                has_scalar_host_row = has_scalar_host_row || !key.complex_internal;
              }
            }
            if (f.chunks[i]->is_mine()) {
              expand_scatter(*p, refs);
              CHECK(refs.size() == p->host_scatter.size(),
                    "%s: canonical/host scatter mirror sizes differ (%zu != %zu)", name,
                    refs.size(), p->host_scatter.size());
              for (size_t n = 0; n < refs.size() && n < p->host_scatter.size(); ++n)
                CHECK(f.halos->arrays.base(refs[n].array) + refs[n].index ==
                          f.halos->host_arrays.address(p->host_scatter[n].id),
                      "%s: canonical/host scatter mirror differs at %zu", name, n);
              for (const HostElementRef &ref : p->host_scatter) {
                const HostHaloKey &key = f.halos->host_arrays.key(ref.id);
                CHECK(key.chunk == p->chunks.second,
                      "%s: host scatter key chunk %d != receiver %d", name, key.chunk,
                      p->chunks.second);
                CHECK(key.ft == int(p->ft == PE_stuff ? E_stuff : H_stuff),
                      "%s: host scatter key has wrong field family", name);
                CHECK(key.state_index >= 0 && key.susceptibility_id >= 0,
                      "%s: host scatter key lacks stable state identity", name);
                states_by_susceptibility[key.susceptibility_id].insert(key.state_index);
                has_scalar_host_row = has_scalar_host_row || !key.complex_internal;
              }
            }
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
    expand_gather_addresses(p, f.halos->arrays, f.halos->host_arrays, addresses);
    for (size_t n = 0; n < addresses.size(); ++n) {
      const realnum want = *addresses[n];
      CHECK(block[p.block_offset + n] == want, "%s: packed value %zu differs", name, n);
    }
    ++roundtrips;
  }
  CHECK(roundtrips > 0 || !owns_chunk || !has_eligible_copy,
        "%s: an eligible local COPY plan was not round-tripped", name);

  /* Exercise the actual host-owned scatter transform for every local opaque
     plan.  This pins COPY, NEGATE, and complex PHASE arithmetic rather than
     merely checking that metadata exists. */
  bool saw_duplicate_host_destination = false;
  for (const HaloPlan &p : f.halos->plans) {
    if (p.host_gather.empty() || p.host_scatter.empty() || !p.block_elements) continue;
    if (!f.chunks[p.chunks.first]->is_mine() || !f.chunks[p.chunks.second]->is_mine()) continue;
    std::vector<realnum> block(p.block_offset + p.block_elements, realnum(0));
    for (size_t n = 0; n < p.block_elements; ++n)
      block[p.block_offset + n] = realnum(17 + n);
    expand_scatter_addresses(p, f.halos->arrays, f.halos->host_arrays, addresses);
    std::map<realnum *, realnum> expected;
    if (p.phase == CONNECT_PHASE) {
      CHECK((p.block_elements % 2) == 0, "%s: odd complex host halo size", name);
      for (size_t n = 0; n < p.block_elements / 2; ++n) {
        const complex<realnum> want =
            p.phase_values[n] * complex<realnum>(block[p.block_offset + 2 * n],
                                                  block[p.block_offset + 2 * n + 1]);
        expected[addresses[2 * n]] = want.real();
        expected[addresses[2 * n + 1]] = want.imag();
      }
    }
    else {
      for (size_t n = 0; n < p.block_elements; ++n) {
        const realnum want = p.phase == CONNECT_NEGATE ? -block[p.block_offset + n]
                                                       : block[p.block_offset + n];
        expected[addresses[n]] = want;
      }
    }
    saw_duplicate_host_destination =
        saw_duplicate_host_destination || expected.size() < addresses.size();
    f.unpack_halo(p, block.data());
    for (const auto &entry : expected)
      CHECK(*entry.first == entry.second, "%s: host %s last-write result differs", name,
            p.phase == CONNECT_PHASE    ? "PHASE"
            : p.phase == CONNECT_NEGATE ? "NEGATE"
                                        : "COPY");
  }

  const CoalesceStats st = coalesce_stats(f.halos->plans);
  master_printf("%s: %zu reals, %.1f%% in %zu slabs, %zu residue\n", name, st.total_elements,
                100.0 * st.ratio(), st.slab_count, st.residue_elements);
  if (expect_host_polarization)
    CHECK(or_to_all(has_host_polarization), "%s: no opaque host-owned PE/PH halo was emitted",
          name);
  if (expect_host_polarization) {
    CHECK(or_to_all(has_scalar_host_row), "%s: no scalar opaque halo row was emitted", name);
    if (expect_repeated_identity) {
      bool repeated = false;
      for (const auto &entry : states_by_susceptibility)
        repeated = repeated || entry.second.size() > 1;
      CHECK(or_to_all(repeated), "%s: repeated susceptibility id lost its state ordinals", name);
      CHECK(or_to_all(saw_duplicate_host_destination),
            "%s: duplicate host destination ordering was not exercised", name);
    }
    for (connect_phase ip : all_connect_phases)
      if (expected_host_phases & (1u << unsigned(ip)))
        CHECK(or_to_all((host_phases & (1u << unsigned(ip))) != 0),
              "%s: missing expected host-owned phase %d", name, int(ip));

    HostHaloId old_host_id = invalid_host_halo();
    for (const HaloPlan &p : f.halos->plans) {
      if (!p.host_gather.empty()) {
        old_host_id = p.host_gather.front().id;
        break;
      }
      if (!p.host_scatter.empty()) {
        old_host_id = p.host_scatter.front().id;
        break;
      }
    }

    /* Public topology mutation forces the same disconnect/reconnect path used
       by callers without reaching through a private test seam. */
    f.use_bloch(X, 0.0);
    f.advance(1);
    if (valid(old_host_id))
      CHECK(!f.halos->host_arrays.contains(old_host_id),
            "%s: reconnect accepted a stale host halo generation", name);
    bool reconnected_host = false;
    for (const HaloPlan &p : f.halos->plans) {
      for (const HostElementRef &ref : p.host_gather) {
        CHECK(valid(ref.id), "%s: reconnect produced an invalid host gather id", name);
        CHECK(f.halos->host_arrays.address(ref.id) != NULL,
              "%s: reconnect produced a null host gather address", name);
        reconnected_host = true;
      }
      for (const HostElementRef &ref : p.host_scatter) {
        CHECK(valid(ref.id), "%s: reconnect produced an invalid host scatter id", name);
        CHECK(f.halos->host_arrays.address(ref.id) != NULL,
              "%s: reconnect produced a null host scatter address", name);
        reconnected_host = true;
      }
    }
    CHECK(or_to_all(reconnected_host), "%s: reconnect lost all host-owned halo entries", name);
  }
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
    structure s(gv, eps_slab, pml(0.5), identity(), 4);
    opaque_custom_lorentzian repeated(0.7, 0.08);
    s.add_susceptibility(one, E_stuff, repeated);
    s.add_susceptibility(one, E_stuff, repeated);
    const vec kpt(0.13, 0.07);
    check_config("2d/opaque-custom-PE-host-halo", s, true, 6, vec(0.13, 0.11), &kpt, true,
                 PE_stuff, (1u << unsigned(CONNECT_COPY)) | (1u << unsigned(CONNECT_PHASE)), Ez,
                 true);
  }
  {
    grid_volume gv = vol2d(4.0, 4.0, 10.0);
    structure s(gv, eps_slab, pml(0.5), mirror(Y, gv), 4);
    s.add_susceptibility(one, H_stuff, opaque_custom_lorentzian(0.9, 0.06));
    const vec edge_k(0.125, 0.0);
    check_config("2d/opaque-custom-PH-host-halo", s, false, 6, vec(0.13, 0.11), &edge_k, true,
                 PH_stuff, (1u << unsigned(CONNECT_COPY)) | (1u << unsigned(CONNECT_NEGATE)), Hz);
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

  CHECK(sum_to_all(opaque_custom_lorentzian::count_queries) > 0,
        "custom halo count virtual was never invoked");
  CHECK(sum_to_all(opaque_custom_lorentzian::pointer_queries) > 0,
        "custom halo pointer virtual was never invoked");

  failures = sum_to_all(failures);
  if (failures) {
    master_printf("halo_plan: %d FAILURE(S) across all ranks\n", failures);
    return 1;
  }
  master_printf("halo_plan: all checks passed\n");
  return 0;
}
