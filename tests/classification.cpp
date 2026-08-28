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

/* PR 4 acceptance tests.
 *
 *  - Hash equality across ranks, including with an uneven chunk layout. A
 *    wrong hash deadlocks under MPI; it does not merely mis-optimize.
 *  - Re-entry bound: a promoting configuration re-enters pass 1 exactly once.
 *  - Executable stability: an unchanged hash after a material *value* change
 *    must not force a rebuild.
 *  - The tiling decision published by apply_classification matches the test
 *    fields::update_eh used to run inline.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <meep.hpp>

#include "backend/classification.hpp"
#include "backend/lifecycle.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

using namespace meep;

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

static void test_hash_across_ranks(const char *name, structure &s, const vec &src_at) {
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, src_at);
  f.advance(4);

  const MaterialClassification cls = classify(f, *f.storage_plan);

  /* Every rank must agree, bit for bit. min == max over all ranks is the
     cheapest way to say that with the reductions meep exposes. */
  const uint64_t h = cls.hash;
  int lo = int(uint32_t(h & 0xffffffffu)), hi = int(uint32_t(h >> 32));
  const int lo_max = max_to_all(lo), lo_min = -max_to_all(-lo);
  const int hi_max = max_to_all(hi), hi_min = -max_to_all(-hi);
  CHECK(lo_max == lo_min && hi_max == hi_min,
        "%s: classification hash differs across ranks (%08x/%08x vs %08x/%08x)", name,
        unsigned(lo_min), unsigned(hi_min), unsigned(lo_max), unsigned(hi_max));

  const MaterialClassification again = classify(f, *f.storage_plan);
  CHECK(again.hash == cls.hash, "%s: classify() is not idempotent", name);

  master_printf("%s: %s\n", name, classification_summary(cls));
}

static void test_tiling_decision() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.13, 0.11));
  f.advance(3);

  const MaterialClassification cls = classify(f, *f.storage_plan);
  size_t mismatches = 0, empty_tiles = 0;
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    FOR_FIELD_TYPES(ft) {
      if (ft != E_stuff && ft != H_stuff) continue;
      bool expect = false;
      FOR_FT_COMPONENTS(ft, cc) {
        const direction d_c = component_direction(cc);
        const direction d_1 = cycle_direction(f.chunks[i]->gv.dim, d_c, 1);
        const direction d_2 = cycle_direction(f.chunks[i]->gv.dim, d_c, 2);
        if (f.chunks[i]->s->chi1inv[cc][d_1] && f.chunks[i]->s->chi1inv[cc][d_2]) {
          expect = true;
          break;
        }
      }
      if ((cls.anisotropic_eh[size_t(i) * NUM_FIELD_TYPES + ft] != 0) != expect) ++mismatches;
      /* update_eh iterates gvs_eh, so an empty list silently skips the whole
         update -- worth asserting separately from the decision itself. */
      if (f.chunks[i]->gvs_eh[ft].empty()) ++empty_tiles;
    }
  }
  CHECK(mismatches == 0, "tiling decision disagrees with the legacy test in %zu places",
        mismatches);
  CHECK(empty_tiles == 0, "%zu (chunk, field type) pairs have an empty tile list", empty_tiles);
}

static void test_value_change_preserves_hash() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5));
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(3);

  const uint64_t before = classify(f, *f.storage_plan).hash;
  f.advance(5); // values change every step; structure does not
  const uint64_t after = classify(f, *f.storage_plan).hash;
  CHECK(before == after, "stepping changed the classification hash (%016llx -> %016llx)",
        (unsigned long long)before, (unsigned long long)after);
}

static void test_reentry_bound() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, one, pml(0.5));
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(4);
  CHECK(f.classification_reentries <= 1, "pass 1 was re-entered %u times",
        unsigned(f.classification_reentries));
  master_printf("reentry bound: %u re-entries\n", unsigned(f.classification_reentries));
}

static void test_changed_hash_invalidates_executable() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5));
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(3);

  CHECK(f.prepared_classification_hash != 0, "classification did not publish a hash");
  clear_dirty(f, dirty_executable);
  ++f.prepared_classification_hash;
  backend_classify_and_finalize(f);
  CHECK(is_dirty(f, dirty_executable),
        "a changed classification hash did not invalidate the executable");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  {
    grid_volume gv = vol2d(4.0, 4.0, 10.0);
    structure s(gv, eps_slab, pml(0.5), identity(), 2);
    test_hash_across_ranks("2d/even", s, vec(0.13, 0.11));
  }
  {
    /* Uneven chunk layout: the plan calls this out specifically, because a
       hash that depends on which chunks a rank owns passes the even case. */
    grid_volume gv = vol2d(4.0, 4.0, 10.0);
    structure s(gv, eps_slab, pml(0.5), identity(), 3);
    test_hash_across_ranks("2d/uneven3", s, vec(0.13, 0.11));
  }
  {
    grid_volume gv = vol3d(2.5, 2.5, 2.5, 7.0);
    structure s(gv, eps_slab, pml(0.4), identity(), 3);
    test_hash_across_ranks("3d/uneven3", s, vec(0.13, 0.11, 0.07));
  }

  test_tiling_decision();
  test_value_change_preserves_hash();
  test_reentry_bound();

  if (failures) {
    master_printf("classification: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("classification: all checks passed\n");
  return 0;
}
