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

/* PR 1 acceptance tests: the invalidation closure table, advance(n) vs. n
   step() calls, and the MEEP_FINITE_CHECK modes. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <meep.hpp>

#include "backend/diagnostics.hpp"
#include "backend/lifecycle.hpp"

using namespace meep;
using std::complex;

static int failures = 0;

static double one(const vec &) { return 1.0; }

#define CHECK(cond, ...)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      master_printf("FAIL (%s:%d): ", __FILE__, __LINE__);                                         \
      master_printf(__VA_ARGS__);                                                                  \
      master_printf("\n");                                                                         \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

/* ------------------------------------------------------------------ */
/* 1. The dirty-closure table (plan section 6.4), one row at a time.   */
/* ------------------------------------------------------------------ */

static void test_closure_table() {
  struct row {
    MutationKind cause;
    DirtyMask expected;
  };
  const row table[] = {
      {MutationKind::field_values, dirty_initialization},
      {MutationKind::source_values, dirty_initialization},
      {MutationKind::source_definition, dirty_source_plan | dirty_regions | dirty_executable},
      {MutationKind::monitor_definition,
       dirty_monitor_plan | dirty_regions | dirty_storage | dirty_executable},
      {MutationKind::material_values, dirty_initialization | dirty_classification},
      {MutationKind::material_region, dirty_initialization | dirty_classification},
      {MutationKind::material_phase,
       dirty_initialization | dirty_classification | dirty_executable},
      {MutationKind::material_definition, dirty_initialization | dirty_storage | dirty_halos |
                                              dirty_classification | dirty_executable},
      {MutationKind::field_layout, dirty_storage | dirty_halos | dirty_executable},
      {MutationKind::boundary_topology, dirty_regions | dirty_halos | dirty_executable},
      {MutationKind::chunk_topology,
       dirty_storage | dirty_regions | dirty_halos | dirty_executable},
      {MutationKind::coordinate_definition, dirty_executable},
      {MutationKind::precision_policy,
       dirty_storage | dirty_initialization | dirty_executable},
  };
  const int n = int(sizeof(table) / sizeof(table[0]));
  CHECK(n == mutation_kind_count, "table covers %d of %d MutationKinds", n, mutation_kind_count);

  bool seen[mutation_kind_count] = {false};
  for (int i = 0; i < n; ++i) {
    const DirtyMask got = invalidation_closure(table[i].cause);
    CHECK(got == table[i].expected, "closure(%s) = 0x%02x, expected 0x%02x",
          mutation_kind_name(table[i].cause), unsigned(got), unsigned(table[i].expected));
    seen[int(table[i].cause)] = true;
  }
  for (int i = 0; i < mutation_kind_count; ++i)
    CHECK(seen[i], "MutationKind %d missing from the table test", i);
}

/* ------------------------------------------------------------------ */
/* 2. Generations advance, and only for the cause that fired.          */
/* ------------------------------------------------------------------ */

static void test_generations() {
  const double a = 8.0;
  grid_volume gv = vol2d(3.0, 3.0, a);
  structure s(gv, one, no_pml());
  fields f(&s);

  uint64_t before[mutation_kind_count];
  for (int i = 0; i < mutation_kind_count; ++i)
    before[i] = f.mutation_generation[i];

  /* The constructor legitimately dirties nearly everything (chunk_topology +
     material_definition), so start from a clean mask to isolate this cause. */
  clear_dirty(f, ~DirtyMask(0));
  CHECK(f.dirty_mask == dirty_none, "clear_dirty(all) left 0x%02x", unsigned(f.dirty_mask));

  invalidate(f, MutationKind::source_values);

  for (int i = 0; i < mutation_kind_count; ++i) {
    const uint64_t expect = before[i] + (i == int(MutationKind::source_values) ? 1 : 0);
    CHECK(f.mutation_generation[i] == expect, "generation[%s] = %llu, expected %llu",
          mutation_kind_name(MutationKind(i)), (unsigned long long)f.mutation_generation[i],
          (unsigned long long)expect);
  }
  CHECK(is_dirty(f, dirty_initialization), "source_values must dirty initialization");
  CHECK(!is_dirty(f, dirty_executable), "source_values must not dirty the executable");

  clear_dirty(f, dirty_initialization);
  CHECK(!is_dirty(f, dirty_initialization), "clear_dirty did not clear");
}

/* ------------------------------------------------------------------ */
/* 3. advance(n) is bitwise-equal to n consecutive step() calls.       */
/* ------------------------------------------------------------------ */

static double eps_slab(const vec &p) { return (fabs(p.y()) < 0.5) ? 9.0 : 1.0; }

static void build(structure **sp, fields **fp) {
  const double a = 10.0;
  grid_volume gv = vol2d(4.0, 4.0, a);
  *sp = new structure(gv, eps_slab, pml(0.5));
  *fp = new fields(*sp);
  gaussian_src_time src(0.3, 0.1);
  (*fp)->add_point_source(Ez, src, vec(0.2, 0.1));
}

/* Compare every allocated field array bit-for-bit. */
static bool identical_state(fields &f1, fields &f2, const char *tag) {
  bool same = true;
  CHECK(f1.num_chunks == f2.num_chunks, "%s: chunk count differs", tag);
  if (f1.num_chunks != f2.num_chunks) return false;
  int compared = 0;
  for (int i = 0; i < f1.num_chunks; ++i) {
    if (!f1.chunks[i]->is_mine()) continue;
    const size_t ntot = size_t(f1.chunks[i]->gv.ntot());
    for (int c = 0; c < NUM_FIELD_COMPONENTS; ++c)
      for (int cmp = 0; cmp < 2; ++cmp) {
        const realnum *a1 = f1.chunks[i]->f[c][cmp];
        const realnum *a2 = f2.chunks[i]->f[c][cmp];
        if (!a1 != !a2) {
          master_printf("%s: allocation mismatch for component %d cmp %d\n", tag, c, cmp);
          same = false;
          continue;
        }
        if (!a1) continue;
        ++compared;
        if (memcmp(a1, a2, ntot * sizeof(realnum)) != 0) {
          master_printf("%s: component %d cmp %d differs\n", tag, c, cmp);
          same = false;
        }
      }
  }
  CHECK(compared > 0, "%s: nothing was compared", tag);
  return same;
}

static void test_advance_equivalence() {
  const int n = 17;
  structure *s1, *s2;
  fields *f1, *f2;
  build(&s1, &f1);
  build(&s2, &f2);

  for (int i = 0; i < n; ++i)
    f1->step();
  f2->advance(n);

  CHECK(f1->t == f2->t, "advance(%d) reached t=%d, %d step()s reached t=%d", n, f2->t, n, f1->t);
  CHECK(identical_state(*f1, *f2, "advance vs step"),
        "advance(%d) is not bitwise-equal to %d step() calls", n, n);

  /* Batched and unbatched advance must also agree with each other. */
  structure *s3;
  fields *f3;
  build(&s3, &f3);
  f3->advance(5);
  f3->advance(7);
  f3->advance(5);
  CHECK(identical_state(*f1, *f3, "split advance"), "advance(5)+advance(7)+advance(5) != %d steps",
        n);

  delete f1;
  delete f2;
  delete f3;
  delete s1;
  delete s2;
  delete s3;
}

/* ------------------------------------------------------------------ */
/* 4. MEEP_FINITE_CHECK modes.                                         */
/* ------------------------------------------------------------------ */

static void test_finite_check_modes() {
  /* `step` is the default and is what the rest of the suite runs under; the
     interesting property here is that `off` and `batch` do not perturb the
     result, since they only change *when* the center point is read. */
  const int n = 11;
  structure *s1, *s2, *s3;
  fields *f1, *f2, *f3;
  build(&s1, &f1);
  build(&s2, &f2);
  build(&s3, &f3);

  set_finite_check_mode(FiniteCheckMode::step);
  f1->advance(n);
  set_finite_check_mode(FiniteCheckMode::batch);
  f2->advance(n);
  set_finite_check_mode(FiniteCheckMode::off);
  f3->advance(n);
  set_finite_check_mode(FiniteCheckMode::step);

  CHECK(identical_state(*f1, *f2, "finite step vs batch"), "MEEP_FINITE_CHECK=batch perturbs state");
  CHECK(identical_state(*f1, *f3, "finite step vs off"), "MEEP_FINITE_CHECK=off perturbs state");

  CHECK(f1->nonfinite_flag == 0, "a healthy simulation must not set nonfinite_flag");
  CHECK(f1->first_bad_step == -1, "first_bad_step should stay -1 when nothing went wrong");

  delete f1;
  delete f2;
  delete f3;
  delete s1;
  delete s2;
  delete s3;
}

/* ------------------------------------------------------------------ */
/* 5. Lifecycle behavior around mutations.                             */
/* ------------------------------------------------------------------ */

static void test_mutation_lifecycle() {
  const double a = 10.0;
  grid_volume gv = vol2d(3.0, 3.0, a);
  structure s(gv, one, pml(0.5));
  fields f(&s);

  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.1, 0.1));
  f.advance(3);

  /* A mid-run source addition must still be honored: PR 3 removes the lazy
     allocation that makes this work today, so the behavior is pinned here
     first. */
  const uint64_t before = generation(f, MutationKind::source_definition);
  gaussian_src_time src2(0.25, 0.1, 0.0, 6.0);
  f.add_point_source(Hz, src2, vec(-0.3, 0.2));
  invalidate(f, MutationKind::source_definition);
  CHECK(generation(f, MutationKind::source_definition) == before + 1,
        "source_definition generation did not advance");
  f.advance(3);
  CHECK(f.t == 6, "expected t=6 after two 3-step batches, got %d", f.t);

  /* change_k_point is a boundary-topology mutation and must invalidate the
     connections. A step() rebuilds them, so connections are current here. */
  f.advance(1);
  const uint64_t built = f.connections_built_generation;
  CHECK(connections_are_current(f), "connections should be current after a step");
  f.use_bloch(X, 0.1);
  CHECK(!connections_are_current(f), "use_bloch must invalidate chunk connections");
  CHECK(f.connections_generation > built, "connection generation did not advance");
  CHECK(is_dirty(f, dirty_halos), "boundary_topology must dirty the halos");

  grid_volume cyl_gv = volcyl(1.0, 1.0, a);
  structure cyl_s(cyl_gv, one, no_pml());
  fields cyl_f(&cyl_s, 1.0);
  clear_dirty(cyl_f, ~DirtyMask(0));
  const uint64_t coordinate_before = generation(cyl_f, MutationKind::coordinate_definition);
  cyl_f.change_m(-1.0);
  CHECK(generation(cyl_f, MutationKind::coordinate_definition) == coordinate_before + 1,
        "change_m must advance the coordinate-state generation");
  CHECK(is_dirty(cyl_f, dirty_executable), "change_m must dirty the executable");
  CHECK(!is_dirty(cyl_f, dirty_storage), "same-layout change_m must preserve storage");
  for (int i = 0; i < cyl_f.num_chunks; ++i)
    CHECK(cyl_f.chunks[i]->m == -1.0, "change_m did not propagate to chunk %d", i);

  clear_dirty(cyl_f, ~DirtyMask(0));
  const uint64_t unchanged_generation =
      generation(cyl_f, MutationKind::coordinate_definition);
  cyl_f.change_m(-1.0);
  CHECK(generation(cyl_f, MutationKind::coordinate_definition) == unchanged_generation,
        "repeating change_m with the live value must not advance its generation");
  CHECK(cyl_f.dirty_mask == dirty_none,
        "repeating change_m with the live value must not dirty prepared state");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  test_closure_table();
  test_generations();
  test_advance_equivalence();
  test_finite_check_modes();
  test_mutation_lifecycle();

  if (failures) {
    master_printf("lifecycle: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("lifecycle: all checks passed\n");
  return 0;
}
