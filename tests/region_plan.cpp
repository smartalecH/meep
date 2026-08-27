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

/* PR 6 acceptance tests: region planning.
 *
 * The region enumerator is the largest purely mechanical move in the stack and
 * the easiest place to silently drop an edge case -- empty-dimension snapping,
 * cylindrical dV factors, m-dependent origin rules, periodic image
 * enumeration. The argument-sequence test is the only thing that catches it.
 *
 * So: record the legacy callback's full argument tuple, in order, and assert
 * the planned regions reproduce it exactly. Both go through the same
 * enumerator now, so what this really pins is that the adapter passes
 * everything through untouched and in the same order -- and it would catch any
 * future attempt to "simplify" ChunkLoopRegion by dropping a field.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <meep.hpp>

#include "backend/region_plan.hpp"
#include "meep_internals.hpp"

using namespace meep;
using std::complex;

static int failures = 0;

#define CHECK(cond, ...)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      printf("[rank %d] FAIL (%s:%d): ", my_rank(), __FILE__, __LINE__);                           \
      printf(__VA_ARGS__);                                                                         \
      printf("\n");                                                                                \
      fflush(stdout);                                                                              \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

static double one(const vec &) { return 1.0; }
static double eps_slab(const vec &p) { return (fabs(p.y()) < 0.4) ? 12.0 : 1.0; }

/* Everything the legacy field_chunkloop is handed. */
struct arg_tuple {
  const fields_chunk *fc;
  int ichunk;
  component cgrid;
  ivec is, ie;
  vec s0, s1, e0, e1;
  double dV0, dV1;
  ivec shift;
  complex<double> shift_phase;
  int sn;
};

static std::vector<arg_tuple> recorded;

static void recorder(fields_chunk *fc, int ichunk, component cgrid, ivec is, ivec ie, vec s0,
                     vec s1, vec e0, vec e1, double dV0, double dV1, ivec shift,
                     complex<double> shift_phase, const symmetry &S, int sn, void *) {
  (void)S;
  arg_tuple a = {fc, ichunk, cgrid, is,  ie,    s0,          s1, e0,
                 e1, dV0,    dV1,   shift, shift_phase, sn};
  recorded.push_back(a);
}

static bool same(const vec &a, const vec &b) {
  LOOP_OVER_DIRECTIONS(a.dim, d) {
    if (a.in_direction(d) != b.in_direction(d)) return false;
  }
  return true;
}

static void check_case(const char *name, fields &f, const volume &where, component cgrid,
                       bool use_symmetry, bool snap) {
  recorded.clear();
  f.loop_in_chunks(recorder, NULL, where, cgrid, use_symmetry, snap);
  const std::vector<arg_tuple> legacy = recorded;

  const ChunkLoopPlan plan = prepare_loop_in_chunks(f, where, cgrid, use_symmetry, snap);

  CHECK(plan.regions.size() == legacy.size(), "%s: %zu planned regions vs %zu callback calls", name,
        plan.regions.size(), legacy.size());
  if (plan.regions.size() != legacy.size()) return;

  size_t bad = 0;
  for (size_t i = 0; i < legacy.size(); ++i) {
    const arg_tuple &a = legacy[i];
    const ChunkLoopRegion &r = plan.regions[i];
    if (r.chunk != a.ichunk) ++bad;
    if (f.chunks[r.chunk] != a.fc) ++bad;
    if (r.transformed_grid_component != a.cgrid) ++bad;
    if (!(r.begin == a.is) || !(r.end == a.ie)) ++bad;
    if (!same(r.weights.s0, a.s0) || !same(r.weights.s1, a.s1)) ++bad;
    if (!same(r.weights.e0, a.e0) || !same(r.weights.e1, a.e1)) ++bad;
    if (r.dV0 != a.dV0 || r.dV1 != a.dV1) ++bad;
    if (!(r.lattice_shift == a.shift)) ++bad;
    if (r.phase != a.shift_phase) ++bad;
    if (r.symmetry_index != a.sn) ++bad;
  }
  CHECK(bad == 0, "%s: %zu argument mismatches across %zu regions", name, bad, legacy.size());
  if (!bad) master_printf("%s: %zu regions match\n", name, legacy.size());
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  { // 2d, uneven chunks, a sub-volume that clips chunk boundaries
    grid_volume gv = vol2d(4.0, 4.0, 10.0);
    structure s(gv, eps_slab, pml(0.5), identity(), 3);
    fields f(&s);
    gaussian_src_time src(0.3, 0.1);
    f.add_point_source(Ez, src, vec(0.13, 0.11));
    f.advance(2);
    check_case("2d/uneven/full", f, gv.surroundings(), Centered, true, false);
    check_case("2d/uneven/sub", f, volume(vec(-0.9, -0.7), vec(1.1, 0.9)), Ez, true, false);
    // an empty dimension: a plane, which exercises the snapping rules
    check_case("2d/plane/snap", f, volume(vec(0.4, -1.2), vec(0.4, 1.2)), Ez, true, true);
    check_case("2d/plane/nosnap", f, volume(vec(0.4, -1.2), vec(0.4, 1.2)), Ez, true, false);
    check_case("2d/nosym", f, volume(vec(-0.9, -0.7), vec(1.1, 0.9)), Ez, false, false);
  }
  { // mirror symmetry: the transform, phase and endpoint-swap paths
    grid_volume gv = vol2d(4.0, 4.0, 10.0);
    structure s(gv, eps_slab, pml(0.5), mirror(Y, gv), 2);
    fields f(&s);
    gaussian_src_time src(0.3, 0.1);
    f.add_point_source(Ez, src, vec(0.13, 0.0));
    f.advance(2);
    check_case("2d/mirror/full", f, gv.surroundings(), Centered, true, false);
    check_case("2d/mirror/sub", f, volume(vec(-1.0, -1.0), vec(1.0, 1.0)), Ez, true, false);
  }
  { // periodic images
    grid_volume gv = vol2d(3.0, 3.0, 10.0);
    structure s(gv, one, no_pml(), identity(), 2);
    fields f(&s);
    f.use_bloch(vec(0.11, 0.07));
    gaussian_src_time src(0.3, 0.1);
    f.add_point_source(Ez, src, vec(0.13, 0.11));
    f.advance(2);
    check_case("2d/periodic", f, volume(vec(-2.2, -2.2), vec(2.2, 2.2)), Ez, true, false);
  }
  { // cylindrical: dV0/dV1 and the m-dependent origin rules
    grid_volume gv = volcyl(2.0, 3.0, 10.0);
    structure s(gv, eps_slab, pml(0.5));
    fields f(&s, 1); // m = 1
    gaussian_src_time src(0.3, 0.1);
    f.add_point_source(Er, src, veccyl(0.7, 0.1));
    f.advance(2);
    check_case("cyl/m1/full", f, gv.surroundings(), Centered, true, false);
    check_case("cyl/m1/sub", f, volume(veccyl(0.2, -0.6), veccyl(1.4, 0.6)), Ez, true, false);
  }
  { // 3d
    grid_volume gv = vol3d(2.5, 2.5, 2.5, 7.0);
    structure s(gv, eps_slab, pml(0.4), identity(), 3);
    fields f(&s);
    gaussian_src_time src(0.3, 0.1);
    f.add_point_source(Ez, src, vec(0.13, 0.11, 0.07));
    f.advance(2);
    check_case("3d/uneven", f, volume(vec(-0.8, -0.8, -0.8), vec(0.8, 0.8, 0.8)), Ez, true, false);
  }

  if (failures) {
    master_printf("region_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("region_plan: all checks passed\n");
  return 0;
}
