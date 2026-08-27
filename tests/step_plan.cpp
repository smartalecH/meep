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

/* PR 5 acceptance tests.
 *
 * The plan calls for a trace of the real fields::step compared against the
 * built plan. Once step() *is* the plan, that comparison is vacuous, so what
 * is asserted here instead is the transcription itself: the expected operation
 * sequence is written out longhand, in the order src/step.cpp:58-138 ran it,
 * and a reviewer can line the two up by eye. That is the property the plan
 * actually cares about -- "if a reviewer cannot line up build_step_plan()
 * against src/step.cpp by eye, it is wrong."
 *
 * Also checked: guard kinds (they determine graph structure in Phase 2 even
 * though the CPU executor ignores the distinction), presence/absence of
 * optional work, the solve_cw program, and plan stability across steps.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include <meep.hpp>

#include "backend/lifecycle.hpp"
#include "backend/step_plan.hpp"
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

static void compare(const char *name, const std::vector<std::string> &got,
                    const char *const *want, size_t nwant) {
  if (got.size() != nwant) {
    master_printf("FAIL: %s: plan has %zu operations, expected %zu\n", name, got.size(), nwant);
    for (size_t i = 0; i < got.size() || i < nwant; ++i)
      master_printf("  %2zu  %-32s  %s\n", i, i < got.size() ? got[i].c_str() : "-",
                    i < nwant ? want[i] : "-");
    ++failures;
    return;
  }
  for (size_t i = 0; i < nwant; ++i)
    if (got[i] != want[i]) {
      master_printf("FAIL: %s: operation %zu is %s, expected %s\n", name, i, got[i].c_str(),
                    want[i]);
      ++failures;
    }
}

/* The ordinary timestep, transcribed from fields::step_once. Compare against
   the schedule in section 10.2 of the plan. */
static const char *const expected_full[] = {
    "restore_magnetic_fields",
    "update_material_coefficients",
    "evaluate_source_scalars",   // time()                 -- B sources
    "update_db(B)",
    "apply_sources(B)",
    "transfer_halo(B)",
    "evaluate_source_scalars",   // time() + 0.5*dt        -- integrated H
    "update_eh(H)",
    "transfer_halo(WH)",
    "update_polarization(H)",
    "transfer_halo(PH)",
    "transfer_halo(H)",
    "update_flux_half",
    "evaluate_source_scalars",   // time() + 0.5*dt        -- D sources
    "update_db(D)",
    "apply_sources(D)",
    "transfer_halo(D)",
    "evaluate_source_scalars",   // time() + dt            -- integrated E
    "update_eh(E)",
    "transfer_halo(WE)",
    "update_polarization(E)",
    "transfer_halo(PE)",
    "transfer_halo(E)",
    "update_flux",
    "increment_time",
    "update_dft",
    "synchronize_magnetic_fields",
    "finite_value_check",
};

/* Same simulation with no sources, no fluxes and no monitors: the optional
   work is omitted, and nothing else moves. */
static const char *const expected_empty[] = {
    "restore_magnetic_fields",
    "update_material_coefficients",
    "update_db(B)",
    "apply_sources(B)",
    "transfer_halo(B)",
    "update_eh(H)",
    "transfer_halo(WH)",
    "update_polarization(H)",
    "transfer_halo(PH)",
    "transfer_halo(H)",
    "update_db(D)",
    "apply_sources(D)",
    "transfer_halo(D)",
    "update_eh(E)",
    "transfer_halo(WE)",
    "update_polarization(E)",
    "transfer_halo(PE)",
    "transfer_halo(E)",
    "increment_time",
    "synchronize_magnetic_fields",
    "finite_value_check",
};

static void test_full_plan() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.13, 0.11));
  volume fv(vec(0.8, -1.0), vec(0.8, 1.0));
  f.add_dft_flux(Z, fv, 0.25, 0.35, 3);
  /* fields::fluxes is the *legacy* flux_vol list, which is what drives
     update_flux_half / update_flux. add_dft_flux does not populate it, so the
     full plan needs a flux plane as well -- a distinction worth having a test
     pin down. */
  f.add_flux_plane(vec(-0.8, -1.0), vec(-0.8, 1.0));
  f.advance(2);

  const StepPlan p = build_step_plan(f, StepProgram::ordinary);
  std::vector<std::string> got;
  format_step_plan(p, got);
  /* Only ranks that actually own a monitor chunk emit update_dft; see the note
     in build_step_plan about why that is deliberately not reduced. Compare the
     full sequence only where the full sequence is expected. */
  bool owns_dft = false;
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine() && f.chunks[i]->dft_chunks) owns_dft = true;
  if (owns_dft)
    compare("ordinary/full", got, expected_full,
            sizeof(expected_full) / sizeof(expected_full[0]));

  /* Guard kinds are load-bearing for Phase 2 even though the CPU executor
     treats them all the same. */
  size_t variants = 0, devices = 0;
  for (const Operation &op : p.operations) {
    if (op.kind == OpKind::restore_magnetic_fields ||
        op.kind == OpKind::synchronize_magnetic_fields) {
      CHECK(op.guard.kind == GuardKind::graph_variant,
            "%s should be a graph_variant guard", op_kind_name(op.kind));
      ++variants;
    }
    if (op.kind == OpKind::update_dft) {
      CHECK(op.guard.kind == GuardKind::device_predicate,
            "update_dft should be a device_predicate guard (the decimation check)");
      ++devices;
    }
  }
  CHECK(variants == 2, "expected 2 magnetic-sync variant guards, got %zu", variants);
  CHECK(devices == (owns_dft ? 1u : 0u), "expected %d decimation guards, got %zu",
        owns_dft ? 1 : 0, devices);

  /* Rebuilding without touching anything gives the same plan. */
  const StepPlan again = build_step_plan(f, StepProgram::ordinary);
  CHECK(again.signature == p.signature, "plan signature is not stable");
}

static void test_empty_plan() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, one, no_pml());
  fields f(&s);
  const StepPlan p = build_step_plan(f, StepProgram::ordinary);
  std::vector<std::string> got;
  format_step_plan(p, got);
  compare("ordinary/empty", got, expected_empty,
          sizeof(expected_empty) / sizeof(expected_empty[0]));
}

/* The CW program must exist and must differ from the ordinary one. A stale or
   missing CW plan is the one failure in this stack that produces wrong physics
   rather than a crash. */
static void test_solve_cw_plan() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5));
  fields f(&s);
  continuous_src_time src(0.3);
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  volume fv(vec(0.6, -0.8), vec(0.6, 0.8));
  f.add_dft_flux(Z, fv, 0.25, 0.35, 3);
  f.advance(2);

  const StepPlan ord = build_step_plan(f, StepProgram::ordinary);
  const StepPlan cw = build_step_plan(f, StepProgram::solve_cw);
  CHECK(cw.program == StepProgram::solve_cw, "CW plan is not marked as such");
  CHECK(cw.signature != ord.signature, "the CW plan is identical to the ordinary plan");

  bool cw_has_dft = false, cw_has_pack = false, cw_has_unpack = false;
  for (const Operation &op : cw.operations) {
    if (op.kind == OpKind::update_dft) cw_has_dft = true;
    if (op.kind == OpKind::pack_state) cw_has_pack = true;
    if (op.kind == OpKind::unpack_state) cw_has_unpack = true;
  }
  CHECK(!cw_has_dft, "update_dfts is disabled under solve_cw; the op must not be emitted");
  CHECK(cw_has_pack && cw_has_unpack, "the CW plan is missing its BiCGSTAB state packing");

  /* And the real thing runs. solve_cw drives fields::step() with
     doing_solve_cw set, so this exercises program selection end to end. */
  const bool ok = f.solve_cw(1e-6, 200, 2);
  CHECK(ok, "solve_cw did not converge");
}

/* Material phasing adds a segment_boundary-guarded reconciliation block whose
   condition is a collective or_to_all. */
static void test_phasing_plan() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, one, pml(0.5));
  structure s2(gv, eps_slab, pml(0.5));
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(2);
  f.phase_in_material(&s2, 1.0);

  const StepPlan p = build_step_plan(f, StepProgram::ordinary);
  size_t segments = 0, phase_ops = 0;
  for (const Operation &op : p.operations) {
    if (op.guard.kind == GuardKind::segment_boundary) ++segments;
    if (op.kind == OpKind::phase_material) ++phase_ops;
  }
  CHECK(phase_ops == 1, "expected one phase_material op, got %zu", phase_ops);
  CHECK(segments == 6, "expected 6 segment-guarded reconciliation ops, got %zu", segments);
  f.advance(3);
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  test_full_plan();
  test_empty_plan();
  test_solve_cw_plan();
  test_phasing_plan();

  if (failures) {
    master_printf("step_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("step_plan: all checks passed\n");
  return 0;
}
