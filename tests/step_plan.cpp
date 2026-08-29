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
    CHECK(op.beta_descriptor_index == 0 && op.beta_descriptor_count == 0,
          "%s has a nonempty beta span in a zero-beta plan", op_kind_name(op.kind));
    CHECK(op.polarization_subtraction_index == 0 && op.polarization_subtraction_count == 0,
          "%s has a nonempty PR6 polarization-subtraction span",
          op_kind_name(op.kind));
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

static void test_polarization_schema_signature() {
  StepPlan plan;
  Operation op = {};
  op.kind = OpKind::update_polarization;
  op.ft = E_stuff;
  op.guard = guard_always();
  op.beta_descriptor_index = 0;
  op.beta_descriptor_count = 0;
  op.polarization_subtraction_index = 0;
  op.polarization_subtraction_count = 0;
  plan.operations.push_back(op);

  PolarizationUpdate update = {};
  update.kind = PolarizationUpdateKind::gyrotropic;
  update.region.chunk = 2;
  update.region.c = Ez;
  update.region.cmp = 1;
  update.region.begin = ivec(1, 3, 5);
  update.region.end = ivec(7, 9, 11);
  update.region.base = 13;
  update.region.counts[0] = 2;
  update.region.counts[1] = 3;
  update.region.counts[2] = 4;
  update.region.strides[0] = 1;
  update.region.strides[1] = 17;
  update.region.strides[2] = 37;
  update.region.variant_key = polarization_one_offdiagonal | polarization_drude;
  update.state_index = 4;
  update.p = ArrayId{1};
  update.p_prev = ArrayId{2};
  update.p_cross1 = ArrayId{8};
  update.p_prev_cross1 = ArrayId{9};
  update.p_cross2 = ArrayId{10};
  update.p_prev_cross2 = ArrayId{11};
  update.primary_w = ArrayId{3};
  update.cross_w1 = ArrayId{4};
  update.cross_w2 = invalid_array();
  update.diagonal_sigma = ArrayId{5};
  update.offdiagonal_sigma1 = ArrayId{6};
  update.offdiagonal_sigma2 = invalid_array();
  update.primary_stride = -7;
  update.cross_stride1 = -11;
  update.cross_stride2 = 0;
  update.omega_0 = 0.25;
  update.gamma = 0.05;
  update.alpha = 0.07;
  update.gyro_tensor[0][1] = 0.11;
  update.gyro_tensor[1][0] = -0.11;
  update.gyro_model = GYROTROPIC_SATURATED;
  update.dt = 0.01;
  plan.polarization_updates.push_back(update);

  PolarizationSubtraction subtraction = {};
  subtraction.chunk = 2;
  subtraction.c = Ez;
  subtraction.cmp = 1;
  subtraction.state_index = 4;
  subtraction.target = ArrayId{7};
  subtraction.p = update.p;
  subtraction.elements = 101;
  plan.polarization_subtractions.push_back(subtraction);

  const uint64_t signature = compute_step_plan_signature(plan);
  CHECK(plan.operations[0].polarization_subtraction_index == 0 &&
            plan.operations[0].polarization_subtraction_count == 0 &&
            plan.operations[0].beta_descriptor_index == 0 &&
            plan.operations[0].beta_descriptor_count == 0,
        "new operation spans are not zero-initialized");

#define CHECK_SIGNATURE_FIELD(expr, message)                                                       \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    expr;                                                                                           \
    CHECK(compute_step_plan_signature(changed) != signature, message);                              \
  } while (0)
  CHECK_SIGNATURE_FIELD(++changed.operations[0].polarization_subtraction_count,
                        "signature ignored polarization subtraction span");
  CHECK_SIGNATURE_FIELD(changed.polarization_updates[0].region.begin.set_direction(
                            X, changed.polarization_updates[0].region.begin.in_direction(X) + 2),
                        "signature ignored polarization region begin");
  CHECK_SIGNATURE_FIELD(++changed.polarization_updates[0].p.value,
                        "signature ignored polarization ArrayId");
  CHECK_SIGNATURE_FIELD(changed.polarization_updates[0].kind =
                            PolarizationUpdateKind::lorentzian,
                        "signature ignored polarization update kind");
  CHECK_SIGNATURE_FIELD(++changed.polarization_updates[0].p_cross1.value,
                        "signature ignored gyrotropic state ArrayId");
  CHECK_SIGNATURE_FIELD(--changed.polarization_updates[0].primary_stride,
                        "signature ignored polarization stride");
  CHECK_SIGNATURE_FIELD(changed.polarization_updates[0].gamma += 0.01,
                        "signature ignored polarization coefficient");
  CHECK_SIGNATURE_FIELD(changed.polarization_updates[0].alpha += 0.01,
                        "signature ignored gyrotropic alpha");
  CHECK_SIGNATURE_FIELD(changed.polarization_updates[0].gyro_tensor[2][1] += 0.01,
                        "signature ignored gyrotropic tensor");
  CHECK_SIGNATURE_FIELD(changed.polarization_updates[0].gyro_model = GYROTROPIC_DRUDE,
                        "signature ignored gyrotropic model");
  CHECK_SIGNATURE_FIELD(++changed.polarization_subtractions[0].elements,
                        "signature ignored polarization subtraction size");
#undef CHECK_SIGNATURE_FIELD
}

static void test_beta_schema_signature() {
  StepPlan plan;
  plan.beta = 0.17;
  Operation op = {};
  op.kind = OpKind::update_db;
  op.ft = D_stuff;
  op.guard = guard_always();
  op.beta_descriptor_index = 0;
  op.beta_descriptor_count = 1;
  plan.operations.push_back(op);

  BetaUpdate update = {};
  update.region.chunk = 3;
  update.region.c = Dx;
  update.region.cmp = 1;
  update.region.begin = ivec(1, 3, 0);
  update.region.end = ivec(7, 9, 0);
  update.region.base = 11;
  update.region.counts[0] = 4;
  update.region.counts[1] = 5;
  update.region.counts[2] = 1;
  update.region.strides[0] = 1;
  update.region.strides[1] = 17;
  update.region.strides[2] = 0;
  update.region.variant_key = beta_has_pml | beta_has_pml_aux | beta_has_conductivity;
  update.target = ArrayId{1};
  update.source = ArrayId{2};
  update.target_u = ArrayId{3};
  update.condinv = ArrayId{4};
  update.target_cond = ArrayId{5};
  update.pml.siginv = ArrayId{6};
  update.pml.base = 7;
  update.pml.strides[1] = 2;
  update.pml_u.siginv = ArrayId{8};
  update.pml_u.base = 9;
  update.pml_u.strides[0] = 2;
  update.betadt = -0.125;
  plan.beta_updates.push_back(update);

  const uint64_t signature = compute_step_plan_signature(plan);
#define CHECK_BETA_SIGNATURE(expr, message)                                                        \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    expr;                                                                                           \
    CHECK(compute_step_plan_signature(changed) != signature, message);                              \
  } while (0)
  CHECK_BETA_SIGNATURE(++changed.operations[0].beta_descriptor_index,
                       "signature ignored beta descriptor index");
  CHECK_BETA_SIGNATURE(changed.beta = -0.17, "signature ignored plan beta");
  CHECK_BETA_SIGNATURE(++changed.operations[0].beta_descriptor_count,
                       "signature ignored beta descriptor count");
  CHECK_BETA_SIGNATURE(++changed.beta_updates[0].region.variant_key,
                       "signature ignored beta variant");
  CHECK_BETA_SIGNATURE(++changed.beta_updates[0].source.value,
                       "signature ignored beta source");
  CHECK_BETA_SIGNATURE(++changed.beta_updates[0].target_u.value,
                       "signature ignored beta auxiliary target");
  CHECK_BETA_SIGNATURE(++changed.beta_updates[0].condinv.value,
                       "signature ignored beta conductivity inverse");
  CHECK_BETA_SIGNATURE(++changed.beta_updates[0].target_cond.value,
                       "signature ignored beta conductivity target");
  CHECK_BETA_SIGNATURE(++changed.beta_updates[0].pml.siginv.value,
                       "signature ignored beta primary PML profile");
  CHECK_BETA_SIGNATURE(++changed.beta_updates[0].pml_u.base,
                       "signature ignored beta auxiliary PML profile");
  CHECK_BETA_SIGNATURE(changed.beta_updates[0].betadt = 0.125,
                       "signature ignored beta coefficient");
#undef CHECK_BETA_SIGNATURE
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  test_full_plan();
  test_empty_plan();
  test_solve_cw_plan();
  test_phasing_plan();
  test_polarization_schema_signature();
  test_beta_schema_signature();

  if (failures) {
    master_printf("step_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("step_plan: all checks passed\n");
  return 0;
}
