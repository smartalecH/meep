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

#include <algorithm>
#include <exception>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include <meep.hpp>

#include "backend/lifecycle.hpp"
#include "backend/prepare.hpp"
#include "backend/random_state.hpp"
#include "backend/storage_plan.hpp"
#include "backend/step_plan.hpp"
#include "meep_internals.hpp"

using namespace meep;

static int failures = 0;

#define CHECK(cond, ...)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      /* printf, not master_printf: a failure on a non-master rank is exactly                      \
         the interesting kind, and master_printf would swallow it. */                              \
      printf("[rank %d] FAIL (%s:%d): ", my_rank(), __FILE__, __LINE__);                           \
      printf(__VA_ARGS__);                                                                         \
      printf("\n");                                                                                \
      fflush(stdout);                                                                              \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

static double one(const vec &) { return 1.0; }
static double eps_slab(const vec &p) { return (fabs(p.y()) < 0.4) ? 12.0 : 1.0; }
static double magnetic_conductivity(const vec &) { return 0.07; }

static void compare(const char *name, const std::vector<std::string> &got, const char *const *want,
                    size_t nwant) {
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

static const BufferAccess *find_access(const Operation &op, ArrayId id) {
  for (const BufferAccess &access : op.accesses)
    if (access.array.id == id) return &access;
  return NULL;
}

static ArrayId canonical_id(const fields &f, ArrayId id) {
  for (size_t depth = 0; is_valid(id) && depth <= f.array_catalog->size(); ++depth) {
    const ArrayId next = f.array_catalog->spec(id).alias_of;
    if (!is_valid(next)) return id;
    id = next;
  }
  return invalid_array();
}

static bool same_ref(const ArrayRef &a, const ArrayRef &b) {
  return a.id == b.id && a.offset == b.offset && a.elements == b.elements;
}

static bool same_access(const BufferAccess &a, const BufferAccess &b) {
  return same_ref(a.array, b.array) && a.mode == b.mode;
}

static std::vector<ptrdiff_t> expand_region(const UpdateRegion &region) {
  std::vector<ptrdiff_t> indices;
  for (size_t i0 = 0; i0 < region.counts[0]; ++i0)
    for (size_t i1 = 0; i1 < region.counts[1]; ++i1)
      for (size_t i2 = 0; i2 < region.counts[2]; ++i2)
        indices.push_back(ptrdiff_t(region.base) + ptrdiff_t(i0) * region.strides[0] +
                          ptrdiff_t(i1) * region.strides[1] + ptrdiff_t(i2) * region.strides[2]);
  return indices;
}

static AccessMode merge_access_mode(AccessMode a, AccessMode b) {
  return a == b ? a : AccessMode::read_write;
}

static void merge_expected_access(std::vector<BufferAccess> &expected, const BufferAccess &access) {
  if (!is_valid(access.array.id)) return;
  for (BufferAccess &entry : expected)
    if (entry.array.id == access.array.id) {
      entry.mode = merge_access_mode(entry.mode, access.mode);
      return;
    }
  expected.push_back(access);
}

static void check_exact_magnetic_access_union(const StepPlan &plan, const Operation &synchronize,
                                              const uint32_t schedule[7]) {
  std::vector<BufferAccess> expected;
  for (const MagneticStateArray &row : plan.magnetic_state_arrays)
    merge_expected_access(expected,
                          BufferAccess{ArrayRef{row.live, 0, row.elements},
                                       row.average ? AccessMode::read_write : AccessMode::read});
  for (int i = 0; i < 7; ++i) {
    if (schedule[i] == UINT32_MAX) continue;
    CHECK(schedule[i] < plan.operations.size(), "magnetic schedule slot %d is out of range", i);
    if (schedule[i] >= plan.operations.size()) continue;
    for (const BufferAccess &access : plan.operations[schedule[i]].accesses)
      merge_expected_access(expected, access);
  }

  CHECK(synchronize.accesses.size() == expected.size(),
        "magnetic synchronize has %zu accesses, expected exact union of %zu",
        synchronize.accesses.size(), expected.size());
  for (const BufferAccess &want : expected) {
    const BufferAccess *got = find_access(synchronize, want.array.id);
    CHECK(got && got->mode == want.mode && got->array.offset == want.array.offset &&
              got->array.elements == want.array.elements,
          "magnetic synchronize has wrong access for ArrayId %u", want.array.id.value);
  }
  for (const BufferAccess &got : synchronize.accesses) {
    const BufferAccess *want = NULL;
    for (const BufferAccess &entry : expected)
      if (entry.array.id == got.array.id) want = &entry;
    CHECK(want, "magnetic synchronize contains unrelated ArrayId %u", got.array.id.value);
  }
}

/* The ordinary timestep, transcribed from fields::step_once. Compare against
   the schedule in section 10.2 of the plan. */
static const char *const expected_full[] = {
    "restore_magnetic_fields",
    "update_material_coefficients",
    "evaluate_source_scalars", // time()                 -- B sources
    "update_db(B)",
    "apply_sources(B)",
    "transfer_halo(B)",
    "evaluate_source_scalars", // time() + 0.5*dt        -- integrated H
    "update_eh(H)",
    "transfer_halo(WH)",
    "update_polarization(H)",
    "transfer_halo(PH)",
    "transfer_halo(H)",
    "update_flux_half",
    "evaluate_source_scalars", // time() + 0.5*dt        -- D sources
    "update_db(D)",
    "apply_sources(D)",
    "transfer_halo(D)",
    "evaluate_source_scalars", // time() + dt            -- integrated E
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
  s.set_conductivity(Bx, magnetic_conductivity);
  const std::vector<double> scaled_k{0.17, -0.11, 0.07};
  fields f(&s, 0, 0, true, 0, 0, scaled_k);
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

  /* Force one exact H==B catalog alias while retaining a different split H
     component. This isolates the snapshot contract from update_eh's choice to
     split trivial H storage during the setup advance. */
  int controlled_chunk = -1;
  for (int chunk = 0; chunk < f.num_chunks && controlled_chunk < 0; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    const StorageKey required[] = {
        {chunk, int(array_kind::f), int(Bx), 0, 0},
        {chunk, int(array_kind::f_cond), int(Bx), 0, 0},
        {chunk, int(array_kind::f_bfast), int(Bx), 0, 0},
        {chunk, int(array_kind::f), int(By), 0, 0},
        {chunk, int(array_kind::f_u), int(By), 0, 0},
        {chunk, int(array_kind::f_bfast), int(By), 0, 0},
        {chunk, int(array_kind::f), int(Hx), 0, 0},
        {chunk, int(array_kind::f_w), int(Hx), 0, 0},
        {chunk, int(array_kind::f), int(Hy), 0, 0},
        {chunk, int(array_kind::f_w), int(Hy), 0, 0},
    };
    bool complete = true;
    for (const StorageKey &key : required)
      complete = complete && is_valid(f.array_catalog->find(key));
    if (complete) controlled_chunk = chunk;
  }
  CHECK(or_to_all(controlled_chunk >= 0),
        "magnetic fixture did not realize the controlled sparse-family pattern");
  if (controlled_chunk >= 0) {
    const ArrayId h = f.array_catalog->find({controlled_chunk, int(array_kind::f), int(Hx), 0, 0});
    const ArrayId b = f.array_catalog->find({controlled_chunk, int(array_kind::f), int(Bx), 0, 0});
    f.array_catalog->set_alias(h, b);
  }

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
    compare("ordinary/full", got, expected_full, sizeof(expected_full) / sizeof(expected_full[0]));

  /* Guard kinds are load-bearing for Phase 2 even though the CPU executor
     treats them all the same. */
  size_t variants = 0, devices = 0;
  const Operation *restore = NULL, *synchronize = NULL;
  for (const Operation &op : p.operations) {
    CHECK(op.beta_descriptor_index == 0 && op.beta_descriptor_count == 0,
          "%s has a nonempty beta span in a zero-beta plan", op_kind_name(op.kind));
    CHECK(op.polarization_subtraction_index == 0 && op.polarization_subtraction_count == 0,
          "%s has a nonempty PR6 polarization-subtraction span", op_kind_name(op.kind));
    if (op.kind == OpKind::restore_magnetic_fields ||
        op.kind == OpKind::synchronize_magnetic_fields) {
      CHECK(op.guard.kind == GuardKind::graph_variant, "%s should be a graph_variant guard",
            op_kind_name(op.kind));
      ++variants;
      if (op.kind == OpKind::restore_magnetic_fields)
        restore = &op;
      else
        synchronize = &op;
    }
    else
      CHECK(op.magnetic_state_index == 0 && op.magnetic_state_count == 0,
            "%s has a nonempty magnetic state span", op_kind_name(op.kind));
    if (op.kind == OpKind::update_flux_half || op.kind == OpKind::update_flux)
      CHECK(op.legacy_flux_index == 0 && op.legacy_flux_count == 1,
            "%s does not cover the one live legacy flux object", op_kind_name(op.kind));
    else
      CHECK(op.legacy_flux_index == 0 && op.legacy_flux_count == 0,
            "%s has a nonempty legacy flux span", op_kind_name(op.kind));
    if (op.kind == OpKind::update_dft) {
      CHECK(op.guard.kind == GuardKind::device_predicate,
            "update_dft should be a device_predicate guard (the decimation check)");
      ++devices;
    }
  }
  CHECK(variants == 2, "expected 2 magnetic-sync variant guards, got %zu", variants);
  CHECK(devices == (owns_dft ? 1u : 0u), "expected %d decimation guards, got %zu", owns_dft ? 1 : 0,
        devices);

  CHECK(restore && synchronize, "magnetic restore/synchronize markers are missing");
  if (restore && synchronize) {
    CHECK(restore->magnetic_state_index == 0 && synchronize->magnetic_state_index == 0,
          "magnetic state span does not begin at zero");
    CHECK(restore->magnetic_state_count == p.magnetic_state_arrays.size() &&
              synchronize->magnetic_state_count == p.magnetic_state_arrays.size(),
          "magnetic markers do not cover the complete state-array span");
  }

  for (size_t i = 0; i < p.magnetic_state_arrays.size(); ++i) {
    const MagneticStateArray &entry = p.magnetic_state_arrays[i];
    CHECK(entry.elements == f.array_catalog->spec(entry.live).elements,
          "magnetic state row %zu has the wrong extent", i);
    CHECK(entry.average == (entry.family == MagneticStateFamily::primary),
          "magnetic state row %zu has the wrong average flag", i);
    const StorageKey &key = f.array_catalog->key(entry.live);
    CHECK(key.chunk == entry.chunk && key.component_ == int(entry.c) && key.cmp == entry.cmp,
          "magnetic state row %zu identity does not match its ArrayId", i);
    MagneticStateFamily expected_family = MagneticStateFamily::primary;
    if (key.kind == int(array_kind::f_u)) expected_family = MagneticStateFamily::u;
    if (key.kind == int(array_kind::f_w)) expected_family = MagneticStateFamily::w;
    if (key.kind == int(array_kind::f_cond)) expected_family = MagneticStateFamily::conductivity;
    if (key.kind == int(array_kind::f_bfast)) expected_family = MagneticStateFamily::bfast;
    CHECK(entry.family == expected_family, "magnetic state row %zu has the wrong family", i);
    if (restore && synchronize) {
      const BufferAccess *restore_access = find_access(*restore, entry.live);
      const BufferAccess *sync_access = find_access(*synchronize, entry.live);
      CHECK(restore_access && restore_access->mode == AccessMode::write,
            "magnetic restore row %zu is not declared write-only", i);
      CHECK(sync_access && sync_access->mode != AccessMode::write,
            "magnetic synchronize row %zu is not readable", i);
      if (entry.average)
        CHECK(sync_access && sync_access->mode == AccessMode::read_write,
              "magnetic primary row %zu is not declared read-write", i);
    }
  }

  if (controlled_chunk >= 0) {
    struct ExpectedMagneticRow {
      component c;
      int cmp;
      MagneticStateFamily family;
      array_kind kind;
    };
    const ExpectedMagneticRow expected[] = {
        {Bx, 0, MagneticStateFamily::primary, array_kind::f},
        {Bx, 0, MagneticStateFamily::conductivity, array_kind::f_cond},
        {Bx, 0, MagneticStateFamily::bfast, array_kind::f_bfast},
        {Bx, 1, MagneticStateFamily::primary, array_kind::f},
        {Bx, 1, MagneticStateFamily::conductivity, array_kind::f_cond},
        {Bx, 1, MagneticStateFamily::bfast, array_kind::f_bfast},
        {By, 0, MagneticStateFamily::primary, array_kind::f},
        {By, 0, MagneticStateFamily::u, array_kind::f_u},
        {By, 0, MagneticStateFamily::bfast, array_kind::f_bfast},
        {By, 1, MagneticStateFamily::primary, array_kind::f},
        {By, 1, MagneticStateFamily::u, array_kind::f_u},
        {By, 1, MagneticStateFamily::bfast, array_kind::f_bfast},
        /* Hx/cmp0 is the forced H==B alias and contributes no row in any family. */
        {Hx, 1, MagneticStateFamily::primary, array_kind::f},
        {Hx, 1, MagneticStateFamily::w, array_kind::f_w},
        {Hy, 0, MagneticStateFamily::primary, array_kind::f},
        {Hy, 0, MagneticStateFamily::w, array_kind::f_w},
        {Hy, 1, MagneticStateFamily::primary, array_kind::f},
        {Hy, 1, MagneticStateFamily::w, array_kind::f_w},
    };
    std::vector<const MagneticStateArray *> actual;
    for (const MagneticStateArray &row : p.magnetic_state_arrays)
      if (row.chunk == controlled_chunk) actual.push_back(&row);
    CHECK(actual.size() == sizeof(expected) / sizeof(expected[0]),
          "controlled magnetic chunk has %zu rows, expected %zu", actual.size(),
          sizeof(expected) / sizeof(expected[0]));
    for (size_t i = 0; i < actual.size() && i < sizeof(expected) / sizeof(expected[0]); ++i) {
      const ExpectedMagneticRow &want = expected[i];
      const MagneticStateArray &got = *actual[i];
      const ArrayId expected_id =
          f.array_catalog->find({controlled_chunk, int(want.kind), int(want.c), want.cmp, 0});
      CHECK(got.c == want.c && got.cmp == want.cmp && got.family == want.family &&
                got.live == expected_id,
            "controlled magnetic row %zu has the wrong identity or order", i);
    }
    for (const MagneticStateArray &row : p.magnetic_state_arrays)
      CHECK(row.chunk != controlled_chunk || row.c != Hx || row.cmp != 0,
            "H==B alias emitted an Hx/cmp0 magnetic row");
  }

  const uint32_t schedule[] = {
      p.magnetic_half_step.evaluate_b_sources, p.magnetic_half_step.update_b,
      p.magnetic_half_step.apply_b_sources,    p.magnetic_half_step.transfer_b,
      p.magnetic_half_step.evaluate_h_sources, p.magnetic_half_step.update_h,
      p.magnetic_half_step.transfer_h};
  const OpKind schedule_kinds[] = {OpKind::evaluate_source_scalars,
                                   OpKind::update_db,
                                   OpKind::apply_sources,
                                   OpKind::transfer_halo,
                                   OpKind::evaluate_source_scalars,
                                   OpKind::update_eh,
                                   OpKind::transfer_halo};
  const field_type schedule_types[] = {field_type(NUM_FIELD_TYPES), B_stuff, B_stuff, B_stuff,
                                       field_type(NUM_FIELD_TYPES), H_stuff, H_stuff};
  for (size_t i = 0; i < sizeof(schedule) / sizeof(schedule[0]); ++i) {
    CHECK(schedule[i] < p.operations.size(), "magnetic half-step slot %zu is out of range", i);
    if (schedule[i] >= p.operations.size()) continue;
    const Operation &op = p.operations[schedule[i]];
    CHECK(op.kind == schedule_kinds[i] && op.ft == schedule_types[i],
          "magnetic half-step slot %zu has the wrong operation", i);
  }
  if (synchronize) check_exact_magnetic_access_union(p, *synchronize, schedule);

  bool promoted_u = false, promoted_w = false, promoted_cond = false, promoted_bfast = false;
  if (synchronize)
    for (const MagneticStateArray &row : p.magnetic_state_arrays) {
      if (row.average) continue;
      const BufferAccess *access = find_access(*synchronize, row.live);
      if (!access || access->mode != AccessMode::read_write) continue;
      promoted_u = promoted_u || row.family == MagneticStateFamily::u;
      promoted_w = promoted_w || row.family == MagneticStateFamily::w;
      promoted_cond = promoted_cond || row.family == MagneticStateFamily::conductivity;
      promoted_bfast = promoted_bfast || row.family == MagneticStateFamily::bfast;
    }
  CHECK(and_to_all(controlled_chunk < 0 ||
                   (promoted_u && promoted_w && promoted_cond && promoted_bfast)),
        "controlled magnetic fixture did not promote every auxiliary family to read-write");
  CHECK(p.operations[schedule[0]].source_time_offset == 0.0 &&
            p.operations[schedule[4]].source_time_offset == 0.5,
        "magnetic half-step source-time offsets are wrong");

  /* Rebuilding without touching anything gives the same plan. */
  const StepPlan again = build_step_plan(f, StepProgram::ordinary);
  CHECK(again.signature == p.signature, "plan signature is not stable");
}

static void test_magnetic_schema_signature() {
  StepPlan plan;
  Operation restore = {};
  restore.kind = OpKind::restore_magnetic_fields;
  restore.guard = guard_variant(0);
  restore.magnetic_state_index = 0;
  restore.magnetic_state_count = 1;
  plan.operations.push_back(restore);

  MagneticStateArray entry = {};
  entry.chunk = 2;
  entry.c = Hy;
  entry.cmp = 1;
  entry.family = MagneticStateFamily::w;
  entry.live = ArrayId{7};
  entry.elements = 257;
  entry.average = false;
  plan.magnetic_state_arrays.push_back(entry);
  MagneticStateArray second = entry;
  second.chunk = 3;
  second.c = Bx;
  second.cmp = 0;
  second.family = MagneticStateFamily::primary;
  second.live = ArrayId{9};
  second.elements = 511;
  second.average = true;
  plan.magnetic_state_arrays.push_back(second);
  plan.operations[0].magnetic_state_count = 2;
  plan.magnetic_half_step.evaluate_b_sources = 1;
  plan.magnetic_half_step.update_b = 2;
  plan.magnetic_half_step.apply_b_sources = 3;
  plan.magnetic_half_step.transfer_b = 4;
  plan.magnetic_half_step.evaluate_h_sources = 5;
  plan.magnetic_half_step.update_h = 6;
  plan.magnetic_half_step.transfer_h = 7;

  const uint64_t signature = compute_step_plan_signature(plan);
#define CHECK_MAGNETIC_SIGNATURE(expr, message)                                                    \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    expr;                                                                                          \
    CHECK(compute_step_plan_signature(changed) != signature, message);                             \
  } while (0)
  CHECK_MAGNETIC_SIGNATURE(++changed.operations[0].magnetic_state_index,
                           "signature ignored magnetic span start");
  CHECK_MAGNETIC_SIGNATURE(++changed.operations[0].magnetic_state_count,
                           "signature ignored magnetic span count");
  CHECK_MAGNETIC_SIGNATURE(++changed.magnetic_state_arrays[0].chunk,
                           "signature ignored magnetic chunk");
  CHECK_MAGNETIC_SIGNATURE(changed.magnetic_state_arrays[0].c = Hz,
                           "signature ignored magnetic component");
  CHECK_MAGNETIC_SIGNATURE(changed.magnetic_state_arrays[0].cmp = 0,
                           "signature ignored magnetic cmp");
  CHECK_MAGNETIC_SIGNATURE(changed.magnetic_state_arrays[0].family =
                               MagneticStateFamily::conductivity,
                           "signature ignored magnetic family");
  CHECK_MAGNETIC_SIGNATURE(++changed.magnetic_state_arrays[0].live.value,
                           "signature ignored magnetic ArrayId");
  CHECK_MAGNETIC_SIGNATURE(++changed.magnetic_state_arrays[0].elements,
                           "signature ignored magnetic extent");
  CHECK_MAGNETIC_SIGNATURE(changed.magnetic_state_arrays[0].average = true,
                           "signature ignored magnetic average flag");
  CHECK_MAGNETIC_SIGNATURE(
      std::swap(changed.magnetic_state_arrays[0], changed.magnetic_state_arrays[1]),
      "signature ignored magnetic row ordering");
  CHECK_MAGNETIC_SIGNATURE(++changed.magnetic_half_step.evaluate_b_sources,
                           "signature ignored magnetic B source schedule");
  CHECK_MAGNETIC_SIGNATURE(++changed.magnetic_half_step.update_b,
                           "signature ignored magnetic B schedule");
  CHECK_MAGNETIC_SIGNATURE(++changed.magnetic_half_step.apply_b_sources,
                           "signature ignored magnetic B source application");
  CHECK_MAGNETIC_SIGNATURE(++changed.magnetic_half_step.transfer_b,
                           "signature ignored magnetic B boundary schedule");
  CHECK_MAGNETIC_SIGNATURE(++changed.magnetic_half_step.evaluate_h_sources,
                           "signature ignored magnetic H source schedule");
  CHECK_MAGNETIC_SIGNATURE(++changed.magnetic_half_step.update_h,
                           "signature ignored magnetic H update schedule");
  CHECK_MAGNETIC_SIGNATURE(++changed.magnetic_half_step.transfer_h,
                           "signature ignored magnetic H schedule");
#undef CHECK_MAGNETIC_SIGNATURE

  plan.clear();
  CHECK(plan.operations.empty() && plan.magnetic_state_arrays.empty() &&
            plan.magnetic_half_step.evaluate_b_sources == UINT32_MAX &&
            plan.magnetic_half_step.update_b == UINT32_MAX &&
            plan.magnetic_half_step.apply_b_sources == UINT32_MAX &&
            plan.magnetic_half_step.transfer_b == UINT32_MAX &&
            plan.magnetic_half_step.evaluate_h_sources == UINT32_MAX &&
            plan.magnetic_half_step.update_h == UINT32_MAX &&
            plan.magnetic_half_step.transfer_h == UINT32_MAX,
        "StepPlan::clear retained magnetic state");
  Operation unrelated = {};
  unrelated.kind = OpKind::increment_time;
  plan.operations.push_back(unrelated);
  CHECK(plan.operations[0].magnetic_state_index == 0 &&
            plan.operations[0].magnetic_state_count == 0 && plan.magnetic_state_arrays.empty(),
        "plan rebuild inherited a cleared magnetic span");
}

static void test_empty_plan() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, one, no_pml());
  fields f(&s);
  f.require_component(Ez);
  f.advance(1);
  const StepPlan p = build_step_plan(f, StepProgram::ordinary);
  std::vector<std::string> got;
  format_step_plan(p, got);
  compare("ordinary/empty", got, expected_empty,
          sizeof(expected_empty) / sizeof(expected_empty[0]));
  const uint32_t schedule[] = {
      p.magnetic_half_step.evaluate_b_sources, p.magnetic_half_step.update_b,
      p.magnetic_half_step.apply_b_sources,    p.magnetic_half_step.transfer_b,
      p.magnetic_half_step.evaluate_h_sources, p.magnetic_half_step.update_h,
      p.magnetic_half_step.transfer_h};
  CHECK(schedule[0] == UINT32_MAX && schedule[4] == UINT32_MAX,
        "source-free magnetic schedule retained source evaluation nodes");
  const OpKind kinds[] = {OpKind::evaluate_source_scalars,
                          OpKind::update_db,
                          OpKind::apply_sources,
                          OpKind::transfer_halo,
                          OpKind::evaluate_source_scalars,
                          OpKind::update_eh,
                          OpKind::transfer_halo};
  const field_type types[] = {field_type(NUM_FIELD_TYPES), B_stuff, B_stuff, B_stuff,
                              field_type(NUM_FIELD_TYPES), H_stuff, H_stuff};
  uint32_t previous = 0;
  for (int i = 0; i < 7; ++i) {
    if (i == 0 || i == 4) continue;
    CHECK(schedule[i] < p.operations.size(), "source-free magnetic slot %d is out of range", i);
    if (schedule[i] >= p.operations.size()) continue;
    CHECK(p.operations[schedule[i]].kind == kinds[i] && p.operations[schedule[i]].ft == types[i],
          "source-free magnetic slot %d has the wrong operation", i);
    CHECK(!previous || previous < schedule[i], "source-free magnetic slots are out of order");
    previous = schedule[i];
  }
  const Operation *synchronize = NULL;
  for (const Operation &op : p.operations)
    if (op.kind == OpKind::synchronize_magnetic_fields) synchronize = &op;
  CHECK(synchronize, "source-free plan has no synchronize marker");
  if (synchronize) {
    CHECK(schedule[6] < size_t(synchronize - &p.operations[0]),
          "source-free synchronize marker precedes its half-step");
    check_exact_magnetic_access_union(p, *synchronize, schedule);
  }
}

static void test_cw_state_layout() {
  grid_volume gv = vol2d(3.1, 2.7, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  s.set_conductivity(Bx, magnetic_conductivity);
  const std::vector<double> scaled_k{0.17, -0.11, 0.07};
  fields f(&s, 0, 0, true, 0, 0, scaled_k);
  continuous_src_time src(0.3);
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(2);

  const StepPlan plan = build_step_plan(f, StepProgram::solve_cw);
  const CwStateLayout &layout = plan.cw_state_layout;
  const CwStateLayout rebuilt = build_cw_state_layout(f);
  CHECK(layout == rebuilt && rebuilt == layout, "independently rebuilt CW layouts differ");
  CHECK(layout.signature == compute_cw_state_layout_signature(layout),
        "stored CW layout signature differs from structural signature");
  std::string validation_error;
  CHECK(validate_cw_state_layout(f, layout, &validation_error),
        "canonical CW layout failed validation: %s", validation_error.c_str());

  struct ExpectedRow {
    int chunk;
    component traversal;
    component storage;
    CwStateFamily family;
    array_kind kind;
    ArrayId real_array;
    ArrayId imag_array;
  };
  std::vector<ExpectedRow> expected;
  bool family_seen[6] = {false, false, false, false, false, false};
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f.chunks[chunk];
    auto append = [&](component traversal, component storage, CwStateFamily family,
                      array_kind kind, realnum *real, realnum *imag) {
      CHECK((real != NULL) == (imag != NULL), "fixture contains a real/imaginary half-pair");
      if (!real || !imag) return;
      const ArrayId real_id = f.array_catalog->find({chunk, int(kind), int(storage), 0, 0});
      const ArrayId imag_id = f.array_catalog->find({chunk, int(kind), int(storage), 1, 0});
      CHECK(is_valid(real_id) && is_valid(imag_id), "fixture live pair is absent from catalog");
      expected.push_back(ExpectedRow{chunk, traversal, storage, family, kind, real_id, imag_id});
      family_seen[uint32_t(family)] = true;
    };
    FOR_COMPONENTS(c) {
      if (!is_D(c) && !is_B(c)) continue;
      append(c, c, CwStateFamily::primary, array_kind::f, fc.f[c][0], fc.f[c][1]);
      append(c, c, CwStateFamily::pml_u, array_kind::f_u, fc.f_u[c][0], fc.f_u[c][1]);
      append(c, c, CwStateFamily::conductivity, array_kind::f_cond, fc.f_cond[c][0],
             fc.f_cond[c][1]);
      append(c, c, CwStateFamily::bfast, array_kind::f_bfast, fc.f_bfast[c][0],
             fc.f_bfast[c][1]);
      const component paired = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
      append(c, paired, CwStateFamily::constitutive_w, array_kind::f_w, fc.f_w[paired][0],
             fc.f_w[paired][1]);
      if (fc.f_w[paired][0])
        append(c, paired, CwStateFamily::paired_primary, array_kind::f, fc.f[paired][0],
               fc.f[paired][1]);
    }
  }

  CHECK(layout.rows.size() == expected.size(), "CW layout has %zu rows, expected %zu",
        layout.rows.size(), expected.size());
  size_t offset = 0;
  std::vector<BufferAccess> expected_pack, expected_unpack;
  const size_t compared = std::min(layout.rows.size(), expected.size());
  for (size_t i = 0; i < compared; ++i) {
    const CwStateRow &row = layout.rows[i];
    const ExpectedRow &want = expected[i];
    CHECK(row.chunk == want.chunk && row.traversal_component == want.traversal &&
              row.storage_component == want.storage && row.family == want.family &&
              row.real_array == want.real_array && row.imag_array == want.imag_array,
          "CW row %zu has the wrong identity or family order", i);
    CHECK(f.chunks[row.chunk]->is_mine(), "CW row %zu names a non-owned chunk", i);
    CHECK(row.owned_region.chunk == row.chunk &&
              row.owned_region.c == row.traversal_component && row.owned_region.cmp == -1 &&
              row.owned_region.begin == f.chunks[row.chunk]->gv.little_owned_corner(want.traversal) &&
              row.owned_region.end == f.chunks[row.chunk]->gv.big_corner(),
          "CW row %zu does not use the traversal component's owned region", i);
    CHECK(row.complex_offset == offset, "CW row %zu has a gap or overlap", i);
    CHECK(row.complex_count == size_t(f.chunks[row.chunk]->gv.nowned(want.traversal)),
          "CW row %zu has the wrong complex count", i);
    offset += row.complex_count;

    std::vector<ptrdiff_t> from_macro;
    LOOP_OVER_VOL_OWNED(f.chunks[row.chunk]->gv, want.traversal, idx) {
      from_macro.push_back(idx);
    }
    CHECK(expand_region(row.owned_region) == from_macro,
          "CW row %zu does not flatten in LOOP_OVER_VOL_OWNED order", i);

    const ArrayId ids[] = {row.real_array, row.imag_array};
    for (ArrayId id : ids) {
      bool already = false;
      for (const BufferAccess &access : expected_pack)
        already = already || access.array.id == id;
      if (already) continue;
      const size_t elements = f.array_catalog->spec(id).elements;
      expected_pack.push_back(BufferAccess{ArrayRef{id, 0, elements}, AccessMode::read});
      expected_unpack.push_back(BufferAccess{ArrayRef{id, 0, elements}, AccessMode::write});
    }
  }
  CHECK(layout.complex_count == offset && layout.real_count == 2 * offset,
        "CW layout totals do not match the row prefix sum");
  CHECK(layout.vector_precision ==
            (sizeof(realnum) == sizeof(float) ? Precision::f32 : Precision::f64),
        "CW layout has the wrong vector precision");
  CHECK(layout.pack_accesses.size() == expected_pack.size() &&
            layout.unpack_accesses.size() == expected_unpack.size(),
        "CW pack/unpack access counts are wrong");
  for (size_t i = 0; i < expected_pack.size() && i < layout.pack_accesses.size(); ++i) {
    CHECK(same_access(layout.pack_accesses[i], expected_pack[i]),
          "CW pack access %zu is not the exact full-allocation read", i);
    CHECK(same_access(layout.unpack_accesses[i], expected_unpack[i]),
          "CW unpack access %zu is not the exact full-allocation write", i);
  }
  CHECK(layout.unpack_prelude.first_boundary == D_stuff &&
            layout.unpack_prelude.constitutive == E_stuff &&
            layout.unpack_prelude.second_boundary == E_stuff &&
            layout.unpack_prelude.skip_w_components &&
            layout.unpack_prelude.invalidate_field_values,
        "CW unpack prelude does not match array_to_fields");

  std::vector<ArrayId> expected_zero_ids;
  const array_kind zero_kinds[] = {
      array_kind::f,           array_kind::f_u,          array_kind::f_w,
      array_kind::f_cond,      array_kind::f_bfast,      array_kind::f_backup,
      array_kind::f_u_backup,  array_kind::f_w_backup,   array_kind::f_cond_backup,
      array_kind::f_bfast_backup};
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f.chunks[chunk];
    realnum *(*families[])[2] = {fc.f,          fc.f_u,          fc.f_w,          fc.f_cond,
                                 fc.f_bfast,    fc.f_backup,     fc.f_u_backup,   fc.f_w_backup,
                                 fc.f_cond_backup, fc.f_bfast_backup};
    for (size_t family = 0; family < sizeof(zero_kinds) / sizeof(zero_kinds[0]); ++family)
      FOR_COMPONENTS(c) for (int cmp = 0; cmp < 2; ++cmp) {
        if (!families[family][c][cmp]) continue;
        const ArrayId id =
            f.array_catalog->find({chunk, int(zero_kinds[family]), int(c), cmp, 0});
        CHECK(is_valid(id), "zero_fields live array is absent from catalog");
        if (is_valid(id)) expected_zero_ids.push_back(canonical_id(f, id));
      }
  }
  std::sort(expected_zero_ids.begin(), expected_zero_ids.end(),
            [](ArrayId a, ArrayId b) { return a.value < b.value; });
  expected_zero_ids.erase(
      std::unique(expected_zero_ids.begin(), expected_zero_ids.end(),
                  [](ArrayId a, ArrayId b) { return a == b; }),
      expected_zero_ids.end());
  CHECK(layout.zero_arrays.size() == expected_zero_ids.size(),
        "CW zero set has %zu arrays, expected %zu", layout.zero_arrays.size(),
        expected_zero_ids.size());
  for (size_t i = 0; i < layout.zero_arrays.size() && i < expected_zero_ids.size(); ++i) {
    const ArrayRef &ref = layout.zero_arrays[i];
    CHECK(ref.id == expected_zero_ids[i] && ref.offset == 0 &&
              ref.elements == f.array_catalog->spec(ref.id).elements,
          "CW zero-set entry %zu is not canonical and full-allocation", i);
    const int kind = f.array_catalog->key(ref.id).kind;
    CHECK(kind != int(array_kind::f_w_prev) && kind != int(array_kind::f_minus_p) &&
              kind != int(array_kind::f_rderiv_int),
          "CW zero set contains an excluded family");
  }

  const Operation *unpack = NULL, *pack = NULL;
  size_t unpack_index = plan.operations.size(), pack_index = plan.operations.size();
  std::vector<double> source_offsets;
  for (size_t i = 0; i < plan.operations.size(); ++i) {
    const Operation &op = plan.operations[i];
    CHECK(op.kind != OpKind::apply_sources, "CW schedule retained ordinary source application");
    CHECK(op.kind != OpKind::update_dft, "CW schedule retained an inner DFT update");
    if (op.kind == OpKind::evaluate_source_scalars) source_offsets.push_back(op.source_time_offset);
    if (op.kind == OpKind::unpack_state) {
      CHECK(!unpack, "CW schedule contains multiple unpack markers");
      unpack = &op;
      unpack_index = i;
    }
    if (op.kind == OpKind::pack_state) {
      CHECK(!pack, "CW schedule contains multiple pack markers");
      pack = &op;
      pack_index = i;
    }
  }
  CHECK(unpack && pack && unpack_index == 0 && pack_index + 1 == plan.operations.size(),
        "CW state markers do not bracket the complete timestep");
  if (unpack && pack) {
    CHECK(unpack->descriptor_index == 0 && unpack->descriptor_count == layout.rows.size() &&
              pack->descriptor_index == 0 && pack->descriptor_count == layout.rows.size(),
          "CW state markers do not cover the complete row span");
    CHECK(unpack->accesses.size() == layout.unpack_accesses.size() &&
              pack->accesses.size() == layout.pack_accesses.size(),
          "CW marker access counts do not match the layout");
    for (size_t i = 0; i < unpack->accesses.size(); ++i)
      CHECK(same_access(unpack->accesses[i], layout.unpack_accesses[i]),
            "unpack marker access %zu differs from the layout", i);
    for (size_t i = 0; i < pack->accesses.size(); ++i)
      CHECK(same_access(pack->accesses[i], layout.pack_accesses[i]),
            "pack marker access %zu differs from the layout", i);
  }
  const double expected_offsets[] = {0.0, 0.5, 0.5, 1.0};
  CHECK(source_offsets.size() == sizeof(expected_offsets) / sizeof(expected_offsets[0]),
        "CW schedule has %zu source evaluations, expected four", source_offsets.size());
  for (size_t i = 0; i < source_offsets.size() && i < 4; ++i)
    CHECK(source_offsets[i] == expected_offsets[i], "CW source evaluation %zu has wrong time", i);
  CHECK(plan.magnetic_half_step.apply_b_sources == UINT32_MAX,
        "CW magnetic half-step references a suppressed source application");

  const uint64_t plan_signature = plan.signature;
#define CHECK_CW_PLAN_MUTATION(mutation, message)                                                  \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    mutation;                                                                                      \
    CHECK(compute_step_plan_signature(changed) != plan_signature, message);                        \
  } while (0)
  CHECK_CW_PLAN_MUTATION(changed.operations.erase(changed.operations.begin()),
                         "signature ignored a missing unpack marker");
  CHECK_CW_PLAN_MUTATION(changed.operations.pop_back(),
                         "signature ignored a missing pack marker");
  CHECK_CW_PLAN_MUTATION(changed.operations[1].kind = OpKind::apply_sources,
                         "signature ignored an injected source application");
  CHECK_CW_PLAN_MUTATION(changed.operations[1].kind = OpKind::update_dft,
                         "signature ignored an injected DFT update");
  if (!source_offsets.empty()) {
    size_t source_index = 0;
    while (source_index < plan.operations.size() &&
           plan.operations[source_index].kind != OpKind::evaluate_source_scalars)
      ++source_index;
    if (source_index < plan.operations.size())
      CHECK_CW_PLAN_MUTATION(changed.operations.erase(changed.operations.begin() + source_index),
                             "signature ignored a missing source evaluation");
  }
#undef CHECK_CW_PLAN_MUTATION

  const bool owns_rows = !layout.rows.empty();
  bool owns_chunk = false;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    owns_chunk = owns_chunk || f.chunks[chunk]->is_mine();
  CHECK(or_to_all(owns_rows), "CW fixture produced no rows on any rank");
  if (!owns_chunk)
    CHECK(layout.rows.empty() && layout.zero_arrays.empty() && layout.pack_accesses.empty() &&
              layout.unpack_accesses.empty() && layout.complex_count == 0 && layout.real_count == 0,
          "idle rank retained local CW state");
  else
    CHECK(!layout.rows.empty() && !layout.zero_arrays.empty() && !layout.pack_accesses.empty() &&
              !layout.unpack_accesses.empty() && layout.complex_count && layout.real_count,
          "owner rank is missing CW state metadata");
  if (count_processors() > 2)
    CHECK(or_to_all(!owns_chunk), "np>2 CW fixture did not exercise a genuinely idle rank");
  for (size_t family = 0; family < 6; ++family)
    CHECK(or_to_all(family_seen[family]), "CW fixture did not exercise state family %zu", family);

  if (!layout.rows.empty()) {
    const CwStateLayout original = layout;
#define CHECK_CW_MUTATION(mutation, message)                                                       \
  do {                                                                                             \
    CwStateLayout changed = original;                                                              \
    mutation;                                                                                      \
    changed.signature = compute_cw_state_layout_signature(changed);                                \
    CHECK(changed != original, message " did not affect equality");                               \
    CHECK(changed.signature != original.signature, message " did not affect signature");          \
    std::string error;                                                                             \
    CHECK(!validate_cw_state_layout(f, changed, &error), message " passed canonical validation");  \
  } while (0)
    CHECK_CW_MUTATION(changed.rows.erase(changed.rows.begin()), "row deletion");
    CHECK_CW_MUTATION(changed.rows.insert(changed.rows.begin(), changed.rows[0]), "row insertion");
    if (original.rows.size() > 1)
      CHECK_CW_MUTATION(std::swap(changed.rows[0], changed.rows[1]), "row ordering");
    CHECK_CW_MUTATION(++changed.rows[0].chunk, "row chunk");
    CHECK_CW_MUTATION(changed.rows[0].traversal_component = Bx, "row traversal component");
    CHECK_CW_MUTATION(changed.rows[0].storage_component = Hx, "row storage component");
    CHECK_CW_MUTATION(changed.rows[0].family = CwStateFamily::bfast, "row family");
    CHECK_CW_MUTATION(++changed.rows[0].real_array.value, "row real ArrayId");
    CHECK_CW_MUTATION(++changed.rows[0].imag_array.value, "row imaginary ArrayId");
    CHECK_CW_MUTATION(changed.rows[0].real_array = invalid_array(), "invalid row ArrayId");
    CHECK_CW_MUTATION(changed.rows[0].imag_array = changed.rows[0].real_array,
                      "same-ID real/imaginary pair");
    CHECK_CW_MUTATION(changed.rows[0].owned_region.begin = ivec(1, 3, 5), "row region begin");
    CHECK_CW_MUTATION(changed.rows[0].owned_region.end = ivec(7, 9, 11), "row region end");
    CHECK_CW_MUTATION(++changed.rows[0].owned_region.base, "row region base");
    CHECK_CW_MUTATION(++changed.rows[0].owned_region.counts[0], "row region count");
    CHECK_CW_MUTATION(++changed.rows[0].owned_region.strides[0], "row region stride");
    CHECK_CW_MUTATION(++changed.rows[0].owned_region.variant_key, "row region variant");
    CHECK_CW_MUTATION(++changed.rows[0].complex_offset, "row vector offset");
    CHECK_CW_MUTATION(++changed.rows[0].complex_count, "row vector count");
    CHECK_CW_MUTATION(changed.zero_arrays.erase(changed.zero_arrays.begin()), "zero-set deletion");
    CHECK_CW_MUTATION(++changed.zero_arrays[0].id.value, "zero-set identity");
    CHECK_CW_MUTATION(changed.zero_arrays.push_back(changed.zero_arrays[0]),
                      "zero-set duplication");
    if (original.zero_arrays.size() > 1)
      CHECK_CW_MUTATION(std::swap(changed.zero_arrays[0], changed.zero_arrays[1]),
                        "zero-set ordering");
    CHECK_CW_MUTATION(++changed.zero_arrays[0].offset, "zero-set offset");
    CHECK_CW_MUTATION(--changed.zero_arrays[0].elements, "zero-set extent");
    CHECK_CW_MUTATION(changed.pack_accesses.erase(changed.pack_accesses.begin()),
                      "pack access deletion");
    CHECK_CW_MUTATION(++changed.pack_accesses[0].array.id.value, "pack access identity");
    CHECK_CW_MUTATION(changed.pack_accesses[0].mode = AccessMode::write, "pack access mode");
    CHECK_CW_MUTATION(++changed.pack_accesses[0].array.offset, "pack access offset");
    CHECK_CW_MUTATION(--changed.pack_accesses[0].array.elements, "pack access extent");
    if (original.pack_accesses.size() > 1)
      CHECK_CW_MUTATION(std::swap(changed.pack_accesses[0], changed.pack_accesses[1]),
                        "pack access ordering");
    CHECK_CW_MUTATION(++changed.unpack_accesses[0].array.id.value, "unpack access identity");
    CHECK_CW_MUTATION(changed.unpack_accesses[0].mode = AccessMode::read, "unpack access mode");
    CHECK_CW_MUTATION(++changed.unpack_accesses[0].array.offset, "unpack access offset");
    CHECK_CW_MUTATION(--changed.unpack_accesses[0].array.elements, "unpack access extent");
    if (original.unpack_accesses.size() > 1)
      CHECK_CW_MUTATION(std::swap(changed.unpack_accesses[0], changed.unpack_accesses[1]),
                        "unpack access ordering");
    CHECK_CW_MUTATION(changed.unpack_prelude.first_boundary = B_stuff,
                      "unpack prelude first boundary");
    CHECK_CW_MUTATION(changed.unpack_prelude.constitutive = H_stuff,
                      "unpack prelude constitutive update");
    CHECK_CW_MUTATION(changed.unpack_prelude.second_boundary = H_stuff,
                      "unpack prelude second boundary");
    CHECK_CW_MUTATION(changed.unpack_prelude.skip_w_components = false,
                      "unpack prelude skip-W flag");
    CHECK_CW_MUTATION(changed.unpack_prelude.invalidate_field_values = false,
                      "unpack prelude invalidation flag");
    CHECK_CW_MUTATION(++changed.complex_count, "complex total");
    CHECK_CW_MUTATION(++changed.real_count, "real total");
    CHECK_CW_MUTATION(changed.vector_precision = changed.vector_precision == Precision::f32
                                                     ? Precision::f64
                                                     : Precision::f32,
                      "vector precision");
    CHECK_CW_MUTATION(++changed.storage_fingerprint, "storage fingerprint");
    CHECK_CW_MUTATION(++changed.coordinate_fingerprint, "coordinate fingerprint");
    CHECK_CW_MUTATION(++changed.material_fingerprint, "material fingerprint");
#undef CHECK_CW_MUTATION

    StepPlan changed_plan = plan;
    ++changed_plan.cw_state_layout.rows[0].complex_count;
    changed_plan.cw_state_layout.signature =
        compute_cw_state_layout_signature(changed_plan.cw_state_layout);
    CHECK(compute_step_plan_signature(changed_plan) != plan.signature,
          "StepPlan signature ignored a re-signed CW layout mutation");

    CwStateLayout cleared = original;
    cleared.clear();
    const CwStateLayout empty;
    CHECK(cleared == empty, "CwStateLayout::clear retained state");
    StepPlan cleared_plan = plan;
    cleared_plan.clear();
    CHECK(cleared_plan.cw_state_layout == empty,
          "StepPlan::clear retained the CW state layout");

    CpuArrayCatalog saved = *f.array_catalog;
    CpuArrayCatalog broken;
    const ArrayId omitted = original.rows[0].real_array;
    for (size_t i = 0; i < saved.size(); ++i) {
      const ArrayId id{uint32_t(i)};
      if (id == omitted) continue;
      const ArraySpec &spec = saved.spec(id);
      broken.register_array(saved.key(id), saved.resolve_untyped(id), spec.elements, spec.role,
                            spec.element_type);
    }
    *f.array_catalog = broken;
    bool rejected_missing = false;
    try {
      build_cw_state_layout(f);
    }
    catch (const std::exception &) { rejected_missing = true; }
    CHECK(rejected_missing, "live state missing from the catalog was silently omitted");
    *f.array_catalog = saved;

    *f.array_catalog = broken;
    invalidate(f, MutationKind::field_layout);
    bool accepted_dirty_partial_catalog = true;
    CwStateLayout dirty_layout;
    try {
      dirty_layout = build_cw_state_layout(f);
    }
    catch (const std::exception &) { accepted_dirty_partial_catalog = false; }
    CHECK(accepted_dirty_partial_catalog,
          "dirty-storage CW layout rejected a transient partial catalog");
    if (accepted_dirty_partial_catalog)
      CHECK(dirty_layout.rows.size() < original.rows.size(),
            "dirty-storage CW layout did not omit the partial-catalog pair");
    *f.array_catalog = saved;
    clear_dirty(f, dirty_storage);

    auto rebuilt_catalog = [&](ArrayId rebound, void *replacement) {
      CpuArrayCatalog result;
      for (size_t i = 0; i < saved.size(); ++i) {
        const ArrayId id{uint32_t(i)};
        const ArraySpec &spec = saved.spec(id);
        void *address = id == rebound ? replacement : saved.resolve_untyped(id);
        result.register_array(saved.key(id), address, spec.elements, spec.role, spec.element_type);
      }
      for (size_t i = 0; i < saved.size(); ++i) {
        const ArrayId id{uint32_t(i)};
        if (is_valid(saved.spec(id).alias_of)) result.set_alias(id, saved.spec(id).alias_of);
      }
      return result;
    };
    realnum *const raw_real =
        static_cast<realnum *>(saved.resolve_untyped(original.rows[0].real_array));
    *f.array_catalog = rebuilt_catalog(original.rows[0].real_array, raw_real + 1);
    bool rejected_binding = false;
    try {
      build_cw_state_layout(f);
    }
    catch (const std::exception &) { rejected_binding = true; }
    CHECK(rejected_binding, "stale CW catalog binding was accepted");
    *f.array_catalog = saved;

    CpuArrayCatalog aliased = saved;
    aliased.set_alias(original.rows[0].real_array, original.rows[0].imag_array);
    *f.array_catalog = aliased;
    bool rejected_alias = false;
    try {
      build_cw_state_layout(f);
    }
    catch (const std::exception &) { rejected_alias = true; }
    CHECK(rejected_alias, "incompatible CW row alias was accepted");
    *f.array_catalog = saved;
  }
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
  CHECK(ord.cw_state_layout == CwStateLayout(),
        "ordinary plan unexpectedly retained CW state metadata");
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

static void test_lazy_cw_layout_before_storage_refresh() {
  {
    grid_volume direct_gv = vol2d(2.0, 2.0, 8.0);
    structure direct_s(direct_gv, one, no_pml());
    fields direct(&direct_s);
    continuous_src_time direct_source(0.23);
    direct_source.is_integrated = false;
    direct.add_point_source(Ez, direct_source, vec(0.11, 0.13));
    CHECK(direct.solve_cw(1e-6, 200, 2),
          "solve_cw before the first ordinary step did not converge");
  }

  grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  fields f(&s);
  f.step(); // freeze an empty catalog before adding the first field component

  continuous_src_time source(0.23);
  source.is_integrated = false;
  f.add_point_source(Ez, source, vec(0.11, 0.13));
  CHECK(is_dirty(f, dirty_storage), "late source did not invalidate prepared storage");

  size_t raw_pairs = 0;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    FOR_COMPONENTS(c) if ((is_D(c) || is_B(c)) && f.chunks[chunk]->f[c][0] &&
                          f.chunks[chunk]->f[c][1]) {
      ++raw_pairs;
    }
  }
  const CwStateLayout transient = build_cw_state_layout(f);
  CHECK(!raw_pairs || transient.rows.size() < raw_pairs,
        "dirty-storage CW layout did not omit uncataloged live rows");

  const bool ok = f.solve_cw(1e-6, 200, 2);
  CHECK(ok, "solve_cw after a late first source did not converge");
  const CwStateLayout prepared = build_cw_state_layout(f);
  std::string error;
  CHECK(validate_cw_state_layout(f, prepared, &error),
        "prepared CW layout after lazy omission is invalid: %s", error.c_str());
  CHECK(!raw_pairs || prepared.rows.size() >= raw_pairs,
        "prepared CW layout did not restore omitted live rows");
  CHECK(or_to_all(!prepared.rows.empty()),
        "late-source solve_cw fixture produced no prepared rows on any rank");
}

/* Material phasing adds a segment_boundary-guarded reconciliation block whose
   condition is a collective or_to_all. */
static void test_phasing_plan() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, one, pml(0.5));
  structure s2(gv, eps_slab, pml(0.5));
  s.set_conductivity(Dz, magnetic_conductivity);
  s2.set_conductivity(Dz, magnetic_conductivity);
  for (int i = 0; i < s.num_chunks; ++i) {
    if (!s.chunks[i]->is_mine()) continue;
    const size_t n = size_t(s.chunks[i]->gv.ntot());
    delete[] s.chunks[i]->chi1inv[Ex][Y];
    s.chunks[i]->chi1inv[Ex][Y] = new realnum[n];
    std::fill(s.chunks[i]->chi1inv[Ex][Y], s.chunks[i]->chi1inv[Ex][Y] + n, realnum(0));
    s.chunks[i]->trivial_chi1inv[Ex][Y] = false;
    delete[] s2.chunks[i]->chi1inv[Ex][Y];
    s2.chunks[i]->chi1inv[Ex][Y] = new realnum[n];
    std::fill(s2.chunks[i]->chi1inv[Ex][Y], s2.chunks[i]->chi1inv[Ex][Y] + n,
              realnum(0.125));
    s2.chunks[i]->trivial_chi1inv[Ex][Y] = false;
  }
  fields f(&s);
  std::unique_ptr<PreparedMaterialPhaseStorage> prepared =
      prepare_material_phase_storage(f, s2);
  prepared->commit();
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(2);
  f.phase_in_material(&s2, 1.0);

  const StepPlan p = build_step_plan(f, StepProgram::ordinary);
  size_t segments = 0, phase_ops = 0, coefficient_ops = 0;
  size_t phase_index = p.operations.size(), coefficient_index = p.operations.size();
  const Operation *phase = NULL, *coefficients = NULL;
  for (size_t i = 0; i < p.operations.size(); ++i) {
    const Operation &op = p.operations[i];
    if (op.guard.kind == GuardKind::segment_boundary) ++segments;
    if (op.kind == OpKind::phase_material) {
      ++phase_ops;
      phase = &op;
      phase_index = i;
    }
    if (op.kind == OpKind::update_material_coefficients) {
      ++coefficient_ops;
      coefficients = &op;
      coefficient_index = i;
    }
  }
  CHECK(phase_ops == 1, "expected one phase_material op, got %zu", phase_ops);
  CHECK(coefficient_ops == 1, "expected one material-coefficient op, got %zu", coefficient_ops);
  CHECK(segments == 6, "expected 6 segment-guarded reconciliation ops, got %zu", segments);
  CHECK(phase_index + 7 == coefficient_index,
        "material refresh boundaries do not enclose exactly six guarded reconciliation ops");
  CHECK(p.material_phase_target_signature == compute_material_phase_target_signature(f),
        "plan did not capture the live target structural fingerprint");

  std::vector<MaterialRefreshArray> expected;
  for (uint32_t family_value = uint32_t(MaterialRefreshFamily::chi1inv);
       family_value <= uint32_t(MaterialRefreshFamily::condinv); ++family_value) {
    const MaterialRefreshFamily family = MaterialRefreshFamily(family_value);
    for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
      if (!f.chunks[chunk]->is_mine()) continue;
      const structure_chunk &sc = *f.chunks[chunk]->s;
      FOR_COMPONENTS(c) for (int d = 0; d < 5; ++d) {
        if (family == MaterialRefreshFamily::condinv && d != int(component_direction(c)))
          continue;
        const realnum *row = family == MaterialRefreshFamily::chi1inv
                                 ? sc.chi1inv[c][d]
                                 : family == MaterialRefreshFamily::conductivity
                                       ? sc.conductivity[c][d]
                                       : sc.condinv[c][d];
        if (!row) continue;
        const array_kind kind = family == MaterialRefreshFamily::chi1inv
                                    ? array_kind::chi1inv
                                    : family == MaterialRefreshFamily::conductivity
                                          ? array_kind::conductivity
                                          : array_kind::condinv;
        const ArrayId id = f.array_catalog->find(StorageKey{chunk, int(kind), int(c), -1, d});
        CHECK(is_valid(id), "expected material row is missing from the current catalog");
        if (!is_valid(id)) continue;
        expected.push_back(MaterialRefreshArray{chunk, c, direction(d), family, id,
                                                f.array_catalog->spec(id).elements});
      }
    }
  }

  CHECK(phase && coefficients, "material refresh operations are absent");
  if (phase && coefficients) {
    size_t phase_rows = 0;
    while (phase_rows < expected.size() &&
           expected[phase_rows].family == MaterialRefreshFamily::chi1inv)
      ++phase_rows;
    CHECK(phase->material_refresh_index == 0 && phase->material_refresh_count == phase_rows,
          "phase refresh span is not the exact chi1inv prefix");
    CHECK(coefficients->material_refresh_index == phase_rows &&
              coefficients->material_refresh_count == expected.size() - phase_rows,
          "coefficient refresh span is not the exact conductivity/condinv suffix");
    CHECK(phase->accesses.size() == phase_rows,
          "phase refresh access count does not match its rows");
    CHECK(coefficients->accesses.size() == expected.size() - phase_rows,
          "coefficient refresh access count does not match its rows");
  }
  CHECK(p.material_refresh_arrays.size() == expected.size(),
        "material refresh row count is %zu, expected %zu", p.material_refresh_arrays.size(),
        expected.size());
  const size_t compared = std::min(p.material_refresh_arrays.size(), expected.size());
  for (size_t i = 0; i < compared; ++i) {
    const MaterialRefreshArray &got = p.material_refresh_arrays[i];
    const MaterialRefreshArray &want = expected[i];
    CHECK(got.chunk == want.chunk && got.c == want.c && got.d == want.d &&
              got.family == want.family && got.current == want.current &&
              got.elements == want.elements,
          "material refresh row %zu is not canonical", i);
    const Operation *owner = got.family == MaterialRefreshFamily::chi1inv ? phase : coefficients;
    const BufferAccess *access = owner ? find_access(*owner, got.current) : NULL;
    CHECK(access && access->mode == AccessMode::write && access->array.offset == 0 &&
              access->array.elements == got.elements,
          "material refresh row %zu lacks its exact device-write access", i);
    const ArraySpec &spec = f.array_catalog->spec(got.current);
    CHECK(!is_valid(spec.alias_of), "material refresh row %zu names an alias", i);
  }

  int controlled_chunk = -1;
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine() && f.chunks[i]->new_s) {
      controlled_chunk = i;
      break;
    }
  if (controlled_chunk >= 0) {
    structure_chunk &target = *f.chunks[controlled_chunk]->new_s;
    component numeric_c = NO_COMPONENT;
    direction numeric_d = NO_DIRECTION;
    FOR_COMPONENTS(c) for (int d = 0; d < 5; ++d)
      if (numeric_c == NO_COMPONENT && target.chi1inv[c][d]) {
        numeric_c = c;
        numeric_d = direction(d);
      }
    CHECK(numeric_c != NO_COMPONENT, "owned material target has no chi1inv row");
    if (numeric_c != NO_COMPONENT) {
      const realnum saved = target.chi1inv[numeric_c][numeric_d][0];
      target.chi1inv[numeric_c][numeric_d][0] = saved + realnum(0.125);
      CHECK(compute_material_phase_target_signature(f) == p.material_phase_target_signature,
            "target numerical values changed the structural fingerprint");
      target.chi1inv[numeric_c][numeric_d][0] = saved;
      target.trivial_chi1inv[numeric_c][numeric_d] =
          !target.trivial_chi1inv[numeric_c][numeric_d];
      CHECK(compute_material_phase_target_signature(f) != p.material_phase_target_signature,
            "target triviality did not change the structural fingerprint");
      target.trivial_chi1inv[numeric_c][numeric_d] =
          !target.trivial_chi1inv[numeric_c][numeric_d];
    }
  }
  f.advance(3);

  /* CPU-only phasing remains lazy.  In particular, a phase configured before
     the first step may create current rows after an empty/incomplete catalog
     was observed; refresh descriptors are resident upload metadata and must
     not make the host executor reject that legacy path. */
  structure lazy_current(gv, one, pml(0.5));
  structure lazy_target(gv, eps_slab, pml(0.5));
  lazy_target.set_conductivity(Dz, magnetic_conductivity);
  fields lazy(&lazy_current);
  lazy.require_component(Ez);
  CHECK(lazy.phase_in_material(&lazy_target, 3.0 * lazy.dt) == 3,
        "lazy CPU material phase setup returned the wrong countdown");
  lazy.advance(4);
  CHECK(!lazy.is_phasing() && lazy.t == 4,
        "lazy CPU material phase did not advance through and beyond its countdown");
}

static void test_material_schema_signature() {
  StepPlan plan;
  plan.material_phase_target_signature = 0x123456789abcdef0ull;
  Operation phase = {};
  phase.kind = OpKind::phase_material;
  phase.guard = guard_static(true);
  phase.material_refresh_index = 0;
  phase.material_refresh_count = 1;
  phase.accesses.push_back(BufferAccess{ArrayRef{ArrayId{7}, 0, 257}, AccessMode::write});
  plan.operations.push_back(phase);
  Operation coefficients = {};
  coefficients.kind = OpKind::update_material_coefficients;
  coefficients.guard = guard_always();
  coefficients.material_refresh_index = 1;
  coefficients.material_refresh_count = 2;
  coefficients.accesses.push_back(
      BufferAccess{ArrayRef{ArrayId{9}, 0, 257}, AccessMode::write});
  coefficients.accesses.push_back(
      BufferAccess{ArrayRef{ArrayId{11}, 0, 257}, AccessMode::write});
  plan.operations.push_back(coefficients);
  plan.material_refresh_arrays.push_back(
      MaterialRefreshArray{2, Ez, Z, MaterialRefreshFamily::chi1inv, ArrayId{7}, 257});
  plan.material_refresh_arrays.push_back(
      MaterialRefreshArray{2, Dz, Z, MaterialRefreshFamily::conductivity, ArrayId{9}, 257});
  plan.material_refresh_arrays.push_back(
      MaterialRefreshArray{2, Dz, Z, MaterialRefreshFamily::condinv, ArrayId{11}, 257});
  const uint64_t signature = compute_step_plan_signature(plan);
#define CHECK_MATERIAL_SIGNATURE(mutation, message)                                                \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    mutation;                                                                                      \
    CHECK(compute_step_plan_signature(changed) != signature, message);                             \
  } while (0)
  CHECK_MATERIAL_SIGNATURE(++changed.operations[0].material_refresh_index,
                           "signature ignored material refresh span start");
  CHECK_MATERIAL_SIGNATURE(++changed.operations[1].material_refresh_count,
                           "signature ignored material refresh span count");
  CHECK_MATERIAL_SIGNATURE(++changed.material_refresh_arrays[0].chunk,
                           "signature ignored material refresh chunk");
  CHECK_MATERIAL_SIGNATURE(changed.material_refresh_arrays[0].c = Ex,
                           "signature ignored material refresh component");
  CHECK_MATERIAL_SIGNATURE(changed.material_refresh_arrays[0].d = X,
                           "signature ignored material refresh direction");
  CHECK_MATERIAL_SIGNATURE(changed.material_refresh_arrays[0].family =
                               MaterialRefreshFamily::conductivity,
                           "signature ignored material refresh family");
  CHECK_MATERIAL_SIGNATURE(++changed.material_refresh_arrays[0].current.value,
                           "signature ignored material refresh ArrayId");
  CHECK_MATERIAL_SIGNATURE(++changed.material_refresh_arrays[0].elements,
                           "signature ignored material refresh extent");
  CHECK_MATERIAL_SIGNATURE(std::swap(changed.material_refresh_arrays[0],
                                     changed.material_refresh_arrays[1]),
                           "signature ignored material refresh row order");
  CHECK_MATERIAL_SIGNATURE(++changed.operations[0].accesses[0].array.id.value,
                           "signature ignored material access identity");
  CHECK_MATERIAL_SIGNATURE(changed.operations[0].accesses[0].mode = AccessMode::read,
                           "signature ignored material access mode");
  CHECK_MATERIAL_SIGNATURE(std::swap(changed.operations[1].accesses[0],
                                     changed.operations[1].accesses[1]),
                           "signature ignored material access order");
  CHECK_MATERIAL_SIGNATURE(++changed.material_phase_target_signature,
                           "signature ignored material target fingerprint");
#undef CHECK_MATERIAL_SIGNATURE

  plan.clear();
  CHECK(plan.material_refresh_arrays.empty() && plan.material_phase_target_signature == 0,
        "StepPlan::clear retained material phase state");
  for (const Operation &op : plan.operations)
    CHECK(op.material_refresh_index == 0 && op.material_refresh_count == 0,
          "StepPlan::clear retained a material operation span");
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
  update.ft = E_stuff;
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
  update.noise_amplitude = 0.0;
  update.noise_algorithm_version = 0;
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
            plan.operations[0].beta_descriptor_count == 0 &&
            plan.operations[0].cylindrical_m_descriptor_index == 0 &&
            plan.operations[0].cylindrical_m_descriptor_count == 0 &&
            plan.operations[0].cylindrical_origin_action_index == 0 &&
            plan.operations[0].cylindrical_origin_action_count == 0,
        "new operation spans are not zero-initialized");

#define CHECK_SIGNATURE_FIELD(expr, message)                                                       \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    expr;                                                                                          \
    CHECK(compute_step_plan_signature(changed) != signature, message);                             \
  } while (0)
  CHECK_SIGNATURE_FIELD(++changed.operations[0].polarization_subtraction_count,
                        "signature ignored polarization subtraction span");
  CHECK_SIGNATURE_FIELD(changed.polarization_updates[0].region.begin.set_direction(
                            X, changed.polarization_updates[0].region.begin.in_direction(X) + 2),
                        "signature ignored polarization region begin");
  CHECK_SIGNATURE_FIELD(changed.polarization_updates[0].ft = H_stuff,
                        "signature ignored polarization field family");
  CHECK_SIGNATURE_FIELD(++changed.polarization_updates[0].state_index,
                        "signature ignored polarization state ordinal");
  CHECK_SIGNATURE_FIELD(++changed.polarization_updates[0].p.value,
                        "signature ignored polarization ArrayId");
  CHECK_SIGNATURE_FIELD(changed.polarization_updates[0].kind = PolarizationUpdateKind::lorentzian,
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
  CHECK_SIGNATURE_FIELD(changed.polarization_updates[0].noise_amplitude = 0.125,
                        "signature ignored noise amplitude");
  CHECK_SIGNATURE_FIELD(changed.polarization_updates[0].noise_algorithm_version = 1,
                        "signature ignored noise algorithm version");
  CHECK_SIGNATURE_FIELD(++changed.polarization_subtractions[0].elements,
                        "signature ignored polarization subtraction size");
#undef CHECK_SIGNATURE_FIELD

  const PolarizationUpdate copy = update;
  CHECK(copy == update, "identical polarization updates compare unequal");
  PolarizationUpdate changed_update = update;
  changed_update.ft = H_stuff;
  CHECK(changed_update != update, "polarization field family did not affect equality");
  changed_update = update;
  changed_update.noise_amplitude = 0.125;
  CHECK(changed_update != update, "noise amplitude did not affect equality");
  changed_update = update;
  changed_update.noise_algorithm_version = 1;
  CHECK(changed_update != update, "noise algorithm version did not affect equality");

  plan.clear();
  CHECK(plan.polarization_updates.empty() && plan.polarization_subtractions.empty(),
        "StepPlan::clear retained polarization state");
  plan.clear();
  CHECK(plan.polarization_updates.empty() && plan.polarization_subtractions.empty(),
        "StepPlan::clear is not idempotent for polarization state");
}

static void test_noisy_polarization_group_schedule() {
  grid_volume gv = vol2d(3.0, 3.0, 8.0);
  structure s(gv, one, no_pml(), identity(), 2);
  fields f(&s);
  FOR_COMPONENTS(c)
  if (gv.has_field(c)) f.require_component(c);
  gaussian_src_time source(0.2, 0.1);
  f.add_point_source(Ez, source, vec(0.0, 0.0));
  f.advance(1);

  bool owns_chunk = false;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    owns_chunk = owns_chunk || f.chunks[chunk]->is_mine();
  std::vector<ArrayId> ids;
  if (f.array_catalog)
    for (size_t i = 0; i < f.array_catalog->size() && ids.size() < 6; ++i)
      if (!is_valid(f.array_catalog->spec(ArrayId{uint32_t(i)}).alias_of))
        ids.push_back(ArrayId{uint32_t(i)});

  bool exercised = false;
  if (ids.size() >= 6) {
    exercised = true;
    auto blank = [&](int state, component c, int cmp) {
      PolarizationUpdate update = {};
      update.kind = PolarizationUpdateKind::lorentzian;
      update.region.chunk = 0;
      update.region.c = c;
      update.region.cmp = cmp;
      update.region.begin = ivec(0, 0, 0);
      update.region.end = ivec(2, 2, 0);
      update.region.base = 0;
      update.region.counts[0] = update.region.counts[1] = 2;
      update.region.counts[2] = 1;
      update.region.strides[0] = 1;
      update.region.strides[1] = 4;
      update.region.strides[2] = 0;
      update.region.variant_key = 0;
      update.ft = is_electric(c) ? E_stuff : H_stuff;
      update.state_index = state;
      update.p = update.p_prev = invalid_array();
      update.p_cross1 = update.p_prev_cross1 = invalid_array();
      update.p_cross2 = update.p_prev_cross2 = invalid_array();
      update.primary_w = update.cross_w1 = update.cross_w2 = invalid_array();
      update.diagonal_sigma = update.offdiagonal_sigma1 = update.offdiagonal_sigma2 =
          invalid_array();
      update.primary_stride = update.cross_stride1 = update.cross_stride2 = 0;
      update.omega_0 = 0.31;
      update.gamma = 0.07;
      update.alpha = 0.0;
      memset(update.gyro_tensor, 0, sizeof(update.gyro_tensor));
      update.gyro_model = GYROTROPIC_LORENTZIAN;
      update.dt = 0.0125;
      update.noise_amplitude = 0.0;
      update.noise_algorithm_version = 0;
      return update;
    };
    auto recurrence = [&](int state, component c, int cmp, size_t base) {
      PolarizationUpdate update = blank(state, c, cmp);
      update.p = ids[base];
      update.p_prev = ids[base + 1];
      update.primary_w = ids[base + 2];
      update.diagonal_sigma = ids[base + 3];
      return update;
    };
    auto noise = [&](const PolarizationUpdate &recurrence_row, ArrayId p, ArrayId sigma) {
      PolarizationUpdate update = blank(recurrence_row.state_index, recurrence_row.region.c,
                                        recurrence_row.region.cmp);
      update.kind = PolarizationUpdateKind::noisy_add;
      update.p = p;
      update.diagonal_sigma = sigma;
      update.omega_0 = recurrence_row.omega_0;
      update.gamma = recurrence_row.gamma;
      update.dt = recurrence_row.dt;
      update.noise_amplitude = 0.125;
      update.noise_algorithm_version = counter_random_algorithm_version;
      return update;
    };

    StepPlan plan;
    Operation op = {};
    op.kind = OpKind::update_polarization;
    op.ft = E_stuff;
    op.guard = guard_always();
    op.descriptor_index = 0;
    op.descriptor_count = 0;
    append_polarization_update_group(f, plan, op, std::vector<PolarizationUpdate>(),
                                     std::vector<PolarizationUpdate>());
    CHECK(plan.polarization_updates.empty() && op.descriptor_count == 0 && op.accesses.empty(),
          "empty polarization group changed the plan");

    const PolarizationUpdate a0 = recurrence(0, Ex, 0, 0);
    const PolarizationUpdate a1 = recurrence(0, Ey, 0, 0);
    const PolarizationUpdate a_noise_only = blank(0, Ez, 0);
    const std::vector<PolarizationUpdate> a_recurrences{a0, a1};
    const std::vector<PolarizationUpdate> a_noise{
        noise(a0, a0.p, a0.diagonal_sigma), noise(a1, a1.p, a1.diagonal_sigma),
        noise(a_noise_only, ids[0], ids[3])};
    append_polarization_update_group(f, plan, op, a_recurrences, a_noise);

    const PolarizationUpdate deterministic = recurrence(1, Ez, 0, 0);
    append_polarization_update_group(f, plan, op, std::vector<PolarizationUpdate>{deterministic},
                                     std::vector<PolarizationUpdate>());

    const PolarizationUpdate b0 = recurrence(2, Ez, 1, 0);
    PolarizationUpdate b0_drude = b0;
    b0_drude.region.variant_key |= polarization_drude;
    const std::vector<PolarizationUpdate> b_noise{
        noise(b0_drude, b0_drude.p, b0_drude.diagonal_sigma)};
    append_polarization_update_group(f, plan, op,
                                     std::vector<PolarizationUpdate>{b0_drude}, b_noise);

    CHECK(op.descriptor_index == 0 && op.descriptor_count == plan.polarization_updates.size(),
          "noisy polarization operation span does not cover every grouped action");
    CHECK(plan.polarization_updates.size() == 8,
          "noisy polarization schedule has %zu actions, expected 8",
          plan.polarization_updates.size());
    const PolarizationUpdateKind expected_kinds[] = {
        PolarizationUpdateKind::lorentzian, PolarizationUpdateKind::lorentzian,
        PolarizationUpdateKind::noisy_add,  PolarizationUpdateKind::noisy_add,
        PolarizationUpdateKind::noisy_add,  PolarizationUpdateKind::lorentzian,
        PolarizationUpdateKind::lorentzian, PolarizationUpdateKind::noisy_add};
    const int expected_states[] = {0, 0, 0, 0, 0, 1, 2, 2};
    for (size_t i = 0; i < plan.polarization_updates.size(); ++i) {
      CHECK(plan.polarization_updates[i].kind == expected_kinds[i] &&
                plan.polarization_updates[i].state_index == expected_states[i],
            "noisy susceptibility group order differs at action %zu", i);
    }
    CHECK((plan.polarization_updates[6].region.variant_key & polarization_drude) != 0,
          "noisy schedule did not retain the Drude recurrence variant");
    CHECK(!is_valid(plan.polarization_updates[4].primary_w),
          "noise-only row unexpectedly requires a primary W array");

    std::vector<BufferAccess> expected_accesses;
    auto expect_access = [&](ArrayId id, AccessMode mode) {
      if (!is_valid(id)) return;
      merge_expected_access(expected_accesses,
                            BufferAccess{ArrayRef{id, 0, f.array_catalog->spec(id).elements}, mode});
    };
    for (const PolarizationUpdate &update : a_recurrences) {
      expect_access(update.p, AccessMode::read_write);
      expect_access(update.p_prev, AccessMode::read_write);
      expect_access(update.primary_w, AccessMode::read);
      expect_access(update.diagonal_sigma, AccessMode::read);
    }
    expect_access(deterministic.p, AccessMode::read_write);
    expect_access(deterministic.p_prev, AccessMode::read_write);
    expect_access(deterministic.primary_w, AccessMode::read);
    expect_access(deterministic.diagonal_sigma, AccessMode::read);
    expect_access(b0_drude.p, AccessMode::read_write);
    expect_access(b0_drude.p_prev, AccessMode::read_write);
    expect_access(b0_drude.primary_w, AccessMode::read);
    expect_access(b0_drude.diagonal_sigma, AccessMode::read);
    for (const PolarizationUpdate &update : a_noise) {
      expect_access(update.p, AccessMode::read_write);
      expect_access(update.diagonal_sigma, AccessMode::read);
    }
    for (const PolarizationUpdate &update : b_noise) {
      expect_access(update.p, AccessMode::read_write);
      expect_access(update.diagonal_sigma, AccessMode::read);
    }
    CHECK(op.accesses.size() == expected_accesses.size(),
          "noisy operation has %zu accesses, expected exact union of %zu", op.accesses.size(),
          expected_accesses.size());
    for (const BufferAccess &want : expected_accesses) {
      const BufferAccess *got = find_access(op, want.array.id);
      CHECK(got && same_access(*got, want), "noisy operation has an incorrect access for ArrayId %u",
            want.array.id.value);
    }

    plan.operations.push_back(op);
    plan.signature = compute_step_plan_signature(plan);
    StepPlan reordered = plan;
    std::swap(reordered.polarization_updates[1], reordered.polarization_updates[2]);
    CHECK(compute_step_plan_signature(reordered) != plan.signature,
          "signature ignored recurrence/noise group ordering");

    bool rejected = false;
    try {
      StepPlan malformed_plan;
      Operation malformed_op = op;
      malformed_op.accesses.clear();
      malformed_op.descriptor_index = malformed_op.descriptor_count = 0;
      PolarizationUpdate malformed_noise = a_noise[0];
      malformed_noise.p_prev = ids[4];
      append_polarization_update_group(f, malformed_plan, malformed_op,
                                       std::vector<PolarizationUpdate>(),
                                       std::vector<PolarizationUpdate>{malformed_noise});
    }
    catch (const std::invalid_argument &) { rejected = true; }
    CHECK(rejected, "noncanonical noisy action was accepted");

    StepPlan noise_only_plan;
    Operation noise_only_op = {};
    noise_only_op.kind = OpKind::update_polarization;
    noise_only_op.ft = E_stuff;
    noise_only_op.guard = guard_always();
    PolarizationUpdate noise_only_identity = blank(5, Ex, 0);
    const PolarizationUpdate isolated_noise = noise(noise_only_identity, ids[4], ids[5]);
    append_polarization_update_group(f, noise_only_plan, noise_only_op,
                                     std::vector<PolarizationUpdate>(),
                                     std::vector<PolarizationUpdate>{isolated_noise});
    CHECK(noise_only_op.accesses.size() == 2,
          "noise-only group has %zu accesses instead of exact P+sigma", noise_only_op.accesses.size());
    const BufferAccess *noise_p = find_access(noise_only_op, isolated_noise.p);
    const BufferAccess *noise_sigma = find_access(noise_only_op, isolated_noise.diagonal_sigma);
    CHECK(noise_p && noise_p->mode == AccessMode::read_write,
          "noise-only P access is not read-write");
    CHECK(noise_sigma && noise_sigma->mode == AccessMode::read,
          "noise-only diagonal sigma access is not read-only");

    auto expect_group_rejection = [&](const std::vector<PolarizationUpdate> &recurrences,
                                      const std::vector<PolarizationUpdate> &noise_rows,
                                      const char *message) {
      StepPlan malformed_plan;
      Operation malformed_op = {};
      malformed_op.kind = OpKind::update_polarization;
      malformed_op.ft = recurrences.empty() ? noise_rows.front().ft : recurrences.front().ft;
      malformed_op.guard = guard_always();
      bool group_rejected = false;
      try {
        append_polarization_update_group(f, malformed_plan, malformed_op, recurrences, noise_rows);
      }
      catch (const std::invalid_argument &) { group_rejected = true; }
      CHECK(group_rejected && malformed_plan.polarization_updates.empty() &&
                malformed_op.descriptor_count == 0 && malformed_op.accesses.empty(),
            "%s", message);
    };

    PolarizationUpdate mixed_kind = a1;
    mixed_kind.kind = PolarizationUpdateKind::gyrotropic;
    expect_group_rejection(std::vector<PolarizationUpdate>{a0, mixed_kind},
                           std::vector<PolarizationUpdate>(),
                           "mixed Lorentz/gyrotropic recurrence group was accepted");

    PolarizationUpdate gyrotropic = a0;
    gyrotropic.kind = PolarizationUpdateKind::gyrotropic;
    expect_group_rejection(std::vector<PolarizationUpdate>{gyrotropic},
                           std::vector<PolarizationUpdate>{a_noise[0]},
                           "gyrotropic recurrence accepted a noisy-add group");

    PolarizationUpdate mismatched_noise = a_noise[0];
    mismatched_noise.p = ids[4];
    expect_group_rejection(std::vector<PolarizationUpdate>{a0},
                           std::vector<PolarizationUpdate>{mismatched_noise},
                           "noise row with mismatched P was accepted");
    mismatched_noise = a_noise[0];
    mismatched_noise.diagonal_sigma = ids[5];
    expect_group_rejection(std::vector<PolarizationUpdate>{a0},
                           std::vector<PolarizationUpdate>{mismatched_noise},
                           "noise row with mismatched diagonal sigma was accepted");
    mismatched_noise = a_noise[0];
    ++mismatched_noise.region.base;
    expect_group_rejection(std::vector<PolarizationUpdate>{a0},
                           std::vector<PolarizationUpdate>{mismatched_noise},
                           "noise row with mismatched region was accepted");
    mismatched_noise = a_noise[0];
    mismatched_noise.gamma += 0.01;
    expect_group_rejection(std::vector<PolarizationUpdate>{a0},
                           std::vector<PolarizationUpdate>{mismatched_noise},
                           "noise row with mismatched recurrence coefficients was accepted");
    std::vector<PolarizationUpdate> inconsistent_noise = a_noise;
    inconsistent_noise[1].noise_amplitude += 0.25;
    expect_group_rejection(a_recurrences, inconsistent_noise,
                           "noise group with inconsistent coefficients was accepted");
    expect_group_rejection(a_recurrences,
                           std::vector<PolarizationUpdate>{a_noise.front()},
                           "noisy recurrence group with an omitted noise row was accepted");

    const size_t before_reappearance = plan.polarization_updates.size();
    rejected = false;
    try {
      append_polarization_update_group(f, plan, op, a_recurrences, a_noise);
    }
    catch (const std::invalid_argument &) { rejected = true; }
    CHECK(rejected && plan.polarization_updates.size() == before_reappearance,
          "noncontiguous repeated polarization identity was accepted");
  }
  CHECK(exercised || !owns_chunk,
        "an owning rank lacked enough catalog rows for noisy schedule coverage");
  CHECK(or_to_all(exercised), "no rank had enough catalog rows for noisy schedule coverage");
}

static void test_legacy_flux_schema_signature() {
  StepPlan plan;
  Operation op = {};
  op.kind = OpKind::update_flux_half;
  op.ft = field_type(NUM_FIELD_TYPES);
  op.guard = guard_static(true);
  op.legacy_flux_index = 0;
  op.legacy_flux_count = 1;
  op.accesses.push_back(BufferAccess{ArrayRef{ArrayId{11}, 0, 101}, AccessMode::read});
  plan.operations.push_back(op);

  LegacyFluxUpdate update = {3, 0, 1, 0x123456789abcdef0ull};
  plan.legacy_flux_updates.push_back(update);

  LegacyFluxTerm term = {};
  term.flux_ordinal = 3;
  term.term_ordinal = 1;
  term.region_ordinal = 2;
  term.sign = -1;
  term.chunk = 4;
  term.e_component = Ey;
  term.h_component = Hx;
  term.e_real = ArrayId{11};
  term.e_imag = ArrayId{12};
  term.h_real = ArrayId{13};
  term.h_imag = ArrayId{14};
  term.begin = ivec(1, 3, 5);
  term.end = ivec(7, 9, 11);
  term.lattice_shift = ivec(2, 0, -2);
  term.symmetry_index = 5;
  term.base = 17;
  term.counts[0] = 2;
  term.counts[1] = 3;
  term.counts[2] = 4;
  term.strides[0] = 1;
  term.strides[1] = 19;
  term.strides[2] = 43;
  term.e_offsets[0] = -2;
  term.e_offsets[1] = 7;
  term.h_offsets[0] = 3;
  term.h_offsets[1] = -11;
  term.phase_real = 0.25;
  term.phase_imag = -0.75;
  for (int axis = 0; axis < 3; ++axis)
    for (int edge = 0; edge < 4; ++edge)
      term.boundary_weights[axis][edge] = 0.125 * (1 + axis * 4 + edge);
  term.dV0 = 0.5;
  term.dV1 = 0.0625;
  plan.legacy_flux_terms.push_back(term);

  const LegacyFluxUpdate update_copy = update;
  const LegacyFluxTerm term_copy = term;
  CHECK(update == update_copy, "identical legacy flux updates compare unequal");
  CHECK(term == term_copy, "identical legacy flux terms compare unequal");
  LegacyFluxUpdate changed_update = update;
  ++changed_update.term_count;
  CHECK(!(update == changed_update), "different legacy flux updates compare equal");
  LegacyFluxTerm changed_term = term;
  ++changed_term.base;
  CHECK(!(term == changed_term), "different legacy flux terms compare equal");

  const uint64_t signature = compute_step_plan_signature(plan);
#define CHECK_FLUX_SIGNATURE(expr, message)                                                       \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    expr;                                                                                          \
    CHECK(compute_step_plan_signature(changed) != signature, message);                             \
  } while (0)
  CHECK_FLUX_SIGNATURE(++changed.operations[0].legacy_flux_index,
                       "signature ignored legacy flux span start");
  CHECK_FLUX_SIGNATURE(++changed.operations[0].legacy_flux_count,
                       "signature ignored legacy flux span count");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_updates[0].flux_ordinal,
                       "signature ignored legacy flux ordinal");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_updates[0].term_index,
                       "signature ignored legacy flux term span");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_updates[0].recipe_signature,
                       "signature ignored legacy flux recipe identity");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].region_ordinal,
                       "signature ignored legacy flux region ordinal");
  CHECK_FLUX_SIGNATURE(changed.legacy_flux_terms[0].sign = 1,
                       "signature ignored legacy flux sign");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].chunk,
                       "signature ignored legacy flux chunk");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].e_real.value,
                       "signature ignored legacy flux field identity");
  CHECK_FLUX_SIGNATURE(changed.legacy_flux_terms[0].begin.set_direction(
                           X, changed.legacy_flux_terms[0].begin.in_direction(X) + 2),
                       "signature ignored legacy flux region extent");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].base,
                       "signature ignored legacy flux base");
  CHECK_FLUX_SIGNATURE(--changed.legacy_flux_terms[0].e_offsets[0],
                       "signature ignored legacy flux interpolation offset");
  CHECK_FLUX_SIGNATURE(changed.legacy_flux_terms[0].phase_imag += 0.25,
                       "signature ignored legacy flux phase");
  CHECK_FLUX_SIGNATURE(changed.legacy_flux_terms[0].boundary_weights[1][2] += 0.25,
                       "signature ignored legacy flux boundary weight");
  CHECK_FLUX_SIGNATURE(changed.legacy_flux_terms[0].dV1 += 0.25,
                       "signature ignored legacy flux radial weight");
  CHECK_FLUX_SIGNATURE(++changed.operations[0].accesses[0].array.id.value,
                       "signature ignored legacy flux access");
#undef CHECK_FLUX_SIGNATURE

  plan.clear();
  CHECK(plan.legacy_flux_updates.empty() && plan.legacy_flux_terms.empty(),
        "StepPlan::clear retained legacy flux state");
}

static void test_live_legacy_flux_spans() {
  grid_volume gv = vol2d(3.0, 3.0, 8.0);
  structure s(gv, one, no_pml());
  fields f(&s);
  f.add_flux_plane(volume(vec(0.0, -1.0), vec(0.0, 1.0)));
  StepPlan one = build_step_plan(f, StepProgram::ordinary);
  CHECK(one.legacy_flux_updates.size() == 1,
        "one live legacy flux produced %zu update rows", one.legacy_flux_updates.size());
  const LegacyFluxUpdate expected_one = {0, 0, 0, 0};
  if (!one.legacy_flux_updates.empty())
    CHECK(one.legacy_flux_updates[0] == expected_one,
          "one live legacy flux has the wrong empty PR5 recipe span");

  f.add_flux_plane(volume(vec(0.5, -1.0), vec(0.5, 1.0)));
  StepPlan two = build_step_plan(f, StepProgram::ordinary);
  CHECK(two.legacy_flux_updates.size() == 2,
        "two live legacy fluxes produced %zu update rows", two.legacy_flux_updates.size());
  for (size_t i = 0; i < two.legacy_flux_updates.size(); ++i) {
    const LegacyFluxUpdate expected = {uint32_t(i), 0, 0, 0};
    CHECK(two.legacy_flux_updates[i] == expected,
          "legacy flux update %zu has the wrong list ordinal or PR5 recipe span", i);
  }

  size_t markers = 0;
  for (const Operation &op : two.operations)
    if (op.kind == OpKind::update_flux_half || op.kind == OpKind::update_flux) {
      ++markers;
      CHECK(op.legacy_flux_index == 0 && op.legacy_flux_count == 2,
            "%s does not cover both live legacy flux updates", op_kind_name(op.kind));
      /* PR5 owns the action vocabulary and live list ordinals. PR6 owns the
         region recipes and is therefore the first layer that can attach the
         exact field-read union to both markers. */
      CHECK(op.accesses.empty(), "%s populated accesses before PR6 flux recipes",
            op_kind_name(op.kind));
    }
  CHECK(markers == 2, "two live legacy fluxes produced %zu flux markers", markers);
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
    expr;                                                                                          \
    CHECK(compute_step_plan_signature(changed) != signature, message);                             \
  } while (0)
  CHECK_BETA_SIGNATURE(++changed.operations[0].beta_descriptor_index,
                       "signature ignored beta descriptor index");
  CHECK_BETA_SIGNATURE(changed.beta = -0.17, "signature ignored plan beta");
  CHECK_BETA_SIGNATURE(++changed.operations[0].beta_descriptor_count,
                       "signature ignored beta descriptor count");
  CHECK_BETA_SIGNATURE(++changed.beta_updates[0].region.variant_key,
                       "signature ignored beta variant");
  CHECK_BETA_SIGNATURE(++changed.beta_updates[0].source.value, "signature ignored beta source");
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

static void test_bfast_schema_signature() {
  StepPlan plan;
  plan.bfast_scaled_k = {0.17, -0.11, 0.07};
  CurlUpdate curl = {};
  curl.bfast_update_index = 0;
  plan.db_updates.push_back(curl);

  BfastUpdate update = {};
  update.region.chunk = 2;
  update.region.c = By;
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
  update.region.variant_key = bfast_has_pml | bfast_has_pml_aux | bfast_has_conductivity;
  update.target = ArrayId{1};
  update.source1 = ArrayId{2};
  update.source2 = ArrayId{3};
  update.stride1 = -7;
  update.stride2 = 11;
  update.f_bfast = ArrayId{4};
  update.target_u = ArrayId{5};
  update.condinv = ArrayId{6};
  update.target_cond = ArrayId{7};
  update.pml.siginv = ArrayId{8};
  update.pml.base = 9;
  update.pml.strides[1] = 2;
  update.pml_u.siginv = ArrayId{10};
  update.pml_u.base = 11;
  update.pml_u.strides[0] = 2;
  update.k1 = -0.17;
  update.k2 = 0.11;
  plan.bfast_updates.push_back(update);

  const uint64_t signature = compute_step_plan_signature(plan);
#define CHECK_BFAST_SIGNATURE(expr, message)                                                       \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    expr;                                                                                          \
    CHECK(compute_step_plan_signature(changed) != signature, message);                             \
  } while (0)
  CHECK_BFAST_SIGNATURE(++changed.db_updates[0].bfast_update_index,
                        "signature ignored paired BFAST index");
  CHECK_BFAST_SIGNATURE(changed.bfast_scaled_k[1] = 0.11,
                        "signature ignored prepared BFAST coordinate");
  CHECK_BFAST_SIGNATURE(++changed.bfast_updates[0].region.variant_key,
                        "signature ignored BFAST variant");
  CHECK_BFAST_SIGNATURE(++changed.bfast_updates[0].source1.value, "signature ignored BFAST source");
  CHECK_BFAST_SIGNATURE(--changed.bfast_updates[0].stride2, "signature ignored BFAST stride");
  CHECK_BFAST_SIGNATURE(++changed.bfast_updates[0].f_bfast.value,
                        "signature ignored BFAST persistent state");
  CHECK_BFAST_SIGNATURE(++changed.bfast_updates[0].target_u.value,
                        "signature ignored BFAST auxiliary target");
  CHECK_BFAST_SIGNATURE(++changed.bfast_updates[0].condinv.value,
                        "signature ignored BFAST conductivity inverse");
  CHECK_BFAST_SIGNATURE(++changed.bfast_updates[0].pml.siginv.value,
                        "signature ignored BFAST primary PML profile");
  CHECK_BFAST_SIGNATURE(++changed.bfast_updates[0].pml_u.base,
                        "signature ignored BFAST auxiliary PML profile");
  CHECK_BFAST_SIGNATURE(changed.bfast_updates[0].k1 = 0.17, "signature ignored BFAST k1");
  CHECK_BFAST_SIGNATURE(changed.bfast_updates[0].k2 = -0.11, "signature ignored BFAST k2");
#undef CHECK_BFAST_SIGNATURE

  plan.clear();
  CHECK(plan.bfast_scaled_k.empty() && plan.bfast_updates.empty(),
        "StepPlan::clear retained BFAST coordinate state");
}

static void test_cylindrical_schema_signature() {
  StepPlan plan;
  plan.coordinate_generation = 7;
  plan.cylindrical_m = -1.0;
  plan.cylindrical_origin_r.push_back(0.0);
  plan.cylindrical_zero_near_origin.push_back(1);

  Operation op = {};
  op.kind = OpKind::update_db;
  op.ft = D_stuff;
  op.guard = guard_always();
  op.cylindrical_m_descriptor_index = 1;
  op.cylindrical_m_descriptor_count = 1;
  op.cylindrical_origin_action_index = 2;
  op.cylindrical_origin_action_count = 2;
  plan.operations.push_back(op);

  CurlUpdate curl = {};
  curl.radial_prefix_index = 0;
  plan.db_updates.push_back(curl);

  CylindricalRadialPrefix prefix = {};
  prefix.chunk = 1;
  prefix.target_component = Dz;
  prefix.source_component = Hp;
  prefix.cmp = 1;
  prefix.source = ArrayId{1};
  prefix.scratch = ArrayId{2};
  prefix.nr = 5;
  prefix.nz = 256;
  prefix.row_stride = 257;
  prefix.source_elements = prefix.scratch_elements = 1542;
  prefix.ir0 = 0.5;
  plan.cylindrical_radial_prefixes.push_back(prefix);

  CylindricalMOverRUpdate mr = {};
  mr.region.chunk = 1;
  mr.region.c = Dz;
  mr.region.cmp = 1;
  mr.region.variant_key =
      cylindrical_m_has_pml | cylindrical_m_has_pml_aux | cylindrical_m_has_conductivity;
  mr.target = ArrayId{3};
  mr.source = ArrayId{4};
  mr.target_u = ArrayId{9};
  mr.condinv = ArrayId{10};
  mr.target_cond = ArrayId{11};
  mr.pml.siginv = ArrayId{12};
  mr.pml.base = 3;
  mr.pml.strides[0] = 2;
  mr.pml_u.siginv = ArrayId{13};
  mr.pml_u.base = 5;
  mr.pml_u.strides[1] = 2;
  mr.numerator = -0.35;
  mr.raw_radial_start = 1;
  plan.cylindrical_m_updates.push_back(mr);

  CylindricalAxisUpdate axis = {};
  axis.kind = CylindricalAxisKind::abs_m1;
  axis.region.chunk = 1;
  axis.region.c = Dp;
  axis.region.cmp = 1;
  axis.region.variant_key =
      cylindrical_axis_has_pml | cylindrical_axis_has_pml_aux | cylindrical_axis_has_conductivity;
  axis.target = ArrayId{14};
  axis.source1 = ArrayId{15};
  axis.source2 = ArrayId{16};
  axis.source1_neighbor_offset = -1;
  axis.source2_offset = 257;
  axis.target_u = ArrayId{17};
  axis.conductivity = ArrayId{18};
  axis.condinv = ArrayId{19};
  axis.target_cond = ArrayId{20};
  axis.pml.sig = ArrayId{21};
  axis.pml.kap = ArrayId{22};
  axis.pml.siginv = ArrayId{23};
  axis.pml.base = 7;
  axis.pml.strides[0] = 2;
  axis.pml_u.sig = ArrayId{24};
  axis.pml_u.kap = ArrayId{25};
  axis.pml_u.siginv = ArrayId{26};
  axis.pml_u.base = 9;
  axis.pml_u.strides[1] = 2;
  axis.source2_multiplier = 2;
  axis.scale = 0.25;
  axis.dt = 0.125;
  plan.cylindrical_axis_updates.push_back(axis);

  SlabRef slab = {};
  slab.array = ArrayId{27};
  slab.base = 257;
  slab.counts[0] = 257;
  slab.counts[1] = slab.counts[2] = 1;
  slab.strides[0] = 1;
  plan.cylindrical_zero_slabs.push_back(slab);
  plan.cylindrical_origin_actions.push_back(
      CylindricalOriginAction{CylindricalOriginActionKind::axis_update, 0});
  plan.cylindrical_origin_actions.push_back(
      CylindricalOriginAction{CylindricalOriginActionKind::zero_slab, 0});

  ConstitutiveUpdate constitutive = {};
  constitutive.region.variant_key = constitutive_axis_override;
  plan.eh_updates.push_back(constitutive);

  const uint64_t signature = compute_step_plan_signature(plan);
  StepPlan next_generation = plan;
  ++next_generation.coordinate_generation;
  CHECK(compute_step_plan_signature(next_generation) == signature,
        "coordinate lifecycle generation changed the executable content signature");
#define CHECK_CYL_SIGNATURE(expr, message)                                                         \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    expr;                                                                                          \
    CHECK(compute_step_plan_signature(changed) != signature, message);                             \
  } while (0)
  CHECK_CYL_SIGNATURE(changed.cylindrical_m = 1.0, "signature ignored cylindrical m");
  CHECK_CYL_SIGNATURE(changed.cylindrical_origin_r[0] = 0.5,
                      "signature ignored cylindrical origin");
  CHECK_CYL_SIGNATURE(changed.cylindrical_zero_near_origin[0] = 0,
                      "signature ignored cylindrical origin policy");
  CHECK_CYL_SIGNATURE(++changed.operations[0].cylindrical_m_descriptor_index,
                      "signature ignored cylindrical m span start");
  CHECK_CYL_SIGNATURE(++changed.operations[0].cylindrical_m_descriptor_count,
                      "signature ignored cylindrical m span");
  CHECK_CYL_SIGNATURE(++changed.operations[0].cylindrical_origin_action_index,
                      "signature ignored cylindrical origin span start");
  CHECK_CYL_SIGNATURE(++changed.operations[0].cylindrical_origin_action_count,
                      "signature ignored cylindrical origin span");
  CHECK_CYL_SIGNATURE(++changed.db_updates[0].radial_prefix_index,
                      "signature ignored radial-prefix pairing");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_radial_prefixes[0].scratch.value,
                      "signature ignored radial scratch identity");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_radial_prefixes[0].chunk,
                      "signature ignored radial-prefix chunk");
  CHECK_CYL_SIGNATURE(changed.cylindrical_radial_prefixes[0].target_component = Dp,
                      "signature ignored radial-prefix target component");
  CHECK_CYL_SIGNATURE(changed.cylindrical_radial_prefixes[0].source_component = Hr,
                      "signature ignored radial-prefix source component");
  CHECK_CYL_SIGNATURE(changed.cylindrical_radial_prefixes[0].cmp = 0,
                      "signature ignored radial-prefix cmp");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_radial_prefixes[0].source.value,
                      "signature ignored radial-prefix source identity");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_radial_prefixes[0].nr,
                      "signature ignored radial-prefix nr");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_radial_prefixes[0].nz,
                      "signature ignored radial-prefix nz");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_radial_prefixes[0].row_stride,
                      "signature ignored radial-prefix row stride");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_radial_prefixes[0].source_elements,
                      "signature ignored radial-prefix source extent");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_radial_prefixes[0].scratch_elements,
                      "signature ignored radial-prefix scratch extent");
  CHECK_CYL_SIGNATURE(changed.cylindrical_radial_prefixes[0].ir0 = 1.5,
                      "signature ignored radial-prefix coefficient");
  CHECK_CYL_SIGNATURE(changed.cylindrical_m_updates[0].numerator = 0.35,
                      "signature ignored cylindrical m/r coefficient");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_m_updates[0].region.variant_key,
                      "signature ignored cylindrical m/r variant");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_m_updates[0].target.value,
                      "signature ignored cylindrical m/r target");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_m_updates[0].source.value,
                      "signature ignored cylindrical m/r source");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_m_updates[0].target_u.value,
                      "signature ignored cylindrical m/r auxiliary target");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_m_updates[0].condinv.value,
                      "signature ignored cylindrical m/r conductivity inverse");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_m_updates[0].target_cond.value,
                      "signature ignored cylindrical m/r conductivity target");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_m_updates[0].pml.base,
                      "signature ignored cylindrical m/r primary PML");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_m_updates[0].pml_u.siginv.value,
                      "signature ignored cylindrical m/r auxiliary PML");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_m_updates[0].raw_radial_start,
                      "signature ignored cylindrical radial coordinate");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_axis_updates[0].source2.value,
                      "signature ignored cylindrical axis source");
  CHECK_CYL_SIGNATURE(changed.cylindrical_axis_updates[0].kind = CylindricalAxisKind::m0_dz,
                      "signature ignored cylindrical axis kind");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_axis_updates[0].region.variant_key,
                      "signature ignored cylindrical axis variant");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_axis_updates[0].target.value,
                      "signature ignored cylindrical axis target");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_axis_updates[0].source1.value,
                      "signature ignored cylindrical axis first source");
  CHECK_CYL_SIGNATURE(--changed.cylindrical_axis_updates[0].source1_neighbor_offset,
                      "signature ignored cylindrical axis neighbor offset");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_axis_updates[0].source2_offset,
                      "signature ignored cylindrical axis second-source offset");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_axis_updates[0].target_u.value,
                      "signature ignored cylindrical axis auxiliary target");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_axis_updates[0].conductivity.value,
                      "signature ignored cylindrical axis conductivity");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_axis_updates[0].condinv.value,
                      "signature ignored cylindrical axis conductivity inverse");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_axis_updates[0].target_cond.value,
                      "signature ignored cylindrical axis conductivity target");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_axis_updates[0].pml.kap.value,
                      "signature ignored cylindrical axis primary PML");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_axis_updates[0].pml_u.base,
                      "signature ignored cylindrical axis auxiliary PML");
  CHECK_CYL_SIGNATURE(changed.cylindrical_axis_updates[0].scale = -0.25,
                      "signature ignored cylindrical axis scale");
  CHECK_CYL_SIGNATURE(changed.cylindrical_axis_updates[0].source2_multiplier = -2,
                      "signature ignored cylindrical axis multiplier");
  CHECK_CYL_SIGNATURE(changed.cylindrical_axis_updates[0].dt = 0.25,
                      "signature ignored cylindrical axis timestep");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_zero_slabs[0].array.value,
                      "signature ignored cylindrical zero-slab array");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_zero_slabs[0].base,
                      "signature ignored cylindrical zero slab");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_zero_slabs[0].counts[0],
                      "signature ignored cylindrical zero-slab count");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_zero_slabs[0].strides[0],
                      "signature ignored cylindrical zero-slab stride");
  CHECK_CYL_SIGNATURE(++changed.cylindrical_origin_actions[0].index,
                      "signature ignored cylindrical origin action");
  CHECK_CYL_SIGNATURE(changed.cylindrical_origin_actions[0].kind =
                          CylindricalOriginActionKind::zero_slab,
                      "signature ignored cylindrical origin action kind");
  CHECK_CYL_SIGNATURE(changed.eh_updates[0].region.variant_key = 0,
                      "signature ignored constitutive axis marker");
#undef CHECK_CYL_SIGNATURE

  plan.clear();
  CHECK(plan.coordinate_generation == 0 && plan.cylindrical_m == 0 &&
            plan.cylindrical_origin_r.empty() && plan.cylindrical_zero_near_origin.empty() &&
            plan.cylindrical_radial_prefixes.empty() && plan.cylindrical_m_updates.empty() &&
            plan.cylindrical_axis_updates.empty() && plan.cylindrical_zero_slabs.empty() &&
            plan.cylindrical_origin_actions.empty(),
        "StepPlan::clear retained cylindrical coordinate state");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  test_full_plan();
  test_empty_plan();
  test_cw_state_layout();
  test_solve_cw_plan();
  test_lazy_cw_layout_before_storage_refresh();
  test_phasing_plan();
  test_material_schema_signature();
  test_magnetic_schema_signature();
  test_polarization_schema_signature();
  test_noisy_polarization_group_schedule();
  test_legacy_flux_schema_signature();
  test_live_legacy_flux_spans();
  test_beta_schema_signature();
  test_bfast_schema_signature();
  test_cylindrical_schema_signature();

  if (failures) {
    master_printf("step_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("step_plan: all checks passed\n");
  return 0;
}
