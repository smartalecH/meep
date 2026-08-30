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
#include <map>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <tuple>
#include <vector>

#include <meep.hpp>

#include "backend/lifecycle.hpp"
#include "backend/prepare.hpp"
#include "backend/random_state.hpp"
#include "backend/storage_plan.hpp"
#include "backend/step_plan.hpp"
#include "meep_internals.hpp"

namespace meep {
/* Test-only friend: exercise the exact CPU dispatch switch without exposing a
   public execution API or adding a production callback hook. */
struct StepPlanTestAccess {
  static void execute(fields &f, const StepPlan &plan) { f.execute_step_plan(plan, 0); }
};
} // namespace meep

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

class multitile_anisotropic_material : public material_function {
public:
  void eff_chi1inv_row(component c, double row[3], const volume &, double, int) override {
    row[0] = row[1] = row[2] = 0.0;
    const int d = component_index(c);
    row[d] = 0.5 + 0.05 * d;
    static const double offdiagonal[3][3] = {
        {0.0, 0.03, -0.02}, {0.03, 0.0, 0.04}, {-0.02, 0.04, 0.0}};
    for (int other = 0; other < 3; ++other)
      if (other != d) row[other] = offdiagonal[d][other];
  }
};
static double magnetic_conductivity(const vec &) { return 0.07; }

class host_segment_counting_lorentzian : public lorentzian_susceptibility {
public:
  host_segment_counting_lorentzian(realnum omega_0, realnum gamma)
      : lorentzian_susceptibility(omega_0, gamma) {}
  susceptibility *clone() const override { return new host_segment_counting_lorentzian(*this); }

  void subtract_P(field_type ft, realnum *f_minus_p[NUM_FIELD_COMPONENTS][2],
                  void *data) const override {
    ++subtract_calls;
    lorentzian_susceptibility::subtract_P(ft, f_minus_p, data);
  }

  void update_P(realnum *W[NUM_FIELD_COMPONENTS][2],
                realnum *W_prev[NUM_FIELD_COMPONENTS][2], realnum dt, const grid_volume &gv,
                void *data) const override {
    ++update_calls;
    lorentzian_susceptibility::update_P(W, W_prev, dt, gv, data);
  }

  static int subtract_calls;
  static int update_calls;
};

int host_segment_counting_lorentzian::subtract_calls = 0;
int host_segment_counting_lorentzian::update_calls = 0;

class cw_custom_source : public continuous_src_time {
public:
  cw_custom_source(double frequency) : continuous_src_time(frequency) {}
  virtual src_time *clone() const { return new cw_custom_source(*this); }
  virtual std::complex<double> current(double time, double dt) const {
    return continuous_src_time::current(time, dt) + std::complex<double>(0.0625, -0.03125);
  }
};
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

static const BufferAccess *find_access(const std::vector<BufferAccess> &accesses, ArrayId id) {
  for (const BufferAccess &access : accesses)
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
  std::string host_error;
  CHECK(p.host_segments.empty(), "PR5 populated live host segments before PR6");
  CHECK(validate_host_segments(p, &host_error), "ordinary plan has invalid host segments: %s",
        host_error.c_str());
  for (const Operation &op : p.operations)
    CHECK(op.kind != OpKind::host_callback, "PR5 emitted a live host callback before PR6");
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

static void test_cw_composite_plan() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  s.set_conductivity(Dz, magnetic_conductivity);
  fields f(&s, 0, 0, true);
  continuous_src_time m0(0.21), mi(0.22), m2(0.23), e0(0.31), ei(0.32);
  cw_custom_source e2(0.33);
  m0.is_integrated = false;
  mi.is_integrated = true;
  m2.is_integrated = false;
  e0.is_integrated = false;
  ei.is_integrated = true;
  e2.is_integrated = false;
  const vec source_point(0.17, 0.19);
  f.add_point_source(Hz, m0, source_point);
  f.add_point_source(Hz, mi, source_point);
  f.add_point_source(Hz, m2, source_point);
  f.add_point_source(Ez, e0, source_point);
  f.add_point_source(Ez, ei, source_point);
  f.add_point_source(Ez, e2, source_point);
  const volume ordinary_monitor(vec(-0.8, -0.6), vec(0.8, 0.6));
  const volume persistent_monitor(vec(-0.8, -0.6), vec(0.8, 0.6));
  f.add_dft(Ez, ordinary_monitor, 0.24, 0.34, 3, true, 1.0, 0, false, 1.0, true, 0,
            2, false);
  f.add_dft(Hz, persistent_monitor, 0.26, 0.36, 3, true, 1.0, 0, false, 1.0, true, 0,
            3, true);
  f.advance(2);

  const StepPlan ordinary = build_step_plan(f, StepProgram::ordinary);
  const StepPlan cw_step = build_step_plan(f, StepProgram::solve_cw);
  const CwPlan plan = build_cw_plan(f, cw_step);
  const CwPlan rebuilt = build_cw_plan(f, cw_step);
  CHECK(plan == rebuilt && rebuilt == plan, "independently rebuilt CwPlans differ");
  CHECK(plan.signature == compute_cw_plan_signature(plan),
        "stored CwPlan signature differs from its structural signature");
  std::string error;
  CHECK(validate_cw_plan(f, cw_step, plan, &error), "canonical CwPlan failed validation: %s",
        error.c_str());

  CHECK(plan.rhs_stages.size() == 2, "CwPlan has %zu RHS stages", plan.rhs_stages.size());
  if (plan.rhs_stages.size() == 2) {
    CHECK(plan.rhs_stages[0].ft == B_stuff && plan.rhs_stages[0].source_time_offset == 0.0,
          "first CW RHS stage is not B at offset 0");
    CHECK(plan.rhs_stages[1].ft == D_stuff && plan.rhs_stages[1].source_time_offset == 0.5,
          "second CW RHS stage is not D at offset 0.5");
    for (const CwRhsStage &stage : plan.rhs_stages)
      CHECK(stage.source_time_index == 0 &&
                stage.source_time_count == f.descriptors->sources.source_times.size(),
            "CW RHS stage does not evaluate all source times in canonical order");
  }
  CHECK(plan.source_time_count == f.descriptors->sources.source_times.size() &&
            plan.rhs_source_count == plan.rhs_sources.size() &&
            plan.final_dft_count == plan.final_dfts.size(),
        "CwPlan checked totals differ from their descriptor vectors");
  CHECK(plan.state_layout_signature == cw_step.cw_state_layout.signature &&
            plan.step_plan_signature == cw_step.signature &&
            plan.source_fingerprint == source_plan_signature(f.descriptors->sources) &&
            plan.monitor_fingerprint == dft_plan_signature(f.descriptors->dfts),
        "CwPlan fingerprints do not bind the canonical inputs");

  size_t expected_source = 0;
  for (size_t stage_index = 0; stage_index < plan.rhs_stages.size(); ++stage_index) {
    const CwRhsStage &stage = plan.rhs_stages[stage_index];
    CHECK(stage.source_index == expected_source, "CW RHS stage has a gap or overlap");
    for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
      if (!f.chunks[chunk]->is_mine()) continue;
      fields_chunk &fc = *f.chunks[chunk];
      const std::vector<src_vol> &live = fc.get_sources(stage.ft);
      for (size_t ordinal = 0; ordinal < live.size(); ++ordinal) {
        const src_vol &sv = live[ordinal];
        const component c =
            direction_component(first_field_component(stage.ft), component_direction(sv.c));
        const bool applies = ((stage.ft == B_stuff && is_magnetic(sv.c)) ||
                              (stage.ft == D_stuff && is_electric(sv.c))) &&
                             fc.f[c][0];
        if (!applies) continue;
        CHECK(expected_source < plan.rhs_sources.size(), "CW RHS omitted a live source");
        if (expected_source >= plan.rhs_sources.size()) continue;
        const CwRhsSourceDescriptor &rhs = plan.rhs_sources[expected_source++];
        CHECK(rhs.source_ordinal == ordinal,
              "CW RHS source order is not chunk then original ordinal");
        CHECK(rhs.mode ==
                  CwRhsSourceMode::primary_subtract_current_dt_including_integrated,
              "CW RHS source has the wrong signed primary-current mode");
        CHECK(rhs.source_descriptor_index < f.descriptors->sources.sources.size(),
              "CW RHS source reference is out of range");
        if (rhs.source_descriptor_index < f.descriptors->sources.sources.size()) {
          const SourceDescriptor &d =
              f.descriptors->sources.sources[rhs.source_descriptor_index];
          CHECK(d.chunk == chunk && d.ft == stage.ft && d.source_ordinal == ordinal,
                "CW RHS reference resolves to the wrong canonical source descriptor");
          CHECK(d.indices.size() == sv.num_points() &&
                    d.complex_amplitudes.size() == sv.num_points(),
                "CW RHS referenced the wrong spatial source row");
          CHECK(d.destination != d.integrated_destination &&
                    d.destination_imag != d.integrated_destination_imag,
                "CW RHS targets the integrated f_minus_p path");
          const BufferAccess *destination_access =
              find_access(stage.accesses, canonical_id(f, d.destination));
          CHECK(destination_access && destination_access->mode == AccessMode::read_write,
                "CW RHS stage lacks a primary destination read_write access");
          if (is_valid(d.condinv)) {
            const BufferAccess *condinv_access =
                find_access(stage.accesses, canonical_id(f, d.condinv));
            CHECK(condinv_access && condinv_access->mode == AccessMode::read,
                  "CW RHS stage lacks a conductivity read access");
          }
        }
      }
    }
    CHECK(expected_source == size_t(stage.source_index) + stage.source_count,
          "CW RHS stage count differs from its exact source sequence");
    const Operation &boundary = cw_step.operations[stage.boundary.operation_index];
    const Operation &constitutive = cw_step.operations[stage.constitutive.operation_index];
    CHECK(boundary.kind == OpKind::transfer_halo && boundary.ft == stage.ft,
          "CW RHS boundary reference is wrong");
    CHECK(constitutive.kind == OpKind::update_eh &&
              constitutive.ft == (stage.ft == B_stuff ? H_stuff : E_stuff),
          "CW RHS constitutive reference is wrong");
    for (const BufferAccess &access : boundary.accesses)
      CHECK(find_access(stage.accesses, canonical_id(f, access.array.id)),
            "CW RHS stage omitted a boundary access");
    for (const BufferAccess &access : constitutive.accesses)
      CHECK(find_access(stage.accesses, canonical_id(f, access.array.id)),
            "CW RHS stage omitted a constitutive access");
  }
  CHECK(expected_source == plan.rhs_sources.size(), "CW RHS contains an extra source row");

  for (const BufferAccess &access : plan.rhs_accesses) {
    const int kind = f.array_catalog->key(access.array.id).kind;
    CHECK(kind != int(array_kind::f_minus_p), "CW RHS access union contains f_minus_p");
    CHECK(access.array.offset == 0 &&
              access.array.elements == f.array_catalog->spec(access.array.id).elements,
          "CW RHS access is not a full allocation");
  }
  for (const ConstitutiveUpdate &d : cw_step.eh_updates)
    CHECK(d.primary == d.base_primary && d.cross1 == d.base_cross1 &&
              d.cross2 == d.base_cross2 &&
              !(d.region.variant_key & constitutive_has_minus_p),
          "solve_cw constitutive descriptors retained f_minus_p semantics");
  for (const Operation &op : cw_step.operations)
    if (op.kind == OpKind::update_eh)
      CHECK(op.source_descriptor_count == 0 && op.polarization_subtraction_count == 0,
            "solve_cw update_eh retained an integrated-source span");
  bool ordinary_integrated_span = false;
  for (const Operation &op : ordinary.operations)
    ordinary_integrated_span =
        ordinary_integrated_span || (op.kind == OpKind::update_eh && op.source_descriptor_count);
  CHECK(or_to_all(ordinary_integrated_span),
        "ordinary plan lost its integrated-source descriptor span");

  CHECK(plan.unpack.skip_w_components && plan.unpack.invalidate_field_values,
        "CW unpack reference lost skip-W or invalidation semantics");
  const CwStepOperationRef unpack_refs[] = {plan.unpack.first_boundary,
                                            plan.unpack.constitutive,
                                            plan.unpack.second_boundary};
  const OpKind unpack_kinds[] = {OpKind::transfer_halo, OpKind::update_eh,
                                 OpKind::transfer_halo};
  const field_type unpack_types[] = {D_stuff, E_stuff, E_stuff};
  for (size_t i = 0; i < 3; ++i) {
    CHECK(unpack_refs[i].operation_index < cw_step.operations.size(),
          "CW unpack operation reference is out of range");
    if (unpack_refs[i].operation_index < cw_step.operations.size()) {
      const Operation &op = cw_step.operations[unpack_refs[i].operation_index];
      CHECK(op.kind == unpack_kinds[i] && op.ft == unpack_types[i],
            "CW unpack operation reference %zu has the wrong semantics", i);
      CHECK(unpack_refs[i].descriptor_index == op.descriptor_index &&
                unpack_refs[i].descriptor_count == op.descriptor_count,
            "CW unpack operation reference %zu has the wrong descriptor span", i);
    }
  }
  for (const BufferAccess &access : cw_step.cw_state_layout.unpack_accesses)
    CHECK(find_access(plan.unpack_accesses, canonical_id(f, access.array.id)),
          "CW unpack access union omitted a state write");

  CHECK(plan.final_dfts.size() == f.descriptors->dfts.size(),
        "CW final DFT reference count differs from the descriptor table");
  bool saw_decimation_two = false, saw_decimation_three = false;
  for (size_t i = 0; i < plan.final_dfts.size(); ++i) {
    const CwDftDescriptorRef &ref = plan.final_dfts[i];
    const DftDescriptor &d = f.descriptors->dfts[i];
    CHECK(ref.descriptor_index == i && ref.chunk == d.chunk && ref.c == d.c &&
              ref.decimation_factor == d.decimation_factor &&
              ref.due_scalar_slot == d.due_scalar_slot,
          "CW final DFT reference %zu differs from its canonical descriptor", i);
    saw_decimation_two = saw_decimation_two || ref.decimation_factor == 2;
    saw_decimation_three = saw_decimation_three || ref.decimation_factor == 3;
    CHECK(find_access(plan.final_dft_accesses, canonical_id(f, d.accumulator)),
          "CW final DFT accesses omit the accumulator");
    CHECK(find_access(plan.final_dft_accesses, canonical_id(f, d.phase_scratch)),
          "CW final DFT accesses omit phase scratch");
    const BufferAccess *accumulator =
        find_access(plan.final_dft_accesses, canonical_id(f, d.accumulator));
    const BufferAccess *phase =
        find_access(plan.final_dft_accesses, canonical_id(f, d.phase_scratch));
    const BufferAccess *source =
        find_access(plan.final_dft_accesses, canonical_id(f, d.source_field.id));
    CHECK(accumulator && accumulator->mode == AccessMode::read_write,
          "CW final DFT accumulator access has the wrong mode");
    CHECK(phase && phase->mode == AccessMode::write,
          "CW final DFT phase access has the wrong mode");
    CHECK(source && source->mode == AccessMode::read,
          "CW final DFT real source access has the wrong mode");
    if (is_valid(d.source_field_imag.id)) {
      const BufferAccess *source_imag =
          find_access(plan.final_dft_accesses, canonical_id(f, d.source_field_imag.id));
      CHECK(source_imag && source_imag->mode == AccessMode::read,
            "CW final DFT imaginary source access has the wrong mode");
    }
  }
  CHECK(or_to_all(saw_decimation_two) && or_to_all(saw_decimation_three),
        "CW final DFT references did not preserve distinct decimation factors");

#define CHECK_CW_COMPOSITE_MUTATION(mutation, message)                                             \
  do {                                                                                             \
    CwPlan changed = plan;                                                                         \
    mutation;                                                                                      \
    changed.signature = compute_cw_plan_signature(changed);                                       \
    std::string changed_error;                                                                     \
    CHECK(changed != plan, message " (equality)");                                                \
    CHECK(!validate_cw_plan(f, cw_step, changed, &changed_error), message " (validation)");       \
  } while (0)
  CHECK_CW_COMPOSITE_MUTATION(changed.rhs_stages[0].source_time_offset = 0.25,
                              "CwPlan accepted a changed RHS time offset");
  CHECK_CW_COMPOSITE_MUTATION(++changed.rhs_stages[0].source_time_index,
                              "CwPlan accepted a changed source-time span");
  CHECK_CW_COMPOSITE_MUTATION(++changed.rhs_stages[0].source_time_count,
                              "CwPlan accepted a changed source-time count");
  CHECK_CW_COMPOSITE_MUTATION(++changed.rhs_stages[0].source_index,
                              "CwPlan accepted a changed RHS source span");
  CHECK_CW_COMPOSITE_MUTATION(++changed.rhs_stages[0].source_count,
                              "CwPlan accepted a changed RHS source count");
  CHECK_CW_COMPOSITE_MUTATION(std::swap(changed.rhs_stages[0], changed.rhs_stages[1]),
                              "CwPlan accepted reordered RHS stages");
  if (!plan.rhs_stages[0].accesses.empty())
    CHECK_CW_COMPOSITE_MUTATION(changed.rhs_stages[0].accesses[0].mode = AccessMode::read,
                                "CwPlan accepted a changed per-stage access");
  CHECK_CW_COMPOSITE_MUTATION(changed.unpack.skip_w_components = false,
                              "CwPlan accepted changed unpack skip-W semantics");
  CHECK_CW_COMPOSITE_MUTATION(changed.unpack.invalidate_field_values = false,
                              "CwPlan accepted changed unpack invalidation semantics");
  CHECK_CW_COMPOSITE_MUTATION(++changed.unpack.constitutive.descriptor_count,
                              "CwPlan accepted a changed unpack descriptor span");
  CHECK_CW_COMPOSITE_MUTATION(++changed.unpack.first_boundary.operation_index,
                              "CwPlan accepted a changed unpack operation index");
  CHECK_CW_COMPOSITE_MUTATION(changed.unpack.second_boundary.kind = OpKind::update_eh,
                              "CwPlan accepted a changed unpack operation kind");
  CHECK_CW_COMPOSITE_MUTATION(changed.unpack.second_boundary.ft = D_stuff,
                              "CwPlan accepted a changed unpack field type");
  CHECK_CW_COMPOSITE_MUTATION(
      std::swap(changed.unpack.first_boundary, changed.unpack.second_boundary),
      "CwPlan accepted reordered unpack operations");
  CHECK_CW_COMPOSITE_MUTATION(++changed.source_fingerprint,
                              "CwPlan accepted a changed source fingerprint");
  CHECK_CW_COMPOSITE_MUTATION(++changed.monitor_fingerprint,
                              "CwPlan accepted a changed monitor fingerprint");
  CHECK_CW_COMPOSITE_MUTATION(++changed.state_layout_signature,
                              "CwPlan accepted a changed layout signature");
  CHECK_CW_COMPOSITE_MUTATION(++changed.step_plan_signature,
                              "CwPlan accepted a changed StepPlan signature");
  CHECK_CW_COMPOSITE_MUTATION(++changed.rhs_source_count,
                              "CwPlan accepted a changed RHS total");
  CHECK_CW_COMPOSITE_MUTATION(++changed.source_time_count,
                              "CwPlan accepted a changed source-time total");
  CHECK_CW_COMPOSITE_MUTATION(++changed.final_dft_count,
                              "CwPlan accepted a changed final-DFT total");
  if (!plan.rhs_sources.empty()) {
    CHECK_CW_COMPOSITE_MUTATION(++changed.rhs_sources[0].source_ordinal,
                                "CwPlan accepted a changed source ordinal");
    CHECK_CW_COMPOSITE_MUTATION(++changed.rhs_sources[0].source_descriptor_index,
                                "CwPlan accepted a changed source reference");
    CHECK_CW_COMPOSITE_MUTATION(
        changed.rhs_sources[0].mode = static_cast<CwRhsSourceMode>(99),
        "CwPlan accepted a changed RHS source mode");
    size_t integrated_rhs = plan.rhs_sources.size();
    for (size_t i = 0; i < plan.rhs_sources.size(); ++i)
      if (f.descriptors->sources.sources[plan.rhs_sources[i].source_descriptor_index].integrated) {
        integrated_rhs = i;
        break;
      }
    CHECK(integrated_rhs < plan.rhs_sources.size(),
          "mixed-source fixture produced no integrated CW RHS row");
    if (integrated_rhs < plan.rhs_sources.size()) {
      const SourceDescriptor &source = f.descriptors->sources.sources[
          plan.rhs_sources[integrated_rhs].source_descriptor_index];
      CHECK(is_valid(source.integrated_destination),
            "integrated source descriptor has no f_minus_p destination");
      if (is_valid(source.integrated_destination)) {
        SourceDescriptor &mutable_source = f.descriptors->sources.sources[
            plan.rhs_sources[integrated_rhs].source_descriptor_index];
        const ArrayId saved_destination = mutable_source.destination;
        mutable_source.destination = mutable_source.integrated_destination;
        CHECK(!validate_cw_plan(f, cw_step, plan, NULL),
              "CwPlan accepted an f_minus_p RHS destination");
        mutable_source.destination = saved_destination;
      }
    }
    CHECK_CW_COMPOSITE_MUTATION(changed.rhs_sources.erase(changed.rhs_sources.begin()),
                                "CwPlan accepted a deleted RHS source");
    CHECK_CW_COMPOSITE_MUTATION(changed.rhs_sources.push_back(plan.rhs_sources[0]),
                                "CwPlan accepted a duplicate RHS source");
    if (plan.rhs_sources.size() > 1)
      CHECK_CW_COMPOSITE_MUTATION(std::swap(changed.rhs_sources[0], changed.rhs_sources[1]),
                                  "CwPlan accepted reordered RHS sources");
    size_t conductive_rhs = plan.rhs_sources.size();
    for (size_t i = 0; i < plan.rhs_sources.size(); ++i)
      if (is_valid(f.descriptors->sources.sources[plan.rhs_sources[i].source_descriptor_index]
                       .condinv)) {
        conductive_rhs = i;
        break;
      }
    CHECK(conductive_rhs < plan.rhs_sources.size(),
          "conductive fixture produced no CW RHS condinv reference");
    if (conductive_rhs < plan.rhs_sources.size()) {
      const uint32_t descriptor_index = plan.rhs_sources[conductive_rhs].source_descriptor_index;
      SourceDescriptor &source = f.descriptors->sources.sources[descriptor_index];
      const ArrayId saved_condinv = source.condinv;
      source.condinv = invalid_array();
      CHECK(!validate_cw_plan(f, cw_step, plan, NULL),
            "CwPlan accepted a changed canonical RHS condinv identity");
      source.condinv = saved_condinv;
    }
  }
  if (!plan.rhs_accesses.empty()) {
    CHECK_CW_COMPOSITE_MUTATION(changed.rhs_accesses[0].mode = AccessMode::read,
                                "CwPlan accepted a changed RHS access mode");
    CHECK_CW_COMPOSITE_MUTATION(++changed.rhs_accesses[0].array.offset,
                                "CwPlan accepted a changed RHS access range");
    CHECK_CW_COMPOSITE_MUTATION(changed.rhs_accesses.pop_back(),
                                "CwPlan accepted a missing RHS access");
    CHECK_CW_COMPOSITE_MUTATION(changed.rhs_accesses.push_back(plan.rhs_accesses[0]),
                                "CwPlan accepted a duplicate RHS access");
  }
  if (!plan.unpack_accesses.empty()) {
    CHECK_CW_COMPOSITE_MUTATION(changed.unpack_accesses.pop_back(),
                                "CwPlan accepted a missing unpack access");
    CHECK_CW_COMPOSITE_MUTATION(++changed.unpack_accesses[0].array.elements,
                                "CwPlan accepted a changed unpack access range");
  }
  if (!plan.final_dfts.empty()) {
    CHECK_CW_COMPOSITE_MUTATION(++changed.final_dfts[0].descriptor_index,
                                "CwPlan accepted a changed final DFT reference");
    CHECK_CW_COMPOSITE_MUTATION(changed.final_dfts.pop_back(),
                                "CwPlan accepted a missing final DFT reference");
    CHECK_CW_COMPOSITE_MUTATION(++changed.final_dfts[0].decimation_factor,
                                "CwPlan accepted a changed final DFT decimation factor");
    CHECK_CW_COMPOSITE_MUTATION(++changed.final_dfts[0].due_scalar_slot,
                                "CwPlan accepted a changed final DFT due slot");
    CHECK_CW_COMPOSITE_MUTATION(++changed.final_dfts[0].chunk,
                                "CwPlan accepted a changed final DFT chunk");
    CHECK_CW_COMPOSITE_MUTATION(changed.final_dfts[0].c = Ex,
                                "CwPlan accepted a changed final DFT component");
    CHECK_CW_COMPOSITE_MUTATION(changed.final_dfts.push_back(plan.final_dfts[0]),
                                "CwPlan accepted a duplicate final DFT reference");
    if (plan.final_dfts.size() > 1)
      CHECK_CW_COMPOSITE_MUTATION(std::swap(changed.final_dfts[0], changed.final_dfts[1]),
                                  "CwPlan accepted reordered final DFT references");
  }
  if (!plan.final_dft_accesses.empty()) {
    CHECK_CW_COMPOSITE_MUTATION(changed.final_dft_accesses[0].mode = AccessMode::read,
                                "CwPlan accepted a changed final DFT access mode");
    CHECK_CW_COMPOSITE_MUTATION(changed.final_dft_accesses.pop_back(),
                                "CwPlan accepted a missing final DFT access");
    CHECK_CW_COMPOSITE_MUTATION(++changed.final_dft_accesses[0].array.elements,
                                "CwPlan accepted a changed final DFT access range");
  }
#undef CHECK_CW_COMPOSITE_MUTATION
  {
    CwPlan changed = plan;
    ++changed.signature;
    CHECK(!validate_cw_plan(f, cw_step, changed, NULL),
          "CwPlan validator accepted a changed composite signature");
  }

  if (!f.descriptors->sources.sources.empty()) {
    SourceDescriptor &source = f.descriptors->sources.sources[0];
    const uint32_t saved_ordinal = source.source_ordinal;
    ++source.source_ordinal;
    bool rejected = false;
    try {
      build_cw_plan(f, build_step_plan(f, StepProgram::solve_cw));
    }
    catch (const std::exception &) { rejected = true; }
    CHECK(rejected, "CwPlan builder accepted a non-live source ordinal");
    source.source_ordinal = saved_ordinal;
  }
  if (!f.descriptors->sources.source_times.empty()) {
    SourceTimeDescriptor &source_time = f.descriptors->sources.source_times[0];
    ++source_time.source_time_id;
    CHECK(!validate_cw_plan(f, cw_step, plan, NULL),
          "CwPlan validator accepted a source-time descriptor unlike the live object");
    --source_time.source_time_id;
  }
  if (!f.descriptors->dfts.empty()) {
    DftDescriptor &dft = f.descriptors->dfts[0];
    dft.omega[0] += 1e-6;
    bool rejected = false;
    try {
      build_cw_plan(f, build_step_plan(f, StepProgram::solve_cw));
    }
    catch (const std::exception &) { rejected = true; }
    CHECK(rejected, "CwPlan builder accepted a DFT descriptor unlike the live monitor");
    dft.omega[0] -= 1e-6;
  }
  if (cw_step.operations.size() > 2) {
    StepPlan changed_step = cw_step;
    ++changed_step.operations[1].guard.variant_index;
    CHECK(!validate_cw_plan(f, changed_step, plan, NULL),
          "CwPlan validator accepted a mutated StepPlan with a stale signature");
    changed_step.signature = compute_step_plan_signature(changed_step);
    bool rejected = false;
    try {
      build_cw_plan(f, changed_step);
    }
    catch (const std::exception &) { rejected = true; }
    CHECK(rejected, "CwPlan builder accepted a re-signed noncanonical solve_cw StepPlan");
  }

  CwPlan cleared = plan;
  cleared.clear();
  CHECK(cleared == CwPlan(), "CwPlan::clear retained descriptor or fingerprint state");

  bool owns_chunk = false, owns_source = false, owns_dft = false;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    owns_chunk = owns_chunk || f.chunks[chunk]->is_mine();
    owns_source = owns_source ||
                  (f.chunks[chunk]->is_mine() &&
                   (!f.chunks[chunk]->get_sources(B_stuff).empty() ||
                    !f.chunks[chunk]->get_sources(D_stuff).empty()));
    owns_dft = owns_dft || (f.chunks[chunk]->is_mine() && f.chunks[chunk]->dft_chunks);
  }
  CHECK(plan.rhs_stages.size() == 2, "idle/owner rank lost fixed-shape RHS stages");
  CHECK(!owns_source || !plan.rhs_sources.empty(), "source owner has no CW RHS rows");
  CHECK(!owns_dft || !plan.final_dfts.empty(), "monitor owner has no final DFT references");
  if (!owns_chunk)
    CHECK(plan.rhs_sources.empty() && plan.final_dfts.empty() && plan.rhs_accesses.empty() &&
              plan.unpack_accesses.empty() && plan.final_dft_accesses.empty(),
          "idle rank retained local CwPlan rows or accesses");
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
  subtraction.transition_index = 3;
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
            plan.operations[0].cylindrical_origin_action_count == 0 &&
            plan.operations[0].polarization_group_index == 0 &&
            plan.operations[0].polarization_group_count == 0,
        "new operation spans are not zero-initialized");

#define CHECK_SIGNATURE_FIELD(expr, message)                                                       \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    expr;                                                                                          \
    CHECK(compute_step_plan_signature(changed) != signature, message);                             \
  } while (0)
  CHECK_SIGNATURE_FIELD(++changed.operations[0].polarization_subtraction_count,
                        "signature ignored polarization subtraction span");
  CHECK_SIGNATURE_FIELD(++changed.operations[0].polarization_group_count,
                        "signature ignored polarization group span");
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
  CHECK_SIGNATURE_FIELD(++changed.polarization_subtractions[0].transition_index,
                        "signature ignored polarization subtraction transition");
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
  CHECK(plan.polarization_groups.empty() && plan.polarization_updates.empty() &&
            plan.polarization_subtractions.empty() &&
            plan.multilevel_population_updates.empty() &&
            plan.multilevel_population_terms.empty() &&
            plan.multilevel_transition_updates.empty() && plan.multilevel_coefficients.empty(),
        "StepPlan::clear retained polarization state");
  plan.clear();
  CHECK(plan.polarization_groups.empty() && plan.polarization_updates.empty() &&
            plan.polarization_subtractions.empty() &&
            plan.multilevel_population_updates.empty() &&
            plan.multilevel_population_terms.empty() &&
            plan.multilevel_transition_updates.empty() && plan.multilevel_coefficients.empty(),
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
    CHECK(plan.polarization_groups.empty() && plan.polarization_updates.empty() &&
              op.descriptor_count == 0 && op.polarization_group_count == 0 &&
              op.accesses.empty(),
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
    CHECK(op.polarization_group_index == 0 &&
              op.polarization_group_count == plan.polarization_groups.size() &&
              plan.polarization_groups.size() == 3,
          "noisy polarization operation does not span three susceptibility groups");
    const int expected_group_states[] = {0, 1, 2};
    for (size_t i = 0; i < plan.polarization_groups.size(); ++i)
      CHECK(plan.polarization_groups[i].kind == PolarizationGroupKind::recurrence &&
                plan.polarization_groups[i].state_index == expected_group_states[i],
            "noisy polarization group schedule differs at state %zu", i);
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
    CHECK(rejected && plan.polarization_updates.size() == before_reappearance &&
              plan.polarization_groups.size() == 3,
          "noncontiguous repeated polarization identity was accepted");
  }
  CHECK(exercised || !owns_chunk,
        "an owning rank lacked enough catalog rows for noisy schedule coverage");
  CHECK(or_to_all(exercised), "no rank had enough catalog rows for noisy schedule coverage");
}

static void test_multilevel_polarization_group_schedule() {
  grid_volume gv = vol2d(3.0, 3.0, 8.0);
  structure s(gv, one, no_pml(), identity(), 2);
  fields f(&s);
  FOR_COMPONENTS(c)
  if (gv.has_field(c)) f.require_component(c);
  gaussian_src_time source(0.2, 0.1);
  f.add_point_source(Ez, source, vec(0.0, 0.0));
  f.advance(1);

  bool owns_chunk = false;
  int owned_chunk = -1;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    if (f.chunks[chunk]->is_mine()) {
      owns_chunk = true;
      if (owned_chunk < 0) owned_chunk = chunk;
    }
  /* This is a synthetic schema test, so provision enough distinct local
     catalog rows instead of depending on how the physical field arrays are
     distributed across MPI owners.  The backing storage remains live for the
     duration of every append/resolve check below. */
  std::vector<realnum> synthetic_storage(27 * 64);
  if (owns_chunk && f.array_catalog) {
    size_t unaliased = 0;
    for (size_t i = 0; i < f.array_catalog->size(); ++i)
      if (!is_valid(f.array_catalog->spec(ArrayId{uint32_t(i)}).alias_of)) ++unaliased;
    size_t synthetic = 0;
    while (unaliased < 27) {
      const StorageKey key{owned_chunk, int(array_kind::dft), -1, -1,
                           UINT64_C(0xf000000000000000) + synthetic};
      ++synthetic;
      if (f.array_catalog->contains(key)) continue;
      f.array_catalog->register_array(key, synthetic_storage.data() + 64 * (synthetic - 1), 64,
                                      array_role::polarization, ElementType::realnum_value);
      ++unaliased;
    }
  }
  std::vector<ArrayId> ids;
  if (f.array_catalog)
    for (size_t i = 0; i < f.array_catalog->size() && ids.size() < 27; ++i)
      if (!is_valid(f.array_catalog->spec(ArrayId{uint32_t(i)}).alias_of))
        ids.push_back(ArrayId{uint32_t(i)});

  bool exercised = false;
  if (ids.size() >= 27) {
    exercised = true;
    const ArraySpec &alias_target_spec = f.array_catalog->spec(ids[0]);
    const StorageKey alias_key{owned_chunk, int(array_kind::dft), -1, -1,
                               UINT64_C(0xefffffffffffffff)};
    const ArrayId alias_of_first = f.array_catalog->register_array(
        alias_key, f.array_catalog->resolve_untyped(ids[0]), alias_target_spec.elements,
        alias_target_spec.role, alias_target_spec.element_type);
    f.array_catalog->set_alias(alias_of_first, ids[0]);
    auto region = [&](component c, int cmp) {
      UpdateRegion r = {};
      r.chunk = owned_chunk;
      r.c = c;
      r.cmp = cmp;
      r.begin = ivec(0, 0, 0);
      r.end = ivec(2, 2, 0);
      r.base = 0;
      r.counts[0] = r.counts[1] = 2;
      r.counts[2] = 1;
      r.strides[0] = 1;
      r.strides[1] = 4;
      r.strides[2] = 0;
      r.variant_key = 0;
      return r;
    };
    auto recurrence = [&](int state, component c, size_t base) {
      PolarizationUpdate update = {};
      update.kind = PolarizationUpdateKind::lorentzian;
      update.region = region(c, 0);
      update.ft = E_stuff;
      update.state_index = state;
      update.p = ids[base];
      update.p_prev = ids[base + 1];
      update.p_cross1 = update.p_prev_cross1 = invalid_array();
      update.p_cross2 = update.p_prev_cross2 = invalid_array();
      update.primary_w = ids[2];
      update.cross_w1 = update.cross_w2 = invalid_array();
      update.diagonal_sigma = ids[6];
      update.offdiagonal_sigma1 = update.offdiagonal_sigma2 = invalid_array();
      update.omega_0 = 0.31;
      update.gamma = 0.07;
      update.gyro_model = GYROTROPIC_LORENTZIAN;
      update.dt = 0.0125;
      return update;
    };

    MultilevelPopulationUpdate population = {};
    population.region = region(Centered, -1);
    population.ft = E_stuff;
    population.state_index = 1;
    population.levels = 3;
    population.transitions = 2;
    population.active_component_cmps = 2;
    population.gamma_inv = ids[0];
    population.populations = ids[1];
    population.scratch_elements_per_point = 3;
    population.dt = 0.0125;

    std::vector<MultilevelPopulationTerm> terms;
    std::vector<MultilevelTransitionUpdate> transitions;
    const component components[] = {Ex, Ey};
    size_t state_id = 8;
    for (int t = 0; t < 2; ++t)
      for (int ci = 0; ci < 2; ++ci) {
        MultilevelPopulationTerm term = {};
        term.transition_index = t;
        term.c = components[ci];
        term.cmp = 0;
        term.w = ids[2 + 2 * ci];
        term.w_prev = ids[3 + 2 * ci];
        term.p = ids[state_id++];
        term.p_prev = ids[state_id++];
        term.centered_offsets[0] = 1 + ci;
        term.centered_offsets[1] = 5 + ci;
        terms.push_back(term);

        MultilevelTransitionUpdate transition = {};
        transition.region = region(term.c, term.cmp);
        transition.ft = E_stuff;
        transition.state_index = 1;
        transition.transition_index = t;
        transition.p = term.p;
        transition.p_prev = term.p_prev;
        transition.w = term.w;
        transition.diagonal_sigma = ids[6 + ci];
        transition.populations = population.populations;
        transition.population_offsets[0] = -term.centered_offsets[0];
        transition.population_offsets[1] = -term.centered_offsets[1];
        transition.population_stride = 3;
        transition.positive_level = t == 0 ? 1 : 2;
        transition.negative_level = t == 0 ? 2 : 1;
        transition.omega = 0.4 + 0.1 * t;
        transition.gamma = 0.03 + 0.01 * t;
        for (int coefficient = 0; coefficient < 5; ++coefficient)
          transition.sigmat[coefficient] = 1.0 + 0.125 * coefficient + t;
        transition.dt = population.dt;
        transitions.push_back(transition);
      }
    const std::vector<double> gamma_matrix{0.01, 0.02, 0.03, 0.04, 0.05,
                                            0.06, 0.07, 0.08, 0.09};
    /* level-major; both transitions have duplicate signs, so the expected
       positive/negative indices pin the CPU's last-sign-wins scan. */
    const std::vector<double> alpha{-0.2, 0.3, 0.4, -0.5, -0.6, 0.7};

    StepPlan plan;
    Operation op = {};
    op.kind = OpKind::update_polarization;
    op.ft = E_stuff;
    op.guard = guard_always();
    const PolarizationUpdate before = recurrence(0, Ex, 16);
    PolarizationUpdate before_noise = {};
    before_noise.kind = PolarizationUpdateKind::noisy_add;
    before_noise.region = before.region;
    before_noise.ft = before.ft;
    before_noise.state_index = before.state_index;
    before_noise.p = before.p;
    before_noise.p_prev = before_noise.p_cross1 = before_noise.p_prev_cross1 = invalid_array();
    before_noise.p_cross2 = before_noise.p_prev_cross2 = invalid_array();
    before_noise.primary_w = before_noise.cross_w1 = before_noise.cross_w2 = invalid_array();
    before_noise.diagonal_sigma = before.diagonal_sigma;
    before_noise.offdiagonal_sigma1 = before_noise.offdiagonal_sigma2 = invalid_array();
    before_noise.omega_0 = before.omega_0;
    before_noise.gamma = before.gamma;
    before_noise.gyro_model = GYROTROPIC_LORENTZIAN;
    before_noise.dt = before.dt;
    before_noise.noise_amplitude = 0.125;
    before_noise.noise_algorithm_version = counter_random_algorithm_version;
    append_polarization_update_group(f, plan, op, std::vector<PolarizationUpdate>{before},
                                     std::vector<PolarizationUpdate>{before_noise});
    append_multilevel_update_group(f, plan, op, population, terms, transitions, gamma_matrix,
                                   alpha);
    const PolarizationUpdate after = recurrence(2, Ey, 18);
    append_polarization_update_group(f, plan, op, std::vector<PolarizationUpdate>{after},
                                     std::vector<PolarizationUpdate>());

    Operation hop = {};
    hop.kind = OpKind::update_polarization;
    hop.ft = H_stuff;
    hop.guard = guard_always();
    hop.descriptor_index = uint32_t(plan.polarization_updates.size());
    hop.polarization_group_index = uint32_t(plan.polarization_groups.size());
    MultilevelPopulationUpdate h_population = {};
    h_population.region = region(Centered, -1);
    h_population.ft = H_stuff;
    h_population.state_index = 0;
    h_population.levels = 2;
    h_population.transitions = 1;
    h_population.active_component_cmps = 1;
    h_population.gamma_inv = ids[20];
    h_population.populations = ids[21];
    h_population.scratch_elements_per_point = 2;
    h_population.dt = 0.0125;
    MultilevelPopulationTerm h_term = {};
    h_term.transition_index = 0;
    h_term.c = Hx;
    h_term.cmp = 0;
    h_term.w = ids[22];
    h_term.w_prev = ids[23];
    h_term.p = ids[24];
    h_term.p_prev = ids[25];
    h_term.centered_offsets[0] = 2;
    h_term.centered_offsets[1] = 6;
    MultilevelTransitionUpdate h_transition = {};
    h_transition.region = region(Hx, 0);
    h_transition.ft = H_stuff;
    h_transition.state_index = 0;
    h_transition.transition_index = 0;
    h_transition.p = h_term.p;
    h_transition.p_prev = h_term.p_prev;
    h_transition.w = h_term.w;
    h_transition.diagonal_sigma = ids[26];
    h_transition.populations = h_population.populations;
    h_transition.population_offsets[0] = -h_term.centered_offsets[0];
    h_transition.population_offsets[1] = -h_term.centered_offsets[1];
    h_transition.population_stride = 2;
    h_transition.positive_level = 1;
    h_transition.negative_level = 0;
    h_transition.omega = 0.71;
    h_transition.gamma = 0.09;
    for (int coefficient = 0; coefficient < 5; ++coefficient)
      h_transition.sigmat[coefficient] = 2.0 + 0.25 * coefficient;
    h_transition.dt = h_population.dt;
    append_multilevel_update_group(f, plan, hop, h_population,
                                   std::vector<MultilevelPopulationTerm>{h_term},
                                   std::vector<MultilevelTransitionUpdate>{h_transition},
                                   std::vector<double>{0.02, 0.0, 0.0, 0.03},
                                   std::vector<double>{-0.4, 0.5});

    CHECK(plan.polarization_groups.size() == 4 && op.polarization_group_index == 0 &&
              op.polarization_group_count == 3,
          "electric polarization operation does not contain exactly three ordered groups");
    CHECK(hop.polarization_group_index == 3 && hop.polarization_group_count == 1,
          "magnetic polarization operation does not contain exactly one group");
    const PolarizationGroupKind expected_kinds[] = {
        PolarizationGroupKind::recurrence, PolarizationGroupKind::multilevel,
        PolarizationGroupKind::recurrence, PolarizationGroupKind::multilevel};
    const int expected_states[] = {0, 1, 2, 0};
    const field_type expected_families[] = {E_stuff, E_stuff, E_stuff, H_stuff};
    for (size_t i = 0; i < plan.polarization_groups.size(); ++i)
      CHECK(plan.polarization_groups[i].kind == expected_kinds[i] &&
                plan.polarization_groups[i].state_index == expected_states[i] &&
                plan.polarization_groups[i].ft == expected_families[i],
            "mixed polarization group order differs at %zu", i);
    CHECK(plan.polarization_groups[0].recurrence_count == 1 &&
              plan.polarization_groups[0].noise_count == 1,
          "noisy group before the multilevel state lost recurrence/noise ordering");
    CHECK(plan.multilevel_population_updates.size() == 2 &&
              plan.multilevel_population_terms.size() == terms.size() + 1 &&
              plan.multilevel_transition_updates.size() == transitions.size() + 1 &&
              plan.multilevel_coefficients.size() == gamma_matrix.size() + alpha.size() + 6,
          "multilevel group did not publish its exact action spans");
    for (size_t i = 0; i < terms.size(); ++i) {
      CHECK(plan.multilevel_population_terms[i] == terms[i],
            "electric multilevel population term differs at %zu", i);
      CHECK(plan.multilevel_transition_updates[i] == transitions[i],
            "electric multilevel transition differs at %zu", i);
    }
    CHECK(plan.multilevel_population_terms.back() == h_term &&
              plan.multilevel_transition_updates.back() == h_transition,
          "nonempty L2/T1 magnetic schema differs from its source rows");
    const MultilevelPopulationUpdate &published = plan.multilevel_population_updates[0];
    CHECK(published.gamma_index == 0 && published.gamma_count == gamma_matrix.size() &&
              published.alpha_index == gamma_matrix.size() &&
              published.alpha_count == alpha.size() && published.term_index == 0 &&
              published.term_count == terms.size(),
          "multilevel coefficient/term spans are incorrect");
    const MultilevelPopulationUpdate &published_h = plan.multilevel_population_updates[1];
    CHECK(published_h.gamma_index == gamma_matrix.size() + alpha.size() &&
              published_h.gamma_count == 4 && published_h.alpha_index ==
                                                     gamma_matrix.size() + alpha.size() + 4 &&
              published_h.alpha_count == 2 && published_h.term_index == terms.size() &&
              published_h.term_count == 1,
          "nonempty L2/T1 coefficient/term spans are incorrect");
    for (size_t i = 0; i < gamma_matrix.size(); ++i)
      CHECK(plan.multilevel_coefficients[i] == gamma_matrix[i],
            "multilevel Gamma coefficient order differs at %zu", i);
    for (size_t i = 0; i < alpha.size(); ++i)
      CHECK(plan.multilevel_coefficients[gamma_matrix.size() + i] == alpha[i],
            "multilevel alpha coefficient order differs at %zu", i);

    std::vector<BufferAccess> expected_accesses;
    auto expect_access = [&](ArrayId id, AccessMode mode) {
      merge_expected_access(expected_accesses,
                            BufferAccess{ArrayRef{id, 0, f.array_catalog->spec(id).elements}, mode});
    };
    for (const PolarizationUpdate *ordinary : {&before, &after}) {
      expect_access(ordinary->p, AccessMode::read_write);
      expect_access(ordinary->p_prev, AccessMode::read_write);
      expect_access(ordinary->primary_w, AccessMode::read);
      expect_access(ordinary->diagonal_sigma, AccessMode::read);
    }
    expect_access(population.gamma_inv, AccessMode::read);
    expect_access(population.populations, AccessMode::read_write);
    for (size_t i = 0; i < terms.size(); ++i) {
      expect_access(terms[i].w, AccessMode::read);
      expect_access(terms[i].w_prev, AccessMode::read);
      expect_access(terms[i].p, AccessMode::read_write);
      expect_access(terms[i].p_prev, AccessMode::read_write);
      expect_access(transitions[i].diagonal_sigma, AccessMode::read);
    }
    CHECK(op.accesses.size() == expected_accesses.size(),
          "multilevel operation has %zu accesses, expected exact union of %zu",
          op.accesses.size(), expected_accesses.size());
    for (const BufferAccess &want : expected_accesses) {
      const BufferAccess *got = find_access(op, want.array.id);
      CHECK(got && same_access(*got, want),
            "multilevel operation has an incorrect access for ArrayId %u", want.array.id.value);
    }
    std::vector<BufferAccess> expected_h_accesses;
    auto expect_h_access = [&](ArrayId id, AccessMode mode) {
      merge_expected_access(expected_h_accesses,
                            BufferAccess{ArrayRef{id, 0, f.array_catalog->spec(id).elements}, mode});
    };
    expect_h_access(h_population.gamma_inv, AccessMode::read);
    expect_h_access(h_population.populations, AccessMode::read_write);
    expect_h_access(h_term.w, AccessMode::read);
    expect_h_access(h_term.w_prev, AccessMode::read);
    expect_h_access(h_term.p, AccessMode::read_write);
    expect_h_access(h_term.p_prev, AccessMode::read_write);
    expect_h_access(h_transition.diagonal_sigma, AccessMode::read);
    CHECK(hop.accesses.size() == expected_h_accesses.size(),
          "L2/T1 magnetic operation has %zu accesses instead of exact union of %zu",
          hop.accesses.size(), expected_h_accesses.size());
    for (const BufferAccess &want : expected_h_accesses) {
      const BufferAccess *got = find_access(hop, want.array.id);
      CHECK(got && same_access(*got, want),
            "L2/T1 magnetic operation has an incorrect access for ArrayId %u",
            want.array.id.value);
    }
    const BufferAccess *h_gamma_access = find_access(hop, h_population.gamma_inv);
    const BufferAccess *h_population_access = find_access(hop, h_population.populations);
    CHECK(h_gamma_access && h_gamma_access->mode == AccessMode::read && h_population_access &&
              h_population_access->mode == AccessMode::read_write,
          "L2/T1 magnetic scalar accesses are incorrect");

    for (const MultilevelPopulationTerm &term : terms) {
      PolarizationSubtraction subtraction = {};
      subtraction.chunk = owned_chunk;
      subtraction.c = term.c;
      subtraction.cmp = term.cmp;
      subtraction.state_index = 1;
      subtraction.transition_index = term.transition_index;
      subtraction.target = term.w;
      subtraction.p = term.p;
      subtraction.elements = 17;
      plan.polarization_subtractions.push_back(subtraction);
    }
    op.polarization_subtraction_index = 0;
    op.polarization_subtraction_count = uint32_t(terms.size());
    PolarizationSubtraction h_subtraction = {};
    h_subtraction.chunk = owned_chunk;
    h_subtraction.c = h_term.c;
    h_subtraction.cmp = h_term.cmp;
    h_subtraction.state_index = h_population.state_index;
    h_subtraction.transition_index = h_term.transition_index;
    h_subtraction.target = h_term.w;
    h_subtraction.p = h_term.p;
    h_subtraction.elements = 19;
    hop.polarization_subtraction_index = uint32_t(plan.polarization_subtractions.size());
    plan.polarization_subtractions.push_back(h_subtraction);
    hop.polarization_subtraction_count = 1;
    CHECK(plan.polarization_subtractions.size() == terms.size() + 1,
          "multilevel subtraction schedule has the wrong row count");
    for (size_t i = 0; i < terms.size(); ++i)
      CHECK(plan.polarization_subtractions[i].transition_index == terms[i].transition_index &&
                plan.polarization_subtractions[i].c == terms[i].c &&
                plan.polarization_subtractions[i].cmp == terms[i].cmp,
            "multilevel subtraction order differs from transition-major term %zu", i);
    CHECK(plan.polarization_subtractions.back() == h_subtraction,
          "L2/T1 magnetic subtraction row differs from its transition");
    plan.operations.push_back(op);
    plan.operations.push_back(hop);
    CHECK(plan.operations.size() == 2 && plan.operations[0].ft == E_stuff &&
              plan.operations[1].ft == H_stuff,
          "multilevel schedule did not retain exactly one operation per field family");
    plan.signature = compute_step_plan_signature(plan);
    const uint64_t signature = plan.signature;

#define CHECK_MULTILEVEL_SIGNATURE(expr, message)                                                  \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    expr;                                                                                          \
    CHECK(compute_step_plan_signature(changed) != signature, message);                             \
  } while (0)
    CHECK_MULTILEVEL_SIGNATURE(++changed.operations[0].polarization_group_count,
                               "signature ignored multilevel group operation span");
    CHECK_MULTILEVEL_SIGNATURE(++changed.operations[0].polarization_group_index,
                               "signature ignored multilevel group operation index");
    CHECK_MULTILEVEL_SIGNATURE(
        changed.polarization_groups[1].kind = PolarizationGroupKind::recurrence,
        "signature ignored multilevel group kind");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_groups[1].chunk,
                               "signature ignored multilevel group chunk");
    CHECK_MULTILEVEL_SIGNATURE(changed.polarization_groups[1].ft = H_stuff,
                               "signature ignored multilevel group field family");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_groups[1].state_index,
                               "signature ignored multilevel group identity");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_groups[1].recurrence_index,
                               "signature ignored multilevel group recurrence index");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_groups[1].recurrence_count,
                               "signature ignored multilevel group recurrence count");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_groups[1].noise_count,
                               "signature ignored multilevel group noise count");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_groups[1].population_index,
                               "signature ignored multilevel group population span");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_groups[1].population_count,
                               "signature ignored multilevel group population count");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_groups[1].transition_index,
                               "signature ignored multilevel group transition index");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_groups[1].transition_count,
                               "signature ignored multilevel group transition span");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].levels,
                               "signature ignored multilevel level count");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].transitions,
                               "signature ignored multilevel transition count");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].active_component_cmps,
                               "signature ignored multilevel active component count");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].gamma_index,
                               "signature ignored multilevel coefficient span");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].gamma_count,
                               "signature ignored multilevel Gamma count");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].alpha_index,
                               "signature ignored multilevel alpha index");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].term_count,
                               "signature ignored multilevel term span");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].term_index,
                               "signature ignored multilevel term index");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].gamma_inv.value,
                               "signature ignored multilevel GammaInv ArrayId");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].region.base,
                               "signature ignored multilevel population region");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].region.chunk,
                               "signature ignored multilevel population chunk");
    CHECK_MULTILEVEL_SIGNATURE(changed.multilevel_population_updates[0].region.c = Ex,
                               "signature ignored multilevel population component");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].region.cmp,
                               "signature ignored multilevel population complex lane");
    CHECK_MULTILEVEL_SIGNATURE(
        changed.multilevel_population_updates[0].region.begin = ivec(2, 0, 0),
        "signature ignored multilevel population region begin");
    CHECK_MULTILEVEL_SIGNATURE(
        changed.multilevel_population_updates[0].region.end = ivec(4, 2, 0),
        "signature ignored multilevel population region end");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].region.counts[0],
                               "signature ignored multilevel population count geometry");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].region.strides[0],
                               "signature ignored multilevel population stride geometry");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].region.variant_key,
                               "signature ignored multilevel population variant");
    CHECK_MULTILEVEL_SIGNATURE(changed.multilevel_population_updates[0].ft = H_stuff,
                               "signature ignored multilevel population field family");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].state_index,
                               "signature ignored multilevel population state");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].populations.value,
                               "signature ignored multilevel population ArrayId");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_updates[0].alpha_count,
                               "signature ignored multilevel alpha span");
    CHECK_MULTILEVEL_SIGNATURE(changed.multilevel_population_updates[0].dt += 0.01,
                               "signature ignored multilevel population timestep");
    CHECK_MULTILEVEL_SIGNATURE(
        ++changed.multilevel_population_updates[0].scratch_elements_per_point,
        "signature ignored multilevel scratch size");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_terms[0].transition_index,
                               "signature ignored multilevel term transition");
    CHECK_MULTILEVEL_SIGNATURE(changed.multilevel_population_terms[0].c = Ey,
                               "signature ignored multilevel term component");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_terms[0].w.value,
                               "signature ignored multilevel term field ArrayId");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_terms[0].w_prev.value,
                               "signature ignored multilevel previous-field ArrayId");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_terms[0].p.value,
                               "signature ignored multilevel term ArrayId");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_terms[0].p_prev.value,
                               "signature ignored multilevel previous-P ArrayId");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_terms[0].cmp,
                               "signature ignored multilevel term complex lane");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_terms[0].centered_offsets[0],
                               "signature ignored multilevel term offset");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_population_terms[0].centered_offsets[1],
                               "signature ignored second multilevel term offset");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].positive_level,
                               "signature ignored multilevel transition level");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].negative_level,
                               "signature ignored multilevel negative level");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].transition_index,
                               "signature ignored multilevel transition ordinal");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].region.chunk,
                               "signature ignored multilevel transition chunk");
    CHECK_MULTILEVEL_SIGNATURE(changed.multilevel_transition_updates[0].region.c = Ez,
                               "signature ignored multilevel transition component");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].region.cmp,
                               "signature ignored multilevel transition complex lane");
    CHECK_MULTILEVEL_SIGNATURE(
        changed.multilevel_transition_updates[0].region.begin = ivec(2, 0, 0),
        "signature ignored multilevel transition region begin");
    CHECK_MULTILEVEL_SIGNATURE(
        changed.multilevel_transition_updates[0].region.end = ivec(4, 2, 0),
        "signature ignored multilevel transition region end");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].region.base,
                               "signature ignored multilevel transition base");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].region.counts[0],
                               "signature ignored multilevel transition count geometry");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].region.strides[0],
                               "signature ignored multilevel transition stride geometry");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].region.variant_key,
                               "signature ignored multilevel transition variant");
    CHECK_MULTILEVEL_SIGNATURE(changed.multilevel_transition_updates[0].ft = H_stuff,
                               "signature ignored multilevel transition field family");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].state_index,
                               "signature ignored multilevel transition state");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].diagonal_sigma.value,
                               "signature ignored multilevel transition sigma ArrayId");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].p.value,
                               "signature ignored multilevel transition P ArrayId");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].p_prev.value,
                               "signature ignored multilevel transition previous-P ArrayId");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].w.value,
                               "signature ignored multilevel transition field ArrayId");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].populations.value,
                               "signature ignored multilevel transition populations ArrayId");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].population_stride,
                               "signature ignored multilevel population stride");
    CHECK_MULTILEVEL_SIGNATURE(++changed.multilevel_transition_updates[0].population_offsets[0],
                               "signature ignored multilevel transition population offset");
    CHECK_MULTILEVEL_SIGNATURE(changed.multilevel_transition_updates[0].omega += 0.01,
                               "signature ignored multilevel transition coefficient");
    CHECK_MULTILEVEL_SIGNATURE(changed.multilevel_transition_updates[0].gamma += 0.01,
                               "signature ignored multilevel transition damping");
    CHECK_MULTILEVEL_SIGNATURE(changed.multilevel_transition_updates[0].sigmat[3] += 0.01,
                               "signature ignored multilevel sigmat coefficient");
    CHECK_MULTILEVEL_SIGNATURE(changed.multilevel_transition_updates[0].dt += 0.01,
                               "signature ignored multilevel transition timestep");
    CHECK_MULTILEVEL_SIGNATURE(changed.multilevel_coefficients[0] += 0.01,
                               "signature ignored multilevel coefficient payload");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_subtractions[1].transition_index,
                               "signature ignored multilevel subtraction transition order");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_subtractions[1].chunk,
                               "signature ignored multilevel subtraction chunk");
    CHECK_MULTILEVEL_SIGNATURE(changed.polarization_subtractions[1].c = Ez,
                               "signature ignored multilevel subtraction component");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_subtractions[1].cmp,
                               "signature ignored multilevel subtraction complex lane");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_subtractions[1].state_index,
                               "signature ignored multilevel subtraction state");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_subtractions[1].target.value,
                               "signature ignored multilevel subtraction target");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_subtractions[1].p.value,
                               "signature ignored multilevel subtraction P");
    CHECK_MULTILEVEL_SIGNATURE(++changed.polarization_subtractions[1].elements,
                               "signature ignored multilevel subtraction extent");
#undef CHECK_MULTILEVEL_SIGNATURE

#define CHECK_TERM_EQUALITY(expr, message)                                                         \
  do {                                                                                             \
    MultilevelPopulationTerm changed_term = terms[0];                                              \
    expr;                                                                                          \
    CHECK(changed_term != terms[0], message);                                                      \
  } while (0)
    CHECK_TERM_EQUALITY(++changed_term.transition_index,
                        "multilevel term equality ignored transition ordinal");
    CHECK_TERM_EQUALITY(changed_term.c = Ey,
                        "multilevel term equality ignored component");
    CHECK_TERM_EQUALITY(++changed_term.cmp, "multilevel term equality ignored complex lane");
    CHECK_TERM_EQUALITY(++changed_term.w.value, "multilevel term equality ignored W");
    CHECK_TERM_EQUALITY(++changed_term.w_prev.value,
                        "multilevel term equality ignored W_prev");
    CHECK_TERM_EQUALITY(++changed_term.p.value, "multilevel term equality ignored P");
    CHECK_TERM_EQUALITY(++changed_term.p_prev.value,
                        "multilevel term equality ignored P_prev");
    CHECK_TERM_EQUALITY(++changed_term.centered_offsets[0],
                        "multilevel term equality ignored first offset");
    CHECK_TERM_EQUALITY(++changed_term.centered_offsets[1],
                        "multilevel term equality ignored second offset");
#undef CHECK_TERM_EQUALITY

#define CHECK_POPULATION_EQUALITY(expr, message)                                                   \
  do {                                                                                             \
    MultilevelPopulationUpdate changed_population = published;                                    \
    expr;                                                                                          \
    CHECK(changed_population != published, message);                                              \
  } while (0)
    CHECK_POPULATION_EQUALITY(++changed_population.region.chunk,
                              "multilevel population equality ignored chunk");
    CHECK_POPULATION_EQUALITY(changed_population.region.c = Ex,
                              "multilevel population equality ignored component");
    CHECK_POPULATION_EQUALITY(++changed_population.region.cmp,
                              "multilevel population equality ignored complex lane");
    CHECK_POPULATION_EQUALITY(changed_population.region.begin = ivec(2, 0, 0),
                              "multilevel population equality ignored begin");
    CHECK_POPULATION_EQUALITY(changed_population.region.end = ivec(4, 2, 0),
                              "multilevel population equality ignored end");
    CHECK_POPULATION_EQUALITY(++changed_population.region.base,
                              "multilevel population equality ignored base");
    CHECK_POPULATION_EQUALITY(++changed_population.region.counts[0],
                              "multilevel population equality ignored count geometry");
    CHECK_POPULATION_EQUALITY(++changed_population.region.strides[0],
                              "multilevel population equality ignored stride geometry");
    CHECK_POPULATION_EQUALITY(++changed_population.region.variant_key,
                              "multilevel population equality ignored region variant");
    CHECK_POPULATION_EQUALITY(changed_population.ft = H_stuff,
                              "multilevel population equality ignored field family");
    CHECK_POPULATION_EQUALITY(++changed_population.state_index,
                              "multilevel population equality ignored state");
    CHECK_POPULATION_EQUALITY(++changed_population.levels,
                              "multilevel population equality ignored levels");
    CHECK_POPULATION_EQUALITY(++changed_population.transitions,
                              "multilevel population equality ignored transitions");
    CHECK_POPULATION_EQUALITY(++changed_population.active_component_cmps,
                              "multilevel population equality ignored active component count");
    CHECK_POPULATION_EQUALITY(++changed_population.gamma_inv.value,
                              "multilevel population equality ignored GammaInv");
    CHECK_POPULATION_EQUALITY(++changed_population.populations.value,
                              "multilevel population equality ignored populations");
    CHECK_POPULATION_EQUALITY(++changed_population.gamma_index,
                              "multilevel population equality ignored Gamma index");
    CHECK_POPULATION_EQUALITY(++changed_population.gamma_count,
                              "multilevel population equality ignored Gamma count");
    CHECK_POPULATION_EQUALITY(++changed_population.alpha_index,
                              "multilevel population equality ignored alpha index");
    CHECK_POPULATION_EQUALITY(++changed_population.alpha_count,
                              "multilevel population equality ignored alpha count");
    CHECK_POPULATION_EQUALITY(++changed_population.term_index,
                              "multilevel population equality ignored term index");
    CHECK_POPULATION_EQUALITY(++changed_population.term_count,
                              "multilevel population equality ignored term count");
    CHECK_POPULATION_EQUALITY(++changed_population.scratch_elements_per_point,
                              "multilevel population equality ignored scratch elements");
    CHECK_POPULATION_EQUALITY(changed_population.dt += 0.01,
                              "multilevel population equality ignored timestep");
#undef CHECK_POPULATION_EQUALITY

#define CHECK_TRANSITION_EQUALITY(expr, message)                                                   \
  do {                                                                                             \
    MultilevelTransitionUpdate changed_transition = transitions[0];                               \
    expr;                                                                                          \
    CHECK(changed_transition != transitions[0], message);                                         \
  } while (0)
    CHECK_TRANSITION_EQUALITY(++changed_transition.region.chunk,
                              "multilevel transition equality ignored chunk");
    CHECK_TRANSITION_EQUALITY(changed_transition.region.c = Ez,
                              "multilevel transition equality ignored component");
    CHECK_TRANSITION_EQUALITY(++changed_transition.region.cmp,
                              "multilevel transition equality ignored complex lane");
    CHECK_TRANSITION_EQUALITY(changed_transition.region.begin = ivec(2, 0, 0),
                              "multilevel transition equality ignored begin");
    CHECK_TRANSITION_EQUALITY(changed_transition.region.end = ivec(4, 2, 0),
                              "multilevel transition equality ignored end");
    CHECK_TRANSITION_EQUALITY(++changed_transition.region.base,
                              "multilevel transition equality ignored base");
    CHECK_TRANSITION_EQUALITY(++changed_transition.region.counts[0],
                              "multilevel transition equality ignored count geometry");
    CHECK_TRANSITION_EQUALITY(++changed_transition.region.strides[0],
                              "multilevel transition equality ignored stride geometry");
    CHECK_TRANSITION_EQUALITY(++changed_transition.region.variant_key,
                              "multilevel transition equality ignored variant");
    CHECK_TRANSITION_EQUALITY(changed_transition.ft = H_stuff,
                              "multilevel transition equality ignored field family");
    CHECK_TRANSITION_EQUALITY(++changed_transition.state_index,
                              "multilevel transition equality ignored state");
    CHECK_TRANSITION_EQUALITY(++changed_transition.transition_index,
                              "multilevel transition equality ignored transition ordinal");
    CHECK_TRANSITION_EQUALITY(++changed_transition.p.value,
                              "multilevel transition equality ignored P");
    CHECK_TRANSITION_EQUALITY(++changed_transition.p_prev.value,
                              "multilevel transition equality ignored P_prev");
    CHECK_TRANSITION_EQUALITY(++changed_transition.w.value,
                              "multilevel transition equality ignored W");
    CHECK_TRANSITION_EQUALITY(++changed_transition.diagonal_sigma.value,
                              "multilevel transition equality ignored sigma");
    CHECK_TRANSITION_EQUALITY(++changed_transition.populations.value,
                              "multilevel transition equality ignored populations");
    CHECK_TRANSITION_EQUALITY(++changed_transition.population_offsets[0],
                              "multilevel transition equality ignored first offset");
    CHECK_TRANSITION_EQUALITY(++changed_transition.population_offsets[1],
                              "multilevel transition equality ignored second offset");
    CHECK_TRANSITION_EQUALITY(++changed_transition.population_stride,
                              "multilevel transition equality ignored population stride");
    CHECK_TRANSITION_EQUALITY(++changed_transition.positive_level,
                              "multilevel transition equality ignored positive level");
    CHECK_TRANSITION_EQUALITY(++changed_transition.negative_level,
                              "multilevel transition equality ignored negative level");
    CHECK_TRANSITION_EQUALITY(changed_transition.omega += 0.01,
                              "multilevel transition equality ignored omega");
    CHECK_TRANSITION_EQUALITY(changed_transition.gamma += 0.01,
                              "multilevel transition equality ignored gamma");
    CHECK_TRANSITION_EQUALITY(changed_transition.sigmat[4] += 0.01,
                              "multilevel transition equality ignored sigmat");
    CHECK_TRANSITION_EQUALITY(changed_transition.dt += 0.01,
                              "multilevel transition equality ignored timestep");
#undef CHECK_TRANSITION_EQUALITY
#define CHECK_GROUP_EQUALITY(expr, message)                                                        \
  do {                                                                                             \
    PolarizationUpdateGroup changed_group = plan.polarization_groups[1];                           \
    expr;                                                                                          \
    CHECK(changed_group != plan.polarization_groups[1], message);                                  \
  } while (0)
    CHECK_GROUP_EQUALITY(changed_group.kind = PolarizationGroupKind::recurrence,
                         "polarization group equality ignored kind");
    CHECK_GROUP_EQUALITY(++changed_group.chunk, "polarization group equality ignored chunk");
    CHECK_GROUP_EQUALITY(changed_group.ft = H_stuff,
                         "polarization group equality ignored field family");
    CHECK_GROUP_EQUALITY(++changed_group.state_index,
                         "polarization group equality ignored state");
    CHECK_GROUP_EQUALITY(++changed_group.recurrence_index,
                         "polarization group equality ignored recurrence index");
    CHECK_GROUP_EQUALITY(++changed_group.recurrence_count,
                         "polarization group equality ignored recurrence count");
    CHECK_GROUP_EQUALITY(++changed_group.noise_count,
                         "polarization group equality ignored noise count");
    CHECK_GROUP_EQUALITY(++changed_group.population_index,
                         "polarization group equality ignored population index");
    CHECK_GROUP_EQUALITY(++changed_group.population_count,
                         "polarization group equality ignored population count");
    CHECK_GROUP_EQUALITY(++changed_group.transition_index,
                         "polarization group equality ignored transition index");
    CHECK_GROUP_EQUALITY(++changed_group.transition_count,
                         "polarization group equality ignored transition count");
#undef CHECK_GROUP_EQUALITY
#define CHECK_SUBTRACTION_EQUALITY(expr, message)                                                  \
  do {                                                                                             \
    PolarizationSubtraction changed_subtraction = plan.polarization_subtractions[0];               \
    expr;                                                                                          \
    CHECK(changed_subtraction != plan.polarization_subtractions[0], message);                      \
  } while (0)
    CHECK_SUBTRACTION_EQUALITY(++changed_subtraction.chunk,
                               "polarization subtraction equality ignored chunk");
    CHECK_SUBTRACTION_EQUALITY(changed_subtraction.c = Ez,
                               "polarization subtraction equality ignored component");
    CHECK_SUBTRACTION_EQUALITY(++changed_subtraction.cmp,
                               "polarization subtraction equality ignored complex lane");
    CHECK_SUBTRACTION_EQUALITY(++changed_subtraction.state_index,
                               "polarization subtraction equality ignored state");
    CHECK_SUBTRACTION_EQUALITY(++changed_subtraction.transition_index,
                               "polarization subtraction equality ignored transition order");
    CHECK_SUBTRACTION_EQUALITY(++changed_subtraction.target.value,
                               "polarization subtraction equality ignored target");
    CHECK_SUBTRACTION_EQUALITY(++changed_subtraction.p.value,
                               "polarization subtraction equality ignored P");
    CHECK_SUBTRACTION_EQUALITY(++changed_subtraction.elements,
                               "polarization subtraction equality ignored extent");
#undef CHECK_SUBTRACTION_EQUALITY

    auto expect_rejection = [&](const MultilevelPopulationUpdate &candidate_population,
                                const std::vector<MultilevelPopulationTerm> &candidate_terms,
                                const std::vector<MultilevelTransitionUpdate> &candidate_transitions,
                                const std::vector<double> &candidate_gamma,
                                const std::vector<double> &candidate_alpha,
                                const char *message) {
      StepPlan rejected_plan;
      Operation rejected_op = {};
      rejected_op.kind = OpKind::update_polarization;
      rejected_op.ft = E_stuff;
      rejected_op.guard = guard_always();
      bool rejected = false;
      try {
        append_multilevel_update_group(f, rejected_plan, rejected_op, candidate_population,
                                       candidate_terms, candidate_transitions, candidate_gamma,
                                       candidate_alpha);
      }
      catch (const std::exception &) { rejected = true; }
      CHECK(rejected && rejected_plan.polarization_groups.empty() &&
                rejected_plan.multilevel_population_updates.empty() &&
                rejected_plan.multilevel_population_terms.empty() &&
                rejected_plan.multilevel_transition_updates.empty() &&
                rejected_plan.multilevel_coefficients.empty() && rejected_op.accesses.empty(),
            "%s", message);
    };

    std::vector<MultilevelPopulationTerm> malformed_terms = terms;
    std::swap(malformed_terms[1], malformed_terms[2]);
    expect_rejection(population, malformed_terms, transitions, gamma_matrix, alpha,
                     "reordered multilevel population terms were accepted");
    std::vector<MultilevelTransitionUpdate> malformed_transitions = transitions;
    malformed_transitions.pop_back();
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "missing multilevel transition row was accepted");
    malformed_terms = terms;
    malformed_terms[0].p_prev = malformed_terms[0].p;
    expect_rejection(population, malformed_terms, transitions, gamma_matrix, alpha,
                     "aliased multilevel P/P_prev was accepted");
    malformed_terms = terms;
    malformed_terms[0].p = invalid_array();
    expect_rejection(population, malformed_terms, transitions, gamma_matrix, alpha,
                     "invalid multilevel ArrayId was accepted");
    MultilevelPopulationUpdate malformed_population = population;
    malformed_population.scratch_elements_per_point = 2;
    expect_rejection(malformed_population, terms, transitions, gamma_matrix, alpha,
                     "incorrect multilevel scratch requirement was accepted");
    std::vector<double> malformed_gamma = gamma_matrix;
    malformed_gamma.pop_back();
    expect_rejection(population, terms, transitions, malformed_gamma, alpha,
                     "incorrect multilevel Gamma extent was accepted");
    std::vector<double> malformed_alpha = alpha;
    malformed_alpha[0] = std::numeric_limits<double>::quiet_NaN();
    expect_rejection(population, terms, transitions, gamma_matrix, malformed_alpha,
                     "nonfinite multilevel alpha was accepted");
    malformed_transitions = transitions;
    malformed_transitions[0].dt += 0.01;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "inconsistent multilevel transition dt was accepted");
    malformed_terms = terms;
    malformed_transitions = transitions;
    malformed_terms[2].w = ids[26];
    malformed_transitions[2].w = ids[26];
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "transition-dependent multilevel W binding was accepted");
    malformed_terms = terms;
    malformed_transitions = transitions;
    ++malformed_terms[2].centered_offsets[0];
    malformed_transitions[2].population_offsets[0] =
        -malformed_terms[2].centered_offsets[0];
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "transition-dependent multilevel centering offset was accepted");
    malformed_transitions = transitions;
    ++malformed_transitions[2].region.base;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "transition-dependent multilevel Yee region was accepted");
    malformed_transitions = transitions;
    malformed_transitions[2].diagonal_sigma = ids[26];
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "transition-dependent multilevel sigma binding was accepted");
    malformed_transitions = transitions;
    malformed_transitions[1].omega += 0.01;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "component-dependent multilevel omega was accepted");
    malformed_transitions = transitions;
    malformed_transitions[1].gamma += 0.01;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "component-dependent multilevel gamma was accepted");
    malformed_transitions = transitions;
    malformed_transitions[1].sigmat[4] += 0.01;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "component-dependent multilevel sigmat was accepted");
    malformed_population = population;
    malformed_population.ft = D_stuff;
    expect_rejection(malformed_population, terms, transitions, gamma_matrix, alpha,
                     "non-E/H multilevel population family was accepted");
    malformed_population = population;
    malformed_population.region.chunk = f.num_chunks;
    expect_rejection(malformed_population, terms, transitions, gamma_matrix, alpha,
                     "out-of-range multilevel chunk identity was accepted");
    malformed_terms = terms;
    malformed_transitions = transitions;
    malformed_terms[0].centered_offsets[0] = std::numeric_limits<ptrdiff_t>::min();
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "PTRDIFF_MIN multilevel offset reached unchecked negation");
    malformed_population = population;
    malformed_population.region.counts[0] = std::numeric_limits<size_t>::max();
    malformed_population.region.strides[0] = 2;
    expect_rejection(malformed_population, terms, transitions, gamma_matrix, alpha,
                     "overflowing multilevel region count/stride product was accepted");
    malformed_population = population;
    malformed_population.region.base = std::numeric_limits<size_t>::max();
    malformed_population.region.counts[0] = 2;
    malformed_population.region.strides[0] = 1;
    expect_rejection(malformed_population, terms, transitions, gamma_matrix, alpha,
                     "overflowing multilevel region base was accepted");
    malformed_transitions = transitions;
    malformed_transitions[0].region.strides[0] = -1;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "negative multilevel region stride was accepted");

    malformed_terms = terms;
    malformed_transitions = transitions;
    malformed_terms[1] = malformed_terms[0];
    malformed_transitions[1] = malformed_transitions[0];
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "duplicate multilevel transition row was accepted");
    malformed_terms = terms;
    malformed_transitions = transitions;
    malformed_terms.push_back(terms.back());
    malformed_transitions.push_back(transitions.back());
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "extra multilevel transition row was accepted");
    malformed_terms = terms;
    malformed_transitions = transitions;
    malformed_terms.pop_back();
    malformed_transitions.pop_back();
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "incomplete multilevel transition row set was accepted");
    malformed_terms = terms;
    malformed_transitions = transitions;
    std::swap(malformed_terms[0], malformed_terms[1]);
    std::swap(malformed_transitions[0], malformed_transitions[1]);
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "paired but reordered multilevel rows were accepted");
    malformed_terms = terms;
    malformed_transitions = transitions;
    malformed_terms[2].c = Ez;
    malformed_transitions[2].region.c = Ez;
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "mismatched multilevel transition row set was accepted");

    malformed_alpha = alpha;
    malformed_alpha[0] = 0.2;
    malformed_alpha[2] = 0.4;
    malformed_alpha[4] = 0.6;
    expect_rejection(population, terms, transitions, gamma_matrix, malformed_alpha,
                     "multilevel alpha without a negative level was accepted");
    malformed_transitions = transitions;
    malformed_transitions[0].positive_level = 0;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "multilevel transition ignored last-sign-wins level selection");
    malformed_transitions = transitions;
    ++malformed_transitions[0].population_offsets[0];
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "multilevel population/transition offset mismatch was accepted");

    malformed_population = population;
    malformed_population.populations = malformed_population.gamma_inv;
    expect_rejection(malformed_population, terms, transitions, gamma_matrix, alpha,
                     "aliased multilevel GammaInv/populations was accepted");
    malformed_population = population;
    malformed_population.gamma_inv = alias_of_first;
    expect_rejection(malformed_population, terms, transitions, gamma_matrix, alpha,
                     "noncanonical alias ArrayId was accepted as multilevel GammaInv");
    malformed_terms = terms;
    malformed_transitions = transitions;
    malformed_terms[1].p = malformed_terms[0].p;
    malformed_transitions[1].p = malformed_terms[0].p;
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "cross-row multilevel P alias was accepted");
    malformed_terms = terms;
    malformed_transitions = transitions;
    malformed_terms[0].p = population.populations;
    malformed_transitions[0].p = population.populations;
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "multilevel P aliasing population storage was accepted");
    malformed_terms = terms;
    malformed_transitions = transitions;
    malformed_transitions[0].diagonal_sigma = malformed_terms[1].p;
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "multilevel sigma aliasing writable P was accepted");
    malformed_terms = terms;
    malformed_transitions = transitions;
    malformed_terms[0].w = malformed_terms[1].p;
    malformed_transitions[0].w = malformed_terms[1].p;
    expect_rejection(population, malformed_terms, malformed_transitions, gamma_matrix, alpha,
                     "multilevel field aliasing writable P was accepted");

    malformed_transitions = transitions;
    malformed_transitions[0].ft = H_stuff;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "multilevel transition with wrong field family was accepted");
    malformed_transitions = transitions;
    ++malformed_transitions[0].region.chunk;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "multilevel transition with wrong chunk was accepted");
    malformed_transitions = transitions;
    ++malformed_transitions[0].state_index;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "multilevel transition with wrong state was accepted");
    malformed_transitions = transitions;
    malformed_transitions[0].region.c = Ez;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "multilevel transition with wrong component was accepted");
    malformed_transitions = transitions;
    malformed_transitions[0].region.cmp = 1;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "multilevel transition with wrong complex lane was accepted");
    malformed_transitions = transitions;
    ++malformed_transitions[0].transition_index;
    expect_rejection(population, terms, malformed_transitions, gamma_matrix, alpha,
                     "multilevel transition with wrong transition ordinal was accepted");

    StepPlan bad_span_plan;
    Operation bad_span_op = {};
    bad_span_op.kind = OpKind::update_polarization;
    bad_span_op.ft = E_stuff;
    bad_span_op.polarization_group_index = std::numeric_limits<uint32_t>::max();
    bool bad_span_rejected = false;
    try {
      append_multilevel_update_group(f, bad_span_plan, bad_span_op, population, terms,
                                     transitions, gamma_matrix, alpha);
    }
    catch (const std::exception &) { bad_span_rejected = true; }
    CHECK(bad_span_rejected && bad_span_plan.polarization_groups.empty() &&
              bad_span_plan.multilevel_population_updates.empty() && bad_span_op.accesses.empty(),
          "out-of-range multilevel group span was accepted or mutated the plan");
    bad_span_plan.clear();
    bad_span_op = {};
    bad_span_op.kind = OpKind::update_polarization;
    bad_span_op.ft = E_stuff;
    bad_span_op.descriptor_count = 1;
    bad_span_rejected = false;
    try {
      append_multilevel_update_group(f, bad_span_plan, bad_span_op, population, terms,
                                     transitions, gamma_matrix, alpha);
    }
    catch (const std::exception &) { bad_span_rejected = true; }
    CHECK(bad_span_rejected && bad_span_plan.polarization_groups.empty() &&
              bad_span_plan.multilevel_population_updates.empty() && bad_span_op.accesses.empty(),
          "corrupt multilevel descriptor span was accepted or mutated the plan");

    StepPlan invalid_family_plan;
    Operation invalid_family_op = {};
    invalid_family_op.kind = OpKind::update_polarization;
    invalid_family_op.ft = D_stuff;
    MultilevelPopulationUpdate invalid_family_population = population;
    invalid_family_population.ft = D_stuff;
    invalid_family_population.state_index = 7;
    invalid_family_population.levels = 2;
    invalid_family_population.transitions = 1;
    invalid_family_population.active_component_cmps = 0;
    invalid_family_population.scratch_elements_per_point = 2;
    bool invalid_family_rejected = false;
    try {
      append_multilevel_update_group(f, invalid_family_plan, invalid_family_op,
                                     invalid_family_population,
                                     std::vector<MultilevelPopulationTerm>(),
                                     std::vector<MultilevelTransitionUpdate>(),
                                     std::vector<double>{1, 0, 0, 1},
                                     std::vector<double>{-1, 1});
    }
    catch (const std::exception &) { invalid_family_rejected = true; }
    CHECK(invalid_family_rejected && invalid_family_plan.polarization_groups.empty() &&
              invalid_family_op.accesses.empty(),
          "non-E/H multilevel operation family was accepted or partially published");

    const size_t groups_before = plan.polarization_groups.size();
    const size_t updates_before = plan.polarization_updates.size();
    bool repeated_rejected = false;
    try {
      append_polarization_update_group(f, plan, op, std::vector<PolarizationUpdate>{before},
                                       std::vector<PolarizationUpdate>());
    }
    catch (const std::invalid_argument &) { repeated_rejected = true; }
    CHECK(repeated_rejected && plan.polarization_groups.size() == groups_before &&
              plan.polarization_updates.size() == updates_before,
          "noncontiguous cross-kind group identity was accepted");

    StepPlan decreasing_plan;
    Operation decreasing_op = {};
    decreasing_op.kind = OpKind::update_polarization;
    decreasing_op.ft = E_stuff;
    append_polarization_update_group(f, decreasing_plan, decreasing_op,
                                     std::vector<PolarizationUpdate>{after},
                                     std::vector<PolarizationUpdate>());
    const size_t decreasing_accesses = decreasing_op.accesses.size();
    bool decreasing_rejected = false;
    try {
      PolarizationUpdate state_one = recurrence(1, Ex, 16);
      append_polarization_update_group(f, decreasing_plan, decreasing_op,
                                       std::vector<PolarizationUpdate>{state_one},
                                       std::vector<PolarizationUpdate>());
    }
    catch (const std::invalid_argument &) { decreasing_rejected = true; }
    CHECK(decreasing_rejected && decreasing_plan.polarization_groups.size() == 1 &&
              decreasing_plan.polarization_updates.size() == 1 &&
              decreasing_op.accesses.size() == decreasing_accesses,
          "unique but decreasing polarization state order was accepted or mutated the plan");

    StepPlan protected_plan;
    Operation protected_op = {};
    protected_op.kind = OpKind::update_polarization;
    protected_op.ft = E_stuff;
    MultilevelPopulationUpdate protected_population = population;
    protected_population.state_index = 0;
    std::vector<MultilevelTransitionUpdate> protected_transitions = transitions;
    for (MultilevelTransitionUpdate &transition : protected_transitions)
      transition.state_index = 0;
    append_multilevel_update_group(f, protected_plan, protected_op, protected_population, terms,
                                   protected_transitions, gamma_matrix, alpha);
    const size_t protected_accesses = protected_op.accesses.size();
    const size_t protected_updates = protected_plan.polarization_updates.size();
    PolarizationUpdate aliased_recurrence = recurrence(1, Ex, 16);
    aliased_recurrence.p = protected_population.gamma_inv;
    bool protected_rejected = false;
    try {
      append_polarization_update_group(f, protected_plan, protected_op,
                                       std::vector<PolarizationUpdate>{aliased_recurrence},
                                       std::vector<PolarizationUpdate>());
    }
    catch (const std::invalid_argument &) { protected_rejected = true; }
    const BufferAccess *protected_gamma =
        find_access(protected_op, protected_population.gamma_inv);
    CHECK(protected_rejected && protected_plan.polarization_groups.size() == 1 &&
              protected_plan.polarization_updates.size() == protected_updates &&
              protected_op.accesses.size() == protected_accesses && protected_gamma &&
              protected_gamma->mode == AccessMode::read,
          "later recurrence reused GammaInv or mutated its read-only access");
    PolarizationUpdate indirect_alias = recurrence(1, Ex, 16);
    indirect_alias.p = alias_of_first;
    protected_rejected = false;
    try {
      append_polarization_update_group(f, protected_plan, protected_op,
                                       std::vector<PolarizationUpdate>{indirect_alias},
                                       std::vector<PolarizationUpdate>());
    }
    catch (const std::invalid_argument &) { protected_rejected = true; }
    protected_gamma = find_access(protected_op, protected_population.gamma_inv);
    CHECK(protected_rejected && protected_plan.polarization_groups.size() == 1 &&
              protected_plan.polarization_updates.size() == protected_updates &&
              protected_op.accesses.size() == protected_accesses && protected_gamma &&
              protected_gamma->mode == AccessMode::read,
          "later recurrence reused an alias of GammaInv or mutated the plan");
    PolarizationUpdate aliased_noise = before_noise;
    aliased_noise.state_index = 1;
    aliased_noise.p = protected_population.gamma_inv;
    protected_rejected = false;
    try {
      append_polarization_update_group(f, protected_plan, protected_op,
                                       std::vector<PolarizationUpdate>(),
                                       std::vector<PolarizationUpdate>{aliased_noise});
    }
    catch (const std::invalid_argument &) { protected_rejected = true; }
    protected_gamma = find_access(protected_op, protected_population.gamma_inv);
    CHECK(protected_rejected && protected_plan.polarization_groups.size() == 1 &&
              protected_plan.polarization_updates.size() == protected_updates &&
              protected_op.accesses.size() == protected_accesses && protected_gamma &&
              protected_gamma->mode == AccessMode::read,
          "later noise action reused GammaInv or mutated the plan");

    Operation protected_hop = {};
    protected_hop.kind = OpKind::update_polarization;
    protected_hop.ft = H_stuff;
    protected_hop.descriptor_index = uint32_t(protected_plan.polarization_updates.size());
    protected_hop.polarization_group_index = uint32_t(protected_plan.polarization_groups.size());
    MultilevelPopulationUpdate aliased_h_population = h_population;
    aliased_h_population.gamma_inv = terms[0].p;
    bool cross_operation_rejected = false;
    try {
      append_multilevel_update_group(
          f, protected_plan, protected_hop, aliased_h_population,
          std::vector<MultilevelPopulationTerm>{h_term},
          std::vector<MultilevelTransitionUpdate>{h_transition},
          std::vector<double>{0.02, 0.0, 0.0, 0.03}, std::vector<double>{-0.4, 0.5});
    }
    catch (const std::invalid_argument &) { cross_operation_rejected = true; }
    CHECK(cross_operation_rejected && protected_plan.polarization_groups.size() == 1 &&
              protected_plan.multilevel_population_updates.size() == 1 &&
              protected_hop.accesses.empty(),
          "later multilevel operation aliased earlier state storage or partially published");

    StepPlan ordinary_first_plan;
    Operation ordinary_first_op = {};
    ordinary_first_op.kind = OpKind::update_polarization;
    ordinary_first_op.ft = E_stuff;
    append_polarization_update_group(f, ordinary_first_plan, ordinary_first_op,
                                     std::vector<PolarizationUpdate>{before},
                                     std::vector<PolarizationUpdate>{before_noise});
    std::vector<MultilevelPopulationTerm> aliases_ordinary_terms = terms;
    std::vector<MultilevelTransitionUpdate> aliases_ordinary_transitions = transitions;
    aliases_ordinary_terms[0].p = before.p;
    aliases_ordinary_transitions[0].p = before.p;
    const size_t ordinary_first_accesses = ordinary_first_op.accesses.size();
    bool ordinary_first_rejected = false;
    try {
      append_multilevel_update_group(f, ordinary_first_plan, ordinary_first_op, population,
                                     aliases_ordinary_terms, aliases_ordinary_transitions,
                                     gamma_matrix, alpha);
    }
    catch (const std::invalid_argument &) { ordinary_first_rejected = true; }
    CHECK(ordinary_first_rejected && ordinary_first_plan.polarization_groups.size() == 1 &&
              ordinary_first_plan.multilevel_population_updates.empty() &&
              ordinary_first_op.accesses.size() == ordinary_first_accesses,
          "multilevel state reused earlier recurrence storage or partially published");

    StepPlan zero_row_plan;
    Operation zero_row_op = {};
    zero_row_op.kind = OpKind::update_polarization;
    zero_row_op.ft = E_stuff;
    MultilevelPopulationUpdate zero_row = population;
    zero_row.state_index = 7;
    zero_row.levels = 2;
    zero_row.transitions = 1;
    zero_row.active_component_cmps = 0;
    zero_row.scratch_elements_per_point = 2;
    append_multilevel_update_group(f, zero_row_plan, zero_row_op, zero_row,
                                   std::vector<MultilevelPopulationTerm>(),
                                   std::vector<MultilevelTransitionUpdate>(),
                                   std::vector<double>{1, 0, 0, 1},
                                   std::vector<double>{-1, 1});
    CHECK(zero_row_plan.polarization_groups.size() == 1 &&
              zero_row_plan.multilevel_population_updates.size() == 1 &&
              zero_row_plan.multilevel_population_terms.empty() &&
              zero_row_plan.multilevel_transition_updates.empty() &&
              zero_row_op.accesses.size() == 2,
          "zero-row multilevel state did not retain its population action");

    plan.clear();
    CHECK(plan.polarization_groups.empty() && plan.multilevel_population_updates.empty() &&
              plan.multilevel_population_terms.empty() &&
              plan.multilevel_transition_updates.empty() && plan.multilevel_coefficients.empty(),
          "StepPlan::clear retained multilevel schedule state");
    plan.clear();
    CHECK(plan.polarization_groups.empty() && plan.multilevel_population_updates.empty() &&
              plan.multilevel_population_terms.empty() &&
              plan.multilevel_transition_updates.empty() && plan.multilevel_coefficients.empty(),
          "StepPlan::clear is not idempotent for multilevel schedule state");
  }
  CHECK(exercised || !owns_chunk,
        "an owning rank lacked enough catalog rows for multilevel schedule coverage");
  CHECK(or_to_all(exercised), "no rank had enough catalog rows for multilevel schedule coverage");
}

static void test_multilevel_previous_w_copy_once() {
  const realnum gamma[] = {realnum(0.02), 0, 0, realnum(0.03)};
  const realnum n0[] = {realnum(0.8), realnum(0.2)};
  const realnum alpha[] = {realnum(-0.4), realnum(0.5)};
  const realnum omega[] = {realnum(0.63)};
  const realnum damping[] = {realnum(0.04)};
  const realnum sigmat[] = {1, 1, 1, 1, 1};
  multilevel_susceptibility multilevel(2, 1, gamma, n0, alpha, omega, damping, sigmat);
  grid_volume gv = vol2d(3.0, 3.0, 8.0);
  multitile_anisotropic_material material;
  structure s(gv, material, pml(0.45, X) + pml(0.45, Y), identity(), 1);
  s.add_susceptibility(one, E_stuff, multilevel);
  fields f(&s, 0, 0, true, 0, 4);
  f.use_real_fields();
  f.require_component(Ex);
  f.require_component(Ey);
  f.require_component(Ez);
  f.advance(1);
  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);

  std::map<uint32_t, size_t> copy_counts;
  std::map<std::tuple<int, int, int>, size_t> tile_counts;
  bool local_multitile = false;
  for (const ConstitutiveUpdate &update : plan.eh_updates) {
    const bool copy =
        (update.region.variant_key & constitutive_copy_w_previous) != 0;
    CHECK(copy == is_valid(update.previous_w),
          "constitutive previous-W operand and copy bit disagree");
    if (copy) ++copy_counts[update.previous_w.value];
    const std::tuple<int, int, int> identity(update.region.chunk, int(update.region.c),
                                             update.region.cmp);
    local_multitile = local_multitile || ++tile_counts[identity] > 1;
  }
  bool owns_previous = false;
  if (f.array_catalog)
    for (size_t i = 0; i < f.array_catalog->size(); ++i) {
      const ArrayId id{uint32_t(i)};
      const StorageKey &key = f.array_catalog->key(id);
      if (key.kind != int(array_kind::f_w_prev)) continue;
      owns_previous = true;
      CHECK(copy_counts[id.value] == 1,
            "f_w_prev ArrayId %u is copied %zu times instead of once", id.value,
            copy_counts[id.value]);
    }
  CHECK(or_to_all(owns_previous),
        "multilevel previous-W fixture produced no owned previous-W storage");
  CHECK(or_to_all(local_multitile),
        "multilevel previous-W copy fixture did not produce multiple constitutive tiles");
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
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_updates[0].term_count,
                       "signature ignored legacy flux term count");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_updates[0].recipe_signature,
                       "signature ignored legacy flux recipe identity");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].term_ordinal,
                       "signature ignored legacy flux signed-product ordinal");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].region_ordinal,
                       "signature ignored legacy flux region ordinal");
  CHECK_FLUX_SIGNATURE(changed.legacy_flux_terms[0].sign = 1,
                       "signature ignored legacy flux sign");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].chunk,
                       "signature ignored legacy flux chunk");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].e_real.value,
                       "signature ignored legacy flux field identity");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].h_imag.value,
                       "signature ignored legacy flux imaginary field identity");
  CHECK_FLUX_SIGNATURE(changed.legacy_flux_terms[0].begin.set_direction(
                           X, changed.legacy_flux_terms[0].begin.in_direction(X) + 2),
                       "signature ignored legacy flux region extent");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].base,
                       "signature ignored legacy flux base");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].counts[2],
                       "signature ignored legacy flux loop count");
  CHECK_FLUX_SIGNATURE(++changed.legacy_flux_terms[0].strides[1],
                       "signature ignored legacy flux loop stride");
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
  f.require_component(Ex);
  f.require_component(Ey);
  f.require_component(Hx);
  f.require_component(Hy);
  f.add_flux_plane(volume(vec(0.0, -1.0), vec(0.0, 1.0)));
  f.advance(1);
  refresh_legacy_flux_descriptors(f);
  StepPlan one = build_step_plan(f, StepProgram::ordinary);
  CHECK(one.legacy_flux_updates.size() == 1,
        "one live legacy flux produced %zu update rows", one.legacy_flux_updates.size());
  const uint32_t one_count = uint32_t(f.descriptors->legacy_fluxes[0].terms.size());
  const LegacyFluxUpdate expected_one = {
      0, 0, one_count, f.descriptors->legacy_fluxes[0].recipe_signature};
  if (!one.legacy_flux_updates.empty())
    CHECK(one.legacy_flux_updates[0] == expected_one,
          "one live legacy flux has the wrong PR6 recipe span");
  CHECK(or_to_all(one_count > 0), "one live legacy flux produced no recipe terms");

  f.add_flux_plane(volume(vec(0.5, -1.0), vec(0.5, 1.0)));
  const StepPlan stale = build_step_plan(f, StepProgram::ordinary);
  CHECK(stale.legacy_flux_updates.size() == 2 && stale.legacy_flux_terms.empty(),
        "stale PR6 recipes were consumed after a legacy flux list mutation");
  for (const Operation &op : stale.operations)
    if (op.kind == OpKind::update_flux_half || op.kind == OpKind::update_flux)
      CHECK(op.accesses.empty(), "%s consumed stale legacy flux accesses",
            op_kind_name(op.kind));
  refresh_legacy_flux_descriptors(f);
  StepPlan two = build_step_plan(f, StepProgram::ordinary);
  CHECK(two.legacy_flux_updates.size() == 2,
        "two live legacy fluxes produced %zu update rows", two.legacy_flux_updates.size());
  uint32_t term_index = 0;
  for (size_t i = 0; i < two.legacy_flux_updates.size(); ++i) {
    const uint32_t count = uint32_t(f.descriptors->legacy_fluxes[i].terms.size());
    const LegacyFluxUpdate expected = {uint32_t(i), term_index, count,
                                       f.descriptors->legacy_fluxes[i].recipe_signature};
    CHECK(two.legacy_flux_updates[i] == expected,
          "legacy flux update %zu has the wrong list ordinal or PR6 recipe span", i);
    term_index += count;
  }
  CHECK(two.legacy_flux_terms.size() == term_index,
        "legacy flux plan has %zu terms for spans totaling %u", two.legacy_flux_terms.size(),
        term_index);

  std::vector<ArrayId> expected_accesses;
  for (const LegacyFluxTerm &term : two.legacy_flux_terms) {
    const ArrayId ids[] = {term.e_real, term.e_imag, term.h_real, term.h_imag};
    for (ArrayId id : ids) {
      if (!is_valid(id)) continue;
      bool duplicate = false;
      for (ArrayId prior : expected_accesses) duplicate = duplicate || prior == id;
      if (!duplicate) expected_accesses.push_back(id);
    }
  }

  size_t markers = 0;
  for (const Operation &op : two.operations)
    if (op.kind == OpKind::update_flux_half || op.kind == OpKind::update_flux) {
      ++markers;
      CHECK(op.legacy_flux_index == 0 && op.legacy_flux_count == 2,
            "%s does not cover both live legacy flux updates", op_kind_name(op.kind));
      CHECK(op.accesses.size() == expected_accesses.size(),
            "%s has %zu accesses, expected the exact %zu-field recipe union",
            op_kind_name(op.kind), op.accesses.size(), expected_accesses.size());
      for (size_t i = 0; i < op.accesses.size() && i < expected_accesses.size(); ++i) {
        const BufferAccess &access = op.accesses[i];
        const ArraySpec &spec = f.array_catalog->spec(expected_accesses[i]);
        CHECK(access.array.id == expected_accesses[i] && access.array.offset == 0 &&
                  access.array.elements == spec.elements && access.mode == AccessMode::read,
              "%s access %zu does not match the exact recipe read union", op_kind_name(op.kind),
              i);
      }
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

static Operation host_test_operation(OpKind kind, field_type ft, Guard guard = guard_always()) {
  Operation op = {};
  op.kind = kind;
  op.ft = ft;
  op.guard = guard;
  return op;
}

static void append_synthetic_host_family(StepPlan &plan, field_type ft,
                                         Guard guard = guard_always()) {
  const field_type w_halo = ft == E_stuff ? WE_stuff : WH_stuff;
  const field_type p_halo = ft == E_stuff ? PE_stuff : PH_stuff;

  Operation constitutive_marker = host_test_operation(OpKind::host_callback, ft, guard);
  constitutive_marker.descriptor_index = uint32_t(plan.host_segments.size());
  constitutive_marker.descriptor_count = 1;
  plan.operations.push_back(constitutive_marker);
  const uint32_t constitutive_index = uint32_t(plan.operations.size());
  plan.operations.push_back(host_test_operation(OpKind::update_eh, ft, guard));
  plan.host_segments.push_back(
      HostSegment{HostSegmentPhase::constitutive, ft, constitutive_index, 1, 0, 0, 0, 0});

  plan.operations.push_back(host_test_operation(OpKind::transfer_halo, w_halo, guard));

  Operation polarization_marker = host_test_operation(OpKind::host_callback, ft, guard);
  polarization_marker.descriptor_index = uint32_t(plan.host_segments.size());
  polarization_marker.descriptor_count = 1;
  plan.operations.push_back(polarization_marker);
  const uint32_t polarization_index = uint32_t(plan.operations.size());
  plan.operations.push_back(host_test_operation(OpKind::update_polarization, ft, guard));
  plan.operations.push_back(host_test_operation(OpKind::transfer_halo, p_halo, guard));
  plan.host_segments.push_back(HostSegment{HostSegmentPhase::polarization_and_halo, ft,
                                           polarization_index, 2, 0, 0, 0, 0});

  plan.operations.push_back(host_test_operation(OpKind::transfer_halo, ft, guard));
}

static size_t count_host_phase(const StepPlan &plan, field_type ft, HostSegmentPhase phase) {
  size_t count = 0;
  for (const HostSegment &segment : plan.host_segments)
    if (segment.ft == ft && segment.phase == phase) ++count;
  return count;
}

static void test_host_segment_topologies() {
  std::string error;
  for (int mask = 1; mask <= 3; ++mask) {
    StepPlan plan;
    if (mask & 1) append_synthetic_host_family(plan, E_stuff);
    if (mask & 2) append_synthetic_host_family(plan, H_stuff);
    CHECK(validate_host_segments(plan, &error), "E/H host topology %d rejected: %s", mask,
          error.c_str());
    CHECK(count_host_phase(plan, E_stuff, HostSegmentPhase::constitutive) ==
              size_t((mask & 1) != 0),
          "E/H topology %d has the wrong E constitutive count", mask);
    CHECK(count_host_phase(plan, H_stuff, HostSegmentPhase::polarization_and_halo) ==
              size_t((mask & 2) != 0),
          "E/H topology %d has the wrong H polarization count", mask);
  }

  StepPlan phasing;
  append_synthetic_host_family(phasing, H_stuff, guard_segment(3));
  append_synthetic_host_family(phasing, H_stuff, guard_always());
  CHECK(validate_host_segments(phasing, &error), "repeated guarded H segments rejected: %s",
        error.c_str());
  CHECK(count_host_phase(phasing, H_stuff, HostSegmentPhase::constitutive) == 2,
        "repeated H update_eh occurrences did not produce two host segments");

  const HostSegment &ordinary_h = phasing.host_segments[2];
  phasing.magnetic_half_step.update_h = ordinary_h.operation_index;
  phasing.magnetic_half_step.transfer_h = uint32_t(phasing.operations.size() - 1);
  CHECK(phasing.operations[phasing.magnetic_half_step.update_h].kind == OpKind::update_eh,
        "magnetic update_h index points at the host marker instead of update_eh");
  CHECK(phasing.operations[phasing.magnetic_half_step.transfer_h].kind == OpKind::transfer_halo &&
            phasing.operations[phasing.magnetic_half_step.transfer_h].ft == H_stuff,
        "magnetic transfer_h index was displaced by host markers");
}

static void test_host_segment_access_union() {
  StepPlan plan;
  Operation first = host_test_operation(OpKind::update_polarization, E_stuff);
  first.accesses.push_back(BufferAccess{ArrayRef{ArrayId{7}, 0, 64}, AccessMode::read});
  first.accesses.push_back(BufferAccess{ArrayRef{ArrayId{8}, 0, 32}, AccessMode::write});
  plan.operations.push_back(first);
  Operation second = host_test_operation(OpKind::transfer_halo, PE_stuff);
  second.accesses.push_back(BufferAccess{ArrayRef{ArrayId{7}, 0, 64}, AccessMode::write});
  plan.operations.push_back(second);
  std::vector<BufferAccess> additional;
  additional.push_back(BufferAccess{ArrayRef{ArrayId{9}, 4, 12}, AccessMode::read_write});
  additional.push_back(BufferAccess{ArrayRef{ArrayId{8}, 0, 32}, AccessMode::read});

  const std::vector<BufferAccess> result =
      build_host_segment_access_union(plan, 0, 2, additional);
  CHECK(result.size() == 3, "host access union has %zu rows instead of 3", result.size());
  if (result.size() == 3) {
    CHECK(result[0].array.id == ArrayId{7} && result[0].mode == AccessMode::read_write,
          "host access union did not promote read/write alias 7");
    CHECK(result[1].array.id == ArrayId{8} && result[1].mode == AccessMode::read_write,
          "host access union did not promote write/read alias 8");
    CHECK(result[2].array.id == ArrayId{9} && result[2].mode == AccessMode::read_write,
          "host access union did not preserve deterministic first-use order");
  }

  bool rejected = false;
  try {
    std::vector<BufferAccess> conflicting;
    conflicting.push_back(BufferAccess{ArrayRef{ArrayId{7}, 1, 63}, AccessMode::read});
    build_host_segment_access_union(plan, 0, 2, conflicting);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  CHECK(rejected, "one ArrayId with conflicting host ranges was accepted");

  rejected = false;
  try { build_host_segment_access_union(plan, UINT32_MAX, 2, additional); }
  catch (const std::out_of_range &) { rejected = true; }
  CHECK(rejected, "out-of-range host access operation span was accepted");
}

static void test_cpu_host_marker_is_noop() {
  grid_volume gv = vol2d(2.0, 2.0, 6.0);
  structure s(gv, one, no_pml());
  s.add_susceptibility(one, E_stuff, host_segment_counting_lorentzian(0.7, 0.08));
  fields f(&s);
  f.use_real_fields();
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.0, 0.0));
  f.advance(1); // realize polarization storage and connections

  StepPlan plain;
  plain.operations.push_back(host_test_operation(OpKind::update_eh, E_stuff));
  plain.operations.push_back(host_test_operation(OpKind::update_polarization, E_stuff));
  plain.operations.push_back(host_test_operation(OpKind::transfer_halo, PE_stuff));

  StepPlan marked;
  append_synthetic_host_family(marked, E_stuff);
  /* Remove the W and final E boundaries so the two test plans differ only by
     the two no-op markers around the same update_eh/update_polarization/PE work. */
  marked.operations.erase(marked.operations.begin() + 2);
  marked.operations.pop_back();
  marked.host_segments[1].operation_index = 3;
  marked.operations[2].descriptor_index = 1;
  std::string error;
  CHECK(validate_host_segments(marked, &error), "CPU marker test plan is invalid: %s",
        error.c_str());

  host_segment_counting_lorentzian::subtract_calls = 0;
  host_segment_counting_lorentzian::update_calls = 0;
  StepPlanTestAccess::execute(f, plain);
  const int plain_subtract = host_segment_counting_lorentzian::subtract_calls;
  const int plain_update = host_segment_counting_lorentzian::update_calls;

  host_segment_counting_lorentzian::subtract_calls = 0;
  host_segment_counting_lorentzian::update_calls = 0;
  StepPlanTestAccess::execute(f, marked);
  CHECK(host_segment_counting_lorentzian::subtract_calls == plain_subtract,
        "CPU host marker duplicated/omitted subtract_P (%d != %d)",
        host_segment_counting_lorentzian::subtract_calls, plain_subtract);
  CHECK(host_segment_counting_lorentzian::update_calls == plain_update,
        "CPU host marker duplicated/omitted update_P (%d != %d)",
        host_segment_counting_lorentzian::update_calls, plain_update);
  CHECK(plain_subtract > 0 && plain_update > 0, "CPU callback-count oracle was vacuous");
}

static void test_host_segment_schema() {
  StepPlan plan;

  Operation constitutive_marker =
      host_test_operation(OpKind::host_callback, E_stuff, guard_segment(7));
  constitutive_marker.descriptor_index = 0;
  constitutive_marker.descriptor_count = 1;
  constitutive_marker.accesses.push_back(
      BufferAccess{ArrayRef{ArrayId{11}, 0, 64}, AccessMode::read_write});
  constitutive_marker.accesses.push_back(
      BufferAccess{ArrayRef{ArrayId{12}, 0, 64}, AccessMode::read_write});
  plan.operations.push_back(constitutive_marker);

  Operation update_eh = host_test_operation(OpKind::update_eh, E_stuff, guard_segment(7));
  update_eh.accesses.push_back(BufferAccess{ArrayRef{ArrayId{11}, 0, 64}, AccessMode::read});
  plan.operations.push_back(update_eh);

  Operation polarization_marker =
      host_test_operation(OpKind::host_callback, H_stuff, guard_always());
  polarization_marker.descriptor_index = 1;
  polarization_marker.descriptor_count = 1;
  polarization_marker.accesses.push_back(
      BufferAccess{ArrayRef{ArrayId{21}, 0, 96}, AccessMode::read_write});
  polarization_marker.accesses.push_back(
      BufferAccess{ArrayRef{ArrayId{22}, 4, 32}, AccessMode::read_write});
  plan.operations.push_back(polarization_marker);

  Operation update_polarization =
      host_test_operation(OpKind::update_polarization, H_stuff, guard_always());
  update_polarization.accesses.push_back(
      BufferAccess{ArrayRef{ArrayId{21}, 0, 96}, AccessMode::write});
  plan.operations.push_back(update_polarization);
  Operation transfer_ph = host_test_operation(OpKind::transfer_halo, PH_stuff, guard_always());
  transfer_ph.accesses.push_back(
      BufferAccess{ArrayRef{ArrayId{22}, 4, 32}, AccessMode::read});
  plan.operations.push_back(transfer_ph);

  const HostSegment constitutive = {HostSegmentPhase::constitutive, E_stuff, 1, 1, 3, 2, 0, 0};
  const HostSegment polarization = {
      HostSegmentPhase::polarization_and_halo, H_stuff, 3, 2, 5, 4, 9, 3};
  plan.host_segments.push_back(constitutive);
  plan.host_segments.push_back(polarization);

  std::string error;
  CHECK(validate_host_segments(plan, &error), "valid host segments were rejected: %s",
        error.c_str());
  CHECK(constitutive == plan.host_segments[0], "identical host segments compare unequal");
  CHECK(constitutive != polarization, "different host segments compare equal");
#define CHECK_HOST_INEQUALITY(expr, message)                                                       \
  do {                                                                                             \
    HostSegment changed = constitutive;                                                            \
    expr;                                                                                          \
    CHECK(changed != constitutive, message);                                                       \
  } while (0)
  CHECK_HOST_INEQUALITY(changed.phase = HostSegmentPhase::polarization_and_halo,
                        "host segment equality ignored phase");
  CHECK_HOST_INEQUALITY(changed.ft = H_stuff, "host segment equality ignored field type");
  CHECK_HOST_INEQUALITY(++changed.operation_index,
                        "host segment equality ignored operation index");
  CHECK_HOST_INEQUALITY(++changed.operation_count,
                        "host segment equality ignored operation count");
  CHECK_HOST_INEQUALITY(++changed.callback_index,
                        "host segment equality ignored callback index");
  CHECK_HOST_INEQUALITY(++changed.callback_count,
                        "host segment equality ignored callback count");
  CHECK_HOST_INEQUALITY(++changed.host_halo_plan_index,
                        "host segment equality ignored host halo index");
  CHECK_HOST_INEQUALITY(++changed.host_halo_plan_count,
                        "host segment equality ignored host halo count");
#undef CHECK_HOST_INEQUALITY
  CHECK(!strcmp(host_segment_phase_name(HostSegmentPhase::constitutive), "constitutive"),
        "constitutive phase has the wrong name");
  CHECK(!strcmp(host_segment_phase_name(HostSegmentPhase::polarization_and_halo),
                "polarization_and_halo"),
        "polarization phase has the wrong name");

  std::vector<std::string> formatted;
  format_step_plan(plan, formatted);
  CHECK(formatted.size() == plan.operations.size(), "host plan formatting lost operations");
  if (formatted.size() == plan.operations.size()) {
    CHECK(formatted[0] ==
              "host_callback(E,constitutive,ops=1+1,callbacks=3+2,halos=0+0)",
          "constitutive host marker formatting is incomplete: %s", formatted[0].c_str());
    CHECK(formatted[2] ==
              "host_callback(H,polarization_and_halo,ops=3+2,callbacks=5+4,halos=9+3)",
          "polarization host marker formatting is incomplete: %s", formatted[2].c_str());
  }

  const uint64_t signature = compute_step_plan_signature(plan);
#define CHECK_HOST_SIGNATURE(expr, message)                                                        \
  do {                                                                                             \
    StepPlan changed = plan;                                                                       \
    expr;                                                                                          \
    CHECK(compute_step_plan_signature(changed) != signature, message);                             \
  } while (0)
  CHECK_HOST_SIGNATURE(changed.host_segments[0].phase =
                           HostSegmentPhase::polarization_and_halo,
                       "signature ignored host segment phase");
  CHECK_HOST_SIGNATURE(changed.host_segments[0].ft = H_stuff,
                       "signature ignored host segment field type");
  CHECK_HOST_SIGNATURE(++changed.host_segments[0].operation_index,
                       "signature ignored host operation span start");
  CHECK_HOST_SIGNATURE(++changed.host_segments[0].operation_count,
                       "signature ignored host operation span count");
  CHECK_HOST_SIGNATURE(++changed.host_segments[0].callback_index,
                       "signature ignored host callback span start");
  CHECK_HOST_SIGNATURE(++changed.host_segments[0].callback_count,
                       "signature ignored host callback span count");
  CHECK_HOST_SIGNATURE(++changed.host_segments[1].host_halo_plan_index,
                       "signature ignored host halo span start");
  CHECK_HOST_SIGNATURE(++changed.host_segments[1].host_halo_plan_count,
                       "signature ignored host halo span count");
  CHECK_HOST_SIGNATURE(++changed.operations[0].guard.scalar_slot,
                       "signature ignored host marker guard");
  CHECK_HOST_SIGNATURE(changed.operations[0].accesses[0].mode = AccessMode::write,
                       "signature ignored host marker access mode");
#undef CHECK_HOST_SIGNATURE

  auto rejected = [&](StepPlan candidate, const char *message) {
    std::string why;
    CHECK(!validate_host_segments(candidate, &why), message);
    CHECK(!why.empty(), "invalid host segment produced no diagnostic");
  };

  {
    StepPlan bad = plan;
    bad.operations[0].descriptor_count = 0;
    rejected(bad, "host marker with an empty segment span was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[0].descriptor_index = 2;
    rejected(bad, "host marker with an out-of-range segment was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[0].polarization_group_count = 1;
    rejected(bad, "host marker with an unrelated payload was accepted");
  }
  {
    StepPlan bad = plan;
    bad.host_segments[0].phase = HostSegmentPhase(99);
    rejected(bad, "host segment with an invalid phase was accepted");
  }
  {
    StepPlan bad = plan;
    bad.host_segments[0].ft = D_stuff;
    rejected(bad, "host segment with a non-E/H field type was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[0].ft = H_stuff;
    rejected(bad, "host callback marker with the wrong field type was accepted");
  }
  {
    StepPlan bad = plan;
    ++bad.host_segments[0].operation_index;
    rejected(bad, "host segment detached from its marker was accepted");
  }
  {
    StepPlan bad = plan;
    bad.host_segments[0].operation_index = 0;
    rejected(bad, "self-covering host segment was accepted");
  }
  {
    StepPlan bad = plan;
    bad.host_segments[0].operation_count = 2;
    rejected(bad, "constitutive host segment with the wrong length was accepted");
  }
  {
    StepPlan bad = plan;
    bad.host_segments[0].host_halo_plan_count = 1;
    rejected(bad, "constitutive host segment with a halo span was accepted");
  }
  {
    StepPlan bad = plan;
    bad.host_segments[0].callback_index = UINT32_MAX;
    bad.host_segments[0].callback_count = 1;
    rejected(bad, "overflowing callback span was accepted");
  }
  {
    StepPlan bad = plan;
    bad.host_segments[1].host_halo_plan_index = UINT32_MAX;
    bad.host_segments[1].host_halo_plan_count = 1;
    rejected(bad, "overflowing host halo span was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[1].kind = OpKind::update_db;
    rejected(bad, "constitutive segment covering the wrong operation was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[4].ft = PE_stuff;
    rejected(bad, "polarization segment covering the wrong halo was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[3].kind = OpKind::update_eh;
    rejected(bad, "polarization segment covering the wrong first operation was accepted");
  }
  {
    StepPlan bad = plan;
    std::swap(bad.operations[3], bad.operations[4]);
    rejected(bad, "reordered polarization/halo segment was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[1].kind = OpKind::host_callback;
    bad.operations[1].descriptor_index = 1;
    bad.operations[1].descriptor_count = 1;
    rejected(bad, "nested host callback marker was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[3].guard = guard_segment(1);
    rejected(bad, "host segment with inconsistent guards was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[0].accesses.erase(bad.operations[0].accesses.begin());
    rejected(bad, "host segment with an incomplete access union was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[0].accesses[0].mode = AccessMode::write;
    rejected(bad, "write-only host access was accepted for a covered read");
  }
  {
    StepPlan bad = plan;
    bad.operations[0].accesses.push_back(bad.operations[0].accesses[0]);
    rejected(bad, "duplicate host segment ArrayId was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[0].accesses[0].array.id = invalid_array();
    rejected(bad, "invalid host segment ArrayId was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[0].accesses[0].array.elements = 0;
    rejected(bad, "empty host segment access was accepted");
  }
  {
    StepPlan bad = plan;
    bad.operations[2].descriptor_index = 0;
    rejected(bad, "host segment referenced by two markers was accepted");
  }
  {
    StepPlan bad = plan;
    bad.host_segments.push_back(constitutive);
    rejected(bad, "unreferenced host segment was accepted");
  }

  plan.clear();
  CHECK(plan.host_segments.empty(), "StepPlan::clear retained host segments");
  plan.clear();
  CHECK(plan.host_segments.empty(), "second StepPlan::clear retained host segments");
  CHECK(validate_host_segments(plan, &error), "empty host segment plan was rejected: %s",
        error.c_str());
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  test_full_plan();
  test_empty_plan();
  test_cw_state_layout();
  test_solve_cw_plan();
  test_lazy_cw_layout_before_storage_refresh();
  test_cw_composite_plan();
  test_phasing_plan();
  test_material_schema_signature();
  test_magnetic_schema_signature();
  test_polarization_schema_signature();
  test_noisy_polarization_group_schedule();
  test_multilevel_polarization_group_schedule();
  test_multilevel_previous_w_copy_once();
  test_legacy_flux_schema_signature();
  test_live_legacy_flux_spans();
  test_beta_schema_signature();
  test_bfast_schema_signature();
  test_cylindrical_schema_signature();
  test_host_segment_topologies();
  test_host_segment_access_union();
  test_cpu_host_marker_is_noop();
  test_host_segment_schema();

  if (failures) {
    master_printf("step_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("step_plan: all checks passed\n");
  return 0;
}
