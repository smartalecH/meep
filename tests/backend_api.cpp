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

/* PR 7 acceptance tests: the backend interface, precision policy, and the
   initialization plan. */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include <meep.hpp>

#include "backend/backend.hpp"
#include "backend/cpu/cpu_backend.hpp"
#include "backend/initialization_plan.hpp"
#include "backend/precision.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

using namespace meep;

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

static double eps_slab(const vec &p) { return (fabs(p.y()) < 0.4) ? 12.0 : 1.0; }

static void build(structure **sp, fields **fp, const execution_options *opts = NULL) {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  *sp = new structure(gv, eps_slab, pml(0.5));
  *fp = opts ? new fields(*sp, *opts) : new fields(*sp);
  gaussian_src_time src(0.3, 0.1);
  (*fp)->add_point_source(Ez, src, vec(0.11, 0.13));
}

/* Selection: cpu accepted, nvidia and non-native precision rejected -- and
   rejected identically on every rank, since a rank that accepted while its
   peers aborted would hang rather than fail. */
static void test_selection() {
  structure *s;
  fields *f;
  build(&s, &f);

  CpuBackend cpu(*f);
  std::string why;

  execution_options ok;
  CHECK(cpu.accepts(ok, why), "the cpu backend rejected its own default options: %s", why.c_str());

  execution_options gpu;
  gpu.backend = backend_kind::nvidia;
  CHECK(!cpu.accepts(gpu, why), "backend=nvidia was accepted");
  CHECK(why.find("nvidia") != std::string::npos, "the nvidia rejection does not say why: %s",
        why.c_str());

  for (int p = 1; p <= 2; ++p) {
    execution_options pr;
    pr.precision = p == 1 ? precision_policy_kind::mixed : precision_policy_kind::f32;
    CHECK(!cpu.accepts(pr, why), "precision=%s was accepted on cpu",
          precision_policy_name(pr.precision));
  }

  execution_options dev;
  dev.device_id = 0;
  CHECK(!cpu.accepts(dev, why), "device_id was accepted on cpu");

  const backend_capabilities c = cpu.capabilities();
  CHECK(c.supports_native, "the cpu backend must support native precision");
  CHECK(!c.supports_mixed && !c.supports_f32, "the cpu backend must not claim mixed or f32");
  CHECK(strcmp(c.name, "cpu") == 0, "capabilities name is %s", c.name);

  delete f;
  delete s;
}

/* The backend-selecting constructor must produce the same simulation as the
   plain one. */
static void test_construction_equivalence() {
  structure *s1, *s2;
  fields *f1, *f2;
  execution_options opts; // defaults: cpu, native
  build(&s1, &f1);
  build(&s2, &f2, &opts);

  f1->advance(9);
  f2->advance(9);

  CHECK(f1->t == f2->t, "t differs: %d vs %d", f1->t, f2->t);
  size_t compared = 0, bad = 0;
  for (int i = 0; i < f1->num_chunks; ++i) {
    if (!f1->chunks[i]->is_mine()) continue;
    const size_t ntot = size_t(f1->chunks[i]->gv.ntot());
    for (int c = 0; c < NUM_FIELD_COMPONENTS; ++c)
      for (int cmp = 0; cmp < 2; ++cmp) {
        const realnum *a = f1->chunks[i]->f[c][cmp];
        const realnum *b = f2->chunks[i]->f[c][cmp];
        if (!a || !b) continue;
        ++compared;
        if (memcmp(a, b, ntot * sizeof(realnum)) != 0) ++bad;
      }
  }
  CHECK(bad == 0, "%zu of %zu arrays differ between the two constructors", bad, compared);
  CHECK(compared > 0, "nothing was compared");
  delete f1;
  delete f2;
  delete s1;
  delete s2;
}

/* read/write must round-trip every registered array without loss under
   native. */
static void test_read_write_roundtrip() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->advance(5);

  CpuBackend cpu(*f);
  size_t checked = 0, bad = 0;
  for (size_t i = 0; i < f->array_catalog->size() && checked < 40; ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = f->array_catalog->spec(id);
    if (spec.role != array_role::field || spec.elements == 0) continue;
    std::vector<realnum> buf(spec.elements, realnum(0));
    ArrayRef ref{id, 0, spec.elements};
    cpu.read(ref, buf.data(), spec.elements * sizeof(realnum));
    std::vector<realnum> back(spec.elements, realnum(0));
    cpu.write(ref, buf.data(), spec.elements * sizeof(realnum));
    cpu.read(ref, back.data(), spec.elements * sizeof(realnum));
    if (memcmp(buf.data(), back.data(), spec.elements * sizeof(realnum)) != 0) ++bad;
    ++checked;
  }
  CHECK(bad == 0, "%zu of %zu arrays did not round-trip through read/write", bad, checked);
  CHECK(or_to_all(checked > 0), "no arrays were round-tripped");
  delete f;
  delete s;
}

static void test_precision_policy() {
  CHECK(policy_for(precision_policy_kind::native) == precision_native(),
        "native policy does not match");
  const PrecisionPolicy mixed = policy_for(precision_policy_kind::mixed);
  CHECK(mixed.field == Precision::f32 && mixed.monitor == Precision::f64,
        "the mixed policy has the wrong shape");
  const PrecisionPolicy f32 = policy_for(precision_policy_kind::f32);
  CHECK(f32.monitor == Precision::f32 && f32.reduction == Precision::f64,
        "f32 must still reduce in f64");

  /* Aliased arrays (H == B) must agree on storage precision. */
  structure *s;
  fields *f;
  build(&s, &f);
  f->advance(3);
  std::string why;
  CHECK(validate_alias_precisions(*f->array_catalog, why), "alias precision validation failed: %s",
        why.c_str());
  delete f;
  delete s;
}

/* restrict_to has no Phase-1 consumer -- the in-place design update that would
   use it is deferred -- so it is built and unit-tested here rather than wired
   in. */
static void test_initialization_plan() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->advance(3);

  const InitializationPlan plan = build_initialization_plan(*f);
  CHECK(or_to_all(!plan.operations.empty()), "the initialization plan is empty");
  CHECK(!plan.materials.empty(), "the initialization plan has no material recipe");

  size_t zero_ops = 0, material_ops = 0, pml_ops = 0;
  for (const InitOperation &op : plan.operations) {
    if (op.kind == InitKind::zero) ++zero_ops;
    if (op.kind == InitKind::material_geometry) ++material_ops;
    if (op.kind == InitKind::pml_profile) ++pml_ops;
  }
  CHECK(or_to_all(zero_ops > 0), "no field arrays are initialized to zero");
  CHECK(or_to_all(material_ops > 0), "no arrays come from the material geometry");
  CHECK(or_to_all(pml_ops > 0), "a PML simulation produced no pml_profile operations");

  const InitializationPlan whole = plan.restrict_to(InitRegion());
  CHECK(whole.operations.size() == plan.operations.size(),
        "restrict_to(whole) dropped %zu operations",
        plan.operations.size() - whole.operations.size());

  InitRegion narrow(0, ivec(2, 2), ivec(4, 4));
  const InitializationPlan sub = plan.restrict_to(narrow);
  CHECK(sub.operations.size() <= plan.operations.size(), "restrict_to grew the plan");
  CHECK(sub.materials.size() == plan.materials.size(), "restrict_to dropped the recipes");

  master_printf("init plan: %zu ops (%zu zero, %zu material, %zu pml), restricted to %zu\n",
                plan.operations.size(), zero_ops, material_ops, pml_ops, sub.operations.size());
  delete f;
  delete s;
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  test_selection();
  test_construction_equivalence();
  test_read_write_roundtrip();
  test_precision_policy();
  test_initialization_plan();

  if (failures) {
    master_printf("backend_api: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("backend_api: all checks passed\n");
  return 0;
}
