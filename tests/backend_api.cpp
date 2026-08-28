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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <meep.hpp>

#include "backend/backend.hpp"
#include "backend/cpu/cpu_backend.hpp"
#include "backend/initialization_plan.hpp"
#include "backend/lifecycle.hpp"
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
static std::complex<double> initial_ez(const vec &) { return std::complex<double>(0.25, -0.5); }

struct lifetime_counts {
  int states_created;
  int states_destroyed;
  int executables_created;
  int executables_destroyed;
  int initialized;
  int classified;
  int finalized;
  int advanced;
  size_t arrays_at_create;

  lifetime_counts()
      : states_created(0), states_destroyed(0), executables_created(0),
        executables_destroyed(0), initialized(0), classified(0), finalized(0), advanced(0),
        arrays_at_create(0) {}
};

struct tracking_state : BackendState {
  explicit tracking_state(lifetime_counts &counts_) : counts(counts_) { ++counts.states_created; }
  ~tracking_state() override { ++counts.states_destroyed; }
  lifetime_counts &counts;
};

struct tracking_executable : Executable {
  explicit tracking_executable(lifetime_counts &counts_) : counts(counts_) {
    ++counts.executables_created;
  }
  ~tracking_executable() override { ++counts.executables_destroyed; }
  lifetime_counts &counts;
};

class tracking_backend : public ExecutionBackend {
public:
  tracking_backend(fields &f_, lifetime_counts &counts_) : f(f_), counts(counts_) {}

  BackendState *create_state(const StoragePlan &plan) override {
    counts.arrays_at_create = plan.arrays.size();
    return new tracking_state(counts);
  }
  void initialize(const InitializationPlan &, BackendState &) override { ++counts.initialized; }
  MaterialClassification classify_state(const StoragePlan &plan, BackendState &) override {
    ++counts.classified;
    return classify(f, plan);
  }
  void finalize_storage(const StoragePlan &, BackendState &) override { ++counts.finalized; }
  Executable *compile(const StepPlan &, BackendState &) override {
    return new tracking_executable(counts);
  }
  void advance(Executable &, BackendState &, int) override { ++counts.advanced; }
  void read(ArrayRef, void *, size_t) override {}
  void write(ArrayRef, const void *, size_t) override {}
  void synchronize() override {}
  backend_capabilities capabilities() const override {
    backend_capabilities c = {true, true, true, 0, "tracking"};
    return c;
  }
  bool requires_full_storage_preparation() const override { return true; }
  void prepare_state_rebuild(BackendState &, DirtyMask) override {}
  bool accepts(const execution_options &, std::string &) const override { return true; }

private:
  fields &f;
  lifetime_counts &counts;
};

struct rebuild_trace {
  std::vector<std::string> events;
  DirtyMask reasons;

  rebuild_trace() : reasons(dirty_none) {}
};

struct rebuild_state : BackendState {
  explicit rebuild_state(rebuild_trace &trace_) : trace(trace_) {}
  ~rebuild_state() override { trace.events.push_back("destroy-state"); }
  rebuild_trace &trace;
};

struct rebuild_executable : Executable {
  explicit rebuild_executable(rebuild_trace &trace_) : trace(trace_) {}
  ~rebuild_executable() override { trace.events.push_back("destroy-executable"); }
  rebuild_trace &trace;
};

class rebuild_backend_base : public ExecutionBackend {
public:
  explicit rebuild_backend_base(rebuild_trace &trace_) : trace(trace_) {}

  BackendState *create_state(const StoragePlan &) override {
    trace.events.push_back("create-state");
    return new rebuild_state(trace);
  }
  void initialize(const InitializationPlan &, BackendState &) override {}
  MaterialClassification classify_state(const StoragePlan &, BackendState &) override {
    return MaterialClassification();
  }
  void finalize_storage(const StoragePlan &, BackendState &) override {}
  Executable *compile(const StepPlan &, BackendState &) override {
    return new rebuild_executable(trace);
  }
  void advance(Executable &, BackendState &, int) override {}
  void read(ArrayRef, void *, size_t) override {}
  void write(ArrayRef, const void *, size_t) override {}
  void synchronize() override {}
  backend_capabilities capabilities() const override {
    backend_capabilities result;
    result.supports_native = true;
    result.supports_mixed = false;
    result.supports_f32 = false;
    result.memory_budget_bytes = 0;
    result.name = "tracking";
    return result;
  }
  bool requires_full_storage_preparation() const override { return true; }
  bool accepts(const execution_options &, std::string &) const override { return true; }

protected:
  rebuild_trace &trace;
};

class rebuild_tracking_backend : public rebuild_backend_base {
public:
  explicit rebuild_tracking_backend(rebuild_trace &trace_) : rebuild_backend_base(trace_) {}

  void prepare_state_rebuild(BackendState &, DirtyMask reasons) override {
    trace.events.push_back("prepare-rebuild");
    trace.reasons = reasons;
  }
};

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

  /* The backend API is typed by ArraySpec, not implicitly by realnum. Register
     one array of every representation and exercise a nonzero element offset. */
  realnum r[4] = {1, 2, 3, 4};
  std::complex<realnum> cr[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  double d[4] = {1, 2, 3, 4};
  std::complex<double> cd[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  int32_t i32[4] = {1, 2, 3, 4};
  size_t iz[4] = {1, 2, 3, 4};
  void *arrays[] = {r, cr, d, cd, i32, iz};
  const ElementType types[] = {ElementType::realnum_value, ElementType::complex_realnum,
                               ElementType::float64, ElementType::complex_float64,
                               ElementType::int32, ElementType::index};
  for (size_t k = 0; k < sizeof(types) / sizeof(types[0]); ++k) {
    const StorageKey key{-1, int(array_kind::num_kinds), -1, -1, int(k)};
    const ArrayId id =
        f->array_catalog->register_array(key, arrays[k], 4, array_role::scratch, types[k]);
    const size_t element_size = host_element_bytes(types[k]);
    std::vector<unsigned char> got(2 * element_size, 0);
    cpu.read(ArrayRef{id, 1, 2}, got.data(), got.size());
    CHECK(memcmp(got.data(), static_cast<unsigned char *>(arrays[k]) + element_size, got.size()) == 0,
          "typed access failed for ElementType %zu", k);
    std::vector<unsigned char> replacement(got.size(), 0x5a);
    cpu.write(ArrayRef{id, 1, 2}, replacement.data(), replacement.size());
    CHECK(memcmp(replacement.data(), static_cast<unsigned char *>(arrays[k]) + element_size,
                 replacement.size()) == 0,
          "typed write failed for ElementType %zu", k);
  }

  bool rejected = false;
  try {
    realnum one = 0;
    cpu.read(ArrayRef{invalid_array(), 0, 1}, &one, sizeof(one));
  }
  catch (const std::out_of_range &) { rejected = true; }
  CHECK(rejected, "an invalid ArrayId was not rejected");

  rejected = false;
  try {
    realnum one = 0;
    cpu.read(ArrayRef{ArrayId{0}, f->array_catalog->spec(ArrayId{0}).elements, 1}, &one,
             sizeof(one));
  }
  catch (const std::out_of_range &) { rejected = true; }
  CHECK(rejected, "an out-of-bounds ArrayRef was not rejected");

  rejected = false;
  try {
    realnum one = 0;
    cpu.read(ArrayRef{ArrayId{0}, 0, 1}, &one, sizeof(one) + 1);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  CHECK(rejected, "a mismatched host byte count was not rejected");
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

  StoragePlan plan;
  plan.arrays.push_back(ArraySpec{ArrayId{0}, array_role::field, ElementType::realnum_value,
                                  native_precision, 10, alignof(realnum), invalid_array(), false});
  plan.arrays.push_back(ArraySpec{ArrayId{1}, array_role::field, ElementType::realnum_value,
                                  native_precision, 10, alignof(realnum), ArrayId{0}, false});
  plan.arrays.push_back(ArraySpec{ArrayId{2}, array_role::material, ElementType::realnum_value,
                                  native_precision, 5, alignof(realnum), invalid_array(), true});
  plan.arrays.push_back(ArraySpec{ArrayId{3}, array_role::dft, ElementType::complex_realnum,
                                  native_precision, 2, alignof(realnum), invalid_array(), false});
  plan.arrays.push_back(ArraySpec{ArrayId{4}, array_role::field, ElementType::float64,
                                  native_precision, 3, alignof(double), invalid_array(), false});
  plan.arrays.push_back(ArraySpec{ArrayId{5}, array_role::scratch, ElementType::int32,
                                  native_precision, 4, alignof(int32_t), invalid_array(), false});
  apply_precision_policy(plan, precision_mixed());
  CHECK(plan.arrays[0].storage == Precision::f32 && plan.arrays[1].storage == Precision::f32,
        "mixed policy did not apply to field/alias storage");
  CHECK(plan.arrays[2].storage == Precision::f32, "mixed policy did not apply to material storage");
  CHECK(plan.arrays[3].storage == Precision::f64, "mixed policy did not preserve monitor precision");
  CHECK(plan.arrays[4].storage == Precision::f64, "fixed float64 storage was narrowed");
  CHECK(plan.provisional_peak_bytes() == 132,
        "precision-aware peak bytes are %zu, expected 132", plan.provisional_peak_bytes());
  CHECK(plan.steady_state_bytes() == 112,
        "precision-aware steady bytes are %zu, expected 112", plan.steady_state_bytes());
  CHECK(validate_alias_precisions(plan, why), "valid plan aliases were rejected: %s", why.c_str());
  plan.arrays[1].storage = Precision::f64;
  CHECK(!validate_alias_precisions(plan, why), "mismatched plan alias precision was accepted");

  ArraySpec huge = {ArrayId{0}, array_role::dft, ElementType::complex_float64, Precision::f64,
                    std::numeric_limits<size_t>::max(), alignof(double), invalid_array(), false};
  bool overflow_rejected = false;
  try {
    (void)storage_bytes(huge);
  }
  catch (const std::overflow_error &) { overflow_rejected = true; }
  CHECK(overflow_rejected, "overflowing storage byte count was accepted");
}

static void test_backend_lifecycle_epoch() {
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);

  f->advance(1);
  BackendState *first_state = f->backend_state;
  CHECK(counts.arrays_at_create > 0, "resident backend state was created from an empty plan");
  CHECK(counts.states_created == 1 && counts.initialized == 1 && counts.classified == 1 &&
            counts.finalized == 1 && counts.executables_created == 1 && counts.advanced == 1,
        "initial resident lifecycle counts are wrong");
  CHECK(!is_dirty(*f, dirty_initialization) && !is_dirty(*f, dirty_classification) &&
            !is_dirty(*f, dirty_executable),
        "resident lifecycle left consumed dirty bits set");

  invalidate(*f, MutationKind::material_values);
  f->advance(1);
  CHECK(f->backend_state == first_state, "value-only refresh reallocated resident state");
  CHECK(counts.states_created == 1 && counts.initialized == 2 && counts.classified == 2 &&
            counts.finalized == 2 && counts.executables_created == 1,
        "value-only refresh rebuilt the wrong backend artifacts");

  f->zero_fields();
  f->advance(1);
  CHECK(f->backend_state == first_state && counts.states_created == 1 && counts.initialized == 3,
        "field-value refresh did not preserve and reinitialize resident state");
  CHECK(counts.classified == 2 && counts.finalized == 2 && counts.executables_created == 1,
        "field-value refresh rebuilt unrelated backend artifacts");

  f->initialize_field(Ez, initial_ez);
  CHECK(is_dirty(*f, dirty_initialization),
        "initialize_field did not invalidate resident values");
  f->advance(1);
  CHECK(f->backend_state == first_state && counts.states_created == 1 && counts.initialized == 4,
        "initialize_field refresh did not preserve and reinitialize resident state");

  invalidate(*f, MutationKind::field_layout);
  f->advance(1);
  CHECK(counts.states_created == 2 && counts.states_destroyed == 1,
        "storage invalidation did not replace resident state");
  CHECK(counts.executables_created == 2 && counts.executables_destroyed == 1,
        "executable invalidation did not replace the compiled artifact");

  delete f;
  CHECK(counts.states_destroyed == 2 && counts.executables_destroyed == 2,
        "polymorphic backend artifacts were not destroyed exactly once");
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

/* A storage rebuild must give the backend a chance to preserve authoritative
   values before either its executable or state is destroyed. A refusal leaves
   both objects live. */
static void test_authority_safe_state_rebuild() {
  structure *s;
  fields *f;
  build(&s, &f);

  rebuild_trace trace;
  rebuild_backend_base *tracking = new rebuild_tracking_backend(trace);
  f->backend = tracking;
  f->backend_state = tracking->create_state(*f->storage_plan);
  StepPlan plan;
  f->executable = tracking->compile(plan, *f->backend_state);
  trace.events.clear();
  clear_dirty(*f, DirtyMask(f->dirty_mask));
  invalidate(*f, MutationKind::field_layout);

  f->init_backend();
  CHECK(trace.events.size() == 4, "rebuild produced %zu events, expected 4", trace.events.size());
  if (trace.events.size() == 4) {
    CHECK(trace.events[0] == "prepare-rebuild", "first rebuild event is %s",
          trace.events[0].c_str());
    CHECK(trace.events[1] == "destroy-executable", "second rebuild event is %s",
          trace.events[1].c_str());
    CHECK(trace.events[2] == "destroy-state", "third rebuild event is %s",
          trace.events[2].c_str());
    CHECK(trace.events[3] == "create-state", "fourth rebuild event is %s",
          trace.events[3].c_str());
  }
  CHECK((trace.reasons & dirty_storage) != 0, "rebuild hook did not receive dirty_storage");

  delete f;
  delete s;

  build(&s, &f);
  rebuild_trace refused;
  tracking = new rebuild_backend_base(refused);
  f->backend = tracking;
  f->backend_state = tracking->create_state(*f->storage_plan);
  BackendState *live_state = f->backend_state;
  refused.events.clear();
  clear_dirty(*f, DirtyMask(f->dirty_mask));
  invalidate(*f, MutationKind::field_layout);
  bool rejected = false;
  try { f->init_backend(); }
  catch (const std::logic_error &) { rejected = true; }
  CHECK(rejected, "backend rebuild refusal did not propagate");
  CHECK(f->backend_state == live_state, "a refused rebuild destroyed the live state");
  CHECK(f->executable == NULL, "refused rebuild unexpectedly created an executable");
  CHECK(refused.events.empty(), "default-safe rebuild destroyed or replaced the live state");

  clear_dirty(*f, DirtyMask(f->dirty_mask));
  delete f;
  delete s;
}

static void test_cpu_state_rebuild_is_safe_noop() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->advance(1);
  CHECK(f->backend_state != NULL && f->executable != NULL,
        "CPU advance did not create backend artifacts");

  invalidate(*f, MutationKind::field_layout);
  f->init_backend();
  CHECK(f->backend_state != NULL, "CPU state rebuild did not create a replacement state");
  CHECK(f->executable == NULL, "CPU state rebuild retained a stale executable");

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
  test_backend_lifecycle_epoch();
  test_initialization_plan();
  test_authority_safe_state_rebuild();
  test_cpu_state_rebuild_is_safe_noop();

  if (failures) {
    master_printf("backend_api: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("backend_api: all checks passed\n");
  return 0;
}
