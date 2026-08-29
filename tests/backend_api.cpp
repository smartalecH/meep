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
#include <stdlib.h>
#include <string.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include <meep.hpp>

#include "config.h"
#include "backend/backend.hpp"
#include "backend/cpu/cpu_backend.hpp"
#include "backend/initialization_plan.hpp"
#include "backend/lifecycle.hpp"
#include "backend/prepare.hpp"
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
static double unit_epsilon(const vec &) { return 1.0; }
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
  int reads;
  int writes;
  int rebuilds;
  int magnetic_synchronizes;
  int magnetic_restores;
  size_t arrays_at_create;
  size_t polarization_arrays_at_create;
  size_t polarization_updates_at_compile;
  size_t polarization_subtractions_at_compile;
  size_t beta_updates_at_compile;
  size_t bfast_updates_at_compile;
  size_t cylindrical_m_updates_at_compile;
  size_t cylindrical_origin_actions_at_compile;
  bool gyrotropic_update_at_compile;
  bool polarization_zero_at_create;
  bool connections_current_at_create;
  bool rebuild_saw_live_imaginary;
  bool fail_rebuild;
  bool fail_compile;
  bool fail_magnetic_synchronize;
  bool fail_magnetic_restore;
  bool fail_magnetic_synchronize_dispatch;

  lifetime_counts()
      : states_created(0), states_destroyed(0), executables_created(0), executables_destroyed(0),
        initialized(0), classified(0), finalized(0), advanced(0), reads(0), writes(0), rebuilds(0),
        magnetic_synchronizes(0), magnetic_restores(0), arrays_at_create(0),
        polarization_arrays_at_create(0), polarization_updates_at_compile(0),
        polarization_subtractions_at_compile(0), beta_updates_at_compile(0),
        bfast_updates_at_compile(0), cylindrical_m_updates_at_compile(0),
        cylindrical_origin_actions_at_compile(0), gyrotropic_update_at_compile(false),
        polarization_zero_at_create(true), connections_current_at_create(false),
        rebuild_saw_live_imaginary(false), fail_rebuild(false), fail_compile(false),
        fail_magnetic_synchronize(false), fail_magnetic_restore(false),
        fail_magnetic_synchronize_dispatch(false) {}
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
  tracking_backend(fields &f_, lifetime_counts &counts_, bool magnetic_supported_ = false)
      : f(f_), counts(counts_), magnetic_supported(magnetic_supported_) {}

  BackendState *create_state(const StoragePlan &plan) override {
    counts.arrays_at_create = plan.arrays.size();
    counts.connections_current_at_create = connections_are_current(f);
    for (size_t i = 0; i < plan.arrays.size(); ++i) {
      const ArraySpec &spec = plan.arrays[i];
      if (spec.role != array_role::polarization || is_valid(spec.alias_of)) continue;
      ++counts.polarization_arrays_at_create;
      const realnum *values = f.array_catalog->resolve<realnum>(spec.id);
      for (size_t j = 0; j < spec.elements; ++j)
        if (values[j] != realnum(0)) counts.polarization_zero_at_create = false;
    }
    return new tracking_state(counts);
  }
  void initialize(const InitializationPlan &, BackendState &) override { ++counts.initialized; }
  MaterialClassification classify_state(const StoragePlan &plan, BackendState &) override {
    ++counts.classified;
    return classify(f, plan);
  }
  void finalize_storage(const StoragePlan &, BackendState &) override { ++counts.finalized; }
  Executable *compile(const StepPlan &plan, BackendState &) override {
    if (counts.fail_compile) throw std::runtime_error("injected executable compilation failure");
    counts.polarization_updates_at_compile = plan.polarization_updates.size();
    counts.polarization_subtractions_at_compile = plan.polarization_subtractions.size();
    counts.beta_updates_at_compile = plan.beta_updates.size();
    counts.bfast_updates_at_compile = plan.bfast_updates.size();
    counts.cylindrical_m_updates_at_compile = plan.cylindrical_m_updates.size();
    counts.cylindrical_origin_actions_at_compile = plan.cylindrical_origin_actions.size();
    for (const PolarizationUpdate &update : plan.polarization_updates)
      if (update.kind == PolarizationUpdateKind::gyrotropic)
        counts.gyrotropic_update_at_compile = true;
    return new tracking_executable(counts);
  }
  void advance(Executable &, BackendState &, int) override { ++counts.advanced; }
  void read(ArrayRef, void *, size_t) override { ++counts.reads; }
  void write(ArrayRef, const void *, size_t) override { ++counts.writes; }
  void synchronize() override {}
  bool supports_magnetic_synchronization() const override { return magnetic_supported; }
  void preflight_magnetic_transition(Executable &, BackendState &, bool synchronize) override {
    if (synchronize && counts.fail_magnetic_synchronize)
      throw std::runtime_error("injected magnetic synchronize failure");
    if (!synchronize && counts.fail_magnetic_restore)
      throw std::runtime_error("injected magnetic restore failure");
  }
  void synchronize_magnetic_fields(Executable &, BackendState &) override {
    if (counts.fail_magnetic_synchronize_dispatch)
      throw std::runtime_error("injected magnetic synchronize dispatch failure");
    ++counts.magnetic_synchronizes;
  }
  void restore_magnetic_fields(Executable &, BackendState &) override {
    ++counts.magnetic_restores;
  }
  backend_capabilities capabilities() const override {
    backend_capabilities c = {true, true, true, 0, "tracking"};
    return c;
  }
  bool requires_full_storage_preparation() const override { return true; }
  void prepare_state_rebuild(BackendState &, DirtyMask) override {
    ++counts.rebuilds;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
      if (!f.chunks[chunk] || !f.chunks[chunk]->is_mine()) continue;
      FOR_COMPONENTS(c) {
        if (!f.chunks[chunk]->f[c][1]) continue;
        const ArrayId id = f.array_catalog->find({chunk, int(array_kind::f), int(c), 1, 0});
        if (is_valid(id) && f.array_catalog->resolve<realnum>(id) == f.chunks[chunk]->f[c][1])
          counts.rebuild_saw_live_imaginary = true;
      }
    }
    if (counts.fail_rebuild) throw std::runtime_error("injected layout migration failure");
  }
  bool accepts(const execution_options &, std::string &) const override { return true; }

private:
  fields &f;
  lifetime_counts &counts;
  bool magnetic_supported;
};

static void build(structure **sp, fields **fp, const execution_options *opts = NULL);

static void test_resident_magnetic_dispatch() {
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts, true);

  counts.fail_compile = my_rank() == 0;
  bool compile_failed = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    compile_failed = true;
  }
  CHECK(sum_to_all(int(compile_failed)) == count_processors(),
        "rank-asymmetric magnetic compile failure was not reconciled");
  CHECK(counts.magnetic_synchronizes == 0,
        "magnetic compile failure entered the synchronization dispatch");
  CHECK(!f->executable && counts.executables_created == counts.executables_destroyed,
        "failed first compile retained a partial executable");
  counts.fail_compile = false;

  f->synchronize_magnetic_fields();
  CHECK(counts.states_created == 1 &&
            counts.executables_created == counts.executables_destroyed + 1,
        "pre-step resident sync did not prepare state and executable");
  CHECK(counts.magnetic_synchronizes == 1 && counts.magnetic_restores == 0,
        "first resident sync dispatched the wrong transition");
  for (int chunk = 0; chunk < f->num_chunks; ++chunk)
    if (f->chunks[chunk]->is_mine()) DOCMP2 FOR_COMPONENTS(c) {
        CHECK(!f->chunks[chunk]->f_backup[c][cmp] && !f->chunks[chunk]->f_u_backup[c][cmp] &&
                  !f->chunks[chunk]->f_w_backup[c][cmp] &&
                  !f->chunks[chunk]->f_cond_backup[c][cmp] &&
                  !f->chunks[chunk]->f_bfast_backup[c][cmp],
              "resident sync allocated legacy host backup storage");
      }

  f->synchronize_magnetic_fields();
  CHECK(counts.magnetic_synchronizes == 1, "nested resident sync performed work");
  f->restore_magnetic_fields();
  CHECK(counts.magnetic_restores == 0, "inner resident restore performed work");

  BackendState *state = f->backend_state;
  Executable *executable = f->executable;
  bool rebuild_rejected = false;
  try {
    backend_prepare_field_layout_change(*f, dirty_storage, "magnetic rebuild test");
  }
  catch (const std::runtime_error &) {
    rebuild_rejected = true;
  }
  CHECK(sum_to_all(int(rebuild_rejected)) == count_processors(),
        "live resident magnetic snapshot did not reject a state rebuild collectively");
  CHECK(f->backend_state == state && f->executable == executable,
        "rejected magnetic rebuild changed state or executable");

  counts.fail_magnetic_restore = my_rank() == 0;
  bool restore_failed = false;
  try {
    f->restore_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    restore_failed = true;
  }
  CHECK(sum_to_all(int(restore_failed)) == count_processors(),
        "resident magnetic restore failure was not reconciled");
  CHECK(counts.magnetic_restores == 0, "failed resident restore completed the transition");
  counts.fail_magnetic_restore = false;
  f->restore_magnetic_fields();
  CHECK(counts.magnetic_restores == 1, "final resident restore did not dispatch exactly once");
  f->restore_magnetic_fields();
  CHECK(counts.magnetic_restores == 1, "unmatched resident restore performed work");

  Executable *compiled = f->executable;
  const int created_before_recompile = counts.executables_created;
  const int destroyed_before_recompile = counts.executables_destroyed;
  f->dirty_mask |= dirty_executable;
  counts.fail_compile = my_rank() == 0;
  compile_failed = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    compile_failed = true;
  }
  CHECK(sum_to_all(int(compile_failed)) == count_processors(),
        "rank-asymmetric replacement compile failure was not reconciled");
  CHECK(f->executable == compiled && is_dirty(*f, dirty_executable),
        "replacement compile failure lost the prior executable or dirty bit");
  CHECK(counts.executables_created - created_before_recompile ==
            counts.executables_destroyed - destroyed_before_recompile,
        "replacement compile failure leaked a candidate executable");
  CHECK(counts.magnetic_synchronizes == 1,
        "replacement compile failure entered the synchronization dispatch");
  counts.fail_compile = false;
  f->synchronize_magnetic_fields();
  CHECK(!is_dirty(*f, dirty_executable) &&
            counts.executables_created == created_before_recompile + 1 + (my_rank() != 0) &&
            counts.executables_destroyed == destroyed_before_recompile + 1 + (my_rank() != 0),
        "replacement compile retry did not replace exactly one live executable");
  f->restore_magnetic_fields();

  counts.fail_magnetic_synchronize = my_rank() == 0;
  bool sync_failed = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    sync_failed = true;
  }
  CHECK(sum_to_all(int(sync_failed)) == count_processors(),
        "resident magnetic synchronize failure was not reconciled");
  CHECK(counts.magnetic_synchronizes == 2, "failed resident sync completed the transition");
  counts.fail_magnetic_synchronize = false;
  f->synchronize_magnetic_fields();
  f->restore_magnetic_fields();
  CHECK(counts.magnetic_synchronizes == 3 && counts.magnetic_restores == 3,
        "resident magnetic transition did not recover after injected failures");

  counts.fail_magnetic_synchronize_dispatch = my_rank() == 0;
  bool dispatch_failed = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    dispatch_failed = true;
  }
  CHECK(sum_to_all(int(dispatch_failed)) == count_processors(),
        "rank-asymmetric dispatch failure was not reconciled");
  CHECK(f->backend->is_poisoned(), "dispatch failure did not poison every resident backend");
  bool poison_rejected = false;
  try {
    f->restore_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    poison_rejected = true;
  }
  CHECK(sum_to_all(int(poison_rejected)) == count_processors(),
        "poisoned resident backend accepted another magnetic transition");
  bool layout_rejected = false;
  try {
    backend_prepare_field_layout_change(*f, dirty_storage, "poisoned magnetic test");
  }
  catch (const std::runtime_error &) {
    layout_rejected = true;
  }
  CHECK(sum_to_all(int(layout_rejected)) == count_processors(),
        "poisoned resident backend accepted a layout rebuild");
  bool advance_rejected = false;
  try {
    f->advance(1);
  }
  catch (const std::runtime_error &) {
    advance_rejected = true;
  }
  CHECK(sum_to_all(int(advance_rejected)) == count_processors(),
        "poisoned resident backend accepted an ordinary advance");

  delete f;
  delete s;
}

struct rebuild_trace {
  std::vector<std::string> events;
  DirtyMask reasons;

  rebuild_trace() : reasons(dirty_none) {}
};

struct access_trace {
  size_t reads;
  size_t writes;
  size_t field_reads;
  size_t dft_reads;
  size_t field_writes;
  size_t dft_writes;
  size_t max_elements;
  int prepare_rebuilds;
  int fail_read_rank;
  int fail_write_rank;
  int fail_dft_read_rank;
  int fail_dft_write_rank;

  access_trace()
      : reads(0), writes(0), field_reads(0), dft_reads(0), field_writes(0), dft_writes(0),
        max_elements(0), prepare_rebuilds(0), fail_read_rank(-1), fail_write_rank(-1),
        fail_dft_read_rank(-1), fail_dft_write_rank(-1) {}
};

struct compact_trace {
  size_t calls;
  size_t returned_bytes;
  int fail_rank;
  int fail_call;
  std::vector<DftReductionRequest> requests;

  compact_trace() : calls(0), returned_bytes(0), fail_rank(-1), fail_call(-1) {}
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
  rebuild_backend_base(fields &f_, rebuild_trace &trace_) : f(f_), trace(trace_) {}

  BackendState *create_state(const StoragePlan &) override {
    trace.events.push_back("create-state");
    return new rebuild_state(trace);
  }
  void initialize(const InitializationPlan &, BackendState &) override {}
  MaterialClassification classify_state(const StoragePlan &plan, BackendState &) override {
    return classify(f, plan);
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
  fields &f;
  rebuild_trace &trace;
};

class rebuild_tracking_backend : public rebuild_backend_base {
public:
  rebuild_tracking_backend(fields &f, rebuild_trace &trace_) : rebuild_backend_base(f, trace_) {}

  void prepare_state_rebuild(BackendState &, DirtyMask reasons) override {
    trace.events.push_back("prepare-rebuild");
    trace.reasons = reasons;
  }
};

class access_tracking_backend : public rebuild_backend_base {
public:
  access_tracking_backend(fields &f, rebuild_trace &rebuilds, access_trace &accesses_)
      : rebuild_backend_base(f, rebuilds), accesses(accesses_) {}
  void advance(Executable &, BackendState &, int num_steps) override { f.t += num_steps; }

  void read(ArrayRef ref, void *host_buffer, size_t bytes) override {
    const ArraySpec &spec = f.array_catalog->spec(ref.id);
    if (my_rank() == accesses.fail_read_rank ||
        (spec.role == array_role::dft && my_rank() == accesses.fail_dft_read_rank))
      throw std::runtime_error("injected rank-local backend read failure");
    CHECK(bytes == ref.elements * host_element_bytes(spec.element_type),
          "backend host-range read byte count is inconsistent");
    ++accesses.reads;
    accesses.field_reads += spec.role == array_role::field;
    accesses.dft_reads += spec.role == array_role::dft;
    if (ref.elements > accesses.max_elements) accesses.max_elements = ref.elements;
    if (spec.element_type == ElementType::realnum_value) {
      realnum *values = static_cast<realnum *>(host_buffer);
      for (size_t i = 0; i < ref.elements; ++i)
        values[i] = realnum(2.5);
    }
    else if (spec.element_type == ElementType::complex_realnum) {
      std::complex<realnum> *values = static_cast<std::complex<realnum> *>(host_buffer);
      for (size_t i = 0; i < ref.elements; ++i)
        values[i] = std::complex<realnum>(realnum(1.25), realnum(-0.75));
    }
  }

  void write(ArrayRef ref, const void *, size_t bytes) override {
    const ArraySpec &spec = f.array_catalog->spec(ref.id);
    if (my_rank() == accesses.fail_write_rank ||
        (spec.role == array_role::dft && my_rank() == accesses.fail_dft_write_rank))
      throw std::runtime_error("injected rank-local backend write failure");
    CHECK(bytes == ref.elements * host_element_bytes(spec.element_type),
          "backend host-range write byte count is inconsistent");
    ++accesses.writes;
    accesses.field_writes += spec.role == array_role::field;
    accesses.dft_writes += spec.role == array_role::dft;
    if (ref.elements > accesses.max_elements) accesses.max_elements = ref.elements;
  }

  void prepare_state_rebuild(BackendState &, DirtyMask reasons) override {
    CHECK((reasons & dirty_storage) != 0,
          "checkpoint replacement did not request storage-safe teardown");
    ++accesses.prepare_rebuilds;
  }

private:
  access_trace &accesses;
};

class compact_access_backend : public access_tracking_backend {
public:
  compact_access_backend(fields &f_, rebuild_trace &rebuilds, access_trace &accesses,
                         compact_trace &compact_)
      : access_tracking_backend(f_, rebuilds, accesses), compact(compact_) {}

  bool supports_compact_dft_reductions() const override { return true; }

  void reduce_dft(const DftReductionRequest &request, std::complex<double> *result,
                  size_t result_count) override {
    const size_t call = compact.calls++;
    if (my_rank() == compact.fail_rank && int(call) == compact.fail_call)
      throw std::runtime_error("injected rank-local compact DFT reduction failure");
    if (request.result_count != result_count)
      throw std::invalid_argument("compact result-count mismatch");
    if ((request.kind == DftReductionKind::norm2) != (result_count == 1) ||
        (request.kind != DftReductionKind::norm2 && result_count == 0))
      throw std::invalid_argument("compact kind/result-count mismatch");
    if (request.accumulation_precision != policy_for(f.options.precision).reduction)
      throw std::invalid_argument("compact reduction precision mismatch");

    for (size_t ti = 0; ti < request.terms.size(); ++ti) {
      const DftReductionTerm &term = request.terms[ti];
      if (!is_valid(term.left) || term.left.value >= f.array_catalog->size())
        throw std::out_of_range("invalid compact left array");
      const ArraySpec &left = f.array_catalog->spec(term.left);
      if (left.role != array_role::dft || left.element_type != ElementType::complex_realnum ||
          is_valid(left.alias_of))
        throw std::invalid_argument("compact left array has the wrong storage kind");
      if (!term.storage_points || !term.frequencies ||
          term.storage_points > std::numeric_limits<size_t>::max() / term.frequencies ||
          term.storage_points * term.frequencies > left.elements)
        throw std::out_of_range("compact left array is too small");
      if (request.kind == DftReductionKind::norm2) {
        if (is_valid(term.right)) throw std::invalid_argument("norm request has a right array");
      }
      else {
        if (!is_valid(term.right) || term.right.value >= f.array_catalog->size())
          throw std::out_of_range("invalid compact right array");
        const ArraySpec &right = f.array_catalog->spec(term.right);
        if (right.role != array_role::dft || right.element_type != ElementType::complex_realnum ||
            is_valid(right.alias_of) || right.storage != left.storage ||
            term.storage_points * term.frequencies > right.elements)
          throw std::invalid_argument("compact DFT pair has incompatible storage");
        if (term.frequencies != result_count)
          throw std::invalid_argument("compact product frequency mismatch");
      }
      size_t maximum = term.region.base;
      for (int axis = 0; axis < 3; ++axis) {
        if (!term.region.counts[axis])
          throw std::invalid_argument("compact region has a zero count");
        const size_t extent = term.region.counts[axis] - 1;
        if (term.region.strides[axis] &&
            extent > (std::numeric_limits<size_t>::max() - maximum) / term.region.strides[axis])
          throw std::overflow_error("compact region overflows");
        maximum += extent * term.region.strides[axis];
      }
      if (maximum >= term.storage_points) throw std::out_of_range("compact region exceeds storage");
    }

    compact.requests.push_back(request);
    compact.returned_bytes += result_count * sizeof(std::complex<double>);
    for (size_t i = 0; i < result_count; ++i) {
      const double value = double((my_rank() + 1) * (i + 1));
      result[i] += request.kind == DftReductionKind::complex_weighted_product
                       ? std::complex<double>(value, -0.25 * value)
                       : std::complex<double>(value, 0.0);
    }
  }

private:
  compact_trace &compact;
};

static void build(structure **sp, fields **fp, const execution_options *opts) {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  *sp = new structure(gv, eps_slab, pml(0.5));
  *fp = opts ? new fields(*sp, *opts) : new fields(*sp);
  gaussian_src_time src(0.3, 0.1);
  (*fp)->add_point_source(Ez, src, vec(0.11, 0.13));
}

static double phase_conductivity(const vec &) { return 0.2; }

static void test_material_phase_transaction() {
  structure *current;
  fields *f;
  build(&current, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts, true);
  f->advance(1);
  BackendState *initial_state = f->backend_state;
  Executable *initial_executable = f->executable;

  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure target_a(gv, eps_slab, pml(0.5));
  structure target_b(gv, unit_epsilon, pml(0.5));
  target_a.set_conductivity(Dz, phase_conductivity);
  target_b.set_conductivity(Dz, phase_conductivity);
  std::vector<int> refs_a(size_t(target_a.num_chunks), 0);
  std::vector<int> refs_b(size_t(target_b.num_chunks), 0);
  for (int i = 0; i < target_a.num_chunks; ++i) {
    refs_a[size_t(i)] = target_a.chunks[i]->refcount;
    refs_b[size_t(i)] = target_b.chunks[i]->refcount;
  }

  const uint32_t initial_dirty = f->dirty_mask;
  const uint64_t initial_generation = generation(*f, MutationKind::material_phase);
  const uint64_t initial_local_generation = f->local_invalidation_generation;
  const int steps = f->phase_in_material(&target_a, 2.25 * f->dt);
  CHECK(steps == 2 && f->phasein_time == 2, "material phase did not preserve integer truncation");
  CHECK(!f->backend_state && !f->executable,
        "successful material setup retained the old resident representation");
  CHECK(counts.rebuilds == 1 && counts.states_destroyed == 1 &&
            counts.executables_destroyed == 1,
        "material setup did not migrate then retire exactly one resident representation");
  CHECK(initial_state != f->backend_state && initial_executable != f->executable,
        "material setup did not clear resident artifacts");
  CHECK(f->dirty_mask == (initial_dirty | invalidation_closure(MutationKind::material_phase)) &&
            generation(*f, MutationKind::material_phase) == initial_generation + 1 &&
            f->local_invalidation_generation == initial_local_generation + 1,
        "successful material setup did not commit the exact lifecycle closure once");
  for (int i = 0; i < f->num_chunks; ++i) {
    if (!f->chunks[i]->is_mine()) continue;
    CHECK(f->chunks[i]->new_s == target_a.chunks[i],
          "material setup did not install the requested target");
    CHECK(target_a.chunks[i]->refcount == refs_a[size_t(i)] + 1,
          "material setup did not retain exactly one target reference");
    CHECK(f->chunks[i]->s->refcount == 1,
          "material setup did not install detached current storage");
    CHECK(f->chunks[i]->s->conductivity[Dz][Z] && f->chunks[i]->s->condinv[Dz][Z],
          "material setup did not realize the stable conductivity union");
  }

  f->init_backend();
  BackendState *live_state = f->backend_state;
  Executable *live_executable = f->executable;
  std::vector<structure_chunk *> current_rows(size_t(f->num_chunks), NULL);
  for (int i = 0; i < f->num_chunks; ++i) current_rows[size_t(i)] = f->chunks[i]->s;
  const uint32_t dirty_before = f->dirty_mask;
  const uint64_t generation_before = generation(*f, MutationKind::material_phase);
  const int countdown_before = f->phasein_time;
  counts.fail_rebuild = my_rank() == 0;
  bool rejected = false;
  try {
    f->phase_in_material(&target_b, 3.5 * f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "rank-asymmetric material rebuild failure was not reconciled");
  CHECK(f->backend_state == live_state && f->executable == live_executable,
        "failed material setup retired the live resident representation");
  CHECK(f->dirty_mask == dirty_before &&
            generation(*f, MutationKind::material_phase) == generation_before &&
            f->phasein_time == countdown_before,
        "failed material setup changed lifecycle state or countdown");
  for (int i = 0; i < f->num_chunks; ++i) {
    CHECK(f->chunks[i]->s == current_rows[size_t(i)],
          "failed material setup rebound current structure storage");
    if (!f->chunks[i]->is_mine()) continue;
    CHECK(f->chunks[i]->new_s == target_a.chunks[i],
          "failed material setup replaced the prior target");
    CHECK(target_a.chunks[i]->refcount == refs_a[size_t(i)] + 1 &&
              target_b.chunks[i]->refcount == refs_b[size_t(i)],
          "failed material setup leaked staged target ownership");
  }

  counts.fail_rebuild = false;
  CHECK(f->phase_in_material(&target_b, 3.5 * f->dt) == 3,
        "material setup retry did not succeed");
  for (int i = 0; i < f->num_chunks; ++i) {
    if (!f->chunks[i]->is_mine()) continue;
    CHECK(f->chunks[i]->new_s == target_b.chunks[i],
          "material retry did not install replacement target");
    CHECK(target_a.chunks[i]->refcount == refs_a[size_t(i)] &&
              target_b.chunks[i]->refcount == refs_b[size_t(i)] + 1,
          "material target replacement has incorrect net ownership");
  }
  CHECK(f->phase_in_material(&target_b, 0.25 * f->dt) == 0 && !f->is_phasing(),
        "zero-step material phase no longer preserves legacy truncation");
  for (int i = 0; i < f->num_chunks; ++i)
    if (f->chunks[i]->is_mine())
      CHECK(target_b.chunks[i]->refcount == refs_b[size_t(i)] + 1,
            "same-target material setup changed net ownership");

  structure incompatible(vol2d(3.25, 3.0, 10.0), unit_epsilon, pml(0.5));
  const int stable_countdown = f->phasein_time;
  const uint32_t topology_dirty = f->dirty_mask;
  const uint64_t topology_generation = generation(*f, MutationKind::material_phase);
  const uint64_t topology_local_generation = f->local_invalidation_generation;
  const bool topology_changed_materials = legacy_material_change_pending(*f);
  rejected = false;
  try {
    f->phase_in_material(&incompatible, f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "incompatible material target was not rejected collectively");
  CHECK(f->phasein_time == stable_countdown && f->dirty_mask == topology_dirty &&
            generation(*f, MutationKind::material_phase) == topology_generation &&
            f->local_invalidation_generation == topology_local_generation &&
            legacy_material_change_pending(*f) == topology_changed_materials,
        "incompatible target changed setup or lifecycle state");
  for (int i = 0; i < f->num_chunks; ++i)
    if (f->chunks[i]->is_mine())
      CHECK(f->chunks[i]->new_s == target_b.chunks[i] &&
                target_b.chunks[i]->refcount == refs_b[size_t(i)] + 1,
            "incompatible target changed target ownership");

  structure chunk_mismatch(gv, unit_epsilon, pml(0.5));
  if (my_rank() == 0 && chunk_mismatch.num_chunks)
    chunk_mismatch.chunks[0]->a += 1.0;
  rejected = false;
  const uint32_t chunk_dirty = f->dirty_mask;
  const uint64_t chunk_generation = generation(*f, MutationKind::material_phase);
  const uint64_t chunk_local_generation = f->local_invalidation_generation;
  const bool chunk_changed_materials = legacy_material_change_pending(*f);
  try {
    f->phase_in_material(&chunk_mismatch, f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "rank-asymmetric chunk-topology mismatch was not rejected collectively");
  CHECK(f->phasein_time == stable_countdown && f->dirty_mask == chunk_dirty &&
            generation(*f, MutationKind::material_phase) == chunk_generation &&
            f->local_invalidation_generation == chunk_local_generation &&
            legacy_material_change_pending(*f) == chunk_changed_materials,
        "chunk-topology rejection changed setup or lifecycle state");

  f->init_backend();
  live_state = f->backend_state;
  live_executable = f->executable;
  structure allocation_failure_target(gv, eps_slab, pml(0.5));
  for (int i = 0; i < allocation_failure_target.num_chunks; ++i) {
    if (!allocation_failure_target.chunks[i]->is_mine()) continue;
    structure_chunk &target = *allocation_failure_target.chunks[i];
    const size_t n = size_t(target.gv.ntot());
    delete[] target.chi1inv[Ex][Y];
    target.chi1inv[Ex][Y] = new realnum[n];
    std::fill(target.chi1inv[Ex][Y], target.chi1inv[Ex][Y] + n, realnum(0.25));
    target.trivial_chi1inv[Ex][Y] = false;
  }
  int owns_chunk = 0;
  for (int i = 0; i < f->num_chunks; ++i) owns_chunk |= f->chunks[i]->is_mine();
  const int failure_rank = min_to_all(owns_chunk ? my_rank() : count_processors());
  set_material_phase_prepare_failure_for_testing(failure_rank, 1);
  const uint32_t allocation_dirty = f->dirty_mask;
  const uint64_t allocation_generation = generation(*f, MutationKind::material_phase);
  const uint64_t allocation_local_generation = f->local_invalidation_generation;
  const bool allocation_changed_materials = legacy_material_change_pending(*f);
  const int allocation_countdown = f->phasein_time;
  std::vector<structure_chunk *> allocation_current(size_t(f->num_chunks), NULL);
  std::vector<int> allocation_target_refs(size_t(f->num_chunks), 0);
  for (int i = 0; i < f->num_chunks; ++i) {
    allocation_current[size_t(i)] = f->chunks[i]->s;
    allocation_target_refs[size_t(i)] = allocation_failure_target.chunks[i]->refcount;
  }
  rejected = false;
  try {
    f->phase_in_material(&allocation_failure_target, 2 * f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  set_material_phase_prepare_failure_for_testing(-1, -1);
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "rank-asymmetric material storage failure was not reconciled");
  CHECK(f->backend_state == live_state && f->executable == live_executable &&
            f->dirty_mask == allocation_dirty &&
            generation(*f, MutationKind::material_phase) == allocation_generation &&
            f->local_invalidation_generation == allocation_local_generation &&
            legacy_material_change_pending(*f) == allocation_changed_materials &&
            f->phasein_time == allocation_countdown,
        "failed material storage preparation changed resident or lifecycle state");
  for (int i = 0; i < f->num_chunks; ++i) {
    CHECK(f->chunks[i]->s == allocation_current[size_t(i)],
          "failed material storage preparation changed current storage");
    CHECK(allocation_failure_target.chunks[i]->refcount ==
              allocation_target_refs[size_t(i)],
          "failed material storage preparation leaked a target reference");
    if (f->chunks[i]->is_mine())
      CHECK(f->chunks[i]->new_s == target_b.chunks[i],
            "failed material storage preparation changed the attached target");
  }

  delete f;
  delete current;

  grid_volume cyl_gv = volcyl(2.0, 3.0, 10.0);
  structure cyl_current(cyl_gv, unit_epsilon, no_pml());
  structure cyl_target(cyl_gv, eps_slab, no_pml());
  fields cyl(&cyl_current, 1.0);
  CHECK(cyl.phase_in_material(&cyl_target, cyl.dt) == 1,
        "valid cylindrical target was rejected after symmetry augmentation");

  build(&current, &f);
  lifetime_counts magnetic_counts;
  f->backend = new tracking_backend(*f, magnetic_counts, true);
  f->advance(1);
  structure magnetic_target(gv, eps_slab, pml(0.5));
  std::vector<int> magnetic_refs(size_t(magnetic_target.num_chunks), 0);
  for (int i = 0; i < magnetic_target.num_chunks; ++i)
    magnetic_refs[size_t(i)] = magnetic_target.chunks[i]->refcount;
  f->synchronize_magnetic_fields();
  f->synchronize_magnetic_fields();
  live_state = f->backend_state;
  live_executable = f->executable;
  const uint32_t magnetic_dirty = f->dirty_mask;
  const uint64_t magnetic_generation = generation(*f, MutationKind::material_phase);
  const uint64_t magnetic_local_generation = f->local_invalidation_generation;
  const bool magnetic_changed_materials = legacy_material_change_pending(*f);
  const int magnetic_countdown = f->phasein_time;
  rejected = false;
  try {
    f->phase_in_material(&magnetic_target, f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "live nested magnetic snapshot did not reject material setup collectively");
  CHECK(f->backend_state == live_state && f->executable == live_executable &&
            f->dirty_mask == magnetic_dirty &&
            generation(*f, MutationKind::material_phase) == magnetic_generation &&
            f->local_invalidation_generation == magnetic_local_generation &&
            legacy_material_change_pending(*f) == magnetic_changed_materials &&
            f->phasein_time == magnetic_countdown,
        "snapshot-rejected material setup changed resident or lifecycle state");
  for (int i = 0; i < f->num_chunks; ++i)
    if (f->chunks[i]->is_mine())
      CHECK(!f->chunks[i]->new_s &&
                magnetic_target.chunks[i]->refcount == magnetic_refs[size_t(i)],
            "snapshot-rejected material setup retained its target");
  f->restore_magnetic_fields();
  f->restore_magnetic_fields();
  CHECK(magnetic_counts.magnetic_restores == 1,
        "material rejection stranded the nested magnetic snapshot");

  f->backend->poison();
  const uint32_t poison_dirty = f->dirty_mask;
  const uint64_t poison_generation = generation(*f, MutationKind::material_phase);
  const uint64_t poison_local_generation = f->local_invalidation_generation;
  const bool poison_changed_materials = legacy_material_change_pending(*f);
  const int poison_countdown = f->phasein_time;
  rejected = false;
  try {
    f->phase_in_material(&magnetic_target, f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "poisoned backend accepted material setup");
  CHECK(f->backend_state == live_state && f->executable == live_executable &&
            f->dirty_mask == poison_dirty &&
            generation(*f, MutationKind::material_phase) == poison_generation &&
            f->local_invalidation_generation == poison_local_generation &&
            legacy_material_change_pending(*f) == poison_changed_materials &&
            f->phasein_time == poison_countdown,
        "poison rejection changed resident or lifecycle state");
  for (int i = 0; i < f->num_chunks; ++i)
    if (f->chunks[i]->is_mine())
      CHECK(!f->chunks[i]->new_s &&
                magnetic_target.chunks[i]->refcount == magnetic_refs[size_t(i)],
            "poison rejection changed target ownership");
  delete f;
  delete current;
}

static void test_material_phase_cpu_to_resident_preparation() {
  const grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure current(gv, eps_slab, pml(0.5));
  structure target(gv, unit_epsilon, pml(0.5));
  target.set_conductivity(Dz, phase_conductivity);
  fields f(&current);
  CHECK(f.phase_in_material(&target, 2.0 * f.dt) == 2,
        "CPU material setup did not retain its countdown");

  std::vector<structure_chunk *> before(size_t(f.num_chunks), NULL);
  std::vector<int> target_refs(size_t(target.num_chunks), 0);
  for (int i = 0; i < f.num_chunks; ++i) {
    before[size_t(i)] = f.chunks[i]->s;
    target_refs[size_t(i)] = target.chunks[i]->refcount;
    if (f.chunks[i]->is_mine())
      CHECK(!f.chunks[i]->s->conductivity[Dz][Z],
            "CPU-only material setup eagerly realized resident storage");
  }
  {
    fields copy(f);
    for (int i = 0; i < copy.num_chunks; ++i) {
      FOR_FIELD_TYPES(ft) CHECK(!copy.chunks[i]->pol[ft],
                                "zero-node fields copy retained a polarization list");
      DOCMP2 FOR_COMPONENTS(c) {
        CHECK(!f.chunks[i]->f_minus_p[c][cmp] || copy.chunks[i]->f_minus_p[c][cmp],
              "fields copy lost an allocated polarization-subtraction row");
        CHECK(!f.chunks[i]->f_w_prev[c][cmp] || copy.chunks[i]->f_w_prev[c][cmp],
              "fields copy lost an allocated previous-W row");
        CHECK(f.chunks[i]->f_minus_p[c][cmp] || !copy.chunks[i]->f_minus_p[c][cmp],
              "fields copy retained an uninitialized polarization-subtraction pointer");
        CHECK(f.chunks[i]->f_w_prev[c][cmp] || !copy.chunks[i]->f_w_prev[c][cmp],
              "fields copy retained an uninitialized previous-W pointer");
      }
    }
  }

  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  set_material_phase_prepare_failure_for_testing(0, 1);
  bool rejected = false;
  try {
    f.init_backend();
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  set_material_phase_prepare_failure_for_testing(-1, -1);
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "rank-asymmetric resident material preparation was not rejected collectively");
  CHECK(!f.backend_state && f.phasein_time == 2,
        "failed resident material preparation changed state or countdown");
  for (int i = 0; i < f.num_chunks; ++i) {
    CHECK(f.chunks[i]->s == before[size_t(i)],
          "failed resident material preparation committed a current chunk");
    if (f.chunks[i]->is_mine())
      CHECK(f.chunks[i]->new_s == target.chunks[i] &&
                target.chunks[i]->refcount == target_refs[size_t(i)],
            "failed resident material preparation changed target ownership");
  }

  f.init_backend();
  CHECK(f.backend_state && f.phasein_time == 2,
        "resident material preparation retry did not create state");
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine()) {
      CHECK(f.chunks[i]->s != before[size_t(i)] && f.chunks[i]->s->refcount == 1,
            "resident material preparation did not detach current storage");
      CHECK(f.chunks[i]->s->conductivity[Dz][Z] && f.chunks[i]->s->condinv[Dz][Z],
            "resident material preparation did not realize the target storage union");
      CHECK(f.chunks[i]->new_s == target.chunks[i] &&
                target.chunks[i]->refcount == target_refs[size_t(i)],
            "resident material preparation retry changed target ownership");
    }
}

static std::complex<double> multiply_fields(const std::complex<realnum> *values, const vec &,
                                            void *) {
  return values[0] * values[1];
}

#ifdef HAVE_HDF5
static std::unique_ptr<binary_partition> uneven_repeated_partition() {
  std::unique_ptr<binary_partition> tree(new binary_partition(3));
  tree.reset(new binary_partition(split_plane{X, 0.9},
                                  std::unique_ptr<binary_partition>(new binary_partition(2)),
                                  std::move(tree)));
  tree.reset(new binary_partition(split_plane{X, 0.3},
                                  std::unique_ptr<binary_partition>(new binary_partition(1)),
                                  std::move(tree)));
  tree.reset(new binary_partition(split_plane{X, -0.3},
                                  std::unique_ptr<binary_partition>(new binary_partition(0)),
                                  std::move(tree)));
  return std::unique_ptr<binary_partition>(new binary_partition(
      split_plane{X, -0.9}, std::unique_ptr<binary_partition>(new binary_partition(0)),
      std::move(tree)));
}

static void test_sharded_dft_checkpoint_ordering() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  std::unique_ptr<binary_partition> partition = uneven_repeated_partition();
  structure *s = new structure(gv, eps_slab, boundary_region(), identity(), 0, 0.5, false,
                               DEFAULT_SUBPIXEL_TOL, DEFAULT_SUBPIXEL_MAXEVAL, partition.get());
  fields *f = new fields(s);
  gaussian_src_time src(0.3, 0.1);
  f->add_point_source(Ez, src, vec(0.11, 0.13));
  f->require_component(Ez);
  component component_ez = Ez;
  dft_fields monitor = f->add_dft_fields(&component_ez, 1, f->v, 0.3, 0.3, 1,
                                         /*use_centered_grid=*/true,
                                         /*decimation_factor=*/1,
                                         /*persist=*/true);

  rebuild_trace rebuilds;
  access_trace accesses;
  f->backend = new access_tracking_backend(*f, rebuilds, accesses);
  f->advance(2);

  int owned_chunks = 0;
  for (int i = 0; i < f->num_chunks; ++i)
    owned_chunks += f->chunks[i]->is_mine();
  if (count_processors() > 1)
    CHECK(min_to_all(owned_chunks) < max_to_all(owned_chunks),
          "custom checkpoint partition did not produce uneven ownership");
  const int dft_failure_rank = min_to_all(monitor.chunks ? my_rank() : count_processors());
  CHECK(dft_failure_rank < count_processors(), "custom checkpoint monitor has no DFT owner");

  char filename[160];
  snprintf(filename, sizeof(filename), "/tmp/meep-sharded-dft-%ld-%d.h5", long(getpid()),
           my_rank());

  accesses.fail_dft_read_rank = dft_failure_rank;
  bool dump_failure = false;
  try {
    f->dump(filename, false);
  }
  catch (const std::runtime_error &) {
    dump_failure = true;
  }
  const int dump_failures = sum_to_all(int(dump_failure));
  CHECK(dump_failures == count_processors(),
        "sharded DFT dump failure was not reconciled at its outer boundary (%d/%d)", dump_failures,
        count_processors());
  accesses.fail_dft_read_rank = -1;
  std::remove(filename);
  all_wait();

  f->dump(filename, false);
  CHECK(sum_to_all(int(accesses.dft_reads)) > 0,
        "sharded DFT dump did not refresh resident accumulators");

  accesses.fail_dft_write_rank = dft_failure_rank;
  bool load_failure = false;
  try {
    h5file file(filename, h5file::READONLY, false, true);
    std::string local_error;
    for (int i = 0; i < f->num_chunks; ++i)
      if (f->chunks[i]->is_mine()) {
        char dataname[32];
        snprintf(dataname, sizeof(dataname), "chunk%02d", i);
        load_dft_hdf5(f->chunks[i]->dft_chunks, dataname, &file, 0, false, &local_error);
      }
    backend_reconcile_host_access(local_error, "backend_api sharded DFT load");
  }
  catch (const std::runtime_error &) {
    load_failure = true;
  }
  const int load_failures = sum_to_all(int(load_failure));
  CHECK(load_failures == count_processors(),
        "sharded DFT load failure was not reconciled at its outer boundary (%d/%d)", load_failures,
        count_processors());
  accesses.fail_dft_write_rank = -1;

  {
    h5file file(filename, h5file::READONLY, false, true);
    std::string local_error;
    for (int i = 0; i < f->num_chunks; ++i)
      if (f->chunks[i]->is_mine()) {
        char dataname[32];
        snprintf(dataname, sizeof(dataname), "chunk%02d", i);
        load_dft_hdf5(f->chunks[i]->dft_chunks, dataname, &file, 0, false, &local_error);
      }
    backend_reconcile_host_access(local_error, "backend_api sharded DFT load recovery");
  }

  f->load(filename, false);
  const int restored_time = f->t;
  f->advance(1);
  CHECK(f->t == restored_time + 1, "execution did not continue after sharded DFT restore");
  int rank = 0;
  size_t dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> restored(f->get_dft_array(monitor, Ez, 0, &rank, dims));
  CHECK(restored.get() != NULL, "restored sharded DFT state could not be queried");

  std::remove(filename);
  monitor.remove();
  delete f;
  delete s;
}
#endif

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
                               ElementType::float64,       ElementType::complex_float64,
                               ElementType::int32,         ElementType::index};
  for (size_t k = 0; k < sizeof(types) / sizeof(types[0]); ++k) {
    const StorageKey key{-1, int(array_kind::num_kinds), -1, -1, int(k)};
    const ArrayId id =
        f->array_catalog->register_array(key, arrays[k], 4, array_role::scratch, types[k]);
    const size_t element_size = host_element_bytes(types[k]);
    std::vector<unsigned char> got(2 * element_size, 0);
    cpu.read(ArrayRef{id, 1, 2}, got.data(), got.size());
    CHECK(memcmp(got.data(), static_cast<unsigned char *>(arrays[k]) + element_size, got.size()) ==
              0,
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
  catch (const std::out_of_range &) {
    rejected = true;
  }
  CHECK(rejected, "an invalid ArrayId was not rejected");

  rejected = false;
  try {
    realnum one = 0;
    cpu.read(ArrayRef{ArrayId{0}, f->array_catalog->spec(ArrayId{0}).elements, 1}, &one,
             sizeof(one));
  }
  catch (const std::out_of_range &) {
    rejected = true;
  }
  CHECK(rejected, "an out-of-bounds ArrayRef was not rejected");

  rejected = false;
  try {
    realnum one = 0;
    cpu.read(ArrayRef{ArrayId{0}, 0, 1}, &one, sizeof(one) + 1);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
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
  CHECK(plan.arrays[3].storage == Precision::f64,
        "mixed policy did not preserve monitor precision");
  CHECK(plan.arrays[4].storage == Precision::f64, "fixed float64 storage was narrowed");
  CHECK(plan.provisional_peak_bytes() == 132, "precision-aware peak bytes are %zu, expected 132",
        plan.provisional_peak_bytes());
  CHECK(plan.steady_state_bytes() == 112, "precision-aware steady bytes are %zu, expected 112",
        plan.steady_state_bytes());
  CHECK(validate_alias_precisions(plan, why), "valid plan aliases were rejected: %s", why.c_str());
  plan.arrays[1].storage = Precision::f64;
  CHECK(!validate_alias_precisions(plan, why), "mismatched plan alias precision was accepted");

  ArraySpec huge = {ArrayId{0},
                    array_role::dft,
                    ElementType::complex_float64,
                    Precision::f64,
                    std::numeric_limits<size_t>::max(),
                    alignof(double),
                    invalid_array(),
                    false};
  bool overflow_rejected = false;
  try {
    (void)storage_bytes(huge);
  }
  catch (const std::overflow_error &) {
    overflow_rejected = true;
  }
  CHECK(overflow_rejected, "overflowing storage byte count was accepted");
}

static void test_dft_access_boundaries() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->require_component(Ez);
  component c = Ez;
  dft_fields monitor = f->add_dft_fields(&c, 1, f->v, 0.3, 0.3, 1);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);

  const int reads_before = counts.reads;
  const int writes_before = counts.writes;
  if (monitor.chunks) monitor.scale_dfts(2.0);
  CHECK(or_to_all(counts.reads > reads_before),
        "DFT host mutation did not read resident accumulator data");
  CHECK(or_to_all(counts.writes > writes_before),
        "DFT host mutation did not publish the accumulator back to the backend");

  monitor.remove();
  CHECK(!f->backend_state, "DFT removal left resident state referencing freed storage");
  CHECK(counts.rebuilds == 1, "DFT removal did not migrate resident state before deletion");
  delete f;
  delete s;
}

static void test_detached_dft_access() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->require_component(Ez);
  component c = Ez;
  dft_fields monitor = f->add_dft_fields(&c, 1, f->v, 0.3, 0.3, 1,
                                         /*use_centered_grid=*/true,
                                         /*decimation_factor=*/1,
                                         /*persist=*/true);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);

  f->clear_dft_monitors();
  CHECK(!f->backend_state, "clearing DFT monitors left resident state alive");
  CHECK(counts.rebuilds == 1, "clearing DFT monitors did not migrate resident state");
  f->advance(1);
  BackendState *rebuilt = f->backend_state;

  /* Detached persistent accumulators are now host-authoritative and absent
     from the rebuilt catalog. Queries/mutations must not try to resolve them. */
  if (monitor.chunks) {
    monitor.chunks->sync_dft_to_host();
    monitor.scale_dfts(2.0);
  }
  monitor.remove();
  CHECK(f->backend_state == rebuilt,
        "removing an already detached DFT monitor retired unrelated backend state");

  delete f;
  delete s;
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
  CHECK(counts.connections_current_at_create,
        "resident backend state was created before halo topology was finalized");
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
  CHECK(is_dirty(*f, dirty_initialization), "initialize_field did not invalidate resident values");
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

static void test_backend_reselection_invalidates_representation() {
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);
  CHECK(f->backend_state && f->executable,
        "backend replacement fixture did not prepare resident execution");
  const uint64_t generation_before = generation(*f, MutationKind::precision_policy);

  execution_options cpu;
  f->select_backend(cpu);
  CHECK(!f->backend_state && !f->executable && counts.states_destroyed == 1 &&
            counts.executables_destroyed == 1,
        "backend replacement did not retire the prior representation exactly once");
  CHECK(is_dirty(*f, dirty_storage) && is_dirty(*f, dirty_initialization) &&
            is_dirty(*f, dirty_executable),
        "backend replacement did not invalidate storage, values, and executable");
  CHECK(generation(*f, MutationKind::precision_policy) == generation_before + 1,
        "backend replacement did not record a representation generation");

  delete f;
  delete s;
}

static void test_resident_polarization_preparation() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  lorentzian_susceptibility susceptibility(1.1, 0.05);
  gyrotropic_susceptibility gyro(vec(0.17, -0.23, 0.31), 0.8, 0.03, 0.07, GYROTROPIC_SATURATED);

  {
    structure single_structure(gv, eps_slab, pml(0.5), identity(), 1);
    single_structure.add_susceptibility(eps_slab, E_stuff, susceptibility);
    fields single(&single_structure);
    single.require_component(Ez);
    single.advance(1);
    fields copy(single);
    for (int i = 0; i < single.num_chunks; ++i)
      if (single.chunks[i]->is_mine()) {
        polarization_state *source = single.chunks[i]->pol[E_stuff];
        polarization_state *duplicate = copy.chunks[i]->pol[E_stuff];
        CHECK(source && !source->next && source->data && duplicate && !duplicate->next &&
                  duplicate->data && duplicate->data != source->data,
              "one-node fields copy did not duplicate polarization state exactly");
      }
  }

  structure resident_structure(gv, eps_slab, pml(0.5), identity(), 2);
  resident_structure.add_susceptibility(eps_slab, E_stuff, susceptibility);
  resident_structure.add_susceptibility(eps_slab, E_stuff, gyro);
  fields resident(&resident_structure);
  resident.require_component(Ez);
  bool initially_null = true;
  for (int i = 0; i < resident.num_chunks; ++i)
    if (resident.chunks[i]->is_mine())
      for (polarization_state *p = resident.chunks[i]->pol[E_stuff]; p; p = p->next)
        initially_null = initially_null && !p->data;
  CHECK(and_to_all(initially_null), "polarization state was allocated before preparation");

  lifetime_counts counts;
  resident.backend = new tracking_backend(resident, counts);
  resident.advance(1);
  {
    fields copy(resident);
    for (int i = 0; i < resident.num_chunks; ++i) {
      polarization_state *source = resident.chunks[i]->pol[E_stuff];
      polarization_state *duplicate = copy.chunks[i]->pol[E_stuff];
      size_t copied = 0;
      while (source && duplicate) {
        CHECK(!source->data || (duplicate->data && duplicate->data != source->data),
              "fields copy did not duplicate populated polarization state");
        ++copied;
        source = source->next;
        duplicate = duplicate->next;
      }
      CHECK(!source && !duplicate, "fields copy changed the polarization-state list shape");
      if (resident.chunks[i]->is_mine())
        CHECK(copied == 2, "two-node fields copy did not retain both polarization states");
    }
  }
  bool owns_chunk = false;
  for (int i = 0; i < resident.num_chunks; ++i)
    owns_chunk = owns_chunk || resident.chunks[i]->is_mine();
  CHECK(and_to_all(!owns_chunk || counts.polarization_arrays_at_create > 0),
        "resident state was created without P/P_prev arrays");
  CHECK(and_to_all(!owns_chunk || (counts.polarization_updates_at_compile > 0 &&
                                   counts.polarization_subtractions_at_compile > 0)),
        "resident executable was compiled without polarization spans");
  CHECK(and_to_all(!owns_chunk || counts.gyrotropic_update_at_compile),
        "resident executable was compiled without a gyrotropic update");
  CHECK(counts.polarization_zero_at_create,
        "resident polarization arrays were not zero-initialized before state creation");
  CHECK(counts.connections_current_at_create,
        "resident state was created before PE/PH topology was finalized");
  CHECK(resident.t == 0, "tracking backend unexpectedly advanced host time");
  const size_t arrays_after_first = counts.polarization_arrays_at_create;
  BackendState *state_after_first = resident.backend_state;
  resident.advance(1);
  CHECK(resident.backend_state == state_after_first && counts.states_created == 1,
        "second resident preparation rebuilt already-realized polarization state");
  CHECK(counts.polarization_arrays_at_create == arrays_after_first,
        "second resident preparation changed the polarization catalog");

  const size_t catalog_after_first = counts.arrays_at_create;
  counts.gyrotropic_update_at_compile = false;
  resident.require_component(Ex);
  resident.advance(1);
  CHECK(and_to_all(!owns_chunk || (counts.states_created == 2 && counts.states_destroyed == 1)),
        "post-allocation field growth did not replace resident state");
  CHECK(and_to_all(!owns_chunk ||
                   (counts.executables_created == 2 && counts.executables_destroyed == 1)),
        "post-allocation field growth did not replace the resident executable");
  CHECK(and_to_all(!owns_chunk || counts.gyrotropic_update_at_compile),
        "rebuilt resident executable lost its gyrotropic update");
  CHECK(!owns_chunk || counts.arrays_at_create > catalog_after_first,
        "post-allocation field growth did not expand the resident field catalog");
  CHECK(counts.polarization_arrays_at_create - arrays_after_first == arrays_after_first,
        "post-allocation field growth changed immutable gyrotropic state storage");

  structure cpu_structure(gv, eps_slab, pml(0.5), identity(), 2);
  cpu_structure.add_susceptibility(eps_slab, E_stuff, susceptibility);
  cpu_structure.add_susceptibility(eps_slab, E_stuff, gyro);
  fields cpu(&cpu_structure);
  cpu.require_component(Ez);
  bool cpu_null = true;
  for (int i = 0; i < cpu.num_chunks; ++i)
    if (cpu.chunks[i]->is_mine())
      for (polarization_state *p = cpu.chunks[i]->pol[E_stuff]; p; p = p->next)
        cpu_null = cpu_null && !p->data;
  CHECK(or_to_all(cpu_null), "CPU polarization state is no longer lazy before update_pols");
  cpu.advance(1);
  bool cpu_allocated = false;
  for (int i = 0; i < cpu.num_chunks; ++i)
    if (cpu.chunks[i]->is_mine())
      for (polarization_state *p = cpu.chunks[i]->pol[E_stuff]; p; p = p->next)
        cpu_allocated = cpu_allocated || p->data;
  CHECK(or_to_all(cpu_allocated), "CPU update_pols did not lazily allocate polarization state");
}

static void test_resident_beta_fingerprint() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  fields f(&s, 0, 0.17);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.1, 0.1));
  f.require_component(Ez);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  f.advance(1);

  bool owns_chunk = false;
  for (int i = 0; i < f.num_chunks; ++i)
    owns_chunk = owns_chunk || f.chunks[i]->is_mine();
  CHECK(and_to_all(!owns_chunk || counts.beta_updates_at_compile > 0),
        "resident executable was compiled without beta updates");
  CHECK(coordinate_state_matches(f, f.step_plans[0]),
        "fresh resident beta fingerprint does not match");

  const double original = f.beta;
  f.beta = -original;
  CHECK(!coordinate_state_matches(f, f.step_plans[0]),
        "outer beta mutation was not rejected by the resident fingerprint");
  f.beta = original;

  CHECK(f.num_chunks > 0, "beta fingerprint test has no chunks");
  if (f.num_chunks > 0) {
    f.chunks[0]->beta = -original;
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "per-chunk beta mutation was not rejected by the resident fingerprint");
    f.chunks[0]->beta = original;
  }
  CHECK(coordinate_state_matches(f, f.step_plans[0]), "restored beta fingerprint does not match");
}

static void test_resident_bfast_fingerprint() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  const std::vector<double> scaled_k{0.17, -0.11, 0.07};
  fields f(&s, 0, 0, true, 0, 0, scaled_k);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.1, 0.1));
  f.require_component(Ez);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  f.advance(1);

  bool owns_chunk = false;
  for (int i = 0; i < f.num_chunks; ++i)
    owns_chunk = owns_chunk || f.chunks[i]->is_mine();
  CHECK(and_to_all(!owns_chunk || counts.bfast_updates_at_compile > 0),
        "resident executable was compiled without BFAST updates");
  CHECK(coordinate_state_matches(f, f.step_plans[0]),
        "fresh resident BFAST fingerprint does not match");

  f.bfast_scaled_k[0] = -scaled_k[0];
  CHECK(!coordinate_state_matches(f, f.step_plans[0]),
        "outer BFAST mutation was not rejected by the resident fingerprint");
  f.bfast_scaled_k = scaled_k;

  CHECK(f.num_chunks > 0, "BFAST fingerprint test has no chunks");
  if (f.num_chunks > 0) {
    f.chunks[0]->bfast_scaled_k[1] = -scaled_k[1];
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "per-chunk BFAST mutation was not rejected by the resident fingerprint");
    f.chunks[0]->bfast_scaled_k = scaled_k;
  }
  CHECK(coordinate_state_matches(f, f.step_plans[0]), "restored BFAST fingerprint does not match");

  f.bfast_scaled_k.resize(2);
  for (int i = 0; i < f.num_chunks; ++i)
    f.chunks[i]->bfast_scaled_k.resize(2);
  CHECK(!coordinate_state_matches(f, f.step_plans[0]),
        "malformed matching BFAST vectors were not rejected before plan rebuild");
}

static void test_resident_cylindrical_fingerprint() {
  grid_volume gv = volcyl(3.0, 4.0, 8.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  fields f(&s, +1.0, 0, true, 64, 64, std::vector<double>{0, 0, 0});
  const component components[] = {Er, Ep, Ez, Hr, Hp, Hz};
  for (component c : components)
    f.require_component(c);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  f.advance(1);

  bool owns_chunk = false, owns_origin_chunk = false;
  for (int i = 0; i < f.num_chunks; ++i) {
    owns_chunk = owns_chunk || f.chunks[i]->is_mine();
    owns_origin_chunk =
        owns_origin_chunk || (f.chunks[i]->is_mine() && f.chunks[i]->gv.origin_r() == 0.0);
  }
  CHECK(and_to_all(!owns_chunk || counts.cylindrical_m_updates_at_compile > 0),
        "resident executable was compiled without cylindrical m/r updates");
  CHECK(and_to_all(!owns_origin_chunk || counts.cylindrical_origin_actions_at_compile > 0),
        "resident executable was compiled without cylindrical origin actions");
  CHECK(coordinate_state_matches(f, f.step_plans[0]),
        "fresh resident cylindrical fingerprint does not match");

  const size_t arrays_after_first = counts.arrays_at_create;
  BackendState *state_after_first = f.backend_state;
  f.change_m(-1.0);
  f.advance(1);
  CHECK(f.backend_state == state_after_first && counts.states_created == 1 &&
            counts.states_destroyed == 0 && counts.arrays_at_create == arrays_after_first,
        "sign-only cylindrical m change unnecessarily rebuilt resident storage");
  CHECK(counts.executables_created == 2 && counts.executables_destroyed == 1,
        "sign-only cylindrical m change did not replace the resident executable");
  CHECK(f.step_plans[0] && f.step_plans[0]->cylindrical_m == -1.0 &&
            coordinate_state_matches(f, f.step_plans[0]),
        "sign-only cylindrical m change did not publish a matching plan");

  f.change_m(0.0);
  CHECK(f.backend_state == NULL && f.executable == NULL && counts.rebuilds == 1 &&
            (!owns_chunk || counts.rebuild_saw_live_imaginary),
        "complex-to-real cylindrical transition did not migrate and retire live storage first");
  f.advance(1);
  CHECK(counts.states_created == 2 && counts.states_destroyed == 1 &&
            counts.arrays_at_create <= arrays_after_first,
        "cylindrical m-to-zero field-layout change did not rebuild resident storage");
  CHECK(counts.executables_created == 3 && counts.executables_destroyed == 2,
        "cylindrical m-to-zero change did not replace the resident executable");
  CHECK(counts.cylindrical_m_updates_at_compile == 0,
        "zero cylindrical m retained m/r descriptors after recompilation");
  CHECK(and_to_all(!owns_origin_chunk || counts.cylindrical_origin_actions_at_compile > 0),
        "zero cylindrical m lost its origin actions after recompilation");
  CHECK(coordinate_state_matches(f, f.step_plans[0]),
        "recompiled zero-m cylindrical fingerprint does not match");

  f.m = +0.5;
  CHECK(and_to_all(!coordinate_state_matches(f, f.step_plans[0])),
        "direct outer cylindrical m mutation was not rejected");
  if (count_processors() == 1) {
    bool rejected = false;
    try {
      f.init_backend();
    }
    catch (const std::runtime_error &) {
      rejected = true;
    }
    CHECK(rejected, "direct outer cylindrical m mutation was not rejected before preparation");
    CHECK(counts.states_created == 2 && counts.states_destroyed == 1 &&
              counts.executables_created == 3,
          "rejected cylindrical mismatch changed resident state or executable counts");
  }
  f.m = 0.0;

  CHECK(f.num_chunks > 0, "cylindrical fingerprint test has no chunks");
  if (f.num_chunks > 0) {
    f.chunks[0]->m = +0.5;
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "per-chunk cylindrical m mutation was not rejected");
    f.chunks[0]->m = 0.0;

    f.chunks[0]->zero_fields_near_cylorigin = !f.chunks[0]->zero_fields_near_cylorigin;
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "per-chunk cylindrical origin-policy mutation was not rejected");
    f.chunks[0]->zero_fields_near_cylorigin = !f.chunks[0]->zero_fields_near_cylorigin;
  }

  const double plan_m = f.step_plans[0]->cylindrical_m;
  f.step_plans[0]->cylindrical_m = +0.5;
  CHECK(!coordinate_state_matches(f, f.step_plans[0]),
        "prepared-plan cylindrical m mutation was not rejected");
  f.step_plans[0]->cylindrical_m = plan_m;

  CHECK(!f.step_plans[0]->cylindrical_origin_r.empty() &&
            !f.step_plans[0]->cylindrical_zero_near_origin.empty(),
        "prepared cylindrical plan lacks per-chunk origin fingerprints");
  if (!f.step_plans[0]->cylindrical_origin_r.empty()) {
    f.step_plans[0]->cylindrical_origin_r[0] += 1.0;
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "prepared-plan cylindrical origin mutation was not rejected");
    f.step_plans[0]->cylindrical_origin_r[0] -= 1.0;
  }
  if (!f.step_plans[0]->cylindrical_zero_near_origin.empty()) {
    f.step_plans[0]->cylindrical_zero_near_origin[0] ^= 1;
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "prepared-plan cylindrical origin-policy mutation was not rejected");
    f.step_plans[0]->cylindrical_zero_near_origin[0] ^= 1;
  }
  CHECK(coordinate_state_matches(f, f.step_plans[0]),
        "restored cylindrical fingerprint does not match");

  f.step_plans[0]->cylindrical_origin_r.pop_back();
  CHECK(!coordinate_state_matches(f, f.step_plans[0]),
        "malformed cylindrical origin vector was not rejected");

  structure refused_structure(gv, eps_slab, pml(0.5), identity(), 2);
  fields refused(&refused_structure, +1.0, 0, true, 64, 64, std::vector<double>{0, 0, 0});
  for (component c : components)
    refused.require_component(c);
  lifetime_counts refused_counts;
  refused.backend = new tracking_backend(refused, refused_counts);
  refused.advance(1);
  bool refused_owns_chunk = false;
  int refused_owned_chunk = -1;
  for (int i = 0; i < refused.num_chunks; ++i)
    if (refused.chunks[i]->is_mine()) {
      refused_owns_chunk = true;
      if (refused_owned_chunk < 0) refused_owned_chunk = i;
    }
  BackendState *refused_state = refused.backend_state;
  Executable *refused_executable = refused.executable;
  realnum *const refused_imaginary =
      refused_owned_chunk < 0 ? NULL : refused.chunks[refused_owned_chunk]->f[Er][1];
  refused_counts.fail_rebuild = true;
  bool rejected = false;
  try {
    refused.change_m(0.0);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(and_to_all(rejected), "failed complex-to-real migration was not rejected collectively");
  CHECK(refused_counts.rebuilds == 1 &&
            (!refused_owns_chunk || refused_counts.rebuild_saw_live_imaginary),
        "failed migration did not observe the old imaginary catalog and pointers");
  CHECK(refused.backend_state == refused_state && refused.executable == refused_executable &&
            refused.m == +1.0 && !refused.is_real &&
            (refused_owned_chunk < 0 ||
             refused.chunks[refused_owned_chunk]->f[Er][1] == refused_imaginary),
        "failed complex-to-real migration partially mutated live resident state");
  for (int i = 0; i < refused.num_chunks; ++i)
    CHECK(refused.chunks[i]->m == +1.0,
          "failed complex-to-real migration partially changed chunk %d m", i);
  refused_counts.fail_rebuild = false;
}

static void test_classification_change_recompiles() {
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);

  CHECK(f->prepared_classification_hash != 0, "classification did not publish a hash");
  ++f->prepared_classification_hash;
  invalidate(*f, MutationKind::material_values, "backend_api:changed_classification_hash");
  f->advance(1);
  CHECK(counts.executables_created == 2 && counts.executables_destroyed == 1,
        "a changed classification hash did not recompile the executable");

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

/* A storage rebuild must give the backend a chance to preserve authoritative
   values before either its executable or state is destroyed. A refusal leaves
   both objects live. */
static void test_authority_safe_state_rebuild() {
  structure *s;
  fields *f;
  build(&s, &f);

  rebuild_trace trace;
  rebuild_backend_base *tracking = new rebuild_tracking_backend(*f, trace);
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
    CHECK(trace.events[2] == "destroy-state", "third rebuild event is %s", trace.events[2].c_str());
    CHECK(trace.events[3] == "create-state", "fourth rebuild event is %s", trace.events[3].c_str());
  }
  CHECK((trace.reasons & dirty_storage) != 0, "rebuild hook did not receive dirty_storage");

  delete f;
  delete s;

  build(&s, &f);
  rebuild_trace refused;
  tracking = new rebuild_backend_base(*f, refused);
  f->backend = tracking;
  f->backend_state = tracking->create_state(*f->storage_plan);
  BackendState *live_state = f->backend_state;
  refused.events.clear();
  clear_dirty(*f, DirtyMask(f->dirty_mask));
  invalidate(*f, MutationKind::field_layout);
  bool rejected = false;
  try {
    f->init_backend();
  }
  catch (const std::logic_error &) {
    rejected = true;
  }
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

  BackendState *state = f->backend_state;
  Executable *executable = f->executable;
  f->backend->prepare_state_rebuild(*state, dirty_storage);
  CHECK(f->backend_state == state && f->executable == executable,
        "CPU authority-safe rebuild hook modified host-authoritative state");

  delete f;
  delete s;
}

static void test_backend_safe_host_access() {
  structure *s;
  fields *f;
  build(&s, &f);
  component components[1] = {Ez};
  const double frequencies[2] = {0.2, 0.3};
  dft_fields monitor =
      f->add_dft_fields(components, 1, volume(vec(0.4, 0.4), vec(1.2, 1.2)), frequencies, 2);

  rebuild_trace rebuilds;
  access_trace accesses;
  f->backend = new access_tracking_backend(*f, rebuilds, accesses);
  f->advance(1);

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  const std::complex<double> point = f->get_field(Ez, vec(0.7, 0.8), true);
  CHECK(fabs(point.real() - 2.5) < 1e-12 && fabs(point.imag() - 2.5) < 1e-12,
        "point query did not consume backend-refreshed real/imaginary values");
  CHECK(sum_to_all(int(accesses.field_reads)) > 0 && max_to_all(int(accesses.max_elements)) == 1,
        "point query did not issue exact one-element backend reads");

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  std::unique_ptr<std::complex<realnum>[]> slice(
      f->get_complex_array_slice(volume(vec(0.5, 0.5), vec(1.0, 1.0)), Ez));
  CHECK(slice.get() != NULL && sum_to_all(int(accesses.field_reads)) > 0,
        "array slice did not issue explicit backend field reads");
  size_t largest_field_allocation = 0;
  for (size_t i = 0; i < f->array_catalog->size(); ++i) {
    const ArraySpec &spec = f->array_catalog->spec(ArrayId{uint32_t(i)});
    if (spec.role == array_role::field && spec.elements > largest_field_allocation)
      largest_field_allocation = spec.elements;
  }
  CHECK(!accesses.field_reads || accesses.max_elements < largest_field_allocation,
        "small array slice refreshed a complete unrelated field allocation");

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  const volume access_region(vec(0.45, 0.45), vec(1.15, 1.05));
  const double field_max = f->max_abs(Ez, access_region);
  CHECK(fabs(field_max - sqrt(12.5)) < 1e-12,
        "integration did not consume backend-refreshed field values: %.17g", field_max);
  CHECK(sum_to_all(int(accesses.field_reads)) > 0,
        "integration did not issue explicit backend field reads");
  CHECK(!accesses.field_reads || accesses.max_elements < largest_field_allocation,
        "small integration refreshed a complete unrelated field allocation");

  accesses.fail_read_rank = 0;
  bool integration_failure = false;
  try {
    (void)f->max_abs(Ez, access_region);
  }
  catch (const std::runtime_error &) {
    integration_failure = true;
  }
  CHECK(sum_to_all(int(integration_failure)) == count_processors(),
        "integration read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;

  structure *s2;
  fields *f2;
  build(&s2, &f2);
  rebuild_trace rebuilds2;
  access_trace accesses2;
  f2->backend = new access_tracking_backend(*f2, rebuilds2, accesses2);
  f2->advance(1);
  accesses2.fail_read_rank = 0;
  bool integration2_failure = false;
  try {
    (void)f->integrate2(*f2, 1, components, 1, components, multiply_fields, NULL, access_region);
  }
  catch (const std::runtime_error &) {
    integration2_failure = true;
  }
  CHECK(sum_to_all(int(integration2_failure)) == count_processors(),
        "second integration read failure was not reconciled on every rank");
  accesses2.fail_read_rank = -1;
  delete f2;
  delete s2;

#ifdef HAVE_HDF5
  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  h5file *field_file = new h5file("backend-api-field-access.h5", h5file::WRITE, true);
  f->output_hdf5(Ez, access_region, field_file);
  CHECK(sum_to_all(int(accesses.field_reads)) > 0,
        "ordinary HDF5 output did not issue explicit backend field reads");
  CHECK(!accesses.field_reads || accesses.max_elements < largest_field_allocation,
        "small HDF5 output refreshed a complete unrelated field allocation");
  field_file->remove();
  delete field_file;

  accesses.fail_read_rank = 0;
  bool hdf5_failure = false;
  field_file = new h5file("backend-api-field-failure.h5", h5file::WRITE, true);
  try {
    f->output_hdf5(Ez, access_region, field_file);
  }
  catch (const std::runtime_error &) {
    hdf5_failure = true;
  }
  CHECK(sum_to_all(int(hdf5_failure)) == count_processors(),
        "ordinary HDF5 read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;
  delete field_file;
  if (am_master()) std::remove("backend-api-field-failure.h5");
  all_wait();
#endif

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  int rank = 0;
  size_t dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> dft(f->get_dft_array(monitor, Ez, 0, &rank, dims));
  CHECK(dft.get() != NULL && sum_to_all(int(accesses.dft_reads)) > 0,
        "DFT array query did not refresh its accumulator storage");
  int local_dft_chunks = 0;
  for (dft_chunk *cur = monitor.chunks; cur; cur = cur->next_in_dft)
    ++local_dft_chunks;
  CHECK(sum_to_all(int(accesses.dft_reads)) == sum_to_all(local_dft_chunks),
        "DFT array query redundantly refreshed accumulator storage");

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  dft_chunk *chunklists[1] = {monitor.chunks};
  std::unique_ptr<std::complex<realnum>[]> encoded_dft;
  std::complex<realnum> *encoded_array = NULL;
  direction dirs[3];
  f->process_dft_component(chunklists, 1, 0, NO_COMPONENT, NULL, &encoded_array, &rank, dims, dirs);
  encoded_dft.reset(encoded_array);
  CHECK(encoded_dft.get() != NULL && sum_to_all(int(accesses.dft_reads)) > 0,
        "encoded DFT component did not refresh its normalized accumulator storage");
  CHECK(sum_to_all(int(accesses.dft_reads)) == sum_to_all(local_dft_chunks),
        "encoded DFT component redundantly refreshed accumulator storage");

  accesses.fail_read_rank = 0;
  bool dft_array_failure = false;
  try {
    std::unique_ptr<std::complex<realnum>[]> failed(f->get_dft_array(monitor, Ez, 0, &rank, dims));
  }
  catch (const std::runtime_error &) {
    dft_array_failure = true;
  }
  CHECK(sum_to_all(int(dft_array_failure)) == count_processors(),
        "DFT array read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;

  accesses.reads = accesses.writes = accesses.field_reads = accesses.dft_reads = 0;
  accesses.field_writes = accesses.dft_writes = accesses.max_elements = 0;
  backend_refresh_dft_chain(monitor.chunks, "backend_api DFT chain read");
  CHECK(sum_to_all(int(accesses.dft_reads)) == sum_to_all(local_dft_chunks),
        "DFT chain boundary did not read each local chunk exactly once");
  backend_publish_dft_chain(monitor.chunks, "backend_api DFT chain write");
  CHECK(sum_to_all(int(accesses.dft_writes)) == sum_to_all(local_dft_chunks),
        "DFT chain boundary did not publish each local chunk exactly once");

  accesses.fail_write_rank = 0;
  bool dft_write_failure = false;
  try {
    backend_publish_dft_chain(monitor.chunks, "backend_api injected DFT write");
  }
  catch (const std::runtime_error &) {
    dft_write_failure = true;
  }
  CHECK(sum_to_all(int(dft_write_failure)) == count_processors(),
        "rank-asymmetric DFT write failure was not reconciled on every rank");
  accesses.fail_write_rank = -1;
  backend_publish_dft_chain(monitor.chunks, "backend_api recovered DFT write");
  const int time_before_recovery = f->t;
  f->advance(1);
  CHECK(f->t == time_before_recovery + 1,
        "execution did not continue after a reconciled DFT write failure");

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  dft_ldos ldos(frequencies, sizeof(frequencies) / sizeof(*frequencies));
  ldos.update(*f);
  CHECK(sum_to_all(int(accesses.field_reads)) > 0,
        "LDOS update did not refresh its source-point fields");
  CHECK(max_to_all(int(accesses.max_elements)) == 1,
        "LDOS update read more than one field element at a time");

  accesses.fail_read_rank = 0;
  bool ldos_failure = false;
  try {
    ldos.update(*f);
  }
  catch (const std::runtime_error &) {
    ldos_failure = true;
  }
  CHECK(sum_to_all(int(ldos_failure)) == count_processors(),
        "rank-asymmetric LDOS read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;
  ldos.update(*f);

  BackendState *state_before_magnetic_rejection = f->backend_state;
  bool direct_magnetic_failure = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    direct_magnetic_failure = true;
  }
  CHECK(sum_to_all(int(direct_magnetic_failure)) == count_processors(),
        "resident magnetic synchronization was not rejected on every rank");
  CHECK(f->backend_state == state_before_magnetic_rejection,
        "magnetic rejection replaced resident state");
  bool repeated_magnetic_failure = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    repeated_magnetic_failure = true;
  }
  CHECK(sum_to_all(int(repeated_magnetic_failure)) == count_processors(),
        "magnetic rejection changed nesting state before returning");

  bool energy_magnetic_failure = false;
  try {
    (void)f->field_energy_in_box(f->v);
  }
  catch (const std::runtime_error &) {
    energy_magnetic_failure = true;
  }
  CHECK(sum_to_all(int(energy_magnetic_failure)) == count_processors(),
        "field_energy_in_box did not reject unsupported magnetic synchronization");

  bool flux_magnetic_failure = false;
  try {
    (void)f->flux_in_box(X, f->v);
  }
  catch (const std::runtime_error &) {
    flux_magnetic_failure = true;
  }
  CHECK(sum_to_all(int(flux_magnetic_failure)) == count_processors(),
        "flux_in_box did not reject unsupported magnetic synchronization");
  const int time_before_magnetic_recovery = f->t;
  f->advance(1);
  CHECK(f->t == time_before_magnetic_recovery + 1,
        "execution did not continue after magnetic-sync rejection");

  dft_flux flux(Ez, Ez, monitor.chunks, monitor.chunks, frequencies, 2, monitor.where, NO_DIRECTION,
                true);
  flux.monitor_lifetime = monitor.monitor_lifetime;
  dft_energy energy(monitor.chunks, monitor.chunks, monitor.chunks, monitor.chunks, frequencies, 2,
                    monitor.where);
  energy.monitor_lifetime = monitor.monitor_lifetime;
  dft_force force(monitor.chunks, monitor.chunks, monitor.chunks, frequencies, 2, monitor.where);
  force.monitor_lifetime = monitor.monitor_lifetime;
  const direction periodic_d[2] = {NO_DIRECTION, NO_DIRECTION};
  const int periodic_n[2] = {0, 0};
  const double periodic_k[2] = {0.0, 0.0};
  const double period[2] = {0.0, 0.0};
  dft_near2far near2far(monitor.chunks, frequencies, 2, 1.0, 1.0, monitor.where, periodic_d,
                        periodic_n, periodic_k, period);
  near2far.monitor_lifetime = monitor.monitor_lifetime;

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  if (am_master()) {
    std::vector<std::complex<double> > local_farfield(6 *
                                                      (sizeof(frequencies) / sizeof(*frequencies)));
    near2far.farfield_lowlevel(local_farfield.data(), vec(2.0, 2.0));
  }
  all_wait();
  CHECK(sum_to_all(int(accesses.dft_reads)) > 0,
        "rank-local far-field query did not refresh its accumulator storage");

  accesses.fail_read_rank = 0;
  bool dft_norm_failure = false;
  try {
    (void)f->dft_norm();
  }
  catch (const std::runtime_error &) {
    dft_norm_failure = true;
  }
  CHECK(sum_to_all(int(dft_norm_failure)) == count_processors(),
        "DFT norm read failure was not reconciled on every rank");

  bool flux_failure = false;
  try {
    delete[] flux.flux();
  }
  catch (const std::runtime_error &) {
    flux_failure = true;
  }
  CHECK(sum_to_all(int(flux_failure)) == count_processors(),
        "flux read failure was not reconciled on every rank");

  bool electric_failure = false;
  try {
    delete[] energy.electric();
  }
  catch (const std::runtime_error &) {
    electric_failure = true;
  }
  CHECK(sum_to_all(int(electric_failure)) == count_processors(),
        "electric-energy read failure was not reconciled on every rank");

  bool magnetic_failure = false;
  try {
    delete[] energy.magnetic();
  }
  catch (const std::runtime_error &) {
    magnetic_failure = true;
  }
  CHECK(sum_to_all(int(magnetic_failure)) == count_processors(),
        "magnetic-energy read failure was not reconciled on every rank");

  bool force_failure = false;
  try {
    delete[] force.force();
  }
  catch (const std::runtime_error &) {
    force_failure = true;
  }
  CHECK(sum_to_all(int(force_failure)) == count_processors(),
        "force read failure was not reconciled on every rank");

  bool farfield_failure = false;
  try {
    delete[] near2far.farfield(vec(2.0, 2.0));
  }
  catch (const std::runtime_error &) {
    farfield_failure = true;
  }
  CHECK(sum_to_all(int(farfield_failure)) == count_processors(),
        "far-field read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;

  accesses.fail_read_rank = 0;
  bool symmetric_failure = false;
  try {
    (void)f->get_field(Ez, vec(0.7, 0.8), true);
  }
  catch (const std::runtime_error &) {
    symmetric_failure = true;
  }
  CHECK(symmetric_failure, "rank-local backend read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;

  char filename[160];
  snprintf(filename, sizeof(filename), "/tmp/meep-backend-access-%ld-%d.h5", long(getpid()),
           my_rank());
  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  f->dump(filename, false);
  CHECK(sum_to_all(int(accesses.field_reads)) > 0 && sum_to_all(int(accesses.dft_reads)) > 0,
        "checkpoint dump did not explicitly read field and DFT storage");
  CHECK(f->backend_state != NULL && f->executable != NULL,
        "checkpoint dump retired resident execution state");

  f->load(filename, false);
  CHECK(accesses.prepare_rebuilds == 1,
        "checkpoint load did not prepare resident authority for replacement");
  CHECK(f->backend_state == NULL && f->executable == NULL,
        "checkpoint load retained artifacts referring to the old catalog");
  CHECK(is_dirty(*f, dirty_storage), "checkpoint load did not invalidate storage layout");
  std::remove(filename);

  delete f;
  delete s;
}

static void test_compact_dft_reduction_boundary() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->require_component(Ez);
  component component_ez = Ez;
  const double frequencies[2] = {0.2, 0.3};
  dft_fields monitor =
      f->add_dft_fields(&component_ez, 1, volume(vec(0.4, 0.4), vec(1.2, 1.2)), frequencies, 2,
                        /*use_centered_grid=*/true,
                        /*decimation_factor=*/1,
                        /*persist=*/true);

  rebuild_trace rebuilds;
  access_trace accesses;
  compact_trace compact;
  f->backend = new compact_access_backend(*f, rebuilds, accesses, compact);
  f->advance(1);

  dft_flux flux(Ez, Ez, monitor.chunks, monitor.chunks, frequencies, 2, monitor.where, NO_DIRECTION,
                true);
  flux.monitor_lifetime = monitor.monitor_lifetime;
  dft_energy energy(monitor.chunks, monitor.chunks, monitor.chunks, monitor.chunks, frequencies, 2,
                    monitor.where);
  energy.monitor_lifetime = monitor.monitor_lifetime;
  dft_force force(monitor.chunks, monitor.chunks, monitor.chunks, frequencies, 2, monitor.where);
  force.monitor_lifetime = monitor.monitor_lifetime;

  accesses.reads = accesses.dft_reads = 0;
  (void)f->dft_norm();
  delete[] flux.flux();
  (void)flux.complexflux();
  delete[] energy.electric();
  delete[] energy.magnetic();
  delete[] energy.total();
  delete[] force.force();

  CHECK(compact.calls == 8, "public compact queries issued %zu calls, expected 8", compact.calls);
  CHECK(sum_to_all(int(accesses.dft_reads)) == 0,
        "compact DFT query unexpectedly read a complete accumulator");
  CHECK(compact.requests.size() == compact.calls,
        "compact mock did not capture every successful request");
  if (compact.requests.size() >= 8) {
    CHECK(compact.requests[0].kind == DftReductionKind::norm2 &&
              compact.requests[0].result_count == 1,
          "DFT norm constructed the wrong compact request");
    CHECK(compact.requests[1].kind == DftReductionKind::real_weighted_product &&
              compact.requests[1].result_count == 2,
          "flux constructed the wrong compact request");
    CHECK(compact.requests[2].kind == DftReductionKind::complex_weighted_product,
          "complex flux constructed the wrong compact request");
    CHECK(compact.requests[3].terms.empty() ||
              fabs(compact.requests[3].terms[0].weight.real() - 0.5) < 1e-15,
          "electric energy lost its one-half weight");
    CHECK(compact.requests[4].terms.empty() ||
              fabs(compact.requests[4].terms[0].weight.real() - 0.5) < 1e-15,
          "magnetic energy lost its one-half weight");
    CHECK(compact.requests[7].kind == DftReductionKind::real_weighted_product,
          "force constructed the wrong compact request");
  }

  bool saw_persistent_subset = false;
  if (!compact.requests.empty())
    for (size_t i = 0; i < compact.requests[0].terms.size(); ++i) {
      const DftReductionTerm &term = compact.requests[0].terms[i];
      size_t selected = 1;
      size_t maximum = term.region.base;
      for (int axis = 0; axis < 3; ++axis) {
        selected *= term.region.counts[axis];
        maximum += (term.region.counts[axis] - 1) * term.region.strides[axis];
      }
      CHECK(maximum < term.storage_points, "persistent compact region exceeds its allocation");
      saw_persistent_subset = saw_persistent_subset || selected < term.storage_points;
    }
  CHECK(or_to_all(saw_persistent_subset),
        "persistent DFT fixture did not produce an unpadded-in-padded compact region");

  if (!compact.requests.empty() && !compact.requests[0].terms.empty()) {
    DftReductionRequest malformed = compact.requests[0];
    std::complex<double> result;
    bool rejected = false;
    malformed.terms[0].left = invalid_array();
    try {
      f->backend->reduce_dft(malformed, &result, 1);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    CHECK(rejected, "compact mock accepted an invalid array id");

    malformed = compact.requests[0];
    malformed.terms[0].region.counts[0] = 3;
    malformed.terms[0].region.strides[0] = std::numeric_limits<size_t>::max();
    rejected = false;
    try {
      f->backend->reduce_dft(malformed, &result, 1);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    CHECK(rejected, "compact mock accepted an overflowing region");

    malformed = compact.requests[0];
    malformed.terms[0].right = malformed.terms[0].left;
    rejected = false;
    try {
      f->backend->reduce_dft(malformed, &result, 1);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    CHECK(rejected, "compact mock accepted a norm request with a right operand");
  }

  compact.fail_rank = 0;
  compact.fail_call = int(compact.calls);
  bool failed = false;
  try {
    delete[] flux.flux();
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors(),
        "rank-asymmetric compact flux failure was not reconciled");

  compact.fail_call = int(compact.calls);
  failed = false;
  try {
    delete[] energy.electric();
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors(),
        "rank-asymmetric compact electric-energy failure was not reconciled");

  compact.fail_call = int(compact.calls);
  failed = false;
  try {
    delete[] energy.magnetic();
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors(),
        "rank-asymmetric compact magnetic-energy failure was not reconciled");

  compact.fail_call = int(compact.calls);
  failed = false;
  try {
    delete[] force.force();
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors(),
        "rank-asymmetric compact force failure was not reconciled");

  compact.fail_call = int(compact.calls + 1);
  failed = false;
  try {
    delete[] energy.total();
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors(),
        "rank-asymmetric compact total-energy second-call failure was not reconciled");
  all_wait();

  monitor.remove();
  delete f;
  delete s;
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  if (getenv("MEEP_BACKEND_API_BETA_ONLY")) {
    test_resident_beta_fingerprint();
    if (failures) return 1;
    master_printf("backend_api: beta checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_BFAST_ONLY")) {
    test_resident_bfast_fingerprint();
    if (failures) return 1;
    master_printf("backend_api: BFAST checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_CYLINDRICAL_ONLY")) {
    test_resident_cylindrical_fingerprint();
    if (failures) return 1;
    master_printf("backend_api: cylindrical checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_MAGNETIC_ONLY")) {
    test_backend_reselection_invalidates_representation();
    test_resident_magnetic_dispatch();
    if (failures) return 1;
    master_printf("backend_api: magnetic checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_MATERIAL_ONLY")) {
    test_material_phase_transaction();
    test_material_phase_cpu_to_resident_preparation();
    if (failures) return 1;
    master_printf("backend_api: material checks passed\n");
    return 0;
  }

  test_selection();
  test_construction_equivalence();
  test_read_write_roundtrip();
  test_precision_policy();
  test_dft_access_boundaries();
  test_detached_dft_access();
  test_backend_lifecycle_epoch();
  test_backend_reselection_invalidates_representation();
  test_resident_magnetic_dispatch();
  test_material_phase_transaction();
  test_material_phase_cpu_to_resident_preparation();
  test_resident_polarization_preparation();
  test_resident_beta_fingerprint();
  test_resident_bfast_fingerprint();
  test_resident_cylindrical_fingerprint();
  test_classification_change_recompiles();
  test_initialization_plan();
  test_authority_safe_state_rebuild();
  test_cpu_state_rebuild_is_safe_noop();
  test_backend_safe_host_access();
#ifdef HAVE_HDF5
  test_sharded_dft_checkpoint_ordering();
#endif
  test_compact_dft_reduction_boundary();

  if (failures) {
    master_printf("backend_api: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("backend_api: all checks passed\n");
  return 0;
}
