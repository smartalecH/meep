/* End-to-end smoke for the first resident NVIDIA solve_cw slice. */

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <meep.hpp>

#include "backend/backend.hpp"
#include "backend/checkpoint.hpp"
#include "backend/step_plan.hpp"
#include "backend/storage_plan.hpp"
#include "backend/nvidia/nvidia_backend.hpp"
#include "backend/nvidia/nvidia_cw.hpp"
#include "backend/nvidia/nvidia_graph.hpp"
#include "backend/nvidia/nvidia_sources.hpp"
#include "backend/nvidia/runtime.hpp"
#include "meep_internals.hpp"

using namespace meep;

static double one(const vec &) { return 1.0; }
static double conductivity(const vec &) { return 0.07; }

struct custom_source_trace {
  std::vector<double> times;
};

class traced_continuous_source : public continuous_src_time {
public:
  traced_continuous_source(double frequency, custom_source_trace *trace)
      : continuous_src_time(frequency, 0.19), trace_(trace) {}
  src_time *clone() const override { return new traced_continuous_source(*this); }
  std::complex<double> current(double time, double dt) const override {
    if (trace_) trace_->times.push_back(time);
    return src_time::current(time, dt);
  }

private:
  custom_source_trace *trace_;
};

static void require(bool condition, const char *message) {
  if (!condition) meep::abort("nvidia_cw: %s", message);
}

static bool profile_mode_requested() {
  const char *value = getenv("MEEP_NVIDIA_CW_PROFILE_ONLY");
  if (value && *value && strcmp(value, "0")) return true;
  value = getenv("MEEP_NVIDIA_TIMESTEP_CW_PROFILE_ONLY");
  return value && *value && strcmp(value, "0");
}

static std::vector<std::complex<realnum> > dft_values(fields &f, dft_fields monitor,
                                                       component c) {
  int rank = 0;
  size_t dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> values(
      f.get_dft_array(monitor, c, 0, &rank, dims));
  size_t elements = 1;
  for (int axis = 0; axis < rank; ++axis) elements *= dims[axis];
  return std::vector<std::complex<realnum> >(values.get(), values.get() + elements);
}

static void compare_dft_values(const std::vector<std::complex<realnum> > &expected,
                               const std::vector<std::complex<realnum> > &observed,
                               double tolerance) {
  require(expected.size() == observed.size(), "CPU/GPU DFT array sizes differ");
  for (size_t i = 0; i < expected.size(); ++i) {
    const double error = std::abs(std::complex<double>(observed[i]) -
                                  std::complex<double>(expected[i]));
    require(error <= tolerance * (1.0 + std::abs(std::complex<double>(expected[i]))),
            "CPU/GPU final DFT accumulators differ");
  }
}

static void require_canonical_cw_times(const std::vector<double> &times, double dt) {
  require(!times.empty(), "host-custom CW source was not evaluated");
  for (double time : times) {
    const int nearest = int(std::floor(time / dt));
    bool matched = false;
    for (int t = nearest - 2; t <= nearest + 2 && !matched; ++t)
      for (int half = 0; half <= 2; ++half)
        matched = matched || time == cw_source_time(t, dt, 0.5 * half);
    require(matched, "host-custom CW source observed a noncanonical time");
  }
}

static void require_exact_cw_trace(const std::vector<double> &times, int entry_t, double dt,
                                   size_t operator_applications) {
  require(operator_applications > 0, "CW trace has no operator applications");
  std::vector<double> expected;
  double cached_time = std::numeric_limits<double>::quiet_NaN();
  const auto evaluate = [&](double time) {
    if (time != cached_time) {
      expected.push_back(time);
      cached_time = time;
    }
  };
  const auto timestep = [&](int t) {
    evaluate(cw_source_time(t, dt, 0.0));
    evaluate(cw_source_time(t, dt, 0.5));
    evaluate(cw_source_time(t, dt, 0.5));
    evaluate(cw_source_time(t, dt, 1.0));
  };
  timestep(entry_t);
  evaluate(cw_source_time(entry_t + 1, dt, 0.0));
  evaluate(cw_source_time(entry_t + 1, dt, 0.5));
  const size_t ordinary_applications = operator_applications - 1;
  for (size_t application = 0; application < ordinary_applications; ++application)
    timestep(entry_t + 1 + int(application));
  const int final_t = entry_t + 1 + int(ordinary_applications);
  timestep(final_t);
  if (times != expected) {
    size_t mismatch = 0;
    while (mismatch < times.size() && mismatch < expected.size() &&
           times[mismatch] == expected[mismatch])
      ++mismatch;
    master_printf("nvidia_cw trace mismatch observed=%zu expected=%zu first=%zu got=%.17g want=%.17g\n",
                  times.size(), expected.size(), mismatch,
                  mismatch < times.size() ? times[mismatch] : -1.0,
                  mismatch < expected.size() ? expected[mismatch] : -1.0);
  }
  require(times == expected,
          "host-custom CW callback sequence includes verification or misses a stage");
}

static void test_first_cw_compile_retry() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  continuous_src_time source(0.30);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), 1.0);
  gpu.advance(1);
  BackendState *const live_state = gpu.backend_state;
  Executable *const live_executable = gpu.executable;
  const nvidia::memory_accounting before = nvidia::current_memory_accounting();
  const auto inject = [&](nvidia::testing::failure_point point, const char *operation) {
    nvidia::testing::fail_next(point);
    bool rejected = false;
    std::string rejection;
    try {
      (void)gpu.solve_cw(1e-4, 1, std::complex<double>(0.30, 0.0), 2);
    }
    catch (const std::exception &error) {
      rejected = true;
      rejection = error.what();
    }
    nvidia::testing::clear_failure();
    const nvidia::memory_accounting after = nvidia::current_memory_accounting();
    if (!(rejected && rejection.find(operation) != std::string::npos &&
          !gpu.backend->is_poisoned() && gpu.backend_state == live_state &&
          gpu.executable == live_executable &&
          after.device_bytes_current == before.device_bytes_current &&
          after.pinned_bytes_current == before.pinned_bytes_current))
      master_printf("CW compile rollback mismatch point=%d rejected=%d rejection=%s "
                    "poisoned=%d state_same=%d executable_same=%d device=%zu/%zu "
                    "pinned=%zu/%zu\n",
                    int(point), int(rejected), rejection.c_str(),
                    int(gpu.backend->is_poisoned()), int(gpu.backend_state == live_state),
                    int(gpu.executable == live_executable), after.device_bytes_current,
                    before.device_bytes_current, after.pinned_bytes_current,
                    before.pinned_bytes_current);
    require(rejected && rejection.find(operation) != std::string::npos &&
                !gpu.backend->is_poisoned() && gpu.backend_state == live_state &&
                gpu.executable == live_executable &&
                after.device_bytes_current == before.device_bytes_current &&
                after.pinned_bytes_current == before.pinned_bytes_current,
            "first CW compile failure changed the live NVIDIA epoch");
  };
  inject(nvidia::testing::failure_point::device_allocate, "cudaMalloc");
  inject(nvidia::testing::failure_point::host_to_device_copy,
         "cudaMemcpyAsync(host-to-device)");
  gpu.advance(1);
  (void)gpu.solve_cw(1e-4, 1, std::complex<double>(0.30, 0.0), 2);
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend && backend->cw_statistics_for_testing().valid && !gpu.backend->is_poisoned(),
          "first CW compile allocation failure was not retryable");
}

static void test_postlaunch_poison(nvidia::testing::failure_point point,
                                   const char *name) {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  continuous_src_time source(0.30);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), 1.0);
  component monitor_component = Ez;
  dft_fields monitor = gpu.add_dft_fields(&monitor_component, 1,
                                          volume(vec(0.5, 0.5), vec(1.5, 1.5)),
                                          0.30, 0.30, 1, true, 1);
  gpu.advance(1);
  const int entry_t = gpu.t;
  BackendState *const entry_state = gpu.backend_state;
  Executable *const entry_executable = gpu.executable;
  require(gpu.array_catalog != NULL, "poison fixture has no array catalog");
  ArrayId chosen = invalid_array();
  for (size_t i = 0; i < gpu.array_catalog->size(); ++i) {
    const ArraySpec &spec = gpu.array_catalog->spec(ArrayId{uint32_t(i)});
    if (!is_valid(spec.alias_of) && spec.role == array_role::field && spec.elements) {
      chosen = spec.id;
      break;
    }
  }
  require(is_valid(chosen), "poison fixture has no readable field row");
  const ArraySpec &chosen_spec = gpu.array_catalog->spec(chosen);
  std::vector<unsigned char> host_before(chosen_spec.elements * sizeof(realnum));
  std::memcpy(host_before.data(), gpu.array_catalog->resolve_untyped(chosen), host_before.size());
  nvidia::testing::fail_next(point);
  bool failed = false;
  try {
    (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 2);
  }
  catch (const std::exception &) {
    failed = true;
  }
  nvidia::testing::clear_failure();
  bool cw_flags_clear = true;
  for (int chunk = 0; chunk < gpu.num_chunks; ++chunk)
    cw_flags_clear = cw_flags_clear && !gpu.chunks[chunk]->is_solving_cw();
  require(failed && gpu.backend->is_poisoned() && gpu.t == entry_t && cw_flags_clear &&
              gpu.backend_state == entry_state && gpu.executable == entry_executable,
          "injected postlaunch CW failure did not poison the backend");
  require(!std::memcmp(host_before.data(), gpu.array_catalog->resolve_untyped(chosen),
                       host_before.size()),
          "postlaunch CW failure published a partial host field");
  bool read_rejected = false, dft_rejected = false, advance_rejected = false;
  bool replacement_rejected = false, solve_rejected = false;
  try {
    (void)gpu.get_field(Ez, vec(1.0, 1.0));
  }
  catch (const std::exception &) { read_rejected = true; }
  try {
    (void)dft_values(gpu, monitor, monitor_component);
  }
  catch (const std::exception &) { dft_rejected = true; }
  try {
    gpu.advance(1);
  }
  catch (const std::exception &) { advance_rejected = true; }
  try {
    execution_options cpu_options;
    cpu_options.backend = backend_kind::cpu;
    gpu.select_backend(cpu_options);
  }
  catch (const std::exception &) { replacement_rejected = true; }
  try {
    (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 2);
  }
  catch (const std::exception &) { solve_rejected = true; }
  require(read_rejected && dft_rejected && advance_rejected && replacement_rejected &&
              solve_rejected,
          "poisoned CW backend accepted a subsequent operation");
  master_printf("nvidia_cw poison %s: PASS\n", name);
}

static void test_normal_breakdown() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  continuous_src_time source(0.30);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), 1.0);
  nvidia::testing::fail_next(nvidia::testing::failure_point::cw_breakdown);
  require(!gpu.solve_cw(1e-6, 1000, std::complex<double>(0.30, 0.0), 2),
          "injected numerical breakdown unexpectedly converged");
  nvidia::testing::clear_failure();
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend != NULL && !backend->is_poisoned(),
          "normal numerical breakdown poisoned the NVIDIA backend");
  const NvidiaCwStatistics failed = backend->cw_statistics_for_testing();
  require(failed.valid && failed.result.status == CwSolveStatus::breakdown &&
              failed.result.iterations == 1,
          "normal numerical breakdown returned the wrong status");
  require(gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2),
          "NVIDIA solve_cw did not recover after normal breakdown");
}

static void test_cross_backend_rejection(fields &live, NvidiaBackend &live_backend) {
  const std::vector<nvidia::device_properties> devices = nvidia::enumerate_devices();
  if (devices.size() < 2) {
    master_printf("nvidia_cw: cross-backend ownership SKIP (one visible GPU)\n");
    return;
  }
  const int live_device = live_backend.device_ordinal_for_testing();
  int other_device = devices[0].id;
  if (other_device == live_device) other_device = devices[1].id;

  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure other_structure(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.device_id = other_device;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields other(&other_structure, options);
  continuous_src_time source(0.30);
  source.is_integrated = false;
  other.add_point_source(Ez, source, vec(1.0, 1.0), 1.0);
  other.advance(1);
  NvidiaBackend *other_backend = dynamic_cast<NvidiaBackend *>(other.backend);
  require(other_backend != NULL, "cross-backend fixture did not select NVIDIA");
  bool wrong_executable = false, wrong_state = false, wrong_owner = false;
  try {
    live_backend.advance(*other.executable, *live.backend_state, 1);
  }
  catch (const std::exception &) { wrong_executable = true; }
  try {
    live_backend.advance(*live.executable, *other.backend_state, 1);
  }
  catch (const std::exception &) { wrong_state = true; }
  try {
    other_backend->advance(*live.executable, *other.backend_state, 1);
  }
  catch (const std::exception &) { wrong_owner = true; }
  require(wrong_executable && wrong_state && wrong_owner,
          "NVIDIA accepted cross-fields state or executable ownership");
  master_printf("nvidia_cw: cross-backend ownership PASS\n");
}

static void test_profile_workload() {
  const grid_volume gv = vol2d(4.0, 3.0, 10.0);
  structure s(gv, one, pml(0.5));
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  continuous_src_time source(0.30);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.5, 1.5), std::complex<double>(1.0, 0.2));
  component monitor_component = Hx;
  dft_fields monitor = gpu.add_dft_fields(&monitor_component, 1,
                                          volume(vec(1.0, 1.0), vec(2.0, 2.0)),
                                          0.30, 0.30, 1, true, 2);
  (void)monitor;
  gpu.t = 2;
  const bool converged =
      gpu.solve_cw(1e-6, 1000, std::complex<double>(0.30, 0.0), 2);
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend != NULL, "profile workload did not select NVIDIA");
  const NvidiaCwStatistics stats = backend->cw_statistics_for_testing();
  const CwStateLayout layout = build_cw_state_layout(gpu);
  const StepPlan plan = build_step_plan(gpu, StepProgram::solve_cw);
  size_t source_evaluation_operations = 0;
  for (const Operation &operation : plan.operations)
    if (operation.kind == OpKind::evaluate_source_scalars)
      ++source_evaluation_operations;
  require(gpu.descriptors != NULL, "profile workload has no descriptor owner");
  const size_t source_scalar_bytes =
      gpu.descriptors->sources.scalars.size() * sizeof(nvidia::source_scalar);
  require(!converged && stats.valid && stats.result.status == CwSolveStatus::not_converged &&
              stats.result.iterations == 3 && stats.result.operator_applications == 14 &&
              stats.iteration_operator_applications == 12 &&
              stats.iteration_reduction_count == 33 &&
              stats.iteration_scalar_device_to_host_calls == 33 &&
              stats.iteration_scalar_device_to_host_bytes == 33 * sizeof(double),
          "fixed L=2,K=3 profile solver counts differ");
  require(stats.iteration_pack_kernel_launches == 12 * layout.rows.size() &&
              stats.iteration_unpack_kernel_launches == 12 * layout.rows.size() &&
              stats.iteration_vector_kernel_launches == 45 &&
              stats.iteration_operator_kernel_launches == 12 &&
              stats.iteration_reduction_kernel_launches == 66 &&
              stats.timestep_kernel_launches_per_operator > 0 &&
              stats.iteration_timestep_kernel_launches ==
                  12 * stats.timestep_kernel_launches_per_operator &&
              stats.reconciliation_kernel_launches_per_operator > 0 &&
              stats.iteration_reconciliation_kernel_launches ==
                  12 * stats.reconciliation_kernel_launches_per_operator,
          "fixed profile kernel categories differ");
  const size_t categorized_iteration_kernels =
      stats.iteration_pack_kernel_launches + stats.iteration_unpack_kernel_launches +
      stats.iteration_reconciliation_kernel_launches + stats.iteration_vector_kernel_launches +
      stats.iteration_operator_kernel_launches + stats.iteration_reduction_kernel_launches +
      stats.iteration_timestep_kernel_launches;
  master_printf("nvidia_cw profile detail: iter_total=%zu categorized=%zu pack=%zu unpack=%zu "
                "reconcile=%zu vector=%zu operator=%zu reduction_kernels=%zu timestep=%zu "
                "finite=%zu diagnostic_d2h=%zu/%zu vector_bytes=%zu/%zu source_h2d=%zu/%zu "
                "final_dft=%zu\n",
                stats.iteration_kernel_launches, categorized_iteration_kernels,
                stats.iteration_pack_kernel_launches, stats.iteration_unpack_kernel_launches,
                stats.iteration_reconciliation_kernel_launches,
                stats.iteration_vector_kernel_launches,
                stats.iteration_operator_kernel_launches,
                stats.iteration_reduction_kernel_launches,
                stats.iteration_timestep_kernel_launches, stats.finite_check_kernel_launches,
                stats.diagnostic_device_to_host_calls, stats.diagnostic_device_to_host_bytes,
                stats.vector_host_to_device_bytes, stats.vector_device_to_host_bytes,
                stats.iteration_source_scalar_host_to_device_calls,
                stats.iteration_source_scalar_host_to_device_bytes,
                stats.final_dft_kernel_launches);
  require(stats.iteration_kernel_launches == categorized_iteration_kernels &&
              stats.finite_check_kernel_launches == 0 &&
              stats.diagnostic_device_to_host_calls == 0 &&
              stats.diagnostic_device_to_host_bytes == 0 &&
              stats.vector_host_to_device_bytes == 0 && stats.vector_device_to_host_bytes == 0 &&
              stats.iteration_source_scalar_host_to_device_calls ==
                  12 * source_evaluation_operations &&
              stats.iteration_source_scalar_host_to_device_bytes ==
                  12 * source_evaluation_operations * source_scalar_bytes &&
              stats.final_dft_kernel_launches == 1,
          "fixed profile accounting contains an untracked launch or transfer");
  master_printf("nvidia_cw profile: iterations=%d operators=%zu reductions=%zu "
                "iteration_kernels=%zu source_h2d=%zu scalar_d2h=%zu\n",
                stats.result.iterations, stats.result.operator_applications,
                stats.iteration_reduction_count, stats.iteration_kernel_launches,
                stats.source_scalar_host_to_device_bytes,
                stats.iteration_scalar_device_to_host_bytes);
}

static void test_mpi_rejection() {
  nvidia::testing::graph_collective_probe required = {
      "required", true, true, true, true, true, true, true, true, true, true};
  required.instantiate_valid = my_rank() != count_processors() - 1;
  bool graph_rejected = false;
  try { (void)nvidia::testing::reconcile_graph_execution_for_testing(required); }
  catch (const std::exception &) { graph_rejected = true; }
  require(and_to_all(graph_rejected),
          "rank-asymmetric required CW graph instantiate failure was not collective");
  nvidia::testing::graph_collective_probe automatic = required;
  automatic.mode = "auto";
  bool graph_fell_back = false;
  try {
    graph_fell_back =
        !nvidia::testing::reconcile_graph_execution_for_testing(automatic).use_graph;
  }
  catch (...) {}
  require(and_to_all(graph_fell_back),
          "rank-asymmetric automatic CW graph failure did not fall back collectively");

  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  fields gpu(&s, options);
  custom_source_trace trace;
  traced_continuous_source source(0.30, &trace);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), 1.0);
  const nvidia::memory_accounting before = nvidia::current_memory_accounting();
  bool rejected = false;
  try {
    const int local_L = my_rank() == 0 ? 0 : 2;
    (void)gpu.solve_cw(1e-6, 20, std::complex<double>(0.30, 0.0), local_L);
  }
  catch (const std::exception &) { rejected = true; }
  const nvidia::memory_accounting after_request = nvidia::current_memory_accounting();
  require(sum_to_all(int(rejected)) == count_processors() && !gpu.backend_state &&
              !gpu.executable && trace.times.empty() &&
              after_request.device_bytes_current == before.device_bytes_current &&
              after_request.pinned_bytes_current == before.pinned_bytes_current,
          "rank-asymmetric NVIDIA CW request crossed the cheap collective gate");

  continuous_src_time second(0.30);
  second.is_integrated = false;
  gpu.add_point_source(Ez, second, vec(1.2, 1.0), 0.5);
  require(gpu.sources && gpu.sources->next,
          "MPI frequency fixture did not register two source times");
  if (my_rank() == 0) gpu.sources->set_frequency(0.40);
  rejected = false;
  try {
    (void)gpu.solve_cw(1e-6, 20, 2);
  }
  catch (const std::exception &) { rejected = true; }
  const nvidia::memory_accounting after_frequency = nvidia::current_memory_accounting();
  require(sum_to_all(int(rejected)) == count_processors() && !gpu.backend_state &&
              !gpu.executable && trace.times.empty() &&
              after_frequency.device_bytes_current == before.device_bytes_current &&
              after_frequency.pinned_bytes_current == before.pinned_bytes_current,
          "rank-asymmetric NVIDIA CW frequency crossed the cheap collective gate");
  master_printf("nvidia_cw: MPI rejection PASS at np=%d\n", count_processors());
}

struct cw_graph_outcome {
  CwSolveResult result;
  NvidiaCwStatistics statistics;
  std::complex<double> field;
  std::vector<std::complex<realnum> > dft;
  std::vector<std::complex<realnum> > not_due_dft;
  std::vector<double> source_times;
  double dt;
};

static cw_graph_outcome run_cw_graph_fixture(const char *mode,
                                             precision_policy_kind precision_policy) {
  setenv("MEEP_NVIDIA_GRAPH_MODE", mode, 1);
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  custom_source_trace trace;
  traced_continuous_source source(0.30, &trace);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), std::complex<double>(1.0, 0.2));
  component monitor_component = Ez;
  dft_fields monitor = gpu.add_dft_fields(&monitor_component, 1,
                                          volume(vec(0.5, 0.5), vec(1.5, 1.5)),
                                          0.30, 0.30, 1, true, 2);
  dft_fields not_due_monitor = gpu.add_dft_fields(&monitor_component, 1,
                                                  volume(vec(0.5, 0.5), vec(1.5, 1.5)),
                                                  0.30, 0.30, 1, true, 3);
  gpu.t = 2;
  (void)gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2);
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend != NULL, "CW graph fixture did not retain NVIDIA");
  cw_graph_outcome outcome;
  outcome.statistics = backend->cw_statistics_for_testing();
  outcome.result = outcome.statistics.result;
  outcome.field = gpu.get_field(Ez, vec(1.0, 1.0));
  outcome.dft = dft_values(gpu, monitor, monitor_component);
  outcome.not_due_dft = dft_values(gpu, not_due_monitor, monitor_component);
  outcome.source_times = trace.times;
  outcome.dt = gpu.dt;
  return outcome;
}

static void test_cw_graph_operator_fixture(precision_policy_kind precision_policy) {
  const cw_graph_outcome eager = run_cw_graph_fixture("eager", precision_policy);
  nvidia::testing::reset_graph_accounting();
  const cw_graph_outcome graph = run_cw_graph_fixture("required", precision_policy);
  const nvidia::graph_accounting teardown = nvidia::testing::current_graph_accounting();
  require(eager.statistics.valid && !eager.statistics.graph_enabled &&
              graph.statistics.valid && graph.statistics.graph_enabled,
          "required CW graph mode was not retained after admission");
  require(graph.statistics.graph_count > 3 &&
              graph.statistics.graph_capture_count == graph.statistics.graph_count &&
              graph.statistics.graph_instantiate_count == graph.statistics.graph_count &&
              graph.statistics.graph_launch_count > 0 &&
              graph.statistics.graph_scalar_write_count > 0,
          "CW graph fixture has incomplete capture/instantiate/replay accounting");
  require(graph.statistics.graph_pack_launch_count > 0 &&
              graph.statistics.graph_unpack_launch_count > 0 &&
              graph.statistics.graph_rhs_launch_count > 0 &&
              graph.statistics.graph_vector_launch_count > 0 &&
              graph.statistics.graph_reduction_launch_count > 0 &&
              graph.statistics.graph_operator_launch_count > 0 &&
              graph.statistics.graph_final_dft_launch_count == 1 &&
              graph.statistics.pack_kernel_launches > 0 &&
              graph.statistics.unpack_kernel_launches > 0 &&
              graph.statistics.rhs_source_kernel_launches > 0 &&
              graph.statistics.vector_kernel_launches > 0 &&
              graph.statistics.reduction_kernel_launches > 0 &&
              graph.statistics.operator_kernel_launches > 0,
          "CW graph variants omitted required workspace or CwPlan-owned work");
  require(eager.result.status == graph.result.status &&
              eager.result.iterations == graph.result.iterations &&
              eager.result.operator_applications == graph.result.operator_applications,
          "CW eager/required solver status or iteration accounting differs");
  const double error = std::abs(eager.field - graph.field);
  const double tolerance = precision_policy == precision_policy_kind::native &&
                                   sizeof(realnum) == sizeof(double)
                               ? 2e-6
                               : 2e-3;
  require(error <= tolerance * (1.0 + std::abs(eager.field)),
          "CW eager/required operator graph field differs");
  compare_dft_values(eager.dft, graph.dft, tolerance);
  compare_dft_values(eager.not_due_dft, graph.not_due_dft, tolerance);
  require(eager.source_times == graph.source_times,
          "CW graph changed host-custom source callback ordering");
  require_canonical_cw_times(graph.source_times, graph.dt);
  require(teardown.end_captures == teardown.graph_destroys &&
              teardown.instantiates == teardown.executable_destroys,
          "CW graph fixture did not tear down every graph owner");
  master_printf("nvidia_cw graph operator: graphs=%zu launches=%zu field_error=%.9g PASS\n",
                graph.statistics.graph_count, graph.statistics.graph_launch_count, error);
}

static void test_cw_graph_checkpoint_roundtrip(precision_policy_kind precision_policy) {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "required", 1);
  nvidia::testing::reset_graph_accounting();
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  continuous_src_time source(0.30, 0.19);
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), std::complex<double>(1.0, 0.2));
  component monitor_component = Ez;
  dft_fields monitor = gpu.add_dft_fields(&monitor_component, 1,
                                          volume(vec(0.5, 0.5), vec(1.5, 1.5)),
                                          0.30, 0.30, 1, true, 2);
  gpu.t = 2;
  require(gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2),
          "CW checkpoint fixture did not converge");
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend && gpu.backend_state && gpu.executable && gpu.backend_state->cw_executable,
          "CW checkpoint fixture did not publish both executable owners");
  require(backend->cw_statistics_for_testing().graph_enabled,
          "CW checkpoint fixture did not retain required graph mode");
  const std::complex<double> saved_field = gpu.get_field(Ez, vec(1.0, 1.0));
  const std::vector<std::complex<realnum> > saved_dft =
      dft_values(gpu, monitor, monitor_component);
  const nvidia::graph_accounting before = nvidia::testing::current_graph_accounting();
  const size_t definitions = before.end_captures - before.graph_destroys;
  const size_t executables = before.instantiates - before.executable_destroys;
  const char *filename = "/tmp/meep-pr6-5-cw-checkpoint.h5";
  std::remove(filename);
  gpu.dump(filename, false);
  require(gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2),
          "CW checkpoint mutation solve did not converge");
  gpu.load(filename, false);
  const nvidia::graph_accounting after_load = nvidia::testing::current_graph_accounting();
  require(!gpu.backend_state && !gpu.executable &&
              after_load.graph_destroys - before.graph_destroys == definitions &&
              after_load.executable_destroys - before.executable_destroys == executables,
          "CW checkpoint load did not retire ordinary/CW graph owners exactly once");
  const double tolerance = precision_policy == precision_policy_kind::native &&
                                   sizeof(realnum) == sizeof(double)
                               ? 2e-6
                               : 2e-3;
  require(std::abs(gpu.get_field(Ez, vec(1.0, 1.0)) - saved_field) <=
              tolerance * (1.0 + std::abs(saved_field)),
          "CW checkpoint did not restore the converged field");
  compare_dft_values(saved_dft, dft_values(gpu, monitor, monitor_component), tolerance);
  require(gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2),
          "CW checkpoint continuation did not converge");
  const nvidia::graph_accounting rebuilt = nvidia::testing::current_graph_accounting();
  require(backend->cw_statistics_for_testing().graph_enabled &&
              rebuilt.instantiates - after_load.instantiates == executables,
          "CW checkpoint continuation did not rebuild fresh ordinary/CW graphs");
  std::remove(filename);
}

static src_vol *first_nonempty_source(fields &owner) {
  for (int chunk = 0; chunk < owner.num_chunks; ++chunk) {
    FOR_FIELD_TYPES(ft) {
      for (src_vol &source : owner.chunks[chunk]->sources[ft])
        if (source.num_points()) return &source;
    }
  }
  return NULL;
}

static void test_cw_source_value_graph_reuse() {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "required", 1);
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure cpu_structure(gv, one, no_pml());
  structure gpu_structure(gv, one, no_pml());
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&gpu_structure, options);
  continuous_src_time cpu_time(0.30), gpu_time(0.30);
  cpu_time.is_integrated = gpu_time.is_integrated = false;
  cpu.add_point_source(Ez, cpu_time, vec(1.0, 1.0), std::complex<double>(1.0, 0.2));
  gpu.add_point_source(Ez, gpu_time, vec(1.0, 1.0), std::complex<double>(1.0, 0.2));
  cpu.t = gpu.t = 2;
  (void)cpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2);
  (void)gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2);
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend && backend->cw_statistics_for_testing().graph_enabled,
          "CW source-value reuse fixture did not retain required graph mode");
  src_vol *cpu_source = first_nonempty_source(cpu);
  src_vol *gpu_source = first_nonempty_source(gpu);
  require(cpu_source && gpu_source &&
              cpu_source->num_points() == gpu_source->num_points(),
          "CW source-value reuse fixture has no corresponding source");
  const std::complex<double> amplitude =
      cpu_source->amplitude_at(0) + std::complex<double>(0.125, -0.0625);
  cpu_source->set_amplitude(0, amplitude);
  gpu_source->set_amplitude(0, amplitude);
  invalidate(cpu, MutationKind::source_values, "CPU CW graph source-value reuse test");
  invalidate(gpu, MutationKind::source_values, "NVIDIA CW graph source-value reuse test");
  BackendState *const state = gpu.backend_state;
  Executable *const ordinary = gpu.executable;
  Executable *const cw = state->cw_executable;
  const NvidiaExecutableCacheStatistics cache_before =
      backend->executable_cache_statistics_for_testing();
  const nvidia::graph_accounting before = nvidia::testing::current_graph_accounting();
  (void)cpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2);
  (void)gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2);
  const nvidia::graph_accounting after = nvidia::testing::current_graph_accounting();
  const NvidiaExecutableCacheStatistics cache_after =
      backend->executable_cache_statistics_for_testing();
  require(gpu.backend_state == state && gpu.executable == ordinary &&
              state->cw_executable == cw,
          "CW source-value mutation replaced a structurally compatible owner");
  require(after.creates == before.creates && after.begin_captures == before.begin_captures &&
              after.instantiates == before.instantiates &&
              after.graph_destroys == before.graph_destroys &&
              after.executable_destroys == before.executable_destroys,
          "CW source-value mutation recaptured or retired a graph");
  require(cache_before.ordinary_resource_generation != 0 &&
              cache_before.cw_resource_generation != 0 &&
              cache_after.ordinary_resource_generation ==
                  cache_before.ordinary_resource_generation &&
              cache_after.cw_resource_generation == cache_before.cw_resource_generation &&
              cache_after.executable_build_count == cache_before.executable_build_count &&
              cache_after.source_value_reuse_count ==
                  cache_before.source_value_reuse_count + 3,
          "CW source-value refresh changed resource generation/build accounting");
  const std::complex<double> expected = cpu.get_field(Ez, vec(1.0, 1.0));
  const std::complex<double> actual = gpu.get_field(Ez, vec(1.0, 1.0));
  require(std::abs(actual - expected) <= 2e-3 * (1.0 + std::abs(expected)),
          "CW source-value graph reuse changed CPU/NVIDIA parity");
}

static void test_cw_source_value_refresh_poison_cleanup() {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "required", 1);
  nvidia::testing::reset_graph_accounting();
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  for (int failure = 0; failure < 2; ++failure) {
    structure s(gv, one, no_pml());
    fields gpu(&s, options);
    continuous_src_time source_time(0.30);
    source_time.is_integrated = false;
    gpu.add_point_source(Ez, source_time, vec(1.0, 1.0), 1.0);
    gpu.t = 2;
    (void)gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2);
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
    src_vol *source = first_nonempty_source(gpu);
    require(backend && source && gpu.backend_state->cw_executable,
            "CW source refresh retry fixture was not prepared");
    BackendState *const state = gpu.backend_state;
    Executable *const ordinary = gpu.executable;
    Executable *const cw = state->cw_executable;
    const NvidiaExecutableCacheStatistics cache_before =
        backend->executable_cache_statistics_for_testing();
    const nvidia::graph_accounting graphs_before =
        nvidia::testing::current_graph_accounting();
    source->set_amplitude(0, source->amplitude_at(0) + 0.125);
    invalidate(gpu, MutationKind::source_values, "CW source refresh retry test");
    if (failure == 0)
      nvidia::testing::fail_next(nvidia::testing::failure_point::cw_source_value_copy);
    else
      backend_set_cw_plan_corruption_for_testing(true);
    bool rejected = false;
    try { (void)gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
    backend_set_cw_plan_corruption_for_testing(false);
    const NvidiaExecutableCacheStatistics cache_failed =
        backend->executable_cache_statistics_for_testing();
    const nvidia::graph_accounting graphs_failed =
        nvidia::testing::current_graph_accounting();
    require(rejected && !gpu.backend->is_poisoned() && gpu.backend_state == state &&
                gpu.executable == ordinary && state->cw_executable == cw &&
                cache_failed.ordinary_resource_generation ==
                    cache_before.ordinary_resource_generation &&
                cache_failed.cw_resource_generation == cache_before.cw_resource_generation &&
                cache_failed.executable_build_count == cache_before.executable_build_count &&
                cache_failed.source_value_reuse_count == cache_before.source_value_reuse_count &&
                graphs_failed.creates == graphs_before.creates &&
                graphs_failed.begin_captures == graphs_before.begin_captures &&
                graphs_failed.instantiates == graphs_before.instantiates &&
                graphs_failed.graph_destroys == graphs_before.graph_destroys &&
                graphs_failed.executable_destroys == graphs_before.executable_destroys,
            failure == 0
                ? "pre-enqueue CW source copy failure changed live owners or accounting"
                : "later CW plan validation failure partially published source refresh");
    (void)gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2);
    const NvidiaExecutableCacheStatistics cache_after =
        backend->executable_cache_statistics_for_testing();
    require(gpu.backend_state == state && gpu.executable == ordinary &&
                state->cw_executable == cw &&
                cache_after.source_value_reuse_count ==
                    cache_before.source_value_reuse_count + 3,
            "retryable CW source refresh did not reuse all three owners");
  }

  nvidia::testing::reset_graph_accounting();
  size_t owned_definitions = 0;
  size_t owned_executables = 0;
  {
    structure s(gv, one, no_pml());
    fields gpu(&s, options);
    continuous_src_time source_time(0.30);
    source_time.is_integrated = false;
    gpu.add_point_source(Ez, source_time, vec(1.0, 1.0), 1.0);
    gpu.t = 2;
    (void)gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2);
    NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
    src_vol *source = first_nonempty_source(gpu);
    require(backend && source && gpu.backend_state->cw_executable,
            "CW source refresh poison fixture was not prepared");
    const nvidia::graph_accounting prepared = nvidia::testing::current_graph_accounting();
    owned_definitions = prepared.end_captures - prepared.graph_destroys;
    owned_executables = prepared.instantiates - prepared.executable_destroys;
    Executable *const ordinary = gpu.executable;
    Executable *const cw = gpu.backend_state->cw_executable;
    source->set_amplitude(0, source->amplitude_at(0) + 0.125);
    invalidate(gpu, MutationKind::source_values, "CW source refresh poison test");
    nvidia::testing::fail_next(nvidia::testing::failure_point::cw_source_value_sync);
    bool rejected = false;
    try { (void)gpu.solve_cw(1e-4, 1000, std::complex<double>(0.30, 0.0), 2); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
    require(rejected && gpu.backend->is_poisoned() && gpu.executable == ordinary &&
                gpu.backend_state->cw_executable == cw,
            "post-enqueue CW source refresh failure did not poison and retain owners");
  }
  const nvidia::graph_accounting cleanup = nvidia::testing::current_graph_accounting();
  require(owned_definitions > 0 && owned_executables > 0 &&
              cleanup.graph_destroys == owned_definitions &&
              cleanup.executable_destroys == owned_executables,
          "poisoned CW source refresh did not release every graph owner exactly once");
}

static void test_cw_graph_compile_rollback() {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "required", 1);
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  continuous_src_time source(0.30);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), 1.0);
  gpu.advance(1);
  BackendState *const state = gpu.backend_state;
  Executable *const ordinary = gpu.executable;
  Executable *const prior_cw = state->cw_executable;
  const nvidia::memory_accounting memory_before = nvidia::current_memory_accounting();
  nvidia::testing::reset_graph_accounting();
  const nvidia::graph_accounting graphs_before = nvidia::testing::current_graph_accounting();
  nvidia::testing::fail_next(nvidia::testing::failure_point::cw_graph_instantiate);
  bool rejected = false;
  try {
    (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 2);
  }
  catch (const std::exception &) { rejected = true; }
  nvidia::testing::clear_failure();
  const nvidia::memory_accounting memory_after = nvidia::current_memory_accounting();
  const nvidia::graph_accounting graphs_after = nvidia::testing::current_graph_accounting();
  if (!(rejected && !gpu.backend->is_poisoned() && gpu.backend_state == state &&
        gpu.executable == ordinary && state->cw_executable == prior_cw &&
        memory_after.device_bytes_current == memory_before.device_bytes_current &&
        memory_after.pinned_bytes_current == memory_before.pinned_bytes_current &&
        graphs_after.end_captures - graphs_after.graph_destroys ==
            graphs_before.end_captures - graphs_before.graph_destroys &&
        graphs_after.instantiates - graphs_after.executable_destroys ==
            graphs_before.instantiates - graphs_before.executable_destroys))
    master_printf("CW graph rollback rejected=%d poison=%d state=%d ordinary=%d cw=%d "
                  "device=%zu/%zu pinned=%zu/%zu graph=%zu/%zu exec=%zu/%zu\n",
                  int(rejected), int(gpu.backend->is_poisoned()), int(gpu.backend_state == state),
                  int(gpu.executable == ordinary), int(state->cw_executable == prior_cw),
                  memory_after.device_bytes_current, memory_before.device_bytes_current,
                  memory_after.pinned_bytes_current, memory_before.pinned_bytes_current,
                  graphs_after.end_captures, graphs_after.graph_destroys,
                  graphs_after.instantiates, graphs_after.executable_destroys);
  require(rejected && !gpu.backend->is_poisoned() && gpu.backend_state == state &&
              gpu.executable == ordinary && state->cw_executable == prior_cw &&
              memory_after.device_bytes_current == memory_before.device_bytes_current &&
              memory_after.pinned_bytes_current == memory_before.pinned_bytes_current &&
              graphs_after.end_captures - graphs_after.graph_destroys ==
                  graphs_before.end_captures - graphs_before.graph_destroys &&
              graphs_after.instantiates - graphs_after.executable_destroys ==
                  graphs_before.instantiates - graphs_before.executable_destroys,
          "failed CW graph preflight published or leaked a replacement");
  (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 2);
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend && backend->cw_statistics_for_testing().graph_enabled,
          "CW graph preflight was not retryable after rollback");
}

static void test_cw_graph_auto_fallback() {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "auto", 1);
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  continuous_src_time source(0.30);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), 1.0);
  gpu.advance(1);
  nvidia::testing::fail_next(nvidia::testing::failure_point::cw_graph_instantiate);
  (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 2);
  nvidia::testing::clear_failure();
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  const NvidiaCwStatistics stats =
      backend ? backend->cw_statistics_for_testing() : NvidiaCwStatistics();
  require(backend && stats.valid && !stats.graph_enabled &&
              stats.graph_launch_count == 0 && stats.pack_kernel_launches > 0 &&
              stats.reduction_kernel_launches > 0 && !gpu.backend->is_poisoned(),
          "automatic CW graph instantiate failure did not select whole-variant eager");
}

static void test_cw_graph_local_cleanup_failure(
    nvidia::testing::failure_point primary,
    nvidia::testing::failure_point cleanup, const char *name) {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "auto", 1);
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  continuous_src_time source(0.30);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), 1.0);
  gpu.advance(1);
  BackendState *const state = gpu.backend_state;
  Executable *const ordinary = gpu.executable;
  const nvidia::memory_accounting memory_before = nvidia::current_memory_accounting();
  nvidia::testing::reset_graph_accounting();
  nvidia::testing::fail_next_then(primary, cleanup);
  bool rejected = false;
  try { (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 2); }
  catch (const std::exception &) { rejected = true; }
  nvidia::testing::clear_failure();
  const nvidia::memory_accounting memory_after = nvidia::current_memory_accounting();
  const nvidia::graph_accounting graphs = nvidia::testing::current_graph_accounting();
  require(rejected && !gpu.backend->is_poisoned() && gpu.backend_state == state &&
              gpu.executable == ordinary && !state->cw_executable &&
              memory_after.device_bytes_current == memory_before.device_bytes_current &&
              memory_after.pinned_bytes_current == memory_before.pinned_bytes_current &&
              graphs.end_captures == graphs.graph_destroys &&
              graphs.instantiates == graphs.executable_destroys,
          "automatic CW graph candidate cleanup failure did not fail closed and roll back");
  (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 2);
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(backend && backend->cw_statistics_for_testing().graph_enabled,
          "CW graph candidate cleanup failure was not retryable");
  master_printf("nvidia_cw graph local rollback (%s): PASS\n", name);
}

static void test_cw_graph_checked_clear(nvidia::testing::failure_point failure,
                                        const char *name, bool alternate_device) {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "required", 1);
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  continuous_src_time source(0.30);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), 1.0);
  (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 2);
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  const NvidiaCwStatistics stats =
      backend ? backend->cw_statistics_for_testing() : NvidiaCwStatistics();
  require(backend && stats.graph_enabled && stats.graph_count > 0,
          "CW checked-clear fixture did not publish graph owners");
  const nvidia::memory_accounting memory_before = nvidia::current_memory_accounting();
  nvidia::testing::reset_graph_accounting();
  bool rejected = false;
  nvidia::testing::fail_next(failure);
  if (alternate_device) {
    const std::vector<nvidia::device_properties> devices = nvidia::enumerate_devices();
    require(devices.size() > 1, "CW device-restore cleanup test requires two visible GPUs");
    const int selected = backend->device_ordinal_for_testing();
    const int alternate = selected == 0 ? 1 : 0;
    nvidia::device_scope scope(alternate);
    try { backend->clear_cw_graphs_for_testing(); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
  }
  else {
    try { backend->clear_cw_graphs_for_testing(); }
    catch (const std::exception &) { rejected = true; }
    nvidia::testing::clear_failure();
  }
  const nvidia::graph_accounting failed = nvidia::testing::current_graph_accounting();
  const nvidia::memory_accounting memory_failed = nvidia::current_memory_accounting();
  require(rejected && memory_failed.device_bytes_current == memory_before.device_bytes_current &&
              memory_failed.pinned_bytes_current == memory_before.pinned_bytes_current,
          "CW checked clear did not retain captured scalar/workspace lifetimes");
  if (failure == nvidia::testing::failure_point::graph_exec_destroy)
    require(failed.graph_destroys > 0 &&
                failed.executable_destroys + 1 == failed.graph_destroys,
            "CW executable-destroy failure did not retain exactly one retryable owner");
  else if (failure == nvidia::testing::failure_point::graph_destroy)
    require(failed.executable_destroys > 0 &&
                failed.graph_destroys + 1 == failed.executable_destroys,
            "CW definition-destroy failure did not retain exactly one retryable owner");
  else
    require(failed.executable_destroys > 0 &&
                failed.executable_destroys == failed.graph_destroys,
            "CW device-restore failure changed graph release accounting");

  backend->clear_cw_graphs_for_testing();
  const nvidia::graph_accounting retried = nvidia::testing::current_graph_accounting();
  const nvidia::memory_accounting memory_retried = nvidia::current_memory_accounting();
  const size_t expected_graphs =
      std::max(failed.executable_destroys, failed.graph_destroys);
  require(retried.executable_destroys == expected_graphs &&
              retried.graph_destroys == expected_graphs &&
              memory_retried.device_bytes_current < memory_before.device_bytes_current &&
              memory_retried.pinned_bytes_current == memory_before.pinned_bytes_current,
          "CW checked clear was not exactly retryable");
  master_printf("nvidia_cw graph checked clear (%s): PASS\n", name);
}

static void test_cw_graph_workspace_replacement() {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "required", 1);
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  continuous_src_time source(0.30);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), 1.0);
  (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 1);
  NvidiaBackend *backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  BackendState *const state = gpu.backend_state;
  Executable *const first = state ? state->cw_executable : NULL;
  NvidiaCwStatistics stats =
      backend ? backend->cw_statistics_for_testing() : NvidiaCwStatistics();
  require(backend && first && stats.graph_enabled && stats.workspace_allocations == 1,
          "initial CW graph workspace was not published once");
  (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 1);
  stats = backend->cw_statistics_for_testing();
  require(gpu.backend_state == state && state->cw_executable == first &&
              stats.workspace_allocations == 1,
          "same-shape CW graph solve did not reuse the current executable/workspace");

  const nvidia::memory_accounting before = nvidia::current_memory_accounting();
  nvidia::testing::fail_next(nvidia::testing::failure_point::cw_graph_instantiate);
  bool rejected = false;
  try { (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 3); }
  catch (const std::exception &) { rejected = true; }
  nvidia::testing::clear_failure();
  const nvidia::memory_accounting after = nvidia::current_memory_accounting();
  require(rejected && !gpu.backend->is_poisoned() && gpu.backend_state == state &&
              state->cw_executable == first &&
              after.device_bytes_current == before.device_bytes_current &&
              after.pinned_bytes_current == before.pinned_bytes_current,
          "failed CW graph workspace growth retired the current executable");
  (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 1);
  require(state->cw_executable == first,
          "current CW graph executable was unusable after failed replacement");
  (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 3);
  stats = backend->cw_statistics_for_testing();
  require(state->cw_executable != first && stats.graph_enabled &&
              stats.workspace_allocations == 2,
          "successful CW graph workspace growth did not publish one replacement");
}

static void test_cw_graph_launch_failure() {
  setenv("MEEP_NVIDIA_GRAPH_MODE", "required", 1);
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, one, no_pml());
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&s, options);
  continuous_src_time source(0.30);
  source.is_integrated = false;
  gpu.add_point_source(Ez, source, vec(1.0, 1.0), 1.0);
  gpu.advance(1);
  nvidia::testing::fail_next(nvidia::testing::failure_point::graph_launch);
  bool failed = false;
  try {
    (void)gpu.solve_cw(1e-4, 10, std::complex<double>(0.30, 0.0), 2);
  }
  catch (const std::exception &) { failed = true; }
  nvidia::testing::clear_failure();
  require(failed && gpu.backend->is_poisoned(),
          "CW graph launch failure did not poison the backend");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  if (count_processors() > 1) {
    test_mpi_rejection();
    return 0;
  }
  if (profile_mode_requested()) {
    require(count_processors() == 1, "initial slice requires one MPI rank");
    test_profile_workload();
    master_printf("nvidia_cw: profile-only PASS\n");
    return 0;
  }
  if (getenv("MEEP_NVIDIA_CW_SOURCE_REUSE_ONLY")) {
    test_cw_source_value_graph_reuse();
    test_cw_source_value_refresh_poison_cleanup();
    master_printf("nvidia_cw: source-value reuse PASS\n");
    return 0;
  }
  bool use_conductivity = true, use_integrated = true, use_magnetic = true;
  precision_policy_kind precision_policy = precision_policy_kind::native;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--no-conductivity")) use_conductivity = false;
    if (!strcmp(argv[i], "--no-integrated")) use_integrated = false;
    if (!strcmp(argv[i], "--electric-only")) use_magnetic = false;
    if (!strcmp(argv[i], "--precision=mixed")) precision_policy = precision_policy_kind::mixed;
    if (!strcmp(argv[i], "--precision=f32")) precision_policy = precision_policy_kind::f32;
    if (!strcmp(argv[i], "--precision=native")) precision_policy = precision_policy_kind::native;
  }
  if (getenv("MEEP_NVIDIA_REQUIRE_NATIVE_SINGLE"))
    require(sizeof(realnum) == sizeof(float) && precision_policy == precision_policy_kind::native,
            "native-single validation requires a single-precision native build");
  if (getenv("MEEP_NVIDIA_CW_CHECKPOINT_ONLY")) {
    require(count_processors() == 1, "CW checkpoint fixture requires one MPI rank");
    test_cw_graph_checkpoint_roundtrip(precision_policy);
    return 0;
  }
  if (getenv("MEEP_NVIDIA_CW_GRAPH_ONLY")) {
    require(count_processors() == 1, "CW graph fixture requires one MPI rank");
    test_cw_graph_compile_rollback();
    test_cw_graph_local_cleanup_failure(
        nvidia::testing::failure_point::cw_graph_capture,
        nvidia::testing::failure_point::graph_destroy, "capture/definition-destroy");
    test_cw_graph_local_cleanup_failure(
        nvidia::testing::failure_point::cw_graph_instantiate,
        nvidia::testing::failure_point::graph_exec_destroy,
        "instantiate/executable-destroy");
    test_cw_graph_auto_fallback();
    test_cw_graph_checked_clear(nvidia::testing::failure_point::graph_exec_destroy,
                                "graph-exec-destroy", false);
    test_cw_graph_checked_clear(nvidia::testing::failure_point::graph_destroy,
                                "graph-destroy", false);
    if (nvidia::enumerate_devices().size() > 1)
      test_cw_graph_checked_clear(nvidia::testing::failure_point::device_restore,
                                  "device-restore", true);
    test_cw_graph_workspace_replacement();
    test_cw_graph_operator_fixture(precision_policy);
    test_cw_graph_checkpoint_roundtrip(precision_policy);
    test_cw_source_value_graph_reuse();
    test_cw_graph_launch_failure();
    return 0;
  }
  require(count_processors() == 1, "initial slice requires one MPI rank");
  test_first_cw_compile_retry();
  test_normal_breakdown();
  test_postlaunch_poison(nvidia::testing::failure_point::cw_pack, "pack");
  test_postlaunch_poison(nvidia::testing::failure_point::cw_timestep, "timestep");
  test_postlaunch_poison(nvidia::testing::failure_point::cw_reduction, "reduction");
  test_postlaunch_poison(nvidia::testing::failure_point::cw_scalar_copy, "scalar-copy");
  test_postlaunch_poison(nvidia::testing::failure_point::cw_unpack, "unpack");
  const grid_volume gv = voltwo(8.0, 3.0, 10.0);
  structure cpu_structure(gv, one, pml(1.0));
  structure gpu_structure(gv, one, pml(1.0));
  if (use_conductivity) {
    cpu_structure.set_conductivity(Dz, conductivity);
    cpu_structure.set_conductivity(Bz, conductivity);
    gpu_structure.set_conductivity(Dz, conductivity);
    gpu_structure.set_conductivity(Bz, conductivity);
  }
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = precision_policy;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields cpu(&cpu_structure);
  fields gpu(&gpu_structure, options);
  continuous_src_time cpu_e(0.30), gpu_e(0.30), cpu_h(0.30), gpu_h(0.30);
  continuous_src_time cpu_e_integrated(0.30), gpu_e_integrated(0.30),
      cpu_h_integrated(0.30), gpu_h_integrated(0.30);
  cpu_e.is_integrated = gpu_e.is_integrated = false;
  cpu_h.is_integrated = gpu_h.is_integrated = false;
  cpu_e_integrated.is_integrated = gpu_e_integrated.is_integrated = true;
  cpu_h_integrated.is_integrated = gpu_h_integrated.is_integrated = true;
  custom_source_trace cpu_custom_trace, gpu_custom_trace;
  traced_continuous_source cpu_custom(0.30, &cpu_custom_trace);
  traced_continuous_source gpu_custom(0.30, &gpu_custom_trace);
  cpu_custom.is_integrated = gpu_custom.is_integrated = false;
  cpu.add_point_source(Ez, cpu_e, vec(2.0, 1.5), std::complex<double>(1.0, 0.2));
  gpu.add_point_source(Ez, gpu_e, vec(2.0, 1.5), std::complex<double>(1.0, 0.2));
  cpu.add_point_source(Ey, cpu_custom, vec(2.1, 1.4), std::complex<double>(0.05, -0.02));
  gpu.add_point_source(Ey, gpu_custom, vec(2.1, 1.4), std::complex<double>(0.05, -0.02));
  if (use_integrated) {
    cpu.add_point_source(Ez, cpu_e_integrated, vec(2.2, 1.5), std::complex<double>(0.2, -0.1));
    gpu.add_point_source(Ez, gpu_e_integrated, vec(2.2, 1.5), std::complex<double>(0.2, -0.1));
  }
  if (use_magnetic) {
    cpu.add_point_source(Hz, cpu_h, vec(2.4, 1.5), std::complex<double>(0.15, 0.05));
    gpu.add_point_source(Hz, gpu_h, vec(2.4, 1.5), std::complex<double>(0.15, 0.05));
    if (use_integrated) {
      cpu.add_point_source(Hz, cpu_h_integrated, vec(2.6, 1.5),
                           std::complex<double>(-0.08, 0.03));
      gpu.add_point_source(Hz, gpu_h_integrated, vec(2.6, 1.5),
                           std::complex<double>(-0.08, 0.03));
    }
  }
  cpu.t = gpu.t = 2;
  const bool reduced_storage = sizeof(realnum) == sizeof(float) ||
                               precision_policy != precision_policy_kind::native;
  const double tolerance = reduced_storage ? 1e-4 : 5e-7;
  require(!gpu.solve_cw(tolerance, 1, std::complex<double>(0.30, 0.0), 2),
          "one-iteration resident solve unexpectedly converged");
  NvidiaBackend *nvidia_backend = dynamic_cast<NvidiaBackend *>(gpu.backend);
  require(nvidia_backend != NULL, "resident solve did not retain the NVIDIA backend");
  const NvidiaCwStatistics failed_stats = nvidia_backend->cw_statistics_for_testing();
  require(failed_stats.valid && failed_stats.result.status == CwSolveStatus::not_converged &&
              failed_stats.result.iterations == 1 &&
              failed_stats.result.operator_applications == 6 && !gpu.backend->is_poisoned(),
          "ordinary CW nonconvergence did not remain retryable");

  component monitor_component = Ez;
  const volume monitor_region(vec(3.2, 1.1), vec(4.8, 1.9));
  dft_fields cpu_due =
      cpu.add_dft_fields(&monitor_component, 1, monitor_region, 0.30, 0.30, 1, true, 2);
  dft_fields gpu_due =
      gpu.add_dft_fields(&monitor_component, 1, monitor_region, 0.30, 0.30, 1, true, 2);
  dft_fields cpu_not_due =
      cpu.add_dft_fields(&monitor_component, 1, monitor_region, 0.30, 0.30, 1, true, 3);
  dft_fields gpu_not_due =
      gpu.add_dft_fields(&monitor_component, 1, monitor_region, 0.30, 0.30, 1, true, 3);
  component magnetic_monitor_component = Hz;
  dft_fields cpu_magnetic_due = cpu.add_dft_fields(&magnetic_monitor_component, 1,
                                                    monitor_region, 0.30, 0.30, 1, true, 1);
  dft_fields gpu_magnetic_due = gpu.add_dft_fields(&magnetic_monitor_component, 1,
                                                    monitor_region, 0.30, 0.30, 1, true, 1);
  BackendState *const pre_monitor_state = gpu.backend_state;
  Executable *const pre_monitor_executable = gpu.executable;
  backend_set_cw_plan_corruption_for_testing(true);
  bool monitor_rollback_rejected = false;
  try {
    (void)gpu.solve_cw(tolerance, 1000, std::complex<double>(0.30, 0.0), 2);
  }
  catch (const std::exception &) {
    monitor_rollback_rejected = true;
  }
  backend_set_cw_plan_corruption_for_testing(false);
  require(monitor_rollback_rejected && !gpu.backend->is_poisoned() &&
              gpu.backend_state == pre_monitor_state && gpu.executable == pre_monitor_executable,
          "NVIDIA staged monitor failure did not preserve the live epoch");
  require(cpu.solve_cw(tolerance, 1000, std::complex<double>(0.30, 0.0), 2),
          "CPU reference solve did not converge");
  require_canonical_cw_times(cpu_custom_trace.times, cpu.dt);
  const int entry_t = gpu.t;
  gpu_custom_trace.times.clear();
  require(gpu.solve_cw(tolerance, 1000, std::complex<double>(0.30, 0.0), 2),
          "resident solve did not converge");
  require(gpu.backend_state != pre_monitor_state && gpu.executable != pre_monitor_executable,
          "NVIDIA staged monitor commit did not publish a new epoch");
  require(gpu.t == entry_t, "resident solve did not restore the entry timestep");
  const StepPlan gpu_step_plan = build_step_plan(gpu, StepProgram::solve_cw);
  const CwPlan gpu_cw_plan = build_cw_plan(gpu, gpu_step_plan);
  const CwStateLayout gpu_layout = build_cw_state_layout(gpu);
  bool has_compiled_w = false, has_w_pair = false;
  for (const ConstitutiveUpdate &update : gpu_step_plan.eh_updates)
    has_compiled_w = has_compiled_w || is_valid(update.target_w);
  for (const CwStateRow &w : gpu_layout.rows) {
    if (w.family != CwStateFamily::constitutive_w) continue;
    for (const CwStateRow &primary : gpu_layout.rows)
      if (primary.family == CwStateFamily::paired_primary && primary.chunk == w.chunk &&
          primary.traversal_component == w.traversal_component &&
          primary.storage_component == w.storage_component &&
          primary.complex_count == w.complex_count) {
        has_w_pair = true;
        break;
      }
  }
  require(has_compiled_w && has_w_pair,
          "fixture lost constitutive-W and paired-primary unpack coverage");
  if (use_magnetic) {
    bool has_ordinary_b_rhs = false, has_integrated_b_rhs = false;
    for (const CwRhsStage &stage : gpu_cw_plan.rhs_stages)
      if (stage.ft == B_stuff)
        for (size_t i = stage.source_index; i < size_t(stage.source_index) + stage.source_count;
             ++i) {
          require(i < gpu_cw_plan.rhs_sources.size(), "CW RHS stage span is out of range");
          const CwRhsSourceDescriptor &rhs = gpu_cw_plan.rhs_sources[i];
          require(rhs.source_descriptor_index < gpu.descriptors->sources.sources.size(),
                  "CW RHS source reference is out of range");
          const SourceDescriptor &source =
              gpu.descriptors->sources.sources[rhs.source_descriptor_index];
          has_ordinary_b_rhs = has_ordinary_b_rhs || !source.integrated;
          has_integrated_b_rhs = has_integrated_b_rhs || source.integrated;
        }
    require(has_ordinary_b_rhs && (!use_integrated || has_integrated_b_rhs),
            "fixture lost ordinary or integrated magnetic CW RHS coverage");
  }
  const NvidiaCwStatistics first_stats = nvidia_backend->cw_statistics_for_testing();
  require_exact_cw_trace(gpu_custom_trace.times, entry_t, gpu.dt,
                         first_stats.result.operator_applications);
  require_canonical_cw_times(gpu_custom_trace.times, gpu.dt);
  size_t expected_due_dft_launches = 0;
  for (const CwDftDescriptorRef &dft : gpu_cw_plan.final_dfts)
    if ((entry_t % dft.decimation_factor) == 0) ++expected_due_dft_launches;
  require(first_stats.valid && first_stats.result.status == CwSolveStatus::converged &&
              first_stats.result.iterations > 0 && first_stats.result.operator_applications > 0 &&
              first_stats.result.operator_applications ==
                  size_t(4 * first_stats.result.iterations + 2) &&
              first_stats.result.recursive_relative_residual <= tolerance &&
              first_stats.result.true_relative_residual <= 5.0 * tolerance,
          "resident solve statistics do not report verified convergence");
  require(first_stats.reduction_count == first_stats.scalar_device_to_host_calls &&
              first_stats.scalar_device_to_host_bytes ==
                  first_stats.scalar_device_to_host_calls * sizeof(double) &&
              first_stats.vector_host_to_device_bytes == 0 &&
              first_stats.vector_device_to_host_bytes == 0 &&
              first_stats.source_scalar_host_to_device_calls > 0 &&
              first_stats.source_scalar_host_to_device_bytes > 0 &&
              first_stats.reduction_kernel_launches == 2 * first_stats.reduction_count &&
              first_stats.final_dft_kernel_launches == expected_due_dft_launches &&
              first_stats.kernel_launches ==
                  first_stats.pack_kernel_launches + first_stats.unpack_kernel_launches +
                      first_stats.zero_kernel_launches + first_stats.rhs_source_kernel_launches +
                      first_stats.reconciliation_kernel_launches +
                      first_stats.vector_kernel_launches + first_stats.operator_kernel_launches +
                      first_stats.reduction_kernel_launches +
                      first_stats.timestep_kernel_launches +
                      first_stats.final_dft_kernel_launches &&
              first_stats.workspace_capacity_bytes > 0 && first_stats.workspace_allocations == 1,
          "resident solve statistics violate the transfer/workspace contract");

  CwSolveRequest valid_request;
  valid_request.tolerance = tolerance;
  valid_request.maxiters = 1000;
  valid_request.frequency = std::complex<double>(0.30, 0.0);
  valid_request.L = 2;
  valid_request.entry_t = gpu.t;
  valid_request.entry_time = cw_source_time(gpu.t, gpu.dt, 0.0);
  const auto expect_preflight_rejected = [&](const CwSolveRequest &request,
                                              const StepPlan &step_plan,
                                              const CwPlan &cw_plan) {
    bool rejected = false;
    Executable *unexpected = NULL;
    try {
      unexpected = nvidia_backend->preflight_cw(request, *gpu.step_plans[0], step_plan,
                                                 cw_plan, *gpu.executable, NULL,
                                                 *gpu.backend_state);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    delete unexpected;
    require(rejected, "NVIDIA CW preflight accepted malformed input");
  };
  CwSolveRequest bad_request = valid_request;
  bad_request.L = 0;
  expect_preflight_rejected(bad_request, gpu_step_plan, gpu_cw_plan);
  bad_request = valid_request;
  bad_request.maxiters = 0;
  expect_preflight_rejected(bad_request, gpu_step_plan, gpu_cw_plan);
  bad_request = valid_request;
  bad_request.frequency = std::complex<double>();
  expect_preflight_rejected(bad_request, gpu_step_plan, gpu_cw_plan);
  bad_request = valid_request;
  bad_request.entry_t += 1;
  expect_preflight_rejected(bad_request, gpu_step_plan, gpu_cw_plan);
  bad_request = valid_request;
  bad_request.entry_time =
      std::nextafter(bad_request.entry_time, std::numeric_limits<double>::infinity());
  expect_preflight_rejected(bad_request, gpu_step_plan, gpu_cw_plan);
  bad_request = valid_request;
  bad_request.eigfrequency = true;
  expect_preflight_rejected(bad_request, gpu_step_plan, gpu_cw_plan);
  const StepPlan ordinary_plan = build_step_plan(gpu, StepProgram::ordinary);
  expect_preflight_rejected(valid_request, ordinary_plan, gpu_cw_plan);
  CwPlan malformed_cw_plan = gpu_cw_plan;
  require(!malformed_cw_plan.final_dfts.empty(), "DFT rejection fixture has no descriptor");
  ++malformed_cw_plan.final_dfts[0].decimation_factor;
  malformed_cw_plan.signature = compute_cw_plan_signature(malformed_cw_plan);
  expect_preflight_rejected(valid_request, gpu_step_plan, malformed_cw_plan);

  const auto expect_layout_rejected = [&](StepPlan malformed_step_plan) {
    malformed_step_plan.cw_state_layout.signature =
        compute_cw_state_layout_signature(malformed_step_plan.cw_state_layout);
    malformed_step_plan.signature = compute_step_plan_signature(malformed_step_plan);
    CwPlan linked = gpu_cw_plan;
    linked.state_layout_signature = malformed_step_plan.cw_state_layout.signature;
    linked.step_plan_signature = malformed_step_plan.signature;
    linked.signature = compute_cw_plan_signature(linked);
    expect_preflight_rejected(valid_request, malformed_step_plan, linked);
  };
  require(gpu_layout.rows.size() > 1 && !gpu_layout.zero_arrays.empty(),
          "canonical-mutation fixture has insufficient CW rows");
  StepPlan malformed_step_plan = gpu_step_plan;
  malformed_step_plan.cw_state_layout.rows.erase(
      malformed_step_plan.cw_state_layout.rows.begin());
  expect_layout_rejected(malformed_step_plan);
  malformed_step_plan = gpu_step_plan;
  std::swap(malformed_step_plan.cw_state_layout.rows[0],
            malformed_step_plan.cw_state_layout.rows[1]);
  expect_layout_rejected(malformed_step_plan);
  malformed_step_plan = gpu_step_plan;
  malformed_step_plan.cw_state_layout.rows.push_back(
      malformed_step_plan.cw_state_layout.rows[0]);
  expect_layout_rejected(malformed_step_plan);
  malformed_step_plan = gpu_step_plan;
  malformed_step_plan.cw_state_layout.rows[0].real_array = invalid_array();
  expect_layout_rejected(malformed_step_plan);
  malformed_step_plan = gpu_step_plan;
  malformed_step_plan.cw_state_layout.vector_precision =
      malformed_step_plan.cw_state_layout.vector_precision == Precision::f32 ? Precision::f64
                                                                             : Precision::f32;
  expect_layout_rejected(malformed_step_plan);
  malformed_step_plan = gpu_step_plan;
  malformed_step_plan.cw_state_layout.real_count = std::numeric_limits<size_t>::max();
  expect_layout_rejected(malformed_step_plan);
  malformed_step_plan = gpu_step_plan;
  malformed_step_plan.cw_state_layout.zero_arrays.clear();
  expect_layout_rejected(malformed_step_plan);
  malformed_step_plan = gpu_step_plan;
  malformed_step_plan.cw_state_layout.zero_arrays.push_back(
      malformed_step_plan.cw_state_layout.zero_arrays[0]);
  expect_layout_rejected(malformed_step_plan);

  const auto expect_cw_plan_rejected = [&](CwPlan malformed) {
    malformed.signature = compute_cw_plan_signature(malformed);
    expect_preflight_rejected(valid_request, gpu_step_plan, malformed);
  };
  require(gpu_cw_plan.rhs_sources.size() > 1,
          "canonical-mutation fixture has insufficient CW RHS rows");
  malformed_cw_plan = gpu_cw_plan;
  malformed_cw_plan.rhs_sources.erase(malformed_cw_plan.rhs_sources.begin());
  expect_cw_plan_rejected(malformed_cw_plan);
  malformed_cw_plan = gpu_cw_plan;
  std::swap(malformed_cw_plan.rhs_sources[0], malformed_cw_plan.rhs_sources[1]);
  expect_cw_plan_rejected(malformed_cw_plan);
  malformed_cw_plan = gpu_cw_plan;
  malformed_cw_plan.rhs_sources.push_back(malformed_cw_plan.rhs_sources[0]);
  expect_cw_plan_rejected(malformed_cw_plan);
  malformed_cw_plan = gpu_cw_plan;
  ++malformed_cw_plan.rhs_sources[0].source_ordinal;
  expect_cw_plan_rejected(malformed_cw_plan);
  malformed_cw_plan = gpu_cw_plan;
  ++malformed_cw_plan.unpack.first_boundary.operation_index;
  expect_cw_plan_rejected(malformed_cw_plan);
  malformed_cw_plan = gpu_cw_plan;
  malformed_cw_plan.unpack.skip_w_components = !malformed_cw_plan.unpack.skip_w_components;
  expect_cw_plan_rejected(malformed_cw_plan);

  for (const CwStateRow &row : gpu_layout.rows) {
    require(row.real_array.value < gpu.storage_plan->keys.size() &&
                row.imag_array.value < gpu.storage_plan->keys.size(),
            "CW row ArrayId is outside the storage plan");
    require(gpu.storage_plan->keys[row.real_array.value].kind != int(array_kind::f_minus_p) &&
                gpu.storage_plan->keys[row.imag_array.value].kind != int(array_kind::f_minus_p),
            "CW state layout names f_minus_p storage");
  }
  for (const CwRhsSourceDescriptor &rhs : gpu_cw_plan.rhs_sources) {
    require(rhs.source_descriptor_index < gpu.descriptors->sources.sources.size(),
            "CW RHS source index is outside its descriptor owner");
    const SourceDescriptor &source =
        gpu.descriptors->sources.sources[rhs.source_descriptor_index];
    require(source.destination.value < gpu.storage_plan->keys.size() &&
                source.destination_imag.value < gpu.storage_plan->keys.size() &&
                gpu.storage_plan->keys[source.destination.value].kind !=
                    int(array_kind::f_minus_p) &&
                gpu.storage_plan->keys[source.destination_imag.value].kind !=
                    int(array_kind::f_minus_p),
            "CW RHS references f_minus_p instead of primary storage");
  }

  const bool saved_is_real = gpu.is_real;
  gpu.is_real = true;
  expect_preflight_rejected(valid_request, gpu_step_plan, gpu_cw_plan);
  gpu.is_real = saved_is_real;
  const double saved_beta = gpu.beta;
  gpu.beta = 0.125;
  expect_preflight_rejected(valid_request, gpu_step_plan, gpu_cw_plan);
  gpu.beta = saved_beta;
  const int saved_phasein_time = gpu.phasein_time;
  gpu.phasein_time = 1;
  expect_preflight_rejected(valid_request, gpu_step_plan, gpu_cw_plan);
  gpu.phasein_time = saved_phasein_time;
  const std::vector<double> saved_bfast = gpu.bfast_scaled_k;
  if (gpu.bfast_scaled_k.empty()) gpu.bfast_scaled_k.resize(3, 0.0);
  gpu.bfast_scaled_k[0] = 0.125;
  expect_preflight_rejected(valid_request, gpu_step_plan, gpu_cw_plan);
  gpu.bfast_scaled_k = saved_bfast;

  {
    nvidia_backend->preflight_initialization(*gpu.initialization_plan);
    std::unique_ptr<BackendState> sibling_state(
        nvidia_backend->create_state(*gpu.storage_plan));
    nvidia_backend->prepare_initialization(*gpu.initialization_plan, *sibling_state);
    nvidia_backend->initialize(*gpu.initialization_plan, *sibling_state);
    const MaterialClassification sibling_classification =
        nvidia_backend->classify_state(*gpu.storage_plan, *sibling_state);
    nvidia_backend->finalize_storage(*gpu.storage_plan, sibling_classification, *sibling_state);
    std::unique_ptr<Executable> sibling_executable(
        nvidia_backend->compile(build_step_plan(gpu, StepProgram::ordinary), *sibling_state));
    bool old_executable_rejected = false, sibling_executable_rejected = false;
    try {
      nvidia_backend->advance(*gpu.executable, *sibling_state, 1);
    }
    catch (const std::exception &) {
      old_executable_rejected = true;
    }
    try {
      nvidia_backend->advance(*sibling_executable, *gpu.backend_state, 1);
    }
    catch (const std::exception &) {
      sibling_executable_rejected = true;
    }
    require(old_executable_rejected && sibling_executable_rejected,
            "same-plan NVIDIA states accepted each other's compiled pointers");
  }
  test_cross_backend_rejection(gpu, *nvidia_backend);

  const CwStateLayout cpu_layout = build_cw_state_layout(cpu);
  require(cpu_layout.rows.size() == gpu_layout.rows.size(),
          "CPU/GPU packed-state row counts differ");
  long double error2 = 0.0L, reference2 = 0.0L;
  double maximum_scaled_error = 0.0;
  int maximum_chunk = -1;
  component maximum_component = NO_COMPONENT;
  CwStateFamily maximum_family = CwStateFamily::primary;
  for (const CwStateRow &row : cpu_layout.rows) {
    const StorageKey &real_key = cpu.array_catalog->key(row.real_array);
    const StorageKey &imag_key = cpu.array_catalog->key(row.imag_array);
    const ArrayId gpu_real_id = gpu.array_catalog->find(real_key);
    const ArrayId gpu_imag_id = gpu.array_catalog->find(imag_key);
    require(is_valid(gpu_real_id) && is_valid(gpu_imag_id),
            "GPU packed state is missing a CPU row");
    const ArraySpec &real_spec = gpu.array_catalog->spec(gpu_real_id);
    const ArraySpec &imag_spec = gpu.array_catalog->spec(gpu_imag_id);
    std::vector<realnum> gpu_real(real_spec.elements), gpu_imag(imag_spec.elements);
    gpu.backend->read(ArrayRef{gpu_real_id, 0, real_spec.elements}, gpu_real.data(),
                      gpu_real.size() * sizeof(realnum));
    gpu.backend->read(ArrayRef{gpu_imag_id, 0, imag_spec.elements}, gpu_imag.data(),
                      gpu_imag.size() * sizeof(realnum));
    const realnum *cpu_real = cpu.array_catalog->resolve<realnum>(row.real_array);
    const realnum *cpu_imag = cpu.array_catalog->resolve<realnum>(row.imag_array);
    for (size_t i0 = 0; i0 < row.owned_region.counts[0]; ++i0)
      for (size_t i1 = 0; i1 < row.owned_region.counts[1]; ++i1)
        for (size_t i2 = 0; i2 < row.owned_region.counts[2]; ++i2) {
          const size_t index = row.owned_region.base + i0 * row.owned_region.strides[0] +
                               i1 * row.owned_region.strides[1] +
                               i2 * row.owned_region.strides[2];
          const double dr = double(gpu_real[index]) - double(cpu_real[index]);
          const double di = double(gpu_imag[index]) - double(cpu_imag[index]);
          error2 += static_cast<long double>(dr) * dr + static_cast<long double>(di) * di;
          reference2 += static_cast<long double>(cpu_real[index]) * cpu_real[index] +
                        static_cast<long double>(cpu_imag[index]) * cpu_imag[index];
          const double scaled =
              std::sqrt(dr * dr + di * di) /
              (1.0 + std::sqrt(double(cpu_real[index]) * cpu_real[index] +
                               double(cpu_imag[index]) * cpu_imag[index]));
          if (scaled > maximum_scaled_error) {
            maximum_scaled_error = scaled;
            maximum_chunk = row.chunk;
            maximum_component = row.storage_component;
            maximum_family = row.family;
          }
        }
  }
  const double relative_l2 = std::sqrt(double(error2 / reference2));
  master_printf("nvidia_cw comparison relative_l2=%.9g max_scaled=%.9g row=(%d,%d,%u)\n",
                relative_l2, maximum_scaled_error, maximum_chunk, int(maximum_component),
                unsigned(maximum_family));
  /* HIP eager native measured 2.25367057e-6 in three consecutive ROCm 7.2
     runs on MI350X, with the same result under ROCm 7.0.  The 3e-6 bound
     allows modest toolchain variation while remaining close to the measured
     error and below the reduced-storage tolerance. */
  const double relative_l2_tolerance =
      reduced_storage ? 5e-4 : (getenv("MEEP_NVIDIA_HIP_EAGER_TOLERANCE") ? 3e-6 : 2e-6);
  require(relative_l2 <= relative_l2_tolerance,
          "resident packed state differs from CPU");
  require(maximum_scaled_error <= (reduced_storage ? 2e-3 : 1e-5),
          "resident packed state maximum scaled error is too large");

  const std::vector<std::complex<realnum> > cpu_due_values =
      dft_values(cpu, cpu_due, monitor_component);
  const std::vector<std::complex<realnum> > gpu_due_values =
      dft_values(gpu, gpu_due, monitor_component);
  const std::vector<std::complex<realnum> > cpu_not_due_values =
      dft_values(cpu, cpu_not_due, monitor_component);
  const std::vector<std::complex<realnum> > gpu_not_due_values =
      dft_values(gpu, gpu_not_due, monitor_component);
  const std::vector<std::complex<realnum> > cpu_magnetic_due_values =
      dft_values(cpu, cpu_magnetic_due, magnetic_monitor_component);
  const std::vector<std::complex<realnum> > gpu_magnetic_due_values =
      dft_values(gpu, gpu_magnetic_due, magnetic_monitor_component);
  compare_dft_values(cpu_due_values, gpu_due_values,
                     reduced_storage ? 2e-3 : 1e-5);
  compare_dft_values(cpu_magnetic_due_values, gpu_magnetic_due_values,
                     reduced_storage ? 2e-3 : 1e-5);
  bool due_nonzero = false;
  for (const std::complex<realnum> &value : cpu_due_values)
    due_nonzero = due_nonzero || value != std::complex<realnum>();
  require(due_nonzero, "due final DFT monitor was not updated");
  bool magnetic_due_nonzero = false;
  for (const std::complex<realnum> &value : cpu_magnetic_due_values)
    magnetic_due_nonzero = magnetic_due_nonzero || value != std::complex<realnum>();
  require(magnetic_due_nonzero, "due magnetic final DFT monitor was not updated");
  for (const std::complex<realnum> &value : cpu_not_due_values)
    require(value == std::complex<realnum>(), "CPU not-due final DFT monitor was updated");
  for (const std::complex<realnum> &value : gpu_not_due_values)
    require(value == std::complex<realnum>(), "NVIDIA not-due final DFT monitor was updated");

  const auto first_live_source = [](fields &owner) -> src_vol * {
    for (int chunk = 0; chunk < owner.num_chunks; ++chunk) {
      FOR_FIELD_TYPES(ft) {
        if (!owner.chunks[chunk]->sources[ft].empty())
          return &owner.chunks[chunk]->sources[ft][0];
      }
    }
    return NULL;
  };
  const auto compare_refresh_point = [&](component c, const vec &point, const char *what) {
    const std::complex<double> expected = cpu.get_field(c, point);
    const std::complex<double> actual = gpu.get_field(c, point);
    const double bound = (reduced_storage ? 2e-3 : 1e-5) * (1.0 + std::abs(expected));
    require(std::abs(actual - expected) <= bound, what);
  };

  BackendState *const source_value_state = gpu.backend_state;
  Executable *const source_value_ordinary = gpu.executable;
  Executable *const source_value_cw = gpu.backend_state->cw_executable;
  const nvidia::graph_accounting source_value_graphs_before =
      nvidia::testing::current_graph_accounting();
  src_vol *const cpu_mutated_source = first_live_source(cpu);
  src_vol *const gpu_mutated_source = first_live_source(gpu);
  require(cpu_mutated_source && gpu_mutated_source &&
              cpu_mutated_source->num_points() == gpu_mutated_source->num_points() &&
              cpu_mutated_source->num_points() > 0,
          "source-value refresh fixture has no corresponding live source");
  const std::complex<double> refreshed_amplitude =
      cpu_mutated_source->amplitude_at(0) + std::complex<double>(0.125, -0.0625);
  cpu_mutated_source->set_amplitude(0, refreshed_amplitude);
  gpu_mutated_source->set_amplitude(0, refreshed_amplitude);
  invalidate(cpu, MutationKind::source_values, "CPU CW source-value refresh test");
  invalidate(gpu, MutationKind::source_values, "NVIDIA CW source-value refresh test");
  require(cpu.solve_cw(tolerance, 1000, std::complex<double>(0.30, 0.0), 2),
          "CPU source-value refresh solve did not converge");
  require(gpu.solve_cw(tolerance, 1000, std::complex<double>(0.30, 0.0), 2),
          "NVIDIA source-value refresh solve did not converge");
  const NvidiaCwStatistics repeat_stats = nvidia_backend->cw_statistics_for_testing();
  const nvidia::graph_accounting source_value_graphs_after =
      nvidia::testing::current_graph_accounting();
  require(gpu.backend_state == source_value_state && gpu.executable == source_value_ordinary &&
              gpu.backend_state->cw_executable == source_value_cw,
          "source-value refresh did not retain state and both compatible executables");
  require(source_value_graphs_after.creates == source_value_graphs_before.creates &&
              source_value_graphs_after.begin_captures ==
                  source_value_graphs_before.begin_captures &&
              source_value_graphs_after.instantiates ==
                  source_value_graphs_before.instantiates &&
              source_value_graphs_after.graph_destroys ==
                  source_value_graphs_before.graph_destroys &&
              source_value_graphs_after.executable_destroys ==
                  source_value_graphs_before.executable_destroys,
          "source-value refresh recaptured or retired a CW graph");
  require(repeat_stats.valid && repeat_stats.workspace_allocations == 1 &&
              repeat_stats.workspace_capacity_bytes == first_stats.workspace_capacity_bytes &&
              repeat_stats.vector_host_to_device_bytes == 0 &&
              repeat_stats.vector_device_to_host_bytes == 0,
          "source-value refresh did not retain its device workspace");
  bool refreshed_descriptor = false;
  for (const SourceDescriptor &source : gpu.descriptors->sources.sources)
    for (std::complex<double> amplitude : source.complex_amplitudes)
      refreshed_descriptor = refreshed_descriptor || amplitude == refreshed_amplitude;
  require(refreshed_descriptor,
          "source-value refresh did not publish the new spatial amplitude");
  compare_refresh_point(Ez, vec(4.0, 1.5),
                        "source-value refresh changed NVIDIA Ez parity");
  compare_refresh_point(Hz, vec(4.0, 1.5),
                        "source-value refresh changed NVIDIA Hz parity");
  compare_dft_values(dft_values(cpu, cpu_due, monitor_component),
                     dft_values(gpu, gpu_due, monitor_component),
                     reduced_storage ? 2e-3 : 1e-5);

  BackendState *const source_definition_state = gpu.backend_state;
  Executable *const source_definition_ordinary = gpu.executable;
  Executable *const source_definition_cw = gpu.backend_state->cw_executable;
  const vec added_source_point(2.4, 1.3);
  const std::complex<double> added_source_amplitude(0.03125, -0.015625);
  cpu.add_point_source(Ez, cpu_e, added_source_point, added_source_amplitude);
  gpu.add_point_source(Ez, gpu_e, added_source_point, added_source_amplitude);
  require(cpu.solve_cw(tolerance, 1000, std::complex<double>(0.30, 0.0), 2),
          "CPU source-definition refresh solve did not converge");
  require(gpu.solve_cw(tolerance, 1000, std::complex<double>(0.30, 0.0), 2),
          "NVIDIA source-definition refresh solve did not converge");
  require(gpu.backend_state != source_definition_state &&
              gpu.executable != source_definition_ordinary &&
              gpu.backend_state->cw_executable != source_definition_cw,
          "source-definition refresh did not replace state and both executables");
  compare_refresh_point(Ez, vec(4.0, 1.5),
                        "source-definition refresh changed NVIDIA Ez parity");
  compare_refresh_point(Hz, vec(4.0, 1.5),
                        "source-definition refresh changed NVIDIA Hz parity");

  const std::complex<double> near = gpu.get_field(Ez, vec(4.0, 1.5));
  const std::complex<double> far = gpu.get_field(Ez, vec(6.0, 1.5));
  const double ratio = std::pow(std::abs(near) / std::abs(far), 2.0);
  require(std::isfinite(ratio), "resident CW field ratio is nonfinite");
  if (!use_conductivity && !use_integrated && !use_magnetic)
    require(ratio > 1.75 && ratio < 2.25,
            "single-source resident CW field failed the radiating-source check");

  require(!gpu_layout.rows.empty(), "poison fixture has no packed state row");
  const CwStateRow &poison_row = gpu_layout.rows[0];
  const realnum nonfinite = std::numeric_limits<realnum>::quiet_NaN();
  gpu.backend->write(ArrayRef{poison_row.real_array, poison_row.owned_region.base, 1},
                     &nonfinite, sizeof(nonfinite));
  bool dispatch_failed = false;
  try {
    (void)gpu.solve_cw(tolerance, 10, std::complex<double>(0.30, 0.0), 2);
  }
  catch (const std::exception &) {
    dispatch_failed = true;
  }
  require(dispatch_failed && gpu.backend->is_poisoned(),
          "nonfinite resident reduction did not poison the backend");
  bool poisoned_retry_rejected = false;
  try {
    (void)gpu.solve_cw(tolerance, 10, std::complex<double>(0.30, 0.0), 2);
  }
  catch (const std::exception &) {
    poisoned_retry_rejected = true;
  }
  require(poisoned_retry_rejected, "poisoned NVIDIA backend accepted another CW solve");
  master_printf("nvidia_cw: PASS realnum_bytes=%zu precision=%d ratio=%g relative_l2=%.9g "
                "max_scaled=%.9g\n",
                sizeof(realnum), int(precision_policy), ratio, relative_l2,
                maximum_scaled_error);
  return 0;
}
