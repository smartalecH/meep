/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "backend/nvidia/cuda_hip_compat.hpp"
#include "backend/nvidia/nvidia_graph.hpp"

using namespace meep;

static int failures = 0;

#define CHECK(condition, message)                                                                  \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      std::fprintf(stderr, "FAIL (%s:%d): %s\n", __FILE__, __LINE__, message);                     \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

static StepScalars values(int64_t timestep) {
  StepScalars result;
  std::memset(&result, 0, sizeof(result));
  result.abi_version = step_scalars_abi_version;
  result.byte_size = sizeof(result);
  result.entry_timestep = timestep;
  result.post_increment_timestep = timestep + 1;
  result.noisy_counter_time = uint64_t(timestep);
  result.source_times[0] = 1.25;
  result.source_times[1] = 1.5;
  result.source_times[2] = 1.75;
  result.batch_count = 1;
  return result;
}

int main() {
  using namespace meep::nvidia;
  if (enumerate_devices().empty()) {
    std::puts("nvidia_graph_smoke: SKIP (no CUDA device)");
    return 0;
  }
  testing::reset_graph_accounting();
  stream work;
  stream same_device_other_stream;
  device_buffer scalar_device(sizeof(StepScalars), work.device());
  pinned_buffer scalar_host(sizeof(StepScalars));

  graph empty;
  empty.create(work.device(), "empty");
  CHECK(empty.opaque_handle() && empty.node_count() == 0, "empty graph creation failed");
  empty.reset();

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count > 1) {
    int before = -1, after = -1;
    cudaGetDevice(&before);
    graph alternate;
    alternate.create(before == 0 ? 1 : 0, "alternate-device");
    cudaGetDevice(&after);
    CHECK(before == after, "graph creation failed to restore the caller's CUDA device");
  }

  graph definition;
  definition.begin_capture(work, "step-scalars");
  launch_step_scalars_write(scalar_device, values(7), work);
  bool rejected = false;
  try {
    definition.end_capture(same_device_other_stream);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected && definition.capturing(),
        "same-device wrong-stream end did not preserve active capture");
  definition.end_capture(work);
  CHECK(definition.node_count() == 1, "capture did not contain exactly one scalar node");

  graph_exec executable;
  executable.instantiate(definition);
  executable.launch(work);
  copy_device_to_host_async(scalar_host.data(), scalar_device, 0, sizeof(StepScalars), work);
  work.synchronize();
  const StepScalars &observed = *static_cast<const StepScalars *>(scalar_host.data());
  CHECK(observed.entry_timestep == 7 && observed.post_increment_timestep == 8,
        "captured scalar writer produced wrong values");

  graph replacement;
  replacement.begin_capture(work, "replacement");
  launch_step_scalars_write(scalar_device, values(9), work);
  replacement.end_capture(work);
  std::string update_diagnostic;
  const graph_update_status update = executable.update(replacement, &update_diagnostic);
#if defined(MEEP_HIP_PORTABILITY)
  if (update != graph_update_status::success)
    std::fprintf(stderr, "HIP same-topology graph update failed: %s\n", update_diagnostic.c_str());
  CHECK(update == graph_update_status::success,
        "HIP same-topology graph update did not succeed");
#else
  CHECK(update == graph_update_status::success || update == graph_update_status::topology_changed ||
            update == graph_update_status::unsupported,
        "graph update returned an invalid status");
#endif
  graph_exec replacement_executable;
  graph_exec *updated_executable = &executable;
  if (update != graph_update_status::success) {
    replacement_executable.instantiate(replacement);
    updated_executable = &replacement_executable;
  }
  updated_executable->launch(work);
  copy_device_to_host_async(scalar_host.data(), scalar_device, 0, sizeof(StepScalars), work);
  work.synchronize();
  CHECK(static_cast<const StepScalars *>(scalar_host.data())->entry_timestep == 9,
        "updated or recaptured graph retained stale scalar parameters");

  graph moved(std::move(definition));
  CHECK(!definition.opaque_handle() && moved.node_count() == 1,
        "graph move did not empty its source");
  graph_exec moved_exec(std::move(executable));
  CHECK(!executable.opaque_handle() && moved_exec.node_count() == 1,
        "graph executable move did not empty its source");

  bool injected = false;
  testing::fail_next(testing::failure_point::graph_launch);
  try {
    moved_exec.launch(work);
  }
  catch (const std::runtime_error &) {
    injected = true;
  }
  CHECK(injected, "graph launch failure injection was not observed");
  testing::clear_failure();

  testing::fail_next(testing::failure_point::graph_update);
  update_diagnostic.clear();
  CHECK(moved_exec.update(replacement, &update_diagnostic) == graph_update_status::failed &&
            !update_diagnostic.empty(),
        "graph update failure injection was not reported");
  testing::clear_failure();

  graph_exec exec_destination, exec_source;
  exec_destination.instantiate(replacement);
  exec_source.instantiate(replacement);
  void *const retained_exec_handle = exec_destination.opaque_handle();
  testing::fail_next(testing::failure_point::graph_exec_destroy);
  exec_destination = std::move(exec_source);
  CHECK(exec_destination.opaque_handle() == retained_exec_handle && exec_source.opaque_handle(),
        "failed executable destroy lost move-assignment ownership");
  testing::clear_failure();
  exec_destination = std::move(exec_source);
  CHECK(exec_destination.opaque_handle() && !exec_source.opaque_handle(),
        "executable destroy cleanup could not be retried");

  graph move_source;
  move_source.create(work.device(), "move-source");
  graph move_active_destination;
  move_active_destination.begin_capture(work, "move-active-destination");
  launch_step_scalars_write(scalar_device, values(10), work);
  move_active_destination = std::move(move_source);
  CHECK(move_active_destination.opaque_handle() && !move_active_destination.capturing() &&
            !move_source.opaque_handle(),
        "move assignment did not clean an active destination capture");

  graph retained_source;
  retained_source.create(work.device(), "retained-source");
  graph retained_destination;
  retained_destination.begin_capture(work, "retained-destination");
  launch_step_scalars_write(scalar_device, values(12), work);
  testing::fail_next(testing::failure_point::graph_end_capture);
  retained_destination = std::move(retained_source);
  CHECK(retained_destination.capturing() && retained_source.opaque_handle(),
        "failed active-destination cleanup did not preserve both graph owners");
  testing::clear_failure();
  retained_destination = std::move(retained_source);
  CHECK(retained_destination.opaque_handle() && !retained_source.opaque_handle(),
        "active-destination cleanup could not be retried");

  graph destroy_destination, destroy_source;
  destroy_destination.create(work.device(), "destroy-destination");
  destroy_source.create(work.device(), "destroy-source");
  void *const retained_handle = destroy_destination.opaque_handle();
  testing::fail_next(testing::failure_point::graph_destroy);
  destroy_destination = std::move(destroy_source);
  CHECK(destroy_destination.opaque_handle() == retained_handle && destroy_source.opaque_handle(),
        "failed graph destroy lost move-assignment ownership");
  testing::clear_failure();
  destroy_destination = std::move(destroy_source);
  CHECK(destroy_destination.opaque_handle() && !destroy_source.opaque_handle(),
        "graph destroy cleanup could not be retried");

  graph_exec cleanup_exec;
  cleanup_exec.instantiate(replacement);
  const graph_accounting exec_cleanup_before = testing::current_graph_accounting();
  void *const cleanup_exec_handle = cleanup_exec.opaque_handle();
  injected = false;
  testing::fail_next(testing::failure_point::graph_exec_destroy);
  try { cleanup_exec.reset(); }
  catch (const std::runtime_error &) { injected = true; }
  const graph_accounting exec_cleanup_failed = testing::current_graph_accounting();
  CHECK(injected && cleanup_exec.opaque_handle() == cleanup_exec_handle &&
            exec_cleanup_failed.executable_destroys ==
                exec_cleanup_before.executable_destroys,
        "failed executable reset lost ownership or reported a false destroy");
  cleanup_exec.reset();
  const graph_accounting exec_cleanup_retried = testing::current_graph_accounting();
  CHECK(!cleanup_exec.opaque_handle() &&
            exec_cleanup_retried.executable_destroys ==
                exec_cleanup_before.executable_destroys + 1,
        "executable reset retry did not release exactly one retained handle");

  graph cleanup_definition;
  cleanup_definition.create(work.device(), "cleanup-definition");
  const graph_accounting graph_cleanup_before = testing::current_graph_accounting();
  void *const cleanup_graph_handle = cleanup_definition.opaque_handle();
  injected = false;
  testing::fail_next(testing::failure_point::graph_destroy);
  try { cleanup_definition.reset(); }
  catch (const std::runtime_error &) { injected = true; }
  const graph_accounting graph_cleanup_failed = testing::current_graph_accounting();
  CHECK(injected && cleanup_definition.opaque_handle() == cleanup_graph_handle &&
            graph_cleanup_failed.graph_destroys == graph_cleanup_before.graph_destroys,
        "failed graph reset lost ownership or reported a false destroy");
  cleanup_definition.reset();
  const graph_accounting graph_cleanup_retried = testing::current_graph_accounting();
  CHECK(!cleanup_definition.opaque_handle() &&
            graph_cleanup_retried.graph_destroys == graph_cleanup_before.graph_destroys + 1,
        "graph reset retry did not release exactly one retained handle");

  if (device_count > 1) {
    const int owner_device = work.device();
    const int caller_device = owner_device == 0 ? 1 : 0;

    graph restore_definition;
    restore_definition.create(owner_device, "restore-definition");
    CHECK(cudaSetDevice(caller_device) == cudaSuccess,
          "could not select alternate caller device for graph cleanup");
    const graph_accounting restore_graph_before = testing::current_graph_accounting();
    injected = false;
    testing::fail_next(testing::failure_point::device_restore);
    try { restore_definition.reset(); }
    catch (const std::runtime_error &) { injected = true; }
    const graph_accounting restore_graph_after = testing::current_graph_accounting();
    CHECK(injected, "graph cleanup device-restoration failure was not reported");
    CHECK(!restore_definition.opaque_handle() && restore_definition.device() == -1,
          "graph restore failure retained an already-destroyed definition");
    CHECK(restore_graph_after.graph_destroys == restore_graph_before.graph_destroys + 1,
          "graph restore failure did not account exactly one successful destroy");
    restore_definition.reset();
    CHECK(testing::current_graph_accounting().graph_destroys ==
              restore_graph_after.graph_destroys,
          "graph restore failure allowed a second destroy attempt");

    graph_exec restore_executable;
    restore_executable.instantiate(replacement);
    CHECK(cudaSetDevice(caller_device) == cudaSuccess,
          "could not select alternate caller device for executable cleanup");
    const graph_accounting restore_exec_before = testing::current_graph_accounting();
    injected = false;
    testing::fail_next(testing::failure_point::device_restore);
    try { restore_executable.reset(); }
    catch (const std::runtime_error &) { injected = true; }
    const graph_accounting restore_exec_after = testing::current_graph_accounting();
    CHECK(injected, "executable cleanup device-restoration failure was not reported");
    CHECK(!restore_executable.opaque_handle() && restore_executable.device() == -1,
          "executable restore failure retained an already-destroyed handle");
    CHECK(restore_exec_after.executable_destroys ==
              restore_exec_before.executable_destroys + 1,
          "executable restore failure did not account exactly one successful destroy");
    restore_executable.reset();
    CHECK(testing::current_graph_accounting().executable_destroys ==
              restore_exec_after.executable_destroys,
          "executable restore failure allowed a second destroy attempt");

    graph restore_move_definition_destination, restore_move_definition_source;
    restore_move_definition_destination.create(owner_device, "restore-move-definition-old");
    restore_move_definition_source.create(owner_device, "restore-move-definition-new");
    void *const replacement_definition = restore_move_definition_source.opaque_handle();
    CHECK(cudaSetDevice(caller_device) == cudaSuccess,
          "could not select alternate caller device for graph move cleanup");
    const graph_accounting restore_move_graph_before = testing::current_graph_accounting();
    testing::fail_next(testing::failure_point::device_restore);
    restore_move_definition_destination = std::move(restore_move_definition_source);
    const graph_accounting restore_move_graph_after = testing::current_graph_accounting();
    CHECK(restore_move_definition_destination.opaque_handle() == replacement_definition &&
              !restore_move_definition_source.opaque_handle() &&
              restore_move_graph_after.graph_destroys ==
                  restore_move_graph_before.graph_destroys + 1,
          "graph move assignment did not transfer ownership after successful destroy");

    graph_exec restore_move_exec_destination, restore_move_exec_source;
    restore_move_exec_destination.instantiate(replacement);
    restore_move_exec_source.instantiate(replacement);
    void *const replacement_executable = restore_move_exec_source.opaque_handle();
    CHECK(cudaSetDevice(caller_device) == cudaSuccess,
          "could not select alternate caller device for executable move cleanup");
    const graph_accounting restore_move_exec_before = testing::current_graph_accounting();
    testing::fail_next(testing::failure_point::device_restore);
    restore_move_exec_destination = std::move(restore_move_exec_source);
    const graph_accounting restore_move_exec_after = testing::current_graph_accounting();
    CHECK(restore_move_exec_destination.opaque_handle() == replacement_executable &&
              !restore_move_exec_source.opaque_handle() &&
              restore_move_exec_after.executable_destroys ==
                  restore_move_exec_before.executable_destroys + 1,
          "executable move assignment did not transfer ownership after successful destroy");
    CHECK(cudaSetDevice(owner_device) == cudaSuccess,
          "could not restore graph-smoke owner device after failure injection");
  }

  const graph_accounting destructor_before = testing::current_graph_accounting();
  {
    graph destructor_definition;
    destructor_definition.create(work.device(), "destructor-definition");
    testing::fail_next(testing::failure_point::graph_destroy);
  }
  const graph_accounting graph_destructor_after = testing::current_graph_accounting();
  CHECK(graph_destructor_after.graph_destroys == destructor_before.graph_destroys + 1,
        "graph destructor did not complete best-effort cleanup after a transient failure");
  {
    graph_exec destructor_executable;
    destructor_executable.instantiate(replacement);
    testing::fail_next(testing::failure_point::graph_exec_destroy);
  }
  const graph_accounting exec_destructor_after = testing::current_graph_accounting();
  CHECK(exec_destructor_after.executable_destroys ==
            graph_destructor_after.executable_destroys + 1,
        "executable destructor did not complete best-effort cleanup after a transient failure");

  graph create_failure;
  injected = false;
  testing::fail_next(testing::failure_point::graph_create);
  try {
    create_failure.create(work.device(), "failure");
  }
  catch (const std::runtime_error &) {
    injected = true;
  }
  CHECK(!create_failure.opaque_handle(), "failed graph create published a handle");
  testing::clear_failure();

  graph begin_failure;
  injected = false;
  testing::fail_next(testing::failure_point::graph_begin_capture);
  try {
    begin_failure.begin_capture(work, "begin-failure");
  }
  catch (const std::runtime_error &) {
    injected = true;
  }
  CHECK(injected && !begin_failure.capturing(), "begin-capture failure was not atomic");
  testing::clear_failure();

  graph end_failure;
  end_failure.begin_capture(work, "end-failure");
  launch_step_scalars_write(scalar_device, values(11), work);
  injected = false;
  testing::fail_next(testing::failure_point::graph_end_capture);
  try {
    end_failure.end_capture(work);
  }
  catch (const runtime_error &) {
    injected = true;
  }
  CHECK(injected && !end_failure.capturing() && !end_failure.opaque_handle(),
        "end-capture failure published a graph or retained capture state");
  testing::clear_failure();

  graph_exec instantiate_failure;
  injected = false;
  testing::fail_next(testing::failure_point::graph_instantiate);
  try {
    instantiate_failure.instantiate(replacement);
  }
  catch (const std::runtime_error &) {
    injected = true;
  }
  CHECK(injected && !instantiate_failure.opaque_handle(),
        "instantiate failure published an executable");
  testing::clear_failure();

  device_buffer undersized(sizeof(StepScalars) - 1, work.device());
  rejected = false;
  try {
    launch_step_scalars_write(undersized, values(1), work);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected, "undersized StepScalars storage was accepted");
  StepScalars bad_abi = values(1);
  bad_abi.abi_version = 0;
  rejected = false;
  try {
    launch_step_scalars_write(scalar_device, bad_abi, work);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected, "invalid StepScalars ABI was accepted");
  testing::fail_next(testing::failure_point::graph_scalar_write);
  rejected = false;
  try {
    launch_step_scalars_write(scalar_device, values(14), work);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(rejected, "scalar-write failure injection was not observed");
  testing::clear_failure();

  /* The buffer is deliberately kept alive through capture, instantiate,
     update, and launch above.  Reuse the same stream after all graph work to
     prove cleanup did not leave it in capture mode. */
  launch_step_scalars_write(scalar_device, values(15), work);
  work.synchronize();

  const graph_capability capability = query_graph_capability();
  CHECK(capability.runtime > 0 && capability.driver > 0 && capability.capture_supported,
        "CUDA graph capability query failed");
  const graph_accounting accounting = testing::current_graph_accounting();
  CHECK(accounting.creates >= 6 && accounting.begin_captures == 5 && accounting.end_captures == 4 &&
            accounting.instantiates >= 4 && accounting.launches >= 1 &&
            accounting.scalar_writes == 6 && accounting.graph_destroys >= 5,
        "CUDA graph accounting is not exact");

  if (failures) {
    std::fprintf(stderr, "nvidia_graph_smoke: %d FAILURE(S)\n", failures);
    return 1;
  }
  std::puts("nvidia_graph_smoke: all checks passed");
  return 0;
}
