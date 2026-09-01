/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <cuda_runtime_api.h>

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
  CHECK(update == graph_update_status::success || update == graph_update_status::topology_changed ||
            update == graph_update_status::unsupported,
        "graph update returned an invalid status");
  if (update == graph_update_status::success) {
    executable.launch(work);
    copy_device_to_host_async(scalar_host.data(), scalar_device, 0, sizeof(StepScalars), work);
    work.synchronize();
    CHECK(static_cast<const StepScalars *>(scalar_host.data())->entry_timestep == 9,
          "updated CUDA graph retained stale scalar parameters");
  }

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
  CHECK(accounting.creates >= 5 && accounting.begin_captures == 5 && accounting.end_captures == 4 &&
            accounting.instantiates == 3 && accounting.launches >= 1 &&
            accounting.scalar_writes == 6 && accounting.graph_destroys >= 5,
        "CUDA graph accounting is not exact");

  if (failures) {
    std::fprintf(stderr, "nvidia_graph_smoke: %d FAILURE(S)\n", failures);
    return 1;
  }
  std::puts("nvidia_graph_smoke: all checks passed");
  return 0;
}
