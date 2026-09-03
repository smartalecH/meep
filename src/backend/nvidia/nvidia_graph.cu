/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_graph.hpp"
#include "backend/nvidia/cuda_hip_compat.hpp"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace meep {
namespace nvidia {
namespace {

std::atomic<size_t> graph_creates(0), graph_begin_captures(0), graph_end_captures(0),
    graph_instantiates(0), graph_updates(0), graph_scalar_writes(0), graph_launches(0),
    graph_destroys(0), graph_executable_destroys(0);

void check_graph(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) return;
  throw runtime_error(operation, static_cast<int>(result), cudaGetErrorName(result),
                      cudaGetErrorString(result));
}

void report_cleanup_failure(const char *operation, cudaError_t result) noexcept {
  std::fprintf(stderr, "Meep CUDA graph cleanup warning: %s failed with %s (%d): %s\n", operation,
               cudaGetErrorName(result), static_cast<int>(result), cudaGetErrorString(result));
}

struct graph_resource_cleanup_result {
  cudaError_t operation_result;
  const char *operation;
  cudaError_t restore_result;
  bool released;
};

graph_resource_cleanup_result destroy_graph_on_device(cudaGraph_t value, int device) noexcept {
  graph_resource_cleanup_result cleanup = {cudaSuccess, NULL, cudaSuccess, true};
  if (!value) return cleanup;
  int previous = -1;
  cleanup.operation_result = cudaGetDevice(&previous);
  cleanup.operation = "cudaGetDevice(graph destroy)";
  if (cleanup.operation_result != cudaSuccess) {
    cleanup.released = false;
    return cleanup;
  }
  const bool restore = previous != device;
  if (restore) {
    cleanup.operation_result = cudaSetDevice(device);
    cleanup.operation = "cudaSetDevice(graph destroy)";
    if (cleanup.operation_result != cudaSuccess) {
      cleanup.released = false;
      return cleanup;
    }
  }
  if (testing::consume_failure_for_testing(testing::failure_point::graph_destroy))
    cleanup.operation_result = cudaErrorUnknown;
  else
    cleanup.operation_result = cudaGraphDestroy(value);
  cleanup.operation = "cudaGraphDestroy";
  cleanup.released = cleanup.operation_result == cudaSuccess;
  if (cleanup.released) graph_destroys.fetch_add(1, std::memory_order_relaxed);
  if (restore)
    cleanup.restore_result =
        testing::consume_failure_for_testing(testing::failure_point::device_restore)
            ? cudaErrorUnknown
            : cudaSetDevice(previous);
  return cleanup;
}

graph_resource_cleanup_result destroy_exec_on_device(cudaGraphExec_t value, int device) noexcept {
  graph_resource_cleanup_result cleanup = {cudaSuccess, NULL, cudaSuccess, true};
  if (!value) return cleanup;
  int previous = -1;
  cleanup.operation_result = cudaGetDevice(&previous);
  cleanup.operation = "cudaGetDevice(graph executable destroy)";
  if (cleanup.operation_result != cudaSuccess) {
    cleanup.released = false;
    return cleanup;
  }
  const bool restore = previous != device;
  if (restore) {
    cleanup.operation_result = cudaSetDevice(device);
    cleanup.operation = "cudaSetDevice(graph executable destroy)";
    if (cleanup.operation_result != cudaSuccess) {
      cleanup.released = false;
      return cleanup;
    }
  }
  if (testing::consume_failure_for_testing(testing::failure_point::graph_exec_destroy))
    cleanup.operation_result = cudaErrorUnknown;
  else
    cleanup.operation_result = cudaGraphExecDestroy(value);
  cleanup.operation = "cudaGraphExecDestroy";
  cleanup.released = cleanup.operation_result == cudaSuccess;
  if (cleanup.released)
    graph_executable_destroys.fetch_add(1, std::memory_order_relaxed);
  if (restore)
    cleanup.restore_result =
        testing::consume_failure_for_testing(testing::failure_point::device_restore)
            ? cudaErrorUnknown
            : cudaSetDevice(previous);
  return cleanup;
}

__global__ void write_step_scalars_kernel(StepScalars *destination, StepScalars values) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *destination = values;
}

} // namespace

struct graph::impl {
  cudaGraph_t value;
  int device;
  bool capturing;
  cudaStream_t capture_stream;
  size_t nodes;
  std::string label;

  impl() : value(NULL), device(-1), capturing(false), capture_stream(NULL), nodes(0) {}
};

namespace {

/* The sole noexcept cleanup path for graph definitions.  It never constructs
   device_scope (whose constructor may throw), and it clears each ownership bit
   only after CUDA has accepted the corresponding release.  This lets noexcept
   move assignment leave both operands untouched when destination teardown can
   still be retried. */
struct graph_definition_cleanup_result {
  cudaError_t operation_result;
  const char *operation;
  cudaError_t restore_result;
  bool complete;
};

graph_definition_cleanup_result cleanup_graph_definition(cudaGraph_t &value, int &device,
                                                         bool &capturing,
                                                         cudaStream_t &capture_stream,
                                                         size_t &nodes,
                                                         std::string &label) noexcept {
  graph_definition_cleanup_result cleanup = {cudaSuccess, NULL, cudaSuccess, false};
  if (!value && !capturing) {
    device = -1;
    capture_stream = NULL;
    nodes = 0;
    label.clear();
    cleanup.complete = true;
    return cleanup;
  }
  int previous = -1;
  cleanup.operation_result = cudaGetDevice(&previous);
  cleanup.operation = "cudaGetDevice(graph definition cleanup)";
  if (cleanup.operation_result != cudaSuccess) return cleanup;
  const bool restore = device >= 0 && previous != device;
  if (restore) {
    cleanup.operation_result = cudaSetDevice(device);
    cleanup.operation = "cudaSetDevice(graph definition cleanup)";
    if (cleanup.operation_result != cudaSuccess) return cleanup;
  }

  if (capturing) {
    cleanup.operation = "cudaStreamEndCapture";
    if (testing::consume_failure_for_testing(testing::failure_point::graph_end_capture))
      cleanup.operation_result = cudaErrorUnknown;
    else {
      cudaGraph_t abandoned = NULL;
      cleanup.operation_result = cudaStreamEndCapture(capture_stream, &abandoned);
      if (cleanup.operation_result == cudaSuccess) {
        graph_end_captures.fetch_add(1, std::memory_order_relaxed);
        capturing = false;
        capture_stream = NULL;
        if (abandoned) {
          const graph_resource_cleanup_result destroy_result =
              destroy_graph_on_device(abandoned, device);
          if (destroy_result.released) abandoned = NULL;
          if (destroy_result.operation_result != cudaSuccess) {
            value = abandoned;
            cleanup.operation_result = destroy_result.operation_result;
            cleanup.operation = destroy_result.operation;
          }
          if (destroy_result.restore_result != cudaSuccess &&
              cleanup.restore_result == cudaSuccess)
            cleanup.restore_result = destroy_result.restore_result;
        }
      }
    }
  }
  if (cleanup.operation_result == cudaSuccess && value) {
    const graph_resource_cleanup_result destroy_result = destroy_graph_on_device(value, device);
    if (destroy_result.released) value = NULL;
    cleanup.operation_result = destroy_result.operation_result;
    cleanup.operation = destroy_result.operation;
    if (destroy_result.restore_result != cudaSuccess && cleanup.restore_result == cudaSuccess)
      cleanup.restore_result = destroy_result.restore_result;
  }
  cleanup.complete = !capturing && !value;
  if (cleanup.complete) {
    device = -1;
    nodes = 0;
    label.clear();
  }
  if (restore) {
    const cudaError_t restore_result =
        testing::consume_failure_for_testing(testing::failure_point::device_restore)
            ? cudaErrorUnknown
            : cudaSetDevice(previous);
    if (cleanup.restore_result == cudaSuccess) cleanup.restore_result = restore_result;
  }
  return cleanup;
}

void report_graph_definition_cleanup(const graph_definition_cleanup_result &result,
                                     const char *operation) noexcept {
  if (result.operation_result != cudaSuccess)
    report_cleanup_failure(result.operation ? result.operation : operation,
                           result.operation_result);
  if (result.restore_result != cudaSuccess)
    report_cleanup_failure("cudaSetDevice(restore after graph definition cleanup)",
                           result.restore_result);
}

} // namespace

graph::graph() : impl_(new impl) {}

graph::~graph() {
  if (!impl_) return;
  graph_definition_cleanup_result result =
      cleanup_graph_definition(impl_->value, impl_->device, impl_->capturing,
                               impl_->capture_stream, impl_->nodes, impl_->label);
  report_graph_definition_cleanup(result, "CUDA graph definition cleanup");
  if (!result.complete) {
    /* A one-shot injected failure must not strand a live stream capture during
       destruction.  The same raw helper performs the best-effort retry. */
    result = cleanup_graph_definition(impl_->value, impl_->device, impl_->capturing,
                                      impl_->capture_stream, impl_->nodes, impl_->label);
    report_graph_definition_cleanup(result, "CUDA graph definition cleanup retry");
  }
  delete impl_;
}

graph::graph(graph &&other) noexcept : impl_(other.impl_) { other.impl_ = NULL; }

graph &graph::operator=(graph &&other) noexcept {
  if (this == &other) return *this;
  if (impl_) {
    const graph_definition_cleanup_result result =
        cleanup_graph_definition(impl_->value, impl_->device, impl_->capturing,
                                 impl_->capture_stream, impl_->nodes, impl_->label);
    report_graph_definition_cleanup(result,
                                    "CUDA graph definition cleanup(move assignment)");
    if (!result.complete) return *this;
    delete impl_;
  }
  impl_ = other.impl_;
  other.impl_ = NULL;
  return *this;
}

void graph::create(int device, const std::string &label) {
  reset();
  if (!impl_) impl_ = new impl;
  device_scope scope(device);
  if (testing::consume_failure_for_testing(testing::failure_point::graph_create))
    throw std::runtime_error("injected CUDA graph-create failure");
  check_graph(cudaGraphCreate(&impl_->value, 0), "cudaGraphCreate");
  impl_->device = device;
  impl_->label = label;
  size_t nodes = 0;
  check_graph(cudaGraphGetNodes(impl_->value, NULL, &nodes), "cudaGraphGetNodes(create)");
  impl_->nodes = nodes;
  graph_creates.fetch_add(1, std::memory_order_relaxed);
}

void graph::begin_capture(const stream &on_stream, const std::string &label) {
  reset();
  if (!impl_) impl_ = new impl;
  if (on_stream.device() < 0) throw std::invalid_argument("graph capture received an empty stream");
  device_scope scope(on_stream.device());
  if (testing::consume_failure_for_testing(testing::failure_point::graph_begin_capture))
    throw std::runtime_error("injected CUDA graph begin-capture failure");
  check_graph(cudaStreamBeginCapture(reinterpret_cast<cudaStream_t>(on_stream.opaque_handle()),
                                     cudaStreamCaptureModeThreadLocal),
              "cudaStreamBeginCapture");
  impl_->device = on_stream.device();
  impl_->capturing = true;
  impl_->capture_stream = reinterpret_cast<cudaStream_t>(on_stream.opaque_handle());
  impl_->label = label;
  graph_begin_captures.fetch_add(1, std::memory_order_relaxed);
}

void graph::end_capture(const stream &on_stream) {
  if (!impl_ || !impl_->capturing) throw std::logic_error("CUDA graph capture is not active");
  if (on_stream.device() != impl_->device)
    throw std::invalid_argument("CUDA graph capture ended on a different device");
  if (reinterpret_cast<cudaStream_t>(on_stream.opaque_handle()) != impl_->capture_stream)
    throw std::invalid_argument("CUDA graph capture ended on a different stream");
  device_scope scope(impl_->device);
  cudaGraph_t captured = NULL;
  cudaError_t result = cudaStreamEndCapture(impl_->capture_stream, &captured);
  const bool injected =
      testing::consume_failure_for_testing(testing::failure_point::graph_end_capture);
  impl_->capturing = false;
  impl_->capture_stream = NULL;
  if (result == cudaSuccess && injected) result = cudaErrorStreamCaptureInvalidated;
  if (result != cudaSuccess) {
    if (captured) {
      const graph_resource_cleanup_result destroy_result =
          destroy_graph_on_device(captured, impl_->device);
      if (destroy_result.operation_result != cudaSuccess)
        report_cleanup_failure(destroy_result.operation, destroy_result.operation_result);
      if (destroy_result.restore_result != cudaSuccess)
        report_cleanup_failure("cudaSetDevice(restore after failed-capture graph destroy)",
                               destroy_result.restore_result);
    }
    check_graph(result, "cudaStreamEndCapture");
  }
  impl_->value = captured;
  check_graph(cudaGraphGetNodes(impl_->value, NULL, &impl_->nodes), "cudaGraphGetNodes(capture)");
  graph_end_captures.fetch_add(1, std::memory_order_relaxed);
}

void graph::reset() {
  if (!impl_) return;
  const graph_definition_cleanup_result result =
      cleanup_graph_definition(impl_->value, impl_->device, impl_->capturing,
                               impl_->capture_stream, impl_->nodes, impl_->label);
  if (result.operation_result != cudaSuccess)
    check_graph(result.operation_result,
                result.operation ? result.operation : "CUDA graph definition cleanup");
  if (result.restore_result != cudaSuccess)
    check_graph(result.restore_result,
                "cudaSetDevice(restore after graph definition cleanup)");
}

int graph::device() const { return impl_ ? impl_->device : -1; }
size_t graph::node_count() const { return impl_ ? impl_->nodes : 0; }
bool graph::capturing() const { return impl_ && impl_->capturing; }
const std::string &graph::label() const {
  static const std::string empty;
  return impl_ ? impl_->label : empty;
}
void *graph::opaque_handle() const { return impl_ ? reinterpret_cast<void *>(impl_->value) : NULL; }

struct graph_exec::impl {
  cudaGraphExec_t value;
  int device;
  size_t nodes;
  std::string label;
  impl() : value(NULL), device(-1), nodes(0) {}
};

graph_exec::graph_exec() : impl_(new impl) {}

graph_exec::~graph_exec() {
  if (!impl_) return;
  graph_resource_cleanup_result result =
      destroy_exec_on_device(impl_->value, impl_->device);
  if (result.operation_result != cudaSuccess)
    report_cleanup_failure(result.operation, result.operation_result);
  if (result.restore_result != cudaSuccess)
    report_cleanup_failure("cudaSetDevice(restore after graph executable destroy)",
                           result.restore_result);
  if (!result.released) {
    result = destroy_exec_on_device(impl_->value, impl_->device);
    if (result.operation_result != cudaSuccess)
      report_cleanup_failure(result.operation, result.operation_result);
    if (result.restore_result != cudaSuccess)
      report_cleanup_failure("cudaSetDevice(restore after graph executable destroy retry)",
                             result.restore_result);
  }
  delete impl_;
}

graph_exec::graph_exec(graph_exec &&other) noexcept : impl_(other.impl_) { other.impl_ = NULL; }

graph_exec &graph_exec::operator=(graph_exec &&other) noexcept {
  if (this == &other) return *this;
  if (impl_) {
    const graph_resource_cleanup_result result =
        destroy_exec_on_device(impl_->value, impl_->device);
    if (result.released) impl_->value = NULL;
    if (result.operation_result != cudaSuccess) {
      report_cleanup_failure(result.operation, result.operation_result);
      return *this;
    }
    if (result.restore_result != cudaSuccess)
      report_cleanup_failure("cudaSetDevice(restore after graph executable destroy)",
                             result.restore_result);
    delete impl_;
  }
  impl_ = other.impl_;
  other.impl_ = NULL;
  return *this;
}

void graph_exec::instantiate(const graph &definition) {
  if (!definition.impl_ || !definition.impl_->value || definition.impl_->capturing)
    throw std::invalid_argument("cannot instantiate an empty or capturing CUDA graph");
  reset();
  if (!impl_) impl_ = new impl;
  device_scope scope(definition.impl_->device);
  if (testing::consume_failure_for_testing(testing::failure_point::graph_instantiate))
    throw std::runtime_error("injected CUDA graph instantiate failure");
  check_graph(cudaGraphInstantiate(&impl_->value, definition.impl_->value, NULL, NULL, 0),
              "cudaGraphInstantiate");
  impl_->device = definition.impl_->device;
  impl_->nodes = definition.impl_->nodes;
  impl_->label = definition.impl_->label;
  graph_instantiates.fetch_add(1, std::memory_order_relaxed);
}

graph_update_status graph_exec::update(const graph &definition, std::string *diagnostic) {
  if (diagnostic) diagnostic->clear();
  if (!impl_ || !impl_->value || !definition.impl_ || !definition.impl_->value)
    throw std::invalid_argument("cannot update an empty CUDA graph executable");
  if (definition.impl_->device != impl_->device)
    throw std::invalid_argument("CUDA graph update crosses devices");
  device_scope scope(impl_->device);
  if (testing::consume_failure_for_testing(testing::failure_point::graph_update)) {
    if (diagnostic) *diagnostic = "injected CUDA graph update failure";
    return graph_update_status::failed;
  }
#if MEEP_GRAPH_EXEC_UPDATE_AVAILABLE
  cudaGraphNode_t error_node = NULL;
  cudaGraphExecUpdateResult update_result = cudaGraphExecUpdateError;
  const cudaError_t result =
      cudaGraphExecUpdate(impl_->value, definition.impl_->value, &error_node, &update_result);
  graph_updates.fetch_add(1, std::memory_order_relaxed);
  if (result != cudaSuccess) {
    if (diagnostic) *diagnostic = cudaGetErrorString(result);
    return graph_update_status::failed;
  }
  if (update_result != cudaGraphExecUpdateSuccess) {
    if (diagnostic) *diagnostic = "CUDA graph topology or node parameters are incompatible";
    return graph_update_status::topology_changed;
  }
  impl_->nodes = definition.impl_->nodes;
  impl_->label = definition.impl_->label;
  return graph_update_status::success;
#else
  (void)definition;
  if (diagnostic) *diagnostic = "CUDA graph executable update is unavailable";
  return graph_update_status::unsupported;
#endif
}

void graph_exec::launch(const stream &on_stream) const {
  if (!impl_ || !impl_->value) throw std::logic_error("cannot launch an empty CUDA graph");
  if (on_stream.device() != impl_->device)
    throw std::invalid_argument("CUDA graph launch crosses devices");
  device_scope scope(impl_->device);
  if (testing::consume_failure_for_testing(testing::failure_point::graph_launch))
    throw std::runtime_error("injected CUDA graph launch failure");
  check_graph(
      cudaGraphLaunch(impl_->value, reinterpret_cast<cudaStream_t>(on_stream.opaque_handle())),
      "cudaGraphLaunch");
  graph_launches.fetch_add(1, std::memory_order_relaxed);
}

void graph_exec::reset() {
  if (!impl_) return;
  if (impl_->value) {
    const graph_resource_cleanup_result result =
        destroy_exec_on_device(impl_->value, impl_->device);
    if (result.released) {
      impl_->value = NULL;
      impl_->device = -1;
      impl_->nodes = 0;
      impl_->label.clear();
    }
    if (result.operation_result != cudaSuccess)
      check_graph(result.operation_result, result.operation);
    if (result.restore_result != cudaSuccess)
      check_graph(result.restore_result,
                  "cudaSetDevice(restore after graph executable destroy)");
  }
  else {
    impl_->device = -1;
    impl_->nodes = 0;
    impl_->label.clear();
  }
}

int graph_exec::device() const { return impl_ ? impl_->device : -1; }
size_t graph_exec::node_count() const { return impl_ ? impl_->nodes : 0; }
const std::string &graph_exec::label() const {
  static const std::string empty;
  return impl_ ? impl_->label : empty;
}
void *graph_exec::opaque_handle() const {
  return impl_ ? reinterpret_cast<void *>(impl_->value) : NULL;
}

void launch_step_scalars_write(device_buffer &destination, const StepScalars &values,
                               const stream &on_stream) {
  if (destination.size() < sizeof(StepScalars) || !destination.opaque_handle())
    throw std::invalid_argument("StepScalars destination is absent or undersized");
  if (destination.device() != on_stream.device())
    throw std::invalid_argument("StepScalars write crosses devices");
  if (values.abi_version != step_scalars_abi_version || values.byte_size != sizeof(StepScalars))
    throw std::invalid_argument("StepScalars ABI header is invalid");
  device_scope scope(on_stream.device());
  if (testing::consume_failure_for_testing(testing::failure_point::graph_scalar_write))
    throw std::runtime_error("injected CUDA graph scalar-write failure");
  write_step_scalars_kernel<<<1, 1, 0,
                              reinterpret_cast<cudaStream_t>(on_stream.opaque_handle())>>>(
      static_cast<StepScalars *>(destination.opaque_handle()), values);
  check_graph(cudaGetLastError(), "write_step_scalars_kernel");
  graph_scalar_writes.fetch_add(1, std::memory_order_relaxed);
}

graph_capability query_graph_capability() {
  graph_capability result;
  result.runtime = runtime_version();
  result.driver = driver_version();
  result.capture_supported = MEEP_GRAPH_RUNTIME_CAPABILITY_ENABLED &&
                             result.runtime >= MEEP_GRAPH_CAPTURE_MIN_RUNTIME_VERSION &&
                             result.driver >= MEEP_GRAPH_CAPTURE_MIN_RUNTIME_VERSION;
  result.update_supported = MEEP_GRAPH_RUNTIME_CAPABILITY_ENABLED &&
                            result.runtime >= MEEP_GRAPH_UPDATE_MIN_RUNTIME_VERSION &&
                            result.driver >= MEEP_GRAPH_UPDATE_MIN_RUNTIME_VERSION;
  return result;
}

namespace testing {
graph_accounting current_graph_accounting() {
  return graph_accounting{graph_creates.load(std::memory_order_relaxed),
                          graph_begin_captures.load(std::memory_order_relaxed),
                          graph_end_captures.load(std::memory_order_relaxed),
                          graph_instantiates.load(std::memory_order_relaxed),
                          graph_updates.load(std::memory_order_relaxed),
                          graph_scalar_writes.load(std::memory_order_relaxed),
                          graph_launches.load(std::memory_order_relaxed),
                          graph_destroys.load(std::memory_order_relaxed),
                          graph_executable_destroys.load(std::memory_order_relaxed)};
}

void reset_graph_accounting() {
  graph_creates.store(0, std::memory_order_relaxed);
  graph_begin_captures.store(0, std::memory_order_relaxed);
  graph_end_captures.store(0, std::memory_order_relaxed);
  graph_instantiates.store(0, std::memory_order_relaxed);
  graph_updates.store(0, std::memory_order_relaxed);
  graph_scalar_writes.store(0, std::memory_order_relaxed);
  graph_launches.store(0, std::memory_order_relaxed);
  graph_destroys.store(0, std::memory_order_relaxed);
  graph_executable_destroys.store(0, std::memory_order_relaxed);
}
} // namespace testing

} // namespace nvidia
} // namespace meep
