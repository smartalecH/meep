/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/runtime.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace meep {
namespace nvidia {

namespace {

std::atomic<size_t> device_current(0);
std::atomic<size_t> device_peak(0);
std::atomic<size_t> pinned_current(0);
std::atomic<size_t> pinned_peak(0);
std::atomic<uint64_t> device_allocation_count(0);
std::atomic<uint64_t> pinned_allocation_count(0);
std::atomic<size_t> host_to_device_calls(0), host_to_device_bytes(0);
std::atomic<size_t> device_to_host_calls(0), device_to_host_bytes(0);
std::atomic<size_t> device_to_device_calls(0), device_to_device_bytes(0);
std::atomic<size_t> material_compact_calls(0), material_compact_bytes(0);
std::atomic<size_t> material_dense_calls(0), material_dense_bytes(0);
std::atomic<size_t> material_tiled_calls(0), material_tiled_bytes(0);
std::atomic<int> injected_failure(static_cast<int>(testing::failure_point::none));
std::atomic<int> injected_followup_failure(static_cast<int>(testing::failure_point::none));

void update_peak(std::atomic<size_t> &peak, size_t value) {
  size_t observed = peak.load(std::memory_order_relaxed);
  while (observed < value &&
         !peak.compare_exchange_weak(observed, value, std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {}
}

void account_event(std::atomic<uint64_t> &counter) {
  uint64_t observed = counter.load(std::memory_order_relaxed);
  while (observed != std::numeric_limits<uint64_t>::max() &&
         !counter.compare_exchange_weak(observed, observed + 1,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {}
}

void account_allocate(std::atomic<size_t> &current, std::atomic<size_t> &peak, size_t bytes) {
  const size_t value = current.fetch_add(bytes, std::memory_order_relaxed) + bytes;
  update_peak(peak, value);
}

void account_free(std::atomic<size_t> &current, size_t bytes) {
  current.fetch_sub(bytes, std::memory_order_relaxed);
}

void check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) return;
  throw runtime_error(operation, static_cast<int>(result), cudaGetErrorName(result),
                      cudaGetErrorString(result));
}

void check_range(size_t allocation_bytes, size_t offset, size_t bytes, const char *operation) {
  if (offset > allocation_bytes || bytes > allocation_bytes - offset)
    throw std::out_of_range(std::string(operation) + ": copy/fill exceeds allocation");
}

int current_device() {
  int device = -1;
  check(cudaGetDevice(&device), "cudaGetDevice");
  return device;
}

int parse_device_id(const char *text) {
  if (!text || !*text) throw std::invalid_argument("MEEP_DEVICE_ID is empty");
  errno = 0;
  char *end = NULL;
  const long value = std::strtol(text, &end, 10);
  if (errno || end == text || *end || value < 0 || value > std::numeric_limits<int>::max())
    throw std::invalid_argument(std::string("invalid MEEP_DEVICE_ID: ") + text);
  return static_cast<int>(value);
}

std::string uuid_string(const cudaUUID_t &uuid) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (size_t i = 0; i < sizeof(uuid.bytes); ++i)
    out << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i]));
  return out.str();
}

bool consume_failure(testing::failure_point point) {
  int expected = static_cast<int>(point);
  const bool consumed = injected_failure.compare_exchange_strong(
      expected, static_cast<int>(testing::failure_point::none), std::memory_order_relaxed,
      std::memory_order_relaxed);
  if (consumed) {
    const int followup = injected_followup_failure.exchange(
        static_cast<int>(testing::failure_point::none), std::memory_order_relaxed);
    if (followup != static_cast<int>(testing::failure_point::none))
      injected_failure.store(followup, std::memory_order_relaxed);
  }
  return consumed;
}

void report_cleanup_failure(const char *operation, cudaError_t result) noexcept {
  std::fprintf(stderr, "Meep CUDA cleanup warning: %s failed with %s (%d): %s\n", operation,
               cudaGetErrorName(result), static_cast<int>(result), cudaGetErrorString(result));
}

struct device_cleanup_result {
  cudaError_t operation_result;
  const char *operation;
  cudaError_t restore_result;
  bool released;
};

device_cleanup_result release_device_allocation(void *address, size_t bytes, int device) noexcept {
  device_cleanup_result result = {cudaSuccess, NULL, cudaSuccess, true};
  if (!address) return result;
  int previous = -1;
  result.operation_result = cudaGetDevice(&previous);
  result.operation = "cudaGetDevice(device free)";
  if (result.operation_result != cudaSuccess) {
    result.released = false;
    return result;
  }
  const bool restore = previous != device;
  if (restore) {
    result.operation_result = cudaSetDevice(device);
    result.operation = "cudaSetDevice(device free)";
    if (result.operation_result != cudaSuccess) {
      result.released = false;
      return result;
    }
  }
  result.operation_result = consume_failure(testing::failure_point::device_free)
                                ? cudaErrorUnknown
                                : cudaFree(address);
  result.operation = "cudaFree";
  result.released = result.operation_result == cudaSuccess;
  if (result.released) account_free(device_current, bytes);
  if (restore) {
    result.restore_result = consume_failure(testing::failure_point::device_restore)
                                ? cudaErrorUnknown
                                : cudaSetDevice(previous);
  }
  return result;
}

void destroy_device_allocation_noexcept(void *address, size_t bytes, int device) noexcept {
  const device_cleanup_result result = release_device_allocation(address, bytes, device);
  if (result.operation_result != cudaSuccess)
    report_cleanup_failure(result.operation, result.operation_result);
  if (result.restore_result != cudaSuccess)
    report_cleanup_failure("cudaSetDevice(restore after device free)", result.restore_result);
}

cudaError_t release_pinned_allocation(void *address, size_t bytes) noexcept {
  if (!address) return cudaSuccess;
  const cudaError_t result = consume_failure(testing::failure_point::pinned_free)
                                 ? cudaErrorUnknown
                                 : cudaFreeHost(address);
  if (result == cudaSuccess) account_free(pinned_current, bytes);
  return result;
}

void destroy_pinned_allocation_noexcept(void *address, size_t bytes) noexcept {
  const cudaError_t result = release_pinned_allocation(address, bytes);
  if (result != cudaSuccess) report_cleanup_failure("cudaFreeHost", result);
}

void destroy_stream(cudaStream_t value, int device) noexcept {
  if (!value) return;
  int previous = -1;
  const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
  if (device >= 0) cudaSetDevice(device);
  cudaStreamDestroy(value);
  if (have_previous && previous != device) cudaSetDevice(previous);
}

void destroy_event(cudaEvent_t value, int device) noexcept {
  if (!value) return;
  int previous = -1;
  const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
  if (device >= 0) cudaSetDevice(device);
  cudaEventDestroy(value);
  if (have_previous && previous != device) cudaSetDevice(previous);
}

} // namespace

runtime_error::runtime_error(const std::string &operation, int code, const std::string &name,
                             const std::string &detail)
    : std::runtime_error(operation + " failed with " + name + " (" + std::to_string(code) +
                         "): " + detail),
      code_(code), operation_(operation), name_(name) {}

std::vector<device_properties> enumerate_devices() {
  int count = 0;
  check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  std::vector<device_properties> devices;
  devices.reserve(static_cast<size_t>(count));
  for (int id = 0; id < count; ++id) devices.push_back(properties_for_device(id));
  return devices;
}

device_properties properties_for_device(int device) {
  cudaDeviceProp prop;
  check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  device_properties result;
  result.id = device;
  result.name = prop.name;
  result.uuid = uuid_string(prop.uuid);
  result.compute_major = prop.major;
  result.compute_minor = prop.minor;
  result.total_memory = prop.totalGlobalMem;
  result.multiprocessors = prop.multiProcessorCount;
  result.max_threads_per_block = prop.maxThreadsPerBlock;
  result.unified_addressing = prop.unifiedAddressing != 0;
  result.managed_memory = prop.managedMemory != 0;
  return result;
}

size_t free_memory_for_device(int device) {
  device_scope scope(device);
  size_t free_bytes = 0, total_bytes = 0;
  check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  return free_bytes;
}

int runtime_version() {
  int version = 0;
  check(cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion");
  return version;
}

int driver_version() {
  int version = 0;
  check(cudaDriverGetVersion(&version), "cudaDriverGetVersion");
  return version;
}

bool devices_can_access_peer(int source, int destination) {
  int can_access = 0;
  check(cudaDeviceCanAccessPeer(&can_access, source, destination), "cudaDeviceCanAccessPeer");
  return can_access != 0;
}

const char *selection_source_name(selection_source source) {
  switch (source) {
    case selection_source::explicit_option: return "explicit";
    case selection_source::environment: return "MEEP_DEVICE_ID";
    case selection_source::node_local_rank: return "node-local-rank";
  }
  return "unknown";
}

device_selection select_device(int explicit_device, int node_local_rank, int node_local_size,
                               bool allow_sharing) {
  if (explicit_device < -1)
    throw std::invalid_argument("explicit NVIDIA device must be -1 or non-negative");
  const int count = static_cast<int>(enumerate_devices().size());
  if (count == 0) throw std::runtime_error("no visible NVIDIA devices");
  if (node_local_rank < 0 || node_local_size < 1 || node_local_rank >= node_local_size)
    throw std::invalid_argument("invalid node-local rank/size");
  if (node_local_size > count && !allow_sharing)
    throw std::runtime_error("more node-local ranks than visible NVIDIA devices");

  int selected = -1;
  selection_source source = selection_source::node_local_rank;
  if (explicit_device >= 0) {
    selected = explicit_device;
    source = selection_source::explicit_option;
  }
  else if (const char *environment_device = std::getenv("MEEP_DEVICE_ID")) {
    selected = parse_device_id(environment_device);
    source = selection_source::environment;
  }
  else {
    selected = node_local_rank % count;
  }
  if (selected < 0 || selected >= count)
    throw std::out_of_range("selected NVIDIA device is outside CUDA_VISIBLE_DEVICES");

  device_selection result;
  result.device = selected;
  result.visible_device_count = count;
  result.node_local_rank = node_local_rank;
  result.node_local_size = node_local_size;
  result.sharing = node_local_size > count;
  result.collective_collision_check_required =
      node_local_size > 1 && source != selection_source::node_local_rank;
  result.source = source;
  return result;
}

device_scope::device_scope(int device) : previous_(-1), device_(device), restore_(false) {
  check(cudaGetDevice(&previous_), "cudaGetDevice");
  restore_ = previous_ != device_;
  if (restore_) check(cudaSetDevice(device_), "cudaSetDevice");
}

device_scope::~device_scope() {
  if (restore_) cudaSetDevice(previous_);
}

struct stream::impl {
  cudaStream_t value;
  int device;
};

stream::stream() : impl_(new impl) {
  try {
    impl_->device = current_device();
    check(cudaStreamCreateWithFlags(&impl_->value, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags");
  }
  catch (...) {
    delete impl_;
    impl_ = NULL;
    throw;
  }
}

stream::~stream() {
  if (impl_) destroy_stream(impl_->value, impl_->device);
  delete impl_;
}

stream::stream(stream &&other) noexcept : impl_(other.impl_) { other.impl_ = NULL; }

stream &stream::operator=(stream &&other) noexcept {
  if (this == &other) return *this;
  if (impl_) destroy_stream(impl_->value, impl_->device);
  delete impl_;
  impl_ = other.impl_;
  other.impl_ = NULL;
  return *this;
}

void stream::synchronize() const {
  if (!impl_) throw std::logic_error("synchronize on moved-from CUDA stream");
  device_scope scope(impl_->device);
  check(cudaStreamSynchronize(impl_->value), "cudaStreamSynchronize");
}

int stream::device() const { return impl_ ? impl_->device : -1; }

void *stream::opaque_handle() const {
  return impl_ ? reinterpret_cast<void *>(impl_->value) : NULL;
}

struct event::impl {
  cudaEvent_t value;
  int device;
};

event::event() : impl_(new impl) {
  try {
    impl_->device = current_device();
    check(cudaEventCreateWithFlags(&impl_->value, cudaEventDisableTiming),
          "cudaEventCreateWithFlags");
  }
  catch (...) {
    delete impl_;
    impl_ = NULL;
    throw;
  }
}

event::~event() {
  if (impl_) destroy_event(impl_->value, impl_->device);
  delete impl_;
}

event::event(event &&other) noexcept : impl_(other.impl_) { other.impl_ = NULL; }

event &event::operator=(event &&other) noexcept {
  if (this == &other) return *this;
  if (impl_) destroy_event(impl_->value, impl_->device);
  delete impl_;
  impl_ = other.impl_;
  other.impl_ = NULL;
  return *this;
}

void event::record(const stream &on_stream) {
  if (!impl_) throw std::logic_error("record on moved-from CUDA event");
  if (impl_->device != on_stream.device())
    throw std::invalid_argument("CUDA event and stream belong to different devices");
  device_scope scope(impl_->device);
  check(cudaEventRecord(impl_->value,
                        reinterpret_cast<cudaStream_t>(on_stream.opaque_handle())),
        "cudaEventRecord");
}

void event::wait(const stream &on_stream) const {
  if (!impl_) throw std::logic_error("wait on moved-from CUDA event");
  if (impl_->device != on_stream.device())
    throw std::invalid_argument("CUDA event and stream belong to different devices");
  device_scope scope(impl_->device);
  check(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(on_stream.opaque_handle()), impl_->value,
                            0),
        "cudaStreamWaitEvent");
}

void event::synchronize() const {
  if (!impl_) throw std::logic_error("synchronize on moved-from CUDA event");
  device_scope scope(impl_->device);
  check(cudaEventSynchronize(impl_->value), "cudaEventSynchronize");
}

bool event::ready() const {
  if (!impl_) throw std::logic_error("query on moved-from CUDA event");
  device_scope scope(impl_->device);
  const cudaError_t result = cudaEventQuery(impl_->value);
  if (result == cudaSuccess) return true;
  if (result == cudaErrorNotReady) return false;
  check(result, "cudaEventQuery");
  return false;
}

int event::device() const { return impl_ ? impl_->device : -1; }

void *event::opaque_handle() const {
  return impl_ ? reinterpret_cast<void *>(impl_->value) : NULL;
}

struct device_buffer::impl {
  void *address;
  size_t bytes;
  int device;
};

device_buffer::device_buffer() : impl_(new impl{NULL, 0, -1}) {}

device_buffer::device_buffer(size_t bytes, int device) : impl_(new impl{NULL, 0, -1}) {
  try {
    allocate(bytes, device);
  }
  catch (...) {
    delete impl_;
    impl_ = NULL;
    throw;
  }
}

device_buffer::~device_buffer() {
  if (impl_)
    destroy_device_allocation_noexcept(impl_->address, impl_->bytes, impl_->device);
  delete impl_;
}

device_buffer::device_buffer(device_buffer &&other) noexcept : impl_(other.impl_) {
  other.impl_ = NULL;
}

device_buffer &device_buffer::operator=(device_buffer &&other) noexcept {
  if (this == &other) return *this;
  if (impl_ && impl_->address) {
    const device_cleanup_result result =
        release_device_allocation(impl_->address, impl_->bytes, impl_->device);
    if (result.operation_result != cudaSuccess) {
      report_cleanup_failure(result.operation, result.operation_result);
      if (result.restore_result != cudaSuccess)
        report_cleanup_failure("cudaSetDevice(restore after device free)", result.restore_result);
      return *this;
    }
    if (result.restore_result != cudaSuccess)
      report_cleanup_failure("cudaSetDevice(restore after device free)", result.restore_result);
  }
  delete impl_;
  impl_ = other.impl_;
  other.impl_ = NULL;
  return *this;
}

void device_buffer::allocate(size_t bytes, int device) {
  if (!impl_) impl_ = new impl{NULL, 0, -1};
  reset();
  if (!bytes) return;
  const int allocation_device = device >= 0 ? device : current_device();
  device_scope scope(allocation_device);
  void *address = NULL;
  const cudaError_t result = consume_failure(testing::failure_point::device_allocate)
                                 ? cudaErrorMemoryAllocation
                                 : cudaMalloc(&address, bytes);
  check(result, "cudaMalloc");
  impl_->address = address;
  impl_->bytes = bytes;
  impl_->device = allocation_device;
  account_allocate(device_current, device_peak, bytes);
  account_event(device_allocation_count);
}

void device_buffer::reset() {
  if (!impl_ || !impl_->address) return;
  const device_cleanup_result result =
      release_device_allocation(impl_->address, impl_->bytes, impl_->device);
  if (result.released) {
    impl_->address = NULL;
    impl_->bytes = 0;
    impl_->device = -1;
  }
  if (result.operation_result != cudaSuccess)
    check(result.operation_result, result.operation);
  if (result.restore_result != cudaSuccess)
    check(result.restore_result, "cudaSetDevice(restore after device free)");
}

size_t device_buffer::size() const { return impl_ ? impl_->bytes : 0; }
int device_buffer::device() const { return impl_ ? impl_->device : -1; }
void *device_buffer::opaque_handle() const { return impl_ ? impl_->address : NULL; }

struct pinned_buffer::impl {
  void *address;
  size_t bytes;
};

pinned_buffer::pinned_buffer() : impl_(new impl{NULL, 0}) {}

pinned_buffer::pinned_buffer(size_t bytes) : impl_(new impl{NULL, 0}) {
  try {
    allocate(bytes);
  }
  catch (...) {
    delete impl_;
    impl_ = NULL;
    throw;
  }
}

pinned_buffer::~pinned_buffer() {
  if (impl_) destroy_pinned_allocation_noexcept(impl_->address, impl_->bytes);
  delete impl_;
}

pinned_buffer::pinned_buffer(pinned_buffer &&other) noexcept : impl_(other.impl_) {
  other.impl_ = NULL;
}

pinned_buffer &pinned_buffer::operator=(pinned_buffer &&other) noexcept {
  if (this == &other) return *this;
  if (impl_ && impl_->address) {
    const cudaError_t result = release_pinned_allocation(impl_->address, impl_->bytes);
    if (result != cudaSuccess) {
      report_cleanup_failure("cudaFreeHost", result);
      return *this;
    }
  }
  delete impl_;
  impl_ = other.impl_;
  other.impl_ = NULL;
  return *this;
}

void pinned_buffer::allocate(size_t bytes) {
  if (!impl_) impl_ = new impl{NULL, 0};
  reset();
  if (!bytes) return;
  void *address = NULL;
  const cudaError_t result = consume_failure(testing::failure_point::pinned_allocate)
                                 ? cudaErrorMemoryAllocation
                                 : cudaHostAlloc(&address, bytes, cudaHostAllocPortable);
  check(result, "cudaHostAlloc");
  impl_->address = address;
  impl_->bytes = bytes;
  account_allocate(pinned_current, pinned_peak, bytes);
  account_event(pinned_allocation_count);
}

void pinned_buffer::reset() {
  if (!impl_ || !impl_->address) return;
  const cudaError_t result = release_pinned_allocation(impl_->address, impl_->bytes);
  if (result != cudaSuccess) check(result, "cudaFreeHost");
  impl_->address = NULL;
  impl_->bytes = 0;
}

size_t pinned_buffer::size() const { return impl_ ? impl_->bytes : 0; }
void *pinned_buffer::data() { return impl_ ? impl_->address : NULL; }
const void *pinned_buffer::data() const { return impl_ ? impl_->address : NULL; }

void copy_host_to_device_async(device_buffer &destination, size_t destination_offset,
                               const void *source, size_t bytes, const stream &on_stream,
                               host_to_device_copy_kind kind) {
  if (kind != host_to_device_copy_kind::general &&
      kind != host_to_device_copy_kind::material_compact_input &&
      kind != host_to_device_copy_kind::material_dense_output &&
      kind != host_to_device_copy_kind::material_tiled_output)
    throw std::invalid_argument("invalid NVIDIA host-to-device copy kind");
  check_range(destination.size(), destination_offset, bytes, "copy_host_to_device_async");
  if (!bytes) return;
  if (destination.device() != on_stream.device())
    throw std::invalid_argument("destination buffer and CUDA stream belong to different devices");
  device_scope scope(destination.device());
  char *destination_address = static_cast<char *>(destination.opaque_handle()) + destination_offset;
  const cudaError_t result = consume_failure(testing::failure_point::host_to_device_copy)
                                 ? cudaErrorUnknown
                                 : cudaMemcpyAsync(destination_address, source, bytes,
                                                   cudaMemcpyHostToDevice,
                                                   reinterpret_cast<cudaStream_t>(
                                                       on_stream.opaque_handle()));
  check(result,
        "cudaMemcpyAsync(host-to-device)");
  host_to_device_calls.fetch_add(1, std::memory_order_relaxed);
  host_to_device_bytes.fetch_add(bytes, std::memory_order_relaxed);
  switch (kind) {
    case host_to_device_copy_kind::general: break;
    case host_to_device_copy_kind::material_compact_input:
      material_compact_calls.fetch_add(1, std::memory_order_relaxed);
      material_compact_bytes.fetch_add(bytes, std::memory_order_relaxed);
      break;
    case host_to_device_copy_kind::material_dense_output:
      material_dense_calls.fetch_add(1, std::memory_order_relaxed);
      material_dense_bytes.fetch_add(bytes, std::memory_order_relaxed);
      break;
    case host_to_device_copy_kind::material_tiled_output:
      material_tiled_calls.fetch_add(1, std::memory_order_relaxed);
      material_tiled_bytes.fetch_add(bytes, std::memory_order_relaxed);
      break;
    default: break;
  }
}

void copy_device_to_host_async(void *destination, const device_buffer &source,
                               size_t source_offset, size_t bytes, const stream &on_stream) {
  check_range(source.size(), source_offset, bytes, "copy_device_to_host_async");
  if (!bytes) return;
  if (source.device() != on_stream.device())
    throw std::invalid_argument("source buffer and CUDA stream belong to different devices");
  device_scope scope(source.device());
  const char *source_address = static_cast<const char *>(source.opaque_handle()) + source_offset;
  const cudaError_t result = consume_failure(testing::failure_point::device_to_host_copy)
                                 ? cudaErrorUnknown
                                 : cudaMemcpyAsync(destination, source_address, bytes,
                                                   cudaMemcpyDeviceToHost,
                                                   reinterpret_cast<cudaStream_t>(
                                                       on_stream.opaque_handle()));
  check(result,
        "cudaMemcpyAsync(device-to-host)");
  device_to_host_calls.fetch_add(1, std::memory_order_relaxed);
  device_to_host_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void copy_device_to_device_async(device_buffer &destination, size_t destination_offset,
                                 const device_buffer &source, size_t source_offset, size_t bytes,
                                 const stream &on_stream) {
  check_range(destination.size(), destination_offset, bytes, "copy_device_to_device_async(dst)");
  check_range(source.size(), source_offset, bytes, "copy_device_to_device_async(src)");
  if (!bytes) return;
  if (destination.device() != source.device() || destination.device() != on_stream.device())
    throw std::invalid_argument("device-to-device copy requires one device and matching stream");
  device_scope scope(destination.device());
  char *destination_address = static_cast<char *>(destination.opaque_handle()) + destination_offset;
  const char *source_address = static_cast<const char *>(source.opaque_handle()) + source_offset;
  check(cudaMemcpyAsync(destination_address, source_address, bytes, cudaMemcpyDeviceToDevice,
                        reinterpret_cast<cudaStream_t>(on_stream.opaque_handle())),
        "cudaMemcpyAsync(device-to-device)");
  device_to_device_calls.fetch_add(1, std::memory_order_relaxed);
  device_to_device_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void fill_byte_async(device_buffer &destination, size_t destination_offset, int value, size_t bytes,
                     const stream &on_stream) {
  check_range(destination.size(), destination_offset, bytes, "fill_byte_async");
  if (!bytes) return;
  if (destination.device() != on_stream.device())
    throw std::invalid_argument("destination buffer and CUDA stream belong to different devices");
  device_scope scope(destination.device());
  char *destination_address = static_cast<char *>(destination.opaque_handle()) + destination_offset;
  check(cudaMemsetAsync(destination_address, value, bytes,
                        reinterpret_cast<cudaStream_t>(on_stream.opaque_handle())),
        "cudaMemsetAsync");
}

memory_accounting current_memory_accounting() {
  memory_accounting result;
  result.device_bytes_current = device_current.load(std::memory_order_relaxed);
  result.device_bytes_peak = device_peak.load(std::memory_order_relaxed);
  result.pinned_bytes_current = pinned_current.load(std::memory_order_relaxed);
  result.pinned_bytes_peak = pinned_peak.load(std::memory_order_relaxed);
  result.device_allocation_count = device_allocation_count.load(std::memory_order_relaxed);
  result.pinned_allocation_count = pinned_allocation_count.load(std::memory_order_relaxed);
  return result;
}

namespace testing {

void fail_next(failure_point point) {
  injected_followup_failure.store(static_cast<int>(failure_point::none),
                                  std::memory_order_relaxed);
  injected_failure.store(static_cast<int>(point), std::memory_order_relaxed);
}

void fail_next_then(failure_point point, failure_point followup) {
  injected_followup_failure.store(static_cast<int>(followup), std::memory_order_relaxed);
  injected_failure.store(static_cast<int>(point), std::memory_order_relaxed);
}

void clear_failure() {
  injected_failure.store(static_cast<int>(failure_point::none), std::memory_order_relaxed);
  injected_followup_failure.store(static_cast<int>(failure_point::none),
                                  std::memory_order_relaxed);
}

bool consume_failure_for_testing(failure_point point) { return consume_failure(point); }

bool opaque_pointer_is_device_for_testing(const void *pointer) {
  cudaPointerAttributes attributes;
  const cudaError_t result = cudaPointerGetAttributes(&attributes, pointer);
  if (result != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
#if CUDART_VERSION >= 10000
  return attributes.type == cudaMemoryTypeDevice;
#else
  return attributes.memoryType == cudaMemoryTypeDevice;
#endif
}

bool copy_opaque_device_to_device_for_testing(void *destination, const void *source, size_t bytes,
                                              int device) {
  try {
    device_scope scope(device);
    if (cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess)
      return false;
    return cudaDeviceSynchronize() == cudaSuccess;
  }
  catch (...) { return false; }
}

transfer_accounting current_transfer_accounting() {
  transfer_accounting result;
  result.host_to_device_calls = host_to_device_calls.load(std::memory_order_relaxed);
  result.host_to_device_bytes = host_to_device_bytes.load(std::memory_order_relaxed);
  result.device_to_host_calls = device_to_host_calls.load(std::memory_order_relaxed);
  result.device_to_host_bytes = device_to_host_bytes.load(std::memory_order_relaxed);
  result.device_to_device_calls = device_to_device_calls.load(std::memory_order_relaxed);
  result.device_to_device_bytes = device_to_device_bytes.load(std::memory_order_relaxed);
  return result;
}

void reset_transfer_accounting() {
  host_to_device_calls.store(0, std::memory_order_relaxed);
  host_to_device_bytes.store(0, std::memory_order_relaxed);
  device_to_host_calls.store(0, std::memory_order_relaxed);
  device_to_host_bytes.store(0, std::memory_order_relaxed);
  device_to_device_calls.store(0, std::memory_order_relaxed);
  device_to_device_bytes.store(0, std::memory_order_relaxed);
}

material_transfer_accounting current_material_transfer_accounting() {
  material_transfer_accounting result;
  result.compact_calls = material_compact_calls.load(std::memory_order_relaxed);
  result.compact_bytes = material_compact_bytes.load(std::memory_order_relaxed);
  result.dense_output_calls = material_dense_calls.load(std::memory_order_relaxed);
  result.dense_output_bytes = material_dense_bytes.load(std::memory_order_relaxed);
  result.tiled_output_calls = material_tiled_calls.load(std::memory_order_relaxed);
  result.tiled_output_bytes = material_tiled_bytes.load(std::memory_order_relaxed);
  return result;
}

void reset_material_transfer_accounting() {
  material_compact_calls.store(0, std::memory_order_relaxed);
  material_compact_bytes.store(0, std::memory_order_relaxed);
  material_dense_calls.store(0, std::memory_order_relaxed);
  material_dense_bytes.store(0, std::memory_order_relaxed);
  material_tiled_calls.store(0, std::memory_order_relaxed);
  material_tiled_bytes.store(0, std::memory_order_relaxed);
}

} // namespace testing

} // namespace nvidia
} // namespace meep
