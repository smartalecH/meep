/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_peer.hpp"

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace meep {
namespace nvidia {
namespace {

static_assert(sizeof(cudaIpcMemHandle_t) == peer_ipc_handle_bytes,
              "CUDA IPC handle size changed");

void check_cuda(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) return;
  throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

__device__ __forceinline__ uint64_t load_acquire_system(const uint64_t *address) {
  uint64_t value;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
#else
  value = *static_cast<const volatile uint64_t *>(address);
  __threadfence_system();
#endif
  return value;
}

__device__ __forceinline__ void store_release_system(uint64_t *address, uint64_t value) {
  /* The full fence orders the preceding same-stream kernels' stores (already
     complete at kernel entry) ahead of the flag for system-scope observers. */
  __threadfence_system();
#if __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
#else
  *static_cast<volatile uint64_t *>(address) = value;
#endif
}

__device__ __forceinline__ uint64_t global_timer_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__global__ void peer_wait_kernel(const uint64_t *flags, size_t count, uint64_t target,
                                 uint64_t timeout_ns) {
  const uint64_t start = timeout_ns ? global_timer_ns() : 0;
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    unsigned backoff = 32;
    while (load_acquire_system(flags + i) < target) {
#if __CUDA_ARCH__ >= 700
      __nanosleep(backoff);
      if (backoff < 1024) backoff *= 2;
#endif
      if (timeout_ns && global_timer_ns() - start > timeout_ns) {
        printf("meep NVIDIA peer transport: flag %llu stuck below generation %llu "
               "(peer rank stopped?)\n",
               (unsigned long long)i, (unsigned long long)target);
        __trap();
      }
    }
  }
  __syncthreads();
  __threadfence_system();
}

__global__ void peer_signal_kernel(uint64_t *const *targets, size_t count, uint64_t value) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) store_release_system(targets[i], value);
}

template <typename T>
__global__ void peer_gather_kernel(const peer_gather_entry *entries, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const peer_gather_entry entry = entries[i];
  *static_cast<T *>(entry.target) = static_cast<const T *>(entry.source)[entry.source_index];
}

} // namespace

peer_ipc_handle export_peer_handle(const device_buffer &buffer) {
  if (!buffer.opaque_handle()) throw std::invalid_argument("cannot export an empty peer arena");
  device_scope scope(buffer.device());
  cudaIpcMemHandle_t handle;
  check_cuda(cudaIpcGetMemHandle(&handle, buffer.opaque_handle()), "cudaIpcGetMemHandle");
  peer_ipc_handle result;
  std::memcpy(result.bytes, &handle, sizeof(handle));
  return result;
}

peer_mapping::peer_mapping() : address_(NULL), device_(-1) {}

peer_mapping::~peer_mapping() {
  try {
    close();
  }
  catch (const std::exception &error) {
    std::fprintf(stderr, "meep NVIDIA peer mapping close failed: %s\n", error.what());
  }
}

peer_mapping::peer_mapping(peer_mapping &&other) noexcept
    : address_(other.address_), device_(other.device_) {
  other.address_ = NULL;
  other.device_ = -1;
}

peer_mapping &peer_mapping::operator=(peer_mapping &&other) noexcept {
  if (this == &other) return *this;
  try {
    close();
  }
  catch (const std::exception &error) {
    std::fprintf(stderr, "meep NVIDIA peer mapping close failed: %s\n", error.what());
  }
  address_ = other.address_;
  device_ = other.device_;
  other.address_ = NULL;
  other.device_ = -1;
  return *this;
}

void peer_mapping::open(const peer_ipc_handle &handle, int device) {
  close();
  device_scope scope(device);
  cudaIpcMemHandle_t ipc;
  std::memcpy(&ipc, handle.bytes, sizeof(ipc));
  void *address = NULL;
  check_cuda(cudaIpcOpenMemHandle(&address, ipc, cudaIpcMemLazyEnablePeerAccess),
             "cudaIpcOpenMemHandle");
  address_ = address;
  device_ = device;
}

void peer_mapping::close() {
  if (!address_) return;
  device_scope scope(device_);
  void *address = address_;
  address_ = NULL;
  check_cuda(cudaIpcCloseMemHandle(address), "cudaIpcCloseMemHandle");
}

std::string device_pci_bus_id(int device) {
  char bus_id[64] = {};
  check_cuda(cudaDeviceGetPCIBusId(bus_id, int(sizeof(bus_id)), device), "cudaDeviceGetPCIBusId");
  return std::string(bus_id);
}

int device_for_pci_bus_id(const std::string &bus_id) {
  int device = -1;
  if (cudaDeviceGetByPCIBusId(&device, bus_id.c_str()) != cudaSuccess) {
    (void)cudaGetLastError();
    return -1;
  }
  return device;
}

void launch_peer_wait(const void *flags, size_t count, uint64_t target, double timeout_seconds,
                      const stream &on_stream) {
  if (!count) return;
  const uint64_t timeout_ns =
      timeout_seconds > 0 ? uint64_t(timeout_seconds * 1e9) : uint64_t(0);
  const unsigned threads = count < 128 ? unsigned((count + 31) / 32 * 32) : 128u;
  peer_wait_kernel<<<1, threads, 0, static_cast<cudaStream_t>(on_stream.opaque_handle())>>>(
      static_cast<const uint64_t *>(flags), count, target, timeout_ns);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA peer wait");
}

void launch_peer_signal(const void *targets, size_t count, uint64_t value,
                        const stream &on_stream) {
  if (!count) return;
  const unsigned threads = 128;
  const unsigned blocks = unsigned((count + threads - 1) / threads);
  peer_signal_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(on_stream.opaque_handle())>>>(
      static_cast<uint64_t *const *>(targets), count, value);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA peer signal");
}

void launch_peer_gather(const void *entries, size_t count, scalar_precision precision,
                        const stream &on_stream) {
  if (!count) return;
  const unsigned threads = 256;
  const size_t blocks = (count + threads - 1) / threads;
  if (blocks > 0x7fffffffu) throw std::overflow_error("NVIDIA peer gather grid overflows");
  cudaStream_t s = static_cast<cudaStream_t>(on_stream.opaque_handle());
  const peer_gather_entry *table = static_cast<const peer_gather_entry *>(entries);
  if (precision == scalar_precision::f32)
    peer_gather_kernel<float><<<unsigned(blocks), threads, 0, s>>>(table, count);
  else
    peer_gather_kernel<double><<<unsigned(blocks), threads, 0, s>>>(table, count);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA peer gather");
}

void read_peer_flags(const device_buffer &flags, size_t first, size_t count, uint64_t *out) {
  if (!count) return;
  device_scope scope(flags.device());
  check_cuda(cudaMemcpy(out, static_cast<const uint64_t *>(flags.opaque_handle()) + first,
                        count * sizeof(uint64_t), cudaMemcpyDeviceToHost),
             "read NVIDIA peer flags");
}

void synchronize_device(int device) {
  device_scope scope(device);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (peer transport drain)");
}

} // namespace nvidia
} // namespace meep
