/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_step.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>

namespace meep {
namespace nvidia {

namespace {

void check_cuda(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) return;
  throw runtime_error(operation, static_cast<int>(result), cudaGetErrorName(result),
                      cudaGetErrorString(result));
}

size_t checked_points(const flat_region &region) {
  size_t points = 1;
  for (int axis = 0; axis < 3; ++axis) {
    if (!region.counts[axis]) throw std::invalid_argument("NVIDIA step region is empty");
    if (region.counts[axis] > std::numeric_limits<size_t>::max() / points)
      throw std::overflow_error("NVIDIA step region size overflow");
    points *= region.counts[axis];
  }
  return points;
}

__device__ ptrdiff_t region_index(const flat_region &region, size_t linear) {
  const size_t i2 = linear % region.counts[2];
  linear /= region.counts[2];
  const size_t i1 = linear % region.counts[1];
  const size_t i0 = linear / region.counts[1];
  return ptrdiff_t(region.base) + ptrdiff_t(i0) * region.strides[0] +
         ptrdiff_t(i1) * region.strides[1] + ptrdiff_t(i2) * region.strides[2];
}

template <typename T> __global__ void curl_kernel(curl_launch update, size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  const ptrdiff_t i = region_index(update.region, linear);
  T *target = static_cast<T *>(update.target);
  const T *plus_source = static_cast<const T *>(update.plus_source);
  const T *minus_source = static_cast<const T *>(update.minus_source);
  T curl = T(0);
  if (plus_source) curl += plus_source[i + update.plus_stride] - plus_source[i];
  if (minus_source) curl += minus_source[i] - minus_source[i + update.minus_stride];
  target[i] -= T(update.dtdx) * curl;
}

template <typename T>
__global__ void constitutive_kernel(constitutive_launch update, size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  const ptrdiff_t i = region_index(update.region, linear);
  T value = static_cast<const T *>(update.primary)[i];
  if (update.diagonal) value *= static_cast<const T *>(update.diagonal)[i];
  static_cast<T *>(update.target)[i] = value;
}

template <typename T> __global__ void zero_kernel(zero_launch update, size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  static_cast<T *>(update.target)[region_index(update.region, linear)] = T(0);
}

template <typename T>
__global__ void halo_gather_kernel(const halo_gather_entry *entries, size_t first, size_t count,
                                   T *buffer) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= count) return;
  const halo_gather_entry entry = entries[first + linear];
  buffer[entry.buffer_index] = static_cast<const T *>(entry.source)[entry.source_index];
}

template <typename T>
__global__ void halo_scatter_kernel(const halo_scatter_entry *entries, size_t first, size_t count,
                                    const T *buffer) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= count) return;
  const halo_scatter_entry entry = entries[first + linear];
  const T input_real = buffer[entry.buffer_index];
  const T phase_real = T(entry.phase_real);
  if (!entry.target_imag) {
    static_cast<T *>(entry.target_real)[entry.target_real_index] = phase_real * input_real;
    return;
  }
  const T input_imag = buffer[entry.buffer_index + 1];
  const T phase_imag = T(entry.phase_imag);
  static_cast<T *>(entry.target_real)[entry.target_real_index] =
      phase_real * input_real - phase_imag * input_imag;
  static_cast<T *>(entry.target_imag)[entry.target_imag_index] =
      phase_real * input_imag + phase_imag * input_real;
}

void launch_geometry(const flat_region &region, unsigned int &blocks, unsigned int &threads) {
  const size_t points = checked_points(region);
  threads = 256;
  const size_t block_count = (points + threads - 1) / threads;
  if (block_count > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA step launch grid overflow");
  blocks = static_cast<unsigned int>(block_count);
}

template <typename T>
void launch_curl_t(const curl_launch &update, const stream &execution_stream) {
  const size_t points = checked_points(update.region);
  unsigned int blocks = 0, threads = 0;
  launch_geometry(update.region, blocks, threads);
  curl_kernel<T><<<blocks, threads, 0,
                   static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(update, points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA curl");
}

template <typename T>
void launch_constitutive_t(const constitutive_launch &update, const stream &execution_stream) {
  const size_t points = checked_points(update.region);
  unsigned int blocks = 0, threads = 0;
  launch_geometry(update.region, blocks, threads);
  constitutive_kernel<T><<<blocks, threads, 0,
                           static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(update,
                                                                                          points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA constitutive update");
}

template <typename T>
void launch_zero_t(const zero_launch &update, const stream &execution_stream) {
  const size_t points = checked_points(update.region);
  unsigned int blocks = 0, threads = 0;
  launch_geometry(update.region, blocks, threads);
  zero_kernel<T><<<blocks, threads, 0,
                   static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(update, points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA zero");
}

void linear_launch_geometry(size_t count, unsigned int &blocks, unsigned int &threads) {
  if (!count) throw std::invalid_argument("NVIDIA halo launch is empty");
  threads = 256;
  const size_t block_count = (count + threads - 1) / threads;
  if (block_count > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA halo launch grid overflow");
  blocks = static_cast<unsigned int>(block_count);
}

template <typename T>
void launch_halo_gather_t(const halo_launch &launch, const void *device_entries,
                          void *device_buffer, const stream &execution_stream) {
  unsigned int blocks = 0, threads = 0;
  linear_launch_geometry(launch.count, blocks, threads);
  halo_gather_kernel<T><<<blocks, threads, 0,
                          static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
      static_cast<const halo_gather_entry *>(device_entries), launch.first, launch.count,
      static_cast<T *>(device_buffer));
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA halo gather");
}

template <typename T>
void launch_halo_scatter_t(const halo_launch &launch, const void *device_entries,
                           const void *device_buffer, const stream &execution_stream) {
  unsigned int blocks = 0, threads = 0;
  linear_launch_geometry(launch.count, blocks, threads);
  halo_scatter_kernel<T><<<blocks, threads, 0,
                           static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
      static_cast<const halo_scatter_entry *>(device_entries), launch.first, launch.count,
      static_cast<const T *>(device_buffer));
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA halo scatter");
}

} // namespace

void launch_curl(const curl_launch &update, const stream &execution_stream) {
  if (!update.target || (!update.plus_source && !update.minus_source))
    throw std::invalid_argument("NVIDIA curl launch has incomplete operands");
  if (update.precision == scalar_precision::f32)
    launch_curl_t<float>(update, execution_stream);
  else
    launch_curl_t<double>(update, execution_stream);
}

void launch_constitutive(const constitutive_launch &update, const stream &execution_stream) {
  if (!update.target || !update.primary)
    throw std::invalid_argument("NVIDIA constitutive launch has incomplete operands");
  if (update.precision == scalar_precision::f32)
    launch_constitutive_t<float>(update, execution_stream);
  else
    launch_constitutive_t<double>(update, execution_stream);
}

void launch_zero(const zero_launch &update, const stream &execution_stream) {
  if (!update.target) throw std::invalid_argument("NVIDIA zero launch has no target");
  if (update.precision == scalar_precision::f32)
    launch_zero_t<float>(update, execution_stream);
  else
    launch_zero_t<double>(update, execution_stream);
}

void launch_halo_gather(const halo_launch &launch, const void *device_entries,
                        void *device_buffer, const stream &execution_stream) {
  if (!device_entries || !device_buffer)
    throw std::invalid_argument("NVIDIA halo gather has incomplete storage");
  if (launch.precision == scalar_precision::f32)
    launch_halo_gather_t<float>(launch, device_entries, device_buffer, execution_stream);
  else
    launch_halo_gather_t<double>(launch, device_entries, device_buffer, execution_stream);
}

void launch_halo_scatter(const halo_launch &launch, const void *device_entries,
                         const void *device_buffer, const stream &execution_stream) {
  if (!device_entries || !device_buffer)
    throw std::invalid_argument("NVIDIA halo scatter has incomplete storage");
  if (launch.precision == scalar_precision::f32)
    launch_halo_scatter_t<float>(launch, device_entries, device_buffer, execution_stream);
  else
    launch_halo_scatter_t<double>(launch, device_entries, device_buffer, execution_stream);
}

} // namespace nvidia
} // namespace meep
