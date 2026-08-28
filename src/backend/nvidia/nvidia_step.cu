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

} // namespace nvidia
} // namespace meep
