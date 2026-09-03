/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_magnetic.hpp"
#include "backend/nvidia/cuda_hip_compat.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace meep {
namespace nvidia {
namespace {

void check_cuda(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) return;
  throw runtime_error(operation, static_cast<int>(result), cudaGetErrorName(result),
                      cudaGetErrorString(result));
}

template <typename T> __global__ void copy_kernel(T *destination, const T *source, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) destination[i] = source[i];
}

template <typename T> __global__ void average_kernel(T *live, const T *backup, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) live[i] = 0.5 * (live[i] + backup[i]);
}

template <typename T>
void launch_copy(void *destination, const void *source, size_t elements, const stream &stream,
                 const char *what) {
  if (!destination || !source) throw std::invalid_argument(std::string(what) + " has a null array");
  if (!elements) throw std::invalid_argument(std::string(what) + " has no elements");
  if (destination == source)
    throw std::invalid_argument(std::string(what) + " aliases live and backup storage");
  const size_t threads = 256;
  const size_t blocks = 1 + (elements - 1) / threads;
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error(std::string(what) + " launch grid overflows");
  copy_kernel<T><<<unsigned(blocks), unsigned(threads), 0,
                   static_cast<cudaStream_t>(stream.opaque_handle())>>>(
      static_cast<T *>(destination), static_cast<const T *>(source), elements);
  check_cuda(cudaPeekAtLastError(), what);
}

template <typename T>
void launch_average_typed(const magnetic_state_launch &launch, const stream &stream) {
  if (!launch.live || !launch.backup)
    throw std::invalid_argument("NVIDIA magnetic average has a null array");
  if (!launch.elements) throw std::invalid_argument("NVIDIA magnetic average has no elements");
  if (launch.live == launch.backup)
    throw std::invalid_argument("NVIDIA magnetic average aliases live and backup storage");
  const size_t threads = 256;
  const size_t blocks = 1 + (launch.elements - 1) / threads;
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA magnetic average launch grid overflows");
  average_kernel<T><<<unsigned(blocks), unsigned(threads), 0,
                      static_cast<cudaStream_t>(stream.opaque_handle())>>>(
      static_cast<T *>(launch.live), static_cast<const T *>(launch.backup), launch.elements);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA magnetic average kernel");
}

} // namespace

void launch_magnetic_backup(const magnetic_state_launch &launch, const stream &stream) {
  switch (launch.precision) {
    case scalar_precision::f32:
      launch_copy<float>(launch.backup, launch.live, launch.elements, stream,
                         "launching NVIDIA magnetic backup kernel");
      break;
    case scalar_precision::f64:
      launch_copy<double>(launch.backup, launch.live, launch.elements, stream,
                          "launching NVIDIA magnetic backup kernel");
      break;
    default: throw std::invalid_argument("NVIDIA magnetic backup precision is invalid");
  }
}

void launch_magnetic_restore(const magnetic_state_launch &launch, const stream &stream) {
  switch (launch.precision) {
    case scalar_precision::f32:
      launch_copy<float>(launch.live, launch.backup, launch.elements, stream,
                         "launching NVIDIA magnetic restore kernel");
      break;
    case scalar_precision::f64:
      launch_copy<double>(launch.live, launch.backup, launch.elements, stream,
                          "launching NVIDIA magnetic restore kernel");
      break;
    default: throw std::invalid_argument("NVIDIA magnetic restore precision is invalid");
  }
}

void launch_magnetic_average(const magnetic_state_launch &launch, const stream &stream) {
  switch (launch.precision) {
    case scalar_precision::f32: launch_average_typed<float>(launch, stream); break;
    case scalar_precision::f64: launch_average_typed<double>(launch, stream); break;
    default: throw std::invalid_argument("NVIDIA magnetic average precision is invalid");
  }
}

} // namespace nvidia
} // namespace meep
