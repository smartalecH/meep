/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_sources.hpp"
#include "backend/nvidia/cuda_hip_compat.hpp"

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

template <typename T>
__device__ void apply_source_point(const source_batch_launch &source, const source_point &point,
                                   const source_scalar &scalar) {
  const double scalar_real = source.integrated ? scalar.dipole_real : scalar.current_real;
  const double scalar_imag = source.integrated ? scalar.dipole_imag : scalar.current_imag;
  double value_real = point.amplitude_real * scalar_real - point.amplitude_imag * scalar_imag;
  double value_imag = point.amplitude_real * scalar_imag + point.amplitude_imag * scalar_real;
  /* Preserve the legacy association. Ordinary sources multiply the complex
     current by dt and condinv; integrated sources subtract the dipole directly
     from the prepared D/B-minus-polarization array. */
  if (!source.integrated) {
    value_real *= source.dt;
    value_imag *= source.dt;
  }
  if (!source.integrated && source.conductivity_inverse) {
    const double condinv =
        double(static_cast<const T *>(source.conductivity_inverse)[point.index]);
    value_real *= condinv;
    value_imag *= condinv;
  }
  static_cast<T *>(source.target_real)[point.index] -= T(value_real);
  if (source.target_imag)
    static_cast<T *>(source.target_imag)[point.index] -= T(value_imag);
}

template <typename T>
__global__ void source_batch_kernel(source_batch_launch source, const source_scalar *scalars) {
  const source_scalar scalar = scalars[source.scalar_slot];
  for (size_t point = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       point < source.point_count; point += size_t(blockDim.x) * gridDim.x)
    apply_source_point<T>(source, source.points[point], scalar);
}

template <typename T>
__global__ void ordered_source_batch_kernel(source_batch_launch source,
                                            const source_scalar *scalars) {
  if (blockIdx.x || threadIdx.x) return;
  const source_scalar scalar = scalars[source.scalar_slot];
  for (size_t point = 0; point < source.point_count; ++point)
    apply_source_point<T>(source, source.points[point], scalar);
}

template <typename T>
__global__ void array_copy_kernel(array_copy_launch copy) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < copy.elements)
    static_cast<T *>(copy.target)[index] = static_cast<const T *>(copy.source)[index];
}

template <typename T>
void launch_source_batch_t(const source_batch_launch &source, const void *device_scalars,
                           const stream &execution_stream) {
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(execution_stream.opaque_handle());
  if (source.sequential)
    ordered_source_batch_kernel<T><<<1, 1, 0, cuda_stream>>>(
        source, static_cast<const source_scalar *>(device_scalars));
  else {
    const unsigned int threads = static_cast<unsigned int>(
        source.point_count < size_t(256) ? source.point_count : size_t(256));
    const size_t blocks = (source.point_count + threads - 1) / threads;
    if (blocks > size_t(std::numeric_limits<unsigned int>::max()))
      throw std::overflow_error("NVIDIA source batch launch grid overflow");
    source_batch_kernel<T><<<static_cast<unsigned int>(blocks), threads, 0, cuda_stream>>>(
        source, static_cast<const source_scalar *>(device_scalars));
  }
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA source batch");
}

template <typename T>
void launch_array_copy_t(const array_copy_launch &copy, const stream &execution_stream) {
  const unsigned int threads = 256;
  const size_t block_count = (copy.elements + threads - 1) / threads;
  if (block_count > size_t(std::numeric_limits<unsigned int>::max()))
    throw std::overflow_error("NVIDIA source copy launch grid overflow");
  array_copy_kernel<T><<<static_cast<unsigned int>(block_count), threads, 0,
                         static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(copy);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA source base copy");
}

} // namespace

void launch_source_batch(const source_batch_launch &source, const void *device_scalars,
                         const stream &execution_stream) {
  if (!source.target_real) throw std::invalid_argument("NVIDIA source batch has no target");
  if (!source.points || !source.point_count)
    throw std::invalid_argument("NVIDIA source batch has no spatial points");
  if (!device_scalars) throw std::invalid_argument("NVIDIA source batch has no scalar block");
  switch (source.precision) {
    case scalar_precision::f32:
      launch_source_batch_t<float>(source, device_scalars, execution_stream);
      return;
    case scalar_precision::f64:
      launch_source_batch_t<double>(source, device_scalars, execution_stream);
      return;
  }
  throw std::invalid_argument("NVIDIA source batch has an invalid precision");
}

void launch_array_copy(const array_copy_launch &copy, const stream &execution_stream) {
  if (!copy.target || !copy.source || !copy.elements)
    throw std::invalid_argument("NVIDIA source base copy has incomplete operands");
  switch (copy.precision) {
    case scalar_precision::f32: launch_array_copy_t<float>(copy, execution_stream); return;
    case scalar_precision::f64: launch_array_copy_t<double>(copy, execution_stream); return;
  }
  throw std::invalid_argument("NVIDIA source base copy has an invalid precision");
}

} // namespace nvidia
} // namespace meep
