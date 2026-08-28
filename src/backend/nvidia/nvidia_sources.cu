/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_sources.hpp"

#include <cuda_runtime_api.h>

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
__global__ void point_source_kernel(point_source_launch source, const source_scalar *scalars) {
  const source_scalar scalar = scalars[source.scalar_slot];
  double value_real =
      source.amplitude_real * scalar.current_real - source.amplitude_imag * scalar.current_imag;
  double value_imag =
      source.amplitude_real * scalar.current_imag + source.amplitude_imag * scalar.current_real;
  /* Preserve fields_chunk::step_source's association: complex amplitude times
     current, then dt, then (when present) the stored conductivity inverse. */
  value_real *= source.dt;
  value_imag *= source.dt;
  if (source.conductivity_inverse) {
    const double condinv =
        double(static_cast<const T *>(source.conductivity_inverse)[source.index]);
    value_real *= condinv;
    value_imag *= condinv;
  }
  static_cast<T *>(source.target_real)[source.index] -= T(value_real);
  if (source.target_imag)
    static_cast<T *>(source.target_imag)[source.index] -= T(value_imag);
}

template <typename T>
void launch_point_source_t(const point_source_launch &source, const void *device_scalars,
                           const stream &execution_stream) {
  point_source_kernel<T><<<1, 1, 0,
                           static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
      source, static_cast<const source_scalar *>(device_scalars));
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA point source");
}

} // namespace

void launch_point_source(const point_source_launch &source, const void *device_scalars,
                         const stream &execution_stream) {
  if (!source.target_real) throw std::invalid_argument("NVIDIA point source has no target");
  if (!device_scalars) throw std::invalid_argument("NVIDIA point source has no scalar block");
  if (source.index < 0) throw std::out_of_range("NVIDIA point source has a negative index");
  switch (source.precision) {
    case scalar_precision::f32:
      launch_point_source_t<float>(source, device_scalars, execution_stream);
      return;
    case scalar_precision::f64:
      launch_point_source_t<double>(source, device_scalars, execution_stream);
      return;
  }
  throw std::invalid_argument("NVIDIA point source has an invalid precision");
}

} // namespace nvidia
} // namespace meep
