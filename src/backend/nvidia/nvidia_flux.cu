/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_flux.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>

namespace meep {
namespace nvidia {
namespace {

const unsigned int flux_threads = 256;

void check_cuda(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) return;
  throw runtime_error(operation, static_cast<int>(result), cudaGetErrorName(result),
                      cudaGetErrorString(result));
}

__device__ double boundary_weight(size_t i, size_t n, double s0, double s1, double e0,
                                  double e1) {
  const ptrdiff_t si = ptrdiff_t(i), sn = ptrdiff_t(n);
  if (si > 1 && si < sn - 2) return 1.0;
  if (si == 0) return s0;
  if (si == 1) return s1;
  if (si == sn - 1) return e0;
  if (si == sn - 2) return e1;
  return 1.0;
}

template <typename T>
__device__ double centered(const T *values, ptrdiff_t index, const ptrdiff_t offsets[2]) {
  if (!values) return 0.0;
  const T sum = (values[index] + values[index + offsets[0]]) +
                (values[index + offsets[1]] + values[index + offsets[0] + offsets[1]]);
  return 0.25 * double(sum);
}

template <typename T>
__global__ void legacy_flux_partial_kernel(legacy_flux_term_launch launch, double *partials) {
  __shared__ double sums[flux_threads];
  double sum = 0.0;
  const T *er = static_cast<const T *>(launch.e_real);
  const T *ei = static_cast<const T *>(launch.e_imag);
  const T *hr = static_cast<const T *>(launch.h_real);
  const T *hi = static_cast<const T *>(launch.h_imag);
  for (size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x; linear < launch.points;
       linear += size_t(gridDim.x) * blockDim.x) {
    size_t rest = linear;
    const size_t i2 = rest % launch.region.counts[2];
    rest /= launch.region.counts[2];
    const size_t i1 = rest % launch.region.counts[1];
    const size_t i0 = rest / launch.region.counts[1];
    const ptrdiff_t index = ptrdiff_t(launch.region.base) +
                            ptrdiff_t(i0) * launch.region.strides[0] +
                            ptrdiff_t(i1) * launch.region.strides[1] +
                            ptrdiff_t(i2) * launch.region.strides[2];
    const double e_real = centered(er, index, launch.e_offsets);
    const double e_imag = centered(ei, index, launch.e_offsets);
    const double h_real = centered(hr, index, launch.h_offsets);
    const double h_imag = centered(hi, index, launch.h_offsets);
    const double product_real = e_real * h_real + e_imag * h_imag;
    const double product_imag = e_real * h_imag - e_imag * h_real;
    const double phased =
        product_real * launch.phase_real - product_imag * launch.phase_imag;
    const double weight =
        boundary_weight(i2, launch.region.counts[2], launch.start0[2], launch.start1[2],
                        launch.end0[2], launch.end1[2]) *
        (boundary_weight(i1, launch.region.counts[1], launch.start0[1], launch.start1[1],
                         launch.end0[1], launch.end1[1]) *
         ((launch.dV0 + launch.dV1 * double(i1)) *
          boundary_weight(i0, launch.region.counts[0], launch.start0[0], launch.start1[0],
                          launch.end0[0], launch.end1[0])));
    sum += double(launch.sign) * phased * weight;
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (!threadIdx.x) partials[blockIdx.x] = sums[0];
}

__global__ void legacy_flux_final_kernel(const double *partials, size_t count, double *result) {
  __shared__ double sums[flux_threads];
  double sum = 0.0;
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) sum += partials[i];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (!threadIdx.x) *result += sums[0];
}

__global__ void legacy_flux_average_kernel(double *current, const double *half, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) current[i] = 0.5 * (current[i] + half[i]);
}

} // namespace

size_t legacy_flux_partial_count(size_t points) {
  if (!points) throw std::invalid_argument("NVIDIA legacy flux launch is empty");
  return 1 + (points - 1) / flux_threads;
}

void launch_legacy_flux_term(const legacy_flux_term_launch &launch, void *partials, void *result,
                             const stream &execution_stream) {
  size_t region_points = 1;
  for (int axis = 0; axis < 3; ++axis) {
    if (!launch.region.counts[axis] || launch.region.strides[axis] < 0)
      throw std::invalid_argument("NVIDIA legacy flux launch has invalid region geometry");
    if (launch.region.counts[axis] > std::numeric_limits<size_t>::max() / region_points)
      throw std::overflow_error("NVIDIA legacy flux launch region size overflow");
    region_points *= launch.region.counts[axis];
  }
  if (!launch.e_real || !launch.h_real || !partials || !result || !launch.points ||
      launch.points != region_points || launch.sign < -1 || launch.sign > 1 || !launch.sign ||
      launch.blocks != legacy_flux_partial_count(launch.points) ||
      launch.blocks > std::numeric_limits<unsigned int>::max() || partials == result ||
      partials == launch.e_real || partials == launch.e_imag || partials == launch.h_real ||
      partials == launch.h_imag || result == launch.e_real || result == launch.e_imag ||
      result == launch.h_real || result == launch.h_imag)
    throw std::invalid_argument("NVIDIA legacy flux launch has invalid operands or geometry");
  switch (launch.precision) {
    case scalar_precision::f32:
      legacy_flux_partial_kernel<float>
          <<<unsigned(launch.blocks), flux_threads, 0,
             static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
              launch, static_cast<double *>(partials));
      break;
    case scalar_precision::f64:
      legacy_flux_partial_kernel<double>
          <<<unsigned(launch.blocks), flux_threads, 0,
             static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
              launch, static_cast<double *>(partials));
      break;
    default: throw std::invalid_argument("NVIDIA legacy flux launch has invalid precision");
  }
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA legacy flux partial reduction");
  legacy_flux_final_kernel<<<1, flux_threads, 0,
                             static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
      static_cast<const double *>(partials), launch.blocks, static_cast<double *>(result));
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA legacy flux final reduction");
}

void launch_legacy_flux_average(void *current, const void *half, size_t count,
                                const stream &execution_stream) {
  if (!current || !half || !count)
    throw std::invalid_argument("NVIDIA legacy flux average has invalid operands");
  const size_t blocks = 1 + (count - 1) / flux_threads;
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA legacy flux average grid overflow");
  legacy_flux_average_kernel<<<unsigned(blocks), flux_threads, 0,
                               static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
      static_cast<double *>(current), static_cast<const double *>(half), count);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA legacy flux average");
}

} // namespace nvidia
} // namespace meep
