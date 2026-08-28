/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_dft.hpp"

#include <cuda_runtime_api.h>

#include <math.h>

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

void launch_geometry(size_t count, unsigned int &blocks, unsigned int &threads) {
  if (!count) throw std::invalid_argument("NVIDIA DFT launch is empty");
  threads = 256;
  const size_t block_count = (count + threads - 1) / threads;
  if (block_count > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA DFT launch grid overflow");
  blocks = static_cast<unsigned int>(block_count);
}

__device__ void region_coordinates(const flat_region &region, size_t linear, size_t &i0, size_t &i1,
                                   size_t &i2) {
  i2 = linear % region.counts[2];
  linear /= region.counts[2];
  i1 = linear % region.counts[1];
  i0 = linear / region.counts[1];
}

__device__ ptrdiff_t region_index(const flat_region &region, size_t i0, size_t i1, size_t i2) {
  return ptrdiff_t(region.base) + ptrdiff_t(i0) * region.strides[0] +
         ptrdiff_t(i1) * region.strides[1] + ptrdiff_t(i2) * region.strides[2];
}

__device__ double boundary_weight(size_t i_value, size_t n_value, double start0, double start1,
                                  double end0, double end1) {
  /* Preserve IVEC_LOOP_WEIGHT1x branch order for overlapping endpoints. */
  const ptrdiff_t i = ptrdiff_t(i_value);
  const ptrdiff_t n = ptrdiff_t(n_value);
  if (i > 1 && i < n - 2) return 1.0;
  if (i == 0) return start0;
  if (i == 1) return start1;
  if (i == n - 1) return end0;
  if (i == n - 2) return end1;
  return 1.0;
}

template <typename MonitorT>
__global__ void dft_phase_kernel(dft_launch launch, double sample_time) {
  const size_t frequency = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (frequency >= launch.frequencies) return;
  const double angle = launch.omega[launch.omega_offset + frequency] * sample_time;
  double sine, cosine;
  sincos(angle, &sine, &cosine);
  MonitorT *phase = static_cast<MonitorT *>(launch.phase_scratch);
  phase[2 * frequency] = MonitorT(cosine * launch.scale_real - sine * launch.scale_imag);
  phase[2 * frequency + 1] = MonitorT(cosine * launch.scale_imag + sine * launch.scale_real);
}

template <typename FieldT, typename MonitorT>
__global__ void dft_accumulate_kernel(dft_launch launch, size_t outputs) {
  const size_t output = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output >= outputs) return;
  const size_t frequency = output % launch.frequencies;
  const size_t voxel = output / launch.frequencies;

  size_t i0, i1, i2;
  region_coordinates(launch.region, voxel, i0, i1, i2);
  const ptrdiff_t index = region_index(launch.region, i0, i1, i2);

  double weight = 1.0;
  if (launch.include_weights) {
    weight = boundary_weight(i2, launch.region.counts[2], launch.start0[2], launch.start1[2],
                             launch.end0[2], launch.end1[2]) *
             (boundary_weight(i1, launch.region.counts[1], launch.start0[1], launch.start1[1],
                              launch.end0[1], launch.end1[1]) *
              ((launch.dV0 + launch.dV1 * double(i1)) *
               boundary_weight(i0, launch.region.counts[0], launch.start0[0], launch.start1[0],
                               launch.end0[0], launch.end1[0])));
    if (launch.sqrt_weights) weight = sqrt(weight);
  }

  const FieldT *source_real = static_cast<const FieldT *>(launch.source_real);
  const FieldT *source_imag = static_cast<const FieldT *>(launch.source_imag);
  const FieldT w = FieldT(weight);
  FieldT field_real;
  FieldT field_imag = FieldT(0);
  if (launch.avg2) {
    const FieldT scale = w * FieldT(0.25);
    field_real = scale * (((source_real[index] + source_real[index + launch.avg1]) +
                           source_real[index + launch.avg2]) +
                          source_real[index + launch.avg1 + launch.avg2]);
    if (source_imag)
      field_imag = scale * (((source_imag[index] + source_imag[index + launch.avg1]) +
                             source_imag[index + launch.avg2]) +
                            source_imag[index + launch.avg1 + launch.avg2]);
  }
  else if (launch.avg1) {
    const FieldT scale = w * FieldT(0.5);
    field_real = scale * (source_real[index] + source_real[index + launch.avg1]);
    if (source_imag) field_imag = scale * (source_imag[index] + source_imag[index + launch.avg1]);
  }
  else {
    field_real = w * source_real[index];
    if (source_imag) field_imag = w * source_imag[index];
  }

  const MonitorT *phase = static_cast<const MonitorT *>(launch.phase_scratch);
  const MonitorT phase_real = phase[2 * frequency];
  const MonitorT phase_imag = phase[2 * frequency + 1];
  const MonitorT value_real = MonitorT(field_real);
  const MonitorT value_imag = MonitorT(field_imag);
  MonitorT *accumulator = static_cast<MonitorT *>(launch.accumulator);
  accumulator[2 * output] += phase_real * value_real - phase_imag * value_imag;
  accumulator[2 * output + 1] += phase_real * value_imag + phase_imag * value_real;
}

template <typename MonitorT>
void launch_phase(const dft_launch &launch, double sample_time, const stream &execution_stream) {
  unsigned int blocks = 0, threads = 0;
  launch_geometry(launch.frequencies, blocks, threads);
  dft_phase_kernel<MonitorT>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          launch, sample_time);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA DFT phase");
}

template <typename FieldT, typename MonitorT>
void launch_accumulation(const dft_launch &launch, const stream &execution_stream) {
  if (launch.points > std::numeric_limits<size_t>::max() / launch.frequencies)
    throw std::overflow_error("NVIDIA DFT output count overflow");
  const size_t outputs = launch.points * launch.frequencies;
  unsigned int blocks = 0, threads = 0;
  launch_geometry(outputs, blocks, threads);
  dft_accumulate_kernel<FieldT, MonitorT>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          launch, outputs);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA DFT accumulation");
}

} // namespace

void launch_dft(const dft_launch &launch, double sample_time, const stream &execution_stream) {
  if (!launch.source_real || !launch.accumulator || !launch.phase_scratch || !launch.omega)
    throw std::invalid_argument("NVIDIA DFT launch has incomplete storage");
  if (!launch.points || !launch.frequencies || launch.decimation_factor <= 0)
    throw std::invalid_argument("NVIDIA DFT launch has invalid dimensions or decimation");
  if (launch.monitor_precision == scalar_precision::f32) {
    launch_phase<float>(launch, sample_time, execution_stream);
    if (launch.field_precision == scalar_precision::f32)
      launch_accumulation<float, float>(launch, execution_stream);
    else
      throw std::invalid_argument("NVIDIA DFT does not support f64 fields with f32 monitors");
  }
  else {
    launch_phase<double>(launch, sample_time, execution_stream);
    if (launch.field_precision == scalar_precision::f32)
      launch_accumulation<float, double>(launch, execution_stream);
    else
      launch_accumulation<double, double>(launch, execution_stream);
  }
}

} // namespace nvidia
} // namespace meep
