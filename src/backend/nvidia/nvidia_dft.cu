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

template <typename T> struct reduction_pair {
  T real;
  T imag;
};

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

__device__ bool graph_predicate(const StepScalars *scalars, uint32_t word, uint32_t bit) {
  return !scalars || ((scalars->predicate_words[word] >> bit) & uint64_t(1));
}

template <typename MonitorT>
__global__ void dft_phase_kernel(dft_launch launch, double sample_time,
                                 const StepScalars *scalars, uint32_t predicate_word,
                                 uint32_t predicate_bit) {
  const size_t frequency = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (frequency >= launch.frequencies ||
      !graph_predicate(scalars, predicate_word, predicate_bit))
    return;
  if (scalars) sample_time = scalars->source_times[launch.magnetic ? 1 : 2];
  const double angle = launch.omega[launch.omega_offset + frequency] * sample_time;
  double sine, cosine;
  sincos(angle, &sine, &cosine);
  MonitorT *phase = static_cast<MonitorT *>(launch.phase_scratch);
  phase[2 * frequency] = MonitorT(cosine * launch.scale_real - sine * launch.scale_imag);
  phase[2 * frequency + 1] = MonitorT(cosine * launch.scale_imag + sine * launch.scale_real);
}

template <typename FieldT, typename MonitorT>
__global__ void dft_accumulate_kernel(dft_launch launch, size_t outputs,
                                      const StepScalars *scalars, uint32_t predicate_word,
                                      uint32_t predicate_bit) {
  const size_t output = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output >= outputs || !graph_predicate(scalars, predicate_word, predicate_bit)) return;
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

template <typename AccumT>
__device__ reduction_pair<AccumT> add_pair(reduction_pair<AccumT> a,
                                           reduction_pair<AccumT> b) {
  return reduction_pair<AccumT>{a.real + b.real, a.imag + b.imag};
}

template <typename MonitorT, typename AccumT>
__global__ void dft_reduction_partials_kernel(dft_reduction_launch launch) {
  const size_t lane = size_t(blockIdx.x) / launch.blocks_per_lane;
  const size_t lane_block = size_t(blockIdx.x) % launch.blocks_per_lane;
  const size_t selected_points = launch.counts[0] * launch.counts[1] * launch.counts[2];
  const size_t lane_work = launch.operation == dft_reduction_operation::norm2
                               ? selected_points * launch.frequencies
                               : selected_points;
  reduction_pair<AccumT> sum = {AccumT(0), AccumT(0)};
  const MonitorT *left = static_cast<const MonitorT *>(launch.left);
  const MonitorT *right = static_cast<const MonitorT *>(launch.right);

  for (size_t linear = lane_block * blockDim.x + threadIdx.x; linear < lane_work;
       linear += launch.blocks_per_lane * blockDim.x) {
    size_t point = linear;
    size_t frequency = lane;
    if (launch.operation == dft_reduction_operation::norm2) {
      frequency = linear % launch.frequencies;
      point = linear / launch.frequencies;
    }
    size_t rest = point;
    const size_t i2 = rest % launch.counts[2];
    rest /= launch.counts[2];
    const size_t i1 = rest % launch.counts[1];
    const size_t i0 = rest / launch.counts[1];
    const size_t voxel = launch.base + i0 * launch.strides[0] + i1 * launch.strides[1] +
                         i2 * launch.strides[2];
    const size_t element = voxel * launch.frequencies + frequency;
    const AccumT ar = AccumT(left[2 * element]);
    const AccumT ai = AccumT(left[2 * element + 1]);
    if (launch.operation == dft_reduction_operation::norm2) {
      sum.real += ar * ar + ai * ai;
      continue;
    }
    const AccumT br = AccumT(right[2 * element]);
    const AccumT bi = AccumT(right[2 * element + 1]);
    const AccumT wr = AccumT(launch.weight_real);
    const AccumT wi = AccumT(launch.weight_imag);
    const AccumT weighted_real = wr * ar - wi * ai;
    const AccumT weighted_imag = wr * ai + wi * ar;
    sum.real += weighted_real * br + weighted_imag * bi;
    if (launch.operation == dft_reduction_operation::complex_weighted_product)
      sum.imag += weighted_imag * br - weighted_real * bi;
  }

  __shared__ reduction_pair<AccumT> shared[256];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset; offset >>= 1) {
    if (threadIdx.x < offset)
      shared[threadIdx.x] = add_pair(shared[threadIdx.x], shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    reduction_pair<AccumT> *partials = static_cast<reduction_pair<AccumT> *>(launch.partials);
    partials[lane * launch.blocks_per_lane + lane_block] = shared[0];
  }
}

template <typename AccumT>
__global__ void dft_reduction_final_kernel(dft_reduction_launch launch) {
  const size_t lane = blockIdx.x;
  reduction_pair<AccumT> sum = {AccumT(0), AccumT(0)};
  const reduction_pair<AccumT> *partials =
      static_cast<const reduction_pair<AccumT> *>(launch.partials);
  for (size_t block = threadIdx.x; block < launch.blocks_per_lane; block += blockDim.x)
    sum = add_pair(sum, partials[lane * launch.blocks_per_lane + block]);
  __shared__ reduction_pair<AccumT> shared[256];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset; offset >>= 1) {
    if (threadIdx.x < offset)
      shared[threadIdx.x] = add_pair(shared[threadIdx.x], shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double2 *result = static_cast<double2 *>(launch.result);
    result[lane].x += double(shared[0].real);
    result[lane].y += double(shared[0].imag);
  }
}

template <typename MonitorT>
void launch_phase(const dft_launch &launch, double sample_time, const StepScalars *scalars,
                  uint32_t predicate_word, uint32_t predicate_bit,
                  const stream &execution_stream) {
  unsigned int blocks = 0, threads = 0;
  launch_geometry(launch.frequencies, blocks, threads);
  dft_phase_kernel<MonitorT>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          launch, sample_time, scalars, predicate_word, predicate_bit);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA DFT phase");
}

template <typename FieldT, typename MonitorT>
void launch_accumulation(const dft_launch &launch, const StepScalars *scalars,
                         uint32_t predicate_word, uint32_t predicate_bit,
                         const stream &execution_stream) {
  if (launch.points > std::numeric_limits<size_t>::max() / launch.frequencies)
    throw std::overflow_error("NVIDIA DFT output count overflow");
  const size_t outputs = launch.points * launch.frequencies;
  unsigned int blocks = 0, threads = 0;
  launch_geometry(outputs, blocks, threads);
  dft_accumulate_kernel<FieldT, MonitorT>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          launch, outputs, scalars, predicate_word, predicate_bit);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA DFT accumulation");
}

} // namespace

void launch_dft(const dft_launch &launch, double sample_time, const stream &execution_stream) {
  if (!launch.source_real || !launch.accumulator || !launch.phase_scratch || !launch.omega)
    throw std::invalid_argument("NVIDIA DFT launch has incomplete storage");
  if (!launch.points || !launch.frequencies || launch.decimation_factor <= 0)
    throw std::invalid_argument("NVIDIA DFT launch has invalid dimensions or decimation");
  if (launch.monitor_precision == scalar_precision::f32) {
    launch_phase<float>(launch, sample_time, NULL, 0, 0, execution_stream);
    if (launch.field_precision == scalar_precision::f32)
      launch_accumulation<float, float>(launch, NULL, 0, 0, execution_stream);
    else
      throw std::invalid_argument("NVIDIA DFT does not support f64 fields with f32 monitors");
  }
  else {
    launch_phase<double>(launch, sample_time, NULL, 0, 0, execution_stream);
    if (launch.field_precision == scalar_precision::f32)
      launch_accumulation<float, double>(launch, NULL, 0, 0, execution_stream);
    else
      launch_accumulation<double, double>(launch, NULL, 0, 0, execution_stream);
  }
}

void launch_dft_graph(const dft_launch &launch, const StepScalars *scalars,
                      uint32_t predicate_word, uint32_t predicate_bit,
                      const stream &execution_stream) {
  if (!scalars) throw std::invalid_argument("NVIDIA graph DFT has no StepScalars");
  if (predicate_word >= step_scalar_predicate_word_count || predicate_bit >= 64)
    throw std::invalid_argument("NVIDIA graph DFT predicate is out of range");
  if (!launch.source_real || !launch.accumulator || !launch.phase_scratch || !launch.omega)
    throw std::invalid_argument("NVIDIA DFT launch has incomplete storage");
  if (!launch.points || !launch.frequencies || launch.decimation_factor <= 0)
    throw std::invalid_argument("NVIDIA DFT launch has invalid dimensions or decimation");
  if (launch.monitor_precision == scalar_precision::f32) {
    launch_phase<float>(launch, 0.0, scalars, predicate_word, predicate_bit, execution_stream);
    if (launch.field_precision == scalar_precision::f32)
      launch_accumulation<float, float>(launch, scalars, predicate_word, predicate_bit,
                                        execution_stream);
    else
      throw std::invalid_argument("NVIDIA DFT does not support f64 fields with f32 monitors");
  }
  else {
    launch_phase<double>(launch, 0.0, scalars, predicate_word, predicate_bit, execution_stream);
    if (launch.field_precision == scalar_precision::f32)
      launch_accumulation<float, double>(launch, scalars, predicate_word, predicate_bit,
                                         execution_stream);
    else
      launch_accumulation<double, double>(launch, scalars, predicate_word, predicate_bit,
                                          execution_stream);
  }
}

template <typename MonitorT, typename AccumT>
void launch_reduction_typed(const dft_reduction_launch &launch,
                            const stream &execution_stream) {
  const unsigned int threads = 256;
  const size_t total_blocks = launch.result_count * launch.blocks_per_lane;
  if (total_blocks > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA DFT reduction grid overflow");
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(execution_stream.opaque_handle());
  dft_reduction_partials_kernel<MonitorT, AccumT>
      <<<static_cast<unsigned int>(total_blocks), threads, 0, cuda_stream>>>(launch);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA DFT reduction partials");
  if (launch.result_count > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA DFT reduction result grid overflow");
  dft_reduction_final_kernel<AccumT>
      <<<static_cast<unsigned int>(launch.result_count), threads, 0, cuda_stream>>>(launch);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA DFT reduction final");
}

void launch_dft_reduction(const dft_reduction_launch &launch,
                          const stream &execution_stream) {
  if (!launch.left || !launch.partials || !launch.result || !launch.storage_points ||
      !launch.frequencies || !launch.result_count || !launch.blocks_per_lane)
    throw std::invalid_argument("NVIDIA DFT reduction launch is incomplete");
  if (launch.operation != dft_reduction_operation::norm2 && !launch.right)
    throw std::invalid_argument("NVIDIA DFT product reduction has no right operand");
  if (launch.accumulation_precision == scalar_precision::f64) {
    if (launch.monitor_precision == scalar_precision::f32)
      launch_reduction_typed<float, double>(launch, execution_stream);
    else
      launch_reduction_typed<double, double>(launch, execution_stream);
  }
  else {
    if (launch.monitor_precision != scalar_precision::f32)
      throw std::invalid_argument("NVIDIA DFT does not reduce f64 monitor storage in f32");
    launch_reduction_typed<float, float>(launch, execution_stream);
  }
}

} // namespace nvidia
} // namespace meep
