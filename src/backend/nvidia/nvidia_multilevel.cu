/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_multilevel.hpp"

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
    if (!region.counts[axis]) throw std::invalid_argument("NVIDIA multilevel region is empty");
    if (region.counts[axis] > std::numeric_limits<size_t>::max() / points)
      throw std::overflow_error("NVIDIA multilevel region size overflow");
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

template <typename T>
__device__ T four_point_sum(const T *values, ptrdiff_t i, ptrdiff_t o1, ptrdiff_t o2) {
  return values[i] + values[i + o1] + values[i + o2] + values[i + o1 + o2];
}

template <typename T>
__device__ T eight_point_sum(const T *current, const T *previous, ptrdiff_t i,
                             ptrdiff_t o1, ptrdiff_t o2) {
  return current[i] + current[i + o1] + current[i + o2] + current[i + o1 + o2] +
         previous[i] + previous[i + o1] + previous[i + o2] +
         previous[i + o1 + o2];
}

template <typename T>
__global__ void multilevel_population_kernel(multilevel_population_launch update,
                                             size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  const ptrdiff_t i = region_index(update.region, linear);
  T *populations = static_cast<T *>(update.populations) + size_t(i) * update.levels;
  T *temporary = static_cast<T *>(update.scratch) + linear * update.levels;
  const T *gamma_inv = static_cast<const T *>(update.gamma_inv);
  const T *gamma = static_cast<const T *>(update.gamma_matrix);
  const T *alpha = static_cast<const T *>(update.alpha);
  const T *gperpdt = static_cast<const T *>(update.transition_gperpdt);

  for (uint32_t l1 = 0; l1 < update.levels; ++l1) {
    temporary[l1] = T(0);
    for (uint32_t l2 = 0; l2 < update.levels; ++l2)
      temporary[l1] += gamma[l1 * update.levels + l2] * populations[l2];
  }

  for (uint32_t transition = 0; transition < update.transitions; ++transition) {
    T edp32 = T(0), epave64 = T(0);
    for (uint32_t row = 0; row < update.term_count; ++row) {
      const multilevel_population_term_launch term = update.terms[row];
      if (term.transition_index != transition) continue;
      const ptrdiff_t o1 = term.centered_offsets[0], o2 = term.centered_offsets[1];
      const T e8 = eight_point_sum(static_cast<const T *>(term.w),
                                   static_cast<const T *>(term.w_prev), i, o1, o2);
      const T p = four_point_sum(static_cast<const T *>(term.p), i, o1, o2);
      const T p_prev = four_point_sum(static_cast<const T *>(term.p_prev), i, o1, o2);
      edp32 += (p - p_prev) * e8;
      epave64 += (p + p_prev) * e8;
    }
    edp32 *= T(0.03125);
    epave64 *= T(0.015625);
    for (uint32_t level = 0; level < update.levels; ++level) {
      const T a = alpha[level * update.transitions + transition];
      temporary[level] += a * edp32 + a * gperpdt[transition] * epave64;
    }
  }

  for (uint32_t l1 = 0; l1 < update.levels; ++l1) {
    T next = T(0);
    for (uint32_t l2 = 0; l2 < update.levels; ++l2)
      next += gamma_inv[l1 * update.levels + l2] * temporary[l2];
    populations[l1] = next;
  }
}

template <typename T>
__global__ void multilevel_transition_kernel(multilevel_transition_launch update,
                                             size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  const ptrdiff_t i = region_index(update.region, linear);
  T *p = static_cast<T *>(update.p);
  T *p_prev = static_cast<T *>(update.p_prev);
  const T *w = static_cast<const T *>(update.w);
  const T *sigma = static_cast<const T *>(update.diagonal_sigma);
  const T *population = static_cast<const T *>(update.populations) + size_t(i) * update.population_stride;
  const T *coefficient = static_cast<const T *>(update.coefficients);
  const ptrdiff_t o1 = update.population_offsets[0], o2 = update.population_offsets[1];
  const uint32_t lp = update.positive_level, lm = update.negative_level;
  const T inversion = T(0.25) *
                      (population[lp] + population[lp + o1] + population[lp + o2] +
                       population[lp + o1 + o2] - population[lm] - population[lm + o1] -
                       population[lm + o2] - population[lm + o1 + o2]);
  const T current = p[i];
  p[i] = coefficient[1] *
         (current * (T(2) - coefficient[0]) - coefficient[2] * p_prev[i] -
          coefficient[3] * (coefficient[4] * sigma[i] * w[i]) * inversion);
  p_prev[i] = current;
}

void launch_geometry(size_t points, unsigned int &blocks, unsigned int &threads) {
  threads = 256;
  const size_t block_count = (points + threads - 1) / threads;
  if (block_count > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA multilevel launch grid overflow");
  blocks = static_cast<unsigned int>(block_count);
}

template <typename T>
void launch_population_t(const multilevel_population_launch &update,
                         const stream &execution_stream) {
  const size_t points = checked_points(update.region);
  if (update.levels > std::numeric_limits<size_t>::max() / points)
    throw std::overflow_error("NVIDIA multilevel scratch index overflow");
  unsigned int blocks = 0, threads = 0;
  launch_geometry(points, blocks, threads);
  multilevel_population_kernel<T>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          update, points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA multilevel population update");
}

template <typename T>
void launch_transition_t(const multilevel_transition_launch &update,
                         const stream &execution_stream) {
  const size_t points = checked_points(update.region);
  unsigned int blocks = 0, threads = 0;
  launch_geometry(points, blocks, threads);
  multilevel_transition_kernel<T>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          update, points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA multilevel transition update");
}

} // namespace

void launch_multilevel_population(const multilevel_population_launch &update,
                                  const stream &execution_stream) {
  if (!update.populations || !update.gamma_inv || !update.gamma_matrix || !update.alpha ||
      !update.transition_gperpdt || !update.scratch || !update.levels || !update.transitions ||
      (update.term_count && !update.terms))
    throw std::invalid_argument("NVIDIA multilevel population update has incomplete operands");
  if (update.precision == scalar_precision::f32)
    launch_population_t<float>(update, execution_stream);
  else if (update.precision == scalar_precision::f64)
    launch_population_t<double>(update, execution_stream);
  else
    throw std::invalid_argument("NVIDIA multilevel population update has invalid precision");
}

void launch_multilevel_transition(const multilevel_transition_launch &update,
                                  const stream &execution_stream) {
  if (!update.p || !update.p_prev || !update.w || !update.diagonal_sigma ||
      !update.populations || !update.coefficients || update.p == update.p_prev ||
      !update.population_stride || update.positive_level >= update.population_stride ||
      update.negative_level >= update.population_stride ||
      update.positive_level == update.negative_level)
    throw std::invalid_argument("NVIDIA multilevel transition update has invalid operands");
  if (update.precision == scalar_precision::f32)
    launch_transition_t<float>(update, execution_stream);
  else if (update.precision == scalar_precision::f64)
    launch_transition_t<double>(update, execution_stream);
  else
    throw std::invalid_argument("NVIDIA multilevel transition update has invalid precision");
}

} // namespace nvidia
} // namespace meep
