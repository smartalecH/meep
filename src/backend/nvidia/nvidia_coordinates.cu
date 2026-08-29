/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_coordinates.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace meep {
namespace nvidia {
namespace {

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

size_t checked_points(const flat_region &region) {
  size_t points = 1;
  for (int axis = 0; axis < 3; ++axis) {
    if (!region.counts[axis]) throw std::invalid_argument("NVIDIA BFAST launch is empty");
    if (points > std::numeric_limits<size_t>::max() / region.counts[axis])
      throw std::overflow_error("NVIDIA BFAST launch point count overflow");
    points *= region.counts[axis];
  }
  return points;
}

__device__ ptrdiff_t region_index(const flat_region &region, size_t i0, size_t i1, size_t i2) {
  return ptrdiff_t(region.base) + ptrdiff_t(i0) * region.strides[0] +
         ptrdiff_t(i1) * region.strides[1] + ptrdiff_t(i2) * region.strides[2];
}

__device__ void region_coordinates(const flat_region &region, size_t linear, size_t &i0,
                                   size_t &i1, size_t &i2) {
  i2 = linear % region.counts[2];
  linear /= region.counts[2];
  i1 = linear % region.counts[1];
  i0 = linear / region.counts[1];
}

__device__ ptrdiff_t profile_index(const pml_profile_launch &profile, size_t i0, size_t i1,
                                   size_t i2) {
  return profile.base + ptrdiff_t(i0) * profile.strides[0] +
         ptrdiff_t(i1) * profile.strides[1] + ptrdiff_t(i2) * profile.strides[2];
}

template <typename T, bool MainPml, bool AuxiliaryPml, bool Conductivity>
__global__ void bfast_kernel(bfast_launch update, size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  size_t i0, i1, i2;
  region_coordinates(update.region, linear, i0, i1, i2);
  const ptrdiff_t i = region_index(update.region, i0, i1, i2);
  T *target = static_cast<T *>(update.target);
  const T *source1 = static_cast<const T *>(update.source1);
  const T *source2 = static_cast<const T *>(update.source2);
  ptrdiff_t stride1 = update.stride1;
  ptrdiff_t stride2 = update.stride2;
  T k1 = T(update.k1);
  T k2 = T(update.k2);
  if (!source1) {
    const T *source_swap = source1;
    source1 = source2;
    source2 = source_swap;
    const ptrdiff_t stride_swap = stride1;
    stride1 = stride2;
    stride2 = stride_swap;
    const T k_swap = k1;
    k1 = k2;
    k2 = k_swap;
  }

  T *state = static_cast<T *>(update.f_bfast);
  const T previous = state[i];
  T next;
  if (source2)
    next = (k1 * (source1[i + stride1] + source1[i]) -
            k2 * (source2[i + stride2] + source2[i])) -
           previous;
  /* Preserve step_bfast's exceptional one-source/no-auxiliary branch: only
     this exact shape assigns the fresh term without subtracting old F. */
  else if (!MainPml && !AuxiliaryPml && !Conductivity)
    next = k1 * (source1[i + stride1] + source1[i]);
  else
    next = k1 * (source1[i + stride1] + source1[i]) - previous;
  state[i] = next;

  T delta = next - previous;
  if (Conductivity)
    delta *= static_cast<const T *>(update.conductivity_inverse)[i];
  if (MainPml) {
    if (Conductivity) static_cast<T *>(update.target_conductivity)[i] += delta;
    const ptrdiff_t k = profile_index(update.pml, i0, i1, i2);
    delta *= static_cast<const T *>(update.pml.inverse)[k];
  }
  if (AuxiliaryPml) {
    static_cast<T *>(update.target_u)[i] += delta;
    const ptrdiff_t ku = profile_index(update.pml_u, i0, i1, i2);
    delta *= static_cast<const T *>(update.pml_u.inverse)[ku];
  }
  target[i] += delta;
}

template <typename T, bool MainPml, bool AuxiliaryPml, bool Conductivity>
void launch_t(const bfast_launch &update, const stream &execution_stream) {
  const size_t points = checked_points(update.region);
  const unsigned int threads = 256;
  const size_t block_count = 1 + (points - 1) / threads;
  if (block_count > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA BFAST launch grid overflow");
  bfast_kernel<T, MainPml, AuxiliaryPml, Conductivity>
      <<<static_cast<unsigned int>(block_count), threads, 0,
         static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(update, points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA BFAST update");
}

} // namespace

void launch_bfast(const bfast_launch &update, const stream &execution_stream) {
  if (!update.target || !update.f_bfast || (!update.source1 && !update.source2))
    throw std::invalid_argument("NVIDIA BFAST launch has incomplete operands");
  const bool main_pml = update.pml.inverse != NULL;
  const bool auxiliary_pml = update.pml_u.inverse != NULL;
  const bool conductivity = update.conductivity_inverse != NULL;
  if (auxiliary_pml != (update.target_u != NULL))
    throw std::invalid_argument("NVIDIA BFAST auxiliary PML state is incomplete");
  if ((main_pml && conductivity) != (update.target_conductivity != NULL))
    throw std::invalid_argument("NVIDIA BFAST conductivity target is inconsistent");
  const void *mutable_arrays[] = {update.target, update.f_bfast, update.target_u,
                                  update.target_conductivity};
  for (size_t i = 0; i < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++i) {
    if (!mutable_arrays[i]) continue;
    if (mutable_arrays[i] == update.source1 || mutable_arrays[i] == update.source2)
      throw std::invalid_argument("NVIDIA BFAST launch aliases mutable and input state");
    for (size_t j = i + 1; j < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++j)
      if (mutable_arrays[j] && mutable_arrays[i] == mutable_arrays[j])
        throw std::invalid_argument("NVIDIA BFAST launch aliases mutable state");
  }

#define LAUNCH_BFAST_VARIANT(T, MP, AP, C) launch_t<T, MP, AP, C>(update, execution_stream)
#define DISPATCH_BFAST(T)                                                                           \
  do {                                                                                              \
    const unsigned int key = unsigned(main_pml) | (unsigned(auxiliary_pml) << 1) |                  \
                             (unsigned(conductivity) << 2);                                         \
    switch (key) {                                                                                  \
      case 0: LAUNCH_BFAST_VARIANT(T, false, false, false); break;                                  \
      case 1: LAUNCH_BFAST_VARIANT(T, true, false, false); break;                                   \
      case 2: LAUNCH_BFAST_VARIANT(T, false, true, false); break;                                   \
      case 3: LAUNCH_BFAST_VARIANT(T, true, true, false); break;                                    \
      case 4: LAUNCH_BFAST_VARIANT(T, false, false, true); break;                                   \
      case 5: LAUNCH_BFAST_VARIANT(T, true, false, true); break;                                    \
      case 6: LAUNCH_BFAST_VARIANT(T, false, true, true); break;                                    \
      case 7: LAUNCH_BFAST_VARIANT(T, true, true, true); break;                                     \
    }                                                                                               \
  } while (0)
  if (update.precision == scalar_precision::f32)
    DISPATCH_BFAST(float);
  else if (update.precision == scalar_precision::f64)
    DISPATCH_BFAST(double);
  else
    throw std::invalid_argument("NVIDIA BFAST launch precision is invalid");
#undef DISPATCH_BFAST
#undef LAUNCH_BFAST_VARIANT
}

} // namespace nvidia
} // namespace meep
