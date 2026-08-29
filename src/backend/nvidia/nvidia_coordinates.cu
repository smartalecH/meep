/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_coordinates.hpp"

#include <cuda_runtime.h>

#include <cmath>
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

size_t checked_points(const flat_region &region, const char *what) {
  size_t points = 1;
  for (int axis = 0; axis < 3; ++axis) {
    if (!region.counts[axis])
      throw std::invalid_argument(std::string("NVIDIA ") + what + " launch is empty");
    if (points > std::numeric_limits<size_t>::max() / region.counts[axis])
      throw std::overflow_error(std::string("NVIDIA ") + what + " launch point count overflow");
    points *= region.counts[axis];
  }
  return points;
}

unsigned int checked_blocks(size_t points, const char *what) {
  const size_t threads = 256;
  const size_t blocks = 1 + (points - 1) / threads;
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error(std::string("NVIDIA ") + what + " launch grid overflow");
  return static_cast<unsigned int>(blocks);
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
    next = fma(-k2, source2[i + stride2] + source2[i],
               k1 * (source1[i + stride1] + source1[i])) -
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

template <typename T>
__global__ void cylindrical_radial_prefix_kernel(cylindrical_radial_prefix_launch update) {
  const size_t iz = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (iz > update.nz) return;
  const T *source = static_cast<const T *>(update.source);
  T *scratch = static_cast<T *>(update.scratch);
  const T ir0 = T(update.ir0);
  scratch[iz] = T(0);
  for (size_t ir = 1; ir <= update.nr; ++ir) {
    const size_t i = ir * update.row_stride + iz;
    const T rinv = 1.0 / ((T(ir) + ir0) - 0.5);
    const T weighted = fma(-source[i - update.row_stride], T(ir - 1) + ir0,
                           source[i] * (T(ir) + ir0));
    scratch[i] = fma(rinv, weighted, scratch[i - update.row_stride]);
  }
}

template <typename T, bool MainPml, bool AuxiliaryPml, bool Conductivity>
__global__ void cylindrical_m_kernel(cylindrical_m_launch update, size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  size_t i0, i1, i2;
  region_coordinates(update.region, linear, i0, i1, i2);
  const size_t coordinates[3] = {i0, i1, i2};
  const ptrdiff_t i = region_index(update.region, i0, i1, i2);
  T *target = static_cast<T *>(update.target);
  const T *source = static_cast<const T *>(update.source);
  T *target_u = static_cast<T *>(update.target_u);
  const T *conductivity_inverse = static_cast<const T *>(update.conductivity_inverse);
  T *target_conductivity = static_cast<T *>(update.target_conductivity);
  const T denominator = T(update.raw_radial_start + 2 * ptrdiff_t(coordinates[update.radial_axis]));
  T delta = T(update.numerator) / denominator * source[i];
  if (Conductivity) delta *= conductivity_inverse[i];
  if (MainPml) {
    if (Conductivity) target_conductivity[i] += delta;
    const ptrdiff_t k = profile_index(update.pml, i0, i1, i2);
    delta *= static_cast<const T *>(update.pml.inverse)[k];
  }
  if (AuxiliaryPml) {
    target_u[i] += delta;
    const ptrdiff_t ku = profile_index(update.pml_u, i0, i1, i2);
    delta *= static_cast<const T *>(update.pml_u.inverse)[ku];
  }
  target[i] += delta;
}

template <typename T, bool MainPml, bool AuxiliaryPml, bool Conductivity>
__global__ void cylindrical_axis_kernel(cylindrical_axis_launch update, size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  size_t i0, i1, i2;
  region_coordinates(update.region, linear, i0, i1, i2);
  const ptrdiff_t i = region_index(update.region, i0, i1, i2);
  T *target = static_cast<T *>(update.target);
  T *target_u = static_cast<T *>(update.target_u);
  const T *source1 = static_cast<const T *>(update.source1);
  const T *source2 = static_cast<const T *>(update.source2);
  T delta;
  if (update.kind == 0)
    delta = T(update.scale) * source1[i];
  else
    delta = T(update.scale) *
            (source1[i] - source1[i + update.source1_neighbor_offset] -
             T(update.source2_multiplier) * source2[i + update.source2_offset]);

  T *primary = AuxiliaryPml ? target_u : target;
  const T previous = primary[i];
  if (Conductivity) {
    T *target_conductivity = static_cast<T *>(update.target_conductivity);
    const T conductivity_previous = target_conductivity[i];
    target_conductivity[i] =
        ((T(1) - T(0.5 * update.dt) * static_cast<const T *>(update.conductivity)[i]) *
             conductivity_previous +
         delta) *
        static_cast<const T *>(update.conductivity_inverse)[i];
    delta = target_conductivity[i] - conductivity_previous;
  }
  if (MainPml) {
    const ptrdiff_t k = profile_index(update.pml, i0, i1, i2);
    primary[i] = ((static_cast<const T *>(update.pml.kappa)[k] -
                   static_cast<const T *>(update.pml.sigma)[k]) *
                      primary[i] +
                  delta) *
                 static_cast<const T *>(update.pml.inverse)[k];
  }
  else
    primary[i] += delta;

  if (AuxiliaryPml) {
    const ptrdiff_t ku = profile_index(update.pml_u, i0, i1, i2);
    target[i] = static_cast<const T *>(update.pml_u.inverse)[ku] *
                (((static_cast<const T *>(update.pml_u.kappa)[ku] -
                   static_cast<const T *>(update.pml_u.sigma)[ku]) *
                      target[i] +
                  primary[i]) -
                 previous);
  }
}

template <typename T, bool MainPml, bool AuxiliaryPml, bool Conductivity>
void launch_t(const bfast_launch &update, const stream &execution_stream) {
  const size_t points = checked_points(update.region, "BFAST");
  const unsigned int threads = 256;
  const unsigned int block_count = checked_blocks(points, "BFAST");
  bfast_kernel<T, MainPml, AuxiliaryPml, Conductivity>
      <<<block_count, threads, 0,
         static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(update, points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA BFAST update");
}

template <typename T, bool MainPml, bool AuxiliaryPml, bool Conductivity>
void launch_cylindrical_m_t(const cylindrical_m_launch &update,
                            const stream &execution_stream) {
  const size_t points = checked_points(update.region, "cylindrical m/r");
  const unsigned int threads = 256;
  cylindrical_m_kernel<T, MainPml, AuxiliaryPml, Conductivity>
      <<<checked_blocks(points, "cylindrical m/r"), threads, 0,
         static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(update, points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA cylindrical m/r update");
}

template <typename T, bool MainPml, bool AuxiliaryPml, bool Conductivity>
void launch_cylindrical_axis_t(const cylindrical_axis_launch &update,
                               const stream &execution_stream) {
  const size_t points = checked_points(update.region, "cylindrical axis");
  const unsigned int threads = 256;
  cylindrical_axis_kernel<T, MainPml, AuxiliaryPml, Conductivity>
      <<<checked_blocks(points, "cylindrical axis"), threads, 0,
         static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(update, points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA cylindrical axis update");
}

} // namespace

void launch_cylindrical_radial_prefix(const cylindrical_radial_prefix_launch &update,
                                      const stream &execution_stream) {
  if (!update.source || !update.scratch || update.source == update.scratch)
    throw std::invalid_argument("NVIDIA cylindrical radial-prefix operands are invalid");
  if (update.nz == std::numeric_limits<size_t>::max() || !update.row_stride ||
      update.row_stride != update.nz + 1)
    throw std::invalid_argument("NVIDIA cylindrical radial-prefix row stride is invalid");
  if (update.nr == std::numeric_limits<size_t>::max() ||
      update.nr + 1 > std::numeric_limits<size_t>::max() / update.row_stride)
    throw std::overflow_error("NVIDIA cylindrical radial-prefix extent overflow");
  const size_t elements = (update.nr + 1) * update.row_stride;
  if (update.source_elements < elements || update.scratch_elements < elements)
    throw std::out_of_range("NVIDIA cylindrical radial-prefix storage is undersized");
  if (!std::isfinite(update.ir0))
    throw std::invalid_argument("NVIDIA cylindrical radial-prefix coefficient is non-finite");
  const double zero_row = 0.5 - update.ir0;
  if (zero_row >= 1.0 && zero_row <= double(update.nr) && std::trunc(zero_row) == zero_row)
    throw std::invalid_argument("NVIDIA cylindrical radial-prefix denominator reaches zero");
  const size_t columns = update.nz + 1;
  const unsigned int threads = 256;
  if (update.precision == scalar_precision::f32)
    cylindrical_radial_prefix_kernel<float>
        <<<checked_blocks(columns, "cylindrical radial-prefix"), threads, 0,
           static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(update);
  else if (update.precision == scalar_precision::f64)
    cylindrical_radial_prefix_kernel<double>
        <<<checked_blocks(columns, "cylindrical radial-prefix"), threads, 0,
           static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(update);
  else
    throw std::invalid_argument("NVIDIA cylindrical radial-prefix precision is invalid");
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA cylindrical radial-prefix update");
}

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

void launch_cylindrical_m(const cylindrical_m_launch &update, const stream &execution_stream) {
  if (!update.target || !update.source || update.target == update.source ||
      update.radial_axis >= 3)
    throw std::invalid_argument("NVIDIA cylindrical m/r launch has invalid operands");
  const bool main_pml = update.pml.inverse != NULL;
  const bool auxiliary_pml = update.pml_u.inverse != NULL;
  const bool conductivity = update.conductivity_inverse != NULL;
  if (main_pml != (update.pml.sigma && update.pml.kappa) ||
      auxiliary_pml != (update.pml_u.sigma && update.pml_u.kappa))
    throw std::invalid_argument("NVIDIA cylindrical m/r PML profile is incomplete");
  if (auxiliary_pml != (update.target_u != NULL) ||
      (main_pml && conductivity) != (update.target_conductivity != NULL))
    throw std::invalid_argument("NVIDIA cylindrical m/r auxiliary state is incomplete");
  if (!std::isfinite(update.numerator))
    throw std::invalid_argument("NVIDIA cylindrical m/r coefficient is non-finite");
  const size_t radial_count = update.region.counts[update.radial_axis];
  if (radial_count && radial_count - 1 > size_t(std::numeric_limits<ptrdiff_t>::max() / 2))
    throw std::overflow_error("NVIDIA cylindrical m/r radial coordinate overflow");
  const ptrdiff_t radial_delta = radial_count ? ptrdiff_t(2 * (radial_count - 1)) : 0;
  if (ptrdiff_t(update.raw_radial_start) >
      std::numeric_limits<ptrdiff_t>::max() - radial_delta)
    throw std::overflow_error("NVIDIA cylindrical m/r radial coordinate overflow");
  const ptrdiff_t radial_end = ptrdiff_t(update.raw_radial_start) + radial_delta;
  const ptrdiff_t first = ptrdiff_t(update.raw_radial_start);
  if (radial_count && first <= 0 && radial_end >= 0 && (-first) % 2 == 0)
    throw std::invalid_argument("NVIDIA cylindrical m/r denominator reaches zero");
  const void *mutable_arrays[] = {update.target, update.target_u, update.target_conductivity};
  const void *input_arrays[] = {update.source,
                                update.conductivity_inverse,
                                update.pml.sigma,
                                update.pml.kappa,
                                update.pml.inverse,
                                update.pml_u.sigma,
                                update.pml_u.kappa,
                                update.pml_u.inverse};
  for (size_t i = 0; i < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++i) {
    if (!mutable_arrays[i]) continue;
    for (const void *input : input_arrays)
      if (input && mutable_arrays[i] == input)
        throw std::invalid_argument("NVIDIA cylindrical m/r aliases mutable and input state");
    for (size_t j = i + 1; j < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++j)
      if (mutable_arrays[j] && mutable_arrays[i] == mutable_arrays[j])
        throw std::invalid_argument("NVIDIA cylindrical m/r aliases mutable state");
  }
#define LAUNCH_CYL_M_VARIANT(T, MP, AP, C)                                                         \
  launch_cylindrical_m_t<T, MP, AP, C>(update, execution_stream)
#define DISPATCH_CYL_M(T)                                                                          \
  do {                                                                                             \
    const unsigned int key = unsigned(main_pml) | (unsigned(auxiliary_pml) << 1) |                 \
                             (unsigned(conductivity) << 2);                                        \
    switch (key) {                                                                                 \
      case 0: LAUNCH_CYL_M_VARIANT(T, false, false, false); break;                                 \
      case 1: LAUNCH_CYL_M_VARIANT(T, true, false, false); break;                                  \
      case 2: LAUNCH_CYL_M_VARIANT(T, false, true, false); break;                                  \
      case 3: LAUNCH_CYL_M_VARIANT(T, true, true, false); break;                                   \
      case 4: LAUNCH_CYL_M_VARIANT(T, false, false, true); break;                                  \
      case 5: LAUNCH_CYL_M_VARIANT(T, true, false, true); break;                                   \
      case 6: LAUNCH_CYL_M_VARIANT(T, false, true, true); break;                                   \
      case 7: LAUNCH_CYL_M_VARIANT(T, true, true, true); break;                                    \
    }                                                                                              \
  } while (0)
  if (update.precision == scalar_precision::f32)
    DISPATCH_CYL_M(float);
  else if (update.precision == scalar_precision::f64)
    DISPATCH_CYL_M(double);
  else
    throw std::invalid_argument("NVIDIA cylindrical m/r launch precision is invalid");
#undef DISPATCH_CYL_M
#undef LAUNCH_CYL_M_VARIANT
}

void launch_cylindrical_axis(const cylindrical_axis_launch &update,
                             const stream &execution_stream) {
  if (update.kind > 1 || !update.target || !update.source1 || update.target == update.source1 ||
      (update.source2 && update.target == update.source2))
    throw std::invalid_argument("NVIDIA cylindrical axis launch has invalid operands");
  if ((update.kind == 0) != (update.source2 == NULL))
    throw std::invalid_argument("NVIDIA cylindrical axis kind and source shape disagree");
  const bool main_pml = update.pml.inverse != NULL;
  const bool auxiliary_pml = update.pml_u.inverse != NULL;
  const bool conductivity = update.conductivity_inverse != NULL;
  if (main_pml != (update.pml.sigma && update.pml.kappa) ||
      auxiliary_pml != (update.pml_u.sigma && update.pml_u.kappa))
    throw std::invalid_argument("NVIDIA cylindrical axis PML profile is incomplete");
  if (auxiliary_pml != (update.target_u != NULL) ||
      conductivity != (update.conductivity != NULL) ||
      conductivity != (update.target_conductivity != NULL))
    throw std::invalid_argument("NVIDIA cylindrical axis auxiliary state is incomplete");
  if (!std::isfinite(update.scale) || !std::isfinite(update.source2_multiplier) ||
      !std::isfinite(update.dt))
    throw std::invalid_argument("NVIDIA cylindrical axis coefficient is non-finite");
  const void *mutable_arrays[] = {update.target, update.target_u, update.target_conductivity};
  const void *input_arrays[] = {update.source1,
                                update.source2,
                                update.conductivity,
                                update.conductivity_inverse,
                                update.pml.sigma,
                                update.pml.kappa,
                                update.pml.inverse,
                                update.pml_u.sigma,
                                update.pml_u.kappa,
                                update.pml_u.inverse};
  for (size_t i = 0; i < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++i) {
    if (!mutable_arrays[i]) continue;
    for (const void *input : input_arrays)
      if (input && mutable_arrays[i] == input)
        throw std::invalid_argument("NVIDIA cylindrical axis aliases mutable and input state");
    for (size_t j = i + 1; j < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++j)
      if (mutable_arrays[j] && mutable_arrays[i] == mutable_arrays[j])
        throw std::invalid_argument("NVIDIA cylindrical axis aliases mutable state");
  }
#define LAUNCH_CYL_AXIS_VARIANT(T, MP, AP, C)                                                      \
  launch_cylindrical_axis_t<T, MP, AP, C>(update, execution_stream)
#define DISPATCH_CYL_AXIS(T)                                                                       \
  do {                                                                                             \
    const unsigned int key = unsigned(main_pml) | (unsigned(auxiliary_pml) << 1) |                 \
                             (unsigned(conductivity) << 2);                                        \
    switch (key) {                                                                                 \
      case 0: LAUNCH_CYL_AXIS_VARIANT(T, false, false, false); break;                              \
      case 1: LAUNCH_CYL_AXIS_VARIANT(T, true, false, false); break;                               \
      case 2: LAUNCH_CYL_AXIS_VARIANT(T, false, true, false); break;                               \
      case 3: LAUNCH_CYL_AXIS_VARIANT(T, true, true, false); break;                                \
      case 4: LAUNCH_CYL_AXIS_VARIANT(T, false, false, true); break;                               \
      case 5: LAUNCH_CYL_AXIS_VARIANT(T, true, false, true); break;                                \
      case 6: LAUNCH_CYL_AXIS_VARIANT(T, false, true, true); break;                                \
      case 7: LAUNCH_CYL_AXIS_VARIANT(T, true, true, true); break;                                 \
    }                                                                                              \
  } while (0)
  if (update.precision == scalar_precision::f32)
    DISPATCH_CYL_AXIS(float);
  else if (update.precision == scalar_precision::f64)
    DISPATCH_CYL_AXIS(double);
  else
    throw std::invalid_argument("NVIDIA cylindrical axis launch precision is invalid");
#undef DISPATCH_CYL_AXIS
#undef LAUNCH_CYL_AXIS_VARIANT
}

} // namespace nvidia
} // namespace meep
