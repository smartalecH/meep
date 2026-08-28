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

__device__ ptrdiff_t profile_index(const pml_profile_launch &profile, size_t i0, size_t i1,
                                   size_t i2) {
  return profile.base + ptrdiff_t(i0) * profile.strides[0] +
         ptrdiff_t(i1) * profile.strides[1] + ptrdiff_t(i2) * profile.strides[2];
}

template <typename T, bool MainPml, bool AuxiliaryPml, bool Conductivity>
__global__ void curl_kernel(curl_launch update, size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  size_t i0, i1, i2;
  region_coordinates(update.region, linear, i0, i1, i2);
  const ptrdiff_t i = region_index(update.region, i0, i1, i2);
  T *target = static_cast<T *>(update.target);
  const T *plus_source = static_cast<const T *>(update.plus_source);
  const T *minus_source = static_cast<const T *>(update.minus_source);
  T curl = T(0);
  if (plus_source) curl += plus_source[i + update.plus_stride] - plus_source[i];
  if (minus_source) curl += minus_source[i] - minus_source[i + update.minus_stride];
  const T delta = T(update.dtdx) * curl;

  const T *conductivity = static_cast<const T *>(update.conductivity);
  const T *conductivity_inverse = static_cast<const T *>(update.conductivity_inverse);
  T *target_conductivity = static_cast<T *>(update.target_conductivity);
  T *target_u = static_cast<T *>(update.target_u);

  if (!MainPml && !AuxiliaryPml) {
    if (Conductivity)
      target[i] =
          ((T(1) - T(0.5 * update.dt) * conductivity[i]) * target[i] - delta) *
          conductivity_inverse[i];
    else
      target[i] -= delta;
    return;
  }

  T intermediate_delta;
  T previous_u = T(0);
  if (MainPml) {
    const ptrdiff_t k = profile_index(update.pml, i0, i1, i2);
    const T *sigma = static_cast<const T *>(update.pml.sigma);
    const T *kappa = static_cast<const T *>(update.pml.kappa);
    const T *inverse = static_cast<const T *>(update.pml.inverse);
    if (Conductivity) {
      const T previous = target_conductivity[i];
      target_conductivity[i] =
          ((T(1) - T(0.5 * update.dt) * conductivity[i]) * previous - delta) *
          conductivity_inverse[i];
      intermediate_delta = target_conductivity[i] - previous;
    }
    else
      intermediate_delta = -delta;

    if (AuxiliaryPml) {
      previous_u = target_u[i];
      target_u[i] =
          ((kappa[k] - sigma[k]) * target_u[i] + intermediate_delta) * inverse[k];
    }
    else {
      target[i] = ((kappa[k] - sigma[k]) * target[i] + intermediate_delta) * inverse[k];
      return;
    }
  }
  else {
    previous_u = target_u[i];
    if (Conductivity)
      target_u[i] =
          ((T(1) - T(0.5 * update.dt) * conductivity[i]) * previous_u - delta) *
          conductivity_inverse[i];
    else
      target_u[i] -= delta;
  }

  const ptrdiff_t ku = profile_index(update.pml_u, i0, i1, i2);
  const T *sigma_u = static_cast<const T *>(update.pml_u.sigma);
  const T *kappa_u = static_cast<const T *>(update.pml_u.kappa);
  const T *inverse_u = static_cast<const T *>(update.pml_u.inverse);
  target[i] = inverse_u[ku] *
              (((kappa_u[ku] - sigma_u[ku]) * target[i] + target_u[i]) - previous_u);
}

template <typename T>
__device__ T offdiagonal_value(const T *coefficient, const T *field, ptrdiff_t i,
                               ptrdiff_t primary_stride, ptrdiff_t cross_stride) {
  return T(0.25) *
         ((field[i] + field[i - cross_stride]) * coefficient[i] +
          (field[i + primary_stride] + field[(i + primary_stride) - cross_stride]) *
              coefficient[i + primary_stride]);
}

template <typename T, bool Pml, unsigned int Offdiagonals>
__global__ void constitutive_kernel(constitutive_launch update, size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  size_t i0, i1, i2;
  region_coordinates(update.region, linear, i0, i1, i2);
  const ptrdiff_t i = region_index(update.region, i0, i1, i2);
  T value = static_cast<const T *>(update.primary)[i];
  if (update.diagonal) value *= static_cast<const T *>(update.diagonal)[i];
  if (Offdiagonals >= 1)
    value += offdiagonal_value(static_cast<const T *>(update.offdiagonal1),
                               static_cast<const T *>(update.cross1), i, update.primary_stride,
                               update.cross1_stride);
  if (Offdiagonals >= 2)
    value += offdiagonal_value(static_cast<const T *>(update.offdiagonal2),
                               static_cast<const T *>(update.cross2), i, update.primary_stride,
                               update.cross2_stride);
  T *target = static_cast<T *>(update.target);
  if (!Pml) {
    target[i] = value;
    return;
  }
  T *target_w = static_cast<T *>(update.target_w);
  const T previous = target_w[i];
  target_w[i] = value;
  const ptrdiff_t k = profile_index(update.pml, i0, i1, i2);
  const T *sigma = static_cast<const T *>(update.pml.sigma);
  const T *kappa = static_cast<const T *>(update.pml.kappa);
  target[i] += (kappa[k] + sigma[k]) * value - (kappa[k] - sigma[k]) * previous;
}

template <typename T> __global__ void zero_kernel(zero_launch update, size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  size_t i0, i1, i2;
  region_coordinates(update.region, linear, i0, i1, i2);
  static_cast<T *>(update.target)[region_index(update.region, i0, i1, i2)] = T(0);
}

template <typename T>
__global__ void halo_gather_kernel(const halo_gather_entry *entries, size_t first, size_t count,
                                   T *buffer) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= count) return;
  const halo_gather_entry entry = entries[first + linear];
  buffer[entry.buffer_index] = static_cast<const T *>(entry.source)[entry.source_index];
}

template <typename T>
__global__ void halo_scatter_kernel(const halo_scatter_entry *entries, size_t first, size_t count,
                                    const T *buffer) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= count) return;
  const halo_scatter_entry entry = entries[first + linear];
  const T input_real = buffer[entry.buffer_index];
  const T phase_real = T(entry.phase_real);
  if (!entry.target_imag) {
    static_cast<T *>(entry.target_real)[entry.target_real_index] = phase_real * input_real;
    return;
  }
  const T input_imag = buffer[entry.buffer_index + 1];
  const T phase_imag = T(entry.phase_imag);
  static_cast<T *>(entry.target_real)[entry.target_real_index] =
      phase_real * input_real - phase_imag * input_imag;
  static_cast<T *>(entry.target_imag)[entry.target_imag_index] =
      phase_real * input_imag + phase_imag * input_real;
}

void launch_geometry(const flat_region &region, unsigned int &blocks, unsigned int &threads) {
  const size_t points = checked_points(region);
  threads = 256;
  const size_t block_count = (points + threads - 1) / threads;
  if (block_count > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA step launch grid overflow");
  blocks = static_cast<unsigned int>(block_count);
}

template <typename T, bool MainPml, bool AuxiliaryPml, bool Conductivity>
void launch_curl_t(const curl_launch &update, const stream &execution_stream) {
  const size_t points = checked_points(update.region);
  unsigned int blocks = 0, threads = 0;
  launch_geometry(update.region, blocks, threads);
  curl_kernel<T, MainPml, AuxiliaryPml, Conductivity>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          update, points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA curl");
}

template <typename T, bool Pml, unsigned int Offdiagonals>
void launch_constitutive_t(const constitutive_launch &update, const stream &execution_stream) {
  const size_t points = checked_points(update.region);
  unsigned int blocks = 0, threads = 0;
  launch_geometry(update.region, blocks, threads);
  constitutive_kernel<T, Pml, Offdiagonals>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          update, points);
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

void linear_launch_geometry(size_t count, unsigned int &blocks, unsigned int &threads) {
  if (!count) throw std::invalid_argument("NVIDIA halo launch is empty");
  threads = 256;
  const size_t block_count = (count + threads - 1) / threads;
  if (block_count > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA halo launch grid overflow");
  blocks = static_cast<unsigned int>(block_count);
}

template <typename T>
void launch_halo_gather_t(const halo_launch &launch, const void *device_entries,
                          void *device_buffer, const stream &execution_stream) {
  unsigned int blocks = 0, threads = 0;
  linear_launch_geometry(launch.count, blocks, threads);
  halo_gather_kernel<T><<<blocks, threads, 0,
                          static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
      static_cast<const halo_gather_entry *>(device_entries), launch.first, launch.count,
      static_cast<T *>(device_buffer));
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA halo gather");
}

template <typename T>
void launch_halo_scatter_t(const halo_launch &launch, const void *device_entries,
                           const void *device_buffer, const stream &execution_stream) {
  unsigned int blocks = 0, threads = 0;
  linear_launch_geometry(launch.count, blocks, threads);
  halo_scatter_kernel<T><<<blocks, threads, 0,
                           static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
      static_cast<const halo_scatter_entry *>(device_entries), launch.first, launch.count,
      static_cast<const T *>(device_buffer));
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA halo scatter");
}

} // namespace

void launch_curl(const curl_launch &update, const stream &execution_stream) {
  if (!update.target || (!update.plus_source && !update.minus_source))
    throw std::invalid_argument("NVIDIA curl launch has incomplete operands");
  const bool main_pml = update.pml.sigma != NULL;
  const bool auxiliary_pml = update.pml_u.sigma != NULL;
  const bool conductivity = update.conductivity != NULL;
  if (main_pml && (!update.pml.kappa || !update.pml.inverse))
    throw std::invalid_argument("NVIDIA curl main PML profile is incomplete");
  if (auxiliary_pml &&
      (!update.target_u || !update.pml_u.kappa || !update.pml_u.inverse))
    throw std::invalid_argument("NVIDIA curl auxiliary PML profile is incomplete");
  if (conductivity &&
      (!update.conductivity_inverse || (main_pml && !update.target_conductivity)))
    throw std::invalid_argument("NVIDIA curl conductivity state is incomplete");

#define LAUNCH_CURL_VARIANT(T, MP, AP, C) launch_curl_t<T, MP, AP, C>(update, execution_stream)
#define DISPATCH_CURL(T)                                                                            \
  do {                                                                                              \
    const unsigned int key = unsigned(main_pml) | (unsigned(auxiliary_pml) << 1) |                  \
                             (unsigned(conductivity) << 2);                                         \
    switch (key) {                                                                                  \
      case 0: LAUNCH_CURL_VARIANT(T, false, false, false); break;                                   \
      case 1: LAUNCH_CURL_VARIANT(T, true, false, false); break;                                    \
      case 2: LAUNCH_CURL_VARIANT(T, false, true, false); break;                                    \
      case 3: LAUNCH_CURL_VARIANT(T, true, true, false); break;                                     \
      case 4: LAUNCH_CURL_VARIANT(T, false, false, true); break;                                    \
      case 5: LAUNCH_CURL_VARIANT(T, true, false, true); break;                                     \
      case 6: LAUNCH_CURL_VARIANT(T, false, true, true); break;                                     \
      case 7: LAUNCH_CURL_VARIANT(T, true, true, true); break;                                      \
    }                                                                                               \
  } while (0)
  if (update.precision == scalar_precision::f32) {
    DISPATCH_CURL(float);
  }
  else {
    DISPATCH_CURL(double);
  }
#undef DISPATCH_CURL
#undef LAUNCH_CURL_VARIANT
}

void launch_constitutive(const constitutive_launch &update, const stream &execution_stream) {
  if (!update.target || !update.primary)
    throw std::invalid_argument("NVIDIA constitutive launch has incomplete operands");
  const bool pml = update.pml.sigma != NULL;
  if (update.offdiagonal2 && !update.offdiagonal1)
    throw std::invalid_argument(
        "NVIDIA constitutive launch has a second off-diagonal without first");
  const unsigned int offdiagonals = update.offdiagonal2 ? 2u : update.offdiagonal1 ? 1u : 0u;
  if ((offdiagonals >= 1 && (!update.cross1 || !update.diagonal)) ||
      (offdiagonals >= 2 && !update.cross2))
    throw std::invalid_argument("NVIDIA constitutive anisotropic state is incomplete");
  if (pml && (!update.target_w || !update.pml.kappa))
    throw std::invalid_argument("NVIDIA constitutive PML state is incomplete");
#define LAUNCH_CONSTITUTIVE(T, P, O) launch_constitutive_t<T, P, O>(update, execution_stream)
#define DISPATCH_CONSTITUTIVE(T)                                                                  \
  do {                                                                                            \
    if (pml) {                                                                                    \
      if (offdiagonals == 2)                                                                      \
        LAUNCH_CONSTITUTIVE(T, true, 2);                                                          \
      else if (offdiagonals == 1)                                                                 \
        LAUNCH_CONSTITUTIVE(T, true, 1);                                                          \
      else                                                                                        \
        LAUNCH_CONSTITUTIVE(T, true, 0);                                                          \
    }                                                                                             \
    else {                                                                                        \
      if (offdiagonals == 2)                                                                      \
        LAUNCH_CONSTITUTIVE(T, false, 2);                                                         \
      else if (offdiagonals == 1)                                                                 \
        LAUNCH_CONSTITUTIVE(T, false, 1);                                                         \
      else                                                                                        \
        LAUNCH_CONSTITUTIVE(T, false, 0);                                                         \
    }                                                                                             \
  } while (0)
  if (update.precision == scalar_precision::f32)
    DISPATCH_CONSTITUTIVE(float);
  else
    DISPATCH_CONSTITUTIVE(double);
#undef DISPATCH_CONSTITUTIVE
#undef LAUNCH_CONSTITUTIVE
}

void launch_zero(const zero_launch &update, const stream &execution_stream) {
  if (!update.target) throw std::invalid_argument("NVIDIA zero launch has no target");
  if (update.precision == scalar_precision::f32)
    launch_zero_t<float>(update, execution_stream);
  else
    launch_zero_t<double>(update, execution_stream);
}

void launch_halo_gather(const halo_launch &launch, const void *device_entries,
                        void *device_buffer, const stream &execution_stream) {
  if (!device_entries || !device_buffer)
    throw std::invalid_argument("NVIDIA halo gather has incomplete storage");
  if (launch.precision == scalar_precision::f32)
    launch_halo_gather_t<float>(launch, device_entries, device_buffer, execution_stream);
  else
    launch_halo_gather_t<double>(launch, device_entries, device_buffer, execution_stream);
}

void launch_halo_scatter(const halo_launch &launch, const void *device_entries,
                         const void *device_buffer, const stream &execution_stream) {
  if (!device_entries || !device_buffer)
    throw std::invalid_argument("NVIDIA halo scatter has incomplete storage");
  if (launch.precision == scalar_precision::f32)
    launch_halo_scatter_t<float>(launch, device_entries, device_buffer, execution_stream);
  else
    launch_halo_scatter_t<double>(launch, device_entries, device_buffer, execution_stream);
}

} // namespace nvidia
} // namespace meep
