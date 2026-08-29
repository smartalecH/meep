/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_polarization.hpp"

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
    if (!region.counts[axis]) throw std::invalid_argument("NVIDIA polarization region is empty");
    if (region.counts[axis] > std::numeric_limits<size_t>::max() / points)
      throw std::overflow_error("NVIDIA polarization region size overflow");
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
__device__ T offdiagonal_value(const T *coefficient, const T *field, ptrdiff_t i,
                               ptrdiff_t primary_stride, ptrdiff_t cross_stride) {
  return T(0.25) *
         ((field[i] + field[i - cross_stride]) * coefficient[i] +
          (field[i + primary_stride] + field[i + primary_stride - cross_stride]) *
              coefficient[i + primary_stride]);
}

template <typename T, unsigned int Offdiagonals, bool Drude>
__global__ void polarization_update_kernel(polarization_update_launch update, size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  const ptrdiff_t i = region_index(update.region, linear);
  T *p = static_cast<T *>(update.p);
  T *p_prev = static_cast<T *>(update.p_prev);
  const T *w = static_cast<const T *>(update.primary_w);
  const T *sigma = static_cast<const T *>(update.diagonal_sigma);
  if (Offdiagonals && sigma[i] == T(0)) return;

  T forcing = sigma[i] * w[i];
  if (Offdiagonals >= 1)
    forcing += offdiagonal_value(static_cast<const T *>(update.offdiagonal_sigma1),
                                 static_cast<const T *>(update.cross_w1), i,
                                 update.primary_stride, update.cross_stride1);
  if (Offdiagonals >= 2)
    forcing += offdiagonal_value(static_cast<const T *>(update.offdiagonal_sigma2),
                                 static_cast<const T *>(update.cross_w2), i,
                                 update.primary_stride, update.cross_stride2);
  const T current = p[i];
  const T denominator = Drude ? T(0) : T(update.omega0dtsqr_denom);
  p[i] = T(update.gamma1inv) *
         (current * (T(2) - denominator) - T(update.gamma1) * p_prev[i] +
          T(update.omega0dtsqr) * forcing);
  p_prev[i] = current;
}

template <typename T>
__global__ void polarization_subtract_kernel(polarization_subtract_launch update) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < update.elements)
    static_cast<T *>(update.target)[i] -= static_cast<const T *>(update.p)[i];
}

void launch_geometry(size_t points, unsigned int &blocks, unsigned int &threads) {
  if (!points) throw std::invalid_argument("NVIDIA polarization launch is empty");
  threads = 256;
  const size_t block_count = (points + threads - 1) / threads;
  if (block_count > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA polarization launch grid overflow");
  blocks = static_cast<unsigned int>(block_count);
}

template <typename T, unsigned int Offdiagonals, bool Drude>
void launch_update_t(const polarization_update_launch &update, const stream &execution_stream) {
  const size_t points = checked_points(update.region);
  unsigned int blocks = 0, threads = 0;
  launch_geometry(points, blocks, threads);
  polarization_update_kernel<T, Offdiagonals, Drude>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          update, points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA polarization update");
}

template <typename T>
void launch_subtract_t(const polarization_subtract_launch &update,
                       const stream &execution_stream) {
  unsigned int blocks = 0, threads = 0;
  launch_geometry(update.elements, blocks, threads);
  polarization_subtract_kernel<T>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(update);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA polarization subtraction");
}

} // namespace

void launch_polarization_update(const polarization_update_launch &update,
                                const stream &execution_stream) {
  if (!update.p || !update.p_prev || !update.primary_w || !update.diagonal_sigma)
    throw std::invalid_argument("NVIDIA polarization update has incomplete operands");
  if (update.p == update.p_prev)
    throw std::invalid_argument("NVIDIA polarization P and P_prev alias");
  if (update.offdiagonals > 2 ||
      (update.offdiagonals >= 1 && (!update.cross_w1 || !update.offdiagonal_sigma1)) ||
      (update.offdiagonals >= 2 && (!update.cross_w2 || !update.offdiagonal_sigma2)))
    throw std::invalid_argument("NVIDIA polarization anisotropic state is incomplete");

#define DISPATCH_UPDATE(T, O)                                                                      \
  do {                                                                                             \
    if (update.drude)                                                                              \
      launch_update_t<T, O, true>(update, execution_stream);                                      \
    else                                                                                           \
      launch_update_t<T, O, false>(update, execution_stream);                                     \
  } while (0)
#define DISPATCH_PRECISION(T)                                                                      \
  do {                                                                                             \
    if (update.offdiagonals == 2) DISPATCH_UPDATE(T, 2);                                           \
    else if (update.offdiagonals == 1) DISPATCH_UPDATE(T, 1);                                      \
    else DISPATCH_UPDATE(T, 0);                                                                    \
  } while (0)
  if (update.precision == scalar_precision::f32)
    DISPATCH_PRECISION(float);
  else if (update.precision == scalar_precision::f64)
    DISPATCH_PRECISION(double);
  else
    throw std::invalid_argument("NVIDIA polarization update has invalid precision");
#undef DISPATCH_PRECISION
#undef DISPATCH_UPDATE
}

void launch_polarization_subtract(const polarization_subtract_launch &update,
                                  const stream &execution_stream) {
  if (!update.target || !update.p || update.target == update.p || !update.elements)
    throw std::invalid_argument("NVIDIA polarization subtraction has invalid operands");
  if (update.precision == scalar_precision::f32)
    launch_subtract_t<float>(update, execution_stream);
  else if (update.precision == scalar_precision::f64)
    launch_subtract_t<double>(update, execution_stream);
  else
    throw std::invalid_argument("NVIDIA polarization subtraction has invalid precision");
}

} // namespace nvidia
} // namespace meep
