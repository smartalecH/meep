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
__device__ double centered_value(const T *field, ptrdiff_t i, ptrdiff_t primary_stride,
                                 ptrdiff_t cross_stride) {
  return 0.25 * (field[i] + field[i - cross_stride] + field[i + primary_stride] +
                 field[i + primary_stride - cross_stride]);
}

template <typename T, gyrotropic_kernel_model Model>
__global__ void gyrotropic_update_kernel(gyrotropic_update_launch update, size_t points) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= points) return;
  const ptrdiff_t i = region_index(update.region, linear);
  T *p0 = static_cast<T *>(update.p[0]);
  T *p1 = static_cast<T *>(update.p[1]);
  T *p2 = static_cast<T *>(update.p[2]);
  T *pp0 = static_cast<T *>(update.p_prev[0]);
  T *pp1 = static_cast<T *>(update.p_prev[1]);
  T *pp2 = static_cast<T *>(update.p_prev[2]);
  const T old0 = p0[i], old1 = p1[i], old2 = p2[i];
  const T prev0 = pp0[i], prev1 = pp1[i], prev2 = pp2[i];
  const T *w0 = static_cast<const T *>(update.w[0]);
  const T *w1 = static_cast<const T *>(update.w[1]);
  const T *w2 = static_cast<const T *>(update.w[2]);
  const T sigma = static_cast<const T *>(update.sigma)[i];
  const double cross1 =
      w1 ? centered_value(w1, i, update.primary_stride, update.cross_stride1) : 0.0;
  const double cross2 =
      w2 ? centered_value(w2, i, update.primary_stride, update.cross_stride2) : 0.0;
  T r0, r1, r2;

  if (Model == gyrotropic_kernel_model::saturated) {
    const T q0 = -T(update.omega) * old0 + 0.5 * T(update.alpha) * prev0 +
                 T(update.dt2pi) * sigma * w0[i];
    const T q1 = -T(update.omega) * old1 + 0.5 * T(update.alpha) * prev1 +
                 T(update.dt2pi) * sigma * cross1;
    const T q2 = -T(update.omega) * old2 + 0.5 * T(update.alpha) * prev2 +
                 T(update.dt2pi) * sigma * cross2;
    r0 = 0.5 * prev0 - T(update.gamma) * old0 + T(update.gyro[0][1]) * q1 +
         T(update.gyro[0][2]) * q2;
    r1 = 0.5 * prev1 - T(update.gamma) * old1 + T(update.gyro[1][2]) * q2 +
         T(update.gyro[1][0]) * q0;
    r2 = 0.5 * prev2 - T(update.gamma) * old2 + T(update.gyro[2][0]) * q0 +
         T(update.gyro[2][1]) * q1;
  }
  else {
    r0 = T(update.diagonal) * old0 - T(update.gamma1) * prev0 +
         T(update.omega0dtsqr) * sigma * w0[i] - T(update.pt) * T(update.gyro[0][1]) * prev1 -
         T(update.pt) * T(update.gyro[0][2]) * prev2;
    r1 = T(update.diagonal) * old1 - T(update.gamma1) * prev1 +
         T(update.omega0dtsqr) * sigma * cross1 - T(update.pt) * T(update.gyro[1][0]) * prev0 -
         T(update.pt) * T(update.gyro[1][2]) * prev2;
    r2 = T(update.diagonal) * old2 - T(update.gamma1) * prev2 +
         T(update.omega0dtsqr) * sigma * cross2 - T(update.pt) * T(update.gyro[2][1]) * prev1 -
         T(update.pt) * T(update.gyro[2][0]) * prev0;
  }

  pp0[i] = old0;
  pp1[i] = old1;
  pp2[i] = old2;
  p0[i] = T(update.inverse[0][0]) * r0 + T(update.inverse[0][1]) * r1 +
          T(update.inverse[0][2]) * r2;
  p1[i] = T(update.inverse[1][0]) * r0 + T(update.inverse[1][1]) * r1 +
          T(update.inverse[1][2]) * r2;
  p2[i] = T(update.inverse[2][0]) * r0 + T(update.inverse[2][1]) * r1 +
          T(update.inverse[2][2]) * r2;
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

template <typename T, gyrotropic_kernel_model Model>
void launch_gyrotropic_t(const gyrotropic_update_launch &update,
                         const stream &execution_stream) {
  const size_t points = checked_points(update.region);
  unsigned int blocks = 0, threads = 0;
  launch_geometry(points, blocks, threads);
  gyrotropic_update_kernel<T, Model>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          update, points);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA gyrotropic update");
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

void launch_gyrotropic_update(const gyrotropic_update_launch &update,
                              const stream &execution_stream) {
  const void *state[6] = {update.p[0],      update.p_prev[0], update.p[1],
                          update.p_prev[1], update.p[2],      update.p_prev[2]};
  for (int i = 0; i < 6; ++i) {
    if (!state[i]) throw std::invalid_argument("NVIDIA gyrotropic update has incomplete state");
    for (int j = i + 1; j < 6; ++j)
      if (state[i] == state[j])
        throw std::invalid_argument("NVIDIA gyrotropic state arrays alias");
  }
  if (!update.w[0] || !update.sigma)
    throw std::invalid_argument("NVIDIA gyrotropic update has incomplete operands");

#define DISPATCH_GYRO(T)                                                                           \
  do {                                                                                             \
    switch (update.model) {                                                                        \
      case gyrotropic_kernel_model::lorentzian:                                                    \
        launch_gyrotropic_t<T, gyrotropic_kernel_model::lorentzian>(update, execution_stream);     \
        break;                                                                                     \
      case gyrotropic_kernel_model::drude:                                                         \
        launch_gyrotropic_t<T, gyrotropic_kernel_model::drude>(update, execution_stream);          \
        break;                                                                                     \
      case gyrotropic_kernel_model::saturated:                                                     \
        launch_gyrotropic_t<T, gyrotropic_kernel_model::saturated>(update, execution_stream);      \
        break;                                                                                     \
      default: throw std::invalid_argument("NVIDIA gyrotropic update has invalid model");         \
    }                                                                                              \
  } while (0)
  if (update.precision == scalar_precision::f32)
    DISPATCH_GYRO(float);
  else if (update.precision == scalar_precision::f64)
    DISPATCH_GYRO(double);
  else
    throw std::invalid_argument("NVIDIA gyrotropic update has invalid precision");
#undef DISPATCH_GYRO
}

void launch_polarization_update(const compiled_polarization_update &update,
                                const stream &execution_stream) {
  switch (update.kind) {
    case compiled_polarization_update::kind_type::lorentzian:
      launch_polarization_update(update.lorentzian, execution_stream);
      break;
    case compiled_polarization_update::kind_type::gyrotropic:
      launch_gyrotropic_update(update.gyrotropic, execution_stream);
      break;
    default: throw std::invalid_argument("NVIDIA polarization update has invalid kind");
  }
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
