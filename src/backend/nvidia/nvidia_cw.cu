/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_cw.hpp"

#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace meep {
namespace nvidia {
namespace {

const unsigned int cw_threads = 256;

void check_cuda(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) return;
  throw runtime_error(operation, static_cast<int>(result), cudaGetErrorName(result),
                      cudaGetErrorString(result));
}

unsigned int launch_blocks(size_t elements, const char *what) {
  if (!elements) throw std::invalid_argument(std::string(what) + " has no elements");
  const size_t blocks = 1 + (elements - 1) / cw_threads;
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error(std::string(what) + " launch grid overflows");
  return unsigned(blocks);
}

size_t region_elements(const flat_region &region, const char *what) {
  if (region.base > size_t(std::numeric_limits<ptrdiff_t>::max()))
    throw std::overflow_error(std::string(what) + " region base overflows");
  size_t result = 1;
  size_t maximum_index = region.base;
  for (int axis = 0; axis < 3; ++axis) {
    if (!region.counts[axis])
      throw std::invalid_argument(std::string(what) + " has an empty region axis");
    if (region.strides[axis] < 0)
      throw std::invalid_argument(std::string(what) + " has a negative region stride");
    if (region.counts[axis] > 1 && region.strides[axis] == 0)
      throw std::invalid_argument(std::string(what) + " has a zero active region stride");
    if (result > std::numeric_limits<size_t>::max() / region.counts[axis])
      throw std::overflow_error(std::string(what) + " region size overflows");
    result *= region.counts[axis];
    const size_t stride = size_t(region.strides[axis]);
    const size_t steps = region.counts[axis] - 1;
    if (steps && stride > std::numeric_limits<size_t>::max() / steps)
      throw std::overflow_error(std::string(what) + " region stride overflows");
    const size_t extent = steps * stride;
    if (extent > size_t(std::numeric_limits<ptrdiff_t>::max()) ||
        maximum_index > size_t(std::numeric_limits<ptrdiff_t>::max()) - extent)
      throw std::overflow_error(std::string(what) + " region index overflows");
    maximum_index += extent;
  }
  return result;
}

void validate_row(const cw_state_row_launch &launch, const void *vector, size_t real_elements,
                  const char *what) {
  if (!launch.real_values || !launch.imaginary_values || !vector)
    throw std::invalid_argument(std::string(what) + " has a null array");
  if (launch.real_values == launch.imaginary_values)
    throw std::invalid_argument(std::string(what) + " aliases real and imaginary storage");
  if (!launch.complex_count || region_elements(launch.region, what) != launch.complex_count)
    throw std::invalid_argument(std::string(what) + " has an inconsistent row extent");
  if (launch.complex_offset > std::numeric_limits<size_t>::max() - launch.complex_count)
    throw std::overflow_error(std::string(what) + " vector extent overflows");
  const size_t complex_end = launch.complex_offset + launch.complex_count;
  if (complex_end > std::numeric_limits<size_t>::max() / 2 || 2 * complex_end > real_elements)
    throw std::out_of_range(std::string(what) + " exceeds the real-vector extent");
  if (vector == launch.real_values || vector == launch.imaginary_values)
    throw std::invalid_argument(std::string(what) + " aliases field and vector storage");
}

__device__ ptrdiff_t row_index(const flat_region &region, size_t linear) {
  const size_t i2 = linear % region.counts[2];
  linear /= region.counts[2];
  const size_t i1 = linear % region.counts[1];
  const size_t i0 = linear / region.counts[1];
  return ptrdiff_t(region.base) + ptrdiff_t(i0) * region.strides[0] +
         ptrdiff_t(i1) * region.strides[1] + ptrdiff_t(i2) * region.strides[2];
}

template <typename T>
__global__ void pack_kernel(cw_state_row_launch launch, T *vector) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= launch.complex_count) return;
  const ptrdiff_t source = row_index(launch.region, i);
  const size_t destination = 2 * (launch.complex_offset + i);
  vector[destination] = static_cast<const T *>(launch.real_values)[source];
  vector[destination + 1] = static_cast<const T *>(launch.imaginary_values)[source];
}

template <typename T>
__global__ void unpack_kernel(cw_state_row_launch launch, const T *vector) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= launch.complex_count) return;
  const ptrdiff_t destination = row_index(launch.region, i);
  const size_t source = 2 * (launch.complex_offset + i);
  static_cast<T *>(launch.real_values)[destination] = vector[source];
  static_cast<T *>(launch.imaginary_values)[destination] = vector[source + 1];
}

template <typename T> __global__ void zero_kernel(T *values, size_t elements) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < elements) values[i] = T(0);
}

template <typename T>
__device__ void apply_cw_source_point(const cw_source_batch_launch &source,
                                      const source_point &point, const source_scalar &scalar) {
  double value_real = point.amplitude_real * scalar.current_real -
                      point.amplitude_imag * scalar.current_imag;
  double value_imag = point.amplitude_real * scalar.current_imag +
                      point.amplitude_imag * scalar.current_real;
  value_real *= source.dt;
  value_imag *= source.dt;
  if (source.conductivity_inverse) {
    const double condinv =
        double(static_cast<const T *>(source.conductivity_inverse)[point.index]);
    value_real *= condinv;
    value_imag *= condinv;
  }
  T *target_real = static_cast<T *>(source.target_real);
  T *target_imag = static_cast<T *>(source.target_imag);
  target_real[point.index] = T(double(target_real[point.index]) - value_real);
  target_imag[point.index] = T(double(target_imag[point.index]) - value_imag);
}

template <typename T>
__global__ void cw_source_kernel(cw_source_batch_launch source, const source_scalar *scalars) {
  const source_scalar scalar = scalars[source.scalar_slot];
  for (size_t point = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       point < source.point_count; point += size_t(blockDim.x) * gridDim.x)
    apply_cw_source_point<T>(source, source.points[point], scalar);
}

template <typename T>
__global__ void ordered_cw_source_kernel(cw_source_batch_launch source,
                                         const source_scalar *scalars) {
  if (blockIdx.x || threadIdx.x) return;
  const source_scalar scalar = scalars[source.scalar_slot];
  for (size_t point = 0; point < source.point_count; ++point)
    apply_cw_source_point<T>(source, source.points[point], scalar);
}

template <typename T, cw_vector_operation Operation>
__global__ void vector_kernel(cw_vector_launch launch) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= launch.elements) return;
  const T x = static_cast<const T *>(launch.x)[i];
  T result = x;
  if (Operation == cw_vector_operation::subtract_field)
    result = T(x - static_cast<const T *>(launch.y)[i]);
  else if (Operation == cw_vector_operation::scale_field_coefficient)
    result = T(x * T(launch.coefficient));
  else if (Operation == cw_vector_operation::scale_f64_coefficient)
    result = T(double(x) * launch.coefficient);
  else if (Operation == cw_vector_operation::linear_f64_coefficient)
    result = T(double(x) + launch.coefficient * double(static_cast<const T *>(launch.y)[i]));
  static_cast<T *>(launch.output)[i] = result;
}

template <typename T> __global__ void operator_kernel(cw_operator_launch launch) {
  const size_t complex_i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (2 * complex_i >= launch.real_elements) return;
  const T *stepped = static_cast<const T *>(launch.stepped);
  const T *input = static_cast<const T *>(launch.input);
  T *output = static_cast<T *>(launch.output);
  const size_t r = 2 * complex_i;
  const T dt_inverse = T(launch.dt_inverse);
  const T omega_real = T(launch.iomega_real);
  const T omega_imaginary = T(launch.iomega_imaginary);
  const T xr = input[r], xi = input[r + 1];
  output[r] = T(T(stepped[r] - xr) * dt_inverse + T(omega_real * xr - omega_imaginary * xi));
  output[r + 1] =
      T(T(stepped[r + 1] - xi) * dt_inverse + T(omega_real * xi + omega_imaginary * xr));
}

template <typename T, int Operation>
__global__ void reduction_partials_kernel(cw_reduction_launch launch) {
  __shared__ double scratch[cw_threads];
  double value = Operation == 1 ? 0.0 : 0.0;
  const T *x = static_cast<const T *>(launch.x);
  const T *y = static_cast<const T *>(launch.y);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < launch.elements;
       i += size_t(launch.blocks) * blockDim.x) {
    const double xv = double(x[i]);
    if (Operation == 0) value += double(T(x[i] * y[i]));
    else if (Operation == 1)
      value = isfinite(xv) ? fmax(value, fabs(xv)) : CUDART_INF;
    else {
      const double scaled = launch.scale * xv;
      value += scaled * scaled;
    }
  }
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int stride = cw_threads / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      scratch[threadIdx.x] = Operation == 1
                                 ? fmax(scratch[threadIdx.x], scratch[threadIdx.x + stride])
                                 : scratch[threadIdx.x] + scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) static_cast<double *>(launch.partials)[blockIdx.x] = scratch[0];
}

template <int Operation>
__global__ void reduction_final_kernel(const double *partials, double *result, size_t count) {
  __shared__ double scratch[cw_threads];
  double value = 0.0;
  for (size_t i = threadIdx.x; i < count; i += blockDim.x)
    value = Operation == 1 ? fmax(value, partials[i]) : value + partials[i];
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int stride = cw_threads / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      scratch[threadIdx.x] = Operation == 1
                                 ? fmax(scratch[threadIdx.x], scratch[threadIdx.x + stride])
                                 : scratch[threadIdx.x] + scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) result[0] = scratch[0];
}

template <typename T, int Operation>
void launch_reduction_typed(const cw_reduction_launch &launch, const stream &stream,
                            const char *what) {
  if (!launch.x || !launch.partials || !launch.result)
    throw std::invalid_argument(std::string(what) + " has a null array");
  if (Operation == 0 && !launch.y)
    throw std::invalid_argument("NVIDIA solve_cw dot has a null right operand");
  if (Operation != 0 && launch.y)
    throw std::invalid_argument(std::string(what) + " has an unexpected right operand");
  if (launch.partials == launch.result || launch.x == launch.partials ||
      launch.x == launch.result || launch.y == launch.partials || launch.y == launch.result)
    throw std::invalid_argument(std::string(what) + " has unsupported aliasing");
  if (!launch.elements || !launch.blocks)
    throw std::invalid_argument(std::string(what) + " has an empty extent");
  const size_t needed = 1 + (launch.elements - 1) / cw_threads;
  if (launch.blocks > 128 || launch.blocks > needed ||
      launch.blocks > std::numeric_limits<unsigned int>::max())
    throw std::invalid_argument(std::string(what) + " has an invalid partial count");
  if (Operation == 2 && (!std::isfinite(launch.scale) || launch.scale <= 0.0))
    throw std::invalid_argument("NVIDIA solve_cw scaled norm has an invalid scale");
  reduction_partials_kernel<T, Operation>
      <<<unsigned(launch.blocks), cw_threads, 0,
         static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch);
  check_cuda(cudaPeekAtLastError(), what);
  reduction_final_kernel<Operation><<<1, cw_threads, 0,
                                      static_cast<cudaStream_t>(stream.opaque_handle())>>>(
      static_cast<const double *>(launch.partials), static_cast<double *>(launch.result),
      launch.blocks);
  check_cuda(cudaPeekAtLastError(), what);
}

template <int Operation>
void launch_reduction(const cw_reduction_launch &launch, const stream &stream, const char *what) {
  switch (launch.precision) {
    case scalar_precision::f32: launch_reduction_typed<float, Operation>(launch, stream, what); break;
    case scalar_precision::f64: launch_reduction_typed<double, Operation>(launch, stream, what); break;
    default: throw std::invalid_argument(std::string(what) + " precision is invalid");
  }
}

} // namespace

void launch_cw_pack(const cw_state_row_launch &launch, void *vector, size_t real_elements,
                    const stream &stream) {
  validate_row(launch, vector, real_elements, "NVIDIA solve_cw pack");
  const unsigned int blocks = launch_blocks(launch.complex_count, "NVIDIA solve_cw pack");
  if (launch.precision == scalar_precision::f32)
    pack_kernel<float><<<blocks, cw_threads, 0,
                         static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch,
                                                                             static_cast<float *>(vector));
  else if (launch.precision == scalar_precision::f64)
    pack_kernel<double><<<blocks, cw_threads, 0,
                          static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch,
                                                                              static_cast<double *>(vector));
  else
    throw std::invalid_argument("NVIDIA solve_cw pack precision is invalid");
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA solve_cw pack kernel");
}

void launch_cw_unpack(const cw_state_row_launch &launch, const void *vector,
                      size_t real_elements, const stream &stream) {
  validate_row(launch, vector, real_elements, "NVIDIA solve_cw unpack");
  const unsigned int blocks = launch_blocks(launch.complex_count, "NVIDIA solve_cw unpack");
  if (launch.precision == scalar_precision::f32)
    unpack_kernel<float><<<blocks, cw_threads, 0,
                           static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch,
                                                                               static_cast<const float *>(vector));
  else if (launch.precision == scalar_precision::f64)
    unpack_kernel<double><<<blocks, cw_threads, 0,
                            static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch,
                                                                                static_cast<const double *>(vector));
  else
    throw std::invalid_argument("NVIDIA solve_cw unpack precision is invalid");
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA solve_cw unpack kernel");
}

void launch_cw_zero(const cw_zero_launch &launch, const stream &stream) {
  if (!launch.values) throw std::invalid_argument("NVIDIA solve_cw zero has a null array");
  const unsigned int blocks = launch_blocks(launch.elements, "NVIDIA solve_cw zero");
  if (launch.precision == scalar_precision::f32)
    zero_kernel<float><<<blocks, cw_threads, 0,
                         static_cast<cudaStream_t>(stream.opaque_handle())>>>(
        static_cast<float *>(launch.values), launch.elements);
  else if (launch.precision == scalar_precision::f64)
    zero_kernel<double><<<blocks, cw_threads, 0,
                          static_cast<cudaStream_t>(stream.opaque_handle())>>>(
        static_cast<double *>(launch.values), launch.elements);
  else
    throw std::invalid_argument("NVIDIA solve_cw zero precision is invalid");
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA solve_cw zero kernel");
}

void launch_cw_source_batch(const cw_source_batch_launch &launch, const void *device_scalars,
                            const stream &stream) {
  if (!launch.target_real || !launch.target_imag || !launch.points || !device_scalars)
    throw std::invalid_argument("NVIDIA solve_cw source has a null array");
  if (launch.target_real == launch.target_imag)
    throw std::invalid_argument("NVIDIA solve_cw source aliases its destinations");
  if (!launch.point_count)
    throw std::invalid_argument("NVIDIA solve_cw source has no spatial points");
  if (!std::isfinite(launch.dt) || launch.dt <= 0.0)
    throw std::invalid_argument("NVIDIA solve_cw source has an invalid timestep");
  const unsigned int threads = launch.sequential ? 1 : unsigned(std::min<size_t>(256, launch.point_count));
  const unsigned int blocks = launch.sequential ? 1 : launch_blocks(launch.point_count,
                                                                    "NVIDIA solve_cw source");
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream.opaque_handle());
  if (launch.precision == scalar_precision::f32) {
    if (launch.sequential)
      ordered_cw_source_kernel<float><<<blocks, threads, 0, cuda_stream>>>(
          launch, static_cast<const source_scalar *>(device_scalars));
    else
      cw_source_kernel<float><<<blocks, threads, 0, cuda_stream>>>(
          launch, static_cast<const source_scalar *>(device_scalars));
  }
  else if (launch.precision == scalar_precision::f64) {
    if (launch.sequential)
      ordered_cw_source_kernel<double><<<blocks, threads, 0, cuda_stream>>>(
          launch, static_cast<const source_scalar *>(device_scalars));
    else
      cw_source_kernel<double><<<blocks, threads, 0, cuda_stream>>>(
          launch, static_cast<const source_scalar *>(device_scalars));
  }
  else throw std::invalid_argument("NVIDIA solve_cw source precision is invalid");
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA solve_cw source kernel");
}

void launch_cw_vector(const cw_vector_launch &launch, const stream &stream) {
  if (!launch.output || !launch.x)
    throw std::invalid_argument("NVIDIA solve_cw vector operation has a null array");
  const bool needs_y = launch.operation == cw_vector_operation::subtract_field ||
                       launch.operation == cw_vector_operation::linear_f64_coefficient;
  if (needs_y && !launch.y)
    throw std::invalid_argument("NVIDIA solve_cw vector operation has a null right operand");
  if (!needs_y && launch.y)
    throw std::invalid_argument("NVIDIA solve_cw vector operation has an unexpected right operand");
  if (launch.x == launch.y)
    throw std::invalid_argument("NVIDIA solve_cw vector operation aliases its inputs");
  if (!std::isfinite(launch.coefficient))
    throw std::invalid_argument("NVIDIA solve_cw vector coefficient is nonfinite");
  const unsigned int blocks = launch_blocks(launch.elements, "NVIDIA solve_cw vector operation");
#define LAUNCH_VECTOR(T)                                                                           \
  switch (launch.operation) {                                                                      \
    case cw_vector_operation::copy:                                                                \
      vector_kernel<T, cw_vector_operation::copy>                                                  \
          <<<blocks, cw_threads, 0, static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch);  \
      break;                                                                                       \
    case cw_vector_operation::subtract_field:                                                      \
      vector_kernel<T, cw_vector_operation::subtract_field>                                        \
          <<<blocks, cw_threads, 0, static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch);  \
      break;                                                                                       \
    case cw_vector_operation::scale_field_coefficient:                                             \
      vector_kernel<T, cw_vector_operation::scale_field_coefficient>                               \
          <<<blocks, cw_threads, 0, static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch);  \
      break;                                                                                       \
    case cw_vector_operation::scale_f64_coefficient:                                               \
      vector_kernel<T, cw_vector_operation::scale_f64_coefficient>                                 \
          <<<blocks, cw_threads, 0, static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch);  \
      break;                                                                                       \
    case cw_vector_operation::linear_f64_coefficient:                                              \
      vector_kernel<T, cw_vector_operation::linear_f64_coefficient>                                \
          <<<blocks, cw_threads, 0, static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch);  \
      break;                                                                                       \
    default: throw std::invalid_argument("NVIDIA solve_cw vector operation is invalid");          \
  }
  if (launch.precision == scalar_precision::f32) {
    LAUNCH_VECTOR(float);
  }
  else if (launch.precision == scalar_precision::f64) {
    LAUNCH_VECTOR(double);
  }
  else throw std::invalid_argument("NVIDIA solve_cw vector precision is invalid");
#undef LAUNCH_VECTOR
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA solve_cw vector kernel");
}

void launch_cw_operator_finalize(const cw_operator_launch &launch, const stream &stream) {
  if (!launch.output || !launch.stepped || !launch.input)
    throw std::invalid_argument("NVIDIA solve_cw operator has a null array");
  if (launch.output == launch.input || launch.input == launch.stepped)
    throw std::invalid_argument("NVIDIA solve_cw operator has unsupported aliasing");
  if (!launch.real_elements || (launch.real_elements & 1))
    throw std::invalid_argument("NVIDIA solve_cw operator has an invalid real-vector extent");
  if (!std::isfinite(launch.dt_inverse) || !std::isfinite(launch.iomega_real) ||
      !std::isfinite(launch.iomega_imaginary))
    throw std::invalid_argument("NVIDIA solve_cw operator coefficient is nonfinite");
  const unsigned int blocks = launch_blocks(launch.real_elements / 2,
                                             "NVIDIA solve_cw operator");
  if (launch.precision == scalar_precision::f32)
    operator_kernel<float><<<blocks, cw_threads, 0,
                             static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch);
  else if (launch.precision == scalar_precision::f64)
    operator_kernel<double><<<blocks, cw_threads, 0,
                              static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch);
  else
    throw std::invalid_argument("NVIDIA solve_cw operator precision is invalid");
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA solve_cw operator kernel");
}

void launch_cw_dot(const cw_reduction_launch &launch, const stream &stream) {
  launch_reduction<0>(launch, stream, "launching NVIDIA solve_cw dot reduction");
}

void launch_cw_max_abs(const cw_reduction_launch &launch, const stream &stream) {
  launch_reduction<1>(launch, stream, "launching NVIDIA solve_cw max reduction");
}

void launch_cw_scaled_norm_sum(const cw_reduction_launch &launch, const stream &stream) {
  launch_reduction<2>(launch, stream, "launching NVIDIA solve_cw scaled-norm reduction");
}

} // namespace nvidia
} // namespace meep
