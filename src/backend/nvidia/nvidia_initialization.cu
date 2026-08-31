/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_initialization.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace meep {
namespace nvidia {
namespace {

void check_cuda(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) return;
  throw runtime_error(operation, static_cast<int>(result), cudaGetErrorName(result),
                      cudaGetErrorString(result));
}

unsigned launch_blocks(size_t elements, const char *what) {
  if (!elements) throw std::invalid_argument(std::string(what) + " has no elements");
  const size_t blocks = 1 + (elements - 1) / 256;
  if (blocks > std::numeric_limits<unsigned>::max())
    throw std::overflow_error(std::string(what) + " launch grid overflows");
  return unsigned(blocks);
}

size_t checked_scalar_bytes(size_t elements, scalar_precision precision, const char *what) {
  size_t width = 0;
  switch (precision) {
    case scalar_precision::f32: width = sizeof(float); break;
    case scalar_precision::f64: width = sizeof(double); break;
    default: throw std::invalid_argument(std::string(what) + " precision is invalid");
  }
  if (elements > std::numeric_limits<size_t>::max() / width)
    throw std::overflow_error(std::string(what) + " byte range overflows");
  return elements * width;
}

void validate_destination(void *destination, size_t bytes, size_t alignment, const char *what) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(destination);
  if (!begin || begin % alignment || bytes > std::numeric_limits<uintptr_t>::max() - begin)
    throw std::invalid_argument(std::string(what) + " destination is invalid");
}

bool ranges_overlap(const void *left, size_t left_bytes, const void *right, size_t right_bytes) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(left);
  const uintptr_t b = reinterpret_cast<uintptr_t>(right);
  return a < b + right_bytes && b < a + left_bytes;
}

template <typename T>
__global__ void material_fill_kernel(T *destination, size_t elements, double value) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < elements) destination[i] = T(value);
}

__device__ double absorber_coordinate(const double coordinates[5], int direction) {
  return direction >= 0 && direction < 5 ? coordinates[direction] : 0.0;
}

template <typename T>
__global__ void material_conductivity_kernel(material_conductivity_launch launch) {
  const size_t point = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (point >= launch.loop_count) return;

  size_t remaining = point;
  const size_t i2 = remaining % launch.loop_extent[2];
  remaining /= launch.loop_extent[2];
  const size_t i1 = remaining % launch.loop_extent[1];
  const size_t i0 = remaining / launch.loop_extent[1];
  const size_t loop_index[3] = {i0, i1, i2};
  size_t destination_index = 0;
  double coordinates[5] = {0, 0, 0, 0, 0};
  for (int axis = 0; axis < 3; ++axis) {
    destination_index += launch.loop_base_offset[axis] +
                         loop_index[axis] * size_t(launch.strides[axis]);
    const int direction = launch.axis_direction[axis];
    if (direction >= 0 && direction < 5)
      coordinates[direction] =
          (0.5 * launch.loop_begin[axis] + double(loop_index[axis])) * launch.inva;
  }

  double conductivity = launch.base_conductivity;
  const material_absorber_header *headers = reinterpret_cast<const material_absorber_header *>(
      launch.compact_inputs + launch.absorber_header_offset);
  for (size_t absorber = 0; absorber < launch.absorber_count; ++absorber) {
    const material_absorber_header header = headers[absorber];
    const double *samples =
        reinterpret_cast<const double *>(launch.compact_inputs + header.sample_offset);
    const double x = absorber_coordinate(coordinates, header.direction);
    const double half_cell = 0.5 * launch.cell_size[header.direction];
    const bool high = header.side == 0;
    const double edge = high ? half_cell - header.thickness : header.thickness - half_cell;
    if ((high && x >= edge) || (!high && x <= edge)) {
      const size_t intervals = size_t(header.sample_count - 1);
      const double u = double(intervals) * (high ? x - edge : edge - x) / header.thickness;
      const int sample = int(u);
      if (sample >= int(intervals))
        conductivity += samples[intervals];
      else {
        const double fraction = u - sample;
        conductivity += samples[sample] * (1 - fraction) + samples[sample + 1] * fraction;
      }
    }
  }

  double inverse;
  if (launch.logical_single) {
    const float logical_conductivity = float(conductivity);
    conductivity = logical_conductivity;
    inverse = float(1 / (1 + double(logical_conductivity) * launch.dt * 0.5));
  }
  else
    inverse = 1 / (1 + conductivity * launch.dt * 0.5);
  static_cast<T *>(launch.conductivity_destination)[destination_index] = T(conductivity);
  static_cast<T *>(launch.condinv_destination)[destination_index] = T(inverse);
}

template <typename T>
__global__ void material_pml_kernel(T *sigma_destination, T *kappa_destination,
                                    T *sigma_inv_destination,
                                    const unsigned char *compact_inputs, size_t profile_offset,
                                    size_t elements, int little_corner, double resolution,
                                    double dt, double thickness, double boundary_location,
                                    double r_asymptotic, double mean_stretch,
                                    double profile_integral, double profile_integral_u,
                                    int thickness_cells,
                                    bool profile_active, bool analytic_quadratic,
                                    bool logical_single) {
  (void)analytic_quadratic; // authority is the captured sample array for both profile kinds
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < elements) {
    double sigma = 0.0, kappa = 1.0, sigma_inv = 1.0;
    if (profile_active) {
      const int logical_index = little_corner + int(i);
      const double here = logical_index * 0.5 / resolution;
      const double x = 0.5 / resolution *
                       (thickness_cells -
                        int(fabs(boundary_location - here) * (2 * resolution) + 0.5));
      if (x > 0.0) {
        const double sample =
            reinterpret_cast<const double *>(compact_inputs + profile_offset)[i];
        const double prefactor = (-log(r_asymptotic)) / (4 * thickness * profile_integral);
        const double kappa_prefactor = (mean_stretch - 1) / profile_integral_u;
        sigma = 0.5 * dt * prefactor * sample;
        kappa = 1 + kappa_prefactor * sample * (x / thickness);
        if (logical_single) {
          const float logical_sigma = float(sigma), logical_kappa = float(kappa);
          sigma = logical_sigma;
          kappa = logical_kappa;
          sigma_inv = float(1 / (logical_kappa + logical_sigma));
        }
        else
          sigma_inv = 1 / (kappa + sigma);
      }
    }
    sigma_destination[i] = T(sigma);
    kappa_destination[i] = T(kappa);
    sigma_inv_destination[i] = T(sigma_inv);
  }
}

template <typename T>
void launch_fill_typed(const material_fill_launch &launch, const stream &stream) {
  material_fill_kernel<T><<<launch_blocks(launch.elements, "NVIDIA material fill"), 256, 0,
                            static_cast<cudaStream_t>(stream.opaque_handle())>>>(
      static_cast<T *>(launch.destination), launch.elements, launch.value);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA material fill kernel");
}

template <typename T>
void launch_conductivity_typed(const material_conductivity_launch &launch,
                               const stream &stream) {
  material_conductivity_kernel<T>
      <<<launch_blocks(launch.loop_count, "NVIDIA material conductivity"), 256, 0,
         static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA material conductivity kernel");
}

template <typename T>
void launch_pml_typed(const material_pml_launch &launch, const stream &stream) {
  material_pml_kernel<T><<<launch_blocks(launch.elements, "NVIDIA material PML"), 256, 0,
                           static_cast<cudaStream_t>(stream.opaque_handle())>>>(
      static_cast<T *>(launch.sigma_destination), static_cast<T *>(launch.kappa_destination),
      static_cast<T *>(launch.sigma_inv_destination), launch.compact_inputs,
      launch.profile_offset, launch.elements, launch.little_corner, launch.resolution,
      launch.dt, launch.thickness,
      launch.boundary_location, launch.r_asymptotic, launch.mean_stretch,
      launch.profile_integral, launch.profile_integral_u, launch.thickness_cells,
      launch.profile_active, launch.analytic_quadratic,
      launch.logical_single);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA material PML kernel");
}

} // namespace

void launch_material_fill(const material_fill_launch &launch, const stream &stream) {
  if (!launch.elements) return;
  if (!launch.destination)
    throw std::invalid_argument("NVIDIA material fill has a null destination");
  const size_t bytes = checked_scalar_bytes(launch.elements, launch.precision,
                                            "NVIDIA material fill");
  const size_t alignment = launch.precision == scalar_precision::f32 ? alignof(float)
                                                                     : alignof(double);
  validate_destination(launch.destination, bytes, alignment, "NVIDIA material fill");
  if (testing::consume_failure_for_testing(testing::failure_point::material_pointwise_launch))
    throw std::runtime_error("injected NVIDIA material pointwise launch failure");
  switch (launch.precision) {
    case scalar_precision::f32: launch_fill_typed<float>(launch, stream); break;
    case scalar_precision::f64: launch_fill_typed<double>(launch, stream); break;
    default: throw std::invalid_argument("NVIDIA material fill precision is invalid");
  }
}

void validate_material_absorber_headers(const unsigned char *compact_inputs,
                                        size_t compact_input_bytes,
                                        size_t absorber_header_offset,
                                        size_t absorber_count) {
  if (!compact_inputs || reinterpret_cast<uintptr_t>(compact_inputs) % alignof(double) ||
      absorber_header_offset % alignof(material_absorber_header) ||
      absorber_header_offset > compact_input_bytes ||
      absorber_count > (compact_input_bytes - absorber_header_offset) /
                           sizeof(material_absorber_header))
    throw std::invalid_argument("NVIDIA material conductivity compact header range is invalid");
  const size_t header_bytes = absorber_count * sizeof(material_absorber_header);
  const size_t header_end = absorber_header_offset + header_bytes;
  const material_absorber_header *headers = reinterpret_cast<const material_absorber_header *>(
      compact_inputs + absorber_header_offset);
  for (size_t i = 0; i < absorber_count; ++i) {
    const material_absorber_header &header = headers[i];
    if (header.version != 1 || header.reserved != 0 || header.direction < 0 ||
        header.direction >= 5 || (header.side != 0 && header.side != 1) ||
        header.sample_count < 2 ||
        header.sample_count > uint64_t(std::numeric_limits<int>::max()) ||
        !std::isfinite(header.thickness) || !(header.thickness > 0) ||
        !std::isfinite(header.sample_spacing) || !(header.sample_spacing > 0) ||
        header.sample_spacing != header.thickness / double(header.sample_count - 1) ||
        header.sample_offset % alignof(double) ||
        header.sample_offset > compact_input_bytes ||
        header.sample_count >
            uint64_t((compact_input_bytes - size_t(header.sample_offset)) / sizeof(double)))
      throw std::invalid_argument("NVIDIA material absorber header is invalid");
    const size_t sample_begin = size_t(header.sample_offset);
    const size_t sample_bytes = size_t(header.sample_count) * sizeof(double);
    const size_t sample_end = sample_begin + sample_bytes;
    if (sample_begin < header_end && absorber_header_offset < sample_end)
      throw std::invalid_argument("NVIDIA material absorber samples overlap compact headers");
    const double *samples = reinterpret_cast<const double *>(compact_inputs + sample_begin);
    for (size_t sample = 0; sample < size_t(header.sample_count); ++sample)
      if (!std::isfinite(samples[sample]))
        throw std::invalid_argument("NVIDIA material absorber sample is non-finite");
    for (size_t previous = 0; previous < i; ++previous) {
      const size_t previous_begin = size_t(headers[previous].sample_offset);
      const size_t previous_end =
          previous_begin + size_t(headers[previous].sample_count) * sizeof(double);
      if (sample_begin < previous_end && previous_begin < sample_end)
        throw std::invalid_argument("NVIDIA material absorber sample ranges overlap");
    }
  }
}

void launch_material_conductivity(const material_conductivity_launch &launch,
                                  const stream &stream) {
  if (!launch.loop_count) return;
  if (!launch.conductivity_destination || !launch.condinv_destination ||
      !launch.compact_inputs || !launch.absorber_count || !launch.elements)
    throw std::invalid_argument("NVIDIA material conductivity launch has a null input");
  const size_t destination_bytes = checked_scalar_bytes(
      launch.elements, launch.precision, "NVIDIA material conductivity");
  const size_t destination_alignment = launch.precision == scalar_precision::f32
                                           ? alignof(float)
                                           : alignof(double);
  validate_destination(launch.conductivity_destination, destination_bytes,
                       destination_alignment, "NVIDIA material conductivity");
  validate_destination(launch.condinv_destination, destination_bytes,
                       destination_alignment, "NVIDIA material conductivity inverse");
  if (ranges_overlap(launch.conductivity_destination, destination_bytes,
                     launch.condinv_destination, destination_bytes) ||
      reinterpret_cast<uintptr_t>(launch.compact_inputs) % alignof(double) ||
      !std::isfinite(launch.inva) || !(launch.inva > 0) ||
      !std::isfinite(launch.base_conductivity) || !std::isfinite(launch.dt) ||
      !(launch.dt > 0) || launch.component < 0 || launch.component >= 20 ||
      launch.dimensions < 0 || launch.dimensions > 3)
    throw std::invalid_argument("NVIDIA material conductivity descriptor is invalid");
  for (int direction = 0; direction < 5; ++direction)
    if (!std::isfinite(launch.cell_size[direction]) || launch.cell_size[direction] < 0)
      throw std::invalid_argument("NVIDIA material conductivity cell size is invalid");
  if (launch.absorber_header_offset % alignof(material_absorber_header) ||
      launch.absorber_count >
          (launch.compact_input_bytes -
           (launch.absorber_header_offset <= launch.compact_input_bytes
                ? launch.absorber_header_offset
                : launch.compact_input_bytes)) /
              sizeof(material_absorber_header))
    throw std::invalid_argument("NVIDIA material conductivity compact header range is invalid");
  size_t loop_product = 1;
  size_t maximum = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const int64_t stagger = int64_t(launch.loop_begin[axis]) -
                            int64_t(launch.little_corner[axis]);
    if (!launch.loop_extent[axis] || launch.strides[axis] < 0 ||
        launch.axis_direction[axis] < 0 || launch.axis_direction[axis] >= 5 ||
        stagger < 0 || stagger > 1 || launch.loop_base_offset[axis] !=
                                       size_t(stagger / 2) * size_t(launch.strides[axis]) ||
        launch.loop_extent[axis] > std::numeric_limits<size_t>::max() / loop_product)
      throw std::invalid_argument("NVIDIA material conductivity loop shape is invalid");
    loop_product *= launch.loop_extent[axis];
    const size_t base = launch.loop_base_offset[axis];
    const long double last_coordinate =
        (0.5L * launch.loop_begin[axis] + launch.loop_extent[axis] - 1) * launch.inva;
    if (!std::isfinite(double(last_coordinate)))
      throw std::invalid_argument("NVIDIA material conductivity coordinate range is invalid");
    const size_t coordinate = base + launch.loop_extent[axis] - 1;
    if (coordinate && size_t(launch.strides[axis]) >
                          (std::numeric_limits<size_t>::max() - maximum) / coordinate)
      throw std::invalid_argument("NVIDIA material conductivity address range overflows");
    maximum += coordinate * size_t(launch.strides[axis]);
  }
  if (loop_product != launch.loop_count || maximum >= launch.elements)
    throw std::invalid_argument("NVIDIA material conductivity loop exceeds its destination");
  if (testing::consume_failure_for_testing(testing::failure_point::material_pointwise_launch))
    throw std::runtime_error("injected NVIDIA material pointwise launch failure");
  switch (launch.precision) {
    case scalar_precision::f32: launch_conductivity_typed<float>(launch, stream); break;
    case scalar_precision::f64: launch_conductivity_typed<double>(launch, stream); break;
    default: throw std::invalid_argument("NVIDIA material conductivity precision is invalid");
  }
}

void launch_material_pml(const material_pml_launch &launch, const stream &stream) {
  if (!launch.elements) return;
  if (!launch.sigma_destination || !launch.kappa_destination ||
      !launch.sigma_inv_destination ||
      (launch.profile_active &&
       (!launch.compact_inputs || launch.profile_offset % alignof(double) ||
        launch.profile_offset > launch.compact_input_bytes ||
        launch.elements >
            (launch.compact_input_bytes - launch.profile_offset) / sizeof(double))))
    throw std::invalid_argument("NVIDIA material PML launch has a null pointer");
  const size_t destination_bytes =
      checked_scalar_bytes(launch.elements, launch.precision, "NVIDIA material PML");
  const size_t destination_alignment = launch.precision == scalar_precision::f32
                                           ? alignof(float)
                                           : alignof(double);
  validate_destination(launch.sigma_destination, destination_bytes, destination_alignment,
                       "NVIDIA material PML sigma");
  validate_destination(launch.kappa_destination, destination_bytes, destination_alignment,
                       "NVIDIA material PML kappa");
  validate_destination(launch.sigma_inv_destination, destination_bytes, destination_alignment,
                       "NVIDIA material PML inverse");
  if (ranges_overlap(launch.sigma_destination, destination_bytes, launch.kappa_destination,
                     destination_bytes) ||
      ranges_overlap(launch.sigma_destination, destination_bytes,
                     launch.sigma_inv_destination, destination_bytes) ||
      ranges_overlap(launch.kappa_destination, destination_bytes,
                     launch.sigma_inv_destination, destination_bytes) ||
      !std::isfinite(launch.resolution) || !(launch.resolution > 0) ||
      !std::isfinite(launch.dt) || !(launch.dt > 0) ||
      launch.elements > size_t(std::numeric_limits<int>::max()) ||
      launch.little_corner > std::numeric_limits<int>::max() - int(launch.elements - 1) ||
      (launch.profile_active &&
       reinterpret_cast<uintptr_t>(launch.compact_inputs) % alignof(double)))
    throw std::invalid_argument("NVIDIA material PML descriptor is invalid");
  if (launch.profile_active &&
      (!std::isfinite(launch.thickness) || !(launch.thickness > 0) ||
       !std::isfinite(launch.boundary_location) || !std::isfinite(launch.r_asymptotic) ||
       !(launch.r_asymptotic > 0) || !(launch.r_asymptotic < 1) ||
       !std::isfinite(launch.mean_stretch) || launch.mean_stretch < 1 ||
       !std::isfinite(launch.profile_integral) || !(launch.profile_integral > 0) ||
       !std::isfinite(launch.profile_integral_u) || !(launch.profile_integral_u > 0)))
    throw std::invalid_argument("NVIDIA material PML profile descriptor is invalid");
  if (launch.profile_active) {
    /* Match structure.cpp:pml_x exactly: changing the operands or association
       to long double can move a half-cell boundary across the integer cast. */
    const double scaled_thickness =
        launch.thickness * (2 * launch.resolution) + 0.5;
    if (!std::isfinite(scaled_thickness) || scaled_thickness <= 0 ||
        scaled_thickness > double(INT_MAX) ||
        launch.thickness_cells != int(scaled_thickness))
      throw std::invalid_argument("NVIDIA material PML thickness cells are inconsistent");
    const long double first_here = static_cast<long double>(launch.little_corner) * 0.5L /
                                   launch.resolution;
    const long double last_here =
        (static_cast<long double>(launch.little_corner) + launch.elements - 1) * 0.5L /
        launch.resolution;
    const long double maximum_distance =
        std::max(std::fabs(static_cast<long double>(launch.boundary_location) - first_here),
                 std::fabs(static_cast<long double>(launch.boundary_location) - last_here)) *
            (2 * launch.resolution) +
        0.5L;
    if (!std::isfinite(double(maximum_distance)) || maximum_distance > INT_MAX)
      throw std::invalid_argument("NVIDIA material PML distance conversion overflows");
  }
  if (testing::consume_failure_for_testing(testing::failure_point::material_pml_launch))
    throw std::runtime_error("injected NVIDIA material PML launch failure");
  switch (launch.precision) {
    case scalar_precision::f32: launch_pml_typed<float>(launch, stream); break;
    case scalar_precision::f64: launch_pml_typed<double>(launch, stream); break;
    default: throw std::invalid_argument("NVIDIA material PML precision is invalid");
  }
}

} // namespace nvidia
} // namespace meep
