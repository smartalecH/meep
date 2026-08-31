/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_initialization.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
bool compact_range(size_t offset, size_t count, size_t bytes) {
  return offset % alignof(T) == 0 && offset <= bytes &&
         count <= (bytes - offset) / sizeof(T);
}

__host__ __device__ int table_mirror_index(int i, int n) {
  const int64_t wide_i = i, wide_n = n;
  const int64_t result = i >= n ? 2 * wide_n - 1 - wide_i
                                : (i < 0 ? -1 - wide_i : wide_i);
  return int(result);
}

__device__ double table_interpolate(const material_table_header &header,
                                    const unsigned char *compact_inputs,
                                    double rx, double ry, double rz) {
  rx = rx < 0.0 ? -rx : (rx > 1.0 ? 1.0 - rx : rx);
  ry = ry < 0.0 ? -ry : (ry > 1.0 ? 1.0 - ry : ry);
  rz = rz < 0.0 ? -rz : (rz > 1.0 ? 1.0 - rz : rz);
  const int nx = int(header.dimensions[0]), ny = int(header.dimensions[1]);
  const int nz = int(header.dimensions[2]);
  const int x1 = table_mirror_index(int(rx * nx), nx);
  const int y1 = table_mirror_index(int(ry * ny), ny);
  const int z1 = table_mirror_index(int(rz * nz), nz);
  double dx = rx * nx - x1 - 0.5;
  double dy = ry * ny - y1 - 0.5;
  double dz = rz * nz - z1 - 0.5;
  const int x2 = table_mirror_index(dx >= 0.0 ? x1 + 1 : x1 - 1, nx);
  const int y2 = table_mirror_index(dy >= 0.0 ? y1 + 1 : y1 - 1, ny);
  const int z2 = table_mirror_index(dz >= 0.0 ? z1 + 1 : z1 - 1, nz);
  dx = fabs(dx); dy = fabs(dy); dz = fabs(dz);
  const double *samples = reinterpret_cast<const double *>(compact_inputs + header.sample_offset);
#define TABLE_D(x, y, z) samples[((size_t(x) * size_t(ny) + size_t(y)) * size_t(nz) + size_t(z))]
  const double result =
      (((TABLE_D(x1, y1, z1) * (1.0 - dx) + TABLE_D(x2, y1, z1) * dx) * (1.0 - dy) +
        (TABLE_D(x1, y2, z1) * (1.0 - dx) + TABLE_D(x2, y2, z1) * dx) * dy) *
           (1.0 - dz) +
       ((TABLE_D(x1, y1, z2) * (1.0 - dx) + TABLE_D(x2, y1, z2) * dx) * (1.0 - dy) +
        (TABLE_D(x1, y2, z2) * (1.0 - dx) + TABLE_D(x2, y2, z2) * dx) * dy) * dz);
#undef TABLE_D
  return result;
}

__device__ double table_grid_value(const material_table_header &header,
                                   const unsigned char *compact_inputs,
                                   double rx, double ry, double rz) {
  const double sample = table_interpolate(header, compact_inputs, rx, ry, rz);
  double reduced = sample;
  if (header.overlap_kind == 0) reduced = sample < 1.0 ? sample : 1.0;
  /* A global default grid has exactly one contributor. U_PROD, U_MEAN, and
     U_DEFAULT therefore all reduce to the sample itself. */
  reduced += header.projection_offset;
  if (header.beta == 0.0) return reduced;
  if (reduced == header.eta) return 0.5;
  const double tanh_beta_eta = tanh(header.beta * header.eta);
  return (tanh_beta_eta + tanh(header.beta * (reduced - header.eta))) /
         (tanh_beta_eta + tanh(header.beta * (1.0 - header.eta)));
}

__device__ void table_point(const material_table_launch &launch, size_t point,
                            size_t &destination_index, double normalized[3],
                            double physical_by_direction[5]) {
  size_t remaining = point;
  const size_t i2 = remaining % launch.loop_extent[2];
  remaining /= launch.loop_extent[2];
  const size_t i1 = remaining % launch.loop_extent[1];
  const size_t i0 = remaining / launch.loop_extent[1];
  const size_t loop_index[3] = {i0, i1, i2};
  double coordinate[3] = {0.0, 0.0, 0.0};
  destination_index = 0;
  for (int axis = 0; axis < 3; ++axis) {
    destination_index += launch.loop_base_offset[axis] +
                         loop_index[axis] * size_t(launch.strides[axis]);
    const double physical =
        0.5 * (double(launch.loop_begin[axis]) + 2.0 * double(loop_index[axis]) +
               double(launch.evaluation_shift[axis])) * launch.inva;
    const int direction = launch.axis_direction[axis];
    if (direction >= 0 && direction < 5) physical_by_direction[direction] = physical;
    if (direction == 0 || direction == 3) coordinate[0] = physical;
    else if (direction == 1) coordinate[1] = physical;
    else if (direction == 2) coordinate[2] = physical;
  }
  if (launch.dimensions == 0) { // D1
    coordinate[0] = 0.0; coordinate[1] = 0.0;
  }
  else if (launch.dimensions == 1) coordinate[2] = 0.0; // D2
  else if (launch.dimensions == 3) coordinate[1] = 0.0; // cylindrical
  for (int axis = 0; axis < 3; ++axis)
    normalized[axis] = launch.cell_size[axis] == 0.0
                           ? 0.0
                           : 0.5 + (coordinate[axis] - launch.cell_center[axis]) /
                                       launch.cell_size[axis];
}

__device__ double table_tensor_member(const double diagonal[3], const double offdiagonal[3],
                                      int row, int column) {
  if (row == column) return diagonal[row];
  if ((row == 0 && column == 1) || (row == 1 && column == 0)) return offdiagonal[0];
  if ((row == 0 && column == 2) || (row == 2 && column == 0)) return offdiagonal[1];
  return offdiagonal[2];
}

__device__ double table_inverse_tensor_member(const double diagonal[3],
                                              const double offdiagonal[3], int row,
                                              int column) {
  const double m00 = diagonal[0], m11 = diagonal[1], m22 = diagonal[2];
  const double m01 = offdiagonal[0], m02 = offdiagonal[1], m12 = offdiagonal[2];
  if (m01 == 0.0 && m02 == 0.0 && m12 == 0.0)
    return row == column ? 1.0 / diagonal[row] : 0.0;
  double determinant = m00 * m11 * m22 - m02 * m11 * m02 +
                       2.0 * m01 * m12 * m02 - m01 * m01 * m22 -
                       m12 * m12 * m00;
  determinant = 1.0 / determinant;
  double inverse[3][3];
  inverse[0][0] = determinant * (m11 * m22 - m12 * m12);
  inverse[1][1] = determinant * (m00 * m22 - m02 * m02);
  inverse[2][2] = determinant * (m11 * m00 - m01 * m01);
  inverse[0][2] = inverse[2][0] = determinant * (m01 * m12 - m11 * m02);
  inverse[0][1] = inverse[1][0] = determinant * (m12 * m02 - m01 * m22);
  inverse[1][2] = inverse[2][1] = determinant * (m01 * m02 - m00 * m12);
  return inverse[row][column];
}

__device__ double table_absorber_conductivity(const material_table_launch &launch,
                                              const double physical[5]) {
  double result = 0.0;
  const material_absorber_header *headers = reinterpret_cast<const material_absorber_header *>(
      launch.compact_inputs + launch.absorber_header_offset);
  for (size_t absorber = 0; absorber < launch.absorber_count; ++absorber) {
    const material_absorber_header header = headers[absorber];
    const double *samples =
        reinterpret_cast<const double *>(launch.compact_inputs + header.sample_offset);
    const double x = physical[header.direction];
    const double half_cell = 0.5 * (header.direction == 0 || header.direction == 3
                                        ? launch.cell_size[0]
                                        : header.direction == 1 ? launch.cell_size[1]
                                                                : launch.cell_size[2]);
    const bool high = header.side == 0;
    const double edge = high ? half_cell - header.thickness : header.thickness - half_cell;
    if ((high && x >= edge) || (!high && x <= edge)) {
      const size_t intervals = size_t(header.sample_count - 1);
      const double u = double(intervals) * (high ? x - edge : edge - x) / header.thickness;
      const int sample = int(u);
      if (sample >= int(intervals)) result += samples[intervals];
      else {
        const double fraction = u - sample;
        result += samples[sample] * (1 - fraction) + samples[sample + 1] * fraction;
      }
    }
  }
  return result;
}

template <typename T>
__global__ void material_table_kernel(material_table_launch launch) {
  const size_t point = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (point >= launch.loop_count) return;
  const material_table_header &table = *reinterpret_cast<const material_table_header *>(
      launch.compact_inputs + launch.table_header_offset);
  const material_medium_header *medium1 = table.medium_1_offset
      ? reinterpret_cast<const material_medium_header *>(launch.compact_inputs + table.medium_1_offset)
      : NULL;
  const material_medium_header *medium2 = table.medium_2_offset
      ? reinterpret_cast<const material_medium_header *>(launch.compact_inputs + table.medium_2_offset)
      : NULL;
  size_t destination_index = 0;
  double normalized[3], physical[5] = {0, 0, 0, 0, 0};
  table_point(launch, point, destination_index, normalized, physical);
  const double sample = launch.table_kind == material_table_kind::material_grid
                            ? table_grid_value(table, launch.compact_inputs, normalized[0],
                                               normalized[1], normalized[2])
                            : table_interpolate(table, launch.compact_inputs, normalized[0],
                                                normalized[1], normalized[2]);
  double value = 0.0;
  if (launch.operation == material_table_operation::file_chi1inv)
    value = launch.tensor_column == launch.tensor_row ? 1.0 / sample : 0.0;
  else if (launch.operation == material_table_operation::grid_chi1inv) {
    double diagonal[3], offdiagonal[3];
    for (int i = 0; i < 3; ++i) {
      diagonal[i] = medium1->epsilon_diagonal[i] +
                    sample * (medium2->epsilon_diagonal[i] - medium1->epsilon_diagonal[i]);
      offdiagonal[i] = medium1->epsilon_offdiagonal[i] +
                       sample * (medium2->epsilon_offdiagonal[i] - medium1->epsilon_offdiagonal[i]);
    }
    value = table_inverse_tensor_member(diagonal, offdiagonal,
                                        launch.tensor_row, launch.tensor_column);
  }
  else if (launch.operation == material_table_operation::grid_sigma) {
    const material_medium_header *medium = launch.source_medium == 1 ? medium1 : medium2;
    const material_susceptibility_record *susceptibilities =
        reinterpret_cast<const material_susceptibility_record *>(
            launch.compact_inputs + medium->electric_susceptibility_offset);
    const material_susceptibility_record &sus = susceptibilities[launch.source_susceptibility];
    value = table_tensor_member(sus.sigma_diagonal, sus.sigma_offdiagonal,
                                launch.tensor_row, launch.tensor_column) *
            (launch.source_medium == 1 ? 1.0 - sample : sample);
  }
  else if (launch.operation == material_table_operation::grid_conductivity) {
    double conductivity = medium1->conductivity[launch.tensor_row] +
                          sample * (medium2->conductivity[launch.tensor_row] -
                                    medium1->conductivity[launch.tensor_row]);
    conductivity += sample * (1.0 - sample) * table.damping;
    conductivity += table_absorber_conductivity(launch, physical);
    double inverse;
    if (launch.logical_single) {
      const float logical_conductivity = float(conductivity);
      conductivity = logical_conductivity;
      inverse = float(1.0 / (1.0 + double(logical_conductivity) * launch.dt * 0.5));
    }
    else inverse = 1.0 / (1.0 + conductivity * launch.dt * 0.5);
    launch.classification[point] =
        1u | (conductivity != launch.trivial_value ? 2u : 0u);
    launch.secondary_classification[point] =
        1u | (inverse != launch.secondary_trivial_value ? 2u : 0u);
    static_cast<T *>(launch.destination)[destination_index] = T(conductivity);
    static_cast<T *>(launch.secondary_destination)[destination_index] = T(inverse);
    return;
  }
  else return;
  if (launch.logical_single) value = double(float(value));
  launch.classification[point] =
      1u | (value != launch.trivial_value ? 2u : 0u);
  static_cast<T *>(launch.destination)[destination_index] = T(value);
}

template <typename T>
__global__ void material_fill_kernel(material_fill_launch launch) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const double value = launch.logical_single ? double(float(launch.value)) : launch.value;
  if (i < launch.elements)
    static_cast<T *>(launch.destination)[i] = T(value);
  if (i < launch.classification_elements)
    launch.classification[i] = 1u | (value != launch.trivial_value ? 2u : 0u);
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
  launch.conductivity_classification[point] =
      1u | (conductivity != 0.0 ? 2u : 0u);
  launch.condinv_classification[point] =
      1u | (inverse != 1.0 ? 2u : 0u);
  static_cast<T *>(launch.conductivity_destination)[destination_index] = T(conductivity);
  static_cast<T *>(launch.condinv_destination)[destination_index] = T(inverse);
}

template <typename T>
__global__ void material_pml_kernel(T *sigma_destination, T *kappa_destination,
                                    T *sigma_inv_destination,
                                    unsigned char *sigma_classification,
                                    unsigned char *kappa_classification,
                                    unsigned char *sigma_inv_classification,
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
    sigma_classification[i] = 1u | (sigma != 0.0 ? 2u : 0u);
    kappa_classification[i] = 1u | (kappa != 1.0 ? 2u : 0u);
    sigma_inv_classification[i] = 1u | (sigma_inv != 1.0 ? 2u : 0u);
    sigma_destination[i] = T(sigma);
    kappa_destination[i] = T(kappa);
    sigma_inv_destination[i] = T(sigma_inv);
  }
}

template <typename T>
void launch_fill_typed(const material_fill_launch &launch, const stream &stream) {
  const size_t work = launch.elements > launch.classification_elements
                          ? launch.elements
                          : launch.classification_elements;
  material_fill_kernel<T><<<launch_blocks(work, "NVIDIA material fill"), 256, 0,
                            static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA material fill kernel");
}

template <typename T>
void launch_table_typed(const material_table_launch &launch, const stream &stream) {
  material_table_kernel<T>
      <<<launch_blocks(launch.loop_count, "NVIDIA material table"), 256, 0,
         static_cast<cudaStream_t>(stream.opaque_handle())>>>(launch);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA material table kernel");
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
      static_cast<T *>(launch.sigma_inv_destination), launch.sigma_classification,
      launch.kappa_classification, launch.sigma_inv_classification, launch.compact_inputs,
      launch.profile_offset, launch.elements, launch.little_corner, launch.resolution,
      launch.dt, launch.thickness,
      launch.boundary_location, launch.r_asymptotic, launch.mean_stretch,
      launch.profile_integral, launch.profile_integral_u, launch.thickness_cells,
      launch.profile_active, launch.analytic_quadratic,
      launch.logical_single);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA material PML kernel");
}

__global__ void material_classification_header_kernel(
    material_classification_header *result, material_classification_header value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *result = value;
}

__global__ void material_classification_row_kernel(
    const unsigned char *status, size_t elements,
    material_classification_row_result *result,
    material_classification_row_result value) {
  __shared__ unsigned int missing;
  __shared__ unsigned int nontrivial;
  if (threadIdx.x == 0) missing = nontrivial = 0;
  __syncthreads();
  for (size_t i = threadIdx.x; i < elements; i += blockDim.x) {
    const unsigned char state = status[i];
    if (!(state & 1u)) atomicExch(&missing, 1u);
    if (state & 2u) atomicExch(&nontrivial, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    value.missing = missing;
    value.nontrivial = nontrivial;
    *result = value;
  }
}

} // namespace

void launch_material_fill(const material_fill_launch &launch, const stream &stream) {
  if (!launch.elements && !launch.classification_elements) return;
  if (!launch.destination || !launch.classification || !std::isfinite(launch.value) ||
      !std::isfinite(launch.trivial_value) ||
      launch.classification_elements > launch.elements)
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

void validate_material_table_headers(const unsigned char *compact_inputs,
                                     size_t compact_input_bytes,
                                     const size_t *table_header_offsets,
                                     size_t table_header_count) {
  if (!compact_inputs || reinterpret_cast<uintptr_t>(compact_inputs) % alignof(double) ||
      (table_header_count && !table_header_offsets))
    throw std::invalid_argument("NVIDIA material table compact input is invalid");
  std::vector<std::pair<size_t, size_t> > ranges;
  const auto add_range = [&](size_t begin, size_t count, size_t width, const char *what) {
    if (count && width > std::numeric_limits<size_t>::max() / count)
      throw std::invalid_argument(std::string(what) + " byte range overflows");
    const size_t bytes = count * width;
    if (begin > compact_input_bytes || bytes > compact_input_bytes - begin)
      throw std::invalid_argument(std::string(what) + " range is invalid");
    const size_t end = begin + bytes;
    for (const std::pair<size_t, size_t> &range : ranges)
      if (begin < range.second && range.first < end)
        throw std::invalid_argument(std::string(what) + " overlaps another table section");
    ranges.push_back(std::make_pair(begin, end));
  };
  size_t previous_end = 0;
  for (size_t table_index = 0; table_index < table_header_count; ++table_index) {
    const size_t offset = table_header_offsets[table_index];
    if (!compact_range<material_table_header>(offset, 1, compact_input_bytes) ||
        offset < previous_end)
      throw std::invalid_argument("NVIDIA material table header range is invalid");
    add_range(offset, 1, sizeof(material_table_header), "NVIDIA material table header");
    size_t cursor = offset + sizeof(material_table_header);
    const material_table_header &header =
        *reinterpret_cast<const material_table_header *>(compact_inputs + offset);
    if (header.version != 1 || header.reserved != 0 ||
        (header.kind != material_table_kind::file_scalar_epsilon &&
         header.kind != material_table_kind::material_grid) ||
        header.overlap_kind > 3 || !std::isfinite(header.beta) ||
        !std::isfinite(header.eta) || !std::isfinite(header.damping) ||
        !std::isfinite(header.projection_offset))
      throw std::invalid_argument("NVIDIA material table header is invalid");
    size_t product = 1;
    for (int axis = 0; axis < 3; ++axis) {
      if (!header.dimensions[axis] || header.dimensions[axis] > uint32_t(INT_MAX) ||
          product > std::numeric_limits<size_t>::max() / header.dimensions[axis])
        throw std::invalid_argument("NVIDIA material table dimensions are invalid");
      product *= header.dimensions[axis];
    }
    const size_t medium_offsets[2] = {size_t(header.medium_1_offset),
                                      size_t(header.medium_2_offset)};
    const size_t medium_count = header.kind == material_table_kind::material_grid ? 2 : 0;
    if ((header.kind == material_table_kind::file_scalar_epsilon &&
         (header.medium_1_offset || header.medium_2_offset)) ||
        (header.kind == material_table_kind::material_grid &&
         (!header.medium_1_offset || !header.medium_2_offset)))
      throw std::invalid_argument("NVIDIA material table medium spans are invalid");
    for (size_t medium_index = 0; medium_index < medium_count; ++medium_index) {
      const size_t medium_offset = medium_offsets[medium_index];
      if (!compact_range<material_medium_header>(medium_offset, 1, compact_input_bytes) ||
          medium_offset < cursor)
        throw std::invalid_argument("NVIDIA material medium header range is invalid");
      add_range(medium_offset, 1, sizeof(material_medium_header),
                "NVIDIA material medium header");
      cursor = medium_offset + sizeof(material_medium_header);
      const material_medium_header &medium =
          *reinterpret_cast<const material_medium_header *>(compact_inputs + medium_offset);
      if (medium.version != 1 ||
          (medium.electric_susceptibility_count &&
           !compact_range<material_susceptibility_record>(
               size_t(medium.electric_susceptibility_offset),
               medium.electric_susceptibility_count, compact_input_bytes)))
        throw std::invalid_argument("NVIDIA material medium header is invalid");
      for (int i = 0; i < 3; ++i)
        if (!std::isfinite(medium.epsilon_diagonal[i]) ||
            !std::isfinite(medium.epsilon_offdiagonal[i]) ||
            !std::isfinite(medium.conductivity[i]))
          throw std::invalid_argument("NVIDIA material medium value is non-finite");
      if (medium.electric_susceptibility_count) {
        const size_t susceptibility_offset =
            size_t(medium.electric_susceptibility_offset);
        if (susceptibility_offset < cursor)
          throw std::invalid_argument("NVIDIA material susceptibility order is invalid");
        add_range(susceptibility_offset, medium.electric_susceptibility_count,
                  sizeof(material_susceptibility_record),
                  "NVIDIA material susceptibility records");
        cursor = susceptibility_offset +
                 size_t(medium.electric_susceptibility_count) *
                     sizeof(material_susceptibility_record);
        const material_susceptibility_record *susceptibilities =
            reinterpret_cast<const material_susceptibility_record *>(
                compact_inputs + susceptibility_offset);
        for (uint32_t i = 0; i < medium.electric_susceptibility_count; ++i) {
          const material_susceptibility_record &sus = susceptibilities[i];
          if (sus.version != 1 || sus.field_type != 0 ||
              sus.material_ordinal != i)
            throw std::invalid_argument("NVIDIA material susceptibility record is invalid");
          for (int j = 0; j < 3; ++j)
            if (!std::isfinite(sus.sigma_diagonal[j]) ||
                !std::isfinite(sus.sigma_offdiagonal[j]))
              throw std::invalid_argument("NVIDIA material susceptibility sigma is non-finite");
        }
      }
    }
    const size_t sample_offset = size_t(header.sample_offset);
    if (header.sample_count != product ||
        !compact_range<double>(sample_offset, product, compact_input_bytes) ||
        sample_offset < cursor)
      throw std::invalid_argument("NVIDIA material table sample range is invalid");
    add_range(sample_offset, product, sizeof(double), "NVIDIA material table samples");
    const double *samples = reinterpret_cast<const double *>(compact_inputs + sample_offset);
    for (size_t i = 0; i < product; ++i)
      if (!std::isfinite(samples[i]))
        throw std::invalid_argument("NVIDIA material table sample is non-finite");
    previous_end = sample_offset + product * sizeof(double);
  }
}

namespace {
int table_component_index(int component) {
  switch (component) {
    case 0: case 2: case 5: case 7: case 10: case 12: case 15: case 17: return 0;
    case 1: case 3: case 6: case 8: case 11: case 13: case 16: case 18: return 1;
    case 4: case 9: case 14: case 19: return 2;
    default: return -1;
  }
}

int table_component_direction(int component) {
  switch (component) {
    case 0: case 5: case 10: case 15: return 0;
    case 1: case 6: case 11: case 16: return 1;
    case 4: case 9: case 14: case 19: return 2;
    case 2: case 7: case 12: case 17: return 3;
    case 3: case 8: case 13: case 18: return 4;
    default: return 5;
  }
}

void validate_material_table_descriptor(const material_table_launch &launch,
                                        bool require_compact_device_pointer) {
  if (!launch.loop_count) return;
  if (!launch.destination || !launch.elements ||
      (require_compact_device_pointer &&
       (!launch.compact_inputs || !launch.classification ||
        (launch.secondary_destination && !launch.secondary_classification) ||
        !compact_range<material_table_header>(launch.table_header_offset, 1,
                                              launch.compact_input_bytes) ||
        reinterpret_cast<uintptr_t>(launch.compact_inputs) % alignof(double))) ||
      launch.dimensions < 0 || launch.dimensions > 3 ||
      launch.destination_component < 0 || launch.destination_component >= 20 ||
      launch.query_component < 0 || launch.query_component >= 20 ||
      launch.tensor_row < 0 || launch.tensor_row > 2 || launch.tensor_column < 0 ||
      launch.tensor_column > 2 || launch.operation_family != 0 ||
      !std::isfinite(launch.inva) || !(launch.inva > 0))
    throw std::invalid_argument("NVIDIA material table descriptor is invalid");
  bool valid_operation = false;
  switch (launch.operation) {
    case material_table_operation::file_chi1inv:
      valid_operation = launch.table_kind == material_table_kind::file_scalar_epsilon &&
                        launch.destination_component >= 0 &&
                        launch.destination_component <= 4 &&
                        launch.tensor_row == launch.tensor_column &&
                        !launch.secondary_destination && !launch.source_medium &&
                        !launch.source_susceptibility && !launch.susceptibility_identity &&
                        launch.susceptibility_field_type == 8 && !launch.absorber_count;
      break;
    case material_table_operation::grid_chi1inv:
      valid_operation = launch.table_kind == material_table_kind::material_grid &&
                        launch.destination_component >= 0 &&
                        launch.destination_component <= 4 &&
                        !launch.secondary_destination && !launch.source_medium &&
                        !launch.source_susceptibility && !launch.susceptibility_identity &&
                        launch.susceptibility_field_type == 8 && !launch.absorber_count;
      break;
    case material_table_operation::grid_conductivity:
      valid_operation = launch.table_kind == material_table_kind::material_grid &&
                        launch.secondary_destination && !launch.source_medium &&
                        !launch.source_susceptibility && !launch.susceptibility_identity &&
                        launch.susceptibility_field_type == 8 &&
                        launch.tensor_row == launch.tensor_column &&
                        launch.destination_component >= 10 && launch.destination_component <= 14 &&
                        launch.query_component >= 10 && launch.query_component <= 14;
      break;
    case material_table_operation::grid_sigma:
      valid_operation = launch.table_kind == material_table_kind::material_grid &&
                        !launch.secondary_destination &&
                        (launch.source_medium == 1 || launch.source_medium == 2) &&
                        launch.susceptibility_field_type == 0 &&
                        launch.destination_component >= 0 && launch.destination_component <= 4 &&
                        launch.query_component >= 0 && launch.query_component <= 4 &&
                        !launch.absorber_count;
      break;
    default: break;
  }
  if ((launch.table_kind != material_table_kind::file_scalar_epsilon &&
       launch.table_kind != material_table_kind::material_grid) ||
      !valid_operation ||
      (launch.operation == material_table_operation::grid_conductivity &&
       (!std::isfinite(launch.dt) || !(launch.dt > 0))))
    throw std::invalid_argument("NVIDIA material table operation is invalid");
  if (launch.destination_component != launch.query_component ||
      launch.tensor_row != table_component_index(launch.destination_component))
    throw std::invalid_argument("NVIDIA material table component mapping is invalid");
  const int expected_axes[4][3] = {{0, 1, 2}, {2, 0, 1}, {0, 1, 2}, {4, 3, 2}};
  for (int axis = 0; axis < 3; ++axis)
    if (launch.axis_direction[axis] != expected_axes[launch.dimensions][axis])
      throw std::invalid_argument("NVIDIA material table dimension axes are invalid");
  const bool shifted = launch.operation == material_table_operation::file_chi1inv ||
                       launch.operation == material_table_operation::grid_chi1inv ||
                       launch.operation == material_table_operation::grid_sigma;
  const int shift_direction = table_component_direction(launch.query_component);
  for (int axis = 0; axis < 3; ++axis) {
    const int expected_shift = shifted && launch.tensor_row != launch.tensor_column &&
                                       launch.axis_direction[axis] == shift_direction
                                   ? -1
                                   : 0;
    if (launch.evaluation_shift[axis] != expected_shift)
      throw std::invalid_argument("NVIDIA material table evaluation shift is invalid");
  }
  const size_t destination_bytes =
      checked_scalar_bytes(launch.elements, launch.precision, "NVIDIA material table");
  const size_t destination_alignment = launch.precision == scalar_precision::f32
                                           ? alignof(float)
                                           : alignof(double);
  validate_destination(launch.destination, destination_bytes, destination_alignment,
                       "NVIDIA material table");
  if (launch.operation == material_table_operation::grid_conductivity) {
    validate_destination(launch.secondary_destination, destination_bytes,
                         destination_alignment, "NVIDIA material table inverse");
    if (ranges_overlap(launch.destination, destination_bytes, launch.secondary_destination,
                       destination_bytes))
      throw std::invalid_argument("NVIDIA material table destinations overlap");
  }
  for (int axis = 0; axis < 3; ++axis)
    if (!std::isfinite(launch.cell_center[axis]) || !std::isfinite(launch.cell_size[axis]) ||
        launch.cell_size[axis] < 0)
      throw std::invalid_argument("NVIDIA material table cell is invalid");
  size_t loop_product = 1, maximum = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const int64_t stagger = int64_t(launch.loop_begin[axis]) -
                            int64_t(launch.little_corner[axis]);
    const int64_t doubled_extent =
        int64_t(launch.loop_end[axis]) - int64_t(launch.loop_begin[axis]);
    if (!launch.loop_extent[axis] || launch.strides[axis] < 0 ||
        launch.axis_direction[axis] < 0 || launch.axis_direction[axis] >= 5 ||
        stagger < 0 || stagger > 1 || doubled_extent < 0 || doubled_extent % 2 ||
        uint64_t(doubled_extent / 2) + 1 != launch.loop_extent[axis] ||
        launch.loop_base_offset[axis] !=
            size_t(stagger / 2) * size_t(launch.strides[axis]) ||
        launch.loop_extent[axis] > std::numeric_limits<size_t>::max() / loop_product)
      throw std::invalid_argument("NVIDIA material table loop shape is invalid");
    loop_product *= launch.loop_extent[axis];
    const size_t last = launch.loop_extent[axis] - 1;
    if (launch.loop_base_offset[axis] > std::numeric_limits<size_t>::max() - maximum)
      throw std::invalid_argument("NVIDIA material table address range overflows");
    maximum += launch.loop_base_offset[axis];
    if (last && size_t(launch.strides[axis]) >
                    (std::numeric_limits<size_t>::max() - maximum) / last)
      throw std::invalid_argument("NVIDIA material table address range overflows");
    maximum += last * size_t(launch.strides[axis]);
    const long double first_coordinate =
        static_cast<long double>(launch.loop_begin[axis]) + launch.evaluation_shift[axis];
    const long double last_coordinate =
        first_coordinate + 2.0L * (launch.loop_extent[axis] - 1);
    if (!std::isfinite(double(first_coordinate * launch.inva)) ||
        !std::isfinite(double(last_coordinate * launch.inva)))
      throw std::invalid_argument("NVIDIA material table coordinate range is invalid");
  }
  if (loop_product != launch.loop_count || maximum >= launch.elements)
    throw std::invalid_argument("NVIDIA material table loop exceeds its destination");
  double coordinate_bounds[2][3] = {};
  for (int endpoint = 0; endpoint < 2; ++endpoint)
    for (int axis = 0; axis < 3; ++axis) {
      const int doubled = endpoint ? launch.loop_end[axis] : launch.loop_begin[axis];
      const double physical =
          0.5 * (double(doubled) + launch.evaluation_shift[axis]) * launch.inva;
      const int direction = launch.axis_direction[axis];
      if (direction == 0 || direction == 3) coordinate_bounds[endpoint][0] = physical;
      else if (direction == 1) coordinate_bounds[endpoint][1] = physical;
      else if (direction == 2) coordinate_bounds[endpoint][2] = physical;
    }
  for (int endpoint = 0; endpoint < 2; ++endpoint) {
    if (launch.dimensions == 0)
      coordinate_bounds[endpoint][0] = coordinate_bounds[endpoint][1] = 0.0;
    else if (launch.dimensions == 1)
      coordinate_bounds[endpoint][2] = 0.0;
    else if (launch.dimensions == 3)
      coordinate_bounds[endpoint][1] = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double normalized = launch.cell_size[axis] == 0.0
                                    ? 0.0
                                    : 0.5 +
                                          (coordinate_bounds[endpoint][axis] -
                                           launch.cell_center[axis]) /
                                              launch.cell_size[axis];
      if (!std::isfinite(normalized) || normalized < -1.0 || normalized > 2.0)
        throw std::invalid_argument(
            "NVIDIA material table coordinate exceeds one mirror reflection");
    }
  }
}
} // namespace

void validate_material_table_launch(const material_table_launch &launch,
                                    const unsigned char *host_compact_inputs,
                                    size_t host_compact_input_bytes) {
  validate_material_table_descriptor(launch, false);
  const size_t offset = launch.table_header_offset;
  validate_material_table_headers(host_compact_inputs, host_compact_input_bytes, &offset, 1);
  const material_table_header &header = *reinterpret_cast<const material_table_header *>(
      host_compact_inputs + launch.table_header_offset);
  if (launch.table_kind != header.kind || launch.source_material_id != header.material_id)
    throw std::invalid_argument("NVIDIA material table launch/header identity differs");
  switch (launch.operation) {
    case material_table_operation::file_chi1inv:
      if (header.kind != material_table_kind::file_scalar_epsilon ||
          launch.secondary_destination || launch.source_medium ||
          launch.source_susceptibility || launch.susceptibility_identity ||
          launch.susceptibility_field_type != 8 || launch.absorber_count)
        throw std::invalid_argument("NVIDIA FILE table operation metadata is invalid");
      break;
    case material_table_operation::grid_chi1inv:
      if (header.kind != material_table_kind::material_grid || launch.secondary_destination ||
          launch.source_medium || launch.source_susceptibility ||
          launch.susceptibility_identity || launch.susceptibility_field_type != 8 ||
          launch.absorber_count)
        throw std::invalid_argument("NVIDIA grid chi1 operation metadata is invalid");
      break;
    case material_table_operation::grid_conductivity:
      if (header.kind != material_table_kind::material_grid ||
          !launch.secondary_destination || launch.source_medium ||
          launch.source_susceptibility || launch.susceptibility_identity ||
          launch.susceptibility_field_type != 8 || launch.tensor_row != launch.tensor_column ||
          launch.destination_component < 10 || launch.destination_component > 14 ||
          launch.query_component < 10 || launch.query_component > 14)
        throw std::invalid_argument("NVIDIA grid conductivity operation metadata is invalid");
      if (launch.absorber_count)
        validate_material_absorber_headers(host_compact_inputs, host_compact_input_bytes,
                                            launch.absorber_header_offset,
                                            launch.absorber_count);
      break;
    case material_table_operation::grid_sigma: {
      if (header.kind != material_table_kind::material_grid || launch.secondary_destination ||
          (launch.source_medium != 1 && launch.source_medium != 2) ||
          launch.susceptibility_field_type != 0 || launch.destination_component < 0 ||
          launch.destination_component > 4 || launch.query_component < 0 ||
          launch.query_component > 4 || launch.absorber_count)
        throw std::invalid_argument("NVIDIA grid sigma operation metadata is invalid");
      const size_t medium_offset = size_t(launch.source_medium == 1 ? header.medium_1_offset
                                                                    : header.medium_2_offset);
      const material_medium_header &medium = *reinterpret_cast<const material_medium_header *>(
          host_compact_inputs + medium_offset);
      if (launch.source_susceptibility >= medium.electric_susceptibility_count)
        throw std::invalid_argument("NVIDIA grid sigma source ordinal is invalid");
      const material_susceptibility_record *selected_records =
          reinterpret_cast<const material_susceptibility_record *>(
              host_compact_inputs + medium.electric_susceptibility_offset);
      const material_susceptibility_record &selected =
          selected_records[launch.source_susceptibility];
      if (selected.identity != launch.susceptibility_identity ||
          selected.field_type != launch.susceptibility_field_type ||
          selected.material_ordinal != launch.source_susceptibility)
        throw std::invalid_argument("NVIDIA grid sigma source identity is invalid");
      const material_medium_header *ordered_media[2] = {
          reinterpret_cast<const material_medium_header *>(host_compact_inputs +
                                                            header.medium_1_offset),
          reinterpret_cast<const material_medium_header *>(host_compact_inputs +
                                                            header.medium_2_offset)};
      uint32_t expected_medium = 0;
      uint64_t expected_ordinal = 0;
      for (uint32_t medium_index = 0; medium_index < 2 && !expected_medium; ++medium_index) {
        const material_medium_header &candidate_medium = *ordered_media[medium_index];
        const material_susceptibility_record *candidate_records =
            reinterpret_cast<const material_susceptibility_record *>(
                host_compact_inputs + candidate_medium.electric_susceptibility_offset);
        for (uint32_t ordinal = 0; ordinal < candidate_medium.electric_susceptibility_count;
             ++ordinal)
          if (candidate_records[ordinal].identity == launch.susceptibility_identity &&
              candidate_records[ordinal].field_type == launch.susceptibility_field_type) {
            expected_medium = medium_index + 1;
            expected_ordinal = ordinal;
            break;
          }
      }
      if (launch.source_medium != expected_medium ||
          launch.source_susceptibility != expected_ordinal)
        throw std::invalid_argument(
            "NVIDIA grid sigma source is not the first equivalent susceptibility");
      break;
    }
    default: throw std::invalid_argument("NVIDIA material table operation is unknown");
  }
}

void launch_material_table(const material_table_launch &launch, const stream &stream) {
  if (!launch.loop_count) return;
  validate_material_table_descriptor(launch, true);
  const testing::failure_point failure =
      launch.table_kind == material_table_kind::file_scalar_epsilon
          ? testing::failure_point::material_file_launch
          : testing::failure_point::material_grid_launch;
  if (testing::consume_failure_for_testing(failure))
    throw std::runtime_error(launch.table_kind == material_table_kind::file_scalar_epsilon
                                 ? "injected NVIDIA material FILE launch failure"
                                 : "injected NVIDIA material grid launch failure");
  switch (launch.precision) {
    case scalar_precision::f32: launch_table_typed<float>(launch, stream); break;
    case scalar_precision::f64: launch_table_typed<double>(launch, stream); break;
    default: throw std::invalid_argument("NVIDIA material table precision is invalid");
  }
}

int material_table_mirror_index_for_testing(int i, int n) {
  if (n <= 0) throw std::invalid_argument("NVIDIA material mirror extent is invalid");
  const int64_t wide_i = i, wide_n = n;
  const int64_t result = i >= n ? 2 * wide_n - 1 - wide_i
                                : (i < 0 ? -1 - wide_i : wide_i);
  if (result < 0 || result >= n)
    throw std::invalid_argument("NVIDIA material mirror index is outside one reflection");
  return int(result);
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
      !launch.conductivity_classification || !launch.condinv_classification ||
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
      !launch.sigma_inv_destination || !launch.sigma_classification ||
      !launch.kappa_classification || !launch.sigma_inv_classification ||
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

void launch_material_classification_header(material_classification_header *result,
                                           material_classification_header value,
                                           const stream &stream) {
  if (!result) throw std::invalid_argument("NVIDIA material classification header is null");
  material_classification_header_kernel<<<1, 1, 0,
      static_cast<cudaStream_t>(stream.opaque_handle())>>>(result, value);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA material classification header kernel");
}

void launch_material_classification_row(
    const unsigned char *status, size_t elements,
    material_classification_row_result *result,
    material_classification_row_result value, const stream &stream) {
  if (!status || !elements || !result)
    throw std::invalid_argument("NVIDIA material classification row is invalid");
  material_classification_row_kernel<<<1, 256, 0,
      static_cast<cudaStream_t>(stream.opaque_handle())>>>(status, elements, result, value);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA material classification row kernel");
}

} // namespace nvidia
} // namespace meep
