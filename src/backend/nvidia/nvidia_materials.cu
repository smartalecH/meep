/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_materials.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cfloat>
#include <climits>
#include <limits>
#include <stdexcept>
#include <string>

#include "backend/material_geometry_numeric.hpp"
#include "backend/nvidia/nvidia_initialization.hpp"

namespace meep {
namespace nvidia {
namespace {

enum { geometry_value_direct = 1u << 0 };

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

template <typename T>
bool compact_range(size_t offset, size_t count, size_t bytes) {
  return offset % alignof(T) == 0 && offset <= bytes &&
         count <= (bytes - offset) / sizeof(T);
}

bool checked_product(size_t left, size_t right, size_t &result) {
  if (left && right > std::numeric_limits<size_t>::max() / left) return false;
  result = left * right;
  return true;
}

size_t scalar_bytes(size_t elements, scalar_precision precision, const char *what) {
  const size_t width = precision == scalar_precision::f32
                           ? sizeof(float)
                           : precision == scalar_precision::f64 ? sizeof(double) : 0;
  if (!width) throw std::invalid_argument(std::string(what) + " precision is invalid");
  if (elements > std::numeric_limits<size_t>::max() / width)
    throw std::overflow_error(std::string(what) + " byte range overflows");
  return elements * width;
}

bool singular_nondiagonal_tensor(const double tensor[6]) {
  if (tensor[3] == 0.0 && tensor[4] == 0.0 && tensor[5] == 0.0) return false;
  const double determinant =
      tensor[0] * tensor[1] * tensor[2] - tensor[4] * tensor[1] * tensor[4] +
      2.0 * tensor[3] * tensor[5] * tensor[4] - tensor[3] * tensor[3] * tensor[2] -
      tensor[5] * tensor[5] * tensor[0];
  return determinant == 0.0;
}

void validate_common(const geometry_launch_common &common, const unsigned char *host_inputs,
                     size_t host_bytes, const char *what) {
  if (!common.destination || !common.point_count || !common.elements ||
      common.compact_input_bytes != host_bytes || !host_inputs ||
      common.dimensions < 0 || common.dimensions > 3 || common.component < 0 ||
      common.tensor_row < 0 || common.tensor_row > 2 || common.tensor_column < -1 ||
      common.tensor_column > 2 || common.property < 0 || common.property > 5 ||
      !(common.inva > 0.0) || !std::isfinite(common.inva) ||
      !(common.dt > 0.0) || !std::isfinite(common.dt) ||
      !compact_range<geometry_object_record>(common.object_offset, common.object_count,
                                             host_bytes) ||
      !compact_range<geometry_image_record>(common.image_offset, common.image_count,
                                            host_bytes) ||
      !compact_range<geometry_value_record>(common.value_offset, common.material_count,
                                            host_bytes) ||
      (common.absorber_count &&
       !compact_range<material_absorber_header>(common.absorber_header_offset,
                                                common.absorber_count, host_bytes)) ||
      common.default_material >= common.material_count)
    throw std::invalid_argument(std::string(what) + " common descriptor is invalid");
  const size_t alignment = common.precision == scalar_precision::f32 ? alignof(float)
                                                                     : alignof(double);
  const size_t bytes = scalar_bytes(common.elements, common.precision, what);
  const uintptr_t destination = reinterpret_cast<uintptr_t>(common.destination);
  if (destination % alignment || bytes > std::numeric_limits<uintptr_t>::max() - destination)
    throw std::invalid_argument(std::string(what) + " destination is invalid");
  if (common.compact_inputs) {
    const uintptr_t compact = reinterpret_cast<uintptr_t>(common.compact_inputs);
    if (host_bytes > std::numeric_limits<uintptr_t>::max() - compact ||
        (destination < compact + host_bytes && compact < destination + bytes))
      throw std::invalid_argument(std::string(what) + " destination aliases compact input");
  }
  size_t loop_count = 1;
  size_t maximum = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (common.loop_extent[axis] == 0 || common.strides[axis] < 0 ||
        common.axis_direction[axis] < 0 || common.axis_direction[axis] > 4 ||
        !std::isfinite(common.cell_center[axis]) ||
        !std::isfinite(common.cell_size[axis]) || common.cell_size[axis] < 0.0)
      throw std::invalid_argument(std::string(what) + " loop geometry is invalid");
    if (loop_count > std::numeric_limits<size_t>::max() / common.loop_extent[axis])
      throw std::overflow_error(std::string(what) + " loop count overflows");
    loop_count *= common.loop_extent[axis];
    const size_t step = common.loop_extent[axis] - 1;
    if (step && size_t(common.strides[axis]) >
                    (std::numeric_limits<size_t>::max() - maximum) / step)
      throw std::overflow_error(std::string(what) + " address range overflows");
    maximum += step * size_t(common.strides[axis]);
    if (common.loop_base_offset[axis] > std::numeric_limits<size_t>::max() - maximum)
      throw std::overflow_error(std::string(what) + " base address overflows");
    maximum += common.loop_base_offset[axis];
    for (int column = 0; column < 3; ++column)
      if (!std::isfinite(common.metric[3 * column + axis]))
        throw std::invalid_argument(std::string(what) + " metric is invalid");
  }
  if (loop_count != common.point_count || maximum >= common.elements)
    throw std::invalid_argument(std::string(what) + " loop exceeds destination");
  const geometry_object_record *objects = reinterpret_cast<const geometry_object_record *>(
      host_inputs + common.object_offset);
  for (size_t i = 0; i < common.object_count; ++i) {
    const geometry_object_record &object = objects[i];
    size_t vertex_values = 0;
    size_t prism_auxiliary_values = 0;
    const bool parameter_count_valid =
        (object.kind == 1 && object.parameter_count == 4) ||
        (object.kind == 2 && object.parameter_count == 29) ||
        (object.kind == 3 &&
         ((object.subtype == 0 && object.parameter_count == 25) ||
          (object.subtype == 1 && object.parameter_count == 28))) ||
        (object.kind == 4 && object.parameter_count == 4) ||
        (object.kind == 5 &&
         ((object.subtype == 0 && object.parameter_count == 9) ||
          (object.subtype == 1 && object.parameter_count == 19) ||
          (object.subtype == 2 && object.parameter_count == 10)));
    if (object.kind < 1 || object.kind > 5 || object.material >= common.material_count ||
        !parameter_count_valid ||
        !compact_range<double>(object.parameter_offset, object.parameter_count, host_bytes) ||
        !checked_product(object.vertex_count, size_t(3), vertex_values) ||
        !compact_range<double>(object.vertex_offset, vertex_values, host_bytes) ||
        !compact_range<double>(object.index_offset, object.index_count, host_bytes) ||
        !compact_range<double>(object.auxiliary_offset, object.auxiliary_count, host_bytes) ||
        !compact_range<geometry_triangle_record>(object.triangle_offset,
                                                 object.triangle_count, host_bytes) ||
        !compact_range<geometry_bvh_record>(object.bvh_offset, object.bvh_count, host_bytes) ||
        !compact_range<uint32_t>(object.face_id_offset, object.face_id_count, host_bytes) ||
        !std::isfinite(object.mesh_lengthscale) || object.mesh_lengthscale < 0.0 ||
        (object.kind == 1 &&
         (object.vertex_count < 4 || object.index_count < 12 || object.index_count % 3 ||
          !(object.mesh_lengthscale > 0.0) || object.triangle_count != object.index_count / 3 ||
          !object.bvh_count || object.face_id_count != object.triangle_count)) ||
        (object.kind == 2 &&
         (object.fixed_vertex_count < 3 || object.vertex_count != object.fixed_vertex_count ||
          !checked_product(object.fixed_vertex_count, size_t(9), prism_auxiliary_values) ||
          object.auxiliary_count != prism_auxiliary_values)) ||
        ((object.kind == 1 || object.kind == 2) && object.closed > 1))
      throw std::invalid_argument(std::string(what) + " object record is invalid");
    const double *parameters = reinterpret_cast<const double *>(host_inputs +
                                                                 object.parameter_offset);
    for (size_t parameter = 0; parameter < object.parameter_count; ++parameter) {
      const bool infinite_extent =
          (object.kind == 3 && parameter >= 12 && parameter < 15) ||
          (object.kind == 2 && parameter == 3) || (object.kind == 5 && parameter == 7);
      if (std::isnan(parameters[parameter]) ||
          (!std::isfinite(parameters[parameter]) &&
           (!infinite_extent || parameters[parameter] < 0.0)))
        throw std::invalid_argument(std::string(what) + " object parameter is invalid");
    }
    const double *vertices = reinterpret_cast<const double *>(host_inputs + object.vertex_offset);
    for (size_t value = 0; value < vertex_values; ++value)
      if (!std::isfinite(vertices[value]))
        throw std::invalid_argument(std::string(what) + " object vertex is invalid");
    const double *indices = reinterpret_cast<const double *>(host_inputs + object.index_offset);
    for (size_t index = 0; index < object.index_count; ++index)
      if (!std::isfinite(indices[index]) || indices[index] < 0.0 ||
          floor(indices[index]) != indices[index] || indices[index] >= object.vertex_count)
        throw std::invalid_argument(std::string(what) + " mesh index is invalid");
    const double *auxiliary = reinterpret_cast<const double *>(host_inputs +
                                                                object.auxiliary_offset);
    for (size_t value = 0; value < object.auxiliary_count; ++value)
      if (!std::isfinite(auxiliary[value]))
        throw std::invalid_argument(std::string(what) + " object auxiliary value is invalid");
    if (object.kind == 1) {
      const geometry_triangle_record *triangles =
          reinterpret_cast<const geometry_triangle_record *>(host_inputs + object.triangle_offset);
      for (size_t triangle = 0; triangle < object.triangle_count; ++triangle) {
        for (int vertex = 0; vertex < 3; ++vertex)
          if (triangles[triangle].vertex[vertex] >= object.vertex_count)
            throw std::invalid_argument(std::string(what) + " triangle vertex is invalid");
        for (int axis = 0; axis < 3; ++axis)
          if (!std::isfinite(triangles[triangle].normal[axis]) ||
              !std::isfinite(triangles[triangle].low[axis]) ||
              !std::isfinite(triangles[triangle].high[axis]) ||
              triangles[triangle].low[axis] > triangles[triangle].high[axis])
            throw std::invalid_argument(std::string(what) + " triangle record is invalid");
      }
      const geometry_bvh_record *nodes = reinterpret_cast<const geometry_bvh_record *>(
          host_inputs + object.bvh_offset);
      const uint32_t *faces = reinterpret_cast<const uint32_t *>(host_inputs +
                                                                 object.face_id_offset);
      for (size_t node = 0; node < object.bvh_count; ++node) {
        const geometry_bvh_record &record = nodes[node];
        for (int axis = 0; axis < 3; ++axis)
          if (!std::isfinite(record.low[axis]) || !std::isfinite(record.high[axis]) ||
              record.low[axis] > record.high[axis])
            throw std::invalid_argument(std::string(what) + " BVH bounds are invalid");
        if (record.escape <= node || record.escape > object.bvh_count ||
            (record.leaf &&
             (record.left != UINT32_MAX || record.right != UINT32_MAX || !record.face_count ||
              record.first_face > object.face_id_count ||
              record.face_count > object.face_id_count - record.first_face)) ||
            (!record.leaf &&
             (record.left <= node || record.left >= object.bvh_count ||
              record.right <= node || record.right >= object.bvh_count ||
              record.face_count)))
          throw std::invalid_argument(std::string(what) + " BVH node is invalid");
      }
      for (size_t face = 0; face < object.face_id_count; ++face)
        if (faces[face] >= object.triangle_count)
          throw std::invalid_argument(std::string(what) + " BVH face identity is invalid");
    }
    for (int axis = 0; axis < 3; ++axis)
      if (std::isnan(object.low[axis]) || std::isnan(object.high[axis]) ||
          object.low[axis] > object.high[axis])
        throw std::invalid_argument(std::string(what) + " object bounds are invalid");
  }
  const geometry_image_record *images = reinterpret_cast<const geometry_image_record *>(
      host_inputs + common.image_offset);
  for (size_t i = 0; i < common.image_count; ++i) {
    const geometry_image_record &image = images[i];
    if (image.object >= common.object_count ||
        (i && (image.ordinal <= images[i - 1].ordinal ||
               image.precedence > images[i - 1].precedence)))
      throw std::invalid_argument(std::string(what) + " image object is invalid");
    for (int axis = 0; axis < 3; ++axis)
      if (!std::isfinite(image.shift[axis]) || std::isnan(image.low[axis]) ||
          std::isnan(image.high[axis]) || image.low[axis] > image.high[axis] ||
          image.low[axis] != objects[image.object].low[axis] + image.shift[axis] ||
          image.high[axis] != objects[image.object].high[axis] + image.shift[axis])
        throw std::invalid_argument(std::string(what) + " image bounds are invalid");
  }
  const geometry_value_record *values = reinterpret_cast<const geometry_value_record *>(
      host_inputs + common.value_offset);
  for (size_t i = 0; i < common.material_count; ++i) {
    const geometry_value_record &value = values[i];
    if (value.kind < geometry_value_kind::constant ||
        value.kind > geometry_value_kind::grid_condinv ||
        (value.flags & ~uint32_t(geometry_value_direct)) ||
        (value.kind != geometry_value_kind::constant && value.flags) ||
        (value.kind == geometry_value_kind::file_epsilon && value.overlap_kind) ||
        ((value.kind == geometry_value_kind::grid_tensor ||
          value.kind == geometry_value_kind::grid_linear ||
          value.kind == geometry_value_kind::grid_condinv) &&
         value.overlap_kind > 3u) ||
        !std::isfinite(value.beta) || !std::isfinite(value.eta) ||
        !std::isfinite(value.damping) || !std::isfinite(value.projection_offset) ||
        !std::isfinite(value.value_1) || !std::isfinite(value.value_2))
      throw std::invalid_argument(std::string(what) + " material value is invalid");
    for (int j = 0; j < 6; ++j)
      if (!std::isfinite(value.tensor_1[j]) || !std::isfinite(value.tensor_2[j]))
        throw std::invalid_argument(std::string(what) + " material tensor is invalid");
    if (((value.kind == geometry_value_kind::constant &&
          !(value.flags & geometry_value_direct)) ||
         value.kind == geometry_value_kind::grid_tensor) &&
        (singular_nondiagonal_tensor(value.tensor_1) ||
         (value.kind == geometry_value_kind::grid_tensor &&
          singular_nondiagonal_tensor(value.tensor_2))))
      throw std::invalid_argument(std::string(what) + " material tensor is singular");
    if (value.kind != geometry_value_kind::constant) {
      size_t samples = 1;
      for (int axis = 0; axis < 3; ++axis) {
        if (!value.dimensions[axis] || value.dimensions[axis] > uint32_t(INT_MAX) ||
            samples > std::numeric_limits<size_t>::max() / value.dimensions[axis])
          throw std::invalid_argument(std::string(what) + " sample dimensions are invalid");
        samples *= value.dimensions[axis];
      }
      if (samples != value.sample_count ||
          !compact_range<double>(value.sample_offset, value.sample_count, host_bytes))
        throw std::invalid_argument(std::string(what) + " sample range is invalid");
    }
  }
  if (common.absorber_count)
    validate_material_absorber_headers(host_inputs, host_bytes,
                                       common.absorber_header_offset,
                                       common.absorber_count);
}

__device__ int mirror_index(int i, int n) {
  const int64_t wide_i = i, wide_n = n;
  return int(i >= n ? 2 * wide_n - 1 - wide_i : (i < 0 ? -1 - wide_i : wide_i));
}

__device__ double interpolate(const geometry_value_record &value,
                              const unsigned char *compact, double x, double y, double z) {
  x = x < 0.0 ? -x : (x > 1.0 ? 1.0 - x : x);
  y = y < 0.0 ? -y : (y > 1.0 ? 1.0 - y : y);
  z = z < 0.0 ? -z : (z > 1.0 ? 1.0 - z : z);
  const int nx = int(value.dimensions[0]), ny = int(value.dimensions[1]);
  const int nz = int(value.dimensions[2]);
  const int x1 = mirror_index(int(x * nx), nx), y1 = mirror_index(int(y * ny), ny);
  const int z1 = mirror_index(int(z * nz), nz);
  double dx = x * nx - x1 - 0.5, dy = y * ny - y1 - 0.5, dz = z * nz - z1 - 0.5;
  const int x2 = mirror_index(dx >= 0.0 ? x1 + 1 : x1 - 1, nx);
  const int y2 = mirror_index(dy >= 0.0 ? y1 + 1 : y1 - 1, ny);
  const int z2 = mirror_index(dz >= 0.0 ? z1 + 1 : z1 - 1, nz);
  dx = fabs(dx); dy = fabs(dy); dz = fabs(dz);
  const double *samples = reinterpret_cast<const double *>(compact + value.sample_offset);
#define GEOMETRY_SAMPLE(ix, iy, iz) \
  samples[(size_t(ix) * size_t(ny) + size_t(iy)) * size_t(nz) + size_t(iz)]
  const double result =
      (((GEOMETRY_SAMPLE(x1, y1, z1) * (1.0 - dx) + GEOMETRY_SAMPLE(x2, y1, z1) * dx) *
            (1.0 - dy) +
        (GEOMETRY_SAMPLE(x1, y2, z1) * (1.0 - dx) + GEOMETRY_SAMPLE(x2, y2, z1) * dx) *
            dy) *
           (1.0 - dz) +
       ((GEOMETRY_SAMPLE(x1, y1, z2) * (1.0 - dx) + GEOMETRY_SAMPLE(x2, y1, z2) * dx) *
            (1.0 - dy) +
        (GEOMETRY_SAMPLE(x1, y2, z2) * (1.0 - dx) + GEOMETRY_SAMPLE(x2, y2, z2) * dx) *
            dy) *
           dz);
#undef GEOMETRY_SAMPLE
  return result;
}

__device__ void point_for(const geometry_launch_common &common, size_t point,
                          size_t &destination_index,
                          material_geometry_numeric::vector &position,
                          double physical_by_direction[5]) {
  size_t remaining = point;
  const size_t i2 = remaining % common.loop_extent[2];
  remaining /= common.loop_extent[2];
  const size_t i1 = remaining % common.loop_extent[1];
  const size_t i0 = remaining / common.loop_extent[1];
  const size_t index[3] = {i0, i1, i2};
  position = {0.0, 0.0, 0.0};
  destination_index = 0;
  for (int axis = 0; axis < 3; ++axis) {
    destination_index += common.loop_base_offset[axis] +
                         index[axis] * size_t(common.strides[axis]);
    const double coordinate =
        0.5 * (double(common.loop_begin[axis]) + 2.0 * double(index[axis]) +
               double(common.evaluation_shift[axis])) * common.inva;
    const int direction = common.axis_direction[axis];
    if (direction >= 0 && direction < 5) physical_by_direction[direction] = coordinate;
    if (direction == 0 || direction == 3) position.x = coordinate;
    else if (direction == 1) position.y = coordinate;
    else if (direction == 2) position.z = coordinate;
  }
  if (common.dimensions == 0) position.x = position.y = 0.0;
  else if (common.dimensions == 1) position.z = 0.0;
  else if (common.dimensions == 3) position.y = 0.0;
}

__device__ double absorber_conductivity(const geometry_launch_common &common,
                                        const double physical[5]) {
  double result = 0.0;
  const material_absorber_header *headers =
      reinterpret_cast<const material_absorber_header *>(
          common.compact_inputs + common.absorber_header_offset);
  for (size_t absorber = 0; absorber < common.absorber_count; ++absorber) {
    const material_absorber_header header = headers[absorber];
    const double *samples = reinterpret_cast<const double *>(
        common.compact_inputs + header.sample_offset);
    const double coordinate = physical[header.direction];
    const double half_cell =
        0.5 * (header.direction == 0 || header.direction == 3
                   ? common.cell_size[0]
                   : header.direction == 1 ? common.cell_size[1] : common.cell_size[2]);
    const bool high = header.side == 0;
    const double edge = high ? half_cell - header.thickness : header.thickness - half_cell;
    if ((high && coordinate >= edge) || (!high && coordinate <= edge)) {
      const size_t intervals = size_t(header.sample_count - 1);
      const double u = double(intervals) *
                       (high ? coordinate - edge : edge - coordinate) / header.thickness;
      const int sample = int(u);
      if (sample >= int(intervals)) result += samples[intervals];
      else {
        const double fraction = u - sample;
        result += samples[sample] * (1.0 - fraction) + samples[sample + 1] * fraction;
      }
    }
  }
  return result;
}

__device__ double geometry_coordinate(material_geometry_numeric::vector value, int axis) {
  return axis == 0 ? value.x : axis == 1 ? value.y : value.z;
}

__device__ bool bvh_ray_interval(const geometry_bvh_record &node,
                                 material_geometry_numeric::vector origin,
                                 material_geometry_numeric::vector direction,
                                 double maximum) {
  double low = 0.0, high = maximum;
  for (int axis = 0; axis < 3; ++axis) {
    const double p = geometry_coordinate(origin, axis);
    const double d = geometry_coordinate(direction, axis);
    if (d == 0.0) {
      if (p < node.low[axis] || p > node.high[axis]) return false;
      continue;
    }
    double first = (node.low[axis] - p) / d;
    double second = (node.high[axis] - p) / d;
    if (first > second) {
      const double swap = first;
      first = second;
      second = swap;
    }
    low = first > low ? first : low;
    high = second < high ? second : high;
    if (low > high) return false;
  }
  return high > 0.0;
}

__device__ bool mesh_hit(const geometry_object_record &object,
                         const unsigned char *compact, uint32_t face,
                         material_geometry_numeric::vector point,
                         material_geometry_numeric::vector direction, double &distance) {
  const geometry_triangle_record *triangles =
      reinterpret_cast<const geometry_triangle_record *>(compact + object.triangle_offset);
  const double *vertices = reinterpret_cast<const double *>(compact + object.vertex_offset);
  const geometry_triangle_record &triangle = triangles[face];
  const material_geometry_numeric::vector a = {
      vertices[3 * triangle.vertex[0]], vertices[3 * triangle.vertex[0] + 1],
      vertices[3 * triangle.vertex[0] + 2]};
  const material_geometry_numeric::vector b = {
      vertices[3 * triangle.vertex[1]], vertices[3 * triangle.vertex[1] + 1],
      vertices[3 * triangle.vertex[1] + 2]};
  const material_geometry_numeric::vector c = {
      vertices[3 * triangle.vertex[2]], vertices[3 * triangle.vertex[2] + 1],
      vertices[3 * triangle.vertex[2] + 2]};
  return material_geometry_numeric::ray_triangle(
      point, direction, a, b, c,
      1e-12 * object.mesh_lengthscale * object.mesh_lengthscale, distance);
}

__device__ int mesh_raw_crossings(const geometry_object_record &object,
                                  const unsigned char *compact,
                                  material_geometry_numeric::vector point,
                                  material_geometry_numeric::vector direction) {
  const geometry_bvh_record *nodes =
      reinterpret_cast<const geometry_bvh_record *>(compact + object.bvh_offset);
  const uint32_t *faces = reinterpret_cast<const uint32_t *>(compact + object.face_id_offset);
  const double epsilon = 1e-12 * object.mesh_lengthscale;
  int crossings = 0;
  uint32_t node_index = 0;
  while (node_index < object.bvh_count) {
    const geometry_bvh_record &node = nodes[node_index];
    if (!bvh_ray_interval(node, point, direction, DBL_MAX)) {
      node_index = node.escape;
      continue;
    }
    if (!node.leaf) {
      node_index = node.left;
      continue;
    }
    for (uint64_t i = 0; i < node.face_count; ++i) {
      double distance = 0.0;
      if (mesh_hit(object, compact, faces[node.first_face + i], point, direction, distance) &&
          distance > epsilon)
        ++crossings;
    }
    node_index = node.escape;
  }
  return crossings;
}

__device__ bool mesh_next_crossing(const geometry_object_record &object,
                                   const unsigned char *compact,
                                   material_geometry_numeric::vector point,
                                   material_geometry_numeric::vector direction,
                                   double after, double &next) {
  const geometry_bvh_record *nodes =
      reinterpret_cast<const geometry_bvh_record *>(compact + object.bvh_offset);
  const uint32_t *faces = reinterpret_cast<const uint32_t *>(compact + object.face_id_offset);
  next = DBL_MAX;
  bool found = false;
  uint32_t node_index = 0;
  while (node_index < object.bvh_count) {
    const geometry_bvh_record &node = nodes[node_index];
    if (!bvh_ray_interval(node, point, direction, next)) {
      node_index = node.escape;
      continue;
    }
    if (!node.leaf) {
      node_index = node.left;
      continue;
    }
    for (uint64_t i = 0; i < node.face_count; ++i) {
      double distance = 0.0;
      if (mesh_hit(object, compact, faces[node.first_face + i], point, direction, distance) &&
          distance > after && distance < next)
        next = distance, found = true;
    }
    node_index = node.escape;
  }
  return found;
}

__device__ int mesh_unique_crossings(const geometry_object_record &object,
                                     const unsigned char *compact,
                                     material_geometry_numeric::vector point,
                                     material_geometry_numeric::vector direction) {
  const double forward = 1e-12 * object.mesh_lengthscale;
  const double duplicate = 1e-10 * object.mesh_lengthscale;
  double cursor = forward;
  int count = 0;
  for (size_t iteration = 0; iteration < object.triangle_count; ++iteration) {
    double next = 0.0;
    if (!mesh_next_crossing(object, compact, point, direction, cursor, next)) break;
    ++count;
    cursor = next + duplicate;
  }
  return count;
}

__device__ bool mesh_contains(const geometry_object_record &object,
                              const unsigned char *compact,
                              material_geometry_numeric::vector point) {
  if (!object.closed) return false;
  const material_geometry_numeric::vector directions[3] = {
      {0.57735026918962576, 0.57735026918962576, 0.57735026918962576},
      {0.80178372573727319, 0.53452248382484879, 0.26726124191242440},
      {0.12309149097933272, 0.49236596391733088, 0.86164043685532904}};
  const int raw = mesh_raw_crossings(object, compact, point, directions[0]);
  const int first = mesh_unique_crossings(object, compact, point, directions[0]);
  if (raw == first) return first % 2 == 1;
  int votes = first % 2;
  votes += mesh_unique_crossings(object, compact, point, directions[1]) % 2;
  votes += mesh_unique_crossings(object, compact, point, directions[2]) % 2;
  return votes >= 2;
}

__device__ bool image_contains(const geometry_launch_common &common,
                               const geometry_image_record &image,
                               material_geometry_numeric::vector point) {
  if (point.x < image.low[0] || point.x > image.high[0] ||
      point.y < image.low[1] || point.y > image.high[1] ||
      point.z < image.low[2] || point.z > image.high[2])
    return false;
  const geometry_object_record *objects = reinterpret_cast<const geometry_object_record *>(
      common.compact_inputs + common.object_offset);
  const geometry_object_record &object = objects[image.object];
  const double *parameters = reinterpret_cast<const double *>(
      common.compact_inputs + object.parameter_offset);
  const double *vertices = reinterpret_cast<const double *>(
      common.compact_inputs + object.vertex_offset);
  const double *indices = reinterpret_cast<const double *>(
      common.compact_inputs + object.index_offset);
  const double *auxiliary = reinterpret_cast<const double *>(
      common.compact_inputs + object.auxiliary_offset);
  const material_geometry_numeric::vector local = {
      point.x - image.shift[0], point.y - image.shift[1], point.z - image.shift[2]};
  if (object.kind == 1) return mesh_contains(object, common.compact_inputs, local);
  return material_geometry_numeric::contains(
      int(object.kind), int(object.subtype), parameters, vertices, object.vertex_count,
      indices, object.index_count, auxiliary, object.mesh_lengthscale,
      common.metric, local, object.closed != 0);
}

__device__ uint32_t selected_material(const geometry_launch_common &common,
                                      material_geometry_numeric::vector point,
                                      size_t &winning_image) {
  const geometry_image_record *images = reinterpret_cast<const geometry_image_record *>(
      common.compact_inputs + common.image_offset);
  const geometry_object_record *objects = reinterpret_cast<const geometry_object_record *>(
      common.compact_inputs + common.object_offset);
  for (size_t i = 0; i < common.image_count; ++i)
    if (image_contains(common, images[i], point)) {
      winning_image = i;
      return objects[images[i].object].material;
    }
  winning_image = size_t(-1);
  return common.default_material;
}

__device__ material_geometry_numeric::vector normalized_global(
    const geometry_launch_common &common, material_geometry_numeric::vector point) {
  return {common.cell_size[0] == 0.0
              ? 0.0
              : 0.5 + (point.x - common.cell_center[0]) / common.cell_size[0],
          common.cell_size[1] == 0.0
              ? 0.0
              : 0.5 + (point.y - common.cell_center[1]) / common.cell_size[1],
          common.cell_size[2] == 0.0
              ? 0.0
              : 0.5 + (point.z - common.cell_center[2]) / common.cell_size[2]};
}

__device__ double grid_weight(const geometry_launch_common &common, uint32_t material,
                              size_t winning_image,
                              material_geometry_numeric::vector point) {
  const geometry_value_record *values = reinterpret_cast<const geometry_value_record *>(
      common.compact_inputs + common.value_offset);
  const geometry_image_record *images = reinterpret_cast<const geometry_image_record *>(
      common.compact_inputs + common.image_offset);
  const geometry_object_record *objects = reinterpret_cast<const geometry_object_record *>(
      common.compact_inputs + common.object_offset);
  const geometry_value_record &winner = values[material];
  double product = 1.0, minimum = 1.0, sum = 0.0, default_value = 0.0;
  size_t count = 0;
  bool exhausted = true;
  if (winning_image != size_t(-1)) {
    for (size_t i = winning_image; i < common.image_count; ++i) {
      if (!image_contains(common, images[i], point)) continue;
      const geometry_object_record &object = objects[images[i].object];
      const geometry_value_record &value = values[object.material];
      if (value.kind != geometry_value_kind::grid_tensor &&
          value.kind != geometry_value_kind::grid_linear &&
          value.kind != geometry_value_kind::grid_condinv) {
        exhausted = false;
        break;
      }
      const double *parameters = reinterpret_cast<const double *>(
          common.compact_inputs + object.parameter_offset);
      material_geometry_numeric::vector local = {
          point.x - images[i].shift[0], point.y - images[i].shift[1],
          point.z - images[i].shift[2]};
      local = material_geometry_numeric::object_coordinates(int(object.kind), parameters, local);
      const double sample = interpolate(value, common.compact_inputs, local.x, local.y, local.z);
      if (winner.overlap_kind == 3u) {
        default_value = sample;
        exhausted = false;
        break;
      }
      minimum = sample < minimum ? sample : minimum;
      product *= sample;
      sum += sample;
      ++count;
    }
  }
  const geometry_value_record &default_value_record = values[common.default_material];
  const bool default_grid = default_value_record.kind == geometry_value_kind::grid_tensor ||
                            default_value_record.kind == geometry_value_kind::grid_linear ||
                            default_value_record.kind == geometry_value_kind::grid_condinv;
  if (exhausted && default_grid) {
    const material_geometry_numeric::vector global = normalized_global(common, point);
    const double sample = interpolate(default_value_record, common.compact_inputs,
                                      global.x, global.y, global.z);
    if (!count) default_value = sample;
    minimum = sample < minimum ? sample : minimum;
    product *= sample;
    sum += sample;
    ++count;
  }
  double result = winner.overlap_kind == 0u
                      ? minimum
                      : winner.overlap_kind == 1u
                            ? product
                            : winner.overlap_kind == 2u ? sum / double(count) : default_value;
  result += winner.projection_offset;
  if (winner.beta == 0.0) return result;
  if (result == winner.eta) return 0.5;
  const double t = tanh(winner.beta * winner.eta);
  return (t + tanh(winner.beta * (result - winner.eta))) /
         (t + tanh(winner.beta * (1.0 - winner.eta)));
}

__device__ double inverse_member(const double tensor[6], int row, int column) {
  const double m00 = tensor[0], m11 = tensor[1], m22 = tensor[2];
  const double m01 = tensor[3], m02 = tensor[4], m12 = tensor[5];
  if (m01 == 0.0 && m02 == 0.0 && m12 == 0.0)
    return row == column ? 1.0 / tensor[row] : 0.0;
  const double inverse_det = 1.0 /
      (m00 * m11 * m22 - m02 * m11 * m02 + 2.0 * m01 * m12 * m02 -
       m01 * m01 * m22 - m12 * m12 * m00);
  if (row == 0 && column == 0) return inverse_det * (m11 * m22 - m12 * m12);
  if (row == 1 && column == 1) return inverse_det * (m00 * m22 - m02 * m02);
  if (row == 2 && column == 2) return inverse_det * (m11 * m00 - m01 * m01);
  if ((row == 0 && column == 2) || (row == 2 && column == 0))
    return inverse_det * (m01 * m12 - m11 * m02);
  if ((row == 0 && column == 1) || (row == 1 && column == 0))
    return inverse_det * (m12 * m02 - m01 * m22);
  return inverse_det * (m01 * m02 - m00 * m12);
}

__device__ double value_at(const geometry_launch_common &common,
                           material_geometry_numeric::vector point,
                           const double physical[5]) {
  size_t image = size_t(-1);
  const uint32_t material = selected_material(common, point, image);
  const geometry_value_record *values = reinterpret_cast<const geometry_value_record *>(
      common.compact_inputs + common.value_offset);
  const geometry_value_record &value = values[material];
  if (common.property == 1 || common.property == 2) {
    double conductivity = value.value_1;
    if (value.kind == geometry_value_kind::grid_linear ||
        value.kind == geometry_value_kind::grid_condinv) {
      const double u = grid_weight(common, material, image, point);
      conductivity = value.value_1 * (1.0 - u) + value.value_2 * u +
                     u * (1.0 - u) * value.damping;
    }
    conductivity += absorber_conductivity(common, physical);
    if (common.property == 1) return conductivity;
    if (common.logical_single) {
      const float stored = float(conductivity);
      return double(float(1.0 / (1.0 + double(stored) * common.dt * 0.5)));
    }
    return 1.0 / (1.0 + conductivity * common.dt * 0.5);
  }
  if (value.kind == geometry_value_kind::constant)
    return value.flags & geometry_value_direct
               ? value.value_1
               : inverse_member(value.tensor_1, common.tensor_row, common.tensor_column);
  material_geometry_numeric::vector coordinate = normalized_global(common, point);
  double u = 0.0;
  if (value.kind == geometry_value_kind::file_epsilon)
    u = interpolate(value, common.compact_inputs, coordinate.x, coordinate.y, coordinate.z);
  else
    u = grid_weight(common, material, image, point);
  if (value.kind == geometry_value_kind::file_epsilon)
    return common.tensor_row == common.tensor_column ? 1.0 / u : 0.0;
  if (value.kind == geometry_value_kind::grid_tensor) {
    double tensor[6];
    for (int i = 0; i < 6; ++i)
      tensor[i] = value.tensor_1[i] * (1.0 - u) + value.tensor_2[i] * u;
    return inverse_member(tensor, common.tensor_row, common.tensor_column);
  }
  const double linear = value.value_1 * (1.0 - u) + value.value_2 * u;
  return linear;
}

__device__ void rotate_tensor(const double input[6], const double rotation[3][3],
                              double output[6]) {
  double matrix[3][3] = {{input[0], input[3], input[4]},
                         {input[3], input[1], input[5]},
                         {input[4], input[5], input[2]}};
  double intermediate[3][3] = {};
  double rotated[3][3] = {};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) intermediate[i][j] += matrix[i][k] * rotation[k][j];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) rotated[i][j] += rotation[k][i] * intermediate[k][j];
  output[0] = rotated[0][0]; output[1] = rotated[1][1]; output[2] = rotated[2][2];
  output[3] = rotated[0][1]; output[4] = rotated[0][2]; output[5] = rotated[1][2];
}

__device__ double analytic_value(const geometry_launch_common &common,
                                 const geometry_analytic_record &job) {
  const geometry_value_record *values = reinterpret_cast<const geometry_value_record *>(
      common.compact_inputs + common.value_offset);
  double rotation[3][3] = {};
  rotation[0][0] = job.normal[0]; rotation[1][0] = job.normal[1];
  rotation[2][0] = job.normal[2];
  if (fabs(job.normal[0]) > 1e-2 || fabs(job.normal[1]) > 1e-2) {
    rotation[0][2] = job.normal[1]; rotation[1][2] = -job.normal[0];
  }
  else {
    rotation[1][2] = -job.normal[2]; rotation[2][2] = job.normal[1];
  }
  const double scale = 1.0 / sqrt(rotation[0][2] * rotation[0][2] +
                                  rotation[1][2] * rotation[1][2] +
                                  rotation[2][2] * rotation[2][2]);
  for (int i = 0; i < 3; ++i) rotation[i][2] *= scale;
  rotation[0][1] = rotation[1][2] * rotation[2][0] - rotation[2][2] * rotation[1][0];
  rotation[1][1] = rotation[2][2] * rotation[0][0] - rotation[0][2] * rotation[2][0];
  rotation[2][1] = rotation[0][2] * rotation[1][0] - rotation[1][2] * rotation[0][0];
  double front[6], behind[6];
  rotate_tensor(values[job.front_material].tensor_1, rotation, front);
  rotate_tensor(values[job.behind_material].tensor_1, rotation, behind);
  const double fill = job.fill, back = 1.0 - fill;
  double delta[6];
  delta[0] = fill * (-1.0 / front[0]) + back * (-1.0 / behind[0]);
  delta[1] = fill * (front[1] - front[3] * front[3] / front[0]) +
             back * (behind[1] - behind[3] * behind[3] / behind[0]);
  delta[2] = fill * (front[2] - front[4] * front[4] / front[0]) +
             back * (behind[2] - behind[4] * behind[4] / behind[0]);
  delta[3] = fill * (front[3] / front[0]) + back * (behind[3] / behind[0]);
  delta[4] = fill * (front[4] / front[0]) + back * (behind[4] / behind[0]);
  delta[5] = fill * (front[5] - front[4] * front[3] / front[0]) +
             back * (behind[5] - behind[4] * behind[3] / behind[0]);
  double mean[6];
  mean[0] = -1.0 / delta[0];
  mean[1] = delta[1] - delta[3] * delta[3] / delta[0];
  mean[2] = delta[2] - delta[4] * delta[4] / delta[0];
  mean[3] = -delta[3] / delta[0];
  mean[4] = -delta[4] / delta[0];
  mean[5] = delta[5] - delta[4] * delta[3] / delta[0];
  double transpose[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) transpose[i][j] = rotation[j][i];
  double unrotated[6];
  rotate_tensor(mean, transpose, unrotated);
  return inverse_member(unrotated, common.tensor_row, common.tensor_column);
}

template <typename T>
__device__ void store_value(const geometry_launch_common &common, size_t destination,
                            double value) {
  if (common.logical_single) value = double(float(value));
  static_cast<T *>(common.destination)[destination] = T(value);
}

template <typename T>
__global__ void geometry_bulk_kernel(geometry_bulk_launch launch) {
  const size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread >= launch.count) return;
  const size_t point = launch.first_point + thread;
  size_t destination = 0;
  material_geometry_numeric::vector position;
  double physical[5] = {};
  point_for(launch.common, point, destination, position, physical);
  store_value<T>(launch.common, destination, value_at(launch.common, position, physical));
}

template <typename T>
__global__ void geometry_analytic_kernel(geometry_analytic_launch launch) {
  const size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread >= launch.count) return;
  const geometry_analytic_record *jobs = reinterpret_cast<const geometry_analytic_record *>(
      launch.common.compact_inputs + launch.job_offset);
  const geometry_analytic_record job = jobs[thread];
  size_t destination = 0;
  material_geometry_numeric::vector ignored;
  double physical[5] = {};
  point_for(launch.common, job.point, destination, ignored, physical);
  store_value<T>(launch.common, destination, analytic_value(launch.common, job));
}

template <typename T>
__global__ void geometry_patch_kernel(geometry_patch_launch launch) {
  const size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread >= launch.count) return;
  const geometry_patch_record *patches = reinterpret_cast<const geometry_patch_record *>(
      launch.common.compact_inputs + launch.patch_offset);
  const geometry_patch_record patch = patches[thread];
  size_t destination = 0;
  material_geometry_numeric::vector ignored;
  double physical[5] = {};
  point_for(launch.common, patch.point, destination, ignored, physical);
  store_value<T>(launch.common, destination, patch.value);
}

template <typename T>
void launch_bulk_typed(const geometry_bulk_launch &launch, const stream &execution) {
  const size_t count = launch.count;
  const dim3 grid(launch_blocks(count, "NVIDIA material geometry"));
  const cudaStream_t cuda_stream = static_cast<cudaStream_t>(execution.opaque_handle());
  geometry_bulk_kernel<T><<<grid, 256, 0, cuda_stream>>>(launch);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA material geometry kernel");
}

template <typename T>
void launch_analytic_typed(const geometry_analytic_launch &launch,
                           const stream &execution) {
  const dim3 grid(launch_blocks(launch.count, "NVIDIA material geometry"));
  const cudaStream_t cuda_stream = static_cast<cudaStream_t>(execution.opaque_handle());
  geometry_analytic_kernel<T><<<grid, 256, 0, cuda_stream>>>(launch);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA material geometry kernel");
}

template <typename T>
void launch_patch_typed(const geometry_patch_launch &launch, const stream &execution) {
  const dim3 grid(launch_blocks(launch.count, "NVIDIA material geometry"));
  const cudaStream_t cuda_stream = static_cast<cudaStream_t>(execution.opaque_handle());
  geometry_patch_kernel<T><<<grid, 256, 0, cuda_stream>>>(launch);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA material geometry kernel");
}

} // namespace

void validate_geometry_bulk_launch(const geometry_bulk_launch &launch,
                                   const unsigned char *host_inputs, size_t host_bytes) {
  validate_common(launch.common, host_inputs, host_bytes, "NVIDIA material geometry bulk");
  if (!launch.count || launch.first_point > launch.common.point_count ||
      launch.count > launch.common.point_count - launch.first_point)
    throw std::invalid_argument("NVIDIA material geometry bulk span is invalid");
}

void validate_geometry_analytic_launch(const geometry_analytic_launch &launch,
                                       const unsigned char *host_inputs, size_t host_bytes) {
  validate_common(launch.common, host_inputs, host_bytes, "NVIDIA material geometry analytic");
  if (!launch.count ||
      !compact_range<geometry_analytic_record>(launch.job_offset, launch.count, host_bytes))
    throw std::invalid_argument("NVIDIA material geometry analytic jobs are invalid");
  const geometry_analytic_record *jobs = reinterpret_cast<const geometry_analytic_record *>(
      host_inputs + launch.job_offset);
  uint64_t previous = 0;
  for (size_t i = 0; i < launch.count; ++i) {
    if (jobs[i].point >= launch.common.point_count ||
        jobs[i].front_material >= launch.common.material_count ||
        jobs[i].behind_material >= launch.common.material_count ||
        !(jobs[i].fill > 0.0 && jobs[i].fill < 1.0) || !std::isfinite(jobs[i].fill) ||
        (i && jobs[i].point <= previous))
      throw std::invalid_argument("NVIDIA material geometry analytic record is invalid");
    double norm2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(jobs[i].normal[axis]))
        throw std::invalid_argument("NVIDIA material geometry analytic normal is invalid");
      norm2 += jobs[i].normal[axis] * jobs[i].normal[axis];
    }
    if (!(norm2 > 0.0))
      throw std::invalid_argument("NVIDIA material geometry analytic normal is zero");
    previous = jobs[i].point;
  }
}

void validate_geometry_patch_launch(const geometry_patch_launch &launch,
                                    const unsigned char *host_inputs, size_t host_bytes) {
  validate_common(launch.common, host_inputs, host_bytes, "NVIDIA material geometry patch");
  if (!launch.count || !compact_range<geometry_patch_record>(
                           launch.patch_offset, launch.count, host_bytes))
    throw std::invalid_argument("NVIDIA material geometry patches are invalid");
  const geometry_patch_record *patches = reinterpret_cast<const geometry_patch_record *>(
      host_inputs + launch.patch_offset);
  uint64_t previous = 0;
  for (size_t i = 0; i < launch.count; ++i) {
    if (patches[i].point >= launch.common.point_count || !std::isfinite(patches[i].value) ||
        (i && patches[i].point <= previous))
      throw std::invalid_argument("NVIDIA material geometry patch record is invalid");
    previous = patches[i].point;
  }
}

void launch_material_geometry_bulk(const geometry_bulk_launch &launch,
                                   const stream &execution) {
  if (!launch.count) return;
  if (testing::consume_failure_for_testing(testing::failure_point::material_geometry_bulk_launch))
    throw std::runtime_error("injected NVIDIA material geometry bulk launch failure");
  if (launch.common.precision == scalar_precision::f32)
    launch_bulk_typed<float>(launch, execution);
  else if (launch.common.precision == scalar_precision::f64)
    launch_bulk_typed<double>(launch, execution);
  else
    throw std::invalid_argument("NVIDIA material geometry precision is invalid");
}

void launch_material_geometry_analytic(const geometry_analytic_launch &launch,
                                       const stream &execution) {
  if (!launch.count) return;
  if (testing::consume_failure_for_testing(
          testing::failure_point::material_geometry_analytic_launch))
    throw std::runtime_error("injected NVIDIA material geometry analytic launch failure");
  if (launch.common.precision == scalar_precision::f32)
    launch_analytic_typed<float>(launch, execution);
  else if (launch.common.precision == scalar_precision::f64)
    launch_analytic_typed<double>(launch, execution);
  else
    throw std::invalid_argument("NVIDIA material geometry precision is invalid");
}

void launch_material_geometry_patch(const geometry_patch_launch &launch,
                                    const stream &execution) {
  if (!launch.count) return;
  if (testing::consume_failure_for_testing(testing::failure_point::material_geometry_patch_launch))
    throw std::runtime_error("injected NVIDIA material geometry patch launch failure");
  if (launch.common.precision == scalar_precision::f32)
    launch_patch_typed<float>(launch, execution);
  else if (launch.common.precision == scalar_precision::f64)
    launch_patch_typed<double>(launch, execution);
  else
    throw std::invalid_argument("NVIDIA material geometry precision is invalid");
}

} // namespace nvidia
} // namespace meep
