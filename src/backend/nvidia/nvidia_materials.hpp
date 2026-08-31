/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_MATERIALS_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_MATERIALS_HPP

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

#include "backend/nvidia/nvidia_step.hpp"

namespace meep {
namespace nvidia {

enum class geometry_value_kind : uint32_t {
  constant,
  file_epsilon,
  grid_tensor,
  grid_linear,
  grid_condinv
};

struct geometry_object_record {
  uint32_t kind;
  uint32_t subtype;
  uint32_t material;
  uint32_t closed;
  uint64_t parameter_offset;
  uint64_t parameter_count;
  uint64_t fixed_vertex_count;
  uint64_t vertex_offset;
  uint64_t vertex_count;
  uint64_t index_offset;
  uint64_t index_count;
  uint64_t auxiliary_offset;
  uint64_t auxiliary_count;
  uint64_t triangle_offset;
  uint64_t triangle_count;
  uint64_t bvh_offset;
  uint64_t bvh_count;
  uint64_t face_id_offset;
  uint64_t face_id_count;
  double mesh_lengthscale;
  double low[3];
  double high[3];
};

struct geometry_triangle_record {
  uint32_t vertex[3];
  uint32_t reserved;
  double normal[3];
  double low[3];
  double high[3];
};

struct geometry_bvh_record {
  double low[3];
  double high[3];
  uint32_t leaf;
  uint32_t left;
  uint32_t right;
  uint32_t escape;
  uint64_t first_face;
  uint64_t face_count;
};

struct geometry_image_record {
  uint32_t object;
  uint32_t ordinal;
  int32_t precedence;
  int32_t image[3];
  double shift[3];
  double low[3];
  double high[3];
};

struct geometry_value_record {
  geometry_value_kind kind;
  uint32_t overlap_kind;
  uint32_t dimensions[3];
  uint32_t flags;
  uint64_t sample_offset;
  uint64_t sample_count;
  double beta;
  double eta;
  double damping;
  double projection_offset;
  double value_1;
  double value_2;
  double tensor_1[6];
  double tensor_2[6];
};

struct geometry_launch_common {
  void *destination;
  unsigned char *classification;
  const unsigned char *compact_inputs;
  size_t compact_input_bytes;
  size_t object_offset;
  size_t object_count;
  size_t image_offset;
  size_t image_count;
  size_t value_offset;
  size_t material_count;
  size_t absorber_header_offset;
  size_t absorber_count;
  uint32_t default_material;
  size_t elements;
  size_t point_count;
  int dimensions;
  int component;
  int tensor_row;
  int tensor_column;
  int property;
  int axis_direction[3];
  int loop_begin[3];
  int little_corner[3];
  size_t loop_base_offset[3];
  size_t loop_extent[3];
  ptrdiff_t strides[3];
  int evaluation_shift[3];
  double cell_center[3];
  double cell_size[3];
  double metric[9];
  double inva;
  double dt;
  double trivial_value;
  bool logical_single;
  scalar_precision precision;
};

struct geometry_bulk_launch {
  geometry_launch_common common;
  uint64_t first_point;
  uint64_t count;
};

struct geometry_analytic_record {
  uint64_t point;
  uint32_t front_material;
  uint32_t behind_material;
  double normal[3];
  double fill;
};

struct geometry_analytic_launch {
  geometry_launch_common common;
  size_t job_offset;
  size_t count;
};

struct geometry_patch_record {
  uint64_t point;
  double value;
};

struct geometry_patch_launch {
  geometry_launch_common common;
  size_t patch_offset;
  size_t count;
};

void validate_geometry_bulk_launch(const geometry_bulk_launch &launch,
                                   const unsigned char *host_inputs, size_t host_bytes);
void validate_geometry_analytic_launch(const geometry_analytic_launch &launch,
                                       const unsigned char *host_inputs, size_t host_bytes);
void validate_geometry_patch_launch(const geometry_patch_launch &launch,
                                    const unsigned char *host_inputs, size_t host_bytes);
void launch_material_geometry_bulk(const geometry_bulk_launch &launch, const stream &stream);
void launch_material_geometry_analytic(const geometry_analytic_launch &launch,
                                       const stream &stream);
void launch_material_geometry_patch(const geometry_patch_launch &launch, const stream &stream);

static_assert(std::is_standard_layout<geometry_object_record>::value,
              "geometry object record must be standard layout");
static_assert(std::is_trivially_copyable<geometry_object_record>::value,
              "geometry object record must be trivially copyable");
static_assert(std::is_standard_layout<geometry_image_record>::value,
              "geometry image record must be standard layout");
static_assert(std::is_trivially_copyable<geometry_image_record>::value,
              "geometry image record must be trivially copyable");
static_assert(std::is_standard_layout<geometry_triangle_record>::value,
              "geometry triangle record must be standard layout");
static_assert(std::is_trivially_copyable<geometry_triangle_record>::value,
              "geometry triangle record must be trivially copyable");
static_assert(std::is_standard_layout<geometry_bvh_record>::value,
              "geometry BVH record must be standard layout");
static_assert(std::is_trivially_copyable<geometry_bvh_record>::value,
              "geometry BVH record must be trivially copyable");
static_assert(std::is_standard_layout<geometry_value_record>::value,
              "geometry value record must be standard layout");
static_assert(std::is_trivially_copyable<geometry_value_record>::value,
              "geometry value record must be trivially copyable");
static_assert(std::is_standard_layout<geometry_analytic_record>::value,
              "geometry analytic record must be standard layout");
static_assert(std::is_trivially_copyable<geometry_analytic_record>::value,
              "geometry analytic record must be trivially copyable");
static_assert(std::is_standard_layout<geometry_patch_record>::value,
              "geometry patch record must be standard layout");
static_assert(std::is_trivially_copyable<geometry_patch_record>::value,
              "geometry patch record must be trivially copyable");
static_assert(std::is_standard_layout<geometry_launch_common>::value,
              "geometry launch common must be standard layout");
static_assert(std::is_trivially_copyable<geometry_launch_common>::value,
              "geometry launch common must be trivially copyable");

} // namespace nvidia
} // namespace meep

#endif
