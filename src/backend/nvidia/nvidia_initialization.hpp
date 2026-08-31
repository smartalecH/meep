/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_INITIALIZATION_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_INITIALIZATION_HPP

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

#include "backend/nvidia/nvidia_step.hpp"

namespace meep {
namespace nvidia {

/* CUDA-private launch records.  They own no host addresses: constants are
   passed by value and PML inputs point into state-owned device storage. */
struct material_fill_launch {
  void *destination;
  size_t elements;
  double value;
  uint32_t phase;
  scalar_precision precision;
};

enum class material_table_kind : uint32_t { file_scalar_epsilon = 1, material_grid = 2 };

/* Pointer-free, versioned compact-table header reserved for PR5.2b.  Offsets
   are byte offsets in the one state-owned compact-input pack. */
struct material_table_header {
  uint32_t version;
  uint32_t material_id;
  material_table_kind kind;
  uint32_t overlap_kind;
  uint64_t sample_offset;
  uint64_t sample_count;
  uint32_t dimensions[3];
  uint32_t reserved;
  uint64_t medium_1_offset;
  uint64_t medium_2_offset;
  double beta;
  double eta;
  double damping;
  double projection_offset;
};

/* Extensible table launch record reserved for PR5.2b.  PR5.2a deliberately
   creates no instances and has no table execution entry point. */
struct material_table_launch {
  void *destination;
  const unsigned char *compact_inputs;
  size_t compact_input_bytes;
  size_t table_header_offset;
  size_t elements;
  size_t loop_count;
  int destination_component;
  int query_component;
  int tensor_column;
  int operation_family;
  int evaluation_shift[3];
  int axis_direction[3];
  int loop_begin[3];
  int little_corner[3];
  size_t loop_base_offset[3];
  size_t loop_extent[3];
  ptrdiff_t strides[3];
  double cell_center[3];
  double cell_size[3];
  double inva;
  bool logical_single;
  scalar_precision precision;
};

struct material_absorber_header {
  uint32_t version;
  int32_t direction;
  int32_t side;
  uint32_t reserved;
  uint64_t sample_offset;
  uint64_t sample_count;
  double thickness;
  double sample_spacing;
};

struct material_conductivity_launch {
  void *conductivity_destination;
  void *condinv_destination;
  const unsigned char *compact_inputs;
  size_t compact_input_bytes;
  size_t absorber_header_offset;
  size_t absorber_count;
  size_t elements;
  size_t loop_count;
  int component;
  int dimensions;
  int axis_direction[3];
  int loop_begin[3];
  int little_corner[3];
  size_t loop_base_offset[3];
  size_t loop_extent[3];
  ptrdiff_t strides[3];
  double cell_size[5];
  double inva;
  double base_conductivity;
  double dt;
  bool logical_single;
  scalar_precision precision;
};

struct material_pml_launch {
  void *sigma_destination;
  void *kappa_destination;
  void *sigma_inv_destination;
  const unsigned char *compact_inputs;
  size_t compact_input_bytes;
  size_t profile_offset;
  size_t elements;
  int little_corner;
  double resolution;
  double dt;
  double thickness;
  double boundary_location;
  double r_asymptotic;
  double mean_stretch;
  double profile_integral;
  double profile_integral_u;
  int thickness_cells;
  bool profile_active;
  bool analytic_quadratic;
  bool logical_single;
  scalar_precision precision;
};

void launch_material_fill(const material_fill_launch &launch, const stream &stream);
void validate_material_absorber_headers(const unsigned char *compact_inputs,
                                        size_t compact_input_bytes,
                                        size_t absorber_header_offset,
                                        size_t absorber_count);
void launch_material_conductivity(const material_conductivity_launch &launch,
                                  const stream &stream);
void launch_material_pml(const material_pml_launch &launch, const stream &stream);

static_assert(std::is_standard_layout<material_table_header>::value,
              "material table header must be standard layout");
static_assert(std::is_trivially_copyable<material_table_header>::value,
              "material table header must be trivially copyable");
static_assert(std::is_standard_layout<material_absorber_header>::value,
              "material absorber header must be standard layout");
static_assert(std::is_trivially_copyable<material_absorber_header>::value,
              "material absorber header must be trivially copyable");

} // namespace nvidia
} // namespace meep

#endif
