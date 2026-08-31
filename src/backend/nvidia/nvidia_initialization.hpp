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
  unsigned char *classification;
  size_t elements;
  size_t classification_elements;
  double value;
  double trivial_value;
  uint32_t phase;
  bool logical_single;
  scalar_precision precision;
};

enum class material_table_kind : uint32_t { file_scalar_epsilon = 1, material_grid = 2 };

enum class material_table_operation : uint32_t {
  file_chi1inv = 1,
  grid_chi1inv = 2,
  grid_conductivity = 3,
  grid_sigma = 4
};

struct material_susceptibility_record {
  uint32_t version;
  uint32_t identity;
  int32_t field_type;
  uint32_t material_ordinal;
  double sigma_diagonal[3];
  double sigma_offdiagonal[3];
};

struct material_medium_header {
  uint32_t version;
  uint32_t electric_susceptibility_count;
  uint64_t electric_susceptibility_offset;
  double epsilon_diagonal[3];
  double epsilon_offdiagonal[3];
  double conductivity[3];
};

/* Pointer-free, versioned compact-table header. Offsets
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

/* Extensible table launch record for FILE and MaterialGrid execution. */
struct material_table_launch {
  void *destination;
  void *secondary_destination;
  unsigned char *classification;
  unsigned char *secondary_classification;
  const unsigned char *compact_inputs;
  size_t compact_input_bytes;
  size_t table_header_offset;
  size_t absorber_header_offset;
  size_t absorber_count;
  size_t elements;
  size_t loop_count;
  material_table_kind table_kind;
  material_table_operation operation;
  uint32_t source_material_id;
  uint32_t source_medium;
  uint64_t source_susceptibility;
  uint32_t susceptibility_identity;
  int susceptibility_field_type;
  int dimensions;
  int destination_component;
  int query_component;
  int tensor_row;
  int tensor_column;
  int operation_family;
  int evaluation_shift[3];
  int axis_direction[3];
  int loop_begin[3];
  int loop_end[3];
  int little_corner[3];
  size_t loop_base_offset[3];
  size_t loop_extent[3];
  ptrdiff_t strides[3];
  double cell_center[3];
  double cell_size[3];
  double inva;
  double dt;
  bool logical_single;
  double trivial_value;
  double secondary_trivial_value;
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
  unsigned char *conductivity_classification;
  unsigned char *condinv_classification;
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
  unsigned char *sigma_classification;
  unsigned char *kappa_classification;
  unsigned char *sigma_inv_classification;
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

struct material_classification_header {
  uint32_t version;
  uint32_t row_count;
  uint64_t recipe_signature;
  uint64_t state_token;
  uint64_t storage_fingerprint;
};

struct material_classification_row_result {
  int32_t chunk;
  int32_t kind;
  int32_t component;
  int32_t cmp;
  uint64_t aux;
  uint32_t missing;
  uint32_t nontrivial;
};

static_assert(std::is_standard_layout<material_classification_header>::value &&
                  std::is_trivially_copyable<material_classification_header>::value,
              "material classification header must be a POD transfer record");
static_assert(std::is_standard_layout<material_classification_row_result>::value &&
                  std::is_trivially_copyable<material_classification_row_result>::value,
              "material classification row must be a POD transfer record");

void launch_material_fill(const material_fill_launch &launch, const stream &stream);
void validate_material_absorber_headers(const unsigned char *compact_inputs,
                                        size_t compact_input_bytes,
                                        size_t absorber_header_offset,
                                        size_t absorber_count);
void validate_material_table_headers(const unsigned char *compact_inputs,
                                     size_t compact_input_bytes,
                                     const size_t *table_header_offsets,
                                     size_t table_header_count);
void validate_material_table_launch(const material_table_launch &launch,
                                    const unsigned char *host_compact_inputs,
                                    size_t host_compact_input_bytes);
void launch_material_table(const material_table_launch &launch, const stream &stream);
int material_table_mirror_index_for_testing(int i, int n);
void launch_material_conductivity(const material_conductivity_launch &launch,
                                  const stream &stream);
void launch_material_pml(const material_pml_launch &launch, const stream &stream);
void launch_material_classification_header(material_classification_header *result,
                                           material_classification_header value,
                                           const stream &stream);
void launch_material_classification_row(
    const unsigned char *status, size_t elements,
    material_classification_row_result *result,
    material_classification_row_result value, const stream &stream);

static_assert(std::is_standard_layout<material_table_header>::value,
              "material table header must be standard layout");
static_assert(std::is_trivially_copyable<material_table_header>::value,
              "material table header must be trivially copyable");
static_assert(std::is_standard_layout<material_medium_header>::value,
              "material medium header must be standard layout");
static_assert(std::is_trivially_copyable<material_medium_header>::value,
              "material medium header must be trivially copyable");
static_assert(std::is_standard_layout<material_susceptibility_record>::value,
              "material susceptibility record must be standard layout");
static_assert(std::is_trivially_copyable<material_susceptibility_record>::value,
              "material susceptibility record must be trivially copyable");
static_assert(std::is_standard_layout<material_absorber_header>::value,
              "material absorber header must be standard layout");
static_assert(std::is_trivially_copyable<material_absorber_header>::value,
              "material absorber header must be trivially copyable");

} // namespace nvidia
} // namespace meep

#endif
