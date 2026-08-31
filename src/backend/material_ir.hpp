/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#ifndef MEEP_BACKEND_MATERIAL_IR_HPP
#define MEEP_BACKEND_MATERIAL_IR_HPP

#include <stdint.h>
#include <memory>
#include <vector>

#include "backend/storage_plan.hpp"

namespace meep {
class structure;
class fields;
}

namespace meep_geom {
class geom_epsilon;
}

namespace meep {

/* Compact owned schema.  Numeric payloads are deliberately normalized into
   vectors, rather than retaining libctl unions, callbacks, or geometry-tree
   nodes.  Shape-specific ordering is versioned by `version`. */
struct MaterialIRMaterial {
  int kind;
  bool host_callback;
  bool do_averaging;
  int material_grid_kind;
  bool material_grid_trivial;
  bool has_conductivity;
  bool has_chi2;
  bool has_chi3;
  uint32_t e_susceptibilities;
  uint32_t h_susceptibilities;
  std::vector<double> parameters;
  std::vector<double> samples;
};

struct MaterialIRObject {
  int kind;
  int material;
  std::vector<uint32_t> children;
  std::vector<double> parameters;
  std::vector<double> vertices;
  std::vector<double> indices;
};

struct MaterialIRPml {
  int direction;
  int side;
  double thickness;
  double r_asymptotic;
  double mean_stretch;
  double sample_spacing;
  std::vector<double> samples;
};

struct MaterialIRPmlAxis {
  int chunk;
  int direction;
  size_t elements;
  int little_corner;
  double resolution;
  bool profile_active;
  bool analytic_quadratic;
  double thickness;
  double boundary_location;
  double r_asymptotic;
  double mean_stretch;
  double profile_integral;
  double profile_integral_u;
  std::vector<double> profile_samples;
  /* CPU-oracle snapshots. Device-native initialization must not consume
     these dense output arrays. */
  std::vector<double> sigma;
  std::vector<double> kappa;
  std::vector<double> sigma_inv;
};

struct MaterialIRTopologyRow {
  StorageKey key;
  ElementType element_type;
  Precision logical_storage;
  size_t elements;
  size_t alignment;
  int yee_component;
  int extents[3];
  ptrdiff_t strides[3];
  int stagger[3];

  bool operator==(const MaterialIRTopologyRow &other) const;
};

struct MaterialIRSusceptibility {
  uint32_t identity;
  uint32_t material;
  int field_type;
  uint32_t material_ordinal;
  std::vector<double> parameters;
};

struct MaterialIRChunk {
  int chunk;
  int dimensions;
  bool owned;
  double resolution;
  double inva;
  size_t elements;
  uint64_t component_bits;
  int extents[3];
  ptrdiff_t strides[3];
  int little_corner[3];
  int big_corner[3];
  double origin[3];
  int stagger[NUM_FIELD_COMPONENTS][3];
  int loop_begin[NUM_FIELD_COMPONENTS][3];
  int loop_end[NUM_FIELD_COMPONENTS][3];
  size_t loop_count[NUM_FIELD_COMPONENTS];
  size_t pml_elements[6];
};

struct MaterialIR {
  uint32_t version;
  bool eps_averaging;
  double subpixel_tol;
  int subpixel_maxeval;
  bool ensure_periodicity;
  bool contains_host_callback;
  bool device_native_eligible;
  int dimensions;
  double projection_offset;
  std::vector<double> cell;
  uint32_t default_material;
  std::vector<MaterialIRMaterial> materials;
  std::vector<MaterialIRObject> objects;
  std::vector<uint32_t> roots;
  std::vector<uint32_t> extra_materials;
  std::vector<MaterialIRSusceptibility> susceptibilities;
  std::vector<MaterialIRChunk> chunks;
  std::vector<MaterialIRPml> absorbers;
  std::vector<MaterialIRPmlAxis> pml_axes;
  std::vector<MaterialIRTopologyRow> topology;
  uint64_t signature;
  uint64_t layout_signature;
};

bool material_ir_equal(const MaterialIR &a, const MaterialIR &b);

/* `absorbers` is an absorber_list pointer, type-erased to keep this private
   header independent of meepgeom.hpp. */
std::shared_ptr<const void> capture_material_ir(const structure &s,
                                               const meep_geom::geom_epsilon &geps,
                                               bool eps_averaging, double tol, int maxeval,
                                               const void *absorbers);
const MaterialIR *material_ir_for(const fields &f);
void validate_material_ir(const MaterialIR &ir);
void refresh_material_ir_signatures_for_testing(MaterialIR &ir);
void set_material_ir_capture_failure_for_testing(int rank, int mode);
int get_material_ir_capture_failure_rank_for_testing();
int get_material_ir_capture_failure_mode_for_testing();

} // namespace meep

#endif
