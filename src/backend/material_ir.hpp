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
  std::vector<double> comparison_medium;
  std::vector<double> parameters;
  std::vector<double> samples;
};

struct MaterialIRObject {
  int kind;
  int material;
  uint32_t source_identity;
  uint32_t root_identity;
  uint32_t leaf_ordinal;
  double parent_shift[3];
  double low[3];
  double high[3];
  uint64_t fixed_vertex_count;
  uint64_t vertex_offset;
  uint64_t vertex_count;
  uint64_t triangle_offset;
  uint64_t triangle_count;
  uint64_t bvh_offset;
  uint64_t bvh_count;
  double mesh_lengthscale;
  std::vector<double> parameters;
  std::vector<double> vertices;
  std::vector<double> indices;
  /* Fixed prism data required by pointer-free containment.  Prism
     records store, in order, vertices_p, top_polygon_diff_vectors_scaled_p,
     and vertices_top_p.  Mesh triangles/BVH live in the top-level geometry tables. */
  std::vector<double> auxiliary;
};

struct MaterialIRTriangle {
  uint32_t vertex[3];
  double normal[3];
  double low[3];
  double high[3];

  bool operator==(const MaterialIRTriangle &other) const;
};

struct MaterialIRBvhNode {
  double low[3];
  double high[3];
  bool leaf;
  uint32_t left;
  uint32_t right;
  uint64_t first_triangle;
  uint64_t triangle_count;

  bool operator==(const MaterialIRBvhNode &other) const;
};

struct MaterialIRGeometryImage {
  uint32_t object;
  uint32_t ordinal;
  int precedence;
  int image[3];
  double shift[3];
  double low[3];
  double high[3];

  bool operator==(const MaterialIRGeometryImage &other) const;
};

enum class MaterialIRProperty : uint32_t {
  chi1inv,
  conductivity,
  condinv,
  chi2,
  chi3,
  sigma
};

struct MaterialIRDestination {
  StorageKey key;
  uint32_t topology_index;
  uint32_t chunk_index;
  MaterialIRProperty property;
  int component;
  int tensor_direction;
  int tensor_column;
  bool offdiagonal;
  uint64_t point_count;

  bool operator==(const MaterialIRDestination &other) const;
};

struct MaterialIRBulkSpan {
  uint32_t destination;
  uint64_t first_point;
  uint64_t count;

  bool operator==(const MaterialIRBulkSpan &other) const;
};

struct MaterialIRAnalyticInterface {
  uint32_t destination;
  uint64_t point;
  uint32_t front_material;
  uint32_t behind_material;
  uint32_t object;
  uint32_t image;
  double normal[3];
  double fill;

  bool operator==(const MaterialIRAnalyticInterface &other) const;
};

enum class MaterialIRPatchReason : uint32_t {
  adaptive_overlap,
  material_grid_averaging,
  ambiguous_front,
  unsupported_analytic_shape,
  negative_material_fallback
};

enum MaterialIRVariableCause : uint32_t {
  material_variable_none = 0,
  material_variable_grid = 1u << 0,
  material_variable_user = 1u << 1,
  material_variable_file_default = 1u << 2
};

struct MaterialIRHybridPatch {
  uint32_t destination;
  uint64_t point;
  double value;
  uint32_t front_material;
  uint32_t behind_material;
  uint32_t object;
  uint32_t image;
  bool ambiguous;
  bool variable_material;
  uint32_t variable_causes;
  bool adaptive_fallback;
  bool negative_fallback;
  MaterialIRPatchReason reason;

  bool operator==(const MaterialIRHybridPatch &other) const;
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
  bool requires_hybrid;
  bool prism_include_boundaries;
  int dimensions;
  double projection_offset;
  std::vector<double> cell;
  double captured_volume[6];
  double lattice_basis_size[3];
  double lattice_basis[9];
  double lattice_metric[9];
  double lattice_inverse[9];
  double lattice_inverse_transpose[9];
  uint32_t default_material;
  uint32_t root_count;
  std::vector<MaterialIRMaterial> materials;
  std::vector<MaterialIRObject> objects;
  std::vector<double> geometry_vertices;
  std::vector<MaterialIRTriangle> geometry_triangles;
  std::vector<MaterialIRBvhNode> geometry_bvh;
  std::vector<uint32_t> geometry_bvh_face_ids;
  std::vector<MaterialIRGeometryImage> images;
  std::vector<uint32_t> active_images;
  std::vector<uint32_t> extra_materials;
  std::vector<MaterialIRSusceptibility> susceptibilities;
  std::vector<MaterialIRChunk> chunks;
  std::vector<MaterialIRPml> absorbers;
  std::vector<MaterialIRPmlAxis> pml_axes;
  std::vector<MaterialIRTopologyRow> topology;
  std::vector<MaterialIRDestination> destinations;
  std::vector<MaterialIRBulkSpan> bulk_spans;
  std::vector<MaterialIRAnalyticInterface> analytic_interfaces;
  std::vector<MaterialIRHybridPatch> hybrid_patches;
  uint64_t signature;
  uint64_t layout_signature;
};

bool material_ir_equal(const MaterialIR &a, const MaterialIR &b);

/* `absorbers` is an absorber_list pointer, type-erased to keep this private
   header independent of meepgeom.hpp. */
std::shared_ptr<const void> capture_material_ir(const structure &s,
                                               meep_geom::geom_epsilon &geps,
                                               bool eps_averaging, double tol, int maxeval,
                                               const void *absorbers);
const MaterialIR *material_ir_for(const fields &f);
void validate_material_ir(const MaterialIR &ir);
uint32_t material_ir_material_at_point(const MaterialIR &ir, const double point[3],
                                       uint32_t *image = NULL);
bool material_ir_materials_equal(const MaterialIR &ir, uint32_t a, uint32_t b);
double material_ir_grid_value_at_point(const MaterialIR &ir, const double point[3],
                                       uint32_t winning_image);
void finalize_material_ir_collective(MaterialIR &ir);
void refresh_material_ir_signatures_for_testing(MaterialIR &ir);
void set_material_ir_capture_failure_for_testing(int rank, int mode);
int get_material_ir_capture_failure_rank_for_testing();
int get_material_ir_capture_failure_mode_for_testing();

} // namespace meep

#endif
