/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/material_ir.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

#include "material_data.hpp"
#include "meepgeom.hpp"
#include "backend/material_callback.hpp"
#include "backend/material_geometry_numeric.hpp"
#include "backend/precision.hpp"

namespace meep {
namespace {

const uint32_t material_ir_version = 6;
int material_ir_capture_failure_rank = -1;
int material_ir_capture_failure_mode = 0;

typedef std::vector<std::shared_ptr<const OwnedMaterialCallback> > MaterialCallbackOwners;
std::mutex material_ir_owners_mutex;
std::map<const MaterialIR *, std::weak_ptr<const MaterialCallbackOwners> > material_ir_owners;

struct CapturedMaterialIR {
  std::shared_ptr<MaterialIR> ir;
  std::shared_ptr<const MaterialCallbackOwners> owners;

  ~CapturedMaterialIR() {
    std::lock_guard<std::mutex> lock(material_ir_owners_mutex);
    material_ir_owners.erase(ir.get());
  }
};

struct CapturedLibctlMeshBvhNode {
  vector3 bbox_low;
  vector3 bbox_high;
  int left_child;
  int right_child;
  int face_start;
  int face_count;
};

struct CapturedLibctlMeshInternal {
  int num_faces;
  int *face_indices;
  vector3 *face_normals;
  double *face_areas;
  int num_bvh_nodes;
  CapturedLibctlMeshBvhNode *bvh;
  int *bvh_face_ids;
  vector3 centroid;
  double lengthscale;
};

bool prism_boundary_policy() {
  static const bool include = []() {
    const char *value = std::getenv("LIBCTL_EXCLUDE_BOUNDARIES");
    return !(value && value[0] == '1');
  }();
  return include;
}

struct StorageKeyLess {
  bool operator()(const StorageKey &a, const StorageKey &b) const {
    if (a.chunk != b.chunk) return a.chunk < b.chunk;
    if (a.kind != b.kind) return a.kind < b.kind;
    if (a.component_ != b.component_) return a.component_ < b.component_;
    if (a.cmp != b.cmp) return a.cmp < b.cmp;
    return a.aux < b.aux;
  }
};

void append_vec(std::vector<double> &out, vector3 v) {
  out.push_back(v.x); out.push_back(v.y); out.push_back(v.z);
}

void append_cvec(std::vector<double> &out, cvector3 v) {
  out.push_back(v.x.re); out.push_back(v.x.im);
  out.push_back(v.y.re); out.push_back(v.y.im);
  out.push_back(v.z.re); out.push_back(v.z.im);
}

void append_susceptibility(std::vector<double> &out, const meep_geom::susceptibility &sus) {
  append_vec(out, sus.sigma_offdiag); append_vec(out, sus.sigma_diag); append_vec(out, sus.bias);
  out.push_back(sus.frequency); out.push_back(sus.gamma); out.push_back(sus.alpha);
  out.push_back(sus.noise_amp); out.push_back(sus.drude ? 1 : 0);
  out.push_back(sus.saturated_gyrotropy ? 1 : 0); out.push_back(sus.is_file ? 1 : 0);
  out.push_back(double(sus.transitions.size()));
  for (const meep_geom::transition &tr : sus.transitions) {
    out.push_back(tr.from_level); out.push_back(tr.to_level); out.push_back(tr.transition_rate);
    out.push_back(tr.frequency); append_vec(out, tr.sigma_diag); out.push_back(tr.gamma);
    out.push_back(tr.pumping_rate);
  }
  out.push_back(double(sus.initial_populations.size()));
  out.insert(out.end(), sus.initial_populations.begin(), sus.initial_populations.end());
}

void append_medium(std::vector<double> &out, const meep_geom::medium_struct &m) {
  append_vec(out, m.epsilon_diag); append_cvec(out, m.epsilon_offdiag);
  append_vec(out, m.mu_diag); append_cvec(out, m.mu_offdiag);
  append_vec(out, m.E_chi2_diag); append_vec(out, m.E_chi3_diag);
  append_vec(out, m.H_chi2_diag); append_vec(out, m.H_chi3_diag);
  append_vec(out, m.D_conductivity_diag); append_vec(out, m.B_conductivity_diag);
  const meep_geom::susceptibility_list *lists[2] = {&m.E_susceptibilities,
                                                    &m.H_susceptibilities};
  for (int ft = 0; ft < 2; ++ft) {
    out.push_back(double(lists[ft]->size()));
    for (const meep_geom::susceptibility &sus : *lists[ft]) append_susceptibility(out, sus);
  }
}

void note_medium(MaterialIRMaterial &out, const meep_geom::medium_struct &m) {
  out.has_conductivity = out.has_conductivity || m.D_conductivity_diag.x != 0 ||
                         m.D_conductivity_diag.y != 0 || m.D_conductivity_diag.z != 0 ||
                         m.B_conductivity_diag.x != 0 || m.B_conductivity_diag.y != 0 ||
                         m.B_conductivity_diag.z != 0;
  out.has_chi2 = out.has_chi2 || m.E_chi2_diag.x != 0 || m.E_chi2_diag.y != 0 ||
                 m.E_chi2_diag.z != 0 || m.H_chi2_diag.x != 0 || m.H_chi2_diag.y != 0 ||
                 m.H_chi2_diag.z != 0;
  out.has_chi3 = out.has_chi3 || m.E_chi3_diag.x != 0 || m.E_chi3_diag.y != 0 ||
                 m.E_chi3_diag.z != 0 || m.H_chi3_diag.x != 0 || m.H_chi3_diag.y != 0 ||
                 m.H_chi3_diag.z != 0;
  if (m.E_susceptibilities.size() > std::numeric_limits<uint32_t>::max() ||
      m.H_susceptibilities.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("material IR susceptibility-list count overflow");
  out.e_susceptibilities = std::max(out.e_susceptibilities,
                                    uint32_t(m.E_susceptibilities.size()));
  out.h_susceptibilities = std::max(out.h_susceptibilities,
                                    uint32_t(m.H_susceptibilities.size()));
}

uint32_t capture_material(MaterialIR &ir, const meep_geom::material_data *md,
                          std::map<const void *, uint32_t> &seen,
                          MaterialCallbackOwners &owners) {
  if (!md) throw std::invalid_argument("material IR contains a null material");
  if (md->which_subclass < meep_geom::material_data::MEDIUM ||
      md->which_subclass > meep_geom::material_data::PERFECT_METAL)
    throw std::invalid_argument("material IR contains an invalid material tag");
  std::map<const void *, uint32_t>::const_iterator existing = seen.find(md);
  if (existing != seen.end()) return existing->second;
  if (ir.materials.size() >= std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("material IR material count overflow");
  const uint32_t id = uint32_t(ir.materials.size());
  seen[md] = id;
  MaterialIRMaterial record;
  record.kind = md->which_subclass;
  record.host_callback = md->which_subclass == meep_geom::material_data::MATERIAL_USER;
  std::shared_ptr<const OwnedMaterialCallback> callback_owner;
  record.owned_callback = record.host_callback &&
                          meep_geom::owned_material_callback(md, &callback_owner);
  record.callback_id = record.owned_callback ? callback_owner->id : UINT64_C(0);
  record.callback_signature = record.owned_callback ? callback_owner->signature : UINT64_C(0);
  record.callback_capabilities = record.owned_callback ? callback_owner->capabilities : UINT64_C(0);
  if (record.owned_callback) {
    bool retained = false;
    for (const std::shared_ptr<const OwnedMaterialCallback> &existing : owners)
      if (existing->id == callback_owner->id) {
        if (existing->signature != callback_owner->signature ||
            existing->capabilities != callback_owner->capabilities)
          throw std::invalid_argument(
              "owned material callback ID has inconsistent signatures or capabilities");
        retained = true;
      }
    if (!retained) owners.push_back(callback_owner);
  }
  record.do_averaging =
      record.host_callback || md->which_subclass == meep_geom::material_data::MATERIAL_GRID
          ? md->do_averaging
          : false;
  record.material_grid_kind = md->which_subclass == meep_geom::material_data::MATERIAL_GRID
                                  ? md->material_grid_kinds
                                  : -1;
  record.material_grid_trivial =
      md->which_subclass == meep_geom::material_data::MATERIAL_GRID && md->damping == 0 &&
      md->medium_1.E_susceptibilities.empty() && md->medium_2.E_susceptibilities.empty();
  record.has_conductivity = record.has_chi2 = record.has_chi3 = false;
  record.e_susceptibilities = record.h_susceptibilities = 0;
  if (record.host_callback) ir.contains_host_callback = true;
  if (md->which_subclass == meep_geom::material_data::MEDIUM ||
      md->which_subclass == meep_geom::material_data::MATERIAL_GRID)
    append_medium(record.comparison_medium, md->medium);
  if (md->which_subclass != meep_geom::material_data::PERFECT_METAL &&
      md->which_subclass != meep_geom::material_data::MATERIAL_GRID &&
      md->which_subclass != meep_geom::material_data::MATERIAL_USER)
    append_medium(record.parameters, md->medium), note_medium(record, md->medium);
  if (md->which_subclass == meep_geom::material_data::MATERIAL_FILE) {
    size_t n = 1;
    for (int d = 0; d < 3; ++d) {
      if (md->epsilon_dims[d] && n > std::numeric_limits<size_t>::max() / md->epsilon_dims[d])
        throw std::overflow_error("material IR file extent overflow");
      n *= md->epsilon_dims[d];
      record.parameters.push_back(double(md->epsilon_dims[d]));
    }
    if (n) {
      if (!md->epsilon_data) throw std::invalid_argument("material IR file has no samples");
      record.samples.assign(md->epsilon_data, md->epsilon_data + n);
    }
  }
  else if (md->which_subclass == meep_geom::material_data::MATERIAL_GRID) {
    append_vec(record.parameters, md->grid_size);
    append_medium(record.parameters, md->medium_1); append_medium(record.parameters, md->medium_2);
    note_medium(record, md->medium_1); note_medium(record, md->medium_2);
    record.parameters.push_back(md->beta); record.parameters.push_back(md->eta);
    record.parameters.push_back(md->damping);
    record.has_conductivity = record.has_conductivity || md->damping != 0.0;
    size_t n = 1;
    const double dims[3] = {md->grid_size.x, md->grid_size.y, md->grid_size.z};
    for (int d = 0; d < 3; ++d) {
      if (!std::isfinite(dims[d]) || dims[d] < 0 || std::floor(dims[d]) != dims[d] ||
          dims[d] > double(std::numeric_limits<size_t>::max()))
        throw std::invalid_argument("material IR grid has an invalid extent");
      const size_t dim = size_t(dims[d]);
      if (dim && n > std::numeric_limits<size_t>::max() / dim)
        throw std::overflow_error("material IR grid extent overflow");
      n *= dim;
    }
    if (n) {
      if (!md->weights) throw std::invalid_argument("material IR grid has no weights");
      record.samples.assign(md->weights, md->weights + n);
    }
  }
  ir.materials.push_back(record);
  return id;
}

void append_matrix(std::vector<double> &out, matrix3x3 m) {
  append_vec(out, m.c0); append_vec(out, m.c1); append_vec(out, m.c2);
}

void capture_object(MaterialIR &ir, const geometric_object &object, uint32_t root_identity,
                    vector3 parent_shift, std::map<const void *, uint32_t> &materials,
                    std::map<const geometric_object *, uint32_t> &objects,
                    MaterialCallbackOwners &owners) {
  if (object.which_subclass == geometric_object::COMPOUND_GEOMETRIC_OBJECT) {
    const geometric_object_list &children =
        object.subclass.compound_geometric_object_data->component_objects;
    if (children.num_items < 0)
      throw std::invalid_argument("material IR compound has a negative child count");
    parent_shift = vector3_plus(parent_shift, object.center);
    for (int i = 0; i < children.num_items; ++i)
      capture_object(ir, children.items[i], root_identity, parent_shift, materials, objects,
                     owners);
    return;
  }
  if (object.which_subclass == geometric_object::GEOMETRIC_OBJECT_SELF) return;
  if (ir.objects.size() >= std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("material IR object count overflow");
  MaterialIRObject out;
  out.kind = object.which_subclass;
  out.source_identity = uint32_t(ir.objects.size());
  out.root_identity = root_identity;
  out.leaf_ordinal = out.source_identity;
  out.parent_shift[0] = parent_shift.x;
  out.parent_shift[1] = parent_shift.y;
  out.parent_shift[2] = parent_shift.z;
  geom_box fixed_bounds;
  geom_get_bounding_box(object, &fixed_bounds);
  out.low[0] = fixed_bounds.low.x;
  out.low[1] = fixed_bounds.low.y;
  out.low[2] = fixed_bounds.low.z;
  out.high[0] = fixed_bounds.high.x;
  out.high[1] = fixed_bounds.high.y;
  out.high[2] = fixed_bounds.high.z;
  out.fixed_vertex_count = 0;
  out.vertex_offset = out.vertex_count = 0;
  out.triangle_offset = out.triangle_count = 0;
  out.bvh_offset = out.bvh_count = 0;
  out.mesh_lengthscale = 0.0;
  const uint32_t material = capture_material(
      ir, static_cast<const meep_geom::material_data *>(object.material), materials, owners);
  if (material > uint32_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("material IR object material ID overflow");
  out.material = int(material);
  append_vec(out.parameters, object.center);
  switch (object.which_subclass) {
    case geometric_object::MESH: {
      const mesh &m = *object.subclass.mesh_data;
      if (m.vertices.num_items < 0 || m.face_indices.num_items < 0)
        throw std::invalid_argument("material IR mesh has a negative count");
      const CapturedLibctlMeshInternal *fixed =
          static_cast<const CapturedLibctlMeshInternal *>(m.internal);
      if (!fixed || fixed->num_faces != m.face_indices.num_items || fixed->num_bvh_nodes < 0)
        throw std::invalid_argument("material IR mesh fixed state is absent");
      out.parameters.push_back(m.is_closed ? 1 : 0);
      for (int i = 0; i < m.vertices.num_items; ++i) append_vec(out.vertices, m.vertices.items[i]);
      for (int i = 0; i < fixed->num_faces * 3; ++i)
        out.indices.push_back(double(fixed->face_indices[i]));
      out.vertex_offset = ir.geometry_vertices.size() / 3;
      out.vertex_count = size_t(m.vertices.num_items);
      ir.geometry_vertices.insert(ir.geometry_vertices.end(), out.vertices.begin(), out.vertices.end());
      out.triangle_offset = ir.geometry_triangles.size();
      out.triangle_count = size_t(m.face_indices.num_items);
      out.mesh_lengthscale = fixed->lengthscale;
      for (int i = 0; i < fixed->num_faces; ++i) {
        const uint32_t ids[3] = {uint32_t(fixed->face_indices[3 * i]),
                                 uint32_t(fixed->face_indices[3 * i + 1]),
                                 uint32_t(fixed->face_indices[3 * i + 2])};
        MaterialIRTriangle triangle = {};
        for (int vertex = 0; vertex < 3; ++vertex)
          triangle.vertex[vertex] = uint32_t(out.vertex_offset) + ids[vertex];
        const vector3 a = m.vertices.items[ids[0]], b = m.vertices.items[ids[1]],
                      c = m.vertices.items[ids[2]];
        const vector3 n = fixed->face_normals[i];
        triangle.normal[0] = n.x; triangle.normal[1] = n.y; triangle.normal[2] = n.z;
        for (int axis = 0; axis < 3; ++axis) {
          const double av = axis == 0 ? a.x : axis == 1 ? a.y : a.z;
          const double bv = axis == 0 ? b.x : axis == 1 ? b.y : b.z;
          const double cv = axis == 0 ? c.x : axis == 1 ? c.y : c.z;
          triangle.low[axis] = std::min(av, std::min(bv, cv));
          triangle.high[axis] = std::max(av, std::max(bv, cv));
        }
        ir.geometry_triangles.push_back(triangle);
      }
      out.bvh_offset = ir.geometry_bvh.size();
      out.bvh_count = size_t(fixed->num_bvh_nodes);
      const uint64_t face_id_offset = ir.geometry_bvh_face_ids.size();
      for (int i = 0; i < fixed->num_faces; ++i) {
        if (fixed->bvh_face_ids[i] < 0 || fixed->bvh_face_ids[i] >= fixed->num_faces)
          throw std::invalid_argument("material IR mesh BVH face identity is invalid");
        ir.geometry_bvh_face_ids.push_back(
            uint32_t(out.triangle_offset + uint64_t(fixed->bvh_face_ids[i])));
      }
      for (int i = 0; i < fixed->num_bvh_nodes; ++i) {
        const CapturedLibctlMeshBvhNode &source = fixed->bvh[i];
        MaterialIRBvhNode node = {};
        node.low[0] = source.bbox_low.x; node.low[1] = source.bbox_low.y;
        node.low[2] = source.bbox_low.z; node.high[0] = source.bbox_high.x;
        node.high[1] = source.bbox_high.y; node.high[2] = source.bbox_high.z;
        node.leaf = source.left_child < 0 && source.right_child < 0;
        if ((source.left_child < 0) != (source.right_child < 0) ||
            (node.leaf && (source.face_start < 0 || source.face_count <= 0)) ||
            (!node.leaf && (source.face_start != -1 || source.face_count != 0)))
          throw std::invalid_argument("material IR mesh BVH node kind is invalid");
        node.left = source.left_child < 0
                        ? std::numeric_limits<uint32_t>::max()
                        : uint32_t(out.bvh_offset + uint64_t(source.left_child));
        node.right = source.right_child < 0
                         ? std::numeric_limits<uint32_t>::max()
                         : uint32_t(out.bvh_offset + uint64_t(source.right_child));
        node.first_triangle = node.leaf ? face_id_offset + uint64_t(source.face_start) : 0;
        node.triangle_count = node.leaf ? uint64_t(source.face_count) : 0;
        ir.geometry_bvh.push_back(node);
      }
      break;
    }
    case geometric_object::PRISM: {
      const prism &p = *object.subclass.prism_data;
      if (p.vertices.num_items < 0 || p.vertices_p.num_items < 0 ||
          p.top_polygon_diff_vectors_scaled_p.num_items != p.vertices_p.num_items ||
          p.vertices_top_p.num_items != p.vertices_p.num_items)
        throw std::invalid_argument("material IR prism has a negative vertex count");
      out.parameters.push_back(p.height); append_vec(out.parameters, p.axis);
      out.parameters.push_back(p.sidewall_angle); append_vec(out.parameters, p.centroid);
      append_matrix(out.parameters, p.m_c2p); append_matrix(out.parameters, p.m_p2c);
      for (int i = 0; i < p.vertices.num_items; ++i) append_vec(out.vertices, p.vertices.items[i]);
      out.fixed_vertex_count = size_t(p.vertices_p.num_items);
      for (int i = 0; i < p.vertices_p.num_items; ++i)
        append_vec(out.auxiliary, p.vertices_p.items[i]);
      for (int i = 0; i < p.vertices_p.num_items; ++i)
        append_vec(out.auxiliary, p.top_polygon_diff_vectors_scaled_p.items[i]);
      for (int i = 0; i < p.vertices_p.num_items; ++i)
        append_vec(out.auxiliary, p.vertices_top_p.items[i]);
      break;
    }
    case geometric_object::BLOCK: {
      const block &b = *object.subclass.block_data;
      if (b.which_subclass < block::BLOCK_SELF || b.which_subclass > block::ELLIPSOID)
        throw std::invalid_argument("material IR block has an invalid subtype");
      append_vec(out.parameters, b.e1); append_vec(out.parameters, b.e2); append_vec(out.parameters, b.e3);
      append_vec(out.parameters, b.size); append_matrix(out.parameters, b.projection_matrix);
      out.parameters.push_back(b.which_subclass);
      if (b.which_subclass == block::ELLIPSOID)
        append_vec(out.parameters, b.subclass.ellipsoid_data->inverse_semi_axes);
      break;
    }
    case geometric_object::SPHERE: out.parameters.push_back(object.subclass.sphere_data->radius); break;
    case geometric_object::CYLINDER: {
      const cylinder &c = *object.subclass.cylinder_data;
      if (c.which_subclass < cylinder::CYLINDER_SELF || c.which_subclass > cylinder::CONE)
        throw std::invalid_argument("material IR cylinder has an invalid subtype");
      append_vec(out.parameters, c.axis); out.parameters.push_back(c.radius);
      out.parameters.push_back(c.height); out.parameters.push_back(c.which_subclass);
      if (c.which_subclass == cylinder::WEDGE) {
        const wedge &w = *c.subclass.wedge_data;
        out.parameters.push_back(w.wedge_angle); append_vec(out.parameters, w.wedge_start);
        append_vec(out.parameters, w.e1); append_vec(out.parameters, w.e2);
      }
      else if (c.which_subclass == cylinder::CONE) out.parameters.push_back(c.subclass.cone_data->radius2);
      break;
    }
    case geometric_object::COMPOUND_GEOMETRIC_OBJECT:
    case geometric_object::GEOMETRIC_OBJECT_SELF:
      throw std::logic_error("material IR attempted to store a non-leaf object");
  }
  const uint32_t id = uint32_t(ir.objects.size());
  ir.objects.push_back(out);
  objects[&object] = id;
}

void mix_byte(uint64_t &h, unsigned char b) {
  h ^= b; h *= UINT64_C(1099511628211);
}
void mix_u64(uint64_t &h, uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) mix_byte(h, (value >> (8 * i)) & 0xffu);
}
void mix_i64(uint64_t &h, int64_t value) { mix_u64(h, uint64_t(value)); }
void mix_bool(uint64_t &h, bool value) { mix_byte(h, value ? 1 : 0); }
void mix_double(uint64_t &h, double value) {
  if (std::isnan(value)) {
    mix_u64(h, UINT64_C(0x7ff8000000000000));
    return;
  }
  if (value == 0.0) value = 0.0;
  uint64_t bits = 0; memcpy(&bits, &value, sizeof(bits)); mix_u64(h, bits);
}
bool equal_double(double a, double b) {
  return a == b || (std::isnan(a) && std::isnan(b));
}
void mix_tag(uint64_t &h, const char *value) {
  const size_t n = strlen(value); mix_u64(h, n);
  for (size_t i = 0; i < n; ++i) mix_byte(h, static_cast<unsigned char>(value[i]));
}
void mix_values(uint64_t &h, const std::vector<double> &values) {
  mix_u64(h, values.size());
  for (double value : values) mix_double(h, value);
}

uint64_t signature(const MaterialIR &ir, bool include_rank_layout) {
  uint64_t h = UINT64_C(1469598103934665603);
  mix_tag(h, "meep.material-ir.v1"); mix_u64(h, ir.version);
  mix_bool(h, ir.eps_averaging); mix_double(h, ir.subpixel_tol);
  mix_i64(h, ir.subpixel_maxeval); mix_bool(h, ir.ensure_periodicity);
  mix_bool(h, ir.contains_host_callback); mix_bool(h, ir.device_native_eligible);
  mix_bool(h, ir.prism_include_boundaries);
  mix_i64(h, ir.dimensions); mix_double(h, ir.projection_offset);
  mix_values(h, ir.cell);
  for (int i = 0; i < 6; ++i) mix_double(h, ir.captured_volume[i]);
  for (int axis = 0; axis < 3; ++axis) mix_double(h, ir.lattice_basis_size[axis]);
  for (int i = 0; i < 9; ++i) {
    mix_double(h, ir.lattice_basis[i]); mix_double(h, ir.lattice_metric[i]);
    mix_double(h, ir.lattice_inverse[i]); mix_double(h, ir.lattice_inverse_transpose[i]);
  }
  mix_u64(h, ir.default_material); mix_u64(h, ir.root_count);
  mix_u64(h, ir.materials.size());
  for (const MaterialIRMaterial &m : ir.materials) {
    mix_tag(h, "material"); mix_i64(h, m.kind); mix_bool(h, m.host_callback);
    mix_bool(h, m.owned_callback); mix_u64(h, m.callback_id); mix_u64(h, m.callback_signature);
    mix_u64(h, m.callback_capabilities);
    mix_bool(h, m.do_averaging); mix_i64(h, m.material_grid_kind);
    mix_bool(h, m.material_grid_trivial);
    mix_bool(h, m.has_conductivity); mix_bool(h, m.has_chi2); mix_bool(h, m.has_chi3);
    mix_u64(h, m.e_susceptibilities); mix_u64(h, m.h_susceptibilities);
    mix_values(h, m.comparison_medium); mix_values(h, m.parameters); mix_values(h, m.samples);
  }
  mix_u64(h, ir.objects.size());
  for (const MaterialIRObject &o : ir.objects) {
    mix_tag(h, "object"); mix_i64(h, o.kind); mix_i64(h, o.material);
    mix_u64(h, o.source_identity); mix_u64(h, o.root_identity); mix_u64(h, o.leaf_ordinal);
    for (int axis = 0; axis < 3; ++axis) {
      mix_double(h, o.parent_shift[axis]); mix_double(h, o.low[axis]); mix_double(h, o.high[axis]);
    }
    mix_u64(h, o.fixed_vertex_count); mix_u64(h, o.vertex_offset); mix_u64(h, o.vertex_count);
    mix_u64(h, o.triangle_offset); mix_u64(h, o.triangle_count);
    mix_u64(h, o.bvh_offset); mix_u64(h, o.bvh_count); mix_double(h, o.mesh_lengthscale);
    mix_values(h, o.parameters); mix_values(h, o.vertices); mix_values(h, o.indices);
    mix_values(h, o.auxiliary);
  }
  mix_values(h, ir.geometry_vertices);
  mix_u64(h, ir.geometry_triangles.size());
  for (const MaterialIRTriangle &triangle : ir.geometry_triangles) {
    for (int i = 0; i < 3; ++i) {
      mix_u64(h, triangle.vertex[i]); mix_double(h, triangle.normal[i]);
      mix_double(h, triangle.low[i]); mix_double(h, triangle.high[i]);
    }
  }
  mix_u64(h, ir.geometry_bvh.size());
  for (const MaterialIRBvhNode &node : ir.geometry_bvh) {
    for (int i = 0; i < 3; ++i) {
      mix_double(h, node.low[i]); mix_double(h, node.high[i]);
    }
    mix_bool(h, node.leaf); mix_u64(h, node.left); mix_u64(h, node.right);
    mix_u64(h, node.first_triangle); mix_u64(h, node.triangle_count);
  }
  mix_u64(h, ir.geometry_bvh_face_ids.size());
  for (uint32_t face : ir.geometry_bvh_face_ids) mix_u64(h, face);
  mix_u64(h, ir.images.size());
  for (const MaterialIRGeometryImage &image : ir.images) {
    mix_tag(h, "geometry-image"); mix_u64(h, image.object); mix_u64(h, image.ordinal);
    mix_i64(h, image.precedence);
    for (int axis = 0; axis < 3; ++axis) {
      mix_i64(h, image.image[axis]); mix_double(h, image.shift[axis]);
      mix_double(h, image.low[axis]); mix_double(h, image.high[axis]);
    }
  }
  mix_u64(h, ir.extra_materials.size());
  for (uint32_t material : ir.extra_materials) mix_u64(h, material);
  mix_u64(h, ir.susceptibilities.size());
  for (const MaterialIRSusceptibility &s : ir.susceptibilities) {
    mix_u64(h, s.identity); mix_u64(h, s.material); mix_i64(h, s.field_type);
    mix_u64(h, s.material_ordinal); mix_values(h, s.parameters);
  }
  if (include_rank_layout) {
    mix_bool(h, ir.requires_hybrid);
    mix_tag(h, "rank-layout"); mix_u64(h, ir.chunks.size());
    for (const MaterialIRChunk &c : ir.chunks) {
      mix_i64(h, c.chunk); mix_i64(h, c.dimensions); mix_bool(h, c.owned);
      mix_double(h, c.resolution); mix_double(h, c.inva);
      mix_u64(h, c.elements); mix_u64(h, c.component_bits);
      for (int axis = 0; axis < 3; ++axis) {
        mix_i64(h, c.extents[axis]); mix_i64(h, c.strides[axis]);
        mix_i64(h, c.little_corner[axis]); mix_i64(h, c.big_corner[axis]);
        mix_double(h, c.origin[axis]);
      }
      for (int component = 0; component < NUM_FIELD_COMPONENTS; ++component)
        {
          for (int axis = 0; axis < 3; ++axis) {
            mix_i64(h, c.stagger[component][axis]);
            mix_i64(h, c.loop_begin[component][axis]);
            mix_i64(h, c.loop_end[component][axis]);
          }
          mix_u64(h, c.loop_count[component]);
        }
      for (int d = 0; d < 6; ++d) mix_u64(h, c.pml_elements[d]);
    }
  }
  if (include_rank_layout) {
    mix_u64(h, ir.active_images.size());
    for (uint32_t image : ir.active_images) mix_u64(h, image);
  }
  mix_u64(h, ir.absorbers.size());
  for (const MaterialIRPml &p : ir.absorbers) {
    mix_tag(h, "absorber"); mix_i64(h, p.direction); mix_i64(h, p.side);
    mix_double(h, p.thickness); mix_double(h, p.r_asymptotic);
    mix_double(h, p.mean_stretch); mix_double(h, p.sample_spacing); mix_values(h, p.samples);
  }
  if (include_rank_layout) {
    mix_u64(h, ir.pml_axes.size());
    for (const MaterialIRPmlAxis &p : ir.pml_axes) {
      mix_tag(h, "pml-axis"); mix_i64(h, p.chunk); mix_i64(h, p.direction);
      mix_u64(h, p.elements); mix_i64(h, p.little_corner); mix_double(h, p.resolution);
      mix_bool(h, p.profile_active);
      mix_bool(h, p.analytic_quadratic); mix_double(h, p.thickness);
      mix_double(h, p.boundary_location); mix_double(h, p.r_asymptotic);
      mix_double(h, p.mean_stretch); mix_double(h, p.profile_integral);
      mix_double(h, p.profile_integral_u); mix_values(h, p.profile_samples);
      mix_values(h, p.sigma); mix_values(h, p.kappa);
      mix_values(h, p.sigma_inv);
    }
    mix_u64(h, ir.destinations.size());
    for (const MaterialIRDestination &destination : ir.destinations) {
      mix_tag(h, "destination");
      mix_i64(h, destination.key.chunk); mix_i64(h, destination.key.kind);
      mix_i64(h, destination.key.component_); mix_i64(h, destination.key.cmp);
      mix_u64(h, destination.key.aux); mix_u64(h, destination.topology_index);
      mix_u64(h, destination.chunk_index); mix_u64(h, uint32_t(destination.property));
      mix_i64(h, destination.component); mix_i64(h, destination.tensor_direction);
      mix_i64(h, destination.tensor_column); mix_bool(h, destination.offdiagonal);
      mix_u64(h, destination.point_count);
    }
    mix_u64(h, ir.bulk_spans.size());
    for (const MaterialIRBulkSpan &span : ir.bulk_spans) {
      mix_u64(h, span.destination); mix_u64(h, span.first_point); mix_u64(h, span.count);
    }
    mix_u64(h, ir.analytic_interfaces.size());
    for (const MaterialIRAnalyticInterface &job : ir.analytic_interfaces) {
      mix_u64(h, job.destination); mix_u64(h, job.point); mix_u64(h, job.front_material);
      mix_u64(h, job.behind_material); mix_u64(h, job.object); mix_u64(h, job.image);
      for (int axis = 0; axis < 3; ++axis) mix_double(h, job.normal[axis]);
      mix_double(h, job.fill);
    }
    mix_u64(h, ir.hybrid_patches.size());
    for (const MaterialIRHybridPatch &patch : ir.hybrid_patches) {
      mix_u64(h, patch.destination); mix_u64(h, patch.point); mix_double(h, patch.value);
      mix_u64(h, patch.front_material); mix_u64(h, patch.behind_material);
      mix_u64(h, patch.object); mix_u64(h, patch.image); mix_bool(h, patch.ambiguous);
      mix_bool(h, patch.variable_material);
      mix_u64(h, patch.variable_causes);
      mix_bool(h, patch.adaptive_fallback); mix_bool(h, patch.negative_fallback);
      mix_u64(h, uint32_t(patch.reason));
    }
  }
  if (include_rank_layout) {
    mix_u64(h, ir.topology.size());
    for (const MaterialIRTopologyRow &row : ir.topology) {
      mix_i64(h, row.key.chunk); mix_i64(h, row.key.kind); mix_i64(h, row.key.component_);
      mix_i64(h, row.key.cmp); mix_u64(h, row.key.aux); mix_i64(h, int(row.element_type));
      mix_i64(h, int(row.logical_storage)); mix_u64(h, row.elements); mix_u64(h, row.alignment);
      mix_i64(h, row.yee_component);
      for (int axis = 0; axis < 3; ++axis) {
        mix_i64(h, row.extents[axis]); mix_i64(h, row.strides[axis]);
        mix_i64(h, row.stagger[axis]);
      }
    }
  }
  return h;
}

size_t checked_count(double value, const char *what) {
  if (!std::isfinite(value) || value < 0 || std::floor(value) != value ||
      value > double(std::numeric_limits<size_t>::max()))
    throw std::invalid_argument(std::string("material IR ") + what + " count is invalid");
  return size_t(value);
}

void validate_susceptibility_payload(const std::vector<double> &values, size_t &offset) {
  const size_t fixed = 17;
  if (offset > values.size() || values.size() - offset < fixed)
    throw std::invalid_argument("material IR susceptibility schema is short");
  for (size_t i = 13; i < 16; ++i)
    if (values[offset + i] != 0.0 && values[offset + i] != 1.0)
      throw std::invalid_argument("material IR susceptibility boolean is invalid");
  const size_t transitions = checked_count(values[offset + 16], "transition");
  offset += fixed;
  if (transitions > (values.size() - offset) / 9)
    throw std::invalid_argument("material IR transition vector is short");
  for (size_t t = 0; t < transitions; ++t) {
    const size_t base = offset + 9 * t;
    for (int endpoint = 0; endpoint < 2; ++endpoint)
      if (values[base + endpoint] < double(std::numeric_limits<int>::min()) ||
          values[base + endpoint] > double(std::numeric_limits<int>::max()) ||
          std::floor(values[base + endpoint]) != values[base + endpoint])
        throw std::invalid_argument("material IR transition level is invalid");
  }
  offset += transitions * 9;
  if (offset >= values.size())
    throw std::invalid_argument("material IR initial-population count is missing");
  const size_t populations = checked_count(values[offset++], "initial-population");
  if (populations > values.size() - offset)
    throw std::invalid_argument("material IR initial-population vector is short");
  offset += populations;
}

void validate_medium_payload(const std::vector<double> &values, size_t &offset,
                             uint32_t *e_count, uint32_t *h_count) {
  if (offset > values.size() || values.size() - offset < 36)
    throw std::invalid_argument("material IR medium schema is short");
  offset += 36;
  uint32_t counts[2] = {0, 0};
  for (int ft = 0; ft < 2; ++ft) {
    if (offset >= values.size())
      throw std::invalid_argument("material IR susceptibility-list count is missing");
    const size_t count = checked_count(values[offset++], "susceptibility-list");
    if (count > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("material IR susceptibility-list count overflow");
    counts[ft] = uint32_t(count);
    for (size_t i = 0; i < count; ++i) validate_susceptibility_payload(values, offset);
  }
  if (e_count) *e_count = counts[0];
  if (h_count) *h_count = counts[1];
}

void validate_medium_tensor_invertibility(const std::vector<double> &values, size_t base,
                                          bool electric, bool magnetic,
                                          const char *description) {
  if (base > values.size() || values.size() - base < 18)
    throw std::invalid_argument(std::string("material IR ") + description +
                                " tensor payload is short");
  const auto validate = [&](size_t diagonal, size_t offdiagonal, const char *field) {
    const double m00 = values[base + diagonal];
    const double m11 = values[base + diagonal + 1];
    const double m22 = values[base + diagonal + 2];
    const double m01 = values[base + offdiagonal];
    const double m02 = values[base + offdiagonal + 2];
    const double m12 = values[base + offdiagonal + 4];
    /* Match sym_matrix_invert exactly.  Its diagonal fast path intentionally
       permits zero diagonal entries and produces signed infinity.  Once any
       real offdiagonal is present, however, the CPU aborts on an exactly-zero
       determinant, so reject that immutable record before a resident backend
       can allocate or publish a candidate state. */
    if ((m01 != 0.0 || m02 != 0.0 || m12 != 0.0) &&
        m00 * m11 * m22 - m02 * m11 * m02 + 2.0 * m01 * m12 * m02 -
                m01 * m01 * m22 - m12 * m12 * m00 ==
            0.0)
      throw std::invalid_argument(std::string("material IR ") + description + " " + field +
                                  " tensor is singular");
  };
  if (electric) validate(0, 3, "electric");
  if (magnetic) validate(9, 12, "magnetic");
}

bool boxes_intersect(const geom_box &a, const geom_box &b) {
  return a.low.x <= b.high.x && a.high.x >= b.low.x &&
         a.low.y <= b.high.y && a.high.y >= b.low.y &&
         a.low.z <= b.high.z && a.high.z >= b.low.z;
}

vector3 array_vector(const double *values) {
  return {values[0], values[1], values[2]};
}

double dot_cross_extent(vector3 a, vector3 b, vector3 c) {
  const vector3 cross = vector3_cross(b, c);
  return fabs(vector3_norm(cross) / vector3_dot(a, cross));
}

double geometry_product(double a, double b) {
  return (a == 0.0 && std::isinf(b)) || (b == 0.0 && std::isinf(a)) ? 0.0 : a * b;
}

double geometry_dot_row(const double *matrix, int row, vector3 value) {
  return geometry_product(matrix[row], value.x) +
         geometry_product(matrix[3 + row], value.y) +
         geometry_product(matrix[6 + row], value.z);
}

geom_box object_bounds_from_ir(const MaterialIR &ir, const MaterialIRObject &object) {
  geom_box box;
  const vector3 center = array_vector(object.parameters.data());
  box.low = box.high = center;
  const auto add = [&](vector3 point) {
    box.low.x = std::min(box.low.x, point.x); box.high.x = std::max(box.high.x, point.x);
    box.low.y = std::min(box.low.y, point.y); box.high.y = std::max(box.high.y, point.y);
    box.low.z = std::min(box.low.z, point.z); box.high.z = std::max(box.high.z, point.z);
  };
  if (object.kind == geometric_object::SPHERE) {
    const vector3 b1 = array_vector(ir.lattice_basis);
    const vector3 b2 = array_vector(ir.lattice_basis + 3);
    const vector3 b3 = array_vector(ir.lattice_basis + 6);
    const double radius = object.parameters[3];
    const double extent[3] = {dot_cross_extent(b1, b2, b3) * radius,
                              dot_cross_extent(b2, b3, b1) * radius,
                              dot_cross_extent(b3, b1, b2) * radius};
    box.low = {center.x - extent[0], center.y - extent[1], center.z - extent[2]};
    box.high = {center.x + extent[0], center.y + extent[1], center.z + extent[2]};
  }
  else if (object.kind == geometric_object::BLOCK) {
    const vector3 axis[3] = {array_vector(object.parameters.data() + 3),
                             array_vector(object.parameters.data() + 6),
                             array_vector(object.parameters.data() + 9)};
    const double size[3] = {object.parameters[12], object.parameters[13], object.parameters[14]};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      double extent = 0.0;
      for (int axis_index = 0; axis_index < 3; ++axis_index) {
        const double component = coordinate == 0 ? axis[axis_index].x
                                                 : coordinate == 1 ? axis[axis_index].y
                                                                   : axis[axis_index].z;
        extent += fabs(geometry_product(0.5 * size[axis_index], component));
      }
      const double c = coordinate == 0 ? center.x : coordinate == 1 ? center.y : center.z;
      if (coordinate == 0) box.low.x = c - extent, box.high.x = c + extent;
      else if (coordinate == 1) box.low.y = c - extent, box.high.y = c + extent;
      else box.low.z = c - extent, box.high.z = c + extent;
    }
  }
  else if (object.kind == geometric_object::CYLINDER) {
    const vector3 axis = array_vector(object.parameters.data() + 3);
    const double half_height = 0.5 * object.parameters[7];
    const vector3 basis1 = array_vector(ir.cell.data() + 6);
    const vector3 basis2 = array_vector(ir.cell.data() + 9);
    const vector3 basis3 = array_vector(ir.cell.data() + 12);
    const vector3 cart_axis = {
        ir.lattice_basis[0] * axis.x + ir.lattice_basis[3] * axis.y +
            ir.lattice_basis[6] * axis.z,
        ir.lattice_basis[1] * axis.x + ir.lattice_basis[4] * axis.y +
            ir.lattice_basis[7] * axis.z,
        ir.lattice_basis[2] * axis.x + ir.lattice_basis[5] * axis.y +
            ir.lattice_basis[8] * axis.z};
    const vector3 crosses[3] = {vector3_cross(basis2, basis3), vector3_cross(basis3, basis1),
                                vector3_cross(basis1, basis2)};
    double radial[3];
    const vector3 effective[3] = {array_vector(ir.lattice_basis),
                                  array_vector(ir.lattice_basis + 3),
                                  array_vector(ir.lattice_basis + 6)};
    for (int i = 0; i < 3; ++i) {
      const double length2 = vector3_dot(crosses[i], crosses[i]);
      const double projection = vector3_dot(crosses[i], cart_axis);
      radial[i] = fabs(sqrt(fabs(length2 - projection * projection)) /
                       vector3_dot(crosses[i], effective[i]));
    }
    const double radius_low = object.parameters[6];
    const double radius_high = int(object.parameters[8]) == cylinder::CONE
                                   ? fabs(object.parameters[9])
                                   : radius_low;
    const double low_center[3] = {center.x - geometry_product(half_height, axis.x),
                                  center.y - geometry_product(half_height, axis.y),
                                  center.z - geometry_product(half_height, axis.z)};
    const double high_center[3] = {center.x + geometry_product(half_height, axis.x),
                                   center.y + geometry_product(half_height, axis.y),
                                   center.z + geometry_product(half_height, axis.z)};
    for (int i = 0; i < 3; ++i) {
      const double lo = std::min(low_center[i] - radial[i] * radius_low,
                                 high_center[i] - radial[i] * radius_high);
      const double hi = std::max(low_center[i] + radial[i] * radius_low,
                                 high_center[i] + radial[i] * radius_high);
      if (i == 0) box.low.x = lo, box.high.x = hi;
      else if (i == 1) box.low.y = lo, box.high.y = hi;
      else box.low.z = lo, box.high.z = hi;
    }
  }
  else if (object.kind == geometric_object::PRISM) {
    const size_t vertices = object.fixed_vertex_count;
    const double *bottom = object.auxiliary.data();
    const double *top = bottom + 6 * vertices;
    const double *matrix = object.parameters.data() + 20;
    const vector3 centroid = array_vector(object.parameters.data() + 8);
    box.low = {infinity, infinity, infinity};
    box.high = {-infinity, -infinity, -infinity};
    for (size_t i = 0; i < vertices; ++i) {
      for (int surface = 0; surface < 2; ++surface) {
        const double *source = (surface ? top : bottom) + 3 * i;
        const vector3 local = array_vector(source);
        const vector3 point = {centroid.x + geometry_dot_row(matrix, 0, local),
                               centroid.y + geometry_dot_row(matrix, 1, local),
                               centroid.z + geometry_dot_row(matrix, 2, local)};
        add(point);
      }
    }
  }
  else if (object.kind == geometric_object::MESH) {
    bool have_vertex = false;
    for (double encoded : object.indices) {
      const size_t index = size_t(encoded);
      const vector3 vertex = array_vector(object.vertices.data() + 3 * index);
      if (!have_vertex) box.low = box.high = vertex, have_vertex = true;
      else add(vertex);
    }
  }
  return box;
}

void shift_box(geom_box &box, vector3 shift) {
  box.low = vector3_plus(box.low, shift);
  box.high = vector3_plus(box.high, shift);
}

size_t count_flat_images(const geometric_object &object, vector3 shift,
                         const geom_box &bounds) {
  if (object.which_subclass == geometric_object::COMPOUND_GEOMETRIC_OBJECT) {
    const geometric_object_list &children =
        object.subclass.compound_geometric_object_data->component_objects;
    shift = vector3_plus(shift, object.center);
    size_t count = 0;
    for (int i = 0; i < children.num_items; ++i) {
      const size_t child = count_flat_images(children.items[i], shift, bounds);
      if (child > std::numeric_limits<size_t>::max() - count)
        throw std::overflow_error("material IR geometry image count overflow");
      count += child;
    }
    return count;
  }
  geom_box box;
  geom_get_bounding_box(object, &box);
  shift_box(box, shift);
  return boxes_intersect(box, bounds) ? 1 : 0;
}

size_t store_flat_images(MaterialIR &ir, const geometric_object &object, vector3 shift,
                         const int image[3], const geom_box &bounds, int precedence,
                         const std::map<const geometric_object *, uint32_t> &objects) {
  if (object.which_subclass == geometric_object::COMPOUND_GEOMETRIC_OBJECT) {
    const geometric_object_list &children =
        object.subclass.compound_geometric_object_data->component_objects;
    shift = vector3_plus(shift, object.center);
    size_t stored = 0;
    for (int i = 0; i < children.num_items; ++i) {
      const int child_precedence =
          stored > size_t(std::numeric_limits<int>::max()) ||
                  precedence < std::numeric_limits<int>::min() + int(stored)
              ? throw std::overflow_error("material IR compound precedence overflow")
              : precedence - int(stored);
      const size_t child = store_flat_images(ir, children.items[i], shift, image, bounds,
                                             child_precedence, objects);
      if (child > std::numeric_limits<size_t>::max() - stored)
        throw std::overflow_error("material IR geometry image count overflow");
      stored += child;
    }
    return stored;
  }
  const std::map<const geometric_object *, uint32_t>::const_iterator found = objects.find(&object);
  if (found == objects.end())
    throw std::logic_error("material IR flattened object identity is absent");
  const MaterialIRObject &captured = ir.objects[found->second];
  geom_box box;
  box.low = {captured.low[0], captured.low[1], captured.low[2]};
  box.high = {captured.high[0], captured.high[1], captured.high[2]};
  shift_box(box, shift);
  if (!boxes_intersect(box, bounds)) return 0;
  if (ir.images.size() >= std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("material IR geometry image ordinal overflow");
  MaterialIRGeometryImage out = {};
  out.object = found->second;
  out.ordinal = uint32_t(ir.images.size());
  out.precedence = precedence;
  for (int axis = 0; axis < 3; ++axis) out.image[axis] = image[axis];
  out.shift[0] = shift.x; out.shift[1] = shift.y; out.shift[2] = shift.z;
  out.low[0] = box.low.x; out.low[1] = box.low.y; out.low[2] = box.low.z;
  out.high[0] = box.high.x; out.high[1] = box.high.y; out.high[2] = box.high.z;
  ir.images.push_back(out);
  return 1;
}

template <typename Function>
void for_periodic_images(const MaterialIR &ir, Function function) {
  const int dimensions = ir.dimensions == D1 ? 1 : ir.dimensions == D3 ? 3 : 2;
  const double size[3] = {ir.cell[3], ir.cell[4], ir.cell[5]};
  const int x0 = ir.ensure_periodicity ? -1 : 0, x1 = ir.ensure_periodicity ? 1 : 0;
  for (int i = x0; i <= x1; ++i) {
    const int y0 = ir.ensure_periodicity && dimensions >= 2 ? -1 : 0;
    const int y1 = ir.ensure_periodicity && dimensions >= 2 ? 1 : 0;
    for (int j = y0; j <= y1; ++j) {
      const int z0 = ir.ensure_periodicity && dimensions >= 3 ? -1 : 0;
      const int z1 = ir.ensure_periodicity && dimensions >= 3 ? 1 : 0;
      for (int k = z0; k <= z1; ++k) {
        const int image[3] = {i, j, k};
        const vector3 shift = {i * size[0], j * size[1], k * size[2]};
        function(image, shift);
        if (dimensions >= 3 && size[2] == 0.0) break;
      }
      if (dimensions >= 2 && size[1] == 0.0) break;
    }
    if (size[0] == 0.0) break;
  }
}

void capture_geometry_images(MaterialIR &ir, const structure &s,
                             const meep_geom::geom_epsilon &geps,
                             const std::map<const geometric_object *, uint32_t> &objects) {
  const geom_box canonical_bounds = meep_geom::gv2box(geps.captured_volume);
  geom_box local_bounds = {};
  bool have_local_bounds = false;
  for (int i = 0; i < s.num_chunks; ++i) {
    if (!s.chunks[i]->is_mine()) continue;
    const geom_box chunk = meep_geom::gv2box(s.chunks[i]->gv.pad().surroundings());
    if (!have_local_bounds) local_bounds = chunk, have_local_bounds = true;
    else {
      local_bounds.low.x = std::min(local_bounds.low.x, chunk.low.x);
      local_bounds.low.y = std::min(local_bounds.low.y, chunk.low.y);
      local_bounds.low.z = std::min(local_bounds.low.z, chunk.low.z);
      local_bounds.high.x = std::max(local_bounds.high.x, chunk.high.x);
      local_bounds.high.y = std::max(local_bounds.high.y, chunk.high.y);
      local_bounds.high.z = std::max(local_bounds.high.z, chunk.high.z);
    }
  }
  size_t total = 0;
  for (int root = geps.geometry.num_items - 1; root >= 0; --root)
    for_periodic_images(ir, [&](const int *, vector3 shift) {
      const size_t count = count_flat_images(geps.geometry.items[root], shift, canonical_bounds);
      if (count > std::numeric_limits<size_t>::max() - total)
        throw std::overflow_error("material IR geometry image count overflow");
      total += count;
    });
  if (total > size_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("material IR geometry precedence overflow");
  size_t index = 0;
  for (int root = geps.geometry.num_items - 1; root >= 0; --root) {
    const int precedence = int(total - index);
    for_periodic_images(ir, [&](const int image[3], vector3 shift) {
      const size_t stored = store_flat_images(ir, geps.geometry.items[root], shift, image,
                                              canonical_bounds, precedence, objects);
      if (stored > total - index)
        throw std::logic_error("material IR geometry image count changed while capturing");
      index += stored;
    });
  }
  if (index != total || ir.images.size() != total)
    throw std::logic_error("material IR geometry image capture is inconsistent");
  if (have_local_bounds)
    for (uint32_t image = 0; image < ir.images.size(); ++image) {
      geom_box box;
      box.low = {ir.images[image].low[0], ir.images[image].low[1], ir.images[image].low[2]};
      box.high = {ir.images[image].high[0], ir.images[image].high[1], ir.images[image].high[2]};
      if (boxes_intersect(box, local_bounds)) ir.active_images.push_back(image);
    }
}

struct FrontObjectResult {
  uint32_t object;
  uint32_t image;
  uint32_t front;
  uint32_t behind;
  int precedence;
  bool ambiguous;
  bool variable_material;
  uint32_t variable_causes;
};

uint32_t material_variable_cause(const MaterialIR &ir, uint32_t material) {
  switch (ir.materials[material].kind) {
    case meep_geom::material_data::MATERIAL_GRID: return material_variable_grid;
    case meep_geom::material_data::MATERIAL_USER: return material_variable_user;
    case meep_geom::material_data::MATERIAL_FILE: return material_variable_file_default;
    default: return material_variable_none;
  }
}

bool material_records_equal(const MaterialIR &ir, uint32_t a_index, uint32_t b_index) {
  if (a_index == b_index) return true;
  const MaterialIRMaterial &a = ir.materials[a_index], &b = ir.materials[b_index];
  if (a.kind != b.kind) return false;
  switch (a.kind) {
    case meep_geom::material_data::MATERIAL_FILE:
    case meep_geom::material_data::PERFECT_METAL: return true;
    case meep_geom::material_data::MEDIUM:
    case meep_geom::material_data::MATERIAL_GRID:
      return a.comparison_medium == b.comparison_medium;
    case meep_geom::material_data::MATERIAL_USER:
    default: return false;
  }
}

int object_subtype(const MaterialIRObject &object) {
  if (object.kind == geometric_object::BLOCK) return int(object.parameters[24]);
  if (object.kind == geometric_object::CYLINDER) return int(object.parameters[8]);
  if (object.kind == geometric_object::MESH) return int(object.parameters[3]);
  return 0;
}

bool image_contains(const MaterialIR &ir, uint32_t image_index, vector3 point) {
  const MaterialIRGeometryImage &image = ir.images[image_index];
  if (point.x < image.low[0] || point.x > image.high[0] || point.y < image.low[1] ||
      point.y > image.high[1] || point.z < image.low[2] || point.z > image.high[2])
    return false;
  const MaterialIRObject &object = ir.objects[image.object];
  const material_geometry_numeric::vector p = {
      point.x - image.shift[0], point.y - image.shift[1], point.z - image.shift[2]};
  return material_geometry_numeric::contains(
      object.kind, object_subtype(object), object.parameters.data(), object.vertices.data(),
      object.kind == geometric_object::PRISM ? object.fixed_vertex_count
                                             : object.vertices.size() / 3,
      object.indices.data(), object.indices.size(),
      object.auxiliary.data(), object.mesh_lengthscale, ir.lattice_metric, p,
      ir.prism_include_boundaries);
}

struct PointObjectResult {
  uint32_t object;
  uint32_t image;
  uint32_t material;
  int precedence;
};

uint32_t point_variable_cause(const MaterialIR &ir, const PointObjectResult &point) {
  if (point.object == std::numeric_limits<uint32_t>::max())
    return material_variable_cause(ir, ir.default_material);
  const uint32_t source = uint32_t(ir.objects[point.object].material);
  if (ir.materials[source].kind == meep_geom::material_data::MATERIAL_FILE)
    return material_variable_cause(ir, ir.default_material);
  return material_variable_cause(ir, source);
}

PointObjectResult object_at_point(const MaterialIR &ir, vector3 point) {
  for (uint32_t active : ir.active_images)
    if (image_contains(ir, active, point)) {
      const MaterialIRGeometryImage &image = ir.images[active];
      return PointObjectResult{image.object, active, uint32_t(ir.objects[image.object].material),
                               image.precedence};
    }
  return PointObjectResult{std::numeric_limits<uint32_t>::max(),
                           std::numeric_limits<uint32_t>::max(), ir.default_material, 0};
}

double grid_sample(const MaterialIRMaterial &material,
                   material_geometry_numeric::vector coordinate) {
  if (material.kind != meep_geom::material_data::MATERIAL_GRID || material.parameters.size() < 3)
    throw std::invalid_argument("material IR grid sample requested from a non-grid material");
  uint32_t dimensions[3];
  for (int axis = 0; axis < 3; ++axis) {
    const size_t extent = checked_count(material.parameters[axis], "grid dimension");
    if (!extent || extent > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument("material IR grid dimension is not compact");
    dimensions[axis] = uint32_t(extent);
  }
  return material_geometry_numeric::interpolate(material.samples.data(), dimensions,
                                                coordinate.x, coordinate.y, coordinate.z);
}

double grid_value_at_point(const MaterialIR &ir, vector3 point, uint32_t winning_image) {
  uint32_t winning_material = ir.default_material;
  size_t active_position = ir.active_images.size();
  if (winning_image != std::numeric_limits<uint32_t>::max()) {
    if (winning_image >= ir.images.size())
      throw std::invalid_argument("material IR grid winner image is invalid");
    winning_material = uint32_t(ir.objects[ir.images[winning_image].object].material);
    const std::vector<uint32_t>::const_iterator found =
        std::lower_bound(ir.active_images.begin(), ir.active_images.end(), winning_image);
    if (found == ir.active_images.end() || *found != winning_image)
      throw std::invalid_argument("material IR grid winner is not locally active");
    active_position = size_t(found - ir.active_images.begin());
  }
  const MaterialIRMaterial &winner = ir.materials[winning_material];
  if (winner.kind != meep_geom::material_data::MATERIAL_GRID)
    throw std::invalid_argument("material IR grid winner is not a MaterialGrid");
  double product = 1.0, minimum = 1.0, sum = 0.0, default_value = 0.0;
  size_t count = 0;
  bool exhausted = true;
  if (winning_image != std::numeric_limits<uint32_t>::max()) {
    for (size_t position = active_position; position < ir.active_images.size(); ++position) {
      const uint32_t image_index = ir.active_images[position];
      if (!image_contains(ir, image_index, point)) continue;
      const MaterialIRGeometryImage &image = ir.images[image_index];
      const MaterialIRObject &object = ir.objects[image.object];
      const MaterialIRMaterial &material = ir.materials[object.material];
      if (material.kind != meep_geom::material_data::MATERIAL_GRID) {
        exhausted = false;
        break;
      }
      material_geometry_numeric::vector local = {
          point.x - image.shift[0], point.y - image.shift[1], point.z - image.shift[2]};
      local = material_geometry_numeric::object_coordinates(object.kind,
                                                            object.parameters.data(), local);
      const double value = grid_sample(material, local);
      if (winner.material_grid_kind == meep_geom::material_data::U_DEFAULT) {
        default_value = value;
        exhausted = false;
        break;
      }
      minimum = std::min(minimum, value);
      product *= value;
      sum += value;
      ++count;
    }
  }
  if (exhausted && ir.materials[ir.default_material].kind ==
                       meep_geom::material_data::MATERIAL_GRID) {
    const material_geometry_numeric::vector global = {
        ir.cell[3] == 0.0 ? 0.0 : 0.5 + (point.x - ir.cell[0]) / ir.cell[3],
        ir.cell[4] == 0.0 ? 0.0 : 0.5 + (point.y - ir.cell[1]) / ir.cell[4],
        ir.cell[5] == 0.0 ? 0.0 : 0.5 + (point.z - ir.cell[2]) / ir.cell[5]};
    const double value = grid_sample(ir.materials[ir.default_material], global);
    if (!count) default_value = value;
    minimum = std::min(minimum, value);
    product *= value;
    sum += value;
    ++count;
  }
  double value = winner.material_grid_kind == meep_geom::material_data::U_MIN
                     ? minimum
                     : winner.material_grid_kind == meep_geom::material_data::U_PROD
                           ? product
                           : winner.material_grid_kind == meep_geom::material_data::U_MEAN
                                 ? sum / double(count)
                                 : default_value;
  value += ir.projection_offset;
  const size_t tail = winner.parameters.size() - 3;
  (void)tail;
  size_t offset = 3;
  uint32_t e = 0, h = 0;
  validate_medium_payload(winner.parameters, offset, &e, &h);
  validate_medium_payload(winner.parameters, offset, &e, &h);
  const double beta = winner.parameters[offset], eta = winner.parameters[offset + 1];
  if (beta == 0.0) return value;
  if (value == eta) return 0.5;
  const double tanh_beta_eta = tanh(beta * eta);
  return (tanh_beta_eta + tanh(beta * (value - eta))) /
         (tanh_beta_eta + tanh(beta * (1.0 - eta)));
}

FrontObjectResult classify_front_object(const MaterialIR &ir, const meep::volume &v) {
  const int num_neighbors[3] = {3, 5, 9};
  const int neighbors[3][9][3] = {{{0, 0, 0}, {0, 0, -1}, {0, 0, 1}},
                                  {{0, 0, 0}, {-1, -1, 0}, {1, 1, 0},
                                   {-1, 1, 0}, {1, -1, 0}},
                                  {{0, 0, 0}, {1, 1, 1}, {1, 1, -1}, {1, -1, 1},
                                   {1, -1, -1}, {-1, 1, 1}, {-1, 1, -1},
                                   {-1, -1, 1}, {-1, -1, -1}}};
  const geom_box pixel = meep_geom::gv2box(v);
  const vector3 center = meep_geom::vec_to_vector3(v.center());
  const double half[3] = {(pixel.high.x - pixel.low.x) * 0.5,
                          (pixel.high.y - pixel.low.y) * 0.5,
                          (pixel.high.z - pixel.low.z) * 0.5};
  uint32_t selected[2] = {std::numeric_limits<uint32_t>::max(),
                          std::numeric_limits<uint32_t>::max()};
  uint32_t selected_images[2] = {std::numeric_limits<uint32_t>::max(),
                                 std::numeric_limits<uint32_t>::max()};
  int ids[2] = {-1, -1};
  uint32_t materials[2] = {ir.default_material, ir.default_material};
  uint32_t variable_causes = material_variable_none;
  const int dimension_index = meep::number_of_directions(v.dim) - 1;
  for (int i = 0; i < num_neighbors[dimension_index]; ++i) {
    vector3 q = {center.x + neighbors[dimension_index][i][0] * half[0],
                 center.y + neighbors[dimension_index][i][1] * half[1],
                 center.z + neighbors[dimension_index][i][2] * half[2]};
    const PointObjectResult point = object_at_point(ir, q);
    variable_causes |= point_variable_cause(ir, point);
    const int id = point.precedence;
    if ((ids[0] != -1 && point.object == selected[0] && point.image == selected_images[0]) ||
        (ids[1] != -1 && point.object == selected[1] && point.image == selected_images[1]))
      continue;
    uint32_t material = point.material;
    if (point.object != std::numeric_limits<uint32_t>::max() &&
        ir.materials[material].kind == meep_geom::material_data::MATERIAL_FILE)
      material = ir.default_material;
    if (ids[0] == -1) {
      selected[0] = point.object; selected_images[0] = point.image; ids[0] = id;
      materials[0] = material;
    }
    else if (ids[1] == -1 ||
             ((id >= ids[0] && id >= ids[1]) &&
              (ids[0] == ids[1] || material_records_equal(ir, materials[0], materials[1])))) {
      selected[1] = point.object; selected_images[1] = point.image; ids[1] = id;
      materials[1] = material;
    }
    else if (!(ids[0] < ids[1] &&
               (ids[0] == id || material_records_equal(ir, materials[0], material))) &&
             !(ids[1] < ids[0] &&
               (ids[1] == id || material_records_equal(ir, materials[1], material))))
      return FrontObjectResult{selected[0], selected_images[0], materials[0], materials[1],
                               ids[0], true, variable_causes != material_variable_none,
                               variable_causes};
  }
  if (ids[1] == -1) {
    selected[1] = selected[0]; selected_images[1] = selected_images[0]; ids[1] = ids[0];
    materials[1] = materials[0];
  }
  if (variable_causes != material_variable_none) {
      const int front = ids[0] >= ids[1] ? 0 : 1;
      const int behind = front ? 0 : 1;
      return FrontObjectResult{selected[front], selected_images[front], materials[front],
                               materials[behind], ids[front], true, true, variable_causes};
    }
  const int front = ids[0] >= ids[1] ? 0 : 1;
  const int behind = front ? 0 : 1;
  return FrontObjectResult{selected[front], selected_images[front], materials[front],
                           materials[behind], ids[front], false, false,
                           material_variable_none};
}

bool ir_material_is_metal(const MaterialIRMaterial &material, field_type ft) {
  if (material.kind == meep_geom::material_data::PERFECT_METAL) return ft == E_stuff;
  if (material.kind != meep_geom::material_data::MEDIUM &&
      material.kind != meep_geom::material_data::MATERIAL_FILE)
    return false;
  const size_t diagonal = ft == E_stuff ? 0 : 9;
  return material.parameters.size() > diagonal + 2 &&
         (material.parameters[diagonal] < 0.0 || material.parameters[diagonal + 1] < 0.0 ||
          material.parameters[diagonal + 2] < 0.0);
}

bool axis_aligned_block_face(const MaterialIR &ir, uint32_t object_index, uint32_t image_index,
                             const meep::volume &v, vector3 &normal, double &fill) {
  const MaterialIRObject &object = ir.objects[object_index];
  const MaterialIRGeometryImage &image = ir.images[image_index];
  if (object.kind != geometric_object::BLOCK || int(object.parameters[24]) != block::BLOCK_SELF)
    return false;
  const double projection[3][3] = {{object.parameters[15], object.parameters[18],
                                    object.parameters[21]},
                                   {object.parameters[16], object.parameters[19],
                                    object.parameters[22]},
                                   {object.parameters[17], object.parameters[20],
                                    object.parameters[23]}};
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column) {
      const double expected = row == column ? 1.0 : 0.0;
      if (projection[row][column] != expected) return false;
    }
  const geom_box pixel = meep_geom::gv2box(v);
  const vector3 point = meep_geom::vec_to_vector3(v.center());
  const double center[3] = {object.parameters[0] + image.shift[0],
                            object.parameters[1] + image.shift[1],
                            object.parameters[2] + image.shift[2]};
  const double size[3] = {object.parameters[12], object.parameters[13], object.parameters[14]};
  const double low[3] = {pixel.low.x, pixel.low.y, pixel.low.z};
  const double high[3] = {pixel.high.x, pixel.high.y, pixel.high.z};
  int partial_axis = -1;
  fill = 1.0;
  for (int axis = 0; axis < 3; ++axis) {
    if (low[axis] == high[axis]) continue;
    const double object_low = center[axis] - 0.5 * size[axis];
    const double object_high = center[axis] + 0.5 * size[axis];
    const double overlap = std::max(0.0, std::min(high[axis], object_high) -
                                             std::max(low[axis], object_low));
    const double fraction = overlap / (high[axis] - low[axis]);
    if (!(fraction > 0.0)) return false;
    if (fraction < 1.0) {
      if (partial_axis >= 0) return false;
      partial_axis = axis;
    }
    fill *= fraction;
  }
  if (partial_axis < 0 || !(fill > 0.0 && fill < 1.0)) return false;
  const double projected[3] = {point.x - center[0], point.y - center[1],
                               point.z - center[2]};
  const double distance[3] = {fabs(fabs(projected[0]) - 0.5 * size[0]),
                              fabs(fabs(projected[1]) - 0.5 * size[1]),
                              fabs(fabs(projected[2]) - 0.5 * size[2])};
  const int nearest = distance[0] < distance[1] && distance[0] < distance[2]
                          ? 0
                          : distance[1] < distance[2] ? 1 : 2;
  if (nearest != partial_axis) return false;
  normal = {0, 0, 0};
  if (partial_axis == 0) normal.x = 1;
  else if (partial_axis == 1) normal.y = 1;
  else normal.z = 1;
  return true;
}

direction ir_yucky_direction(ndim dimensions, int axis) {
  if (dimensions == Dcyl) {
    static const direction cylindrical[3] = {P, R, Z};
    return cylindrical[axis];
  }
  if (dimensions == D2) {
    static const direction planar[3] = {Z, X, Y};
    return planar[axis];
  }
  return direction(axis);
}

meep::volume destination_evaluation_volume(const MaterialIR &ir,
                                           const MaterialIRDestination &destination,
                                           uint64_t point) {
  const MaterialIRChunk &chunk = ir.chunks[destination.chunk_index];
  const component c = component(destination.component);
  uint64_t count[3];
  for (int axis = 0; axis < 3; ++axis)
    count[axis] = uint64_t((int64_t(chunk.loop_end[c][axis]) -
                            int64_t(chunk.loop_begin[c][axis])) /
                               2 +
                           1);
  if (!count[1] || !count[2] || point >= count[0] * count[1] * count[2])
    throw std::invalid_argument("material IR destination point ordinal is invalid");
  const uint64_t plane = count[1] * count[2];
  const uint64_t coordinate[3] = {point / plane, (point % plane) / count[2], point % count[2]};
  int here[3];
  for (int axis = 0; axis < 3; ++axis)
    here[axis] = chunk.loop_begin[c][axis] + int(2 * coordinate[axis]);
  if (destination.offdiagonal) {
    const direction shifted = component_direction(c);
    for (int axis = 0; axis < 3; ++axis)
      if (ir_yucky_direction(ndim(chunk.dimensions), axis) == shifted)
        here[axis] -= type(c) == E_stuff ? 1 : -1;
  }
  meep::volume evaluation(ndim(chunk.dimensions));
  const double half = 0.5 * chunk.inva;
  for (int axis = 0; axis < 3; ++axis) {
    const direction d = ir_yucky_direction(ndim(chunk.dimensions), axis);
    if (!has_direction(ndim(chunk.dimensions), d)) continue;
    const double center = 0.5 * chunk.inva * here[axis];
    evaluation.set_direction_min(d, center - half);
    evaluation.set_direction_max(d, center + half);
  }
  if (chunk.dimensions == Dcyl) {
    for (int axis = 0; axis < 3; ++axis)
      if (ir_yucky_direction(Dcyl, axis) == R && here[axis] == 0)
        evaluation.set_direction_min(R, 0.0);
  }
  return evaluation;
}

int tensor_column_for(const MaterialIRTopologyRow &row, int dimensions) {
  const component c = component(row.key.component_);
  const direction d = direction(row.key.aux);
  if (dimensions == Dcyl) {
    if (d == R) return 0;
    if (d == P) return 1;
    if (d == Z) return 2;
  }
  else {
    if (d == X) return 0;
    if (d == Y) return 1;
    if (d == Z) return 2;
  }
  (void)c;
  return -1;
}

MaterialIRProperty property_for(array_kind kind) {
  switch (kind) {
    case array_kind::chi1inv: return MaterialIRProperty::chi1inv;
    case array_kind::conductivity: return MaterialIRProperty::conductivity;
    case array_kind::condinv: return MaterialIRProperty::condinv;
    case array_kind::chi2: return MaterialIRProperty::chi2;
    case array_kind::chi3: return MaterialIRProperty::chi3;
    case array_kind::sigma: return MaterialIRProperty::sigma;
    default: throw std::invalid_argument("material IR destination property is unsupported");
  }
}

void append_bulk_point(MaterialIR &ir, uint32_t destination, uint64_t point) {
  if (!ir.bulk_spans.empty()) {
    MaterialIRBulkSpan &last = ir.bulk_spans.back();
    if (last.destination == destination && last.first_point + last.count == point) {
      if (last.count == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("material IR bulk span count overflow");
      ++last.count;
      return;
    }
  }
  ir.bulk_spans.push_back(MaterialIRBulkSpan{destination, point, 1});
}

} // namespace

uint32_t material_ir_material_at_point(const MaterialIR &ir, const double point[3],
                                       uint32_t *image) {
  if (!point) throw std::invalid_argument("material IR point is null");
  const PointObjectResult result = object_at_point(ir, {point[0], point[1], point[2]});
  if (image) *image = result.image;
  return result.material;
}

bool material_ir_materials_equal(const MaterialIR &ir, uint32_t a, uint32_t b) {
  if (a >= ir.materials.size() || b >= ir.materials.size())
    throw std::invalid_argument("material IR material equality index is invalid");
  return material_records_equal(ir, a, b);
}

double material_ir_grid_value_at_point(const MaterialIR &ir, const double point[3],
                                       uint32_t winning_image) {
  if (!point) throw std::invalid_argument("material IR point is null");
  return grid_value_at_point(ir, {point[0], point[1], point[2]}, winning_image);
}

vec material_ir_destination_center(const MaterialIR &ir,
                                   const MaterialIRDestination &destination,
                                   uint64_t point) {
  return destination_evaluation_volume(ir, destination, point).center();
}

size_t material_ir_destination_storage_index(const MaterialIR &ir,
                                             const MaterialIRDestination &destination,
                                             uint64_t point) {
  if (destination.chunk_index >= ir.chunks.size())
    throw std::invalid_argument("material destination chunk is invalid");
  const MaterialIRChunk &chunk = ir.chunks[destination.chunk_index];
  const component c = component(destination.component);
  size_t extent[3];
  for (int axis = 0; axis < 3; ++axis) {
    const int64_t doubled = int64_t(chunk.loop_end[c][axis]) - chunk.loop_begin[c][axis];
    if (doubled < 0 || doubled % 2)
      throw std::invalid_argument("material destination loop extent is invalid");
    extent[axis] = size_t(doubled / 2) + 1;
  }
  if (!extent[1] || !extent[2] || point >= extent[0] * extent[1] * extent[2])
    throw std::invalid_argument("material destination point ordinal is invalid");
  const size_t index[3] = {size_t(point / (extent[1] * extent[2])),
                           size_t((point / extent[2]) % extent[1]),
                           size_t(point % extent[2])};
  size_t destination_index = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const int64_t stagger = int64_t(chunk.loop_begin[c][axis]) - chunk.little_corner[axis];
    if (stagger < 0 || stagger > 1 || chunk.strides[axis] < 0)
      throw std::invalid_argument("material destination scatter metadata is invalid");
    const size_t coordinate = size_t(stagger / 2) + index[axis];
    const size_t stride = size_t(chunk.strides[axis]);
    if (stride && coordinate > std::numeric_limits<size_t>::max() / stride)
      throw std::overflow_error("material destination scatter overflows");
    const size_t term = coordinate * stride;
    if (term > std::numeric_limits<size_t>::max() - destination_index)
      throw std::overflow_error("material destination scatter overflows");
    destination_index += term;
  }
  if (destination.topology_index >= ir.topology.size() ||
      destination_index >= ir.topology[destination.topology_index].elements)
    throw std::invalid_argument("material destination scatter exceeds its row");
  return destination_index;
}

std::shared_ptr<const void> capture_material_ir(const structure &s,
                                               meep_geom::geom_epsilon &geps,
                                               bool eps_averaging, double tol, int maxeval,
                                               const void *absorbers) {
  if (material_ir_capture_failure_rank == my_rank()) {
    if (material_ir_capture_failure_mode == 1)
      throw std::invalid_argument("injected material IR capture failure");
    if (material_ir_capture_failure_mode == 2) throw std::bad_alloc();
  }
  std::shared_ptr<CapturedMaterialIR> captured(new CapturedMaterialIR);
  captured->ir.reset(new MaterialIR);
  std::shared_ptr<MaterialCallbackOwners> captured_owners(new MaterialCallbackOwners);
  captured->owners = captured_owners;
  std::shared_ptr<MaterialIR> ir = captured->ir;
  ir->version = material_ir_version; ir->eps_averaging = eps_averaging;
  ir->subpixel_tol = tol; ir->subpixel_maxeval = eps_averaging ? maxeval : 0;
  ir->ensure_periodicity = geps.captured_ensure_periodicity; ir->contains_host_callback = false;
  ir->requires_hybrid = false;
  ir->prism_include_boundaries = prism_boundary_policy();
  if (s.num_chunks <= 0 || !s.chunks[0])
    throw std::invalid_argument("material IR has no live chunk dimension authority");
  ir->dimensions = int(s.chunks[0]->gv.dim);
  ir->projection_offset = geps.u_p;
  ir->signature = 0; ir->layout_signature = 0;
  append_vec(ir->cell, geps.captured_geometry_center);
  append_vec(ir->cell, geps.captured_geometry_lattice.size);
  append_vec(ir->cell, geps.captured_geometry_lattice.basis1);
  append_vec(ir->cell, geps.captured_geometry_lattice.basis2);
  append_vec(ir->cell, geps.captured_geometry_lattice.basis3);
  const geom_box captured_bounds = meep_geom::gv2box(geps.captured_volume);
  ir->captured_volume[0] = captured_bounds.low.x;
  ir->captured_volume[1] = captured_bounds.low.y;
  ir->captured_volume[2] = captured_bounds.low.z;
  ir->captured_volume[3] = captured_bounds.high.x;
  ir->captured_volume[4] = captured_bounds.high.y;
  ir->captured_volume[5] = captured_bounds.high.z;
  const lattice &lattice = geps.captured_geometry_lattice;
  const matrix3x3 inverse = matrix3x3_inverse(lattice.basis);
  const matrix3x3 inverse_transpose = matrix3x3_transpose(inverse);
  const vector3 basis_size = lattice.basis_size;
  const vector3 basis_columns[3] = {lattice.basis.c0, lattice.basis.c1, lattice.basis.c2};
  const vector3 metric_columns[3] = {lattice.metric.c0, lattice.metric.c1, lattice.metric.c2};
  const vector3 inverse_columns[3] = {inverse.c0, inverse.c1, inverse.c2};
  const vector3 inverse_transpose_columns[3] = {inverse_transpose.c0, inverse_transpose.c1,
                                                inverse_transpose.c2};
  ir->lattice_basis_size[0] = basis_size.x;
  ir->lattice_basis_size[1] = basis_size.y;
  ir->lattice_basis_size[2] = basis_size.z;
  for (int column = 0; column < 3; ++column) {
    const vector3 values[4] = {basis_columns[column], metric_columns[column],
                               inverse_columns[column], inverse_transpose_columns[column]};
    for (int row = 0; row < 3; ++row) {
      const double element[4] = {
          row == 0 ? values[0].x : row == 1 ? values[0].y : values[0].z,
          row == 0 ? values[1].x : row == 1 ? values[1].y : values[1].z,
          row == 0 ? values[2].x : row == 1 ? values[2].y : values[2].z,
          row == 0 ? values[3].x : row == 1 ? values[3].y : values[3].z};
      ir->lattice_basis[3 * column + row] = element[0];
      ir->lattice_metric[3 * column + row] = element[1];
      ir->lattice_inverse[3 * column + row] = element[2];
      ir->lattice_inverse_transpose[3 * column + row] = element[3];
    }
  }
  std::map<const void *, uint32_t> materials;
  std::map<const geometric_object *, uint32_t> objects;
  ir->default_material =
      capture_material(*ir, &geps.owned_default_material(), materials, *captured_owners);
  if (geps.geometry.num_items < 0)
    throw std::invalid_argument("material IR geometry has a negative root count");
  ir->root_count = uint32_t(geps.geometry.num_items);
  const vector3 no_parent_shift = {0, 0, 0};
  for (int i = 0; i < geps.geometry.num_items; ++i)
    capture_object(*ir, geps.geometry.items[i], uint32_t(i), no_parent_shift, materials, objects,
                   *captured_owners);
  const meep_geom::material_type_list &extra = geps.owned_extra_materials();
  if (extra.num_items < 0)
    throw std::invalid_argument("material IR extra-material count is negative");
  for (int i = 0; i < extra.num_items; ++i)
    ir->extra_materials.push_back(
        capture_material(*ir, extra.items[i], materials, *captured_owners));
  for (int which = 0; which < 2; ++which) {
    const field_type ft = which == 0 ? E_stuff : H_stuff;
    const std::vector<meep_geom::susceptibility> unique =
        geps.owned_unique_susceptibilities(ft);
    if (unique.size() > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("material IR susceptibility count overflow");
    /* geom_epsilon builds the CPU's temporary unique list by prepending, then
       structure_chunk prepends each created state once more.  Reverse the
       owned temporary order here so identity matches the final CPU state_index. */
    for (uint32_t si = 0; si < unique.size(); ++si) {
      MaterialIRSusceptibility sus;
      sus.identity = si; sus.material = std::numeric_limits<uint32_t>::max();
      sus.field_type = ft; sus.material_ordinal = si;
      append_susceptibility(sus.parameters, unique[unique.size() - 1 - si]);
      ir->susceptibilities.push_back(sus);
    }
  }
  for (int i = 0; i < s.num_chunks; ++i) {
    const grid_volume &gv = s.chunks[i]->gv;
    MaterialIRChunk chunk;
    chunk.chunk = i; chunk.dimensions = int(gv.dim); chunk.owned = s.chunks[i]->is_mine();
    chunk.resolution = gv.a; chunk.inva = gv.inva;
    if (gv.ntot() <= 0) throw std::invalid_argument("material IR chunk extent is invalid");
    chunk.elements = size_t(gv.ntot());
    chunk.component_bits = 0;
    FOR_COMPONENTS(c) if (gv.has_field(c))
      chunk.component_bits |= uint64_t(1) << int(c);
    for (int axis = 0; axis < 3; ++axis) {
      const direction d = gv.yucky_direction(axis);
      chunk.little_corner[axis] = gv.little_corner().yucky_val(axis);
      chunk.big_corner[axis] = gv.big_corner().yucky_val(axis);
      const int64_t doubled_extent = int64_t(chunk.big_corner[axis]) -
                                     int64_t(chunk.little_corner[axis]);
      if (doubled_extent < 0 || doubled_extent % 2 ||
          doubled_extent / 2 == std::numeric_limits<int>::max())
        throw std::invalid_argument("material IR chunk axis extent is invalid");
      chunk.extents[axis] = int(doubled_extent / 2 + 1);
      chunk.strides[axis] = gv.stride(d);
      if (chunk.strides[axis] < 0)
        throw std::invalid_argument("material IR chunk stride is negative");
      chunk.origin[axis] = gv.get_origin().in_direction(d);
      FOR_COMPONENTS(c) {
        const int shift = gv.has_field(c) ? gv.iyee_shift(c).in_direction(d) : 0;
        const int64_t begin = int64_t(chunk.little_corner[axis]) + shift;
        const int64_t end = int64_t(chunk.big_corner[axis]) + shift;
        if (begin < std::numeric_limits<int>::min() ||
            begin > std::numeric_limits<int>::max() ||
            end < std::numeric_limits<int>::min() || end > std::numeric_limits<int>::max())
          throw std::overflow_error("material IR component loop bound overflow");
        chunk.stagger[c][axis] = shift;
        chunk.loop_begin[c][axis] = gv.has_field(c) ? int(begin) : 0;
        chunk.loop_end[c][axis] = gv.has_field(c) ? int(end) : 0;
      }
    }
    FOR_COMPONENTS(c) {
      chunk.loop_count[c] = 0;
      if (!gv.has_field(c)) continue;
      size_t count = 1;
      ptrdiff_t maximum = 0;
      for (int axis = 0; axis < 3; ++axis) {
        const int64_t delta = int64_t(chunk.loop_end[c][axis]) -
                              int64_t(chunk.loop_begin[c][axis]);
        if (delta < 0 || delta % 2)
          throw std::invalid_argument("material IR component loop bounds are invalid");
        const size_t axis_count = size_t(delta / 2) + 1;
        if (axis_count && count > std::numeric_limits<size_t>::max() / axis_count)
          throw std::overflow_error("material IR component loop count overflow");
        count *= axis_count;
        const int64_t coordinate64 =
            (int64_t(chunk.loop_end[c][axis]) - chunk.little_corner[axis]) / 2;
        if (coordinate64 < 0 || uint64_t(coordinate64) > uint64_t(PTRDIFF_MAX))
          throw std::overflow_error("material IR component address range overflow");
        const ptrdiff_t coordinate = ptrdiff_t(coordinate64);
        if ((coordinate && chunk.strides[axis] >
                                           std::numeric_limits<ptrdiff_t>::max() / coordinate) ||
            maximum > std::numeric_limits<ptrdiff_t>::max() - coordinate * chunk.strides[axis])
          throw std::overflow_error("material IR component address range overflow");
        maximum += coordinate * chunk.strides[axis];
      }
      if (count != chunk.elements || maximum < 0 || size_t(maximum) >= chunk.elements)
        throw std::invalid_argument("material IR component loop exceeds chunk storage");
      chunk.loop_count[c] = count;
    }
    for (int d = 0; d < 6; ++d) {
      if (s.chunks[i]->sigsize[d] < 0)
        throw std::invalid_argument("material IR PML extent is negative");
      if (uint64_t(s.chunks[i]->sigsize[d]) >
          uint64_t(std::numeric_limits<int>::max()))
        throw std::overflow_error("material IR PML extent exceeds integer layout range");
      chunk.pml_elements[d] = size_t(s.chunks[i]->sigsize[d]);
    }
    ir->chunks.push_back(chunk);
    if (chunk.owned) for (int d = 0; d < 6; ++d) if (s.chunks[i]->sigsize[d]) {
      const size_t n = size_t(s.chunks[i]->sigsize[d]);
      if (!s.chunks[i]->sig[d] || !s.chunks[i]->kap[d] || !s.chunks[i]->siginv[d])
        throw std::invalid_argument("material IR PML axis has a null source");
      MaterialIRPmlAxis axis;
      axis.chunk = i; axis.direction = d; axis.elements = n;
      axis.little_corner = s.chunks[i]->gv.little_corner().in_direction(direction(d));
      axis.resolution = s.chunks[i]->a;
      const structure_chunk::pml_initialization_recipe &recipe = s.chunks[i]->pml_recipe[d];
      axis.profile_active = recipe.active;
      axis.analytic_quadratic = recipe.analytic_quadratic;
      axis.thickness = recipe.thickness;
      axis.boundary_location = recipe.boundary_location;
      axis.r_asymptotic = recipe.r_asymptotic;
      axis.mean_stretch = recipe.mean_stretch;
      axis.profile_integral = recipe.profile_integral;
      axis.profile_integral_u = recipe.profile_integral_u;
      axis.profile_samples = recipe.profile_samples;
      axis.sigma.assign(s.chunks[i]->sig[d], s.chunks[i]->sig[d] + n);
      axis.kappa.assign(s.chunks[i]->kap[d], s.chunks[i]->kap[d] + n);
      axis.sigma_inv.assign(s.chunks[i]->siginv[d], s.chunks[i]->siginv[d] + n);
      ir->pml_axes.push_back(axis);
    }
  }
  capture_geometry_images(*ir, s, geps, objects);
  for (int d = 0; d < 5; ++d) for (int side = 0; side < 2; ++side) {
    const meep_geom::cond_profile &profile = geps.cond[d][side];
    if (!profile.prof) continue;
    if (profile.N < 0) throw std::invalid_argument("material IR absorber count is negative");
    MaterialIRPml p;
    p.direction = d; p.side = side; p.thickness = profile.L;
    p.r_asymptotic = 1e-15; p.mean_stretch = 1;
    p.sample_spacing = profile.N ? profile.L / profile.N : 0;
    p.samples.assign(profile.prof, profile.prof + size_t(profile.N) + 1);
    ir->absorbers.push_back(p);
  }
  if (absorbers) {
    const meep_geom::absorber_list_type &layers =
        *static_cast<const meep_geom::absorber_list_type *>(absorbers);
    for (const meep_geom::absorber &layer : layers)
      for (MaterialIRPml &p : ir->absorbers)
        if ((layer.direction == meep_geom::ALL_DIRECTIONS || layer.direction == p.direction) &&
            (layer.side == meep_geom::ALL_SIDES || layer.side == p.side) && layer.thickness == p.thickness) {
          p.r_asymptotic = layer.R_asymptotic; p.mean_stretch = layer.mean_stretch;
        }
  }
  ir->contains_host_callback = ir->materials[ir->default_material].host_callback;
  for (const MaterialIRObject &object : ir->objects)
    ir->contains_host_callback =
        ir->contains_host_callback || ir->materials[object.material].host_callback;
  ir->device_native_eligible = !ir->contains_host_callback;
  /* Absorbing layers also populate the legacy conductivity/condinv material
     rows used by the CPU update path, independently of medium conductivity. */
  bool conductivity = !ir->absorbers.empty(), chi2 = false, chi3 = false;
  for (const MaterialIRMaterial &material : ir->materials) {
    conductivity = conductivity || material.has_conductivity;
    chi2 = chi2 || material.has_chi2;
    chi3 = chi3 || material.has_chi3;
  }
  const auto add_topology = [&](const MaterialIRChunk &chunk, const StorageKey &key,
                                component c, size_t elements) {
    MaterialIRTopologyRow row;
    row.key = key; row.element_type = ElementType::realnum_value;
    row.logical_storage = native_precision; row.elements = elements;
    row.alignment = alignof(realnum); row.yee_component = int(c);
    const grid_volume &layout = s.chunks[chunk.chunk]->gv;
    const ivec shift = c == NO_COMPONENT ? ivec(D1) : layout.iyee_shift(c);
    for (int axis = 0; axis < 3; ++axis) {
      row.extents[axis] = c == NO_COMPONENT ? (axis == 0 ? int(elements) : 1)
                                             : chunk.extents[axis];
      row.strides[axis] = c == NO_COMPONENT ? (axis == 0 ? 1 : 0)
                                             : chunk.strides[axis];
      row.stagger[axis] = c == NO_COMPONENT ? 0 : shift.in_direction(
          s.chunks[chunk.chunk]->gv.yucky_direction(axis));
    }
    ir->topology.push_back(row);
  };
  for (const MaterialIRChunk &chunk : ir->chunks) {
    if (!chunk.owned) continue;
    const grid_volume &gv = s.chunks[chunk.chunk]->gv;
    FOR_COMPONENTS(c) if (gv.has_field(c)) {
      for (int d = 0; d < 5; ++d) {
        add_topology(chunk, {chunk.chunk, int(array_kind::chi1inv), int(c), -1, uint64_t(d)},
                     c, chunk.elements);
        if (conductivity) {
          add_topology(chunk, {chunk.chunk, int(array_kind::conductivity), int(c), -1,
                               uint64_t(d)}, c, chunk.elements);
          add_topology(chunk, {chunk.chunk, int(array_kind::condinv), int(c), -1, uint64_t(d)},
                       c, chunk.elements);
        }
      }
      if (chi2 || chi3) {
        add_topology(chunk, {chunk.chunk, int(array_kind::chi2), int(c), -1, 0}, c,
                     chunk.elements);
        add_topology(chunk, {chunk.chunk, int(array_kind::chi3), int(c), -1, 0}, c,
                     chunk.elements);
      }
      for (const MaterialIRSusceptibility &sus : ir->susceptibilities) {
        if (sus.field_type != type(c)) continue;
        for (int d = 0; d < 5; ++d)
          add_topology(chunk,
                       {chunk.chunk, int(array_kind::sigma), int(c), d,
                        uint64_t(sus.identity) * NUM_FIELD_TYPES + uint64_t(sus.field_type)},
                       c, chunk.elements);
      }
    }
    for (int d = 0; d < 6; ++d) if (chunk.pml_elements[d]) {
      add_topology(chunk, {chunk.chunk, int(array_kind::pml_sig), -1, -1, uint64_t(d)},
                   NO_COMPONENT, chunk.pml_elements[d]);
      add_topology(chunk, {chunk.chunk, int(array_kind::pml_kap), -1, -1, uint64_t(d)},
                   NO_COMPONENT, chunk.pml_elements[d]);
      add_topology(chunk, {chunk.chunk, int(array_kind::pml_siginv), -1, -1, uint64_t(d)},
                   NO_COMPONENT, chunk.pml_elements[d]);
    }
  }

  if (ir->topology.size() > std::numeric_limits<uint32_t>::max() ||
      ir->chunks.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("material IR destination identity overflow");
  std::map<int, uint32_t> chunk_indices;
  for (uint32_t i = 0; i < ir->chunks.size(); ++i) chunk_indices[ir->chunks[i].chunk] = i;
  for (uint32_t topology_index = 0; topology_index < ir->topology.size(); ++topology_index) {
    const MaterialIRTopologyRow &row = ir->topology[topology_index];
    const array_kind kind = static_cast<array_kind>(row.key.kind);
    if (kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
        kind == array_kind::pml_siginv)
      continue;
    const std::map<int, uint32_t>::const_iterator chunk_found = chunk_indices.find(row.key.chunk);
    if (chunk_found == chunk_indices.end())
      throw std::logic_error("material IR destination chunk is absent");
    if (ir->destinations.size() >= std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("material IR destination count overflow");
    const component c = component(row.key.component_);
    MaterialIRDestination destination;
    destination.key = row.key;
    destination.topology_index = topology_index;
    destination.chunk_index = chunk_found->second;
    destination.property = property_for(kind);
    destination.component = int(c);
    destination.tensor_direction = kind == array_kind::sigma
                                       ? row.key.cmp
                                       : (kind == array_kind::chi1inv ||
                                                  kind == array_kind::conductivity ||
                                                  kind == array_kind::condinv
                                              ? int(row.key.aux)
                                              : int(component_direction(c)));
    destination.tensor_column = kind == array_kind::chi1inv
                                    ? tensor_column_for(row, ir->dimensions)
                                    : component_index(c);
    destination.offdiagonal = kind == array_kind::chi1inv &&
                              destination.tensor_column >= 0 &&
                              destination.tensor_column != component_index(c);
    destination.point_count = ir->chunks[destination.chunk_index].loop_count[c];
    const uint32_t destination_index = uint32_t(ir->destinations.size());
    ir->destinations.push_back(destination);

    if (kind != array_kind::chi1inv || !ir->eps_averaging ||
        destination.tensor_column < 0 || ir->contains_host_callback) {
      if (destination.point_count)
        ir->bulk_spans.push_back(
            MaterialIRBulkSpan{destination_index, 0, destination.point_count});
      continue;
    }

    const grid_volume &gv = s.chunks[row.key.chunk]->gv;
    const direction dc = component_direction(c);
    const ivec shift1(unit_ivec(gv.dim, dc) * (type(c) == E_stuff ? 1 : -1));
    uint64_t point = 0;
    LOOP_OVER_VOL(gv, c, storage_index) {
      (void)storage_index;
      IVEC_LOOP_ILOC(gv, here);
      const meep::volume evaluation = gv.dV(destination.offdiagonal ? here - shift1 : here, 1.0);
      const FrontObjectResult front = classify_front_object(*ir, evaluation);
      bool bulk = false;
      MaterialIRPatchReason reason = MaterialIRPatchReason::unsupported_analytic_shape;
      if (!front.ambiguous &&
          material_records_equal(*ir, front.front, front.behind))
        bulk = true;
      else if (!front.ambiguous &&
               (ir_material_is_metal(ir->materials[front.front], type(c)) ||
                ir_material_is_metal(ir->materials[front.behind], type(c))))
        bulk = true;
      else if (front.ambiguous)
        reason = MaterialIRPatchReason::ambiguous_front;

      vector3 normal = {0, 0, 0};
      double fill = 0.0;
      bool analytic = false;
      uint32_t image_index = std::numeric_limits<uint32_t>::max();
      if (!bulk && !front.ambiguous &&
          front.object != std::numeric_limits<uint32_t>::max() &&
          ir->materials[front.front].kind == meep_geom::material_data::MEDIUM &&
          ir->materials[front.behind].kind == meep_geom::material_data::MEDIUM &&
          axis_aligned_block_face(*ir, front.object, front.image, evaluation, normal, fill)) {
        image_index = front.image;
        analytic = true;
      }
      if (bulk) append_bulk_point(*ir, destination_index, point);
      else if (analytic) {
        MaterialIRAnalyticInterface job = {};
        job.destination = destination_index;
        job.point = point;
        job.front_material = front.front;
        job.behind_material = front.behind;
        job.object = front.object;
        job.image = image_index;
        job.normal[0] = normal.x; job.normal[1] = normal.y; job.normal[2] = normal.z;
        job.fill = fill;
        ir->analytic_interfaces.push_back(job);
      }
      else {
        double values[3] = {0, 0, 0};
        bool adaptive_fallback = false, negative_fallback = false;
        geps.eff_chi1inv_row_with_outcome(c, values, evaluation, tol, maxeval,
                                          adaptive_fallback, negative_fallback);
        const realnum rounded = realnum(values[destination.tensor_column]);
        MaterialIRHybridPatch patch;
        patch.destination = destination_index;
        patch.point = point;
        patch.value = double(rounded);
        patch.front_material = front.front;
        patch.behind_material = front.behind;
        patch.object = front.object;
        patch.image = front.image;
        patch.ambiguous = front.ambiguous;
        patch.variable_material = front.variable_material;
        patch.variable_causes = front.variable_causes;
        patch.adaptive_fallback = adaptive_fallback;
        patch.negative_fallback = negative_fallback;
        patch.reason = reason;
        if (negative_fallback)
          patch.reason = MaterialIRPatchReason::negative_material_fallback;
        else if (front.variable_causes & material_variable_grid)
          patch.reason = MaterialIRPatchReason::material_grid_averaging;
        else if (adaptive_fallback)
          patch.reason = MaterialIRPatchReason::adaptive_overlap;
        ir->hybrid_patches.push_back(patch);
      }
      if (point == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("material IR destination point ordinal overflow");
      ++point;
    }
    if (point != destination.point_count)
      throw std::logic_error("material IR partition traversal count changed");
  }
  ir->requires_hybrid = !ir->hybrid_patches.empty();
  ir->signature = signature(*ir, false);
  ir->layout_signature = signature(*ir, true);
  validate_material_ir(*ir);
  {
    std::lock_guard<std::mutex> lock(material_ir_owners_mutex);
    material_ir_owners[ir.get()] = captured->owners;
  }
  return std::shared_ptr<const void>(captured, ir.get());
}

std::vector<std::shared_ptr<const OwnedMaterialCallback> >
material_ir_callback_owners(const MaterialIR &ir) {
  std::lock_guard<std::mutex> lock(material_ir_owners_mutex);
  const auto found = material_ir_owners.find(&ir);
  if (found == material_ir_owners.end()) return MaterialCallbackOwners();
  const std::shared_ptr<const MaterialCallbackOwners> owners = found->second.lock();
  return owners ? *owners : MaterialCallbackOwners();
}

const MaterialIR *material_ir_for(const fields &f) {
  return static_cast<const MaterialIR *>(f.material_ir.get());
}

void refresh_material_ir_signatures_for_testing(MaterialIR &ir) {
  ir.signature = signature(ir, false);
  ir.layout_signature = signature(ir, true);
}

void finalize_material_ir_collective(MaterialIR &ir) {
  ir.requires_hybrid = or_to_all(ir.requires_hybrid);
  ir.signature = signature(ir, false);
  ir.layout_signature = signature(ir, true);
  validate_material_ir(ir);
}

void set_material_ir_capture_failure_for_testing(int rank, int mode) {
  material_ir_capture_failure_rank = rank;
  material_ir_capture_failure_mode = mode;
}

int get_material_ir_capture_failure_rank_for_testing() {
  return material_ir_capture_failure_rank;
}

int get_material_ir_capture_failure_mode_for_testing() {
  return material_ir_capture_failure_mode;
}

void validate_material_ir(const MaterialIR &ir) {
  if (ir.version != material_ir_version || !std::isfinite(ir.subpixel_tol) || ir.subpixel_tol <= 0 ||
      (ir.eps_averaging ? ir.subpixel_maxeval <= 0 : ir.subpixel_maxeval != 0) ||
      ir.dimensions < D1 || ir.dimensions > Dcyl ||
      !std::isfinite(ir.projection_offset) ||
      ir.signature != signature(ir, false) ||
      ir.layout_signature != signature(ir, true))
    throw std::invalid_argument("material IR is malformed or stale");
  if (!ir.requires_hybrid && !ir.hybrid_patches.empty())
    throw std::invalid_argument("material IR global hybrid route omits local patches");
  if (ir.materials.empty() || ir.default_material >= ir.materials.size() || ir.cell.size() != 15)
    throw std::invalid_argument("material IR root metadata is invalid");
  const auto finite = [](const std::vector<double> &values) {
    for (double value : values) if (!std::isfinite(value)) return false;
    return true;
  };
  if (!finite(ir.cell)) throw std::invalid_argument("material IR cell is non-finite");
  for (int axis = 0; axis < 3; ++axis)
    if (!std::isfinite(ir.captured_volume[axis]) ||
        !std::isfinite(ir.captured_volume[axis + 3]) ||
        ir.captured_volume[axis] > ir.captured_volume[axis + 3])
      throw std::invalid_argument("material IR captured volume is invalid");
  if (ir.root_count > uint32_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("material IR root count is invalid");
  for (int i = 0; i < 3; ++i)
    if (!std::isfinite(ir.lattice_basis_size[i]) || !(ir.lattice_basis_size[i] > 0.0))
      throw std::invalid_argument("material IR lattice basis size is invalid");
  for (int i = 0; i < 9; ++i)
    if (!std::isfinite(ir.lattice_basis[i]) || !std::isfinite(ir.lattice_metric[i]) ||
        !std::isfinite(ir.lattice_inverse[i]) ||
        !std::isfinite(ir.lattice_inverse_transpose[i]))
      throw std::invalid_argument("material IR lattice transform is non-finite");
  matrix3x3 basis, metric, inverse, inverse_transpose;
  vector3 *columns[4][3] = {{&basis.c0, &basis.c1, &basis.c2},
                            {&metric.c0, &metric.c1, &metric.c2},
                            {&inverse.c0, &inverse.c1, &inverse.c2},
                            {&inverse_transpose.c0, &inverse_transpose.c1,
                             &inverse_transpose.c2}};
  const double *sources[4] = {ir.lattice_basis, ir.lattice_metric, ir.lattice_inverse,
                              ir.lattice_inverse_transpose};
  for (int matrix = 0; matrix < 4; ++matrix)
    for (int column = 0; column < 3; ++column) {
      columns[matrix][column]->x = sources[matrix][3 * column];
      columns[matrix][column]->y = sources[matrix][3 * column + 1];
      columns[matrix][column]->z = sources[matrix][3 * column + 2];
    }
  if (matrix3x3_determinant(basis) == 0.0 ||
      !matrix3x3_equal(matrix3x3_mult(matrix3x3_transpose(basis), basis), metric) ||
      !matrix3x3_equal(matrix3x3_inverse(basis), inverse) ||
      !matrix3x3_equal(matrix3x3_transpose(inverse), inverse_transpose))
    throw std::invalid_argument("material IR lattice transforms are inconsistent");
  bool contains_host_callback = false;
  for (const MaterialIRMaterial &m : ir.materials) {
    if (m.kind < meep_geom::material_data::MEDIUM ||
        m.kind > meep_geom::material_data::PERFECT_METAL ||
        (m.kind == meep_geom::material_data::MATERIAL_USER) != m.host_callback ||
        (m.owned_callback && (!m.callback_id || !m.callback_signature ||
                              !m.callback_capabilities)) ||
        (!m.owned_callback && (m.callback_id != 0 || m.callback_signature != 0 ||
                               m.callback_capabilities != 0)) ||
        (m.kind != meep_geom::material_data::MATERIAL_GRID &&
         (m.material_grid_kind != -1 || m.material_grid_trivial)) ||
        (m.kind != meep_geom::material_data::MATERIAL_GRID &&
         m.kind != meep_geom::material_data::MATERIAL_USER && m.do_averaging) ||
        (m.kind == meep_geom::material_data::MATERIAL_GRID &&
         (m.material_grid_kind < meep_geom::material_data::U_MIN ||
          m.material_grid_kind > meep_geom::material_data::U_DEFAULT)))
      throw std::invalid_argument("material IR material record is invalid");
    if (!finite(m.comparison_medium) || !finite(m.parameters) || !finite(m.samples))
      throw std::invalid_argument("material IR material numeric payload is non-finite");
    contains_host_callback = contains_host_callback || m.host_callback;
    if ((m.kind == meep_geom::material_data::PERFECT_METAL ||
         m.kind == meep_geom::material_data::MATERIAL_USER) &&
        (!m.parameters.empty() || !m.samples.empty()))
      throw std::invalid_argument("material IR tag has an unexpected payload");
    size_t payload_offset = 0;
    uint32_t expected_e = 0, expected_h = 0;
    if (m.kind == meep_geom::material_data::MEDIUM ||
        m.kind == meep_geom::material_data::MATERIAL_GRID) {
      size_t comparison_offset = 0;
      validate_medium_payload(m.comparison_medium, comparison_offset, NULL, NULL);
      if (comparison_offset != m.comparison_medium.size())
        throw std::invalid_argument("material IR comparison medium has trailing data");
      validate_medium_tensor_invertibility(m.comparison_medium, 0, true, true,
                                           m.kind == meep_geom::material_data::MATERIAL_GRID
                                               ? "MaterialGrid comparison"
                                               : "medium comparison");
    }
    else if (!m.comparison_medium.empty())
      throw std::invalid_argument("material IR variant has an unexpected comparison medium");
    if (m.kind == meep_geom::material_data::MEDIUM ||
        m.kind == meep_geom::material_data::MATERIAL_FILE) {
      validate_medium_payload(m.parameters, payload_offset, &expected_e, &expected_h);
      if (m.e_susceptibilities != expected_e || m.h_susceptibilities != expected_h)
        throw std::invalid_argument("material IR medium susceptibility count is inconsistent");
      /* FILE replaces the electric diagonal/offdiagonal at every sampled
         point, but retains and inverts the captured magnetic tensor. */
      validate_medium_tensor_invertibility(
          m.parameters, 0, m.kind == meep_geom::material_data::MEDIUM, true,
          m.kind == meep_geom::material_data::MATERIAL_FILE ? "FILE medium" : "medium");
    }
    if (m.kind == meep_geom::material_data::MEDIUM &&
        (payload_offset != m.parameters.size() || m.comparison_medium != m.parameters ||
         !m.samples.empty()))
      throw std::invalid_argument("material IR medium payload is invalid");
    if (m.kind == meep_geom::material_data::MATERIAL_FILE) {
      if (m.parameters.size() - payload_offset != 3)
        throw std::invalid_argument("material IR file schema has the wrong length");
      size_t product = 1;
      for (int d = 0; d < 3; ++d) {
        const size_t extent = checked_count(m.parameters[payload_offset + d], "file dimension");
        if (extent && product > std::numeric_limits<size_t>::max() / extent)
          throw std::overflow_error("material IR file product overflow");
        product *= extent;
      }
      if (product != m.samples.size())
        throw std::invalid_argument("material IR file sample count is invalid");
    }
    if (m.kind == meep_geom::material_data::MATERIAL_GRID) {
      if (m.parameters.size() < 3) throw std::invalid_argument("material IR grid schema is short");
      size_t product = 1;
      for (int d = 0; d < 3; ++d) {
        const double value = m.parameters[d];
        if (value < 0 || std::floor(value) != value ||
            value > double(std::numeric_limits<size_t>::max()))
          throw std::invalid_argument("material IR grid dimension is invalid");
        const size_t extent = size_t(value);
        if (extent && product > std::numeric_limits<size_t>::max() / extent)
          throw std::overflow_error("material IR grid product overflow");
        product *= extent;
      }
      payload_offset = 3;
      uint32_t e0 = 0, h0 = 0, e1 = 0, h1 = 0;
      const size_t first_medium_offset = payload_offset;
      validate_medium_payload(m.parameters, payload_offset, &e0, &h0);
      const size_t second_medium_offset = payload_offset;
      validate_medium_payload(m.parameters, payload_offset, &e1, &h1);
      validate_medium_tensor_invertibility(m.parameters, first_medium_offset, true, true,
                                           "MaterialGrid first endpoint");
      validate_medium_tensor_invertibility(m.parameters, second_medium_offset, true, true,
                                           "MaterialGrid second endpoint");
      if (m.parameters.size() - payload_offset != 3 ||
          m.e_susceptibilities != std::max(e0, e1) ||
          m.h_susceptibilities != std::max(h0, h1))
        throw std::invalid_argument("material IR grid medium schema is inconsistent");
      payload_offset += 3;
      if (payload_offset != m.parameters.size())
        throw std::invalid_argument("material IR grid schema has trailing data");
      if (product != m.samples.size())
        throw std::invalid_argument("material IR grid weight count is invalid");
    }
  }
  bool reachable_host_callback = ir.materials[ir.default_material].host_callback;
  for (const MaterialIRObject &object : ir.objects)
    reachable_host_callback = reachable_host_callback || ir.materials[object.material].host_callback;
  const bool expected_native = !reachable_host_callback;
  if (ir.contains_host_callback != reachable_host_callback ||
      ir.device_native_eligible != expected_native)
    throw std::invalid_argument("material IR callback eligibility is inconsistent");
  for (size_t oi = 0; oi < ir.objects.size(); ++oi) {
    const MaterialIRObject &o = ir.objects[oi];
    if (o.kind < geometric_object::MESH || o.kind > geometric_object::CYLINDER ||
        o.material < 0 || size_t(o.material) >= ir.materials.size() ||
        o.source_identity != oi || o.leaf_ordinal != oi || o.root_identity >= ir.root_count ||
        !finite(o.vertices) || !finite(o.indices) || !finite(o.auxiliary))
      throw std::invalid_argument("material IR object record is invalid");
    bool referenced_by_image = false;
    for (const MaterialIRGeometryImage &image : ir.images)
      referenced_by_image = referenced_by_image || image.object == oi;
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(o.parent_shift[axis]))
        throw std::invalid_argument("material IR object parent shift is invalid");
      if (referenced_by_image &&
          (std::isnan(o.low[axis]) || std::isnan(o.high[axis]) || o.low[axis] > o.high[axis]))
        throw std::invalid_argument("material IR referenced object bounds are invalid");
    }
    const size_t parameter_count = o.parameters.size();
    if ((o.kind == geometric_object::MESH &&
         (parameter_count != 4 || o.vertices.size() % 3 || o.indices.size() % 3 ||
          !o.auxiliary.empty())) ||
        (o.kind == geometric_object::PRISM &&
         (parameter_count != 29 || o.vertices.size() % 3 || !o.indices.empty() ||
          o.fixed_vertex_count < 3 ||
          o.fixed_vertex_count > std::numeric_limits<size_t>::max() / 9 ||
          o.auxiliary.size() != 9 * o.fixed_vertex_count)) ||
        (o.kind == geometric_object::SPHERE &&
         (parameter_count != 4 || !o.vertices.empty() || !o.indices.empty() ||
          !o.auxiliary.empty())) ||
        (o.kind == geometric_object::BLOCK &&
         ((parameter_count != 25 && parameter_count != 28) || !o.vertices.empty() ||
          !o.indices.empty() || !o.auxiliary.empty())) ||
        (o.kind == geometric_object::CYLINDER &&
         ((parameter_count != 9 && parameter_count != 10 && parameter_count != 19) ||
          !o.vertices.empty() || !o.indices.empty() || !o.auxiliary.empty())))
      throw std::invalid_argument("material IR object shape payload is invalid");
    for (size_t pi = 0; pi < o.parameters.size(); ++pi) {
      const bool infinite_extent =
          (o.kind == geometric_object::BLOCK && pi >= 12 && pi < 15) ||
          (o.kind == geometric_object::PRISM && pi == 3) ||
          (o.kind == geometric_object::CYLINDER && pi == 7);
      if (std::isnan(o.parameters[pi]) ||
          (!std::isfinite(o.parameters[pi]) &&
           (!infinite_extent || o.parameters[pi] < 0.0)))
        throw std::invalid_argument("material IR object parameter is non-finite");
    }
    if (o.kind == geometric_object::MESH && o.parameters[3] != 0.0 && o.parameters[3] != 1.0)
      throw std::invalid_argument("material IR mesh closed flag is invalid");
    if (o.kind == geometric_object::PRISM) {
      const double *top = o.auxiliary.data() + 6 * o.fixed_vertex_count;
      for (size_t vertex = 0; vertex < o.fixed_vertex_count; ++vertex)
        if (!equal_double(top[3 * vertex + 2], o.parameters[3]))
          throw std::invalid_argument("material IR prism fixed-height state is stale");
    }
    if (o.kind == geometric_object::BLOCK) {
      const double subtype = o.parameters[24];
      if (std::floor(subtype) != subtype || subtype < double(std::numeric_limits<int>::min()) ||
          subtype > double(std::numeric_limits<int>::max()) ||
          (parameter_count == 25 && int(subtype) != block::BLOCK_SELF) ||
          (parameter_count == 28 && int(subtype) != block::ELLIPSOID))
        throw std::invalid_argument("material IR block subtype is invalid");
    }
    if (o.kind == geometric_object::CYLINDER) {
      const double subtype = o.parameters[8];
      if (std::floor(subtype) != subtype || subtype < double(std::numeric_limits<int>::min()) ||
          subtype > double(std::numeric_limits<int>::max()) ||
          (parameter_count == 9 && int(subtype) != cylinder::CYLINDER_SELF) ||
          (parameter_count == 10 && int(subtype) != cylinder::CONE) ||
          (parameter_count == 19 && int(subtype) != cylinder::WEDGE))
        throw std::invalid_argument("material IR cylinder subtype is invalid");
    }
    if (o.kind == geometric_object::MESH) {
      if (o.fixed_vertex_count)
        throw std::invalid_argument("material IR mesh has prism fixed vertices");
      const size_t vertices = o.vertices.size() / 3;
      if (vertices < 4 || o.indices.size() < 12)
        throw std::invalid_argument("material IR mesh cardinality is invalid");
      for (double index : o.indices)
        if (index < 0 || std::floor(index) != index || index >= double(vertices))
          throw std::invalid_argument("material IR mesh index is invalid");
      if (o.vertex_offset > ir.geometry_vertices.size() / 3 ||
          o.vertex_count > ir.geometry_vertices.size() / 3 - o.vertex_offset ||
          o.vertex_count != vertices || o.triangle_offset > ir.geometry_triangles.size() ||
          o.triangle_count > ir.geometry_triangles.size() - o.triangle_offset ||
          o.triangle_count != o.indices.size() / 3 ||
          o.bvh_offset > ir.geometry_bvh.size() ||
          o.bvh_count > ir.geometry_bvh.size() - o.bvh_offset ||
          !std::isfinite(o.mesh_lengthscale) || !(o.mesh_lengthscale > 0.0))
        throw std::invalid_argument("material IR mesh compact spans are invalid");
      for (size_t vertex = 0; vertex < vertices; ++vertex)
        for (int axis = 0; axis < 3; ++axis)
          if (ir.geometry_vertices[3 * (o.vertex_offset + vertex) + axis] !=
              o.vertices[3 * vertex + axis])
            throw std::invalid_argument("material IR mesh vertex span is stale");
      vector3 vertex_low = array_vector(o.vertices.data()), vertex_high = vertex_low;
      for (size_t vertex = 1; vertex < vertices; ++vertex) {
        const vector3 value = array_vector(o.vertices.data() + 3 * vertex);
        vertex_low.x = std::min(vertex_low.x, value.x);
        vertex_low.y = std::min(vertex_low.y, value.y);
        vertex_low.z = std::min(vertex_low.z, value.z);
        vertex_high.x = std::max(vertex_high.x, value.x);
        vertex_high.y = std::max(vertex_high.y, value.y);
        vertex_high.z = std::max(vertex_high.z, value.z);
      }
      const double dx = vertex_high.x - vertex_low.x, dy = vertex_high.y - vertex_low.y,
                   dz = vertex_high.z - vertex_low.z;
      double expected_lengthscale = sqrt(dx * dx + dy * dy + dz * dz);
      if (expected_lengthscale == 0.0) expected_lengthscale = 1.0;
      if (o.mesh_lengthscale != expected_lengthscale)
        throw std::invalid_argument("material IR mesh lengthscale is stale");
      const double area_epsilon = 1e-20 * o.mesh_lengthscale * o.mesh_lengthscale;
      for (size_t triangle_index = 0; triangle_index < o.triangle_count; ++triangle_index) {
        const MaterialIRTriangle &triangle =
            ir.geometry_triangles[size_t(o.triangle_offset + triangle_index)];
        const uint32_t local[3] = {uint32_t(o.indices[3 * triangle_index]),
                                   uint32_t(o.indices[3 * triangle_index + 1]),
                                   uint32_t(o.indices[3 * triangle_index + 2])};
        const vector3 a = array_vector(o.vertices.data() + 3 * local[0]);
        const vector3 b = array_vector(o.vertices.data() + 3 * local[1]);
        const vector3 c = array_vector(o.vertices.data() + 3 * local[2]);
        vector3 normal = vector3_cross(vector3_minus(b, a), vector3_minus(c, a));
        const double length = vector3_norm(normal);
        if (length > area_epsilon) normal = vector3_scale(1.0 / length, normal);
        const double expected_normal[3] = {normal.x, normal.y, normal.z};
        for (int axis = 0; axis < 3; ++axis) {
          const double av = axis == 0 ? a.x : axis == 1 ? a.y : a.z;
          const double bv = axis == 0 ? b.x : axis == 1 ? b.y : b.z;
          const double cv = axis == 0 ? c.x : axis == 1 ? c.y : c.z;
          if (triangle.vertex[axis] != o.vertex_offset + local[axis] ||
              triangle.normal[axis] != expected_normal[axis] ||
              triangle.low[axis] != std::min(av, std::min(bv, cv)) ||
              triangle.high[axis] != std::max(av, std::max(bv, cv)))
            throw std::invalid_argument("material IR mesh triangle span is stale");
        }
      }
      std::set<uint32_t> ordered_faces;
      for (uint64_t node_index = o.bvh_offset; node_index < o.bvh_offset + o.bvh_count;
           ++node_index) {
        const MaterialIRBvhNode &node = ir.geometry_bvh[size_t(node_index)];
        if (node.leaf) {
          if (node.left != std::numeric_limits<uint32_t>::max() ||
              node.right != std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("material IR mesh BVH leaf has children");
          for (uint64_t face = node.first_triangle;
               face < node.first_triangle + node.triangle_count; ++face)
            if (face >= ir.geometry_bvh_face_ids.size() ||
                ir.geometry_bvh_face_ids[size_t(face)] < o.triangle_offset ||
                ir.geometry_bvh_face_ids[size_t(face)] >= o.triangle_offset + o.triangle_count)
              throw std::invalid_argument("material IR mesh BVH face span is invalid");
            else if (!ordered_faces.insert(ir.geometry_bvh_face_ids[size_t(face)]).second)
              throw std::invalid_argument("material IR mesh BVH face identity is duplicated");
        }
        else if (node.left < o.bvh_offset || node.left >= o.bvh_offset + o.bvh_count ||
                 node.right < o.bvh_offset || node.right >= o.bvh_offset + o.bvh_count ||
                 node.first_triangle != 0 || node.triangle_count != 0)
          throw std::invalid_argument("material IR mesh BVH internal node is invalid");
      }
      if (ordered_faces.size() != o.triangle_count)
        throw std::invalid_argument("material IR mesh BVH face coverage is incomplete");
    }
    else if (o.kind != geometric_object::PRISM && o.fixed_vertex_count)
      throw std::invalid_argument("material IR non-prism has fixed prism vertices");
    else if (o.vertex_offset || o.vertex_count || o.triangle_offset || o.triangle_count ||
             o.bvh_offset || o.bvh_count || o.mesh_lengthscale != 0.0)
      throw std::invalid_argument("non-mesh material IR object has mesh spans");
    const geom_box rebuilt = object_bounds_from_ir(ir, o);
    const double rebuilt_low[3] = {rebuilt.low.x, rebuilt.low.y, rebuilt.low.z};
    const double rebuilt_high[3] = {rebuilt.high.x, rebuilt.high.y, rebuilt.high.z};
    for (int axis = 0; axis < 3; ++axis) {
      if (std::isnan(o.low[axis]) || std::isnan(o.high[axis])) continue;
      const bool finite_axis = std::isfinite(o.low[axis]) && std::isfinite(o.high[axis]);
      if ((finite_axis && (!std::isfinite(rebuilt_low[axis]) ||
                           !std::isfinite(rebuilt_high[axis]) ||
                           o.low[axis] != rebuilt_low[axis] ||
                           o.high[axis] != rebuilt_high[axis])) ||
          (!finite_axis && !std::isnan(rebuilt_low[axis]) &&
           !std::isnan(rebuilt_high[axis]) &&
           (o.low[axis] != rebuilt_low[axis] || o.high[axis] != rebuilt_high[axis])))
        throw std::invalid_argument("material IR object bounds are stale for kind " +
                                    std::to_string(o.kind) + " axis " +
                                    std::to_string(axis) + " (stored " +
                                    std::to_string(o.low[axis]) + "," +
                                    std::to_string(o.high[axis]) + "; rebuilt " +
                                    std::to_string(rebuilt_low[axis]) + "," +
                                    std::to_string(rebuilt_high[axis]) + ")");
    }
  }
  if (ir.geometry_vertices.size() % 3 || !finite(ir.geometry_vertices))
    throw std::invalid_argument("material IR geometry vertex table is invalid");
  const size_t geometry_vertex_count = ir.geometry_vertices.size() / 3;
  for (const MaterialIRTriangle &triangle : ir.geometry_triangles)
    for (int axis = 0; axis < 3; ++axis) {
      if (triangle.vertex[axis] >= geometry_vertex_count ||
          !std::isfinite(triangle.normal[axis]) || std::isnan(triangle.low[axis]) ||
          std::isnan(triangle.high[axis]) || triangle.low[axis] > triangle.high[axis])
        throw std::invalid_argument("material IR triangle table is invalid");
    }
  for (const MaterialIRBvhNode &node : ir.geometry_bvh) {
    for (int axis = 0; axis < 3; ++axis)
      if (std::isnan(node.low[axis]) || std::isnan(node.high[axis]) ||
          node.low[axis] > node.high[axis])
        throw std::invalid_argument("material IR BVH bounds are invalid");
    if ((node.leaf &&
         (node.left != std::numeric_limits<uint32_t>::max() ||
          node.right != std::numeric_limits<uint32_t>::max() || !node.triangle_count)) ||
        (!node.leaf &&
         (node.left == std::numeric_limits<uint32_t>::max() ||
          node.right == std::numeric_limits<uint32_t>::max() || node.first_triangle != 0 ||
          node.triangle_count != 0)) ||
        node.first_triangle > ir.geometry_bvh_face_ids.size() ||
        node.triangle_count > ir.geometry_bvh_face_ids.size() - node.first_triangle)
      throw std::invalid_argument("material IR BVH node is invalid");
  }
  for (uint32_t triangle : ir.geometry_bvh_face_ids)
    if (triangle >= ir.geometry_triangles.size())
      throw std::invalid_argument("material IR BVH face identity is invalid");
  const geom_box canonical_bounds = {{ir.captured_volume[0], ir.captured_volume[1],
                                      ir.captured_volume[2]},
                                     {ir.captured_volume[3], ir.captured_volume[4],
                                      ir.captured_volume[5]}};
  if (ir.images.size() > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("material IR geometry image count is invalid");
  size_t expected_image = 0;
  for (int root = int(ir.root_count) - 1; root >= 0; --root) {
    if (expected_image > ir.images.size())
      throw std::invalid_argument("material IR geometry image count is invalid");
    const int precedence = int(ir.images.size() - expected_image);
    for_periodic_images(ir, [&](const int periodic[3], vector3 periodic_shift) {
      size_t stored = 0;
      for (uint32_t object_index = 0; object_index < ir.objects.size(); ++object_index) {
        const MaterialIRObject &object = ir.objects[object_index];
        if (object.root_identity != uint32_t(root)) continue;
        geom_box box = {{object.low[0], object.low[1], object.low[2]},
                        {object.high[0], object.high[1], object.high[2]}};
        const vector3 shift = {object.parent_shift[0] + periodic_shift.x,
                               object.parent_shift[1] + periodic_shift.y,
                               object.parent_shift[2] + periodic_shift.z};
        shift_box(box, shift);
        if (!boxes_intersect(box, canonical_bounds)) continue;
        if (expected_image >= ir.images.size())
          throw std::invalid_argument("material IR geometry image list is short");
        const MaterialIRGeometryImage &image = ir.images[expected_image];
        if (image.object != object_index || image.ordinal != expected_image ||
            stored > size_t(std::numeric_limits<int>::max()) ||
            image.precedence != precedence - int(stored))
          throw std::invalid_argument("material IR geometry image order is stale");
        for (int axis = 0; axis < 3; ++axis) {
          const double expected_shift = axis == 0 ? shift.x : axis == 1 ? shift.y : shift.z;
          const double expected_low = axis == 0 ? box.low.x : axis == 1 ? box.low.y : box.low.z;
          const double expected_high =
              axis == 0 ? box.high.x : axis == 1 ? box.high.y : box.high.z;
          if (image.image[axis] != periodic[axis] || image.shift[axis] != expected_shift ||
              image.low[axis] != expected_low || image.high[axis] != expected_high)
            throw std::invalid_argument("material IR geometry image payload is stale");
        }
        ++expected_image;
        ++stored;
      }
    });
  }
  if (expected_image != ir.images.size())
    throw std::invalid_argument("material IR geometry image list has trailing records");
  uint32_t previous_active = 0;
  bool have_active = false;
  for (uint32_t active : ir.active_images) {
    if (active >= ir.images.size() || (have_active && active <= previous_active))
      throw std::invalid_argument("material IR active image list is invalid");
    previous_active = active;
    have_active = true;
  }
  geom_box expected_local_bounds = {};
  bool have_expected_local_bounds = false;
  for (const MaterialIRChunk &chunk : ir.chunks) {
    if (!chunk.owned) continue;
    geom_box box = {};
    for (int axis = 0; axis < 3; ++axis) {
      const direction d = ir_yucky_direction(ndim(chunk.dimensions), axis);
      if (!has_direction(ndim(chunk.dimensions), d)) continue;
      const double low = 0.5 * chunk.inva * (int64_t(chunk.little_corner[axis]) - 2);
      const double high = 0.5 * chunk.inva * (int64_t(chunk.big_corner[axis]) + 2);
      double *box_low = d == X || d == R ? &box.low.x : d == Y || d == P ? &box.low.y
                                                                         : &box.low.z;
      double *box_high = d == X || d == R ? &box.high.x : d == Y || d == P ? &box.high.y
                                                                            : &box.high.z;
      *box_low = low;
      *box_high = high;
    }
    if (!have_expected_local_bounds) {
      expected_local_bounds = box;
      have_expected_local_bounds = true;
    }
    else {
      expected_local_bounds.low.x = std::min(expected_local_bounds.low.x, box.low.x);
      expected_local_bounds.low.y = std::min(expected_local_bounds.low.y, box.low.y);
      expected_local_bounds.low.z = std::min(expected_local_bounds.low.z, box.low.z);
      expected_local_bounds.high.x = std::max(expected_local_bounds.high.x, box.high.x);
      expected_local_bounds.high.y = std::max(expected_local_bounds.high.y, box.high.y);
      expected_local_bounds.high.z = std::max(expected_local_bounds.high.z, box.high.z);
    }
  }
  std::vector<uint32_t> expected_active_images;
  if (have_expected_local_bounds)
    for (uint32_t image = 0; image < ir.images.size(); ++image) {
      geom_box box;
      box.low = {ir.images[image].low[0], ir.images[image].low[1], ir.images[image].low[2]};
      box.high = {ir.images[image].high[0], ir.images[image].high[1],
                  ir.images[image].high[2]};
      if (boxes_intersect(box, expected_local_bounds)) expected_active_images.push_back(image);
    }
  if (ir.active_images != expected_active_images)
    throw std::invalid_argument("material IR active image partition is stale");
  for (uint32_t material : ir.extra_materials)
    if (material >= ir.materials.size())
      throw std::invalid_argument("material IR extra-material reference is invalid");
  uint32_t next_identity[2] = {0, 0};
  for (const MaterialIRSusceptibility &sus : ir.susceptibilities) {
    const int slot = sus.field_type == E_stuff ? 0 : sus.field_type == H_stuff ? 1 : -1;
    if (slot < 0 || sus.material != std::numeric_limits<uint32_t>::max() ||
        sus.identity != next_identity[slot] || sus.material_ordinal != next_identity[slot]++ ||
        !finite(sus.parameters))
      throw std::invalid_argument("material IR susceptibility identity is invalid");
    size_t offset = 0;
    validate_susceptibility_payload(sus.parameters, offset);
    if (offset != sus.parameters.size())
      throw std::invalid_argument("material IR susceptibility schema has trailing data");
  }
  std::set<int> chunks;
  for (const MaterialIRChunk &chunk : ir.chunks) {
    if (chunk.chunk < 0 || !chunks.insert(chunk.chunk).second ||
        chunk.dimensions < D1 || chunk.dimensions > Dcyl ||
        chunk.dimensions != ir.dimensions ||
        !std::isfinite(chunk.resolution) || !(chunk.resolution > 0) ||
        !std::isfinite(chunk.inva) || !(chunk.inva > 0) ||
        chunk.inva != 1.0 / chunk.resolution || !chunk.elements ||
        chunk.component_bits >> NUM_FIELD_COMPONENTS)
      throw std::invalid_argument("material IR chunk record is invalid");
    for (int axis = 0; axis < 3; ++axis) {
      const int64_t doubled_extent = int64_t(chunk.big_corner[axis]) -
                                     int64_t(chunk.little_corner[axis]);
      if (chunk.extents[axis] <= 0 || chunk.strides[axis] < 0 ||
          doubled_extent < 0 || doubled_extent % 2 ||
          doubled_extent / 2 >= std::numeric_limits<int>::max() ||
          chunk.extents[axis] != doubled_extent / 2 + 1 ||
          !std::isfinite(chunk.origin[axis]))
        throw std::invalid_argument("material IR chunk layout is invalid");
    }
    FOR_COMPONENTS(c) {
      const bool present = (chunk.component_bits & (uint64_t(1) << int(c))) != 0;
      size_t count = present ? 1 : 0;
      ptrdiff_t maximum = 0;
      for (int axis = 0; axis < 3; ++axis) {
        if (chunk.stagger[c][axis] != 0 && chunk.stagger[c][axis] != 1)
          throw std::invalid_argument("material IR chunk stagger is invalid");
        if (!present) {
          if (chunk.loop_begin[c][axis] || chunk.loop_end[c][axis])
            throw std::invalid_argument("absent material IR component has loop bounds");
          continue;
        }
        const int64_t expected_begin =
            int64_t(chunk.little_corner[axis]) + chunk.stagger[c][axis];
        const int64_t expected_end =
            int64_t(chunk.big_corner[axis]) + chunk.stagger[c][axis];
        if (expected_begin < std::numeric_limits<int>::min() ||
            expected_begin > std::numeric_limits<int>::max() ||
            expected_end < std::numeric_limits<int>::min() ||
            expected_end > std::numeric_limits<int>::max() ||
            chunk.loop_begin[c][axis] != expected_begin ||
            chunk.loop_end[c][axis] != expected_end)
          throw std::invalid_argument("material IR component loop bounds are inconsistent");
        const int64_t delta = int64_t(chunk.loop_end[c][axis]) -
                              int64_t(chunk.loop_begin[c][axis]);
        if (delta < 0 || delta % 2)
          throw std::invalid_argument("material IR component loop bounds are invalid");
        const size_t axis_count = size_t(delta / 2) + 1;
        if (axis_count && count > std::numeric_limits<size_t>::max() / axis_count)
          throw std::overflow_error("material IR component loop count overflow");
        count *= axis_count;
        const int64_t coordinate64 =
            (int64_t(chunk.loop_end[c][axis]) - chunk.little_corner[axis]) / 2;
        if (coordinate64 < 0 || uint64_t(coordinate64) > uint64_t(PTRDIFF_MAX))
          throw std::overflow_error("material IR component address range overflow");
        const ptrdiff_t coordinate = ptrdiff_t(coordinate64);
        if ((coordinate && chunk.strides[axis] >
                               std::numeric_limits<ptrdiff_t>::max() / coordinate) ||
            maximum > std::numeric_limits<ptrdiff_t>::max() - coordinate * chunk.strides[axis])
          throw std::overflow_error("material IR component address range overflow");
        maximum += coordinate * chunk.strides[axis];
      }
      if (chunk.loop_count[c] != count ||
          (present && (maximum < 0 || size_t(maximum) >= chunk.elements)))
        throw std::invalid_argument("material IR component loop exceeds chunk storage");
    }
  }
  for (const MaterialIRPml &p : ir.absorbers)
    if (p.direction < 0 || p.direction >= 5 || p.side < 0 || p.side >= 2 ||
        !std::isfinite(p.thickness) || !(p.thickness > 0) ||
        !std::isfinite(p.r_asymptotic) || p.r_asymptotic <= 0 || p.r_asymptotic >= 1 ||
        !std::isfinite(p.mean_stretch) || p.mean_stretch < 1 ||
        !std::isfinite(p.sample_spacing) || p.sample_spacing < 0 || p.samples.empty() ||
        !finite(p.samples))
      throw std::invalid_argument("material IR PML record is invalid");
  for (const MaterialIRPmlAxis &p : ir.pml_axes)
    if (p.chunk < 0 || !chunks.count(p.chunk) || p.direction < 0 || p.direction >= 6 ||
        !p.elements || p.elements > size_t(std::numeric_limits<int>::max()) ||
        !std::isfinite(p.resolution) || !(p.resolution > 0) ||
        (p.profile_active &&
         (!std::isfinite(p.thickness) || !(p.thickness > 0) ||
          !std::isfinite(p.boundary_location) || !std::isfinite(p.r_asymptotic) ||
          !(p.r_asymptotic > 0) || !(p.r_asymptotic < 1) ||
          !std::isfinite(p.mean_stretch) || p.mean_stretch < 1 ||
          !std::isfinite(p.profile_integral) || !(p.profile_integral > 0) ||
          !std::isfinite(p.profile_integral_u) || !(p.profile_integral_u > 0) ||
          p.profile_samples.size() != p.elements || !finite(p.profile_samples))) ||
        (!p.profile_active &&
         (p.analytic_quadratic || p.thickness != 0 || p.boundary_location != 0 ||
          p.r_asymptotic != 0 || p.mean_stretch != 1 || p.profile_integral != 0 ||
          p.profile_integral_u != 0 || !p.profile_samples.empty())) ||
        p.sigma.size() != p.elements || p.kappa.size() != p.elements ||
        p.sigma_inv.size() != p.elements || !finite(p.sigma) || !finite(p.kappa) ||
        !finite(p.sigma_inv))
      throw std::invalid_argument("material IR PML axis is invalid");
  std::set<StorageKey, StorageKeyLess> topology_keys;
  std::set<StorageKey, StorageKeyLess> expected_keys;
  std::map<int, const MaterialIRChunk *> chunk_by_id;
  for (const MaterialIRChunk &chunk : ir.chunks) chunk_by_id[chunk.chunk] = &chunk;
  std::set<std::pair<int, int> > pml_axes;
  for (const MaterialIRPmlAxis &axis : ir.pml_axes) {
    if (!pml_axes.insert(std::make_pair(axis.chunk, axis.direction)).second)
      throw std::invalid_argument("material IR PML axis is duplicated");
    const MaterialIRChunk &chunk = *chunk_by_id[axis.chunk];
    if (chunk.pml_elements[axis.direction] != axis.elements)
      throw std::invalid_argument("material IR PML axis extent differs from its chunk");
  }
  size_t pml_topology_rows = 0;
  for (const MaterialIRTopologyRow &row : ir.topology) {
    if (row.key.chunk < 0 || !chunks.count(row.key.chunk) ||
        row.element_type != ElementType::realnum_value || row.logical_storage != native_precision ||
        !row.elements || row.alignment != alignof(realnum) ||
        !topology_keys.insert(row.key).second)
      throw std::invalid_argument("material IR topology row is invalid");
    if (row.elements > std::numeric_limits<size_t>::max() / sizeof(realnum))
      throw std::overflow_error("material IR topology byte count overflow");
    const array_kind kind = static_cast<array_kind>(row.key.kind);
    const bool pml = kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
                     kind == array_kind::pml_siginv;
    const bool diagonal = kind == array_kind::chi2 || kind == array_kind::chi3;
    const bool tensor = kind == array_kind::chi1inv || kind == array_kind::conductivity ||
                        kind == array_kind::condinv;
    if ((!pml && !diagonal && !tensor && kind != array_kind::sigma) ||
        (pml && (row.key.component_ != -1 || row.key.cmp != -1 || row.key.aux >= 6)) ||
        (diagonal && (row.key.component_ < 0 || row.key.component_ >= NUM_FIELD_COMPONENTS ||
                      row.key.cmp != -1 || row.key.aux != 0)) ||
        (tensor && (row.key.component_ < 0 || row.key.component_ >= NUM_FIELD_COMPONENTS ||
                    row.key.cmp != -1 || row.key.aux >= 5)) ||
        (kind == array_kind::sigma &&
         (row.key.component_ < 0 || row.key.component_ >= NUM_FIELD_COMPONENTS ||
          row.key.cmp < 0 || row.key.cmp >= 5)))
      throw std::invalid_argument("material IR topology key schema is invalid");
    const MaterialIRChunk &chunk = *chunk_by_id[row.key.chunk];
    if (pml) {
      ++pml_topology_rows;
      if (!pml_axes.count(std::make_pair(row.key.chunk, int(row.key.aux))) ||
          row.yee_component != NO_COMPONENT || row.extents[0] != int(row.elements) ||
          row.extents[1] != 1 || row.extents[2] != 1 || row.strides[0] != 1 ||
          row.strides[1] != 0 || row.strides[2] != 0)
        throw std::invalid_argument("material IR PML topology does not match its axis");
    }
    else {
      if (row.yee_component != row.key.component_ || row.elements != chunk.elements)
        throw std::invalid_argument("material IR Yee topology target is invalid");
      for (int axis = 0; axis < 3; ++axis)
        if (row.extents[axis] != chunk.extents[axis] ||
            row.strides[axis] != chunk.strides[axis] ||
            row.stagger[axis] != chunk.stagger[row.key.component_][axis])
          throw std::invalid_argument("material IR Yee topology shape is invalid");
    }
    if (kind == array_kind::sigma) {
      const field_type ft = field_type(row.key.aux % uint64_t(NUM_FIELD_TYPES));
      const uint64_t state = row.key.aux / uint64_t(NUM_FIELD_TYPES);
      bool found = false;
      for (const MaterialIRSusceptibility &sus : ir.susceptibilities)
        found = found || (sus.field_type == ft && sus.identity == state);
      if (!found || (ft == E_stuff ? !is_electric(component(row.key.component_))
                                   : !is_magnetic(component(row.key.component_))))
        throw std::invalid_argument("material IR sigma topology identity is invalid");
    }
    for (int axis = 0; axis < 3; ++axis)
      if (row.extents[axis] <= 0 || row.strides[axis] < 0 ||
          (row.stagger[axis] != 0 && row.stagger[axis] != 1))
        throw std::invalid_argument("material IR topology layout is invalid");
  }
  if (ir.pml_axes.size() > std::numeric_limits<size_t>::max() / 3 ||
      pml_topology_rows != ir.pml_axes.size() * 3)
    throw std::invalid_argument("material IR PML topology coverage is incomplete");
  bool conductivity = !ir.absorbers.empty(), chi2 = false, chi3 = false;
  for (const MaterialIRMaterial &material : ir.materials) {
    conductivity = conductivity || material.has_conductivity;
    chi2 = chi2 || material.has_chi2;
    chi3 = chi3 || material.has_chi3;
  }
  for (const MaterialIRChunk &chunk : ir.chunks) {
    if (!chunk.owned) continue;
    FOR_COMPONENTS(c) if (chunk.component_bits & (uint64_t(1) << int(c))) {
      for (int d = 0; d < 5; ++d) {
        expected_keys.insert({chunk.chunk, int(array_kind::chi1inv), int(c), -1, uint64_t(d)});
        if (conductivity) {
          expected_keys.insert(
              {chunk.chunk, int(array_kind::conductivity), int(c), -1, uint64_t(d)});
          expected_keys.insert({chunk.chunk, int(array_kind::condinv), int(c), -1, uint64_t(d)});
        }
      }
      if (chi2 || chi3) {
        expected_keys.insert({chunk.chunk, int(array_kind::chi2), int(c), -1, 0});
        expected_keys.insert({chunk.chunk, int(array_kind::chi3), int(c), -1, 0});
      }
      for (const MaterialIRSusceptibility &sus : ir.susceptibilities) {
        if (sus.field_type != type(c)) continue;
        for (int d = 0; d < 5; ++d)
          expected_keys.insert(
              {chunk.chunk, int(array_kind::sigma), int(c), d,
               uint64_t(sus.identity) * NUM_FIELD_TYPES + uint64_t(sus.field_type)});
      }
    }
    for (int d = 0; d < 6; ++d) if (chunk.pml_elements[d]) {
      expected_keys.insert({chunk.chunk, int(array_kind::pml_sig), -1, -1, uint64_t(d)});
      expected_keys.insert({chunk.chunk, int(array_kind::pml_kap), -1, -1, uint64_t(d)});
      expected_keys.insert({chunk.chunk, int(array_kind::pml_siginv), -1, -1, uint64_t(d)});
    }
  }
  if (topology_keys != expected_keys)
    throw std::invalid_argument("material IR topology is not the exact conservative superset");

  size_t expected_destinations = 0;
  for (const MaterialIRTopologyRow &row : ir.topology) {
    const array_kind kind = static_cast<array_kind>(row.key.kind);
    if (kind != array_kind::pml_sig && kind != array_kind::pml_kap &&
        kind != array_kind::pml_siginv)
      ++expected_destinations;
  }
  if (ir.destinations.size() != expected_destinations)
    throw std::invalid_argument("material IR destination coverage is incomplete");
  size_t expected_destination_index = 0;
  for (size_t topology_index = 0; topology_index < ir.topology.size(); ++topology_index) {
    const array_kind kind = static_cast<array_kind>(ir.topology[topology_index].key.kind);
    if (kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
        kind == array_kind::pml_siginv)
      continue;
    if (expected_destination_index >= ir.destinations.size() ||
        ir.destinations[expected_destination_index].topology_index != topology_index ||
        !(ir.destinations[expected_destination_index].key == ir.topology[topology_index].key) ||
        ir.destinations[expected_destination_index].property != property_for(kind))
      throw std::invalid_argument("material IR destination order differs from topology");
    ++expected_destination_index;
  }
  for (size_t i = 0; i < ir.destinations.size(); ++i) {
    const MaterialIRDestination &destination = ir.destinations[i];
    if (destination.topology_index >= ir.topology.size())
      throw std::invalid_argument("material IR destination topology index is invalid");
    const MaterialIRTopologyRow &row = ir.topology[destination.topology_index];
    const array_kind kind = static_cast<array_kind>(row.key.kind);
    const component c = component(row.key.component_);
    const int expected_direction = kind == array_kind::sigma
                                       ? row.key.cmp
                                       : (kind == array_kind::chi1inv ||
                                                  kind == array_kind::conductivity ||
                                                  kind == array_kind::condinv
                                              ? int(row.key.aux)
                                              : int(component_direction(c)));
    const int expected_column = kind == array_kind::chi1inv
                                    ? tensor_column_for(row, ir.dimensions)
                                    : component_index(c);
    const bool expected_offdiagonal = kind == array_kind::chi1inv && expected_column >= 0 &&
                                      expected_column != component_index(c);
    if (destination.topology_index >= ir.topology.size() ||
        destination.chunk_index >= ir.chunks.size() ||
        !(destination.key == row.key) ||
        destination.key.chunk != ir.chunks[destination.chunk_index].chunk ||
        destination.component != destination.key.component_ ||
        destination.component < 0 || destination.component >= NUM_FIELD_COMPONENTS ||
        destination.property != property_for(kind) ||
        destination.tensor_direction != expected_direction ||
        destination.tensor_column != expected_column ||
        destination.offdiagonal != expected_offdiagonal ||
        destination.point_count !=
            ir.chunks[destination.chunk_index].loop_count[destination.component] ||
        (destination.offdiagonal && destination.property != MaterialIRProperty::chi1inv))
      throw std::invalid_argument("material IR destination record is invalid");
  }
  uint32_t previous_destination = 0;
  uint64_t previous_end = 0;
  bool have_span = false;
  for (const MaterialIRBulkSpan &span : ir.bulk_spans) {
    if (span.destination >= ir.destinations.size() || !span.count ||
        span.first_point > ir.destinations[span.destination].point_count ||
        span.count > ir.destinations[span.destination].point_count - span.first_point ||
        (have_span && (span.destination < previous_destination ||
                       (span.destination == previous_destination &&
                        span.first_point <= previous_end))))
      throw std::invalid_argument("material IR bulk span is invalid");
    const MaterialIRDestination &destination = ir.destinations[span.destination];
    if (destination.property == MaterialIRProperty::chi1inv && ir.eps_averaging &&
        destination.tensor_column >= 0 && !ir.contains_host_callback)
      for (uint64_t point = span.first_point; point < span.first_point + span.count; ++point) {
        const meep::volume evaluation = destination_evaluation_volume(ir, destination, point);
        const FrontObjectResult front = classify_front_object(ir, evaluation);
        const bool expected_bulk =
            !front.ambiguous &&
            (material_records_equal(ir, front.front, front.behind) ||
             ir_material_is_metal(ir.materials[front.front],
                                  type(component(destination.component))) ||
             ir_material_is_metal(ir.materials[front.behind],
                                  type(component(destination.component))));
        if (!expected_bulk)
          throw std::invalid_argument("material IR bulk span classification is stale");
      }
    previous_destination = span.destination;
    previous_end = span.first_point + span.count;
    have_span = true;
  }
  uint32_t previous_job_destination = 0;
  uint64_t previous_job_point = 0;
  bool have_job = false;
  for (const MaterialIRAnalyticInterface &job : ir.analytic_interfaces) {
    if (job.destination >= ir.destinations.size() ||
        job.point >= ir.destinations[job.destination].point_count ||
        job.front_material >= ir.materials.size() || job.behind_material >= ir.materials.size() ||
        job.object >= ir.objects.size() || job.image >= ir.images.size() ||
        ir.images[job.image].object != job.object || !std::isfinite(job.fill) ||
        !(job.fill > 0.0 && job.fill < 1.0) ||
        (have_job && (job.destination < previous_job_destination ||
                      (job.destination == previous_job_destination &&
                       job.point <= previous_job_point))))
      throw std::invalid_argument("material IR analytic interface is invalid");
    double norm2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(job.normal[axis]))
        throw std::invalid_argument("material IR analytic normal is non-finite");
      norm2 += job.normal[axis] * job.normal[axis];
    }
    if (!(norm2 > 0.0))
      throw std::invalid_argument("material IR analytic interface has a zero normal");
    const MaterialIRDestination &destination = ir.destinations[job.destination];
    const meep::volume evaluation = destination_evaluation_volume(ir, destination, job.point);
    const FrontObjectResult front = classify_front_object(ir, evaluation);
    vector3 expected_normal = {0, 0, 0};
    double expected_fill = 0.0;
    if (destination.property != MaterialIRProperty::chi1inv || !ir.eps_averaging ||
        destination.tensor_column < 0 || ir.contains_host_callback || front.ambiguous ||
        front.object == std::numeric_limits<uint32_t>::max() ||
        ir.materials[front.front].kind != meep_geom::material_data::MEDIUM ||
        ir.materials[front.behind].kind != meep_geom::material_data::MEDIUM ||
        !axis_aligned_block_face(ir, front.object, front.image, evaluation, expected_normal,
                                 expected_fill) ||
        front.object != job.object || front.image != job.image ||
        front.front != job.front_material || front.behind != job.behind_material ||
        expected_fill != job.fill || expected_normal.x != job.normal[0] ||
        expected_normal.y != job.normal[1] || expected_normal.z != job.normal[2])
      throw std::invalid_argument("material IR analytic interface is stale");
    previous_job_destination = job.destination;
    previous_job_point = job.point;
    have_job = true;
  }
  uint32_t previous_patch_destination = 0;
  uint64_t previous_patch_point = 0;
  bool have_patch = false;
  for (const MaterialIRHybridPatch &patch : ir.hybrid_patches) {
    if (patch.destination >= ir.destinations.size() ||
        patch.point >= ir.destinations[patch.destination].point_count ||
        patch.front_material >= ir.materials.size() ||
        patch.behind_material >= ir.materials.size() || !std::isfinite(patch.value) ||
        uint32_t(patch.reason) > uint32_t(MaterialIRPatchReason::negative_material_fallback) ||
        (patch.variable_causes & ~(material_variable_grid | material_variable_user |
                                  material_variable_file_default)) != 0 ||
        patch.variable_material != (patch.variable_causes != material_variable_none) ||
        (have_patch && (patch.destination < previous_patch_destination ||
                        (patch.destination == previous_patch_destination &&
                         patch.point <= previous_patch_point))))
      throw std::invalid_argument("material IR hybrid patch is invalid");
    const MaterialIRDestination &destination = ir.destinations[patch.destination];
    if (destination.property != MaterialIRProperty::chi1inv || !ir.eps_averaging ||
        destination.tensor_column < 0 || ir.contains_host_callback)
      throw std::invalid_argument("material IR hybrid patch route is invalid");
    const meep::volume evaluation = destination_evaluation_volume(ir, destination, patch.point);
    const FrontObjectResult front = classify_front_object(ir, evaluation);
    if (front.front != patch.front_material || front.behind != patch.behind_material ||
        front.object != patch.object || front.image != patch.image ||
        front.ambiguous != patch.ambiguous || front.variable_material != patch.variable_material ||
        front.variable_causes != patch.variable_causes)
      throw std::invalid_argument("material IR hybrid patch classification is stale");
    if (!front.ambiguous &&
        (material_records_equal(ir, front.front, front.behind) ||
         ir_material_is_metal(ir.materials[front.front], type(component(destination.component))) ||
         ir_material_is_metal(ir.materials[front.behind], type(component(destination.component)))))
      throw std::invalid_argument("material IR hybrid patch should be bulk work");
    vector3 analytic_normal = {0, 0, 0};
    double analytic_fill = 0.0;
    if (!front.ambiguous && front.object != std::numeric_limits<uint32_t>::max() &&
        ir.materials[front.front].kind == meep_geom::material_data::MEDIUM &&
        ir.materials[front.behind].kind == meep_geom::material_data::MEDIUM &&
        axis_aligned_block_face(ir, front.object, front.image, evaluation, analytic_normal,
                                analytic_fill))
      throw std::invalid_argument("material IR hybrid patch should be analytic work");
    MaterialIRPatchReason expected_reason = front.ambiguous
                                                ? MaterialIRPatchReason::ambiguous_front
                                                : MaterialIRPatchReason::unsupported_analytic_shape;
    if (patch.negative_fallback)
      expected_reason = MaterialIRPatchReason::negative_material_fallback;
    else if (front.variable_causes & material_variable_grid)
      expected_reason = MaterialIRPatchReason::material_grid_averaging;
    else if (patch.adaptive_fallback)
      expected_reason = MaterialIRPatchReason::adaptive_overlap;
    if (patch.reason != expected_reason)
      throw std::invalid_argument("material IR hybrid patch reason is stale");
    previous_patch_destination = patch.destination;
    previous_patch_point = patch.point;
    have_patch = true;
  }
  size_t bulk = 0, analytic = 0, patch = 0;
  const uint64_t absent = std::numeric_limits<uint64_t>::max();
  for (uint32_t destination = 0; destination < ir.destinations.size(); ++destination) {
    uint64_t expected = 0;
    while (expected < ir.destinations[destination].point_count) {
      const uint64_t bulk_point = bulk < ir.bulk_spans.size() &&
                                          ir.bulk_spans[bulk].destination == destination
                                      ? ir.bulk_spans[bulk].first_point
                                      : absent;
      const uint64_t analytic_point =
          analytic < ir.analytic_interfaces.size() &&
                  ir.analytic_interfaces[analytic].destination == destination
              ? ir.analytic_interfaces[analytic].point
              : absent;
      const uint64_t patch_point = patch < ir.hybrid_patches.size() &&
                                           ir.hybrid_patches[patch].destination == destination
                                       ? ir.hybrid_patches[patch].point
                                       : absent;
      const uint64_t next = std::min(bulk_point, std::min(analytic_point, patch_point));
      if (next != expected || (bulk_point == next) + (analytic_point == next) +
                                      (patch_point == next) !=
                                  1)
        throw std::invalid_argument("material IR destination partition overlaps or has a gap");
      if (bulk_point == next) {
        const uint64_t end = ir.bulk_spans[bulk].first_point + ir.bulk_spans[bulk].count;
        if ((analytic_point != absent && analytic_point < end) ||
            (patch_point != absent && patch_point < end))
          throw std::invalid_argument("material IR bulk span overlaps interface work");
        expected = end;
        ++bulk;
      }
      else if (analytic_point == next) {
        ++expected;
        ++analytic;
      }
      else {
        ++expected;
        ++patch;
      }
    }
  }
  if (bulk != ir.bulk_spans.size() || analytic != ir.analytic_interfaces.size() ||
      patch != ir.hybrid_patches.size())
    throw std::invalid_argument("material IR partition contains trailing work");
}

bool MaterialIRTopologyRow::operator==(const MaterialIRTopologyRow &other) const {
  if (!(key == other.key) || element_type != other.element_type ||
      logical_storage != other.logical_storage || elements != other.elements ||
      alignment != other.alignment || yee_component != other.yee_component)
    return false;
  for (int axis = 0; axis < 3; ++axis)
    if (extents[axis] != other.extents[axis] || strides[axis] != other.strides[axis] ||
        stagger[axis] != other.stagger[axis])
      return false;
  return true;
}

bool MaterialIRGeometryImage::operator==(const MaterialIRGeometryImage &other) const {
  if (object != other.object || ordinal != other.ordinal || precedence != other.precedence)
    return false;
  for (int axis = 0; axis < 3; ++axis)
    if (image[axis] != other.image[axis] || !equal_double(shift[axis], other.shift[axis]) ||
        !equal_double(low[axis], other.low[axis]) || !equal_double(high[axis], other.high[axis]))
      return false;
  return true;
}

bool MaterialIRDestination::operator==(const MaterialIRDestination &other) const {
  return key == other.key && topology_index == other.topology_index &&
         chunk_index == other.chunk_index && property == other.property &&
         component == other.component && tensor_direction == other.tensor_direction &&
         tensor_column == other.tensor_column && offdiagonal == other.offdiagonal &&
         point_count == other.point_count;
}

bool MaterialIRBulkSpan::operator==(const MaterialIRBulkSpan &other) const {
  return destination == other.destination && first_point == other.first_point &&
         count == other.count;
}

bool MaterialIRAnalyticInterface::operator==(const MaterialIRAnalyticInterface &other) const {
  if (destination != other.destination || point != other.point ||
      front_material != other.front_material || behind_material != other.behind_material ||
      object != other.object || image != other.image || fill != other.fill)
    return false;
  for (int axis = 0; axis < 3; ++axis)
    if (!equal_double(normal[axis], other.normal[axis])) return false;
  return true;
}

bool MaterialIRHybridPatch::operator==(const MaterialIRHybridPatch &other) const {
  return destination == other.destination && point == other.point && value == other.value &&
         front_material == other.front_material && behind_material == other.behind_material &&
         object == other.object && image == other.image && ambiguous == other.ambiguous &&
         variable_material == other.variable_material &&
         variable_causes == other.variable_causes &&
         adaptive_fallback == other.adaptive_fallback &&
         negative_fallback == other.negative_fallback &&
         reason == other.reason;
}

bool MaterialIRTriangle::operator==(const MaterialIRTriangle &other) const {
  for (int i = 0; i < 3; ++i)
    if (vertex[i] != other.vertex[i] || !equal_double(normal[i], other.normal[i]) ||
        !equal_double(low[i], other.low[i]) || !equal_double(high[i], other.high[i]))
      return false;
  return true;
}

bool MaterialIRBvhNode::operator==(const MaterialIRBvhNode &other) const {
  if (leaf != other.leaf || left != other.left || right != other.right ||
      first_triangle != other.first_triangle || triangle_count != other.triangle_count)
    return false;
  for (int i = 0; i < 3; ++i)
    if (!equal_double(low[i], other.low[i]) || !equal_double(high[i], other.high[i])) return false;
  return true;
}

bool material_ir_equal(const MaterialIR &a, const MaterialIR &b) {
  if (a.version != b.version || a.eps_averaging != b.eps_averaging ||
      a.subpixel_tol != b.subpixel_tol || a.subpixel_maxeval != b.subpixel_maxeval ||
      a.ensure_periodicity != b.ensure_periodicity ||
      a.contains_host_callback != b.contains_host_callback ||
      a.device_native_eligible != b.device_native_eligible ||
      a.requires_hybrid != b.requires_hybrid ||
      a.prism_include_boundaries != b.prism_include_boundaries ||
      a.dimensions != b.dimensions ||
      a.projection_offset != b.projection_offset || a.cell != b.cell ||
      a.default_material != b.default_material || a.root_count != b.root_count ||
      a.geometry_vertices != b.geometry_vertices ||
      a.geometry_triangles != b.geometry_triangles || a.geometry_bvh != b.geometry_bvh ||
      a.geometry_bvh_face_ids != b.geometry_bvh_face_ids ||
      a.images != b.images || a.active_images != b.active_images ||
      a.extra_materials != b.extra_materials || a.signature != b.signature ||
      a.layout_signature != b.layout_signature || a.topology != b.topology ||
      a.destinations != b.destinations || a.bulk_spans != b.bulk_spans ||
      a.analytic_interfaces != b.analytic_interfaces || a.hybrid_patches != b.hybrid_patches ||
      a.materials.size() != b.materials.size() || a.objects.size() != b.objects.size() ||
      a.susceptibilities.size() != b.susceptibilities.size() ||
      a.chunks.size() != b.chunks.size() || a.absorbers.size() != b.absorbers.size() ||
      a.pml_axes.size() != b.pml_axes.size())
    return false;
  for (int i = 0; i < 6; ++i)
    if (!equal_double(a.captured_volume[i], b.captured_volume[i])) return false;
  for (int i = 0; i < 3; ++i)
    if (!equal_double(a.lattice_basis_size[i], b.lattice_basis_size[i])) return false;
  for (int i = 0; i < 9; ++i)
    if (!equal_double(a.lattice_basis[i], b.lattice_basis[i]) ||
        !equal_double(a.lattice_metric[i], b.lattice_metric[i]) ||
        !equal_double(a.lattice_inverse[i], b.lattice_inverse[i]) ||
        !equal_double(a.lattice_inverse_transpose[i], b.lattice_inverse_transpose[i]))
      return false;
  for (size_t i = 0; i < a.materials.size(); ++i) {
    const MaterialIRMaterial &x = a.materials[i], &y = b.materials[i];
    if (x.kind != y.kind || x.host_callback != y.host_callback ||
        x.owned_callback != y.owned_callback ||
        x.callback_id != y.callback_id ||
        x.callback_signature != y.callback_signature ||
        x.callback_capabilities != y.callback_capabilities ||
        x.do_averaging != y.do_averaging || x.material_grid_kind != y.material_grid_kind ||
        x.material_grid_trivial != y.material_grid_trivial ||
        x.has_conductivity != y.has_conductivity || x.has_chi2 != y.has_chi2 ||
        x.has_chi3 != y.has_chi3 || x.e_susceptibilities != y.e_susceptibilities ||
        x.h_susceptibilities != y.h_susceptibilities ||
        x.comparison_medium != y.comparison_medium || x.parameters != y.parameters ||
        x.samples != y.samples)
      return false;
  }
  for (size_t i = 0; i < a.objects.size(); ++i) {
    const MaterialIRObject &x = a.objects[i], &y = b.objects[i];
    if (x.kind != y.kind || x.material != y.material ||
        x.source_identity != y.source_identity || x.root_identity != y.root_identity ||
        x.leaf_ordinal != y.leaf_ordinal || x.fixed_vertex_count != y.fixed_vertex_count ||
        x.vertex_offset != y.vertex_offset ||
        x.vertex_count != y.vertex_count || x.triangle_offset != y.triangle_offset ||
        x.triangle_count != y.triangle_count || x.bvh_offset != y.bvh_offset ||
        x.bvh_count != y.bvh_count || x.mesh_lengthscale != y.mesh_lengthscale ||
        x.parameters != y.parameters || x.vertices != y.vertices || x.indices != y.indices ||
        x.auxiliary != y.auxiliary)
      return false;
    for (int axis = 0; axis < 3; ++axis)
      if (!equal_double(x.parent_shift[axis], y.parent_shift[axis]) ||
          !equal_double(x.low[axis], y.low[axis]) ||
          !equal_double(x.high[axis], y.high[axis]))
        return false;
  }
  for (size_t i = 0; i < a.susceptibilities.size(); ++i) {
    const MaterialIRSusceptibility &x = a.susceptibilities[i], &y = b.susceptibilities[i];
    if (x.identity != y.identity || x.material != y.material || x.field_type != y.field_type ||
        x.material_ordinal != y.material_ordinal || x.parameters != y.parameters)
      return false;
  }
  for (size_t i = 0; i < a.chunks.size(); ++i) {
    const MaterialIRChunk &x = a.chunks[i], &y = b.chunks[i];
    if (x.chunk != y.chunk || x.dimensions != y.dimensions || x.owned != y.owned ||
        x.resolution != y.resolution || x.inva != y.inva || x.elements != y.elements ||
        x.component_bits != y.component_bits)
      return false;
    for (int axis = 0; axis < 3; ++axis)
      if (x.extents[axis] != y.extents[axis] || x.strides[axis] != y.strides[axis] ||
          x.little_corner[axis] != y.little_corner[axis] ||
          x.big_corner[axis] != y.big_corner[axis] || x.origin[axis] != y.origin[axis])
        return false;
    for (int component = 0; component < NUM_FIELD_COMPONENTS; ++component)
      {
        if (x.loop_count[component] != y.loop_count[component]) return false;
        for (int axis = 0; axis < 3; ++axis)
        if (x.stagger[component][axis] != y.stagger[component][axis]) return false;
        else if (x.loop_begin[component][axis] != y.loop_begin[component][axis] ||
                 x.loop_end[component][axis] != y.loop_end[component][axis])
          return false;
      }
    for (int d = 0; d < 6; ++d) if (x.pml_elements[d] != y.pml_elements[d]) return false;
  }
  for (size_t i = 0; i < a.absorbers.size(); ++i) {
    const MaterialIRPml &x = a.absorbers[i], &y = b.absorbers[i];
    if (x.direction != y.direction || x.side != y.side || x.thickness != y.thickness ||
        x.r_asymptotic != y.r_asymptotic || x.mean_stretch != y.mean_stretch ||
        x.sample_spacing != y.sample_spacing || x.samples != y.samples)
      return false;
  }
  for (size_t i = 0; i < a.pml_axes.size(); ++i) {
    const MaterialIRPmlAxis &x = a.pml_axes[i], &y = b.pml_axes[i];
    if (x.chunk != y.chunk || x.direction != y.direction || x.elements != y.elements ||
        x.little_corner != y.little_corner || x.resolution != y.resolution ||
        x.profile_active != y.profile_active ||
        x.analytic_quadratic != y.analytic_quadratic || x.thickness != y.thickness ||
        x.boundary_location != y.boundary_location || x.r_asymptotic != y.r_asymptotic ||
        x.mean_stretch != y.mean_stretch || x.profile_integral != y.profile_integral ||
        x.profile_integral_u != y.profile_integral_u ||
        x.profile_samples != y.profile_samples ||
        x.sigma != y.sigma || x.kappa != y.kappa || x.sigma_inv != y.sigma_inv)
      return false;
  }
  return true;
}

} // namespace meep
