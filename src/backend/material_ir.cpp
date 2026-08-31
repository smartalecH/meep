/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/material_ir.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

#include "material_data.hpp"
#include "meepgeom.hpp"
#include "backend/precision.hpp"

namespace meep {
namespace {

const uint32_t material_ir_version = 2;
int material_ir_capture_failure_rank = -1;
int material_ir_capture_failure_mode = 0;

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
                          std::map<const void *, uint32_t> &seen) {
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

uint32_t capture_object(MaterialIR &ir, const geometric_object &object,
                        std::map<const void *, uint32_t> &materials) {
  if (ir.objects.size() >= std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("material IR object count overflow");
  MaterialIRObject out;
  out.kind = object.which_subclass;
  const uint32_t material = capture_material(
      ir, static_cast<const meep_geom::material_data *>(object.material), materials);
  if (material > uint32_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("material IR object material ID overflow");
  out.material = int(material);
  append_vec(out.parameters, object.center);
  switch (object.which_subclass) {
    case geometric_object::MESH: {
      const mesh &m = *object.subclass.mesh_data;
      if (m.vertices.num_items < 0 || m.face_indices.num_items < 0)
        throw std::invalid_argument("material IR mesh has a negative count");
      out.parameters.push_back(m.is_closed ? 1 : 0);
      for (int i = 0; i < m.vertices.num_items; ++i) append_vec(out.vertices, m.vertices.items[i]);
      for (int i = 0; i < m.face_indices.num_items; ++i) append_vec(out.indices, m.face_indices.items[i]);
      break;
    }
    case geometric_object::PRISM: {
      const prism &p = *object.subclass.prism_data;
      if (p.vertices.num_items < 0)
        throw std::invalid_argument("material IR prism has a negative vertex count");
      out.parameters.push_back(p.height); append_vec(out.parameters, p.axis);
      out.parameters.push_back(p.sidewall_angle); append_vec(out.parameters, p.centroid);
      append_matrix(out.parameters, p.m_c2p); append_matrix(out.parameters, p.m_p2c);
      for (int i = 0; i < p.vertices.num_items; ++i) append_vec(out.vertices, p.vertices.items[i]);
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
    case geometric_object::COMPOUND_GEOMETRIC_OBJECT: break;
    case geometric_object::GEOMETRIC_OBJECT_SELF: break;
  }
  const uint32_t id = uint32_t(ir.objects.size());
  ir.objects.push_back(out);
  if (object.which_subclass == geometric_object::COMPOUND_GEOMETRIC_OBJECT) {
    const geometric_object_list &children =
        object.subclass.compound_geometric_object_data->component_objects;
    if (children.num_items < 0)
      throw std::invalid_argument("material IR compound has a negative child count");
    for (int i = 0; i < children.num_items; ++i) {
      const uint32_t child = capture_object(ir, children.items[i], materials);
      /* Recursive capture may reallocate ir.objects; resolve the parent again
         only after the child has been appended. */
      ir.objects[id].children.push_back(child);
    }
  }
  return id;
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
  uint64_t bits = 0; memcpy(&bits, &value, sizeof(bits)); mix_u64(h, bits);
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
  mix_i64(h, ir.dimensions); mix_values(h, ir.cell); mix_u64(h, ir.default_material);
  mix_u64(h, ir.materials.size());
  for (const MaterialIRMaterial &m : ir.materials) {
    mix_tag(h, "material"); mix_i64(h, m.kind); mix_bool(h, m.host_callback);
    mix_bool(h, m.do_averaging); mix_i64(h, m.material_grid_kind);
    mix_bool(h, m.material_grid_trivial);
    mix_bool(h, m.has_conductivity); mix_bool(h, m.has_chi2); mix_bool(h, m.has_chi3);
    mix_u64(h, m.e_susceptibilities); mix_u64(h, m.h_susceptibilities);
    mix_values(h, m.parameters); mix_values(h, m.samples);
  }
  mix_u64(h, ir.objects.size());
  for (const MaterialIRObject &o : ir.objects) {
    mix_tag(h, "object"); mix_i64(h, o.kind); mix_i64(h, o.material);
    mix_u64(h, o.children.size()); for (uint32_t child : o.children) mix_u64(h, child);
    mix_values(h, o.parameters); mix_values(h, o.vertices); mix_values(h, o.indices);
  }
  mix_u64(h, ir.roots.size()); for (uint32_t root : ir.roots) mix_u64(h, root);
  mix_u64(h, ir.extra_materials.size());
  for (uint32_t material : ir.extra_materials) mix_u64(h, material);
  mix_u64(h, ir.susceptibilities.size());
  for (const MaterialIRSusceptibility &s : ir.susceptibilities) {
    mix_u64(h, s.identity); mix_u64(h, s.material); mix_i64(h, s.field_type);
    mix_u64(h, s.material_ordinal); mix_values(h, s.parameters);
  }
  if (include_rank_layout) {
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

} // namespace

std::shared_ptr<const void> capture_material_ir(const structure &s,
                                               const meep_geom::geom_epsilon &geps,
                                               bool eps_averaging, double tol, int maxeval,
                                               const void *absorbers) {
  if (material_ir_capture_failure_rank == my_rank()) {
    if (material_ir_capture_failure_mode == 1)
      throw std::invalid_argument("injected material IR capture failure");
    if (material_ir_capture_failure_mode == 2) throw std::bad_alloc();
  }
  std::shared_ptr<MaterialIR> ir(new MaterialIR);
  ir->version = material_ir_version; ir->eps_averaging = eps_averaging;
  ir->subpixel_tol = tol; ir->subpixel_maxeval = eps_averaging ? maxeval : 0;
  ir->ensure_periodicity = geps.captured_ensure_periodicity; ir->contains_host_callback = false;
  if (s.num_chunks <= 0 || !s.chunks[0])
    throw std::invalid_argument("material IR has no live chunk dimension authority");
  ir->dimensions = int(s.chunks[0]->gv.dim);
  ir->signature = 0; ir->layout_signature = 0;
  append_vec(ir->cell, geps.captured_geometry_center);
  append_vec(ir->cell, geps.captured_geometry_lattice.size);
  append_vec(ir->cell, geps.captured_geometry_lattice.basis1);
  append_vec(ir->cell, geps.captured_geometry_lattice.basis2);
  append_vec(ir->cell, geps.captured_geometry_lattice.basis3);
  std::map<const void *, uint32_t> materials;
  ir->default_material = capture_material(*ir, &geps.owned_default_material(), materials);
  if (geps.geometry.num_items < 0)
    throw std::invalid_argument("material IR geometry has a negative root count");
  for (int i = 0; i < geps.geometry.num_items; ++i)
    ir->roots.push_back(capture_object(*ir, geps.geometry.items[i], materials));
  const meep_geom::material_type_list &extra = geps.owned_extra_materials();
  if (extra.num_items < 0)
    throw std::invalid_argument("material IR extra-material count is negative");
  for (int i = 0; i < extra.num_items; ++i)
    ir->extra_materials.push_back(capture_material(*ir, extra.items[i], materials));
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
  const MaterialIRMaterial &root_material = ir->materials[ir->default_material];
  ir->device_native_eligible =
      ir->objects.empty() && !root_material.host_callback &&
      !(root_material.kind == meep_geom::material_data::MATERIAL_GRID &&
        root_material.do_averaging);
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
  ir->signature = signature(*ir, false);
  ir->layout_signature = signature(*ir, true);
  validate_material_ir(*ir);
  return std::static_pointer_cast<const void>(ir);
}

const MaterialIR *material_ir_for(const fields &f) {
  return static_cast<const MaterialIR *>(f.material_ir.get());
}

void refresh_material_ir_signatures_for_testing(MaterialIR &ir) {
  ir.signature = signature(ir, false);
  ir.layout_signature = signature(ir, true);
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
      ir.signature != signature(ir, false) ||
      ir.layout_signature != signature(ir, true))
    throw std::invalid_argument("material IR is malformed or stale");
  if (ir.materials.empty() || ir.default_material >= ir.materials.size() || ir.cell.size() != 15)
    throw std::invalid_argument("material IR root metadata is invalid");
  const auto finite = [](const std::vector<double> &values) {
    for (double value : values) if (!std::isfinite(value)) return false;
    return true;
  };
  if (!finite(ir.cell)) throw std::invalid_argument("material IR cell is non-finite");
  bool contains_host_callback = false;
  for (const MaterialIRMaterial &m : ir.materials) {
    if (m.kind < meep_geom::material_data::MEDIUM ||
        m.kind > meep_geom::material_data::PERFECT_METAL ||
        (m.kind == meep_geom::material_data::MATERIAL_USER) != m.host_callback ||
        (m.kind != meep_geom::material_data::MATERIAL_GRID &&
         (m.material_grid_kind != -1 || m.material_grid_trivial)) ||
        (m.kind != meep_geom::material_data::MATERIAL_GRID &&
         m.kind != meep_geom::material_data::MATERIAL_USER && m.do_averaging) ||
        (m.kind == meep_geom::material_data::MATERIAL_GRID &&
         (m.material_grid_kind < meep_geom::material_data::U_MIN ||
          m.material_grid_kind > meep_geom::material_data::U_DEFAULT)) ||
        !finite(m.parameters) || !finite(m.samples))
      throw std::invalid_argument("material IR material record is invalid");
    contains_host_callback = contains_host_callback || m.host_callback;
    if ((m.kind == meep_geom::material_data::PERFECT_METAL ||
         m.kind == meep_geom::material_data::MATERIAL_USER) &&
        (!m.parameters.empty() || !m.samples.empty()))
      throw std::invalid_argument("material IR tag has an unexpected payload");
    size_t payload_offset = 0;
    uint32_t expected_e = 0, expected_h = 0;
    if (m.kind == meep_geom::material_data::MEDIUM ||
        m.kind == meep_geom::material_data::MATERIAL_FILE) {
      validate_medium_payload(m.parameters, payload_offset, &expected_e, &expected_h);
      if (m.e_susceptibilities != expected_e || m.h_susceptibilities != expected_h)
        throw std::invalid_argument("material IR medium susceptibility count is inconsistent");
    }
    if (m.kind == meep_geom::material_data::MEDIUM &&
        (payload_offset != m.parameters.size() || !m.samples.empty()))
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
      validate_medium_payload(m.parameters, payload_offset, &e0, &h0);
      validate_medium_payload(m.parameters, payload_offset, &e1, &h1);
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
  const MaterialIRMaterial &root_material = ir.materials[ir.default_material];
  const bool expected_native =
      ir.objects.empty() && !root_material.host_callback &&
      !(root_material.kind == meep_geom::material_data::MATERIAL_GRID &&
        root_material.do_averaging);
  if (ir.contains_host_callback != contains_host_callback ||
      ir.device_native_eligible != expected_native)
    throw std::invalid_argument("material IR callback eligibility is inconsistent");
  std::vector<unsigned> parents(ir.objects.size(), 0);
  for (size_t oi = 0; oi < ir.objects.size(); ++oi) {
    const MaterialIRObject &o = ir.objects[oi];
    if (o.kind < geometric_object::GEOMETRIC_OBJECT_SELF ||
        o.kind > geometric_object::COMPOUND_GEOMETRIC_OBJECT || o.material < 0 ||
        size_t(o.material) >= ir.materials.size() || !finite(o.vertices) || !finite(o.indices))
      throw std::invalid_argument("material IR object record is invalid");
    for (uint32_t child : o.children)
      if (child >= ir.objects.size() || child <= oi || ++parents[child] != 1)
        throw std::invalid_argument("material IR child reference is invalid");
    const size_t parameter_count = o.parameters.size();
    if (o.kind != geometric_object::COMPOUND_GEOMETRIC_OBJECT && !o.children.empty())
      throw std::invalid_argument("material IR non-compound object has children");
    if ((o.kind == geometric_object::MESH &&
         (parameter_count != 4 || o.vertices.size() % 3 || o.indices.size() % 3)) ||
        (o.kind == geometric_object::PRISM &&
         (parameter_count != 29 || o.vertices.size() % 3 || !o.indices.empty())) ||
        (o.kind == geometric_object::SPHERE &&
         (parameter_count != 4 || !o.vertices.empty() || !o.indices.empty())) ||
        (o.kind == geometric_object::BLOCK &&
         ((parameter_count != 25 && parameter_count != 28) || !o.vertices.empty() ||
          !o.indices.empty())) ||
        (o.kind == geometric_object::CYLINDER &&
         ((parameter_count != 9 && parameter_count != 10 && parameter_count != 19) ||
          !o.vertices.empty() || !o.indices.empty())) ||
        (o.kind == geometric_object::COMPOUND_GEOMETRIC_OBJECT &&
         (parameter_count != 3 || !o.vertices.empty() || !o.indices.empty())) ||
        (o.kind == geometric_object::GEOMETRIC_OBJECT_SELF &&
         (parameter_count != 3 || !o.vertices.empty() || !o.indices.empty())))
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
      const size_t vertices = o.vertices.size() / 3;
      for (double index : o.indices)
        if (index < 0 || std::floor(index) != index || index >= double(vertices))
          throw std::invalid_argument("material IR mesh index is invalid");
    }
  }
  std::set<uint32_t> root_ids;
  for (uint32_t root : ir.roots)
    if (root >= ir.objects.size() || parents[root] != 0)
      throw std::invalid_argument("material IR root is invalid");
    else if (!root_ids.insert(root).second)
      throw std::invalid_argument("material IR root is duplicated");
  std::vector<uint8_t> reached(ir.objects.size(), 0), active(ir.objects.size(), 0);
  std::function<void(uint32_t)> visit = [&](uint32_t id) {
    if (active[id]) throw std::invalid_argument("material IR object graph is cyclic");
    if (reached[id]) return;
    active[id] = 1;
    for (uint32_t child : ir.objects[id].children) visit(child);
    active[id] = 0; reached[id] = 1;
  };
  for (uint32_t root : ir.roots) visit(root);
  for (size_t i = 0; i < reached.size(); ++i)
    if (!reached[i]) throw std::invalid_argument("material IR object is unreachable");
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

bool material_ir_equal(const MaterialIR &a, const MaterialIR &b) {
  if (a.version != b.version || a.eps_averaging != b.eps_averaging ||
      a.subpixel_tol != b.subpixel_tol || a.subpixel_maxeval != b.subpixel_maxeval ||
      a.ensure_periodicity != b.ensure_periodicity ||
      a.contains_host_callback != b.contains_host_callback ||
      a.device_native_eligible != b.device_native_eligible || a.dimensions != b.dimensions ||
      a.cell != b.cell || a.default_material != b.default_material || a.roots != b.roots ||
      a.extra_materials != b.extra_materials || a.signature != b.signature ||
      a.layout_signature != b.layout_signature || a.topology != b.topology ||
      a.materials.size() != b.materials.size() || a.objects.size() != b.objects.size() ||
      a.susceptibilities.size() != b.susceptibilities.size() ||
      a.chunks.size() != b.chunks.size() || a.absorbers.size() != b.absorbers.size() ||
      a.pml_axes.size() != b.pml_axes.size())
    return false;
  for (size_t i = 0; i < a.materials.size(); ++i) {
    const MaterialIRMaterial &x = a.materials[i], &y = b.materials[i];
    if (x.kind != y.kind || x.host_callback != y.host_callback ||
        x.do_averaging != y.do_averaging || x.material_grid_kind != y.material_grid_kind ||
        x.material_grid_trivial != y.material_grid_trivial ||
        x.has_conductivity != y.has_conductivity || x.has_chi2 != y.has_chi2 ||
        x.has_chi3 != y.has_chi3 || x.e_susceptibilities != y.e_susceptibilities ||
        x.h_susceptibilities != y.h_susceptibilities || x.parameters != y.parameters ||
        x.samples != y.samples)
      return false;
  }
  for (size_t i = 0; i < a.objects.size(); ++i) {
    const MaterialIRObject &x = a.objects[i], &y = b.objects[i];
    if (x.kind != y.kind || x.material != y.material || x.children != y.children ||
        x.parameters != y.parameters || x.vertices != y.vertices || x.indices != y.indices)
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
