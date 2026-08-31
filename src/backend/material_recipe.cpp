/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/material_recipe.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <sstream>
#include <set>
#include <stdexcept>

#include "backend/classification.hpp"
#include "backend/precision.hpp"
#include "material_data.hpp"
#include "meep_internals.hpp"

namespace meep {

namespace {

const uint32_t material_recipe_format_version = 1;
int material_recipe_failure_rank_for_testing = -1;
int material_recipe_failure_mode_for_testing = 0;

void mix_byte(uint64_t &hash, unsigned char value) {
  hash ^= uint64_t(value);
  hash *= UINT64_C(1099511628211);
}

void mix_u64(uint64_t &hash, uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    mix_byte(hash, static_cast<unsigned char>((value >> (8 * i)) & 0xffu));
}

void mix_bytes(uint64_t &hash, const void *data, size_t bytes) {
  const unsigned char *p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < bytes; ++i) mix_byte(hash, p[i]);
}

void mix_realnum_values(uint64_t &hash, const std::vector<unsigned char> &values) {
  if (values.size() % sizeof(realnum))
    throw std::invalid_argument("material recipe row has a partial scalar");
  mix_u64(hash, values.size() / sizeof(realnum));
  for (size_t offset = 0; offset < values.size(); offset += sizeof(realnum)) {
    realnum value = 0;
    memcpy(&value, values.data() + offset, sizeof(value));
    if (sizeof(realnum) == sizeof(float)) {
      uint32_t bits = 0;
      const float narrowed = float(value);
      memcpy(&bits, &narrowed, sizeof(bits));
      mix_u64(hash, bits);
    }
    else {
      uint64_t bits = 0;
      const double widened = double(value);
      memcpy(&bits, &widened, sizeof(bits));
      mix_u64(hash, bits);
    }
  }
}

void mix_string(uint64_t &hash, const std::string &value) {
  mix_u64(hash, value.size());
  if (!value.empty()) mix_bytes(hash, value.data(), value.size());
}

void mix_key(uint64_t &hash, const StorageKey &key) {
  mix_u64(hash, uint64_t(int64_t(key.chunk)));
  mix_u64(hash, uint64_t(int64_t(key.kind)));
  mix_u64(hash, uint64_t(int64_t(key.component_)));
  mix_u64(hash, uint64_t(int64_t(key.cmp)));
  mix_u64(hash, key.aux);
}

void mix_topology(uint64_t &hash, const MaterialIRTopologyRow &row) {
  mix_key(hash, row.key);
  mix_u64(hash, uint64_t(row.element_type));
  mix_u64(hash, uint64_t(row.logical_storage));
  mix_u64(hash, row.elements);
  mix_u64(hash, row.alignment);
  mix_u64(hash, uint64_t(int64_t(row.yee_component)));
  for (int axis = 0; axis < 3; ++axis) {
    mix_u64(hash, uint64_t(int64_t(row.extents[axis])));
    mix_u64(hash, uint64_t(int64_t(row.strides[axis])));
    mix_u64(hash, uint64_t(int64_t(row.stagger[axis])));
  }
}

size_t checked_host_bytes(ElementType type, size_t elements) {
  const size_t width = host_element_bytes(type);
  if (elements && width > std::numeric_limits<size_t>::max() / elements)
    throw std::overflow_error("material recipe row byte count overflow");
  return width * elements;
}

uint64_t compute_signature(uint32_t version, MaterialRecipeDisposition disposition,
                           const std::string &description, bool eps_averaging,
                           double subpixel_tol, int subpixel_maxeval,
                           uint32_t host_callback_id, bool from_host_callback,
                           uint64_t support_reason_bits,
                           const std::vector<MaterialRecipeRow> &rows,
                           const std::vector<MaterialIRTopologyRow> &topology,
                           const std::shared_ptr<const MaterialIR> &ir) {
  uint64_t hash = UINT64_C(1469598103934665603);
  mix_u64(hash, version);
  mix_u64(hash, uint64_t(disposition));
  mix_string(hash, description);
  mix_u64(hash, eps_averaging ? 1 : 0);
  uint64_t tolerance_bits = 0;
  static_assert(sizeof(tolerance_bits) == sizeof(subpixel_tol), "unexpected double size");
  memcpy(&tolerance_bits, &subpixel_tol, sizeof(tolerance_bits));
  mix_u64(hash, tolerance_bits);
  mix_u64(hash, uint64_t(int64_t(subpixel_maxeval)));
  mix_u64(hash, host_callback_id);
  mix_u64(hash, from_host_callback ? 1 : 0);
  mix_u64(hash, support_reason_bits);
  mix_u64(hash, rows.size());
  for (const MaterialRecipeRow &row : rows) {
    mix_key(hash, row.key);
    mix_u64(hash, uint64_t(row.role));
    mix_u64(hash, uint64_t(row.element_type));
    mix_u64(hash, uint64_t(row.storage));
    mix_u64(hash, row.elements);
    mix_u64(hash, row.alignment);
    mix_realnum_values(hash, row.values);
  }
  mix_u64(hash, topology.size());
  for (const MaterialIRTopologyRow &row : topology) mix_topology(hash, row);
  mix_u64(hash, ir ? ir->signature : 0);
  mix_u64(hash, ir ? ir->layout_signature : 0);
  return hash;
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

uint64_t checked_u64_add(uint64_t a, uint64_t b, const char *what) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error(std::string("material support ") + what + " overflow");
  return a + b;
}

uint64_t checked_u64_bytes(size_t count, size_t width, const char *what) {
  if (count && width > std::numeric_limits<uint64_t>::max() / count)
    throw std::overflow_error(std::string("material support ") + what + " overflow");
  return uint64_t(count) * uint64_t(width);
}

void add_compact_bytes(uint64_t &total, size_t count, size_t width) {
  total = checked_u64_add(total, checked_u64_bytes(count, width, "compact input bytes"),
                          "compact input bytes");
}

uint64_t compact_material_ir_bytes(const MaterialIR &ir) {
  uint64_t total = 0;
#define ADD_SCALAR(member) add_compact_bytes(total, 1, sizeof(member))
  ADD_SCALAR(ir.version);
  ADD_SCALAR(ir.eps_averaging);
  ADD_SCALAR(ir.subpixel_tol);
  ADD_SCALAR(ir.subpixel_maxeval);
  ADD_SCALAR(ir.ensure_periodicity);
  ADD_SCALAR(ir.contains_host_callback);
  ADD_SCALAR(ir.device_native_eligible);
  ADD_SCALAR(ir.dimensions);
  ADD_SCALAR(ir.default_material);
  ADD_SCALAR(ir.signature);
  ADD_SCALAR(ir.layout_signature);
  add_compact_bytes(total, ir.cell.size(), sizeof(double));
  add_compact_bytes(total, ir.roots.size(), sizeof(uint32_t));
  add_compact_bytes(total, ir.extra_materials.size(), sizeof(uint32_t));
  for (const MaterialIRMaterial &m : ir.materials) {
    ADD_SCALAR(m.kind);
    ADD_SCALAR(m.host_callback);
    ADD_SCALAR(m.do_averaging);
    ADD_SCALAR(m.material_grid_kind);
    ADD_SCALAR(m.material_grid_trivial);
    ADD_SCALAR(m.has_conductivity);
    ADD_SCALAR(m.has_chi2);
    ADD_SCALAR(m.has_chi3);
    ADD_SCALAR(m.e_susceptibilities);
    ADD_SCALAR(m.h_susceptibilities);
    add_compact_bytes(total, m.parameters.size(), sizeof(double));
    add_compact_bytes(total, m.samples.size(), sizeof(double));
  }
  for (const MaterialIRObject &o : ir.objects) {
    ADD_SCALAR(o.kind);
    ADD_SCALAR(o.material);
    add_compact_bytes(total, o.children.size(), sizeof(uint32_t));
    add_compact_bytes(total, o.parameters.size(), sizeof(double));
    add_compact_bytes(total, o.vertices.size(), sizeof(double));
    add_compact_bytes(total, o.indices.size(), sizeof(double));
  }
  for (const MaterialIRSusceptibility &s : ir.susceptibilities) {
    ADD_SCALAR(s.identity);
    ADD_SCALAR(s.material);
    ADD_SCALAR(s.field_type);
    ADD_SCALAR(s.material_ordinal);
    add_compact_bytes(total, s.parameters.size(), sizeof(double));
  }
  for (const MaterialIRChunk &c : ir.chunks) {
    ADD_SCALAR(c.chunk);
    ADD_SCALAR(c.dimensions);
    ADD_SCALAR(c.owned);
    ADD_SCALAR(c.resolution);
    ADD_SCALAR(c.inva);
    ADD_SCALAR(c.elements);
    ADD_SCALAR(c.component_bits);
    add_compact_bytes(total, 3, sizeof(c.extents[0]));
    add_compact_bytes(total, 3, sizeof(c.strides[0]));
    add_compact_bytes(total, 3, sizeof(c.little_corner[0]));
    add_compact_bytes(total, 3, sizeof(c.big_corner[0]));
    add_compact_bytes(total, 3, sizeof(c.origin[0]));
    add_compact_bytes(total, NUM_FIELD_COMPONENTS * 3, sizeof(c.stagger[0][0]));
    add_compact_bytes(total, NUM_FIELD_COMPONENTS * 3, sizeof(c.loop_begin[0][0]));
    add_compact_bytes(total, NUM_FIELD_COMPONENTS * 3, sizeof(c.loop_end[0][0]));
    add_compact_bytes(total, NUM_FIELD_COMPONENTS, sizeof(c.loop_count[0]));
    add_compact_bytes(total, 6, sizeof(c.pml_elements[0]));
  }
  for (const MaterialIRPml &p : ir.absorbers) {
    ADD_SCALAR(p.direction);
    ADD_SCALAR(p.side);
    ADD_SCALAR(p.thickness);
    ADD_SCALAR(p.r_asymptotic);
    ADD_SCALAR(p.mean_stretch);
    ADD_SCALAR(p.sample_spacing);
    add_compact_bytes(total, p.samples.size(), sizeof(double));
  }
  for (const MaterialIRPmlAxis &p : ir.pml_axes) {
    ADD_SCALAR(p.chunk);
    ADD_SCALAR(p.direction);
    ADD_SCALAR(p.elements);
    ADD_SCALAR(p.little_corner);
    ADD_SCALAR(p.resolution);
    ADD_SCALAR(p.profile_active);
    ADD_SCALAR(p.analytic_quadratic);
    ADD_SCALAR(p.thickness);
    ADD_SCALAR(p.boundary_location);
    ADD_SCALAR(p.r_asymptotic);
    ADD_SCALAR(p.mean_stretch);
    ADD_SCALAR(p.profile_integral);
    ADD_SCALAR(p.profile_integral_u);
    add_compact_bytes(total, p.profile_samples.size(), sizeof(double));
    add_compact_bytes(total, p.sigma.size(), sizeof(double));
    add_compact_bytes(total, p.kappa.size(), sizeof(double));
    add_compact_bytes(total, p.sigma_inv.size(), sizeof(double));
  }
  for (const MaterialIRTopologyRow &row : ir.topology) {
    ADD_SCALAR(row.key.chunk);
    ADD_SCALAR(row.key.kind);
    ADD_SCALAR(row.key.component_);
    ADD_SCALAR(row.key.cmp);
    ADD_SCALAR(row.key.aux);
    ADD_SCALAR(row.element_type);
    ADD_SCALAR(row.logical_storage);
    ADD_SCALAR(row.elements);
    ADD_SCALAR(row.alignment);
    ADD_SCALAR(row.yee_component);
    add_compact_bytes(total, 3, sizeof(row.extents[0]));
    add_compact_bytes(total, 3, sizeof(row.strides[0]));
    add_compact_bytes(total, 3, sizeof(row.stagger[0]));
  }
#undef ADD_SCALAR
  return total;
}

MaterialSupportDecision classify_support_input(
    const std::shared_ptr<const MaterialIR> &ir,
    const std::vector<MaterialRecipeRow> &rows) {
  MaterialSupportDecision decision = {MaterialRecipeDisposition::host_reference,
                                      material_support_no_owned_ir,
                                      0, 0, 0, 0, 0, 0};
  for (const MaterialRecipeRow &row : rows)
    decision.dense_fallback_bytes = checked_u64_add(
        decision.dense_fallback_bytes,
        checked_u64_bytes(row.values.size(), 1, "dense byte count"), "dense byte count");
  if (!ir) {
    return decision;
  }

  decision.reason_bits = material_support_none;
  if (!ir->objects.empty()) decision.reason_bits |= material_support_object_lookup;
  if (ir->default_material >= ir->materials.size())
    throw std::invalid_argument("material support has an invalid default material");
  std::vector<uint8_t> used(ir->materials.size(), 0);
  used[ir->default_material] = 1;
  for (const MaterialIRObject &object : ir->objects) {
    if (object.material < 0 || size_t(object.material) >= used.size())
      throw std::invalid_argument("material support has an invalid object material");
    used[size_t(object.material)] = 1;
  }
  for (size_t i = 0; i < ir->materials.size(); ++i) {
    if (!used[i]) continue;
    const MaterialIRMaterial &material = ir->materials[i];
    if (material.host_callback)
      decision.reason_bits |= material_support_unowned_callback;
    if (material.kind == meep_geom::material_data::MATERIAL_GRID && material.do_averaging)
      decision.reason_bits |= material_support_adaptive_averaging;
  }

  decision.compact_input_bytes = compact_material_ir_bytes(*ir);
  for (const MaterialIRTopologyRow &row : ir->topology)
    decision.native_points =
        checked_u64_add(decision.native_points, uint64_t(row.elements), "native point count");

  decision.disposition = decision.reason_bits == material_support_none
                             ? MaterialRecipeDisposition::device_native
                             : MaterialRecipeDisposition::host_reference;
  return decision;
}

void validate_input(const MaterialRecipeInput &input) {
  if (input.description.empty()) throw std::invalid_argument("material recipe has no description");
  if (!std::isfinite(input.subpixel_tol) || input.subpixel_tol <= 0.0)
    throw std::invalid_argument("material recipe has an invalid subpixel tolerance");
  if (input.subpixel_maxeval < 0 || (input.eps_averaging && input.subpixel_maxeval == 0))
    throw std::invalid_argument("material recipe has an invalid subpixel evaluation limit");
  if (input.disposition == MaterialRecipeDisposition::hybrid_interface ||
      input.disposition == MaterialRecipeDisposition::tiled_callback)
    throw std::invalid_argument("material recipe disposition lacks owned PR5.1 work records");
  if (input.disposition != MaterialRecipeDisposition::host_reference &&
      input.disposition != MaterialRecipeDisposition::device_native)
    throw std::invalid_argument("material recipe disposition is invalid");
  if (input.from_host_callback || input.host_callback_id != invalid_array_value)
    throw std::invalid_argument("host-reference material recipe cannot contain a callback");
  if (input.ir) {
    validate_material_ir(*input.ir);
    if (input.eps_averaging != input.ir->eps_averaging ||
        input.subpixel_tol != input.ir->subpixel_tol ||
        input.subpixel_maxeval != input.ir->subpixel_maxeval)
      throw std::invalid_argument("material recipe policy differs from its immutable IR");
  }
  const MaterialSupportDecision expected = classify_support_input(input.ir, input.rows);
  if (input.disposition != expected.disposition ||
      input.support_reason_bits != expected.reason_bits)
    throw std::invalid_argument("material recipe support disposition is stale or inconsistent");

  std::map<StorageKey, const MaterialIRTopologyRow *, StorageKeyLess> ir_rows;
  if (input.ir)
    for (const MaterialIRTopologyRow &row : input.ir->topology)
      ir_rows.insert(std::make_pair(row.key, &row));

  std::set<StorageKey, StorageKeyLess> keys;
  size_t total_bytes = 0;
  for (const MaterialRecipeRow &row : input.rows) {
    if (row.role != array_role::material)
      throw std::invalid_argument("material recipe row has a non-material role");
    const array_kind kind = static_cast<array_kind>(row.key.kind);
    if (kind != array_kind::chi1inv && kind != array_kind::conductivity &&
        kind != array_kind::condinv && kind != array_kind::chi2 &&
        kind != array_kind::chi3 && kind != array_kind::sigma &&
        kind != array_kind::pml_sig && kind != array_kind::pml_kap &&
        kind != array_kind::pml_siginv)
      throw std::invalid_argument("material recipe row has a non-material storage kind");
    if (row.key.chunk < 0)
      throw std::invalid_argument("material recipe row has an invalid storage key");
    const bool pml = kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
                     kind == array_kind::pml_siginv;
    const bool diagonal = kind == array_kind::chi2 || kind == array_kind::chi3;
    const bool tensor = kind == array_kind::chi1inv || kind == array_kind::conductivity ||
                        kind == array_kind::condinv;
    const bool valid_key =
        (pml && row.key.component_ == -1 && row.key.cmp == -1 && row.key.aux < 6) ||
        (diagonal && row.key.component_ >= 0 && row.key.component_ < NUM_FIELD_COMPONENTS &&
         row.key.cmp == -1 && row.key.aux == 0) ||
        (tensor && row.key.component_ >= 0 && row.key.component_ < NUM_FIELD_COMPONENTS &&
         row.key.cmp == -1 && row.key.aux < 5) ||
        (kind == array_kind::sigma && row.key.component_ >= 0 &&
         row.key.component_ < NUM_FIELD_COMPONENTS && row.key.cmp >= 0 && row.key.cmp < 5);
    if (!valid_key) throw std::invalid_argument("material recipe row has an invalid storage key");
    if (kind == array_kind::sigma) {
      const field_type ft = field_type(row.key.aux % uint64_t(NUM_FIELD_TYPES));
      const uint64_t state = row.key.aux / uint64_t(NUM_FIELD_TYPES);
      if ((ft != E_stuff && ft != H_stuff) || state > uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("material recipe sigma identity is invalid");
    }
    if (row.element_type != ElementType::realnum_value || row.storage != native_precision)
      throw std::invalid_argument("host-reference material row is not native realnum storage");
    if (row.alignment != alignof(realnum))
      throw std::invalid_argument("material recipe row has an invalid alignment");
    if (!row.elements)
      throw std::invalid_argument("material recipe row has zero extent");
    const size_t bytes = checked_host_bytes(row.element_type, row.elements);
    if (row.values.size() != bytes)
      throw std::invalid_argument("material recipe row byte count does not match its extent");
    if (bytes > std::numeric_limits<size_t>::max() - total_bytes)
      throw std::overflow_error("material recipe total byte count overflow");
    total_bytes += bytes;
    if (!keys.insert(row.key).second)
      throw std::invalid_argument("material recipe contains a duplicate storage key");
    if (input.ir) {
      const std::map<StorageKey, const MaterialIRTopologyRow *, StorageKeyLess>::const_iterator
          expected = ir_rows.find(row.key);
      if (expected == ir_rows.end() || expected->second->element_type != row.element_type ||
          expected->second->logical_storage != row.storage ||
          expected->second->elements != row.elements ||
          expected->second->alignment != row.alignment) {
        std::ostringstream message;
        message << "material recipe row differs from immutable IR topology (chunk="
                << row.key.chunk << ", kind=" << row.key.kind << ", component="
                << row.key.component_ << ", cmp=" << row.key.cmp << ", aux=" << row.key.aux
                << ", elements=" << row.elements;
        if (expected != ir_rows.end())
          message << ", expected_elements=" << expected->second->elements
                  << ", storage=" << int(row.storage)
                  << ", expected_storage=" << int(expected->second->logical_storage)
                  << ", alignment=" << row.alignment
                  << ", expected_alignment=" << expected->second->alignment;
        message << ')';
        throw std::invalid_argument(message.str());
      }
    }
  }
  for (const MaterialIRTopologyRow &row : input.topology) {
    if (row.key.chunk < 0 || row.element_type != ElementType::realnum_value ||
        row.logical_storage != native_precision || !row.elements ||
        row.alignment != alignof(realnum))
      throw std::invalid_argument("material topology row is malformed");
    for (int axis = 0; axis < 3; ++axis)
      if (row.extents[axis] <= 0 || row.strides[axis] < 0 ||
          (row.stagger[axis] != 0 && row.stagger[axis] != 1))
        throw std::invalid_argument("material topology Yee layout is malformed");
    const array_kind kind = static_cast<array_kind>(row.key.kind);
    if (kind != array_kind::chi1inv && kind != array_kind::conductivity &&
        kind != array_kind::condinv && kind != array_kind::chi2 && kind != array_kind::chi3 &&
        kind != array_kind::sigma && kind != array_kind::pml_sig && kind != array_kind::pml_kap &&
        kind != array_kind::pml_siginv)
      throw std::invalid_argument("material topology row has an invalid kind");
    const size_t bytes = checked_host_bytes(row.element_type, row.elements);
    if (bytes > std::numeric_limits<size_t>::max() - total_bytes)
      throw std::overflow_error("material recipe total byte count overflow");
    total_bytes += bytes;
    if (!keys.insert(row.key).second)
      throw std::invalid_argument("material topology contains a duplicate storage key");
    if (input.ir) {
      const std::map<StorageKey, const MaterialIRTopologyRow *, StorageKeyLess>::const_iterator
          expected = ir_rows.find(row.key);
      if (expected == ir_rows.end() || !(row == *expected->second))
        throw std::invalid_argument("provisional material topology differs from immutable IR");
    }
  }
  if (input.ir) {
    std::set<StorageKey, StorageKeyLess> expected_keys;
    for (const MaterialIRTopologyRow &row : input.ir->topology) expected_keys.insert(row.key);
    if (keys != expected_keys)
      throw std::invalid_argument("material recipe does not exactly cover immutable IR topology");
  }
}

bool same_spec(const MaterialRecipeRow &row, const ArraySpec &spec) {
  return spec.role == row.role && spec.element_type == row.element_type &&
         spec.storage == row.storage && spec.elements == row.elements &&
         spec.alignment == row.alignment && !is_valid(spec.alias_of);
}

void validate_plan_shape(const StoragePlan &plan) {
  if (plan.arrays.size() != plan.keys.size())
    throw std::invalid_argument("material storage plan arrays/keys size mismatch");
  std::set<StorageKey, StorageKeyLess> keys;
  for (size_t i = 0; i < plan.arrays.size(); ++i)
    if (plan.arrays[i].id.value != i)
      throw std::invalid_argument("material storage plan has noncanonical ArrayIds");
    else if (!keys.insert(plan.keys[i]).second)
      throw std::invalid_argument("material storage plan has duplicate StorageKeys");
}

} // namespace

const char *material_recipe_disposition_name(MaterialRecipeDisposition disposition) {
  switch (disposition) {
    case MaterialRecipeDisposition::device_native: return "device-native";
    case MaterialRecipeDisposition::host_reference: return "host-reference";
    case MaterialRecipeDisposition::hybrid_interface: return "hybrid-interface";
    case MaterialRecipeDisposition::tiled_callback: return "tiled-callback";
  }
  return "?";
}

MaterialSupportDecision classify_material_support(const MaterialRecipe &recipe) {
  validate_material_recipe(recipe);
  return classify_support_input(recipe.ir(), recipe.rows());
}

bool MaterialRecipeRow::operator==(const MaterialRecipeRow &other) const {
  return key == other.key && role == other.role && element_type == other.element_type &&
         storage == other.storage && elements == other.elements && alignment == other.alignment &&
         values == other.values;
}

MaterialRecipeInput::MaterialRecipeInput()
    : disposition(MaterialRecipeDisposition::host_reference), eps_averaging(true),
      subpixel_tol(1e-4), subpixel_maxeval(100000), host_callback_id(invalid_array_value),
      from_host_callback(false), support_reason_bits(material_support_no_owned_ir) {}

MaterialRecipe::MaterialRecipe(const MaterialRecipeInput &input)
    : version_(material_recipe_format_version), disposition_(input.disposition),
      description_(input.description), eps_averaging_(input.eps_averaging),
      subpixel_tol_(input.subpixel_tol), subpixel_maxeval_(input.subpixel_maxeval),
      host_callback_id_(input.host_callback_id), from_host_callback_(input.from_host_callback),
      support_reason_bits_(input.support_reason_bits),
      rows_(input.rows), topology_(input.topology), ir_(input.ir), signature_(0) {
  validate_input(input);
  signature_ = compute_signature(version_, disposition_, description_, eps_averaging_,
                                 subpixel_tol_, subpixel_maxeval_, host_callback_id_,
                                 from_host_callback_, support_reason_bits_, rows_, topology_, ir_);
}

bool MaterialRecipe::operator==(const MaterialRecipe &other) const {
  return version_ == other.version_ && disposition_ == other.disposition_ &&
         description_ == other.description_ && eps_averaging_ == other.eps_averaging_ &&
         subpixel_tol_ == other.subpixel_tol_ && subpixel_maxeval_ == other.subpixel_maxeval_ &&
         host_callback_id_ == other.host_callback_id_ &&
         from_host_callback_ == other.from_host_callback_ &&
         support_reason_bits_ == other.support_reason_bits_ && rows_ == other.rows_ &&
         topology_ == other.topology_ &&
         ((!ir_ && !other.ir_) || (ir_ && other.ir_ && material_ir_equal(*ir_, *other.ir_))) &&
         signature_ == other.signature_;
}

void validate_material_recipe(const MaterialRecipe &recipe) {
  MaterialRecipeInput input;
  input.disposition = recipe.disposition();
  input.description = recipe.description();
  input.eps_averaging = recipe.eps_averaging();
  input.subpixel_tol = recipe.subpixel_tol();
  input.subpixel_maxeval = recipe.subpixel_maxeval();
  input.host_callback_id = recipe.host_callback_id();
  input.from_host_callback = recipe.from_host_callback();
  input.support_reason_bits = recipe.support_reason_bits();
  input.rows = recipe.rows();
  input.topology = recipe.topology();
  input.ir = recipe.ir();
  validate_input(input);
  const uint64_t signature =
      compute_signature(recipe.version(), input.disposition, input.description,
                        input.eps_averaging, input.subpixel_tol, input.subpixel_maxeval,
                        input.host_callback_id, input.from_host_callback,
                        input.support_reason_bits, input.rows,
                        input.topology, input.ir);
  if (recipe.version() != material_recipe_format_version || signature != recipe.signature())
    throw std::invalid_argument("material recipe has a stale or unsupported signature");
}

MaterialRecipe build_host_reference_material_recipe(const fields &f) {
  if (material_recipe_failure_rank_for_testing == my_rank()) {
    if (material_recipe_failure_mode_for_testing == 1)
      throw std::invalid_argument("injected material recipe validation failure");
    if (material_recipe_failure_mode_for_testing == 2) throw std::bad_alloc();
  }
  if (!f.storage_plan || !f.array_catalog)
    throw std::logic_error("material recipe requires a prepared storage catalog");
  validate_plan_shape(*f.storage_plan);

  MaterialRecipeInput input;
  input.description = "cpu:eager-host-reference";
  if (f.material_ir)
    input.ir = std::shared_ptr<const MaterialIR>(f.material_ir, material_ir_for(f));
  std::set<StorageKey, StorageKeyLess> present;
  for (size_t i = 0; i < f.storage_plan->arrays.size(); ++i) {
    const ArraySpec &spec = f.storage_plan->arrays[i];
    if (spec.role != array_role::material) continue;
    if (is_valid(spec.alias_of))
      throw std::logic_error("host-reference material recipe cannot own an aliased row");
    const void *source = f.array_catalog->resolve_untyped(spec.id);
    if (!source) throw std::logic_error("host-reference material recipe has a null source row");
    MaterialRecipeRow row;
    row.key = f.storage_plan->keys[i];
    row.role = spec.role;
    row.element_type = spec.element_type;
    row.storage = spec.storage;
    row.elements = spec.elements;
    row.alignment = spec.alignment;
    const size_t bytes = checked_host_bytes(spec.element_type, spec.elements);
    const unsigned char *begin = static_cast<const unsigned char *>(source);
    row.values.assign(begin, begin + bytes);
    input.rows.push_back(row);
    present.insert(row.key);
  }
  const MaterialIR *ir = material_ir_for(f);
  if (ir) {
    validate_material_ir(*ir);
    input.eps_averaging = ir->eps_averaging;
    input.subpixel_tol = ir->subpixel_tol;
    input.subpixel_maxeval = ir->subpixel_maxeval;
    for (const MaterialIRTopologyRow &row : ir->topology)
      if (!present.count(row.key)) input.topology.push_back(row), present.insert(row.key);
  }
  const MaterialSupportDecision decision = classify_support_input(input.ir, input.rows);
  input.disposition = decision.disposition;
  input.support_reason_bits = decision.reason_bits;
  input.description = decision.disposition == MaterialRecipeDisposition::device_native
                          ? "owned-ir:device-native-ready"
                          : "cpu:eager-host-reference";
  return MaterialRecipe(input);
}

void mark_material_storage_provisional(const MaterialRecipe &recipe, StoragePlan &plan) {
  validate_material_recipe(recipe);
  validate_plan_shape(plan);
  StoragePlan candidate = plan;
  size_t recipe_index = 0;
  for (size_t i = 0; i < candidate.arrays.size(); ++i) {
    if (candidate.arrays[i].role == array_role::material) {
      if (recipe_index >= recipe.rows().size() ||
          !(candidate.keys[i] == recipe.rows()[recipe_index].key) ||
          !same_spec(recipe.rows()[recipe_index], candidate.arrays[i]))
        throw std::invalid_argument(
            "material recipe order or row shape disagrees with provisional storage");
      candidate.arrays[i].classification_provisional = true;
      candidate.arrays[i].classification_elided = false;
      ++recipe_index;
    }
    else if (candidate.arrays[i].classification_provisional)
      throw std::invalid_argument("non-material storage is classification-provisional");
  }
  if (recipe_index != recipe.rows().size())
    throw std::invalid_argument("material recipe contains a row absent from storage");
  std::set<StorageKey, StorageKeyLess> keys(candidate.keys.begin(), candidate.keys.end());
  for (const MaterialIRTopologyRow &row : recipe.topology()) {
    if (keys.count(row.key)) continue;
    if (candidate.arrays.size() >= invalid_array_value)
      throw std::overflow_error("provisional material ArrayId overflow");
    ArraySpec spec;
    spec.id = ArrayId{uint32_t(candidate.arrays.size())};
    spec.role = array_role::material; spec.element_type = row.element_type;
    spec.storage = row.logical_storage; spec.elements = row.elements; spec.alignment = row.alignment;
    spec.alias_of = invalid_array(); spec.classification_provisional = true;
    spec.classification_elided = false;
    candidate.arrays.push_back(spec); candidate.keys.push_back(row.key); keys.insert(row.key);
  }
  plan = candidate;
}

void resolve_material_storage(const MaterialRecipe &recipe,
                              const MaterialClassification &classification,
                              const StoragePlan &authoritative, StoragePlan &provisional,
                              const PrecisionPolicy &policy) {
  validate_material_recipe(recipe);
  validate_plan_shape(authoritative);
  validate_plan_shape(provisional);
  if (recipe.disposition() != MaterialRecipeDisposition::host_reference &&
      recipe.disposition() != MaterialRecipeDisposition::device_native)
    throw std::invalid_argument("material storage resolver received an unsupported disposition");
  if (authoritative.arrays.size() > provisional.arrays.size())
    throw std::invalid_argument("provisional material storage lost the authoritative prefix");
  StoragePlan candidate = provisional;
  StoragePlan expected_device = authoritative;
  apply_precision_policy(expected_device, policy);
  if (classification.provisional_row_state.size() != candidate.arrays.size())
    throw std::invalid_argument("classification does not cover every candidate ArrayId");
  std::set<uint32_t> elided;
  uint32_t previous_elided = 0;
  bool have_previous_elided = false;
  for (ArrayId id : classification.elided) {
    if (!is_valid(id) || id.value >= candidate.arrays.size() ||
        id.value < authoritative.arrays.size() ||
        candidate.arrays[id.value].role != array_role::material ||
        classification.provisional_row_state[id.value] !=
            MaterialClassification::elided_row)
      throw std::invalid_argument("classification elided a non-provisional material row");
    if (have_previous_elided && id.value <= previous_elided)
      throw std::invalid_argument("classification elided rows are not sorted and unique");
    previous_elided = id.value;
    have_previous_elided = true;
    if (!elided.insert(id.value).second)
      throw std::invalid_argument("classification contains a duplicate elided row");
  }
  size_t recipe_index = 0;
  for (size_t i = 0; i < authoritative.arrays.size(); ++i) {
    const ArraySpec &expected = authoritative.arrays[i];
    const ArraySpec &actual = candidate.arrays[i];
    const bool was_provisional = actual.classification_provisional;
    if (!(authoritative.keys[i] == candidate.keys[i]) || expected.id != actual.id ||
        expected.role != actual.role || expected.element_type != actual.element_type ||
        expected.elements != actual.elements ||
        expected.alignment != actual.alignment || expected.alias_of != actual.alias_of)
      throw std::invalid_argument("provisional material storage changed before classification");
    if (actual.storage != expected_device.arrays[i].storage)
      throw std::invalid_argument("provisional material storage violates its precision policy");
    if (expected.classification_provisional)
      throw std::invalid_argument("authoritative material storage is still provisional");
    if (actual.role == array_role::material) {
      if (recipe_index >= recipe.rows().size() ||
          !(authoritative.keys[i] == recipe.rows()[recipe_index].key) ||
          !same_spec(recipe.rows()[recipe_index], expected))
        throw std::invalid_argument("authoritative material row disagrees with the recipe");
      if (actual.classification_elided)
        throw std::invalid_argument("authoritative material row is classification-elided");
      candidate.arrays[i].classification_provisional = false;
      candidate.arrays[i].classification_elided = false;
      ++recipe_index;
    }
    else if (actual.classification_provisional != expected.classification_provisional ||
             actual.classification_elided != expected.classification_elided)
      throw std::invalid_argument("classification changed non-material storage");
    const uint8_t state = classification.provisional_row_state[i];
    const bool valid_state =
        actual.role == array_role::material
            ? (state == MaterialClassification::retained ||
               (!was_provisional && state == MaterialClassification::not_provisional))
            : state == MaterialClassification::not_provisional;
    if (!valid_state)
      throw std::invalid_argument(
          "classification status disagrees with authoritative prefix ArrayId " +
          std::to_string(i) + " (observed " +
          std::to_string(state) + ")");
  }
  if (recipe_index != recipe.rows().size())
    throw std::invalid_argument("material recipe contains a row absent from authoritative storage");
  std::set<StorageKey, StorageKeyLess> topology;
  for (const MaterialIRTopologyRow &row : recipe.topology()) topology.insert(row.key);
  for (size_t i = authoritative.arrays.size(); i < candidate.arrays.size(); ++i) {
    ArraySpec &actual = candidate.arrays[i];
    const uint8_t state = classification.provisional_row_state[i];
    const bool was_provisional = actual.classification_provisional;
    if (actual.id.value != i || actual.role != array_role::material ||
        actual.element_type != ElementType::realnum_value || actual.storage != policy.material ||
        is_valid(actual.alias_of) ||
        !topology.count(candidate.keys[i]))
      throw std::invalid_argument("provisional material suffix is not recipe-derived");
    if (was_provisional) {
      if (state != MaterialClassification::retained &&
          state != MaterialClassification::elided_row)
        throw std::invalid_argument("classification omitted a provisional material row");
      if ((state == MaterialClassification::elided_row) !=
          (elided.count(uint32_t(i)) != 0))
        throw std::invalid_argument("classification row status and elision list disagree");
      actual.classification_provisional = false;
      actual.classification_elided = state == MaterialClassification::elided_row;
    }
    else if (state != MaterialClassification::not_provisional &&
             (state != MaterialClassification::retained || actual.classification_elided) &&
             (state != MaterialClassification::elided_row || !actual.classification_elided))
      throw std::invalid_argument("resolved material row classification changed");
  }
  provisional = candidate;
}

void resolve_material_storage(const MaterialRecipe &recipe,
                              const MaterialClassification &classification,
                              const StoragePlan &authoritative, StoragePlan &provisional) {
  resolve_material_storage(recipe, classification, authoritative, provisional,
                           precision_native());
}

bool has_provisional_material_storage(const StoragePlan &plan) {
  for (const ArraySpec &spec : plan.arrays)
    if (spec.role == array_role::material && spec.classification_provisional) return true;
  return false;
}

void set_material_recipe_failure_for_testing(int rank, int mode) {
  material_recipe_failure_rank_for_testing = rank;
  material_recipe_failure_mode_for_testing = mode;
}

} // namespace meep
