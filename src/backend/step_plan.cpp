/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "backend/descriptors.hpp"
#include "backend/step_plan.hpp"
#include "backend/halo_plan.hpp"
#include "backend/lifecycle.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

namespace meep {

const char *op_kind_name(OpKind k) {
  switch (k) {
    case OpKind::restore_magnetic_fields: return "restore_magnetic_fields";
    case OpKind::phase_material: return "phase_material";
    case OpKind::update_material_coefficients: return "update_material_coefficients";
    case OpKind::evaluate_source_scalars: return "evaluate_source_scalars";
    case OpKind::update_db: return "update_db";
    case OpKind::update_eh: return "update_eh";
    case OpKind::update_polarization: return "update_polarization";
    case OpKind::apply_sources: return "apply_sources";
    case OpKind::zero_boundary: return "zero_boundary";
    case OpKind::pack_halo: return "pack_halo";
    case OpKind::transfer_halo: return "transfer_halo";
    case OpKind::exchange_local: return "exchange_local";
    case OpKind::unpack_halo: return "unpack_halo";
    case OpKind::update_flux_half: return "update_flux_half";
    case OpKind::update_flux: return "update_flux";
    case OpKind::increment_time: return "increment_time";
    case OpKind::update_dft: return "update_dft";
    case OpKind::synchronize_magnetic_fields: return "synchronize_magnetic_fields";
    case OpKind::finite_value_check: return "finite_value_check";
    case OpKind::reduction: return "reduction";
    case OpKind::host_callback: return "host_callback";
    case OpKind::pack_state: return "pack_state";
    case OpKind::unpack_state: return "unpack_state";
    case OpKind::num_kinds: break;
  }
  return "?";
}

static const char *ft_name(field_type ft) {
  switch (ft) {
    case E_stuff: return "E";
    case H_stuff: return "H";
    case D_stuff: return "D";
    case B_stuff: return "B";
    case PE_stuff: return "PE";
    case PH_stuff: return "PH";
    case WE_stuff: return "WE";
    case WH_stuff: return "WH";
    default: return "?";
  }
}

namespace {

void target_fingerprint_mix(uint64_t &sig, uint64_t value) {
  sig ^= value + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
}

uint64_t material_phase_target_signature(const fields &f) {
  if (f.phasein_time <= 0) return 0;
  uint64_t sig = 0xcbf29ce484222325ull;
  target_fingerprint_mix(sig, uint64_t(f.num_chunks));
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    const fields_chunk &fc = *f.chunks[chunk];
    const bool owned = fc.is_mine();
    const structure_chunk *target = fc.new_s;
    target_fingerprint_mix(sig, uint64_t(chunk));
    target_fingerprint_mix(sig, uint64_t(owned));
    target_fingerprint_mix(sig, uint64_t(target != NULL));
    if (!owned || !target) continue;
    target_fingerprint_mix(sig, uint64_t(target->gv.dim));
    target_fingerprint_mix(sig, uint64_t(target->gv.ntot()));
    for (int axis = 0; axis < 3; ++axis) {
      target_fingerprint_mix(sig, uint64_t(target->gv.yucky_direction(axis)));
      target_fingerprint_mix(sig, uint64_t(target->gv.little_corner().yucky_val(axis)));
      target_fingerprint_mix(sig, uint64_t(target->gv.big_corner().yucky_val(axis)));
    }
    FOR_COMPONENTS(c) for (int d = 0; d < 5; ++d) {
      const bool has_chi = target->chi1inv[c][d] != NULL;
      target_fingerprint_mix(sig, uint64_t(c));
      target_fingerprint_mix(sig, uint64_t(d));
      target_fingerprint_mix(sig, uint64_t(has_chi));
      target_fingerprint_mix(sig, uint64_t(has_chi && target->trivial_chi1inv[c][d]));
      target_fingerprint_mix(sig, uint64_t(target->conductivity[c][d] != NULL));
    }
  }
  return sig;
}

ArrayId find_array(fields &f, int chunk, array_kind kind, int c, int cmp, int aux) {
  if (!f.array_catalog) return invalid_array();
  return f.array_catalog->find(StorageKey{chunk, int(kind), c, cmp, aux});
}

UpdateRegion make_region(const grid_volume &gv, int chunk, component c, int cmp, const ivec &begin,
                         const ivec &end) {
  UpdateRegion r;
  r.chunk = chunk;
  r.c = c;
  r.cmp = cmp;
  r.begin = begin;
  r.end = end;
  r.base = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const direction d = gv.yucky_direction(axis);
    r.counts[axis] = size_t((end.yucky_val(axis) - begin.yucky_val(axis)) / 2 + 1);
    r.strides[axis] = gv.stride(d);
    r.base += size_t((begin.yucky_val(axis) - gv.little_corner().yucky_val(axis)) / 2) *
              size_t(r.strides[axis]);
  }
  r.variant_key = 0;
  return r;
}

PmlProfile no_pml_profile() {
  PmlProfile p;
  p.sig = p.kap = p.siginv = invalid_array();
  p.base = 0;
  p.strides[0] = p.strides[1] = p.strides[2] = 0;
  return p;
}

PmlProfile make_pml_profile(fields &f, const fields_chunk &fc, int chunk, direction d,
                            const ivec &begin) {
  if (d == NO_DIRECTION) return no_pml_profile();
  PmlProfile p;
  p.sig = find_array(f, chunk, array_kind::pml_sig, -1, -1, int(d));
  p.kap = find_array(f, chunk, array_kind::pml_kap, -1, -1, int(d));
  p.siginv = find_array(f, chunk, array_kind::pml_siginv, -1, -1, int(d));
  p.base = begin.in_direction(d) - fc.gv.little_corner().in_direction(d);
  for (int axis = 0; axis < 3; ++axis)
    p.strides[axis] = fc.gv.yucky_direction(axis) == d ? 2 : 0;
  return p;
}

void add_access(fields &f, Operation &op, ArrayId id, AccessMode mode) {
  if (!is_valid(id) || !f.array_catalog || id.value >= f.array_catalog->size()) return;
  for (size_t i = 0; i < op.accesses.size(); ++i) {
    BufferAccess &existing = op.accesses[i];
    if (existing.array.id != id) continue;
    if (existing.mode != mode) existing.mode = AccessMode::read_write;
    return;
  }
  const ArraySpec &spec = f.array_catalog->spec(id);
  op.accesses.push_back(BufferAccess{ArrayRef{id, 0, spec.elements}, mode});
}

namespace {

bool same_polarization_group(const PolarizationUpdate &a, const PolarizationUpdate &b) {
  return a.region.chunk == b.region.chunk && a.ft == b.ft && a.state_index == b.state_index;
}

bool same_polarization_region(const UpdateRegion &a, const UpdateRegion &b) {
  if (a.chunk != b.chunk || a.c != b.c || a.cmp != b.cmp || !(a.begin == b.begin) ||
      !(a.end == b.end) || a.base != b.base)
    return false;
  for (int axis = 0; axis < 3; ++axis)
    if (a.counts[axis] != b.counts[axis] || a.strides[axis] != b.strides[axis]) return false;
  return true;
}

bool ordered_polarization_rows(const std::vector<PolarizationUpdate> &rows) {
  for (size_t i = 1; i < rows.size(); ++i) {
    const PolarizationUpdate &previous = rows[i - 1];
    const PolarizationUpdate &current = rows[i];
    if (int(previous.region.c) > int(current.region.c) ||
        (previous.region.c == current.region.c && previous.region.cmp >= current.region.cmp))
      return false;
  }
  return true;
}

bool canonical_noisy_add(const PolarizationUpdate &d) {
  if (d.kind != PolarizationUpdateKind::noisy_add || !is_valid(d.p) ||
      !is_valid(d.diagonal_sigma) || is_valid(d.p_prev) || is_valid(d.p_cross1) ||
      is_valid(d.p_prev_cross1) || is_valid(d.p_cross2) || is_valid(d.p_prev_cross2) ||
      is_valid(d.primary_w) || is_valid(d.cross_w1) || is_valid(d.cross_w2) ||
      is_valid(d.offdiagonal_sigma1) || is_valid(d.offdiagonal_sigma2) ||
      d.primary_stride != 0 || d.cross_stride1 != 0 || d.cross_stride2 != 0 || d.alpha != 0.0 ||
      d.gyro_model != GYROTROPIC_LORENTZIAN || d.region.variant_key != 0 ||
      d.noise_algorithm_version == 0)
    return false;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (d.gyro_tensor[i][j] != 0.0) return false;
  return true;
}

void add_polarization_recurrence_accesses(fields &f, Operation &op,
                                          const PolarizationUpdate &update) {
  add_access(f, op, update.p, AccessMode::read_write);
  add_access(f, op, update.p_prev, AccessMode::read_write);
  add_access(f, op, update.p_cross1, AccessMode::read_write);
  add_access(f, op, update.p_prev_cross1, AccessMode::read_write);
  add_access(f, op, update.p_cross2, AccessMode::read_write);
  add_access(f, op, update.p_prev_cross2, AccessMode::read_write);
  add_access(f, op, update.primary_w, AccessMode::read);
  add_access(f, op, update.cross_w1, AccessMode::read);
  add_access(f, op, update.cross_w2, AccessMode::read);
  add_access(f, op, update.diagonal_sigma, AccessMode::read);
  add_access(f, op, update.offdiagonal_sigma1, AccessMode::read);
  add_access(f, op, update.offdiagonal_sigma2, AccessMode::read);
}

} // namespace

void append_polarization_update_group_impl(fields &f, StepPlan &plan, Operation &op,
                                           const std::vector<PolarizationUpdate> &recurrences,
                                           const std::vector<PolarizationUpdate> &noise_additions) {
  if (op.kind != OpKind::update_polarization)
    throw std::invalid_argument("polarization group requires an update_polarization operation");
  if (op.descriptor_index > plan.polarization_updates.size() ||
      op.descriptor_count != plan.polarization_updates.size() - op.descriptor_index)
    throw std::invalid_argument("polarization group is not appended to its operation span");
  if (recurrences.empty() && noise_additions.empty()) return;

  const PolarizationUpdate &identity =
      recurrences.empty() ? noise_additions.front() : recurrences.front();
  if (identity.ft != op.ft)
    throw std::invalid_argument("polarization group field family differs from its operation");
  for (const PolarizationUpdate &previous : plan.polarization_updates)
    if (same_polarization_group(identity, previous))
      throw std::invalid_argument("polarization group identity is not contiguous");
  PolarizationUpdateKind recurrence_kind = PolarizationUpdateKind::lorentzian;
  if (!recurrences.empty()) recurrence_kind = recurrences.front().kind;
  for (const PolarizationUpdate &update : recurrences) {
    if ((update.kind != PolarizationUpdateKind::lorentzian &&
         update.kind != PolarizationUpdateKind::gyrotropic) ||
        update.kind != recurrence_kind ||
        update.noise_amplitude != 0.0 || update.noise_algorithm_version != 0 ||
        !same_polarization_group(identity, update))
      throw std::invalid_argument("malformed polarization recurrence group");
  }
  if (!noise_additions.empty() && !recurrences.empty() &&
      recurrence_kind != PolarizationUpdateKind::lorentzian)
    throw std::invalid_argument("noise additions require a Lorentz-family recurrence");
  for (const PolarizationUpdate &update : noise_additions) {
    if (!canonical_noisy_add(update) || !same_polarization_group(identity, update))
      throw std::invalid_argument("malformed noisy polarization addition group");
  }
  if (!noise_additions.empty()) {
    const PolarizationUpdate &coefficients = noise_additions.front();
    for (const PolarizationUpdate &recurrence : recurrences) {
      bool matched = false;
      for (const PolarizationUpdate &noise : noise_additions)
        matched = matched || (recurrence.region.c == noise.region.c &&
                              recurrence.region.cmp == noise.region.cmp);
      if (!matched)
        throw std::invalid_argument("noisy polarization recurrence is missing its noise row");
    }
    for (const PolarizationUpdate &noise : noise_additions) {
      if (noise.omega_0 != coefficients.omega_0 || noise.gamma != coefficients.gamma ||
          noise.dt != coefficients.dt || noise.noise_amplitude != coefficients.noise_amplitude ||
          noise.noise_algorithm_version != coefficients.noise_algorithm_version)
        throw std::invalid_argument("noisy polarization group has inconsistent coefficients");
      for (const PolarizationUpdate &recurrence : recurrences) {
        if (recurrence.region.c != noise.region.c || recurrence.region.cmp != noise.region.cmp)
          continue;
        if (!same_polarization_region(recurrence.region, noise.region) || recurrence.p != noise.p ||
            recurrence.diagonal_sigma != noise.diagonal_sigma ||
            recurrence.omega_0 != noise.omega_0 || recurrence.gamma != noise.gamma ||
            recurrence.dt != noise.dt)
          throw std::invalid_argument("noisy addition differs from its recurrence row");
      }
    }
  }
  if (!ordered_polarization_rows(recurrences) || !ordered_polarization_rows(noise_additions))
    throw std::invalid_argument("polarization group rows are not in component/cmp order");

  for (const PolarizationUpdate &update : recurrences) {
    plan.polarization_updates.push_back(update);
    add_polarization_recurrence_accesses(f, op, update);
  }
  for (const PolarizationUpdate &update : noise_additions) {
    plan.polarization_updates.push_back(update);
    add_access(f, op, update.p, AccessMode::read_write);
    add_access(f, op, update.diagonal_sigma, AccessMode::read);
  }
  op.descriptor_count = uint32_t(plan.polarization_updates.size()) - op.descriptor_index;
}

bool polarization_updates_equal(const PolarizationUpdate &a, const PolarizationUpdate &b) {
  if (a.kind != b.kind || a.region.chunk != b.region.chunk || a.region.c != b.region.c ||
      a.region.cmp != b.region.cmp || !(a.region.begin == b.region.begin) ||
      !(a.region.end == b.region.end) || a.region.base != b.region.base ||
      a.region.variant_key != b.region.variant_key || a.ft != b.ft ||
      a.state_index != b.state_index || a.p != b.p || a.p_prev != b.p_prev ||
      a.p_cross1 != b.p_cross1 || a.p_prev_cross1 != b.p_prev_cross1 ||
      a.p_cross2 != b.p_cross2 || a.p_prev_cross2 != b.p_prev_cross2 ||
      a.primary_w != b.primary_w || a.cross_w1 != b.cross_w1 || a.cross_w2 != b.cross_w2 ||
      a.diagonal_sigma != b.diagonal_sigma || a.offdiagonal_sigma1 != b.offdiagonal_sigma1 ||
      a.offdiagonal_sigma2 != b.offdiagonal_sigma2 ||
      a.primary_stride != b.primary_stride || a.cross_stride1 != b.cross_stride1 ||
      a.cross_stride2 != b.cross_stride2 || a.omega_0 != b.omega_0 || a.gamma != b.gamma ||
      a.alpha != b.alpha || a.gyro_model != b.gyro_model || a.dt != b.dt ||
      a.noise_amplitude != b.noise_amplitude ||
      a.noise_algorithm_version != b.noise_algorithm_version)
    return false;
  for (int axis = 0; axis < 3; ++axis) {
    if (a.region.counts[axis] != b.region.counts[axis] ||
        a.region.strides[axis] != b.region.strides[axis])
      return false;
    for (int cross = 0; cross < 3; ++cross)
      if (a.gyro_tensor[axis][cross] != b.gyro_tensor[axis][cross]) return false;
  }
  return true;
}

size_t checked_product(size_t a, size_t b, const char *what) {
  if (a && b > std::numeric_limits<size_t>::max() / a) throw std::overflow_error(what);
  return a * b;
}

size_t checked_sum(size_t a, size_t b, const char *what) {
  if (b > std::numeric_limits<size_t>::max() - a) throw std::overflow_error(what);
  return a + b;
}

ArrayId canonical_array(const CpuArrayCatalog &catalog, ArrayId id) {
  if (!is_valid(id) || id.value >= catalog.size())
    throw std::invalid_argument("solve_cw state row names an invalid array");
  for (size_t depth = 0; depth <= catalog.size(); ++depth) {
    const ArrayId next = catalog.spec(id).alias_of;
    if (!is_valid(next)) return id;
    if (next.value >= catalog.size())
      throw std::invalid_argument("solve_cw state row names an invalid alias");
    if (catalog.resolve_untyped(id) != catalog.resolve_untyped(next))
      throw std::invalid_argument("solve_cw state row contains a stale alias binding");
    id = next;
  }
  throw std::invalid_argument("solve_cw state row contains an alias cycle");
}

bool same_region(const UpdateRegion &a, const UpdateRegion &b) {
  if (a.chunk != b.chunk || a.c != b.c || a.cmp != b.cmp || a.base != b.base ||
      a.variant_key != b.variant_key)
    return false;
  for (int axis = 0; axis < 3; ++axis)
    if (a.begin.yucky_val(axis) != b.begin.yucky_val(axis) ||
        a.end.yucky_val(axis) != b.end.yucky_val(axis) || a.counts[axis] != b.counts[axis] ||
        a.strides[axis] != b.strides[axis])
      return false;
  return true;
}

bool same_array_ref(const ArrayRef &a, const ArrayRef &b) {
  return a.id == b.id && a.offset == b.offset && a.elements == b.elements;
}

bool same_access(const BufferAccess &a, const BufferAccess &b) {
  return same_array_ref(a.array, b.array) && a.mode == b.mode;
}

void hash_array_spec(uint64_t &sig, const StorageKey &key, const ArraySpec &spec) {
  target_fingerprint_mix(sig, uint64_t(key.chunk));
  target_fingerprint_mix(sig, uint64_t(key.kind));
  target_fingerprint_mix(sig, uint64_t(key.component_));
  target_fingerprint_mix(sig, uint64_t(key.cmp));
  target_fingerprint_mix(sig, uint64_t(key.aux));
  target_fingerprint_mix(sig, uint64_t(spec.id.value));
  target_fingerprint_mix(sig, uint64_t(spec.role));
  target_fingerprint_mix(sig, uint64_t(spec.element_type));
  target_fingerprint_mix(sig, uint64_t(spec.storage));
  target_fingerprint_mix(sig, uint64_t(spec.elements));
  target_fingerprint_mix(sig, uint64_t(spec.alignment));
  target_fingerprint_mix(sig, uint64_t(spec.alias_of.value));
  target_fingerprint_mix(sig, uint64_t(spec.classification_provisional));
}

uint64_t storage_fingerprint(const fields &f, const CwStateLayout &layout) {
  uint64_t sig = 0xcbf29ce484222325ull;
  if (!f.array_catalog) return sig;
  std::vector<ArrayId> ids;
  for (const CwStateRow &row : layout.rows) {
    ids.push_back(row.real_array);
    ids.push_back(row.imag_array);
  }
  for (const ArrayRef &ref : layout.zero_arrays)
    ids.push_back(ref.id);
  std::sort(ids.begin(), ids.end(), [](ArrayId a, ArrayId b) { return a.value < b.value; });
  ids.erase(std::unique(ids.begin(), ids.end(), [](ArrayId a, ArrayId b) { return a == b; }),
            ids.end());
  target_fingerprint_mix(sig, uint64_t(ids.size()));
  for (ArrayId id : ids)
    hash_array_spec(sig, f.array_catalog->key(id), f.array_catalog->spec(id));
  return sig;
}

void fingerprint_double(uint64_t &sig, double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double is not 64-bit");
  memcpy(&bits, &value, sizeof(bits));
  target_fingerprint_mix(sig, bits);
}

uint64_t coordinate_fingerprint(const fields &f) {
  uint64_t sig = 0xcbf29ce484222325ull;
  target_fingerprint_mix(sig, uint64_t(f.num_chunks));
  fingerprint_double(sig, f.beta);
  fingerprint_double(sig, f.m);
  for (double k : f.bfast_scaled_k)
    fingerprint_double(sig, k);
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    const fields_chunk &fc = *f.chunks[chunk];
    target_fingerprint_mix(sig, uint64_t(chunk));
    target_fingerprint_mix(sig, uint64_t(fc.is_mine()));
    target_fingerprint_mix(sig, uint64_t(fc.gv.dim));
    target_fingerprint_mix(sig, uint64_t(fc.gv.ntot()));
    for (int axis = 0; axis < 3; ++axis) {
      target_fingerprint_mix(sig, uint64_t(fc.gv.yucky_direction(axis)));
      target_fingerprint_mix(sig, uint64_t(fc.gv.little_corner().yucky_val(axis)));
      target_fingerprint_mix(sig, uint64_t(fc.gv.big_corner().yucky_val(axis)));
      target_fingerprint_mix(sig,
                             uint64_t(fc.gv.stride(fc.gv.yucky_direction(axis))));
    }
    fingerprint_double(sig, fc.gv.origin_r());
    target_fingerprint_mix(sig, uint64_t(fc.zero_fields_near_cylorigin));
  }
  return sig;
}

uint64_t material_fingerprint(const fields &f) {
  uint64_t sig = 0xcbf29ce484222325ull;
  target_fingerprint_mix(sig, material_phase_target_signature(f));
  return sig;
}

uint64_t cw_layout_signature(const CwStateLayout &layout) {
  uint64_t sig = 0xcbf29ce484222325ull;
  for (const CwStateRow &row : layout.rows) {
    target_fingerprint_mix(sig, uint64_t(row.chunk));
    target_fingerprint_mix(sig, uint64_t(row.traversal_component));
    target_fingerprint_mix(sig, uint64_t(row.storage_component));
    target_fingerprint_mix(sig, uint64_t(row.family));
    target_fingerprint_mix(sig, uint64_t(row.real_array.value));
    target_fingerprint_mix(sig, uint64_t(row.imag_array.value));
    target_fingerprint_mix(sig, uint64_t(row.owned_region.chunk));
    target_fingerprint_mix(sig, uint64_t(row.owned_region.c));
    target_fingerprint_mix(sig, uint64_t(row.owned_region.cmp));
    for (int axis = 0; axis < 3; ++axis) {
      target_fingerprint_mix(sig, uint64_t(row.owned_region.begin.yucky_val(axis)));
      target_fingerprint_mix(sig, uint64_t(row.owned_region.end.yucky_val(axis)));
    }
    target_fingerprint_mix(sig, uint64_t(row.owned_region.base));
    for (int axis = 0; axis < 3; ++axis) {
      target_fingerprint_mix(sig, uint64_t(row.owned_region.counts[axis]));
      target_fingerprint_mix(sig, uint64_t(row.owned_region.strides[axis]));
    }
    target_fingerprint_mix(sig, uint64_t(row.owned_region.variant_key));
    target_fingerprint_mix(sig, uint64_t(row.complex_offset));
    target_fingerprint_mix(sig, uint64_t(row.complex_count));
  }
  for (const ArrayRef &ref : layout.zero_arrays) {
    target_fingerprint_mix(sig, uint64_t(ref.id.value));
    target_fingerprint_mix(sig, uint64_t(ref.offset));
    target_fingerprint_mix(sig, uint64_t(ref.elements));
  }
  for (const BufferAccess &access : layout.pack_accesses) {
    target_fingerprint_mix(sig, uint64_t(access.array.id.value));
    target_fingerprint_mix(sig, uint64_t(access.array.offset));
    target_fingerprint_mix(sig, uint64_t(access.array.elements));
    target_fingerprint_mix(sig, uint64_t(access.mode));
  }
  for (const BufferAccess &access : layout.unpack_accesses) {
    target_fingerprint_mix(sig, uint64_t(access.array.id.value));
    target_fingerprint_mix(sig, uint64_t(access.array.offset));
    target_fingerprint_mix(sig, uint64_t(access.array.elements));
    target_fingerprint_mix(sig, uint64_t(access.mode));
  }
  target_fingerprint_mix(sig, uint64_t(layout.unpack_prelude.first_boundary));
  target_fingerprint_mix(sig, uint64_t(layout.unpack_prelude.constitutive));
  target_fingerprint_mix(sig, uint64_t(layout.unpack_prelude.second_boundary));
  target_fingerprint_mix(sig, uint64_t(layout.unpack_prelude.skip_w_components));
  target_fingerprint_mix(sig, uint64_t(layout.unpack_prelude.invalidate_field_values));
  target_fingerprint_mix(sig, uint64_t(layout.complex_count));
  target_fingerprint_mix(sig, uint64_t(layout.real_count));
  target_fingerprint_mix(sig, uint64_t(layout.vector_precision));
  target_fingerprint_mix(sig, layout.storage_fingerprint);
  target_fingerprint_mix(sig, layout.coordinate_fingerprint);
  target_fingerprint_mix(sig, layout.material_fingerprint);
  return sig;
}

struct CurlSources {
  bool have_plus;
  bool have_minus;
  component plus_component;
  component minus_component;
  direction plus_direction;
  direction minus_direction;

  CurlSources()
      : have_plus(false), have_minus(false), plus_component(NO_COMPONENT),
        minus_component(NO_COMPONENT), plus_direction(NO_DIRECTION), minus_direction(NO_DIRECTION) {
  }
};

bool cross_is_negative(direction a, direction b) {
  if (a >= R) a = direction(a - 3);
  if (b >= R) b = direction(b - 3);
  return ((3 + b - a) % 3) == 2;
}

direction cross_direction(direction a, direction b) {
  if (a == b) meep::abort("bug - cross_direction expects different directions");
  const bool cylindrical = a >= R || b >= R;
  if (a >= R) a = direction(a - 3);
  if (b >= R) b = direction(b - 3);
  direction result = direction((3 + 2 * a - b) % 3);
  if (cylindrical && result < Z) result = direction(result + 3);
  return result;
}

CurlSources curl_sources_for(const fields_chunk &fc, component target) {
  CurlSources result;
  const direction target_direction = component_direction(target);
  FOR_COMPONENTS(source) {
    if (!((is_electric(target) && is_magnetic(source)) || (is_D(target) && is_magnetic(source)) ||
          (is_magnetic(target) && is_electric(source)) || (is_B(target) && is_electric(source))))
      continue;
    const direction source_direction = component_direction(source);
    if (target_direction == source_direction || !fc.gv.has_field(source) ||
        !fc.gv.has_field(target))
      continue;
    const direction derivative = cross_direction(target_direction, source_direction);
    if (!(has_direction(fc.gv.dim, derivative) ||
          (fc.gv.dim == Dcyl && has_field_direction(fc.gv.dim, derivative))))
      continue;
    if (cross_is_negative(source_direction, target_direction)) {
      result.have_minus = true;
      result.minus_component = source;
      result.minus_direction = derivative;
    }
    else {
      result.have_plus = true;
      result.plus_component = source;
      result.plus_direction = derivative;
    }
  }
  return result;
}

class StepPlanBuilder {
public:
  explicit StepPlanBuilder(fields &f, StepProgram program) : f_(f) {
    plan_.program = program;
    plan_.coordinate_generation = generation(f, MutationKind::coordinate_definition);
    plan_.beta = f.beta;
    plan_.cylindrical_m = f.m;
    plan_.bfast_scaled_k = f.bfast_scaled_k;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
      const fields_chunk &fc = *f.chunks[chunk];
      plan_.cylindrical_origin_r.push_back(fc.gv.origin_r());
      plan_.cylindrical_zero_near_origin.push_back(fc.zero_fields_near_cylorigin ? 1 : 0);
    }
    plan_.material_phase_target_signature = material_phase_target_signature(f);
    plan_.source_signature = f_.descriptors ? source_plan_signature(f_.descriptors->sources) : 0;
  }

  Operation &add(OpKind k, field_type ft = field_type(NUM_FIELD_TYPES), Guard g = guard_always(),
                 double src_offset = 0.0) {
    Operation op;
    op.kind = k;
    op.descriptor_index = 0;
    op.descriptor_count = 0;
    op.material_refresh_index = 0;
    op.material_refresh_count = 0;
    op.beta_descriptor_index = 0;
    op.beta_descriptor_count = 0;
    op.cylindrical_m_descriptor_index = 0;
    op.cylindrical_m_descriptor_count = 0;
    op.cylindrical_origin_action_index = 0;
    op.cylindrical_origin_action_count = 0;
    op.polarization_subtraction_index = 0;
    op.polarization_subtraction_count = 0;
    op.magnetic_state_index = 0;
    op.magnetic_state_count = 0;
    op.legacy_flux_index = 0;
    op.legacy_flux_count = 0;
    op.source_descriptor_index = 0;
    op.source_descriptor_count = 0;
    op.guard = g;
    op.ft = ft;
    op.source_time_offset = src_offset;
    plan_.operations.push_back(op);
    return plan_.operations.back();
  }

  void add_db(field_type ft);
  void add_eh(field_type ft, Guard guard = guard_always());
  void add_polarizations(field_type ft);
  void add_source_evaluation(Guard guard, double src_offset);
  void add_sources(field_type ft);
  void add_dfts();

  Operation &add_material_refresh(OpKind op_kind) {
    Operation &op = add(op_kind, field_type(NUM_FIELD_TYPES),
                        op_kind == OpKind::phase_material ? guard_static(true) : guard_always());
    op.material_refresh_index = uint32_t(plan_.material_refresh_arrays.size());
    if (f_.phasein_time <= 0) return op;

    const MaterialRefreshFamily first = op_kind == OpKind::phase_material
                                            ? MaterialRefreshFamily::chi1inv
                                            : MaterialRefreshFamily::conductivity;
    const MaterialRefreshFamily last = op_kind == OpKind::phase_material
                                           ? MaterialRefreshFamily::chi1inv
                                           : MaterialRefreshFamily::condinv;
    for (uint32_t family_value = uint32_t(first); family_value <= uint32_t(last);
         ++family_value) {
      const MaterialRefreshFamily family = MaterialRefreshFamily(family_value);
      const array_kind storage_kind = family == MaterialRefreshFamily::chi1inv
                                          ? array_kind::chi1inv
                                          : family == MaterialRefreshFamily::conductivity
                                                ? array_kind::conductivity
                                                : array_kind::condinv;
      for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
        if (!f_.chunks[chunk]->is_mine()) continue;
        const structure_chunk &sc = *f_.chunks[chunk]->s;
        FOR_COMPONENTS(c) for (int d = 0; d < 5; ++d) {
          if (family == MaterialRefreshFamily::condinv && d != int(component_direction(c)))
            continue;
          const realnum *row = family == MaterialRefreshFamily::chi1inv
                                   ? sc.chi1inv[c][d]
                                   : family == MaterialRefreshFamily::conductivity
                                         ? sc.conductivity[c][d]
                                         : sc.condinv[c][d];
          if (!row) continue;
          const ArrayId current = find_array(f_, chunk, storage_kind, int(c), -1, d);
          if (!is_valid(current)) {
            /* The CPU backend intentionally keeps material preparation lazy.
               A phase configured before its first step may therefore acquire
               current rows after the last catalog build; CPU arithmetic does
               not consume these upload descriptors. Resident backends freeze
               the complete union before plan construction and must fail
               closed if any such row is absent. */
            continue;
          }
          const ArraySpec &spec = f_.array_catalog->spec(current);
          plan_.material_refresh_arrays.push_back(MaterialRefreshArray{
              chunk, c, direction(d), family, current, spec.elements});
          add_access(f_, op, current, AccessMode::write);
        }
      }
    }
    op.material_refresh_count =
        uint32_t(plan_.material_refresh_arrays.size()) - op.material_refresh_index;
    return op;
  }

  uint32_t operation_count() const { return uint32_t(plan_.operations.size()); }

  Operation &add_magnetic_marker(OpKind kind, AccessMode mode) {
    Operation &op = add(kind, field_type(NUM_FIELD_TYPES), guard_variant(0));
    if (plan_.magnetic_state_arrays.empty()) build_magnetic_state_arrays();
    op.magnetic_state_index = 0;
    op.magnetic_state_count = uint32_t(plan_.magnetic_state_arrays.size());
    for (const MagneticStateArray &entry : plan_.magnetic_state_arrays)
      add_access(f_, op, entry.live,
                 mode == AccessMode::write || entry.average ? mode : AccessMode::read);
    if (kind == OpKind::synchronize_magnetic_fields) {
      const uint32_t schedule[] = {
          plan_.magnetic_half_step.evaluate_b_sources, plan_.magnetic_half_step.update_b,
          plan_.magnetic_half_step.apply_b_sources,    plan_.magnetic_half_step.transfer_b,
          plan_.magnetic_half_step.evaluate_h_sources, plan_.magnetic_half_step.update_h,
          plan_.magnetic_half_step.transfer_h};
      const size_t marker = plan_.operations.size() - 1;
      for (size_t i = 0; i < sizeof(schedule) / sizeof(schedule[0]); ++i) {
        if (schedule[i] == UINT32_MAX) continue;
        if (schedule[i] >= marker) meep::abort("invalid magnetic half-step operation index");
        const Operation &referenced = plan_.operations[schedule[i]];
        for (const BufferAccess &access : referenced.accesses)
          add_access(f_, op, access.array.id, access.mode);
      }
    }
    return op;
  }

  void set_magnetic_half_step(uint32_t evaluate_b_sources, uint32_t update_b,
                              uint32_t apply_b_sources, uint32_t transfer_b,
                              uint32_t evaluate_h_sources, uint32_t update_h, uint32_t transfer_h) {
    plan_.magnetic_half_step.evaluate_b_sources = evaluate_b_sources;
    plan_.magnetic_half_step.update_b = update_b;
    plan_.magnetic_half_step.apply_b_sources = apply_b_sources;
    plan_.magnetic_half_step.transfer_b = transfer_b;
    plan_.magnetic_half_step.evaluate_h_sources = evaluate_h_sources;
    plan_.magnetic_half_step.update_h = update_h;
    plan_.magnetic_half_step.transfer_h = transfer_h;
  }

  void add_finite_value_check() {
    Operation &op = add(OpKind::finite_value_check);
    if (!f_.storage_plan) return;

    /* Device diagnostics scan physical field arrays only. Preserve catalog
       order so the first failing access is deterministic, and skip aliases so
       one allocation cannot be attributed twice. StorageKey supplies the
       chunk/component/cmp identity used by device backends for diagnostics. */
    const StoragePlan &storage = *f_.storage_plan;
    for (size_t i = 0; i < storage.arrays.size(); ++i) {
      const ArraySpec &spec = storage.arrays[i];
      const StorageKey &key = storage.keys[i];
      if (key.kind != int(array_kind::f) || spec.element_type != ElementType::realnum_value ||
          !spec.elements || is_valid(spec.alias_of))
        continue;
      add_access(f_, op, spec.id, AccessMode::read);
    }
  }

  void set_cw_state_layout(const CwStateLayout &layout) { plan_.cw_state_layout = layout; }

  void add_cw_state_marker(OpKind kind) {
    if (plan_.cw_state_layout.rows.size() > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("solve_cw state row span overflow");
    Operation &op = add(kind);
    op.descriptor_index = 0;
    op.descriptor_count = uint32_t(plan_.cw_state_layout.rows.size());
    op.accesses = kind == OpKind::pack_state ? plan_.cw_state_layout.pack_accesses
                                             : plan_.cw_state_layout.unpack_accesses;
  }

  void add_if(bool present, OpKind k, field_type ft = field_type(NUM_FIELD_TYPES),
              double src_offset = 0.0) {
    if (present) add(k, ft, guard_static(true), src_offset);
  }

  void add_legacy_flux(bool present, OpKind kind) {
    if (!present) return;
    if (plan_.legacy_flux_updates.empty()) {
      uint32_t ordinal = 0;
      for (const flux_vol *flux = f_.fluxes; flux; flux = flux->next, ++ordinal)
        plan_.legacy_flux_updates.push_back(LegacyFluxUpdate{ordinal, 0, 0, 0});
    }
    Operation &op = add(kind, field_type(NUM_FIELD_TYPES), guard_static(true));
    op.legacy_flux_index = 0;
    op.legacy_flux_count = uint32_t(plan_.legacy_flux_updates.size());
  }

  /* One semantic boundary step.
   *
   * The plan asks for this to expand into zero-metal, pack, transfer and
   * unpack. On CPU it stays fused into a single transfer_halo, for two
   * concrete reasons in step_boundaries: comms_manager's lifetime spans all
   * three communication phases (its destructor is what completes the
   * outstanding requests), and the receives are posted *before* the metal
   * zeroing, so the plan's nominal order is not the implementation's order
   * either. Splitting it would change the request-completion point and the
   * timing scopes and buys nothing on CPU.
   *
   * The four OpKinds exist in the enum because Phase 2 needs the vocabulary;
   * see the deviation note in ~/meep-phase1-pr5.md. */
  void add_boundaries(field_type ft, Guard g = guard_always()) {
    add(OpKind::transfer_halo, ft, g);
  }

  StepPlan finish() {
    plan_.signature = signature_for(plan_);
    return plan_;
  }

  static uint64_t signature_for(const StepPlan &plan) {
    uint64_t sig = 0xcbf29ce484222325ull;
    mix(sig, uint64_t(plan.program));
    mix_double(sig, plan.beta);
    mix_double(sig, plan.cylindrical_m);
    for (double k : plan.bfast_scaled_k) mix_double(sig, k);
    for (double origin : plan.cylindrical_origin_r) mix_double(sig, origin);
    for (uint8_t zero : plan.cylindrical_zero_near_origin) mix(sig, uint64_t(zero));
    mix(sig, plan.source_signature);
    for (const Operation &op : plan.operations) {
      sig ^= uint64_t(op.kind) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.ft) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.guard.kind) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.guard.scalar_slot) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.guard.variant_index) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.descriptor_index) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.descriptor_count) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      mix(sig, uint64_t(op.material_refresh_index));
      mix(sig, uint64_t(op.material_refresh_count));
      sig ^= uint64_t(op.beta_descriptor_index) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.beta_descriptor_count) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      mix(sig, uint64_t(op.cylindrical_m_descriptor_index));
      mix(sig, uint64_t(op.cylindrical_m_descriptor_count));
      mix(sig, uint64_t(op.cylindrical_origin_action_index));
      mix(sig, uint64_t(op.cylindrical_origin_action_count));
      sig ^= uint64_t(op.polarization_subtraction_index) + 0x9e3779b97f4a7c15ull + (sig << 6) +
             (sig >> 2);
      sig ^= uint64_t(op.polarization_subtraction_count) + 0x9e3779b97f4a7c15ull + (sig << 6) +
             (sig >> 2);
      mix(sig, uint64_t(op.magnetic_state_index));
      mix(sig, uint64_t(op.magnetic_state_count));
      mix(sig, uint64_t(op.legacy_flux_index));
      mix(sig, uint64_t(op.legacy_flux_count));
      sig ^= uint64_t(op.source_descriptor_index) + 0x9e3779b97f4a7c15ull + (sig << 6) +
             (sig >> 2);
      sig ^= uint64_t(op.source_descriptor_count) + 0x9e3779b97f4a7c15ull + (sig << 6) +
             (sig >> 2);
      uint64_t source_bits = 0;
      static_assert(sizeof(source_bits) == sizeof(op.source_time_offset), "double is not 64-bit");
      memcpy(&source_bits, &op.source_time_offset, sizeof(source_bits));
      sig ^= source_bits + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      for (const BufferAccess &access : op.accesses) {
        sig ^= uint64_t(access.array.id.value) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
        sig ^= uint64_t(access.array.offset) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
        sig ^= uint64_t(access.array.elements) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
        sig ^= uint64_t(access.mode) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      }
    }
    for (const CurlUpdate &d : plan.db_updates)
      hash_curl(sig, d);
    for (const CylindricalRadialPrefix &d : plan.cylindrical_radial_prefixes)
      hash_cylindrical_radial_prefix(sig, d);
    for (const BfastUpdate &d : plan.bfast_updates)
      hash_bfast(sig, d);
    for (const BetaUpdate &d : plan.beta_updates)
      hash_beta(sig, d);
    for (const CylindricalMOverRUpdate &d : plan.cylindrical_m_updates)
      hash_cylindrical_m(sig, d);
    for (const CylindricalAxisUpdate &d : plan.cylindrical_axis_updates)
      hash_cylindrical_axis(sig, d);
    for (const SlabRef &d : plan.cylindrical_zero_slabs)
      hash_slab(sig, d);
    for (const CylindricalOriginAction &d : plan.cylindrical_origin_actions) {
      mix(sig, uint64_t(d.kind));
      mix(sig, uint64_t(d.index));
    }
    for (const ConstitutiveUpdate &d : plan.eh_updates)
      hash_constitutive(sig, d);
    for (const PolarizationUpdate &d : plan.polarization_updates)
      hash_polarization(sig, d);
    for (const PolarizationSubtraction &d : plan.polarization_subtractions)
      hash_polarization_subtraction(sig, d);
    for (const LegacyFluxUpdate &d : plan.legacy_flux_updates)
      hash_legacy_flux_update(sig, d);
    for (const LegacyFluxTerm &d : plan.legacy_flux_terms)
      hash_legacy_flux_term(sig, d);
    for (const MagneticStateArray &d : plan.magnetic_state_arrays)
      hash_magnetic_state(sig, d);
    for (const MaterialRefreshArray &d : plan.material_refresh_arrays)
      hash_material_refresh(sig, d);
    if (plan.program == StepProgram::solve_cw)
      mix(sig, cw_layout_signature(plan.cw_state_layout));
    hash_magnetic_half_step(sig, plan.magnetic_half_step);
    mix(sig, plan.material_phase_target_signature);
    for (const DftDescriptor &d : plan.dft_updates) hash_dft(sig, d);
    return sig;
  }

private:
  static void mix(uint64_t &sig, uint64_t value) {
    sig ^= value + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
  }
  static void mix_double(uint64_t &sig, double value) {
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    mix(sig, bits);
  }
  static void hash_region(uint64_t &sig, const UpdateRegion &r) {
    mix(sig, uint64_t(r.chunk));
    mix(sig, uint64_t(r.c));
    mix(sig, uint64_t(r.cmp));
    for (int i = 0; i < 3; ++i) {
      mix(sig, uint64_t(r.begin.yucky_val(i)));
      mix(sig, uint64_t(r.end.yucky_val(i)));
    }
    mix(sig, uint64_t(r.base));
    for (int i = 0; i < 3; ++i) {
      mix(sig, uint64_t(r.counts[i]));
      mix(sig, uint64_t(r.strides[i]));
    }
    mix(sig, uint64_t(r.variant_key));
  }
  static void hash_id(uint64_t &sig, ArrayId id) { mix(sig, uint64_t(id.value)); }
  static void hash_ref(uint64_t &sig, const ArrayRef &ref) {
    hash_id(sig, ref.id);
    mix(sig, uint64_t(ref.offset));
    mix(sig, uint64_t(ref.elements));
  }
  static void hash_ivec(uint64_t &sig, const ivec &v) {
    mix(sig, uint64_t(v.dim));
    LOOP_OVER_DIRECTIONS(v.dim, d) { mix(sig, uint64_t(v.in_direction(d))); }
  }
  static void hash_vec(uint64_t &sig, const vec &v) {
    mix(sig, uint64_t(v.dim));
    LOOP_OVER_DIRECTIONS(v.dim, d) { mix_double(sig, v.in_direction(d)); }
  }
  static void hash_pml(uint64_t &sig, const PmlProfile &p) {
    hash_id(sig, p.sig);
    hash_id(sig, p.kap);
    hash_id(sig, p.siginv);
    mix(sig, uint64_t(p.base));
    for (int i = 0; i < 3; ++i)
      mix(sig, uint64_t(p.strides[i]));
  }
  static void hash_curl(uint64_t &sig, const CurlUpdate &d) {
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.plus_source);
    hash_id(sig, d.minus_source);
    mix(sig, uint64_t(d.plus_stride));
    mix(sig, uint64_t(d.minus_stride));
    hash_id(sig, d.target_u);
    hash_id(sig, d.conductivity);
    hash_id(sig, d.condinv);
    hash_id(sig, d.target_cond);
    hash_pml(sig, d.pml);
    hash_pml(sig, d.pml_u);
    mix_double(sig, d.dtdx);
    mix_double(sig, d.dt);
    mix(sig, uint64_t(d.radial_prefix_index));
    mix(sig, uint64_t(d.bfast_update_index));
  }
  static void hash_cylindrical_radial_prefix(uint64_t &sig, const CylindricalRadialPrefix &d) {
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.target_component));
    mix(sig, uint64_t(d.source_component));
    mix(sig, uint64_t(d.cmp));
    hash_id(sig, d.source);
    hash_id(sig, d.scratch);
    mix(sig, uint64_t(d.nr));
    mix(sig, uint64_t(d.nz));
    mix(sig, uint64_t(d.row_stride));
    mix(sig, uint64_t(d.source_elements));
    mix(sig, uint64_t(d.scratch_elements));
    mix_double(sig, d.ir0);
  }
  static void hash_bfast(uint64_t &sig, const BfastUpdate &d) {
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.source1);
    hash_id(sig, d.source2);
    mix(sig, uint64_t(d.stride1));
    mix(sig, uint64_t(d.stride2));
    hash_id(sig, d.f_bfast);
    hash_id(sig, d.target_u);
    hash_id(sig, d.condinv);
    hash_id(sig, d.target_cond);
    hash_pml(sig, d.pml);
    hash_pml(sig, d.pml_u);
    mix_double(sig, d.k1);
    mix_double(sig, d.k2);
  }
  static void hash_beta(uint64_t &sig, const BetaUpdate &d) {
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.source);
    hash_id(sig, d.target_u);
    hash_id(sig, d.condinv);
    hash_id(sig, d.target_cond);
    hash_pml(sig, d.pml);
    hash_pml(sig, d.pml_u);
    mix_double(sig, d.betadt);
  }
  static void hash_cylindrical_m(uint64_t &sig, const CylindricalMOverRUpdate &d) {
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.source);
    hash_id(sig, d.target_u);
    hash_id(sig, d.condinv);
    hash_id(sig, d.target_cond);
    hash_pml(sig, d.pml);
    hash_pml(sig, d.pml_u);
    mix_double(sig, d.numerator);
    mix(sig, uint64_t(d.raw_radial_start));
  }
  static void hash_cylindrical_axis(uint64_t &sig, const CylindricalAxisUpdate &d) {
    mix(sig, uint64_t(d.kind));
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.source1);
    hash_id(sig, d.source2);
    mix(sig, uint64_t(d.source1_neighbor_offset));
    mix(sig, uint64_t(d.source2_offset));
    hash_id(sig, d.target_u);
    hash_id(sig, d.conductivity);
    hash_id(sig, d.condinv);
    hash_id(sig, d.target_cond);
    hash_pml(sig, d.pml);
    hash_pml(sig, d.pml_u);
    mix_double(sig, d.scale);
    mix_double(sig, d.source2_multiplier);
    mix_double(sig, d.dt);
  }
  static void hash_slab(uint64_t &sig, const SlabRef &d) {
    hash_id(sig, d.array);
    mix(sig, uint64_t(d.base));
    for (int i = 0; i < 3; ++i) {
      mix(sig, uint64_t(d.counts[i]));
      mix(sig, uint64_t(d.strides[i]));
    }
  }
  static void hash_constitutive(uint64_t &sig, const ConstitutiveUpdate &d) {
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.base_primary);
    hash_id(sig, d.base_cross1);
    hash_id(sig, d.base_cross2);
    hash_id(sig, d.primary);
    hash_id(sig, d.cross1);
    hash_id(sig, d.cross2);
    hash_id(sig, d.diagonal);
    hash_id(sig, d.offdiagonal1);
    hash_id(sig, d.offdiagonal2);
    mix(sig, uint64_t(d.primary_stride));
    mix(sig, uint64_t(d.cross1_stride));
    mix(sig, uint64_t(d.cross2_stride));
    hash_id(sig, d.chi2);
    hash_id(sig, d.chi3);
    hash_id(sig, d.target_w);
    hash_id(sig, d.previous_w);
    hash_pml(sig, d.pml);
  }
  static void hash_polarization(uint64_t &sig, const PolarizationUpdate &d) {
    mix(sig, uint64_t(d.kind));
    hash_region(sig, d.region);
    mix(sig, uint64_t(d.ft));
    mix(sig, uint64_t(d.state_index));
    hash_id(sig, d.p);
    hash_id(sig, d.p_prev);
    hash_id(sig, d.p_cross1);
    hash_id(sig, d.p_prev_cross1);
    hash_id(sig, d.p_cross2);
    hash_id(sig, d.p_prev_cross2);
    hash_id(sig, d.primary_w);
    hash_id(sig, d.cross_w1);
    hash_id(sig, d.cross_w2);
    hash_id(sig, d.diagonal_sigma);
    hash_id(sig, d.offdiagonal_sigma1);
    hash_id(sig, d.offdiagonal_sigma2);
    mix(sig, uint64_t(d.primary_stride));
    mix(sig, uint64_t(d.cross_stride1));
    mix(sig, uint64_t(d.cross_stride2));
    mix_double(sig, d.omega_0);
    mix_double(sig, d.gamma);
    mix_double(sig, d.alpha);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        mix_double(sig, d.gyro_tensor[i][j]);
    mix(sig, uint64_t(d.gyro_model));
    mix_double(sig, d.dt);
    mix_double(sig, d.noise_amplitude);
    mix(sig, uint64_t(d.noise_algorithm_version));
  }
  static void hash_polarization_subtraction(uint64_t &sig, const PolarizationSubtraction &d) {
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.c));
    mix(sig, uint64_t(d.cmp));
    mix(sig, uint64_t(d.state_index));
    hash_id(sig, d.target);
    hash_id(sig, d.p);
    mix(sig, uint64_t(d.elements));
  }
  static void hash_dft(uint64_t &sig, const DftDescriptor &d) {
    hash_id(sig, d.accumulator);
    hash_id(sig, d.phase_scratch);
    hash_ref(sig, d.source_field);
    hash_ref(sig, d.source_field_imag);
    mix(sig, uint64_t(d.omega.size()));
    for (size_t i = 0; i < d.omega.size(); ++i) mix_double(sig, d.omega[i]);
    mix_double(sig, d.scale.real());
    mix_double(sig, d.scale.imag());
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.c));
    mix(sig, uint64_t(d.avg1));
    mix(sig, uint64_t(d.avg2));
    hash_ivec(sig, d.is);
    hash_ivec(sig, d.ie);
    hash_ivec(sig, d.is_old);
    hash_ivec(sig, d.ie_old);
    mix(sig, uint64_t(d.persist));
    mix(sig, uint64_t(d.decimation_factor));
    mix(sig, uint64_t(d.due_scalar_slot));
    hash_vec(sig, d.weights.s0);
    hash_vec(sig, d.weights.s1);
    hash_vec(sig, d.weights.e0);
    hash_vec(sig, d.weights.e1);
    mix_double(sig, d.dV0);
    mix_double(sig, d.dV1);
    mix(sig, uint64_t(d.include_dV_and_interp_weights));
    mix(sig, uint64_t(d.sqrt_dV_and_interp_weights));
    mix(sig, uint64_t(d.N));
    mix(sig, uint64_t(d.Nomega));
  }

  static void hash_legacy_flux_update(uint64_t &sig, const LegacyFluxUpdate &d) {
    mix(sig, uint64_t(d.flux_ordinal));
    mix(sig, uint64_t(d.term_index));
    mix(sig, uint64_t(d.term_count));
    mix(sig, d.recipe_signature);
  }

  static void hash_legacy_flux_term(uint64_t &sig, const LegacyFluxTerm &d) {
    mix(sig, uint64_t(d.flux_ordinal));
    mix(sig, uint64_t(d.term_ordinal));
    mix(sig, uint64_t(d.region_ordinal));
    mix(sig, uint64_t(d.sign));
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.e_component));
    mix(sig, uint64_t(d.h_component));
    hash_id(sig, d.e_real);
    hash_id(sig, d.e_imag);
    hash_id(sig, d.h_real);
    hash_id(sig, d.h_imag);
    for (int i = 0; i < 3; ++i) {
      mix(sig, uint64_t(d.begin.yucky_val(i)));
      mix(sig, uint64_t(d.end.yucky_val(i)));
      mix(sig, uint64_t(d.lattice_shift.yucky_val(i)));
    }
    mix(sig, uint64_t(d.symmetry_index));
    mix(sig, uint64_t(d.base));
    for (int i = 0; i < 3; ++i) {
      mix(sig, uint64_t(d.counts[i]));
      mix(sig, uint64_t(d.strides[i]));
      for (int j = 0; j < 4; ++j)
        mix_double(sig, d.boundary_weights[i][j]);
    }
    for (int i = 0; i < 2; ++i) {
      mix(sig, uint64_t(d.e_offsets[i]));
      mix(sig, uint64_t(d.h_offsets[i]));
    }
    mix_double(sig, d.phase_real);
    mix_double(sig, d.phase_imag);
    mix_double(sig, d.dV0);
    mix_double(sig, d.dV1);
  }

  static void hash_magnetic_state(uint64_t &sig, const MagneticStateArray &d) {
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.c));
    mix(sig, uint64_t(d.cmp));
    mix(sig, uint64_t(d.family));
    hash_id(sig, d.live);
    mix(sig, uint64_t(d.elements));
    mix(sig, uint64_t(d.average));
  }

  static void hash_material_refresh(uint64_t &sig, const MaterialRefreshArray &d) {
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.c));
    mix(sig, uint64_t(d.d));
    mix(sig, uint64_t(d.family));
    hash_id(sig, d.current);
    mix(sig, uint64_t(d.elements));
  }

  static void hash_magnetic_half_step(uint64_t &sig, const MagneticHalfStep &d) {
    mix(sig, uint64_t(d.evaluate_b_sources));
    mix(sig, uint64_t(d.update_b));
    mix(sig, uint64_t(d.apply_b_sources));
    mix(sig, uint64_t(d.transfer_b));
    mix(sig, uint64_t(d.evaluate_h_sources));
    mix(sig, uint64_t(d.update_h));
    mix(sig, uint64_t(d.transfer_h));
  }

  void build_magnetic_state_arrays() {
    const array_kind kinds[] = {array_kind::f, array_kind::f_u, array_kind::f_w, array_kind::f_cond,
                                array_kind::f_bfast};
    const MagneticStateFamily families[] = {
        MagneticStateFamily::primary, MagneticStateFamily::u, MagneticStateFamily::w,
        MagneticStateFamily::conductivity, MagneticStateFamily::bfast};
    for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
      if (!f_.chunks[chunk]->is_mine()) continue;
      const fields_chunk &fc = *f_.chunks[chunk];
      const int components = fc.is_real ? 1 : 2;
      for (int family_type = 0; family_type < 2; ++family_type) {
        const field_type ft = family_type == 0 ? B_stuff : H_stuff;
        FOR_FT_COMPONENTS(ft, c) for (int cmp = 0; cmp < components; ++cmp) {
          const ArrayId primary = find_array(f_, chunk, array_kind::f, int(c), cmp, 0);
          if (!is_valid(primary)) continue;
          if (ft == H_stuff && is_valid(f_.array_catalog->spec(primary).alias_of)) continue;
          for (size_t family = 0; family < sizeof(kinds) / sizeof(kinds[0]); ++family) {
            const ArrayId live = find_array(f_, chunk, kinds[family], int(c), cmp, 0);
            if (!is_valid(live)) continue;
            const ArraySpec &spec = f_.array_catalog->spec(live);
            plan_.magnetic_state_arrays.push_back(
                MagneticStateArray{chunk, c, cmp, families[family], live, spec.elements,
                                   families[family] == MagneticStateFamily::primary});
          }
        }
      }
    }
  }

  fields &f_;
  StepPlan plan_;

  void attach_source_span(Operation &op, field_type ft, bool integrated) {
    if (!f_.descriptors) return;
    const std::vector<SourceDescriptor> &sources = f_.descriptors->sources.sources;
    bool started = false, finished = false;
    for (size_t i = 0; i < sources.size(); ++i) {
      const SourceDescriptor &d = sources[i];
      const bool matches = d.ft == ft && d.integrated == integrated;
      if (matches) {
        if (finished) meep::abort("source descriptors for one operation are not contiguous");
        if (!started) {
          op.source_descriptor_index = uint32_t(i);
          started = true;
        }
        ++op.source_descriptor_count;
        if (integrated) {
          add_access(f_, op, d.destination, AccessMode::read);
          add_access(f_, op, d.destination_imag, AccessMode::read);
          add_access(f_, op, d.integrated_destination, AccessMode::read_write);
          add_access(f_, op, d.integrated_destination_imag, AccessMode::read_write);
        }
        else {
          add_access(f_, op, d.destination, AccessMode::read_write);
          add_access(f_, op, d.destination_imag, AccessMode::read_write);
          add_access(f_, op, d.condinv, AccessMode::read);
        }
      }
      else if (started)
        finished = true;
    }
  }
};

void StepPlanBuilder::add_source_evaluation(Guard guard, double src_offset) {
  Operation &op = add(OpKind::evaluate_source_scalars, field_type(NUM_FIELD_TYPES), guard,
                      src_offset);
  if (f_.descriptors)
    op.descriptor_count = uint32_t(f_.descriptors->sources.source_times.size());
}

void StepPlanBuilder::add_sources(field_type ft) {
  Operation &op = add(OpKind::apply_sources, ft);
  attach_source_span(op, ft, false);
}

void StepPlanBuilder::add_dfts() {
  Operation &op = add(OpKind::update_dft, field_type(NUM_FIELD_TYPES), guard_device(0));
  op.descriptor_index = uint32_t(plan_.dft_updates.size());
  if (f_.descriptors) {
    for (size_t i = 0; i < f_.descriptors->dfts.size(); ++i) {
      const DftDescriptor &d = f_.descriptors->dfts[i];
      plan_.dft_updates.push_back(d);
      add_access(f_, op, d.accumulator, AccessMode::read_write);
      add_access(f_, op, d.phase_scratch, AccessMode::write);
      add_access(f_, op, d.source_field.id, AccessMode::read);
      add_access(f_, op, d.source_field_imag.id, AccessMode::read);
    }
  }
  op.descriptor_count = uint32_t(plan_.dft_updates.size()) - op.descriptor_index;
}

void StepPlanBuilder::add_polarizations(field_type ft) {
  Operation &op = add(OpKind::update_polarization, ft);
  op.descriptor_index = uint32_t(plan_.polarization_updates.size());
  if (!f_.descriptors) return;

  for (size_t di = 0; di < f_.descriptors->polarizations.size(); ++di) {
    const PolarizationDescriptor &descriptor = f_.descriptors->polarizations[di];
    if (descriptor.ft != ft ||
        (descriptor.kind != SusceptibilityKind::lorentzian &&
         descriptor.kind != SusceptibilityKind::gyrotropic))
      continue;
    if (descriptor.chunk < 0 || descriptor.chunk >= f_.num_chunks)
      meep::abort("polarization descriptor has invalid chunk");
    fields_chunk &fc = *f_.chunks[descriptor.chunk];

    for (size_t si = 0; descriptor.kind == SusceptibilityKind::lorentzian &&
                        si < descriptor.lorentzian_states.size();
         ++si) {
      const LorentzianStateArrays &state = descriptor.lorentzian_states[si];
      const direction primary_direction = component_direction(state.c);
      direction cross_direction1 = cycle_direction(fc.gv.dim, primary_direction, 1);
      direction cross_direction2 = cycle_direction(fc.gv.dim, primary_direction, 2);
      component cross_component1 = direction_component(state.c, cross_direction1);
      component cross_component2 = direction_component(state.c, cross_direction2);
      const int sigma_aux = descriptor.state_index * NUM_FIELD_TYPES + int(ft);

      PolarizationUpdate update = {};
      update.kind = PolarizationUpdateKind::lorentzian;
      update.region = make_region(fc.gv, descriptor.chunk, state.c, state.cmp,
                                  fc.gv.little_owned_corner(state.c), fc.gv.big_corner());
      update.state_index = descriptor.state_index;
      update.p = state.p;
      update.p_prev = state.p_prev;
      update.p_cross1 = invalid_array();
      update.p_prev_cross1 = invalid_array();
      update.p_cross2 = invalid_array();
      update.p_prev_cross2 = invalid_array();
      update.primary_w = find_array(f_, descriptor.chunk, array_kind::f_w, int(state.c),
                                    state.cmp, 0);
      if (!is_valid(update.primary_w))
        update.primary_w =
            find_array(f_, descriptor.chunk, array_kind::f, int(state.c), state.cmp, 0);
      update.cross_w1 = find_array(f_, descriptor.chunk, array_kind::f_w,
                                   int(cross_component1), state.cmp, 0);
      if (!is_valid(update.cross_w1))
        update.cross_w1 = find_array(f_, descriptor.chunk, array_kind::f,
                                     int(cross_component1), state.cmp, 0);
      update.cross_w2 = find_array(f_, descriptor.chunk, array_kind::f_w,
                                   int(cross_component2), state.cmp, 0);
      if (!is_valid(update.cross_w2))
        update.cross_w2 = find_array(f_, descriptor.chunk, array_kind::f,
                                     int(cross_component2), state.cmp, 0);
      update.diagonal_sigma = find_array(f_, descriptor.chunk, array_kind::sigma, int(state.c),
                                         int(primary_direction), sigma_aux);
      update.offdiagonal_sigma1 =
          is_valid(update.cross_w1)
              ? find_array(f_, descriptor.chunk, array_kind::sigma, int(state.c),
                           int(cross_direction1), sigma_aux)
              : invalid_array();
      update.offdiagonal_sigma2 =
          is_valid(update.cross_w2)
              ? find_array(f_, descriptor.chunk, array_kind::sigma, int(state.c),
                           int(cross_direction2), sigma_aux)
              : invalid_array();
      const ptrdiff_t stride_sign = is_magnetic(state.c) ? -1 : 1;
      update.primary_stride = stride_sign * fc.gv.stride(primary_direction);
      update.cross_stride1 = stride_sign * fc.gv.stride(cross_direction1);
      update.cross_stride2 = stride_sign * fc.gv.stride(cross_direction2);

      if (is_valid(update.offdiagonal_sigma2) && !is_valid(update.offdiagonal_sigma1)) {
        std::swap(update.cross_w1, update.cross_w2);
        std::swap(update.offdiagonal_sigma1, update.offdiagonal_sigma2);
        std::swap(update.cross_stride1, update.cross_stride2);
      }

      if (!is_valid(update.primary_w) || !is_valid(update.diagonal_sigma)) continue;
      if (is_valid(update.offdiagonal_sigma1))
        update.region.variant_key |= polarization_one_offdiagonal;
      else
        update.cross_w1 = invalid_array();
      if (is_valid(update.offdiagonal_sigma2))
        update.region.variant_key |= polarization_two_offdiagonals;
      else
        update.cross_w2 = invalid_array();
      if (descriptor.lorentzian.drude) update.region.variant_key |= polarization_drude;
      update.omega_0 = descriptor.lorentzian.omega_0;
      update.gamma = descriptor.lorentzian.gamma;
      update.alpha = 0.0;
      memset(update.gyro_tensor, 0, sizeof(update.gyro_tensor));
      update.gyro_model = GYROTROPIC_LORENTZIAN;
      update.dt = fc.dt;

      plan_.polarization_updates.push_back(update);
      add_access(f_, op, update.p, AccessMode::read_write);
      add_access(f_, op, update.p_prev, AccessMode::read_write);
      add_access(f_, op, update.primary_w, AccessMode::read);
      add_access(f_, op, update.cross_w1, AccessMode::read);
      add_access(f_, op, update.cross_w2, AccessMode::read);
      add_access(f_, op, update.diagonal_sigma, AccessMode::read);
      add_access(f_, op, update.offdiagonal_sigma1, AccessMode::read);
      add_access(f_, op, update.offdiagonal_sigma2, AccessMode::read);
    }

    for (size_t si = 0; descriptor.kind == SusceptibilityKind::gyrotropic &&
                        si < descriptor.gyrotropic_states.size();
         ++si) {
      const GyrotropicStateArrays &state = descriptor.gyrotropic_states[si];
      const direction d0 = component_direction(state.c);
      const direction d1 = cycle_direction(fc.gv.dim, d0, 1);
      const direction d2 = cycle_direction(fc.gv.dim, d0, 2);
      const component c1 = direction_component(state.c, d1);
      const component c2 = direction_component(state.c, d2);
      const int sigma_aux = descriptor.state_index * NUM_FIELD_TYPES + int(ft);

      PolarizationUpdate update = {};
      update.kind = PolarizationUpdateKind::gyrotropic;
      update.region = make_region(fc.gv, descriptor.chunk, state.c, state.cmp,
                                  fc.gv.little_owned_corner(state.c), fc.gv.big_corner());
      update.state_index = descriptor.state_index;
      update.p = state.p[int(d0)];
      update.p_prev = state.p_prev[int(d0)];
      update.p_cross1 = state.p[int(d1)];
      update.p_prev_cross1 = state.p_prev[int(d1)];
      update.p_cross2 = state.p[int(d2)];
      update.p_prev_cross2 = state.p_prev[int(d2)];
      update.primary_w =
          find_array(f_, descriptor.chunk, array_kind::f_w, int(state.c), state.cmp, 0);
      if (!is_valid(update.primary_w))
        update.primary_w =
            find_array(f_, descriptor.chunk, array_kind::f, int(state.c), state.cmp, 0);
      update.cross_w1 =
          find_array(f_, descriptor.chunk, array_kind::f_w, int(c1), state.cmp, 0);
      if (!is_valid(update.cross_w1))
        update.cross_w1 = find_array(f_, descriptor.chunk, array_kind::f, int(c1), state.cmp, 0);
      update.cross_w2 =
          find_array(f_, descriptor.chunk, array_kind::f_w, int(c2), state.cmp, 0);
      if (!is_valid(update.cross_w2))
        update.cross_w2 = find_array(f_, descriptor.chunk, array_kind::f, int(c2), state.cmp, 0);
      update.diagonal_sigma = find_array(f_, descriptor.chunk, array_kind::sigma, int(state.c),
                                         int(d0), sigma_aux);
      update.offdiagonal_sigma1 = find_array(f_, descriptor.chunk, array_kind::sigma, int(state.c),
                                             int(d1), sigma_aux);
      update.offdiagonal_sigma2 = find_array(f_, descriptor.chunk, array_kind::sigma, int(state.c),
                                             int(d2), sigma_aux);
      const ptrdiff_t sign = is_magnetic(state.c) ? -1 : 1;
      update.primary_stride = sign * fc.gv.stride(d0);
      update.cross_stride1 = sign * fc.gv.stride(d1);
      update.cross_stride2 = sign * fc.gv.stride(d2);
      update.omega_0 = descriptor.gyrotropic.omega_0;
      update.gamma = descriptor.gyrotropic.gamma;
      update.alpha = descriptor.gyrotropic.alpha;
      memcpy(update.gyro_tensor, descriptor.gyrotropic.gyro_tensor,
             sizeof(update.gyro_tensor));
      update.gyro_model = descriptor.gyrotropic.model;
      update.dt = fc.dt;

      plan_.polarization_updates.push_back(update);
      add_access(f_, op, update.p, AccessMode::read_write);
      add_access(f_, op, update.p_prev, AccessMode::read_write);
      add_access(f_, op, update.p_cross1, AccessMode::read_write);
      add_access(f_, op, update.p_prev_cross1, AccessMode::read_write);
      add_access(f_, op, update.p_cross2, AccessMode::read_write);
      add_access(f_, op, update.p_prev_cross2, AccessMode::read_write);
      add_access(f_, op, update.primary_w, AccessMode::read);
      add_access(f_, op, update.cross_w1, AccessMode::read);
      add_access(f_, op, update.cross_w2, AccessMode::read);
      add_access(f_, op, update.diagonal_sigma, AccessMode::read);
      add_access(f_, op, update.offdiagonal_sigma1, AccessMode::read);
      add_access(f_, op, update.offdiagonal_sigma2, AccessMode::read);
    }
  }
  op.descriptor_count = uint32_t(plan_.polarization_updates.size()) - op.descriptor_index;
}

void StepPlanBuilder::add_db(field_type ft) {
  Operation &op = add(OpKind::update_db, ft);
  op.descriptor_index = uint32_t(plan_.db_updates.size());
  op.beta_descriptor_index = uint32_t(plan_.beta_updates.size());
  op.cylindrical_m_descriptor_index = uint32_t(plan_.cylindrical_m_updates.size());
  op.cylindrical_origin_action_index = uint32_t(plan_.cylindrical_origin_actions.size());

  for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
    if (!f_.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f_.chunks[chunk];
    const int components = fc.is_real ? 1 : 2;
    for (size_t tile = 0; tile < fc.gvs_tiled.size(); ++tile) {
      const grid_volume &sub = fc.gvs_tiled[tile];
      for (int cmp = 0; cmp < components; ++cmp)
        FOR_FT_COMPONENTS(ft, cc) {
          const ArrayId target = find_array(f_, chunk, array_kind::f, int(cc), cmp, 0);
          if (!is_valid(target)) continue;

          const CurlSources sources = curl_sources_for(fc, cc);
          const direction dc = component_direction(cc);
          const direction dsig0 = cycle_direction(fc.gv.dim, dc, 1);
          const direction dsig = fc.s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
          const direction dsigu0 = cycle_direction(fc.gv.dim, dc, 2);
          const direction dsigu = fc.s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;

          CurlUpdate d;
          d.region =
              make_region(fc.gv, chunk, cc, cmp, sub.little_owned_corner0(cc), sub.big_corner());
          d.target = target;
          d.plus_source = sources.have_plus ? find_array(f_, chunk, array_kind::f,
                                                         int(sources.plus_component), cmp, 0)
                                            : invalid_array();
          d.minus_source = sources.have_minus ? find_array(f_, chunk, array_kind::f,
                                                           int(sources.minus_component), cmp, 0)
                                              : invalid_array();
          d.plus_stride = sources.have_plus ? fc.gv.stride(sources.plus_direction) : 0;
          d.minus_stride = sources.have_minus ? fc.gv.stride(sources.minus_direction) : 0;
          if (ft == D_stuff) {
            d.plus_stride = -d.plus_stride;
            d.minus_stride = -d.minus_stride;
          }
          d.target_u = find_array(f_, chunk, array_kind::f_u, int(cc), cmp, 0);
          d.conductivity = find_array(f_, chunk, array_kind::conductivity, int(cc), -1, int(dc));
          d.condinv = find_array(f_, chunk, array_kind::condinv, int(cc), -1, int(dc));
          d.target_cond = find_array(f_, chunk, array_kind::f_cond, int(cc), cmp, 0);
          d.pml = make_pml_profile(f_, fc, chunk, dsig, d.region.begin);
          d.pml_u = make_pml_profile(f_, fc, chunk, dsigu, d.region.begin);
          d.dtdx = fc.Courant;
          d.dt = fc.dt;
          d.radial_prefix_index = UINT32_MAX;
          d.bfast_update_index = UINT32_MAX;

          if (fc.gv.dim == Dcyl) {
            switch (dc) {
              case R: d.plus_source = invalid_array(); break;
              case P: break;
              case Z: {
                CylindricalRadialPrefix prefix;
                prefix.chunk = chunk;
                prefix.target_component = cc;
                prefix.source_component = sources.plus_component;
                prefix.cmp = cmp;
                prefix.source = d.plus_source;
                prefix.scratch = find_array(f_, chunk, array_kind::f_rderiv_int, -1, -1, 0);
                prefix.nr = size_t(fc.gv.nr());
                prefix.nz = size_t(fc.gv.nz());
                prefix.row_stride = prefix.nz + 1;
                prefix.source_elements =
                    is_valid(prefix.source) ? f_.array_catalog->spec(prefix.source).elements : 0;
                prefix.scratch_elements =
                    is_valid(prefix.scratch) ? f_.array_catalog->spec(prefix.scratch).elements : 0;
                const realnum ir0 = fc.gv.origin_r() * fc.gv.a +
                                    0.5 * fc.gv.iyee_shift(sources.plus_component).in_direction(R);
                prefix.ir0 = ir0;
                d.radial_prefix_index = uint32_t(plan_.cylindrical_radial_prefixes.size());
                plan_.cylindrical_radial_prefixes.push_back(prefix);
                d.plus_source = prefix.scratch;
                d.minus_source = invalid_array();
                add_access(f_, op, prefix.source, AccessMode::read);
                add_access(f_, op, prefix.scratch, AccessMode::read_write);
                break;
              }
              default: meep::abort("bug - non-cylindrical field component in Dcyl");
            }
          }

          if (is_valid(d.plus_source) && is_valid(d.minus_source))
            d.region.variant_key |= curl_has_second_derivative;
          if (dsig != NO_DIRECTION) d.region.variant_key |= curl_has_pml;
          if (dsigu != NO_DIRECTION) d.region.variant_key |= curl_has_pml_aux;
          if (is_valid(d.conductivity)) d.region.variant_key |= curl_has_conductivity;
          if (fc.bfast_scaled_k[0] || fc.bfast_scaled_k[1] || fc.bfast_scaled_k[2]) {
            d.region.variant_key |= curl_has_bfast;

            BfastUpdate b;
            b.region = d.region;
            b.region.variant_key = 0;
            b.target = d.target;
            b.source1 = d.plus_source;
            b.source2 = d.minus_source;
            b.stride1 = d.plus_stride;
            b.stride2 = d.minus_stride;
            b.f_bfast = find_array(f_, chunk, array_kind::f_bfast, int(cc), cmp, 0);
            b.target_u = d.target_u;
            b.condinv = d.condinv;
            b.target_cond = d.target_cond;
            b.pml = d.pml;
            b.pml_u = d.pml_u;
            realnum k1 = sources.have_minus
                             ? fc.bfast_scaled_k[component_index(sources.minus_component)]
                             : 0;
            realnum k2 =
                sources.have_plus ? fc.bfast_scaled_k[component_index(sources.plus_component)] : 0;
            if (ft == D_stuff) {
              k1 = -k1;
              k2 = -k2;
            }
            b.k1 = k1;
            b.k2 = k2;
            if (dsig != NO_DIRECTION) b.region.variant_key |= bfast_has_pml;
            if (dsigu != NO_DIRECTION) b.region.variant_key |= bfast_has_pml_aux;
            if (is_valid(b.condinv)) b.region.variant_key |= bfast_has_conductivity;
            d.bfast_update_index = uint32_t(plan_.bfast_updates.size());
            plan_.bfast_updates.push_back(b);

            add_access(f_, op, b.target, AccessMode::read_write);
            add_access(f_, op, b.source1, AccessMode::read);
            add_access(f_, op, b.source2, AccessMode::read);
            add_access(f_, op, b.f_bfast, AccessMode::read_write);
            add_access(f_, op, b.target_u, AccessMode::read_write);
            add_access(f_, op, b.condinv, AccessMode::read);
            add_access(f_, op, b.target_cond, AccessMode::read_write);
            add_access(f_, op, b.pml.siginv, AccessMode::read);
            add_access(f_, op, b.pml_u.siginv, AccessMode::read);
          }

          plan_.db_updates.push_back(d);
          add_access(f_, op, d.target, AccessMode::read_write);
          add_access(f_, op, d.plus_source, AccessMode::read);
          add_access(f_, op, d.minus_source, AccessMode::read);
          add_access(f_, op, d.target_u, AccessMode::read_write);
          add_access(f_, op, d.conductivity, AccessMode::read);
          add_access(f_, op, d.condinv, AccessMode::read);
          add_access(f_, op, d.target_cond, AccessMode::read_write);
          add_access(f_, op, d.pml.sig, AccessMode::read);
          add_access(f_, op, d.pml.kap, AccessMode::read);
          add_access(f_, op, d.pml.siginv, AccessMode::read);
          add_access(f_, op, d.pml_u.sig, AccessMode::read);
          add_access(f_, op, d.pml_u.kap, AccessMode::read);
          add_access(f_, op, d.pml_u.siginv, AccessMode::read);
        }
    }
  }
  op.descriptor_count = uint32_t(plan_.db_updates.size()) - op.descriptor_index;

  /* Match step_db.cpp exactly: special-kz rows run only after every ordinary
     curl tile in this half-step has completed, and cover the full chunk rather
     than an individual tile. */
  for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
    if (!f_.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f_.chunks[chunk];
    if (fc.gv.dim != D2 || fc.beta == 0) continue;
    const int components = fc.is_real ? 1 : 2;
    for (int cmp = 0; cmp < components; ++cmp)
      for (direction dc = X; dc <= Y; dc = direction(dc + 1)) {
        const component target_component = direction_component(first_field_component(ft), dc);
        const component source_component =
            direction_component(ft == D_stuff ? Hx : Ex, dc == X ? Y : X);
        const ArrayId target = find_array(f_, chunk, array_kind::f, int(target_component), cmp, 0);
        const ArrayId opposite_source =
            find_array(f_, chunk, array_kind::f, int(source_component), 1 - cmp, 0);
        const ArrayId same_source =
            find_array(f_, chunk, array_kind::f, int(source_component), cmp, 0);
        const ArrayId source = is_valid(opposite_source) ? opposite_source : same_source;
        /* step_beta is a no-op without either operand. Do not turn such a row
           into executable backend work. */
        if (!is_valid(target) || !is_valid(source)) continue;

        const direction dsig0 = cycle_direction(fc.gv.dim, dc, 1);
        const direction dsig = fc.s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
        const direction dsigu0 = cycle_direction(fc.gv.dim, dc, 2);
        const direction dsigu = fc.s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;

        BetaUpdate d;
        d.region = make_region(fc.gv, chunk, target_component, cmp,
                               fc.gv.little_owned_corner0(target_component), fc.gv.big_corner());
        d.target = target;
        d.source = source;
        d.target_u = find_array(f_, chunk, array_kind::f_u, int(target_component), cmp, 0);
        d.condinv = find_array(f_, chunk, array_kind::condinv, int(target_component), -1, int(dc));
        d.target_cond = find_array(f_, chunk, array_kind::f_cond, int(target_component), cmp, 0);
        d.pml = make_pml_profile(f_, fc, chunk, dsig, d.region.begin);
        d.pml_u = make_pml_profile(f_, fc, chunk, dsigu, d.region.begin);
        const realnum betadt =
            2 * pi * fc.beta * fc.dt * (dc == X ? +1 : -1) *
            (is_valid(opposite_source) ? (ft == D_stuff ? -1 : +1) * (2 * cmp - 1) : 1);
        d.betadt = betadt;
        if (dsig != NO_DIRECTION) d.region.variant_key |= beta_has_pml;
        if (dsigu != NO_DIRECTION) d.region.variant_key |= beta_has_pml_aux;
        if (is_valid(d.condinv)) d.region.variant_key |= beta_has_conductivity;

        plan_.beta_updates.push_back(d);
        add_access(f_, op, d.target, AccessMode::read_write);
        add_access(f_, op, d.source, AccessMode::read);
        add_access(f_, op, d.target_u, AccessMode::read_write);
        add_access(f_, op, d.condinv, AccessMode::read);
        add_access(f_, op, d.target_cond, AccessMode::read_write);
        add_access(f_, op, d.pml.siginv, AccessMode::read);
        add_access(f_, op, d.pml_u.siginv, AccessMode::read);
      }
  }
  op.beta_descriptor_count = uint32_t(plan_.beta_updates.size()) - op.beta_descriptor_index;

  /* Cylindrical m/r terms are chunk-wide tails after every tiled curl and
     paired BFAST postpass. */
  for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
    if (!f_.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f_.chunks[chunk];
    if (fc.gv.dim != Dcyl || fc.m == 0) continue;
    const int components = fc.is_real ? 1 : 2;
    for (int cmp = 0; cmp < components; ++cmp)
      FOR_FT_COMPONENTS(ft, cc) {
        const direction dc = component_direction(cc);
        if (dc != R && dc != Z) continue;
        const ArrayId target = find_array(f_, chunk, array_kind::f, int(cc), cmp, 0);
        const CurlSources tail_sources = curl_sources_for(fc, cc);
        const component source_component =
            dc == R ? tail_sources.plus_component : tail_sources.minus_component;
        if (source_component == NO_COMPONENT) continue;
        const ArrayId source =
            find_array(f_, chunk, array_kind::f, int(source_component), 1 - cmp, 0);
        if (!is_valid(target) || !is_valid(source)) continue;

        const direction dsig0 = cycle_direction(fc.gv.dim, dc, 1);
        const direction dsig = fc.s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
        const direction dsigu0 = cycle_direction(fc.gv.dim, dc, 2);
        const direction dsigu = fc.s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;

        CylindricalMOverRUpdate d;
        d.region =
            make_region(fc.gv, chunk, cc, cmp, fc.gv.little_owned_corner0(cc), fc.gv.big_corner());
        d.target = target;
        d.source = source;
        d.target_u = find_array(f_, chunk, array_kind::f_u, int(cc), cmp, 0);
        d.condinv = find_array(f_, chunk, array_kind::condinv, int(cc), -1, int(dc));
        d.target_cond = dsig != NO_DIRECTION && is_valid(d.condinv)
                            ? find_array(f_, chunk, array_kind::f_cond, int(cc), cmp, 0)
                            : invalid_array();
        d.pml = make_pml_profile(f_, fc, chunk, dsig, d.region.begin);
        d.pml_u = make_pml_profile(f_, fc, chunk, dsigu, d.region.begin);
        const realnum numerator =
            2 * fc.m * (1 - 2 * cmp) * (1 - 2 * (ft == B_stuff)) * (1 - 2 * (dc == R)) * fc.Courant;
        d.numerator = numerator;
        d.raw_radial_start = d.region.begin.in_direction(R);
        if (dsig != NO_DIRECTION) d.region.variant_key |= cylindrical_m_has_pml;
        if (dsigu != NO_DIRECTION) d.region.variant_key |= cylindrical_m_has_pml_aux;
        if (is_valid(d.condinv)) d.region.variant_key |= cylindrical_m_has_conductivity;
        plan_.cylindrical_m_updates.push_back(d);

        add_access(f_, op, d.target, AccessMode::read_write);
        add_access(f_, op, d.source, AccessMode::read);
        add_access(f_, op, d.target_u, AccessMode::read_write);
        add_access(f_, op, d.condinv, AccessMode::read);
        add_access(f_, op, d.target_cond, AccessMode::read_write);
        add_access(f_, op, d.pml.siginv, AccessMode::read);
        add_access(f_, op, d.pml_u.siginv, AccessMode::read);
      }
  }
  op.cylindrical_m_descriptor_count =
      uint32_t(plan_.cylindrical_m_updates.size()) - op.cylindrical_m_descriptor_index;

  auto add_zero_slab = [&](fields_chunk &fc, ArrayId id, int radial_row) {
    if (!is_valid(id)) return;
    SlabRef slab;
    slab.array = id;
    slab.base = ptrdiff_t(radial_row) * ptrdiff_t(fc.gv.nz() + 1);
    slab.counts[0] = fc.gv.nz() + 1;
    slab.counts[1] = slab.counts[2] = 1;
    slab.strides[0] = 1;
    slab.strides[1] = slab.strides[2] = 0;
    plan_.cylindrical_zero_slabs.push_back(slab);
    plan_.cylindrical_origin_actions.push_back(CylindricalOriginAction{
        CylindricalOriginActionKind::zero_slab, uint32_t(plan_.cylindrical_zero_slabs.size() - 1)});
    add_access(f_, op, id, AccessMode::write);
  };

  auto add_component_zero = [&](fields_chunk &fc, int chunk, component c, int cmp, int radial_row) {
    add_zero_slab(fc, find_array(f_, chunk, array_kind::f, int(c), cmp, 0), radial_row);
    add_zero_slab(fc, find_array(f_, chunk, array_kind::f_cond, int(c), cmp, 0), radial_row);
    add_zero_slab(fc, find_array(f_, chunk, array_kind::f_u, int(c), cmp, 0), radial_row);
  };

  auto add_family_zero = [&](fields_chunk &fc, int chunk, field_type family, int cmp,
                             int radial_row) {
    const array_kind kinds[] = {array_kind::f, array_kind::f_cond, array_kind::f_u};
    for (array_kind kind : kinds)
      FOR_FT_COMPONENTS(family, c)
    add_zero_slab(fc, find_array(f_, chunk, kind, int(c), cmp, 0), radial_row);
  };

  /* Origin arithmetic and zero slabs execute after the m/r tail. */
  for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
    if (!f_.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f_.chunks[chunk];
    if (fc.gv.dim != Dcyl || fc.gv.origin_r() != 0.0) continue;
    const int components = fc.is_real ? 1 : 2;
    for (int cmp = 0; cmp < components; ++cmp) {
      component target_component = NO_COMPONENT;
      CylindricalAxisKind kind = CylindricalAxisKind::m0_dz;
      ArrayId source1 = invalid_array(), source2 = invalid_array();
      ptrdiff_t source1_neighbor_offset = 0, source2_offset = 0;
      realnum scale = 0, source2_multiplier = 0;
      component zero_after_axis = NO_COMPONENT;

      if (fc.m == 0 && ft == D_stuff) {
        target_component = Dz;
        source1 = find_array(f_, chunk, array_kind::f, int(Hp), cmp, 0);
        scale = fc.Courant * 4;
        zero_after_axis = Dp;
      }
      else if (fc.m == 0 && ft == B_stuff) {
        if (is_valid(find_array(f_, chunk, array_kind::f, int(Br), cmp, 0)))
          add_component_zero(fc, chunk, Br, cmp, 0);
      }
      else if (fabs(fc.m) == 1) {
        kind = CylindricalAxisKind::abs_m1;
        target_component = ft == D_stuff ? Dp : Br;
        const int sd = ft == D_stuff ? +1 : -1;
        source1 = find_array(f_, chunk, array_kind::f, int(ft == D_stuff ? Hr : Ep), cmp, 0);
        source2 = find_array(f_, chunk, array_kind::f, int(ft == D_stuff ? Hz : Ez),
                             ft == D_stuff ? cmp : 1 - cmp, 0);
        source1_neighbor_offset = -sd;
        source2_offset = ft == D_stuff ? 0 : fc.gv.nz() + 1;
        scale = sd * fc.Courant;
        source2_multiplier = ft == D_stuff ? 2 : (1 - 2 * cmp) * fc.m;
        if (ft == D_stuff) zero_after_axis = Dz;
      }
      else if (fc.m != 0) {
        int radial_rows = 1;
        if (fc.zero_fields_near_cylorigin) {
          radial_rows = 0;
          const double rmax = fabs(fc.m) - int(fc.gv.origin_r() * fc.gv.a + 0.5);
          while (radial_rows <= fc.gv.nr() && radial_rows < rmax)
            ++radial_rows;
        }
        for (int row = 0; row < radial_rows; ++row)
          add_family_zero(fc, chunk, ft, cmp, row);
      }

      const ArrayId target =
          target_component == NO_COMPONENT
              ? invalid_array()
              : find_array(f_, chunk, array_kind::f, int(target_component), cmp, 0);
      if (is_valid(target) && is_valid(source1) &&
          (kind == CylindricalAxisKind::m0_dz || is_valid(source2))) {
        const direction dc = component_direction(target_component);
        const direction dsig0 = cycle_direction(fc.gv.dim, dc, 1);
        const direction dsig = fc.s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
        const direction dsigu0 = cycle_direction(fc.gv.dim, dc, 2);
        const direction dsigu = fc.s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;
        ivec begin = fc.gv.little_owned_corner(target_component);
        ivec end = fc.gv.big_owned_corner(target_component);
        end.set_direction(R, 0);

        CylindricalAxisUpdate d;
        d.kind = kind;
        d.region = make_region(fc.gv, chunk, target_component, cmp, begin, end);
        d.target = target;
        d.source1 = source1;
        d.source2 = source2;
        d.source1_neighbor_offset = source1_neighbor_offset;
        d.source2_offset = source2_offset;
        d.target_u = find_array(f_, chunk, array_kind::f_u, int(target_component), cmp, 0);
        d.conductivity =
            find_array(f_, chunk, array_kind::conductivity, int(target_component), -1, int(dc));
        d.condinv = find_array(f_, chunk, array_kind::condinv, int(target_component), -1, int(dc));
        d.target_cond = find_array(f_, chunk, array_kind::f_cond, int(target_component), cmp, 0);
        if (!is_valid(d.target_cond)) {
          d.conductivity = invalid_array();
          d.condinv = invalid_array();
        }
        d.pml = make_pml_profile(f_, fc, chunk, dsig, d.region.begin);
        d.pml_u = make_pml_profile(f_, fc, chunk, dsigu, d.region.begin);
        d.scale = scale;
        d.source2_multiplier = source2_multiplier;
        d.dt = fc.dt;
        if (dsig != NO_DIRECTION) d.region.variant_key |= cylindrical_axis_has_pml;
        if (dsigu != NO_DIRECTION) d.region.variant_key |= cylindrical_axis_has_pml_aux;
        if (is_valid(d.target_cond)) d.region.variant_key |= cylindrical_axis_has_conductivity;
        plan_.cylindrical_axis_updates.push_back(d);
        plan_.cylindrical_origin_actions.push_back(
            CylindricalOriginAction{CylindricalOriginActionKind::axis_update,
                                    uint32_t(plan_.cylindrical_axis_updates.size() - 1)});

        add_access(f_, op, d.target, AccessMode::read_write);
        add_access(f_, op, d.source1, AccessMode::read);
        add_access(f_, op, d.source2, AccessMode::read);
        add_access(f_, op, d.target_u, AccessMode::read_write);
        add_access(f_, op, d.conductivity, AccessMode::read);
        add_access(f_, op, d.condinv, AccessMode::read);
        add_access(f_, op, d.target_cond, AccessMode::read_write);
        add_access(f_, op, d.pml.sig, AccessMode::read);
        add_access(f_, op, d.pml.kap, AccessMode::read);
        add_access(f_, op, d.pml.siginv, AccessMode::read);
        add_access(f_, op, d.pml_u.sig, AccessMode::read);
        add_access(f_, op, d.pml_u.kap, AccessMode::read);
        add_access(f_, op, d.pml_u.siginv, AccessMode::read);
        if (zero_after_axis != NO_COMPONENT) add_component_zero(fc, chunk, zero_after_axis, cmp, 0);
      }
    }
  }
  op.cylindrical_origin_action_count =
      uint32_t(plan_.cylindrical_origin_actions.size()) - op.cylindrical_origin_action_index;
}

void StepPlanBuilder::add_eh(field_type ft, Guard guard) {
  Operation &op = add(OpKind::update_eh, ft, guard);
  op.descriptor_index = uint32_t(plan_.eh_updates.size());
  op.polarization_subtraction_index = uint32_t(plan_.polarization_subtractions.size());
  const field_type ft2 = ft == E_stuff ? D_stuff : B_stuff;

  if (f_.descriptors && plan_.program != StepProgram::solve_cw) {
    for (size_t di = 0; di < f_.descriptors->polarizations.size(); ++di) {
      const PolarizationDescriptor &descriptor = f_.descriptors->polarizations[di];
      if (descriptor.ft != ft) continue;
      const size_t state_count = descriptor.kind == SusceptibilityKind::lorentzian
                                     ? descriptor.lorentzian_states.size()
                                 : descriptor.kind == SusceptibilityKind::gyrotropic
                                     ? descriptor.gyrotropic_states.size()
                                     : 0;
      for (size_t si = 0; si < state_count; ++si) {
        const component state_c = descriptor.kind == SusceptibilityKind::lorentzian
                                      ? descriptor.lorentzian_states[si].c
                                      : descriptor.gyrotropic_states[si].c;
        const int state_cmp = descriptor.kind == SusceptibilityKind::lorentzian
                                  ? descriptor.lorentzian_states[si].cmp
                                  : descriptor.gyrotropic_states[si].cmp;
        const ArrayId state_p = descriptor.kind == SusceptibilityKind::lorentzian
                                    ? descriptor.lorentzian_states[si].p
                                    : descriptor.gyrotropic_states[si]
                                          .p[int(component_direction(state_c))];
        const size_t state_elements = descriptor.kind == SusceptibilityKind::lorentzian
                                          ? descriptor.lorentzian_states[si].elements
                                          : descriptor.gyrotropic_states[si].elements;
        const component target_component = field_type_component(ft2, state_c);
        const ArrayId target = find_array(f_, descriptor.chunk, array_kind::f_minus_p,
                                          int(target_component), state_cmp, 0);
        if (!is_valid(target)) continue;
        PolarizationSubtraction subtraction;
        subtraction.chunk = descriptor.chunk;
        subtraction.c = state_c;
        subtraction.cmp = state_cmp;
        subtraction.state_index = descriptor.state_index;
        subtraction.target = target;
        subtraction.p = state_p;
        subtraction.elements = state_elements;
        plan_.polarization_subtractions.push_back(subtraction);
        add_access(f_, op, subtraction.target, AccessMode::read_write);
        add_access(f_, op, subtraction.p, AccessMode::read);
      }
    }
  }
  op.polarization_subtraction_count =
      uint32_t(plan_.polarization_subtractions.size()) - op.polarization_subtraction_index;

  for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
    if (!f_.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f_.chunks[chunk];
    const int components = fc.is_real ? 1 : 2;
    for (size_t tile = 0; tile < fc.gvs_eh[ft].size(); ++tile) {
      const grid_volume &sub = fc.gvs_eh[ft][tile];
      for (int cmp = 0; cmp < components; ++cmp)
        FOR_FT_COMPONENTS(ft, ec) {
          if (!fc.f[ec][cmp]) continue;
          const component dc = field_type_component(ft2, ec);
          if (fc.f[ec][cmp] == fc.f[dc][cmp]) continue;

          const direction dec = component_direction(ec);
          const direction d1 = cycle_direction(fc.gv.dim, dec, 1);
          const direction d2 = cycle_direction(fc.gv.dim, dec, 2);
          const component dc1 = direction_component(dc, d1);
          const component dc2 = direction_component(dc, d2);
          const direction dsigw = fc.s->sigsize[dec] > 1 ? dec : NO_DIRECTION;

          ConstitutiveUpdate d;
          d.region =
              make_region(fc.gv, chunk, ec, cmp, sub.little_owned_corner0(ec), sub.big_corner());
          d.target = find_array(f_, chunk, array_kind::f, int(ec), cmp, 0);
          const bool use_minus_p = plan_.program != StepProgram::solve_cw;
          const ArrayId primary_minus_p =
              use_minus_p ? find_array(f_, chunk, array_kind::f_minus_p, int(dc), cmp, 0)
                          : invalid_array();
          const ArrayId cross1_minus_p =
              use_minus_p ? find_array(f_, chunk, array_kind::f_minus_p, int(dc1), cmp, 0)
                          : invalid_array();
          const ArrayId cross2_minus_p =
              use_minus_p ? find_array(f_, chunk, array_kind::f_minus_p, int(dc2), cmp, 0)
                          : invalid_array();
          d.base_primary = find_array(f_, chunk, array_kind::f, int(dc), cmp, 0);
          d.base_cross1 = find_array(f_, chunk, array_kind::f, int(dc1), cmp, 0);
          d.base_cross2 = find_array(f_, chunk, array_kind::f, int(dc2), cmp, 0);
          d.primary = is_valid(primary_minus_p) ? primary_minus_p : d.base_primary;
          d.cross1 = is_valid(cross1_minus_p) ? cross1_minus_p : d.base_cross1;
          d.cross2 = is_valid(cross2_minus_p) ? cross2_minus_p : d.base_cross2;
          d.diagonal = find_array(f_, chunk, array_kind::chi1inv, int(ec), -1, int(dec));
          d.offdiagonal1 = find_array(f_, chunk, array_kind::chi1inv, int(ec), -1, int(d1));
          d.offdiagonal2 = find_array(f_, chunk, array_kind::chi1inv, int(ec), -1, int(d2));
          d.primary_stride = fc.gv.stride(dec) * (ft == H_stuff ? -1 : 1);
          d.cross1_stride = fc.gv.stride(d1) * (ft == H_stuff ? -1 : 1);
          d.cross2_stride = fc.gv.stride(d2) * (ft == H_stuff ? -1 : 1);

          /* Match step_update_EDHB's normalization: the one surviving
             off-diagonal term is always slot 1. */
          if ((!is_valid(d.cross1) && is_valid(d.cross2)) ||
              (is_valid(d.cross1) && is_valid(d.cross2) && !is_valid(d.offdiagonal1) &&
               is_valid(d.offdiagonal2))) {
            std::swap(d.base_cross1, d.base_cross2);
            std::swap(d.cross1, d.cross2);
            std::swap(d.offdiagonal1, d.offdiagonal2);
            std::swap(d.cross1_stride, d.cross2_stride);
          }

          d.chi2 = find_array(f_, chunk, array_kind::chi2, int(ec), -1, 0);
          d.chi3 = find_array(f_, chunk, array_kind::chi3, int(ec), -1, 0);
          d.target_w = find_array(f_, chunk, array_kind::f_w, int(ec), cmp, 0);
          d.previous_w = find_array(f_, chunk, array_kind::f_w_prev, int(ec), cmp, 0);
          d.pml = make_pml_profile(f_, fc, chunk, dsigw, d.region.begin);
          if (is_valid(d.offdiagonal1)) d.region.variant_key |= constitutive_one_offdiagonal;
          if (is_valid(d.offdiagonal2)) d.region.variant_key |= constitutive_two_offdiagonals;
          if (dsigw != NO_DIRECTION) d.region.variant_key |= constitutive_has_pml;
          if (is_valid(d.chi2) || is_valid(d.chi3))
            d.region.variant_key |= constitutive_has_nonlinearity;
          if (is_valid(primary_minus_p) || is_valid(cross1_minus_p) || is_valid(cross2_minus_p))
            d.region.variant_key |= constitutive_has_minus_p;
          if (tile == 0 && is_valid(d.previous_w))
            d.region.variant_key |= constitutive_copy_w_previous;

          plan_.eh_updates.push_back(d);
          add_access(f_, op, d.target, AccessMode::read_write);
          add_access(f_, op, d.base_primary, AccessMode::read);
          add_access(f_, op, d.base_cross1, AccessMode::read);
          add_access(f_, op, d.base_cross2, AccessMode::read);
          add_access(f_, op, d.primary,
                     d.primary != d.base_primary ? AccessMode::read_write : AccessMode::read);
          add_access(f_, op, d.cross1,
                     d.cross1 != d.base_cross1 ? AccessMode::read_write : AccessMode::read);
          add_access(f_, op, d.cross2,
                     d.cross2 != d.base_cross2 ? AccessMode::read_write : AccessMode::read);
          add_access(f_, op, d.diagonal, AccessMode::read);
          add_access(f_, op, d.offdiagonal1, AccessMode::read);
          add_access(f_, op, d.offdiagonal2, AccessMode::read);
          add_access(f_, op, d.chi2, AccessMode::read);
          add_access(f_, op, d.chi3, AccessMode::read);
          add_access(f_, op, d.target_w, AccessMode::read_write);
          add_access(f_, op, d.previous_w, AccessMode::write);
          add_access(f_, op, d.pml.sig, AccessMode::read);
          add_access(f_, op, d.pml.kap, AccessMode::read);
          add_access(f_, op, d.pml.siginv, AccessMode::read);

          if (fc.gv.dim == Dcyl) {
            ivec axis_begin = sub.little_owned_corner(ec);
            if (axis_begin.in_direction(R) == 0) {
              ConstitutiveUpdate axis = d;
              ivec axis_end = sub.big_corner();
              axis_end.set_direction(R, 0);
              axis.region = make_region(fc.gv, chunk, ec, cmp, axis_begin, axis_end);
              axis.region.variant_key =
                  d.region.variant_key &
                  ~(constitutive_one_offdiagonal | constitutive_two_offdiagonals |
                    constitutive_has_minus_p | constitutive_copy_w_previous);
              if (axis.primary != axis.base_primary)
                axis.region.variant_key |= constitutive_has_minus_p;
              axis.region.variant_key |= constitutive_axis_override;
              axis.base_cross1 = axis.base_cross2 = invalid_array();
              axis.cross1 = axis.cross2 = invalid_array();
              axis.offdiagonal1 = axis.offdiagonal2 = invalid_array();
              axis.cross1_stride = axis.cross2_stride = 0;
              axis.previous_w = invalid_array();
              plan_.eh_updates.push_back(axis);

              add_access(f_, op, axis.target, AccessMode::read_write);
              add_access(f_, op, axis.base_primary, AccessMode::read);
              add_access(f_, op, axis.primary,
                         axis.primary != axis.base_primary ? AccessMode::read_write
                                                           : AccessMode::read);
              add_access(f_, op, axis.diagonal, AccessMode::read);
              add_access(f_, op, axis.chi2, AccessMode::read);
              add_access(f_, op, axis.chi3, AccessMode::read);
              add_access(f_, op, axis.target_w, AccessMode::read_write);
              add_access(f_, op, axis.pml.sig, AccessMode::read);
              add_access(f_, op, axis.pml.kap, AccessMode::read);
              add_access(f_, op, axis.pml.siginv, AccessMode::read);
            }
          }
        }
    }
  }
  op.descriptor_count = uint32_t(plan_.eh_updates.size()) - op.descriptor_index;
  if (plan_.program != StepProgram::solve_cw) attach_source_span(op, ft2, true);
}

} // namespace

void append_polarization_update_group(fields &f, StepPlan &plan, Operation &op,
                                      const std::vector<PolarizationUpdate> &recurrences,
                                      const std::vector<PolarizationUpdate> &noise_additions) {
  append_polarization_update_group_impl(f, plan, op, recurrences, noise_additions);
}

bool operator==(const PolarizationUpdate &a, const PolarizationUpdate &b) {
  return polarization_updates_equal(a, b);
}

bool operator==(const CwStateRow &a, const CwStateRow &b) {
  return a.chunk == b.chunk && a.traversal_component == b.traversal_component &&
         a.storage_component == b.storage_component && a.family == b.family &&
         a.real_array == b.real_array && a.imag_array == b.imag_array &&
         same_region(a.owned_region, b.owned_region) && a.complex_offset == b.complex_offset &&
         a.complex_count == b.complex_count;
}

bool operator==(const CwUnpackPrelude &a, const CwUnpackPrelude &b) {
  return a.first_boundary == b.first_boundary && a.constitutive == b.constitutive &&
         a.second_boundary == b.second_boundary &&
         a.skip_w_components == b.skip_w_components &&
         a.invalidate_field_values == b.invalidate_field_values;
}

bool operator==(const CwStateLayout &a, const CwStateLayout &b) {
  if (a.rows != b.rows || a.zero_arrays.size() != b.zero_arrays.size() ||
      a.pack_accesses.size() != b.pack_accesses.size() ||
      a.unpack_accesses.size() != b.unpack_accesses.size() ||
      a.unpack_prelude != b.unpack_prelude || a.complex_count != b.complex_count ||
      a.real_count != b.real_count || a.vector_precision != b.vector_precision ||
      a.storage_fingerprint != b.storage_fingerprint ||
      a.coordinate_fingerprint != b.coordinate_fingerprint ||
      a.material_fingerprint != b.material_fingerprint || a.signature != b.signature)
    return false;
  for (size_t i = 0; i < a.zero_arrays.size(); ++i)
    if (!same_array_ref(a.zero_arrays[i], b.zero_arrays[i])) return false;
  for (size_t i = 0; i < a.pack_accesses.size(); ++i)
    if (!same_access(a.pack_accesses[i], b.pack_accesses[i])) return false;
  for (size_t i = 0; i < a.unpack_accesses.size(); ++i)
    if (!same_access(a.unpack_accesses[i], b.unpack_accesses[i])) return false;
  return true;
}

uint64_t compute_cw_state_layout_signature(const CwStateLayout &layout) {
  return cw_layout_signature(layout);
}

CwStateLayout build_cw_state_layout(fields &f) {
  CwStateLayout layout;
  layout.unpack_prelude.first_boundary = D_stuff;
  layout.unpack_prelude.constitutive = E_stuff;
  layout.unpack_prelude.second_boundary = E_stuff;
  layout.unpack_prelude.skip_w_components = true;
  layout.unpack_prelude.invalidate_field_values = true;
  layout.coordinate_fingerprint = coordinate_fingerprint(f);
  layout.material_fingerprint = material_fingerprint(f);

  if (!f.array_catalog) {
    layout.signature = compute_cw_state_layout_signature(layout);
    return layout;
  }

  std::vector<ArrayId> used_allocations;
  auto add_access_once = [&](std::vector<BufferAccess> &accesses, ArrayId id, AccessMode mode) {
    for (const BufferAccess &access : accesses)
      if (access.array.id == id) return;
    const ArraySpec &spec = f.array_catalog->spec(id);
    accesses.push_back(BufferAccess{ArrayRef{id, 0, spec.elements}, mode});
  };

  auto validate_array = [&](ArrayId id, const realnum *raw, int chunk, array_kind kind, component c,
                            int cmp, const UpdateRegion &region) {
    if (!is_valid(id) || id.value >= f.array_catalog->size())
      throw std::invalid_argument("solve_cw state row is absent from the array catalog");
    const StorageKey &key = f.array_catalog->key(id);
    const ArraySpec &spec = f.array_catalog->spec(id);
    if (key.chunk != chunk || key.kind != int(kind) || key.component_ != int(c) || key.cmp != cmp ||
        key.aux != 0)
      throw std::invalid_argument("solve_cw state row has the wrong catalog identity");
    if (spec.role != array_role::field || spec.element_type != ElementType::realnum_value ||
        spec.storage != layout.vector_precision)
      throw std::invalid_argument("solve_cw state row has incompatible storage");
    if (f.array_catalog->resolve_untyped(id) != raw)
      throw std::invalid_argument("solve_cw state row has a stale catalog binding");
    if (spec.elements != size_t(f.chunks[chunk]->gv.ntot()))
      throw std::invalid_argument("solve_cw state row has the wrong allocation extent");

    size_t last = region.base;
    for (int axis = 0; axis < 3; ++axis) {
      if (region.strides[axis] < 0)
        throw std::invalid_argument("solve_cw state row has a negative stride");
      if (!region.counts[axis]) continue;
      const size_t tail = checked_product(region.counts[axis] - 1,
                                          size_t(region.strides[axis]),
                                          "solve_cw state region extent overflow");
      last = checked_sum(last, tail, "solve_cw state region extent overflow");
    }
    if (region.counts[0] && region.counts[1] && region.counts[2] && last >= spec.elements)
      throw std::invalid_argument("solve_cw state region exceeds its backing allocation");
  };

  auto append_pair = [&](int chunk, fields_chunk &fc, component traversal, component storage,
                         CwStateFamily family, array_kind kind, realnum *raw_real,
                         realnum *raw_imag, bool primary_present) {
    if ((raw_real != NULL) != (raw_imag != NULL))
      throw std::invalid_argument("solve_cw state contains a real/imaginary half-pair");
    if (!raw_real) return;
    if (!primary_present)
      throw std::invalid_argument("solve_cw optional state exists without its primary field");
    const ArrayId real_id = find_array(f, chunk, kind, int(storage), 0, 0);
    const ArrayId imag_id = find_array(f, chunk, kind, int(storage), 1, 0);
    if ((!is_valid(real_id) || !is_valid(imag_id)) && is_dirty(f, dirty_storage)) return;
    if (!is_valid(real_id) || !is_valid(imag_id))
      throw std::invalid_argument("solve_cw live state is absent from the array catalog");

    const ivec begin = fc.gv.little_owned_corner(traversal);
    const ivec end = fc.gv.big_corner();
    const UpdateRegion region = make_region(fc.gv, chunk, traversal, -1, begin, end);
    size_t count = 1;
    for (int axis = 0; axis < 3; ++axis)
      count = checked_product(count, region.counts[axis], "solve_cw state row count overflow");
    if (count != size_t(fc.gv.nowned(traversal)))
      throw std::invalid_argument("solve_cw state row does not match LOOP_OVER_VOL_OWNED");

    validate_array(real_id, raw_real, chunk, kind, storage, 0, region);
    validate_array(imag_id, raw_imag, chunk, kind, storage, 1, region);
    const ArrayId canonical_real = canonical_array(*f.array_catalog, real_id);
    const ArrayId canonical_imag = canonical_array(*f.array_catalog, imag_id);
    if (canonical_real == canonical_imag)
      throw std::invalid_argument("solve_cw state real and imaginary arrays alias");
    for (ArrayId used : used_allocations)
      if (used == canonical_real || used == canonical_imag)
        throw std::invalid_argument("solve_cw state rows alias the same allocation");

    if (!count) return;
    used_allocations.push_back(canonical_real);
    used_allocations.push_back(canonical_imag);
    CwStateRow row;
    row.chunk = chunk;
    row.traversal_component = traversal;
    row.storage_component = storage;
    row.family = family;
    row.real_array = real_id;
    row.imag_array = imag_id;
    row.owned_region = region;
    row.complex_offset = layout.complex_count;
    row.complex_count = count;
    layout.complex_count = checked_sum(layout.complex_count, count,
                                       "solve_cw state vector length overflow");
    layout.rows.push_back(row);
    add_access_once(layout.pack_accesses, real_id, AccessMode::read);
    add_access_once(layout.pack_accesses, imag_id, AccessMode::read);
    add_access_once(layout.unpack_accesses, real_id, AccessMode::write);
    add_access_once(layout.unpack_accesses, imag_id, AccessMode::write);
  };

  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f.chunks[chunk];
    FOR_COMPONENTS(c) {
      if (!is_D(c) && !is_B(c)) continue;
      realnum *const primary_real = fc.f[c][0];
      realnum *const primary_imag = fc.f[c][1];
      if ((primary_real != NULL) != (primary_imag != NULL))
        throw std::invalid_argument("solve_cw primary state contains a real/imaginary half-pair");
      const bool primary_present = primary_real && primary_imag;
      append_pair(chunk, fc, c, c, CwStateFamily::primary, array_kind::f, primary_real,
                  primary_imag, primary_present);
      append_pair(chunk, fc, c, c, CwStateFamily::pml_u, array_kind::f_u, fc.f_u[c][0],
                  fc.f_u[c][1], primary_present);
      append_pair(chunk, fc, c, c, CwStateFamily::conductivity, array_kind::f_cond,
                  fc.f_cond[c][0], fc.f_cond[c][1], primary_present);
      append_pair(chunk, fc, c, c, CwStateFamily::bfast, array_kind::f_bfast,
                  fc.f_bfast[c][0], fc.f_bfast[c][1], primary_present);

      const component paired = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
      realnum *const w_real = fc.f_w[paired][0];
      realnum *const w_imag = fc.f_w[paired][1];
      if ((w_real != NULL) != (w_imag != NULL))
        throw std::invalid_argument("solve_cw constitutive state contains a real/imaginary half-pair");
      append_pair(chunk, fc, c, paired, CwStateFamily::constitutive_w, array_kind::f_w,
                  w_real, w_imag, primary_present);
      if (w_real)
        append_pair(chunk, fc, c, paired, CwStateFamily::paired_primary, array_kind::f,
                    fc.f[paired][0], fc.f[paired][1], primary_present);
    }
  }

  layout.real_count = checked_product(layout.complex_count, size_t(2),
                                      "solve_cw real vector length overflow");

  const int zero_kinds[] = {
      int(array_kind::f),           int(array_kind::f_u),          int(array_kind::f_w),
      int(array_kind::f_cond),      int(array_kind::f_bfast),      int(array_kind::f_backup),
      int(array_kind::f_u_backup),  int(array_kind::f_w_backup),   int(array_kind::f_cond_backup),
      int(array_kind::f_bfast_backup)};
  std::vector<ArrayId> zero_ids;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f.chunks[chunk];
    realnum *(*families[])[2] = {fc.f,          fc.f_u,          fc.f_w,          fc.f_cond,
                                 fc.f_bfast,    fc.f_backup,     fc.f_u_backup,   fc.f_w_backup,
                                 fc.f_cond_backup, fc.f_bfast_backup};
    for (size_t family = 0; family < sizeof(zero_kinds) / sizeof(zero_kinds[0]); ++family)
      FOR_COMPONENTS(c) for (int cmp = 0; cmp < 2; ++cmp) {
        realnum *const raw = families[family][c][cmp];
        if (!raw) continue;
        const ArrayId id = find_array(f, chunk, array_kind(zero_kinds[family]), int(c), cmp, 0);
        if (!is_valid(id) && is_dirty(f, dirty_storage)) continue;
        if (!is_valid(id))
          throw std::invalid_argument("solve_cw zeroed field is absent from the array catalog");
        const ArraySpec &spec = f.array_catalog->spec(id);
        if (f.array_catalog->resolve_untyped(id) != raw || spec.role != array_role::field ||
            spec.element_type != ElementType::realnum_value ||
            spec.storage != layout.vector_precision || spec.elements != size_t(fc.gv.ntot()))
          throw std::invalid_argument("solve_cw zero set has incompatible catalog storage");
        zero_ids.push_back(canonical_array(*f.array_catalog, id));
      }
  }
  std::sort(zero_ids.begin(), zero_ids.end(),
            [](ArrayId a, ArrayId b) { return a.value < b.value; });
  zero_ids.erase(std::unique(zero_ids.begin(), zero_ids.end(),
                             [](ArrayId a, ArrayId b) { return a == b; }),
                 zero_ids.end());
  for (ArrayId id : zero_ids) {
    const ArraySpec &spec = f.array_catalog->spec(id);
    if (spec.role != array_role::field || spec.element_type != ElementType::realnum_value ||
        spec.storage != layout.vector_precision)
      throw std::invalid_argument("solve_cw canonical zero allocation has incompatible storage");
    layout.zero_arrays.push_back(ArrayRef{id, 0, spec.elements});
  }

  layout.storage_fingerprint = storage_fingerprint(f, layout);
  layout.signature = compute_cw_state_layout_signature(layout);
  return layout;
}

bool validate_cw_state_layout(fields &f, const CwStateLayout &layout, std::string *error) {
  if (error) error->clear();
  try {
    const CwStateLayout expected = build_cw_state_layout(f);
    if (layout == expected) return true;
    if (error) *error = "solve_cw state layout differs from the canonical layout";
  }
  catch (const std::exception &e) {
    if (error) *error = e.what();
  }
  return false;
}

void CwPlan::clear() {
  state_layout_signature = 0;
  step_plan_signature = 0;
  rhs_stages.clear();
  rhs_sources.clear();
  unpack = CwUnpackDescriptorRefs();
  final_dfts.clear();
  rhs_accesses.clear();
  unpack_accesses.clear();
  final_dft_accesses.clear();
  source_time_count = 0;
  rhs_source_count = 0;
  final_dft_count = 0;
  source_fingerprint = 0;
  monitor_fingerprint = 0;
  signature = 0;
}

bool operator==(const CwRhsSourceDescriptor &a, const CwRhsSourceDescriptor &b) {
  return a.source_descriptor_index == b.source_descriptor_index &&
         a.source_ordinal == b.source_ordinal && a.mode == b.mode;
}

bool operator==(const CwStepOperationRef &a, const CwStepOperationRef &b) {
  return a.operation_index == b.operation_index && a.kind == b.kind && a.ft == b.ft &&
         a.descriptor_index == b.descriptor_index && a.descriptor_count == b.descriptor_count &&
         a.polarization_subtraction_index == b.polarization_subtraction_index &&
         a.polarization_subtraction_count == b.polarization_subtraction_count;
}

bool operator==(const CwRhsStage &a, const CwRhsStage &b) {
  return a.ft == b.ft && a.source_time_offset == b.source_time_offset &&
         a.source_time_index == b.source_time_index &&
         a.source_time_count == b.source_time_count &&
         a.source_index == b.source_index && a.source_count == b.source_count &&
         a.boundary == b.boundary && a.constitutive == b.constitutive &&
         a.accesses.size() == b.accesses.size() &&
         std::equal(a.accesses.begin(), a.accesses.end(), b.accesses.begin(), same_access);
}

bool operator==(const CwUnpackDescriptorRefs &a, const CwUnpackDescriptorRefs &b) {
  return a.first_boundary == b.first_boundary && a.constitutive == b.constitutive &&
         a.second_boundary == b.second_boundary &&
         a.skip_w_components == b.skip_w_components &&
         a.invalidate_field_values == b.invalidate_field_values;
}

bool operator==(const CwDftDescriptorRef &a, const CwDftDescriptorRef &b) {
  return a.descriptor_index == b.descriptor_index && a.chunk == b.chunk && a.c == b.c &&
         a.decimation_factor == b.decimation_factor && a.due_scalar_slot == b.due_scalar_slot;
}

bool operator==(const CwPlan &a, const CwPlan &b) {
  if (a.state_layout_signature != b.state_layout_signature ||
      a.step_plan_signature != b.step_plan_signature || a.rhs_stages != b.rhs_stages ||
      a.rhs_sources != b.rhs_sources || a.unpack != b.unpack || a.final_dfts != b.final_dfts ||
      a.rhs_accesses.size() != b.rhs_accesses.size() ||
      a.unpack_accesses.size() != b.unpack_accesses.size() ||
      a.final_dft_accesses.size() != b.final_dft_accesses.size() ||
      a.source_time_count != b.source_time_count || a.rhs_source_count != b.rhs_source_count ||
      a.final_dft_count != b.final_dft_count ||
      a.source_fingerprint != b.source_fingerprint ||
      a.monitor_fingerprint != b.monitor_fingerprint || a.signature != b.signature)
    return false;
  for (size_t i = 0; i < a.rhs_accesses.size(); ++i)
    if (!same_access(a.rhs_accesses[i], b.rhs_accesses[i])) return false;
  for (size_t i = 0; i < a.unpack_accesses.size(); ++i)
    if (!same_access(a.unpack_accesses[i], b.unpack_accesses[i])) return false;
  for (size_t i = 0; i < a.final_dft_accesses.size(); ++i)
    if (!same_access(a.final_dft_accesses[i], b.final_dft_accesses[i])) return false;
  return true;
}

namespace {

void cw_hash_ref(uint64_t &sig, const ArrayRef &ref) {
  target_fingerprint_mix(sig, ref.id.value);
  target_fingerprint_mix(sig, ref.offset);
  target_fingerprint_mix(sig, ref.elements);
}

void cw_hash_access(uint64_t &sig, const BufferAccess &access) {
  cw_hash_ref(sig, access.array);
  target_fingerprint_mix(sig, uint64_t(access.mode));
}

void cw_hash_operation_ref(uint64_t &sig, const CwStepOperationRef &ref) {
  target_fingerprint_mix(sig, ref.operation_index);
  target_fingerprint_mix(sig, uint64_t(ref.kind));
  target_fingerprint_mix(sig, uint64_t(ref.ft));
  target_fingerprint_mix(sig, ref.descriptor_index);
  target_fingerprint_mix(sig, ref.descriptor_count);
  target_fingerprint_mix(sig, ref.polarization_subtraction_index);
  target_fingerprint_mix(sig, ref.polarization_subtraction_count);
}

CwStepOperationRef cw_operation_ref(const StepPlan &plan, uint32_t operation_index,
                                    OpKind kind, field_type ft) {
  if (operation_index >= plan.operations.size())
    throw std::invalid_argument("CW operation reference is out of range");
  const Operation &op = plan.operations[operation_index];
  if (op.kind != kind || op.ft != ft)
    throw std::invalid_argument("CW operation reference has the wrong kind or field type");
  CwStepOperationRef ref;
  ref.operation_index = operation_index;
  ref.kind = op.kind;
  ref.ft = op.ft;
  ref.descriptor_index = op.descriptor_index;
  ref.descriptor_count = op.descriptor_count;
  ref.polarization_subtraction_index = op.polarization_subtraction_index;
  ref.polarization_subtraction_count = op.polarization_subtraction_count;
  return ref;
}

uint32_t cw_find_operation(const StepPlan &plan, uint32_t begin, OpKind kind, field_type ft) {
  for (size_t i = begin; i < plan.operations.size(); ++i)
    if (plan.operations[i].kind == kind && plan.operations[i].ft == ft) {
      if (i > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("CW operation index overflow");
      return uint32_t(i);
    }
  throw std::invalid_argument("canonical solve_cw plan is missing a required operation");
}

void cw_add_access(fields &f, std::vector<BufferAccess> &accesses, ArrayId id, AccessMode mode) {
  if (!is_valid(id)) return;
  if (!f.array_catalog || id.value >= f.array_catalog->size())
    throw std::invalid_argument("CW plan access names an invalid array");
  id = canonical_array(*f.array_catalog, id);
  const ArraySpec &spec = f.array_catalog->spec(id);
  for (BufferAccess &access : accesses) {
    if (access.array.id != id) continue;
    if (access.array.offset != 0 || access.array.elements != spec.elements)
      throw std::invalid_argument("CW plan contains inconsistent access spans");
    if (access.mode != mode) access.mode = AccessMode::read_write;
    return;
  }
  accesses.push_back(BufferAccess{ArrayRef{id, 0, spec.elements}, mode});
}

void cw_add_access_ref(fields &f, std::vector<BufferAccess> &accesses, const ArrayRef &ref,
                       AccessMode mode) {
  if (!is_valid(ref.id)) return;
  if (!f.array_catalog || ref.id.value >= f.array_catalog->size())
    throw std::invalid_argument("CW final DFT access names an invalid array");
  const ArraySpec &original_spec = f.array_catalog->spec(ref.id);
  if (ref.offset != 0 || ref.elements != original_spec.elements)
    throw std::invalid_argument("CW plan access is not a full catalog allocation");
  ArrayRef canonical = ref;
  canonical.id = canonical_array(*f.array_catalog, ref.id);
  const ArraySpec &canonical_spec = f.array_catalog->spec(canonical.id);
  if (canonical_spec.elements != original_spec.elements || canonical.offset != 0 ||
      canonical.elements != canonical_spec.elements)
    throw std::invalid_argument("CW final DFT alias has an incompatible extent");
  for (BufferAccess &access : accesses) {
    if (access.array.id != canonical.id) continue;
    if (!same_array_ref(access.array, canonical))
      throw std::invalid_argument("CW final DFT contains inconsistent access spans");
    if (access.mode != mode) access.mode = AccessMode::read_write;
    return;
  }
  accesses.push_back(BufferAccess{canonical, mode});
}

void cw_merge_accesses(fields &f, std::vector<BufferAccess> &destination,
                       const std::vector<BufferAccess> &source) {
  for (const BufferAccess &access : source) {
    if (!is_valid(access.array.id) || access.array.id.value >= f.array_catalog->size())
      throw std::invalid_argument("referenced CW operation has an invalid access");
    const ArraySpec &spec = f.array_catalog->spec(access.array.id);
    if (access.array.offset != 0 || access.array.elements != spec.elements)
      throw std::invalid_argument("referenced CW operation access is not full-allocation");
    cw_add_access(f, destination, access.array.id, access.mode);
  }
}

uint32_t cw_source_time_id(const fields &f, const src_time *wanted) {
  size_t id = 0;
  for (const src_time *st = f.sources; st; st = st->next, ++id)
    if (st == wanted) {
      if (id > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("CW source-time index overflow");
      return uint32_t(id);
    }
  throw std::invalid_argument("CW source is absent from the source-time list");
}

size_t cw_find_source_descriptor(const SourcePlan &plan, int chunk, field_type ft,
                                 uint32_t ordinal) {
  size_t found = plan.sources.size();
  for (size_t i = 0; i < plan.sources.size(); ++i) {
    const SourceDescriptor &d = plan.sources[i];
    if (d.chunk != chunk || d.ft != ft || d.source_ordinal != ordinal) continue;
    if (found != plan.sources.size())
      throw std::invalid_argument("CW source plan contains a duplicate source ordinal");
    found = i;
  }
  if (found == plan.sources.size())
    throw std::invalid_argument("CW source plan is missing a live source ordinal");
  return found;
}

bool cw_same_source_time(const SourceTimeDescriptor &a, const SourceTimeDescriptor &b) {
  return a.source_time_id == b.source_time_id && a.kind == b.kind &&
         a.parameters == b.parameters && a.scalar_slot == b.scalar_slot &&
         a.host_callback_id == b.host_callback_id && a.is_integrated == b.is_integrated;
}

bool cw_same_source_descriptor(const SourceDescriptor &a, const SourceDescriptor &b) {
  return a.destination == b.destination && a.destination_imag == b.destination_imag &&
         a.integrated_destination == b.integrated_destination &&
         a.integrated_destination_imag == b.integrated_destination_imag && a.chunk == b.chunk &&
         a.c == b.c && a.indices == b.indices && a.complex_amplitudes == b.complex_amplitudes &&
         a.condinv == b.condinv && a.source_time_id == b.source_time_id &&
         a.source_ordinal == b.source_ordinal && a.integrated == b.integrated && a.ft == b.ft;
}

bool cw_same_source_plan_recipe(const SourcePlan &a, const SourcePlan &b) {
  if (a.source_times.size() != b.source_times.size() || a.sources.size() != b.sources.size())
    return false;
  for (size_t i = 0; i < a.source_times.size(); ++i)
    if (!cw_same_source_time(a.source_times[i], b.source_times[i])) return false;
  for (size_t i = 0; i < a.sources.size(); ++i)
    if (!cw_same_source_descriptor(a.sources[i], b.sources[i])) return false;
  return true;
}

bool cw_same_ivec(const ivec &a, const ivec &b) {
  if (a.dim != b.dim) return false;
  LOOP_OVER_DIRECTIONS(a.dim, d) {
    if (a.in_direction(d) != b.in_direction(d)) return false;
  }
  return true;
}

bool cw_same_vec(const vec &a, const vec &b) {
  if (a.dim != b.dim) return false;
  LOOP_OVER_DIRECTIONS(a.dim, d) {
    if (a.in_direction(d) != b.in_direction(d)) return false;
  }
  return true;
}

bool cw_same_dft(const DftDescriptor &a, const DftDescriptor &b) {
  return a.accumulator == b.accumulator && a.phase_scratch == b.phase_scratch &&
         same_array_ref(a.source_field, b.source_field) &&
         same_array_ref(a.source_field_imag, b.source_field_imag) && a.omega == b.omega &&
         a.scale == b.scale && a.chunk == b.chunk && a.c == b.c && a.avg1 == b.avg1 &&
         a.avg2 == b.avg2 && cw_same_ivec(a.is, b.is) && cw_same_ivec(a.ie, b.ie) &&
         cw_same_ivec(a.is_old, b.is_old) && cw_same_ivec(a.ie_old, b.ie_old) &&
         a.persist == b.persist && a.decimation_factor == b.decimation_factor &&
         a.due_scalar_slot == b.due_scalar_slot && cw_same_vec(a.weights.s0, b.weights.s0) &&
         cw_same_vec(a.weights.s1, b.weights.s1) && cw_same_vec(a.weights.e0, b.weights.e0) &&
         cw_same_vec(a.weights.e1, b.weights.e1) && a.dV0 == b.dV0 && a.dV1 == b.dV1 &&
         a.include_dV_and_interp_weights == b.include_dV_and_interp_weights &&
         a.sqrt_dV_and_interp_weights == b.sqrt_dV_and_interp_weights && a.N == b.N &&
         a.Nomega == b.Nomega;
}

void cw_validate_dft_storage(fields &f, const fields_chunk &fc, int chunk, int local_index,
                             const dft_chunk &live, const DftDescriptor &d) {
  const Precision native =
      sizeof(realnum) == sizeof(float) ? Precision::f32 : Precision::f64;
  const size_t accumulator_elements =
      checked_product(live.N, live.omega.size(), "CW final DFT extent overflow");
  const ArrayId accumulator =
      f.array_catalog->find({chunk, int(array_kind::dft), int(live.c), -1, local_index});
  const ArrayId phase =
      f.array_catalog->find({chunk, int(array_kind::dft_phase), int(live.c), -1, local_index});
  const ArrayId source =
      f.array_catalog->find({chunk, int(array_kind::f), int(live.c), 0, 0});
  const ArrayId source_imag =
      f.array_catalog->find({chunk, int(array_kind::f), int(live.c), 1, 0});
  if (!is_valid(accumulator) || !is_valid(phase) || d.accumulator != accumulator ||
      d.phase_scratch != phase || d.source_field.id != source ||
      d.source_field_imag.id != source_imag)
    throw std::invalid_argument("CW final DFT has the wrong catalog identity");
  const ArraySpec &accumulator_spec = f.array_catalog->spec(accumulator);
  const ArraySpec &phase_spec = f.array_catalog->spec(phase);
  if (accumulator_spec.role != array_role::dft ||
      accumulator_spec.element_type != ElementType::complex_realnum ||
      accumulator_spec.storage != native || accumulator_spec.elements != accumulator_elements ||
      phase_spec.role != array_role::dft ||
      phase_spec.element_type != ElementType::complex_realnum || phase_spec.storage != native ||
      phase_spec.elements != live.omega.size())
    throw std::invalid_argument("CW final DFT has incompatible catalog storage");
  if (f.array_catalog->resolve_untyped(accumulator) != live.dft ||
      f.array_catalog->resolve_untyped(phase) != live.dft_phase)
    throw std::invalid_argument("CW final DFT has a stale catalog binding");

  const ArrayRef fields[] = {d.source_field, d.source_field_imag};
  realnum *const raw[] = {fc.f[live.c][0], fc.f[live.c][1]};
  for (size_t cmp = 0; cmp < 2; ++cmp) {
    if (!raw[cmp]) {
      if (is_valid(fields[cmp].id) || fields[cmp].elements != 0)
        throw std::invalid_argument("CW final DFT has a spurious source-field reference");
      continue;
    }
    if (!is_valid(fields[cmp].id) || fields[cmp].offset != 0 ||
        fields[cmp].elements != size_t(fc.gv.ntot()))
      throw std::invalid_argument("CW final DFT has the wrong source-field extent");
    const ArraySpec &spec = f.array_catalog->spec(fields[cmp].id);
    if (spec.role != array_role::field || spec.element_type != ElementType::realnum_value ||
        spec.storage != native || spec.elements != size_t(fc.gv.ntot()) ||
        f.array_catalog->resolve_untyped(fields[cmp].id) != raw[cmp])
      throw std::invalid_argument("CW final DFT source field has incompatible storage");
  }
}

void cw_validate_source_descriptor(fields &f, const fields_chunk &fc, int chunk, field_type ft,
                                   uint32_t ordinal, const src_vol &sv,
                                   const SourceDescriptor &d) {
  const component c = direction_component(first_field_component(ft), component_direction(sv.c));
  const ArrayId destination =
      f.array_catalog->find({chunk, int(array_kind::f), int(c), 0, 0});
  const ArrayId destination_imag =
      f.array_catalog->find({chunk, int(array_kind::f), int(c), 1, 0});
  const ArrayId condinv = f.array_catalog->find(
      {chunk, int(array_kind::condinv), int(c), -1, int(component_direction(sv.c))});
  const ArrayId integrated_destination =
      sv.t()->is_integrated
          ? f.array_catalog->find({chunk, int(array_kind::f_minus_p), int(c), 0, 0})
          : invalid_array();
  const ArrayId integrated_destination_imag =
      sv.t()->is_integrated
          ? f.array_catalog->find({chunk, int(array_kind::f_minus_p), int(c), 1, 0})
          : invalid_array();
  const Precision native =
      sizeof(realnum) == sizeof(float) ? Precision::f32 : Precision::f64;
  if (d.chunk != chunk || d.ft != ft || d.source_ordinal != ordinal || d.c != c ||
      d.integrated != sv.t()->is_integrated || d.destination != destination ||
      d.destination_imag != destination_imag || d.condinv != condinv ||
      d.integrated_destination != integrated_destination ||
      d.integrated_destination_imag != integrated_destination_imag ||
      d.source_time_id != cw_source_time_id(f, sv.t()) || d.indices.size() != sv.num_points() ||
      d.complex_amplitudes.size() != sv.num_points())
    throw std::invalid_argument("CW source descriptor differs from its live source");
  for (size_t i = 0; i < sv.num_points(); ++i)
    if (d.indices[i] != sv.index_at(i) || d.complex_amplitudes[i] != sv.amplitude_at(i))
      throw std::invalid_argument("CW source spatial table differs from its live source");
  const ArrayId destinations[] = {destination, destination_imag};
  realnum *const raw_destinations[] = {fc.f[c][0], fc.f[c][1]};
  for (size_t cmp = 0; cmp < 2; ++cmp) {
    if (is_valid(destinations[cmp]) != (raw_destinations[cmp] != NULL))
      throw std::invalid_argument("CW source destination catalog presence differs from live storage");
    if (!is_valid(destinations[cmp])) continue;
    const ArraySpec &spec = f.array_catalog->spec(destinations[cmp]);
    if (spec.role != array_role::field || spec.element_type != ElementType::realnum_value ||
        spec.storage != native || spec.elements != size_t(fc.gv.ntot()) ||
        f.array_catalog->resolve_untyped(destinations[cmp]) != raw_destinations[cmp])
      throw std::invalid_argument("CW source destination has incompatible catalog storage");
  }
  const realnum *raw_condinv = fc.s->condinv[c][component_direction(sv.c)];
  if (is_valid(condinv) != (raw_condinv != NULL))
    throw std::invalid_argument("CW source conductivity catalog presence differs from live storage");
  if (is_valid(condinv)) {
    const ArraySpec &spec = f.array_catalog->spec(condinv);
    if (spec.role != array_role::material ||
        spec.element_type != ElementType::realnum_value || spec.storage != native ||
        spec.elements != size_t(fc.gv.ntot()) ||
        f.array_catalog->resolve_untyped(condinv) != raw_condinv)
      throw std::invalid_argument("CW source conductivity has incompatible catalog storage");
  }
}

} // namespace

uint64_t compute_cw_plan_signature(const CwPlan &plan) {
  uint64_t sig = 0xcbf29ce484222325ull;
  target_fingerprint_mix(sig, plan.state_layout_signature);
  target_fingerprint_mix(sig, plan.step_plan_signature);
  target_fingerprint_mix(sig, plan.rhs_stages.size());
  for (const CwRhsStage &stage : plan.rhs_stages) {
    target_fingerprint_mix(sig, uint64_t(stage.ft));
    fingerprint_double(sig, stage.source_time_offset);
    target_fingerprint_mix(sig, stage.source_time_index);
    target_fingerprint_mix(sig, stage.source_time_count);
    target_fingerprint_mix(sig, stage.source_index);
    target_fingerprint_mix(sig, stage.source_count);
    cw_hash_operation_ref(sig, stage.boundary);
    cw_hash_operation_ref(sig, stage.constitutive);
    target_fingerprint_mix(sig, stage.accesses.size());
    for (const BufferAccess &access : stage.accesses)
      cw_hash_access(sig, access);
  }
  target_fingerprint_mix(sig, plan.rhs_sources.size());
  for (const CwRhsSourceDescriptor &source : plan.rhs_sources) {
    target_fingerprint_mix(sig, source.source_descriptor_index);
    target_fingerprint_mix(sig, source.source_ordinal);
    target_fingerprint_mix(sig, uint64_t(source.mode));
  }
  cw_hash_operation_ref(sig, plan.unpack.first_boundary);
  cw_hash_operation_ref(sig, plan.unpack.constitutive);
  cw_hash_operation_ref(sig, plan.unpack.second_boundary);
  target_fingerprint_mix(sig, plan.unpack.skip_w_components);
  target_fingerprint_mix(sig, plan.unpack.invalidate_field_values);
  target_fingerprint_mix(sig, plan.final_dfts.size());
  for (const CwDftDescriptorRef &dft : plan.final_dfts) {
    target_fingerprint_mix(sig, dft.descriptor_index);
    target_fingerprint_mix(sig, uint64_t(dft.chunk));
    target_fingerprint_mix(sig, uint64_t(dft.c));
    target_fingerprint_mix(sig, uint64_t(dft.decimation_factor));
    target_fingerprint_mix(sig, dft.due_scalar_slot);
  }
  target_fingerprint_mix(sig, plan.rhs_accesses.size());
  for (const BufferAccess &access : plan.rhs_accesses)
    cw_hash_access(sig, access);
  target_fingerprint_mix(sig, plan.unpack_accesses.size());
  for (const BufferAccess &access : plan.unpack_accesses)
    cw_hash_access(sig, access);
  target_fingerprint_mix(sig, plan.final_dft_accesses.size());
  for (const BufferAccess &access : plan.final_dft_accesses)
    cw_hash_access(sig, access);
  target_fingerprint_mix(sig, plan.source_fingerprint);
  target_fingerprint_mix(sig, plan.monitor_fingerprint);
  target_fingerprint_mix(sig, plan.source_time_count);
  target_fingerprint_mix(sig, plan.rhs_source_count);
  target_fingerprint_mix(sig, plan.final_dft_count);
  return sig;
}

CwPlan build_cw_plan(fields &f, const StepPlan &step_plan) {
  if (step_plan.program != StepProgram::solve_cw)
    throw std::invalid_argument("CwPlan requires a solve_cw StepPlan");
  if (step_plan.signature != compute_step_plan_signature(step_plan))
    throw std::invalid_argument("CwPlan received a stale solve_cw StepPlan signature");
  const StepPlan canonical_step_plan = build_step_plan(f, StepProgram::solve_cw);
  if (step_plan.signature != canonical_step_plan.signature)
    throw std::invalid_argument("CwPlan received a noncanonical solve_cw StepPlan");
  std::string layout_error;
  if (!validate_cw_state_layout(f, step_plan.cw_state_layout, &layout_error))
    throw std::invalid_argument(layout_error);
  if (!f.descriptors || !f.array_catalog)
    throw std::invalid_argument("CwPlan requires prepared source/DFT descriptors and storage");

  SourcePlan canonical_sources;
  build_source_descriptors(f, canonical_sources);
  if (!cw_same_source_plan_recipe(f.descriptors->sources, canonical_sources))
    throw std::invalid_argument("CW source descriptors differ from the live source objects");
  std::vector<DftDescriptor> canonical_dfts;
  build_dft_descriptors(f, canonical_dfts);
  if (canonical_dfts.size() != f.descriptors->dfts.size())
    throw std::invalid_argument("CW DFT descriptor count differs from the live monitors");
  for (size_t i = 0; i < canonical_dfts.size(); ++i)
    if (!cw_same_dft(f.descriptors->dfts[i], canonical_dfts[i]))
      throw std::invalid_argument("CW DFT descriptor differs from its live monitor");

  CwPlan plan;
  plan.state_layout_signature = step_plan.cw_state_layout.signature;
  plan.step_plan_signature = step_plan.signature;
  plan.source_fingerprint = source_plan_signature(f.descriptors->sources);
  plan.monitor_fingerprint = dft_plan_signature(f.descriptors->dfts);
  if (f.descriptors->sources.source_times.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("CW source-time span overflow");
  plan.source_time_count = uint32_t(f.descriptors->sources.source_times.size());
  for (const ArrayRef &zero : step_plan.cw_state_layout.zero_arrays)
    cw_add_access(f, plan.rhs_accesses, zero.id, AccessMode::write);

  const uint32_t b_update = cw_find_operation(step_plan, 0, OpKind::update_db, B_stuff);
  const uint32_t b_boundary =
      cw_find_operation(step_plan, b_update + 1, OpKind::transfer_halo, B_stuff);
  const uint32_t h_update =
      cw_find_operation(step_plan, b_boundary + 1, OpKind::update_eh, H_stuff);
  const uint32_t d_update =
      cw_find_operation(step_plan, h_update + 1, OpKind::update_db, D_stuff);
  const uint32_t d_boundary =
      cw_find_operation(step_plan, d_update + 1, OpKind::transfer_halo, D_stuff);
  const uint32_t e_update =
      cw_find_operation(step_plan, d_boundary + 1, OpKind::update_eh, E_stuff);
  const uint32_t e_boundary =
      cw_find_operation(step_plan, e_update + 1, OpKind::transfer_halo, E_stuff);

  const field_type stage_ft[] = {B_stuff, D_stuff};
  const double stage_offset[] = {0.0, 0.5};
  const uint32_t stage_boundary[] = {b_boundary, d_boundary};
  const uint32_t stage_constitutive[] = {h_update, e_update};
  std::vector<uint8_t> consumed(f.descriptors->sources.sources.size(), 0);
  for (size_t stage_index = 0; stage_index < 2; ++stage_index) {
    CwRhsStage stage;
    stage.ft = stage_ft[stage_index];
    stage.source_time_offset = stage_offset[stage_index];
    stage.source_time_index = 0;
    stage.source_time_count = plan.source_time_count;
    if (plan.rhs_sources.size() > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("CW RHS source index overflow");
    stage.source_index = uint32_t(plan.rhs_sources.size());
    stage.boundary = cw_operation_ref(step_plan, stage_boundary[stage_index],
                                      OpKind::transfer_halo, stage.ft);
    const field_type constitutive_ft = stage.ft == B_stuff ? H_stuff : E_stuff;
    stage.constitutive = cw_operation_ref(step_plan, stage_constitutive[stage_index],
                                          OpKind::update_eh, constitutive_ft);

    for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
      if (!f.chunks[chunk]->is_mine()) continue;
      fields_chunk &fc = *f.chunks[chunk];
      const std::vector<src_vol> &sources = fc.get_sources(stage.ft);
      if (sources.size() > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("CW per-chunk source ordinal overflow");
      for (uint32_t ordinal = 0; ordinal < sources.size(); ++ordinal) {
        const src_vol &sv = sources[ordinal];
        const size_t descriptor_index =
            cw_find_source_descriptor(f.descriptors->sources, chunk, stage.ft, ordinal);
        if (descriptor_index > std::numeric_limits<uint32_t>::max())
          throw std::overflow_error("CW source descriptor index overflow");
        const SourceDescriptor &d = f.descriptors->sources.sources[descriptor_index];
        cw_validate_source_descriptor(f, fc, chunk, stage.ft, ordinal, sv, d);
        consumed[descriptor_index] = 1;

        const bool family_matches =
            (stage.ft == B_stuff && is_magnetic(sv.c)) ||
            (stage.ft == D_stuff && is_electric(sv.c));
        if (!family_matches || !fc.f[d.c][0]) continue;
        if (!is_valid(d.destination))
          throw std::invalid_argument("CW RHS source has no primary D/B destination");

        CwRhsSourceDescriptor rhs;
        rhs.source_descriptor_index = uint32_t(descriptor_index);
        rhs.source_ordinal = ordinal;
        rhs.mode = CwRhsSourceMode::primary_subtract_current_dt_including_integrated;
        plan.rhs_sources.push_back(rhs);
        cw_add_access(f, stage.accesses, d.destination, AccessMode::read_write);
        cw_add_access(f, stage.accesses, d.destination_imag, AccessMode::read_write);
        cw_add_access(f, stage.accesses, d.condinv, AccessMode::read);
      }
    }
    const size_t source_count = plan.rhs_sources.size() - stage.source_index;
    if (source_count > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("CW RHS source span overflow");
    stage.source_count = uint32_t(source_count);
    cw_merge_accesses(f, stage.accesses, step_plan.operations[stage_boundary[stage_index]].accesses);
    cw_merge_accesses(f, stage.accesses,
                      step_plan.operations[stage_constitutive[stage_index]].accesses);
    cw_merge_accesses(f, plan.rhs_accesses, stage.accesses);
    plan.rhs_stages.push_back(stage);
  }

  /* SourcePlan also contains empty/non-applicable field-family rows. Validate
     those against the live vectors so a stale descriptor cannot hide outside
     the two emitted RHS spans. */
  FOR_FIELD_TYPES(ft) {
    if (ft == B_stuff || ft == D_stuff) continue;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
      if (!f.chunks[chunk]->is_mine()) continue;
      fields_chunk &fc = *f.chunks[chunk];
      const std::vector<src_vol> &sources = fc.get_sources(ft);
      if (sources.size() > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("CW per-chunk source ordinal overflow");
      for (uint32_t ordinal = 0; ordinal < sources.size(); ++ordinal) {
        const size_t descriptor_index =
            cw_find_source_descriptor(f.descriptors->sources, chunk, ft, ordinal);
        cw_validate_source_descriptor(f, fc, chunk, ft, ordinal, sources[ordinal],
                                      f.descriptors->sources.sources[descriptor_index]);
        consumed[descriptor_index] = 1;
      }
    }
  }
  for (uint8_t row_consumed : consumed)
    if (!row_consumed)
      throw std::invalid_argument("CW source plan contains a non-live source descriptor");

  plan.unpack.first_boundary =
      cw_operation_ref(step_plan, d_boundary, OpKind::transfer_halo, D_stuff);
  plan.unpack.constitutive =
      cw_operation_ref(step_plan, e_update, OpKind::update_eh, E_stuff);
  plan.unpack.second_boundary =
      cw_operation_ref(step_plan, e_boundary, OpKind::transfer_halo, E_stuff);
  plan.unpack.skip_w_components = step_plan.cw_state_layout.unpack_prelude.skip_w_components;
  plan.unpack.invalidate_field_values =
      step_plan.cw_state_layout.unpack_prelude.invalidate_field_values;
  for (const BufferAccess &access : step_plan.cw_state_layout.unpack_accesses)
    cw_add_access(f, plan.unpack_accesses, access.array.id, access.mode);
  cw_merge_accesses(f, plan.unpack_accesses, step_plan.operations[d_boundary].accesses);
  cw_merge_accesses(f, plan.unpack_accesses, step_plan.operations[e_update].accesses);
  cw_merge_accesses(f, plan.unpack_accesses, step_plan.operations[e_boundary].accesses);

  if (f.descriptors->dfts.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("CW final DFT descriptor span overflow");
  size_t live_dft_index = 0;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    int local_index = 0;
    for (dft_chunk *live = f.chunks[chunk]->dft_chunks; live;
         live = live->next_in_chunk, ++local_index, ++live_dft_index)
      cw_validate_dft_storage(f, *f.chunks[chunk], chunk, local_index, *live,
                              f.descriptors->dfts[live_dft_index]);
  }
  if (live_dft_index != f.descriptors->dfts.size())
    throw std::invalid_argument("CW final DFT order differs from the live monitor order");
  for (size_t i = 0; i < f.descriptors->dfts.size(); ++i) {
    const DftDescriptor &d = f.descriptors->dfts[i];
    if (d.decimation_factor < 1)
      throw std::invalid_argument("CW final DFT has an invalid decimation factor");
    plan.final_dfts.push_back(CwDftDescriptorRef{uint32_t(i), d.chunk, d.c,
                                                 d.decimation_factor, d.due_scalar_slot});
    cw_add_access(f, plan.final_dft_accesses, d.accumulator, AccessMode::read_write);
    cw_add_access(f, plan.final_dft_accesses, d.phase_scratch, AccessMode::write);
    cw_add_access_ref(f, plan.final_dft_accesses, d.source_field, AccessMode::read);
    cw_add_access_ref(f, plan.final_dft_accesses, d.source_field_imag, AccessMode::read);
  }

  if (plan.rhs_sources.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("CW RHS source total overflow");
  if (plan.final_dfts.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("CW final DFT total overflow");
  plan.rhs_source_count = uint32_t(plan.rhs_sources.size());
  plan.final_dft_count = uint32_t(plan.final_dfts.size());

  plan.signature = compute_cw_plan_signature(plan);
  return plan;
}

bool validate_cw_plan(fields &f, const StepPlan &step_plan, const CwPlan &plan,
                      std::string *error) {
  if (error) error->clear();
  try {
    /* Validate the caller-supplied plan itself before replacing it with the
       independently rebuilt canonical plan below. This catches both stale
       signatures and structurally mutated/re-signed timestep plans. */
    (void)build_cw_plan(f, step_plan);
    const StepPlan canonical_step_plan = build_step_plan(f, StepProgram::solve_cw);
    if (step_plan.signature != canonical_step_plan.signature)
      throw std::invalid_argument("solve_cw StepPlan differs from the canonical live plan");
    const CwPlan expected = build_cw_plan(f, canonical_step_plan);
    if (plan == expected) return true;
    if (error) *error = "CwPlan differs from the canonical source/monitor plan";
  }
  catch (const std::exception &e) {
    if (error) *error = e.what();
  }
  return false;
}

uint64_t compute_material_phase_target_signature(const fields &f) {
  return material_phase_target_signature(f);
}

uint64_t compute_step_plan_signature(const StepPlan &plan) {
  return StepPlanBuilder::signature_for(plan);
}

/* Transcribed from fields::step_once. Read the two side by side.
 *
 * step_boundaries() begins with zero_metal for every owned chunk and then does
 * pack/transfer/unpack, which is why add_boundaries emits zero_boundary first.
 */
StepPlan build_step_plan(fields &f, StepProgram program) {
  StepPlanBuilder p(f, program);
  const bool cw = program == StepProgram::solve_cw;
  if (cw) {
    p.set_cw_state_layout(build_cw_state_layout(f));
    p.add_cw_state_marker(OpKind::unpack_state);
  }

  const bool has_sources = f.sources != NULL;
  const bool has_fluxes = f.fluxes != NULL;
  const bool phasing = f.is_phasing();
  /* update_dfts is disabled under solve_cw (see dft.cpp), so the operation is
     statically absent rather than guarded.

     dft_chunks are per-chunk, so a rank owning no chunk that intersects a
     monitor builds a program of a different shape from its peers. On CPU that
     is harmless -- update_dfts() is a no-op with no chunks -- and it is
     deliberately left alone.

     Reducing it with or_to_all here DEADLOCKS, and it is worth being precise
     about why, because it is the plan's section 6.4 hazard showing up for
     real. build_step_plan runs from step_plan_for(), which rebuilds when
     dirty_executable is set -- and dirty_executable can be set on a *subset*
     of ranks, because the lazy-allocation sites in step_db, update_eh and
     update_pols are rank-local by design (each returns a flag saying "I
     allocated, reconnect"). One rank rebuilds, enters the reduction, and waits
     for peers that never arrive. Observed: tests/flux hung indefinitely at
     np=2 with the reduction in place.

     The real fix is to make the *rebuild decision* collective, the way
     connect_chunks() gates on sync_chunk_connections(). That is follow-up
     work, recorded in ~/meep-phase1-pr5.md -- and it applies equally to the
     collectives PR 4 put inside classify(). */
  bool has_dfts = false;
  for (int i = 0; i < f.num_chunks && !has_dfts; ++i)
    if (f.chunks[i]->is_mine() && f.chunks[i]->dft_chunks) has_dfts = true;

  /* Magnetic re-synchronization is a graph_variant: the whole program differs
     when synchronized fields are active. */
  p.add_magnetic_marker(OpKind::restore_magnetic_fields, AccessMode::write);

  /* phase_material's conditional E/H reconciliation is a segment_boundary: the
     condition is an or_to_all over all chunks, so the host has to evaluate it
     between device segments. */
  if (phasing) {
    p.add_material_refresh(OpKind::phase_material);
    p.add_source_evaluation(guard_segment(0), 0.5);
    p.add_eh(H_stuff, guard_segment(0));
    p.add_boundaries(H_stuff, guard_segment(0));
    p.add_source_evaluation(guard_segment(0), 1.0);
    p.add_eh(E_stuff, guard_segment(0));
    p.add_boundaries(E_stuff, guard_segment(0));
  }

  p.add_material_refresh(OpKind::update_material_coefficients);

  uint32_t magnetic_evaluate_b = UINT32_MAX;
  if (has_sources) {
    magnetic_evaluate_b = p.operation_count();
    p.add_source_evaluation(guard_static(true), 0.0);
  }
  const uint32_t magnetic_update_b = p.operation_count();
  p.add_db(B_stuff);
  uint32_t magnetic_apply_b = UINT32_MAX;
  if (!cw) {
    magnetic_apply_b = p.operation_count();
    p.add_sources(B_stuff);
  }
  const uint32_t magnetic_transfer_b = p.operation_count();
  p.add_boundaries(B_stuff);

  uint32_t magnetic_evaluate_h = UINT32_MAX;
  if (has_sources) {
    magnetic_evaluate_h = p.operation_count();
    p.add_source_evaluation(guard_static(true), 0.5);
  }
  const uint32_t magnetic_update_h = p.operation_count();
  p.add_eh(H_stuff);
  p.add_boundaries(WH_stuff);
  p.add_polarizations(H_stuff);
  p.add_boundaries(PH_stuff);
  const uint32_t magnetic_transfer_h = p.operation_count();
  p.add_boundaries(H_stuff);

  p.add_legacy_flux(has_fluxes, OpKind::update_flux_half);

  if (has_sources) p.add_source_evaluation(guard_static(true), 0.5);
  p.add_db(D_stuff);
  if (!cw) p.add_sources(D_stuff);
  p.add_boundaries(D_stuff);

  if (has_sources) p.add_source_evaluation(guard_static(true), 1.0);
  p.add_eh(E_stuff);
  p.add_boundaries(WE_stuff);
  p.add_polarizations(E_stuff);
  p.add_boundaries(PE_stuff);
  p.add_boundaries(E_stuff);

  p.add_legacy_flux(has_fluxes, OpKind::update_flux);
  p.add(OpKind::increment_time);
  /* The decimation predicate is a device_predicate: the node stays in the
     graph and the kernel returns early when the step is not due. */
  if (has_dfts && !cw) p.add_dfts();
  p.set_magnetic_half_step(magnetic_evaluate_b, magnetic_update_b, magnetic_apply_b,
                           magnetic_transfer_b, magnetic_evaluate_h, magnetic_update_h,
                           magnetic_transfer_h);
  p.add_magnetic_marker(OpKind::synchronize_magnetic_fields, AccessMode::read_write);
  p.add_finite_value_check();
  /* The state markers bracket one complete source-suppressed matrix
     application.  CPU continues to perform the actual gather/scatter in
     cw_fields.cpp; resident backends consume these descriptors. */
  if (cw) p.add_cw_state_marker(OpKind::pack_state);

  return p.finish();
}

void format_step_plan(const StepPlan &p, std::vector<std::string> &out) {
  out.clear();
  char buf[128];
  for (const Operation &op : p.operations) {
    if (op.ft == field_type(NUM_FIELD_TYPES))
      snprintf(buf, sizeof buf, "%s", op_kind_name(op.kind));
    else
      snprintf(buf, sizeof buf, "%s(%s)", op_kind_name(op.kind), ft_name(op.ft));
    out.push_back(buf);
  }
}

} // namespace meep
