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
#include <cmath>
#include <limits>
#include <set>
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

ArrayId canonical_array(const CpuArrayCatalog &catalog, ArrayId id);

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

bool same_polarization_group_identity(const PolarizationUpdateGroup &group, int chunk,
                                      field_type ft, int state_index) {
  return group.chunk == chunk && group.ft == ft && group.state_index == state_index;
}

bool ordered_polarization_group_identity(const PolarizationUpdateGroup &previous, int chunk,
                                         field_type ft, int state_index) {
  return previous.ft == ft &&
         (previous.chunk < chunk || (previous.chunk == chunk && previous.state_index < state_index));
}

bool same_polarization_region(const UpdateRegion &a, const UpdateRegion &b) {
  if (a.chunk != b.chunk || a.c != b.c || a.cmp != b.cmp || !(a.begin == b.begin) ||
      !(a.end == b.end) || a.base != b.base)
    return false;
  for (int axis = 0; axis < 3; ++axis)
    if (a.counts[axis] != b.counts[axis] || a.strides[axis] != b.strides[axis]) return false;
  return true;
}

bool same_multilevel_region(const UpdateRegion &a, const UpdateRegion &b) {
  return same_polarization_region(a, b) && a.variant_key == b.variant_key;
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

bool polarization_update_aliases(const fields &f, const PolarizationUpdate &d, ArrayId id) {
  if (!f.array_catalog || !is_valid(id)) return false;
  const ArrayId protected_id = canonical_array(*f.array_catalog, id);
  const ArrayId candidates[] = {d.p,          d.p_prev,       d.p_cross1,
                                d.p_prev_cross1, d.p_cross2,    d.p_prev_cross2,
                                d.primary_w,  d.cross_w1,     d.cross_w2,
                                d.diagonal_sigma, d.offdiagonal_sigma1,
                                d.offdiagonal_sigma2};
  for (ArrayId candidate : candidates)
    if (is_valid(candidate) &&
        canonical_array(*f.array_catalog, candidate) == protected_id)
      return true;
  return false;
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
  if (op.polarization_group_index > plan.polarization_groups.size() ||
      op.polarization_group_count !=
          plan.polarization_groups.size() - op.polarization_group_index)
    throw std::invalid_argument("polarization group schedule is not appended to its operation");
  if (recurrences.empty() && noise_additions.empty()) return;

  const PolarizationUpdate &identity =
      recurrences.empty() ? noise_additions.front() : recurrences.front();
  if (identity.ft != op.ft)
    throw std::invalid_argument("polarization group field family differs from its operation");
  for (const PolarizationUpdateGroup &previous : plan.polarization_groups)
    if (same_polarization_group_identity(previous, identity.region.chunk, identity.ft,
                                         identity.state_index))
      throw std::invalid_argument("polarization group identity is not contiguous");
  if (op.polarization_group_count &&
      !ordered_polarization_group_identity(plan.polarization_groups.back(), identity.region.chunk,
                                           identity.ft, identity.state_index))
    throw std::invalid_argument("polarization groups are not in live state order");
  for (const MultilevelPopulationUpdate &population : plan.multilevel_population_updates)
    for (const PolarizationUpdate &update : recurrences)
      if (polarization_update_aliases(f, update, population.gamma_inv) ||
          polarization_update_aliases(f, update, population.populations))
        throw std::invalid_argument("polarization recurrence aliases multilevel scalar storage");
  for (const MultilevelPopulationUpdate &population : plan.multilevel_population_updates)
    for (const PolarizationUpdate &update : noise_additions)
      if (polarization_update_aliases(f, update, population.gamma_inv) ||
          polarization_update_aliases(f, update, population.populations))
        throw std::invalid_argument("polarization noise aliases multilevel scalar storage");
  for (const MultilevelPopulationTerm &term : plan.multilevel_population_terms) {
    for (const PolarizationUpdate &update : recurrences)
      if (polarization_update_aliases(f, update, term.p) ||
          polarization_update_aliases(f, update, term.p_prev))
        throw std::invalid_argument("polarization recurrence aliases multilevel state storage");
    for (const PolarizationUpdate &update : noise_additions)
      if (polarization_update_aliases(f, update, term.p) ||
          polarization_update_aliases(f, update, term.p_prev))
        throw std::invalid_argument("polarization noise aliases multilevel state storage");
  }
  PolarizationUpdateKind recurrence_kind = PolarizationUpdateKind::lorentzian;
  if (!recurrences.empty()) recurrence_kind = recurrences.front().kind;
  const uint32_t recurrence_index = uint32_t(plan.polarization_updates.size());
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
  const size_t update_count = recurrences.size() + noise_additions.size();
  if (update_count < recurrences.size() ||
      update_count > std::numeric_limits<uint32_t>::max() ||
      plan.polarization_updates.size() > std::numeric_limits<uint32_t>::max() - update_count ||
      plan.polarization_groups.size() >= std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("polarization group span overflow");

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
  plan.polarization_groups.push_back(
      PolarizationUpdateGroup{PolarizationGroupKind::recurrence,
                              identity.region.chunk,
                              identity.ft,
                              identity.state_index,
                              recurrence_index,
                              uint32_t(recurrences.size()),
                              uint32_t(noise_additions.size()),
                              0,
                              0,
                              0,
                              0});
  op.polarization_group_count =
      uint32_t(plan.polarization_groups.size()) - op.polarization_group_index;
  for (const MultilevelPopulationUpdate &population : plan.multilevel_population_updates)
    for (const BufferAccess &access : op.accesses)
      if (access.array.id == population.gamma_inv && access.mode != AccessMode::read)
        throw std::logic_error("multilevel GammaInv access is not read-only");
}

bool valid_catalog_array(const fields &f, ArrayId id) {
  return f.array_catalog && is_valid(id) && id.value < f.array_catalog->size() &&
         !is_valid(f.array_catalog->spec(id).alias_of) &&
         f.array_catalog->resolve_untyped(id) != NULL && f.array_catalog->spec(id).elements != 0;
}

void validate_multilevel_region_arithmetic(const UpdateRegion &region) {
  size_t last = region.base;
  for (int axis = 0; axis < 3; ++axis) {
    if (!region.counts[axis] || region.strides[axis] < 0)
      throw std::invalid_argument("malformed multilevel update region");
    const size_t stride = size_t(region.strides[axis]);
    const size_t count_minus_one = region.counts[axis] - 1;
    if (count_minus_one && stride > std::numeric_limits<size_t>::max() / count_minus_one)
      throw std::overflow_error("multilevel update region stride overflow");
    const size_t tail = count_minus_one * stride;
    if (tail > std::numeric_limits<size_t>::max() - last)
      throw std::overflow_error("multilevel update region base overflow");
    last += tail;
  }
}

bool ordered_multilevel_key(int previous_transition, component previous_component,
                            int previous_cmp, int transition, component c, int cmp) {
  return previous_transition < transition ||
         (previous_transition == transition &&
          (int(previous_component) < int(c) ||
           (previous_component == c && previous_cmp < cmp)));
}

void append_multilevel_update_group_impl(
    fields &f, StepPlan &plan, Operation &op, const MultilevelPopulationUpdate &input_population,
    const std::vector<MultilevelPopulationTerm> &terms,
    const std::vector<MultilevelTransitionUpdate> &transitions,
    const std::vector<double> &gamma_matrix, const std::vector<double> &alpha) {
  if (op.kind != OpKind::update_polarization)
    throw std::invalid_argument("multilevel group requires an update_polarization operation");
  if (op.descriptor_index > plan.polarization_updates.size() ||
      op.descriptor_count != plan.polarization_updates.size() - op.descriptor_index)
    throw std::invalid_argument("multilevel group is not appended to its descriptor span");
  if (op.polarization_group_index > plan.polarization_groups.size() ||
      op.polarization_group_count !=
          plan.polarization_groups.size() - op.polarization_group_index)
    throw std::invalid_argument("multilevel group is not appended to its operation span");
  if ((input_population.ft != E_stuff && input_population.ft != H_stuff) ||
      input_population.ft != op.ft || input_population.levels == 0 ||
      input_population.transitions == 0 || input_population.region.c != Centered ||
      input_population.region.cmp != -1 || input_population.region.chunk < 0 ||
      input_population.region.chunk >= f.num_chunks ||
      input_population.state_index < 0 || input_population.scratch_elements_per_point !=
                                                input_population.levels ||
      !std::isfinite(input_population.dt) || input_population.dt <= 0.0 ||
      !valid_catalog_array(f, input_population.gamma_inv) ||
      !valid_catalog_array(f, input_population.populations) ||
      input_population.gamma_inv == input_population.populations)
    throw std::invalid_argument("malformed multilevel population update");
  validate_multilevel_region_arithmetic(input_population.region);

  const size_t levels = input_population.levels;
  const size_t transition_count = input_population.transitions;
  if (levels > std::numeric_limits<size_t>::max() / levels ||
      levels > std::numeric_limits<size_t>::max() / transition_count ||
      gamma_matrix.size() != levels * levels || alpha.size() != levels * transition_count)
    throw std::invalid_argument("multilevel coefficient extent mismatch");
  for (double value : gamma_matrix)
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite multilevel Gamma value");
  for (double value : alpha)
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite multilevel alpha value");
  std::vector<int> positive_level(transition_count, -1);
  std::vector<int> negative_level(transition_count, -1);
  for (size_t level = 0; level < levels; ++level)
    for (size_t transition = 0; transition < transition_count; ++transition) {
      const double value = alpha[level * transition_count + transition];
      if (value > 0.0) positive_level[transition] = int(level);
      if (value < 0.0) negative_level[transition] = int(level);
    }
  for (size_t transition = 0; transition < transition_count; ++transition)
    if (positive_level[transition] < 0 || negative_level[transition] < 0)
      throw std::invalid_argument("multilevel alpha lacks a positive or negative level");
  if (terms.size() != transitions.size())
    throw std::invalid_argument("multilevel population/transition row count mismatch");

  for (const PolarizationUpdateGroup &previous : plan.polarization_groups)
    if (same_polarization_group_identity(previous, input_population.region.chunk,
                                         input_population.ft, input_population.state_index))
      throw std::invalid_argument("multilevel group identity is not contiguous");
  if (op.polarization_group_count &&
      !ordered_polarization_group_identity(plan.polarization_groups.back(),
                                           input_population.region.chunk, input_population.ft,
                                           input_population.state_index))
    throw std::invalid_argument("polarization groups are not in live state order");
  for (const BufferAccess &access : op.accesses)
    if (access.array.id == input_population.gamma_inv ||
        access.array.id == input_population.populations)
      throw std::invalid_argument("multilevel scalar storage is reused by an earlier group");

  if (input_population.active_component_cmps >
          std::numeric_limits<size_t>::max() / transition_count ||
      terms.size() != transition_count * input_population.active_component_cmps)
    throw std::invalid_argument("multilevel active-row count mismatch");
  std::set<uint32_t> mutable_state_arrays;
  mutable_state_arrays.insert(input_population.populations.value);
  std::set<uint32_t> read_only_dynamic_arrays;
  read_only_dynamic_arrays.insert(input_population.gamma_inv.value);

  std::set<uint32_t> prior_state_arrays;
  std::set<uint32_t> prior_dynamic_arrays;
  for (const PolarizationUpdate &update : plan.polarization_updates) {
    const ArrayId state_ids[] = {update.p,       update.p_prev,       update.p_cross1,
                                 update.p_prev_cross1, update.p_cross2, update.p_prev_cross2};
    for (ArrayId id : state_ids)
      if (is_valid(id)) {
        const uint32_t canonical = canonical_array(*f.array_catalog, id).value;
        prior_state_arrays.insert(canonical);
        prior_dynamic_arrays.insert(canonical);
      }
    const ArrayId read_ids[] = {update.primary_w, update.cross_w1, update.cross_w2,
                                update.diagonal_sigma, update.offdiagonal_sigma1,
                                update.offdiagonal_sigma2};
    for (ArrayId id : read_ids)
      if (is_valid(id))
        prior_dynamic_arrays.insert(canonical_array(*f.array_catalog, id).value);
  }
  for (const MultilevelPopulationUpdate &population : plan.multilevel_population_updates) {
    prior_state_arrays.insert(population.gamma_inv.value);
    prior_state_arrays.insert(population.populations.value);
    prior_dynamic_arrays.insert(population.gamma_inv.value);
    prior_dynamic_arrays.insert(population.populations.value);
  }
  for (const MultilevelPopulationTerm &term : plan.multilevel_population_terms) {
    prior_state_arrays.insert(term.p.value);
    prior_state_arrays.insert(term.p_prev.value);
    prior_dynamic_arrays.insert(term.p.value);
    prior_dynamic_arrays.insert(term.p_prev.value);
    prior_dynamic_arrays.insert(term.w.value);
    prior_dynamic_arrays.insert(term.w_prev.value);
  }
  for (const MultilevelTransitionUpdate &transition : plan.multilevel_transition_updates)
    prior_dynamic_arrays.insert(transition.diagonal_sigma.value);

  for (size_t i = 0; i < terms.size(); ++i) {
    const MultilevelPopulationTerm &term = terms[i];
    const MultilevelTransitionUpdate &transition = transitions[i];
    if (term.transition_index < 0 || size_t(term.transition_index) >= transition_count ||
        term.c == Centered || type(term.c) != input_population.ft || term.cmp < 0 || term.cmp > 1 ||
        !valid_catalog_array(f, term.w) || !valid_catalog_array(f, term.w_prev) ||
        !valid_catalog_array(f, term.p) || !valid_catalog_array(f, term.p_prev) ||
        term.p == term.p_prev || term.w == term.w_prev)
      throw std::invalid_argument("malformed multilevel population term");
    if (!mutable_state_arrays.insert(term.p.value).second ||
        !mutable_state_arrays.insert(term.p_prev.value).second)
      throw std::invalid_argument("multilevel transition state arrays are aliased");
    read_only_dynamic_arrays.insert(term.w.value);
    read_only_dynamic_arrays.insert(term.w_prev.value);
    read_only_dynamic_arrays.insert(transition.diagonal_sigma.value);
    if (term.p == input_population.gamma_inv || term.p_prev == input_population.gamma_inv ||
        term.w == input_population.gamma_inv || term.w_prev == input_population.gamma_inv ||
        term.p == input_population.populations || term.p_prev == input_population.populations ||
        term.w == input_population.populations || term.w_prev == input_population.populations)
      throw std::invalid_argument("multilevel scalar storage aliases a term array");
    if (i && !ordered_multilevel_key(terms[i - 1].transition_index, terms[i - 1].c,
                                     terms[i - 1].cmp, term.transition_index, term.c, term.cmp))
      throw std::invalid_argument("multilevel population terms are not transition-major");
    if (transition.ft != input_population.ft ||
        transition.region.chunk != input_population.region.chunk ||
        transition.state_index != input_population.state_index ||
        transition.transition_index != term.transition_index || transition.region.c != term.c ||
        transition.region.cmp != term.cmp || transition.p != term.p ||
        transition.p_prev != term.p_prev || transition.w != term.w ||
        transition.populations != input_population.populations ||
        term.centered_offsets[0] == std::numeric_limits<ptrdiff_t>::min() ||
        term.centered_offsets[1] == std::numeric_limits<ptrdiff_t>::min() ||
        transition.population_offsets[0] != -term.centered_offsets[0] ||
        transition.population_offsets[1] != -term.centered_offsets[1] ||
        transition.population_stride != input_population.levels ||
        transition.positive_level != positive_level[size_t(term.transition_index)] ||
        transition.negative_level != negative_level[size_t(term.transition_index)] ||
        transition.positive_level == transition.negative_level ||
        !valid_catalog_array(f, transition.p) ||
        !valid_catalog_array(f, transition.p_prev) || !valid_catalog_array(f, transition.w) ||
        !valid_catalog_array(f, transition.diagonal_sigma) ||
        !valid_catalog_array(f, transition.populations) || !std::isfinite(transition.omega) ||
        !std::isfinite(transition.gamma) || !std::isfinite(transition.dt) ||
        transition.dt != input_population.dt)
      throw std::invalid_argument("multilevel transition differs from its population term");
    validate_multilevel_region_arithmetic(transition.region);
    if (transition.diagonal_sigma == input_population.gamma_inv ||
        transition.diagonal_sigma == input_population.populations)
      throw std::invalid_argument("multilevel scalar storage aliases diagonal sigma");
    for (double value : transition.sigmat)
      if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite multilevel transition coefficient");
  }
  for (uint32_t id : mutable_state_arrays)
    if (read_only_dynamic_arrays.count(id))
      throw std::invalid_argument("multilevel writable storage aliases a read-only array");
  std::set<uint32_t> new_state_arrays = mutable_state_arrays;
  new_state_arrays.insert(input_population.gamma_inv.value);
  for (uint32_t id : new_state_arrays)
    if (prior_dynamic_arrays.count(id))
      throw std::invalid_argument("multilevel state storage aliases an earlier group");
  for (uint32_t id : read_only_dynamic_arrays)
    if (prior_state_arrays.count(id))
      throw std::invalid_argument("multilevel read-only storage aliases earlier state");
  if (!terms.empty()) {
    size_t rows_per_transition = 0;
    while (rows_per_transition < terms.size() && terms[rows_per_transition].transition_index == 0)
      ++rows_per_transition;
    if (!rows_per_transition || terms.size() != rows_per_transition * transition_count)
      throw std::invalid_argument("multilevel transition rows are incomplete");
    for (size_t row = 0; row < rows_per_transition; ++row)
      if (terms[row].cmp == 1 &&
          (row == 0 || terms[row - 1].c != terms[row].c || terms[row - 1].cmp != 0))
        throw std::invalid_argument("multilevel imaginary row lacks its real row");
    for (size_t transition = 0; transition < transition_count; ++transition) {
      const MultilevelTransitionUpdate &coefficient_reference =
          transitions[transition * rows_per_transition];
      for (size_t row = 0; row < rows_per_transition; ++row) {
        const MultilevelPopulationTerm &actual = terms[transition * rows_per_transition + row];
        const MultilevelPopulationTerm &expected = terms[row];
        const MultilevelTransitionUpdate &actual_transition =
            transitions[transition * rows_per_transition + row];
        const MultilevelTransitionUpdate &expected_transition = transitions[row];
        if (actual.transition_index != int(transition) || actual.c != expected.c ||
            actual.cmp != expected.cmp || actual.w != expected.w ||
            actual.w_prev != expected.w_prev ||
            actual.centered_offsets[0] != expected.centered_offsets[0] ||
            actual.centered_offsets[1] != expected.centered_offsets[1] ||
            !same_multilevel_region(actual_transition.region, expected_transition.region) ||
            actual_transition.diagonal_sigma != expected_transition.diagonal_sigma)
          throw std::invalid_argument("multilevel transition row sets differ");
        if (actual_transition.omega != coefficient_reference.omega ||
            actual_transition.gamma != coefficient_reference.gamma)
          throw std::invalid_argument("multilevel transition coefficients differ by row");
        for (int coefficient = 0; coefficient < 5; ++coefficient)
          if (actual_transition.sigmat[coefficient] !=
              coefficient_reference.sigmat[coefficient])
            throw std::invalid_argument("multilevel transition coefficients differ by row");
      }
    }
  }

  const size_t u32_max = std::numeric_limits<uint32_t>::max();
  if (terms.size() > u32_max || transitions.size() > u32_max || gamma_matrix.size() > u32_max ||
      alpha.size() > u32_max || plan.multilevel_population_updates.size() >= u32_max ||
      plan.multilevel_population_terms.size() > u32_max - terms.size() ||
      plan.multilevel_transition_updates.size() > u32_max - transitions.size() ||
      plan.multilevel_coefficients.size() > u32_max - gamma_matrix.size() ||
      plan.multilevel_coefficients.size() + gamma_matrix.size() > u32_max - alpha.size() ||
      plan.polarization_groups.size() >= u32_max)
    throw std::overflow_error("multilevel action span overflow");

  MultilevelPopulationUpdate population = input_population;
  population.gamma_index = uint32_t(plan.multilevel_coefficients.size());
  population.gamma_count = uint32_t(gamma_matrix.size());
  population.alpha_index = population.gamma_index + population.gamma_count;
  population.alpha_count = uint32_t(alpha.size());
  population.term_index = uint32_t(plan.multilevel_population_terms.size());
  population.term_count = uint32_t(terms.size());
  const uint32_t population_index = uint32_t(plan.multilevel_population_updates.size());
  const uint32_t transition_index = uint32_t(plan.multilevel_transition_updates.size());

  plan.multilevel_coefficients.insert(plan.multilevel_coefficients.end(), gamma_matrix.begin(),
                                      gamma_matrix.end());
  plan.multilevel_coefficients.insert(plan.multilevel_coefficients.end(), alpha.begin(),
                                      alpha.end());
  plan.multilevel_population_terms.insert(plan.multilevel_population_terms.end(), terms.begin(),
                                          terms.end());
  plan.multilevel_transition_updates.insert(plan.multilevel_transition_updates.end(),
                                            transitions.begin(), transitions.end());
  plan.multilevel_population_updates.push_back(population);
  plan.polarization_groups.push_back(
      PolarizationUpdateGroup{PolarizationGroupKind::multilevel,
                              population.region.chunk,
                              population.ft,
                              population.state_index,
                              0,
                              0,
                              0,
                              population_index,
                              1,
                              transition_index,
                              uint32_t(transitions.size())});

  add_access(f, op, population.gamma_inv, AccessMode::read);
  add_access(f, op, population.populations, AccessMode::read_write);
  for (const MultilevelPopulationTerm &term : terms) {
    add_access(f, op, term.w, AccessMode::read);
    add_access(f, op, term.w_prev, AccessMode::read);
    add_access(f, op, term.p, AccessMode::read);
    add_access(f, op, term.p_prev, AccessMode::read);
  }
  for (const MultilevelTransitionUpdate &transition : transitions) {
    add_access(f, op, transition.p, AccessMode::read_write);
    add_access(f, op, transition.p_prev, AccessMode::read_write);
    add_access(f, op, transition.w, AccessMode::read);
    add_access(f, op, transition.diagonal_sigma, AccessMode::read);
    add_access(f, op, transition.populations, AccessMode::read);
  }
  op.polarization_group_count =
      uint32_t(plan.polarization_groups.size()) - op.polarization_group_index;
  for (const BufferAccess &access : op.accesses)
    if (access.array.id == population.gamma_inv && access.mode != AccessMode::read)
      throw std::logic_error("multilevel GammaInv access is not read-only");
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
    throw std::invalid_argument("portable plan row names an invalid array");
  for (size_t depth = 0; depth <= catalog.size(); ++depth) {
    const ArrayId next = catalog.spec(id).alias_of;
    if (!is_valid(next)) return id;
    if (next.value >= catalog.size())
      throw std::invalid_argument("portable plan row names an invalid alias");
    if (catalog.resolve_untyped(id) != catalog.resolve_untyped(next))
      throw std::invalid_argument("portable plan row contains a stale alias binding");
    id = next;
  }
  throw std::invalid_argument("portable plan row contains an alias cycle");
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
    op.polarization_group_index = 0;
    op.polarization_group_count = 0;
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
      mix(sig, uint64_t(op.polarization_group_index));
      mix(sig, uint64_t(op.polarization_group_count));
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
    for (const PolarizationUpdateGroup &d : plan.polarization_groups)
      hash_polarization_group(sig, d);
    for (const PolarizationUpdate &d : plan.polarization_updates)
      hash_polarization(sig, d);
    for (const PolarizationSubtraction &d : plan.polarization_subtractions)
      hash_polarization_subtraction(sig, d);
    for (const MultilevelPopulationUpdate &d : plan.multilevel_population_updates)
      hash_multilevel_population(sig, d);
    for (const MultilevelPopulationTerm &d : plan.multilevel_population_terms)
      hash_multilevel_term(sig, d);
    for (const MultilevelTransitionUpdate &d : plan.multilevel_transition_updates)
      hash_multilevel_transition(sig, d);
    for (double coefficient : plan.multilevel_coefficients)
      mix_double(sig, coefficient);
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
  static void hash_polarization_group(uint64_t &sig, const PolarizationUpdateGroup &d) {
    mix(sig, uint64_t(d.kind));
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.ft));
    mix(sig, uint64_t(d.state_index));
    mix(sig, uint64_t(d.recurrence_index));
    mix(sig, uint64_t(d.recurrence_count));
    mix(sig, uint64_t(d.noise_count));
    mix(sig, uint64_t(d.population_index));
    mix(sig, uint64_t(d.population_count));
    mix(sig, uint64_t(d.transition_index));
    mix(sig, uint64_t(d.transition_count));
  }
  static void hash_multilevel_term(uint64_t &sig, const MultilevelPopulationTerm &d) {
    mix(sig, uint64_t(d.transition_index));
    mix(sig, uint64_t(d.c));
    mix(sig, uint64_t(d.cmp));
    hash_id(sig, d.w);
    hash_id(sig, d.w_prev);
    hash_id(sig, d.p);
    hash_id(sig, d.p_prev);
    mix(sig, uint64_t(d.centered_offsets[0]));
    mix(sig, uint64_t(d.centered_offsets[1]));
  }
  static void hash_multilevel_population(uint64_t &sig, const MultilevelPopulationUpdate &d) {
    hash_region(sig, d.region);
    mix(sig, uint64_t(d.ft));
    mix(sig, uint64_t(d.state_index));
    mix(sig, uint64_t(d.levels));
    mix(sig, uint64_t(d.transitions));
    mix(sig, uint64_t(d.active_component_cmps));
    hash_id(sig, d.gamma_inv);
    hash_id(sig, d.populations);
    mix(sig, uint64_t(d.gamma_index));
    mix(sig, uint64_t(d.gamma_count));
    mix(sig, uint64_t(d.alpha_index));
    mix(sig, uint64_t(d.alpha_count));
    mix(sig, uint64_t(d.term_index));
    mix(sig, uint64_t(d.term_count));
    mix(sig, uint64_t(d.scratch_elements_per_point));
    mix_double(sig, d.dt);
  }
  static void hash_multilevel_transition(uint64_t &sig, const MultilevelTransitionUpdate &d) {
    hash_region(sig, d.region);
    mix(sig, uint64_t(d.ft));
    mix(sig, uint64_t(d.state_index));
    mix(sig, uint64_t(d.transition_index));
    hash_id(sig, d.p);
    hash_id(sig, d.p_prev);
    hash_id(sig, d.w);
    hash_id(sig, d.diagonal_sigma);
    hash_id(sig, d.populations);
    mix(sig, uint64_t(d.population_offsets[0]));
    mix(sig, uint64_t(d.population_offsets[1]));
    mix(sig, uint64_t(d.population_stride));
    mix(sig, uint64_t(d.positive_level));
    mix(sig, uint64_t(d.negative_level));
    mix_double(sig, d.omega);
    mix_double(sig, d.gamma);
    for (double value : d.sigmat)
      mix_double(sig, value);
    mix_double(sig, d.dt);
  }
  static void hash_polarization_subtraction(uint64_t &sig, const PolarizationSubtraction &d) {
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.c));
    mix(sig, uint64_t(d.cmp));
    mix(sig, uint64_t(d.state_index));
    mix(sig, uint64_t(d.transition_index));
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

  if (f_.descriptors) {
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
          const ArrayId primary_minus_p =
              find_array(f_, chunk, array_kind::f_minus_p, int(dc), cmp, 0);
          const ArrayId cross1_minus_p =
              find_array(f_, chunk, array_kind::f_minus_p, int(dc1), cmp, 0);
          const ArrayId cross2_minus_p =
              find_array(f_, chunk, array_kind::f_minus_p, int(dc2), cmp, 0);
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
          if (tile != 0) d.previous_w = invalid_array();
          d.pml = make_pml_profile(f_, fc, chunk, dsigw, d.region.begin);
          if (is_valid(d.offdiagonal1)) d.region.variant_key |= constitutive_one_offdiagonal;
          if (is_valid(d.offdiagonal2)) d.region.variant_key |= constitutive_two_offdiagonals;
          if (dsigw != NO_DIRECTION) d.region.variant_key |= constitutive_has_pml;
          if (is_valid(d.chi2) || is_valid(d.chi3))
            d.region.variant_key |= constitutive_has_nonlinearity;
          if (is_valid(primary_minus_p) || is_valid(cross1_minus_p) || is_valid(cross2_minus_p))
            d.region.variant_key |= constitutive_has_minus_p;
          if (is_valid(d.previous_w))
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
  attach_source_span(op, ft2, true);
}

} // namespace

void append_polarization_update_group(fields &f, StepPlan &plan, Operation &op,
                                      const std::vector<PolarizationUpdate> &recurrences,
                                      const std::vector<PolarizationUpdate> &noise_additions) {
  append_polarization_update_group_impl(f, plan, op, recurrences, noise_additions);
}

void append_multilevel_update_group(fields &f, StepPlan &plan, Operation &op,
                                    const MultilevelPopulationUpdate &population,
                                    const std::vector<MultilevelPopulationTerm> &terms,
                                    const std::vector<MultilevelTransitionUpdate> &transitions,
                                    const std::vector<double> &gamma_matrix,
                                    const std::vector<double> &alpha) {
  append_multilevel_update_group_impl(f, plan, op, population, terms, transitions, gamma_matrix,
                                      alpha);
}

bool operator==(const PolarizationUpdate &a, const PolarizationUpdate &b) {
  return polarization_updates_equal(a, b);
}

bool operator==(const PolarizationSubtraction &a, const PolarizationSubtraction &b) {
  return a.chunk == b.chunk && a.c == b.c && a.cmp == b.cmp &&
         a.state_index == b.state_index && a.transition_index == b.transition_index &&
         a.target == b.target && a.p == b.p && a.elements == b.elements;
}

bool operator==(const PolarizationUpdateGroup &a, const PolarizationUpdateGroup &b) {
  return a.kind == b.kind && a.chunk == b.chunk && a.ft == b.ft &&
         a.state_index == b.state_index && a.recurrence_index == b.recurrence_index &&
         a.recurrence_count == b.recurrence_count && a.noise_count == b.noise_count &&
         a.population_index == b.population_index && a.population_count == b.population_count &&
         a.transition_index == b.transition_index && a.transition_count == b.transition_count;
}

bool operator==(const MultilevelPopulationTerm &a, const MultilevelPopulationTerm &b) {
  return a.transition_index == b.transition_index && a.c == b.c && a.cmp == b.cmp &&
         a.w == b.w && a.w_prev == b.w_prev && a.p == b.p && a.p_prev == b.p_prev &&
         a.centered_offsets[0] == b.centered_offsets[0] &&
         a.centered_offsets[1] == b.centered_offsets[1];
}

bool operator==(const MultilevelPopulationUpdate &a, const MultilevelPopulationUpdate &b) {
  return same_multilevel_region(a.region, b.region) && a.ft == b.ft &&
         a.state_index == b.state_index && a.levels == b.levels &&
         a.transitions == b.transitions &&
         a.active_component_cmps == b.active_component_cmps && a.gamma_inv == b.gamma_inv &&
         a.populations == b.populations && a.gamma_index == b.gamma_index &&
         a.gamma_count == b.gamma_count && a.alpha_index == b.alpha_index &&
         a.alpha_count == b.alpha_count && a.term_index == b.term_index &&
         a.term_count == b.term_count &&
         a.scratch_elements_per_point == b.scratch_elements_per_point && a.dt == b.dt;
}

bool operator==(const MultilevelTransitionUpdate &a, const MultilevelTransitionUpdate &b) {
  if (!same_multilevel_region(a.region, b.region) || a.ft != b.ft ||
      a.state_index != b.state_index || a.transition_index != b.transition_index || a.p != b.p ||
      a.p_prev != b.p_prev || a.w != b.w || a.diagonal_sigma != b.diagonal_sigma ||
      a.populations != b.populations || a.population_offsets[0] != b.population_offsets[0] ||
      a.population_offsets[1] != b.population_offsets[1] ||
      a.population_stride != b.population_stride || a.positive_level != b.positive_level ||
      a.negative_level != b.negative_level || a.omega != b.omega || a.gamma != b.gamma ||
      a.dt != b.dt)
    return false;
  for (int i = 0; i < 5; ++i)
    if (a.sigmat[i] != b.sigmat[i]) return false;
  return true;
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
