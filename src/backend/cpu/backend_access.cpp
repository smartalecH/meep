/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* Backend selection, lifecycle, and the backend-safe access points. */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "backend/backend.hpp"
#include "backend/cpu/cpu_backend.hpp"
#include "backend/descriptors.hpp"
#include "backend/halo_plan.hpp"
#include "backend/initialization_plan.hpp"
#include "backend/lifecycle.hpp"
#include "backend/precision.hpp"
#include "backend/prepare.hpp"
#include "backend/random_state.hpp"
#include "backend/step_plan.hpp"

namespace meep {

static int cw_clone_fail_after_for_testing = -1;
static bool cw_plan_corruption_for_testing = false;
static int legacy_flux_prepare_failure_rank_for_testing = -1;
static int noisy_preflight_failure_rank_for_testing = -1;
static int noisy_preflight_failure_mode_for_testing = 0;
static size_t noisy_collective_count_for_testing = 0;
static int multilevel_preflight_failure_rank_for_testing = -1;
static int multilevel_preflight_failure_mode_for_testing = 0;
static size_t multilevel_collective_count_for_testing = 0;

void backend_set_cw_clone_fail_after_for_testing(int checkpoints) {
  cw_clone_fail_after_for_testing = checkpoints;
}

void backend_cw_clone_checkpoint() {
  if (cw_clone_fail_after_for_testing < 0) return;
  if (cw_clone_fail_after_for_testing-- == 0) throw std::bad_alloc();
}

void backend_set_cw_plan_corruption_for_testing(bool enabled) {
  cw_plan_corruption_for_testing = enabled;
}

void backend_set_legacy_flux_prepare_failure_for_testing(int rank) {
  legacy_flux_prepare_failure_rank_for_testing = rank;
}

void backend_set_noisy_preflight_failure_for_testing(int rank, int mode) {
  noisy_preflight_failure_rank_for_testing = rank;
  noisy_preflight_failure_mode_for_testing = mode;
}

void backend_reset_noisy_collective_count_for_testing() {
  noisy_collective_count_for_testing = 0;
}

size_t backend_noisy_collective_count_for_testing() {
  return noisy_collective_count_for_testing;
}

void backend_note_noisy_collective_for_testing() {
  ++noisy_collective_count_for_testing;
}

void backend_set_multilevel_preflight_failure_for_testing(int rank, int mode) {
  multilevel_preflight_failure_rank_for_testing = rank;
  multilevel_preflight_failure_mode_for_testing = mode;
}

void backend_reset_multilevel_collective_count_for_testing() {
  multilevel_collective_count_for_testing = 0;
}

size_t backend_multilevel_collective_count_for_testing() {
  return multilevel_collective_count_for_testing;
}

void backend_note_multilevel_collective_for_testing() {
  ++multilevel_collective_count_for_testing;
}

namespace {

typedef std::tuple<int, int, int> NoisyGroupTuple;
typedef std::tuple<int, int, int, int, int> NoisyStreamTuple;

bool finite_in_realnum(double value) {
  return std::isfinite(value) && std::isfinite(double(realnum(value)));
}

bool noisy_region_fits(const fields &f, ArrayId id, array_role role,
                       const UpdateRegion &region) {
  if (!f.array_catalog || !is_valid(id) || id.value >= f.array_catalog->size()) return false;
  const ArraySpec &spec = f.array_catalog->spec(id);
  if (spec.role != role || spec.element_type != ElementType::realnum_value ||
      is_valid(spec.alias_of) || !f.array_catalog->resolve_untyped(id))
    return false;
  __int128 low = region.base, high = region.base;
  for (int axis = 0; axis < 3; ++axis) {
    if (!region.counts[axis] || (region.counts[axis] > 1 && region.strides[axis] == 0))
      return false;
    const __int128 delta = __int128(region.counts[axis] - 1) * region.strides[axis];
    if (delta < 0)
      low += delta;
    else
      high += delta;
  }
  return low >= 0 && high >= low && high < __int128(spec.elements);
}

bool same_noisy_access(const BufferAccess &a, const BufferAccess &b) {
  return a.array.id == b.array.id && a.array.offset == b.array.offset &&
         a.array.elements == b.array.elements && a.mode == b.mode;
}

bool same_polarization_operation(const Operation &a, const Operation &b) {
  if (a.kind != b.kind || a.ft != b.ft || a.descriptor_index != b.descriptor_index ||
      a.descriptor_count != b.descriptor_count ||
      a.material_refresh_index != b.material_refresh_index ||
      a.material_refresh_count != b.material_refresh_count ||
      a.beta_descriptor_index != b.beta_descriptor_index ||
      a.beta_descriptor_count != b.beta_descriptor_count ||
      a.cylindrical_m_descriptor_index != b.cylindrical_m_descriptor_index ||
      a.cylindrical_m_descriptor_count != b.cylindrical_m_descriptor_count ||
      a.cylindrical_origin_action_index != b.cylindrical_origin_action_index ||
      a.cylindrical_origin_action_count != b.cylindrical_origin_action_count ||
      a.polarization_group_index != b.polarization_group_index ||
      a.polarization_group_count != b.polarization_group_count ||
      a.polarization_subtraction_index != b.polarization_subtraction_index ||
      a.polarization_subtraction_count != b.polarization_subtraction_count ||
      a.magnetic_state_index != b.magnetic_state_index ||
      a.magnetic_state_count != b.magnetic_state_count ||
      a.legacy_flux_index != b.legacy_flux_index ||
      a.legacy_flux_count != b.legacy_flux_count ||
      a.source_descriptor_index != b.source_descriptor_index ||
      a.source_descriptor_count != b.source_descriptor_count ||
      a.guard.kind != b.guard.kind || a.guard.scalar_slot != b.guard.scalar_slot ||
      a.guard.variant_index != b.guard.variant_index ||
      a.source_time_offset != b.source_time_offset || a.accesses.size() != b.accesses.size())
    return false;
  for (size_t i = 0; i < a.accesses.size(); ++i)
    if (!same_noisy_access(a.accesses[i], b.accesses[i])) return false;
  return true;
}

bool same_internal_layout(const InternalArrayLayout &a, const InternalArrayLayout &b) {
  return ((!a.name && !b.name) || (a.name && b.name && !strcmp(a.name, b.name))) &&
         a.element_type == b.element_type && a.offset_elements == b.offset_elements &&
         a.elements == b.elements && a.c == b.c && a.cmp == b.cmp;
}

bool same_multilevel_descriptor(const PolarizationDescriptor &a,
                                const PolarizationDescriptor &b) {
  if (a.kind != SusceptibilityKind::multilevel || b.kind != SusceptibilityKind::multilevel ||
      a.chunk != b.chunk || a.ft != b.ft || a.state_index != b.state_index ||
      a.has_internal_state != b.has_internal_state ||
      a.multilevel.levels != b.multilevel.levels ||
      a.multilevel.transitions != b.multilevel.transitions ||
      a.multilevel.gamma_matrix != b.multilevel.gamma_matrix ||
      a.multilevel.initial_populations != b.multilevel.initial_populations ||
      a.multilevel.alpha != b.multilevel.alpha || a.multilevel.omega != b.multilevel.omega ||
      a.multilevel.transition_gamma != b.multilevel.transition_gamma ||
      a.multilevel.sigmat != b.multilevel.sigmat ||
      a.multilevel_gamma_inv != b.multilevel_gamma_inv ||
      a.multilevel_populations != b.multilevel_populations ||
      a.multilevel_population_points != b.multilevel_population_points ||
      a.per_thread_scratch_elements != b.per_thread_scratch_elements ||
      a.required_w != b.required_w || a.required_w_prev != b.required_w_prev ||
      a.needs_halo != b.needs_halo || a.multilevel_states.size() != b.multilevel_states.size() ||
      a.internal_arrays.size() != b.internal_arrays.size())
    return false;
  for (size_t i = 0; i < a.multilevel_states.size(); ++i) {
    const MultilevelStateArrays &x = a.multilevel_states[i];
    const MultilevelStateArrays &y = b.multilevel_states[i];
    if (x.transition_index != y.transition_index || x.c != y.c || x.cmp != y.cmp ||
        x.p != y.p || x.p_prev != y.p_prev || x.elements != y.elements)
      return false;
  }
  for (size_t i = 0; i < a.internal_arrays.size(); ++i)
    if (!same_internal_layout(a.internal_arrays[i], b.internal_arrays[i])) return false;
  return true;
}

std::string validate_multilevel_plan(const fields &f, const StepPlan &plan,
                                     bool &local_multilevel_actions) {
  local_multilevel_actions = false;
  if (multilevel_preflight_failure_rank_for_testing == my_rank()) {
    if (multilevel_preflight_failure_mode_for_testing == 3)
      return "injected multilevel static validation failure";
    if (multilevel_preflight_failure_mode_for_testing == 4) throw std::bad_alloc();
  }
  if (!f.array_catalog || !f.descriptors)
    return "multilevel preflight has no catalog or descriptor set";
  if (compute_step_plan_signature(plan) != plan.signature)
    return "installed multilevel plan signature is stale";

  std::vector<const PolarizationDescriptor *> installed_descriptors;
  for (const PolarizationDescriptor &descriptor : f.descriptors->polarizations)
    if (descriptor.kind == SusceptibilityKind::multilevel)
      installed_descriptors.push_back(&descriptor);
  for (const PolarizationUpdateGroup &group : plan.polarization_groups)
    local_multilevel_actions =
        local_multilevel_actions || group.kind == PolarizationGroupKind::multilevel;
  local_multilevel_actions = local_multilevel_actions || !installed_descriptors.empty() ||
                             has_local_exact_multilevel(f);

  fields &mutable_f = const_cast<fields &>(f);
  std::vector<PolarizationDescriptor> rebuilt_descriptors;
  StepPlan canonical;
  DescriptorSet *const installed_set = mutable_f.descriptors;
  try {
    build_polarization_descriptors(mutable_f, rebuilt_descriptors);
    DescriptorSet staged = *installed_set;
    staged.polarizations = rebuilt_descriptors;
    mutable_f.descriptors = &staged;
    canonical = build_step_plan(mutable_f, StepProgram::ordinary);
    mutable_f.descriptors = installed_set;
  }
  catch (...) {
    mutable_f.descriptors = installed_set;
    throw;
  }

  std::vector<const PolarizationDescriptor *> canonical_descriptors;
  for (const PolarizationDescriptor &descriptor : rebuilt_descriptors)
    if (descriptor.kind == SusceptibilityKind::multilevel)
      canonical_descriptors.push_back(&descriptor);
  if (installed_descriptors.size() != canonical_descriptors.size())
    return "installed multilevel descriptors differ from live exact states";
  for (size_t i = 0; i < installed_descriptors.size(); ++i)
    if (!same_multilevel_descriptor(*installed_descriptors[i], *canonical_descriptors[i]))
      return "installed multilevel descriptor is not canonical";

  std::vector<const Operation *> installed_ops, canonical_ops;
  for (const Operation &op : plan.operations)
    if (op.kind == OpKind::update_polarization || op.kind == OpKind::update_eh)
      installed_ops.push_back(&op);
  for (const Operation &op : canonical.operations)
    if (op.kind == OpKind::update_polarization || op.kind == OpKind::update_eh)
      canonical_ops.push_back(&op);
  if (plan.coordinate_generation != canonical.coordinate_generation)
    return "installed multilevel coordinate generation is not canonical";
  if (plan.polarization_updates != canonical.polarization_updates)
    return "installed ordinary polarization rows are not canonical";
  if (plan.polarization_subtractions != canonical.polarization_subtractions)
    return "installed polarization subtraction rows are not canonical";
  if (plan.polarization_groups != canonical.polarization_groups)
    return "installed polarization groups are not canonical";
  if (plan.multilevel_population_updates != canonical.multilevel_population_updates)
    return "installed multilevel population updates are not canonical";
  if (plan.multilevel_population_terms != canonical.multilevel_population_terms)
    return "installed multilevel population terms are not canonical";
  if (plan.multilevel_transition_updates != canonical.multilevel_transition_updates)
    return "installed multilevel transition updates are not canonical";
  if (plan.multilevel_coefficients != canonical.multilevel_coefficients)
    return "installed multilevel coefficients are not canonical";
  if (installed_ops.size() != canonical_ops.size())
    return "installed multilevel polarization operation count is not canonical";
  for (size_t i = 0; i < installed_ops.size(); ++i)
    if (!same_polarization_operation(*installed_ops[i], *canonical_ops[i]))
      return "installed multilevel polarization/subtraction operation is not canonical";
  return std::string();
}

bool full_seed_snapshot_equal(const RandomSeedSnapshot &a, const RandomSeedSnapshot &b) {
  return a.semantic_seed == b.semantic_seed &&
         a.saved_semantic_seed == b.saved_semantic_seed && a.generation == b.generation &&
         a.algorithm_version == b.algorithm_version && a.initialized == b.initialized &&
         a.semantic_seed_valid == b.semantic_seed_valid &&
         a.saved_semantic_seed_valid == b.saved_semantic_seed_valid &&
         a.explicit_seed == b.explicit_seed &&
         a.saved_explicit_seed == b.saved_explicit_seed;
}

bool valid_noisy_coefficient(const PolarizationUpdate &update) {
  if (!finite_in_realnum(update.omega_0) || !finite_in_realnum(update.gamma) ||
      !finite_in_realnum(update.dt) || update.dt <= 0.0 ||
      !finite_in_realnum(update.noise_amplitude))
    return false;
  const realnum noise_amplitude = realnum(update.noise_amplitude);
  const realnum gamma = realnum(update.gamma);
  const realnum omega_0 = realnum(update.omega_0);
  const realnum dt = realnum(update.dt);
  const realnum g2pi = gamma * 2 * pi;
  const realnum w2pi = omega_0 * 2 * pi;
  if (!std::isfinite(double(g2pi)) || g2pi < realnum(0) ||
      !std::isfinite(double(w2pi)))
    return false;
  const realnum root = sqrt(g2pi);
  const realnum denominator = 1 + g2pi * dt / 2;
  if (!std::isfinite(double(root)) || !std::isfinite(double(denominator)) ||
      denominator == realnum(0))
    return false;
  /* Exact finite zero follows the deterministic recurrence without creating a
     signed-zero noise store, but the CPU has already evaluated the common
     sqrt/denominator coefficient path above. */
  if (noise_amplitude == realnum(0)) return true;
  const realnum amplitude =
      w2pi * noise_amplitude * root * dt * dt / denominator;
  return std::isfinite(double(amplitude));
}

std::string validate_noisy_plan(const fields &f, const StepPlan &plan,
                                bool &local_noisy_actions, size_t &stream_count,
                                uint64_t &first_stream_tag) {
  local_noisy_actions = false;
  stream_count = 0;
  first_stream_tag = 0;
  if (!f.array_catalog) return "noisy RNG preflight has no array catalog";

  std::vector<NoisyGroupTuple> live_groups;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk] || !f.chunks[chunk]->is_mine()) continue;
    FOR_FIELD_TYPES(ft) {
      int state_index = 0;
      for (const polarization_state *p = f.chunks[chunk]->pol[ft]; p;
           p = p->next, ++state_index)
        if (p->s && typeid(*p->s) == typeid(noisy_lorentzian_susceptibility))
          live_groups.push_back(NoisyGroupTuple(chunk, int(ft), state_index));
    }
  }
  std::vector<NoisyGroupTuple> descriptor_groups;
  if (f.descriptors)
    for (const PolarizationDescriptor &descriptor : f.descriptors->polarizations)
      if (descriptor.kind == SusceptibilityKind::noisy_lorentzian)
        descriptor_groups.push_back(
            NoisyGroupTuple(descriptor.chunk, int(descriptor.ft), descriptor.state_index));
  if (live_groups != descriptor_groups)
    return "noisy susceptibility descriptors do not match the live linked-list order";

  StepPlan canonical;
  try {
    canonical = build_polarization_validation_plan(const_cast<fields &>(f));
  }
  catch (const std::exception &e) {
    return e.what();
  }
  catch (...) {
    return "unknown canonical noisy polarization-plan failure";
  }
  std::vector<const Operation *> installed_ops, canonical_ops;
  for (const Operation &op : plan.operations)
    if (op.kind == OpKind::update_polarization) installed_ops.push_back(&op);
  for (const Operation &op : canonical.operations)
    if (op.kind == OpKind::update_polarization) canonical_ops.push_back(&op);
  const int global_rank = my_global_rank();
  if (global_rank < 0 || uint64_t(global_rank) > uint64_t(UINT32_MAX))
    return "noisy RNG global rank is out of range";
  if (f.t < 0) return "noisy RNG timestep is negative";

  std::set<NoisyStreamTuple> tuples;
  std::set<uint64_t> tags;
  size_t noisy_seen = 0;
  for (const PolarizationUpdate &update : plan.polarization_updates) {
    if (update.kind != PolarizationUpdateKind::noisy_add) continue;
    local_noisy_actions = true;
    ++noisy_seen;
    if (noisy_preflight_failure_mode_for_testing == 4 &&
        noisy_preflight_failure_rank_for_testing == my_rank() && noisy_seen == 1)
      return "injected noisy group validation failure";
    const int c = int(update.region.c);
    if (update.region.chunk < 0 || update.region.chunk >= f.num_chunks ||
        uint64_t(update.region.chunk) > uint64_t(UINT32_MAX) ||
        (update.ft != E_stuff && update.ft != H_stuff) ||
        update.state_index < 0 || uint64_t(update.state_index) > uint64_t(UINT32_MAX) ||
        c < 0 || c >= NUM_FIELD_COMPONENTS ||
        (update.ft == E_stuff ? !is_electric(update.region.c)
                              : !is_magnetic(update.region.c)) ||
        (update.region.cmp != 0 && update.region.cmp != 1) ||
        update.noise_algorithm_version != counter_random_algorithm_version ||
        !valid_noisy_coefficient(update))
      return "noisy polarization action metadata, range, or coefficient is invalid";

    uint64_t points = 1;
    for (int axis = 0; axis < 3; ++axis) {
      if (!update.region.counts[axis] ||
          points > UINT64_MAX / uint64_t(update.region.counts[axis]))
        return "noisy owned-point extent overflows uint64";
      points *= uint64_t(update.region.counts[axis]);
    }
    if (!points) return "noisy owned-point extent is empty";
    if (!noisy_region_fits(f, update.p, array_role::polarization, update.region) ||
        !noisy_region_fits(f, update.diagonal_sigma, array_role::material, update.region))
      return "noisy polarization action array range is invalid";

    const NoisyStreamTuple tuple(update.region.chunk, int(update.ft), update.state_index, c,
                                 update.region.cmp);
    if (!tuples.insert(tuple).second) return "duplicate noisy RNG stream tuple";
    uint64_t tag = counter_random_stream_tag(
        update.noise_algorithm_version, uint32_t(global_rank), uint32_t(update.region.chunk),
        uint32_t(update.ft), uint32_t(update.state_index), uint32_t(c),
        uint32_t(update.region.cmp));
    if (noisy_preflight_failure_mode_for_testing == 6 &&
        noisy_preflight_failure_rank_for_testing == my_rank() && noisy_seen == 2 && !tags.empty())
      tag = *tags.begin();
    if (!tags.insert(tag).second) return "noisy RNG static stream tag collision";
    if (!stream_count) first_stream_tag = tag;
    ++stream_count;
  }
  if (plan.polarization_updates != canonical.polarization_updates)
    return "installed polarization rows differ from the descriptor-authoritative plan";
  if (installed_ops.size() != canonical_ops.size())
    return "installed polarization operation count is noncanonical";
  for (size_t i = 0; i < installed_ops.size(); ++i)
    if (!same_polarization_operation(*installed_ops[i], *canonical_ops[i]))
      return "installed polarization operation span or access set is noncanonical";
  return std::string();
}

std::string validate_noisy_snapshot(const RandomSeedSnapshot &snapshot) {
  if (!snapshot.initialized || !snapshot.semantic_seed_valid || snapshot.generation == 0)
    return "noisy RNG semantic seed is not valid";
  if (snapshot.algorithm_version != counter_random_algorithm_version)
    return "noisy RNG algorithm version is unsupported";
  if (!snapshot.saved_semantic_seed_valid && snapshot.saved_explicit_seed)
    return "noisy RNG saved-seed metadata is inconsistent";
  return std::string();
}

} // namespace

std::string backend_validate_multilevel_plan(const fields &f, const StepPlan &plan,
                                             bool &local_multilevel_actions) {
  return validate_multilevel_plan(f, plan, local_multilevel_actions);
}

void backend_refresh_noisy_seed(fields &f, const StepPlan &plan, const char *site) {
  if (!f.backend || !f.backend->requires_full_storage_preparation()) return;
  if (!f.backend_state) throw std::logic_error(std::string(site) + ": missing backend state");
  if (!f.backend_state->noisy_preflight_required) return;

  BackendState &state = *f.backend_state;
  std::string local_error;
  bool local_noisy_actions = false;
  size_t local_stream_count = state.noisy_stream_count;
  uint64_t local_first_stream_tag = state.noisy_first_stream_tag;
  for (const PolarizationUpdate &update : plan.polarization_updates)
    local_noisy_actions =
        local_noisy_actions || update.kind == PolarizationUpdateKind::noisy_add;
  const bool injected_static_validation = noisy_preflight_failure_mode_for_testing == 4 ||
                                          noisy_preflight_failure_mode_for_testing == 6;
  const bool validate_static = state.noisy_static_validation_required ||
                               injected_static_validation;

  if (validate_static) {
    try {
      local_error = validate_noisy_plan(f, plan, local_noisy_actions, local_stream_count,
                                        local_first_stream_tag);
    }
    catch (const std::exception &e) {
      local_error = e.what();
    }
    catch (...) {
      local_error = "unknown noisy plan validation failure";
    }
    size_t local_status = size_t(!local_error.empty()) | (size_t(local_noisy_actions) << 1);
    size_t global_status = 0;
    backend_note_noisy_collective_for_testing();
    bw_or_to_all(&local_status, &global_status, 1);
    if (global_status & 1) {
      if (local_error.empty())
        throw std::runtime_error(std::string(site) +
                                 ": noisy plan validation failed on another MPI rank");
      throw std::runtime_error(std::string(site) + ": " + local_error);
    }
    state.noisy_preflight_required = (global_status & 2) != 0;
    state.noisy_static_validation_required = false;
    state.noisy_plan_validated = true;
    state.noisy_validated_plan_signature = plan.signature;
    state.noisy_stream_count = local_stream_count;
    state.noisy_first_stream_tag = local_first_stream_tag;
    if (!state.noisy_preflight_required) return;
  }

  RandomSeedSnapshot candidate = ensure_random_seed_snapshot();
  if (noisy_preflight_failure_rank_for_testing == my_rank()) {
    if (noisy_preflight_failure_mode_for_testing == 1) candidate.semantic_seed_valid = false;
    if (noisy_preflight_failure_mode_for_testing == 2) ++candidate.algorithm_version;
    if (noisy_preflight_failure_mode_for_testing == 3) candidate.generation = 0;
    if (noisy_preflight_failure_mode_for_testing == 5) ++candidate.saved_semantic_seed;
    if (noisy_preflight_failure_mode_for_testing == 7)
      candidate.generation = state.accepted_random_seed.generation > 1
                                 ? state.accepted_random_seed.generation - 1
                                 : 0;
  }
  local_error = validate_noisy_snapshot(candidate);
  if (local_error.empty() && state.random_seed_snapshot_accepted &&
      candidate.generation < state.accepted_random_seed.generation)
    local_error = "noisy RNG generation regressed";
  if (local_error.empty() && state.random_seed_snapshot_accepted &&
      candidate.generation == state.accepted_random_seed.generation &&
      !full_seed_snapshot_equal(candidate, state.accepted_random_seed))
    local_error = "noisy RNG snapshot changed without a new generation";
  const bool refresh = local_error.empty() &&
                       (!state.random_seed_snapshot_accepted ||
                        state.accepted_random_seed.generation != candidate.generation);
  size_t local_status = size_t(!local_error.empty()) | (size_t(refresh) << 1);
  size_t global_status = 0;
  backend_note_noisy_collective_for_testing();
  bw_or_to_all(&local_status, &global_status, 1);
  if (global_status & 1) {
    if (local_error.empty())
      throw std::runtime_error(std::string(site) +
                               ": noisy seed validation failed on another MPI rank");
    throw std::runtime_error(std::string(site) + ": " + local_error);
  }
  const bool any_refresh = (global_status & 2) != 0;
  if (any_refresh) {
    try {
      if (refresh) f.backend->refresh_noisy_seed(candidate, state);
    }
    catch (const std::exception &e) {
      local_error = e.what();
    }
    catch (...) {
      local_error = "unknown noisy seed refresh failure";
    }
    local_status = size_t(!local_error.empty()) | (size_t(f.backend->is_poisoned()) << 1);
    global_status = 0;
    backend_note_noisy_collective_for_testing();
    bw_or_to_all(&local_status, &global_status, 1);
    const bool hook_failed = (global_status & 1) != 0;
    const bool hook_poisoned = (global_status & 2) != 0;
    if (hook_poisoned) f.backend->poison();
    if (hook_failed || hook_poisoned) {
      if (refresh) f.backend->discard_noisy_seed(state);
      if (local_error.empty())
        throw std::runtime_error(std::string(site) +
                                 (hook_poisoned ? ": noisy seed refresh poisoned another MPI rank"
                                                : ": noisy seed refresh failed on another MPI rank"));
      throw std::runtime_error(std::string(site) + ": " + local_error);
    }
    if (refresh) f.backend->commit_noisy_seed(state);
  }
  if (refresh) {
    state.accepted_random_seed = candidate;
    state.random_seed_snapshot_accepted = true;
  }
}

namespace {

bool has_cw_source_amplitude(const fields &f) {
  bool present = false;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    const fields_chunk &fc = *f.chunks[chunk];
    if (!fc.is_mine()) continue;
    FOR_FIELD_TYPES(ft) for (const src_vol &source : fc.get_sources(ft))
      for (size_t point = 0; point < source.num_points(); ++point)
        present = present || source.amplitude_at(point) != std::complex<double>(0.0, 0.0);
  }
  return or_to_all(present);
}

bool has_cw_material_topology(const fields &f) {
  bool unsupported = false;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    const structure_chunk *s = f.chunks[chunk]->s;
    if (!s) continue;
    unsupported = unsupported || s->has_nonlinearities();
    FOR_FIELD_TYPES(ft) unsupported = unsupported || s->chiP[ft] != NULL;
  }
  return or_to_all(unsupported);
}

void audit_resident_cw_layout(const fields &f, const CwStateLayout &layout) {
  if (is_dirty(f, dirty_storage))
    throw std::logic_error("resident solve_cw requires clean prepared storage");
  if (!f.array_catalog || audit_storage_catalog(const_cast<fields &>(f), *f.array_catalog, false))
    throw std::logic_error("resident solve_cw requires a complete live storage catalog");

  size_t row_index = 0;
  auto require_pair = [&](int chunk, const fields_chunk &fc, component traversal, component storage,
                          CwStateFamily family, realnum *real_array, realnum *imag_array,
                          bool primary_present) {
    if ((real_array != NULL) != (imag_array != NULL))
      throw std::logic_error("resident solve_cw found a live half-pair");
    if (!real_array) return;
    if (!primary_present)
      throw std::logic_error("resident solve_cw found optional state without its primary field");
    if (row_index >= layout.rows.size())
      throw std::logic_error("resident solve_cw layout omits live field state");
    const CwStateRow &row = layout.rows[row_index++];
    if (row.chunk != chunk || row.traversal_component != traversal ||
        row.storage_component != storage || row.family != family ||
        !is_valid(row.real_array) || !is_valid(row.imag_array) ||
        f.array_catalog->resolve_untyped(row.real_array) != real_array ||
        f.array_catalog->resolve_untyped(row.imag_array) != imag_array ||
        row.complex_count != size_t(fc.gv.nowned(traversal)))
      throw std::logic_error("resident solve_cw layout does not exactly cover live field state");
  };

  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    const fields_chunk &fc = *f.chunks[chunk];
    FOR_COMPONENTS(c) {
      if (!is_D(c) && !is_B(c)) continue;
      const bool primary_present = fc.f[c][0] && fc.f[c][1];
      require_pair(chunk, fc, c, c, CwStateFamily::primary, fc.f[c][0], fc.f[c][1],
                   primary_present);
      require_pair(chunk, fc, c, c, CwStateFamily::pml_u, fc.f_u[c][0], fc.f_u[c][1],
                   primary_present);
      require_pair(chunk, fc, c, c, CwStateFamily::conductivity, fc.f_cond[c][0],
                   fc.f_cond[c][1], primary_present);
      require_pair(chunk, fc, c, c, CwStateFamily::bfast, fc.f_bfast[c][0], fc.f_bfast[c][1],
                   primary_present);
      const component paired = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
      require_pair(chunk, fc, c, paired, CwStateFamily::constitutive_w, fc.f_w[paired][0],
                   fc.f_w[paired][1], primary_present);
      if (fc.f_w[paired][0])
        require_pair(chunk, fc, c, paired, CwStateFamily::paired_primary, fc.f[paired][0],
                     fc.f[paired][1], primary_present);
    }
  }
  if (row_index != layout.rows.size())
    throw std::logic_error("resident solve_cw layout contains a non-live field row");
}

std::string cheap_cw_rejection(const fields &f, const CwSolveRequest &request,
                               bool live_magnetic_snapshot) {
  if (f.backend->is_poisoned()) return "resident backend is poisoned";
  if (!std::isfinite(request.tolerance) || request.tolerance <= 0.0)
    return "solve_cw tolerance must be finite and positive";
  if (!std::isfinite(real(request.frequency)) || !std::isfinite(imag(request.frequency)) ||
      request.frequency == std::complex<double>(0.0, 0.0))
    return "solve_cw frequency must be finite and nonzero";
  if (request.L < 1) return "solve_cw requires L >= 1";
  if (request.maxiters < 1) return "solve_cw requires maxiters >= 1";
  if (request.eigfrequency)
    return "resident solve_cw does not support eigfrequency/shift-invert requests";
  if (f.is_real) return "resident solve_cw does not support real fields";
  if (count_processors() != 1) return "resident solve_cw does not support MPI decomposition";
  if (f.gv.dim == Dcyl) return "resident solve_cw supports only Cartesian coordinates";
  if (f.beta != 0.0) return "resident solve_cw does not support beta coordinates";
  for (double value : f.bfast_scaled_k)
    if (value != 0.0) return "resident solve_cw does not support BFAST coordinates";
  if (f.phasein_time > 0) return "resident solve_cw does not support active material phasing";
  if (live_magnetic_snapshot)
    return "resident solve_cw does not support a live magnetic snapshot";
  if (f.fluxes) return "resident solve_cw does not support legacy flux accumulators";
  if (has_cw_material_topology(f))
    return "resident solve_cw does not support dispersion, polarization, or nonlinear media";
  if (!has_cw_source_amplitude(f)) return "resident solve_cw requires a nonzero source amplitude";
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    if (f.chunks[chunk]->is_solving_cw()) return "resident solve_cw is already active";
  return std::string();
}

struct field_refresh_data {
  fields *owner;
  int count;
  const component *components;
  std::string local_error;
};

void refresh_field_chunk(fields_chunk *fc, int, component cgrid, ivec is, ivec ie, vec, vec, vec,
                         vec, double, double, ivec, std::complex<double>, const symmetry &S, int sn,
                         void *data_) {
  field_refresh_data *data = static_cast<field_refresh_data *>(data_);
  if (!data->local_error.empty()) return;

  for (int i = 0; i < data->count; ++i) {
    const component c = S.transform(data->components[i], -sn);
    if (c == Dielectric || c == Permeability || c == NO_COMPONENT) continue;

    ptrdiff_t offset1 = 0, offset2 = 0;
    if (cgrid == Centered) fc->gv.yee2cent_offsets(c, offset1, offset2);
    ptrdiff_t minimum = std::numeric_limits<ptrdiff_t>::max();
    ptrdiff_t maximum = std::numeric_limits<ptrdiff_t>::min();
    LOOP_OVER_IVECS(fc->gv, is, ie, idx) {
      const ptrdiff_t candidates[4] = {idx, idx + offset1, idx + offset2, idx + offset1 + offset2};
      for (int k = 0; k < 4; ++k) {
        minimum = std::min(minimum, candidates[k]);
        maximum = std::max(maximum, candidates[k]);
      }
    }
    if (minimum < 0 || maximum < minimum) {
      data->local_error = "invalid field range during resident host refresh";
      return;
    }
    for (int cmp = 0; cmp < 2; ++cmp)
      if (fc->f[c][cmp] &&
          !backend_read_host_range(*data->owner, fc->f[c][cmp] + minimum,
                                   size_t(maximum - minimum) + 1, data->local_error))
        return;
  }
}

struct point_sampler {
  component c;
  vec loc;
  int interval;
  std::vector<std::complex<double> > samples;
};

/* Registered samplers, keyed by the id handed back. Kept here rather than in
   `fields` so the container types stay out of meep.hpp. */
std::vector<std::vector<point_sampler> > &sampler_registry() {
  static std::vector<std::vector<point_sampler> > registry;
  return registry;
}

/* The hook runs before either object is destroyed, so a resident backend can
   synchronize/migrate authoritative values or reject the rebuild without
   losing its live state. */
void release_backend_state_for_rebuild(fields &f, DirtyMask reasons) {
  if (f.backend_state) f.backend->prepare_state_rebuild(*f.backend_state, reasons);
  delete f.executable;
  f.executable = NULL;
  destroy_backend_state(f.backend_state);
}

} // namespace

double cw_source_time(int t, double dt, double offset_in_dt) {
  volatile double tnow = double(t) * dt;
  return tnow + offset_in_dt * dt;
}

void destroy_backend_state(BackendState *&state) {
  if (!state) return;
  state->clear_cw_executable();
  delete state;
  state = NULL;
}

CwSolveSession::CwSolveSession(fields &owner, const CwSolveRequest &request)
    : owner_(owner), entry_t_(request.entry_t), boundary_called_(false) {
  for (int i = 0; i < owner_.num_chunks; ++i)
    owner_.chunks[i]->set_solve_cw_omega(2.0 * pi * request.frequency);
}

CwSolveSession::~CwSolveSession() {
  for (int i = 0; i < owner_.num_chunks; ++i)
    owner_.chunks[i]->unset_solve_cw_omega();
  owner_.t = entry_t_;
}

void CwSolveSession::restore_before_final_dft() noexcept {
  for (int i = 0; i < owner_.num_chunks; ++i)
    owner_.chunks[i]->unset_solve_cw_omega();
  owner_.t = entry_t_;
  boundary_called_ = true;
}

bool CwSolveSession::at_entry_state() const {
  if (owner_.t != entry_t_) return false;
  for (int i = 0; i < owner_.num_chunks; ++i)
    if (owner_.chunks[i]->is_solving_cw()) return false;
  return true;
}

class PreparedBackendEpoch {
public:
  explicit PreparedBackendEpoch(fields &owner)
      : owner_(owner), committed_(false), old_chunks_(owner.chunks), old_halos_(owner.halos),
        old_catalog_(owner.array_catalog), old_storage_(owner.storage_plan),
        old_descriptors_(owner.descriptors), old_initialization_(owner.initialization_plan),
        old_state_(owner.backend_state), old_executable_(owner.executable),
        old_dirty_mask_(owner.dirty_mask), old_components_allocated_(owner.components_allocated),
        old_connections_generation_(owner.connections_generation),
        old_connections_built_generation_(owner.connections_built_generation),
        old_local_invalidation_generation_(owner.local_invalidation_generation),
        old_local_invalidation_synced_(owner.local_invalidation_synced),
        old_storage_prepared_mask_(owner.storage_prepared_mask),
        old_prepared_classification_hash_(owner.prepared_classification_hash),
        old_classification_reentries_(owner.classification_reentries),
        old_chunk_connections_valid_(owner.chunk_connections_valid),
        old_changed_materials_(owner.changed_materials) {
    old_step_plans_[0] = owner.step_plans[0];
    old_step_plans_[1] = owner.step_plans[1];
    for (int i = 0; i < fields::num_mutation_kinds; ++i)
      old_mutation_generation_[i] = owner.mutation_generation[i];
    for (int ft = 0; ft < NUM_FIELD_TYPES; ++ft) old_comm_blocks_[ft] = owner.comm_blocks[ft];

    std::vector<std::unique_ptr<fields_chunk> > chunks(size_t(owner.num_chunks));
    for (int i = 0; i < owner.num_chunks; ++i) {
      chunks[size_t(i)].reset(new fields_chunk(*old_chunks_[i], i));
      fields_chunk &staged = *chunks[size_t(i)];
      const fields_chunk &old = *old_chunks_[i];
      std::unique_ptr<structure_chunk> staged_structure(new structure_chunk(old.s));
      staged_structure->update_condinv();
      if (staged.s->refcount-- <= 1) delete staged.s;
      staged.s = staged_structure.release();
      FOR_FIELD_TYPES(ft) {
        std::vector<src_vol> staged_sources(old.sources[ft]);
        staged.sources[ft].swap(staged_sources);
        staged.npol[ft] = old.npol[ft];
      }
      DOCMP2 FOR_COMPONENTS(c) {
        const size_t n = size_t(old.gv.ntot());
#define COPY_BACKUP(name)                                                                          \
  if (old.name[c][cmp]) {                                                                          \
    staged.name[c][cmp] = new realnum[n];                                                          \
    memcpy(staged.name[c][cmp], old.name[c][cmp], n * sizeof(realnum));                            \
    backend_cw_clone_checkpoint();                                                                 \
  }
        COPY_BACKUP(f_backup)
        COPY_BACKUP(f_u_backup)
        COPY_BACKUP(f_w_backup)
        COPY_BACKUP(f_cond_backup)
        COPY_BACKUP(f_bfast_backup)
#undef COPY_BACKUP
      }
      if (old.f_rderiv_int) {
        staged.f_rderiv_int = new realnum[old.gv.ntot()];
        memcpy(staged.f_rderiv_int, old.f_rderiv_int, old.gv.ntot() * sizeof(realnum));
        backend_cw_clone_checkpoint();
      }
      backend_cw_clone_checkpoint();
    }

    std::unique_ptr<fields_chunk *[]> chunk_array(new fields_chunk *[size_t(owner.num_chunks)]);
    backend_cw_clone_checkpoint();
    for (int i = 0; i < owner.num_chunks; ++i) chunk_array[size_t(i)] = chunks[size_t(i)].get();
    std::unique_ptr<halo_plan_set> halos(new halo_plan_set);
    backend_cw_clone_checkpoint();
    std::unique_ptr<CpuArrayCatalog> catalog(new CpuArrayCatalog);
    backend_cw_clone_checkpoint();
    std::unique_ptr<StoragePlan> storage(new StoragePlan);
    backend_cw_clone_checkpoint();
    std::unique_ptr<DescriptorSet> descriptors(new DescriptorSet);
    backend_cw_clone_checkpoint();
    std::unique_ptr<realnum *[]> comm_blocks[NUM_FIELD_TYPES];
    for (int ft = 0; ft < NUM_FIELD_TYPES; ++ft) {
      comm_blocks[ft].reset(new realnum *[size_t(owner.num_chunks) * size_t(owner.num_chunks)]);
      backend_cw_clone_checkpoint();
      std::fill(comm_blocks[ft].get(),
                comm_blocks[ft].get() + size_t(owner.num_chunks) * size_t(owner.num_chunks),
                static_cast<realnum *>(NULL));
    }

    /* No operation below allocates or copies. DFT chains remain attached to
       the live chunks until this final publication tail. */
    for (int i = 0; i < owner.num_chunks; ++i) {
      chunks[size_t(i)]->dft_chunks = old_chunks_[i]->dft_chunks;
      old_chunks_[i]->dft_chunks = NULL;
      for (dft_chunk *dft = chunks[size_t(i)]->dft_chunks; dft; dft = dft->next_in_chunk)
        dft->fc = chunks[size_t(i)].get();
    }
    owner.chunks = chunk_array.release();
    for (int i = 0; i < owner.num_chunks; ++i) chunks[size_t(i)].release();
    owner.halos = halos.release();
    owner.array_catalog = catalog.release();
    owner.storage_plan = storage.release();
    owner.descriptors = descriptors.release();
    owner.initialization_plan = NULL;
    owner.step_plans[0] = owner.step_plans[1] = NULL;
    owner.backend_state = NULL;
    owner.executable = NULL;
    old_comm_sizes_.swap(owner.comm_sizes);
    for (int ft = 0; ft < NUM_FIELD_TYPES; ++ft) {
      owner.comm_blocks[ft] = comm_blocks[ft].release();
      std::swap(old_comms_sequence_[ft], owner.comms_sequence_for_field[ft]);
    }
    owner.dirty_mask |= dirty_initialization | dirty_source_plan | dirty_monitor_plan |
                        dirty_storage | dirty_regions | dirty_halos | dirty_executable |
                        dirty_classification;
    owner.storage_prepared_mask = 0;
    owner.prepared_classification_hash = 0;
    owner.classification_reentries = 0;
    owner.connections_built_generation = 0;
    note_connections_invalidated(owner);
    mark_local_invalidation(owner);
    owner.chunk_connections_valid = false;
    owner.changed_materials = true;
  }

  ~PreparedBackendEpoch() {
    if (!committed_) restore();
  }

  void commit() {
    if (committed_) return;
    fields_chunk **staged_chunks = owner_.chunks;
    for (int i = 0; i < owner_.num_chunks; ++i) {
      fields_chunk &live = *old_chunks_[i];
      fields_chunk &staged = *staged_chunks[i];
      live.swap_prepared_state(staged);
      live.dft_chunks = staged.dft_chunks;
      staged.dft_chunks = NULL;
      for (dft_chunk *dft = live.dft_chunks; dft; dft = dft->next_in_chunk) dft->fc = &live;
    }
    owner_.chunks = old_chunks_;
    if (old_state_) old_state_->clear_cw_executable();
    delete old_executable_;
    destroy_backend_state(old_state_);
    delete old_initialization_;
    delete old_step_plans_[0];
    delete old_step_plans_[1];
    delete old_descriptors_;
    delete old_catalog_;
    delete old_storage_;
    delete old_halos_;
    delete_comm_blocks(old_comm_blocks_);
    for (int i = 0; i < owner_.num_chunks; ++i) delete staged_chunks[i];
    delete[] staged_chunks;
    committed_ = true;
  }

private:
  PreparedBackendEpoch(const PreparedBackendEpoch &);
  PreparedBackendEpoch &operator=(const PreparedBackendEpoch &);

  void delete_comm_blocks(realnum **blocks[NUM_FIELD_TYPES]) {
    for (int ft = 0; ft < NUM_FIELD_TYPES; ++ft) {
      if (!blocks[ft]) continue;
      for (int i = 0; i < owner_.num_chunks * owner_.num_chunks; ++i) delete[] blocks[ft][i];
      delete[] blocks[ft];
      blocks[ft] = NULL;
    }
  }

  void restore() {
    fields_chunk **staged_chunks = owner_.chunks;
    halo_plan_set *staged_halos = owner_.halos;
    CpuArrayCatalog *staged_catalog = owner_.array_catalog;
    StoragePlan *staged_storage = owner_.storage_plan;
    DescriptorSet *staged_descriptors = owner_.descriptors;
    InitializationPlan *staged_initialization = owner_.initialization_plan;
    StepPlan *staged_step_plans[2] = {owner_.step_plans[0], owner_.step_plans[1]};
    Executable *staged_executable = owner_.executable;
    BackendState *staged_state = owner_.backend_state;
    realnum **staged_comm_blocks[NUM_FIELD_TYPES];
    for (int ft = 0; ft < NUM_FIELD_TYPES; ++ft)
      staged_comm_blocks[ft] = owner_.comm_blocks[ft];

    if (staged_state) staged_state->clear_cw_executable();
    delete staged_executable;
    destroy_backend_state(staged_state);
    for (int i = 0; i < owner_.num_chunks; ++i) {
      old_chunks_[i]->dft_chunks = staged_chunks[i]->dft_chunks;
      for (dft_chunk *dft = old_chunks_[i]->dft_chunks; dft; dft = dft->next_in_chunk)
        dft->fc = old_chunks_[i];
      staged_chunks[i]->dft_chunks = NULL;
    }
    owner_.chunks = old_chunks_;
    owner_.halos = old_halos_;
    owner_.array_catalog = old_catalog_;
    owner_.storage_plan = old_storage_;
    owner_.descriptors = old_descriptors_;
    owner_.initialization_plan = old_initialization_;
    owner_.step_plans[0] = old_step_plans_[0];
    owner_.step_plans[1] = old_step_plans_[1];
    owner_.backend_state = old_state_;
    owner_.executable = old_executable_;
    for (int ft = 0; ft < NUM_FIELD_TYPES; ++ft) {
      owner_.comm_blocks[ft] = old_comm_blocks_[ft];
      std::swap(owner_.comms_sequence_for_field[ft], old_comms_sequence_[ft]);
    }
    owner_.comm_sizes.swap(old_comm_sizes_);
    owner_.dirty_mask = old_dirty_mask_;
    owner_.components_allocated = old_components_allocated_;
    for (int i = 0; i < fields::num_mutation_kinds; ++i)
      owner_.mutation_generation[i] = old_mutation_generation_[i];
    owner_.connections_generation = old_connections_generation_;
    owner_.connections_built_generation = old_connections_built_generation_;
    owner_.local_invalidation_generation = old_local_invalidation_generation_;
    owner_.local_invalidation_synced = old_local_invalidation_synced_;
    owner_.storage_prepared_mask = old_storage_prepared_mask_;
    owner_.prepared_classification_hash = old_prepared_classification_hash_;
    owner_.classification_reentries = old_classification_reentries_;
    owner_.chunk_connections_valid = old_chunk_connections_valid_;
    owner_.changed_materials = old_changed_materials_;

    delete staged_initialization;
    delete staged_step_plans[0];
    delete staged_step_plans[1];
    delete staged_descriptors;
    delete staged_catalog;
    delete staged_storage;
    delete staged_halos;
    delete_comm_blocks(staged_comm_blocks);
    for (int i = 0; i < owner_.num_chunks; ++i) delete staged_chunks[i];
    delete[] staged_chunks;
  }

  fields &owner_;
  bool committed_;
  fields_chunk **old_chunks_;
  halo_plan_set *old_halos_;
  CpuArrayCatalog *old_catalog_;
  StoragePlan *old_storage_;
  DescriptorSet *old_descriptors_;
  InitializationPlan *old_initialization_;
  StepPlan *old_step_plans_[2];
  BackendState *old_state_;
  Executable *old_executable_;
  realnum **old_comm_blocks_[NUM_FIELD_TYPES];
  std::unordered_map<comms_key, size_t, comms_key_hash_fn> old_comm_sizes_;
  comms_sequence old_comms_sequence_[NUM_FIELD_TYPES];
  DirtyMask old_dirty_mask_;
  bool old_components_allocated_;
  uint64_t old_mutation_generation_[fields::num_mutation_kinds];
  uint64_t old_connections_generation_;
  uint64_t old_connections_built_generation_;
  uint64_t old_local_invalidation_generation_;
  uint64_t old_local_invalidation_synced_;
  uint32_t old_storage_prepared_mask_;
  uint64_t old_prepared_classification_hash_;
  uint32_t old_classification_reentries_;
  bool old_chunk_connections_valid_;
  bool old_changed_materials_;
};

/* Reversible storage/connection epoch used only by resident solve_cw
   preparation. The supported PR7 slice has no polarization, BFAST,
   cylindrical scratch, material phase, or magnetic backup topology, so the
   existing storage preparation may only fill previously-null raw slots. */
backend_capabilities fields::backend_caps() const {
  if (backend) return backend->capabilities();
  backend_capabilities c;
  c.supports_native = true;
  c.supports_mixed = c.supports_f32 = false;
  c.memory_budget_bytes = 0;
  c.name = "none";
  return c;
}

bool backend_host_refresh_required(const fields &f) {
  return f.backend && f.backend_state && f.backend->requires_full_storage_preparation();
}

bool backend_read_host_range(const fields &f, const void *host_address, size_t elements,
                             std::string &local_error) {
  if (!local_error.empty()) return false;
  try {
    if (!host_address || !elements || !backend_host_refresh_required(f)) return true;
    if (f.backend->is_poisoned())
      throw std::logic_error("resident backend is poisoned by a failed transition");
    if (!f.array_catalog)
      throw std::logic_error("resident backend access requires a storage catalog");

    ArrayId id;
    ptrdiff_t offset;
    if (!f.array_catalog->locate(host_address, id, offset) || offset < 0)
      throw std::out_of_range("host access does not name catalogued backend storage");
    const ArraySpec &spec = f.array_catalog->spec(id);
    if (size_t(offset) > spec.elements || elements > spec.elements - size_t(offset))
      throw std::out_of_range("host access exceeds catalogued backend storage");
    const size_t element_bytes = host_element_bytes(spec.element_type);
    if (elements > std::numeric_limits<size_t>::max() / element_bytes)
      throw std::overflow_error("backend host-access byte count overflow");
    f.backend->read(ArrayRef{id, size_t(offset), elements}, const_cast<void *>(host_address),
                    elements * element_bytes);
    return true;
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown backend host-access failure";
  }
  return false;
}

bool backend_write_host_range(fields &f, const void *host_address, size_t elements,
                              std::string &local_error) {
  if (!local_error.empty()) return false;
  try {
    if (!host_address || !elements || !backend_host_refresh_required(f)) return true;
    if (f.backend->is_poisoned())
      throw std::logic_error("resident backend is poisoned by a failed transition");
    if (!f.array_catalog)
      throw std::logic_error("resident backend access requires a storage catalog");

    ArrayId id;
    ptrdiff_t offset;
    if (!f.array_catalog->locate(host_address, id, offset) || offset < 0)
      throw std::out_of_range("host access does not name catalogued backend storage");
    const ArraySpec &spec = f.array_catalog->spec(id);
    if (size_t(offset) > spec.elements || elements > spec.elements - size_t(offset))
      throw std::out_of_range("host access exceeds catalogued backend storage");
    const size_t element_bytes = host_element_bytes(spec.element_type);
    if (elements > std::numeric_limits<size_t>::max() / element_bytes)
      throw std::overflow_error("backend host-access byte count overflow");
    f.backend->write(ArrayRef{id, size_t(offset), elements}, host_address,
                     elements * element_bytes);
    return true;
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown backend host-access failure";
  }
  return false;
}

void backend_reconcile_host_access(const std::string &local_error, const char *site) {
  if (!or_to_all(!local_error.empty())) return;
  if (local_error.empty())
    throw std::runtime_error(std::string(site) +
                             ": backend host access failed on another MPI rank");
  throw std::runtime_error(std::string(site) + ": " + local_error);
}

void backend_publish_legacy_flux(fields &f, const double *values, size_t count, const char *site) {
  std::string local_error;
  size_t live_count = 0;
  for (const flux_vol *flux = f.fluxes; flux; flux = flux->next) ++live_count;
  if (count && !values)
    local_error = "legacy flux publication has no result buffer";
  else if (count != live_count)
    local_error = "legacy flux publication result count differs from the live list";
  else if (!f.descriptors || f.descriptors->legacy_fluxes.size() != live_count ||
           f.descriptors->legacy_flux_generation !=
               generation(f, MutationKind::legacy_flux_definition))
    local_error = "legacy flux publication has stale descriptors";
  else
    for (size_t ordinal = 0; ordinal < live_count; ++ordinal)
      if (f.descriptors->legacy_fluxes[ordinal].flux_ordinal != ordinal) {
        local_error = "legacy flux publication has a noncanonical list ordinal";
        break;
      }
  backend_reconcile_host_access(local_error, site);

  size_t ordinal = 0;
  for (flux_vol *flux = f.fluxes; flux; flux = flux->next, ++ordinal)
    flux->cur_flux = values[ordinal];
}

static bool is_legacy_flux_marker(OpKind kind) {
  return kind == OpKind::update_flux_half || kind == OpKind::update_flux;
}

StepPlan build_legacy_flux_only_step_plan(fields &f, StepProgram program,
                                          const StepPlan &stable) {
  if (stable.program != program)
    throw std::logic_error("legacy flux refresh stable plan has the wrong program");
  const StepPlan fresh = build_step_plan(f, program);
  StepPlan replacement = stable;
  replacement.legacy_flux_updates = fresh.legacy_flux_updates;
  replacement.legacy_flux_terms = fresh.legacy_flux_terms;
  replacement.operations.clear();
  replacement.operations.reserve(fresh.operations.size());

  size_t stable_index = 0;
  for (const Operation &fresh_op : fresh.operations) {
    if (is_legacy_flux_marker(fresh_op.kind)) {
      replacement.operations.push_back(fresh_op);
      continue;
    }
    while (stable_index < stable.operations.size() &&
           is_legacy_flux_marker(stable.operations[stable_index].kind))
      ++stable_index;
    if (stable_index == stable.operations.size())
      throw std::logic_error("legacy flux refresh changed the non-flux operation count");
    const Operation &stable_op = stable.operations[stable_index++];
    if (stable_op.kind != fresh_op.kind || stable_op.ft != fresh_op.ft ||
        stable_op.source_time_offset != fresh_op.source_time_offset ||
        stable_op.guard.kind != fresh_op.guard.kind ||
        stable_op.guard.scalar_slot != fresh_op.guard.scalar_slot ||
        stable_op.guard.variant_index != fresh_op.guard.variant_index)
      throw std::logic_error("legacy flux refresh changed the non-flux operation schedule");
    replacement.operations.push_back(stable_op);
  }
  while (stable_index < stable.operations.size() &&
         is_legacy_flux_marker(stable.operations[stable_index].kind))
    ++stable_index;
  if (stable_index != stable.operations.size())
    throw std::logic_error("legacy flux refresh dropped a non-flux operation");
  replacement.signature = compute_step_plan_signature(replacement);
  return replacement;
}

bool backend_try_refresh_legacy_flux(fields &f, const char *site) {
  if (!f.backend || !f.backend->requires_full_storage_preparation()) return false;
  const bool refresh = or_to_all(is_dirty(f, dirty_flux_plan));
  if (!refresh) return false;

  /* The small replacement below is valid only when the remaining dirty state
     is exactly the legacy-flux closure.  In particular, it must not clear
     dirty_executable while a boundary/material/storage mutation still needs
     the conservative staged epoch.  Source/monitor descriptors have already
     been refreshed by init_backend before this hook is reached. */
  const DirtyMask flux_closure =
      DirtyMask(dirty_flux_plan | dirty_regions | dirty_executable);
  bool stable_provenance_mismatch = false;
  if (f.step_plans[0]) {
    stable_provenance_mismatch =
        !f.descriptors ||
        f.step_plans[0]->source_signature != source_plan_signature(f.descriptors->sources) ||
        dft_plan_signature(f.step_plans[0]->dft_updates) !=
            dft_plan_signature(f.descriptors->dfts);
  }
  const bool local_multilevel_presence =
      has_local_exact_multilevel(f) ||
      (f.backend_state && f.backend_state->multilevel_preflight_required);
  size_t local_shape =
      size_t((f.dirty_mask & ~flux_closure) != dirty_none || stable_provenance_mismatch) |
      (size_t(local_multilevel_presence) << 1);
  size_t global_shape = 0;
  bw_or_to_all(&local_shape, &global_shape, 1);
  const bool mixed_structural = (global_shape & 1) != 0;
  const bool any_multilevel = (global_shape & 2) != 0;

  std::string local_error;
  size_t local_definition = size_t(legacy_flux_definition_signature(f));
  size_t reference_definition = local_definition;
  broadcast(0, &reference_definition, 1);
  if (local_definition != reference_definition)
    local_error = "legacy flux definitions differ across MPI ranks";
  if (legacy_flux_prepare_failure_rank_for_testing == my_rank())
    local_error = "injected legacy flux descriptor preparation failure";
  backend_reconcile_host_access(local_error, site);

  if (mixed_structural) {
    if (f.backend_state) try {
        f.backend->prepare_state_rebuild(*f.backend_state, DirtyMask(f.dirty_mask));
      }
      catch (const std::exception &e) {
        local_error = e.what();
      }
      catch (...) {
        local_error = "unknown legacy flux state-migration failure";
      }
    backend_reconcile_host_access(local_error, site);

    std::unique_ptr<PreparedBackendEpoch> prepared_epoch;
    try {
      prepared_epoch.reset(new PreparedBackendEpoch(f));
    }
    catch (const std::exception &e) {
      local_error = e.what();
    }
    catch (...) {
      local_error = "unknown legacy flux epoch preparation failure";
    }
    const bool prepare_failed = or_to_all(!local_error.empty());
    if (prepare_failed) {
      prepared_epoch.reset();
      if (local_error.empty())
        throw std::runtime_error(std::string(site) +
                                 ": legacy flux epoch preparation failed on another MPI rank");
      throw std::runtime_error(std::string(site) + ": " + local_error);
    }

    try {
      f.require_source_components();
      f.init_backend();
      f.ensure_backend_executable();
    }
    catch (const std::exception &e) {
      local_error = e.what();
    }
    catch (...) {
      local_error = "unknown legacy flux epoch compilation failure";
    }
    const bool compile_failed = or_to_all(!local_error.empty());
    if (compile_failed) {
      prepared_epoch.reset();
      if (local_error.empty())
        throw std::runtime_error(std::string(site) +
                                 ": legacy flux epoch compilation failed on another MPI rank");
      throw std::runtime_error(std::string(site) + ": " + local_error);
    }
    prepared_epoch->commit();
    return true;
  }

  std::unique_ptr<DescriptorSet> replacement_descriptors;
  std::unique_ptr<StepPlan> replacement_plan;
  Executable *replacement_executable = NULL;
  DescriptorSet *live_descriptors = f.descriptors;
  try {
    replacement_descriptors.reset(new DescriptorSet(*live_descriptors));
    build_legacy_flux_descriptors(f, replacement_descriptors->legacy_fluxes);
    replacement_descriptors->legacy_flux_generation =
        generation(f, MutationKind::legacy_flux_definition);
    replacement_descriptors->regions.clear();

    f.descriptors = replacement_descriptors.get();
    replacement_plan.reset(new StepPlan(
        f.step_plans[0]
            ? build_legacy_flux_only_step_plan(f, StepProgram::ordinary, *f.step_plans[0])
            : build_step_plan(f, StepProgram::ordinary)));
    if (any_multilevel) {
      bool local_multilevel_actions = false;
      local_error = validate_multilevel_plan(f, *replacement_plan, local_multilevel_actions);
      if (local_error.empty() && local_multilevel_actions != has_local_exact_multilevel(f))
        local_error = "multilevel live state and legacy-flux replacement actions differ";
    }
    if (local_error.empty()) {
      replacement_executable = f.backend->compile(*replacement_plan, *f.backend_state);
      if (!replacement_executable)
        throw std::runtime_error("backend returned no legacy-flux replacement executable");
    }
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown legacy flux refresh failure";
  }
  f.descriptors = live_descriptors;

  if (any_multilevel) backend_note_multilevel_collective_for_testing();
  const bool failed = or_to_all(!local_error.empty());
  if (failed) {
    delete replacement_executable;
    if (local_error.empty())
      throw std::runtime_error(std::string(site) +
                               ": legacy flux refresh failed on another MPI rank");
    throw std::runtime_error(std::string(site) + ": " + local_error);
  }

  DescriptorSet *old_descriptors = f.descriptors;
  StepPlan *old_plans[2] = {f.step_plans[0], f.step_plans[1]};
  Executable *old_executable = f.executable;
  f.descriptors = replacement_descriptors.release();
  f.step_plans[0] = replacement_plan.release();
  f.step_plans[1] = old_plans[1];
  f.executable = replacement_executable;
  if (f.backend_state) {
    f.backend_state->multilevel_preflight_required = any_multilevel;
    f.backend_state->multilevel_static_validation_required = false;
    f.backend_state->multilevel_plan_validated = any_multilevel;
    f.backend_state->multilevel_validated_plan_signature = f.step_plans[0]->signature;
  }
  clear_dirty(f, dirty_flux_plan | dirty_regions | dirty_executable);
  delete old_executable;
  delete old_plans[0];
  delete old_descriptors;
  return true;
}

void backend_preflight_field_layout_change(fields &f, DirtyMask reasons, const char *site) {
  std::string local_error;
  if (f.backend_state && f.backend && f.backend->requires_full_storage_preparation()) {
    if (f.backend->is_poisoned())
      local_error = "cannot change field layout after the resident backend was poisoned";
    else if (f.synchronized_magnetic_fields)
      local_error = "cannot change resident field layout while magnetic fields are synchronized";
  }
  if (f.backend_state) try {
      if (local_error.empty()) f.backend->prepare_state_rebuild(*f.backend_state, reasons);
    }
    catch (const std::exception &e) {
      local_error = e.what();
    }
    catch (...) {
      local_error = "unknown backend field-layout preparation failure";
    }
  backend_reconcile_host_access(local_error, site);
}

void backend_commit_field_layout_change(fields &f) {
  if (f.backend_state) {
    delete f.executable;
    f.executable = NULL;
    destroy_backend_state(f.backend_state);
  }
}

void backend_prepare_field_layout_change(fields &f, DirtyMask reasons, const char *site) {
  backend_preflight_field_layout_change(f, reasons, site);
  backend_commit_field_layout_change(f);
}

void backend_refresh_host_fields(fields &owner, int count, const component *components,
                                 const volume &where, component cgrid, bool use_symmetry,
                                 bool snap_empty_dimensions, const char *site) {
  if (!backend_host_refresh_required(owner)) return;
  field_refresh_data data = {&owner, count, components, std::string()};
  owner.loop_in_chunks(refresh_field_chunk, &data, where, cgrid, use_symmetry,
                       snap_empty_dimensions);
  backend_reconcile_host_access(data.local_error, site);
}

bool backend_read_dft_chunk(const dft_chunk *chunk, std::string &local_error) {
  if (!chunk || !local_error.empty()) return local_error.empty();
  fields *owner = chunk->monitor_lifetime ? chunk->monitor_lifetime->owner : NULL;
  if (!owner || !backend_host_refresh_required(*owner)) return true;

  ArrayId id;
  ptrdiff_t offset;
  if (!owner->array_catalog || !owner->array_catalog->locate(chunk->dft, id, offset)) {
    if (!chunk->attached_to_fields || is_dirty(*owner, dirty_storage)) return true;
  }
  return backend_read_host_range(*owner, chunk->dft, chunk->N * chunk->omega.size(), local_error);
}

bool backend_read_dft_chain(const dft_chunk *head, std::string &local_error) {
  for (const dft_chunk *cur = head; cur; cur = cur->next_in_dft)
    if (!backend_read_dft_chunk(cur, local_error)) return false;
  return true;
}

bool backend_write_dft_chunk(dft_chunk *chunk, std::string &local_error) {
  if (!chunk || !local_error.empty()) return local_error.empty();
  fields *owner = chunk->monitor_lifetime ? chunk->monitor_lifetime->owner : NULL;
  if (!owner || !backend_host_refresh_required(*owner)) return true;

  ArrayId id;
  ptrdiff_t offset;
  if (!owner->array_catalog || !owner->array_catalog->locate(chunk->dft, id, offset)) {
    if (!chunk->attached_to_fields || is_dirty(*owner, dirty_storage)) return true;
  }
  return backend_write_host_range(*owner, chunk->dft, chunk->N * chunk->omega.size(), local_error);
}

bool backend_write_dft_chain(dft_chunk *head, std::string &local_error) {
  for (dft_chunk *cur = head; cur; cur = cur->next_in_dft)
    if (!backend_write_dft_chunk(cur, local_error)) return false;
  return true;
}

void backend_refresh_dft_chains(fields &owner, int count, dft_chunk *const *heads,
                                const char *site) {
  if (!backend_host_refresh_required(owner)) return;
  std::string local_error;
  for (int i = 0; i < count; ++i)
    backend_read_dft_chain(heads[i], local_error);
  backend_reconcile_host_access(local_error, site);
}

void backend_refresh_dft_chain(const dft_chunk *head, const char *site) {
  bool local_refresh = false;
  for (const dft_chunk *cur = head; cur; cur = cur->next_in_dft) {
    fields *owner = cur->monitor_lifetime ? cur->monitor_lifetime->owner : NULL;
    local_refresh = local_refresh || (owner && backend_host_refresh_required(*owner));
  }
  std::string local_error;
  backend_read_dft_chain(head, local_error);
  if (or_to_all(local_refresh)) backend_reconcile_host_access(local_error, site);
}

void backend_publish_dft_chain(dft_chunk *head, const char *site) {
  bool local_publish = false;
  for (dft_chunk *cur = head; cur; cur = cur->next_in_dft) {
    fields *owner = cur->monitor_lifetime ? cur->monitor_lifetime->owner : NULL;
    local_publish = local_publish || (owner && backend_host_refresh_required(*owner));
  }
  std::string local_error;
  backend_write_dft_chain(head, local_error);
  if (or_to_all(local_publish)) backend_reconcile_host_access(local_error, site);
}

void backend_publish_dft_chains(fields &owner, int count, dft_chunk *const *heads,
                                const char *site) {
  if (!backend_host_refresh_required(owner)) return;
  std::string local_error;
  for (int i = 0; i < count; ++i)
    backend_write_dft_chain(heads[i], local_error);
  backend_reconcile_host_access(local_error, site);
}

void backend_require_magnetic_synchronization(const fields &f, const char *site) {
  if (!f.backend || !f.backend->requires_full_storage_preparation()) return;
  const bool poisoned = f.backend->is_poisoned();
  const bool unsupported = !f.backend->supports_magnetic_synchronization();
  if (!or_to_all(poisoned || unsupported)) return;
  throw std::runtime_error(
      std::string(site) +
      (poisoned      ? ": resident backend is poisoned by a failed magnetic transition"
       : unsupported ? ": magnetic synchronization is not supported by the resident backend"
                     : ": magnetic synchronization is unavailable on another MPI rank"));
}

static void reconcile_magnetic_dispatch(fields &f, const std::string &local_error,
                                        const char *site) {
  if (!or_to_all(!local_error.empty())) return;
  f.backend->poison();
  if (local_error.empty())
    throw std::runtime_error(std::string(site) +
                             ": magnetic transition failed on another MPI rank; resident backend "
                             "is poisoned");
  throw std::runtime_error(std::string(site) + ": " + local_error +
                           "; resident backend is poisoned");
}

bool backend_try_synchronize_magnetic_fields(fields &f, const char *site) {
  if (!f.backend || !f.backend->requires_full_storage_preparation()) return false;
  backend_require_magnetic_synchronization(f, site);
  std::string local_error;
  try {
    f.init_backend();
    f.ensure_backend_executable();
    f.backend->preflight_magnetic_transition(*f.executable, *f.backend_state, true);
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown resident magnetic synchronize failure";
  }
  backend_reconcile_host_access(local_error, site);
  local_error.clear();
  try {
    f.backend->synchronize_magnetic_fields(*f.executable, *f.backend_state);
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown resident magnetic synchronize failure";
  }
  reconcile_magnetic_dispatch(f, local_error, site);
  return true;
}

bool backend_try_restore_magnetic_fields(fields &f, const char *site) {
  if (!f.backend || !f.backend->requires_full_storage_preparation()) return false;
  backend_require_magnetic_synchronization(f, site);
  std::string local_error;
  try {
    if (!f.backend_state || !f.executable)
      throw std::logic_error("resident magnetic restore has no prepared executable");
    f.backend->preflight_magnetic_transition(*f.executable, *f.backend_state, false);
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown resident magnetic restore failure";
  }
  backend_reconcile_host_access(local_error, site);
  local_error.clear();
  try {
    f.backend->restore_magnetic_fields(*f.executable, *f.backend_state);
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown resident magnetic restore failure";
  }
  reconcile_magnetic_dispatch(f, local_error, site);
  return true;
}

bool backend_try_reduce_dft(fields &owner, const DftReductionRequest &request,
                            std::complex<double> *local_result, size_t result_count,
                            std::string &local_error, const char *site) {
  if (!backend_host_refresh_required(owner) || !owner.backend->supports_compact_dft_reductions())
    return false;

  if (local_error.empty() && owner.backend->is_poisoned())
    local_error = "resident backend is poisoned by a failed transition";
  if (local_error.empty()) try {
      if (!local_result && result_count)
        throw std::invalid_argument("compact DFT reduction has no result buffer");
      if (request.result_count != result_count)
        throw std::invalid_argument("compact DFT reduction result-count mismatch");
      if (result_count)
        std::fill(local_result, local_result + result_count, std::complex<double>(0.0, 0.0));
      owner.backend->reduce_dft(request, local_result, result_count);
    }
    catch (const std::exception &e) {
      local_error = e.what();
    }
    catch (...) {
      local_error = "unknown compact DFT reduction failure";
    }

  backend_reconcile_host_access(local_error, site);
  return true;
}

void backend_prepare_checkpoint_load(fields &f) {
  backend_prepare_field_layout_change(
      f, DirtyMask(dirty_storage | dirty_initialization | dirty_executable), "fields::load");
  /* A load can create or remove lazily allocated arrays. Always rebuild the
     catalog before the next resident execution rather than trying to infer
     whether this particular checkpoint happened to retain its old shape. */
  invalidate(f, MutationKind::field_layout, "fields::load checkpoint replacement");
}

/* Select the backend for `opts`, or abort with a clear message.
 *
 * The abort is reached identically on every rank: every rank evaluates the same
 * options against the same capabilities. A rank that accepted while its peers
 * rejected would hang at the next reduction rather than fail. */
void fields::select_backend(const execution_options &opts) {
  options = opts;
  apply_execution_environment(options);

  std::string why;
  ExecutionBackend *b = make_backend(*this, options, why);
  if (!b) {
    if (options.strict || options.fallback == fallback_policy::error)
      meep::abort("meep: cannot use backend=%s precision=%s: %s",
                  backend_kind_name(options.backend), precision_policy_name(options.precision),
                  why.c_str());
    master_printf("meep: falling back to the cpu backend (%s)\n", why.c_str());
    options.backend = backend_kind::cpu;
    options.precision = precision_policy_kind::native;
    options.device_id = automatic_device;
    b = make_backend(*this, options, why);
    if (!b) meep::abort("meep: no usable backend: %s", why.c_str());
  }

  /* Selecting another backend replaces the storage representation even when no
     simulation mutation happened. Keep the old state alive until its
     replacement has been selected, and do not leak the replacement if the old
     backend refuses or fails migration. */
  try {
    std::string local_error;
    if (backend_state && backend && backend->requires_full_storage_preparation()) {
      if (backend->is_poisoned())
        local_error = "cannot replace a poisoned resident backend";
      else if (synchronized_magnetic_fields)
        local_error = "cannot replace resident backend while magnetic fields are synchronized";
    }
    backend_reconcile_host_access(local_error, "fields::select_backend");
    release_backend_state_for_rebuild(*this,
                                      DirtyMask(dirty_mask | dirty_storage | dirty_executable));
  }
  catch (...) {
    delete b;
    throw;
  }
  delete backend;
  backend = b;
  delete initialization_plan;
  initialization_plan = NULL;
  /* Backend selection changes the storage representation even when the host
     field layout is unchanged (for example CPU native -> NVIDIA mixed).  The
     old backend has been retired successfully at this point, so invalidate
     only now: a failed replacement must leave both the old state and lifecycle
     bookkeeping untouched. */
  invalidate(*this, MutationKind::precision_policy, "fields::select_backend");
}

void fields::init_backend() {
  if (!backend) {
    execution_options opts;
    select_backend(opts);
  }

  /* One collective carries the health, lifecycle, and legacy-flux presence
     facts needed below.  Keeping them in this existing init-entry reduction
     avoids adding steady-state collectives to CPU advance(). */
  const DirtyMask relevant = dirty_source_plan | dirty_monitor_plan | dirty_storage |
                             dirty_regions | dirty_initialization | dirty_classification |
                             dirty_executable | dirty_flux_plan;
  const size_t poison_bit = size_t(1) << (std::numeric_limits<size_t>::digits - 1);
  const size_t flux_present_bit = size_t(1) << (std::numeric_limits<size_t>::digits - 2);
  const size_t multilevel_present_bit =
      size_t(1) << (std::numeric_limits<size_t>::digits - 3);
  size_t local_status = size_t(dirty_mask & relevant);
  if (backend->is_poisoned()) local_status |= poison_bit;
  if (fluxes || (descriptors && !descriptors->legacy_fluxes.empty()))
    local_status |= flux_present_bit;
  if (has_local_exact_multilevel(*this)) local_status |= multilevel_present_bit;
  size_t global_status = 0;
  bw_or_to_all(&local_status, &global_status, 1);
  if (global_status & poison_bit) {
    if (backend->requires_full_storage_preparation()) backend->poison();
    throw std::runtime_error("fields::init_backend: resident backend is poisoned");
  }
  const bool any_flux = (global_status & flux_present_bit) != 0;
  const bool any_multilevel = (global_status & multilevel_present_bit) != 0;

  if (!backend->requires_full_storage_preparation()) {
    /* Most CPU execution keeps its historical lazy storage preparation.
       Legacy flux is the exception: its pointer-free recipes need final
       catalog ArrayIds before the first StepPlan is built, so a cold CPU flux
       step completes storage here. Once storage is current, value-only plan
       mutations refresh without another preparation pass. */
    if (global_status & size_t(dirty_flux_plan))
      dirty_mask |= dirty_flux_plan | dirty_regions | dirty_executable;
    if (any_flux && (global_status & size_t(dirty_storage))) dirty_mask |= dirty_storage;
    const bool refresh_flux_after_cpu_rebuild = any_flux && is_dirty(*this, dirty_storage);
    if (refresh_flux_after_cpu_rebuild) {
      dirty_mask |= dirty_storage;
      prepare_storage();
      dirty_mask |= dirty_flux_plan | dirty_regions | dirty_executable;
    }
    if (!is_dirty(*this, dirty_storage)) refresh_operation_descriptors(*this);
    if (!backend_state) backend_state = backend->create_state(*storage_plan);
    /* CPU storage is the host storage, so value mutations require no transfer. */
    clear_dirty(*this, dirty_initialization);
    return;
  }

  dirty_mask |= DirtyMask(global_status & size_t(relevant));

  /* A phase may have been configured while the CPU backend was still lazy and
     only then moved to a resident backend.  Before that backend freezes its
     catalog, detach shared current structure chunks and realize the retained
     current/target storage union transactionally.  CPU-only execution remains
     lazy, and no rank publishes replacement pointers until every rank has
     completed the fallible preparation. */
  std::unique_ptr<PreparedMaterialPhaseStorage> prepared_material;
  std::string material_error;
  if (phasein_time > 0 && !backend_state) try {
      prepared_material = prepare_material_phase_storage(*this);
    }
    catch (const std::exception &e) {
      material_error = e.what();
    }
    catch (...) {
      material_error = "unknown resident material phase preparation failure";
    }
  backend_reconcile_host_access(material_error, "fields::init_backend material phase");
  if (prepared_material) prepared_material->commit();

  bool coordinates_match = coordinate_state_matches(*this, step_plans[0]);
  if (step_plans[1]) coordinates_match &= coordinate_state_matches(*this, step_plans[1]);
  if (!and_to_all(coordinates_match))
    meep::abort("meep: fields coordinate state changed without invalidation; recreate fields so "
                "the per-chunk coordinate state and resident executable agree");
  /* A value-only material update can change classification without changing
     the existing storage layout. Reconcile the host representation first; any
     promotion it discovers will set dirty_storage for the rebuild below. */
  if (backend_state && is_dirty(*this, dirty_classification) && !is_dirty(*this, dirty_storage))
    classify_and_finalize();

  const bool rebuild_state = !backend_state || is_dirty(*this, dirty_storage);
  const bool refresh_flux_after_rebuild =
      rebuild_state &&
      (fluxes || (descriptors && !descriptors->legacy_fluxes.empty()) ||
       is_dirty(*this, dirty_flux_plan));
  std::unique_ptr<PreparedMaterialCoefficientStorage> prepared_coefficients;
  std::unique_ptr<StepPlan> preclassification_ordinary;
  if (rebuild_state) {
    std::string preflight_error;
    if (any_multilevel) try {
        if (multilevel_preflight_failure_rank_for_testing == my_rank()) {
          if (multilevel_preflight_failure_mode_for_testing == 1)
            preflight_error = "injected multilevel recipe validation failure";
          else if (multilevel_preflight_failure_mode_for_testing == 2)
            throw std::bad_alloc();
        }
        if (preflight_error.empty())
          preflight_error = validate_resident_multilevel_recipes(*this);
      }
      catch (const std::exception &e) {
        preflight_error = e.what();
      }
      catch (...) {
        preflight_error = "unknown resident multilevel recipe preflight failure";
      }
    size_t local_flux_definition = size_t(legacy_flux_definition_signature(*this));
    size_t reference_flux_definition = local_flux_definition;
    broadcast(0, &reference_flux_definition, 1);
    if (local_flux_definition != reference_flux_definition && preflight_error.empty())
      preflight_error = "legacy flux definitions differ across MPI ranks";
    if (any_multilevel) backend_note_multilevel_collective_for_testing();
    backend_reconcile_host_access(preflight_error, "fields::init_backend definition preflight");

    std::string coefficient_error;
    try {
      prepared_coefficients = prepare_material_coefficient_storage(*this);
    }
    catch (const std::exception &e) {
      coefficient_error = e.what();
    }
    catch (...) {
      coefficient_error = "unknown resident material coefficient preparation failure";
    }
    backend_reconcile_host_access(coefficient_error,
                                  "fields::init_backend material coefficients");

    if (backend_state) {
      std::string local_error;
      if (synchronized_magnetic_fields)
        local_error = "cannot rebuild resident state while magnetic fields are synchronized";
      backend_reconcile_host_access(local_error, "fields::init_backend");
      release_backend_state_for_rebuild(*this, DirtyMask(dirty_mask));
    }

    /* Unlike the CPU path, a resident backend needs the complete catalog before
       it allocates. This is intentionally gated by the backend capability so
       default CPU preparation remains lazy and checkpoint-bitwise identical. */
    if (prepared_coefficients) prepared_coefficients->commit();
    prepare_storage();
    /* Boundary construction is also lazy on CPU. Resident compilation needs
       the finalized metal-zero and halo schedules before it snapshots state. */
    connect_chunks();
    if (refresh_flux_after_rebuild) {
      std::unique_ptr<DescriptorSet> staged_descriptors;
      DescriptorSet *const live_descriptors = descriptors;
      std::string plan_error;
      try {
        staged_descriptors.reset(new DescriptorSet(*live_descriptors));
        build_legacy_flux_descriptors(*this, staged_descriptors->legacy_fluxes);
        staged_descriptors->legacy_flux_generation =
            generation(*this, MutationKind::legacy_flux_definition);
        descriptors = staged_descriptors.get();
        preclassification_ordinary.reset(
            new StepPlan(build_step_plan(*this, StepProgram::ordinary)));
      }
      catch (const std::exception &e) {
        plan_error = e.what();
      }
      catch (...) {
        plan_error = "unknown pre-classification ordinary-plan failure";
      }
      descriptors = live_descriptors;
      backend_reconcile_host_access(plan_error,
                                    "fields::init_backend legacy flux stable plan");
    }
    backend_state = backend->create_state(*storage_plan);
    dirty_mask |= dirty_initialization | dirty_classification;
  }

  /* A source-definition change does not necessarily alter storage. Refresh it
     while the old executable is still alive; advance() destroys that artifact
     only after this succeeds and a new StepPlan exists. */
  refresh_operation_descriptors(*this);

  /* A resident catalog replacement invalidates every ArrayId in an existing
     flux recipe. Re-dirty the flux closure without advancing the public
     definition generation; backend_try_refresh_legacy_flux performs its
     collective definition preflight and rebuilds against the complete new
     catalog before compilation. */
  if (refresh_flux_after_rebuild)
    dirty_mask |= dirty_flux_plan | dirty_regions | dirty_executable;

  if (is_dirty(*this, dirty_initialization)) {
    delete initialization_plan;
    initialization_plan = new InitializationPlan(build_initialization_plan(*this));
    backend->initialize(*initialization_plan, *backend_state);
    clear_dirty(*this, dirty_initialization);
  }

  if (is_dirty(*this, dirty_classification)) {
    const MaterialClassification cls = backend->classify_state(*storage_plan, *backend_state);
    if (prepared_classification_hash && cls.hash != prepared_classification_hash)
      meep::abort("meep: backend classification disagrees with the prepared host state");
    backend->finalize_storage(*storage_plan, *backend_state);
    clear_dirty(*this, dirty_classification);
  }
  if (preclassification_ordinary) {
    delete step_plans[0];
    step_plans[0] = preclassification_ordinary.release();
  }
}

bool backend_try_solve_cw(fields &f, const CwSolveRequest &request, CwSolveResult &result) {
  /* A missing backend and the host-authoritative CPU backend retain the exact
     legacy solve_cw path, including its lazy allocation and MPI behavior. */
  if (!f.backend || !f.backend->requires_full_storage_preparation()) return false;

  std::string local_error;
  try {
    std::string why;
    if (!f.backend->supports_cw(request, why))
      local_error = why.empty() ? "resident backend does not support solve_cw" : why;
    if (local_error.empty())
      local_error = cheap_cw_rejection(f, request, f.synchronized_magnetic_fields != 0);
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown resident solve_cw capability failure";
  }
  backend_reconcile_host_access(local_error, "fields::solve_cw cheap preflight");

  /* A pure source-amplitude mutation has stable indices, components, storage,
     and coordinate topology. It can therefore replace the descriptors and
     both executables transactionally against the existing state, retaining
     the state-owned CW workspace. Every other executable-invalidating change
     keeps the conservative full staged-epoch path below. */
  const DirtyMask source_value_forbidden =
      DirtyMask(dirty_initialization | dirty_monitor_plan | dirty_storage | dirty_regions |
                dirty_halos | dirty_classification);
  const bool source_value_refresh =
      f.backend_state && f.executable && is_dirty(f, dirty_source_plan) &&
      is_dirty(f, dirty_executable) && !(f.dirty_mask & source_value_forbidden) &&
      coordinate_state_matches(f, f.step_plans[0]) &&
      (!f.step_plans[1] || coordinate_state_matches(f, f.step_plans[1]));
  const DirtyMask structural_dirty =
      DirtyMask(dirty_source_plan | dirty_monitor_plan | dirty_storage | dirty_regions |
                dirty_halos | dirty_executable | dirty_classification);
  const bool stage_epoch = !source_value_refresh &&
                           (!f.backend_state || !f.executable ||
                            (f.dirty_mask & structural_dirty));
  std::unique_ptr<PreparedBackendEpoch> prepared_epoch;
  if (stage_epoch && f.backend_state) {
    try {
      f.backend->prepare_state_rebuild(*f.backend_state, DirtyMask(f.dirty_mask));
    }
    catch (const std::exception &e) {
      local_error = e.what();
    }
    catch (...) {
      local_error = "unknown solve_cw state-migration failure";
    }
    backend_reconcile_host_access(local_error, "fields::solve_cw state migration");
  }

  if (stage_epoch) {
    PreparedBackendEpoch *local_epoch = NULL;
    try {
      local_epoch = new PreparedBackendEpoch(f);
    }
    catch (const std::exception &e) {
      local_error = e.what();
    }
    catch (...) {
      local_error = "unknown solve_cw epoch preparation failure";
    }
    const bool failed = or_to_all(!local_error.empty());
    if (failed) {
      delete local_epoch;
      if (local_error.empty())
        throw std::runtime_error("fields::solve_cw epoch preparation failed on another MPI rank");
      throw std::runtime_error(std::string("fields::solve_cw epoch preparation: ") + local_error);
    }
    prepared_epoch.reset(local_epoch);
  }

  Executable *ordinary_executable = NULL;
  BackendState *state = NULL;
  const StepPlan *cw_step_plan = NULL;
  CwPlan cw_plan;
  uint64_t storage_fingerprint = 0;
  bool cache_matches = false;
  Executable *old_cw_executable = NULL;
  Executable *replacement = NULL;
  Executable *rogue_cache_executable = NULL;
  uint64_t old_cw_storage_fingerprint = 0;
  uint64_t old_cw_step_plan_signature = 0;
  uint64_t old_cw_plan_signature = 0;
  bool captured_cw_cache = false;
  std::unique_ptr<DescriptorSet> source_value_descriptors;
  std::unique_ptr<StepPlan> source_value_step_plans[2];
  Executable *source_value_ordinary = NULL;
  DescriptorSet *live_descriptors = NULL;
  try {
    if (source_value_refresh) {
      source_value_descriptors.reset(new DescriptorSet(*f.descriptors));
      build_source_descriptors(f, source_value_descriptors->sources);
      live_descriptors = f.descriptors;
      f.descriptors = source_value_descriptors.get();
      source_value_step_plans[0].reset(new StepPlan(build_step_plan(f, StepProgram::ordinary)));
      source_value_step_plans[1].reset(new StepPlan(build_step_plan(f, StepProgram::solve_cw)));
      source_value_ordinary = f.backend->compile(*source_value_step_plans[0], *f.backend_state);
      if (!source_value_ordinary)
        throw std::runtime_error("backend returned no source-refresh executable");
      ordinary_executable = source_value_ordinary;
      state = f.backend_state;
      cw_step_plan = source_value_step_plans[1].get();
    }
    else {
      if (stage_epoch) {
      /* Source promotion and boundary relocation belong to the staged epoch:
         both may rewrite chunk-local source rows and realize new field slots. */
        f.require_source_components();
        f.init_backend();
        f.ensure_backend_executable();
      }
      ordinary_executable = f.executable;
      state = f.backend_state;
      cw_step_plan = &f.step_plan_for(StepProgram::solve_cw);
    }
    if (!ordinary_executable || !state)
      throw std::logic_error("resident solve_cw has no prepared ordinary backend epoch");
    audit_resident_cw_layout(f, cw_step_plan->cw_state_layout);
    cw_plan = build_cw_plan(f, *cw_step_plan);
    if (cw_plan_corruption_for_testing) cw_plan.signature ^= 1;
    std::string validation_error;
    if (!validate_cw_plan(f, *cw_step_plan, cw_plan, &validation_error))
      throw std::logic_error(validation_error.empty() ? "invalid canonical solve_cw plan"
                                                       : validation_error);

    storage_fingerprint = cw_step_plan->cw_state_layout.storage_fingerprint;
    cache_matches = !stage_epoch && state->cw_executable &&
                    state->cw_storage_fingerprint == storage_fingerprint &&
                    state->cw_step_plan_signature == cw_step_plan->signature &&
                    state->cw_plan_signature == cw_plan.signature;
    old_cw_executable = state->cw_executable;
    old_cw_storage_fingerprint = state->cw_storage_fingerprint;
    old_cw_step_plan_signature = state->cw_step_plan_signature;
    old_cw_plan_signature = state->cw_plan_signature;
    captured_cw_cache = true;
    replacement = f.backend->preflight_cw(request, *cw_step_plan, cw_plan,
                                            cache_matches ? old_cw_executable : NULL, *state);
    if (!replacement) throw std::runtime_error("backend returned no solve_cw executable");
    if (replacement == ordinary_executable)
      throw std::runtime_error("backend aliased the ordinary and solve_cw executables");
    if (!cache_matches && replacement == old_cw_executable)
      throw std::runtime_error("backend reused a solve_cw executable with a stale cache key");
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown resident solve_cw compiled preflight failure";
  }
  if (live_descriptors) f.descriptors = live_descriptors;

  if (captured_cw_cache && state &&
      (state->cw_executable != old_cw_executable ||
                state->cw_storage_fingerprint != old_cw_storage_fingerprint ||
                state->cw_step_plan_signature != old_cw_step_plan_signature ||
       state->cw_plan_signature != old_cw_plan_signature)) {
    rogue_cache_executable = state->cw_executable;
    state->cw_executable = old_cw_executable;
    state->cw_storage_fingerprint = old_cw_storage_fingerprint;
    state->cw_step_plan_signature = old_cw_step_plan_signature;
    state->cw_plan_signature = old_cw_plan_signature;
    if (local_error.empty())
      local_error = "backend mutated the solve_cw cache during preflight";
  }

  const bool preflight_failed = or_to_all(!local_error.empty());
  if (preflight_failed) {
    if (rogue_cache_executable && rogue_cache_executable != old_cw_executable &&
        rogue_cache_executable != replacement && rogue_cache_executable != ordinary_executable)
      delete rogue_cache_executable;
    if (replacement && replacement != old_cw_executable && replacement != ordinary_executable)
      delete replacement;
    delete source_value_ordinary;
    prepared_epoch.reset();
    if (local_error.empty())
      throw std::runtime_error("fields::solve_cw compiled preflight failed on another MPI rank");
    throw std::runtime_error(std::string("fields::solve_cw compiled preflight: ") + local_error);
  }

  if (source_value_refresh) {
    DescriptorSet *old_descriptors = f.descriptors;
    StepPlan *old_step_plans[2] = {f.step_plans[0], f.step_plans[1]};
    Executable *old_ordinary = f.executable;
    f.descriptors = source_value_descriptors.release();
    f.step_plans[0] = source_value_step_plans[0].release();
    f.step_plans[1] = source_value_step_plans[1].release();
    f.executable = source_value_ordinary;
    source_value_ordinary = NULL;
    clear_dirty(f, dirty_source_plan | dirty_executable);
    delete old_ordinary;
    delete old_step_plans[0];
    delete old_step_plans[1];
    delete old_descriptors;
  }
  if (replacement != old_cw_executable) {
    state->cw_executable = replacement;
    delete old_cw_executable;
  }
  state->cw_storage_fingerprint = storage_fingerprint;
  state->cw_step_plan_signature = cw_step_plan->signature;
  state->cw_plan_signature = cw_plan.signature;
  if (prepared_epoch) prepared_epoch->commit();

  local_error.clear();
  try {
    /* Field/material value refreshes are deliberately beyond the retryable
       preflight boundary: their plans and executables remain reusable, but a
       partially failed device upload cannot be rolled back. Source amplitudes
       were refreshed transactionally with their plans/executables above. */
    if (!stage_epoch && is_dirty(f, dirty_initialization)) f.init_backend();
    {
      CwSolveSession session(f, request);
      result = f.backend->solve_cw(request, *cw_step_plan, cw_plan, *ordinary_executable,
                                   *state->cw_executable, *state, session);
      if (!session.boundary_called() || !session.at_entry_state())
        throw std::runtime_error("backend did not restore solve_cw state before final DFT");
      switch (result.status) {
        case CwSolveStatus::converged:
        case CwSolveStatus::not_converged:
        case CwSolveStatus::breakdown: break;
        default: throw std::runtime_error("backend returned an invalid solve_cw status");
      }
      if (result.iterations < 0 || result.iterations > request.maxiters ||
          result.operator_applications == 0 ||
          !std::isfinite(result.recursive_relative_residual) ||
          result.recursive_relative_residual < 0.0 ||
          !std::isfinite(result.true_relative_residual) || result.true_relative_residual < 0.0)
        throw std::runtime_error("backend returned an invalid solve_cw result");
    }
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown resident solve_cw dispatch failure";
  }

  if (or_to_all(!local_error.empty())) {
    f.backend->poison();
    if (local_error.empty())
      throw std::runtime_error("fields::solve_cw dispatch failed on another MPI rank; backend poisoned");
    throw std::runtime_error(std::string("fields::solve_cw dispatch failed; backend poisoned: ") +
                             local_error);
  }
  return true;
}

/* --- Backend-safe access -------------------------------------------------- */

uint32_t fields::register_point_sampler(component c, const vec &loc, int interval) {
  std::vector<std::vector<point_sampler> > &reg = sampler_registry();
  const uint32_t id = uint32_t(reg.size());
  reg.push_back(std::vector<point_sampler>());
  point_sampler s;
  s.c = c;
  s.loc = loc;
  s.interval = interval < 1 ? 1 : interval;
  reg.back().push_back(s);
  return id;
}

uint32_t fields::register_reduction(reduction_kind kind, const volume &where, int interval) {
  /* The registration API is what PR 7 owes; device-side accumulation is Phase 2
     (§14). On CPU a reduction is still evaluated on demand from the host, so
     this records the request and read_samples() computes it. */
  (void)kind;
  (void)where;
  (void)interval;
  std::vector<std::vector<point_sampler> > &reg = sampler_registry();
  const uint32_t id = uint32_t(reg.size());
  reg.push_back(std::vector<point_sampler>());
  return id;
}

void fields::read_samples(uint32_t id, std::vector<std::complex<double> > &out) {
  out.clear();
  std::vector<std::vector<point_sampler> > &reg = sampler_registry();
  if (id >= reg.size() || reg[id].empty()) return;
  out = reg[id][0].samples;
}

void fields::collect_samples() {
  std::vector<std::vector<point_sampler> > &reg = sampler_registry();
  for (std::vector<point_sampler> &v : reg)
    for (point_sampler &s : v)
      if (s.interval > 0 && (t % s.interval) == 0) s.samples.push_back(get_field(s.c, s.loc));
}

} // namespace meep
