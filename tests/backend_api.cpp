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

/* PR 7 acceptance tests: the backend interface, precision policy, and the
   initialization plan. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unistd.h>
#include <vector>

#include <meep.hpp>

#include "config.h"
#include "backend/backend.hpp"
#include "backend/checkpoint.hpp"
#include "backend/cpu/cpu_backend.hpp"
#include "backend/descriptors.hpp"
#include "backend/halo_plan.hpp"
#include "backend/initialization_plan.hpp"
#include "backend/lifecycle.hpp"
#include "backend/material_ir.hpp"
#include "backend/material_callback.hpp"
#include "backend/prepare.hpp"
#include "backend/precision.hpp"
#include "backend/random_state.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"
#include "meepgeom.hpp"

using namespace meep;

static int failures = 0;

#define CHECK(cond, ...)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      printf("[rank %d] FAIL (%s:%d): ", my_rank(), __FILE__, __LINE__);                           \
      printf(__VA_ARGS__);                                                                         \
      printf("\n");                                                                                \
      fflush(stdout);                                                                              \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

static double eps_slab(const vec &p) { return (fabs(p.y()) < 0.4) ? 12.0 : 1.0; }
static double unit_epsilon(const vec &) { return 1.0; }
static double two_epsilon(const vec &) { return 2.0; }
static realnum checkpoint_identity_epsilon_value = realnum(1.0);
static double checkpoint_identity_epsilon(const vec &) {
  return double(checkpoint_identity_epsilon_value);
}
static double phase_conductivity(const vec &);
static std::complex<double> initial_ez(const vec &) { return std::complex<double>(0.25, -0.5); }
static int material_ir_user_calls = 0;
static void material_ir_user_function(vector3, void *, meep_geom::medium_struct *medium) {
  ++material_ir_user_calls;
  medium->epsilon_diag = meep_geom::make_vector3(2.0, 2.0, 2.0);
}

class lifecycle_custom_susceptibility : public lorentzian_susceptibility {
public:
  lifecycle_custom_susceptibility(realnum omega, realnum gamma, bool publish_layout_ = false)
      : lorentzian_susceptibility(omega, gamma), publish_layout(publish_layout_) {}
  lifecycle_custom_susceptibility(const lifecycle_custom_susceptibility &other)
      : lorentzian_susceptibility(other), publish_layout(other.publish_layout) {}
  susceptibility *clone() const override { return new lifecycle_custom_susceptibility(*this); }
  void *new_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2],
                          const grid_volume &gv) const override {
    ++allocations;
    return lorentzian_susceptibility::new_internal_data(W, gv);
  }
  void init_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2], realnum dt,
                          const grid_volume &gv, void *data) const override {
    ++initializations;
    lorentzian_susceptibility::init_internal_data(W, dt, gv, data);
  }
  bool internal_layout(std::vector<InternalArrayLayout> &out, const grid_volume &gv,
                       void *data) const override {
    ++layout_queries;
    if (!publish_layout) {
      out.clear();
      return false;
    }
    return lorentzian_susceptibility::internal_layout(out, gv, data);
  }

  static void reset_counts() { allocations = initializations = layout_queries = 0; }
  static int allocations;
  static int initializations;
  static int layout_queries;

private:
  bool publish_layout;
};

int lifecycle_custom_susceptibility::allocations = 0;
int lifecycle_custom_susceptibility::initializations = 0;
int lifecycle_custom_susceptibility::layout_queries = 0;

template <typename T> static void check_susceptibility_clone(T &source, const char *name) {
  source.ntot = 3;
  source.sigma[Ex][X] = new realnum[source.ntot];
  source.sigma[Ex][X][0] = realnum(0.25);
  source.sigma[Ex][X][1] = realnum(-0.5);
  source.sigma[Ex][X][2] = realnum(0.75);
  source.trivial_sigma[Ex][X] = false;
  source.trivial_sigma[Ey][Y] = false;
  source.next = new susceptibility;
  std::unique_ptr<susceptibility> clone(source.clone());
  CHECK(clone && typeid(*clone) == typeid(source), "%s clone changed dynamic type", name);
  CHECK(clone && clone->get_id() == source.get_id() && clone->ntot == source.ntot &&
            clone->next == NULL,
        "%s clone changed identity, extent, or list ownership", name);
  FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
    CHECK(clone->trivial_sigma[c][d] == source.trivial_sigma[c][d],
          "%s clone changed trivial-sigma flags", name);
    CHECK((clone->sigma[c][d] == NULL) == (source.sigma[c][d] == NULL),
          "%s clone changed sigma nullability", name);
    if (source.sigma[c][d]) {
      CHECK(clone->sigma[c][d] != source.sigma[c][d], "%s clone aliased sigma storage", name);
      for (size_t i = 0; i < source.ntot; ++i)
        CHECK(clone->sigma[c][d][i] == source.sigma[c][d][i],
              "%s clone changed sigma values", name);
    }
  }
  clone->sigma[Ex][X][1] = realnum(9.5);
  CHECK(source.sigma[Ex][X][1] == realnum(-0.5), "%s clone mutation aliased source", name);
}

static void test_susceptibility_clone_coefficients() {
  susceptibility base;
  lorentzian_susceptibility lorentz(realnum(0.51), realnum(0.07));
  noisy_lorentzian_susceptibility noisy(realnum(0.02), realnum(0.53), realnum(0.08));
  gyrotropic_susceptibility gyro(vec(0.1, -0.2, 0.3), realnum(0.61), realnum(0.09));
  const realnum Gamma[4] = {realnum(0.01), realnum(0.02), realnum(0.03), realnum(0.04)};
  const realnum N0[2] = {realnum(0.7), realnum(0.3)};
  const realnum alpha[2] = {realnum(-0.5), realnum(0.5)};
  const realnum omega[1] = {realnum(0.73)};
  const realnum gamma[1] = {realnum(0.06)};
  const realnum sigmat[5] = {realnum(1), realnum(0), realnum(0), realnum(1), realnum(0)};
  multilevel_susceptibility multilevel(2, 1, Gamma, N0, alpha, omega, gamma, sigmat);
  check_susceptibility_clone(base, "base susceptibility");
  check_susceptibility_clone(lorentz, "Lorentz susceptibility");
  check_susceptibility_clone(noisy, "noisy susceptibility");
  check_susceptibility_clone(gyro, "gyrotropic susceptibility");
  check_susceptibility_clone(multilevel, "multilevel susceptibility");

  const grid_volume gv = vol1d(1.0, 4.0);
  structure owner(gv, unit_epsilon, no_pml(), identity(), 1);
  lorentzian_susceptibility resident(realnum(0.47), realnum(0.05));
  gyrotropic_susceptibility resident_gyro(vec(0, 0, 0.11), realnum(0.63),
                                          realnum(0.04));
  owner.add_susceptibility(unit_epsilon, E_stuff, resident);
  owner.add_susceptibility(unit_epsilon, E_stuff, resident_gyro);
  structure_chunk *source = owner.chunks[0];
  std::unique_ptr<structure_chunk> copy(new structure_chunk(source));
  const susceptibility *left = source->chiP[E_stuff];
  const susceptibility *right = copy->chiP[E_stuff];
  size_t nodes = 0;
  for (; left && right; left = left->next, right = right->next, ++nodes) {
    CHECK(typeid(*left) == typeid(*right) && left->get_id() == right->get_id(),
          "structure COW changed susceptibility type, identity, or list order");
    FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
      CHECK(left->trivial_sigma[c][d] == right->trivial_sigma[c][d],
            "structure COW changed trivial-sigma flags");
      CHECK((left->sigma[c][d] == NULL) == (right->sigma[c][d] == NULL),
            "structure COW changed sigma nullability");
      if (left->sigma[c][d]) {
        CHECK(left->sigma[c][d] != right->sigma[c][d],
              "structure COW aliased susceptibility sigma storage");
        for (size_t i = 0; i < left->ntot; ++i)
          CHECK(left->sigma[c][d][i] == right->sigma[c][d][i],
                "structure COW changed susceptibility sigma values");
      }
    }
  }
  CHECK(!left && !right && nodes == 2, "structure COW changed susceptibility list length");
}

class lifecycle_stateless_custom_susceptibility : public susceptibility {
public:
  susceptibility *clone() const override {
    return new lifecycle_stateless_custom_susceptibility(*this);
  }
};

struct lifetime_counts {
  int states_created;
  int states_destroyed;
  int executables_created;
  int executables_destroyed;
  int initialized;
  int classified;
  int finalized;
  int advance_attempts;
  int advanced;
  int reads;
  int writes;
  int rebuilds;
  int magnetic_synchronizes;
  int magnetic_restores;
  int cw_executables_created;
  int cw_executables_destroyed;
  int cw_preflights;
  int cw_solves;
  int cw_callback_effects;
  int malformed_cw_result;
  int noisy_seed_refresh_attempts;
  int noisy_seed_refreshes;
  int custom_preflights;
  RandomSeedSnapshot last_noisy_seed;
  size_t arrays_at_create;
  size_t material_arrays_at_create;
  size_t provisional_material_arrays_at_create;
  size_t material_recipe_rows_at_initialize;
  uint64_t material_recipe_signature_at_initialize;
  size_t arrays_at_compile;
  size_t retained_logical_suffix_at_compile;
  size_t provisional_material_arrays_at_compile;
  size_t polarization_arrays_at_create;
  size_t polarization_updates_at_compile;
  size_t polarization_subtractions_at_compile;
  size_t multilevel_population_updates_at_compile;
  size_t multilevel_transition_updates_at_compile;
  size_t beta_updates_at_compile;
  size_t bfast_updates_at_compile;
  size_t cylindrical_m_updates_at_compile;
  size_t cylindrical_origin_actions_at_compile;
  size_t legacy_flux_updates_at_compile;
  size_t legacy_flux_terms_at_compile;
  size_t legacy_flux_half_accesses_at_compile;
  size_t legacy_flux_final_accesses_at_compile;
  bool gyrotropic_update_at_compile;
  bool polarization_zero_at_create;
  bool connections_current_at_create;
  bool rebuild_saw_live_imaginary;
  bool migrate_authoritative_value;
  realnum authoritative_value;
  bool migrate_multilevel_values;
  realnum authoritative_population;
  realnum authoritative_p;
  realnum authoritative_p_prev;
  int multilevel_migrations;
  bool fail_rebuild;
  bool fail_create_state;
  int fail_create_state_on_call;
  bool fail_initialize;
  int fail_initialize_on_call;
  bool fail_classify;
  int fail_classify_on_call;
  int force_promotion_on_classify_call;
  bool force_aniso2d;
  component_mask force_required_components;
  int malformed_material_classification;
  bool fail_finalize;
  int fail_finalize_on_call;
  bool fail_compile;
  int fail_compile_on_call;
  bool fail_advance;
  bool retain_all_provisional_material_rows;
  bool corrupt_catalog_after_compile;
  bool fail_magnetic_synchronize;
  bool fail_magnetic_restore;
  bool fail_magnetic_synchronize_dispatch;
  bool fail_cw_preflight;
  bool fail_cw_dispatch;
  bool alias_cw_to_ordinary;
  bool mutate_cw_cache_during_preflight;
  bool mutate_after_cw_boundary;
  bool fail_noisy_seed_refresh;
  bool poison_noisy_seed_refresh;
  bool fail_custom_preflight;
  bool poison_custom_preflight;
  bool fail_custom_before_entry;
  bool fail_custom_later_before_entry;
  bool fail_custom_after_entry;
  bool reenter_custom_callback;
  bool omit_custom_last_segment;
  bool undercount_custom_last_segment;
  bool redistribute_custom_callbacks;
  bool reorder_custom_first_step;
  bool reorder_custom_second_step;
  bool cw_saw_transient_mode;
  bool cw_final_dft_at_entry_time;
  CwSolveStatus cw_status;

  lifetime_counts()
      : states_created(0), states_destroyed(0), executables_created(0), executables_destroyed(0),
        initialized(0), classified(0), finalized(0), advance_attempts(0), advanced(0), reads(0),
        writes(0), rebuilds(0),
        magnetic_synchronizes(0), magnetic_restores(0),
        cw_executables_created(0), cw_executables_destroyed(0), cw_preflights(0), cw_solves(0),
        cw_callback_effects(0), malformed_cw_result(0), noisy_seed_refresh_attempts(0),
        noisy_seed_refreshes(0), custom_preflights(0), last_noisy_seed(), arrays_at_create(0),
        material_arrays_at_create(0), provisional_material_arrays_at_create(0),
        material_recipe_rows_at_initialize(0), material_recipe_signature_at_initialize(0),
        arrays_at_compile(0), retained_logical_suffix_at_compile(0),
        provisional_material_arrays_at_compile(0),
        polarization_arrays_at_create(0), polarization_updates_at_compile(0),
        polarization_subtractions_at_compile(0), beta_updates_at_compile(0),
        multilevel_population_updates_at_compile(0),
        multilevel_transition_updates_at_compile(0),
        bfast_updates_at_compile(0), cylindrical_m_updates_at_compile(0),
        cylindrical_origin_actions_at_compile(0), legacy_flux_updates_at_compile(0),
        legacy_flux_terms_at_compile(0), legacy_flux_half_accesses_at_compile(0),
        legacy_flux_final_accesses_at_compile(0), gyrotropic_update_at_compile(false),
        polarization_zero_at_create(true), connections_current_at_create(false),
        rebuild_saw_live_imaginary(false), migrate_authoritative_value(false),
        authoritative_value(0), migrate_multilevel_values(false),
        authoritative_population(realnum(0.625)), authoritative_p(realnum(0.375)),
        authoritative_p_prev(realnum(-0.25)), multilevel_migrations(0), fail_rebuild(false),
        fail_create_state(false), fail_create_state_on_call(0), fail_initialize(false),
        fail_initialize_on_call(0), fail_classify(false), fail_classify_on_call(0),
        force_promotion_on_classify_call(0), force_aniso2d(false),
        force_required_components(0),
        malformed_material_classification(0), fail_finalize(false), fail_finalize_on_call(0),
        fail_compile(false), fail_compile_on_call(0), fail_advance(false),
        retain_all_provisional_material_rows(false),
        corrupt_catalog_after_compile(false),
        fail_magnetic_synchronize(false), fail_magnetic_restore(false),
        fail_magnetic_synchronize_dispatch(false), fail_cw_preflight(false),
        fail_cw_dispatch(false), alias_cw_to_ordinary(false),
        mutate_cw_cache_during_preflight(false), mutate_after_cw_boundary(false),
        fail_noisy_seed_refresh(false), poison_noisy_seed_refresh(false),
        fail_custom_preflight(false),
        poison_custom_preflight(false), fail_custom_before_entry(false),
        fail_custom_later_before_entry(false), fail_custom_after_entry(false),
        reenter_custom_callback(false), omit_custom_last_segment(false),
        undercount_custom_last_segment(false), redistribute_custom_callbacks(false),
        reorder_custom_first_step(false), reorder_custom_second_step(false),
        cw_saw_transient_mode(false),
        cw_final_dft_at_entry_time(false),
        cw_status(CwSolveStatus::converged) {}
};

struct tracking_state : BackendState {
  tracking_state(lifetime_counts &counts_, const StoragePlan &plan_)
      : counts(counts_), plan(plan_), staged_noisy_seed(), noisy_seed_staged(false) {
    ++counts.states_created;
  }
  ~tracking_state() override { ++counts.states_destroyed; }
  lifetime_counts &counts;
  StoragePlan plan;
  RandomSeedSnapshot staged_noisy_seed;
  bool noisy_seed_staged;
};

struct tracking_custom_segment {
  uint32_t operation_index;
  HostSegment identity;
  size_t callback_count;
};

struct tracking_executable : Executable {
  tracking_executable(lifetime_counts &counts_,
                      const std::vector<tracking_custom_segment> &custom_segments_)
      : counts(counts_), custom_segments(custom_segments_) {
    ++counts.executables_created;
  }
  ~tracking_executable() override { ++counts.executables_destroyed; }
  lifetime_counts &counts;
  std::vector<tracking_custom_segment> custom_segments;
};

struct tracking_cw_executable : Executable {
  explicit tracking_cw_executable(lifetime_counts &counts_) : counts(counts_) {
    ++counts.cw_executables_created;
  }
  ~tracking_cw_executable() override { ++counts.cw_executables_destroyed; }
  lifetime_counts &counts;
};

class tracking_backend : public ExecutionBackend {
public:
  tracking_backend(fields &f_, lifetime_counts &counts_, bool magnetic_supported_ = false,
                   bool cw_supported_ = false, bool custom_supported_ = false,
                   bool execute_custom_ = false, bool material_policy_enabled_ = false)
      : f(f_), counts(counts_), magnetic_supported(magnetic_supported_),
        cw_supported(cw_supported_), custom_supported(custom_supported_),
        execute_custom(execute_custom_), material_policy_enabled(material_policy_enabled_),
        custom_staging_prepared(false) {}

  BackendState *create_state(const StoragePlan &plan) override {
    if (counts.fail_create_state ||
        (counts.fail_create_state_on_call &&
         counts.states_created + 1 == counts.fail_create_state_on_call))
      throw std::runtime_error("injected state creation failure");
    counts.arrays_at_create = plan.arrays.size();
    counts.material_arrays_at_create = 0;
    counts.provisional_material_arrays_at_create = 0;
    counts.connections_current_at_create = connections_are_current(f);
    for (size_t i = 0; i < plan.arrays.size(); ++i) {
      const ArraySpec &spec = plan.arrays[i];
      if (spec.role == array_role::material) {
        ++counts.material_arrays_at_create;
        if (spec.classification_provisional)
          ++counts.provisional_material_arrays_at_create;
      }
      if (spec.role != array_role::polarization || is_valid(spec.alias_of)) continue;
      ++counts.polarization_arrays_at_create;
      const realnum *values = f.array_catalog->resolve<realnum>(spec.id);
      for (size_t j = 0; j < spec.elements; ++j)
        if (values[j] != realnum(0)) counts.polarization_zero_at_create = false;
    }
    return new tracking_state(counts, plan);
  }
  void initialize(const InitializationPlan &plan, BackendState &) override {
    ++counts.initialized;
    CHECK(plan.materials.size() == 1,
          "tracking backend initialization did not receive one material recipe");
    counts.material_recipe_rows_at_initialize =
        plan.materials.empty() ? 0 : plan.materials[0].rows().size();
    if (!plan.materials.empty()) {
      validate_material_recipe(plan.materials[0]);
      counts.material_recipe_signature_at_initialize = plan.materials[0].signature();
    }
    if (counts.fail_initialize ||
        (counts.fail_initialize_on_call && counts.initialized == counts.fail_initialize_on_call))
      throw std::runtime_error("injected initialization failure");
  }
  MaterialClassification classify_state(const StoragePlan &plan, BackendState &raw_state) override {
    ++counts.classified;
    const std::string error =
        counts.fail_classify ||
                (counts.fail_classify_on_call && counts.classified == counts.fail_classify_on_call)
            ? "injected material classify failure"
            : "";
    backend_reconcile_host_access(error, "tracking material classify");
    tracking_state &state = dynamic_cast<tracking_state &>(raw_state);
    CHECK(plan.arrays.size() <= state.plan.arrays.size(),
          "tracking classification plan exceeds resident storage");
    MaterialClassification result = classify(f, state.plan);
    if (counts.retain_all_provisional_material_rows) {
      result.elided.clear();
      for (size_t i = 0; i < state.plan.arrays.size(); ++i)
        if (state.plan.arrays[i].role == array_role::material &&
            state.plan.arrays[i].classification_provisional)
          result.provisional_row_state[i] = MaterialClassification::retained;
      refresh_material_classification_variants(f, state.plan, result);
    }
    if (counts.malformed_material_classification == 1 &&
        !result.provisional_row_state.empty())
      result.provisional_row_state.pop_back();
    if (counts.malformed_material_classification == 2)
      result.required_components = ~component_mask(0);
    if (counts.malformed_material_classification == 3) {
      const ArrayId duplicate = result.elided.empty() ? ArrayId{0} : result.elided.back();
      result.elided.push_back(duplicate);
      result.elided.push_back(duplicate);
    }
    if (counts.malformed_material_classification == 4 && !result.anisotropic_eh.empty())
      result.anisotropic_eh.pop_back();
    if (counts.malformed_material_classification == 5 && !result.variant_facts.empty())
      result.variant_facts.push_back(result.variant_facts.back());
    if (counts.force_promotion_on_classify_call &&
        counts.classified == counts.force_promotion_on_classify_call)
      result.aniso2d = true;
    if (counts.force_aniso2d) result.aniso2d = true;
    result.required_components |= counts.force_required_components;
    return result;
  }
  void finalize_storage(const StoragePlan &plan, const MaterialClassification &classification,
                        BackendState &raw_state) override {
    if (counts.fail_finalize ||
        (counts.fail_finalize_on_call && counts.finalized + 1 == counts.fail_finalize_on_call))
      throw std::runtime_error("injected material finalize failure");
    tracking_state &state = dynamic_cast<tracking_state &>(raw_state);
    CHECK(f.initialization_plan && f.initialization_plan->materials.size() == 1,
          "tracking backend did not receive one frozen material recipe");
    if (f.initialization_plan && f.initialization_plan->materials.size() == 1)
      resolve_material_storage(f.initialization_plan->materials[0], classification, plan,
                               state.plan);
    CHECK(!has_provisional_material_storage(state.plan),
          "tracking backend retained provisional material storage after classification");
    ++counts.finalized;
  }
  Executable *compile(const StepPlan &plan, BackendState &raw_state) override {
    if (counts.fail_compile ||
        (counts.fail_compile_on_call &&
         counts.executables_created + 1 == counts.fail_compile_on_call))
      throw std::runtime_error("injected executable compilation failure");
    tracking_state &state = dynamic_cast<tracking_state &>(raw_state);
    counts.arrays_at_compile = f.storage_plan ? f.storage_plan->arrays.size() : 0;
    counts.retained_logical_suffix_at_compile = 0;
    counts.provisional_material_arrays_at_compile = 0;
    for (size_t i = 0; i < state.plan.arrays.size(); ++i) {
      const ArraySpec &spec = state.plan.arrays[i];
      if (spec.role == array_role::material && spec.classification_provisional)
        ++counts.provisional_material_arrays_at_compile;
      if (spec.role == array_role::material && !spec.classification_provisional &&
          !spec.classification_elided && f.array_catalog && i < f.array_catalog->size() &&
          !f.array_catalog->resolve_untyped(ArrayId{uint32_t(i)})) {
        ++counts.retained_logical_suffix_at_compile;
        CHECK(f.array_catalog->find(f.storage_plan->keys[i]) == ArrayId{uint32_t(i)},
              "retained logical suffix key was invisible during compilation");
      }
    }
    counts.polarization_updates_at_compile = plan.polarization_updates.size();
    counts.polarization_subtractions_at_compile = plan.polarization_subtractions.size();
    counts.multilevel_population_updates_at_compile =
        plan.multilevel_population_updates.size();
    counts.multilevel_transition_updates_at_compile =
        plan.multilevel_transition_updates.size();
    counts.beta_updates_at_compile = plan.beta_updates.size();
    counts.bfast_updates_at_compile = plan.bfast_updates.size();
    counts.cylindrical_m_updates_at_compile = plan.cylindrical_m_updates.size();
    counts.cylindrical_origin_actions_at_compile = plan.cylindrical_origin_actions.size();
    counts.legacy_flux_updates_at_compile = plan.legacy_flux_updates.size();
    counts.legacy_flux_terms_at_compile = plan.legacy_flux_terms.size();
    counts.legacy_flux_half_accesses_at_compile = 0;
    counts.legacy_flux_final_accesses_at_compile = 0;
    for (const Operation &op : plan.operations) {
      if (op.kind == OpKind::update_flux_half)
        counts.legacy_flux_half_accesses_at_compile = op.accesses.size();
      if (op.kind == OpKind::update_flux)
        counts.legacy_flux_final_accesses_at_compile = op.accesses.size();
    }
    for (const PolarizationUpdate &update : plan.polarization_updates)
      if (update.kind == PolarizationUpdateKind::gyrotropic)
        counts.gyrotropic_update_at_compile = true;
    std::vector<tracking_custom_segment> custom_segments;
    for (size_t operation_index = 0; operation_index < plan.operations.size();
         ++operation_index) {
      const Operation &op = plan.operations[operation_index];
      if (op.kind != OpKind::host_callback || op.descriptor_count != 1 ||
          op.descriptor_index >= plan.host_segments.size())
        continue;
      const HostSegment &segment = plan.host_segments[op.descriptor_index];
      size_t callback_count = 0;
      for (uint32_t i = 0; i < segment.callback_count; ++i) {
        const HostCallbackDescriptor &callback =
            plan.host_callbacks[size_t(segment.callback_index) + i];
        if (segment.phase != HostSegmentPhase::constitutive || callback.has_internal_state)
          ++callback_count;
      }
      custom_segments.push_back(tracking_custom_segment{
          uint32_t(operation_index), segment, callback_count});
    }
    Executable *result = new tracking_executable(counts, custom_segments);
    if (counts.corrupt_catalog_after_compile) f.array_catalog->clear();
    return result;
  }
  void advance(Executable &executable, BackendState &, int num_steps) override {
    ++counts.advance_attempts;
    if (counts.fail_advance) throw std::runtime_error("injected backend advance failure");
    if (execute_custom && host_custom_fallback_enabled()) {
      tracking_executable &compiled = static_cast<tracking_executable &>(executable);
      for (int step = 0; step < num_steps; ++step)
        for (size_t segment = 0; segment < compiled.custom_segments.size(); ++segment) {
          if (counts.omit_custom_last_segment &&
              segment + 1 == compiled.custom_segments.size())
            continue;
          const size_t callback_segment =
              ((counts.reorder_custom_first_step && step == 0) ||
               (counts.reorder_custom_second_step && step == 1))
                  ? compiled.custom_segments.size() - segment - 1
                  : segment;
          const tracking_custom_segment &actual = compiled.custom_segments[callback_segment];
          HostCustomFallbackSession session(*this, actual.operation_index, actual.identity);
          if ((counts.fail_custom_before_entry && segment == 0) ||
              (counts.fail_custom_later_before_entry && segment == 1))
            throw std::runtime_error("injected pre-callback host custom failure");
          session.record_download(32);
          size_t callback_count = actual.callback_count;
          if (counts.undercount_custom_last_segment &&
              segment + 1 == compiled.custom_segments.size() && callback_count)
            --callback_count;
          if (counts.redistribute_custom_callbacks && compiled.custom_segments.size() >= 2) {
            if (segment == 0 && callback_count)
              --callback_count;
            else if (segment == 1)
              ++callback_count;
          }
          session.enter_callback(callback_count);
          if (counts.reenter_custom_callback && segment == 0) {
            HostCustomFallbackSession nested(*this, actual.operation_index, actual.identity);
            (void)nested;
          }
          if (counts.fail_custom_after_entry && segment == 0)
            throw std::runtime_error("injected host custom callback failure");
          session.record_upload(48);
          session.complete();
        }
    }
    ++counts.advanced;
  }
  void refresh_noisy_seed(const RandomSeedSnapshot &candidate, BackendState &state) override {
    ++counts.noisy_seed_refresh_attempts;
    if (counts.fail_noisy_seed_refresh)
      throw std::runtime_error("injected noisy seed refresh failure");
    if (counts.poison_noisy_seed_refresh) poison();
    tracking_state &tracking = dynamic_cast<tracking_state &>(state);
    tracking.staged_noisy_seed = candidate;
    tracking.noisy_seed_staged = true;
  }
  void commit_noisy_seed(BackendState &state) noexcept override {
    tracking_state &tracking = static_cast<tracking_state &>(state);
    if (!tracking.noisy_seed_staged) return;
    counts.last_noisy_seed = tracking.staged_noisy_seed;
    ++counts.noisy_seed_refreshes;
    tracking.noisy_seed_staged = false;
  }
  void discard_noisy_seed(BackendState &state) noexcept override {
    static_cast<tracking_state &>(state).noisy_seed_staged = false;
  }
  bool supports_host_custom_fallback() const override { return custom_supported; }
  void preflight_host_custom_fallback(Executable &, BackendState &) override {
    ++counts.custom_preflights;
    if (counts.poison_custom_preflight) poison();
    if (counts.fail_custom_preflight)
      throw std::runtime_error("injected host custom fallback preflight failure");
    if (!custom_staging_prepared) {
      note_host_custom_staging_allocation(64);
      custom_staging_prepared = true;
    }
  }
  bool supports_cw(const CwSolveRequest &, std::string &why) const override {
    if (cw_supported) return true;
    why = "tracking backend CW support is disabled";
    return false;
  }
  Executable *preflight_cw(const CwSolveRequest &, const StepPlan &,
                           const StepPlan &step_plan, const CwPlan &cw_plan,
                           Executable &, Executable *cached, BackendState &state) override {
    ++counts.cw_preflights;
    if (counts.fail_cw_preflight)
      throw std::runtime_error("injected solve_cw preflight failure");
    CHECK(step_plan.program == StepProgram::solve_cw && cw_plan.step_plan_signature == step_plan.signature,
          "CW preflight received mismatched canonical plans");
    if (counts.alias_cw_to_ordinary) return f.executable;
    if (counts.mutate_cw_cache_during_preflight) {
      state.cw_executable = new tracking_cw_executable(counts);
      state.cw_storage_fingerprint = 1;
      state.cw_step_plan_signature = 2;
      state.cw_plan_signature = 3;
      return state.cw_executable;
    }
    return cached ? cached : new tracking_cw_executable(counts);
  }
  CwSolveResult solve_cw(const CwSolveRequest &request, const StepPlan &step_plan,
                         const CwPlan &cw_plan,
                         Executable &ordinary, Executable &cw, BackendState &,
                         CwSolveSession &session) override {
    ++counts.cw_solves;
    ++counts.cw_callback_effects;
    counts.cw_saw_transient_mode = true;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk)
      counts.cw_saw_transient_mode =
          counts.cw_saw_transient_mode && f.chunks[chunk]->is_solving_cw();
    CHECK(&ordinary != &cw, "CW dispatch aliased the ordinary executable");
    CHECK(step_plan.program == StepProgram::solve_cw && cw_plan.step_plan_signature == step_plan.signature,
          "CW dispatch received mismatched canonical plans");
    f.t += 7;
    if (counts.fail_cw_dispatch)
      throw std::runtime_error("injected solve_cw dispatch failure");
    session.restore_before_final_dft();
    counts.cw_final_dft_at_entry_time = f.t == request.entry_t && f.time() == request.entry_time;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk)
      counts.cw_final_dft_at_entry_time =
          counts.cw_final_dft_at_entry_time && !f.chunks[chunk]->is_solving_cw();
    if (counts.mutate_after_cw_boundary) {
      f.t += 9;
      f.set_solve_cw_omega(2.0 * pi * request.frequency);
    }
    CwSolveResult result;
    result.status = counts.cw_status;
    result.iterations = 3;
    result.operator_applications = 11;
    result.recursive_relative_residual = 1e-7;
    result.true_relative_residual = 2e-7;
    if (counts.malformed_cw_result == 1) result.status = CwSolveStatus(99);
    if (counts.malformed_cw_result == 2)
      result.recursive_relative_residual = std::numeric_limits<double>::quiet_NaN();
    if (counts.malformed_cw_result == 3) result.true_relative_residual = -1.0;
    if (counts.malformed_cw_result == 4) result.iterations = 21;
    if (counts.malformed_cw_result == 5) result.operator_applications = 0;
    if (counts.malformed_cw_result == 6) {
      result.iterations = 0;
      result.operator_applications = 0;
    }
    return result;
  }
  void read(ArrayRef, void *, size_t) override { ++counts.reads; }
  void write(ArrayRef, const void *, size_t) override { ++counts.writes; }
  void synchronize() override {}
  bool supports_magnetic_synchronization() const override { return magnetic_supported; }
  void preflight_magnetic_transition(Executable &, BackendState &, bool synchronize) override {
    if (synchronize && counts.fail_magnetic_synchronize)
      throw std::runtime_error("injected magnetic synchronize failure");
    if (!synchronize && counts.fail_magnetic_restore)
      throw std::runtime_error("injected magnetic restore failure");
  }
  void synchronize_magnetic_fields(Executable &, BackendState &) override {
    if (counts.fail_magnetic_synchronize_dispatch)
      throw std::runtime_error("injected magnetic synchronize dispatch failure");
    ++counts.magnetic_synchronizes;
  }
  void restore_magnetic_fields(Executable &, BackendState &) override {
    ++counts.magnetic_restores;
  }
  backend_capabilities capabilities() const override {
    backend_capabilities c = {true, true, true, 0, "tracking"};
    return c;
  }
  bool requires_full_storage_preparation() const override { return true; }
  bool enforces_material_fallback_policy() const override {
    return material_policy_enabled;
  }
  void prepare_state_rebuild(BackendState &, DirtyMask) override {
    ++counts.rebuilds;
    bool migrated = false;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
      if (!f.chunks[chunk] || !f.chunks[chunk]->is_mine()) continue;
      FOR_COMPONENTS(c) {
        if (counts.migrate_authoritative_value && !migrated && f.chunks[chunk]->f[c][0]) {
          f.chunks[chunk]->f[c][0][0] = counts.authoritative_value;
          migrated = true;
        }
        if (!f.chunks[chunk]->f[c][1]) continue;
        const ArrayId id = f.array_catalog->find({chunk, int(array_kind::f), int(c), 1, 0});
        if (is_valid(id) && f.array_catalog->resolve<realnum>(id) == f.chunks[chunk]->f[c][1])
          counts.rebuild_saw_live_imaginary = true;
      }
    }
    if (counts.migrate_multilevel_values && f.descriptors && f.array_catalog) {
      for (const PolarizationDescriptor &descriptor : f.descriptors->polarizations) {
        if (descriptor.kind != SusceptibilityKind::multilevel ||
            !is_valid(descriptor.multilevel_populations))
          continue;
        realnum *populations =
            f.array_catalog->resolve<realnum>(descriptor.multilevel_populations);
        const size_t population_elements =
            f.array_catalog->spec(descriptor.multilevel_populations).elements;
        for (size_t i = 0; i < population_elements; ++i)
          populations[i] = counts.authoritative_population + realnum(i) * realnum(0.000125);
        for (const MultilevelStateArrays &row : descriptor.multilevel_states) {
          realnum *p = f.array_catalog->resolve<realnum>(row.p);
          realnum *p_prev = f.array_catalog->resolve<realnum>(row.p_prev);
          for (size_t i = 0; i < row.elements; ++i) {
            p[i] = counts.authoritative_p + realnum(i) * realnum(0.00025);
            p_prev[i] = counts.authoritative_p_prev - realnum(i) * realnum(0.000375);
          }
        }
        ++counts.multilevel_migrations;
      }
    }
    if (counts.fail_rebuild) throw std::runtime_error("injected layout migration failure");
  }
  bool accepts(const execution_options &, std::string &) const override { return true; }

private:
  fields &f;
  lifetime_counts &counts;
  bool magnetic_supported;
  bool cw_supported;
  bool custom_supported;
  bool execute_custom;
  bool material_policy_enabled;
  bool custom_staging_prepared;
};

static void build(structure **sp, fields **fp, const execution_options *opts = NULL);
static bool checkpoint_payload_equal(const CheckpointImage &expected,
                                     const CheckpointImage &actual);

static CwSolveRequest cw_request() {
  CwSolveRequest request;
  request.tolerance = 1e-6;
  request.maxiters = 20;
  request.frequency = std::complex<double>(0.3, 0.0);
  request.L = 2;
  return request;
}

static std::vector<const void *> cw_chunk_storage_addresses(const fields &f) {
  std::vector<const void *> result;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    const fields_chunk &fc = *f.chunks[chunk];
    result.push_back(&fc);
    result.push_back(fc.s);
    result.push_back(fc.dft_chunks);
    DOCMP2 FOR_COMPONENTS(c) {
      result.push_back(fc.f[c][cmp]);
      result.push_back(fc.f_u[c][cmp]);
      result.push_back(fc.f_w[c][cmp]);
      result.push_back(fc.f_cond[c][cmp]);
      result.push_back(fc.f_bfast[c][cmp]);
      result.push_back(fc.f_minus_p[c][cmp]);
      result.push_back(fc.f_w_prev[c][cmp]);
      result.push_back(fc.f_backup[c][cmp]);
      result.push_back(fc.f_u_backup[c][cmp]);
      result.push_back(fc.f_w_backup[c][cmp]);
      result.push_back(fc.f_cond_backup[c][cmp]);
      result.push_back(fc.f_bfast_backup[c][cmp]);
    }
    result.push_back(fc.f_rderiv_int);
  }
  return result;
}

static realnum *first_owned_real_field(fields &f) {
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    FOR_COMPONENTS(c) if (f.chunks[chunk]->f[c][0]) return f.chunks[chunk]->f[c][0];
  }
  return NULL;
}

struct cw_source_snapshot {
  std::vector<int> metadata;
  std::vector<ptrdiff_t> indices;
  std::vector<std::complex<double> > amplitudes;
  bool operator==(const cw_source_snapshot &other) const {
    return metadata == other.metadata && indices == other.indices && amplitudes == other.amplitudes;
  }
};

static cw_source_snapshot snapshot_cw_sources(const fields &f) {
  cw_source_snapshot result;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    FOR_FIELD_TYPES(ft) for (const src_vol &source : f.chunks[chunk]->sources[ft]) {
      result.metadata.push_back(chunk);
      result.metadata.push_back(int(ft));
      result.metadata.push_back(int(source.c));
      result.metadata.push_back(source.needs_boundary_fix ? 1 : 0);
      result.metadata.push_back(int(source.num_points()));
      for (size_t point = 0; point < source.num_points(); ++point) {
        result.indices.push_back(source.index_at(point));
        result.amplitudes.push_back(source.amplitude_at(point));
      }
    }
  return result;
}

namespace meep {
struct BackendEpochSnapshot {
  fields_chunk **chunks;
  halo_plan_set *halos;
  CpuArrayCatalog *catalog;
  StoragePlan *storage;
  DescriptorSet *descriptors;
  InitializationPlan *initialization;
  StepPlan *steps[2];
  BackendState *state;
  Executable *ordinary;
  Executable *cw;
  std::vector<const void *> raw;
  std::vector<const void *> dft_fc;
  std::vector<const void *> susceptibility_nodes;
  std::vector<const void *> polarization_nodes;
  std::vector<const void *> comm;
  std::vector<uint64_t> catalog_values;
  std::vector<std::vector<unsigned char> > catalog_bytes;
  std::vector<int> source_metadata;
  std::vector<ptrdiff_t> source_indices;
  std::vector<std::complex<double> > source_amplitudes;
  std::vector<bool> cw_flags;
  std::vector<std::complex<double> > cw_frequencies;
  uint64_t catalog_hash, storage_hash, descriptor_hash, halo_hash, comm_hash, comm_block_hash,
      comm_sequence_hash, comm_map_hash, raw_hash;
  uint32_t dirty, prepared_mask, classification_reentries;
  uint64_t classification_hash, connections_generation, connections_built_generation;
  uint64_t local_generation, local_synced, mutations[fields::num_mutation_kinds];
  bool components_allocated, connections_valid, changed_materials;
  bool host_custom_enabled;
  HostCustomFallbackStats host_custom_stats;
  MaterialRecipeDisposition material_global_route, material_local_route;
  uint64_t material_support_reasons, material_recipe_signature,
      material_state_classification_hash;
  bool material_fallback_local_presence, material_fallback_global_presence,
      material_fallback_presence_validated, material_fallback_policy_pending,
      material_phase_active;
  MaterialFallbackStatistics material_fallback_stats;
  uint64_t material_warning_count;
  bool material_warning_emitted;
  int time_step;
  uint64_t cw_storage_key, cw_step_key, cw_plan_key;

  static void mix(uint64_t &h, uint64_t v) {
    h ^= v;
    h *= 1099511628211ULL;
  }
  static void mix_bytes(uint64_t &h, const void *p, size_t n) {
    const unsigned char *bytes = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; ++i) mix(h, bytes[i]);
  }
  static size_t element_bytes(ElementType type) {
    switch (type) {
      case ElementType::realnum_value: return sizeof(realnum);
      case ElementType::complex_realnum: return sizeof(std::complex<realnum>);
      case ElementType::float64: return sizeof(double);
      case ElementType::complex_float64: return sizeof(std::complex<double>);
      case ElementType::int32: return sizeof(int32_t);
      case ElementType::index: return sizeof(ptrdiff_t);
    }
    return 0;
  }
  template <typename T> static void mix_vector(uint64_t &h, const std::vector<T> &values) {
    mix(h, values.size());
    if (!values.empty()) mix_bytes(h, values.data(), values.size() * sizeof(T));
  }

  explicit BackendEpochSnapshot(const fields &f)
      : chunks(f.chunks), halos(f.halos), catalog(f.array_catalog), storage(f.storage_plan),
        descriptors(f.descriptors), initialization(f.initialization_plan), state(f.backend_state),
        ordinary(f.executable), cw(f.backend_state ? f.backend_state->cw_executable : NULL),
        raw(cw_chunk_storage_addresses(f)), catalog_hash(1469598103934665603ULL),
        storage_hash(1469598103934665603ULL), descriptor_hash(1469598103934665603ULL),
        halo_hash(1469598103934665603ULL), comm_hash(1469598103934665603ULL),
        comm_block_hash(1469598103934665603ULL),
        comm_sequence_hash(1469598103934665603ULL), comm_map_hash(0),
        raw_hash(1469598103934665603ULL), dirty(f.dirty_mask),
        prepared_mask(f.storage_prepared_mask), classification_reentries(f.classification_reentries),
        classification_hash(f.prepared_classification_hash),
        connections_generation(f.connections_generation),
        connections_built_generation(f.connections_built_generation),
        local_generation(f.local_invalidation_generation), local_synced(f.local_invalidation_synced),
        components_allocated(f.components_allocated), connections_valid(f.chunk_connections_valid),
        changed_materials(f.changed_materials),
        host_custom_enabled(f.backend && f.backend->host_custom_fallback_enabled()),
        host_custom_stats(f.backend ? f.backend->host_custom_fallback_stats()
                                    : HostCustomFallbackStats()),
        material_global_route(f.backend_state ? f.backend_state->material_route
                                              : MaterialRecipeDisposition::device_native),
        material_local_route(f.backend_state ? f.backend_state->material_local_route
                                             : MaterialRecipeDisposition::device_native),
        material_support_reasons(f.backend_state ? f.backend_state->material_support_reasons : 0),
        material_recipe_signature(f.backend_state ? f.backend_state->material_recipe_signature : 0),
        material_state_classification_hash(
            f.backend_state ? f.backend_state->material_classification_hash : 0),
        material_fallback_local_presence(
            f.backend_state && f.backend_state->material_fallback_local_presence),
        material_fallback_global_presence(
            f.backend_state && f.backend_state->material_fallback_global_presence),
        material_fallback_presence_validated(
            f.backend_state && f.backend_state->material_fallback_presence_validated),
        material_fallback_policy_pending(
            f.backend_state && f.backend_state->material_fallback_policy_pending),
        material_phase_active(f.backend_state && f.backend_state->material_phase_active),
        material_fallback_stats(f.backend_state
                                    ? f.backend_state->material_fallback_statistics
                                    : MaterialFallbackStatistics()),
        material_warning_count(f.backend ? f.backend->material_fallback_warning_count() : 0),
        material_warning_emitted(
            f.backend && f.backend->material_fallback_warning_emitted()),
        time_step(f.t), cw_storage_key(0), cw_step_key(0),
        cw_plan_key(0) {
    steps[0] = f.step_plans[0];
    steps[1] = f.step_plans[1];
    for (int i = 0; i < fields::num_mutation_kinds; ++i) mutations[i] = f.mutation_generation[i];
    if (f.backend_state) {
      cw_storage_key = f.backend_state->cw_storage_fingerprint;
      cw_step_key = f.backend_state->cw_step_plan_signature;
      cw_plan_key = f.backend_state->cw_plan_signature;
    }
    for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
      cw_flags.push_back(f.chunks[chunk]->doing_solve_cw);
      cw_frequencies.push_back(f.chunks[chunk]->solve_cw_omega);
      FOR_FIELD_TYPES(ft) for (const src_vol &source : f.chunks[chunk]->sources[ft]) {
        source_metadata.push_back(chunk);
        source_metadata.push_back(int(ft));
        source_metadata.push_back(int(source.c));
        source_metadata.push_back(source.needs_boundary_fix ? 1 : 0);
        source_metadata.push_back(int(source.num_points()));
        for (size_t point = 0; point < source.num_points(); ++point) {
          source_indices.push_back(source.index_at(point));
          source_amplitudes.push_back(source.amplitude_at(point));
        }
      }
      for (dft_chunk *dft = f.chunks[chunk]->dft_chunks; dft; dft = dft->next_in_chunk) {
        dft_fc.push_back(dft);
        dft_fc.push_back(dft->fc);
      }
      if (f.chunks[chunk]->s)
        FOR_FIELD_TYPES(ft)
          for (const susceptibility *sus = f.chunks[chunk]->s->chiP[ft]; sus;
               sus = sus->next)
            susceptibility_nodes.push_back(sus);
      FOR_FIELD_TYPES(ft)
        for (const polarization_state *pol = f.chunks[chunk]->pol[ft]; pol; pol = pol->next) {
          polarization_nodes.push_back(pol);
          polarization_nodes.push_back(pol->s);
          polarization_nodes.push_back(pol->data);
        }
    }
    if (catalog) for (size_t i = 0; i < catalog->size(); ++i) {
      const ArrayId id = {uint32_t(i)};
      const StorageKey &key = catalog->key(id);
      const ArraySpec &spec = catalog->spec(id);
      const void *base = catalog->resolve_untyped(id);
      mix(catalog_hash, uint64_t(reinterpret_cast<uintptr_t>(base)));
      mix_bytes(catalog_hash, &key, sizeof(key));
      mix(catalog_hash, spec.id.value);
      mix(catalog_hash, uint64_t(spec.role));
      mix(catalog_hash, uint64_t(spec.element_type));
      mix(catalog_hash, uint64_t(spec.storage));
      mix(catalog_hash, spec.elements);
      mix(catalog_hash, spec.alignment);
      mix(catalog_hash, spec.alias_of.value);
      mix(catalog_hash, spec.classification_provisional);
      mix(catalog_hash, spec.classification_elided);
      uint64_t value_hash = 1469598103934665603ULL;
      std::vector<unsigned char> value_bytes;
      if (base && !is_valid(spec.alias_of)) {
        const size_t bytes = spec.elements * element_bytes(spec.element_type);
        mix_bytes(raw_hash, base, bytes);
        mix_bytes(value_hash, base, bytes);
        const unsigned char *begin = static_cast<const unsigned char *>(base);
        value_bytes.assign(begin, begin + bytes);
      }
      catalog_values.push_back(value_hash);
      catalog_bytes.push_back(value_bytes);
    }
    if (storage) for (size_t i = 0; i < storage->arrays.size(); ++i) {
      const ArraySpec &spec = storage->arrays[i];
      mix_bytes(storage_hash, &storage->keys[i], sizeof(StorageKey));
      mix(storage_hash, spec.id.value);
      mix(storage_hash, uint64_t(spec.role));
      mix(storage_hash, uint64_t(spec.element_type));
      mix(storage_hash, uint64_t(spec.storage));
      mix(storage_hash, spec.elements);
      mix(storage_hash, spec.alignment);
      mix(storage_hash, spec.alias_of.value);
      mix(storage_hash, spec.classification_provisional);
      mix(storage_hash, spec.classification_elided);
    }
    if (descriptors) {
      mix(descriptor_hash, source_plan_signature(descriptors->sources));
      mix(descriptor_hash, dft_plan_signature(descriptors->dfts));
      mix(descriptor_hash, descriptors->legacy_flux_generation);
      mix(descriptor_hash, descriptors->legacy_fluxes.size());
      for (const LegacyFluxDescriptor &flux : descriptors->legacy_fluxes) {
        mix(descriptor_hash, flux.flux_ordinal);
        mix(descriptor_hash, flux.recipe_signature);
        mix(descriptor_hash, flux.terms.size());
      }
      mix(descriptor_hash, descriptors->polarizations.size());
      mix_vector(descriptor_hash, descriptors->regions);
    }
    if (steps[0]) mix(descriptor_hash, steps[0]->signature);
    if (steps[1]) mix(descriptor_hash, steps[1]->signature);
    if (initialization) {
      mix(descriptor_hash, initialization->material_values_generation);
      mix(descriptor_hash, initialization->material_region_generation);
      mix_vector(descriptor_hash, initialization->operations);
      for (const MaterialRecipe &recipe : initialization->materials) {
        mix(descriptor_hash, recipe.signature());
      }
      mix(descriptor_hash, initialization->pml.size());
      for (const PmlRecipe &recipe : initialization->pml) {
        mix(descriptor_hash, uint64_t(recipe.chunk));
        mix(descriptor_hash, uint64_t(recipe.direction_));
        mix_vector(descriptor_hash, recipe.sigma);
        mix_vector(descriptor_hash, recipe.kappa);
        mix_vector(descriptor_hash, recipe.sigma_inv);
      }
      for (const HostCallbackRecipe &recipe : initialization->host_callbacks) {
        mix(descriptor_hash, recipe.id);
        mix_bytes(descriptor_hash, recipe.description.data(), recipe.description.size());
      }
    }
    if (halos) {
      mix(halo_hash, halos->arrays.size());
      mix(halo_hash, halos->index.size());
      for (size_t i = 0; i < halos->arrays.size(); ++i) {
        const ArrayId id = {uint32_t(i)};
        mix(halo_hash, uint64_t(reinterpret_cast<uintptr_t>(halos->arrays.base(id))));
        const ArraySpec &spec = halos->arrays.spec(id);
        mix(halo_hash, spec.id.value);
        mix(halo_hash, uint64_t(spec.role));
        mix(halo_hash, uint64_t(spec.element_type));
        mix(halo_hash, spec.elements);
      }
      for (const HaloPlan &plan : halos->plans) {
        mix(halo_hash, uint64_t(plan.ft));
        mix_bytes(halo_hash, &plan.chunks, sizeof(plan.chunks));
        mix(halo_hash, uint64_t(plan.phase));
        mix(halo_hash, plan.peer_rank);
        mix(halo_hash, plan.tag);
        mix(halo_hash, plan.same_rank);
        mix(halo_hash, plan.sequence_index);
        mix(halo_hash, plan.block_offset);
        mix(halo_hash, plan.block_elements);
        mix_vector(halo_hash, plan.gather_slabs);
        mix_vector(halo_hash, plan.scatter_slabs);
        mix_vector(halo_hash, plan.gather);
        mix_vector(halo_hash, plan.scatter);
        mix_vector(halo_hash, plan.gather_order);
        mix_vector(halo_hash, plan.scatter_order);
        mix_vector(halo_hash, plan.phase_values);
      }
      FOR_FIELD_TYPES(ft) for (const ZeroPlan &zero : halos->zeros[ft]) {
        mix_vector(halo_hash, zero.slabs);
        mix_vector(halo_hash, zero.residue);
      }
    }
    FOR_FIELD_TYPES(ft) {
      comm.push_back(f.comm_blocks[ft]);
      if (f.comm_blocks[ft])
        for (int i = 0; i < f.num_chunks * f.num_chunks; ++i) {
          comm.push_back(f.comm_blocks[ft][i]);
          const chunk_pair pair(i % f.num_chunks, i / f.num_chunks);
          const size_t count = f.comm_size_tot(field_type(ft), pair);
          if (f.comm_blocks[ft][i] && count)
            mix_bytes(comm_block_hash, f.comm_blocks[ft][i], count * sizeof(realnum));
        }
      for (const comms_operation &op : f.comms_sequence_for_field[ft].receive_ops)
        mix_bytes(comm_sequence_hash, &op, sizeof(op));
      for (const comms_operation &op : f.comms_sequence_for_field[ft].send_ops)
        mix_bytes(comm_sequence_hash, &op, sizeof(op));
    }
    uint64_t map_hash = 0;
    for (const auto &entry : f.comm_sizes) {
      uint64_t item = 1469598103934665603ULL;
      mix_bytes(item, &entry.first, sizeof(entry.first));
      mix(item, entry.second);
      map_hash ^= item + 0x9e3779b97f4a7c15ULL;
    }
    comm_map_hash = map_hash;
    mix(comm_hash, comm_block_hash);
    mix(comm_hash, comm_sequence_hash);
    mix(comm_hash, comm_map_hash);
  }

  bool matches(const BackendEpochSnapshot &other) const {
    if (chunks != other.chunks || halos != other.halos || catalog != other.catalog ||
        storage != other.storage || descriptors != other.descriptors ||
        initialization != other.initialization || steps[0] != other.steps[0] ||
        steps[1] != other.steps[1] || state != other.state || ordinary != other.ordinary ||
        cw != other.cw || raw != other.raw || dft_fc != other.dft_fc ||
        susceptibility_nodes != other.susceptibility_nodes ||
        polarization_nodes != other.polarization_nodes || comm != other.comm ||
        catalog_values != other.catalog_values ||
        catalog_bytes != other.catalog_bytes ||
        source_metadata != other.source_metadata || source_indices != other.source_indices ||
        source_amplitudes != other.source_amplitudes || cw_flags != other.cw_flags ||
        cw_frequencies != other.cw_frequencies ||
        catalog_hash != other.catalog_hash || storage_hash != other.storage_hash ||
        descriptor_hash != other.descriptor_hash || halo_hash != other.halo_hash ||
        comm_hash != other.comm_hash || raw_hash != other.raw_hash || dirty != other.dirty ||
        prepared_mask != other.prepared_mask || classification_hash != other.classification_hash ||
        classification_reentries != other.classification_reentries ||
        connections_generation != other.connections_generation ||
        connections_built_generation != other.connections_built_generation ||
        local_generation != other.local_generation || local_synced != other.local_synced ||
        components_allocated != other.components_allocated ||
        connections_valid != other.connections_valid || changed_materials != other.changed_materials ||
        host_custom_enabled != other.host_custom_enabled ||
        host_custom_stats.warnings != other.host_custom_stats.warnings ||
        host_custom_stats.preflights != other.host_custom_stats.preflights ||
        host_custom_stats.sessions != other.host_custom_stats.sessions ||
        host_custom_stats.callbacks != other.host_custom_stats.callbacks ||
        host_custom_stats.completed_sessions != other.host_custom_stats.completed_sessions ||
        host_custom_stats.staging_allocations != other.host_custom_stats.staging_allocations ||
        host_custom_stats.staging_bytes != other.host_custom_stats.staging_bytes ||
        host_custom_stats.downloads != other.host_custom_stats.downloads ||
        host_custom_stats.download_bytes != other.host_custom_stats.download_bytes ||
        host_custom_stats.uploads != other.host_custom_stats.uploads ||
        host_custom_stats.upload_bytes != other.host_custom_stats.upload_bytes ||
        host_custom_stats.retryable_failures != other.host_custom_stats.retryable_failures ||
        host_custom_stats.poisoned_failures != other.host_custom_stats.poisoned_failures ||
        material_global_route != other.material_global_route ||
        material_local_route != other.material_local_route ||
        material_support_reasons != other.material_support_reasons ||
        material_recipe_signature != other.material_recipe_signature ||
        material_state_classification_hash != other.material_state_classification_hash ||
        material_fallback_local_presence != other.material_fallback_local_presence ||
        material_fallback_global_presence != other.material_fallback_global_presence ||
        material_fallback_presence_validated != other.material_fallback_presence_validated ||
        material_fallback_policy_pending != other.material_fallback_policy_pending ||
        material_phase_active != other.material_phase_active ||
        material_fallback_stats.warnings != other.material_fallback_stats.warnings ||
        material_fallback_stats.dense_rows != other.material_fallback_stats.dense_rows ||
        material_fallback_stats.dense_bytes != other.material_fallback_stats.dense_bytes ||
        material_fallback_stats.interface_points != other.material_fallback_stats.interface_points ||
        material_fallback_stats.callback_tiles != other.material_fallback_stats.callback_tiles ||
        material_fallback_stats.callback_points != other.material_fallback_stats.callback_points ||
        material_fallback_stats.callback_calls != other.material_fallback_stats.callback_calls ||
        material_fallback_stats.classification_launches !=
            other.material_fallback_stats.classification_launches ||
        material_fallback_stats.classification_device_to_host_calls !=
            other.material_fallback_stats.classification_device_to_host_calls ||
        material_fallback_stats.classification_device_to_host_bytes !=
            other.material_fallback_stats.classification_device_to_host_bytes ||
        material_warning_count != other.material_warning_count ||
        material_warning_emitted != other.material_warning_emitted ||
        time_step != other.time_step ||
        cw_storage_key != other.cw_storage_key || cw_step_key != other.cw_step_key ||
        cw_plan_key != other.cw_plan_key)
      return false;
    for (int i = 0; i < fields::num_mutation_kinds; ++i)
      if (mutations[i] != other.mutations[i]) return false;
    return true;
  }
  bool matches(const fields &f) const { return matches(BackendEpochSnapshot(f)); }
};
} // namespace meep

static void expect_cheap_cw_rejection(fields &f, lifetime_counts &counts,
                                      const CwSolveRequest &request, const char *label) {
  const int saved_t = f.t;
  const uint32_t saved_dirty = f.dirty_mask;
  CwSolveResult result;
  bool rejected = false;
  try {
    (void)backend_try_solve_cw(f, request, result);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(sum_to_all(int(rejected)) == count_processors(), "%s was not rejected collectively", label);
  CHECK(!f.backend_state && !f.executable, "%s initialized resident state", label);
  CHECK(counts.states_created == 0 && counts.executables_created == 0 &&
            counts.cw_preflights == 0 && counts.cw_solves == 0 && counts.cw_callback_effects == 0,
        "%s crossed the cheap preflight boundary", label);
  CHECK(f.t == saved_t && f.dirty_mask == saved_dirty, "%s changed host time or lifecycle state",
        label);
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    CHECK(!f.chunks[chunk]->is_solving_cw(), "%s left transient CW mode active", label);
}

static void test_resident_cw_lifecycle() {
  structure *s;
  fields *f;
  build(&s, &f);
  component cw_monitor_component = Ez;
  dft_fields cw_monitor =
      f->add_dft_fields(&cw_monitor_component, 1, f->v, 0.3, 0.3, 1, 2);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts, false, true);
  bool marked_boundary_source = false;
  for (int chunk = 0; chunk < f->num_chunks && !marked_boundary_source; ++chunk)
    FOR_FIELD_TYPES(ft) if (!f->chunks[chunk]->sources[ft].empty()) {
      f->chunks[chunk]->sources[ft][0].needs_boundary_fix = true;
      marked_boundary_source = true;
      break;
    }
  CHECK(or_to_all(marked_boundary_source),
        "CW fixture has no source row for staged boundary fix coverage");

  if (count_processors() != 1) {
    expect_cheap_cw_rejection(*f, counts, cw_request(), "MPI resident solve_cw");
    delete f;
    delete s;
    return;
  }

  const uint32_t cold_dirty = f->dirty_mask;
  counts.fail_cw_preflight = true;
  bool failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && !f->backend_state && !f->executable && f->dirty_mask == cold_dirty,
        "cold CW preflight failure published resident artifacts or dirty-state changes");
  CHECK(counts.cw_solves == 0 && counts.cw_callback_effects == 0,
        "cold CW preflight failure entered dispatch/callback territory");
  bool boundary_fix_restored = false;
  for (int chunk = 0; chunk < f->num_chunks; ++chunk)
    FOR_FIELD_TYPES(ft) for (const src_vol &source : f->chunks[chunk]->sources[ft])
      boundary_fix_restored = boundary_fix_restored || source.needs_boundary_fix;
  CHECK(boundary_fix_restored,
        "cold CW preflight rollback changed the live source boundary-fix state");
  counts.fail_cw_preflight = false;

  counts.cw_status = CwSolveStatus::not_converged;
  const int entry_t = f->t;
  CHECK(!f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
        "ordinary nonconvergence was reported as success");
  CHECK(!f->backend->is_poisoned(), "ordinary nonconvergence poisoned the backend");
  CHECK(f->t == entry_t && counts.cw_saw_transient_mode,
        "resident solve did not restore time or enter transient CW mode");
  CHECK(counts.cw_final_dft_at_entry_time,
        "resident backend did not restore solve-entry time before its final DFT boundary");
  for (int chunk = 0; chunk < f->num_chunks; ++chunk)
    FOR_FIELD_TYPES(ft) for (const src_vol &source : f->chunks[chunk]->sources[ft])
      CHECK(!source.needs_boundary_fix,
            "staged source-boundary rewrite was not transferred to the stable chunk");
  for (int chunk = 0; chunk < f->num_chunks; ++chunk)
    CHECK(!f->chunks[chunk]->is_solving_cw(), "resident solve did not restore chunk CW flags");
  CHECK(counts.cw_preflights == 2 && counts.cw_solves == 1 &&
            counts.cw_executables_created == 1 && counts.cw_executables_destroyed == 0,
        "first resident solve did not create exactly one private CW executable");
  BackendState *const state = f->backend_state;
  Executable *ordinary = f->executable;
  Executable *cached_cw = state->cw_executable;
  CHECK(cached_cw && cached_cw != ordinary, "ordinary and CW executables are not separately owned");

  counts.cw_status = CwSolveStatus::converged;
  CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
        "converged resident solve was reported as failure");
  CHECK(f->backend_state == state && f->executable == ordinary &&
            f->backend_state->cw_executable == cached_cw,
        "unchanged resident solve did not reuse state and both executables");
  CHECK(counts.cw_preflights == 3 && counts.cw_solves == 2 &&
            counts.cw_executables_created == 1,
        "unchanged resident solve rebuilt its private executable");

  Executable *const ordinary_before_values = ordinary;
  Executable *const cw_before_values = cached_cw;
  src_vol *mutated_source = NULL;
  for (int chunk = 0; chunk < f->num_chunks && !mutated_source; ++chunk)
    FOR_FIELD_TYPES(ft) if (!f->chunks[chunk]->sources[ft].empty()) {
      mutated_source = &f->chunks[chunk]->sources[ft][0];
      break;
    }
  CHECK(mutated_source && mutated_source->num_points() > 0,
        "source-value refresh fixture has no mutable amplitude");
  const std::complex<double> refreshed_amplitude =
      mutated_source->amplitude_at(0) + std::complex<double>(0.125, -0.0625);
  mutated_source->set_amplitude(0, refreshed_amplitude);
  invalidate(*f, MutationKind::source_values, "CW source-value refresh test");
  const BackendEpochSnapshot source_value_entry(*f);
  DescriptorSet *const source_value_descriptors = f->descriptors;
  StepPlan *const source_value_ordinary_plan = f->step_plans[0];
  StepPlan *const source_value_cw_plan = f->step_plans[1];
  counts.fail_compile = true;
  bool source_value_compile_failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    source_value_compile_failed = true;
  }
  counts.fail_compile = false;
  CHECK(source_value_compile_failed && source_value_entry.matches(*f) &&
            mutated_source->amplitude_at(0) == refreshed_amplitude &&
            f->backend_state == state && f->executable == ordinary_before_values &&
            f->backend_state->cw_executable == cw_before_values &&
            f->descriptors == source_value_descriptors &&
            f->step_plans[0] == source_value_ordinary_plan &&
            f->step_plans[1] == source_value_cw_plan,
        "source-value ordinary compile failure partially published the refresh");

  counts.fail_cw_preflight = true;
  bool source_value_cw_failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    source_value_cw_failed = true;
  }
  counts.fail_cw_preflight = false;
  CHECK(source_value_cw_failed && source_value_entry.matches(*f) &&
            mutated_source->amplitude_at(0) == refreshed_amplitude &&
            f->backend_state == state && f->executable == ordinary_before_values &&
            f->backend_state->cw_executable == cw_before_values &&
            f->descriptors == source_value_descriptors &&
            f->step_plans[0] == source_value_ordinary_plan &&
            f->step_plans[1] == source_value_cw_plan,
        "source-value CW preflight failure partially published the refresh");

  const int ordinary_created_before_success = counts.executables_created;
  const int ordinary_destroyed_before_success = counts.executables_destroyed;
  const int cw_created_before_success = counts.cw_executables_created;
  const int cw_destroyed_before_success = counts.cw_executables_destroyed;
  CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
        "source-value-only resident solve failed");
  ordinary = f->executable;
  cached_cw = f->backend_state->cw_executable;
  CHECK(f->backend_state == state && ordinary != ordinary_before_values &&
            cached_cw != cw_before_values,
        "source-value-only refresh did not retain state and replace both executables");
  CHECK(counts.executables_created == ordinary_created_before_success + 1 &&
            counts.executables_destroyed == ordinary_destroyed_before_success + 1 &&
            counts.cw_executables_created == cw_created_before_success + 1 &&
            counts.cw_executables_destroyed == cw_destroyed_before_success + 1,
        "source-value-only refresh did not replace each executable exactly once");
  bool descriptor_refreshed = false;
  for (const SourceDescriptor &source : f->descriptors->sources.sources)
    for (std::complex<double> amplitude : source.complex_amplitudes)
      descriptor_refreshed = descriptor_refreshed || amplitude == refreshed_amplitude;
  CHECK(descriptor_refreshed,
        "source-value-only refresh did not publish the live amplitude in SourcePlan");

  counts.cw_status = CwSolveStatus::breakdown;
  CHECK(!f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2) &&
            !f->backend->is_poisoned(),
        "ordinary solver breakdown did not remain a non-poisoning result");
  counts.cw_status = CwSolveStatus::converged;
  CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
        "retry after ordinary solver breakdown did not succeed");

  counts.fail_cw_preflight = true;
  const int solves_before_preflight = counts.cw_solves;
  const int callbacks_before_preflight = counts.cw_callback_effects;
  failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && !f->backend->is_poisoned(),
        "compiled CW preflight failure was not retryable");
  CHECK(f->backend_state == state && f->executable == ordinary &&
            f->backend_state->cw_executable == cached_cw && f->t == entry_t,
        "compiled CW preflight failure changed live executable/time state");
  CHECK(counts.cw_solves == solves_before_preflight &&
            counts.cw_callback_effects == callbacks_before_preflight,
        "compiled CW preflight entered dispatch/callback territory");
  counts.fail_cw_preflight = false;
  CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
        "retry after CW preflight failure did not succeed");

  int created_before = counts.cw_executables_created;
  int destroyed_before = counts.cw_executables_destroyed;
  invalidate(*f, MutationKind::source_definition, "CW source-definition invalidation test");
  const BackendEpochSnapshot source_entry_epoch(*f);
  const uint32_t dirty_before_source_preflight = f->dirty_mask;
  fields_chunk **const chunks_before_source_preflight = f->chunks;
  fields_chunk *const first_chunk_before_source_preflight = f->chunks[0];
  CpuArrayCatalog *const catalog_before_source_preflight = f->array_catalog;
  StoragePlan *const storage_before_source_preflight = f->storage_plan;
  DescriptorSet *const descriptors_before_source_preflight = f->descriptors;
  StepPlan *const ordinary_plan_before_source_preflight = f->step_plans[0];
  StepPlan *const cw_plan_before_source_preflight = f->step_plans[1];
  counts.fail_cw_preflight = true;
  failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && !f->backend->is_poisoned() && f->backend_state == state &&
            f->executable == ordinary && f->backend_state->cw_executable == cached_cw &&
            f->dirty_mask == dirty_before_source_preflight,
        "dirty source preflight failure did not restore the live backend epoch");
  CHECK(f->chunks == chunks_before_source_preflight &&
            f->chunks[0] == first_chunk_before_source_preflight &&
            f->array_catalog == catalog_before_source_preflight &&
            f->storage_plan == storage_before_source_preflight &&
            f->descriptors == descriptors_before_source_preflight &&
            f->step_plans[0] == ordinary_plan_before_source_preflight &&
            f->step_plans[1] == cw_plan_before_source_preflight,
        "dirty source preflight failure changed stable chunk/plan metadata identity");
  CHECK(source_entry_epoch.matches(*f),
        "dirty source preflight failure changed exact epoch content");
  counts.fail_cw_preflight = false;
  CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
        "source-definition-invalidated resident solve failed");
  CHECK(counts.cw_executables_created == created_before + 1 &&
            counts.cw_executables_destroyed == destroyed_before + 1,
        "source-definition invalidation did not replace the private CW executable");

  created_before = counts.cw_executables_created;
  destroyed_before = counts.cw_executables_destroyed;
  invalidate(*f, MutationKind::monitor_definition, "CW monitor invalidation test");
  const BackendEpochSnapshot monitor_entry_epoch(*f);
  BackendState *const monitor_entry_state = f->backend_state;
  Executable *const monitor_entry_ordinary = f->executable;
  Executable *const monitor_entry_cw = f->backend_state->cw_executable;
  fields_chunk **const monitor_entry_chunks = f->chunks;
  dft_chunk *const monitor_entry_dft = f->chunks[0]->dft_chunks;
  CHECK(monitor_entry_dft, "dirty monitor rollback fixture has no live DFT chain");
  const uint32_t monitor_entry_dirty = f->dirty_mask;
  counts.fail_cw_preflight = true;
  failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && !f->backend->is_poisoned() && f->backend_state == monitor_entry_state &&
            f->executable == monitor_entry_ordinary &&
            f->backend_state->cw_executable == monitor_entry_cw &&
            f->chunks == monitor_entry_chunks && f->chunks[0]->dft_chunks == monitor_entry_dft &&
            f->dirty_mask == monitor_entry_dirty,
        "dirty monitor preflight failure did not restore the live epoch/DFT ownership");
  CHECK(monitor_entry_epoch.matches(*f),
        "dirty monitor preflight failure changed exact epoch content");
  counts.fail_cw_preflight = false;
  CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
        "monitor-invalidated resident solve failed");
  CHECK(counts.cw_executables_created == created_before + 1 &&
            counts.cw_executables_destroyed == destroyed_before + 1,
        "monitor invalidation did not replace/destroy the private CW executable");

  created_before = counts.cw_executables_created;
  destroyed_before = counts.cw_executables_destroyed;
  invalidate(*f, MutationKind::field_layout, "CW layout invalidation test");
  CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
        "layout-invalidated resident solve failed");
  CHECK(counts.cw_executables_created == created_before + 1 &&
            counts.cw_executables_destroyed == destroyed_before + 1,
        "layout invalidation did not replace/destroy the private CW executable");

  counts.fail_cw_dispatch = true;
  const int callbacks_before_failure = counts.cw_callback_effects;
  failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && f->backend->is_poisoned(), "post-dispatch failure did not poison the backend");
  CHECK(counts.cw_callback_effects == callbacks_before_failure + 1,
        "post-dispatch callback side effect was incorrectly rolled back");
  CHECK(f->t == entry_t, "post-dispatch failure did not restore host time");
  for (int chunk = 0; chunk < f->num_chunks; ++chunk)
    CHECK(!f->chunks[chunk]->is_solving_cw(), "post-dispatch failure did not restore CW flags");

  const int callbacks_after_poison = counts.cw_callback_effects;
  failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && counts.cw_callback_effects == callbacks_after_poison,
        "poisoned retry reached the resident CW dispatch");

  delete f;
  CHECK(counts.cw_executables_created == counts.cw_executables_destroyed,
        "backend-state destruction leaked the private CW executable");
  delete s;
}

static void test_resident_cw_rejection_surface() {
  CwSolveRequest request = cw_request();

  {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, false, true);
    CwSolveRequest bad = request;
    bad.eigfrequency = true;
    expect_cheap_cw_rejection(*f, counts, bad, "eigfrequency request");
    bad = request;
    bad.L = 0;
    expect_cheap_cw_rejection(*f, counts, bad, "L=0 request");
    bad = request;
    bad.maxiters = 0;
    expect_cheap_cw_rejection(*f, counts, bad, "maxiters=0 request");
    bad = request;
    bad.frequency = 0.0;
    expect_cheap_cw_rejection(*f, counts, bad, "zero-frequency request");
    bad = request;
    bad.frequency = std::complex<double>(std::numeric_limits<double>::infinity(), 0.0);
    expect_cheap_cw_rejection(*f, counts, bad, "nonfinite-frequency request");
    delete f;
    delete s;
  }
  {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, false, true);
    CwSolveRequest bad = request;
    bad.tolerance = std::numeric_limits<double>::quiet_NaN();
    expect_cheap_cw_rejection(*f, counts, bad, "nonfinite tolerance");
    delete f;
    delete s;
  }
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml());
    fields f(&s);
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts, false, true);
    expect_cheap_cw_rejection(f, counts, request, "missing source");
  }
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml());
    fields f(&s);
    gaussian_src_time source(0.3, 0.1);
    f.add_point_source(Ez, source, vec(0.0, 0.0), std::complex<double>(0.0, 0.0));
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts, false, true);
    expect_cheap_cw_rejection(f, counts, request, "zero source amplitude");
  }
  {
    structure *s;
    fields *f;
    build(&s, &f);
    f->use_real_fields();
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, false, true);
    expect_cheap_cw_rejection(*f, counts, request, "real fields");
    delete f;
    delete s;
  }
  {
    structure *s;
    fields *f;
    build(&s, &f);
    f->phasein_time = 1;
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, false, true);
    expect_cheap_cw_rejection(*f, counts, request, "active material phase");
    delete f;
    delete s;
  }
  {
    grid_volume gv = volcyl(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml());
    fields f(&s);
    gaussian_src_time source(0.3, 0.1);
    f.add_point_source(Ez, source, gv.center());
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts, false, true);
    expect_cheap_cw_rejection(f, counts, request, "cylindrical coordinates");
  }
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml());
    lorentzian_susceptibility susceptibility(1.1, 0.05);
    s.add_susceptibility(unit_epsilon, E_stuff, susceptibility);
    fields f(&s);
    gaussian_src_time source(0.3, 0.1);
    f.add_point_source(Ez, source, vec(0.0, 0.0));
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts, false, true);
    expect_cheap_cw_rejection(f, counts, request, "polarization topology");
  }
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml());
    fields f(&s, 0.0, 0.2);
    gaussian_src_time source(0.3, 0.1);
    f.add_point_source(Ez, source, vec(0.0, 0.0));
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts, false, true);
    expect_cheap_cw_rejection(f, counts, request, "beta coordinates");
  }
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml());
    std::vector<double> scaled_k(3, 0.0);
    scaled_k[0] = 0.1;
    fields f(&s, 0.0, 0.0, true, 0, 0, scaled_k);
    gaussian_src_time source(0.3, 0.1);
    f.add_point_source(Ez, source, vec(0.0, 0.0));
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts, false, true);
    expect_cheap_cw_rejection(f, counts, request, "BFAST coordinates");
  }
  {
    structure *s;
    fields *f;
    build(&s, &f);
    f->add_flux_plane(vec(-0.8, -1.0), vec(-0.8, 1.0));
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, false, true);
    expect_cheap_cw_rejection(*f, counts, request, "legacy flux accumulator");
    delete f;
    delete s;
  }
  if (count_processors() == 1) {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, true, true);
    f->synchronize_magnetic_fields();
    BackendState *const state = f->backend_state;
    const int preflights = counts.cw_preflights;
    bool rejected = false;
    try {
      (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
    }
    catch (const std::runtime_error &) {
      rejected = true;
    }
    CHECK(rejected && f->backend_state == state && counts.cw_preflights == preflights,
          "live magnetic snapshot crossed the cheap CW preflight boundary");
    f->restore_magnetic_fields();
    delete f;
    delete s;
  }
  {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, false, true);
    f->backend->poison();
    expect_cheap_cw_rejection(*f, counts, request, "poisoned backend");
    delete f;
    delete s;
  }
  {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, false, false);
    expect_cheap_cw_rejection(*f, counts, request, "resident backend without CW support");
    delete f;
    delete s;
  }
  if (count_processors() > 1) {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, false, my_rank() != 0);
    expect_cheap_cw_rejection(*f, counts, request, "rank-asymmetric CW support");
    delete f;
    delete s;
    build(&s, &f);
    lifetime_counts request_counts;
    f->backend = new tracking_backend(*f, request_counts, false, true);
    CwSolveRequest asymmetric = request;
    if (my_rank() == 0) asymmetric.L = 0;
    expect_cheap_cw_rejection(*f, request_counts, asymmetric,
                              "rank-asymmetric invalid CW request");
    delete f;
    delete s;

    build(&s, &f);
    lifetime_counts implicit_absent_counts;
    f->backend = new tracking_backend(*f, implicit_absent_counts, false, true);
    src_time *saved_sources = f->sources;
    if (my_rank() == 0) f->sources = NULL;
    bool implicit_rejected = false;
    try {
      (void)f->solve_cw(1e-6, 20, 2);
    }
    catch (const std::runtime_error &) {
      implicit_rejected = true;
    }
    if (my_rank() == 0) f->sources = saved_sources;
    CHECK(sum_to_all(int(implicit_rejected)) == count_processors() &&
              implicit_absent_counts.cw_preflights == 0 && !f->backend_state,
          "rank-asymmetric absent implicit frequency crossed the collective preflight");
    delete f;
    delete s;

    build(&s, &f);
    lifetime_counts implicit_mismatch_counts;
    f->backend = new tracking_backend(*f, implicit_mismatch_counts, false, true);
    gaussian_src_time other_source(0.3, 0.2);
    f->add_point_source(Ez, other_source, vec(0.2, 0.0));
    CHECK(f->sources && f->sources->next,
          "implicit-frequency mismatch fixture did not register two source times");
    if (my_rank() == 0) f->sources->set_frequency(0.4);
    implicit_rejected = false;
    try {
      (void)f->solve_cw(1e-6, 20, 2);
    }
    catch (const std::runtime_error &) {
      implicit_rejected = true;
    }
    CHECK(sum_to_all(int(implicit_rejected)) == count_processors() &&
              implicit_mismatch_counts.cw_preflights == 0 && !f->backend_state,
          "rank-asymmetric inconsistent implicit frequency crossed the collective preflight");
    delete f;
    delete s;
  }
}

static void test_resident_cw_malformed_results_and_alias() {
  if (count_processors() != 1) return;
  for (int malformed = 1; malformed <= 6; ++malformed) {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    counts.malformed_cw_result = malformed;
    f->backend = new tracking_backend(*f, counts, false, true);
    const int entry_t = f->t;
    bool failed = false;
    try {
      (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
    }
    catch (const std::runtime_error &) {
      failed = true;
    }
    CHECK(failed && f->backend->is_poisoned(),
          "malformed CW result %d did not poison the backend", malformed);
    CHECK(f->t == entry_t, "malformed CW result %d did not restore host time", malformed);
    for (int chunk = 0; chunk < f->num_chunks; ++chunk)
      CHECK(!f->chunks[chunk]->is_solving_cw(),
            "malformed CW result %d left transient mode active", malformed);
    delete f;
    delete s;
  }

  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  counts.alias_cw_to_ordinary = true;
  f->backend = new tracking_backend(*f, counts, false, true);
  bool failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && !f->backend->is_poisoned() && !f->backend_state && !f->executable,
        "ordinary/CW executable alias was published or treated as a dispatch failure");
  CHECK(counts.executables_created == 1 && counts.executables_destroyed == 1,
        "ordinary/CW alias rejection did not roll back the staged ordinary executable");
  delete f;
  CHECK(counts.executables_created == counts.executables_destroyed,
        "ordinary executable ownership was corrupted by CW alias rejection");
  delete s;

  build(&s, &f);
  lifetime_counts mutation_counts;
  mutation_counts.mutate_cw_cache_during_preflight = true;
  f->backend = new tracking_backend(*f, mutation_counts, false, true);
  failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && !f->backend->is_poisoned() && !f->backend_state && !f->executable,
        "preflight cache mutation escaped the retryable staged boundary");
  CHECK(mutation_counts.cw_executables_created ==
            mutation_counts.cw_executables_destroyed,
        "preflight cache mutation leaked or double-owned its executable");
  delete f;
  delete s;

  build(&s, &f);
  lifetime_counts boundary_counts;
  boundary_counts.mutate_after_cw_boundary = true;
  f->backend = new tracking_backend(*f, boundary_counts, false, true);
  const int boundary_entry_t = f->t;
  failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && f->backend->is_poisoned() && f->t == boundary_entry_t,
        "post-boundary state mutation was not rejected and defensively restored");
  for (int chunk = 0; chunk < f->num_chunks; ++chunk)
    CHECK(!f->chunks[chunk]->is_solving_cw(),
          "post-boundary state mutation escaped defensive session cleanup");
  delete f;
  delete s;
}

static void test_cw_clone_failure_is_atomic() {
  if (count_processors() != 1) return;
  bool reached_end_of_clone = false;
  for (int checkpoint = 0; checkpoint < 512 && !reached_end_of_clone; ++checkpoint) {
    structure *s;
    fields *f;
    build(&s, &f);
    simple_material_function conductivity_material(phase_conductivity);
    std::vector<realnum *> entry_condinv;
    for (int chunk = 0; chunk < f->num_chunks; ++chunk) {
      fields_chunk &fc = *f->chunks[chunk];
      structure_chunk *old_structure = fc.s;
      fc.s = new structure_chunk(old_structure);
      if (old_structure->refcount-- <= 1) delete old_structure;
      if (fc.s->is_mine()) {
        fc.s->set_conductivity(Dz, conductivity_material);
        fc.s->update_condinv();
      }
      entry_condinv.push_back(fc.s->condinv[Dz][Z]);
    }
    CHECK(or_to_all(entry_condinv[0] != NULL),
          "clone failpoint fixture did not realize live conductivity/condinv storage");
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, false, true);
    const BackendEpochSnapshot entry_epoch(*f);
    fields_chunk **const entry_chunks = f->chunks;
    const uint32_t entry_dirty = f->dirty_mask;
    std::vector<int> entry_refcounts;
    for (int chunk = 0; chunk < f->num_chunks; ++chunk)
      entry_refcounts.push_back(f->chunks[chunk]->s->refcount);
    backend_set_cw_clone_fail_after_for_testing(checkpoint);
    bool failed = false;
    try {
      (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
    }
    catch (const std::runtime_error &) {
      failed = true;
    }
    backend_set_cw_clone_fail_after_for_testing(-1);
    if (failed) {
      CHECK(!f->backend_state && !f->executable && f->chunks == entry_chunks &&
                f->dirty_mask == entry_dirty && entry_epoch.matches(*f),
            "clone checkpoint %d published a partial staged epoch", checkpoint);
      for (int chunk = 0; chunk < f->num_chunks; ++chunk)
        CHECK(f->chunks[chunk]->s->refcount == entry_refcounts[size_t(chunk)] &&
                  f->chunks[chunk]->s->condinv[Dz][Z] == entry_condinv[size_t(chunk)],
              "clone checkpoint %d leaked a structure reference", checkpoint);
    }
    else
      reached_end_of_clone = true;
    delete f;
    delete s;
  }
  CHECK(reached_end_of_clone,
        "clone failpoint sweep did not reach metadata/communication completion");
}

static void test_cw_staged_pipeline_failures() {
  if (count_processors() != 1) return;
  for (int injection = 0; injection < 5; ++injection) {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    if (injection == 0) counts.fail_create_state = true;
    if (injection == 1) counts.fail_initialize = true;
    if (injection == 2) counts.fail_compile = true;
    if (injection == 3) counts.corrupt_catalog_after_compile = true;
    if (injection == 4) backend_set_cw_plan_corruption_for_testing(true);
    f->backend = new tracking_backend(*f, counts, false, true);
    const BackendEpochSnapshot entry_epoch(*f);

    fields_chunk **const entry_chunks = f->chunks;
    halo_plan_set *const entry_halos = f->halos;
    CpuArrayCatalog *const entry_catalog = f->array_catalog;
    StoragePlan *const entry_storage = f->storage_plan;
    DescriptorSet *const entry_descriptors = f->descriptors;
    InitializationPlan *const entry_initialization = f->initialization_plan;
    StepPlan *const entry_steps[2] = {f->step_plans[0], f->step_plans[1]};
    const uint32_t entry_dirty = f->dirty_mask;
    const uint32_t entry_prepared = f->storage_prepared_mask;
    const uint64_t entry_classification = f->prepared_classification_hash;
    const uint32_t entry_reentries = f->classification_reentries;
    uint64_t entry_generations[fields::num_mutation_kinds];
    for (int i = 0; i < fields::num_mutation_kinds; ++i)
      entry_generations[i] = f->mutation_generation[i];
    const std::vector<const void *> entry_raw = cw_chunk_storage_addresses(*f);

    bool failed = false;
    try {
      (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
    }
    catch (const std::runtime_error &) {
      failed = true;
    }
    CHECK(failed && !f->backend->is_poisoned() && !f->backend_state && !f->executable,
          "staged pipeline injection %d was not retryable", injection);
    CHECK(entry_epoch.matches(*f),
          "staged pipeline injection %d changed the exact host epoch", injection);
    CHECK(f->chunks == entry_chunks && f->halos == entry_halos &&
              f->array_catalog == entry_catalog && f->storage_plan == entry_storage &&
              f->descriptors == entry_descriptors && f->initialization_plan == entry_initialization &&
              f->step_plans[0] == entry_steps[0] && f->step_plans[1] == entry_steps[1] &&
              f->dirty_mask == entry_dirty && f->storage_prepared_mask == entry_prepared &&
              f->prepared_classification_hash == entry_classification &&
              f->classification_reentries == entry_reentries &&
              cw_chunk_storage_addresses(*f) == entry_raw,
          "staged pipeline injection %d changed the host epoch", injection);
    for (int i = 0; i < fields::num_mutation_kinds; ++i)
      CHECK(f->mutation_generation[i] == entry_generations[i],
            "staged pipeline injection %d changed mutation generation %d", injection, i);

    counts.fail_create_state = false;
    counts.fail_initialize = false;
    counts.fail_compile = false;
    counts.corrupt_catalog_after_compile = false;
    backend_set_cw_plan_corruption_for_testing(false);
    CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
          "staged pipeline injection %d did not permit retry", injection);
    delete f;
    CHECK(counts.states_created == counts.states_destroyed &&
              counts.executables_created == counts.executables_destroyed &&
              counts.cw_executables_created == counts.cw_executables_destroyed,
          "staged pipeline injection %d leaked a backend artifact", injection);
    delete s;
  }
}

static void test_cw_structural_rebuild_migrates_authority() {
  if (count_processors() != 1) return;
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts, false, true);
  CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
        "authority-migration fixture did not create a resident epoch");
  realnum *entry_field = first_owned_real_field(*f);
  CHECK(entry_field, "authority-migration fixture has no owned real field");
  if (entry_field) *entry_field = realnum(-17);
  counts.migrate_authoritative_value = true;
  counts.authoritative_value = realnum(23);
  counts.fail_cw_preflight = true;
  BackendState *const entry_state = f->backend_state;
  fields_chunk **const entry_chunks = f->chunks;
  invalidate(*f, MutationKind::source_definition, "CW authority migration rollback");
  bool failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && !f->backend->is_poisoned() && f->backend_state == entry_state &&
            f->chunks == entry_chunks && first_owned_real_field(*f) == entry_field &&
            (!entry_field || *entry_field == realnum(23)),
        "failed structural rebuild lost the migrated authoritative value");
  counts.fail_cw_preflight = false;
  CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2) &&
            f->chunks == entry_chunks && first_owned_real_field(*f) &&
            *first_owned_real_field(*f) == realnum(23),
        "successful structural rebuild did not clone the migrated authoritative value");
  delete f;
  delete s;
}

static void test_cw_warm_staged_pipeline_failures() {
  if (count_processors() != 1) return;
  for (int injection = 0; injection < 5; ++injection) {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, false, true);
    CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
          "warm staged-failure fixture did not create its entry epoch");
    invalidate(*f, MutationKind::source_definition, "warm staged pipeline failure");
    if (injection == 0) counts.fail_create_state = true;
    if (injection == 1) counts.fail_initialize = true;
    if (injection == 2) counts.fail_compile = true;
    if (injection == 3) counts.corrupt_catalog_after_compile = true;
    if (injection == 4) backend_set_cw_plan_corruption_for_testing(true);
    const BackendEpochSnapshot entry_epoch(*f);
    const int solves_before = counts.cw_solves;
    bool failed = false;
    try {
      (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
    }
    catch (const std::runtime_error &) {
      failed = true;
    }
    const BackendEpochSnapshot after_failure(*f);
    CHECK(failed && !f->backend->is_poisoned() && entry_epoch.matches(after_failure) &&
              counts.cw_solves == solves_before,
          "warm staged pipeline injection %d changed the entry epoch", injection);
    if (!entry_epoch.matches(after_failure))
      printf("[rank %d] warm injection %d snapshot: ptr=%d raw=%d catalog=%d storage=%d "
             "descriptor=%d halo=%d comm=%d dirty=%d lifecycle=%d cache=%d\n",
             my_rank(), injection,
             int(entry_epoch.chunks == after_failure.chunks &&
                 entry_epoch.state == after_failure.state &&
                 entry_epoch.ordinary == after_failure.ordinary),
             int(entry_epoch.raw == after_failure.raw &&
                 entry_epoch.raw_hash == after_failure.raw_hash),
             int(entry_epoch.catalog == after_failure.catalog &&
                 entry_epoch.catalog_hash == after_failure.catalog_hash),
             int(entry_epoch.storage == after_failure.storage &&
                 entry_epoch.storage_hash == after_failure.storage_hash),
             int(entry_epoch.descriptors == after_failure.descriptors &&
                 entry_epoch.descriptor_hash == after_failure.descriptor_hash),
             int(entry_epoch.halos == after_failure.halos &&
                 entry_epoch.halo_hash == after_failure.halo_hash),
             int(entry_epoch.comm == after_failure.comm &&
                 entry_epoch.comm_hash == after_failure.comm_hash),
             int(entry_epoch.dirty == after_failure.dirty),
             int(entry_epoch.connections_generation == after_failure.connections_generation &&
                 entry_epoch.connections_built_generation ==
                     after_failure.connections_built_generation &&
                 entry_epoch.local_generation == after_failure.local_generation &&
                 entry_epoch.local_synced == after_failure.local_synced),
             int(entry_epoch.cw == after_failure.cw &&
                 entry_epoch.cw_storage_key == after_failure.cw_storage_key &&
                 entry_epoch.cw_step_key == after_failure.cw_step_key &&
                 entry_epoch.cw_plan_key == after_failure.cw_plan_key));
    if (!entry_epoch.matches(after_failure))
      printf("[rank %d] warm injection %d snapshot2: dft=%d prepared=%d class=%d flags=%d "
             "mutations=%d steps=%d catalog_values=%d metadata_ptrs=%d\n",
             my_rank(), injection, int(entry_epoch.dft_fc == after_failure.dft_fc),
             int(entry_epoch.prepared_mask == after_failure.prepared_mask),
             int(entry_epoch.classification_hash == after_failure.classification_hash &&
                 entry_epoch.classification_reentries == after_failure.classification_reentries),
             int(entry_epoch.components_allocated == after_failure.components_allocated &&
                 entry_epoch.connections_valid == after_failure.connections_valid &&
                 entry_epoch.changed_materials == after_failure.changed_materials),
             int(std::equal(entry_epoch.mutations,
                            entry_epoch.mutations + fields::num_mutation_kinds,
                            after_failure.mutations)),
             int(entry_epoch.initialization == after_failure.initialization &&
                 entry_epoch.steps[0] == after_failure.steps[0] &&
                 entry_epoch.steps[1] == after_failure.steps[1]),
             int(entry_epoch.catalog_values == after_failure.catalog_values),
             int(entry_epoch.catalog == after_failure.catalog &&
                 entry_epoch.storage == after_failure.storage &&
                 entry_epoch.descriptors == after_failure.descriptors &&
                 entry_epoch.halos == after_failure.halos));
    if (entry_epoch.catalog_values != after_failure.catalog_values)
      for (size_t i = 0; i < entry_epoch.catalog_values.size(); ++i)
        if (entry_epoch.catalog_values[i] != after_failure.catalog_values[i]) {
          size_t changed_byte = 0;
          while (changed_byte < entry_epoch.catalog_bytes[i].size() &&
                 entry_epoch.catalog_bytes[i][changed_byte] ==
                     after_failure.catalog_bytes[i][changed_byte])
            ++changed_byte;
          printf("[rank %d] warm injection %d changed catalog value row %zu kind=%d component=%d\n",
                 my_rank(), injection, i, int(f->array_catalog->key(ArrayId{uint32_t(i)}).kind),
                 f->array_catalog->key(ArrayId{uint32_t(i)}).component_);
          printf("[rank %d] first changed byte %zu: %u -> %u\n", my_rank(), changed_byte,
                 unsigned(entry_epoch.catalog_bytes[i][changed_byte]),
                 unsigned(after_failure.catalog_bytes[i][changed_byte]));
        }
    counts.fail_create_state = false;
    counts.fail_initialize = false;
    counts.fail_compile = false;
    counts.corrupt_catalog_after_compile = false;
    backend_set_cw_plan_corruption_for_testing(false);
    CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
          "warm staged pipeline injection %d did not permit retry", injection);
    delete f;
    CHECK(counts.states_created == counts.states_destroyed &&
              counts.executables_created == counts.executables_destroyed &&
              counts.cw_executables_created == counts.cw_executables_destroyed,
          "warm staged pipeline injection %d leaked a backend artifact", injection);
    delete s;
  }
}

static void test_cw_cold_conductivity_transaction() {
  if (count_processors() != 1) return;
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure *s = new structure(gv, eps_slab, pml(0.5));
  s->set_conductivity(Dz, phase_conductivity);
  fields *f = new fields(s);
  gaussian_src_time source(0.3, 0.1);
  f->add_point_source(Ez, source, vec(0.11, 0.13));
  lifetime_counts counts;
  counts.fail_cw_preflight = true;
  f->backend = new tracking_backend(*f, counts, false, true);
  fields_chunk *const entry_chunk = f->chunks[0];
  structure_chunk *const entry_structure = entry_chunk->s;
  realnum *const entry_condinv = entry_structure->condinv[Dz][Z];
  const cw_source_snapshot entry_sources = snapshot_cw_sources(*f);
  bool failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && counts.cw_preflights == 1 && f->chunks[0] == entry_chunk &&
            f->chunks[0]->s == entry_structure &&
            f->chunks[0]->s->condinv[Dz][Z] == entry_condinv && !f->backend_state,
        "cold conductive preflight rollback published staged condinv/structure state");
  CHECK(snapshot_cw_sources(*f) == entry_sources,
        "cold conductive preflight rollback changed source indices/amplitudes/boundary state");
  counts.fail_cw_preflight = false;
  CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
        "cold conductive resident solve did not retry");
  CHECK(f->chunks[0] == entry_chunk && f->chunks[0]->s != entry_structure &&
            f->chunks[0]->s->condinv[Dz][Z] &&
            is_valid(f->array_catalog->find(
                {0, int(array_kind::condinv), int(Dz), -1, int(Z)})),
        "cold conductive commit did not publish catalogued staged condinv storage");
  delete f;
  delete s;
}

static void test_cw_warm_monitor_pipeline_failures() {
  if (count_processors() != 1) return;
  for (int injection = 0; injection < 2; ++injection) {
    structure *s;
    fields *f;
    build(&s, &f);
    component c = Ez;
    dft_fields monitor = f->add_dft_fields(&c, 1, f->v, 0.3, 0.3, 1, 3);
    (void)monitor;
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts, false, true);
    CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
          "warm monitor failure fixture did not create its entry epoch");
    invalidate(*f, MutationKind::monitor_definition, "warm monitor staged pipeline failure");
    if (injection == 0) counts.fail_create_state = true;
    if (injection == 1) counts.fail_compile = true;
    const BackendEpochSnapshot entry_epoch(*f);
    bool failed = false;
    try {
      (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
    }
    catch (const std::runtime_error &) {
      failed = true;
    }
    const BackendEpochSnapshot after_failure(*f);
    CHECK(failed && !f->backend->is_poisoned() && entry_epoch.matches(after_failure),
          "warm monitor pipeline injection %d changed DFT/backend epoch state", injection);
    if (!entry_epoch.matches(after_failure))
      for (size_t i = 0; i < entry_epoch.catalog_bytes.size(); ++i)
        if (entry_epoch.catalog_bytes[i] != after_failure.catalog_bytes[i])
          printf("[rank %d] warm monitor injection %d changed catalog row %zu kind=%d\n",
                 my_rank(), injection, i,
                 int(f->array_catalog->key(ArrayId{uint32_t(i)}).kind));
    counts.fail_create_state = false;
    counts.fail_compile = false;
    CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
          "warm monitor pipeline injection %d did not permit retry", injection);
    delete f;
    delete s;
  }
}

static void test_cold_cw_preflight_restores_existing_plans() {
  if (count_processors() != 1) return;
  structure *s;
  fields *f;
  build(&s, &f);
  f->advance(1);
  if (!f->initialization_plan)
    f->initialization_plan = new InitializationPlan(build_initialization_plan(*f));
  CHECK(f->backend_state && f->executable && f->initialization_plan && f->step_plans[0],
        "CPU setup did not create the expected preparation artifacts");

  const uint64_t ordinary_signature = f->step_plans[0]->signature;
  const bool had_cw_plan = f->step_plans[1] != NULL;
  const uint64_t cw_signature = had_cw_plan ? f->step_plans[1]->signature : 0;
  const size_t init_ops = f->initialization_plan->operations.size();

  delete f->executable;
  f->executable = NULL;
  delete f->backend_state;
  f->backend_state = NULL;
  delete f->backend;
  lifetime_counts counts;
  counts.fail_cw_preflight = true;
  f->backend = new tracking_backend(*f, counts, false, true);
  invalidate(*f, MutationKind::precision_policy, "CPU-to-resident CW test");
  const uint32_t entry_dirty = f->dirty_mask;
  const BackendEpochSnapshot entry_epoch(*f);

  bool failed = false;
  try {
    (void)f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(failed && !f->backend_state && !f->executable && f->dirty_mask == entry_dirty,
        "cold CPU-to-resident preflight failure published state or dirty changes");
  CHECK(f->initialization_plan && f->initialization_plan->operations.size() == init_ops,
        "cold preflight failure did not restore the initialization plan");
  CHECK(f->step_plans[0] && f->step_plans[0]->signature == ordinary_signature &&
            (!!f->step_plans[1] == had_cw_plan) &&
            (!had_cw_plan || f->step_plans[1]->signature == cw_signature),
        "cold preflight failure did not restore exact StepPlan signatures");
  const BackendEpochSnapshot after_cold_failure(*f);
  CHECK(entry_epoch.matches(after_cold_failure),
        "cold CPU-prepared preflight failure changed exact epoch content");
  if (!entry_epoch.matches(after_cold_failure))
    printf("[rank %d] cold CPU snapshot: raw=%d catalog=%d storage=%d descriptor=%d halo=%d "
           "comm_ptr=%d comm_hash=%d(block=%d seq=%d map=%d) source=%d dft=%d lifecycle=%d\n",
           my_rank(),
           int(entry_epoch.raw == after_cold_failure.raw &&
               entry_epoch.raw_hash == after_cold_failure.raw_hash &&
               entry_epoch.catalog_bytes == after_cold_failure.catalog_bytes),
           int(entry_epoch.catalog == after_cold_failure.catalog &&
               entry_epoch.catalog_hash == after_cold_failure.catalog_hash),
           int(entry_epoch.storage == after_cold_failure.storage &&
               entry_epoch.storage_hash == after_cold_failure.storage_hash),
           int(entry_epoch.descriptors == after_cold_failure.descriptors &&
               entry_epoch.descriptor_hash == after_cold_failure.descriptor_hash),
           int(entry_epoch.halos == after_cold_failure.halos &&
               entry_epoch.halo_hash == after_cold_failure.halo_hash),
           int(entry_epoch.comm == after_cold_failure.comm),
           int(entry_epoch.comm_hash == after_cold_failure.comm_hash),
           int(entry_epoch.comm_block_hash == after_cold_failure.comm_block_hash),
           int(entry_epoch.comm_sequence_hash == after_cold_failure.comm_sequence_hash),
           int(entry_epoch.comm_map_hash == after_cold_failure.comm_map_hash),
           int(entry_epoch.source_metadata == after_cold_failure.source_metadata &&
               entry_epoch.source_indices == after_cold_failure.source_indices &&
               entry_epoch.source_amplitudes == after_cold_failure.source_amplitudes),
           int(entry_epoch.dft_fc == after_cold_failure.dft_fc),
           int(entry_epoch.dirty == after_cold_failure.dirty &&
               entry_epoch.connections_generation == after_cold_failure.connections_generation &&
               entry_epoch.local_generation == after_cold_failure.local_generation));

  counts.fail_cw_preflight = false;
  CHECK(f->solve_cw(1e-6, 20, std::complex<double>(0.3, 0.0), 2),
        "cold preflight rollback did not permit retry");
  delete f;
  delete s;
}

static void test_cpu_cw_hook_declines_without_initialization() {
  grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, unit_epsilon, no_pml());
  fields f(&s);
  gaussian_src_time source(0.3, 0.1);
  f.add_point_source(Ez, source, vec(0.0, 0.0));
  CwSolveResult result;
  CHECK(!backend_try_solve_cw(f, cw_request(), result),
        "unselected CPU/default path claimed resident solve_cw");
  CHECK(!f.backend && !f.backend_state && !f.executable,
        "declined CPU/default hook initialized backend state");
  f.init_backend();
  BackendState *state = f.backend_state;
  CHECK(!backend_try_solve_cw(f, cw_request(), result) && f.backend_state == state,
        "initialized CPU backend did not retain the legacy solve_cw path");
}

static void test_resident_magnetic_dispatch() {
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts, true);

  counts.fail_compile = my_rank() == 0;
  bool compile_failed = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    compile_failed = true;
  }
  CHECK(sum_to_all(int(compile_failed)) == count_processors(),
        "rank-asymmetric magnetic compile failure was not reconciled");
  CHECK(counts.magnetic_synchronizes == 0,
        "magnetic compile failure entered the synchronization dispatch");
  CHECK(!f->executable && counts.executables_created == counts.executables_destroyed,
        "failed first compile retained a partial executable");
  counts.fail_compile = false;

  f->synchronize_magnetic_fields();
  CHECK(counts.states_created == 2 && counts.states_destroyed == 1 &&
            counts.executables_created == counts.executables_destroyed + 1,
        "pre-step resident sync did not prepare state and executable");
  CHECK(counts.magnetic_synchronizes == 1 && counts.magnetic_restores == 0,
        "first resident sync dispatched the wrong transition");
  for (int chunk = 0; chunk < f->num_chunks; ++chunk)
    if (f->chunks[chunk]->is_mine()) DOCMP2 FOR_COMPONENTS(c) {
        CHECK(!f->chunks[chunk]->f_backup[c][cmp] && !f->chunks[chunk]->f_u_backup[c][cmp] &&
                  !f->chunks[chunk]->f_w_backup[c][cmp] &&
                  !f->chunks[chunk]->f_cond_backup[c][cmp] &&
                  !f->chunks[chunk]->f_bfast_backup[c][cmp],
              "resident sync allocated legacy host backup storage");
      }

  f->synchronize_magnetic_fields();
  CHECK(counts.magnetic_synchronizes == 1, "nested resident sync performed work");
  f->restore_magnetic_fields();
  CHECK(counts.magnetic_restores == 0, "inner resident restore performed work");

  BackendState *state = f->backend_state;
  Executable *executable = f->executable;
  bool rebuild_rejected = false;
  try {
    backend_prepare_field_layout_change(*f, dirty_storage, "magnetic rebuild test");
  }
  catch (const std::runtime_error &) {
    rebuild_rejected = true;
  }
  CHECK(sum_to_all(int(rebuild_rejected)) == count_processors(),
        "live resident magnetic snapshot did not reject a state rebuild collectively");
  CHECK(f->backend_state == state && f->executable == executable,
        "rejected magnetic rebuild changed state or executable");

  counts.fail_magnetic_restore = my_rank() == 0;
  bool restore_failed = false;
  try {
    f->restore_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    restore_failed = true;
  }
  CHECK(sum_to_all(int(restore_failed)) == count_processors(),
        "resident magnetic restore failure was not reconciled");
  CHECK(counts.magnetic_restores == 0, "failed resident restore completed the transition");
  counts.fail_magnetic_restore = false;
  f->restore_magnetic_fields();
  CHECK(counts.magnetic_restores == 1, "final resident restore did not dispatch exactly once");
  f->restore_magnetic_fields();
  CHECK(counts.magnetic_restores == 1, "unmatched resident restore performed work");

  Executable *compiled = f->executable;
  const int created_before_recompile = counts.executables_created;
  const int destroyed_before_recompile = counts.executables_destroyed;
  f->dirty_mask |= dirty_executable;
  counts.fail_compile = my_rank() == 0;
  compile_failed = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    compile_failed = true;
  }
  CHECK(sum_to_all(int(compile_failed)) == count_processors(),
        "rank-asymmetric replacement compile failure was not reconciled");
  CHECK(f->executable == compiled && is_dirty(*f, dirty_executable),
        "replacement compile failure lost the prior executable or dirty bit");
  CHECK(counts.executables_created - created_before_recompile ==
            counts.executables_destroyed - destroyed_before_recompile,
        "replacement compile failure leaked a candidate executable");
  CHECK(counts.magnetic_synchronizes == 1,
        "replacement compile failure entered the synchronization dispatch");
  counts.fail_compile = false;
  f->synchronize_magnetic_fields();
  CHECK(!is_dirty(*f, dirty_executable) &&
            counts.executables_created == created_before_recompile + 1 + (my_rank() != 0) &&
            counts.executables_destroyed == destroyed_before_recompile + 1 + (my_rank() != 0),
        "replacement compile retry did not replace exactly one live executable");
  f->restore_magnetic_fields();

  counts.fail_magnetic_synchronize = my_rank() == 0;
  bool sync_failed = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    sync_failed = true;
  }
  CHECK(sum_to_all(int(sync_failed)) == count_processors(),
        "resident magnetic synchronize failure was not reconciled");
  CHECK(counts.magnetic_synchronizes == 2, "failed resident sync completed the transition");
  counts.fail_magnetic_synchronize = false;
  f->synchronize_magnetic_fields();
  f->restore_magnetic_fields();
  CHECK(counts.magnetic_synchronizes == 3 && counts.magnetic_restores == 3,
        "resident magnetic transition did not recover after injected failures");

  counts.fail_magnetic_synchronize_dispatch = my_rank() == 0;
  bool dispatch_failed = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    dispatch_failed = true;
  }
  CHECK(sum_to_all(int(dispatch_failed)) == count_processors(),
        "rank-asymmetric dispatch failure was not reconciled");
  CHECK(f->backend->is_poisoned(), "dispatch failure did not poison every resident backend");
  bool poison_rejected = false;
  try {
    f->restore_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    poison_rejected = true;
  }
  CHECK(sum_to_all(int(poison_rejected)) == count_processors(),
        "poisoned resident backend accepted another magnetic transition");
  bool layout_rejected = false;
  try {
    backend_prepare_field_layout_change(*f, dirty_storage, "poisoned magnetic test");
  }
  catch (const std::runtime_error &) {
    layout_rejected = true;
  }
  CHECK(sum_to_all(int(layout_rejected)) == count_processors(),
        "poisoned resident backend accepted a layout rebuild");
  bool advance_rejected = false;
  try {
    f->advance(1);
  }
  catch (const std::runtime_error &) {
    advance_rejected = true;
  }
  CHECK(sum_to_all(int(advance_rejected)) == count_processors(),
        "poisoned resident backend accepted an ordinary advance");

  delete f;
  delete s;
}

struct rebuild_trace {
  std::vector<std::string> events;
  DirtyMask reasons;

  rebuild_trace() : reasons(dirty_none) {}
};

struct access_trace {
  size_t reads;
  size_t writes;
  size_t field_reads;
  size_t dft_reads;
  size_t field_writes;
  size_t dft_writes;
  size_t max_elements;
  int prepare_rebuilds;
  int fail_read_rank;
  int fail_write_rank;
  int fail_dft_read_rank;
  int fail_dft_write_rank;

  access_trace()
      : reads(0), writes(0), field_reads(0), dft_reads(0), field_writes(0), dft_writes(0),
        max_elements(0), prepare_rebuilds(0), fail_read_rank(-1), fail_write_rank(-1),
        fail_dft_read_rank(-1), fail_dft_write_rank(-1) {}
};

struct compact_trace {
  size_t calls;
  size_t returned_bytes;
  int fail_rank;
  int fail_call;
  std::vector<DftReductionRequest> requests;

  compact_trace() : calls(0), returned_bytes(0), fail_rank(-1), fail_call(-1) {}
};

struct rebuild_state : BackendState {
  explicit rebuild_state(rebuild_trace &trace_) : trace(trace_) {}
  ~rebuild_state() override { trace.events.push_back("destroy-state"); }
  rebuild_trace &trace;
};

struct rebuild_executable : Executable {
  explicit rebuild_executable(rebuild_trace &trace_) : trace(trace_) {}
  ~rebuild_executable() override { trace.events.push_back("destroy-executable"); }
  rebuild_trace &trace;
};

class rebuild_backend_base : public ExecutionBackend {
public:
  rebuild_backend_base(fields &f_, rebuild_trace &trace_) : f(f_), trace(trace_) {}

  BackendState *create_state(const StoragePlan &) override {
    trace.events.push_back("create-state");
    return new rebuild_state(trace);
  }
  void initialize(const InitializationPlan &, BackendState &) override {}
  MaterialClassification classify_state(const StoragePlan &plan, BackendState &) override {
    return classify(f, plan);
  }
  void finalize_storage(const StoragePlan &, const MaterialClassification &,
                        BackendState &) override {}
  Executable *compile(const StepPlan &, BackendState &) override {
    return new rebuild_executable(trace);
  }
  void advance(Executable &, BackendState &, int) override {}
  void read(ArrayRef, void *, size_t) override {}
  void write(ArrayRef, const void *, size_t) override {}
  void synchronize() override {}
  backend_capabilities capabilities() const override {
    backend_capabilities result;
    result.supports_native = true;
    result.supports_mixed = false;
    result.supports_f32 = false;
    result.memory_budget_bytes = 0;
    result.name = "tracking";
    return result;
  }
  bool requires_full_storage_preparation() const override { return true; }
  bool accepts(const execution_options &, std::string &) const override { return true; }

protected:
  fields &f;
  rebuild_trace &trace;
};

class rebuild_tracking_backend : public rebuild_backend_base {
public:
  rebuild_tracking_backend(fields &f, rebuild_trace &trace_) : rebuild_backend_base(f, trace_) {}

  void prepare_state_rebuild(BackendState &, DirtyMask reasons) override {
    trace.events.push_back("prepare-rebuild");
    trace.reasons = reasons;
  }
};

class access_tracking_backend : public rebuild_backend_base {
public:
  access_tracking_backend(fields &f, rebuild_trace &rebuilds, access_trace &accesses_)
      : rebuild_backend_base(f, rebuilds), accesses(accesses_) {}
  void advance(Executable &, BackendState &, int num_steps) override { f.t += num_steps; }

  void read(ArrayRef ref, void *host_buffer, size_t bytes) override {
    const ArraySpec &spec = f.array_catalog->spec(ref.id);
    if (my_rank() == accesses.fail_read_rank ||
        (spec.role == array_role::dft && my_rank() == accesses.fail_dft_read_rank))
      throw std::runtime_error("injected rank-local backend read failure");
    CHECK(bytes == ref.elements * host_element_bytes(spec.element_type),
          "backend host-range read byte count is inconsistent");
    ++accesses.reads;
    accesses.field_reads += spec.role == array_role::field;
    accesses.dft_reads += spec.role == array_role::dft;
    if (ref.elements > accesses.max_elements) accesses.max_elements = ref.elements;
    if (spec.element_type == ElementType::realnum_value) {
      realnum *values = static_cast<realnum *>(host_buffer);
      for (size_t i = 0; i < ref.elements; ++i)
        values[i] = realnum(2.5);
    }
    else if (spec.element_type == ElementType::complex_realnum) {
      std::complex<realnum> *values = static_cast<std::complex<realnum> *>(host_buffer);
      for (size_t i = 0; i < ref.elements; ++i)
        values[i] = std::complex<realnum>(realnum(1.25), realnum(-0.75));
    }
  }

  void write(ArrayRef ref, const void *, size_t bytes) override {
    const ArraySpec &spec = f.array_catalog->spec(ref.id);
    if (my_rank() == accesses.fail_write_rank ||
        (spec.role == array_role::dft && my_rank() == accesses.fail_dft_write_rank))
      throw std::runtime_error("injected rank-local backend write failure");
    CHECK(bytes == ref.elements * host_element_bytes(spec.element_type),
          "backend host-range write byte count is inconsistent");
    ++accesses.writes;
    accesses.field_writes += spec.role == array_role::field;
    accesses.dft_writes += spec.role == array_role::dft;
    if (ref.elements > accesses.max_elements) accesses.max_elements = ref.elements;
  }

  void prepare_state_rebuild(BackendState &, DirtyMask reasons) override {
    CHECK((reasons & dirty_storage) != 0,
          "checkpoint replacement did not request storage-safe teardown");
    ++accesses.prepare_rebuilds;
  }

private:
  access_trace &accesses;
};

class compact_access_backend : public access_tracking_backend {
public:
  compact_access_backend(fields &f_, rebuild_trace &rebuilds, access_trace &accesses,
                         compact_trace &compact_)
      : access_tracking_backend(f_, rebuilds, accesses), compact(compact_) {}

  bool supports_compact_dft_reductions() const override { return true; }

  void reduce_dft(const DftReductionRequest &request, std::complex<double> *result,
                  size_t result_count) override {
    const size_t call = compact.calls++;
    if (my_rank() == compact.fail_rank && int(call) == compact.fail_call)
      throw std::runtime_error("injected rank-local compact DFT reduction failure");
    if (request.result_count != result_count)
      throw std::invalid_argument("compact result-count mismatch");
    if ((request.kind == DftReductionKind::norm2) != (result_count == 1) ||
        (request.kind != DftReductionKind::norm2 && result_count == 0))
      throw std::invalid_argument("compact kind/result-count mismatch");
    if (request.accumulation_precision != policy_for(f.options.precision).reduction)
      throw std::invalid_argument("compact reduction precision mismatch");

    for (size_t ti = 0; ti < request.terms.size(); ++ti) {
      const DftReductionTerm &term = request.terms[ti];
      if (!is_valid(term.left) || term.left.value >= f.array_catalog->size())
        throw std::out_of_range("invalid compact left array");
      const ArraySpec &left = f.array_catalog->spec(term.left);
      if (left.role != array_role::dft || left.element_type != ElementType::complex_realnum ||
          is_valid(left.alias_of))
        throw std::invalid_argument("compact left array has the wrong storage kind");
      if (!term.storage_points || !term.frequencies ||
          term.storage_points > std::numeric_limits<size_t>::max() / term.frequencies ||
          term.storage_points * term.frequencies > left.elements)
        throw std::out_of_range("compact left array is too small");
      if (request.kind == DftReductionKind::norm2) {
        if (is_valid(term.right)) throw std::invalid_argument("norm request has a right array");
      }
      else {
        if (!is_valid(term.right) || term.right.value >= f.array_catalog->size())
          throw std::out_of_range("invalid compact right array");
        const ArraySpec &right = f.array_catalog->spec(term.right);
        if (right.role != array_role::dft || right.element_type != ElementType::complex_realnum ||
            is_valid(right.alias_of) || right.storage != left.storage ||
            term.storage_points * term.frequencies > right.elements)
          throw std::invalid_argument("compact DFT pair has incompatible storage");
        if (term.frequencies != result_count)
          throw std::invalid_argument("compact product frequency mismatch");
      }
      size_t maximum = term.region.base;
      for (int axis = 0; axis < 3; ++axis) {
        if (!term.region.counts[axis])
          throw std::invalid_argument("compact region has a zero count");
        const size_t extent = term.region.counts[axis] - 1;
        if (term.region.strides[axis] &&
            extent > (std::numeric_limits<size_t>::max() - maximum) / term.region.strides[axis])
          throw std::overflow_error("compact region overflows");
        maximum += extent * term.region.strides[axis];
      }
      if (maximum >= term.storage_points) throw std::out_of_range("compact region exceeds storage");
    }

    compact.requests.push_back(request);
    compact.returned_bytes += result_count * sizeof(std::complex<double>);
    for (size_t i = 0; i < result_count; ++i) {
      const double value = double((my_rank() + 1) * (i + 1));
      result[i] += request.kind == DftReductionKind::complex_weighted_product
                       ? std::complex<double>(value, -0.25 * value)
                       : std::complex<double>(value, 0.0);
    }
  }

private:
  compact_trace &compact;
};

static void build(structure **sp, fields **fp, const execution_options *opts) {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  *sp = new structure(gv, eps_slab, pml(0.5));
  *fp = opts ? new fields(*sp, *opts) : new fields(*sp);
  gaussian_src_time src(0.3, 0.1);
  (*fp)->add_point_source(Ez, src, vec(0.11, 0.13));
}

static double phase_conductivity(const vec &) { return 0.2; }

static void test_resident_material_coefficient_preparation() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5));
  s.set_conductivity(Dz, phase_conductivity);
  fields f(&s);
  f.require_component(Ez);
  bool missing_before = false;
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine())
      missing_before |= f.chunks[i]->s->conductivity[Dz][Z] &&
                        !f.chunks[i]->s->condinv[Dz][Z];
  CHECK(or_to_all(missing_before),
        "resident coefficient fixture did not begin with a lazy conductivity inverse");

  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts, true);
  f.advance(1);
  bool all_catalogued = true, saw_catalogued = false;
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    structure_chunk &chunk = *f.chunks[i]->s;
    if (!chunk.conductivity[Dz][Z]) continue;
    saw_catalogued = true;
    const ArrayId id = f.array_catalog->find(
        StorageKey{i, int(array_kind::condinv), int(Dz), -1, int(Z)});
    all_catalogued &= chunk.condinv[Dz][Z] && !chunk.condinv_stale && is_valid(id) &&
                       f.array_catalog->resolve<realnum>(id) == chunk.condinv[Dz][Z];
  }
  const bool global_catalogued = and_to_all(all_catalogued);
  const bool global_saw_catalogued = or_to_all(saw_catalogued);
  bool all_curl_inverses = true, saw_conductive_curl = false;
  for (const CurlUpdate &update : f.step_plans[0]->db_updates)
    if (update.region.variant_key & curl_has_conductivity) {
      saw_conductive_curl = true;
      all_curl_inverses &= is_valid(update.condinv);
    }
  const bool global_curl_inverses = and_to_all(all_curl_inverses);
  const bool global_saw_conductive_curl = or_to_all(saw_conductive_curl);
  CHECK(global_catalogued && global_saw_catalogued && global_curl_inverses &&
            global_saw_conductive_curl,
        "resident preparation did not materialize and catalog conductivity inverse storage");
}

static void test_material_phase_transaction() {
  structure *current;
  fields *f;
  build(&current, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts, true);
  f->advance(1);
  BackendState *initial_state = f->backend_state;
  Executable *initial_executable = f->executable;

  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure target_a(gv, eps_slab, pml(0.5));
  structure target_b(gv, unit_epsilon, pml(0.5));
  target_a.set_conductivity(Dz, phase_conductivity);
  target_b.set_conductivity(Dz, phase_conductivity);
  std::vector<int> refs_a(size_t(target_a.num_chunks), 0);
  std::vector<int> refs_b(size_t(target_b.num_chunks), 0);
  for (int i = 0; i < target_a.num_chunks; ++i) {
    refs_a[size_t(i)] = target_a.chunks[i]->refcount;
    refs_b[size_t(i)] = target_b.chunks[i]->refcount;
  }

  const uint32_t initial_dirty = f->dirty_mask;
  const uint64_t initial_generation = generation(*f, MutationKind::material_phase);
  const uint64_t initial_local_generation = f->local_invalidation_generation;
  const int steps = f->phase_in_material(&target_a, 2.25 * f->dt);
  CHECK(steps == 2 && f->phasein_time == 2, "material phase did not preserve integer truncation");
  CHECK(!f->backend_state && !f->executable,
        "successful material setup retained the old resident representation");
  CHECK(counts.rebuilds == 1 && counts.states_destroyed == 1 &&
            counts.executables_destroyed == 1,
        "material setup did not migrate then retire exactly one resident representation");
  CHECK(initial_state != f->backend_state && initial_executable != f->executable,
        "material setup did not clear resident artifacts");
  CHECK(f->dirty_mask == (initial_dirty | invalidation_closure(MutationKind::material_phase)) &&
            generation(*f, MutationKind::material_phase) == initial_generation + 1 &&
            f->local_invalidation_generation == initial_local_generation + 1,
        "successful material setup did not commit the exact lifecycle closure once");
  for (int i = 0; i < f->num_chunks; ++i) {
    if (!f->chunks[i]->is_mine()) continue;
    CHECK(f->chunks[i]->new_s == target_a.chunks[i],
          "material setup did not install the requested target");
    CHECK(target_a.chunks[i]->refcount == refs_a[size_t(i)] + 1,
          "material setup did not retain exactly one target reference");
    CHECK(f->chunks[i]->s->refcount == 1,
          "material setup did not install detached current storage");
    CHECK(f->chunks[i]->s->conductivity[Dz][Z] && f->chunks[i]->s->condinv[Dz][Z],
          "material setup did not realize the stable conductivity union");
  }

  f->init_backend();
  BackendState *live_state = f->backend_state;
  Executable *live_executable = f->executable;
  std::vector<structure_chunk *> current_rows(size_t(f->num_chunks), NULL);
  for (int i = 0; i < f->num_chunks; ++i) current_rows[size_t(i)] = f->chunks[i]->s;
  const uint32_t dirty_before = f->dirty_mask;
  const uint64_t generation_before = generation(*f, MutationKind::material_phase);
  const int countdown_before = f->phasein_time;
  counts.fail_rebuild = my_rank() == 0;
  bool rejected = false;
  try {
    f->phase_in_material(&target_b, 3.5 * f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "rank-asymmetric material rebuild failure was not reconciled");
  CHECK(f->backend_state == live_state && f->executable == live_executable,
        "failed material setup retired the live resident representation");
  CHECK(f->dirty_mask == dirty_before &&
            generation(*f, MutationKind::material_phase) == generation_before &&
            f->phasein_time == countdown_before,
        "failed material setup changed lifecycle state or countdown");
  for (int i = 0; i < f->num_chunks; ++i) {
    CHECK(f->chunks[i]->s == current_rows[size_t(i)],
          "failed material setup rebound current structure storage");
    if (!f->chunks[i]->is_mine()) continue;
    CHECK(f->chunks[i]->new_s == target_a.chunks[i],
          "failed material setup replaced the prior target");
    CHECK(target_a.chunks[i]->refcount == refs_a[size_t(i)] + 1 &&
              target_b.chunks[i]->refcount == refs_b[size_t(i)],
          "failed material setup leaked staged target ownership");
  }

  counts.fail_rebuild = false;
  CHECK(f->phase_in_material(&target_b, 3.5 * f->dt) == 3,
        "material setup retry did not succeed");
  for (int i = 0; i < f->num_chunks; ++i) {
    if (!f->chunks[i]->is_mine()) continue;
    CHECK(f->chunks[i]->new_s == target_b.chunks[i],
          "material retry did not install replacement target");
    CHECK(target_a.chunks[i]->refcount == refs_a[size_t(i)] &&
              target_b.chunks[i]->refcount == refs_b[size_t(i)] + 1,
          "material target replacement has incorrect net ownership");
  }
  CHECK(f->phase_in_material(&target_b, 0.25 * f->dt) == 0 && !f->is_phasing(),
        "zero-step material phase no longer preserves legacy truncation");
  for (int i = 0; i < f->num_chunks; ++i)
    if (f->chunks[i]->is_mine())
      CHECK(target_b.chunks[i]->refcount == refs_b[size_t(i)] + 1,
            "same-target material setup changed net ownership");

  structure incompatible(vol2d(3.25, 3.0, 10.0), unit_epsilon, pml(0.5));
  const int stable_countdown = f->phasein_time;
  const uint32_t topology_dirty = f->dirty_mask;
  const uint64_t topology_generation = generation(*f, MutationKind::material_phase);
  const uint64_t topology_local_generation = f->local_invalidation_generation;
  const bool topology_changed_materials = legacy_material_change_pending(*f);
  rejected = false;
  try {
    f->phase_in_material(&incompatible, f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "incompatible material target was not rejected collectively");
  CHECK(f->phasein_time == stable_countdown && f->dirty_mask == topology_dirty &&
            generation(*f, MutationKind::material_phase) == topology_generation &&
            f->local_invalidation_generation == topology_local_generation &&
            legacy_material_change_pending(*f) == topology_changed_materials,
        "incompatible target changed setup or lifecycle state");
  for (int i = 0; i < f->num_chunks; ++i)
    if (f->chunks[i]->is_mine())
      CHECK(f->chunks[i]->new_s == target_b.chunks[i] &&
                target_b.chunks[i]->refcount == refs_b[size_t(i)] + 1,
            "incompatible target changed target ownership");

  structure chunk_mismatch(gv, unit_epsilon, pml(0.5));
  if (my_rank() == 0 && chunk_mismatch.num_chunks)
    chunk_mismatch.chunks[0]->a += 1.0;
  rejected = false;
  const uint32_t chunk_dirty = f->dirty_mask;
  const uint64_t chunk_generation = generation(*f, MutationKind::material_phase);
  const uint64_t chunk_local_generation = f->local_invalidation_generation;
  const bool chunk_changed_materials = legacy_material_change_pending(*f);
  try {
    f->phase_in_material(&chunk_mismatch, f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "rank-asymmetric chunk-topology mismatch was not rejected collectively");
  CHECK(f->phasein_time == stable_countdown && f->dirty_mask == chunk_dirty &&
            generation(*f, MutationKind::material_phase) == chunk_generation &&
            f->local_invalidation_generation == chunk_local_generation &&
            legacy_material_change_pending(*f) == chunk_changed_materials,
        "chunk-topology rejection changed setup or lifecycle state");

  f->init_backend();
  live_state = f->backend_state;
  live_executable = f->executable;
  structure allocation_failure_target(gv, eps_slab, pml(0.5));
  for (int i = 0; i < allocation_failure_target.num_chunks; ++i) {
    if (!allocation_failure_target.chunks[i]->is_mine()) continue;
    structure_chunk &target = *allocation_failure_target.chunks[i];
    const size_t n = size_t(target.gv.ntot());
    delete[] target.chi1inv[Ex][Y];
    target.chi1inv[Ex][Y] = new realnum[n];
    std::fill(target.chi1inv[Ex][Y], target.chi1inv[Ex][Y] + n, realnum(0.25));
    target.trivial_chi1inv[Ex][Y] = false;
  }
  int owns_chunk = 0;
  for (int i = 0; i < f->num_chunks; ++i) owns_chunk |= f->chunks[i]->is_mine();
  const int failure_rank = min_to_all(owns_chunk ? my_rank() : count_processors());
  set_material_phase_prepare_failure_for_testing(failure_rank, 1);
  const uint32_t allocation_dirty = f->dirty_mask;
  const uint64_t allocation_generation = generation(*f, MutationKind::material_phase);
  const uint64_t allocation_local_generation = f->local_invalidation_generation;
  const bool allocation_changed_materials = legacy_material_change_pending(*f);
  const int allocation_countdown = f->phasein_time;
  std::vector<structure_chunk *> allocation_current(size_t(f->num_chunks), NULL);
  std::vector<int> allocation_target_refs(size_t(f->num_chunks), 0);
  for (int i = 0; i < f->num_chunks; ++i) {
    allocation_current[size_t(i)] = f->chunks[i]->s;
    allocation_target_refs[size_t(i)] = allocation_failure_target.chunks[i]->refcount;
  }
  rejected = false;
  try {
    f->phase_in_material(&allocation_failure_target, 2 * f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  set_material_phase_prepare_failure_for_testing(-1, -1);
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "rank-asymmetric material storage failure was not reconciled");
  CHECK(f->backend_state == live_state && f->executable == live_executable &&
            f->dirty_mask == allocation_dirty &&
            generation(*f, MutationKind::material_phase) == allocation_generation &&
            f->local_invalidation_generation == allocation_local_generation &&
            legacy_material_change_pending(*f) == allocation_changed_materials &&
            f->phasein_time == allocation_countdown,
        "failed material storage preparation changed resident or lifecycle state");
  for (int i = 0; i < f->num_chunks; ++i) {
    CHECK(f->chunks[i]->s == allocation_current[size_t(i)],
          "failed material storage preparation changed current storage");
    CHECK(allocation_failure_target.chunks[i]->refcount ==
              allocation_target_refs[size_t(i)],
          "failed material storage preparation leaked a target reference");
    if (f->chunks[i]->is_mine())
      CHECK(f->chunks[i]->new_s == target_b.chunks[i],
            "failed material storage preparation changed the attached target");
  }

  delete f;
  delete current;

  {
    using namespace meep_geom;
    geometric_object_list empty_geometry = {0, NULL};
    material_type current_material = make_dielectric(2.0);
    material_type target_material = make_dielectric(3.0);
    structure geometry_current(gv, unit_epsilon, no_pml(), identity(), 2);
    structure geometry_target(gv, unit_epsilon, no_pml(), identity(), 2);
    set_materials_from_geometry(&geometry_current, empty_geometry, make_vector3(), false, 1e-5,
                                128, false, current_material);
    set_materials_from_geometry(&geometry_target, empty_geometry, make_vector3(), false, 1e-5,
                                128, false, target_material);
    fields geometry_fields(&geometry_current);
    lifetime_counts geometry_counts;
    geometry_fields.backend = new tracking_backend(geometry_fields, geometry_counts, true);
    geometry_fields.advance(1);
    const std::shared_ptr<const void> original_ir = geometry_fields.material_ir;
    CHECK(original_ir && geometry_target.material_ir,
          "geometry-backed material phase fixture has no immutable IR");
    geometry_counts.fail_rebuild = true;
    rejected = false;
    try { geometry_fields.phase_in_material(&geometry_target, geometry_fields.dt); }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(and_to_all(rejected) && geometry_fields.material_ir == original_ir,
          "failed geometry-backed material phase discarded the committed IR");
    geometry_counts.fail_rebuild = false;
    CHECK(geometry_fields.phase_in_material(&geometry_target, geometry_fields.dt) == 1 &&
              !geometry_fields.material_ir,
          "successful geometry-backed material phase retained a stale source IR");
    geometry_fields.init_backend();
    CHECK(geometry_fields.initialization_plan &&
              geometry_fields.initialization_plan->materials.size() == 1 &&
              geometry_fields.initialization_plan->materials[0].disposition() ==
                  MaterialRecipeDisposition::host_reference &&
              geometry_fields.initialization_plan->materials[0].support_reason_bits() ==
                  material_support_no_owned_ir,
          "material phase retry did not rebuild a dense host-reference recipe");
    material_free(target_material);
    material_free(current_material);
  }

  grid_volume cyl_gv = volcyl(2.0, 3.0, 10.0);
  structure cyl_current(cyl_gv, unit_epsilon, no_pml());
  structure cyl_target(cyl_gv, eps_slab, no_pml());
  fields cyl(&cyl_current, 1.0);
  CHECK(cyl.phase_in_material(&cyl_target, cyl.dt) == 1,
        "valid cylindrical target was rejected after symmetry augmentation");

  build(&current, &f);
  lifetime_counts magnetic_counts;
  f->backend = new tracking_backend(*f, magnetic_counts, true);
  f->advance(1);
  structure magnetic_target(gv, eps_slab, pml(0.5));
  std::vector<int> magnetic_refs(size_t(magnetic_target.num_chunks), 0);
  for (int i = 0; i < magnetic_target.num_chunks; ++i)
    magnetic_refs[size_t(i)] = magnetic_target.chunks[i]->refcount;
  f->synchronize_magnetic_fields();
  f->synchronize_magnetic_fields();
  live_state = f->backend_state;
  live_executable = f->executable;
  const uint32_t magnetic_dirty = f->dirty_mask;
  const uint64_t magnetic_generation = generation(*f, MutationKind::material_phase);
  const uint64_t magnetic_local_generation = f->local_invalidation_generation;
  const bool magnetic_changed_materials = legacy_material_change_pending(*f);
  const int magnetic_countdown = f->phasein_time;
  rejected = false;
  try {
    f->phase_in_material(&magnetic_target, f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "live nested magnetic snapshot did not reject material setup collectively");
  CHECK(f->backend_state == live_state && f->executable == live_executable &&
            f->dirty_mask == magnetic_dirty &&
            generation(*f, MutationKind::material_phase) == magnetic_generation &&
            f->local_invalidation_generation == magnetic_local_generation &&
            legacy_material_change_pending(*f) == magnetic_changed_materials &&
            f->phasein_time == magnetic_countdown,
        "snapshot-rejected material setup changed resident or lifecycle state");
  for (int i = 0; i < f->num_chunks; ++i)
    if (f->chunks[i]->is_mine())
      CHECK(!f->chunks[i]->new_s &&
                magnetic_target.chunks[i]->refcount == magnetic_refs[size_t(i)],
            "snapshot-rejected material setup retained its target");
  f->restore_magnetic_fields();
  f->restore_magnetic_fields();
  CHECK(magnetic_counts.magnetic_restores == 1,
        "material rejection stranded the nested magnetic snapshot");

  f->backend->poison();
  const uint32_t poison_dirty = f->dirty_mask;
  const uint64_t poison_generation = generation(*f, MutationKind::material_phase);
  const uint64_t poison_local_generation = f->local_invalidation_generation;
  const bool poison_changed_materials = legacy_material_change_pending(*f);
  const int poison_countdown = f->phasein_time;
  rejected = false;
  try {
    f->phase_in_material(&magnetic_target, f->dt);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "poisoned backend accepted material setup");
  CHECK(f->backend_state == live_state && f->executable == live_executable &&
            f->dirty_mask == poison_dirty &&
            generation(*f, MutationKind::material_phase) == poison_generation &&
            f->local_invalidation_generation == poison_local_generation &&
            legacy_material_change_pending(*f) == poison_changed_materials &&
            f->phasein_time == poison_countdown,
        "poison rejection changed resident or lifecycle state");
  for (int i = 0; i < f->num_chunks; ++i)
    if (f->chunks[i]->is_mine())
      CHECK(!f->chunks[i]->new_s &&
                magnetic_target.chunks[i]->refcount == magnetic_refs[size_t(i)],
            "poison rejection changed target ownership");
  delete f;
  delete current;
}

static void test_material_phase_cpu_to_resident_preparation() {
  const grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure current(gv, eps_slab, pml(0.5));
  structure target(gv, unit_epsilon, pml(0.5));
  target.set_conductivity(Dz, phase_conductivity);
  fields f(&current);
  CHECK(f.phase_in_material(&target, 2.0 * f.dt) == 2,
        "CPU material setup did not retain its countdown");

  std::vector<structure_chunk *> before(size_t(f.num_chunks), NULL);
  std::vector<int> target_refs(size_t(target.num_chunks), 0);
  for (int i = 0; i < f.num_chunks; ++i) {
    before[size_t(i)] = f.chunks[i]->s;
    target_refs[size_t(i)] = target.chunks[i]->refcount;
    if (f.chunks[i]->is_mine())
      CHECK(!f.chunks[i]->s->conductivity[Dz][Z],
            "CPU-only material setup eagerly realized resident storage");
  }
  {
    fields copy(f);
    for (int i = 0; i < copy.num_chunks; ++i) {
      FOR_FIELD_TYPES(ft) CHECK(!copy.chunks[i]->pol[ft],
                                "zero-node fields copy retained a polarization list");
      DOCMP2 FOR_COMPONENTS(c) {
        CHECK(!f.chunks[i]->f_minus_p[c][cmp] || copy.chunks[i]->f_minus_p[c][cmp],
              "fields copy lost an allocated polarization-subtraction row");
        CHECK(!f.chunks[i]->f_w_prev[c][cmp] || copy.chunks[i]->f_w_prev[c][cmp],
              "fields copy lost an allocated previous-W row");
        CHECK(f.chunks[i]->f_minus_p[c][cmp] || !copy.chunks[i]->f_minus_p[c][cmp],
              "fields copy retained an uninitialized polarization-subtraction pointer");
        CHECK(f.chunks[i]->f_w_prev[c][cmp] || !copy.chunks[i]->f_w_prev[c][cmp],
              "fields copy retained an uninitialized previous-W pointer");
      }
    }
  }

  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  set_material_phase_prepare_failure_for_testing(0, 1);
  bool rejected = false;
  try {
    f.init_backend();
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  set_material_phase_prepare_failure_for_testing(-1, -1);
  CHECK(sum_to_all(int(rejected)) == count_processors(),
        "rank-asymmetric resident material preparation was not rejected collectively");
  CHECK(!f.backend_state && f.phasein_time == 2,
        "failed resident material preparation changed state or countdown");
  for (int i = 0; i < f.num_chunks; ++i) {
    CHECK(f.chunks[i]->s == before[size_t(i)],
          "failed resident material preparation committed a current chunk");
    if (f.chunks[i]->is_mine())
      CHECK(f.chunks[i]->new_s == target.chunks[i] &&
                target.chunks[i]->refcount == target_refs[size_t(i)],
            "failed resident material preparation changed target ownership");
  }

  f.init_backend();
  CHECK(f.backend_state && f.phasein_time == 2,
        "resident material preparation retry did not create state");
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine()) {
      CHECK(f.chunks[i]->s != before[size_t(i)] && f.chunks[i]->s->refcount == 1,
            "resident material preparation did not detach current storage");
      CHECK(f.chunks[i]->s->conductivity[Dz][Z] && f.chunks[i]->s->condinv[Dz][Z],
            "resident material preparation did not realize the target storage union");
      CHECK(f.chunks[i]->new_s == target.chunks[i] &&
                target.chunks[i]->refcount == target_refs[size_t(i)],
            "resident material preparation retry changed target ownership");
    }
}

static void test_material_phase_early_fallback_preflight() {
  const grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure current(gv, unit_epsilon, pml(0.5));
  structure target(gv, eps_slab, pml(0.5));
  target.set_conductivity(Dz, phase_conductivity);
  fields f(&current);
  CHECK(f.phase_in_material(&target, 2.0 * f.dt) == 2,
        "early material preflight fixture did not retain its phase countdown");
  std::vector<structure_chunk *> current_rows(size_t(f.num_chunks), NULL);
  std::vector<int> target_refs(size_t(target.num_chunks), 0);
  for (int i = 0; i < f.num_chunks; ++i) {
    current_rows[size_t(i)] = f.chunks[i]->s;
    target_refs[size_t(i)] = target.chunks[i]->refcount;
  }

  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts, false, false, false, false, true);
  f.options.strict = true;
  f.options.fallback = fallback_policy::error;
  bool rejected = false;
  try { f.init_backend(); }
  catch (const std::runtime_error &) { rejected = true; }
  CHECK(and_to_all(rejected) && !f.backend_state && !f.executable &&
            counts.states_created == 0 && counts.rebuilds == 0 && counts.initialized == 0,
        "strict phased fallback rejection performed resident allocation or migration");
  for (int i = 0; i < f.num_chunks; ++i) {
    CHECK(f.chunks[i]->s == current_rows[size_t(i)] &&
              target.chunks[i]->refcount == target_refs[size_t(i)],
          "strict phased fallback rejection mutated material ownership");
    if (f.chunks[i]->is_mine())
      CHECK(!f.chunks[i]->s->conductivity[Dz][Z] && !f.chunks[i]->s->condinv[Dz][Z],
            "strict phased fallback rejection allocated material-phase coefficients");
  }

  if (count_processors() > 1) {
    using namespace meep_geom;
    geometric_object_list empty_geometry = {0, NULL};
    material_type ir_material = make_dielectric(2.25);
    structure ir_source(gv, unit_epsilon, no_pml(), identity(), 2);
    set_materials_from_geometry(&ir_source, empty_geometry, make_vector3(), false, 1e-5, 64,
                                false, ir_material);
    material_free(ir_material);
    CHECK(ir_source.material_ir, "semantic mismatch fixture has no owned material IR");
    std::shared_ptr<MaterialIR> local_ir(
        new MaterialIR(*static_cast<const MaterialIR *>(ir_source.material_ir.get())));
    if (my_rank() == 0) {
      local_ir->subpixel_tol *= 2.0;
      refresh_material_ir_signatures_for_testing(*local_ir);
    }
    f.material_ir = std::static_pointer_cast<const void>(local_ir);
    f.options.strict = false;
    f.options.fallback = fallback_policy::warn;
    rejected = false;
    try { f.init_backend(); }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(and_to_all(rejected) && !f.backend_state && !f.executable &&
              counts.states_created == 0 && counts.rebuilds == 0 && counts.initialized == 0,
          "phased semantic-signature mismatch reached allocation or migration");
    for (int i = 0; i < f.num_chunks; ++i)
      CHECK(f.chunks[i]->s == current_rows[size_t(i)] &&
                target.chunks[i]->refcount == target_refs[size_t(i)],
            "phased semantic-signature mismatch mutated material ownership");
    f.material_ir.reset();
  }

  f.options.strict = false;
  f.options.fallback = fallback_policy::warn;
  f.init_backend();
  CHECK(f.backend_state && f.executable && f.phasein_time == 2,
        "early material preflight rejection was not retryable");
}

static std::complex<double> multiply_fields(const std::complex<realnum> *values, const vec &,
                                            void *) {
  return values[0] * values[1];
}

#ifdef HAVE_HDF5
static std::unique_ptr<binary_partition> uneven_repeated_partition() {
  std::unique_ptr<binary_partition> tree(new binary_partition(3));
  tree.reset(new binary_partition(split_plane{X, 0.9},
                                  std::unique_ptr<binary_partition>(new binary_partition(2)),
                                  std::move(tree)));
  tree.reset(new binary_partition(split_plane{X, 0.3},
                                  std::unique_ptr<binary_partition>(new binary_partition(1)),
                                  std::move(tree)));
  tree.reset(new binary_partition(split_plane{X, -0.3},
                                  std::unique_ptr<binary_partition>(new binary_partition(0)),
                                  std::move(tree)));
  return std::unique_ptr<binary_partition>(new binary_partition(
      split_plane{X, -0.9}, std::unique_ptr<binary_partition>(new binary_partition(0)),
      std::move(tree)));
}

static void test_sharded_dft_checkpoint_ordering() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  std::unique_ptr<binary_partition> partition = uneven_repeated_partition();
  structure *s = new structure(gv, eps_slab, boundary_region(), identity(), 0, 0.5, false,
                               DEFAULT_SUBPIXEL_TOL, DEFAULT_SUBPIXEL_MAXEVAL, partition.get());
  fields *f = new fields(s);
  gaussian_src_time src(0.3, 0.1);
  f->add_point_source(Ez, src, vec(0.11, 0.13));
  f->require_component(Ez);
  component component_ez = Ez;
  dft_fields monitor = f->add_dft_fields(&component_ez, 1, f->v, 0.3, 0.3, 1,
                                         /*use_centered_grid=*/true,
                                         /*decimation_factor=*/1,
                                         /*persist=*/true);

  rebuild_trace rebuilds;
  access_trace accesses;
  f->backend = new access_tracking_backend(*f, rebuilds, accesses);
  f->advance(2);

  int owned_chunks = 0;
  for (int i = 0; i < f->num_chunks; ++i)
    owned_chunks += f->chunks[i]->is_mine();
  if (count_processors() > 1)
    CHECK(min_to_all(owned_chunks) < max_to_all(owned_chunks),
          "custom checkpoint partition did not produce uneven ownership");
  const int dft_failure_rank = min_to_all(monitor.chunks ? my_rank() : count_processors());
  CHECK(dft_failure_rank < count_processors(), "custom checkpoint monitor has no DFT owner");

  char filename[160];
  snprintf(filename, sizeof(filename), "/tmp/meep-sharded-dft-%ld-%d.h5", long(getpid()),
           my_rank());

  accesses.fail_dft_read_rank = dft_failure_rank;
  bool dump_failure = false;
  try {
    f->dump(filename, false);
  }
  catch (const std::runtime_error &) {
    dump_failure = true;
  }
  const int dump_failures = sum_to_all(int(dump_failure));
  CHECK(dump_failures == count_processors(),
        "sharded DFT dump failure was not reconciled at its outer boundary (%d/%d)", dump_failures,
        count_processors());
  accesses.fail_dft_read_rank = -1;
  std::remove(filename);
  all_wait();

  f->dump(filename, false);
  CHECK(sum_to_all(int(accesses.dft_reads)) > 0,
        "sharded DFT dump did not refresh resident accumulators");

  accesses.fail_dft_write_rank = dft_failure_rank;
  bool load_failure = false;
  try {
    h5file file(filename, h5file::READONLY, false, true);
    std::string local_error;
    for (int i = 0; i < f->num_chunks; ++i)
      if (f->chunks[i]->is_mine()) {
        char dataname[32];
        snprintf(dataname, sizeof(dataname), "chunk%02d", i);
        load_dft_hdf5(f->chunks[i]->dft_chunks, dataname, &file, 0, false, &local_error);
      }
    backend_reconcile_host_access(local_error, "backend_api sharded DFT load");
  }
  catch (const std::runtime_error &) {
    load_failure = true;
  }
  const int load_failures = sum_to_all(int(load_failure));
  CHECK(load_failures == count_processors(),
        "sharded DFT load failure was not reconciled at its outer boundary (%d/%d)", load_failures,
        count_processors());
  accesses.fail_dft_write_rank = -1;

  {
    h5file file(filename, h5file::READONLY, false, true);
    std::string local_error;
    for (int i = 0; i < f->num_chunks; ++i)
      if (f->chunks[i]->is_mine()) {
        char dataname[32];
        snprintf(dataname, sizeof(dataname), "chunk%02d", i);
        load_dft_hdf5(f->chunks[i]->dft_chunks, dataname, &file, 0, false, &local_error);
      }
    backend_reconcile_host_access(local_error, "backend_api sharded DFT load recovery");
  }

  f->load(filename, false);
  const int restored_time = f->t;
  f->advance(1);
  CHECK(f->t == restored_time + 1, "execution did not continue after sharded DFT restore");
  int rank = 0;
  size_t dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> restored(f->get_dft_array(monitor, Ez, 0, &rank, dims));
  CHECK(restored.get() != NULL, "restored sharded DFT state could not be queried");

  std::remove(filename);
  monitor.remove();
  delete f;
  delete s;
}
#endif

/* Selection: cpu accepted, nvidia and non-native precision rejected -- and
   rejected identically on every rank, since a rank that accepted while its
   peers aborted would hang rather than fail. */
static void test_selection() {
  structure *s;
  fields *f;
  build(&s, &f);

  CpuBackend cpu(*f);
  std::string why;

  execution_options ok;
  CHECK(cpu.accepts(ok, why), "the cpu backend rejected its own default options: %s", why.c_str());

  execution_options gpu;
  gpu.backend = backend_kind::nvidia;
  CHECK(!cpu.accepts(gpu, why), "backend=nvidia was accepted");
  CHECK(why.find("nvidia") != std::string::npos, "the nvidia rejection does not say why: %s",
        why.c_str());

  for (int p = 1; p <= 2; ++p) {
    execution_options pr;
    pr.precision = p == 1 ? precision_policy_kind::mixed : precision_policy_kind::f32;
    CHECK(!cpu.accepts(pr, why), "precision=%s was accepted on cpu",
          precision_policy_name(pr.precision));
  }

  execution_options dev;
  dev.device_id = 0;
  CHECK(!cpu.accepts(dev, why), "device_id was accepted on cpu");

  const backend_capabilities c = cpu.capabilities();
  CHECK(c.supports_native, "the cpu backend must support native precision");
  CHECK(!c.supports_mixed && !c.supports_f32, "the cpu backend must not claim mixed or f32");
  CHECK(strcmp(c.name, "cpu") == 0, "capabilities name is %s", c.name);

  delete f;
  delete s;
}

/* The backend-selecting constructor must produce the same simulation as the
   plain one. */
static void test_construction_equivalence() {
  structure *s1, *s2;
  fields *f1, *f2;
  execution_options opts; // defaults: cpu, native
  build(&s1, &f1);
  build(&s2, &f2, &opts);

  f1->advance(9);
  f2->advance(9);

  CHECK(f1->t == f2->t, "t differs: %d vs %d", f1->t, f2->t);
  size_t compared = 0, bad = 0;
  for (int i = 0; i < f1->num_chunks; ++i) {
    if (!f1->chunks[i]->is_mine()) continue;
    const size_t ntot = size_t(f1->chunks[i]->gv.ntot());
    for (int c = 0; c < NUM_FIELD_COMPONENTS; ++c)
      for (int cmp = 0; cmp < 2; ++cmp) {
        const realnum *a = f1->chunks[i]->f[c][cmp];
        const realnum *b = f2->chunks[i]->f[c][cmp];
        if (!a || !b) continue;
        ++compared;
        if (memcmp(a, b, ntot * sizeof(realnum)) != 0) ++bad;
      }
  }
  CHECK(bad == 0, "%zu of %zu arrays differ between the two constructors", bad, compared);
  CHECK(compared > 0, "nothing was compared");
  delete f1;
  delete f2;
  delete s1;
  delete s2;
}

/* read/write must round-trip every registered array without loss under
   native. */
static void test_read_write_roundtrip() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->advance(5);

  CpuBackend cpu(*f);
  size_t checked = 0, bad = 0;
  for (size_t i = 0; i < f->array_catalog->size() && checked < 40; ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = f->array_catalog->spec(id);
    if (spec.role != array_role::field || spec.elements == 0) continue;
    std::vector<realnum> buf(spec.elements, realnum(0));
    ArrayRef ref{id, 0, spec.elements};
    cpu.read(ref, buf.data(), spec.elements * sizeof(realnum));
    std::vector<realnum> back(spec.elements, realnum(0));
    cpu.write(ref, buf.data(), spec.elements * sizeof(realnum));
    cpu.read(ref, back.data(), spec.elements * sizeof(realnum));
    if (memcmp(buf.data(), back.data(), spec.elements * sizeof(realnum)) != 0) ++bad;
    ++checked;
  }
  CHECK(bad == 0, "%zu of %zu arrays did not round-trip through read/write", bad, checked);
  CHECK(or_to_all(checked > 0), "no arrays were round-tripped");

  /* The backend API is typed by ArraySpec, not implicitly by realnum. Register
     one array of every representation and exercise a nonzero element offset. */
  realnum r[4] = {1, 2, 3, 4};
  std::complex<realnum> cr[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  double d[4] = {1, 2, 3, 4};
  std::complex<double> cd[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  int32_t i32[4] = {1, 2, 3, 4};
  size_t iz[4] = {1, 2, 3, 4};
  void *arrays[] = {r, cr, d, cd, i32, iz};
  const ElementType types[] = {ElementType::realnum_value, ElementType::complex_realnum,
                               ElementType::float64,       ElementType::complex_float64,
                               ElementType::int32,         ElementType::index};
  for (size_t k = 0; k < sizeof(types) / sizeof(types[0]); ++k) {
    const StorageKey key{-1, int(array_kind::num_kinds), -1, -1, int(k)};
    const ArrayId id =
        f->array_catalog->register_array(key, arrays[k], 4, array_role::scratch, types[k]);
    const size_t element_size = host_element_bytes(types[k]);
    std::vector<unsigned char> got(2 * element_size, 0);
    cpu.read(ArrayRef{id, 1, 2}, got.data(), got.size());
    CHECK(memcmp(got.data(), static_cast<unsigned char *>(arrays[k]) + element_size, got.size()) ==
              0,
          "typed access failed for ElementType %zu", k);
    std::vector<unsigned char> replacement(got.size(), 0x5a);
    cpu.write(ArrayRef{id, 1, 2}, replacement.data(), replacement.size());
    CHECK(memcmp(replacement.data(), static_cast<unsigned char *>(arrays[k]) + element_size,
                 replacement.size()) == 0,
          "typed write failed for ElementType %zu", k);
  }

  bool rejected = false;
  try {
    realnum one = 0;
    cpu.read(ArrayRef{invalid_array(), 0, 1}, &one, sizeof(one));
  }
  catch (const std::out_of_range &) {
    rejected = true;
  }
  CHECK(rejected, "an invalid ArrayId was not rejected");

  rejected = false;
  try {
    realnum one = 0;
    cpu.read(ArrayRef{ArrayId{0}, f->array_catalog->spec(ArrayId{0}).elements, 1}, &one,
             sizeof(one));
  }
  catch (const std::out_of_range &) {
    rejected = true;
  }
  CHECK(rejected, "an out-of-bounds ArrayRef was not rejected");

  rejected = false;
  try {
    realnum one = 0;
    cpu.read(ArrayRef{ArrayId{0}, 0, 1}, &one, sizeof(one) + 1);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected, "a mismatched host byte count was not rejected");
  delete f;
  delete s;
}

static void test_precision_policy() {
  CHECK(policy_for(precision_policy_kind::native) == precision_native(),
        "native policy does not match");
  const PrecisionPolicy mixed = policy_for(precision_policy_kind::mixed);
  CHECK(mixed.field == Precision::f32 && mixed.monitor == Precision::f64,
        "the mixed policy has the wrong shape");
  const PrecisionPolicy f32 = policy_for(precision_policy_kind::f32);
  CHECK(f32.monitor == Precision::f32 && f32.reduction == Precision::f64,
        "f32 must still reduce in f64");

  /* Aliased arrays (H == B) must agree on storage precision. */
  structure *s;
  fields *f;
  build(&s, &f);
  f->advance(3);
  std::string why;
  CHECK(validate_alias_precisions(*f->array_catalog, why), "alias precision validation failed: %s",
        why.c_str());
  delete f;
  delete s;

  StoragePlan plan;
  plan.arrays.push_back(ArraySpec{ArrayId{0}, array_role::field, ElementType::realnum_value,
                                  native_precision, 10, alignof(realnum), invalid_array(), false,
                                  false});
  plan.arrays.push_back(ArraySpec{ArrayId{1}, array_role::field, ElementType::realnum_value,
                                  native_precision, 10, alignof(realnum), ArrayId{0}, false,
                                  false});
  plan.arrays.push_back(ArraySpec{ArrayId{2}, array_role::material, ElementType::realnum_value,
                                  native_precision, 5, alignof(realnum), invalid_array(), true,
                                  false});
  plan.arrays.push_back(ArraySpec{ArrayId{3}, array_role::dft, ElementType::complex_realnum,
                                  native_precision, 2, alignof(realnum), invalid_array(), false,
                                  false});
  plan.arrays.push_back(ArraySpec{ArrayId{4}, array_role::field, ElementType::float64,
                                  native_precision, 3, alignof(double), invalid_array(), false,
                                  false});
  plan.arrays.push_back(ArraySpec{ArrayId{5}, array_role::scratch, ElementType::int32,
                                  native_precision, 4, alignof(int32_t), invalid_array(), false,
                                  false});
  apply_precision_policy(plan, precision_mixed());
  CHECK(plan.arrays[0].storage == Precision::f32 && plan.arrays[1].storage == Precision::f32,
        "mixed policy did not apply to field/alias storage");
  CHECK(plan.arrays[2].storage == Precision::f32, "mixed policy did not apply to material storage");
  CHECK(plan.arrays[3].storage == Precision::f64,
        "mixed policy did not preserve monitor precision");
  CHECK(plan.arrays[4].storage == Precision::f64, "fixed float64 storage was narrowed");
  CHECK(plan.provisional_peak_bytes() == 132, "precision-aware peak bytes are %zu, expected 132",
        plan.provisional_peak_bytes());
  CHECK(plan.steady_state_bytes() == 112, "precision-aware steady bytes are %zu, expected 112",
        plan.steady_state_bytes());
  CHECK(validate_alias_precisions(plan, why), "valid plan aliases were rejected: %s", why.c_str());
  plan.arrays[1].storage = Precision::f64;
  CHECK(!validate_alias_precisions(plan, why), "mismatched plan alias precision was accepted");

  ArraySpec huge = {ArrayId{0},
                    array_role::dft,
                    ElementType::complex_float64,
                    Precision::f64,
                    std::numeric_limits<size_t>::max(),
                    alignof(double),
                    invalid_array(),
                    false,
                    false};
  bool overflow_rejected = false;
  try {
    (void)storage_bytes(huge);
  }
  catch (const std::overflow_error &) {
    overflow_rejected = true;
  }
  CHECK(overflow_rejected, "overflowing storage byte count was accepted");
}

static void test_dft_access_boundaries() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->require_component(Ez);
  component c = Ez;
  dft_fields monitor = f->add_dft_fields(&c, 1, f->v, 0.3, 0.3, 1);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);

  const int reads_before = counts.reads;
  const int writes_before = counts.writes;
  if (monitor.chunks) monitor.scale_dfts(2.0);
  CHECK(or_to_all(counts.reads > reads_before),
        "DFT host mutation did not read resident accumulator data");
  CHECK(or_to_all(counts.writes > writes_before),
        "DFT host mutation did not publish the accumulator back to the backend");

  monitor.remove();
  CHECK(!f->backend_state, "DFT removal left resident state referencing freed storage");
  CHECK(counts.rebuilds == 1, "DFT removal did not migrate resident state before deletion");
  delete f;
  delete s;
}

static void test_detached_dft_access() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->require_component(Ez);
  component c = Ez;
  dft_fields monitor = f->add_dft_fields(&c, 1, f->v, 0.3, 0.3, 1,
                                         /*use_centered_grid=*/true,
                                         /*decimation_factor=*/1,
                                         /*persist=*/true);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);

  f->clear_dft_monitors();
  CHECK(!f->backend_state, "clearing DFT monitors left resident state alive");
  CHECK(counts.rebuilds == 1, "clearing DFT monitors did not migrate resident state");
  f->advance(1);
  BackendState *rebuilt = f->backend_state;

  /* Detached persistent accumulators are now host-authoritative and absent
     from the rebuilt catalog. Queries/mutations must not try to resolve them. */
  if (monitor.chunks) {
    monitor.chunks->sync_dft_to_host();
    monitor.scale_dfts(2.0);
  }
  monitor.remove();
  CHECK(f->backend_state == rebuilt,
        "removing an already detached DFT monitor retired unrelated backend state");

  delete f;
  delete s;
}

static void test_backend_lifecycle_epoch() {
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);

  f->advance(1);
  BackendState *first_state = f->backend_state;
  CHECK(counts.arrays_at_create > 0, "resident backend state was created from an empty plan");
  CHECK(counts.connections_current_at_create,
        "resident backend state was created before halo topology was finalized");
  CHECK(counts.states_created == 1 && counts.initialized == 1 && counts.classified == 1 &&
            counts.finalized == 1 && counts.executables_created == 1 && counts.advanced == 1,
        "initial resident lifecycle counts are wrong");
  CHECK(!is_dirty(*f, dirty_initialization) && !is_dirty(*f, dirty_classification) &&
            !is_dirty(*f, dirty_executable),
        "resident lifecycle left consumed dirty bits set");

  invalidate(*f, MutationKind::material_values);
  f->advance(1);
  CHECK(f->backend_state != first_state, "material refresh did not replace resident state");
  CHECK(counts.states_created == 2 && counts.states_destroyed == 1 && counts.initialized == 2 &&
            counts.classified == 2 && counts.finalized == 2 &&
            counts.executables_created == 2 && counts.executables_destroyed == 1,
        "material refresh did not commit one complete replacement epoch");
  first_state = f->backend_state;

  f->zero_fields();
  f->advance(1);
  CHECK(f->backend_state == first_state && counts.states_created == 2 && counts.initialized == 3,
        "field-value refresh did not preserve and reinitialize resident state");
  CHECK(counts.classified == 2 && counts.finalized == 2 && counts.executables_created == 2,
        "field-value refresh rebuilt unrelated backend artifacts");

  f->initialize_field(Ez, initial_ez);
  CHECK(is_dirty(*f, dirty_initialization), "initialize_field did not invalidate resident values");
  f->advance(1);
  CHECK(f->backend_state == first_state && counts.states_created == 2 && counts.initialized == 4,
        "initialize_field refresh did not preserve and reinitialize resident state");

  invalidate(*f, MutationKind::field_layout);
  f->advance(1);
  CHECK(counts.states_created == 3 && counts.states_destroyed == 2,
        "storage invalidation did not replace resident state");
  CHECK(counts.executables_created == 3 && counts.executables_destroyed == 2,
        "executable invalidation did not replace the compiled artifact");

  delete f;
  CHECK(counts.states_destroyed == 3 && counts.executables_destroyed == 3,
        "polymorphic backend artifacts were not destroyed exactly once");
  delete s;
}

static uint64_t non_flux_step_plan_signature(const StepPlan &source) {
  StepPlan plan = source;
  plan.legacy_flux_updates.clear();
  plan.legacy_flux_terms.clear();
  std::vector<Operation> operations;
  operations.reserve(plan.operations.size());
  for (const Operation &op : plan.operations)
    if (op.kind != OpKind::update_flux_half && op.kind != OpKind::update_flux)
      operations.push_back(op);
  plan.operations.swap(operations);
  plan.signature = compute_step_plan_signature(plan);
  return plan.signature;
}

static void test_resident_legacy_flux_lifecycle() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->require_component(Ey);
  f->require_component(Hy);
  f->require_component(Hz);
  flux_vol *older = f->add_flux_vol(X, volume(vec(0.1, -0.8), vec(0.1, 0.8)));
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);

  CHECK(f->backend_state && f->executable && f->descriptors->legacy_fluxes.size() == 1 &&
            counts.legacy_flux_updates_at_compile == 1,
        "initial resident flux setup did not publish one compiled recipe");
  CHECK(or_to_all(counts.legacy_flux_terms_at_compile > 0),
        "resident flux setup produced no integration terms on any rank");
  CHECK(counts.legacy_flux_half_accesses_at_compile ==
            counts.legacy_flux_final_accesses_at_compile,
        "legacy flux markers do not have identical access sets");

  BackendState *const state = f->backend_state;
  Executable *const old_executable = f->executable;
  DescriptorSet *const old_descriptors = f->descriptors;
  StepPlan *const old_ordinary = f->step_plans[0];
  StepPlan *const old_cw = f->step_plans[1];
  const uint64_t stable_nonflux_signature = non_flux_step_plan_signature(*old_ordinary);
  flux_vol *newer = f->add_flux_vol(Y, volume(vec(-0.8, -0.2), vec(0.8, -0.2)));
  f->descriptors->regions.push_back(ChunkLoopRegion());
  const BackendEpochSnapshot failed_entry(*f);
  CpuArrayCatalog *const live_catalog = f->array_catalog;
  if (my_rank() == 0) f->array_catalog = NULL;
  bool failed = false;
  try {
    f->advance(1);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  if (my_rank() == 0) f->array_catalog = live_catalog;
  CHECK(sum_to_all(int(failed)) == count_processors() && failed_entry.matches(*f) &&
            f->backend_state == state && f->executable == old_executable &&
            is_dirty(*f, dirty_flux_plan),
        "rank-asymmetric missing legacy flux catalog was not rejected collectively");

  backend_set_legacy_flux_descriptor_failure_for_testing(0, 0);
  failed = false;
  try {
    f->advance(1);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  backend_set_legacy_flux_descriptor_failure_for_testing(-1, -1);
  CHECK(sum_to_all(int(failed)) == count_processors() && failed_entry.matches(*f) &&
            f->backend_state == state && f->executable == old_executable &&
            f->descriptors == old_descriptors && f->step_plans[0] == old_ordinary &&
            f->step_plans[1] == old_cw && is_dirty(*f, dirty_flux_plan),
        "per-monitor legacy flux failure was not reconciled before the next monitor");

  counts.fail_compile = my_rank() == 0;
  failed = false;
  try {
    f->advance(1);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  counts.fail_compile = false;
  CHECK(sum_to_all(int(failed)) == count_processors(),
        "rank-asymmetric legacy flux compile failure was not reconciled");
  CHECK(failed_entry.matches(*f) && f->backend_state == state && f->executable == old_executable &&
            f->descriptors == old_descriptors && f->step_plans[0] == old_ordinary &&
            f->step_plans[1] == old_cw && is_dirty(*f, dirty_flux_plan) &&
            is_dirty(*f, dirty_regions) && is_dirty(*f, dirty_executable),
        "failed legacy flux refresh partially published its resident epoch");
  CHECK(older->flux() == 0.0 && newer->flux() == 0.0,
        "failed legacy flux refresh changed a public scalar");

  const int created_before_retry = counts.executables_created;
  const int destroyed_before_retry = counts.executables_destroyed;
  f->advance(1);
  CHECK(f->backend_state == state && f->executable != old_executable &&
            f->descriptors != old_descriptors && f->step_plans[0] != old_ordinary &&
            f->step_plans[1] == old_cw &&
            counts.states_created == 1 &&
            counts.states_destroyed == 0 &&
            counts.executables_created == created_before_retry + 1 &&
            counts.executables_destroyed == destroyed_before_retry + 1 &&
            counts.legacy_flux_updates_at_compile == 2 &&
            non_flux_step_plan_signature(*f->step_plans[0]) == stable_nonflux_signature &&
            !is_dirty(*f, dirty_flux_plan) && !is_dirty(*f, dirty_regions) &&
            !is_dirty(*f, dirty_executable),
        "successful legacy flux refresh changed non-flux work or did not atomically replace code");

  const double values[2] = {1.25, -2.5};
  backend_publish_legacy_flux(*f, values, 2, "legacy flux publication test");
  CHECK(newer->flux() == values[0] && older->flux() == values[1],
        "legacy flux publication did not follow newest-first checked ordinals");
  const double before_bad[2] = {newer->flux(), older->flux()};
  failed = false;
  try {
    backend_publish_legacy_flux(*f, values, 1, "legacy flux bad-count test");
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors() && newer->flux() == before_bad[0] &&
            older->flux() == before_bad[1],
        "bad legacy flux publication count partially changed public scalars");

  const uint32_t saved_ordinal = f->descriptors->legacy_fluxes[0].flux_ordinal;
  if (my_rank() == 0) ++f->descriptors->legacy_fluxes[0].flux_ordinal;
  failed = false;
  try {
    backend_publish_legacy_flux(*f, values, 2, "legacy flux bad-ordinal test");
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors() && newer->flux() == before_bad[0] &&
            older->flux() == before_bad[1],
        "bad legacy flux ordinal partially changed public scalars");
  f->descriptors->legacy_fluxes[0].flux_ordinal = saved_ordinal;

  /* A legacy-flux mutation may be composed with an unrelated structural
     invalidation.  That combination must take the conservative staged-epoch
     route: the small descriptor/executable replacement must neither consume
     the other dirty bits nor hide its required state replacement. */
  flux_vol *composed =
      f->add_flux_vol(X, volume(vec(-0.4, -0.5), vec(-0.4, 0.5)));
  invalidate(*f, MutationKind::boundary_topology);
  BackendState *const composition_old_state = f->backend_state;
  Executable *const composition_old_executable = f->executable;
  const BackendEpochSnapshot composition_entry(*f);
  counts.fail_compile = my_rank() == 0;
  failed = false;
  try {
    f->advance(1);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  counts.fail_compile = false;
  CHECK(sum_to_all(int(failed)) == count_processors() && composition_entry.matches(*f) &&
            f->backend_state == composition_old_state &&
            f->executable == composition_old_executable && is_dirty(*f, dirty_flux_plan) &&
            is_dirty(*f, dirty_halos) && is_dirty(*f, dirty_executable) &&
            composed->flux() == 0.0,
        "failed mixed legacy-flux epoch partially consumed structural invalidation");
  f->advance(1);
  BackendState *const composition_state = f->backend_state;
  CHECK(composition_state != composition_old_state && f->executable != composition_old_executable &&
            f->descriptors->legacy_fluxes.size() == 3 && !is_dirty(*f, dirty_flux_plan) &&
            !is_dirty(*f, dirty_halos) && !is_dirty(*f, dirty_executable),
        "mixed legacy-flux invalidation did not replace the full resident epoch");

  gaussian_src_time composed_source(0.23, 0.1);
  composed_source.is_integrated = false;
  f->add_point_source(Ey, composed_source, vec(0.0, 0.0));
  f->add_flux_vol(X, volume(vec(0.35, -0.4), vec(0.35, 0.4)));
  f->advance(1);
  const bool source_present = or_to_all(!f->descriptors->sources.sources.empty());
  CHECK(f->backend_state != composition_state && source_present &&
            f->descriptors->legacy_fluxes.size() == 4 && !is_dirty(*f, dirty_source_plan) &&
            !is_dirty(*f, dirty_flux_plan) && !is_dirty(*f, dirty_executable) &&
            f->step_plans[0]->source_signature ==
                source_plan_signature(f->descriptors->sources),
        "same-advance source and legacy-flux refresh did not publish one complete executable");

  component dft_component = Ez;
  dft_fields dft_monitor =
      f->add_dft_fields(&dft_component, 1, f->v, 0.3, 0.3, 1, 2);
  (void)dft_monitor;
  f->add_flux_vol(Y, volume(vec(-0.35, -0.4), vec(-0.35, 0.4)));
  BackendState *const dft_old_state = f->backend_state;
  f->advance(1);
  CHECK(f->backend_state != dft_old_state && f->descriptors->legacy_fluxes.size() == 5 &&
            f->step_plans[0]->source_signature ==
                source_plan_signature(f->descriptors->sources) &&
            dft_plan_signature(f->step_plans[0]->dft_updates) ==
                dft_plan_signature(f->descriptors->dfts) &&
            !is_dirty(*f, dirty_flux_plan) && !is_dirty(*f, dirty_executable),
        "same-advance DFT and legacy-flux refresh preserved a stale non-flux plan");

  const uint64_t remove_nonflux_signature =
      non_flux_step_plan_signature(*f->step_plans[0]);
  f->remove_fluxes();
  BackendState *const remove_state = f->backend_state;
  Executable *const remove_executable = f->executable;
  backend_set_legacy_flux_prepare_failure_for_testing(0);
  failed = false;
  try {
    f->advance(1);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  backend_set_legacy_flux_prepare_failure_for_testing(-1);
  CHECK(sum_to_all(int(failed)) == count_processors() && f->backend_state == remove_state &&
            f->executable == remove_executable && is_dirty(*f, dirty_flux_plan),
        "failed zero-monitor legacy flux refresh changed resident artifacts");
  f->advance(1);
  CHECK(f->backend_state == remove_state && f->descriptors->legacy_fluxes.empty() &&
            counts.legacy_flux_updates_at_compile == 0 &&
            non_flux_step_plan_signature(*f->step_plans[0]) == remove_nonflux_signature,
        "legacy flux removal did not publish an empty recipe set without rebuilding state");

  delete f;
  delete s;
}

static void test_resident_legacy_flux_rank_mismatch() {
  if (count_processors() == 1) return;

  {
    structure *s;
    fields *f;
    build(&s, &f);
    f->require_component(Ey);
    f->require_component(Hy);
    f->require_component(Hz);
    f->add_flux_vol(X, volume(vec(0.1, -0.8), vec(0.1, 0.8)));
    f->advance(1);
    StepPlan *const old_plan = f->step_plans[0];
    if (my_rank() == 0)
      f->add_flux_vol(Y, volume(vec(-0.8, -0.2), vec(0.8, -0.2)));
    bool failed = false;
    try {
      f->advance(1);
    }
    catch (const std::runtime_error &) {
      failed = true;
    }
    CHECK(sum_to_all(int(failed)) == count_processors() && f->step_plans[0] == old_plan &&
              is_dirty(*f, dirty_flux_plan) && is_dirty(*f, dirty_executable),
          "warm CPU rank-asymmetric legacy flux mutation was not rejected collectively");
    delete f;
    delete s;
  }

  {
    structure *s;
    fields *f;
    build(&s, &f);
    f->require_component(Ey);
    f->require_component(Hy);
    f->require_component(Hz);
    if (my_rank() == 0)
      f->add_flux_vol(X, volume(vec(0.1, -0.8), vec(0.1, 0.8)));
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts);
    bool failed = false;
    try {
      f->advance(1);
    }
    catch (const std::runtime_error &) {
      failed = true;
    }
    CHECK(sum_to_all(int(failed)) == count_processors() && !f->backend_state && !f->executable,
          "cold rank-asymmetric legacy flux definition reached resident construction");
    delete f;
    delete s;
  }

  {
    structure *s;
    fields *f;
    build(&s, &f);
    f->require_component(Ey);
    f->require_component(Hy);
    f->require_component(Hz);
    flux_vol *flux =
        f->add_flux_vol(X, volume(vec(0.1, -0.8), vec(0.1, 0.8)));
    f->advance(1);
    delete f->executable;
    f->executable = NULL;
    delete f->backend_state;
    f->backend_state = NULL;
    delete f->backend;
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts);
    if (my_rank() == 0)
      f->add_flux_vol(Y, volume(vec(-0.8, -0.2), vec(0.8, -0.2)));
    invalidate(*f, MutationKind::precision_policy, "legacy flux backend reselection test");
    bool failed = false;
    try {
      f->advance(1);
    }
    catch (const std::runtime_error &) {
      failed = true;
    }
    CHECK(sum_to_all(int(failed)) == count_processors() && !f->backend_state && !f->executable,
          "backend reselection accepted rank-asymmetric legacy flux definitions");
    delete f;
    delete s;
  }

  structure *s;
  fields *f;
  build(&s, &f);
  f->require_component(Ey);
  f->require_component(Hy);
  f->require_component(Hz);
  f->add_flux_vol(X, volume(vec(0.1, -0.8), vec(0.1, 0.8)));
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);
  BackendState *const state = f->backend_state;
  Executable *const executable = f->executable;

  if (my_rank() == 0)
    f->add_flux_vol(Y, volume(vec(-0.8, -0.2), vec(0.8, -0.2)));
  bool failed = false;
  try {
    f->advance(1);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors() && f->backend_state == state &&
            f->executable == executable,
        "rank-asymmetric legacy flux definition was not rejected before replacement");

  delete f;
  delete s;
}

static void test_resident_legacy_flux_catalog_rebuild() {
  {
    structure *s;
    fields *f;
    build(&s, &f);
    f->require_component(Ey);
    f->require_component(Hy);
    f->require_component(Hz);
    flux_vol *flux =
        f->add_flux_vol(X, volume(vec(0.1, -0.8), vec(0.1, 0.8)));
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts);
    f->advance(1);
    BackendState *old_state = f->backend_state;
    const uint64_t flux_generation =
        generation(*f, MutationKind::legacy_flux_definition);
    const double published = 3.25;
    backend_publish_legacy_flux(*f, &published, 1, "legacy flux catalog continuity test");

    invalidate(*f, MutationKind::material_values, "legacy flux material candidate rebuild test");
    f->advance(1);
    CHECK(f->backend_state != old_state && f->descriptors->legacy_fluxes.size() == 1 &&
              counts.legacy_flux_updates_at_compile == 1 &&
              f->step_plans[0]->legacy_flux_updates.size() == 1 &&
              generation(*f, MutationKind::legacy_flux_definition) == flux_generation &&
              !is_dirty(*f, dirty_flux_plan),
          "warm material candidate dropped a clean legacy flux recipe");
    old_state = f->backend_state;

    invalidate(*f, MutationKind::field_layout, "legacy flux catalog rebuild test");
    f->advance(1);
    CHECK(f->backend_state != old_state && f->descriptors->legacy_fluxes.size() == 1 &&
              counts.legacy_flux_updates_at_compile == 1 &&
              generation(*f, MutationKind::legacy_flux_definition) == flux_generation &&
              !is_dirty(*f, dirty_flux_plan) && flux->flux() == published,
          "clean legacy flux recipe was not rebound after resident catalog replacement");
    delete f;
    delete s;
  }

  {
    structure *s;
    fields *f;
    build(&s, &f);
    f->require_component(Ey);
    f->require_component(Hy);
    f->require_component(Hz);
    flux_vol *flux =
        f->add_flux_vol(X, volume(vec(0.1, -0.8), vec(0.1, 0.8)));
    f->advance(2);
    CHECK(f->descriptors->legacy_flux_generation ==
                generation(*f, MutationKind::legacy_flux_definition) &&
              f->descriptors->legacy_fluxes.size() == 1 && f->step_plans[0] &&
              f->step_plans[0]->legacy_flux_updates.size() == 1 &&
              f->step_plans[0]->legacy_flux_terms.size() ==
                  f->descriptors->legacy_fluxes[0].terms.size(),
          "cold CPU multi-step advance retained an empty or stale legacy flux recipe");
    CHECK(or_to_all(!f->descriptors->legacy_fluxes[0].terms.empty()),
          "cold CPU multi-step advance produced no legacy flux terms on any rank");
    size_t half_accesses = 0, final_accesses = 0;
    for (const Operation &op : f->step_plans[0]->operations) {
      if (op.kind == OpKind::update_flux_half) half_accesses = op.accesses.size();
      if (op.kind == OpKind::update_flux) final_accesses = op.accesses.size();
    }
    const bool any_cpu_flux_access = or_to_all(half_accesses > 0);
    CHECK(half_accesses == final_accesses && any_cpu_flux_access,
          "cold CPU legacy flux plan omitted or mismatched marker accesses");

    const uint64_t cpu_flux_generation =
        generation(*f, MutationKind::legacy_flux_definition);
    if (my_rank() == 0)
      invalidate(*f, MutationKind::field_layout, "CPU legacy flux catalog rebuild test");
    f->advance(1);
    const StepPlan rebuilt_cpu_plan = build_step_plan(*f, StepProgram::ordinary);
    const bool any_rebuilt_cpu_flux_term =
        or_to_all(!f->step_plans[0]->legacy_flux_terms.empty());
    CHECK(f->descriptors->legacy_flux_generation == cpu_flux_generation &&
              f->descriptors->legacy_fluxes.size() == 1 &&
              f->step_plans[0]->signature == rebuilt_cpu_plan.signature &&
              f->step_plans[0]->legacy_flux_updates.size() == 1 &&
              f->step_plans[0]->legacy_flux_terms.size() ==
                  f->descriptors->legacy_fluxes[0].terms.size() &&
              any_rebuilt_cpu_flux_term,
          "established CPU legacy flux recipe was stale after catalog replacement");
    const double cpu_flux = flux->flux();
    delete f->executable;
    f->executable = NULL;
    delete f->backend_state;
    f->backend_state = NULL;
    delete f->backend;
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts);
    invalidate(*f, MutationKind::precision_policy, "CPU-to-resident legacy flux test");
    f->advance(1);
    CHECK(f->backend_state && f->executable && f->descriptors->legacy_fluxes.size() == 1 &&
              counts.legacy_flux_updates_at_compile == 1 && !is_dirty(*f, dirty_flux_plan) &&
              flux->flux() == cpu_flux,
          "CPU-refreshed legacy flux recipe was not rebound during resident selection");
    delete f;
    delete s;
  }
}

static void test_backend_reselection_invalidates_representation() {
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);
  CHECK(f->backend_state && f->executable,
        "backend replacement fixture did not prepare resident execution");
  const uint64_t generation_before = generation(*f, MutationKind::precision_policy);

  execution_options cpu;
  f->select_backend(cpu);
  CHECK(!f->backend_state && !f->executable && counts.states_destroyed == 1 &&
            counts.executables_destroyed == 1,
        "backend replacement did not retire the prior representation exactly once");
  CHECK(is_dirty(*f, dirty_storage) && is_dirty(*f, dirty_initialization) &&
            is_dirty(*f, dirty_executable),
        "backend replacement did not invalidate storage, values, and executable");
  CHECK(generation(*f, MutationKind::precision_policy) == generation_before + 1,
        "backend replacement did not record a representation generation");

  delete f;
  delete s;
}

static void add_multilevel_lifecycle_states(structure &s) {
  const realnum e_gamma[] = {realnum(0.02), 0, 0, 0, realnum(0.03), 0, 0, 0,
                             realnum(0.04)};
  const realnum e_n0[] = {realnum(0.7), realnum(0.2), realnum(0.1)};
  const realnum e_alpha[] = {realnum(-0.2), realnum(0.3), realnum(0.4), realnum(-0.5),
                             realnum(-0.6), realnum(0.7)};
  const realnum e_omega[] = {realnum(0.73), realnum(0.91)};
  const realnum e_damping[] = {realnum(0.06), realnum(0.08)};
  const realnum e_sigmat[] = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
  multilevel_susceptibility electric(3, 2, e_gamma, e_n0, e_alpha, e_omega, e_damping,
                                     e_sigmat);
  const realnum h_gamma[] = {realnum(0.01), realnum(0.005), 0, realnum(0.025)};
  const realnum h_n0[] = {realnum(0.8), realnum(0.2)};
  const realnum h_alpha[] = {realnum(-0.4), realnum(0.5)};
  const realnum h_omega[] = {realnum(0.63)};
  const realnum h_damping[] = {realnum(0.04)};
  const realnum h_sigmat[] = {3, 3, 3, 3, 3};
  multilevel_susceptibility magnetic(2, 1, h_gamma, h_n0, h_alpha, h_omega, h_damping,
                                     h_sigmat);
  s.add_susceptibility(unit_epsilon, E_stuff, electric);
  s.add_susceptibility(unit_epsilon, H_stuff, magnetic);
}

struct multilevel_value_snapshot {
  StorageKey key;
  const void *address;
  std::vector<realnum> values;
};

static std::vector<multilevel_value_snapshot> capture_multilevel_values(const fields &f) {
  std::vector<multilevel_value_snapshot> result;
  if (!f.array_catalog) return result;
  for (size_t i = 0; i < f.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const StorageKey key = f.array_catalog->key(id);
    if (key.kind != int(array_kind::polarization_internal)) continue;
    const ArraySpec &spec = f.array_catalog->spec(id);
    if (is_valid(spec.alias_of) || spec.element_type != ElementType::realnum_value) continue;
    const realnum *values = f.array_catalog->resolve<realnum>(id);
    result.push_back(multilevel_value_snapshot{
        key, values, std::vector<realnum>(values, values + spec.elements)});
  }
  return result;
}

static bool multilevel_addresses_match(const std::vector<multilevel_value_snapshot> &expected,
                                       const fields &actual, bool should_match) {
  if (!actual.array_catalog) return expected.empty();
  for (const multilevel_value_snapshot &row : expected) {
    const ArrayId id = actual.array_catalog->find(row.key);
    if (!is_valid(id)) return false;
    if ((actual.array_catalog->resolve_untyped(id) == row.address) != should_match) return false;
  }
  return true;
}

static bool multilevel_values_equal(const std::vector<multilevel_value_snapshot> &expected,
                                    const fields &actual) {
  if (!actual.array_catalog) return expected.empty();
  for (const multilevel_value_snapshot &row : expected) {
    const ArrayId id = actual.array_catalog->find(row.key);
    if (!is_valid(id) || id.value >= actual.array_catalog->size()) return false;
    const ArraySpec &spec = actual.array_catalog->spec(id);
    if (spec.elements != row.values.size()) return false;
    const realnum *values = actual.array_catalog->resolve<realnum>(id);
    if (memcmp(values, row.values.data(), row.values.size() * sizeof(realnum))) return false;
  }
  return true;
}

static void test_resident_multilevel_lifecycle() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);

  {
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_multilevel_lifecycle_states(s);
    fields f(&s);
    f.use_real_fields();
    f.require_component(Ez);
    f.require_component(Hz);
    gaussian_src_time source(0.3, 0.1);
    f.add_point_source(Ez, source, vec(0.11, 0.13));
    f.add_flux_vol(X, volume(vec(0.1, -0.8), vec(0.1, 0.8)));
    bool owns_chunk = false;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk)
      owns_chunk = owns_chunk || f.chunks[chunk]->is_mine();
    bool initially_null = true;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk)
      if (f.chunks[chunk]->is_mine())
        FOR_FIELD_TYPES(ft)
          for (polarization_state *state = f.chunks[chunk]->pol[ft]; state;
               state = state->next)
            initially_null = initially_null && !state->data;
    CHECK(and_to_all(initially_null),
          "multilevel state was allocated before resident preflight");

    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts);
    f.advance(1);
    CHECK(or_to_all(counts.multilevel_population_updates_at_compile > 0 &&
                    counts.multilevel_transition_updates_at_compile > 0),
          "resident cold build omitted multilevel actions");
    CHECK(or_to_all(counts.polarization_arrays_at_create > 0) &&
              and_to_all(counts.connections_current_at_create),
          "resident state was created before multilevel storage/topology preparation");
    BackendState *const state = f.backend_state;
    Executable *const executable = f.executable;
    const int entry_t = f.t;
    const std::vector<multilevel_value_snapshot> initialized_values =
        capture_multilevel_values(f);

    ArrayId gamma_id = invalid_array();
    for (const PolarizationDescriptor &descriptor : f.descriptors->polarizations)
      if (descriptor.kind == SusceptibilityKind::multilevel) {
        gamma_id = descriptor.multilevel_gamma_inv;
        break;
      }
    CHECK(is_valid(gamma_id) || !owns_chunk,
          "resident multilevel fixture has no GammaInv row on an owning rank");
    realnum authoritative_gamma = 0;
    if (is_valid(gamma_id)) {
      realnum *gamma = f.array_catalog->resolve<realnum>(gamma_id);
      gamma[0] += realnum(0.125);
      authoritative_gamma = gamma[0];
    }
    counts.migrate_multilevel_values = true;

    fields copy(f);
    CHECK(f.backend_state == state && f.executable == executable && f.t == entry_t,
          "fields clone retired or advanced its resident source epoch");
    CHECK(or_to_all(counts.multilevel_migrations > 0),
          "fields clone did not materialize resident multilevel state");
    const std::vector<multilevel_value_snapshot> source_values = capture_multilevel_values(f);
    build_storage_catalog(copy, *copy.array_catalog, *copy.storage_plan);
    build_polarization_descriptors(copy, copy.descriptors->polarizations);
    CHECK(multilevel_values_equal(source_values, copy),
          "fields clone lost multilevel GammaInv/N/P/P_prev values");
    CHECK(multilevel_addresses_match(source_values, copy, false),
          "fields clone aliased source multilevel storage");
    if (is_valid(gamma_id))
      CHECK(f.array_catalog->resolve<realnum>(gamma_id)[0] == authoritative_gamma,
            "clone migration overwrote host-authoritative GammaInv");

    counts.fail_rebuild = true;
    bool clone_rejected = false;
    try { fields rejected(f); }
    catch (const std::runtime_error &) { clone_rejected = true; }
    CHECK(and_to_all(clone_rejected) && f.backend_state == state && f.executable == executable &&
              f.t == entry_t,
          "failed resident clone changed the source epoch");
    counts.fail_rebuild = false;

    const std::vector<multilevel_value_snapshot> before_zero_addresses =
        capture_multilevel_values(f);
    const std::vector<multilevel_value_snapshot> independent_copy =
        capture_multilevel_values(copy);
    f.zero_fields();
    CHECK(f.backend_state == state && f.executable == executable && f.t == entry_t,
          "zero_fields replaced resident multilevel state or advanced time");
    CHECK(multilevel_values_equal(initialized_values, f) &&
              multilevel_addresses_match(before_zero_addresses, f, true),
          "zero_fields did not restore exact GammaInv/N0/zero-P values in-place");
    CHECK(multilevel_values_equal(independent_copy, copy),
          "zero_fields mutated the independent multilevel clone");
    f.zero_fields();
    CHECK(f.backend_state == state && f.executable == executable && f.t == entry_t &&
              multilevel_values_equal(initialized_values, f) &&
              multilevel_addresses_match(before_zero_addresses, f, true),
          "repeated multilevel zero_fields changed the resident epoch, rows, or addresses");
    f.advance(f.t + 1);
    CHECK(f.backend_state == state && f.executable == executable,
          "advance rebuilt resident state after multilevel zero_fields");

    BackendState *const rebuilt = f.backend_state;
    Executable *const rebuilt_executable = f.executable;
    src_time *const source_before_reset = f.sources;
    flux_vol *const flux_before_reset = f.fluxes;
    DescriptorSet *const descriptors_before_reset = f.descriptors;
    StepPlan *const ordinary_plan_before_reset = f.step_plans[0];
    StepPlan *const cw_plan_before_reset = f.step_plans[1];
    const DirtyMask dirty_before_reset = DirtyMask(f.dirty_mask);
    counts.fail_rebuild = true;
    bool reset_rejected = false;
    try { f.reset(); }
    catch (const std::runtime_error &) { reset_rejected = true; }
    CHECK(and_to_all(reset_rejected) && f.backend_state == rebuilt &&
              f.sources == source_before_reset && f.fluxes == flux_before_reset &&
              f.descriptors == descriptors_before_reset &&
              f.step_plans[0] == ordinary_plan_before_reset &&
              f.step_plans[1] == cw_plan_before_reset &&
              DirtyMask(f.dirty_mask) == dirty_before_reset && f.t == entry_t,
          "failed multilevel reset changed definitions, plans, time, dirtiness, or resident state");
    counts.fail_rebuild = false;
    const std::vector<multilevel_value_snapshot> before_reset_addresses =
        capture_multilevel_values(f);
    f.reset();
    CHECK(!f.backend_state && !f.executable && f.t == 0 &&
              !f.sources && !f.fluxes &&
              multilevel_values_equal(initialized_values, f) &&
              multilevel_addresses_match(before_reset_addresses, f, true),
          "multilevel reset replaced state or failed to clear definitions and restore exact rows");
    f.reset();
    CHECK(!f.backend_state && !f.executable && f.t == 0 &&
              !f.sources && !f.fluxes && multilevel_values_equal(initialized_values, f) &&
              multilevel_addresses_match(before_reset_addresses, f, true),
          "repeated multilevel reset changed the resident epoch, rows, or addresses");
  }

  {
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_multilevel_lifecycle_states(s);
    fields f(&s);
    f.use_real_fields();
    f.require_component(Ez);
    bool owns_chunk = false;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk)
      owns_chunk = owns_chunk || f.chunks[chunk]->is_mine();
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts);
    f.advance(1);
    counts.migrate_multilevel_values = true;
    f.backend->prepare_state_rebuild(*f.backend_state, dirty_storage);
    ArrayId gamma_id = invalid_array();
    for (const PolarizationDescriptor &descriptor : f.descriptors->polarizations)
      if (descriptor.kind == SusceptibilityKind::multilevel) {
        gamma_id = descriptor.multilevel_gamma_inv;
        break;
      }
    const StorageKey gamma_key =
        is_valid(gamma_id) ? f.array_catalog->key(gamma_id) : StorageKey{-1, -1, -1, -1, 0};
    realnum authoritative_gamma = 0;
    if (is_valid(gamma_id)) {
      realnum *gamma = f.array_catalog->resolve<realnum>(gamma_id);
      gamma[0] += realnum(0.15625);
      authoritative_gamma = gamma[0];
    }
    const std::vector<multilevel_value_snapshot> values_before_growth =
        capture_multilevel_values(f);
    const int states_before_growth = counts.states_created;
    const int destroyed_before_growth = counts.states_destroyed;
    const size_t arrays_before_growth = counts.arrays_at_create;
    f.require_component(Hz);
    f.advance(f.t + 1);
    CHECK(counts.states_created == states_before_growth + 1 &&
              counts.states_destroyed == destroyed_before_growth + 1 && f.backend_state &&
              f.executable && (!owns_chunk || counts.arrays_at_create > arrays_before_growth),
          "multilevel field growth did not replace and expand the resident epoch");
    CHECK(multilevel_values_equal(values_before_growth, f),
          "multilevel field growth lost resident N/P/P_prev values");
    if (is_valid(gamma_id)) {
      gamma_id = f.array_catalog->find(gamma_key);
      CHECK(is_valid(gamma_id) &&
                f.array_catalog->resolve<realnum>(gamma_id)[0] == authoritative_gamma,
            "field-growth rebuild overwrote or lost host-authoritative GammaInv");
    }
  }

  {
    structure s(gv, unit_epsilon, no_pml(), identity(), 3);
    add_multilevel_lifecycle_states(s);
    fields f(&s);
    f.use_real_fields();
    f.require_component(Ez);
    f.require_component(Hz);
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts);
    f.advance(1);
    counts.migrate_multilevel_values = true;
    BackendState *const state_before_remove = f.backend_state;
    Executable *const executable_before_remove = f.executable;
    const DirtyMask dirty_before_remove = DirtyMask(f.dirty_mask);
    const int t_before_remove = f.t;
    CpuArrayCatalog *const catalog_before_remove = f.array_catalog;
    DescriptorSet *const descriptors_before_remove = f.descriptors;
    StepPlan *const ordinary_plan_before_remove = f.step_plans[0];
    std::vector<polarization_state *> polarization_before_remove;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk)
      FOR_FIELD_TYPES(ft) polarization_before_remove.push_back(f.chunks[chunk]->pol[ft]);
    counts.fail_rebuild = true;
    bool remove_rejected = false;
    try { f.remove_susceptibilities(); }
    catch (const std::runtime_error &) { remove_rejected = true; }
    bool polarization_unchanged = true;
    size_t polarization_index = 0;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk)
      FOR_FIELD_TYPES(ft) polarization_unchanged =
          polarization_unchanged &&
          f.chunks[chunk]->pol[ft] == polarization_before_remove[polarization_index++];
    CHECK(and_to_all(remove_rejected) && f.backend_state == state_before_remove &&
              f.executable == executable_before_remove &&
              f.t == t_before_remove && f.array_catalog == catalog_before_remove &&
              f.descriptors == descriptors_before_remove &&
              f.step_plans[0] == ordinary_plan_before_remove &&
              DirtyMask(f.dirty_mask) == dirty_before_remove && polarization_unchanged,
          "failed multilevel removal changed the live epoch or polarization lists");
    counts.fail_rebuild = false;
    const int migrations_before = counts.multilevel_migrations;
    f.remove_susceptibilities();
    CHECK(!f.backend_state && !f.executable &&
              or_to_all(counts.multilevel_migrations > migrations_before),
          "multilevel removal did not migrate and retire resident state first");
    bool empty = true;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk)
      FOR_FIELD_TYPES(ft) empty = empty && !f.chunks[chunk]->pol[ft];
    CHECK(and_to_all(empty), "distributed multilevel removal retained polarization state");
    f.remove_susceptibilities();
    f.advance(f.t + 1);
    bool stale_catalog = false, stale_descriptors = false, stale_actions = false,
         stale_halos = false;
    for (size_t i = 0; i < f.array_catalog->size(); ++i)
      stale_catalog = stale_catalog ||
                      f.array_catalog->key(ArrayId{uint32_t(i)}).kind ==
                          int(array_kind::polarization_internal);
    for (const PolarizationDescriptor &descriptor : f.descriptors->polarizations)
      stale_descriptors = stale_descriptors || descriptor.kind == SusceptibilityKind::multilevel;
    const StepPlan &removed_plan = *f.step_plans[0];
    stale_actions = !removed_plan.multilevel_population_updates.empty() ||
                    !removed_plan.multilevel_population_terms.empty() ||
                    !removed_plan.multilevel_transition_updates.empty();
    for (const HaloPlan &plan : f.halos->plans)
      if (plan.ft == PE_stuff || plan.ft == PH_stuff) {
        std::vector<ElementRef> refs;
        expand_gather(plan, refs);
        stale_halos = stale_halos || !refs.empty();
        expand_scatter(plan, refs);
        stale_halos = stale_halos || !refs.empty();
      }
    CHECK(and_to_all(!stale_catalog && !stale_descriptors && !stale_actions && !stale_halos),
          "same-object removal rebuild retained multilevel catalog, plan, or halo state");

    fields readded(&s);
    readded.use_real_fields();
    readded.require_component(Ez);
    readded.require_component(Hz);
    lifetime_counts readded_counts;
    readded.backend = new tracking_backend(readded, readded_counts);
    readded.advance(1);
    CHECK(or_to_all(readded_counts.multilevel_population_updates_at_compile > 0),
          "fresh remove/re-add construction lost multilevel actions");
  }

  {
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_multilevel_lifecycle_states(s);
    fields f(&s);
    f.use_real_fields();
    f.require_component(Ez);
    f.require_component(Hz);
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts);
    f.options.precision = precision_policy_kind::mixed;
    f.advance(1);
    counts.migrate_multilevel_values = true;
    f.backend->prepare_state_rebuild(*f.backend_state, dirty_storage);
    ArrayId gamma_id = invalid_array();
    for (const PolarizationDescriptor &descriptor : f.descriptors->polarizations)
      if (descriptor.kind == SusceptibilityKind::multilevel) {
        gamma_id = descriptor.multilevel_gamma_inv;
        break;
      }
    realnum authoritative_gamma = 0;
    if (is_valid(gamma_id)) {
      realnum *gamma = f.array_catalog->resolve<realnum>(gamma_id);
      gamma[0] += realnum(0.1875);
      authoritative_gamma = gamma[0];
    }
    const std::vector<multilevel_value_snapshot> values_before_reselection =
        capture_multilevel_values(f);
    execution_options cpu;
    f.select_backend(cpu);
    CHECK(!f.backend_state && !f.executable &&
              or_to_all(counts.multilevel_migrations > 0) &&
              multilevel_values_equal(values_before_reselection, f) &&
              multilevel_addresses_match(values_before_reselection, f, true),
          "CPU reselection did not migrate and retire multilevel resident state");
    if (is_valid(gamma_id))
      CHECK(f.array_catalog->resolve<realnum>(gamma_id)[0] == authoritative_gamma,
            "precision/backend reselection overwrote host-authoritative GammaInv");
  }
}

static execution_options warned_custom_options(
    precision_policy_kind precision = precision_policy_kind::native) {
  execution_options opts;
  opts.backend = backend_kind::automatic;
  opts.precision = precision;
  opts.fallback = fallback_policy::warn;
  opts.strict = false;
  return opts;
}

static void add_custom_lifecycle_state(structure &s, bool publish_layout = false) {
  s.add_susceptibility(unit_epsilon, E_stuff,
                       lifecycle_custom_susceptibility(0.85, 0.07, publish_layout));
}

static uint64_t custom_counter_value(const HostCustomFallbackStats &stats,
                                     HostCustomFallbackCounter counter) {
  switch (counter) {
    case HostCustomFallbackCounter::warnings: return stats.warnings;
    case HostCustomFallbackCounter::preflights: return stats.preflights;
    case HostCustomFallbackCounter::sessions: return stats.sessions;
    case HostCustomFallbackCounter::callbacks: return stats.callbacks;
    case HostCustomFallbackCounter::completed_sessions: return stats.completed_sessions;
    case HostCustomFallbackCounter::staging_allocations: return stats.staging_allocations;
    case HostCustomFallbackCounter::staging_bytes: return stats.staging_bytes;
    case HostCustomFallbackCounter::downloads: return stats.downloads;
    case HostCustomFallbackCounter::download_bytes: return stats.download_bytes;
    case HostCustomFallbackCounter::uploads: return stats.uploads;
    case HostCustomFallbackCounter::upload_bytes: return stats.upload_bytes;
    case HostCustomFallbackCounter::retryable_failures: return stats.retryable_failures;
    case HostCustomFallbackCounter::poisoned_failures: return stats.poisoned_failures;
  }
  return 0;
}

static bool same_host_custom_stats(const HostCustomFallbackStats &a,
                                   const HostCustomFallbackStats &b) {
  return a.warnings == b.warnings && a.preflights == b.preflights &&
         a.sessions == b.sessions && a.callbacks == b.callbacks &&
         a.completed_sessions == b.completed_sessions &&
         a.staging_allocations == b.staging_allocations &&
         a.staging_bytes == b.staging_bytes && a.downloads == b.downloads &&
         a.download_bytes == b.download_bytes && a.uploads == b.uploads &&
         a.upload_bytes == b.upload_bytes &&
         a.retryable_failures == b.retryable_failures &&
         a.poisoned_failures == b.poisoned_failures;
}

static void test_resident_host_custom_policy_lifecycle() {
  auto expect_early_rejection = [](const char *label, const execution_options &opts,
                                   bool backend_support, const char *message,
                                   int phasein_time = 0) {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s, true);
    fields f(&s);
    f.require_component(Ez);
    f.phasein_time = phasein_time;
    lifecycle_custom_susceptibility::reset_counts();
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts, false, false, backend_support, true);
    f.options = opts;
    bool rejected = false;
    std::string what;
    try {
      f.advance(1);
    }
    catch (const std::exception &e) {
      rejected = true;
      what = e.what();
    }
    CHECK(rejected && (!message || what.find(message) != std::string::npos),
          "%s was not rejected by the early custom capability gate: %s", label, what.c_str());
    CHECK(!f.backend_state && !f.executable && counts.states_created == 0 &&
              counts.executables_created == 0 && lifecycle_custom_susceptibility::allocations == 0 &&
              lifecycle_custom_susceptibility::initializations == 0 &&
              lifecycle_custom_susceptibility::layout_queries == 0 && !f.backend->is_poisoned() &&
              f.backend->host_custom_fallback_stats().warnings == 0,
          "%s reached allocation, descriptor callbacks, publication, warning, or poison", label);
  };

  {
    execution_options strict = warned_custom_options();
    strict.strict = true;
    expect_early_rejection("strict custom fallback", strict, true, "strict=false");
  }
  {
    execution_options error = warned_custom_options();
    error.fallback = fallback_policy::error;
    expect_early_rejection("error-policy custom fallback", error, true, "fallback=warn");
  }
  expect_early_rejection("mixed-precision custom fallback",
                         warned_custom_options(precision_policy_kind::mixed), true,
                         "precision=native");
  expect_early_rejection("f32 custom fallback", warned_custom_options(precision_policy_kind::f32),
                         true, "precision=native");
  expect_early_rejection("unsupported-backend custom fallback", warned_custom_options(), false,
                         count_processors() == 1 ? "does not implement" : "single MPI rank");
  if (count_processors() == 1)
    expect_early_rejection("material-phasing custom fallback", warned_custom_options(), true,
                           "material phasing", 2);

  {
    grid_volume gv = vol2d(1.0, 1.0, 4.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 1);
    fields f(&s);
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts);
    f.backend = tracking;
    const HostCustomFallbackCounter counters[] = {
        HostCustomFallbackCounter::warnings,
        HostCustomFallbackCounter::preflights,
        HostCustomFallbackCounter::sessions,
        HostCustomFallbackCounter::callbacks,
        HostCustomFallbackCounter::completed_sessions,
        HostCustomFallbackCounter::staging_allocations,
        HostCustomFallbackCounter::staging_bytes,
        HostCustomFallbackCounter::downloads,
        HostCustomFallbackCounter::download_bytes,
        HostCustomFallbackCounter::uploads,
        HostCustomFallbackCounter::upload_bytes,
        HostCustomFallbackCounter::retryable_failures,
        HostCustomFallbackCounter::poisoned_failures};
    for (HostCustomFallbackCounter counter : counters) {
      backend_set_host_custom_counter_for_testing(*tracking, counter,
                                                  std::numeric_limits<uint64_t>::max());
      bool overflowed = false;
      try { backend_increment_host_custom_counter_for_testing(*tracking, counter); }
      catch (const std::overflow_error &) { overflowed = true; }
      CHECK(overflowed &&
                custom_counter_value(tracking->host_custom_fallback_stats(), counter) ==
                    std::numeric_limits<uint64_t>::max(),
            "host custom fallback counter wrapped instead of rejecting overflow");
      backend_set_host_custom_counter_for_testing(*tracking, counter, 0);
    }
  }

  /* Exact built-ins never enter the custom policy path, even with strict mode
     and a backend that deliberately declines the fallback hook. */
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    s.add_susceptibility(unit_epsilon, E_stuff, lorentzian_susceptibility(0.85, 0.07));
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts);
    f.advance(1);
    CHECK(counts.states_created == 1 && counts.advanced == 1 &&
              f.backend->host_custom_fallback_stats().warnings == 0,
          "exact built-in was misclassified as host custom fallback");
  }

  if (count_processors() != 1) {
    expect_early_rejection("MPI custom fallback", warned_custom_options(), true,
                           "single MPI rank");
    return;
  }

  /* Warning capacity is part of enablement publication. Overflow leaves the
     fallback disabled and the untouched epoch remains retryable. */
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    backend_set_host_custom_counter_for_testing(
        *tracking, HostCustomFallbackCounter::warnings,
        std::numeric_limits<uint64_t>::max());
    bool overflowed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { overflowed = true; }
    CHECK(overflowed && !tracking->host_custom_fallback_enabled() && !f.backend_state &&
              !f.executable && counts.custom_preflights == 0 && counts.advance_attempts == 0,
          "warning overflow partially published custom enablement");
    backend_set_host_custom_counter_for_testing(*tracking, HostCustomFallbackCounter::warnings,
                                                0);
    f.advance(1);
    CHECK(tracking->host_custom_fallback_enabled() &&
              tracking->host_custom_fallback_stats().warnings == 1 &&
              tracking->host_custom_fallback_stats().completed_sessions == 2 &&
              !tracking->is_poisoned(),
          "warning overflow did not leave a retryable custom epoch");
  }

  /* All dispatch-wide lifecycle deltas are reserved before backend preflight,
     so no session can become reachable if a later exact counter would wrap. */
  for (HostCustomFallbackCounter counter : {HostCustomFallbackCounter::sessions,
                                            HostCustomFallbackCounter::callbacks,
                                            HostCustomFallbackCounter::completed_sessions}) {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    f.advance(1);
    const int preflights_before = counts.custom_preflights;
    const int advances_before = counts.advance_attempts;
    backend_set_host_custom_counter_for_testing(*tracking, counter,
                                                std::numeric_limits<uint64_t>::max());
    bool overflowed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { overflowed = true; }
    CHECK(overflowed && counts.custom_preflights == preflights_before &&
              counts.advance_attempts == advances_before &&
              !tracking->is_poisoned(),
          "dispatch counter capacity failure reached preflight or callback entry");
  }

  /* The concrete staging and transfer recorders precheck both halves of each
     pair. Neither a count nor byte overflow may partially update its peer. */
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    backend_set_host_custom_counter_for_testing(
        *tracking, HostCustomFallbackCounter::staging_allocations,
        std::numeric_limits<uint64_t>::max());
    backend_set_host_custom_counter_for_testing(*tracking,
                                                HostCustomFallbackCounter::staging_bytes, 17);
    bool overflowed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { overflowed = true; }
    const HostCustomFallbackStats &stats = tracking->host_custom_fallback_stats();
    CHECK(overflowed &&
              stats.staging_allocations == std::numeric_limits<uint64_t>::max() &&
              stats.staging_bytes == 17 && stats.sessions == 0 &&
              stats.retryable_failures == 1 && !tracking->is_poisoned(),
          "staging allocation count overflow changed its paired byte field");
  }
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    backend_set_host_custom_counter_for_testing(
        *tracking, HostCustomFallbackCounter::staging_bytes,
        std::numeric_limits<uint64_t>::max());
    bool overflowed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { overflowed = true; }
    const HostCustomFallbackStats &stats = tracking->host_custom_fallback_stats();
    CHECK(overflowed && stats.staging_allocations == 0 &&
              stats.staging_bytes == std::numeric_limits<uint64_t>::max() &&
              stats.sessions == 0 && stats.retryable_failures == 1 && !tracking->is_poisoned(),
          "staging byte overflow partially incremented its allocation count");
    backend_set_host_custom_counter_for_testing(*tracking,
                                                HostCustomFallbackCounter::staging_bytes, 0);
    f.advance(1);
    CHECK(stats.staging_allocations == 1 && stats.staging_bytes == 64 &&
              stats.completed_sessions == 2 && !tracking->is_poisoned(),
          "staging accounting overflow was not retryable");
  }
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    f.advance(1);
    const uint64_t bytes_before = tracking->host_custom_fallback_stats().download_bytes;
    backend_set_host_custom_counter_for_testing(*tracking, HostCustomFallbackCounter::downloads,
                                                std::numeric_limits<uint64_t>::max());
    bool overflowed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { overflowed = true; }
    const HostCustomFallbackStats &stats = tracking->host_custom_fallback_stats();
    CHECK(overflowed && stats.downloads == std::numeric_limits<uint64_t>::max() &&
              stats.download_bytes == bytes_before && stats.callbacks == 4 &&
              stats.retryable_failures == 1 && !tracking->is_poisoned(),
          "download count overflow changed its paired byte field");
  }
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    f.advance(1);
    backend_set_host_custom_counter_for_testing(*tracking, HostCustomFallbackCounter::downloads,
                                                7);
    backend_set_host_custom_counter_for_testing(
        *tracking, HostCustomFallbackCounter::download_bytes,
        std::numeric_limits<uint64_t>::max());
    bool overflowed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { overflowed = true; }
    const HostCustomFallbackStats &stats = tracking->host_custom_fallback_stats();
    CHECK(overflowed && stats.downloads == 7 &&
              stats.download_bytes == std::numeric_limits<uint64_t>::max() &&
              stats.callbacks == 4 && stats.retryable_failures == 1 &&
              !tracking->is_poisoned(),
          "download byte overflow partially incremented its transfer count");
    backend_set_host_custom_counter_for_testing(*tracking,
                                                HostCustomFallbackCounter::download_bytes, 0);
    f.advance(1);
    CHECK(stats.downloads == 9 && stats.download_bytes == 64 &&
              stats.completed_sessions == 4 && !tracking->is_poisoned(),
          "download accounting overflow was not retryable");
  }
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    f.advance(1);
    const uint64_t bytes_before = tracking->host_custom_fallback_stats().upload_bytes;
    backend_set_host_custom_counter_for_testing(*tracking, HostCustomFallbackCounter::uploads,
                                                std::numeric_limits<uint64_t>::max());
    bool overflowed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { overflowed = true; }
    const HostCustomFallbackStats &stats = tracking->host_custom_fallback_stats();
    CHECK(overflowed && stats.uploads == std::numeric_limits<uint64_t>::max() &&
              stats.upload_bytes == bytes_before && stats.callbacks == 6 &&
              stats.poisoned_failures == 1 && tracking->is_poisoned(),
          "upload count overflow changed its paired byte field or escaped poison");
  }
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    f.advance(1);
    backend_set_host_custom_counter_for_testing(*tracking, HostCustomFallbackCounter::uploads, 7);
    backend_set_host_custom_counter_for_testing(
        *tracking, HostCustomFallbackCounter::upload_bytes,
        std::numeric_limits<uint64_t>::max());
    bool overflowed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { overflowed = true; }
    const HostCustomFallbackStats &stats = tracking->host_custom_fallback_stats();
    CHECK(overflowed && stats.uploads == 7 &&
              stats.upload_bytes == std::numeric_limits<uint64_t>::max() &&
              stats.callbacks == 6 && stats.poisoned_failures == 1 && tracking->is_poisoned(),
          "upload byte overflow partially incremented its transfer count or escaped poison");
  }

  /* Native descriptors are ignored by the fallback schedule, while stateful
     and stateless custom descriptors contribute [2,4] in either live-list
     order: constitutive skips stateless, polarization includes both. */
  for (int order = 0; order < 2; ++order) {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    if (order == 0) {
      s.add_susceptibility(unit_epsilon, E_stuff, lorentzian_susceptibility(0.7, 0.03));
      add_custom_lifecycle_state(s);
      s.add_susceptibility(unit_epsilon, E_stuff,
                           lifecycle_stateless_custom_susceptibility());
    }
    else {
      s.add_susceptibility(unit_epsilon, E_stuff,
                           lifecycle_stateless_custom_susceptibility());
      add_custom_lifecycle_state(s);
      s.add_susceptibility(unit_epsilon, E_stuff, lorentzian_susceptibility(0.7, 0.03));
    }
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    f.advance(1);
    const HostCustomFallbackStats &stats = tracking->host_custom_fallback_stats();
    CHECK(stats.sessions == 2 && stats.callbacks == 6 && stats.completed_sessions == 2 &&
              !tracking->is_poisoned(),
          "mixed native/stateful/stateless custom order %d changed the exact [2,4] schedule",
          order);
  }

  auto expect_schedule_poison = [](const char *label, bool redistribute, bool reorder) {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    s.add_susceptibility(unit_epsilon, E_stuff,
                         lifecycle_stateless_custom_susceptibility());
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    counts.redistribute_custom_callbacks = redistribute;
    counts.reorder_custom_second_step = reorder;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    bool rejected = false;
    try { f.advance(reorder ? 2 : 1); }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected && tracking->is_poisoned() &&
              tracking->host_custom_fallback_stats().poisoned_failures == 1,
          "%s was accepted because its aggregate callback total matched", label);
  };
  expect_schedule_poison("same-total callback redistribution", true, false);
  expect_schedule_poison("second-step callback reorder", false, true);

  /* A stateful-only plan has the same callback count in both segments. The
     canonical operation/segment identity must still reject either a first-step
     swap or a swap after one complete step, before charging the bad session. */
  auto expect_equal_count_reorder = [](const char *label, bool second_step) {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    counts.reorder_custom_first_step = !second_step;
    counts.reorder_custom_second_step = second_step;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    bool rejected = false;
    try { f.advance(second_step ? 2 : 1); }
    catch (const std::runtime_error &) { rejected = true; }
    const HostCustomFallbackStats &stats = tracking->host_custom_fallback_stats();
    const uint64_t completed_before_reorder = second_step ? 2 : 0;
    const uint64_t callbacks_before_reorder = second_step ? 4 : 0;
    CHECK(rejected && tracking->is_poisoned() &&
              stats.sessions == completed_before_reorder &&
              stats.completed_sessions == completed_before_reorder &&
              stats.callbacks == callbacks_before_reorder && stats.poisoned_failures == 1,
          "%s was accepted or charged before its identity mismatch", label);
  };
  expect_equal_count_reorder("equal-count first-step segment reorder", false);
  expect_equal_count_reorder("equal-count second-step segment reorder", true);

  /* A stateless custom susceptibility contributes no constitutive virtual but
     still receives update_P. advance(2) also pins exact per-step multiplication
     of the two host-segment sessions. */
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    s.add_susceptibility(unit_epsilon, E_stuff,
                         lifecycle_stateless_custom_susceptibility());
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    f.advance(2);
    const HostCustomFallbackStats &stats = tracking->host_custom_fallback_stats();
    CHECK(stats.sessions == 4 && stats.completed_sessions == 4 && stats.callbacks == 4 &&
              !tracking->is_poisoned(),
          "stateless/batched custom dispatch did not preserve exact segment and callback counts");
  }

  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifecycle_custom_susceptibility::reset_counts();
    lifetime_counts counts;
    tracking_backend *tracking = new tracking_backend(f, counts, true, true, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    f.advance(1);
    const HostCustomFallbackStats &stats = tracking->host_custom_fallback_stats();
    CHECK(f.backend_state && f.executable && tracking->host_custom_fallback_enabled() &&
              lifecycle_custom_susceptibility::allocations > 0 && stats.warnings == 1 &&
              stats.preflights == 1 && stats.sessions == 2 && stats.callbacks == 4 &&
              stats.completed_sessions == 2 && stats.retryable_failures == 0 &&
              stats.poisoned_failures == 0 && stats.staging_allocations == 1 &&
              stats.staging_bytes == 64 && stats.downloads == 2 && stats.download_bytes == 64 &&
              stats.uploads == 2 && stats.upload_bytes == 96,
          "warned custom fallback did not publish exactly one complete first session");

    f.advance(1);
    CHECK(stats.warnings == 1 && stats.preflights == 2 && stats.sessions == 4 &&
              stats.callbacks == 8 && stats.completed_sessions == 4 &&
              stats.staging_allocations == 1 && stats.downloads == 4 && stats.uploads == 4,
          "steady custom fallback repeated its warning or lost session accounting");

    BackendState *const state = f.backend_state;
    Executable *const executable = f.executable;
    const int attempts = counts.advance_attempts;
    counts.fail_custom_preflight = true;
    bool rejected = false;
    try {
      f.advance(1);
    }
    catch (const std::runtime_error &) {
      rejected = true;
    }
    CHECK(rejected && f.backend_state == state && f.executable == executable &&
              counts.advance_attempts == attempts && !tracking->is_poisoned() &&
              stats.retryable_failures == 1,
          "custom pre-entry failure dispatched, replaced the epoch, or poisoned the backend");
    counts.fail_custom_preflight = false;
    f.advance(1);
    CHECK(stats.completed_sessions == 6 && counts.advance_attempts == attempts + 1,
          "custom pre-entry failure was not retryable");

    const int attempts_before_session_failure = counts.advance_attempts;
    const uint64_t callbacks_before_session_failure = stats.callbacks;
    counts.fail_custom_before_entry = true;
    rejected = false;
    try {
      f.advance(1);
    }
    catch (const std::runtime_error &) {
      rejected = true;
    }
    CHECK(rejected && !tracking->is_poisoned() && f.backend_state == state &&
              f.executable == executable &&
              counts.advance_attempts == attempts_before_session_failure + 1 &&
              stats.callbacks == callbacks_before_session_failure &&
              stats.retryable_failures == 2,
          "failure before custom callback entry was not retryable");
    counts.fail_custom_before_entry = false;
    f.advance(1);
    CHECK(stats.completed_sessions == 8,
          "custom session did not recover after a pre-callback failure");

    rejected = false;
    try {
      backend_preflight_host_custom_fallback(f, HostCustomFallbackUse::solve_cw,
                                             "backend_api custom solve_cw");
    }
    catch (const std::runtime_error &e) {
      rejected = std::string(e.what()).find("time-domain") != std::string::npos;
    }
    CHECK(rejected && !tracking->is_poisoned() && f.backend_state == state &&
              f.executable == executable && stats.warnings == 1,
          "custom solve_cw scope rejection changed the installed epoch");

    rejected = false;
    try {
      f.synchronize_magnetic_fields();
    }
    catch (const std::runtime_error &e) {
      rejected = std::string(e.what()).find("custom susceptibility") != std::string::npos;
    }
    CHECK(rejected && !tracking->is_poisoned() && f.backend_state == state &&
              f.executable == executable && counts.magnetic_synchronizes == 0,
          "custom magnetic synchronization entered a transition or changed the epoch");

    f.reset();
    f.advance(1);
    CHECK(stats.warnings == 1 && stats.completed_sessions == 10,
          "custom reset lost the installed policy or repeated its warning");

    const uint64_t preflights_before_remove = stats.preflights;
    const uint64_t sessions_before_remove = stats.sessions;
    f.remove_susceptibilities();
    f.advance(1);
    CHECK(!tracking->host_custom_fallback_enabled() && stats.warnings == 1 &&
              stats.preflights == preflights_before_remove &&
              stats.sessions == sessions_before_remove,
          "custom removal retained a fallback dispatch or repeated its warning");
  }

  /* fields has no public live add-susceptibility API symmetric with removal.
     Re-add is therefore a fresh fields construction; clone and backend
     reselection below pin the supported reconstruction paths. */
  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s, true);
    fields readded(&s);
    readded.require_component(Ez);
    lifetime_counts initial_counts;
    tracking_backend *initial = new tracking_backend(readded, initial_counts, false, false, true,
                                                     true);
    readded.backend = initial;
    readded.options = warned_custom_options();
    readded.advance(1);
    CHECK(initial->host_custom_fallback_stats().warnings == 1 &&
              initial->host_custom_fallback_stats().sessions == 2 &&
              initial->host_custom_fallback_stats().completed_sessions == 2,
          "fresh custom reconstruction did not restore the exact segment lifecycle");

    fields copied(readded);
    lifetime_counts copied_counts;
    tracking_backend *copied_backend =
        new tracking_backend(copied, copied_counts, false, false, true, true);
    copied.backend = copied_backend;
    copied.options = warned_custom_options();
    copied.advance(1);
    CHECK(copied_backend->host_custom_fallback_enabled() &&
              copied_backend->host_custom_fallback_stats().warnings == 1 &&
              copied_backend->host_custom_fallback_stats().sessions == 2 &&
              copied_backend->host_custom_fallback_stats().callbacks == 4 &&
              copied_backend->host_custom_fallback_stats().completed_sessions == 2,
          "fields clone did not rebuild the exact custom fallback session set");

    execution_options cpu;
    readded.select_backend(cpu);
    CHECK(!readded.backend_state && !readded.executable,
          "custom backend reselection retained the resident epoch");
    delete readded.backend;
    lifetime_counts reselection_counts;
    tracking_backend *reselected =
        new tracking_backend(readded, reselection_counts, false, false, true, true);
    readded.backend = reselected;
    readded.options = warned_custom_options();
    readded.advance(1);
    CHECK(reselected->host_custom_fallback_enabled() &&
              reselected->host_custom_fallback_stats().warnings == 1 &&
              reselected->host_custom_fallback_stats().sessions == 2 &&
              reselected->host_custom_fallback_stats().callbacks == 4 &&
              reselected->host_custom_fallback_stats().completed_sessions == 2,
          "backend reselection did not rebuild the exact custom fallback session set");
  }

  auto expect_poison = [](const char *label, bool fail_after_entry, bool reenter,
                          bool execute_session, bool fail_advance,
                          bool fail_later_before_entry, bool omit_last_segment,
                          bool undercount_last_segment) {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    counts.fail_custom_after_entry = fail_after_entry;
    counts.reenter_custom_callback = reenter;
    counts.fail_advance = fail_advance;
    counts.fail_custom_later_before_entry = fail_later_before_entry;
    counts.omit_custom_last_segment = omit_last_segment;
    counts.undercount_custom_last_segment = undercount_last_segment;
    tracking_backend *tracking =
        new tracking_backend(f, counts, false, false, true, execute_session);
    f.backend = tracking;
    f.options = warned_custom_options();
    bool rejected = false;
    try {
      f.advance(1);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    const HostCustomFallbackStats &stats = tracking->host_custom_fallback_stats();
    CHECK(rejected && tracking->is_poisoned() && stats.poisoned_failures == 1,
          "%s did not poison exactly once", label);
    if (fail_advance)
      CHECK(counts.advance_attempts == 1 && stats.callbacks == 0 &&
                stats.retryable_failures == 0,
            "%s was misclassified as a retryable pre-callback session failure", label);
    if (fail_later_before_entry)
      CHECK(counts.advance_attempts == 1 && stats.callbacks == 2 &&
                stats.completed_sessions == 1 && stats.retryable_failures == 0,
            "%s ignored the earlier irreversible callback boundary", label);
    const uint64_t callbacks = stats.callbacks;
    bool poison_rejected = false;
    try {
      f.advance(1);
    }
    catch (const std::exception &) {
      poison_rejected = true;
    }
    CHECK(poison_rejected && stats.callbacks == callbacks,
          "%s allowed a callback after poison", label);
  };

  expect_poison("generic resident advance failure with custom enabled", false, false, true, true,
                false, false, false);
  expect_poison("later segment pre-entry failure", false, false, true, false, true, false,
                false);
  expect_poison("post-entry custom callback failure", true, false, true, false, false, false,
                false);
  expect_poison("reentrant custom callback", false, true, true, false, false, false, false);
  expect_poison("omitted custom callback session", false, false, false, false, false, false,
                false);
  expect_poison("omitted later custom segment", false, false, true, false, false, true, false);
  expect_poison("partial custom segment callback count", false, false, true, false, false, false,
                true);

  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    counts.poison_custom_preflight = true;
    tracking_backend *tracking = new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    bool rejected = false;
    try {
      f.advance(1);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    CHECK(rejected && tracking->is_poisoned() && counts.advance_attempts == 0 &&
              tracking->host_custom_fallback_stats().poisoned_failures == 1,
          "self-poisoning custom preflight published or dispatched");
  }
}

static void test_resident_host_custom_collective_preflight() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);

  /* The public policy still rejects multi-rank fallback before any legacy or
     resident allocation. The explicit override below exists only to exercise
     later collective boundaries that are otherwise unreachable by contract. */
  if (count_processors() > 1) {
    for (int target = 0; target < count_processors(); ++target) {
      structure s(gv, unit_epsilon, no_pml(), identity(), 2);
      add_custom_lifecycle_state(s, true);
      fields f(&s);
      f.require_component(Ez);
      lifecycle_custom_susceptibility::reset_counts();
      lifetime_counts counts;
      f.backend = new tracking_backend(f, counts, false, false, true, true);
      f.options = warned_custom_options();
      if (my_rank() == target) f.options.strict = true;
      backend_set_host_custom_mpi_override_for_testing(true);
      bool failed = false;
      try { f.advance(1); }
      catch (const std::runtime_error &) { failed = true; }
      backend_set_host_custom_mpi_override_for_testing(false);
      CHECK(and_to_all(failed) && !f.backend_state && !f.executable &&
                counts.states_created == 0 && counts.advance_attempts == 0 &&
                lifecycle_custom_susceptibility::allocations == 0 &&
                lifecycle_custom_susceptibility::initializations == 0 &&
                lifecycle_custom_susceptibility::layout_queries == 0 &&
                !f.backend->host_custom_fallback_enabled() && !f.backend->is_poisoned(),
            "rank-asymmetric custom policy failure crossed allocation or callback");
    }

    for (int target = 0; target < count_processors(); ++target) {
      structure s(gv, unit_epsilon, no_pml(), identity(), 2);
      add_custom_lifecycle_state(s, true);
      fields f(&s);
      f.require_component(Ez);
      lifecycle_custom_susceptibility::reset_counts();
      lifetime_counts counts;
      tracking_backend *tracking =
          new tracking_backend(f, counts, false, false, true, true);
      f.backend = tracking;
      f.options = warned_custom_options();
      if (my_rank() == target)
        backend_set_host_custom_counter_for_testing(
            *tracking, HostCustomFallbackCounter::warnings,
            std::numeric_limits<uint64_t>::max());
      backend_set_host_custom_mpi_override_for_testing(true);
      bool failed = false;
      try { f.advance(1); }
      catch (const std::runtime_error &) { failed = true; }
      backend_set_host_custom_mpi_override_for_testing(false);
      CHECK(and_to_all(failed) && !f.backend_state && !f.executable &&
                counts.states_created == 0 && counts.advance_attempts == 0 &&
                !f.backend->host_custom_fallback_enabled() && !f.backend->is_poisoned(),
            "rank-asymmetric custom policy publication overflow was not atomic");
    }

    for (int target = 0; target < count_processors(); ++target)
      for (int mode = 1; mode <= 3; ++mode) {
        structure s(gv, unit_epsilon, no_pml(), identity(), 2);
        add_custom_lifecycle_state(s, true);
        fields f(&s);
        f.require_component(Ez);
        lifecycle_custom_susceptibility::reset_counts();
        lifetime_counts counts;
        f.backend = new tracking_backend(f, counts, false, false, true, true);
        f.options = warned_custom_options();
        backend_set_host_custom_mpi_override_for_testing(true);
        backend_set_host_custom_collective_failure_for_testing(target, mode);
        bool failed = false;
        try { f.advance(1); }
        catch (const std::runtime_error &) { failed = true; }
        backend_set_host_custom_collective_failure_for_testing(-1, 0);
        backend_set_host_custom_mpi_override_for_testing(false);
        CHECK(and_to_all(failed) && !f.backend_state && !f.executable &&
                  counts.states_created == 0 && counts.advance_attempts == 0 &&
                  lifecycle_custom_susceptibility::allocations == 0 &&
                  lifecycle_custom_susceptibility::initializations == 0 &&
                  lifecycle_custom_susceptibility::layout_queries == 0 &&
                  !f.backend->host_custom_fallback_enabled() && !f.backend->is_poisoned(),
              "rank-asymmetric custom identity/range/rebuild failure published partial state");
      }

    for (int target = 0; target < count_processors(); ++target) {
      structure s(gv, unit_epsilon, no_pml(), identity(), 2);
      add_custom_lifecycle_state(s);
      fields f(&s);
      f.require_component(Ez);
      lifetime_counts counts;
      f.backend = new tracking_backend(f, counts, false, false, true, true);
      f.options = warned_custom_options();
      backend_set_host_custom_mpi_override_for_testing(true);
      backend_set_host_custom_collective_failure_for_testing(target, 4);
      bool failed = false;
      try { f.advance(1); }
      catch (const std::runtime_error &) { failed = true; }
      backend_set_host_custom_collective_failure_for_testing(-1, 0);
      CHECK(and_to_all(failed) && !f.backend_state && !f.executable &&
                counts.states_created == 1 && counts.states_destroyed == 1 &&
                counts.advance_attempts == 0 &&
                f.backend->host_custom_fallback_stats().callbacks == 0 &&
                !f.backend->is_poisoned(),
            "rank-asymmetric custom plan readiness failure published or dispatched");
      f.advance(1);
      CHECK(f.executable && f.backend_state->host_custom_plan_validated &&
                counts.advance_attempts == 1,
            "custom plan readiness failure was not retryable");
      backend_set_host_custom_mpi_override_for_testing(false);
    }
  }

  backend_set_host_custom_mpi_override_for_testing(count_processors() > 1);

  /* Policy and its one-shot warning belong to the committed resident epoch,
     not merely to the early custom capability reconciliation. Exercise every
     fallible step after that gate, including ordinary executable compilation,
     for addition and removal. */
  for (int injection = 0; injection < 4; ++injection) {
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking =
        new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    const HostCustomFallbackStats entry_stats = tracking->host_custom_fallback_stats();
    counts.fail_create_state = injection == 0;
    counts.fail_initialize = injection == 1;
    counts.fail_finalize = injection == 2;
    counts.fail_compile = injection == 3;
    bool failed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    CHECK(and_to_all(failed) && !tracking->host_custom_fallback_enabled() &&
              same_host_custom_stats(entry_stats, tracking->host_custom_fallback_stats()) &&
              !tracking->is_poisoned(),
          "custom addition rebuild failure %d published policy, warning, stats, or poison",
          injection);
    if (injection == 3)
      CHECK(!f.backend_state && !f.executable && is_dirty(f, dirty_executable),
            "custom addition compile failure published a candidate epoch");
    counts.fail_create_state = false;
    counts.fail_initialize = false;
    counts.fail_finalize = false;
    counts.fail_compile = false;
    f.advance(1);
    const bool any_enabled = or_to_all(tracking->host_custom_fallback_enabled());
    CHECK(f.backend_state->host_custom_preflight_required &&
              tracking->host_custom_fallback_enabled() ==
                  f.backend_state->host_custom_local_presence &&
              any_enabled && !f.backend_state->host_custom_policy_pending &&
              tracking->host_custom_fallback_stats().warnings == 1,
          "custom addition rebuild failure %d did not publish on retry", injection);
  }

  for (int injection = 0; injection < 4; ++injection) {
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking =
        new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    f.advance(1);
    const bool entry_enabled = tracking->host_custom_fallback_enabled();
    const HostCustomFallbackStats entry_stats = tracking->host_custom_fallback_stats();
    f.remove_susceptibilities();
    counts.fail_create_state = injection == 0;
    counts.fail_initialize = injection == 1;
    counts.fail_finalize = injection == 2;
    counts.fail_compile = injection == 3;
    bool failed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    CHECK(and_to_all(failed) &&
              tracking->host_custom_fallback_enabled() == entry_enabled &&
              same_host_custom_stats(entry_stats, tracking->host_custom_fallback_stats()) &&
              !tracking->is_poisoned(),
          "custom removal rebuild failure %d replaced the committed policy or stats",
          injection);
    if (injection == 3)
      CHECK(!f.backend_state && !f.executable && is_dirty(f, dirty_executable),
            "custom removal compile failure published a candidate epoch");
    counts.fail_create_state = false;
    counts.fail_initialize = false;
    counts.fail_finalize = false;
    counts.fail_compile = false;
    f.advance(1);
    CHECK(!tracking->host_custom_fallback_enabled() &&
              !f.backend_state->host_custom_policy_pending &&
              same_host_custom_stats(entry_stats, tracking->host_custom_fallback_stats()),
          "custom removal rebuild failure %d did not publish removal on retry", injection);
  }

  {
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking =
        new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    backend_reset_host_custom_collective_count_for_testing();
    f.advance(1);
    CHECK(f.backend_state && f.executable &&
              f.backend_state->host_custom_preflight_required &&
              f.backend_state->host_custom_plan_validated,
          "collective custom rebuild did not publish validated presence");
    const size_t first_collectives = backend_host_custom_collective_count_for_testing();
    BackendState *const state = f.backend_state;
    Executable *const executable = f.executable;
    const int attempts = counts.advance_attempts;
    f.advance(1);
    CHECK(f.backend_state == state && f.executable == executable &&
              counts.advance_attempts == attempts + 1 &&
              backend_host_custom_collective_count_for_testing() == first_collectives + 2,
          "steady custom dispatch rebuilt state or missed prepare/dispatch reconciliation");

    /* One owner rejects staging after the executable exists. Every successful
       peer must discard its pending marker, and the installed epoch remains
       retryable without a callback or dispatch. */
    const int staging_target = 0;
    CHECK(my_rank() != staging_target || tracking->host_custom_fallback_enabled(),
          "custom staging failure target does not own a callback segment");
    counts.fail_custom_preflight = my_rank() == staging_target;
    const int attempts_before_staging = counts.advance_attempts;
    const uint64_t callbacks_before_staging =
        tracking->host_custom_fallback_stats().callbacks;
    bool staging_failed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { staging_failed = true; }
    CHECK(and_to_all(staging_failed) && f.backend_state == state && f.executable == executable &&
              counts.advance_attempts == attempts_before_staging &&
              tracking->host_custom_fallback_stats().callbacks == callbacks_before_staging &&
              !f.backend->is_poisoned(),
          "asymmetric custom staging failure dispatched, replaced, or poisoned the epoch");
    counts.fail_custom_preflight = false;
    f.advance(1);

    /* A callback failure on an owner, or a generic resident failure on an idle
       rank, poisons and rejects every participant in this dispatch. */
    for (int target = 0; target < count_processors(); ++target) {
      structure ps(gv, unit_epsilon, no_pml(), identity(), 2);
      add_custom_lifecycle_state(ps);
      fields poisoned(&ps);
      poisoned.require_component(Ez);
      lifetime_counts poison_counts;
      tracking_backend *poison_tracking =
          new tracking_backend(poisoned, poison_counts, false, false, true, true);
      poisoned.backend = poison_tracking;
      poisoned.options = warned_custom_options();
      poisoned.advance(1);
      const bool local_callback_owner = poison_tracking->host_custom_fallback_enabled();
      poison_counts.fail_custom_after_entry = local_callback_owner && my_rank() == target;
      poison_counts.fail_advance = !local_callback_owner && my_rank() == target;
      bool failed = false;
      try { poisoned.advance(1); }
      catch (const std::runtime_error &) { failed = true; }
      CHECK(and_to_all(failed) && poisoned.backend->is_poisoned(),
            "rank-asymmetric custom postdispatch failure did not poison every rank");
    }

    if (count_processors() > 1) {
      structure ps(gv, unit_epsilon, no_pml(), identity(), 2);
      add_custom_lifecycle_state(ps);
      fields poisoned(&ps);
      poisoned.require_component(Ez);
      lifetime_counts poison_counts;
      tracking_backend *poison_tracking =
          new tracking_backend(poisoned, poison_counts, false, false, true, true);
      poisoned.backend = poison_tracking;
      poisoned.options = warned_custom_options();
      poisoned.advance(1);
      const bool local_owner = poison_tracking->host_custom_fallback_enabled();
      const bool target_owner = and_to_all(my_rank() != 0 || local_owner);
      const bool peer_owner = or_to_all(local_owner && my_rank() != 0);
      poison_counts.fail_custom_before_entry = local_owner && my_rank() == 0;
      bool failed = false;
      try { poisoned.advance(1); }
      catch (const std::runtime_error &) { failed = true; }
      CHECK(target_owner && peer_owner && and_to_all(failed) &&
                poisoned.backend->is_poisoned(),
            "peer callback completion did not poison a collectively failed pre-entry dispatch");
    }

    f.remove_susceptibilities();
    f.advance(1);
    CHECK(!f.backend_state->host_custom_preflight_required &&
              !f.backend_state->host_custom_plan_validated &&
              !tracking->host_custom_fallback_enabled(),
          "removing the final custom state retained collective presence");
    backend_reset_host_custom_collective_count_for_testing();
    f.advance(1);
    CHECK(backend_host_custom_collective_count_for_testing() == 0,
          "post-removal steady path retained a custom-specific collective");
  }
  backend_set_host_custom_mpi_override_for_testing(false);

  /* A resident plan that never contains a custom state must not pay a custom
     validation or dispatch collective, including on clean steady steps. */
  {
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts);
    backend_reset_host_custom_collective_count_for_testing();
    f.advance(1);
    backend_reset_host_custom_presence_scan_count_for_testing();
    f.advance(1);
    CHECK(backend_host_custom_collective_count_for_testing() == 0 &&
              backend_host_custom_presence_scan_count_for_testing() == 0 &&
              !f.backend_state->host_custom_preflight_required &&
              !f.backend_state->host_custom_plan_validated,
          "never-custom steady path entered custom collective validation or rescanned states");
  }
}

static void test_resident_host_custom_split_communicator() {
  const int groups = count_processors();
  divide_parallel_processes(groups);
  CHECK(my_rank() == 0, "one-rank custom split communicator did not have local rank zero");
  {
    const grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 1);
    add_custom_lifecycle_state(s);
    fields f(&s);
    f.require_component(Ez);
    lifetime_counts counts;
    tracking_backend *tracking =
        new tracking_backend(f, counts, false, false, true, true);
    f.backend = tracking;
    f.options = warned_custom_options();
    backend_reset_host_custom_collective_count_for_testing();
    f.advance(1);
    const size_t rebuild_collectives = backend_host_custom_collective_count_for_testing();
    f.advance(1);
    CHECK(f.backend_state && f.executable &&
              f.backend_state->host_custom_preflight_required &&
              f.backend_state->host_custom_plan_validated && counts.advance_attempts == 2 &&
              backend_host_custom_collective_count_for_testing() == rebuild_collectives + 2,
          "split-communicator custom validation used the wrong communicator");
  }
  end_divide_parallel();
}

static void expect_collective_multilevel_static_failure(fields &f, lifetime_counts &counts,
                                                        const char *message) {
  BackendState *const state = f.backend_state;
  Executable *const executable = f.executable;
  StepPlan *const plan = f.step_plans[0];
  StoragePlan *const storage = f.storage_plan;
  CpuArrayCatalog *const catalog = f.array_catalog;
  const DirtyMask dirty = DirtyMask(f.dirty_mask);
  const int entry_t = f.t;
  const int dispatches = counts.advance_attempts;
  bool failed = false;
  try { f.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  CHECK(and_to_all(failed) && f.backend_state == state && f.executable == executable &&
            f.step_plans[0] == plan && f.storage_plan == storage &&
            f.array_catalog == catalog && DirtyMask(f.dirty_mask) == dirty && f.t == entry_t &&
            counts.advance_attempts == dispatches && !f.backend->is_poisoned(),
        message);
}

static void test_resident_multilevel_collective_preflight() {
  const grid_volume gv = vol2d(3.0, 3.0, 10.0);

  /* Rank-local recipe validation and allocation failures must be reconciled
     before prepare_storage reaches the legacy allocator/LAPACK path.  Two
     chunks deliberately leave idle ranks when this runs at np4. */
  for (int target = 0; target < count_processors(); ++target)
    for (int mode = 1; mode <= 2; ++mode) {
      structure s(gv, unit_epsilon, no_pml(), identity(), 2);
      add_multilevel_lifecycle_states(s);
      fields f(&s);
      f.use_real_fields();
      f.require_component(Ez);
      f.require_component(Hz);
      lifetime_counts counts;
      f.backend = new tracking_backend(f, counts);
      CpuArrayCatalog *const catalog = f.array_catalog;
      StoragePlan *const storage = f.storage_plan;
      DescriptorSet *const descriptors = f.descriptors;
      StepPlan *const plan = f.step_plans[0];
      const DirtyMask dirty = DirtyMask(f.dirty_mask);
      backend_set_multilevel_preflight_failure_for_testing(target, mode);
      bool failed = false;
      try { f.advance(1); }
      catch (const std::runtime_error &) { failed = true; }
      backend_set_multilevel_preflight_failure_for_testing(-1, 0);
      bool unallocated = true;
      for (int chunk = 0; chunk < f.num_chunks; ++chunk)
        if (f.chunks[chunk]->is_mine())
          FOR_FIELD_TYPES(ft)
            for (polarization_state *state = f.chunks[chunk]->pol[ft]; state;
                 state = state->next)
              unallocated = unallocated && !state->data;
      CHECK(and_to_all(failed) && !f.backend_state && !f.executable &&
                f.array_catalog == catalog && f.storage_plan == storage &&
                f.descriptors == descriptors && f.step_plans[0] == plan &&
                DirtyMask(f.dirty_mask) == dirty && counts.states_created == 0 &&
                counts.advance_attempts == 0 && and_to_all(unallocated) &&
                !f.backend->is_poisoned(),
            "rank-asymmetric multilevel recipe preflight failure published partial state");
      f.advance(1);
      CHECK(f.backend_state && f.executable && counts.advance_attempts == 1 &&
                !f.backend->is_poisoned(),
            "multilevel recipe preflight failure was not retryable");
    }

  for (int malformed = 0; malformed < 8; ++malformed) {
    structure invalid_structure(gv, unit_epsilon, no_pml(), identity(), 2);
    realnum gamma[] = {realnum(0.02), 0, 0, realnum(0.03)};
    realnum n0[] = {realnum(0.8), realnum(0.2)};
    realnum alpha[] = {realnum(-0.4), realnum(0.5)};
    realnum omega[] = {realnum(0.63)};
    realnum damping[] = {realnum(0.04)};
    realnum sigmat[] = {1, 1, 1, 1, 1};
    if (malformed == 0) gamma[1] = std::numeric_limits<realnum>::quiet_NaN();
    if (malformed == 1) n0[0] = std::numeric_limits<realnum>::infinity();
    if (malformed == 2) alpha[0] = std::numeric_limits<realnum>::quiet_NaN();
    if (malformed == 3) alpha[0] = realnum(0.4);
    if (malformed == 4) omega[0] = std::numeric_limits<realnum>::infinity();
    if (malformed == 5) damping[0] = std::numeric_limits<realnum>::quiet_NaN();
    if (malformed == 6) sigmat[3] = std::numeric_limits<realnum>::infinity();
    if (malformed == 7) gamma[0] = realnum(-2.0 / invalid_structure.dt);
    multilevel_susceptibility invalid(2, 1, gamma, n0, alpha, omega, damping, sigmat);
    invalid_structure.add_susceptibility(unit_epsilon, E_stuff, invalid);
    fields rejected(&invalid_structure);
    rejected.use_real_fields();
    rejected.require_component(Ez);
    lifetime_counts rejected_counts;
    rejected.backend = new tracking_backend(rejected, rejected_counts);
    bool failed = false;
    try { rejected.advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    bool unallocated = true;
    for (int chunk = 0; chunk < rejected.num_chunks; ++chunk)
      if (rejected.chunks[chunk]->is_mine())
        for (polarization_state *state = rejected.chunks[chunk]->pol[E_stuff]; state;
             state = state->next)
          unallocated = unallocated && !state->data;
    CHECK(and_to_all(failed) && !rejected.backend_state && !rejected.executable &&
              rejected_counts.states_created == 0 && rejected_counts.advance_attempts == 0 &&
              and_to_all(unallocated) && !rejected.backend->is_poisoned(),
          "malformed live multilevel recipe reached allocation or dispatch");
  }

  {
    structure cylindrical_structure(volcyl(2.0, 3.0, 8.0), unit_epsilon, no_pml(),
                                     identity(), 2);
    add_multilevel_lifecycle_states(cylindrical_structure);
    fields cylindrical(&cylindrical_structure);
    cylindrical.use_real_fields();
    cylindrical.require_component(Ez);
    lifetime_counts cylindrical_counts;
    cylindrical.backend = new tracking_backend(cylindrical, cylindrical_counts);
    bool failed = false;
    try { cylindrical.advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    CHECK(and_to_all(failed) && !cylindrical.backend_state && !cylindrical.executable &&
              cylindrical_counts.states_created == 0 &&
              cylindrical_counts.advance_attempts == 0 &&
              !cylindrical.backend->is_poisoned(),
          "unsupported cylindrical multilevel recipe crossed allocation or dispatch");
  }

  structure s(gv, unit_epsilon, no_pml(), identity(), 2);
  add_multilevel_lifecycle_states(s);
  fields f(&s);
  f.use_real_fields();
  f.require_component(Ez);
  f.require_component(Hz);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  backend_reset_multilevel_collective_count_for_testing();
  f.advance(1);
  CHECK(f.backend_state->multilevel_preflight_required &&
            f.backend_state->multilevel_plan_validated &&
            !f.backend_state->multilevel_static_validation_required,
        "valid multilevel plan did not publish cached validation state");
  const bool local_multilevel = has_local_exact_multilevel(f);
  CHECK(or_to_all(local_multilevel), "collective multilevel fixture has no live exact state");
  CHECK(local_multilevel || (f.step_plans[0]->multilevel_population_updates.empty() &&
                             f.step_plans[0]->multilevel_transition_updates.empty()),
        "idle multilevel rank published local update rows");
  const size_t collectives_after_build = backend_multilevel_collective_count_for_testing();
  BackendState *const stable_state = f.backend_state;
  Executable *const stable_executable = f.executable;
  const int compiles_after_build = counts.executables_created;
  f.advance(1);
  CHECK(f.backend_state == stable_state && f.executable == stable_executable &&
            counts.executables_created == compiles_after_build &&
            backend_multilevel_collective_count_for_testing() == collectives_after_build,
        "steady multilevel advance rebuilt state or ran a static-validation collective");

  for (int target = 0; target < count_processors(); ++target)
    for (int mode = 3; mode <= 4; ++mode) {
      f.backend_state->multilevel_static_validation_required = true;
      backend_set_multilevel_preflight_failure_for_testing(target, mode);
      expect_collective_multilevel_static_failure(
          f, counts, "rank-asymmetric multilevel static failure crossed dispatch");
      backend_set_multilevel_preflight_failure_for_testing(-1, 0);
      f.advance(1);
      CHECK(!f.backend->is_poisoned(),
            "rank-asymmetric multilevel static failure was not retryable");
    }

  const StepPlan stable_plan = *f.step_plans[0];
  if (my_rank() == 0) f.step_plans[0]->signature ^= UINT64_C(1);
  expect_collective_multilevel_static_failure(
      f, counts, "stale multilevel plan signature crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  bool changed_group = false;
  if (my_rank() == 0)
    for (Operation &op : f.step_plans[0]->operations)
      if (!changed_group && op.kind == OpKind::update_polarization &&
          op.polarization_group_count) {
        --op.polarization_group_count;
        changed_group = true;
      }
  CHECK(or_to_all(changed_group), "multilevel operation mutation found no group span");
  if (my_rank() == 0)
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed multilevel operation group span crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool changed_access = false;
  if (my_rank() == 0)
    for (Operation &op : f.step_plans[0]->operations)
      if (!changed_access && op.kind == OpKind::update_polarization &&
          op.polarization_group_count && !op.accesses.empty()) {
        op.accesses.pop_back();
        f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
        changed_access = true;
      }
  CHECK(or_to_all(changed_access), "multilevel operation mutation found no access set");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed multilevel operation access mutation crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool removed_action = false;
  if (my_rank() == 0 && !f.step_plans[0]->multilevel_population_updates.empty()) {
    f.step_plans[0]->multilevel_population_updates.erase(
        f.step_plans[0]->multilevel_population_updates.begin());
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    removed_action = true;
  }
  CHECK(or_to_all(removed_action), "multilevel missing-action mutation found no population row");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed missing multilevel action crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool added_action = false;
  if (my_rank() == 0 && !f.step_plans[0]->multilevel_population_updates.empty()) {
    f.step_plans[0]->multilevel_population_updates.push_back(
        f.step_plans[0]->multilevel_population_updates.back());
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    added_action = true;
  }
  CHECK(or_to_all(added_action), "multilevel extra-action mutation found no population row");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed extra multilevel action crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool reordered_actions = false;
  if (my_rank() == 0 && f.step_plans[0]->multilevel_population_updates.size() > 1) {
    std::swap(f.step_plans[0]->multilevel_population_updates[0],
              f.step_plans[0]->multilevel_population_updates[1]);
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    reordered_actions = true;
  }
  CHECK(or_to_all(reordered_actions),
        "multilevel reordered-action mutation found fewer than two population rows");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed reordered multilevel actions crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool changed_action = false;
  if (my_rank() == 0 && !f.step_plans[0]->multilevel_population_updates.empty()) {
    ++f.step_plans[0]->multilevel_population_updates[0].levels;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    changed_action = true;
  }
  CHECK(or_to_all(changed_action), "multilevel action mutation found no population row");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed multilevel action mutation crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool changed_array_id = false;
  if (my_rank() == 0 && !f.step_plans[0]->multilevel_population_updates.empty()) {
    f.step_plans[0]->multilevel_population_updates[0].gamma_inv =
        ArrayId{std::numeric_limits<uint32_t>::max()};
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    changed_array_id = true;
  }
  CHECK(or_to_all(changed_array_id), "multilevel ArrayId mutation found no population row");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed out-of-range multilevel ArrayId crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool changed_region = false;
  if (my_rank() == 0 && !f.step_plans[0]->multilevel_population_updates.empty()) {
    ++f.step_plans[0]->multilevel_population_updates[0].region.base;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    changed_region = true;
  }
  CHECK(or_to_all(changed_region), "multilevel range mutation found no population row");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed multilevel region range crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool changed_offset = false;
  if (my_rank() == 0 && !f.step_plans[0]->multilevel_population_terms.empty()) {
    ++f.step_plans[0]->multilevel_population_terms[0].centered_offsets[0];
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    changed_offset = true;
  }
  CHECK(or_to_all(changed_offset), "multilevel offset mutation found no population term");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed multilevel centered offset crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool changed_scratch = false;
  if (my_rank() == 0 && !f.step_plans[0]->multilevel_population_updates.empty()) {
    f.step_plans[0]->multilevel_population_updates[0].scratch_elements_per_point =
        std::numeric_limits<size_t>::max();
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    changed_scratch = true;
  }
  CHECK(or_to_all(changed_scratch), "multilevel scratch mutation found no population row");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed multilevel scratch budget crossed dispatch");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0) {
    ++f.mutation_generation[static_cast<int>(MutationKind::coordinate_definition)];
    f.backend_state->multilevel_static_validation_required = true;
  }
  expect_collective_multilevel_static_failure(
      f, counts, "stale multilevel topology generation crossed dispatch");
  if (my_rank() == 0)
    --f.mutation_generation[static_cast<int>(MutationKind::coordinate_definition)];

  bool changed_term = false;
  if (my_rank() == 0 && !f.step_plans[0]->multilevel_population_terms.empty()) {
    ++f.step_plans[0]->multilevel_population_terms[0].transition_index;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    changed_term = true;
  }
  CHECK(or_to_all(changed_term), "multilevel term mutation found no population term");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed multilevel population term crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool changed_transition = false;
  if (my_rank() == 0 && !f.step_plans[0]->multilevel_transition_updates.empty()) {
    ++f.step_plans[0]->multilevel_transition_updates[0].positive_level;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    changed_transition = true;
  }
  CHECK(or_to_all(changed_transition), "multilevel transition mutation found no update row");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed multilevel transition action crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool changed_coefficient = false;
  if (my_rank() == 0 && !f.step_plans[0]->multilevel_coefficients.empty()) {
    f.step_plans[0]->multilevel_coefficients[0] += 0.125;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    changed_coefficient = true;
  }
  CHECK(or_to_all(changed_coefficient), "multilevel coefficient mutation found no coefficient");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed multilevel coefficient mutation crossed dispatch");
  *f.step_plans[0] = stable_plan;

  bool changed_subtraction = false;
  if (my_rank() == 0 && !f.step_plans[0]->polarization_subtractions.empty()) {
    ++f.step_plans[0]->polarization_subtractions[0].transition_index;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
    changed_subtraction = true;
  }
  CHECK(or_to_all(changed_subtraction),
        "multilevel subtraction mutation found no subtraction row");
  expect_collective_multilevel_static_failure(
      f, counts, "re-signed multilevel subtraction crossed dispatch");
  *f.step_plans[0] = stable_plan;

  const DescriptorSet stable_descriptors = *f.descriptors;
  bool changed_descriptor = false;
  if (my_rank() == 0)
    for (PolarizationDescriptor &descriptor : f.descriptors->polarizations)
      if (!changed_descriptor && descriptor.kind == SusceptibilityKind::multilevel &&
          !descriptor.multilevel.gamma_matrix.empty()) {
        descriptor.multilevel.gamma_matrix[0] += 0.125;
        changed_descriptor = true;
      }
  CHECK(or_to_all(changed_descriptor), "multilevel descriptor mutation found no exact state");
  f.backend_state->multilevel_static_validation_required = true;
  expect_collective_multilevel_static_failure(
      f, counts, "multilevel descriptor mutation crossed the dispatch boundary");
  *f.descriptors = stable_descriptors;

  ArrayId gamma_id = invalid_array();
  if (my_rank() == 0)
    for (const PolarizationDescriptor &descriptor : f.descriptors->polarizations)
      if (descriptor.kind == SusceptibilityKind::multilevel) {
        gamma_id = descriptor.multilevel_gamma_inv;
        break;
      }
  bool changed_catalog_role = false;
  array_role saved_role = array_role::polarization;
  if (my_rank() == 0 && is_valid(gamma_id)) {
    ArraySpec &spec = const_cast<ArraySpec &>(f.array_catalog->spec(gamma_id));
    saved_role = spec.role;
    spec.role = array_role::field;
    f.backend_state->multilevel_static_validation_required = true;
    changed_catalog_role = true;
  }
  CHECK(or_to_all(changed_catalog_role), "multilevel catalog mutation found no GammaInv row");
  expect_collective_multilevel_static_failure(
      f, counts, "multilevel catalog-role mutation crossed dispatch");
  if (my_rank() == 0 && is_valid(gamma_id))
    const_cast<ArraySpec &>(f.array_catalog->spec(gamma_id)).role = saved_role;

  bool changed_catalog_extent = false;
  size_t saved_elements = 0;
  if (my_rank() == 0 && is_valid(gamma_id)) {
    ArraySpec &spec = const_cast<ArraySpec &>(f.array_catalog->spec(gamma_id));
    saved_elements = spec.elements;
    spec.elements = 0;
    f.backend_state->multilevel_static_validation_required = true;
    changed_catalog_extent = true;
  }
  CHECK(or_to_all(changed_catalog_extent), "multilevel catalog extent found no GammaInv row");
  expect_collective_multilevel_static_failure(
      f, counts, "multilevel catalog-extent mutation crossed dispatch");
  if (my_rank() == 0 && is_valid(gamma_id))
    const_cast<ArraySpec &>(f.array_catalog->spec(gamma_id)).elements = saved_elements;

  bool changed_gamma_value = false;
  realnum saved_gamma = 0;
  if (my_rank() == 0 && is_valid(gamma_id)) {
    realnum *gamma = f.array_catalog->resolve<realnum>(gamma_id);
    saved_gamma = gamma[0];
    gamma[0] = std::numeric_limits<realnum>::infinity();
    f.backend_state->multilevel_static_validation_required = true;
    changed_gamma_value = true;
  }
  CHECK(or_to_all(changed_gamma_value), "multilevel GammaInv mutation found no authoritative row");
  expect_collective_multilevel_static_failure(
      f, counts, "nonfinite authoritative multilevel GammaInv crossed dispatch");
  if (my_rank() == 0 && is_valid(gamma_id))
    f.array_catalog->resolve<realnum>(gamma_id)[0] = saved_gamma;
  f.advance(1);

  f.add_flux_vol(X, volume(vec(0.1, -0.8), vec(0.1, 0.8)));
  f.advance(1);
  CHECK(f.backend_state == stable_state && f.backend_state->multilevel_plan_validated,
        "legacy-flux refresh lost validated multilevel state");
  const StepPlan stable_flux_plan = *f.step_plans[0];
  DescriptorSet *const flux_descriptors = f.descriptors;
  Executable *const flux_executable = f.executable;
  changed_group = false;
  if (my_rank() == 0)
    for (Operation &op : f.step_plans[0]->operations)
      if (!changed_group && op.kind == OpKind::update_polarization &&
          op.polarization_group_count) {
        --op.polarization_group_count;
        f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
        changed_group = true;
      }
  CHECK(or_to_all(changed_group), "flux graft mutation found no multilevel group span");
  f.add_flux_vol(Y, volume(vec(-0.8, -0.2), vec(0.8, -0.2)));
  const int dispatches_before_flux_failure = counts.advance_attempts;
  bool flux_failed = false;
  try { f.advance(1); }
  catch (const std::runtime_error &) { flux_failed = true; }
  CHECK(and_to_all(flux_failed) && f.backend_state == stable_state &&
            f.descriptors == flux_descriptors && f.executable == flux_executable &&
            counts.advance_attempts == dispatches_before_flux_failure &&
            is_dirty(f, dirty_flux_plan) && !f.backend->is_poisoned(),
        "invalid multilevel flux graft was published or dispatched");
  *f.step_plans[0] = stable_flux_plan;
  f.advance(1);
  CHECK(f.backend_state == stable_state && f.backend_state->multilevel_plan_validated,
        "multilevel flux graft failure was not retryable");

  f.remove_susceptibilities();
  f.advance(1);
  CHECK(!f.backend_state->multilevel_preflight_required &&
            !f.backend_state->multilevel_plan_validated &&
            !f.backend_state->multilevel_static_validation_required,
        "removing the last multilevel state retained cached presence or validation");
  backend_reset_multilevel_collective_count_for_testing();
  f.advance(1);
  CHECK(backend_multilevel_collective_count_for_testing() == 0,
        "post-removal steady path retained a multilevel-specific collective");

  /* A never-multilevel resident plan must not enter any multilevel-specific
     validation or post-dispatch collective. */
  structure plain_structure(gv, unit_epsilon, no_pml(), identity(), 2);
  fields plain(&plain_structure);
  plain.require_component(Ez);
  lifetime_counts plain_counts;
  plain.backend = new tracking_backend(plain, plain_counts);
  backend_reset_multilevel_collective_count_for_testing();
  plain.advance(1);
  plain.advance(1);
  CHECK(backend_multilevel_collective_count_for_testing() == 0 &&
            !plain.backend_state->multilevel_preflight_required &&
            !plain.backend_state->multilevel_plan_validated,
        "never-multilevel resident path entered multilevel collective validation");

  /* A post-dispatch failure poisons its local backend without imposing a
     clean-step collective.  The next public entry reduction propagates that
     poison to every owner and idle rank before another dispatch can begin. */
  for (int target = 0; target < count_processors(); ++target) {
    structure poison_structure(gv, unit_epsilon, no_pml(), identity(), 2);
    add_multilevel_lifecycle_states(poison_structure);
    fields poisoned(&poison_structure);
    poisoned.use_real_fields();
    poisoned.require_component(Ez);
    poisoned.require_component(Hz);
    lifetime_counts poison_counts;
    poisoned.backend = new tracking_backend(poisoned, poison_counts);
    poisoned.advance(1);
    poison_counts.fail_advance = my_rank() == target;
    const int attempts = poison_counts.advance_attempts;
    bool failed = false;
    try { poisoned.advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    CHECK(sum_to_all(int(failed)) == 1 &&
              poisoned.backend->is_poisoned() == (my_rank() == target) &&
              poison_counts.advance_attempts == attempts + 1,
          "rank-asymmetric multilevel dispatch failure did not poison its local backend");
    const int attempts_after_failure = poison_counts.advance_attempts;
    bool propagated = false;
    try { poisoned.advance(1); }
    catch (const std::runtime_error &) { propagated = true; }
    CHECK(and_to_all(propagated) && poisoned.backend->is_poisoned() &&
              poison_counts.advance_attempts == attempts_after_failure,
          "multilevel post-dispatch poison was not propagated before the next dispatch");
  }

  std::string error;
  const std::vector<double> identity_gamma = {0.0, 0.0, 0.0, 0.0};
  CHECK(!preflight_multilevel_internal_data(identity_gamma, 0, 1, 1, 1, realnum(0.1), error),
        "zero-level multilevel allocation preflight was accepted");
  CHECK(!preflight_multilevel_internal_data(identity_gamma, 2, 0, 1, 1, realnum(0.1), error),
        "zero-transition multilevel allocation preflight was accepted");
  CHECK(!preflight_multilevel_internal_data(std::vector<double>(3, 0.0), 2, 1, 1, 1,
                                             realnum(0.1), error),
        "mismatched multilevel Gamma extent was accepted");
  CHECK(!preflight_multilevel_internal_data(identity_gamma, 2, 1, 1, 1, realnum(0), error),
        "zero multilevel timestep was accepted");
  CHECK(!preflight_multilevel_internal_data(
            identity_gamma, 2, 1, 1, 1, std::numeric_limits<realnum>::infinity(), error),
        "nonfinite multilevel timestep was accepted");
  CHECK(!preflight_multilevel_internal_data(identity_gamma, 2, 1,
                                             std::numeric_limits<size_t>::max(), 1,
                                             realnum(0.1), error),
        "overflowing multilevel allocation extent was accepted");
  CHECK(!preflight_multilevel_internal_data(identity_gamma, 2,
                                             std::numeric_limits<size_t>::max(), 1, 2,
                                             realnum(0.1), error),
        "overflowing multilevel pointer-table extent was accepted");
  const std::vector<double> singular_gamma = {-20.0, 0.0, 0.0, 0.0};
  CHECK(!preflight_multilevel_internal_data(singular_gamma, 2, 1, 1, 1, realnum(0.1), error),
        "singular multilevel timestep matrix was accepted");
}

static void test_resident_multilevel_split_communicator() {
  const int groups = count_processors();
  divide_parallel_processes(groups);
  CHECK(my_rank() == 0, "one-rank multilevel split communicator did not have local rank zero");
  {
    const grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 1);
    add_multilevel_lifecycle_states(s);
    fields f(&s);
    f.use_real_fields();
    f.require_component(Ez);
    f.require_component(Hz);
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts);
    backend_reset_multilevel_collective_count_for_testing();
    f.advance(1);
    const size_t rebuild_collectives = backend_multilevel_collective_count_for_testing();
    f.advance(1);
    CHECK(f.backend_state && f.executable && f.backend_state->multilevel_plan_validated &&
              f.backend_state->multilevel_preflight_required &&
              counts.advance_attempts == 2 &&
              backend_multilevel_collective_count_for_testing() == rebuild_collectives,
          "split-communicator multilevel validation used the wrong communicator or repeated");
  }
  end_divide_parallel();
}

static void test_resident_polarization_preparation() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  lorentzian_susceptibility susceptibility(1.1, 0.05);
  gyrotropic_susceptibility gyro(vec(0.17, -0.23, 0.31), 0.8, 0.03, 0.07, GYROTROPIC_SATURATED);

  {
    structure single_structure(gv, eps_slab, pml(0.5), identity(), 1);
    single_structure.add_susceptibility(eps_slab, E_stuff, susceptibility);
    fields single(&single_structure);
    single.require_component(Ez);
    single.advance(1);
    fields copy(single);
    for (int i = 0; i < single.num_chunks; ++i)
      if (single.chunks[i]->is_mine()) {
        polarization_state *source = single.chunks[i]->pol[E_stuff];
        polarization_state *duplicate = copy.chunks[i]->pol[E_stuff];
        CHECK(source && !source->next && source->data && duplicate && !duplicate->next &&
                  duplicate->data && duplicate->data != source->data,
              "one-node fields copy did not duplicate polarization state exactly");
      }
  }

  structure resident_structure(gv, eps_slab, pml(0.5), identity(), 2);
  resident_structure.add_susceptibility(eps_slab, E_stuff, susceptibility);
  resident_structure.add_susceptibility(eps_slab, E_stuff, gyro);
  fields resident(&resident_structure);
  resident.require_component(Ez);
  bool initially_null = true;
  for (int i = 0; i < resident.num_chunks; ++i)
    if (resident.chunks[i]->is_mine())
      for (polarization_state *p = resident.chunks[i]->pol[E_stuff]; p; p = p->next)
        initially_null = initially_null && !p->data;
  CHECK(and_to_all(initially_null), "polarization state was allocated before preparation");

  lifetime_counts counts;
  resident.backend = new tracking_backend(resident, counts);
  resident.advance(1);
  {
    fields copy(resident);
    for (int i = 0; i < resident.num_chunks; ++i) {
      polarization_state *source = resident.chunks[i]->pol[E_stuff];
      polarization_state *duplicate = copy.chunks[i]->pol[E_stuff];
      size_t copied = 0;
      while (source && duplicate) {
        CHECK(!source->data || (duplicate->data && duplicate->data != source->data),
              "fields copy did not duplicate populated polarization state");
        ++copied;
        source = source->next;
        duplicate = duplicate->next;
      }
      CHECK(!source && !duplicate, "fields copy changed the polarization-state list shape");
      if (resident.chunks[i]->is_mine())
        CHECK(copied == 2, "two-node fields copy did not retain both polarization states");
    }
  }
  bool owns_chunk = false;
  for (int i = 0; i < resident.num_chunks; ++i)
    owns_chunk = owns_chunk || resident.chunks[i]->is_mine();
  CHECK(and_to_all(!owns_chunk || counts.polarization_arrays_at_create > 0),
        "resident state was created without P/P_prev arrays");
  CHECK(and_to_all(!owns_chunk || (counts.polarization_updates_at_compile > 0 &&
                                   counts.polarization_subtractions_at_compile > 0)),
        "resident executable was compiled without polarization spans");
  CHECK(and_to_all(!owns_chunk || counts.gyrotropic_update_at_compile),
        "resident executable was compiled without a gyrotropic update");
  CHECK(counts.polarization_zero_at_create,
        "resident polarization arrays were not zero-initialized before state creation");
  CHECK(counts.connections_current_at_create,
        "resident state was created before PE/PH topology was finalized");
  CHECK(resident.t == 0, "tracking backend unexpectedly advanced host time");
  const size_t arrays_after_first = counts.polarization_arrays_at_create;
  BackendState *state_after_first = resident.backend_state;
  resident.advance(1);
  CHECK(resident.backend_state == state_after_first && counts.states_created == 1,
        "second resident preparation rebuilt already-realized polarization state");
  CHECK(counts.polarization_arrays_at_create == arrays_after_first,
        "second resident preparation changed the polarization catalog");

  const size_t catalog_after_first = counts.arrays_at_create;
  counts.gyrotropic_update_at_compile = false;
  resident.require_component(Ex);
  resident.advance(1);
  CHECK(and_to_all(!owns_chunk || (counts.states_created == 2 && counts.states_destroyed == 1)),
        "post-allocation field growth did not replace resident state");
  CHECK(and_to_all(!owns_chunk ||
                   (counts.executables_created == 2 && counts.executables_destroyed == 1)),
        "post-allocation field growth did not replace the resident executable");
  CHECK(and_to_all(!owns_chunk || counts.gyrotropic_update_at_compile),
        "rebuilt resident executable lost its gyrotropic update");
  CHECK(!owns_chunk || counts.arrays_at_create > catalog_after_first,
        "post-allocation field growth did not expand the resident field catalog");
  CHECK(counts.polarization_arrays_at_create - arrays_after_first == arrays_after_first,
        "post-allocation field growth changed immutable gyrotropic state storage");

  structure cpu_structure(gv, eps_slab, pml(0.5), identity(), 2);
  cpu_structure.add_susceptibility(eps_slab, E_stuff, susceptibility);
  cpu_structure.add_susceptibility(eps_slab, E_stuff, gyro);
  fields cpu(&cpu_structure);
  cpu.require_component(Ez);
  bool cpu_null = true;
  for (int i = 0; i < cpu.num_chunks; ++i)
    if (cpu.chunks[i]->is_mine())
      for (polarization_state *p = cpu.chunks[i]->pol[E_stuff]; p; p = p->next)
        cpu_null = cpu_null && !p->data;
  CHECK(or_to_all(cpu_null), "CPU polarization state is no longer lazy before update_pols");
  cpu.advance(1);
  bool cpu_allocated = false;
  for (int i = 0; i < cpu.num_chunks; ++i)
    if (cpu.chunks[i]->is_mine())
      for (polarization_state *p = cpu.chunks[i]->pol[E_stuff]; p; p = p->next)
        cpu_allocated = cpu_allocated || p->data;
  CHECK(or_to_all(cpu_allocated), "CPU update_pols did not lazily allocate polarization state");
}

static bool same_seed_snapshot(const RandomSeedSnapshot &a, const RandomSeedSnapshot &b) {
  return a.semantic_seed == b.semantic_seed && a.saved_semantic_seed == b.saved_semantic_seed &&
         a.generation == b.generation && a.algorithm_version == b.algorithm_version &&
         a.initialized == b.initialized && a.semantic_seed_valid == b.semantic_seed_valid &&
         a.saved_semantic_seed_valid == b.saved_semantic_seed_valid &&
         a.explicit_seed == b.explicit_seed && a.saved_explicit_seed == b.saved_explicit_seed;
}

static bool has_local_noisy_actions(const fields &f) {
  if (!f.step_plans[0]) return false;
  for (const PolarizationUpdate &update : f.step_plans[0]->polarization_updates)
    if (update.kind == PolarizationUpdateKind::noisy_add) return true;
  return false;
}

static void add_noisy_fixture(structure &s, fields &f) {
  gaussian_src_time source(0.3, 0.1);
  f.add_point_source(Ez, source, vec(0.11, 0.13));
}

static void test_resident_noisy_lazy_default() {
  const RandomSeedSnapshot initial = random_seed_snapshot();
  CHECK(!initial.initialized && !initial.semantic_seed_valid,
        "isolated lazy-default fixture did not begin before RNG initialization");

  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  noisy_lorentzian_susceptibility noisy(0.03125, 1.1, 0.05);
  s.add_susceptibility(unit_epsilon, E_stuff, noisy);
  fields f(&s);
  add_noisy_fixture(s, f);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  f.advance(1);

  const bool local_noisy = has_local_noisy_actions(f);
  const RandomSeedSnapshot current = random_seed_snapshot();
  CHECK(current.initialized && current.semantic_seed_valid && !current.explicit_seed,
        "resident noisy lazy default did not initialize participating rank metadata");
  CHECK(counts.noisy_seed_refresh_attempts == 1 && counts.noisy_seed_refreshes == 1,
        "resident noisy lazy default did not refresh every participating rank");
  CHECK(f.backend_state->random_seed_snapshot_accepted &&
            same_seed_snapshot(f.backend_state->accepted_random_seed, current),
        "resident noisy lazy default accepted the wrong rank-local snapshot");
  CHECK(local_noisy || f.step_plans[0]->polarization_updates.empty(),
        "idle noisy rank installed a local polarization action");
  CHECK(or_to_all(local_noisy), "resident noisy lazy-default fixture had no owning rank");
}

static void test_resident_noisy_seed_lifecycle() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  noisy_lorentzian_susceptibility noisy(0.03125, 1.1, 0.05);
  s.add_susceptibility(unit_epsilon, E_stuff, noisy);
  fields f(&s);
  gaussian_src_time source(0.3, 0.1);
  f.add_point_source(Ez, source, vec(0.11, 0.13));
  component monitor_component = Ez;
  dft_fields monitor = f.add_dft_fields(&monitor_component, 1, f.v, 0.3, 0.3, 1);
  f.add_flux_vol(X, volume(vec(0.1, -0.8), vec(0.1, 0.8)));
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);

  set_random_seed(1001);
  const int expected_first_draw = random_int(0, 1000000);
  set_random_seed(1001);
  const RandomSeedSnapshot first_seed = random_seed_snapshot();
  f.advance(1);
  const bool local_noisy = has_local_noisy_actions(f);
  CHECK(f.backend_state && f.backend_state->random_seed_snapshot_accepted &&
            same_seed_snapshot(f.backend_state->accepted_random_seed, first_seed),
        "first resident noisy advance did not accept the current seed snapshot");
  CHECK(counts.noisy_seed_refresh_attempts == 1 && counts.noisy_seed_refreshes == 1 &&
            same_seed_snapshot(counts.last_noisy_seed, first_seed),
        "first resident noisy advance did not refresh exactly one seed snapshot");
  CHECK(random_int(0, 1000000) == expected_first_draw,
        "resident noisy compile/refresh/advance consumed the host MT stream");
  CHECK(or_to_all(local_noisy), "resident noisy seed fixture installed no noisy action");

  BackendState *const state = f.backend_state;
  Executable *const executable = f.executable;
  StepPlan *const ordinary_plan = f.step_plans[0];
  StoragePlan *const storage = f.storage_plan;
  CpuArrayCatalog *const catalog = f.array_catalog;
  const uint64_t plan_signature = ordinary_plan->signature;
  const int entry_t = f.t;
  const BackendEpochSnapshot stable_epoch(f);
  const int first_local_refreshes = 1;

  set_random_seed(2002);
  const RandomSeedSnapshot second_seed = random_seed_snapshot();
  f.advance(1);
  CHECK(f.backend_state == state && f.executable == executable && f.step_plans[0] == ordinary_plan &&
            f.storage_plan == storage && f.array_catalog == catalog &&
            f.step_plans[0]->signature == plan_signature && f.t == entry_t &&
            stable_epoch.matches(f),
        "reseed rebuilt or mutated the resident noisy execution epoch");
  CHECK(counts.noisy_seed_refresh_attempts == 2 * first_local_refreshes &&
            counts.noisy_seed_refreshes == 2 * first_local_refreshes &&
            same_seed_snapshot(state->accepted_random_seed, second_seed),
        "reseed did not publish exactly one refreshed snapshot");

  set_random_seed(3003);
  const RandomSeedSnapshot third_seed = random_seed_snapshot();
  Executable *const before_flux_executable = f.executable;
  StepPlan *const before_flux_plan = f.step_plans[0];
  f.add_flux_vol(Y, volume(vec(-0.8, -0.2), vec(0.8, -0.2)));
  f.advance(1);
  CHECK(f.backend_state == state && f.executable != before_flux_executable &&
            f.step_plans[0] != before_flux_plan && f.storage_plan == storage &&
            f.array_catalog == catalog && f.t == entry_t &&
            counts.noisy_seed_refreshes == 3 * first_local_refreshes &&
            same_seed_snapshot(state->accepted_random_seed, third_seed),
        "concurrent seed and legacy-flux refresh did not preserve state and publish both updates");

  const RandomSeedSnapshot before_restore = state->accepted_random_seed;
  restore_random_seed();
  const RandomSeedSnapshot restored_seed = random_seed_snapshot();
  f.advance(1);
  CHECK(restored_seed.generation != before_restore.generation &&
            counts.noisy_seed_refreshes == 4 * first_local_refreshes &&
            same_seed_snapshot(state->accepted_random_seed, restored_seed),
        "valid seed restore did not refresh the resident noisy snapshot");
  restore_random_seed();
  const RandomSeedSnapshot repeated_restore = random_seed_snapshot();
  f.advance(1);
  CHECK(counts.noisy_seed_refreshes == 5 * first_local_refreshes &&
            same_seed_snapshot(state->accepted_random_seed, repeated_restore),
        "repeated valid restore did not refresh by generation");

  set_random_seed(2002);
  const RandomSeedSnapshot repeated_seed = random_seed_snapshot();
  f.advance(1);
  CHECK(repeated_seed.generation != second_seed.generation &&
            counts.noisy_seed_refresh_attempts == 6 * first_local_refreshes &&
            counts.noisy_seed_refreshes == 6 * first_local_refreshes &&
            same_seed_snapshot(state->accepted_random_seed, repeated_seed),
        "same-value reseed did not refresh by generation without rebuilding");

  set_random_seed(5005);
  const int expected_failure_draw = random_int(0, 1000000);
  const int expected_retry_draw = random_int(0, 1000000);
  set_random_seed(5005);
  const RandomSeedSnapshot failed_candidate = random_seed_snapshot();
  const RandomSeedSnapshot accepted_before_failure = state->accepted_random_seed;
  const BackendEpochSnapshot failure_epoch(f);
  const int advances_before_failure = counts.advanced;
  counts.fail_noisy_seed_refresh = true;
  bool failed = false;
  try { f.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  CHECK(failed &&
            counts.noisy_seed_refresh_attempts == 7 * first_local_refreshes &&
            counts.noisy_seed_refreshes == 6 * first_local_refreshes &&
            counts.advanced == advances_before_failure &&
            same_seed_snapshot(state->accepted_random_seed, accepted_before_failure) &&
            failure_epoch.matches(f) && !f.backend->is_poisoned(),
        "failed noisy seed refresh published state, dispatched, rebuilt, or poisoned");
  CHECK(random_int(0, 1000000) == expected_failure_draw,
        "failed noisy seed refresh consumed the host MT stream");

  counts.fail_noisy_seed_refresh = false;
  f.advance(1);
  CHECK(counts.noisy_seed_refresh_attempts == 8 * first_local_refreshes &&
            counts.noisy_seed_refreshes == 7 * first_local_refreshes &&
            counts.advanced == advances_before_failure + 1 &&
            same_seed_snapshot(state->accepted_random_seed, failed_candidate),
        "retry after noisy seed refresh failure did not publish the pending snapshot exactly once");
  CHECK(random_int(0, 1000000) == expected_retry_draw,
        "noisy seed refresh retry consumed the host MT stream");

  const int refreshes_before_reset = counts.noisy_seed_refreshes;
  Executable *const executable_before_reset = f.executable;
  StepPlan *const plan_before_reset = f.step_plans[0];
  f.zero_fields();
  f.advance(1);
  CHECK(f.backend_state == state && f.executable == executable_before_reset &&
            f.step_plans[0] == plan_before_reset &&
            counts.noisy_seed_refreshes == refreshes_before_reset &&
            same_seed_snapshot(state->accepted_random_seed, failed_candidate),
        "field reset rebuilt or reset the accepted noisy seed snapshot");

  const int refreshes_before_remove = counts.noisy_seed_refreshes;
  BackendState *const flux_state = f.backend_state;
  f.remove_fluxes();
  f.advance(1);
  CHECK(f.backend_state == flux_state && counts.legacy_flux_updates_at_compile == 0 &&
            counts.noisy_seed_refreshes == refreshes_before_remove,
        "legacy-flux removal rebuilt state or reset the noisy seed snapshot");
  f.add_flux_vol(X, volume(vec(-0.2, -0.8), vec(-0.2, 0.8)));
  f.advance(1);
  CHECK(f.backend_state == flux_state && counts.legacy_flux_updates_at_compile == 1 &&
            counts.noisy_seed_refreshes == refreshes_before_remove,
        "legacy-flux add-after-remove rebuilt state or reset the noisy seed snapshot");

  const int states_before_growth = counts.states_created;
  const int destroyed_before_growth = counts.states_destroyed;
  set_random_seed(6006);
  const RandomSeedSnapshot growth_seed = random_seed_snapshot();
  f.require_component(Ex);
  f.advance(1);
  const bool growth_noisy = has_local_noisy_actions(f);
  CHECK(!growth_noisy ||
            (counts.states_created == states_before_growth + 1 &&
             counts.states_destroyed == destroyed_before_growth + 1),
        "owned catalog growth did not replace resident state exactly once");
  CHECK(f.backend_state->random_seed_snapshot_accepted &&
            same_seed_snapshot(f.backend_state->accepted_random_seed, growth_seed),
        "catalog/state replacement accepted the wrong rank-local noisy seed snapshot");

  execution_options cpu_options;
  f.select_backend(cpu_options);
  CHECK(!f.backend_state && !f.executable,
        "backend reselection retained the previous resident execution epoch");
  delete f.backend;
  f.backend = new tracking_backend(f, counts);
  const int refreshes_before_reselection = counts.noisy_seed_refreshes;
  f.advance(1);
  const bool reselection_noisy = has_local_noisy_actions(f);
  CHECK(f.backend_state && f.backend_state->random_seed_snapshot_accepted &&
            counts.noisy_seed_refreshes == refreshes_before_reselection + 1 &&
            same_seed_snapshot(f.backend_state->accepted_random_seed, growth_seed),
        "backend reselection did not accept the current noisy seed snapshot");

  monitor.remove();
  f.remove_fluxes();
  f.advance(1);
  const int noisy_attempts_before_failure = counts.advance_attempts;
  counts.fail_advance = my_rank() == 0;
  bool dispatch_failed = false;
  try { f.advance(1); }
  catch (const std::runtime_error &) { dispatch_failed = true; }
  CHECK(dispatch_failed && f.backend->is_poisoned() &&
            counts.advance_attempts == noisy_attempts_before_failure + 1,
        "noisy post-dispatch failure did not poison the resident backend");

  structure plain_structure(gv, eps_slab, pml(0.5), identity(), 2);
  fields plain(&plain_structure);
  plain.add_point_source(Ez, source, vec(0.11, 0.13));
  lifetime_counts plain_counts;
  plain.backend = new tracking_backend(plain, plain_counts);
  set_random_seed(4004);
  plain.advance(1);
  CHECK(plain_counts.noisy_seed_refresh_attempts == 0 && plain.backend_state &&
            !plain.backend_state->random_seed_snapshot_accepted,
        "non-noisy resident plan sampled or accepted RNG metadata");
}

static void test_resident_noisy_prelaunch_failures() {
  grid_volume gv = vol2d(2.0, 2.0, 8.0);
  gaussian_src_time source(0.3, 0.1);
  noisy_lorentzian_susceptibility noisy(0.03125, 1.1, 0.05);

  {
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    s.add_susceptibility(unit_epsilon, E_stuff, noisy);
    fields f(&s);
    f.add_point_source(Ez, source, vec(0.0, 0.0));
    lifetime_counts counts;
    counts.fail_create_state = true;
    f.backend = new tracking_backend(f, counts);
    set_random_seed(7007);
    bool failed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    CHECK(failed && !f.backend_state && !f.executable &&
              counts.noisy_seed_refresh_attempts == 0 && counts.advance_attempts == 0 &&
              !f.backend->is_poisoned(),
          "cold noisy state-creation failure crossed the seed/dispatch boundary");
    counts.fail_create_state = false;
    f.advance(1);
    CHECK(f.backend_state && f.executable && !f.backend->is_poisoned(),
          "cold noisy state-creation failure was not retryable");
  }

  {
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    s.add_susceptibility(unit_epsilon, E_stuff, noisy);
    fields f(&s);
    f.add_point_source(Ez, source, vec(0.0, 0.0));
    lifetime_counts counts;
    counts.fail_compile = true;
    f.backend = new tracking_backend(f, counts);
    set_random_seed(8008);
    bool failed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    CHECK(failed && !f.backend_state && !f.executable &&
              counts.noisy_seed_refresh_attempts == 0 && counts.advance_attempts == 0 &&
              !f.backend->is_poisoned(),
          "cold noisy compile failure crossed the seed/dispatch boundary");
    counts.fail_compile = false;
    f.advance(1);
    CHECK(f.executable && !f.backend->is_poisoned(),
          "cold noisy compile failure was not retryable");
  }

  {
    structure s(gv, unit_epsilon, no_pml(), identity(), 2);
    s.add_susceptibility(unit_epsilon, E_stuff, noisy);
    fields f(&s);
    f.add_point_source(Ez, source, vec(0.0, 0.0));
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts);
    set_random_seed(8108 + my_global_rank());
    f.advance(1);
    const RandomSeedSnapshot accepted = f.backend_state->accepted_random_seed;
    const int dispatches = counts.advance_attempts;
    set_random_seed(8208 + my_global_rank());
    counts.poison_noisy_seed_refresh = my_rank() == 0;
    bool failed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    CHECK(failed && f.backend->is_poisoned() && counts.advance_attempts == dispatches &&
              same_seed_snapshot(f.backend_state->accepted_random_seed, accepted),
          "poisoned noisy seed staging committed or dispatched");
  }
}

static void expect_collective_noisy_preflight_failure(fields &f, lifetime_counts &counts,
                                                      const char *message) {
  BackendState *const state = f.backend_state;
  Executable *const executable = f.executable;
  StepPlan *const plan = f.step_plans[0];
  StoragePlan *const storage = f.storage_plan;
  CpuArrayCatalog *const catalog = f.array_catalog;
  const RandomSeedSnapshot accepted = state->accepted_random_seed;
  const int entry_t = f.t;
  const int attempts = counts.noisy_seed_refresh_attempts;
  const int refreshes = counts.noisy_seed_refreshes;
  const int dispatches = counts.advance_attempts;
  bool failed = false;
  try { f.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  CHECK(failed && f.backend_state == state && f.executable == executable &&
            f.step_plans[0] == plan && f.storage_plan == storage && f.array_catalog == catalog &&
            f.t == entry_t && counts.noisy_seed_refresh_attempts == attempts &&
            counts.noisy_seed_refreshes == refreshes && counts.advance_attempts == dispatches &&
            same_seed_snapshot(state->accepted_random_seed, accepted) && !f.backend->is_poisoned(),
        message);
}

static void test_resident_noisy_collective_preflight() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  noisy_lorentzian_susceptibility noisy0(0.03125, 1.1, 0.05);
  noisy_lorentzian_susceptibility noisy1(0.0625, 0.9, 0.07);
  s.add_susceptibility(unit_epsilon, E_stuff, noisy0);
  s.add_susceptibility(unit_epsilon, E_stuff, noisy1);
  fields f(&s);
  add_noisy_fixture(s, f);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);

  for (int i = 0; i <= my_rank(); ++i) set_random_seed(9000 + my_global_rank());
  const RandomSeedSnapshot local_seed = random_seed_snapshot();
  f.advance(1);
  CHECK(f.backend_state->random_seed_snapshot_accepted &&
            same_seed_snapshot(f.backend_state->accepted_random_seed, local_seed),
        "collective noisy preflight rejected valid unequal rank-local seed metadata");
  CHECK(counts.noisy_seed_refreshes == 1,
        "collective noisy preflight did not refresh every participating rank");
  const bool local_noisy = has_local_noisy_actions(f);
  CHECK(or_to_all(local_noisy), "collective noisy fixture installed no noisy action");
  CHECK(local_noisy || f.step_plans[0]->polarization_updates.empty(),
        "idle collective noisy rank installed a local polarization row");

  const int dynamic_modes[] = {1, 2, 3, 5, 7};
  for (int target = 0; target < count_processors(); ++target) {
    for (size_t mode_index = 0; mode_index < sizeof(dynamic_modes) / sizeof(dynamic_modes[0]);
         ++mode_index) {
      backend_set_noisy_preflight_failure_for_testing(target, dynamic_modes[mode_index]);
      expect_collective_noisy_preflight_failure(
          f, counts, "rank-asymmetric noisy metadata failure crossed the dispatch boundary");
      backend_set_noisy_preflight_failure_for_testing(-1, 0);
      const int dispatches = counts.advance_attempts;
      f.advance(1);
      CHECK(counts.advance_attempts == dispatches + 1 && !f.backend->is_poisoned(),
            "rank-asymmetric noisy metadata failure was not retryable");
    }
    const bool target_has_noisy = or_to_all(my_rank() == target && local_noisy);
    if (target_has_noisy)
      for (int mode = 4; mode <= 6; mode += 2) {
        backend_set_noisy_preflight_failure_for_testing(target, mode);
        expect_collective_noisy_preflight_failure(
            f, counts, "rank-asymmetric noisy static failure crossed the dispatch boundary");
        backend_set_noisy_preflight_failure_for_testing(-1, 0);
        const int dispatches = counts.advance_attempts;
        f.advance(1);
        CHECK(counts.advance_attempts == dispatches + 1 && !f.backend->is_poisoned(),
              "rank-asymmetric noisy static failure was not retryable");
      }
  }

  StepPlan stable_plan = *f.step_plans[0];
  std::vector<size_t> noise_rows;
  for (size_t i = 0; i < f.step_plans[0]->polarization_updates.size(); ++i)
    if (f.step_plans[0]->polarization_updates[i].kind == PolarizationUpdateKind::noisy_add)
      noise_rows.push_back(i);
  const bool enough_rows = noise_rows.size() >= 2;
  CHECK(or_to_all(enough_rows), "collective noisy mutation fixture has fewer than two noise rows");
  if (my_rank() == 0 && enough_rows) {
    std::swap(f.step_plans[0]->polarization_updates[noise_rows[0]],
              f.step_plans[0]->polarization_updates[noise_rows[1]]);
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "re-signed noisy action order mutation crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0 && enough_rows) {
    PolarizationUpdate &first = f.step_plans[0]->polarization_updates[noise_rows[0]];
    PolarizationUpdate &second = f.step_plans[0]->polarization_updates[noise_rows[1]];
    second.region.chunk = first.region.chunk;
    second.region.c = first.region.c;
    second.region.cmp = first.region.cmp;
    second.ft = first.ft;
    second.state_index = first.state_index;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "duplicate re-signed noisy stream tuple crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0 && !noise_rows.empty())
    f.step_plans[0]->polarization_updates[noise_rows[0]].p =
        f.step_plans[0]->polarization_updates[noise_rows[0]].diagonal_sigma;
  expect_collective_noisy_preflight_failure(
      f, counts, "stale-signature noisy ArrayId mutation crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0)
    for (PolarizationUpdate &row : f.step_plans[0]->polarization_updates)
      if (row.kind == PolarizationUpdateKind::noisy_add)
        row.kind = PolarizationUpdateKind::lorentzian;
  expect_collective_noisy_preflight_failure(
      f, counts, "stale-signature removal of all noisy actions crossed dispatch");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0 && !noise_rows.empty()) {
    PolarizationUpdate &row = f.step_plans[0]->polarization_updates[noise_rows[0]];
    row.p = row.diagonal_sigma;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "re-signed noisy ArrayId mutation crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0 && !noise_rows.empty()) {
    f.step_plans[0]->polarization_updates[noise_rows[0]].region.counts[0] = 0;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "re-signed noisy range mutation crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0 && !noise_rows.empty()) {
    UpdateRegion &region = f.step_plans[0]->polarization_updates[noise_rows[0]].region;
    bool changed = false;
    for (int axis = 0; axis < 3; ++axis)
      if (!changed && region.counts[axis] > 1 && region.strides[axis] != 0) {
        region.strides[axis] = 0;
        changed = true;
      }
    CHECK(changed, "noisy zero-stride mutation found no traversed axis");
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "re-signed noisy zero-stride mutation crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0) {
    for (Operation &op : f.step_plans[0]->operations)
      if (op.kind == OpKind::update_polarization && op.descriptor_count) {
        --op.descriptor_count;
        if (!op.accesses.empty()) op.accesses.pop_back();
        break;
      }
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "re-signed noisy operation span/access mutation crossed dispatch");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0) {
    for (Operation &op : f.step_plans[0]->operations)
      if (op.kind == OpKind::update_polarization) {
        op.guard = guard_device(17);
        op.source_time_offset = 0.25;
        break;
      }
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "re-signed noisy operation guard/timing mutation crossed dispatch");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0 && !noise_rows.empty()) {
    f.step_plans[0]->polarization_updates.erase(
        f.step_plans[0]->polarization_updates.begin() + noise_rows[0]);
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "missing re-signed noisy action crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0 && !noise_rows.empty()) {
    f.step_plans[0]->polarization_updates.push_back(
        f.step_plans[0]->polarization_updates[noise_rows[0]]);
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "extra duplicate noisy action crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0 && !noise_rows.empty()) {
    UpdateRegion &region = f.step_plans[0]->polarization_updates[noise_rows[0]].region;
    region.counts[0] = std::numeric_limits<size_t>::max();
    region.counts[1] = 2;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "noisy owned-point extent overflow crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0 && !noise_rows.empty()) {
    f.step_plans[0]->polarization_updates[noise_rows[0]].omega_0 =
        std::numeric_limits<double>::max();
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "overflowing noisy omega coefficient crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0 && !noise_rows.empty()) {
    f.step_plans[0]->polarization_updates[noise_rows[0]].gamma = -0.1;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "invalid noisy gamma sqrt domain crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0 && !noise_rows.empty()) {
    f.step_plans[0]->polarization_updates[noise_rows[0]].dt = 0.0;
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "invalid noisy timestep coefficient crossed the dispatch boundary");
  *f.step_plans[0] = stable_plan;

  CpuArrayCatalog *const live_catalog = f.array_catalog;
  if (my_rank() == 0) {
    f.array_catalog = NULL;
    f.step_plans[0]->signature ^= UINT64_C(1);
  }
  const int dispatches_before_missing_catalog = counts.advance_attempts;
  bool missing_catalog_failed = false;
  try { f.advance(1); }
  catch (const std::runtime_error &) { missing_catalog_failed = true; }
  if (my_rank() == 0) f.array_catalog = live_catalog;
  *f.step_plans[0] = stable_plan;
  CHECK(missing_catalog_failed && counts.advance_attempts == dispatches_before_missing_catalog &&
            !f.backend->is_poisoned(),
        "rank-asymmetric missing noisy catalog crossed the dispatch boundary");

  DescriptorSet stable_descriptors = *f.descriptors;
  if (my_rank() == 0) {
    f.descriptors->polarizations.clear();
    f.step_plans[0]->polarization_updates.clear();
    for (Operation &op : f.step_plans[0]->operations)
      if (op.kind == OpKind::update_polarization) {
        op.descriptor_index = 0;
        op.descriptor_count = 0;
        op.accesses.clear();
      }
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "missing noisy descriptor and plan rows crossed the dispatch boundary");
  *f.descriptors = stable_descriptors;
  *f.step_plans[0] = stable_plan;

  if (my_rank() == 0) {
    bool changed = false;
    for (PolarizationDescriptor &descriptor : f.descriptors->polarizations)
      if (!changed && descriptor.kind == SusceptibilityKind::noisy_lorentzian) {
        descriptor.noise_amplitude = std::numeric_limits<double>::infinity();
        for (PolarizationUpdate &row : f.step_plans[0]->polarization_updates)
          if (row.kind == PolarizationUpdateKind::noisy_add &&
              row.region.chunk == descriptor.chunk && row.ft == descriptor.ft &&
              row.state_index == descriptor.state_index)
            row.noise_amplitude = descriptor.noise_amplitude;
        changed = true;
      }
    f.step_plans[0]->signature = compute_step_plan_signature(*f.step_plans[0]);
  }
  expect_collective_noisy_preflight_failure(
      f, counts, "descriptor-consistent nonfinite noisy coefficient crossed dispatch");
  *f.descriptors = stable_descriptors;
  *f.step_plans[0] = stable_plan;

  for (int target = 0; target < count_processors(); ++target) {
    set_random_seed(10000 + 31 * target + my_global_rank());
    const RandomSeedSnapshot pending = random_seed_snapshot();
    const RandomSeedSnapshot accepted = f.backend_state->accepted_random_seed;
    const int dispatches = counts.advance_attempts;
    const int refresh_attempts = counts.noisy_seed_refresh_attempts;
    counts.fail_noisy_seed_refresh = my_rank() == target;
    bool failed = false;
    try { f.advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    CHECK(failed && counts.advance_attempts == dispatches &&
              counts.noisy_seed_refresh_attempts == refresh_attempts + 1 &&
              same_seed_snapshot(f.backend_state->accepted_random_seed, accepted) &&
              !f.backend->is_poisoned(),
          "rank-asymmetric noisy seed hook failure dispatched or published acceptance");
    counts.fail_noisy_seed_refresh = false;
    f.advance(1);
    CHECK(same_seed_snapshot(f.backend_state->accepted_random_seed, pending) &&
              counts.advance_attempts == dispatches + 1,
          "rank-asymmetric noisy seed hook failure was not retryable");
  }

  const int attempts_before_poison = counts.advance_attempts;
  counts.fail_advance = my_rank() == 0;
  bool failed = false;
  try { f.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  CHECK(failed && f.backend->is_poisoned() &&
            counts.advance_attempts == attempts_before_poison + 1,
        "rank-asymmetric noisy dispatch failure did not collectively poison");
  failed = false;
  try { f.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  CHECK(failed && counts.advance_attempts == attempts_before_poison + 1,
        "collectively poisoned noisy backend accepted another dispatch");
}

static void test_resident_noisy_zero_row() {
  grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, unit_epsilon, no_pml(), identity(), 2);
  noisy_lorentzian_susceptibility noisy(0.03125, 1.1, 0.05);
  s.add_susceptibility(unit_epsilon, E_stuff, noisy);
  fields f(&s);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  backend_reset_noisy_collective_count_for_testing();
  f.advance(1);
  f.advance(1);
  CHECK(counts.noisy_seed_refresh_attempts == 0 && counts.noisy_seed_refreshes == 0 &&
            f.backend_state && !f.backend_state->random_seed_snapshot_accepted &&
            !f.backend_state->noisy_preflight_required &&
            backend_noisy_collective_count_for_testing() == 0,
        "zero-row noisy definition sampled or refreshed RNG metadata");
}

static void test_resident_noisy_invalid_first_restore() {
  restore_random_seed();
  const RandomSeedSnapshot invalid = random_seed_snapshot();
  CHECK(invalid.initialized && !invalid.semantic_seed_valid,
        "isolated first restore did not produce invalid semantic metadata");
  grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, unit_epsilon, no_pml(), identity(), 2);
  noisy_lorentzian_susceptibility noisy(0.03125, 1.1, 0.05);
  s.add_susceptibility(unit_epsilon, E_stuff, noisy);
  fields f(&s);
  add_noisy_fixture(s, f);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  bool failed = false;
  try { f.advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  CHECK(failed && counts.noisy_seed_refresh_attempts == 0 && counts.advance_attempts == 0 &&
            !f.backend->is_poisoned(),
        "invalid first restore crossed the noisy seed dispatch boundary");
  set_random_seed(12345 + my_global_rank());
  f.advance(1);
  CHECK(counts.noisy_seed_refreshes == 1 && counts.advance_attempts == 1,
        "valid seed did not retry after invalid first restore");
}

static void test_resident_noisy_split_global_rank() {
  const int groups = count_processors();
  divide_parallel_processes(groups);
  CHECK(my_rank() == 0, "one-rank split communicator did not have local rank zero");
  const uint64_t local_tag = counter_random_stream_tag(
      counter_random_algorithm_version, uint32_t(my_global_rank()), 0, uint32_t(E_stuff), 0,
      uint32_t(Ez), 0);
  const uint64_t rank_zero_tag = counter_random_stream_tag(
      counter_random_algorithm_version, 0, 0, uint32_t(E_stuff), 0, uint32_t(Ez), 0);
  CHECK(rank_zero_tag == UINT64_C(0xbef1dcd6d5312d4d),
        "version-1 noisy FNV stream-tag KAT changed");
  CHECK(my_global_rank() == 0 || local_tag != rank_zero_tag,
        "split-communicator noisy stream tag used local rather than global rank");

  {
    grid_volume gv = vol2d(2.0, 2.0, 8.0);
    structure s(gv, unit_epsilon, no_pml(), identity(), 1);
    noisy_lorentzian_susceptibility noisy(0.03125, 1.1, 0.05);
    s.add_susceptibility(unit_epsilon, E_stuff, noisy);
    fields f(&s);
    add_noisy_fixture(s, f);
    lifetime_counts counts;
    f.backend = new tracking_backend(f, counts);
    set_random_seed(777);
    f.advance(1);
    CHECK(counts.noisy_seed_refreshes == 1 && counts.advance_attempts == 1,
          "split-communicator noisy preflight did not accept local rank metadata");
    const PolarizationUpdate *first = NULL;
    for (const PolarizationUpdate &update : f.step_plans[0]->polarization_updates)
      if (!first && update.kind == PolarizationUpdateKind::noisy_add) first = &update;
    CHECK(first && f.backend_state->noisy_stream_count > 0,
          "split-communicator noisy preflight recorded no production stream");
    if (first) {
      const uint64_t expected = counter_random_stream_tag(
          first->noise_algorithm_version, uint32_t(my_global_rank()),
          uint32_t(first->region.chunk), uint32_t(first->ft), uint32_t(first->state_index),
          uint32_t(first->region.c), uint32_t(first->region.cmp));
      CHECK(f.backend_state->noisy_first_stream_tag == expected,
            "production noisy preflight did not use global-rank stream identity");
    }
  }
  end_divide_parallel();
}

static void test_resident_advance_failure_poison() {
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  counts.fail_advance = true;
  f->backend = new tracking_backend(*f, counts);
  bool failed = false;
  try { f->advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  CHECK(failed && f->backend->is_poisoned() && counts.advance_attempts == 1 &&
            counts.advanced == 0,
        "resident post-dispatch failure did not poison the backend");
  failed = false;
  try { f->advance(1); }
  catch (const std::runtime_error &) { failed = true; }
  CHECK(failed && counts.advance_attempts == 1,
        "poisoned resident backend accepted another dispatch");
  delete f;
  delete s;
}

static void test_resident_beta_fingerprint() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  fields f(&s, 0, 0.17);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.1, 0.1));
  f.require_component(Ez);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  f.advance(1);

  bool owns_chunk = false;
  for (int i = 0; i < f.num_chunks; ++i)
    owns_chunk = owns_chunk || f.chunks[i]->is_mine();
  CHECK(and_to_all(!owns_chunk || counts.beta_updates_at_compile > 0),
        "resident executable was compiled without beta updates");
  CHECK(coordinate_state_matches(f, f.step_plans[0]),
        "fresh resident beta fingerprint does not match");

  const double original = f.beta;
  f.beta = -original;
  CHECK(!coordinate_state_matches(f, f.step_plans[0]),
        "outer beta mutation was not rejected by the resident fingerprint");
  f.beta = original;

  CHECK(f.num_chunks > 0, "beta fingerprint test has no chunks");
  if (f.num_chunks > 0) {
    f.chunks[0]->beta = -original;
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "per-chunk beta mutation was not rejected by the resident fingerprint");
    f.chunks[0]->beta = original;
  }
  CHECK(coordinate_state_matches(f, f.step_plans[0]), "restored beta fingerprint does not match");
}

static void test_resident_bfast_fingerprint() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  const std::vector<double> scaled_k{0.17, -0.11, 0.07};
  fields f(&s, 0, 0, true, 0, 0, scaled_k);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.1, 0.1));
  f.require_component(Ez);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  f.advance(1);

  bool owns_chunk = false;
  for (int i = 0; i < f.num_chunks; ++i)
    owns_chunk = owns_chunk || f.chunks[i]->is_mine();
  CHECK(and_to_all(!owns_chunk || counts.bfast_updates_at_compile > 0),
        "resident executable was compiled without BFAST updates");
  CHECK(coordinate_state_matches(f, f.step_plans[0]),
        "fresh resident BFAST fingerprint does not match");

  f.bfast_scaled_k[0] = -scaled_k[0];
  CHECK(!coordinate_state_matches(f, f.step_plans[0]),
        "outer BFAST mutation was not rejected by the resident fingerprint");
  f.bfast_scaled_k = scaled_k;

  CHECK(f.num_chunks > 0, "BFAST fingerprint test has no chunks");
  if (f.num_chunks > 0) {
    f.chunks[0]->bfast_scaled_k[1] = -scaled_k[1];
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "per-chunk BFAST mutation was not rejected by the resident fingerprint");
    f.chunks[0]->bfast_scaled_k = scaled_k;
  }
  CHECK(coordinate_state_matches(f, f.step_plans[0]), "restored BFAST fingerprint does not match");

  f.bfast_scaled_k.resize(2);
  for (int i = 0; i < f.num_chunks; ++i)
    f.chunks[i]->bfast_scaled_k.resize(2);
  CHECK(!coordinate_state_matches(f, f.step_plans[0]),
        "malformed matching BFAST vectors were not rejected before plan rebuild");
}

static void test_resident_cylindrical_fingerprint() {
  grid_volume gv = volcyl(3.0, 4.0, 8.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  fields f(&s, +1.0, 0, true, 64, 64, std::vector<double>{0, 0, 0});
  const component components[] = {Er, Ep, Ez, Hr, Hp, Hz};
  for (component c : components)
    f.require_component(c);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  f.advance(1);

  bool owns_chunk = false, owns_origin_chunk = false;
  for (int i = 0; i < f.num_chunks; ++i) {
    owns_chunk = owns_chunk || f.chunks[i]->is_mine();
    owns_origin_chunk =
        owns_origin_chunk || (f.chunks[i]->is_mine() && f.chunks[i]->gv.origin_r() == 0.0);
  }
  CHECK(and_to_all(!owns_chunk || counts.cylindrical_m_updates_at_compile > 0),
        "resident executable was compiled without cylindrical m/r updates");
  CHECK(and_to_all(!owns_origin_chunk || counts.cylindrical_origin_actions_at_compile > 0),
        "resident executable was compiled without cylindrical origin actions");
  CHECK(coordinate_state_matches(f, f.step_plans[0]),
        "fresh resident cylindrical fingerprint does not match");

  const size_t arrays_after_first = counts.arrays_at_create;
  BackendState *state_after_first = f.backend_state;
  f.change_m(-1.0);
  f.advance(1);
  CHECK(f.backend_state == state_after_first && counts.states_created == 1 &&
            counts.states_destroyed == 0 && counts.arrays_at_create == arrays_after_first,
        "sign-only cylindrical m change unnecessarily rebuilt resident storage");
  CHECK(counts.executables_created == 2 && counts.executables_destroyed == 1,
        "sign-only cylindrical m change did not replace the resident executable");
  CHECK(f.step_plans[0] && f.step_plans[0]->cylindrical_m == -1.0 &&
            coordinate_state_matches(f, f.step_plans[0]),
        "sign-only cylindrical m change did not publish a matching plan");

  f.change_m(0.0);
  CHECK(f.backend_state == NULL && f.executable == NULL && counts.rebuilds == 1 &&
            (!owns_chunk || counts.rebuild_saw_live_imaginary),
        "complex-to-real cylindrical transition did not migrate and retire live storage first");
  f.advance(1);
  CHECK(counts.states_created == 2 && counts.states_destroyed == 1 &&
            counts.arrays_at_create <= arrays_after_first,
        "cylindrical m-to-zero field-layout change did not rebuild resident storage");
  CHECK(counts.executables_created == 3 && counts.executables_destroyed == 2,
        "cylindrical m-to-zero change did not replace the resident executable");
  CHECK(counts.cylindrical_m_updates_at_compile == 0,
        "zero cylindrical m retained m/r descriptors after recompilation");
  CHECK(and_to_all(!owns_origin_chunk || counts.cylindrical_origin_actions_at_compile > 0),
        "zero cylindrical m lost its origin actions after recompilation");
  CHECK(coordinate_state_matches(f, f.step_plans[0]),
        "recompiled zero-m cylindrical fingerprint does not match");

  f.m = +0.5;
  CHECK(and_to_all(!coordinate_state_matches(f, f.step_plans[0])),
        "direct outer cylindrical m mutation was not rejected");
  if (count_processors() == 1) {
    bool rejected = false;
    try {
      f.init_backend();
    }
    catch (const std::runtime_error &) {
      rejected = true;
    }
    CHECK(rejected, "direct outer cylindrical m mutation was not rejected before preparation");
    CHECK(counts.states_created == 2 && counts.states_destroyed == 1 &&
              counts.executables_created == 3,
          "rejected cylindrical mismatch changed resident state or executable counts");
  }
  f.m = 0.0;

  CHECK(f.num_chunks > 0, "cylindrical fingerprint test has no chunks");
  if (f.num_chunks > 0) {
    f.chunks[0]->m = +0.5;
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "per-chunk cylindrical m mutation was not rejected");
    f.chunks[0]->m = 0.0;

    f.chunks[0]->zero_fields_near_cylorigin = !f.chunks[0]->zero_fields_near_cylorigin;
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "per-chunk cylindrical origin-policy mutation was not rejected");
    f.chunks[0]->zero_fields_near_cylorigin = !f.chunks[0]->zero_fields_near_cylorigin;
  }

  const double plan_m = f.step_plans[0]->cylindrical_m;
  f.step_plans[0]->cylindrical_m = +0.5;
  CHECK(!coordinate_state_matches(f, f.step_plans[0]),
        "prepared-plan cylindrical m mutation was not rejected");
  f.step_plans[0]->cylindrical_m = plan_m;

  CHECK(!f.step_plans[0]->cylindrical_origin_r.empty() &&
            !f.step_plans[0]->cylindrical_zero_near_origin.empty(),
        "prepared cylindrical plan lacks per-chunk origin fingerprints");
  if (!f.step_plans[0]->cylindrical_origin_r.empty()) {
    f.step_plans[0]->cylindrical_origin_r[0] += 1.0;
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "prepared-plan cylindrical origin mutation was not rejected");
    f.step_plans[0]->cylindrical_origin_r[0] -= 1.0;
  }
  if (!f.step_plans[0]->cylindrical_zero_near_origin.empty()) {
    f.step_plans[0]->cylindrical_zero_near_origin[0] ^= 1;
    CHECK(!coordinate_state_matches(f, f.step_plans[0]),
          "prepared-plan cylindrical origin-policy mutation was not rejected");
    f.step_plans[0]->cylindrical_zero_near_origin[0] ^= 1;
  }
  CHECK(coordinate_state_matches(f, f.step_plans[0]),
        "restored cylindrical fingerprint does not match");

  f.step_plans[0]->cylindrical_origin_r.pop_back();
  CHECK(!coordinate_state_matches(f, f.step_plans[0]),
        "malformed cylindrical origin vector was not rejected");

  structure refused_structure(gv, eps_slab, pml(0.5), identity(), 2);
  fields refused(&refused_structure, +1.0, 0, true, 64, 64, std::vector<double>{0, 0, 0});
  for (component c : components)
    refused.require_component(c);
  lifetime_counts refused_counts;
  refused.backend = new tracking_backend(refused, refused_counts);
  refused.advance(1);
  bool refused_owns_chunk = false;
  int refused_owned_chunk = -1;
  for (int i = 0; i < refused.num_chunks; ++i)
    if (refused.chunks[i]->is_mine()) {
      refused_owns_chunk = true;
      if (refused_owned_chunk < 0) refused_owned_chunk = i;
    }
  BackendState *refused_state = refused.backend_state;
  Executable *refused_executable = refused.executable;
  realnum *const refused_imaginary =
      refused_owned_chunk < 0 ? NULL : refused.chunks[refused_owned_chunk]->f[Er][1];
  refused_counts.fail_rebuild = true;
  bool rejected = false;
  try {
    refused.change_m(0.0);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(and_to_all(rejected), "failed complex-to-real migration was not rejected collectively");
  CHECK(refused_counts.rebuilds == 1 &&
            (!refused_owns_chunk || refused_counts.rebuild_saw_live_imaginary),
        "failed migration did not observe the old imaginary catalog and pointers");
  CHECK(refused.backend_state == refused_state && refused.executable == refused_executable &&
            refused.m == +1.0 && !refused.is_real &&
            (refused_owned_chunk < 0 ||
             refused.chunks[refused_owned_chunk]->f[Er][1] == refused_imaginary),
        "failed complex-to-real migration partially mutated live resident state");
  for (int i = 0; i < refused.num_chunks; ++i)
    CHECK(refused.chunks[i]->m == +1.0,
          "failed complex-to-real migration partially changed chunk %d m", i);
  refused_counts.fail_rebuild = false;
}

static void test_classification_change_recompiles() {
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);

  CHECK(f->prepared_classification_hash != 0, "classification did not publish a hash");
  ++f->prepared_classification_hash;
  invalidate(*f, MutationKind::material_values, "backend_api:changed_classification_hash");
  f->advance(1);
  CHECK(counts.executables_created == 2 && counts.executables_destroyed == 1,
        "a changed classification hash did not recompile the executable");

  delete f;
  delete s;
}

static MaterialRecipeInput material_recipe_input(const MaterialRecipe &recipe) {
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
  input.dense_fallback_rows = recipe.dense_fallback_rows();
  input.callback_tiles = recipe.callback_tiles();
  input.callback_owners = recipe.callback_owners();
  input.topology = recipe.topology();
  input.ir = recipe.ir();
  return input;
}

static MaterialClassificationFacts classification_facts_input(
    const fields &f, const StoragePlan &plan, const MaterialClassification &classification) {
  MaterialClassificationFacts facts;
  facts.required_components = classification.required_components;
  facts.has_nonlinearities = classification.has_nonlinearities;
  facts.aniso2d = classification.aniso2d && f.beta == 0;
  facts.min_decimation_factor = classification.min_decimation_factor;
  for (size_t i = 0; i < plan.arrays.size(); ++i)
    if (plan.arrays[i].role == array_role::material)
      facts.rows.push_back(
          MaterialRowClassificationFact{plan.keys[i], classification.provisional_row_state[i]});
  for (const MaterialVariantClassificationFact &variant : classification.variant_facts)
    if (f.chunks[variant.chunk]->is_mine()) facts.variants.push_back(variant);
  return facts;
}

static void expect_material_classification_facts_rejected(
    fields &f, const StoragePlan &plan, const MaterialRecipe &recipe,
    const MaterialClassificationFacts &facts, const char *label) {
  bool rejected = false;
  try { (void)assemble_material_classification(f, plan, recipe, facts); }
  catch (const std::runtime_error &) { rejected = true; }
  catch (const std::invalid_argument &) { rejected = true; }
  CHECK(and_to_all(rejected), "%s was accepted", label);
}

static void expect_material_recipe_rejected(const MaterialRecipeInput &input,
                                            const char *label) {
  bool rejected = false;
  try {
    MaterialRecipe invalid(input);
    (void)invalid;
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  catch (const std::overflow_error &) {
    rejected = true;
  }
  CHECK(rejected, "%s was accepted", label);
}

static void expect_material_ir_rejected(MaterialIR malformed, const char *label) {
  refresh_material_ir_signatures_for_testing(malformed);
  bool rejected = false;
  try { validate_material_ir(malformed); }
  catch (const std::invalid_argument &) { rejected = true; }
  catch (const std::overflow_error &) { rejected = true; }
  CHECK(rejected, "%s was accepted", label);
}

static uint64_t expected_compact_material_ir_bytes(const MaterialIR &ir) {
  uint64_t total = 0;
  const auto add = [&total](size_t count, size_t width) {
    CHECK(count <= std::numeric_limits<uint64_t>::max() / width &&
              uint64_t(count) * uint64_t(width) <=
                  std::numeric_limits<uint64_t>::max() - total,
          "material IR compact-byte test fixture overflowed");
    total += uint64_t(count) * uint64_t(width);
  };
#define ADD_IR_SCALAR(member) add(1, sizeof(member))
  ADD_IR_SCALAR(ir.version);
  ADD_IR_SCALAR(ir.eps_averaging);
  ADD_IR_SCALAR(ir.subpixel_tol);
  ADD_IR_SCALAR(ir.subpixel_maxeval);
  ADD_IR_SCALAR(ir.ensure_periodicity);
  ADD_IR_SCALAR(ir.contains_host_callback);
  ADD_IR_SCALAR(ir.device_native_eligible);
  ADD_IR_SCALAR(ir.requires_hybrid);
  ADD_IR_SCALAR(ir.prism_include_boundaries);
  ADD_IR_SCALAR(ir.dimensions);
  ADD_IR_SCALAR(ir.projection_offset);
  ADD_IR_SCALAR(ir.default_material);
  ADD_IR_SCALAR(ir.root_count);
  ADD_IR_SCALAR(ir.signature);
  ADD_IR_SCALAR(ir.layout_signature);
  add(ir.cell.size(), sizeof(double));
  add(6, sizeof(ir.captured_volume[0]));
  add(3, sizeof(ir.lattice_basis_size[0]));
  add(9, sizeof(ir.lattice_basis[0]));
  add(9, sizeof(ir.lattice_metric[0]));
  add(9, sizeof(ir.lattice_inverse[0]));
  add(9, sizeof(ir.lattice_inverse_transpose[0]));
  add(ir.extra_materials.size(), sizeof(uint32_t));
  for (const MaterialIRMaterial &m : ir.materials) {
    ADD_IR_SCALAR(m.kind);
    ADD_IR_SCALAR(m.host_callback);
    ADD_IR_SCALAR(m.owned_callback);
    ADD_IR_SCALAR(m.callback_id);
    ADD_IR_SCALAR(m.callback_signature);
    ADD_IR_SCALAR(m.callback_capabilities);
    ADD_IR_SCALAR(m.do_averaging);
    ADD_IR_SCALAR(m.material_grid_kind);
    ADD_IR_SCALAR(m.material_grid_trivial);
    ADD_IR_SCALAR(m.has_conductivity);
    ADD_IR_SCALAR(m.has_chi2);
    ADD_IR_SCALAR(m.has_chi3);
    ADD_IR_SCALAR(m.e_susceptibilities);
    ADD_IR_SCALAR(m.h_susceptibilities);
    add(m.comparison_medium.size(), sizeof(double));
    add(m.parameters.size(), sizeof(double));
    add(m.samples.size(), sizeof(double));
  }
  for (const MaterialIRObject &o : ir.objects) {
    ADD_IR_SCALAR(o.kind);
    ADD_IR_SCALAR(o.material);
    ADD_IR_SCALAR(o.source_identity);
    ADD_IR_SCALAR(o.root_identity);
    ADD_IR_SCALAR(o.leaf_ordinal);
    add(3, sizeof(o.parent_shift[0]));
    add(3, sizeof(o.low[0]));
    add(3, sizeof(o.high[0]));
    ADD_IR_SCALAR(o.fixed_vertex_count);
    ADD_IR_SCALAR(o.vertex_offset);
    ADD_IR_SCALAR(o.vertex_count);
    ADD_IR_SCALAR(o.triangle_offset);
    ADD_IR_SCALAR(o.triangle_count);
    ADD_IR_SCALAR(o.bvh_offset);
    ADD_IR_SCALAR(o.bvh_count);
    ADD_IR_SCALAR(o.mesh_lengthscale);
    add(o.parameters.size(), sizeof(double));
    add(o.vertices.size(), sizeof(double));
    add(o.indices.size(), sizeof(double));
    add(o.auxiliary.size(), sizeof(double));
  }
  add(ir.geometry_vertices.size(), sizeof(double));
  add(ir.geometry_triangles.size(), sizeof(MaterialIRTriangle));
  add(ir.geometry_bvh.size(), sizeof(MaterialIRBvhNode));
  add(ir.geometry_bvh_face_ids.size(), sizeof(uint32_t));
  for (const MaterialIRGeometryImage &image : ir.images) {
    ADD_IR_SCALAR(image.object);
    ADD_IR_SCALAR(image.ordinal);
    ADD_IR_SCALAR(image.precedence);
    add(3, sizeof(image.image[0]));
    add(3, sizeof(image.shift[0]));
    add(3, sizeof(image.low[0]));
    add(3, sizeof(image.high[0]));
  }
  add(ir.active_images.size(), sizeof(uint32_t));
  for (const MaterialIRSusceptibility &s : ir.susceptibilities) {
    ADD_IR_SCALAR(s.identity);
    ADD_IR_SCALAR(s.material);
    ADD_IR_SCALAR(s.field_type);
    ADD_IR_SCALAR(s.material_ordinal);
    add(s.parameters.size(), sizeof(double));
  }
  for (const MaterialIRChunk &c : ir.chunks) {
    ADD_IR_SCALAR(c.chunk);
    ADD_IR_SCALAR(c.dimensions);
    ADD_IR_SCALAR(c.owned);
    ADD_IR_SCALAR(c.resolution);
    ADD_IR_SCALAR(c.inva);
    ADD_IR_SCALAR(c.elements);
    ADD_IR_SCALAR(c.component_bits);
    add(3, sizeof(c.extents[0]));
    add(3, sizeof(c.strides[0]));
    add(3, sizeof(c.little_corner[0]));
    add(3, sizeof(c.big_corner[0]));
    add(3, sizeof(c.origin[0]));
    add(NUM_FIELD_COMPONENTS * 3, sizeof(c.stagger[0][0]));
    add(NUM_FIELD_COMPONENTS * 3, sizeof(c.loop_begin[0][0]));
    add(NUM_FIELD_COMPONENTS * 3, sizeof(c.loop_end[0][0]));
    add(NUM_FIELD_COMPONENTS, sizeof(c.loop_count[0]));
    add(6, sizeof(c.pml_elements[0]));
  }
  for (const MaterialIRPml &p : ir.absorbers) {
    ADD_IR_SCALAR(p.direction);
    ADD_IR_SCALAR(p.side);
    ADD_IR_SCALAR(p.thickness);
    ADD_IR_SCALAR(p.r_asymptotic);
    ADD_IR_SCALAR(p.mean_stretch);
    ADD_IR_SCALAR(p.sample_spacing);
    add(p.samples.size(), sizeof(double));
  }
  for (const MaterialIRPmlAxis &p : ir.pml_axes) {
    ADD_IR_SCALAR(p.chunk);
    ADD_IR_SCALAR(p.direction);
    ADD_IR_SCALAR(p.elements);
    ADD_IR_SCALAR(p.little_corner);
    ADD_IR_SCALAR(p.resolution);
    ADD_IR_SCALAR(p.profile_active);
    ADD_IR_SCALAR(p.analytic_quadratic);
    ADD_IR_SCALAR(p.thickness);
    ADD_IR_SCALAR(p.boundary_location);
    ADD_IR_SCALAR(p.r_asymptotic);
    ADD_IR_SCALAR(p.mean_stretch);
    ADD_IR_SCALAR(p.profile_integral);
    ADD_IR_SCALAR(p.profile_integral_u);
    add(p.profile_samples.size(), sizeof(double));
    add(p.sigma.size(), sizeof(double));
    add(p.kappa.size(), sizeof(double));
    add(p.sigma_inv.size(), sizeof(double));
  }
  for (const MaterialIRTopologyRow &row : ir.topology) {
    ADD_IR_SCALAR(row.key.chunk);
    ADD_IR_SCALAR(row.key.kind);
    ADD_IR_SCALAR(row.key.component_);
    ADD_IR_SCALAR(row.key.cmp);
    ADD_IR_SCALAR(row.key.aux);
    ADD_IR_SCALAR(row.element_type);
    ADD_IR_SCALAR(row.logical_storage);
    ADD_IR_SCALAR(row.elements);
    ADD_IR_SCALAR(row.alignment);
    ADD_IR_SCALAR(row.yee_component);
    add(3, sizeof(row.extents[0]));
    add(3, sizeof(row.strides[0]));
    add(3, sizeof(row.stagger[0]));
  }
  for (const MaterialIRDestination &destination : ir.destinations) {
    ADD_IR_SCALAR(destination.key.chunk);
    ADD_IR_SCALAR(destination.key.kind);
    ADD_IR_SCALAR(destination.key.component_);
    ADD_IR_SCALAR(destination.key.cmp);
    ADD_IR_SCALAR(destination.key.aux);
    ADD_IR_SCALAR(destination.topology_index);
    ADD_IR_SCALAR(destination.chunk_index);
    ADD_IR_SCALAR(destination.property);
    ADD_IR_SCALAR(destination.component);
    ADD_IR_SCALAR(destination.tensor_direction);
    ADD_IR_SCALAR(destination.tensor_column);
    ADD_IR_SCALAR(destination.offdiagonal);
    ADD_IR_SCALAR(destination.point_count);
  }
  for (const MaterialIRBulkSpan &span : ir.bulk_spans) {
    ADD_IR_SCALAR(span.destination);
    ADD_IR_SCALAR(span.first_point);
    ADD_IR_SCALAR(span.count);
  }
  for (const MaterialIRAnalyticInterface &job : ir.analytic_interfaces) {
    ADD_IR_SCALAR(job.destination);
    ADD_IR_SCALAR(job.point);
    ADD_IR_SCALAR(job.front_material);
    ADD_IR_SCALAR(job.behind_material);
    ADD_IR_SCALAR(job.object);
    ADD_IR_SCALAR(job.image);
    add(3, sizeof(job.normal[0]));
    ADD_IR_SCALAR(job.fill);
  }
  for (const MaterialIRHybridPatch &patch : ir.hybrid_patches) {
    ADD_IR_SCALAR(patch.destination);
    ADD_IR_SCALAR(patch.point);
    ADD_IR_SCALAR(patch.value);
    ADD_IR_SCALAR(patch.front_material);
    ADD_IR_SCALAR(patch.behind_material);
    ADD_IR_SCALAR(patch.object);
    ADD_IR_SCALAR(patch.image);
    ADD_IR_SCALAR(patch.ambiguous);
    ADD_IR_SCALAR(patch.variable_material);
    ADD_IR_SCALAR(patch.variable_causes);
    ADD_IR_SCALAR(patch.adaptive_fallback);
    ADD_IR_SCALAR(patch.negative_fallback);
    ADD_IR_SCALAR(patch.reason);
  }
#undef ADD_IR_SCALAR
  return total;
}

static void test_geometry_backed_material_ir() {
  using namespace meep_geom;
  grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, unit_epsilon, pml(0.25), identity(), 2);
  material_type dielectric = make_dielectric(3.25);
  meep_geom::susceptibility e0 = meep_geom::susceptibility();
  e0.frequency = 0.61; e0.gamma = 0.07; e0.sigma_diag = make_vector3(1.0, 0.0, 0.0);
  meep_geom::susceptibility e1 = meep_geom::susceptibility();
  e1.frequency = 0.83; e1.gamma = 0.09; e1.sigma_diag = make_vector3(0.0, 1.0, 0.0);
  meep_geom::susceptibility e0_duplicate = e0;
  e0_duplicate.sigma_diag = make_vector3(0.0, 0.0, 2.0);
  e0_duplicate.sigma_offdiag = make_vector3(0.25, 0.5, 0.75);
  dielectric->medium.E_susceptibilities.push_back(e0);
  dielectric->medium.E_susceptibilities.push_back(e1);
  dielectric->medium.E_susceptibilities.push_back(e0_duplicate);
  meep_geom::susceptibility h0 = meep_geom::susceptibility();
  h0.frequency = 0.47; h0.gamma = 0.04; h0.sigma_diag = make_vector3(1.0, 0.0, 0.0);
  meep_geom::susceptibility h1 = meep_geom::susceptibility();
  h1.frequency = 0.72; h1.gamma = 0.06; h1.sigma_diag = make_vector3(0.0, 1.0, 0.0);
  meep_geom::susceptibility h0_duplicate = h0;
  h0_duplicate.sigma_diag = make_vector3(0.0, 0.0, 3.0);
  h0_duplicate.sigma_offdiag = make_vector3(0.35, 0.55, 0.85);
  dielectric->medium.H_susceptibilities.push_back(h0);
  dielectric->medium.H_susceptibilities.push_back(h1);
  dielectric->medium.H_susceptibilities.push_back(h0_duplicate);
  geometric_object object = make_block(dielectric, make_vector3(), make_vector3(1, 0, 0),
                                       make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                                       make_vector3(0.7, 0.5, 1.0));
  geometric_object inner = make_geometric_object(dielectric, make_vector3());
  inner.which_subclass = geometric_object::COMPOUND_GEOMETRIC_OBJECT;
  inner.subclass.compound_geometric_object_data =
      static_cast<compound_geometric_object *>(calloc(1, sizeof(compound_geometric_object)));
  inner.subclass.compound_geometric_object_data->component_objects.num_items = 2;
  inner.subclass.compound_geometric_object_data->component_objects.items =
      static_cast<geometric_object *>(calloc(2, sizeof(geometric_object)));
  geometric_object_copy(&object,
                        inner.subclass.compound_geometric_object_data->component_objects.items);
  geometric_object_copy(&object,
                        inner.subclass.compound_geometric_object_data->component_objects.items + 1);
  inner.subclass.compound_geometric_object_data->component_objects.items[1].center.x += 0.125;
  geometric_object outer = make_geometric_object(dielectric, make_vector3());
  outer.which_subclass = geometric_object::COMPOUND_GEOMETRIC_OBJECT;
  outer.subclass.compound_geometric_object_data =
      static_cast<compound_geometric_object *>(calloc(1, sizeof(compound_geometric_object)));
  outer.subclass.compound_geometric_object_data->component_objects.num_items = 1;
  outer.subclass.compound_geometric_object_data->component_objects.items =
      static_cast<geometric_object *>(calloc(1, sizeof(geometric_object)));
  geometric_object_copy(&inner,
                        outer.subclass.compound_geometric_object_data->component_objects.items);
  const double infinite_extent = std::numeric_limits<double>::infinity();
  geometric_object infinite_block =
      make_block(dielectric, make_vector3(), make_vector3(1, 0, 0), make_vector3(0, 1, 0),
                 make_vector3(0, 0, 1), make_vector3(infinite_extent, 0.25, 1.0));
  const vector3 prism_vertices[3] = {make_vector3(-0.2, -0.2), make_vector3(0.2, -0.2),
                                     make_vector3(0.0, 0.2)};
  geometric_object infinite_prism =
      make_prism_with_center(dielectric, make_vector3(), prism_vertices, 3, 1.0,
                             make_vector3(0, 0, 1));
  geometric_object infinite_cylinder =
      make_cylinder(dielectric, make_vector3(), 0.1, 1.0, make_vector3(0, 0, 1));
  geometric_object geometry_items[4] = {outer, infinite_block, infinite_prism, infinite_cylinder};
  geometric_object_list geometry = {4, geometry_items};
  material_type file_material = new material_data();
  file_material->which_subclass = material_data::MATERIAL_FILE;
  file_material->epsilon_dims[0] = 2;
  file_material->epsilon_dims[1] = file_material->epsilon_dims[2] = 1;
  file_material->epsilon_data = new double[2]{1.5, 2.5};
  material_type grid_material = make_material_grid(true, 8.0, 0.45, 0.0);
  grid_material->material_grid_kinds = material_data::U_MEAN;
  grid_material->grid_size = make_vector3(2, 1, 1);
  grid_material->weights = new double[2]{0.2, 0.8};
  grid_material->medium_1 = medium_struct(1.25);
  grid_material->medium_2 = medium_struct(4.5);
  grid_material->trivial = false; // mutable eager cache; IR must derive rather than copy it
  material_type user_material = make_user_material(material_ir_user_function, NULL, true);
  material_type extra_items[3] = {file_material, grid_material, user_material};
  material_type_list extra_materials;
  extra_materials.num_items = 3;
  extra_materials.items = extra_items;
  absorber_list absorbers = create_absorber_list();
  add_absorbing_layer(absorbers, 0.125, ALL_DIRECTIONS, ALL_SIDES, 1e-12, 1.25);
  set_materials_from_geometry(&s, geometry, make_vector3(), true, 1e-5, 4321, true, vacuum,
                              absorbers, extra_materials);
  CHECK(s.material_ir.get() != NULL, "geometry-backed material setup did not publish an IR");
  const std::shared_ptr<const void> retained = s.material_ir;
  const MaterialIR *ir = static_cast<const MaterialIR *>(retained.get());
  validate_material_ir(*ir);
  CHECK(ir->eps_averaging && ir->subpixel_maxeval == 4321 && ir->ensure_periodicity &&
            ir->projection_offset == 0.0 &&
            ir->root_count == 4 && ir->objects.size() == 5 &&
            ir->objects[0].root_identity == 0 && ir->objects[0].leaf_ordinal == 0 &&
            !ir->images.empty() &&
            !ir->absorbers.empty(),
        "geometry-backed IR omitted geometry or absorber input");
  {
    int root_precedence = -1;
    int periodic_groups = 0;
    for (size_t i = 0; i + 1 < ir->images.size(); ++i) {
      const MaterialIRGeometryImage &first = ir->images[i];
      const MaterialIRGeometryImage &second = ir->images[i + 1];
      if (ir->objects[first.object].root_identity != 0 ||
          ir->objects[second.object].root_identity != 0 || first.object != 0 ||
          second.object != 1)
        continue;
      bool same_image = true;
      for (int axis = 0; axis < 3; ++axis)
        same_image = same_image && first.image[axis] == second.image[axis];
      if (!same_image) continue;
      CHECK(second.precedence == first.precedence - 1,
            "compound leaf precedence did not decrement within a periodic image");
      if (root_precedence < 0) root_precedence = first.precedence;
      CHECK(first.precedence == root_precedence,
            "compound root precedence did not reset across periodic images");
      ++periodic_groups;
    }
    CHECK(periodic_groups > 1,
          "multi-leaf compound fixture did not retain multiple periodic image groups");
  }
  bool saw_infinite_block = false;
  for (const MaterialIRObject &captured : ir->objects) {
    saw_infinite_block =
        saw_infinite_block ||
        (captured.kind == geometric_object::BLOCK && captured.parameters.size() >= 15 &&
         std::isinf(captured.parameters[12]) && captured.parameters[12] > 0.0);
  }
  CHECK(saw_infinite_block, "material IR rejected or altered a legal infinite block extent");
  {
    MaterialIR malformed = *ir;
    bool found = false;
    for (size_t object_index = 0; object_index < malformed.objects.size(); ++object_index) {
      MaterialIRObject &captured = malformed.objects[object_index];
      if (captured.kind != geometric_object::BLOCK || captured.parameters.size() < 15 ||
          !std::isinf(captured.parameters[12]))
        continue;
      captured.low[1] = -0.09375;
      captured.high[1] = 0.125;
      for (MaterialIRGeometryImage &image : malformed.images)
        if (image.object == object_index)
          image.low[1] = -0.09375 + image.shift[1],
          image.high[1] = 0.125 + image.shift[1];
      found = true;
      break;
    }
    CHECK(found, "material IR infinite-AABB mutation fixture has no infinite block");
    if (found)
      expect_material_ir_rejected(malformed, "coordinated infinite-shape transverse AABB");
  }
  {
    MaterialIR legal = *ir;
    bool found = false;
    for (size_t object_index = 0; object_index < legal.objects.size(); ++object_index) {
      MaterialIRObject &captured = legal.objects[object_index];
      if (captured.kind == geometric_object::PRISM) {
        captured.parameters[3] = infinite_extent;
        found = true;
        break;
      }
    }
    CHECK(found, "material IR infinity fixture has no prism");
    if (found) expect_material_ir_rejected(legal, "stale infinite prism fixed state");
    legal = *ir;
    found = false;
    for (size_t object_index = 0; object_index < legal.objects.size(); ++object_index) {
      MaterialIRObject &captured = legal.objects[object_index];
      if (captured.kind == geometric_object::CYLINDER) {
        captured.parameters[7] = infinite_extent;
        found = true;
        break;
      }
    }
    CHECK(found, "material IR infinity fixture has no cylinder");
    if (found) expect_material_ir_rejected(legal, "stale infinite cylinder fixed state");
  }
  bool saw_file = false, saw_grid = false, saw_user = false;
  for (const MaterialIRMaterial &material : ir->materials) {
    saw_file = saw_file || material.kind == material_data::MATERIAL_FILE;
    saw_grid = saw_grid ||
               (material.kind == material_data::MATERIAL_GRID && material.do_averaging &&
                material.material_grid_kind == material_data::U_MEAN &&
                material.material_grid_trivial);
    saw_user = saw_user ||
               (material.kind == material_data::MATERIAL_USER && material.host_callback &&
                material.do_averaging);
  }
  CHECK(saw_file && saw_grid && saw_user && !ir->contains_host_callback &&
            ir->device_native_eligible,
        "material IR omitted or altered FILE/GRID/USER variant metadata");
  CHECK(or_to_all(!ir->pml_axes.empty()) && or_to_all(!ir->topology.empty()),
        "geometry-backed IR omitted distributed PML or topology input");
  for (const MaterialIRChunk &chunk : ir->chunks) {
    CHECK(chunk.resolution > 0 && chunk.inva == 1.0 / chunk.resolution,
          "material IR omitted the chunk resolution");
    FOR_COMPONENTS(c) {
      const bool present = (chunk.component_bits & (uint64_t(1) << int(c))) != 0;
      CHECK(chunk.loop_count[c] == (present ? chunk.elements : 0),
            "material IR component loop count differs from exact Yee traversal");
      if (!present) continue;
      ptrdiff_t maximum = 0;
      for (int axis = 0; axis < 3; ++axis) {
        CHECK(chunk.loop_begin[c][axis] ==
                  chunk.little_corner[axis] + chunk.stagger[c][axis] &&
                  chunk.loop_end[c][axis] ==
                      chunk.big_corner[axis] + chunk.stagger[c][axis],
              "material IR component loop bounds are not exact");
        maximum += ((chunk.loop_end[c][axis] - chunk.little_corner[axis]) / 2) *
                   chunk.strides[axis];
      }
      CHECK(maximum >= 0 && size_t(maximum) < chunk.elements,
            "material IR component loop address exceeds owned storage");
    }
  }
  {
    MaterialIR malformed = *ir;
    malformed.objects[0].root_identity = malformed.root_count;
    expect_material_ir_rejected(malformed, "out-of-range material IR root identity");
    malformed = *ir;
    ++malformed.objects[0].leaf_ordinal;
    expect_material_ir_rejected(malformed, "noncanonical material IR leaf ordinal");
    malformed = *ir;
    malformed.materials[malformed.default_material].kind = 999;
    expect_material_ir_rejected(malformed, "invalid material IR variant tag");
    malformed = *ir;
    for (MaterialIRMaterial &material : malformed.materials)
      if (material.kind == material_data::MEDIUM && !material.parameters.empty()) {
        material.parameters.pop_back();
        break;
      }
    expect_material_ir_rejected(malformed, "short material IR medium schema");
    const auto make_singular_tensor = [](std::vector<double> &values, size_t base,
                                         bool magnetic) {
      const size_t diagonal = base + (magnetic ? 9 : 0);
      const size_t offdiagonal = base + (magnetic ? 12 : 3);
      values[diagonal] = values[diagonal + 1] = values[diagonal + 2] = 1.0;
      for (size_t i = 0; i < 6; ++i) values[offdiagonal + i] = 0.0;
      values[offdiagonal] = 1.0;
    };
    malformed = *ir;
    for (MaterialIRMaterial &material : malformed.materials)
      if (material.kind == material_data::MEDIUM && !material.parameters.empty()) {
        make_singular_tensor(material.parameters, 0, false);
        make_singular_tensor(material.comparison_medium, 0, false);
        break;
      }
    expect_material_ir_rejected(malformed, "singular ordinary geometry tensor");
    malformed = *ir;
    for (MaterialIRMaterial &material : malformed.materials)
      if (material.kind == material_data::MATERIAL_GRID && material.parameters.size() >= 41) {
        make_singular_tensor(material.parameters, 3, false);
        break;
      }
    expect_material_ir_rejected(malformed, "singular MaterialGrid endpoint tensor");
    malformed = *ir;
    for (MaterialIRMaterial &material : malformed.materials)
      if (material.kind == material_data::MATERIAL_GRID &&
          material.comparison_medium.size() >= 18) {
        make_singular_tensor(material.comparison_medium, 0, true);
        break;
      }
    expect_material_ir_rejected(malformed, "singular MaterialGrid comparison tensor");
    malformed = *ir;
    for (MaterialIRMaterial &material : malformed.materials)
      if (material.kind == material_data::MATERIAL_FILE && material.parameters.size() >= 18) {
        make_singular_tensor(material.parameters, 0, true);
        break;
      }
    expect_material_ir_rejected(malformed, "singular FILE magnetic tensor");
    {
      MaterialIR allowed = *ir;
      for (MaterialIRMaterial &material : allowed.materials)
        if (material.kind == material_data::MEDIUM && !material.parameters.empty()) {
          material.parameters[0] = 0.0;
          material.comparison_medium[0] = 0.0;
          break;
        }
      refresh_material_ir_signatures_for_testing(allowed);
      validate_material_ir(allowed);
    }
    {
      MaterialIR allowed = *ir;
      for (MaterialIRMaterial &material : allowed.materials)
        if (material.kind == material_data::MATERIAL_FILE && material.parameters.size() >= 9) {
          make_singular_tensor(material.parameters, 0, false);
          break;
        }
      refresh_material_ir_signatures_for_testing(allowed);
      validate_material_ir(allowed);
    }
    malformed = *ir;
    malformed.susceptibilities[0].parameters.resize(8);
    expect_material_ir_rejected(malformed, "short material IR susceptibility schema");
    malformed = *ir;
    ++malformed.susceptibilities[0].material_ordinal;
    expect_material_ir_rejected(malformed, "wrong material IR susceptibility ordinal");
    malformed = *ir;
    malformed.susceptibilities[0].parameters.push_back(1.0);
    expect_material_ir_rejected(malformed, "trailing material IR susceptibility payload");
    if (!ir->topology.empty()) {
      malformed = *ir;
      malformed.topology[0].extents[0] = 0;
      expect_material_ir_rejected(malformed, "zero material IR topology extent");
      malformed = *ir;
      malformed.topology.pop_back();
      expect_material_ir_rejected(malformed, "missing material IR topology row");
      malformed = *ir;
      MaterialIRTopologyRow extra = malformed.topology.back();
      extra.key.chunk += int(malformed.chunks.size()) + 1;
      malformed.topology.push_back(extra);
      expect_material_ir_rejected(malformed, "extra material IR topology row");
      malformed = *ir;
      malformed.topology[0].elements = std::numeric_limits<size_t>::max();
      expect_material_ir_rejected(malformed, "overflowing material IR topology row");
    }
    if (!ir->pml_axes.empty()) {
      malformed = *ir;
      malformed.pml_axes[0].sigma.pop_back();
      expect_material_ir_rejected(malformed, "short material IR PML profile");
      malformed = *ir;
      malformed.pml_axes[0].profile_samples.pop_back();
      expect_material_ir_rejected(malformed, "short material IR raw PML profile");
      malformed = *ir;
      malformed.pml_axes[0].profile_integral = 0.0;
      expect_material_ir_rejected(malformed, "zero material IR PML profile integral");
      malformed = *ir;
      malformed.pml_axes[0].boundary_location =
          std::numeric_limits<double>::quiet_NaN();
      expect_material_ir_rejected(malformed, "nonfinite material IR PML boundary location");
      malformed = *ir;
      ++malformed.pml_axes[0].elements;
      expect_material_ir_rejected(malformed, "wrong material IR PML extent");
      malformed = *ir;
      malformed.pml_axes.push_back(malformed.pml_axes[0]);
      expect_material_ir_rejected(malformed, "duplicate material IR PML axis");
      malformed = *ir;
      malformed.pml_axes[0].elements = size_t(std::numeric_limits<int>::max()) + 1;
      expect_material_ir_rejected(malformed, "overflowing material IR PML integer extent");
    }
    malformed = *ir;
    malformed.projection_offset = std::numeric_limits<double>::quiet_NaN();
    expect_material_ir_rejected(malformed, "nonfinite material IR projection offset");
    malformed = *ir;
    --malformed.version;
    expect_material_ir_rejected(malformed, "stale material IR schema version");
    malformed = *ir;
    malformed.dimensions = D3;
    expect_material_ir_rejected(malformed, "D2/D3 material IR dimension mismatch");
    malformed = *ir;
    malformed.dimensions = Dcyl;
    expect_material_ir_rejected(malformed, "D2/cylindrical material IR dimension mismatch");
    if (!ir->chunks.empty()) {
      malformed = *ir;
      malformed.chunks[0].resolution = 0.0;
      expect_material_ir_rejected(malformed, "zero material IR chunk resolution");
      malformed = *ir;
      component first = NO_COMPONENT;
      FOR_COMPONENTS(c)
        if (first == NO_COMPONENT &&
            (malformed.chunks[0].component_bits & (uint64_t(1) << int(c))))
          first = c;
      if (first != NO_COMPONENT) {
        ++malformed.chunks[0].loop_end[first][0];
        expect_material_ir_rejected(malformed, "misaligned material IR component loop bound");
        malformed = *ir;
        ++malformed.chunks[0].loop_count[first];
        expect_material_ir_rejected(malformed, "wrong material IR component loop count");
      }
    }
    malformed = *ir;
    MaterialIRObject &mesh = malformed.objects.back();
    mesh.kind = geometric_object::MESH;
    mesh.parameters.assign(4, 0.0);
    mesh.vertices = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    mesh.indices = {0.0, 1.0, 3.0};
    expect_material_ir_rejected(malformed, "out-of-range material IR mesh index");
    malformed = *ir;
    for (MaterialIRObject &candidate : malformed.objects)
      if (candidate.kind == geometric_object::BLOCK) {
        candidate.parameters[24] = double(std::numeric_limits<int>::max()) + 1024.0;
        break;
      }
    expect_material_ir_rejected(malformed, "overflowing material IR block subtype");
    malformed = *ir;
    malformed.objects[0].parameters[0] =
        std::numeric_limits<double>::quiet_NaN();
    expect_material_ir_rejected(malformed, "NaN material IR object center");
    malformed = *ir;
    malformed.objects[0].parameters[0] = infinite_extent;
    expect_material_ir_rejected(malformed, "illegal infinite material IR object center");
    malformed = *ir;
    for (MaterialIRObject &candidate : malformed.objects)
      if (candidate.kind == geometric_object::BLOCK) {
        candidate.parameters[12] = -infinite_extent;
        break;
      }
    expect_material_ir_rejected(malformed, "negative infinite material IR block extent");
    if (!ir->chunks.empty()) {
      malformed = *ir;
      malformed.chunks[0].little_corner[0] = std::numeric_limits<int>::min();
      malformed.chunks[0].big_corner[0] = std::numeric_limits<int>::max() - 1;
      expect_material_ir_rejected(malformed, "overflowing material IR chunk coordinate span");
    }
  }
  std::vector<double> ir_frequencies[2];
  for (const MaterialIRSusceptibility &sus : ir->susceptibilities) {
    const int slot = sus.field_type == E_stuff ? 0 : sus.field_type == H_stuff ? 1 : -1;
    CHECK(slot >= 0 && sus.parameters.size() > 9,
          "material IR susceptibility has an invalid field or short payload");
    if (slot >= 0 && sus.parameters.size() > 9) ir_frequencies[slot].push_back(sus.parameters[9]);
  }
  CHECK(ir_frequencies[0].size() == 2 && ir_frequencies[0][0] == e0.frequency &&
            ir_frequencies[0][1] == e1.frequency && ir_frequencies[1].size() == 2 &&
            ir_frequencies[1][0] == h0.frequency && ir_frequencies[1][1] == h1.frequency,
        "material IR susceptibility identities do not preserve CPU state-index order");
  const uint64_t signature = ir->signature;
  const uint64_t layout_signature = ir->layout_signature;
  {
    MaterialIR route_only = *ir;
    route_only.requires_hybrid = !route_only.requires_hybrid;
    refresh_material_ir_signatures_for_testing(route_only);
    CHECK(route_only.signature == signature && route_only.layout_signature != layout_signature,
          "rank/layout-derived hybrid route polluted the semantic signature");
  }
  {
    const double point[3] = {0.0, 0.0, 0.0};
    uint32_t image = std::numeric_limits<uint32_t>::max();
    const uint32_t material = material_ir_material_at_point(*ir, point, &image);
    const bool selected = material < ir->materials.size() && image < ir->images.size() &&
                          material == uint32_t(ir->objects[ir->images[image].object].material);
    CHECK(or_to_all(selected),
          "pointer-free material evaluator did not select a live geometry image");
    MaterialIR default_grid = *ir;
    default_grid.default_material = default_grid.extra_materials[1];
    const double grid_value = material_ir_grid_value_at_point(
        default_grid, point, std::numeric_limits<uint32_t>::max());
    const double raw = 0.5;
    const double expected_grid =
        (tanh(8.0 * 0.45) + tanh(8.0 * (raw - 0.45))) /
        (tanh(8.0 * 0.45) + tanh(8.0 * (1.0 - 0.45)));
    CHECK(fabs(grid_value - expected_grid) < 1e-14,
          "pointer-free default MaterialGrid evaluator changed interpolation/projection");
  }
  CHECK(or_to_all(!ir->bulk_spans.empty()),
        "geometry partition fixture did not exercise bulk work");
  CHECK(or_to_all(!ir->hybrid_patches.empty()),
        "geometry partition fixture did not exercise patch work");
  const int patch_ranks = sum_to_all(int(!ir->hybrid_patches.empty()));
  CHECK(ir->requires_hybrid && patch_ranks > 0,
        "global hybrid route did not reconcile a rank-local patch");
  if (count_processors() >= 4)
    CHECK(patch_ranks < count_processors(),
          "rank-asymmetric fixture unexpectedly gave every rank a local patch");
  {
    MaterialIR malformed = *ir;
    bool swapped = false;
    for (size_t i = 1; i < malformed.images.size(); ++i)
      if (malformed.images[i - 1].precedence == malformed.images[i].precedence) {
        std::swap(malformed.images[i - 1], malformed.images[i]);
        malformed.images[i - 1].ordinal = uint32_t(i - 1);
        malformed.images[i].ordinal = uint32_t(i);
        swapped = true;
        break;
      }
    CHECK(swapped, "periodic image fixture has no equal-precedence tie");
    if (swapped) expect_material_ir_rejected(malformed, "re-signed equal-precedence image swap");
    malformed = *ir;
    malformed.objects[0].low[0] += 0.03125;
    for (MaterialIRGeometryImage &candidate : malformed.images)
      if (candidate.object == 0) candidate.low[0] += 0.03125;
    expect_material_ir_rejected(malformed, "coordinated stale object/image AABB");
    if (!ir->bulk_spans.empty() && ir->bulk_spans[0].count > 1) {
      malformed = *ir;
      MaterialIRBulkSpan tail = malformed.bulk_spans[0];
      tail.first_point += 1;
      --tail.count;
      malformed.bulk_spans[0].count = 1;
      malformed.bulk_spans.insert(malformed.bulk_spans.begin() + 1, tail);
      expect_material_ir_rejected(malformed, "nonmaximal material IR bulk spans");
    }
    if (!ir->hybrid_patches.empty()) {
      malformed = *ir;
      malformed.hybrid_patches[0].reason = MaterialIRPatchReason::negative_material_fallback;
      malformed.hybrid_patches[0].negative_fallback = false;
      expect_material_ir_rejected(malformed, "re-signed hybrid patch reason");
    }
  }
  {
    MaterialIR projected = *ir;
    projected.projection_offset = 0.125;
    refresh_material_ir_signatures_for_testing(projected);
    validate_material_ir(projected);
    CHECK(projected.signature != signature && projected.layout_signature != layout_signature &&
              !material_ir_equal(projected, *ir),
          "material IR projection offset did not affect both immutable signatures");
  }
  std::vector<double> frozen_file_samples, frozen_grid_samples;
  for (const MaterialIRMaterial &material : ir->materials) {
    if (material.kind == material_data::MATERIAL_FILE) frozen_file_samples = material.samples;
    if (material.kind == material_data::MATERIAL_GRID) frozen_grid_samples = material.samples;
  }
  dielectric->medium.epsilon_diag.x = 99.0;
  file_material->epsilon_data[0] = 77.0;
  grid_material->weights[0] = 0.99;
  grid_material->medium_1.epsilon_diag.x = 88.0;
  bool owned_file = false, owned_grid = false;
  for (const MaterialIRMaterial &material : ir->materials) {
    if (material.kind == material_data::MATERIAL_FILE)
      owned_file = material.samples == frozen_file_samples;
    if (material.kind == material_data::MATERIAL_GRID)
      owned_grid = material.samples == frozen_grid_samples && material.parameters.size() > 3 &&
                   material.parameters[3] != 88.0;
  }
  CHECK(ir->signature == signature && ir->layout_signature == layout_signature && owned_file &&
            owned_grid,
        "material IR borrowed its caller's material payload");
  structure copied(s);
  structure mutated(s);
  CHECK(mutated.material_ir.get() == retained.get(),
        "structure copy did not retain immutable material IR");
  mutated.set_conductivity(Dz, unit_epsilon);
  CHECK(!mutated.material_ir,
        "public coefficient mutation retained stale geometry-backed material IR");
  set_materials_from_geometry(&mutated, geometry, make_vector3(), true, 1e-5, 4321, true,
                              vacuum, absorbers, extra_materials);
  CHECK(mutated.material_ir && mutated.material_ir.get() != retained.get(),
        "geometry material rebuild did not publish a replacement immutable IR");
  validate_material_ir(*static_cast<const MaterialIR *>(mutated.material_ir.get()));
  fields f(&copied);
  CHECK(f.material_ir.get() == retained.get(), "structure/fields copy lost immutable IR ownership");
  validate_material_ir(*material_ir_for(f));
  lifetime_counts counts;
  counts.retain_all_provisional_material_rows = true;
  f.backend = new tracking_backend(f, counts);
  f.advance(1);
  CHECK(f.initialization_plan && f.initialization_plan->materials.size() == 1 &&
            f.initialization_plan->materials[0].eps_averaging() == ir->eps_averaging &&
            f.initialization_plan->materials[0].subpixel_tol() == ir->subpixel_tol &&
            f.initialization_plan->materials[0].subpixel_maxeval() == ir->subpixel_maxeval,
        "material recipe did not inherit its immutable IR policy");
  CHECK(f.initialization_plan && f.initialization_plan->pml.size() == ir->pml_axes.size(),
        "initialization plan did not preserve every exact PML axis recipe");
  CHECK(f.storage_plan && f.array_catalog &&
            f.storage_plan->arrays.size() == f.array_catalog->size() &&
            counts.arrays_at_compile == counts.arrays_at_create &&
            or_to_all(counts.retained_logical_suffix_at_compile > 0),
        "retained provisional material IDs were not published through compile");
  if (f.initialization_plan) {
    const MaterialRecipe &geometry_recipe = f.initialization_plan->materials[0];
    const MaterialSupportDecision support = classify_material_support(geometry_recipe);
    uint64_t exact_dense_bytes = 0;
    for (const MaterialRecipeRow &row : geometry_recipe.dense_fallback_rows())
      exact_dense_bytes += row.values.size();
    CHECK(geometry_recipe.disposition() == MaterialRecipeDisposition::hybrid_interface &&
              support.disposition == MaterialRecipeDisposition::hybrid_interface &&
              support.reason_bits == material_support_none &&
              support.compact_input_bytes == expected_compact_material_ir_bytes(*ir) &&
              support.dense_fallback_bytes == exact_dense_bytes,
          "geometry-backed recipe did not freeze its hybrid support route");
    CHECK(ir->extra_materials.size() == 3 && !ir->contains_host_callback,
          "support fixture lost its unused extra material variants");

    MaterialRecipeInput malformed_recipe = material_recipe_input(geometry_recipe);
    malformed_recipe.support_reason_bits ^= material_support_unowned_callback;
    expect_material_recipe_rejected(malformed_recipe,
                                    "stale material support reason signature");
    for (size_t row_index = 0; row_index < geometry_recipe.rows().size(); ++row_index)
      if (geometry_recipe.rows()[row_index].key.kind == int(array_kind::sigma)) {
        malformed_recipe = material_recipe_input(geometry_recipe);
        malformed_recipe.ir.reset();
        malformed_recipe.topology.clear();
        malformed_recipe.support_reason_bits = material_support_no_owned_ir;
        malformed_recipe.rows[row_index].key.aux = uint64_t(D_stuff);
        expect_material_recipe_rejected(malformed_recipe,
                                        "invalid material sigma field-type encoding");
        malformed_recipe = material_recipe_input(geometry_recipe);
        malformed_recipe.ir.reset();
        malformed_recipe.topology.clear();
        malformed_recipe.support_reason_bits = material_support_no_owned_ir;
        malformed_recipe.rows[row_index].key.aux =
            (uint64_t(std::numeric_limits<int>::max()) + 1) * NUM_FIELD_TYPES + E_stuff;
        expect_material_recipe_rejected(malformed_recipe,
                                        "overflowing material sigma state identity");
        break;
      }
    malformed_recipe = material_recipe_input(geometry_recipe);
    if (!malformed_recipe.rows.empty()) {
      malformed_recipe.rows.pop_back();
      expect_material_recipe_rejected(malformed_recipe,
                                      "geometry recipe missing dense IR topology key");
    }
    malformed_recipe = material_recipe_input(geometry_recipe);
    if (!malformed_recipe.topology.empty()) {
      malformed_recipe.topology.pop_back();
      expect_material_recipe_rejected(malformed_recipe,
                                      "geometry recipe missing provisional IR topology key");
      malformed_recipe = material_recipe_input(geometry_recipe);
      ++malformed_recipe.topology[0].elements;
      expect_material_recipe_rejected(malformed_recipe,
                                      "geometry recipe replaced IR topology metadata");
    }
    malformed_recipe = material_recipe_input(geometry_recipe);
    if (!ir->topology.empty()) {
      MaterialIRTopologyRow extra = ir->topology.front();
      extra.key.chunk += f.num_chunks + 1;
      malformed_recipe.topology.push_back(extra);
      expect_material_recipe_rejected(malformed_recipe,
                                      "geometry recipe added non-IR topology key");
    }
  }
  {
    const size_t resolved_size = f.storage_plan->arrays.size();
    const size_t host_prefix = f.array_catalog->host_backed_size();
    CHECK(or_to_all(resolved_size > host_prefix),
          "warm material visibility fixture has no provisional logical suffix on any rank");
    BackendState *const retained_state = f.backend_state;
    Executable *const retained_executable = f.executable;
    counts.retain_all_provisional_material_rows = false;
    invalidate(f, MutationKind::material_values,
               "backend_api:warm material suffix elision");
    f.advance(1);
    CHECK(f.backend_state != retained_state && f.executable != retained_executable &&
              f.storage_plan->arrays.size() == resolved_size &&
              f.array_catalog->size() == resolved_size &&
              f.array_catalog->host_backed_size() == host_prefix &&
              counts.arrays_at_compile == resolved_size &&
              counts.retained_logical_suffix_at_compile == 0,
          "warm material reclassification did not rebuild the resolved logical epoch");
    for (size_t i = host_prefix; i < resolved_size; ++i)
      CHECK(f.storage_plan->arrays[i].classification_elided &&
                f.array_catalog->find(f.storage_plan->keys[i]) == invalid_array(),
            "warm material reclassification exposed tombstoned ArrayId %zu", i);

    BackendState *const elided_state = f.backend_state;
    Executable *const elided_executable = f.executable;
    counts.retain_all_provisional_material_rows = true;
    invalidate(f, MutationKind::material_values,
               "backend_api:warm material suffix retention");
    f.advance(1);
    CHECK(f.backend_state != elided_state && f.executable != elided_executable &&
              f.storage_plan->arrays.size() == resolved_size &&
              f.array_catalog->size() == resolved_size &&
              counts.retained_logical_suffix_at_compile == resolved_size - host_prefix,
          "warm material reclassification did not republish retained logical IDs");
    for (size_t i = host_prefix; i < resolved_size; ++i)
      CHECK(!f.storage_plan->arrays[i].classification_elided &&
                f.array_catalog->find(f.storage_plan->keys[i]) == ArrayId{uint32_t(i)},
            "warm material reclassification hid retained ArrayId %zu", i);
  }
  if (f.initialization_plan)
    for (const PmlRecipe &p : f.initialization_plan->pml)
      CHECK(!p.sigma.empty() && p.sigma.size() == p.kappa.size() &&
                p.sigma.size() == p.sigma_inv.size(),
            "initialization plan contains an incomplete exact PML axis recipe");
  std::vector<double> live_frequencies[2];
  for (const PolarizationDescriptor &descriptor : f.descriptors->polarizations) {
    const int slot = descriptor.ft == E_stuff ? 0 : descriptor.ft == H_stuff ? 1 : -1;
    if (slot >= 0 && descriptor.kind == SusceptibilityKind::lorentzian &&
        descriptor.chunk >= 0 && descriptor.state_index == int(live_frequencies[slot].size()))
      live_frequencies[slot].push_back(descriptor.lorentzian.omega_0);
  }
  for (int slot = 0; slot < 2; ++slot) {
    CHECK(or_to_all(!live_frequencies[slot].empty()),
          "geometry-backed susceptibility fixture has no live descriptor");
    bool same_order = live_frequencies[slot].size() == ir_frequencies[slot].size();
    for (size_t i = 0; same_order && i < live_frequencies[slot].size(); ++i)
      same_order = live_frequencies[slot][i] == double(realnum(ir_frequencies[slot][i]));
    CHECK(live_frequencies[slot].empty() || same_order,
          "material IR susceptibility order differs from live descriptor state_index order");
  }
  uint64_t reference_signature = signature;
  broadcast(0, &reference_signature, 1);
  CHECK(signature == reference_signature, "global material IR signature differs across ranks");
  uint64_t rank_zero_layout = layout_signature;
  broadcast(0, &rank_zero_layout, 1);
  CHECK(count_processors() == 1 || or_to_all(layout_signature != rank_zero_layout),
        "rank-local material IR layout signature omitted ownership/layout identity");
  geometric_object_list empty_geometry = {0, NULL};
  structure d1_structure(vol1d(1.5, 8.0), unit_epsilon, pml(0.25), identity(), 1);
  set_materials_from_geometry(&d1_structure, empty_geometry, make_vector3(), true, 1e-5, 128,
                              false, vacuum);
  const MaterialIR *d1_ir = static_cast<const MaterialIR *>(d1_structure.material_ir.get());
  validate_material_ir(*d1_ir);
  CHECK(or_to_all(!d1_ir->pml_axes.empty()), "D1 material IR omitted its PML axis");
  for (const MaterialIRPmlAxis &axis : d1_ir->pml_axes) {
    CHECK(axis.elements > 0 && axis.sigma.size() == axis.elements &&
              axis.kappa.size() == axis.elements && axis.sigma_inv.size() == axis.elements,
          "D1 material IR PML recipe has the wrong extent");
    if (axis.profile_active)
      CHECK(axis.analytic_quadratic && axis.thickness == 0.25 &&
                axis.profile_integral == 1.0 / 3.0 &&
                axis.profile_integral_u == 1.0 / 4.0 &&
                axis.profile_samples.size() == axis.elements,
            "D1 material IR PML recipe omitted compact analytic provenance");
    else
      CHECK(!axis.analytic_quadratic && axis.profile_samples.empty(),
            "inactive D1 PML axis retained an executable profile");
  }
  {
    fields native_fields(&d1_structure);
    lifetime_counts native_counts;
    native_fields.backend = new tracking_backend(native_fields, native_counts);
    native_fields.advance(1);
    const MaterialRecipe &native_recipe = native_fields.initialization_plan->materials[0];
    const MaterialSupportDecision support = classify_material_support(native_recipe);
    CHECK(native_recipe.disposition() == MaterialRecipeDisposition::device_native &&
              support.disposition == MaterialRecipeDisposition::device_native &&
              support.reason_bits == material_support_none &&
              or_to_all(support.native_points > 0) &&
              support.compact_input_bytes > 0,
          "uniform owned material IR was not classified device-native before allocation");
  }
  {
    material_type lower_grid = new material_data();
    material_type upper_grid = new material_data();
    material_type default_grid = new material_data();
    material_type grids[3] = {lower_grid, upper_grid, default_grid};
    const double weights[3] = {0.2, 0.8, 0.4};
    for (int i = 0; i < 3; ++i) {
      grids[i]->which_subclass = material_data::MATERIAL_GRID;
      grids[i]->do_averaging = true;
      grids[i]->material_grid_kinds = material_data::U_MEAN;
      grids[i]->grid_size = make_vector3(1, 1, 1);
      grids[i]->weights = new double[1]{weights[i]};
      grids[i]->medium_1 = medium_struct(1.0);
      grids[i]->medium_2 = medium_struct(4.0);
      grids[i]->beta = 0.0;
      grids[i]->eta = 0.5;
      grids[i]->damping = 0.0;
    }
    geometric_object grid_objects[2] = {
        make_block(lower_grid, make_vector3(), make_vector3(1, 0, 0), make_vector3(0, 1, 0),
                   make_vector3(0, 0, 1), make_vector3(0.75, 0.75, 1.0)),
        make_block(upper_grid, make_vector3(), make_vector3(1, 0, 0), make_vector3(0, 1, 0),
                   make_vector3(0, 0, 1), make_vector3(0.5, 0.5, 1.0))};
    geometric_object_list grid_geometry = {2, grid_objects};
    structure grid_structure(vol2d(1.0, 1.0, 8.0), unit_epsilon, no_pml(), identity(), 1);
    set_materials_from_geometry(&grid_structure, grid_geometry, make_vector3(), true, 1e-5,
                                128, false, default_grid);
    const MaterialIR &grid_ir =
        *static_cast<const MaterialIR *>(grid_structure.material_ir.get());
    const double point[3] = {0.0, 0.0, 0.0};
    uint32_t winner = std::numeric_limits<uint32_t>::max();
    material_ir_material_at_point(grid_ir, point, &winner);
    const bool local_grid_match = winner < grid_ir.images.size() &&
                                  fabs(material_ir_grid_value_at_point(grid_ir, point, winner) -
                                       (weights[1] + weights[0] + weights[2]) / 3.0) < 1e-14;
    CHECK(or_to_all(local_grid_match),
          "ordered overlapping/default MaterialGrid mean composition changed");
    const bool local_grid_patch = !grid_ir.hybrid_patches.empty() &&
                                  grid_ir.hybrid_patches[0].reason ==
                                      MaterialIRPatchReason::material_grid_averaging &&
                                  grid_ir.hybrid_patches[0].variable_material;
    CHECK(or_to_all(local_grid_patch),
          "MaterialGrid patch lost variable-material provenance");
    {
      geometric_object_list object_grid_geometry = {1, &grid_objects[0]};
      structure object_grid_structure(vol2d(1.0, 1.0, 8.0), unit_epsilon, no_pml(),
                                      identity(), 1);
      set_materials_from_geometry(&object_grid_structure, object_grid_geometry, make_vector3(),
                                  true, 1e-5, 128, false, vacuum);
      const MaterialIR &object_grid_ir =
          *static_cast<const MaterialIR *>(object_grid_structure.material_ir.get());
      bool local_object_grid_patch = false;
      for (const MaterialIRHybridPatch &patch : object_grid_ir.hybrid_patches)
        local_object_grid_patch = local_object_grid_patch ||
                                  (patch.variable_causes & material_variable_grid) != 0 &&
                                      patch.variable_material &&
                                      patch.reason ==
                                          MaterialIRPatchReason::material_grid_averaging;
      CHECK(or_to_all(local_object_grid_patch),
            "object MaterialGrid over constant default lost its exact grid cause");
    }
    {
      const int kinds[4] = {material_data::U_DEFAULT, material_data::U_MIN,
                            material_data::U_PROD, material_data::U_MEAN};
      material_type_list no_extra;
      no_extra.num_items = 0;
      no_extra.items = NULL;
      for (int kind : kinds) {
        for (material_type grid : grids)
          grid->material_grid_kinds =
              static_cast<decltype(grid->material_grid_kinds)>(kind);
        structure reducer_structure(vol2d(1.0, 1.0, 8.0), unit_epsilon, no_pml(), identity(), 1);
        set_materials_from_geometry(&reducer_structure, grid_geometry, make_vector3(), false,
                                    1e-5, 128, false, default_grid);
        const MaterialIR &reducer_ir =
            *static_cast<const MaterialIR *>(reducer_structure.material_ir.get());
        geom_epsilon oracle(grid_geometry, no_extra,
                            reducer_structure.gv.pad().surroundings());
        int object_index = -1;
        geom_box_tree tree = geom_tree_search(make_vector3(), oracle.geometry_tree,
                                              &object_index);
        CHECK(tree && object_index >= 0, "MaterialGrid CPU oracle found no winning object");
        const double cpu_value =
            matgrid_val(make_vector3(), tree, object_index,
                        static_cast<material_data *>(tree->objects[object_index].o->material));
        uint32_t reducer_winner = std::numeric_limits<uint32_t>::max();
        material_ir_material_at_point(reducer_ir, point, &reducer_winner);
        const bool local_reducer_match =
            reducer_winner < reducer_ir.images.size() &&
            fabs(material_ir_grid_value_at_point(reducer_ir, point, reducer_winner) - cpu_value) <=
                16.0 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, fabs(cpu_value));
        CHECK(or_to_all(local_reducer_match),
              "MaterialGrid reducer %d diverged from the CPU contributor walk", kind);
        CHECK(reducer_ir.hybrid_patches.empty() && reducer_ir.analytic_interfaces.empty(),
              "non-averaged MaterialGrid reducer %d produced interface work", kind);
      }
      for (material_type grid : grids) grid->material_grid_kinds = material_data::U_MEAN;
    }
    geometric_object_destroy(grid_objects[1]);
    geometric_object_destroy(grid_objects[0]);
    for (material_type grid : grids) material_free(grid);
  }
  {
    material_type block_medium = make_dielectric(5.0);
    geometric_object analytic_block =
        make_block(block_medium, make_vector3(0.03, -0.02, 0.0), make_vector3(1, 0, 0),
                   make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                   make_vector3(0.63, 0.61, 1.0));
    geometric_object_list analytic_geometry = {1, &analytic_block};
    structure analytic_structure(vol2d(1.0, 1.0, 8.0), unit_epsilon, no_pml(), identity(), 1);
    set_materials_from_geometry(&analytic_structure, analytic_geometry, make_vector3(), true,
                                1e-5, 256, false, vacuum);
    const MaterialIR &analytic_ir =
        *static_cast<const MaterialIR *>(analytic_structure.material_ir.get());
    size_t eligible_destinations = 0;
    for (const MaterialIRDestination &destination : analytic_ir.destinations)
      eligible_destinations += destination.property == MaterialIRProperty::chi1inv &&
                               destination.tensor_column >= 0;
    const double inside_point[3] = {0.0, 0.0, 0.0};
    const double outside_point[3] = {0.49, 0.49, 0.0};
    CHECK(analytic_ir.eps_averaging && or_to_all(eligible_destinations > 0),
          "axis-aligned block fixture has no averaging-eligible destination");
    const uint32_t inside_material = material_ir_material_at_point(analytic_ir, inside_point);
    const uint32_t outside_material = material_ir_material_at_point(analytic_ir, outside_point);
    const bool local_interface = inside_material != outside_material;
    CHECK(or_to_all(local_interface),
          "axis-aligned block fixture did not preserve its material interface");
    const bool local_distinct_payloads =
        local_interface && inside_material < analytic_ir.materials.size() &&
        outside_material < analytic_ir.materials.size() &&
        analytic_ir.materials[inside_material].comparison_medium !=
            analytic_ir.materials[outside_material].comparison_medium;
    CHECK(or_to_all(local_distinct_payloads),
          "axis-aligned block fixture collapsed distinct CPU medium equality payloads");
    CHECK(or_to_all(local_interface &&
                    !material_ir_materials_equal(analytic_ir, inside_material,
                                                 outside_material)),
          "axis-aligned block fixture compared distinct CPU media equal");
    CHECK(or_to_all(!analytic_ir.bulk_spans.empty()),
          "axis-aligned block fixture omitted bulk work");
    CHECK(or_to_all(!analytic_ir.analytic_interfaces.empty()),
          "axis-aligned block fixture omitted planar-face analytic work (count=%zu)",
          analytic_ir.analytic_interfaces.size());
    CHECK(or_to_all(!analytic_ir.hybrid_patches.empty()),
          "axis-aligned block fixture omitted edge/corner patch work (count=%zu)",
          analytic_ir.hybrid_patches.size());
    if (!analytic_ir.analytic_interfaces.empty()) {
      const MaterialIRAnalyticInterface &job = analytic_ir.analytic_interfaces[0];
      int nonzero = 0;
      for (int axis = 0; axis < 3; ++axis) nonzero += job.normal[axis] != 0.0;
      CHECK(nonzero == 1 && job.fill > 0.0 && job.fill < 1.0,
            "analytic block face record has a non-planar normal/fill");
      MaterialIR malformed = analytic_ir;
      malformed.analytic_interfaces[0].fill = 0.25;
      expect_material_ir_rejected(malformed, "re-signed analytic fill");
    }
    geometric_object_destroy(analytic_block);
    material_free(block_medium);
  }
  {
    material_type mesh_medium = make_dielectric(6.0);
    const vector3 vertices[8] = {
        make_vector3(-1, -1, -1), make_vector3(1, -1, -1),
        make_vector3(1, 1, -1), make_vector3(-1, 1, -1),
        make_vector3(-1, -1, 1), make_vector3(1, -1, 1),
        make_vector3(1, 1, 1), make_vector3(-1, 1, 1)};
    const int triangles[36] = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
                               0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
                               2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7};
    geometric_object mesh_object = make_mesh(mesh_medium, vertices, 8, triangles, 12);
    geom_fix_object_ptr(&mesh_object);
    geometric_object_list mesh_geometry = {1, &mesh_object};
    structure mesh_structure(vol3d(3.0, 3.0, 3.0, 8.0), unit_epsilon, no_pml(), identity(), 1);
    set_materials_from_geometry(&mesh_structure, mesh_geometry, make_vector3(), false, 1e-5,
                                128, false, vacuum);
    const MaterialIR &mesh_ir =
        *static_cast<const MaterialIR *>(mesh_structure.material_ir.get());
    CHECK(mesh_ir.objects.size() == 1 && mesh_ir.objects[0].kind == geometric_object::MESH &&
              mesh_ir.objects[0].triangle_count == 12 && mesh_ir.objects[0].bvh_count > 1 &&
              mesh_ir.geometry_bvh_face_ids.size() == 12 &&
              fabs(mesh_ir.objects[0].mesh_lengthscale - sqrt(12.0)) <=
                  4.0 * std::numeric_limits<double>::epsilon() * sqrt(12.0),
          "fixed cube mesh triangles/BVH/lengthscale were not captured");
    size_t internal_nodes = 0, leaf_nodes = 0, covered_faces = 0;
    for (size_t i = 0; i < mesh_ir.objects[0].bvh_count; ++i) {
      const MaterialIRBvhNode &node =
          mesh_ir.geometry_bvh[mesh_ir.objects[0].bvh_offset + i];
      if (node.leaf) {
        ++leaf_nodes;
        covered_faces += node.triangle_count;
        CHECK(node.triangle_count >= 1 && node.triangle_count <= 4,
              "mesh BVH leaf has a noncanonical face count");
      }
      else {
        ++internal_nodes;
        CHECK(node.triangle_count == 0 && node.first_triangle == 0,
              "mesh BVH internal node retained a leaf span");
      }
    }
    CHECK(internal_nodes > 0 && leaf_nodes > 1 && covered_faces == 12,
          "cube mesh did not force a complete multi-node BVH");
    MaterialIR global_mesh_ir = mesh_ir;
    global_mesh_ir.active_images.clear();
    for (uint32_t image = 0; image < global_mesh_ir.images.size(); ++image)
      global_mesh_ir.active_images.push_back(image);
    const double mesh_points[][3] = {{0, 0, 0},
                                     {1.25, 0, 0},
                                     {1, 0.125, -0.25},
                                     {std::nextafter(1.0, 0.0), 0.125, -0.25},
                                     {1.0 - 1e-10, 0.125, -0.25},
                                     {std::nextafter(1.0, infinite_extent), 0.125, -0.25},
                                     {-1, 0.125, -0.25}};
    for (const auto &point_values : mesh_points) {
      const vector3 point = make_vector3(point_values[0], point_values[1], point_values[2]);
      const bool cpu_inside = point_in_fixed_objectp(point, mesh_object);
      const bool ir_inside =
          material_ir_material_at_point(global_mesh_ir, point_values) ==
          uint32_t(global_mesh_ir.objects[0].material);
      CHECK(ir_inside == cpu_inside,
            "pointer-free cube mesh containment diverged at (%g,%g,%g)", point_values[0],
            point_values[1], point_values[2]);
    }
    const double inside[3] = {0.0, 0.0, 0.0};
    const bool local_mesh_inside = material_ir_material_at_point(global_mesh_ir, inside) ==
                                   uint32_t(global_mesh_ir.objects[0].material);
    MaterialIR open_mesh = mesh_ir;
    open_mesh.active_images = global_mesh_ir.active_images;
    open_mesh.objects[0].parameters[3] = 0.0;
    CHECK(local_mesh_inside && material_ir_material_at_point(open_mesh, inside) ==
                                   open_mesh.default_material,
          "pointer-free mesh evaluator did not preserve open-mesh non-containment");
    geometric_object_destroy(mesh_object);
    material_free(mesh_medium);
  }
  {
    material_type atlas_medium = make_dielectric(7.0);
    const auto check_shape = [&](const char *label, geometric_object &shape,
                                 const std::vector<vector3> &points) {
      geom_fix_object_ptr(&shape);
      geometric_object_list atlas_geometry = {1, &shape};
      structure atlas_structure(vol3d(8.0, 8.0, 8.0, 4.0), unit_epsilon, no_pml(),
                                identity(), 1);
      set_materials_from_geometry(&atlas_structure, atlas_geometry, make_vector3(), false,
                                  1e-5, 128, false, vacuum);
      MaterialIR atlas_ir =
          *static_cast<const MaterialIR *>(atlas_structure.material_ir.get());
      atlas_ir.active_images.clear();
      for (uint32_t image = 0; image < atlas_ir.images.size(); ++image)
        atlas_ir.active_images.push_back(image);
      bool saw_inside = false, saw_outside = false;
      for (const vector3 &point : points) {
        const bool cpu_inside = point_in_fixed_objectp(point, shape);
        const double encoded[3] = {point.x, point.y, point.z};
        const bool ir_inside = material_ir_material_at_point(atlas_ir, encoded) ==
                               uint32_t(atlas_ir.objects[0].material);
        CHECK(ir_inside == cpu_inside,
              "%s pointer-free containment diverged at (%.17g,%.17g,%.17g)", label, point.x,
              point.y, point.z);
        saw_inside = saw_inside || cpu_inside;
        saw_outside = saw_outside || !cpu_inside;
      }
      CHECK(saw_inside && saw_outside, "%s containment oracle was vacuous", label);
      CHECK(atlas_ir.hybrid_patches.empty() && atlas_ir.analytic_interfaces.empty() &&
                or_to_all(!atlas_ir.bulk_spans.empty()),
            "%s non-averaged geometry did not remain bulk-only", label);
    };
    geometric_object sphere = make_sphere(atlas_medium, make_vector3(), 1.0);
    check_shape("sphere", sphere,
                {make_vector3(), make_vector3(1, 0, 0),
                 make_vector3(std::nextafter(1.0, 0.0), 0, 0),
                 make_vector3(std::nextafter(1.0, infinite_extent), 0, 0)});
    geometric_object_destroy(sphere);
    geometric_object ellipsoid =
        make_ellipsoid(atlas_medium, make_vector3(), make_vector3(1, 0, 0),
                       make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                       make_vector3(4, 2, 6));
    check_shape("ellipsoid", ellipsoid,
                {make_vector3(), make_vector3(2, 0, 0), make_vector3(0, 1, 0),
                 make_vector3(0, 0, 3),
                 make_vector3(std::nextafter(2.0, infinite_extent), 0, 0)});
    geometric_object_destroy(ellipsoid);
    geometric_object cylinder =
        make_cylinder(atlas_medium, make_vector3(), 1.0, 2.0, make_vector3(0, 0, 1));
    check_shape("cylinder", cylinder,
                {make_vector3(), make_vector3(1, 0, 0), make_vector3(0, 0, 1),
                 make_vector3(std::nextafter(1.0, infinite_extent), 0, 0),
                 make_vector3(0, 0, 1.01)});
    geometric_object_destroy(cylinder);
    geometric_object cone =
        make_cone(atlas_medium, make_vector3(), 1.0, 2.0, make_vector3(0, 0, 1), 0.5);
    check_shape("cone", cone,
                {make_vector3(1, 0, -1), make_vector3(0.75, 0, 0),
                 make_vector3(0.5, 0, 1),
                 make_vector3(std::nextafter(0.75, infinite_extent), 0, 0)});
    geometric_object_destroy(cone);
    geometric_object wedge = make_wedge(atlas_medium, make_vector3(), 1.0, 2.0,
                                         make_vector3(0, 0, 1), 0.5 * M_PI,
                                         make_vector3(1, 0, 0));
    check_shape("wedge", wedge,
                {make_vector3(0.5, 0.5, 0), make_vector3(-0.5, 0.5, 0),
                 make_vector3(0.5, -0.5, 0), make_vector3(0, 0.5, 0)});
    geometric_object_destroy(wedge);
    geometric_object negative_wedge =
        make_wedge(atlas_medium, make_vector3(), 1.0, 2.0, make_vector3(0, 0, 1),
                   -0.5 * M_PI, make_vector3(1, 0, 0));
    check_shape("negative wedge", negative_wedge,
                {make_vector3(0.5, -0.5, 0), make_vector3(0.5, 0.5, 0),
                 make_vector3(0, -0.5, 0), make_vector3(-1e-12, -0.5, 0)});
    geometric_object_destroy(negative_wedge);
    geometric_object transformed_block =
        make_block(atlas_medium, make_vector3(0.25, -0.5, 0.125), make_vector3(0, 1, 0),
                   make_vector3(-1, 0, 0), make_vector3(0, 0, 1), make_vector3(2, 1, 3));
    check_shape("transformed block", transformed_block,
                {make_vector3(0.25, -0.5, 0.125), make_vector3(-0.15, 0.4, 1.525),
                 make_vector3(0.25, 0.51, 0.125)});
    geometric_object_destroy(transformed_block);
    const vector3 square[4] = {make_vector3(-1, -1), make_vector3(1, -1),
                               make_vector3(1, 1), make_vector3(-1, 1)};
    geometric_object square_prism =
        make_prism_with_center(atlas_medium, make_vector3(), square, 4, 2.0,
                               make_vector3(0, 0, 1));
    check_shape("prism", square_prism,
                {make_vector3(), make_vector3(1, 0, 0), make_vector3(1.01, 0, 0)});
    {
      geometric_object_list prism_geometry = {1, &square_prism};
      structure prism_structure(vol3d(4.0, 4.0, 4.0, 4.0), unit_epsilon, no_pml(),
                                identity(), 1);
      set_materials_from_geometry(&prism_structure, prism_geometry, make_vector3(), false,
                                  1e-5, 128, false, vacuum);
      MaterialIR include_ir =
          *static_cast<const MaterialIR *>(prism_structure.material_ir.get());
      include_ir.active_images.clear();
      for (uint32_t image = 0; image < include_ir.images.size(); ++image)
        include_ir.active_images.push_back(image);
      MaterialIR exclude_ir = include_ir;
      include_ir.prism_include_boundaries = true;
      exclude_ir.prism_include_boundaries = false;
      const double edge[3] = {1.0, 0.0, 0.0};
      CHECK(material_ir_material_at_point(include_ir, edge) ==
                uint32_t(include_ir.objects[0].material) &&
                material_ir_material_at_point(exclude_ir, edge) == exclude_ir.default_material,
            "prism side-boundary include/exclude policy diverged");
      const prism &fixed_prism = *square_prism.subclass.prism_data;
      CHECK(node_in_or_on_polygon(make_vector3(1, 0, 0), fixed_prism.vertices_p.items,
                                  fixed_prism.vertices_p.num_items, true) &&
                !node_in_or_on_polygon(make_vector3(1, 0, 0), fixed_prism.vertices_p.items,
                                       fixed_prism.vertices_p.num_items, false),
            "libctl prism boundary oracle did not distinguish include/exclude");
    }
    geometric_object_destroy(square_prism);
    material_free(atlas_medium);
  }
  {
    material_type periodic_media[3] = {make_dielectric(2.0), make_dielectric(3.0),
                                       make_dielectric(3.0)};
    geometric_object periodic_objects[3] = {
        make_sphere(periodic_media[0], make_vector3(), 1.3),
        make_sphere(periodic_media[1], make_vector3(), 0.25),
        make_sphere(periodic_media[2], make_vector3(), 0.10)};
    geometric_object_list periodic_geometry = {3, periodic_objects};
    grid_volume periodic_volume = vol2d(2.0, 2.0, 8.0);
    periodic_volume.center_origin();
    structure periodic_structure(periodic_volume, unit_epsilon, no_pml(), identity(), 1);
    geom_epsilon *periodic_oracle =
        make_geom_epsilon(&periodic_structure, &periodic_geometry, make_vector3(), true, vacuum);
    const double tie_x = -0.8;
    const vector3 tie_point = make_vector3(tie_x, 0, 0);
    vector3 cpu_shift = make_vector3();
    bool cpu_object = false;
    for (int ix = -1; ix <= 1 && !cpu_object; ++ix)
      for (int iy = -1; iy <= 1 && !cpu_object; ++iy) {
        const vector3 shift = make_vector3(ix * geometry_lattice.size.x,
                                           iy * geometry_lattice.size.y, 0);
        if (point_in_fixed_objectp(vector3_minus(tie_point, shift),
                                   periodic_oracle->geometry.items[0]))
          cpu_shift = shift, cpu_object = true;
      }
    MaterialIR periodic_ir = *static_cast<const MaterialIR *>(
        capture_material_ir(periodic_structure, *periodic_oracle, false, 1e-5, 128, NULL).get());
    periodic_ir.active_images.clear();
    for (uint32_t image = 0; image < periodic_ir.images.size(); ++image)
      periodic_ir.active_images.push_back(image);
    const double selection_points[3][3] = {{0, 0, 0}, {0.20, 0, 0}, {0.50, 0, 0}};
    const uint32_t expected_roots[3] = {2, 1, 0};
    for (int i = 0; i < 3; ++i) {
      uint32_t image = std::numeric_limits<uint32_t>::max();
      material_ir_material_at_point(periodic_ir, selection_points[i], &image);
      CHECK(image < periodic_ir.images.size() &&
                periodic_ir.objects[periodic_ir.images[image].object].root_identity ==
                    expected_roots[i],
            "periodic overlap precedence selected the wrong root at sample %d", i);
    }
    const double encoded_tie[3] = {tie_x, 0, 0};
    uint32_t tie_image = std::numeric_limits<uint32_t>::max();
    material_ir_material_at_point(periodic_ir, encoded_tie, &tie_image);
    uint32_t first_tie_image = std::numeric_limits<uint32_t>::max();
    bool first_tie_contains = false;
    for (uint32_t image = 0; image < periodic_ir.images.size(); ++image)
      if (periodic_ir.objects[periodic_ir.images[image].object].root_identity == 0 &&
          periodic_ir.images[image].shift[0] == cpu_shift.x &&
          periodic_ir.images[image].shift[1] == cpu_shift.y) {
        first_tie_image = image;
        MaterialIR one_image = periodic_ir;
        one_image.active_images.assign(1, image);
        first_tie_contains = material_ir_material_at_point(one_image, encoded_tie) ==
                             uint32_t(one_image.objects[one_image.images[image].object].material);
        break;
      }
    CHECK(cpu_object && tie_image < periodic_ir.images.size() &&
              periodic_ir.objects[periodic_ir.images[tie_image].object].root_identity == 0 &&
              periodic_ir.images[tie_image].shift[0] == cpu_shift.x &&
              periodic_ir.images[tie_image].shift[1] == cpu_shift.y &&
              periodic_ir.images[tie_image].shift[2] == cpu_shift.z &&
              periodic_ir.images[tie_image].image[0] == -1,
          "equal-precedence periodic tie diverged: image=%u integer=%d shift=%g cpu_shift=%g "
          "precedence=%d cpu_object=%d first=%u first_contains=%d images=%zu cell=%g bounds=%g,%g",
          tie_image,
          tie_image < periodic_ir.images.size() ? periodic_ir.images[tie_image].image[0] : 99,
          tie_image < periodic_ir.images.size() ? periodic_ir.images[tie_image].shift[0] : 99.0,
          cpu_shift.x,
          tie_image < periodic_ir.images.size() ? periodic_ir.images[tie_image].precedence : -99,
          cpu_object, first_tie_image, first_tie_contains, periodic_ir.images.size(),
          periodic_ir.cell[3], periodic_ir.captured_volume[0], periodic_ir.captured_volume[3]);
    delete periodic_oracle;
    for (int i = 2; i >= 0; --i) {
      geometric_object_destroy(periodic_objects[i]);
      material_free(periodic_media[i]);
    }
  }
  {
    geom_initialize();
    dimensions = 3;
    geometry_center = make_vector3();
    ensure_periodicity = false;
    set_default_material(vacuum);
    geometry_lattice.size = make_vector3(4, 4, 4);
    geometry_lattice.basis_size = make_vector3(1, 1, 1);
    geometry_lattice.basis1 = make_vector3(1, 0, 0);
    geometry_lattice.basis2 = make_vector3(0.5, sqrt(3.0) * 0.5, 0);
    geometry_lattice.basis3 = make_vector3(0, 0, 1);
    geom_fix_lattice();
    material_type skew_medium = make_dielectric(8.0);
    geometric_object skew_sphere = make_sphere(skew_medium, make_vector3(), 1.0);
    geometric_object_list skew_geometry = {1, &skew_sphere};
    material_type_list no_extra;
    no_extra.num_items = 0;
    no_extra.items = NULL;
    structure skew_structure(vol3d(4.0, 4.0, 4.0, 8.0), unit_epsilon, no_pml(), identity(), 1);
    geom_epsilon skew_oracle(skew_geometry, no_extra,
                             skew_structure.gv.pad().surroundings());
    MaterialIR skew_ir = *static_cast<const MaterialIR *>(
        capture_material_ir(skew_structure, skew_oracle, false, 1e-5, 128, NULL).get());
    skew_ir.active_images.clear();
    for (uint32_t image = 0; image < skew_ir.images.size(); ++image)
      skew_ir.active_images.push_back(image);
    const vector3 skew_points[5] = {
        make_vector3(-0.5, 1, 0), make_vector3(0, 1, 0), make_vector3(0.5, 1, 0),
        make_vector3(std::nextafter(1.0, 0.0), 0, 0),
        make_vector3(std::nextafter(1.0, infinite_extent), 0, 0)};
    for (const vector3 &point : skew_points) {
      const bool cpu_inside = point_in_fixed_objectp(point, skew_oracle.geometry.items[0]);
      const double encoded[3] = {point.x, point.y, point.z};
      const bool ir_inside = material_ir_material_at_point(skew_ir, encoded) ==
                             uint32_t(skew_ir.objects[0].material);
      CHECK(ir_inside == cpu_inside,
            "skew-lattice sphere containment diverged at (%g,%g,%g)", point.x, point.y,
            point.z);
    }
    CHECK(skew_ir.lattice_metric[1] == geometry_lattice.metric.c0.y &&
              skew_ir.lattice_metric[3] == geometry_lattice.metric.c1.x &&
              skew_ir.lattice_metric[3] != 0.0,
          "skew-lattice metric was not frozen into the material IR");
    geometric_object_destroy(skew_sphere);
    material_free(skew_medium);
    geom_initialize();
  }
  {
    geometric_object callback_block =
        make_block(user_material, make_vector3(), make_vector3(1, 0, 0),
                   make_vector3(0, 1, 0), make_vector3(0, 0, 1),
                   make_vector3(2, 2, 1));
    geometric_object_list callback_geometry = {1, &callback_block};
    structure callback_structure(vol2d(1.0, 1.0, 8.0), unit_epsilon, no_pml(), identity(), 1);
    geom_epsilon *callback_oracle = make_geom_epsilon(
        &callback_structure, &callback_geometry, make_vector3(), false, vacuum);
    material_ir_user_calls = 0;
    std::shared_ptr<const void> callback_capture =
        capture_material_ir(callback_structure, *callback_oracle, false, 1e-5, 128, NULL);
    const MaterialIR &callback_ir =
        *static_cast<const MaterialIR *>(callback_capture.get());
    CHECK(material_ir_user_calls == 0 && callback_ir.contains_host_callback &&
              !callback_ir.device_native_eligible && callback_ir.hybrid_patches.empty() &&
              callback_ir.analytic_interfaces.empty(),
          "reachable callback was evaluated during capture or entered geometry work");
    set_materials_from_geom_epsilon(&callback_structure, callback_oracle, false, 1e-5, 128);
    fields callback_fields(&callback_structure);
    lifetime_counts callback_counts;
    callback_fields.backend = new tracking_backend(callback_fields, callback_counts);
    callback_fields.advance(1);
    CHECK(callback_fields.initialization_plan &&
              callback_fields.initialization_plan->materials.size() == 1,
          "reachable callback fixture produced no material recipe");
    if (callback_fields.initialization_plan &&
        callback_fields.initialization_plan->materials.size() == 1) {
      const MaterialRecipe &recipe = callback_fields.initialization_plan->materials[0];
      const MaterialSupportDecision support = classify_material_support(recipe);
      CHECK(recipe.disposition() == MaterialRecipeDisposition::host_reference &&
                support.disposition == MaterialRecipeDisposition::host_reference &&
                support.reason_bits == material_support_unowned_callback,
            "reachable callback did not select the exact PR5.3 host-reference route");
    }
    delete callback_oracle;
    geometric_object_destroy(callback_block);
  }
  destroy_absorber_list(absorbers);
  material_free(file_material);
  material_free(grid_material);
  material_free(user_material);
  material_free(dielectric);
  geometric_object_destroy(infinite_cylinder);
  geometric_object_destroy(infinite_prism);
  geometric_object_destroy(infinite_block);
  geometric_object_destroy(outer);
  geometric_object_destroy(inner);
  geometric_object_destroy(object);
  validate_material_ir(*static_cast<const MaterialIR *>(retained.get()));
}

static void test_material_ir_capture_atomicity() {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.5, 1.25, 8.0);
  material_type dielectric = make_dielectric(2.75);
  geometric_object object =
      make_block(dielectric, make_vector3(), make_vector3(1, 0, 0), make_vector3(0, 1, 0),
                 make_vector3(0, 0, 1), make_vector3(0.5, 0.5, 1.0));
  geometric_object_list geometry = {1, &object};
  for (int target = 0; target < count_processors(); ++target) {
    const int maximum_mode = count_processors() > 1 ? 3 : 2;
    for (int mode = 1; mode <= maximum_mode; ++mode) {
      structure s(gv, unit_epsilon, no_pml(), identity(), 2);
      s.set_conductivity(Dz, unit_epsilon);
      realnum *const address = s.chunks[0]->conductivity[Dz][Z];
      const realnum value = address ? address[0] : realnum(0);
      set_material_ir_capture_failure_for_testing(target, mode);
      bool failed = false;
      try {
        set_materials_from_geometry(&s, geometry, make_vector3(), true, 1e-5, 256, true,
                                    vacuum);
      }
      catch (const std::runtime_error &) { failed = true; }
      set_material_ir_capture_failure_for_testing(-1, 0);
      CHECK(and_to_all(failed) && !s.material_ir && s.chunks[0]->conductivity[Dz][Z] == address &&
                (!address || address[0] == value),
            "rank-asymmetric material IR capture failure mutated eager material state");
    }
  }
  geometric_object_destroy(object);
  material_free(dielectric);
}

static void test_geometry_backed_material_ir_removal() {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.5, 1.25, 8.0);
  material_type dielectric = make_dielectric(2.75);
  meep_geom::susceptibility lorentz = meep_geom::susceptibility();
  lorentz.frequency = 0.73;
  lorentz.gamma = 0.04;
  lorentz.sigma_diag = make_vector3(1.0, 0.5, 0.25);
  dielectric->medium.E_susceptibilities.push_back(lorentz);
  geometric_object object =
      make_block(dielectric, make_vector3(), make_vector3(1, 0, 0), make_vector3(0, 1, 0),
                 make_vector3(0, 0, 1), make_vector3(0.75, 0.5, 1.0));
  geometric_object_list geometry = {1, &object};
  structure s(gv, unit_epsilon, no_pml(), identity(), 2);
  set_materials_from_geometry(&s, geometry, make_vector3(), true, 1e-5, 256, true, vacuum);
  fields f(&s);
  f.require_component(Ez);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  f.advance(1);

  const std::shared_ptr<const void> retained_ir = f.material_ir;
  const MaterialIR *const retained = material_ir_for(f);
  CHECK(retained_ir && retained && !retained->susceptibilities.empty(),
        "geometry-backed removal fixture has no susceptibility IR");
  const uint64_t retained_signature = retained ? retained->signature : 0;
  const uint64_t retained_layout_signature = retained ? retained->layout_signature : 0;
  const BackendEpochSnapshot retained_epoch(f);

  counts.fail_rebuild = true;
  bool failed = false;
  try { f.remove_susceptibilities(); }
  catch (const std::runtime_error &) { failed = true; }
  const MaterialIR *const after_failure = material_ir_for(f);
  CHECK(and_to_all(failed) && f.material_ir == retained_ir && after_failure &&
            after_failure->signature == retained_signature &&
            after_failure->layout_signature == retained_layout_signature &&
            retained_epoch.matches(f),
        "failed geometry-backed removal changed IR ownership, signatures, or live state");

  counts.fail_rebuild = false;
  f.remove_susceptibilities();
  bool live_susceptibility = false;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    FOR_FIELD_TYPES(ft) {
      live_susceptibility = live_susceptibility || f.chunks[chunk]->s->chiP[ft] ||
                            f.chunks[chunk]->pol[ft];
    }
  }
  CHECK(and_to_all(!f.material_ir && !live_susceptibility),
        "successful geometry-backed removal retained IR or live susceptibility state");

  f.advance(1);
  bool stale_sigma = false;
  if (f.initialization_plan) {
    CHECK(f.initialization_plan->materials.size() == 1,
          "post-removal rebuild did not publish one material recipe");
    if (f.initialization_plan->materials.size() == 1) {
      const MaterialRecipe &recipe = f.initialization_plan->materials[0];
      stale_sigma = stale_sigma || bool(recipe.ir());
      for (const MaterialRecipeRow &row : recipe.rows())
        stale_sigma = stale_sigma || row.key.kind == int(array_kind::sigma);
      for (const MaterialRecipeRow &row : recipe.dense_fallback_rows())
        stale_sigma = stale_sigma || row.key.kind == int(array_kind::sigma);
      for (const MaterialIRTopologyRow &row : recipe.topology())
        stale_sigma = stale_sigma || row.key.kind == int(array_kind::sigma);
    }
  }
  else {
    stale_sigma = true;
  }
  if (f.storage_plan)
    for (const StorageKey &key : f.storage_plan->keys)
      stale_sigma = stale_sigma || key.kind == int(array_kind::sigma);
  if (f.descriptors) stale_sigma = stale_sigma || !f.descriptors->polarizations.empty();
  CHECK(and_to_all(!stale_sigma),
        "post-removal recipe or provisional topology retained susceptibility sigma definitions");

  geometric_object_destroy(object);
  material_free(dielectric);
}

static void test_material_recipe_and_provisional_storage() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->advance(1);

  const InitializationPlan initialization = build_initialization_plan(*f);
  CHECK(initialization.materials.size() == 1,
        "material recipe capture did not produce exactly one recipe");
  const MaterialRecipe &recipe = initialization.materials[0];
  validate_material_recipe(recipe);
  CHECK(recipe.disposition() == MaterialRecipeDisposition::host_reference &&
            !recipe.from_host_callback() && recipe.host_callback_id() == invalid_array_value,
        "eager CPU material capture has the wrong fallback disposition");

  size_t local_material_rows = 0;
  for (const ArraySpec &spec : f->storage_plan->arrays)
    if (spec.role == array_role::material) ++local_material_rows;
  CHECK(recipe.rows().size() == local_material_rows,
        "material recipe captured %zu rows for %zu material arrays", recipe.rows().size(),
        local_material_rows);
  CHECK(or_to_all(local_material_rows > 0), "material recipe fixture has no material rows");

  for (const MaterialRecipeRow &row : recipe.rows()) {
    const ArrayId id = f->array_catalog->find(row.key);
    CHECK(is_valid(id), "material recipe row does not resolve in the CPU catalog");
    if (is_valid(id)) {
      const void *live = f->array_catalog->resolve_untyped(id);
      CHECK(live && row.values.size() == row.elements * sizeof(realnum) &&
                !memcmp(live, row.values.data(), row.values.size()),
            "material recipe did not exactly own a CPU material row");
    }
  }

  if (!recipe.rows().empty() && !recipe.rows()[0].values.empty()) {
    const MaterialRecipeRow &row = recipe.rows()[0];
    const ArrayId id = f->array_catalog->find(row.key);
    CHECK(is_valid(id), "material recipe row does not resolve in the CPU catalog");
    unsigned char *live = static_cast<unsigned char *>(f->array_catalog->resolve_untyped(id));
    const unsigned char frozen = row.values[0];
    const uint64_t signature = recipe.signature();
    live[0] ^= 0x5a;
    CHECK(row.values[0] == frozen && recipe.signature() == signature,
          "material recipe borrowed mutable CPU coefficient storage");
    live[0] ^= 0x5a;
  }

  const MaterialRecipe copied(recipe);
  CHECK(copied == recipe && copied.signature() == recipe.signature(),
        "material recipe copy changed identity");
  MaterialRecipeInput detached_input = material_recipe_input(recipe);
  const MaterialRecipe detached(detached_input);
  if (!detached_input.rows.empty() && !detached_input.rows[0].values.empty()) {
    const unsigned char frozen = detached.rows()[0].values[0];
    detached_input.rows[0].values[0] ^= 0xff;
    detached_input.description.clear();
    CHECK(detached.rows()[0].values[0] == frozen && !detached.description().empty(),
          "material recipe borrowed its constructor input");
  }

  MaterialRecipeInput changed = material_recipe_input(recipe);
  changed.description += ":changed";
  CHECK(MaterialRecipe(changed).signature() == recipe.signature() &&
            MaterialRecipe(changed) == recipe,
        "diagnostic material description changed semantic identity");
  changed = material_recipe_input(recipe);
  changed.eps_averaging = !changed.eps_averaging;
  if (recipe.ir())
    expect_material_recipe_rejected(changed, "material recipe policy/IR averaging mismatch");
  else
    CHECK(MaterialRecipe(changed).signature() != recipe.signature(),
          "material recipe signature ignores averaging policy");
  changed = material_recipe_input(recipe);
  changed.subpixel_tol *= 2.0;
  if (recipe.ir())
    expect_material_recipe_rejected(changed, "material recipe policy/IR tolerance mismatch");
  else
    CHECK(MaterialRecipe(changed).signature() != recipe.signature(),
          "material recipe signature ignores subpixel tolerance");
  changed = material_recipe_input(recipe);
  ++changed.subpixel_maxeval;
  if (recipe.ir())
    expect_material_recipe_rejected(changed,
                                    "material recipe policy/IR evaluation-limit mismatch");
  else
    CHECK(MaterialRecipe(changed).signature() != recipe.signature(),
          "material recipe signature ignores subpixel evaluation limit");
  changed = material_recipe_input(recipe);
  changed.disposition = MaterialRecipeDisposition::device_native;
  expect_material_recipe_rejected(changed, "unimplemented device-native material recipe");
  changed = material_recipe_input(recipe);
  changed.disposition = MaterialRecipeDisposition::tiled_callback;
  changed.from_host_callback = true;
  changed.host_callback_id = 7;
  expect_material_recipe_rejected(changed, "unimplemented tiled material recipe");
  changed = material_recipe_input(recipe);
  changed.disposition = MaterialRecipeDisposition::hybrid_interface;
  expect_material_recipe_rejected(changed, "unimplemented hybrid material recipe");
  if (!recipe.rows().empty() && !recipe.rows()[0].values.empty()) {
    changed = material_recipe_input(recipe);
    changed.rows[0].values[0] ^= 1;
    CHECK(MaterialRecipe(changed).signature() != recipe.signature(),
          "material recipe signature ignores owned coefficient bytes");
    changed = material_recipe_input(recipe);
    changed.rows[0].key.chunk += f->num_chunks + 1;
    if (recipe.ir())
      expect_material_recipe_rejected(changed, "material recipe row absent from IR topology");
    else
      CHECK(MaterialRecipe(changed).signature() != recipe.signature(),
            "material recipe signature ignores storage identity");
    changed = material_recipe_input(recipe);
    changed.rows[0].storage = changed.rows[0].storage == Precision::f32 ? Precision::f64
                                                                       : Precision::f32;
    expect_material_recipe_rejected(changed, "non-native material row storage");
    changed = material_recipe_input(recipe);
    changed.rows[0].alignment *= 2;
    expect_material_recipe_rejected(changed, "noncanonical material row alignment");
    changed = material_recipe_input(recipe);
    if (changed.rows[0].elements > 1) {
      --changed.rows[0].elements;
      changed.rows[0].values.resize(changed.rows[0].elements *
                                    host_element_bytes(changed.rows[0].element_type));
    }
    else {
      ++changed.rows[0].elements;
      changed.rows[0].values.resize(changed.rows[0].elements *
                                    host_element_bytes(changed.rows[0].element_type));
    }
    if (recipe.ir())
      expect_material_recipe_rejected(changed,
                                      "material recipe row extent differs from IR topology");
    else
      CHECK(MaterialRecipe(changed).signature() != recipe.signature(),
            "material recipe signature ignores row extent");
  }
  if (recipe.rows().size() > 1) {
    changed = material_recipe_input(recipe);
    std::swap(changed.rows[0], changed.rows[1]);
    CHECK(MaterialRecipe(changed).signature() != recipe.signature(),
          "material recipe signature ignores row order");
  }

  changed = material_recipe_input(recipe);
  changed.description.clear();
  expect_material_recipe_rejected(changed, "empty material recipe description");
  changed = material_recipe_input(recipe);
  changed.subpixel_tol = std::numeric_limits<double>::quiet_NaN();
  expect_material_recipe_rejected(changed, "nonfinite material recipe tolerance");
  changed = material_recipe_input(recipe);
  changed.subpixel_maxeval = 0;
  expect_material_recipe_rejected(changed, "zero material recipe evaluation limit");
  changed = material_recipe_input(recipe);
  changed.host_callback_id = 7;
  expect_material_recipe_rejected(changed, "callback identity on host-reference recipe");
  if (!recipe.rows().empty()) {
    changed = material_recipe_input(recipe);
    changed.rows.push_back(changed.rows[0]);
    expect_material_recipe_rejected(changed, "duplicate material recipe row");
    changed = material_recipe_input(recipe);
    changed.rows[0].role = array_role::field;
    expect_material_recipe_rejected(changed, "non-material recipe row");
    changed = material_recipe_input(recipe);
    changed.rows[0].alignment = 3;
    expect_material_recipe_rejected(changed, "non-power-of-two material row alignment");
    changed = material_recipe_input(recipe);
    changed.rows[0].key.kind = int(array_kind::f);
    expect_material_recipe_rejected(changed, "non-material recipe storage kind");
    changed = material_recipe_input(recipe);
    changed.rows[0].key.chunk = -1;
    expect_material_recipe_rejected(changed, "negative material recipe chunk");
    changed = material_recipe_input(recipe);
    changed.rows[0].key.component_ = NUM_FIELD_COMPONENTS;
    expect_material_recipe_rejected(changed, "out-of-range material recipe component");
    changed = material_recipe_input(recipe);
    changed.rows[0].key.cmp = 6;
    expect_material_recipe_rejected(changed, "out-of-range material recipe complex part");
    changed = material_recipe_input(recipe);
    changed.rows[0].element_type = static_cast<ElementType>(999);
    expect_material_recipe_rejected(changed, "invalid material recipe element type");
    changed = material_recipe_input(recipe);
    changed.rows[0].elements = 0;
    changed.rows[0].values.clear();
    expect_material_recipe_rejected(changed, "zero-extent material recipe row");
    changed = material_recipe_input(recipe);
    changed.rows[0].values.pop_back();
    expect_material_recipe_rejected(changed, "short material recipe row payload");
    changed = material_recipe_input(recipe);
    changed.rows[0].elements = std::numeric_limits<size_t>::max();
    expect_material_recipe_rejected(changed, "overflowing material recipe row extent");
  }
  if (!recipe.topology().empty()) {
    changed = material_recipe_input(recipe);
    changed.topology.pop_back();
    expect_material_recipe_rejected(changed, "missing provisional IR topology row");
    changed = material_recipe_input(recipe);
    MaterialIRTopologyRow extra = changed.topology.back();
    extra.key.chunk += f->num_chunks + 1;
    changed.topology.push_back(extra);
    expect_material_recipe_rejected(changed, "extra provisional IR topology row");
    changed = material_recipe_input(recipe);
    ++changed.topology[0].elements;
    expect_material_recipe_rejected(changed, "changed provisional IR topology metadata");
  }

  StoragePlan provisional = *f->storage_plan;
  mark_material_storage_provisional(recipe, provisional);
  size_t expected_peak = 0, expected_steady = 0, expected_suffix = 0;
  for (const MaterialIRTopologyRow &row : recipe.topology())
    if (!is_valid(f->array_catalog->find(row.key))) ++expected_suffix;
  for (size_t i = 0; i < provisional.arrays.size(); ++i)
    if (!is_valid(provisional.arrays[i].alias_of)) {
      const size_t bytes = storage_bytes(provisional.arrays[i]);
      CHECK(bytes <= std::numeric_limits<size_t>::max() - expected_peak,
            "provisional material peak-byte oracle overflowed");
      expected_peak += bytes;
      if (!provisional.arrays[i].classification_provisional) {
        CHECK(bytes <= std::numeric_limits<size_t>::max() - expected_steady,
              "provisional material steady-byte oracle overflowed");
        expected_steady += bytes;
      }
    }
  for (size_t i = 0; i < provisional.arrays.size(); ++i) {
    CHECK(provisional.arrays[i].classification_provisional ==
              (provisional.arrays[i].role == array_role::material),
          "provisional marking changed the wrong storage role at %zu", i);
  }
  CHECK(provisional.arrays.size() == f->storage_plan->arrays.size() + expected_suffix &&
            provisional.provisional_peak_bytes() == expected_peak &&
            provisional.steady_state_bytes() == expected_steady && expected_peak >= expected_steady,
        "provisional suffix or peak/steady byte accounting differs from the exact oracle");

  if (recipe.rows().size() > 1) {
    changed = material_recipe_input(recipe);
    std::swap(changed.rows[0], changed.rows[1]);
    const MaterialRecipe reordered(changed);
    StoragePlan unchanged = *f->storage_plan;
    bool rejected = false;
    try { mark_material_storage_provisional(reordered, unchanged); }
    catch (const std::invalid_argument &) { rejected = true; }
    CHECK(rejected, "noncanonical material recipe row order was accepted");
    CHECK(!has_provisional_material_storage(unchanged),
          "failed reordered material recipe partially marked storage");
  }
  if (!recipe.rows().empty()) {
    changed = material_recipe_input(recipe);
    changed.rows.pop_back();
    if (recipe.ir())
      expect_material_recipe_rejected(changed, "missing material recipe row");
    else {
      const MaterialRecipe missing(changed);
      StoragePlan unchanged = *f->storage_plan;
      bool rejected = false;
      try { mark_material_storage_provisional(missing, unchanged); }
      catch (const std::invalid_argument &) { rejected = true; }
      CHECK(rejected && !has_provisional_material_storage(unchanged),
            "missing material recipe row was accepted or partially published");
    }

    changed = material_recipe_input(recipe);
    MaterialRecipeRow extra = changed.rows.back();
    extra.key.chunk += f->num_chunks + 1;
    changed.rows.push_back(extra);
    if (recipe.ir())
      expect_material_recipe_rejected(changed, "extra material recipe row");
    else {
      const MaterialRecipe oversized(changed);
      StoragePlan unchanged = *f->storage_plan;
      bool rejected = false;
      try { mark_material_storage_provisional(oversized, unchanged); }
      catch (const std::invalid_argument &) { rejected = true; }
      CHECK(rejected && !has_provisional_material_storage(unchanged),
            "extra material recipe row was accepted or partially published");
    }
  }

  const StoragePlan fully_provisional = provisional;
  MaterialClassification classification;
  classification.provisional_row_state.assign(
      provisional.arrays.size(), MaterialClassification::not_provisional);
  for (size_t i = 0; i < f->storage_plan->arrays.size(); ++i)
    if (provisional.arrays[i].role == array_role::material)
      classification.provisional_row_state[i] = MaterialClassification::retained;
  for (size_t i = f->storage_plan->arrays.size(); i < provisional.arrays.size(); ++i) {
    classification.elided.push_back(ArrayId{uint32_t(i)});
    classification.provisional_row_state[i] = MaterialClassification::elided_row;
  }
  resolve_material_storage(recipe, classification, *f->storage_plan, provisional);
  CHECK(!has_provisional_material_storage(provisional) &&
            provisional.arrays.size() == fully_provisional.arrays.size() &&
            provisional.provisional_peak_bytes() ==
                fully_provisional.provisional_peak_bytes() &&
            provisional.physical_resident_bytes() ==
                fully_provisional.provisional_peak_bytes() &&
            provisional.steady_state_bytes() == f->storage_plan->steady_state_bytes(),
        "material classification did not resolve provisional rows");
  resolve_material_storage(recipe, classification, *f->storage_plan, provisional);
  CHECK(!has_provisional_material_storage(provisional),
        "material classification resolution is not idempotent");

  if (fully_provisional.arrays.size() > f->storage_plan->arrays.size()) {
    StoragePlan all_retained = fully_provisional;
    MaterialClassification retain_all;
    retain_all.provisional_row_state.assign(
        all_retained.arrays.size(), MaterialClassification::not_provisional);
    for (size_t i = 0; i < all_retained.arrays.size(); ++i)
      if (all_retained.arrays[i].role == array_role::material)
        retain_all.provisional_row_state[i] = MaterialClassification::retained;
    resolve_material_storage(recipe, retain_all, *f->storage_plan, all_retained);
    CHECK(all_retained.arrays.size() == fully_provisional.arrays.size() &&
              !has_provisional_material_storage(all_retained) &&
              all_retained.steady_state_bytes() == all_retained.physical_resident_bytes(),
          "all-retained material classification changed stable IDs or accounting");

    StoragePlan retained = fully_provisional;
    MaterialClassification selective;
    selective.provisional_row_state.assign(
        retained.arrays.size(), MaterialClassification::not_provisional);
    for (size_t i = 0; i < f->storage_plan->arrays.size(); ++i)
      if (retained.arrays[i].role == array_role::material)
        selective.provisional_row_state[i] = MaterialClassification::retained;
    size_t retained_suffix = 0;
    for (size_t i = f->storage_plan->arrays.size(); i < retained.arrays.size(); ++i) {
      if ((i - f->storage_plan->arrays.size()) % 2) {
        selective.elided.push_back(ArrayId{uint32_t(i)});
        selective.provisional_row_state[i] = MaterialClassification::elided_row;
      }
      else {
        selective.provisional_row_state[i] = MaterialClassification::retained;
        ++retained_suffix;
      }
    }
    resolve_material_storage(recipe, selective, *f->storage_plan, retained);
    CHECK(retained.arrays.size() == fully_provisional.arrays.size() &&
              retained_suffix > 0 && !has_provisional_material_storage(retained) &&
              retained.steady_state_bytes() > f->storage_plan->steady_state_bytes(),
          "material classification did not preserve alternating stable-ID suffix rows");
    for (size_t i = f->storage_plan->arrays.size(); i < retained.arrays.size(); ++i) {
      const bool expected_elided = (i - f->storage_plan->arrays.size()) % 2;
      CHECK(retained.arrays[i].id.value == i &&
                retained.arrays[i].classification_elided == expected_elided,
            "material resolution renumbered or misclassified stable ArrayId %zu", i);
    }
  }

  StoragePlan mixed = *f->storage_plan;
  mark_material_storage_provisional(recipe, mixed);
  apply_precision_policy(mixed, precision_mixed());
  const size_t mixed_peak_bytes = mixed.provisional_peak_bytes();
  StoragePlan mixed_authoritative = *f->storage_plan;
  apply_precision_policy(mixed_authoritative, precision_mixed());
  resolve_material_storage(recipe, classification, *f->storage_plan, mixed,
                           precision_mixed());
  CHECK(mixed.arrays.size() == fully_provisional.arrays.size() &&
            !has_provisional_material_storage(mixed) &&
            mixed.physical_resident_bytes() == mixed_peak_bytes &&
            mixed.steady_state_bytes() == mixed_authoritative.steady_state_bytes(),
        "mixed material storage did not preserve stable tombstones and byte accounting "
        "(arrays=%zu expected=%zu peak=%zu expected_peak=%zu steady=%zu expected_steady=%zu)",
        mixed.arrays.size(), fully_provisional.arrays.size(), mixed.physical_resident_bytes(),
        mixed_peak_bytes, mixed.steady_state_bytes(), mixed_authoritative.steady_state_bytes());
  for (size_t i = f->storage_plan->arrays.size(); i < mixed.arrays.size(); ++i)
    CHECK(mixed.arrays[i].classification_elided,
          "mixed material storage exposed an elided suffix key at %zu", i);

  if (!recipe.rows().empty()) {
    StoragePlan malformed = *f->storage_plan;
    for (size_t i = 0; i < malformed.arrays.size(); ++i)
      if (malformed.arrays[i].role == array_role::material) {
        ++malformed.arrays[i].elements;
        break;
      }
    const StoragePlan entry = malformed;
    bool rejected = false;
    try {
      mark_material_storage_provisional(recipe, malformed);
    }
    catch (const std::invalid_argument &) {
      rejected = true;
    }
    CHECK(rejected, "mismatched material storage extent was accepted");
    for (size_t i = 0; i < malformed.arrays.size(); ++i)
      CHECK(malformed.arrays[i].classification_provisional ==
                entry.arrays[i].classification_provisional,
            "failed provisional marking partially changed storage at %zu", i);

    provisional = *f->storage_plan;
    mark_material_storage_provisional(recipe, provisional);
    classification.elided.push_back(f->array_catalog->find(recipe.rows()[0].key));
    const StoragePlan before_elision = provisional;
    rejected = false;
    try {
      resolve_material_storage(recipe, classification, *f->storage_plan, provisional);
    }
    catch (const std::invalid_argument &) {
      rejected = true;
    }
    CHECK(rejected, "host-reference material recipe accepted device row elision");
    for (size_t i = 0; i < provisional.arrays.size(); ++i)
      CHECK(provisional.arrays[i].classification_provisional ==
                before_elision.arrays[i].classification_provisional,
            "failed material resolution partially changed storage at %zu", i);

    classification.elided.clear();
    provisional = *f->storage_plan;
    mark_material_storage_provisional(recipe, provisional);
    for (size_t i = 0; i < provisional.arrays.size(); ++i)
      if (provisional.arrays[i].role == array_role::material) {
        provisional.arrays[i].storage =
            provisional.arrays[i].storage == Precision::f32 ? Precision::f64 : Precision::f32;
        break;
      }
    const StoragePlan before_precision = provisional;
    rejected = false;
    try {
      resolve_material_storage(recipe, classification, *f->storage_plan, provisional);
    }
    catch (const std::invalid_argument &) {
      rejected = true;
    }
    CHECK(rejected, "material resolution accepted a changed storage precision");
    for (size_t i = 0; i < provisional.arrays.size(); ++i)
      CHECK(provisional.arrays[i].storage == before_precision.arrays[i].storage &&
                provisional.arrays[i].classification_provisional ==
                    before_precision.arrays[i].classification_provisional,
            "failed precision resolution partially changed storage at %zu", i);
  }

  delete f;
  delete s;
}

static void test_owned_tiled_material_route() {
  using namespace meep_geom;
  const grid_volume gv = vol2d(2.0, 2.0, 16.0);
  std::shared_ptr<size_t> calls(new size_t(0));
  std::shared_ptr<const OwnedMaterialCallback> owner(
      new OwnedMaterialCallback(
          UINT64_C(0x74696c65642d6964), UINT64_C(0x74696c65642d7631),
          owned_material_callback_tiled_capabilities,
          [calls](vector3 point, medium_struct &medium) {
            ++*calls;
            medium = medium_struct(2.0 + 0.125 * point.x + 0.0625 * point.y);
          }));
  std::shared_ptr<const OwnedMaterialCallback> equivalent_owner(
      new OwnedMaterialCallback(
          owner->id, owner->signature, owner->capabilities,
          [](vector3, medium_struct &medium) { medium = medium_struct(99.0); }));
  std::shared_ptr<const OwnedMaterialCallback> changed_owner(
      new OwnedMaterialCallback(
          owner->id, owner->signature + 1, owner->capabilities,
          [](vector3, medium_struct &medium) { medium = medium_struct(99.0); }));
  std::shared_ptr<const OwnedMaterialCallback> changed_capabilities(
      new OwnedMaterialCallback(
          owner->id, owner->signature,
          owned_material_callback_pure_replay_stable,
          [](vector3, medium_struct &medium) { medium = medium_struct(99.0); }));
  material_type equivalent = make_owned_user_material_for_backend(equivalent_owner, false);
  material_type changed = make_owned_user_material_for_backend(changed_owner, false);
  material_type changed_contract =
      make_owned_user_material_for_backend(changed_capabilities, false);
  std::weak_ptr<const OwnedMaterialCallback> lifetime(owner);
  material_type material = make_owned_user_material_for_backend(owner, false);
  CHECK(material_type_equal(material, equivalent) && !material_type_equal(material, changed) &&
            !material_type_equal(material, changed_contract),
        "owned callback equality used owner identity instead of stable behavior signature");
  material_free(equivalent);
  material_free(changed);
  material_free(changed_contract);
  geometric_object_list geometry = {0, NULL};
  structure s(gv, unit_epsilon, no_pml(), identity(), 1);
  set_materials_from_geometry(&s, geometry, make_vector3(), false, 1e-5, 64, false,
                              material);
  material_free(material);
  owner.reset();
  const bool callback_evaluated = or_to_all(*calls > 0);
  CHECK(!lifetime.expired() && callback_evaluated,
        "owned material callback token did not survive eager construction");

  fields f(&s);
  f.use_real_fields();
  f.require_component(Ez);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts);
  f.advance(1);
  CHECK(f.initialization_plan && f.initialization_plan->materials.size() == 1,
        "owned callback produced no material recipe");
  if (f.initialization_plan && f.initialization_plan->materials.size() == 1) {
    const MaterialRecipe &recipe = f.initialization_plan->materials[0];
    fields oracle(&s);
    oracle.use_real_fields();
    oracle.require_component(Ez);
    oracle.advance(1);
    const InitializationPlan oracle_initialization = build_initialization_plan(oracle);
    CHECK(oracle_initialization.materials.size() == 1,
          "owned callback CPU oracle produced no unselected material recipe");
    const MaterialRecipe unselected_recipe =
        oracle_initialization.materials.empty()
            ? recipe
            : oracle_initialization.materials[0];
    const MaterialSupportDecision support = classify_material_support(recipe);
    uint64_t points = 0;
    uint64_t destination_points = 0;
    bool dense_payloads_elided = true;
    for (const MaterialCallbackTile &tile : recipe.callback_tiles()) points += tile.count;
    for (const MaterialIRDestination &destination : recipe.ir()->destinations)
      destination_points += destination.point_count;
    const size_t global_points = sum_to_all(size_t(points));
    CHECK(f.backend_state &&
              f.backend_state->material_fallback_local_presence ==
                  !recipe.callback_tiles().empty(),
          "owned callback local fallback presence did not match local tile work");
    for (const MaterialRecipeRow &row : recipe.rows())
      dense_payloads_elided = dense_payloads_elided && row.values.empty();
    std::set<StorageKey, bool (*)(const StorageKey &, const StorageKey &)> primary_keys(
        [](const StorageKey &a, const StorageKey &b) {
          if (a.chunk != b.chunk) return a.chunk < b.chunk;
          if (a.kind != b.kind) return a.kind < b.kind;
          if (a.component_ != b.component_) return a.component_ < b.component_;
          if (a.cmp != b.cmp) return a.cmp < b.cmp;
          return a.aux < b.aux;
        });
    for (const MaterialRecipeRow &row : unselected_recipe.rows()) primary_keys.insert(row.key);
    bool saw_synthesized_identity = false;
    for (const MaterialRecipeRow &row : unselected_recipe.dense_fallback_rows()) {
      if (primary_keys.count(row.key) || row.key.kind != int(array_kind::chi1inv) ||
          row.key.component_ < 0 ||
          int(row.key.aux) != int(component_direction(component(row.key.component_))))
        continue;
      saw_synthesized_identity = true;
      CHECK(row.values.size() == row.elements * sizeof(realnum),
            "synthesized chi1 identity row has the wrong byte count");
      for (size_t point = 0; point < row.elements; ++point) {
        realnum value = 0;
        memcpy(&value, row.values.data() + point * sizeof(realnum), sizeof(value));
        CHECK(value == realnum(1),
              "dense fallback synthesized chi1 diagonal as %.17g instead of +1",
              double(value));
      }
    }
    const bool globally_saw_synthesized_identity = or_to_all(saw_synthesized_identity);
    const MaterialRecipe coerced =
        select_material_recipe_route(unselected_recipe,
                                     MaterialRecipeDisposition::host_reference);
    CHECK(coerced.dense_fallback_rows() == unselected_recipe.dense_fallback_rows() &&
              material_recipe_has_local_fallback_work(
                  coerced, MaterialRecipeDisposition::host_reference) ==
                  !coerced.dense_fallback_rows().empty(),
          "host coercion changed or misclassified its complete dense snapshot");
    const MaterialRecipeDisposition expected_local_callback_route =
        points ? MaterialRecipeDisposition::tiled_callback
               : MaterialRecipeDisposition::device_native;
    CHECK(recipe.disposition() == expected_local_callback_route &&
              support.disposition == expected_local_callback_route &&
              (support.reason_bits & material_support_owned_callback) &&
              !(support.reason_bits & material_support_unowned_callback) &&
              points == destination_points && points == support.callback_points &&
              (recipe.callback_tiles().empty() == (points == 0)) && global_points > 256 &&
              dense_payloads_elided && globally_saw_synthesized_identity &&
              !lifetime.expired(),
          "owned pointwise callback did not select a nonvacuous bounded tiled route");

    MaterialRecipeInput malformed = material_recipe_input(recipe);
    malformed.callback_owners.clear();
    expect_material_recipe_rejected(malformed, "missing owned callback token");
    malformed = material_recipe_input(recipe);
    malformed.callback_owners.push_back(std::shared_ptr<const OwnedMaterialCallback>(
        new OwnedMaterialCallback(UINT64_C(0x65787472612d6f77),
                                  UINT64_C(0x65787472612d7369),
                                  owned_material_callback_tiled_capabilities,
                                  [](vector3, medium_struct &medium) {
                                    medium = medium_struct(4.0);
                                  })));
    expect_material_recipe_rejected(malformed, "unreferenced owned callback token");
    malformed = material_recipe_input(recipe);
    malformed.callback_owners.push_back(malformed.callback_owners.front());
    expect_material_recipe_rejected(malformed, "duplicate owned callback token");
    if (!recipe.callback_tiles().empty()) {
      malformed = material_recipe_input(recipe);
      malformed.callback_tiles[0].material = uint32_t(recipe.ir()->materials.size());
      expect_material_recipe_rejected(malformed, "wrong callback material");
      malformed = material_recipe_input(recipe);
      malformed.callback_tiles[0].first_point = 1;
      expect_material_recipe_rejected(malformed, "callback tile gap");
      malformed = material_recipe_input(recipe);
      malformed.callback_tiles.insert(malformed.callback_tiles.begin(),
                                      malformed.callback_tiles.front());
      expect_material_recipe_rejected(malformed, "duplicate callback tile");
      if (recipe.callback_tiles().size() > 1) {
        malformed = material_recipe_input(recipe);
        std::swap(malformed.callback_tiles[0], malformed.callback_tiles[1]);
        expect_material_recipe_rejected(malformed, "reordered callback tiles");
      }
      malformed = material_recipe_input(recipe);
      --malformed.callback_tiles.back().count;
      expect_material_recipe_rejected(malformed, "short final callback tile");
    }
    malformed = material_recipe_input(recipe);
    malformed.callback_owners[0].reset(new OwnedMaterialCallback(
        recipe.ir()->materials[recipe.ir()->default_material].callback_id,
        recipe.ir()->materials[recipe.ir()->default_material].callback_signature + 1,
        owned_material_callback_tiled_capabilities,
        [](vector3, medium_struct &medium) { medium = medium_struct(3.0); }));
    expect_material_recipe_rejected(malformed, "callback signature mutation");
    malformed = material_recipe_input(recipe);
    malformed.callback_owners[0].reset(new OwnedMaterialCallback(
        recipe.ir()->materials[recipe.ir()->default_material].callback_id,
        recipe.ir()->materials[recipe.ir()->default_material].callback_signature,
        owned_material_callback_pure_replay_stable,
        [](vector3, medium_struct &medium) { medium = medium_struct(3.0); }));
    expect_material_recipe_rejected(malformed, "callback capability mutation");

    MaterialIR effective = *recipe.ir();
    effective.eps_averaging = false;
    effective.subpixel_maxeval = 0;
    effective.materials[effective.default_material].do_averaging = true;
    refresh_material_ir_signatures_for_testing(effective);
    CHECK(classify_material_ir_support(std::shared_ptr<const MaterialIR>(new MaterialIR(effective)))
              .disposition == expected_local_callback_route,
          "material-local averaging enabled a tiled callback while global averaging was off");
    effective.eps_averaging = true;
    effective.subpixel_maxeval = 64;
    effective.materials[effective.default_material].do_averaging = false;
    refresh_material_ir_signatures_for_testing(effective);
    CHECK(classify_material_ir_support(std::shared_ptr<const MaterialIR>(new MaterialIR(effective)))
              .disposition == expected_local_callback_route,
          "global averaging enabled a tiled callback while material averaging was off");
    effective.materials[effective.default_material].do_averaging = true;
    refresh_material_ir_signatures_for_testing(effective);
    CHECK(classify_material_ir_support(std::shared_ptr<const MaterialIR>(new MaterialIR(effective)))
              .disposition == MaterialRecipeDisposition::host_reference,
          "effective callback averaging did not select host-reference fallback");

    if (count_processors() > 1) {
      const std::shared_ptr<const void> original_ir = f.material_ir;
      BackendState *const live_state = f.backend_state;
      const int rebuilds = counts.rebuilds;
      if (my_rank() == 0) {
        std::shared_ptr<MaterialIR> changed_ir(new MaterialIR(*recipe.ir()));
        changed_ir->subpixel_tol *= 2.0;
        refresh_material_ir_signatures_for_testing(*changed_ir);
        f.material_ir = std::static_pointer_cast<const void>(changed_ir);
      }
      invalidate(f, MutationKind::material_values,
                 "backend_api:rank_asymmetric_material_signature");
      bool failed = false;
      try { f.advance(1); }
      catch (const std::runtime_error &) { failed = true; }
      CHECK(and_to_all(failed) && f.backend_state == live_state && counts.rebuilds == rebuilds,
            "rank-asymmetric semantic material signature reached migration or publication");
      f.material_ir = original_ir;
    }
  }
}

static void test_material_fallback_policy_transaction() {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.0, 1.0, 8.0);
  std::shared_ptr<const OwnedMaterialCallback> owner(new OwnedMaterialCallback(
      UINT64_C(0x706f6c6963792d69), UINT64_C(0x706f6c6963792d73),
      owned_material_callback_tiled_capabilities,
      [](vector3 point, medium_struct &medium) {
        medium = medium_struct(2.0 + 0.01 * point.x);
      }));
  material_type material = make_owned_user_material_for_backend(owner, false);
  geometric_object_list geometry = {0, NULL};
  structure s(gv, unit_epsilon, no_pml(), identity(), 1);
  set_materials_from_geometry(&s, geometry, make_vector3(), false, 1e-5, 64, false,
                              material);
  material_free(material);
  fields f(&s);
  f.require_component(Ez);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts, false, false, false, false, true);
  f.options.fallback = fallback_policy::warn;
  f.options.strict = false;

  counts.fail_compile = true;
  bool failed = false;
  try { f.advance(1); }
  catch (const std::exception &) { failed = true; }
  CHECK(failed && !f.backend_state && !f.executable &&
            f.backend->material_fallback_warning_count() == 0,
        "failed material candidate published warning/state before commit");
  counts.fail_compile = false;
  f.advance(1);
  CHECK(f.backend_state && f.executable &&
            f.backend->material_fallback_warning_count() == 1,
        "successful material fallback retry did not publish exactly one warning");
  const bool any_callback_points = or_to_all(
      f.backend_state && f.backend_state->material_fallback_statistics.callback_points > 0);
  CHECK(f.backend_state &&
            f.backend_state->material_route == MaterialRecipeDisposition::tiled_callback &&
            f.backend_state->material_fallback_statistics.dense_rows == 0 &&
            f.backend_state->material_fallback_statistics.dense_bytes == 0 &&
            any_callback_points,
        "tiled material route published overlapping dense/native ownership accounting");

  BackendState *committed_state = f.backend_state;
  const int rebuilds = counts.rebuilds;
  f.options.strict = true;
  invalidate(f, MutationKind::material_values, "backend_api:strict_material_retry");
  failed = false;
  try { f.advance(1); }
  catch (const std::exception &) { failed = true; }
  CHECK(failed && f.backend_state == committed_state && counts.rebuilds == rebuilds,
        "strict warm material rejection ran migration or replaced the live epoch");

  f.options.strict = false;
  backend_set_material_fallback_warning_for_testing(
      *f.backend, std::numeric_limits<uint64_t>::max(), false);
  failed = false;
  try { f.advance(1); }
  catch (const std::exception &) { failed = true; }
  CHECK(failed && f.backend_state == committed_state && counts.rebuilds == rebuilds,
        "warning overflow ran migration or replaced the live material epoch");
  backend_set_material_fallback_warning_for_testing(*f.backend, 1, true);
  f.advance(1);
  CHECK(f.backend_state != committed_state,
        "material fallback did not recover after warning-overflow retry");
}

static void replace_isotropic_material_definition_for_testing(
    fields &f, const std::shared_ptr<const void> &material_ir,
    double (*epsilon)(const vec &), const char *site) {
  backend_prepare_field_layout_change(
      f, invalidation_closure(MutationKind::material_definition), site);
  simple_material_function material(epsilon);
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    FOR_COMPONENTS(c)
      if (f.chunks[chunk]->gv.has_field(c) && (is_electric(c) || is_magnetic(c)))
        f.chunks[chunk]->s->set_chi1inv(c, material, false, 1e-5, 0);
    f.chunks[chunk]->s->remove_susceptibilities();
  }
  f.material_ir = material_ir;
  invalidate(f, MutationKind::material_definition, site);
  note_connections_invalidated(f);
  mark_local_invalidation(f);
  backend_note_material_definition_changed_for_testing(f);
}

static void test_material_fallback_route_removal() {
  using namespace meep_geom;
  const grid_volume gv = vol2d(1.0, 1.0, 8.0);
  geometric_object_list empty_geometry = {0, NULL};
  std::shared_ptr<const OwnedMaterialCallback> owner(new OwnedMaterialCallback(
      UINT64_C(0x72656d6f76652d69), UINT64_C(0x72656d6f76652d73),
      owned_material_callback_tiled_capabilities,
      [](vector3 point, medium_struct &medium) {
        medium = medium_struct(2.5 + 0.01 * point.x);
      }));
  material_type fallback_material = make_owned_user_material_for_backend(owner, false);
  structure fallback_definition(gv, unit_epsilon, no_pml(), identity(), 1);
  set_materials_from_geometry(&fallback_definition, empty_geometry, make_vector3(), false,
                              1e-5, 64, false, fallback_material);
  material_free(fallback_material);

  material_type native_material = make_dielectric(2.0);
  structure native_definition(gv, unit_epsilon, no_pml(), identity(), 1);
  set_materials_from_geometry(&native_definition, empty_geometry, make_vector3(), false,
                              1e-5, 64, false, native_material);
  material_free(native_material);
  structure vacuum_definition(gv, unit_epsilon, no_pml(), identity(), 1);
  set_materials_from_geometry(&vacuum_definition, empty_geometry, make_vector3(), false,
                              1e-5, 64, false, vacuum);

  fields f(&fallback_definition);
  f.require_component(Ez);
  lifetime_counts counts;
  f.backend = new tracking_backend(f, counts, false, false, false, false, true);
  f.options.strict = false;
  f.options.fallback = fallback_policy::warn;
  f.advance(1);
  const bool initial_local_callback_work =
      f.initialization_plan && f.initialization_plan->materials.size() == 1 &&
      !f.initialization_plan->materials[0].callback_tiles().empty();
  CHECK(f.backend_state &&
            f.backend_state->material_route == MaterialRecipeDisposition::tiled_callback &&
            f.backend_state->material_fallback_global_presence &&
            f.backend_state->material_fallback_local_presence ==
                initial_local_callback_work &&
            f.backend->material_fallback_warning_count() == 1,
        "fallback removal fixture did not publish its initial tiled route");

  replace_isotropic_material_definition_for_testing(
      f, native_definition.material_ir, two_epsilon,
      "backend_api:fallback_to_native_definition");
  f.advance(1);
  CHECK(f.backend_state &&
            f.backend_state->material_route == MaterialRecipeDisposition::device_native &&
            !f.backend_state->material_fallback_global_presence &&
            !f.backend_state->material_fallback_local_presence &&
            f.backend_state->material_fallback_presence_validated &&
            f.backend_state->material_fallback_statistics.warnings == 0 &&
            f.backend_state->material_fallback_statistics.dense_rows == 0 &&
            f.backend_state->material_fallback_statistics.callback_tiles == 0 &&
            f.backend_state->material_fallback_statistics.callback_points == 0 &&
            f.backend->material_fallback_warning_count() == 1,
        "fallback-to-native replacement retained stale fallback state");
  CHECK(f.initialization_plan && f.initialization_plan->materials.size() == 1 &&
            f.initialization_plan->materials[0].callback_owners().empty() &&
            f.initialization_plan->materials[0].callback_tiles().empty() &&
            f.initialization_plan->materials[0].dense_fallback_rows().empty(),
        "native replacement retained stale callback, tile, or dense fallback work");

  replace_isotropic_material_definition_for_testing(
      f, vacuum_definition.material_ir, unit_epsilon,
      "backend_api:native_to_vacuum_definition");
  f.advance(1);
  bool stale_material_state = !f.backend_state ||
                              f.backend_state->material_route !=
                                  MaterialRecipeDisposition::device_native ||
                              f.backend_state->material_fallback_global_presence ||
                              f.backend_state->material_fallback_local_presence;
  if (f.initialization_plan && f.initialization_plan->materials.size() == 1) {
    const MaterialRecipe &recipe = f.initialization_plan->materials[0];
    stale_material_state = stale_material_state || !recipe.callback_owners().empty() ||
                           !recipe.callback_tiles().empty() ||
                           !recipe.dense_fallback_rows().empty();
    for (const MaterialRecipeRow &row : recipe.rows())
      stale_material_state = stale_material_state ||
                             row.key.kind == int(array_kind::chi2) ||
                             row.key.kind == int(array_kind::chi3) ||
                             row.key.kind == int(array_kind::conductivity) ||
                             row.key.kind == int(array_kind::condinv) ||
                             row.key.kind == int(array_kind::sigma);
  }
  else
    stale_material_state = true;
  CHECK(and_to_all(!stale_material_state) &&
            f.backend->material_fallback_warning_count() == 1,
        "native-to-vacuum replacement retained stale rows, callbacks, or fallback presence");
}

static void test_material_promotion_transaction() {
  for (int failure = 0; failure < 6; ++failure) {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts);
    f->advance(1);
    counts.force_promotion_on_classify_call = 2;
    invalidate(*f, MutationKind::material_values,
               "backend_api:warm_classification_promotion");
    const BackendEpochSnapshot entry(*f);
    if (failure == 0) backend_set_cw_clone_fail_after_for_testing(0);
    counts.fail_create_state_on_call = failure == 1 ? 3 : 0;
    counts.fail_initialize_on_call = failure == 2 ? 3 : 0;
    counts.fail_classify_on_call = failure == 3 ? 3 : 0;
    counts.fail_finalize_on_call = failure == 4 ? 2 : 0;
    counts.fail_compile_on_call = failure == 5 ? 2 : 0;
    bool failed = false;
    try { f->advance(1); }
    catch (const std::exception &) { failed = true; }
    backend_set_cw_clone_fail_after_for_testing(-1);
    CHECK(and_to_all(failed) && entry.matches(*f) && !f->backend->is_poisoned(),
          "promotion second-pass failure %d changed the live epoch", failure);
    counts.fail_create_state_on_call = 0;
    counts.fail_initialize_on_call = 0;
    counts.fail_classify_on_call = 0;
    counts.fail_finalize_on_call = 0;
    counts.fail_compile_on_call = 0;
    counts.force_promotion_on_classify_call = counts.classified + 1;
    BackendState *old_state = f->backend_state;
    f->advance(1);
    CHECK(f->backend_state && f->backend_state != old_state &&
              f->classification_reentries == 1,
          "promotion second-pass failure %d did not retry exactly once", failure);
    delete f;
    delete s;
  }
}

static component_mask global_component_mask(fields &f) {
  int local[NUM_FIELD_COMPONENTS], global[NUM_FIELD_COMPONENTS];
  FOR_COMPONENTS(c) local[c] = f.have_component(c) ? 1 : 0;
  or_to_all(local, global, NUM_FIELD_COMPONENTS);
  component_mask result = 0;
  FOR_COMPONENTS(c)
    if (global[c]) result |= component_mask(1) << int(c);
  return result;
}

static void check_idle_rank_material_promotion(fields &f, lifetime_counts &counts,
                                               component_mask expected, const char *name) {
  bool owns_chunk = false;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    owns_chunk = owns_chunk || f.chunks[chunk]->is_mine();
  CHECK(sum_to_all(owns_chunk ? 1 : 0) == 1,
        "%s fixture did not assign its one chunk to exactly one rank", name);
  if (count_processors() > 1)
    CHECK(or_to_all(!owns_chunk), "%s fixture did not include an idle rank", name);

  f.backend = new tracking_backend(f, counts);
  f.advance(1);
  const component_mask present = global_component_mask(f);
  CHECK((present & expected) == expected,
        "%s promotion did not realize the globally required component layout", name);
  CHECK(and_to_all(f.classification_reentries == 1),
        "%s promotion did not re-enter preparation exactly once", name);
  CHECK(counts.classified >= 1, "%s promotion never reached backend classification", name);
  if (!owns_chunk)
    CHECK(!f.have_component(Ex) && !f.have_component(Ez),
          "%s idle rank acquired rank-local field storage", name);
}

static void test_material_idle_rank_component_promotion() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);

  {
    structure s(gv, unit_epsilon, no_pml(), identity(), 1);
    fields f(&s, 0, 0.2);
    f._require_component(Ez, false);
    lifetime_counts counts;
    component_mask expected = 0;
    FOR_COMPONENTS(c)
      if (f.gv.has_field(c)) expected |= component_mask(1) << int(c);
    check_idle_rank_material_promotion(f, counts, expected,
                                       "beta/aniso2d idle-rank");
  }

  {
    structure s(gv, unit_epsilon, no_pml(), identity(), 1);
    fields f(&s);
    f._require_component(Ez, false);
    lifetime_counts counts;
    counts.force_required_components = component_mask(1) << int(Ex);
    check_idle_rank_material_promotion(f, counts, counts.force_required_components,
                                       "required-component idle-rank");
  }
}

static void test_material_route_lattice() {
  if (count_processors() < 2) return;
  const bool first = my_rank() == 0;
  CHECK(reconcile_material_recipe_route(
            first ? MaterialRecipeDisposition::tiled_callback
                  : MaterialRecipeDisposition::hybrid_interface,
            true) == MaterialRecipeDisposition::host_reference,
        "asymmetric tiled/hybrid routes did not reconcile to host-reference");
  CHECK(reconcile_material_recipe_route(
            first ? MaterialRecipeDisposition::host_reference
                  : MaterialRecipeDisposition::device_native,
            true) == MaterialRecipeDisposition::host_reference,
        "host-reference did not dominate an asymmetric native route");
  CHECK(reconcile_material_recipe_route(
            first ? MaterialRecipeDisposition::tiled_callback
                  : MaterialRecipeDisposition::device_native,
            true) == MaterialRecipeDisposition::tiled_callback,
        "tiled callback did not dominate an asymmetric native route");
  CHECK(reconcile_material_recipe_route(
            first ? MaterialRecipeDisposition::hybrid_interface
                  : MaterialRecipeDisposition::device_native,
            true) == MaterialRecipeDisposition::hybrid_interface,
        "hybrid interface did not dominate an asymmetric native route");
  bool rejected = false;
  try {
    reconcile_material_recipe_route(
        first ? MaterialRecipeDisposition::host_reference
              : MaterialRecipeDisposition::device_native,
        !first);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  CHECK(and_to_all(rejected),
        "global host-reference route accepted an incomplete dense snapshot");
}

static void test_material_classification_fact_contract() {
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);
  CHECK(f->initialization_plan && f->initialization_plan->materials.size() == 1,
        "classification fact fixture has no material recipe");
  if (!f->initialization_plan || f->initialization_plan->materials.size() != 1) {
    delete f;
    delete s;
    return;
  }
  const MaterialRecipe &recipe = f->initialization_plan->materials[0];
  const MaterialClassification classification = classify(*f, *f->storage_plan);
  const MaterialClassificationFacts baseline =
      classification_facts_input(*f, *f->storage_plan, classification);
  CHECK(f->step_plans[0], "classification fixture has no rebuilt ordinary StepPlan");
  if (f->step_plans[0])
    validate_material_classification_consumers(*f->storage_plan, *f->initialization_plan,
                                               *f->step_plans[0], classification);

  CHECK(baseline.variants.size() > 1,
        "classification fixture did not produce multiple exact update regions");
  std::set<uint32_t> observed_variant_keys;
  for (const MaterialVariantClassificationFact &variant : baseline.variants)
    observed_variant_keys.insert(variant.variant_key);
  CHECK(observed_variant_keys.size() > 1,
        "classification fixture did not exercise multiple update variants");

  if (f->step_plans[0] && !classification.variant_facts.empty()) {
    MaterialClassification changed = classification;
    const bool changed_variant_owned =
        f->chunks[changed.variant_facts[0].chunk]->is_mine();
    changed.variant_facts[0].variant_key ^= constitutive_has_nonlinearity;
    bool rejected = false;
    try {
      validate_material_classification_consumers(*f->storage_plan, *f->initialization_plan,
                                                 *f->step_plans[0], changed);
    }
    catch (const std::invalid_argument &) { rejected = true; }
    CHECK(or_to_all(rejected),
          "post-plan validation accepted a changed material variant key collectively");
    CHECK(and_to_all(!changed_variant_owned || rejected),
          "post-plan validation accepted a changed material variant key on its owning rank");

    changed = classification;
    const bool missing_variant_owned =
        f->chunks[changed.variant_facts.back().chunk]->is_mine();
    changed.variant_facts.pop_back();
    rejected = false;
    try {
      validate_material_classification_consumers(*f->storage_plan, *f->initialization_plan,
                                                 *f->step_plans[0], changed);
    }
    catch (const std::invalid_argument &) { rejected = true; }
    CHECK(or_to_all(rejected),
          "post-plan validation accepted a missing material update region collectively");
    CHECK(and_to_all(!missing_variant_owned || rejected),
          "post-plan validation accepted a missing material update region on its owning rank");

    changed = classification;
    changed.variant_facts.push_back(changed.variant_facts.back());
    rejected = false;
    try {
      validate_material_classification_consumers(*f->storage_plan, *f->initialization_plan,
                                                 *f->step_plans[0], changed);
    }
    catch (const std::invalid_argument &) { rejected = true; }
    CHECK(rejected, "post-plan validation accepted an extra material update region");

    int owned_chunk = -1;
    for (const MaterialVariantClassificationFact &variant : classification.variant_facts)
      if (f->chunks[variant.chunk]->is_mine()) {
        owned_chunk = variant.chunk;
        break;
      }
    CHECK(owned_chunk >= 0,
          "post-plan validation fixture has no classified update region for an owned chunk");
    if (owned_chunk >= 0) {
      StepPlan missing_chunk = *f->step_plans[0];
      std::vector<ConstitutiveUpdate> retained;
      for (Operation &operation : missing_chunk.operations) {
        if (operation.kind != OpKind::update_eh) continue;
        const size_t begin = operation.descriptor_index;
        const size_t end = begin + operation.descriptor_count;
        operation.descriptor_index = uint32_t(retained.size());
        for (size_t i = begin; i < end; ++i)
          if (missing_chunk.eh_updates[i].region.chunk != owned_chunk)
            retained.push_back(missing_chunk.eh_updates[i]);
        operation.descriptor_count = uint32_t(retained.size()) - operation.descriptor_index;
      }
      missing_chunk.eh_updates.swap(retained);
      rejected = false;
      try {
        validate_material_classification_consumers(*f->storage_plan, *f->initialization_plan,
                                                   missing_chunk, classification);
      }
      catch (const std::invalid_argument &) { rejected = true; }
      CHECK(rejected,
            "post-plan validation accepted deletion of every update region for an owned chunk");
    }
  }

  MaterialClassificationFacts malformed = baseline;
  ++malformed.version;
  expect_material_classification_facts_rejected(
      *f, *f->storage_plan, recipe, malformed, "wrong material classification version");

  malformed = baseline;
  if (!malformed.variants.empty()) {
    malformed.variants[0].variant_key ^= UINT32_C(1) << 31;
    expect_material_classification_facts_rejected(
        *f, *f->storage_plan, recipe, malformed, "malformed material variant fact");
  }

  malformed = baseline;
  if (malformed.variants.size() > 1) {
    std::swap(malformed.variants[0], malformed.variants[1]);
    expect_material_classification_facts_rejected(
        *f, *f->storage_plan, recipe, malformed, "reordered material update regions");
  }

  malformed = baseline;
  if (!malformed.variants.empty()) {
    ++malformed.variants[0].base;
    expect_material_classification_facts_rejected(
        *f, *f->storage_plan, recipe, malformed, "changed material update region");
  }

  malformed = baseline;
  if (!malformed.variants.empty()) {
    malformed.variants.pop_back();
    expect_material_classification_facts_rejected(
        *f, *f->storage_plan, recipe, malformed, "missing material update region");
  }

  malformed = baseline;
  if (!malformed.variants.empty()) {
    malformed.variants.push_back(malformed.variants.back());
    expect_material_classification_facts_rejected(
        *f, *f->storage_plan, recipe, malformed, "extra material update region");
  }

  malformed = baseline;
  if (!malformed.rows.empty()) {
    malformed.rows.pop_back();
    expect_material_classification_facts_rejected(
        *f, *f->storage_plan, recipe, malformed, "missing material row fact");
  }

  malformed = baseline;
  if (!malformed.rows.empty()) {
    malformed.rows.push_back(malformed.rows.back());
    expect_material_classification_facts_rejected(
        *f, *f->storage_plan, recipe, malformed, "extra material row fact");
  }

  malformed = baseline;
  bool changed_group = false;
  for (MaterialRowClassificationFact &row : malformed.rows)
    if (row.key.kind == int(array_kind::pml_sig) &&
        row.state == MaterialClassification::retained) {
      row.state = MaterialClassification::elided_row;
      changed_group = true;
      break;
    }
  CHECK(or_to_all(changed_group), "classification fact fixture has no retained PML group");
  if (or_to_all(changed_group))
    expect_material_classification_facts_rejected(
        *f, *f->storage_plan, recipe, malformed, "broken material PML group fact");

  MaterialClassificationFacts asymmetric = baseline;
  if (my_rank() == 0) asymmetric.has_nonlinearities = true;
  const MaterialClassification combined =
      assemble_material_classification(*f, *f->storage_plan, recipe, asymmetric);
  CHECK(and_to_all(combined.has_nonlinearities),
        "rank-asymmetric scalar facts were not normalized globally");
  uint64_t reference_hash = combined.hash;
  broadcast(0, &reference_hash, 1);
  CHECK(combined.hash == reference_hash,
        "rank-asymmetric fact normalization produced different hashes");

  for (int target = 0; target < count_processors(); ++target)
    for (int mode = 1; mode <= 4; ++mode) {
      backend_set_material_classification_failure_for_testing(target, mode);
      bool failed = false;
      try { (void)assemble_material_classification(*f, *f->storage_plan, recipe, baseline); }
      catch (const std::runtime_error &) { failed = true; }
      backend_set_material_classification_failure_for_testing(-1, 0);
      CHECK(and_to_all(failed),
            "rank-targeted material classification failure mode %d did not reconcile", mode);
      const MaterialClassification retry =
          assemble_material_classification(*f, *f->storage_plan, recipe, baseline);
      CHECK(retry.hash == classification.hash,
            "material classification failure mode %d was not retryable", mode);
    }

  StoragePlan malformed_plan = *f->storage_plan;
  ArraySpec tombstone = malformed_plan.arrays.front();
  tombstone.id = ArrayId{uint32_t(malformed_plan.arrays.size())};
  tombstone.role = array_role::material;
  tombstone.alias_of = invalid_array();
  tombstone.classification_provisional = false;
  tombstone.classification_elided = true;
  malformed_plan.arrays.push_back(tombstone);
  StorageKey tombstone_key = malformed_plan.keys.front();
  tombstone_key.chunk = f->num_chunks + 17;
  tombstone_key.kind = int(array_kind::chi2);
  tombstone_key.component_ = Ez;
  tombstone_key.cmp = -1;
  tombstone_key.aux = 0;
  malformed_plan.keys.push_back(tombstone_key);
  const ArrayId tombstone_id = tombstone.id;
  InitializationPlan empty_initialization;
  StepPlan empty_steps;

  StoragePlan alias_plan = malformed_plan;
  alias_plan.arrays[0].alias_of = tombstone_id;
  bool rejected = false;
  try {
    validate_material_classification_consumers(alias_plan, empty_initialization, empty_steps,
                                               classification);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  CHECK(rejected, "material tombstone alias dependency was accepted");

  InitializationPlan malformed_initialization;
  InitOperation init = {};
  init.kind = InitKind::zero;
  init.destination = ArrayRef{tombstone_id, 0, 1};
  malformed_initialization.operations.push_back(init);
  rejected = false;
  try {
    validate_material_classification_consumers(malformed_plan, malformed_initialization,
                                               empty_steps, classification);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  CHECK(rejected, "material tombstone initialization action was accepted");

  StepPlan malformed_access;
  Operation operation = {};
  operation.accesses.push_back(BufferAccess{ArrayRef{tombstone_id, 0, 1}, AccessMode::read});
  malformed_access.operations.push_back(operation);
  rejected = false;
  try {
    validate_material_classification_consumers(malformed_plan, empty_initialization,
                                               malformed_access, classification);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  CHECK(rejected, "material tombstone step access was accepted");

  StepPlan malformed_action;
  ConstitutiveUpdate action = {};
  action.diagonal = tombstone_id;
  malformed_action.eh_updates.push_back(action);
  rejected = false;
  try {
    validate_material_classification_consumers(malformed_plan, empty_initialization,
                                               malformed_action, classification);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  CHECK(rejected, "material tombstone direct action was accepted");

  delete f;
  delete s;
}

static void test_material_classification_collective_failures() {
  structure *s;
  fields *f;
  build(&s, &f);
  lifetime_counts counts;
  f->backend = new tracking_backend(*f, counts);
  f->advance(1);
  for (int target = 0; target < count_processors(); ++target)
    for (int mode = 1; mode <= 4; ++mode) {
      invalidate(*f, MutationKind::material_values,
                 "backend_api:classification_collective_failure");
      const BackendEpochSnapshot entry(*f);
      backend_set_material_classification_failure_for_testing(target, mode);
      bool failed = false;
      try { f->advance(1); }
      catch (const std::runtime_error &) { failed = true; }
      backend_set_material_classification_failure_for_testing(-1, 0);
      CHECK(and_to_all(failed) && entry.matches(*f) && !f->backend->is_poisoned(),
            "classification collective failure mode %d changed the live epoch", mode);
      BackendState *const old_state = f->backend_state;
      f->advance(1);
      CHECK(f->backend_state && f->backend_state != old_state,
            "classification collective failure mode %d was not retryable", mode);
    }
  delete f;
  delete s;
}

static void test_resident_material_recipe_lifecycle() {
  for (int target = 0; target < count_processors(); ++target)
    for (int mode = 1; mode <= 2; ++mode) {
      structure *s;
      fields *f;
      build(&s, &f);
      lifetime_counts counts;
      f->backend = new tracking_backend(*f, counts);
      InitializationPlan *const entry_initialization = f->initialization_plan;
      set_material_recipe_failure_for_testing(target, mode);
      bool failed = false;
      try { f->advance(1); }
      catch (const std::runtime_error &) { failed = true; }
      set_material_recipe_failure_for_testing(-1, 0);
      const bool all_failed = and_to_all(failed);
      CHECK(all_failed && !f->backend_state && !f->executable &&
                f->initialization_plan == entry_initialization && counts.states_created == 0 &&
                counts.initialized == 0 && counts.advance_attempts == 0,
            "rank-asymmetric material recipe capture failure published an epoch");
      f->advance(1);
      CHECK(f->backend_state && f->initialization_plan && f->executable,
            "material recipe capture failure was not retryable");
      delete f;
      delete s;
    }

  {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts);
    InitializationPlan *const entry_initialization = f->initialization_plan;
    counts.fail_create_state = true;
    bool failed = false;
    try { f->advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    counts.fail_create_state = false;
    const bool all_failed = and_to_all(failed);
    CHECK(all_failed && !f->backend_state && !f->executable &&
              f->initialization_plan == entry_initialization,
          "failed material recipe state creation partially published the staged epoch");
    f->advance(1);
    CHECK(counts.states_created == 1 && counts.initialized == 1 && counts.classified == 1 &&
              counts.finalized == 1 && counts.executables_created == 1,
          "resident material recipe did not traverse one complete lifecycle");
    CHECK(counts.material_arrays_at_create == counts.provisional_material_arrays_at_create,
          "resident state creation saw non-provisional material storage");
    CHECK(counts.material_recipe_rows_at_initialize == counts.material_arrays_at_create &&
              counts.material_recipe_signature_at_initialize ==
                  f->initialization_plan->materials[0].signature(),
          "resident initialization recipe does not cover its provisional rows");
    CHECK(counts.provisional_material_arrays_at_compile == 0,
          "resident compile ran before material classification resolution");

    const MaterialRecipe retained_recipe = f->initialization_plan->materials[0];
    const uint64_t retained_signature = retained_recipe.signature();
    InitializationPlan *const live_initialization = f->initialization_plan;
    invalidate(*f, MutationKind::material_values, "material recipe refresh rollback");
    set_material_recipe_failure_for_testing(my_rank() == 0 ? 0 : -1, 1);
    failed = false;
    try { f->advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    set_material_recipe_failure_for_testing(-1, 0);
    const bool warm_all_failed = and_to_all(failed);
    CHECK(warm_all_failed && f->initialization_plan == live_initialization &&
              f->initialization_plan->materials[0].signature() == retained_signature &&
              retained_recipe.signature() == retained_signature,
          "failed warm recipe refresh replaced or mutated the owned recipe");
    f->advance(1);
    CHECK(f->initialization_plan &&
              f->initialization_plan->materials[0].signature() == retained_signature,
          "successful warm recipe refresh changed unchanged material identity");
    delete f;
    validate_material_recipe(retained_recipe);
    CHECK(retained_recipe.signature() == retained_signature,
          "owned material recipe lifetime depended on its fields object");
    delete s;
  }

  for (int target = 0; target < count_processors(); ++target)
    for (int failure = 0; failure < 8; ++failure) {
      structure *s;
      fields *f;
      build(&s, &f);
      lifetime_counts counts;
      f->backend = new tracking_backend(*f, counts);
      f->advance(1);
      invalidate(*f, MutationKind::material_values, "material candidate failure matrix");
      const BackendEpochSnapshot entry(*f);
      const int created_before = counts.states_created;
      const int destroyed_before = counts.states_destroyed;
      const int executables_before = counts.executables_created;
      const int executables_destroyed_before = counts.executables_destroyed;
      const bool inject = my_rank() == target;
      counts.fail_create_state = inject && failure == 0;
      counts.fail_initialize = inject && failure == 1;
      counts.fail_classify = inject && failure == 2;
      counts.fail_finalize = inject && failure == 3;
      if (failure == 4)
        backend_set_material_candidate_plan_failure_for_testing(target, 1);
      counts.fail_compile = inject && failure == 5;
      counts.corrupt_catalog_after_compile = inject && failure == 6;
      counts.fail_rebuild = inject && failure == 7;
      bool failed = false;
      try { f->advance(1); }
      catch (const std::runtime_error &) { failed = true; }
      backend_set_material_candidate_plan_failure_for_testing(-1, 0);
      CHECK(and_to_all(failed) && entry.matches(*f) && !f->backend->is_poisoned(),
            "warm material candidate failure %d changed the installed epoch", failure);
      CHECK(counts.states_created - created_before == counts.states_destroyed - destroyed_before &&
                counts.executables_created - executables_before ==
                    counts.executables_destroyed - executables_destroyed_before,
            "warm material candidate failure %d leaked staged state or executable", failure);
      counts.fail_create_state = false;
      counts.fail_initialize = false;
      counts.fail_classify = false;
      counts.fail_finalize = false;
      counts.fail_compile = false;
      counts.corrupt_catalog_after_compile = false;
      counts.fail_rebuild = false;
      BackendState *const old_state = f->backend_state;
      Executable *const old_executable = f->executable;
      f->advance(1);
      CHECK(f->backend_state && f->executable && f->backend_state != old_state &&
                f->executable != old_executable,
            "warm material candidate failure %d was not retryable", failure);
      delete f;
      delete s;
    }

  for (int target = 0; target < count_processors(); ++target) {
    structure *s;
    fields *f;
    build(&s, &f);
    f->add_flux_vol(X, volume(vec(0.1, -0.8), vec(0.1, 0.8)));
    lifetime_counts counts;
    f->backend = new tracking_backend(*f, counts);
    f->advance(1);
    CHECK(f->descriptors && !f->descriptors->legacy_fluxes.empty(),
          "warm material/flux failure fixture has no live legacy-flux recipe");
    invalidate(*f, MutationKind::material_values,
               "material candidate live-flux plan failure");
    const BackendEpochSnapshot entry(*f);
    backend_set_material_candidate_plan_failure_for_testing(target, 1);
    bool failed = false;
    try { f->advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    backend_set_material_candidate_plan_failure_for_testing(-1, 0);
    CHECK(and_to_all(failed) && entry.matches(*f) && !f->backend->is_poisoned(),
          "rank-asymmetric warm material/live-flux plan failure changed the installed epoch");
    f->advance(1);
    CHECK(f->descriptors && !f->descriptors->legacy_fluxes.empty(),
          "warm material/live-flux plan failure was not retryable");
    delete f;
    delete s;
  }

  for (int failure = 0; failure < 3; ++failure) {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    counts.fail_initialize = failure == 0;
    counts.fail_classify = failure == 1;
    counts.fail_finalize = failure == 2;
    f->backend = new tracking_backend(*f, counts);
    bool failed = false;
    try { f->advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    const bool all_failed = and_to_all(failed);
    CHECK(all_failed && !f->backend_state && !f->initialization_plan && !f->executable &&
              f->t == 0,
          "cold material initialize/finalize failure published a partial epoch");
    counts.fail_initialize = false;
    counts.fail_classify = false;
    counts.fail_finalize = false;
    f->advance(1);
    const bool retry_ok = f->backend_state && f->initialization_plan && f->executable &&
                          counts.provisional_material_arrays_at_compile == 0;
    if (!retry_ok)
      fprintf(stderr,
              "material retry %d state=%d initialization=%d executable=%d provisional=%zu "
              "initialized=%d classified=%d finalized=%d\n",
              failure, int(f->backend_state != NULL),
              int(f->initialization_plan != NULL), int(f->executable != NULL),
              counts.provisional_material_arrays_at_compile, counts.initialized,
              counts.classified, counts.finalized);
    CHECK(retry_ok,
          "material initialize/finalize retry replaced or incompletely resolved its epoch");
    delete f;
    delete s;
  }

  for (int malformed = 1; malformed <= 5; ++malformed) {
    structure *s;
    fields *f;
    build(&s, &f);
    lifetime_counts counts;
    counts.malformed_material_classification = malformed;
    f->backend = new tracking_backend(*f, counts);
    bool failed = false;
    try { f->advance(1); }
    catch (const std::runtime_error &) { failed = true; }
    CHECK(and_to_all(failed) && !f->backend_state && !f->executable && f->t == 0,
          "malformed classification mode %d published a partial epoch", malformed);
    counts.malformed_material_classification = 0;
    f->advance(1);
    CHECK(f->backend_state && f->executable,
          "malformed classification mode %d was not retryable", malformed);
    delete f;
    delete s;
  }
}

/* restrict_to has no Phase-1 consumer -- the in-place design update that would
   use it is deferred -- so it is built and unit-tested here rather than wired
   in. */
static void test_initialization_plan() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->advance(3);

  const InitializationPlan plan = build_initialization_plan(*f);
  CHECK(or_to_all(!plan.operations.empty()), "the initialization plan is empty");
  CHECK(!plan.materials.empty(), "the initialization plan has no material recipe");

  size_t zero_ops = 0, material_ops = 0, pml_ops = 0;
  for (const InitOperation &op : plan.operations) {
    if (op.kind == InitKind::zero) ++zero_ops;
    if (op.kind == InitKind::material_geometry) ++material_ops;
    if (op.kind == InitKind::pml_profile) ++pml_ops;
  }
  CHECK(or_to_all(zero_ops > 0), "no field arrays are initialized to zero");
  CHECK(or_to_all(material_ops > 0), "no arrays come from the material geometry");
  CHECK(or_to_all(pml_ops > 0), "a PML simulation produced no pml_profile operations");

  const InitializationPlan whole = plan.restrict_to(InitRegion());
  CHECK(whole.operations.size() == plan.operations.size(),
        "restrict_to(whole) dropped %zu operations",
        plan.operations.size() - whole.operations.size());
  CHECK(whole.material_values_generation == plan.material_values_generation &&
            whole.material_region_generation == plan.material_region_generation,
        "restrict_to(whole) changed material recipe generations");

  InitRegion narrow(0, ivec(2, 2), ivec(4, 4));
  const InitializationPlan sub = plan.restrict_to(narrow);
  CHECK(sub.operations.size() <= plan.operations.size(), "restrict_to grew the plan");
  CHECK(sub.materials.size() == plan.materials.size(), "restrict_to dropped the recipes");
  CHECK(sub.material_values_generation == plan.material_values_generation &&
            sub.material_region_generation == plan.material_region_generation,
        "restrict_to changed material recipe generations");

  master_printf("init plan: %zu ops (%zu zero, %zu material, %zu pml), restricted to %zu\n",
                plan.operations.size(), zero_ops, material_ops, pml_ops, sub.operations.size());
  delete f;
  delete s;
}

/* A storage rebuild must give the backend a chance to preserve authoritative
   values before either its executable or state is destroyed. A refusal leaves
   both objects live. */
static void test_authority_safe_state_rebuild() {
  structure *s;
  fields *f;
  build(&s, &f);

  rebuild_trace trace;
  rebuild_backend_base *tracking = new rebuild_tracking_backend(*f, trace);
  f->backend = tracking;
  f->backend_state = tracking->create_state(*f->storage_plan);
  StepPlan plan;
  f->executable = tracking->compile(plan, *f->backend_state);
  trace.events.clear();
  clear_dirty(*f, DirtyMask(f->dirty_mask));
  invalidate(*f, MutationKind::field_layout);

  f->init_backend();
  CHECK(trace.events.size() == 4, "rebuild produced %zu events, expected 4", trace.events.size());
  if (trace.events.size() == 4) {
    CHECK(trace.events[0] == "prepare-rebuild", "first rebuild event is %s",
          trace.events[0].c_str());
    CHECK(trace.events[1] == "destroy-executable", "second rebuild event is %s",
          trace.events[1].c_str());
    CHECK(trace.events[2] == "destroy-state", "third rebuild event is %s", trace.events[2].c_str());
    CHECK(trace.events[3] == "create-state", "fourth rebuild event is %s", trace.events[3].c_str());
  }
  CHECK((trace.reasons & dirty_storage) != 0, "rebuild hook did not receive dirty_storage");

  delete f;
  delete s;

  build(&s, &f);
  rebuild_trace refused;
  tracking = new rebuild_backend_base(*f, refused);
  f->backend = tracking;
  f->backend_state = tracking->create_state(*f->storage_plan);
  BackendState *live_state = f->backend_state;
  refused.events.clear();
  clear_dirty(*f, DirtyMask(f->dirty_mask));
  invalidate(*f, MutationKind::field_layout);
  bool rejected = false;
  try {
    f->init_backend();
  }
  catch (const std::logic_error &) {
    rejected = true;
  }
  CHECK(rejected, "backend rebuild refusal did not propagate");
  CHECK(f->backend_state == live_state, "a refused rebuild destroyed the live state");
  CHECK(f->executable == NULL, "refused rebuild unexpectedly created an executable");
  CHECK(refused.events.empty(), "default-safe rebuild destroyed or replaced the live state");

  clear_dirty(*f, DirtyMask(f->dirty_mask));
  delete f;
  delete s;
}

static void test_cpu_state_rebuild_is_safe_noop() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->advance(1);
  CHECK(f->backend_state != NULL && f->executable != NULL,
        "CPU advance did not create backend artifacts");

  BackendState *state = f->backend_state;
  Executable *executable = f->executable;
  f->backend->prepare_state_rebuild(*state, dirty_storage);
  CHECK(f->backend_state == state && f->executable == executable,
        "CPU authority-safe rebuild hook modified host-authoritative state");

  delete f;
  delete s;
}

static void test_backend_safe_host_access() {
  structure *s;
  fields *f;
  build(&s, &f);
  component components[1] = {Ez};
  const double frequencies[2] = {0.2, 0.3};
  dft_fields monitor =
      f->add_dft_fields(components, 1, volume(vec(0.4, 0.4), vec(1.2, 1.2)), frequencies, 2);

  rebuild_trace rebuilds;
  access_trace accesses;
  f->backend = new access_tracking_backend(*f, rebuilds, accesses);
  f->advance(1);

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  const std::complex<double> point = f->get_field(Ez, vec(0.7, 0.8), true);
  CHECK(fabs(point.real() - 2.5) < 1e-12 && fabs(point.imag() - 2.5) < 1e-12,
        "point query did not consume backend-refreshed real/imaginary values");
  CHECK(sum_to_all(int(accesses.field_reads)) > 0 && max_to_all(int(accesses.max_elements)) == 1,
        "point query did not issue exact one-element backend reads");

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  std::unique_ptr<std::complex<realnum>[]> slice(
      f->get_complex_array_slice(volume(vec(0.5, 0.5), vec(1.0, 1.0)), Ez));
  CHECK(slice.get() != NULL && sum_to_all(int(accesses.field_reads)) > 0,
        "array slice did not issue explicit backend field reads");
  size_t largest_field_allocation = 0;
  for (size_t i = 0; i < f->array_catalog->size(); ++i) {
    const ArraySpec &spec = f->array_catalog->spec(ArrayId{uint32_t(i)});
    if (spec.role == array_role::field && spec.elements > largest_field_allocation)
      largest_field_allocation = spec.elements;
  }
  CHECK(!accesses.field_reads || accesses.max_elements < largest_field_allocation,
        "small array slice refreshed a complete unrelated field allocation");

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  const volume access_region(vec(0.45, 0.45), vec(1.15, 1.05));
  const double field_max = f->max_abs(Ez, access_region);
  CHECK(fabs(field_max - sqrt(12.5)) < 1e-12,
        "integration did not consume backend-refreshed field values: %.17g", field_max);
  CHECK(sum_to_all(int(accesses.field_reads)) > 0,
        "integration did not issue explicit backend field reads");
  CHECK(!accesses.field_reads || accesses.max_elements < largest_field_allocation,
        "small integration refreshed a complete unrelated field allocation");

  accesses.fail_read_rank = 0;
  bool integration_failure = false;
  try {
    (void)f->max_abs(Ez, access_region);
  }
  catch (const std::runtime_error &) {
    integration_failure = true;
  }
  CHECK(sum_to_all(int(integration_failure)) == count_processors(),
        "integration read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;

  structure *s2;
  fields *f2;
  build(&s2, &f2);
  rebuild_trace rebuilds2;
  access_trace accesses2;
  f2->backend = new access_tracking_backend(*f2, rebuilds2, accesses2);
  f2->advance(1);
  accesses2.fail_read_rank = 0;
  bool integration2_failure = false;
  try {
    (void)f->integrate2(*f2, 1, components, 1, components, multiply_fields, NULL, access_region);
  }
  catch (const std::runtime_error &) {
    integration2_failure = true;
  }
  CHECK(sum_to_all(int(integration2_failure)) == count_processors(),
        "second integration read failure was not reconciled on every rank");
  accesses2.fail_read_rank = -1;
  delete f2;
  delete s2;

#ifdef HAVE_HDF5
  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  h5file *field_file = new h5file("backend-api-field-access.h5", h5file::WRITE, true);
  f->output_hdf5(Ez, access_region, field_file);
  CHECK(sum_to_all(int(accesses.field_reads)) > 0,
        "ordinary HDF5 output did not issue explicit backend field reads");
  CHECK(!accesses.field_reads || accesses.max_elements < largest_field_allocation,
        "small HDF5 output refreshed a complete unrelated field allocation");
  field_file->remove();
  delete field_file;

  accesses.fail_read_rank = 0;
  bool hdf5_failure = false;
  field_file = new h5file("backend-api-field-failure.h5", h5file::WRITE, true);
  try {
    f->output_hdf5(Ez, access_region, field_file);
  }
  catch (const std::runtime_error &) {
    hdf5_failure = true;
  }
  CHECK(sum_to_all(int(hdf5_failure)) == count_processors(),
        "ordinary HDF5 read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;
  delete field_file;
  if (am_master()) std::remove("backend-api-field-failure.h5");
  all_wait();
#endif

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  int rank = 0;
  size_t dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> dft(f->get_dft_array(monitor, Ez, 0, &rank, dims));
  CHECK(dft.get() != NULL && sum_to_all(int(accesses.dft_reads)) > 0,
        "DFT array query did not refresh its accumulator storage");
  int local_dft_chunks = 0;
  for (dft_chunk *cur = monitor.chunks; cur; cur = cur->next_in_dft)
    ++local_dft_chunks;
  CHECK(sum_to_all(int(accesses.dft_reads)) == sum_to_all(local_dft_chunks),
        "DFT array query redundantly refreshed accumulator storage");

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  dft_chunk *chunklists[1] = {monitor.chunks};
  std::unique_ptr<std::complex<realnum>[]> encoded_dft;
  std::complex<realnum> *encoded_array = NULL;
  direction dirs[3];
  f->process_dft_component(chunklists, 1, 0, NO_COMPONENT, NULL, &encoded_array, &rank, dims, dirs);
  encoded_dft.reset(encoded_array);
  CHECK(encoded_dft.get() != NULL && sum_to_all(int(accesses.dft_reads)) > 0,
        "encoded DFT component did not refresh its normalized accumulator storage");
  CHECK(sum_to_all(int(accesses.dft_reads)) == sum_to_all(local_dft_chunks),
        "encoded DFT component redundantly refreshed accumulator storage");

  accesses.fail_read_rank = 0;
  bool dft_array_failure = false;
  try {
    std::unique_ptr<std::complex<realnum>[]> failed(f->get_dft_array(monitor, Ez, 0, &rank, dims));
  }
  catch (const std::runtime_error &) {
    dft_array_failure = true;
  }
  CHECK(sum_to_all(int(dft_array_failure)) == count_processors(),
        "DFT array read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;

  accesses.reads = accesses.writes = accesses.field_reads = accesses.dft_reads = 0;
  accesses.field_writes = accesses.dft_writes = accesses.max_elements = 0;
  backend_refresh_dft_chain(monitor.chunks, "backend_api DFT chain read");
  CHECK(sum_to_all(int(accesses.dft_reads)) == sum_to_all(local_dft_chunks),
        "DFT chain boundary did not read each local chunk exactly once");
  backend_publish_dft_chain(monitor.chunks, "backend_api DFT chain write");
  CHECK(sum_to_all(int(accesses.dft_writes)) == sum_to_all(local_dft_chunks),
        "DFT chain boundary did not publish each local chunk exactly once");

  accesses.fail_write_rank = 0;
  bool dft_write_failure = false;
  try {
    backend_publish_dft_chain(monitor.chunks, "backend_api injected DFT write");
  }
  catch (const std::runtime_error &) {
    dft_write_failure = true;
  }
  CHECK(sum_to_all(int(dft_write_failure)) == count_processors(),
        "rank-asymmetric DFT write failure was not reconciled on every rank");
  accesses.fail_write_rank = -1;
  backend_publish_dft_chain(monitor.chunks, "backend_api recovered DFT write");
  const int time_before_recovery = f->t;
  f->advance(1);
  CHECK(f->t == time_before_recovery + 1,
        "execution did not continue after a reconciled DFT write failure");

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  dft_ldos ldos(frequencies, sizeof(frequencies) / sizeof(*frequencies));
  ldos.update(*f);
  CHECK(sum_to_all(int(accesses.field_reads)) > 0,
        "LDOS update did not refresh its source-point fields");
  CHECK(max_to_all(int(accesses.max_elements)) == 1,
        "LDOS update read more than one field element at a time");

  accesses.fail_read_rank = 0;
  bool ldos_failure = false;
  try {
    ldos.update(*f);
  }
  catch (const std::runtime_error &) {
    ldos_failure = true;
  }
  CHECK(sum_to_all(int(ldos_failure)) == count_processors(),
        "rank-asymmetric LDOS read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;
  ldos.update(*f);

  BackendState *state_before_magnetic_rejection = f->backend_state;
  bool direct_magnetic_failure = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    direct_magnetic_failure = true;
  }
  CHECK(sum_to_all(int(direct_magnetic_failure)) == count_processors(),
        "resident magnetic synchronization was not rejected on every rank");
  CHECK(f->backend_state == state_before_magnetic_rejection,
        "magnetic rejection replaced resident state");
  bool repeated_magnetic_failure = false;
  try {
    f->synchronize_magnetic_fields();
  }
  catch (const std::runtime_error &) {
    repeated_magnetic_failure = true;
  }
  CHECK(sum_to_all(int(repeated_magnetic_failure)) == count_processors(),
        "magnetic rejection changed nesting state before returning");

  bool energy_magnetic_failure = false;
  try {
    (void)f->field_energy_in_box(f->v);
  }
  catch (const std::runtime_error &) {
    energy_magnetic_failure = true;
  }
  CHECK(sum_to_all(int(energy_magnetic_failure)) == count_processors(),
        "field_energy_in_box did not reject unsupported magnetic synchronization");

  bool flux_magnetic_failure = false;
  try {
    (void)f->flux_in_box(X, f->v);
  }
  catch (const std::runtime_error &) {
    flux_magnetic_failure = true;
  }
  CHECK(sum_to_all(int(flux_magnetic_failure)) == count_processors(),
        "flux_in_box did not reject unsupported magnetic synchronization");
  const int time_before_magnetic_recovery = f->t;
  f->advance(1);
  CHECK(f->t == time_before_magnetic_recovery + 1,
        "execution did not continue after magnetic-sync rejection");

  dft_flux flux(Ez, Ez, monitor.chunks, monitor.chunks, frequencies, 2, monitor.where, NO_DIRECTION,
                true);
  flux.monitor_lifetime = monitor.monitor_lifetime;
  dft_energy energy(monitor.chunks, monitor.chunks, monitor.chunks, monitor.chunks, frequencies, 2,
                    monitor.where);
  energy.monitor_lifetime = monitor.monitor_lifetime;
  dft_force force(monitor.chunks, monitor.chunks, monitor.chunks, frequencies, 2, monitor.where);
  force.monitor_lifetime = monitor.monitor_lifetime;
  const direction periodic_d[2] = {NO_DIRECTION, NO_DIRECTION};
  const int periodic_n[2] = {0, 0};
  const double periodic_k[2] = {0.0, 0.0};
  const double period[2] = {0.0, 0.0};
  dft_near2far near2far(monitor.chunks, frequencies, 2, 1.0, 1.0, monitor.where, periodic_d,
                        periodic_n, periodic_k, period);
  near2far.monitor_lifetime = monitor.monitor_lifetime;

  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  if (am_master()) {
    std::vector<std::complex<double> > local_farfield(6 *
                                                      (sizeof(frequencies) / sizeof(*frequencies)));
    near2far.farfield_lowlevel(local_farfield.data(), vec(2.0, 2.0));
  }
  all_wait();
  CHECK(sum_to_all(int(accesses.dft_reads)) > 0,
        "rank-local far-field query did not refresh its accumulator storage");

  accesses.fail_read_rank = 0;
  bool dft_norm_failure = false;
  try {
    (void)f->dft_norm();
  }
  catch (const std::runtime_error &) {
    dft_norm_failure = true;
  }
  CHECK(sum_to_all(int(dft_norm_failure)) == count_processors(),
        "DFT norm read failure was not reconciled on every rank");

  bool flux_failure = false;
  try {
    delete[] flux.flux();
  }
  catch (const std::runtime_error &) {
    flux_failure = true;
  }
  CHECK(sum_to_all(int(flux_failure)) == count_processors(),
        "flux read failure was not reconciled on every rank");

  bool electric_failure = false;
  try {
    delete[] energy.electric();
  }
  catch (const std::runtime_error &) {
    electric_failure = true;
  }
  CHECK(sum_to_all(int(electric_failure)) == count_processors(),
        "electric-energy read failure was not reconciled on every rank");

  bool magnetic_failure = false;
  try {
    delete[] energy.magnetic();
  }
  catch (const std::runtime_error &) {
    magnetic_failure = true;
  }
  CHECK(sum_to_all(int(magnetic_failure)) == count_processors(),
        "magnetic-energy read failure was not reconciled on every rank");

  bool force_failure = false;
  try {
    delete[] force.force();
  }
  catch (const std::runtime_error &) {
    force_failure = true;
  }
  CHECK(sum_to_all(int(force_failure)) == count_processors(),
        "force read failure was not reconciled on every rank");

  bool farfield_failure = false;
  try {
    delete[] near2far.farfield(vec(2.0, 2.0));
  }
  catch (const std::runtime_error &) {
    farfield_failure = true;
  }
  CHECK(sum_to_all(int(farfield_failure)) == count_processors(),
        "far-field read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;

  accesses.fail_read_rank = 0;
  bool symmetric_failure = false;
  try {
    (void)f->get_field(Ez, vec(0.7, 0.8), true);
  }
  catch (const std::runtime_error &) {
    symmetric_failure = true;
  }
  CHECK(symmetric_failure, "rank-local backend read failure was not reconciled on every rank");
  accesses.fail_read_rank = -1;

  char filename[160];
  snprintf(filename, sizeof(filename), "/tmp/meep-backend-access-%ld-%d.h5", long(getpid()),
           my_rank());
  accesses.reads = accesses.field_reads = accesses.dft_reads = accesses.max_elements = 0;
  const uint64_t checkpoint_generation = std::numeric_limits<uint64_t>::max();
  f->mutation_generation[static_cast<int>(MutationKind::source_values)] =
      checkpoint_generation;
  f->dump(filename, false);
  CHECK(sum_to_all(int(accesses.field_reads)) > 0 && sum_to_all(int(accesses.dft_reads)) > 0,
        "checkpoint dump did not explicitly read field and DFT storage");
  CHECK(f->backend_state != NULL && f->executable != NULL,
        "checkpoint dump retired resident execution state");
  CHECK(accesses.prepare_rebuilds == 1,
        "checkpoint dump did not batch a storage-safe resident snapshot");
  CheckpointImage expected_checkpoint;
  {
    h5file stored(filename, h5file::READONLY, false, true);
    expected_checkpoint = CheckpointTransaction::read_manifest(stored, false);
  }
  f->mutation_generation[static_cast<int>(MutationKind::source_values)] = 1;

  const CheckpointFailurePoint dump_failures[] = {
      CheckpointFailurePoint::snapshot, CheckpointFailurePoint::write,
      CheckpointFailurePoint::rename_publish};
  for (CheckpointFailurePoint point : dump_failures) {
    BackendState *const state_before = f->backend_state;
    Executable *const executable_before = f->executable;
    checkpoint_set_failure_for_testing(point, 0);
    bool rejected = false;
    try { f->dump(filename, false); }
    catch (const std::runtime_error &) { rejected = true; }
    checkpoint_clear_failure_for_testing();
    bool target_complete = false;
    try {
      h5file existing(filename, h5file::READONLY, false, true);
      target_complete = CheckpointTransaction::has_manifest(existing);
    }
    catch (...) { target_complete = false; }
    CHECK(rejected && target_complete && access((std::string(filename) + ".tmp").c_str(), F_OK) != 0 &&
              access((std::string(filename) + ".bak").c_str(), F_OK) != 0 &&
              f->backend_state == state_before && f->executable == executable_before,
          "failed checkpoint dump changed the live epoch, target, or left a temporary file "
          "(point=%d rejected=%d complete=%d tmp=%d bak=%d)",
          int(point), int(rejected), int(target_complete),
          int(access((std::string(filename) + ".tmp").c_str(), F_OK) == 0),
          int(access((std::string(filename) + ".bak").c_str(), F_OK) == 0));
  }
  if (count_processors() > 1) {
    checkpoint_set_failure_for_testing(CheckpointFailurePoint::rename_backup, 0);
    checkpoint_set_secondary_failure_for_testing(CheckpointFailurePoint::rename_restore, 1);
    bool rejected = false;
    try { f->dump(filename, false); }
    catch (const std::runtime_error &) { rejected = true; }
    checkpoint_clear_failure_for_testing();
    const std::string backup = std::string(filename) + ".bak";
    const bool target_exists = access(filename, F_OK) == 0;
    const bool backup_exists = access(backup.c_str(), F_OK) == 0;
    bool authoritative_complete = false;
    try {
      const char *authoritative = target_exists ? filename : backup.c_str();
      h5file existing(authoritative, h5file::READONLY, false, true);
      authoritative_complete = CheckpointTransaction::has_manifest(existing);
    }
    catch (...) { authoritative_complete = false; }
    CHECK(sum_to_all(int(rejected)) == count_processors() && authoritative_complete &&
              access((std::string(filename) + ".tmp").c_str(), F_OK) != 0 &&
              (my_rank() != 1 || (!target_exists && backup_exists)),
          "asymmetric backup-restore failure lost prior authority or exposed a partial target");
    if (!target_exists && backup_exists) CHECK(std::rename(backup.c_str(), filename) == 0,
                                                "checkpoint test could not recover prior backup");
    all_wait();
  }
  const int prepares_before_load = accesses.prepare_rebuilds;

  if (count_processors() == 1) {
    int rank = 0;
    size_t count = 0, start = 0;
    std::vector<double> values;
    double original = 0.0;
    {
      h5file corrupt(filename, h5file::READWRITE, false, true);
      corrupt.read_size("backend_checkpoint_values", &rank, &count, 1);
      values.resize(count);
      corrupt.read_chunk(1, &start, &count, values.data());
      original = values[0];
      values[0] = original + 1.0;
      corrupt.write("backend_checkpoint_values", 1, &count, values.data(), false);
    }
    bool rejected = false;
    try { f->load(filename, false); }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected && f->backend_state && f->executable,
          "checkpoint hash corruption changed the live epoch");
    {
      values[0] = original;
      h5file restore(filename, h5file::READWRITE, false, true);
      restore.write("backend_checkpoint_values", 1, &count, values.data(), false);
    }
  }

  if (count_processors() == 1) {
    int rank = 0;
    size_t dims[2] = {0, 0}, start[2] = {0, 0};
    std::vector<double> rows;
    double original = 0.0;
    {
      h5file corrupt(filename, h5file::READWRITE, false, true);
      corrupt.read_size("backend_checkpoint_rows", &rank, dims, 2);
      rows.resize(dims[0] * dims[1]);
      corrupt.read_chunk(2, start, dims, rows.data());
      original = rows[7];
      rows[7] = 99.0;
      corrupt.write("backend_checkpoint_rows", 2, dims, rows.data(), false);
    }
    bool rejected = false;
    try { f->load(filename, false); }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected && f->backend_state && f->executable,
          "unsupported checkpoint precision changed the live epoch");
    {
      rows[7] = original;
      h5file restore(filename, h5file::READWRITE, false, true);
      restore.write("backend_checkpoint_rows", 2, dims, rows.data(), false);
    }
  }

  if (count_processors() == 1 &&
      std::numeric_limits<size_t>::max() > size_t(std::numeric_limits<int>::max())) {
    int rank = 0;
    size_t count = 0, start = 0;
    std::vector<size_t> header;
    {
      h5file corrupt(filename, h5file::READWRITE, false, true);
      corrupt.read_size("backend_checkpoint_header", &rank, &count, 1);
      header.resize(count);
      corrupt.read_chunk(1, &start, &count, header.data());
      header[9] = size_t(std::numeric_limits<int>::max()) + 1;
      corrupt.remove_data("backend_checkpoint_header");
      corrupt.create_data("backend_checkpoint_header", 1, &count, false, false);
      corrupt.write_chunk(1, &start, &count, header.data());
      corrupt.done_writing_chunks();
    }
    bool rejected = false;
    try { f->load(filename, false); }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected && f->backend_state && f->executable,
          "out-of-range disk timestep changed the live epoch");
    {
      header[9] = size_t(expected_checkpoint.timestep);
      h5file restore(filename, h5file::READWRITE, false, true);
      restore.remove_data("backend_checkpoint_header");
      restore.create_data("backend_checkpoint_header", 1, &count, false, false);
      restore.write_chunk(1, &start, &count, header.data());
      restore.done_writing_chunks();
    }
  }

  const CheckpointFailurePoint load_failures[] = {
      CheckpointFailurePoint::read, CheckpointFailurePoint::allocation,
      CheckpointFailurePoint::validation, CheckpointFailurePoint::precommit};
  for (CheckpointFailurePoint point : load_failures) {
    BackendState *const state_before = f->backend_state;
    Executable *const executable_before = f->executable;
    const int t_before = f->t;
    const uint32_t dirty_before = f->dirty_mask;
    checkpoint_set_failure_for_testing(point, 0);
    bool rejected = false;
    try { f->load(filename, false); }
    catch (const std::runtime_error &) { rejected = true; }
    checkpoint_clear_failure_for_testing();
    CHECK(rejected && f->backend_state == state_before && f->executable == executable_before &&
              f->t == t_before && f->dirty_mask == dirty_before,
          "failed staged checkpoint load changed the live resident epoch (point=%d rejected=%d "
          "state=%d executable=%d time=%d dirty=%u/%u)",
          int(point), int(rejected), int(f->backend_state == state_before),
          int(f->executable == executable_before), f->t - t_before, f->dirty_mask,
          dirty_before);
  }

  if (count_processors() == 1) {
    const uint64_t layout_generation =
        f->mutation_generation[static_cast<int>(MutationKind::field_layout)];
    f->mutation_generation[static_cast<int>(MutationKind::field_layout)] =
        std::numeric_limits<uint64_t>::max();
    bool generation_rejected = false;
    try { f->load(filename, false); }
    catch (const std::runtime_error &) { generation_rejected = true; }
    CHECK(generation_rejected && f->backend_state && f->executable,
          "checkpoint field-layout overflow changed the live epoch");
    f->mutation_generation[static_cast<int>(MutationKind::field_layout)] = layout_generation;
  }

  f->load(filename, false);
  CHECK(accesses.prepare_rebuilds == prepares_before_load + 1,
        "checkpoint load did not prepare resident authority for replacement");
  CHECK(f->backend_state == NULL && f->executable == NULL,
        "checkpoint load retained artifacts referring to the old catalog");
  CHECK(is_dirty(*f, dirty_storage), "checkpoint load did not invalidate storage layout");
  CHECK(f->mutation_generation[static_cast<int>(MutationKind::source_values)] ==
            checkpoint_generation,
        "checkpoint generation split/merge did not preserve a value above 2^53");
  const CheckpointImage restored_checkpoint = CheckpointTransaction::snapshot(*f);
  CHECK(checkpoint_payload_equal(expected_checkpoint, restored_checkpoint),
        "checkpoint did not exactly restore field, material, DFT, alias, or legacy flux rows");
  std::remove(filename);

  delete f;
  delete s;
}

static void test_checkpoint_metadata_boundaries() {
  CHECK(checkpoint_decode_signed_for_testing(checkpoint_encode_signed_for_testing(
            std::numeric_limits<int>::min())) == std::numeric_limits<int>::min() &&
            checkpoint_decode_signed_for_testing(checkpoint_encode_signed_for_testing(-1)) == -1 &&
            checkpoint_decode_signed_for_testing(checkpoint_encode_signed_for_testing(0)) == 0 &&
            checkpoint_decode_signed_for_testing(checkpoint_encode_signed_for_testing(
                std::numeric_limits<int>::max())) == std::numeric_limits<int>::max(),
        "checkpoint signed-key encoding did not preserve int bounds");
  bool wide_rejected = false;
  try { (void)checkpoint_decode_signed_for_testing(std::numeric_limits<uint64_t>::max()); }
  catch (const std::invalid_argument &) { wide_rejected = true; }
  CHECK(wide_rejected, "checkpoint signed-key decoder accepted an out-of-domain disk integer");
}

static void test_checkpoint_susceptibility_identity() {
  lorentzian_susceptibility lorentz_a(realnum(1.125), realnum(0.0625), false);
  lorentzian_susceptibility lorentz_b(realnum(1.25), realnum(0.0625), false);
  lorentzian_susceptibility lorentz_gamma(realnum(1.125), realnum(0.125), false);
  lorentzian_susceptibility drude(realnum(1.125), realnum(0.0625), true);
  CHECK(susceptibility_chain_signature(&lorentz_a) !=
            susceptibility_chain_signature(&lorentz_b) &&
            susceptibility_chain_signature(&lorentz_a) !=
                susceptibility_chain_signature(&lorentz_gamma) &&
            susceptibility_chain_signature(&lorentz_a) !=
                susceptibility_chain_signature(&drude),
        "checkpoint susceptibility identity omitted Lorentzian dynamics");

  noisy_lorentzian_susceptibility noisy_a(realnum(0.03125), realnum(1.125),
                                           realnum(0.0625));
  noisy_lorentzian_susceptibility noisy_b(realnum(0.046875), realnum(1.125),
                                           realnum(0.0625));
  CHECK(susceptibility_chain_signature(&noisy_a) !=
            susceptibility_chain_signature(&noisy_b),
        "checkpoint susceptibility identity omitted noisy amplitude");

  gyrotropic_susceptibility gyro_a(vec(0.125, -0.25, 0.375), realnum(1.125),
                                    realnum(0.0625), realnum(0.015625),
                                    GYROTROPIC_LORENTZIAN);
  gyrotropic_susceptibility gyro_b(vec(0.125, -0.25, 0.5), realnum(1.125),
                                    realnum(0.0625), realnum(0.015625),
                                    GYROTROPIC_LORENTZIAN);
  gyrotropic_susceptibility gyro_c(vec(0.125, -0.25, 0.375), realnum(1.125),
                                    realnum(0.0625), realnum(0.03125), GYROTROPIC_DRUDE);
  CHECK(susceptibility_chain_signature(&gyro_a) != susceptibility_chain_signature(&gyro_b) &&
            susceptibility_chain_signature(&gyro_a) != susceptibility_chain_signature(&gyro_c),
        "checkpoint susceptibility identity omitted gyrotropic bias/model/alpha dynamics");

  const realnum gamma_matrix[] = {realnum(0.0), realnum(0.125), realnum(0.25), realnum(0.0)};
  const realnum populations[] = {realnum(1.0), realnum(0.0)};
  const realnum alpha[] = {realnum(-0.5), realnum(0.5)};
  const realnum omega[] = {realnum(0.75)};
  const realnum transition_gamma[] = {realnum(0.03125)};
  const realnum sigmat[] = {realnum(1.0), realnum(0.5), realnum(0.25), realnum(0.125),
                            realnum(0.0625)};
  multilevel_susceptibility multilevel_a(2, 1, gamma_matrix, populations, alpha, omega,
                                          transition_gamma, sigmat);
  multilevel_susceptibility multilevel_same(2, 1, gamma_matrix, populations, alpha, omega,
                                             transition_gamma, sigmat);
  CHECK(susceptibility_chain_signature(&multilevel_a) ==
            susceptibility_chain_signature(&multilevel_same),
        "checkpoint multilevel identity is not exact for equal definitions");
  realnum changed_gamma_matrix[4];
  memcpy(changed_gamma_matrix, gamma_matrix, sizeof(gamma_matrix));
  changed_gamma_matrix[1] += realnum(0.03125);
  multilevel_susceptibility multilevel_changed(2, 1, changed_gamma_matrix, populations, alpha,
                                                omega, transition_gamma, sigmat);
  CHECK(susceptibility_chain_signature(&multilevel_a) !=
            susceptibility_chain_signature(&multilevel_changed),
        "checkpoint multilevel identity omitted relaxation rates");
  realnum changed_populations[2];
  memcpy(changed_populations, populations, sizeof(populations));
  changed_populations[0] -= realnum(0.125);
  multilevel_susceptibility multilevel_population(2, 1, gamma_matrix, changed_populations, alpha,
                                                   omega, transition_gamma, sigmat);
  realnum changed_alpha[2];
  memcpy(changed_alpha, alpha, sizeof(alpha));
  changed_alpha[0] -= realnum(0.125);
  multilevel_susceptibility multilevel_alpha(2, 1, gamma_matrix, populations, changed_alpha,
                                              omega, transition_gamma, sigmat);
  realnum changed_omega[1] = {omega[0] + realnum(0.125)};
  multilevel_susceptibility multilevel_omega(2, 1, gamma_matrix, populations, alpha,
                                              changed_omega, transition_gamma, sigmat);
  realnum changed_transition_gamma[1] = {transition_gamma[0] + realnum(0.015625)};
  multilevel_susceptibility multilevel_loss(2, 1, gamma_matrix, populations, alpha, omega,
                                             changed_transition_gamma, sigmat);
  realnum changed_sigmat[5];
  memcpy(changed_sigmat, sigmat, sizeof(sigmat));
  changed_sigmat[4] += realnum(0.03125);
  multilevel_susceptibility multilevel_sigma(2, 1, gamma_matrix, populations, alpha, omega,
                                              transition_gamma, changed_sigmat);
  const uint64_t multilevel_signature = susceptibility_chain_signature(&multilevel_a);
  CHECK(multilevel_signature != susceptibility_chain_signature(&multilevel_population) &&
            multilevel_signature != susceptibility_chain_signature(&multilevel_alpha) &&
            multilevel_signature != susceptibility_chain_signature(&multilevel_omega) &&
            multilevel_signature != susceptibility_chain_signature(&multilevel_loss) &&
            multilevel_signature != susceptibility_chain_signature(&multilevel_sigma),
        "checkpoint multilevel identity omitted an evolution-affecting parameter family");

  susceptibility *ordered = new lorentzian_susceptibility(realnum(0.75), realnum(0.0625));
  ordered->next = new noisy_lorentzian_susceptibility(realnum(0.03125), realnum(1.0),
                                                       realnum(0.125));
  susceptibility *reversed = new noisy_lorentzian_susceptibility(
      realnum(0.03125), realnum(1.0), realnum(0.125));
  reversed->next = new lorentzian_susceptibility(realnum(0.75), realnum(0.0625));
  CHECK(susceptibility_chain_signature(ordered) != susceptibility_chain_signature(reversed),
        "checkpoint susceptibility identity omitted chain order");
  delete ordered;
  delete reversed;

  susceptibility unsupported;
  bool custom_rejected = false;
  try { (void)susceptibility_chain_signature(&unsupported); }
  catch (const std::invalid_argument &) { custom_rejected = true; }
  CHECK(custom_rejected, "checkpoint susceptibility identity accepted a custom chain");

  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure structure_a(gv, unit_epsilon, no_pml(), identity(), 1);
  structure structure_b(gv, unit_epsilon, no_pml(), identity(), 1);
  structure_a.add_susceptibility(unit_epsilon, E_stuff, lorentz_a);
  structure_b.add_susceptibility(unit_epsilon, E_stuff, lorentz_b);
  fields fields_a(&structure_a), fields_b(&structure_b);
  const size_t signature_a =
      size_t(CheckpointTransaction::snapshot(fields_a).material_recipe_signature);
  const size_t signature_b =
      size_t(CheckpointTransaction::snapshot(fields_b).material_recipe_signature);
  CHECK(sum_to_all(signature_a) != sum_to_all(signature_b),
        "checkpoint material signature omitted same-shape susceptibility parameter changes");

  if (sizeof(realnum) == sizeof(double)) {
    const auto adjacent = [](realnum value) {
      return realnum(std::nextafter(double(value), std::numeric_limits<double>::infinity()));
    };
    const auto exact_only = [](const susceptibility &baseline, const susceptibility &changed,
                               const char *what) {
      CHECK(susceptibility_chain_native_signature(&baseline) !=
                    susceptibility_chain_native_signature(&changed) &&
                susceptibility_chain_signature(&baseline) ==
                    susceptibility_chain_signature(&changed),
            "checkpoint susceptibility identity did not preserve exact %s bits", what);
    };

    lorentzian_susceptibility lorentz_adjacent_omega(adjacent(realnum(1.125)),
                                                      realnum(0.0625), false);
    lorentzian_susceptibility lorentz_adjacent_gamma(realnum(1.125),
                                                      adjacent(realnum(0.0625)), false);
    exact_only(lorentz_a, lorentz_adjacent_omega, "Lorentzian omega");
    exact_only(lorentz_a, lorentz_adjacent_gamma, "Lorentzian gamma");
    noisy_lorentzian_susceptibility noisy_adjacent(adjacent(realnum(0.03125)),
                                                    realnum(1.125), realnum(0.0625));
    exact_only(noisy_a, noisy_adjacent, "noisy amplitude");

    const realnum gyro_base[6] = {realnum(0.125), realnum(-0.25), realnum(0.375),
                                  realnum(1.125), realnum(0.0625), realnum(0.015625)};
    for (int parameter = 0; parameter < 6; ++parameter) {
      realnum changed[6];
      memcpy(changed, gyro_base, sizeof(changed));
      changed[parameter] = adjacent(changed[parameter]);
      gyrotropic_susceptibility baseline(
          vec(gyro_base[0], gyro_base[1], gyro_base[2]), gyro_base[3], gyro_base[4],
          gyro_base[5], GYROTROPIC_LORENTZIAN);
      gyrotropic_susceptibility modified(
          vec(changed[0], changed[1], changed[2]), changed[3], changed[4], changed[5],
          GYROTROPIC_LORENTZIAN);
      exact_only(baseline, modified, "gyrotropic numeric parameter");
    }

    for (int family = 0; family < 6; ++family) {
      realnum gamma_changed[4], populations_changed[2], alpha_changed[2], omega_changed[1],
          transition_gamma_changed[1], sigmat_changed[5];
      memcpy(gamma_changed, gamma_matrix, sizeof(gamma_changed));
      memcpy(populations_changed, populations, sizeof(populations_changed));
      memcpy(alpha_changed, alpha, sizeof(alpha_changed));
      memcpy(omega_changed, omega, sizeof(omega_changed));
      memcpy(transition_gamma_changed, transition_gamma, sizeof(transition_gamma_changed));
      memcpy(sigmat_changed, sigmat, sizeof(sigmat_changed));
      realnum *families[] = {gamma_changed, populations_changed, alpha_changed, omega_changed,
                             transition_gamma_changed, sigmat_changed};
      families[family][0] = adjacent(families[family][0]);
      multilevel_susceptibility modified(
          2, 1, gamma_changed, populations_changed, alpha_changed, omega_changed,
          transition_gamma_changed, sigmat_changed);
      exact_only(multilevel_a, modified, "multilevel numeric parameter family");
    }

    checkpoint_identity_epsilon_value = realnum(2.0);
    structure material(gv, checkpoint_identity_epsilon, no_pml(), identity(), 1);
    material.set_chi3(checkpoint_identity_epsilon);
    fields material_fields(&material);
    material_fields.require_component(Ez);
    gaussian_src_time material_source(0.25, 0.0625);
    material_fields.add_point_source(Ez, material_source, vec(0.0, 0.0));
    const CheckpointImage material_image_a = CheckpointTransaction::snapshot(material_fields);
    bool changed_material_sample = false;
    for (int chunk = 0; chunk < material_fields.num_chunks && !changed_material_sample; ++chunk) {
      structure_chunk *sc = material_fields.chunks[chunk]->s;
      if (!material_fields.chunks[chunk]->is_mine()) continue;
      realnum *row = sc->chi3[int(Ez)];
      if (!row) continue;
      for (int i = 0; i < sc->gv.ntot(); ++i) {
        if (!sc->gv.owns(sc->gv.iloc(Ez, i))) continue;
        row[i] = adjacent(row[i]);
        changed_material_sample = true;
        break;
      }
    }
    const CheckpointImage material_image_b = CheckpointTransaction::snapshot(material_fields);
    CHECK(or_to_all(changed_material_sample),
          "checkpoint material identity fixture found no owned material sample");
    const size_t native_a = sum_to_all(size_t(material_image_a.material_native_signature));
    const size_t native_b = sum_to_all(size_t(material_image_b.material_native_signature));
    const size_t portable_a = sum_to_all(size_t(material_image_a.material_recipe_signature));
    const size_t portable_b = sum_to_all(size_t(material_image_b.material_recipe_signature));
    CHECK(native_a != native_b && portable_a == portable_b,
          "checkpoint material identity collapsed adjacent binary64 samples");
    checkpoint_identity_epsilon_value = realnum(1.0);
  }
}

static void test_checkpoint_dft_identity_precision_and_order() {
  if (sizeof(realnum) != sizeof(double)) return;
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, unit_epsilon, no_pml(), identity(), 1);
  fields f(&s);
  f.use_real_fields();
  component monitored = Ez;
  f.require_component(monitored);
  const double frequency = 0.25;
  dft_fields monitor = f.add_dft_fields(&monitored, 1, f.v, &frequency, 1);
  (void)monitor;
  const CheckpointImage baseline = CheckpointTransaction::snapshot(f);
  const auto reduced = [](uint64_t signature) {
    return uint64_t(sum_to_all(size_t(signature))) & ((UINT64_C(1) << 52) - 1);
  };
  const auto verify_exact_only = [&](const CheckpointImage &changed, const char *what) {
    CHECK(reduced(baseline.dft_native_signature) != reduced(changed.dft_native_signature) &&
              reduced(baseline.dft_recipe_signature) == reduced(changed.dft_recipe_signature),
          "checkpoint DFT identity did not preserve exact %s bits", what);
  };
  dft_chunk *local = NULL;
  for (int chunk = 0; chunk < f.num_chunks && !local; ++chunk)
    if (f.chunks[chunk]->is_mine()) local = f.chunks[chunk]->dft_chunks;

  const double original_omega = local ? local->omega[0] : 0.0;
  if (local) local->omega[0] = std::nextafter(local->omega[0], INFINITY);
  verify_exact_only(CheckpointTransaction::snapshot(f), "frequency");
  if (local) local->omega[0] = original_omega;

  const std::complex<double> original_scale = local ? local->scale : std::complex<double>();
  if (local)
    local->scale = std::complex<double>(std::nextafter(local->scale.real(), INFINITY),
                                        local->scale.imag());
  verify_exact_only(CheckpointTransaction::snapshot(f), "scale");
  if (local) local->scale = original_scale;

  double original_weight = 0.0;
  if (local) {
    original_weight = local->s0.in_direction(X);
    local->s0.set_direction(X, std::nextafter(original_weight, INFINITY));
  }
  verify_exact_only(CheckpointTransaction::snapshot(f), "boundary weight");
  if (local) local->s0.set_direction(X, original_weight);

  const double frequency_a = 0.1875, frequency_b = 0.3125;
  structure ordered_structure(gv, unit_epsilon, no_pml(), identity(), 1);
  fields ordered(&ordered_structure);
  ordered.use_real_fields();
  ordered.require_component(monitored);
  dft_fields ordered_a = ordered.add_dft_fields(&monitored, 1, ordered.v, &frequency_a, 1);
  dft_fields ordered_b = ordered.add_dft_fields(&monitored, 1, ordered.v, &frequency_b, 1);
  (void)ordered_a;
  (void)ordered_b;
  structure swapped_structure(gv, unit_epsilon, no_pml(), identity(), 1);
  fields swapped(&swapped_structure);
  swapped.use_real_fields();
  swapped.require_component(monitored);
  dft_fields swapped_b = swapped.add_dft_fields(&monitored, 1, swapped.v, &frequency_b, 1);
  dft_fields swapped_a = swapped.add_dft_fields(&monitored, 1, swapped.v, &frequency_a, 1);
  (void)swapped_a;
  (void)swapped_b;
  const CheckpointImage ordered_image = CheckpointTransaction::snapshot(ordered);
  const CheckpointImage swapped_image = CheckpointTransaction::snapshot(swapped);
  CHECK(reduced(ordered_image.dft_native_signature) !=
                reduced(swapped_image.dft_native_signature) &&
            reduced(ordered_image.dft_recipe_signature) !=
                reduced(swapped_image.dft_recipe_signature),
        "checkpoint DFT identity omitted equal-shape monitor ordering");
}

static bool checkpoint_payload_equal(const CheckpointImage &expected,
                                     const CheckpointImage &actual) {
  if (expected.rows.size() != actual.rows.size() ||
      expected.legacy_flux_signatures != actual.legacy_flux_signatures ||
      expected.legacy_flux_values != actual.legacy_flux_values ||
      !same_seed_snapshot(expected.random_seed, actual.random_seed))
    return false;
  for (const CheckpointRow &row : expected.rows) {
    const CheckpointRow *match = NULL;
    for (const CheckpointRow &candidate : actual.rows)
      if (candidate.key == row.key) {
        if (match) return false;
        match = &candidate;
      }
    if (!match || match->spec.element_type != row.spec.element_type ||
        match->spec.elements != row.spec.elements || match->has_alias != row.has_alias ||
        (row.has_alias && !(match->alias_key == row.alias_key)) || match->values != row.values)
      return false;
  }
  return true;
}

static void test_checkpoint_polarization_dft_roundtrip() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, unit_epsilon, pml(0.25), identity(), 2);
  add_multilevel_lifecycle_states(s);
  fields f(&s);
  f.use_real_fields();
  f.require_component(Ez);
  f.require_component(Hz);
  gaussian_src_time source(0.31, 0.09);
  f.add_point_source(Ez, source, vec(0.13, -0.17));
  component monitored = Ez;
  const double frequencies[2] = {0.21, 0.37};
  dft_fields monitor = f.add_dft_fields(&monitored, 1, f.v, frequencies, 2);
  (void)monitor;
  f.add_flux_vol(X, volume(vec(0.1, -0.7), vec(0.1, 0.7)));
  set_random_seed(41000 + my_global_rank());
  f.advance(3);

  const CheckpointImage expected = CheckpointTransaction::snapshot(f);
  const RandomSeedSnapshot expected_seed = random_seed_snapshot();
  size_t polarization_rows = 0, dft_rows = 0, aliases = 0;
  bool saw_gamma_inv = false;
  for (const CheckpointRow &row : expected.rows) {
    if (row.key.kind == int(array_kind::polarization_internal)) {
      ++polarization_rows;
      saw_gamma_inv |= row.key.component_ == int(Centered) && row.key.cmp == -1 &&
                       row.spec.elements == 9;
    }
    if (row.key.kind == int(array_kind::dft) || row.key.kind == int(array_kind::dft_phase))
      ++dft_rows;
    aliases += row.has_alias;
  }
  CHECK(or_to_all(polarization_rows > 0) && or_to_all(dft_rows > 0) &&
            or_to_all(saw_gamma_inv),
        "checkpoint fixture omitted state rows (polarization=%zu DFT=%zu aliases=%zu GammaInv=%d)",
        polarization_rows, dft_rows, aliases, int(saw_gamma_inv));

  char filename[160];
  snprintf(filename, sizeof(filename), "/tmp/meep-checkpoint-state-%ld-%d.h5", long(getpid()),
           my_rank());
  f.dump(filename, false);
  f.advance(2);
  const vec continuation_probes[] = {vec(-0.45, -0.35), vec(0.15, 0.25), vec(0.55, -0.15)};
  std::complex<double> expected_continuation[3];
  for (size_t i = 0; i < 3; ++i)
    expected_continuation[i] = f.get_field(Ez, continuation_probes[i]);
  CHECK(f.phase_in_material(&s, f.dt) == 1,
        "checkpoint continuation fixture did not enter material mutation");
  f.advance(1);
  set_random_seed(42000 + my_global_rank());
  f.load(filename, false);
  const CheckpointImage restored = CheckpointTransaction::snapshot(f);
  bool rebound = true;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    FOR_FIELD_TYPES(ft) {
      const polarization_state *state = f.chunks[chunk]->pol[ft];
      const susceptibility *sus = f.chunks[chunk]->s->chiP[ft];
      while (state && sus) {
        rebound &= state->s == sus;
        state = state->next;
        sus = sus->next;
      }
      rebound &= !state && !sus;
    }
  CHECK(f.t == expected.timestep && checkpoint_payload_equal(expected, restored) &&
            same_seed_snapshot(random_seed_snapshot(), expected_seed) && rebound,
        "checkpoint did not exactly restore multilevel/GammaInv/DFT/alias state");
  f.advance(2);
  bool continued = true;
  for (size_t i = 0; i < 3; ++i)
    continued &= std::abs(f.get_field(Ez, continuation_probes[i]) -
                          expected_continuation[i]) < 1e-10;
  CHECK(continued, "checkpoint did not continue after a cloned material structure");
  std::remove(filename);

  structure alias_structure(gv, unit_epsilon, no_pml(), identity(), 1);
  fields alias_fields(&alias_structure);
  alias_fields.use_real_fields();
  alias_fields.require_component(Hx);
  const CheckpointImage expected_alias = CheckpointTransaction::snapshot(alias_fields);
  size_t alias_rows = 0;
  for (const CheckpointRow &row : expected_alias.rows) alias_rows += row.has_alias;
  snprintf(filename, sizeof(filename), "/tmp/meep-checkpoint-alias-%ld-%d.h5", long(getpid()),
           my_rank());
  alias_fields.dump(filename, false);
  alias_fields.load(filename, false);
  const CheckpointImage restored_alias = CheckpointTransaction::snapshot(alias_fields);
  CHECK(or_to_all(alias_rows > 0) && checkpoint_payload_equal(expected_alias, restored_alias),
        "checkpoint did not preserve canonical H/B alias ownership");
  CheckpointImage cyclic_alias = expected_alias;
  for (CheckpointRow &row : cyclic_alias.rows)
    if (row.has_alias) {
      row.alias_key = row.key;
      row.checksum = checkpoint_row_checksum(row);
      break;
    }
  bool cycle_rejected = false;
  try { CheckpointTransaction::validate_target(alias_fields, cyclic_alias); }
  catch (const std::runtime_error &) { cycle_rejected = true; }
  CHECK(cycle_rejected, "checkpoint accepted a cyclic alias graph");
  CheckpointImage dangling_alias = expected_alias;
  for (CheckpointRow &row : dangling_alias.rows)
    if (row.has_alias) {
      row.alias_key.chunk = std::numeric_limits<int>::max();
      row.checksum = checkpoint_row_checksum(row);
      break;
    }
  bool dangling_rejected = false;
  try { CheckpointTransaction::validate_target(alias_fields, dangling_alias); }
  catch (const std::runtime_error &) { dangling_rejected = true; }
  CHECK(dangling_rejected, "checkpoint accepted a dangling alias target");

  const auto rejected_metadata = [&](CheckpointImage malformed, const char *what) {
    bool rejected = false;
    try { CheckpointTransaction::validate_target(alias_fields, malformed); }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected, "checkpoint accepted checksum-consistent malformed %s", what);
  };
  for (int kind = 0; kind <= int(array_kind::num_kinds); ++kind) {
    CheckpointImage malformed = expected_alias;
    if (!malformed.rows.empty()) {
      CheckpointRow &row = malformed.rows.front();
      row.key.kind = kind;
      row.key.component_ = -1;
      row.key.cmp = -1;
      row.key.aux = std::numeric_limits<uint64_t>::max();
      row.checksum = checkpoint_row_checksum(row);
    }
    rejected_metadata(malformed, "array-kind key domain");
  }
  CheckpointImage malformed_component = expected_alias;
  if (!malformed_component.rows.empty()) {
    malformed_component.rows.front().key.component_ = NUM_FIELD_COMPONENTS;
    malformed_component.rows.front().checksum =
        checkpoint_row_checksum(malformed_component.rows.front());
  }
  rejected_metadata(malformed_component, "component");
  CheckpointImage malformed_cmp = expected_alias;
  if (!malformed_cmp.rows.empty()) {
    malformed_cmp.rows.front().key.cmp = 2;
    malformed_cmp.rows.front().checksum = checkpoint_row_checksum(malformed_cmp.rows.front());
  }
  rejected_metadata(malformed_cmp, "complex lane");
  CheckpointImage malformed_aux = expected_alias;
  if (!malformed_aux.rows.empty()) {
    malformed_aux.rows.front().key.aux = 1;
    malformed_aux.rows.front().checksum = checkpoint_row_checksum(malformed_aux.rows.front());
  }
  rejected_metadata(malformed_aux, "auxiliary key");
  CheckpointImage malformed_timestep = expected_alias;
  malformed_timestep.timestep = -1;
  rejected_metadata(malformed_timestep, "timestep");
  CheckpointImage wrong_configuration = expected_alias;
  wrong_configuration.configuration_signature ^= 1;
  bool configuration_rejected = false;
  try { CheckpointTransaction::validate_target(alias_fields, wrong_configuration); }
  catch (const std::runtime_error &) { configuration_rejected = true; }
  CHECK(configuration_rejected,
        "checkpoint accepted mismatched global grid/boundary/symmetry metadata");
  CheckpointImage unsupported_host_precision = expected_alias;
  unsupported_host_precision.host_realnum_bytes = 16;
  bool host_precision_rejected = false;
  try { CheckpointTransaction::validate_target(alias_fields, unsupported_host_precision); }
  catch (const std::runtime_error &) { host_precision_rejected = true; }
  CHECK(host_precision_rejected, "checkpoint accepted an unsupported host precision");
  if (count_processors() == 1) {
    CheckpointImage opposite_precision = expected_alias;
    const bool native_f32 = sizeof(realnum) == sizeof(float);
    opposite_precision.host_realnum_bytes = native_f32 ? 8 : 4;
    for (CheckpointRow &row : opposite_precision.rows) {
      row.spec.storage = native_f32 ? Precision::f64 : Precision::f32;
      row.spec.alignment = native_f32 ? alignof(double) : alignof(float);
      if (!native_f32)
        for (double &value : row.values)
          if (std::isfinite(value)) value = double(float(value));
      row.checksum = checkpoint_row_checksum(row);
    }
    bool opposite_precision_accepted = true;
    try { CheckpointTransaction::validate_target(alias_fields, opposite_precision); }
    catch (const std::runtime_error &) { opposite_precision_accepted = false; }
    CHECK(opposite_precision_accepted,
          "checkpoint rejected a format-correct opposite-precision row layout");
    CheckpointImage wrong_saved_alignment = opposite_precision;
    if (!wrong_saved_alignment.rows.empty()) {
      ++wrong_saved_alignment.rows.front().spec.alignment;
      wrong_saved_alignment.rows.front().checksum =
          checkpoint_row_checksum(wrong_saved_alignment.rows.front());
      bool saved_alignment_rejected = false;
      try { CheckpointTransaction::validate_target(alias_fields, wrong_saved_alignment); }
      catch (const std::runtime_error &) { saved_alignment_rejected = true; }
      CHECK(saved_alignment_rejected,
            "checkpoint accepted alignment inconsistent with the saved precision format");
    }
    CheckpointImage malformed_f32 = expected_alias;
    malformed_f32.host_realnum_bytes = 4;
    bool injected_noncanonical_f32 = false;
    for (CheckpointRow &row : malformed_f32.rows) {
      row.spec.storage = Precision::f32;
      row.spec.alignment = alignof(float);
      if (!row.has_alias && !row.values.empty() && !injected_noncanonical_f32) {
        row.values.front() = 0.1;
        injected_noncanonical_f32 = true;
      }
      else
        for (double &value : row.values)
          if (std::isfinite(value)) value = double(float(value));
      row.checksum = checkpoint_row_checksum(row);
    }
    bool malformed_f32_rejected = false;
    try { CheckpointTransaction::validate_target(alias_fields, malformed_f32); }
    catch (const std::runtime_error &) { malformed_f32_rejected = true; }
    CHECK(injected_noncanonical_f32 && malformed_f32_rejected,
          "checkpoint accepted a noncanonical binary32 payload");
    if (native_f32) {
      CheckpointImage overflowing_f64 = opposite_precision;
      bool injected_overflow = false;
      for (CheckpointRow &row : overflowing_f64.rows) {
        if (!row.has_alias && !row.values.empty() && !injected_overflow) {
          row.values.front() = std::numeric_limits<double>::max();
          injected_overflow = true;
        }
        row.checksum = checkpoint_row_checksum(row);
      }
      bool overflow_rejected = false;
      try { CheckpointTransaction::validate_target(alias_fields, overflowing_f64); }
      catch (const std::runtime_error &) { overflow_rejected = true; }
      CHECK(injected_overflow && overflow_rejected,
            "checkpoint accepted an f64 payload outside binary32 range");
    }
  }
  if (count_processors() == 1) {
    CheckpointImage wrong_source_recipe = expected_alias;
    wrong_source_recipe.source_definition_signature ^= 1;
    bool source_recipe_rejected = false;
    try { CheckpointTransaction::validate_target(alias_fields, wrong_source_recipe); }
    catch (const std::runtime_error &) { source_recipe_rejected = true; }
    CHECK(source_recipe_rejected, "checkpoint accepted a changed source recipe fingerprint");

    CheckpointImage wrong_dft_recipe = expected_alias;
    wrong_dft_recipe.dft_native_signature ^= 1;
    bool dft_recipe_rejected = false;
    try { CheckpointTransaction::validate_target(alias_fields, wrong_dft_recipe); }
    catch (const std::runtime_error &) { dft_recipe_rejected = true; }
    CHECK(dft_recipe_rejected,
          "checkpoint accepted a changed same-precision DFT recipe fingerprint");

    CheckpointImage ambiguous_seed = expected_alias;
    ambiguous_seed.saved_rank_count = 2;
    ambiguous_seed.random_seed_ranks = {0, 1};
    ambiguous_seed.random_seeds = {expected_alias.random_seed, expected_alias.random_seed};
    ++ambiguous_seed.random_seeds[1].semantic_seed;
    bool seed_remap_rejected = false;
    try { CheckpointTransaction::validate_target(alias_fields, ambiguous_seed); }
    catch (const std::runtime_error &) { seed_remap_rejected = true; }
    CHECK(seed_remap_rejected, "checkpoint accepted an ambiguous noisy rank-count remap");
  }
  std::remove(filename);
}

static void test_checkpoint_noisy_rank_seed_continuation() {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure reference_structure(gv, unit_epsilon, no_pml(), identity(), 2);
  structure resumed_structure(gv, unit_epsilon, no_pml(), identity(), 2);
  noisy_lorentzian_susceptibility noisy(0.03125, 1.1, 0.05);
  reference_structure.add_susceptibility(unit_epsilon, E_stuff, noisy);
  resumed_structure.add_susceptibility(unit_epsilon, E_stuff, noisy);
  fields reference(&reference_structure);
  fields resumed(&resumed_structure);
  gaussian_src_time source(0.31, 0.09);
  reference.add_point_source(Ez, source, vec(0.13, -0.17));
  resumed.add_point_source(Ez, source, vec(0.13, -0.17));
  lifetime_counts reference_counts, resumed_counts;
  reference.backend = new tracking_backend(reference, reference_counts);
  resumed.backend = new tracking_backend(resumed, resumed_counts);

  const unsigned long seed = 51000 + unsigned(my_global_rank());
  set_random_seed(seed);
  reference.advance(5);
  const vec probes[] = {vec(-0.6, -0.4), vec(0.1, 0.2), vec(0.7, -0.3)};
  std::complex<double> expected[3];
  for (size_t i = 0; i < 3; ++i) expected[i] = reference.get_field(Ez, probes[i]);

  set_random_seed(seed);
  resumed.advance(3);
  const RandomSeedSnapshot saved_seed = random_seed_snapshot();
  char filename[160];
  snprintf(filename, sizeof(filename), "/tmp/meep-checkpoint-noisy-%ld-%d.h5", long(getpid()),
           my_rank());
  resumed.dump(filename, false);
  set_random_seed(52000 + unsigned(my_global_rank()));
  resumed.load(filename, false);
  CHECK(same_seed_snapshot(random_seed_snapshot(), saved_seed),
        "checkpoint did not restore the rank-local noisy seed snapshot");
  resumed.advance(2);
  bool continued = true;
  for (size_t i = 0; i < 3; ++i)
    continued &= std::abs(resumed.get_field(Ez, probes[i]) - expected[i]) < 1e-10;
  CHECK(continued, "checkpoint did not preserve noisy rank-local continuation");
  std::remove(filename);
}

static void test_checkpoint_single_rank_repartition() {
  if (count_processors() != 1) return;
  const grid_volume gv = vol2d(3.0, 2.0, 8.0);
  structure source_structure(gv, eps_slab, no_pml(), identity(), 3);
  structure target_structure(gv, eps_slab, no_pml(), identity(), 1);
  fields source(&source_structure);
  fields target(&target_structure);
  source.use_real_fields();
  target.use_real_fields();
  source.require_component(Hx);
  target.require_component(Hx);
  gaussian_src_time source_time(0.27, 0.08);
  const vec source_point(0.17, -0.21);
  source.add_point_source(Ez, source_time, source_point);
  target.add_point_source(Ez, source_time, source_point);
  source.advance(3);
  target.advance(1);
  CHECK(source.num_chunks != target.num_chunks,
        "checkpoint repartition fixture did not select distinct chunk layouts");
  const CheckpointImage source_image = CheckpointTransaction::snapshot(source);
  const CheckpointImage target_image = CheckpointTransaction::snapshot(target);
  CHECK(source_image.storage_signature != target_image.storage_signature,
        "checkpoint repartition fixture unexpectedly had an identical storage signature");

  CheckpointImage missing_persistent = source_image;
  for (std::vector<CheckpointRow>::iterator row = missing_persistent.rows.begin();
       row != missing_persistent.rows.end(); ++row)
    if (!row->has_alias && row->key.kind == int(array_kind::f)) {
      missing_persistent.rows.erase(row);
      break;
    }
  BackendState *const state_before_reject = target.backend_state;
  Executable *const executable_before_reject = target.executable;
  const int time_before_reject = target.t;
  bool missing_rejected = false;
  try {
    CheckpointTransaction::validate_target(target, missing_persistent);
    CheckpointTransaction::commit(target, missing_persistent);
  }
  catch (const std::runtime_error &) { missing_rejected = true; }
  CHECK(missing_rejected && target.backend_state == state_before_reject &&
            target.executable == executable_before_reject && target.t == time_before_reject,
        "checkpoint repartition accepted a missing persistent physical row");

  CheckpointImage extra_persistent = source_image;
  for (const CheckpointRow &row : source_image.rows)
    if (!row.has_alias && row.key.kind == int(array_kind::f)) {
      CheckpointRow extra = row;
      extra.key.aux ^= UINT64_C(1) << 63;
      extra.checksum = checkpoint_row_checksum(extra);
      extra_persistent.rows.push_back(extra);
      break;
    }
  bool extra_rejected = false;
  try {
    CheckpointTransaction::validate_target(target, extra_persistent);
    CheckpointTransaction::commit(target, extra_persistent);
  }
  catch (const std::runtime_error &) { extra_rejected = true; }
  CHECK(extra_rejected && target.t == time_before_reject,
        "checkpoint silently discarded an unmatched saved persistent row");

  CheckpointImage trailing_persistent = source_image;
  for (CheckpointRow &row : trailing_persistent.rows)
    if (!row.has_alias && row.key.kind == int(array_kind::f)) {
      ++row.spec.elements;
      row.values.push_back(0.0);
      row.checksum = checkpoint_row_checksum(row);
      break;
    }
  bool trailing_rejected = false;
  try {
    CheckpointTransaction::validate_target(target, trailing_persistent);
    CheckpointTransaction::commit(target, trailing_persistent);
  }
  catch (const std::runtime_error &) { trailing_rejected = true; }
  CHECK(trailing_rejected && target.t == time_before_reject,
        "checkpoint repartition accepted a trailing spatial scalar");

  CheckpointImage nonintersecting = source_image;
  for (const CheckpointRow &row : source_image.rows)
    if (!row.has_alias && row.key.kind == int(array_kind::f)) {
      CheckpointRow extra = row;
      extra.key.chunk += source.num_chunks + 100;
      for (int axis = 0; axis < 3; ++axis) {
        extra.little_corner[axis] += 100000;
        extra.big_corner[axis] += 100000;
      }
      extra.checksum = checkpoint_row_checksum(extra);
      nonintersecting.rows.push_back(extra);
      break;
    }
  bool nonintersecting_rejected = false;
  try {
    CheckpointTransaction::validate_target(target, nonintersecting);
    CheckpointTransaction::commit(target, nonintersecting);
  }
  catch (const std::runtime_error &) { nonintersecting_rejected = true; }
  CHECK(nonintersecting_rejected && target.t == time_before_reject,
        "checkpoint repartition accepted an unconsumed nonintersecting saved row");

  structure changed_structure(gv, two_epsilon, no_pml(), identity(), 1);
  fields changed_target(&changed_structure);
  changed_target.use_real_fields();
  changed_target.require_component(Hx);
  changed_target.add_point_source(Ez, source_time, source_point);
  changed_target.advance(1);
  bool changed_material_rejected = false;
  try { CheckpointTransaction::validate_target(changed_target, source_image); }
  catch (const std::runtime_error &) { changed_material_rejected = true; }
  CHECK(changed_material_rejected,
        "checkpoint repartition accepted a changed global material definition");

  char filename[160];
  snprintf(filename, sizeof(filename), "/tmp/meep-checkpoint-repartition-%ld.h5", long(getpid()));
  source.dump(filename, false);
  target.load(filename, false);
  CHECK(target.t == source.t, "checkpoint repartition did not restore the saved timestep");
  const vec probes[] = {vec(-1.1, -0.6), vec(-0.35, 0.2), vec(0.25, -0.3), vec(1.0, 0.55)};
  bool restored = true;
  for (const vec &probe : probes)
    restored &= std::abs(source.get_field(Ez, probe) - target.get_field(Ez, probe)) < 1e-12;
  CHECK(restored, "checkpoint repartition changed restored global Yee field values");
  source.advance(2);
  target.advance(2);
  bool continued = true;
  for (const vec &probe : probes)
    continued &= std::abs(source.get_field(Ez, probe) - target.get_field(Ez, probe)) < 1e-10;
  CHECK(continued, "checkpoint repartition did not preserve continuation semantics");
  std::remove(filename);
}

static void checkpoint_mpi_repartition_file(const char *filename, bool write_only) {
  const grid_volume gv = vol2d(3.0, 2.0, 8.0);
  gaussian_src_time source_time(0.27, 0.08);
  const vec source_point(0.17, -0.21);
  if (write_only) {
    structure source_structure(gv, eps_slab, no_pml(), identity(), 1);
    fields source(&source_structure);
    source.use_real_fields();
    source.add_point_source(Ez, source_time, source_point);
    source.advance(3);
    if (am_master()) std::remove(filename);
    all_wait();
    source.dump(filename, true);
    master_printf("backend_api: MPI checkpoint writer PASS at np=%d\n", count_processors());
    return;
  }

  structure reference_structure(gv, eps_slab, no_pml(), identity(), 1);
  structure target_structure(gv, eps_slab, no_pml(), identity(), 3);
  fields reference(&reference_structure);
  fields target(&target_structure);
  reference.use_real_fields();
  target.use_real_fields();
  reference.add_point_source(Ez, source_time, source_point);
  target.add_point_source(Ez, source_time, source_point);
  reference.advance(3);
  target.advance(1);
  if (count_processors() > 1) {
    const int t_before_failure = target.t;
    checkpoint_set_failure_for_testing(CheckpointFailurePoint::read, 0);
    bool read_rejected = false;
    try { target.load(filename, true); }
    catch (const std::runtime_error &) { read_rejected = true; }
    checkpoint_clear_failure_for_testing();
    CHECK(sum_to_all(int(read_rejected)) == count_processors() && target.t == t_before_failure,
          "shared checkpoint asymmetric read preflight did not reject collectively");
  }
  target.load(filename, true);
  CHECK(target.t == reference.t,
        "MPI checkpoint repartition did not restore the saved timestep");
  const vec probes[] = {vec(-1.1, -0.6), vec(-0.35, 0.2), vec(0.25, -0.3), vec(1.0, 0.55)};
  bool restored = true;
  for (const vec &probe : probes)
    restored &= std::abs(reference.get_field(Ez, probe) - target.get_field(Ez, probe)) < 1e-12;
  CHECK(restored, "MPI checkpoint repartition changed restored global Yee field values");
  reference.advance(2);
  target.advance(2);
  bool continued = true;
  for (const vec &probe : probes)
    continued &= std::abs(reference.get_field(Ez, probe) - target.get_field(Ez, probe)) < 1e-10;
  CHECK(continued, "MPI checkpoint repartition did not preserve continuation semantics");
  if (failures) return;
  master_printf("backend_api: MPI checkpoint reader PASS at np=%d\n", count_processors());
}

static void checkpoint_cross_precision_file(const char *filename, bool write_only) {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  gaussian_src_time source_time(0.25, 0.0625);
  const vec source_point(0.125, -0.1875);
  const double frequencies[2] = {0.1875, 0.3125};
  component monitored = Ez;
  lorentzian_susceptibility susceptibility(realnum(1.125), realnum(0.0625));

  if (write_only) {
    structure source_structure(gv, unit_epsilon, no_pml(), identity(), 1);
    source_structure.add_susceptibility(unit_epsilon, E_stuff, susceptibility);
    fields source(&source_structure);
    source.use_real_fields();
    source.add_point_source(Ez, source_time, source_point);
    dft_fields monitor = source.add_dft_fields(&monitored, 1, source.v, frequencies, 2);
    (void)monitor;
    source.advance(4);
    if (am_master()) std::remove(filename);
    all_wait();
    source.dump(filename, false);
    master_printf("backend_api: %zu-bit checkpoint writer PASS\n", 8 * sizeof(realnum));
    return;
  }

  CheckpointImage saved;
  {
    h5file stored(filename, h5file::READONLY, false, true);
    saved = CheckpointTransaction::read_manifest(stored, false);
  }
  CHECK(saved.host_realnum_bytes != sizeof(realnum),
        "cross-precision checkpoint reader received a native-precision file");

  structure target_structure(gv, unit_epsilon, no_pml(), identity(), 1);
  target_structure.add_susceptibility(unit_epsilon, E_stuff, susceptibility);
  fields target(&target_structure);
  target.use_real_fields();
  target.add_point_source(Ez, source_time, source_point);
  dft_fields monitor = target.add_dft_fields(&monitored, 1, target.v, frequencies, 2);
  (void)monitor;
  target.advance(1);
  const CheckpointImage before_load = CheckpointTransaction::snapshot(target);
  CHECK(saved.dft_recipe_signature == before_load.dft_recipe_signature,
        "cross-precision DFT recipe identity depends on catalog numbering");
  target.load(filename, false);
  const CheckpointImage restored = CheckpointTransaction::snapshot(target);
  bool converted = restored.rows.size() == saved.rows.size();
  for (const CheckpointRow &source_row : saved.rows) {
    const CheckpointRow *target_row = NULL;
    for (const CheckpointRow &candidate : restored.rows)
      if (candidate.key == source_row.key) {
        target_row = &candidate;
        break;
      }
    if (!target_row || target_row->has_alias != source_row.has_alias ||
        target_row->values.size() != source_row.values.size()) {
      converted = false;
      continue;
    }
    for (size_t i = 0; i < source_row.values.size(); ++i) {
      const double expected = double(realnum(source_row.values[i]));
      converted &= (std::isnan(target_row->values[i]) && std::isnan(expected)) ||
                   target_row->values[i] == expected;
    }
  }
  CHECK(converted && target.t == saved.timestep,
        "cross-precision checkpoint did not explicitly convert every real/complex row");
  target.advance(2);
  const std::complex<double> probe = target.get_field(Ez, vec(0.25, -0.125));
  CHECK(target.t == saved.timestep + 2 && std::isfinite(probe.real()) &&
            std::isfinite(probe.imag()),
        "cross-precision checkpoint did not continue with finite state");
  master_printf("backend_api: %zu-bit checkpoint reader PASS\n", 8 * sizeof(realnum));
}

static void test_checkpoint_transient_rejections() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->advance(1);
  char filename[160];
  snprintf(filename, sizeof(filename), "/tmp/meep-checkpoint-transient-%ld-%d.h5",
           long(getpid()), my_rank());

  f->synchronize_magnetic_fields();
  bool magnetic_rejected = false;
  try { f->dump(filename, false); }
  catch (const std::runtime_error &) { magnetic_rejected = true; }
  CHECK(magnetic_rejected, "checkpoint accepted an active magnetic synchronization");
  f->restore_magnetic_fields();

  f->phasein_time = 1;
  bool phase_rejected = false;
  try { f->dump(filename, false); }
  catch (const std::runtime_error &) { phase_rejected = true; }
  CHECK(phase_rejected, "checkpoint accepted an active material phase");
  f->phasein_time = 0;

  int active_chunk = -1;
  for (int chunk = 0; chunk < f->num_chunks; ++chunk)
    if (f->chunks[chunk]->is_mine()) {
      active_chunk = chunk;
      f->chunks[chunk]->set_solve_cw_omega(0.3);
      break;
    }
  bool cw_rejected = false;
  try { f->dump(filename, false); }
  catch (const std::runtime_error &) { cw_rejected = true; }
  CHECK(cw_rejected, "checkpoint accepted an active CW solve");
  if (active_chunk >= 0) f->chunks[active_chunk]->unset_solve_cw_omega();
  std::remove(filename);
  std::remove((std::string(filename) + ".tmp").c_str());
  delete f;
  delete s;
}

static void test_compact_dft_reduction_boundary() {
  structure *s;
  fields *f;
  build(&s, &f);
  f->require_component(Ez);
  component component_ez = Ez;
  const double frequencies[2] = {0.2, 0.3};
  dft_fields monitor =
      f->add_dft_fields(&component_ez, 1, volume(vec(0.4, 0.4), vec(1.2, 1.2)), frequencies, 2,
                        /*use_centered_grid=*/true,
                        /*decimation_factor=*/1,
                        /*persist=*/true);

  rebuild_trace rebuilds;
  access_trace accesses;
  compact_trace compact;
  f->backend = new compact_access_backend(*f, rebuilds, accesses, compact);
  f->advance(1);

  dft_flux flux(Ez, Ez, monitor.chunks, monitor.chunks, frequencies, 2, monitor.where, NO_DIRECTION,
                true);
  flux.monitor_lifetime = monitor.monitor_lifetime;
  dft_energy energy(monitor.chunks, monitor.chunks, monitor.chunks, monitor.chunks, frequencies, 2,
                    monitor.where);
  energy.monitor_lifetime = monitor.monitor_lifetime;
  dft_force force(monitor.chunks, monitor.chunks, monitor.chunks, frequencies, 2, monitor.where);
  force.monitor_lifetime = monitor.monitor_lifetime;

  accesses.reads = accesses.dft_reads = 0;
  (void)f->dft_norm();
  delete[] flux.flux();
  (void)flux.complexflux();
  delete[] energy.electric();
  delete[] energy.magnetic();
  delete[] energy.total();
  delete[] force.force();

  CHECK(compact.calls == 8, "public compact queries issued %zu calls, expected 8", compact.calls);
  CHECK(sum_to_all(int(accesses.dft_reads)) == 0,
        "compact DFT query unexpectedly read a complete accumulator");
  CHECK(compact.requests.size() == compact.calls,
        "compact mock did not capture every successful request");
  if (compact.requests.size() >= 8) {
    CHECK(compact.requests[0].kind == DftReductionKind::norm2 &&
              compact.requests[0].result_count == 1,
          "DFT norm constructed the wrong compact request");
    CHECK(compact.requests[1].kind == DftReductionKind::real_weighted_product &&
              compact.requests[1].result_count == 2,
          "flux constructed the wrong compact request");
    CHECK(compact.requests[2].kind == DftReductionKind::complex_weighted_product,
          "complex flux constructed the wrong compact request");
    CHECK(compact.requests[3].terms.empty() ||
              fabs(compact.requests[3].terms[0].weight.real() - 0.5) < 1e-15,
          "electric energy lost its one-half weight");
    CHECK(compact.requests[4].terms.empty() ||
              fabs(compact.requests[4].terms[0].weight.real() - 0.5) < 1e-15,
          "magnetic energy lost its one-half weight");
    CHECK(compact.requests[7].kind == DftReductionKind::real_weighted_product,
          "force constructed the wrong compact request");
  }

  bool saw_persistent_subset = false;
  if (!compact.requests.empty())
    for (size_t i = 0; i < compact.requests[0].terms.size(); ++i) {
      const DftReductionTerm &term = compact.requests[0].terms[i];
      size_t selected = 1;
      size_t maximum = term.region.base;
      for (int axis = 0; axis < 3; ++axis) {
        selected *= term.region.counts[axis];
        maximum += (term.region.counts[axis] - 1) * term.region.strides[axis];
      }
      CHECK(maximum < term.storage_points, "persistent compact region exceeds its allocation");
      saw_persistent_subset = saw_persistent_subset || selected < term.storage_points;
    }
  CHECK(or_to_all(saw_persistent_subset),
        "persistent DFT fixture did not produce an unpadded-in-padded compact region");

  if (!compact.requests.empty() && !compact.requests[0].terms.empty()) {
    DftReductionRequest malformed = compact.requests[0];
    std::complex<double> result;
    bool rejected = false;
    malformed.terms[0].left = invalid_array();
    try {
      f->backend->reduce_dft(malformed, &result, 1);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    CHECK(rejected, "compact mock accepted an invalid array id");

    malformed = compact.requests[0];
    malformed.terms[0].region.counts[0] = 3;
    malformed.terms[0].region.strides[0] = std::numeric_limits<size_t>::max();
    rejected = false;
    try {
      f->backend->reduce_dft(malformed, &result, 1);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    CHECK(rejected, "compact mock accepted an overflowing region");

    malformed = compact.requests[0];
    malformed.terms[0].right = malformed.terms[0].left;
    rejected = false;
    try {
      f->backend->reduce_dft(malformed, &result, 1);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    CHECK(rejected, "compact mock accepted a norm request with a right operand");
  }

  compact.fail_rank = 0;
  compact.fail_call = int(compact.calls);
  bool failed = false;
  try {
    delete[] flux.flux();
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors(),
        "rank-asymmetric compact flux failure was not reconciled");

  compact.fail_call = int(compact.calls);
  failed = false;
  try {
    delete[] energy.electric();
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors(),
        "rank-asymmetric compact electric-energy failure was not reconciled");

  compact.fail_call = int(compact.calls);
  failed = false;
  try {
    delete[] energy.magnetic();
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors(),
        "rank-asymmetric compact magnetic-energy failure was not reconciled");

  compact.fail_call = int(compact.calls);
  failed = false;
  try {
    delete[] force.force();
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors(),
        "rank-asymmetric compact force failure was not reconciled");

  compact.fail_call = int(compact.calls + 1);
  failed = false;
  try {
    delete[] energy.total();
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  CHECK(sum_to_all(int(failed)) == count_processors(),
        "rank-asymmetric compact total-energy second-call failure was not reconciled");
  all_wait();

  monitor.remove();
  delete f;
  delete s;
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  if (const char *filename = getenv("MEEP_BACKEND_API_CHECKPOINT_WRITE")) {
    checkpoint_mpi_repartition_file(filename, true);
    return failures ? 1 : 0;
  }
  if (const char *filename = getenv("MEEP_BACKEND_API_CHECKPOINT_READ")) {
    checkpoint_mpi_repartition_file(filename, false);
    return failures ? 1 : 0;
  }
  if (const char *filename = getenv("MEEP_BACKEND_API_PRECISION_WRITE")) {
    checkpoint_cross_precision_file(filename, true);
    return failures ? 1 : 0;
  }
  if (const char *filename = getenv("MEEP_BACKEND_API_PRECISION_READ")) {
    checkpoint_cross_precision_file(filename, false);
    return failures ? 1 : 0;
  }
  if (getenv("MEEP_BACKEND_API_CHECKPOINT_ONLY")) {
#ifdef HAVE_HDF5
    test_backend_safe_host_access();
    test_checkpoint_metadata_boundaries();
    test_checkpoint_susceptibility_identity();
    test_checkpoint_dft_identity_precision_and_order();
    test_checkpoint_transient_rejections();
    test_checkpoint_polarization_dft_roundtrip();
    test_checkpoint_noisy_rank_seed_continuation();
    test_checkpoint_single_rank_repartition();
#endif
    if (sum_to_all(failures)) return 1;
    master_printf("backend_api: checkpoint checks passed\n");
    return 0;
  }

  if (getenv("MEEP_BACKEND_API_BETA_ONLY")) {
    test_resident_beta_fingerprint();
    test_material_idle_rank_component_promotion();
    if (failures) return 1;
    master_printf("backend_api: beta checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_BFAST_ONLY")) {
    test_resident_bfast_fingerprint();
    if (failures) return 1;
    master_printf("backend_api: BFAST checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_CYLINDRICAL_ONLY")) {
    test_resident_cylindrical_fingerprint();
    if (failures) return 1;
    master_printf("backend_api: cylindrical checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_FLUX_ONLY")) {
    test_resident_legacy_flux_lifecycle();
    test_resident_legacy_flux_rank_mismatch();
    test_resident_legacy_flux_catalog_rebuild();
    if (failures) return 1;
    master_printf("backend_api: legacy flux checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_MAGNETIC_ONLY")) {
    test_backend_reselection_invalidates_representation();
    test_resident_magnetic_dispatch();
    if (failures) return 1;
    master_printf("backend_api: magnetic checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_MATERIAL_ONLY")) {
    test_geometry_backed_material_ir();
    test_geometry_backed_material_ir_removal();
    test_material_recipe_and_provisional_storage();
    test_resident_material_recipe_lifecycle();
    test_resident_material_coefficient_preparation();
    test_material_phase_transaction();
    test_material_phase_cpu_to_resident_preparation();
    if (failures) return 1;
    master_printf("backend_api: material checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_MATERIAL_RECIPE_ONLY")) {
    test_geometry_backed_material_ir();
    test_material_ir_capture_atomicity();
    test_geometry_backed_material_ir_removal();
    test_material_recipe_and_provisional_storage();
    test_resident_material_recipe_lifecycle();
    if (failures) return 1;
    master_printf("backend_api: material recipe checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_MATERIAL_RECIPE_UNIT_ONLY")) {
    test_geometry_backed_material_ir();
    test_material_ir_capture_atomicity();
    test_geometry_backed_material_ir_removal();
    test_material_recipe_and_provisional_storage();
    if (failures) return 1;
    master_printf("backend_api: material recipe unit checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_MATERIAL_RECIPE_LIFECYCLE_ONLY")) {
    test_resident_material_recipe_lifecycle();
    if (failures) return 1;
    master_printf("backend_api: material recipe lifecycle checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_PR54_ONLY")) {
    test_owned_tiled_material_route();
    test_material_fallback_policy_transaction();
    test_material_fallback_route_removal();
    test_material_phase_early_fallback_preflight();
    test_material_promotion_transaction();
    test_material_idle_rank_component_promotion();
    test_material_route_lattice();
    test_material_classification_fact_contract();
    test_material_classification_collective_failures();
    if (failures) return 1;
    master_printf("backend_api: PR5.4 backend-neutral checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_NOISY_ONLY")) {
    test_resident_noisy_seed_lifecycle();
    test_resident_noisy_prelaunch_failures();
    test_resident_noisy_collective_preflight();
    test_resident_noisy_zero_row();
    test_resident_advance_failure_poison();
    if (failures) return 1;
    master_printf("backend_api: noisy checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_MULTILEVEL_ONLY")) {
    test_resident_multilevel_lifecycle();
    test_resident_multilevel_collective_preflight();
    failures = sum_to_all(failures);
    if (failures) return 1;
    master_printf("backend_api: multilevel checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_CUSTOM_ONLY")) {
    test_resident_host_custom_policy_lifecycle();
    test_resident_host_custom_collective_preflight();
    failures = sum_to_all(failures);
    if (failures) return 1;
    master_printf("backend_api: custom fallback checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_CUSTOM_SPLIT_ONLY")) {
    test_resident_host_custom_split_communicator();
    if (failures) return 1;
    master_printf("backend_api: custom split-communicator checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_MULTILEVEL_SPLIT_ONLY")) {
    test_resident_multilevel_split_communicator();
    if (failures) return 1;
    master_printf("backend_api: multilevel split-communicator checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_NOISY_DEFAULT_ONLY")) {
    test_resident_noisy_lazy_default();
    if (failures) return 1;
    master_printf("backend_api: noisy lazy-default checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_NOISY_INVALID_ONLY")) {
    test_resident_noisy_invalid_first_restore();
    if (failures) return 1;
    master_printf("backend_api: noisy invalid-restore checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_NOISY_SPLIT_ONLY")) {
    test_resident_noisy_split_global_rank();
    if (failures) return 1;
    master_printf("backend_api: noisy split-communicator checks passed\n");
    return 0;
  }
  if (getenv("MEEP_BACKEND_API_CW_ONLY")) {
    test_resident_cw_lifecycle();
    test_resident_cw_rejection_surface();
    test_resident_cw_malformed_results_and_alias();
    test_cw_clone_failure_is_atomic();
    test_cw_staged_pipeline_failures();
    test_cw_structural_rebuild_migrates_authority();
    test_cw_warm_staged_pipeline_failures();
    test_cw_cold_conductivity_transaction();
    test_cw_warm_monitor_pipeline_failures();
    test_cold_cw_preflight_restores_existing_plans();
    test_cpu_cw_hook_declines_without_initialization();
    if (failures) return 1;
    master_printf("backend_api: CW checks passed\n");
    return 0;
  }

  test_selection();
  test_susceptibility_clone_coefficients();
  test_construction_equivalence();
  test_read_write_roundtrip();
  test_precision_policy();
  test_dft_access_boundaries();
  test_detached_dft_access();
  test_backend_lifecycle_epoch();
  test_backend_reselection_invalidates_representation();
  test_resident_magnetic_dispatch();
  test_resident_material_coefficient_preparation();
  test_material_phase_transaction();
  test_material_phase_cpu_to_resident_preparation();
  test_resident_cw_lifecycle();
  test_resident_cw_rejection_surface();
  test_resident_cw_malformed_results_and_alias();
  test_cw_clone_failure_is_atomic();
  test_cw_staged_pipeline_failures();
  test_cw_structural_rebuild_migrates_authority();
  test_cw_warm_staged_pipeline_failures();
  test_cw_cold_conductivity_transaction();
  test_cw_warm_monitor_pipeline_failures();
  test_cold_cw_preflight_restores_existing_plans();
  test_cpu_cw_hook_declines_without_initialization();
  test_resident_polarization_preparation();
  test_resident_multilevel_lifecycle();
  test_resident_host_custom_policy_lifecycle();
  test_resident_host_custom_collective_preflight();
  test_resident_multilevel_collective_preflight();
  test_resident_noisy_seed_lifecycle();
  test_resident_noisy_prelaunch_failures();
  test_resident_noisy_collective_preflight();
  test_resident_noisy_zero_row();
  test_resident_advance_failure_poison();
  test_resident_beta_fingerprint();
  test_resident_bfast_fingerprint();
  test_resident_cylindrical_fingerprint();
  test_classification_change_recompiles();
  test_geometry_backed_material_ir();
  test_material_ir_capture_atomicity();
  test_geometry_backed_material_ir_removal();
  test_material_recipe_and_provisional_storage();
  test_owned_tiled_material_route();
  test_material_fallback_policy_transaction();
  test_material_fallback_route_removal();
  test_material_phase_early_fallback_preflight();
  test_material_promotion_transaction();
  test_material_idle_rank_component_promotion();
  test_material_route_lattice();
  test_material_classification_fact_contract();
  test_material_classification_collective_failures();
  test_resident_material_recipe_lifecycle();
  test_initialization_plan();
  test_authority_safe_state_rebuild();
  test_cpu_state_rebuild_is_safe_noop();
  test_backend_safe_host_access();
#ifdef HAVE_HDF5
  test_checkpoint_metadata_boundaries();
  test_checkpoint_susceptibility_identity();
  test_checkpoint_dft_identity_precision_and_order();
  test_checkpoint_transient_rejections();
  test_checkpoint_polarization_dft_roundtrip();
  test_checkpoint_noisy_rank_seed_continuation();
  test_checkpoint_single_rank_repartition();
  test_sharded_dft_checkpoint_ordering();
#endif
  test_compact_dft_reduction_boundary();

  if (failures) {
    master_printf("backend_api: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("backend_api: all checks passed\n");
  return 0;
}
