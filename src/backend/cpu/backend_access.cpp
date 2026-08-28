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
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

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

void backend_refresh_noisy_seed(fields &f, const StepPlan &plan, const char *site) {
  if (!f.backend || !f.backend->requires_full_storage_preparation()) return;
  bool has_noisy_actions = false;
  for (const PolarizationUpdate &update : plan.polarization_updates)
    has_noisy_actions = has_noisy_actions || update.kind == PolarizationUpdateKind::noisy_add;
  if (!has_noisy_actions) return;
  if (!f.backend_state) throw std::logic_error(std::string(site) + ": missing backend state");

  const RandomSeedSnapshot candidate = ensure_random_seed_snapshot();
  BackendState &state = *f.backend_state;
  if (state.random_seed_snapshot_accepted &&
      state.accepted_random_seed.generation == candidate.generation)
    return;

  try {
    f.backend->refresh_noisy_seed(candidate, state);
  }
  catch (const std::exception &e) {
    throw std::runtime_error(std::string(site) + ": " + e.what());
  }
  catch (...) {
    throw std::runtime_error(std::string(site) + ": unknown noisy seed refresh failure");
  }
  state.accepted_random_seed = candidate;
  state.random_seed_snapshot_accepted = true;
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
  const bool mixed_structural = or_to_all(
      (f.dirty_mask & ~flux_closure) != dirty_none || stable_provenance_mismatch);

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
    replacement_executable = f.backend->compile(*replacement_plan, *f.backend_state);
    if (!replacement_executable)
      throw std::runtime_error("backend returned no legacy-flux replacement executable");
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown legacy flux refresh failure";
  }
  f.descriptors = live_descriptors;

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
  if (or_to_all(backend->is_poisoned()))
    throw std::runtime_error("fields::init_backend: resident backend is poisoned");
  if (!backend->requires_full_storage_preparation()) {
    /* Most CPU execution keeps its historical lazy storage preparation.
       Legacy flux is the exception: its pointer-free recipes need final
       catalog ArrayIds before the first StepPlan is built, so a cold CPU flux
       step completes storage here. Once storage is current, value-only plan
       mutations refresh without another preparation pass. */
    const bool refresh_flux_after_cpu_rebuild = fluxes && is_dirty(*this, dirty_storage);
    if (refresh_flux_after_cpu_rebuild) {
      prepare_storage();
      dirty_mask |= dirty_flux_plan | dirty_regions | dirty_executable;
    }
    if (!is_dirty(*this, dirty_storage)) refresh_operation_descriptors(*this);
    if (!backend_state) backend_state = backend->create_state(*storage_plan);
    /* CPU storage is the host storage, so value mutations require no transfer. */
    clear_dirty(*this, dirty_initialization);
    return;
  }

  /* Resident lifecycle decisions guard collective preparation. Reconcile the
     relevant causes before any rank branches into classification or rebuild. */
  const DirtyMask relevant = dirty_source_plan | dirty_monitor_plan | dirty_storage |
                             dirty_regions | dirty_initialization | dirty_classification |
                             dirty_executable;
  const size_t local_dirty = size_t(dirty_mask & relevant);
  size_t global_dirty = 0;
  bw_or_to_all(&local_dirty, &global_dirty, 1);
  dirty_mask |= DirtyMask(global_dirty);

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
    std::string flux_error;
    size_t local_flux_definition = size_t(legacy_flux_definition_signature(*this));
    size_t reference_flux_definition = local_flux_definition;
    broadcast(0, &reference_flux_definition, 1);
    if (local_flux_definition != reference_flux_definition)
      flux_error = "legacy flux definitions differ across MPI ranks";
    backend_reconcile_host_access(flux_error, "fields::init_backend legacy flux preflight");

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
