/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* Backend selection, lifecycle, and the backend-safe access points. */

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "backend/backend.hpp"
#include "backend/cpu/cpu_backend.hpp"
#include "backend/descriptors.hpp"
#include "backend/initialization_plan.hpp"
#include "backend/lifecycle.hpp"
#include "backend/precision.hpp"

namespace meep {

namespace {

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
      const ptrdiff_t candidates[4] = {idx, idx + offset1, idx + offset2,
                                       idx + offset1 + offset2};
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
  delete f.backend_state;
  f.backend_state = NULL;
}

} // namespace

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

void backend_reconcile_host_access(const std::string &local_error, const char *site) {
  if (!or_to_all(!local_error.empty())) return;
  if (local_error.empty())
    throw std::runtime_error(std::string(site) + ": backend host access failed on another MPI rank");
  throw std::runtime_error(std::string(site) + ": " + local_error);
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

void backend_refresh_dft_chains(fields &owner, int count, dft_chunk *const *heads,
                                const char *site) {
  if (!backend_host_refresh_required(owner)) return;
  std::string local_error;
  for (int i = 0; i < count; ++i)
    backend_read_dft_chain(heads[i], local_error);
  backend_reconcile_host_access(local_error, site);
}

bool backend_try_reduce_dft(fields &owner, const DftReductionRequest &request,
                            std::complex<double> *local_result, size_t result_count,
                            std::string &local_error, const char *site) {
  if (!backend_host_refresh_required(owner) ||
      !owner.backend->supports_compact_dft_reductions())
    return false;

  if (local_error.empty())
    try {
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
  std::string local_error;
  if (f.backend_state)
    try {
      f.backend->prepare_state_rebuild(
          *f.backend_state, DirtyMask(dirty_storage | dirty_initialization | dirty_executable));
    }
    catch (const std::exception &e) {
      local_error = e.what();
    }
    catch (...) {
      local_error = "unknown backend checkpoint-preparation failure";
    }
  backend_reconcile_host_access(local_error, "fields::load");
  if (f.backend_state) {
    delete f.executable;
    f.executable = NULL;
    delete f.backend_state;
    f.backend_state = NULL;
  }
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
    release_backend_state_for_rebuild(
        *this, DirtyMask(dirty_mask | dirty_storage | dirty_executable));
  }
  catch (...) {
    delete b;
    throw;
  }
  delete backend;
  backend = b;
  delete initialization_plan;
  initialization_plan = NULL;
}

void fields::init_backend() {
  if (!backend) {
    execution_options opts;
    select_backend(opts);
  }
  if (!backend->requires_full_storage_preparation()) {
    /* On the initial CPU step the lazy storage pass below the executor builds
       descriptors after the catalog is complete. Once storage is current,
       source-only mutations can refresh their descriptors without promoting
       the CPU path to eager storage preparation. */
    if (!is_dirty(*this, dirty_storage)) refresh_operation_descriptors(*this);
    if (!backend_state) backend_state = backend->create_state(*storage_plan);
    /* CPU storage is the host storage, so value mutations require no transfer. */
    clear_dirty(*this, dirty_initialization);
    return;
  }

  /* Resident lifecycle decisions guard collective preparation. Reconcile the
     relevant causes before any rank branches into classification or rebuild. */
  const DirtyMask relevant = dirty_storage | dirty_initialization | dirty_classification |
                             dirty_executable;
  const size_t local_dirty = size_t(dirty_mask & relevant);
  size_t global_dirty = 0;
  bw_or_to_all(&local_dirty, &global_dirty, 1);
  dirty_mask |= DirtyMask(global_dirty);

  /* A value-only material update can change classification without changing
     the existing storage layout. Reconcile the host representation first; any
     promotion it discovers will set dirty_storage for the rebuild below. */
  if (backend_state && is_dirty(*this, dirty_classification) &&
      !is_dirty(*this, dirty_storage))
    classify_and_finalize();

  const bool rebuild_state = !backend_state || is_dirty(*this, dirty_storage);
  if (rebuild_state) {
    if (backend_state) release_backend_state_for_rebuild(*this, DirtyMask(dirty_mask));

    /* Unlike the CPU path, a resident backend needs the complete catalog before
       it allocates. This is intentionally gated by the backend capability so
       default CPU preparation remains lazy and checkpoint-bitwise identical. */
    prepare_storage();
    /* Boundary construction is also lazy on CPU. Resident compilation needs
       the finalized metal-zero and halo schedules before it snapshots state. */
    connect_chunks();
    backend_state = backend->create_state(*storage_plan);
    dirty_mask |= dirty_initialization | dirty_classification;
  }

  /* A source-definition change does not necessarily alter storage. Refresh it
     while the old executable is still alive; advance() destroys that artifact
     only after this succeeds and a new StepPlan exists. */
  refresh_operation_descriptors(*this);

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
