/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* Backend selection, lifecycle, and the backend-safe access points. */

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
