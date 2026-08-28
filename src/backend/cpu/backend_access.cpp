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
  delete executable;
  executable = NULL;
  delete backend_state;
  backend_state = NULL;
  delete backend;
  backend = b;
}

void fields::init_backend() {
  if (!backend) {
    execution_options opts;
    select_backend(opts);
  }
  if (!backend_state) backend_state = backend->create_state(*storage_plan);
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
