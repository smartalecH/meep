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

#include <stdlib.h>
#include <string.h>

#include "backend/cpu/cpu_backend.hpp"
#include "backend/initialization_plan.hpp"
#include "meep_internals.hpp"

namespace meep {

/* On CPU these are thin: the state is the catalog `fields` already owns, and
   the executable is the StepPlan it already built. They exist so the boundary
   has the shape a device backend needs. */
struct BackendState {
  CpuArrayCatalog *catalog;
};

struct Executable {
  const StepPlan *plan;
};

BackendState *CpuBackend::create_state(const StoragePlan &) {
  BackendState *st = new BackendState;
  st->catalog = f_.array_catalog;
  return st;
}

void CpuBackend::initialize(const InitializationPlan &, BackendState &) {
  /* Material initialization on CPU is unchanged: geom_epsilon,
     structure::set_materials, structure_chunk::set_chi1inv, eff_chi1inv_row and
     libctl adaptive integration populate the coefficient arrays before the
     catalog is frozen, with their geometry queries, staggered volumes,
     averaging formulas, tolerances and fallback behavior untouched (§12.4).
     There is nothing for the backend to replay. */
}

MaterialClassification CpuBackend::classify_state(const StoragePlan &plan, BackendState &) {
  return classify(f_, plan);
}

void CpuBackend::finalize_storage(const StoragePlan &, BackendState &) {
  /* Nothing to elide on CPU: set_chi1inv already deleted the trivial rows
     before preparation ever ran, so the provisional superset and the steady
     state coincide. */
}

Executable *CpuBackend::compile(const StepPlan &plan, BackendState &) {
  Executable *e = new Executable;
  e->plan = &plan;
  return e;
}

void CpuBackend::advance(Executable &, BackendState &, int num_steps) {
  f_.advance_cpu(num_steps);
}

void CpuBackend::read(ArrayRef ref, void *host_buffer, size_t bytes) {
  const realnum *src = f_.array_catalog->resolve<realnum>(ref.id);
  memcpy(host_buffer, src + ref.offset, bytes);
}

void CpuBackend::write(ArrayRef ref, const void *host_buffer, size_t bytes) {
  realnum *dst = f_.array_catalog->resolve<realnum>(ref.id);
  memcpy(dst + ref.offset, host_buffer, bytes);
}

backend_capabilities CpuBackend::capabilities() const {
  backend_capabilities c;
  /* Native only. `mixed` and `f32` would require every kernel to be
     instantiated at a second precision, which is Phase 2 work. */
  c.supports_native = true;
  c.supports_mixed = false;
  c.supports_f32 = false;
  c.memory_budget_bytes = 0; // host memory; not preflighted
  c.name = "cpu";
  return c;
}

bool CpuBackend::accepts(const execution_options &opts, std::string &why) const {
  if (opts.backend == backend_kind::nvidia) {
    why = "the nvidia backend is not built into this copy of Meep (Phase 2)";
    return false;
  }
  if (opts.precision != precision_policy_kind::native) {
    why = std::string("the cpu backend supports only precision=native, not ") +
          precision_policy_name(opts.precision);
    return false;
  }
  if (opts.device_id != automatic_device) {
    why = "the cpu backend has no devices to select";
    return false;
  }
  return true;
}

ExecutionBackend *make_backend(fields &f, const execution_options &opts, std::string &why) {
  why.clear();
  CpuBackend *cpu = new CpuBackend(f);

  if (opts.backend == backend_kind::automatic) {
    /* Only one backend exists, so `automatic` resolves to it. When a device
       backend arrives this is where the probe goes. */
    execution_options relaxed = opts;
    relaxed.backend = backend_kind::cpu;
    if (cpu->accepts(relaxed, why)) return cpu;
  }
  else if (cpu->accepts(opts, why)) {
    return cpu;
  }

  delete cpu;
  /* Collective by construction: every rank evaluates the same options against
     the same capabilities, so every rank reaches the same verdict. A rank that
     accepted while its peers rejected would hang at the next reduction. */
  return NULL;
}

} // namespace meep
