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

#include <stdexcept>

#include "config.h"
#include "backend/cpu/cpu_backend.hpp"
#include "backend/initialization_plan.hpp"
#include "meep_internals.hpp"

#ifdef HAVE_NVIDIA_BACKEND
#include <unistd.h>

#include "backend/nvidia/nvidia_backend.hpp"
#include "backend/nvidia/runtime.hpp"
#endif

namespace meep {

/* On CPU these are thin: the state is the catalog `fields` already owns, and
   the executable is the StepPlan it already built. They exist so the boundary
   has the shape a device backend needs. */
struct CpuBackendState : BackendState {
  explicit CpuBackendState(CpuArrayCatalog *catalog_) : catalog(catalog_) {}
  CpuArrayCatalog *catalog;
};

struct CpuExecutable : Executable {
  explicit CpuExecutable(const StepPlan *plan_) : plan(plan_) {}
  const StepPlan *plan;
};

static const ArraySpec &checked_access(const CpuArrayCatalog &catalog, ArrayRef ref,
                                       const void *host_buffer, size_t bytes) {
  if (!is_valid(ref.id) || ref.id.value >= catalog.size())
    throw std::out_of_range("backend array access uses an invalid ArrayId");
  const ArraySpec &spec = catalog.spec(ref.id);
  if (ref.offset > spec.elements || ref.elements > spec.elements - ref.offset)
    throw std::out_of_range("backend array access exceeds the registered array");
  const size_t element_size = host_element_bytes(spec.element_type);
  if (ref.offset && element_size > size_t(-1) / ref.offset)
    throw std::overflow_error("backend host byte offset overflow");
  if (ref.elements && element_size > size_t(-1) / ref.elements)
    throw std::overflow_error("backend host byte count overflow");
  const size_t expected = ref.elements * element_size;
  if (bytes != expected)
    throw std::invalid_argument("backend array access byte count does not match ArrayRef");
  if (bytes && !host_buffer)
    throw std::invalid_argument("backend array access has a null host buffer");
  if (bytes && !catalog.resolve_untyped(ref.id))
    throw std::invalid_argument("backend array access has a null storage address");
  return spec;
}

BackendState *CpuBackend::create_state(const StoragePlan &) {
  return new CpuBackendState(f_.array_catalog);
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

void CpuBackend::finalize_storage(const StoragePlan &, const MaterialClassification &,
                                  BackendState &) {
  /* Nothing to elide on CPU: set_chi1inv already deleted the trivial rows
     before preparation ever ran, so the provisional superset and the steady
     state coincide. */
}

Executable *CpuBackend::compile(const StepPlan &plan, BackendState &) {
  return new CpuExecutable(&plan);
}

void CpuBackend::advance(Executable &, BackendState &, int num_steps) {
  f_.advance_cpu(num_steps);
}

void CpuBackend::read(ArrayRef ref, void *host_buffer, size_t bytes) {
  const ArraySpec &spec = checked_access(*f_.array_catalog, ref, host_buffer, bytes);
  if (!bytes) return;
  const size_t element_size = host_element_bytes(spec.element_type);
  const char *src = static_cast<const char *>(f_.array_catalog->resolve_untyped(ref.id));
  memcpy(host_buffer, src + ref.offset * element_size, bytes);
}

void CpuBackend::write(ArrayRef ref, const void *host_buffer, size_t bytes) {
  const ArraySpec &spec = checked_access(*f_.array_catalog, ref, host_buffer, bytes);
  if (!bytes) return;
  const size_t element_size = host_element_bytes(spec.element_type);
  char *dst = static_cast<char *>(f_.array_catalog->resolve_untyped(ref.id));
  memcpy(dst + ref.offset * element_size, host_buffer, bytes);
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

#ifdef HAVE_NVIDIA_BACKEND
  if (opts.backend == backend_kind::nvidia || opts.backend == backend_kind::automatic) {
    std::string local_error;
    ExecutionBackend *nvidia_backend = NULL;
    nvidia::device_selection selection;
    nvidia::device_properties properties;
    char local_host[256] = {0};
    bool allow_sharing = false;
    try {
      int local_rank = 0;
      int local_size = 1;
      node_local_process_info(&local_rank, &local_size);
      if (const char *sharing = getenv("MEEP_ALLOW_GPU_SHARING")) {
        if (!strcmp(sharing, "1") || !strcmp(sharing, "true") || !strcmp(sharing, "yes"))
          allow_sharing = true;
        else if (strcmp(sharing, "0") && strcmp(sharing, "false") && strcmp(sharing, "no"))
          throw std::invalid_argument(
              "MEEP_ALLOW_GPU_SHARING must be one of 0|false|no|1|true|yes");
      }
      selection = nvidia::select_device(opts.device_id, local_rank, local_size, allow_sharing);
      properties = nvidia::properties_for_device(selection.device);
      if (gethostname(local_host, sizeof(local_host) - 1) != 0)
        throw std::runtime_error("could not determine the local hostname for GPU selection");
    }
    catch (const std::exception &error) { local_error = error.what(); }
    catch (...) { local_error = "unknown NVIDIA backend selection failure"; }

    /* Never enter rank-matching collectives until every rank completed its
       local CUDA probe. Otherwise one bad visibility/device setting can strand
       its peers inside broadcast. */
    if (or_to_all(!local_error.empty())) {
      if (opts.backend == backend_kind::nvidia) {
        why = local_error.empty() ? "another rank failed NVIDIA backend selection" : local_error;
        return NULL;
      }
    }
    else {
      try {
        bool collision = false;
        for (int root = 0; root < count_processors(); ++root) {
          char peer_host[256] = {0};
          char peer_uuid[128] = {0};
          if (my_rank() == root) {
            strncpy(peer_host, local_host, sizeof(local_host) - 1);
            strncpy(peer_uuid, properties.uuid.c_str(), sizeof(peer_uuid) - 1);
          }
          broadcast(root, peer_host, int(sizeof(peer_host)));
          broadcast(root, peer_uuid, int(sizeof(peer_uuid)));
          if (!allow_sharing && root != my_rank() && !strcmp(peer_host, local_host) &&
              properties.uuid == peer_uuid)
            collision = true;
        }
        if (or_to_all(collision))
          throw std::runtime_error(
              "multiple ranks selected the same GPU; set MEEP_ALLOW_GPU_SHARING only if intentional");
        nvidia_backend = make_nvidia_backend(f, opts, selection.device, local_error);
        if (!nvidia_backend && local_error.empty())
          local_error = "the NVIDIA backend could not be constructed";
      }
      catch (const std::exception &error) { local_error = error.what(); }
      catch (...) { local_error = "unknown NVIDIA backend construction failure"; }

      if (!or_to_all(!local_error.empty())) return nvidia_backend;
      delete nvidia_backend;
      if (opts.backend == backend_kind::nvidia) {
        why = local_error.empty() ? "another rank failed NVIDIA backend construction" : local_error;
        return NULL;
      }
    }
  }
#endif

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
