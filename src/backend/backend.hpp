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

/* The backend boundary.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 *
 * One virtual call per large lifecycle action -- never per operation, and
 * certainly never per voxel. The CPU backend continues to use STEP_CURL,
 * STEP_UPDATE_EDHB, the stride-one variants, PLOOP_OVER_*, tiling and OpenMP,
 * untouched.
 */

#ifndef MEEP_BACKEND_BACKEND_HPP
#define MEEP_BACKEND_BACKEND_HPP

#include <stdexcept>
#include <string>

#include "meep.hpp"
#include "backend/array_ref.hpp"
#include "backend/classification.hpp"
#include "backend/lifecycle.hpp"
#include "backend/precision.hpp"
#include "backend/step_plan.hpp"
#include "backend/storage_plan.hpp"

namespace meep {

/* Type-erased backend ownership. Concrete CPU/device implementations derive
   from these private bases so fields can destroy them without knowing their
   representation. */
struct BackendState {
  virtual ~BackendState() {}
};

struct Executable {
  virtual ~Executable() {}
};

struct InitializationPlan; // src/backend/initialization_plan.hpp

class ExecutionBackend {
public:
  virtual ~ExecutionBackend() {}

  virtual BackendState *create_state(const StoragePlan &) = 0;
  virtual void initialize(const InitializationPlan &, BackendState &) = 0;

  // Pass 2: report what initialization actually produced.
  virtual MaterialClassification classify_state(const StoragePlan &, BackendState &) = 0;
  virtual void finalize_storage(const StoragePlan &, BackendState &) = 0;

  virtual Executable *compile(const StepPlan &, BackendState &) = 0;
  virtual void advance(Executable &, BackendState &, int num_steps) = 0;

  virtual void read(ArrayRef, void *host_buffer, size_t bytes) = 0;  // converts from storage
  virtual void write(ArrayRef, const void *host_buffer, size_t bytes) = 0; // converts to storage
  virtual void synchronize() = 0;
  virtual backend_capabilities capabilities() const = 0;

  /* Device-resident backends need one collective, complete storage snapshot
     before create_state; the CPU backend must preserve its lazy preparation. */
  virtual bool requires_full_storage_preparation() const = 0;

  /* Called while the old state and executable are still alive, before a
     storage-layout rebuild or backend replacement can destroy authoritative
     resident values. Host-authoritative backends may override this as a no-op;
     the default refuses the rebuild so a future device backend cannot silently
     discard data merely because it forgot to implement migration. */
  virtual void prepare_state_rebuild(BackendState &, DirtyMask) {
    throw std::logic_error("backend does not support authority-safe state rebuild");
  }

  /* Reject an unsupported request clearly and *collectively*: every rank has to
     reach the same verdict, or the ones that accept will wait forever on the
     ones that abort. */
  virtual bool accepts(const execution_options &opts, std::string &why) const = 0;
};

/* Selects and constructs a backend, or fails with a clear collective error.
   Returns NULL and fills `why` when the request cannot be satisfied. */
ExecutionBackend *make_backend(fields &f, const execution_options &opts, std::string &why);

} // namespace meep

#endif // MEEP_BACKEND_BACKEND_HPP
