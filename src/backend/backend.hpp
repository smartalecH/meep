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

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

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

enum class DftReductionKind { norm2, real_weighted_product, complex_weighted_product };

struct DftReductionRegion {
  size_t base;
  size_t counts[3];
  size_t strides[3];
};

struct DftReductionTerm {
  ArrayId left;
  ArrayId right;
  size_t storage_points;
  size_t frequencies;
  DftReductionRegion region;
  std::complex<double> weight;
};

struct DftReductionRequest {
  DftReductionKind kind;
  Precision accumulation_precision;
  size_t result_count;
  std::vector<DftReductionTerm> terms;
};

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
  virtual bool supports_compact_dft_reductions() const { return false; }
  virtual void reduce_dft(const DftReductionRequest &, std::complex<double> *, size_t) {
    throw std::logic_error("backend does not support compact DFT reductions");
  }
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

/* True only while a device/resident backend has authoritative storage. This
   also lets CPU callers skip range-discovery work entirely. */
bool backend_host_refresh_required(const fields &f);

/* Refresh exactly one catalogued host range before legacy host-side query or
   output code reads it. Returns false and records the first local error rather
   than unwinding one rank ahead of a later collective. */
bool backend_read_host_range(const fields &f, const void *host_address, size_t elements,
                             std::string &local_error);

/* Every participating rank must call this at the boundary following a batch
   of backend reads and before entering the next MPI/HDF5 collective. */
void backend_reconcile_host_access(const std::string &local_error, const char *site);

/* Refresh the exact contiguous field envelopes touched by a legacy
   loop_in_chunks consumer, then reconcile any rank-local read failure before
   the consumer reaches an MPI or HDF5 collective. */
void backend_refresh_host_fields(fields &owner, int count, const component *components,
                                 const volume &where, component cgrid, bool use_symmetry,
                                 bool snap_empty_dimensions, const char *site);

/* Queue resident-to-host reads for one DFT accumulator or an entire
   next_in_dft chain. These helpers are deliberately noncollective so callers
   can batch every local read before reconciling once. */
bool backend_read_dft_chunk(const dft_chunk *chunk, std::string &local_error);
bool backend_read_dft_chain(const dft_chunk *head, std::string &local_error);
void backend_refresh_dft_chains(fields &owner, int count, dft_chunk *const *heads,
                                const char *site);

/* Execute one synchronous, rank-local compact DFT reduction, then reconcile
   construction or backend failures before the caller enters its numeric MPI
   reduction. Returns false only when compact reductions are unsupported. */
bool backend_try_reduce_dft(fields &owner, const DftReductionRequest &request,
                            std::complex<double> *local_result, size_t result_count,
                            std::string &local_error, const char *site);

/* A checkpoint load may replace array allocations. Preserve authoritative
   resident state before those host pointers change, then retire the stale
   state/catalog consumers. */
void backend_prepare_checkpoint_load(fields &f);

} // namespace meep

#endif // MEEP_BACKEND_BACKEND_HPP
