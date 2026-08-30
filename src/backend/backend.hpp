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
#include "backend/random_state.hpp"
#include "backend/step_plan.hpp"
#include "backend/storage_plan.hpp"

namespace meep {

struct Executable {
  virtual ~Executable() {}
};

/* Type-erased backend ownership. Concrete CPU/device implementations derive
   from these private bases so fields can destroy them without knowing their
   representation. The ordinary executable remains fields-owned. A resident
   CW executable is deliberately separate and state-owned: it may retain
   workspace and resolved array references that cannot outlive the state.
   Because the base destructor releases it after derived members have been
   destroyed, its destructor must not dereference derived BackendState data. */
struct BackendState {
  BackendState()
      : cw_executable(NULL), cw_storage_fingerprint(0), cw_step_plan_signature(0),
        cw_plan_signature(0), accepted_random_seed(), random_seed_snapshot_accepted(false) {}
  virtual ~BackendState() { delete cw_executable; }

  void clear_cw_executable() {
    delete cw_executable;
    cw_executable = NULL;
    cw_storage_fingerprint = 0;
    cw_step_plan_signature = 0;
    cw_plan_signature = 0;
  }

  Executable *cw_executable;
  uint64_t cw_storage_fingerprint;
  uint64_t cw_step_plan_signature;
  uint64_t cw_plan_signature;
  RandomSeedSnapshot accepted_random_seed;
  bool random_seed_snapshot_accepted;
};

struct InitializationPlan; // src/backend/initialization_plan.hpp

/* Retire a backend epoch while its derived state is still intact. This clears
   the state-owned CW executable before derived-state members are destroyed. */
void destroy_backend_state(BackendState *&state);

enum class CwSolveStatus { converged, not_converged, breakdown };

struct CwSolveRequest {
  double tolerance;
  int maxiters;
  std::complex<double> frequency;
  int L;
  bool eigfrequency;
  int entry_t;
  double entry_time;

  CwSolveRequest()
      : tolerance(0.0), maxiters(0), frequency(0.0, 0.0), L(0), eigfrequency(false), entry_t(0),
        entry_time(0.0) {}
};

struct CwSolveResult {
  CwSolveStatus status;
  int iterations;
  size_t operator_applications;
  double recursive_relative_residual;
  double true_relative_residual;

  CwSolveResult()
      : status(CwSolveStatus::breakdown), iterations(0), operator_applications(0),
        recursive_relative_residual(0.0), true_relative_residual(0.0) {}
};

class CwSolveSession {
public:
  CwSolveSession(fields &owner, const CwSolveRequest &request);
  ~CwSolveSession();
  void restore_before_final_dft() noexcept;
  bool boundary_called() const { return boundary_called_; }
  bool at_entry_state() const;

private:
  CwSolveSession(const CwSolveSession &);
  CwSolveSession &operator=(const CwSolveSession &);
  fields &owner_;
  int entry_t_;
  bool boundary_called_;
};

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
  ExecutionBackend() : poisoned_(false) {}
  virtual ~ExecutionBackend() {}

  virtual BackendState *create_state(const StoragePlan &) = 0;
  virtual void initialize(const InitializationPlan &, BackendState &) = 0;

  // Pass 2: report what initialization actually produced.
  virtual MaterialClassification classify_state(const StoragePlan &, BackendState &) = 0;
  virtual void finalize_storage(const StoragePlan &, BackendState &) = 0;

  virtual Executable *compile(const StepPlan &, BackendState &) = 0;
  virtual void advance(Executable &, BackendState &, int num_steps) = 0;

  /* Refresh backend-private noisy-RNG metadata without rebuilding storage or
     executable state. Throwing must leave the previously active seed usable;
     an implementation may poison itself only after an irreversible/enqueued
     transfer failure. The caller publishes the candidate host snapshot only
     after this hook returns successfully. */
  virtual void refresh_noisy_seed(const RandomSeedSnapshot &, BackendState &) {
    throw std::logic_error("backend does not implement noisy seed refresh");
  }

  /* A resident CW solve is one coarse operation. CPU declines this hook and
     keeps the legacy solver unchanged. preflight_cw must not invoke source
     callbacks or mutate fields; it may return `cached` after validating and
     reserving workspace, or a replacement compiled artifact. solve_cw owns
     the complete post-preflight operation, including final synchronization
     and the single due-filtered DFT action. */
  virtual bool supports_cw(const CwSolveRequest &, std::string &why) const {
    why = "backend does not implement resident solve_cw";
    return false;
  }
  virtual Executable *preflight_cw(const CwSolveRequest &, const StepPlan &, const CwPlan &,
                                   Executable *, BackendState &) {
    throw std::logic_error("backend does not implement resident solve_cw preflight");
  }
  /* Immediately before its owned final due-filtered DFT action, the coarse
     hook must call session.restore_before_final_dft(), which restores both the
     solve-entry clock and every chunk's transient CW flag. The wrapper checks
     that boundary and restores the same state defensively on every return. */
  virtual CwSolveResult solve_cw(const CwSolveRequest &, const StepPlan &, const CwPlan &,
                                 Executable &, Executable &, BackendState &, CwSolveSession &) {
    throw std::logic_error("backend does not implement resident solve_cw");
  }

  virtual void read(ArrayRef, void *host_buffer, size_t bytes) = 0;        // converts from storage
  virtual void write(ArrayRef, const void *host_buffer, size_t bytes) = 0; // converts to storage
  /* Legacy magnetic synchronization mutates several coupled field families
     through host loops. Resident backends must opt into a complete lowering;
     merely copying the stale host mirrors is not correct. */
  virtual bool supports_magnetic_synchronization() const {
    return !requires_full_storage_preparation();
  }
  /* Validate and reserve everything required by the following transition
     without mutating live fields or the backend-private snapshot. All ranks
     reconcile this preflight before any rank enters the dispatch hook. A
     dispatch must therefore contain no recoverable validation or allocation;
     an exception after it starts is a poisoned-backend failure, not a
     retryable transition. */
  virtual void preflight_magnetic_transition(Executable &, BackendState &, bool) {}
  virtual void synchronize_magnetic_fields(Executable &, BackendState &) {
    throw std::logic_error("backend does not implement magnetic synchronization");
  }
  virtual void restore_magnetic_fields(Executable &, BackendState &) {
    throw std::logic_error("backend does not implement magnetic restoration");
  }
  void poison() { poisoned_ = true; }
  bool is_poisoned() const { return poisoned_; }
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
     resident values. Mutable field, N, P, and P_prev rows must be materialized
     on the host, but immutable host-authored rows such as multilevel GammaInv
     must not be overwritten by a lower-precision resident round trip.
     Host-authoritative backends may override this as a no-op; the default
     refuses the rebuild so a future device backend cannot silently discard
     data merely because it forgot to implement migration. */
  virtual void prepare_state_rebuild(BackendState &, DirtyMask) {
    throw std::logic_error("backend does not support authority-safe state rebuild");
  }

  /* Reject an unsupported request clearly and *collectively*: every rank has to
     reach the same verdict, or the ones that accept will wait forever on the
     ones that abort. */
  virtual bool accepts(const execution_options &opts, std::string &why) const = 0;

private:
  bool poisoned_;
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

/* Publish exactly one catalogued host range after a legacy host-side mutation.
   Like the read counterpart, this records the first local error so all ranks
   can reach a single reconciliation boundary. */
bool backend_write_host_range(fields &f, const void *host_address, size_t elements,
                              std::string &local_error);

/* Every participating rank must call this at the boundary following a batch
   of backend reads and before entering the next MPI/HDF5 collective. */
void backend_reconcile_host_access(const std::string &local_error, const char *site);

/* Publish final legacy-flux scalars by checked newest-first ordinal. The
   half-step samples remain backend-private. */
void backend_publish_legacy_flux(fields &f, const double *values, size_t count, const char *site);
/* Rebuild only the legacy-flux recipe vectors and marker operations while
   preserving the already-compiled ordinary operands. Resident classification
   may deliberately remove host-only coefficient rows from the live catalog,
   so a flux-only mutation must not regenerate unrelated curl/constitutive
   descriptors from that reduced catalog. */
StepPlan build_legacy_flux_only_step_plan(fields &f, StepProgram program,
                                          const StepPlan &stable);
bool backend_try_refresh_legacy_flux(fields &f, const char *site);
void backend_set_legacy_flux_prepare_failure_for_testing(int rank);
void backend_refresh_noisy_seed(fields &f, const StepPlan &plan, const char *site);
void backend_set_legacy_flux_descriptor_failure_for_testing(int rank, int flux_ordinal);

/* Preserve resident-authoritative values and retire the old backend objects
   before a host-side field-layout mutation can delete or replace their
   catalogued storage. Every rank must enter this boundary together. */
void backend_prepare_field_layout_change(fields &f, DirtyMask reasons, const char *site);

/* Split form used by multi-resource transactions: preflight/migrate while the
   old representation remains live, then retire it only after every other
   fallible preparation has succeeded collectively. */
void backend_preflight_field_layout_change(fields &f, DirtyMask reasons, const char *site);
void backend_commit_field_layout_change(fields &f);

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
void backend_publish_dft_chains(fields &owner, int count, dft_chunk *const *heads,
                                const char *site);
bool backend_write_dft_chunk(dft_chunk *chunk, std::string &local_error);
bool backend_write_dft_chain(dft_chunk *head, std::string &local_error);

/* Owner-free chain boundaries are used by Python's opaque DFT data helpers.
   A null local chain is valid: every rank still participates in the global
   decision and the one subsequent error reconciliation. */
void backend_refresh_dft_chain(const dft_chunk *head, const char *site);
void backend_publish_dft_chain(dft_chunk *head, const char *site);

/* Refuse legacy magnetic synchronization before it changes its nesting
   counter or allocates backup arrays on an unsupported resident backend. */
void backend_require_magnetic_synchronization(const fields &f, const char *site);
bool backend_try_synchronize_magnetic_fields(fields &f, const char *site);
bool backend_try_restore_magnetic_fields(fields &f, const char *site);

/* Dispatch a complete resident CW solve. Returns false only for the CPU/default
   backend, which preserves the legacy fields::solve_cw implementation. A
   resident backend that declines is fail-closed. */
bool backend_try_solve_cw(fields &f, const CwSolveRequest &request, CwSolveResult &result);

/* Shared, noncontracting source time used by both legacy and resident CW
   paths. Keeping this out of line prevents an optimizer from fusing t*dt with
   the half-step addition and changing the last bit at a callback boundary. */
double cw_source_time(int t, double dt, double offset_in_dt);
/* Deterministic allocation-failure seam for lifecycle tests. Negative disables it. */
void backend_set_cw_clone_fail_after_for_testing(int checkpoints);
void backend_cw_clone_checkpoint();
void backend_set_cw_plan_corruption_for_testing(bool enabled);

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
