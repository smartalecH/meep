/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
% Private staged CUDA/MPI boundary transport. MPI and CUDA implementation
% types are deliberately hidden from this header.
*/

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_MPI_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_MPI_HPP

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>

#include "backend/dependency_region.hpp"
#include "backend/nvidia/nvidia_boundaries.hpp"

namespace meep {
namespace nvidia {

struct staged_transport_statistics {
  uint64_t messages_sent;
  uint64_t messages_received;
  uint64_t bytes_sent;
  uint64_t bytes_received;
  uint64_t device_to_host_calls;
  uint64_t device_to_host_bytes;
  uint64_t host_to_device_calls;
  uint64_t host_to_device_bytes;
  uint64_t direct_bytes;
  uint64_t gather_launches;
  uint64_t scatter_launches;
  uint64_t testsome_polls;
  uint64_t waitall_calls;
  uint64_t request_completions;
  uint64_t slot_reuses;
  uint64_t overlap_stages;
  uint64_t overlap_interior_launches;
  uint64_t overlap_boundary_launches;
  uint64_t high_water_requests;
  /* Monotonic host elapsed-time counters for this transport epoch.  CUDA
     work remains asynchronous; synchronization time is charged to the first
     later phase that must observe completion. */
  uint64_t gather_pack_nanoseconds;
  uint64_t device_to_host_nanoseconds;
  uint64_t mpi_progress_nanoseconds;
  uint64_t mpi_wait_nanoseconds;
  uint64_t host_to_device_nanoseconds;
  uint64_t scatter_unpack_nanoseconds;
  size_t device_bytes;
  size_t pinned_bytes;

  staged_transport_statistics();
};

/* Immutable cache identity for one remote transport epoch. Rolling per-stage
   slot generations are deliberately excluded. */
struct transport_structural_identity {
  static const uint32_t schema_version = 1;
  uint32_t version;
  uint32_t slot_layout_version;
  uint32_t slot_count;
  uint64_t communicator_generation;
  int communicator_rank;
  int communicator_size;
  int tag_ub;
  GpuMpiPolicy requested_policy;
  GpuMpiRoute resolved_route;
  DependencyOverlapPolicy overlap_policy;
  uint64_t wire_signature;
  uint64_t authority_signature;
  uint64_t provider_signature;
  uint64_t device_signature;
  uint64_t owner_map_signature;
  uint64_t arena_layout_signature;
  uint64_t dependency_signature;
  uint64_t signature;

  transport_structural_identity();
};

bool operator==(const transport_structural_identity &a,
                const transport_structural_identity &b);
uint64_t compute_transport_structural_identity_signature(
    const transport_structural_identity &identity);

/* One executable-owned request/event/buffer epoch borrowing the active shared
   communicator context. begin_stage is called after
   zeroing and local gathers have been enqueued. It posts receives first and
   fences remote gather ahead of the caller's local scatters. finish_stage is
   called after those local scatters have been enqueued; it completes the
   staged exchange, publishes remote scatters canonically, and fences the
   compute stream before dependent operations. */
class staged_transport_epoch {
public:
  ~staged_transport_epoch();

  staged_transport_epoch(staged_transport_epoch &&) noexcept;
  staged_transport_epoch &operator=(staged_transport_epoch &&) noexcept;

  bool has_stage(field_type ft) const;
  bool begin_stage(field_type ft, const stream &compute, std::string &why);
  bool finish_stage(field_type ft, const stream &compute, std::string &why);
  [[noreturn]] void abort_published_stage(field_type ft, const char *operation) noexcept;
  bool participate_idle_stage(std::string &why);
  void record_dependency_overlap(size_t interior_launches, size_t boundary_launches);
  bool retire(std::string &why) noexcept;
  const staged_transport_statistics &statistics() const;
  const transport_structural_identity &structural_identity() const;
  bool structural_identity_matches_current(uint64_t device_signature,
                                           GpuMpiPolicy requested_policy,
                                           DependencyOverlapPolicy overlap_policy,
                                           std::string &why) const;

private:
  staged_transport_epoch();
  staged_transport_epoch(const staged_transport_epoch &);
  staged_transport_epoch &operator=(const staged_transport_epoch &);
  struct impl;
  impl *impl_;

  friend std::unique_ptr<staged_transport_epoch>
  create_staged_transport_epoch(const compiled_boundary_artifact &, int, stream *, std::string &,
                                uint64_t, uint64_t, DependencyOverlapPolicy);
};

std::unique_ptr<staged_transport_epoch>
create_staged_transport_epoch(const compiled_boundary_artifact &artifact, int device,
                              stream *communication, std::string &why,
                              uint64_t device_signature = 0,
                              uint64_t dependency_signature = 0,
                              DependencyOverlapPolicy overlap_policy =
                                  DependencyOverlapPolicy::off);
std::unique_ptr<staged_transport_epoch>
create_transport_epoch_with_fallback(compiled_boundary_artifact &artifact, int device,
                                     stream *communication, GpuMpiPolicy agreed_policy,
                                     DependencyOverlapPolicy overlap_policy, bool &fell_back,
                                     std::string &why, uint64_t device_signature = 0,
                                     uint64_t dependency_signature = 0);

namespace testing {
enum class transport_presend_action { poll, enqueue_device_to_host, post_sends };
transport_presend_action transport_presend_action_for_testing(
    GpuMpiRoute route, bool gather_ready, bool staging_ready, bool copies_enqueued);
/* Narrow request seam used to exercise the production direct-transport state
   machine without requiring a CUDA-aware MPI provider.  Request storage is
   still owned by the epoch; production uses the exact MPI calls whenever no
   facade is installed. */
class mpi_request_operations {
public:
  virtual ~mpi_request_operations() {}
  virtual int irecv(void *buffer, size_t bytes, int source, int tag,
                    void *request_storage) = 0;
  virtual int isend(const void *buffer, size_t bytes, int destination, int tag,
                    void *request_storage) = 0;
  virtual int testsome(size_t request_count, void *request_storage, size_t request_stride,
                       int &completed_count, int *completed_indices) = 0;
  virtual int waitall(size_t request_count, void *request_storage, size_t request_stride) = 0;
  virtual bool request_is_null(const void *request_storage) const = 0;
  virtual void clear_request(void *request_storage) = 0;
};
void set_mpi_request_operations_for_testing(mpi_request_operations *operations);
enum class staged_transport_failure_point {
  none,
  direct_validation,
  before_receive_post,
  after_receive_post,
  gather,
  device_to_host,
  before_send_post,
  after_send_post,
  host_to_device,
  scatter
};
void fail_staged_transport_once(staged_transport_failure_point point);
void set_staged_transport_minimum_presend_polls(size_t polls);
} // namespace testing

} // namespace nvidia
} // namespace meep

#endif
