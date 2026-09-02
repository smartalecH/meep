/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  Backend-private MPI context. MPI types never cross the installed/public API.
*/

#ifndef MEEP_BACKEND_MPI_CONTEXT_HPP
#define MEEP_BACKEND_MPI_CONTEXT_HPP

#include "config.h"

#include <stdint.h>
#include <memory>
#include <string>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "backend/transport_plan.hpp"

namespace meep {

struct BackendCommunicatorContext;

struct BackendCommunicatorInfo {
  uint64_t generation;
  int rank;
  int size;
  int tag_ub;
  int thread_level;
  bool thread_main;
  bool provider_query_available;
  bool provider_supports_direct;
  std::string provider;
};

class BackendCommunicatorLease {
public:
  BackendCommunicatorLease();
  ~BackendCommunicatorLease();
  BackendCommunicatorLease(BackendCommunicatorLease &&other) noexcept;
  BackendCommunicatorLease &operator=(BackendCommunicatorLease &&other) noexcept;

  bool valid() const;
  const BackendCommunicatorInfo &info() const;

private:
  BackendCommunicatorLease(const BackendCommunicatorLease &);
  BackendCommunicatorLease &operator=(const BackendCommunicatorLease &);
  std::shared_ptr<BackendCommunicatorContext> context_;
  BackendCommunicatorInfo info_;
  friend bool create_backend_communicator_lease(BackendCommunicatorLease &, std::string &);
  friend bool retire_backend_communicator_lease(BackendCommunicatorLease &, std::string &);
  friend bool collective_validate_remote_halo_agreement(const BackendCommunicatorLease &,
                                                        const RemoteHaloProgram &,
                                                        std::string &);
#ifdef HAVE_MPI
  friend MPI_Comm backend_communicator(const BackendCommunicatorLease &);
#endif
};

uint64_t current_backend_communicator_generation();
#ifdef HAVE_MPI
MPI_Comm current_backend_communicator();
MPI_Comm backend_communicator(const BackendCommunicatorLease &lease);
#endif
bool next_backend_communicator_generation(uint64_t current, uint64_t &next, std::string &why);
bool backend_mpi_thread_ready(std::string &why);
bool query_gpu_aware_mpi_provider(bool &query_available, bool &supports_direct,
                                  std::string &provider, std::string &why);
bool collective_resolve_gpu_mpi_policy(bool local_parse_valid, GpuMpiPolicy local_policy,
                                       bool local_query_available, bool local_direct_support,
                                       GpuMpiPolicy &agreed_policy, GpuMpiRoute &route,
                                       std::string &why);
/* These operations only acquire/release a local borrow of the shared active
   context. The duplicated communicator itself is created and freed by Meep's
   explicit initialization and communicator-transition transactions. */
bool create_backend_communicator_lease(BackendCommunicatorLease &lease, std::string &why);
bool retire_backend_communicator_lease(BackendCommunicatorLease &lease, std::string &why);
bool collective_validate_remote_halo_agreement(const BackendCommunicatorLease &lease,
                                               const RemoteHaloProgram &program,
                                               std::string &why);
void set_backend_communicator_failure_for_testing(const char *point);
const void *backend_communicator_context_identity_for_testing();
size_t backend_communicator_context_use_count_for_testing();

} // namespace meep

#endif
