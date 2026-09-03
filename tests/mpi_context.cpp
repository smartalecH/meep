/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include <cstdio>
#include <limits>
#include <string>
#include <thread>

#include <meep.hpp>

#include "config.h"
#include "backend/backend.hpp"
#include "backend/mpi_context.hpp"

using namespace meep;

namespace {
int failures = 0;

#define CHECK(condition, message)                                                                 \
  do {                                                                                            \
    if (!(condition)) {                                                                           \
      std::fprintf(stderr, "[rank %d] FAIL %s:%d: %s\n", my_rank(), __FILE__, __LINE__,         \
                   message);                                                                      \
      ++failures;                                                                                 \
    }                                                                                             \
  } while (0)

RemoteHaloMessage message(bool outgoing, int rank) {
  RemoteHaloMessage result;
  result.key = RemoteHaloWireKey{0, 1, E_stuff, 0, 1, 5, 2};
  result.direction = outgoing ? RemoteHaloDirection::outgoing : RemoteHaloDirection::incoming;
  result.local_schedule_ordinal = outgoing ? 9 : 0;
  result.phases.push_back(RemoteHaloPhaseSpan{CONNECT_COPY, uint32_t(CONNECT_COPY), 0, 2});
  result.total_elements = 2;
  result.storage_precision = Precision::f64;
  result.element_bytes = sizeof(double);
  result.wire_bytes = 2 * sizeof(double);
  result.wire_digest = compute_remote_halo_wire_digest(result);
  (void)rank;
  return result;
}

void test_lease_and_agreement() {
  std::string why;
  CHECK(backend_mpi_thread_ready(why), why.c_str());
  BackendCommunicatorLease lease;
  CHECK(create_backend_communicator_lease(lease, why), why.c_str());
  CHECK(lease.valid(), "communicator lease was not published");
  if (!lease.valid()) return;
  BackendCommunicatorLease sibling;
  CHECK(create_backend_communicator_lease(sibling, why), why.c_str());
  CHECK(sibling.valid() && backend_communicator_context_use_count_for_testing() >= 3,
        "communicator leases did not share the active context");
  CHECK(lease.info().generation == current_backend_communicator_generation(),
        "lease generation is stale at creation");
  CHECK(lease.info().rank == my_rank() && lease.info().size == count_processors(),
        "lease rank/size disagree with active communicator");
#ifdef HAVE_MPI
  int comparison = MPI_UNEQUAL;
  CHECK(current_backend_communicator() != MPI_COMM_NULL,
        "current communicator accessor returned MPI_COMM_NULL");
  CHECK(backend_communicator(lease) != MPI_COMM_NULL,
        "bound communicator accessor returned MPI_COMM_NULL");
  CHECK(MPI_Comm_compare(current_backend_communicator(), backend_communicator(lease),
                         &comparison) == MPI_SUCCESS &&
            (comparison == MPI_CONGRUENT || comparison == MPI_IDENT),
        "duplicated lease is not congruent with the active communicator");
#endif

  RemoteHaloProgram program;
  program.version = RemoteHaloProgram::schema_version;
  program.communicator_rank = lease.info().rank;
  program.communicator_size = lease.info().size;
  program.communicator_generation = lease.info().generation;
  program.requested_policy = GpuMpiPolicy::staged;
  program.resolved_route = GpuMpiRoute::staged;
  const bool owner = lease.info().size < 2 || lease.info().rank < 2;
  program.participation = RemoteHaloParticipation{owner, owner ? 1u : 0u,
                                                  owner && lease.info().size >= 2 ? 1u : 0u,
                                                  owner ? 1u : 0u};
  if (lease.info().size >= 2 && lease.info().rank < 2) {
    RemoteHaloStage stage;
    stage.ft = E_stuff;
    if (lease.info().rank == 0)
      stage.sends.push_back(message(true, lease.info().rank));
    else
      stage.receives.push_back(message(false, lease.info().rank));
    stage.signature = compute_remote_halo_stage_signature(stage);
    program.stages.push_back(stage);
  }
  program.signature = compute_remote_halo_program_signature(program);
  if (lease.info().size > 1) {
    RemoteHaloProgram invalid = program;
    if (lease.info().rank == lease.info().size - 1) {
      invalid.participation.device_owner = false;
      invalid.participation.cuda_required_operations = 1;
      invalid.signature = compute_remote_halo_program_signature(invalid);
    }
    CHECK(!collective_validate_remote_halo_agreement(lease, invalid, why),
          "asymmetric invalid idle-rank participation was accepted");
  }
  CHECK(collective_validate_remote_halo_agreement(lease, program, why), why.c_str());

  GpuMpiPolicy agreed = GpuMpiPolicy::automatic;
  GpuMpiRoute route = GpuMpiRoute::direct;
  CHECK(collective_resolve_gpu_mpi_policy(true, GpuMpiPolicy::staged, false, false,
                                          agreed, route, why) &&
            agreed == GpuMpiPolicy::staged && route == GpuMpiRoute::staged,
        why.c_str());
  if (count_processors() > 1) {
    const GpuMpiPolicy disagree = my_rank() == 0 ? GpuMpiPolicy::staged
                                                 : GpuMpiPolicy::automatic;
    CHECK(!collective_resolve_gpu_mpi_policy(true, disagree, false, false, agreed, route, why),
          "asymmetric requested policy was accepted");
  }
  CHECK(retire_backend_communicator_lease(lease, why), why.c_str());
  CHECK(!lease.valid(), "communicator lease remained valid after retirement");
  CHECK(sibling.valid(), "releasing one borrower invalidated a sibling lease");
  CHECK(retire_backend_communicator_lease(sibling, why), why.c_str());
}

void test_collective_context_rollback() {
#ifdef HAVE_MPI
  if (count_processors() < 2) return;
  const uint64_t generation = current_backend_communicator_generation();
  const void *identity = backend_communicator_context_identity_for_testing();
  if (my_rank() == 0) set_backend_communicator_failure_for_testing("before_dup");
  bool rejected = false;
  try { (void)divide_parallel_processes(2); }
  catch (const std::runtime_error &) { rejected = true; }
  CHECK(rejected, "asymmetric communicator preflight failure did not reconcile");
  CHECK(current_backend_communicator_generation() == generation &&
            backend_communicator_context_identity_for_testing() == identity,
        "failed communicator preflight replaced the active context");
  set_backend_communicator_failure_for_testing(NULL);
  if (my_rank() == 0) set_backend_communicator_failure_for_testing("after_dup");
  rejected = false;
  try { (void)divide_parallel_processes(2); }
  catch (const std::runtime_error &) { rejected = true; }
  CHECK(rejected, "asymmetric post-duplication failure did not roll back collectively");
  CHECK(current_backend_communicator_generation() == generation &&
            backend_communicator_context_identity_for_testing() == identity,
        "failed post-duplication candidate replaced the active context");
  set_backend_communicator_failure_for_testing(NULL);
  if (my_rank() == 0) set_backend_communicator_failure_for_testing("before_retire");
  rejected = false;
  try { (void)divide_parallel_processes(2); }
  catch (const std::runtime_error &) { rejected = true; }
  CHECK(rejected, "asymmetric context retirement failure did not reject transition");
  CHECK(current_backend_communicator_generation() == generation &&
            backend_communicator_context_identity_for_testing() == identity,
        "failed context retirement changed the active context");
  set_backend_communicator_failure_for_testing(NULL);
#endif
}

void test_thread_and_generation_helpers() {
  std::string why;
  uint64_t next = 0;
  CHECK(next_backend_communicator_generation(7, next, why) && next == 8,
        "checked communicator generation did not increment");
  CHECK(!next_backend_communicator_generation(std::numeric_limits<uint64_t>::max(), next, why),
        "communicator generation overflow was accepted");

  bool worker_ready = true;
  bool worker_mutation_rejected = false;
  std::thread worker([&]() {
    std::string worker_why;
    worker_ready = backend_mpi_thread_ready(worker_why);
#ifdef HAVE_MPI
    try { begin_global_communications(); }
    catch (const std::exception &) { worker_mutation_rejected = true; }
#endif
  });
  worker.join();
#ifdef HAVE_MPI
  CHECK(!worker_ready, "non-main thread passed MPI transport admission");
  CHECK(worker_mutation_rejected, "non-main thread changed the public communicator state");
#else
  CHECK(worker_ready, "non-MPI build unexpectedly imposed an MPI thread restriction");
#endif
}

void test_frontend_json_allgather() {
  const std::vector<std::string> empty = active_communicator_allgather_json_records("");
  CHECK(empty.size() == size_t(count_processors()),
        "empty frontend JSON gather returned the wrong rank count");
  for (const std::string &value : empty)
    CHECK(value.empty(), "empty frontend JSON gather changed the payload");

  const std::string payload = std::string("{\"rank\":") + std::to_string(my_rank()) + "}";
  const std::vector<std::string> gathered =
      active_communicator_allgather_json_records(payload);
  CHECK(gathered.size() == size_t(count_processors()),
        "frontend JSON gather returned the wrong rank count");
  for (int rank = 0; rank < count_processors(); ++rank)
    CHECK(gathered[size_t(rank)] ==
              std::string("{\"rank\":") + std::to_string(rank) + "}",
          "frontend JSON gather did not preserve rank order or bytes");

  if (count_processors() > 1) {
    bool rejected = false;
    try {
      (void)active_communicator_allgather_json_records(
          payload, my_rank() != count_processors() - 1, "injected invalid JSON payload");
    }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected, "asymmetric invalid frontend JSON payload was not rejected collectively");
  }

  for (int point = 1; point <= 3; ++point) {
    backend_set_runtime_report_failure_for_testing(0, point);
    bool rejected = false;
    try { (void)active_communicator_allgather_json_records(payload); }
    catch (const std::exception &) { rejected = true; }
    CHECK(rejected, "asymmetric frontend JSON allocation failure was not reconciled");
    backend_set_runtime_report_failure_for_testing(-1, 0);
  }

#ifdef HAVE_MPI
  bool worker_rejected = false;
  std::thread worker([&]() {
    try { (void)active_communicator_allgather_json_records(payload); }
    catch (const std::runtime_error &) { worker_rejected = true; }
  });
  worker.join();
  CHECK(worker_rejected, "frontend JSON gather accepted a non-main MPI thread");
#endif
}

void test_split_transitions() {
  if (count_processors() < 2) return;
  const uint64_t before = current_backend_communicator_generation();
  const void *world_context = backend_communicator_context_identity_for_testing();
  divide_parallel_processes(2);
  CHECK(current_backend_communicator_generation() == before + 1,
        "divide did not advance communicator generation once");
  CHECK(backend_communicator_context_identity_for_testing() != world_context,
        "divide retained the old active communicator context");
  const std::vector<std::string> split_records =
      active_communicator_allgather_json_records(std::to_string(my_rank()));
  CHECK(split_records.size() == size_t(count_processors()),
        "frontend JSON gather used the wrong split communicator");
  {
    std::string why;
    BackendCommunicatorLease stale;
    CHECK(create_backend_communicator_lease(stale, why), why.c_str());
    RemoteHaloProgram program;
    program.version = RemoteHaloProgram::schema_version;
    program.communicator_rank = stale.info().rank;
    program.communicator_size = stale.info().size;
    program.communicator_generation = stale.info().generation;
    program.requested_policy = GpuMpiPolicy::staged;
    program.resolved_route = GpuMpiRoute::staged;
    program.participation = RemoteHaloParticipation{false, 0, 0, 0};
    program.signature = compute_remote_halo_program_signature(program);
    begin_global_communications();
    CHECK(!stale.valid(), "communicator transition did not invalidate an old borrower");
    CHECK(!collective_validate_remote_halo_agreement(stale, program, why),
          "stale lease/program remained usable after a communicator transition");
    end_global_communications();
    CHECK(retire_backend_communicator_lease(stale, why), why.c_str());
  }
  test_lease_and_agreement();
  const uint64_t divided = current_backend_communicator_generation();
  const void *divided_context = backend_communicator_context_identity_for_testing();
  begin_global_communications();
  CHECK(current_backend_communicator_generation() == divided + 1,
        "begin_global did not advance communicator generation");
  CHECK(backend_communicator_context_identity_for_testing() != divided_context,
        "begin_global retained the divided communicator context");
  const void *global_context = backend_communicator_context_identity_for_testing();
  end_global_communications();
  CHECK(current_backend_communicator_generation() == divided + 2,
        "end_global did not advance communicator generation");
  CHECK(backend_communicator_context_identity_for_testing() != global_context,
        "end_global retained the global communicator context");
  end_divide_parallel();
  CHECK(current_backend_communicator_generation() == divided + 3,
        "end_divide did not advance communicator generation");

  const uint64_t repeated = current_backend_communicator_generation();
  divide_parallel_processes(2);
  divide_parallel_processes(2);
  CHECK(current_backend_communicator_generation() == repeated + 2,
        "repeated divide did not commit exactly one generation per transition");
  end_divide_parallel();
}

} // namespace

int main(int argc, char **argv) {
  int total = 0;
  int saved_rank = 0;
  {
  initialize mpi(argc, argv);
  saved_rank = my_rank();
#ifdef HAVE_MPI
  bool query_available = false, supports_direct = false;
  std::string provider, provider_why;
  CHECK(query_gpu_aware_mpi_provider(query_available, supports_direct, provider, provider_why),
        provider_why.c_str());
  CHECK(!provider.empty(), "MPI provider identity is empty");
#ifdef HAVE_MPIX_QUERY_CUDA_SUPPORT
  CHECK(query_available, "configured MPIX CUDA-support query was not used at runtime");
#else
  CHECK(!query_available, "runtime reported a provider query absent from configure results");
#endif
#endif
  test_thread_and_generation_helpers();
  test_frontend_json_allgather();
  test_lease_and_agreement();
  test_collective_context_rollback();
  test_split_transitions();
  total = sum_to_all(failures);
  }
#ifdef HAVE_MPI
  bool finalized_rejected = false;
  try { (void)active_communicator_allgather_json_records("{}"); }
  catch (const std::runtime_error &) { finalized_rejected = true; }
  if (!finalized_rejected) ++total;
#endif
  if (saved_rank == 0)
    std::printf("mpi_context: %s (%d failures)\n", total ? "FAIL" : "PASS", total);
  return total ? 1 : 0;
}
