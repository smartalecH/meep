/* True-MPI synthetic exercise of the PR7.2 staged CUDA transport epoch. */

#include <meep.hpp>

#include "backend/mpi_context.hpp"
#include "backend/nvidia/nvidia_mpi.hpp"
#include "backend/nvidia/runtime.hpp"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>

using namespace meep;

namespace {
int failures = 0;

#define CHECK(condition, message)                                                                  \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      std::fprintf(stderr, "[rank %d] FAIL %s:%d: %s\n", my_rank(), __FILE__, __LINE__, message);  \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

void test_retirement_resolution() {
  bool retired = false;
  CHECK(meep::nvidia::testing::resolve_staged_transport_retirement_for_testing(
            false, false, retired) &&
            retired,
        "finalized local communicator discard was not accepted as retirement");
  retired = false;
  CHECK(!meep::nvidia::testing::resolve_staged_transport_retirement_for_testing(
            false, true, retired) &&
            !retired,
        "failed retirement with a live communicator lease was accepted");
}

std::unique_ptr<meep::nvidia::staged_transport_epoch>
make_live_finalization_epoch(int device, meep::nvidia::stream &communication, std::string &why) {
  const int rank = my_rank(), size = count_processors();
  if (size < 2) return std::unique_ptr<meep::nvidia::staged_transport_epoch>();
  meep::nvidia::compiled_boundary_artifact artifact;
  artifact.wire.version = RemoteHaloProgram::schema_version;
  artifact.wire.communicator_rank = rank;
  artifact.wire.communicator_size = size;
  artifact.wire.communicator_generation = current_backend_communicator_generation();
  artifact.wire.requested_policy = GpuMpiPolicy::staged;
  artifact.wire.resolved_route = GpuMpiRoute::staged;
  artifact.wire.participation = RemoteHaloParticipation{true, 1, 0, 1};
  RemoteHaloStage wire_stage;
  wire_stage.ft = E_stuff;
  wire_stage.signature = compute_remote_halo_stage_signature(wire_stage);
  artifact.wire.stages.push_back(wire_stage);
  artifact.wire.signature = compute_remote_halo_program_signature(artifact.wire);
  meep::nvidia::bound_boundary_stage bound_stage;
  bound_stage.ft = E_stuff;
  bound_stage.receive_slot_bytes = 0;
  bound_stage.send_slot_bytes = 0;
  bound_stage.publication = RemoteHaloPublicationMode::parallel_unique;
  artifact.bound.program_signature = artifact.wire.signature;
  artifact.bound.stages.push_back(bound_stage);
  return meep::nvidia::create_staged_transport_epoch(artifact, device, &communication, why);
}

RemoteHaloMessage make_message(int source, int destination, int size, RemoteHaloDirection direction,
                               size_t elements, Precision precision) {
  RemoteHaloMessage message;
  message.key = RemoteHaloWireKey{source,
                                  destination,
                                  E_stuff,
                                  source,
                                  destination,
                                  29,
                                  uint64_t(source) * uint64_t(size) + uint64_t(destination)};
  message.direction = direction;
  message.local_schedule_ordinal = 0;
  message.phases.push_back(RemoteHaloPhaseSpan{CONNECT_COPY, uint32_t(CONNECT_COPY), 0, elements});
  message.total_elements = elements;
  message.storage_precision = precision;
  message.element_bytes = precision == Precision::f32 ? sizeof(float) : sizeof(double);
  message.wire_bytes = elements * message.element_bytes;
  message.wire_digest = compute_remote_halo_wire_digest(message);
  return message;
}

template <typename T> int run_case(int device) {
  const int rank = my_rank(), size = count_processors();
  if (size < 2) return 0;
  const int previous = (rank + size - 1) % size;
  const int next = (rank + 1) % size;
  const size_t elements = 259;
  const Precision precision = sizeof(T) == sizeof(float) ? Precision::f32 : Precision::f64;

  meep::nvidia::device_scope scope(device);
  meep::nvidia::stream compute, communication;
  meep::nvidia::device_buffer source(elements * sizeof(T), device);
  meep::nvidia::device_buffer target(elements * sizeof(T), device);
  meep::nvidia::pinned_buffer source_host(elements * sizeof(T));
  meep::nvidia::pinned_buffer target_host(elements * sizeof(T));

  meep::nvidia::compiled_boundary_artifact artifact;
  artifact.wire.version = RemoteHaloProgram::schema_version;
  artifact.wire.communicator_rank = rank;
  artifact.wire.communicator_size = size;
  artifact.wire.communicator_generation = current_backend_communicator_generation();
  artifact.wire.requested_policy = GpuMpiPolicy::staged;
  artifact.wire.resolved_route = GpuMpiRoute::staged;
  artifact.wire.participation = RemoteHaloParticipation{true, 1, 2, 1};
  RemoteHaloStage wire_stage;
  wire_stage.ft = E_stuff;
  wire_stage.receives.push_back(
      make_message(previous, rank, size, RemoteHaloDirection::incoming, elements, precision));
  wire_stage.sends.push_back(
      make_message(rank, next, size, RemoteHaloDirection::outgoing, elements, precision));
  wire_stage.signature = compute_remote_halo_stage_signature(wire_stage);
  artifact.wire.stages.push_back(wire_stage);
  artifact.wire.signature = compute_remote_halo_program_signature(artifact.wire);

  meep::nvidia::bound_boundary_stage bound;
  bound.ft = E_stuff;
  bound.receive_slot_bytes = elements * sizeof(T);
  bound.send_slot_bytes = elements * sizeof(T);
  bound.publication = RemoteHaloPublicationMode::parallel_unique;
  bound.canonical_receive_order.push_back(0);
  for (size_t i = 0; i < elements; ++i) {
    bound.gathers.push_back(
        meep::nvidia::boundary_gather_entry{source.opaque_handle(), ptrdiff_t(i), i * sizeof(T)});
    bound.scatters.push_back(meep::nvidia::boundary_scatter_entry{
        target.opaque_handle(), ptrdiff_t(i), NULL, 0, i * sizeof(T), 1.0, 0.0});
  }
  const meep::nvidia::boundary_launch launch{0, elements, elements,
                                             sizeof(T) == sizeof(float)
                                                 ? meep::nvidia::scalar_precision::f32
                                                 : meep::nvidia::scalar_precision::f64};
  const RemoteHaloMessage &receive = artifact.wire.stages[0].receives[0];
  const RemoteHaloMessage &send = artifact.wire.stages[0].sends[0];
  bound.receives.push_back(meep::nvidia::bound_boundary_message{receive.key,
                                                                receive.direction,
                                                                launch,
                                                                receive.wire_bytes,
                                                                {0, receive.wire_bytes},
                                                                {0, receive.wire_bytes}});
  bound.sends.push_back(meep::nvidia::bound_boundary_message{send.key,
                                                             send.direction,
                                                             launch,
                                                             send.wire_bytes,
                                                             {0, send.wire_bytes},
                                                             {0, send.wire_bytes}});
  artifact.bound.program_signature = artifact.wire.signature;
  artifact.bound.stages.push_back(bound);

  std::string why;
  meep::nvidia::compiled_boundary_artifact malformed = artifact;
  if (rank == 0) malformed.bound.program_signature ^= 1;
  std::unique_ptr<meep::nvidia::staged_transport_epoch> rejected =
      meep::nvidia::create_staged_transport_epoch(malformed, device, &communication, why);
  CHECK(!rejected && why.find("authority") != std::string::npos,
        "rank-asymmetric transport authority mismatch did not reject collectively");
  const meep::nvidia::memory_accounting before_failed_create =
      meep::nvidia::current_memory_accounting();
  if (rank == 0)
    meep::nvidia::testing::fail_next(meep::nvidia::testing::failure_point::device_allocate);
  rejected = meep::nvidia::create_staged_transport_epoch(artifact, device, &communication, why);
  CHECK(!rejected, "rank-asymmetric transport allocation failure did not roll back collectively");
  const meep::nvidia::memory_accounting after_failed_create =
      meep::nvidia::current_memory_accounting();
  CHECK(before_failed_create.device_bytes_current == after_failed_create.device_bytes_current &&
            before_failed_create.pinned_bytes_current == after_failed_create.pinned_bytes_current,
        "failed transport creation leaked device or pinned ownership");
  std::unique_ptr<meep::nvidia::staged_transport_epoch> epoch =
      meep::nvidia::create_staged_transport_epoch(artifact, device, &communication, why);
  CHECK(bool(epoch), why.c_str());
  if (!epoch) return failures;
  const meep::nvidia::memory_accounting steady_memory = meep::nvidia::current_memory_accounting();

  const char *fatal_mode = getenv("MEEP_NVIDIA_STAGED_FATAL");
  if (!fatal_mode && getenv("MEEP_NVIDIA_STAGED_FATAL_AFTER_RECV")) fatal_mode = "receive";
  if (fatal_mode) {
    if (!strcmp(fatal_mode, "retire")) {
      if (rank == 0) set_backend_communicator_failure_for_testing("before_retire");
      /* A collectively unretired, still-live communicator is not the
         finalized-MPI discard case. Destruction must remain fatal. */
      epoch.reset();
      CHECK(false, "live communicator retirement failure unexpectedly returned");
      return failures;
    }
    meep::nvidia::testing::staged_transport_failure_point failure =
        meep::nvidia::testing::staged_transport_failure_point::none;
    if (!strcmp(fatal_mode, "receive"))
      failure = meep::nvidia::testing::staged_transport_failure_point::after_receive_post;
    else if (!strcmp(fatal_mode, "gather"))
      failure = meep::nvidia::testing::staged_transport_failure_point::gather;
    else if (!strcmp(fatal_mode, "d2h"))
      failure = meep::nvidia::testing::staged_transport_failure_point::device_to_host;
    else if (!strcmp(fatal_mode, "isend"))
      failure = meep::nvidia::testing::staged_transport_failure_point::after_send_post;
    else {
      CHECK(false, "unknown fatal staged-transport failure mode");
      return failures;
    }
    if (rank == 0) meep::nvidia::testing::fail_staged_transport_once(failure);
    const bool begun = epoch->begin_stage(E_stuff, compute, why);
    if (begun) (void)epoch->finish_stage(E_stuff, compute, why);
    CHECK(false, "published-request failure unexpectedly returned from the fatal path");
    return failures;
  }

  meep::nvidia::testing::set_staged_transport_minimum_presend_polls(32);
  if (rank == 0)
    meep::nvidia::testing::fail_staged_transport_once(
        meep::nvidia::testing::staged_transport_failure_point::before_receive_post);
  CHECK(!epoch->begin_stage(E_stuff, compute, why) && why.find("preflight") != std::string::npos,
        "receive preflight failure did not remain retryable");
  for (int iteration = 0; iteration < 3; ++iteration) {
    T *input = static_cast<T *>(source_host.data());
    T *observed = static_cast<T *>(target_host.data());
    for (size_t i = 0; i < elements; ++i) {
      input[i] = T(rank * 1000 + iteration * 10) + T(i) / T(17);
      observed[i] = T(-1);
    }
    meep::nvidia::copy_host_to_device_async(source, 0, input, elements * sizeof(T), compute);
    meep::nvidia::copy_host_to_device_async(target, 0, observed, elements * sizeof(T), compute);
    CHECK(epoch->begin_stage(E_stuff, compute, why), why.c_str());
    if (iteration == 0)
      CHECK(!epoch->retire(why) && why.find("live request slot") != std::string::npos,
            "transport retirement accepted an occupied request slot");
    CHECK(epoch->finish_stage(E_stuff, compute, why), why.c_str());
    meep::nvidia::copy_device_to_host_async(observed, target, 0, elements * sizeof(T), compute);
    compute.synchronize();
    for (size_t i = 0; i < elements; ++i) {
      const T expected = T(previous * 1000 + iteration * 10) + T(i) / T(17);
      if (observed[i] != expected) {
        CHECK(false, "staged transport payload differs from previous rank");
        break;
      }
    }
  }
  const meep::nvidia::staged_transport_statistics &stats = epoch->statistics();
  CHECK(stats.messages_sent == 3 && stats.messages_received == 3,
        "staged transport message accounting differs");
  CHECK(stats.bytes_sent == 3 * elements * sizeof(T) &&
            stats.bytes_received == 3 * elements * sizeof(T),
        "staged transport byte accounting differs");
  CHECK(stats.device_to_host_calls == 3 && stats.host_to_device_calls == 3 &&
            stats.device_to_host_bytes == 3 * elements * sizeof(T) &&
            stats.host_to_device_bytes == 3 * elements * sizeof(T),
        "staged transport copy accounting differs");
  CHECK(stats.gather_launches == 3 && stats.scatter_launches == 3 &&
            stats.request_completions == 6 && stats.high_water_requests == 2,
        "staged transport kernel/request accounting differs");
  CHECK(stats.direct_bytes == 0 && stats.slot_reuses == 1,
        "staged-only route or two-slot reuse accounting differs");
  CHECK(stats.testsome_polls && stats.waitall_calls == 3,
        "staged progress did not exercise Testsome then Waitall");
  CHECK(stats.testsome_polls >= 96,
        "delayed send-ready fixture did not remain in nonblocking progress");
  const meep::nvidia::memory_accounting after_warmup = meep::nvidia::current_memory_accounting();
  CHECK(steady_memory.device_bytes_current == after_warmup.device_bytes_current &&
            steady_memory.pinned_bytes_current == after_warmup.pinned_bytes_current,
        "staged transport grew device or pinned storage after warmup");
  meep::nvidia::testing::set_staged_transport_minimum_presend_polls(0);
  if (rank == 0)
    meep::nvidia::testing::fail_staged_transport_once(
        meep::nvidia::testing::staged_transport_failure_point::host_to_device);
  CHECK(epoch->begin_stage(E_stuff, compute, why), why.c_str());
  CHECK(!epoch->finish_stage(E_stuff, compute, why),
        "post-completion H2D failure did not fail the whole stage");
  CHECK(!epoch->begin_stage(E_stuff, compute, why) && why.find("poisoned") != std::string::npos,
        "failed published stage left the transport retryable");
  CHECK(epoch->retire(why), why.c_str());

  epoch = meep::nvidia::create_staged_transport_epoch(artifact, device, &communication, why);
  CHECK(bool(epoch), why.c_str());
  if (!epoch) return failures;
  if (rank == 0)
    meep::nvidia::testing::fail_staged_transport_once(
        meep::nvidia::testing::staged_transport_failure_point::scatter);
  CHECK(epoch->begin_stage(E_stuff, compute, why), why.c_str());
  CHECK(!epoch->finish_stage(E_stuff, compute, why),
        "scatter publication failure did not fail the whole stage");
  CHECK(!epoch->begin_stage(E_stuff, compute, why) && why.find("poisoned") != std::string::npos,
        "failed scatter publication left the transport retryable");
  CHECK(epoch->retire(why), why.c_str());
  return failures;
}

void run_idle_np4_case(const std::vector<meep::nvidia::device_properties> &devices) {
  if (count_processors() != 4) return;
  const int rank = my_rank();
  const bool owner = rank < 2;
  const size_t elements = 17;
  std::unique_ptr<meep::nvidia::device_scope> selected;
  std::unique_ptr<meep::nvidia::stream> compute, communication;
  std::unique_ptr<meep::nvidia::device_buffer> source, target;
  meep::nvidia::compiled_boundary_artifact artifact;
  artifact.wire.version = RemoteHaloProgram::schema_version;
  artifact.wire.communicator_rank = rank;
  artifact.wire.communicator_size = 4;
  artifact.wire.communicator_generation = current_backend_communicator_generation();
  artifact.wire.requested_policy = GpuMpiPolicy::staged;
  artifact.wire.resolved_route = GpuMpiRoute::staged;
  artifact.wire.participation =
      RemoteHaloParticipation{owner, owner ? 1u : 0u, owner ? 2u : 0u, owner ? 1u : 0u};
  if (owner) {
    selected.reset(new meep::nvidia::device_scope(rank));
    compute.reset(new meep::nvidia::stream);
    communication.reset(new meep::nvidia::stream);
    source.reset(new meep::nvidia::device_buffer(elements * sizeof(double), rank));
    target.reset(new meep::nvidia::device_buffer(elements * sizeof(double), rank));
    const int peer = 1 - rank;
    RemoteHaloStage wire_stage;
    wire_stage.ft = E_stuff;
    wire_stage.receives.push_back(
        make_message(peer, rank, 4, RemoteHaloDirection::incoming, elements, Precision::f64));
    wire_stage.sends.push_back(
        make_message(rank, peer, 4, RemoteHaloDirection::outgoing, elements, Precision::f64));
    wire_stage.signature = compute_remote_halo_stage_signature(wire_stage);
    artifact.wire.stages.push_back(wire_stage);
    meep::nvidia::bound_boundary_stage bound;
    bound.ft = E_stuff;
    bound.receive_slot_bytes = bound.send_slot_bytes = elements * sizeof(double);
    bound.publication = RemoteHaloPublicationMode::parallel_unique;
    bound.canonical_receive_order.push_back(0);
    for (size_t i = 0; i < elements; ++i) {
      bound.gathers.push_back(meep::nvidia::boundary_gather_entry{
          source->opaque_handle(), ptrdiff_t(i), i * sizeof(double)});
      bound.scatters.push_back(meep::nvidia::boundary_scatter_entry{
          target->opaque_handle(), ptrdiff_t(i), NULL, 0, i * sizeof(double), 1.0, 0.0});
    }
    const meep::nvidia::boundary_launch launch{0, elements, elements,
                                               meep::nvidia::scalar_precision::f64};
    const RemoteHaloMessage &receive = artifact.wire.stages[0].receives[0];
    const RemoteHaloMessage &send = artifact.wire.stages[0].sends[0];
    bound.receives.push_back(meep::nvidia::bound_boundary_message{receive.key,
                                                                  receive.direction,
                                                                  launch,
                                                                  receive.wire_bytes,
                                                                  {0, receive.wire_bytes},
                                                                  {0, receive.wire_bytes}});
    bound.sends.push_back(meep::nvidia::bound_boundary_message{send.key,
                                                               send.direction,
                                                               launch,
                                                               send.wire_bytes,
                                                               {0, send.wire_bytes},
                                                               {0, send.wire_bytes}});
    artifact.bound.stages.push_back(bound);
  }
  artifact.wire.signature = compute_remote_halo_program_signature(artifact.wire);
  artifact.bound.program_signature = artifact.wire.signature;
  std::string why;
  std::unique_ptr<meep::nvidia::staged_transport_epoch> epoch =
      meep::nvidia::create_staged_transport_epoch(artifact, owner ? rank : -1,
                                                  owner ? communication.get() : NULL, why);
  CHECK(bool(epoch), why.c_str());
  if (epoch && owner) {
    std::vector<double> input(elements), observed(elements, -1.0);
    for (size_t i = 0; i < elements; ++i)
      input[i] = rank * 100.0 + double(i);
    meep::nvidia::copy_host_to_device_async(*source, 0, input.data(), elements * sizeof(double),
                                            *compute);
    meep::nvidia::copy_host_to_device_async(*target, 0, observed.data(), elements * sizeof(double),
                                            *compute);
    CHECK(epoch->begin_stage(E_stuff, *compute, why), why.c_str());
    CHECK(epoch->finish_stage(E_stuff, *compute, why), why.c_str());
    meep::nvidia::copy_device_to_host_async(observed.data(), *target, 0, elements * sizeof(double),
                                            *compute);
    compute->synchronize();
    for (size_t i = 0; i < elements; ++i)
      if (observed[i] != (1 - rank) * 100.0 + double(i)) {
        CHECK(false, "idle-rank fixture payload differs");
        break;
      }
  }
  else if (epoch) {
    CHECK(epoch->participate_idle_stage(why), why.c_str());
    CHECK(epoch->participate_idle_stage(why), why.c_str());
  }
  CHECK(epoch && epoch->retire(why), why.c_str());
  (void)devices;
}

} // namespace

int main(int argc, char **argv) {
#ifdef HAVE_MPI
  int provided = MPI_THREAD_SINGLE;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS ||
      provided < MPI_THREAD_FUNNELED)
    return 2;
#endif
  initialize mpi(argc, argv);
  try {
    test_retirement_resolution();
    const std::vector<meep::nvidia::device_properties> devices = meep::nvidia::enumerate_devices();
    CHECK(!devices.empty(), "no CUDA device is available");
    if (!devices.empty()) {
      const int device = my_rank() % int(devices.size());
      run_case<double>(device);
      run_case<float>(device);
      run_idle_np4_case(devices);
      if (count_processors() == 4) {
        divide_parallel_processes(2);
        run_case<double>(my_rank() % int(devices.size()));
        end_divide_parallel();
      }
    }

#ifdef HAVE_MPI
    /* Keep a real device-owning transport epoch alive across externally-owned
       MPI finalization. Its stage owns CUDA events and a duplicated backend
       communicator, so destruction exercises the production teardown path. */
    std::unique_ptr<meep::nvidia::device_scope> final_scope;
    std::unique_ptr<meep::nvidia::stream> final_communication;
    std::unique_ptr<meep::nvidia::staged_transport_epoch> final_epoch;
    if (!devices.empty() && count_processors() >= 2) {
      const int device = my_rank() % int(devices.size());
      final_scope.reset(new meep::nvidia::device_scope(device));
      final_communication.reset(new meep::nvidia::stream);
      std::string why;
      final_epoch = make_live_finalization_epoch(device, *final_communication, why);
      CHECK(bool(final_epoch), why.c_str());
    }
#endif

    const int total = sum_to_all(failures);
    if (my_rank() == 0)
      std::printf("nvidia_staged_transport: %s (%d failures)\n", total ? "FAIL" : "PASS", total);
#ifdef HAVE_MPI
    const int finalize_result = MPI_Finalize();
    if (finalize_result != MPI_SUCCESS) return 3;
    /* This must neither abort nor report a failed retirement: finalized MPI
       discarded the duplicated lease locally and no live request exists. */
    final_epoch.reset();
#endif
    return total ? 1 : 0;
  }
  catch (const std::exception &error) {
    std::fprintf(stderr, "[rank %d] exception: %s\n", my_rank(), error.what());
    ++failures;
  }
#ifdef HAVE_MPI
  const int total = sum_to_all(failures);
  if (MPI_Finalize() != MPI_SUCCESS) return 3;
  return total ? 1 : 0;
#else
  return failures ? 1 : 0;
#endif
}
