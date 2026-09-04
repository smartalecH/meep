/* True-MPI synthetic exercise of the NVIDIA-compatible staged GPU transport epoch. */

#include <meep.hpp>

#include "backend/mpi_context.hpp"
#include "backend/nvidia/nvidia_mpi.hpp"
#include "backend/nvidia/runtime.hpp"

#if defined(HAVE_MPI) && defined(MEEP_HIP_PORTABILITY)
#include <hip/hip_runtime_api.h>
#include <glob.h>
#include <sched.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

void test_route_state_machine(bool staged_only) {
  using meep::nvidia::testing::transport_presend_action;
  using meep::nvidia::testing::transport_presend_action_for_testing;
  if (!staged_only) {
    CHECK(transport_presend_action_for_testing(GpuMpiRoute::direct, false, false, false) ==
              transport_presend_action::poll,
          "direct transport posted a send before the producer event");
    CHECK(transport_presend_action_for_testing(GpuMpiRoute::direct, true, false, true) ==
              transport_presend_action::post_sends,
          "direct transport did not post from the device arena after the producer event");
  }
  CHECK(transport_presend_action_for_testing(GpuMpiRoute::staged, true, false, false) ==
            transport_presend_action::enqueue_device_to_host,
        "staged transport omitted its device-to-host transition");
  CHECK(transport_presend_action_for_testing(GpuMpiRoute::staged, true, false, true) ==
            transport_presend_action::poll,
        "staged transport posted before its host copy completed");
  CHECK(transport_presend_action_for_testing(GpuMpiRoute::staged, true, true, true) ==
            transport_presend_action::post_sends,
        "staged transport did not post after its host copy completed");
}

#ifdef HAVE_MPI
class direct_request_mock : public meep::nvidia::testing::mpi_request_operations {
public:
  int device_;
  struct request_record {
    void *storage;
    void *buffer;
    size_t bytes;
    bool receive;
    bool active;
  };

  explicit direct_request_mock(int device)
      : device_(device), irecvs(0), isends(0), testsome_calls(0), waitalls(0), clears(0),
        polls_since_receive(0), pointers_are_device(true), delayed_send_window_armed(false),
        delayed_send_window_expected(0), delayed_send_window_sends(0), delayed_send_window_polls(0),
        delayed_send_window_premature_sends(0), copies_completed(0) {}

  void arm_delayed_send_window(size_t expected_sends) {
    delayed_send_window_armed = true;
    delayed_send_window_expected = expected_sends;
    delayed_send_window_sends = 0;
    delayed_send_window_polls = 0;
    delayed_send_window_premature_sends = 0;
  }

  int irecv(void *buffer, size_t bytes, int, int, void *request_storage) override {
    records.push_back(request_record{request_storage, buffer, bytes, true, true});
    request_slots.insert(request_storage);
    ++irecvs;
    polls_since_receive = 0;
    pointers_are_device = pointers_are_device && is_device_pointer(buffer);
    return MPI_SUCCESS;
  }

  int isend(const void *buffer, size_t bytes, int, int, void *request_storage) override {
    records.push_back(
        request_record{request_storage, const_cast<void *>(buffer), bytes, false, true});
    request_slots.insert(request_storage);
    ++isends;
    if (delayed_send_window_armed) {
      ++delayed_send_window_sends;
      if (polls_since_receive < 32) ++delayed_send_window_premature_sends;
      if (delayed_send_window_sends == delayed_send_window_expected)
        delayed_send_window_armed = false;
    }
    pointers_are_device = pointers_are_device && is_device_pointer(buffer);
    return MPI_SUCCESS;
  }

  int testsome(size_t, void *, size_t, int &completed_count, int *) override {
    ++testsome_calls;
    ++polls_since_receive;
    if (delayed_send_window_armed) ++delayed_send_window_polls;
    completed_count = 0;
    return MPI_SUCCESS;
  }

  int waitall(size_t request_count, void *request_storage, size_t request_stride) override {
    ++waitalls;
    request_record *receive = NULL, *send = NULL;
    unsigned char *base = static_cast<unsigned char *>(request_storage);
    for (size_t i = 0; i < request_count; ++i) {
      void *slot = base + i * request_stride;
      for (request_record &record : records)
        if (record.storage == slot && record.active) {
          if (record.receive)
            receive = &record;
          else
            send = &record;
        }
    }
    if (!receive || !send || receive->bytes != send->bytes) return MPI_ERR_OTHER;
    if (!meep::nvidia::testing::copy_opaque_device_to_device_for_testing(
            receive->buffer, send->buffer, send->bytes, device_))
      return MPI_ERR_OTHER;
    receive->active = false;
    send->active = false;
    ++copies_completed;
    return MPI_SUCCESS;
  }

  bool request_is_null(const void *request_storage) const override {
    for (const request_record &record : records)
      if (record.storage == request_storage && record.active) return false;
    return true;
  }

  void clear_request(void *request_storage) override {
    for (request_record &record : records)
      if (record.storage == request_storage) record.active = false;
    ++clears;
  }

  void verify(const char *precision_name) const {
    CHECK(pointers_are_device, "direct request facade did not receive device arena pointers");
    CHECK(!delayed_send_window_armed && delayed_send_window_sends == delayed_send_window_expected &&
              delayed_send_window_expected == 3 && delayed_send_window_premature_sends == 0,
          "direct request facade observed Isend before gather readiness");
    CHECK(delayed_send_window_polls >= 96,
          "direct request facade did not poll Testsome through all delayed sends");
    CHECK(irecvs >= 4 && irecvs == isends && waitalls == isends && copies_completed == waitalls,
          "direct request facade did not exercise complete Irecv/Isend/Wait transitions");
    CHECK(testsome_calls >= 96,
          "direct request facade did not exercise delayed nonblocking Test progress");
    CHECK(request_slots.size() >= 4,
          "direct request facade did not exercise both request slots and reuse");
    CHECK(clears >= 2 * waitalls,
          "direct request facade did not clear completed receive/send requests");
    (void)precision_name;
  }

  size_t irecvs, isends, testsome_calls, waitalls, clears, polls_since_receive;
  bool pointers_are_device, delayed_send_window_armed;
  size_t delayed_send_window_expected, delayed_send_window_sends, delayed_send_window_polls,
      delayed_send_window_premature_sends;
  size_t copies_completed;
  std::vector<request_record> records;
  std::set<void *> request_slots;

private:
  static bool is_device_pointer(const void *pointer) {
    return meep::nvidia::testing::opaque_pointer_is_device_for_testing(pointer);
  }
};

class request_operations_scope {
public:
  explicit request_operations_scope(meep::nvidia::testing::mpi_request_operations *operations) {
    meep::nvidia::testing::set_mpi_request_operations_for_testing(operations);
  }
  ~request_operations_scope() {
    meep::nvidia::testing::set_mpi_request_operations_for_testing(NULL);
  }
};
#endif

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

template <typename T>
int run_case(int device, bool direct = false, bool direct_mock_local_payload = false,
             bool staged_only = false) {
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
  artifact.wire.requested_policy = direct ? GpuMpiPolicy::direct : GpuMpiPolicy::staged;
  artifact.wire.resolved_route = direct ? GpuMpiRoute::direct : GpuMpiRoute::staged;
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
  if (!direct && !staged_only) {
    const DependencyOverlapPolicy fallback_policies[] = {DependencyOverlapPolicy::off,
                                                         DependencyOverlapPolicy::automatic};
    for (DependencyOverlapPolicy overlap : fallback_policies)
      for (int failure_kind = 0; failure_kind < 2; ++failure_kind) {
        meep::nvidia::compiled_boundary_artifact fallback_artifact = artifact;
        fallback_artifact.wire.requested_policy = GpuMpiPolicy::automatic;
        fallback_artifact.wire.resolved_route = GpuMpiRoute::direct;
        fallback_artifact.wire.signature =
            compute_remote_halo_program_signature(fallback_artifact.wire);
        fallback_artifact.bound.program_signature = fallback_artifact.wire.signature;
        if (rank == 0) {
          if (failure_kind == 0)
            meep::nvidia::testing::fail_next(meep::nvidia::testing::failure_point::device_allocate);
          else
            meep::nvidia::testing::fail_staged_transport_once(
                meep::nvidia::testing::staged_transport_failure_point::direct_validation);
        }
        bool fell_back = false;
        std::unique_ptr<meep::nvidia::staged_transport_epoch> fallback_epoch =
            meep::nvidia::create_transport_epoch_with_fallback(
                fallback_artifact, device, &communication, GpuMpiPolicy::automatic, overlap,
                fell_back, why);
        CHECK(bool(fallback_epoch) && fell_back,
              "auto direct creation failure did not fall back collectively");
        CHECK(fallback_artifact.wire.resolved_route == GpuMpiRoute::staged &&
                  fallback_artifact.wire.signature ==
                      compute_remote_halo_program_signature(fallback_artifact.wire) &&
                  fallback_artifact.lowered.program_signature == fallback_artifact.wire.signature &&
                  fallback_artifact.lowered.authority_signature ==
                      compute_remote_lowered_authority_signature(
                          fallback_artifact.lowered.program_signature,
                          fallback_artifact.lowered.storage_signature) &&
                  fallback_artifact.bound.program_signature == fallback_artifact.wire.signature &&
                  fallback_artifact.bound.authority_signature ==
                      fallback_artifact.lowered.authority_signature,
              "fallback published stale route or signature authority");
        if (fallback_epoch) {
          const meep::nvidia::staged_transport_statistics &fallback_stats =
              fallback_epoch->statistics();
          CHECK(fallback_stats.pinned_bytes != 0 && fallback_stats.direct_bytes == 0 &&
                    fallback_stats.overlap_stages == 0,
                "fallback statistics did not match the published staged route");
          CHECK(fallback_epoch->retire(why), why.c_str());
        }
      }

    for (int failure_kind = 0; failure_kind < 2; ++failure_kind) {
      meep::nvidia::compiled_boundary_artifact required_artifact = artifact;
      required_artifact.wire.requested_policy = GpuMpiPolicy::automatic;
      required_artifact.wire.resolved_route = GpuMpiRoute::direct;
      required_artifact.wire.signature =
          compute_remote_halo_program_signature(required_artifact.wire);
      required_artifact.bound.program_signature = required_artifact.wire.signature;
      const meep::nvidia::memory_accounting before = meep::nvidia::current_memory_accounting();
      if (rank == 0) {
        if (failure_kind == 0)
          meep::nvidia::testing::fail_next(meep::nvidia::testing::failure_point::device_allocate);
        else
          meep::nvidia::testing::fail_staged_transport_once(
              meep::nvidia::testing::staged_transport_failure_point::direct_validation);
      }
      bool fell_back = false;
      std::unique_ptr<meep::nvidia::staged_transport_epoch> required_epoch =
          meep::nvidia::create_transport_epoch_with_fallback(
              required_artifact, device, &communication, GpuMpiPolicy::automatic,
              DependencyOverlapPolicy::required, fell_back, why);
      CHECK(!required_epoch && !fell_back &&
                required_artifact.wire.resolved_route == GpuMpiRoute::direct &&
                required_artifact.wire.signature ==
                    compute_remote_halo_program_signature(required_artifact.wire),
            "required overlap silently downgraded after direct creation failure");
      const meep::nvidia::memory_accounting after = meep::nvidia::current_memory_accounting();
      CHECK(before.device_bytes_current == after.device_bytes_current &&
                before.pinned_bytes_current == after.pinned_bytes_current,
            "required-overlap direct creation failure leaked transport ownership");
    }
  }
  meep::nvidia::compiled_boundary_artifact malformed = artifact;
  if (rank == 0) malformed.bound.program_signature ^= 1;
  std::unique_ptr<meep::nvidia::staged_transport_epoch> rejected =
      meep::nvidia::create_staged_transport_epoch(malformed, device, &communication, why);
  CHECK(!rejected && why.find("authority") != std::string::npos,
        "rank-asymmetric transport authority mismatch did not reject collectively");
  const meep::nvidia::memory_accounting before_failed_create =
      meep::nvidia::current_memory_accounting();
  const size_t context_users_before_failed_create =
      backend_communicator_context_use_count_for_testing();
  if (rank == 0)
    meep::nvidia::testing::fail_next(meep::nvidia::testing::failure_point::device_allocate);
  rejected = meep::nvidia::create_staged_transport_epoch(artifact, device, &communication, why);
  CHECK(!rejected, "rank-asymmetric transport allocation failure did not roll back collectively");
  const meep::nvidia::memory_accounting after_failed_create =
      meep::nvidia::current_memory_accounting();
  CHECK(before_failed_create.device_bytes_current == after_failed_create.device_bytes_current &&
            before_failed_create.pinned_bytes_current == after_failed_create.pinned_bytes_current,
        "failed transport creation leaked device or pinned ownership");
  CHECK(backend_communicator_context_use_count_for_testing() == context_users_before_failed_create,
        "failed transport candidate leaked a communicator-context borrow");
  std::unique_ptr<meep::nvidia::staged_transport_epoch> epoch =
      meep::nvidia::create_staged_transport_epoch(artifact, device, &communication, why,
                                                  UINT64_C(0x1234), UINT64_C(0x5678));
  CHECK(bool(epoch), why.c_str());
  if (!epoch) return failures;
  const meep::nvidia::transport_structural_identity &identity = epoch->structural_identity();
  CHECK(identity.version == meep::nvidia::transport_structural_identity::schema_version &&
            identity.slot_count == 2 && identity.slot_layout_version == 1 &&
            identity.communicator_generation == current_backend_communicator_generation() &&
            identity.wire_signature == artifact.wire.signature &&
            identity.device_signature == UINT64_C(0x1234) &&
            identity.dependency_signature == UINT64_C(0x5678) &&
            identity.signature ==
                meep::nvidia::compute_transport_structural_identity_signature(identity),
        "transport structural identity is incomplete or stale at creation");
  meep::nvidia::transport_structural_identity changed = identity;
  changed.requested_policy = meep::GpuMpiPolicy::automatic;
  CHECK(!(changed == identity), "transport policy mutation preserved structural identity");
  changed = identity;
  changed.overlap_policy = meep::DependencyOverlapPolicy::required;
  CHECK(!(changed == identity), "overlap policy mutation preserved structural identity");
  changed = identity;
  ++changed.provider_signature;
  CHECK(!(changed == identity), "MPI provider mutation preserved structural identity");
  const meep::nvidia::memory_accounting steady_memory = meep::nvidia::current_memory_accounting();

  const char *fatal_mode = getenv("MEEP_NVIDIA_STAGED_FATAL");
  if (!fatal_mode && getenv("MEEP_NVIDIA_STAGED_FATAL_AFTER_RECV")) fatal_mode = "receive";
  if (fatal_mode) {
    if (!strcmp(fatal_mode, "retire")) {
      const size_t before = backend_communicator_context_use_count_for_testing();
      epoch.reset();
      CHECK(backend_communicator_context_use_count_for_testing() + 1 == before,
            "local transport destruction retained its communicator-context borrow");
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
    if (direct_mock_local_payload) epoch->record_dependency_overlap(3, 2);
    meep::nvidia::copy_device_to_host_async(observed, target, 0, elements * sizeof(T), compute);
    compute.synchronize();
    for (size_t i = 0; i < elements; ++i) {
      const int payload_rank = direct_mock_local_payload ? rank : previous;
      const T expected = T(payload_rank * 1000 + iteration * 10) + T(i) / T(17);
      if (observed[i] != expected) {
        std::fprintf(stderr,
                     "[rank %d] transport payload detail direct=%d mock=%d bytes=%zu "
                     "iteration=%d index=%zu observed=%.17g expected=%.17g\n",
                     rank, direct ? 1 : 0, direct_mock_local_payload ? 1 : 0, sizeof(T), iteration,
                     i, double(observed[i]), double(expected));
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
  CHECK(direct ? (stats.device_to_host_calls == 0 && stats.host_to_device_calls == 0 &&
                  stats.device_to_host_bytes == 0 && stats.host_to_device_bytes == 0)
               : (stats.device_to_host_calls == 3 && stats.host_to_device_calls == 3 &&
                  stats.device_to_host_bytes == 3 * elements * sizeof(T) &&
                  stats.host_to_device_bytes == 3 * elements * sizeof(T)),
        "transport copy accounting differs from the selected route");
  CHECK(stats.gather_launches == 3 && stats.scatter_launches == 3 &&
            stats.request_completions == 6 && stats.high_water_requests == 2,
        "staged transport kernel/request accounting differs");
  CHECK(stats.direct_bytes == (direct ? 6 * elements * sizeof(T) : 0) && stats.slot_reuses == 1,
        "route byte or two-slot reuse accounting differs");
  CHECK(direct ? stats.pinned_bytes == 0 : stats.pinned_bytes > 0,
        "transport pinned-storage accounting differs from the selected route");
  CHECK(direct_mock_local_payload
            ? (stats.overlap_stages == 3 && stats.overlap_interior_launches == 9 &&
               stats.overlap_boundary_launches == 6)
            : stats.overlap_stages == 0,
        "overlap accounting differs from the published execution path");
  CHECK(stats.testsome_polls && stats.waitall_calls == 3,
        "staged progress did not exercise Testsome then Waitall");
  CHECK(stats.testsome_polls >= 96,
        "delayed send-ready fixture did not remain in nonblocking progress");
  CHECK(stats.gather_pack_nanoseconds > 0 && stats.mpi_progress_nanoseconds > 0 &&
            stats.mpi_wait_nanoseconds > 0 && stats.scatter_unpack_nanoseconds > 0,
        "transport phase timing counters did not advance");
  CHECK(direct ? (stats.device_to_host_nanoseconds == 0 && stats.host_to_device_nanoseconds == 0)
               : (stats.device_to_host_nanoseconds > 0 && stats.host_to_device_nanoseconds > 0),
        "transport copy timing counters differ from the selected route");
  const meep::nvidia::memory_accounting after_warmup = meep::nvidia::current_memory_accounting();
  CHECK(steady_memory.device_bytes_current == after_warmup.device_bytes_current &&
            steady_memory.pinned_bytes_current == after_warmup.pinned_bytes_current &&
            steady_memory.device_allocation_count == after_warmup.device_allocation_count &&
            steady_memory.pinned_allocation_count == after_warmup.pinned_allocation_count,
        "staged transport grew device or pinned storage after warmup");
  meep::nvidia::testing::set_staged_transport_minimum_presend_polls(0);
  if (!direct) {
    if (rank == 0)
      meep::nvidia::testing::fail_staged_transport_once(
          meep::nvidia::testing::staged_transport_failure_point::host_to_device);
    CHECK(epoch->begin_stage(E_stuff, compute, why), why.c_str());
    CHECK(!epoch->finish_stage(E_stuff, compute, why),
          "post-completion H2D failure did not fail the whole stage");
    CHECK(!epoch->begin_stage(E_stuff, compute, why) && why.find("poisoned") != std::string::npos,
          "failed published stage left the transport retryable");
    CHECK(epoch->retire(why), why.c_str());
  }
  else
    CHECK(epoch->retire(why), why.c_str());

  /* Two executable-like epochs borrow one active communicator context. Local
     destruction of one must not free or invalidate the shared communicator. */
  const size_t context_users_before_pair = backend_communicator_context_use_count_for_testing();
  std::unique_ptr<meep::nvidia::staged_transport_epoch> first =
      meep::nvidia::create_staged_transport_epoch(artifact, device, &communication, why);
  std::unique_ptr<meep::nvidia::staged_transport_epoch> second =
      meep::nvidia::create_staged_transport_epoch(artifact, device, &communication, why);
  CHECK(first && second, why.c_str());
  CHECK(backend_communicator_context_use_count_for_testing() == context_users_before_pair + 2,
        "same-communicator transport rebuild did not retain shared context ownership");
  first.reset();
  CHECK(second && second->structural_identity().communicator_generation ==
                      current_backend_communicator_generation(),
        "local executable destruction invalidated a sibling transport epoch");
  CHECK(backend_communicator_context_use_count_for_testing() == context_users_before_pair + 1,
        "local executable destruction retained its communicator-context borrow");
  if (second) CHECK(second->retire(why), why.c_str());

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

#if defined(HAVE_MPI) && defined(MEEP_HIP_PORTABILITY)
struct topology_record {
  int valid;
  int rank;
  int owner;
  int logical_device;
  int gpu_numa_node;
  int cpu_numa_count;
  char runtime_uuid[96];
  char pci_bus_id[32];
  char cpu_affinity[4096];
  char cpu_numa_nodes[128];
  char error[256];
};

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

std::string normalize_pci_bus_id(std::string value) {
  value = lowercase(value);
  if (value.size() == 7) value = "0000:" + value;
  return value;
}

bool valid_pci_bus_id(const std::string &value) {
  if (value.size() != 12 || value[4] != ':' || value[7] != ':' || value[10] != '.') return false;
  for (size_t i = 0; i < value.size(); ++i)
    if (i != 4 && i != 7 && i != 10 && !std::isxdigit(static_cast<unsigned char>(value[i])))
      return false;
  return value[11] >= '0' && value[11] <= '7';
}

std::vector<std::string> expected_owner_bdfs(size_t required, const char *context) {
  const char *configured = std::getenv("MEEP_AMD_EXPECTED_OWNER_BDFS");
  if (!configured || !*configured)
    throw std::runtime_error(std::string("MEEP_AMD_EXPECTED_OWNER_BDFS is required for ") +
                             context);
  std::vector<std::string> result;
  std::stringstream values(configured);
  std::string value;
  while (std::getline(values, value, ','))
    if (!value.empty()) result.push_back(normalize_pci_bus_id(value));
  std::set<std::string> unique;
  for (const std::string &bus_id : result)
    if (!valid_pci_bus_id(bus_id))
      throw std::runtime_error("MEEP_AMD_EXPECTED_OWNER_BDFS contains an invalid PCI BDF");
    else
      unique.insert(bus_id);
  if (result.size() != required || unique.size() != required) {
    std::ostringstream message;
    message << "MEEP_AMD_EXPECTED_OWNER_BDFS must name " << required << " distinct PCI BDFs";
    throw std::runtime_error(message.str());
  }
  return result;
}

void copy_topology_text(char *target, size_t capacity, const std::string &value) {
  if (value.size() >= capacity) throw std::runtime_error("topology field exceeds fixed record");
  std::memset(target, 0, capacity);
  std::memcpy(target, value.data(), value.size());
}

void local_cpu_placement(std::string &affinity_text, std::string &nodes_text, int &node_count) {
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0)
    throw std::runtime_error("sched_getaffinity failed during topology attestation");

  std::vector<int> cpus;
  std::set<int> nodes;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &affinity)) {
      cpus.push_back(cpu);
      const std::string pattern =
          "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/node[0-9]*";
      glob_t paths = {};
      const int glob_result = glob(pattern.c_str(), 0, NULL, &paths);
      if (glob_result != 0 && glob_result != GLOB_NOMATCH) {
        globfree(&paths);
        throw std::runtime_error("cannot inspect CPU NUMA placement");
      }
      for (size_t i = 0; i < paths.gl_pathc; ++i) {
        const std::string path(paths.gl_pathv[i]);
        const size_t marker = path.rfind("/node");
        if (marker != std::string::npos) nodes.insert(std::stoi(path.substr(marker + 5)));
      }
      globfree(&paths);
    }
  if (cpus.empty()) throw std::runtime_error("CPU affinity is empty");

  std::ostringstream affinity_stream, nodes_stream;
  for (size_t i = 0; i < cpus.size(); ++i) {
    if (i) affinity_stream << ',';
    affinity_stream << cpus[i];
  }
  size_t index = 0;
  for (int node : nodes) {
    if (index++) nodes_stream << ',';
    nodes_stream << node;
  }
  affinity_text = affinity_stream.str();
  nodes_text = nodes_stream.str();
  node_count = static_cast<int>(nodes.size());
}

int gpu_numa_node(const std::string &pci_bus_id) {
  const std::string path = "/sys/bus/pci/devices/" + pci_bus_id + "/numa_node";
  std::ifstream input(path.c_str());
  int node = -1;
  if (!(input >> node) || node < 0)
    throw std::runtime_error("cannot resolve GPU NUMA node for " + pci_bus_id);
  return node;
}

topology_record make_topology_record(bool owner, int logical_device,
                                     const std::vector<meep::nvidia::device_properties> &devices) {
  topology_record record = {};
  record.rank = my_rank();
  record.owner = owner ? 1 : 0;
  record.logical_device = owner ? logical_device : -1;
  record.gpu_numa_node = -1;

  std::string affinity, nodes;
  local_cpu_placement(affinity, nodes, record.cpu_numa_count);
  copy_topology_text(record.cpu_affinity, sizeof(record.cpu_affinity), affinity);
  copy_topology_text(record.cpu_numa_nodes, sizeof(record.cpu_numa_nodes), nodes);
  if (owner) {
    char bus_id[32] = {};
    const hipError_t status = hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), logical_device);
    if (status != hipSuccess)
      throw std::runtime_error(std::string("hipDeviceGetPCIBusId failed: ") +
                               hipGetErrorString(status));
    const std::string normalized_bus_id = normalize_pci_bus_id(bus_id);
    copy_topology_text(record.runtime_uuid, sizeof(record.runtime_uuid),
                       devices.at(static_cast<size_t>(logical_device)).uuid);
    copy_topology_text(record.pci_bus_id, sizeof(record.pci_bus_id), normalized_bus_id);
    record.gpu_numa_node = gpu_numa_node(normalized_bus_id);
  }
  record.valid = 1;
  return record;
}

bool validate_np4_topology(const std::vector<topology_record> &records,
                           const std::vector<std::string> &expected_bdfs, std::string &why) {
  if (records.size() != 4 || expected_bdfs.size() != 2) {
    why = "np4 topology requires four rank records and two expected owner BDFs";
    return false;
  }
  std::set<std::string> owner_uuids, owner_bdfs;
  for (size_t i = 0; i < records.size(); ++i) {
    const topology_record &record = records[i];
    if (!record.valid) {
      why = std::string("np4 rank topology collection failed: ") + record.error;
      return false;
    }
    const bool expected_owner = i == 0 || i == 2;
    if (record.rank != static_cast<int>(i) || (record.owner != 0) != expected_owner) {
      why = "np4 rank role differs from owner0/idle1/owner2/idle3";
      return false;
    }
    if (!record.cpu_affinity[0] || record.cpu_numa_count != 1 || !record.cpu_numa_nodes[0]) {
      why = "np4 rank CPU affinity is not confined to one NUMA node";
      return false;
    }
    if (expected_owner) {
      const size_t owner_index = i / 2;
      if (record.logical_device != static_cast<int>(owner_index) ||
          normalize_pci_bus_id(record.pci_bus_id) != expected_bdfs[owner_index]) {
        why = "np4 owner did not resolve to its expected physical PCI BDF";
        return false;
      }
      char *end = NULL;
      const long cpu_numa_node = std::strtol(record.cpu_numa_nodes, &end, 10);
      if (!record.runtime_uuid[0] || !record.pci_bus_id[0] || record.gpu_numa_node < 0 ||
          end == record.cpu_numa_nodes || *end || cpu_numa_node != record.gpu_numa_node) {
        why = "np4 owner UUID/BDF/NUMA/affinity attestation is incomplete or non-local";
        return false;
      }
      owner_uuids.insert(record.runtime_uuid);
      owner_bdfs.insert(record.pci_bus_id);
    }
    else if (record.logical_device != -1 || record.runtime_uuid[0] || record.pci_bus_id[0] ||
             record.gpu_numa_node != -1) {
      why = "np4 idle rank unexpectedly reports device ownership";
      return false;
    }
  }
  if (owner_uuids.size() != 2 || owner_bdfs.size() != 2) {
    why = "np4 owners have duplicate runtime UUID or PCI BDF";
    return false;
  }
  char *first_idle_end = NULL, *second_idle_end = NULL;
  const long first_idle_node = std::strtol(records[1].cpu_numa_nodes, &first_idle_end, 10);
  const long second_idle_node = std::strtol(records[3].cpu_numa_nodes, &second_idle_end, 10);
  if (first_idle_end == records[1].cpu_numa_nodes || *first_idle_end ||
      second_idle_end == records[3].cpu_numa_nodes || *second_idle_end ||
      first_idle_node != records[0].gpu_numa_node || second_idle_node != records[2].gpu_numa_node) {
    why = "np4 idle rank is not local to the owner in its package";
    return false;
  }
  return true;
}

void attest_idle_np4_topology(bool owner, int owner_index,
                              const std::vector<meep::nvidia::device_properties> &devices) {
  topology_record local = {};
  std::vector<std::string> expected_bdfs;
  try {
    expected_bdfs = expected_owner_bdfs(2, "np4 owner/idle topology");
    local = make_topology_record(owner, owner_index, devices);
  }
  catch (const std::exception &error) {
    local.rank = my_rank();
    local.owner = owner ? 1 : 0;
    local.logical_device = owner ? owner_index : -1;
    local.gpu_numa_node = -1;
    copy_topology_text(local.error, sizeof(local.error), error.what());
  }
  std::vector<topology_record> records(4);
  if (MPI_Allgather(&local, sizeof(local), MPI_BYTE, records.data(), sizeof(local), MPI_BYTE,
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
    CHECK(false, "np4 topology record allgather failed");
    return;
  }

  if (my_rank() == 0) {
    for (const topology_record &record : records)
      std::printf("HIP_TOPOLOGY rank=%d role=%s logical_device=%d uuid=%s bdf=%s "
                  "gpu_numa=%d cpu_numa=%s cpu_affinity=%s\n",
                  record.rank, record.owner ? "owner" : "idle", record.logical_device,
                  record.runtime_uuid[0] ? record.runtime_uuid : "-",
                  record.pci_bus_id[0] ? record.pci_bus_id : "-", record.gpu_numa_node,
                  record.cpu_numa_nodes, record.cpu_affinity);

    std::string why;
    if (validate_np4_topology(records, expected_bdfs, why)) {
      std::vector<topology_record> duplicate = records;
      std::memcpy(duplicate[2].runtime_uuid, duplicate[0].runtime_uuid,
                  sizeof(duplicate[2].runtime_uuid));
      std::string duplicate_why;
      CHECK(!validate_np4_topology(duplicate, expected_bdfs, duplicate_why) &&
                duplicate_why.find("duplicate") != std::string::npos,
            "np4 topology validator accepted duplicate owner identity");
    }
    else { CHECK(false, why.c_str()); }
  }
}

bool validate_all_owner_topology(const std::vector<topology_record> &records,
                                 const std::vector<std::string> &expected_bdfs, std::string &why) {
  if (records.size() != expected_bdfs.size() || records.empty()) {
    why = "all-owner topology requires one expected BDF per rank";
    return false;
  }
  std::set<std::string> owner_uuids, owner_bdfs;
  for (size_t i = 0; i < records.size(); ++i) {
    const topology_record &record = records[i];
    if (!record.valid) {
      why = std::string("all-owner rank topology collection failed: ") + record.error;
      return false;
    }
    if (record.rank != static_cast<int>(i) || !record.owner ||
        record.logical_device != static_cast<int>(i)) {
      why = "all-owner rank did not own its process-visible logical device";
      return false;
    }
    if (!record.cpu_affinity[0] || record.cpu_numa_count != 1 || !record.cpu_numa_nodes[0]) {
      why = "all-owner rank CPU affinity is not confined to one NUMA node";
      return false;
    }
    char *end = NULL;
    const long cpu_numa_node = std::strtol(record.cpu_numa_nodes, &end, 10);
    if (!record.runtime_uuid[0] || !record.pci_bus_id[0] || record.gpu_numa_node < 0 ||
        end == record.cpu_numa_nodes || *end || cpu_numa_node != record.gpu_numa_node) {
      why = "all-owner UUID/BDF/NUMA/affinity attestation is incomplete or non-local";
      return false;
    }
    if (normalize_pci_bus_id(record.pci_bus_id) != expected_bdfs[i]) {
      why = "all-owner rank did not resolve to its expected physical PCI BDF";
      return false;
    }
    owner_uuids.insert(record.runtime_uuid);
    owner_bdfs.insert(record.pci_bus_id);
  }
  if (owner_uuids.size() != records.size() || owner_bdfs.size() != records.size()) {
    why = "all-owner ranks have duplicate runtime UUID or PCI BDF";
    return false;
  }
  return true;
}

bool attest_all_owner_topology(int logical_device,
                               const std::vector<meep::nvidia::device_properties> &devices) {
  const size_t ranks = static_cast<size_t>(count_processors());
  topology_record local = {};
  std::vector<std::string> expected_bdfs;
  try {
    expected_bdfs = expected_owner_bdfs(ranks, "all-owner topology");
    local = make_topology_record(true, logical_device, devices);
  }
  catch (const std::exception &error) {
    local.rank = my_rank();
    local.owner = 1;
    local.logical_device = logical_device;
    local.gpu_numa_node = -1;
    copy_topology_text(local.error, sizeof(local.error), error.what());
  }
  std::vector<topology_record> records(ranks);
  if (MPI_Allgather(&local, sizeof(local), MPI_BYTE, records.data(), sizeof(local), MPI_BYTE,
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
    CHECK(false, "all-owner topology record allgather failed");
    return false;
  }
  if (my_rank() == 0) {
    for (const topology_record &record : records)
      std::printf("HIP_TOPOLOGY rank=%d role=owner logical_device=%d uuid=%s bdf=%s "
                  "gpu_numa=%d cpu_numa=%s cpu_affinity=%s\n",
                  record.rank, record.logical_device,
                  record.runtime_uuid[0] ? record.runtime_uuid : "-",
                  record.pci_bus_id[0] ? record.pci_bus_id : "-", record.gpu_numa_node,
                  record.cpu_numa_nodes, record.cpu_affinity);
  }

  /* Reduce even though the gathered records are identical: a rank-local
     expected-BDF environment mistake must also fail every rank closed before
     any device-pointer MPI. */
  std::string why;
  const bool topology_valid = validate_all_owner_topology(records, expected_bdfs, why);
  const bool topology_admitted = and_to_all(topology_valid);
  if (!topology_valid)
    CHECK(false, why.c_str());
  else
    CHECK(topology_admitted, "another rank rejected all-owner topology admission");
  if (topology_admitted) {
    std::vector<topology_record> duplicate = records;
    if (duplicate.size() > 1)
      std::memcpy(duplicate[1].runtime_uuid, duplicate[0].runtime_uuid,
                  sizeof(duplicate[1].runtime_uuid));
    std::string duplicate_why;
    CHECK(duplicate.size() < 2 ||
              (!validate_all_owner_topology(duplicate, expected_bdfs, duplicate_why) &&
               duplicate_why.find("duplicate") != std::string::npos),
          "all-owner topology validator accepted duplicate owner identity");
  }
  return topology_admitted;
}
#endif

void run_idle_np4_case(const std::vector<meep::nvidia::device_properties> &devices,
                       bool direct = false) {
  if (count_processors() != 4) return;
  const int rank = my_rank();
  /* The launcher requests two ranks per package. Runtime attestation below
     verifies that owners 0/2 are local to the explicitly named PCI devices;
     no ROCr ordinal-to-physical-device equivalence is assumed. */
  const bool owner = rank == 0 || rank == 2;
  const int owner_index = rank / 2;
  const int peer = rank == 0 ? 2 : 0;
#if defined(HAVE_MPI) && defined(MEEP_HIP_PORTABILITY)
  attest_idle_np4_topology(owner, owner_index, devices);
#endif
  const size_t elements = 17;
  std::unique_ptr<meep::nvidia::device_scope> selected;
  std::unique_ptr<meep::nvidia::stream> compute, communication;
  std::unique_ptr<meep::nvidia::device_buffer> source, target;
  meep::nvidia::compiled_boundary_artifact artifact;
  artifact.wire.version = RemoteHaloProgram::schema_version;
  artifact.wire.communicator_rank = rank;
  artifact.wire.communicator_size = 4;
  artifact.wire.communicator_generation = current_backend_communicator_generation();
  artifact.wire.requested_policy = direct ? GpuMpiPolicy::direct : GpuMpiPolicy::staged;
  artifact.wire.resolved_route = direct ? GpuMpiRoute::direct : GpuMpiRoute::staged;
  artifact.wire.participation =
      RemoteHaloParticipation{owner, owner ? 1u : 0u, owner ? 2u : 0u, owner ? 1u : 0u};
  if (owner) {
    selected.reset(new meep::nvidia::device_scope(owner_index));
    compute.reset(new meep::nvidia::stream);
    communication.reset(new meep::nvidia::stream);
    source.reset(new meep::nvidia::device_buffer(elements * sizeof(double), owner_index));
    target.reset(new meep::nvidia::device_buffer(elements * sizeof(double), owner_index));
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
      meep::nvidia::create_staged_transport_epoch(artifact, owner ? owner_index : -1,
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
      if (observed[i] != peer * 100.0 + double(i)) {
        CHECK(false, "idle-rank fixture payload differs");
        break;
      }
    const meep::nvidia::staged_transport_statistics &stats = epoch->statistics();
    CHECK(stats.messages_sent == 1 && stats.messages_received == 1 && stats.bytes_sent > 0 &&
              stats.bytes_sent == stats.bytes_received && stats.direct_bytes == 0 &&
              stats.device_to_host_bytes == stats.bytes_sent &&
              stats.host_to_device_bytes == stats.bytes_received && stats.pinned_bytes > 0,
          "idle-rank owner accounting does not prove balanced staged wire traffic");
  }
  else if (epoch) {
    CHECK(epoch->participate_idle_stage(why), why.c_str());
    CHECK(epoch->participate_idle_stage(why), why.c_str());
    const meep::nvidia::staged_transport_statistics &idle = epoch->statistics();
    CHECK(idle.gather_pack_nanoseconds == 0 && idle.device_to_host_nanoseconds == 0 &&
              idle.mpi_progress_nanoseconds == 0 && idle.mpi_wait_nanoseconds == 0 &&
              idle.host_to_device_nanoseconds == 0 && idle.scatter_unpack_nanoseconds == 0,
          "idle transport participant accumulated device/transport timing");
  }
  CHECK(epoch && epoch->retire(why), why.c_str());
  (void)devices;
}

} // namespace

int main(int argc, char **argv) {
  bool staged_only = false, idle_np4_only = false, direct_all_owners_only = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--staged-only"))
      staged_only = true;
    else if (!std::strcmp(argv[i], "--idle-np4-only")) {
      idle_np4_only = true;
      staged_only = true;
    }
    else if (!std::strcmp(argv[i], "--direct-all-owners-only")) { direct_all_owners_only = true; }
    else {
      std::fprintf(stderr, "usage: nvidia_staged_transport [--staged-only] [--idle-np4-only] "
                           "[--direct-all-owners-only]\n");
      return 2;
    }
  }
  if (direct_all_owners_only && (staged_only || idle_np4_only)) {
    std::fprintf(stderr, "direct all-owner mode cannot be combined with staged-only modes\n");
    return 2;
  }
#ifdef HAVE_MPI
  int provided = MPI_THREAD_SINGLE;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS ||
      provided < MPI_THREAD_FUNNELED)
    return 2;
#endif
  initialize mpi(argc, argv);
  try {
    test_route_state_machine(staged_only);
    const std::vector<meep::nvidia::device_properties> devices = meep::nvidia::enumerate_devices();
    CHECK(!devices.empty(), "no GPU device is available");
    if (!devices.empty()) {
      const meep::nvidia::memory_accounting before = meep::nvidia::current_memory_accounting();
      const int device = my_rank() % int(devices.size());
      if (idle_np4_only) {
        CHECK(count_processors() == 4, "--idle-np4-only requires exactly four ranks");
        run_idle_np4_case(devices);
      }
      else if (direct_all_owners_only) {
        const bool rank_count_valid =
            and_to_all(count_processors() == static_cast<int>(devices.size()));
        CHECK(rank_count_valid, "--direct-all-owners-only requires one visible device per rank");
        bool topology_admitted = rank_count_valid;
#if defined(HAVE_MPI) && defined(MEEP_HIP_PORTABILITY)
        if (topology_admitted) topology_admitted = attest_all_owner_topology(device, devices);
#endif
        bool query_available = false, supports_direct = false;
        std::string provider, provider_error;
        const bool query_ok = query_gpu_aware_mpi_provider(query_available, supports_direct,
                                                           provider, provider_error);
        GpuMpiPolicy agreed_policy = GpuMpiPolicy::automatic;
        GpuMpiRoute agreed_route = GpuMpiRoute::staged;
        std::string agreement_error;
        const bool admitted = collective_resolve_gpu_mpi_policy(
            query_ok, GpuMpiPolicy::direct, query_available, supports_direct, agreed_policy,
            agreed_route, agreement_error);
        CHECK(admitted && agreed_policy == GpuMpiPolicy::direct &&
                  agreed_route == GpuMpiRoute::direct,
              query_ok ? agreement_error.c_str() : provider_error.c_str());
        if (topology_admitted && admitted) {
          if (my_rank() == 0) std::printf("DIRECT_RING_BEGIN ranks=%d\n", count_processors());
          run_case<double>(device, true);
          run_case<float>(device, true);
        }
        else if (!topology_admitted && my_rank() == 0) {
          std::printf("DIRECT_RING_SKIPPED topology admission failed\n");
        }
      }
      else {
        run_case<double>(device, false, false, staged_only);
        const char *fatal_mode = getenv("MEEP_NVIDIA_STAGED_FATAL");
        if (fatal_mode && !strcmp(fatal_mode, "retire")) {
          const int total = sum_to_all(failures);
          if (my_rank() == 0)
            std::printf("nvidia_staged_transport: %s (%d failures)\n", total ? "FAIL" : "PASS",
                        total);
#ifdef HAVE_MPI
          if (MPI_Finalize() != MPI_SUCCESS) return 3;
#endif
          return total ? 1 : 0;
        }
        run_case<float>(device, false, false, staged_only);
        if (!staged_only) run_idle_np4_case(devices);
#ifdef HAVE_MPI
        if (!staged_only) {
          {
            direct_request_mock requests(device);
            requests.arm_delayed_send_window(3);
            request_operations_scope request_scope(&requests);
            run_case<double>(device, true, true);
            requests.verify("native");
          }
          {
            direct_request_mock requests(device);
            requests.arm_delayed_send_window(3);
            request_operations_scope request_scope(&requests);
            run_case<float>(device, true, true);
            requests.verify("f32");
          }
        }
#endif
        if (!staged_only) {
          bool query_available = false, supports_direct = false;
          std::string provider, provider_error;
          CHECK(query_gpu_aware_mpi_provider(query_available, supports_direct, provider,
                                             provider_error),
                provider_error.c_str());
          if (query_available && supports_direct) {
            run_case<double>(device, true);
            run_case<float>(device, true);
            run_idle_np4_case(devices, true);
          }
          if (count_processors() == 4) {
            divide_parallel_processes(2);
            run_case<double>(my_rank() % int(devices.size()));
            end_divide_parallel();
          }
        }
      }
      const meep::nvidia::memory_accounting after = meep::nvidia::current_memory_accounting();
      CHECK(before.device_bytes_current == after.device_bytes_current &&
                before.pinned_bytes_current == after.pinned_bytes_current,
            "completed staged transport cases leaked device or pinned storage");
    }

#ifdef HAVE_MPI
    /* Keep a real device-owning transport epoch alive across externally-owned
       MPI finalization. Its stage owns GPU events and a duplicated backend
       communicator, so destruction exercises the production teardown path. */
    std::unique_ptr<meep::nvidia::device_scope> final_scope;
    std::unique_ptr<meep::nvidia::stream> final_communication;
    std::unique_ptr<meep::nvidia::staged_transport_epoch> final_epoch;
    if (!idle_np4_only && !direct_all_owners_only && !devices.empty() && count_processors() >= 2) {
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
