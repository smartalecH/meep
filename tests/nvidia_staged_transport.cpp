/* True-MPI synthetic exercise of the PR7.2 staged CUDA transport epoch. */

#include <meep.hpp>

#include "backend/mpi_context.hpp"
#include "backend/nvidia/nvidia_mpi.hpp"
#include "backend/nvidia/runtime.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <exception>
#include <memory>
#include <set>
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

void test_route_state_machine() {
  using meep::nvidia::testing::transport_presend_action;
  using meep::nvidia::testing::transport_presend_action_for_testing;
  CHECK(transport_presend_action_for_testing(GpuMpiRoute::direct, false, false, false) ==
            transport_presend_action::poll,
        "direct transport posted a send before the producer event");
  CHECK(transport_presend_action_for_testing(GpuMpiRoute::direct, true, false, true) ==
            transport_presend_action::post_sends,
        "direct transport did not post from the device arena after the producer event");
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
        delayed_send_window_expected(0), delayed_send_window_sends(0),
        delayed_send_window_polls(0), delayed_send_window_premature_sends(0),
        copies_completed(0) {}

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
    CHECK(!delayed_send_window_armed &&
              delayed_send_window_sends == delayed_send_window_expected &&
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
int run_case(int device, bool direct = false, bool direct_mock_local_payload = false) {
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
  if (!direct) {
    const DependencyOverlapPolicy fallback_policies[] = {
        DependencyOverlapPolicy::off, DependencyOverlapPolicy::automatic};
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
            meep::nvidia::testing::fail_next(
                meep::nvidia::testing::failure_point::device_allocate);
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
                  fallback_artifact.lowered.program_signature ==
                      fallback_artifact.wire.signature &&
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
      const meep::nvidia::memory_accounting before =
          meep::nvidia::current_memory_accounting();
      if (rank == 0) {
        if (failure_kind == 0)
          meep::nvidia::testing::fail_next(
              meep::nvidia::testing::failure_point::device_allocate);
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
      const meep::nvidia::memory_accounting after =
          meep::nvidia::current_memory_accounting();
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
  CHECK(backend_communicator_context_use_count_for_testing() ==
            context_users_before_failed_create,
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
                     rank, direct ? 1 : 0, direct_mock_local_payload ? 1 : 0, sizeof(T),
                     iteration, i, double(observed[i]), double(expected));
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
  CHECK(stats.direct_bytes == (direct ? 6 * elements * sizeof(T) : 0) &&
            stats.slot_reuses == 1,
        "route byte or two-slot reuse accounting differs");
  CHECK(!direct || stats.pinned_bytes == 0,
        "direct transport unexpectedly retained pinned staging storage");
  CHECK(direct_mock_local_payload
            ? (stats.overlap_stages == 3 && stats.overlap_interior_launches == 9 &&
               stats.overlap_boundary_launches == 6)
            : stats.overlap_stages == 0,
        "overlap accounting differs from the published execution path");
  CHECK(stats.testsome_polls && stats.waitall_calls == 3,
        "staged progress did not exercise Testsome then Waitall");
  CHECK(stats.testsome_polls >= 96,
        "delayed send-ready fixture did not remain in nonblocking progress");
  const meep::nvidia::memory_accounting after_warmup = meep::nvidia::current_memory_accounting();
  CHECK(steady_memory.device_bytes_current == after_warmup.device_bytes_current &&
            steady_memory.pinned_bytes_current == after_warmup.pinned_bytes_current,
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

void run_idle_np4_case(const std::vector<meep::nvidia::device_properties> &devices,
                       bool direct = false) {
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
  artifact.wire.requested_policy = direct ? GpuMpiPolicy::direct : GpuMpiPolicy::staged;
  artifact.wire.resolved_route = direct ? GpuMpiRoute::direct : GpuMpiRoute::staged;
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
  test_route_state_machine();
    const std::vector<meep::nvidia::device_properties> devices = meep::nvidia::enumerate_devices();
    CHECK(!devices.empty(), "no CUDA device is available");
    if (!devices.empty()) {
      const int device = my_rank() % int(devices.size());
      run_case<double>(device);
      const char *fatal_mode = getenv("MEEP_NVIDIA_STAGED_FATAL");
      if (fatal_mode && !strcmp(fatal_mode, "retire")) {
        const int total = sum_to_all(failures);
        if (my_rank() == 0)
          std::printf("nvidia_staged_transport: %s (%d failures)\n",
                      total ? "FAIL" : "PASS", total);
#ifdef HAVE_MPI
        if (MPI_Finalize() != MPI_SUCCESS) return 3;
#endif
        return total ? 1 : 0;
      }
      run_case<float>(device);
      run_idle_np4_case(devices);
#ifdef HAVE_MPI
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
#endif
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
