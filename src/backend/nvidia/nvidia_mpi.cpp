/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_mpi.hpp"

#include "backend/mpi_context.hpp"
#include "backend/nvidia/nvidia_peer.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace meep {
namespace nvidia {
namespace {

typedef std::chrono::steady_clock transport_clock;

uint64_t elapsed_nanoseconds(const transport_clock::time_point &start) {
  const std::chrono::nanoseconds elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(transport_clock::now() - start);
  return elapsed.count() > 0 ? uint64_t(elapsed.count()) : uint64_t(0);
}

testing::staged_transport_failure_point injected_failure =
    testing::staged_transport_failure_point::none;
size_t minimum_presend_polls = 0;
testing::mpi_request_operations *injected_request_operations = NULL;

uint64_t identity_mix(uint64_t hash, uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) {
    hash ^= (value >> (8 * i)) & 0xffu;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

uint64_t identity_string(const std::string &value) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (unsigned char c : value) {
    hash ^= c;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

uint64_t provider_identity(const BackendCommunicatorInfo &info) {
  uint64_t hash = identity_string(info.provider);
  hash = identity_mix(hash, info.provider_query_available ? 1 : 0);
  return identity_mix(hash, info.provider_supports_direct ? 1 : 0);
}

uint64_t provider_identity(bool query_available, bool supports_direct,
                           const std::string &provider) {
  uint64_t hash = identity_string(provider);
  hash = identity_mix(hash, query_available ? 1 : 0);
  return identity_mix(hash, supports_direct ? 1 : 0);
}

uint64_t arena_layout_identity(const compiled_boundary_artifact &artifact) {
  uint64_t hash = UINT64_C(1469598103934665603);
  hash = identity_mix(hash, 1); // slot-layout schema
  hash = identity_mix(hash, 2); // fixed double buffering
  hash = identity_mix(hash, uint64_t(artifact.wire.resolved_route));
  for (const bound_boundary_stage &stage : artifact.bound.stages) {
    hash = identity_mix(hash, uint64_t(stage.ft));
    hash = identity_mix(hash, stage.receive_slot_bytes);
    hash = identity_mix(hash, stage.send_slot_bytes);
    hash = identity_mix(hash, stage.receives.size());
    for (const bound_boundary_message &message : stage.receives) {
      hash = identity_mix(hash, message.wire_bytes);
      hash = identity_mix(hash, message.arena_offsets[0]);
      hash = identity_mix(hash, message.arena_offsets[1]);
    }
    hash = identity_mix(hash, stage.sends.size());
    for (const bound_boundary_message &message : stage.sends) {
      hash = identity_mix(hash, message.wire_bytes);
      hash = identity_mix(hash, message.arena_offsets[0]);
      hash = identity_mix(hash, message.arena_offsets[1]);
    }
  }
  return hash;
}

bool consume_failure(testing::staged_transport_failure_point point) {
  if (injected_failure != point) return false;
  injected_failure = testing::staged_transport_failure_point::none;
  return true;
}

testing::transport_presend_action presend_action(GpuMpiRoute route, bool gather_ready,
                                                 bool staging_ready, bool copies_enqueued) {
  if (!gather_ready) return testing::transport_presend_action::poll;
  if (route == GpuMpiRoute::direct) return testing::transport_presend_action::post_sends;
  if (!copies_enqueued) return testing::transport_presend_action::enqueue_device_to_host;
  return staging_ready ? testing::transport_presend_action::post_sends
                       : testing::transport_presend_action::poll;
}

uint64_t checked_counter_add(uint64_t value, uint64_t add, const char *what) {
  if (add > std::numeric_limits<uint64_t>::max() - value)
    throw std::overflow_error(std::string("NVIDIA MPI statistic overflow: ") + what);
  return value + add;
}

size_t checked_size_add(size_t value, size_t add, const char *what) {
  if (add > std::numeric_limits<size_t>::max() - value)
    throw std::overflow_error(std::string("NVIDIA MPI allocation overflow: ") + what);
  return value + add;
}

#ifdef HAVE_MPI
int transport_irecv(void *buffer, int bytes, int source, int tag, MPI_Comm communicator,
                    MPI_Request *request) {
  if (injected_request_operations)
    return injected_request_operations->irecv(buffer, size_t(bytes), source, tag, request);
  return MPI_Irecv(buffer, bytes, MPI_BYTE, source, tag, communicator, request);
}

int transport_isend(const void *buffer, int bytes, int destination, int tag,
                    MPI_Comm communicator, MPI_Request *request) {
  if (injected_request_operations)
    return injected_request_operations->isend(buffer, size_t(bytes), destination, tag, request);
  return MPI_Isend(buffer, bytes, MPI_BYTE, destination, tag, communicator, request);
}

int transport_testsome(std::vector<MPI_Request> &requests, int &count, int *completed,
                       MPI_Status *statuses) {
  if (injected_request_operations)
    return injected_request_operations->testsome(requests.size(), requests.data(),
                                                 sizeof(MPI_Request), count, completed);
  return MPI_Testsome(int(requests.size()), requests.data(), &count, completed, statuses);
}

int transport_waitall(std::vector<MPI_Request> &requests, MPI_Status *statuses) {
  if (injected_request_operations)
    return injected_request_operations->waitall(requests.size(), requests.data(),
                                                sizeof(MPI_Request));
  return MPI_Waitall(int(requests.size()), requests.data(), statuses);
}

bool transport_request_is_null(const MPI_Request *request) {
  return injected_request_operations ? injected_request_operations->request_is_null(request)
                                     : *request == MPI_REQUEST_NULL;
}

void transport_clear_request(MPI_Request *request) {
  if (injected_request_operations)
    injected_request_operations->clear_request(request);
  else
    *request = MPI_REQUEST_NULL;
}

[[noreturn]] void fatal_transport(MPI_Comm communicator, const char *operation, int code) {
  std::fprintf(stderr, "fatal NVIDIA staged MPI transport failure in %s (MPI code %d)\n", operation,
               code);
  std::fflush(stderr);
  MPI_Abort(communicator, code == MPI_SUCCESS ? 1 : code);
  std::_Exit(1);
}

void mpi_require(int code, MPI_Comm communicator, const char *operation) {
  if (code != MPI_SUCCESS) fatal_transport(communicator, operation, code);
}
#endif

} // namespace

staged_transport_statistics::staged_transport_statistics()
    : messages_sent(0), messages_received(0), bytes_sent(0), bytes_received(0),
      device_to_host_calls(0), device_to_host_bytes(0), host_to_device_calls(0),
      host_to_device_bytes(0), direct_bytes(0), gather_launches(0), scatter_launches(0),
      testsome_polls(0), waitall_calls(0), request_completions(0), slot_reuses(0),
      overlap_stages(0), overlap_interior_launches(0), overlap_boundary_launches(0),
      high_water_requests(0), gather_pack_nanoseconds(0), device_to_host_nanoseconds(0),
      mpi_progress_nanoseconds(0), mpi_wait_nanoseconds(0), host_to_device_nanoseconds(0),
      scatter_unpack_nanoseconds(0), device_bytes(0), pinned_bytes(0) {}

transport_structural_identity::transport_structural_identity()
    : version(schema_version), slot_layout_version(1), slot_count(2),
      communicator_generation(0), communicator_rank(0), communicator_size(1), tag_ub(0),
      requested_policy(GpuMpiPolicy::automatic), resolved_route(GpuMpiRoute::staged),
      overlap_policy(DependencyOverlapPolicy::off), wire_signature(0), authority_signature(0),
      provider_signature(0), device_signature(0), owner_map_signature(0),
      arena_layout_signature(0), dependency_signature(0), signature(0) {}

uint64_t compute_transport_structural_identity_signature(
    const transport_structural_identity &identity) {
  uint64_t hash = UINT64_C(1469598103934665603);
  hash = identity_mix(hash, identity.version);
  hash = identity_mix(hash, identity.slot_layout_version);
  hash = identity_mix(hash, identity.slot_count);
  hash = identity_mix(hash, identity.communicator_generation);
  hash = identity_mix(hash, uint64_t(uint32_t(identity.communicator_rank)));
  hash = identity_mix(hash, uint64_t(uint32_t(identity.communicator_size)));
  hash = identity_mix(hash, uint64_t(uint32_t(identity.tag_ub)));
  hash = identity_mix(hash, uint64_t(identity.requested_policy));
  hash = identity_mix(hash, uint64_t(identity.resolved_route));
  hash = identity_mix(hash, uint64_t(identity.overlap_policy));
  hash = identity_mix(hash, identity.wire_signature);
  hash = identity_mix(hash, identity.authority_signature);
  hash = identity_mix(hash, identity.provider_signature);
  hash = identity_mix(hash, identity.device_signature);
  hash = identity_mix(hash, identity.owner_map_signature);
  hash = identity_mix(hash, identity.arena_layout_signature);
  return identity_mix(hash, identity.dependency_signature);
}

bool operator==(const transport_structural_identity &a,
                const transport_structural_identity &b) {
  return a.version == b.version && a.slot_layout_version == b.slot_layout_version &&
         a.slot_count == b.slot_count &&
         a.communicator_generation == b.communicator_generation &&
         a.communicator_rank == b.communicator_rank &&
         a.communicator_size == b.communicator_size && a.tag_ub == b.tag_ub &&
         a.requested_policy == b.requested_policy && a.resolved_route == b.resolved_route &&
         a.overlap_policy == b.overlap_policy && a.wire_signature == b.wire_signature &&
         a.authority_signature == b.authority_signature &&
         a.provider_signature == b.provider_signature &&
         a.device_signature == b.device_signature &&
         a.owner_map_signature == b.owner_map_signature &&
         a.arena_layout_signature == b.arena_layout_signature &&
         a.dependency_signature == b.dependency_signature && a.signature == b.signature;
}

struct staged_transport_epoch::impl {
  enum class slot_phase { idle, begun };

  struct slot_state {
    slot_phase phase;
    event compute_ready;
    event gather_ready;
    event local_scatter_ready;
    event device_to_host_ready;
    event transport_done;
#ifdef HAVE_MPI
    std::vector<MPI_Request> requests;
    std::vector<int> completed;
    std::vector<MPI_Status> statuses;
#endif
    slot_state() : phase(slot_phase::idle) {}
  };

  struct stage_state {
    bound_boundary_stage plan;
    device_buffer gather_entries;
    device_buffer scatter_entries;
    device_buffer send_arena;
    device_buffer receive_arena;
    pinned_buffer send_staging;
    pinned_buffer receive_staging;
    slot_state slots[2];
    uint64_t generation;
    stage_state() : generation(0) {}
  };

  /* Node-local peer route (parallel to `stages`): sends gather straight into
     the receiver's mapped arena slot, then release-store generation+1 into
     the receiver's ready flag; the receiver waits on its ready flags on the
     device, scatters, and stores generation+1 into the sender's ack flag. A
     sender reuses slot g&1 only after the ack for generation g-2. */
  struct peer_stage {
    size_t ready_first; // local flags, one per receive
    size_t ack_first;   // local flags, one per send
    std::vector<unsigned char *> send_targets[2];
    device_buffer ready_targets; // uint64_t* per send: the receiver's ready flag
    device_buffer ack_targets;   // uint64_t* per receive: the sender's ack flag
    peer_stage() : ready_first(0), ack_first(0) {}
  };

  int device;
  stream *communication;
  BackendCommunicatorLease communicator;
  RemoteHaloProgram wire;
  std::vector<stage_state> stages;
  staged_transport_statistics stats;
  transport_structural_identity identity;
  bool retired;
  bool poisoned;
  bool peer;
  double peer_timeout_seconds;
  device_buffer peer_flags;
  std::vector<peer_mapping> peer_mappings;
  std::vector<peer_stage> peer_stages;

  impl()
      : device(-1), communication(NULL), retired(false), poisoned(false), peer(false),
        peer_timeout_seconds(0) {}

  stage_state *find(field_type ft) {
    for (stage_state &stage : stages)
      if (stage.plan.ft == ft) return &stage;
    return NULL;
  }
  const stage_state *find(field_type ft) const {
    for (const stage_state &stage : stages)
      if (stage.plan.ft == ft) return &stage;
    return NULL;
  }
};

staged_transport_epoch::staged_transport_epoch() : impl_(new impl) {}

staged_transport_epoch::~staged_transport_epoch() {
  if (!impl_) return;
  if (!impl_->retired && impl_->communicator.valid()) {
    std::string ignored;
    if (!retire(ignored) && impl_->communicator.valid()) {
      std::fprintf(stderr, "NVIDIA staged transport epoch was not collectively retired\n");
      std::abort();
    }
  }
  delete impl_;
}

staged_transport_epoch::staged_transport_epoch(staged_transport_epoch &&other) noexcept
    : impl_(other.impl_) {
  other.impl_ = NULL;
}

staged_transport_epoch &staged_transport_epoch::operator=(staged_transport_epoch &&other) noexcept {
  if (this == &other) return *this;
  if (impl_) {
    std::string ignored;
    if (!retire(ignored) && impl_->communicator.valid()) std::abort();
    delete impl_;
  }
  impl_ = other.impl_;
  other.impl_ = NULL;
  return *this;
}

bool staged_transport_epoch::has_stage(field_type ft) const {
  const impl::stage_state *stage = impl_ ? impl_->find(ft) : NULL;
  return stage != NULL;
}

void staged_transport_epoch::record_dependency_overlap(size_t interior_launches,
                                                       size_t boundary_launches) {
  if (!impl_ || impl_->retired || impl_->poisoned)
    throw std::logic_error("cannot account dependency overlap on an inactive transport");
  impl_->stats.overlap_stages =
      checked_counter_add(impl_->stats.overlap_stages, 1, "overlap stage");
  impl_->stats.overlap_interior_launches = checked_counter_add(
      impl_->stats.overlap_interior_launches, interior_launches, "overlap interior launch");
  impl_->stats.overlap_boundary_launches = checked_counter_add(
      impl_->stats.overlap_boundary_launches, boundary_launches, "overlap boundary launch");
}

bool staged_transport_epoch::begin_stage(field_type ft, const stream &compute, std::string &why) {
  why.clear();
  if (!impl_) {
    why = "NVIDIA staged transport epoch has no implementation";
    return false;
  }
  std::string local_error;
  if (impl_->retired || impl_->poisoned)
    local_error = impl_->retired ? "NVIDIA staged transport epoch is retired"
                                 : "NVIDIA staged transport epoch is poisoned";
  if (local_error.empty() &&
      impl_->wire.communicator_generation != current_backend_communicator_generation())
    local_error = "NVIDIA staged transport communicator generation is stale";
  impl::stage_state *stage = impl_->find(ft);
  if (!stage && local_error.empty()) local_error = "NVIDIA staged transport has no field stage";
  if (!local_error.empty()) {
    why = local_error;
    return false;
  }
  const size_t slot_index = size_t(stage->generation & 1u);
  impl::slot_state &slot = stage->slots[slot_index];
  if (slot.phase != impl::slot_phase::idle)
    local_error = "NVIDIA staged transport slot was reused before completion";
  if (impl_->peer) {
    /* Everything is enqueued on the compute stream behind the caller's
       zeroing and local gathers; the host never waits. */
    if (!local_error.empty()) {
      why = local_error;
      return false;
    }
    impl::peer_stage &peer = impl_->peer_stages[size_t(stage - impl_->stages.data())];
    const uint64_t generation = stage->generation;
    const size_t sends = stage->plan.sends.size();
    try {
      if (generation >= 2)
        launch_peer_wait(static_cast<const uint64_t *>(impl_->peer_flags.opaque_handle()) +
                             peer.ack_first,
                         sends, generation - 1, impl_->peer_timeout_seconds, compute);
      for (size_t i = 0; i < sends; ++i)
        launch_boundary_gather(stage->plan.sends[i].launch, stage->gather_entries.opaque_handle(),
                               peer.send_targets[slot_index][i], compute);
      launch_peer_signal(peer.ready_targets.opaque_handle(), sends, generation + 1, compute);
    }
    catch (const std::exception &error) {
      why = error.what();
      return false;
    }
    impl_->stats.gather_launches =
        checked_counter_add(impl_->stats.gather_launches, sends, "gather launch");
    if (generation >= 2)
      impl_->stats.slot_reuses = checked_counter_add(impl_->stats.slot_reuses, 1, "slot reuse");
    slot.phase = impl::slot_phase::begun;
    return true;
  }
#ifdef HAVE_MPI
  for (MPI_Request &request : slot.requests)
    if (!transport_request_is_null(&request))
      local_error = "NVIDIA staged transport slot retained a live MPI request";
#endif
  if (stage->generation >= 2)
    impl_->stats.slot_reuses = checked_counter_add(impl_->stats.slot_reuses, 1, "slot reuse");
#ifdef HAVE_MPI
  std::string thread_error;
  if (!backend_mpi_thread_ready(thread_error) && local_error.empty()) local_error = thread_error;
  MPI_Comm communicator = backend_communicator(impl_->communicator);
  if (communicator == MPI_COMM_NULL) {
    why = "NVIDIA staged transport communicator is invalid";
    return false;
  }
  if (consume_failure(testing::staged_transport_failure_point::before_receive_post))
    local_error = "injected NVIDIA staged transport receive preflight failure";
  if (local_error.empty()) {
    try {
      const size_t request_count = stage->plan.receives.size() + stage->plan.sends.size();
      slot.requests.assign(request_count, MPI_REQUEST_NULL);
      slot.completed.resize(request_count);
      slot.statuses.resize(request_count);
    }
    catch (const std::exception &error) {
      local_error = error.what();
    }
    catch (...) {
      local_error = "unknown NVIDIA staged transport request allocation failure";
    }
  }
  int local_failed = local_error.empty() ? 0 : 1, any_failed = 0;
  mpi_require(MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_LOR, communicator),
              communicator, "transport receive preflight reconciliation");
  if (any_failed) {
    why = local_error.empty() ? "NVIDIA staged transport preflight failed on another rank"
                              : local_error;
    return false;
  }
  for (size_t i = 0; i < stage->plan.receives.size(); ++i) {
    const bound_boundary_message &message = stage->plan.receives[i];
    unsigned char *destination =
        impl_->wire.resolved_route == GpuMpiRoute::direct
            ? static_cast<unsigned char *>(stage->receive_arena.opaque_handle()) +
                  message.arena_offsets[slot_index]
            : static_cast<unsigned char *>(stage->receive_staging.data()) +
                  message.arena_offsets[slot_index];
    const int code = transport_irecv(destination, int(message.wire_bytes),
                                     message.key.source_rank, message.key.tag, communicator,
                                     &slot.requests[i]);
    if (code != MPI_SUCCESS) fatal_transport(communicator, "MPI_Irecv", code);
  }
  slot.phase = impl::slot_phase::begun;
  const uint64_t live = uint64_t(stage->plan.receives.size());
  impl_->stats.high_water_requests = std::max(impl_->stats.high_water_requests, live);
  if (consume_failure(testing::staged_transport_failure_point::after_receive_post))
    fatal_transport(communicator, "injected failure after MPI_Irecv publication", 1);
  try {
    const transport_clock::time_point gather_start = transport_clock::now();
    if (consume_failure(testing::staged_transport_failure_point::gather))
      throw std::runtime_error("injected NVIDIA staged transport gather failure");
    slot.compute_ready.record(compute);
    slot.compute_ready.wait(*impl_->communication);
    for (const bound_boundary_message &message : stage->plan.sends) {
      launch_boundary_gather(message.launch, stage->gather_entries.opaque_handle(),
                             static_cast<unsigned char *>(stage->send_arena.opaque_handle()) +
                                 message.arena_offsets[slot_index],
                             *impl_->communication);
      impl_->stats.gather_launches =
          checked_counter_add(impl_->stats.gather_launches, 1, "gather launch");
    }
    slot.gather_ready.record(*impl_->communication);
    slot.gather_ready.wait(compute);
    if (!stage->plan.sends.empty())
      impl_->stats.gather_pack_nanoseconds = checked_counter_add(
          impl_->stats.gather_pack_nanoseconds, elapsed_nanoseconds(gather_start),
          "gather/pack elapsed nanoseconds");
  }
  catch (...) {
    fatal_transport(communicator, "CUDA gather after request publication", 1);
  }
  return true;
#else
  (void)compute;
  why = "NVIDIA staged transport requires an MPI build";
  return false;
#endif
}

bool staged_transport_epoch::finish_stage(field_type ft, const stream &compute, std::string &why) {
  why.clear();
  if (!impl_ || impl_->retired) {
    why = "NVIDIA staged transport epoch is retired";
    return false;
  }
  impl::stage_state *stage = impl_->find(ft);
  if (!stage) {
    why = "NVIDIA staged transport has no field stage";
    return false;
  }
  const size_t slot_index = size_t(stage->generation & 1u);
  impl::slot_state &slot = stage->slots[slot_index];
  if (slot.phase != impl::slot_phase::begun) {
    why = "NVIDIA staged transport stage was not begun";
    return false;
  }
  if (impl_->peer) {
    impl::peer_stage &peer = impl_->peer_stages[size_t(stage - impl_->stages.data())];
    const uint64_t generation = stage->generation;
    const size_t receives = stage->plan.receives.size();
    try {
      launch_peer_wait(static_cast<const uint64_t *>(impl_->peer_flags.opaque_handle()) +
                           peer.ready_first,
                       receives, generation + 1, impl_->peer_timeout_seconds, compute);
      for (uint32_t index : stage->plan.canonical_receive_order) {
        if (index >= receives) throw std::logic_error("NVIDIA peer transport receive order is stale");
        const bound_boundary_message &message = stage->plan.receives[index];
        const void *arena = static_cast<const unsigned char *>(stage->receive_arena.opaque_handle()) +
                            message.arena_offsets[slot_index];
        if (stage->plan.publication == RemoteHaloPublicationMode::canonical_serial)
          launch_boundary_scatter_serial(message.launch, stage->scatter_entries.opaque_handle(),
                                         arena, compute);
        else
          launch_boundary_scatter_parallel(message.launch, stage->scatter_entries.opaque_handle(),
                                           arena, compute);
      }
      launch_peer_signal(peer.ack_targets.opaque_handle(), receives, generation + 1, compute);
    }
    catch (const std::exception &error) {
      /* Peers may already be spinning on this generation; they trap after
         the peer timeout instead of hanging. */
      impl_->poisoned = true;
      why = error.what();
      return false;
    }
    impl_->stats.scatter_launches =
        checked_counter_add(impl_->stats.scatter_launches, receives, "scatter launch");
    impl_->stats.messages_received =
        checked_counter_add(impl_->stats.messages_received, receives, "received messages");
    impl_->stats.messages_sent = checked_counter_add(
        impl_->stats.messages_sent, stage->plan.sends.size(), "sent messages");
    for (const bound_boundary_message &message : stage->plan.receives)
      impl_->stats.bytes_received =
          checked_counter_add(impl_->stats.bytes_received, message.wire_bytes, "received bytes");
    for (const bound_boundary_message &message : stage->plan.sends) {
      impl_->stats.bytes_sent =
          checked_counter_add(impl_->stats.bytes_sent, message.wire_bytes, "sent bytes");
      impl_->stats.direct_bytes =
          checked_counter_add(impl_->stats.direct_bytes, message.wire_bytes, "direct bytes");
    }
    slot.phase = impl::slot_phase::idle;
    ++stage->generation;
    return true;
  }
#ifdef HAVE_MPI
  if (!backend_mpi_thread_ready(why)) {
    fatal_transport(backend_communicator(impl_->communicator),
                    "MPI thread contract after request publication", 1);
  }
  MPI_Comm communicator = backend_communicator(impl_->communicator);
  try {
    slot.local_scatter_ready.record(compute);
  }
  catch (...) {
    fatal_transport(communicator, "CUDA local-scatter event record", 1);
  }

  const bool direct = impl_->wire.resolved_route == GpuMpiRoute::direct;
  bool copies_enqueued = direct || stage->plan.sends.empty();
  bool all_sends_posted = stage->plan.sends.empty();
  size_t presend_polls = 0;
  const transport_clock::time_point progress_start = transport_clock::now();
  while (!all_sends_posted) {
    int count = 0;
    mpi_require(transport_testsome(slot.requests, count, slot.completed.data(),
                                   slot.statuses.data()),
                communicator, "MPI_Testsome before all sends are posted");
    try {
      impl_->stats.testsome_polls =
          checked_counter_add(impl_->stats.testsome_polls, 1, "Testsome poll");
      if (count != MPI_UNDEFINED)
        impl_->stats.request_completions = checked_counter_add(
            impl_->stats.request_completions, uint64_t(count), "request completion");
      ++presend_polls;
      const bool gather_ready =
          presend_polls >= minimum_presend_polls && slot.gather_ready.ready();
      const testing::transport_presend_action action =
          presend_action(impl_->wire.resolved_route, gather_ready,
                         !direct && copies_enqueued && slot.device_to_host_ready.ready(),
                         copies_enqueued);
      if (action == testing::transport_presend_action::enqueue_device_to_host) {
        const transport_clock::time_point copy_start = transport_clock::now();
        if (consume_failure(testing::staged_transport_failure_point::device_to_host))
          throw std::runtime_error("injected NVIDIA staged transport D2H failure");
        for (const bound_boundary_message &message : stage->plan.sends) {
          void *destination = static_cast<unsigned char *>(stage->send_staging.data()) +
                              message.arena_offsets[slot_index];
          copy_device_to_host_async(destination, stage->send_arena,
                                    message.arena_offsets[slot_index], message.wire_bytes,
                                    *impl_->communication);
          impl_->stats.device_to_host_calls =
              checked_counter_add(impl_->stats.device_to_host_calls, 1, "D2H call");
          impl_->stats.device_to_host_bytes = checked_counter_add(impl_->stats.device_to_host_bytes,
                                                                  message.wire_bytes, "D2H bytes");
        }
        slot.device_to_host_ready.record(*impl_->communication);
        impl_->stats.device_to_host_nanoseconds = checked_counter_add(
            impl_->stats.device_to_host_nanoseconds, elapsed_nanoseconds(copy_start),
            "D2H elapsed nanoseconds");
        copies_enqueued = true;
      }
      const bool staging_ready =
          !direct && copies_enqueued && slot.device_to_host_ready.ready();
      if (presend_action(impl_->wire.resolved_route, gather_ready, staging_ready,
                         copies_enqueued) == testing::transport_presend_action::post_sends) {
        if (consume_failure(testing::staged_transport_failure_point::before_send_post))
          fatal_transport(communicator, "injected failure before MPI_Isend publication", 1);
        const size_t first_send = stage->plan.receives.size();
        for (size_t i = 0; i < stage->plan.sends.size(); ++i) {
          const bound_boundary_message &message = stage->plan.sends[i];
          const unsigned char *source =
              direct ? static_cast<const unsigned char *>(stage->send_arena.opaque_handle()) +
                           message.arena_offsets[slot_index]
                     : static_cast<const unsigned char *>(stage->send_staging.data()) +
                           message.arena_offsets[slot_index];
          mpi_require(transport_isend(source, int(message.wire_bytes),
                                      message.key.destination_rank, message.key.tag, communicator,
                                      &slot.requests[first_send + i]),
                      communicator, "MPI_Isend");
          if (consume_failure(testing::staged_transport_failure_point::after_send_post))
            fatal_transport(communicator, "injected failure after MPI_Isend publication", 1);
        }
        all_sends_posted = true;
        impl_->stats.high_water_requests =
            std::max(impl_->stats.high_water_requests, uint64_t(slot.requests.size()));
      }
    }
    catch (...) {
      fatal_transport(communicator, "CUDA staging before all sends are posted", 1);
    }
  }
  if (!stage->plan.sends.empty())
    impl_->stats.mpi_progress_nanoseconds = checked_counter_add(
        impl_->stats.mpi_progress_nanoseconds, elapsed_nanoseconds(progress_start),
        "MPI progress elapsed nanoseconds");

  size_t waited_requests = 0;
  for (MPI_Request &request : slot.requests)
    if (!transport_request_is_null(&request)) ++waited_requests;
  if (!slot.requests.empty()) {
    const transport_clock::time_point wait_start = transport_clock::now();
    mpi_require(transport_waitall(slot.requests, slot.statuses.data()),
                communicator, "MPI_Waitall after all sends are posted");
    impl_->stats.mpi_wait_nanoseconds = checked_counter_add(
        impl_->stats.mpi_wait_nanoseconds, elapsed_nanoseconds(wait_start),
        "MPI wait elapsed nanoseconds");
  }
  for (MPI_Request &request : slot.requests)
    transport_clear_request(&request);

  std::string local_error;
  try {
    if (!slot.requests.empty())
      impl_->stats.waitall_calls =
          checked_counter_add(impl_->stats.waitall_calls, 1, "Waitall call");
    impl_->stats.request_completions = checked_counter_add(
        impl_->stats.request_completions, waited_requests, "Waitall request completion");
    impl_->stats.messages_received = checked_counter_add(
        impl_->stats.messages_received, stage->plan.receives.size(), "received messages");
    impl_->stats.messages_sent =
        checked_counter_add(impl_->stats.messages_sent, stage->plan.sends.size(), "sent messages");
    for (const bound_boundary_message &message : stage->plan.receives)
      impl_->stats.bytes_received =
          checked_counter_add(impl_->stats.bytes_received, message.wire_bytes, "received bytes");
    for (const bound_boundary_message &message : stage->plan.sends)
      impl_->stats.bytes_sent =
          checked_counter_add(impl_->stats.bytes_sent, message.wire_bytes, "sent bytes");
    if (!direct && consume_failure(testing::staged_transport_failure_point::host_to_device))
      throw std::runtime_error("injected NVIDIA staged transport H2D failure");
    const transport_clock::time_point host_to_device_start = transport_clock::now();
    for (const bound_boundary_message &message : stage->plan.receives)
      if (direct)
        impl_->stats.direct_bytes =
            checked_counter_add(impl_->stats.direct_bytes, message.wire_bytes, "direct bytes");
      else {
        const void *source = static_cast<const unsigned char *>(stage->receive_staging.data()) +
                             message.arena_offsets[slot_index];
        copy_host_to_device_async(stage->receive_arena, message.arena_offsets[slot_index], source,
                                  message.wire_bytes, *impl_->communication);
        impl_->stats.host_to_device_calls =
            checked_counter_add(impl_->stats.host_to_device_calls, 1, "H2D call");
        impl_->stats.host_to_device_bytes = checked_counter_add(
            impl_->stats.host_to_device_bytes, message.wire_bytes, "H2D bytes");
      }
    if (!direct && !stage->plan.receives.empty())
      impl_->stats.host_to_device_nanoseconds = checked_counter_add(
          impl_->stats.host_to_device_nanoseconds,
          elapsed_nanoseconds(host_to_device_start), "H2D elapsed nanoseconds");
    if (direct)
      for (const bound_boundary_message &message : stage->plan.sends)
        impl_->stats.direct_bytes =
            checked_counter_add(impl_->stats.direct_bytes, message.wire_bytes, "direct bytes");
    const transport_clock::time_point scatter_start = transport_clock::now();
    slot.local_scatter_ready.wait(*impl_->communication);
    if (consume_failure(testing::staged_transport_failure_point::scatter))
      throw std::runtime_error("injected NVIDIA staged transport scatter failure");
    const std::vector<uint32_t> &order = stage->plan.canonical_receive_order;
    for (uint32_t index : order) {
      if (index >= stage->plan.receives.size())
        throw std::logic_error("NVIDIA staged transport receive order is stale");
      const bound_boundary_message &message = stage->plan.receives[index];
      const void *arena = static_cast<const unsigned char *>(stage->receive_arena.opaque_handle()) +
                          message.arena_offsets[slot_index];
      if (stage->plan.publication == RemoteHaloPublicationMode::canonical_serial)
        launch_boundary_scatter_serial(message.launch, stage->scatter_entries.opaque_handle(),
                                       arena, *impl_->communication);
      else
        launch_boundary_scatter_parallel(message.launch, stage->scatter_entries.opaque_handle(),
                                         arena, *impl_->communication);
      impl_->stats.scatter_launches =
          checked_counter_add(impl_->stats.scatter_launches, 1, "scatter launch");
    }
    impl_->communication->synchronize();
    if (!stage->plan.receives.empty())
      impl_->stats.scatter_unpack_nanoseconds = checked_counter_add(
          impl_->stats.scatter_unpack_nanoseconds, elapsed_nanoseconds(scatter_start),
          "scatter/unpack elapsed nanoseconds");
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown NVIDIA staged transport publication failure";
  }
  int local_failed = local_error.empty() ? 0 : 1, any_failed = 0;
  mpi_require(MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_LOR, communicator),
              communicator, "same-stage transport failure reconciliation");
  if (any_failed) {
    why = local_error.empty() ? "NVIDIA staged transport failed on another rank" : local_error;
    slot.phase = impl::slot_phase::idle;
    impl_->poisoned = true;
    if (stage->generation == std::numeric_limits<uint64_t>::max())
      fatal_transport(communicator, "transport slot generation overflow", 1);
    ++stage->generation;
    return false;
  }
  try {
    slot.transport_done.record(*impl_->communication);
    slot.transport_done.wait(compute);
  }
  catch (...) {
    fatal_transport(communicator, "CUDA transport completion fence", 1);
  }
  slot.phase = impl::slot_phase::idle;
  if (stage->generation == std::numeric_limits<uint64_t>::max())
    fatal_transport(communicator, "transport slot generation overflow", 1);
  ++stage->generation;
  return true;
#else
  (void)compute;
  why = "NVIDIA staged transport requires an MPI build";
  return false;
#endif
}

[[noreturn]] void staged_transport_epoch::abort_published_stage(field_type ft,
                                                                const char *operation) noexcept {
#ifdef HAVE_MPI
  MPI_Comm communicator = impl_ ? backend_communicator(impl_->communicator) : MPI_COMM_NULL;
  const impl::stage_state *stage = impl_ ? impl_->find(ft) : NULL;
  if (communicator != MPI_COMM_NULL && stage) {
    const size_t slot_index = size_t(stage->generation & 1u);
    if (stage->slots[slot_index].phase == impl::slot_phase::begun)
      fatal_transport(communicator, operation ? operation : "failure after request publication", 1);
  }
#else
  (void)ft;
  (void)operation;
#endif
  std::fprintf(stderr, "fatal NVIDIA staged transport ownership failure\n");
  std::fflush(stderr);
  std::abort();
}

bool staged_transport_epoch::participate_idle_stage(std::string &why) {
  why.clear();
  if (!impl_ || impl_->retired || impl_->poisoned || impl_->device >= 0 || !impl_->stages.empty()) {
    why = "NVIDIA idle transport participation has invalid ownership";
    return false;
  }
#ifdef HAVE_MPI
  if (!backend_mpi_thread_ready(why)) return false;
  MPI_Comm communicator = backend_communicator(impl_->communicator);
  int local_failed = 0, any_failed = 0;
  mpi_require(MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_LOR, communicator),
              communicator, "idle same-stage transport reconciliation");
  return any_failed == 0;
#else
  why = "NVIDIA staged transport requires an MPI build";
  return false;
#endif
}

bool staged_transport_epoch::retire(std::string &why) noexcept {
  why.clear();
  if (!impl_ || impl_->retired) return true;
  try {
    for (impl::stage_state &stage : impl_->stages)
      for (size_t i = 0; i < 2; ++i) {
        if (stage.slots[i].phase != impl::slot_phase::idle) {
          why = "cannot retire NVIDIA staged transport with a live request slot";
          return false;
        }
#ifdef HAVE_MPI
        for (MPI_Request &request : stage.slots[i].requests)
          if (!transport_request_is_null(&request)) {
            why = "cannot retire NVIDIA staged transport with a live MPI request";
            return false;
          }
#endif
      }
    if (impl_->device >= 0) {
      device_scope scope(impl_->device);
      if (impl_->communication) impl_->communication->synchronize();
      if (impl_->peer) {
        /* Peer stores (ours into mapped peer memory, and the acks peers send
           into our flags after consuming our last messages) must land before
           the mappings close and the flags are freed. No collective here:
           the wait is bounded so a peer that already exited cannot hang us. */
        synchronize_device(impl_->device);
        const transport_clock::time_point deadline =
            transport_clock::now() + std::chrono::seconds(30);
        for (size_t s = 0; s < impl_->stages.size(); ++s) {
          const size_t sends = impl_->stages[s].plan.sends.size();
          const uint64_t generation = impl_->stages[s].generation;
          if (!sends || !generation) continue;
          std::vector<uint64_t> acks(sends);
          for (;;) {
            read_peer_flags(impl_->peer_flags, impl_->peer_stages[s].ack_first, sends,
                            acks.data());
            if (*std::min_element(acks.begin(), acks.end()) >= generation) break;
            if (transport_clock::now() > deadline) {
              std::fprintf(stderr, "NVIDIA peer transport: retiring without final peer acks\n");
              break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
        }
        impl_->peer_mappings.clear();
        impl_->peer_stages.clear();
        impl_->peer_flags.reset();
        impl_->peer = false;
      }
      /* Requests are drained before entering this path. Destroy CUDA events and
         staging arenas while the owning device is selected, then retire the
         duplicated communicator collectively. */
      impl_->stages.clear();
    }
    else if (!impl_->stages.empty()) {
      why = "idle NVIDIA staged transport owns device stages";
      return false;
    }
  }
  catch (const std::exception &error) {
    why = error.what();
    return false;
  }
  catch (...) {
    why = "unknown NVIDIA staged transport drain failure";
    return false;
  }
  /* The active communicator context owns the duplicated MPI communicator.
     Transport retirement only releases this epoch's local borrow after every
     request and CUDA resource is gone; no executable teardown may initiate an
     MPI collective. */
  if (!retire_backend_communicator_lease(impl_->communicator, why)) return false;
  impl_->retired = true;
  why.clear();
  return true;
}

const staged_transport_statistics &staged_transport_epoch::statistics() const {
  if (!impl_) throw std::logic_error("NVIDIA staged transport has no implementation");
  return impl_->stats;
}

const transport_structural_identity &staged_transport_epoch::structural_identity() const {
  if (!impl_) throw std::logic_error("NVIDIA staged transport has no implementation");
  return impl_->identity;
}

bool staged_transport_epoch::structural_identity_matches_current(
    uint64_t device_signature, GpuMpiPolicy requested_policy,
    DependencyOverlapPolicy overlap_policy, std::string &why) const {
  why.clear();
  if (!impl_ || impl_->retired || !impl_->communicator.valid()) {
    why = "NVIDIA transport structural identity has no active communicator context";
    return false;
  }
  const transport_structural_identity &identity = impl_->identity;
  BackendCommunicatorLease current;
  if (!create_backend_communicator_lease(current, why)) return false;
  const BackendCommunicatorInfo &current_info = current.info();
  GpuMpiRoute route = GpuMpiRoute::staged;
  if (!resolve_gpu_mpi_route(requested_policy, current_info.provider_query_available,
                             current_info.provider_supports_direct, route, why))
    return false;
  const bool route_is_current =
      identity.resolved_route == route ||
      (requested_policy == GpuMpiPolicy::automatic && route == GpuMpiRoute::direct &&
       identity.resolved_route == GpuMpiRoute::staged);
  if (identity.signature != compute_transport_structural_identity_signature(identity) ||
      identity.communicator_generation != current_info.generation ||
      identity.communicator_rank != current_info.rank ||
      identity.communicator_size != current_info.size || identity.tag_ub != current_info.tag_ub ||
      identity.requested_policy != requested_policy || !route_is_current ||
      identity.overlap_policy != overlap_policy || identity.device_signature != device_signature ||
      identity.provider_signature != provider_identity(current_info)) {
    why = "NVIDIA transport structural identity is stale";
    return false;
  }
  return true;
}

namespace {

/* Fixed-size records exchanged once per epoch to wire the peer route. */
struct peer_record {
  peer_ipc_handle flags;
  peer_ipc_handle arenas[NUM_FIELD_TYPES];
  unsigned char has_arena[NUM_FIELD_TYPES];
};

struct peer_entry {
  int32_t outgoing; // 0: a receive on the publishing rank, 1: a send
  int32_t ft;
  int32_t source_rank;
  int32_t destination_rank;
  int32_t source_chunk;
  int32_t destination_chunk;
  int32_t tag;
  int32_t reserved;
  uint64_t canonical_ordinal;
  uint64_t arena_offsets[2];
  uint64_t wire_bytes;
  uint64_t flag_index;
};

bool peer_entry_matches(const peer_entry &entry, const RemoteHaloWireKey &key) {
  return entry.ft == int32_t(key.ft) && entry.source_rank == key.source_rank &&
         entry.destination_rank == key.destination_rank &&
         entry.source_chunk == key.source_chunk &&
         entry.destination_chunk == key.destination_chunk && entry.tag == key.tag &&
         entry.canonical_ordinal == key.canonical_ordinal;
}

/* -1 invalid, 0 off, 1 auto, 2 required. */
int peer_policy_from_environment() {
  const char *value = std::getenv("MEEP_NVIDIA_PEER_TRANSPORT");
  if (!value || !*value || !std::strcmp(value, "auto")) return 1;
  if (!std::strcmp(value, "off") || !std::strcmp(value, "no") || !std::strcmp(value, "0"))
    return 0;
  if (!std::strcmp(value, "required") || !std::strcmp(value, "yes") || !std::strcmp(value, "1"))
    return 2;
  return -1;
}

double peer_timeout_from_environment() {
  const char *value = std::getenv("MEEP_NVIDIA_PEER_TIMEOUT");
  if (!value || !*value) return 1800.0;
  char *end = NULL;
  const double seconds = std::strtod(value, &end);
  return end && !*end && seconds >= 0 ? seconds : 1800.0;
}

} // namespace

bool staged_transport_epoch::enable_peer_route(std::string &why) {
  why.clear();
#ifdef HAVE_MPI
  impl &state = *impl_;
  MPI_Comm communicator = backend_communicator(state.communicator);
  const int rank = state.communicator.info().rank;
  const int size = state.communicator.info().size;

  const int policy = peer_policy_from_environment();
  int policy_min = 0, policy_max = 0;
  mpi_require(MPI_Allreduce(&policy, &policy_min, 1, MPI_INT, MPI_MIN, communicator),
              communicator, "peer policy reconciliation");
  mpi_require(MPI_Allreduce(&policy, &policy_max, 1, MPI_INT, MPI_MAX, communicator),
              communicator, "peer policy reconciliation");
  if (policy_min < 0 || (policy_min != policy_max && policy_max == 2)) {
    why = "MEEP_NVIDIA_PEER_TRANSPORT must be off, auto or required, identically on every rank";
    return false;
  }
  if (policy_min != policy_max || policy_min == 0) return true;
  const bool required = policy_min == 2;

  /* Every phase below runs every collective even after a local failure, so
     ranks agree on the outcome without diverging control flow. */
  std::string local_error;
  int failed = 0, any_failed = 0;
  if (!state.wire.participation.device_owner || state.device < 0 || !state.communication)
    local_error = "a rank owns no device chunks";

  MPI_Comm node = MPI_COMM_NULL;
  int node_size = 0;
  mpi_require(MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node),
              communicator, "peer node split");
  mpi_require(MPI_Comm_size(node, &node_size), communicator, "peer node size");
  mpi_require(MPI_Comm_free(&node), communicator, "peer node free");
  if (node_size != size && local_error.empty()) local_error = "ranks span more than one node";

  const int bus_bytes = 32;
  std::vector<char> bus(bus_bytes, 0), buses(size_t(bus_bytes) * size, 0);
  if (local_error.empty()) try {
      std::strncpy(bus.data(), device_pci_bus_id(state.device).c_str(), bus_bytes - 1);
    }
    catch (const std::exception &error) {
      local_error = error.what();
    }
  mpi_require(MPI_Allgather(bus.data(), bus_bytes, MPI_CHAR, buses.data(), bus_bytes, MPI_CHAR,
                            communicator),
              communicator, "peer bus-id exchange");

  std::vector<bool> talks_to(size, false);
  size_t total_receives = 0, total_sends = 0;
  for (const impl::stage_state &stage : state.stages) {
    for (const bound_boundary_message &message : stage.plan.sends)
      if (message.key.destination_rank >= 0 && message.key.destination_rank < size)
        talks_to[message.key.destination_rank] = true;
    for (const bound_boundary_message &message : stage.plan.receives)
      if (message.key.source_rank >= 0 && message.key.source_rank < size)
        talks_to[message.key.source_rank] = true;
    total_receives += stage.plan.receives.size();
    total_sends += stage.plan.sends.size();
  }
  for (int r = 0; local_error.empty() && r < size; ++r) {
    if (!talks_to[r] || r == rank) continue;
    const int peer_device = device_for_pci_bus_id(std::string(&buses[size_t(r) * bus_bytes]));
    if (peer_device < 0)
      local_error = "the GPU of rank " + std::to_string(r) + " is not visible to rank " +
                    std::to_string(rank);
    else if (peer_device != state.device && !devices_can_access_peer(state.device, peer_device))
      local_error = "no CUDA peer access between the GPUs of ranks " + std::to_string(rank) +
                    " and " + std::to_string(r);
  }
  failed = local_error.empty() ? 0 : 1;
  mpi_require(MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_LOR, communicator), communicator,
              "peer capability reconciliation");
  if (any_failed) {
    if (!required) return true;
    why = "MEEP_NVIDIA_PEER_TRANSPORT=required: " +
          (local_error.empty() ? std::string("unavailable on another rank") : local_error);
    return false;
  }

  /* Phase 2: local flags, exported handles and the message tables. */
  peer_record record;
  std::memset(&record, 0, sizeof(record));
  std::vector<peer_entry> entries;
  try {
    device_scope scope(state.device);
    state.peer_stages.clear();
    state.peer_stages.resize(state.stages.size());
    size_t next_flag = 0;
    for (size_t s = 0; s < state.stages.size(); ++s) {
      const impl::stage_state &stage = state.stages[s];
      impl::peer_stage &peer = state.peer_stages[s];
      peer.ready_first = next_flag;
      next_flag += stage.plan.receives.size();
      peer.ack_first = next_flag;
      next_flag += stage.plan.sends.size();
      const int ft = int(stage.plan.ft);
      if (stage.receive_arena.opaque_handle() && ft >= 0 && ft < NUM_FIELD_TYPES) {
        record.arenas[ft] = export_peer_handle(stage.receive_arena);
        record.has_arena[ft] = 1;
      }
      for (size_t i = 0; i < stage.plan.receives.size(); ++i) {
        const bound_boundary_message &message = stage.plan.receives[i];
        peer_entry entry = {};
        entry.outgoing = 0;
        entry.ft = int32_t(message.key.ft);
        entry.source_rank = message.key.source_rank;
        entry.destination_rank = message.key.destination_rank;
        entry.source_chunk = message.key.source_chunk;
        entry.destination_chunk = message.key.destination_chunk;
        entry.tag = message.key.tag;
        entry.canonical_ordinal = message.key.canonical_ordinal;
        entry.arena_offsets[0] = message.arena_offsets[0];
        entry.arena_offsets[1] = message.arena_offsets[1];
        entry.wire_bytes = message.wire_bytes;
        entry.flag_index = peer.ready_first + i;
        entries.push_back(entry);
      }
      for (size_t i = 0; i < stage.plan.sends.size(); ++i) {
        const bound_boundary_message &message = stage.plan.sends[i];
        peer_entry entry = {};
        entry.outgoing = 1;
        entry.ft = int32_t(message.key.ft);
        entry.source_rank = message.key.source_rank;
        entry.destination_rank = message.key.destination_rank;
        entry.source_chunk = message.key.source_chunk;
        entry.destination_chunk = message.key.destination_chunk;
        entry.tag = message.key.tag;
        entry.canonical_ordinal = message.key.canonical_ordinal;
        entry.wire_bytes = message.wire_bytes;
        entry.flag_index = peer.ack_first + i;
        entries.push_back(entry);
      }
    }
    if (next_flag != total_receives + total_sends)
      throw std::logic_error("NVIDIA peer flag layout is inconsistent");
    const size_t flag_bytes = std::max<size_t>(next_flag, 1) * sizeof(uint64_t);
    state.peer_flags.allocate(flag_bytes, state.device);
    fill_byte_async(state.peer_flags, 0, 0, flag_bytes, *state.communication);
    state.communication->synchronize();
    record.flags = export_peer_handle(state.peer_flags);
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  std::vector<peer_record> records(size);
  mpi_require(MPI_Allgather(&record, int(sizeof(record)), MPI_BYTE, records.data(),
                            int(sizeof(record)), MPI_BYTE, communicator),
              communicator, "peer handle exchange");
  int local_count = local_error.empty() ? int(entries.size()) : 0;
  std::vector<int> counts(size), byte_counts(size), displacements(size);
  mpi_require(MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, communicator),
              communicator, "peer table size exchange");
  size_t all_entries = 0;
  for (int r = 0; r < size; ++r) {
    byte_counts[r] = counts[r] * int(sizeof(peer_entry));
    displacements[r] = int(all_entries * sizeof(peer_entry));
    all_entries += size_t(counts[r]);
  }
  std::vector<peer_entry> table(std::max<size_t>(all_entries, 1));
  mpi_require(MPI_Allgatherv(entries.data(), local_count * int(sizeof(peer_entry)), MPI_BYTE,
                             table.data(), byte_counts.data(), displacements.data(), MPI_BYTE,
                             communicator),
              communicator, "peer table exchange");

  /* Phase 3: map peer allocations and resolve every message endpoint. */
  if (local_error.empty()) try {
      device_scope scope(state.device);
      std::map<int, size_t> flag_mapping;
      std::map<std::pair<int, int>, size_t> arena_mapping;
      auto mapped = [&](const peer_ipc_handle &handle) -> size_t {
        state.peer_mappings.emplace_back();
        state.peer_mappings.back().open(handle, state.device);
        return state.peer_mappings.size() - 1;
      };
      /* A rank's own allocations are reached through the same IPC path only
         if a message ever targets self; MPI never routes those here. */
      auto find_entry = [&](int owner, int outgoing,
                            const RemoteHaloWireKey &key) -> const peer_entry & {
        const peer_entry *found = NULL;
        size_t first = size_t(displacements[owner]) / sizeof(peer_entry);
        for (size_t k = first; k < first + size_t(counts[owner]); ++k)
          if (table[k].outgoing == outgoing && peer_entry_matches(table[k], key)) {
            if (found) throw std::logic_error("NVIDIA peer message key is ambiguous");
            found = &table[k];
          }
        if (!found)
          throw std::logic_error("NVIDIA peer message has no matching endpoint on rank " +
                                 std::to_string(owner));
        return *found;
      };
      auto flags_of = [&](int owner) -> uint64_t * {
        std::map<int, size_t>::iterator it = flag_mapping.find(owner);
        if (it == flag_mapping.end())
          it = flag_mapping.insert(std::make_pair(owner, mapped(records[owner].flags))).first;
        return static_cast<uint64_t *>(state.peer_mappings[it->second].address());
      };
      auto arena_of = [&](int owner, int ft) -> unsigned char * {
        if (ft < 0 || ft >= NUM_FIELD_TYPES || !records[owner].has_arena[ft])
          throw std::logic_error("NVIDIA peer receive arena is missing");
        const std::pair<int, int> key(owner, ft);
        std::map<std::pair<int, int>, size_t>::iterator it = arena_mapping.find(key);
        if (it == arena_mapping.end())
          it = arena_mapping.insert(std::make_pair(key, mapped(records[owner].arenas[ft]))).first;
        return static_cast<unsigned char *>(state.peer_mappings[it->second].address());
      };
      state.peer_mappings.reserve(2 * size_t(size) * (1 + NUM_FIELD_TYPES));
      for (size_t s = 0; s < state.stages.size(); ++s) {
        const impl::stage_state &stage = state.stages[s];
        impl::peer_stage &peer = state.peer_stages[s];
        std::vector<uint64_t *> ready_targets, ack_targets;
        for (const bound_boundary_message &message : stage.plan.sends) {
          const int owner = message.key.destination_rank;
          if (owner < 0 || owner >= size || owner == rank)
            throw std::logic_error("NVIDIA peer send has an invalid destination");
          const peer_entry &entry = find_entry(owner, 0, message.key);
          if (entry.wire_bytes != message.wire_bytes)
            throw std::logic_error("NVIDIA peer message sizes disagree");
          unsigned char *arena = arena_of(owner, entry.ft);
          peer.send_targets[0].push_back(arena + entry.arena_offsets[0]);
          peer.send_targets[1].push_back(arena + entry.arena_offsets[1]);
          ready_targets.push_back(flags_of(owner) + entry.flag_index);
        }
        for (const bound_boundary_message &message : stage.plan.receives) {
          const int owner = message.key.source_rank;
          if (owner < 0 || owner >= size || owner == rank)
            throw std::logic_error("NVIDIA peer receive has an invalid source");
          const peer_entry &entry = find_entry(owner, 1, message.key);
          ack_targets.push_back(flags_of(owner) + entry.flag_index);
        }
        if (!ready_targets.empty()) {
          const size_t bytes = ready_targets.size() * sizeof(uint64_t *);
          peer.ready_targets.allocate(bytes, state.device);
          copy_host_to_device_async(peer.ready_targets, 0, ready_targets.data(), bytes,
                                    *state.communication);
        }
        if (!ack_targets.empty()) {
          const size_t bytes = ack_targets.size() * sizeof(uint64_t *);
          peer.ack_targets.allocate(bytes, state.device);
          copy_host_to_device_async(peer.ack_targets, 0, ack_targets.data(), bytes,
                                    *state.communication);
        }
      }
      state.communication->synchronize();
    }
    catch (const std::exception &error) {
      local_error = error.what();
    }
  failed = local_error.empty() ? 0 : 1;
  mpi_require(MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_LOR, communicator), communicator,
              "peer mapping reconciliation");
  if (any_failed) {
    try {
      device_scope scope(state.device >= 0 ? state.device : 0);
      state.peer_mappings.clear();
      state.peer_stages.clear();
      state.peer_flags.reset();
    }
    catch (...) {
      fatal_transport(communicator, "peer route cleanup", 1);
    }
    if (!required) {
      if (rank == 0)
        std::fprintf(stderr, "meep: NVIDIA peer halo transport unavailable (%s); using MPI\n",
                     local_error.empty() ? "failed on another rank" : local_error.c_str());
      return true;
    }
    why = "MEEP_NVIDIA_PEER_TRANSPORT=required: " +
          (local_error.empty() ? std::string("failed on another rank") : local_error);
    return false;
  }
  state.peer = true;
  state.peer_timeout_seconds = peer_timeout_from_environment();
  if (rank == 0)
    std::printf("meep: NVIDIA halo exchange uses CUDA IPC peer stores (%d ranks, one node)\n",
                size);
  return true;
#else
  return true;
#endif
}

std::unique_ptr<staged_transport_epoch>
create_staged_transport_epoch(const compiled_boundary_artifact &artifact, int device,
                              stream *communication, std::string &why,
                              uint64_t device_signature, uint64_t dependency_signature,
                              DependencyOverlapPolicy overlap_policy) {
  why.clear();
  if (artifact.wire.communicator_size <= 1) return std::unique_ptr<staged_transport_epoch>();
  std::unique_ptr<staged_transport_epoch> result;
  std::string construction_error;
  try {
    result.reset(new staged_transport_epoch);
    result->impl_->device = device;
    result->impl_->communication = communication;
    result->impl_->wire = artifact.wire;
  }
  catch (const std::exception &error) {
    construction_error = error.what();
  }
  catch (...) {
    construction_error = "unknown NVIDIA staged transport host allocation failure";
  }
#ifdef HAVE_MPI
  MPI_Comm source_communicator = current_backend_communicator();
  int local_construction_failed = construction_error.empty() ? 0 : 1;
  int any_construction_failed = 0;
  mpi_require(MPI_Allreduce(&local_construction_failed, &any_construction_failed, 1, MPI_INT,
                            MPI_LOR, source_communicator),
              source_communicator, "transport host-allocation reconciliation");
  if (any_construction_failed) {
    why = construction_error.empty()
              ? "NVIDIA staged transport host allocation failed on another rank"
              : construction_error;
    return NULL;
  }
#else
  if (!construction_error.empty()) {
    why = construction_error;
    return NULL;
  }
#endif
  if (!create_backend_communicator_lease(result->impl_->communicator, why)) return NULL;
  result->impl_->identity.communicator_generation =
      result->impl_->communicator.info().generation;
  result->impl_->identity.communicator_rank = result->impl_->communicator.info().rank;
  result->impl_->identity.communicator_size = result->impl_->communicator.info().size;
  result->impl_->identity.tag_ub = result->impl_->communicator.info().tag_ub;
  result->impl_->identity.requested_policy = artifact.wire.requested_policy;
  result->impl_->identity.resolved_route = artifact.wire.resolved_route;
  result->impl_->identity.overlap_policy = overlap_policy;
  result->impl_->identity.wire_signature = artifact.wire.signature;
  result->impl_->identity.authority_signature = artifact.bound.authority_signature;
  result->impl_->identity.provider_signature =
      provider_identity(result->impl_->communicator.info());
  result->impl_->identity.device_signature = device_signature;
  result->impl_->identity.arena_layout_signature = arena_layout_identity(artifact);
  result->impl_->identity.dependency_signature = dependency_signature;
#ifdef HAVE_MPI
  {
    uint64_t local_owner = UINT64_C(1469598103934665603);
    local_owner = identity_mix(local_owner,
                               uint64_t(result->impl_->communicator.info().rank));
    local_owner = identity_mix(local_owner,
                               artifact.wire.participation.device_owner ? 1 : 0);
    local_owner = identity_mix(local_owner, device_signature);
    const unsigned long long local = static_cast<unsigned long long>(local_owner);
    unsigned long long owners = 0;
    mpi_require(MPI_Allreduce(&local, &owners, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR,
                              backend_communicator(result->impl_->communicator)),
                backend_communicator(result->impl_->communicator),
                "transport owner-map reconciliation");
    result->impl_->identity.owner_map_signature = static_cast<uint64_t>(owners);
  }
#else
  result->impl_->identity.owner_map_signature =
      artifact.wire.participation.device_owner ? (device_signature ? device_signature : 1) : 0;
#endif
  result->impl_->identity.signature =
      compute_transport_structural_identity_signature(result->impl_->identity);
  std::string authority_error;
  if (artifact.wire.resolved_route == GpuMpiRoute::direct &&
      consume_failure(testing::staged_transport_failure_point::direct_validation))
    authority_error = "injected NVIDIA direct MPI transport validation failure";
  else if (artifact.wire.resolved_route != GpuMpiRoute::staged &&
           artifact.wire.resolved_route != GpuMpiRoute::direct)
    authority_error = "NVIDIA MPI transport route is invalid";
  else if (artifact.wire.participation.device_owner != (device >= 0 && communication != NULL))
    authority_error = "NVIDIA staged transport device ownership is inconsistent";
  else if (!artifact.wire.participation.device_owner &&
           (!artifact.bound.stages.empty() || !artifact.wire.stages.empty()))
    authority_error = "idle NVIDIA staged transport rank has device boundary stages";
  else if (artifact.bound.program_signature != artifact.wire.signature ||
           artifact.bound.stages.size() != artifact.wire.stages.size())
    authority_error = "NVIDIA staged transport artifact authority is stale";
  for (size_t i = 0; authority_error.empty() && i < artifact.bound.stages.size(); ++i) {
    const bound_boundary_stage &bound = artifact.bound.stages[i];
    const RemoteHaloStage &wire = artifact.wire.stages[i];
    if (bound.ft != wire.ft || bound.receives.size() != wire.receives.size() ||
        bound.sends.size() != wire.sends.size() || bound.receives.size() > size_t(INT_MAX) ||
        bound.sends.size() > size_t(INT_MAX) - bound.receives.size())
      authority_error = "NVIDIA staged transport stage does not match wire authority";
    for (size_t j = 0; authority_error.empty() && j < bound.receives.size(); ++j)
      if (bound.receives[j].key != wire.receives[j].key ||
          bound.receives[j].direction != RemoteHaloDirection::incoming ||
          bound.receives[j].wire_bytes != wire.receives[j].wire_bytes ||
          bound.receives[j].wire_bytes > size_t(INT_MAX))
        authority_error = "NVIDIA staged transport receive does not match wire authority";
    for (size_t j = 0; authority_error.empty() && j < bound.sends.size(); ++j)
      if (bound.sends[j].key != wire.sends[j].key ||
          bound.sends[j].direction != RemoteHaloDirection::outgoing ||
          bound.sends[j].wire_bytes != wire.sends[j].wire_bytes ||
          bound.sends[j].wire_bytes > size_t(INT_MAX))
        authority_error = "NVIDIA staged transport send does not match wire authority";
  }
#ifdef HAVE_MPI
  MPI_Comm authority_communicator = backend_communicator(result->impl_->communicator);
  int local_authority_failed = authority_error.empty() ? 0 : 1;
  int any_authority_failed = 0;
  mpi_require(MPI_Allreduce(&local_authority_failed, &any_authority_failed, 1, MPI_INT, MPI_LOR,
                            authority_communicator),
              authority_communicator, "transport authority reconciliation");
  if (any_authority_failed) {
    std::string retire_error;
    (void)retire_backend_communicator_lease(result->impl_->communicator, retire_error);
    why = authority_error.empty() ? "NVIDIA staged transport authority failed on another rank"
                                  : authority_error;
    result->impl_->retired = true;
    return NULL;
  }
#else
  if (!authority_error.empty()) {
    why = authority_error;
    return NULL;
  }
#endif
  if (!collective_validate_remote_halo_agreement(result->impl_->communicator, artifact.wire, why)) {
    std::string retire_error;
    if (!retire_backend_communicator_lease(result->impl_->communicator, retire_error) && why.empty())
      why = retire_error;
    return NULL;
  }

  bool global_stage[NUM_FIELD_TYPES] = {};
#ifdef HAVE_MPI
  int local_stage[NUM_FIELD_TYPES] = {}, reconciled_stage[NUM_FIELD_TYPES] = {};
  for (const bound_boundary_stage &stage : artifact.bound.stages)
    if (int(stage.ft) >= 0 && int(stage.ft) < NUM_FIELD_TYPES) local_stage[int(stage.ft)] = 1;
  MPI_Comm stage_communicator = backend_communicator(result->impl_->communicator);
  mpi_require(MPI_Allreduce(local_stage, reconciled_stage, NUM_FIELD_TYPES, MPI_INT, MPI_LOR,
                            stage_communicator),
              stage_communicator, "transport stage-presence reconciliation");
  for (int ft = 0; ft < NUM_FIELD_TYPES; ++ft)
    global_stage[ft] = reconciled_stage[ft] != 0;
#else
  for (const bound_boundary_stage &stage : artifact.bound.stages)
    global_stage[int(stage.ft)] = true;
#endif

  std::string local_error;
  try {
    std::unique_ptr<device_scope> scope;
    if (device >= 0) scope.reset(new device_scope(device));
    result->impl_->stages.reserve(NUM_FIELD_TYPES);
    for (int ft = 0; ft < NUM_FIELD_TYPES; ++ft) {
      if (!artifact.wire.participation.device_owner || !global_stage[ft]) continue;
      const bound_boundary_stage *local_plan = NULL;
      for (const bound_boundary_stage &candidate : artifact.bound.stages)
        if (int(candidate.ft) == ft) local_plan = &candidate;
      bound_boundary_stage empty_plan;
      empty_plan.ft = field_type(ft);
      empty_plan.receive_slot_bytes = 0;
      empty_plan.send_slot_bytes = 0;
      empty_plan.publication = RemoteHaloPublicationMode::parallel_unique;
      const bound_boundary_stage &plan = local_plan ? *local_plan : empty_plan;
      result->impl_->stages.emplace_back();
      staged_transport_epoch::impl::stage_state &stage = result->impl_->stages.back();
      stage.plan = plan;
      if (plan.gathers.size() >
              std::numeric_limits<size_t>::max() / sizeof(boundary_gather_entry) ||
          plan.scatters.size() >
              std::numeric_limits<size_t>::max() / sizeof(boundary_scatter_entry))
        throw std::overflow_error("NVIDIA MPI descriptor byte count overflows");
      const size_t gather_bytes = plan.gathers.size() * sizeof(boundary_gather_entry);
      const size_t scatter_bytes = plan.scatters.size() * sizeof(boundary_scatter_entry);
      const size_t send_bytes =
          checked_size_add(plan.send_slot_bytes, plan.send_slot_bytes, "send slots");
      const size_t receive_bytes =
          checked_size_add(plan.receive_slot_bytes, plan.receive_slot_bytes, "receive slots");
      if (gather_bytes) {
        stage.gather_entries.allocate(gather_bytes, device);
        copy_host_to_device_async(stage.gather_entries, 0, plan.gathers.data(), gather_bytes,
                                  *communication);
      }
      if (scatter_bytes) {
        stage.scatter_entries.allocate(scatter_bytes, device);
        copy_host_to_device_async(stage.scatter_entries, 0, plan.scatters.data(), scatter_bytes,
                                  *communication);
      }
      if (send_bytes) {
        stage.send_arena.allocate(send_bytes, device);
        if (artifact.wire.resolved_route == GpuMpiRoute::staged)
          stage.send_staging.allocate(send_bytes);
      }
      if (receive_bytes) {
        stage.receive_arena.allocate(receive_bytes, device);
        if (artifact.wire.resolved_route == GpuMpiRoute::staged)
          stage.receive_staging.allocate(receive_bytes);
      }
      result->impl_->stats.device_bytes = checked_size_add(
          result->impl_->stats.device_bytes,
          checked_size_add(checked_size_add(gather_bytes, scatter_bytes, "descriptor bytes"),
                           checked_size_add(send_bytes, receive_bytes, "arena bytes"),
                           "device epoch bytes"),
          "device epoch total");
      if (artifact.wire.resolved_route == GpuMpiRoute::staged)
        result->impl_->stats.pinned_bytes = checked_size_add(
            result->impl_->stats.pinned_bytes,
            checked_size_add(send_bytes, receive_bytes, "pinned epoch bytes"),
            "pinned epoch total");
    }
    if (communication) communication->synchronize();
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown NVIDIA staged transport allocation failure";
  }

#ifdef HAVE_MPI
  MPI_Comm communicator = backend_communicator(result->impl_->communicator);
  int local_failed = local_error.empty() ? 0 : 1, any_failed = 0;
  mpi_require(MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_LOR, communicator),
              communicator, "transport allocation reconciliation");
  if (any_failed) {
    try {
      if (device >= 0) {
        device_scope scope(device);
        result->impl_->stages.clear();
      }
      else
        result->impl_->stages.clear();
    }
    catch (...) {
      fatal_transport(communicator, "transport allocation cleanup", 1);
    }
    std::string retire_error;
    (void)retire_backend_communicator_lease(result->impl_->communicator, retire_error);
    why = local_error.empty() ? "NVIDIA staged transport allocation failed on another rank"
                              : local_error;
    result->impl_->retired = true;
    return NULL;
  }
#else
  if (!local_error.empty()) {
    why = local_error;
    return NULL;
  }
#endif
  if (!result->enable_peer_route(why)) {
    try {
      if (device >= 0) {
        device_scope scope(device);
        result->impl_->stages.clear();
      }
    }
    catch (...) {
    }
    std::string retire_error;
    (void)retire_backend_communicator_lease(result->impl_->communicator, retire_error);
    result->impl_->retired = true;
    return NULL;
  }
  return result;
}

std::unique_ptr<staged_transport_epoch>
create_transport_epoch_with_fallback(compiled_boundary_artifact &artifact, int device,
                                     stream *communication, GpuMpiPolicy agreed_policy,
                                     DependencyOverlapPolicy overlap_policy, bool &fell_back,
                                     std::string &why, uint64_t device_signature,
                                     uint64_t dependency_signature) {
  fell_back = false;
  std::unique_ptr<staged_transport_epoch> result =
      create_staged_transport_epoch(artifact, device, communication, why, device_signature,
                                    dependency_signature, overlap_policy);
  if (result || artifact.wire.resolved_route != GpuMpiRoute::direct ||
      agreed_policy != GpuMpiPolicy::automatic ||
      overlap_policy == DependencyOverlapPolicy::required)
    return result;

  compiled_boundary_artifact fallback = artifact;
  fallback.wire.resolved_route = GpuMpiRoute::staged;
  fallback.wire.signature = compute_remote_halo_program_signature(fallback.wire);
  fallback.lowered.program_signature = fallback.wire.signature;
  fallback.lowered.authority_signature = compute_remote_lowered_authority_signature(
      fallback.lowered.program_signature, fallback.lowered.storage_signature);
  fallback.bound.program_signature = fallback.lowered.program_signature;
  fallback.bound.authority_signature = fallback.lowered.authority_signature;
  why.clear();
  result = create_staged_transport_epoch(fallback, device, communication, why, device_signature,
                                         0, overlap_policy);
  if (result) {
    artifact = std::move(fallback);
    fell_back = true;
  }
  return result;
}

namespace testing {
transport_presend_action transport_presend_action_for_testing(
    GpuMpiRoute route, bool gather_ready, bool staging_ready, bool copies_enqueued) {
  return presend_action(route, gather_ready, staging_ready, copies_enqueued);
}
void set_mpi_request_operations_for_testing(mpi_request_operations *operations) {
  injected_request_operations = operations;
}
void fail_staged_transport_once(staged_transport_failure_point point) { injected_failure = point; }
void set_staged_transport_minimum_presend_polls(size_t polls) { minimum_presend_polls = polls; }
} // namespace testing

} // namespace nvidia
} // namespace meep
