/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_mpi.hpp"

#include "backend/mpi_context.hpp"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace meep {
namespace nvidia {
namespace {

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
      high_water_requests(0), device_bytes(0), pinned_bytes(0) {}

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

  int device;
  stream *communication;
  BackendCommunicatorLease communicator;
  RemoteHaloProgram wire;
  std::vector<stage_state> stages;
  staged_transport_statistics stats;
  transport_structural_identity identity;
  bool retired;
  bool poisoned;

  impl() : device(-1), communication(NULL), retired(false), poisoned(false) {}

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

  size_t waited_requests = 0;
  for (MPI_Request &request : slot.requests)
    if (!transport_request_is_null(&request)) ++waited_requests;
  if (!slot.requests.empty()) {
    mpi_require(transport_waitall(slot.requests, slot.statuses.data()),
                communicator, "MPI_Waitall after all sends are posted");
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
    if (direct)
      for (const bound_boundary_message &message : stage->plan.sends)
        impl_->stats.direct_bytes =
            checked_counter_add(impl_->stats.direct_bytes, message.wire_bytes, "direct bytes");
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
