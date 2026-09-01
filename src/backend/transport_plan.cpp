/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/transport_plan.hpp"

#include <algorithm>
#include <climits>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>

#include "backend/halo_plan.hpp"
#include "backend/precision.hpp"
#include "backend/storage_plan.hpp"

namespace meep {
namespace {

uint64_t hash_u64(uint64_t h, uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) {
    h ^= (value >> (8 * i)) & 0xffu;
    h *= UINT64_C(1099511628211);
  }
  return h;
}

uint64_t hash_key(uint64_t h, const RemoteHaloWireKey &key) {
  h = hash_u64(h, uint64_t(uint32_t(key.source_rank)));
  h = hash_u64(h, uint64_t(uint32_t(key.destination_rank)));
  h = hash_u64(h, uint64_t(uint32_t(key.ft)));
  h = hash_u64(h, uint64_t(uint32_t(key.source_chunk)));
  h = hash_u64(h, uint64_t(uint32_t(key.destination_chunk)));
  h = hash_u64(h, uint64_t(uint32_t(key.tag)));
  return hash_u64(h, key.canonical_ordinal);
}

bool checked_multiply(size_t a, size_t b, size_t &result) {
  if (a && b > std::numeric_limits<size_t>::max() / a) return false;
  result = a * b;
  return true;
}

bool valid_policy(GpuMpiPolicy policy) {
  return policy == GpuMpiPolicy::staged || policy == GpuMpiPolicy::automatic ||
         policy == GpuMpiPolicy::direct;
}

bool valid_route(GpuMpiRoute route) {
  return route == GpuMpiRoute::staged || route == GpuMpiRoute::direct;
}

uint64_t stage_signature(const RemoteHaloStage &stage) {
  uint64_t h = UINT64_C(1469598103934665603);
  h = hash_u64(h, uint64_t(uint32_t(stage.ft)));
  for (const RemoteHaloMessage &message : stage.receives) {
    h = hash_u64(h, uint64_t(RemoteHaloDirection::incoming));
    h = hash_u64(h, message.wire_digest);
    h = hash_u64(h, message.local_schedule_ordinal);
  }
  for (const RemoteHaloMessage &message : stage.sends) {
    h = hash_u64(h, uint64_t(RemoteHaloDirection::outgoing));
    h = hash_u64(h, message.wire_digest);
    h = hash_u64(h, message.local_schedule_ordinal);
  }
  return h;
}

bool collect_precision(const HaloPlan &plan, RemoteHaloDirection direction,
                       const StoragePlan &storage, bool &have_precision,
                       Precision &precision, std::string &why) {
  std::vector<ElementRef> refs;
  if (direction == RemoteHaloDirection::outgoing)
    expand_gather(plan, refs);
  else
    expand_scatter(plan, refs);
  if (refs.size() != plan.block_elements) {
    why = "canonical halo endpoint element count disagrees with block span";
    return false;
  }
  for (const ElementRef &ref : refs) {
    if (ref.array.value >= storage.arrays.size()) {
      why = "remote halo references an absent canonical array";
      return false;
    }
    const ArraySpec &spec = storage.arrays[ref.array.value];
    if (spec.id != ref.array || spec.classification_provisional || spec.classification_elided) {
      why = "remote halo references provisional, stale, or elided canonical storage";
      return false;
    }
    if (ref.index < 0 || size_t(ref.index) >= spec.elements) {
      why = "remote halo canonical element index is out of range";
      return false;
    }
    if (!have_precision) {
      precision = spec.storage;
      have_precision = true;
    }
    else if (precision != spec.storage) {
      why = "remote halo communication block mixes storage precision";
      return false;
    }
  }
  return true;
}

bool build_message(const comms_operation &op, uint32_t local_ordinal, field_type ft,
                   RemoteHaloDirection direction, const halo_plan_set &halos,
                   const StoragePlan &storage, const std::vector<int> &chunk_ranks,
                   int communicator_rank, RemoteHaloMessage &message, std::string &why) {
  const int source_chunk = direction == RemoteHaloDirection::outgoing
                               ? int(op.my_chunk_idx)
                               : int(op.other_chunk_idx);
  const int destination_chunk = direction == RemoteHaloDirection::outgoing
                                    ? int(op.other_chunk_idx)
                                    : int(op.my_chunk_idx);
  const in_or_out expected_direction =
      direction == RemoteHaloDirection::outgoing ? Outgoing : Incoming;
  if (source_chunk < 0 || destination_chunk < 0 ||
      size_t(source_chunk) >= chunk_ranks.size() ||
      size_t(destination_chunk) >= chunk_ranks.size()) {
    why = "remote halo communication operation has an invalid chunk index";
    return false;
  }
  const size_t expected_pair = size_t(source_chunk) + size_t(destination_chunk) * chunk_ranks.size();
  if (op.comm_direction != expected_direction || op.pair_idx < 0 ||
      size_t(op.pair_idx) != expected_pair) {
    why = "remote halo communication operation direction or pair index is inconsistent";
    return false;
  }
  const int source_rank = chunk_ranks[source_chunk];
  const int destination_rank = chunk_ranks[destination_chunk];
  if ((direction == RemoteHaloDirection::outgoing && source_rank != communicator_rank) ||
      (direction == RemoteHaloDirection::incoming && destination_rank != communicator_rank) ||
      op.other_proc_id !=
          (direction == RemoteHaloDirection::outgoing ? destination_rank : source_rank)) {
    why = "remote halo operation ownership or peer rank is inconsistent";
    return false;
  }
  if (source_rank == destination_rank) {
    why = "same-rank communication operation reached remote message lowering";
    return false;
  }

  message = RemoteHaloMessage();
  message.key.source_rank = source_rank;
  message.key.destination_rank = destination_rank;
  message.key.ft = ft;
  message.key.source_chunk = source_chunk;
  message.key.destination_chunk = destination_chunk;
  message.key.tag = op.tag;
  message.key.canonical_ordinal =
      uint64_t(uint32_t(ft)) * uint64_t(chunk_ranks.size()) * uint64_t(chunk_ranks.size()) +
      uint64_t(source_chunk) * uint64_t(chunk_ranks.size()) + uint64_t(destination_chunk);
  message.direction = direction;
  message.local_schedule_ordinal = local_ordinal;
  message.total_elements = op.transfer_size;
  message.element_bytes = 0;
  message.wire_bytes = 0;
  message.wire_digest = 0;

  size_t expected_offset = 0;
  bool have_precision = false;
  Precision precision = Precision::f64;
  for (connect_phase phase : all_connect_phases) {
    const HaloPlan *plan = halos.find(comms_key{ft, phase, chunk_pair{source_chunk,
                                                                     destination_chunk}});
    if (!plan || !plan->block_elements) continue;
    if (plan->ft != ft || plan->chunks != chunk_pair{source_chunk, destination_chunk} ||
        plan->phase != phase || plan->sequence_index != uint32_t(phase)) {
      why = "remote halo phase identity is inconsistent";
      return false;
    }
    if (plan->storage != HaloStorageDisposition::canonical) {
      why = "remote halo complete block contains host-owned storage";
      return false;
    }
    if (plan->peer_rank != op.other_proc_id || plan->same_rank) {
      why = "remote halo phase peer disposition is inconsistent";
      return false;
    }
    if (plan->block_offset != expected_offset) {
      why = "remote halo phase spans do not form an exact ordered cover";
      return false;
    }
    if (plan->block_elements > std::numeric_limits<size_t>::max() - expected_offset) {
      why = "remote halo phase element coverage overflows";
      return false;
    }
    message.phases.push_back(RemoteHaloPhaseSpan{phase, plan->sequence_index,
                                                 plan->block_offset, plan->block_elements});
    expected_offset += plan->block_elements;
    if (!collect_precision(*plan, direction, storage, have_precision, precision, why)) return false;
  }
  if (expected_offset != op.transfer_size || (!expected_offset && op.transfer_size)) {
    why = "remote halo phase coverage does not equal communication transfer size";
    return false;
  }
  if (!have_precision && op.transfer_size) {
    why = "remote halo nonempty communication block has no canonical endpoint storage";
    return false;
  }
  message.storage_precision = precision;
  message.element_bytes = precision == Precision::f32 ? sizeof(float) : sizeof(double);
  if (!checked_multiply(message.total_elements, message.element_bytes, message.wire_bytes)) {
    why = "remote halo byte count overflows size_t";
    return false;
  }
  if (message.wire_bytes > size_t(INT_MAX)) {
    why = "remote halo byte count exceeds MPI int count";
    return false;
  }
  message.wire_digest = compute_remote_halo_wire_digest(message);
  return true;
}

} // namespace

GpuMpiPolicyParse parse_gpu_mpi_policy(const char *value) {
  GpuMpiPolicyParse result{true, GpuMpiPolicy::automatic, std::string()};
  if (!value || !*value || std::string(value) == "auto") return result;
  if (std::string(value) == "no") result.requested = GpuMpiPolicy::staged;
  else if (std::string(value) == "yes") result.requested = GpuMpiPolicy::direct;
  else {
    result.valid = false;
    result.error = "MEEP_GPU_AWARE_MPI must be no, yes, or auto";
  }
  return result;
}

bool resolve_gpu_mpi_route(GpuMpiPolicy policy, bool provider_query_available,
                           bool provider_supports_direct, GpuMpiRoute &route,
                           std::string &why) {
  why.clear();
  if (!valid_policy(policy)) {
    why = "GPU MPI policy enum is invalid";
    return false;
  }
  if (policy == GpuMpiPolicy::staged) {
    route = GpuMpiRoute::staged;
    return true;
  }
  if (policy == GpuMpiPolicy::direct) {
    if (!provider_query_available || !provider_supports_direct) {
      why = "forced direct GPU MPI requires a positive provider query";
      return false;
    }
    route = GpuMpiRoute::direct;
    return true;
  }
  route = provider_query_available && provider_supports_direct ? GpuMpiRoute::direct
                                                               : GpuMpiRoute::staged;
  return true;
}

const char *gpu_mpi_policy_name(GpuMpiPolicy policy) {
  switch (policy) {
    case GpuMpiPolicy::staged: return "staged";
    case GpuMpiPolicy::automatic: return "auto";
    case GpuMpiPolicy::direct: return "direct";
  }
  return "invalid";
}

const char *gpu_mpi_route_name(GpuMpiRoute route) {
  return route == GpuMpiRoute::direct ? "direct" : "staged";
}

bool operator==(const RemoteHaloWireKey &a, const RemoteHaloWireKey &b) {
  return a.source_rank == b.source_rank && a.destination_rank == b.destination_rank &&
         a.ft == b.ft && a.source_chunk == b.source_chunk &&
         a.destination_chunk == b.destination_chunk && a.tag == b.tag &&
         a.canonical_ordinal == b.canonical_ordinal;
}

bool remote_halo_wire_key_less(const RemoteHaloWireKey &a, const RemoteHaloWireKey &b) {
  return std::tie(a.source_rank, a.destination_rank, a.ft, a.source_chunk,
                  a.destination_chunk, a.tag, a.canonical_ordinal) <
         std::tie(b.source_rank, b.destination_rank, b.ft, b.source_chunk,
                  b.destination_chunk, b.tag, b.canonical_ordinal);
}

bool operator==(const RemoteHaloPhaseSpan &a, const RemoteHaloPhaseSpan &b) {
  return a.phase == b.phase && a.sequence_index == b.sequence_index &&
         a.block_offset == b.block_offset && a.block_elements == b.block_elements;
}

uint64_t compute_remote_halo_wire_digest(const RemoteHaloMessage &message) {
  uint64_t h = UINT64_C(1469598103934665603);
  h = hash_key(h, message.key);
  h = hash_u64(h, message.total_elements);
  h = hash_u64(h, uint64_t(message.storage_precision));
  h = hash_u64(h, message.element_bytes);
  h = hash_u64(h, message.wire_bytes);
  h = hash_u64(h, message.phases.size());
  for (const RemoteHaloPhaseSpan &phase : message.phases) {
    h = hash_u64(h, uint64_t(uint32_t(phase.phase)));
    h = hash_u64(h, phase.sequence_index);
    h = hash_u64(h, phase.block_offset);
    h = hash_u64(h, phase.block_elements);
  }
  return h;
}

uint64_t compute_remote_halo_stage_signature(const RemoteHaloStage &stage) {
  return stage_signature(stage);
}

uint64_t compute_remote_halo_program_signature(const RemoteHaloProgram &program) {
  uint64_t h = UINT64_C(1469598103934665603);
  h = hash_u64(h, program.version);
  h = hash_u64(h, uint64_t(uint32_t(program.communicator_rank)));
  h = hash_u64(h, uint64_t(uint32_t(program.communicator_size)));
  h = hash_u64(h, program.communicator_generation);
  h = hash_u64(h, uint64_t(program.requested_policy));
  h = hash_u64(h, uint64_t(program.resolved_route));
  h = hash_u64(h, program.participation.device_owner ? 1 : 0);
  h = hash_u64(h, program.participation.owned_field_chunks);
  h = hash_u64(h, program.participation.transport_messages);
  h = hash_u64(h, program.participation.cuda_required_operations);
  h = hash_u64(h, program.stages.size());
  for (const RemoteHaloStage &stage : program.stages) {
    h = hash_u64(h, compute_remote_halo_stage_signature(stage));
  }
  return h;
}

bool validate_remote_halo_participation(const RemoteHaloParticipation &participation,
                                        std::string &why) {
  why.clear();
  if (!participation.device_owner &&
      (participation.owned_field_chunks || participation.transport_messages ||
       participation.cuda_required_operations)) {
    why = "idle/no-device rank has owned chunks, transport messages, or CUDA-required work";
    return false;
  }
  return true;
}

bool validate_remote_halo_program(const RemoteHaloProgram &program, int mpi_tag_ub,
                                  std::string &why) {
  why.clear();
  if (program.version != RemoteHaloProgram::schema_version) {
    why = "remote halo program schema version is unsupported";
    return false;
  }
  if (program.communicator_size < 1 || program.communicator_rank < 0 ||
      program.communicator_rank >= program.communicator_size ||
      !program.communicator_generation) {
    why = "remote halo program has an invalid communicator rank or size";
    return false;
  }
  if (!valid_policy(program.requested_policy) || !valid_route(program.resolved_route)) {
    why = "remote halo program has an invalid policy or route enum";
    return false;
  }
  if ((program.requested_policy == GpuMpiPolicy::staged &&
       program.resolved_route != GpuMpiRoute::staged) ||
      (program.requested_policy == GpuMpiPolicy::direct &&
       program.resolved_route != GpuMpiRoute::direct)) {
    why = "remote halo resolved route contradicts the requested policy";
    return false;
  }
  if (!validate_remote_halo_participation(program.participation, why)) return false;
  size_t transport_messages = 0;
  for (const RemoteHaloStage &stage : program.stages) {
    if (stage.signature != compute_remote_halo_stage_signature(stage)) {
      why = "remote halo stage signature is stale";
      return false;
    }
    if (stage.receives.size() > std::numeric_limits<size_t>::max() - transport_messages ||
        stage.sends.size() > std::numeric_limits<size_t>::max() -
                                 transport_messages - stage.receives.size()) {
      why = "remote halo transport message count overflows";
      return false;
    }
    transport_messages += stage.receives.size() + stage.sends.size();
    std::set<std::tuple<int, int, int> > live;
    const std::vector<const std::vector<RemoteHaloMessage> *> sides{&stage.receives, &stage.sends};
    for (const std::vector<RemoteHaloMessage> *messages : sides)
      for (const RemoteHaloMessage &message : *messages) {
        if (int(stage.ft) < 0 || int(stage.ft) >= NUM_FIELD_TYPES ||
            int(message.key.ft) < 0 || int(message.key.ft) >= NUM_FIELD_TYPES ||
            message.key.ft != stage.ft || message.key.tag < 0 ||
            message.key.tag > mpi_tag_ub) {
          why = "remote halo message has an invalid field type or MPI tag";
          return false;
        }
        if (message.key.source_rank < 0 ||
            message.key.source_rank >= program.communicator_size ||
            message.key.destination_rank < 0 ||
            message.key.destination_rank >= program.communicator_size ||
            message.key.source_rank == message.key.destination_rank ||
            message.key.source_chunk < 0 || message.key.destination_chunk < 0) {
          why = "remote halo message has invalid endpoint identity";
          return false;
        }
        if (message.direction == RemoteHaloDirection::incoming) {
          if (message.key.destination_rank != program.communicator_rank) {
            why = "incoming remote halo message has the wrong local endpoint";
            return false;
          }
        }
        else if (message.direction == RemoteHaloDirection::outgoing) {
          if (message.key.source_rank != program.communicator_rank) {
            why = "outgoing remote halo message has the wrong local endpoint";
            return false;
          }
        }
        else {
          why = "remote halo message has an invalid direction";
          return false;
        }
        if ((message.storage_precision != Precision::f32 &&
             message.storage_precision != Precision::f64) ||
            message.element_bytes !=
                (message.storage_precision == Precision::f32 ? sizeof(float) : sizeof(double))) {
          why = "remote halo message has an invalid storage precision";
          return false;
        }
        size_t expected_wire_bytes = 0;
        if (!checked_multiply(message.total_elements, message.element_bytes,
                              expected_wire_bytes) ||
            expected_wire_bytes != message.wire_bytes) {
          why = "remote halo message byte extent is inconsistent";
          return false;
        }
        size_t expected_offset = 0;
        uint32_t previous_sequence = 0;
        bool have_phase = false;
        for (const RemoteHaloPhaseSpan &phase : message.phases) {
          if (int(phase.phase) < int(CONNECT_PHASE) || int(phase.phase) > int(CONNECT_COPY) ||
              phase.sequence_index != uint32_t(phase.phase) || !phase.block_elements ||
              phase.block_offset != expected_offset ||
              (have_phase && phase.sequence_index <= previous_sequence) ||
              phase.block_elements > std::numeric_limits<size_t>::max() - expected_offset) {
            why = "remote halo message phase layout is invalid";
            return false;
          }
          expected_offset += phase.block_elements;
          previous_sequence = phase.sequence_index;
          have_phase = true;
        }
        if (expected_offset != message.total_elements ||
            (message.total_elements != 0 && !have_phase)) {
          why = "remote halo message phase coverage is incomplete";
          return false;
        }
        if (message.wire_bytes > size_t(INT_MAX) ||
            message.wire_digest != compute_remote_halo_wire_digest(message)) {
          why = "remote halo message byte count or digest is invalid";
          return false;
        }
        const std::tuple<int, int, int> liveness(
            message.direction == RemoteHaloDirection::incoming ? message.key.source_rank
                                                               : message.key.destination_rank,
            message.key.tag, int(message.direction));
        if (!live.insert(liveness).second) {
          why = "remote halo stage contains duplicate simultaneously live peer/tag requests";
          return false;
        }
      }
  }
  if (transport_messages != program.participation.transport_messages) {
    why = "remote halo participation message count is stale";
    return false;
  }
  if (program.signature != compute_remote_halo_program_signature(program)) {
    why = "remote halo program signature is stale";
    return false;
  }
  return true;
}

void remote_halo_agreement_records(const RemoteHaloProgram &program,
                                   std::vector<RemoteHaloAgreementRecord> &records) {
  records.clear();
  for (const RemoteHaloStage &stage : program.stages) {
    for (const RemoteHaloMessage &message : stage.receives)
      records.push_back(RemoteHaloAgreementRecord{message.key, message.direction,
                                                  message.wire_digest, message.wire_bytes});
    for (const RemoteHaloMessage &message : stage.sends)
      records.push_back(RemoteHaloAgreementRecord{message.key, message.direction,
                                                  message.wire_digest, message.wire_bytes});
  }
}

bool validate_remote_halo_agreement(const std::vector<RemoteHaloAgreementRecord> &records,
                                    std::string &why) {
  why.clear();
  std::vector<RemoteHaloAgreementRecord> ordered = records;
  std::sort(ordered.begin(), ordered.end(), [](const RemoteHaloAgreementRecord &a,
                                                const RemoteHaloAgreementRecord &b) {
    if (a.key != b.key) return remote_halo_wire_key_less(a.key, b.key);
    return int(a.direction) < int(b.direction);
  });
  for (size_t i = 0; i < ordered.size();) {
    size_t j = i + 1;
    while (j < ordered.size() && ordered[j].key == ordered[i].key) ++j;
    if (j - i != 2 || ordered[i].direction != RemoteHaloDirection::incoming ||
        ordered[i + 1].direction != RemoteHaloDirection::outgoing ||
        ordered[i].wire_digest != ordered[i + 1].wire_digest ||
        ordered[i].wire_bytes != ordered[i + 1].wire_bytes) {
      why = "remote halo sender/receiver wire agreement is incomplete or mismatched";
      return false;
    }
    i = j;
  }
  return true;
}

bool build_remote_halo_program(const comms_sequence sequences[NUM_FIELD_TYPES],
                               const halo_plan_set &halos, const StoragePlan &storage,
                               const std::vector<int> &chunk_ranks, int communicator_rank,
                               int communicator_size, uint64_t communicator_generation,
                               GpuMpiPolicy requested_policy, GpuMpiRoute resolved_route,
                               bool device_owner, size_t cuda_required_operations,
                               int mpi_tag_ub, RemoteHaloProgram &program, std::string &why) {
  why.clear();
  RemoteHaloProgram staged;
  staged.version = RemoteHaloProgram::schema_version;
  staged.communicator_rank = communicator_rank;
  staged.communicator_size = communicator_size;
  staged.communicator_generation = communicator_generation;
  staged.requested_policy = requested_policy;
  staged.resolved_route = resolved_route;
  staged.participation = RemoteHaloParticipation{device_owner, 0, 0,
                                                 cuda_required_operations};
  for (int rank : chunk_ranks) {
    if (rank < 0 || rank >= communicator_size) {
      why = "remote halo chunk ownership rank is outside the communicator";
      return false;
    }
    if (rank == communicator_rank) ++staged.participation.owned_field_chunks;
  }

  FOR_FIELD_TYPES(ft) {
    RemoteHaloStage stage;
    stage.ft = ft;
    const comms_sequence &sequence = sequences[ft];
    for (size_t i = 0; i < sequence.receive_ops.size(); ++i) {
      if (i > std::numeric_limits<uint32_t>::max()) {
        why = "remote halo receive schedule ordinal overflows";
        return false;
      }
      const comms_operation &op = sequence.receive_ops[i];
      if (op.other_proc_id == communicator_rank) continue;
      RemoteHaloMessage message;
      if (!build_message(op, uint32_t(i), ft, RemoteHaloDirection::incoming, halos, storage,
                         chunk_ranks, communicator_rank, message, why))
        return false;
      stage.receives.push_back(message);
    }
    for (size_t i = 0; i < sequence.send_ops.size(); ++i) {
      if (i > std::numeric_limits<uint32_t>::max()) {
        why = "remote halo send schedule ordinal overflows";
        return false;
      }
      const comms_operation &op = sequence.send_ops[i];
      if (op.other_proc_id == communicator_rank) continue;
      RemoteHaloMessage message;
      if (!build_message(op, uint32_t(i), ft, RemoteHaloDirection::outgoing, halos, storage,
                         chunk_ranks, communicator_rank, message, why))
        return false;
      stage.sends.push_back(message);
    }
    stage.signature = compute_remote_halo_stage_signature(stage);
    if (!stage.receives.empty() || !stage.sends.empty()) staged.stages.push_back(stage);
  }
  for (const RemoteHaloStage &stage : staged.stages)
    staged.participation.transport_messages += stage.receives.size() + stage.sends.size();
  staged.signature = compute_remote_halo_program_signature(staged);
  if (!validate_remote_halo_program(staged, mpi_tag_ub, why)) return false;
  program = staged;
  return true;
}

} // namespace meep
