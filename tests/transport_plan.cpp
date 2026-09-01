/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include <climits>
#include <cstdio>
#include <string>
#include <vector>

#include <meep.hpp>

#include "backend/halo_plan.hpp"
#include "backend/storage_plan.hpp"
#include "backend/transport_plan.hpp"

using namespace meep;

namespace {
int failures = 0;

#define CHECK(condition, message)                                                                 \
  do {                                                                                            \
    if (!(condition)) {                                                                           \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, message);                    \
      ++failures;                                                                                 \
    }                                                                                             \
  } while (0)

ArraySpec array_spec(uint32_t id, Precision precision, size_t elements) {
  ArraySpec result;
  result.id = ArrayId{id};
  result.role = array_role::field;
  result.element_type = ElementType::realnum_value;
  result.storage = precision;
  result.elements = elements;
  result.alignment = precision == Precision::f32 ? alignof(float) : alignof(double);
  result.alias_of = invalid_array();
  result.classification_provisional = false;
  result.classification_elided = false;
  return result;
}

HaloPlan &phase(halo_plan_set &halos, field_type ft, connect_phase p, size_t offset,
                size_t elements, int peer, bool outgoing, ArrayId array) {
  HaloPlan &result = halos.get_or_create(comms_key{ft, p, chunk_pair{0, 1}});
  result.peer_rank = peer;
  result.same_rank = false;
  result.storage = HaloStorageDisposition::canonical;
  result.block_offset = offset;
  result.block_elements = elements;
  std::vector<ElementRef> &refs = outgoing ? result.gather : result.scatter;
  for (size_t i = 0; i < elements; ++i) refs.push_back(ElementRef{array, ptrdiff_t(offset + i)});
  if (elements)
    (outgoing ? result.gather_order : result.scatter_order)
        .push_back(HaloSegment{0, 0, 0, uint32_t(elements)});
  return result;
}

void make_endpoint(bool outgoing, uint32_t local_ordinal, RemoteHaloProgram &program,
                   halo_plan_set &halos, StoragePlan &storage) {
  storage.arrays.push_back(array_spec(0, Precision::f32, 8));
  storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Ex), 0, 0});
  phase(halos, E_stuff, CONNECT_PHASE, 0, 2, outgoing ? 1 : 0, outgoing, ArrayId{0});
  phase(halos, E_stuff, CONNECT_COPY, 2, 1, outgoing ? 1 : 0, outgoing, ArrayId{0});

  comms_sequence sequences[NUM_FIELD_TYPES];
  comms_operation op;
  op.my_chunk_idx = outgoing ? 0 : 1;
  op.other_chunk_idx = outgoing ? 1 : 0;
  op.other_proc_id = outgoing ? 1 : 0;
  op.pair_idx = 2;
  op.transfer_size = 3;
  op.comm_direction = outgoing ? Outgoing : Incoming;
  op.tag = 7;
  std::vector<comms_operation> &ops = outgoing ? sequences[E_stuff].send_ops
                                               : sequences[E_stuff].receive_ops;
  ops.resize(local_ordinal + 1, op);
  for (uint32_t i = 0; i < local_ordinal; ++i) ops[i].other_proc_id = outgoing ? 0 : 1;

  std::string why;
  CHECK(build_remote_halo_program(sequences, halos, storage, std::vector<int>{0, 1},
                                  outgoing ? 0 : 1, 2, 11, GpuMpiPolicy::staged,
                                  GpuMpiRoute::staged, true, 1, 100, program, why),
        why.c_str());
}

void test_policy() {
  CHECK(parse_gpu_mpi_policy(NULL).valid, "unset policy should be auto");
  CHECK(parse_gpu_mpi_policy("no").requested == GpuMpiPolicy::staged,
        "no policy should request staged");
  CHECK(parse_gpu_mpi_policy("yes").requested == GpuMpiPolicy::direct,
        "yes policy should request direct");
  CHECK(!parse_gpu_mpi_policy("maybe").valid, "invalid policy should reject");
  GpuMpiRoute route = GpuMpiRoute::direct;
  std::string why;
  CHECK(resolve_gpu_mpi_route(GpuMpiPolicy::automatic, false, false, route, why) &&
            route == GpuMpiRoute::staged,
        "auto without provider query should stage");
  CHECK(resolve_gpu_mpi_route(GpuMpiPolicy::automatic, true, true, route, why) &&
            route == GpuMpiRoute::direct,
        "auto with positive provider query should be direct");
  CHECK(!resolve_gpu_mpi_route(GpuMpiPolicy::direct, true, false, route, why),
        "forced direct with negative query should reject");
  CHECK(!resolve_gpu_mpi_route(static_cast<GpuMpiPolicy>(99), true, true, route, why),
        "invalid policy enum was accepted");
}

void test_endpoint_invariant_identity() {
  RemoteHaloProgram sender, receiver;
  halo_plan_set send_halos, receive_halos;
  StoragePlan send_storage, receive_storage;
  make_endpoint(true, 3, sender, send_halos, send_storage);
  make_endpoint(false, 0, receiver, receive_halos, receive_storage);
  CHECK(sender.stages.size() == 1 && sender.stages[0].sends.size() == 1,
        "sender message was not lowered");
  CHECK(receiver.stages.size() == 1 && receiver.stages[0].receives.size() == 1,
        "receiver message was not lowered");
  if (sender.stages.empty() || receiver.stages.empty()) return;
  const RemoteHaloMessage &send = sender.stages[0].sends[0];
  const RemoteHaloMessage &receive = receiver.stages[0].receives[0];
  CHECK(send.local_schedule_ordinal == 3 && receive.local_schedule_ordinal == 0,
        "fixture did not create distinct local schedule positions");
  CHECK(send.key == receive.key, "wire key depends on endpoint-local schedule order");
  CHECK(send.key.canonical_ordinal ==
            uint64_t(uint32_t(E_stuff)) * 4 + 1,
        "canonical wire ordinal does not follow the global source/destination loop");
  CHECK(send.wire_digest == receive.wire_digest,
        "wire digest depends on endpoint-local direction or schedule order");
  CHECK(send.wire_bytes == 3 * sizeof(float), "f32 wire byte count is incorrect");

  std::vector<RemoteHaloAgreementRecord> records;
  remote_halo_agreement_records(sender, records);
  std::vector<RemoteHaloAgreementRecord> other;
  remote_halo_agreement_records(receiver, other);
  records.insert(records.end(), other.begin(), other.end());
  std::string why;
  CHECK(validate_remote_halo_agreement(records, why), why.c_str());
  records.back().wire_digest ^= 1;
  CHECK(!validate_remote_halo_agreement(records, why),
        "wire agreement accepted a digest mismatch");
}

void test_complete_block_rejections() {
  RemoteHaloProgram program;
  halo_plan_set halos;
  StoragePlan storage;
  storage.arrays.push_back(array_spec(0, Precision::f64, 8));
  storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Ex), 0, 0});
  HaloPlan &p = phase(halos, E_stuff, CONNECT_COPY, 0, 1, 1, true, ArrayId{0});
  comms_sequence sequences[NUM_FIELD_TYPES];
  sequences[E_stuff].send_ops.push_back(comms_operation{0, 1, 1, 2, 1, Outgoing, 9});
  std::string why;
  p.storage = HaloStorageDisposition::host_owned;
  CHECK(!build_remote_halo_program(sequences, halos, storage, std::vector<int>{0, 1}, 0, 2,
                                   1, GpuMpiPolicy::staged, GpuMpiRoute::staged, true, 1, 100,
                                   program, why),
        "host-owned phase did not reject the complete block");
  p.storage = HaloStorageDisposition::canonical;
  p.block_offset = 1;
  CHECK(!build_remote_halo_program(sequences, halos, storage, std::vector<int>{0, 1}, 0, 2,
                                   1, GpuMpiPolicy::staged, GpuMpiRoute::staged, true, 1, 100,
                                   program, why),
        "gapped phase coverage was accepted");
  p.block_offset = 0;
  sequences[E_stuff].send_ops[0].tag = 101;
  CHECK(!build_remote_halo_program(sequences, halos, storage, std::vector<int>{0, 1}, 0, 2,
                                   1, GpuMpiPolicy::staged, GpuMpiRoute::staged, true, 1, 100,
                                   program, why),
        "tag above MPI_TAG_UB was accepted");
  sequences[E_stuff].send_ops[0].tag = 9;
  storage.arrays[0].classification_provisional = true;
  CHECK(!build_remote_halo_program(sequences, halos, storage, std::vector<int>{0, 1}, 0, 2,
                                   1, GpuMpiPolicy::staged, GpuMpiRoute::staged, true, 1, 100,
                                   program, why),
        "provisional canonical storage was accepted");

  halo_plan_set mixed_halos;
  StoragePlan mixed_storage;
  mixed_storage.arrays.push_back(array_spec(0, Precision::f64, 4));
  mixed_storage.arrays.push_back(array_spec(1, Precision::f32, 4));
  mixed_storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Ex), 0, 0});
  mixed_storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Ey), 0, 0});
  phase(mixed_halos, E_stuff, CONNECT_NEGATE, 0, 1, 1, true, ArrayId{0});
  phase(mixed_halos, E_stuff, CONNECT_COPY, 1, 1, 1, true, ArrayId{1});
  comms_sequence mixed_sequences[NUM_FIELD_TYPES];
  mixed_sequences[E_stuff].send_ops.push_back(comms_operation{0, 1, 1, 2, 2, Outgoing, 10});
  CHECK(!build_remote_halo_program(mixed_sequences, mixed_halos, mixed_storage,
                                   std::vector<int>{0, 1}, 0, 2, 1,
                                   GpuMpiPolicy::staged, GpuMpiRoute::staged, true, 1, 100,
                                   program, why),
        "mixed-precision complete communication block was accepted");
}

void test_manual_validation() {
  RemoteHaloProgram program;
  program.version = RemoteHaloProgram::schema_version;
  program.communicator_rank = 0;
  program.communicator_size = 2;
  program.communicator_generation = 1;
  program.requested_policy = GpuMpiPolicy::staged;
  program.resolved_route = GpuMpiRoute::staged;
  program.participation = RemoteHaloParticipation{true, 1, 1, 1};
  RemoteHaloStage stage;
  stage.ft = E_stuff;
  RemoteHaloMessage message;
  message.key = RemoteHaloWireKey{0, 1, E_stuff, 0, 1, 3, 2};
  message.direction = RemoteHaloDirection::outgoing;
  message.local_schedule_ordinal = 0;
  message.total_elements = size_t(INT_MAX) + 1;
  message.storage_precision = Precision::f32;
  message.element_bytes = 1;
  message.wire_bytes = size_t(INT_MAX) + 1;
  message.wire_digest = compute_remote_halo_wire_digest(message);
  stage.sends.push_back(message);
  stage.signature = compute_remote_halo_stage_signature(stage);
  program.stages.push_back(stage);
  program.signature = compute_remote_halo_program_signature(program);
  std::string why;
  CHECK(!validate_remote_halo_program(program, INT_MAX, why),
        "wire bytes above INT_MAX were accepted");

  program.stages[0].sends[0].wire_bytes = 4;
  program.stages[0].sends[0].wire_digest =
      compute_remote_halo_wire_digest(program.stages[0].sends[0]);
  program.stages[0].sends.push_back(program.stages[0].sends[0]);
  program.stages[0].signature = compute_remote_halo_stage_signature(program.stages[0]);
  program.signature = compute_remote_halo_program_signature(program);
  CHECK(!validate_remote_halo_program(program, INT_MAX, why),
        "duplicate live peer/tag request was accepted");

  program.stages[0].sends.pop_back();
  program.participation.transport_messages = 1;
  program.stages[0].signature = compute_remote_halo_stage_signature(program.stages[0]) ^ 1;
  program.signature = compute_remote_halo_program_signature(program);
  CHECK(!validate_remote_halo_program(program, INT_MAX, why),
        "stale remote halo stage signature was accepted");
}

void test_idle_admission() {
  std::string why;
  CHECK(validate_remote_halo_participation(RemoteHaloParticipation{false, 0, 0, 0}, why),
        why.c_str());
  CHECK(!validate_remote_halo_participation(RemoteHaloParticipation{false, 1, 0, 0}, why),
        "idle rank with an owned field chunk was accepted");
  CHECK(!validate_remote_halo_participation(RemoteHaloParticipation{false, 0, 1, 0}, why),
        "idle rank with a transport message was accepted");
  CHECK(!validate_remote_halo_participation(RemoteHaloParticipation{false, 0, 0, 1}, why),
        "idle rank with a CUDA-required operation was accepted");

  comms_sequence sequences[NUM_FIELD_TYPES];
  halo_plan_set halos;
  StoragePlan storage;
  RemoteHaloProgram idle;
  CHECK(build_remote_halo_program(sequences, halos, storage, std::vector<int>{0, 0}, 1, 2, 5,
                                  GpuMpiPolicy::staged, GpuMpiRoute::staged, false, 0, 100,
                                  idle, why),
        why.c_str());
  CHECK(!idle.participation.device_owner && !idle.participation.owned_field_chunks &&
            !idle.participation.transport_messages &&
            !idle.participation.cuda_required_operations,
        "idle program did not preserve the zero-work proof");
}

void test_zero_length_wire_message() {
  RemoteHaloProgram sender, receiver;
  halo_plan_set send_halos, receive_halos;
  StoragePlan send_storage, receive_storage;
  comms_sequence send_sequences[NUM_FIELD_TYPES];
  comms_sequence receive_sequences[NUM_FIELD_TYPES];
  send_sequences[E_stuff].send_ops.push_back(comms_operation{0, 1, 1, 2, 0, Outgoing, 4});
  receive_sequences[E_stuff].receive_ops.push_back(
      comms_operation{1, 0, 0, 2, 0, Incoming, 4});
  std::string why;
  CHECK(build_remote_halo_program(send_sequences, send_halos, send_storage,
                                  std::vector<int>{0, 1}, 0, 2, 3,
                                  GpuMpiPolicy::staged, GpuMpiRoute::staged, true, 1, 100,
                                  sender, why),
        why.c_str());
  CHECK(build_remote_halo_program(receive_sequences, receive_halos, receive_storage,
                                  std::vector<int>{0, 1}, 1, 2, 3,
                                  GpuMpiPolicy::staged, GpuMpiRoute::staged, true, 1, 100,
                                  receiver, why),
        why.c_str());
  std::vector<RemoteHaloAgreementRecord> records;
  remote_halo_agreement_records(sender, records);
  std::vector<RemoteHaloAgreementRecord> receive_records;
  remote_halo_agreement_records(receiver, receive_records);
  records.insert(records.end(), receive_records.begin(), receive_records.end());
  CHECK(validate_remote_halo_agreement(records, why), why.c_str());
}

} // namespace

int main() {
  test_policy();
  test_endpoint_invariant_identity();
  test_complete_block_rejections();
  test_manual_validation();
  test_idle_admission();
  test_zero_length_wire_message();
  if (failures) {
    std::fprintf(stderr, "transport_plan: %d failures\n", failures);
    return 1;
  }
  std::printf("transport_plan: PASS\n");
  return 0;
}
