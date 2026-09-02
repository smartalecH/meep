/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include <climits>
#include <algorithm>
#include <cstdio>
#include <limits>
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

void test_neutral_remote_lowering() {
  RemoteHaloProgram sender, receiver;
  halo_plan_set send_halos, receive_halos;
  StoragePlan send_storage, receive_storage;
  make_endpoint(true, 0, sender, send_halos, send_storage);
  make_endpoint(false, 0, receiver, receive_halos, receive_storage);
  receive_halos.find(comms_key{E_stuff, CONNECT_PHASE, chunk_pair{0, 1}})
      ->phase_values.push_back(std::complex<realnum>(realnum(0.6), realnum(0.8)));

  LoweredRemoteHaloProgram lowered_send, lowered_receive;
  std::string why;
  CHECK(lower_remote_halo_program(sender, send_halos, send_storage, false, 100,
                                  lowered_send, why),
        why.c_str());
  CHECK(lower_remote_halo_program(receiver, receive_halos, receive_storage, false, 100,
                                  lowered_receive, why),
        why.c_str());
  CHECK(lowered_send.stages.empty() || lowered_receive.stages.empty() ||
            (lowered_send.stages[0].sends[0].slot_offsets[0] ==
                 lowered_receive.stages[0].receives[0].slot_offsets[0] &&
             lowered_send.stages[0].sends[0].slot_offsets[1] ==
                 lowered_receive.stages[0].receives[0].slot_offsets[1]),
        "sender and receiver reconstructed different wire-local slot offsets");
  CHECK(lowered_send.version == LoweredRemoteHaloProgram::schema_version &&
            lowered_send.program_signature == sender.signature &&
            lowered_send.storage_signature ==
                compute_remote_storage_authority_signature(send_storage) &&
            lowered_send.authority_signature == compute_remote_lowered_authority_signature(
                                                     sender.signature,
                                                     lowered_send.storage_signature) &&
            lowered_send.stages.size() == 1,
        "lowered sender did not retain its authority identity");
  if (!lowered_send.stages.empty()) {
    const LoweredRemoteHaloStage &stage = lowered_send.stages[0];
    CHECK(stage.send_slot_bytes == 3 * sizeof(float) && stage.sends.size() == 1,
          "lowered sender has an incorrect byte layout");
    if (!stage.sends.empty()) {
      const LoweredRemoteHaloMessage &message = stage.sends[0];
      CHECK(message.gathers.size() == 3 && message.slot_offsets[0] == 0 &&
                message.slot_offsets[1] == 3 * sizeof(float) &&
                message.arena_offsets[0] == 0 &&
                message.arena_offsets[1] == 3 * sizeof(float),
            "lowered sender did not build two exact arena slots");
      CHECK(message.gathers.size() < 3 ||
                (message.gathers[0].buffer_byte_offset == 0 &&
                 message.gathers[1].buffer_byte_offset == sizeof(float) &&
                 message.gathers[2].buffer_byte_offset == 2 * sizeof(float)),
            "lowered gather order is not byte-exact");
    }
  }
  if (!lowered_receive.stages.empty()) {
    const LoweredRemoteHaloStage &stage = lowered_receive.stages[0];
    CHECK(stage.receive_slot_bytes == 3 * sizeof(float) && stage.receives.size() == 1 &&
              stage.publication == RemoteHaloPublicationMode::parallel_unique,
          "unique receive lowering did not select parallel publication");
    if (!stage.receives.empty()) {
      const LoweredRemoteHaloMessage &message = stage.receives[0];
      CHECK(message.scatters.size() == 2,
            "PHASE pair plus COPY did not compile to two scatter descriptors");
      CHECK(message.scatters.size() < 2 ||
                (message.scatters[0].phase_real == 0.6 &&
                 message.scatters[0].phase_imag == 0.8 &&
                 message.scatters[1].phase_real == 1.0 &&
                 !is_valid(message.scatters[1].target_imag.root)),
            "lowered transforms differ from canonical phase order");
    }
  }

  RemoteHaloProgram overlapping = receiver;
  overlapping.communicator_size = 3;
  RemoteHaloMessage second = overlapping.stages[0].receives[0];
  second.key.source_rank = 2;
  second.key.source_chunk = 2;
  second.key.tag = 8;
  second.key.canonical_ordinal += 8;
  second.wire_digest = compute_remote_halo_wire_digest(second);
  overlapping.stages[0].receives.insert(overlapping.stages[0].receives.begin(), second);
  overlapping.participation.transport_messages = 2;
  overlapping.stages[0].signature =
      compute_remote_halo_stage_signature(overlapping.stages[0]);
  overlapping.signature = compute_remote_halo_program_signature(overlapping);
  for (connect_phase p : all_connect_phases) {
    const HaloPlan *original =
        receive_halos.find(comms_key{E_stuff, p, chunk_pair{0, 1}});
    if (!original) continue;
    const HaloPlan copy = *original;
    HaloPlan &other = receive_halos.get_or_create(comms_key{E_stuff, p, chunk_pair{2, 1}});
    other = copy;
    other.chunks = chunk_pair{2, 1};
    other.peer_rank = 2;
  }
  CHECK(lower_remote_halo_program(overlapping, receive_halos, receive_storage, false, 100,
                                  lowered_receive, why),
        why.c_str());
  CHECK(lowered_receive.stages.empty() ||
            (lowered_receive.stages[0].publication ==
                 RemoteHaloPublicationMode::canonical_serial &&
             lowered_receive.stages[0].canonical_receive_order.size() == 2 &&
             lowered_receive.stages[0].canonical_receive_order[0] == 1 &&
             lowered_receive.stages[0].canonical_receive_order[1] == 0),
        "cross-message overlap did not select endpoint-invariant publication order");

  HaloPlan *copy =
      receive_halos.find(comms_key{E_stuff, CONNECT_COPY, chunk_pair{0, 1}});
  CHECK(copy && copy->scatter.size() == 1, "receive fixture lacks COPY target");
  if (copy && copy->scatter.size() == 1) copy->scatter[0].index = 0;
  CHECK(lower_remote_halo_program(receiver, receive_halos, receive_storage, false, 100,
                                  lowered_receive, why),
        why.c_str());
  CHECK(lowered_receive.stages.empty() ||
            lowered_receive.stages[0].publication ==
                RemoteHaloPublicationMode::canonical_serial,
        "duplicate physical destination did not select canonical serial publication");

  if (copy && copy->scatter.size() == 1) copy->scatter[0].index = 2;
  receive_storage.arrays.push_back(array_spec(1, Precision::f32, 8));
  receive_storage.arrays[1].alias_of = ArrayId{0};
  receive_storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Ey), 0, 0});
  if (copy && copy->scatter.size() == 1) {
    copy->scatter[0].array = ArrayId{1};
    copy->scatter[0].index = 0;
  }
  CHECK(lower_remote_halo_program(receiver, receive_halos, receive_storage, false, 100,
                                  lowered_receive, why),
        why.c_str());
  CHECK(lowered_receive.stages.empty() ||
            lowered_receive.stages[0].publication ==
                RemoteHaloPublicationMode::canonical_serial,
        "alias-root destination overlap was not detected");

  receive_storage.arrays[0].alias_of = ArrayId{1};
  CHECK(!lower_remote_halo_program(receiver, receive_halos, receive_storage, false, 100,
                                   lowered_receive, why),
        "cyclic storage aliases were accepted by remote lowering");
  receive_storage.arrays[0].alias_of = invalid_array();
  receive_storage.arrays[1].alias_of = ArrayId{0};
  receive_halos.find(comms_key{E_stuff, CONNECT_PHASE, chunk_pair{0, 1}})
      ->phase_values.clear();
  CHECK(!lower_remote_halo_program(receiver, receive_halos, receive_storage, false, 100,
                                   lowered_receive, why),
        "malformed PHASE literal coverage was accepted by remote lowering");

  RemoteHaloProgram malformed_receiver;
  halo_plan_set malformed_halos;
  StoragePlan malformed_storage;
  make_endpoint(false, 0, malformed_receiver, malformed_halos, malformed_storage);
  malformed_halos.find(comms_key{E_stuff, CONNECT_PHASE, chunk_pair{0, 1}})
      ->phase_values.push_back(std::complex<realnum>(realnum(0.6), realnum(0.8)));
  malformed_halos.find(comms_key{E_stuff, CONNECT_COPY, chunk_pair{0, 1}})->block_offset = 1;
  CHECK(!lower_remote_halo_program(malformed_receiver, malformed_halos, malformed_storage,
                                   false, 100, lowered_receive, why),
        "receiver reconstruction accepted a HaloPlan phase-span mismatch");

  RemoteHaloProgram real_receiver;
  halo_plan_set real_halos;
  StoragePlan real_storage;
  make_endpoint(false, 0, real_receiver, real_halos, real_storage);
  real_halos.find(comms_key{E_stuff, CONNECT_PHASE, chunk_pair{0, 1}})
      ->phase_values.push_back(std::complex<realnum>(realnum(0.6), realnum(0.8)));
  CHECK(!lower_remote_halo_program(real_receiver, real_halos, real_storage,
                                   true, 100, lowered_receive, why),
        "real-field lowering accepted a PHASE communication block");

  StoragePlan zero_storage;
  zero_storage.arrays.push_back(array_spec(0, Precision::f64, 8));
  zero_storage.arrays.push_back(array_spec(1, Precision::f64, 8));
  zero_storage.arrays[1].alias_of = ArrayId{0};
  zero_storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Ex), 0, 0});
  zero_storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Hx), 0, 0});
  ZeroPlan zero;
  zero.residue.push_back(ElementRef{ArrayId{0}, 3});
  zero.residue.push_back(ElementRef{ArrayId{1}, 4});
  std::vector<LoweredHaloZeroDescriptor> lowered_zeroes;
  CHECK(lower_canonical_halo_zeroes(std::vector<ZeroPlan>(1, zero), zero_storage,
                                    lowered_zeroes, why),
        why.c_str());
  CHECK(lowered_zeroes.size() == 2 && lowered_zeroes[0].target.root == ArrayId{0} &&
            lowered_zeroes[1].target.root == ArrayId{0} &&
            lowered_zeroes[1].target.index == 4,
        "canonical halo zero lowering did not preserve alias-root identity");
  zero.residue[1].index = -1;
  CHECK(!lower_canonical_halo_zeroes(std::vector<ZeroPlan>(1, zero), zero_storage,
                                     lowered_zeroes, why),
        "canonical halo zero lowering accepted a negative scalar index");
}

void test_checked_descriptor_inputs() {
  RemoteHaloProgram sender;
  halo_plan_set halos;
  StoragePlan storage;
  make_endpoint(true, 0, sender, halos, storage);
  LoweredRemoteHaloProgram baseline;
  std::string why;
  CHECK(lower_remote_halo_program(sender, halos, storage, false, 100, baseline, why),
        why.c_str());
  const uint64_t baseline_signature = baseline.authority_signature;
  const size_t baseline_stages = baseline.stages.size();

  HaloPlan *copy = halos.find(comms_key{E_stuff, CONNECT_COPY, chunk_pair{0, 1}});
  CHECK(copy && !copy->gather_order.empty() && !copy->gather.empty(),
        "checked-input fixture has no COPY residue");
  if (!copy || copy->gather_order.empty() || copy->gather.empty()) return;
  const HaloPlan saved = *copy;
  copy->gather_order[0].residue = UINT32_MAX;
  CHECK(!lower_remote_halo_program(sender, halos, storage, false, 100, baseline, why) &&
            baseline.authority_signature == baseline_signature &&
            baseline.stages.size() == baseline_stages,
        "out-of-range residue segment did not reject transactionally");
  *copy = saved;
  copy->gather[0].index = std::numeric_limits<ptrdiff_t>::max();
  CHECK(!lower_remote_halo_program(sender, halos, storage, false, 100, baseline, why),
        "out-of-allocation residue index was expanded before rejection");
  *copy = saved;
  copy->gather.clear();
  copy->gather_order.clear();
  SlabRef malformed;
  malformed.array = ArrayId{0};
  malformed.base = 0;
  malformed.counts[0] = -1;
  malformed.counts[1] = malformed.counts[2] = 1;
  malformed.strides[0] = 1;
  malformed.strides[1] = malformed.strides[2] = 0;
  copy->gather_slabs.push_back(malformed);
  copy->gather_order.push_back(HaloSegment{0, 1, 1, 0});
  CHECK(!lower_remote_halo_program(sender, halos, storage, false, 100, baseline, why),
        "negative halo slab count was accepted");
  copy->gather_slabs[0].counts[0] = 2;
  copy->gather_slabs[0].base = std::numeric_limits<ptrdiff_t>::max();
  copy->gather_slabs[0].strides[0] = std::numeric_limits<ptrdiff_t>::max();
  copy->gather_order[0].count = 2;
  copy->block_elements = 2;
  sender.stages[0].sends[0].phases.back().block_elements = 2;
  sender.stages[0].sends[0].total_elements = 4;
  sender.stages[0].sends[0].wire_bytes = 4 * sizeof(float);
  sender.stages[0].sends[0].wire_digest =
      compute_remote_halo_wire_digest(sender.stages[0].sends[0]);
  sender.stages[0].signature = compute_remote_halo_stage_signature(sender.stages[0]);
  sender.signature = compute_remote_halo_program_signature(sender);
  CHECK(!lower_remote_halo_program(sender, halos, storage, false, 100, baseline, why),
        "signed halo slab arithmetic overflow was accepted");

  StoragePlan zero_storage;
  zero_storage.arrays.push_back(array_spec(0, Precision::f64, 8));
  zero_storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Ex), 0, 0});
  ZeroPlan zero;
  zero.residue.push_back(ElementRef{ArrayId{0}, 2});
  zero.residue.push_back(ElementRef{ArrayId{0}, 2});
  std::vector<LoweredHaloZeroDescriptor> lowered_zeroes;
  CHECK(lower_canonical_halo_zeroes(std::vector<ZeroPlan>(1, zero), zero_storage,
                                    lowered_zeroes, why) && lowered_zeroes.size() == 1,
        "duplicate physical zero target was not deterministically deduplicated");
  zero.residue.clear();
  SlabRef bad_zero;
  bad_zero.array = ArrayId{0};
  bad_zero.base = 0;
  bad_zero.counts[0] = -1;
  bad_zero.counts[1] = bad_zero.counts[2] = 1;
  bad_zero.strides[0] = 1;
  bad_zero.strides[1] = bad_zero.strides[2] = 0;
  zero.slabs.push_back(bad_zero);
  CHECK(!lower_canonical_halo_zeroes(std::vector<ZeroPlan>(1, zero), zero_storage,
                                     lowered_zeroes, why) && lowered_zeroes.size() == 1,
        "negative zero slab count did not reject transactionally");
  zero.slabs[0].counts[0] = 2;
  zero.slabs[0].base = std::numeric_limits<ptrdiff_t>::max();
  zero.slabs[0].strides[0] = std::numeric_limits<ptrdiff_t>::max();
  CHECK(!lower_canonical_halo_zeroes(std::vector<ZeroPlan>(1, zero), zero_storage,
                                     lowered_zeroes, why),
        "signed zero slab arithmetic overflow was accepted");
}

void test_mixed_precision_slot_alignment() {
  StoragePlan storage;
  storage.arrays.push_back(array_spec(0, Precision::f32, 4));
  storage.arrays.push_back(array_spec(1, Precision::f64, 4));
  storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Ex), 0, 0});
  storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Ey), 0, 0});
  halo_plan_set halos;
  const auto add_plan = [&](int destination, ArrayId array) {
    HaloPlan &plan = halos.get_or_create(
        comms_key{E_stuff, CONNECT_COPY, chunk_pair{0, destination}});
    plan.peer_rank = destination;
    plan.same_rank = false;
    plan.storage = HaloStorageDisposition::canonical;
    plan.block_offset = 0;
    plan.block_elements = 1;
    plan.gather.push_back(ElementRef{array, 0});
    plan.gather_order.push_back(HaloSegment{0, 0, 0, 1});
  };
  add_plan(1, ArrayId{0});
  add_plan(2, ArrayId{1});
  const auto operation = [](int destination, int tag) {
    comms_operation op;
    op.my_chunk_idx = 0;
    op.other_chunk_idx = destination;
    op.other_proc_id = destination;
    op.pair_idx = destination * 3;
    op.transfer_size = 1;
    op.comm_direction = Outgoing;
    op.tag = tag;
    return op;
  };
  comms_sequence sequences[NUM_FIELD_TYPES];
  sequences[E_stuff].send_ops.push_back(operation(1, 7));
  sequences[E_stuff].send_ops.push_back(operation(2, 8));
  std::string why;
  RemoteHaloProgram program;
  LoweredRemoteHaloProgram lowered;
  CHECK(build_remote_halo_program(sequences, halos, storage, std::vector<int>{0, 1, 2},
                                  0, 3, 1, GpuMpiPolicy::staged, GpuMpiRoute::staged,
                                  true, 1, 100, program, why) &&
            lower_remote_halo_program(program, halos, storage, false, 100, lowered, why),
        why.c_str());
  CHECK(lowered.stages.size() == 1 && lowered.stages[0].send_slot_bytes == 16 &&
            lowered.stages[0].sends[0].arena_offsets[0] == 0 &&
            lowered.stages[0].sends[1].arena_offsets[0] == 8 &&
            lowered.stages[0].sends[1].arena_offsets[1] == 24,
        "f32-before-f64 stage slot was not padded to maximum alignment");
  std::reverse(sequences[E_stuff].send_ops.begin(), sequences[E_stuff].send_ops.end());
  CHECK(build_remote_halo_program(sequences, halos, storage, std::vector<int>{0, 1, 2},
                                  0, 3, 1, GpuMpiPolicy::staged, GpuMpiRoute::staged,
                                  true, 1, 100, program, why) &&
            lower_remote_halo_program(program, halos, storage, false, 100, lowered, why),
        why.c_str());
  CHECK(lowered.stages.size() == 1 && lowered.stages[0].send_slot_bytes == 16 &&
            lowered.stages[0].sends[0].arena_offsets[0] == 0 &&
            lowered.stages[0].sends[1].arena_offsets[0] == 8 &&
            lowered.stages[0].sends[1].arena_offsets[1] == 24,
        "f64-before-f32 stage slot was not padded to maximum alignment");
}

} // namespace

int main() {
  test_policy();
  test_endpoint_invariant_identity();
  test_complete_block_rejections();
  test_manual_validation();
  test_idle_admission();
  test_zero_length_wire_message();
  test_neutral_remote_lowering();
  test_checked_descriptor_inputs();
  test_mixed_precision_slot_alignment();
  if (failures) {
    std::fprintf(stderr, "transport_plan: %d failures\n", failures);
    return 1;
  }
  std::printf("transport_plan: PASS\n");
  return 0;
}
