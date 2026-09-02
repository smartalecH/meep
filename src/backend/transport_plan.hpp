/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#ifndef MEEP_BACKEND_TRANSPORT_PLAN_HPP
#define MEEP_BACKEND_TRANSPORT_PLAN_HPP

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "meep.hpp"
#include "backend/array_ref.hpp"
#include "backend/halo_plan.hpp"
#include "backend/storage_plan.hpp"

namespace meep {

enum class GpuMpiPolicy : uint32_t { staged = 0, automatic = 1, direct = 2 };
enum class GpuMpiRoute : uint32_t { staged = 0, direct = 1 };
enum class RemoteHaloDirection : uint32_t { incoming = 0, outgoing = 1 };

struct GpuMpiPolicyParse {
  bool valid;
  GpuMpiPolicy requested;
  std::string error;
};

GpuMpiPolicyParse parse_gpu_mpi_policy(const char *value);
bool resolve_gpu_mpi_route(GpuMpiPolicy policy, bool provider_query_available,
                           bool provider_supports_direct, GpuMpiRoute &route,
                           std::string &why);
const char *gpu_mpi_policy_name(GpuMpiPolicy policy);
const char *gpu_mpi_route_name(GpuMpiRoute route);

/* Endpoint-invariant identity. canonical_ordinal is derived from the global
   field/source-chunk/destination-chunk construction order before local
   ownership filtering and before send optimization. */
struct RemoteHaloWireKey {
  int source_rank;
  int destination_rank;
  field_type ft;
  int source_chunk;
  int destination_chunk;
  int tag;
  uint64_t canonical_ordinal;
};

bool operator==(const RemoteHaloWireKey &a, const RemoteHaloWireKey &b);
inline bool operator!=(const RemoteHaloWireKey &a, const RemoteHaloWireKey &b) {
  return !(a == b);
}
bool remote_halo_wire_key_less(const RemoteHaloWireKey &a, const RemoteHaloWireKey &b);

struct RemoteHaloPhaseSpan {
  connect_phase phase;
  uint32_t sequence_index;
  size_t block_offset;
  size_t block_elements;
};

bool operator==(const RemoteHaloPhaseSpan &a, const RemoteHaloPhaseSpan &b);

struct RemoteHaloMessage {
  RemoteHaloWireKey key;
  RemoteHaloDirection direction;
  uint32_t local_schedule_ordinal;
  std::vector<RemoteHaloPhaseSpan> phases;
  size_t total_elements;
  Precision storage_precision;
  size_t element_bytes;
  size_t wire_bytes;
  uint64_t wire_digest;
};

struct RemoteHaloStage {
  field_type ft;
  std::vector<RemoteHaloMessage> receives;
  std::vector<RemoteHaloMessage> sends;
  uint64_t signature;
};

/* Rank participation is explicit so an idle/no-device rank cannot silently
   omit work. A non-owner is valid only when all three work counts are zero. */
struct RemoteHaloParticipation {
  bool device_owner;
  size_t owned_field_chunks;
  size_t transport_messages;
  size_t cuda_required_operations;
};

struct RemoteHaloProgram {
  static const uint32_t schema_version = 1;

  uint32_t version;
  int communicator_rank;
  int communicator_size;
  uint64_t communicator_generation;
  GpuMpiPolicy requested_policy;
  GpuMpiRoute resolved_route;
  RemoteHaloParticipation participation;
  std::vector<RemoteHaloStage> stages;
  uint64_t signature;
};

/* Backend-neutral, allocation-free-at-launch lowering of the canonical wire
   program. Array references are resolved through aliases to their owning
   storage row so destination overlap is a physical scalar property rather
   than an ArrayId property. Descriptor and slot offsets are wire-message
   local and therefore endpoint invariant; arena_offsets are rank-local. */
struct RemoteHaloScalarRef {
  ArrayId root;
  StorageKey key;
  array_role role;
  ElementType element_type;
  Precision storage_precision;
  size_t elements;
  size_t alignment;
  size_t byte_extent;
  ptrdiff_t index;
};

inline RemoteHaloScalarRef invalid_remote_halo_scalar_ref() {
  RemoteHaloScalarRef result;
  result.root = invalid_array();
  result.key = StorageKey{-1, -1, -1, -1, 0};
  result.role = array_role::scratch;
  result.element_type = ElementType::realnum_value;
  result.storage_precision = Precision::f64;
  result.elements = 0;
  result.alignment = 0;
  result.byte_extent = 0;
  result.index = 0;
  return result;
}

struct RemoteHaloGatherDescriptor {
  RemoteHaloScalarRef source;
  size_t buffer_byte_offset;
};

struct RemoteHaloScatterDescriptor {
  RemoteHaloScalarRef target_real;
  RemoteHaloScalarRef target_imag; // invalid root for COPY/NEGATE
  size_t buffer_byte_offset;
  double phase_real;
  double phase_imag;
  uint64_t canonical_element_ordinal;
};

enum class RemoteHaloPublicationMode : uint32_t {
  parallel_unique = 0,
  canonical_serial = 1
};

struct LoweredRemoteHaloMessage {
  RemoteHaloWireKey key;
  RemoteHaloDirection direction;
  Precision storage_precision;
  size_t element_bytes;
  size_t wire_bytes;
  size_t slot_offsets[2];  // message-local: {0, wire_bytes}
  size_t arena_offsets[2]; // rank-local placement within each stage slot
  std::vector<RemoteHaloGatherDescriptor> gathers;
  std::vector<RemoteHaloScatterDescriptor> scatters;
};

struct LoweredRemoteHaloStage {
  field_type ft;
  std::vector<LoweredRemoteHaloMessage> receives;
  std::vector<LoweredRemoteHaloMessage> sends;
  size_t receive_slot_bytes;
  size_t send_slot_bytes;
  RemoteHaloPublicationMode publication;
  std::vector<uint32_t> canonical_receive_order;
};

struct LoweredRemoteHaloProgram {
  static const uint32_t schema_version = 1;
  uint32_t version;
  uint64_t program_signature;
  uint64_t storage_signature;
  uint64_t authority_signature;
  std::vector<LoweredRemoteHaloStage> stages;
};

struct LoweredHaloZeroDescriptor {
  RemoteHaloScalarRef target;
  Precision storage_precision;
};

/* One fixed-width record per local endpoint, suitable for collective exchange.
   Local schedule ordinals and local ArrayIds are intentionally absent. */
struct RemoteHaloAgreementRecord {
  RemoteHaloWireKey key;
  RemoteHaloDirection direction;
  uint64_t wire_digest;
  uint64_t wire_bytes;
};

uint64_t compute_remote_halo_wire_digest(const RemoteHaloMessage &message);
uint64_t compute_remote_halo_stage_signature(const RemoteHaloStage &stage);
uint64_t compute_remote_halo_program_signature(const RemoteHaloProgram &program);
inline uint64_t remote_authority_hash_u64(uint64_t h, uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) {
    h ^= (value >> (8 * i)) & 0xffu;
    h *= UINT64_C(1099511628211);
  }
  return h;
}
inline uint64_t compute_remote_storage_authority_signature(const StoragePlan &storage) {
  uint64_t h = UINT64_C(1469598103934665603);
  h = remote_authority_hash_u64(h, storage.arrays.size());
  h = remote_authority_hash_u64(h, storage.keys.size());
  const size_t count = storage.arrays.size() < storage.keys.size()
                           ? storage.arrays.size() : storage.keys.size();
  for (size_t i = 0; i < count; ++i) {
    const ArraySpec &spec = storage.arrays[i];
    const StorageKey &key = storage.keys[i];
    h = remote_authority_hash_u64(h, i);
    h = remote_authority_hash_u64(h, spec.id.value);
    h = remote_authority_hash_u64(h, uint64_t(uint32_t(spec.role)));
    h = remote_authority_hash_u64(h, uint64_t(uint32_t(spec.element_type)));
    h = remote_authority_hash_u64(h, uint64_t(uint32_t(spec.storage)));
    h = remote_authority_hash_u64(h, spec.elements);
    h = remote_authority_hash_u64(h, spec.alignment);
    h = remote_authority_hash_u64(
        h, is_valid(spec.alias_of) ? spec.alias_of.value : invalid_array_value);
    h = remote_authority_hash_u64(h, spec.classification_provisional ? 1 : 0);
    h = remote_authority_hash_u64(h, spec.classification_elided ? 1 : 0);
    h = remote_authority_hash_u64(h, uint64_t(int64_t(key.chunk)));
    h = remote_authority_hash_u64(h, uint64_t(int64_t(key.kind)));
    h = remote_authority_hash_u64(h, uint64_t(int64_t(key.component_)));
    h = remote_authority_hash_u64(h, uint64_t(int64_t(key.cmp)));
    h = remote_authority_hash_u64(h, key.aux);
  }
  return h;
}
inline uint64_t compute_remote_lowered_authority_signature(uint64_t program_signature,
                                                           uint64_t storage_signature) {
  return remote_authority_hash_u64(
      remote_authority_hash_u64(UINT64_C(1469598103934665603), program_signature),
      storage_signature);
}

bool validate_remote_halo_program(const RemoteHaloProgram &program, int mpi_tag_ub,
                                  std::string &why);
bool validate_remote_halo_participation(const RemoteHaloParticipation &participation,
                                        std::string &why);
void remote_halo_agreement_records(const RemoteHaloProgram &program,
                                   std::vector<RemoteHaloAgreementRecord> &records);
bool validate_remote_halo_agreement(const std::vector<RemoteHaloAgreementRecord> &records,
                                    std::string &why);

/* Independently revalidates the TransportProgram, exact HaloPlan phase cover,
   and installed StoragePlan. No CUDA or MPI types are involved. */
bool lower_remote_halo_program(const RemoteHaloProgram &program, const halo_plan_set &halos,
                               const StoragePlan &storage, bool real_fields, int mpi_tag_ub,
                               LoweredRemoteHaloProgram &lowered, std::string &why);
bool lower_canonical_halo_zeroes(const std::vector<ZeroPlan> &zeroes,
                                 const StoragePlan &storage,
                                 std::vector<LoweredHaloZeroDescriptor> &lowered,
                                 std::string &why);

/* Build from the legacy communication authority and canonicalized HaloPlans.
   chunk_ranks is indexed by global chunk id. The catalog supplies storage
   precision for canonical ArrayIds. Remote-only messages are emitted; local
   schedule order remains endpoint-local. */
bool build_remote_halo_program(const comms_sequence sequences[NUM_FIELD_TYPES],
                               const halo_plan_set &halos, const StoragePlan &storage,
                               const std::vector<int> &chunk_ranks, int communicator_rank,
                               int communicator_size, uint64_t communicator_generation,
                               GpuMpiPolicy requested_policy, GpuMpiRoute resolved_route,
                               bool device_owner, size_t cuda_required_operations,
                               int mpi_tag_ub, RemoteHaloProgram &program, std::string &why);

} // namespace meep

#endif
