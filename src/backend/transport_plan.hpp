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

namespace meep {

class halo_plan_set;
struct StoragePlan;

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

bool validate_remote_halo_program(const RemoteHaloProgram &program, int mpi_tag_ub,
                                  std::string &why);
bool validate_remote_halo_participation(const RemoteHaloParticipation &participation,
                                        std::string &why);
void remote_halo_agreement_records(const RemoteHaloProgram &program,
                                   std::vector<RemoteHaloAgreementRecord> &records);
bool validate_remote_halo_agreement(const std::vector<RemoteHaloAgreementRecord> &records,
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
