/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* CUDA-free graph lowering schema.
 *
 * This is a backend-private description of which parts of one canonical
 * StepPlan may be captured.  It intentionally carries no CUDA handle or raw
 * address.  Halo locality comes from a separately signed HaloPlan authority;
 * it is never inferred from OpKind or BufferAccess.
 */

#ifndef MEEP_BACKEND_GRAPH_PLAN_HPP
#define MEEP_BACKEND_GRAPH_PLAN_HPP

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "backend/halo_plan.hpp"
#include "backend/storage_plan.hpp"
#include "backend/step_plan.hpp"

namespace meep {

enum class GraphNodeClass { device, device_predicate, host_boundary };

enum class GraphBoundaryKind {
  none,
  source_evaluation,
  host_callback,
  material_phase,
  material_refresh,
  remote_halo,
  legacy_flux_publish,
  finite_diagnostic,
  time_update,
  magnetic_transition,
  segment_guard
};

enum class GraphVariantKind { ordinary, magnetic_half_step, magnetic_restore, cw_operator };

enum class GraphExecutionMode { automatic, eager, required };

const char *graph_execution_mode_name(GraphExecutionMode mode);
GraphExecutionMode parse_graph_execution_mode(const char *value);

struct GraphRankModeSupport {
  GraphExecutionMode requested;
  bool runtime_capture_supported;
  bool program_graphable;
};

struct GraphModeResolution {
  GraphExecutionMode requested;
  bool use_graph;
};

/* Pure collective-resolution seam.  Callers gather one row per rank, then all
   ranks run this deterministic resolver over the same vector.  Host boundaries
   do not make a program ungraphable; program_graphable describes only whether
   every selected device segment has a graph representation. */
GraphModeResolution
resolve_collective_graph_execution_mode(const std::vector<GraphRankModeSupport> &ranks);

/* Graph-required accepts plans containing supported host boundaries.  Every
   device segment still has to be capturable.  Auto is deliberately
   all-or-nothing: before the first dispatch it may select the eager program,
   but it never mixes eager and graph execution segment by segment. */
enum class GraphFallbackPolicy { whole_program_eager };

enum class GraphHaloRoute { local_device, remote_host, host_owned };

struct GraphArrayIdentity {
  ArrayId source_id; // lookup only; deliberately excluded from stable signatures
  StorageKey key;
  array_role role;
  ElementType element_type;
  Precision storage;
  size_t elements;
  size_t alignment;
  bool classification_provisional;
  bool classification_elided;
  uint64_t signature;
};

bool operator==(const GraphArrayIdentity &a, const GraphArrayIdentity &b);
inline bool operator!=(const GraphArrayIdentity &a, const GraphArrayIdentity &b) {
  return !(a == b);
}

struct GraphHaloRow {
  field_type ft;
  chunk_pair chunks;
  connect_phase phase;
  int peer_rank;
  int tag;
  GraphHaloRoute route;
  uint32_t sequence_index;
  size_t block_offset;
  size_t block_elements;
  uint64_t signature;
};

bool operator==(const GraphHaloRow &a, const GraphHaloRow &b);
inline bool operator!=(const GraphHaloRow &a, const GraphHaloRow &b) { return !(a == b); }

struct GraphHaloDisposition {
  uint32_t operation_index;
  field_type ft;
  uint32_t row_index;
  uint32_t row_count;
  bool entirely_local_canonical;
  uint64_t signature;
};

bool operator==(const GraphHaloDisposition &a, const GraphHaloDisposition &b);
inline bool operator!=(const GraphHaloDisposition &a, const GraphHaloDisposition &b) {
  return !(a == b);
}

struct GraphZeroRow {
  field_type ft;
  uint32_t chunk;
  size_t elements;
  uint64_t signature;
};

bool operator==(const GraphZeroRow &a, const GraphZeroRow &b);
inline bool operator!=(const GraphZeroRow &a, const GraphZeroRow &b) { return !(a == b); }

struct GraphZeroDisposition {
  uint32_t operation_index;
  field_type ft;
  uint32_t row_index;
  uint32_t row_count;
  uint64_t signature;
};

bool operator==(const GraphZeroDisposition &a, const GraphZeroDisposition &b);
inline bool operator!=(const GraphZeroDisposition &a, const GraphZeroDisposition &b) {
  return !(a == b);
}

struct GraphHostInterval {
  uint32_t marker_operation;
  uint32_t first_covered_operation;
  uint32_t covered_operation_count;
  uint32_t host_segment_index;
  uint64_t signature;
};

bool operator==(const GraphHostInterval &a, const GraphHostInterval &b);
inline bool operator!=(const GraphHostInterval &a, const GraphHostInterval &b) { return !(a == b); }

struct GraphRemoteOverlap {
  uint32_t halo_operation;
  uint32_t update_operation;
  uint64_t dependency_signature;
  uint64_t signature;
};

bool operator==(const GraphRemoteOverlap &a, const GraphRemoteOverlap &b);
inline bool operator!=(const GraphRemoteOverlap &a, const GraphRemoteOverlap &b) {
  return !(a == b);
}
uint64_t compute_graph_remote_overlap_signature(const GraphRemoteOverlap &overlap);

struct GraphLoweringAuthorities {
  uint64_t step_plan_signature;
  uint64_t halo_signature;
  uint64_t cw_plan_signature;
  uint64_t cw_stable_signature;
  std::vector<GraphArrayIdentity> array_identities;
  std::vector<GraphHaloRow> halo_rows;
  std::vector<GraphHaloDisposition> halo_dispositions;
  std::vector<GraphZeroRow> zero_rows;
  std::vector<GraphZeroDisposition> zero_dispositions;
  std::vector<GraphHostInterval> host_intervals;
  std::vector<GraphRemoteOverlap> remote_overlaps;
  uint64_t signature;

  GraphLoweringAuthorities()
      : step_plan_signature(0), halo_signature(0), cw_plan_signature(0), cw_stable_signature(0),
        signature(0) {}
};

GraphLoweringAuthorities build_graph_lowering_authorities(const StepPlan &plan,
                                                          const halo_plan_set *halos,
                                                          const CpuArrayCatalog *catalog = NULL,
                                                          int field_interleave = 1,
                                                          const CwPlan *cw_plan = NULL);
uint64_t compute_graph_lowering_authorities_signature(const GraphLoweringAuthorities &authority);
bool validate_graph_lowering_authorities(const StepPlan &plan,
                                         const GraphLoweringAuthorities &authority,
                                         const halo_plan_set *halos = NULL,
                                         const CpuArrayCatalog *catalog = NULL,
                                         int field_interleave = 1, const CwPlan *cw_plan = NULL,
                                         std::string *error = NULL);

/* The fixed scalar block is written by a tiny same-stream kernel.  Source
   callback values continue to use their existing separately owned device
   buffer.  Predicate bits are assigned by StepScalarLayout; each distinct DFT
   decimation factor gets its own bit. */
const uint32_t step_scalars_abi_version = 1;
const size_t step_scalar_predicate_word_count = 64;
static_assert(step_scalar_predicate_word_count * 64 == cw_dft_predicate_capacity,
              "CW final-DFT predicate capacity must match StepScalars");

struct StepScalars {
  uint32_t abi_version;
  uint32_t byte_size;
  int64_t entry_timestep;
  int64_t post_increment_timestep;
  uint64_t noisy_counter_time;
  double source_times[3];
  uint64_t batch_ordinal;
  uint64_t batch_count;
  int64_t dft_timestep;
  uint64_t noisy_seed_generation;
  int32_t active_noisy_seed_slot;
  uint32_t finite_check_mode;
  uint32_t finite_check_due;
  uint32_t graph_variant;
  uint32_t material_phase_result;
  uint32_t reserved;
  uint64_t predicate_words[step_scalar_predicate_word_count];
};

enum class StepScalarSemantic {
  abi_version,
  byte_size,
  entry_timestep,
  post_increment_timestep,
  noisy_counter_time,
  source_time_0,
  source_time_half,
  source_time_1,
  batch_ordinal,
  batch_count,
  dft_timestep,
  noisy_seed_generation,
  active_noisy_seed_slot,
  finite_check_mode,
  finite_check_due,
  graph_variant,
  material_phase_result,
  guard_predicate,
  dft_due_predicate
};

enum class StepScalarType { u32, i32, u64, i64, f64, predicate_bit };

struct StepScalarSlot {
  StepScalarSemantic semantic;
  StepScalarType type;
  uint32_t semantic_index;
  uint32_t byte_offset;
  uint32_t bit_offset;
  uint32_t byte_size;
};

bool operator==(const StepScalarSlot &a, const StepScalarSlot &b);
inline bool operator!=(const StepScalarSlot &a, const StepScalarSlot &b) { return !(a == b); }

struct StepScalarLayout {
  uint32_t abi_version;
  uint32_t total_bytes;
  std::vector<StepScalarSlot> slots;
  uint64_t signature;

  StepScalarLayout() : abi_version(step_scalars_abi_version), total_bytes(0), signature(0) {}
};

StepScalarLayout build_step_scalar_layout(const StepPlan &plan);

bool operator==(const StepScalarLayout &a, const StepScalarLayout &b);
inline bool operator!=(const StepScalarLayout &a, const StepScalarLayout &b) { return !(a == b); }

struct GraphOperationRef {
  uint32_t operation_index;
  GraphNodeClass node_class;
  Operation operation;
  std::vector<uint32_t> scalar_slots;
  uint64_t signature;
};

bool operator==(const GraphOperationRef &a, const GraphOperationRef &b);
inline bool operator!=(const GraphOperationRef &a, const GraphOperationRef &b) { return !(a == b); }

struct GraphSegment {
  GraphVariantKind variant;
  uint32_t first_operation;
  uint32_t operation_count;
  std::vector<GraphOperationRef> operations;
  GraphBoundaryKind exit_boundary;
  uint64_t signature;
};

bool operator==(const GraphSegment &a, const GraphSegment &b);
inline bool operator!=(const GraphSegment &a, const GraphSegment &b) { return !(a == b); }

struct GraphBoundary {
  GraphBoundaryKind kind;
  uint32_t first_operation;
  uint32_t operation_count;
  bool completion_only;
  GraphOperationRef operation;
  uint64_t signature;
};

bool operator==(const GraphBoundary &a, const GraphBoundary &b);
inline bool operator!=(const GraphBoundary &a, const GraphBoundary &b) { return !(a == b); }

enum class GraphScheduleKind { segment, boundary };

struct GraphScheduleEntry {
  GraphScheduleKind kind;
  uint32_t index;
};

bool operator==(const GraphScheduleEntry &a, const GraphScheduleEntry &b);
inline bool operator!=(const GraphScheduleEntry &a, const GraphScheduleEntry &b) {
  return !(a == b);
}

struct GraphProgram {
  StepProgram program;
  GraphVariantKind variant;
  GraphFallbackPolicy fallback_policy;
  uint64_t step_plan_signature;
  uint64_t authority_signature;
  uint64_t cw_plan_signature;
  StepScalarLayout scalar_layout;
  std::vector<GraphSegment> segments;
  std::vector<GraphBoundary> boundaries;
  std::vector<GraphScheduleEntry> schedule;
  uint64_t signature;

  GraphProgram()
      : program(StepProgram::ordinary), variant(GraphVariantKind::ordinary),
        fallback_policy(GraphFallbackPolicy::whole_program_eager), step_plan_signature(0),
        authority_signature(0), cw_plan_signature(0), signature(0) {}
};

/* Host boundaries are legal in required mode.  This query is false only when
   lowering failed to represent a selected device operation as a graph node;
   runtime/capture capability is checked separately before dispatch. */
bool graph_required_compatible(const GraphProgram &program);

bool operator==(const GraphProgram &a, const GraphProgram &b);
inline bool operator!=(const GraphProgram &a, const GraphProgram &b) { return !(a == b); }

GraphProgram build_graph_program(const StepPlan &plan, const GraphLoweringAuthorities &authority,
                                 GraphVariantKind variant);
bool validate_graph_program(const StepPlan &plan, const GraphLoweringAuthorities &authority,
                            const GraphProgram &program, std::string *error = NULL);
uint64_t compute_graph_program_signature(const GraphProgram &program);
void format_graph_program(const GraphProgram &program, std::vector<std::string> &out);

} // namespace meep

#endif // MEEP_BACKEND_GRAPH_PLAN_HPP
