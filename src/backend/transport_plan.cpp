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

bool checked_add(size_t a, size_t b, size_t &result) {
  if (b > std::numeric_limits<size_t>::max() - a) return false;
  result = a + b;
  return true;
}

bool checked_align(size_t value, size_t alignment, size_t &result) {
  if (!alignment || (alignment & (alignment - 1))) return false;
  const size_t mask = alignment - 1;
  if (value > std::numeric_limits<size_t>::max() - mask) return false;
  result = (value + mask) & ~mask;
  return true;
}

struct ScalarIdentity {
  StorageKey key;
  size_t byte_offset;
  bool operator<(const ScalarIdentity &other) const {
    return std::tie(key.chunk, key.kind, key.component_, key.cmp, key.aux, byte_offset) <
           std::tie(other.key.chunk, other.key.kind, other.key.component_, other.key.cmp,
                    other.key.aux, other.byte_offset);
  }
};

bool valid_precision(Precision precision) {
  return precision == Precision::f32 || precision == Precision::f64;
}

bool valid_role(array_role role) {
  return role == array_role::field || role == array_role::material ||
         role == array_role::polarization || role == array_role::dft ||
         role == array_role::communication || role == array_role::scratch;
}

bool valid_element_type(ElementType type) {
  return type == ElementType::realnum_value || type == ElementType::complex_realnum ||
         type == ElementType::float64 || type == ElementType::complex_float64 ||
         type == ElementType::int32 || type == ElementType::index;
}

bool scalar_storage_bytes(const ArraySpec &spec, size_t &bytes) {
  size_t element_bytes = 0;
  switch (spec.element_type) {
    case ElementType::realnum_value:
      element_bytes = spec.storage == Precision::f32 ? sizeof(float) : sizeof(double);
      break;
    case ElementType::complex_realnum:
      element_bytes = spec.storage == Precision::f32 ? 2 * sizeof(float) : 2 * sizeof(double);
      break;
    case ElementType::float64: element_bytes = sizeof(double); break;
    case ElementType::complex_float64: element_bytes = 2 * sizeof(double); break;
    case ElementType::int32: element_bytes = sizeof(int32_t); break;
    case ElementType::index: element_bytes = sizeof(size_t); break;
    default: return false;
  }
  return checked_multiply(spec.elements, element_bytes, bytes);
}

bool validate_storage_authority(const StoragePlan &storage, std::string &why) {
  if (storage.arrays.size() != storage.keys.size()) {
    why = "remote halo lowering received inconsistent storage authority";
    return false;
  }
  std::set<std::tuple<int, int, int, int, uint64_t> > keys;
  for (size_t i = 0; i < storage.arrays.size(); ++i) {
    const ArraySpec &spec = storage.arrays[i];
    const StorageKey &key = storage.keys[i];
    size_t bytes = 0;
    if (spec.id.value != i || !valid_role(spec.role) || !valid_element_type(spec.element_type) ||
        !valid_precision(spec.storage) || !spec.elements || !spec.alignment ||
        (spec.alignment & (spec.alignment - 1)) ||
        (is_valid(spec.alias_of) && spec.alias_of.value >= storage.arrays.size()) ||
        !scalar_storage_bytes(spec, bytes)) {
      why = "remote halo lowering received malformed storage authority";
      return false;
    }
    if (!keys.insert(std::make_tuple(key.chunk, key.kind, key.component_, key.cmp, key.aux)).second) {
      why = "remote halo lowering received duplicate StorageKeys";
      return false;
    }
    if (is_valid(spec.alias_of)) {
      ArrayId current = spec.alias_of;
      std::set<uint32_t> visited;
      visited.insert(spec.id.value);
      while (is_valid(current)) {
        if (current.value >= storage.arrays.size() || !visited.insert(current.value).second) {
          why = "remote halo lowering received a dangling or cyclic storage alias";
          return false;
        }
        const ArraySpec &target = storage.arrays[current.value];
        if (target.elements != spec.elements || target.element_type != spec.element_type ||
            target.storage != spec.storage || target.role != spec.role) {
          why = "remote halo lowering received an alias with incompatible storage metadata";
          return false;
        }
        current = target.alias_of;
      }
    }
  }
  return true;
}

bool resolve_scalar(const StoragePlan &storage, const ElementRef &source,
                    Precision expected_precision, RemoteHaloScalarRef &result,
                    std::string &why) {
  if (!is_valid(source.array) || source.array.value >= storage.arrays.size()) {
    why = "remote halo lowering references an absent ArrayId";
    return false;
  }
  if (source.index < 0) {
    why = "remote halo lowering references a negative scalar index";
    return false;
  }
  ArrayId current = source.array;
  std::set<uint32_t> visited;
  const ArraySpec *logical = NULL;
  while (true) {
    if (current.value >= storage.arrays.size() || !visited.insert(current.value).second) {
      why = "remote halo storage alias is dangling or cyclic";
      return false;
    }
    const ArraySpec &spec = storage.arrays[current.value];
    if (spec.id != current || spec.classification_provisional || spec.classification_elided ||
        spec.element_type != ElementType::realnum_value || spec.storage != expected_precision ||
        size_t(source.index) >= spec.elements) {
      why = "remote halo lowering references incompatible or stale scalar storage";
      return false;
    }
    if (!logical) logical = &spec;
    else if (spec.elements != logical->elements || spec.storage != logical->storage ||
             spec.element_type != logical->element_type) {
      why = "remote halo alias changes scalar layout or precision";
      return false;
    }
    if (!is_valid(spec.alias_of)) {
      result.root = current;
      result.key = storage.keys[current.value];
      result.role = spec.role;
      result.element_type = spec.element_type;
      result.storage_precision = spec.storage;
      result.elements = spec.elements;
      result.alignment = spec.alignment;
      if (!scalar_storage_bytes(spec, result.byte_extent)) {
        why = "remote halo canonical storage byte extent overflows";
        return false;
      }
      result.index = source.index;
      return true;
    }
    current = spec.alias_of;
  }
}

bool checked_scalar_index(ptrdiff_t base, int count0, ptrdiff_t stride0,
                          int count1, ptrdiff_t stride1, int count2,
                          ptrdiff_t stride2, size_t elements, std::string &why) {
  if (count0 <= 0 || count1 <= 0 || count2 <= 0) {
    why = "remote halo slab contains a nonpositive count";
    return false;
  }
  __int128 minimum = base, maximum = base;
  const int counts[3] = {count0, count1, count2};
  const ptrdiff_t strides[3] = {stride0, stride1, stride2};
  for (unsigned dimension = 0; dimension < 3; ++dimension) {
    const __int128 delta = __int128(counts[dimension] - 1) * strides[dimension];
    if (delta < 0) minimum += delta;
    else maximum += delta;
  }
  if (minimum < 0 || maximum < minimum ||
      maximum >= __int128(elements) ||
      minimum < __int128(std::numeric_limits<ptrdiff_t>::min()) ||
      maximum > __int128(std::numeric_limits<ptrdiff_t>::max())) {
    why = "remote halo slab scalar extent is outside its allocation";
    return false;
  }
  return true;
}

bool validate_element_ref(const StoragePlan &storage, const ElementRef &ref,
                          Precision expected_precision, std::string &why) {
  RemoteHaloScalarRef ignored;
  return resolve_scalar(storage, ref, expected_precision, ignored, why);
}

bool validate_halo_side(const HaloPlan &plan, bool gather, const StoragePlan &storage,
                        Precision expected_precision, std::string &why) {
  const std::vector<SlabRef> &slabs = gather ? plan.gather_slabs : plan.scatter_slabs;
  const std::vector<ElementRef> &residue = gather ? plan.gather : plan.scatter;
  const std::vector<HaloSegment> &order = gather ? plan.gather_order : plan.scatter_order;
  std::vector<unsigned char> used_slabs(slabs.size(), 0);
  size_t residue_cursor = 0, expanded = 0;
  for (const HaloSegment &segment : order) {
    if (segment.nslabs) {
      size_t slab_end = 0, contribution = 0;
      if (!segment.count || segment.residue ||
          !checked_add(size_t(segment.first_slab), size_t(segment.nslabs), slab_end) ||
          slab_end > slabs.size() ||
          !checked_multiply(size_t(segment.nslabs), size_t(segment.count), contribution) ||
          !checked_add(expanded, contribution, expanded)) {
        why = "remote halo slab segment is malformed or overflows";
        return false;
      }
      for (size_t i = segment.first_slab; i < slab_end; ++i) {
        const SlabRef &slab = slabs[i];
        if (used_slabs[i] || slab.counts[0] != int(segment.count) ||
            slab.counts[1] != 1 || slab.counts[2] != 1 ||
            !validate_element_ref(storage, ElementRef{slab.array, slab.base},
                                  expected_precision, why) ||
            !checked_scalar_index(slab.base, slab.counts[0], slab.strides[0],
                                  slab.counts[1], slab.strides[1], slab.counts[2],
                                  slab.strides[2], storage.arrays[slab.array.value].elements,
                                  why)) {
          if (why.empty()) why = "remote halo slab segment reuses or changes a slab";
          return false;
        }
        used_slabs[i] = 1;
      }
    }
    else {
      size_t next = 0;
      if (segment.count || !segment.residue ||
          !checked_add(residue_cursor, size_t(segment.residue), next) ||
          next > residue.size() || !checked_add(expanded, size_t(segment.residue), expanded)) {
        why = "remote halo residue segment is malformed or overflows";
        return false;
      }
      for (size_t i = residue_cursor; i < next; ++i)
        if (!validate_element_ref(storage, residue[i], expected_precision, why)) return false;
      residue_cursor = next;
    }
  }
  if (residue_cursor != residue.size() ||
      std::find(used_slabs.begin(), used_slabs.end(), 0) != used_slabs.end() ||
      expanded != plan.block_elements) {
    why = "remote halo descriptor order does not exactly cover its phase";
    return false;
  }
  return true;
}

bool find_phase_plan(const halo_plan_set &halos, const RemoteHaloMessage &message,
                     const RemoteHaloPhaseSpan &span, const HaloPlan *&plan,
                     std::string &why) {
  plan = halos.find(comms_key{message.key.ft, span.phase,
                              chunk_pair{message.key.source_chunk,
                                         message.key.destination_chunk}});
  const int peer = message.direction == RemoteHaloDirection::outgoing
                       ? message.key.destination_rank
                       : message.key.source_rank;
  if (!plan || plan->ft != message.key.ft ||
      plan->chunks != chunk_pair{message.key.source_chunk, message.key.destination_chunk} ||
      plan->phase != span.phase || plan->sequence_index != span.sequence_index ||
      plan->block_offset != span.block_offset || plan->block_elements != span.block_elements ||
      plan->same_rank || plan->peer_rank != peer) {
    why = "remote halo lowering phase identity disagrees with HaloPlan authority";
    return false;
  }
  if (plan->storage != HaloStorageDisposition::canonical) {
    why = "remote halo lowering encountered a host-owned complete block";
    return false;
  }
  return true;
}

bool lower_message(const RemoteHaloMessage &message, const halo_plan_set &halos,
                   const StoragePlan &storage, bool real_fields, size_t arena_zero_offset,
                   LoweredRemoteHaloMessage &lowered, std::string &why) {
  lowered = LoweredRemoteHaloMessage();
  lowered.key = message.key;
  lowered.direction = message.direction;
  lowered.storage_precision = message.storage_precision;
  lowered.element_bytes = message.element_bytes;
  lowered.wire_bytes = message.wire_bytes;
  lowered.slot_offsets[0] = 0;
  lowered.slot_offsets[1] = message.wire_bytes;
  lowered.arena_offsets[0] = arena_zero_offset;
  lowered.arena_offsets[1] = 0;
  size_t authoritative_elements = 0;
  for (connect_phase phase : all_connect_phases) {
    const HaloPlan *plan = halos.find(comms_key{message.key.ft, phase,
                                                chunk_pair{message.key.source_chunk,
                                                           message.key.destination_chunk}});
    if (!plan || !plan->block_elements) continue;
    if (authoritative_elements > std::numeric_limits<size_t>::max() - plan->block_elements) {
      why = "remote halo authoritative phase coverage overflows";
      return false;
    }
    authoritative_elements += plan->block_elements;
    const auto match = std::find_if(message.phases.begin(), message.phases.end(),
                                    [&](const RemoteHaloPhaseSpan &span) {
                                      return span.phase == phase;
                                    });
    if (match == message.phases.end() || match->block_offset != plan->block_offset ||
        match->block_elements != plan->block_elements ||
        match->sequence_index != plan->sequence_index) {
      why = "remote halo message omits or changes an authoritative phase";
      return false;
    }
  }
  if (authoritative_elements != message.total_elements) {
    why = "remote halo authoritative phase coverage differs from the wire message";
    return false;
  }
  uint64_t element_ordinal = 0;
  for (const RemoteHaloPhaseSpan &span : message.phases) {
    const HaloPlan *plan = NULL;
    if (!find_phase_plan(halos, message, span, plan, why)) return false;
    if (!plan->host_gather.empty() || !plan->host_scatter.empty() ||
        !validate_halo_side(*plan, message.direction == RemoteHaloDirection::outgoing,
                            storage, message.storage_precision, why))
      return false;
    if (span.phase == CONNECT_PHASE && message.direction == RemoteHaloDirection::incoming &&
        (plan->block_elements % 2 ||
         plan->phase_values.size() != plan->block_elements / 2)) {
      why = "remote phase halo does not contain exact complex phase coverage";
      return false;
    }
    std::vector<ElementRef> refs;
    if (message.direction == RemoteHaloDirection::outgoing)
      expand_gather(*plan, refs);
    else
      expand_scatter(*plan, refs);
    if (refs.size() != span.block_elements) {
      why = "remote halo lowering expansion does not match its phase span";
      return false;
    }
    if (span.phase == CONNECT_PHASE && real_fields) {
      why = "remote phase halo requires complex field storage";
      return false;
    }
    size_t phase_byte_offset = 0;
    if (!checked_multiply(span.block_offset, message.element_bytes, phase_byte_offset)) {
      why = "remote halo lowering phase byte offset overflows";
      return false;
    }
    if (message.direction == RemoteHaloDirection::outgoing) {
      for (size_t i = 0; i < refs.size(); ++i) {
        RemoteHaloScalarRef source;
        if (!resolve_scalar(storage, refs[i], message.storage_precision, source, why)) return false;
        size_t relative = 0, byte_offset = 0;
        if (!checked_multiply(i, message.element_bytes, relative) ||
            !checked_add(phase_byte_offset, relative, byte_offset)) {
          why = "remote halo gather byte offset overflows";
          return false;
        }
        lowered.gathers.push_back(RemoteHaloGatherDescriptor{source, byte_offset});
      }
    }
    else if (span.phase == CONNECT_COPY || span.phase == CONNECT_NEGATE) {
      const double sign = span.phase == CONNECT_NEGATE ? -1.0 : 1.0;
      for (size_t i = 0; i < refs.size(); ++i, ++element_ordinal) {
        RemoteHaloScalarRef target;
        if (!resolve_scalar(storage, refs[i], message.storage_precision, target, why)) return false;
        size_t relative = 0, byte_offset = 0;
        if (!checked_multiply(i, message.element_bytes, relative) ||
            !checked_add(phase_byte_offset, relative, byte_offset)) {
          why = "remote halo scatter byte offset overflows";
          return false;
        }
        lowered.scatters.push_back(RemoteHaloScatterDescriptor{
            target, invalid_remote_halo_scalar_ref(), byte_offset, sign, 0.0,
            element_ordinal});
      }
    }
    else if (span.phase == CONNECT_PHASE) {
      if (refs.size() % 2 || plan->phase_values.size() != refs.size() / 2) {
        why = "remote phase halo does not contain valid complex scalar pairs";
        return false;
      }
      for (size_t i = 0; i < plan->phase_values.size(); ++i, element_ordinal += 2) {
        RemoteHaloScalarRef target_real, target_imag;
        if (!resolve_scalar(storage, refs[2 * i], message.storage_precision, target_real, why) ||
            !resolve_scalar(storage, refs[2 * i + 1], message.storage_precision, target_imag, why))
          return false;
        size_t relative = 0, byte_offset = 0;
        if (!checked_multiply(2 * i, message.element_bytes, relative) ||
            !checked_add(phase_byte_offset, relative, byte_offset)) {
          why = "remote phase scatter byte offset overflows";
          return false;
        }
        lowered.scatters.push_back(RemoteHaloScatterDescriptor{
            target_real, target_imag, byte_offset, double(plan->phase_values[i].real()),
            double(plan->phase_values[i].imag()), element_ordinal});
      }
    }
    else {
      why = "remote halo lowering encountered an invalid transform";
      return false;
    }
  }
  if (message.direction == RemoteHaloDirection::outgoing &&
      lowered.gathers.size() != message.total_elements) {
    why = "remote halo gather lowering does not cover the complete block";
    return false;
  }
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
  const bool gather = direction == RemoteHaloDirection::outgoing;
  const std::vector<SlabRef> &slabs = gather ? plan.gather_slabs : plan.scatter_slabs;
  const std::vector<ElementRef> &residue = gather ? plan.gather : plan.scatter;
  if (!plan.host_gather.empty() || !plan.host_scatter.empty()) {
    why = "canonical remote halo contains opaque host references";
    return false;
  }
  const auto observe = [&](ArrayId array) -> bool {
    if (!is_valid(array) || array.value >= storage.arrays.size()) {
      why = "remote halo references an absent canonical array";
      return false;
    }
    const ArraySpec &spec = storage.arrays[array.value];
    if (spec.id != array || spec.classification_provisional || spec.classification_elided ||
        spec.element_type != ElementType::realnum_value || !valid_precision(spec.storage)) {
      why = "remote halo references provisional, stale, or elided canonical storage";
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
    return true;
  };
  for (const SlabRef &slab : slabs)
    if (!observe(slab.array)) return false;
  for (const ElementRef &ref : residue)
    if (!observe(ref.array)) return false;
  if (!have_precision && plan.block_elements) {
    why = "canonical halo endpoint has no storage descriptors";
    return false;
  }
  return !have_precision || validate_halo_side(plan, gather, storage, precision, why);
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

bool lower_remote_halo_program(const RemoteHaloProgram &program, const halo_plan_set &halos,
                               const StoragePlan &storage, bool real_fields, int mpi_tag_ub,
                               LoweredRemoteHaloProgram &lowered, std::string &why) {
  why.clear();
  if (!validate_remote_halo_program(program, mpi_tag_ub, why)) return false;
  if (!validate_storage_authority(storage, why)) return false;

  LoweredRemoteHaloProgram staged;
  staged.version = LoweredRemoteHaloProgram::schema_version;
  staged.program_signature = program.signature;
  staged.storage_signature = compute_remote_storage_authority_signature(storage);
  staged.authority_signature = compute_remote_lowered_authority_signature(
      staged.program_signature, staged.storage_signature);
  for (const RemoteHaloStage &source_stage : program.stages) {
    LoweredRemoteHaloStage stage;
    stage.ft = source_stage.ft;
    stage.receive_slot_bytes = 0;
    stage.send_slot_bytes = 0;
    stage.publication = RemoteHaloPublicationMode::parallel_unique;

    const auto lower_side = [&](const std::vector<RemoteHaloMessage> &messages,
                                std::vector<LoweredRemoteHaloMessage> &destination,
                                size_t &slot_bytes) -> bool {
      size_t cursor = 0;
      destination.reserve(messages.size());
      for (const RemoteHaloMessage &message : messages) {
        size_t aligned = 0;
        if (!checked_align(cursor, message.element_bytes, aligned)) {
          why = "remote halo arena alignment overflows";
          return false;
        }
        LoweredRemoteHaloMessage compiled;
        if (!lower_message(message, halos, storage, real_fields, aligned, compiled, why))
          return false;
        if (!checked_add(aligned, message.wire_bytes, cursor)) {
          why = "remote halo arena byte extent overflows";
          return false;
        }
        destination.push_back(compiled);
      }
      size_t max_alignment = 1;
      for (const RemoteHaloMessage &message : messages)
        max_alignment = std::max(max_alignment, message.element_bytes);
      if (!checked_align(cursor, max_alignment, slot_bytes)) {
        why = "remote halo slot padding overflows";
        return false;
      }
      size_t both_slots = 0;
      if (!checked_multiply(slot_bytes, size_t(2), both_slots)) {
        why = "remote halo double-buffered arena size overflows";
        return false;
      }
      (void)both_slots;
      for (LoweredRemoteHaloMessage &message : destination)
        if (!checked_add(slot_bytes, message.arena_offsets[0], message.arena_offsets[1])) {
          why = "remote halo second-slot byte offset overflows";
          return false;
        }
      return true;
    };

    if (!lower_side(source_stage.receives, stage.receives, stage.receive_slot_bytes) ||
        !lower_side(source_stage.sends, stage.sends, stage.send_slot_bytes))
      return false;

    std::set<ScalarIdentity> destinations;
    for (const LoweredRemoteHaloMessage &message : stage.receives)
      for (const RemoteHaloScatterDescriptor &scatter : message.scatters) {
        size_t real_byte_offset = 0;
        if (!checked_multiply(size_t(scatter.target_real.index), message.element_bytes,
                              real_byte_offset)) {
          why = "remote halo destination byte offset overflows";
          return false;
        }
        const ScalarIdentity real{scatter.target_real.key, real_byte_offset};
        if (!destinations.insert(real).second)
          stage.publication = RemoteHaloPublicationMode::canonical_serial;
        if (is_valid(scatter.target_imag.root)) {
          size_t imag_byte_offset = 0;
          if (!checked_multiply(size_t(scatter.target_imag.index), message.element_bytes,
                                imag_byte_offset)) {
            why = "remote halo destination byte offset overflows";
            return false;
          }
          const ScalarIdentity imag{scatter.target_imag.key, imag_byte_offset};
          if (!destinations.insert(imag).second)
            stage.publication = RemoteHaloPublicationMode::canonical_serial;
        }
      }
    stage.canonical_receive_order.resize(stage.receives.size());
    for (size_t i = 0; i < stage.receives.size(); ++i) {
      if (i > std::numeric_limits<uint32_t>::max()) {
        why = "remote halo canonical receive index overflows";
        return false;
      }
      stage.canonical_receive_order[i] = uint32_t(i);
    }
    std::sort(stage.canonical_receive_order.begin(), stage.canonical_receive_order.end(),
              [&](uint32_t a, uint32_t b) {
                return remote_halo_wire_key_less(stage.receives[a].key,
                                                 stage.receives[b].key);
              });
    staged.stages.push_back(stage);
  }
  lowered = staged;
  return true;
}

bool lower_canonical_halo_zeroes(const std::vector<ZeroPlan> &zeroes,
                                 const StoragePlan &storage,
                                 std::vector<LoweredHaloZeroDescriptor> &lowered,
                                 std::string &why) {
  why.clear();
  if (!validate_storage_authority(storage, why)) return false;
  std::vector<LoweredHaloZeroDescriptor> staged;
  std::set<ScalarIdentity> targets;
  for (const ZeroPlan &zero : zeroes) {
    std::vector<ElementRef> refs;
    size_t expected = zero.residue.size();
    for (const SlabRef &slab : zero.slabs) {
      if (!is_valid(slab.array) || slab.array.value >= storage.arrays.size()) {
        why = "halo zero lowering references an absent slab ArrayId";
        return false;
      }
      size_t slab_elements = 1;
      for (unsigned dimension = 0; dimension < 3; ++dimension) {
        if (slab.counts[dimension] <= 0 ||
            !checked_multiply(slab_elements, size_t(slab.counts[dimension]), slab_elements)) {
          why = "halo zero slab count is invalid or overflows";
          return false;
        }
      }
      if (!checked_add(expected, slab_elements, expected) ||
          !checked_scalar_index(slab.base, slab.counts[0], slab.strides[0],
                                slab.counts[1], slab.strides[1], slab.counts[2],
                                slab.strides[2], storage.arrays[slab.array.value].elements,
                                why))
        return false;
      for (int i0 = 0; i0 < slab.counts[0]; ++i0)
        for (int i1 = 0; i1 < slab.counts[1]; ++i1)
          for (int i2 = 0; i2 < slab.counts[2]; ++i2)
            refs.push_back(ElementRef{
                slab.array, ptrdiff_t(__int128(slab.base) + __int128(i0) * slab.strides[0] +
                                     __int128(i1) * slab.strides[1] +
                                     __int128(i2) * slab.strides[2])});
    }
    refs.insert(refs.end(), zero.residue.begin(), zero.residue.end());
    if (refs.size() != expected) {
      why = "halo zero descriptor expansion changed its checked extent";
      return false;
    }
    for (const ElementRef &ref : refs) {
      if (!is_valid(ref.array) || ref.array.value >= storage.arrays.size()) {
        why = "halo zero lowering references an absent ArrayId";
        return false;
      }
      const Precision precision = storage.arrays[ref.array.value].storage;
      RemoteHaloScalarRef target;
      if (!resolve_scalar(storage, ref, precision, target, why)) return false;
      size_t byte_offset = 0;
      const size_t element_bytes = precision == Precision::f32 ? sizeof(float) : sizeof(double);
      if (!checked_multiply(size_t(target.index), element_bytes, byte_offset)) {
        why = "halo zero target byte offset overflows";
        return false;
      }
      if (!targets.insert(ScalarIdentity{target.key, byte_offset}).second) {
        /* Metal scans may name the same scalar through two canonical aliases.
           Preserve one deterministic first occurrence rather than issuing
           racing identical stores. */
        continue;
      }
      staged.push_back(LoweredHaloZeroDescriptor{target, precision});
    }
  }
  lowered = staged;
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
