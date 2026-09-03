/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_boundaries.hpp"
#include "backend/nvidia/cuda_hip_compat.hpp"
#include "backend/storage_plan.hpp"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace meep {
namespace nvidia {
namespace {

void check_cuda(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) return;
  throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

size_t element_bytes(const ArraySpec &spec) {
  switch (spec.element_type) {
    case ElementType::realnum_value:
      return spec.storage == Precision::f32 ? sizeof(float) : sizeof(double);
    case ElementType::complex_realnum:
      return spec.storage == Precision::f32 ? 2 * sizeof(float) : 2 * sizeof(double);
    case ElementType::float64: return sizeof(double);
    case ElementType::complex_float64: return 2 * sizeof(double);
    case ElementType::int32: return sizeof(int32_t);
    case ElementType::index: return sizeof(size_t);
  }
  return 0;
}

bool checked_storage_bytes(const ArraySpec &spec, size_t &bytes) {
  const size_t scalar = element_bytes(spec);
  if (!scalar || (spec.elements && scalar > std::numeric_limits<size_t>::max() / spec.elements))
    return false;
  bytes = scalar * spec.elements;
  return true;
}

arena_role expected_arena_role(array_role role) {
  switch (role) {
    case array_role::field: return arena_role::field;
    case array_role::material: return arena_role::material;
    case array_role::polarization: return arena_role::polarization;
    case array_role::dft: return arena_role::dft;
    case array_role::communication: return arena_role::communication;
    case array_role::scratch: return arena_role::scratch;
  }
  return arena_role::count;
}

bool resolve_allocations(const StoragePlan &storage, const device_arenas &arenas,
                         std::vector<boundary_device_allocation> &resolved,
                         std::string &why) {
  resolved.clear();
  if (storage.arrays.size() != storage.keys.size()) {
    why = "NVIDIA remote boundary storage authority has inconsistent rows";
    return false;
  }
  try {
    for (size_t i = 0; i < storage.arrays.size(); ++i) {
      const ArraySpec &spec = storage.arrays[i];
      if (is_valid(spec.alias_of) || spec.classification_elided) continue;
      const device_allocation allocation = arenas.resolve(i);
      if (allocation.id != i || allocation.canonical_id != i) {
        why = "NVIDIA remote boundary arena did not resolve a canonical owner";
        return false;
      }
      resolved.push_back(boundary_device_allocation{
          spec.id, ArrayId{uint32_t(allocation.canonical_id)}, storage.keys[i], spec.role,
          allocation.role, allocation.offset, allocation.address, allocation.bytes,
          allocation.alignment, arenas.device()});
    }
  }
  catch (const std::exception &error) {
    why = error.what();
    return false;
  }
  return true;
}

bool validate_allocations(const StoragePlan &storage,
                          const std::vector<boundary_device_allocation> &allocations,
                          int owning_device,
                          std::vector<const boundary_device_allocation *> &by_id,
                          std::string &why) {
  by_id.assign(storage.arrays.size(), NULL);
  std::vector<std::pair<uintptr_t, uintptr_t> > ranges;
  std::vector<uintptr_t> role_bases(size_t(arena_role::count), 0);
  for (const boundary_device_allocation &allocation : allocations) {
    if (!is_valid(allocation.id) || allocation.id.value >= storage.arrays.size() ||
        by_id[allocation.id.value]) {
      why = "NVIDIA remote boundary allocation has a wrong or duplicate ArrayId";
      return false;
    }
    const ArraySpec &spec = storage.arrays[allocation.id.value];
    const arena_role required_arena = expected_arena_role(spec.role);
    size_t expected_bytes = 0;
    if (spec.id != allocation.id || allocation.canonical_id != allocation.id ||
        is_valid(spec.alias_of) ||
        !(storage.keys[allocation.id.value] == allocation.key) ||
        spec.role != allocation.role || required_arena == arena_role::count ||
        allocation.arena != required_arena ||
        !checked_storage_bytes(spec, expected_bytes) || allocation.bytes != expected_bytes ||
        !spec.elements || !spec.alignment || (spec.alignment & (spec.alignment - 1)) ||
        allocation.alignment < spec.alignment || allocation.alignment % spec.alignment ||
        !allocation.alignment || !allocation.address || allocation.device != owning_device ||
        reinterpret_cast<uintptr_t>(allocation.address) % allocation.alignment) {
      why = "NVIDIA remote boundary allocation disagrees with canonical storage authority";
      return false;
    }
    cudaPointerAttributes attributes;
    cudaError_t pointer_result = cudaPointerGetAttributes(&attributes, allocation.address);
    if (pointer_result != cudaSuccess) {
      (void)cudaGetLastError();
      why = "NVIDIA remote boundary allocation is not CUDA device storage";
      return false;
    }
    if (MEEP_POINTER_MEMORY_TYPE(attributes) != cudaMemoryTypeDevice ||
        attributes.device != owning_device) {
      why = "NVIDIA remote boundary allocation is on the wrong CUDA device";
      return false;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(allocation.address);
    if (allocation.bytes > std::numeric_limits<uintptr_t>::max() - begin) {
      why = "NVIDIA remote boundary allocation address range overflows";
      return false;
    }
    const uintptr_t end = begin + allocation.bytes;
    if (allocation.arena_offset > begin) {
      why = "NVIDIA remote boundary arena offset underflows its base";
      return false;
    }
    const size_t role = size_t(allocation.arena);
    const uintptr_t role_base = begin - allocation.arena_offset;
    if (!role_bases[role]) role_bases[role] = role_base;
    else if (role_bases[role] != role_base) {
      why = "NVIDIA remote boundary allocations do not share authoritative role-arena identity";
      return false;
    }
    for (const std::pair<uintptr_t, uintptr_t> &range : ranges)
      if (begin < range.second && range.first < end) {
        why = "NVIDIA remote boundary canonical allocations overlap physically";
        return false;
      }
    ranges.push_back(std::make_pair(begin, end));
    by_id[allocation.id.value] = &allocation;
  }
  for (size_t i = 0; i < storage.arrays.size(); ++i)
    if (!is_valid(storage.arrays[i].alias_of) && !storage.arrays[i].classification_elided &&
        !by_id[i]) {
      why = "NVIDIA remote boundary binding is missing a canonical allocation";
      return false;
    }
  return true;
}

bool bind_scalar(const RemoteHaloScalarRef &source, Precision precision, const StoragePlan &storage,
                 const std::vector<const boundary_device_allocation *> &device_allocations,
                 void *&address, ptrdiff_t &index,
                 std::string &why) {
  if (!is_valid(source.root) || source.root.value >= storage.arrays.size() ||
      source.root.value >= device_allocations.size() || !device_allocations[source.root.value]) {
    why = "NVIDIA remote boundary binding references an absent device allocation";
    return false;
  }
  const ArraySpec &spec = storage.arrays[source.root.value];
  size_t bytes = 0;
  if (spec.id != source.root || is_valid(spec.alias_of) || spec.classification_provisional ||
      spec.classification_elided || spec.element_type != ElementType::realnum_value ||
      spec.storage != precision || source.index < 0 || size_t(source.index) >= spec.elements ||
      !(storage.keys[source.root.value] == source.key) || source.role != spec.role ||
      source.element_type != spec.element_type || source.storage_precision != spec.storage ||
      source.elements != spec.elements || source.alignment != spec.alignment ||
      !checked_storage_bytes(spec, bytes) || source.byte_extent != bytes) {
    why = "NVIDIA remote boundary binding references incompatible canonical storage";
    return false;
  }
  const boundary_device_allocation &allocation = *device_allocations[source.root.value];
  if (allocation.id != source.root || !(allocation.key == source.key) ||
      allocation.bytes != source.byte_extent || allocation.alignment != source.alignment) {
    why = "NVIDIA remote boundary allocation identity changed after lowering";
    return false;
  }
  address = allocation.address;
  index = source.index;
  return true;
}

bool valid_message_layout(const LoweredRemoteHaloMessage &message, size_t slot_bytes,
                          std::string &why) {
  if (message.storage_precision != Precision::f32 && message.storage_precision != Precision::f64) {
    why = "NVIDIA remote boundary binding received invalid storage precision";
    return false;
  }
  const size_t element_bytes =
      message.storage_precision == Precision::f32 ? sizeof(float) : sizeof(double);
  size_t arena_bytes = 0;
  if (slot_bytes > std::numeric_limits<size_t>::max() / 2) {
    why = "NVIDIA remote boundary arena byte extent overflows";
    return false;
  }
  arena_bytes = 2 * slot_bytes;
  if (message.element_bytes != element_bytes || message.wire_bytes % element_bytes ||
      message.slot_offsets[0] != 0 ||
      message.slot_offsets[1] != message.wire_bytes || message.arena_offsets[0] % element_bytes ||
      message.arena_offsets[0] > slot_bytes ||
      message.wire_bytes > slot_bytes - message.arena_offsets[0] ||
      message.arena_offsets[0] > std::numeric_limits<size_t>::max() - slot_bytes ||
      message.arena_offsets[1] != slot_bytes + message.arena_offsets[0] ||
      message.arena_offsets[1] % element_bytes ||
      message.arena_offsets[1] > arena_bytes ||
      message.wire_bytes > arena_bytes - message.arena_offsets[1]) {
    why = "NVIDIA remote boundary binding received an invalid message layout";
    return false;
  }
  return true;
}

bool mark_exact_cover(std::vector<unsigned char> &cover, size_t byte_offset,
                      size_t scalar_count, size_t element_bytes, std::string &why) {
  if (byte_offset % element_bytes) {
    why = "NVIDIA remote boundary descriptor is not scalar aligned";
    return false;
  }
  const size_t first = byte_offset / element_bytes;
  if (first > cover.size() || scalar_count > cover.size() - first) {
    why = "NVIDIA remote boundary descriptor exceeds its wire message";
    return false;
  }
  for (size_t i = 0; i < scalar_count; ++i)
    if (cover[first + i]++) {
      why = "NVIDIA remote boundary descriptor covers a wire scalar more than once";
      return false;
    }
  return true;
}

bool complete_cover(const std::vector<unsigned char> &cover) {
  return std::find(cover.begin(), cover.end(), 0) == cover.end();
}

bool valid_stage_layout(const std::vector<LoweredRemoteHaloMessage> &messages,
                        size_t slot_bytes, std::string &why) {
  size_t max_alignment = 1, cursor = 0;
  for (const LoweredRemoteHaloMessage &message : messages) {
    if (!valid_message_layout(message, slot_bytes, why)) return false;
    max_alignment = std::max(max_alignment, message.element_bytes);
    const size_t mask = message.element_bytes - 1;
    if (cursor > std::numeric_limits<size_t>::max() - mask) {
      why = "NVIDIA remote boundary stage alignment overflows";
      return false;
    }
    const size_t aligned = (cursor + mask) & ~mask;
    if (message.arena_offsets[0] != aligned) {
      why = "NVIDIA remote boundary stage does not replay canonical lowering order";
      return false;
    }
    if (message.wire_bytes > std::numeric_limits<size_t>::max() - aligned) {
      why = "NVIDIA remote boundary stage byte extent overflows";
      return false;
    }
    cursor = aligned + message.wire_bytes;
  }
  if (cursor > std::numeric_limits<size_t>::max() - (max_alignment - 1)) {
    why = "NVIDIA remote boundary stage alignment overflows";
    return false;
  }
  const size_t expected_slot = (cursor + max_alignment - 1) & ~(max_alignment - 1);
  if (slot_bytes != expected_slot) {
    why = "NVIDIA remote boundary stage slot padding is stale";
    return false;
  }
  return true;
}

bool wire_key_less(const RemoteHaloWireKey &a, const RemoteHaloWireKey &b) {
  return std::tie(a.source_rank, a.destination_rank, a.ft, a.source_chunk, a.destination_chunk,
                  a.tag, a.canonical_ordinal) < std::tie(b.source_rank, b.destination_rank, b.ft,
                                                         b.source_chunk, b.destination_chunk, b.tag,
                                                         b.canonical_ordinal);
}

template <typename T>
__global__ void boundary_gather_kernel(const boundary_gather_entry *entries, size_t first,
                                       size_t count, unsigned char *arena) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= count) return;
  const boundary_gather_entry entry = entries[first + linear];
  *reinterpret_cast<T *>(arena + entry.buffer_byte_offset) =
      static_cast<const T *>(entry.source)[entry.source_index];
}

template <typename T>
__device__ void apply_scatter(const boundary_scatter_entry &entry, const unsigned char *arena) {
  const T *input = reinterpret_cast<const T *>(arena + entry.buffer_byte_offset);
  const T input_real = input[0];
  const T phase_real = T(entry.phase_real);
  if (!entry.target_imag) {
    static_cast<T *>(entry.target_real)[entry.target_real_index] = phase_real * input_real;
    return;
  }
  const T input_imag = input[1];
  const T phase_imag = T(entry.phase_imag);
  static_cast<T *>(entry.target_real)[entry.target_real_index] =
      phase_real * input_real - phase_imag * input_imag;
  static_cast<T *>(entry.target_imag)[entry.target_imag_index] =
      phase_real * input_imag + phase_imag * input_real;
}

template <typename T>
__global__ void boundary_scatter_kernel(const boundary_scatter_entry *entries, size_t first,
                                        size_t count, const unsigned char *arena) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear < count) apply_scatter<T>(entries[first + linear], arena);
}

template <typename T>
__global__ void boundary_scatter_serial_kernel(const boundary_scatter_entry *entries, size_t first,
                                               size_t count, const unsigned char *arena) {
  if (blockIdx.x || threadIdx.x) return;
  for (size_t i = 0; i < count; ++i)
    apply_scatter<T>(entries[first + i], arena);
}

template <typename T>
__global__ void boundary_zero_kernel(const boundary_zero_entry *entries, size_t first,
                                     size_t count) {
  const size_t linear = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= count) return;
  const boundary_zero_entry entry = entries[first + linear];
  static_cast<T *>(entry.target)[entry.target_index] = T(0);
}

void launch_geometry(const boundary_launch &launch, const stream &execution_stream,
                     unsigned int &blocks, unsigned int &threads) {
  if (!launch.count) throw std::invalid_argument("NVIDIA boundary launch is empty");
  if (launch.first > launch.extent || launch.count > launch.extent - launch.first)
    throw std::out_of_range("NVIDIA boundary launch exceeds its immutable descriptor table");
  threads = 256;
  const size_t block_count = 1 + (launch.count - 1) / threads;
  int maximum_grid_x = 0;
  {
    device_scope scope(execution_stream.device());
    check_cuda(cudaDeviceGetAttribute(&maximum_grid_x, cudaDevAttrMaxGridDimX,
                                      execution_stream.device()),
               "query NVIDIA boundary launch grid bound");
  }
  if (maximum_grid_x <= 0 || block_count > size_t(maximum_grid_x) ||
      block_count > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("NVIDIA boundary launch grid overflows");
  blocks = static_cast<unsigned int>(block_count);
}

template <typename T>
void gather(const boundary_launch &launch, const void *entries, void *arena,
            const stream &execution_stream) {
  unsigned int blocks = 0, threads = 0;
  launch_geometry(launch, execution_stream, blocks, threads);
  boundary_gather_kernel<T>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          static_cast<const boundary_gather_entry *>(entries), launch.first, launch.count,
          static_cast<unsigned char *>(arena));
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA remote boundary gather");
}

template <typename T>
void scatter_parallel(const boundary_launch &launch, const void *entries, const void *arena,
                      const stream &execution_stream) {
  unsigned int blocks = 0, threads = 0;
  launch_geometry(launch, execution_stream, blocks, threads);
  boundary_scatter_kernel<T>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          static_cast<const boundary_scatter_entry *>(entries), launch.first, launch.count,
          static_cast<const unsigned char *>(arena));
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA remote boundary scatter");
}

template <typename T>
void scatter_serial(const boundary_launch &launch, const void *entries, const void *arena,
                    const stream &execution_stream) {
  unsigned int ignored_blocks = 0, ignored_threads = 0;
  launch_geometry(launch, execution_stream, ignored_blocks, ignored_threads);
  boundary_scatter_serial_kernel<T>
      <<<1, 1, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          static_cast<const boundary_scatter_entry *>(entries), launch.first, launch.count,
          static_cast<const unsigned char *>(arena));
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA ordered remote boundary scatter");
}

template <typename T>
void zero(const boundary_launch &launch, const void *entries, const stream &execution_stream) {
  unsigned int blocks = 0, threads = 0;
  launch_geometry(launch, execution_stream, blocks, threads);
  boundary_zero_kernel<T>
      <<<blocks, threads, 0, static_cast<cudaStream_t>(execution_stream.opaque_handle())>>>(
          static_cast<const boundary_zero_entry *>(entries), launch.first, launch.count);
  check_cuda(cudaPeekAtLastError(), "launch NVIDIA remote boundary zero");
}

} // namespace

bool bind_remote_boundaries_from_records(
    const LoweredRemoteHaloProgram &lowered, const StoragePlan &storage,
    const std::vector<boundary_device_allocation> &device_allocations,
    int owning_device, bound_boundary_program &bound, std::string &why) {
  why.clear();
  if (lowered.version != LoweredRemoteHaloProgram::schema_version ||
      storage.arrays.size() != storage.keys.size() ||
      lowered.storage_signature != compute_remote_storage_authority_signature(storage) ||
      lowered.authority_signature != compute_remote_lowered_authority_signature(
                                         lowered.program_signature,
                                         lowered.storage_signature)) {
    why = "NVIDIA remote boundary binding received stale lowering authority";
    return false;
  }
  std::vector<const boundary_device_allocation *> allocations_by_id;
  if (!validate_allocations(storage, device_allocations, owning_device,
                            allocations_by_id, why))
    return false;
  bound_boundary_program staged;
  staged.program_signature = lowered.program_signature;
  staged.storage_signature = lowered.storage_signature;
  staged.authority_signature = lowered.authority_signature;
  for (const LoweredRemoteHaloStage &source_stage : lowered.stages) {
    bound_boundary_stage stage;
    stage.ft = source_stage.ft;
    stage.receive_slot_bytes = source_stage.receive_slot_bytes;
    stage.send_slot_bytes = source_stage.send_slot_bytes;
    stage.publication = source_stage.publication;
    stage.canonical_receive_order = source_stage.canonical_receive_order;
    if ((stage.publication != RemoteHaloPublicationMode::parallel_unique &&
         stage.publication != RemoteHaloPublicationMode::canonical_serial) ||
        int(stage.ft) < 0 || int(stage.ft) >= NUM_FIELD_TYPES) {
      why = "NVIDIA remote boundary stage has invalid publication metadata";
      return false;
    }
    if (!valid_stage_layout(source_stage.sends, source_stage.send_slot_bytes, why) ||
        !valid_stage_layout(source_stage.receives, source_stage.receive_slot_bytes, why))
      return false;
    for (const LoweredRemoteHaloMessage &message : source_stage.sends) {
      if (message.direction != RemoteHaloDirection::outgoing ||
          !valid_message_layout(message, source_stage.send_slot_bytes, why))
        return false;
      const size_t first = stage.gathers.size();
      std::vector<unsigned char> cover(message.wire_bytes / message.element_bytes, 0);
      for (const RemoteHaloGatherDescriptor &source : message.gathers) {
        if (source.buffer_byte_offset % message.element_bytes ||
            source.buffer_byte_offset > message.wire_bytes ||
            message.element_bytes > message.wire_bytes - source.buffer_byte_offset) {
          why = "NVIDIA remote gather descriptor exceeds its wire message";
          return false;
        }
        if (!mark_exact_cover(cover, source.buffer_byte_offset, 1, message.element_bytes, why))
          return false;
        void *address = NULL;
        ptrdiff_t index = 0;
        if (!bind_scalar(source.source, message.storage_precision, storage, allocations_by_id,
                         address, index, why))
          return false;
        stage.gathers.push_back(boundary_gather_entry{address, index, source.buffer_byte_offset});
      }
      if (message.gathers.size() != message.wire_bytes / message.element_bytes ||
          !complete_cover(cover)) {
        why = "NVIDIA remote gather descriptors do not cover the complete wire message";
        return false;
      }
      stage.sends.push_back(bound_boundary_message{
          message.key,
          message.direction,
          boundary_launch{first, message.gathers.size(), 0,
                          message.storage_precision == Precision::f32 ? scalar_precision::f32
                                                                      : scalar_precision::f64},
          message.wire_bytes,
          {message.slot_offsets[0], message.slot_offsets[1]},
          {message.arena_offsets[0], message.arena_offsets[1]}});
    }
    std::set<uintptr_t> destination_scalars;
    bool overlap = false;
    for (const LoweredRemoteHaloMessage &message : source_stage.receives) {
      if (message.direction != RemoteHaloDirection::incoming ||
          !valid_message_layout(message, source_stage.receive_slot_bytes, why))
        return false;
      const size_t first = stage.scatters.size();
      size_t covered_scalars = 0;
      std::vector<unsigned char> cover(message.wire_bytes / message.element_bytes, 0);
      for (const RemoteHaloScatterDescriptor &source : message.scatters) {
        const size_t scalars = is_valid(source.target_imag.root) ? 2 : 1;
        size_t bytes = 0;
        if (source.buffer_byte_offset % message.element_bytes ||
            scalars > std::numeric_limits<size_t>::max() / message.element_bytes ||
            (bytes = scalars * message.element_bytes) > message.wire_bytes ||
            source.buffer_byte_offset > message.wire_bytes - bytes) {
          why = "NVIDIA remote scatter descriptor exceeds its wire message";
          return false;
        }
        if (source.canonical_element_ordinal != covered_scalars) {
          why = "NVIDIA remote scatter canonical element order is invalid";
          return false;
        }
        if (!mark_exact_cover(cover, source.buffer_byte_offset, scalars,
                              message.element_bytes, why))
          return false;
        covered_scalars += scalars;
        void *target_real = NULL, *target_imag = NULL;
        ptrdiff_t target_real_index = 0, target_imag_index = 0;
        if (!bind_scalar(source.target_real, message.storage_precision, storage, allocations_by_id,
                         target_real, target_real_index, why))
          return false;
        if (is_valid(source.target_imag.root) &&
            !bind_scalar(source.target_imag, message.storage_precision, storage, allocations_by_id,
                         target_imag, target_imag_index, why))
          return false;
        stage.scatters.push_back(boundary_scatter_entry{
            target_real, target_real_index, target_imag, target_imag_index,
            source.buffer_byte_offset, source.phase_real, source.phase_imag});
        const uintptr_t real_address = reinterpret_cast<uintptr_t>(target_real) +
                                       size_t(target_real_index) * message.element_bytes;
        overlap = !destination_scalars.insert(real_address).second || overlap;
        if (is_valid(source.target_imag.root))
          overlap = !destination_scalars
                         .insert(reinterpret_cast<uintptr_t>(target_imag) +
                                 size_t(target_imag_index) * message.element_bytes)
                         .second || overlap;
      }
      if (covered_scalars != message.wire_bytes / message.element_bytes) {
        why = "NVIDIA remote scatter descriptors do not cover the complete wire message";
        return false;
      }
      if (!complete_cover(cover)) {
        why = "NVIDIA remote scatter descriptors leave holes in the wire message";
        return false;
      }
      stage.receives.push_back(bound_boundary_message{
          message.key,
          message.direction,
          boundary_launch{first, message.scatters.size(), 0,
                          message.storage_precision == Precision::f32 ? scalar_precision::f32
                                                                      : scalar_precision::f64},
          message.wire_bytes,
          {message.slot_offsets[0], message.slot_offsets[1]},
          {message.arena_offsets[0], message.arena_offsets[1]}});
    }
    if ((overlap && stage.publication != RemoteHaloPublicationMode::canonical_serial) ||
        (!overlap && stage.publication != RemoteHaloPublicationMode::parallel_unique)) {
      why = "NVIDIA remote boundary overlap disposition is stale";
      return false;
    }
    if (stage.canonical_receive_order.size() != stage.receives.size()) {
      why = "NVIDIA remote boundary canonical receive order is incomplete";
      return false;
    }
    std::vector<unsigned char> seen(stage.receives.size(), 0);
    for (size_t i = 0; i < stage.canonical_receive_order.size(); ++i) {
      const uint32_t index = stage.canonical_receive_order[i];
      if (index >= stage.receives.size() || seen[index] ||
          (i && wire_key_less(stage.receives[index].key,
                              stage.receives[stage.canonical_receive_order[i - 1]].key))) {
        why = "NVIDIA remote boundary canonical receive order is invalid";
        return false;
      }
      seen[index] = 1;
    }
    for (bound_boundary_message &message : stage.sends)
      message.launch.extent = stage.gathers.size();
    for (bound_boundary_message &message : stage.receives)
      message.launch.extent = stage.scatters.size();
    staged.stages.push_back(stage);
  }
  bound = staged;
  return true;
}

bool bind_boundary_zeroes_from_records(
    const std::vector<LoweredHaloZeroDescriptor> &lowered, const StoragePlan &storage,
    const std::vector<boundary_device_allocation> &device_allocations,
    int owning_device, bound_boundary_zeroes &bound, std::string &why) {
  why.clear();
  if (storage.arrays.size() != storage.keys.size()) {
    why = "NVIDIA boundary zero binding received stale storage authority";
    return false;
  }
  std::vector<const boundary_device_allocation *> allocations_by_id;
  if (!validate_allocations(storage, device_allocations, owning_device,
                            allocations_by_id, why))
    return false;
  bound_boundary_zeroes staged;
  std::set<uintptr_t> targets;
  for (const LoweredHaloZeroDescriptor &source : lowered) {
    void *address = NULL;
    ptrdiff_t index = 0;
    if (!bind_scalar(source.target, source.storage_precision, storage, allocations_by_id, address,
                     index, why))
      return false;
    const size_t scalar_bytes = source.storage_precision == Precision::f32 ? sizeof(float)
                                                                           : sizeof(double);
    if (!targets.insert(reinterpret_cast<uintptr_t>(address) + size_t(index) * scalar_bytes).second) {
      why = "NVIDIA boundary zero binding contains a duplicate physical target";
      return false;
    }
    const scalar_precision precision =
        source.storage_precision == Precision::f32 ? scalar_precision::f32 : scalar_precision::f64;
    if (staged.launches.empty() || staged.launches.back().precision != precision) {
      staged.launches.push_back(boundary_launch{staged.entries.size(), 0, 0, precision});
    }
    staged.entries.push_back(boundary_zero_entry{address, index});
    ++staged.launches.back().count;
  }
  for (boundary_launch &launch : staged.launches) launch.extent = staged.entries.size();
  bound = staged;
  return true;
}

bool bind_remote_boundaries(const LoweredRemoteHaloProgram &lowered,
                            const StoragePlan &storage,
                            const device_arenas &device_allocations,
                            bound_boundary_program &bound, std::string &why) {
  std::vector<boundary_device_allocation> resolved;
  if (!resolve_allocations(storage, device_allocations, resolved, why)) return false;
  return bind_remote_boundaries_from_records(lowered, storage, resolved,
                                             device_allocations.device(), bound, why);
}

bool bind_boundary_zeroes(const std::vector<LoweredHaloZeroDescriptor> &lowered,
                          const StoragePlan &storage,
                          const device_arenas &device_allocations,
                          bound_boundary_zeroes &bound, std::string &why) {
  std::vector<boundary_device_allocation> resolved;
  if (!resolve_allocations(storage, device_allocations, resolved, why)) return false;
  return bind_boundary_zeroes_from_records(lowered, storage, resolved,
                                           device_allocations.device(), bound, why);
}

namespace testing {
bool bind_remote_boundaries_with_allocations(
    const LoweredRemoteHaloProgram &lowered, const StoragePlan &storage,
    const std::vector<boundary_device_allocation> &device_allocations,
    int owning_device, bound_boundary_program &bound, std::string &why) {
  return bind_remote_boundaries_from_records(lowered, storage, device_allocations,
                                             owning_device, bound, why);
}

bool bind_boundary_zeroes_with_allocations(
    const std::vector<LoweredHaloZeroDescriptor> &lowered, const StoragePlan &storage,
    const std::vector<boundary_device_allocation> &device_allocations,
    int owning_device, bound_boundary_zeroes &bound, std::string &why) {
  return bind_boundary_zeroes_from_records(lowered, storage, device_allocations,
                                           owning_device, bound, why);
}
} // namespace testing

void launch_boundary_gather(const boundary_launch &launch, const void *device_entries,
                            void *device_arena, const stream &execution_stream) {
  if (!device_entries || !device_arena)
    throw std::invalid_argument("NVIDIA remote boundary gather has incomplete storage");
  if (launch.precision == scalar_precision::f32)
    gather<float>(launch, device_entries, device_arena, execution_stream);
  else if (launch.precision == scalar_precision::f64)
    gather<double>(launch, device_entries, device_arena, execution_stream);
  else
    throw std::invalid_argument("NVIDIA remote boundary gather has invalid precision");
}

void launch_boundary_scatter_parallel(const boundary_launch &launch, const void *device_entries,
                                      const void *device_arena, const stream &execution_stream) {
  if (!device_entries || !device_arena)
    throw std::invalid_argument("NVIDIA remote boundary scatter has incomplete storage");
  if (launch.precision == scalar_precision::f32)
    scatter_parallel<float>(launch, device_entries, device_arena, execution_stream);
  else if (launch.precision == scalar_precision::f64)
    scatter_parallel<double>(launch, device_entries, device_arena, execution_stream);
  else
    throw std::invalid_argument("NVIDIA remote boundary scatter has invalid precision");
}

void launch_boundary_scatter_serial(const boundary_launch &launch, const void *device_entries,
                                    const void *device_arena, const stream &execution_stream) {
  if (!device_entries || !device_arena)
    throw std::invalid_argument("NVIDIA ordered remote boundary scatter has incomplete storage");
  if (!launch.count) throw std::invalid_argument("NVIDIA boundary launch is empty");
  if (launch.precision == scalar_precision::f32)
    scatter_serial<float>(launch, device_entries, device_arena, execution_stream);
  else if (launch.precision == scalar_precision::f64)
    scatter_serial<double>(launch, device_entries, device_arena, execution_stream);
  else
    throw std::invalid_argument("NVIDIA ordered remote boundary scatter has invalid precision");
}

void launch_boundary_zero(const boundary_launch &launch, const void *device_entries,
                          const stream &execution_stream) {
  if (!device_entries)
    throw std::invalid_argument("NVIDIA remote boundary zero has incomplete storage");
  if (launch.precision == scalar_precision::f32)
    zero<float>(launch, device_entries, execution_stream);
  else if (launch.precision == scalar_precision::f64)
    zero<double>(launch, device_entries, execution_stream);
  else
    throw std::invalid_argument("NVIDIA remote boundary zero has invalid precision");
}

} // namespace nvidia
} // namespace meep
