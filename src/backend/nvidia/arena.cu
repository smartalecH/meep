/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
*/

#include "backend/nvidia/arena.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace meep {
namespace nvidia {

namespace {

size_t role_index(arena_role role) {
  const size_t value = static_cast<size_t>(role);
  if (value >= static_cast<size_t>(arena_role::count))
    throw std::invalid_argument("invalid NVIDIA arena role");
  return value;
}

void validate_element_type(arena_element_type type) {
  if (static_cast<size_t>(type) >= static_cast<size_t>(arena_element_type::count))
    throw std::invalid_argument("invalid NVIDIA arena element type");
}

void validate_storage_precision(arena_storage_precision precision) {
  if (static_cast<size_t>(precision) >= static_cast<size_t>(arena_storage_precision::count))
    throw std::invalid_argument("invalid NVIDIA arena storage precision");
}

bool is_power_of_two(size_t value) { return value && !(value & (value - 1)); }

size_t checked_add(size_t left, size_t right, const char *what) {
  if (right > std::numeric_limits<size_t>::max() - left)
    throw std::overflow_error(std::string("NVIDIA arena size overflow while ") + what);
  return left + right;
}

size_t align_up(size_t value, size_t alignment) {
  const size_t padding = (alignment - (value & (alignment - 1))) & (alignment - 1);
  return checked_add(value, padding, "aligning an allocation");
}

uintptr_t align_pointer(uintptr_t value, size_t alignment) {
  const uintptr_t mask = static_cast<uintptr_t>(alignment - 1);
  if (mask > std::numeric_limits<uintptr_t>::max() - value)
    throw std::overflow_error("NVIDIA arena address alignment overflow");
  return (value + mask) & ~mask;
}

void check_cuda(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) return;
  throw runtime_error(operation, static_cast<int>(result), cudaGetErrorName(result),
                      cudaGetErrorString(result));
}

void check_subrange(size_t allocation_bytes, size_t offset, size_t bytes, const char *operation) {
  if (offset > allocation_bytes || bytes > allocation_bytes - offset)
    throw std::out_of_range(std::string(operation) + ": range exceeds logical allocation");
}

struct request_record {
  allocation_request request;
  size_t canonical_index;
  size_t effective_alignment;

  explicit request_record(const allocation_request &request_)
      : request(request_), canonical_index(0), effective_alignment(request_.alignment) {}
};

bool request_id_less(const request_record &left, const request_record &right) {
  return left.request.id < right.request.id;
}

bool layout_id_less(const allocation_layout &left, allocation_id right) { return left.id < right; }

} // namespace

const char *arena_role_name(arena_role role) {
  switch (role) {
    case arena_role::field: return "field";
    case arena_role::material: return "material";
    case arena_role::polarization: return "polarization";
    case arena_role::dft: return "dft";
    case arena_role::communication: return "communication";
    case arena_role::scratch: return "scratch";
    case arena_role::count: break;
  }
  return "invalid";
}

arena_plan::arena_plan()
    : total_high_water_bytes_(0), total_reserved_bytes_(0), canonical_allocation_count_(0),
      alias_count_(0) {}

arena_plan::arena_plan(const std::vector<allocation_request> &requests)
    : total_high_water_bytes_(0), total_reserved_bytes_(0), canonical_allocation_count_(0),
      alias_count_(0) {
  std::vector<request_record> records;
  records.reserve(requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    const allocation_request &request = requests[i];
    if (request.id == no_allocation)
      throw std::invalid_argument("allocation ID uses the reserved no-allocation value");
    (void)role_index(request.role);
    validate_element_type(request.element_type);
    validate_storage_precision(request.storage_precision);
    if (!request.bytes) throw std::invalid_argument("NVIDIA arena allocations must be nonempty");
    if (!is_power_of_two(request.alignment))
      throw std::invalid_argument("NVIDIA arena alignment must be a nonzero power of two");
    records.push_back(request_record(request));
  }
  std::sort(records.begin(), records.end(), request_id_less);

  std::map<allocation_id, size_t> by_id;
  for (size_t i = 0; i < records.size(); ++i) {
    if (i && records[i - 1].request.id == records[i].request.id)
      throw std::invalid_argument("duplicate NVIDIA arena allocation ID");
    by_id[records[i].request.id] = i;
  }
  for (size_t i = 0; i < records.size(); ++i) {
    if (records[i].request.alias_of != no_allocation &&
        by_id.find(records[i].request.alias_of) == by_id.end())
      throw std::invalid_argument("NVIDIA arena alias refers to an unknown allocation ID");
  }

  /* Resolve every alias chain without depending on request order. 0 is unseen,
     1 is on the current chain, and 2 is fully resolved. */
  std::vector<unsigned char> state(records.size(), 0);
  for (size_t start = 0; start < records.size(); ++start) {
    if (state[start] == 2) continue;
    std::vector<size_t> chain;
    size_t current = start;
    while (state[current] != 2) {
      if (state[current] == 1) throw std::invalid_argument("NVIDIA arena alias cycle");
      state[current] = 1;
      chain.push_back(current);
      const allocation_id target = records[current].request.alias_of;
      if (target == no_allocation) break;
      current = by_id.find(target)->second;
    }
    const size_t canonical = state[current] == 2 ? records[current].canonical_index : current;
    for (std::vector<size_t>::reverse_iterator it = chain.rbegin(); it != chain.rend(); ++it) {
      records[*it].canonical_index = canonical;
      state[*it] = 2;
    }
  }

  for (size_t i = 0; i < records.size(); ++i) {
    const size_t canonical = records[i].canonical_index;
    const allocation_request &request = records[i].request;
    const allocation_request &owner = records[canonical].request;
    if (request.role != owner.role)
      throw std::invalid_argument("NVIDIA arena alias and target have different roles");
    if (request.bytes != owner.bytes)
      throw std::invalid_argument("NVIDIA arena alias and target have different byte extents");
    if (request.element_type != owner.element_type)
      throw std::invalid_argument("NVIDIA arena alias and target have different element types");
    if (request.storage_precision != owner.storage_precision)
      throw std::invalid_argument("NVIDIA arena alias and target have different storage precision");
    records[canonical].effective_alignment =
        std::max(records[canonical].effective_alignment, request.alignment);
  }

  const size_t role_count = static_cast<size_t>(arena_role::count);
  std::vector<size_t> high_water(role_count, 0);
  std::vector<size_t> max_alignment(role_count, 1);
  std::vector<size_t> canonical_counts(role_count, 0);
  std::vector<size_t> alias_counts(role_count, 0);
  std::vector<size_t> canonical_offsets(records.size(), 0);

  /* records are sorted by ID, so each role's allocation order is stable. */
  for (size_t i = 0; i < records.size(); ++i) {
    if (records[i].canonical_index != i) continue;
    const size_t role = role_index(records[i].request.role);
    const size_t alignment = records[i].effective_alignment;
    const size_t offset = align_up(high_water[role], alignment);
    high_water[role] = checked_add(offset, records[i].request.bytes, "placing an allocation");
    max_alignment[role] = std::max(max_alignment[role], alignment);
    canonical_offsets[i] = offset;
    ++canonical_counts[role];
    ++canonical_allocation_count_;
  }

  allocations_.reserve(records.size());
  for (size_t i = 0; i < records.size(); ++i) {
    const request_record &record = records[i];
    const request_record &owner = records[record.canonical_index];
    allocation_layout layout;
    layout.id = record.request.id;
    layout.role = record.request.role;
    layout.offset = canonical_offsets[record.canonical_index];
    layout.bytes = record.request.bytes;
    layout.alignment = record.request.alignment;
    layout.effective_alignment = owner.effective_alignment;
    layout.canonical_id = owner.request.id;
    layout.element_type = record.request.element_type;
    layout.storage_precision = record.request.storage_precision;
    allocations_.push_back(layout);
    if (layout.is_alias()) {
      ++alias_counts[role_index(layout.role)];
      ++alias_count_;
    }
  }

  for (size_t role = 0; role < role_count; ++role) {
    if (!high_water[role]) continue;
    role_layout layout;
    layout.role = static_cast<arena_role>(role);
    layout.high_water_bytes = high_water[role];
    layout.max_alignment = max_alignment[role];
    layout.reserved_bytes =
        checked_add(high_water[role], max_alignment[role] - 1, "reserving arena alignment padding");
    layout.canonical_allocations = canonical_counts[role];
    layout.aliases = alias_counts[role];
    roles_.push_back(layout);
    total_high_water_bytes_ = checked_add(total_high_water_bytes_, layout.high_water_bytes,
                                          "summing arena high-water marks");
    total_reserved_bytes_ =
        checked_add(total_reserved_bytes_, layout.reserved_bytes, "summing arena reservations");
  }
}

const allocation_layout &arena_plan::layout(allocation_id id) const {
  const std::vector<allocation_layout>::const_iterator found =
      std::lower_bound(allocations_.begin(), allocations_.end(), id, layout_id_less);
  if (found == allocations_.end() || found->id != id)
    throw std::out_of_range("unknown NVIDIA arena allocation ID");
  return *found;
}

bool arena_plan::contains(allocation_id id) const {
  const std::vector<allocation_layout>::const_iterator found =
      std::lower_bound(allocations_.begin(), allocations_.end(), id, layout_id_less);
  return found != allocations_.end() && found->id == id;
}

struct device_arenas::impl {
  struct role_storage {
    arena_role role;
    size_t base_offset;
    device_buffer buffer;

    explicit role_storage(arena_role role_) : role(role_), base_offset(0), buffer() {}
  };

  arena_plan plan;
  int device;
  arena_accounting accounting;
  std::vector<role_storage> storage;

  impl(const arena_plan &plan_, int device_, size_t reserve_bytes) : plan(plan_), device(device_) {
    device_scope selected(device);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    if (reserve_bytes > free_bytes || plan.total_reserved_bytes() > free_bytes - reserve_bytes) {
      std::ostringstream message;
      message << "NVIDIA arena preflight requires " << plan.total_reserved_bytes() << " bytes plus "
              << reserve_bytes << " reserved bytes, but device " << device << " has only "
              << free_bytes << " bytes free";
      throw std::runtime_error(message.str());
    }

    accounting.device = device;
    accounting.free_bytes_before = free_bytes;
    accounting.total_bytes_on_device = total_bytes;
    accounting.reserve_bytes = reserve_bytes;
    accounting.high_water_bytes = plan.total_high_water_bytes();
    accounting.allocated_bytes = 0;
    accounting.allocation_count = plan.allocations().size();
    accounting.canonical_allocation_count = plan.canonical_allocation_count();
    accounting.alias_count = plan.alias_count();

    storage.reserve(plan.roles().size());
    accounting.roles.reserve(plan.roles().size());
    for (size_t i = 0; i < plan.roles().size(); ++i) {
      const role_layout &role = plan.roles()[i];
      storage.push_back(role_storage(role.role));
      role_storage &owned = storage.back();
      owned.buffer.allocate(role.reserved_bytes, device);
      const uintptr_t raw = reinterpret_cast<uintptr_t>(owned.buffer.opaque_handle());
      const uintptr_t aligned = align_pointer(raw, role.max_alignment);
      owned.base_offset = static_cast<size_t>(aligned - raw);
      if (owned.base_offset > owned.buffer.size() ||
          role.high_water_bytes > owned.buffer.size() - owned.base_offset)
        throw std::logic_error("aligned NVIDIA arena exceeds its backing allocation");

      role_arena_accounting role_accounting;
      role_accounting.role = role.role;
      role_accounting.high_water_bytes = role.high_water_bytes;
      role_accounting.allocated_bytes = owned.buffer.size();
      role_accounting.canonical_allocations = role.canonical_allocations;
      role_accounting.aliases = role.aliases;
      accounting.roles.push_back(role_accounting);
      accounting.allocated_bytes = checked_add(accounting.allocated_bytes, owned.buffer.size(),
                                               "accounting for device arenas");
    }
  }

  role_storage &storage_for(arena_role role) {
    for (size_t i = 0; i < storage.size(); ++i)
      if (storage[i].role == role) return storage[i];
    throw std::logic_error("NVIDIA arena has no storage for requested role");
  }

  const role_storage &storage_for(arena_role role) const {
    for (size_t i = 0; i < storage.size(); ++i)
      if (storage[i].role == role) return storage[i];
    throw std::logic_error("NVIDIA arena has no storage for requested role");
  }

  size_t buffer_offset(const allocation_layout &layout, size_t suboffset) const {
    const role_storage &owned = storage_for(layout.role);
    return checked_add(checked_add(owned.base_offset, layout.offset, "resolving arena offset"),
                       suboffset, "resolving allocation suboffset");
  }
};

device_arenas::device_arenas(const arena_plan &plan, int device, size_t reserve_bytes)
    : impl_(new impl(plan, device, reserve_bytes)) {}

device_arenas::~device_arenas() { delete impl_; }

device_arenas::device_arenas(device_arenas &&other) noexcept : impl_(other.impl_) {
  other.impl_ = NULL;
}

device_arenas &device_arenas::operator=(device_arenas &&other) noexcept {
  if (this == &other) return *this;
  std::swap(impl_, other.impl_);
  return *this;
}

device_allocation device_arenas::resolve(allocation_id id) const {
  if (!impl_) throw std::logic_error("resolve on moved-from NVIDIA arenas");
  const allocation_layout &layout = impl_->plan.layout(id);
  const impl::role_storage &owned = impl_->storage_for(layout.role);
  char *base = static_cast<char *>(owned.buffer.opaque_handle()) + owned.base_offset;
  device_allocation result;
  result.id = layout.id;
  result.canonical_id = layout.canonical_id;
  result.role = layout.role;
  result.address = base + layout.offset;
  result.bytes = layout.bytes;
  result.offset = layout.offset;
  result.alignment = layout.alignment;
  return result;
}

int device_arenas::device() const { return impl_ ? impl_->device : -1; }

const arena_accounting &device_arenas::accounting() const {
  if (!impl_) throw std::logic_error("accounting on moved-from NVIDIA arenas");
  return impl_->accounting;
}

void device_arenas::copy_from_host_async(allocation_id destination, size_t destination_offset,
                                         const void *source, size_t bytes,
                                         const stream &on_stream,
                                         host_to_device_copy_kind kind) {
  if (!impl_) throw std::logic_error("copy on moved-from NVIDIA arenas");
  const allocation_layout &layout = impl_->plan.layout(destination);
  check_subrange(layout.bytes, destination_offset, bytes, "arena host-to-device copy");
  impl::role_storage &owned = impl_->storage_for(layout.role);
  nvidia::copy_host_to_device_async(owned.buffer, impl_->buffer_offset(layout, destination_offset),
                                    source, bytes, on_stream, kind);
}

void device_arenas::copy_to_host_async(void *destination, allocation_id source,
                                       size_t source_offset, size_t bytes,
                                       const stream &on_stream) const {
  if (!impl_) throw std::logic_error("copy on moved-from NVIDIA arenas");
  const allocation_layout &layout = impl_->plan.layout(source);
  check_subrange(layout.bytes, source_offset, bytes, "arena device-to-host copy");
  const impl::role_storage &owned = impl_->storage_for(layout.role);
  nvidia::copy_device_to_host_async(destination, owned.buffer,
                                    impl_->buffer_offset(layout, source_offset), bytes, on_stream);
}

void device_arenas::copy_from_device_async(allocation_id destination, size_t destination_offset,
                                           const device_arenas &source_arenas, allocation_id source,
                                           size_t source_offset, size_t bytes,
                                           const stream &on_stream) {
  if (!impl_ || !source_arenas.impl_) throw std::logic_error("copy on moved-from NVIDIA arenas");
  if (impl_->device != source_arenas.impl_->device)
    throw std::invalid_argument("arena device-to-device copy requires matching devices");
  const allocation_layout &destination_layout = impl_->plan.layout(destination);
  const allocation_layout &source_layout = source_arenas.impl_->plan.layout(source);
  if (destination_layout.element_type != source_layout.element_type ||
      destination_layout.storage_precision != source_layout.storage_precision)
    throw std::invalid_argument("arena device-to-device copy requires compatible storage types");
  check_subrange(destination_layout.bytes, destination_offset, bytes,
                 "arena device-to-device copy destination");
  check_subrange(source_layout.bytes, source_offset, bytes, "arena device-to-device copy source");
  impl::role_storage &destination_storage = impl_->storage_for(destination_layout.role);
  const impl::role_storage &source_storage = source_arenas.impl_->storage_for(source_layout.role);
  nvidia::copy_device_to_device_async(
      destination_storage.buffer, impl_->buffer_offset(destination_layout, destination_offset),
      source_storage.buffer, source_arenas.impl_->buffer_offset(source_layout, source_offset),
      bytes, on_stream);
}

void device_arenas::fill_async(allocation_id destination, size_t destination_offset, int value,
                               size_t bytes, const stream &on_stream) {
  if (!impl_) throw std::logic_error("fill on moved-from NVIDIA arenas");
  const allocation_layout &layout = impl_->plan.layout(destination);
  check_subrange(layout.bytes, destination_offset, bytes, "arena fill");
  impl::role_storage &owned = impl_->storage_for(layout.role);
  nvidia::fill_byte_async(owned.buffer, impl_->buffer_offset(layout, destination_offset), value,
                          bytes, on_stream);
}

} // namespace nvidia
} // namespace meep
