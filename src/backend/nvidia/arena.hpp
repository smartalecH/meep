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

/* Private NVIDIA role-arena planning and allocation.
 *
 * This header is deliberately usable by an ordinary C++11 compiler. CUDA
 * resources remain in the implementation and are exposed only as opaque
 * addresses through the runtime layer.
 */

#ifndef MEEP_BACKEND_NVIDIA_ARENA_HPP
#define MEEP_BACKEND_NVIDIA_ARENA_HPP

#include "backend/nvidia/runtime.hpp"

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace meep {
namespace nvidia {

typedef uint64_t allocation_id;
const allocation_id no_allocation = UINT64_MAX;

enum class arena_role { field, material, polarization, dft, communication, scratch, count };

/* Backend-private metadata mirrored by the eventual ArraySpec adapter. Keeping
   these enums independent avoids a Phase-1 header dependency in this layer. */
enum class arena_element_type {
  opaque_bytes,
  realnum_value,
  complex_realnum,
  float64,
  complex_float64,
  int32,
  index,
  count
};
enum class arena_storage_precision { not_applicable, f32, f64, count };

const char *arena_role_name(arena_role role);

/* alias_of == no_allocation denotes an owning request. Aliases have their own
   stable IDs but share the canonical request's address and byte extent. */
struct allocation_request {
  allocation_id id;
  arena_role role;
  size_t bytes;
  size_t alignment;
  allocation_id alias_of;
  arena_element_type element_type;
  arena_storage_precision storage_precision;

  allocation_request(
      allocation_id id_, arena_role role_, size_t bytes_, size_t alignment_,
      allocation_id alias_of_ = no_allocation,
      arena_element_type element_type_ = arena_element_type::opaque_bytes,
      arena_storage_precision storage_precision_ = arena_storage_precision::not_applicable)
      : id(id_), role(role_), bytes(bytes_), alignment(alignment_), alias_of(alias_of_),
        element_type(element_type_), storage_precision(storage_precision_) {}
};

struct allocation_layout {
  allocation_id id;
  arena_role role;
  size_t offset;
  size_t bytes;
  size_t alignment;
  size_t effective_alignment;
  allocation_id canonical_id;
  arena_element_type element_type;
  arena_storage_precision storage_precision;

  bool is_alias() const { return id != canonical_id; }
};

struct role_layout {
  arena_role role;
  size_t high_water_bytes;
  /* Includes worst-case leading padding needed to align a cudaMalloc base. */
  size_t reserved_bytes;
  size_t max_alignment;
  size_t canonical_allocations;
  size_t aliases;
};

/* Planning is deterministic: canonical allocations are laid out by role and
   numeric ID, independent of request order. */
class arena_plan {
public:
  arena_plan();
  explicit arena_plan(const std::vector<allocation_request> &requests);

  const allocation_layout &layout(allocation_id id) const;
  bool contains(allocation_id id) const;
  const std::vector<allocation_layout> &allocations() const { return allocations_; }
  const std::vector<role_layout> &roles() const { return roles_; }

  size_t total_high_water_bytes() const { return total_high_water_bytes_; }
  size_t total_reserved_bytes() const { return total_reserved_bytes_; }
  size_t canonical_allocation_count() const { return canonical_allocation_count_; }
  size_t alias_count() const { return alias_count_; }

private:
  std::vector<allocation_layout> allocations_;
  std::vector<role_layout> roles_;
  size_t total_high_water_bytes_;
  size_t total_reserved_bytes_;
  size_t canonical_allocation_count_;
  size_t alias_count_;
};

struct device_allocation {
  allocation_id id;
  allocation_id canonical_id;
  arena_role role;
  void *address;
  size_t bytes;
  size_t offset;

  bool is_alias() const { return id != canonical_id; }
};

struct role_arena_accounting {
  arena_role role;
  size_t high_water_bytes;
  size_t allocated_bytes;
  size_t canonical_allocations;
  size_t aliases;
};

struct arena_accounting {
  int device;
  size_t free_bytes_before;
  size_t total_bytes_on_device;
  size_t reserve_bytes;
  size_t high_water_bytes;
  size_t allocated_bytes;
  size_t allocation_count;
  size_t canonical_allocation_count;
  size_t alias_count;
  std::vector<role_arena_accounting> roles;
};

/* Allocates exactly one device_buffer for each role with a nonzero high-water
   mark. reserve_bytes leaves caller-selected headroom during the preflight. */
class device_arenas {
public:
  device_arenas(const arena_plan &plan, int device, size_t reserve_bytes = 0);
  ~device_arenas();
  device_arenas(device_arenas &&other) noexcept;
  /* No CUDA teardown occurs during assignment: the PIMPLs are exchanged, so
     the source remains valid and owns the destination's previous arenas. */
  device_arenas &operator=(device_arenas &&other) noexcept;

  device_arenas(const device_arenas &) = delete;
  device_arenas &operator=(const device_arenas &) = delete;

  device_allocation resolve(allocation_id id) const;
  int device() const;
  const arena_accounting &accounting() const;

  void copy_from_host_async(allocation_id destination, size_t destination_offset,
                            const void *source, size_t bytes, const stream &on_stream,
                            host_to_device_copy_kind kind = host_to_device_copy_kind::general);
  void copy_to_host_async(void *destination, allocation_id source, size_t source_offset,
                          size_t bytes, const stream &on_stream) const;
  void copy_from_device_async(allocation_id destination, size_t destination_offset,
                              const device_arenas &source_arenas, allocation_id source,
                              size_t source_offset, size_t bytes, const stream &on_stream);
  void fill_async(allocation_id destination, size_t destination_offset, int value, size_t bytes,
                  const stream &on_stream);

private:
  struct impl;
  impl *impl_;
};

} // namespace nvidia
} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_ARENA_HPP
