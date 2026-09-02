/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_BOUNDARIES_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_BOUNDARIES_HPP

#include <stddef.h>
#include <string>
#include <vector>

#include "backend/nvidia/arena.hpp"
#include "backend/nvidia/nvidia_step.hpp"
#include "backend/transport_plan.hpp"

namespace meep {
namespace nvidia {

/* Remote transport descriptors use byte offsets so one arena can contain
   independently aligned f32 and f64 messages. They are uploaded during
   executable construction; every launch below is allocation-free. */
struct boundary_gather_entry {
  const void *source;
  ptrdiff_t source_index;
  size_t buffer_byte_offset;
};

struct boundary_scatter_entry {
  void *target_real;
  ptrdiff_t target_real_index;
  void *target_imag;
  ptrdiff_t target_imag_index;
  size_t buffer_byte_offset;
  double phase_real;
  double phase_imag;
};

struct boundary_zero_entry {
  void *target;
  ptrdiff_t target_index;
};

struct boundary_launch {
  size_t first;
  size_t count;
  size_t extent;
  scalar_precision precision;
};

struct bound_boundary_message {
  RemoteHaloWireKey key;
  RemoteHaloDirection direction;
  boundary_launch launch;
  size_t wire_bytes;
  size_t slot_offsets[2];
  size_t arena_offsets[2];
};

struct bound_boundary_stage {
  field_type ft;
  std::vector<boundary_gather_entry> gathers;
  std::vector<boundary_scatter_entry> scatters;
  std::vector<bound_boundary_message> receives;
  std::vector<bound_boundary_message> sends;
  size_t receive_slot_bytes;
  size_t send_slot_bytes;
  RemoteHaloPublicationMode publication;
  std::vector<uint32_t> canonical_receive_order;
};

struct bound_boundary_program {
  uint64_t program_signature;
  uint64_t storage_signature;
  uint64_t authority_signature;
  std::vector<bound_boundary_stage> stages;
};

struct bound_boundary_zeroes {
  std::vector<boundary_zero_entry> entries;
  std::vector<boundary_launch> launches;
};

/* Complete allocation identity derived from the owning device_arenas. The
   production binder constructs these records internally; the public shape is
   exposed only so hostile tests can prove every identity field is checked. */
struct boundary_device_allocation {
  ArrayId id;
  ArrayId canonical_id;
  StorageKey key;
  array_role role;
  arena_role arena;
  size_t arena_offset;
  void *address;
  size_t bytes;
  size_t alignment;
  int device;
};

/* Immutable output of the production compile-only PR7.1 boundary path. PR7.2
   adds transport ownership around this artifact; PR7.1 deliberately never
   posts a request or launches it from timestep dispatch. */
struct compiled_boundary_artifact {
  RemoteHaloProgram wire;
  LoweredRemoteHaloProgram lowered;
  std::vector<LoweredHaloZeroDescriptor> lowered_zeroes;
  bound_boundary_program bound;
  bound_boundary_zeroes bound_zeroes;
};

/* Bind neutral canonical roots to already-created device allocations. This is
   compile-time work; launch functions below never allocate or rewrite it. */
bool bind_remote_boundaries(const LoweredRemoteHaloProgram &lowered, const StoragePlan &storage,
                            const device_arenas &device_allocations,
                            bound_boundary_program &bound, std::string &why);
bool bind_boundary_zeroes(const std::vector<LoweredHaloZeroDescriptor> &lowered,
                          const StoragePlan &storage,
                          const device_arenas &device_allocations,
                          bound_boundary_zeroes &bound, std::string &why);

namespace testing {
/* Malformed-authority coverage only. Production binding never accepts caller-
   assembled allocation records. */
bool bind_remote_boundaries_with_allocations(
    const LoweredRemoteHaloProgram &lowered, const StoragePlan &storage,
    const std::vector<boundary_device_allocation> &device_allocations,
    int owning_device, bound_boundary_program &bound, std::string &why);
bool bind_boundary_zeroes_with_allocations(
    const std::vector<LoweredHaloZeroDescriptor> &lowered, const StoragePlan &storage,
    const std::vector<boundary_device_allocation> &device_allocations,
    int owning_device, bound_boundary_zeroes &bound, std::string &why);
} // namespace testing

void launch_boundary_gather(const boundary_launch &launch, const void *device_entries,
                            void *device_arena, const stream &stream);
void launch_boundary_scatter_parallel(const boundary_launch &launch, const void *device_entries,
                                      const void *device_arena, const stream &stream);
void launch_boundary_scatter_serial(const boundary_launch &launch, const void *device_entries,
                                    const void *device_arena, const stream &stream);
void launch_boundary_zero(const boundary_launch &launch, const void *device_entries,
                          const stream &stream);

} // namespace nvidia
} // namespace meep

#endif
