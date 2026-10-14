/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* CUDA-free contract for node-local peer halo transport. Ranks on one node
   map each other's receive arenas and flag words through CUDA IPC; a halo is
   then a gather kernel that stores straight into the peer's arena followed by
   a release store of a per-message generation flag. The receiver's stream
   waits on its local flag words on the device, so no host synchronization or
   MPI call sits on the per-step path. */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_PEER_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_PEER_HPP

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "backend/nvidia/nvidia_step.hpp"
#include "backend/nvidia/runtime.hpp"

namespace meep {
namespace nvidia {

const size_t peer_ipc_handle_bytes = 64;

struct peer_ipc_handle {
  unsigned char bytes[peer_ipc_handle_bytes];
};

/* Export the allocation that starts at buffer's base address. */
peer_ipc_handle export_peer_handle(const device_buffer &buffer);

/* One imported peer allocation; closed on destruction (device selected). */
class peer_mapping {
public:
  peer_mapping();
  ~peer_mapping();
  peer_mapping(peer_mapping &&other) noexcept;
  peer_mapping &operator=(peer_mapping &&other) noexcept;
  peer_mapping(const peer_mapping &) = delete;
  peer_mapping &operator=(const peer_mapping &) = delete;

  void open(const peer_ipc_handle &handle, int device);
  void close();
  void *address() const { return address_; }

private:
  void *address_;
  int device_;
};

/* PCI bus id of a local ordinal, and the local ordinal of a bus id (-1 when
   the device is not visible to this process). Ordinals are process-relative
   under CUDA_VISIBLE_DEVICES, bus ids are not. */
std::string device_pci_bus_id(int device);
int device_for_pci_bus_id(const std::string &bus_id);

/* Spin on the stream until flags[i] >= target for every i < count, reading
   with system-scope acquire. A wait longer than timeout_seconds (0 = never)
   traps, turning a dead peer into a CUDA error instead of a silent hang. */
void launch_peer_wait(const void *flags, size_t count, uint64_t target, double timeout_seconds,
                      const stream &on_stream);

/* *targets[i] = value for i < count with system-scope release stores, so
   every write that happened before on this stream (in particular the gather
   into a peer arena) is visible to a peer that observes the flag. `targets`
   is a device array of (possibly peer-mapped) uint64_t pointers. */
void launch_peer_signal(const void *targets, size_t count, uint64_t value,
                        const stream &on_stream);

/* One stage's sends in one launch: target[0] = source[source_index] with
   `target` an absolute (usually peer-mapped) address. */
struct peer_gather_entry {
  const void *source;
  ptrdiff_t source_index;
  void *target;
};

void launch_peer_gather(const void *entries, size_t count, scalar_precision precision,
                        const stream &on_stream);

/* Device-generation variants: `generation` is a local uint64_t counter for
   one stage, read on the device, so these launches have static parameters
   and can be captured into a CUDA graph. Slot = generation & 1. */
void launch_peer_wait_relative(const void *flags, size_t count, const void *generation,
                               int64_t delta, double timeout_seconds, const stream &on_stream);
/* targets[i] <- generation + delta (release); with `advance`, the counter is
   then incremented. Always one block, so it may run with count == 0. */
void launch_peer_signal_relative(const void *targets, size_t count, const void *generation,
                                 int64_t delta, bool advance, const stream &on_stream);
void launch_peer_gather_slotted(const void *table0, const void *table1, size_t count,
                                const void *generation, scalar_precision precision,
                                const stream &on_stream);
/* boundary_scatter_entry tables with byte offsets relative to `arena`. */
void launch_peer_scatter_slotted(const void *table0, const void *table1, size_t count,
                                 const void *arena, const void *generation,
                                 scalar_precision precision, const stream &on_stream);

/* Host read of local flag words (for draining before teardown). */
void read_peer_flags(const device_buffer &flags, size_t first, size_t count, uint64_t *out);

/* Wait for all work on a device (teardown only: peer stores may still be
   queued on a stream this epoch does not own). */
void synchronize_device(int device);

} // namespace nvidia
} // namespace meep

#endif
