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

#ifndef MEEP_BACKEND_NVIDIA_RUNTIME_HPP
#define MEEP_BACKEND_NVIDIA_RUNTIME_HPP

#include <stddef.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace meep {
namespace nvidia {

class runtime_error : public std::runtime_error {
public:
  runtime_error(const std::string &operation, int code, const std::string &name,
                const std::string &detail);

  int code() const { return code_; }
  const std::string &operation() const { return operation_; }
  const std::string &name() const { return name_; }

private:
  int code_;
  std::string operation_;
  std::string name_;
};

struct device_properties {
  int id;
  std::string name;
  std::string uuid;
  int compute_major;
  int compute_minor;
  size_t total_memory;
  int multiprocessors;
  int max_threads_per_block;
  bool unified_addressing;
  bool managed_memory;
};

std::vector<device_properties> enumerate_devices();
device_properties properties_for_device(int device);
int runtime_version();
int driver_version();
bool devices_can_access_peer(int source, int destination);

enum class selection_source { explicit_option, environment, node_local_rank };

struct device_selection {
  int device;
  int visible_device_count;
  int node_local_rank;
  int node_local_size;
  /* True when rank/device cardinality alone proves that sharing is required.
     False is not proof of a collision-free explicit/environment assignment. */
  bool sharing;
  /* select_device only has rank-local inputs. Explicit/environment choices can
     collide even when node_local_size <= visible_device_count, so PR1
     integration must collectively compare the selected device UUIDs before
     creating backend state. */
  bool collective_collision_check_required;
  selection_source source;
};

/* CUDA_VISIBLE_DEVICES is already reflected in CUDA's device numbering.
   Precedence is explicit_device, MEEP_DEVICE_ID, then node-local rank. This is
   a local selection and capacity precheck, not a collective collision check. */
device_selection select_device(int explicit_device, int node_local_rank, int node_local_size,
                               bool allow_sharing = false);
const char *selection_source_name(selection_source source);

class device_scope {
public:
  explicit device_scope(int device);
  ~device_scope();

  device_scope(const device_scope &) = delete;
  device_scope &operator=(const device_scope &) = delete;

  int device() const { return device_; }

private:
  int previous_;
  int device_;
  bool restore_;
};

class stream {
public:
  stream();
  ~stream();
  stream(stream &&other) noexcept;
  stream &operator=(stream &&other) noexcept;

  stream(const stream &) = delete;
  stream &operator=(const stream &) = delete;

  void synchronize() const;
  int device() const;
  void *opaque_handle() const;

private:
  struct impl;
  impl *impl_;
};

class event {
public:
  event();
  ~event();
  event(event &&other) noexcept;
  event &operator=(event &&other) noexcept;

  event(const event &) = delete;
  event &operator=(const event &) = delete;

  void record(const stream &on_stream);
  void wait(const stream &on_stream) const;
  void synchronize() const;
  bool ready() const;
  int device() const;
  void *opaque_handle() const;

private:
  struct impl;
  impl *impl_;
};

class device_buffer {
public:
  device_buffer();
  explicit device_buffer(size_t bytes, int device = -1);
  ~device_buffer();
  device_buffer(device_buffer &&other) noexcept;
  device_buffer &operator=(device_buffer &&other) noexcept;

  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;

  /* reset reports teardown failures and retains ownership when cudaFree did
     not release the allocation. allocate first performs that checked reset;
     if the subsequent allocation fails, the buffer is empty. Destruction is
     best effort. A noexcept move assignment leaves both operands unchanged if
     destination cleanup fails and reports the failure on stderr. */
  void allocate(size_t bytes, int device = -1);
  void reset();
  size_t size() const;
  int device() const;
  void *opaque_handle() const;

private:
  struct impl;
  impl *impl_;
};

class pinned_buffer {
public:
  pinned_buffer();
  explicit pinned_buffer(size_t bytes);
  ~pinned_buffer();
  pinned_buffer(pinned_buffer &&other) noexcept;
  pinned_buffer &operator=(pinned_buffer &&other) noexcept;

  pinned_buffer(const pinned_buffer &) = delete;
  pinned_buffer &operator=(const pinned_buffer &) = delete;

  /* The same checked-reset and noexcept move-assignment rules as
     device_buffer apply. Storage is portable across CUDA devices. */
  void allocate(size_t bytes);
  void reset();
  size_t size() const;
  void *data();
  const void *data() const;

private:
  struct impl;
  impl *impl_;
};

void copy_host_to_device_async(device_buffer &destination, size_t destination_offset,
                               const void *source, size_t bytes, const stream &on_stream);
void copy_device_to_host_async(void *destination, const device_buffer &source,
                               size_t source_offset, size_t bytes, const stream &on_stream);
void copy_device_to_device_async(device_buffer &destination, size_t destination_offset,
                                 const device_buffer &source, size_t source_offset, size_t bytes,
                                 const stream &on_stream);
void fill_byte_async(device_buffer &destination, size_t destination_offset, int value, size_t bytes,
                     const stream &on_stream);

struct memory_accounting {
  size_t device_bytes_current;
  size_t device_bytes_peak;
  size_t pinned_bytes_current;
  size_t pinned_bytes_peak;
};

memory_accounting current_memory_accounting();

/* Failure injection for the standalone runtime tests. These hooks are private
   to the backend prototype and are not part of Meep's public API. */
namespace testing {
struct transfer_accounting {
  size_t host_to_device_calls;
  size_t host_to_device_bytes;
  size_t device_to_host_calls;
  size_t device_to_host_bytes;
  size_t device_to_device_calls;
  size_t device_to_device_bytes;
};

enum class failure_point {
  none,
  device_allocate,
  device_free,
  device_restore,
  pinned_allocate,
  pinned_free
};
void fail_next(failure_point point);
void clear_failure();
transfer_accounting current_transfer_accounting();
void reset_transfer_accounting();
} // namespace testing

} // namespace nvidia
} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_RUNTIME_HPP
