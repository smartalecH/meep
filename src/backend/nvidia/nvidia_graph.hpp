/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#ifndef MEEP_BACKEND_NVIDIA_GRAPH_HPP
#define MEEP_BACKEND_NVIDIA_GRAPH_HPP

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "backend/graph_plan.hpp"
#include "backend/nvidia/runtime.hpp"

namespace meep {
namespace nvidia {

struct graph_capability {
  int runtime;
  int driver;
  bool capture_supported;
  bool update_supported;
};

graph_capability query_graph_capability();

struct graph_accounting {
  size_t creates;
  size_t begin_captures;
  size_t end_captures;
  size_t instantiates;
  size_t updates;
  size_t scalar_writes;
  size_t launches;
  size_t graph_destroys;
  size_t executable_destroys;
};

enum class graph_update_status { success, topology_changed, unsupported, failed };

class graph {
public:
  graph();
  ~graph();
  graph(graph &&other) noexcept;
  graph &operator=(graph &&other) noexcept;

  graph(const graph &) = delete;
  graph &operator=(const graph &) = delete;

  void create(int device, const std::string &label);
  void begin_capture(const stream &on_stream, const std::string &label);
  void end_capture(const stream &on_stream);
  void reset();
  int device() const;
  size_t node_count() const;
  bool capturing() const;
  const std::string &label() const;
  void *opaque_handle() const;

private:
  struct impl;
  impl *impl_;
  friend class graph_exec;
};

class graph_exec {
public:
  graph_exec();
  ~graph_exec();
  graph_exec(graph_exec &&other) noexcept;
  graph_exec &operator=(graph_exec &&other) noexcept;

  graph_exec(const graph_exec &) = delete;
  graph_exec &operator=(const graph_exec &) = delete;

  void instantiate(const graph &definition);
  graph_update_status update(const graph &definition, std::string *diagnostic = NULL);
  void launch(const stream &on_stream) const;
  void reset();
  int device() const;
  size_t node_count() const;
  const std::string &label() const;
  void *opaque_handle() const;

private:
  struct impl;
  impl *impl_;
};

/* Writes the complete fixed-layout block by value on the same stream that is
   subsequently captured/launched.  No pinned staging buffer is involved. */
void launch_step_scalars_write(device_buffer &destination, const StepScalars &values,
                               const stream &on_stream);

namespace testing {
graph_accounting current_graph_accounting();
void reset_graph_accounting();
} // namespace testing

} // namespace nvidia
} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_GRAPH_HPP
