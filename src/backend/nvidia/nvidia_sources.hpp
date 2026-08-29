/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* CUDA-free launch contract for the first NVIDIA point-source slice. */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_SOURCES_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_SOURCES_HPP

#include <stddef.h>
#include <stdint.h>

#include <set>

#include "backend/nvidia/nvidia_step.hpp"

namespace meep {
namespace nvidia {

struct source_scalar {
  double current_real;
  double current_imag;
  double dipole_real;
  double dipole_imag;
};

struct source_point {
  ptrdiff_t index;
  double amplitude_real;
  double amplitude_imag;
};

struct source_batch_launch {
  void *target_real;
  void *target_imag;
  const void *conductivity_inverse;
  const source_point *points;
  size_t point_offset;
  size_t point_count;
  uint32_t scalar_slot;
  double dt;
  bool integrated;
  bool sequential;
  scalar_precision precision;
};

struct array_copy_launch {
  void *target;
  const void *source;
  size_t elements;
  scalar_precision precision;
};

/* This is part of descriptor compilation, not launch-time policy. Keeping the
   classifier CUDA-free makes the exact lowering decision directly testable. */
inline bool source_indices_require_sequential(const ptrdiff_t *indices, size_t count) {
  std::set<ptrdiff_t> unique_indices;
  for (size_t i = 0; i < count; ++i)
    if (!unique_indices.insert(indices[i]).second) return true;
  return false;
}

/* One launch applies one spatial descriptor. Descriptors with unique indices
   use a grid-stride kernel. A descriptor containing duplicate indices uses a
   one-thread ordered fallback so its floating-point association is unchanged.
   Calling descriptors in order on one stream preserves inter-descriptor order. */
void launch_source_batch(const source_batch_launch &source, const void *device_scalars,
                         const stream &execution_stream);
void launch_array_copy(const array_copy_launch &copy, const stream &execution_stream);

} // namespace nvidia
} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_NVIDIA_SOURCES_HPP
