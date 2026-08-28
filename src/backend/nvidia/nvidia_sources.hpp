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

#include "backend/nvidia/nvidia_step.hpp"

namespace meep {
namespace nvidia {

struct source_scalar {
  double current_real;
  double current_imag;
  double dipole_real;
  double dipole_imag;
};

struct point_source_launch {
  void *target_real;
  void *target_imag;
  const void *conductivity_inverse;
  ptrdiff_t index;
  uint32_t scalar_slot;
  double amplitude_real;
  double amplitude_imag;
  double dt;
  bool integrated;
  scalar_precision precision;
};

struct array_copy_launch {
  void *target;
  const void *source;
  size_t elements;
  scalar_precision precision;
};

/* One launch applies one point descriptor. Calling these in descriptor order
   on one stream preserves the legacy ordering when points alias. */
void launch_point_source(const point_source_launch &source, const void *device_scalars,
                         const stream &execution_stream);
void launch_array_copy(const array_copy_launch &copy, const stream &execution_stream);

} // namespace nvidia
} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_NVIDIA_SOURCES_HPP
