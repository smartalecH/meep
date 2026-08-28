/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* CUDA-free launch contract for the first eager NVIDIA timestep kernels. */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_STEP_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_STEP_HPP

#include <stddef.h>

#include "backend/nvidia/runtime.hpp"

namespace meep {
namespace nvidia {

enum class scalar_precision { f32, f64 };

struct flat_region {
  size_t base;
  size_t counts[3];
  ptrdiff_t strides[3];
};

struct curl_launch {
  flat_region region;
  void *target;
  const void *plus_source;
  const void *minus_source;
  ptrdiff_t plus_stride;
  ptrdiff_t minus_stride;
  double dtdx;
  scalar_precision precision;
};

struct constitutive_launch {
  flat_region region;
  void *target;
  const void *primary;
  const void *diagonal;
  scalar_precision precision;
};

struct zero_launch {
  flat_region region;
  void *target;
  scalar_precision precision;
};

/* All launches are asynchronous on `stream`. Invalid geometry or a CUDA launch
   failure throws before returning. */
void launch_curl(const curl_launch &update, const stream &stream);
void launch_constitutive(const constitutive_launch &update, const stream &stream);
void launch_zero(const zero_launch &update, const stream &stream);

} // namespace nvidia
} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_NVIDIA_STEP_HPP
