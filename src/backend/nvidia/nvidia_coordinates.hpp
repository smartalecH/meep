/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_COORDINATES_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_COORDINATES_HPP

#include "backend/nvidia/nvidia_step.hpp"

namespace meep {
namespace nvidia {

struct bfast_launch {
  flat_region region;
  void *target;
  const void *source1;
  const void *source2;
  ptrdiff_t stride1;
  ptrdiff_t stride2;
  void *f_bfast;
  void *target_u;
  const void *conductivity_inverse;
  void *target_conductivity;
  pml_profile_launch pml;
  pml_profile_launch pml_u;
  double k1;
  double k2;
  scalar_precision precision;
};

void launch_bfast(const bfast_launch &update, const stream &stream);

} // namespace nvidia
} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_NVIDIA_COORDINATES_HPP
