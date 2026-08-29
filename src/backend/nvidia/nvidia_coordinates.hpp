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

struct cylindrical_radial_prefix_launch {
  const void *source;
  void *scratch;
  size_t nr;
  size_t nz;
  size_t row_stride;
  size_t source_elements;
  size_t scratch_elements;
  double ir0;
  scalar_precision precision;
};

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

struct cylindrical_m_launch {
  flat_region region;
  void *target;
  const void *source;
  void *target_u;
  const void *conductivity_inverse;
  void *target_conductivity;
  pml_profile_launch pml;
  pml_profile_launch pml_u;
  double numerator;
  int raw_radial_start;
  unsigned int radial_axis;
  scalar_precision precision;
};

struct cylindrical_axis_launch {
  flat_region region;
  void *target;
  const void *source1;
  const void *source2;
  ptrdiff_t source1_neighbor_offset;
  ptrdiff_t source2_offset;
  void *target_u;
  const void *conductivity;
  const void *conductivity_inverse;
  void *target_conductivity;
  pml_profile_launch pml;
  pml_profile_launch pml_u;
  double scale;
  double source2_multiplier;
  double dt;
  uint32_t kind;
  scalar_precision precision;
};

void launch_cylindrical_radial_prefix(const cylindrical_radial_prefix_launch &update,
                                      const stream &stream);
void launch_bfast(const bfast_launch &update, const stream &stream);
void launch_cylindrical_m(const cylindrical_m_launch &update, const stream &stream);
void launch_cylindrical_axis(const cylindrical_axis_launch &update, const stream &stream);

} // namespace nvidia
} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_NVIDIA_COORDINATES_HPP
