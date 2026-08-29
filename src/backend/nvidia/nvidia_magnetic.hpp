/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_MAGNETIC_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_MAGNETIC_HPP

#include <stddef.h>

#include "backend/nvidia/nvidia_step.hpp"

namespace meep {
namespace nvidia {

struct magnetic_state_launch {
  void *live;
  void *backup;
  size_t elements;
  scalar_precision precision;
};

void launch_magnetic_backup(const magnetic_state_launch &launch, const stream &stream);
void launch_magnetic_restore(const magnetic_state_launch &launch, const stream &stream);
void launch_magnetic_average(const magnetic_state_launch &launch, const stream &stream);

} // namespace nvidia
} // namespace meep

#endif
