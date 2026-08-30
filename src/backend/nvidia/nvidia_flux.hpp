/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_FLUX_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_FLUX_HPP

#include <stddef.h>

#include "backend/nvidia/nvidia_step.hpp"

namespace meep {
namespace nvidia {

struct legacy_flux_term_launch {
  flat_region region;
  const void *e_real;
  const void *e_imag;
  const void *h_real;
  const void *h_imag;
  ptrdiff_t e_offsets[2];
  ptrdiff_t h_offsets[2];
  double phase_real;
  double phase_imag;
  double start0[3];
  double start1[3];
  double end0[3];
  double end1[3];
  double dV0;
  double dV1;
  int sign;
  scalar_precision precision;
  size_t points;
  size_t blocks;
};

size_t legacy_flux_partial_count(size_t points);
void launch_legacy_flux_term(const legacy_flux_term_launch &launch, void *partials,
                             void *result, const stream &execution_stream);
void launch_legacy_flux_average(void *current, const void *half, size_t count,
                                const stream &execution_stream);

} // namespace nvidia
} // namespace meep

#endif
