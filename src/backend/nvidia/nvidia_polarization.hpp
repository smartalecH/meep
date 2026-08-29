/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_POLARIZATION_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_POLARIZATION_HPP

#include "backend/nvidia/nvidia_step.hpp"

namespace meep {
namespace nvidia {

struct polarization_update_launch {
  flat_region region;
  void *p;
  void *p_prev;
  const void *primary_w;
  const void *cross_w1;
  const void *cross_w2;
  const void *diagonal_sigma;
  const void *offdiagonal_sigma1;
  const void *offdiagonal_sigma2;
  ptrdiff_t primary_stride;
  ptrdiff_t cross_stride1;
  ptrdiff_t cross_stride2;
  double omega0dtsqr;
  double gamma1inv;
  double gamma1;
  double omega0dtsqr_denom;
  unsigned int offdiagonals;
  bool drude;
  scalar_precision precision;
};

struct polarization_subtract_launch {
  void *target;
  const void *p;
  size_t elements;
  scalar_precision precision;
};

void launch_polarization_update(const polarization_update_launch &update, const stream &stream);
void launch_polarization_subtract(const polarization_subtract_launch &update,
                                  const stream &stream);

} // namespace nvidia
} // namespace meep

#endif
