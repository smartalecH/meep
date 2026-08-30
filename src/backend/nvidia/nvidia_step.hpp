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
#include <stdint.h>

#include "backend/nvidia/runtime.hpp"

namespace meep {
namespace nvidia {

enum class scalar_precision { f32, f64 };

struct flat_region {
  size_t base;
  size_t counts[3];
  ptrdiff_t strides[3];
};

struct pml_profile_launch {
  const void *sigma;
  const void *kappa;
  const void *inverse;
  ptrdiff_t base;
  ptrdiff_t strides[3];
};

struct curl_launch {
  flat_region region;
  void *target;
  const void *plus_source;
  const void *minus_source;
  ptrdiff_t plus_stride;
  ptrdiff_t minus_stride;
  void *target_u;
  const void *conductivity;
  const void *conductivity_inverse;
  void *target_conductivity;
  pml_profile_launch pml;
  pml_profile_launch pml_u;
  double dtdx;
  double dt;
  scalar_precision precision;
  uint32_t radial_prefix_index;
  uint32_t bfast_update_index;
};

struct beta_launch {
  flat_region region;
  void *target;
  const void *source;
  void *target_u;
  const void *conductivity_inverse;
  void *target_conductivity;
  pml_profile_launch pml;
  pml_profile_launch pml_u;
  double betadt;
  scalar_precision precision;
};

struct constitutive_launch {
  flat_region region;
  void *target;
  const void *primary;
  const void *cross1;
  const void *cross2;
  const void *diagonal;
  const void *offdiagonal1;
  const void *offdiagonal2;
  const void *chi2;
  const void *chi3;
  ptrdiff_t primary_stride;
  ptrdiff_t cross1_stride;
  ptrdiff_t cross2_stride;
  void *target_w;
  void *previous_w;
  const void *previous_w_source;
  size_t previous_w_elements;
  pml_profile_launch pml;
  scalar_precision precision;
};

struct zero_launch {
  flat_region region;
  void *target;
  scalar_precision precision;
};

/* Relocatable same-device halo metadata. The descriptors themselves are
   uploaded once when an executable is compiled. `buffer_index` is measured in
   scalar elements, not bytes. A null imaginary target denotes COPY/NEGATE;
   otherwise the entry applies a complex phase to two adjacent buffer values. */
struct halo_gather_entry {
  const void *source;
  ptrdiff_t source_index;
  size_t buffer_index;
};

struct halo_scatter_entry {
  void *target_real;
  ptrdiff_t target_real_index;
  void *target_imag;
  ptrdiff_t target_imag_index;
  size_t buffer_index;
  double phase_real;
  double phase_imag;
};

struct halo_launch {
  size_t first;
  size_t count;
  scalar_precision precision;
};

struct finite_check_launch {
  const void *values;
  size_t elements;
  uint64_t ordinal_base;
  scalar_precision precision;
};

/* All launches are asynchronous on `stream`. Invalid geometry or a CUDA launch
   failure throws before returning. */
void launch_curl(const curl_launch &update, const stream &stream);
void launch_beta(const beta_launch &update, const stream &stream);
void launch_constitutive(const constitutive_launch &update, const stream &stream);
void launch_zero(const zero_launch &update, const stream &stream);
void launch_halo_gather(const halo_launch &launch, const void *device_entries, void *device_buffer,
                        const stream &stream);
void launch_halo_scatter(const halo_launch &launch, const void *device_entries,
                         const void *device_buffer, const stream &stream);
void launch_finite_check(const finite_check_launch &launch, void *device_first_bad,
                         const stream &stream);

} // namespace nvidia
} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_NVIDIA_STEP_HPP
