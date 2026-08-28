/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* CUDA-free launch contract for device-resident DFT accumulation. */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_DFT_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_DFT_HPP

#include <stddef.h>

#include "backend/nvidia/nvidia_step.hpp"

namespace meep {
namespace nvidia {

struct dft_launch {
  flat_region region;
  const void *source_real;
  const void *source_imag;
  void *accumulator;   // interleaved complex monitor scalars
  void *phase_scratch; // interleaved complex monitor scalars
  const double *omega;
  size_t omega_offset;
  size_t points;
  size_t frequencies;
  ptrdiff_t avg1;
  ptrdiff_t avg2;
  double start0[3];
  double start1[3];
  double end0[3];
  double end1[3];
  double dV0;
  double dV1;
  double scale_real;
  double scale_imag;
  int decimation_factor;
  bool include_weights;
  bool sqrt_weights;
  bool magnetic;
  scalar_precision field_precision;
  scalar_precision monitor_precision;
};

enum class dft_reduction_operation { norm2, real_weighted_product, complex_weighted_product };

struct dft_reduction_launch {
  const void *left;  // interleaved complex monitor scalars
  const void *right; // null for norm2
  void *partials;    // bounded per-lane block partials
  void *result;      // double2-compatible compact output
  size_t storage_points;
  size_t frequencies;
  size_t base;
  size_t counts[3];
  size_t strides[3];
  size_t result_count;
  size_t blocks_per_lane;
  double weight_real;
  double weight_imag;
  dft_reduction_operation operation;
  scalar_precision monitor_precision;
  scalar_precision accumulation_precision;
};

/* Launches phase generation followed by accumulation on the same stream. */
void launch_dft(const dft_launch &launch, double sample_time, const stream &stream);

/* Two-stage, bounded reduction. Terms are launched sequentially and this adds
   the term's contribution to the existing compact result buffer. */
void launch_dft_reduction(const dft_reduction_launch &launch, const stream &stream);

} // namespace nvidia
} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_NVIDIA_DFT_HPP
