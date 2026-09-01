/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* CUDA-free launch contracts for the resident continuous-wave solver. */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_CW_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_CW_HPP

#include <stddef.h>

#include "backend/nvidia/nvidia_step.hpp"
#include "backend/nvidia/nvidia_sources.hpp"

namespace meep {
namespace nvidia {

struct cw_state_row_launch {
  flat_region region;
  void *real_values;
  void *imaginary_values;
  /* Logical complex-value units. Pack/unpack map one entry to the adjacent
     real scalars [2*i, 2*i+1] in the Krylov vector. */
  size_t complex_offset;
  size_t complex_count;
  scalar_precision precision;
};

struct cw_zero_launch {
  void *values;
  size_t elements;
  scalar_precision precision;
};

struct cw_source_batch_launch {
  void *target_real;
  void *target_imag;
  const void *conductivity_inverse;
  const source_point *points;
  size_t point_offset;
  size_t point_count;
  uint32_t scalar_slot;
  double dt;
  bool sequential;
  scalar_precision precision;
};

struct cw_workspace_shape {
  size_t vector_elements;
  size_t vector_count;
  size_t vector_bytes;
  size_t reduction_partial_bytes;
};

enum class cw_vector_operation {
  copy,
  subtract_field,
  scale_field_coefficient,
  scale_f64_coefficient,
  linear_f64_coefficient
};

struct cw_vector_launch {
  void *output;
  const void *x;
  const void *y;
  size_t elements;
  double coefficient;
  scalar_precision precision;
  cw_vector_operation operation;
};

struct cw_operator_launch {
  void *output;
  const void *stepped;
  const void *input;
  size_t real_elements;
  double dt_inverse;
  double iomega_real;
  double iomega_imaginary;
  scalar_precision precision;
};

struct cw_reduction_launch {
  const void *x;
  const void *y;
  void *partials;
  void *result;
  size_t elements;
  size_t blocks;
  double scale;
  scalar_precision precision;
};

const uint32_t cw_graph_scalars_abi_version = 1;

/* Mutable values consumed by fixed-address CW graph kernels.  The complete
   block is written by one same-stream kernel before each graph replay, so no
   captured node depends on pageable/pinned host staging or stale host values. */
struct cw_graph_scalars {
  uint32_t abi_version;
  uint32_t byte_size;
  void *output;
  const void *x;
  const void *y;
  double coefficient;
  double reduction_scale;
  double dt_inverse;
  double iomega_real;
  double iomega_imaginary;
};

void launch_cw_pack(const cw_state_row_launch &launch, void *vector, size_t real_elements,
                    const stream &stream);
void launch_cw_unpack(const cw_state_row_launch &launch, const void *vector,
                      size_t real_elements, const stream &stream);
void launch_cw_zero(const cw_zero_launch &launch, const stream &stream);
void launch_cw_source_batch(const cw_source_batch_launch &launch, const void *device_scalars,
                            const stream &stream);
void launch_cw_vector(const cw_vector_launch &launch, const stream &stream);
void launch_cw_operator_finalize(const cw_operator_launch &launch, const stream &stream);
void launch_cw_dot(const cw_reduction_launch &launch, const stream &stream);
void launch_cw_max_abs(const cw_reduction_launch &launch, const stream &stream);
void launch_cw_scaled_norm_sum(const cw_reduction_launch &launch, const stream &stream);
void launch_cw_graph_scalars_write(device_buffer &destination,
                                   const cw_graph_scalars &values,
                                   const stream &stream);
void launch_cw_pack_graph(const cw_state_row_launch &launch,
                          const cw_graph_scalars *scalars, size_t real_elements,
                          const stream &stream);
void launch_cw_unpack_graph(const cw_state_row_launch &launch,
                            const cw_graph_scalars *scalars, size_t real_elements,
                            const stream &stream);
void launch_cw_vector_graph(const cw_vector_launch &launch,
                            const cw_graph_scalars *scalars, const stream &stream);
void launch_cw_operator_finalize_graph(const cw_operator_launch &launch,
                                       const cw_graph_scalars *scalars,
                                       const stream &stream);
void launch_cw_dot_graph(const cw_reduction_launch &launch,
                         const cw_graph_scalars *scalars, const stream &stream);
void launch_cw_max_abs_graph(const cw_reduction_launch &launch,
                             const cw_graph_scalars *scalars, const stream &stream);
void launch_cw_scaled_norm_sum_graph(const cw_reduction_launch &launch,
                                     const cw_graph_scalars *scalars,
                                     const stream &stream);

} // namespace nvidia
} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_NVIDIA_CW_HPP
