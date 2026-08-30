/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_MULTILEVEL_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_MULTILEVEL_HPP

#include "backend/nvidia/nvidia_step.hpp"

namespace meep {
namespace nvidia {

struct multilevel_population_term_launch {
  const void *w;
  const void *w_prev;
  const void *p;
  const void *p_prev;
  ptrdiff_t centered_offsets[2];
  uint32_t transition_index;
};

struct multilevel_population_launch {
  flat_region region;
  void *populations;
  const void *gamma_inv;
  const void *gamma_matrix;
  const void *alpha;
  const void *transition_gperpdt;
  const multilevel_population_term_launch *terms;
  void *scratch;
  size_t gamma_byte_offset;
  size_t alpha_byte_offset;
  size_t transition_byte_offset;
  uint32_t term_index;
  uint32_t term_count;
  uint32_t levels;
  uint32_t transitions;
  scalar_precision precision;
};

struct multilevel_transition_launch {
  flat_region region;
  void *p;
  void *p_prev;
  const void *w;
  const void *diagonal_sigma;
  const void *populations;
  const void *coefficients;
  size_t coefficient_byte_offset;
  ptrdiff_t population_offsets[2];
  uint32_t population_stride;
  uint32_t positive_level;
  uint32_t negative_level;
  scalar_precision precision;
};

void launch_multilevel_population(const multilevel_population_launch &update,
                                  const stream &stream);
void launch_multilevel_transition(const multilevel_transition_launch &update,
                                  const stream &stream);

} // namespace nvidia
} // namespace meep

#endif
