/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_POLARIZATION_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_POLARIZATION_HPP

#include "backend/graph_plan.hpp"
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

enum class gyrotropic_kernel_model : unsigned int { lorentzian, drude, saturated };

struct gyrotropic_update_launch {
  flat_region region;
  void *p[3];
  void *p_prev[3];
  const void *w[3];
  const void *sigma;
  ptrdiff_t primary_stride;
  ptrdiff_t cross_stride1;
  ptrdiff_t cross_stride2;
  double omega0dtsqr;
  double gamma1;
  double diagonal;
  double pt;
  double omega;
  double gamma;
  double alpha;
  double dt2pi;
  double gyro[3][3];
  double inverse[3][3];
  gyrotropic_kernel_model model;
  scalar_precision precision;
};

struct noisy_seed_block {
  uint32_t semantic_seed;
  uint32_t algorithm_version;
};

struct counter_random_words {
  uint32_t lane[4];
};

struct counter_random_input {
  uint32_t semantic_seed;
  uint64_t stream_tag;
  uint64_t point_ordinal;
  uint64_t timestep;
};

void launch_counter_random_samples_for_testing(const counter_random_input *inputs,
                                               counter_random_words *words,
                                               double *uniform_pairs, double *normals,
                                               size_t count, unsigned int threads,
                                               const stream &stream);

struct noisy_add_launch {
  flat_region region;
  void *p;
  const void *diagonal_sigma;
  double amplitude;
  uint64_t stream_tag;
  uint64_t point_ordinal_base;
  scalar_precision precision;
};

struct compiled_polarization_update {
  enum class kind_type : unsigned int { lorentzian, gyrotropic, noisy_add } kind;
  polarization_update_launch lorentzian;
  gyrotropic_update_launch gyrotropic;
  noisy_add_launch noisy;
};

struct polarization_subtract_launch {
  void *target;
  const void *p;
  size_t elements;
  scalar_precision precision;
};

void launch_polarization_update(const polarization_update_launch &update, const stream &stream);
void launch_gyrotropic_update(const gyrotropic_update_launch &update, const stream &stream);
void launch_polarization_update(const compiled_polarization_update &update,
                                const noisy_seed_block *seed, uint64_t timestep,
                                const stream &stream);
void launch_polarization_update_graph(const compiled_polarization_update &update,
                                      const noisy_seed_block *seed_slots,
                                      const StepScalars *scalars, const stream &stream);
void launch_polarization_subtract(const polarization_subtract_launch &update,
                                  const stream &stream);

} // namespace nvidia
} // namespace meep

#endif
