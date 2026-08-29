/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_step.hpp"
#include "backend/nvidia/nvidia_coordinates.hpp"
#include "backend/nvidia/nvidia_sources.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using meep::nvidia::constitutive_launch;
using meep::nvidia::array_copy_launch;
using meep::nvidia::beta_launch;
using meep::nvidia::bfast_launch;
using meep::nvidia::copy_device_to_host_async;
using meep::nvidia::copy_host_to_device_async;
using meep::nvidia::curl_launch;
using meep::nvidia::device_buffer;
using meep::nvidia::device_properties;
using meep::nvidia::device_scope;
using meep::nvidia::enumerate_devices;
using meep::nvidia::fill_byte_async;
using meep::nvidia::finite_check_launch;
using meep::nvidia::flat_region;
using meep::nvidia::halo_gather_entry;
using meep::nvidia::halo_launch;
using meep::nvidia::halo_scatter_entry;
using meep::nvidia::launch_constitutive;
using meep::nvidia::launch_array_copy;
using meep::nvidia::launch_beta;
using meep::nvidia::launch_bfast;
using meep::nvidia::launch_curl;
using meep::nvidia::launch_finite_check;
using meep::nvidia::launch_halo_gather;
using meep::nvidia::launch_halo_scatter;
using meep::nvidia::launch_source_batch;
using meep::nvidia::launch_zero;
using meep::nvidia::scalar_precision;
using meep::nvidia::source_batch_launch;
using meep::nvidia::source_indices_require_sequential;
using meep::nvidia::source_point;
using meep::nvidia::source_scalar;
using meep::nvidia::stream;
using meep::nvidia::zero_launch;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename T>
static T nonlinear_reference(T dsqr, T di, T chi1inv, T chi2, T chi3) {
  const T c2 = di * chi2 * (chi1inv * chi1inv);
  const T c3 = dsqr * chi3 * (chi1inv * chi1inv * chi1inv);
  return (T(1) + c2 + T(2) * c3) / (T(1) + T(2) * c2 + T(3) * c3);
}

template <typename T>
static void check_beta_variants(int device, stream &execution, scalar_precision precision) {
  const size_t elements = 257;
  const size_t bytes = elements * sizeof(T);
  std::vector<T> initial(elements), source(elements), initial_u(elements), initial_cond(elements),
      condinv(elements), inverse(elements), inverse_u(elements), observed(elements),
      observed_u(elements), observed_cond(elements);
  for (size_t i = 0; i < elements; ++i) {
    initial[i] = T(0.2 + 0.001 * double(i));
    source[i] = T(std::sin(0.031 * double(i + 1)));
    initial_u[i] = T(-0.1 + 0.0007 * double(i));
    initial_cond[i] = T(0.05 - 0.0002 * double(i));
    condinv[i] = T(1.0 / (1.0 + 0.0003 * double(i + 1)));
    inverse[i] = T(1.0 / (1.0 + 0.0004 * double(i + 1)));
    inverse_u[i] = T(1.0 / (1.0 + 0.0005 * double(i + 1)));
  }

  device_buffer d_target(bytes, device), d_source(bytes, device), d_u(bytes, device),
      d_cond(bytes, device), d_condinv(bytes, device), d_inverse(bytes, device),
      d_inverse_u(bytes, device);
  copy_host_to_device_async(d_source, 0, source.data(), bytes, execution);
  copy_host_to_device_async(d_condinv, 0, condinv.data(), bytes, execution);
  copy_host_to_device_async(d_inverse, 0, inverse.data(), bytes, execution);
  copy_host_to_device_async(d_inverse_u, 0, inverse_u.data(), bytes, execution);

  for (unsigned int variant = 0; variant < 8; ++variant)
    for (int sign = -1; sign <= 1; sign += 2) {
      const bool main_pml = (variant & 1) != 0;
      const bool auxiliary_pml = (variant & 2) != 0;
      const bool conductive = (variant & 4) != 0;
      copy_host_to_device_async(d_target, 0, initial.data(), bytes, execution);
      copy_host_to_device_async(d_u, 0, initial_u.data(), bytes, execution);
      copy_host_to_device_async(d_cond, 0, initial_cond.data(), bytes, execution);

      beta_launch beta = {};
      beta.region.base = 0;
      beta.region.counts[0] = 1;
      beta.region.counts[1] = 1;
      beta.region.counts[2] = elements;
      beta.region.strides[2] = 1;
      beta.target = d_target.opaque_handle();
      beta.source = d_source.opaque_handle();
      beta.betadt = sign * 0.0375;
      beta.precision = precision;
      if (main_pml) {
        beta.pml.inverse = d_inverse.opaque_handle();
        beta.pml.strides[2] = 1;
      }
      if (auxiliary_pml) {
        beta.target_u = d_u.opaque_handle();
        beta.pml_u.inverse = d_inverse_u.opaque_handle();
        beta.pml_u.strides[2] = 1;
      }
      if (conductive) beta.conductivity_inverse = d_condinv.opaque_handle();
      if (main_pml && conductive) beta.target_conductivity = d_cond.opaque_handle();
      launch_beta(beta, execution);

      copy_device_to_host_async(observed.data(), d_target, 0, bytes, execution);
      copy_device_to_host_async(observed_u.data(), d_u, 0, bytes, execution);
      copy_device_to_host_async(observed_cond.data(), d_cond, 0, bytes, execution);
      execution.synchronize();

      std::vector<T> expected = initial;
      std::vector<T> expected_u = initial_u;
      std::vector<T> expected_cond = initial_cond;
      for (size_t i = 0; i < elements; ++i) {
        T delta = T(beta.betadt) * source[i];
        if (conductive) delta *= condinv[i];
        if (main_pml) {
          if (conductive) expected_cond[i] += delta;
          delta *= inverse[i];
        }
        if (auxiliary_pml) {
          expected_u[i] += delta;
          delta *= inverse_u[i];
        }
        expected[i] += delta;
      }
      const double tolerance = 16.0 * std::numeric_limits<T>::epsilon();
      for (size_t i = 0; i < elements; ++i) {
        require(std::fabs(double(observed[i] - expected[i])) <=
                    tolerance * (1.0 + std::fabs(double(expected[i]))),
                "beta target differs from host recurrence");
        require(std::fabs(double(observed_u[i] - expected_u[i])) <=
                    tolerance * (1.0 + std::fabs(double(expected_u[i]))),
                "beta auxiliary target differs");
        require(std::fabs(double(observed_cond[i] - expected_cond[i])) <=
                    tolerance * (1.0 + std::fabs(double(expected_cond[i]))),
                "beta conductivity target differs");
      }
    }

  beta_launch malformed = {};
  malformed.region.counts[0] = malformed.region.counts[1] = malformed.region.counts[2] = 1;
  malformed.target = d_target.opaque_handle();
  malformed.source = d_source.opaque_handle();
  malformed.precision = precision;
  bool rejected = false;
  try {
    beta_launch missing = malformed;
    missing.source = NULL;
    launch_beta(missing, execution);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "beta launch accepted a missing source");
  rejected = false;
  try {
    beta_launch aliased = malformed;
    aliased.source = aliased.target;
    launch_beta(aliased, execution);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "beta launch accepted aliased target/source state");
  rejected = false;
  try {
    beta_launch empty = malformed;
    empty.region.counts[1] = 0;
    launch_beta(empty, execution);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "beta launch accepted an empty region");
  rejected = false;
  try {
    beta_launch overflow = malformed;
    overflow.region.counts[0] = std::numeric_limits<size_t>::max();
    overflow.region.counts[1] = 2;
    launch_beta(overflow, execution);
  }
  catch (const std::overflow_error &) { rejected = true; }
  require(rejected, "beta launch accepted an overflowing region");
  rejected = false;
  try {
    beta_launch invalid_precision = malformed;
    invalid_precision.precision = static_cast<scalar_precision>(99);
    launch_beta(invalid_precision, execution);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "beta launch accepted an invalid precision");
}

template <typename T>
static void check_bfast_variants(int device, stream &execution, scalar_precision precision) {
  const size_t elements = 273;
  const size_t bytes = elements * sizeof(T);
  std::vector<T> initial(elements), source1(elements), source2(elements), initial_state(elements),
      initial_u(elements), initial_cond(elements), condinv(elements), inverse(elements),
      inverse_u(elements), observed(elements), observed_state(elements), observed_u(elements),
      observed_cond(elements);
  for (size_t i = 0; i < elements; ++i) {
    initial[i] = T(0.2 + 0.001 * double(i));
    source1[i] = T(std::sin(0.019 * double(i + 1)));
    source2[i] = T(std::cos(0.023 * double(i + 1)));
    initial_state[i] = T(-0.07 + 0.0003 * double(i));
    initial_u[i] = T(-0.1 + 0.0007 * double(i));
    initial_cond[i] = T(0.05 - 0.0002 * double(i));
    condinv[i] = T(1.0 / (1.0 + 0.0003 * double(i + 1)));
    inverse[i] = T(1.0 / (1.0 + 0.0004 * double(i + 1)));
    inverse_u[i] = T(1.0 / (1.0 + 0.0005 * double(i + 1)));
  }

  device_buffer d_target(bytes, device), d_source1(bytes, device), d_source2(bytes, device),
      d_state(bytes, device), d_u(bytes, device), d_cond(bytes, device), d_condinv(bytes, device),
      d_inverse(bytes, device), d_inverse_u(bytes, device);
  copy_host_to_device_async(d_source1, 0, source1.data(), bytes, execution);
  copy_host_to_device_async(d_source2, 0, source2.data(), bytes, execution);
  copy_host_to_device_async(d_condinv, 0, condinv.data(), bytes, execution);
  copy_host_to_device_async(d_inverse, 0, inverse.data(), bytes, execution);
  copy_host_to_device_async(d_inverse_u, 0, inverse_u.data(), bytes, execution);

  bool exercised_exceptional_branch = false, exercised_source_swap = false,
       exercised_zero_k = false;
  for (unsigned int variant = 0; variant < 8; ++variant)
    for (int source_shape = 0; source_shape < 3; ++source_shape) {
      const bool main_pml = (variant & 1) != 0;
      const bool auxiliary_pml = (variant & 2) != 0;
      const bool conductive = (variant & 4) != 0;
      copy_host_to_device_async(d_target, 0, initial.data(), bytes, execution);
      copy_host_to_device_async(d_state, 0, initial_state.data(), bytes, execution);
      copy_host_to_device_async(d_u, 0, initial_u.data(), bytes, execution);
      copy_host_to_device_async(d_cond, 0, initial_cond.data(), bytes, execution);

      bfast_launch update = {};
      update.region.base = 8;
      update.region.counts[0] = 1;
      update.region.counts[1] = 1;
      update.region.counts[2] = 257;
      update.region.strides[2] = 1;
      update.target = d_target.opaque_handle();
      update.source1 = source_shape == 2 ? NULL : d_source1.opaque_handle();
      update.source2 = source_shape == 1 ? NULL : d_source2.opaque_handle();
      update.stride1 = 2;
      update.stride2 = -3;
      update.f_bfast = d_state.opaque_handle();
      update.k1 = (variant & 1) ? -0.0375 : 0.0375;
      update.k2 = (source_shape == 2 && variant == 0) ? 0.0 : -0.02125;
      update.precision = precision;
      if (main_pml) {
        update.pml.inverse = d_inverse.opaque_handle();
        update.pml.base = 0;
        update.pml.strides[2] = 1;
      }
      if (auxiliary_pml) {
        update.target_u = d_u.opaque_handle();
        update.pml_u.inverse = d_inverse_u.opaque_handle();
        update.pml_u.base = 0;
        update.pml_u.strides[2] = 1;
      }
      if (conductive) update.conductivity_inverse = d_condinv.opaque_handle();
      if (main_pml && conductive) update.target_conductivity = d_cond.opaque_handle();

      std::vector<T> expected = initial;
      std::vector<T> expected_state = initial_state;
      std::vector<T> expected_u = initial_u;
      std::vector<T> expected_cond = initial_cond;
      for (int repeat = 0; repeat < 2; ++repeat) {
        launch_bfast(update, execution);
        for (size_t n = 0; n < update.region.counts[2]; ++n) {
          const ptrdiff_t i = ptrdiff_t(update.region.base + n);
          const T *g1 = source_shape == 2 ? NULL : source1.data();
          const T *g2 = source_shape == 1 ? NULL : source2.data();
          ptrdiff_t s1 = update.stride1, s2 = update.stride2;
          T k1 = T(update.k1), k2 = T(update.k2);
          if (!g1) {
            std::swap(g1, g2);
            std::swap(s1, s2);
            std::swap(k1, k2);
          }
          const T previous = expected_state[i];
          T next;
          if (g2)
            next = (k1 * (g1[i + s1] + g1[i]) - k2 * (g2[i + s2] + g2[i])) - previous;
          else if (!main_pml && !auxiliary_pml && !conductive)
            next = k1 * (g1[i + s1] + g1[i]);
          else
            next = k1 * (g1[i + s1] + g1[i]) - previous;
          expected_state[i] = next;
          T delta = next - previous;
          if (conductive) delta *= condinv[i];
          if (main_pml) {
            if (conductive) expected_cond[i] += delta;
            delta *= inverse[n];
          }
          if (auxiliary_pml) {
            expected_u[i] += delta;
            delta *= inverse_u[n];
          }
          expected[i] += delta;
        }
      }
      exercised_exceptional_branch |= variant == 0 && source_shape == 1;
      exercised_source_swap |= source_shape == 2;
      exercised_zero_k |= source_shape == 2 && variant == 0;

      copy_device_to_host_async(observed.data(), d_target, 0, bytes, execution);
      copy_device_to_host_async(observed_state.data(), d_state, 0, bytes, execution);
      copy_device_to_host_async(observed_u.data(), d_u, 0, bytes, execution);
      copy_device_to_host_async(observed_cond.data(), d_cond, 0, bytes, execution);
      execution.synchronize();
      for (size_t i = 0; i < elements; ++i) {
        require(observed[i] == expected[i],
                "BFAST target differs from host recurrence or sentinel");
        require(observed_state[i] == expected_state[i],
                "BFAST persistent state differs from host recurrence or sentinel");
        require(observed_u[i] == expected_u[i],
                "BFAST auxiliary state differs from host recurrence or sentinel");
        require(observed_cond[i] == expected_cond[i],
                "BFAST conductivity state differs from host recurrence or sentinel");
      }
    }
  require(exercised_exceptional_branch, "BFAST exceptional one-source branch was not covered");
  require(exercised_source_swap, "BFAST missing-first-source swap was not covered");
  require(exercised_zero_k, "BFAST zero-k one-source row was not covered");

  bfast_launch malformed = {};
  malformed.region.counts[0] = malformed.region.counts[1] = malformed.region.counts[2] = 1;
  malformed.target = d_target.opaque_handle();
  malformed.source1 = d_source1.opaque_handle();
  malformed.f_bfast = d_state.opaque_handle();
  malformed.precision = precision;
  bool rejected = false;
  try {
    bfast_launch missing = malformed;
    missing.f_bfast = NULL;
    launch_bfast(missing, execution);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "BFAST launch accepted missing persistent state");
  rejected = false;
  try {
    bfast_launch missing = malformed;
    missing.source1 = NULL;
    launch_bfast(missing, execution);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "BFAST launch accepted missing sources");
  rejected = false;
  try {
    bfast_launch aliased = malformed;
    aliased.f_bfast = aliased.target;
    launch_bfast(aliased, execution);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "BFAST launch accepted aliased mutable state");
  rejected = false;
  try {
    bfast_launch empty = malformed;
    empty.region.counts[1] = 0;
    launch_bfast(empty, execution);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "BFAST launch accepted an empty region");
  rejected = false;
  try {
    bfast_launch overflow = malformed;
    overflow.region.counts[0] = std::numeric_limits<size_t>::max();
    overflow.region.counts[1] = 2;
    launch_bfast(overflow, execution);
  }
  catch (const std::overflow_error &) { rejected = true; }
  require(rejected, "BFAST launch accepted an overflowing region");
  rejected = false;
  try {
    bfast_launch grid_overflow = malformed;
    grid_overflow.region.counts[0] = std::numeric_limits<size_t>::max();
    launch_bfast(grid_overflow, execution);
  }
  catch (const std::overflow_error &) { rejected = true; }
  require(rejected, "BFAST launch accepted a wrapping grid size");
  rejected = false;
  try {
    bfast_launch invalid_precision = malformed;
    invalid_precision.precision = static_cast<scalar_precision>(99);
    launch_bfast(invalid_precision, execution);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "BFAST launch accepted an invalid precision");
}

template <typename T> static void check_device(int device) {
  device_scope selected(device);
  stream execution;
  const size_t elements = 96;
  const size_t bytes = elements * sizeof(T);
  std::vector<T> target(elements), plus(elements), minus(elements), primary(elements),
      diagonal(elements), offdiagonal1(elements), offdiagonal2(elements), chi2(elements),
      chi3(elements),
      output(elements, T(-17.25)), observed(elements);
  for (size_t i = 0; i < elements; ++i) {
    target[i] = T(0.01 * double(i + 1));
    plus[i] = T(std::sin(0.13 * double(i)));
    minus[i] = T(std::cos(0.07 * double(i)));
    primary[i] = T(0.25 + 0.005 * double(i));
    diagonal[i] = T(0.5 + 0.002 * double(i));
    offdiagonal1[i] = T(0.03 - 0.0001 * double(i));
    offdiagonal2[i] = T(-0.02 + 0.00015 * double(i));
    chi2[i] = T(0.07 + 0.0002 * double(i));
    chi3[i] = T(0.11 - 0.0003 * double(i));
  }

  device_buffer d_target(bytes, device), d_plus(bytes, device), d_minus(bytes, device),
      d_primary(bytes, device), d_diagonal(bytes, device), d_offdiagonal1(bytes, device),
      d_offdiagonal2(bytes, device), d_chi2(bytes, device), d_chi3(bytes, device),
      d_output(bytes, device);
  copy_host_to_device_async(d_target, 0, target.data(), bytes, execution);
  copy_host_to_device_async(d_plus, 0, plus.data(), bytes, execution);
  copy_host_to_device_async(d_minus, 0, minus.data(), bytes, execution);
  copy_host_to_device_async(d_primary, 0, primary.data(), bytes, execution);
  copy_host_to_device_async(d_diagonal, 0, diagonal.data(), bytes, execution);
  copy_host_to_device_async(d_offdiagonal1, 0, offdiagonal1.data(), bytes, execution);
  copy_host_to_device_async(d_offdiagonal2, 0, offdiagonal2.data(), bytes, execution);
  copy_host_to_device_async(d_chi2, 0, chi2.data(), bytes, execution);
  copy_host_to_device_async(d_chi3, 0, chi3.data(), bytes, execution);
  copy_host_to_device_async(d_output, 0, output.data(), bytes, execution);

  flat_region region;
  region.base = 9;
  region.counts[0] = 2;
  region.counts[1] = 3;
  region.counts[2] = 7;
  region.strides[0] = 32;
  region.strides[1] = 9;
  region.strides[2] = 1;
  const scalar_precision precision =
      sizeof(T) == sizeof(float) ? scalar_precision::f32 : scalar_precision::f64;
  check_bfast_variants<T>(device, execution, precision);
  if (std::getenv("MEEP_NVIDIA_STEP_BFAST_ONLY")) return;
  check_beta_variants<T>(device, execution, precision);
  if (std::getenv("MEEP_NVIDIA_STEP_BETA_ONLY")) return;

  curl_launch curl = {};
  curl.region = region;
  curl.target = d_target.opaque_handle();
  curl.plus_source = d_plus.opaque_handle();
  curl.minus_source = d_minus.opaque_handle();
  curl.plus_stride = 2;
  curl.minus_stride = -3;
  curl.dtdx = 0.125;
  curl.precision = precision;
  launch_curl(curl, execution);
  copy_device_to_host_async(observed.data(), d_target, 0, bytes, execution);
  execution.synchronize();

  std::vector<T> expected = target;
  for (size_t i0 = 0; i0 < region.counts[0]; ++i0)
    for (size_t i1 = 0; i1 < region.counts[1]; ++i1)
      for (size_t i2 = 0; i2 < region.counts[2]; ++i2) {
        const ptrdiff_t i = ptrdiff_t(region.base) + ptrdiff_t(i0) * region.strides[0] +
                            ptrdiff_t(i1) * region.strides[1] + ptrdiff_t(i2) * region.strides[2];
        expected[i] -= T(curl.dtdx) * (plus[i + curl.plus_stride] - plus[i] + minus[i] -
                                       minus[i + curl.minus_stride]);
      }
  for (size_t i = 0; i < elements; ++i)
    require(std::fabs(double(observed[i] - expected[i])) <=
                4.0 * std::numeric_limits<T>::epsilon() * (1.0 + std::fabs(double(expected[i]))),
            "curl result differs from host reference");

  constitutive_launch constitutive = {};
  constitutive.region = region;
  constitutive.target = d_output.opaque_handle();
  constitutive.primary = d_primary.opaque_handle();
  constitutive.diagonal = d_diagonal.opaque_handle();
  constitutive.precision = precision;
  launch_constitutive(constitutive, execution);
  copy_device_to_host_async(observed.data(), d_output, 0, bytes, execution);
  execution.synchronize();
  std::vector<T> expected_output = output;
  for (size_t i0 = 0; i0 < region.counts[0]; ++i0)
    for (size_t i1 = 0; i1 < region.counts[1]; ++i1)
      for (size_t i2 = 0; i2 < region.counts[2]; ++i2) {
        const ptrdiff_t i = ptrdiff_t(region.base) + ptrdiff_t(i0) * region.strides[0] +
                            ptrdiff_t(i1) * region.strides[1] + ptrdiff_t(i2) * region.strides[2];
        expected_output[i] = primary[i] * diagonal[i];
      }
  for (size_t i = 0; i < elements; ++i)
    require(std::fabs(double(observed[i] - expected_output[i])) <=
                2.0 * std::numeric_limits<T>::epsilon() *
                    (1.0 + std::fabs(double(expected_output[i]))),
            "constitutive result or out-of-region sentinel differs");

  zero_launch zero = {};
  zero.region = region;
  zero.target = d_target.opaque_handle();
  zero.precision = precision;
  launch_zero(zero, execution);
  copy_device_to_host_async(observed.data(), d_target, 0, bytes, execution);
  execution.synchronize();
  std::vector<T> expected_zero = expected;
  for (size_t i0 = 0; i0 < region.counts[0]; ++i0)
    for (size_t i1 = 0; i1 < region.counts[1]; ++i1)
      for (size_t i2 = 0; i2 < region.counts[2]; ++i2) {
        const ptrdiff_t i = ptrdiff_t(region.base) + ptrdiff_t(i0) * region.strides[0] +
                            ptrdiff_t(i1) * region.strides[1] + ptrdiff_t(i2) * region.strides[2];
        expected_zero[i] = T(0);
      }
  for (size_t i = 0; i < elements; ++i)
    require(observed[i] == expected_zero[i], "zero result or out-of-region sentinel differs");

  const double pml_dt = 0.075;
  flat_region pml_region = {};
  pml_region.base = 2;
  pml_region.counts[0] = 1;
  pml_region.counts[1] = 1;
  pml_region.counts[2] = 6;
  pml_region.strides[0] = 0;
  pml_region.strides[1] = 0;
  pml_region.strides[2] = 1;
  const size_t profile_elements = 20;
  std::vector<T> pml_target(elements), pml_u(elements), pml_cond_target(elements),
      conductivity(elements), conductivity_inverse(elements), sigma(profile_elements),
      kappa(profile_elements), inverse(profile_elements), sigma_u(profile_elements),
      kappa_u(profile_elements), inverse_u(profile_elements);
  for (size_t i = 0; i < elements; ++i) {
    pml_target[i] = T(0.4 + 0.003 * double(i));
    pml_u[i] = T(-0.2 + 0.002 * double(i));
    pml_cond_target[i] = T(0.1 - 0.001 * double(i));
    conductivity[i] = T(0.03 + 0.0004 * double(i));
    conductivity_inverse[i] = T(1) / (T(1) + T(0.5 * pml_dt) * conductivity[i]);
  }
  for (size_t i = 0; i < profile_elements; ++i) {
    sigma[i] = T(0.004 * double(i + 1));
    kappa[i] = T(1.0 + 0.003 * double(i));
    inverse[i] = T(1) / (kappa[i] + sigma[i]);
    sigma_u[i] = T(0.003 * double(i + 1));
    kappa_u[i] = T(1.0 + 0.002 * double(i));
    inverse_u[i] = T(1) / (kappa_u[i] + sigma_u[i]);
  }

  device_buffer d_pml_target(bytes, device), d_pml_u(bytes, device),
      d_pml_cond_target(bytes, device), d_conductivity(bytes, device),
      d_conductivity_inverse(bytes, device), d_sigma(profile_elements * sizeof(T), device),
      d_kappa(profile_elements * sizeof(T), device), d_inverse(profile_elements * sizeof(T), device),
      d_sigma_u(profile_elements * sizeof(T), device),
      d_kappa_u(profile_elements * sizeof(T), device),
      d_inverse_u(profile_elements * sizeof(T), device);
  copy_host_to_device_async(d_conductivity, 0, conductivity.data(), bytes, execution);
  copy_host_to_device_async(d_conductivity_inverse, 0, conductivity_inverse.data(), bytes,
                            execution);
  copy_host_to_device_async(d_sigma, 0, sigma.data(), profile_elements * sizeof(T), execution);
  copy_host_to_device_async(d_kappa, 0, kappa.data(), profile_elements * sizeof(T), execution);
  copy_host_to_device_async(d_inverse, 0, inverse.data(), profile_elements * sizeof(T), execution);
  copy_host_to_device_async(d_sigma_u, 0, sigma_u.data(), profile_elements * sizeof(T), execution);
  copy_host_to_device_async(d_kappa_u, 0, kappa_u.data(), profile_elements * sizeof(T), execution);
  copy_host_to_device_async(d_inverse_u, 0, inverse_u.data(), profile_elements * sizeof(T),
                            execution);

  for (unsigned int variant = 0; variant < 8; ++variant) {
    const bool main_pml = (variant & 1) != 0;
    const bool auxiliary_pml = (variant & 2) != 0;
    const bool conductive = (variant & 4) != 0;
    copy_host_to_device_async(d_pml_target, 0, pml_target.data(), bytes, execution);
    copy_host_to_device_async(d_pml_u, 0, pml_u.data(), bytes, execution);
    copy_host_to_device_async(d_pml_cond_target, 0, pml_cond_target.data(), bytes, execution);

    curl_launch pml_curl = {};
    pml_curl.region = pml_region;
    pml_curl.target = d_pml_target.opaque_handle();
    pml_curl.plus_source = d_plus.opaque_handle();
    pml_curl.minus_source = d_minus.opaque_handle();
    pml_curl.plus_stride = 1;
    pml_curl.minus_stride = 1;
    pml_curl.dtdx = 0.125;
    pml_curl.dt = pml_dt;
    pml_curl.precision = precision;
    if (main_pml) {
      pml_curl.pml.sigma = d_sigma.opaque_handle();
      pml_curl.pml.kappa = d_kappa.opaque_handle();
      pml_curl.pml.inverse = d_inverse.opaque_handle();
      pml_curl.pml.base = 3;
      pml_curl.pml.strides[2] = 2;
    }
    if (auxiliary_pml) {
      pml_curl.target_u = d_pml_u.opaque_handle();
      pml_curl.pml_u.sigma = d_sigma_u.opaque_handle();
      pml_curl.pml_u.kappa = d_kappa_u.opaque_handle();
      pml_curl.pml_u.inverse = d_inverse_u.opaque_handle();
      pml_curl.pml_u.base = 2;
      pml_curl.pml_u.strides[2] = 2;
    }
    if (conductive) {
      pml_curl.conductivity = d_conductivity.opaque_handle();
      pml_curl.conductivity_inverse = d_conductivity_inverse.opaque_handle();
      if (main_pml) pml_curl.target_conductivity = d_pml_cond_target.opaque_handle();
    }
    launch_curl(pml_curl, execution);

    std::vector<T> expected_target = pml_target;
    std::vector<T> expected_u = pml_u;
    std::vector<T> expected_cond_target = pml_cond_target;
    for (size_t n = 0; n < pml_region.counts[2]; ++n) {
      const size_t i = pml_region.base + n;
      const size_t k = 3 + 2 * n;
      const size_t ku = 2 + 2 * n;
      const T curl_value = plus[i + 1] - plus[i] + minus[i] - minus[i + 1];
      const T delta = T(pml_curl.dtdx) * curl_value;
      const T damping = T(1) - T(0.5 * pml_dt) * conductivity[i];
      if (!main_pml && !auxiliary_pml) {
        expected_target[i] = conductive
                                 ? (damping * expected_target[i] - delta) *
                                       conductivity_inverse[i]
                                 : expected_target[i] - delta;
        continue;
      }
      T intermediate;
      T previous_u = T(0);
      if (main_pml) {
        if (conductive) {
          const T previous = expected_cond_target[i];
          expected_cond_target[i] =
              (damping * previous - delta) * conductivity_inverse[i];
          intermediate = expected_cond_target[i] - previous;
        }
        else
          intermediate = -delta;
        if (auxiliary_pml) {
          previous_u = expected_u[i];
          expected_u[i] =
              ((kappa[k] - sigma[k]) * expected_u[i] + intermediate) * inverse[k];
        }
        else {
          expected_target[i] =
              ((kappa[k] - sigma[k]) * expected_target[i] + intermediate) * inverse[k];
          continue;
        }
      }
      else {
        previous_u = expected_u[i];
        expected_u[i] = conductive
                            ? (damping * previous_u - delta) * conductivity_inverse[i]
                            : previous_u - delta;
      }
      expected_target[i] =
          inverse_u[ku] * (((kappa_u[ku] - sigma_u[ku]) * expected_target[i] + expected_u[i]) -
                           previous_u);
    }

    copy_device_to_host_async(observed.data(), d_pml_target, 0, bytes, execution);
    std::vector<T> observed_u(elements), observed_cond_target(elements);
    copy_device_to_host_async(observed_u.data(), d_pml_u, 0, bytes, execution);
    copy_device_to_host_async(observed_cond_target.data(), d_pml_cond_target, 0, bytes, execution);
    execution.synchronize();
    const double variant_tolerance = 16.0 * std::numeric_limits<T>::epsilon();
    for (size_t i = 0; i < elements; ++i) {
      require(std::fabs(double(observed[i] - expected_target[i])) <=
                  variant_tolerance * (1.0 + std::fabs(double(expected_target[i]))),
              "curl PML/conductivity target differs");
      require(std::fabs(double(observed_u[i] - expected_u[i])) <=
                  variant_tolerance * (1.0 + std::fabs(double(expected_u[i]))),
              "curl PML auxiliary or sentinel differs");
      require(std::fabs(double(observed_cond_target[i] - expected_cond_target[i])) <=
                  variant_tolerance * (1.0 + std::fabs(double(expected_cond_target[i]))),
              "curl conductivity auxiliary or sentinel differs");
    }
  }

  copy_host_to_device_async(d_pml_target, 0, pml_target.data(), bytes, execution);
  copy_host_to_device_async(d_pml_u, 0, pml_u.data(), bytes, execution);
  constitutive_launch pml_constitutive = {};
  pml_constitutive.region = pml_region;
  pml_constitutive.target = d_pml_target.opaque_handle();
  pml_constitutive.primary = d_primary.opaque_handle();
  pml_constitutive.diagonal = d_diagonal.opaque_handle();
  pml_constitutive.target_w = d_pml_u.opaque_handle();
  pml_constitutive.pml.sigma = d_sigma.opaque_handle();
  pml_constitutive.pml.kappa = d_kappa.opaque_handle();
  pml_constitutive.pml.inverse = d_inverse.opaque_handle();
  pml_constitutive.pml.base = 3;
  pml_constitutive.pml.strides[2] = 2;
  pml_constitutive.precision = precision;
  launch_constitutive(pml_constitutive, execution);
  copy_device_to_host_async(observed.data(), d_pml_target, 0, bytes, execution);
  std::vector<T> observed_w(elements);
  copy_device_to_host_async(observed_w.data(), d_pml_u, 0, bytes, execution);
  execution.synchronize();
  std::vector<T> expected_pml_target = pml_target;
  std::vector<T> expected_w = pml_u;
  for (size_t n = 0; n < pml_region.counts[2]; ++n) {
    const size_t i = pml_region.base + n;
    const size_t k = 3 + 2 * n;
    const T value = primary[i] * diagonal[i];
    expected_pml_target[i] +=
        (kappa[k] + sigma[k]) * value - (kappa[k] - sigma[k]) * expected_w[i];
    expected_w[i] = value;
  }
  const double constitutive_tolerance = 12.0 * std::numeric_limits<T>::epsilon();
  for (size_t i = 0; i < elements; ++i) {
    require(std::fabs(double(observed[i] - expected_pml_target[i])) <=
                constitutive_tolerance * (1.0 + std::fabs(double(expected_pml_target[i]))),
            "constitutive PML target or sentinel differs");
    require(std::fabs(double(observed_w[i] - expected_w[i])) <=
                constitutive_tolerance * (1.0 + std::fabs(double(expected_w[i]))),
            "constitutive PML auxiliary or sentinel differs");
  }

  flat_region anisotropic_region = {};
  anisotropic_region.base = 16;
  anisotropic_region.counts[0] = 1;
  anisotropic_region.counts[1] = 1;
  anisotropic_region.counts[2] = 8;
  anisotropic_region.strides[0] = 0;
  anisotropic_region.strides[1] = 0;
  anisotropic_region.strides[2] = 1;
  const ptrdiff_t primary_stride = 9;
  const ptrdiff_t cross1_stride = 2;
  const ptrdiff_t cross2_stride = 3;
  for (unsigned int offdiagonals = 1; offdiagonals <= 2; ++offdiagonals)
    for (unsigned int pml_enabled = 0; pml_enabled <= 1; ++pml_enabled) {
      copy_host_to_device_async(d_output, 0, output.data(), bytes, execution);
      copy_host_to_device_async(d_pml_u, 0, pml_u.data(), bytes, execution);

      constitutive_launch anisotropic = {};
      anisotropic.region = anisotropic_region;
      anisotropic.target = d_output.opaque_handle();
      anisotropic.primary = d_primary.opaque_handle();
      anisotropic.cross1 = d_plus.opaque_handle();
      anisotropic.diagonal = d_diagonal.opaque_handle();
      anisotropic.offdiagonal1 = d_offdiagonal1.opaque_handle();
      anisotropic.primary_stride = primary_stride;
      anisotropic.cross1_stride = cross1_stride;
      if (offdiagonals == 2) {
        anisotropic.cross2 = d_minus.opaque_handle();
        anisotropic.offdiagonal2 = d_offdiagonal2.opaque_handle();
        anisotropic.cross2_stride = cross2_stride;
      }
      if (pml_enabled) {
        anisotropic.target_w = d_pml_u.opaque_handle();
        anisotropic.pml.sigma = d_sigma.opaque_handle();
        anisotropic.pml.kappa = d_kappa.opaque_handle();
        anisotropic.pml.inverse = d_inverse.opaque_handle();
        anisotropic.pml.base = 3;
        anisotropic.pml.strides[2] = 1;
      }
      anisotropic.precision = precision;
      launch_constitutive(anisotropic, execution);

      std::vector<T> expected_anisotropic = output;
      std::vector<T> expected_anisotropic_w = pml_u;
      for (size_t n = 0; n < anisotropic_region.counts[2]; ++n) {
        const ptrdiff_t i = ptrdiff_t(anisotropic_region.base + n);
        T value = primary[i] * diagonal[i];
        const T first =
            T(0.25) * ((plus[i] + plus[i - cross1_stride]) * offdiagonal1[i] +
                       (plus[i + primary_stride] + plus[(i + primary_stride) - cross1_stride]) *
                           offdiagonal1[i + primary_stride]);
        value += first;
        if (offdiagonals == 2) {
          const T second =
              T(0.25) *
              ((minus[i] + minus[i - cross2_stride]) * offdiagonal2[i] +
               (minus[i + primary_stride] + minus[(i + primary_stride) - cross2_stride]) *
                   offdiagonal2[i + primary_stride]);
          value += second;
        }
        if (pml_enabled) {
          const ptrdiff_t k = 3 + ptrdiff_t(n);
          const T previous = expected_anisotropic_w[i];
          expected_anisotropic_w[i] = value;
          expected_anisotropic[i] +=
              (kappa[k] + sigma[k]) * value - (kappa[k] - sigma[k]) * previous;
        }
        else
          expected_anisotropic[i] = value;
      }

      copy_device_to_host_async(observed.data(), d_output, 0, bytes, execution);
      copy_device_to_host_async(observed_w.data(), d_pml_u, 0, bytes, execution);
      execution.synchronize();
      const double anisotropic_tolerance = 20.0 * std::numeric_limits<T>::epsilon();
      for (size_t i = 0; i < elements; ++i) {
        require(std::fabs(double(observed[i] - expected_anisotropic[i])) <=
                    anisotropic_tolerance *
                        (1.0 + std::fabs(double(expected_anisotropic[i]))),
                "anisotropic constitutive target or sentinel differs");
        require(std::fabs(double(observed_w[i] - expected_anisotropic_w[i])) <=
                    anisotropic_tolerance *
                        (1.0 + std::fabs(double(expected_anisotropic_w[i]))),
                "anisotropic constitutive PML auxiliary or sentinel differs");
      }
    }

  struct nonlinear_case {
    unsigned int nonlinear_crosses;
    unsigned int offdiagonals;
    bool pml;
    bool use_chi2;
    bool use_chi3;
  };
  const nonlinear_case nonlinear_cases[] = {
      {0, 0, false, true, false}, {2, 0, false, false, true},
      {1, 0, false, true, true},  {2, 0, true, true, true},
      {2, 1, false, true, true},  {2, 1, true, true, true},
      {2, 2, false, true, true},  {2, 2, true, true, true}};
  for (size_t c = 0; c < sizeof(nonlinear_cases) / sizeof(nonlinear_cases[0]); ++c) {
    const nonlinear_case &test = nonlinear_cases[c];
    std::vector<T> case_chi2 = chi2, case_chi3 = chi3;
    if (!test.use_chi2) std::fill(case_chi2.begin(), case_chi2.end(), T(0));
    if (!test.use_chi3) std::fill(case_chi3.begin(), case_chi3.end(), T(0));
    copy_host_to_device_async(d_chi2, 0, case_chi2.data(), bytes, execution);
    copy_host_to_device_async(d_chi3, 0, case_chi3.data(), bytes, execution);
    copy_host_to_device_async(d_output, 0, output.data(), bytes, execution);
    copy_host_to_device_async(d_pml_u, 0, pml_u.data(), bytes, execution);

    constitutive_launch nonlinear = {};
    nonlinear.region = anisotropic_region;
    nonlinear.target = d_output.opaque_handle();
    nonlinear.primary = d_primary.opaque_handle();
    nonlinear.diagonal = d_diagonal.opaque_handle();
    nonlinear.chi2 = d_chi2.opaque_handle();
    nonlinear.chi3 = d_chi3.opaque_handle();
    nonlinear.primary_stride = primary_stride;
    nonlinear.cross1_stride = cross1_stride;
    nonlinear.cross2_stride = cross2_stride;
    if (test.nonlinear_crosses >= 1) nonlinear.cross1 = d_plus.opaque_handle();
    if (test.nonlinear_crosses >= 2) nonlinear.cross2 = d_minus.opaque_handle();
    if (test.offdiagonals >= 1) {
      nonlinear.cross1 = d_plus.opaque_handle();
      nonlinear.offdiagonal1 = d_offdiagonal1.opaque_handle();
    }
    if (test.offdiagonals >= 2) {
      nonlinear.cross2 = d_minus.opaque_handle();
      nonlinear.offdiagonal2 = d_offdiagonal2.opaque_handle();
    }
    if (test.pml) {
      nonlinear.target_w = d_pml_u.opaque_handle();
      nonlinear.pml.sigma = d_sigma.opaque_handle();
      nonlinear.pml.kappa = d_kappa.opaque_handle();
      nonlinear.pml.inverse = d_inverse.opaque_handle();
      nonlinear.pml.base = 3;
      nonlinear.pml.strides[2] = 1;
    }
    nonlinear.precision = precision;
    launch_constitutive(nonlinear, execution);

    std::vector<T> nonlinear_expected = output;
    std::vector<T> nonlinear_w_expected = pml_u;
    for (size_t n = 0; n < anisotropic_region.counts[2]; ++n) {
      const ptrdiff_t i = ptrdiff_t(anisotropic_region.base + n);
      T value = primary[i] * diagonal[i];
      if (test.offdiagonals >= 1)
        value += T(0.25) *
                 ((plus[i] + plus[i - cross1_stride]) * offdiagonal1[i] +
                  (plus[i + primary_stride] + plus[i + primary_stride - cross1_stride]) *
                      offdiagonal1[i + primary_stride]);
      if (test.offdiagonals >= 2)
        value += T(0.25) *
                 ((minus[i] + minus[i - cross2_stride]) * offdiagonal2[i] +
                  (minus[i + primary_stride] + minus[i + primary_stride - cross2_stride]) *
                      offdiagonal2[i + primary_stride]);
      T dsqr;
      if (test.nonlinear_crosses == 2) {
        const T g1s = plus[i] + plus[i + primary_stride] + plus[i - cross1_stride] +
                      plus[i + primary_stride - cross1_stride];
        const T g2s = minus[i] + minus[i + primary_stride] + minus[i - cross2_stride] +
                      minus[i + primary_stride - cross2_stride];
        dsqr = primary[i] * primary[i] + T(0.0625) * (g1s * g1s + g2s * g2s);
      }
      else if (test.nonlinear_crosses == 1) {
        const T g1s = plus[i] + plus[i + primary_stride] + plus[i - cross1_stride] +
                      plus[i + primary_stride - cross1_stride];
        dsqr = primary[i] * primary[i] + T(0.0625) * (g1s * g1s);
      }
      else
        dsqr = primary[i] * primary[i];
      value *=
          nonlinear_reference(dsqr, primary[i], diagonal[i], case_chi2[i], case_chi3[i]);
      if (test.pml) {
        const ptrdiff_t k = 3 + ptrdiff_t(n);
        const T previous = nonlinear_w_expected[i];
        nonlinear_w_expected[i] = value;
        nonlinear_expected[i] +=
            (kappa[k] + sigma[k]) * value - (kappa[k] - sigma[k]) * previous;
      }
      else
        nonlinear_expected[i] = value;
    }

    copy_device_to_host_async(observed.data(), d_output, 0, bytes, execution);
    copy_device_to_host_async(observed_w.data(), d_pml_u, 0, bytes, execution);
    execution.synchronize();
    const double tolerance = 64.0 * std::numeric_limits<T>::epsilon();
    for (size_t i = 0; i < elements; ++i) {
      require(std::fabs(double(observed[i] - nonlinear_expected[i])) <=
                  tolerance * (1.0 + std::fabs(double(nonlinear_expected[i]))),
              "nonlinear constitutive target or sentinel differs");
      require(std::fabs(double(observed_w[i] - nonlinear_w_expected[i])) <=
                  tolerance * (1.0 + std::fabs(double(nonlinear_w_expected[i]))),
              "nonlinear constitutive PML auxiliary or sentinel differs");
    }
  }

  {
    constitutive_launch invalid = {};
    invalid.region = anisotropic_region;
    invalid.target = d_output.opaque_handle();
    invalid.primary = d_primary.opaque_handle();
    invalid.diagonal = d_diagonal.opaque_handle();
    invalid.chi3 = d_chi3.opaque_handle();
    invalid.precision = precision;
    bool rejected = false;
    try {
      launch_constitutive(invalid, execution);
    }
    catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "constitutive launch accepted a partial nonlinear descriptor");

    invalid.chi2 = d_chi2.opaque_handle();
    invalid.cross2 = d_minus.opaque_handle();
    rejected = false;
    try {
      launch_constitutive(invalid, execution);
    }
    catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "constitutive launch accepted nonlinear cross2 without cross1");
  }
  if (std::getenv("MEEP_NVIDIA_STEP_NONLINEAR_ONLY")) return;

  std::vector<T> finite_values(elements);
  for (size_t i = 0; i < elements; ++i) finite_values[i] = T(0.125 * double(i) - 3.0);
  device_buffer d_finite_values(bytes, device), d_finite_result(sizeof(uint64_t), device);
  finite_check_launch finite = {};
  finite.values = d_finite_values.opaque_handle();
  finite.elements = elements;
  finite.ordinal_base = 37;
  finite.precision = precision;
  uint64_t first_bad = 0;

  copy_host_to_device_async(d_finite_values, 0, finite_values.data(), bytes, execution);
  fill_byte_async(d_finite_result, 0, 0xff, sizeof(first_bad), execution);
  launch_finite_check(finite, d_finite_result.opaque_handle(), execution);
  copy_device_to_host_async(&first_bad, d_finite_result, 0, sizeof(first_bad), execution);
  execution.synchronize();
  require(first_bad == std::numeric_limits<uint64_t>::max(),
          "finite-value check rejected finite input");

  finite_values[13] = std::numeric_limits<T>::quiet_NaN();
  finite_values[7] = std::numeric_limits<T>::infinity();
  copy_host_to_device_async(d_finite_values, 0, finite_values.data(), bytes, execution);
  fill_byte_async(d_finite_result, 0, 0xff, sizeof(first_bad), execution);
  launch_finite_check(finite, d_finite_result.opaque_handle(), execution);
  copy_device_to_host_async(&first_bad, d_finite_result, 0, sizeof(first_bad), execution);
  execution.synchronize();
  require(first_bad == finite.ordinal_base + 7,
          "finite-value check did not select the lowest non-finite element");

  finite_values[7] = T(0.875);
  copy_host_to_device_async(d_finite_values, 0, finite_values.data(), bytes, execution);
  fill_byte_async(d_finite_result, 0, 0xff, sizeof(first_bad), execution);
  launch_finite_check(finite, d_finite_result.opaque_handle(), execution);
  copy_device_to_host_async(&first_bad, d_finite_result, 0, sizeof(first_bad), execution);
  execution.synchronize();
  require(first_bad == finite.ordinal_base + 13, "finite-value check did not report NaN");

  std::vector<T> halo_values(8), halo_observed(8);
  for (size_t i = 0; i < halo_values.size(); ++i) halo_values[i] = T(i + 1);
  device_buffer d_halo(halo_values.size() * sizeof(T), device);
  device_buffer d_halo_scratch(6 * sizeof(T), device);
  copy_host_to_device_async(d_halo, 0, halo_values.data(), halo_values.size() * sizeof(T),
                            execution);

  std::vector<halo_gather_entry> gather_entries;
  for (size_t i = 0; i < 6; ++i)
    gather_entries.push_back(halo_gather_entry{d_halo.opaque_handle(), ptrdiff_t(i), i});
  std::vector<halo_scatter_entry> scatter_entries;
  /* Swap 0/1 to prove every gather completed before any scatter. */
  scatter_entries.push_back(
      halo_scatter_entry{d_halo.opaque_handle(), 1, NULL, 0, 0, 1.0, 0.0});
  scatter_entries.push_back(
      halo_scatter_entry{d_halo.opaque_handle(), 0, NULL, 0, 1, 1.0, 0.0});
  scatter_entries.push_back(
      halo_scatter_entry{d_halo.opaque_handle(), 2, NULL, 0, 2, -1.0, 0.0});
  scatter_entries.push_back(
      halo_scatter_entry{d_halo.opaque_handle(), 4, d_halo.opaque_handle(), 5, 4, 0.6, 0.8});

  device_buffer d_gather_entries(gather_entries.size() * sizeof(gather_entries[0]), device);
  device_buffer d_scatter_entries(scatter_entries.size() * sizeof(scatter_entries[0]), device);
  copy_host_to_device_async(d_gather_entries, 0, gather_entries.data(),
                            gather_entries.size() * sizeof(gather_entries[0]), execution);
  copy_host_to_device_async(d_scatter_entries, 0, scatter_entries.data(),
                            scatter_entries.size() * sizeof(scatter_entries[0]), execution);
  launch_halo_gather(halo_launch{0, gather_entries.size(), precision},
                     d_gather_entries.opaque_handle(), d_halo_scratch.opaque_handle(), execution);
  launch_halo_scatter(halo_launch{0, 2, precision}, d_scatter_entries.opaque_handle(),
                      d_halo_scratch.opaque_handle(), execution);
  launch_halo_scatter(halo_launch{2, 1, precision}, d_scatter_entries.opaque_handle(),
                      d_halo_scratch.opaque_handle(), execution);
  launch_halo_scatter(halo_launch{3, 1, precision}, d_scatter_entries.opaque_handle(),
                      d_halo_scratch.opaque_handle(), execution);
  copy_device_to_host_async(halo_observed.data(), d_halo, 0,
                            halo_observed.size() * sizeof(T), execution);
  execution.synchronize();

  require(halo_observed[0] == halo_values[1] && halo_observed[1] == halo_values[0],
          "halo COPY did not preserve gather-before-scatter semantics");
  require(halo_observed[2] == -halo_values[2], "halo NEGATE result differs");
  const T phase_real = T(0.6) * halo_values[4] - T(0.8) * halo_values[5];
  const T phase_imag = T(0.6) * halo_values[5] + T(0.8) * halo_values[4];
  const double phase_tolerance = 4.0 * std::numeric_limits<T>::epsilon();
  require(std::fabs(double(halo_observed[4] - phase_real)) <=
              phase_tolerance * (1.0 + std::fabs(double(phase_real))) &&
              std::fabs(double(halo_observed[5] - phase_imag)) <=
                  phase_tolerance * (1.0 + std::fabs(double(phase_imag))),
          "halo PHASE result differs");

  /* A 259-point descriptor crosses the 256-thread launch boundary and models
     the packed plane data used by volume/eigenmode sources. The following
     descriptors overlap it, and the last contains duplicate indices. */
  const size_t source_elements = 300;
  std::vector<T> source_real(source_elements), source_imag(source_elements),
      source_condinv(source_elements), source_observed(source_elements),
      source_imag_observed(source_elements);
  for (size_t i = 0; i < source_real.size(); ++i) {
    source_real[i] = T(0.25 + 0.01 * double(i));
    source_imag[i] = T(-0.4 + 0.02 * double(i));
    source_condinv[i] = T(0.8 + 0.01 * double(i));
  }
  const source_scalar scalar = {0.75, -0.25, 1.25, 0.5};
  device_buffer d_source_real(source_real.size() * sizeof(T), device);
  device_buffer d_source_imag(source_imag.size() * sizeof(T), device);
  device_buffer d_source_condinv(source_condinv.size() * sizeof(T), device);
  device_buffer d_source_scalar(sizeof(scalar), device);
  std::vector<source_point> source_points(259 + 3 + 3);
  for (size_t i = 0; i < 259; ++i) {
    source_points[i].index = ptrdiff_t(i);
    source_points[i].amplitude_real = 0.003 * double(i + 1);
    source_points[i].amplitude_imag = -0.002 * double((i % 13) + 1);
  }
  const ptrdiff_t overlap_indices[] = {3, 127, 258};
  const ptrdiff_t duplicate_indices[] = {41, 41, 41};
  require(!source_indices_require_sequential(overlap_indices, 3),
          "source compiler classified unique indices as sequential");
  require(source_indices_require_sequential(duplicate_indices, 3),
          "source compiler missed duplicate indices");
  for (size_t i = 0; i < 3; ++i) {
    source_points[259 + i].index = overlap_indices[i];
    source_points[259 + i].amplitude_real = -0.11 * double(i + 1);
    source_points[259 + i].amplitude_imag = 0.07 * double(i + 1);
    source_points[262 + i].index = duplicate_indices[i];
    source_points[262 + i].amplitude_real =
        i == 0 ? 8.0e19 : (i == 1 ? -8.0e19 : 2.5132741228718345);
    source_points[262 + i].amplitude_imag = 0.0;
  }
  device_buffer d_source_points(source_points.size() * sizeof(source_points[0]), device);
  copy_host_to_device_async(d_source_real, 0, source_real.data(),
                            source_real.size() * sizeof(T), execution);
  copy_host_to_device_async(d_source_imag, 0, source_imag.data(),
                            source_imag.size() * sizeof(T), execution);
  copy_host_to_device_async(d_source_condinv, 0, source_condinv.data(),
                            source_condinv.size() * sizeof(T), execution);
  copy_host_to_device_async(d_source_scalar, 0, &scalar, sizeof(scalar), execution);
  copy_host_to_device_async(d_source_points, 0, source_points.data(),
                            source_points.size() * sizeof(source_points[0]), execution);
  source_batch_launch batch = {};
  batch.target_real = d_source_real.opaque_handle();
  batch.target_imag = d_source_imag.opaque_handle();
  batch.conductivity_inverse = d_source_condinv.opaque_handle();
  batch.points = static_cast<const source_point *>(d_source_points.opaque_handle());
  batch.point_count = 259;
  batch.scalar_slot = 0;
  batch.dt = 0.125;
  batch.precision = precision;
  launch_source_batch(batch, d_source_scalar.opaque_handle(), execution);
  source_batch_launch overlap = batch;
  overlap.points += 259;
  overlap.point_count = 3;
  launch_source_batch(overlap, d_source_scalar.opaque_handle(), execution);
  source_batch_launch duplicate = batch;
  duplicate.points += 262;
  duplicate.point_count = 3;
  duplicate.sequential =
      source_indices_require_sequential(duplicate_indices, duplicate.point_count);
  launch_source_batch(duplicate, d_source_scalar.opaque_handle(), execution);
  copy_device_to_host_async(source_observed.data(), d_source_real, 0,
                            source_observed.size() * sizeof(T), execution);
  copy_device_to_host_async(source_imag_observed.data(), d_source_imag, 0,
                            source_imag_observed.size() * sizeof(T), execution);
  execution.synchronize();
  std::vector<T> source_expected = source_real;
  std::vector<T> source_imag_expected = source_imag;
  for (size_t p = 0; p < source_points.size(); ++p) {
    const source_point &point = source_points[p];
    double source_value_real =
        point.amplitude_real * scalar.current_real - point.amplitude_imag * scalar.current_imag;
    double source_value_imag =
        point.amplitude_real * scalar.current_imag + point.amplitude_imag * scalar.current_real;
    source_value_real *= batch.dt * double(source_condinv[point.index]);
    source_value_imag *= batch.dt * double(source_condinv[point.index]);
    source_expected[point.index] -= T(source_value_real);
    source_imag_expected[point.index] -= T(source_value_imag);
  }
  const double source_tolerance = 4.0 * std::numeric_limits<T>::epsilon();
  for (size_t i = 0; i < source_observed.size(); ++i) {
    require(std::fabs(double(source_observed[i] - source_expected[i])) <=
                source_tolerance * (1.0 + std::fabs(double(source_expected[i]))),
            "point-source real result or sentinel differs");
    require(std::fabs(double(source_imag_observed[i] - source_imag_expected[i])) <=
                source_tolerance * (1.0 + std::fabs(double(source_imag_expected[i]))),
            "point-source imaginary result or sentinel differs");
  }

  std::vector<T> integrated_target(source_real.size(), T(-3.0));
  copy_host_to_device_async(d_source_real, 0, integrated_target.data(),
                            integrated_target.size() * sizeof(T), execution);
  array_copy_launch source_copy = {};
  source_copy.target = d_source_real.opaque_handle();
  source_copy.source = d_source_imag.opaque_handle();
  source_copy.elements = source_real.size();
  source_copy.precision = precision;
  launch_array_copy(source_copy, execution);
  source_batch_launch integrated = {};
  integrated.target_real = d_source_real.opaque_handle();
  integrated.points = static_cast<const source_point *>(d_source_points.opaque_handle()) + 262;
  integrated.point_count = 3;
  integrated.scalar_slot = 0;
  integrated.integrated = true;
  integrated.sequential =
      source_indices_require_sequential(duplicate_indices, integrated.point_count);
  integrated.precision = precision;
  launch_source_batch(integrated, d_source_scalar.opaque_handle(), execution);
  copy_device_to_host_async(source_observed.data(), d_source_real, 0,
                            source_observed.size() * sizeof(T), execution);
  execution.synchronize();
  std::vector<T> integrated_expected = source_imag_observed;
  for (size_t p = 262; p < 265; ++p) {
    const source_point &point = source_points[p];
    const double dipole_real =
        point.amplitude_real * scalar.dipole_real - point.amplitude_imag * scalar.dipole_imag;
    integrated_expected[point.index] -= T(dipole_real);
  }
  for (size_t i = 0; i < source_observed.size(); ++i)
    require(std::fabs(double(source_observed[i] - integrated_expected[i])) <=
                source_tolerance * (1.0 + std::fabs(double(integrated_expected[i]))),
            "integrated-source copy/application result or sentinel differs");
  require(source_observed[41] == integrated_expected[41],
          "ordered duplicate-source fallback changed sequential floating-point behavior");
  T reordered = source_imag_observed[41];
  const size_t reordered_points[] = {262, 264, 263};
  for (size_t i = 0; i < 3; ++i) {
    const source_point &point = source_points[reordered_points[i]];
    const double dipole_real =
        point.amplitude_real * scalar.dipole_real - point.amplitude_imag * scalar.dipole_imag;
    reordered -= T(dipole_real);
  }
  require(source_observed[41] != reordered,
          "duplicate-source test values do not distinguish descriptor order");
}

int main() {
  try {
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA devices found");
    for (size_t i = 0; i < devices.size(); ++i) {
      check_device<float>(devices[i].id);
      check_device<double>(devices[i].id);
      std::cout << "device " << devices[i].id << " (" << devices[i].name
                << "): NVIDIA step kernels PASS\n";
    }
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_step_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
  std::cout << "nvidia_step_smoke: PASS\n";
  return 0;
}
