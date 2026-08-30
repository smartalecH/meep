/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_flux.hpp"
#include "backend/nvidia/runtime.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace meep::nvidia;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

static double boundary_weight(size_t i, size_t n, double s0, double s1, double e0,
                              double e1) {
  const ptrdiff_t si = ptrdiff_t(i), sn = ptrdiff_t(n);
  if (si > 1 && si < sn - 2) return 1.0;
  if (si == 0) return s0;
  if (si == 1) return s1;
  if (si == sn - 1) return e0;
  if (si == sn - 2) return e1;
  return 1.0;
}

template <typename T>
static double centered(const std::vector<T> &v, ptrdiff_t i, ptrdiff_t o0, ptrdiff_t o1) {
  const T sum = (v[i] + v[i + o0]) + (v[i + o1] + v[i + o0 + o1]);
  return 0.25 * double(sum);
}

template <typename T>
static void check_precision(int device, scalar_precision precision) {
  const size_t n0 = 2, n1 = 3, n2 = 43;
  const size_t points = n0 * n1 * n2, elements = points + 5;
  std::vector<T> er(elements), ei(elements), hr(elements), hi(elements);
  for (size_t i = 0; i < elements; ++i) {
    er[i] = T(0.2 + 0.001 * double(i));
    ei[i] = T(-0.1 + 0.0007 * double(i));
    hr[i] = T(std::sin(0.019 * double(i + 1)));
    hi[i] = T(std::cos(0.023 * double(i + 1)));
  }

  stream execution;
  device_buffer d_er(elements * sizeof(T), device), d_ei(elements * sizeof(T), device),
      d_hr(elements * sizeof(T), device), d_hi(elements * sizeof(T), device);
  copy_host_to_device_async(d_er, 0, er.data(), elements * sizeof(T), execution);
  copy_host_to_device_async(d_ei, 0, ei.data(), elements * sizeof(T), execution);
  copy_host_to_device_async(d_hr, 0, hr.data(), elements * sizeof(T), execution);
  copy_host_to_device_async(d_hi, 0, hi.data(), elements * sizeof(T), execution);

  legacy_flux_term_launch launch = {};
  launch.region.base = 0;
  launch.region.counts[0] = n0;
  launch.region.counts[1] = n1;
  launch.region.counts[2] = n2;
  launch.region.strides[0] = n1 * n2;
  launch.region.strides[1] = n2;
  launch.region.strides[2] = 1;
  launch.e_real = d_er.opaque_handle();
  launch.e_imag = d_ei.opaque_handle();
  launch.h_real = d_hr.opaque_handle();
  launch.h_imag = d_hi.opaque_handle();
  launch.e_offsets[0] = 1;
  launch.e_offsets[1] = 3;
  launch.h_offsets[0] = 2;
  launch.h_offsets[1] = 3;
  launch.phase_real = 0.8;
  launch.phase_imag = -0.6;
  for (int axis = 0; axis < 3; ++axis) {
    launch.start0[axis] = 1.0;
    launch.start1[axis] = 1.0;
    launch.end0[axis] = 1.0;
    launch.end1[axis] = 1.0;
  }
  launch.start0[0] = 0.55;
  launch.start1[0] = 0.85;
  launch.end0[0] = 0.65;
  launch.end1[0] = 0.95;
  launch.start0[1] = 0.45;
  launch.start1[1] = 0.75;
  launch.end0[1] = 0.35;
  launch.end1[1] = 0.8;
  launch.start0[2] = 0.3;
  launch.start1[2] = 0.7;
  launch.end0[2] = 0.4;
  launch.end1[2] = 0.9;
  launch.dV0 = 0.125;
  launch.dV1 = 0.03125;
  launch.sign = -1;
  launch.precision = precision;
  launch.points = points;
  launch.blocks = legacy_flux_partial_count(points);

  device_buffer partials(launch.blocks * sizeof(double), device);
  device_buffer current(sizeof(double), device), half(sizeof(double), device);
  fill_byte_async(current, 0, 0, sizeof(double), execution);
  launch_legacy_flux_term(launch, partials.opaque_handle(), current.opaque_handle(), execution);
  double observed = 0.0;
  std::vector<double> observed_partials(launch.blocks, 0.0);
  copy_device_to_host_async(observed_partials.data(), partials, 0,
                            observed_partials.size() * sizeof(double), execution);
  copy_device_to_host_async(&observed, current, 0, sizeof(double), execution);
  execution.synchronize();

  double expected = 0.0;
  for (size_t linear = 0; linear < points; ++linear) {
    size_t rest = linear;
    const size_t i2 = rest % n2;
    rest /= n2;
    const size_t i1 = rest % n1;
    const size_t i0 = rest / n1;
    const ptrdiff_t index = ptrdiff_t(i0 * n1 * n2 + i1 * n2 + i2);
    const double e_real = centered(er, index, 1, 3);
    const double e_imag = centered(ei, index, 1, 3);
    const double h_real = centered(hr, index, 2, 3);
    const double h_imag = centered(hi, index, 2, 3);
    const double product_real = e_real * h_real + e_imag * h_imag;
    const double product_imag = e_real * h_imag - e_imag * h_real;
    const double weight =
        boundary_weight(i2, n2, launch.start0[2], launch.start1[2], launch.end0[2],
                        launch.end1[2]) *
        (boundary_weight(i1, n1, launch.start0[1], launch.start1[1], launch.end0[1],
                         launch.end1[1]) *
         ((launch.dV0 + launch.dV1 * double(i1)) *
          boundary_weight(i0, n0, launch.start0[0], launch.start1[0], launch.end0[0],
                          launch.end1[0])));
    expected -= (product_real * 0.8 - product_imag * -0.6) * weight;
  }
  const double tolerance = precision == scalar_precision::f32 ? 2e-6 : 2e-13;
  if (std::fabs(observed - expected) > tolerance * (1.0 + std::fabs(expected)))
    std::cerr << "precision=" << int(precision) << " observed=" << observed
              << " expected=" << expected << " difference=" << observed - expected
              << " partial0=" << observed_partials[0] << " partial1=" << observed_partials[1]
              << "\n";
  require(std::fabs(observed - expected) <= tolerance * (1.0 + std::fabs(expected)),
          "legacy flux deterministic reduction differs from reference");

  const double current_value = observed;
  const double half_value = current_value * 0.25;
  copy_host_to_device_async(half, 0, &half_value, sizeof(double), execution);
  launch_legacy_flux_average(current.opaque_handle(), half.opaque_handle(), 1, execution);
  copy_device_to_host_async(&observed, current, 0, sizeof(double), execution);
  execution.synchronize();
  require(observed == 0.5 * (current_value + half_value),
          "legacy flux half-step averaging differs");

  bool rejected = false;
  try {
    legacy_flux_term_launch malformed = launch;
    malformed.blocks = 0;
    launch_legacy_flux_term(malformed, partials.opaque_handle(), current.opaque_handle(), execution);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "legacy flux malformed launch was accepted");
  rejected = false;
  try {
    legacy_flux_term_launch malformed = launch;
    --malformed.points;
    malformed.blocks = legacy_flux_partial_count(malformed.points);
    launch_legacy_flux_term(malformed, partials.opaque_handle(), current.opaque_handle(), execution);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "legacy flux region/product mismatch was accepted");
}

int main() {
  try {
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA devices found");
    for (size_t i = 0; i < devices.size(); ++i) {
      device_scope scope(devices[i].id);
      check_precision<float>(devices[i].id, scalar_precision::f32);
      check_precision<double>(devices[i].id, scalar_precision::f64);
      std::cout << "device " << devices[i].id << " (" << devices[i].name
                << "): NVIDIA legacy flux kernels PASS\n";
    }
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_flux_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
  std::cout << "nvidia_flux_smoke: PASS\n";
  return 0;
}
