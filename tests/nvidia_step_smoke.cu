/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_step.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using meep::nvidia::constitutive_launch;
using meep::nvidia::copy_device_to_host_async;
using meep::nvidia::copy_host_to_device_async;
using meep::nvidia::curl_launch;
using meep::nvidia::device_buffer;
using meep::nvidia::device_properties;
using meep::nvidia::device_scope;
using meep::nvidia::enumerate_devices;
using meep::nvidia::flat_region;
using meep::nvidia::launch_constitutive;
using meep::nvidia::launch_curl;
using meep::nvidia::launch_zero;
using meep::nvidia::scalar_precision;
using meep::nvidia::stream;
using meep::nvidia::zero_launch;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename T> static void check_device(int device) {
  device_scope selected(device);
  stream execution;
  const size_t elements = 96;
  const size_t bytes = elements * sizeof(T);
  std::vector<T> target(elements), plus(elements), minus(elements), primary(elements),
      diagonal(elements), observed(elements);
  for (size_t i = 0; i < elements; ++i) {
    target[i] = T(0.01 * double(i + 1));
    plus[i] = T(std::sin(0.13 * double(i)));
    minus[i] = T(std::cos(0.07 * double(i)));
    primary[i] = T(0.25 + 0.005 * double(i));
    diagonal[i] = T(0.5 + 0.002 * double(i));
  }

  device_buffer d_target(bytes, device), d_plus(bytes, device), d_minus(bytes, device),
      d_primary(bytes, device), d_diagonal(bytes, device), d_output(bytes, device);
  copy_host_to_device_async(d_target, 0, target.data(), bytes, execution);
  copy_host_to_device_async(d_plus, 0, plus.data(), bytes, execution);
  copy_host_to_device_async(d_minus, 0, minus.data(), bytes, execution);
  copy_host_to_device_async(d_primary, 0, primary.data(), bytes, execution);
  copy_host_to_device_async(d_diagonal, 0, diagonal.data(), bytes, execution);

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

  curl_launch curl;
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

  constitutive_launch constitutive;
  constitutive.region = region;
  constitutive.target = d_output.opaque_handle();
  constitutive.primary = d_primary.opaque_handle();
  constitutive.diagonal = d_diagonal.opaque_handle();
  constitutive.precision = precision;
  launch_constitutive(constitutive, execution);
  copy_device_to_host_async(observed.data(), d_output, 0, bytes, execution);
  execution.synchronize();
  for (size_t i0 = 0; i0 < region.counts[0]; ++i0)
    for (size_t i1 = 0; i1 < region.counts[1]; ++i1)
      for (size_t i2 = 0; i2 < region.counts[2]; ++i2) {
        const ptrdiff_t i = ptrdiff_t(region.base) + ptrdiff_t(i0) * region.strides[0] +
                            ptrdiff_t(i1) * region.strides[1] + ptrdiff_t(i2) * region.strides[2];
        const T expected_value = primary[i] * diagonal[i];
        require(std::fabs(double(observed[i] - expected_value)) <=
                    2.0 * std::numeric_limits<T>::epsilon() *
                        (1.0 + std::fabs(double(expected_value))),
                "constitutive result differs from host reference");
      }

  zero_launch zero;
  zero.region = region;
  zero.target = d_target.opaque_handle();
  zero.precision = precision;
  launch_zero(zero, execution);
  copy_device_to_host_async(observed.data(), d_target, 0, bytes, execution);
  execution.synchronize();
  for (size_t i0 = 0; i0 < region.counts[0]; ++i0)
    for (size_t i1 = 0; i1 < region.counts[1]; ++i1)
      for (size_t i2 = 0; i2 < region.counts[2]; ++i2) {
        const ptrdiff_t i = ptrdiff_t(region.base) + ptrdiff_t(i0) * region.strides[0] +
                            ptrdiff_t(i1) * region.strides[1] + ptrdiff_t(i2) * region.strides[2];
        require(observed[i] == T(0), "zero result is nonzero");
      }
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
