/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_polarization.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace meep::nvidia;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename T>
static T offdiag(const std::vector<T> &coefficient, const std::vector<T> &field, ptrdiff_t i,
                 ptrdiff_t primary_stride, ptrdiff_t cross_stride) {
  return T(0.25) *
         ((field[i] + field[i - cross_stride]) * coefficient[i] +
          (field[i + primary_stride] + field[i + primary_stride - cross_stride]) *
              coefficient[i + primary_stride]);
}

template <typename T> static void check_device(int device) {
  device_scope selected(device);
  stream execution;
  const size_t elements = 320;
  const size_t bytes = elements * sizeof(T);
  std::vector<T> p(elements), p_prev(elements), w(elements), w1(elements), w2(elements),
      sigma(elements), sigma1(elements), sigma2(elements), observed(elements),
      observed_prev(elements);
  for (size_t i = 0; i < elements; ++i) {
    p[i] = T(0.2 + 0.001 * double(i));
    p_prev[i] = T(-0.1 + 0.0007 * double(i));
    w[i] = T(0.4 - 0.0009 * double(i));
    w1[i] = T(-0.3 + 0.0011 * double(i));
    w2[i] = T(0.1 + 0.0005 * double(i));
    sigma[i] = T(0.8 + 0.0003 * double(i));
    sigma1[i] = T(0.04 - 0.0001 * double(i));
    sigma2[i] = T(-0.03 + 0.0002 * double(i));
  }

  device_buffer d_p(bytes, device), d_p_prev(bytes, device), d_w(bytes, device),
      d_w1(bytes, device), d_w2(bytes, device), d_sigma(bytes, device),
      d_sigma1(bytes, device), d_sigma2(bytes, device);
  copy_host_to_device_async(d_w, 0, w.data(), bytes, execution);
  copy_host_to_device_async(d_w1, 0, w1.data(), bytes, execution);
  copy_host_to_device_async(d_w2, 0, w2.data(), bytes, execution);
  copy_host_to_device_async(d_sigma1, 0, sigma1.data(), bytes, execution);
  copy_host_to_device_async(d_sigma2, 0, sigma2.data(), bytes, execution);

  flat_region region = {};
  region.base = 30;
  region.counts[0] = 1;
  region.counts[1] = 1;
  region.counts[2] = 257;
  region.strides[2] = 1;
  const scalar_precision precision =
      sizeof(T) == sizeof(float) ? scalar_precision::f32 : scalar_precision::f64;
  size_t cases = 0;
  for (unsigned int offdiagonals = 0; offdiagonals <= 2; ++offdiagonals)
    for (int drude = 0; drude <= 1; ++drude) {
      std::vector<T> expected = p, expected_prev = p_prev;
      std::vector<T> diagonal = sigma;
      if (offdiagonals) diagonal[44] = T(0);
      copy_host_to_device_async(d_p, 0, p.data(), bytes, execution);
      copy_host_to_device_async(d_p_prev, 0, p_prev.data(), bytes, execution);
      copy_host_to_device_async(d_sigma, 0, diagonal.data(), bytes, execution);

      polarization_update_launch launch = {};
      launch.region = region;
      launch.p = d_p.opaque_handle();
      launch.p_prev = d_p_prev.opaque_handle();
      launch.primary_w = d_w.opaque_handle();
      launch.diagonal_sigma = d_sigma.opaque_handle();
      launch.offdiagonals = offdiagonals;
      launch.drude = drude != 0;
      launch.precision = precision;
      launch.primary_stride = -2;
      launch.cross_stride1 = -5;
      launch.cross_stride2 = -7;
      if (offdiagonals >= 1) {
        launch.cross_w1 = d_w1.opaque_handle();
        launch.offdiagonal_sigma1 = d_sigma1.opaque_handle();
      }
      if (offdiagonals >= 2) {
        launch.cross_w2 = d_w2.opaque_handle();
        launch.offdiagonal_sigma2 = d_sigma2.opaque_handle();
      }
      const double omega = 0.73 + 0.09 * drude;
      const double gamma = drude ? 0.0 : 0.06;
      const double dt = 0.017;
      const double omega2pi = 2 * 3.14159265358979323846 * omega;
      const double gamma2pi = 2 * 3.14159265358979323846 * gamma;
      launch.omega0dtsqr = omega2pi * omega2pi * dt * dt;
      launch.gamma1inv = 1 / (1 + gamma2pi * dt / 2);
      launch.gamma1 = 1 - gamma2pi * dt / 2;
      launch.omega0dtsqr_denom = drude ? 0 : launch.omega0dtsqr;

      for (size_t linear = 0; linear < region.counts[2]; ++linear) {
        const ptrdiff_t i = ptrdiff_t(region.base + linear);
        if (offdiagonals && diagonal[i] == T(0)) continue;
        T forcing = diagonal[i] * w[i];
        if (offdiagonals >= 1)
          forcing += offdiag(sigma1, w1, i, launch.primary_stride, launch.cross_stride1);
        if (offdiagonals >= 2)
          forcing += offdiag(sigma2, w2, i, launch.primary_stride, launch.cross_stride2);
        const T current = expected[i];
        expected[i] = T(launch.gamma1inv) *
                      (current * (T(2) - T(launch.omega0dtsqr_denom)) -
                       T(launch.gamma1) * expected_prev[i] + T(launch.omega0dtsqr) * forcing);
        expected_prev[i] = current;
      }

      launch_polarization_update(launch, execution);
      copy_device_to_host_async(observed.data(), d_p, 0, bytes, execution);
      copy_device_to_host_async(observed_prev.data(), d_p_prev, 0, bytes, execution);
      execution.synchronize();
      const double tolerance = 16 * std::numeric_limits<T>::epsilon();
      for (size_t i = 0; i < elements; ++i) {
        require(std::fabs(double(observed[i] - expected[i])) <=
                    tolerance * (1 + std::fabs(double(expected[i]))),
                "polarization recurrence or sentinel differs");
        require(std::fabs(double(observed_prev[i] - expected_prev[i])) <=
                    tolerance * (1 + std::fabs(double(expected_prev[i]))),
                "polarization P_prev or sentinel differs");
      }
      ++cases;
    }

  std::vector<T> target(elements), p2(elements);
  for (size_t i = 0; i < elements; ++i) {
    target[i] = T(2 + 0.01 * double(i));
    p2[i] = T(0.05 - 0.0002 * double(i));
  }
  copy_host_to_device_async(d_w, 0, target.data(), bytes, execution);
  copy_host_to_device_async(d_p, 0, p.data(), bytes, execution);
  copy_host_to_device_async(d_p_prev, 0, p2.data(), bytes, execution);
  polarization_subtract_launch subtraction = {};
  subtraction.target = d_w.opaque_handle();
  subtraction.p = d_p.opaque_handle();
  subtraction.elements = elements;
  subtraction.precision = precision;
  launch_polarization_subtract(subtraction, execution);
  copy_device_to_host_async(observed.data(), d_w, 0, bytes, execution);
  execution.synchronize();
  for (size_t i = 0; i < elements; ++i)
    require(observed[i] == T(target[i] - p[i]),
            "single full-array polarization subtraction differs");
  subtraction.p = d_p_prev.opaque_handle();
  launch_polarization_subtract(subtraction, execution);
  copy_device_to_host_async(observed.data(), d_w, 0, bytes, execution);
  execution.synchronize();
  for (size_t i = 0; i < elements; ++i)
    require(observed[i] == T(target[i] - p[i] - p2[i]),
            "ordered full-array polarization subtraction differs");

  bool rejected = false;
  polarization_update_launch malformed = {};
  malformed.region = region;
  malformed.p = d_p.opaque_handle();
  malformed.p_prev = d_p_prev.opaque_handle();
  malformed.primary_w = d_w.opaque_handle();
  malformed.diagonal_sigma = d_sigma.opaque_handle();
  malformed.offdiagonals = 2;
  malformed.precision = precision;
  try { launch_polarization_update(malformed, execution); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "malformed second-offdiagonal launch was accepted");
  malformed = {};
  malformed.region = region;
  malformed.p = d_p.opaque_handle();
  malformed.p_prev = d_p_prev.opaque_handle();
  malformed.primary_w = d_w.opaque_handle();
  malformed.diagonal_sigma = d_sigma.opaque_handle();
  malformed.precision = static_cast<scalar_precision>(99);
  rejected = false;
  try { launch_polarization_update(malformed, execution); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "invalid polarization precision was accepted");
  std::cout << "device " << device << ": " << cases << " "
            << (sizeof(T) == sizeof(float) ? "f32" : "f64") << " polarization cases PASS\n";
}

int main() {
  try {
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA devices found");
    for (size_t i = 0; i < devices.size(); ++i) {
      check_device<float>(devices[i].id);
      check_device<double>(devices[i].id);
      std::cout << "device " << devices[i].id << " (" << devices[i].uuid
                << "): NVIDIA polarization kernels PASS\n";
    }
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_polarization_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
  std::cout << "nvidia_polarization_smoke: PASS\n";
  return 0;
}
