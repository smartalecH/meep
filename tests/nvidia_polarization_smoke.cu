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

template <typename T>
static double gyrotropic_centered_host(const std::vector<T> &field, ptrdiff_t i,
                                       ptrdiff_t primary_stride, ptrdiff_t cross_stride) {
  return 0.25 * (field[i] + field[i - cross_stride] + field[i + primary_stride] +
                 field[i + primary_stride - cross_stride]);
}

template <typename T>
static void fill_gyro_coefficients(gyrotropic_update_launch &launch, int primary,
                                   gyrotropic_kernel_model model) {
  T bias[3] = {T(0.17), T(-0.23), T(0.31)};
  if (model == gyrotropic_kernel_model::saturated) {
    const T norm = std::sqrt(bias[0] * bias[0] + bias[1] * bias[1] + bias[2] * bias[2]);
    for (int i = 0; i < 3; ++i) bias[i] /= norm;
  }
  T global[3][3] = {};
  global[0][1] = bias[2];
  global[1][0] = -bias[2];
  global[1][2] = bias[0];
  global[2][1] = -bias[0];
  global[2][0] = bias[1];
  global[0][2] = -bias[1];
  const int order[3] = {primary, (primary + 1) % 3, (primary + 2) % 3};
  const T dt = T(0.017), omega0 = T(0.73), gamma0 = T(0.06), alpha = T(0.19);
  const T omega = 2 * 3.14159265358979323846 * omega0 * dt;
  const T gamma = 2 * 3.14159265358979323846 * gamma0 * dt;
  T gd, gx, gy, gz;
  launch.omega = double(omega);
  launch.gamma = double(gamma);
  launch.alpha = double(alpha);
  launch.dt2pi = double(T(2 * 3.14159265358979323846 * dt));
  if (model == gyrotropic_kernel_model::saturated) {
    gd = 0.5;
    gx = -0.5 * alpha * global[1][2];
    gy = -0.5 * alpha * global[2][0];
    gz = -0.5 * alpha * global[0][1];
  }
  else {
    const T a = omega * omega;
    launch.omega0dtsqr = double(a);
    launch.gamma1 = double(T(1) - gamma / T(2));
    launch.diagonal = double(T(2) -
                             (model == gyrotropic_kernel_model::drude ? T(0) : a));
    launch.pt = double(T(3.14159265358979323846 * dt));
    gd = T(1) + gamma / T(2);
    gx = T(launch.pt) * global[1][2];
    gy = T(launch.pt) * global[2][0];
    gz = T(launch.pt) * global[0][1];
  }
  const T invdet = 1.0 / gd / (gd * gd + gx * gx + gy * gy + gz * gz);
  const T inverse[3][3] = {
      {invdet * (gd * gd + gx * gx), invdet * (gx * gy + gd * gz),
       invdet * (gx * gz - gd * gy)},
      {invdet * (gy * gx - gd * gz), invdet * (gd * gd + gy * gy),
       invdet * (gy * gz + gd * gx)},
      {invdet * (gz * gx + gd * gy), invdet * (gz * gy - gd * gx),
       invdet * (gd * gd + gz * gz)}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      launch.gyro[i][j] = double(global[order[i]][order[j]]);
      launch.inverse[i][j] = double(inverse[order[i]][order[j]]);
    }
  launch.model = model;
}

template <typename T> static void check_gyrotropic_device(int device) {
  device_scope selected(device);
  stream execution;
  const size_t elements = 320, bytes = elements * sizeof(T);
  std::vector<T> p[3], p_prev[3], expected[3], expected_prev[3];
  std::vector<T> w[3], sigma(elements);
  for (int d = 0; d < 3; ++d) {
    p[d].resize(elements);
    p_prev[d].resize(elements);
    w[d].resize(elements);
    for (size_t i = 0; i < elements; ++i) {
      p[d][i] = T(0.11 * (d + 1) + 0.0007 * double(i));
      p_prev[d][i] = T(-0.07 * (d + 1) + 0.0003 * double(i));
      w[d][i] = T(0.23 * (d + 1) - 0.0005 * double(i));
    }
  }
  for (size_t i = 0; i < elements; ++i) sigma[i] = T(0.8 + 0.0002 * double(i));
  device_buffer d_p[3], d_pp[3], d_w[3];
  for (int d = 0; d < 3; ++d) {
    d_p[d].allocate(bytes, device);
    d_pp[d].allocate(bytes, device);
    d_w[d].allocate(bytes, device);
    copy_host_to_device_async(d_w[d], 0, w[d].data(), bytes, execution);
  }
  device_buffer d_sigma(bytes, device);
  copy_host_to_device_async(d_sigma, 0, sigma.data(), bytes, execution);

  flat_region region = {};
  region.base = 30;
  region.counts[0] = region.counts[1] = 1;
  region.counts[2] = 257;
  region.strides[2] = 1;
  const scalar_precision precision =
      sizeof(T) == sizeof(float) ? scalar_precision::f32 : scalar_precision::f64;
  size_t cases = 0;
  for (int primary = 0; primary < 3; ++primary)
    for (int model_index = 0; model_index < 3; ++model_index) {
      const gyrotropic_kernel_model model = static_cast<gyrotropic_kernel_model>(model_index);
      const bool magnetic = ((primary + model_index) & 1) != 0;
      const bool have_w1 = model_index != 1;
      const bool have_w2 = model_index != 0;
      for (int d = 0; d < 3; ++d) {
        expected[d] = p[d];
        expected_prev[d] = p_prev[d];
        copy_host_to_device_async(d_p[d], 0, p[d].data(), bytes, execution);
        copy_host_to_device_async(d_pp[d], 0, p_prev[d].data(), bytes, execution);
      }
      gyrotropic_update_launch launch = {};
      launch.region = region;
      launch.precision = precision;
      launch.primary_stride = magnetic ? -2 : 2;
      launch.cross_stride1 = magnetic ? -5 : 5;
      launch.cross_stride2 = magnetic ? -7 : 7;
      for (int d = 0; d < 3; ++d) {
        launch.p[d] = d_p[d].opaque_handle();
        launch.p_prev[d] = d_pp[d].opaque_handle();
      }
      launch.w[0] = d_w[0].opaque_handle();
      launch.w[1] = have_w1 ? d_w[1].opaque_handle() : NULL;
      launch.w[2] = have_w2 ? d_w[2].opaque_handle() : NULL;
      launch.sigma = d_sigma.opaque_handle();
      fill_gyro_coefficients<T>(launch, primary, model);

      for (size_t linear = 0; linear < region.counts[2]; ++linear) {
        const ptrdiff_t i = ptrdiff_t(region.base + linear);
        const T old0 = expected[0][i], old1 = expected[1][i], old2 = expected[2][i];
        const T prev0 = expected_prev[0][i], prev1 = expected_prev[1][i],
                prev2 = expected_prev[2][i];
        const double c1 = have_w1 ? gyrotropic_centered_host(
                                          w[1], i, launch.primary_stride, launch.cross_stride1)
                                  : 0.0;
        const double c2 = have_w2 ? gyrotropic_centered_host(
                                          w[2], i, launch.primary_stride, launch.cross_stride2)
                                  : 0.0;
        T r0, r1, r2;
        if (model == gyrotropic_kernel_model::saturated) {
          const T q0 = -T(launch.omega) * old0 + 0.5 * T(launch.alpha) * prev0 +
                       T(launch.dt2pi) * sigma[i] * w[0][i];
          const T q1 = -T(launch.omega) * old1 + 0.5 * T(launch.alpha) * prev1 +
                       T(launch.dt2pi) * sigma[i] * c1;
          const T q2 = -T(launch.omega) * old2 + 0.5 * T(launch.alpha) * prev2 +
                       T(launch.dt2pi) * sigma[i] * c2;
          r0 = 0.5 * prev0 - T(launch.gamma) * old0 + T(launch.gyro[0][1]) * q1 +
               T(launch.gyro[0][2]) * q2;
          r1 = 0.5 * prev1 - T(launch.gamma) * old1 + T(launch.gyro[1][2]) * q2 +
               T(launch.gyro[1][0]) * q0;
          r2 = 0.5 * prev2 - T(launch.gamma) * old2 + T(launch.gyro[2][0]) * q0 +
               T(launch.gyro[2][1]) * q1;
        }
        else {
          r0 = T(launch.diagonal) * old0 - T(launch.gamma1) * prev0 +
               T(launch.omega0dtsqr) * sigma[i] * w[0][i] -
               T(launch.pt) * T(launch.gyro[0][1]) * prev1 -
               T(launch.pt) * T(launch.gyro[0][2]) * prev2;
          r1 = T(launch.diagonal) * old1 - T(launch.gamma1) * prev1 +
               T(launch.omega0dtsqr) * sigma[i] * c1 -
               T(launch.pt) * T(launch.gyro[1][0]) * prev0 -
               T(launch.pt) * T(launch.gyro[1][2]) * prev2;
          r2 = T(launch.diagonal) * old2 - T(launch.gamma1) * prev2 +
               T(launch.omega0dtsqr) * sigma[i] * c2 -
               T(launch.pt) * T(launch.gyro[2][1]) * prev1 -
               T(launch.pt) * T(launch.gyro[2][0]) * prev0;
        }
        expected_prev[0][i] = old0;
        expected_prev[1][i] = old1;
        expected_prev[2][i] = old2;
        for (int d = 0; d < 3; ++d)
          expected[d][i] = T(launch.inverse[d][0]) * r0 + T(launch.inverse[d][1]) * r1 +
                           T(launch.inverse[d][2]) * r2;
      }

      launch_gyrotropic_update(launch, execution);
      const double tolerance = 16 * std::numeric_limits<T>::epsilon();
      for (int d = 0; d < 3; ++d) {
        std::vector<T> observed(elements), observed_prev(elements);
        copy_device_to_host_async(observed.data(), d_p[d], 0, bytes, execution);
        copy_device_to_host_async(observed_prev.data(), d_pp[d], 0, bytes, execution);
        execution.synchronize();
        for (size_t i = 0; i < elements; ++i) {
          require((sizeof(T) == sizeof(float) && observed[i] == expected[d][i]) ||
                      (sizeof(T) != sizeof(float) &&
                       std::fabs(double(observed[i] - expected[d][i])) <=
                           tolerance * (1 + std::fabs(double(expected[d][i])))),
                  "gyrotropic recurrence or sentinel differs");
          require(std::fabs(double(observed_prev[i] - expected_prev[d][i])) <=
                      tolerance * (1 + std::fabs(double(expected_prev[d][i]))),
                  "gyrotropic P_prev rotation or sentinel differs");
        }
      }
      ++cases;
    }

  gyrotropic_update_launch malformed = {};
  malformed.region = region;
  malformed.precision = precision;
  malformed.primary_stride = 2;
  malformed.cross_stride1 = 5;
  malformed.cross_stride2 = 7;
  for (int d = 0; d < 3; ++d) {
    malformed.p[d] = d_p[d].opaque_handle();
    malformed.p_prev[d] = d_pp[d].opaque_handle();
    malformed.w[d] = d_w[d].opaque_handle();
  }
  malformed.sigma = d_sigma.opaque_handle();
  fill_gyro_coefficients<T>(malformed, 0, gyrotropic_kernel_model::lorentzian);
  const auto rejected = [&](const gyrotropic_update_launch &candidate) {
    try {
      launch_gyrotropic_update(candidate, execution);
      return false;
    }
    catch (const std::invalid_argument &) { return true; }
    catch (const std::overflow_error &) { return true; }
  };
  gyrotropic_update_launch bad = malformed;
  bad.p_prev[2] = NULL;
  require(rejected(bad), "gyrotropic launch accepted missing state");
  bad = malformed;
  bad.p_prev[2] = bad.p[0];
  require(rejected(bad), "gyrotropic launch accepted aliased state");
  bad = malformed;
  bad.model = static_cast<gyrotropic_kernel_model>(99);
  require(rejected(bad), "gyrotropic launch accepted invalid model");
  bad = malformed;
  bad.precision = static_cast<scalar_precision>(99);
  require(rejected(bad), "gyrotropic launch accepted invalid precision");
  bad = malformed;
  bad.region.counts[2] = 0;
  require(rejected(bad), "gyrotropic launch accepted an empty region");
  bad = malformed;
  bad.region.counts[0] = std::numeric_limits<size_t>::max();
  bad.region.counts[1] = 2;
  require(rejected(bad), "gyrotropic launch accepted a region-size overflow");
  std::cout << "device " << device << ": " << cases << " "
            << (sizeof(T) == sizeof(float) ? "f32" : "f64") << " gyrotropic cases PASS\n";
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

int main(int argc, char **argv) {
  try {
    const bool gyro_only = argc == 2 && std::string(argv[1]) == "--gyro-only";
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA devices found");
    for (size_t i = 0; i < devices.size(); ++i) {
      if (!gyro_only) {
        check_device<float>(devices[i].id);
        check_device<double>(devices[i].id);
      }
      check_gyrotropic_device<float>(devices[i].id);
      check_gyrotropic_device<double>(devices[i].id);
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
