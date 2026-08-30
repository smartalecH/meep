/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_multilevel.hpp"
#include "backend/nvidia/runtime.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace meep::nvidia;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename T>
static void copy_to_device(device_buffer &target, const std::vector<T> &source,
                           stream &execution) {
  copy_host_to_device_async(target, 0, source.data(), source.size() * sizeof(T), execution);
}

template <typename T>
static void compare(const std::vector<T> &observed, const std::vector<T> &expected,
                    double tolerance, const char *what) {
  require(observed.size() == expected.size(), "multilevel result size differs");
  for (size_t i = 0; i < observed.size(); ++i) {
    const double error = std::fabs(double(observed[i]) - double(expected[i]));
    if (error > tolerance * (1.0 + std::fabs(double(expected[i])))) {
      std::cerr << what << "[" << i << "] observed=" << double(observed[i])
                << " expected=" << double(expected[i]) << " error=" << error << "\n";
      throw std::runtime_error("NVIDIA multilevel kernel differs from reference");
    }
  }
}

template <typename T>
static void check_precision(int device, scalar_precision precision) {
  const size_t points = 259;
  const uint32_t levels = 3, transitions = 2;
  std::vector<T> population(points * levels), expected_population;
  std::vector<T> gamma_inv = {T(1.02), T(-0.01), T(0.005), T(0.015), T(0.98), T(-0.004),
                              T(0.002), T(-0.006), T(1.01)};
  std::vector<T> gamma = {T(0.99), T(0.01), T(0.0), T(0.005), T(0.985), T(0.004),
                          T(0.0), T(0.006), T(0.9875)};
  std::vector<T> alpha = {T(-0.4), T(-0.3), T(0.1), T(0.2), T(0.3), T(0.1)};
  std::vector<T> gperpdt = {T(0.013), T(0.021)};
  std::vector<T> w[2], w_prev[2], p[2], p_prev[2], sigma[2];
  for (uint32_t t = 0; t < transitions; ++t) {
    w[t].resize(points);
    w_prev[t].resize(points);
    p[t].resize(points);
    p_prev[t].resize(points);
    sigma[t].resize(points);
  }
  for (size_t i = 0; i < points; ++i) {
    for (uint32_t l = 0; l < levels; ++l)
      population[i * levels + l] = T(0.11 + 0.003 * l + 0.0002 * i);
    for (uint32_t t = 0; t < transitions; ++t) {
      w[t][i] = T(0.2 + 0.0007 * i + 0.01 * t);
      w_prev[t][i] = T(-0.08 + 0.0003 * i - 0.005 * t);
      p[t][i] = T(0.03 + 0.00011 * i + 0.004 * t);
      p_prev[t][i] = T(-0.02 + 0.00009 * i - 0.003 * t);
      sigma[t][i] = T(0.7 + 0.0002 * i + 0.05 * t);
    }
  }
  expected_population = population;
  std::vector<T> expected_p[2] = {p[0], p[1]};
  std::vector<T> expected_p_prev[2] = {p_prev[0], p_prev[1]};

  for (size_t i = 0; i < points; ++i) {
    T temporary[levels];
    for (uint32_t l1 = 0; l1 < levels; ++l1) {
      temporary[l1] = T(0);
      for (uint32_t l2 = 0; l2 < levels; ++l2)
        temporary[l1] += gamma[l1 * levels + l2] * population[i * levels + l2];
    }
    for (uint32_t t = 0; t < transitions; ++t) {
      const T e8 = w[t][i] + w[t][i] + w[t][i] + w[t][i] +
                   w_prev[t][i] + w_prev[t][i] + w_prev[t][i] + w_prev[t][i];
      const T ps = p[t][i] + p[t][i] + p[t][i] + p[t][i];
      const T pps = p_prev[t][i] + p_prev[t][i] + p_prev[t][i] + p_prev[t][i];
      const T edp32 = (ps - pps) * e8 * T(0.03125);
      const T epave64 = (ps + pps) * e8 * T(0.015625);
      for (uint32_t l = 0; l < levels; ++l)
        temporary[l] += alpha[l * transitions + t] * edp32 +
                        alpha[l * transitions + t] * gperpdt[t] * epave64;
    }
    for (uint32_t l1 = 0; l1 < levels; ++l1) {
      T next = T(0);
      for (uint32_t l2 = 0; l2 < levels; ++l2)
        next += gamma_inv[l1 * levels + l2] * temporary[l2];
      expected_population[i * levels + l1] = next;
    }
  }

  stream execution;
  device_buffer d_population(population.size() * sizeof(T), device);
  device_buffer d_gamma_inv(gamma_inv.size() * sizeof(T), device);
  device_buffer d_gamma(gamma.size() * sizeof(T), device);
  device_buffer d_alpha(alpha.size() * sizeof(T), device);
  device_buffer d_gperpdt(gperpdt.size() * sizeof(T), device);
  device_buffer d_scratch(population.size() * sizeof(T), device);
  device_buffer d_w[2], d_w_prev[2], d_p[2], d_p_prev[2], d_sigma[2];
  copy_to_device(d_population, population, execution);
  copy_to_device(d_gamma_inv, gamma_inv, execution);
  copy_to_device(d_gamma, gamma, execution);
  copy_to_device(d_alpha, alpha, execution);
  copy_to_device(d_gperpdt, gperpdt, execution);
  std::vector<multilevel_population_term_launch> terms(transitions);
  for (uint32_t t = 0; t < transitions; ++t) {
    d_w[t].allocate(points * sizeof(T), device);
    d_w_prev[t].allocate(points * sizeof(T), device);
    d_p[t].allocate(points * sizeof(T), device);
    d_p_prev[t].allocate(points * sizeof(T), device);
    d_sigma[t].allocate(points * sizeof(T), device);
    copy_to_device(d_w[t], w[t], execution);
    copy_to_device(d_w_prev[t], w_prev[t], execution);
    copy_to_device(d_p[t], p[t], execution);
    copy_to_device(d_p_prev[t], p_prev[t], execution);
    copy_to_device(d_sigma[t], sigma[t], execution);
    terms[t] = {d_w[t].opaque_handle(), d_w_prev[t].opaque_handle(),
                d_p[t].opaque_handle(), d_p_prev[t].opaque_handle(), {0, 0}, t};
  }
  device_buffer d_terms(terms.size() * sizeof(terms[0]), device);
  copy_to_device(d_terms, terms, execution);

  multilevel_population_launch population_launch = {};
  population_launch.region.counts[0] = 1;
  population_launch.region.counts[1] = 1;
  population_launch.region.counts[2] = points;
  population_launch.region.strides[0] = points;
  population_launch.region.strides[1] = points;
  population_launch.region.strides[2] = 1;
  population_launch.populations = d_population.opaque_handle();
  population_launch.gamma_inv = d_gamma_inv.opaque_handle();
  population_launch.gamma_matrix = d_gamma.opaque_handle();
  population_launch.alpha = d_alpha.opaque_handle();
  population_launch.transition_gperpdt = d_gperpdt.opaque_handle();
  population_launch.terms = static_cast<const multilevel_population_term_launch *>(
      d_terms.opaque_handle());
  population_launch.scratch = d_scratch.opaque_handle();
  population_launch.term_count = transitions;
  population_launch.levels = levels;
  population_launch.transitions = transitions;
  population_launch.precision = precision;
  launch_multilevel_population(population_launch, execution);

  const T transition_coefficients[2][5] = {
      {T(0.31), T(0.96), T(0.94), T(0.015), T(0.8)},
      {T(0.47), T(0.93), T(0.91), T(0.015), T(1.1)}};
  device_buffer d_transition_coefficients[2];
  for (uint32_t t = 0; t < transitions; ++t) {
    for (size_t i = 1; i + 2 < points; ++i) {
      const size_t base = i * levels;
      const T inversion = T(0.25) *
                          (expected_population[base + 2] +
                           expected_population[base - levels + 2] +
                           expected_population[base + 2 * levels + 2] +
                           expected_population[base + levels + 2] -
                           expected_population[base + 0] -
                           expected_population[base - levels + 0] -
                           expected_population[base + 2 * levels + 0] -
                           expected_population[base + levels + 0]);
      const T current = expected_p[t][i];
      expected_p[t][i] = transition_coefficients[t][1] *
                         (current * (T(2) - transition_coefficients[t][0]) -
                          transition_coefficients[t][2] * expected_p_prev[t][i] -
                          transition_coefficients[t][3] *
                              (transition_coefficients[t][4] * sigma[t][i] * w[t][i]) *
                              inversion);
      expected_p_prev[t][i] = current;
    }
    d_transition_coefficients[t].allocate(sizeof(transition_coefficients[t]), device);
    copy_host_to_device_async(d_transition_coefficients[t], 0, transition_coefficients[t],
                              sizeof(transition_coefficients[t]), execution);
    multilevel_transition_launch transition_launch = {};
    transition_launch.region = population_launch.region;
    transition_launch.region.base = 1;
    transition_launch.region.counts[2] = points - 3;
    transition_launch.p = d_p[t].opaque_handle();
    transition_launch.p_prev = d_p_prev[t].opaque_handle();
    transition_launch.w = d_w[t].opaque_handle();
    transition_launch.diagonal_sigma = d_sigma[t].opaque_handle();
    transition_launch.populations = d_population.opaque_handle();
    transition_launch.coefficients = d_transition_coefficients[t].opaque_handle();
    transition_launch.population_stride = levels;
    transition_launch.population_offsets[0] = -ptrdiff_t(levels);
    transition_launch.population_offsets[1] = 2 * ptrdiff_t(levels);
    /* alpha has multiple positive entries for transition 1; the CPU's
       ascending scan makes level 2 authoritative (last-sign-wins). */
    transition_launch.positive_level = 2;
    transition_launch.negative_level = 0;
    transition_launch.precision = precision;
    launch_multilevel_transition(transition_launch, execution);
  }

  std::vector<T> observed_population(population.size()), observed_p[2], observed_p_prev[2];
  copy_device_to_host_async(observed_population.data(), d_population, 0,
                            observed_population.size() * sizeof(T), execution);
  for (uint32_t t = 0; t < transitions; ++t) {
    observed_p[t].resize(points);
    observed_p_prev[t].resize(points);
    copy_device_to_host_async(observed_p[t].data(), d_p[t], 0, points * sizeof(T), execution);
    copy_device_to_host_async(observed_p_prev[t].data(), d_p_prev[t], 0,
                              points * sizeof(T), execution);
  }
  execution.synchronize();
  const double tolerance = precision == scalar_precision::f32 ? 3e-6 : 2e-13;
  compare(observed_population, expected_population, tolerance, "population");
  for (uint32_t t = 0; t < transitions; ++t) {
    compare(observed_p[t], expected_p[t], tolerance, "P");
    compare(observed_p_prev[t], expected_p_prev[t], tolerance, "P_prev");
  }

  std::vector<T> zero_row = population;
  copy_to_device(d_population, zero_row, execution);
  population_launch.term_count = 0;
  population_launch.terms = NULL;
  launch_multilevel_population(population_launch, execution);
  std::vector<T> expected_zero = zero_row;
  for (size_t i = 0; i < points; ++i) {
    T temporary[levels];
    for (uint32_t l1 = 0; l1 < levels; ++l1) {
      temporary[l1] = T(0);
      for (uint32_t l2 = 0; l2 < levels; ++l2)
        temporary[l1] += gamma[l1 * levels + l2] * zero_row[i * levels + l2];
    }
    for (uint32_t l1 = 0; l1 < levels; ++l1) {
      T next = T(0);
      for (uint32_t l2 = 0; l2 < levels; ++l2)
        next += gamma_inv[l1 * levels + l2] * temporary[l2];
      expected_zero[i * levels + l1] = next;
    }
  }
  copy_device_to_host_async(observed_population.data(), d_population, 0,
                            observed_population.size() * sizeof(T), execution);
  execution.synchronize();
  compare(observed_population, expected_zero, tolerance, "zero-row population");

  bool rejected = false;
  try {
    multilevel_population_launch malformed = population_launch;
    malformed.scratch = NULL;
    launch_multilevel_population(malformed, execution);
  }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "malformed multilevel population launch was accepted");
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
                << "): NVIDIA multilevel kernels PASS\n";
    }
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_multilevel_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
  std::cout << "nvidia_multilevel_smoke: PASS\n";
  return 0;
}
