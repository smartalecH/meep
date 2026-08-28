/* Standalone numerical and shape coverage for compact NVIDIA DFT reductions. */

#include "backend/nvidia/nvidia_dft.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace meep::nvidia;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename MonitorT>
static std::complex<long double> reference_value(
    const std::vector<MonitorT> &left, const std::vector<MonitorT> &right, size_t points,
    size_t frequencies, size_t lane, dft_reduction_operation operation,
    std::complex<long double> weight, long double &sum_abs) {
  std::complex<long double> result(0.0L, 0.0L);
  sum_abs = 0.0L;
  const size_t work = operation == dft_reduction_operation::norm2 ? points * frequencies : points;
  for (size_t i = 0; i < work; ++i) {
    const size_t frequency = operation == dft_reduction_operation::norm2 ? i % frequencies : lane;
    const size_t point = operation == dft_reduction_operation::norm2 ? i / frequencies : i;
    const size_t element = point * frequencies + frequency;
    const std::complex<long double> a(left[2 * element], left[2 * element + 1]);
    std::complex<long double> value;
    if (operation == dft_reduction_operation::norm2)
      value = std::complex<long double>(std::norm(a), 0.0L);
    else {
      const std::complex<long double> b(right[2 * element], right[2 * element + 1]);
      value = weight * a * std::conj(b);
      if (operation == dft_reduction_operation::real_weighted_product)
        value = std::complex<long double>(value.real(), 0.0L);
    }
    result += value;
    sum_abs += std::abs(value);
  }
  return result;
}

template <typename MonitorT>
static void run_shape(int device, size_t points, size_t frequencies,
                      dft_reduction_operation operation, bool alias, int repetitions) {
  device_scope selected(device);
  stream execution;
  std::vector<MonitorT> left(2 * points * frequencies), right(left.size());
  for (size_t p = 0; p < points; ++p)
    for (size_t f = 0; f < frequencies; ++f) {
      const size_t e = p * frequencies + f;
      left[2 * e] = MonitorT(0.013 * double(p + 1) - 0.071 * double(f + 1));
      left[2 * e + 1] = MonitorT(-0.009 * double(p + 2) + 0.043 * double(f + 1));
      right[2 * e] = MonitorT(std::sin(0.017 * double(p + 3) * double(f + 1)));
      right[2 * e + 1] = MonitorT(std::cos(0.011 * double(p + 5) * double(f + 2)));
    }
  if (alias) right = left;

  const size_t result_count = operation == dft_reduction_operation::norm2 ? 1 : frequencies;
  const size_t work = operation == dft_reduction_operation::norm2 ? points * frequencies : points;
  const size_t blocks = std::min<size_t>(128, std::max<size_t>(1, (work + 255) / 256));
  device_buffer d_left(left.size() * sizeof(MonitorT), device);
  device_buffer d_right(right.size() * sizeof(MonitorT), device);
  device_buffer d_partials(result_count * blocks * 2 * sizeof(double), device);
  device_buffer d_result(result_count * 2 * sizeof(double), device);
  copy_host_to_device_async(d_left, 0, left.data(), d_left.size(), execution);
  copy_host_to_device_async(d_right, 0, right.data(), d_right.size(), execution);
  fill_byte_async(d_result, 0, 0, d_result.size(), execution);

  dft_reduction_launch launch = {};
  launch.left = d_left.opaque_handle();
  launch.right = operation == dft_reduction_operation::norm2
                     ? NULL
                     : (alias ? d_left.opaque_handle() : d_right.opaque_handle());
  launch.partials = d_partials.opaque_handle();
  launch.result = d_result.opaque_handle();
  launch.storage_points = points;
  launch.frequencies = frequencies;
  launch.base = 0;
  launch.counts[0] = points;
  launch.counts[1] = launch.counts[2] = 1;
  launch.strides[0] = 1;
  launch.strides[1] = launch.strides[2] = 0;
  launch.result_count = result_count;
  launch.blocks_per_lane = blocks;
  launch.weight_real = -0.37;
  launch.weight_imag = 0.19;
  launch.operation = operation;
  launch.monitor_precision = sizeof(MonitorT) == sizeof(float) ? scalar_precision::f32
                                                               : scalar_precision::f64;
  launch.accumulation_precision = scalar_precision::f64;
  for (int i = 0; i < repetitions; ++i)
    launch_dft_reduction(launch, execution);

  std::vector<double> observed(2 * result_count);
  copy_device_to_host_async(observed.data(), d_result, 0, d_result.size(), execution);
  execution.synchronize();
  const std::complex<long double> weight(-0.37L, 0.19L);
  for (size_t lane = 0; lane < result_count; ++lane) {
    long double sum_abs = 0.0L;
    const std::complex<long double> expected =
        reference_value(left, right, points, frequencies, lane, operation, weight, sum_abs) *
        static_cast<long double>(repetitions);
    const std::complex<long double> got(observed[2 * lane], observed[2 * lane + 1]);
    const long double tolerance = 5e-11L * (1.0L + sum_abs * repetitions);
    require(std::abs(got - expected) <= tolerance, "compact DFT reduction differs from reference");
  }
}

template <typename MonitorT> static void run_precision(int device) {
  const size_t shapes[] = {1, 7, 31, 32, 33, 255, 256, 257, 513, 4097};
  for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i) {
    const size_t frequencies = shapes[i] == 513 ? 5 : 3;
    run_shape<MonitorT>(device, shapes[i], frequencies, dft_reduction_operation::norm2, false, 1);
    run_shape<MonitorT>(device, shapes[i], frequencies,
                        dft_reduction_operation::complex_weighted_product, false, 1);
  }
  run_shape<MonitorT>(device, 257, 3, dft_reduction_operation::real_weighted_product, false, 1);
  run_shape<MonitorT>(device, 257, 3, dft_reduction_operation::complex_weighted_product, true, 1);
  run_shape<MonitorT>(device, 257, 3, dft_reduction_operation::complex_weighted_product, false, 2);
}

template <typename MonitorT> static void run_padded_region(int device) {
  device_scope selected(device);
  stream execution;
  const size_t points = 20, frequencies = 3;
  std::vector<MonitorT> values(2 * points * frequencies, MonitorT(1000));
  const size_t indices[] = {2, 3, 5, 6, 11, 12, 14, 15};
  long double expected = 0.0L;
  for (size_t i = 0; i < sizeof(indices) / sizeof(indices[0]); ++i)
    for (size_t f = 0; f < frequencies; ++f) {
      const size_t e = indices[i] * frequencies + f;
      values[2 * e] = MonitorT(0.125 * double(i + 1) + 0.03125 * double(f));
      values[2 * e + 1] = MonitorT(-0.0625 * double(i + 2) + 0.015625 * double(f));
      expected += static_cast<long double>(values[2 * e]) * values[2 * e] +
                  static_cast<long double>(values[2 * e + 1]) * values[2 * e + 1];
    }
  device_buffer input(values.size() * sizeof(MonitorT), device);
  device_buffer partials(2 * 16 * sizeof(double), device);
  device_buffer result(2 * sizeof(double), device);
  copy_host_to_device_async(input, 0, values.data(), input.size(), execution);
  fill_byte_async(result, 0, 0, result.size(), execution);
  dft_reduction_launch launch = {};
  launch.left = input.opaque_handle();
  launch.partials = partials.opaque_handle();
  launch.result = result.opaque_handle();
  launch.storage_points = points;
  launch.frequencies = frequencies;
  launch.base = 2;
  launch.counts[0] = launch.counts[1] = launch.counts[2] = 2;
  launch.strides[0] = 9;
  launch.strides[1] = 3;
  launch.strides[2] = 1;
  launch.result_count = 1;
  launch.blocks_per_lane = 1;
  launch.operation = dft_reduction_operation::norm2;
  launch.monitor_precision = sizeof(MonitorT) == sizeof(float) ? scalar_precision::f32
                                                               : scalar_precision::f64;
  launch.accumulation_precision = scalar_precision::f64;
  launch_dft_reduction(launch, execution);
  double observed[2] = {0.0, 0.0};
  copy_device_to_host_async(observed, result, 0, result.size(), execution);
  execution.synchronize();
  require(std::fabs(static_cast<long double>(observed[0]) - expected) <=
              5e-12L * (1.0L + expected),
          "padded compact DFT norm included sentinel storage");
  require(observed[1] == 0.0, "real compact DFT norm returned an imaginary component");
}

int main(int argc, char **argv) {
  try {
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA device available");
    if (argc == 2 && std::string(argv[1]) == "--profile-only") {
      run_shape<double>(0, 4097, 3, dft_reduction_operation::complex_weighted_product, false, 1);
      std::cout << "nvidia_dft_reduction_smoke: profile-only PASS\n";
      return 0;
    }
    run_precision<double>(0);
    run_precision<float>(0);
    run_padded_region<double>(0);
    run_padded_region<float>(0);
    std::cout << "nvidia_dft_reduction_smoke: PASS\n";
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_dft_reduction_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
}
