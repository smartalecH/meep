/* Low-level CUDA coverage for resident solve_cw vector primitives. */

#include "backend/nvidia/nvidia_cw.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace meep::nvidia;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename F> static void require_throws(F operation, const char *message) {
  bool threw = false;
  try {
    operation();
  }
  catch (const std::exception &) {
    threw = true;
  }
  require(threw, message);
}

template <typename T> static bool same_bits(T a, T b) {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <typename T>
static void run_precision(int device, scalar_precision precision) {
  device_scope selected(device);
  stream execution;
  const size_t points = 257;
  const size_t field_elements = 2 * points + 9;
  const size_t real_elements = 2 * (points + 3) + 5;
  std::vector<T> real_values(field_elements), imaginary_values(field_elements);
  for (size_t i = 0; i < field_elements; ++i) {
    real_values[i] = T(0.125 * double(i) - 3.0);
    imaginary_values[i] = T(2.0 - 0.0625 * double(i));
  }
  device_buffer d_real(field_elements * sizeof(T), device);
  device_buffer d_imag(field_elements * sizeof(T), device);
  device_buffer d_vector(real_elements * sizeof(T), device);
  copy_host_to_device_async(d_real, 0, real_values.data(), d_real.size(), execution);
  copy_host_to_device_async(d_imag, 0, imaginary_values.data(), d_imag.size(), execution);
  std::vector<T> packed(real_elements, T(-9));
  copy_host_to_device_async(d_vector, 0, packed.data(), d_vector.size(), execution);

  cw_state_row_launch row = {};
  row.region.base = 4;
  row.region.counts[0] = points;
  row.region.counts[1] = row.region.counts[2] = 1;
  row.region.strides[0] = 2;
  row.region.strides[1] = row.region.strides[2] = 1;
  row.real_values = d_real.opaque_handle();
  row.imaginary_values = d_imag.opaque_handle();
  row.complex_offset = 3;
  row.complex_count = points;
  row.precision = precision;
  launch_cw_pack(row, d_vector.opaque_handle(), real_elements, execution);
  copy_device_to_host_async(packed.data(), d_vector, 0, d_vector.size(), execution);
  execution.synchronize();
  for (size_t i = 0; i < points; ++i) {
    require(same_bits(packed[2 * (3 + i)], real_values[4 + 2 * i]),
            "solve_cw pack real value differs");
    require(same_bits(packed[2 * (3 + i) + 1], imaginary_values[4 + 2 * i]),
            "solve_cw pack imaginary value differs");
  }
  for (size_t i = 0; i < 2 * row.complex_offset; ++i)
    require(packed[i] == T(-9), "solve_cw pack changed prefix padding");
  for (size_t i = 2 * (row.complex_offset + row.complex_count); i < packed.size(); ++i)
    require(packed[i] == T(-9), "solve_cw pack changed suffix padding");

  for (size_t i = 0; i < points; ++i) {
    packed[2 * (3 + i)] = T(10.0 + 0.25 * double(i));
    packed[2 * (3 + i) + 1] = T(-7.0 - 0.125 * double(i));
  }
  copy_host_to_device_async(d_vector, 0, packed.data(), d_vector.size(), execution);
  const std::vector<T> real_before = real_values, imaginary_before = imaginary_values;
  launch_cw_unpack(row, d_vector.opaque_handle(), real_elements, execution);
  copy_device_to_host_async(real_values.data(), d_real, 0, d_real.size(), execution);
  copy_device_to_host_async(imaginary_values.data(), d_imag, 0, d_imag.size(), execution);
  execution.synchronize();
  for (size_t i = 0; i < points; ++i) {
    require(same_bits(real_values[4 + 2 * i], packed[2 * (3 + i)]),
            "solve_cw unpack real value differs");
    require(same_bits(imaginary_values[4 + 2 * i], packed[2 * (3 + i) + 1]),
            "solve_cw unpack imaginary value differs");
  }
  for (size_t i = 0; i < field_elements; ++i) {
    if (i >= 4 && i <= 4 + 2 * (points - 1) && ((i - 4) % 2 == 0)) continue;
    require(same_bits(real_values[i], real_before[i]),
            "solve_cw unpack changed a real value outside the row");
    require(same_bits(imaginary_values[i], imaginary_before[i]),
            "solve_cw unpack changed an imaginary value outside the row");
  }

  launch_cw_zero(cw_zero_launch{d_real.opaque_handle(), field_elements, precision}, execution);
  copy_device_to_host_async(real_values.data(), d_real, 0, d_real.size(), execution);
  execution.synchronize();
  for (T value : real_values) require(value == T(0), "solve_cw zero left a nonzero value");

  std::vector<T> source_real(3, T(0)), source_imag(3, T(0)), condinv(3, T(1));
  source_real[1] = T(0.08718129992485046);
  device_buffer d_source_real(source_real.size() * sizeof(T), device);
  device_buffer d_source_imag(source_imag.size() * sizeof(T), device);
  device_buffer d_condinv(condinv.size() * sizeof(T), device);
  device_buffer d_point(sizeof(source_point), device);
  device_buffer d_scalar(sizeof(source_scalar), device);
  const source_point point = {1, 1.0, 0.0};
  const source_scalar scalar = {2.179294922780449e-06, 0.0, 99.0, 99.0};
  copy_host_to_device_async(d_source_real, 0, source_real.data(), d_source_real.size(), execution);
  copy_host_to_device_async(d_source_imag, 0, source_imag.data(), d_source_imag.size(), execution);
  copy_host_to_device_async(d_condinv, 0, condinv.data(), d_condinv.size(), execution);
  copy_host_to_device_async(d_point, 0, &point, sizeof(point), execution);
  copy_host_to_device_async(d_scalar, 0, &scalar, sizeof(scalar), execution);
  cw_source_batch_launch source = {d_source_real.opaque_handle(), d_source_imag.opaque_handle(),
                                   d_condinv.opaque_handle(),
                                   static_cast<const source_point *>(d_point.opaque_handle()),
                                   0, 1, 0, 1.0, false, precision};
  launch_cw_source_batch(source, d_scalar.opaque_handle(), execution);
  copy_device_to_host_async(source_real.data(), d_source_real, 0, d_source_real.size(), execution);
  execution.synchronize();
  require(same_bits(source_real[1], T(double(T(0.08718129992485046)) -
                                      2.179294922780449e-06)),
          "solve_cw source subtraction differs from CPU compound assignment");

  const size_t n = 257;
  std::vector<T> x(n), y(n), output(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = T(0.2 * double(i) - 11.0);
    y[i] = T(3.0 - 0.07 * double(i));
  }
  device_buffer d_x(n * sizeof(T), device), d_y(n * sizeof(T), device),
      d_output(n * sizeof(T), device);
  copy_host_to_device_async(d_x, 0, x.data(), d_x.size(), execution);
  copy_host_to_device_async(d_y, 0, y.data(), d_y.size(), execution);

  cw_vector_launch vector = {d_output.opaque_handle(), d_x.opaque_handle(), d_y.opaque_handle(),
                             n, -0.123456789012345, precision,
                             cw_vector_operation::subtract_field};
  launch_cw_vector(vector, execution);
  copy_device_to_host_async(output.data(), d_output, 0, d_output.size(), execution);
  execution.synchronize();
  for (size_t i = 0; i < n; ++i)
    require(same_bits(output[i], T(x[i] - y[i])), "solve_cw T subtraction differs");

  vector.y = NULL;
  vector.operation = cw_vector_operation::scale_field_coefficient;
  launch_cw_vector(vector, execution);
  copy_device_to_host_async(output.data(), d_output, 0, d_output.size(), execution);
  execution.synchronize();
  const T narrowed = T(vector.coefficient);
  for (size_t i = 0; i < n; ++i)
    require(same_bits(output[i], T(x[i] * narrowed)), "solve_cw T-coefficient scale differs");

  vector.y = d_y.opaque_handle();
  vector.operation = cw_vector_operation::linear_f64_coefficient;
  launch_cw_vector(vector, execution);
  copy_device_to_host_async(output.data(), d_output, 0, d_output.size(), execution);
  execution.synchronize();
  for (size_t i = 0; i < n; ++i)
    require(same_bits(output[i], T(double(x[i]) + vector.coefficient * double(y[i]))),
            "solve_cw f64-coefficient linear update differs");

  vector.y = NULL;
  vector.coefficient = 0.0;
  vector.operation = cw_vector_operation::copy;
  launch_cw_vector(vector, execution);
  copy_device_to_host_async(output.data(), d_output, 0, d_output.size(), execution);
  execution.synchronize();
  for (size_t i = 0; i < n; ++i)
    require(same_bits(output[i], x[i]), "solve_cw copy differs");

  cw_operator_launch op = {d_output.opaque_handle(), d_y.opaque_handle(), d_x.opaque_handle(),
                           n - 1, 3.141592653589793, -0.2718281828459045,
                           0.1618033988749895, precision};
  launch_cw_operator_finalize(op, execution);
  copy_device_to_host_async(output.data(), d_output, 0, d_output.size(), execution);
  execution.synchronize();
  const T dt_inverse = T(op.dt_inverse), wr = T(op.iomega_real), wi = T(op.iomega_imaginary);
  for (size_t i = 0; i < op.real_elements / 2; ++i) {
    const size_t r = 2 * i;
    const T expected_real = T(T(y[r] - x[r]) * dt_inverse + T(wr * x[r] - wi * x[r + 1]));
    const T expected_imag =
        T(T(y[r + 1] - x[r + 1]) * dt_inverse + T(wr * x[r + 1] + wi * x[r]));
    require(same_bits(output[r], expected_real) && same_bits(output[r + 1], expected_imag),
            "solve_cw operator expression differs");
  }

  const size_t blocks = 2;
  device_buffer partials(blocks * sizeof(double), device), result(sizeof(double), device);
  double observed = 0.0;
  cw_reduction_launch reduction = {d_x.opaque_handle(), d_y.opaque_handle(),
                                   partials.opaque_handle(), result.opaque_handle(), n, blocks,
                                   1.0, precision};
  launch_cw_dot(reduction, execution);
  copy_device_to_host_async(&observed, result, 0, sizeof(observed), execution);
  execution.synchronize();
  long double expected_long = 0.0L;
  for (size_t i = 0; i < n; ++i) expected_long += static_cast<long double>(T(x[i] * y[i]));
  double expected = double(expected_long);
  require(std::fabs(observed - expected) <=
              4e-13 * (1.0 + double(std::fabs(expected_long))),
          "solve_cw dot reduction differs");
  uint64_t repeat_bits = 0;
  std::memcpy(&repeat_bits, &observed, sizeof(repeat_bits));
  for (int repeat = 0; repeat < 20; ++repeat) {
    launch_cw_dot(reduction, execution);
    copy_device_to_host_async(&observed, result, 0, sizeof(observed), execution);
    execution.synchronize();
    uint64_t bits = 0;
    std::memcpy(&bits, &observed, sizeof(bits));
    require(bits == repeat_bits, "solve_cw dot reduction is not bit deterministic");
  }

  reduction.y = NULL;
  launch_cw_max_abs(reduction, execution);
  copy_device_to_host_async(&observed, result, 0, sizeof(observed), execution);
  execution.synchronize();
  expected = 0.0;
  for (T value : x) expected = std::max(expected, std::fabs(double(value)));
  require(observed == expected, "solve_cw max reduction differs");

  x[17] = std::numeric_limits<T>::quiet_NaN();
  copy_host_to_device_async(d_x, 0, x.data(), d_x.size(), execution);
  launch_cw_max_abs(reduction, execution);
  copy_device_to_host_async(&observed, result, 0, sizeof(observed), execution);
  execution.synchronize();
  require(std::isinf(observed), "solve_cw max reduction suppressed a nonfinite input");
  x[17] = T(0.2 * 17.0 - 11.0);
  copy_host_to_device_async(d_x, 0, x.data(), d_x.size(), execution);

  reduction.scale = 1.0 / expected;
  launch_cw_scaled_norm_sum(reduction, execution);
  copy_device_to_host_async(&observed, result, 0, sizeof(observed), execution);
  execution.synchronize();
  expected = 0.0;
  for (T value : x) {
    const double scaled = reduction.scale * double(value);
    expected += scaled * scaled;
  }
  require(std::fabs(observed - expected) <= 2e-12 * (1.0 + expected),
          "solve_cw scaled norm reduction differs");

  cw_state_row_launch invalid_row = row;
  invalid_row.complex_count = 0;
  require_throws([&]() { launch_cw_pack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw pack accepted an empty row");
  invalid_row = row;
  invalid_row.complex_offset = std::numeric_limits<size_t>::max();
  require_throws([&]() { launch_cw_pack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw pack accepted an overflowing vector range");
  vector.x = vector.y;
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted aliased inputs");
  vector.x = d_x.opaque_handle();
  vector.output = d_output.opaque_handle();
  vector.precision = static_cast<scalar_precision>(99);
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted invalid precision");
  vector.precision = precision;
  vector.elements = 0;
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted an empty extent");

  const size_t tail_lengths[] = {1, 7, 31, 32, 33, 255, 256, 257, 511, 512, 513, 4097};
  const size_t tail_capacity = 4097;
  std::vector<T> tail_x(tail_capacity), tail_y(tail_capacity), tail_out(tail_capacity);
  for (size_t i = 0; i < tail_capacity; ++i) {
    tail_x[i] = T(0.03125 * double(i) - 4.0);
    tail_y[i] = T(1.5 - 0.015625 * double(i));
  }
  device_buffer d_tail_x(tail_capacity * sizeof(T), device);
  device_buffer d_tail_y(tail_capacity * sizeof(T), device);
  device_buffer d_tail_out(tail_capacity * sizeof(T), device);
  copy_host_to_device_async(d_tail_x, 0, tail_x.data(), d_tail_x.size(), execution);
  copy_host_to_device_async(d_tail_y, 0, tail_y.data(), d_tail_y.size(), execution);
  for (size_t tail : tail_lengths) {
    cw_vector_launch tail_launch = {d_tail_out.opaque_handle(), d_tail_x.opaque_handle(),
                                    d_tail_y.opaque_handle(), tail, 0.375, precision,
                                    cw_vector_operation::linear_f64_coefficient};
    launch_cw_vector(tail_launch, execution);
    copy_device_to_host_async(tail_out.data(), d_tail_out, 0, tail * sizeof(T), execution);
    execution.synchronize();
    for (size_t i = 0; i < tail; ++i)
      require(same_bits(tail_out[i], T(double(tail_x[i]) + 0.375 * double(tail_y[i]))),
              "solve_cw tail launch differs");
  }
}

int main() {
  try {
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA device available");
    run_precision<float>(0, scalar_precision::f32);
    run_precision<double>(0, scalar_precision::f64);
    std::cout << "nvidia_cw_smoke: PASS\n";
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_cw_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
}
