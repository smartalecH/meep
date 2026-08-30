/* Low-level CUDA coverage for resident solve_cw vector primitives. */

#include "backend/nvidia/nvidia_cw.hpp"

#include <cmath>
#include <cstdint>
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

template <typename T> static void digest_value(uint64_t &digest, T value) {
  unsigned char bits[sizeof(T)];
  std::memcpy(bits, &value, sizeof(T));
  for (size_t i = 0; i < sizeof(T); ++i) {
    digest ^= uint64_t(bits[i]);
    digest *= UINT64_C(1099511628211);
  }
}

template <typename T>
static uint64_t run_precision(int device, scalar_precision precision) {
  device_scope selected(device);
  stream execution;
  uint64_t digest = UINT64_C(1469598103934665603);
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
  reduction.y = d_y.opaque_handle();
  launch_cw_dot(reduction, execution);
  copy_device_to_host_async(&observed, result, 0, sizeof(observed), execution);
  execution.synchronize();
  require(!std::isfinite(observed), "solve_cw dot reduction suppressed a nonfinite input");
  x[17] = std::numeric_limits<T>::infinity();
  copy_host_to_device_async(d_x, 0, x.data(), d_x.size(), execution);
  reduction.y = NULL;
  launch_cw_max_abs(reduction, execution);
  copy_device_to_host_async(&observed, result, 0, sizeof(observed), execution);
  execution.synchronize();
  require(std::isinf(observed), "solve_cw max reduction suppressed an infinite input");
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

  require_throws([&]() { launch_cw_pack(row, NULL, real_elements, execution); },
                 "solve_cw pack accepted a null vector");
  cw_state_row_launch invalid_row = row;
  invalid_row.real_values = NULL;
  require_throws([&]() { launch_cw_pack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw pack accepted a null real array");
  invalid_row = row;
  invalid_row.imaginary_values = NULL;
  require_throws([&]() { launch_cw_unpack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw unpack accepted a null imaginary array");
  invalid_row = row;
  invalid_row.imaginary_values = invalid_row.real_values;
  require_throws([&]() { launch_cw_pack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw pack accepted aliased field arrays");
  invalid_row = row;
  invalid_row.complex_count = 0;
  require_throws([&]() { launch_cw_pack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw pack accepted an empty row");
  invalid_row = row;
  ++invalid_row.complex_count;
  require_throws([&]() { launch_cw_unpack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw unpack accepted inconsistent region and row counts");
  invalid_row = row;
  invalid_row.region.counts[1] = 0;
  require_throws([&]() { launch_cw_pack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw pack accepted an empty region axis");
  invalid_row = row;
  invalid_row.region.strides[0] = -1;
  require_throws([&]() { launch_cw_unpack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw unpack accepted a negative stride");
  invalid_row = row;
  invalid_row.region.strides[0] = 0;
  require_throws([&]() { launch_cw_pack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw pack accepted a zero active stride");
  invalid_row = row;
  invalid_row.region.base = std::numeric_limits<size_t>::max();
  require_throws([&]() { launch_cw_pack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw pack accepted an overflowing region base");
  invalid_row = row;
  invalid_row.region.strides[0] = std::numeric_limits<ptrdiff_t>::max();
  require_throws([&]() { launch_cw_unpack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw unpack accepted an overflowing region stride");
  invalid_row = row;
  invalid_row.complex_offset = std::numeric_limits<size_t>::max();
  require_throws([&]() { launch_cw_pack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw pack accepted an overflowing vector range");
  invalid_row = row;
  require_throws([&]() { launch_cw_unpack(invalid_row, d_vector.opaque_handle(), 2, execution); },
                 "solve_cw unpack accepted a short vector extent");
  invalid_row = row;
  invalid_row.real_values = d_vector.opaque_handle();
  require_throws([&]() { launch_cw_pack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw pack accepted field/vector aliasing");
  invalid_row = row;
  invalid_row.precision = static_cast<scalar_precision>(99);
  require_throws([&]() { launch_cw_unpack(invalid_row, d_vector.opaque_handle(), real_elements, execution); },
                 "solve_cw unpack accepted invalid precision");

  cw_zero_launch invalid_zero = {NULL, 1, precision};
  require_throws([&]() { launch_cw_zero(invalid_zero, execution); },
                 "solve_cw zero accepted a null array");
  invalid_zero = cw_zero_launch{d_real.opaque_handle(), 0, precision};
  require_throws([&]() { launch_cw_zero(invalid_zero, execution); },
                 "solve_cw zero accepted an empty extent");
  invalid_zero = cw_zero_launch{d_real.opaque_handle(), 1, static_cast<scalar_precision>(99)};
  require_throws([&]() { launch_cw_zero(invalid_zero, execution); },
                 "solve_cw zero accepted invalid precision");

  cw_source_batch_launch invalid_source = source;
  invalid_source.target_real = NULL;
  require_throws([&]() { launch_cw_source_batch(invalid_source, d_scalar.opaque_handle(), execution); },
                 "solve_cw source accepted a null destination");
  invalid_source = source;
  invalid_source.target_imag = invalid_source.target_real;
  require_throws([&]() { launch_cw_source_batch(invalid_source, d_scalar.opaque_handle(), execution); },
                 "solve_cw source accepted aliased destinations");
  invalid_source = source;
  invalid_source.points = NULL;
  require_throws([&]() { launch_cw_source_batch(invalid_source, d_scalar.opaque_handle(), execution); },
                 "solve_cw source accepted null points");
  invalid_source = source;
  require_throws([&]() { launch_cw_source_batch(invalid_source, NULL, execution); },
                 "solve_cw source accepted null scalars");
  invalid_source = source;
  invalid_source.point_count = 0;
  require_throws([&]() { launch_cw_source_batch(invalid_source, d_scalar.opaque_handle(), execution); },
                 "solve_cw source accepted an empty point range");
  invalid_source = source;
  invalid_source.dt = std::numeric_limits<double>::quiet_NaN();
  require_throws([&]() { launch_cw_source_batch(invalid_source, d_scalar.opaque_handle(), execution); },
                 "solve_cw source accepted a nonfinite timestep");
  invalid_source = source;
  invalid_source.dt = 0.0;
  require_throws([&]() { launch_cw_source_batch(invalid_source, d_scalar.opaque_handle(), execution); },
                 "solve_cw source accepted a zero timestep");
  invalid_source = source;
  invalid_source.precision = static_cast<scalar_precision>(99);
  require_throws([&]() { launch_cw_source_batch(invalid_source, d_scalar.opaque_handle(), execution); },
                 "solve_cw source accepted invalid precision");

  vector.y = d_y.opaque_handle();
  vector.operation = cw_vector_operation::linear_f64_coefficient;
  vector.output = NULL;
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted a null output");
  vector.output = d_output.opaque_handle();
  vector.x = NULL;
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted a null input");
  vector.x = d_x.opaque_handle();
  vector.x = vector.y;
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted aliased inputs");
  vector.x = d_x.opaque_handle();
  vector.y = NULL;
  vector.operation = cw_vector_operation::linear_f64_coefficient;
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted a missing required right operand");
  vector.y = d_y.opaque_handle();
  vector.operation = cw_vector_operation::copy;
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted an unexpected right operand");
  vector.x = d_x.opaque_handle();
  vector.output = d_output.opaque_handle();
  vector.y = d_y.opaque_handle();
  vector.operation = cw_vector_operation::linear_f64_coefficient;
  vector.precision = static_cast<scalar_precision>(99);
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted invalid precision");
  vector.precision = precision;
  vector.elements = 0;
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted an empty extent");
  vector.elements = n;
  vector.coefficient = std::numeric_limits<double>::infinity();
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted a nonfinite coefficient");
  vector.coefficient = 1.0;
  vector.y = NULL;
  vector.operation = static_cast<cw_vector_operation>(99);
  require_throws([&]() { launch_cw_vector(vector, execution); },
                 "solve_cw vector accepted an invalid operation");

  cw_operator_launch invalid_operator = op;
  invalid_operator.output = NULL;
  require_throws([&]() { launch_cw_operator_finalize(invalid_operator, execution); },
                 "solve_cw operator accepted a null output");
  invalid_operator = op;
  invalid_operator.stepped = NULL;
  require_throws([&]() { launch_cw_operator_finalize(invalid_operator, execution); },
                 "solve_cw operator accepted a null stepped vector");
  invalid_operator = op;
  invalid_operator.input = NULL;
  require_throws([&]() { launch_cw_operator_finalize(invalid_operator, execution); },
                 "solve_cw operator accepted a null input vector");
  invalid_operator = op;
  invalid_operator.output = const_cast<void *>(invalid_operator.input);
  require_throws([&]() { launch_cw_operator_finalize(invalid_operator, execution); },
                 "solve_cw operator accepted output/input aliasing");
  invalid_operator = op;
  invalid_operator.stepped = invalid_operator.input;
  require_throws([&]() { launch_cw_operator_finalize(invalid_operator, execution); },
                 "solve_cw operator accepted stepped/input aliasing");
  invalid_operator = op;
  invalid_operator.real_elements = 0;
  require_throws([&]() { launch_cw_operator_finalize(invalid_operator, execution); },
                 "solve_cw operator accepted an empty extent");
  invalid_operator = op;
  invalid_operator.real_elements = 3;
  require_throws([&]() { launch_cw_operator_finalize(invalid_operator, execution); },
                 "solve_cw operator accepted an odd extent");
  invalid_operator = op;
  invalid_operator.dt_inverse = std::numeric_limits<double>::quiet_NaN();
  require_throws([&]() { launch_cw_operator_finalize(invalid_operator, execution); },
                 "solve_cw operator accepted a nonfinite coefficient");
  invalid_operator = op;
  invalid_operator.precision = static_cast<scalar_precision>(99);
  require_throws([&]() { launch_cw_operator_finalize(invalid_operator, execution); },
                 "solve_cw operator accepted invalid precision");

  cw_reduction_launch invalid_reduction = reduction;
  invalid_reduction.y = d_y.opaque_handle();
  invalid_reduction.x = NULL;
  require_throws([&]() { launch_cw_dot(invalid_reduction, execution); },
                 "solve_cw dot accepted a null left operand");
  invalid_reduction = reduction;
  invalid_reduction.y = NULL;
  require_throws([&]() { launch_cw_dot(invalid_reduction, execution); },
                 "solve_cw dot accepted a null right operand");
  invalid_reduction = reduction;
  invalid_reduction.y = d_y.opaque_handle();
  invalid_reduction.partials = NULL;
  require_throws([&]() { launch_cw_dot(invalid_reduction, execution); },
                 "solve_cw dot accepted null partials");
  invalid_reduction = reduction;
  invalid_reduction.y = d_y.opaque_handle();
  invalid_reduction.result = NULL;
  require_throws([&]() { launch_cw_dot(invalid_reduction, execution); },
                 "solve_cw dot accepted a null result");
  invalid_reduction = reduction;
  invalid_reduction.y = d_y.opaque_handle();
  invalid_reduction.elements = 0;
  require_throws([&]() { launch_cw_dot(invalid_reduction, execution); },
                 "solve_cw dot accepted an empty extent");
  invalid_reduction = reduction;
  invalid_reduction.y = d_y.opaque_handle();
  invalid_reduction.blocks = 0;
  require_throws([&]() { launch_cw_dot(invalid_reduction, execution); },
                 "solve_cw dot accepted zero partials");
  invalid_reduction = reduction;
  invalid_reduction.y = d_y.opaque_handle();
  invalid_reduction.blocks = 3;
  require_throws([&]() { launch_cw_dot(invalid_reduction, execution); },
                 "solve_cw dot accepted too many partials");
  invalid_reduction.blocks = 129;
  require_throws([&]() { launch_cw_dot(invalid_reduction, execution); },
                 "solve_cw dot accepted more than 128 partials");
  invalid_reduction = reduction;
  invalid_reduction.y = d_y.opaque_handle();
  invalid_reduction.precision = static_cast<scalar_precision>(99);
  require_throws([&]() { launch_cw_dot(invalid_reduction, execution); },
                 "solve_cw dot accepted invalid precision");
  invalid_reduction = reduction;
  invalid_reduction.y = d_y.opaque_handle();
  require_throws([&]() { launch_cw_max_abs(invalid_reduction, execution); },
                 "solve_cw max accepted a right operand");
  invalid_reduction.y = NULL;
  invalid_reduction.scale = 0.0;
  require_throws([&]() { launch_cw_scaled_norm_sum(invalid_reduction, execution); },
                 "solve_cw scaled norm accepted zero scale");
  invalid_reduction.scale = std::numeric_limits<double>::infinity();
  require_throws([&]() { launch_cw_scaled_norm_sum(invalid_reduction, execution); },
                 "solve_cw scaled norm accepted nonfinite scale");
  invalid_reduction = reduction;
  invalid_reduction.y = d_y.opaque_handle();
  invalid_reduction.partials = const_cast<void *>(invalid_reduction.x);
  require_throws([&]() { launch_cw_dot(invalid_reduction, execution); },
                 "solve_cw dot accepted aliased partial storage");
  invalid_reduction = reduction;
  invalid_reduction.y = d_y.opaque_handle();
  invalid_reduction.result = invalid_reduction.partials;
  require_throws([&]() { launch_cw_dot(invalid_reduction, execution); },
                 "solve_cw dot accepted aliased partial/result storage");

  const size_t tail_lengths[] = {1, 7, 31, 32, 33, 255, 256, 257, 511, 512, 513, 4097};
  const size_t tail_capacity = 4097;
  const size_t tail_field_capacity = 2 * tail_capacity + 4;
  const size_t tail_vector_capacity = 2 * (tail_capacity + 3);
  std::vector<T> tail_x(tail_field_capacity), tail_y(tail_field_capacity),
      tail_out(tail_vector_capacity);
  for (size_t i = 0; i < tail_capacity; ++i) {
    tail_x[i] = T(0.03125 * double(i) - 4.0);
    tail_y[i] = T(1.5 - 0.015625 * double(i));
  }
  device_buffer d_tail_x(tail_field_capacity * sizeof(T), device);
  device_buffer d_tail_y(tail_field_capacity * sizeof(T), device);
  device_buffer d_tail_out(tail_vector_capacity * sizeof(T), device);
  device_buffer d_tail_partials(128 * sizeof(double), device);
  device_buffer d_tail_result(sizeof(double), device);
  copy_host_to_device_async(d_tail_x, 0, tail_x.data(), d_tail_x.size(), execution);
  copy_host_to_device_async(d_tail_y, 0, tail_y.data(), d_tail_y.size(), execution);
  for (size_t tail : tail_lengths) {
    std::fill(tail_out.begin(), tail_out.end(), T(-19));
    copy_host_to_device_async(d_tail_out, 0, tail_out.data(), d_tail_out.size(), execution);
    cw_state_row_launch tail_row = {};
    tail_row.region.base = 2;
    tail_row.region.counts[0] = tail;
    tail_row.region.counts[1] = tail_row.region.counts[2] = 1;
    tail_row.region.strides[0] = tail_row.region.strides[1] =
        tail_row.region.strides[2] = 1;
    tail_row.real_values = d_tail_x.opaque_handle();
    tail_row.imaginary_values = d_tail_y.opaque_handle();
    tail_row.complex_offset = 1;
    tail_row.complex_count = tail;
    tail_row.precision = precision;
    launch_cw_pack(tail_row, d_tail_out.opaque_handle(), tail_vector_capacity, execution);
    copy_device_to_host_async(tail_out.data(), d_tail_out, 0, d_tail_out.size(), execution);
    execution.synchronize();
    require(tail_out[0] == T(-19) && tail_out[1] == T(-19),
            "solve_cw tail pack changed prefix padding");
    for (size_t i = 0; i < tail; ++i) {
      require(same_bits(tail_out[2 * (i + 1)], tail_x[i + 2]),
              "solve_cw tail pack real differs");
      require(same_bits(tail_out[2 * (i + 1) + 1], tail_y[i + 2]),
              "solve_cw tail pack imaginary differs");
      digest_value(digest, tail_out[2 * (i + 1)]);
      digest_value(digest, tail_out[2 * (i + 1) + 1]);
    }
    require(tail_out[2 * (tail + 1)] == T(-19),
            "solve_cw tail pack changed suffix padding");

    const std::vector<T> tail_x_before = tail_x, tail_y_before = tail_y;
    for (size_t i = 0; i < tail; ++i) {
      tail_out[2 * (i + 1)] = T(7.0 + 0.015625 * double(i));
      tail_out[2 * (i + 1) + 1] = T(-6.0 - 0.03125 * double(i));
    }
    copy_host_to_device_async(d_tail_out, 0, tail_out.data(), d_tail_out.size(), execution);
    launch_cw_unpack(tail_row, d_tail_out.opaque_handle(), tail_vector_capacity, execution);
    copy_device_to_host_async(tail_x.data(), d_tail_x, 0, d_tail_x.size(), execution);
    copy_device_to_host_async(tail_y.data(), d_tail_y, 0, d_tail_y.size(), execution);
    execution.synchronize();
    require(tail_x[0] == tail_x_before[0] && tail_x[1] == tail_x_before[1] &&
                tail_y[0] == tail_y_before[0] && tail_y[1] == tail_y_before[1] &&
                tail_x[tail + 2] == tail_x_before[tail + 2] &&
                tail_y[tail + 2] == tail_y_before[tail + 2],
            "solve_cw tail unpack changed values outside the row");
    for (size_t i = 0; i < tail; ++i) {
      require(same_bits(tail_x[i + 2], tail_out[2 * (i + 1)]) &&
                  same_bits(tail_y[i + 2], tail_out[2 * (i + 1) + 1]),
              "solve_cw tail unpack differs");
      digest_value(digest, tail_x[i + 2]);
      digest_value(digest, tail_y[i + 2]);
    }
    tail_x = tail_x_before;
    tail_y = tail_y_before;
    copy_host_to_device_async(d_tail_x, 0, tail_x_before.data(), d_tail_x.size(), execution);
    copy_host_to_device_async(d_tail_y, 0, tail_y_before.data(), d_tail_y.size(), execution);

    std::fill(tail_out.begin(), tail_out.end(), T(-23));
    copy_host_to_device_async(d_tail_out, 0, tail_out.data(), d_tail_out.size(), execution);
    launch_cw_zero(cw_zero_launch{d_tail_out.opaque_handle(), tail, precision}, execution);
    copy_device_to_host_async(tail_out.data(), d_tail_out, 0, d_tail_out.size(), execution);
    execution.synchronize();
    for (size_t i = 0; i < tail; ++i)
      require(tail_out[i] == T(0), "solve_cw tail zero differs");
    require(tail_out[tail] == T(-23), "solve_cw tail zero changed its sentinel");

    std::fill(tail_out.begin(), tail_out.end(), T(-29));
    copy_host_to_device_async(d_tail_out, 0, tail_out.data(), d_tail_out.size(), execution);
    cw_vector_launch tail_launch = {d_tail_out.opaque_handle(), d_tail_x.opaque_handle(),
                                    d_tail_y.opaque_handle(), tail, 0.375, precision,
                                    cw_vector_operation::linear_f64_coefficient};
    launch_cw_vector(tail_launch, execution);
    copy_device_to_host_async(tail_out.data(), d_tail_out, 0, d_tail_out.size(), execution);
    execution.synchronize();
    for (size_t i = 0; i < tail; ++i)
      require(same_bits(tail_out[i], T(double(tail_x_before[i]) + 0.375 * double(tail_y_before[i]))),
              "solve_cw tail launch differs");
    require(tail_out[tail] == T(-29), "solve_cw tail vector changed its sentinel");

    std::fill(tail_out.begin(), tail_out.end(), T(-31));
    copy_host_to_device_async(d_tail_out, 0, tail_out.data(), d_tail_out.size(), execution);
    cw_operator_launch tail_operator = {
        d_tail_out.opaque_handle(), d_tail_y.opaque_handle(), d_tail_x.opaque_handle(),
        2 * tail, 1.25, -0.5, 0.125, precision};
    launch_cw_operator_finalize(tail_operator, execution);
    copy_device_to_host_async(tail_out.data(), d_tail_out, 0, d_tail_out.size(), execution);
    execution.synchronize();
    const T tail_dt_inverse = T(tail_operator.dt_inverse);
    const T tail_wr = T(tail_operator.iomega_real);
    const T tail_wi = T(tail_operator.iomega_imaginary);
    for (size_t i = 0; i < tail; ++i) {
      const size_t r = 2 * i;
      const T expected_real =
          T(T(tail_y_before[r] - tail_x_before[r]) * tail_dt_inverse +
            T(tail_wr * tail_x_before[r] - tail_wi * tail_x_before[r + 1]));
      const T expected_imag =
          T(T(tail_y_before[r + 1] - tail_x_before[r + 1]) * tail_dt_inverse +
            T(tail_wr * tail_x_before[r + 1] + tail_wi * tail_x_before[r]));
      require(same_bits(tail_out[r], expected_real) &&
                  same_bits(tail_out[r + 1], expected_imag),
              "solve_cw tail operator expression differs");
      digest_value(digest, tail_out[r]);
      digest_value(digest, tail_out[r + 1]);
    }
    require(tail_out[2 * tail] == T(-31),
            "solve_cw tail operator changed its sentinel");

    const size_t tail_blocks = std::min<size_t>(128, 1 + (tail - 1) / 256);
    cw_reduction_launch tail_reduction = {
        d_tail_x.opaque_handle(), d_tail_y.opaque_handle(), d_tail_partials.opaque_handle(),
        d_tail_result.opaque_handle(), tail, tail_blocks, 1.0, precision};
    double tail_results[3] = {0, 0, 0};
    long double dot_reference = 0.0L;
    double maximum_reference = 0.0;
    for (size_t i = 0; i < tail; ++i) {
      dot_reference += static_cast<long double>(T(tail_x_before[i] * tail_y_before[i]));
      maximum_reference = std::max(maximum_reference, std::fabs(double(tail_x_before[i])));
    }
    for (int repeat = 0; repeat < 20; ++repeat) {
      launch_cw_dot(tail_reduction, execution);
      copy_device_to_host_async(&observed, d_tail_result, 0, sizeof(observed), execution);
      execution.synchronize();
      require(std::fabs(observed - double(dot_reference)) <=
                  4e-13 * (1.0 + double(std::fabs(dot_reference))),
              "solve_cw tail dot differs from long-double reference");
      if (!repeat) tail_results[0] = observed;
      else require(same_bits(observed, tail_results[0]),
                   "solve_cw tail dot is not bit deterministic");

      tail_reduction.y = NULL;
      launch_cw_max_abs(tail_reduction, execution);
      copy_device_to_host_async(&observed, d_tail_result, 0, sizeof(observed), execution);
      execution.synchronize();
      require(observed == maximum_reference, "solve_cw tail max differs");
      if (!repeat) tail_results[1] = observed;
      else require(same_bits(observed, tail_results[1]),
                   "solve_cw tail max is not bit deterministic");

      tail_reduction.scale = 1.0 / maximum_reference;
      launch_cw_scaled_norm_sum(tail_reduction, execution);
      copy_device_to_host_async(&observed, d_tail_result, 0, sizeof(observed), execution);
      execution.synchronize();
      long double norm_reference = 0.0L;
      for (size_t i = 0; i < tail; ++i) {
        const long double scaled = static_cast<long double>(tail_reduction.scale) *
                                   static_cast<long double>(tail_x_before[i]);
        norm_reference += scaled * scaled;
      }
      require(std::fabs(observed - double(norm_reference)) <=
                  2e-12 * (1.0 + double(norm_reference)),
              "solve_cw tail scaled norm differs from long-double reference");
      if (!repeat) tail_results[2] = observed;
      else require(same_bits(observed, tail_results[2]),
                   "solve_cw tail scaled norm is not bit deterministic");
      tail_reduction.y = d_tail_y.opaque_handle();
      tail_reduction.scale = 1.0;
    }
    for (double value : tail_results) digest_value(digest, value);
  }
  return digest;
}

int main() {
  try {
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA device available");
    const uint64_t f32_digest = run_precision<float>(0, scalar_precision::f32);
    const uint64_t f64_digest = run_precision<double>(0, scalar_precision::f64);
    std::cout << "nvidia_cw_smoke: PASS f32_digest=" << std::hex << f32_digest
              << " f64_digest=" << f64_digest << std::dec << "\n";
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_cw_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
}
