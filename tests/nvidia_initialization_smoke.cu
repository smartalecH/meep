/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_initialization.hpp"
#include "backend/nvidia/runtime.hpp"

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

template <typename Callable>
static void require_invalid(Callable callable, const char *message) {
  bool rejected = false;
  try { callable(); }
  catch (const std::invalid_argument &) { rejected = true; }
  catch (const std::overflow_error &) { rejected = true; }
  require(rejected, message);
}

template <typename T>
static void require_exact(const std::vector<T> &observed, const std::vector<T> &expected,
                          const char *message) {
  require(observed.size() == expected.size(), message);
  for (size_t i = 0; i < observed.size(); ++i)
    if (observed[i] != expected[i]) throw std::runtime_error(message);
}

template <typename T>
static void check_length(int device, scalar_precision precision, size_t elements,
                         bool logical_single = false) {
  stream execution;
  const T sentinel = T(-19.25);
  const double fill_value = 0.12345678901234567;
  std::vector<T> destination(elements + 2, sentinel);
  device_buffer device_destination(destination.size() * sizeof(T), device);
  copy_host_to_device_async(device_destination, 0, destination.data(),
                            destination.size() * sizeof(T), execution);
  material_fill_launch fill = {};
  fill.destination = static_cast<unsigned char *>(device_destination.opaque_handle()) + sizeof(T);
  fill.elements = elements;
  fill.value = fill_value;
  fill.phase = 0;
  fill.precision = precision;
  launch_material_fill(fill, execution);
  copy_device_to_host_async(destination.data(), device_destination, 0,
                            destination.size() * sizeof(T), execution);
  execution.synchronize();
  require(destination.front() == sentinel && destination.back() == sentinel,
          "material fill overwrote a guard element");
  for (size_t i = 0; i < elements; ++i)
    require(destination[i + 1] == T(fill_value), "material fill value differs");
  if (elements) {
    fill.value = -0.0;
    launch_material_fill(fill, execution);
    copy_device_to_host_async(destination.data(), device_destination, 0,
                              destination.size() * sizeof(T), execution);
    execution.synchronize();
    for (size_t i = 0; i < elements; ++i)
      require(destination[i + 1] == T(0) && std::signbit(destination[i + 1]),
              "material fill did not preserve negative zero");
  }

  std::vector<double> profile(elements);
  for (size_t i = 0; i < elements; ++i) {
    profile[i] = 0.03125 + 0.00013 * i;
  }
  const size_t profile_offset = 16;
  std::vector<unsigned char> compact(profile_offset + elements * sizeof(double), 0xa5);
  if (elements)
    std::memcpy(compact.data() + profile_offset, profile.data(), elements * sizeof(double));
  device_buffer device_profile(compact.size(), device);
  device_buffer device_sigma((elements + 2) * sizeof(T), device);
  device_buffer device_kappa((elements + 2) * sizeof(T), device);
  device_buffer device_inverse((elements + 2) * sizeof(T), device);
  std::vector<T> guards(elements + 2, sentinel);
  copy_host_to_device_async(device_profile, 0, compact.data(), compact.size(), execution);
  copy_host_to_device_async(device_sigma, 0, guards.data(), guards.size() * sizeof(T), execution);
  copy_host_to_device_async(device_kappa, 0, guards.data(), guards.size() * sizeof(T), execution);
  copy_host_to_device_async(device_inverse, 0, guards.data(), guards.size() * sizeof(T), execution);
  material_pml_launch pml = {};
  pml.sigma_destination = static_cast<unsigned char *>(device_sigma.opaque_handle()) + sizeof(T);
  pml.kappa_destination = static_cast<unsigned char *>(device_kappa.opaque_handle()) + sizeof(T);
  pml.sigma_inv_destination =
      static_cast<unsigned char *>(device_inverse.opaque_handle()) + sizeof(T);
  pml.compact_inputs = static_cast<const unsigned char *>(device_profile.opaque_handle());
  pml.compact_input_bytes = compact.size();
  pml.profile_offset = profile_offset;
  pml.elements = elements;
  pml.little_corner = 0;
  pml.resolution = 8.0;
  pml.dt = 0.03;
  pml.thickness = 10.0;
  pml.boundary_location = 0.0;
  pml.r_asymptotic = 1e-8;
  pml.mean_stretch = 1.5;
  pml.profile_integral = 0.25;
  pml.profile_integral_u = 0.2;
  pml.thickness_cells = int(pml.thickness * (2 * pml.resolution) + 0.5);
  pml.profile_active = true;
  pml.logical_single = logical_single;
  pml.precision = precision;
  launch_material_pml(pml, execution);
  std::vector<T> observed_sigma(elements + 2), observed_kappa(elements + 2),
      observed_inverse(elements + 2);
  copy_device_to_host_async(observed_sigma.data(), device_sigma, 0,
                            observed_sigma.size() * sizeof(T), execution);
  copy_device_to_host_async(observed_kappa.data(), device_kappa, 0,
                            observed_kappa.size() * sizeof(T), execution);
  copy_device_to_host_async(observed_inverse.data(), device_inverse, 0,
                            observed_inverse.size() * sizeof(T), execution);
  execution.synchronize();
  require(observed_sigma.front() == sentinel && observed_sigma.back() == sentinel &&
              observed_kappa.front() == sentinel && observed_kappa.back() == sentinel &&
              observed_inverse.front() == sentinel && observed_inverse.back() == sentinel,
          "material PML initialization overwrote a guard element");
  for (size_t i = 0; i < elements; ++i) {
    const double here = double(i) * 0.5 / pml.resolution;
    const double x = 0.5 / pml.resolution *
                     (int(pml.thickness * (2 * pml.resolution) + 0.5) -
                      int(std::fabs(pml.boundary_location - here) * (2 * pml.resolution) + 0.5));
    double sigma = 0.0, kappa = 1.0, inverse = 1.0;
    if (x > 0) {
      sigma = 0.5 * pml.dt *
              ((-std::log(pml.r_asymptotic)) /
               (4 * pml.thickness * pml.profile_integral)) *
              profile[i];
      kappa = 1 + ((pml.mean_stretch - 1) / pml.profile_integral_u) * profile[i] *
                      (x / pml.thickness);
      if (logical_single) {
        sigma = float(sigma);
        kappa = float(kappa);
        inverse = float(1 / (float(kappa) + float(sigma)));
      }
      else
        inverse = 1 / (kappa + sigma);
    }
    require(observed_sigma[i + 1] == T(sigma), "material PML sigma differs");
    require(observed_kappa[i + 1] == T(kappa), "material PML kappa differs");
    require(observed_inverse[i + 1] == T(inverse), "material PML sigma inverse differs");
  }
}

template <typename T>
static void check_failures(int device, scalar_precision precision) {
  stream execution;
  std::vector<T> initial(5, T(-7));
  device_buffer destination(initial.size() * sizeof(T), device);
  copy_host_to_device_async(destination, 0, initial.data(), initial.size() * sizeof(T), execution);
  execution.synchronize();
  material_fill_launch fill = {destination.opaque_handle(), initial.size(), 3.0, 0, precision};
  testing::fail_next(testing::failure_point::material_pointwise_launch);
  bool rejected = false;
  try { launch_material_fill(fill, execution); }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "injected material pointwise launch failure was ignored");
  std::vector<T> observed(initial.size());
  copy_device_to_host_async(observed.data(), destination, 0, observed.size() * sizeof(T), execution);
  execution.synchronize();
  require_exact(observed, initial, "failed material pointwise launch changed output");

  material_fill_launch malformed = fill;
  malformed.destination = NULL;
  require_invalid([&]() { launch_material_fill(malformed, execution); },
                  "null material fill destination was accepted");
  malformed = fill;
  malformed.destination = static_cast<unsigned char *>(destination.opaque_handle()) + 1;
  require_invalid([&]() { launch_material_fill(malformed, execution); },
                  "misaligned material fill destination was accepted");
  malformed = fill;
  malformed.precision = static_cast<scalar_precision>(99);
  require_invalid([&]() { launch_material_fill(malformed, execution); },
                  "invalid material fill precision was accepted");
  malformed = fill;
  malformed.elements = std::numeric_limits<size_t>::max();
  require_invalid([&]() { launch_material_fill(malformed, execution); },
                  "overflowing material fill range was accepted");
  material_fill_launch empty = {NULL, 0, 1.0, 0, precision};
  launch_material_fill(empty, execution);

  const size_t profile_offset = 16;
  const double profile[2] = {0.25, 0.5};
  std::vector<unsigned char> compact_bytes(profile_offset + sizeof(profile), 0);
  std::memcpy(compact_bytes.data() + profile_offset, profile, sizeof(profile));
  device_buffer compact(compact_bytes.size(), device), kappa(initial.size() * sizeof(T), device),
      inverse(initial.size() * sizeof(T), device);
  copy_host_to_device_async(compact, 0, compact_bytes.data(), compact_bytes.size(), execution);
  copy_host_to_device_async(destination, 0, initial.data(), initial.size() * sizeof(T), execution);
  copy_host_to_device_async(kappa, 0, initial.data(), initial.size() * sizeof(T), execution);
  copy_host_to_device_async(inverse, 0, initial.data(), initial.size() * sizeof(T), execution);
  execution.synchronize();
  material_pml_launch pml = {};
  pml.sigma_destination = destination.opaque_handle();
  pml.kappa_destination = kappa.opaque_handle();
  pml.sigma_inv_destination = inverse.opaque_handle();
  pml.compact_inputs = static_cast<const unsigned char *>(compact.opaque_handle());
  pml.compact_input_bytes = compact.size();
  pml.profile_offset = profile_offset;
  pml.elements = 2;
  pml.resolution = 8;
  pml.dt = 0.05;
  pml.thickness = 0.25;
  pml.boundary_location = 0;
  pml.r_asymptotic = 1e-8;
  pml.mean_stretch = 1.2;
  pml.profile_integral = 1.0 / 3.0;
  pml.profile_integral_u = 0.25;
  pml.thickness_cells = 4;
  pml.profile_active = true;
  pml.precision = precision;
  const auto reject_pml = [&](const material_pml_launch &candidate, const char *message) {
    require_invalid([&]() { launch_material_pml(candidate, execution); }, message);
  };
  material_pml_launch bad_pml = pml;
  bad_pml.profile_offset = 1;
  reject_pml(bad_pml, "misaligned material PML compact offset was accepted");
  bad_pml = pml;
  bad_pml.profile_offset = compact.size();
  reject_pml(bad_pml, "out-of-range material PML compact input was accepted");
  bad_pml = pml;
  bad_pml.compact_inputs += 1;
  reject_pml(bad_pml, "misaligned material PML compact base was accepted");
  bad_pml = pml;
  bad_pml.kappa_destination = bad_pml.sigma_destination;
  reject_pml(bad_pml, "aliased material PML destinations were accepted");
  bad_pml = pml;
  bad_pml.kappa_destination = static_cast<unsigned char *>(bad_pml.sigma_destination) + sizeof(T);
  reject_pml(bad_pml, "overlapping material PML destinations were accepted");
  bad_pml = pml;
  bad_pml.precision = static_cast<scalar_precision>(99);
  reject_pml(bad_pml, "invalid material PML precision was accepted");
  bad_pml = pml;
  bad_pml.resolution = std::numeric_limits<double>::infinity();
  reject_pml(bad_pml, "non-finite material PML resolution was accepted");
  bad_pml = pml;
  bad_pml.dt = 0;
  reject_pml(bad_pml, "zero material PML timestep was accepted");
  bad_pml = pml;
  bad_pml.r_asymptotic = 1;
  reject_pml(bad_pml, "invalid material PML asymptotic reflection was accepted");
  bad_pml = pml;
  bad_pml.profile_integral = 0;
  reject_pml(bad_pml, "zero material PML profile integral was accepted");
  bad_pml = pml;
  bad_pml.thickness_cells = 0;
  reject_pml(bad_pml, "zero material PML thickness cells were accepted");
  bad_pml = pml;
  --bad_pml.thickness_cells;
  reject_pml(bad_pml, "inconsistent material PML thickness cells were accepted");
  bad_pml = pml;
  bad_pml.little_corner = std::numeric_limits<int>::max();
  bad_pml.elements = 2;
  reject_pml(bad_pml, "overflowing material PML logical index was accepted");
  bad_pml = pml;
  bad_pml.boundary_location = double(std::numeric_limits<int>::max());
  reject_pml(bad_pml, "overflowing material PML distance cast was accepted");

  std::vector<T> unchanged(initial.size()), unchanged_kappa(initial.size()),
      unchanged_inverse(initial.size());
  copy_device_to_host_async(unchanged.data(), destination, 0,
                            unchanged.size() * sizeof(T), execution);
  copy_device_to_host_async(unchanged_kappa.data(), kappa, 0,
                            unchanged_kappa.size() * sizeof(T), execution);
  copy_device_to_host_async(unchanged_inverse.data(), inverse, 0,
                            unchanged_inverse.size() * sizeof(T), execution);
  execution.synchronize();
  require_exact(unchanged, initial, "rejected material descriptor changed sigma output");
  require_exact(unchanged_kappa, initial, "rejected material descriptor changed kappa output");
  require_exact(unchanged_inverse, initial, "rejected material descriptor changed inverse output");
}

template <typename T>
static void check_absorber(int device, scalar_precision precision,
                           bool logical_single = false) {
  stream execution;
  const size_t elements = 9;
  const T sentinel = T(-31.5);
  std::vector<double> low = {0.0, 0.4, 0.9};
  std::vector<double> high = {0.0, 0.7, 1.3};
  const size_t header_offset = 16;
  const size_t header_bytes = 2 * sizeof(material_absorber_header);
  const size_t low_offset = header_offset + header_bytes;
  const size_t high_offset = low_offset + low.size() * sizeof(double);
  std::vector<unsigned char> compact(high_offset + high.size() * sizeof(double), 0);
  material_absorber_header headers[2] = {};
  headers[0].version = headers[1].version = 1;
  headers[0].direction = headers[1].direction = 2;
  headers[0].side = 1;
  headers[1].side = 0;
  headers[0].sample_offset = low_offset;
  headers[1].sample_offset = high_offset;
  headers[0].sample_count = low.size();
  headers[1].sample_count = high.size();
  headers[0].thickness = headers[1].thickness = 1.0;
  headers[0].sample_spacing = headers[1].sample_spacing = 0.5;
  std::memcpy(compact.data() + header_offset, headers, sizeof(headers));
  std::memcpy(compact.data() + low_offset, low.data(), low.size() * sizeof(double));
  std::memcpy(compact.data() + high_offset, high.data(), high.size() * sizeof(double));
  validate_material_absorber_headers(compact.data(), compact.size(), header_offset, 2);
  const auto reject_headers = [&](const std::vector<unsigned char> &candidate,
                                  size_t offset, size_t count, const char *message) {
    require_invalid(
        [&]() {
          validate_material_absorber_headers(candidate.data(), candidate.size(), offset, count);
        },
        message);
  };
  std::vector<unsigned char> malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].version = 2;
  reject_headers(malformed, header_offset, 2, "stale absorber header version was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].reserved = 1;
  reject_headers(malformed, header_offset, 2, "nonzero absorber reserved field was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].direction = 5;
  reject_headers(malformed, header_offset, 2, "invalid absorber direction was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].side = 2;
  reject_headers(malformed, header_offset, 2, "invalid absorber side was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].sample_count = 1;
  reject_headers(malformed, header_offset, 2, "short absorber sample count was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].sample_count =
      std::numeric_limits<uint64_t>::max();
  reject_headers(malformed, header_offset, 2, "overflowing absorber sample count was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].sample_offset =
      header_offset;
  reject_headers(malformed, header_offset, 2, "absorber samples overlapping headers were accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].sample_offset =
      low_offset + 1;
  reject_headers(malformed, header_offset, 2, "misaligned absorber sample offset was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].thickness = 0;
  reject_headers(malformed, header_offset, 2, "zero absorber thickness was accepted");
  malformed = compact;
  reinterpret_cast<material_absorber_header *>(malformed.data() + header_offset)[0].sample_spacing =
      0.25;
  reject_headers(malformed, header_offset, 2, "inconsistent absorber spacing was accepted");
  malformed = compact;
  reinterpret_cast<double *>(malformed.data() + low_offset)[1] =
      std::numeric_limits<double>::quiet_NaN();
  reject_headers(malformed, header_offset, 2, "non-finite absorber sample was accepted");
  reject_headers(compact, header_offset + 1, 2, "misaligned absorber header block was accepted");
  reject_headers(compact, compact.size(), 1, "out-of-range absorber header block was accepted");
  require_invalid(
      [&]() {
        validate_material_absorber_headers(compact.data() + 1, compact.size() - 1,
                                            header_offset, 2);
      },
      "misaligned absorber compact base was accepted");

  device_buffer device_compact(compact.size(), device);
  copy_host_to_device_async(device_compact, 0, compact.data(), compact.size(), execution);
  std::vector<T> guards(elements + 2, sentinel), observed_cnd(elements + 2),
      observed_inv(elements + 2);
  device_buffer device_cnd(guards.size() * sizeof(T), device);
  device_buffer device_inv(guards.size() * sizeof(T), device);
  copy_host_to_device_async(device_cnd, 0, guards.data(), guards.size() * sizeof(T), execution);
  copy_host_to_device_async(device_inv, 0, guards.data(), guards.size() * sizeof(T), execution);

  material_conductivity_launch launch = {};
  launch.conductivity_destination =
      static_cast<unsigned char *>(device_cnd.opaque_handle()) + sizeof(T);
  launch.condinv_destination =
      static_cast<unsigned char *>(device_inv.opaque_handle()) + sizeof(T);
  launch.compact_inputs = static_cast<const unsigned char *>(device_compact.opaque_handle());
  launch.compact_input_bytes = compact.size();
  launch.absorber_header_offset = header_offset;
  launch.absorber_count = 2;
  launch.elements = elements;
  launch.loop_count = elements;
  launch.component = 14;
  launch.dimensions = 0;
  launch.axis_direction[0] = 0;
  launch.axis_direction[1] = 1;
  launch.axis_direction[2] = 2;
  launch.loop_begin[0] = launch.loop_begin[1] = 0;
  launch.loop_begin[2] = -8;
  launch.little_corner[0] = launch.little_corner[1] = 0;
  launch.little_corner[2] = -8;
  launch.loop_extent[0] = launch.loop_extent[1] = 1;
  launch.loop_extent[2] = elements;
  launch.strides[0] = launch.strides[1] = 0;
  launch.strides[2] = 1;
  launch.cell_size[2] = 4.0;
  launch.inva = 0.5;
  launch.base_conductivity = 0.25;
  launch.dt = 0.03;
  launch.logical_single = logical_single;
  launch.precision = precision;
  launch_material_conductivity(launch, execution);
  copy_device_to_host_async(observed_cnd.data(), device_cnd, 0,
                            observed_cnd.size() * sizeof(T), execution);
  copy_device_to_host_async(observed_inv.data(), device_inv, 0,
                            observed_inv.size() * sizeof(T), execution);
  execution.synchronize();
  require(observed_cnd.front() == sentinel && observed_cnd.back() == sentinel &&
              observed_inv.front() == sentinel && observed_inv.back() == sentinel,
          "material conductivity initialization overwrote a guard element");
  for (size_t i = 0; i < elements; ++i) {
    const double x = -2.0 + 0.5 * i;
    double expected = 0.25;
    if (x <= -1.0) {
      const double u = 2 * (-1.0 - x);
      const int sample = int(u);
      expected += sample >= 2 ? low[2]
                              : low[sample] * (1 - (u - sample)) + low[sample + 1] * (u - sample);
    }
    if (x >= 1.0) {
      const double u = 2 * (x - 1.0);
      const int sample = int(u);
      expected += sample >= 2
                      ? high[2]
                      : high[sample] * (1 - (u - sample)) + high[sample + 1] * (u - sample);
    }
    double expected_inverse;
    if (logical_single) {
      expected = float(expected);
      expected_inverse = float(1 / (1 + double(float(expected)) * launch.dt * 0.5));
    }
    else
      expected_inverse = 1 / (1 + expected * launch.dt * 0.5);
    require(observed_cnd[i + 1] == T(expected), "material absorber conductivity differs");
    require(observed_inv[i + 1] == T(expected_inverse),
            "material absorber conductivity inverse differs");
  }

  copy_host_to_device_async(device_cnd, 0, guards.data(), guards.size() * sizeof(T), execution);
  copy_host_to_device_async(device_inv, 0, guards.data(), guards.size() * sizeof(T), execution);
  execution.synchronize();
  const auto reject_launch = [&](const material_conductivity_launch &candidate,
                                 const char *message) {
    require_invalid([&]() { launch_material_conductivity(candidate, execution); }, message);
  };
  material_conductivity_launch bad = launch;
  bad.compact_inputs += 1;
  reject_launch(bad, "misaligned conductivity compact base was accepted");
  bad = launch;
  bad.absorber_header_offset += 1;
  reject_launch(bad, "misaligned conductivity header offset was accepted");
  bad = launch;
  bad.absorber_count = std::numeric_limits<size_t>::max();
  reject_launch(bad, "overflowing conductivity header count was accepted");
  bad = launch;
  bad.condinv_destination = bad.conductivity_destination;
  reject_launch(bad, "aliased conductivity destinations were accepted");
  bad = launch;
  bad.condinv_destination = static_cast<unsigned char *>(bad.conductivity_destination) + sizeof(T);
  reject_launch(bad, "overlapping conductivity destinations were accepted");
  bad = launch;
  bad.precision = static_cast<scalar_precision>(99);
  reject_launch(bad, "invalid conductivity precision was accepted");
  bad = launch;
  bad.dimensions = 4;
  reject_launch(bad, "invalid conductivity dimensions were accepted");
  bad = launch;
  bad.axis_direction[1] = 5;
  reject_launch(bad, "invalid conductivity axis direction was accepted");
  bad = launch;
  bad.loop_extent[1] = 0;
  reject_launch(bad, "zero conductivity loop extent was accepted");
  bad = launch;
  bad.loop_extent[0] = std::numeric_limits<size_t>::max();
  bad.loop_extent[1] = 2;
  reject_launch(bad, "overflowing conductivity loop product was accepted");
  bad = launch;
  bad.loop_count -= 1;
  reject_launch(bad, "mismatched conductivity loop count was accepted");
  bad = launch;
  bad.strides[2] = -1;
  reject_launch(bad, "negative conductivity stride was accepted");
  bad = launch;
  bad.elements -= 1;
  reject_launch(bad, "out-of-range conductivity destination index was accepted");
  bad = launch;
  bad.loop_begin[0] = std::numeric_limits<int>::max();
  bad.little_corner[0] = std::numeric_limits<int>::min();
  reject_launch(bad, "overflowing conductivity stagger subtraction was accepted");
  bad = launch;
  bad.loop_base_offset[0] = 1;
  reject_launch(bad, "inconsistent conductivity base offset was accepted");
  bad = launch;
  bad.dt = std::numeric_limits<double>::quiet_NaN();
  reject_launch(bad, "non-finite conductivity timestep was accepted");
  bad = launch;
  bad.cell_size[2] = std::numeric_limits<double>::infinity();
  reject_launch(bad, "non-finite conductivity cell size was accepted");
  bad = launch;
  bad.inva = std::numeric_limits<double>::max();
  bad.loop_begin[2] = std::numeric_limits<int>::max();
  reject_launch(bad, "overflowing conductivity coordinate was accepted");
  copy_device_to_host_async(observed_cnd.data(), device_cnd, 0,
                            observed_cnd.size() * sizeof(T), execution);
  copy_device_to_host_async(observed_inv.data(), device_inv, 0,
                            observed_inv.size() * sizeof(T), execution);
  execution.synchronize();
  require_exact(observed_cnd, guards,
                "rejected conductivity descriptor changed conductivity output");
  require_exact(observed_inv, guards,
                "rejected conductivity descriptor changed inverse output");
}

int main() {
  try {
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA devices found");
    const size_t lengths[] = {0, 1, 255, 256, 257};
    for (size_t di = 0; di < devices.size(); ++di) {
      device_scope scope(devices[di].id);
      for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
        check_length<float>(devices[di].id, scalar_precision::f32, lengths[i]);
        check_length<double>(devices[di].id, scalar_precision::f64, lengths[i]);
      }
      check_length<float>(devices[di].id, scalar_precision::f32, 17, true);
      check_length<double>(devices[di].id, scalar_precision::f64, 17, true);
      check_failures<float>(devices[di].id, scalar_precision::f32);
      check_failures<double>(devices[di].id, scalar_precision::f64);
      check_absorber<float>(devices[di].id, scalar_precision::f32);
      check_absorber<double>(devices[di].id, scalar_precision::f64);
      check_absorber<float>(devices[di].id, scalar_precision::f32, true);
      check_absorber<double>(devices[di].id, scalar_precision::f64, true);
      std::cout << "device " << devices[di].id << " (" << devices[di].name
                << "): NVIDIA material initialization kernels PASS\n";
    }
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_initialization_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
  std::cout << "nvidia_initialization_smoke: PASS\n";
  return 0;
}
