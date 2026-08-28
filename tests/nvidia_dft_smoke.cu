/* Low-level parity test for NVIDIA DFT phase generation, interpolation,
   boundary weighting, complex fields, and precision pairs. */

#include "backend/nvidia/nvidia_dft.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace meep::nvidia;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

static double boundary_weight(ptrdiff_t i, ptrdiff_t n, double s0, double s1, double e0,
                              double e1) {
  if (i > 1 && i < n - 2) return 1.0;
  if (i == 0) return s0;
  if (i == 1) return s1;
  if (i == n - 1) return e0;
  if (i == n - 2) return e1;
  return 1.0;
}

template <typename FieldT, typename MonitorT>
static void run_precision_pair(int device, scalar_precision field_precision,
                               scalar_precision monitor_precision) {
  device_scope selected(device);
  stream execution;
  const size_t source_elements = 80;
  const size_t points = 6;
  const size_t frequencies = 3;
  std::vector<FieldT> source_real(source_elements), source_imag(source_elements);
  for (size_t i = 0; i < source_elements; ++i) {
    source_real[i] = FieldT(std::sin(0.11 * double(i)) + 0.01 * double(i));
    source_imag[i] = FieldT(std::cos(0.07 * double(i)) - 0.005 * double(i));
  }
  std::vector<MonitorT> accumulator(2 * points * frequencies, MonitorT(0));
  std::vector<MonitorT> phase(2 * frequencies, MonitorT(0));
  const std::vector<double> omega = {-91.0, 0.4, 0.7, 1.1};

  device_buffer d_real(source_real.size() * sizeof(FieldT), device);
  device_buffer d_imag(source_imag.size() * sizeof(FieldT), device);
  device_buffer d_accumulator(accumulator.size() * sizeof(MonitorT), device);
  device_buffer d_phase(phase.size() * sizeof(MonitorT), device);
  device_buffer d_omega(omega.size() * sizeof(double), device);
  copy_host_to_device_async(d_real, 0, source_real.data(), d_real.size(), execution);
  copy_host_to_device_async(d_imag, 0, source_imag.data(), d_imag.size(), execution);
  copy_host_to_device_async(d_accumulator, 0, accumulator.data(), d_accumulator.size(), execution);
  copy_host_to_device_async(d_phase, 0, phase.data(), d_phase.size(), execution);
  copy_host_to_device_async(d_omega, 0, omega.data(), d_omega.size(), execution);

  dft_launch launch = {};
  launch.region.base = 4;
  launch.region.counts[0] = 2;
  launch.region.counts[1] = 1;
  launch.region.counts[2] = 3;
  launch.region.strides[0] = 24;
  launch.region.strides[1] = 8;
  launch.region.strides[2] = 2;
  launch.source_real = d_real.opaque_handle();
  launch.source_imag = d_imag.opaque_handle();
  launch.accumulator = d_accumulator.opaque_handle();
  launch.phase_scratch = d_phase.opaque_handle();
  launch.omega = static_cast<const double *>(d_omega.opaque_handle());
  launch.omega_offset = 1;
  launch.points = points;
  launch.frequencies = frequencies;
  launch.avg1 = 1;
  launch.avg2 = 7;
  launch.dV0 = 0.8;
  launch.dV1 = 0.13;
  launch.scale_real = 0.31;
  launch.scale_imag = -0.27;
  launch.decimation_factor = 2;
  launch.include_weights = true;
  launch.sqrt_weights = true;
  launch.field_precision = field_precision;
  launch.monitor_precision = monitor_precision;
  for (int axis = 0; axis < 3; ++axis) {
    launch.start0[axis] = 0.55 + 0.03 * axis;
    launch.start1[axis] = 0.65 + 0.03 * axis;
    launch.end0[axis] = 0.75 + 0.03 * axis;
    launch.end1[axis] = 0.85 + 0.03 * axis;
  }

  const double sample_times[] = {0.375, 0.625};
  for (double sample_time : sample_times) {
    launch_dft(launch, sample_time, execution);
    for (size_t voxel = 0; voxel < points; ++voxel) {
      size_t linear = voxel;
      const size_t i2 = linear % launch.region.counts[2];
      linear /= launch.region.counts[2];
      const size_t i1 = linear % launch.region.counts[1];
      const size_t i0 = linear / launch.region.counts[1];
      const ptrdiff_t index = ptrdiff_t(launch.region.base) + ptrdiff_t(i0) * 24 +
                              ptrdiff_t(i1) * 8 + ptrdiff_t(i2) * 2;
      double weight = boundary_weight(i2, 3, launch.start0[2], launch.start1[2], launch.end0[2],
                                      launch.end1[2]) *
                      (boundary_weight(i1, 1, launch.start0[1], launch.start1[1], launch.end0[1],
                                       launch.end1[1]) *
                       ((launch.dV0 + launch.dV1 * double(i1)) *
                        boundary_weight(i0, 2, launch.start0[0], launch.start1[0], launch.end0[0],
                                        launch.end1[0])));
      weight = std::sqrt(weight);
      const FieldT scale = FieldT(weight) * FieldT(0.25);
      const FieldT fr = scale * (((source_real[index] + source_real[index + launch.avg1]) +
                                  source_real[index + launch.avg2]) +
                                 source_real[index + launch.avg1 + launch.avg2]);
      const FieldT fi = scale * (((source_imag[index] + source_imag[index + launch.avg1]) +
                                  source_imag[index + launch.avg2]) +
                                 source_imag[index + launch.avg1 + launch.avg2]);
      for (size_t frequency = 0; frequency < frequencies; ++frequency) {
        const double angle = omega[1 + frequency] * sample_time;
        const MonitorT pr =
            MonitorT(std::cos(angle) * launch.scale_real - std::sin(angle) * launch.scale_imag);
        const MonitorT pi =
            MonitorT(std::cos(angle) * launch.scale_imag + std::sin(angle) * launch.scale_real);
        const size_t output = voxel * frequencies + frequency;
        accumulator[2 * output] += pr * MonitorT(fr) - pi * MonitorT(fi);
        accumulator[2 * output + 1] += pr * MonitorT(fi) + pi * MonitorT(fr);
        phase[2 * frequency] = pr;
        phase[2 * frequency + 1] = pi;
      }
    }
  }

  std::vector<MonitorT> observed_accumulator(accumulator.size());
  std::vector<MonitorT> observed_phase(phase.size());
  copy_device_to_host_async(observed_accumulator.data(), d_accumulator, 0, d_accumulator.size(),
                            execution);
  copy_device_to_host_async(observed_phase.data(), d_phase, 0, d_phase.size(), execution);
  execution.synchronize();
  const double tolerance = sizeof(MonitorT) == sizeof(float) ? 2e-6 : 2e-13;
  for (size_t i = 0; i < accumulator.size(); ++i)
    require(std::fabs(double(observed_accumulator[i] - accumulator[i])) <=
                tolerance * (1.0 + std::fabs(double(accumulator[i]))),
            "DFT accumulator differs from host reference");
  for (size_t i = 0; i < phase.size(); ++i)
    require(std::fabs(double(observed_phase[i] - phase[i])) <=
                tolerance * (1.0 + std::fabs(double(phase[i]))),
            "DFT phase differs from host reference");
}

int main() {
  try {
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA device available");
    run_precision_pair<double, double>(0, scalar_precision::f64, scalar_precision::f64);
    run_precision_pair<float, double>(0, scalar_precision::f32, scalar_precision::f64);
    run_precision_pair<float, float>(0, scalar_precision::f32, scalar_precision::f32);
    std::cout << "nvidia_dft_smoke: PASS\n";
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_dft_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
}
