/* Deterministic low-level parity and failure test for NVIDIA adjoint accumulation. */

#include "backend/nvidia/nvidia_adjoint.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace meep::nvidia;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  try {
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA device available");
    device_scope selected(0);
    stream execution;

    const adjoint_complex64 forward[] = {{1.0, 2.0}, {-0.5, 0.25}, {3.0, -1.0}};
    const adjoint_complex64 adjoint[] = {{0.75, -0.5}, {2.0, 1.0}};
    adjoint_contribution contributions[3] = {};
    contributions[0].adjoint_index = 0;
    contributions[0].forward_index[0] = 0;
    contributions[0].forward_weight[0] = 1.0;
    contributions[0].forward_count = 1;
    contributions[0].adjoint_coefficient_real = 1.0;
    contributions[0].material_coefficient_real = 0.5;
    contributions[0].material_coefficient_imag = -0.25;
    contributions[0].accumulation_scale = 2.0;
    contributions[1] = contributions[0];
    contributions[1].forward_index[0] = UINT64_MAX;
    contributions[1].forward_weight[0] = 0.5;
    contributions[2] = contributions[0];
    contributions[2].adjoint_index = 1;
    contributions[2].forward_index[0] = 2;
    contributions[2].accumulation_scale = -0.25;
    const uint64_t offsets[] = {0, 2, 2, 3};
    double result[3] = {};

    device_buffer d_forward(sizeof(forward), 0), d_adjoint(sizeof(adjoint), 0);
    device_buffer d_contributions(sizeof(contributions), 0), d_offsets(sizeof(offsets), 0);
    device_buffer d_result(sizeof(result), 0);
    copy_host_to_device_async(d_forward, 0, forward, sizeof(forward), execution);
    copy_host_to_device_async(d_adjoint, 0, adjoint, sizeof(adjoint), execution);
    copy_host_to_device_async(d_contributions, 0, contributions, sizeof(contributions), execution);
    copy_host_to_device_async(d_offsets, 0, offsets, sizeof(offsets), execution);
    adjoint_launch launch = {
        static_cast<const adjoint_complex64 *>(d_forward.opaque_handle()),
        static_cast<const adjoint_complex64 *>(d_adjoint.opaque_handle()),
        static_cast<const adjoint_contribution *>(d_contributions.opaque_handle()),
        static_cast<const uint64_t *>(d_offsets.opaque_handle()),
        static_cast<double *>(d_result.opaque_handle()), 3, 2, 3, 3};

    adjoint_launch overflowing = launch;
    overflowing.result_count = std::numeric_limits<size_t>::max();
    bool rejected = false;
    try { validate_adjoint_launch(overflowing); }
    catch (const std::overflow_error &) { rejected = true; }
    require(rejected, "adjoint launch accepted an overflowing CUDA result grid");

    testing::fail_next(testing::failure_point::adjoint_launch);
    rejected = false;
    try { launch_adjoint_gradient(launch, execution); }
    catch (const std::exception &) { rejected = true; }
    require(rejected, "injected adjoint launch failure was not observed");
    testing::clear_failure();

    launch_adjoint_gradient(launch, execution);
    copy_device_to_host_async(result, d_result, 0, sizeof(result), execution);
    execution.synchronize();
    require(std::fabs(result[0] - 2.25) < 1e-14, "adjoint output 0 differs from oracle");
    require(result[1] == 0.0, "empty adjoint output was not zero");
    require(std::fabs(result[2] + 0.9375) < 1e-14, "adjoint output 2 differs from oracle");
    std::cout << "nvidia_adjoint_smoke: PASS\n";
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_adjoint_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
}
