/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_adjoint.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>

namespace meep {
namespace nvidia {
namespace {

void check_cuda(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) return;
  throw runtime_error(operation, static_cast<int>(result), cudaGetErrorName(result),
                      cudaGetErrorString(result));
}

__device__ adjoint_complex64 multiply(adjoint_complex64 a, adjoint_complex64 b) {
  return {a.real * b.real - a.imag * b.imag,
          a.real * b.imag + a.imag * b.real};
}

__global__ void adjoint_gradient_kernel(adjoint_launch launch) {
  const size_t output = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output >= launch.result_count) return;
  double total = 0.0;
  for (uint64_t index = launch.offsets[output]; index < launch.offsets[output + 1]; ++index) {
    const adjoint_contribution contribution = launch.contributions[index];
    adjoint_complex64 forward = {0.0, 0.0};
    for (uint32_t i = 0; i < contribution.forward_count; ++i)
      if (contribution.forward_index[i] != UINT64_MAX) {
        const adjoint_complex64 value = launch.forward[contribution.forward_index[i]];
        forward.real += value.real * contribution.forward_weight[i];
        forward.imag += value.imag * contribution.forward_weight[i];
      }
    const adjoint_complex64 adjoint = launch.adjoint[contribution.adjoint_index];
    const adjoint_complex64 adjusted =
        multiply(adjoint, {contribution.adjoint_coefficient_real,
                           contribution.adjoint_coefficient_imag});
    const adjoint_complex64 material =
        multiply(forward, {contribution.material_coefficient_real,
                           contribution.material_coefficient_imag});
    total += multiply(adjusted, material).real * contribution.accumulation_scale;
  }
  launch.result[output] = total;
}

} // namespace

void validate_adjoint_launch(const adjoint_launch &launch) {
  if (!launch.forward || !launch.adjoint || !launch.contributions || !launch.offsets ||
      !launch.result || !launch.forward_values || !launch.adjoint_values ||
      !launch.contribution_count || !launch.result_count)
    throw std::invalid_argument("NVIDIA adjoint launch is incomplete");
  if ((launch.result_count - 1) / size_t(256) >= std::numeric_limits<unsigned>::max())
    throw std::overflow_error("NVIDIA adjoint result grid overflows");
}

void launch_adjoint_gradient(const adjoint_launch &launch, const stream &execution) {
  validate_adjoint_launch(launch);
  if (testing::consume_failure_for_testing(testing::failure_point::adjoint_launch))
    throw runtime_error("launching NVIDIA adjoint gradient", int(cudaErrorLaunchFailure),
                        cudaGetErrorName(cudaErrorLaunchFailure),
                        "injected CUDA adjoint launch failure");
  const unsigned blocks = unsigned(1 + (launch.result_count - 1) / 256);
  const cudaStream_t cuda_stream = static_cast<cudaStream_t>(execution.opaque_handle());
  adjoint_gradient_kernel<<<blocks, 256, 0, cuda_stream>>>(launch);
  check_cuda(cudaPeekAtLastError(), "launching NVIDIA adjoint gradient");
}

} // namespace nvidia
} // namespace meep
