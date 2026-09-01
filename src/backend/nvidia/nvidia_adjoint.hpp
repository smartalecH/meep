/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_ADJOINT_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_ADJOINT_HPP

#include <stddef.h>
#include <stdint.h>

#include "backend/nvidia/runtime.hpp"

namespace meep {
namespace nvidia {

struct adjoint_complex64 { double real, imag; };

struct adjoint_contribution {
  uint64_t adjoint_index;
  uint64_t forward_index[2];
  double forward_weight[2];
  uint32_t forward_count;
  uint32_t reserved;
  double adjoint_coefficient_real;
  double adjoint_coefficient_imag;
  double material_coefficient_real;
  double material_coefficient_imag;
  double accumulation_scale;
};

struct adjoint_launch {
  const adjoint_complex64 *forward;
  const adjoint_complex64 *adjoint;
  const adjoint_contribution *contributions;
  const uint64_t *offsets;
  double *result;
  size_t forward_values;
  size_t adjoint_values;
  size_t contribution_count;
  size_t result_count;
};

void validate_adjoint_launch(const adjoint_launch &launch);
void launch_adjoint_gradient(const adjoint_launch &launch, const stream &stream);

} // namespace nvidia
} // namespace meep

#endif
