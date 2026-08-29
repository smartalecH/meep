/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_backend.hpp"
#include "backend/nvidia/nvidia_coordinates.hpp"

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "backend/initialization_plan.hpp"
#include "backend/descriptors.hpp"
#include "backend/diagnostics.hpp"
#include "backend/halo_plan.hpp"
#include "backend/nvidia/arena.hpp"
#include "backend/nvidia/nvidia_dft.hpp"
#include "backend/nvidia/nvidia_magnetic.hpp"
#include "backend/nvidia/nvidia_polarization.hpp"
#include "backend/nvidia/runtime.hpp"
#include "backend/nvidia/nvidia_sources.hpp"
#include "backend/nvidia/nvidia_step.hpp"
#include "meep_internals.hpp"

namespace meep {

namespace nvidia {

polarization_coefficients derive_polarization_coefficients(double omega_0, double gamma,
                                                           double dt_value, bool drude) {
  const realnum omega2pi = 2 * pi * omega_0, g2pi = gamma * 2 * pi;
  const realnum dt = dt_value;
  const realnum omega0dtsqr = omega2pi * omega2pi * dt * dt;
  const realnum gamma1inv = 1 / (1 + g2pi * dt / 2), gamma1 = (1 - g2pi * dt / 2);
  polarization_coefficients result = {double(omega0dtsqr), double(gamma1inv), double(gamma1),
                                      drude ? 0.0 : double(omega0dtsqr)};
  return result;
}

gyrotropic_coefficients derive_gyrotropic_coefficients(double omega_0, double gamma,
                                                       double alpha,
                                                       const double gyro_tensor[3][3],
                                                       double dt_value, gyrotropy_model model,
                                                       const direction order[3]) {
  const realnum omega_0_r = omega_0, gamma_r = gamma, alpha_r = alpha, dt = dt_value;
  realnum gyro[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) gyro[i][j] = gyro_tensor[i][j];
  const realnum omega = 2 * pi * omega_0_r * dt;
  const realnum gamma_dt = 2 * pi * gamma_r * dt;
  const realnum dt2pi = 2 * pi * dt;
  realnum gd, gx, gy, gz;
  gyrotropic_coefficients result = {};
  result.omega = omega;
  result.gamma = gamma_dt;
  result.alpha = alpha_r;
  result.dt2pi = dt2pi;
  if (model == GYROTROPIC_SATURATED) {
    gd = 0.5;
    gx = -0.5 * alpha_r * gyro[Y][Z];
    gy = -0.5 * alpha_r * gyro[Z][X];
    gz = -0.5 * alpha_r * gyro[X][Y];
  }
  else {
    result.omega0dtsqr = omega * omega;
    result.gamma1 = 1 - gamma_dt / 2;
    result.diagonal = 2 - (model == GYROTROPIC_DRUDE ? 0 : realnum(result.omega0dtsqr));
    const realnum pt = pi * dt;
    result.pt = pt;
    gd = 1 + gamma_dt / 2;
    gx = pt * gyro[Y][Z];
    gy = pt * gyro[Z][X];
    gz = pt * gyro[X][Y];
  }
  const realnum invdet = 1.0 / gd / (gd * gd + gx * gx + gy * gy + gz * gz);
  const realnum inverse[3][3] = {
      {invdet * (gd * gd + gx * gx), invdet * (gx * gy + gd * gz),
       invdet * (gx * gz - gd * gy)},
      {invdet * (gy * gx - gd * gz), invdet * (gd * gd + gy * gy),
       invdet * (gy * gz + gd * gx)},
      {invdet * (gz * gx + gd * gy), invdet * (gz * gy - gd * gx),
       invdet * (gd * gd + gz * gz)}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      result.gyro[i][j] = gyro[int(order[i])][int(order[j])];
      result.inverse[i][j] = inverse[int(order[i])][int(order[j])];
    }
  return result;
}

} // namespace nvidia

namespace {

size_t checked_product(size_t left, size_t right, const char *what) {
  if (left && right > std::numeric_limits<size_t>::max() / left)
    throw std::overflow_error(std::string("overflow while ") + what);
  return left * right;
}

size_t checked_add(size_t left, size_t right, const char *what) {
  if (right > std::numeric_limits<size_t>::max() - left)
    throw std::overflow_error(std::string("overflow while ") + what);
  return left + right;
}

nvidia::arena_role arena_role_for(array_role role) {
  switch (role) {
    case array_role::field: return nvidia::arena_role::field;
    case array_role::material: return nvidia::arena_role::material;
    case array_role::polarization: return nvidia::arena_role::polarization;
    case array_role::dft: return nvidia::arena_role::dft;
    case array_role::communication: return nvidia::arena_role::communication;
    case array_role::scratch: return nvidia::arena_role::scratch;
  }
  throw std::invalid_argument("storage plan contains an invalid array role");
}

nvidia::arena_element_type arena_element_type_for(ElementType type) {
  switch (type) {
    case ElementType::realnum_value: return nvidia::arena_element_type::realnum_value;
    case ElementType::complex_realnum: return nvidia::arena_element_type::complex_realnum;
    case ElementType::float64: return nvidia::arena_element_type::float64;
    case ElementType::complex_float64: return nvidia::arena_element_type::complex_float64;
    case ElementType::int32: return nvidia::arena_element_type::int32;
    case ElementType::index: return nvidia::arena_element_type::index;
  }
  throw std::invalid_argument("storage plan contains an invalid element type");
}

nvidia::arena_storage_precision arena_precision_for(ElementType type, Precision precision) {
  switch (type) {
    case ElementType::realnum_value:
    case ElementType::complex_realnum:
      switch (precision) {
        case Precision::f32: return nvidia::arena_storage_precision::f32;
        case Precision::f64: return nvidia::arena_storage_precision::f64;
      }
      break;
    case ElementType::float64:
    case ElementType::complex_float64:
    case ElementType::int32:
    case ElementType::index: return nvidia::arena_storage_precision::not_applicable;
  }
  throw std::invalid_argument("storage plan contains an invalid storage precision");
}

size_t storage_alignment_for(const ArraySpec &spec) {
  size_t required = 1;
  switch (spec.element_type) {
    case ElementType::realnum_value:
      required = spec.storage == Precision::f32 ? alignof(float) : alignof(double);
      break;
    case ElementType::complex_realnum:
      required = spec.storage == Precision::f32 ? alignof(std::complex<float>)
                                                : alignof(std::complex<double>);
      break;
    case ElementType::float64: required = alignof(double); break;
    case ElementType::complex_float64: required = alignof(std::complex<double>); break;
    case ElementType::int32: required = alignof(int32_t); break;
    case ElementType::index: required = alignof(size_t); break;
  }
  return spec.alignment < required ? required : spec.alignment;
}

std::vector<nvidia::allocation_request> allocation_requests_for(const StoragePlan &plan) {
  if (plan.arrays.size() != plan.keys.size())
    throw std::invalid_argument("storage plan array/key lengths differ");

  std::vector<nvidia::allocation_request> requests;
  requests.reserve(plan.arrays.size());
  for (size_t i = 0; i < plan.arrays.size(); ++i) {
    const ArraySpec &spec = plan.arrays[i];
    if (!is_valid(spec.id) || spec.id.value != i)
      throw std::invalid_argument("storage plan ArrayIds are not dense and canonical");
    if (!spec.elements)
      throw std::invalid_argument("storage plan contains a zero-element allocation");
    if (!spec.alignment || (spec.alignment & (spec.alignment - 1)))
      throw std::invalid_argument("storage plan contains a non-power-of-two alignment");
    if (is_valid(spec.alias_of) && spec.alias_of.value >= plan.arrays.size())
      throw std::out_of_range("storage plan alias target is outside the plan");

    requests.push_back(nvidia::allocation_request(
        spec.id.value, arena_role_for(spec.role), storage_bytes(spec), storage_alignment_for(spec),
        is_valid(spec.alias_of) ? spec.alias_of.value : nvidia::no_allocation,
        arena_element_type_for(spec.element_type),
        arena_precision_for(spec.element_type, spec.storage)));
  }
  return requests;
}

uint64_t mix_fingerprint(uint64_t hash, uint64_t value) {
  for (unsigned int i = 0; i < 8; ++i) {
    hash ^= (value >> (i * 8)) & 0xffu;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

uint64_t storage_fingerprint(const StoragePlan &plan) {
  uint64_t hash = UINT64_C(1469598103934665603);
  hash = mix_fingerprint(hash, plan.arrays.size());
  for (size_t i = 0; i < plan.arrays.size(); ++i) {
    const ArraySpec &spec = plan.arrays[i];
    const StorageKey &key = plan.keys[i];
    hash = mix_fingerprint(hash, spec.id.value);
    hash = mix_fingerprint(hash, static_cast<unsigned int>(spec.role));
    hash = mix_fingerprint(hash, static_cast<unsigned int>(spec.element_type));
    hash = mix_fingerprint(hash, static_cast<unsigned int>(spec.storage));
    hash = mix_fingerprint(hash, spec.elements);
    hash = mix_fingerprint(hash, spec.alignment);
    hash =
        mix_fingerprint(hash, is_valid(spec.alias_of) ? spec.alias_of.value : invalid_array_value);
    hash = mix_fingerprint(hash, static_cast<uint64_t>(static_cast<int64_t>(key.chunk)));
    hash = mix_fingerprint(hash, static_cast<uint64_t>(static_cast<int64_t>(key.kind)));
    hash = mix_fingerprint(hash, static_cast<uint64_t>(static_cast<int64_t>(key.component_)));
    hash = mix_fingerprint(hash, static_cast<uint64_t>(static_cast<int64_t>(key.cmp)));
    hash = mix_fingerprint(hash, static_cast<uint64_t>(static_cast<int64_t>(key.aux)));
  }
  return hash;
}

template <typename Source, typename Destination>
void convert_real_values(void *destination, const void *source, size_t count) {
  const char *input = static_cast<const char *>(source);
  char *output = static_cast<char *>(destination);
  for (size_t i = 0; i < count; ++i) {
    Source source_value;
    memcpy(&source_value, input + i * sizeof(Source), sizeof(Source));
    const Destination destination_value = static_cast<Destination>(source_value);
    memcpy(output + i * sizeof(Destination), &destination_value, sizeof(Destination));
  }
}

template <typename Source, typename Destination>
void convert_complex_values(void *destination, const void *source, size_t count) {
  const char *input = static_cast<const char *>(source);
  char *output = static_cast<char *>(destination);
  for (size_t i = 0; i < count; ++i) {
    std::complex<Source> source_value;
    memcpy(&source_value, input + i * sizeof(source_value), sizeof(source_value));
    const std::complex<Destination> destination_value(
        static_cast<Destination>(source_value.real()),
        static_cast<Destination>(source_value.imag()));
    memcpy(output + i * sizeof(destination_value), &destination_value, sizeof(destination_value));
  }
}

void host_to_storage(void *destination, const void *source, const ArraySpec &spec,
                     size_t elements) {
  switch (spec.element_type) {
    case ElementType::realnum_value:
      if (spec.storage == native_precision)
        memcpy(destination, source,
               checked_product(elements, sizeof(realnum), "copying host data"));
      else if (spec.storage == Precision::f32)
        convert_real_values<realnum, float>(destination, source, elements);
      else
        convert_real_values<realnum, double>(destination, source, elements);
      return;
    case ElementType::complex_realnum:
      static_assert(sizeof(std::complex<float>) == 2 * sizeof(float),
                    "NVIDIA complex<float> storage must be packed");
      static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
                    "NVIDIA complex<double> storage must be packed");
      if (spec.storage == native_precision)
        memcpy(destination, source,
               checked_product(elements, sizeof(std::complex<realnum>), "copying host data"));
      else if (spec.storage == Precision::f32)
        convert_complex_values<realnum, float>(destination, source, elements);
      else
        convert_complex_values<realnum, double>(destination, source, elements);
      return;
    case ElementType::float64:
    case ElementType::complex_float64:
    case ElementType::int32:
    case ElementType::index:
      memcpy(destination, source,
             checked_product(elements, host_element_bytes(spec.element_type),
                             "copying fixed-width host data"));
      return;
  }
  throw std::invalid_argument("cannot convert an invalid element type");
}

void storage_to_host(void *destination, const void *source, const ArraySpec &spec,
                     size_t elements) {
  switch (spec.element_type) {
    case ElementType::realnum_value:
      if (spec.storage == native_precision)
        memcpy(destination, source,
               checked_product(elements, sizeof(realnum), "reading host data"));
      else if (spec.storage == Precision::f32)
        convert_real_values<float, realnum>(destination, source, elements);
      else
        convert_real_values<double, realnum>(destination, source, elements);
      return;
    case ElementType::complex_realnum:
      if (spec.storage == native_precision)
        memcpy(destination, source,
               checked_product(elements, sizeof(std::complex<realnum>), "reading host data"));
      else if (spec.storage == Precision::f32)
        convert_complex_values<float, realnum>(destination, source, elements);
      else
        convert_complex_values<double, realnum>(destination, source, elements);
      return;
    case ElementType::float64:
    case ElementType::complex_float64:
    case ElementType::int32:
    case ElementType::index:
      memcpy(destination, source,
             checked_product(elements, host_element_bytes(spec.element_type),
                             "reading fixed-width host data"));
      return;
  }
  throw std::invalid_argument("cannot convert an invalid element type");
}

struct AccessRange {
  const ArraySpec *spec;
  size_t storage_offset;
  size_t storage_bytes;
};

AccessRange checked_access(const StoragePlan &plan, ArrayRef ref, const void *host_buffer,
                           size_t host_bytes) {
  if (!is_valid(ref.id) || ref.id.value >= plan.arrays.size())
    throw std::out_of_range("NVIDIA backend access uses an invalid ArrayId");
  const ArraySpec &spec = plan.arrays[ref.id.value];
  if (ref.offset > spec.elements || ref.elements > spec.elements - ref.offset)
    throw std::out_of_range("NVIDIA backend access exceeds the registered array");
  const size_t expected_host_bytes = checked_product(
      ref.elements, host_element_bytes(spec.element_type), "validating NVIDIA host byte count");
  if (host_bytes != expected_host_bytes)
    throw std::invalid_argument(
        "NVIDIA backend byte count does not match the ArrayRef host representation");
  if (host_bytes && !host_buffer)
    throw std::invalid_argument("NVIDIA backend access has a null host buffer");
  AccessRange result;
  result.spec = &spec;
  result.storage_offset =
      checked_product(ref.offset, storage_element_bytes(spec.element_type, spec.storage),
                      "resolving NVIDIA storage offset");
  result.storage_bytes =
      checked_product(ref.elements, storage_element_bytes(spec.element_type, spec.storage),
                      "resolving NVIDIA storage byte count");
  return result;
}

} // namespace

class NvidiaBackendState : public BackendState {
public:
  NvidiaBackendState(NvidiaBackend *owner, StoragePlan plan, int device, uint64_t state_token)
      : owner_(owner), plan_(plan), layout_(allocation_requests_for(plan_)), device_(device),
        state_token_(state_token), fingerprint_(storage_fingerprint(plan_)), initialized_(false),
        device_authoritative_(false), magnetic_snapshot_bytes_(0),
        magnetic_snapshot_layout_fingerprint_(0), magnetic_snapshot_valid_(false) {
    nvidia::device_scope scope(device_);
    transfer_.reset(new nvidia::stream);
    arenas_.reset(new nvidia::device_arenas(layout_, device_));
  }

  ~NvidiaBackendState() override {}

  void ensure_staging(size_t bytes) {
    if (staging_.size() < bytes) staging_.allocate(bytes);
  }

  void ensure_dft_reduction_buffers(size_t result_bytes, size_t partial_bytes) {
    if (dft_reduction_result_.size() < result_bytes)
      dft_reduction_result_.allocate(result_bytes, device_);
    if (dft_reduction_partials_.size() < partial_bytes)
      dft_reduction_partials_.allocate(partial_bytes, device_);
    ensure_staging(result_bytes);
  }

  NvidiaBackend *owner_;
  StoragePlan plan_;
  nvidia::arena_plan layout_;
  int device_;
  const uint64_t state_token_;
  uint64_t fingerprint_;
  bool initialized_;
  bool transfer_failed_;
  bool device_authoritative_;
  std::unique_ptr<nvidia::device_arenas> arenas_;
  std::unique_ptr<nvidia::stream> transfer_;
  nvidia::pinned_buffer staging_;
  nvidia::device_buffer dft_reduction_result_;
  nvidia::device_buffer dft_reduction_partials_;
  nvidia::device_buffer magnetic_snapshot_;
  size_t magnetic_snapshot_bytes_;
  uint64_t magnetic_snapshot_layout_fingerprint_;
  bool magnetic_snapshot_valid_;
};

struct NvidiaCompiledOperation {
  OpKind kind;
  size_t first;
  size_t count;
  size_t beta_first;
  size_t beta_count;
  size_t cylindrical_m_first;
  size_t cylindrical_m_count;
  size_t cylindrical_origin_first;
  size_t cylindrical_origin_count;
  size_t halo_first;
  size_t halo_count;
  size_t source_first;
  size_t source_count;
  size_t copy_first;
  size_t copy_count;
  size_t polarization_first;
  size_t polarization_count;
  size_t subtraction_first;
  size_t subtraction_count;
  size_t source_staging_offset;
  double source_time_offset;
};

struct NvidiaCompiledCylindricalOriginAction {
  CylindricalOriginActionKind kind;
  size_t index;
};

struct NvidiaCompiledHalo {
  nvidia::halo_launch gather;
  nvidia::halo_launch scatter;
};

struct NvidiaFiniteCheck {
  nvidia::finite_check_launch launch;
  StorageKey key;
};

struct NvidiaCompiledMagneticState {
  void *live;
  size_t backup_offset;
  size_t elements;
  size_t bytes;
  nvidia::scalar_precision precision;
  bool average;
};

class NvidiaExecutable : public Executable {
public:
  NvidiaExecutable(const NvidiaBackend *owner, uint64_t signature, uint64_t storage_fingerprint,
                   uint64_t state_token,
                   const std::vector<NvidiaCompiledOperation> &operations,
                   const std::vector<nvidia::curl_launch> &curl_updates,
                   const std::vector<nvidia::cylindrical_radial_prefix_launch>
                       &cylindrical_radial_prefixes,
                   const std::vector<nvidia::bfast_launch> &bfast_updates,
                   const std::vector<nvidia::beta_launch> &beta_updates,
                   const std::vector<nvidia::cylindrical_m_launch> &cylindrical_m_updates,
                   const std::vector<nvidia::cylindrical_axis_launch> &cylindrical_axis_updates,
                   const std::vector<NvidiaCompiledCylindricalOriginAction>
                       &cylindrical_origin_actions,
                   const std::vector<nvidia::constitutive_launch> &constitutive_updates,
                   const std::vector<nvidia::zero_launch> &zero_updates,
                   const std::vector<NvidiaCompiledHalo> &halo_plans,
                   const std::vector<nvidia::halo_gather_entry> &halo_gathers,
                   const std::vector<nvidia::halo_scatter_entry> &halo_scatters,
                   size_t halo_scratch_bytes, const std::vector<NvidiaFiniteCheck> &finite_checks,
                   const std::vector<nvidia::source_batch_launch> &source_batches,
                   const std::vector<nvidia::source_point> &source_points,
                   const std::vector<nvidia::array_copy_launch> &source_copies,
                   const std::vector<nvidia::compiled_polarization_update>
                       &polarization_updates,
                   const std::vector<nvidia::polarization_subtract_launch>
                       &polarization_subtractions,
                   const std::vector<nvidia::dft_launch> &dft_updates,
                   const std::vector<double> &dft_omega,
                   const std::vector<NvidiaCompiledMagneticState> &magnetic_states,
                   size_t magnetic_snapshot_bytes, uint64_t magnetic_layout_fingerprint,
                   const MagneticHalfStep &magnetic_half_step,
                   size_t source_scalar_count, size_t source_staging_elements,
                   NvidiaBackendState &state)
      : owner_(owner), state_token_(state_token), signature_(signature),
        storage_fingerprint_(storage_fingerprint),
        operations_(operations), curl_updates_(curl_updates),
        cylindrical_radial_prefixes_(cylindrical_radial_prefixes),
        bfast_updates_(bfast_updates), beta_updates_(beta_updates),
        cylindrical_m_updates_(cylindrical_m_updates),
        cylindrical_axis_updates_(cylindrical_axis_updates),
        cylindrical_origin_actions_(cylindrical_origin_actions),
        constitutive_updates_(constitutive_updates), zero_updates_(zero_updates),
        halo_plans_(halo_plans), finite_checks_(finite_checks), source_batches_(source_batches),
        source_copies_(source_copies), polarization_updates_(polarization_updates),
        polarization_subtractions_(polarization_subtractions), dft_updates_(dft_updates),
        magnetic_states_(magnetic_states), magnetic_snapshot_bytes_(magnetic_snapshot_bytes),
        magnetic_layout_fingerprint_(magnetic_layout_fingerprint),
        magnetic_half_step_(magnetic_half_step), source_scalar_count_(source_scalar_count) {
    try {
      nvidia::device_scope scope(state.device_);
      if (!halo_gathers.empty()) {
        halo_gathers_.allocate(checked_product(halo_gathers.size(), sizeof(halo_gathers[0]),
                                               "allocating NVIDIA halo gather descriptors"),
                               state.device_);
        nvidia::copy_host_to_device_async(halo_gathers_, 0, halo_gathers.data(),
                                          halo_gathers_.size(), *state.transfer_);
      }
      if (!halo_scatters.empty()) {
        halo_scatters_.allocate(checked_product(halo_scatters.size(), sizeof(halo_scatters[0]),
                                                "allocating NVIDIA halo scatter descriptors"),
                                state.device_);
        nvidia::copy_host_to_device_async(halo_scatters_, 0, halo_scatters.data(),
                                          halo_scatters_.size(), *state.transfer_);
      }
      if (halo_scratch_bytes) halo_scratch_.allocate(halo_scratch_bytes, state.device_);
      if (!finite_checks_.empty()) {
        finite_result_.allocate(sizeof(uint64_t), state.device_);
        finite_result_host_.allocate(sizeof(uint64_t));
      }
      if (source_scalar_count_) {
        source_scalars_.allocate(checked_product(source_scalar_count_,
                                                 sizeof(nvidia::source_scalar),
                                                 "allocating NVIDIA source scalars"),
                                 state.device_);
        source_staging_.allocate(checked_product(source_staging_elements,
                                                 sizeof(nvidia::source_scalar),
                                                 "allocating NVIDIA source-scalar staging"));
      }
      if (!source_points.empty()) {
        source_points_.allocate(checked_product(source_points.size(), sizeof(source_points[0]),
                                                "allocating NVIDIA source points"),
                                state.device_);
        nvidia::copy_host_to_device_async(source_points_, 0, source_points.data(),
                                          source_points_.size(), *state.transfer_);
        const nvidia::source_point *point_base =
            static_cast<const nvidia::source_point *>(source_points_.opaque_handle());
        for (size_t i = 0; i < source_batches_.size(); ++i)
          source_batches_[i].points = point_base + source_batches_[i].point_offset;
      }
      if (!dft_omega.empty()) {
        dft_omega_.allocate(checked_product(dft_omega.size(), sizeof(double),
                                            "allocating NVIDIA DFT frequencies"),
                            state.device_);
        nvidia::copy_host_to_device_async(dft_omega_, 0, dft_omega.data(), dft_omega_.size(),
                                          *state.transfer_);
        const double *omega_base = static_cast<const double *>(dft_omega_.opaque_handle());
        for (size_t i = 0; i < dft_updates_.size(); ++i)
          dft_updates_[i].omega = omega_base;
      }
      if (!halo_gathers.empty() || !halo_scatters.empty() || !source_points.empty() ||
          !dft_omega.empty())
        state.transfer_->synchronize();
    }
    catch (...) {
      /* Every destination above belongs to this unpublished executable.
         Cleanup is local, so a preflight compile failure leaves the live
         state and its stream retryable. */
      throw;
    }
  }

  const NvidiaBackend *owner_;
  uint64_t state_token_;
  uint64_t signature_;
  uint64_t storage_fingerprint_;
  std::vector<NvidiaCompiledOperation> operations_;
  std::vector<nvidia::curl_launch> curl_updates_;
  std::vector<nvidia::cylindrical_radial_prefix_launch> cylindrical_radial_prefixes_;
  std::vector<nvidia::bfast_launch> bfast_updates_;
  std::vector<nvidia::beta_launch> beta_updates_;
  std::vector<nvidia::cylindrical_m_launch> cylindrical_m_updates_;
  std::vector<nvidia::cylindrical_axis_launch> cylindrical_axis_updates_;
  std::vector<NvidiaCompiledCylindricalOriginAction> cylindrical_origin_actions_;
  std::vector<nvidia::constitutive_launch> constitutive_updates_;
  std::vector<nvidia::zero_launch> zero_updates_;
  std::vector<NvidiaCompiledHalo> halo_plans_;
  std::vector<NvidiaFiniteCheck> finite_checks_;
  std::vector<nvidia::source_batch_launch> source_batches_;
  std::vector<nvidia::array_copy_launch> source_copies_;
  std::vector<nvidia::compiled_polarization_update> polarization_updates_;
  std::vector<nvidia::polarization_subtract_launch> polarization_subtractions_;
  std::vector<nvidia::dft_launch> dft_updates_;
  std::vector<NvidiaCompiledMagneticState> magnetic_states_;
  size_t magnetic_snapshot_bytes_;
  uint64_t magnetic_layout_fingerprint_;
  MagneticHalfStep magnetic_half_step_;
  size_t source_scalar_count_;
  nvidia::device_buffer halo_gathers_;
  nvidia::device_buffer halo_scatters_;
  nvidia::device_buffer halo_scratch_;
  nvidia::device_buffer finite_result_;
  nvidia::pinned_buffer finite_result_host_;
  nvidia::device_buffer source_scalars_;
  nvidia::pinned_buffer source_staging_;
  nvidia::device_buffer source_points_;
  nvidia::device_buffer dft_omega_;
};

namespace {

nvidia::scalar_precision scalar_precision_for(const StoragePlan &plan, ArrayId id,
                                              const char *what) {
  if (!is_valid(id) || id.value >= plan.arrays.size())
    throw std::invalid_argument(std::string(what) + " uses an invalid ArrayId");
  const ArraySpec &spec = plan.arrays[id.value];
  if (spec.element_type != ElementType::realnum_value)
    throw std::invalid_argument(std::string(what) + " is not a realnum array");
  return spec.storage == Precision::f32 ? nvidia::scalar_precision::f32
                                        : nvidia::scalar_precision::f64;
}

nvidia::scalar_precision complex_precision_for(const StoragePlan &plan, ArrayId id,
                                               const char *what) {
  if (!is_valid(id) || id.value >= plan.arrays.size())
    throw std::invalid_argument(std::string(what) + " uses an invalid ArrayId");
  const ArraySpec &spec = plan.arrays[id.value];
  if (spec.element_type != ElementType::complex_realnum)
    throw std::invalid_argument(std::string(what) + " is not a complex-realnum array");
  return spec.storage == Precision::f32 ? nvidia::scalar_precision::f32
                                        : nvidia::scalar_precision::f64;
}

nvidia::scalar_precision scalar_precision_for(Precision precision) {
  return precision == Precision::f32 ? nvidia::scalar_precision::f32
                                     : nvidia::scalar_precision::f64;
}

const ArraySpec &validate_dft_reduction_array(const StoragePlan &plan, ArrayId id,
                                              size_t storage_points, size_t frequencies,
                                              const char *what) {
  if (!is_valid(id) || id.value >= plan.arrays.size())
    throw std::out_of_range(std::string(what) + " uses an invalid ArrayId");
  const ArraySpec &spec = plan.arrays[id.value];
  if (spec.role != array_role::dft || spec.element_type != ElementType::complex_realnum)
    throw std::invalid_argument(std::string(what) + " is not a DFT complex-realnum array");
  if (is_valid(spec.alias_of))
    throw std::invalid_argument(std::string(what) + " must not be an alias");
  const size_t elements =
      checked_product(storage_points, frequencies, "validating DFT reduction array shape");
  if (!storage_points || !frequencies || elements != spec.elements)
    throw std::out_of_range(std::string(what) + " has an incompatible logical shape");
  return spec;
}

size_t validate_dft_reduction_region(const DftReductionTerm &term) {
  size_t selected = 1;
  size_t maximum = term.region.base;
  for (int axis = 0; axis < 3; ++axis) {
    if (!term.region.counts[axis])
      throw std::invalid_argument("NVIDIA DFT reduction region has a zero count");
    selected = checked_product(selected, term.region.counts[axis],
                               "validating DFT reduction region size");
    const size_t extent = term.region.counts[axis] - 1;
    if (term.region.strides[axis] &&
        extent > (std::numeric_limits<size_t>::max() - maximum) / term.region.strides[axis])
      throw std::overflow_error("NVIDIA DFT reduction region index overflow");
    maximum += extent * term.region.strides[axis];
  }
  if (maximum >= term.storage_points)
    throw std::out_of_range("NVIDIA DFT reduction region exceeds monitor storage");
  return selected;
}

void require_same_precision(const StoragePlan &plan, ArrayId id, nvidia::scalar_precision precision,
                            const char *what) {
  if (!is_valid(id)) return;
  if (scalar_precision_for(plan, id, what) != precision)
    throw std::invalid_argument(std::string(what) + " has a different storage precision");
}

void *device_address(NvidiaBackendState &state, ArrayId id, const char *what) {
  (void)scalar_precision_for(state.plan_, id, what);
  return state.arenas_->resolve(id.value).address;
}

void *complex_device_address(NvidiaBackendState &state, ArrayId id, const char *what) {
  (void)complex_precision_for(state.plan_, id, what);
  return state.arenas_->resolve(id.value).address;
}

const void *optional_device_address(NvidiaBackendState &state, ArrayId id,
                                    nvidia::scalar_precision precision, const char *what) {
  if (!is_valid(id)) return NULL;
  require_same_precision(state.plan_, id, precision, what);
  return state.arenas_->resolve(id.value).address;
}

void *optional_mutable_device_address(NvidiaBackendState &state, ArrayId id,
                                      nvidia::scalar_precision precision, const char *what) {
  if (!is_valid(id)) return NULL;
  require_same_precision(state.plan_, id, precision, what);
  return state.arenas_->resolve(id.value).address;
}

array_kind magnetic_array_kind(MagneticStateFamily family) {
  switch (family) {
    case MagneticStateFamily::primary: return array_kind::f;
    case MagneticStateFamily::u: return array_kind::f_u;
    case MagneticStateFamily::w: return array_kind::f_w;
    case MagneticStateFamily::conductivity: return array_kind::f_cond;
    case MagneticStateFamily::bfast: return array_kind::f_bfast;
  }
  throw std::invalid_argument("NVIDIA magnetic state family is invalid");
}

NvidiaCompiledMagneticState compile_magnetic_state(const MagneticStateArray &source,
                                                   NvidiaBackendState &state, const fields &f,
                                                   size_t &next_offset) {
  if (source.chunk < 0 || source.chunk >= f.num_chunks || !f.chunks[source.chunk] ||
      !f.chunks[source.chunk]->is_mine())
    throw std::out_of_range("NVIDIA magnetic state has an invalid chunk");
  if (!is_B(source.c) && !is_magnetic(source.c))
    throw std::invalid_argument("NVIDIA magnetic state has a nonmagnetic component");
  if (source.cmp < 0 || source.cmp > 1)
    throw std::invalid_argument("NVIDIA magnetic state has an invalid complex plane");
  const nvidia::scalar_precision precision =
      scalar_precision_for(state.plan_, source.live, "NVIDIA magnetic live state");
  const ArraySpec &spec = state.plan_.arrays[source.live.value];
  const StorageKey &key = state.plan_.keys[source.live.value];
  if (spec.role != array_role::field || is_valid(spec.alias_of))
    throw std::invalid_argument("NVIDIA magnetic state must name canonical field storage");
  if (key.chunk != source.chunk || key.kind != int(magnetic_array_kind(source.family)) ||
      key.component_ != int(source.c) || key.cmp != source.cmp || key.aux != 0)
    throw std::invalid_argument("NVIDIA magnetic state identity does not match its ArrayId");
  if (!source.elements || source.elements != spec.elements)
    throw std::invalid_argument("NVIDIA magnetic state extent does not match its ArrayId");
  if (source.average != (source.family == MagneticStateFamily::primary))
    throw std::invalid_argument("NVIDIA magnetic state has an invalid average flag");

  const size_t element_bytes = precision == nvidia::scalar_precision::f32 ? sizeof(float)
                                                                           : sizeof(double);
  const size_t padding = (element_bytes - next_offset % element_bytes) % element_bytes;
  next_offset = checked_add(next_offset, padding, "aligning NVIDIA magnetic snapshot");
  NvidiaCompiledMagneticState result = {
      state.arenas_->resolve(source.live.value).address, next_offset, source.elements,
      checked_product(source.elements, element_bytes, "sizing NVIDIA magnetic snapshot"),
      precision, source.average};
  next_offset = checked_add(next_offset, result.bytes, "laying out NVIDIA magnetic snapshot");
  return result;
}

uint64_t magnetic_layout_fingerprint(uint64_t storage_fingerprint,
                                     const std::vector<MagneticStateArray> &rows) {
  uint64_t hash = mix_fingerprint(storage_fingerprint, rows.size());
  for (const MagneticStateArray &row : rows) {
    hash = mix_fingerprint(hash, uint64_t(int64_t(row.chunk)));
    hash = mix_fingerprint(hash, uint64_t(int64_t(row.c)));
    hash = mix_fingerprint(hash, uint64_t(int64_t(row.cmp)));
    hash = mix_fingerprint(hash, uint64_t(row.family));
    hash = mix_fingerprint(hash, row.live.value);
    hash = mix_fingerprint(hash, row.elements);
    hash = mix_fingerprint(hash, row.average ? 1 : 0);
  }
  return hash;
}

void validate_index_range(const StoragePlan &plan, ArrayId id, ptrdiff_t minimum, ptrdiff_t maximum,
                          const char *what);

void validate_ref_index_range(const StoragePlan &plan, const ArrayRef &ref, ptrdiff_t minimum,
                              ptrdiff_t maximum, const char *what) {
  if (!is_valid(ref.id) || ref.id.value >= plan.arrays.size())
    throw std::invalid_argument(std::string(what) + " uses an invalid ArrayId");
  const ArraySpec &spec = plan.arrays[ref.id.value];
  if (ref.offset > spec.elements || ref.elements > spec.elements - ref.offset)
    throw std::out_of_range(std::string(what) + " declares an invalid source span");
  if (minimum < 0 || maximum < minimum || size_t(minimum) < ref.offset ||
      size_t(maximum) >= ref.offset + ref.elements)
    throw std::out_of_range(std::string(what) + " index range exceeds its declared span");
}

nvidia::source_batch_launch compile_source_batch(const SourceDescriptor &source,
                                                 const SourcePlan &source_plan,
                                                 const fields &f, NvidiaBackendState &state,
                                                 std::vector<nvidia::source_point> &points) {
  if (source.indices.empty() || source.indices.size() != source.complex_amplitudes.size())
    throw std::invalid_argument("NVIDIA source descriptor has invalid spatial data");
  if (source.source_time_id >= source_plan.source_times.size())
    throw std::out_of_range("source descriptor has an invalid source-time ID");
  const SourceTimeDescriptor &time = source_plan.source_times[source.source_time_id];
  if (time.source_time_id != source.source_time_id ||
      time.scalar_slot >= source_plan.scalars.size())
    throw std::invalid_argument("source descriptor has a non-canonical scalar mapping");
  if (time.is_integrated != source.integrated)
    throw std::invalid_argument("source descriptor and source-time integration modes disagree");

  nvidia::source_batch_launch result = {};
  const ArrayId target = source.integrated ? source.integrated_destination : source.destination;
  const ArrayId target_imag =
      source.integrated ? source.integrated_destination_imag : source.destination_imag;
  result.precision = scalar_precision_for(state.plan_, target, "source target");
  result.target_real = device_address(state, target, "source target");
  result.target_imag = optional_mutable_device_address(state, target_imag, result.precision,
                                                       "source imaginary target");
  result.conductivity_inverse =
      source.integrated ? NULL
                        : optional_device_address(state, source.condinv, result.precision,
                                                  "source conductivity inverse");
  result.point_offset = points.size();
  result.point_count = source.indices.size();
  result.scalar_slot = time.scalar_slot;
  result.dt = f.dt;
  result.integrated = source.integrated;
  result.sequential =
      nvidia::source_indices_require_sequential(source.indices.data(), source.indices.size());
  const std::pair<std::vector<ptrdiff_t>::const_iterator,
                  std::vector<ptrdiff_t>::const_iterator> bounds =
      std::minmax_element(source.indices.begin(), source.indices.end());
  validate_index_range(state.plan_, target, *bounds.first, *bounds.second, "source target");
  if (is_valid(target_imag))
    validate_index_range(state.plan_, target_imag, *bounds.first, *bounds.second,
                         "source imaginary target");
  if (!source.integrated && is_valid(source.condinv))
    validate_index_range(state.plan_, source.condinv, *bounds.first, *bounds.second,
                         "source conductivity inverse");
  if (f.is_real && result.target_imag)
    throw std::invalid_argument("real NVIDIA fields have an imaginary source target");
  if (!f.is_real && !result.target_imag)
    throw std::invalid_argument("complex NVIDIA fields have no imaginary source target");
  for (size_t point = 0; point < source.indices.size(); ++point) {
    nvidia::source_point packed = {};
    packed.index = source.indices[point];
    packed.amplitude_real = source.complex_amplitudes[point].real();
    packed.amplitude_imag = source.complex_amplitudes[point].imag();
    points.push_back(packed);
  }
  return result;
}

nvidia::array_copy_launch compile_source_copy(ArrayId target, ArrayId source,
                                              NvidiaBackendState &state) {
  if (!is_valid(target) || !is_valid(source))
    throw std::invalid_argument("integrated-source copy has an invalid array");
  if (target.value >= state.plan_.arrays.size() || source.value >= state.plan_.arrays.size())
    throw std::out_of_range("integrated-source copy ArrayId is out of range");
  const ArraySpec &target_spec = state.plan_.arrays[target.value];
  const ArraySpec &source_spec = state.plan_.arrays[source.value];
  if (target_spec.elements != source_spec.elements)
    throw std::invalid_argument("integrated-source copy arrays have different sizes");
  nvidia::array_copy_launch result = {};
  result.precision = scalar_precision_for(state.plan_, target, "integrated-source target");
  require_same_precision(state.plan_, source, result.precision, "integrated-source base");
  result.target = device_address(state, target, "integrated-source target");
  result.source = optional_device_address(state, source, result.precision,
                                          "integrated-source base");
  result.elements = target_spec.elements;
  return result;
}

void evaluate_supported_source_scalars(fields &f, double offset_in_dt) {
  const size_t time_slot = offset_in_dt == 0.0 ? 0 : offset_in_dt == 0.5 ? 1 : 2;
  for (src_time *source = f.sources; source; source = source->next)
    source->update(f.step_source_times[time_slot], f.dt);
  populate_source_scalars(f, f.descriptors->sources);
}

nvidia::pml_profile_launch compile_pml_profile(const PmlProfile &source,
                                               const nvidia::flat_region &region,
                                               nvidia::scalar_precision precision,
                                               NvidiaBackendState &state, const char *what) {
  nvidia::pml_profile_launch result = {};
  const bool have_sigma = is_valid(source.sig);
  if (!have_sigma) {
    if (is_valid(source.kap) || is_valid(source.siginv))
      throw std::invalid_argument(std::string(what) + " has an incomplete disabled profile");
    return result;
  }
  if (!is_valid(source.kap) || !is_valid(source.siginv))
    throw std::invalid_argument(std::string(what) + " has an incomplete profile");
  if (source.base < 0) throw std::out_of_range(std::string(what) + " has a negative base index");

  result.sigma = optional_device_address(state, source.sig, precision, what);
  result.kappa = optional_device_address(state, source.kap, precision, what);
  result.inverse = optional_device_address(state, source.siginv, precision, what);
  result.base = source.base;
  ptrdiff_t maximum = source.base;
  for (int axis = 0; axis < 3; ++axis) {
    if (source.strides[axis] < 0)
      throw std::invalid_argument(std::string(what) + " has a negative stride");
    result.strides[axis] = source.strides[axis];
    const size_t steps = region.counts[axis] - 1;
    if (steps && size_t(source.strides[axis]) >
                     size_t(std::numeric_limits<ptrdiff_t>::max() - maximum) / steps)
      throw std::overflow_error(std::string(what) + " index range overflow");
    maximum += ptrdiff_t(steps) * source.strides[axis];
  }
  validate_index_range(state.plan_, source.sig, source.base, maximum, what);
  validate_index_range(state.plan_, source.kap, source.base, maximum, what);
  validate_index_range(state.plan_, source.siginv, source.base, maximum, what);
  return result;
}

nvidia::flat_region flat_region_for(const UpdateRegion &source) {
  nvidia::flat_region result;
  result.base = source.base;
  for (int axis = 0; axis < 3; ++axis) {
    if (!source.counts[axis]) throw std::invalid_argument("update descriptor has an empty axis");
    if (source.strides[axis] < 0)
      throw std::invalid_argument("update descriptor has a negative region stride");
    result.counts[axis] = source.counts[axis];
    result.strides[axis] = source.strides[axis];
  }
  return result;
}

ptrdiff_t checked_region_max(const nvidia::flat_region &region) {
  if (region.base > size_t(std::numeric_limits<ptrdiff_t>::max()))
    throw std::overflow_error("update descriptor base exceeds ptrdiff_t");
  ptrdiff_t maximum = ptrdiff_t(region.base);
  for (int axis = 0; axis < 3; ++axis) {
    const size_t steps = region.counts[axis] - 1;
    if (steps && size_t(region.strides[axis]) >
                     size_t(std::numeric_limits<ptrdiff_t>::max() - maximum) / steps)
      throw std::overflow_error("update descriptor index range overflow");
    maximum += ptrdiff_t(steps) * region.strides[axis];
  }
  return maximum;
}

void validate_index_range(const StoragePlan &plan, ArrayId id, ptrdiff_t minimum, ptrdiff_t maximum,
                          const char *what) {
  if (!is_valid(id) || id.value >= plan.arrays.size())
    throw std::invalid_argument(std::string(what) + " uses an invalid ArrayId");
  if (minimum < 0 || maximum < minimum || size_t(maximum) >= plan.arrays[id.value].elements)
    throw std::out_of_range(std::string(what) + " index range exceeds its array");
}

ptrdiff_t checked_shift(ptrdiff_t value, ptrdiff_t offset, const char *what) {
  if ((offset > 0 && value > std::numeric_limits<ptrdiff_t>::max() - offset) ||
      (offset < 0 && value < std::numeric_limits<ptrdiff_t>::min() - offset))
    throw std::overflow_error(std::string(what) + " index range overflow");
  return value + offset;
}

ptrdiff_t checked_negate(ptrdiff_t value, const char *what) {
  if (value == std::numeric_limits<ptrdiff_t>::min())
    throw std::overflow_error(std::string(what) + " index offset overflow");
  return -value;
}

void validate_shifted_index_range(const StoragePlan &plan, ArrayId id, ptrdiff_t region_min,
                                  ptrdiff_t region_max, ptrdiff_t offset0, ptrdiff_t offset1,
                                  ptrdiff_t offset2, ptrdiff_t offset3, const char *what) {
  const ptrdiff_t minimum_offset = std::min(std::min(offset0, offset1), std::min(offset2, offset3));
  const ptrdiff_t maximum_offset = std::max(std::max(offset0, offset1), std::max(offset2, offset3));
  validate_index_range(plan, id, checked_shift(region_min, minimum_offset, what),
                       checked_shift(region_max, maximum_offset, what), what);
}

nvidia::polarization_update_launch compile_lorentzian_update(
    const PolarizationUpdate &source, NvidiaBackendState &state) {
  const uint32_t supported = polarization_one_offdiagonal | polarization_two_offdiagonals |
                             polarization_drude;
  if (source.region.variant_key & ~supported)
    throw std::invalid_argument("polarization descriptor has unknown variant bits");
  const bool have_cross1 = source.region.variant_key & polarization_one_offdiagonal;
  const bool have_cross2 = source.region.variant_key & polarization_two_offdiagonals;
  if (have_cross2 && !have_cross1)
    throw std::invalid_argument("polarization descriptor has a second off-diagonal without first");
  if (have_cross1 != (is_valid(source.cross_w1) && is_valid(source.offdiagonal_sigma1)) ||
      have_cross2 != (is_valid(source.cross_w2) && is_valid(source.offdiagonal_sigma2)))
    throw std::invalid_argument("polarization descriptor anisotropy bits and operands disagree");
  if (!is_valid(source.p) || !is_valid(source.p_prev) || !is_valid(source.primary_w) ||
      !is_valid(source.diagonal_sigma) || source.p == source.p_prev)
    throw std::invalid_argument("polarization descriptor has incomplete state");

  nvidia::polarization_update_launch result = {};
  result.region = flat_region_for(source.region);
  result.precision = scalar_precision_for(state.plan_, source.p, "polarization P");
  require_same_precision(state.plan_, source.p_prev, result.precision, "polarization P_prev");
  require_same_precision(state.plan_, source.primary_w, result.precision, "polarization W");
  require_same_precision(state.plan_, source.diagonal_sigma, result.precision,
                         "polarization diagonal sigma");
  require_same_precision(state.plan_, source.cross_w1, result.precision,
                         "polarization cross W1");
  require_same_precision(state.plan_, source.cross_w2, result.precision,
                         "polarization cross W2");
  require_same_precision(state.plan_, source.offdiagonal_sigma1, result.precision,
                         "polarization off-diagonal sigma1");
  require_same_precision(state.plan_, source.offdiagonal_sigma2, result.precision,
                         "polarization off-diagonal sigma2");
  result.p = device_address(state, source.p, "polarization P");
  result.p_prev = device_address(state, source.p_prev, "polarization P_prev");
  result.primary_w = device_address(state, source.primary_w, "polarization W");
  result.diagonal_sigma =
      device_address(state, source.diagonal_sigma, "polarization diagonal sigma");
  result.cross_w1 = optional_device_address(state, source.cross_w1, result.precision,
                                            "polarization cross W1");
  result.cross_w2 = optional_device_address(state, source.cross_w2, result.precision,
                                            "polarization cross W2");
  result.offdiagonal_sigma1 = optional_device_address(
      state, source.offdiagonal_sigma1, result.precision, "polarization off-diagonal sigma1");
  result.offdiagonal_sigma2 = optional_device_address(
      state, source.offdiagonal_sigma2, result.precision, "polarization off-diagonal sigma2");
  result.primary_stride = source.primary_stride;
  result.cross_stride1 = source.cross_stride1;
  result.cross_stride2 = source.cross_stride2;
  result.drude = (source.region.variant_key & polarization_drude) != 0;
  const nvidia::polarization_coefficients coefficients = nvidia::derive_polarization_coefficients(
      source.omega_0, source.gamma, source.dt, result.drude);
  result.omega0dtsqr = coefficients.omega0dtsqr;
  result.gamma1inv = coefficients.gamma1inv;
  result.gamma1 = coefficients.gamma1;
  result.omega0dtsqr_denom = coefficients.omega0dtsqr_denom;
  result.offdiagonals = have_cross2 ? 2 : have_cross1 ? 1 : 0;

  const ptrdiff_t region_min = ptrdiff_t(result.region.base);
  const ptrdiff_t region_max = checked_region_max(result.region);
  validate_index_range(state.plan_, source.p, region_min, region_max, "polarization P");
  validate_index_range(state.plan_, source.p_prev, region_min, region_max, "polarization P_prev");
  validate_index_range(state.plan_, source.primary_w, region_min, region_max, "polarization W");
  validate_index_range(state.plan_, source.diagonal_sigma, region_min, region_max,
                       "polarization diagonal sigma");
  if (have_cross1) {
    const ptrdiff_t negative_cross =
        checked_negate(source.cross_stride1, "polarization cross1");
    const ptrdiff_t combined =
        checked_shift(source.primary_stride, negative_cross, "polarization cross1");
    validate_shifted_index_range(state.plan_, source.cross_w1, region_min, region_max, 0,
                                 negative_cross, source.primary_stride, combined,
                                 "polarization cross W1");
    validate_shifted_index_range(state.plan_, source.offdiagonal_sigma1, region_min, region_max, 0,
                                 source.primary_stride, 0, source.primary_stride,
                                 "polarization off-diagonal sigma1");
  }
  if (have_cross2) {
    const ptrdiff_t negative_cross =
        checked_negate(source.cross_stride2, "polarization cross2");
    const ptrdiff_t combined =
        checked_shift(source.primary_stride, negative_cross, "polarization cross2");
    validate_shifted_index_range(state.plan_, source.cross_w2, region_min, region_max, 0,
                                 negative_cross, source.primary_stride, combined,
                                 "polarization cross W2");
    validate_shifted_index_range(state.plan_, source.offdiagonal_sigma2, region_min, region_max, 0,
                                 source.primary_stride, 0, source.primary_stride,
                                 "polarization off-diagonal sigma2");
  }
  return result;
}

nvidia::gyrotropic_update_launch compile_gyrotropic_update(
    const PolarizationUpdate &source, NvidiaBackendState &state) {
  if (source.region.variant_key)
    throw std::invalid_argument("gyrotropic descriptor has Lorentzian variant bits");
  if ((!is_electric(source.region.c) && !is_magnetic(source.region.c)) ||
      component_direction(source.region.c) < X || component_direction(source.region.c) > Z)
    throw std::invalid_argument("gyrotropic descriptor requires a Cartesian E/H component");
  if (source.gyro_model != GYROTROPIC_LORENTZIAN && source.gyro_model != GYROTROPIC_DRUDE &&
      source.gyro_model != GYROTROPIC_SATURATED)
    throw std::invalid_argument("gyrotropic descriptor has an invalid model");
  const ArrayId state_ids[6] = {source.p,          source.p_prev,  source.p_cross1,
                                source.p_prev_cross1, source.p_cross2, source.p_prev_cross2};
  for (int i = 0; i < 6; ++i) {
    if (!is_valid(state_ids[i]) || state_ids[i].value >= state.plan_.arrays.size())
      throw std::invalid_argument("gyrotropic descriptor has incomplete state");
    for (int j = i + 1; j < 6; ++j)
      if (state_ids[i] == state_ids[j])
        throw std::invalid_argument("gyrotropic descriptor state arrays alias");
  }
  if (!is_valid(source.primary_w) || !is_valid(source.diagonal_sigma))
    throw std::invalid_argument("gyrotropic descriptor has incomplete driving operands");
  if (is_valid(source.offdiagonal_sigma1) || is_valid(source.offdiagonal_sigma2))
    throw std::invalid_argument("gyrotropic media do not support anisotropic sigma");

  nvidia::gyrotropic_update_launch result = {};
  result.region = flat_region_for(source.region);
  result.precision = scalar_precision_for(state.plan_, source.p, "gyrotropic P0");
  const size_t state_elements = state.plan_.arrays[source.p.value].elements;
  for (int i = 0; i < 6; ++i) {
    const ArraySpec &spec = state.plan_.arrays[state_ids[i].value];
    if (spec.role != array_role::polarization)
      throw std::invalid_argument("gyrotropic state is not polarization storage");
    if (spec.elements != state_elements)
      throw std::invalid_argument("gyrotropic state arrays have incompatible extents");
    require_same_precision(state.plan_, state_ids[i], result.precision,
                           "gyrotropic polarization state");
  }
  require_same_precision(state.plan_, source.primary_w, result.precision, "gyrotropic W0");
  require_same_precision(state.plan_, source.cross_w1, result.precision, "gyrotropic W1");
  require_same_precision(state.plan_, source.cross_w2, result.precision, "gyrotropic W2");
  require_same_precision(state.plan_, source.diagonal_sigma, result.precision,
                         "gyrotropic diagonal sigma");
  result.p[0] = device_address(state, source.p, "gyrotropic P0");
  result.p_prev[0] = device_address(state, source.p_prev, "gyrotropic P0 previous");
  result.p[1] = device_address(state, source.p_cross1, "gyrotropic P1");
  result.p_prev[1] = device_address(state, source.p_prev_cross1, "gyrotropic P1 previous");
  result.p[2] = device_address(state, source.p_cross2, "gyrotropic P2");
  result.p_prev[2] = device_address(state, source.p_prev_cross2, "gyrotropic P2 previous");
  result.w[0] = device_address(state, source.primary_w, "gyrotropic W0");
  result.w[1] = optional_device_address(state, source.cross_w1, result.precision, "gyrotropic W1");
  result.w[2] = optional_device_address(state, source.cross_w2, result.precision, "gyrotropic W2");
  result.sigma = device_address(state, source.diagonal_sigma, "gyrotropic diagonal sigma");
  result.primary_stride = source.primary_stride;
  result.cross_stride1 = source.cross_stride1;
  result.cross_stride2 = source.cross_stride2;

  const direction order[3] = {
      component_direction(source.region.c),
      cycle_direction(source.region.begin.dim, component_direction(source.region.c), 1),
      cycle_direction(source.region.begin.dim, component_direction(source.region.c), 2)};
  const nvidia::gyrotropic_coefficients coefficients = nvidia::derive_gyrotropic_coefficients(
      source.omega_0, source.gamma, source.alpha, source.gyro_tensor, source.dt,
      source.gyro_model, order);
  result.omega0dtsqr = coefficients.omega0dtsqr;
  result.gamma1 = coefficients.gamma1;
  result.diagonal = coefficients.diagonal;
  result.pt = coefficients.pt;
  result.omega = coefficients.omega;
  result.gamma = coefficients.gamma;
  result.alpha = coefficients.alpha;
  result.dt2pi = coefficients.dt2pi;
  memcpy(result.gyro, coefficients.gyro, sizeof(result.gyro));
  memcpy(result.inverse, coefficients.inverse, sizeof(result.inverse));
  result.model = source.gyro_model == GYROTROPIC_LORENTZIAN
                     ? nvidia::gyrotropic_kernel_model::lorentzian
                 : source.gyro_model == GYROTROPIC_DRUDE
                     ? nvidia::gyrotropic_kernel_model::drude
                     : nvidia::gyrotropic_kernel_model::saturated;

  const ptrdiff_t region_min = ptrdiff_t(result.region.base);
  const ptrdiff_t region_max = checked_region_max(result.region);
  for (int i = 0; i < 6; ++i)
    validate_index_range(state.plan_, state_ids[i], region_min, region_max,
                         "gyrotropic polarization state");
  validate_index_range(state.plan_, source.primary_w, region_min, region_max, "gyrotropic W0");
  validate_index_range(state.plan_, source.diagonal_sigma, region_min, region_max,
                       "gyrotropic diagonal sigma");
  if (is_valid(source.cross_w1)) {
    const ptrdiff_t negative_cross = checked_negate(source.cross_stride1, "gyrotropic W1");
    const ptrdiff_t combined =
        checked_shift(source.primary_stride, negative_cross, "gyrotropic W1");
    validate_shifted_index_range(state.plan_, source.cross_w1, region_min, region_max, 0,
                                 negative_cross, source.primary_stride, combined,
                                 "gyrotropic W1");
  }
  if (is_valid(source.cross_w2)) {
    const ptrdiff_t negative_cross = checked_negate(source.cross_stride2, "gyrotropic W2");
    const ptrdiff_t combined =
        checked_shift(source.primary_stride, negative_cross, "gyrotropic W2");
    validate_shifted_index_range(state.plan_, source.cross_w2, region_min, region_max, 0,
                                 negative_cross, source.primary_stride, combined,
                                 "gyrotropic W2");
  }
  return result;
}

nvidia::compiled_polarization_update compile_polarization_update(
    const PolarizationUpdate &source, NvidiaBackendState &state) {
  nvidia::compiled_polarization_update result = {};
  switch (source.kind) {
    case PolarizationUpdateKind::lorentzian:
      result.kind = nvidia::compiled_polarization_update::kind_type::lorentzian;
      result.lorentzian = compile_lorentzian_update(source, state);
      break;
    case PolarizationUpdateKind::gyrotropic:
      result.kind = nvidia::compiled_polarization_update::kind_type::gyrotropic;
      result.gyrotropic = compile_gyrotropic_update(source, state);
      break;
    default: throw std::invalid_argument("polarization descriptor has an invalid update kind");
  }
  return result;
}

nvidia::polarization_subtract_launch compile_polarization_subtraction(
    const PolarizationSubtraction &source, NvidiaBackendState &state) {
  if (!is_valid(source.target) || !is_valid(source.p) || source.target == source.p ||
      !source.elements)
    throw std::invalid_argument("polarization subtraction has invalid operands");
  if (source.target.value >= state.plan_.arrays.size() || source.p.value >= state.plan_.arrays.size())
    throw std::out_of_range("polarization subtraction ArrayId is out of range");
  const ArraySpec &target_spec = state.plan_.arrays[source.target.value];
  const ArraySpec &p_spec = state.plan_.arrays[source.p.value];
  if (source.elements != target_spec.elements || source.elements != p_spec.elements)
    throw std::out_of_range("polarization subtraction is not a full-array operation");
  nvidia::polarization_subtract_launch result = {};
  result.precision = scalar_precision_for(state.plan_, source.target,
                                          "polarization subtraction target");
  require_same_precision(state.plan_, source.p, result.precision, "polarization subtraction P");
  result.target = device_address(state, source.target, "polarization subtraction target");
  result.p = device_address(state, source.p, "polarization subtraction P");
  result.elements = source.elements;
  return result;
}

nvidia::dft_launch compile_dft(const DftDescriptor &source, const fields &f,
                               NvidiaBackendState &state, std::vector<double> &packed_omega) {
  if (source.chunk < 0 || source.chunk >= f.num_chunks || !f.chunks[source.chunk] ||
      !f.chunks[source.chunk]->is_mine())
    throw std::invalid_argument("DFT descriptor has an invalid local chunk");
  if (!source.N || !source.Nomega || source.omega.size() != source.Nomega)
    throw std::invalid_argument("DFT descriptor has invalid dimensions");
  if (source.decimation_factor <= 0)
    throw std::invalid_argument("DFT descriptor has a nonpositive decimation factor");

  const ArrayId accumulator = source.accumulator;
  const ArrayId phase = source.phase_scratch;
  if (!is_valid(accumulator) || accumulator.value >= state.plan_.arrays.size() ||
      !is_valid(phase) || phase.value >= state.plan_.arrays.size())
    throw std::invalid_argument("DFT descriptor has invalid monitor storage");
  const ArraySpec &accumulator_spec = state.plan_.arrays[accumulator.value];
  const ArraySpec &phase_spec = state.plan_.arrays[phase.value];
  const StorageKey &accumulator_key = state.plan_.keys[accumulator.value];
  const StorageKey &phase_key = state.plan_.keys[phase.value];
  if (accumulator_spec.role != array_role::dft || phase_spec.role != array_role::dft ||
      accumulator_key.kind != int(array_kind::dft) ||
      phase_key.kind != int(array_kind::dft_phase) || accumulator_key.chunk != source.chunk ||
      phase_key.chunk != source.chunk || accumulator_key.component_ != int(source.c) ||
      phase_key.component_ != int(source.c) || accumulator_key.aux != phase_key.aux ||
      is_valid(accumulator_spec.alias_of) || is_valid(phase_spec.alias_of))
    throw std::invalid_argument("DFT descriptor storage has the wrong role");
  const nvidia::scalar_precision monitor_precision =
      complex_precision_for(state.plan_, accumulator, "DFT accumulator");
  if (complex_precision_for(state.plan_, phase, "DFT phase scratch") != monitor_precision)
    throw std::invalid_argument("DFT accumulator and phase scratch precisions differ");

  if (!is_valid(source.source_field.id) ||
      source.source_field.id.value >= state.plan_.arrays.size())
    throw std::invalid_argument("DFT descriptor has no real source field");
  const ArraySpec &real_spec = state.plan_.arrays[source.source_field.id.value];
  const StorageKey &real_key = state.plan_.keys[source.source_field.id.value];
  if (real_spec.role != array_role::field || real_spec.element_type != ElementType::realnum_value)
    throw std::invalid_argument("DFT real source has the wrong storage type");
  if (real_key.kind != int(array_kind::f) || real_key.chunk != source.chunk ||
      real_key.component_ != int(source.c) || real_key.cmp != 0)
    throw std::invalid_argument("DFT real source has the wrong storage identity");
  const nvidia::scalar_precision field_precision =
      scalar_precision_for(state.plan_, source.source_field.id, "DFT real source");
  if (f.is_real && is_valid(source.source_field_imag.id))
    throw std::invalid_argument("real NVIDIA fields have an imaginary DFT source");
  if (!f.is_real && !is_valid(source.source_field_imag.id))
    throw std::invalid_argument("complex NVIDIA fields have no imaginary DFT source");
  if (is_valid(source.source_field_imag.id)) {
    if (source.source_field_imag.id.value >= state.plan_.arrays.size())
      throw std::out_of_range("DFT imaginary source ArrayId is out of range");
    const ArraySpec &imag_spec = state.plan_.arrays[source.source_field_imag.id.value];
    const StorageKey &imag_key = state.plan_.keys[source.source_field_imag.id.value];
    if (imag_spec.role != array_role::field ||
        imag_spec.element_type != ElementType::realnum_value)
      throw std::invalid_argument("DFT imaginary source has the wrong storage type");
    if (imag_key.kind != int(array_kind::f) || imag_key.chunk != source.chunk ||
        imag_key.component_ != int(source.c) || imag_key.cmp != 1)
      throw std::invalid_argument("DFT imaginary source has the wrong storage identity");
    require_same_precision(state.plan_, source.source_field_imag.id, field_precision,
                           "DFT imaginary source");
  }
  if (field_precision == nvidia::scalar_precision::f64 &&
      monitor_precision == nvidia::scalar_precision::f32)
    throw std::invalid_argument("NVIDIA DFT does not support f64 fields with f32 monitors");

  nvidia::dft_launch result = {};
  result.field_precision = field_precision;
  result.monitor_precision = monitor_precision;
  result.source_real = device_address(state, source.source_field.id, "DFT real source");
  result.source_imag = is_valid(source.source_field_imag.id)
                           ? device_address(state, source.source_field_imag.id,
                                            "DFT imaginary source")
                           : NULL;
  result.accumulator = complex_device_address(state, accumulator, "DFT accumulator");
  result.phase_scratch = complex_device_address(state, phase, "DFT phase scratch");
  result.points = source.N;
  result.frequencies = source.Nomega;
  result.avg1 = source.avg1;
  result.avg2 = source.avg2;
  result.dV0 = source.dV0;
  result.dV1 = source.dV1;
  result.scale_real = source.scale.real();
  result.scale_imag = source.scale.imag();
  result.decimation_factor = source.decimation_factor;
  result.include_weights = source.include_dV_and_interp_weights;
  result.sqrt_weights = source.sqrt_dV_and_interp_weights;
  result.magnetic = is_H_or_B(source.c);

  const grid_volume &gv = f.chunks[source.chunk]->gv;
  ptrdiff_t base = 0;
  size_t points = 1;
  for (int axis = 0; axis < 3; ++axis) {
    const ptrdiff_t begin = source.is.yucky_val(axis);
    const ptrdiff_t end = source.ie.yucky_val(axis);
    const ptrdiff_t extent = end - begin;
    if (extent < 0) {
      std::ostringstream message;
      message << "DFT descriptor has a reversed extent (chunk=" << source.chunk
              << ", component=" << int(source.c) << ", axis=" << axis << ", begin=" << begin
              << ", end=" << end << ")";
      throw std::invalid_argument(message.str());
    }
    const size_t count = size_t(extent / 2) + 1;
    points = checked_product(points, count, "sizing NVIDIA DFT region");
    const direction d = gv.yucky_direction(axis);
    const ptrdiff_t stride = gv.stride(d);
    if (stride < 0) throw std::invalid_argument("DFT descriptor has a negative field stride");
    const ptrdiff_t relative = begin - gv.little_corner().yucky_val(axis);
    /* Match PLOOP_OVER_IVECS exactly: component Yee shifts can make this
       difference odd (including -1), and C++ integer division truncates it
       toward zero. */
    const ptrdiff_t coordinate = relative / 2;
    if (coordinate > 0 && stride > std::numeric_limits<ptrdiff_t>::max() / coordinate)
      throw std::overflow_error("DFT descriptor base index overflow");
    if (coordinate < 0) {
      if (coordinate == std::numeric_limits<ptrdiff_t>::min() ||
          stride > std::numeric_limits<ptrdiff_t>::max() / -coordinate)
        throw std::overflow_error("DFT descriptor base index overflow");
    }
    base = checked_shift(base, coordinate * stride, "building DFT descriptor base");
    result.region.counts[axis] = count;
    result.region.strides[axis] = stride;
    result.start0[axis] = source.weights.s0.in_direction(d);
    result.start1[axis] = source.weights.s1.in_direction(d);
    result.end0[axis] = source.weights.e0.in_direction(d);
    result.end1[axis] = source.weights.e1.in_direction(d);
  }
  if (points != source.N)
    throw std::invalid_argument("DFT descriptor region size does not match N");
  if (base < 0) throw std::out_of_range("DFT descriptor has a negative source base");
  result.region.base = size_t(base);

  const ptrdiff_t region_max = checked_region_max(result.region);
  const ptrdiff_t offsets[4] = {0, source.avg1, source.avg2,
                                checked_shift(source.avg1, source.avg2,
                                              "validating DFT interpolation")};
  const ptrdiff_t minimum_offset = *std::min_element(offsets, offsets + 4);
  const ptrdiff_t maximum_offset = *std::max_element(offsets, offsets + 4);
  validate_ref_index_range(state.plan_, source.source_field,
                           checked_shift(base, minimum_offset, "validating DFT real source"),
                           checked_shift(region_max, maximum_offset,
                                         "validating DFT real source"),
                           "DFT real source");
  if (is_valid(source.source_field_imag.id))
    validate_ref_index_range(state.plan_, source.source_field_imag,
                             checked_shift(base, minimum_offset,
                                           "validating DFT imaginary source"),
                             checked_shift(region_max, maximum_offset,
                                           "validating DFT imaginary source"),
                             "DFT imaginary source");

  const size_t outputs = checked_product(source.N, source.Nomega,
                                         "sizing NVIDIA DFT accumulator");
  if (accumulator_spec.elements < outputs || phase_spec.elements < source.Nomega)
    throw std::out_of_range("DFT monitor storage is smaller than its descriptor");
  const size_t monitor_bytes = monitor_precision == nvidia::scalar_precision::f32
                                   ? sizeof(float)
                                   : sizeof(double);
  (void)checked_product(checked_product(outputs, size_t(2),
                                        "sizing NVIDIA DFT scalar output"),
                        monitor_bytes, "sizing NVIDIA DFT output bytes");
  (void)checked_product(checked_product(source.Nomega, size_t(2),
                                        "sizing NVIDIA DFT phase scalars"),
                        monitor_bytes, "sizing NVIDIA DFT phase bytes");

  result.omega_offset = packed_omega.size();
  if (source.omega.size() > std::numeric_limits<size_t>::max() - packed_omega.size())
    throw std::overflow_error("NVIDIA DFT frequency table size overflow");
  packed_omega.insert(packed_omega.end(), source.omega.begin(), source.omega.end());
  return result;
}

nvidia::curl_launch compile_curl(const CurlUpdate &source, NvidiaBackendState &state) {
  const uint32_t supported_variants =
      curl_has_second_derivative | curl_has_pml | curl_has_pml_aux | curl_has_conductivity |
      curl_has_bfast;
  if (source.region.variant_key & ~supported_variants)
    throw std::invalid_argument("curl descriptor has an unsupported variant bit");

  const bool have_pml = (source.region.variant_key & curl_has_pml) != 0;
  const bool have_pml_u = (source.region.variant_key & curl_has_pml_aux) != 0;
  const bool have_conductivity = (source.region.variant_key & curl_has_conductivity) != 0;
  if (have_pml != is_valid(source.pml.sig) || have_pml_u != is_valid(source.pml_u.sig) ||
      have_pml_u != is_valid(source.target_u) ||
      have_conductivity != is_valid(source.conductivity) ||
      have_conductivity != is_valid(source.condinv) ||
      (have_pml && have_conductivity) != is_valid(source.target_cond))
    throw std::invalid_argument("curl descriptor variant bits and auxiliary arrays disagree");

  nvidia::curl_launch result = {};
  result.region = flat_region_for(source.region);
  result.precision = scalar_precision_for(state.plan_, source.target, "curl target");
  result.target = device_address(state, source.target, "curl target");
  result.plus_source =
      optional_device_address(state, source.plus_source, result.precision, "curl plus source");
  result.minus_source =
      optional_device_address(state, source.minus_source, result.precision, "curl minus source");
  if (!result.plus_source && !result.minus_source)
    throw std::invalid_argument("curl descriptor has no source field");
  result.plus_stride = source.plus_stride;
  result.minus_stride = source.minus_stride;
  result.target_u = optional_mutable_device_address(state, source.target_u, result.precision,
                                                    "curl auxiliary target");
  result.conductivity =
      optional_device_address(state, source.conductivity, result.precision, "curl conductivity");
  result.conductivity_inverse =
      optional_device_address(state, source.condinv, result.precision, "curl conductivity inverse");
  result.target_conductivity = optional_mutable_device_address(
      state, source.target_cond, result.precision, "curl conductivity target");
  result.pml =
      compile_pml_profile(source.pml, result.region, result.precision, state, "curl main PML");
  result.pml_u = compile_pml_profile(source.pml_u, result.region, result.precision, state,
                                     "curl auxiliary PML");
  result.dtdx = source.dtdx;
  result.dt = source.dt;
  result.radial_prefix_index = UINT32_MAX;
  result.bfast_update_index = UINT32_MAX;

  const ptrdiff_t region_max = checked_region_max(result.region);
  validate_index_range(state.plan_, source.target, ptrdiff_t(result.region.base), region_max,
                       "curl target");
  if (is_valid(source.plus_source))
    validate_index_range(state.plan_, source.plus_source,
                         ptrdiff_t(result.region.base) + std::min<ptrdiff_t>(0, source.plus_stride),
                         region_max + std::max<ptrdiff_t>(0, source.plus_stride),
                         "curl plus source");
  if (is_valid(source.minus_source))
    validate_index_range(
        state.plan_, source.minus_source,
        ptrdiff_t(result.region.base) + std::min<ptrdiff_t>(0, source.minus_stride),
        region_max + std::max<ptrdiff_t>(0, source.minus_stride), "curl minus source");
  if (is_valid(source.target_u))
    validate_index_range(state.plan_, source.target_u, ptrdiff_t(result.region.base), region_max,
                         "curl auxiliary target");
  if (is_valid(source.conductivity))
    validate_index_range(state.plan_, source.conductivity, ptrdiff_t(result.region.base),
                         region_max, "curl conductivity");
  if (is_valid(source.condinv))
    validate_index_range(state.plan_, source.condinv, ptrdiff_t(result.region.base), region_max,
                         "curl conductivity inverse");
  if (is_valid(source.target_cond))
    validate_index_range(state.plan_, source.target_cond, ptrdiff_t(result.region.base), region_max,
                         "curl conductivity target");
  return result;
}

nvidia::cylindrical_radial_prefix_launch
compile_cylindrical_radial_prefix(const CylindricalRadialPrefix &source,
                                  NvidiaBackendState &state, const fields &f) {
  if (source.chunk < 0 || source.chunk >= f.num_chunks || !f.chunks[source.chunk] ||
      !f.chunks[source.chunk]->is_mine())
    throw std::invalid_argument("cylindrical radial prefix references a nonlocal chunk");
  const fields_chunk &fc = *f.chunks[source.chunk];
  const bool valid_pair =
      (source.target_component == Dz && source.source_component == Hp) ||
      (source.target_component == Bz && source.source_component == Ep);
  if (fc.gv.dim != Dcyl || !valid_pair || source.cmp < 0 || source.cmp > 1)
    throw std::invalid_argument("cylindrical radial prefix has invalid component identity");
  const StorageKey expected_source{source.chunk, int(array_kind::f), int(source.source_component),
                                   source.cmp, 0};
  const StorageKey expected_scratch{source.chunk, int(array_kind::f_rderiv_int), -1, -1, 0};
  if (!is_valid(source.source) || source.source.value >= state.plan_.keys.size() ||
      !(state.plan_.keys[source.source.value] == expected_source) ||
      !is_valid(source.scratch) || source.scratch.value >= state.plan_.keys.size() ||
      !(state.plan_.keys[source.scratch.value] == expected_scratch) ||
      is_valid(state.plan_.arrays[source.scratch.value].alias_of) || source.source == source.scratch)
    throw std::invalid_argument("cylindrical radial-prefix storage identity is invalid");
  if (source.nr != size_t(fc.gv.nr()) || source.nz != size_t(fc.gv.nz()) ||
      source.row_stride != source.nz + 1)
    throw std::invalid_argument("cylindrical radial-prefix shape does not match its chunk");
  if (source.nr == std::numeric_limits<size_t>::max())
    throw std::overflow_error("cylindrical radial-prefix radial extent overflow");
  const size_t required = checked_product(source.nr + 1, source.row_stride,
                                          "sizing cylindrical radial-prefix storage");
  if (source.source_elements != state.plan_.arrays[source.source.value].elements ||
      source.scratch_elements != state.plan_.arrays[source.scratch.value].elements ||
      source.source_elements < required || source.scratch_elements < required)
    throw std::out_of_range("cylindrical radial-prefix storage extent is invalid");
  const realnum expected_ir0 =
      fc.gv.origin_r() * fc.gv.a +
      0.5 * fc.gv.iyee_shift(source.source_component).in_direction(R);
  if (!std::isfinite(source.ir0) || source.ir0 != double(expected_ir0))
    throw std::invalid_argument("cylindrical radial-prefix coefficient is stale");

  nvidia::cylindrical_radial_prefix_launch result = {};
  result.precision = scalar_precision_for(state.plan_, source.scratch,
                                          "cylindrical radial-prefix scratch");
  require_same_precision(state.plan_, source.source, result.precision,
                         "cylindrical radial-prefix source");
  result.source = device_address(state, source.source, "cylindrical radial-prefix source");
  result.scratch = device_address(state, source.scratch, "cylindrical radial-prefix scratch");
  result.nr = source.nr;
  result.nz = source.nz;
  result.row_stride = source.row_stride;
  result.source_elements = source.source_elements;
  result.scratch_elements = source.scratch_elements;
  result.ir0 = source.ir0;
  return result;
}

nvidia::cylindrical_m_launch compile_cylindrical_m(const CylindricalMOverRUpdate &source,
                                                   NvidiaBackendState &state, const fields &f) {
  const uint32_t supported = cylindrical_m_has_pml | cylindrical_m_has_pml_aux |
                             cylindrical_m_has_conductivity;
  if (source.region.variant_key & ~supported)
    throw std::invalid_argument("cylindrical m/r descriptor has an unsupported variant bit");
  if (source.region.chunk < 0 || source.region.chunk >= f.num_chunks ||
      !f.chunks[source.region.chunk] || !f.chunks[source.region.chunk]->is_mine())
    throw std::invalid_argument("cylindrical m/r descriptor references a nonlocal chunk");
  const fields_chunk &fc = *f.chunks[source.region.chunk];
  const direction dc = component_direction(source.region.c);
  if (fc.gv.dim != Dcyl || f.m == 0 || (dc != R && dc != Z) ||
      !(is_D(source.region.c) || is_B(source.region.c)) || source.region.cmp < 0 ||
      source.region.cmp > 1)
    throw std::invalid_argument("cylindrical m/r descriptor has invalid target identity");
  const bool have_pml = source.region.variant_key & cylindrical_m_has_pml;
  const bool have_pml_u = source.region.variant_key & cylindrical_m_has_pml_aux;
  const bool have_conductivity = source.region.variant_key & cylindrical_m_has_conductivity;
  const bool complete_pml = is_valid(source.pml.sig) && is_valid(source.pml.kap) &&
                            is_valid(source.pml.siginv);
  const bool complete_pml_u = is_valid(source.pml_u.sig) && is_valid(source.pml_u.kap) &&
                              is_valid(source.pml_u.siginv);
  if (have_pml != complete_pml || have_pml_u != complete_pml_u ||
      have_pml_u != is_valid(source.target_u) ||
      have_conductivity != is_valid(source.condinv) ||
      (have_pml && have_conductivity) != is_valid(source.target_cond))
    throw std::invalid_argument("cylindrical m/r variant bits and auxiliary arrays disagree");
  const component expected_source_component =
      is_D(source.region.c) ? (dc == R ? Hz : Hr) : (dc == R ? Ez : Er);
  const StorageKey expected_target{source.region.chunk, int(array_kind::f), int(source.region.c),
                                   source.region.cmp, 0};
  const StorageKey expected_source{source.region.chunk, int(array_kind::f),
                                   int(expected_source_component), 1 - source.region.cmp, 0};
  if (!is_valid(source.target) || source.target.value >= state.plan_.keys.size() ||
      !(state.plan_.keys[source.target.value] == expected_target) || !is_valid(source.source) ||
      source.source.value >= state.plan_.keys.size() ||
      !(state.plan_.keys[source.source.value] == expected_source))
    throw std::invalid_argument("cylindrical m/r target or source identity is invalid");
  const auto has_key = [&](ArrayId id, const StorageKey &key) {
    return !is_valid(id) || (id.value < state.plan_.keys.size() && state.plan_.keys[id.value] == key);
  };
  if (!has_key(source.target_u,
               StorageKey{source.region.chunk, int(array_kind::f_u), int(source.region.c),
                          source.region.cmp, 0}) ||
      !has_key(source.condinv,
               StorageKey{source.region.chunk, int(array_kind::condinv), int(source.region.c), -1,
                          int(dc)}) ||
      !has_key(source.target_cond,
               StorageKey{source.region.chunk, int(array_kind::f_cond), int(source.region.c),
                          source.region.cmp, 0}))
    throw std::invalid_argument("cylindrical m/r auxiliary storage identity is invalid");
  const realnum expected_numerator =
      2 * fc.m * (1 - 2 * source.region.cmp) * (1 - 2 * is_B(source.region.c)) *
      (1 - 2 * (dc == R)) * fc.Courant;
  if (!std::isfinite(source.numerator) || source.numerator != double(expected_numerator))
    throw std::invalid_argument("cylindrical m/r coefficient is stale");
  if (source.raw_radial_start != source.region.begin.in_direction(R))
    throw std::invalid_argument("cylindrical m/r raw radial coordinate is stale");
  const ArrayId mutable_arrays[] = {source.target, source.target_u, source.target_cond};
  const ArrayId input_arrays[] = {source.source, source.condinv, source.pml.sig,
                                  source.pml.kap, source.pml.siginv, source.pml_u.sig,
                                  source.pml_u.kap, source.pml_u.siginv};
  for (size_t i = 0; i < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++i) {
    if (!is_valid(mutable_arrays[i])) continue;
    for (ArrayId input : input_arrays)
      if (is_valid(input) && mutable_arrays[i] == input)
        throw std::invalid_argument("cylindrical m/r aliases mutable and input state");
    for (size_t j = i + 1; j < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++j)
      if (is_valid(mutable_arrays[j]) && mutable_arrays[i] == mutable_arrays[j])
        throw std::invalid_argument("cylindrical m/r aliases mutable state");
  }

  nvidia::cylindrical_m_launch result = {};
  result.region = flat_region_for(source.region);
  result.precision = scalar_precision_for(state.plan_, source.target, "cylindrical m/r target");
  result.target = device_address(state, source.target, "cylindrical m/r target");
  result.source = optional_device_address(state, source.source, result.precision,
                                          "cylindrical m/r source");
  result.target_u = optional_mutable_device_address(state, source.target_u, result.precision,
                                                     "cylindrical m/r auxiliary target");
  result.conductivity_inverse = optional_device_address(
      state, source.condinv, result.precision, "cylindrical m/r conductivity inverse");
  result.target_conductivity = optional_mutable_device_address(
      state, source.target_cond, result.precision, "cylindrical m/r conductivity target");
  result.pml = compile_pml_profile(source.pml, result.region, result.precision, state,
                                   "cylindrical m/r main PML");
  result.pml_u = compile_pml_profile(source.pml_u, result.region, result.precision, state,
                                     "cylindrical m/r auxiliary PML");
  result.numerator = source.numerator;
  result.raw_radial_start = source.raw_radial_start;
  result.radial_axis = 3;
  for (unsigned int axis = 0; axis < 3; ++axis)
    if (fc.gv.yucky_direction(axis) == R) result.radial_axis = axis;
  if (result.radial_axis >= 3)
    throw std::invalid_argument("cylindrical m/r region has no radial axis");
  const size_t radial_count = result.region.counts[result.radial_axis];
  if (!radial_count)
    throw std::invalid_argument("cylindrical m/r region has no radial points");
  if (radial_count - 1 > size_t(std::numeric_limits<ptrdiff_t>::max() / 2))
    throw std::overflow_error("cylindrical m/r radial coordinate overflow");
  const ptrdiff_t last_raw = checked_shift(
      source.raw_radial_start, ptrdiff_t(2 * (radial_count - 1)),
      "validating cylindrical m/r radial coordinate");
  if ((source.raw_radial_start <= 0 && last_raw >= 0) ||
      (source.raw_radial_start >= 0 && last_raw <= 0))
    throw std::invalid_argument("cylindrical m/r denominator reaches zero");

  const ptrdiff_t region_max = checked_region_max(result.region);
  validate_index_range(state.plan_, source.target, ptrdiff_t(result.region.base), region_max,
                       "cylindrical m/r target");
  validate_index_range(state.plan_, source.source, ptrdiff_t(result.region.base), region_max,
                       "cylindrical m/r source");
  if (is_valid(source.target_u))
    validate_index_range(state.plan_, source.target_u, ptrdiff_t(result.region.base), region_max,
                         "cylindrical m/r auxiliary target");
  if (is_valid(source.condinv))
    validate_index_range(state.plan_, source.condinv, ptrdiff_t(result.region.base), region_max,
                         "cylindrical m/r conductivity inverse");
  if (is_valid(source.target_cond))
    validate_index_range(state.plan_, source.target_cond, ptrdiff_t(result.region.base), region_max,
                         "cylindrical m/r conductivity target");
  return result;
}

nvidia::cylindrical_axis_launch compile_cylindrical_axis(const CylindricalAxisUpdate &source,
                                                         NvidiaBackendState &state,
                                                         const fields &f) {
  const uint32_t supported = cylindrical_axis_has_pml | cylindrical_axis_has_pml_aux |
                             cylindrical_axis_has_conductivity;
  if (source.region.variant_key & ~supported)
    throw std::invalid_argument("cylindrical axis descriptor has an unsupported variant bit");
  if (source.region.chunk < 0 || source.region.chunk >= f.num_chunks ||
      !f.chunks[source.region.chunk] || !f.chunks[source.region.chunk]->is_mine())
    throw std::invalid_argument("cylindrical axis descriptor references a nonlocal chunk");
  const fields_chunk &fc = *f.chunks[source.region.chunk];
  if (fc.gv.dim != Dcyl || fc.gv.origin_r() != 0.0 || source.region.cmp < 0 ||
      source.region.cmp > 1 ||
      source.region.begin.in_direction(R) != 0 || source.region.end.in_direction(R) != 0)
    throw std::invalid_argument("cylindrical axis descriptor has invalid geometry");
  const bool have_pml = source.region.variant_key & cylindrical_axis_has_pml;
  const bool have_pml_u = source.region.variant_key & cylindrical_axis_has_pml_aux;
  const bool have_conductivity = source.region.variant_key & cylindrical_axis_has_conductivity;
  const bool complete_pml = is_valid(source.pml.sig) && is_valid(source.pml.kap) &&
                            is_valid(source.pml.siginv);
  const bool complete_pml_u = is_valid(source.pml_u.sig) && is_valid(source.pml_u.kap) &&
                              is_valid(source.pml_u.siginv);
  if (have_pml != complete_pml || have_pml_u != complete_pml_u ||
      have_pml_u != is_valid(source.target_u) ||
      have_conductivity != is_valid(source.conductivity) ||
      have_conductivity != is_valid(source.condinv) ||
      have_conductivity != is_valid(source.target_cond))
    throw std::invalid_argument("cylindrical axis variant bits and auxiliary arrays disagree");

  component expected_target = NO_COMPONENT, expected_source1 = NO_COMPONENT,
            expected_source2 = NO_COMPONENT;
  int expected_source2_cmp = source.region.cmp;
  ptrdiff_t expected_neighbor = 0, expected_source2_offset = 0;
  realnum expected_scale = 0, expected_multiplier = 0;
  if (source.kind == CylindricalAxisKind::m0_dz) {
    if (fc.m != 0) throw std::invalid_argument("m=0 cylindrical axis row has stale live m");
    expected_target = Dz;
    expected_source1 = Hp;
    expected_scale = fc.Courant * 4;
  }
  else if (source.kind == CylindricalAxisKind::abs_m1) {
    if (fabs(fc.m) != 1)
      throw std::invalid_argument("|m|=1 cylindrical axis row has stale live m");
    const bool electric = is_D(source.region.c);
    expected_target = electric ? Dp : Br;
    expected_source1 = electric ? Hr : Ep;
    expected_source2 = electric ? Hz : Ez;
    expected_source2_cmp = electric ? source.region.cmp : 1 - source.region.cmp;
    expected_neighbor = electric ? -1 : +1;
    expected_source2_offset = electric ? 0 : fc.gv.nz() + 1;
    expected_scale = (electric ? +1 : -1) * fc.Courant;
    expected_multiplier = electric ? 2 : (1 - 2 * source.region.cmp) * fc.m;
  }
  else
    throw std::invalid_argument("cylindrical axis descriptor kind is invalid");
  const StorageKey expected_target_key{source.region.chunk, int(array_kind::f),
                                       int(expected_target), source.region.cmp, 0};
  const StorageKey expected_source1_key{source.region.chunk, int(array_kind::f),
                                        int(expected_source1), source.region.cmp, 0};
  if (!is_valid(source.target) || source.target.value >= state.plan_.keys.size() ||
      !(state.plan_.keys[source.target.value] == expected_target_key) ||
      !is_valid(source.source1) || source.source1.value >= state.plan_.keys.size() ||
      !(state.plan_.keys[source.source1.value] == expected_source1_key))
    throw std::invalid_argument("cylindrical axis target or first-source identity is invalid");
  const auto has_key = [&](ArrayId id, const StorageKey &key) {
    return !is_valid(id) || (id.value < state.plan_.keys.size() && state.plan_.keys[id.value] == key);
  };
  const direction target_direction = component_direction(expected_target);
  if (!has_key(source.target_u,
               StorageKey{source.region.chunk, int(array_kind::f_u), int(expected_target),
                          source.region.cmp, 0}) ||
      !has_key(source.conductivity,
               StorageKey{source.region.chunk, int(array_kind::conductivity), int(expected_target),
                          -1, int(target_direction)}) ||
      !has_key(source.condinv,
               StorageKey{source.region.chunk, int(array_kind::condinv), int(expected_target), -1,
                          int(target_direction)}) ||
      !has_key(source.target_cond,
               StorageKey{source.region.chunk, int(array_kind::f_cond), int(expected_target),
                          source.region.cmp, 0}))
    throw std::invalid_argument("cylindrical axis auxiliary storage identity is invalid");
  if (expected_source2 == NO_COMPONENT) {
    if (is_valid(source.source2))
      throw std::invalid_argument("m=0 cylindrical axis row has an unexpected second source");
  }
  else {
    const StorageKey expected_source2_key{source.region.chunk, int(array_kind::f),
                                          int(expected_source2), expected_source2_cmp, 0};
    if (!is_valid(source.source2) || source.source2.value >= state.plan_.keys.size() ||
        !(state.plan_.keys[source.source2.value] == expected_source2_key))
      throw std::invalid_argument("cylindrical axis second-source identity is invalid");
  }
  if (source.region.c != expected_target ||
      source.source1_neighbor_offset != expected_neighbor ||
      source.source2_offset != expected_source2_offset ||
      source.scale != double(expected_scale) ||
      source.source2_multiplier != double(expected_multiplier) || source.dt != double(fc.dt) ||
      !std::isfinite(source.scale) || !std::isfinite(source.source2_multiplier) ||
      !std::isfinite(source.dt))
    throw std::invalid_argument("cylindrical axis coefficient or offset is stale");

  const ArrayId mutable_arrays[] = {source.target, source.target_u, source.target_cond};
  const ArrayId input_arrays[] = {source.source1, source.source2, source.conductivity,
                                  source.condinv, source.pml.sig, source.pml.kap,
                                  source.pml.siginv, source.pml_u.sig, source.pml_u.kap,
                                  source.pml_u.siginv};
  for (ArrayId output : mutable_arrays) {
    if (!is_valid(output)) continue;
    for (ArrayId input : input_arrays)
      if (is_valid(input) && output == input)
        throw std::invalid_argument("cylindrical axis aliases mutable and input state");
  }
  for (size_t i = 0; i < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++i)
    for (size_t j = i + 1; j < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++j)
      if (is_valid(mutable_arrays[i]) && mutable_arrays[i] == mutable_arrays[j])
        throw std::invalid_argument("cylindrical axis aliases mutable state");

  nvidia::cylindrical_axis_launch result = {};
  result.region = flat_region_for(source.region);
  result.precision = scalar_precision_for(state.plan_, source.target, "cylindrical axis target");
  result.target = device_address(state, source.target, "cylindrical axis target");
  result.source1 = optional_device_address(state, source.source1, result.precision,
                                           "cylindrical axis first source");
  result.source2 = optional_device_address(state, source.source2, result.precision,
                                           "cylindrical axis second source");
  result.source1_neighbor_offset = source.source1_neighbor_offset;
  result.source2_offset = source.source2_offset;
  result.target_u = optional_mutable_device_address(state, source.target_u, result.precision,
                                                     "cylindrical axis auxiliary target");
  result.conductivity = optional_device_address(state, source.conductivity, result.precision,
                                                "cylindrical axis conductivity");
  result.conductivity_inverse = optional_device_address(
      state, source.condinv, result.precision, "cylindrical axis conductivity inverse");
  result.target_conductivity = optional_mutable_device_address(
      state, source.target_cond, result.precision, "cylindrical axis conductivity target");
  result.pml = compile_pml_profile(source.pml, result.region, result.precision, state,
                                   "cylindrical axis main PML");
  result.pml_u = compile_pml_profile(source.pml_u, result.region, result.precision, state,
                                     "cylindrical axis auxiliary PML");
  result.scale = source.scale;
  result.source2_multiplier = source.source2_multiplier;
  result.dt = source.dt;
  result.kind = static_cast<uint32_t>(source.kind);

  const ptrdiff_t region_min = ptrdiff_t(result.region.base);
  const ptrdiff_t region_max = checked_region_max(result.region);
  validate_index_range(state.plan_, source.target, region_min, region_max,
                       "cylindrical axis target");
  validate_shifted_index_range(state.plan_, source.source1, region_min, region_max, 0,
                               source.source1_neighbor_offset, 0, 0,
                               "cylindrical axis first source");
  if (is_valid(source.source2))
    validate_shifted_index_range(state.plan_, source.source2, region_min, region_max, 0,
                                 source.source2_offset, 0, 0,
                                 "cylindrical axis second source");
  if (is_valid(source.target_u))
    validate_index_range(state.plan_, source.target_u, region_min, region_max,
                         "cylindrical axis auxiliary target");
  if (is_valid(source.conductivity))
    validate_index_range(state.plan_, source.conductivity, region_min, region_max,
                         "cylindrical axis conductivity");
  if (is_valid(source.condinv))
    validate_index_range(state.plan_, source.condinv, region_min, region_max,
                         "cylindrical axis conductivity inverse");
  if (is_valid(source.target_cond))
    validate_index_range(state.plan_, source.target_cond, region_min, region_max,
                         "cylindrical axis conductivity target");
  return result;
}

nvidia::bfast_launch compile_bfast(const BfastUpdate &source, NvidiaBackendState &state,
                                   const fields &f) {
  const uint32_t supported_variants =
      bfast_has_pml | bfast_has_pml_aux | bfast_has_conductivity;
  if (source.region.variant_key & ~supported_variants)
    throw std::invalid_argument("BFAST descriptor has an unsupported variant bit");

  const bool have_pml = (source.region.variant_key & bfast_has_pml) != 0;
  const bool have_pml_u = (source.region.variant_key & bfast_has_pml_aux) != 0;
  const bool have_conductivity =
      (source.region.variant_key & bfast_has_conductivity) != 0;
  const bool complete_pml = is_valid(source.pml.sig) && is_valid(source.pml.kap) &&
                            is_valid(source.pml.siginv);
  const bool complete_pml_u = is_valid(source.pml_u.sig) && is_valid(source.pml_u.kap) &&
                              is_valid(source.pml_u.siginv);
  if (have_pml != complete_pml || have_pml_u != complete_pml_u ||
      have_pml_u != is_valid(source.target_u) ||
      have_conductivity != is_valid(source.condinv) ||
      (have_pml && have_conductivity) != is_valid(source.target_cond))
    throw std::invalid_argument("BFAST descriptor variant bits and auxiliary arrays disagree");
  if ((!have_pml && (is_valid(source.pml.sig) || is_valid(source.pml.kap) ||
                     is_valid(source.pml.siginv))) ||
      (!have_pml_u && (is_valid(source.pml_u.sig) || is_valid(source.pml_u.kap) ||
                       is_valid(source.pml_u.siginv))))
    throw std::invalid_argument("BFAST descriptor has a partial disabled PML profile");
  if (!std::isfinite(source.k1) || !std::isfinite(source.k2))
    throw std::invalid_argument("BFAST descriptor coefficient is non-finite");
  if (!is_valid(source.source1) && !is_valid(source.source2))
    throw std::invalid_argument("BFAST descriptor has no source field");
  if (source.region.chunk < 0 || source.region.chunk >= f.num_chunks ||
      !(is_D(source.region.c) || is_B(source.region.c)) || source.region.cmp < 0 ||
      source.region.cmp > 1)
    throw std::invalid_argument("BFAST descriptor has invalid target identity");
  if (!f.chunks[source.region.chunk] || !f.chunks[source.region.chunk]->is_mine())
    throw std::invalid_argument("BFAST descriptor references a nonlocal chunk");
  const fields_chunk &fc = *f.chunks[source.region.chunk];
  const StorageKey expected_target{source.region.chunk, int(array_kind::f), int(source.region.c),
                                   source.region.cmp, 0};
  const StorageKey expected_state{source.region.chunk, int(array_kind::f_bfast),
                                  int(source.region.c), source.region.cmp, 0};
  if (!is_valid(source.target) || source.target.value >= state.plan_.keys.size() ||
      !(state.plan_.keys[source.target.value] == expected_target) ||
      !is_valid(source.f_bfast) || source.f_bfast.value >= state.plan_.keys.size() ||
      !(state.plan_.keys[source.f_bfast.value] == expected_state) ||
      is_valid(state.plan_.arrays[source.f_bfast.value].alias_of))
    throw std::invalid_argument("BFAST target or persistent-state identity is invalid");

  const component target = source.region.c;
  bool have_plus = false, have_minus = false;
  component plus = NO_COMPONENT, minus = NO_COMPONENT;
  direction plus_direction = NO_DIRECTION, minus_direction = NO_DIRECTION;
  const direction target_direction = component_direction(target);
  FOR_COMPONENTS(candidate) {
    if (!((is_electric(target) && is_magnetic(candidate)) ||
          (is_D(target) && is_magnetic(candidate)) ||
          (is_magnetic(target) && is_electric(candidate)) ||
          (is_B(target) && is_electric(candidate))))
      continue;
    direction candidate_direction = component_direction(candidate);
    if (target_direction == candidate_direction || !fc.gv.has_field(candidate) ||
        !fc.gv.has_field(target))
      continue;
    direction target_xyz = target_direction >= R ? direction(target_direction - 3) : target_direction;
    direction candidate_xyz =
        candidate_direction >= R ? direction(candidate_direction - 3) : candidate_direction;
    direction derivative = direction((3 + 2 * target_xyz - candidate_xyz) % 3);
    if ((target_direction >= R || candidate_direction >= R) && derivative < Z)
      derivative = direction(derivative + 3);
    if (!(has_direction(fc.gv.dim, derivative) ||
          (fc.gv.dim == Dcyl && has_field_direction(fc.gv.dim, derivative))))
      continue;
    const bool negative = ((3 + target_xyz - candidate_xyz) % 3) == 2;
    if (negative) {
      have_minus = true;
      minus = candidate;
      minus_direction = derivative;
    }
    else {
      have_plus = true;
      plus = candidate;
      plus_direction = derivative;
    }
  }
  ArrayId expected_source1 =
      have_plus ? f.array_catalog->find(StorageKey{source.region.chunk, int(array_kind::f),
                                                   int(plus), source.region.cmp, 0})
                : invalid_array();
  ArrayId expected_source2 =
      have_minus ? f.array_catalog->find(StorageKey{source.region.chunk, int(array_kind::f),
                                                    int(minus), source.region.cmp, 0})
                 : invalid_array();
  if (fc.gv.dim == Dcyl) {
    if (target_direction == R)
      expected_source1 = invalid_array();
    else if (target_direction == Z) {
      expected_source1 = f.array_catalog->find(
          StorageKey{source.region.chunk, int(array_kind::f_rderiv_int), -1, -1, 0});
      expected_source2 = invalid_array();
    }
  }
  if (source.source1 != expected_source1 || source.source2 != expected_source2)
    throw std::invalid_argument(
        "BFAST source identity does not match its curl target (got " +
        std::to_string(source.source1.value) + "/" + std::to_string(source.source2.value) +
        ", expected " + std::to_string(expected_source1.value) + "/" +
        std::to_string(expected_source2.value) + ")");
  ptrdiff_t expected_stride1 = have_plus ? fc.gv.stride(plus_direction) : 0;
  ptrdiff_t expected_stride2 = have_minus ? fc.gv.stride(minus_direction) : 0;
  if (is_D(target)) {
    expected_stride1 = -expected_stride1;
    expected_stride2 = -expected_stride2;
  }
  if (source.stride1 != expected_stride1 || source.stride2 != expected_stride2)
    throw std::invalid_argument("BFAST source stride does not match its curl target");
  realnum expected_k1 = have_minus ? f.bfast_scaled_k[component_index(minus)] : 0;
  realnum expected_k2 = have_plus ? f.bfast_scaled_k[component_index(plus)] : 0;
  if (is_D(target)) {
    expected_k1 = -expected_k1;
    expected_k2 = -expected_k2;
  }
  if (source.k1 != double(expected_k1) || source.k2 != double(expected_k2))
    throw std::invalid_argument("BFAST coefficients do not match live coordinate routing");
  const ArrayId mutable_arrays[] = {source.target, source.f_bfast, source.target_u,
                                    source.target_cond};
  const ArrayId input_arrays[] = {source.source1, source.source2};
  for (size_t i = 0; i < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++i) {
    if (!is_valid(mutable_arrays[i])) continue;
    for (size_t j = 0; j < sizeof(input_arrays) / sizeof(input_arrays[0]); ++j)
      if (mutable_arrays[i] == input_arrays[j])
        throw std::invalid_argument("BFAST descriptor aliases mutable and input state");
    for (size_t j = i + 1; j < sizeof(mutable_arrays) / sizeof(mutable_arrays[0]); ++j)
      if (is_valid(mutable_arrays[j]) && mutable_arrays[i] == mutable_arrays[j])
        throw std::invalid_argument("BFAST descriptor aliases mutable state");
  }

  nvidia::bfast_launch result = {};
  result.region = flat_region_for(source.region);
  result.precision = scalar_precision_for(state.plan_, source.target, "BFAST target");
  result.target = device_address(state, source.target, "BFAST target");
  result.source1 =
      optional_device_address(state, source.source1, result.precision, "BFAST first source");
  result.source2 =
      optional_device_address(state, source.source2, result.precision, "BFAST second source");
  result.stride1 = source.stride1;
  result.stride2 = source.stride2;
  result.f_bfast = optional_mutable_device_address(state, source.f_bfast, result.precision,
                                                    "BFAST persistent state");
  if (!result.f_bfast)
    throw std::invalid_argument("BFAST descriptor has no persistent state");
  result.target_u = optional_mutable_device_address(state, source.target_u, result.precision,
                                                     "BFAST auxiliary target");
  result.conductivity_inverse = optional_device_address(
      state, source.condinv, result.precision, "BFAST conductivity inverse");
  result.target_conductivity = optional_mutable_device_address(
      state, source.target_cond, result.precision, "BFAST conductivity target");
  result.pml =
      compile_pml_profile(source.pml, result.region, result.precision, state, "BFAST main PML");
  result.pml_u = compile_pml_profile(source.pml_u, result.region, result.precision, state,
                                     "BFAST auxiliary PML");
  result.k1 = source.k1;
  result.k2 = source.k2;

  const ptrdiff_t region_max = checked_region_max(result.region);
  validate_index_range(state.plan_, source.target, ptrdiff_t(result.region.base), region_max,
                       "BFAST target");
  validate_index_range(state.plan_, source.f_bfast, ptrdiff_t(result.region.base), region_max,
                       "BFAST persistent state");
  if (is_valid(source.source1))
    validate_shifted_index_range(state.plan_, source.source1, ptrdiff_t(result.region.base),
                                 region_max, 0, source.stride1, 0, 0,
                                 "BFAST first source");
  if (is_valid(source.source2))
    validate_shifted_index_range(state.plan_, source.source2, ptrdiff_t(result.region.base),
                                 region_max, 0, source.stride2, 0, 0,
                                 "BFAST second source");
  if (is_valid(source.target_u))
    validate_index_range(state.plan_, source.target_u, ptrdiff_t(result.region.base), region_max,
                         "BFAST auxiliary target");
  if (is_valid(source.condinv))
    validate_index_range(state.plan_, source.condinv, ptrdiff_t(result.region.base), region_max,
                         "BFAST conductivity inverse");
  if (is_valid(source.target_cond))
    validate_index_range(state.plan_, source.target_cond, ptrdiff_t(result.region.base), region_max,
                         "BFAST conductivity target");
  return result;
}

nvidia::beta_launch compile_beta(const BetaUpdate &source, NvidiaBackendState &state) {
  const uint32_t supported_variants =
      beta_has_pml | beta_has_pml_aux | beta_has_conductivity;
  if (source.region.variant_key & ~supported_variants)
    throw std::invalid_argument("beta descriptor has an unsupported variant bit");

  const bool have_pml = (source.region.variant_key & beta_has_pml) != 0;
  const bool have_pml_u = (source.region.variant_key & beta_has_pml_aux) != 0;
  const bool have_conductivity =
      (source.region.variant_key & beta_has_conductivity) != 0;
  const bool complete_pml = is_valid(source.pml.sig) && is_valid(source.pml.kap) &&
                            is_valid(source.pml.siginv);
  const bool complete_pml_u = is_valid(source.pml_u.sig) && is_valid(source.pml_u.kap) &&
                              is_valid(source.pml_u.siginv);
  if (have_pml != complete_pml || have_pml_u != complete_pml_u ||
      have_pml_u != is_valid(source.target_u) ||
      have_conductivity != is_valid(source.condinv) ||
      (have_pml && have_conductivity) != is_valid(source.target_cond))
    throw std::invalid_argument("beta descriptor variant bits and auxiliary arrays disagree");
  if ((!have_pml && (is_valid(source.pml.sig) || is_valid(source.pml.kap) ||
                     is_valid(source.pml.siginv))) ||
      (!have_pml_u && (is_valid(source.pml_u.sig) || is_valid(source.pml_u.kap) ||
                       is_valid(source.pml_u.siginv))))
    throw std::invalid_argument("beta descriptor has a partial disabled PML profile");
  if (!std::isfinite(source.betadt))
    throw std::invalid_argument("beta descriptor coefficient is non-finite");
  if (source.target == source.source || source.target == source.target_u ||
      source.target == source.target_cond ||
      (is_valid(source.target_u) && source.target_u == source.target_cond) ||
      (is_valid(source.target_u) && source.target_u == source.source) ||
      (is_valid(source.target_cond) && source.target_cond == source.source))
    throw std::invalid_argument("beta descriptor aliases mutable and input state");

  nvidia::beta_launch result = {};
  result.region = flat_region_for(source.region);
  result.precision = scalar_precision_for(state.plan_, source.target, "beta target");
  result.target = device_address(state, source.target, "beta target");
  result.source = optional_device_address(state, source.source, result.precision, "beta source");
  if (!result.source) throw std::invalid_argument("beta descriptor has no source field");
  result.target_u = optional_mutable_device_address(state, source.target_u, result.precision,
                                                     "beta auxiliary target");
  result.conductivity_inverse = optional_device_address(
      state, source.condinv, result.precision, "beta conductivity inverse");
  result.target_conductivity = optional_mutable_device_address(
      state, source.target_cond, result.precision, "beta conductivity target");
  result.pml =
      compile_pml_profile(source.pml, result.region, result.precision, state, "beta main PML");
  result.pml_u = compile_pml_profile(source.pml_u, result.region, result.precision, state,
                                     "beta auxiliary PML");
  result.betadt = source.betadt;

  const ptrdiff_t region_max = checked_region_max(result.region);
  validate_index_range(state.plan_, source.target, ptrdiff_t(result.region.base), region_max,
                       "beta target");
  validate_index_range(state.plan_, source.source, ptrdiff_t(result.region.base), region_max,
                       "beta source");
  if (is_valid(source.target_u))
    validate_index_range(state.plan_, source.target_u, ptrdiff_t(result.region.base), region_max,
                         "beta auxiliary target");
  if (is_valid(source.condinv))
    validate_index_range(state.plan_, source.condinv, ptrdiff_t(result.region.base), region_max,
                         "beta conductivity inverse");
  if (is_valid(source.target_cond))
    validate_index_range(state.plan_, source.target_cond, ptrdiff_t(result.region.base), region_max,
                         "beta conductivity target");
  return result;
}

nvidia::constitutive_launch compile_constitutive(const ConstitutiveUpdate &source,
                                                 NvidiaBackendState &state) {
  const uint32_t supported_variants = constitutive_has_pml | constitutive_one_offdiagonal |
                                      constitutive_two_offdiagonals |
                                      constitutive_has_nonlinearity | constitutive_has_minus_p |
                                      constitutive_axis_override;
  if (source.region.variant_key & ~supported_variants)
    throw std::invalid_argument("constitutive descriptor requires unsupported auxiliary state");
  if (is_valid(source.previous_w))
    throw std::invalid_argument("constitutive descriptor contains unsupported auxiliary arrays");

  const bool have_pml = (source.region.variant_key & constitutive_has_pml) != 0;
  const bool have_nonlinearity =
      (source.region.variant_key & constitutive_has_nonlinearity) != 0;
  const bool have_offdiagonal1 = (source.region.variant_key & constitutive_one_offdiagonal) != 0;
  const bool have_offdiagonal2 = (source.region.variant_key & constitutive_two_offdiagonals) != 0;
  if (have_offdiagonal2 && !have_offdiagonal1)
    throw std::invalid_argument("constitutive descriptor has a second off-diagonal without first");
  if (have_offdiagonal1 != is_valid(source.offdiagonal1) ||
      have_offdiagonal2 != is_valid(source.offdiagonal2) ||
      (have_offdiagonal1 && (!is_valid(source.cross1) || !is_valid(source.diagonal))) ||
      (have_offdiagonal2 && !is_valid(source.cross2)))
    throw std::invalid_argument(
        "constitutive descriptor anisotropy bits and operand arrays disagree");
  if (have_pml != is_valid(source.pml.sig) || have_pml != is_valid(source.target_w))
    throw std::invalid_argument("constitutive descriptor PML bit and auxiliary arrays disagree");
  if (!have_pml && (is_valid(source.pml.kap) || is_valid(source.pml.siginv)))
    throw std::invalid_argument("constitutive descriptor has a partial disabled PML profile");
  if (have_nonlinearity != is_valid(source.chi2) ||
      have_nonlinearity != is_valid(source.chi3) ||
      (have_nonlinearity && !is_valid(source.diagonal)))
    throw std::invalid_argument(
        "constitutive descriptor nonlinearity bit and operand arrays disagree");
  if (have_nonlinearity && is_valid(source.cross2) && !is_valid(source.cross1))
    throw std::invalid_argument(
        "constitutive descriptor has a second nonlinear cross field without first");

  nvidia::constitutive_launch result = {};
  result.region = flat_region_for(source.region);
  result.precision = scalar_precision_for(state.plan_, source.target, "constitutive target");
  result.target = device_address(state, source.target, "constitutive target");
  result.primary =
      optional_device_address(state, source.primary, result.precision, "constitutive primary");
  result.cross1 =
      (have_offdiagonal1 || (have_nonlinearity && is_valid(source.cross1)))
          ? optional_device_address(state, source.cross1, result.precision, "constitutive cross1")
          : NULL;
  result.cross2 =
      (have_offdiagonal2 || (have_nonlinearity && is_valid(source.cross2)))
          ? optional_device_address(state, source.cross2, result.precision, "constitutive cross2")
          : NULL;
  result.diagonal =
      optional_device_address(state, source.diagonal, result.precision, "constitutive diagonal");
  result.offdiagonal1 = optional_device_address(state, source.offdiagonal1, result.precision,
                                                "constitutive off-diagonal1");
  result.offdiagonal2 = optional_device_address(state, source.offdiagonal2, result.precision,
                                                "constitutive off-diagonal2");
  result.chi2 = optional_device_address(state, source.chi2, result.precision, "constitutive chi2");
  result.chi3 = optional_device_address(state, source.chi3, result.precision, "constitutive chi3");
  result.primary_stride = source.primary_stride;
  result.cross1_stride = source.cross1_stride;
  result.cross2_stride = source.cross2_stride;
  result.target_w = optional_mutable_device_address(state, source.target_w, result.precision,
                                                    "constitutive PML target");
  result.pml =
      compile_pml_profile(source.pml, result.region, result.precision, state, "constitutive PML");
  if (!result.primary) throw std::invalid_argument("constitutive descriptor has no primary field");

  const ptrdiff_t region_max = checked_region_max(result.region);
  validate_index_range(state.plan_, source.target, ptrdiff_t(result.region.base), region_max,
                       "constitutive target");
  validate_index_range(state.plan_, source.primary, ptrdiff_t(result.region.base), region_max,
                       "constitutive primary");
  if (have_nonlinearity) {
    validate_index_range(state.plan_, source.chi2, ptrdiff_t(result.region.base), region_max,
                         "constitutive chi2");
    validate_index_range(state.plan_, source.chi3, ptrdiff_t(result.region.base), region_max,
                         "constitutive chi3");
    if (is_valid(source.cross1)) {
      const ptrdiff_t negative_cross_stride =
          checked_negate(source.cross1_stride, "constitutive nonlinear cross1");
      const ptrdiff_t combined_stride = checked_shift(
          source.primary_stride, negative_cross_stride, "constitutive nonlinear cross1");
      validate_shifted_index_range(state.plan_, source.cross1, ptrdiff_t(result.region.base),
                                   region_max, 0,
                                   negative_cross_stride, source.primary_stride, combined_stride,
                                   "constitutive nonlinear cross1");
    }
    if (is_valid(source.cross2)) {
      const ptrdiff_t negative_cross_stride =
          checked_negate(source.cross2_stride, "constitutive nonlinear cross2");
      const ptrdiff_t combined_stride = checked_shift(
          source.primary_stride, negative_cross_stride, "constitutive nonlinear cross2");
      validate_shifted_index_range(state.plan_, source.cross2, ptrdiff_t(result.region.base),
                                   region_max, 0,
                                   negative_cross_stride, source.primary_stride, combined_stride,
                                   "constitutive nonlinear cross2");
    }
  }
  if (is_valid(source.diagonal))
    validate_index_range(state.plan_, source.diagonal, ptrdiff_t(result.region.base), region_max,
                         "constitutive diagonal");
  if (have_offdiagonal1) {
    const ptrdiff_t negative_cross_stride =
        checked_negate(source.cross1_stride, "constitutive cross1");
    const ptrdiff_t combined_stride =
        checked_shift(source.primary_stride, negative_cross_stride, "constitutive cross1");
    validate_shifted_index_range(state.plan_, source.cross1, ptrdiff_t(result.region.base),
                                 region_max, 0, negative_cross_stride, source.primary_stride,
                                 combined_stride, "constitutive cross1");
    validate_shifted_index_range(state.plan_, source.offdiagonal1, ptrdiff_t(result.region.base),
                                 region_max, 0, source.primary_stride, 0, source.primary_stride,
                                 "constitutive off-diagonal1");
  }
  if (have_offdiagonal2) {
    const ptrdiff_t negative_cross_stride =
        checked_negate(source.cross2_stride, "constitutive cross2");
    const ptrdiff_t combined_stride =
        checked_shift(source.primary_stride, negative_cross_stride, "constitutive cross2");
    validate_shifted_index_range(state.plan_, source.cross2, ptrdiff_t(result.region.base),
                                 region_max, 0, negative_cross_stride, source.primary_stride,
                                 combined_stride, "constitutive cross2");
    validate_shifted_index_range(state.plan_, source.offdiagonal2, ptrdiff_t(result.region.base),
                                 region_max, 0, source.primary_stride, 0, source.primary_stride,
                                 "constitutive off-diagonal2");
  }
  if (is_valid(source.target_w))
    validate_index_range(state.plan_, source.target_w, ptrdiff_t(result.region.base), region_max,
                         "constitutive PML target");
  return result;
}

nvidia::zero_launch compile_zero(const SlabRef &source, NvidiaBackendState &state) {
  nvidia::zero_launch result = {};
  result.precision = scalar_precision_for(state.plan_, source.array, "metal-zero target");
  result.target = device_address(state, source.array, "metal-zero target");
  if (source.base < 0) throw std::out_of_range("metal-zero descriptor has a negative base");
  result.region.base = size_t(source.base);
  for (int axis = 0; axis < 3; ++axis) {
    if (source.counts[axis] <= 0 || source.strides[axis] < 0)
      throw std::invalid_argument("metal-zero descriptor has invalid geometry");
    result.region.counts[axis] = size_t(source.counts[axis]);
    result.region.strides[axis] = source.strides[axis];
  }
  validate_index_range(state.plan_, source.array, source.base, checked_region_max(result.region),
                       "metal-zero target");
  return result;
}

nvidia::zero_launch compile_zero(const ElementRef &source, NvidiaBackendState &state) {
  SlabRef slab;
  slab.array = source.array;
  slab.base = source.index;
  for (int axis = 0; axis < 3; ++axis) {
    slab.counts[axis] = 1;
    slab.strides[axis] = 0;
  }
  return compile_zero(slab, state);
}

NvidiaFiniteCheck compile_finite_check(const BufferAccess &source, uint64_t ordinal_base,
                                       NvidiaBackendState &state) {
  if (source.mode != AccessMode::read)
    throw std::invalid_argument("finite-value check access is not read-only");
  if (!is_valid(source.array.id) || source.array.id.value >= state.plan_.arrays.size())
    throw std::out_of_range("finite-value check uses an invalid ArrayId");
  const ArraySpec &spec = state.plan_.arrays[source.array.id.value];
  const StorageKey &key = state.plan_.keys[source.array.id.value];
  if (key.kind != int(array_kind::f) || spec.element_type != ElementType::realnum_value ||
      is_valid(spec.alias_of) || !spec.elements)
    throw std::invalid_argument("finite-value check access is not a canonical physical field");
  if (source.array.offset != 0 || source.array.elements != spec.elements)
    throw std::invalid_argument("finite-value check access does not cover its full allocation");
  if (key.chunk < 0 || key.component_ < 0 || key.component_ >= NUM_FIELD_COMPONENTS ||
      (key.cmp != 0 && key.cmp != 1))
    throw std::invalid_argument("finite-value check access has invalid diagnostic attribution");
  if (source.array.elements > std::numeric_limits<uint64_t>::max() - ordinal_base)
    throw std::overflow_error("finite-value check ordinal range overflow");

  NvidiaFiniteCheck result;
  result.launch.values = device_address(state, source.array.id, "finite-value check input");
  result.launch.elements = source.array.elements;
  result.launch.ordinal_base = ordinal_base;
  result.launch.precision =
      scalar_precision_for(state.plan_, source.array.id, "finite-value check input");
  result.key = key;
  return result;
}

struct CompiledHaloRef {
  void *address;
  ptrdiff_t index;
  nvidia::scalar_precision precision;
};

CompiledHaloRef compile_halo_ref(const ElementRef &source, NvidiaBackendState &state,
                                 const char *what) {
  if (source.index < 0)
    throw std::out_of_range(std::string(what) + " has a negative element index");
  const nvidia::scalar_precision precision = scalar_precision_for(state.plan_, source.array, what);
  validate_index_range(state.plan_, source.array, source.index, source.index, what);
  return CompiledHaloRef{device_address(state, source.array, what), source.index, precision};
}

size_t scalar_bytes(nvidia::scalar_precision precision) {
  return precision == nvidia::scalar_precision::f32 ? sizeof(float) : sizeof(double);
}

NvidiaCompiledHalo compile_halo(const HaloPlan &source, const fields &f, NvidiaBackendState &state,
                                size_t buffer_base, std::vector<nvidia::halo_gather_entry> &gathers,
                                std::vector<nvidia::halo_scatter_entry> &scatters) {
  if (!source.same_rank)
    throw std::invalid_argument("NVIDIA PR2 does not support remote halo exchange");

  HaloPlan canonical;
  std::string why;
  if (!remap_halo_plan(source, f.halos->arrays, *f.array_catalog, f.is_real ? 1 : 2, canonical,
                       why))
    throw std::logic_error(std::string("cannot remap local halo plan: ") + why);

  std::vector<ElementRef> gather_refs, scatter_refs;
  expand_gather(canonical, gather_refs);
  expand_scatter(canonical, scatter_refs);
  if (gather_refs.size() != source.block_elements || scatter_refs.size() != source.block_elements)
    throw std::logic_error("local halo plan expansion does not match its communication block");
  if (!source.block_elements)
    throw std::invalid_argument("cannot compile an empty local halo plan");
  if (buffer_base > std::numeric_limits<size_t>::max() - source.block_elements)
    throw std::overflow_error("NVIDIA local halo scratch index overflow");

  NvidiaCompiledHalo result;
  result.gather.first = gathers.size();
  result.gather.count = gather_refs.size();
  result.scatter.first = scatters.size();
  result.scatter.count = 0;

  bool have_precision = false;
  nvidia::scalar_precision precision = nvidia::scalar_precision::f64;
  for (size_t i = 0; i < gather_refs.size(); ++i) {
    const CompiledHaloRef ref = compile_halo_ref(gather_refs[i], state, "halo gather source");
    if (have_precision && ref.precision != precision)
      throw std::invalid_argument("local halo gather mixes storage precisions");
    precision = ref.precision;
    have_precision = true;
    gathers.push_back(nvidia::halo_gather_entry{ref.address, ref.index, buffer_base + i});
  }

  switch (source.phase) {
    case CONNECT_COPY:
    case CONNECT_NEGATE: {
      const double sign = source.phase == CONNECT_NEGATE ? -1.0 : 1.0;
      for (size_t i = 0; i < scatter_refs.size(); ++i) {
        const CompiledHaloRef ref = compile_halo_ref(scatter_refs[i], state, "halo scatter target");
        if (ref.precision != precision)
          throw std::invalid_argument("local halo scatter mixes storage precisions");
        scatters.push_back(nvidia::halo_scatter_entry{ref.address, ref.index, NULL, 0,
                                                      buffer_base + i, sign, 0.0});
      }
      break;
    }
    case CONNECT_PHASE: {
      if (f.is_real) throw std::invalid_argument("phase halo requires complex field storage");
      if (source.block_elements % 2)
        throw std::invalid_argument("phase halo block does not contain complex pairs");
      if (source.phase_values.size() != source.block_elements / 2)
        throw std::invalid_argument("phase halo values do not match the transfer count");
      for (size_t i = 0; i < source.phase_values.size(); ++i) {
        const CompiledHaloRef real_ref =
            compile_halo_ref(scatter_refs[2 * i], state, "phase halo real target");
        const CompiledHaloRef imag_ref =
            compile_halo_ref(scatter_refs[2 * i + 1], state, "phase halo imaginary target");
        if (real_ref.precision != precision || imag_ref.precision != precision)
          throw std::invalid_argument("phase halo mixes storage precisions");
        scatters.push_back(nvidia::halo_scatter_entry{
            real_ref.address, real_ref.index, imag_ref.address, imag_ref.index, buffer_base + 2 * i,
            double(source.phase_values[i].real()), double(source.phase_values[i].imag())});
      }
      break;
    }
    default: throw std::invalid_argument("local halo plan has an invalid transform");
  }

  result.gather.precision = precision;
  result.scatter.count = scatters.size() - result.scatter.first;
  result.scatter.precision = precision;
  return result;
}

bool has_polarization(const fields &f) {
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    FOR_FIELD_TYPES(ft) if (f.chunks[chunk]->pol[ft]) return true;
  }
  return false;
}

void set_reason(std::string &why, size_t operation, const char *detail) {
  std::ostringstream message;
  message << "NVIDIA PR2 unsupported operation at index " << operation << ": " << detail;
  why = message.str();
}

} // namespace

NvidiaBackend::NvidiaBackend(fields &f, const execution_options &options, int selected_device)
    : f_(f), options_(options), device_(selected_device), device_memory_bytes_(0),
      next_state_token_(1) {
  if (device_ < 0) throw std::invalid_argument("NVIDIA backend requires a resolved device ID");
  device_memory_bytes_ = nvidia::properties_for_device(device_).total_memory;
}

NvidiaBackend::~NvidiaBackend() {}

BackendState *NvidiaBackend::create_state(const StoragePlan &plan) {
  std::unique_ptr<NvidiaBackendState> state;
  std::string local_error;
  try {
    StoragePlan device_plan = plan;
    apply_precision_policy(device_plan, policy_for(options_.precision));
    std::string why;
    if (!validate_alias_precisions(device_plan, why))
      throw std::invalid_argument(std::string("invalid NVIDIA storage aliases: ") + why);
    if (next_state_token_ == 0)
      throw std::overflow_error("NVIDIA backend state token overflow");
    state.reset(new NvidiaBackendState(this, device_plan, device_, next_state_token_++));
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown NVIDIA state-construction failure";
  }
  if (or_to_all(!local_error.empty())) {
    state.reset();
    if (local_error.empty()) local_error = "another rank failed to construct NVIDIA backend state";
    throw std::runtime_error(local_error);
  }
  return state.release();
}

void NvidiaBackend::initialize(const InitializationPlan &, BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  state.initialized_ = false;
  std::string local_error;
  try {
    if (state.transfer_failed_)
      throw std::logic_error("NVIDIA transfer stream failed; recreate backend state");
    if (state.device_authoritative_)
      throw std::logic_error(
          "cannot refresh NVIDIA storage from a stale host mirror after device stepping");
    if (!f_.array_catalog)
      throw std::logic_error("NVIDIA initialization requires a prepared CPU catalog");
    const CpuArrayCatalog &catalog = *f_.array_catalog;
    if (catalog.size() != state.plan_.arrays.size())
      throw std::logic_error("CPU catalog changed after NVIDIA storage was finalized");

    struct Upload {
      nvidia::allocation_id id;
      size_t bytes;
    };
    std::vector<Upload> uploads;
    uploads.reserve(state.plan_.arrays.size());
    size_t staging_bytes = 0;
    for (size_t i = 0; i < state.plan_.arrays.size(); ++i) {
      const ArraySpec &device_spec = state.plan_.arrays[i];
      const ArraySpec &host_spec = catalog.spec(ArrayId{uint32_t(i)});
      if (host_spec.id != device_spec.id || host_spec.role != device_spec.role ||
          host_spec.element_type != device_spec.element_type ||
          host_spec.elements != device_spec.elements || host_spec.alias_of != device_spec.alias_of)
        throw std::logic_error("CPU catalog no longer matches the NVIDIA storage plan");
      if (is_valid(device_spec.alias_of)) {
        if (catalog.resolve_untyped(device_spec.id) !=
            catalog.resolve_untyped(device_spec.alias_of))
          throw std::logic_error("NVIDIA storage alias does not match the CPU catalog alias");
        continue;
      }
      const size_t bytes = storage_bytes(device_spec);
      uploads.push_back(Upload{device_spec.id.value, bytes});
      if (bytes > staging_bytes) staging_bytes = bytes;
    }

    state.ensure_staging(staging_bytes);
    for (size_t i = 0; i < uploads.size(); ++i) {
      const Upload &upload = uploads[i];
      const ArraySpec &spec = state.plan_.arrays[upload.id];
      const void *source = catalog.resolve_untyped(spec.id);
      if (!source) throw std::logic_error("CPU catalog contains a null canonical allocation");
      host_to_storage(state.staging_.data(), source, spec, spec.elements);
      state.arenas_->copy_from_host_async(upload.id, 0, state.staging_.data(), upload.bytes,
                                          *state.transfer_);
      /* Reuse one bounded pinned allocation instead of pinning a mirror of the
         complete simulation. Initialization is off the steady-state path. */
      state.transfer_->synchronize();
    }
  }
  catch (const std::exception &error) {
    local_error = error.what();
    state.transfer_failed_ = true;
    try {
      state.transfer_->synchronize();
    }
    catch (...) {
    }
  }
  catch (...) {
    local_error = "unknown NVIDIA initialization failure";
    state.transfer_failed_ = true;
    try {
      state.transfer_->synchronize();
    }
    catch (...) {
    }
  }
  if (or_to_all(!local_error.empty())) {
    if (local_error.empty()) local_error = "another rank failed to initialize NVIDIA storage";
    throw std::runtime_error(local_error);
  }
  state.initialized_ = true;
  state.device_authoritative_ = false;
}

MaterialClassification NvidiaBackend::classify_state(const StoragePlan &plan,
                                                     BackendState &raw_state) {
  checked_state(raw_state);
  /* PR1 deliberately uses the already-populated CPU catalog as compatibility
     initialization. Classification therefore runs on those exact host values
     before device execution can make them stale. */
  return classify(f_, plan);
}

void NvidiaBackend::finalize_storage(const StoragePlan &, BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  if (!state.initialized_) throw std::logic_error("cannot finalize uninitialized NVIDIA storage");
  /* PR1 retains the complete provisional arena. Device-side elision and
     migration require the PR2 operation lowering and are not guessed here. */
}

Executable *NvidiaBackend::compile(const StepPlan &plan, BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  if (!state.initialized_)
    throw std::logic_error("cannot compile against uninitialized NVIDIA storage");
  std::unique_ptr<NvidiaExecutable> executable;
  std::string local_error;
  try {
    if (plan.signature != compute_step_plan_signature(plan))
      throw std::invalid_argument("NVIDIA PR2 received a stale StepPlan signature");
    if (plan.program != StepProgram::ordinary)
      throw std::invalid_argument("NVIDIA PR2 does not support solve_cw");
    if (count_processors() != 1)
      throw std::invalid_argument("NVIDIA PR2 does not yet support MPI timestepping");
    if (f_.gv.dim != Dcyl && f_.m != 0.0)
      throw std::invalid_argument("NVIDIA nonzero m requires cylindrical coordinates");
    if (plan.cylindrical_m != f_.m ||
        plan.cylindrical_origin_r.size() != size_t(f_.num_chunks) ||
        plan.cylindrical_zero_near_origin.size() != size_t(f_.num_chunks))
      throw std::invalid_argument("NVIDIA cylindrical coordinate fingerprint is stale");
    for (int i = 0; i < f_.num_chunks; ++i)
      if (!f_.chunks[i] || f_.chunks[i]->m != f_.m ||
          plan.cylindrical_origin_r[i] != f_.chunks[i]->gv.origin_r() ||
          bool(plan.cylindrical_zero_near_origin[i]) !=
              f_.chunks[i]->zero_fields_near_cylorigin)
        throw std::invalid_argument("NVIDIA cylindrical chunk fingerprint is stale");
    if (f_.beta != 0.0 && f_.gv.dim != D2)
      throw std::invalid_argument("NVIDIA nonzero beta requires a 2-D Cartesian grid");
    if (plan.beta != f_.beta)
      throw std::invalid_argument("NVIDIA beta coordinate fingerprint is stale");
    if (plan.bfast_scaled_k != f_.bfast_scaled_k)
      throw std::invalid_argument("NVIDIA BFAST coordinate fingerprint is stale");
    if (f_.beta != 0.0 && plan.beta_updates.empty())
      throw std::invalid_argument("NVIDIA nonzero-beta plan has no beta descriptors");
    if (f_.bfast_scaled_k.size() != 3)
      throw std::invalid_argument("NVIDIA BFAST coordinate must contain exactly three values");
    const bool use_bfast = std::any_of(f_.bfast_scaled_k.begin(), f_.bfast_scaled_k.end(),
                                      [](double k) { return k != 0.0; });
    if (use_bfast != !plan.bfast_updates.empty())
      throw std::invalid_argument("NVIDIA BFAST coordinate and descriptor state disagree");
    if (f_.is_phasing())
      throw std::invalid_argument("NVIDIA PR2 does not support material phasing");
    if (f_.fluxes)
      throw std::invalid_argument("NVIDIA PR3 does not support legacy time-domain flux monitors");
    if (has_polarization(f_) && !f_.descriptors)
      throw std::invalid_argument("NVIDIA dispersion state has no prepared descriptors");
    if (f_.descriptors) {
      size_t live_states = 0;
      for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
        if (!f_.chunks[chunk]->is_mine()) continue;
        FOR_FIELD_TYPES(ft) for (polarization_state *p = f_.chunks[chunk]->pol[ft]; p;
                                p = p->next) ++live_states;
      }
      if (live_states != f_.descriptors->polarizations.size())
        throw std::invalid_argument("NVIDIA dispersion descriptors are stale");
      for (size_t i = 0; i < f_.descriptors->polarizations.size(); ++i) {
        const PolarizationDescriptor &d = f_.descriptors->polarizations[i];
        if (d.kind != SusceptibilityKind::lorentzian &&
            d.kind != SusceptibilityKind::gyrotropic)
          throw std::invalid_argument(std::string("NVIDIA does not support polarization kind ") +
                                      susceptibility_kind_name(d.kind));
        if (d.kind == SusceptibilityKind::lorentzian && d.lorentzian_states.empty())
          throw std::invalid_argument("NVIDIA Lorentzian descriptor has no resident state");
        if (d.kind == SusceptibilityKind::gyrotropic && d.gyrotropic_states.empty())
          throw std::invalid_argument("NVIDIA gyrotropic descriptor has no resident state");
        if (f_.gv.dim == Dcyl && d.kind == SusceptibilityKind::gyrotropic)
          throw std::invalid_argument("NVIDIA cylindrical gyrotropic media are not supported");
      }
    }
    if (!connections_are_current(f_))
      throw std::invalid_argument(
          "NVIDIA PR2 requires Phase 1 to finalize halo topology before backend compilation");
    if (!f_.halos || !f_.array_catalog)
      throw std::logic_error("NVIDIA PR2 requires prepared halo and storage plans");

    const SourcePlan *source_plan = f_.descriptors ? &f_.descriptors->sources : NULL;
    if (f_.sources && !source_plan)
      throw std::logic_error("NVIDIA PR3 requires prepared source descriptors");
    if (source_plan) {
      if (source_plan->source_times.size() != source_plan->scalars.size())
        throw std::invalid_argument("NVIDIA source scalar and source-time counts differ");
      for (size_t i = 0; i < source_plan->source_times.size(); ++i) {
        const SourceTimeDescriptor &time = source_plan->source_times[i];
        if (time.source_time_id != i || time.scalar_slot != i)
          throw std::invalid_argument("NVIDIA source-time descriptors are not canonical");
        switch (time.kind) {
          case SourceTimeKind::gaussian:
          case SourceTimeKind::continuous:
          case SourceTimeKind::host_custom: break;
          default: throw std::invalid_argument("NVIDIA source-time kind is invalid");
        }
      }
    }

    std::vector<NvidiaCompiledOperation> operations;
    std::vector<nvidia::curl_launch> curl_updates;
    std::vector<nvidia::cylindrical_radial_prefix_launch> cylindrical_radial_prefixes;
    std::vector<nvidia::bfast_launch> bfast_updates;
    std::vector<nvidia::beta_launch> beta_updates;
    std::vector<nvidia::cylindrical_m_launch> cylindrical_m_updates;
    std::vector<nvidia::cylindrical_axis_launch> cylindrical_axis_updates;
    std::vector<NvidiaCompiledCylindricalOriginAction> cylindrical_origin_actions;
    std::vector<nvidia::constitutive_launch> constitutive_updates;
    std::vector<nvidia::zero_launch> zero_updates;
    std::vector<NvidiaCompiledHalo> halo_plans;
    std::vector<nvidia::halo_gather_entry> halo_gathers;
    std::vector<nvidia::halo_scatter_entry> halo_scatters;
    std::vector<NvidiaFiniteCheck> finite_checks;
    size_t halo_scratch_bytes = 0;
    uint64_t finite_elements = 0;
    std::vector<nvidia::source_batch_launch> source_batches;
    std::vector<nvidia::source_point> source_points;
    std::vector<nvidia::array_copy_launch> source_copies;
    std::vector<nvidia::compiled_polarization_update> polarization_updates;
    std::vector<nvidia::polarization_subtract_launch> polarization_subtractions;
    std::vector<nvidia::dft_launch> dft_updates;
    std::vector<double> dft_omega;
    std::vector<NvidiaCompiledMagneticState> magnetic_states;
    size_t magnetic_snapshot_bytes = 0;
    size_t source_staging_elements = 0;
    operations.reserve(plan.operations.size());
    cylindrical_radial_prefixes.reserve(plan.cylindrical_radial_prefixes.size());
    for (const CylindricalRadialPrefix &prefix : plan.cylindrical_radial_prefixes)
      cylindrical_radial_prefixes.push_back(
          compile_cylindrical_radial_prefix(prefix, state, f_));
    std::vector<unsigned int> radial_prefix_references(plan.cylindrical_radial_prefixes.size(), 0);
    std::vector<unsigned int> cylindrical_m_references(plan.cylindrical_m_updates.size(), 0);
    std::vector<unsigned int> cylindrical_axis_references(plan.cylindrical_axis_updates.size(), 0);
    std::vector<unsigned int> cylindrical_zero_references(plan.cylindrical_zero_slabs.size(), 0);
    std::vector<unsigned int> cylindrical_origin_references(plan.cylindrical_origin_actions.size(),
                                                            0);
    bfast_updates.reserve(plan.bfast_updates.size());
    for (const BfastUpdate &update : plan.bfast_updates)
      bfast_updates.push_back(compile_bfast(update, state, f_));
    std::vector<unsigned int> bfast_references(plan.bfast_updates.size(), 0);

    const StepPlan canonical = build_step_plan(f_, plan.program);
    if (canonical.magnetic_state_arrays.size() != plan.magnetic_state_arrays.size())
      throw std::invalid_argument("NVIDIA magnetic snapshot descriptor count is non-canonical");
    for (size_t i = 0; i < plan.magnetic_state_arrays.size(); ++i) {
      const MagneticStateArray &got = plan.magnetic_state_arrays[i];
      const MagneticStateArray &want = canonical.magnetic_state_arrays[i];
      if (got.chunk != want.chunk || got.c != want.c || got.cmp != want.cmp ||
          got.family != want.family || got.live != want.live || got.elements != want.elements ||
          got.average != want.average)
        throw std::invalid_argument("NVIDIA magnetic snapshot descriptors are non-canonical");
      magnetic_states.push_back(
          compile_magnetic_state(got, state, f_, magnetic_snapshot_bytes));
    }
    const MagneticHalfStep &got_half = plan.magnetic_half_step;
    const MagneticHalfStep &want_half = canonical.magnetic_half_step;
    if (got_half.evaluate_b_sources != want_half.evaluate_b_sources ||
        got_half.update_b != want_half.update_b ||
        got_half.apply_b_sources != want_half.apply_b_sources ||
        got_half.transfer_b != want_half.transfer_b ||
        got_half.evaluate_h_sources != want_half.evaluate_h_sources ||
        got_half.update_h != want_half.update_h || got_half.transfer_h != want_half.transfer_h)
      throw std::invalid_argument("NVIDIA magnetic half-step schedule is non-canonical");

    /* Capability validation is deliberately fail-fast. The first local reason
       is stable and actionable; the collective below only establishes that no
       rank can enter execution after any rank rejected its plan. */
    for (size_t oi = 0; oi < plan.operations.size(); ++oi) {
      const Operation &op = plan.operations[oi];
      NvidiaCompiledOperation compiled = {};
      compiled.kind = op.kind;
      compiled.source_time_offset = op.source_time_offset;
      switch (op.kind) {
        case OpKind::update_db: {
          if (size_t(op.descriptor_index) + op.descriptor_count > plan.db_updates.size()) {
            set_reason(local_error, oi, "curl descriptor span is out of range");
            break;
          }
          compiled.first = curl_updates.size();
          for (size_t i = op.descriptor_index;
               i < size_t(op.descriptor_index) + op.descriptor_count; ++i) {
            const CurlUpdate &source = plan.db_updates[i];
            nvidia::curl_launch curl = compile_curl(source, state);
            const bool cylindrical = f_.chunks[source.region.chunk]->gv.dim == Dcyl;
            const direction target_direction = component_direction(source.region.c);
            const bool has_prefix = source.radial_prefix_index != UINT32_MAX;
            if (cylindrical) {
              const fields_chunk &fc = *f_.chunks[source.region.chunk];
              const component source_base = is_D(source.region.c) ? Hx : Ex;
              direction plus_component_direction = NO_DIRECTION,
                        minus_component_direction = NO_DIRECTION;
              direction plus_direction = NO_DIRECTION, minus_direction = NO_DIRECTION;
              if (target_direction == R) {
                plus_component_direction = Z;
                minus_component_direction = P;
                plus_direction = P;
                minus_direction = Z;
              }
              else if (target_direction == P) {
                plus_component_direction = R;
                minus_component_direction = Z;
                plus_direction = Z;
                minus_direction = R;
              }
              else if (target_direction == Z) {
                plus_component_direction = P;
                minus_component_direction = R;
                plus_direction = R;
                minus_direction = P;
              }
              else {
                set_reason(local_error, oi, "cylindrical curl has an invalid target direction");
                break;
              }
              const component plus_component =
                  direction_component(source_base, plus_component_direction);
              const component minus_component =
                  direction_component(source_base, minus_component_direction);
              ArrayId expected_plus = f_.array_catalog->find(
                  StorageKey{source.region.chunk, int(array_kind::f), int(plus_component),
                             source.region.cmp, 0});
              ArrayId expected_minus = f_.array_catalog->find(
                  StorageKey{source.region.chunk, int(array_kind::f), int(minus_component),
                             source.region.cmp, 0});
              if (target_direction == R) expected_plus = invalid_array();
              if (target_direction == Z) {
                expected_plus = f_.array_catalog->find(
                    StorageKey{source.region.chunk, int(array_kind::f_rderiv_int), -1, -1, 0});
                expected_minus = invalid_array();
              }
              ptrdiff_t expected_plus_stride = fc.gv.stride(plus_direction);
              ptrdiff_t expected_minus_stride = fc.gv.stride(minus_direction);
              if (is_D(source.region.c)) {
                expected_plus_stride = -expected_plus_stride;
                expected_minus_stride = -expected_minus_stride;
              }
              const bool expected_second = is_valid(expected_plus) && is_valid(expected_minus);
              if (source.plus_source != expected_plus || source.minus_source != expected_minus ||
                  source.plus_stride != expected_plus_stride ||
                  source.minus_stride != expected_minus_stride ||
                  bool(source.region.variant_key & curl_has_second_derivative) != expected_second) {
                set_reason(local_error, oi,
                           "cylindrical curl source identity, stride, or variant is stale");
                break;
              }
            }
            if (cylindrical && target_direction == Z) {
              if (!has_prefix || source.radial_prefix_index >= plan.cylindrical_radial_prefixes.size()) {
                set_reason(local_error, oi,
                           "cylindrical Z curl lacks a valid radial-prefix descriptor");
                break;
              }
              const CylindricalRadialPrefix &prefix =
                  plan.cylindrical_radial_prefixes[source.radial_prefix_index];
              if (prefix.chunk != source.region.chunk ||
                  prefix.target_component != source.region.c || prefix.cmp != source.region.cmp ||
                  source.plus_source != prefix.scratch || is_valid(source.minus_source) ||
                  prefix.source_component !=
                      direction_component(is_D(source.region.c) ? Hx : Ex, P)) {
                set_reason(local_error, oi,
                           "cylindrical Z curl and radial-prefix descriptors disagree");
                break;
              }
              curl.radial_prefix_index = source.radial_prefix_index;
              ++radial_prefix_references[source.radial_prefix_index];
            }
            else {
              if (has_prefix) {
                set_reason(local_error, oi,
                           "non-Z or Cartesian curl has a radial-prefix descriptor");
                break;
              }
              if (cylindrical && target_direction == R && is_valid(source.plus_source)) {
                set_reason(local_error, oi,
                           "cylindrical R curl did not suppress its plus source");
                break;
              }
            }
            const bool has_bfast_bit = (source.region.variant_key & curl_has_bfast) != 0;
            const bool has_bfast_index = source.bfast_update_index != UINT32_MAX;
            if (use_bfast && !has_bfast_index) {
              set_reason(local_error, oi, "live BFAST coordinate has an unpaired curl row");
              break;
            }
            if (has_bfast_bit != has_bfast_index) {
              set_reason(local_error, oi, "curl BFAST bit and paired index disagree");
              break;
            }
            if (has_bfast_index) {
              if (source.bfast_update_index >= plan.bfast_updates.size()) {
                set_reason(local_error, oi, "curl BFAST paired index is out of range");
                break;
              }
              const BfastUpdate &bfast = plan.bfast_updates[source.bfast_update_index];
              bool same_region = source.region.chunk == bfast.region.chunk &&
                                 source.region.c == bfast.region.c &&
                                 source.region.cmp == bfast.region.cmp &&
                                 source.region.base == bfast.region.base;
              bool same_profiles = source.pml.sig == bfast.pml.sig &&
                                   source.pml.kap == bfast.pml.kap &&
                                   source.pml.siginv == bfast.pml.siginv &&
                                   source.pml.base == bfast.pml.base &&
                                   source.pml_u.sig == bfast.pml_u.sig &&
                                   source.pml_u.kap == bfast.pml_u.kap &&
                                   source.pml_u.siginv == bfast.pml_u.siginv &&
                                   source.pml_u.base == bfast.pml_u.base;
              for (int axis = 0; axis < 3; ++axis)
                same_region = same_region &&
                              source.region.begin.yucky_val(axis) ==
                                  bfast.region.begin.yucky_val(axis) &&
                              source.region.end.yucky_val(axis) ==
                                  bfast.region.end.yucky_val(axis) &&
                              source.region.counts[axis] == bfast.region.counts[axis] &&
                              source.region.strides[axis] == bfast.region.strides[axis];
              for (int axis = 0; axis < 3; ++axis)
                same_profiles = same_profiles &&
                                source.pml.strides[axis] == bfast.pml.strides[axis] &&
                                source.pml_u.strides[axis] == bfast.pml_u.strides[axis];
              if (!same_region || !same_profiles || source.target != bfast.target ||
                  source.plus_source != bfast.source1 || source.minus_source != bfast.source2 ||
                  source.plus_stride != bfast.stride1 || source.minus_stride != bfast.stride2 ||
                  source.target_u != bfast.target_u || source.condinv != bfast.condinv ||
                  source.target_cond != bfast.target_cond) {
                set_reason(local_error, oi, "curl and paired BFAST descriptors disagree");
                break;
              }
              curl.bfast_update_index = source.bfast_update_index;
              ++bfast_references[source.bfast_update_index];
            }
            curl_updates.push_back(curl);
          }
          if (!local_error.empty()) break;
          compiled.count = curl_updates.size() - compiled.first;
          if (!compiled.count) set_reason(local_error, oi, "curl descriptor span is empty");
          if (size_t(op.beta_descriptor_index) + op.beta_descriptor_count >
              plan.beta_updates.size()) {
            set_reason(local_error, oi, "beta descriptor span is out of range");
            break;
          }
          compiled.beta_first = beta_updates.size();
          for (size_t i = op.beta_descriptor_index;
               i < size_t(op.beta_descriptor_index) + op.beta_descriptor_count; ++i)
            beta_updates.push_back(compile_beta(plan.beta_updates[i], state));
          compiled.beta_count = beta_updates.size() - compiled.beta_first;
          if (size_t(op.cylindrical_m_descriptor_index) +
                  op.cylindrical_m_descriptor_count >
              plan.cylindrical_m_updates.size()) {
            set_reason(local_error, oi, "cylindrical m/r descriptor span is out of range");
            break;
          }
          compiled.cylindrical_m_first = cylindrical_m_updates.size();
          for (size_t i = op.cylindrical_m_descriptor_index;
               i < size_t(op.cylindrical_m_descriptor_index) +
                       op.cylindrical_m_descriptor_count;
               ++i)
          {
            cylindrical_m_updates.push_back(
                compile_cylindrical_m(plan.cylindrical_m_updates[i], state, f_));
            ++cylindrical_m_references[i];
          }
          compiled.cylindrical_m_count =
              cylindrical_m_updates.size() - compiled.cylindrical_m_first;
          if (size_t(op.cylindrical_origin_action_index) +
                  op.cylindrical_origin_action_count >
              plan.cylindrical_origin_actions.size()) {
            set_reason(local_error, oi, "cylindrical origin-action span is out of range");
            break;
          }
          compiled.cylindrical_origin_first = cylindrical_origin_actions.size();
          for (size_t i = op.cylindrical_origin_action_index;
               i < size_t(op.cylindrical_origin_action_index) +
                       op.cylindrical_origin_action_count;
               ++i) {
            const CylindricalOriginAction &action = plan.cylindrical_origin_actions[i];
            ++cylindrical_origin_references[i];
            NvidiaCompiledCylindricalOriginAction compiled_action = {};
            compiled_action.kind = action.kind;
            if (action.kind == CylindricalOriginActionKind::axis_update) {
              if (action.index >= plan.cylindrical_axis_updates.size()) {
                set_reason(local_error, oi, "cylindrical axis action index is out of range");
                break;
              }
              compiled_action.index = cylindrical_axis_updates.size();
              cylindrical_axis_updates.push_back(
                  compile_cylindrical_axis(plan.cylindrical_axis_updates[action.index], state, f_));
              ++cylindrical_axis_references[action.index];
            }
            else if (action.kind == CylindricalOriginActionKind::zero_slab) {
              if (action.index >= plan.cylindrical_zero_slabs.size()) {
                set_reason(local_error, oi, "cylindrical zero action index is out of range");
                break;
              }
              compiled_action.index = zero_updates.size();
              zero_updates.push_back(compile_zero(plan.cylindrical_zero_slabs[action.index], state));
              ++cylindrical_zero_references[action.index];
            }
            else {
              set_reason(local_error, oi, "cylindrical origin action kind is invalid");
              break;
            }
            cylindrical_origin_actions.push_back(compiled_action);
          }
          if (!local_error.empty()) break;
          compiled.cylindrical_origin_count =
              cylindrical_origin_actions.size() - compiled.cylindrical_origin_first;
          break;
        }
        case OpKind::update_eh: {
          if (size_t(op.descriptor_index) + op.descriptor_count > plan.eh_updates.size()) {
            set_reason(local_error, oi, "constitutive descriptor span is out of range");
            break;
          }
          compiled.first = constitutive_updates.size();
          compiled.copy_first = source_copies.size();
          for (size_t i = op.descriptor_index;
               i < size_t(op.descriptor_index) + op.descriptor_count; ++i) {
            const ConstitutiveUpdate &update = plan.eh_updates[i];
            const bool axis = update.region.variant_key & constitutive_axis_override;
            if (axis) {
              if (i == op.descriptor_index ||
                  (plan.eh_updates[i - 1].region.variant_key & constitutive_axis_override)) {
                set_reason(local_error, oi,
                           "cylindrical constitutive axis replay lacks an adjacent ordinary row");
                break;
              }
              const ConstitutiveUpdate &ordinary = plan.eh_updates[i - 1];
              bool same_profile = update.pml.sig == ordinary.pml.sig &&
                                  update.pml.kap == ordinary.pml.kap &&
                                  update.pml.siginv == ordinary.pml.siginv &&
                                  update.pml.base == ordinary.pml.base;
              for (int axis_index = 0; axis_index < 3; ++axis_index)
                same_profile = same_profile &&
                               update.pml.strides[axis_index] ==
                                   ordinary.pml.strides[axis_index];
              uint32_t expected_variant =
                  ordinary.region.variant_key &
                  ~(constitutive_one_offdiagonal | constitutive_two_offdiagonals |
                    constitutive_has_minus_p | constitutive_copy_w_previous);
              if (ordinary.primary != ordinary.base_primary)
                expected_variant |= constitutive_has_minus_p;
              expected_variant |= constitutive_axis_override;
              if (update.target != ordinary.target ||
                  update.region.chunk != ordinary.region.chunk ||
                  update.region.c != ordinary.region.c ||
                  update.region.cmp != ordinary.region.cmp ||
                  update.region.begin.in_direction(R) != 0 ||
                  update.region.end.in_direction(R) != 0 ||
                  update.base_primary != ordinary.base_primary ||
                  update.primary != ordinary.primary || update.diagonal != ordinary.diagonal ||
                  update.chi2 != ordinary.chi2 || update.chi3 != ordinary.chi3 ||
                  update.target_w != ordinary.target_w ||
                  update.primary_stride != ordinary.primary_stride || !same_profile ||
                  update.region.variant_key != expected_variant ||
                  is_valid(update.base_cross1) || is_valid(update.base_cross2) ||
                  is_valid(update.cross1) || is_valid(update.cross2) ||
                  is_valid(update.offdiagonal1) || is_valid(update.offdiagonal2) ||
                  update.cross1_stride != 0 || update.cross2_stride != 0 ||
                  is_valid(update.previous_w)) {
                set_reason(local_error, oi,
                           "cylindrical constitutive axis replay does not match its ordinary row");
                break;
              }
            }
            constitutive_updates.push_back(compile_constitutive(update, state));
            if (axis) continue;
            const ArrayId targets[] = {update.primary, update.cross1, update.cross2};
            const ArrayId bases[] = {update.base_primary, update.base_cross1, update.base_cross2};
            for (size_t j = 0; j < 3; ++j) {
              if (!is_valid(targets[j]) || targets[j] == bases[j]) continue;
              const void *target_address = device_address(state, targets[j],
                                                          "integrated-source target");
              bool already_copied = false;
              const void *source_address =
                  device_address(state, bases[j], "integrated-source base");
              for (size_t k = compiled.copy_first; k < source_copies.size(); ++k) {
                if (source_copies[k].target != target_address) continue;
                if (source_copies[k].source != source_address)
                  throw std::invalid_argument(
                      "integrated-source target has inconsistent base arrays");
                already_copied = true;
              }
              if (!already_copied)
                source_copies.push_back(compile_source_copy(targets[j], bases[j], state));
            }
          }
          compiled.count = constitutive_updates.size() - compiled.first;
          compiled.copy_count = source_copies.size() - compiled.copy_first;

          if (size_t(op.polarization_subtraction_index) + op.polarization_subtraction_count >
              plan.polarization_subtractions.size()) {
            set_reason(local_error, oi, "polarization subtraction span is out of range");
            break;
          }
          compiled.subtraction_first = polarization_subtractions.size();
          for (size_t i = op.polarization_subtraction_index;
               i < size_t(op.polarization_subtraction_index) +
                       op.polarization_subtraction_count;
               ++i)
            polarization_subtractions.push_back(
                compile_polarization_subtraction(plan.polarization_subtractions[i], state));
          compiled.subtraction_count =
              polarization_subtractions.size() - compiled.subtraction_first;

          if (op.source_descriptor_count) {
            if (!source_plan) {
              set_reason(local_error, oi, "integrated source operation has no prepared plan");
              break;
            }
            if (size_t(op.source_descriptor_index) + op.source_descriptor_count >
                source_plan->sources.size()) {
              set_reason(local_error, oi, "integrated source descriptor span is out of range");
              break;
            }
            compiled.source_first = source_batches.size();
            const field_type source_ft = op.ft == E_stuff ? D_stuff : B_stuff;
            for (size_t i = op.source_descriptor_index;
                 i < size_t(op.source_descriptor_index) + op.source_descriptor_count; ++i) {
              const SourceDescriptor &source = source_plan->sources[i];
              if (!source.integrated || source.ft != source_ft)
                throw std::invalid_argument(
                    "integrated source descriptor span has the wrong field type");
              source_batches.push_back(
                  compile_source_batch(source, *source_plan, f_, state, source_points));
            }
            compiled.source_count = source_batches.size() - compiled.source_first;
          }
          /* Vacuum H==B and E==D aliases legitimately produce no E/H work. */
          break;
        }
        case OpKind::transfer_halo: {
          if (op.ft < 0 || op.ft >= NUM_FIELD_TYPES) {
            set_reason(local_error, oi, "boundary operation has an invalid field type");
            break;
          }
          compiled.first = zero_updates.size();
          const std::vector<ZeroPlan> &zeros = f_.halos->zeros[op.ft];
          for (size_t chunk = 0; chunk < zeros.size(); ++chunk) {
            ZeroPlan canonical;
            std::string why;
            if (!remap_zero_plan(zeros[chunk], f_.halos->arrays, *f_.array_catalog, canonical, why))
              throw std::logic_error(std::string("cannot remap metal-zero plan: ") + why);
            for (size_t i = 0; i < canonical.slabs.size(); ++i)
              zero_updates.push_back(compile_zero(canonical.slabs[i], state));
            for (size_t i = 0; i < canonical.residue.size(); ++i)
              zero_updates.push_back(compile_zero(canonical.residue[i], state));
          }
          compiled.count = zero_updates.size() - compiled.first;

          compiled.halo_first = halo_plans.size();
          size_t buffer_elements = 0;
          size_t operation_scratch_bytes = 0;
          bool have_halo_precision = false;
          nvidia::scalar_precision halo_precision = nvidia::scalar_precision::f64;
          for (size_t i = 0; i < f_.halos->plans.size(); ++i) {
            const HaloPlan &halo = f_.halos->plans[i];
            if (halo.ft != op.ft || !halo.block_elements) continue;
            const NvidiaCompiledHalo lowered =
                compile_halo(halo, f_, state, buffer_elements, halo_gathers, halo_scatters);
            if (have_halo_precision && lowered.gather.precision != halo_precision)
              throw std::invalid_argument(
                  "one NVIDIA boundary operation mixes halo storage precisions");
            halo_precision = lowered.gather.precision;
            have_halo_precision = true;
            halo_plans.push_back(lowered);
            buffer_elements += halo.block_elements;
            operation_scratch_bytes = checked_product(buffer_elements, scalar_bytes(halo_precision),
                                                      "sizing NVIDIA local halo scratch");
          }
          compiled.halo_count = halo_plans.size() - compiled.halo_first;
          halo_scratch_bytes = std::max(halo_scratch_bytes, operation_scratch_bytes);
          break;
        }
        case OpKind::finite_value_check: {
          compiled.first = finite_checks.size();
          uint32_t previous_id = 0;
          bool have_previous = false;
          for (size_t i = 0; i < op.accesses.size(); ++i) {
            const BufferAccess &access = op.accesses[i];
            if (have_previous && access.array.id.value <= previous_id)
              throw std::invalid_argument(
                  "finite-value check accesses are not in stable ArrayId order");
            finite_checks.push_back(compile_finite_check(access, finite_elements, state));
            finite_elements += access.array.elements;
            previous_id = access.array.id.value;
            have_previous = true;
          }
          compiled.count = finite_checks.size() - compiled.first;
          if (!compiled.count) set_reason(local_error, oi, "finite-value check span is empty");
          break;
        }
        case OpKind::apply_sources: {
          if (!source_plan) {
            set_reason(local_error, oi, "source operation has no prepared source plan");
            break;
          }
          if (size_t(op.source_descriptor_index) + op.source_descriptor_count >
              source_plan->sources.size()) {
            set_reason(local_error, oi, "source descriptor span is out of range");
            break;
          }
          compiled.first = source_batches.size();
          for (size_t i = op.source_descriptor_index;
               i < size_t(op.source_descriptor_index) + op.source_descriptor_count; ++i) {
            if (source_plan->sources[i].ft != op.ft)
              throw std::invalid_argument("source descriptor span has the wrong field type");
            const SourceDescriptor &source = source_plan->sources[i];
            source_batches.push_back(
                compile_source_batch(source, *source_plan, f_, state, source_points));
          }
          compiled.count = source_batches.size() - compiled.first;
          break;
        }
        case OpKind::evaluate_source_scalars: {
          if (!source_plan) {
            set_reason(local_error, oi, "source evaluation has no prepared source plan");
            break;
          }
          if (op.descriptor_index != 0 || op.descriptor_count != source_plan->source_times.size()) {
            set_reason(local_error, oi, "source-time descriptor span is not complete");
            break;
          }
          if (op.source_time_offset != 0.0 && op.source_time_offset != 0.5 &&
              op.source_time_offset != 1.0) {
            set_reason(local_error, oi, "source-time offset is invalid");
            break;
          }
          compiled.count = source_plan->scalars.size();
          compiled.source_staging_offset = source_staging_elements;
          if (compiled.count > std::numeric_limits<size_t>::max() - source_staging_elements)
            throw std::overflow_error("NVIDIA source-scalar staging size overflow");
          source_staging_elements += compiled.count;
          break;
        }
        case OpKind::update_dft: {
          if (size_t(op.descriptor_index) + op.descriptor_count > plan.dft_updates.size()) {
            set_reason(local_error, oi, "DFT descriptor span is out of range");
            break;
          }
          compiled.first = dft_updates.size();
          for (size_t i = op.descriptor_index;
               i < size_t(op.descriptor_index) + op.descriptor_count; ++i)
            dft_updates.push_back(compile_dft(plan.dft_updates[i], f_, state, dft_omega));
          compiled.count = dft_updates.size() - compiled.first;
          break;
        }
        case OpKind::update_polarization: {
          if (size_t(op.descriptor_index) + op.descriptor_count >
              plan.polarization_updates.size()) {
            set_reason(local_error, oi, "polarization update span is out of range");
            break;
          }
          compiled.polarization_first = polarization_updates.size();
          for (size_t i = op.descriptor_index;
               i < size_t(op.descriptor_index) + op.descriptor_count; ++i)
            polarization_updates.push_back(
                compile_polarization_update(plan.polarization_updates[i], state));
          compiled.polarization_count =
              polarization_updates.size() - compiled.polarization_first;
          break;
        }
        case OpKind::restore_magnetic_fields:
        case OpKind::synchronize_magnetic_fields:
          if (op.magnetic_state_index != 0 ||
              op.magnetic_state_count != plan.magnetic_state_arrays.size())
            set_reason(local_error, oi, "magnetic snapshot span is incomplete");
          break;
        case OpKind::update_material_coefficients:
        case OpKind::increment_time:
          break;

        case OpKind::phase_material:
        case OpKind::zero_boundary:
        case OpKind::pack_halo:
        case OpKind::exchange_local:
        case OpKind::unpack_halo:
        case OpKind::update_flux_half:
        case OpKind::update_flux:
        case OpKind::reduction:
        case OpKind::host_callback:
        case OpKind::pack_state:
        case OpKind::unpack_state:
        case OpKind::num_kinds: set_reason(local_error, oi, op_kind_name(op.kind)); break;
      }
      if (!local_error.empty()) break;
      operations.push_back(compiled);
    }

    if (local_error.empty())
      for (size_t i = 0; i < bfast_references.size(); ++i)
        if (bfast_references[i] != 1) {
          local_error = "NVIDIA BFAST descriptor is not paired with exactly one curl";
          break;
        }

    const auto require_single_reference = [&](const std::vector<unsigned int> &references,
                                              const char *what) {
      if (!local_error.empty()) return;
      for (size_t i = 0; i < references.size(); ++i)
        if (references[i] != 1) {
          local_error = std::string("NVIDIA ") + what +
                        " descriptor is not referenced exactly once";
          return;
        }
    };
    require_single_reference(radial_prefix_references, "cylindrical radial-prefix");
    require_single_reference(cylindrical_m_references, "cylindrical m/r");
    require_single_reference(cylindrical_axis_references, "cylindrical axis");
    require_single_reference(cylindrical_zero_references, "cylindrical zero-slab");
    require_single_reference(cylindrical_origin_references, "cylindrical origin-action");

    /* Reference counts prove that descriptors which survived into the submitted
       plan are wired exactly once.  They cannot prove that a re-signed plan did
       not remove both a descriptor and its operation span.  Cylindrical update
       construction is deterministic from the live fields state, so compare the
       complete canonical content signature after the detailed diagnostics above
       have had a chance to identify malformed individual operands. */
    if (local_error.empty() && canonical.signature != plan.signature)
      local_error = "NVIDIA timestep plan is incomplete or non-canonical";

    if (local_error.empty())
      executable.reset(new NvidiaExecutable(
          this, plan.signature, state.fingerprint_, state.state_token_, operations, curl_updates,
          cylindrical_radial_prefixes, bfast_updates, beta_updates, cylindrical_m_updates,
          cylindrical_axis_updates, cylindrical_origin_actions, constitutive_updates, zero_updates,
          halo_plans, halo_gathers, halo_scatters, halo_scratch_bytes, finite_checks,
          source_batches, source_points, source_copies, polarization_updates,
          polarization_subtractions, dft_updates, dft_omega,
          magnetic_states, magnetic_snapshot_bytes,
          magnetic_layout_fingerprint(state.fingerprint_, plan.magnetic_state_arrays),
          plan.magnetic_half_step,
          source_plan ? source_plan->scalars.size() : 0,
          source_staging_elements, state));
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown NVIDIA PR2 compilation failure";
  }

  if (or_to_all(!local_error.empty())) {
    executable.reset();
    if (local_error.empty()) local_error = "NVIDIA PR2 compilation was rejected on another rank";
    throw std::runtime_error(local_error);
  }
  return executable.release();
}

void NvidiaBackend::preflight_magnetic_transition(Executable &raw_executable,
                                                  BackendState &raw_state, bool synchronize) {
  NvidiaBackendState &state = checked_state(raw_state);
  NvidiaExecutable &executable = checked_executable(raw_executable, state);
  if (!state.initialized_) throw std::logic_error("NVIDIA magnetic transition has no initialized state");
  if (state.transfer_failed_)
    throw std::logic_error("NVIDIA magnetic transition stream is unavailable");
  if (executable.storage_fingerprint_ != state.fingerprint_)
    throw std::logic_error("NVIDIA magnetic executable uses a stale storage layout");
  if (synchronize) {
    if (state.magnetic_snapshot_valid_)
      throw std::logic_error("NVIDIA magnetic snapshot is already live");
    if (state.magnetic_snapshot_.size() < executable.magnetic_snapshot_bytes_)
      state.magnetic_snapshot_.allocate(executable.magnetic_snapshot_bytes_, state.device_);
  }
  else {
    if (!state.magnetic_snapshot_valid_)
      throw std::logic_error("NVIDIA magnetic restore has no live snapshot");
    if (state.magnetic_snapshot_layout_fingerprint_ !=
            executable.magnetic_layout_fingerprint_ ||
        state.magnetic_snapshot_bytes_ != executable.magnetic_snapshot_bytes_)
      throw std::logic_error("NVIDIA magnetic snapshot layout is stale");
  }
}

void NvidiaBackend::execute_magnetic_half_step(NvidiaExecutable &executable,
                                               NvidiaBackendState &state) {
  /* The ordinary executor owns the same pinned source-staging ranges.  A
     synchronization transition can follow an asynchronously queued step, so
     finish those copies before rewriting the host staging memory. */
  state.transfer_->synchronize();
  f_.step_source_times[0] = f_.time();
  f_.step_source_times[1] = std::fma(double(f_.t), f_.dt, 0.5 * f_.dt);
  f_.step_source_times[2] = std::fma(double(f_.t), f_.dt, f_.dt);
  const uint32_t schedule[] = {
      executable.magnetic_half_step_.evaluate_b_sources,
      executable.magnetic_half_step_.update_b,
      executable.magnetic_half_step_.apply_b_sources,
      executable.magnetic_half_step_.transfer_b,
      executable.magnetic_half_step_.evaluate_h_sources,
      executable.magnetic_half_step_.update_h,
      executable.magnetic_half_step_.transfer_h};
  for (int slot = 0; slot < 7; ++slot) {
    if (schedule[slot] == UINT32_MAX) continue;
    if (schedule[slot] >= executable.operations_.size())
      throw std::logic_error("NVIDIA magnetic half-step operation is out of range");
    const NvidiaCompiledOperation &op = executable.operations_[schedule[slot]];
    switch (op.kind) {
      case OpKind::evaluate_source_scalars: {
        evaluate_supported_source_scalars(f_, op.source_time_offset);
        const SourcePlan &source_plan = f_.descriptors->sources;
        if (source_plan.scalars.size() != executable.source_scalar_count_ ||
            op.count != executable.source_scalar_count_)
          throw std::logic_error("NVIDIA magnetic source scalar block changed after compilation");
        nvidia::source_scalar *staging =
            static_cast<nvidia::source_scalar *>(executable.source_staging_.data()) +
            op.source_staging_offset;
        for (size_t i = 0; i < op.count; ++i) {
          staging[i].current_real = source_plan.scalars[i].current.real();
          staging[i].current_imag = source_plan.scalars[i].current.imag();
          staging[i].dipole_real = source_plan.scalars[i].dipole.real();
          staging[i].dipole_imag = source_plan.scalars[i].dipole.imag();
        }
        nvidia::copy_host_to_device_async(
            executable.source_scalars_, 0, staging,
            checked_product(op.count, sizeof(nvidia::source_scalar),
                            "uploading NVIDIA magnetic source scalars"),
            *state.transfer_);
        break;
      }
      case OpKind::update_db:
        for (size_t i = op.first; i < op.first + op.count; ++i) {
          const uint32_t prefix = executable.curl_updates_[i].radial_prefix_index;
          if (prefix != UINT32_MAX)
            nvidia::launch_cylindrical_radial_prefix(
                executable.cylindrical_radial_prefixes_[prefix], *state.transfer_);
          nvidia::launch_curl(executable.curl_updates_[i], *state.transfer_);
          const uint32_t bfast = executable.curl_updates_[i].bfast_update_index;
          if (bfast != UINT32_MAX)
            nvidia::launch_bfast(executable.bfast_updates_[bfast], *state.transfer_);
        }
        for (size_t i = op.beta_first; i < op.beta_first + op.beta_count; ++i)
          nvidia::launch_beta(executable.beta_updates_[i], *state.transfer_);
        for (size_t i = op.cylindrical_m_first;
             i < op.cylindrical_m_first + op.cylindrical_m_count; ++i)
          nvidia::launch_cylindrical_m(executable.cylindrical_m_updates_[i], *state.transfer_);
        for (size_t i = op.cylindrical_origin_first;
             i < op.cylindrical_origin_first + op.cylindrical_origin_count; ++i) {
          const NvidiaCompiledCylindricalOriginAction &action =
              executable.cylindrical_origin_actions_[i];
          if (action.kind == CylindricalOriginActionKind::axis_update)
            nvidia::launch_cylindrical_axis(executable.cylindrical_axis_updates_[action.index],
                                            *state.transfer_);
          else
            nvidia::launch_zero(executable.zero_updates_[action.index], *state.transfer_);
        }
        break;
      case OpKind::apply_sources:
        for (size_t i = op.first; i < op.first + op.count; ++i)
          nvidia::launch_source_batch(executable.source_batches_[i],
                                      executable.source_scalars_.opaque_handle(),
                                      *state.transfer_);
        break;
      case OpKind::update_eh:
        for (size_t i = op.copy_first; i < op.copy_first + op.copy_count; ++i)
          nvidia::launch_array_copy(executable.source_copies_[i], *state.transfer_);
        for (size_t i = op.subtraction_first; i < op.subtraction_first + op.subtraction_count; ++i)
          nvidia::launch_polarization_subtract(executable.polarization_subtractions_[i],
                                               *state.transfer_);
        for (size_t i = op.source_first; i < op.source_first + op.source_count; ++i)
          nvidia::launch_source_batch(executable.source_batches_[i],
                                      executable.source_scalars_.opaque_handle(),
                                      *state.transfer_);
        for (size_t i = op.first; i < op.first + op.count; ++i)
          nvidia::launch_constitutive(executable.constitutive_updates_[i], *state.transfer_);
        break;
      case OpKind::transfer_halo:
        for (size_t i = op.first; i < op.first + op.count; ++i)
          nvidia::launch_zero(executable.zero_updates_[i], *state.transfer_);
        for (size_t i = op.halo_first; i < op.halo_first + op.halo_count; ++i)
          nvidia::launch_halo_gather(executable.halo_plans_[i].gather,
                                     executable.halo_gathers_.opaque_handle(),
                                     executable.halo_scratch_.opaque_handle(), *state.transfer_);
        for (size_t i = op.halo_first; i < op.halo_first + op.halo_count; ++i)
          nvidia::launch_halo_scatter(executable.halo_plans_[i].scatter,
                                      executable.halo_scatters_.opaque_handle(),
                                      executable.halo_scratch_.opaque_handle(), *state.transfer_);
        break;
      default: throw std::logic_error("NVIDIA magnetic half-step contains an invalid operation");
    }
  }
}

void NvidiaBackend::synchronize_magnetic_fields(Executable &raw_executable,
                                                BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  NvidiaExecutable &executable = checked_executable(raw_executable, state);
  nvidia::device_scope scope(state.device_);
  char *backup = static_cast<char *>(state.magnetic_snapshot_.opaque_handle());
  for (const NvidiaCompiledMagneticState &row : executable.magnetic_states_)
    nvidia::launch_magnetic_backup(
        nvidia::magnetic_state_launch{row.live, backup + row.backup_offset, row.elements,
                                      row.precision},
        *state.transfer_);
  execute_magnetic_half_step(executable, state);
  for (const NvidiaCompiledMagneticState &row : executable.magnetic_states_)
    if (row.average)
      nvidia::launch_magnetic_average(
          nvidia::magnetic_state_launch{row.live, backup + row.backup_offset, row.elements,
                                        row.precision},
          *state.transfer_);
  state.transfer_->synchronize();
  state.magnetic_snapshot_bytes_ = executable.magnetic_snapshot_bytes_;
  state.magnetic_snapshot_layout_fingerprint_ = executable.magnetic_layout_fingerprint_;
  state.magnetic_snapshot_valid_ = true;
  state.device_authoritative_ = true;
}

void NvidiaBackend::restore_magnetic_fields(Executable &raw_executable,
                                            BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  NvidiaExecutable &executable = checked_executable(raw_executable, state);
  nvidia::device_scope scope(state.device_);
  char *backup = static_cast<char *>(state.magnetic_snapshot_.opaque_handle());
  for (const NvidiaCompiledMagneticState &row : executable.magnetic_states_)
    nvidia::launch_magnetic_restore(
        nvidia::magnetic_state_launch{row.live, backup + row.backup_offset, row.elements,
                                      row.precision},
        *state.transfer_);
  state.transfer_->synchronize();
  state.magnetic_snapshot_valid_ = false;
  state.device_authoritative_ = true;
}

void NvidiaBackend::advance(Executable &raw_executable, BackendState &raw_state, int num_steps) {
  NvidiaBackendState &state = checked_state(raw_state);
  NvidiaExecutable &executable = checked_executable(raw_executable, state);
  if (!or_to_all(num_steps > 0)) return;
  if (!state.initialized_) throw std::logic_error("cannot advance uninitialized NVIDIA storage");
  if (executable.storage_fingerprint_ != state.fingerprint_)
    throw std::logic_error("NVIDIA executable was compiled for a different storage layout");
  if (state.transfer_failed_)
    throw std::logic_error("NVIDIA execution stream failed; recreate backend state");

  const FiniteCheckMode finite_mode = finite_check_mode();
  try {
    nvidia::device_scope scope(state.device_);
    for (int step = 0; step < num_steps; ++step) {
      f_.step_source_times[0] = f_.time();
      f_.step_source_times[1] = std::fma(double(f_.t), f_.dt, 0.5 * f_.dt);
      f_.step_source_times[2] = std::fma(double(f_.t), f_.dt, f_.dt);
      for (size_t oi = 0; oi < executable.operations_.size(); ++oi) {
        const NvidiaCompiledOperation &op = executable.operations_[oi];
        switch (op.kind) {
          case OpKind::restore_magnetic_fields:
            if (f_.synchronized_magnetic_fields) {
              preflight_magnetic_transition(executable, state, false);
              restore_magnetic_fields(executable, state);
            }
            break;
          case OpKind::update_db:
            for (size_t i = op.first; i < op.first + op.count; ++i) {
              const uint32_t prefix = executable.curl_updates_[i].radial_prefix_index;
              if (prefix != UINT32_MAX)
                nvidia::launch_cylindrical_radial_prefix(
                    executable.cylindrical_radial_prefixes_[prefix], *state.transfer_);
              nvidia::launch_curl(executable.curl_updates_[i], *state.transfer_);
              const uint32_t bfast = executable.curl_updates_[i].bfast_update_index;
              if (bfast != UINT32_MAX)
                nvidia::launch_bfast(executable.bfast_updates_[bfast], *state.transfer_);
            }
            for (size_t i = op.beta_first; i < op.beta_first + op.beta_count; ++i)
              nvidia::launch_beta(executable.beta_updates_[i], *state.transfer_);
            for (size_t i = op.cylindrical_m_first;
                 i < op.cylindrical_m_first + op.cylindrical_m_count; ++i)
              nvidia::launch_cylindrical_m(executable.cylindrical_m_updates_[i],
                                           *state.transfer_);
            for (size_t i = op.cylindrical_origin_first;
                 i < op.cylindrical_origin_first + op.cylindrical_origin_count; ++i) {
              const NvidiaCompiledCylindricalOriginAction &action =
                  executable.cylindrical_origin_actions_[i];
              if (action.kind == CylindricalOriginActionKind::axis_update)
                nvidia::launch_cylindrical_axis(
                    executable.cylindrical_axis_updates_[action.index], *state.transfer_);
              else
                nvidia::launch_zero(executable.zero_updates_[action.index], *state.transfer_);
            }
            break;
          case OpKind::update_eh:
            for (size_t i = op.copy_first; i < op.copy_first + op.copy_count; ++i)
              nvidia::launch_array_copy(executable.source_copies_[i], *state.transfer_);
            for (size_t i = op.subtraction_first; i < op.subtraction_first + op.subtraction_count;
                 ++i)
              nvidia::launch_polarization_subtract(executable.polarization_subtractions_[i],
                                                   *state.transfer_);
            for (size_t i = op.source_first; i < op.source_first + op.source_count; ++i)
              nvidia::launch_source_batch(executable.source_batches_[i],
                                          executable.source_scalars_.opaque_handle(),
                                          *state.transfer_);
            for (size_t i = op.first; i < op.first + op.count; ++i)
              nvidia::launch_constitutive(executable.constitutive_updates_[i], *state.transfer_);
            break;
          case OpKind::update_polarization:
            for (size_t i = op.polarization_first;
                 i < op.polarization_first + op.polarization_count; ++i)
              nvidia::launch_polarization_update(executable.polarization_updates_[i],
                                                 *state.transfer_);
            break;
          case OpKind::transfer_halo:
            for (size_t i = op.first; i < op.first + op.count; ++i)
              nvidia::launch_zero(executable.zero_updates_[i], *state.transfer_);
            /* All source values must be captured before any destination is
               overwritten: local chunk boundaries can alias another plan's
               gather side. Stream ordering supplies the gather/scatter
               barrier without a host synchronization. */
            for (size_t i = op.halo_first; i < op.halo_first + op.halo_count; ++i)
              nvidia::launch_halo_gather(
                  executable.halo_plans_[i].gather, executable.halo_gathers_.opaque_handle(),
                  executable.halo_scratch_.opaque_handle(), *state.transfer_);
            for (size_t i = op.halo_first; i < op.halo_first + op.halo_count; ++i)
              nvidia::launch_halo_scatter(
                  executable.halo_plans_[i].scatter, executable.halo_scatters_.opaque_handle(),
                  executable.halo_scratch_.opaque_handle(), *state.transfer_);
            break;
          case OpKind::evaluate_source_scalars: {
            evaluate_supported_source_scalars(f_, op.source_time_offset);
            const SourcePlan &source_plan = f_.descriptors->sources;
            if (source_plan.scalars.size() != executable.source_scalar_count_ ||
                op.count != executable.source_scalar_count_)
              throw std::logic_error("NVIDIA source scalar block changed after compilation");
            nvidia::source_scalar *staging =
                static_cast<nvidia::source_scalar *>(executable.source_staging_.data()) +
                op.source_staging_offset;
            for (size_t i = 0; i < op.count; ++i) {
              staging[i].current_real = source_plan.scalars[i].current.real();
              staging[i].current_imag = source_plan.scalars[i].current.imag();
              staging[i].dipole_real = source_plan.scalars[i].dipole.real();
              staging[i].dipole_imag = source_plan.scalars[i].dipole.imag();
            }
            nvidia::copy_host_to_device_async(executable.source_scalars_, 0, staging,
                                              checked_product(op.count,
                                                              sizeof(nvidia::source_scalar),
                                                              "uploading NVIDIA source scalars"),
                                              *state.transfer_);
            break;
          }
          case OpKind::apply_sources:
            for (size_t i = op.first; i < op.first + op.count; ++i)
              nvidia::launch_source_batch(executable.source_batches_[i],
                                          executable.source_scalars_.opaque_handle(),
                                          *state.transfer_);
            break;
          case OpKind::increment_time: ++f_.t; break;
          case OpKind::synchronize_magnetic_fields:
            if (f_.synchronized_magnetic_fields) {
              preflight_magnetic_transition(executable, state, true);
              synchronize_magnetic_fields(executable, state);
            }
            break;
          case OpKind::update_dft:
            for (size_t i = op.first; i < op.first + op.count; ++i) {
              const nvidia::dft_launch &dft = executable.dft_updates_[i];
              if ((f_.t % dft.decimation_factor) != 0) continue;
              const double sample_time =
                  dft.magnetic ? f_.time() - 0.5 * f_.dt : f_.time();
              nvidia::launch_dft(dft, sample_time, *state.transfer_);
            }
            break;
          case OpKind::finite_value_check: {
            const bool due = finite_mode == FiniteCheckMode::step ||
                             (finite_mode == FiniteCheckMode::batch && step + 1 == num_steps);
            if (!due) break;
            nvidia::fill_byte_async(executable.finite_result_, 0, 0xff, sizeof(uint64_t),
                                    *state.transfer_);
            for (size_t i = op.first; i < op.first + op.count; ++i)
              nvidia::launch_finite_check(executable.finite_checks_[i].launch,
                                          executable.finite_result_.opaque_handle(),
                                          *state.transfer_);
            nvidia::copy_device_to_host_async(executable.finite_result_host_.data(),
                                              executable.finite_result_, 0, sizeof(uint64_t),
                                              *state.transfer_);
            state.transfer_->synchronize();

            uint64_t first_bad = std::numeric_limits<uint64_t>::max();
            memcpy(&first_bad, executable.finite_result_host_.data(), sizeof(first_bad));
            if (first_bad == std::numeric_limits<uint64_t>::max()) break;

            const NvidiaFiniteCheck *found = NULL;
            size_t element = 0;
            for (size_t i = op.first; i < op.first + op.count; ++i) {
              const NvidiaFiniteCheck &scan = executable.finite_checks_[i];
              if (first_bad >= scan.launch.ordinal_base &&
                  first_bad - scan.launch.ordinal_base < scan.launch.elements) {
                found = &scan;
                element = size_t(first_bad - scan.launch.ordinal_base);
                break;
              }
            }
            if (!found)
              throw std::logic_error("NVIDIA finite-value diagnostic returned an invalid ordinal");
            if (!f_.nonfinite_flag) {
              f_.nonfinite_flag = 1;
              f_.first_bad_step = f_.t;
              f_.first_bad_component = found->key.component_;
            }
            meep::abort(
                "simulation fields are NaN or Inf (chunk %d, component %d, cmp %d, element %zu)",
                found->key.chunk, found->key.component_, found->key.cmp, element);
          }
          default: break; // capability validation proved these operations are no-ops
        }
      }
      state.transfer_->synchronize();
    }
  }
  catch (...) {
    state.transfer_failed_ = true;
    poison();
    throw;
  }
  state.device_authoritative_ = true;
}

void NvidiaBackend::read(ArrayRef ref, void *host_buffer, size_t bytes) {
  NvidiaBackendState *current = current_state();
  if (!current) throw std::logic_error("NVIDIA backend has no active state");
  NvidiaBackendState &state = *current;
  if (!state.initialized_) throw std::logic_error("NVIDIA backend storage is not initialized");
  if (state.transfer_failed_)
    throw std::logic_error("NVIDIA transfer stream failed; recreate backend state");
  const AccessRange access = checked_access(state.plan_, ref, host_buffer, bytes);
  if (!bytes) return;
  state.ensure_staging(access.storage_bytes);
  try {
    state.arenas_->copy_to_host_async(state.staging_.data(), ref.id.value, access.storage_offset,
                                      access.storage_bytes, *state.transfer_);
    state.transfer_->synchronize();
  }
  catch (...) {
    state.transfer_failed_ = true;
    throw;
  }
  storage_to_host(host_buffer, state.staging_.data(), *access.spec, ref.elements);
}

void NvidiaBackend::write(ArrayRef ref, const void *host_buffer, size_t bytes) {
  NvidiaBackendState *current = current_state();
  if (!current) throw std::logic_error("NVIDIA backend has no active state");
  NvidiaBackendState &state = *current;
  if (!state.initialized_) throw std::logic_error("NVIDIA backend storage is not initialized");
  if (state.transfer_failed_)
    throw std::logic_error("NVIDIA transfer stream failed; recreate backend state");
  const AccessRange access = checked_access(state.plan_, ref, host_buffer, bytes);
  if (!bytes) return;
  state.ensure_staging(access.storage_bytes);
  host_to_storage(state.staging_.data(), host_buffer, *access.spec, ref.elements);
  try {
    state.arenas_->copy_from_host_async(ref.id.value, access.storage_offset, state.staging_.data(),
                                        access.storage_bytes, *state.transfer_);
    /* Writes are complete on return, matching the CPU backend and making reuse
       of the single pinned staging allocation unambiguous. */
    state.transfer_->synchronize();

    /* Keep the explicitly written host range coherent. Other host arrays may
       remain stale after stepping; prepare_state_rebuild migrates the complete
       device-authoritative state before a storage rebuild. */
    char *host = static_cast<char *>(f_.array_catalog->resolve_untyped(ref.id));
    if (!host) throw std::logic_error("NVIDIA backend host mirror is null");
    const size_t host_offset =
        checked_product(ref.offset, host_element_bytes(access.spec->element_type),
                        "resolving NVIDIA host mirror offset");
    storage_to_host(host + host_offset, state.staging_.data(), *access.spec, ref.elements);
  }
  catch (...) {
    state.transfer_failed_ = true;
    throw;
  }
}

void NvidiaBackend::reduce_dft(const DftReductionRequest &request,
                               std::complex<double> *local_result, size_t result_count) {
  if (!active_state_) throw std::logic_error("NVIDIA backend has no active state");
  NvidiaBackendState &state = *active_state_;
  if (!state.initialized_) throw std::logic_error("NVIDIA backend storage is not initialized");
  if (state.transfer_failed_)
    throw std::logic_error("NVIDIA transfer stream failed; recreate backend state");
  if (!local_result && result_count)
    throw std::invalid_argument("NVIDIA DFT reduction has no result buffer");
  if (request.result_count != result_count || !result_count)
    throw std::invalid_argument("NVIDIA DFT reduction result-count mismatch");
  if ((request.kind == DftReductionKind::norm2 && result_count != 1) ||
      (request.kind != DftReductionKind::norm2 && !result_count))
    throw std::invalid_argument("NVIDIA DFT reduction kind/result-count mismatch");
  switch (request.kind) {
    case DftReductionKind::norm2:
    case DftReductionKind::real_weighted_product:
    case DftReductionKind::complex_weighted_product: break;
    default: throw std::invalid_argument("NVIDIA DFT reduction kind is invalid");
  }
  if (request.accumulation_precision != policy_for(options_.precision).reduction)
    throw std::invalid_argument("NVIDIA DFT reduction precision violates backend policy");

  std::vector<size_t> selected_points(request.terms.size());
  std::vector<size_t> blocks_per_lane(request.terms.size());
  size_t maximum_blocks = 1;
  for (size_t i = 0; i < request.terms.size(); ++i) {
    const DftReductionTerm &term = request.terms[i];
    const ArraySpec &left = validate_dft_reduction_array(
        state.plan_, term.left, term.storage_points, term.frequencies, "DFT reduction left array");
    selected_points[i] = validate_dft_reduction_region(term);
    if (request.kind == DftReductionKind::norm2) {
      if (is_valid(term.right))
        throw std::invalid_argument("NVIDIA DFT norm reduction has a right operand");
    }
    else {
      if (term.frequencies != result_count)
        throw std::invalid_argument("NVIDIA DFT product frequency/result mismatch");
      const ArraySpec &right = validate_dft_reduction_array(state.plan_, term.right,
                                                            term.storage_points, term.frequencies,
                                                            "DFT reduction right array");
      if (right.storage != left.storage)
        throw std::invalid_argument("NVIDIA DFT pair has different monitor precision");
    }
    if (request.accumulation_precision == Precision::f32 && left.storage != Precision::f32)
      throw std::invalid_argument("NVIDIA DFT f64 monitor storage cannot reduce in f32");
    size_t work = selected_points[i];
    if (request.kind == DftReductionKind::norm2)
      work = checked_product(work, term.frequencies, "sizing NVIDIA DFT norm reduction");
    const size_t needed = (work + 255) / 256;
    blocks_per_lane[i] = std::min<size_t>(128, std::max<size_t>(1, needed));
    maximum_blocks = std::max(maximum_blocks, blocks_per_lane[i]);
  }

  const size_t result_bytes =
      checked_product(result_count, sizeof(double) * 2, "sizing NVIDIA DFT reduction result");
  const size_t partial_values = checked_product(result_count, maximum_blocks,
                                                "sizing NVIDIA DFT reduction partial count");
  const size_t partial_bytes = checked_product(
      partial_values, sizeof(double) * 2, "sizing NVIDIA DFT reduction partial storage");
  state.ensure_dft_reduction_buffers(result_bytes, partial_bytes);

  try {
    nvidia::fill_byte_async(state.dft_reduction_result_, 0, 0, result_bytes, *state.transfer_);
    for (size_t i = 0; i < request.terms.size(); ++i) {
      const DftReductionTerm &term = request.terms[i];
      nvidia::dft_reduction_launch launch;
      launch.left = complex_device_address(state, term.left, "DFT reduction left array");
      launch.right = request.kind == DftReductionKind::norm2
                         ? NULL
                         : complex_device_address(state, term.right, "DFT reduction right array");
      launch.partials = state.dft_reduction_partials_.opaque_handle();
      launch.result = state.dft_reduction_result_.opaque_handle();
      launch.storage_points = term.storage_points;
      launch.frequencies = term.frequencies;
      launch.base = term.region.base;
      for (int axis = 0; axis < 3; ++axis) {
        launch.counts[axis] = term.region.counts[axis];
        launch.strides[axis] = term.region.strides[axis];
      }
      launch.result_count = result_count;
      launch.blocks_per_lane = blocks_per_lane[i];
      launch.weight_real = term.weight.real();
      launch.weight_imag = term.weight.imag();
      launch.operation = request.kind == DftReductionKind::norm2
                             ? nvidia::dft_reduction_operation::norm2
                             : request.kind == DftReductionKind::real_weighted_product
                                   ? nvidia::dft_reduction_operation::real_weighted_product
                                   : nvidia::dft_reduction_operation::complex_weighted_product;
      launch.monitor_precision = complex_precision_for(state.plan_, term.left,
                                                       "DFT reduction monitor");
      launch.accumulation_precision = scalar_precision_for(request.accumulation_precision);
      nvidia::launch_dft_reduction(launch, *state.transfer_);
    }
    nvidia::copy_device_to_host_async(state.staging_.data(), state.dft_reduction_result_, 0,
                                      result_bytes, *state.transfer_);
    state.transfer_->synchronize();
  }
  catch (...) {
    state.transfer_failed_ = true;
    throw;
  }

  const double *values = static_cast<const double *>(state.staging_.data());
  for (size_t i = 0; i < result_count; ++i)
    local_result[i] = std::complex<double>(values[2 * i], values[2 * i + 1]);
}

void NvidiaBackend::synchronize() {
  NvidiaBackendState *state = current_state();
  if (!state) return;
  if (state->transfer_failed_)
    throw std::logic_error("NVIDIA transfer stream failed; recreate backend state");
  try {
    state->transfer_->synchronize();
  }
  catch (...) {
    state->transfer_failed_ = true;
    throw;
  }
}

void NvidiaBackend::prepare_state_rebuild(BackendState &raw_state, DirtyMask) {
  NvidiaBackendState &state = checked_state(raw_state);
  if (state.transfer_failed_)
    throw std::logic_error("NVIDIA transfer stream failed; recreate backend state");
  try {
    state.transfer_->synchronize();
  }
  catch (...) {
    state.transfer_failed_ = true;
    throw;
  }
  if (!state.device_authoritative_) return;
  if (!f_.array_catalog || f_.array_catalog->size() != state.plan_.arrays.size())
    throw std::logic_error("cannot migrate NVIDIA state into a changed host catalog");

  try {
    for (size_t i = 0; i < state.plan_.arrays.size(); ++i) {
      const ArraySpec &spec = state.plan_.arrays[i];
      if (is_valid(spec.alias_of)) continue;
      const size_t bytes = storage_bytes(spec);
      state.ensure_staging(bytes);
      state.arenas_->copy_to_host_async(state.staging_.data(), spec.id.value, 0, bytes,
                                        *state.transfer_);
      state.transfer_->synchronize();
      void *destination = f_.array_catalog->resolve_untyped(spec.id);
      if (!destination) throw std::logic_error("NVIDIA migration found a null host allocation");
      storage_to_host(destination, state.staging_.data(), spec, spec.elements);
    }
  }
  catch (...) {
    state.transfer_failed_ = true;
    throw;
  }
  state.device_authoritative_ = false;
}

backend_capabilities NvidiaBackend::capabilities() const {
  backend_capabilities result;
  result.supports_native = true;
  result.supports_mixed = true;
  result.supports_f32 = true;
  result.memory_budget_bytes = device_memory_bytes_;
  result.name = "nvidia";
  return result;
}

bool NvidiaBackend::accepts(const execution_options &options, std::string &why) const {
  why.clear();
  if (options.backend == backend_kind::cpu) {
    why = "the NVIDIA backend cannot satisfy an explicit cpu request";
    return false;
  }
  switch (options.precision) {
    case precision_policy_kind::native:
    case precision_policy_kind::mixed:
    case precision_policy_kind::f32: break;
    default: why = "the NVIDIA backend received an invalid precision policy"; return false;
  }
  if (options.device_id != automatic_device && options.device_id != device_) {
    why = "the resolved NVIDIA device does not match the explicit device request";
    return false;
  }
  return true;
}

NvidiaBackendState &NvidiaBackend::checked_state(BackendState &raw_state) const {
  NvidiaBackendState *state = dynamic_cast<NvidiaBackendState *>(&raw_state);
  if (!state || state->owner_ != this)
    throw std::invalid_argument("NVIDIA backend received state owned by another backend");
  return *state;
}

NvidiaBackendState *NvidiaBackend::current_state() const {
  NvidiaBackendState *state = dynamic_cast<NvidiaBackendState *>(f_.backend_state);
  return state && state->owner_ == this ? state : NULL;
}

NvidiaExecutable &NvidiaBackend::checked_executable(Executable &raw_executable,
                                                    const NvidiaBackendState &state) const {
  NvidiaExecutable *executable = dynamic_cast<NvidiaExecutable *>(&raw_executable);
  if (!executable || executable->owner_ != this || executable->state_token_ != state.state_token_)
    throw std::invalid_argument("NVIDIA backend received an executable of the wrong type");
  return *executable;
}

ExecutionBackend *make_nvidia_backend(fields &f, const execution_options &options,
                                      int selected_device, std::string &why) {
  try {
    std::unique_ptr<NvidiaBackend> backend(new NvidiaBackend(f, options, selected_device));
    if (!backend->accepts(options, why)) return NULL;
    return backend.release();
  }
  catch (const std::exception &error) {
    why = error.what();
    return NULL;
  }
}

} // namespace meep
