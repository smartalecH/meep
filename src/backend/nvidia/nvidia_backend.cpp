/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_backend.hpp"
#include "backend/adjoint_plan.hpp"
#include "backend/nvidia/nvidia_adjoint.hpp"
#include "backend/nvidia/nvidia_coordinates.hpp"
#include "backend/nvidia/nvidia_cw.hpp"

#include <stdint.h>
#include <string.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "backend/initialization_plan.hpp"
#include "backend/graph_plan.hpp"
#include "backend/material_recipe.hpp"
#include "backend/descriptors.hpp"
#include "backend/diagnostics.hpp"
#include "backend/halo_plan.hpp"
#include "backend/nvidia/arena.hpp"
#include "backend/nvidia/nvidia_dft.hpp"
#include "backend/nvidia/nvidia_flux.hpp"
#include "backend/nvidia/nvidia_graph.hpp"
#include "backend/nvidia/nvidia_magnetic.hpp"
#include "backend/nvidia/nvidia_initialization.hpp"
#include "backend/nvidia/nvidia_materials.hpp"
#include "backend/nvidia/nvidia_multilevel.hpp"
#include "backend/nvidia/nvidia_polarization.hpp"
#include "backend/nvidia/runtime.hpp"
#include "backend/nvidia/nvidia_sources.hpp"
#include "backend/nvidia/nvidia_step.hpp"
#include "material_data.hpp"
#include "meepgeom.hpp"
#include "meep_internals.hpp"

namespace meep {

namespace nvidia {

namespace {
std::atomic<size_t> initialization_memory_budget_override(
    std::numeric_limits<size_t>::max());
}

namespace testing {
void set_initialization_memory_budget_for_testing(size_t bytes) {
  initialization_memory_budget_override.store(bytes, std::memory_order_relaxed);
}
size_t initialization_memory_budget_for_testing() {
  return initialization_memory_budget_override.load(std::memory_order_relaxed);
}
} // namespace testing

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

double derive_noisy_amplitude(double omega_0, double gamma, double noise_amplitude,
                              double dt_value) {
  const realnum gamma2pi = realnum(gamma) * 2 * pi;
  const realnum omega2pi = realnum(omega_0) * 2 * pi;
  const realnum dt = realnum(dt_value);
  const realnum amplitude = omega2pi * realnum(noise_amplitude) * sqrt(gamma2pi) * dt * dt /
                            (1 + gamma2pi * dt / 2);
  return double(amplitude);
}

bool valid_noisy_coefficients(double omega_0, double gamma, double noise_amplitude,
                              double dt_value) {
  if (!std::isfinite(omega_0) || !std::isfinite(gamma) ||
      !std::isfinite(noise_amplitude) || !std::isfinite(dt_value) || dt_value <= 0.0 ||
      !std::isfinite(double(realnum(omega_0))) || !std::isfinite(double(realnum(gamma))) ||
      !std::isfinite(double(realnum(noise_amplitude))) ||
      !std::isfinite(double(realnum(dt_value))))
    return false;
  const realnum gamma2pi = realnum(gamma) * 2 * pi;
  const realnum omega2pi = realnum(omega_0) * 2 * pi;
  const realnum dt = realnum(dt_value);
  if (!std::isfinite(double(gamma2pi)) || gamma2pi < realnum(0) ||
      !std::isfinite(double(omega2pi)))
    return false;
  const realnum root = sqrt(gamma2pi);
  const realnum denominator = 1 + gamma2pi * dt / 2;
  if (!std::isfinite(double(root)) || !std::isfinite(double(denominator)) ||
      denominator == realnum(0))
    return false;
  const realnum amplitude = omega2pi * realnum(noise_amplitude) * root * dt * dt / denominator;
  return std::isfinite(double(amplitude));
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

class NvtxApi {
public:
  typedef int (*push_type)(const char *);
  typedef int (*pop_type)();

  NvtxApi()
      : handle_(dlopen("libnvToolsExt.so.1", RTLD_NOW | RTLD_LOCAL)), push_(NULL), pop_(NULL) {
    if (!handle_) return;
    push_ = reinterpret_cast<push_type>(dlsym(handle_, "nvtxRangePushA"));
    pop_ = reinterpret_cast<pop_type>(dlsym(handle_, "nvtxRangePop"));
  }

  bool available() const { return push_ && pop_; }
  void push(const char *name) const { push_(name); }
  void pop() const { pop_(); }

private:
  void *handle_;
  push_type push_;
  pop_type pop_;
};

const NvtxApi &nvtx_api() {
  static const NvtxApi api;
  return api;
}

bool cw_profile_mode_requested() {
  const char *value = std::getenv("MEEP_NVIDIA_CW_PROFILE_ONLY");
  if (value && *value && strcmp(value, "0") != 0) return true;
  value = std::getenv("MEEP_NVIDIA_TIMESTEP_CW_PROFILE_ONLY");
  return value && *value && strcmp(value, "0") != 0;
}

class NvtxRange {
public:
  NvtxRange(const NvtxApi *api, const char *name) : api_(api) {
    if (api_) api_->push(name);
  }
  ~NvtxRange() {
    if (api_) api_->pop();
  }

private:
  const NvtxApi *api_;
};

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
    hash = mix_fingerprint(hash, spec.classification_provisional ? 1 : 0);
    hash = mix_fingerprint(hash, spec.classification_elided ? 1 : 0);
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
  if (spec.classification_elided)
    throw std::out_of_range("NVIDIA backend access names an elided ArrayId");
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

class NvidiaCwWorkspace {
public:
  NvidiaCwWorkspace(const nvidia::cw_workspace_shape &shape, nvidia::scalar_precision precision,
                    int L, int device)
      : shape_(shape), precision_(precision), L_(L), gamma_(size_t(L) + 1),
        gamma_p_(size_t(L) + 1), gamma_pp_(size_t(L) + 1),
        tau_(checked_product(size_t(L), size_t(L), "allocating solve_cw tau storage")),
        sigma_(size_t(L) + 1), r_(size_t(L) + 1), u_(size_t(L) + 1),
        operator_applications_(0), reduction_count_(0), source_scalar_h2d_calls_(0),
        source_scalar_h2d_bytes_(0), pack_kernel_launches_(0), unpack_kernel_launches_(0),
        zero_kernel_launches_(0), rhs_source_kernel_launches_(0),
        reconciliation_kernel_launches_(0), vector_kernel_launches_(0),
        operator_kernel_launches_(0), reduction_kernel_launches_(0),
        timestep_kernel_launches_(0), finite_check_kernel_launches_(0),
        diagnostic_d2h_calls_(0), diagnostic_d2h_bytes_(0),
        final_dft_kernel_launches_(0) {
    vectors_.allocate(shape_.vector_bytes, device);
    reduction_partials_.allocate(shape_.reduction_partial_bytes, device);
    reduction_result_.allocate(2 * sizeof(double), device);
    reduction_host_.allocate(2 * sizeof(double));
  }

  bool accommodates(const nvidia::cw_workspace_shape &shape,
                    nvidia::scalar_precision precision, int L) const {
    return precision_ == precision && L_ >= L && shape_.vector_elements >= shape.vector_elements &&
           shape_.vector_count >= shape.vector_count && shape_.vector_bytes >= shape.vector_bytes &&
           shape_.reduction_partial_bytes >= shape.reduction_partial_bytes;
  }

  nvidia::cw_workspace_shape shape_;
  nvidia::scalar_precision precision_;
  int L_;
  nvidia::device_buffer vectors_;
  nvidia::device_buffer reduction_partials_;
  nvidia::device_buffer reduction_result_;
  nvidia::pinned_buffer reduction_host_;
  std::vector<double> gamma_;
  std::vector<double> gamma_p_;
  std::vector<double> gamma_pp_;
  std::vector<double> tau_;
  std::vector<double> sigma_;
  std::vector<void *> r_;
  std::vector<void *> u_;
  size_t operator_applications_;
  size_t reduction_count_;
  size_t source_scalar_h2d_calls_;
  size_t source_scalar_h2d_bytes_;
  size_t pack_kernel_launches_;
  size_t unpack_kernel_launches_;
  size_t zero_kernel_launches_;
  size_t rhs_source_kernel_launches_;
  size_t reconciliation_kernel_launches_;
  size_t vector_kernel_launches_;
  size_t operator_kernel_launches_;
  size_t reduction_kernel_launches_;
  size_t timestep_kernel_launches_;
  size_t finite_check_kernel_launches_;
  size_t diagnostic_d2h_calls_;
  size_t diagnostic_d2h_bytes_;
  size_t final_dft_kernel_launches_;
};

struct NvidiaMaterialTableAuthority {
  MaterialIRTopologyRow row;
  ArrayId destination;
  ArrayId secondary_destination;
  size_t table_header_offset;
};

enum class NvidiaMaterialGeometryKind { bulk, analytic, patch };

struct NvidiaMaterialGeometryAuthority {
  NvidiaMaterialGeometryKind kind;
  uint32_t destination;
  size_t source_index;
  size_t source_count;
  ArrayId array;
  MaterialIRTopologyRow topology;
  size_t object_offset;
  size_t object_count;
  size_t image_offset;
  size_t image_count;
  size_t value_offset;
  size_t absorber_header_offset;
  size_t absorber_count;
  uint64_t first_point;
  size_t record_offset;
};

struct NvidiaInitializationUpload {
  uint32_t allocation;
  size_t bytes;
  size_t staging_offset;
  size_t logical_offset;
  bool material;
  struct Range {
    size_t offset;
    size_t bytes;
  };
  std::vector<Range> ranges;
};

struct NvidiaMaterialClassificationRow {
  StorageKey key;
  uint32_t allocation;
  size_t status_offset;
  size_t logical_elements;
  double trivial_value;
};

class NvidiaBackendState : public BackendState {
public:
  NvidiaBackendState(NvidiaBackend *owner, StoragePlan plan, int device, uint64_t state_token,
                     size_t initialization_reserve_bytes,
                     size_t initialization_compact_input_bytes,
                     size_t initialization_staging_bytes,
                     const std::vector<std::pair<StorageKey, size_t> >
                         &classification_rows)
      : owner_(owner), plan_(plan), layout_(allocation_requests_for(plan_)), device_(device),
        state_token_(state_token), fingerprint_(storage_fingerprint(plan_)), initialized_(false),
        transfer_failed_(false), material_refresh_pending_(false),
        device_authoritative_(false), magnetic_snapshot_bytes_(0),
        magnetic_snapshot_layout_fingerprint_(0), magnetic_snapshot_valid_(false),
        cw_skip_source_evaluation_(false), cw_workspace_allocations_(0),
        material_geometry_compact_hash_(0),
        initialization_compact_input_bytes_(initialization_compact_input_bytes),
        initialization_staging_bytes_(initialization_staging_bytes),
        initialization_prepared_(false),
        prepared_material_route_(MaterialRecipeDisposition::device_native),
        prepared_recipe_signature_(0), prepared_material_rows_(0),
        prepared_authoritative_arrays_(0),
        prepared_compact_staging_offset_(0), prepared_status_staging_offset_(0),
        prepared_result_staging_offset_(0), prepared_callback_scratch_offset_(0),
        noisy_seed_active_slot_(-1), noisy_seed_staged_slot_(-1),
        executable_build_count_(0), source_value_reuse_count_(0) {
    nvidia::device_scope scope(device_);
    transfer_.reset(new nvidia::stream);
    arenas_.reset(new nvidia::device_arenas(layout_, device_, initialization_reserve_bytes));
    noisy_seed_slots_.allocate(2 * sizeof(nvidia::noisy_seed_block), device_);
    if (initialization_staging_bytes_) staging_.allocate(initialization_staging_bytes_);
    size_t status_bytes = 0;
    for (size_t i = 0; i < plan_.arrays.size(); ++i)
      if (plan_.arrays[i].role == array_role::material &&
          !is_valid(plan_.arrays[i].alias_of)) {
        NvidiaMaterialClassificationRow row;
        row.key = plan_.keys[i];
        row.allocation = uint32_t(i);
        row.status_offset = status_bytes;
        row.logical_elements = 0;
        for (const std::pair<StorageKey, size_t> &candidate : classification_rows)
          if (candidate.first == row.key) {
            if (row.logical_elements)
              throw std::invalid_argument(
                  "duplicate NVIDIA logical material classification row");
            row.logical_elements = candidate.second;
          }
        if (!row.logical_elements)
          throw std::invalid_argument(
              "NVIDIA material classification row has no logical span");
        row.trivial_value = 0.0;
        material_classification_rows_.push_back(row);
        status_bytes = checked_add(status_bytes, row.logical_elements,
                                   "allocating NVIDIA material status");
      }
    if (material_classification_rows_.size() > UINT32_MAX)
      throw std::overflow_error("NVIDIA material classification row count overflows");
    if (status_bytes) material_classification_status_.allocate(status_bytes, device_);
    material_classification_result_bytes_ = checked_add(
        sizeof(nvidia::material_classification_header),
        checked_product(material_classification_rows_.size(),
                        sizeof(nvidia::material_classification_row_result),
                        "allocating NVIDIA classification rows"),
        "allocating NVIDIA classification result");
    material_classification_result_.allocate(material_classification_result_bytes_, device_);
  }

  ~NvidiaBackendState() override {}

  void ensure_staging(size_t bytes) {
    if (staging_.size() >= bytes) return;
    /* Preserve the installed epoch's staging on allocation failure.  This is
       required when an unpublished replacement executable asks for a larger
       host segment while the old executable remains retryable. */
    nvidia::pinned_buffer replacement;
    replacement.allocate(bytes);
    staging_ = std::move(replacement);
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
  bool material_refresh_pending_;
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
  bool cw_skip_source_evaluation_;
  size_t cw_workspace_allocations_;
  NvidiaCwStatistics cw_statistics_;
  NvidiaHostFallbackStatistics host_fallback_statistics_;
  NvidiaMaterialInitializationStatistics material_initialization_statistics_;
  NvidiaGraphStatistics graph_statistics_;
  nvidia::device_buffer material_ir_inputs_;
  std::vector<nvidia::material_fill_launch> material_fill_launches_;
  std::vector<nvidia::material_table_launch> material_table_launches_;
  std::vector<NvidiaMaterialTableAuthority> material_table_authorities_;
  std::vector<nvidia::material_conductivity_launch> material_conductivity_launches_;
  std::vector<nvidia::material_pml_launch> material_pml_launches_;
  std::vector<nvidia::geometry_bulk_launch> material_geometry_bulk_launches_;
  std::vector<nvidia::geometry_analytic_launch> material_geometry_analytic_launches_;
  std::vector<nvidia::geometry_patch_launch> material_geometry_patch_launches_;
  std::vector<NvidiaMaterialGeometryAuthority> material_geometry_authorities_;
  uint64_t material_geometry_compact_hash_;
  size_t initialization_compact_input_bytes_;
  size_t initialization_staging_bytes_;
  bool initialization_prepared_;
  MaterialRecipeDisposition prepared_material_route_;
  uint64_t prepared_recipe_signature_;
  size_t prepared_material_rows_;
  size_t prepared_authoritative_arrays_;
  size_t prepared_compact_staging_offset_;
  size_t prepared_status_staging_offset_;
  size_t prepared_result_staging_offset_;
  size_t prepared_callback_scratch_offset_;
  std::vector<NvidiaInitializationUpload> initialization_uploads_;
  std::vector<NvidiaMaterialClassificationRow> material_classification_rows_;
  nvidia::device_buffer material_classification_status_;
  nvidia::device_buffer material_classification_result_;
  size_t material_classification_result_bytes_;
  MaterialClassificationFacts material_classification_facts_;
  nvidia::device_buffer noisy_seed_slots_;
  int noisy_seed_active_slot_;
  int noisy_seed_staged_slot_;
  size_t executable_build_count_;
  size_t source_value_reuse_count_;
};

struct NvidiaCompiledOperation {
  OpKind kind;
  Guard guard;
  size_t first;
  size_t count;
  size_t material_first;
  size_t material_count;
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
  size_t legacy_flux_first;
  size_t legacy_flux_count;
  size_t source_staging_offset;
  size_t host_segment_index;
  double source_time_offset;
};

struct NvidiaCompiledHostTransfer {
  ArrayRef array;
  AccessMode mode;
  ArraySpec spec;
  size_t staging_offset;
  size_t storage_offset;
  size_t storage_bytes;
  size_t host_offset;
};

struct NvidiaCompiledHostSegment {
  HostSegment segment;
  std::vector<HostCallbackDescriptor> callbacks;
  StepPlan host_plan;
  std::vector<NvidiaCompiledHostTransfer> transfers;
  std::vector<InternalArrayLayout> resolution_layout;
  size_t staging_bytes;
};

struct NvidiaCompiledPolarizationAction {
  enum class Kind { ordinary, multilevel_population, multilevel_transition } kind;
  size_t index;
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

struct NvidiaCompiledMaterialRefresh {
  ArrayId current;
  ArraySpec spec;
  size_t staging_offset;
  size_t bytes;
};

struct NvidiaCompiledLegacyFluxUpdate {
  uint32_t flux_ordinal;
  size_t term_first;
  size_t term_count;
  uint64_t recipe_signature;
};

struct NvidiaCompiledGraphSegment {
  GraphSegment plan;
  nvidia::graph definition;
  nvidia::graph_exec executable;

  explicit NvidiaCompiledGraphSegment(const GraphSegment &source) : plan(source) {}
  NvidiaCompiledGraphSegment(NvidiaCompiledGraphSegment &&) noexcept = default;
  NvidiaCompiledGraphSegment &operator=(NvidiaCompiledGraphSegment &&) noexcept = default;
  NvidiaCompiledGraphSegment(const NvidiaCompiledGraphSegment &) = delete;
  NvidiaCompiledGraphSegment &operator=(const NvidiaCompiledGraphSegment &) = delete;
};

struct NvidiaCapturedGraph {
  nvidia::graph definition;
  nvidia::graph_exec executable;

  NvidiaCapturedGraph() {}
  NvidiaCapturedGraph(NvidiaCapturedGraph &&) noexcept = default;
  NvidiaCapturedGraph &operator=(NvidiaCapturedGraph &&) noexcept = default;
  NvidiaCapturedGraph(const NvidiaCapturedGraph &) = delete;
  NvidiaCapturedGraph &operator=(const NvidiaCapturedGraph &) = delete;
};

void remember_cw_graph_cleanup_error(std::string &error, const std::exception &failure) {
  if (error.empty()) error = failure.what();
}

bool cleanup_cw_graph(NvidiaCapturedGraph &graph, std::string &error) {
  try { graph.executable.reset(); }
  catch (const std::exception &failure) { remember_cw_graph_cleanup_error(error, failure); }
  catch (...) {
    if (error.empty()) error = "unknown CUDA CW graph executable cleanup failure";
  }
  try { graph.definition.reset(); }
  catch (const std::exception &failure) { remember_cw_graph_cleanup_error(error, failure); }
  catch (...) {
    if (error.empty()) error = "unknown CUDA CW graph definition cleanup failure";
  }
  return !graph.executable.opaque_handle() && !graph.definition.opaque_handle() &&
         !graph.definition.capturing();
}

bool cleanup_cw_graphs(std::vector<NvidiaCapturedGraph> &graphs, std::string &error) {
  bool released = true;
  for (NvidiaCapturedGraph &graph : graphs)
    released = cleanup_cw_graph(graph, error) && released;
  return released;
}

class NvidiaExecutable : public Executable {
public:
  NvidiaExecutable(const NvidiaBackend *owner, StepProgram program, uint64_t signature,
                   uint64_t structural_signature, uint64_t source_topology_signature,
                   const uint64_t *freshness, uint64_t resource_generation,
                   uint64_t storage_fingerprint,
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
                   const std::vector<NvidiaCompiledPolarizationAction>
                       &polarization_actions,
                   const std::vector<nvidia::multilevel_population_launch>
                       &multilevel_population_updates,
                   const std::vector<nvidia::multilevel_population_term_launch>
                       &multilevel_population_terms,
                   const std::vector<nvidia::multilevel_transition_launch>
                       &multilevel_transition_updates,
                   const std::vector<unsigned char> &multilevel_coefficients,
                   size_t multilevel_scratch_bytes,
                   const std::vector<nvidia::polarization_subtract_launch>
                       &polarization_subtractions,
                   const std::vector<nvidia::dft_launch> &dft_updates,
                   const std::vector<double> &dft_omega,
                   const std::vector<NvidiaCompiledLegacyFluxUpdate> &legacy_flux_updates,
                   const std::vector<nvidia::legacy_flux_term_launch> &legacy_flux_terms,
                   size_t legacy_flux_partial_count,
                   const std::vector<NvidiaCompiledMagneticState> &magnetic_states,
                   size_t magnetic_snapshot_bytes, uint64_t magnetic_layout_fingerprint,
                   const MagneticHalfStep &magnetic_half_step,
                   const std::vector<NvidiaCompiledMaterialRefresh> &material_refreshes,
                   size_t material_staging_bytes, uint64_t material_target_signature,
                   size_t source_scalar_count, size_t source_staging_elements,
                   const std::vector<NvidiaCompiledHostSegment> &host_segments,
                   size_t host_staging_bytes,
                   NvidiaBackendState &state)
      : Executable(state.material_phase_active), owner_(owner), state_token_(state_token),
        program_(program), signature_(signature), structural_signature_(structural_signature),
        source_topology_signature_(source_topology_signature),
        resource_generation_(resource_generation),
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
        polarization_actions_(polarization_actions),
        multilevel_population_updates_(multilevel_population_updates),
        multilevel_transition_updates_(multilevel_transition_updates), has_noisy_updates_(false),
        polarization_subtractions_(polarization_subtractions), dft_updates_(dft_updates),
        legacy_flux_updates_(legacy_flux_updates), legacy_flux_terms_(legacy_flux_terms),
        legacy_flux_global_(legacy_flux_updates.size(), 0.0),
        magnetic_states_(magnetic_states), magnetic_snapshot_bytes_(magnetic_snapshot_bytes),
        magnetic_layout_fingerprint_(magnetic_layout_fingerprint),
        magnetic_half_step_(magnetic_half_step), material_refreshes_(material_refreshes),
        material_target_signature_(material_target_signature),
        source_scalar_count_(source_scalar_count), host_segments_(host_segments),
        graph_enabled_(false), cw_workspace_(NULL) {
    for (int i = 0; i < mutation_kind_count; ++i) freshness_[i] = freshness[i];
    for (const nvidia::compiled_polarization_update &update : polarization_updates_)
      has_noisy_updates_ = has_noisy_updates_ ||
                           update.kind ==
                               nvidia::compiled_polarization_update::kind_type::noisy_add;
    try {
      nvidia::device_scope scope(state.device_);
      size_t executable_device_bytes = 0;
      const auto budget = [&](size_t bytes, const char *what) {
        executable_device_bytes = checked_add(executable_device_bytes, bytes, what);
      };
      budget(checked_product(halo_gathers.size(), sizeof(halo_gathers[0]),
                             "budgeting NVIDIA halo gather descriptors"),
             "budgeting NVIDIA executable storage");
      budget(checked_product(halo_scatters.size(), sizeof(halo_scatters[0]),
                             "budgeting NVIDIA halo scatter descriptors"),
             "budgeting NVIDIA executable storage");
      budget(halo_scratch_bytes, "budgeting NVIDIA executable storage");
      if (!finite_checks_.empty()) budget(sizeof(uint64_t), "budgeting NVIDIA executable storage");
      budget(checked_product(source_scalar_count_, sizeof(nvidia::source_scalar),
                             "budgeting NVIDIA source scalars"),
             "budgeting NVIDIA executable storage");
      budget(checked_product(source_points.size(), sizeof(source_points[0]),
                             "budgeting NVIDIA source points"),
             "budgeting NVIDIA executable storage");
      budget(checked_product(dft_omega.size(), sizeof(double),
                             "budgeting NVIDIA DFT frequencies"),
             "budgeting NVIDIA executable storage");
      budget(checked_product(legacy_flux_updates_.size(), 2 * sizeof(double),
                             "budgeting NVIDIA legacy flux scalars"),
             "budgeting NVIDIA executable storage");
      budget(checked_product(legacy_flux_partial_count, sizeof(double),
                             "budgeting NVIDIA legacy flux partials"),
             "budgeting NVIDIA executable storage");
      budget(multilevel_coefficients.size(), "budgeting NVIDIA executable storage");
      budget(checked_product(multilevel_population_terms.size(),
                             sizeof(multilevel_population_terms[0]),
                             "budgeting NVIDIA multilevel term descriptors"),
             "budgeting NVIDIA executable storage");
      budget(multilevel_scratch_bytes, "budgeting NVIDIA executable storage");
      const size_t free_device_bytes = nvidia::free_memory_for_device(state.device_);
      if (executable_device_bytes > free_device_bytes)
        throw std::runtime_error("NVIDIA executable memory budget exceeds available device memory");
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
      if (!multilevel_coefficients.empty()) {
        multilevel_coefficients_.allocate(multilevel_coefficients.size(), state.device_);
        nvidia::copy_host_to_device_async(multilevel_coefficients_, 0,
                                          multilevel_coefficients.data(),
                                          multilevel_coefficients.size(), *state.transfer_);
      }
      if (!multilevel_population_terms.empty()) {
        multilevel_population_terms_.allocate(
            checked_product(multilevel_population_terms.size(),
                            sizeof(multilevel_population_terms[0]),
                            "allocating NVIDIA multilevel term descriptors"),
            state.device_);
        nvidia::copy_host_to_device_async(multilevel_population_terms_, 0,
                                          multilevel_population_terms.data(),
                                          multilevel_population_terms_.size(), *state.transfer_);
      }
      if (multilevel_scratch_bytes)
        multilevel_scratch_.allocate(multilevel_scratch_bytes, state.device_);
      unsigned char *coefficient_base = static_cast<unsigned char *>(
          multilevel_coefficients_.opaque_handle());
      const nvidia::multilevel_population_term_launch *term_base =
          static_cast<const nvidia::multilevel_population_term_launch *>(
              multilevel_population_terms_.opaque_handle());
      for (nvidia::multilevel_population_launch &update : multilevel_population_updates_) {
        update.gamma_matrix = coefficient_base + update.gamma_byte_offset;
        update.alpha = coefficient_base + update.alpha_byte_offset;
        update.transition_gperpdt = coefficient_base + update.transition_byte_offset;
        update.terms = update.term_count ? term_base + update.term_index : NULL;
        update.scratch = multilevel_scratch_.opaque_handle();
      }
      for (nvidia::multilevel_transition_launch &update : multilevel_transition_updates_)
        update.coefficients = coefficient_base + update.coefficient_byte_offset;
      if (material_staging_bytes) material_staging_.allocate(material_staging_bytes);
      if (host_staging_bytes) state.ensure_staging(host_staging_bytes);
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
      if (!legacy_flux_updates_.empty()) {
        const size_t scalar_bytes = checked_product(legacy_flux_updates_.size(), sizeof(double),
                                                    "allocating NVIDIA legacy flux scalars");
        legacy_flux_half_.allocate(scalar_bytes, state.device_);
        legacy_flux_current_.allocate(scalar_bytes, state.device_);
        legacy_flux_host_.allocate(scalar_bytes);
        nvidia::fill_byte_async(legacy_flux_half_, 0, 0, scalar_bytes, *state.transfer_);
        nvidia::fill_byte_async(legacy_flux_current_, 0, 0, scalar_bytes, *state.transfer_);
      }
      if (legacy_flux_partial_count)
        legacy_flux_partials_.allocate(
            checked_product(legacy_flux_partial_count, sizeof(double),
                            "allocating NVIDIA legacy flux reduction scratch"),
            state.device_);
      if (!halo_gathers.empty() || !halo_scatters.empty() || !source_points.empty() ||
          !dft_omega.empty() || !legacy_flux_updates_.empty() ||
          !multilevel_coefficients.empty() || !multilevel_population_terms.empty())
        state.transfer_->synchronize();
    }
    catch (...) {
      /* Every destination above belongs to this unpublished executable.
         Cleanup is local, so a preflight compile failure leaves the live
         state and its stream retryable. */
      throw;
    }
  }

  ~NvidiaExecutable() override {
    /* Captured graph nodes retain raw addresses owned by the buffers below.
       Destroy executable graphs and their definitions while every captured
       allocation is still alive; member reverse-order destruction would do
       the opposite because graph_segments_ precedes those buffers. */
    magnetic_restore_graph_segments_.clear();
    magnetic_half_graph_segments_.clear();
    magnetic_auxiliary_graphs_.clear();
    graph_segments_.clear();
  }

  const NvidiaBackend *owner_;
  uint64_t state_token_;
  StepProgram program_;
  uint64_t signature_;
  uint64_t structural_signature_;
  uint64_t source_topology_signature_;
  uint64_t freshness_[mutation_kind_count];
  uint64_t resource_generation_;
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
  std::vector<NvidiaCompiledPolarizationAction> polarization_actions_;
  std::vector<nvidia::multilevel_population_launch> multilevel_population_updates_;
  std::vector<nvidia::multilevel_transition_launch> multilevel_transition_updates_;
  bool has_noisy_updates_;
  std::vector<nvidia::polarization_subtract_launch> polarization_subtractions_;
  std::vector<nvidia::dft_launch> dft_updates_;
  std::vector<NvidiaCompiledLegacyFluxUpdate> legacy_flux_updates_;
  std::vector<nvidia::legacy_flux_term_launch> legacy_flux_terms_;
  std::vector<double> legacy_flux_global_;
  std::vector<NvidiaCompiledMagneticState> magnetic_states_;
  size_t magnetic_snapshot_bytes_;
  uint64_t magnetic_layout_fingerprint_;
  MagneticHalfStep magnetic_half_step_;
  std::vector<NvidiaCompiledMaterialRefresh> material_refreshes_;
  uint64_t material_target_signature_;
  size_t source_scalar_count_;
  std::vector<NvidiaCompiledHostSegment> host_segments_;
  GraphProgram graph_program_;
  std::vector<NvidiaCompiledGraphSegment> graph_segments_;
  std::vector<size_t> graph_cw_timestep_launches_;
  std::vector<size_t> graph_cw_finite_launches_;
  GraphProgram magnetic_half_graph_program_;
  std::vector<NvidiaCompiledGraphSegment> magnetic_half_graph_segments_;
  GraphProgram magnetic_restore_graph_program_;
  std::vector<NvidiaCompiledGraphSegment> magnetic_restore_graph_segments_;
  std::vector<NvidiaCapturedGraph> magnetic_auxiliary_graphs_;
  nvidia::device_buffer step_scalars_;
  nvidia::device_buffer magnetic_step_scalars_;
  bool graph_enabled_;
  NvidiaCwWorkspace *cw_workspace_;
  NvidiaGraphStatistics graph_statistics_;
  nvidia::device_buffer halo_gathers_;
  nvidia::device_buffer halo_scatters_;
  nvidia::device_buffer halo_scratch_;
  nvidia::device_buffer finite_result_;
  nvidia::pinned_buffer finite_result_host_;
  nvidia::device_buffer source_scalars_;
  nvidia::pinned_buffer source_staging_;
  nvidia::pinned_buffer material_staging_;
  nvidia::device_buffer source_points_;
  nvidia::device_buffer dft_omega_;
  nvidia::device_buffer multilevel_coefficients_;
  nvidia::device_buffer multilevel_population_terms_;
  nvidia::device_buffer multilevel_scratch_;
  nvidia::device_buffer legacy_flux_half_;
  nvidia::device_buffer legacy_flux_current_;
  nvidia::device_buffer legacy_flux_partials_;
  nvidia::pinned_buffer legacy_flux_host_;
};

class NvidiaCwExecutable : public Executable {
public:
  struct Stage {
    double source_time_offset;
    size_t source_first;
    size_t source_count;
    uint32_t boundary_operation;
    uint32_t constitutive_operation;
  };

  NvidiaCwExecutable(const NvidiaBackend *owner, uint64_t layout_storage_fingerprint,
                     uint64_t device_storage_fingerprint,
                     uint64_t step_plan_signature, uint64_t cw_plan_signature,
                     uint64_t structural_step_signature,
                     uint64_t structural_cw_signature, uint64_t source_topology_signature,
                     const uint64_t *freshness, uint64_t resource_generation,
                     nvidia::scalar_precision precision, size_t real_count,
                     const std::vector<nvidia::cw_state_row_launch> &rows,
                     const std::vector<nvidia::cw_zero_launch> &zeroes,
                     const std::vector<Stage> &rhs_stages,
                     const std::vector<nvidia::cw_source_batch_launch> &rhs_sources,
                     const std::vector<nvidia::source_point> &source_points,
                     const std::vector<nvidia::dft_launch> &final_dfts,
                     const std::vector<double> &dft_omega,
                     const std::vector<CwDftDescriptorRef> &final_dft_refs,
                     const CwUnpackDescriptorRefs &unpack,
                     std::unique_ptr<NvidiaCwWorkspace> workspace,
                     std::unique_ptr<NvidiaExecutable> timestep, NvidiaBackendState &state)
      : owner_(owner), layout_storage_fingerprint_(layout_storage_fingerprint),
        device_storage_fingerprint_(device_storage_fingerprint),
        state_token_(state.state_token_), step_plan_signature_(step_plan_signature),
        cw_plan_signature_(cw_plan_signature), structural_step_signature_(structural_step_signature),
        structural_cw_signature_(structural_cw_signature),
        source_topology_signature_(source_topology_signature),
        resource_generation_(resource_generation),
        precision_(precision), real_count_(real_count), rows_(rows), zeroes_(zeroes),
        rhs_stages_(rhs_stages), rhs_sources_(rhs_sources), final_dfts_(final_dfts),
        unpack_skip_w_components_(unpack.skip_w_components), workspace_(std::move(workspace)),
        timestep_(std::move(timestep)) {
    for (int i = 0; i < mutation_kind_count; ++i) freshness_[i] = freshness[i];
    unpack_operations_[0] = unpack.first_boundary.operation_index;
    unpack_operations_[1] = unpack.constitutive.operation_index;
    unpack_operations_[2] = unpack.second_boundary.operation_index;
    timestep_->cw_workspace_ = workspace_.get();
    final_dft_due_slots_.reserve(final_dft_refs.size());
    for (const CwDftDescriptorRef &reference : final_dft_refs)
      final_dft_due_slots_.push_back(reference.due_scalar_slot);
    nvidia::device_scope scope(state.device_);
    if (!source_points.empty()) {
      source_points_.allocate(checked_product(source_points.size(), sizeof(source_points[0]),
                                              "allocating NVIDIA solve_cw source points"),
                              state.device_);
      nvidia::copy_host_to_device_async(source_points_, 0, source_points.data(),
                                        source_points_.size(), *state.transfer_);
      const nvidia::source_point *base =
          static_cast<const nvidia::source_point *>(source_points_.opaque_handle());
      for (nvidia::cw_source_batch_launch &source : rhs_sources_)
        source.points = base + source.point_offset;
    }
    if (!dft_omega.empty()) {
      dft_omega_.allocate(checked_product(dft_omega.size(), sizeof(dft_omega[0]),
                                         "allocating NVIDIA solve_cw DFT frequencies"),
                          state.device_);
      nvidia::copy_host_to_device_async(dft_omega_, 0, dft_omega.data(), dft_omega_.size(),
                                        *state.transfer_);
      const double *base = static_cast<const double *>(dft_omega_.opaque_handle());
      for (nvidia::dft_launch &dft : final_dfts_) dft.omega = base;
    }
    if (!source_points.empty() || !dft_omega.empty()) state.transfer_->synchronize();
  }

  ~NvidiaCwExecutable() override {
    /* Every graph is gone before its scalar block, nested timestep, workspace,
       descriptor buffers, or backend state can be retired. */
    std::string cleanup_error;
    bool released = cleanup_cw_graphs(rhs_stage_graphs_, cleanup_error);
    released = cleanup_cw_graphs(vector_graphs_, cleanup_error) && released;
    released = cleanup_cw_graphs(reduction_graphs_, cleanup_error) && released;
    released = cleanup_cw_graph(final_dft_graph_, cleanup_error) && released;
    released = cleanup_cw_graph(operator_finalize_graph_, cleanup_error) && released;
    released = cleanup_cw_graph(operator_prelude_graph_, cleanup_error) && released;
    released = cleanup_cw_graph(pack_graph_, cleanup_error) && released;
    released = cleanup_cw_graph(rhs_zero_graph_, cleanup_error) && released;
    if (!released || !cleanup_error.empty()) {
      /* Failure injection is one-shot. Retry while the captured buffers are
         still members of this owner; the graph RAII destructors remain the
         final best-effort path for a persistent CUDA failure. */
      cleanup_error.clear();
      released = cleanup_cw_graphs(rhs_stage_graphs_, cleanup_error);
      released = cleanup_cw_graphs(vector_graphs_, cleanup_error) && released;
      released = cleanup_cw_graphs(reduction_graphs_, cleanup_error) && released;
      released = cleanup_cw_graph(final_dft_graph_, cleanup_error) && released;
      released = cleanup_cw_graph(operator_finalize_graph_, cleanup_error) && released;
      released = cleanup_cw_graph(operator_prelude_graph_, cleanup_error) && released;
      released = cleanup_cw_graph(pack_graph_, cleanup_error) && released;
      released = cleanup_cw_graph(rhs_zero_graph_, cleanup_error) && released;
    }
    if (released) {
      rhs_stage_graphs_.clear();
      rhs_stage_reconciliation_kernel_counts_.clear();
      vector_graphs_.clear();
      reduction_graphs_.clear();
    }
    timestep_.reset();
  }

  const NvidiaBackend *owner_;
  uint64_t layout_storage_fingerprint_;
  uint64_t device_storage_fingerprint_;
  uint64_t state_token_;
  uint64_t step_plan_signature_;
  uint64_t cw_plan_signature_;
  uint64_t structural_step_signature_;
  uint64_t structural_cw_signature_;
  uint64_t source_topology_signature_;
  uint64_t freshness_[mutation_kind_count];
  uint64_t resource_generation_;
  nvidia::scalar_precision precision_;
  size_t real_count_;
  std::vector<nvidia::cw_state_row_launch> rows_;
  std::vector<nvidia::cw_zero_launch> zeroes_;
  std::vector<Stage> rhs_stages_;
  std::vector<nvidia::cw_source_batch_launch> rhs_sources_;
  std::vector<nvidia::dft_launch> final_dfts_;
  std::vector<uint32_t> final_dft_due_slots_;
  uint32_t unpack_operations_[3];
  bool unpack_skip_w_components_;
  nvidia::device_buffer source_points_;
  nvidia::device_buffer dft_omega_;
  std::unique_ptr<NvidiaCwWorkspace> workspace_;
  std::unique_ptr<NvidiaExecutable> timestep_;
  nvidia::device_buffer graph_scalars_;
  nvidia::device_buffer final_dft_scalars_;
  NvidiaCapturedGraph rhs_zero_graph_;
  std::vector<NvidiaCapturedGraph> rhs_stage_graphs_;
  std::vector<size_t> rhs_stage_reconciliation_kernel_counts_;
  NvidiaCapturedGraph operator_prelude_graph_;
  NvidiaCapturedGraph pack_graph_;
  NvidiaCapturedGraph operator_finalize_graph_;
  std::vector<NvidiaCapturedGraph> vector_graphs_;
  std::vector<NvidiaCapturedGraph> reduction_graphs_;
  NvidiaCapturedGraph final_dft_graph_;
  bool graphs_enabled_ = false;
  size_t graph_capture_count_ = 0;
  size_t graph_instantiate_count_ = 0;
  size_t graph_launch_count_ = 0;
  size_t graph_scalar_write_count_ = 0;
  size_t graph_rhs_launch_count_ = 0;
  size_t graph_unpack_launch_count_ = 0;
  size_t graph_pack_launch_count_ = 0;
  size_t graph_vector_launch_count_ = 0;
  size_t graph_reduction_launch_count_ = 0;
  size_t graph_operator_launch_count_ = 0;
  size_t graph_final_dft_launch_count_ = 0;
  size_t graph_operator_reconciliation_kernel_count_ = 0;
};

namespace {

nvidia::scalar_precision scalar_precision_for(const StoragePlan &plan, ArrayId id,
                                              const char *what) {
  if (!is_valid(id) || id.value >= plan.arrays.size())
    throw std::invalid_argument(std::string(what) + " uses an invalid ArrayId");
  const ArraySpec &spec = plan.arrays[id.value];
  if (spec.classification_elided)
    throw std::invalid_argument(std::string(what) + " uses an elided ArrayId");
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
  if (spec.classification_elided)
    throw std::invalid_argument(std::string(what) + " uses an elided ArrayId");
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
  if (spec.classification_elided)
    throw std::invalid_argument(std::string(what) + " uses an elided ArrayId");
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

NvidiaCompiledMaterialRefresh compile_material_refresh(const MaterialRefreshArray &source,
                                                        NvidiaBackendState &state,
                                                        const fields &f, size_t &next_offset) {
  if (source.chunk < 0 || source.chunk >= f.num_chunks || !f.chunks[source.chunk] ||
      !f.chunks[source.chunk]->is_mine())
    throw std::out_of_range("NVIDIA material refresh has an invalid chunk");
  if (source.c < 0 || source.c >= NUM_FIELD_COMPONENTS || source.d < 0 || source.d >= 5)
    throw std::invalid_argument("NVIDIA material refresh has an invalid component or direction");
  const array_kind expected_kind = source.family == MaterialRefreshFamily::chi1inv
                                       ? array_kind::chi1inv
                                       : source.family == MaterialRefreshFamily::conductivity
                                             ? array_kind::conductivity
                                             : source.family == MaterialRefreshFamily::condinv
                                                   ? array_kind::condinv
                                                   : array_kind::num_kinds;
  if (expected_kind == array_kind::num_kinds)
    throw std::invalid_argument("NVIDIA material refresh has an invalid family");
  if (source.family == MaterialRefreshFamily::condinv &&
      source.d != component_direction(source.c))
    throw std::invalid_argument("NVIDIA material refresh has a non-diagonal condinv row");
  const nvidia::scalar_precision precision =
      scalar_precision_for(state.plan_, source.current, "NVIDIA material refresh");
  const ArraySpec &spec = state.plan_.arrays[source.current.value];
  const StorageKey &key = state.plan_.keys[source.current.value];
  if (spec.role != array_role::material || is_valid(spec.alias_of) || !source.elements ||
      source.elements != spec.elements)
    throw std::invalid_argument("NVIDIA material refresh has invalid storage metadata");
  if (key.chunk != source.chunk || key.kind != int(expected_kind) ||
      key.component_ != int(source.c) || key.cmp != -1 || key.aux != int(source.d))
    throw std::invalid_argument("NVIDIA material refresh identity does not match its ArrayId");
  const structure_chunk &sc = *f.chunks[source.chunk]->s;
  const realnum *expected = source.family == MaterialRefreshFamily::chi1inv
                                ? sc.chi1inv[source.c][source.d]
                                : source.family == MaterialRefreshFamily::conductivity
                                      ? sc.conductivity[source.c][source.d]
                                      : sc.condinv[source.c][source.d];
  if (!expected || !f.array_catalog ||
      f.array_catalog->resolve<realnum>(source.current) != expected)
    throw std::invalid_argument("NVIDIA material refresh does not name the current material row");

  const size_t alignment =
      precision == nvidia::scalar_precision::f32 ? alignof(float) : alignof(double);
  const size_t padding = (alignment - next_offset % alignment) % alignment;
  next_offset = checked_add(next_offset, padding, "aligning NVIDIA material refresh staging");
  NvidiaCompiledMaterialRefresh result = {
      source.current, spec, next_offset,
      checked_product(source.elements,
                      precision == nvidia::scalar_precision::f32 ? sizeof(float) : sizeof(double),
                      "sizing NVIDIA material refresh")};
  next_offset = checked_add(next_offset, result.bytes, "laying out NVIDIA material refresh staging");
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

nvidia::cw_source_batch_launch compile_cw_source_batch(
    const SourceDescriptor &source, const SourcePlan &source_plan, const fields &f,
    NvidiaBackendState &state, std::vector<nvidia::source_point> &points) {
  if (source.indices.empty() || source.indices.size() != source.complex_amplitudes.size())
    throw std::invalid_argument("NVIDIA solve_cw source has invalid spatial data");
  if (source.source_time_id >= source_plan.source_times.size())
    throw std::out_of_range("NVIDIA solve_cw source has an invalid source-time ID");
  const SourceTimeDescriptor &time = source_plan.source_times[source.source_time_id];
  if (time.source_time_id != source.source_time_id || time.scalar_slot >= source_plan.scalars.size())
    throw std::invalid_argument("NVIDIA solve_cw source has a non-canonical scalar mapping");

  nvidia::cw_source_batch_launch result = {};
  result.precision = scalar_precision_for(state.plan_, source.destination,
                                          "solve_cw primary source target");
  result.target_real = device_address(state, source.destination,
                                      "solve_cw primary source target");
  result.target_imag = optional_mutable_device_address(
      state, source.destination_imag, result.precision, "solve_cw imaginary source target");
  result.conductivity_inverse = optional_device_address(
      state, source.condinv, result.precision, "solve_cw source conductivity inverse");
  if (!result.target_imag)
    throw std::invalid_argument("complex NVIDIA solve_cw source has no imaginary target");
  result.point_offset = points.size();
  result.point_count = source.indices.size();
  result.scalar_slot = time.scalar_slot;
  result.dt = f.dt;
  result.sequential =
      nvidia::source_indices_require_sequential(source.indices.data(), source.indices.size());
  const std::pair<std::vector<ptrdiff_t>::const_iterator,
                  std::vector<ptrdiff_t>::const_iterator> bounds =
      std::minmax_element(source.indices.begin(), source.indices.end());
  validate_index_range(state.plan_, source.destination, *bounds.first, *bounds.second,
                       "solve_cw primary source target");
  validate_index_range(state.plan_, source.destination_imag, *bounds.first, *bounds.second,
                       "solve_cw imaginary source target");
  if (is_valid(source.condinv))
    validate_index_range(state.plan_, source.condinv, *bounds.first, *bounds.second,
                         "solve_cw source conductivity inverse");
  for (size_t point = 0; point < source.indices.size(); ++point)
    points.push_back(nvidia::source_point{source.indices[point],
                                          source.complex_amplitudes[point].real(),
                                          source.complex_amplitudes[point].imag()});
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

bool same_polarization_access(const BufferAccess &a, const BufferAccess &b) {
  return a.array.id == b.array.id && a.array.offset == b.array.offset &&
         a.array.elements == b.array.elements && a.mode == b.mode;
}

bool same_polarization_operation(const Operation &a, const Operation &b) {
  if (a.kind != b.kind || a.ft != b.ft || a.descriptor_index != b.descriptor_index ||
      a.descriptor_count != b.descriptor_count ||
      a.material_refresh_index != b.material_refresh_index ||
      a.material_refresh_count != b.material_refresh_count ||
      a.beta_descriptor_index != b.beta_descriptor_index ||
      a.beta_descriptor_count != b.beta_descriptor_count ||
      a.cylindrical_m_descriptor_index != b.cylindrical_m_descriptor_index ||
      a.cylindrical_m_descriptor_count != b.cylindrical_m_descriptor_count ||
      a.cylindrical_origin_action_index != b.cylindrical_origin_action_index ||
      a.cylindrical_origin_action_count != b.cylindrical_origin_action_count ||
      a.polarization_group_index != b.polarization_group_index ||
      a.polarization_group_count != b.polarization_group_count ||
      a.polarization_subtraction_index != b.polarization_subtraction_index ||
      a.polarization_subtraction_count != b.polarization_subtraction_count ||
      a.magnetic_state_index != b.magnetic_state_index ||
      a.magnetic_state_count != b.magnetic_state_count ||
      a.legacy_flux_index != b.legacy_flux_index ||
      a.legacy_flux_count != b.legacy_flux_count ||
      a.source_descriptor_index != b.source_descriptor_index ||
      a.source_descriptor_count != b.source_descriptor_count ||
      a.guard.kind != b.guard.kind || a.guard.scalar_slot != b.guard.scalar_slot ||
      a.guard.variant_index != b.guard.variant_index ||
      a.source_time_offset != b.source_time_offset || a.accesses.size() != b.accesses.size())
    return false;
  for (size_t i = 0; i < a.accesses.size(); ++i)
    if (!same_polarization_access(a.accesses[i], b.accesses[i])) return false;
  return true;
}

bool same_multilevel_descriptor(const PolarizationDescriptor &a,
                                const PolarizationDescriptor &b) {
  if (a.kind != SusceptibilityKind::multilevel || b.kind != SusceptibilityKind::multilevel ||
      a.chunk != b.chunk || a.ft != b.ft || a.state_index != b.state_index ||
      a.has_internal_state != b.has_internal_state ||
      a.multilevel.levels != b.multilevel.levels ||
      a.multilevel.transitions != b.multilevel.transitions ||
      a.multilevel.gamma_matrix != b.multilevel.gamma_matrix ||
      a.multilevel.initial_populations != b.multilevel.initial_populations ||
      a.multilevel.alpha != b.multilevel.alpha || a.multilevel.omega != b.multilevel.omega ||
      a.multilevel.transition_gamma != b.multilevel.transition_gamma ||
      a.multilevel.sigmat != b.multilevel.sigmat ||
      a.multilevel_gamma_inv != b.multilevel_gamma_inv ||
      a.multilevel_populations != b.multilevel_populations ||
      a.multilevel_population_points != b.multilevel_population_points ||
      a.per_thread_scratch_elements != b.per_thread_scratch_elements ||
      a.required_w != b.required_w || a.required_w_prev != b.required_w_prev ||
      a.needs_halo != b.needs_halo ||
      a.multilevel_states.size() != b.multilevel_states.size() ||
      a.internal_arrays.size() != b.internal_arrays.size())
    return false;
  for (size_t i = 0; i < a.multilevel_states.size(); ++i) {
    const MultilevelStateArrays &x = a.multilevel_states[i];
    const MultilevelStateArrays &y = b.multilevel_states[i];
    if (x.transition_index != y.transition_index || x.c != y.c || x.cmp != y.cmp ||
        x.p != y.p || x.p_prev != y.p_prev || x.elements != y.elements)
      return false;
  }
  for (size_t i = 0; i < a.internal_arrays.size(); ++i) {
    const InternalArrayLayout &x = a.internal_arrays[i];
    const InternalArrayLayout &y = b.internal_arrays[i];
    if (((x.name || y.name) && (!x.name || !y.name || strcmp(x.name, y.name))) ||
        x.element_type != y.element_type || x.offset_elements != y.offset_elements ||
        x.elements != y.elements || x.c != y.c || x.cmp != y.cmp)
      return false;
  }
  return true;
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
    case PolarizationUpdateKind::noisy_add: {
      if (source.region.variant_key || !is_valid(source.p) || !is_valid(source.diagonal_sigma) ||
          is_valid(source.p_prev) || is_valid(source.p_cross1) ||
          is_valid(source.p_prev_cross1) || is_valid(source.p_cross2) ||
          is_valid(source.p_prev_cross2) || is_valid(source.primary_w) ||
          is_valid(source.cross_w1) || is_valid(source.cross_w2) ||
          is_valid(source.offdiagonal_sigma1) || is_valid(source.offdiagonal_sigma2) ||
          source.primary_stride || source.cross_stride1 || source.cross_stride2 ||
          source.noise_algorithm_version != counter_random_algorithm_version ||
          source.state_index < 0 || uint64_t(source.state_index) > UINT32_MAX ||
          source.region.chunk < 0 || uint64_t(source.region.chunk) > UINT32_MAX ||
          (source.ft != E_stuff && source.ft != H_stuff) ||
          source.region.cmp < 0 || source.region.cmp > 1 ||
          (source.ft == E_stuff ? !is_electric(source.region.c)
                                : !is_magnetic(source.region.c)) ||
          source.alpha != 0.0 || source.gyro_model != GYROTROPIC_LORENTZIAN ||
          !nvidia::valid_noisy_coefficients(source.omega_0, source.gamma,
                                            source.noise_amplitude, source.dt))
        throw std::invalid_argument("noisy polarization descriptor is noncanonical");
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          if (source.gyro_tensor[i][j] != 0.0)
            throw std::invalid_argument("noisy polarization descriptor has a gyro tensor");
      if (source.p.value >= state.plan_.arrays.size() ||
          source.p.value >= state.plan_.keys.size() ||
          source.diagonal_sigma.value >= state.plan_.arrays.size() ||
          source.diagonal_sigma.value >= state.plan_.keys.size())
        throw std::out_of_range("noisy polarization descriptor ArrayId is out of range");
      const ArraySpec &p_spec = state.plan_.arrays[source.p.value];
      const StorageKey &p_key = state.plan_.keys[source.p.value];
      const ArraySpec &sigma_spec = state.plan_.arrays[source.diagonal_sigma.value];
      const StorageKey &sigma_key = state.plan_.keys[source.diagonal_sigma.value];
      if (source.state_index >
          (std::numeric_limits<int>::max() - int(source.ft)) / NUM_FIELD_TYPES)
        throw std::overflow_error("noisy polarization sigma identity overflows");
      const int sigma_aux = source.state_index * NUM_FIELD_TYPES + int(source.ft);
      if (p_spec.id != source.p || p_spec.role != array_role::polarization ||
          p_spec.element_type != ElementType::realnum_value || is_valid(p_spec.alias_of) ||
          p_key.chunk != source.region.chunk ||
          p_key.kind != int(array_kind::polarization_internal) ||
          p_key.component_ != int(source.region.c) || p_key.cmp != source.region.cmp ||
          polarization_storage_field_type(p_key.aux) != source.ft ||
          polarization_storage_state_index(p_key.aux) != source.state_index)
        throw std::invalid_argument("noisy polarization P has the wrong storage identity");
      if (sigma_spec.id != source.diagonal_sigma || sigma_spec.role != array_role::material ||
          sigma_spec.element_type != ElementType::realnum_value ||
          is_valid(sigma_spec.alias_of) || sigma_key.chunk != source.region.chunk ||
          sigma_key.kind != int(array_kind::sigma) ||
          sigma_key.component_ != int(source.region.c) ||
          sigma_key.cmp != int(component_direction(source.region.c)) ||
          sigma_key.aux != uint64_t(sigma_aux))
        throw std::invalid_argument("noisy polarization sigma has the wrong storage identity");
      nvidia::noisy_add_launch launch = {};
      launch.region = flat_region_for(source.region);
      launch.precision = scalar_precision_for(state.plan_, source.p, "noisy polarization P");
      require_same_precision(state.plan_, source.diagonal_sigma, launch.precision,
                             "noisy polarization sigma");
      launch.p = device_address(state, source.p, "noisy polarization P");
      launch.diagonal_sigma =
          device_address(state, source.diagonal_sigma, "noisy polarization sigma");
      launch.amplitude = nvidia::derive_noisy_amplitude(
          source.omega_0, source.gamma, source.noise_amplitude, source.dt);
      if (!std::isfinite(launch.amplitude))
        throw std::invalid_argument("noisy polarization amplitude is not finite");
      const int global_rank = my_global_rank();
      if (global_rank < 0 || uint64_t(global_rank) > UINT32_MAX)
        throw std::invalid_argument("noisy polarization global rank is out of range");
      launch.stream_tag = counter_random_stream_tag(
          source.noise_algorithm_version, uint32_t(global_rank), uint32_t(source.region.chunk),
          uint32_t(source.ft), uint32_t(source.state_index), uint32_t(source.region.c),
          uint32_t(source.region.cmp));
      launch.point_ordinal_base = 0;
      const ptrdiff_t region_min = ptrdiff_t(launch.region.base);
      const ptrdiff_t region_max = checked_region_max(launch.region);
      validate_index_range(state.plan_, source.p, region_min, region_max,
                           "noisy polarization P");
      validate_index_range(state.plan_, source.diagonal_sigma, region_min, region_max,
                           "noisy polarization sigma");
      result.kind = nvidia::compiled_polarization_update::kind_type::noisy_add;
      result.noisy = launch;
      break;
    }
    default: throw std::invalid_argument("polarization descriptor has an invalid update kind");
  }
  return result;
}

size_t append_multilevel_coefficients(std::vector<unsigned char> &storage,
                                      nvidia::scalar_precision precision,
                                      const std::vector<double> &values) {
  const size_t bytes = precision == nvidia::scalar_precision::f32 ? sizeof(float)
                                                                  : sizeof(double);
  const size_t padding = (bytes - storage.size() % bytes) % bytes;
  const size_t offset = checked_add(storage.size(), padding,
                                    "aligning NVIDIA multilevel coefficients");
  const size_t payload = checked_product(values.size(), bytes,
                                         "sizing NVIDIA multilevel coefficients");
  storage.resize(checked_add(offset, payload, "packing NVIDIA multilevel coefficients"), 0);
  if (precision == nvidia::scalar_precision::f32) {
    for (size_t i = 0; i < values.size(); ++i) {
      const float value = float(values[i]);
      memcpy(storage.data() + offset + i * sizeof(value), &value, sizeof(value));
    }
  }
  else {
    for (size_t i = 0; i < values.size(); ++i)
      memcpy(storage.data() + offset + i * sizeof(values[i]), &values[i],
             sizeof(values[i]));
  }
  return offset;
}

size_t multilevel_region_points(const nvidia::flat_region &region) {
  size_t points = 1;
  for (int axis = 0; axis < 3; ++axis)
    points = checked_product(points, region.counts[axis],
                             "sizing NVIDIA multilevel region");
  return points;
}

ptrdiff_t checked_index_product(ptrdiff_t value, uint32_t multiplier, const char *what) {
  const __int128 product = __int128(value) * multiplier;
  if (product < std::numeric_limits<ptrdiff_t>::min() ||
      product > std::numeric_limits<ptrdiff_t>::max())
    throw std::overflow_error(std::string(what) + " index overflow");
  return ptrdiff_t(product);
}

void validate_multilevel_storage_identity(const StoragePlan &plan, ArrayId id,
                                          array_role role, int chunk, field_type ft,
                                          int state_index, component c, int cmp,
                                          uint32_t layout_ordinal,
                                          nvidia::scalar_precision precision,
                                          const char *what) {
  if (!is_valid(id) || id.value >= plan.arrays.size() || id.value >= plan.keys.size())
    throw std::out_of_range(std::string(what) + " ArrayId is out of range");
  const ArraySpec &spec = plan.arrays[id.value];
  const StorageKey &key = plan.keys[id.value];
  if (spec.id != id || spec.role != role || spec.element_type != ElementType::realnum_value ||
      is_valid(spec.alias_of) || key.chunk != chunk ||
      key.kind != int(array_kind::polarization_internal) || key.component_ != int(c) ||
      key.cmp != cmp || polarization_storage_field_type(key.aux) != ft ||
      polarization_storage_state_index(key.aux) != state_index ||
      polarization_storage_layout_ordinal(key.aux) != layout_ordinal ||
      scalar_precision_for(plan, id, what) != precision)
    throw std::invalid_argument(std::string(what) + " has a noncanonical storage identity");
}

nvidia::multilevel_population_launch compile_multilevel_population(
    const MultilevelPopulationUpdate &source, uint32_t transition_index, const StepPlan &plan,
    NvidiaBackendState &state,
    std::vector<nvidia::multilevel_population_term_launch> &compiled_terms,
    std::vector<unsigned char> &coefficient_storage, size_t &scratch_bytes) {
  if (source.region.c != Centered || source.region.cmp != -1 || source.region.variant_key ||
      source.region.chunk < 0 || source.state_index < 0 ||
      (source.ft != E_stuff && source.ft != H_stuff) || !source.levels || !source.transitions ||
      source.scratch_elements_per_point != source.levels || !std::isfinite(source.dt) ||
      source.dt <= 0.0 ||
      size_t(source.levels) > std::numeric_limits<size_t>::max() / source.levels ||
      size_t(source.levels) > std::numeric_limits<size_t>::max() / source.transitions ||
      source.gamma_count != size_t(source.levels) * source.levels ||
      source.alpha_count != size_t(source.levels) * source.transitions ||
      size_t(source.gamma_index) + source.gamma_count > plan.multilevel_coefficients.size() ||
      size_t(source.alpha_index) + source.alpha_count > plan.multilevel_coefficients.size() ||
      size_t(source.term_index) + source.term_count > plan.multilevel_population_terms.size())
    throw std::invalid_argument("NVIDIA multilevel population descriptor is noncanonical");

  nvidia::multilevel_population_launch result = {};
  result.region = flat_region_for(source.region);
  result.precision = scalar_precision_for(state.plan_, source.populations,
                                          "multilevel populations");
  require_same_precision(state.plan_, source.gamma_inv, result.precision,
                         "multilevel GammaInv");
  validate_multilevel_storage_identity(state.plan_, source.gamma_inv,
                                       array_role::polarization, source.region.chunk, source.ft,
                                       source.state_index, Centered, -1, 0, result.precision,
                                       "multilevel GammaInv");
  const size_t population_ordinal =
      checked_add(1, checked_product(checked_product(2, size_t(source.active_component_cmps),
                                                    "computing NVIDIA multilevel population ordinal"),
                                     size_t(source.transitions),
                                     "computing NVIDIA multilevel population ordinal"),
                  "computing NVIDIA multilevel population ordinal");
  if (population_ordinal > UINT32_MAX)
    throw std::overflow_error("NVIDIA multilevel population ordinal overflows");
  validate_multilevel_storage_identity(state.plan_, source.populations,
                                       array_role::polarization, source.region.chunk, source.ft,
                                       source.state_index, Centered, -1,
                                       uint32_t(population_ordinal), result.precision,
                                       "multilevel populations");
  const ArraySpec &gamma_spec = state.plan_.arrays[source.gamma_inv.value];
  const ArraySpec &population_spec = state.plan_.arrays[source.populations.value];
  if (gamma_spec.elements != size_t(source.levels) * source.levels)
    throw std::invalid_argument("NVIDIA multilevel GammaInv extent is noncanonical");
  const ptrdiff_t region_min = ptrdiff_t(result.region.base);
  const ptrdiff_t region_max = checked_region_max(result.region);
  const ptrdiff_t population_min = checked_index_product(region_min, source.levels,
                                                         "multilevel population");
  const ptrdiff_t population_max = checked_shift(
      checked_index_product(region_max, source.levels, "multilevel population"),
      ptrdiff_t(source.levels - 1), "multilevel population");
  validate_index_range(state.plan_, source.populations, population_min, population_max,
                       "multilevel populations");
  if (population_spec.elements < size_t(population_max) + 1)
    throw std::out_of_range("NVIDIA multilevel population extent is too small");
  result.populations = device_address(state, source.populations, "multilevel populations");
  result.gamma_inv = device_address(state, source.gamma_inv, "multilevel GammaInv");
  result.levels = source.levels;
  result.transitions = source.transitions;
  if (compiled_terms.size() > UINT32_MAX ||
      size_t(source.term_count) > size_t(UINT32_MAX) - compiled_terms.size())
    throw std::overflow_error("NVIDIA multilevel population term span overflows");
  result.term_index = uint32_t(compiled_terms.size());
  result.term_count = source.term_count;

  std::vector<double> gamma(source.gamma_count);
  const realnum dt2 = realnum(source.dt) / 2;
  for (uint32_t l1 = 0; l1 < source.levels; ++l1)
    for (uint32_t l2 = 0; l2 < source.levels; ++l2) {
      const realnum value =
          (l1 == l2 ? realnum(1) : realnum(0)) -
          realnum(plan.multilevel_coefficients[source.gamma_index + l1 * source.levels + l2]) *
              dt2;
      if (!std::isfinite(double(value)))
        throw std::invalid_argument("NVIDIA multilevel Gamma coefficient is nonfinite");
      gamma[l1 * source.levels + l2] = double(value);
    }
  std::vector<double> alpha(source.alpha_count);
  for (uint32_t i = 0; i < source.alpha_count; ++i) {
    const realnum value = realnum(plan.multilevel_coefficients[source.alpha_index + i]);
    if (!std::isfinite(double(value)))
      throw std::invalid_argument("NVIDIA multilevel alpha coefficient is nonfinite");
    alpha[i] = double(value);
  }
  std::vector<double> gperpdt(source.transitions, 0.0);
  std::vector<bool> have_transition(source.transitions, false);
  for (uint32_t i = 0; i < source.term_count; ++i) {
    const MultilevelPopulationTerm &term = plan.multilevel_population_terms[source.term_index + i];
    if (term.transition_index < 0 || uint32_t(term.transition_index) >= source.transitions)
      throw std::invalid_argument("NVIDIA multilevel population term has invalid transition");
    if (size_t(transition_index) + i >= plan.multilevel_transition_updates.size())
      throw std::out_of_range("NVIDIA multilevel transition span is out of range");
    const MultilevelTransitionUpdate &transition =
        plan.multilevel_transition_updates[transition_index + i];
    if (transition.transition_index != term.transition_index)
      throw std::invalid_argument("NVIDIA multilevel term/action ordering differs");
    const uint32_t t = uint32_t(term.transition_index);
    const realnum value = realnum(transition.gamma) * pi * realnum(source.dt);
    if (!std::isfinite(double(value)))
      throw std::invalid_argument("NVIDIA multilevel transition damping is nonfinite");
    if (have_transition[t] && gperpdt[t] != double(value))
      throw std::invalid_argument("NVIDIA multilevel transition damping differs by row");
    have_transition[t] = true;
    gperpdt[t] = double(value);

    nvidia::multilevel_population_term_launch compiled = {};
    compiled.transition_index = t;
    compiled.centered_offsets[0] = term.centered_offsets[0];
    compiled.centered_offsets[1] = term.centered_offsets[1];
    const ArrayId ids[] = {term.w, term.w_prev, term.p, term.p_prev};
    const char *names[] = {"multilevel W", "multilevel W_prev", "multilevel P",
                           "multilevel P_prev"};
    for (size_t operand = 0; operand < 4; ++operand) {
      require_same_precision(state.plan_, ids[operand], result.precision, names[operand]);
      const ptrdiff_t combined = checked_shift(term.centered_offsets[0],
                                               term.centered_offsets[1], names[operand]);
      validate_shifted_index_range(state.plan_, ids[operand], region_min, region_max, 0,
                                   term.centered_offsets[0], term.centered_offsets[1], combined,
                                   names[operand]);
    }
    if (!source.active_component_cmps ||
        size_t(source.term_count) !=
            size_t(source.active_component_cmps) * size_t(source.transitions))
      throw std::invalid_argument("NVIDIA multilevel population row count is noncanonical");
    const uint32_t row = i % source.active_component_cmps;
    const size_t expected_ordinal =
        1 + 2 * (size_t(row) * source.transitions + uint32_t(term.transition_index));
    if (expected_ordinal >= UINT32_MAX)
      throw std::overflow_error("NVIDIA multilevel transition ordinal overflows");
    const uint32_t expected_p_ordinal = uint32_t(expected_ordinal);
    validate_multilevel_storage_identity(state.plan_, term.p, array_role::polarization,
                                         source.region.chunk, source.ft, source.state_index,
                                         term.c, term.cmp, expected_p_ordinal, result.precision,
                                         names[2]);
    validate_multilevel_storage_identity(state.plan_, term.p_prev, array_role::polarization,
                                         source.region.chunk, source.ft, source.state_index,
                                         term.c, term.cmp, expected_p_ordinal + 1,
                                         result.precision, names[3]);
    for (size_t operand = 0; operand < 2; ++operand) {
      const ArraySpec &spec = state.plan_.arrays[ids[operand].value];
      const StorageKey &key = state.plan_.keys[ids[operand].value];
      const array_kind expected = operand == 0 ? array_kind::f_w : array_kind::f_w_prev;
      const bool primary_fallback = operand == 0 && key.kind == int(array_kind::f);
      if (spec.role != array_role::field || is_valid(spec.alias_of) ||
          key.chunk != source.region.chunk ||
          (!primary_fallback && key.kind != int(expected)) || key.component_ != int(term.c) ||
          key.cmp != term.cmp || key.aux != 0)
        throw std::invalid_argument(std::string(names[operand]) +
                                    " has a noncanonical storage identity");
    }
    compiled.w = device_address(state, term.w, names[0]);
    compiled.w_prev = device_address(state, term.w_prev, names[1]);
    compiled.p = device_address(state, term.p, names[2]);
    compiled.p_prev = device_address(state, term.p_prev, names[3]);
    compiled_terms.push_back(compiled);
  }
  result.gamma_byte_offset = append_multilevel_coefficients(coefficient_storage,
                                                            result.precision, gamma);
  result.alpha_byte_offset = append_multilevel_coefficients(coefficient_storage,
                                                            result.precision, alpha);
  result.transition_byte_offset = append_multilevel_coefficients(
      coefficient_storage, result.precision, gperpdt);
  const size_t scratch_elements = checked_product(multilevel_region_points(result.region),
                                                  source.levels,
                                                  "sizing NVIDIA multilevel scratch");
  const size_t bytes = result.precision == nvidia::scalar_precision::f32 ? sizeof(float)
                                                                         : sizeof(double);
  scratch_bytes = std::max(scratch_bytes,
                           checked_product(scratch_elements, bytes,
                                           "sizing NVIDIA multilevel scratch"));
  return result;
}

nvidia::multilevel_transition_launch compile_multilevel_transition(
    const MultilevelTransitionUpdate &source, uint32_t expected_p_ordinal,
    uint32_t expected_population_ordinal, NvidiaBackendState &state,
    std::vector<unsigned char> &coefficient_storage) {
  if (source.region.variant_key || source.region.chunk < 0 || source.state_index < 0 ||
      source.transition_index < 0 || (source.ft != E_stuff && source.ft != H_stuff) ||
      source.region.c == Centered || source.region.cmp < 0 || source.region.cmp > 1 ||
      source.population_stride == 0 || source.positive_level < 0 || source.negative_level < 0 ||
      uint32_t(source.positive_level) >= source.population_stride ||
      uint32_t(source.negative_level) >= source.population_stride ||
      source.positive_level == source.negative_level || !std::isfinite(source.omega) ||
      !std::isfinite(source.gamma) || !std::isfinite(source.dt) || source.dt <= 0.0)
    throw std::invalid_argument("NVIDIA multilevel transition descriptor is noncanonical");
  for (double value : source.sigmat)
    if (!std::isfinite(value))
      throw std::invalid_argument("NVIDIA multilevel transition coefficient is nonfinite");

  nvidia::multilevel_transition_launch result = {};
  result.region = flat_region_for(source.region);
  result.precision = scalar_precision_for(state.plan_, source.p, "multilevel P");
  const ArrayId ids[] = {source.p_prev, source.w, source.diagonal_sigma, source.populations};
  const char *names[] = {"multilevel P_prev", "multilevel W", "multilevel sigma",
                         "multilevel populations"};
  for (size_t i = 0; i < 4; ++i)
    require_same_precision(state.plan_, ids[i], result.precision, names[i]);
  result.p = device_address(state, source.p, "multilevel P");
  result.p_prev = device_address(state, source.p_prev, names[0]);
  result.w = device_address(state, source.w, names[1]);
  result.diagonal_sigma = device_address(state, source.diagonal_sigma, names[2]);
  result.populations = device_address(state, source.populations, names[3]);
  if (result.p == result.p_prev)
    throw std::invalid_argument("NVIDIA multilevel P and P_prev alias");
  validate_multilevel_storage_identity(state.plan_, source.p, array_role::polarization,
                                       source.region.chunk, source.ft, source.state_index,
                                       source.region.c, source.region.cmp, expected_p_ordinal,
                                       result.precision,
                                       "multilevel P");
  validate_multilevel_storage_identity(state.plan_, source.p_prev, array_role::polarization,
                                       source.region.chunk, source.ft, source.state_index,
                                       source.region.c, source.region.cmp,
                                       expected_p_ordinal + 1, result.precision,
                                       "multilevel P_prev");
  validate_multilevel_storage_identity(state.plan_, source.populations,
                                       array_role::polarization, source.region.chunk, source.ft,
                                       source.state_index, Centered, -1,
                                       expected_population_ordinal, result.precision,
                                       "multilevel populations");
  const ArraySpec &w_spec = state.plan_.arrays[source.w.value];
  const StorageKey &w_key = state.plan_.keys[source.w.value];
  if (w_spec.role != array_role::field || is_valid(w_spec.alias_of) ||
      w_key.chunk != source.region.chunk ||
      (w_key.kind != int(array_kind::f_w) && w_key.kind != int(array_kind::f)) ||
      w_key.component_ != int(source.region.c) || w_key.cmp != source.region.cmp || w_key.aux != 0)
    throw std::invalid_argument("NVIDIA multilevel W has a noncanonical storage identity");
  const ArraySpec &sigma_spec = state.plan_.arrays[source.diagonal_sigma.value];
  const StorageKey &sigma_key = state.plan_.keys[source.diagonal_sigma.value];
  if (source.state_index >
      (std::numeric_limits<int>::max() - int(source.ft)) / NUM_FIELD_TYPES)
    throw std::overflow_error("NVIDIA multilevel sigma identity overflows");
  const int sigma_aux = source.state_index * NUM_FIELD_TYPES + int(source.ft);
  if (sigma_spec.role != array_role::material || is_valid(sigma_spec.alias_of) ||
      sigma_key.chunk != source.region.chunk || sigma_key.kind != int(array_kind::sigma) ||
      sigma_key.component_ != int(source.region.c) ||
      sigma_key.cmp != int(component_direction(source.region.c)) ||
      sigma_key.aux != uint64_t(sigma_aux))
    throw std::invalid_argument("NVIDIA multilevel sigma has a noncanonical storage identity");
  result.population_stride = source.population_stride;
  result.population_offsets[0] = checked_index_product(
      source.population_offsets[0], result.population_stride,
      "scaling NVIDIA multilevel population offset");
  result.population_offsets[1] = checked_index_product(
      source.population_offsets[1], result.population_stride,
      "scaling NVIDIA multilevel population offset");
  result.positive_level = uint32_t(source.positive_level);
  result.negative_level = uint32_t(source.negative_level);

  const ptrdiff_t region_min = ptrdiff_t(result.region.base);
  const ptrdiff_t region_max = checked_region_max(result.region);
  validate_index_range(state.plan_, source.p, region_min, region_max, "multilevel P");
  validate_index_range(state.plan_, source.p_prev, region_min, region_max, names[0]);
  validate_index_range(state.plan_, source.w, region_min, region_max, names[1]);
  validate_index_range(state.plan_, source.diagonal_sigma, region_min, region_max, names[2]);
  const ptrdiff_t population_min = checked_index_product(region_min, source.population_stride,
                                                         "multilevel population");
  const ptrdiff_t population_max = checked_index_product(region_max, source.population_stride,
                                                         "multilevel population");
  const ptrdiff_t combined = checked_shift(result.population_offsets[0],
                                           result.population_offsets[1],
                                           "multilevel population");
  validate_shifted_index_range(state.plan_, source.populations, population_min, population_max,
                               0, result.population_offsets[0], result.population_offsets[1],
                               combined, names[3]);

  const realnum dt = realnum(source.dt);
  const realnum omega2pi = 2 * pi * realnum(source.omega);
  const realnum g2pi = realnum(source.gamma) * 2 * pi;
  const realnum gperp = realnum(source.gamma) * pi;
  const realnum corrected = omega2pi * omega2pi * dt * dt + gperp * gperp * dt * dt;
  const realnum gamma1inv = 1 / (1 + g2pi * dt / 2);
  const realnum gamma1 = 1 - g2pi * dt / 2;
  const realnum dtsqr = dt * dt;
  const realnum st = realnum(source.sigmat[component_direction(source.region.c)]);
  const std::vector<double> coefficients = {double(corrected), double(gamma1inv),
                                            double(gamma1), double(dtsqr), double(st)};
  for (double value : coefficients)
    if (!std::isfinite(value))
      throw std::invalid_argument("NVIDIA multilevel derived coefficient is nonfinite");
  result.coefficient_byte_offset = append_multilevel_coefficients(
      coefficient_storage, result.precision, coefficients);
  return result;
}

nvidia::polarization_subtract_launch compile_polarization_subtraction(
    const PolarizationSubtraction &source, const StepPlan &plan, NvidiaBackendState &state) {
  if (!is_valid(source.target) || !is_valid(source.p) || source.target == source.p ||
      !source.elements)
    throw std::invalid_argument("polarization subtraction has invalid operands");
  if (source.target.value >= state.plan_.arrays.size() || source.p.value >= state.plan_.arrays.size())
    throw std::out_of_range("polarization subtraction ArrayId is out of range");
  const ArraySpec &target_spec = state.plan_.arrays[source.target.value];
  const ArraySpec &p_spec = state.plan_.arrays[source.p.value];
  const StorageKey &target_key = state.plan_.keys[source.target.value];
  const StorageKey &p_key = state.plan_.keys[source.p.value];
  const field_type source_ft = type(source.c);
  const field_type target_ft = source_ft == E_stuff ? D_stuff : B_stuff;
  const component target_component = field_type_component(target_ft, source.c);
  if ((source_ft != E_stuff && source_ft != H_stuff) || source.chunk < 0 ||
      source.state_index < 0 || source.cmp < 0 || source.cmp > 1 ||
      source.elements != target_spec.elements || source.elements != p_spec.elements)
    throw std::out_of_range("polarization subtraction is not a full-array operation");
  if (target_spec.role != array_role::field || is_valid(target_spec.alias_of) ||
      target_key.chunk != source.chunk || target_key.kind != int(array_kind::f_minus_p) ||
      target_key.component_ != int(target_component) || target_key.cmp != source.cmp ||
      target_key.aux != 0 || p_spec.role != array_role::polarization ||
      is_valid(p_spec.alias_of) || p_key.chunk != source.chunk ||
      p_key.kind != int(array_kind::polarization_internal) ||
      p_key.component_ != int(source.c) || p_key.cmp != source.cmp ||
      polarization_storage_field_type(p_key.aux) != source_ft ||
      polarization_storage_state_index(p_key.aux) != source.state_index)
    throw std::invalid_argument("polarization subtraction has a noncanonical storage identity");
  if (source.transition_index >= 0) {
    const MultilevelPopulationUpdate *population = NULL;
    for (const MultilevelPopulationUpdate &candidate : plan.multilevel_population_updates)
      if (candidate.region.chunk == source.chunk && candidate.ft == source_ft &&
          candidate.state_index == source.state_index) {
        if (population)
          throw std::invalid_argument("multilevel subtraction has duplicate population owners");
        population = &candidate;
      }
    if (!population || uint32_t(source.transition_index) >= population->transitions)
      throw std::invalid_argument("multilevel subtraction has no matching population action");
    const uint32_t ordinal = polarization_storage_layout_ordinal(p_key.aux);
    if (!ordinal || ((ordinal - 1) / 2) % population->transitions !=
                        uint32_t(source.transition_index) ||
        (ordinal - 1) % 2)
      throw std::invalid_argument("multilevel subtraction P has the wrong transition ordinal");
  }
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

const ArraySpec &validate_legacy_flux_field(const NvidiaBackendState &state, ArrayId id,
                                            int chunk, component c, int cmp,
                                            const char *what) {
  if (!is_valid(id) || id.value >= state.plan_.arrays.size() ||
      id.value >= state.plan_.keys.size())
    throw std::out_of_range(std::string(what) + " uses an invalid ArrayId");
  const ArraySpec &spec = state.plan_.arrays[id.value];
  const StorageKey &key = state.plan_.keys[id.value];
  if (spec.role != array_role::field || spec.element_type != ElementType::realnum_value ||
      key.kind != int(array_kind::f) || key.chunk != chunk || key.component_ != int(c) ||
      key.cmp != cmp)
    throw std::invalid_argument(std::string(what) + " has the wrong field identity");
  return spec;
}

nvidia::legacy_flux_term_launch compile_legacy_flux_term(const LegacyFluxTerm &source,
                                                          const fields &f,
                                                          NvidiaBackendState &state) {
  if (source.chunk < 0 || source.chunk >= f.num_chunks || !f.chunks[source.chunk] ||
      !f.chunks[source.chunk]->is_mine())
    throw std::invalid_argument("legacy flux term has an invalid local chunk");
  if (source.sign != 1 && source.sign != -1)
    throw std::invalid_argument("legacy flux term has an invalid sign");
  if (!std::isfinite(source.phase_real) || !std::isfinite(source.phase_imag) ||
      !std::isfinite(source.dV0) || !std::isfinite(source.dV1))
    throw std::invalid_argument("legacy flux term has a nonfinite coefficient");

  nvidia::legacy_flux_term_launch result = {};
  result.region.base = source.base;
  size_t points = 1;
  for (int axis = 0; axis < 3; ++axis) {
    if (!source.counts[axis] || source.strides[axis] < 0)
      throw std::invalid_argument("legacy flux term has invalid region geometry");
    result.region.counts[axis] = source.counts[axis];
    result.region.strides[axis] = source.strides[axis];
    points = checked_product(points, source.counts[axis],
                             "sizing NVIDIA legacy flux region");
    result.start0[axis] = source.boundary_weights[axis][0];
    result.start1[axis] = source.boundary_weights[axis][1];
    result.end0[axis] = source.boundary_weights[axis][2];
    result.end1[axis] = source.boundary_weights[axis][3];
    for (int endpoint = 0; endpoint < 4; ++endpoint)
      if (!std::isfinite(source.boundary_weights[axis][endpoint]))
        throw std::invalid_argument("legacy flux term has a nonfinite boundary weight");
  }
  result.points = points;
  result.blocks = nvidia::legacy_flux_partial_count(points);
  result.phase_real = source.phase_real;
  result.phase_imag = source.phase_imag;
  result.dV0 = source.dV0;
  result.dV1 = source.dV1;
  result.sign = source.sign;
  for (int i = 0; i < 2; ++i) {
    result.e_offsets[i] = source.e_offsets[i];
    result.h_offsets[i] = source.h_offsets[i];
  }

  if ((!is_valid(source.e_real) && is_valid(source.e_imag)) ||
      (!is_valid(source.h_real) && is_valid(source.h_imag)))
    throw std::invalid_argument("legacy flux imaginary field has no real counterpart");
  if (f.is_real && (is_valid(source.e_imag) || is_valid(source.h_imag)))
    throw std::invalid_argument("real legacy flux term has imaginary field storage");
  if (!f.is_real && ((is_valid(source.e_real) && !is_valid(source.e_imag)) ||
                     (is_valid(source.h_real) && !is_valid(source.h_imag))))
    throw std::invalid_argument("complex legacy flux term lacks imaginary field storage");

  bool have_precision = false;
  nvidia::scalar_precision precision = nvidia::scalar_precision::f64;
  const auto bind = [&](ArrayId id, component c, int cmp, const char *what) -> const void * {
    if (!is_valid(id)) return NULL;
    (void)validate_legacy_flux_field(state, id, source.chunk, c, cmp, what);
    const nvidia::scalar_precision current = scalar_precision_for(state.plan_, id, what);
    if (have_precision && current != precision)
      throw std::invalid_argument("legacy flux term mixes field storage precisions");
    precision = current;
    have_precision = true;
    return device_address(state, id, what);
  };
  result.e_real = bind(source.e_real, source.e_component, 0, "legacy flux E real field");
  result.e_imag = bind(source.e_imag, source.e_component, 1, "legacy flux E imaginary field");
  result.h_real = bind(source.h_real, source.h_component, 0, "legacy flux H real field");
  result.h_imag = bind(source.h_imag, source.h_component, 1, "legacy flux H imaginary field");
  result.precision = precision;

  const ptrdiff_t region_min = source.base > size_t(std::numeric_limits<ptrdiff_t>::max())
                                   ? throw std::overflow_error(
                                         "legacy flux term base exceeds ptrdiff_t")
                                   : ptrdiff_t(source.base);
  const ptrdiff_t region_max = checked_region_max(result.region);
  const ptrdiff_t e_both = checked_shift(source.e_offsets[0], source.e_offsets[1],
                                         "validating legacy flux E interpolation");
  const ptrdiff_t h_both = checked_shift(source.h_offsets[0], source.h_offsets[1],
                                         "validating legacy flux H interpolation");
  if (is_valid(source.e_real)) {
    validate_shifted_index_range(state.plan_, source.e_real, region_min, region_max, 0,
                                 source.e_offsets[0], source.e_offsets[1], e_both,
                                 "legacy flux E real field");
    if (is_valid(source.e_imag))
      validate_shifted_index_range(state.plan_, source.e_imag, region_min, region_max, 0,
                                   source.e_offsets[0], source.e_offsets[1], e_both,
                                   "legacy flux E imaginary field");
  }
  if (is_valid(source.h_real)) {
    validate_shifted_index_range(state.plan_, source.h_real, region_min, region_max, 0,
                                 source.h_offsets[0], source.h_offsets[1], h_both,
                                 "legacy flux H real field");
    if (is_valid(source.h_imag))
      validate_shifted_index_range(state.plan_, source.h_imag, region_min, region_max, 0,
                                   source.h_offsets[0], source.h_offsets[1], h_both,
                                   "legacy flux H imaginary field");
  }
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
                                      constitutive_axis_override |
                                      constitutive_copy_w_previous;
  if (source.region.variant_key & ~supported_variants)
    throw std::invalid_argument("constitutive descriptor requires unsupported auxiliary state");

  const bool have_pml = (source.region.variant_key & constitutive_has_pml) != 0;
  const bool copy_previous =
      (source.region.variant_key & constitutive_copy_w_previous) != 0;
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
  if (have_pml != is_valid(source.pml.sig) ||
      (have_pml && !is_valid(source.target_w)) ||
      (!have_pml && is_valid(source.target_w) && !copy_previous) ||
      copy_previous != is_valid(source.previous_w))
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
  result.previous_w = optional_mutable_device_address(
      state, source.previous_w, result.precision, "constitutive previous W");
  const ArrayId previous_source = is_valid(source.target_w) ? source.target_w : source.target;
  result.previous_w_source = copy_previous
                                 ? device_address(state, previous_source,
                                                  "constitutive previous-W source")
                                 : NULL;
  result.previous_w_elements = 0;
  if (copy_previous) {
    const ArraySpec &previous_spec = state.plan_.arrays[source.previous_w.value];
    const ArraySpec &source_spec = state.plan_.arrays[previous_source.value];
    require_same_precision(state.plan_, source.previous_w, result.precision,
                           "constitutive previous W");
    if (previous_spec.role != array_role::field || source_spec.role != array_role::field ||
        is_valid(previous_spec.alias_of) || previous_spec.elements != source_spec.elements)
      throw std::invalid_argument("constitutive previous-W arrays have incompatible storage");
    result.previous_w_elements = previous_spec.elements;
  }
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
  if (source.storage != HaloStorageDisposition::canonical)
    throw std::invalid_argument("NVIDIA cannot lower a host-owned halo plan");

  std::vector<ElementRef> gather_refs, scatter_refs;
  expand_gather(source, gather_refs);
  expand_scatter(source, scatter_refs);
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

bool has_live_host_custom(const fields &f) {
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk] || !f.chunks[chunk]->is_mine()) continue;
    FOR_FIELD_TYPES(ft) for (const polarization_state *p = f.chunks[chunk]->pol[ft]; p;
                            p = p->next)
      if (p->s && typeid(*p->s) != typeid(noisy_lorentzian_susceptibility) &&
          typeid(*p->s) != typeid(gyrotropic_susceptibility) &&
          typeid(*p->s) != typeid(multilevel_susceptibility) &&
          typeid(*p->s) != typeid(lorentzian_susceptibility))
        return true;
  }
  return false;
}

bool has_descriptor_host_custom(const fields &f) {
  if (!f.descriptors) return false;
  for (const PolarizationDescriptor &descriptor : f.descriptors->polarizations)
    if (descriptor.kind == SusceptibilityKind::host_custom) return true;
  return false;
}

bool has_plan_host_custom(const StepPlan &plan) {
  bool marker = false;
  for (const Operation &operation : plan.operations)
    marker = marker || operation.kind == OpKind::host_callback;
  return marker || !plan.host_segments.empty() || !plan.host_callbacks.empty();
}

void set_reason(std::string &why, size_t operation, const char *detail) {
  std::ostringstream message;
  message << "NVIDIA PR2 unsupported operation at index " << operation << ": " << detail;
  why = message.str();
}

GraphExecutionMode parse_graph_mode_locally(const char *value, bool &valid) noexcept {
  valid = true;
  if (!value || !*value || strcmp(value, "auto") == 0)
    return GraphExecutionMode::automatic;
  if (strcmp(value, "eager") == 0) return GraphExecutionMode::eager;
  if (strcmp(value, "required") == 0) return GraphExecutionMode::required;
  valid = false;
  return GraphExecutionMode::automatic;
}

void reconcile_graph_stage(const std::string &local_error, const char *remote_error) {
  const bool any_error = or_to_all(!local_error.empty());
  if (!any_error) return;
  throw std::runtime_error(local_error.empty() ? remote_error : local_error);
}

GraphExecutionMode reconcile_graph_mode(GraphExecutionMode local_mode,
                                        bool local_parse_valid) {
  const bool any_parse_error = or_to_all(!local_parse_valid);
  if (any_parse_error)
    throw std::invalid_argument(
        local_parse_valid ? "NVIDIA graph mode was invalid on another rank"
                          : "invalid NVIDIA graph mode (expected auto, eager, or required)");
  const int requested = int(local_mode);
  const int minimum = min_to_all(requested);
  const int maximum = max_to_all(requested);
  if (minimum != maximum)
    throw std::invalid_argument("NVIDIA graph execution mode differs across ranks");
  return local_mode;
}

bool reconcile_graph_support(GraphExecutionMode requested, bool local_supported) {
  const bool all_supported = and_to_all(local_supported);
  const std::vector<GraphRankModeSupport> collective_support(
      1, GraphRankModeSupport{requested, all_supported, all_supported});
  return resolve_collective_graph_execution_mode(collective_support).use_graph;
}

bool reconcile_graph_attempt(GraphExecutionMode requested, const std::string &local_error,
                             const char *remote_error) {
  const bool any_error = or_to_all(!local_error.empty());
  if (!any_error) return true;
  if (requested == GraphExecutionMode::required)
    throw std::runtime_error(local_error.empty() ? remote_error : local_error);
  return false;
}

} // namespace

NvidiaBackend::NvidiaBackend(fields &f, const execution_options &options, int selected_device)
    : f_(f), options_(options), device_(selected_device),
      graph_mode_(GraphExecutionMode::automatic), graph_mode_parse_valid_(true),
      device_memory_bytes_(0),
      next_state_token_(1), next_executable_generation_(0),
      pending_initialization_reserve_bytes_(0),
      pending_initialization_compact_bytes_(0), pending_initialization_scratch_bytes_(0),
      pending_initialization_route_(MaterialRecipeDisposition::device_native),
      pending_initialization_reserve_valid_(false) {
  graph_mode_ =
      parse_graph_mode_locally(std::getenv("MEEP_NVIDIA_GRAPH_MODE"), graph_mode_parse_valid_);
  if (device_ < 0) throw std::invalid_argument("NVIDIA backend requires a resolved device ID");
  device_memory_bytes_ = nvidia::properties_for_device(device_).total_memory;
}

NvidiaBackend::~NvidiaBackend() {}

uint64_t NvidiaBackend::claim_executable_generation() {
  if (next_executable_generation_ == std::numeric_limits<uint64_t>::max())
    throw std::overflow_error("NVIDIA executable resource generation overflow");
  return ++next_executable_generation_;
}

namespace nvidia {
namespace testing {

GraphModeResolution reconcile_graph_execution_for_testing(
    const graph_collective_probe &probe) {
  bool parse_valid = true;
  const GraphExecutionMode local_mode = parse_graph_mode_locally(probe.mode, parse_valid);
  const GraphExecutionMode requested = reconcile_graph_mode(local_mode, parse_valid);
  reconcile_graph_stage(probe.lowering_valid ? std::string()
                                             : "injected graph lowering failure",
                        "graph lowering failed on another rank");
  reconcile_graph_stage(probe.validation_valid ? std::string()
                                               : "injected graph validation failure",
                        "graph validation failed on another rank");
  if (!reconcile_graph_support(
          requested, probe.runtime_capture_supported && probe.program_graphable))
    return GraphModeResolution{requested, false};
  const auto reconcile_attempt = [&](bool local_valid, const char *local_failure,
                                     const char *remote_failure) {
    const bool any_failure = or_to_all(!local_valid);
    if (!any_failure) return true;
    std::string cleanup_error;
    if (!probe.graph_exec_destroy_valid)
      cleanup_error = "injected CUDA graph executable cleanup failure";
    if (!probe.graph_destroy_valid && cleanup_error.empty())
      cleanup_error = "injected CUDA graph definition cleanup failure";
    if (!probe.graph_device_restore_valid && cleanup_error.empty())
      cleanup_error = "injected CUDA graph cleanup device-restoration failure";
    reconcile_graph_stage(cleanup_error, "CUDA graph cleanup failed on another rank");
    return reconcile_graph_attempt(requested,
                                   local_valid ? std::string() : local_failure,
                                   remote_failure);
  };
  if (!reconcile_attempt(probe.allocation_valid,
                         "injected graph scalar allocation failure",
                         "graph scalar allocation failed on another rank"))
    return GraphModeResolution{requested, false};
  if (!reconcile_attempt(probe.capture_valid, "injected graph capture failure",
                         "graph capture failed on another rank"))
    return GraphModeResolution{requested, false};
  if (!reconcile_attempt(probe.instantiate_valid,
                         "injected graph instantiate failure",
                         "graph instantiate failed on another rank"))
    return GraphModeResolution{requested, false};
  return GraphModeResolution{requested, true};
}

} // namespace testing
} // namespace nvidia

namespace {
void preflight_native_table_ir(const MaterialIR &ir);
void preflight_geometry_ir(const MaterialIR &ir);
size_t exact_material_compact_input_bytes(const MaterialIR &ir);

std::vector<std::pair<StorageKey, size_t> >
logical_material_classification_rows(const MaterialRecipe &recipe) {
  std::vector<std::pair<StorageKey, size_t> > result;
  const MaterialIR *const ir = recipe.ir().get();
  std::vector<MaterialIRTopologyRow> rows;
  if (ir)
    rows = ir->topology;
  else {
    rows.reserve(recipe.rows().size() + recipe.topology().size());
    for (const MaterialRecipeRow &row : recipe.rows()) {
      MaterialIRTopologyRow topology = {};
      topology.key = row.key;
      topology.element_type = row.element_type;
      topology.logical_storage = row.storage;
      topology.elements = row.elements;
      topology.alignment = row.alignment;
      rows.push_back(topology);
    }
    rows.insert(rows.end(), recipe.topology().begin(), recipe.topology().end());
  }
  result.reserve(rows.size());
  for (const MaterialIRTopologyRow &row : rows) {
    size_t logical_elements = 0;
    if (ir) {
      const array_kind kind = static_cast<array_kind>(row.key.kind);
      if (kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
          kind == array_kind::pml_siginv) {
        size_t matches = 0;
        for (const MaterialIRPmlAxis &axis : ir->pml_axes)
          if (axis.chunk == row.key.chunk && uint64_t(axis.direction) == row.key.aux) {
            logical_elements = axis.elements;
            ++matches;
          }
        if (matches != 1)
          throw std::invalid_argument(
              "NVIDIA PML classification row has ambiguous logical extent");
      }
      else {
        size_t matches = 0;
        for (const MaterialIRDestination &destination : ir->destinations)
          if (destination.key == row.key) {
            if (destination.point_count > uint64_t(std::numeric_limits<size_t>::max()))
              throw std::overflow_error(
                  "NVIDIA material classification logical extent overflows size_t");
            logical_elements = size_t(destination.point_count);
            ++matches;
          }
        if (matches != 1)
          throw std::invalid_argument(
              "NVIDIA material classification row has ambiguous logical destination");
      }
    }
    else
      logical_elements = row.elements;
    if (!logical_elements || logical_elements > row.elements)
      throw std::invalid_argument(
          "NVIDIA material classification logical span exceeds physical storage");
    result.push_back(std::make_pair(row.key, logical_elements));
  }
  return result;
}

size_t logical_material_classification_elements(const MaterialIR &ir,
                                                const StorageKey &key) {
  const array_kind kind = static_cast<array_kind>(key.kind);
  if (kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
      kind == array_kind::pml_siginv) {
    for (const MaterialIRPmlAxis &axis : ir.pml_axes)
      if (axis.chunk == key.chunk && uint64_t(axis.direction) == key.aux)
        return axis.elements;
  }
  else
    for (const MaterialIRDestination &destination : ir.destinations)
      if (destination.key == key) {
        if (destination.point_count > uint64_t(std::numeric_limits<size_t>::max()))
          throw std::overflow_error(
              "NVIDIA material classification logical extent overflows size_t");
        return size_t(destination.point_count);
      }
  throw std::invalid_argument(
      "NVIDIA material classification has no logical span for storage key");
}

size_t logical_material_storage_index(const MaterialRecipe &recipe,
                                      const StorageKey &key, size_t point) {
  const MaterialIR *const ir = recipe.ir().get();
  if (!ir) return point;
  const array_kind kind = static_cast<array_kind>(key.kind);
  if (kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
      kind == array_kind::pml_siginv)
    return point;
  const MaterialIRDestination *match = NULL;
  for (const MaterialIRDestination &destination : ir->destinations)
    if (destination.key == key) {
      if (match)
        throw std::invalid_argument(
            "NVIDIA material classification has an ambiguous logical destination");
      match = &destination;
    }
  if (!match)
    throw std::invalid_argument(
        "NVIDIA material classification has no logical destination");
  return material_ir_destination_storage_index(*ir, *match, point);
}
}

void NvidiaBackend::preflight_initialization(const InitializationPlan &initialization) const {
  pending_initialization_reserve_valid_ = false;
  pending_initialization_reserve_bytes_ = 0;
  pending_initialization_compact_bytes_ = 0;
  pending_initialization_scratch_bytes_ = 0;
  pending_initialization_classification_rows_.clear();
  pending_initialization_route_ = MaterialRecipeDisposition::device_native;
  if (initialization.materials.size() != 1)
    throw std::invalid_argument("NVIDIA initialization requires one frozen material recipe");
  const MaterialRecipe &recipe = initialization.materials[0];
  validate_material_recipe(recipe);
  if (initialization.regional) {
    if (!initialization.regional_supported)
      throw std::invalid_argument(std::string("unsupported regional initialization: ") +
                                  regional_support_reason_name(initialization.regional_reason));
    if (recipe.disposition() != MaterialRecipeDisposition::host_reference)
      throw std::invalid_argument("regional initialization requires clipped host row kernels");
  }
  pending_initialization_classification_rows_ =
      logical_material_classification_rows(recipe);
  const MaterialSupportDecision support = classify_material_support(recipe);
  pending_initialization_route_ = recipe.disposition();
  if (support.scratch_bytes > uint64_t(std::numeric_limits<size_t>::max()))
    throw std::overflow_error("NVIDIA initialization callback scratch budget overflows size_t");
  pending_initialization_scratch_bytes_ = size_t(support.scratch_bytes);
  if (support.compact_input_bytes > uint64_t(std::numeric_limits<size_t>::max()))
    throw std::overflow_error("NVIDIA initialization compact-input budget overflows size_t");
  size_t compact_bytes = 0;
  const size_t fixed_bytes = 2 * sizeof(nvidia::noisy_seed_block);
  if (compact_bytes > std::numeric_limits<size_t>::max() - fixed_bytes)
    throw std::overflow_error("NVIDIA initialization auxiliary-memory budget overflows");
  pending_initialization_reserve_bytes_ = compact_bytes + fixed_bytes;
  if (recipe.disposition() != MaterialRecipeDisposition::device_native &&
      recipe.disposition() != MaterialRecipeDisposition::hybrid_interface) {
    pending_initialization_reserve_valid_ = true;
    return;
  }
  if (!recipe.ir())
    throw std::invalid_argument("NVIDIA native material initialization has no owned IR");
  const MaterialIR &ir = *recipe.ir();
  validate_material_ir(ir);
  const MaterialIR *const canonical = material_ir_for(f_);
  if (!canonical || !material_ir_equal(ir, *canonical))
    throw std::invalid_argument(
        "NVIDIA material IR differs from the live canonical owned snapshot");
  if (ir.default_material >= ir.materials.size())
    throw std::invalid_argument("NVIDIA native material root is absent");
  if (pending_initialization_classification_rows_.empty()) {
    pending_initialization_reserve_bytes_ = fixed_bytes;
    pending_initialization_compact_bytes_ = 0;
    pending_initialization_reserve_valid_ = true;
    return;
  }
  const int kind = ir.materials[ir.default_material].kind;
  if (kind != meep_geom::material_data::MEDIUM &&
      kind != meep_geom::material_data::PERFECT_METAL &&
      kind != meep_geom::material_data::MATERIAL_FILE &&
      kind != meep_geom::material_data::MATERIAL_GRID)
    throw std::invalid_argument("NVIDIA native material opcode is unsupported");
  if (ir.objects.empty() && ir.analytic_interfaces.empty() && ir.hybrid_patches.empty())
    preflight_native_table_ir(ir);
  else
    preflight_geometry_ir(ir);
  compact_bytes = pending_initialization_classification_rows_.empty()
                      ? 0
                      : exact_material_compact_input_bytes(ir);
  if (compact_bytes > std::numeric_limits<size_t>::max() - fixed_bytes)
    throw std::overflow_error("NVIDIA initialization auxiliary-memory budget overflows");
  pending_initialization_reserve_bytes_ = compact_bytes + fixed_bytes;
  pending_initialization_compact_bytes_ = compact_bytes;
  pending_initialization_reserve_valid_ = true;
}

void NvidiaBackend::validate_host_custom_rebuild() {
  if (options_.precision != precision_policy_kind::native)
    throw std::invalid_argument("NVIDIA host-custom fallback requires native precision");
  if (f_.phasein_time > 0)
    throw std::invalid_argument("NVIDIA host-custom fallback does not support material phasing");
}

void NvidiaBackend::validate_host_custom_plan(const StepPlan &plan, BackendState &raw_state) {
  (void)checked_state(raw_state);
  validate_host_custom_rebuild();
  if (plan.program != StepProgram::ordinary || !has_plan_host_custom(plan))
    throw std::invalid_argument("NVIDIA host-custom fallback requires an ordinary host-segment plan");
}

void NvidiaBackend::preflight_host_custom_fallback(Executable &raw_executable,
                                                   BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  NvidiaExecutable &executable = checked_executable(raw_executable, state);
  if (!state.initialized_ || state.transfer_failed_)
    throw std::logic_error("NVIDIA host-custom fallback state is unavailable");
  if (executable.program_ != StepProgram::ordinary ||
      executable.storage_fingerprint_ != state.fingerprint_)
    throw std::logic_error("NVIDIA host-custom fallback executable is stale");
  if (!f_.step_plans[0] || f_.step_plans[0]->signature != executable.signature_ ||
      compute_step_plan_signature(*f_.step_plans[0]) != executable.signature_)
    throw std::logic_error("NVIDIA host-custom fallback plan signature is stale");

  const StepPlan &plan = *f_.step_plans[0];
  if (plan.host_segments.size() != executable.host_segments_.size() ||
      plan.host_segments.empty())
    throw std::logic_error("NVIDIA host-custom fallback segment schedule changed");
  for (size_t si = 0; si < executable.host_segments_.size(); ++si) {
    const NvidiaCompiledHostSegment &compiled = executable.host_segments_[si];
    const HostSegment &segment = plan.host_segments[si];
    if (compiled.segment != segment ||
        size_t(segment.callback_index) + segment.callback_count > plan.host_callbacks.size() ||
        compiled.callbacks.size() != segment.callback_count ||
        state.staging_.size() < compiled.staging_bytes)
      throw std::logic_error("NVIDIA host-custom fallback segment identity changed");
    for (size_t ci = 0; ci < compiled.callbacks.size(); ++ci)
      if (compiled.callbacks[ci] != plan.host_callbacks[size_t(segment.callback_index) + ci])
        throw std::logic_error("NVIDIA host-custom fallback callback identity changed");
  }
}

namespace {

struct OwnedSusceptibilitySpan {
  size_t begin;
  size_t end;
};

struct OwnedMediumView {
  const std::vector<double> *values;
  size_t base;
  size_t end;
  std::vector<OwnedSusceptibilitySpan> susceptibilities[2];
};

size_t material_checked_add(size_t left, size_t right, const char *what) {
  if (right > std::numeric_limits<size_t>::max() - left)
    throw std::overflow_error(std::string("NVIDIA material initialization overflow while ") +
                              what);
  return left + right;
}

size_t material_checked_product(size_t left, size_t right, const char *what) {
  if (left && right > std::numeric_limits<size_t>::max() / left)
    throw std::overflow_error(std::string("NVIDIA material initialization overflow while ") +
                              what);
  return left * right;
}

size_t owned_count(double value, const char *what) {
  if (!std::isfinite(value) || value < 0 || std::floor(value) != value ||
      value > double(std::numeric_limits<size_t>::max()))
    throw std::invalid_argument(std::string("NVIDIA material IR has an invalid ") + what);
  return size_t(value);
}

uint32_t material_checked_uint32(uint64_t value, const char *what) {
  if (value > UINT32_MAX)
    throw std::overflow_error(std::string("NVIDIA material initialization overflow while ") +
                              what);
  return uint32_t(value);
}

OwnedSusceptibilitySpan parse_owned_susceptibility(const std::vector<double> &values,
                                                   size_t &offset) {
  const size_t begin = offset;
  if (offset > values.size() || values.size() - offset < 17)
    throw std::invalid_argument("NVIDIA material susceptibility payload is short");
  const size_t transitions = owned_count(values[offset + 16], "transition count");
  offset += 17;
  const size_t transition_values =
      material_checked_product(transitions, size_t(9), "reading transition values");
  if (transition_values > values.size() - offset)
    throw std::invalid_argument("NVIDIA material susceptibility transition payload is short");
  offset += transition_values;
  if (offset >= values.size())
    throw std::invalid_argument("NVIDIA material susceptibility populations are missing");
  const size_t populations = owned_count(values[offset++], "population count");
  if (populations > values.size() - offset)
    throw std::invalid_argument("NVIDIA material susceptibility population payload is short");
  offset += populations;
  return OwnedSusceptibilitySpan{begin, offset};
}

OwnedMediumView parse_owned_medium_at(const std::vector<double> &values, size_t &offset) {
  if (offset > values.size() || values.size() - offset < 38)
    throw std::invalid_argument("NVIDIA material medium payload is short");
  OwnedMediumView result;
  result.values = &values;
  result.base = offset;
  offset += 36;
  for (int ft = 0; ft < 2; ++ft) {
    if (offset >= values.size())
      throw std::invalid_argument("NVIDIA material medium susceptibility count is missing");
    const size_t count = owned_count(values[offset++], "susceptibility count");
    result.susceptibilities[ft].reserve(count);
    for (size_t i = 0; i < count; ++i)
      result.susceptibilities[ft].push_back(parse_owned_susceptibility(values, offset));
  }
  result.end = offset;
  return result;
}

OwnedMediumView parse_owned_medium(const MaterialIRMaterial &material) {
  if (material.kind != meep_geom::material_data::MEDIUM)
    throw std::invalid_argument("NVIDIA homogeneous initialization requires MEDIUM");
  size_t offset = 0;
  OwnedMediumView result = parse_owned_medium_at(material.parameters, offset);
  if (offset != material.parameters.size())
    throw std::invalid_argument("NVIDIA homogeneous medium payload has trailing values");
  return result;
}

void validate_table_medium_source(const OwnedMediumView &medium) {
  const std::vector<double> &p = *medium.values;
  for (int i = 0; i < 3; ++i)
    if (p[medium.base + 4 + 2 * i] != 0.0)
      throw std::invalid_argument(
          "NVIDIA table material has a non-real electric offdiagonal");
}

void preflight_native_table_ir(const MaterialIR &ir) {
  const MaterialIRMaterial &root = ir.materials[ir.default_material];
  if (root.kind != meep_geom::material_data::MATERIAL_FILE &&
      root.kind != meep_geom::material_data::MATERIAL_GRID)
    return;
  if (!ir.objects.empty() || root.host_callback ||
      (root.kind == meep_geom::material_data::MATERIAL_GRID && root.do_averaging))
    throw std::invalid_argument("NVIDIA table material route is not object-free native");
  size_t offset = root.kind == meep_geom::material_data::MATERIAL_FILE ? 0 : 3;
  const OwnedMediumView first = parse_owned_medium_at(root.parameters, offset);
  if (root.kind == meep_geom::material_data::MATERIAL_GRID) {
    const OwnedMediumView second = parse_owned_medium_at(root.parameters, offset);
    validate_table_medium_source(first);
    validate_table_medium_source(second);
    if (root.parameters.size() - offset != 3)
      throw std::invalid_argument("NVIDIA MaterialGrid payload tail is invalid");
  }
  else if (root.parameters.size() - offset != 3)
    throw std::invalid_argument("NVIDIA FILE payload tail is invalid");
  size_t product = 1;
  for (int axis = 0; axis < 3; ++axis) {
    const size_t extent = owned_count(
        root.parameters[(root.kind == meep_geom::material_data::MATERIAL_FILE ? offset : 0) +
                        axis],
        "table dimension");
    if (!extent || extent > size_t(INT_MAX))
      throw std::invalid_argument("NVIDIA material table dimension is not positive int-sized");
    product = material_checked_product(product, extent, "validating material table samples");
  }
  if (product != root.samples.size())
    throw std::invalid_argument("NVIDIA material table sample count is inconsistent");
}

struct SymmetricMaterialTensor {
  double value[3][3];
};

SymmetricMaterialTensor inverse_owned_tensor(const OwnedMediumView &medium, field_type ft) {
  const std::vector<double> &p = *medium.values;
  const size_t diagonal = medium.base + (ft == E_stuff ? 0 : 9);
  const size_t offdiagonal = medium.base + (ft == E_stuff ? 3 : 12);
  if (p[offdiagonal + 1] != 0.0 || p[offdiagonal + 3] != 0.0 ||
      p[offdiagonal + 5] != 0.0)
    throw std::invalid_argument(
        "NVIDIA native initialization does not support imaginary material offdiagonals");
  const double m00 = p[diagonal], m11 = p[diagonal + 1], m22 = p[diagonal + 2];
  const double m01 = p[offdiagonal], m02 = p[offdiagonal + 2],
               m12 = p[offdiagonal + 4];
  SymmetricMaterialTensor result = {};
  const bool diagonal_tensor = m01 == 0.0 && m02 == 0.0 && m12 == 0.0;
  if (diagonal_tensor) {
    result.value[0][0] = 1.0 / m00;
    result.value[1][1] = 1.0 / m11;
    result.value[2][2] = 1.0 / m22;
  }
  else {
    double detinv = m00 * m11 * m22 - m02 * m11 * m02 +
                    2.0 * m01 * m12 * m02 - m01 * m01 * m22 -
                    m12 * m12 * m00;
    if (detinv == 0.0)
      throw std::invalid_argument("NVIDIA homogeneous material tensor is singular");
    detinv = 1.0 / detinv;
    result.value[0][0] = detinv * (m11 * m22 - m12 * m12);
    result.value[1][1] = detinv * (m00 * m22 - m02 * m02);
    result.value[2][2] = detinv * (m11 * m00 - m01 * m01);
    result.value[0][2] = result.value[2][0] = detinv * (m01 * m12 - m11 * m02);
    result.value[0][1] = result.value[1][0] = detinv * (m12 * m02 - m01 * m22);
    result.value[1][2] = result.value[2][1] = detinv * (m01 * m02 - m00 * m12);
  }
  for (int i = 0; !diagonal_tensor && i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (!std::isfinite(result.value[i][j]))
        throw std::invalid_argument("NVIDIA homogeneous tensor inverse is non-finite");
  return result;
}

SymmetricMaterialTensor inverse_callback_tensor(const meep_geom::medium_struct &medium,
                                                field_type ft) {
  const vector3 &diagonal = ft == E_stuff ? medium.epsilon_diag : medium.mu_diag;
  const cvector3 &offdiagonal = ft == E_stuff ? medium.epsilon_offdiag : medium.mu_offdiag;
  if (offdiagonal.x.im != 0.0 || offdiagonal.y.im != 0.0 || offdiagonal.z.im != 0.0)
    throw std::invalid_argument(
        "owned tiled material callback returned an imaginary tensor offdiagonal");
  const double m00 = diagonal.x, m11 = diagonal.y, m22 = diagonal.z;
  const double m01 = offdiagonal.x.re, m02 = offdiagonal.y.re, m12 = offdiagonal.z.re;
  SymmetricMaterialTensor result = {};
  const bool diagonal_tensor = m01 == 0.0 && m02 == 0.0 && m12 == 0.0;
  if (diagonal_tensor) {
    result.value[0][0] = 1.0 / m00;
    result.value[1][1] = 1.0 / m11;
    result.value[2][2] = 1.0 / m22;
  }
  else {
    double determinant = m00 * m11 * m22 - m02 * m11 * m02 +
                         2.0 * m01 * m12 * m02 - m01 * m01 * m22 -
                         m12 * m12 * m00;
    if (determinant == 0.0)
      throw std::invalid_argument("owned tiled material callback returned a singular tensor");
    const double detinv = 1.0 / determinant;
    result.value[0][0] = detinv * (m11 * m22 - m12 * m12);
    result.value[1][1] = detinv * (m00 * m22 - m02 * m02);
    result.value[2][2] = detinv * (m11 * m00 - m01 * m01);
    result.value[0][2] = result.value[2][0] = detinv * (m01 * m12 - m11 * m02);
    result.value[0][1] = result.value[1][0] = detinv * (m12 * m02 - m01 * m22);
    result.value[1][2] = result.value[2][1] = detinv * (m01 * m02 - m00 * m12);
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (!std::isfinite(result.value[i][j]))
        throw std::invalid_argument(
            "owned tiled material callback returned a non-finite tensor inverse");
  return result;
}

bool zero_vector3(const vector3 &value) {
  return value.x == 0.0 && value.y == 0.0 && value.z == 0.0;
}

void validate_tiled_callback_output(const meep_geom::medium_struct &medium) {
  if (!medium.E_susceptibilities.empty() || !medium.H_susceptibilities.empty() ||
      !zero_vector3(medium.E_chi2_diag) || !zero_vector3(medium.E_chi3_diag) ||
      !zero_vector3(medium.H_chi2_diag) || !zero_vector3(medium.H_chi3_diag) ||
      !zero_vector3(medium.D_conductivity_diag) ||
      !zero_vector3(medium.B_conductivity_diag))
    throw std::invalid_argument(
        "owned tiled material callback returned an undeclared output family");
}

int material_tensor_column(component c, int encoded_direction) {
  const direction d = direction(encoded_direction);
  if (component_direction(c) == R || component_direction(c) == P) {
    if (d == R) return 0;
    if (d == P) return 1;
    if (d == Z) return 2;
  }
  else {
    if (d == X) return 0;
    if (d == Y) return 1;
    if (d == Z) return 2;
  }
  return -1;
}

bool same_owned_susceptibility(const std::vector<double> &medium,
                               const OwnedSusceptibilitySpan &candidate,
                               const std::vector<double> &identity) {
  /* susceptibility_equiv ignores only the six sigma values serialized first;
     bias is part of recurrence identity and begins at offset 6. */
  if (candidate.end - candidate.begin != identity.size() || identity.size() < 6) return false;
  for (size_t i = 6; i < identity.size(); ++i)
    if (medium[candidate.begin + i] != identity[i]) return false;
  return true;
}

int material_tensor_column(component c, int encoded_direction);

double owned_sigma_value(const OwnedMediumView &medium,
                         const MaterialIRSusceptibility &identity, component c, int d) {
  const int ft_slot = identity.field_type == E_stuff ? 0 : identity.field_type == H_stuff ? 1 : -1;
  if (ft_slot < 0 || type(c) != identity.field_type) return 0.0;
  const OwnedSusceptibilitySpan *match = NULL;
  for (const OwnedSusceptibilitySpan &candidate : medium.susceptibilities[ft_slot])
    if (same_owned_susceptibility(*medium.values, candidate, identity.parameters)) {
      match = &candidate;
      break;
    }
  if (!match) return 0.0;
  const std::vector<double> &p = *medium.values;
  const int row = component_index(c), column = material_tensor_column(c, d);
  if (column < 0) return 0.0;
  if (row == column) return p[match->begin + 3 + row];
  if ((row == 0 && column == 1) || (row == 1 && column == 0))
    return p[match->begin];
  if ((row == 0 && column == 2) || (row == 2 && column == 0))
    return p[match->begin + 1];
  return p[match->begin + 2];
}

double homogeneous_row_value(const MaterialIR &ir, const OwnedMediumView *medium,
                             const SymmetricMaterialTensor tensors[2], const StorageKey &key,
                             double dt) {
  const array_kind kind = static_cast<array_kind>(key.kind);
  const component c = component(key.component_);
  const int row = key.component_ >= 0 ? component_index(c) : -1;
  const int column = key.component_ >= 0 ? material_tensor_column(c, int(key.aux)) : -1;
  const bool perfect = ir.materials[ir.default_material].kind ==
                       meep_geom::material_data::PERFECT_METAL;
  if (kind == array_kind::chi1inv) {
    if (column < 0) return component_direction(c) == direction(key.aux) ? 1.0 : 0.0;
    if (is_electric(c)) {
      if (perfect) return row == column ? -0.0 : 0.0;
      return tensors[0].value[row][column];
    }
    if (is_magnetic(c)) {
      if (perfect) return row == column ? 1.0 : 0.0;
      return tensors[1].value[row][column];
    }
    return row == column ? 1.0 : 0.0;
  }
  if (kind == array_kind::chi2 || kind == array_kind::chi3) {
    if (!medium || (!is_electric(c) && !is_magnetic(c))) return 0.0;
    const size_t base = medium->base +
                        (is_electric(c) ? (kind == array_kind::chi2 ? 18 : 21)
                                        : (kind == array_kind::chi2 ? 24 : 27));
    return (*medium->values)[base + row];
  }
  if (kind == array_kind::conductivity) {
    if (!medium || direction(key.aux) != component_direction(c) ||
        (!is_D(c) && !is_B(c)))
      return 0.0;
    return (*medium->values)[medium->base + (is_D(c) ? 30 : 33) + row];
  }
  if (kind == array_kind::condinv) {
    if (!medium || direction(key.aux) != component_direction(c) ||
        (!is_D(c) && !is_B(c)))
      return 1.0;
    const double conductivity =
        (*medium->values)[medium->base + (is_D(c) ? 30 : 33) + row];
    if (sizeof(realnum) == sizeof(float)) {
      const realnum stored_conductivity = realnum(conductivity);
      return double(realnum(1 / (1 + double(stored_conductivity) * dt * 0.5)));
    }
    const double result = 1.0 / (1.0 + conductivity * dt * 0.5);
    if (!std::isfinite(result))
      throw std::invalid_argument("NVIDIA homogeneous condinv is non-finite");
    return result;
  }
  if (kind == array_kind::sigma) {
    if (!medium || (!is_electric(c) && !is_magnetic(c))) return 0.0;
    const field_type ft = field_type(key.aux % uint64_t(NUM_FIELD_TYPES));
    const uint64_t identity = key.aux / uint64_t(NUM_FIELD_TYPES);
    for (const MaterialIRSusceptibility &sus : ir.susceptibilities)
      if (sus.field_type == ft && sus.identity == identity)
        return owned_sigma_value(*medium, sus, c, key.cmp);
    throw std::invalid_argument("NVIDIA homogeneous sigma identity is missing");
  }
  throw std::invalid_argument("NVIDIA homogeneous initializer received an unsupported row kind");
}

ArrayId find_storage_key(const StoragePlan &plan, const StorageKey &key) {
  for (size_t i = 0; i < plan.keys.size(); ++i)
    if (plan.keys[i] == key) return ArrayId{uint32_t(i)};
  return invalid_array();
}

size_t material_compact_append_extent(size_t current, size_t count, size_t alignment,
                                      const char *what, size_t *offset = NULL) {
  if (!alignment || (alignment & (alignment - 1)))
    throw std::logic_error("NVIDIA material compact-input alignment is invalid");
  const size_t padding = (alignment - current % alignment) % alignment;
  const size_t aligned = material_checked_add(current, padding, what);
  const size_t total = material_checked_add(aligned, count, what);
  if (offset) *offset = aligned;
  return total;
}

struct MaterialCompactPack {
  std::vector<unsigned char> bytes;
  std::vector<size_t> table_header_offsets;
  size_t absorber_header_offset;
  size_t absorber_profile_bytes;
  size_t pml_profile_bytes;
  size_t file_sample_bytes;
  size_t grid_weight_bytes;
  size_t geometry_object_bytes;
  size_t geometry_image_bytes;
  size_t geometry_value_bytes;
  size_t geometry_analytic_bytes;
  size_t geometry_patch_bytes;

  MaterialCompactPack()
      : absorber_header_offset(0), absorber_profile_bytes(0), pml_profile_bytes(0),
        file_sample_bytes(0), grid_weight_bytes(0), geometry_object_bytes(0),
        geometry_image_bytes(0), geometry_value_bytes(0), geometry_analytic_bytes(0),
        geometry_patch_bytes(0) {}

  size_t append_space(size_t count, size_t alignment, const char *what) {
    size_t offset = 0;
    const size_t total =
        material_compact_append_extent(bytes.size(), count, alignment, what, &offset);
    bytes.resize(total, 0);
    return offset;
  }

  size_t append_doubles(const std::vector<double> &values, const char *what) {
    const size_t count = material_checked_product(values.size(), sizeof(double), what);
    const size_t offset = append_space(count, alignof(double), what);
    if (count) memcpy(bytes.data() + offset, values.data(), count);
    return offset;
  }

  template <typename T>
  size_t append_record(const T &value, const char *what) {
    const size_t offset = append_space(sizeof(T), alignof(T), what);
    memcpy(bytes.data() + offset, &value, sizeof(T));
    return offset;
  }

  template <typename T>
  size_t append_records(const std::vector<T> &values, const char *what) {
    const size_t count = material_checked_product(values.size(), sizeof(T), what);
    const size_t offset = append_space(count, alignof(T), what);
    if (count) memcpy(bytes.data() + offset, values.data(), count);
    return offset;
  }
};

template <typename T>
bool material_compact_range(size_t offset, size_t count, size_t bytes) {
  return offset % alignof(T) == 0 && offset <= bytes &&
         count <= (bytes - offset) / sizeof(T);
}

size_t pack_material_medium(MaterialCompactPack &compact, const OwnedMediumView &medium,
                            const MaterialIR &ir) {
  const std::vector<double> &p = *medium.values;
  nvidia::material_medium_header header = {};
  header.version = 1;
  if (medium.susceptibilities[0].size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("NVIDIA material susceptibility count overflows");
  header.electric_susceptibility_count =
      uint32_t(medium.susceptibilities[0].size());
  for (int i = 0; i < 3; ++i) {
    header.epsilon_diagonal[i] = p[medium.base + i];
    header.epsilon_offdiagonal[i] = p[medium.base + 3 + 2 * i];
    if (p[medium.base + 4 + 2 * i] != 0.0)
      throw std::invalid_argument(
          "NVIDIA table material has a non-real electric offdiagonal");
    header.conductivity[i] = p[medium.base + 30 + i];
  }
  std::vector<nvidia::material_susceptibility_record> records;
  records.reserve(medium.susceptibilities[0].size());
  for (size_t ordinal = 0; ordinal < medium.susceptibilities[0].size(); ++ordinal) {
    const OwnedSusceptibilitySpan &span = medium.susceptibilities[0][ordinal];
    nvidia::material_susceptibility_record record = {};
    record.version = 1;
    const MaterialIRSusceptibility *identity = NULL;
    for (const MaterialIRSusceptibility &candidate : ir.susceptibilities)
      if (candidate.field_type == E_stuff &&
          same_owned_susceptibility(p, span, candidate.parameters)) {
        identity = &candidate;
        break;
      }
    if (!identity)
      throw std::invalid_argument(
          "NVIDIA material susceptibility has no canonical recurrence identity");
    if (ordinal > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("NVIDIA material susceptibility ordinal overflows");
    record.identity = identity->identity;
    record.field_type = int32_t(identity->field_type);
    record.material_ordinal = uint32_t(ordinal);
    for (int i = 0; i < 3; ++i) {
      record.sigma_offdiagonal[i] = p[span.begin + i];
      record.sigma_diagonal[i] = p[span.begin + 3 + i];
    }
    records.push_back(record);
  }
  const size_t header_offset = compact.append_space(
      sizeof(header), alignof(decltype(header)), "packing material medium header");
  if (!records.empty())
    header.electric_susceptibility_offset = compact.append_records(
        records, "packing material susceptibility records");
  memcpy(compact.bytes.data() + header_offset, &header, sizeof(header));
  return header_offset;
}

struct MaterialCompactSizer {
  size_t bytes;

  MaterialCompactSizer() : bytes(0) {}

  void append(size_t count, size_t alignment, const char *what) {
    bytes = material_compact_append_extent(bytes, count, alignment, what);
  }

  template <typename T> void append_records(size_t count, const char *what) {
    append(material_checked_product(count, sizeof(T), what), alignof(T), what);
  }

  void append_doubles(size_t count, const char *what) {
    append_records<double>(count, what);
  }
};

void size_material_medium(MaterialCompactSizer &compact, const OwnedMediumView &medium) {
  compact.append_records<nvidia::material_medium_header>(
      1, "sizing packed material medium header");
  const size_t susceptibility_count = medium.susceptibilities[0].size();
  if (susceptibility_count)
    compact.append_records<nvidia::material_susceptibility_record>(
        susceptibility_count, "sizing packed material susceptibility records");
}

size_t exact_material_compact_input_bytes(const MaterialIR &ir) {
  MaterialCompactSizer compact;
  if (ir.topology.empty()) return 0;

  const bool geometry_partition = !ir.objects.empty() || !ir.analytic_interfaces.empty() ||
                                  !ir.hybrid_patches.empty();
  const MaterialIRMaterial &root = ir.materials[ir.default_material];
  const bool file_table = root.kind == meep_geom::material_data::MATERIAL_FILE;
  const bool grid_table = root.kind == meep_geom::material_data::MATERIAL_GRID;

  if (!geometry_partition && (file_table || grid_table)) {
    compact.append_records<nvidia::material_table_header>(
        1, "sizing packed material table header");
    size_t parameter_offset = file_table ? 0 : 3;
    const OwnedMediumView first = parse_owned_medium_at(root.parameters, parameter_offset);
    if (grid_table) {
      const OwnedMediumView second = parse_owned_medium_at(root.parameters, parameter_offset);
      size_material_medium(compact, first);
      size_material_medium(compact, second);
    }
    compact.append_doubles(root.samples.size(),
                           file_table ? "sizing packed FILE samples"
                                      : "sizing packed MaterialGrid weights");
  }

  if (!ir.absorbers.empty()) {
    compact.append_records<nvidia::material_absorber_header>(
        ir.absorbers.size(), "sizing packed absorber headers");
    for (const MaterialIRPml &absorber : ir.absorbers)
      compact.append_doubles(absorber.samples.size(),
                             "sizing packed absorber profile samples");
  }

  if (geometry_partition) {
    for (const MaterialIRObject &object : ir.objects) {
      compact.append_doubles(object.parameters.size(), "sizing packed geometry parameters");
      compact.append_doubles(object.vertices.size(), "sizing packed geometry vertices");
      compact.append_doubles(object.indices.size(), "sizing packed geometry indices");
      compact.append_doubles(object.auxiliary.size(),
                             "sizing packed geometry auxiliary data");
      if (object.kind == geometric_object::MESH) {
        compact.append_records<nvidia::geometry_triangle_record>(
            size_t(object.triangle_count), "sizing packed geometry triangles");
        compact.append_records<uint32_t>(size_t(object.triangle_count),
                                         "sizing packed geometry BVH faces");
        compact.append_records<nvidia::geometry_bvh_record>(
            size_t(object.bvh_count), "sizing packed geometry BVH nodes");
      }
    }
    compact.append_records<nvidia::geometry_object_record>(
        ir.objects.size(), "sizing packed geometry objects");
    compact.append_records<nvidia::geometry_image_record>(
        ir.active_images.size(), "sizing packed geometry images");

    std::vector<uint8_t> used(ir.materials.size(), 0);
    used[ir.default_material] = 1;
    for (const MaterialIRObject &object : ir.objects) used[size_t(object.material)] = 1;
    for (size_t material = 0; material < ir.materials.size(); ++material)
      if (used[material] && !ir.materials[material].samples.empty())
        compact.append_doubles(ir.materials[material].samples.size(),
                               "sizing packed geometry material samples");

    for (size_t destination = 0; destination < ir.destinations.size(); ++destination)
      compact.append_records<nvidia::geometry_value_record>(
          ir.materials.size(), "sizing packed geometry material values");

    size_t analytic = 0;
    while (analytic < ir.analytic_interfaces.size()) {
      const uint32_t destination = ir.analytic_interfaces[analytic].destination;
      const size_t begin = analytic++;
      while (analytic < ir.analytic_interfaces.size() &&
             ir.analytic_interfaces[analytic].destination == destination)
        ++analytic;
      compact.append_records<nvidia::geometry_analytic_record>(
          analytic - begin, "sizing packed geometry analytic jobs");
    }

    size_t patch = 0;
    while (patch < ir.hybrid_patches.size()) {
      const uint32_t destination = ir.hybrid_patches[patch].destination;
      const size_t begin = patch++;
      while (patch < ir.hybrid_patches.size() &&
             ir.hybrid_patches[patch].destination == destination)
        ++patch;
      compact.append_records<nvidia::geometry_patch_record>(
          patch - begin, "sizing packed geometry patches");
    }
  }

  for (const MaterialIRPmlAxis &axis : ir.pml_axes)
    if (axis.profile_active)
      compact.append_doubles(axis.profile_samples.size(),
                             "sizing packed PML profile samples");
  return compact.bytes;
}

struct MaterialStorageKeyLess {
  bool operator()(const StorageKey &left, const StorageKey &right) const {
    if (left.chunk != right.chunk) return left.chunk < right.chunk;
    if (left.kind != right.kind) return left.kind < right.kind;
    if (left.component_ != right.component_) return left.component_ < right.component_;
    if (left.cmp != right.cmp) return left.cmp < right.cmp;
    return left.aux < right.aux;
  }
};

int material_axis_direction(int dimensions, int axis) {
  if (dimensions == int(Dcyl)) {
    const int directions[3] = {int(P), int(R), int(Z)};
    return directions[axis];
  }
  if (dimensions == int(D2)) {
    const int directions[3] = {int(Z), int(X), int(Y)};
    return directions[axis];
  }
  return axis;
}

const MaterialIRChunk &material_chunk(const MaterialIR &ir, int chunk) {
  for (const MaterialIRChunk &candidate : ir.chunks)
    if (candidate.chunk == chunk) return candidate;
  throw std::invalid_argument("NVIDIA material topology references a missing chunk");
}

bool same_material_double(double left, double right) {
  return memcmp(&left, &right, sizeof(double)) == 0;
}

const MaterialIRSusceptibility &canonical_material_susceptibility(
    const MaterialIR &ir, const OwnedMediumView &medium,
    const OwnedSusceptibilitySpan &span) {
  for (const MaterialIRSusceptibility &candidate : ir.susceptibilities)
    if (candidate.field_type == E_stuff &&
        same_owned_susceptibility(*medium.values, span, candidate.parameters))
      return candidate;
  throw std::invalid_argument(
      "NVIDIA table susceptibility has no canonical owned identity");
}

void validate_packed_material_medium(const MaterialIR &ir, const OwnedMediumView &expected,
                                     const unsigned char *compact_inputs,
                                     const nvidia::material_medium_header &observed) {
  if (observed.version != 1 ||
      observed.electric_susceptibility_count != expected.susceptibilities[0].size())
    throw std::invalid_argument("NVIDIA packed table medium identity is stale");
  const std::vector<double> &parameters = *expected.values;
  for (int i = 0; i < 3; ++i)
    if (!same_material_double(observed.epsilon_diagonal[i], parameters[expected.base + i]) ||
        !same_material_double(observed.epsilon_offdiagonal[i],
                              parameters[expected.base + 3 + 2 * i]) ||
        !same_material_double(observed.conductivity[i],
                              parameters[expected.base + 30 + i]))
      throw std::invalid_argument("NVIDIA packed table medium values are stale");
  const nvidia::material_susceptibility_record *records =
      observed.electric_susceptibility_count
          ? reinterpret_cast<const nvidia::material_susceptibility_record *>(
                compact_inputs + observed.electric_susceptibility_offset)
          : NULL;
  for (size_t ordinal = 0; ordinal < expected.susceptibilities[0].size(); ++ordinal) {
    const OwnedSusceptibilitySpan &span = expected.susceptibilities[0][ordinal];
    const MaterialIRSusceptibility &identity =
        canonical_material_susceptibility(ir, expected, span);
    const nvidia::material_susceptibility_record &record = records[ordinal];
    if (record.version != 1 || record.identity != identity.identity ||
        record.field_type != identity.field_type || record.material_ordinal != ordinal)
      throw std::invalid_argument("NVIDIA packed susceptibility identity is stale");
    for (int i = 0; i < 3; ++i)
      if (!same_material_double(record.sigma_offdiagonal[i], parameters[span.begin + i]) ||
          !same_material_double(record.sigma_diagonal[i], parameters[span.begin + 3 + i]))
        throw std::invalid_argument("NVIDIA packed susceptibility values are stale");
  }
}

void validate_packed_material_table(const MaterialIR &ir,
                                    const unsigned char *compact_inputs,
                                    size_t compact_input_bytes, size_t table_header_offset) {
  nvidia::validate_material_table_headers(compact_inputs, compact_input_bytes,
                                           &table_header_offset, 1);
  const MaterialIRMaterial &root = ir.materials[ir.default_material];
  const bool file_table = root.kind == meep_geom::material_data::MATERIAL_FILE;
  const bool grid_table = root.kind == meep_geom::material_data::MATERIAL_GRID;
  if (!file_table && !grid_table)
    throw std::invalid_argument("NVIDIA packed table has no authoritative table material");
  const nvidia::material_table_header &header =
      *reinterpret_cast<const nvidia::material_table_header *>(compact_inputs +
                                                               table_header_offset);
  const nvidia::material_table_kind expected_kind =
      file_table ? nvidia::material_table_kind::file_scalar_epsilon
                 : nvidia::material_table_kind::material_grid;
  if (header.version != 1 || header.material_id != ir.default_material ||
      header.kind != expected_kind ||
      header.overlap_kind != uint32_t(file_table ? meep_geom::material_data::U_DEFAULT
                                                 : root.material_grid_kind))
    throw std::invalid_argument("NVIDIA packed table header identity is stale");
  size_t parameter_offset = file_table ? 0 : 3;
  OwnedMediumView first = parse_owned_medium_at(root.parameters, parameter_offset);
  std::unique_ptr<OwnedMediumView> second;
  if (grid_table) second.reset(new OwnedMediumView(parse_owned_medium_at(root.parameters,
                                                                         parameter_offset)));
  const size_t dimension_offset = file_table ? parameter_offset : 0;
  size_t sample_count = 1;
  for (int axis = 0; axis < 3; ++axis) {
    const size_t extent = owned_count(root.parameters[dimension_offset + axis],
                                      "validating material table dimensions");
    sample_count = material_checked_product(sample_count, extent,
                                            "validating material table sample count");
    if (extent != header.dimensions[axis])
      throw std::invalid_argument("NVIDIA packed table dimensions are stale");
  }
  if (sample_count != root.samples.size() || header.sample_count != root.samples.size())
    throw std::invalid_argument("NVIDIA packed table sample count is stale");
  const double *samples =
      reinterpret_cast<const double *>(compact_inputs + size_t(header.sample_offset));
  for (size_t i = 0; i < root.samples.size(); ++i)
    if (!same_material_double(samples[i], root.samples[i]))
      throw std::invalid_argument("NVIDIA packed table samples are stale");
  if (file_table) {
    if (header.medium_1_offset || header.medium_2_offset ||
        !same_material_double(header.beta, 0.0) ||
        !same_material_double(header.eta, 0.0) ||
        !same_material_double(header.damping, 0.0) ||
        !same_material_double(header.projection_offset, 0.0))
      throw std::invalid_argument("NVIDIA packed FILE metadata is stale");
  }
  else {
    if (root.parameters.size() - parameter_offset != 3 ||
        !same_material_double(header.beta, root.parameters[parameter_offset]) ||
        !same_material_double(header.eta, root.parameters[parameter_offset + 1]) ||
        !same_material_double(header.damping, root.parameters[parameter_offset + 2]) ||
        !same_material_double(header.projection_offset, ir.projection_offset))
      throw std::invalid_argument("NVIDIA packed MaterialGrid metadata is stale");
    const nvidia::material_medium_header &first_header =
        *reinterpret_cast<const nvidia::material_medium_header *>(
            compact_inputs + size_t(header.medium_1_offset));
    const nvidia::material_medium_header &second_header =
        *reinterpret_cast<const nvidia::material_medium_header *>(
            compact_inputs + size_t(header.medium_2_offset));
    validate_packed_material_medium(ir, first, compact_inputs, first_header);
    validate_packed_material_medium(ir, *second, compact_inputs, second_header);
  }
}

void validate_material_table_authority(const MaterialIR &ir, const NvidiaBackendState &state,
                                       const MaterialCompactPack &compact, double dt,
                                       const nvidia::material_table_launch &launch,
                                       const NvidiaMaterialTableAuthority &authority,
                                       bool require_device_pointer) {
  if (authority.table_header_offset != launch.table_header_offset ||
      authority.destination.value >= state.plan_.arrays.size() ||
      !(state.plan_.keys[authority.destination.value] == authority.row.key))
    throw std::invalid_argument("NVIDIA table launch has stale storage authority");
  size_t row_matches = 0;
  for (const MaterialIRTopologyRow &row : ir.topology)
    if (row.key == authority.row.key) {
      ++row_matches;
      if (!(row == authority.row))
        throw std::invalid_argument("NVIDIA table launch topology row is stale");
    }
  if (row_matches != 1)
    throw std::invalid_argument("NVIDIA table launch topology authority is ambiguous");
  const ArraySpec &destination_spec = state.plan_.arrays[authority.destination.value];
  if (destination_spec.id != authority.destination ||
      destination_spec.role != array_role::material ||
      destination_spec.element_type != authority.row.element_type ||
      destination_spec.elements != authority.row.elements ||
      destination_spec.alignment != authority.row.alignment ||
      is_valid(destination_spec.alias_of) || launch.elements != authority.row.elements ||
      launch.destination != state.arenas_->resolve(authority.destination.value).address ||
      launch.precision != scalar_precision_for(state.plan_, authority.destination,
                                               "NVIDIA table authority destination"))
    throw std::invalid_argument("NVIDIA table launch destination authority differs");
  const MaterialIRChunk &chunk = material_chunk(ir, authority.row.key.chunk);
  const component c = component(authority.row.key.component_);
  if (authority.row.yee_component != int(c) || !chunk.owned ||
      !(chunk.component_bits & (uint64_t(1) << int(c))) ||
      launch.destination_component != int(c) || launch.query_component != int(c) ||
      launch.tensor_row != component_index(c) || launch.dimensions != ir.dimensions)
    throw std::invalid_argument("NVIDIA table launch component authority differs");
  if (launch.loop_count != chunk.loop_count[c] ||
      !same_material_double(launch.inva, chunk.inva) || !same_material_double(launch.dt, dt) ||
      launch.logical_single != (sizeof(realnum) == sizeof(float)))
    throw std::invalid_argument("NVIDIA table launch numeric chunk authority differs");
  if (launch.compact_input_bytes != compact.bytes.size() ||
      (require_device_pointer
           ? launch.compact_inputs != static_cast<const unsigned char *>(
                                         state.material_ir_inputs_.opaque_handle())
           : launch.compact_inputs != NULL))
    throw std::invalid_argument("NVIDIA table launch compact-input authority differs");
  for (int axis = 0; axis < 3; ++axis) {
    const int64_t stagger = int64_t(chunk.loop_begin[c][axis]) -
                            int64_t(chunk.little_corner[axis]);
    const int64_t doubled_extent = int64_t(chunk.loop_end[c][axis]) -
                                   int64_t(chunk.loop_begin[c][axis]);
    if (authority.row.extents[axis] != chunk.extents[axis] ||
        authority.row.strides[axis] != chunk.strides[axis] ||
        authority.row.stagger[axis] != chunk.stagger[c][axis] ||
        launch.axis_direction[axis] != material_axis_direction(ir.dimensions, axis) ||
        launch.loop_begin[axis] != chunk.loop_begin[c][axis] ||
        launch.loop_end[axis] != chunk.loop_end[c][axis] ||
        launch.little_corner[axis] != chunk.little_corner[axis] ||
        launch.loop_base_offset[axis] !=
            size_t(stagger / 2) * size_t(chunk.strides[axis]) ||
        launch.loop_extent[axis] != size_t(doubled_extent / 2) + 1 ||
        launch.strides[axis] != chunk.strides[axis] ||
        !same_material_double(launch.cell_center[axis], ir.cell[axis]) ||
        !same_material_double(launch.cell_size[axis], ir.cell[axis + 3]))
      throw std::invalid_argument("NVIDIA table launch geometry authority differs");
  }
  validate_packed_material_table(ir, compact.bytes.data(), compact.bytes.size(),
                                 authority.table_header_offset);
  const MaterialIRMaterial &root = ir.materials[ir.default_material];
  const bool file_table = root.kind == meep_geom::material_data::MATERIAL_FILE;
  const nvidia::material_table_kind expected_kind =
      file_table ? nvidia::material_table_kind::file_scalar_epsilon
                 : nvidia::material_table_kind::material_grid;
  if (launch.table_kind != expected_kind || launch.source_material_id != ir.default_material ||
      launch.operation_family != 0)
    throw std::invalid_argument("NVIDIA table launch source identity differs");
  const array_kind row_kind = static_cast<array_kind>(authority.row.key.kind);
  const int expected_column = material_tensor_column(c,
      row_kind == array_kind::sigma ? authority.row.key.cmp : int(authority.row.key.aux));
  nvidia::material_table_operation expected_operation;
  ArrayId expected_secondary = invalid_array();
  uint32_t expected_source_medium = 0;
  size_t expected_source_ordinal = 0;
  uint32_t expected_identity = 0;
  int expected_field_type = int(NO_FIELD_TYPE);
  size_t expected_absorber_count = 0;
  size_t expected_absorber_offset = 0;
  if (row_kind == array_kind::chi1inv && is_electric(c) && expected_column >= 0 &&
      (!file_table || expected_column == component_index(c)))
    expected_operation = file_table ? nvidia::material_table_operation::file_chi1inv
                                    : nvidia::material_table_operation::grid_chi1inv;
  else if (!file_table && row_kind == array_kind::conductivity && is_D(c) &&
           direction(authority.row.key.aux) == component_direction(c)) {
    expected_operation = nvidia::material_table_operation::grid_conductivity;
    StorageKey inverse_key = authority.row.key;
    inverse_key.kind = int(array_kind::condinv);
    expected_secondary = find_storage_key(state.plan_, inverse_key);
    expected_absorber_count = ir.absorbers.size();
    expected_absorber_offset = compact.absorber_header_offset;
  }
  else if (!file_table && row_kind == array_kind::sigma && is_electric(c) &&
           expected_column >= 0) {
    expected_operation = nvidia::material_table_operation::grid_sigma;
    const field_type ft = field_type(authority.row.key.aux % uint64_t(NUM_FIELD_TYPES));
    const uint64_t identity_value = authority.row.key.aux / uint64_t(NUM_FIELD_TYPES);
    if (ft != E_stuff || identity_value > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument("NVIDIA table sigma topology identity is invalid");
    const MaterialIRSusceptibility *identity = NULL;
    for (const MaterialIRSusceptibility &candidate : ir.susceptibilities)
      if (candidate.field_type == ft && candidate.identity == identity_value) {
        identity = &candidate;
        break;
      }
    if (!identity)
      throw std::invalid_argument("NVIDIA table sigma identity is absent from owned IR");
    size_t parameter_offset = 3;
    OwnedMediumView first = parse_owned_medium_at(root.parameters, parameter_offset);
    OwnedMediumView second = parse_owned_medium_at(root.parameters, parameter_offset);
    const OwnedMediumView *ordered[2] = {&first, &second};
    for (size_t medium_index = 0; medium_index < 2 && !expected_source_medium;
         ++medium_index)
      for (size_t ordinal = 0; ordinal < ordered[medium_index]->susceptibilities[0].size();
           ++ordinal)
        if (same_owned_susceptibility(*ordered[medium_index]->values,
                                      ordered[medium_index]->susceptibilities[0][ordinal],
                                      identity->parameters)) {
          expected_source_medium = uint32_t(medium_index + 1);
          expected_source_ordinal = ordinal;
          break;
        }
    if (!expected_source_medium)
      throw std::invalid_argument("NVIDIA table sigma has no first-equivalent source");
    expected_identity = uint32_t(identity_value);
    expected_field_type = int(ft);
  }
  else
    throw std::invalid_argument("NVIDIA table launch is not authorized by its topology row");
  if (launch.operation != expected_operation || launch.tensor_column != expected_column ||
      launch.source_medium != expected_source_medium ||
      launch.source_susceptibility != expected_source_ordinal ||
      launch.susceptibility_identity != expected_identity ||
      launch.susceptibility_field_type != expected_field_type ||
      authority.secondary_destination != expected_secondary ||
      launch.absorber_count != expected_absorber_count ||
      launch.absorber_header_offset != expected_absorber_offset)
    throw std::invalid_argument("NVIDIA table launch operation authority differs");
  for (int axis = 0; axis < 3; ++axis) {
    const int expected_shift = expected_column != component_index(c) &&
                                       launch.axis_direction[axis] == int(component_direction(c))
                                   ? -1
                                   : 0;
    if (launch.evaluation_shift[axis] != expected_shift)
      throw std::invalid_argument("NVIDIA table launch evaluation authority differs");
  }
  if (is_valid(expected_secondary)) {
    if (expected_secondary.value >= state.plan_.arrays.size() ||
        launch.secondary_destination !=
            state.arenas_->resolve(expected_secondary.value).address)
      throw std::invalid_argument("NVIDIA table launch secondary destination differs");
    const ArraySpec &secondary_spec = state.plan_.arrays[expected_secondary.value];
    if (secondary_spec.role != array_role::material ||
        secondary_spec.element_type != authority.row.element_type ||
        secondary_spec.elements != authority.row.elements ||
        secondary_spec.alignment != authority.row.alignment ||
        is_valid(secondary_spec.alias_of) ||
        scalar_precision_for(state.plan_, expected_secondary,
                             "NVIDIA table authority secondary") != launch.precision)
      throw std::invalid_argument("NVIDIA table secondary storage authority differs");
  }
  else if (launch.secondary_destination)
    throw std::invalid_argument("NVIDIA table launch has an unauthorized secondary destination");
  nvidia::validate_material_table_launch(launch, compact.bytes.data(), compact.bytes.size());
}

void validate_material_ir_against_live(const MaterialIR &ir, const fields &f, double dt) {
  const MaterialIR *const canonical = material_ir_for(f);
  if (!canonical || !material_ir_equal(ir, *canonical))
    throw std::invalid_argument(
        "NVIDIA material IR differs from the live canonical owned snapshot");
  if (ir.chunks.size() != size_t(f.num_chunks))
    throw std::invalid_argument("NVIDIA material IR chunk count differs from live geometry");
  for (const MaterialIRChunk &chunk : ir.chunks) {
    if (chunk.chunk < 0 || chunk.chunk >= f.num_chunks || !f.chunks[chunk.chunk] ||
        !f.chunks[chunk.chunk]->s)
      throw std::invalid_argument("NVIDIA material IR chunk is absent from live geometry");
    const structure_chunk &live = *f.chunks[chunk.chunk]->s;
    const grid_volume &gv = live.gv;
    if (chunk.dimensions != int(gv.dim) || chunk.owned != live.is_mine() ||
        chunk.resolution != gv.a || chunk.inva != gv.inva ||
        chunk.elements != size_t(gv.ntot()))
      throw std::invalid_argument("NVIDIA material IR chunk scalar metadata is stale");
    uint64_t component_bits = 0;
    FOR_COMPONENTS(c) if (gv.has_field(c)) component_bits |= uint64_t(1) << int(c);
    if (chunk.component_bits != component_bits)
      throw std::invalid_argument("NVIDIA material IR chunk component mask is stale");
    for (int axis = 0; axis < 3; ++axis) {
      const direction d = gv.yucky_direction(axis);
      if (material_axis_direction(chunk.dimensions, axis) != int(d) ||
          chunk.little_corner[axis] != gv.little_corner().yucky_val(axis) ||
          chunk.big_corner[axis] != gv.big_corner().yucky_val(axis) ||
          chunk.strides[axis] != gv.stride(d))
        throw std::invalid_argument("NVIDIA material IR chunk layout metadata is stale");
      FOR_COMPONENTS(c) if (gv.has_field(c)) {
        const int shift = gv.iyee_shift(c).in_direction(d);
        if (chunk.stagger[c][axis] != shift ||
            chunk.loop_begin[c][axis] != chunk.little_corner[axis] + shift ||
            chunk.loop_end[c][axis] != chunk.big_corner[axis] + shift)
          throw std::invalid_argument("NVIDIA material IR Yee metadata is stale");
      }
    }
    for (int d = 0; d < 6; ++d)
      if (chunk.pml_elements[d] != size_t(live.sigsize[d]))
        throw std::invalid_argument("NVIDIA material IR PML extent is stale");
  }

  for (const MaterialIRPmlAxis &axis : ir.pml_axes) {
    const structure_chunk &live = *f.chunks[axis.chunk]->s;
    if (axis.elements != size_t(live.sigsize[axis.direction]) ||
        !live.sig[axis.direction] || !live.kap[axis.direction] ||
        !live.siginv[axis.direction])
      throw std::invalid_argument("NVIDIA material IR PML destination is stale");
    if (axis.elements > size_t(std::numeric_limits<int>::max()) ||
        axis.little_corner > std::numeric_limits<int>::max() - int(axis.elements - 1))
      throw std::overflow_error("NVIDIA material PML integer index overflows");
    if (axis.profile_active) {
      /* This is the authoritative structure.cpp:pml_x expression.  Keep the
         evaluation in double so half-cell rounding is bit-for-bit identical. */
      const double scaled_thickness =
          axis.thickness * (2 * axis.resolution) + 0.5;
      if (!std::isfinite(scaled_thickness) || scaled_thickness <= 0 ||
          scaled_thickness > double(INT_MAX))
        throw std::overflow_error("NVIDIA material PML thickness conversion overflows");
    }
    for (size_t i = 0; i < axis.elements; ++i) {
      realnum expected_sigma = 0, expected_kappa = 1, expected_inverse = 1;
      if (axis.profile_active) {
        const int logical_index = axis.little_corner + int(i);
        const double here = logical_index * 0.5 / axis.resolution;
        const long double scaled_distance =
            std::fabs(static_cast<long double>(axis.boundary_location) - here) *
                (2 * axis.resolution) +
            0.5L;
        if (!std::isfinite(double(scaled_distance)) || scaled_distance > INT_MAX)
          throw std::overflow_error("NVIDIA material PML distance conversion overflows");
        const double x = 0.5 / axis.resolution *
                         (int(axis.thickness * (2 * axis.resolution) + 0.5) -
                          int(std::fabs(axis.boundary_location - here) *
                                  (2 * axis.resolution) +
                              0.5));
        if (x > 0) {
          const double sample = axis.profile_samples[i];
          if (axis.analytic_quadratic && sample != (x / axis.thickness) * (x / axis.thickness))
            throw std::invalid_argument("NVIDIA analytic PML sample authority is inconsistent");
          const double prefactor =
              (-std::log(axis.r_asymptotic)) /
              (4 * axis.thickness * axis.profile_integral);
          const double kappa_prefactor =
              (axis.mean_stretch - 1) / axis.profile_integral_u;
          expected_sigma = realnum(0.5 * dt * prefactor * sample);
          expected_kappa = realnum(1 + kappa_prefactor * sample * (x / axis.thickness));
          expected_inverse = realnum(1 / (expected_kappa + expected_sigma));
        }
      }
      if (axis.sigma[i] != double(expected_sigma) ||
          axis.kappa[i] != double(expected_kappa) ||
          axis.sigma_inv[i] != double(expected_inverse) ||
          axis.sigma[i] != double(live.sig[axis.direction][i]) ||
          axis.kappa[i] != double(live.kap[axis.direction][i]) ||
          axis.sigma_inv[i] != double(live.siginv[axis.direction][i]))
        throw std::invalid_argument("NVIDIA material PML oracle differs from live geometry");
    }
  }
}

double default_material_row_value(const StorageKey &key) {
  const array_kind kind = static_cast<array_kind>(key.kind);
  if (kind == array_kind::chi1inv || kind == array_kind::condinv) {
    const component c = component(key.component_);
    return component_direction(c) == direction(key.aux) ? 1.0 : 0.0;
  }
  if (kind == array_kind::chi2 || kind == array_kind::chi3 ||
      kind == array_kind::conductivity || kind == array_kind::sigma ||
      kind == array_kind::pml_sig)
    return 0.0;
  if (kind == array_kind::pml_kap || kind == array_kind::pml_siginv) return 1.0;
  throw std::invalid_argument("NVIDIA material default initializer received an unsupported row");
}

int material_fill_phase(array_kind kind) {
  switch (kind) {
    case array_kind::chi1inv: return 1;
    case array_kind::chi2:
    case array_kind::chi3: return 2;
    case array_kind::conductivity: return 3;
    case array_kind::condinv: return 4;
    case array_kind::sigma: return 5;
    default: return 6;
  }
}

int geometry_property_phase(int property) {
  switch (MaterialIRProperty(property)) {
    case MaterialIRProperty::chi1inv: return 1;
    case MaterialIRProperty::chi2:
    case MaterialIRProperty::chi3: return 2;
    case MaterialIRProperty::conductivity: return 3;
    case MaterialIRProperty::condinv: return 4;
    case MaterialIRProperty::sigma: return 5;
  }
  return 6;
}

void geometry_tensor_from_medium(nvidia::geometry_value_record &record,
                                 const OwnedMediumView &medium, field_type ft,
                                 double tensor[6]) {
  const std::vector<double> &p = *medium.values;
  const size_t diagonal = medium.base + (ft == E_stuff ? 0 : 9);
  const size_t offdiagonal = medium.base + (ft == E_stuff ? 3 : 12);
  if (p[offdiagonal + 1] != 0.0 || p[offdiagonal + 3] != 0.0 ||
      p[offdiagonal + 5] != 0.0)
    throw std::invalid_argument(
        "NVIDIA geometry material has a non-real tensor offdiagonal");
  tensor[0] = p[diagonal];
  tensor[1] = p[diagonal + 1];
  tensor[2] = p[diagonal + 2];
  tensor[3] = p[offdiagonal];
  tensor[4] = p[offdiagonal + 2];
  tensor[5] = p[offdiagonal + 4];
  (void)record;
}

const MaterialIRSusceptibility &geometry_sigma_identity(const MaterialIR &ir,
                                                        const MaterialIRDestination &destination) {
  const field_type ft = field_type(destination.key.aux % uint64_t(NUM_FIELD_TYPES));
  const uint64_t identity = destination.key.aux / uint64_t(NUM_FIELD_TYPES);
  for (const MaterialIRSusceptibility &candidate : ir.susceptibilities)
    if (candidate.field_type == ft && candidate.identity == identity) return candidate;
  throw std::invalid_argument("NVIDIA geometry sigma identity is absent");
}

double geometry_medium_scalar(const MaterialIR &ir, const OwnedMediumView &medium,
                              const MaterialIRDestination &destination) {
  const component c = component(destination.component);
  const int row = component_index(c);
  const std::vector<double> &p = *medium.values;
  switch (destination.property) {
    case MaterialIRProperty::conductivity:
    case MaterialIRProperty::condinv:
      if (direction(destination.tensor_direction) != component_direction(c) ||
          (!is_D(c) && !is_B(c)))
        return 0.0;
      return p[medium.base + (is_D(c) ? 30 : 33) + row];
    case MaterialIRProperty::chi2:
      if (!is_electric(c) && !is_magnetic(c)) return 0.0;
      return p[medium.base + (is_electric(c) ? 18 : 24) + row];
    case MaterialIRProperty::chi3:
      if (!is_electric(c) && !is_magnetic(c)) return 0.0;
      return p[medium.base + (is_electric(c) ? 21 : 27) + row];
    case MaterialIRProperty::sigma:
      return owned_sigma_value(medium, geometry_sigma_identity(ir, destination), c,
                               destination.tensor_direction);
    case MaterialIRProperty::chi1inv: break;
  }
  throw std::logic_error("NVIDIA geometry scalar requested for tensor property");
}

bool geometry_medium_sigma(const MaterialIR &ir, const OwnedMediumView &medium,
                           const MaterialIRDestination &destination, double &value) {
  const MaterialIRSusceptibility &identity = geometry_sigma_identity(ir, destination);
  const component c = component(destination.component);
  const int slot = identity.field_type == E_stuff ? 0 : identity.field_type == H_stuff ? 1 : -1;
  if (slot < 0 || type(c) != identity.field_type) return false;
  for (const OwnedSusceptibilitySpan &candidate : medium.susceptibilities[slot])
    if (same_owned_susceptibility(*medium.values, candidate, identity.parameters)) {
      value = owned_sigma_value(medium, identity, c, destination.tensor_direction);
      return true;
    }
  return false;
}

nvidia::geometry_value_record geometry_value_for(
    const MaterialIR &ir, uint32_t material_index,
    const MaterialIRDestination &destination, size_t sample_offset) {
  if (material_index >= ir.materials.size())
    throw std::invalid_argument("NVIDIA geometry material identity is invalid");
  const MaterialIRMaterial &material = ir.materials[material_index];
  const component c = component(destination.component);
  nvidia::geometry_value_record result = {};
  result.kind = nvidia::geometry_value_kind::constant;
  result.flags = 1u; // geometry_value_direct

  if (material.kind == meep_geom::material_data::PERFECT_METAL) {
    if (destination.property == MaterialIRProperty::chi1inv) {
      const bool diagonal = destination.tensor_column == component_index(c);
      result.value_1 = is_electric(c) && diagonal ? -0.0
                       : is_magnetic(c) && diagonal ? 1.0 : 0.0;
    }
    return result;
  }
  if (material.kind == meep_geom::material_data::MATERIAL_USER)
    throw std::invalid_argument("NVIDIA geometry cannot pack a reachable user callback");

  size_t offset = material.kind == meep_geom::material_data::MATERIAL_GRID ? 3 : 0;
  OwnedMediumView first = parse_owned_medium_at(material.parameters, offset);
  if (material.kind == meep_geom::material_data::MEDIUM) {
    if (offset != material.parameters.size())
      throw std::invalid_argument("NVIDIA geometry medium payload has trailing values");
    if (destination.property == MaterialIRProperty::chi1inv &&
        (is_electric(c) || is_magnetic(c)) && destination.tensor_column >= 0) {
      result.flags = 0;
      geometry_tensor_from_medium(result, first, is_electric(c) ? E_stuff : H_stuff,
                                  result.tensor_1);
    }
    else if (destination.property == MaterialIRProperty::condinv ||
             destination.property == MaterialIRProperty::conductivity)
      result.value_1 = geometry_medium_scalar(ir, first, destination);
    else if (destination.property != MaterialIRProperty::chi1inv)
      result.value_1 = geometry_medium_scalar(ir, first, destination);
    else
      result.value_1 = destination.tensor_column == component_index(c) ? 1.0 : 0.0;
    return result;
  }

  if (material.kind == meep_geom::material_data::MATERIAL_FILE) {
    if (material.parameters.size() - offset != 3)
      throw std::invalid_argument("NVIDIA geometry FILE payload has a wrong tail");
    if (destination.property == MaterialIRProperty::chi1inv && is_electric(c) &&
        destination.tensor_column == component_index(c)) {
      result.kind = nvidia::geometry_value_kind::file_epsilon;
      result.flags = 0;
      result.sample_offset = sample_offset;
      result.sample_count = material.samples.size();
      for (int axis = 0; axis < 3; ++axis)
        result.dimensions[axis] = material_checked_uint32(
            owned_count(material.parameters[offset + axis], "geometry FILE dimension"),
            "packing geometry FILE dimension");
    }
    else if (destination.property == MaterialIRProperty::chi1inv && is_magnetic(c) &&
             destination.tensor_column >= 0) {
      result.flags = 0;
      geometry_tensor_from_medium(result, first, H_stuff, result.tensor_1);
    }
    else if (destination.property == MaterialIRProperty::chi1inv)
      result.value_1 = destination.tensor_column == component_index(c) ? 1.0 : 0.0;
    return result;
  }

  if (material.kind != meep_geom::material_data::MATERIAL_GRID)
    throw std::invalid_argument("NVIDIA geometry material kind is unsupported");
  OwnedMediumView second = parse_owned_medium_at(material.parameters, offset);
  if (material.parameters.size() - offset != 3)
    throw std::invalid_argument("NVIDIA geometry MaterialGrid payload has a wrong tail");
  size_t comparison_offset = 0;
  OwnedMediumView comparison = parse_owned_medium_at(material.comparison_medium,
                                                      comparison_offset);
  if (comparison_offset != material.comparison_medium.size())
    throw std::invalid_argument("NVIDIA geometry MaterialGrid comparison medium is stale");
  result.overlap_kind = uint32_t(material.material_grid_kind);
  result.sample_offset = sample_offset;
  result.sample_count = material.samples.size();
  for (int axis = 0; axis < 3; ++axis)
    result.dimensions[axis] = material_checked_uint32(
        owned_count(material.parameters[axis], "geometry MaterialGrid dimension"),
        "packing geometry MaterialGrid dimension");
  result.beta = material.parameters[offset];
  result.eta = material.parameters[offset + 1];
  result.damping = material.parameters[offset + 2];
  result.projection_offset = ir.projection_offset;
  result.flags = 0;
  if (destination.property == MaterialIRProperty::chi1inv && is_electric(c) &&
      destination.tensor_column >= 0) {
    result.kind = nvidia::geometry_value_kind::grid_tensor;
    result.flags = 0;
    geometry_tensor_from_medium(result, first, E_stuff, result.tensor_1);
    geometry_tensor_from_medium(result, second, E_stuff, result.tensor_2);
  }
  else if (destination.property == MaterialIRProperty::chi1inv && is_magnetic(c) &&
           destination.tensor_column >= 0) {
    result.kind = nvidia::geometry_value_kind::constant;
    result.flags = 0;
    geometry_tensor_from_medium(result, comparison, H_stuff, result.tensor_1);
  }
  else if ((destination.property == MaterialIRProperty::conductivity ||
            destination.property == MaterialIRProperty::condinv) && is_D(c) &&
           direction(destination.tensor_direction) == component_direction(c)) {
    result.kind = destination.property == MaterialIRProperty::conductivity
                      ? nvidia::geometry_value_kind::grid_linear
                      : nvidia::geometry_value_kind::grid_condinv;
    result.value_1 = geometry_medium_scalar(ir, first, destination);
    result.value_2 = geometry_medium_scalar(ir, second, destination);
  }
  else if (destination.property == MaterialIRProperty::sigma && is_electric(c)) {
    result.kind = nvidia::geometry_value_kind::grid_linear;
    double sigma = 0.0;
    if (geometry_medium_sigma(ir, first, destination, sigma)) result.value_1 = sigma;
    sigma = 0.0;
    if (geometry_medium_sigma(ir, second, destination, sigma)) result.value_2 = sigma;
  }
  else if (destination.property == MaterialIRProperty::chi1inv) {
    result.kind = nvidia::geometry_value_kind::constant;
    result.flags = 1u;
    result.value_1 = destination.tensor_column == component_index(c) ? 1.0 : 0.0;
  }
  else {
    result.kind = nvidia::geometry_value_kind::constant;
    result.flags = 1u;
    result.value_1 = geometry_medium_scalar(ir, comparison, destination);
  }
  return result;
}

void preflight_geometry_ir(const MaterialIR &ir) {
  std::vector<uint8_t> used(ir.materials.size(), 0);
  used[ir.default_material] = 1;
  for (const MaterialIRObject &object : ir.objects) used[size_t(object.material)] = 1;
  for (const MaterialIRDestination &destination : ir.destinations)
    for (uint32_t material = 0; material < ir.materials.size(); ++material)
      if (used[material]) (void)geometry_value_for(ir, material, destination, 0);
}

int geometry_object_subtype(const MaterialIRObject &object) {
  if (object.kind == geometric_object::BLOCK) return int(object.parameters[24]);
  if (object.kind == geometric_object::CYLINDER) return int(object.parameters[8]);
  if (object.kind == geometric_object::MESH) return int(object.parameters[3]);
  return 0;
}

uint64_t geometry_compact_hash(const std::vector<unsigned char> &bytes) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (unsigned char byte : bytes) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

bool same_geometry_common(nvidia::geometry_launch_common left,
                          nvidia::geometry_launch_common right) {
  if (left.destination != right.destination ||
      left.compact_input_bytes != right.compact_input_bytes ||
      left.object_offset != right.object_offset || left.object_count != right.object_count ||
      left.image_offset != right.image_offset || left.image_count != right.image_count ||
      left.value_offset != right.value_offset || left.material_count != right.material_count ||
      left.absorber_header_offset != right.absorber_header_offset ||
      left.absorber_count != right.absorber_count || left.default_material != right.default_material ||
      left.elements != right.elements || left.point_count != right.point_count ||
      left.dimensions != right.dimensions || left.component != right.component ||
      left.tensor_row != right.tensor_row || left.tensor_column != right.tensor_column ||
      left.property != right.property || left.inva != right.inva || left.dt != right.dt ||
      left.trivial_value != right.trivial_value ||
      left.logical_single != right.logical_single || left.precision != right.precision)
    return false;
  for (int axis = 0; axis < 3; ++axis)
    if (left.axis_direction[axis] != right.axis_direction[axis] ||
        left.loop_begin[axis] != right.loop_begin[axis] ||
        left.little_corner[axis] != right.little_corner[axis] ||
        left.loop_base_offset[axis] != right.loop_base_offset[axis] ||
        left.loop_extent[axis] != right.loop_extent[axis] ||
        left.strides[axis] != right.strides[axis] ||
        left.evaluation_shift[axis] != right.evaluation_shift[axis] ||
        left.cell_center[axis] != right.cell_center[axis] ||
        left.cell_size[axis] != right.cell_size[axis])
      return false;
  for (int i = 0; i < 9; ++i)
    if (left.metric[i] != right.metric[i]) return false;
  return true;
}

nvidia::geometry_launch_common geometry_common_for(
    const MaterialIR &ir, const MaterialIRDestination &destination,
    const NvidiaBackendState &state, size_t object_offset, size_t object_count,
    size_t image_offset, size_t image_count, size_t value_offset,
    size_t absorber_header_offset, size_t absorber_count, double dt) {
  const ArrayId id = find_storage_key(state.plan_, destination.key);
  if (!is_valid(id)) throw std::invalid_argument("NVIDIA geometry destination is absent");
  const MaterialIRChunk &chunk = ir.chunks[destination.chunk_index];
  const component c = component(destination.component);
  nvidia::geometry_launch_common common = {};
  common.destination = state.arenas_->resolve(id.value).address;
  common.classification = NULL;
  common.object_offset = object_offset;
  common.object_count = object_count;
  common.image_offset = image_offset;
  common.image_count = image_count;
  common.value_offset = value_offset;
  common.material_count = ir.materials.size();
  common.default_material = ir.default_material;
  common.absorber_header_offset = absorber_header_offset;
  common.absorber_count =
      (destination.property == MaterialIRProperty::conductivity ||
       destination.property == MaterialIRProperty::condinv) &&
              (is_D(c) || is_B(c)) &&
              direction(destination.tensor_direction) == component_direction(c)
          ? absorber_count
          : 0;
  common.elements = state.plan_.arrays[id.value].elements;
  common.point_count = destination.point_count;
  common.dimensions = ir.dimensions;
  common.component = destination.component;
  common.tensor_row = component_index(c);
  common.tensor_column = destination.tensor_column;
  common.property = int(destination.property);
  for (int axis = 0; axis < 3; ++axis) {
    common.axis_direction[axis] = material_axis_direction(ir.dimensions, axis);
    common.loop_begin[axis] = chunk.loop_begin[c][axis];
    common.little_corner[axis] = chunk.little_corner[axis];
    const int64_t stagger = int64_t(chunk.loop_begin[c][axis]) - chunk.little_corner[axis];
    const int64_t doubled_extent = int64_t(chunk.loop_end[c][axis]) - chunk.loop_begin[c][axis];
    if (stagger < 0 || stagger > 1 || doubled_extent < 0 || doubled_extent % 2)
      throw std::invalid_argument("NVIDIA geometry loop metadata is invalid");
    common.loop_base_offset[axis] = size_t(stagger / 2) * size_t(chunk.strides[axis]);
    common.loop_extent[axis] = size_t(doubled_extent / 2) + 1;
    common.strides[axis] = chunk.strides[axis];
    const bool shifted = destination.offdiagonal ||
                         (destination.property == MaterialIRProperty::sigma &&
                          direction(destination.tensor_direction) != component_direction(c));
    common.evaluation_shift[axis] =
        shifted &&
                common.axis_direction[axis] == int(component_direction(c))
            ? (type(c) == E_stuff ? -1 : 1)
            : 0;
    common.cell_center[axis] = ir.cell[axis];
    common.cell_size[axis] = ir.cell[axis + 3];
  }
  for (int i = 0; i < 9; ++i) common.metric[i] = ir.lattice_metric[i];
  common.inva = chunk.inva;
  common.dt = dt;
  common.trivial_value = default_material_row_value(destination.key);
  common.logical_single = sizeof(realnum) == sizeof(float);
  common.precision = scalar_precision_for(state.plan_, id, "NVIDIA geometry destination");
  return common;
}

void compile_material_geometry(const MaterialIR &ir, NvidiaBackendState &state, double dt,
                               MaterialCompactPack &compact, size_t absorber_header_offset,
                               size_t absorber_count) {
  std::vector<nvidia::geometry_object_record> objects;
  objects.reserve(ir.objects.size());
  for (const MaterialIRObject &source : ir.objects) {
    nvidia::geometry_object_record object = {};
    if (source.kind < 0 || source.material < 0)
      throw std::invalid_argument("NVIDIA geometry object identity is negative");
    const int subtype = geometry_object_subtype(source);
    if (subtype < 0)
      throw std::invalid_argument("NVIDIA geometry object subtype is negative");
    object.kind = material_checked_uint32(uint64_t(source.kind),
                                          "packing geometry object kind");
    object.subtype = material_checked_uint32(uint64_t(subtype),
                                             "packing geometry object subtype");
    object.material = material_checked_uint32(uint64_t(source.material),
                                              "packing geometry material identity");
    object.closed = source.kind == geometric_object::PRISM
                        ? uint32_t(ir.prism_include_boundaries)
                        : source.kind == geometric_object::MESH
                              ? uint32_t(source.parameters[3] != 0.0)
                              : 0;
    object.parameter_offset = compact.append_doubles(source.parameters,
                                                     "packing geometry parameters");
    object.parameter_count = source.parameters.size();
    object.fixed_vertex_count = source.fixed_vertex_count;
    object.vertex_offset = compact.append_doubles(source.vertices,
                                                  "packing geometry vertices");
    object.vertex_count = source.kind == geometric_object::PRISM
                              ? source.fixed_vertex_count
                              : source.vertices.size() / 3;
    object.index_offset = compact.append_doubles(source.indices, "packing geometry indices");
    object.index_count = source.indices.size();
    object.auxiliary_offset = compact.append_doubles(source.auxiliary,
                                                     "packing geometry auxiliary data");
    object.auxiliary_count = source.auxiliary.size();
    if (source.kind == geometric_object::MESH) {
      std::vector<nvidia::geometry_triangle_record> triangles;
      triangles.reserve(source.triangle_count);
      for (uint64_t i = 0; i < source.triangle_count; ++i) {
        const MaterialIRTriangle &captured =
            ir.geometry_triangles[size_t(source.triangle_offset + i)];
        nvidia::geometry_triangle_record triangle = {};
        for (int vertex = 0; vertex < 3; ++vertex) {
          if (source.vertex_offset > UINT32_MAX ||
              uint64_t(captured.vertex[vertex]) < source.vertex_offset)
            throw std::invalid_argument("NVIDIA geometry triangle vertex is stale");
          triangle.vertex[vertex] = material_checked_uint32(
              uint64_t(captured.vertex[vertex]) - source.vertex_offset,
              "packing geometry triangle vertex");
        }
        for (int axis = 0; axis < 3; ++axis) {
          triangle.normal[axis] = captured.normal[axis];
          triangle.low[axis] = captured.low[axis];
          triangle.high[axis] = captured.high[axis];
        }
        triangles.push_back(triangle);
      }
      object.triangle_offset = compact.append_records(triangles,
                                                      "packing geometry triangles");
      object.triangle_count = triangles.size();
      uint64_t captured_face_begin = std::numeric_limits<uint64_t>::max();
      for (uint64_t i = 0; i < source.bvh_count; ++i) {
        const MaterialIRBvhNode &node = ir.geometry_bvh[size_t(source.bvh_offset + i)];
        if (node.leaf) captured_face_begin = std::min(captured_face_begin, node.first_triangle);
      }
      if (captured_face_begin == std::numeric_limits<uint64_t>::max() ||
          captured_face_begin > ir.geometry_bvh_face_ids.size() ||
          source.triangle_count > ir.geometry_bvh_face_ids.size() - captured_face_begin)
        throw std::invalid_argument("NVIDIA geometry mesh BVH face span is invalid");
      std::vector<uint32_t> faces;
      faces.reserve(source.triangle_count);
      for (uint64_t i = 0; i < source.triangle_count; ++i)
        {
          const uint32_t face = ir.geometry_bvh_face_ids[size_t(captured_face_begin + i)];
          if (uint64_t(face) < source.triangle_offset)
            throw std::invalid_argument("NVIDIA geometry BVH face identity is stale");
          faces.push_back(material_checked_uint32(
              uint64_t(face) - source.triangle_offset,
              "packing geometry BVH face identity"));
        }
      object.face_id_offset = compact.append_records(faces, "packing geometry BVH faces");
      object.face_id_count = faces.size();
      std::vector<nvidia::geometry_bvh_record> nodes;
      nodes.reserve(source.bvh_count);
      for (uint64_t i = 0; i < source.bvh_count; ++i) {
        const MaterialIRBvhNode &captured = ir.geometry_bvh[size_t(source.bvh_offset + i)];
        nvidia::geometry_bvh_record node = {};
        for (int axis = 0; axis < 3; ++axis) {
          node.low[axis] = captured.low[axis];
          node.high[axis] = captured.high[axis];
        }
        node.leaf = captured.leaf;
        if (!captured.leaf &&
            (uint64_t(captured.left) < source.bvh_offset ||
             uint64_t(captured.right) < source.bvh_offset))
          throw std::invalid_argument("NVIDIA geometry BVH child identity is stale");
        node.left = captured.leaf
                        ? UINT32_MAX
                        : material_checked_uint32(
                              uint64_t(captured.left) - source.bvh_offset,
                              "packing geometry BVH left child");
        node.right = captured.leaf
                         ? UINT32_MAX
                         : material_checked_uint32(
                               uint64_t(captured.right) - source.bvh_offset,
                               "packing geometry BVH right child");
        node.first_face = captured.leaf
                              ? captured.first_triangle - captured_face_begin
                              : 0;
        node.face_count = captured.triangle_count;
        nodes.push_back(node);
      }
      if (nodes.empty()) throw std::invalid_argument("NVIDIA geometry mesh BVH is empty");
      std::vector<std::pair<uint32_t, uint32_t> > stack;
      stack.push_back(std::make_pair(
          0u, material_checked_uint32(nodes.size(), "packing geometry BVH node count")));
      while (!stack.empty()) {
        const std::pair<uint32_t, uint32_t> entry = stack.back();
        stack.pop_back();
        if (entry.first >= nodes.size())
          throw std::invalid_argument("NVIDIA geometry mesh BVH child is invalid");
        nvidia::geometry_bvh_record &node = nodes[entry.first];
        node.escape = entry.second;
        if (!node.leaf) {
          stack.push_back(std::make_pair(node.right, entry.second));
          stack.push_back(std::make_pair(node.left, node.right));
        }
      }
      object.bvh_offset = compact.append_records(nodes, "packing geometry BVH nodes");
      object.bvh_count = nodes.size();
      compact.geometry_object_bytes = material_checked_add(
          compact.geometry_object_bytes,
          material_checked_add(
              material_checked_product(triangles.size(),
                                       sizeof(nvidia::geometry_triangle_record),
                                       "accounting geometry triangles"),
              material_checked_add(
                  material_checked_product(nodes.size(), sizeof(nvidia::geometry_bvh_record),
                                           "accounting geometry BVH nodes"),
                  material_checked_product(faces.size(), sizeof(uint32_t),
                                           "accounting geometry BVH faces"),
                  "accounting geometry BVH"),
              "accounting geometry mesh records"),
          "accounting geometry mesh records");
    }
    object.mesh_lengthscale = source.mesh_lengthscale;
    for (int axis = 0; axis < 3; ++axis) {
      object.low[axis] = source.low[axis];
      object.high[axis] = source.high[axis];
    }
    objects.push_back(object);
    compact.geometry_object_bytes = material_checked_add(
        compact.geometry_object_bytes,
        material_checked_product(source.parameters.size() + source.vertices.size() +
                                     source.indices.size() + source.auxiliary.size(),
                                 sizeof(double), "accounting geometry payload"),
        "accounting geometry payload");
  }
  const size_t object_offset = compact.append_records(objects, "packing geometry objects");
  compact.geometry_object_bytes = material_checked_add(
      compact.geometry_object_bytes,
      material_checked_product(objects.size(), sizeof(nvidia::geometry_object_record),
                               "accounting geometry object records"),
      "accounting geometry object records");

  std::vector<nvidia::geometry_image_record> images;
  images.reserve(ir.active_images.size());
  for (uint32_t image_index : ir.active_images) {
    const MaterialIRGeometryImage &source = ir.images[image_index];
    nvidia::geometry_image_record image = {};
    image.object = source.object;
    image.ordinal = source.ordinal;
    image.precedence = source.precedence;
    for (int axis = 0; axis < 3; ++axis) {
      image.image[axis] = source.image[axis];
      image.shift[axis] = source.shift[axis];
      image.low[axis] = source.low[axis];
      image.high[axis] = source.high[axis];
    }
    images.push_back(image);
  }
  const size_t image_offset = compact.append_records(images, "packing geometry images");
  compact.geometry_image_bytes = material_checked_product(
      images.size(), sizeof(nvidia::geometry_image_record), "accounting geometry images");

  std::vector<size_t> sample_offsets(ir.materials.size(), 0);
  std::vector<uint8_t> used(ir.materials.size(), 0);
  used[ir.default_material] = 1;
  for (const MaterialIRObject &object : ir.objects) used[size_t(object.material)] = 1;
  for (size_t i = 0; i < ir.materials.size(); ++i)
    if (used[i] && !ir.materials[i].samples.empty()) {
      sample_offsets[i] = compact.append_doubles(ir.materials[i].samples,
                                                 "packing geometry material samples");
      const size_t bytes = material_checked_product(ir.materials[i].samples.size(),
                                                    sizeof(double),
                                                    "accounting geometry samples");
      if (ir.materials[i].kind == meep_geom::material_data::MATERIAL_FILE)
        compact.file_sample_bytes = material_checked_add(
            compact.file_sample_bytes, bytes, "accounting geometry FILE samples");
      else
        compact.grid_weight_bytes = material_checked_add(
            compact.grid_weight_bytes, bytes, "accounting geometry grid samples");
    }

  std::vector<size_t> value_offsets(ir.destinations.size(), 0);
  for (size_t destination_index = 0; destination_index < ir.destinations.size();
       ++destination_index) {
    std::vector<nvidia::geometry_value_record> values;
    values.reserve(ir.materials.size());
    for (uint32_t material = 0; material < ir.materials.size(); ++material) {
      if (used[material])
        values.push_back(geometry_value_for(ir, material, ir.destinations[destination_index],
                                            sample_offsets[material]));
      else {
        nvidia::geometry_value_record unused = {};
        unused.kind = nvidia::geometry_value_kind::constant;
        unused.flags = 1u;
        values.push_back(unused);
      }
    }
    value_offsets[destination_index] =
        compact.append_records(values, "packing geometry material values");
    compact.geometry_value_bytes = material_checked_add(
        compact.geometry_value_bytes,
        material_checked_product(values.size(), sizeof(nvidia::geometry_value_record),
                                 "accounting geometry values"),
        "accounting geometry values");
  }

  for (size_t source_index = 0; source_index < ir.bulk_spans.size(); ++source_index) {
    const MaterialIRBulkSpan &source = ir.bulk_spans[source_index];
    nvidia::geometry_bulk_launch launch = {};
    launch.common = geometry_common_for(ir, ir.destinations[source.destination], state,
                                        object_offset, objects.size(), image_offset,
                                        images.size(), value_offsets[source.destination],
                                        absorber_header_offset, absorber_count, dt);
    launch.first_point = source.first_point;
    launch.count = source.count;
    state.material_geometry_bulk_launches_.push_back(launch);
    NvidiaMaterialGeometryAuthority authority = {};
    authority.kind = NvidiaMaterialGeometryKind::bulk;
    authority.destination = source.destination;
    authority.source_index = source_index;
    authority.source_count = source.count;
    authority.array = find_storage_key(state.plan_, ir.destinations[source.destination].key);
    authority.topology = ir.topology[ir.destinations[source.destination].topology_index];
    authority.object_offset = launch.common.object_offset;
    authority.object_count = launch.common.object_count;
    authority.image_offset = launch.common.image_offset;
    authority.image_count = launch.common.image_count;
    authority.value_offset = launch.common.value_offset;
    authority.absorber_header_offset = launch.common.absorber_header_offset;
    authority.absorber_count = launch.common.absorber_count;
    authority.first_point = source.first_point;
    state.material_geometry_authorities_.push_back(authority);
  }
  size_t analytic = 0;
  while (analytic < ir.analytic_interfaces.size()) {
    const size_t source_begin = analytic;
    const uint32_t destination = ir.analytic_interfaces[analytic].destination;
    std::vector<nvidia::geometry_analytic_record> jobs;
    do {
      const MaterialIRAnalyticInterface &source = ir.analytic_interfaces[analytic++];
      nvidia::geometry_analytic_record job = {};
      job.point = source.point;
      job.front_material = source.front_material;
      job.behind_material = source.behind_material;
      for (int axis = 0; axis < 3; ++axis) job.normal[axis] = source.normal[axis];
      job.fill = source.fill;
      jobs.push_back(job);
    } while (analytic < ir.analytic_interfaces.size() &&
             ir.analytic_interfaces[analytic].destination == destination);
    nvidia::geometry_analytic_launch launch = {};
    launch.common = geometry_common_for(ir, ir.destinations[destination], state,
                                        object_offset, objects.size(), image_offset,
                                        images.size(), value_offsets[destination],
                                        absorber_header_offset, absorber_count, dt);
    launch.job_offset = compact.append_records(jobs, "packing geometry analytic jobs");
    launch.count = jobs.size();
    state.material_geometry_analytic_launches_.push_back(launch);
    NvidiaMaterialGeometryAuthority authority = {};
    authority.kind = NvidiaMaterialGeometryKind::analytic;
    authority.destination = destination;
    authority.source_index = source_begin;
    authority.source_count = jobs.size();
    authority.array = find_storage_key(state.plan_, ir.destinations[destination].key);
    authority.topology = ir.topology[ir.destinations[destination].topology_index];
    authority.object_offset = launch.common.object_offset;
    authority.object_count = launch.common.object_count;
    authority.image_offset = launch.common.image_offset;
    authority.image_count = launch.common.image_count;
    authority.value_offset = launch.common.value_offset;
    authority.absorber_header_offset = launch.common.absorber_header_offset;
    authority.absorber_count = launch.common.absorber_count;
    authority.record_offset = launch.job_offset;
    state.material_geometry_authorities_.push_back(authority);
    compact.geometry_analytic_bytes = material_checked_add(
        compact.geometry_analytic_bytes,
        material_checked_product(jobs.size(), sizeof(nvidia::geometry_analytic_record),
                                 "accounting geometry analytic jobs"),
        "accounting geometry analytic jobs");
  }
  size_t patch = 0;
  while (patch < ir.hybrid_patches.size()) {
    const size_t source_begin = patch;
    const uint32_t destination = ir.hybrid_patches[patch].destination;
    std::vector<nvidia::geometry_patch_record> patches;
    do {
      const MaterialIRHybridPatch &source = ir.hybrid_patches[patch++];
      patches.push_back(nvidia::geometry_patch_record{source.point, source.value});
    } while (patch < ir.hybrid_patches.size() &&
             ir.hybrid_patches[patch].destination == destination);
    nvidia::geometry_patch_launch launch = {};
    launch.common = geometry_common_for(ir, ir.destinations[destination], state,
                                        object_offset, objects.size(), image_offset,
                                        images.size(), value_offsets[destination],
                                        absorber_header_offset, absorber_count, dt);
    launch.patch_offset = compact.append_records(patches, "packing geometry patches");
    launch.count = patches.size();
    state.material_geometry_patch_launches_.push_back(launch);
    NvidiaMaterialGeometryAuthority authority = {};
    authority.kind = NvidiaMaterialGeometryKind::patch;
    authority.destination = destination;
    authority.source_index = source_begin;
    authority.source_count = patches.size();
    authority.array = find_storage_key(state.plan_, ir.destinations[destination].key);
    authority.topology = ir.topology[ir.destinations[destination].topology_index];
    authority.object_offset = launch.common.object_offset;
    authority.object_count = launch.common.object_count;
    authority.image_offset = launch.common.image_offset;
    authority.image_count = launch.common.image_count;
    authority.value_offset = launch.common.value_offset;
    authority.absorber_header_offset = launch.common.absorber_header_offset;
    authority.absorber_count = launch.common.absorber_count;
    authority.record_offset = launch.patch_offset;
    state.material_geometry_authorities_.push_back(authority);
    compact.geometry_patch_bytes = material_checked_add(
        compact.geometry_patch_bytes,
        material_checked_product(patches.size(), sizeof(nvidia::geometry_patch_record),
                                 "accounting geometry patches"),
        "accounting geometry patches");
  }
}

void validate_material_geometry_authority(
    const MaterialIR &ir, const NvidiaBackendState &state, const MaterialCompactPack &compact,
    const NvidiaMaterialGeometryAuthority &authority,
    const nvidia::geometry_launch_common &common, uint64_t first_point, size_t record_offset,
    size_t count, double dt, bool require_device_pointer) {
  if (authority.destination >= ir.destinations.size())
    throw std::invalid_argument("NVIDIA geometry authority destination is invalid");
  if (authority.array.value >= state.plan_.arrays.size())
    throw std::invalid_argument("NVIDIA geometry authority ArrayId is invalid");
  if (!(state.plan_.keys[authority.array.value] ==
        ir.destinations[authority.destination].key))
    throw std::invalid_argument("NVIDIA geometry authority StorageKey is stale");
  if (!(authority.topology ==
        ir.topology[ir.destinations[authority.destination].topology_index]))
    throw std::invalid_argument("NVIDIA geometry authority topology is stale");
  if (!(authority.array == find_storage_key(state.plan_, authority.topology.key)))
    throw std::invalid_argument("NVIDIA geometry authority binding is stale");
  if (authority.source_count != count || authority.first_point != first_point ||
      authority.record_offset != record_offset)
    throw std::invalid_argument("NVIDIA geometry authority partition is stale");
  nvidia::geometry_launch_common expected = geometry_common_for(
      ir, ir.destinations[authority.destination], state, authority.object_offset,
      authority.object_count, authority.image_offset, authority.image_count,
      authority.value_offset, authority.absorber_header_offset, authority.absorber_count,
      dt);
  expected.compact_input_bytes = compact.bytes.size();
  expected.compact_inputs = require_device_pointer
                                ? static_cast<const unsigned char *>(
                                      state.material_ir_inputs_.opaque_handle())
                                : NULL;
  if (!same_geometry_common(expected, common))
    throw std::invalid_argument("NVIDIA geometry launch descriptor authority is stale");
  const void *expected_destination = state.arenas_->resolve(authority.array.value).address;
  const unsigned char *expected_inputs = require_device_pointer
                                             ? static_cast<const unsigned char *>(
                                                   state.material_ir_inputs_.opaque_handle())
                                             : NULL;
  if (common.destination != expected_destination || common.compact_inputs != expected_inputs ||
      common.compact_input_bytes != compact.bytes.size())
    throw std::invalid_argument("NVIDIA geometry launch pointer authority is stale");
  if (authority.kind == NvidiaMaterialGeometryKind::bulk) {
    if (authority.source_index >= ir.bulk_spans.size())
      throw std::invalid_argument("NVIDIA geometry bulk authority is absent");
    const MaterialIRBulkSpan &source = ir.bulk_spans[authority.source_index];
    if (source.destination != authority.destination || source.first_point != first_point ||
        source.count != count)
      throw std::invalid_argument("NVIDIA geometry bulk partition authority is stale");
  }
  else if (authority.kind == NvidiaMaterialGeometryKind::analytic) {
    if (authority.source_index > ir.analytic_interfaces.size() ||
        count > ir.analytic_interfaces.size() - authority.source_index ||
        !material_compact_range<nvidia::geometry_analytic_record>(record_offset, count,
                                                                  compact.bytes.size()))
      throw std::invalid_argument("NVIDIA geometry analytic authority is absent");
    const nvidia::geometry_analytic_record *records =
        reinterpret_cast<const nvidia::geometry_analytic_record *>(compact.bytes.data() +
                                                                    record_offset);
    for (size_t i = 0; i < count; ++i) {
      const MaterialIRAnalyticInterface &source = ir.analytic_interfaces[authority.source_index + i];
      if (source.destination != authority.destination || records[i].point != source.point ||
          records[i].front_material != source.front_material ||
          records[i].behind_material != source.behind_material ||
          !same_material_double(records[i].fill, source.fill))
        throw std::invalid_argument("NVIDIA geometry analytic record authority is stale");
      for (int axis = 0; axis < 3; ++axis)
        if (!same_material_double(records[i].normal[axis], source.normal[axis]))
          throw std::invalid_argument("NVIDIA geometry analytic normal authority is stale");
    }
  }
  else {
    if (authority.source_index > ir.hybrid_patches.size() ||
        count > ir.hybrid_patches.size() - authority.source_index ||
        !material_compact_range<nvidia::geometry_patch_record>(record_offset, count,
                                                               compact.bytes.size()))
      throw std::invalid_argument("NVIDIA geometry patch authority is absent");
    const nvidia::geometry_patch_record *records =
        reinterpret_cast<const nvidia::geometry_patch_record *>(compact.bytes.data() +
                                                                 record_offset);
    for (size_t i = 0; i < count; ++i) {
      const MaterialIRHybridPatch &source = ir.hybrid_patches[authority.source_index + i];
      if (source.destination != authority.destination || records[i].point != source.point ||
          !same_material_double(records[i].value, source.value))
        throw std::invalid_argument("NVIDIA geometry patch record authority is stale");
    }
  }
}

void validate_all_material_geometry_authority(const MaterialIR &ir,
                                              const NvidiaBackendState &state,
                                              const MaterialCompactPack &compact,
                                              double dt, bool require_device_pointer) {
  if (state.material_geometry_compact_hash_ != geometry_compact_hash(compact.bytes))
    throw std::invalid_argument("NVIDIA geometry compact authority is stale");
  size_t bulk = 0, analytic = 0, patch = 0;
  for (const NvidiaMaterialGeometryAuthority &authority :
       state.material_geometry_authorities_) {
    if (authority.kind == NvidiaMaterialGeometryKind::bulk) {
      if (bulk >= state.material_geometry_bulk_launches_.size())
        throw std::invalid_argument("NVIDIA geometry bulk launch authority is incomplete");
      const nvidia::geometry_bulk_launch &launch = state.material_geometry_bulk_launches_[bulk++];
      validate_material_geometry_authority(ir, state, compact, authority, launch.common,
                                           launch.first_point, 0, launch.count,
                                           dt, require_device_pointer);
      nvidia::validate_geometry_bulk_launch(launch, compact.bytes.data(), compact.bytes.size());
    }
    else if (authority.kind == NvidiaMaterialGeometryKind::analytic) {
      if (analytic >= state.material_geometry_analytic_launches_.size())
        throw std::invalid_argument("NVIDIA geometry analytic launch authority is incomplete");
      const nvidia::geometry_analytic_launch &launch =
          state.material_geometry_analytic_launches_[analytic++];
      validate_material_geometry_authority(ir, state, compact, authority, launch.common, 0,
                                           launch.job_offset, launch.count,
                                           dt, require_device_pointer);
      nvidia::validate_geometry_analytic_launch(launch, compact.bytes.data(),
                                                compact.bytes.size());
    }
    else {
      if (patch >= state.material_geometry_patch_launches_.size())
        throw std::invalid_argument("NVIDIA geometry patch launch authority is incomplete");
      const nvidia::geometry_patch_launch &launch = state.material_geometry_patch_launches_[patch++];
      validate_material_geometry_authority(ir, state, compact, authority, launch.common, 0,
                                           launch.patch_offset, launch.count,
                                           dt, require_device_pointer);
      nvidia::validate_geometry_patch_launch(launch, compact.bytes.data(), compact.bytes.size());
    }
  }
  if (bulk != state.material_geometry_bulk_launches_.size() ||
      analytic != state.material_geometry_analytic_launches_.size() ||
      patch != state.material_geometry_patch_launches_.size())
    throw std::invalid_argument("NVIDIA geometry launch authority count is stale");
}

void clear_material_initialization_launches(NvidiaBackendState &state) {
  state.material_fill_launches_.clear();
  state.material_table_launches_.clear();
  state.material_table_authorities_.clear();
  state.material_conductivity_launches_.clear();
  state.material_pml_launches_.clear();
  state.material_geometry_bulk_launches_.clear();
  state.material_geometry_analytic_launches_.clear();
  state.material_geometry_patch_launches_.clear();
  state.material_geometry_authorities_.clear();
  state.material_geometry_compact_hash_ = 0;
}

void compile_device_native_material_initialization(const MaterialRecipe &recipe,
                                                   NvidiaBackendState &state, const fields &f,
                                                   double dt,
                                                   MaterialCompactPack &compact) {
  if ((recipe.disposition() != MaterialRecipeDisposition::device_native &&
       recipe.disposition() != MaterialRecipeDisposition::hybrid_interface) ||
      !recipe.ir())
    throw std::invalid_argument("NVIDIA native material initialization requires owned native IR");
  const MaterialIR &ir = *recipe.ir();
  validate_material_ir(ir);
  validate_material_ir_against_live(ir, f, dt);
  const bool geometry_partition = !ir.objects.empty() || !ir.analytic_interfaces.empty() ||
                                  !ir.hybrid_patches.empty();
  if (!geometry_partition) preflight_native_table_ir(ir);
  const MaterialIRMaterial &root = ir.materials[ir.default_material];
  if (root.kind != meep_geom::material_data::MEDIUM &&
      root.kind != meep_geom::material_data::PERFECT_METAL &&
      root.kind != meep_geom::material_data::MATERIAL_FILE &&
      root.kind != meep_geom::material_data::MATERIAL_GRID)
    throw std::invalid_argument("NVIDIA native material kind is unsupported");
  if (!geometry_partition && root.kind == meep_geom::material_data::MATERIAL_GRID &&
      root.do_averaging)
    throw std::invalid_argument("NVIDIA table initialization does not support grid averaging");
  if (!std::isfinite(dt) || !(dt > 0.0))
    throw std::invalid_argument("NVIDIA native material initialization has an invalid timestep");

  std::unique_ptr<OwnedMediumView> medium;
  SymmetricMaterialTensor tensors[2] = {};
  if (root.kind == meep_geom::material_data::MEDIUM) {
    medium.reset(new OwnedMediumView(parse_owned_medium(root)));
    tensors[0] = inverse_owned_tensor(*medium, E_stuff);
    tensors[1] = inverse_owned_tensor(*medium, H_stuff);
  }
  const bool file_table = root.kind == meep_geom::material_data::MATERIAL_FILE;
  const bool grid_table = root.kind == meep_geom::material_data::MATERIAL_GRID;
  std::unique_ptr<OwnedMediumView> table_medium_1, table_medium_2;
  size_t table_header_offset = size_t(-1);

  clear_material_initialization_launches(state);
  compact = MaterialCompactPack();
  if (ir.topology.empty()) return;
  std::set<uint32_t> destinations;
  std::vector<std::pair<uintptr_t, uintptr_t> > destination_ranges;
  std::map<StorageKey, const MaterialIRTopologyRow *, MaterialStorageKeyLess> rows;

  for (const MaterialIRTopologyRow &row_spec : ir.topology) {
    const ArrayId id = find_storage_key(state.plan_, row_spec.key);
    if (!is_valid(id) || id.value >= state.plan_.arrays.size())
      throw std::invalid_argument("NVIDIA material topology destination is absent from storage");
    const ArraySpec &spec = state.plan_.arrays[id.value];
    if (spec.role != array_role::material || spec.element_type != row_spec.element_type ||
        spec.elements != row_spec.elements || spec.alignment != row_spec.alignment ||
        is_valid(spec.alias_of) ||
        (spec.storage != Precision::f32 && spec.storage != Precision::f64))
      throw std::invalid_argument("NVIDIA material topology destination shape is inconsistent");
    if (!destinations.insert(id.value).second)
      throw std::invalid_argument("NVIDIA material topology writes a destination twice");
    rows[row_spec.key] = &row_spec;
    const nvidia::device_allocation allocation = state.arenas_->resolve(id.value);
    const size_t bytes = storage_bytes(spec);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(allocation.address);
    if (!begin || begin % (spec.storage == Precision::f32 ? alignof(float) : alignof(double)) ||
        bytes > std::numeric_limits<uintptr_t>::max() - begin)
      throw std::invalid_argument("NVIDIA material destination alignment is invalid");
    const uintptr_t end = begin + bytes;
    for (const std::pair<uintptr_t, uintptr_t> &range : destination_ranges)
      if (begin < range.second && range.first < end)
        throw std::invalid_argument("NVIDIA material destinations overlap");
    destination_ranges.push_back(std::make_pair(begin, end));
  }
  for (size_t i = 0; i < state.plan_.arrays.size(); ++i)
    if (state.plan_.arrays[i].role == array_role::material &&
        !is_valid(state.plan_.arrays[i].alias_of) && !rows.count(state.plan_.keys[i]))
      throw std::invalid_argument("NVIDIA material storage is absent from owned topology");

  if (!geometry_partition && (file_table || grid_table)) {
    nvidia::material_table_header header = {};
    header.version = 1;
    header.material_id = ir.default_material;
    header.kind = file_table ? nvidia::material_table_kind::file_scalar_epsilon
                             : nvidia::material_table_kind::material_grid;
    header.overlap_kind = grid_table ? uint32_t(root.material_grid_kind)
                                     : uint32_t(meep_geom::material_data::U_DEFAULT);
    size_t parameter_offset = file_table ? 0 : 3;
    OwnedMediumView first = parse_owned_medium_at(root.parameters, parameter_offset);
    if (grid_table) {
      OwnedMediumView second = parse_owned_medium_at(root.parameters, parameter_offset);
      if (root.parameters.size() - parameter_offset != 3)
        throw std::invalid_argument("NVIDIA material grid payload has the wrong tail");
      header.beta = root.parameters[parameter_offset];
      header.eta = root.parameters[parameter_offset + 1];
      header.damping = root.parameters[parameter_offset + 2];
      header.projection_offset = ir.projection_offset;
      table_medium_1.reset(new OwnedMediumView(first));
      table_medium_2.reset(new OwnedMediumView(second));
    }
    else if (root.parameters.size() - parameter_offset != 3)
      throw std::invalid_argument("NVIDIA FILE payload has the wrong dimension tail");
    size_t product = 1;
    for (int axis = 0; axis < 3; ++axis) {
      const size_t extent = owned_count(
          root.parameters[(file_table ? parameter_offset : 0) + axis],
          file_table ? "FILE dimension" : "MaterialGrid dimension");
      if (!extent || extent > size_t(INT_MAX))
        throw std::invalid_argument("NVIDIA material table dimension is not positive int-sized");
      product = material_checked_product(product, extent, "sizing material table samples");
      header.dimensions[axis] = uint32_t(extent);
    }
    if (product != root.samples.size())
      throw std::invalid_argument("NVIDIA material table sample count is inconsistent");
    table_header_offset = compact.append_space(sizeof(header), alignof(decltype(header)),
                                               "packing material table header");
    if (grid_table) {
      header.medium_1_offset = pack_material_medium(compact, *table_medium_1, ir);
      header.medium_2_offset = pack_material_medium(compact, *table_medium_2, ir);
    }
    header.sample_offset = compact.append_doubles(
        root.samples, file_table ? "packing FILE samples" : "packing MaterialGrid weights");
    header.sample_count = root.samples.size();
    memcpy(compact.bytes.data() + table_header_offset, &header, sizeof(header));
    compact.table_header_offsets.push_back(table_header_offset);
    const size_t sample_bytes = material_checked_product(root.samples.size(), sizeof(double),
                                                         "accounting table samples");
    if (file_table) compact.file_sample_bytes = sample_bytes;
    else compact.grid_weight_bytes = sample_bytes;
  }

  /* Phase zero establishes every padding/default element before any
     dependency-bearing override. */
  for (const MaterialIRTopologyRow &row_spec : ir.topology) {
    const array_kind kind = static_cast<array_kind>(row_spec.key.kind);
    if (kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
        kind == array_kind::pml_siginv)
      continue;
    const ArrayId id = find_storage_key(state.plan_, row_spec.key);
    nvidia::material_fill_launch launch;
    launch.destination = state.arenas_->resolve(id.value).address;
    launch.classification = NULL;
    launch.elements = state.plan_.arrays[id.value].elements;
    launch.classification_elements =
        logical_material_classification_elements(ir, row_spec.key);
    launch.value = default_material_row_value(row_spec.key);
    launch.trivial_value = default_material_row_value(row_spec.key);
    launch.phase = 0;
    launch.logical_single = sizeof(realnum) == sizeof(float);
    launch.precision = scalar_precision_for(state.plan_, id, "NVIDIA material destination");
    state.material_fill_launches_.push_back(launch);
  }

  std::vector<nvidia::material_absorber_header> absorber_headers;
  if (!ir.absorbers.empty()) {
    const size_t header_bytes = material_checked_product(
        ir.absorbers.size(), sizeof(nvidia::material_absorber_header),
        "sizing absorber headers");
    compact.absorber_header_offset = compact.append_space(
        header_bytes, alignof(nvidia::material_absorber_header), "packing absorber headers");
    absorber_headers.reserve(ir.absorbers.size());
    for (const MaterialIRPml &absorber : ir.absorbers) {
      bool active_direction = false;
      for (int axis = 0; axis < 3; ++axis)
        active_direction = active_direction ||
                           material_axis_direction(ir.dimensions, axis) == absorber.direction;
      if (!active_direction || absorber.samples.size() < 2 || !(absorber.sample_spacing > 0.0))
        throw std::invalid_argument("NVIDIA absorber profile has an unsupported shape");
      nvidia::material_absorber_header header = {};
      header.version = 1;
      header.direction = absorber.direction;
      header.side = absorber.side;
      header.sample_offset =
          compact.append_doubles(absorber.samples, "packing absorber profile samples");
      header.sample_count = absorber.samples.size();
      header.thickness = absorber.thickness;
      header.sample_spacing = absorber.sample_spacing;
      if (header.sample_offset % alignof(double) || header.sample_count < 2 ||
          header.sample_offset > compact.bytes.size() ||
          header.sample_count >
              (compact.bytes.size() - size_t(header.sample_offset)) / sizeof(double) ||
          header.direction < int(X) || header.direction > int(P) ||
          (header.side != int(High) && header.side != int(Low)) ||
          header.sample_spacing != header.thickness / double(header.sample_count - 1))
        throw std::invalid_argument("NVIDIA absorber compact header is invalid");
      absorber_headers.push_back(header);
      compact.absorber_profile_bytes = material_checked_add(
          compact.absorber_profile_bytes,
          material_checked_product(absorber.samples.size(), sizeof(double),
                                   "accounting absorber profile samples"),
          "accounting absorber profile samples");
    }
    memcpy(compact.bytes.data() + compact.absorber_header_offset, absorber_headers.data(),
           header_bytes);
    if (compact.absorber_header_offset % alignof(nvidia::material_absorber_header) ||
        compact.absorber_header_offset > compact.bytes.size() ||
        header_bytes > compact.bytes.size() - compact.absorber_header_offset)
      throw std::logic_error("NVIDIA absorber compact header range changed while packing");
    nvidia::validate_material_absorber_headers(compact.bytes.data(), compact.bytes.size(),
                                                compact.absorber_header_offset,
                                                absorber_headers.size());
  }

  /* Constant overrides execute in the legacy dependency order.  Absorber
     conductivity and its derived inverse are emitted as one ordered kernel
     below, so their logical predecessor is never reread after narrowing. */
  if (!geometry_partition && !file_table && !grid_table)
    for (int phase = 1; phase <= 5; ++phase)
      for (const MaterialIRTopologyRow &row_spec : ir.topology) {
      const array_kind kind = static_cast<array_kind>(row_spec.key.kind);
      if (material_fill_phase(kind) != phase) continue;
      const component c = component(row_spec.key.component_);
      const bool absorber_pair = !ir.absorbers.empty() &&
                                 (kind == array_kind::conductivity ||
                                  kind == array_kind::condinv) &&
                                 (is_D(c) || is_B(c)) &&
                                 direction(row_spec.key.aux) == component_direction(c);
      if (absorber_pair) continue;
      const ArrayId id = find_storage_key(state.plan_, row_spec.key);
      nvidia::material_fill_launch launch;
      launch.destination = state.arenas_->resolve(id.value).address;
      launch.classification = NULL;
      launch.elements = state.plan_.arrays[id.value].elements;
      launch.classification_elements =
          logical_material_classification_elements(ir, row_spec.key);
      launch.value = homogeneous_row_value(ir, medium.get(), tensors, row_spec.key, dt);
      launch.trivial_value = default_material_row_value(row_spec.key);
      launch.phase = uint32_t(phase);
      launch.logical_single = sizeof(realnum) == sizeof(float);
      launch.precision = scalar_precision_for(state.plan_, id, "NVIDIA material destination");
      state.material_fill_launches_.push_back(launch);
      }

  if (!geometry_partition && !ir.absorbers.empty())
    for (const MaterialIRTopologyRow &row_spec : ir.topology) {
      if (static_cast<array_kind>(row_spec.key.kind) != array_kind::conductivity) continue;
      const component c = component(row_spec.key.component_);
      if ((!is_D(c) && !is_B(c)) || direction(row_spec.key.aux) != component_direction(c))
        continue;
      if (grid_table && is_D(c)) continue;
      StorageKey inverse_key = row_spec.key;
      inverse_key.kind = int(array_kind::condinv);
      const ArrayId conductivity = find_storage_key(state.plan_, row_spec.key);
      const ArrayId inverse = find_storage_key(state.plan_, inverse_key);
      if (!is_valid(inverse))
        throw std::invalid_argument("NVIDIA absorber conductivity inverse destination is absent");
      const nvidia::scalar_precision precision = scalar_precision_for(
          state.plan_, conductivity, "NVIDIA absorber conductivity destination");
      require_same_precision(state.plan_, inverse, precision,
                             "NVIDIA absorber conductivity inverse");
      const MaterialIRChunk &chunk = material_chunk(ir, row_spec.key.chunk);
      nvidia::material_conductivity_launch launch = {};
      launch.conductivity_destination = state.arenas_->resolve(conductivity.value).address;
      launch.condinv_destination = state.arenas_->resolve(inverse.value).address;
      launch.conductivity_classification = NULL;
      launch.condinv_classification = NULL;
      launch.compact_inputs = NULL;
      launch.compact_input_bytes = compact.bytes.size();
      launch.absorber_header_offset = compact.absorber_header_offset;
      launch.absorber_count = absorber_headers.size();
      launch.elements = row_spec.elements;
      launch.loop_count = chunk.loop_count[c];
      launch.component = int(c);
      launch.dimensions = ir.dimensions;
      for (int axis = 0; axis < 3; ++axis) {
        launch.axis_direction[axis] = material_axis_direction(ir.dimensions, axis);
        launch.loop_begin[axis] = chunk.loop_begin[c][axis];
        launch.little_corner[axis] = chunk.little_corner[axis];
        const int64_t stagger = int64_t(chunk.loop_begin[c][axis]) -
                                int64_t(chunk.little_corner[axis]);
        const int64_t doubled_extent = int64_t(chunk.loop_end[c][axis]) -
                                       int64_t(chunk.loop_begin[c][axis]);
        if (stagger < 0 || stagger > 1 || doubled_extent < 0 || doubled_extent % 2)
          throw std::invalid_argument("NVIDIA absorber loop geometry is invalid");
        launch.loop_base_offset[axis] =
            size_t(stagger / 2) * size_t(chunk.strides[axis]);
        launch.loop_extent[axis] = size_t(doubled_extent / 2) + 1;
        launch.strides[axis] = chunk.strides[axis];
      }
      for (int direction = 0; direction < 5; ++direction) launch.cell_size[direction] = 0.0;
      launch.cell_size[X] = ir.cell[3];
      launch.cell_size[Y] = ir.cell[4];
      launch.cell_size[Z] = ir.cell[5];
      launch.cell_size[R] = ir.cell[3];
      launch.inva = chunk.inva;
      launch.base_conductivity = medium
                                     ? (*medium->values)[medium->base +
                                                         (is_D(c) ? 30 : 33) +
                                                         component_index(c)]
                                     : 0.0;
      launch.dt = dt;
      launch.logical_single = sizeof(realnum) == sizeof(float);
      launch.precision = precision;
      state.material_conductivity_launches_.push_back(launch);
    }

  if (!geometry_partition && (file_table || grid_table)) {
    const auto make_table_launch = [&](const MaterialIRTopologyRow &row_spec,
                                       nvidia::material_table_operation operation) {
      const component c = component(row_spec.key.component_);
      const ArrayId destination = find_storage_key(state.plan_, row_spec.key);
      nvidia::material_table_launch launch = {};
      launch.destination = state.arenas_->resolve(destination.value).address;
      launch.classification = NULL;
      launch.secondary_classification = NULL;
      launch.compact_inputs = NULL;
      launch.compact_input_bytes = compact.bytes.size();
      launch.table_header_offset = table_header_offset;
      launch.elements = row_spec.elements;
      const MaterialIRChunk &chunk = material_chunk(ir, row_spec.key.chunk);
      launch.loop_count = chunk.loop_count[c];
      launch.table_kind = file_table ? nvidia::material_table_kind::file_scalar_epsilon
                                     : nvidia::material_table_kind::material_grid;
      launch.operation = operation;
      launch.source_material_id = ir.default_material;
      launch.destination_component = int(c);
      launch.query_component = int(c);
      launch.tensor_row = component_index(c);
      launch.tensor_column = -1;
      launch.susceptibility_field_type = int(NO_FIELD_TYPE);
      launch.dimensions = ir.dimensions;
      for (int axis = 0; axis < 3; ++axis) {
        launch.axis_direction[axis] = material_axis_direction(ir.dimensions, axis);
        launch.loop_begin[axis] = chunk.loop_begin[c][axis];
        launch.loop_end[axis] = chunk.loop_end[c][axis];
        launch.little_corner[axis] = chunk.little_corner[axis];
        const int64_t stagger = int64_t(chunk.loop_begin[c][axis]) -
                                int64_t(chunk.little_corner[axis]);
        const int64_t doubled_extent = int64_t(chunk.loop_end[c][axis]) -
                                       int64_t(chunk.loop_begin[c][axis]);
        if (stagger < 0 || stagger > 1 || doubled_extent < 0 || doubled_extent % 2)
          throw std::invalid_argument("NVIDIA material table loop geometry is invalid");
        launch.loop_base_offset[axis] =
            size_t(stagger / 2) * size_t(chunk.strides[axis]);
        launch.loop_extent[axis] = size_t(doubled_extent / 2) + 1;
        launch.strides[axis] = chunk.strides[axis];
        launch.evaluation_shift[axis] = 0;
      }
      launch.cell_center[0] = ir.cell[0];
      launch.cell_center[1] = ir.cell[1];
      launch.cell_center[2] = ir.cell[2];
      launch.cell_size[0] = ir.cell[3];
      launch.cell_size[1] = ir.cell[4];
      launch.cell_size[2] = ir.cell[5];
      launch.inva = chunk.inva;
      launch.dt = dt;
      launch.logical_single = sizeof(realnum) == sizeof(float);
      launch.trivial_value = default_material_row_value(row_spec.key);
      launch.secondary_trivial_value = 1.0;
      launch.precision =
          scalar_precision_for(state.plan_, destination, "NVIDIA material table destination");
      return launch;
    };
    const auto add_table_launch = [&](const MaterialIRTopologyRow &row_spec,
                                      const nvidia::material_table_launch &launch,
                                      ArrayId secondary_destination = invalid_array()) {
      const ArrayId destination = find_storage_key(state.plan_, row_spec.key);
      if (!is_valid(destination))
        throw std::invalid_argument("NVIDIA table authority destination is absent");
      NvidiaMaterialTableAuthority authority;
      authority.row = row_spec;
      authority.destination = destination;
      authority.secondary_destination = secondary_destination;
      authority.table_header_offset = table_header_offset;
      state.material_table_launches_.push_back(launch);
      state.material_table_authorities_.push_back(authority);
    };

    for (const MaterialIRTopologyRow &row_spec : ir.topology) {
      const array_kind kind = static_cast<array_kind>(row_spec.key.kind);
      if (kind == array_kind::chi1inv) {
        const component c = component(row_spec.key.component_);
        const int column = material_tensor_column(c, int(row_spec.key.aux));
        if (!is_electric(c) || column < 0 ||
            (file_table && column != component_index(c)))
          continue;
        nvidia::material_table_launch launch = make_table_launch(
            row_spec, file_table ? nvidia::material_table_operation::file_chi1inv
                                 : nvidia::material_table_operation::grid_chi1inv);
        launch.tensor_column = column;
        if (column != launch.tensor_row)
          for (int axis = 0; axis < 3; ++axis)
            if (launch.axis_direction[axis] == int(component_direction(c)))
              launch.evaluation_shift[axis] = -1;
        add_table_launch(row_spec, launch);
      }
      else if (grid_table && kind == array_kind::conductivity) {
        const component c = component(row_spec.key.component_);
        if (!is_D(c) || direction(row_spec.key.aux) != component_direction(c)) continue;
        StorageKey inverse_key = row_spec.key;
        inverse_key.kind = int(array_kind::condinv);
        const ArrayId inverse = find_storage_key(state.plan_, inverse_key);
        if (!is_valid(inverse))
          throw std::invalid_argument("NVIDIA grid conductivity inverse is absent");
        nvidia::material_table_launch launch = make_table_launch(
            row_spec, nvidia::material_table_operation::grid_conductivity);
        launch.secondary_destination = state.arenas_->resolve(inverse.value).address;
        launch.absorber_header_offset = compact.absorber_header_offset;
        launch.absorber_count = absorber_headers.size();
        require_same_precision(state.plan_, inverse, launch.precision,
                               "NVIDIA grid conductivity inverse");
        launch.tensor_column = launch.tensor_row;
        add_table_launch(row_spec, launch, inverse);
      }
      else if (grid_table && kind == array_kind::sigma) {
        const component c = component(row_spec.key.component_);
        const field_type ft = field_type(row_spec.key.aux % uint64_t(NUM_FIELD_TYPES));
        const uint64_t identity_value = row_spec.key.aux / uint64_t(NUM_FIELD_TYPES);
        const int column = material_tensor_column(c, row_spec.key.cmp);
        if (!is_electric(c) || ft != E_stuff || column < 0 ||
            identity_value > std::numeric_limits<uint32_t>::max())
          continue;
        const MaterialIRSusceptibility *identity = NULL;
        for (const MaterialIRSusceptibility &sus : ir.susceptibilities)
          if (sus.field_type == ft && sus.identity == identity_value) {
            identity = &sus;
            break;
          }
        if (!identity)
          throw std::invalid_argument("NVIDIA grid sigma identity is absent");
        unsigned source_medium = 0;
        size_t source_ordinal = 0;
        for (size_t i = 0; i < table_medium_1->susceptibilities[0].size(); ++i)
          if (same_owned_susceptibility(*table_medium_1->values,
                                        table_medium_1->susceptibilities[0][i],
                                        identity->parameters)) {
            source_medium = 1;
            source_ordinal = i;
            break;
          }
        if (!source_medium)
          for (size_t i = 0; i < table_medium_2->susceptibilities[0].size(); ++i)
            if (same_owned_susceptibility(*table_medium_2->values,
                                          table_medium_2->susceptibilities[0][i],
                                          identity->parameters)) {
              source_medium = 2;
              source_ordinal = i;
              break;
            }
        if (!source_medium) continue;
        const OwnedMediumView &source =
            source_medium == 1 ? *table_medium_1 : *table_medium_2;
        if (owned_sigma_value(source, *identity, c, row_spec.key.cmp) == 0.0) continue;
        nvidia::material_table_launch launch =
            make_table_launch(row_spec, nvidia::material_table_operation::grid_sigma);
        launch.source_medium = source_medium;
        launch.source_susceptibility = source_ordinal;
        launch.susceptibility_identity = uint32_t(identity_value);
        launch.susceptibility_field_type = int(ft);
        launch.tensor_column = column;
        if (column != launch.tensor_row)
          for (int axis = 0; axis < 3; ++axis)
            if (launch.axis_direction[axis] == int(component_direction(c)))
              launch.evaluation_shift[axis] = -1;
        add_table_launch(row_spec, launch);
      }
    }
  }

  if (geometry_partition)
    compile_material_geometry(ir, state, dt, compact, compact.absorber_header_offset,
                              absorber_headers.size());

  std::vector<size_t> pml_profile_offsets(ir.pml_axes.size(), size_t(-1));
  for (size_t i = 0; i < ir.pml_axes.size(); ++i)
    if (ir.pml_axes[i].profile_active) {
      pml_profile_offsets[i] = compact.append_doubles(
          ir.pml_axes[i].profile_samples, "packing PML profile samples");
      compact.pml_profile_bytes = material_checked_add(
          compact.pml_profile_bytes,
          material_checked_product(ir.pml_axes[i].profile_samples.size(), sizeof(double),
                                   "accounting PML profile samples"),
          "accounting PML profile samples");
    }

  size_t pml_axis_index = 0;
  for (const MaterialIRPmlAxis &axis : ir.pml_axes) {
    const StorageKey sig_key = {axis.chunk, int(array_kind::pml_sig), -1, -1,
                                uint64_t(axis.direction)};
    const StorageKey kap_key = {axis.chunk, int(array_kind::pml_kap), -1, -1,
                                uint64_t(axis.direction)};
    const StorageKey inv_key = {axis.chunk, int(array_kind::pml_siginv), -1, -1,
                                uint64_t(axis.direction)};
    const ArrayId sig = find_storage_key(state.plan_, sig_key);
    const ArrayId kap = find_storage_key(state.plan_, kap_key);
    const ArrayId inv = find_storage_key(state.plan_, inv_key);
    if (!is_valid(sig) || !is_valid(kap) || !is_valid(inv))
      throw std::invalid_argument("NVIDIA material PML group is incomplete");
    const nvidia::scalar_precision precision =
        scalar_precision_for(state.plan_, sig, "NVIDIA material PML sigma");
    require_same_precision(state.plan_, kap, precision, "NVIDIA material PML kappa");
    require_same_precision(state.plan_, inv, precision, "NVIDIA material PML sigma inverse");
    if (state.plan_.arrays[sig.value].elements != axis.elements ||
        state.plan_.arrays[kap.value].elements != axis.elements ||
        state.plan_.arrays[inv.value].elements != axis.elements)
      throw std::invalid_argument("NVIDIA material PML extent differs from owned IR");
    nvidia::material_pml_launch launch;
    launch.sigma_destination = state.arenas_->resolve(sig.value).address;
    launch.kappa_destination = state.arenas_->resolve(kap.value).address;
    launch.sigma_inv_destination = state.arenas_->resolve(inv.value).address;
    launch.sigma_classification = NULL;
    launch.kappa_classification = NULL;
    launch.sigma_inv_classification = NULL;
    launch.compact_inputs = NULL;
    launch.compact_input_bytes = compact.bytes.size();
    launch.profile_offset = axis.profile_active ? pml_profile_offsets[pml_axis_index] : 0;
    launch.elements = axis.elements;
    launch.little_corner = axis.little_corner;
    launch.resolution = axis.resolution;
    launch.dt = dt;
    launch.thickness = axis.thickness;
    launch.boundary_location = axis.boundary_location;
    launch.r_asymptotic = axis.r_asymptotic;
    launch.mean_stretch = axis.mean_stretch;
    launch.profile_integral = axis.profile_integral;
    launch.profile_integral_u = axis.profile_integral_u;
    launch.thickness_cells = 0;
    if (axis.profile_active) {
      /* Match structure.cpp:pml_x exactly; a long-double intermediate changes
         the integer result for representable half-cell boundary cases. */
      const double thickness_cells =
          axis.thickness * (2 * axis.resolution) + 0.5;
      if (!std::isfinite(thickness_cells) || thickness_cells <= 0 ||
          thickness_cells > double(INT_MAX))
        throw std::overflow_error("NVIDIA material PML thickness conversion overflows");
      launch.thickness_cells = int(thickness_cells);
    }
    launch.profile_active = axis.profile_active;
    launch.analytic_quadratic = axis.analytic_quadratic;
    launch.logical_single = sizeof(realnum) == sizeof(float);
    launch.precision = precision;
    state.material_pml_launches_.push_back(launch);
    ++pml_axis_index;
  }

  const size_t input_bytes = compact.bytes.size();
  if (input_bytes) {
    for (nvidia::material_table_launch &launch : state.material_table_launches_)
      launch.compact_input_bytes = input_bytes;
    for (nvidia::geometry_bulk_launch &launch : state.material_geometry_bulk_launches_)
      launch.common.compact_input_bytes = input_bytes;
    for (nvidia::geometry_analytic_launch &launch : state.material_geometry_analytic_launches_)
      launch.common.compact_input_bytes = input_bytes;
    for (nvidia::geometry_patch_launch &launch : state.material_geometry_patch_launches_)
      launch.common.compact_input_bytes = input_bytes;
    state.material_geometry_compact_hash_ = geometry_compact_hash(compact.bytes);
    if (!state.material_geometry_bulk_launches_.empty() &&
        nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_geometry_descriptor_mutation))
      ++state.material_geometry_bulk_launches_.front().common.evaluation_shift[0];
    if (!state.material_geometry_authorities_.empty() && !compact.bytes.empty() &&
        nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_geometry_compact_mutation))
      compact.bytes.back() ^= 1u;
    if (!state.material_table_launches_.empty() &&
        nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_table_semantic_mutation))
      state.material_table_launches_.front().axis_direction[0] = 2;
    if (!state.material_table_launches_.empty() &&
        nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_table_far_coordinate_mutation)) {
      nvidia::material_table_launch &launch = state.material_table_launches_.front();
      launch.loop_begin[2] += 1000000;
      launch.loop_end[2] += 1000000;
      launch.little_corner[2] += 1000000;
    }
    if (!state.material_table_launches_.empty() &&
        nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_table_component_mutation)) {
      nvidia::material_table_launch &launch = state.material_table_launches_.front();
      launch.destination_component = launch.query_component =
          launch.destination_component == int(Ex) ? int(Ey) : int(Ex);
      launch.tensor_row = component_index(component(launch.destination_component));
      launch.tensor_column = launch.tensor_row;
    }
    if (!state.material_table_launches_.empty() &&
        nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_table_tensor_mutation))
      state.material_table_launches_.front().tensor_column =
          (state.material_table_launches_.front().tensor_column + 1) % 3;
    if (!state.material_table_launches_.empty() &&
        nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_table_loop_mutation)) {
      nvidia::material_table_launch &launch = state.material_table_launches_.front();
      launch.loop_begin[2] += 2;
      launch.loop_end[2] += 2;
      launch.little_corner[2] += 2;
    }
    if (!state.material_table_launches_.empty() &&
        nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_table_shift_mutation))
      state.material_table_launches_.front().evaluation_shift[2] = -1;
    if (!state.material_table_launches_.empty() &&
        nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_table_destination_mutation)) {
      nvidia::material_table_launch &launch = state.material_table_launches_.front();
      const size_t width = launch.precision == nvidia::scalar_precision::f32
                               ? sizeof(float)
                               : sizeof(double);
      launch.destination = static_cast<unsigned char *>(launch.destination) + width;
    }
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_table_source_mutation)) {
      bool mutated = false;
      for (nvidia::material_table_launch &launch : state.material_table_launches_)
        if (launch.operation == nvidia::material_table_operation::grid_sigma) {
          launch.source_medium = launch.source_medium == 1 ? 2 : 1;
          launch.source_susceptibility = 0;
          mutated = true;
          break;
        }
      if (!mutated)
        throw std::logic_error("NVIDIA table source mutation fixture has no sigma launch");
    }
    if (!state.material_table_launches_.empty() &&
        nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_table_header_mutation)) {
      nvidia::material_table_header *header =
          reinterpret_cast<nvidia::material_table_header *>(
              compact.bytes.data() + compact.table_header_offsets.front());
      ++header->material_id;
      ++state.material_table_launches_.front().source_material_id;
    }
    if (!compact.table_header_offsets.empty())
      nvidia::validate_material_table_headers(compact.bytes.data(), compact.bytes.size(),
                                               compact.table_header_offsets.data(),
                                               compact.table_header_offsets.size());
    if (state.material_table_launches_.size() != state.material_table_authorities_.size())
      throw std::logic_error("NVIDIA material table authority count differs");
    for (size_t i = 0; i < state.material_table_launches_.size(); ++i)
      validate_material_table_authority(ir, state, compact, dt,
                                        state.material_table_launches_[i],
                                        state.material_table_authorities_[i], false);
    if (!absorber_headers.empty())
      nvidia::validate_material_absorber_headers(compact.bytes.data(), compact.bytes.size(),
                                                  compact.absorber_header_offset,
                                                  absorber_headers.size());
    validate_all_material_geometry_authority(ir, state, compact, dt, false);
  }
}

void bind_material_compact_inputs(NvidiaBackendState &state, size_t input_bytes) {
  if (!input_bytes) {
    state.material_ir_inputs_.reset();
    return;
  }
  if (input_bytes > nvidia::free_memory_for_device(state.device_))
    throw std::runtime_error("NVIDIA material compact input exceeds available device memory");
  if (nvidia::testing::consume_failure_for_testing(
          nvidia::testing::failure_point::material_compact_allocate))
    throw std::runtime_error("injected NVIDIA material compact allocation failure");
  state.material_ir_inputs_.allocate(input_bytes, state.device_);
  const unsigned char *base = static_cast<const unsigned char *>(
      state.material_ir_inputs_.opaque_handle());
  for (nvidia::material_table_launch &launch : state.material_table_launches_) {
    launch.compact_inputs = base;
    launch.compact_input_bytes = input_bytes;
  }
  for (nvidia::material_conductivity_launch &launch : state.material_conductivity_launches_) {
    launch.compact_inputs = base;
    launch.compact_input_bytes = input_bytes;
  }
  for (nvidia::material_pml_launch &launch : state.material_pml_launches_) {
    launch.compact_inputs = base;
    launch.compact_input_bytes = input_bytes;
  }
  for (nvidia::geometry_bulk_launch &launch : state.material_geometry_bulk_launches_) {
    launch.common.compact_inputs = base;
    launch.common.compact_input_bytes = input_bytes;
  }
  for (nvidia::geometry_analytic_launch &launch : state.material_geometry_analytic_launches_) {
    launch.common.compact_inputs = base;
    launch.common.compact_input_bytes = input_bytes;
  }
  for (nvidia::geometry_patch_launch &launch : state.material_geometry_patch_launches_) {
    launch.common.compact_inputs = base;
    launch.common.compact_input_bytes = input_bytes;
  }
}

void bind_material_classification(NvidiaBackendState &state) {
  unsigned char *const base = static_cast<unsigned char *>(
      state.material_classification_status_.opaque_handle());
  const auto status_for_destination = [&](const void *destination) -> unsigned char * {
    for (const NvidiaMaterialClassificationRow &row : state.material_classification_rows_)
      if (state.arenas_->resolve(row.allocation).address == destination)
        return base + row.status_offset;
    throw std::logic_error("NVIDIA material writer has no classification row");
  };
  for (nvidia::material_fill_launch &launch : state.material_fill_launches_)
    launch.classification = status_for_destination(launch.destination);
  for (nvidia::material_table_launch &launch : state.material_table_launches_) {
    launch.classification = status_for_destination(launch.destination);
    if (launch.secondary_destination)
      launch.secondary_classification = status_for_destination(launch.secondary_destination);
  }
  for (nvidia::material_conductivity_launch &launch : state.material_conductivity_launches_) {
    launch.conductivity_classification =
        status_for_destination(launch.conductivity_destination);
    launch.condinv_classification = status_for_destination(launch.condinv_destination);
  }
  for (nvidia::material_pml_launch &launch : state.material_pml_launches_) {
    launch.sigma_classification = status_for_destination(launch.sigma_destination);
    launch.kappa_classification = status_for_destination(launch.kappa_destination);
    launch.sigma_inv_classification = status_for_destination(launch.sigma_inv_destination);
  }
  for (nvidia::geometry_bulk_launch &launch : state.material_geometry_bulk_launches_)
    launch.common.classification = status_for_destination(launch.common.destination);
  for (nvidia::geometry_analytic_launch &launch : state.material_geometry_analytic_launches_)
    launch.common.classification = status_for_destination(launch.common.destination);
  for (nvidia::geometry_patch_launch &launch : state.material_geometry_patch_launches_)
    launch.common.classification = status_for_destination(launch.common.destination);
}

size_t exact_initialization_staging_bytes(const StoragePlan &plan,
                                          MaterialRecipeDisposition route,
                                          size_t compact_input_bytes,
                                          size_t callback_scratch_bytes,
                                          size_t material_status_bytes,
                                          size_t material_rows) {
  size_t bytes = 0;
  size_t descriptors = 0;
  const bool native_material = route == MaterialRecipeDisposition::device_native ||
                               route == MaterialRecipeDisposition::hybrid_interface;
  for (const ArraySpec &spec : plan.arrays) {
    if (is_valid(spec.alias_of) || (native_material && spec.role == array_role::material))
      continue;
    const size_t array_bytes = storage_bytes(spec);
    bytes = material_checked_add(bytes, array_bytes,
                                 "sizing NVIDIA initialization staging");
    ++descriptors;
  }
  if (!native_material)
    for (const ArraySpec &spec : plan.arrays)
      if (!is_valid(spec.alias_of) && spec.role == array_role::material)
        bytes = material_checked_add(
            bytes,
            material_checked_product(spec.elements, sizeof(realnum),
                                     "sizing NVIDIA logical material staging"),
            "sizing NVIDIA logical material staging");
  bytes = material_checked_add(bytes, material_status_bytes,
                               "sizing NVIDIA material classification status");
  const size_t result_bytes = material_checked_add(
      sizeof(nvidia::material_classification_header),
      material_checked_product(material_rows,
                               sizeof(nvidia::material_classification_row_result),
                               "sizing NVIDIA material classification rows"),
      "sizing NVIDIA material classification result");
  bytes = material_compact_append_extent(
      bytes, result_bytes, alignof(nvidia::material_classification_row_result),
      "sizing aligned NVIDIA material classification result");
  if (native_material)
    bytes = material_checked_add(bytes, compact_input_bytes,
                                 "sizing NVIDIA compact-input staging");
  bytes = material_checked_add(bytes, callback_scratch_bytes,
                               "sizing NVIDIA callback scratch");
  bytes = material_checked_add(
      bytes,
      material_checked_product(descriptors, sizeof(NvidiaInitializationUpload),
                               "sizing NVIDIA initialization descriptors"),
      "sizing NVIDIA initialization descriptors");
  return bytes;
}

} // namespace

BackendState *NvidiaBackend::create_state(const StoragePlan &plan) {
  std::unique_ptr<NvidiaBackendState> state;
  std::string local_error;
  try {
    StoragePlan device_plan = plan;
    apply_precision_policy(device_plan, policy_for(options_.precision));
    std::string why;
    if (!validate_alias_precisions(device_plan, why))
      throw std::invalid_argument(std::string("invalid NVIDIA storage aliases: ") + why);
    if (next_state_token_ == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("NVIDIA backend state token overflow");
    const uint64_t state_token = next_state_token_++;
    const nvidia::arena_plan layout(allocation_requests_for(device_plan));
    size_t classification_status_bytes = 0;
    for (const std::pair<StorageKey, size_t> &row :
         pending_initialization_classification_rows_)
      classification_status_bytes = material_checked_add(
          classification_status_bytes, row.second,
          "sizing NVIDIA device material classification status");
    size_t material_rows = 0;
    for (const ArraySpec &spec : device_plan.arrays)
      if (!is_valid(spec.alias_of) && spec.role == array_role::material) ++material_rows;
    if (material_rows != pending_initialization_classification_rows_.size())
      throw std::logic_error(
          "NVIDIA material classification preflight row count changed");
    size_t classification_device_bytes = material_checked_add(
        classification_status_bytes,
        material_checked_add(
            sizeof(nvidia::material_classification_header),
            material_checked_product(material_rows,
                                     sizeof(nvidia::material_classification_row_result),
                                     "sizing NVIDIA device material classification rows"),
            "sizing NVIDIA device material classification result"),
        "sizing NVIDIA device material classification buffers");
    const size_t base_reserve_bytes = pending_initialization_reserve_valid_
                                          ? pending_initialization_reserve_bytes_
                                          : 2 * sizeof(nvidia::noisy_seed_block);
    const size_t device_reserve_bytes = material_checked_add(
        base_reserve_bytes, classification_device_bytes,
        "sizing NVIDIA initialization classification reserve");
    const size_t compact_input_bytes = pending_initialization_reserve_valid_
                                           ? pending_initialization_compact_bytes_
                                           : 0;
    const MaterialRecipeDisposition route = pending_initialization_reserve_valid_
                                                ? pending_initialization_route_
                                                : MaterialRecipeDisposition::host_reference;
    const size_t staging_bytes = exact_initialization_staging_bytes(
        device_plan, route, compact_input_bytes,
        pending_initialization_reserve_valid_ ? pending_initialization_scratch_bytes_ : 0,
        classification_status_bytes, material_rows);
    if (device_reserve_bytes >
        std::numeric_limits<size_t>::max() - layout.total_reserved_bytes())
      throw std::overflow_error("NVIDIA initialization peak-memory admission overflows");
    const size_t device_peak = layout.total_reserved_bytes() + device_reserve_bytes;
    if (staging_bytes > std::numeric_limits<size_t>::max() - device_peak)
      throw std::overflow_error("NVIDIA initialization staging peak overflows");
    const size_t combined_peak = device_peak + staging_bytes;
    const size_t physical_free = nvidia::free_memory_for_device(device_);
    const size_t test_budget = nvidia::testing::initialization_memory_budget_for_testing();
    if (device_peak > physical_free || combined_peak > test_budget) {
      std::ostringstream message;
      message << "NVIDIA initialization peak requires " << combined_peak
              << " bytes (resident arenas " << layout.total_reserved_bytes()
              << " plus device compact/auxiliary reserve " << device_reserve_bytes
              << " plus pinned staging " << staging_bytes << "), but device "
              << device_ << " has " << physical_free << " device bytes and the admission budget is "
              << test_budget << " bytes";
      throw std::runtime_error(message.str());
    }
    state.reset(new NvidiaBackendState(this, device_plan, device_, state_token,
                                       device_reserve_bytes, compact_input_bytes,
                                       staging_bytes,
                                       pending_initialization_classification_rows_));
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
  pending_initialization_reserve_valid_ = false;
  pending_initialization_reserve_bytes_ = 0;
  pending_initialization_compact_bytes_ = 0;
  pending_initialization_scratch_bytes_ = 0;
  pending_initialization_classification_rows_.clear();
  pending_initialization_route_ = MaterialRecipeDisposition::device_native;
  return state.release();
}

void NvidiaBackend::prepare_initialization(const InitializationPlan &initialization,
                                           BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  state.initialization_prepared_ = false;
  state.initialization_uploads_.clear();
  state.prepared_material_rows_ = 0;
  state.prepared_compact_staging_offset_ = 0;
  state.prepared_callback_scratch_offset_ = 0;
  state.material_initialization_statistics_ = NvidiaMaterialInitializationStatistics();
  if (state.transfer_failed_)
    throw std::logic_error("NVIDIA transfer stream failed; recreate backend state");
  if (state.device_authoritative_)
    throw std::logic_error(
        "cannot prepare NVIDIA storage from a stale host mirror after device stepping");
  if (!f_.array_catalog)
    throw std::logic_error("NVIDIA initialization requires a prepared CPU catalog");
  if (initialization.materials.size() != 1)
    throw std::logic_error("NVIDIA initialization requires one frozen material recipe");
  const MaterialRecipe &material = initialization.materials[0];
  validate_material_recipe(material);
  const MaterialSupportDecision support = classify_material_support(material);
  if (material.disposition() != MaterialRecipeDisposition::device_native &&
      material.disposition() != MaterialRecipeDisposition::hybrid_interface &&
      material.disposition() != MaterialRecipeDisposition::host_reference &&
      material.disposition() != MaterialRecipeDisposition::tiled_callback)
    throw std::invalid_argument("NVIDIA initialization received an unsupported route");
  const bool native_material =
      material.disposition() == MaterialRecipeDisposition::device_native ||
      material.disposition() == MaterialRecipeDisposition::hybrid_interface;
  const CpuArrayCatalog &catalog = *f_.array_catalog;
  if (catalog.size() > state.plan_.arrays.size())
    throw std::logic_error("CPU catalog exceeds the provisional NVIDIA storage plan");
  /* The authoritative prefix is fixed when the state is created. A resolved
     live catalog also contains classification tombstones; promoting that
     larger size here would silently turn previously elidable rows into
     retained rows during a regional/value refresh. */
  if (!state.initialized_) state.prepared_authoritative_arrays_ = catalog.size();
  else if (catalog.size() < state.prepared_authoritative_arrays_)
    throw std::logic_error("NVIDIA live catalog lost its authoritative prefix");

  std::map<uint32_t, const InitOperation *> regional_operations;
  if (initialization.regional)
    for (const InitOperation &operation : initialization.operations) {
      if (operation.kind != InitKind::material_geometry &&
          operation.kind != InitKind::pml_profile)
        throw std::invalid_argument("regional initialization contains a non-material producer");
      if (!regional_operations.insert(
              std::make_pair(operation.destination.id.value, &operation)).second)
        throw std::invalid_argument("regional initialization repeats a destination");
    }

  MaterialCompactPack compact_inputs;
  if (native_material && !state.material_classification_rows_.empty())
    compile_device_native_material_initialization(material, state, f_, double(f_.dt),
                                                  compact_inputs);
  else
    clear_material_initialization_launches(state);
  if (compact_inputs.bytes.size() != state.initialization_compact_input_bytes_)
    throw std::logic_error(
        "NVIDIA material compact-input size differs from pre-allocation admission");

  size_t cursor = 0;
  for (size_t i = 0; i < state.plan_.arrays.size(); ++i) {
    const ArraySpec &device_spec = state.plan_.arrays[i];
    if (i < catalog.size()) {
      const ArraySpec &host_spec = catalog.spec(ArrayId{uint32_t(i)});
      if (host_spec.id != device_spec.id || host_spec.role != device_spec.role ||
          host_spec.element_type != device_spec.element_type ||
          host_spec.elements != device_spec.elements || host_spec.alias_of != device_spec.alias_of)
        throw std::logic_error("CPU catalog no longer matches the NVIDIA storage plan");
    }
    else if (device_spec.role != array_role::material ||
             !device_spec.classification_provisional || is_valid(device_spec.alias_of))
      throw std::logic_error("NVIDIA provisional suffix is not an owned material row");
    if (is_valid(device_spec.alias_of)) {
      if (catalog.resolve_untyped(device_spec.id) != catalog.resolve_untyped(device_spec.alias_of))
        throw std::logic_error("NVIDIA storage alias does not match the CPU catalog alias");
      continue;
    }
    const auto regional_operation = regional_operations.find(device_spec.id.value);
    if (initialization.regional &&
        (device_spec.role != array_role::material ||
         regional_operation == regional_operations.end() || device_spec.classification_elided))
      continue;
    if (native_material && device_spec.role == array_role::material) continue;
    const size_t bytes = storage_bytes(device_spec);
    NvidiaInitializationUpload upload = {
        device_spec.id.value, bytes, cursor, size_t(-1),
        device_spec.role == array_role::material, {}};
    if (!initialization.regional || regional_operation->second->point_spans.empty())
      upload.ranges.push_back(NvidiaInitializationUpload::Range{0, bytes});
    else {
      if (!material.ir() ||
          regional_operation->second->descriptor_index >= material.ir()->destinations.size())
        throw std::invalid_argument("regional upload has no canonical destination");
      const MaterialIRDestination &destination =
          material.ir()->destinations[regional_operation->second->descriptor_index];
      const size_t element_bytes = storage_bytes(device_spec) / device_spec.elements;
      std::vector<size_t> physical;
      for (const InitPointSpan &span : regional_operation->second->point_spans)
        for (uint64_t logical = span.first; logical - span.first < span.count; ++logical)
          physical.push_back(
              material_ir_destination_storage_index(*material.ir(), destination, logical));
      std::sort(physical.begin(), physical.end());
      physical.erase(std::unique(physical.begin(), physical.end()), physical.end());
      for (size_t index : physical) {
        if (index >= device_spec.elements)
          throw std::out_of_range("regional upload point exceeds destination storage");
        const size_t offset = material_checked_product(
            index, element_bytes, "laying out regional material upload");
        if (!upload.ranges.empty() &&
            upload.ranges.back().offset + upload.ranges.back().bytes == offset)
          upload.ranges.back().bytes = material_checked_add(
              upload.ranges.back().bytes, element_bytes,
              "coalescing regional material upload");
        else
          upload.ranges.push_back(NvidiaInitializationUpload::Range{offset, element_bytes});
      }
      if (upload.ranges.empty()) continue;
    }
    state.initialization_uploads_.push_back(upload);
    cursor = material_checked_add(cursor, bytes, "laying out NVIDIA initialization uploads");
  }
  state.prepared_compact_staging_offset_ = cursor;
  cursor = material_checked_add(cursor, compact_inputs.bytes.size(),
                                "laying out NVIDIA compact input staging");
  if (!native_material)
    for (NvidiaInitializationUpload &upload : state.initialization_uploads_)
      if (upload.material) {
        upload.logical_offset = cursor;
        const ArraySpec &spec = state.plan_.arrays[upload.allocation];
        cursor = material_checked_add(
            cursor,
            material_checked_product(spec.elements, sizeof(realnum),
                                     "laying out NVIDIA logical material output"),
            "laying out NVIDIA logical material output");
      }
  state.prepared_status_staging_offset_ = cursor;
  size_t classification_status_bytes = 0;
  for (NvidiaMaterialClassificationRow &row : state.material_classification_rows_) {
    row.trivial_value = default_material_row_value(row.key);
    classification_status_bytes = material_checked_add(
        classification_status_bytes, row.logical_elements,
        "laying out NVIDIA classification status");
  }
  cursor = material_checked_add(cursor, classification_status_bytes,
                                "laying out NVIDIA classification status");
  cursor = material_compact_append_extent(
      cursor, state.material_classification_result_bytes_,
      alignof(nvidia::material_classification_row_result),
      "laying out aligned NVIDIA classification result",
      &state.prepared_result_staging_offset_);
  state.prepared_callback_scratch_offset_ = cursor;
  cursor = material_checked_add(cursor, size_t(support.scratch_bytes),
                                "laying out NVIDIA callback scratch");
  cursor = material_checked_add(
      cursor,
      material_checked_product(state.initialization_uploads_.size(),
                               sizeof(NvidiaInitializationUpload),
                               "laying out NVIDIA initialization descriptors"),
      "laying out NVIDIA initialization descriptors");
  if ((!initialization.regional && cursor != state.initialization_staging_bytes_) ||
      cursor > state.initialization_staging_bytes_ ||
      state.staging_.size() != state.initialization_staging_bytes_)
    throw std::logic_error(
        "NVIDIA initialization staging differs from pre-allocation admission");

  unsigned char *const staging = static_cast<unsigned char *>(state.staging_.data());
  for (const NvidiaInitializationUpload &upload : state.initialization_uploads_) {
    const ArraySpec &spec = state.plan_.arrays[upload.allocation];
    if (!upload.material) {
      const void *source = catalog.resolve_untyped(spec.id);
      if (!source) throw std::logic_error("CPU catalog contains a null canonical allocation");
      host_to_storage(staging + upload.staging_offset, source, spec, spec.elements);
      continue;
    }
    if (upload.logical_offset == size_t(-1))
      throw std::logic_error("NVIDIA material upload has no logical staging span");
    const MaterialRecipeRow *row = NULL;
    const MaterialRecipeRow *primary = NULL;
    const MaterialIRTopologyRow *topology = NULL;
    for (const MaterialRecipeRow &candidate : material.dense_fallback_rows())
      if (candidate.key == state.plan_.keys[upload.allocation]) row = &candidate;
    for (const MaterialRecipeRow &candidate : material.rows())
      if (candidate.key == state.plan_.keys[upload.allocation]) primary = &candidate;
    for (const MaterialIRTopologyRow &candidate : material.topology())
      if (candidate.key == state.plan_.keys[upload.allocation]) topology = &candidate;
    const bool row_matches = row && row->role == spec.role &&
                             row->element_type == spec.element_type &&
                             row->elements == spec.elements && row->alignment == spec.alignment;
    const bool topology_matches = topology && spec.role == array_role::material &&
                                  topology->element_type == spec.element_type &&
                                  topology->elements == spec.elements &&
                                  topology->alignment == spec.alignment;
    const bool primary_matches = primary && primary->role == spec.role &&
                                 primary->element_type == spec.element_type &&
                                 primary->elements == spec.elements &&
                                 primary->alignment == spec.alignment;
    if ((material.disposition() == MaterialRecipeDisposition::host_reference &&
         !row_matches) ||
        (material.disposition() == MaterialRecipeDisposition::tiled_callback &&
         !topology_matches && !primary_matches)) {
      std::ostringstream message;
      message << "material recipe row no longer matches device storage (route="
              << material_recipe_disposition_name(material.disposition())
              << ", id=" << upload.allocation << ", chunk="
              << state.plan_.keys[upload.allocation].chunk << ", kind="
              << state.plan_.keys[upload.allocation].kind << ", component="
              << state.plan_.keys[upload.allocation].component_ << ", cmp="
              << state.plan_.keys[upload.allocation].cmp << ", aux="
              << state.plan_.keys[upload.allocation].aux << ", dense=" << bool(row)
              << ", topology=" << bool(topology) << ")";
      throw std::logic_error(message.str());
    }
    unsigned char *const logical = staging + upload.logical_offset;
    const size_t logical_bytes = material_checked_product(
        spec.elements, sizeof(realnum), "staging logical material output");
    if (material.disposition() == MaterialRecipeDisposition::host_reference) {
      if (row->values.size() != logical_bytes)
        throw std::logic_error("host-reference material row is incomplete");
      memcpy(logical, row->values.data(), logical_bytes);
    }
    else {
      const realnum value = realnum(default_material_row_value(state.plan_.keys[upload.allocation]));
      for (size_t point = 0; point < spec.elements; ++point)
        memcpy(logical + point * sizeof(realnum), &value, sizeof(value));
    }
    ++state.prepared_material_rows_;
  }

  if (material.disposition() == MaterialRecipeDisposition::tiled_callback) {
    if (!material.ir()) throw std::logic_error("tiled callback route has no immutable IR");
    const MaterialIR &ir = *material.ir();
    const MaterialIRMaterial &callback = ir.materials[ir.default_material];
    const OwnedMaterialCallback *owner = NULL;
    for (const std::shared_ptr<const OwnedMaterialCallback> &candidate :
         material.callback_owners())
      if (candidate && candidate->id == callback.callback_id) owner = candidate.get();
    if (!owner || owner->signature != callback.callback_signature ||
        owner->capabilities != callback.callback_capabilities ||
        owner->capabilities != owned_material_callback_tiled_capabilities)
      throw std::logic_error("tiled callback lifetime/capability token is stale");
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_callback_prepare))
      throw std::runtime_error("injected owned tiled callback preparation failure");
    static thread_local bool callback_active = false;
    if (callback_active)
      throw std::logic_error("recursive owned tiled material callback is unsupported");
    struct CallbackScope {
      bool &active;
      explicit CallbackScope(bool &active_) : active(active_) { active = true; }
      ~CallbackScope() { active = false; }
    } scope(callback_active);
    /* The declared pure/replay-stable contract makes destination-major calls
       observationally equivalent to the legacy shared-evaluation grouping.
       Exact shifted centers and scatter indices still come from the frozen IR. */
    realnum *const scratch = reinterpret_cast<realnum *>(
        staging + state.prepared_callback_scratch_offset_);
    const size_t scratch_points = size_t(support.scratch_bytes) / sizeof(realnum);
    const std::vector<MaterialCallbackTile> &callback_tiles =
        initialization.regional ? initialization.material_callback_tiles
                                : material.callback_tiles();
    for (const MaterialCallbackTile &tile : callback_tiles) {
      if (tile.destination >= ir.destinations.size())
        throw std::logic_error("tiled callback destination is out of range");
      const MaterialIRDestination &destination = ir.destinations[tile.destination];
      if (destination.property != MaterialIRProperty::chi1inv ||
          tile.material != ir.default_material)
        throw std::logic_error("tiled callback contains an unsupported output");
      const ArrayId id = find_storage_key(state.plan_, destination.key);
      NvidiaInitializationUpload *upload = NULL;
      for (NvidiaInitializationUpload &candidate : state.initialization_uploads_)
        if (candidate.material && candidate.allocation == id.value) upload = &candidate;
      if (!upload || upload->logical_offset == size_t(-1))
        throw std::logic_error("tiled callback destination has no staged output");
      if (!tile.count || tile.count > scratch_points || tile.count > 256)
        throw std::logic_error("tiled callback exceeds its bounded scratch span");
      unsigned char *const logical = staging + upload->logical_offset;
      for (uint64_t local = 0; local < tile.count; ++local) {
        const uint64_t point = tile.first_point + local;
        meep_geom::medium_struct medium;
        owner->function(meep_geom::vec_to_vector3(
                            material_ir_destination_center(ir, destination, point)),
                        medium);
        validate_tiled_callback_output(medium);
        const component c = component(destination.component);
        const SymmetricMaterialTensor inverse = inverse_callback_tensor(medium, type(c));
        const int row_index = component_index(c);
        const double value = destination.tensor_column < 0
                                 ? 0.0
                                 : inverse.value[row_index][destination.tensor_column];
        scratch[local] = realnum(value);
        ++state.material_fallback_statistics.callback_calls;
      }
      for (uint64_t local = 0; local < tile.count; ++local) {
        const uint64_t point = tile.first_point + local;
        const size_t index =
            material_ir_destination_storage_index(ir, destination, point);
        memcpy(logical + index * sizeof(realnum), scratch + local, sizeof(realnum));
      }
    }
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_callback_dispatch))
      throw std::runtime_error("injected owned tiled callback dispatch failure");
  }

  if (!native_material)
    for (const NvidiaInitializationUpload &upload : state.initialization_uploads_)
      if (upload.material) {
        const ArraySpec &spec = state.plan_.arrays[upload.allocation];
        host_to_storage(staging + upload.staging_offset,
                        staging + upload.logical_offset, spec, spec.elements);
      }
  if (!native_material)
    for (const NvidiaMaterialClassificationRow &row : state.material_classification_rows_) {
      const NvidiaInitializationUpload *upload = NULL;
      for (const NvidiaInitializationUpload &candidate : state.initialization_uploads_)
        if (candidate.material && candidate.allocation == row.allocation) upload = &candidate;
      const unsigned char *logical = NULL;
      if (upload && upload->logical_offset != size_t(-1))
        logical = staging + upload->logical_offset;
      else if (initialization.regional) {
        for (const MaterialRecipeRow &candidate : material.rows())
          if (candidate.key == row.key && !candidate.values.empty()) {
            logical = candidate.values.data();
            break;
          }
        if (!logical)
          for (const MaterialRecipeRow &candidate : material.dense_fallback_rows())
            if (candidate.key == row.key && !candidate.values.empty()) {
              logical = candidate.values.data();
              break;
            }
      }
      if (!logical) throw std::logic_error("material classification row has no logical output");
      unsigned char *status = staging + state.prepared_status_staging_offset_ + row.status_offset;
      for (size_t i = 0; i < row.logical_elements; ++i) {
        realnum value;
        const size_t physical = logical_material_storage_index(material, row.key, i);
        if (physical >= state.plan_.arrays[row.allocation].elements)
          throw std::logic_error(
              "NVIDIA material logical classification index exceeds physical storage");
        memcpy(&value, logical + physical * sizeof(realnum), sizeof(value));
        status[i] = 1u | (double(value) != row.trivial_value ? 2u : 0u);
      }
    }
  if (!compact_inputs.bytes.empty())
    memcpy(staging + state.prepared_compact_staging_offset_, compact_inputs.bytes.data(),
           compact_inputs.bytes.size());

  state.material_initialization_statistics_.owned_ir_bytes = support.compact_input_bytes;
  state.material_initialization_statistics_.dense_oracle_bytes = support.dense_fallback_bytes;
  for (const NvidiaInitializationUpload &upload : state.initialization_uploads_)
    if (upload.material)
      state.material_initialization_statistics_.logical_output_bytes = material_checked_add(
          state.material_initialization_statistics_.logical_output_bytes,
          material_checked_product(state.plan_.arrays[upload.allocation].elements,
                                   sizeof(realnum),
                                   "accounting logical material outputs"),
          "accounting logical material outputs");
  state.material_initialization_statistics_.callback_scratch_bytes = support.scratch_bytes;
  state.material_initialization_statistics_.upload_descriptor_bytes =
      material_checked_product(state.initialization_uploads_.size(),
                               sizeof(NvidiaInitializationUpload),
                               "accounting initialization upload descriptors");
  state.material_initialization_statistics_.classification_status_bytes =
      state.material_classification_status_.size();
  state.material_initialization_statistics_.classification_result_bytes =
      state.material_classification_result_bytes_;
  state.material_initialization_statistics_.absorber_profile_bytes =
      compact_inputs.absorber_profile_bytes;
  state.material_initialization_statistics_.pml_profile_bytes = compact_inputs.pml_profile_bytes;
  state.material_initialization_statistics_.file_sample_bytes = compact_inputs.file_sample_bytes;
  state.material_initialization_statistics_.grid_weight_bytes = compact_inputs.grid_weight_bytes;
  state.material_initialization_statistics_.geometry_object_bytes =
      compact_inputs.geometry_object_bytes;
  state.material_initialization_statistics_.geometry_image_bytes =
      compact_inputs.geometry_image_bytes;
  state.material_initialization_statistics_.geometry_value_bytes =
      compact_inputs.geometry_value_bytes;
  state.material_initialization_statistics_.geometry_analytic_bytes =
      compact_inputs.geometry_analytic_bytes;
  state.material_initialization_statistics_.geometry_patch_bytes =
      compact_inputs.geometry_patch_bytes;
  state.material_initialization_statistics_.decoded_parameter_bytes =
      compact_inputs.bytes.size() - compact_inputs.absorber_profile_bytes -
      compact_inputs.pml_profile_bytes - compact_inputs.file_sample_bytes -
      compact_inputs.grid_weight_bytes;
  state.prepared_material_route_ = material.disposition();
  state.prepared_recipe_signature_ = material.signature();
  state.initialization_prepared_ = true;
}

void NvidiaBackend::initialize(const InitializationPlan &initialization, BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  const bool live_refresh = state.initialized_;
  std::string local_error;
  try {
    nvidia::device_scope device_scope(state.device_);
    if (state.transfer_failed_)
      throw std::logic_error("NVIDIA transfer stream failed; recreate backend state");
    if (!state.initialization_prepared_)
      throw std::logic_error("NVIDIA initialization was not prepared collectively");
    if (initialization.materials.size() != 1 ||
        initialization.materials[0].signature() != state.prepared_recipe_signature_ ||
        initialization.materials[0].disposition() != state.prepared_material_route_)
      throw std::logic_error("NVIDIA prepared initialization identity is stale");
    const bool native_material =
        state.prepared_material_route_ == MaterialRecipeDisposition::device_native ||
        state.prepared_material_route_ == MaterialRecipeDisposition::hybrid_interface;
    const size_t compact_input_bytes = state.initialization_compact_input_bytes_;
    if (native_material) bind_material_compact_inputs(state, compact_input_bytes);
    else state.material_ir_inputs_.reset();
    bind_material_classification(state);
    bool has_tiled_upload = false;
    for (const NvidiaInitializationUpload &upload : state.initialization_uploads_)
      has_tiled_upload = has_tiled_upload ||
                         (upload.material &&
                          state.prepared_material_route_ !=
                              MaterialRecipeDisposition::host_reference);
    if (has_tiled_upload && nvidia::testing::consume_failure_for_testing(
                                nvidia::testing::failure_point::material_tiled_upload))
      throw std::runtime_error("injected NVIDIA tiled material upload failure");
    if (native_material && compact_input_bytes &&
        nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_ir_upload))
      throw std::runtime_error("injected NVIDIA material IR upload failure");
    if (live_refresh && nvidia::testing::consume_failure_for_testing(
                            nvidia::testing::failure_point::material_refresh_before_upload))
      throw std::runtime_error("injected NVIDIA material refresh pre-upload failure");
  }
  catch (const std::exception &error) { local_error = error.what(); }
  catch (...) { local_error = "unknown NVIDIA initialization preflight failure"; }
  if (or_to_all(!local_error.empty())) {
    if (local_error.empty())
      local_error = "another rank failed NVIDIA initialization preflight";
    throw std::runtime_error(local_error);
  }

  state.initialized_ = false;
  bool enqueued = false;
  try {
    nvidia::device_scope device_scope(state.device_);
    const bool native_material =
        state.prepared_material_route_ == MaterialRecipeDisposition::device_native ||
        state.prepared_material_route_ == MaterialRecipeDisposition::hybrid_interface;
    const size_t compact_input_bytes = state.initialization_compact_input_bytes_;
    /* The classification header below guarantees at least one launch. Mark
       the live transaction irreversible before the first CUDA API call so an
       API error cannot be misreported as a retryable pre-enqueue failure. */
    enqueued = true;
    if (native_material && state.material_classification_status_.size())
      nvidia::fill_byte_async(state.material_classification_status_, 0, 0,
                              state.material_classification_status_.size(),
                              *state.transfer_);
    else if (state.material_classification_status_.size())
      nvidia::copy_host_to_device_async(
          state.material_classification_status_, 0,
          static_cast<const unsigned char *>(state.staging_.data()) +
              state.prepared_status_staging_offset_,
          state.material_classification_status_.size(), *state.transfer_,
          nvidia::host_to_device_copy_kind::general);

    for (const NvidiaInitializationUpload &upload : state.initialization_uploads_) {
      const nvidia::host_to_device_copy_kind copy_kind =
          !upload.material
              ? nvidia::host_to_device_copy_kind::general
              : state.prepared_material_route_ == MaterialRecipeDisposition::host_reference
                    ? nvidia::host_to_device_copy_kind::material_dense_output
                    : nvidia::host_to_device_copy_kind::material_tiled_output;
      for (const NvidiaInitializationUpload::Range &range : upload.ranges)
        state.arenas_->copy_from_host_async(
            upload.allocation, range.offset,
            static_cast<const unsigned char *>(state.staging_.data()) +
                upload.staging_offset + range.offset,
            range.bytes, *state.transfer_, copy_kind);
      if (upload.material) {
        if (copy_kind == nvidia::host_to_device_copy_kind::material_dense_output) {
          state.material_initialization_statistics_.dense_output_host_to_device_calls +=
              upload.ranges.size();
          for (const NvidiaInitializationUpload::Range &range : upload.ranges)
            state.material_initialization_statistics_.dense_output_host_to_device_bytes =
                material_checked_add(
                    state.material_initialization_statistics_.dense_output_host_to_device_bytes,
                    range.bytes, "accounting dense material output uploads");
        }
        else {
          state.material_initialization_statistics_.tiled_output_host_to_device_calls +=
              upload.ranges.size();
          for (const NvidiaInitializationUpload::Range &range : upload.ranges)
            state.material_initialization_statistics_.tiled_output_host_to_device_bytes =
                material_checked_add(
                    state.material_initialization_statistics_.tiled_output_host_to_device_bytes,
                    range.bytes, "accounting tiled material output uploads");
        }
      }
    }
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_refresh_after_upload))
      throw std::runtime_error("injected NVIDIA material refresh post-upload failure");
    if (native_material && compact_input_bytes) {
      nvidia::copy_host_to_device_async(
          state.material_ir_inputs_, 0,
          static_cast<const unsigned char *>(state.staging_.data()) +
              state.prepared_compact_staging_offset_,
          compact_input_bytes, *state.transfer_,
          nvidia::host_to_device_copy_kind::material_compact_input);
      state.material_initialization_statistics_.compact_input_host_to_device_calls = 1;
      state.material_initialization_statistics_.compact_input_host_to_device_bytes =
          compact_input_bytes;
    }
    if (native_material) {
      for (uint32_t phase = 0; phase <= 5; ++phase) {
        for (const nvidia::material_fill_launch &launch : state.material_fill_launches_)
          if (launch.phase == phase) {
            nvidia::launch_material_fill(launch, *state.transfer_);
            ++state.material_initialization_statistics_.constant_fill_kernel_launches;
            ++state.material_initialization_statistics_.pointwise_kernel_launches;
          }
        for (const nvidia::geometry_bulk_launch &launch :
             state.material_geometry_bulk_launches_)
          if (geometry_property_phase(launch.common.property) == int(phase)) {
            nvidia::launch_material_geometry_bulk(launch, *state.transfer_);
            ++state.material_initialization_statistics_.geometry_bulk_kernel_launches;
            ++state.material_initialization_statistics_.pointwise_kernel_launches;
            state.material_initialization_statistics_.geometry_bulk_points =
                material_checked_add(state.material_initialization_statistics_.geometry_bulk_points,
                                     launch.count, "accounting geometry bulk points");
          }
        if (phase == 1) {
          for (const nvidia::geometry_analytic_launch &launch :
               state.material_geometry_analytic_launches_) {
            nvidia::launch_material_geometry_analytic(launch, *state.transfer_);
            ++state.material_initialization_statistics_.geometry_analytic_kernel_launches;
            ++state.material_initialization_statistics_.pointwise_kernel_launches;
            state.material_initialization_statistics_.geometry_analytic_points =
                material_checked_add(
                    state.material_initialization_statistics_.geometry_analytic_points,
                    launch.count, "accounting geometry analytic points");
          }
          for (const nvidia::geometry_patch_launch &launch :
               state.material_geometry_patch_launches_) {
            nvidia::launch_material_geometry_patch(launch, *state.transfer_);
            ++state.material_initialization_statistics_.geometry_patch_kernel_launches;
            ++state.material_initialization_statistics_.pointwise_kernel_launches;
            state.material_initialization_statistics_.geometry_patch_points =
                material_checked_add(state.material_initialization_statistics_.geometry_patch_points,
                                     launch.count, "accounting geometry patch points");
          }
        }
        if (phase == 3)
          for (const nvidia::material_conductivity_launch &launch :
               state.material_conductivity_launches_) {
            nvidia::launch_material_conductivity(launch, *state.transfer_);
            ++state.material_initialization_statistics_.conductivity_kernel_launches;
            ++state.material_initialization_statistics_.pointwise_kernel_launches;
            state.material_initialization_statistics_.absorber_points_evaluated =
                material_checked_add(
                    state.material_initialization_statistics_.absorber_points_evaluated,
                    launch.loop_count, "accounting absorber points");
          }
        for (const nvidia::material_table_launch &launch : state.material_table_launches_) {
          const uint32_t table_phase =
              launch.operation == nvidia::material_table_operation::file_chi1inv ||
                      launch.operation == nvidia::material_table_operation::grid_chi1inv
                  ? 1
                  : launch.operation == nvidia::material_table_operation::grid_conductivity
                        ? 3
                        : 5;
          if (table_phase != phase) continue;
          nvidia::launch_material_table(launch, *state.transfer_);
          ++state.material_initialization_statistics_.pointwise_kernel_launches;
          if (launch.table_kind == nvidia::material_table_kind::file_scalar_epsilon) {
            ++state.material_initialization_statistics_.file_table_kernel_launches;
            state.material_initialization_statistics_.file_points_evaluated =
                material_checked_add(state.material_initialization_statistics_.file_points_evaluated,
                                     launch.loop_count, "accounting FILE table points");
          }
          else {
            ++state.material_initialization_statistics_.grid_table_kernel_launches;
            state.material_initialization_statistics_.grid_points_evaluated =
                material_checked_add(state.material_initialization_statistics_.grid_points_evaluated,
                                     launch.loop_count, "accounting MaterialGrid table points");
          }
        }
      }
      for (const nvidia::material_pml_launch &launch : state.material_pml_launches_) {
        nvidia::launch_material_pml(launch, *state.transfer_);
        ++state.material_initialization_statistics_.pml_kernel_launches;
      }
    }
    nvidia::material_classification_header header = {};
    header.version = 1;
    header.row_count = uint32_t(state.material_classification_rows_.size());
    header.recipe_signature = state.prepared_recipe_signature_;
    header.state_token = state.state_token_;
    header.storage_fingerprint = state.fingerprint_;
    nvidia::material_classification_header *const device_header =
        static_cast<nvidia::material_classification_header *>(
            state.material_classification_result_.opaque_handle());
    nvidia::launch_material_classification_header(device_header, header, *state.transfer_);
    nvidia::material_classification_row_result *const device_rows =
        reinterpret_cast<nvidia::material_classification_row_result *>(device_header + 1);
    const unsigned char *const status_base = static_cast<const unsigned char *>(
        state.material_classification_status_.opaque_handle());
    for (size_t i = 0; i < state.material_classification_rows_.size(); ++i) {
      const NvidiaMaterialClassificationRow &row = state.material_classification_rows_[i];
      nvidia::material_classification_row_result value = {};
      value.chunk = row.key.chunk;
      value.kind = row.key.kind;
      value.component = row.key.component_;
      value.cmp = row.key.cmp;
      value.aux = row.key.aux;
      nvidia::launch_material_classification_row(status_base + row.status_offset,
                                                 row.logical_elements, device_rows + i, value,
                                                 *state.transfer_);
    }
    state.material_fallback_statistics.classification_launches = material_checked_add(
        state.material_fallback_statistics.classification_launches,
        state.material_classification_rows_.size() + 1,
        "accounting material classification launches");
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_classification_d2h))
      throw std::runtime_error("injected NVIDIA material classification result copy failure");
    nvidia::copy_device_to_host_async(
        static_cast<unsigned char *>(state.staging_.data()) +
            state.prepared_result_staging_offset_,
        state.material_classification_result_, 0,
        state.material_classification_result_bytes_, *state.transfer_);
    ++state.material_fallback_statistics.classification_device_to_host_calls;
    state.material_fallback_statistics.classification_device_to_host_bytes =
        material_checked_add(
            state.material_fallback_statistics.classification_device_to_host_bytes,
            state.material_classification_result_bytes_,
            "accounting material classification transfer");
    {
      if (nvidia::testing::consume_failure_for_testing(
              nvidia::testing::failure_point::material_initialization_sync))
        throw std::runtime_error("injected NVIDIA material initialization sync failure");
      state.transfer_->synchronize();
      state.material_initialization_statistics_.synchronizations = 1;
    }
    const unsigned char *const result =
        static_cast<const unsigned char *>(state.staging_.data()) +
        state.prepared_result_staging_offset_;
    nvidia::material_classification_header actual_header;
    memcpy(&actual_header, result, sizeof(actual_header));
    if (actual_header.version != header.version ||
        actual_header.row_count != header.row_count ||
        actual_header.recipe_signature != header.recipe_signature ||
        actual_header.state_token != header.state_token ||
        actual_header.storage_fingerprint != header.storage_fingerprint)
      throw std::runtime_error("NVIDIA material classification header is stale");
    state.material_classification_facts_ = MaterialClassificationFacts();
    std::set<StorageKey, MaterialStorageKeyLess> seen;
    for (size_t i = 0; i < state.material_classification_rows_.size(); ++i) {
      const NvidiaMaterialClassificationRow &expected = state.material_classification_rows_[i];
      nvidia::material_classification_row_result actual;
      memcpy(&actual,
             result + sizeof(actual_header) +
                 i * sizeof(nvidia::material_classification_row_result),
             sizeof(actual));
      if (i == 0 && nvidia::testing::consume_failure_for_testing(
                        nvidia::testing::failure_point::material_classification_result_mutation))
        actual.nontrivial = 2;
      const StorageKey key{actual.chunk, actual.kind, actual.component, actual.cmp, actual.aux};
      if (!(key == expected.key) || actual.missing > 1 || actual.nontrivial > 1 ||
          actual.missing ||
          !seen.insert(key).second)
        throw std::runtime_error(
            "NVIDIA material classification producer coverage is incomplete");
      const ArraySpec &spec = state.plan_.arrays[expected.allocation];
      const uint8_t row_state = expected.allocation < state.prepared_authoritative_arrays_ ||
                                        (!spec.classification_provisional &&
                                         !spec.classification_elided) ||
                                        actual.nontrivial
                                    ? uint8_t(MaterialClassification::retained)
                                    : uint8_t(MaterialClassification::elided_row);
      state.material_classification_facts_.rows.push_back(
          MaterialRowClassificationFact{key, row_state});
    }
    state.material_initialization_statistics_.device_native = native_material;
    state.material_initialization_statistics_.valid = true;
  }
  catch (const std::exception &error) {
    local_error = error.what();
    try { state.transfer_->synchronize(); }
    catch (...) {}
  }
  catch (...) {
    local_error = "unknown NVIDIA initialization failure";
    try { state.transfer_->synchronize(); }
    catch (...) {}
  }
  if (or_to_all(!local_error.empty())) {
    state.transfer_failed_ = true;
    if (live_refresh && enqueued) poison();
    if (local_error.empty()) local_error = "another rank failed to initialize NVIDIA storage";
    throw std::runtime_error(local_error);
  }
  state.initialized_ = true;
  state.material_refresh_pending_ = live_refresh;
  state.device_authoritative_ = false;
}

bool NvidiaBackend::preview_prepared_material_classification(
    const StoragePlan &plan, BackendState &raw_state, MaterialClassification &result) {
  NvidiaBackendState &state = checked_state(raw_state);
  if (!state.initialization_prepared_ || state.transfer_failed_)
    throw std::logic_error("NVIDIA material preview requires prepared usable staging");
  if (!f_.initialization_plan || f_.initialization_plan->materials.size() != 1)
    throw std::logic_error("NVIDIA material preview has no installed recipe authority");
  MaterialClassificationFacts facts;
  const unsigned char *const status =
      static_cast<const unsigned char *>(state.staging_.data()) +
      state.prepared_status_staging_offset_;
  for (const NvidiaMaterialClassificationRow &row : state.material_classification_rows_) {
    bool missing = false;
    bool nontrivial = false;
    for (size_t i = 0; i < row.logical_elements; ++i) {
      missing = missing || !(status[row.status_offset + i] & 1u);
      nontrivial = nontrivial || (status[row.status_offset + i] & 2u);
    }
    if (missing)
      throw std::invalid_argument("NVIDIA prepared regional row has incomplete coverage");
    const ArraySpec &spec = state.plan_.arrays[row.allocation];
    const uint8_t row_state = row.allocation < state.prepared_authoritative_arrays_ ||
                                      (!spec.classification_provisional &&
                                       !spec.classification_elided) ||
                                      nontrivial
                                  ? uint8_t(MaterialClassification::retained)
                                  : uint8_t(MaterialClassification::elided_row);
    facts.rows.push_back(MaterialRowClassificationFact{row.key, row_state});
  }
  const MaterialClassificationFacts installed = state.material_classification_facts_;
  state.material_classification_facts_ = facts;
  try { result = classify_state(plan, state); }
  catch (...) {
    state.material_classification_facts_ = installed;
    throw;
  }
  state.material_classification_facts_ = installed;
  return true;
}

namespace {

bool same_regional_ivec(const ivec &a, const ivec &b) {
  if (a.dim != b.dim) return false;
  LOOP_OVER_DIRECTIONS(a.dim, d)
    if (a.in_direction(d) != b.in_direction(d)) return false;
  return true;
}

bool same_regional_region(const InitRegion &a, const InitRegion &b) {
  return a.chunk == b.chunk && a.whole == b.whole &&
         (a.whole || (same_regional_ivec(a.begin, b.begin) &&
                      same_regional_ivec(a.end, b.end)));
}

bool same_regional_operation(const InitOperation &a, const InitOperation &b) {
  if (a.kind != b.kind || a.destination.id != b.destination.id ||
      a.destination.offset != b.destination.offset ||
      a.destination.elements != b.destination.elements ||
      a.descriptor_index != b.descriptor_index ||
      !same_regional_region(a.region, b.region) ||
      a.point_spans.size() != b.point_spans.size())
    return false;
  for (size_t i = 0; i < a.point_spans.size(); ++i)
    if (a.point_spans[i].first != b.point_spans[i].first ||
        a.point_spans[i].count != b.point_spans[i].count)
      return false;
  return true;
}

bool same_pml_recipe(const PmlRecipe &a, const PmlRecipe &b) {
  return a.chunk == b.chunk && a.direction_ == b.direction_ &&
         a.sigma == b.sigma && a.kappa == b.kappa &&
         a.sigma_inv == b.sigma_inv;
}

bool same_regional_authority(const InitializationPlan &actual,
                             const InitializationPlan &expected) {
  if (!actual.regional || !expected.regional ||
      actual.material_values_generation != expected.material_values_generation ||
      actual.material_region_generation != expected.material_region_generation ||
      actual.requires_remote_transport != expected.requires_remote_transport ||
      actual.regional_supported != expected.regional_supported ||
      actual.regional_reason != expected.regional_reason ||
      actual.regional_unsupported_reason != expected.regional_unsupported_reason ||
      !same_regional_region(actual.requested_region, expected.requested_region) ||
      actual.operations.size() != expected.operations.size() ||
      actual.materials.size() != expected.materials.size() ||
      actual.pml.size() != expected.pml.size() ||
      actual.host_callbacks.size() != expected.host_callbacks.size() ||
      actual.material_destinations != expected.material_destinations ||
      actual.material_bulk_spans != expected.material_bulk_spans ||
      actual.material_analytic_interfaces != expected.material_analytic_interfaces ||
      actual.material_hybrid_patches != expected.material_hybrid_patches ||
      actual.material_callback_tiles != expected.material_callback_tiles)
    return false;
  for (size_t i = 0; i < actual.operations.size(); ++i)
    if (!same_regional_operation(actual.operations[i], expected.operations[i])) return false;
  for (size_t i = 0; i < actual.materials.size(); ++i)
    if (actual.materials[i].signature() != expected.materials[i].signature()) return false;
  for (size_t i = 0; i < actual.pml.size(); ++i)
    if (!same_pml_recipe(actual.pml[i], expected.pml[i])) return false;
  for (size_t i = 0; i < actual.host_callbacks.size(); ++i)
    if (actual.host_callbacks[i].id != expected.host_callbacks[i].id ||
        actual.host_callbacks[i].description != expected.host_callbacks[i].description)
      return false;
  return true;
}

bool same_array_spec(const ArraySpec &a, const ArraySpec &b) {
  return a.id == b.id && a.role == b.role && a.element_type == b.element_type &&
         a.storage == b.storage && a.elements == b.elements &&
         a.alignment == b.alignment && a.alias_of == b.alias_of &&
         a.classification_provisional == b.classification_provisional &&
         a.classification_elided == b.classification_elided;
}

bool pml_storage_key(const StorageKey &key) {
  return key.kind == int(array_kind::pml_sig) ||
         key.kind == int(array_kind::pml_kap) ||
         key.kind == int(array_kind::pml_siginv);
}

struct RegionalExpectedUpload {
  size_t bytes;
  size_t staging_offset;
  size_t logical_offset;
  std::vector<NvidiaInitializationUpload::Range> ranges;
};

bool append_regional_expected_upload(
    const InitializationPlan &canonical, const InitOperation &operation,
    const StoragePlan &resolved, const MaterialIR &ir,
    std::map<uint32_t, RegionalExpectedUpload> &expected) {
  if (!is_valid(operation.destination.id) ||
      operation.destination.id.value >= resolved.arrays.size())
    return false;
  const uint32_t id = operation.destination.id.value;
  const ArraySpec &spec = resolved.arrays[id];
  if (spec.id.value != id || spec.role != array_role::material ||
      is_valid(spec.alias_of) || spec.classification_provisional ||
      operation.destination.offset != 0 ||
      operation.destination.elements != spec.elements)
    return false;
  if (spec.classification_elided) return true;
  if (expected.count(id)) return false;
  const size_t bytes = storage_bytes(spec);
  if (!spec.elements || bytes % spec.elements) return false;
  std::vector<NvidiaInitializationUpload::Range> ranges;
  if (operation.kind == InitKind::pml_profile) {
    if (!operation.point_spans.empty() || !pml_storage_key(resolved.keys[id])) return false;
    ranges.push_back(NvidiaInitializationUpload::Range{0, bytes});
  }
  else if (operation.kind == InitKind::material_geometry) {
    if (operation.descriptor_index >= ir.destinations.size() ||
        !(ir.destinations[operation.descriptor_index].key == resolved.keys[id]) ||
        operation.point_spans.empty())
      return false;
    const MaterialIRDestination &destination =
        ir.destinations[operation.descriptor_index];
    const size_t element_bytes = bytes / spec.elements;
    std::vector<size_t> physical;
    uint64_t prior_end = 0;
    bool have_prior = false;
    for (const InitPointSpan &span : operation.point_spans) {
      if (!span.count || span.first >= destination.point_count ||
          span.count > destination.point_count - span.first ||
          (have_prior && span.first < prior_end))
        return false;
      prior_end = span.first + span.count;
      have_prior = true;
      for (uint64_t logical = span.first; logical < prior_end; ++logical) {
        const size_t index =
            material_ir_destination_storage_index(ir, destination, logical);
        if (index >= spec.elements) return false;
        physical.push_back(index);
      }
    }
    std::sort(physical.begin(), physical.end());
    if (std::adjacent_find(physical.begin(), physical.end()) != physical.end()) return false;
    for (size_t index : physical) {
      const size_t offset = material_checked_product(
          index, element_bytes, "validating regional material upload");
      if (!ranges.empty() && ranges.back().offset + ranges.back().bytes == offset)
        ranges.back().bytes = material_checked_add(
            ranges.back().bytes, element_bytes,
            "coalescing validated regional material upload");
      else
        ranges.push_back(NvidiaInitializationUpload::Range{offset, element_bytes});
    }
  }
  else return false;
  if (ranges.empty()) return false;
  expected.insert(std::make_pair(
      id, RegionalExpectedUpload{bytes, 0, size_t(-1), ranges}));
  (void)canonical;
  return true;
}

} // namespace

bool NvidiaBackend::prepared_material_classification_preserves_storage(
    const InitializationPlan &candidate, const InitializationPlan &execution,
    const StoragePlan &authoritative, const StoragePlan &candidate_storage,
    const MaterialClassification &classification, BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  if (candidate.materials.size() != 1 || execution.materials.size() != 1 ||
      authoritative.arrays.size() > candidate_storage.arrays.size() ||
      candidate_storage.arrays.size() != state.plan_.arrays.size() ||
      candidate_storage.keys.size() != candidate_storage.arrays.size() ||
      authoritative.keys.size() != authoritative.arrays.size() ||
      classification.provisional_row_state.size() != candidate_storage.arrays.size())
    return false;

  StoragePlan expected_storage = candidate_storage;
  try {
    apply_precision_policy(expected_storage, policy_for(options_.precision));
    resolve_material_storage(candidate.materials[0], classification, authoritative,
                             expected_storage, policy_for(options_.precision));
  }
  catch (...) { return false; }
  if (expected_storage.arrays.size() != state.plan_.arrays.size() ||
      expected_storage.keys.size() != state.plan_.keys.size())
    return false;
  std::set<uint32_t> installed_ids;
  for (size_t i = 0; i < state.plan_.arrays.size(); ++i) {
    const ArraySpec &installed = state.plan_.arrays[i];
    if (installed.id.value != i || !installed_ids.insert(installed.id.value).second ||
        !(expected_storage.keys[i] == state.plan_.keys[i]) ||
        !same_array_spec(expected_storage.arrays[i], installed) ||
        (is_valid(installed.alias_of) &&
         (installed.alias_of.value >= state.plan_.arrays.size() ||
          state.plan_.arrays[installed.alias_of.value].classification_elided)))
      return false;
  }

  if (execution.regional) {
    InitializationPlan canonical;
    try { canonical = candidate.restrict_to(execution.requested_region); }
    catch (...) { return false; }
    if (!canonical.regional_supported ||
        canonical.regional_reason != RegionalSupportReason::supported ||
        !same_regional_authority(execution, canonical) ||
        candidate.materials[0].disposition() !=
            MaterialRecipeDisposition::host_reference ||
        !f_.initialization_plan)
      return false;
    std::string preservation_error;
    if (!regional_replacement_preserves_unselected(
            *f_.initialization_plan, candidate, canonical, &preservation_error))
      return false;
    const MaterialIR *ir = candidate.materials[0].ir().get();
    if (!ir) return false;
    std::map<uint32_t, RegionalExpectedUpload> expected_uploads;
    try {
      for (const InitOperation &operation : canonical.operations)
        if (!append_regional_expected_upload(canonical, operation, expected_storage,
                                             *ir, expected_uploads))
          return false;
    }
    catch (...) { return false; }
    size_t staging_cursor = 0;
    try {
      for (auto &entry : expected_uploads) {
        entry.second.staging_offset = staging_cursor;
        staging_cursor = material_checked_add(
            staging_cursor, entry.second.bytes,
            "validating regional upload staging layout");
      }
      if (staging_cursor != state.prepared_compact_staging_offset_) return false;
      size_t logical_cursor = staging_cursor;
      for (auto &entry : expected_uploads) {
        entry.second.logical_offset = logical_cursor;
        const ArraySpec &spec = expected_storage.arrays[entry.first];
        logical_cursor = material_checked_add(
            logical_cursor,
            material_checked_product(spec.elements, sizeof(realnum),
                                     "validating regional logical staging layout"),
            "validating regional logical staging layout");
      }
      if (logical_cursor != state.prepared_status_staging_offset_ ||
          state.prepared_status_staging_offset_ > state.staging_.size())
        return false;
    }
    catch (...) { return false; }
    if (expected_uploads.size() != state.initialization_uploads_.size()) return false;
    std::set<uint32_t> actual_ids;
    for (const NvidiaInitializationUpload &upload : state.initialization_uploads_) {
      if (upload.allocation >= state.plan_.arrays.size() || !upload.material ||
          upload.bytes != storage_bytes(state.plan_.arrays[upload.allocation]) ||
          !actual_ids.insert(upload.allocation).second)
        return false;
      const auto expected = expected_uploads.find(upload.allocation);
      if (expected == expected_uploads.end() ||
          upload.bytes != expected->second.bytes ||
          upload.staging_offset != expected->second.staging_offset ||
          upload.logical_offset != expected->second.logical_offset ||
          upload.ranges.size() != expected->second.ranges.size())
        return false;
      size_t prior_end = 0;
      for (size_t i = 0; i < upload.ranges.size(); ++i) {
        const NvidiaInitializationUpload::Range &actual = upload.ranges[i];
        const NvidiaInitializationUpload::Range &wanted = expected->second.ranges[i];
        if (!actual.bytes || actual.offset < prior_end ||
            actual.offset > upload.bytes || actual.bytes > upload.bytes - actual.offset ||
            actual.offset != wanted.offset || actual.bytes != wanted.bytes)
          return false;
        prior_end = actual.offset + actual.bytes;
      }
    }
    return actual_ids.size() == expected_uploads.size();
  }

  std::set<uint32_t> upload_ids;
  for (const NvidiaInitializationUpload &upload : state.initialization_uploads_) {
    if (upload.allocation >= state.plan_.arrays.size() ||
        !upload_ids.insert(upload.allocation).second)
      return false;
    size_t prior_end = 0;
    for (const NvidiaInitializationUpload::Range &range : upload.ranges) {
      if (!range.bytes || range.offset < prior_end || range.offset > upload.bytes ||
          range.bytes > upload.bytes - range.offset)
        return false;
      prior_end = range.offset + range.bytes;
    }
  }
  return true;
}

void NvidiaBackend::mutate_prepared_initialization_upload_for_testing(
    BackendState &raw_state, int mode) {
  NvidiaBackendState &state = checked_state(raw_state);
  if (!state.initialization_prepared_ || state.initialization_uploads_.empty())
    throw std::logic_error("NVIDIA test upload mutation requires a prepared upload");
  NvidiaInitializationUpload &upload = state.initialization_uploads_[0];
  switch (mode) {
    case 0:
      state.initialization_uploads_.pop_back();
      break;
    case 1:
      {
        const NvidiaInitializationUpload duplicate = upload;
        state.initialization_uploads_.push_back(duplicate);
      }
      break;
    case 2:
      upload.allocation = upload.allocation + 1 < state.plan_.arrays.size()
                              ? upload.allocation + 1
                              : uint32_t(state.plan_.arrays.size());
      break;
    case 3:
      if (upload.ranges.empty())
        throw std::logic_error("NVIDIA test upload mutation found no range");
      ++upload.ranges[0].offset;
      break;
    case 4:
      ++upload.staging_offset;
      break;
    default:
      throw std::invalid_argument("unknown NVIDIA prepared-upload test mutation");
  }
}


MaterialClassification NvidiaBackend::classify_state(const StoragePlan &plan,
                                                     BackendState &raw_state) {
  std::string local_error;
  NvidiaBackendState *state = NULL;
  MaterialClassificationFacts facts;
  try {
    state = &checked_state(raw_state);
    if (!state->initialized_ || state->transfer_failed_)
      local_error = "NVIDIA classification requires an initialized usable state";
    else if (plan.arrays.size() > state->plan_.arrays.size())
      local_error = "NVIDIA classification plan exceeds resident storage";
    else if (!f_.initialization_plan || f_.initialization_plan->materials.size() != 1)
      local_error = "NVIDIA classification has no frozen material recipe";
    else {
      facts = state->material_classification_facts_;
      std::map<StorageKey, uint8_t, MaterialStorageKeyLess> row_states;
      for (const MaterialRowClassificationFact &row : facts.rows)
        if (!row_states.insert(std::make_pair(row.key, row.state)).second)
          throw std::invalid_argument("NVIDIA classification contains duplicate rows");
      const auto retain_group = [&](const std::vector<StorageKey> &keys) {
        bool keep = false;
        for (const StorageKey &key : keys) {
          const auto found = row_states.find(key);
          keep = keep ||
                 (found != row_states.end() &&
                  found->second == MaterialClassification::retained);
        }
        if (keep)
          for (const StorageKey &key : keys) {
            const auto found = row_states.find(key);
            if (found != row_states.end()) found->second = MaterialClassification::retained;
          }
      };
      for (const auto &entry : row_states) {
        const StorageKey key = entry.first;
        const array_kind kind = static_cast<array_kind>(key.kind);
        if (kind == array_kind::conductivity || kind == array_kind::condinv)
          retain_group({StorageKey{key.chunk, int(array_kind::conductivity), key.component_,
                                   key.cmp, key.aux},
                        StorageKey{key.chunk, int(array_kind::condinv), key.component_,
                                   key.cmp, key.aux}});
        else if (kind == array_kind::chi2 || kind == array_kind::chi3) {
          retain_group({StorageKey{key.chunk, int(array_kind::chi2), key.component_, -1, 0},
                        StorageKey{key.chunk, int(array_kind::chi3), key.component_, -1, 0}});
          const StorageKey chi2{key.chunk, int(array_kind::chi2), key.component_, -1, 0};
          const StorageKey chi3{key.chunk, int(array_kind::chi3), key.component_, -1, 0};
          const auto chi2_state = row_states.find(chi2);
          const auto chi3_state = row_states.find(chi3);
          const bool nonlinear =
              (chi2_state != row_states.end() &&
               chi2_state->second == MaterialClassification::retained) ||
              (chi3_state != row_states.end() &&
               chi3_state->second == MaterialClassification::retained);
          if (nonlinear) {
            const StorageKey diagonal{
                key.chunk, int(array_kind::chi1inv), key.component_, -1,
                uint64_t(component_direction(component(key.component_)))};
            const auto found = row_states.find(diagonal);
            if (found != row_states.end()) found->second = MaterialClassification::retained;
          }
        }
        else if (kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
                 kind == array_kind::pml_siginv)
          retain_group({StorageKey{key.chunk, int(array_kind::pml_sig), -1, -1, key.aux},
                        StorageKey{key.chunk, int(array_kind::pml_kap), -1, -1, key.aux},
                        StorageKey{key.chunk, int(array_kind::pml_siginv), -1, -1,
                                   key.aux}});
        const auto current = row_states.find(key);
        if (kind == array_kind::chi1inv && key.component_ >= 0 &&
            direction(key.aux) != component_direction(component(key.component_))) {
          const StorageKey diagonal{key.chunk, int(array_kind::chi1inv), key.component_, -1,
                                    uint64_t(component_direction(component(key.component_)))};
          const auto found = row_states.find(diagonal);
          if (current != row_states.end() &&
              current->second == MaterialClassification::retained &&
              found != row_states.end())
            found->second = MaterialClassification::retained;
        }
        if (kind == array_kind::sigma && current != row_states.end() &&
            current->second == MaterialClassification::retained) {
          const StorageKey diagonal{
              key.chunk, int(array_kind::sigma), key.component_,
              int(component_direction(component(key.component_))), key.aux};
          const auto found = row_states.find(diagonal);
          if (found != row_states.end()) found->second = MaterialClassification::retained;
        }
      }
      for (MaterialRowClassificationFact &row : facts.rows)
        row.state = row_states[row.key];
      MaterialClassification local;
      local.provisional_row_state.assign(plan.arrays.size(),
                                         MaterialClassification::not_provisional);
      local.anisotropic_eh.assign(size_t(f_.num_chunks) * NUM_FIELD_TYPES, 0);
      for (size_t i = 0; i < plan.arrays.size(); ++i) {
        if (plan.arrays[i].role != array_role::material) continue;
        const auto found = row_states.find(plan.keys[i]);
        if (found == row_states.end())
          throw std::invalid_argument("NVIDIA classification omitted a material row");
        local.provisional_row_state[i] = found->second;
        if (found->second == MaterialClassification::elided_row)
          local.elided.push_back(ArrayId{uint32_t(i)});
        else {
          const array_kind kind = static_cast<array_kind>(plan.keys[i].kind);
          local.has_nonlinearities = local.has_nonlinearities ||
                                     kind == array_kind::chi2 || kind == array_kind::chi3;
          if (f_.gv.dim == D2 && kind == array_kind::chi1inv) {
            const component c = component(plan.keys[i].component_);
            const direction d = direction(plan.keys[i].aux);
            local.aniso2d = local.aniso2d ||
                            ((c == Ex || c == Ey || c == Hx || c == Hy) && d == Z) ||
                            ((c == Ez || c == Hz) && (d == X || d == Y));
          }
        }
      }
      FOR_COMPONENTS(c)
        if (f_.have_component(c)) local.required_components |= component_mask(1) << int(c);
      int minimum = 0;
      for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
        if (!f_.chunks[chunk]->is_mine()) continue;
        for (dft_chunk *dft = f_.chunks[chunk]->dft_chunks; dft; dft = dft->next_in_chunk) {
          const int factor = dft->get_decimation_factor();
          if (factor > 0 && (!minimum || factor < minimum)) minimum = factor;
        }
      }
      local.min_decimation_factor = minimum ? minimum : 1;
      refresh_material_classification_variants(f_, plan, local);
      facts.required_components = local.required_components;
      facts.has_nonlinearities = local.has_nonlinearities;
      facts.aniso2d = local.aniso2d;
      facts.min_decimation_factor = minimum;
      facts.variants.clear();
      for (const MaterialVariantClassificationFact &variant : local.variant_facts)
        if (variant.chunk >= 0 && variant.chunk < f_.num_chunks &&
            f_.chunks[variant.chunk]->is_mine())
          facts.variants.push_back(variant);
    }
  }
  catch (const std::exception &e) {
    local_error = e.what();
  }
  catch (...) {
    local_error = "unknown NVIDIA classification precondition failure";
  }
  if (!local_error.empty() && state && state->material_refresh_pending_) {
    state->transfer_failed_ = true;
    poison();
  }
  backend_reconcile_host_access(local_error, "NVIDIA classification preflight");
  return assemble_material_classification(
      f_, plan, f_.initialization_plan->materials[0], facts);
}

void NvidiaBackend::finalize_storage(const StoragePlan &plan,
                                     const MaterialClassification &classification,
                                     BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  try {
    if (!state.initialized_)
      throw std::logic_error("cannot finalize uninitialized NVIDIA storage");
    if (!f_.initialization_plan || f_.initialization_plan->materials.size() != 1)
      throw std::logic_error("NVIDIA material finalization requires one frozen recipe");
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::material_finalize))
      throw std::runtime_error("injected NVIDIA material finalization failure");
    resolve_material_storage(f_.initialization_plan->materials[0], classification, plan,
                             state.plan_, policy_for(options_.precision));
    if (has_provisional_material_storage(state.plan_))
      throw std::logic_error("NVIDIA material finalization left provisional storage");
    /* phase_in_material is a collective public operation, so this bit is
       identical on active and idle ranks.  It is copied into the executable at
       compile and cleared only after all ranks finish the last phase step. */
    state.material_phase_active = f_.phasein_time > 0;
    state.material_refresh_pending_ = false;
  }
  catch (...) {
    if (state.material_refresh_pending_) {
      state.transfer_failed_ = true;
      poison();
    }
    throw;
  }
}

void NvidiaBackend::refresh_noisy_seed(const RandomSeedSnapshot &candidate,
                                       BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  if (!state.initialized_ || state.transfer_failed_)
    throw std::logic_error("NVIDIA noisy seed refresh requires an initialized usable state");
  if (!candidate.initialized || !candidate.semantic_seed_valid ||
      candidate.algorithm_version != counter_random_algorithm_version)
    throw std::invalid_argument("NVIDIA noisy seed refresh received invalid metadata");
  const int slot = state.noisy_seed_active_slot_ == 0 ? 1 : 0;
  const nvidia::noisy_seed_block block = {candidate.semantic_seed,
                                          candidate.algorithm_version};
  nvidia::device_scope scope(state.device_);
  if (nvidia::testing::consume_failure_for_testing(
          nvidia::testing::failure_point::noisy_seed_copy))
    throw std::runtime_error("injected NVIDIA noisy seed copy failure");
  try {
    nvidia::copy_host_to_device_async(
        state.noisy_seed_slots_, size_t(slot) * sizeof(block), &block, sizeof(block),
        *state.transfer_);
  }
  catch (...) {
    state.noisy_seed_staged_slot_ = -1;
    throw;
  }
  try {
    state.transfer_->synchronize();
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::noisy_seed_sync))
      throw std::runtime_error("injected NVIDIA noisy seed synchronization failure");
  }
  catch (...) {
    state.noisy_seed_staged_slot_ = -1;
    state.transfer_failed_ = true;
    poison();
    throw;
  }
  state.noisy_seed_staged_slot_ = slot;
}

void NvidiaBackend::commit_noisy_seed(BackendState &raw_state) noexcept {
  NvidiaBackendState *state = dynamic_cast<NvidiaBackendState *>(&raw_state);
  if (!state || state->noisy_seed_staged_slot_ < 0) return;
  state->noisy_seed_active_slot_ = state->noisy_seed_staged_slot_;
  state->noisy_seed_staged_slot_ = -1;
}

void NvidiaBackend::discard_noisy_seed(BackendState &raw_state) noexcept {
  NvidiaBackendState *state = dynamic_cast<NvidiaBackendState *>(&raw_state);
  if (state) state->noisy_seed_staged_slot_ = -1;
}

namespace {

uint64_t source_structural_step_signature(const StepPlan &plan,
                                          const SourcePlan *source_plan) {
  StepPlan structural = plan;
  structural.source_signature =
      source_plan ? source_plan_topology_signature(*source_plan)
                  : source_plan_topology_signature(SourcePlan());
  return compute_step_plan_signature(structural);
}

bool same_source_launch_topology(const nvidia::source_batch_launch &a,
                                 const nvidia::source_batch_launch &b) {
  return a.target_real == b.target_real && a.target_imag == b.target_imag &&
         a.conductivity_inverse == b.conductivity_inverse &&
         a.point_offset == b.point_offset && a.point_count == b.point_count &&
         a.scalar_slot == b.scalar_slot && a.dt == b.dt && a.integrated == b.integrated &&
         a.sequential == b.sequential && a.precision == b.precision;
}

bool same_cw_source_launch_topology(const nvidia::cw_source_batch_launch &a,
                                    const nvidia::cw_source_batch_launch &b) {
  return a.target_real == b.target_real && a.target_imag == b.target_imag &&
         a.conductivity_inverse == b.conductivity_inverse &&
         a.point_offset == b.point_offset && a.point_count == b.point_count &&
         a.scalar_slot == b.scalar_slot && a.dt == b.dt &&
         a.sequential == b.sequential && a.precision == b.precision;
}

uint64_t source_structural_cw_signature(const CwPlan &plan,
                                        uint64_t structural_step_signature,
                                        uint64_t source_topology_signature) {
  CwPlan structural = plan;
  structural.step_plan_signature = structural_step_signature;
  structural.source_fingerprint = source_topology_signature;
  return compute_cw_plan_signature(structural);
}

struct NvidiaStagedSourceRefresh {
  std::vector<nvidia::source_point> points;
  uint64_t topology;
  bool advanced;

  NvidiaStagedSourceRefresh() : topology(0), advanced(false) {}
};

bool stage_source_value_refresh(fields &f, const StepPlan &plan,
                                NvidiaExecutable &executable,
                                NvidiaBackendState &state,
                                NvidiaStagedSourceRefresh &staged) {
  if (!f.descriptors || plan.program != executable.program_ ||
      plan.signature != compute_step_plan_signature(plan) ||
      plan.source_signature != source_plan_signature(f.descriptors->sources) ||
      executable.storage_fingerprint_ != state.fingerprint_)
    return false;

  const SourcePlan &source_plan = f.descriptors->sources;
  staged.topology = source_plan_topology_signature(source_plan);
  if (staged.topology != executable.source_topology_signature_ ||
      source_structural_step_signature(plan, &source_plan) !=
          executable.structural_signature_)
    return false;

  for (int i = 0; i < mutation_kind_count; ++i) {
    const uint64_t current = f.mutation_generation[i];
    const uint64_t installed = executable.freshness_[i];
    const MutationKind kind = MutationKind(i);
    if (kind == MutationKind::source_values) {
      if (current < installed) return false;
      staged.advanced = current != installed;
    }
    else if (mutation_may_preserve_executable_structure(kind)) {
      if (current < installed) return false;
    }
    else if (current != installed)
      return false;
  }

  std::vector<nvidia::source_batch_launch> batches;
  std::vector<unsigned int> references(source_plan.sources.size(), 0);
  batches.reserve(executable.source_batches_.size());
  for (const Operation &operation : plan.operations) {
    if (operation.kind != OpKind::apply_sources && operation.kind != OpKind::update_eh)
      continue;
    if (!operation.source_descriptor_count) continue;
    if (size_t(operation.source_descriptor_index) + operation.source_descriptor_count >
        source_plan.sources.size())
      throw std::invalid_argument("NVIDIA source refresh descriptor span is out of range");
    for (size_t i = operation.source_descriptor_index;
         i < size_t(operation.source_descriptor_index) + operation.source_descriptor_count; ++i) {
      ++references[i];
      batches.push_back(compile_source_batch(source_plan.sources[i], source_plan, f, state,
                                             staged.points));
    }
  }
  size_t referenced_sources = 0;
  for (unsigned int count : references) {
    if (count > 1) return false;
    referenced_sources += count;
  }
  if (referenced_sources != batches.size() ||
      batches.size() != executable.source_batches_.size() ||
      staged.points.size() * sizeof(nvidia::source_point) != executable.source_points_.size())
    return false;
  for (size_t i = 0; i < batches.size(); ++i)
    if (!same_source_launch_topology(batches[i], executable.source_batches_[i])) return false;
  return true;
}

void commit_source_value_refresh(fields &f, const StepPlan &plan,
                                 NvidiaExecutable &executable,
                                 const NvidiaStagedSourceRefresh &staged) {
  executable.signature_ = plan.signature;
  executable.source_topology_signature_ = staged.topology;
  for (int i = 0; i < mutation_kind_count; ++i)
    executable.freshness_[i] = f.mutation_generation[i];
}

} // namespace

Executable *NvidiaBackend::compile(const StepPlan &plan, BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  if (!state.initialized_)
    throw std::logic_error("cannot compile against uninitialized NVIDIA storage");
  std::unique_ptr<NvidiaExecutable> executable;
  std::string local_error;
  try {
    if (has_provisional_material_storage(state.plan_))
      throw std::logic_error("NVIDIA compile received unresolved material storage");
    if (plan.signature != compute_step_plan_signature(plan))
      throw std::invalid_argument("NVIDIA PR2 received a stale StepPlan signature");
    if (plan.program != StepProgram::ordinary && plan.program != StepProgram::solve_cw)
      throw std::invalid_argument("NVIDIA received an invalid timestep program");
    if (plan.program == StepProgram::solve_cw) {
      std::string layout_error;
      if (!validate_cw_state_layout(f_, plan.cw_state_layout, &layout_error))
        throw std::invalid_argument(layout_error.empty() ? "invalid solve_cw state layout"
                                                         : layout_error);
    }
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
            d.kind != SusceptibilityKind::noisy_lorentzian &&
            d.kind != SusceptibilityKind::gyrotropic &&
            d.kind != SusceptibilityKind::multilevel &&
            d.kind != SusceptibilityKind::host_custom)
          throw std::invalid_argument(std::string("NVIDIA does not support polarization kind ") +
                                      susceptibility_kind_name(d.kind));
        if (d.kind == SusceptibilityKind::lorentzian && d.lorentzian_states.empty())
          throw std::invalid_argument("NVIDIA Lorentzian descriptor has no resident state");
        if (d.kind == SusceptibilityKind::gyrotropic && d.gyrotropic_states.empty())
          throw std::invalid_argument("NVIDIA gyrotropic descriptor has no resident state");
        if (d.kind == SusceptibilityKind::multilevel && !d.has_internal_state)
          throw std::invalid_argument("NVIDIA multilevel descriptor has no resident state");
        if (f_.gv.dim == Dcyl && d.kind == SusceptibilityKind::gyrotropic)
          throw std::invalid_argument("NVIDIA cylindrical gyrotropic media are not supported");
        if (f_.gv.dim == Dcyl && d.kind == SusceptibilityKind::multilevel)
          throw std::invalid_argument("NVIDIA cylindrical multilevel media are not supported");
        if (plan.program == StepProgram::solve_cw && d.kind == SusceptibilityKind::multilevel)
          throw std::invalid_argument("NVIDIA solve_cw does not support multilevel media");
        if (plan.program == StepProgram::solve_cw && d.kind == SusceptibilityKind::host_custom)
          throw std::invalid_argument("NVIDIA solve_cw does not support host-custom media");
      }
    }
    const bool live_host_custom = has_live_host_custom(f_);
    const bool descriptor_host_custom = has_descriptor_host_custom(f_);
    const bool plan_host_custom = has_plan_host_custom(plan);
    if (live_host_custom != descriptor_host_custom ||
        descriptor_host_custom != plan_host_custom)
      throw std::invalid_argument(
          "NVIDIA host-callback live, descriptor, and plan presence differ");
    if (live_host_custom) {
      if (options_.precision != precision_policy_kind::native)
        throw std::invalid_argument("NVIDIA host-custom fallback requires native precision");
      if (f_.phasein_time > 0)
        throw std::invalid_argument("NVIDIA host-custom fallback does not support material phasing");
      std::string host_error;
      if (plan.program != StepProgram::ordinary ||
          !validate_host_callback_plan(f_, plan, &host_error))
        throw std::invalid_argument(host_error.empty()
                                        ? "NVIDIA host-callback plan is invalid"
                                        : host_error);
    }
    bool has_noisy_updates = false, has_noisy_descriptors = false;
    for (const PolarizationUpdate &update : plan.polarization_updates)
      has_noisy_updates = has_noisy_updates || update.kind == PolarizationUpdateKind::noisy_add;
    typedef std::tuple<int, int, int> NoisyGroupTuple;
    std::vector<NoisyGroupTuple> live_noisy_groups, descriptor_noisy_groups;
    for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
      if (!f_.chunks[chunk] || !f_.chunks[chunk]->is_mine()) continue;
      FOR_FIELD_TYPES(ft) {
        int state_index = 0;
        for (const polarization_state *p = f_.chunks[chunk]->pol[ft]; p;
             p = p->next, ++state_index)
          if (p->s && typeid(*p->s) == typeid(noisy_lorentzian_susceptibility))
            live_noisy_groups.push_back(NoisyGroupTuple(chunk, int(ft), state_index));
      }
    }
    if (f_.descriptors)
      for (const PolarizationDescriptor &descriptor : f_.descriptors->polarizations)
        if (descriptor.kind == SusceptibilityKind::noisy_lorentzian) {
          has_noisy_descriptors = true;
          descriptor_noisy_groups.push_back(
              NoisyGroupTuple(descriptor.chunk, int(descriptor.ft), descriptor.state_index));
        }
    if (!live_noisy_groups.empty() || has_noisy_updates || has_noisy_descriptors) {
      if (live_noisy_groups != descriptor_noisy_groups)
        throw std::invalid_argument(
            "NVIDIA noisy susceptibility descriptors differ from live linked-list order");
      const StepPlan canonical_polarization = build_polarization_validation_plan(f_);
      if (plan.polarization_updates != canonical_polarization.polarization_updates)
        throw std::invalid_argument(
            "NVIDIA noisy polarization rows differ from descriptor authority");
      std::vector<const Operation *> installed_operations, canonical_operations;
      for (const Operation &operation : plan.operations)
        if (operation.kind == OpKind::update_polarization)
          installed_operations.push_back(&operation);
      for (const Operation &operation : canonical_polarization.operations)
        if (operation.kind == OpKind::update_polarization)
          canonical_operations.push_back(&operation);
      if (installed_operations.size() != canonical_operations.size())
        throw std::invalid_argument(
            "NVIDIA noisy polarization operation count is noncanonical");
      for (size_t i = 0; i < installed_operations.size(); ++i)
        if (!same_polarization_operation(*installed_operations[i], *canonical_operations[i]))
          throw std::invalid_argument(
              "NVIDIA noisy polarization operation span or access set is noncanonical");
      typedef std::tuple<int, int, int, int, int> StreamTuple;
      std::set<StreamTuple> tuples;
      std::set<uint64_t> tags;
      const int global_rank = my_global_rank();
      if (global_rank < 0 || uint64_t(global_rank) > UINT32_MAX)
        throw std::invalid_argument("NVIDIA noisy polarization global rank is out of range");
      for (const PolarizationUpdate &update : plan.polarization_updates) {
        if (update.kind != PolarizationUpdateKind::noisy_add) continue;
        const StreamTuple tuple(update.region.chunk, int(update.ft), update.state_index,
                                int(update.region.c), update.region.cmp);
        if (!tuples.insert(tuple).second)
          throw std::invalid_argument("NVIDIA noisy polarization stream tuple is duplicated");
        const uint64_t tag = counter_random_stream_tag(
            update.noise_algorithm_version, uint32_t(global_rank),
            uint32_t(update.region.chunk), uint32_t(update.ft), uint32_t(update.state_index),
            uint32_t(update.region.c), uint32_t(update.region.cmp));
        if (!tags.insert(tag).second)
          throw std::invalid_argument("NVIDIA noisy polarization stream tag collides");
      }
    }
    bool has_multilevel = has_local_exact_multilevel(f_);
    if (f_.descriptors)
      for (const PolarizationDescriptor &descriptor : f_.descriptors->polarizations)
        has_multilevel = has_multilevel || descriptor.kind == SusceptibilityKind::multilevel;
    for (const PolarizationUpdateGroup &group : plan.polarization_groups)
      has_multilevel = has_multilevel || group.kind == PolarizationGroupKind::multilevel;
    has_multilevel = has_multilevel || !plan.multilevel_population_updates.empty() ||
                     !plan.multilevel_population_terms.empty() ||
                     !plan.multilevel_transition_updates.empty() ||
                     !plan.multilevel_coefficients.empty();
    if (has_multilevel) {
      if (!f_.descriptors)
        throw std::invalid_argument("NVIDIA live multilevel state has no descriptors");
      std::vector<const PolarizationDescriptor *> installed_descriptors;
      for (const PolarizationDescriptor &descriptor : f_.descriptors->polarizations)
        if (descriptor.kind == SusceptibilityKind::multilevel)
          installed_descriptors.push_back(&descriptor);
      std::vector<PolarizationDescriptor> rebuilt_descriptors;
      build_polarization_descriptors(f_, rebuilt_descriptors);
      std::vector<const PolarizationDescriptor *> canonical_descriptors;
      for (const PolarizationDescriptor &descriptor : rebuilt_descriptors)
        if (descriptor.kind == SusceptibilityKind::multilevel)
          canonical_descriptors.push_back(&descriptor);
      if (installed_descriptors.size() != canonical_descriptors.size())
        throw std::invalid_argument(
            "NVIDIA multilevel descriptors differ from live exact states");
      for (size_t i = 0; i < installed_descriptors.size(); ++i)
        if (!same_multilevel_descriptor(*installed_descriptors[i],
                                        *canonical_descriptors[i]))
          throw std::invalid_argument(
              "NVIDIA multilevel descriptor order or state differs from live authority");

      DescriptorSet staged_descriptors = *f_.descriptors;
      staged_descriptors.polarizations = rebuilt_descriptors;
      DescriptorSet *const installed_set = f_.descriptors;
      StepPlan canonical_polarization;
      try {
        f_.descriptors = &staged_descriptors;
        canonical_polarization = build_step_plan(f_, StepProgram::ordinary);
        f_.descriptors = installed_set;
      }
      catch (...) {
        f_.descriptors = installed_set;
        throw;
      }
      if (plan.polarization_groups != canonical_polarization.polarization_groups)
        throw std::invalid_argument(
            "NVIDIA multilevel polarization groups differ from descriptor authority");
      if (plan.polarization_updates != canonical_polarization.polarization_updates)
        throw std::invalid_argument(
            "NVIDIA multilevel ordinary polarization rows differ from descriptor authority");
      if (plan.polarization_subtractions != canonical_polarization.polarization_subtractions)
        throw std::invalid_argument(
            "NVIDIA multilevel subtraction rows differ from descriptor authority");
      if (plan.multilevel_population_updates !=
          canonical_polarization.multilevel_population_updates)
        throw std::invalid_argument(
            "NVIDIA multilevel population rows differ from descriptor authority");
      if (plan.multilevel_population_terms != canonical_polarization.multilevel_population_terms)
        throw std::invalid_argument(
            "NVIDIA multilevel population terms differ from descriptor authority");
      if (plan.multilevel_transition_updates !=
          canonical_polarization.multilevel_transition_updates)
        throw std::invalid_argument(
            "NVIDIA multilevel transition rows differ from descriptor authority");
      if (plan.multilevel_coefficients != canonical_polarization.multilevel_coefficients)
        throw std::invalid_argument(
            "NVIDIA multilevel coefficients differ from descriptor authority");
      std::vector<const Operation *> installed_operations, canonical_operations;
      for (const Operation &operation : plan.operations)
        if (operation.kind == OpKind::update_polarization || operation.kind == OpKind::update_eh)
          installed_operations.push_back(&operation);
      for (const Operation &operation : canonical_polarization.operations)
        if (operation.kind == OpKind::update_polarization || operation.kind == OpKind::update_eh)
          canonical_operations.push_back(&operation);
      if (installed_operations.size() != canonical_operations.size())
        throw std::invalid_argument(
            "NVIDIA multilevel polarization operation count is noncanonical");
      for (size_t i = 0; i < installed_operations.size(); ++i)
        if (!same_polarization_operation(*installed_operations[i], *canonical_operations[i]))
          throw std::invalid_argument(
              "NVIDIA multilevel polarization operation span or access set is noncanonical");
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
    std::vector<NvidiaCompiledPolarizationAction> polarization_actions;
    std::vector<nvidia::multilevel_population_launch> multilevel_population_updates;
    std::vector<nvidia::multilevel_population_term_launch> multilevel_population_terms;
    std::vector<nvidia::multilevel_transition_launch> multilevel_transition_updates;
    std::vector<unsigned char> multilevel_coefficients;
    size_t multilevel_scratch_bytes = 0;
    std::vector<nvidia::polarization_subtract_launch> polarization_subtractions;
    std::vector<nvidia::dft_launch> dft_updates;
    std::vector<double> dft_omega;
    std::vector<NvidiaCompiledLegacyFluxUpdate> legacy_flux_updates;
    std::vector<nvidia::legacy_flux_term_launch> legacy_flux_terms;
    size_t legacy_flux_partial_count = 0;
    std::vector<NvidiaCompiledMagneticState> magnetic_states;
    std::vector<NvidiaCompiledMaterialRefresh> material_refreshes;
    std::vector<NvidiaCompiledHostSegment> host_segments;
    size_t magnetic_snapshot_bytes = 0;
    size_t material_staging_bytes = 0;
    size_t host_staging_bytes = 0;
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

    size_t expected_material_refreshes = 0;
    if (f_.phasein_time > 0)
      for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
        if (!f_.chunks[chunk] || !f_.chunks[chunk]->is_mine()) continue;
        const structure_chunk &current = *f_.chunks[chunk]->s;
        for (uint32_t family_value = uint32_t(MaterialRefreshFamily::chi1inv);
             family_value <= uint32_t(MaterialRefreshFamily::condinv); ++family_value) {
          const MaterialRefreshFamily family = MaterialRefreshFamily(family_value);
          const array_kind kind = family == MaterialRefreshFamily::chi1inv
                                      ? array_kind::chi1inv
                                      : family == MaterialRefreshFamily::conductivity
                                            ? array_kind::conductivity
                                            : array_kind::condinv;
          FOR_COMPONENTS(c) for (int d = 0; d < 5; ++d) {
            if (family == MaterialRefreshFamily::condinv &&
                d != int(component_direction(c)))
              continue;
            const realnum *row = family == MaterialRefreshFamily::chi1inv
                                     ? current.chi1inv[c][d]
                                     : family == MaterialRefreshFamily::conductivity
                                           ? current.conductivity[c][d]
                                           : current.condinv[c][d];
            if (!row) continue;
            ++expected_material_refreshes;
            const ArrayId id = f_.array_catalog->find(
                StorageKey{chunk, int(kind), int(c), -1, d});
            if (!is_valid(id) || f_.array_catalog->resolve<realnum>(id) != row)
              throw std::invalid_argument(
                  "NVIDIA material phase current storage is absent from the device catalog");
          }
        }
      }
    const size_t stable_index = plan.program == StepProgram::ordinary ? 0 : 1;
    const StepPlan *stable_plan = f_.step_plans[stable_index];
    const DirtyMask flux_closure =
        DirtyMask(dirty_flux_plan | dirty_regions | dirty_executable);
    const bool stable_provenance_matches =
        stable_plan && f_.descriptors &&
        stable_plan->source_signature == source_plan_signature(f_.descriptors->sources) &&
        dft_plan_signature(stable_plan->dft_updates) ==
            dft_plan_signature(f_.descriptors->dfts);
    const bool flux_only_refresh = is_dirty(f_, dirty_flux_plan) &&
                                   (f_.dirty_mask & ~flux_closure) == dirty_none &&
                                   stable_provenance_matches;
    const StepPlan canonical =
        flux_only_refresh
            ? build_legacy_flux_only_step_plan(f_, plan.program, *stable_plan)
            : build_step_plan(f_, plan.program);
    if (plan.legacy_flux_updates != canonical.legacy_flux_updates ||
        plan.legacy_flux_terms != canonical.legacy_flux_terms)
      throw std::invalid_argument("NVIDIA legacy flux descriptors are non-canonical");
    legacy_flux_updates.reserve(plan.legacy_flux_updates.size());
    legacy_flux_terms.reserve(plan.legacy_flux_terms.size());
    if (plan.legacy_flux_updates.size() > size_t(std::numeric_limits<int>::max()))
      throw std::overflow_error("NVIDIA legacy flux monitor count exceeds MPI range");
    for (size_t i = 0; i < plan.legacy_flux_updates.size(); ++i) {
      const LegacyFluxUpdate &update = plan.legacy_flux_updates[i];
      if (update.flux_ordinal != i ||
          size_t(update.term_index) + update.term_count > plan.legacy_flux_terms.size())
        throw std::invalid_argument("NVIDIA legacy flux update has an invalid ordered span");
      const size_t first = legacy_flux_terms.size();
      for (size_t j = update.term_index; j < size_t(update.term_index) + update.term_count; ++j) {
        if (plan.legacy_flux_terms[j].flux_ordinal != update.flux_ordinal)
          throw std::invalid_argument("NVIDIA legacy flux term belongs to the wrong monitor");
        nvidia::legacy_flux_term_launch launch =
            compile_legacy_flux_term(plan.legacy_flux_terms[j], f_, state);
        legacy_flux_partial_count = std::max(legacy_flux_partial_count, launch.blocks);
        legacy_flux_terms.push_back(launch);
      }
      legacy_flux_updates.push_back(NvidiaCompiledLegacyFluxUpdate{
          update.flux_ordinal, first, legacy_flux_terms.size() - first,
          update.recipe_signature});
    }
    if (plan.material_phase_target_signature != compute_material_phase_target_signature(f_))
      throw std::invalid_argument("NVIDIA material phase target fingerprint is stale");
    if (plan.material_refresh_arrays.size() != expected_material_refreshes ||
        canonical.material_phase_target_signature != plan.material_phase_target_signature ||
        canonical.material_refresh_arrays.size() != plan.material_refresh_arrays.size())
      throw std::invalid_argument("NVIDIA material refresh descriptors are non-canonical");
    for (size_t i = 0; i < plan.material_refresh_arrays.size(); ++i) {
      const MaterialRefreshArray &got = plan.material_refresh_arrays[i];
      const MaterialRefreshArray &want = canonical.material_refresh_arrays[i];
      if (got.chunk != want.chunk || got.c != want.c || got.d != want.d ||
          got.family != want.family || got.current != want.current ||
          got.elements != want.elements)
        throw std::invalid_argument("NVIDIA material refresh descriptors are non-canonical");
      material_refreshes.push_back(
          compile_material_refresh(got, state, f_, material_staging_bytes));
    }
    if (f_.array_catalog)
      for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
        if (!f_.chunks[chunk] || !f_.chunks[chunk]->is_mine() || !f_.chunks[chunk]->new_s)
          continue;
        const structure_chunk &target = *f_.chunks[chunk]->new_s;
        FOR_COMPONENTS(c) for (int d = 0; d < 5; ++d) {
          if ((target.chi1inv[c][d] &&
               f_.array_catalog->contains_address(target.chi1inv[c][d])) ||
              (target.conductivity[c][d] &&
               f_.array_catalog->contains_address(target.conductivity[c][d])) ||
              (target.condinv[c][d] &&
               f_.array_catalog->contains_address(target.condinv[c][d])))
            throw std::invalid_argument("NVIDIA material phase target entered the device catalog");
        }
      }
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

    std::vector<size_t> host_segment_owner(
        plan.operations.size(), std::numeric_limits<size_t>::max());
    for (size_t segment_index = 0; segment_index < plan.host_segments.size(); ++segment_index) {
      const HostSegment &segment = plan.host_segments[segment_index];
      for (size_t i = segment.operation_index;
           i < size_t(segment.operation_index) + segment.operation_count; ++i)
        host_segment_owner[i] = segment_index;
    }

    /* Capability validation is deliberately fail-fast. The first local reason
       is stable and actionable; the collective below only establishes that no
       rank can enter execution after any rank rejected its plan. */
    for (size_t oi = 0; oi < plan.operations.size(); ++oi) {
      const Operation &op = plan.operations[oi];
      NvidiaCompiledOperation compiled = {};
      compiled.kind = op.kind;
      compiled.guard = op.guard;
      compiled.host_segment_index = std::numeric_limits<size_t>::max();
      compiled.source_time_offset = op.source_time_offset;
      /* A host marker owns these complete operations.  Do not lower even the
         built-in subset: a mixed list must retain legacy order, and a
         host-owned PE/PH halo has no device ArrayIds to lower. */
      if (host_segment_owner[oi] != std::numeric_limits<size_t>::max()) {
        operations.push_back(compiled);
        continue;
      }
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
                compile_polarization_subtraction(plan.polarization_subtractions[i], plan, state));
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
            HaloPlan canonical;
            std::string why;
            if (!remap_halo_plan(halo, f_.halos->arrays, f_.halos->host_arrays,
                                 *f_.array_catalog, f_.is_real ? 1 : 2, canonical, why))
              throw std::logic_error(std::string("cannot remap local halo plan: ") + why);
            /* Genuinely opaque polarization halos execute as part of the
               encompassing host-callback segment.  A source plan starts as
               host-owned, however, so only skip it after canonical remapping
               proves that some row has no device ArrayId. */
            if (canonical.storage == HaloStorageDisposition::host_owned) continue;
            const NvidiaCompiledHalo lowered =
                compile_halo(canonical, f_, state, buffer_elements, halo_gathers, halo_scatters);
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
        case OpKind::update_flux_half:
        case OpKind::update_flux: {
          if (plan.program != StepProgram::ordinary) {
            set_reason(local_error, oi, "legacy flux marker in solve_cw plan");
            break;
          }
          if (op.legacy_flux_index != 0 ||
              op.legacy_flux_count != plan.legacy_flux_updates.size() ||
              op.legacy_flux_count != legacy_flux_updates.size()) {
            set_reason(local_error, oi, "legacy flux marker span is incomplete");
            break;
          }
          if (oi >= canonical.operations.size()) {
            set_reason(local_error, oi, "legacy flux marker is absent from canonical plan");
            break;
          }
          const Operation &expected = canonical.operations[oi];
          bool same_accesses = op.accesses.size() == expected.accesses.size();
          for (size_t i = 0; same_accesses && i < op.accesses.size(); ++i)
            same_accesses = op.accesses[i].array.id == expected.accesses[i].array.id &&
                            op.accesses[i].array.offset == expected.accesses[i].array.offset &&
                            op.accesses[i].array.elements == expected.accesses[i].array.elements &&
                            op.accesses[i].mode == expected.accesses[i].mode;
          if (expected.kind != op.kind || expected.legacy_flux_index != op.legacy_flux_index ||
              expected.legacy_flux_count != op.legacy_flux_count || !same_accesses) {
            set_reason(local_error, oi, "legacy flux marker identity or accesses are stale");
            break;
          }
          compiled.legacy_flux_first = op.legacy_flux_index;
          compiled.legacy_flux_count = op.legacy_flux_count;
          break;
        }
        case OpKind::update_polarization: {
          if (size_t(op.descriptor_index) + op.descriptor_count >
                  plan.polarization_updates.size() ||
              size_t(op.polarization_group_index) + op.polarization_group_count >
                  plan.polarization_groups.size()) {
            set_reason(local_error, oi, "polarization update span is out of range");
            break;
          }
          compiled.polarization_first = polarization_actions.size();
          size_t next_ordinary = op.descriptor_index;
          for (size_t gi = op.polarization_group_index;
               gi < size_t(op.polarization_group_index) + op.polarization_group_count; ++gi) {
            const PolarizationUpdateGroup &group = plan.polarization_groups[gi];
            if (group.ft != op.ft)
              throw std::invalid_argument("polarization group has the wrong field type");
            if (group.kind == PolarizationGroupKind::recurrence) {
              const size_t rows = size_t(group.recurrence_count) + group.noise_count;
              if (group.recurrence_index != next_ordinary || group.population_count ||
                  group.transition_count || next_ordinary + rows >
                                                size_t(op.descriptor_index) + op.descriptor_count)
                throw std::invalid_argument(
                    "NVIDIA recurrence polarization group has a noncanonical span");
              for (size_t i = 0; i < rows; ++i) {
                const size_t update_index = polarization_updates.size();
                polarization_updates.push_back(compile_polarization_update(
                    plan.polarization_updates[next_ordinary++], state));
                polarization_actions.push_back(NvidiaCompiledPolarizationAction{
                    NvidiaCompiledPolarizationAction::Kind::ordinary, update_index});
              }
            }
            else if (group.kind == PolarizationGroupKind::multilevel) {
              if (group.recurrence_count || group.noise_count || group.population_count != 1 ||
                  size_t(group.population_index) + group.population_count >
                      plan.multilevel_population_updates.size() ||
                  size_t(group.transition_index) + group.transition_count >
                      plan.multilevel_transition_updates.size())
                throw std::invalid_argument(
                    "NVIDIA multilevel polarization group has a noncanonical span");
              const MultilevelPopulationUpdate &population =
                  plan.multilevel_population_updates[group.population_index];
              if (population.region.chunk != group.chunk || population.ft != group.ft ||
                  population.state_index != group.state_index ||
                  population.term_count != group.transition_count)
                throw std::invalid_argument(
                    "NVIDIA multilevel group identity differs from its action span");
              const size_t population_index = multilevel_population_updates.size();
              multilevel_population_updates.push_back(compile_multilevel_population(
                  population, group.transition_index, plan, state, multilevel_population_terms,
                  multilevel_coefficients, multilevel_scratch_bytes));
              polarization_actions.push_back(NvidiaCompiledPolarizationAction{
                  NvidiaCompiledPolarizationAction::Kind::multilevel_population,
                  population_index});
              for (size_t i = group.transition_index;
                   i < size_t(group.transition_index) + group.transition_count; ++i) {
                if (!population.active_component_cmps)
                  throw std::invalid_argument(
                      "NVIDIA multilevel transition group has no active rows");
                const size_t relative = i - group.transition_index;
                const size_t row = relative % population.active_component_cmps;
                const size_t transition = relative / population.active_component_cmps;
                const size_t p_ordinal =
                    1 + 2 * (row * population.transitions + transition);
                const size_t population_ordinal =
                    1 + 2 * size_t(population.active_component_cmps) * population.transitions;
                if (p_ordinal >= UINT32_MAX || population_ordinal > UINT32_MAX)
                  throw std::overflow_error("NVIDIA multilevel storage ordinal overflows");
                const size_t transition_index = multilevel_transition_updates.size();
                multilevel_transition_updates.push_back(compile_multilevel_transition(
                    plan.multilevel_transition_updates[i], uint32_t(p_ordinal),
                    uint32_t(population_ordinal), state, multilevel_coefficients));
                polarization_actions.push_back(NvidiaCompiledPolarizationAction{
                    NvidiaCompiledPolarizationAction::Kind::multilevel_transition,
                    transition_index});
              }
            }
            else
              throw std::invalid_argument("NVIDIA polarization group has an invalid kind");
          }
          if (next_ordinary != size_t(op.descriptor_index) + op.descriptor_count)
            throw std::invalid_argument(
                "NVIDIA polarization groups do not cover the ordinary descriptor span");
          compiled.polarization_count =
              polarization_actions.size() - compiled.polarization_first;
          break;
        }
        case OpKind::restore_magnetic_fields:
        case OpKind::synchronize_magnetic_fields:
          if (op.magnetic_state_index != 0 ||
              op.magnetic_state_count != plan.magnetic_state_arrays.size())
            set_reason(local_error, oi, "magnetic snapshot span is incomplete");
          break;
        case OpKind::phase_material:
        case OpKind::update_material_coefficients:
          if (size_t(op.material_refresh_index) + op.material_refresh_count >
              plan.material_refresh_arrays.size()) {
            set_reason(local_error, oi, "material refresh descriptor span is out of range");
            break;
          }
          compiled.material_first = op.material_refresh_index;
          compiled.material_count = op.material_refresh_count;
          for (size_t i = compiled.material_first;
               i < compiled.material_first + compiled.material_count; ++i) {
            const MaterialRefreshFamily expected = op.kind == OpKind::phase_material
                                                       ? MaterialRefreshFamily::chi1inv
                                                       : plan.material_refresh_arrays[i].family;
            if (plan.material_refresh_arrays[i].family != expected ||
                (op.kind == OpKind::update_material_coefficients &&
                 expected == MaterialRefreshFamily::chi1inv)) {
              set_reason(local_error, oi, "material refresh span has the wrong family");
              break;
            }
          }
          break;
        case OpKind::increment_time:
          break;

        case OpKind::host_callback: {
          if (op.descriptor_count != 1 || op.descriptor_index >= plan.host_segments.size()) {
            set_reason(local_error, oi, "host segment span is out of range");
            break;
          }
          const HostSegment &source = plan.host_segments[op.descriptor_index];
          NvidiaCompiledHostSegment segment = {};
          segment.segment = source;
          segment.host_plan.program = StepProgram::ordinary;
          segment.staging_bytes = 0;
          for (size_t i = source.callback_index;
               i < size_t(source.callback_index) + source.callback_count; ++i) {
            segment.callbacks.push_back(plan.host_callbacks[i]);
            segment.resolution_layout.resize(
                std::max(segment.resolution_layout.size(),
                         plan.host_callbacks[i].published_layout.size()));
          }
          for (size_t i = source.operation_index;
               i < size_t(source.operation_index) + source.operation_count; ++i) {
            Operation covered = plan.operations[i];
            /* The resident scheduler has already evaluated the marker guard.
               A standalone CPU segment must not evaluate the same guard again. */
            covered.guard = guard_always();
            segment.host_plan.operations.push_back(covered);
          }
          for (const BufferAccess &access : op.accesses) {
            if (!is_valid(access.array.id) || access.array.id.value >= state.plan_.arrays.size()) {
              set_reason(local_error, oi, "host segment access has an invalid ArrayId");
              break;
            }
            const ArraySpec &spec = state.plan_.arrays[access.array.id.value];
            const size_t host_bytes = checked_product(
                access.array.elements, host_element_bytes(spec.element_type),
                "sizing NVIDIA host-segment native transfer");
            const char *host_base = static_cast<const char *>(
                f_.array_catalog->resolve_untyped(access.array.id));
            if (!host_base) {
              set_reason(local_error, oi, "host segment access resolves to a null host row");
              break;
            }
            const AccessRange range = checked_access(state.plan_, access.array, host_base,
                                                     host_bytes);
            NvidiaCompiledHostTransfer transfer = {};
            transfer.array = access.array;
            transfer.mode = access.mode;
            transfer.spec = spec;
            transfer.staging_offset = segment.staging_bytes;
            transfer.storage_offset = range.storage_offset;
            transfer.storage_bytes = range.storage_bytes;
            transfer.host_offset = checked_product(
                access.array.offset, host_element_bytes(spec.element_type),
                "resolving NVIDIA host-segment native offset");
            segment.staging_bytes = checked_add(segment.staging_bytes, range.storage_bytes,
                                                "sizing NVIDIA host-segment staging");
            segment.transfers.push_back(transfer);
          }
          if (!local_error.empty()) break;
          host_staging_bytes = std::max(host_staging_bytes, segment.staging_bytes);
          compiled.host_segment_index = host_segments.size();
          host_segments.push_back(segment);
          break;
        }
        case OpKind::zero_boundary:
        case OpKind::pack_halo:
        case OpKind::exchange_local:
        case OpKind::unpack_halo:
        case OpKind::reduction: set_reason(local_error, oi, op_kind_name(op.kind)); break;
        case OpKind::pack_state:
        case OpKind::unpack_state:
          if (plan.program != StepProgram::solve_cw)
            set_reason(local_error, oi, "CW state marker in an ordinary timestep plan");
          break;
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
      if (nvidia::testing::consume_failure_for_testing(
              nvidia::testing::failure_point::material_compile))
        throw std::runtime_error("injected NVIDIA material executable compilation failure");
    if (local_error.empty()) {
      const uint64_t source_topology =
          source_plan ? source_plan_topology_signature(*source_plan)
                      : source_plan_topology_signature(SourcePlan());
      executable.reset(new NvidiaExecutable(
          this, plan.program, plan.signature,
          source_structural_step_signature(plan, source_plan), source_topology,
          f_.mutation_generation, claim_executable_generation(), state.fingerprint_,
          state.state_token_, operations,
          curl_updates,
          cylindrical_radial_prefixes, bfast_updates, beta_updates, cylindrical_m_updates,
          cylindrical_axis_updates, cylindrical_origin_actions, constitutive_updates, zero_updates,
          halo_plans, halo_gathers, halo_scatters, halo_scratch_bytes, finite_checks,
          source_batches, source_points, source_copies, polarization_updates,
          polarization_actions, multilevel_population_updates, multilevel_population_terms,
          multilevel_transition_updates, multilevel_coefficients, multilevel_scratch_bytes,
          polarization_subtractions, dft_updates, dft_omega, legacy_flux_updates,
          legacy_flux_terms, legacy_flux_partial_count,
          magnetic_states, magnetic_snapshot_bytes,
          magnetic_layout_fingerprint(state.fingerprint_, plan.magnetic_state_arrays),
          plan.magnetic_half_step, material_refreshes, material_staging_bytes,
          plan.material_phase_target_signature,
          source_plan ? source_plan->scalars.size() : 0,
          source_staging_elements, host_segments, host_staging_bytes, state));
    }
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
  local_error.clear();
  try {
    configure_graph_execution(plan, *executable, state);
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown NVIDIA graph configuration failure";
  }
  if (or_to_all(!local_error.empty())) {
    executable.reset();
    if (local_error.empty())
      local_error = "NVIDIA graph configuration was rejected on another rank";
    throw std::runtime_error(local_error);
  }
  if (plan.program == StepProgram::ordinary)
    state.graph_statistics_ = executable->graph_statistics_;
  ++state.executable_build_count_;
  return executable.release();
}

bool NvidiaBackend::refresh_source_values(const StepPlan &plan,
                                          Executable &raw_executable,
                                          BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  NvidiaExecutable &executable = checked_executable(raw_executable, state);
  if (!state.initialized_ || state.transfer_failed_ || is_poisoned())
    throw std::logic_error("NVIDIA source refresh requires an initialized usable state");
  NvidiaStagedSourceRefresh staged;
  if (!stage_source_value_refresh(f_, plan, executable, state, staged) || !staged.advanced)
    return false;

  if (!staged.points.empty()) {
    bool enqueued = false;
    try {
      nvidia::device_scope scope(state.device_);
      if (nvidia::testing::consume_failure_for_testing(
              nvidia::testing::failure_point::source_value_copy))
        throw std::runtime_error("injected NVIDIA source-value copy failure");
      enqueued = true;
      nvidia::copy_host_to_device_async(executable.source_points_, 0, staged.points.data(),
                                        staged.points.size() * sizeof(staged.points[0]),
                                        *state.transfer_);
      state.transfer_->synchronize();
      if (nvidia::testing::consume_failure_for_testing(
              nvidia::testing::failure_point::source_value_sync))
        throw std::runtime_error("injected NVIDIA source-value synchronization failure");
    }
    catch (...) {
      if (enqueued) {
        state.transfer_failed_ = true;
        poison();
      }
      throw;
    }
  }

  commit_source_value_refresh(f_, plan, executable, staged);
  ++state.source_value_reuse_count_;
  if (plan.program == StepProgram::ordinary)
    state.graph_statistics_ = executable.graph_statistics_;
  return true;
}

namespace {

array_kind cw_array_kind(CwStateFamily family) {
  switch (family) {
    case CwStateFamily::primary:
    case CwStateFamily::paired_primary: return array_kind::f;
    case CwStateFamily::pml_u: return array_kind::f_u;
    case CwStateFamily::conductivity: return array_kind::f_cond;
    case CwStateFamily::bfast: return array_kind::f_bfast;
    case CwStateFamily::constitutive_w: return array_kind::f_w;
  }
  throw std::invalid_argument("NVIDIA solve_cw row has an invalid state family");
}

const ArraySpec &validate_cw_field_array(const NvidiaBackendState &state, ArrayId id,
                                         int chunk, array_kind kind, component c, int cmp,
                                         const char *what) {
  if (!is_valid(id) || id.value >= state.plan_.arrays.size() ||
      id.value >= state.plan_.keys.size())
    throw std::out_of_range(std::string(what) + " uses an invalid ArrayId");
  const ArraySpec &spec = state.plan_.arrays[id.value];
  const StorageKey &key = state.plan_.keys[id.value];
  if (spec.id != id || spec.role != array_role::field ||
      spec.element_type != ElementType::realnum_value || is_valid(spec.alias_of))
    throw std::invalid_argument(std::string(what) + " has incompatible device storage");
  if (key.chunk != chunk || key.kind != int(kind) || key.component_ != int(c) ||
      key.cmp != cmp || key.aux != 0)
    throw std::invalid_argument(std::string(what) + " has the wrong storage identity");
  return spec;
}

nvidia::cw_workspace_shape cw_workspace_shape(size_t real_count, int L,
                                               nvidia::scalar_precision precision) {
  if (!real_count) throw std::invalid_argument("NVIDIA solve_cw state vector is empty");
  if (L < 1) throw std::invalid_argument("NVIDIA solve_cw requires L >= 1");
  const size_t two_l = checked_product(size_t(L), size_t(2),
                                       "sizing NVIDIA solve_cw vector count");
  const size_t vector_count = checked_add(two_l, size_t(5),
                                          "sizing NVIDIA solve_cw vector count");
  const size_t scalar_size = precision == nvidia::scalar_precision::f32 ? sizeof(float)
                                                                        : sizeof(double);
  const size_t vector_elements = checked_product(real_count, vector_count,
                                                  "sizing NVIDIA solve_cw workspace");
  const size_t blocks = std::min<size_t>(128, checked_add(real_count, size_t(255),
                                                          "sizing solve_cw reduction grid") /
                                                  256);
  nvidia::cw_workspace_shape shape;
  shape.vector_elements = real_count;
  shape.vector_count = vector_count;
  shape.vector_bytes = checked_product(vector_elements, scalar_size,
                                       "sizing NVIDIA solve_cw workspace bytes");
  shape.reduction_partial_bytes = checked_product(
      checked_product(blocks, size_t(2), "sizing NVIDIA solve_cw reduction partials"),
      sizeof(double), "sizing NVIDIA solve_cw reduction bytes");
  return shape;
}

size_t launch_cw_reconciliation_device(NvidiaCwExecutable &cw, NvidiaBackendState &state,
                                       uint32_t operation_index, bool skip_w_components) {
  if (operation_index >= cw.timestep_->operations_.size())
    throw std::logic_error("NVIDIA solve_cw reconciliation operation is out of range");
  const NvidiaCompiledOperation &op = cw.timestep_->operations_[operation_index];
  size_t launches = 0;
  if (op.kind == OpKind::transfer_halo) {
    for (size_t i = op.first; i < op.first + op.count; ++i) {
      nvidia::launch_zero(cw.timestep_->zero_updates_[i], *state.transfer_);
      ++launches;
    }
    for (size_t i = op.halo_first; i < op.halo_first + op.halo_count; ++i) {
      nvidia::launch_halo_gather(cw.timestep_->halo_plans_[i].gather,
                                 cw.timestep_->halo_gathers_.opaque_handle(),
                                 cw.timestep_->halo_scratch_.opaque_handle(), *state.transfer_);
      ++launches;
    }
    for (size_t i = op.halo_first; i < op.halo_first + op.halo_count; ++i) {
      nvidia::launch_halo_scatter(cw.timestep_->halo_plans_[i].scatter,
                                  cw.timestep_->halo_scatters_.opaque_handle(),
                                  cw.timestep_->halo_scratch_.opaque_handle(), *state.transfer_);
      ++launches;
    }
    return launches;
  }
  if (op.kind == OpKind::update_eh) {
    if (op.copy_count || op.subtraction_count || op.source_count)
      throw std::logic_error("NVIDIA solve_cw constitutive operation retained source state");
    for (size_t i = op.first; i < op.first + op.count; ++i) {
      const nvidia::constitutive_launch &update = cw.timestep_->constitutive_updates_[i];
      if (skip_w_components && update.target_w) continue;
      nvidia::launch_constitutive(update, *state.transfer_);
      ++launches;
    }
    return launches;
  }
  throw std::logic_error("NVIDIA solve_cw reconciliation reference has the wrong kind");
}

class CwGraphCleanupFailure : public std::runtime_error {
public:
  explicit CwGraphCleanupFailure(const std::string &message) : std::runtime_error(message) {}
};

template <typename Launch>
NvidiaCapturedGraph capture_cw_graph(NvidiaBackendState &state, const std::string &label,
                                     Launch launch) {
  NvidiaCapturedGraph captured;
  try {
    captured.definition.begin_capture(*state.transfer_, label);
    launch();
    captured.definition.end_capture(*state.transfer_);
  }
  catch (...) {
    std::string cleanup_error;
    cleanup_cw_graph(captured, cleanup_error);
    if (!cleanup_error.empty())
      throw CwGraphCleanupFailure(std::string("CUDA CW graph capture rollback failed: ") +
                                  cleanup_error);
    throw;
  }
  return captured;
}

void clear_cw_device_graphs(NvidiaCwExecutable &cw) {
  cw.graphs_enabled_ = false;
  std::string cleanup_error;
  bool released = cleanup_cw_graphs(cw.rhs_stage_graphs_, cleanup_error);
  released = cleanup_cw_graphs(cw.vector_graphs_, cleanup_error) && released;
  released = cleanup_cw_graphs(cw.reduction_graphs_, cleanup_error) && released;
  released = cleanup_cw_graph(cw.final_dft_graph_, cleanup_error) && released;
  released = cleanup_cw_graph(cw.operator_finalize_graph_, cleanup_error) && released;
  released = cleanup_cw_graph(cw.operator_prelude_graph_, cleanup_error) && released;
  released = cleanup_cw_graph(cw.pack_graph_, cleanup_error) && released;
  released = cleanup_cw_graph(cw.rhs_zero_graph_, cleanup_error) && released;
  if (released && cleanup_error.empty()) {
    cw.rhs_stage_graphs_.clear();
    cw.rhs_stage_reconciliation_kernel_counts_.clear();
    cw.vector_graphs_.clear();
    cw.reduction_graphs_.clear();
    try {
      cw.graph_scalars_.reset();
      cw.final_dft_scalars_.reset();
    }
    catch (const std::exception &failure) {
      remember_cw_graph_cleanup_error(cleanup_error, failure);
    }
    catch (...) { cleanup_error = "unknown CUDA CW graph scalar cleanup failure"; }
  }
  reconcile_graph_stage(cleanup_error, "CUDA CW workspace graph cleanup failed on another rank");
  cw.graph_capture_count_ = 0;
  cw.graph_instantiate_count_ = 0;
  cw.graph_operator_reconciliation_kernel_count_ = 0;
}

bool configure_cw_device_graphs(NvidiaCwExecutable &cw, NvidiaBackendState &state,
                                GraphExecutionMode requested) {
  bool runtime_supported = false;
  std::string local_error;
  try { runtime_supported = nvidia::query_graph_capability().capture_supported; }
  catch (const std::exception &error) { local_error = error.what(); }
  catch (...) { local_error = "unknown CUDA CW workspace graph capability-query failure"; }
  reconcile_graph_stage(local_error,
                        "CUDA CW workspace graph capability query failed on another rank");
  if (!reconcile_graph_support(requested, runtime_supported)) return false;

  nvidia::device_buffer scalars;
  nvidia::device_buffer final_dft_scalars;
  NvidiaCapturedGraph rhs_zero;
  std::vector<NvidiaCapturedGraph> rhs_stages;
  std::vector<size_t> rhs_reconciliation_launches;
  NvidiaCapturedGraph prelude;
  NvidiaCapturedGraph pack;
  NvidiaCapturedGraph finalize;
  std::vector<NvidiaCapturedGraph> vectors;
  std::vector<NvidiaCapturedGraph> reductions;
  NvidiaCapturedGraph final_dft;
  size_t captures = 0;
  size_t reconciliation_launches = 0;
  bool capture_cleanup_failed = false;
  const auto discard_candidates = [&]() {
    std::string cleanup_error;
    bool released = cleanup_cw_graphs(rhs_stages, cleanup_error);
    released = cleanup_cw_graphs(vectors, cleanup_error) && released;
    released = cleanup_cw_graphs(reductions, cleanup_error) && released;
    released = cleanup_cw_graph(final_dft, cleanup_error) && released;
    released = cleanup_cw_graph(finalize, cleanup_error) && released;
    released = cleanup_cw_graph(prelude, cleanup_error) && released;
    released = cleanup_cw_graph(pack, cleanup_error) && released;
    released = cleanup_cw_graph(rhs_zero, cleanup_error) && released;
    if (released && cleanup_error.empty()) {
      rhs_stages.clear();
      vectors.clear();
      reductions.clear();
      try {
        scalars.reset();
        final_dft_scalars.reset();
        state.transfer_->synchronize();
      }
      catch (const std::exception &failure) {
        remember_cw_graph_cleanup_error(cleanup_error, failure);
      }
      catch (...) { cleanup_error = "unknown CUDA CW graph candidate cleanup failure"; }
    }
    reconcile_graph_stage(cleanup_error,
                          "CUDA CW workspace graph candidate cleanup failed on another rank");
  };
  local_error.clear();
  try {
    scalars.allocate(sizeof(nvidia::cw_graph_scalars), state.device_);
    final_dft_scalars.allocate(sizeof(StepScalars), state.device_);
    state.transfer_->synchronize();
    const nvidia::cw_graph_scalars *device_scalars =
        static_cast<const nvidia::cw_graph_scalars *>(scalars.opaque_handle());
    const StepScalars *device_dft_scalars =
        static_cast<const StepScalars *>(final_dft_scalars.opaque_handle());
    rhs_zero = capture_cw_graph(state, "cw-rhs-zero", [&]() {
      for (const nvidia::cw_zero_launch &zero : cw.zeroes_)
        nvidia::launch_cw_zero(zero, *state.transfer_);
    });
    ++captures;
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::cw_graph_capture))
      throw std::runtime_error("injected CUDA CW graph capture failure");
    rhs_stages.reserve(cw.rhs_stages_.size());
    rhs_reconciliation_launches.reserve(cw.rhs_stages_.size());
    for (size_t stage_index = 0; stage_index < cw.rhs_stages_.size(); ++stage_index) {
      const NvidiaCwExecutable::Stage &stage = cw.rhs_stages_[stage_index];
      size_t stage_reconciliation = 0;
      std::ostringstream label;
      label << "cw-rhs-stage-" << stage_index;
      rhs_stages.push_back(capture_cw_graph(state, label.str(), [&]() {
        for (size_t i = stage.source_first;
             i < stage.source_first + stage.source_count; ++i)
          nvidia::launch_cw_source_batch(cw.rhs_sources_[i],
                                         cw.timestep_->source_scalars_.opaque_handle(),
                                         *state.transfer_);
        stage_reconciliation += launch_cw_reconciliation_device(
            cw, state, stage.boundary_operation, false);
        stage_reconciliation += launch_cw_reconciliation_device(
            cw, state, stage.constitutive_operation, false);
      }));
      rhs_reconciliation_launches.push_back(stage_reconciliation);
      ++captures;
    }
    prelude = capture_cw_graph(state, "cw-unpack-reconciliation", [&]() {
      for (const nvidia::cw_state_row_launch &row : cw.rows_)
        nvidia::launch_cw_unpack_graph(row, device_scalars, cw.real_count_, *state.transfer_);
      for (int i = 0; i < 3; ++i)
        reconciliation_launches += launch_cw_reconciliation_device(
            cw, state, cw.unpack_operations_[i],
            i == 1 && cw.unpack_skip_w_components_);
    });
    ++captures;
    pack = capture_cw_graph(state, "cw-pack", [&]() {
      for (const nvidia::cw_state_row_launch &row : cw.rows_)
        nvidia::launch_cw_pack_graph(row, device_scalars, cw.real_count_, *state.transfer_);
    });
    ++captures;
    nvidia::cw_operator_launch operator_launch = {
        NULL, NULL, NULL, cw.real_count_, 0.0, 0.0, 0.0, cw.precision_};
    finalize = capture_cw_graph(state, "cw-operator-finalize", [&]() {
      nvidia::launch_cw_operator_finalize_graph(operator_launch, device_scalars,
                                                *state.transfer_);
    });
    ++captures;
    vectors.reserve(5);
    for (int operation = int(nvidia::cw_vector_operation::copy);
         operation <= int(nvidia::cw_vector_operation::linear_f64_coefficient); ++operation) {
      nvidia::cw_vector_launch launch = {
          NULL, NULL, NULL, cw.real_count_, 0.0, cw.precision_,
          nvidia::cw_vector_operation(operation)};
      std::ostringstream label;
      label << "cw-vector-" << operation;
      vectors.push_back(capture_cw_graph(state, label.str(), [&]() {
        nvidia::launch_cw_vector_graph(launch, device_scalars, *state.transfer_);
      }));
      ++captures;
    }
    const size_t reduction_blocks =
        std::min<size_t>(128, checked_add(cw.real_count_, size_t(255),
                                          "sizing CW graph reduction") /
                                  256);
    reductions.reserve(3);
    for (int operation = 0; operation < 3; ++operation) {
      nvidia::cw_reduction_launch launch = {
          NULL, NULL, cw.workspace_->reduction_partials_.opaque_handle(),
          cw.workspace_->reduction_result_.opaque_handle(), cw.real_count_,
          reduction_blocks, 1.0, cw.precision_};
      std::ostringstream label;
      label << "cw-reduction-" << operation;
      reductions.push_back(capture_cw_graph(state, label.str(), [&]() {
        if (operation == 0)
          nvidia::launch_cw_dot_graph(launch, device_scalars, *state.transfer_);
        else if (operation == 1)
          nvidia::launch_cw_max_abs_graph(launch, device_scalars, *state.transfer_);
        else
          nvidia::launch_cw_scaled_norm_sum_graph(launch, device_scalars,
                                                  *state.transfer_);
      }));
      ++captures;
    }
    if (!cw.final_dfts_.empty()) {
      if (cw.final_dfts_.size() != cw.final_dft_due_slots_.size())
        throw std::logic_error("NVIDIA CW final DFT graph authority is incomplete");
      final_dft = capture_cw_graph(state, "cw-final-dft", [&]() {
        for (size_t i = 0; i < cw.final_dfts_.size(); ++i) {
          const uint32_t slot = cw.final_dft_due_slots_[i];
          nvidia::launch_dft_graph(cw.final_dfts_[i], device_dft_scalars, slot / 64,
                                   slot % 64, *state.transfer_);
        }
      });
      ++captures;
    }
  }
  catch (const CwGraphCleanupFailure &error) {
    local_error = error.what();
    capture_cleanup_failed = true;
  }
  catch (const std::exception &error) { local_error = error.what(); }
  catch (...) { local_error = "unknown CUDA CW workspace graph capture failure"; }
  if (or_to_all(!local_error.empty())) {
    discard_candidates();
    reconcile_graph_stage(capture_cleanup_failed ? local_error : std::string(),
                          "CUDA CW graph capture rollback failed on another rank");
    if (requested == GraphExecutionMode::required)
      throw std::runtime_error(local_error.empty()
                                   ? "CUDA CW workspace graph capture failed on another rank"
                                   : local_error);
    return false;
  }

  size_t instantiates = 0;
  local_error.clear();
  try {
    rhs_zero.executable.instantiate(rhs_zero.definition);
    ++instantiates;
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::cw_graph_instantiate))
      throw std::runtime_error("injected CUDA CW graph instantiate failure");
    for (NvidiaCapturedGraph &stage : rhs_stages) {
      stage.executable.instantiate(stage.definition);
      ++instantiates;
    }
    prelude.executable.instantiate(prelude.definition);
    ++instantiates;
    pack.executable.instantiate(pack.definition);
    ++instantiates;
    finalize.executable.instantiate(finalize.definition);
    ++instantiates;
    for (NvidiaCapturedGraph &vector : vectors) {
      vector.executable.instantiate(vector.definition);
      ++instantiates;
    }
    for (NvidiaCapturedGraph &reduction : reductions) {
      reduction.executable.instantiate(reduction.definition);
      ++instantiates;
    }
    if (!cw.final_dfts_.empty()) {
      final_dft.executable.instantiate(final_dft.definition);
      ++instantiates;
    }
  }
  catch (const std::exception &error) { local_error = error.what(); }
  catch (...) { local_error = "unknown CUDA CW workspace graph instantiate failure"; }
  if (or_to_all(!local_error.empty())) {
    discard_candidates();
    if (requested == GraphExecutionMode::required)
      throw std::runtime_error(local_error.empty()
                                   ? "CUDA CW workspace graph instantiate failed on another rank"
                                   : local_error);
    return false;
  }

  cw.graph_scalars_ = std::move(scalars);
  cw.final_dft_scalars_ = std::move(final_dft_scalars);
  cw.rhs_zero_graph_ = std::move(rhs_zero);
  cw.rhs_stage_graphs_ = std::move(rhs_stages);
  cw.rhs_stage_reconciliation_kernel_counts_ = std::move(rhs_reconciliation_launches);
  cw.operator_prelude_graph_ = std::move(prelude);
  cw.pack_graph_ = std::move(pack);
  cw.operator_finalize_graph_ = std::move(finalize);
  cw.vector_graphs_ = std::move(vectors);
  cw.reduction_graphs_ = std::move(reductions);
  cw.final_dft_graph_ = std::move(final_dft);
  cw.graph_capture_count_ = captures;
  cw.graph_instantiate_count_ = instantiates;
  cw.graph_operator_reconciliation_kernel_count_ = reconciliation_launches;
  cw.graphs_enabled_ = true;
  return true;
}

} // namespace

bool NvidiaBackend::supports_cw(const CwSolveRequest &, std::string &why) const {
  why.clear();
  return true;
}

Executable *NvidiaBackend::preflight_cw(const CwSolveRequest &request,
                                        const StepPlan &ordinary_plan,
                                        const StepPlan &step_plan, const CwPlan &cw_plan,
                                        Executable &raw_ordinary, Executable *cached,
                                        BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  NvidiaExecutable &ordinary = checked_executable(raw_ordinary, state);
  if (!state.initialized_ || state.transfer_failed_)
    throw std::logic_error("NVIDIA solve_cw requires an initialized usable state");
  if (!std::isfinite(request.tolerance) || request.tolerance <= 0.0 || request.maxiters < 1 ||
      request.L < 1 || !std::isfinite(request.frequency.real()) ||
      !std::isfinite(request.frequency.imag()) || request.frequency == std::complex<double>() ||
      request.eigfrequency)
    throw std::invalid_argument("NVIDIA solve_cw request is unsupported or invalid");
  const bool profile_only = cw_profile_mode_requested();
  if (profile_only && (request.L != 2 || request.maxiters < 3))
    throw std::invalid_argument("NVIDIA solve_cw profile mode requires L=2 and maxiters>=3");
  if (profile_only && !nvtx_api().available())
    throw std::runtime_error("NVIDIA solve_cw profile mode requires libnvToolsExt");
  if (request.entry_t != f_.t || request.entry_time != cw_source_time(f_.t, f_.dt, 0.0))
    throw std::invalid_argument("NVIDIA solve_cw request has stale entry-time state");
  if (count_processors() != 1)
    throw std::invalid_argument("NVIDIA solve_cw does not support MPI decomposition");
  if (f_.is_real || f_.gv.dim == Dcyl || f_.beta != 0.0 || f_.m != 0.0 ||
      f_.phasein_time > 0 || f_.synchronized_magnetic_fields || f_.fluxes)
    throw std::invalid_argument("NVIDIA solve_cw field state is outside the initial slice");
  for (double value : f_.bfast_scaled_k)
    if (value != 0.0)
      throw std::invalid_argument("NVIDIA solve_cw does not support BFAST coordinates");
  for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
    if (!f_.chunks[chunk] || !f_.chunks[chunk]->is_mine()) continue;
    if (f_.chunks[chunk]->new_s || f_.chunks[chunk]->s->has_nonlinearities())
      throw std::invalid_argument("NVIDIA solve_cw requires stable linear material storage");
    FOR_FIELD_TYPES(ft) if (f_.chunks[chunk]->pol[ft])
      throw std::invalid_argument("NVIDIA solve_cw does not support polarization state");
  }
  if (!step_plan.polarization_updates.empty() || !step_plan.polarization_subtractions.empty())
    throw std::invalid_argument("NVIDIA solve_cw plan contains polarization work");
  if (step_plan.program != StepProgram::solve_cw ||
      step_plan.signature != compute_step_plan_signature(step_plan))
    throw std::invalid_argument("NVIDIA solve_cw received a stale or ordinary StepPlan");
  std::string validation_error;
  if (!validate_cw_state_layout(f_, step_plan.cw_state_layout, &validation_error))
    throw std::invalid_argument(validation_error.empty() ? "invalid NVIDIA solve_cw state layout"
                                                         : validation_error);
  if (cw_plan.signature != compute_cw_plan_signature(cw_plan) ||
      !validate_cw_plan(f_, step_plan, cw_plan, &validation_error))
    throw std::invalid_argument(validation_error.empty() ? "invalid NVIDIA solve_cw plan"
                                                         : validation_error);
  if (cw_plan.state_layout_signature != step_plan.cw_state_layout.signature ||
      cw_plan.step_plan_signature != step_plan.signature)
    throw std::invalid_argument("NVIDIA solve_cw plan fingerprints disagree");

  const CwStateLayout &layout = step_plan.cw_state_layout;
  std::vector<nvidia::cw_state_row_launch> rows;
  rows.reserve(layout.rows.size());
  size_t expected_complex_offset = 0;
  bool have_precision = false;
  nvidia::scalar_precision precision = nvidia::scalar_precision::f64;
  for (const CwStateRow &row : layout.rows) {
    if (row.chunk < 0 || row.chunk >= f_.num_chunks || !f_.chunks[row.chunk] ||
        !f_.chunks[row.chunk]->is_mine())
      throw std::invalid_argument("NVIDIA solve_cw row references an unowned chunk");
    if (row.complex_offset != expected_complex_offset || !row.complex_count)
      throw std::invalid_argument("NVIDIA solve_cw row offsets are not contiguous");
    const array_kind kind = cw_array_kind(row.family);
    const ArraySpec &real_spec = validate_cw_field_array(
        state, row.real_array, row.chunk, kind, row.storage_component, 0,
        "NVIDIA solve_cw real row");
    const ArraySpec &imag_spec = validate_cw_field_array(
        state, row.imag_array, row.chunk, kind, row.storage_component, 1,
        "NVIDIA solve_cw imaginary row");
    if (real_spec.storage != imag_spec.storage || real_spec.elements != imag_spec.elements)
      throw std::invalid_argument("NVIDIA solve_cw row halves have incompatible storage");
    const nvidia::scalar_precision row_precision =
        real_spec.storage == Precision::f32 ? nvidia::scalar_precision::f32
                                            : nvidia::scalar_precision::f64;
    if (have_precision && precision != row_precision)
      throw std::invalid_argument("NVIDIA solve_cw state rows mix storage precision");
    precision = row_precision;
    have_precision = true;

    nvidia::cw_state_row_launch compiled = {};
    compiled.region = flat_region_for(row.owned_region);
    const ptrdiff_t maximum = checked_region_max(compiled.region);
    validate_index_range(state.plan_, row.real_array, ptrdiff_t(compiled.region.base), maximum,
                         "NVIDIA solve_cw real row");
    validate_index_range(state.plan_, row.imag_array, ptrdiff_t(compiled.region.base), maximum,
                         "NVIDIA solve_cw imaginary row");
    size_t region_count = 1;
    for (int axis = 0; axis < 3; ++axis)
      region_count = checked_product(region_count, compiled.region.counts[axis],
                                     "counting NVIDIA solve_cw row points");
    if (region_count != row.complex_count)
      throw std::invalid_argument("NVIDIA solve_cw row region/count mismatch");
    const size_t complex_end = checked_add(row.complex_offset, row.complex_count,
                                           "validating NVIDIA solve_cw vector extent");
    const size_t real_end = checked_product(complex_end, size_t(2),
                                            "validating NVIDIA solve_cw real-vector extent");
    if (real_end > layout.real_count)
      throw std::out_of_range("NVIDIA solve_cw row exceeds the real-vector extent");
    compiled.real_values = state.arenas_->resolve(row.real_array.value).address;
    compiled.imaginary_values = state.arenas_->resolve(row.imag_array.value).address;
    compiled.complex_offset = row.complex_offset;
    compiled.complex_count = row.complex_count;
    compiled.precision = row_precision;
    rows.push_back(compiled);
    expected_complex_offset = complex_end;
  }
  if (!have_precision || expected_complex_offset != layout.complex_count ||
      layout.real_count != checked_product(layout.complex_count, size_t(2),
                                           "validating NVIDIA solve_cw vector length"))
    throw std::invalid_argument("NVIDIA solve_cw state vector totals are inconsistent");

  std::vector<nvidia::cw_zero_launch> zeroes;
  zeroes.reserve(layout.zero_arrays.size());
  uint32_t previous_zero = 0;
  bool have_previous_zero = false;
  for (const ArrayRef &ref : layout.zero_arrays) {
    if (!is_valid(ref.id) || ref.id.value >= state.plan_.arrays.size())
      throw std::out_of_range("NVIDIA solve_cw zero set uses an invalid ArrayId");
    const ArraySpec &spec = state.plan_.arrays[ref.id.value];
    if (spec.role != array_role::field || spec.element_type != ElementType::realnum_value ||
        is_valid(spec.alias_of) ||
        (spec.storage == Precision::f32 ? nvidia::scalar_precision::f32
                                        : nvidia::scalar_precision::f64) != precision ||
        ref.offset != 0 ||
        ref.elements != spec.elements)
      throw std::invalid_argument("NVIDIA solve_cw zero set has incompatible storage");
    if (have_previous_zero && ref.id.value <= previous_zero)
      throw std::invalid_argument("NVIDIA solve_cw zero set is not canonical");
    zeroes.push_back(nvidia::cw_zero_launch{state.arenas_->resolve(ref.id.value).address,
                                           ref.elements, precision});
    previous_zero = ref.id.value;
    have_previous_zero = true;
  }

  if (!f_.descriptors)
    throw std::logic_error("NVIDIA solve_cw requires prepared source/DFT descriptors");
  const SourcePlan &source_plan = f_.descriptors->sources;
  std::vector<NvidiaCwExecutable::Stage> rhs_stages;
  std::vector<nvidia::cw_source_batch_launch> rhs_sources;
  std::vector<nvidia::source_point> source_points;
  rhs_stages.reserve(cw_plan.rhs_stages.size());
  rhs_sources.reserve(cw_plan.rhs_sources.size());
  for (const CwRhsStage &stage : cw_plan.rhs_stages) {
    if (stage.source_time_index != 0 || stage.source_time_count != source_plan.source_times.size() ||
        size_t(stage.source_index) + stage.source_count > cw_plan.rhs_sources.size() ||
        stage.boundary.operation_index >= step_plan.operations.size() ||
        stage.constitutive.operation_index >= step_plan.operations.size())
      throw std::invalid_argument("NVIDIA solve_cw RHS stage has invalid descriptor spans");
    const Operation &boundary = step_plan.operations[stage.boundary.operation_index];
    const Operation &constitutive = step_plan.operations[stage.constitutive.operation_index];
    if (boundary.kind != OpKind::transfer_halo || boundary.ft != stage.ft ||
        constitutive.kind != OpKind::update_eh ||
        constitutive.ft != (stage.ft == B_stuff ? H_stuff : E_stuff))
      throw std::invalid_argument("NVIDIA solve_cw RHS stage references the wrong operations");
    NvidiaCwExecutable::Stage compiled_stage = {stage.source_time_offset, rhs_sources.size(),
                                                stage.source_count,
                                                stage.boundary.operation_index,
                                                stage.constitutive.operation_index};
    for (size_t i = stage.source_index; i < size_t(stage.source_index) + stage.source_count; ++i) {
      const CwRhsSourceDescriptor &reference = cw_plan.rhs_sources[i];
      if (reference.mode !=
              CwRhsSourceMode::primary_subtract_current_dt_including_integrated ||
          reference.source_descriptor_index >= source_plan.sources.size())
        throw std::invalid_argument("NVIDIA solve_cw RHS source reference is invalid");
      const SourceDescriptor &source = source_plan.sources[reference.source_descriptor_index];
      if (source.ft != stage.ft || source.source_ordinal != reference.source_ordinal)
        throw std::invalid_argument("NVIDIA solve_cw RHS source order is non-canonical");
      rhs_sources.push_back(
          compile_cw_source_batch(source, source_plan, f_, state, source_points));
    }
    rhs_stages.push_back(compiled_stage);
  }
  if (rhs_sources.size() != cw_plan.rhs_sources.size())
    throw std::invalid_argument("NVIDIA solve_cw RHS source coverage is incomplete");

  std::vector<nvidia::dft_launch> final_dfts;
  std::vector<double> dft_omega;
  final_dfts.reserve(cw_plan.final_dfts.size());
  for (size_t i = 0; i < cw_plan.final_dfts.size(); ++i) {
    const CwDftDescriptorRef &reference = cw_plan.final_dfts[i];
    if (reference.descriptor_index >= f_.descriptors->dfts.size())
      throw std::out_of_range("NVIDIA solve_cw final DFT reference is out of range");
    const DftDescriptor &descriptor = f_.descriptors->dfts[reference.descriptor_index];
    if (reference.descriptor_index != i || reference.chunk != descriptor.chunk ||
        reference.c != descriptor.c || reference.decimation_factor != descriptor.decimation_factor ||
        reference.due_scalar_slot != descriptor.due_scalar_slot)
      throw std::invalid_argument("NVIDIA solve_cw final DFT reference is non-canonical");
    final_dfts.push_back(compile_dft(descriptor, f_, state, dft_omega));
  }

  const nvidia::cw_workspace_shape shape =
      cw_workspace_shape(layout.real_count, request.L, precision);
  const uint64_t source_topology = source_plan_topology_signature(source_plan);
  const uint64_t structural_step =
      source_structural_step_signature(step_plan, &source_plan);
  const uint64_t structural_cw =
      source_structural_cw_signature(cw_plan, structural_step, source_topology);
  NvidiaStagedSourceRefresh ordinary_refresh;
  if (!stage_source_value_refresh(f_, ordinary_plan, ordinary, state, ordinary_refresh))
    throw std::invalid_argument("NVIDIA solve_cw ordinary source targets are not reusable");
  if (cached) {
    NvidiaCwExecutable *existing = dynamic_cast<NvidiaCwExecutable *>(cached);
    if (!existing || existing->owner_ != this ||
        existing->state_token_ != state.state_token_ ||
        existing->layout_storage_fingerprint_ != layout.storage_fingerprint ||
        existing->device_storage_fingerprint_ != state.fingerprint_ ||
        existing->precision_ != precision || existing->real_count_ != layout.real_count)
      throw std::invalid_argument("NVIDIA solve_cw cache entry has the wrong identity");
    const bool exact_identity = existing->step_plan_signature_ == step_plan.signature &&
                                existing->cw_plan_signature_ == cw_plan.signature;
    if (exact_identity) {
      if (existing->workspace_->accommodates(shape, precision, request.L)) return existing;
    }
    else {
      bool source_values_advanced = false;
      for (int i = 0; i < mutation_kind_count; ++i) {
        const uint64_t current = f_.mutation_generation[i];
        const uint64_t installed = existing->freshness_[i];
        const MutationKind kind = MutationKind(i);
        if (kind == MutationKind::source_values) {
          if (current < installed)
            throw std::invalid_argument("NVIDIA solve_cw cache freshness regressed");
          source_values_advanced = current != installed;
        }
        else if (mutation_may_preserve_executable_structure(kind)) {
          if (current < installed)
            throw std::invalid_argument("NVIDIA solve_cw cache freshness regressed");
        }
        else if (current != installed)
          throw std::invalid_argument("NVIDIA solve_cw cache has stale structural freshness");
      }
      const bool structurally_compatible =
          source_values_advanced && existing->structural_step_signature_ == structural_step &&
          existing->structural_cw_signature_ == structural_cw &&
          existing->source_topology_signature_ == source_topology &&
          existing->workspace_->accommodates(shape, precision, request.L) &&
          existing->rhs_sources_.size() == rhs_sources.size() &&
          existing->source_points_.size() ==
              source_points.size() * sizeof(nvidia::source_point);
      if (structurally_compatible) {
        for (size_t i = 0; i < rhs_sources.size(); ++i)
          if (!same_cw_source_launch_topology(existing->rhs_sources_[i], rhs_sources[i]))
            throw std::invalid_argument("NVIDIA solve_cw source targets are not reusable");
        NvidiaStagedSourceRefresh timestep_refresh;
        if (!ordinary_refresh.advanced ||
            !stage_source_value_refresh(f_, step_plan, *existing->timestep_, state,
                                        timestep_refresh) ||
            !timestep_refresh.advanced)
          throw std::invalid_argument("NVIDIA solve_cw timestep source targets are not reusable");

        /* All identities above are validated before the first transfer.  The
           three live buffers then share one stream transaction, and none of
           their freshness metadata is published until its final sync. */
        const bool have_ordinary_points = !ordinary_refresh.points.empty();
        const bool have_timestep_points = !timestep_refresh.points.empty();
        const bool have_rhs_points = !source_points.empty();
        if ((have_ordinary_points || have_timestep_points) &&
            nvidia::testing::consume_failure_for_testing(
                nvidia::testing::failure_point::source_value_copy))
          throw std::runtime_error("injected NVIDIA source-value copy failure");
        if (have_rhs_points && nvidia::testing::consume_failure_for_testing(
                                   nvidia::testing::failure_point::cw_source_value_copy))
          throw std::runtime_error("injected NVIDIA CW source-value copy failure");
        bool enqueued = false;
        try {
          nvidia::device_scope scope(state.device_);
          if (have_ordinary_points) {
            enqueued = true;
            nvidia::copy_host_to_device_async(
                ordinary.source_points_, 0, ordinary_refresh.points.data(),
                ordinary_refresh.points.size() * sizeof(ordinary_refresh.points[0]),
                *state.transfer_);
          }
          if (have_timestep_points) {
            enqueued = true;
            nvidia::copy_host_to_device_async(
                existing->timestep_->source_points_, 0, timestep_refresh.points.data(),
                timestep_refresh.points.size() * sizeof(timestep_refresh.points[0]),
                *state.transfer_);
          }
          if (have_rhs_points) {
            enqueued = true;
            nvidia::copy_host_to_device_async(
                existing->source_points_, 0, source_points.data(),
                source_points.size() * sizeof(source_points[0]), *state.transfer_);
          }
          if (enqueued) state.transfer_->synchronize();
          if (enqueued && nvidia::testing::consume_failure_for_testing(
                              nvidia::testing::failure_point::source_value_sync))
            throw std::runtime_error("injected NVIDIA source-value synchronization failure");
          if (enqueued && nvidia::testing::consume_failure_for_testing(
                              nvidia::testing::failure_point::cw_source_value_sync))
            throw std::runtime_error(
                "injected NVIDIA CW source-value synchronization failure");
        }
        catch (...) {
          if (enqueued) {
            state.transfer_failed_ = true;
            poison();
          }
          throw;
        }
        commit_source_value_refresh(f_, ordinary_plan, ordinary, ordinary_refresh);
        commit_source_value_refresh(f_, step_plan, *existing->timestep_, timestep_refresh);
        existing->step_plan_signature_ = step_plan.signature;
        existing->cw_plan_signature_ = cw_plan.signature;
        for (int i = 0; i < mutation_kind_count; ++i)
          existing->freshness_[i] = f_.mutation_generation[i];
        state.source_value_reuse_count_ += 3;
        state.graph_statistics_ = ordinary.graph_statistics_;
        return existing;
      }
      if (!source_values_advanced)
        throw std::invalid_argument("NVIDIA solve_cw cache entry has the wrong identity");
    }
  }

  std::unique_ptr<NvidiaCwWorkspace> workspace(
      new NvidiaCwWorkspace(shape, precision, request.L, state.device_));
  std::unique_ptr<Executable> compiled_step(compile(step_plan, state));
  NvidiaExecutable *timestep = dynamic_cast<NvidiaExecutable *>(compiled_step.get());
  if (!timestep) throw std::logic_error("NVIDIA solve_cw compiler returned the wrong type");
  std::unique_ptr<NvidiaExecutable> timestep_owner(timestep);
  compiled_step.release();
  std::unique_ptr<NvidiaCwExecutable> compiled_cw(new NvidiaCwExecutable(
      this, layout.storage_fingerprint, state.fingerprint_, step_plan.signature,
      cw_plan.signature, structural_step, structural_cw, source_topology,
      f_.mutation_generation, claim_executable_generation(), precision, layout.real_count,
      rows, zeroes, rhs_stages, rhs_sources,
      source_points, final_dfts, dft_omega, cw_plan.final_dfts, cw_plan.unpack,
      std::move(workspace), std::move(timestep_owner), state));
  const GraphExecutionMode requested =
      reconcile_graph_mode(graph_mode_, graph_mode_parse_valid_);
  if (configure_cw_device_graphs(*compiled_cw, state, requested)) {
    configure_cw_graph_execution(step_plan, cw_plan, *compiled_cw->timestep_, state);
    if (!compiled_cw->timestep_->graph_enabled_) clear_cw_device_graphs(*compiled_cw);
  }
  if (ordinary_refresh.advanced &&
      !refresh_source_values(ordinary_plan, ordinary, state))
    throw std::invalid_argument("NVIDIA solve_cw ordinary source targets are not reusable");
  ++state.cw_workspace_allocations_;
  ++state.executable_build_count_;
  return compiled_cw.release();
}

CwSolveResult NvidiaBackend::solve_cw(const CwSolveRequest &request, const StepPlan &step_plan,
                                      const CwPlan &cw_plan, Executable &raw_ordinary,
                                      Executable &raw_cw, BackendState &raw_state,
                                      CwSolveSession &session) {
  NvidiaBackendState &state = checked_state(raw_state);
  (void)checked_executable(raw_ordinary, state);
  nvidia::device_scope device_scope(state.device_);
  NvidiaCwExecutable *cw = dynamic_cast<NvidiaCwExecutable *>(&raw_cw);
  if (!cw || cw->owner_ != this || cw->state_token_ != state.state_token_ ||
      !cw->timestep_ || cw->timestep_->state_token_ != state.state_token_ || !cw->workspace_)
    throw std::invalid_argument("NVIDIA solve_cw received an invalid compiled artifact");
  if (cw->layout_storage_fingerprint_ != step_plan.cw_state_layout.storage_fingerprint ||
      cw->device_storage_fingerprint_ != state.fingerprint_ ||
      cw->step_plan_signature_ != step_plan.signature ||
      cw->cw_plan_signature_ != cw_plan.signature || cw->real_count_ == 0)
    throw std::logic_error("NVIDIA solve_cw executable is stale");
  NvidiaCwWorkspace &workspace = *cw->workspace_;
  if (!workspace.accommodates(cw_workspace_shape(cw->real_count_, request.L, cw->precision_),
                              cw->precision_, request.L))
    throw std::logic_error("NVIDIA solve_cw workspace is too small");
  const bool profile_only = cw_profile_mode_requested();
  const NvtxApi *profile_api = NULL;
  if (profile_only) {
    profile_api = &nvtx_api();
    if (!profile_api->available())
      throw std::logic_error("NVIDIA solve_cw NVTX preflight invariant was lost");
  }

  const size_t n = cw->real_count_;
  const size_t Ls = size_t(request.L);
  const size_t two_l = checked_product(Ls, size_t(2), "indexing solve_cw workspace");
  const size_t scalar_size = cw->precision_ == nvidia::scalar_precision::f32 ? sizeof(float)
                                                                            : sizeof(double);
  const size_t vector_bytes = checked_product(n, scalar_size, "indexing solve_cw workspace");
  char *const workspace_base = static_cast<char *>(workspace.vectors_.opaque_handle());
  const auto vector_at = [&](size_t slot) -> void * {
    return workspace_base + checked_product(slot, vector_bytes, "indexing solve_cw vector");
  };
  for (int i = 0; i <= request.L; ++i) {
    workspace.r_[i] = vector_at(size_t(i));
    workspace.u_[i] = vector_at(checked_add(checked_add(Ls, size_t(1),
                                                       "indexing solve_cw workspace"),
                                            size_t(i), "indexing solve_cw workspace"));
  }
  void *const rtilde = vector_at(checked_add(two_l, size_t(2), "indexing solve_cw workspace"));
  void *const x = vector_at(checked_add(two_l, size_t(3), "indexing solve_cw workspace"));
  void *const b = vector_at(checked_add(two_l, size_t(4), "indexing solve_cw workspace"));
  workspace.operator_applications_ = 0;
  workspace.reduction_count_ = 0;
  workspace.source_scalar_h2d_calls_ = 0;
  workspace.source_scalar_h2d_bytes_ = 0;
  workspace.pack_kernel_launches_ = 0;
  workspace.unpack_kernel_launches_ = 0;
  workspace.zero_kernel_launches_ = 0;
  workspace.rhs_source_kernel_launches_ = 0;
  workspace.reconciliation_kernel_launches_ = 0;
  workspace.vector_kernel_launches_ = 0;
  workspace.operator_kernel_launches_ = 0;
  workspace.reduction_kernel_launches_ = 0;
  workspace.timestep_kernel_launches_ = 0;
  workspace.finite_check_kernel_launches_ = 0;
  workspace.diagnostic_d2h_calls_ = 0;
  workspace.diagnostic_d2h_bytes_ = 0;
  workspace.final_dft_kernel_launches_ = 0;
  cw->graph_launch_count_ = 0;
  cw->graph_scalar_write_count_ = 0;
  cw->graph_rhs_launch_count_ = 0;
  cw->graph_unpack_launch_count_ = 0;
  cw->graph_pack_launch_count_ = 0;
  cw->graph_vector_launch_count_ = 0;
  cw->graph_reduction_launch_count_ = 0;
  cw->graph_operator_launch_count_ = 0;
  cw->graph_final_dft_launch_count_ = 0;
  cw->timestep_->graph_statistics_.scalar_write_count = 0;
  cw->timestep_->graph_statistics_.launch_count = 0;
  cw->timestep_->graph_statistics_.boundary_count = 0;
  state.cw_statistics_ = NvidiaCwStatistics();
  const nvidia::testing::transfer_accounting transfer_start =
      nvidia::testing::current_transfer_accounting();
  nvidia::testing::transfer_accounting transfer_setup_end = transfer_start;
  nvidia::testing::transfer_accounting transfer_iteration_end = transfer_start;
  size_t setup_kernel_launches = 0;
  size_t setup_source_h2d_calls = 0, setup_source_h2d_bytes = 0;
  size_t setup_scalar_d2h_calls = 0, setup_scalar_d2h_bytes = 0;
  size_t setup_diagnostic_d2h_calls = 0, setup_diagnostic_d2h_bytes = 0;
  size_t setup_operator_applications = 0, setup_reduction_count = 0;
  size_t setup_pack_kernels = 0, setup_unpack_kernels = 0;
  size_t setup_reconciliation_kernels = 0, setup_vector_kernels = 0;
  size_t setup_operator_kernels = 0, setup_reduction_kernels = 0;
  size_t setup_timestep_kernels = 0;

  const auto total_kernel_launches = [&]() {
    return workspace.pack_kernel_launches_ + workspace.unpack_kernel_launches_ +
           workspace.zero_kernel_launches_ + workspace.rhs_source_kernel_launches_ +
           workspace.reconciliation_kernel_launches_ + workspace.vector_kernel_launches_ +
           workspace.operator_kernel_launches_ + workspace.reduction_kernel_launches_ +
           workspace.timestep_kernel_launches_ + workspace.final_dft_kernel_launches_;
  };

  const std::complex<double> iomega =
      (1.0 - std::exp(std::complex<double>(0.0, -1.0) *
                      (2 * pi * request.frequency) * f_.dt)) *
      (1.0 / f_.dt);
  const auto write_graph_scalars = [&](void *output, const void *left, const void *right,
                                       double coefficient, double reduction_scale) {
    nvidia::cw_graph_scalars scalars = {};
    scalars.abi_version = nvidia::cw_graph_scalars_abi_version;
    scalars.byte_size = sizeof(scalars);
    scalars.output = output;
    scalars.x = left;
    scalars.y = right;
    scalars.coefficient = coefficient;
    scalars.reduction_scale = reduction_scale;
    scalars.dt_inverse = 1.0 / f_.dt;
    scalars.iomega_real = iomega.real();
    scalars.iomega_imaginary = iomega.imag();
    nvidia::launch_cw_graph_scalars_write(cw->graph_scalars_, scalars, *state.transfer_);
    ++cw->graph_scalar_write_count_;
  };

  const auto vector_op = [&](void *output, const void *left, const void *right, double coefficient,
                             nvidia::cw_vector_operation operation) {
    if (!output || !left ||
        ((operation == nvidia::cw_vector_operation::subtract_field ||
          operation == nvidia::cw_vector_operation::linear_f64_coefficient) && !right))
      throw std::logic_error("NVIDIA solve_cw vector graph received an invalid operand");
    if (cw->graphs_enabled_) {
      const size_t index = size_t(operation);
      if (index >= cw->vector_graphs_.size())
        throw std::logic_error("NVIDIA solve_cw vector graph is missing");
      write_graph_scalars(output, left, right, coefficient, 1.0);
      cw->vector_graphs_[index].executable.launch(*state.transfer_);
      ++cw->graph_launch_count_;
      ++cw->graph_vector_launch_count_;
    }
    else {
      nvidia::cw_vector_launch launch = {output, left, right, n, coefficient, cw->precision_,
                                         operation};
      nvidia::launch_cw_vector(launch, *state.transfer_);
    }
    ++workspace.vector_kernel_launches_;
  };
  const auto pack = [&](void *destination) {
    if (cw->graphs_enabled_) {
      if (!destination) throw std::logic_error("NVIDIA solve_cw graph pack has no destination");
      write_graph_scalars(destination, destination, NULL, 0.0, 1.0);
      cw->pack_graph_.executable.launch(*state.transfer_);
      ++cw->graph_launch_count_;
      ++cw->graph_pack_launch_count_;
      workspace.pack_kernel_launches_ += cw->rows_.size();
      if (nvidia::testing::consume_failure_for_testing(
              nvidia::testing::failure_point::cw_pack))
        throw std::runtime_error("injected NVIDIA solve_cw pack failure");
      return;
    }
    for (const nvidia::cw_state_row_launch &row : cw->rows_) {
      nvidia::launch_cw_pack(row, destination, n, *state.transfer_);
      ++workspace.pack_kernel_launches_;
      if (nvidia::testing::consume_failure_for_testing(
              nvidia::testing::failure_point::cw_pack))
        throw std::runtime_error("injected NVIDIA solve_cw pack failure");
    }
  };
  const auto unpack = [&](const void *source) {
    if (cw->graphs_enabled_) {
      if (!source) throw std::logic_error("NVIDIA solve_cw graph unpack has no source");
      write_graph_scalars(NULL, source, NULL, 0.0, 1.0);
      cw->operator_prelude_graph_.executable.launch(*state.transfer_);
      ++cw->graph_launch_count_;
      ++cw->graph_unpack_launch_count_;
      workspace.unpack_kernel_launches_ += cw->rows_.size();
      workspace.reconciliation_kernel_launches_ +=
          cw->graph_operator_reconciliation_kernel_count_;
      if (nvidia::testing::consume_failure_for_testing(
              nvidia::testing::failure_point::cw_unpack))
        throw std::runtime_error("injected NVIDIA solve_cw unpack failure");
      return;
    }
    for (const nvidia::cw_state_row_launch &row : cw->rows_) {
      nvidia::launch_cw_unpack(row, source, n, *state.transfer_);
      ++workspace.unpack_kernel_launches_;
      if (nvidia::testing::consume_failure_for_testing(
              nvidia::testing::failure_point::cw_unpack))
        throw std::runtime_error("injected NVIDIA solve_cw unpack failure");
    }
  };
  const auto upload_source_scalars = [&](double offset) {
    f_.step_source_times[0] = cw_source_time(f_.t, f_.dt, 0.0);
    f_.step_source_times[1] = cw_source_time(f_.t, f_.dt, 0.5);
    f_.step_source_times[2] = cw_source_time(f_.t, f_.dt, 1.0);
    evaluate_supported_source_scalars(f_, offset);
    const SourcePlan &sources = f_.descriptors->sources;
    if (sources.scalars.size() != cw->timestep_->source_scalar_count_)
      throw std::logic_error("NVIDIA solve_cw source scalar count changed after compilation");
    nvidia::source_scalar *staging =
        static_cast<nvidia::source_scalar *>(cw->timestep_->source_staging_.data());
    for (size_t i = 0; i < sources.scalars.size(); ++i) {
      staging[i].current_real = sources.scalars[i].current.real();
      staging[i].current_imag = sources.scalars[i].current.imag();
      staging[i].dipole_real = sources.scalars[i].dipole.real();
      staging[i].dipole_imag = sources.scalars[i].dipole.imag();
    }
    const size_t upload_bytes =
        checked_product(sources.scalars.size(), sizeof(nvidia::source_scalar),
                        "uploading solve_cw source scalars");
    nvidia::copy_host_to_device_async(
        cw->timestep_->source_scalars_, 0, staging,
        upload_bytes,
        *state.transfer_);
    ++workspace.source_scalar_h2d_calls_;
    workspace.source_scalar_h2d_bytes_ += upload_bytes;
    /* The next RHS stage reuses this bounded pinned block after invoking more
       host callbacks. Complete the compact upload before that overwrite. */
    state.transfer_->synchronize();
  };
  const auto execute_reconciliation = [&](uint32_t operation_index, bool skip_w_components) {
    if (operation_index >= cw->timestep_->operations_.size())
      throw std::logic_error("NVIDIA solve_cw reconciliation operation is out of range");
    const NvidiaCompiledOperation &op = cw->timestep_->operations_[operation_index];
    if (op.kind == OpKind::transfer_halo) {
      for (size_t i = op.first; i < op.first + op.count; ++i) {
        nvidia::launch_zero(cw->timestep_->zero_updates_[i], *state.transfer_);
        ++workspace.reconciliation_kernel_launches_;
      }
      for (size_t i = op.halo_first; i < op.halo_first + op.halo_count; ++i) {
        nvidia::launch_halo_gather(cw->timestep_->halo_plans_[i].gather,
                                   cw->timestep_->halo_gathers_.opaque_handle(),
                                   cw->timestep_->halo_scratch_.opaque_handle(), *state.transfer_);
        ++workspace.reconciliation_kernel_launches_;
      }
      for (size_t i = op.halo_first; i < op.halo_first + op.halo_count; ++i) {
        nvidia::launch_halo_scatter(cw->timestep_->halo_plans_[i].scatter,
                                    cw->timestep_->halo_scatters_.opaque_handle(),
                                    cw->timestep_->halo_scratch_.opaque_handle(), *state.transfer_);
        ++workspace.reconciliation_kernel_launches_;
      }
      return;
    }
    if (op.kind == OpKind::update_eh) {
      if (op.copy_count || op.subtraction_count || op.source_count)
        throw std::logic_error("NVIDIA solve_cw constitutive operation retained source state");
      for (size_t i = op.first; i < op.first + op.count; ++i) {
        const nvidia::constitutive_launch &update = cw->timestep_->constitutive_updates_[i];
        /* array_to_fields alone calls update_eh(E_stuff, true): the packed CW
           vector already carries both W and its paired primary field.  RHS
           reconciliation and the full T_cw step use the ordinary false mode
           and must retain these rows. */
        if (skip_w_components && update.target_w) continue;
        nvidia::launch_constitutive(update, *state.transfer_);
        ++workspace.reconciliation_kernel_launches_;
      }
      return;
    }
    throw std::logic_error("NVIDIA solve_cw reconciliation reference has the wrong kind");
  };
  const auto apply_operator = [&](const void *input, void *output) {
    const size_t reconciliation_before = workspace.reconciliation_kernel_launches_;
    unpack(input);
    if (!cw->graphs_enabled_)
      for (int i = 0; i < 3; ++i)
        execute_reconciliation(cw->unpack_operations_[i],
                               i == 1 && cw->unpack_skip_w_components_);
    const size_t reconciliation_launches =
        workspace.reconciliation_kernel_launches_ - reconciliation_before;
    if (!state.cw_statistics_.reconciliation_kernel_launches_per_operator)
      state.cw_statistics_.reconciliation_kernel_launches_per_operator =
          reconciliation_launches;
    else if (state.cw_statistics_.reconciliation_kernel_launches_per_operator !=
             reconciliation_launches)
      throw std::logic_error("NVIDIA solve_cw reconciliation shape changed during solve");
    const size_t timestep_before = workspace.timestep_kernel_launches_;
    advance(*cw->timestep_, state, 1);
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::cw_timestep))
      throw std::runtime_error("injected NVIDIA solve_cw timestep failure");
    const size_t timestep_launches = workspace.timestep_kernel_launches_ - timestep_before;
    if (!state.cw_statistics_.timestep_kernel_launches_per_operator)
      state.cw_statistics_.timestep_kernel_launches_per_operator = timestep_launches;
    else if (state.cw_statistics_.timestep_kernel_launches_per_operator != timestep_launches)
      throw std::logic_error("NVIDIA solve_cw operator launch shape changed during solve");
    pack(output);
    if (cw->graphs_enabled_) {
      write_graph_scalars(output, input, NULL, 0.0, 1.0);
      cw->operator_finalize_graph_.executable.launch(*state.transfer_);
      ++cw->graph_launch_count_;
      ++cw->graph_operator_launch_count_;
    }
    else {
      nvidia::cw_operator_launch launch = {output, output, input, n, 1.0 / f_.dt,
                                           iomega.real(), iomega.imag(), cw->precision_};
      nvidia::launch_cw_operator_finalize(launch, *state.transfer_);
    }
    ++workspace.operator_kernel_launches_;
    ++workspace.operator_applications_;
  };
  const size_t reduction_blocks = std::min<size_t>(128, (n + 255) / 256);
  const auto reduce = [&](const void *left, const void *right, double scale, int operation) {
    if (!left || (operation == 0 && !right) || operation < 0 || operation > 2 ||
        (operation == 2 && (!std::isfinite(scale) || scale <= 0.0)))
      throw std::logic_error("NVIDIA solve_cw reduction graph received invalid operands");
    if (cw->graphs_enabled_) {
      if (size_t(operation) >= cw->reduction_graphs_.size())
        throw std::logic_error("NVIDIA solve_cw reduction graph is missing");
      write_graph_scalars(NULL, left, right, 0.0, scale);
      cw->reduction_graphs_[size_t(operation)].executable.launch(*state.transfer_);
      ++cw->graph_launch_count_;
      ++cw->graph_reduction_launch_count_;
    }
    else {
      nvidia::cw_reduction_launch launch = {
          left, right, workspace.reduction_partials_.opaque_handle(),
          workspace.reduction_result_.opaque_handle(), n, reduction_blocks, scale, cw->precision_};
      if (operation == 0) nvidia::launch_cw_dot(launch, *state.transfer_);
      else if (operation == 1) nvidia::launch_cw_max_abs(launch, *state.transfer_);
      else nvidia::launch_cw_scaled_norm_sum(launch, *state.transfer_);
    }
    workspace.reduction_kernel_launches_ += 2;
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::cw_reduction))
      throw std::runtime_error("injected NVIDIA solve_cw reduction failure");
    nvidia::copy_device_to_host_async(workspace.reduction_host_.data(),
                                      workspace.reduction_result_, 0, sizeof(double),
                                      *state.transfer_);
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::cw_scalar_copy))
      throw std::runtime_error("injected NVIDIA solve_cw scalar-copy failure");
    state.transfer_->synchronize();
    double result = 0.0;
    memcpy(&result, workspace.reduction_host_.data(), sizeof(result));
    if (!std::isfinite(result) || (operation != 0 && result < 0.0))
      throw std::runtime_error("NVIDIA solve_cw reduction returned a nonfinite result");
    ++workspace.reduction_count_;
    ++state.cw_statistics_.scalar_device_to_host_calls;
    state.cw_statistics_.scalar_device_to_host_bytes += sizeof(double);
    return result;
  };
  const auto dot = [&](const void *left, const void *right) {
    return reduce(left, right, 1.0, 0);
  };
  const auto norm = [&](const void *values) {
    const double maximum = reduce(values, NULL, 1.0, 1);
    if (maximum == 0.0) return 0.0;
    const double sum = reduce(values, NULL, 1.0 / maximum, 2);
    const double result = maximum * std::sqrt(sum);
    if (!std::isfinite(result))
      throw std::runtime_error("NVIDIA solve_cw norm is nonfinite");
    return result;
  };

  double bnrm = 0.0, rtilde_norm = 0.0;
  {
    NvtxRange setup_range(profile_api, "meep.solve_cw.setup");
    /* Match the legacy entry sequence: one source-suppressed timestep produces
       the initial guess before RHS construction. */
    advance(*cw->timestep_, state, 1);
    pack(x);
    if (cw->graphs_enabled_) {
      cw->rhs_zero_graph_.executable.launch(*state.transfer_);
      ++cw->graph_launch_count_;
      ++cw->graph_rhs_launch_count_;
      workspace.zero_kernel_launches_ += cw->zeroes_.size();
    }
    else
      for (const nvidia::cw_zero_launch &zero : cw->zeroes_) {
        nvidia::launch_cw_zero(zero, *state.transfer_);
        ++workspace.zero_kernel_launches_;
      }
    for (size_t stage_index = 0; stage_index < cw->rhs_stages_.size(); ++stage_index) {
      const NvidiaCwExecutable::Stage &stage = cw->rhs_stages_[stage_index];
      upload_source_scalars(stage.source_time_offset);
      if (cw->graphs_enabled_) {
        if (stage_index >= cw->rhs_stage_graphs_.size() ||
            stage_index >= cw->rhs_stage_reconciliation_kernel_counts_.size())
          throw std::logic_error("NVIDIA solve_cw RHS graph schedule is incomplete");
        cw->rhs_stage_graphs_[stage_index].executable.launch(*state.transfer_);
        ++cw->graph_launch_count_;
        ++cw->graph_rhs_launch_count_;
        workspace.rhs_source_kernel_launches_ += stage.source_count;
        workspace.reconciliation_kernel_launches_ +=
            cw->rhs_stage_reconciliation_kernel_counts_[stage_index];
      }
      else {
        for (size_t i = stage.source_first; i < stage.source_first + stage.source_count; ++i) {
          nvidia::launch_cw_source_batch(cw->rhs_sources_[i],
                                         cw->timestep_->source_scalars_.opaque_handle(),
                                         *state.transfer_);
          ++workspace.rhs_source_kernel_launches_;
        }
        execute_reconciliation(stage.boundary_operation, false);
        execute_reconciliation(stage.constitutive_operation, false);
      }
    }
    pack(b);
    vector_op(b, b, NULL, -1.0 / f_.dt,
              nvidia::cw_vector_operation::scale_field_coefficient);
    bnrm = norm(b);
    if (bnrm == 0.0) throw std::runtime_error("zero current amplitudes in NVIDIA solve_cw");

    apply_operator(x, workspace.r_[0]);
    vector_op(workspace.r_[0], b, workspace.r_[0], 0.0,
              nvidia::cw_vector_operation::subtract_field);
    vector_op(rtilde, workspace.r_[0], NULL, 0.0, nvidia::cw_vector_operation::copy);
    rtilde_norm = norm(rtilde);
    if (rtilde_norm != 0.0)
      vector_op(rtilde, rtilde, NULL, 1.0 / rtilde_norm,
                nvidia::cw_vector_operation::scale_f64_coefficient);
    nvidia::fill_byte_async(workspace.vectors_,
                            checked_product(checked_add(Ls, size_t(1), "indexing solve_cw u0"),
                                            vector_bytes,
                                            "zeroing solve_cw u0"),
                            0, vector_bytes, *state.transfer_);
    if (profile_only) state.transfer_->synchronize();
  }
  transfer_setup_end = nvidia::testing::current_transfer_accounting();
  setup_kernel_launches = total_kernel_launches();
  setup_source_h2d_calls = workspace.source_scalar_h2d_calls_;
  setup_source_h2d_bytes = workspace.source_scalar_h2d_bytes_;
  setup_scalar_d2h_calls = state.cw_statistics_.scalar_device_to_host_calls;
  setup_scalar_d2h_bytes = state.cw_statistics_.scalar_device_to_host_bytes;
  setup_diagnostic_d2h_calls = workspace.diagnostic_d2h_calls_;
  setup_diagnostic_d2h_bytes = workspace.diagnostic_d2h_bytes_;
  setup_operator_applications = workspace.operator_applications_;
  setup_reduction_count = workspace.reduction_count_;
  setup_pack_kernels = workspace.pack_kernel_launches_;
  setup_unpack_kernels = workspace.unpack_kernel_launches_;
  setup_reconciliation_kernels = workspace.reconciliation_kernel_launches_;
  setup_vector_kernels = workspace.vector_kernel_launches_;
  setup_operator_kernels = workspace.operator_kernel_launches_;
  setup_reduction_kernels = workspace.reduction_kernel_launches_;
  setup_timestep_kernels = workspace.timestep_kernel_launches_;

  double rho = 1.0, alpha = 0.0, omega = 1.0;
  double residual = rtilde_norm;
  int iterations = 0;
  CwSolveStatus status = CwSolveStatus::converged;
  const double break_tolerance = 1e-30;
  while (true) {
    if (profile_only && iterations == 3) {
      status = CwSolveStatus::not_converged;
      break;
    }
    char iteration_range_name[64];
    std::snprintf(iteration_range_name, sizeof(iteration_range_name),
                  "meep.solve_cw.iteration.%d", iterations + 1);
    NvtxRange iteration_range(profile_api, iteration_range_name);
    residual = norm(workspace.r_[0]);
    if (!profile_only && residual <= request.tolerance * bnrm) break;
    ++iterations;
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::cw_breakdown))
      rho = 0.0;
    rho = -omega * rho;
    bool breakdown = false;
    for (int j = 0; j < request.L; ++j) {
      if (std::fabs(rho) < break_tolerance) {
        breakdown = true;
        break;
      }
      const double rho1 = dot(workspace.r_[j], rtilde);
      const double beta = alpha * rho1 / rho;
      if (!std::isfinite(beta)) throw std::runtime_error("NVIDIA solve_cw beta is nonfinite");
      rho = rho1;
      for (int i = 0; i <= j; ++i)
        vector_op(workspace.u_[i], workspace.r_[i], workspace.u_[i], -beta,
                  nvidia::cw_vector_operation::linear_f64_coefficient);
      apply_operator(workspace.u_[j], workspace.u_[j + 1]);
      const double denominator = dot(workspace.u_[j + 1], rtilde);
      alpha = rho / denominator;
      if (!std::isfinite(alpha)) throw std::runtime_error("NVIDIA solve_cw alpha is nonfinite");
      for (int i = 0; i <= j; ++i)
        vector_op(workspace.r_[i], workspace.r_[i], workspace.u_[i + 1], -alpha,
                  nvidia::cw_vector_operation::linear_f64_coefficient);
      apply_operator(workspace.r_[j], workspace.r_[j + 1]);
      vector_op(x, x, workspace.u_[0], alpha,
                nvidia::cw_vector_operation::linear_f64_coefficient);
    }
    if (breakdown) {
      status = CwSolveStatus::breakdown;
      if (profile_only) state.transfer_->synchronize();
      break;
    }
    for (int j = 1; j <= request.L; ++j) {
      for (int i = 1; i < j; ++i) {
        const size_t ij = size_t(j - 1) * size_t(request.L) + size_t(i - 1);
        workspace.tau_[ij] = dot(workspace.r_[j], workspace.r_[i]) / workspace.sigma_[i];
        if (!std::isfinite(workspace.tau_[ij]))
          throw std::runtime_error("NVIDIA solve_cw tau is nonfinite");
        vector_op(workspace.r_[j], workspace.r_[j], workspace.r_[i], -workspace.tau_[ij],
                  nvidia::cw_vector_operation::linear_f64_coefficient);
      }
      if (breakdown) break;
      workspace.sigma_[j] = dot(workspace.r_[j], workspace.r_[j]);
      workspace.gamma_p_[j] = dot(workspace.r_[0], workspace.r_[j]) / workspace.sigma_[j];
      if (!std::isfinite(workspace.gamma_p_[j]))
        throw std::runtime_error("NVIDIA solve_cw gamma-prime is nonfinite");
    }
    if (breakdown) {
      status = CwSolveStatus::breakdown;
      if (profile_only) state.transfer_->synchronize();
      break;
    }
    omega = workspace.gamma_[request.L] = workspace.gamma_p_[request.L];
    for (int j = request.L - 1; j >= 1; --j) {
      workspace.gamma_[j] = workspace.gamma_p_[j];
      for (int i = j + 1; i <= request.L; ++i)
        workspace.gamma_[j] -=
            workspace.tau_[size_t(i - 1) * size_t(request.L) + size_t(j - 1)] *
            workspace.gamma_[i];
    }
    for (int j = 1; j < request.L; ++j) {
      workspace.gamma_pp_[j] = workspace.gamma_[j + 1];
      for (int i = j + 1; i < request.L; ++i)
        workspace.gamma_pp_[j] +=
            workspace.tau_[size_t(i - 1) * size_t(request.L) + size_t(j - 1)] *
            workspace.gamma_[i + 1];
    }
    vector_op(x, x, workspace.r_[0], workspace.gamma_[1],
              nvidia::cw_vector_operation::linear_f64_coefficient);
    vector_op(workspace.r_[0], workspace.r_[0], workspace.r_[request.L],
              -workspace.gamma_p_[request.L],
              nvidia::cw_vector_operation::linear_f64_coefficient);
    vector_op(workspace.u_[0], workspace.u_[0], workspace.u_[request.L],
              -workspace.gamma_[request.L],
              nvidia::cw_vector_operation::linear_f64_coefficient);
    for (int j = 1; j < request.L; ++j) {
      vector_op(x, x, workspace.r_[j], workspace.gamma_pp_[j],
                nvidia::cw_vector_operation::linear_f64_coefficient);
      vector_op(workspace.r_[0], workspace.r_[0], workspace.r_[j],
                -workspace.gamma_p_[j], nvidia::cw_vector_operation::linear_f64_coefficient);
      vector_op(workspace.u_[0], workspace.u_[0], workspace.u_[j], -workspace.gamma_[j],
                nvidia::cw_vector_operation::linear_f64_coefficient);
    }
    if (iterations == request.maxiters) {
      status = CwSolveStatus::not_converged;
      if (profile_only) state.transfer_->synchronize();
      break;
    }
    if (profile_only) state.transfer_->synchronize();
  }

  transfer_iteration_end = nvidia::testing::current_transfer_accounting();
  const size_t iteration_kernel_launches = total_kernel_launches() - setup_kernel_launches;
  const size_t iteration_source_h2d_calls =
      workspace.source_scalar_h2d_calls_ - setup_source_h2d_calls;
  const size_t iteration_source_h2d_bytes =
      workspace.source_scalar_h2d_bytes_ - setup_source_h2d_bytes;
  const size_t iteration_scalar_d2h_calls =
      state.cw_statistics_.scalar_device_to_host_calls - setup_scalar_d2h_calls;
  const size_t iteration_scalar_d2h_bytes =
      state.cw_statistics_.scalar_device_to_host_bytes - setup_scalar_d2h_bytes;
  const size_t iteration_diagnostic_d2h_calls =
      workspace.diagnostic_d2h_calls_ - setup_diagnostic_d2h_calls;
  const size_t iteration_diagnostic_d2h_bytes =
      workspace.diagnostic_d2h_bytes_ - setup_diagnostic_d2h_bytes;
  const size_t iteration_operator_applications =
      workspace.operator_applications_ - setup_operator_applications;
  const size_t iteration_reduction_count = workspace.reduction_count_ - setup_reduction_count;
  const size_t iteration_pack_kernels = workspace.pack_kernel_launches_ - setup_pack_kernels;
  const size_t iteration_unpack_kernels = workspace.unpack_kernel_launches_ - setup_unpack_kernels;
  const size_t iteration_reconciliation_kernels =
      workspace.reconciliation_kernel_launches_ - setup_reconciliation_kernels;
  const size_t iteration_vector_kernels = workspace.vector_kernel_launches_ - setup_vector_kernels;
  const size_t iteration_operator_kernels =
      workspace.operator_kernel_launches_ - setup_operator_kernels;
  const size_t iteration_reduction_kernels =
      workspace.reduction_kernel_launches_ - setup_reduction_kernels;
  const size_t iteration_timestep_kernels =
      workspace.timestep_kernel_launches_ - setup_timestep_kernels;

  const double recursive_relative_residual = residual / bnrm;
  double true_relative_residual = 0.0;
  {
    NvtxRange true_residual_range(profile_api, "meep.solve_cw.true_residual");
    const int verification_t = f_.t;
    state.cw_skip_source_evaluation_ = true;
    try {
      apply_operator(x, workspace.r_[0]);
    }
    catch (...) {
      state.cw_skip_source_evaluation_ = false;
      f_.t = verification_t;
      throw;
    }
    state.cw_skip_source_evaluation_ = false;
    f_.t = verification_t;
    vector_op(workspace.r_[0], b, workspace.r_[0], 0.0,
              nvidia::cw_vector_operation::subtract_field);
    true_relative_residual = norm(workspace.r_[0]) / bnrm;
  }
  const double reduction_error_bound =
      64.0 * (cw->precision_ == nvidia::scalar_precision::f32
                  ? double(std::numeric_limits<float>::epsilon())
                  : std::numeric_limits<double>::epsilon()) *
      std::sqrt(double(n));
  if (status == CwSolveStatus::converged &&
      true_relative_residual > std::max(5.0 * request.tolerance, reduction_error_bound))
    status = CwSolveStatus::not_converged;
  {
    NvtxRange install_range(profile_api, "meep.solve_cw.install_and_sync");
    unpack(x);
    if (!cw->graphs_enabled_)
      for (int i = 0; i < 3; ++i)
        execute_reconciliation(cw->unpack_operations_[i],
                               i == 1 && cw->unpack_skip_w_components_);
    advance(*cw->timestep_, state, 1);
    session.restore_before_final_dft();
    if (profile_only) state.transfer_->synchronize();
  }
  size_t final_dft_kernel_launches = 0;
  {
    NvtxRange dft_range(profile_api, "meep.solve_cw.final_dft");
    if (cw->graphs_enabled_ && !cw->final_dfts_.empty()) {
      StepScalars scalars = {};
      scalars.abi_version = step_scalars_abi_version;
      scalars.byte_size = sizeof(scalars);
      scalars.entry_timestep = f_.t;
      scalars.post_increment_timestep = int64_t(f_.t) + 1;
      scalars.dft_timestep = f_.t;
      scalars.source_times[1] = f_.time() - 0.5 * f_.dt;
      scalars.source_times[2] = f_.time();
      scalars.graph_variant = uint32_t(GraphVariantKind::cw_operator);
      for (size_t i = 0; i < cw->final_dfts_.size(); ++i) {
        const uint32_t slot = cw->final_dft_due_slots_[i];
        if ((f_.t % cw->final_dfts_[i].decimation_factor) == 0) {
          scalars.predicate_words[slot / 64] |= uint64_t(1) << (slot % 64);
          ++workspace.final_dft_kernel_launches_;
          ++final_dft_kernel_launches;
        }
      }
      nvidia::launch_step_scalars_write(cw->final_dft_scalars_, scalars, *state.transfer_);
      ++cw->graph_scalar_write_count_;
      cw->final_dft_graph_.executable.launch(*state.transfer_);
      ++cw->graph_launch_count_;
      ++cw->graph_final_dft_launch_count_;
    }
    else {
      for (const nvidia::dft_launch &dft : cw->final_dfts_) {
        if ((f_.t % dft.decimation_factor) != 0) continue;
        const double sample_time = dft.magnetic ? f_.time() - 0.5 * f_.dt : f_.time();
        nvidia::launch_dft(dft, sample_time, *state.transfer_);
        ++workspace.final_dft_kernel_launches_;
        ++final_dft_kernel_launches;
      }
    }
    state.transfer_->synchronize();
  }
  state.device_authoritative_ = true;
  const nvidia::testing::transfer_accounting transfer_end =
      nvidia::testing::current_transfer_accounting();

  const auto counter_delta = [](size_t end, size_t begin, const char *what) {
    if (end < begin) throw std::logic_error(std::string("NVIDIA solve_cw ") + what +
                                            " accounting moved backwards");
    return end - begin;
  };
  const size_t setup_total_h2d_calls =
      counter_delta(transfer_setup_end.host_to_device_calls, transfer_start.host_to_device_calls,
                    "setup H2D");
  const size_t setup_total_h2d_bytes =
      counter_delta(transfer_setup_end.host_to_device_bytes, transfer_start.host_to_device_bytes,
                    "setup H2D byte");
  const size_t setup_total_d2h_calls =
      counter_delta(transfer_setup_end.device_to_host_calls, transfer_start.device_to_host_calls,
                    "setup D2H");
  const size_t setup_total_d2h_bytes =
      counter_delta(transfer_setup_end.device_to_host_bytes, transfer_start.device_to_host_bytes,
                    "setup D2H byte");
  const size_t iteration_total_h2d_calls = counter_delta(
      transfer_iteration_end.host_to_device_calls, transfer_setup_end.host_to_device_calls,
      "iteration H2D");
  const size_t iteration_total_h2d_bytes = counter_delta(
      transfer_iteration_end.host_to_device_bytes, transfer_setup_end.host_to_device_bytes,
      "iteration H2D byte");
  const size_t iteration_total_d2h_calls = counter_delta(
      transfer_iteration_end.device_to_host_calls, transfer_setup_end.device_to_host_calls,
      "iteration D2H");
  const size_t iteration_total_d2h_bytes = counter_delta(
      transfer_iteration_end.device_to_host_bytes, transfer_setup_end.device_to_host_bytes,
      "iteration D2H byte");
  const size_t final_total_h2d_calls = counter_delta(
      transfer_end.host_to_device_calls, transfer_iteration_end.host_to_device_calls,
      "final H2D");
  const size_t final_total_h2d_bytes = counter_delta(
      transfer_end.host_to_device_bytes, transfer_iteration_end.host_to_device_bytes,
      "final H2D byte");
  const size_t final_total_d2h_calls = counter_delta(
      transfer_end.device_to_host_calls, transfer_iteration_end.device_to_host_calls,
      "final D2H");
  const size_t final_total_d2h_bytes = counter_delta(
      transfer_end.device_to_host_bytes, transfer_iteration_end.device_to_host_bytes,
      "final D2H byte");
  const size_t final_source_h2d_calls =
      workspace.source_scalar_h2d_calls_ - setup_source_h2d_calls -
      iteration_source_h2d_calls;
  const size_t final_source_h2d_bytes =
      workspace.source_scalar_h2d_bytes_ - setup_source_h2d_bytes -
      iteration_source_h2d_bytes;
  const size_t final_scalar_d2h_calls =
      state.cw_statistics_.scalar_device_to_host_calls - setup_scalar_d2h_calls -
      iteration_scalar_d2h_calls;
  const size_t final_scalar_d2h_bytes =
      state.cw_statistics_.scalar_device_to_host_bytes - setup_scalar_d2h_bytes -
      iteration_scalar_d2h_bytes;
  const size_t final_diagnostic_d2h_calls =
      workspace.diagnostic_d2h_calls_ - setup_diagnostic_d2h_calls -
      iteration_diagnostic_d2h_calls;
  const size_t final_diagnostic_d2h_bytes =
      workspace.diagnostic_d2h_bytes_ - setup_diagnostic_d2h_bytes -
      iteration_diagnostic_d2h_bytes;
  if (setup_total_h2d_calls < setup_source_h2d_calls ||
      setup_total_h2d_bytes < setup_source_h2d_bytes ||
      iteration_total_h2d_calls < iteration_source_h2d_calls ||
      iteration_total_h2d_bytes < iteration_source_h2d_bytes ||
      final_total_h2d_calls < final_source_h2d_calls ||
      final_total_h2d_bytes < final_source_h2d_bytes ||
      setup_total_d2h_calls < setup_scalar_d2h_calls + setup_diagnostic_d2h_calls ||
      setup_total_d2h_bytes < setup_scalar_d2h_bytes + setup_diagnostic_d2h_bytes ||
      iteration_total_d2h_calls <
          iteration_scalar_d2h_calls + iteration_diagnostic_d2h_calls ||
      iteration_total_d2h_bytes <
          iteration_scalar_d2h_bytes + iteration_diagnostic_d2h_bytes ||
      final_total_d2h_calls < final_scalar_d2h_calls + final_diagnostic_d2h_calls ||
      final_total_d2h_bytes < final_scalar_d2h_bytes + final_diagnostic_d2h_bytes)
    throw std::logic_error("NVIDIA solve_cw compact transfer accounting is inconsistent");

  CwSolveResult result;
  result.status = status;
  result.iterations = iterations;
  result.operator_applications = workspace.operator_applications_;
  result.recursive_relative_residual = recursive_relative_residual;
  result.true_relative_residual = true_relative_residual;
  state.cw_statistics_.result = result;
  state.cw_statistics_.reduction_count = workspace.reduction_count_;
  state.cw_statistics_.source_scalar_host_to_device_calls =
      workspace.source_scalar_h2d_calls_;
  state.cw_statistics_.source_scalar_host_to_device_bytes =
      workspace.source_scalar_h2d_bytes_;
  state.cw_statistics_.vector_host_to_device_bytes =
      setup_total_h2d_bytes - setup_source_h2d_bytes +
      iteration_total_h2d_bytes - iteration_source_h2d_bytes +
      final_total_h2d_bytes - final_source_h2d_bytes;
  state.cw_statistics_.vector_device_to_host_bytes =
      setup_total_d2h_bytes - setup_scalar_d2h_bytes - setup_diagnostic_d2h_bytes +
      iteration_total_d2h_bytes - iteration_scalar_d2h_bytes -
      iteration_diagnostic_d2h_bytes + final_total_d2h_bytes - final_scalar_d2h_bytes -
      final_diagnostic_d2h_bytes;
  state.cw_statistics_.setup_scalar_device_to_host_calls = setup_scalar_d2h_calls;
  state.cw_statistics_.setup_scalar_device_to_host_bytes = setup_scalar_d2h_bytes;
  state.cw_statistics_.setup_source_scalar_host_to_device_calls = setup_source_h2d_calls;
  state.cw_statistics_.setup_source_scalar_host_to_device_bytes = setup_source_h2d_bytes;
  state.cw_statistics_.iteration_scalar_device_to_host_calls = iteration_scalar_d2h_calls;
  state.cw_statistics_.iteration_scalar_device_to_host_bytes = iteration_scalar_d2h_bytes;
  state.cw_statistics_.iteration_source_scalar_host_to_device_calls =
      iteration_source_h2d_calls;
  state.cw_statistics_.iteration_source_scalar_host_to_device_bytes =
      iteration_source_h2d_bytes;
  state.cw_statistics_.final_scalar_device_to_host_calls = final_scalar_d2h_calls;
  state.cw_statistics_.final_scalar_device_to_host_bytes = final_scalar_d2h_bytes;
  state.cw_statistics_.final_source_scalar_host_to_device_calls = final_source_h2d_calls;
  state.cw_statistics_.final_source_scalar_host_to_device_bytes = final_source_h2d_bytes;
  state.cw_statistics_.diagnostic_device_to_host_calls = workspace.diagnostic_d2h_calls_;
  state.cw_statistics_.diagnostic_device_to_host_bytes = workspace.diagnostic_d2h_bytes_;
  state.cw_statistics_.pack_kernel_launches = workspace.pack_kernel_launches_;
  state.cw_statistics_.unpack_kernel_launches = workspace.unpack_kernel_launches_;
  state.cw_statistics_.zero_kernel_launches = workspace.zero_kernel_launches_;
  state.cw_statistics_.rhs_source_kernel_launches = workspace.rhs_source_kernel_launches_;
  state.cw_statistics_.reconciliation_kernel_launches =
      workspace.reconciliation_kernel_launches_;
  state.cw_statistics_.vector_kernel_launches = workspace.vector_kernel_launches_;
  state.cw_statistics_.operator_kernel_launches = workspace.operator_kernel_launches_;
  state.cw_statistics_.reduction_kernel_launches = workspace.reduction_kernel_launches_;
  state.cw_statistics_.timestep_kernel_launches = workspace.timestep_kernel_launches_;
  state.cw_statistics_.finite_check_kernel_launches = workspace.finite_check_kernel_launches_;
  state.cw_statistics_.final_dft_kernel_launches = final_dft_kernel_launches;
  state.cw_statistics_.kernel_launches = total_kernel_launches();
  state.cw_statistics_.setup_kernel_launches = setup_kernel_launches;
  state.cw_statistics_.iteration_kernel_launches = iteration_kernel_launches;
  state.cw_statistics_.final_kernel_launches = total_kernel_launches() -
                                               setup_kernel_launches -
                                               iteration_kernel_launches;
  state.cw_statistics_.iteration_operator_applications = iteration_operator_applications;
  state.cw_statistics_.iteration_reduction_count = iteration_reduction_count;
  state.cw_statistics_.iteration_pack_kernel_launches = iteration_pack_kernels;
  state.cw_statistics_.iteration_unpack_kernel_launches = iteration_unpack_kernels;
  state.cw_statistics_.iteration_reconciliation_kernel_launches =
      iteration_reconciliation_kernels;
  state.cw_statistics_.iteration_vector_kernel_launches = iteration_vector_kernels;
  state.cw_statistics_.iteration_operator_kernel_launches = iteration_operator_kernels;
  state.cw_statistics_.iteration_reduction_kernel_launches = iteration_reduction_kernels;
  state.cw_statistics_.iteration_timestep_kernel_launches = iteration_timestep_kernels;
  state.cw_statistics_.workspace_capacity_bytes = workspace.shape_.vector_bytes +
                                                  workspace.shape_.reduction_partial_bytes +
                                                  2 * sizeof(double);
  state.cw_statistics_.workspace_allocations = state.cw_workspace_allocations_;
  state.cw_statistics_.graph_enabled =
      cw->graphs_enabled_ && cw->timestep_->graph_enabled_;
  state.cw_statistics_.graph_count =
      cw->graph_capture_count_ + cw->timestep_->graph_segments_.size();
  state.cw_statistics_.graph_capture_count =
      cw->graph_capture_count_ + cw->timestep_->graph_statistics_.capture_count;
  state.cw_statistics_.graph_instantiate_count =
      cw->graph_instantiate_count_ + cw->timestep_->graph_statistics_.instantiate_count;
  state.cw_statistics_.graph_scalar_write_count =
      cw->graph_scalar_write_count_ + cw->timestep_->graph_statistics_.scalar_write_count;
  state.cw_statistics_.graph_launch_count =
      cw->graph_launch_count_ + cw->timestep_->graph_statistics_.launch_count;
  state.cw_statistics_.graph_rhs_launch_count = cw->graph_rhs_launch_count_;
  state.cw_statistics_.graph_unpack_launch_count = cw->graph_unpack_launch_count_;
  state.cw_statistics_.graph_pack_launch_count = cw->graph_pack_launch_count_;
  state.cw_statistics_.graph_vector_launch_count = cw->graph_vector_launch_count_;
  state.cw_statistics_.graph_reduction_launch_count = cw->graph_reduction_launch_count_;
  state.cw_statistics_.graph_operator_launch_count = cw->graph_operator_launch_count_;
  state.cw_statistics_.graph_final_dft_launch_count = cw->graph_final_dft_launch_count_;
  state.cw_statistics_.valid = true;
  return result;
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
  if (executable.graph_enabled_) {
    StepScalars scalars = {};
    scalars.abi_version = step_scalars_abi_version;
    scalars.byte_size = sizeof(StepScalars);
    scalars.entry_timestep = f_.t;
    scalars.post_increment_timestep = int64_t(f_.t) + 1;
    scalars.noisy_counter_time = uint64_t(f_.t);
    for (int i = 0; i < 3; ++i) scalars.source_times[i] = f_.step_source_times[i];
    scalars.batch_count = 1;
    scalars.dft_timestep = int64_t(f_.t) + 1;
    scalars.graph_variant = uint32_t(GraphVariantKind::magnetic_half_step);
    nvidia::launch_step_scalars_write(executable.magnetic_step_scalars_, scalars,
                                      *state.transfer_);
    ++executable.graph_statistics_.magnetic_scalar_write_count;
    ++state.graph_statistics_.magnetic_scalar_write_count;
    for (const GraphScheduleEntry &entry : executable.magnetic_half_graph_program_.schedule) {
      if (entry.kind == GraphScheduleKind::segment) {
        if (entry.index >= executable.magnetic_half_graph_segments_.size())
          throw std::logic_error("NVIDIA magnetic graph segment is out of range");
        executable.magnetic_half_graph_segments_[entry.index].executable.launch(
            *state.transfer_);
        ++executable.graph_statistics_.magnetic_launch_count;
        ++state.graph_statistics_.magnetic_launch_count;
        continue;
      }
      if (entry.index >= executable.magnetic_half_graph_program_.boundaries.size())
        throw std::logic_error("NVIDIA magnetic graph boundary is out of range");
      const GraphBoundary &boundary =
          executable.magnetic_half_graph_program_.boundaries[entry.index];
      ++executable.graph_statistics_.magnetic_boundary_count;
      ++state.graph_statistics_.magnetic_boundary_count;
      if (boundary.kind != GraphBoundaryKind::source_evaluation ||
          boundary.first_operation >= executable.operations_.size())
        throw std::logic_error("NVIDIA magnetic graph has an invalid host boundary");
      const NvidiaCompiledOperation &op = executable.operations_[boundary.first_operation];
      evaluate_supported_source_scalars(f_, op.source_time_offset);
      const SourcePlan &source_plan = f_.descriptors->sources;
      if (source_plan.scalars.size() != executable.source_scalar_count_ ||
          op.count != executable.source_scalar_count_)
        throw std::logic_error("NVIDIA magnetic source scalar block changed after compilation");
      if (!op.count) continue;
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
    }
    return;
  }
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
  if (executable.graph_enabled_) {
    if (executable.magnetic_auxiliary_graphs_.size() != 2)
      throw std::logic_error("NVIDIA magnetic graph auxiliary set is incomplete");
    executable.magnetic_auxiliary_graphs_[0].executable.launch(*state.transfer_);
    ++executable.graph_statistics_.magnetic_launch_count;
    ++state.graph_statistics_.magnetic_launch_count;
  }
  else
    for (const NvidiaCompiledMagneticState &row : executable.magnetic_states_)
      nvidia::launch_magnetic_backup(
          nvidia::magnetic_state_launch{row.live, backup + row.backup_offset, row.elements,
                                        row.precision},
          *state.transfer_);
  execute_magnetic_half_step(executable, state);
  if (executable.graph_enabled_) {
    executable.magnetic_auxiliary_graphs_[1].executable.launch(*state.transfer_);
    ++executable.graph_statistics_.magnetic_launch_count;
    ++state.graph_statistics_.magnetic_launch_count;
  }
  else
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
  if (executable.graph_enabled_) {
    if (executable.magnetic_restore_graph_segments_.size() != 1)
      throw std::logic_error("NVIDIA magnetic restore graph set is incomplete");
    executable.magnetic_restore_graph_segments_[0].executable.launch(*state.transfer_);
    ++executable.graph_statistics_.magnetic_launch_count;
    ++state.graph_statistics_.magnetic_launch_count;
  }
  else
    for (const NvidiaCompiledMagneticState &row : executable.magnetic_states_)
      nvidia::launch_magnetic_restore(
          nvidia::magnetic_state_launch{row.live, backup + row.backup_offset, row.elements,
                                        row.precision},
          *state.transfer_);
  state.transfer_->synchronize();
  state.magnetic_snapshot_valid_ = false;
  state.device_authoritative_ = true;
}

void nvidia::validate_material_phase_state(const fields &f,
                                           uint64_t expected_target_signature) {
  const bool local_active = f.phasein_time > 0;
  backend_note_material_phase_collective_for_testing();
  const int minimum_countdown = min_to_all(f.phasein_time);
  backend_note_material_phase_collective_for_testing();
  const int maximum_countdown = max_to_all(f.phasein_time);
  backend_note_material_phase_collective_for_testing();
  const bool every_active = and_to_all(local_active);
  backend_note_material_phase_collective_for_testing();
  const bool any_active = or_to_all(local_active);
  backend_note_material_phase_scan_for_testing();
  const bool local_target_matches =
      !local_active || expected_target_signature == compute_material_phase_target_signature(f);
  backend_note_material_phase_collective_for_testing();
  const bool every_target_matches = and_to_all(local_target_matches);
  bool local_storage_detached = true;
  backend_note_material_phase_scan_for_testing();
  if (local_active)
    for (int i = 0; i < f.num_chunks; ++i)
      if (f.chunks[i] && f.chunks[i]->is_mine() &&
          (!f.chunks[i]->s || f.chunks[i]->s->refcount != 1)) {
        local_storage_detached = false;
          break;
      }
  backend_note_material_phase_collective_for_testing();
  const bool every_storage_detached = and_to_all(local_storage_detached);
  if (minimum_countdown != maximum_countdown || any_active != every_active ||
      !every_target_matches || !every_storage_detached)
    throw std::logic_error(
        "NVIDIA material phase state, target, or current storage changed after compilation on "
        "an MPI rank");
}

void NvidiaBackend::preflight_advance(Executable &raw_executable,
                                      BackendState &raw_state, int num_steps) {
  NvidiaBackendState &state = checked_state(raw_state);
  NvidiaExecutable &executable = checked_executable(raw_executable, state);
  if (num_steps <= 0) return;
  if (!state.initialized_) throw std::logic_error("cannot advance uninitialized NVIDIA storage");
  if (executable.storage_fingerprint_ != state.fingerprint_)
    throw std::logic_error("NVIDIA executable was compiled for a different storage layout");
  if (state.transfer_failed_)
    throw std::logic_error("NVIDIA execution stream failed; recreate backend state");
  if (executable.has_noisy_updates_ && f_.t < 0)
    throw std::invalid_argument("NVIDIA noisy polarization timestep is negative");
  if (executable.material_phase_active != state.material_phase_active)
    throw std::logic_error("NVIDIA executable and storage disagree on material phase activity");
  if (!executable.material_phase_active) return;
  nvidia::validate_material_phase_state(f_, executable.material_target_signature_);
}

bool NvidiaBackend::launch_device_operation(NvidiaExecutable &executable,
                                            NvidiaBackendState &state,
                                            const NvidiaCompiledOperation &op,
                                            const StepScalars *graph_scalars,
                                            bool account_cw) {
  const auto count_cw_kernel = [&]() {
    if (account_cw) ++executable.cw_workspace_->timestep_kernel_launches_;
  };
  const auto accumulate_legacy_flux = [&]() {
    for (size_t i = op.legacy_flux_first;
         i < op.legacy_flux_first + op.legacy_flux_count; ++i) {
      const NvidiaCompiledLegacyFluxUpdate &update = executable.legacy_flux_updates_[i];
      double *result = static_cast<double *>(executable.legacy_flux_current_.opaque_handle()) +
                       update.flux_ordinal;
      for (size_t j = update.term_first; j < update.term_first + update.term_count; ++j) {
        const nvidia::legacy_flux_term_launch &term = executable.legacy_flux_terms_[j];
        if (!term.e_real || !term.h_real) continue;
        nvidia::launch_legacy_flux_term(term, executable.legacy_flux_partials_.opaque_handle(),
                                        result, *state.transfer_);
      }
    }
  };
  switch (op.kind) {
    case OpKind::update_db:
      for (size_t i = op.first; i < op.first + op.count; ++i) {
        const uint32_t prefix = executable.curl_updates_[i].radial_prefix_index;
        if (prefix != UINT32_MAX)
          nvidia::launch_cylindrical_radial_prefix(
              executable.cylindrical_radial_prefixes_[prefix], *state.transfer_);
        if (prefix != UINT32_MAX) count_cw_kernel();
        nvidia::launch_curl(executable.curl_updates_[i], *state.transfer_);
        count_cw_kernel();
        const uint32_t bfast = executable.curl_updates_[i].bfast_update_index;
        if (bfast != UINT32_MAX)
          nvidia::launch_bfast(executable.bfast_updates_[bfast], *state.transfer_);
        if (bfast != UINT32_MAX) count_cw_kernel();
      }
      for (size_t i = op.beta_first; i < op.beta_first + op.beta_count; ++i) {
        nvidia::launch_beta(executable.beta_updates_[i], *state.transfer_);
        count_cw_kernel();
      }
      for (size_t i = op.cylindrical_m_first;
           i < op.cylindrical_m_first + op.cylindrical_m_count; ++i) {
        nvidia::launch_cylindrical_m(executable.cylindrical_m_updates_[i], *state.transfer_);
        count_cw_kernel();
      }
      for (size_t i = op.cylindrical_origin_first;
           i < op.cylindrical_origin_first + op.cylindrical_origin_count; ++i) {
        const NvidiaCompiledCylindricalOriginAction &action =
            executable.cylindrical_origin_actions_[i];
        if (action.kind == CylindricalOriginActionKind::axis_update)
          nvidia::launch_cylindrical_axis(executable.cylindrical_axis_updates_[action.index],
                                          *state.transfer_);
        else
          nvidia::launch_zero(executable.zero_updates_[action.index], *state.transfer_);
        count_cw_kernel();
      }
      return true;
    case OpKind::update_eh:
      for (size_t i = op.copy_first; i < op.copy_first + op.copy_count; ++i) {
        nvidia::launch_array_copy(executable.source_copies_[i], *state.transfer_);
        count_cw_kernel();
      }
      for (size_t i = op.subtraction_first; i < op.subtraction_first + op.subtraction_count; ++i) {
        nvidia::launch_polarization_subtract(executable.polarization_subtractions_[i],
                                             *state.transfer_);
        count_cw_kernel();
      }
      for (size_t i = op.source_first; i < op.source_first + op.source_count; ++i) {
        nvidia::launch_source_batch(executable.source_batches_[i],
                                    executable.source_scalars_.opaque_handle(), *state.transfer_);
        count_cw_kernel();
      }
      for (size_t i = op.first; i < op.first + op.count; ++i) {
        nvidia::launch_constitutive(executable.constitutive_updates_[i], *state.transfer_);
        count_cw_kernel();
      }
      return true;
    case OpKind::update_polarization: {
      const nvidia::noisy_seed_block *seed = NULL;
      if (graph_scalars)
        seed = static_cast<const nvidia::noisy_seed_block *>(
            state.noisy_seed_slots_.opaque_handle());
      else if (state.noisy_seed_active_slot_ >= 0)
        seed = static_cast<const nvidia::noisy_seed_block *>(
                   state.noisy_seed_slots_.opaque_handle()) +
               state.noisy_seed_active_slot_;
      for (size_t i = op.polarization_first;
           i < op.polarization_first + op.polarization_count; ++i) {
        const NvidiaCompiledPolarizationAction &action = executable.polarization_actions_[i];
        if (action.kind == NvidiaCompiledPolarizationAction::Kind::ordinary) {
          const nvidia::compiled_polarization_update &update =
              executable.polarization_updates_[action.index];
          if (graph_scalars)
            nvidia::launch_polarization_update_graph(update, seed, graph_scalars,
                                                      *state.transfer_);
          else
            nvidia::launch_polarization_update(update, seed, uint64_t(f_.t), *state.transfer_);
          if (!graph_scalars &&
              update.kind == nvidia::compiled_polarization_update::kind_type::noisy_add &&
              nvidia::testing::consume_failure_for_testing(
                  nvidia::testing::failure_point::noisy_add))
            throw std::runtime_error("injected NVIDIA noisy-add postlaunch failure");
        }
        else if (action.kind == NvidiaCompiledPolarizationAction::Kind::multilevel_population) {
          nvidia::launch_multilevel_population(
              executable.multilevel_population_updates_[action.index], *state.transfer_);
          if (!graph_scalars && nvidia::testing::consume_failure_for_testing(
                                    nvidia::testing::failure_point::multilevel_population))
            throw std::runtime_error("injected NVIDIA multilevel population postlaunch failure");
        }
        else if (action.kind == NvidiaCompiledPolarizationAction::Kind::multilevel_transition) {
          nvidia::launch_multilevel_transition(
              executable.multilevel_transition_updates_[action.index], *state.transfer_);
          if (!graph_scalars && nvidia::testing::consume_failure_for_testing(
                                    nvidia::testing::failure_point::multilevel_transition))
            throw std::runtime_error("injected NVIDIA multilevel transition postlaunch failure");
        }
        else
          throw std::logic_error("NVIDIA polarization schedule has an invalid action");
        count_cw_kernel();
      }
      return true;
    }
    case OpKind::transfer_halo:
      for (size_t i = op.first; i < op.first + op.count; ++i) {
        nvidia::launch_zero(executable.zero_updates_[i], *state.transfer_);
        count_cw_kernel();
      }
      for (size_t i = op.halo_first; i < op.halo_first + op.halo_count; ++i) {
        nvidia::launch_halo_gather(executable.halo_plans_[i].gather,
                                   executable.halo_gathers_.opaque_handle(),
                                   executable.halo_scratch_.opaque_handle(), *state.transfer_);
        count_cw_kernel();
      }
      for (size_t i = op.halo_first; i < op.halo_first + op.halo_count; ++i) {
        nvidia::launch_halo_scatter(executable.halo_plans_[i].scatter,
                                    executable.halo_scatters_.opaque_handle(),
                                    executable.halo_scratch_.opaque_handle(), *state.transfer_);
        count_cw_kernel();
      }
      return true;
    case OpKind::apply_sources:
      for (size_t i = op.first; i < op.first + op.count; ++i) {
        nvidia::launch_source_batch(executable.source_batches_[i],
                                    executable.source_scalars_.opaque_handle(), *state.transfer_);
        count_cw_kernel();
      }
      return true;
    case OpKind::update_dft:
      for (size_t i = op.first; i < op.first + op.count; ++i) {
        const nvidia::dft_launch &dft = executable.dft_updates_[i];
        if (graph_scalars) {
          const StepScalarSlot *predicate = NULL;
          for (const StepScalarSlot &slot : executable.graph_program_.scalar_layout.slots)
            if (slot.semantic == StepScalarSemantic::dft_due_predicate &&
                slot.semantic_index == uint32_t(dft.decimation_factor)) {
              predicate = &slot;
              break;
            }
          if (!predicate) throw std::logic_error("NVIDIA graph DFT predicate is missing");
          const size_t word = (predicate->byte_offset - offsetof(StepScalars, predicate_words)) /
                              sizeof(uint64_t);
          nvidia::launch_dft_graph(dft, graph_scalars, uint32_t(word), predicate->bit_offset,
                                   *state.transfer_);
        }
        else {
          if ((f_.t % dft.decimation_factor) != 0) continue;
          const double sample_time = dft.magnetic ? f_.time() - 0.5 * f_.dt : f_.time();
          nvidia::launch_dft(dft, sample_time, *state.transfer_);
        }
        count_cw_kernel();
      }
      return true;
    case OpKind::update_flux_half: {
      const size_t bytes = checked_product(op.legacy_flux_count, sizeof(double),
                                           "clearing NVIDIA legacy flux half sample");
      nvidia::fill_byte_async(executable.legacy_flux_current_, 0, 0, bytes, *state.transfer_);
      accumulate_legacy_flux();
      nvidia::copy_device_to_device_async(executable.legacy_flux_half_, 0,
                                          executable.legacy_flux_current_, 0, bytes,
                                          *state.transfer_);
      return true;
    }
    case OpKind::update_flux: {
      const size_t bytes = checked_product(op.legacy_flux_count, sizeof(double),
                                           "copying NVIDIA legacy flux result");
      nvidia::fill_byte_async(executable.legacy_flux_current_, 0, 0, bytes, *state.transfer_);
      accumulate_legacy_flux();
      nvidia::launch_legacy_flux_average(executable.legacy_flux_current_.opaque_handle(),
                                         executable.legacy_flux_half_.opaque_handle(),
                                         op.legacy_flux_count, *state.transfer_);
      return true;
    }
    case OpKind::finite_value_check:
      nvidia::fill_byte_async(executable.finite_result_, 0, 0xff, sizeof(uint64_t),
                              *state.transfer_);
      for (size_t i = op.first; i < op.first + op.count; ++i) {
        if (graph_scalars)
          nvidia::launch_finite_check_graph(executable.finite_checks_[i].launch,
                                            executable.finite_result_.opaque_handle(),
                                            graph_scalars, *state.transfer_);
        else
          nvidia::launch_finite_check(executable.finite_checks_[i].launch,
                                      executable.finite_result_.opaque_handle(),
                                      *state.transfer_);
      }
      if (account_cw) executable.cw_workspace_->timestep_kernel_launches_ += op.count;
      if (account_cw) executable.cw_workspace_->finite_check_kernel_launches_ += op.count;
      return true;
    case OpKind::pack_state:
    case OpKind::unpack_state:
    case OpKind::reduction:
      /* These neutral CW markers carry authority for the separately owned
         workspace graphs.  The StepProgram graph must retain their order,
         but the concrete pack/unpack/reduction launches are compiled from
         CwPlan and issued by solve_cw, not duplicated here. */
      return executable.program_ == StepProgram::solve_cw;
    default: return false;
  }
}

void NvidiaBackend::configure_graph_execution(const StepPlan &plan,
                                              NvidiaExecutable &executable,
                                              NvidiaBackendState &state) {
  executable.graph_statistics_ = NvidiaGraphStatistics();
  executable.graph_statistics_.requested = graph_mode_;
  executable.graph_statistics_.valid = true;

  const GraphExecutionMode requested =
      reconcile_graph_mode(graph_mode_, graph_mode_parse_valid_);
  executable.graph_statistics_.requested = requested;
  if (requested != GraphExecutionMode::eager && state.magnetic_snapshot_valid_)
    throw std::logic_error(
        "NVIDIA graph executable replacement is forbidden while a magnetic snapshot is live");
  const int local_program = int(plan.program);
  const int minimum_program = min_to_all(local_program);
  const int maximum_program = max_to_all(local_program);
  if (minimum_program != maximum_program)
    throw std::invalid_argument("NVIDIA timestep program differs across ranks");

  if (plan.program != StepProgram::ordinary) {
    return;
  }

  GraphLoweringAuthorities authority;
  GraphProgram graph_program;
  GraphProgram magnetic_half_program;
  GraphProgram magnetic_restore_program;
  std::string local_error;
  try {
    authority = build_graph_lowering_authorities(plan, f_.halos, f_.array_catalog,
                                                  f_.is_real ? 1 : 2);
    graph_program = build_graph_program(plan, authority, GraphVariantKind::ordinary);
    magnetic_half_program =
        build_graph_program(plan, authority, GraphVariantKind::magnetic_half_step);
    magnetic_restore_program =
        build_graph_program(plan, authority, GraphVariantKind::magnetic_restore);
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown NVIDIA graph lowering failure";
  }
  reconcile_graph_stage(local_error, "NVIDIA graph lowering failed on another rank");

  local_error.clear();
  bool program_graphable = false;
  try {
    std::string graph_error;
    if (!validate_graph_program(plan, authority, graph_program, &graph_error))
      throw std::invalid_argument(graph_error.empty() ? "invalid NVIDIA graph program"
                                                       : graph_error);
    if (!validate_graph_program(plan, authority, magnetic_half_program, &graph_error) ||
        !validate_graph_program(plan, authority, magnetic_restore_program, &graph_error))
      throw std::invalid_argument(graph_error.empty() ? "invalid NVIDIA magnetic graph program"
                                                       : graph_error);
    program_graphable = graph_required_compatible(graph_program) &&
                        graph_required_compatible(magnetic_half_program) &&
                        graph_required_compatible(magnetic_restore_program) &&
                        !executable.magnetic_states_.empty() &&
                        executable.magnetic_snapshot_bytes_ != 0;
    const GraphProgram *programs[] = {
        &graph_program, &magnetic_half_program, &magnetic_restore_program};
    for (const GraphProgram *candidate : programs) {
      for (const GraphBoundary &boundary : candidate->boundaries)
        if (boundary.kind == GraphBoundaryKind::remote_halo) program_graphable = false;
      for (const GraphSegment &segment : candidate->segments)
        for (const GraphOperationRef &operation : segment.operations)
          if (operation.operation.guard.kind == GuardKind::device_predicate &&
              operation.operation.kind != OpKind::update_dft)
            program_graphable = false;
    }
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown NVIDIA graph validation failure";
  }
  reconcile_graph_stage(local_error, "NVIDIA graph validation failed on another rank");

  bool runtime_supported = false;
  local_error.clear();
  try {
    runtime_supported = nvidia::query_graph_capability().capture_supported;
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown CUDA graph capability-query failure";
  }
  reconcile_graph_stage(local_error,
                        "CUDA graph capability query failed on another rank");
  if (!reconcile_graph_support(requested, runtime_supported && program_graphable)) return;

  const auto discard_graph_state = [&]() {
    executable.graph_enabled_ = false;
    executable.graph_statistics_.enabled = false;
    executable.graph_statistics_.segment_count = 0;
    executable.graph_statistics_.capture_count = 0;
    executable.graph_statistics_.instantiate_count = 0;
    executable.graph_statistics_.magnetic_segment_count = 0;
    executable.graph_statistics_.magnetic_capture_count = 0;
    executable.graph_statistics_.magnetic_instantiate_count = 0;
    std::string cleanup_error;
    const auto remember_cleanup_error = [&](const std::exception &error) {
      if (cleanup_error.empty()) cleanup_error = error.what();
    };
    const auto reset_segments = [&](std::vector<NvidiaCompiledGraphSegment> &segments) {
      for (NvidiaCompiledGraphSegment &segment : segments) {
        try { segment.executable.reset(); }
        catch (const std::exception &error) { remember_cleanup_error(error); }
        catch (...) {
          if (cleanup_error.empty())
            cleanup_error = "unknown CUDA graph executable cleanup failure";
        }
        try { segment.definition.reset(); }
        catch (const std::exception &error) { remember_cleanup_error(error); }
        catch (...) {
          if (cleanup_error.empty())
            cleanup_error = "unknown CUDA graph definition cleanup failure";
        }
      }
    };
    reset_segments(executable.graph_segments_);
    reset_segments(executable.magnetic_half_graph_segments_);
    reset_segments(executable.magnetic_restore_graph_segments_);
    for (NvidiaCapturedGraph &graph : executable.magnetic_auxiliary_graphs_) {
      try { graph.executable.reset(); }
      catch (const std::exception &error) { remember_cleanup_error(error); }
      catch (...) {
        if (cleanup_error.empty())
          cleanup_error = "unknown CUDA graph executable cleanup failure";
      }
      try { graph.definition.reset(); }
      catch (const std::exception &error) { remember_cleanup_error(error); }
      catch (...) {
        if (cleanup_error.empty())
          cleanup_error = "unknown CUDA graph definition cleanup failure";
      }
    }
    bool graphs_released = true;
    const auto segments_released = [&](const std::vector<NvidiaCompiledGraphSegment> &segments) {
      bool released = true;
      for (const NvidiaCompiledGraphSegment &segment : segments)
        released = released && !segment.executable.opaque_handle() &&
                   !segment.definition.opaque_handle() && !segment.definition.capturing();
      return released;
    };
    graphs_released = segments_released(executable.graph_segments_) &&
                      segments_released(executable.magnetic_half_graph_segments_) &&
                      segments_released(executable.magnetic_restore_graph_segments_);
    for (const NvidiaCapturedGraph &graph : executable.magnetic_auxiliary_graphs_)
      graphs_released = graphs_released && !graph.executable.opaque_handle() &&
                        !graph.definition.opaque_handle() && !graph.definition.capturing();
    if (graphs_released) {
      executable.graph_segments_.clear();
      executable.magnetic_half_graph_segments_.clear();
      executable.magnetic_restore_graph_segments_.clear();
      executable.magnetic_auxiliary_graphs_.clear();
      executable.graph_program_ = GraphProgram();
      executable.magnetic_half_graph_program_ = GraphProgram();
      executable.magnetic_restore_graph_program_ = GraphProgram();
      try {
        executable.step_scalars_.reset();
        executable.magnetic_step_scalars_.reset();
        state.transfer_->synchronize();
      }
      catch (const std::exception &error) { remember_cleanup_error(error); }
      catch (...) {
        if (cleanup_error.empty()) cleanup_error = "unknown CUDA graph cleanup failure";
      }
    }
    reconcile_graph_stage(cleanup_error, "CUDA graph cleanup failed on another rank");
  };

  nvidia::device_buffer staged_magnetic_snapshot;
  void *magnetic_snapshot = state.magnetic_snapshot_.opaque_handle();
  bool publish_magnetic_snapshot = false;
  local_error.clear();
  try {
    executable.graph_program_ = graph_program;
    executable.magnetic_half_graph_program_ = magnetic_half_program;
    executable.magnetic_restore_graph_program_ = magnetic_restore_program;
    executable.step_scalars_.allocate(sizeof(StepScalars), state.device_);
    executable.magnetic_step_scalars_.allocate(sizeof(StepScalars), state.device_);
    if (state.magnetic_snapshot_.size() < executable.magnetic_snapshot_bytes_) {
      if (state.magnetic_snapshot_.size())
        throw std::logic_error(
            "NVIDIA magnetic graph snapshot growth requires a replacement epoch");
      staged_magnetic_snapshot.allocate(executable.magnetic_snapshot_bytes_, state.device_);
      magnetic_snapshot = staged_magnetic_snapshot.opaque_handle();
      publish_magnetic_snapshot = true;
    }
    state.transfer_->synchronize();
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown CUDA graph scalar-allocation failure";
  }
  const bool allocation_failed = or_to_all(!local_error.empty());
  if (allocation_failed) {
    discard_graph_state();
    if (requested == GraphExecutionMode::required)
      throw std::runtime_error(local_error.empty()
                                   ? "CUDA graph scalar allocation failed on another rank"
                                   : local_error);
    return;
  }

  local_error.clear();
  try {
    executable.graph_segments_.reserve(graph_program.segments.size());
    const StepScalars *device_scalars =
        static_cast<const StepScalars *>(executable.step_scalars_.opaque_handle());
    for (size_t i = 0; i < graph_program.segments.size(); ++i) {
      std::ostringstream label;
      label << "ordinary-step-segment-" << i;
      executable.graph_segments_.push_back(NvidiaCompiledGraphSegment(graph_program.segments[i]));
      NvidiaCompiledGraphSegment &compiled = executable.graph_segments_.back();
      compiled.definition.begin_capture(*state.transfer_, label.str());
      for (const GraphOperationRef &operation : compiled.plan.operations) {
        if (operation.operation_index >= executable.operations_.size())
          throw std::logic_error("NVIDIA graph operation index is out of range");
        if (!launch_device_operation(executable, state,
                                     executable.operations_[operation.operation_index],
                                     device_scalars, false))
          throw std::logic_error("NVIDIA graph segment contains a host-only operation");
      }
      compiled.definition.end_capture(*state.transfer_);
      ++executable.graph_statistics_.capture_count;
    }

    executable.magnetic_auxiliary_graphs_.reserve(2);
    executable.magnetic_auxiliary_graphs_.push_back(NvidiaCapturedGraph());
    NvidiaCapturedGraph &backup = executable.magnetic_auxiliary_graphs_.back();
    backup.definition.begin_capture(*state.transfer_, "magnetic-backup");
    char *snapshot = static_cast<char *>(magnetic_snapshot);
    for (const NvidiaCompiledMagneticState &row : executable.magnetic_states_)
      nvidia::launch_magnetic_backup(
          nvidia::magnetic_state_launch{row.live, snapshot + row.backup_offset, row.elements,
                                        row.precision},
          *state.transfer_);
    backup.definition.end_capture(*state.transfer_);
    ++executable.graph_statistics_.magnetic_capture_count;

    executable.magnetic_half_graph_segments_.reserve(magnetic_half_program.segments.size());
    const StepScalars *magnetic_scalars =
        static_cast<const StepScalars *>(executable.magnetic_step_scalars_.opaque_handle());
    for (size_t i = 0; i < magnetic_half_program.segments.size(); ++i) {
      std::ostringstream label;
      label << "magnetic-half-step-segment-" << i;
      executable.magnetic_half_graph_segments_.push_back(
          NvidiaCompiledGraphSegment(magnetic_half_program.segments[i]));
      NvidiaCompiledGraphSegment &compiled = executable.magnetic_half_graph_segments_.back();
      compiled.definition.begin_capture(*state.transfer_, label.str());
      for (const GraphOperationRef &operation : compiled.plan.operations) {
        if (operation.operation_index >= executable.operations_.size())
          throw std::logic_error("NVIDIA magnetic graph operation index is out of range");
        if (!launch_device_operation(executable, state,
                                     executable.operations_[operation.operation_index],
                                     magnetic_scalars, false))
          throw std::logic_error("NVIDIA magnetic graph segment contains a host-only operation");
      }
      compiled.definition.end_capture(*state.transfer_);
      ++executable.graph_statistics_.magnetic_capture_count;
    }

    executable.magnetic_auxiliary_graphs_.push_back(NvidiaCapturedGraph());
    NvidiaCapturedGraph &average = executable.magnetic_auxiliary_graphs_.back();
    average.definition.begin_capture(*state.transfer_, "magnetic-average");
    for (const NvidiaCompiledMagneticState &row : executable.magnetic_states_)
      if (row.average)
        nvidia::launch_magnetic_average(
            nvidia::magnetic_state_launch{row.live, snapshot + row.backup_offset, row.elements,
                                          row.precision},
            *state.transfer_);
    average.definition.end_capture(*state.transfer_);
    ++executable.graph_statistics_.magnetic_capture_count;

    executable.magnetic_restore_graph_segments_.reserve(
        magnetic_restore_program.segments.size());
    for (size_t i = 0; i < magnetic_restore_program.segments.size(); ++i) {
      std::ostringstream label;
      label << "magnetic-restore-segment-" << i;
      executable.magnetic_restore_graph_segments_.push_back(
          NvidiaCompiledGraphSegment(magnetic_restore_program.segments[i]));
      NvidiaCompiledGraphSegment &compiled = executable.magnetic_restore_graph_segments_.back();
      compiled.definition.begin_capture(*state.transfer_, label.str());
      for (const GraphOperationRef &operation : compiled.plan.operations) {
        if (operation.operation.kind != OpKind::restore_magnetic_fields)
          throw std::logic_error("NVIDIA magnetic restore graph has an invalid operation");
        for (const NvidiaCompiledMagneticState &row : executable.magnetic_states_)
          nvidia::launch_magnetic_restore(
              nvidia::magnetic_state_launch{row.live, snapshot + row.backup_offset, row.elements,
                                            row.precision},
              *state.transfer_);
      }
      compiled.definition.end_capture(*state.transfer_);
      ++executable.graph_statistics_.magnetic_capture_count;
    }
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown CUDA graph capture failure";
  }
  const bool capture_failed = or_to_all(!local_error.empty());
  if (capture_failed) {
    discard_graph_state();
    if (requested == GraphExecutionMode::required)
      throw std::runtime_error(local_error.empty() ? "CUDA graph capture failed on another rank"
                                                   : local_error);
    return;
  }

  local_error.clear();
  try {
    const auto instantiate_segments = [&](std::vector<NvidiaCompiledGraphSegment> &segments) {
      for (NvidiaCompiledGraphSegment &compiled : segments) {
        compiled.executable.instantiate(compiled.definition);
        ++executable.graph_statistics_.instantiate_count;
      }
    };
    instantiate_segments(executable.graph_segments_);
    if (nvidia::testing::consume_failure_for_testing(
            nvidia::testing::failure_point::magnetic_graph_instantiate))
      throw std::runtime_error("injected CUDA magnetic graph instantiate failure");
    const auto instantiate_magnetic_segments =
        [&](std::vector<NvidiaCompiledGraphSegment> &segments) {
          for (NvidiaCompiledGraphSegment &compiled : segments) {
            compiled.executable.instantiate(compiled.definition);
            ++executable.graph_statistics_.magnetic_instantiate_count;
          }
        };
    instantiate_magnetic_segments(executable.magnetic_half_graph_segments_);
    instantiate_magnetic_segments(executable.magnetic_restore_graph_segments_);
    for (NvidiaCapturedGraph &graph : executable.magnetic_auxiliary_graphs_) {
      graph.executable.instantiate(graph.definition);
      ++executable.graph_statistics_.magnetic_instantiate_count;
    }
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown CUDA graph instantiate failure";
  }
  const bool instantiate_failed = or_to_all(!local_error.empty());
  if (instantiate_failed) {
    discard_graph_state();
    if (requested == GraphExecutionMode::required)
      throw std::runtime_error(local_error.empty()
                                   ? "CUDA graph instantiate failed on another rank"
                                   : local_error);
    return;
  }

  if (publish_magnetic_snapshot) {
    void *const expected = staged_magnetic_snapshot.opaque_handle();
    state.magnetic_snapshot_ = std::move(staged_magnetic_snapshot);
    if (state.magnetic_snapshot_.opaque_handle() != expected) {
      discard_graph_state();
      throw std::runtime_error("NVIDIA magnetic graph snapshot publication failed");
    }
  }
  executable.graph_enabled_ = true;
  executable.graph_statistics_.enabled = true;
  executable.graph_statistics_.segment_count = executable.graph_segments_.size();
  executable.graph_statistics_.magnetic_segment_count =
      executable.magnetic_half_graph_segments_.size() +
      executable.magnetic_restore_graph_segments_.size() +
      executable.magnetic_auxiliary_graphs_.size();
}

void NvidiaBackend::configure_cw_graph_execution(const StepPlan &plan, const CwPlan &cw_plan,
                                                 NvidiaExecutable &executable,
                                                 NvidiaBackendState &state) {
  const GraphExecutionMode requested =
      reconcile_graph_mode(graph_mode_, graph_mode_parse_valid_);
  executable.graph_statistics_ = NvidiaGraphStatistics();
  executable.graph_statistics_.requested = requested;
  executable.graph_statistics_.valid = true;

  GraphLoweringAuthorities authority;
  GraphProgram program;
  std::string local_error;
  try {
    authority = build_graph_lowering_authorities(plan, f_.halos, f_.array_catalog,
                                                  f_.is_real ? 1 : 2, &cw_plan);
    program = build_graph_program(plan, authority, GraphVariantKind::cw_operator);
    std::string graph_error;
    if (!validate_graph_program(plan, authority, program, &graph_error))
      throw std::invalid_argument(graph_error.empty() ? "invalid NVIDIA CW graph program"
                                                       : graph_error);
  }
  catch (const std::exception &error) {
    local_error = error.what();
  }
  catch (...) {
    local_error = "unknown NVIDIA CW graph lowering failure";
  }
  reconcile_graph_stage(local_error, "NVIDIA CW graph lowering failed on another rank");

  bool graphable = graph_required_compatible(program);
  for (const GraphBoundary &boundary : program.boundaries)
    if (boundary.kind == GraphBoundaryKind::remote_halo) graphable = false;
  for (const GraphSegment &segment : program.segments)
    for (const GraphOperationRef &operation : segment.operations)
      if (operation.operation.guard.kind == GuardKind::device_predicate &&
          operation.operation.kind != OpKind::update_dft &&
          operation.operation.kind != OpKind::finite_value_check)
        graphable = false;

  bool runtime_supported = false;
  local_error.clear();
  try { runtime_supported = nvidia::query_graph_capability().capture_supported; }
  catch (const std::exception &error) { local_error = error.what(); }
  catch (...) { local_error = "unknown CUDA CW graph capability-query failure"; }
  reconcile_graph_stage(local_error, "CUDA CW graph capability query failed on another rank");
  if (!reconcile_graph_support(requested, runtime_supported && graphable)) return;

  const auto discard = [&]() {
    executable.graph_enabled_ = false;
    executable.graph_statistics_.enabled = false;
    std::string cleanup_error;
    for (NvidiaCompiledGraphSegment &segment : executable.graph_segments_) {
      try { segment.executable.reset(); }
      catch (const std::exception &error) {
        if (cleanup_error.empty()) cleanup_error = error.what();
      }
      try { segment.definition.reset(); }
      catch (const std::exception &error) {
        if (cleanup_error.empty()) cleanup_error = error.what();
      }
    }
    bool released = true;
    for (const NvidiaCompiledGraphSegment &segment : executable.graph_segments_)
      released = released && !segment.executable.opaque_handle() &&
                 !segment.definition.opaque_handle() && !segment.definition.capturing();
    if (released) {
      executable.graph_segments_.clear();
      executable.graph_cw_timestep_launches_.clear();
      executable.graph_cw_finite_launches_.clear();
      executable.graph_program_ = GraphProgram();
      try { executable.step_scalars_.reset(); }
      catch (const std::exception &error) {
        if (cleanup_error.empty()) cleanup_error = error.what();
      }
    }
    reconcile_graph_stage(cleanup_error, "CUDA CW graph cleanup failed on another rank");
  };

  local_error.clear();
  try {
    executable.graph_program_ = program;
    executable.step_scalars_.allocate(sizeof(StepScalars), state.device_);
    state.transfer_->synchronize();
  }
  catch (const std::exception &error) { local_error = error.what(); }
  catch (...) { local_error = "unknown CUDA CW graph scalar-allocation failure"; }
  if (or_to_all(!local_error.empty())) {
    discard();
    if (requested == GraphExecutionMode::required)
      throw std::runtime_error(local_error.empty()
                                   ? "CUDA CW graph scalar allocation failed on another rank"
                                   : local_error);
    return;
  }

  local_error.clear();
  try {
    executable.graph_segments_.reserve(program.segments.size());
    executable.graph_cw_timestep_launches_.reserve(program.segments.size());
    executable.graph_cw_finite_launches_.reserve(program.segments.size());
    const StepScalars *device_scalars =
        static_cast<const StepScalars *>(executable.step_scalars_.opaque_handle());
    for (size_t i = 0; i < program.segments.size(); ++i) {
      std::ostringstream label;
      label << "cw-operator-segment-" << i;
      executable.graph_segments_.push_back(NvidiaCompiledGraphSegment(program.segments[i]));
      NvidiaCompiledGraphSegment &compiled = executable.graph_segments_.back();
      compiled.definition.begin_capture(*state.transfer_, label.str());
      size_t timestep_launches = 0;
      size_t finite_launches = 0;
      for (const GraphOperationRef &operation : compiled.plan.operations) {
        if (operation.operation_index >= executable.operations_.size() ||
            !launch_device_operation(executable, state,
                                     executable.operations_[operation.operation_index],
                                     device_scalars, false))
          throw std::logic_error("NVIDIA CW graph segment contains an invalid operation at " +
                                 std::to_string(operation.operation_index) + " kind " +
                                 std::to_string(int(operation.operation.kind)));
        const NvidiaCompiledOperation &op =
            executable.operations_[operation.operation_index];
        switch (op.kind) {
          case OpKind::update_db:
            for (size_t j = op.first; j < op.first + op.count; ++j) {
              ++timestep_launches;
              timestep_launches += executable.curl_updates_[j].radial_prefix_index != UINT32_MAX;
              timestep_launches += executable.curl_updates_[j].bfast_update_index != UINT32_MAX;
            }
            timestep_launches += op.beta_count + op.cylindrical_m_count +
                                 op.cylindrical_origin_count;
            break;
          case OpKind::update_eh:
            timestep_launches += op.copy_count + op.subtraction_count + op.source_count + op.count;
            break;
          case OpKind::update_polarization: timestep_launches += op.polarization_count; break;
          case OpKind::apply_sources:
          case OpKind::zero_boundary: timestep_launches += op.count; break;
          case OpKind::transfer_halo:
            timestep_launches += op.count + 2 * op.halo_count;
            break;
          case OpKind::finite_value_check:
            timestep_launches += op.count;
            finite_launches += op.count;
            break;
          default: break;
        }
      }
      compiled.definition.end_capture(*state.transfer_);
      executable.graph_cw_timestep_launches_.push_back(timestep_launches);
      executable.graph_cw_finite_launches_.push_back(finite_launches);
      ++executable.graph_statistics_.capture_count;
    }
  }
  catch (const std::exception &error) { local_error = error.what(); }
  catch (...) { local_error = "unknown CUDA CW graph capture failure"; }
  if (or_to_all(!local_error.empty())) {
    discard();
    if (requested == GraphExecutionMode::required)
      throw std::runtime_error(local_error.empty() ? "CUDA CW graph capture failed on another rank"
                                                   : local_error);
    return;
  }

  local_error.clear();
  try {
    for (NvidiaCompiledGraphSegment &segment : executable.graph_segments_) {
      segment.executable.instantiate(segment.definition);
      ++executable.graph_statistics_.instantiate_count;
    }
  }
  catch (const std::exception &error) { local_error = error.what(); }
  catch (...) { local_error = "unknown CUDA CW graph instantiate failure"; }
  if (or_to_all(!local_error.empty())) {
    discard();
    if (requested == GraphExecutionMode::required)
      throw std::runtime_error(local_error.empty()
                                   ? "CUDA CW graph instantiate failed on another rank"
                                   : local_error);
    return;
  }

  executable.graph_enabled_ = true;
  executable.graph_statistics_.enabled = true;
  executable.graph_statistics_.segment_count = executable.graph_segments_.size();
}

void NvidiaBackend::advance(Executable &raw_executable, BackendState &raw_state, int num_steps) {
  NvidiaBackendState &state = checked_state(raw_state);
  NvidiaExecutable &executable = checked_executable(raw_executable, state);
  if (!or_to_all(num_steps > 0)) return;
  if (count_processors() != 1)
    throw std::invalid_argument("NVIDIA PR2 does not yet support MPI timestepping");
  if (!state.initialized_) throw std::logic_error("cannot advance uninitialized NVIDIA storage");
  if (executable.storage_fingerprint_ != state.fingerprint_)
    throw std::logic_error("NVIDIA executable was compiled for a different storage layout");
  if (state.transfer_failed_)
    throw std::logic_error("NVIDIA execution stream failed; recreate backend state");
  if (executable.has_noisy_updates_ && f_.t < 0)
    throw std::invalid_argument("NVIDIA noisy polarization timestep is negative");

  const FiniteCheckMode finite_mode = finite_check_mode();
  try {
    nvidia::device_scope scope(state.device_);
    const bool account_cw = executable.program_ == StepProgram::solve_cw &&
                            executable.cw_workspace_;
    const auto count_cw_source_upload = [&](size_t bytes) {
      if (!account_cw) return;
      ++executable.cw_workspace_->source_scalar_h2d_calls_;
      executable.cw_workspace_->source_scalar_h2d_bytes_ += bytes;
    };
    const auto upload_material = [&](const NvidiaCompiledOperation &op) {
      char *staging = static_cast<char *>(executable.material_staging_.data());
      for (size_t i = op.material_first; i < op.material_first + op.material_count; ++i) {
        const NvidiaCompiledMaterialRefresh &row = executable.material_refreshes_[i];
        const void *source = f_.array_catalog->resolve_untyped(row.current);
        if (!source) throw std::logic_error("NVIDIA material refresh resolved a null host row");
        host_to_storage(staging + row.staging_offset, source, row.spec, row.spec.elements);
        state.arenas_->copy_from_host_async(row.current.value, 0,
                                            staging + row.staging_offset, row.bytes,
                                            *state.transfer_);
      }
      if (op.material_count) state.transfer_->synchronize();
    };
    const auto publish_legacy_flux = [&](const NvidiaCompiledOperation &op) {
      const size_t bytes = checked_product(op.legacy_flux_count, sizeof(double),
                                           "copying NVIDIA legacy flux result");
      nvidia::copy_device_to_host_async(executable.legacy_flux_host_.data(),
                                        executable.legacy_flux_current_, 0, bytes,
                                        *state.transfer_);
      state.transfer_->synchronize();
      sum_to_all(static_cast<const double *>(executable.legacy_flux_host_.data()),
                 executable.legacy_flux_global_.data(), int(op.legacy_flux_count));
      backend_publish_legacy_flux(f_, executable.legacy_flux_global_.data(),
                                  executable.legacy_flux_global_.size(),
                                  "NVIDIA legacy flux publication");
    };
    const auto complete_finite_check = [&](const NvidiaCompiledOperation &op) {
      nvidia::copy_device_to_host_async(executable.finite_result_host_.data(),
                                        executable.finite_result_, 0, sizeof(uint64_t),
                                        *state.transfer_);
      if (account_cw) {
        ++executable.cw_workspace_->diagnostic_d2h_calls_;
        executable.cw_workspace_->diagnostic_d2h_bytes_ += sizeof(uint64_t);
      }
      state.transfer_->synchronize();
      uint64_t first_bad = std::numeric_limits<uint64_t>::max();
      memcpy(&first_bad, executable.finite_result_host_.data(), sizeof(first_bad));
      if (first_bad == std::numeric_limits<uint64_t>::max()) return;
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
      meep::abort("simulation fields are NaN or Inf (chunk %d, component %d, cmp %d, element %zu)",
                  found->key.chunk, found->key.component_, found->key.cmp, element);
    };
    for (int step = 0; step < num_steps; ++step) {
      bool segment_guard = false;
      if (executable.program_ == StepProgram::solve_cw) {
        f_.step_source_times[0] = cw_source_time(f_.t, f_.dt, 0.0);
        f_.step_source_times[1] = cw_source_time(f_.t, f_.dt, 0.5);
        f_.step_source_times[2] = cw_source_time(f_.t, f_.dt, 1.0);
      }
      else {
        f_.step_source_times[0] = f_.time();
        f_.step_source_times[1] = std::fma(double(f_.t), f_.dt, 0.5 * f_.dt);
        f_.step_source_times[2] = std::fma(double(f_.t), f_.dt, f_.dt);
      }
      const bool finite_due = finite_mode == FiniteCheckMode::step ||
                              (finite_mode == FiniteCheckMode::batch && step + 1 == num_steps);
      if (executable.graph_enabled_) {
        StepScalars scalars = {};
        scalars.abi_version = step_scalars_abi_version;
        scalars.byte_size = sizeof(StepScalars);
        scalars.entry_timestep = f_.t;
        scalars.post_increment_timestep = int64_t(f_.t) + 1;
        scalars.noisy_counter_time = uint64_t(f_.t);
        for (int i = 0; i < 3; ++i) scalars.source_times[i] = f_.step_source_times[i];
        scalars.batch_ordinal = uint64_t(step);
        scalars.batch_count = uint64_t(num_steps);
        scalars.dft_timestep = int64_t(f_.t) + 1;
        scalars.noisy_seed_generation = state.random_seed_snapshot_accepted
                                            ? state.accepted_random_seed.generation
                                            : 0;
        scalars.active_noisy_seed_slot = state.noisy_seed_active_slot_;
        scalars.finite_check_mode = uint32_t(finite_mode);
        scalars.finite_check_due = finite_due ? 1u : 0u;
        scalars.graph_variant = uint32_t(executable.program_ == StepProgram::solve_cw
                                             ? GraphVariantKind::cw_operator
                                             : GraphVariantKind::ordinary);
        for (const StepScalarSlot &slot : executable.graph_program_.scalar_layout.slots) {
          bool value = false;
          if (slot.semantic == StepScalarSemantic::dft_due_predicate)
            value = slot.semantic_index &&
                    scalars.dft_timestep % int64_t(slot.semantic_index) == 0;
          else if (slot.semantic == StepScalarSemantic::guard_predicate)
            value = true;
          if (value && slot.type == StepScalarType::predicate_bit) {
            const size_t word =
                (slot.byte_offset - offsetof(StepScalars, predicate_words)) / sizeof(uint64_t);
            scalars.predicate_words[word] |= uint64_t(1) << slot.bit_offset;
          }
        }
        nvidia::launch_step_scalars_write(executable.step_scalars_, scalars, *state.transfer_);
        ++executable.graph_statistics_.scalar_write_count;
        ++state.graph_statistics_.scalar_write_count;
        bool graph_segment_guard = false;
        for (const GraphScheduleEntry &entry : executable.graph_program_.schedule) {
          if (entry.kind == GraphScheduleKind::segment) {
            if (entry.index >= executable.graph_segments_.size())
              throw std::logic_error("NVIDIA graph schedule segment is out of range");
            executable.graph_segments_[entry.index].executable.launch(*state.transfer_);
            ++executable.graph_statistics_.launch_count;
            ++state.graph_statistics_.launch_count;
            if (account_cw) {
              if (entry.index >= executable.graph_cw_timestep_launches_.size() ||
                  entry.index >= executable.graph_cw_finite_launches_.size())
                throw std::logic_error("NVIDIA CW graph launch accounting is incomplete");
              executable.cw_workspace_->timestep_kernel_launches_ +=
                  executable.graph_cw_timestep_launches_[entry.index];
              executable.cw_workspace_->finite_check_kernel_launches_ +=
                  executable.graph_cw_finite_launches_[entry.index];
            }
            /* Preserve the existing post-dispatch failure seams.  Captured
               nodes no longer pass through their launch wrappers at replay,
               so consume these test-only failures immediately after the
               irreversible graph launch. */
            for (const GraphOperationRef &operation :
                 executable.graph_segments_[entry.index].plan.operations) {
              const NvidiaCompiledOperation &launched =
                  executable.operations_[operation.operation_index];
              if (launched.kind != OpKind::update_polarization) continue;
              for (size_t i = launched.polarization_first;
                   i < launched.polarization_first + launched.polarization_count; ++i) {
                const NvidiaCompiledPolarizationAction &action =
                    executable.polarization_actions_[i];
                if (action.kind == NvidiaCompiledPolarizationAction::Kind::ordinary &&
                    executable.polarization_updates_[action.index].kind ==
                        nvidia::compiled_polarization_update::kind_type::noisy_add &&
                    nvidia::testing::consume_failure_for_testing(
                        nvidia::testing::failure_point::noisy_add))
                  throw std::runtime_error("injected NVIDIA noisy-add postlaunch failure");
                if (action.kind ==
                        NvidiaCompiledPolarizationAction::Kind::multilevel_population &&
                    nvidia::testing::consume_failure_for_testing(
                        nvidia::testing::failure_point::multilevel_population))
                  throw std::runtime_error(
                      "injected NVIDIA multilevel population postlaunch failure");
                if (action.kind ==
                        NvidiaCompiledPolarizationAction::Kind::multilevel_transition &&
                    nvidia::testing::consume_failure_for_testing(
                        nvidia::testing::failure_point::multilevel_transition))
                  throw std::runtime_error(
                      "injected NVIDIA multilevel transition postlaunch failure");
              }
            }
            continue;
          }
          if (entry.index >= executable.graph_program_.boundaries.size())
            throw std::logic_error("NVIDIA graph schedule boundary is out of range");
          const GraphBoundary &boundary = executable.graph_program_.boundaries[entry.index];
          ++executable.graph_statistics_.boundary_count;
          ++state.graph_statistics_.boundary_count;
          if (boundary.first_operation >= executable.operations_.size())
            throw std::logic_error("NVIDIA graph boundary operation is out of range");
          const NvidiaCompiledOperation &op = executable.operations_[boundary.first_operation];
          if (boundary.completion_only) {
            if (boundary.kind == GraphBoundaryKind::legacy_flux_publish)
              publish_legacy_flux(op);
            else if (boundary.kind == GraphBoundaryKind::finite_diagnostic && finite_due)
              complete_finite_check(op);
            continue;
          }
          if (op.guard.kind == GuardKind::segment_boundary && !graph_segment_guard &&
              op.kind != OpKind::phase_material)
            continue;
          switch (op.kind) {
            case OpKind::host_callback:
              execute_host_segment(executable, state, boundary.first_operation,
                                   op.host_segment_index);
              break;
            case OpKind::phase_material:
              graph_segment_guard = f_.phase_material_mix();
              if (graph_segment_guard) upload_material(op);
              break;
            case OpKind::update_material_coefficients:
              for (int i = 0; i < f_.num_chunks; ++i) f_.chunks[i]->s->update_condinv();
              if (graph_segment_guard) upload_material(op);
              break;
            case OpKind::evaluate_source_scalars: {
              if (state.cw_skip_source_evaluation_) break;
              evaluate_supported_source_scalars(f_, op.source_time_offset);
              const SourcePlan &source_plan = f_.descriptors->sources;
              if (source_plan.scalars.size() != executable.source_scalar_count_ ||
                  op.count != executable.source_scalar_count_)
                throw std::logic_error("NVIDIA source scalar block changed after compilation");
              if (!op.count) break;
              nvidia::source_scalar *staging =
                  static_cast<nvidia::source_scalar *>(executable.source_staging_.data()) +
                  op.source_staging_offset;
              for (size_t i = 0; i < op.count; ++i) {
                staging[i].current_real = source_plan.scalars[i].current.real();
                staging[i].current_imag = source_plan.scalars[i].current.imag();
                staging[i].dipole_real = source_plan.scalars[i].dipole.real();
                staging[i].dipole_imag = source_plan.scalars[i].dipole.imag();
              }
              const size_t bytes = checked_product(op.count, sizeof(nvidia::source_scalar),
                                                   "uploading NVIDIA source scalars");
              nvidia::copy_host_to_device_async(executable.source_scalars_, 0, staging, bytes,
                                                 *state.transfer_);
              break;
            }
            case OpKind::increment_time: ++f_.t; break;
            case OpKind::restore_magnetic_fields:
              if (f_.synchronized_magnetic_fields) {
                preflight_magnetic_transition(executable, state, false);
                restore_magnetic_fields(executable, state);
              }
              break;
            case OpKind::synchronize_magnetic_fields:
              if (f_.synchronized_magnetic_fields) {
                preflight_magnetic_transition(executable, state, true);
                synchronize_magnetic_fields(executable, state);
              }
              break;
            default:
              if (!launch_device_operation(executable, state, op, NULL, false))
                throw std::logic_error("NVIDIA graph boundary has an invalid host operation");
              break;
          }
        }
        /* PR6.2 deliberately retains this synchronization: one pinned source
           staging allocation is reused by the next step. */
        state.transfer_->synchronize();
        continue;
      }
      for (size_t oi = 0; oi < executable.operations_.size(); ++oi) {
        const NvidiaCompiledOperation &op = executable.operations_[oi];
        if (op.guard.kind == GuardKind::segment_boundary && !segment_guard) continue;
        if (op.kind == OpKind::finite_value_check && account_cw && cw_profile_mode_requested())
          continue;
        if (op.kind == OpKind::finite_value_check && !finite_due) continue;
        if (launch_device_operation(executable, state, op, NULL, account_cw)) {
          if (op.kind == OpKind::update_flux) publish_legacy_flux(op);
          else if (op.kind == OpKind::finite_value_check) complete_finite_check(op);
          continue;
        }
        switch (op.kind) {
          case OpKind::host_callback:
            execute_host_segment(executable, state, oi, op.host_segment_index);
            oi = checked_add(oi, executable.host_segments_[op.host_segment_index]
                                     .segment.operation_count,
                             "skipping NVIDIA host-segment operations");
            break;
          case OpKind::restore_magnetic_fields:
            if (f_.synchronized_magnetic_fields) {
              preflight_magnetic_transition(executable, state, false);
              restore_magnetic_fields(executable, state);
            }
            break;
          case OpKind::phase_material:
            segment_guard = f_.phase_material_mix();
            if (segment_guard) upload_material(op);
            break;
          case OpKind::update_material_coefficients:
            for (int i = 0; i < f_.num_chunks; ++i)
              f_.chunks[i]->s->update_condinv();
            if (segment_guard) upload_material(op);
            break;
          case OpKind::evaluate_source_scalars: {
            if (state.cw_skip_source_evaluation_) break;
            evaluate_supported_source_scalars(f_, op.source_time_offset);
            const SourcePlan &source_plan = f_.descriptors->sources;
            if (source_plan.scalars.size() != executable.source_scalar_count_ ||
                op.count != executable.source_scalar_count_)
              throw std::logic_error("NVIDIA source scalar block changed after compilation");
            if (!op.count) break;
            nvidia::source_scalar *staging =
                static_cast<nvidia::source_scalar *>(executable.source_staging_.data()) +
                op.source_staging_offset;
            for (size_t i = 0; i < op.count; ++i) {
              staging[i].current_real = source_plan.scalars[i].current.real();
              staging[i].current_imag = source_plan.scalars[i].current.imag();
              staging[i].dipole_real = source_plan.scalars[i].dipole.real();
              staging[i].dipole_imag = source_plan.scalars[i].dipole.imag();
            }
            const size_t upload_bytes =
                checked_product(op.count, sizeof(nvidia::source_scalar),
                                "uploading NVIDIA source scalars");
            nvidia::copy_host_to_device_async(executable.source_scalars_, 0, staging,
                                              upload_bytes, *state.transfer_);
            count_cw_source_upload(upload_bytes);
            break;
          }
          case OpKind::increment_time: ++f_.t; break;
          case OpKind::synchronize_magnetic_fields:
            if (f_.synchronized_magnetic_fields) {
              preflight_magnetic_transition(executable, state, true);
              synchronize_magnetic_fields(executable, state);
            }
            break;
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
  if (executable.material_phase_active && f_.phasein_time <= 0) {
    executable.material_phase_active = false;
    state.material_phase_active = false;
  }
}

void NvidiaBackend::execute_host_segment(NvidiaExecutable &executable,
                                         NvidiaBackendState &state,
                                         size_t operation_index, size_t segment_index) {
  if (segment_index >= executable.host_segments_.size())
    throw std::logic_error("NVIDIA host segment index is out of range");
  if (operation_index > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("NVIDIA host segment operation index overflow");
  NvidiaCompiledHostSegment &segment = executable.host_segments_[segment_index];
  HostCustomFallbackSession session(*this, uint32_t(operation_index), segment.segment);
  ++state.host_fallback_statistics_.segment_executions;

  /* Resolve pointer-free identities immediately before entering the host
     session.  The resolved pointers are deliberately not retained: a later
     rebuild must recompile, and an identity mismatch fails before any D2H. */
  for (const HostCallbackDescriptor &callback : segment.callbacks) {
    ResolvedHostCallback resolved;
    std::string error;
    const size_t capacity = segment.resolution_layout.capacity();
    if (!resolve_host_callback(f_, callback, resolved, segment.resolution_layout, &error))
      throw std::invalid_argument(error.empty() ? "NVIDIA host callback identity is stale"
                                                : error);
    if (segment.resolution_layout.capacity() != capacity)
      ++state.host_fallback_statistics_.steady_capacity_growths;
    ++state.host_fallback_statistics_.callback_resolutions;
  }

  char *staging = static_cast<char *>(state.staging_.data());
  for (const NvidiaCompiledHostTransfer &transfer : segment.transfers) {
    if (transfer.mode == AccessMode::write) continue;
    state.arenas_->copy_to_host_async(staging + transfer.staging_offset,
                                      transfer.array.id.value, transfer.storage_offset,
                                      transfer.storage_bytes, *state.transfer_);
    session.record_download(transfer.storage_bytes);
    ++state.host_fallback_statistics_.device_to_host_calls;
    state.host_fallback_statistics_.device_to_host_bytes += transfer.storage_bytes;
  }
  /* Even a segment whose declared catalog union is write-only may consult or
     mutate opaque host-owned state.  It still forms a hard ordering boundary
     with all previously enqueued device work. */
  state.transfer_->synchronize();
  ++state.host_fallback_statistics_.synchronizations;
  for (const NvidiaCompiledHostTransfer &transfer : segment.transfers) {
    if (transfer.mode == AccessMode::write) continue;
    char *host = static_cast<char *>(f_.array_catalog->resolve_untyped(transfer.array.id));
    if (!host) throw std::logic_error("NVIDIA host segment resolved a null native row");
    storage_to_host(host + transfer.host_offset, staging + transfer.staging_offset,
                    transfer.spec, transfer.array.elements);
  }
  if (nvidia::testing::consume_failure_for_testing(
          nvidia::testing::failure_point::host_segment_after_download))
    throw std::runtime_error("injected NVIDIA host-segment post-download failure");

  size_t callback_count = 0;
  for (const HostCallbackDescriptor &callback : segment.callbacks)
    if (segment.segment.phase != HostSegmentPhase::constitutive || callback.has_internal_state)
      ++callback_count;
  session.enter_callback(callback_count);
  f_.execute_step_plan(segment.host_plan, 0);
  if (nvidia::testing::consume_failure_for_testing(
          nvidia::testing::failure_point::host_segment_after_callback))
    throw std::runtime_error("injected NVIDIA host-segment post-callback failure");

  for (const NvidiaCompiledHostTransfer &transfer : segment.transfers) {
    if (transfer.mode == AccessMode::read) continue;
    const char *host = static_cast<const char *>(
        f_.array_catalog->resolve_untyped(transfer.array.id));
    if (!host) throw std::logic_error("NVIDIA host segment resolved a null native row");
    host_to_storage(staging + transfer.staging_offset, host + transfer.host_offset,
                    transfer.spec, transfer.array.elements);
    state.arenas_->copy_from_host_async(transfer.array.id.value, transfer.storage_offset,
                                        staging + transfer.staging_offset,
                                        transfer.storage_bytes, *state.transfer_);
    session.record_upload(transfer.storage_bytes);
    ++state.host_fallback_statistics_.host_to_device_calls;
    state.host_fallback_statistics_.host_to_device_bytes += transfer.storage_bytes;
  }
  if (nvidia::testing::consume_failure_for_testing(
          nvidia::testing::failure_point::host_segment_after_upload))
    throw std::runtime_error("injected NVIDIA host-segment post-upload failure");
  session.complete();
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
  NvidiaBackendState *current = current_state();
  if (!current) throw std::logic_error("NVIDIA backend has no active state");
  NvidiaBackendState &state = *current;
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

bool NvidiaBackend::supports_adjoint_gradient(const AdjointGradientRequest &request,
                                              std::string &why) const {
  try {
    validate_adjoint_request(request);
    validate_adjoint_snapshot_freshness(*request.forward, f_);
    validate_adjoint_snapshot_freshness(*request.adjoint, f_);
    const MaterialIR *ir = material_ir_for(f_);
    if (!ir || request.material_recipe.get() != f_.material_ir.get() ||
        request.material_signature != ir->signature ||
        request.material_layout_signature != ir->layout_signature)
      throw std::invalid_argument("adjoint material recipe is stale");
    if (request.disposition != AdjointSupportDisposition::device_native ||
        request.support_reasons != adjoint_support_none)
      throw std::invalid_argument("adjoint request is outside the native support matrix");
    for (const AdjointDftTerm &term : request.terms)
      if (!is_valid(term.adjoint.id))
        throw std::invalid_argument("adjoint request has no resident adjoint array");
    why.clear();
    return true;
  }
  catch (const std::exception &error) {
    why = error.what();
    return false;
  }
}

void NvidiaBackend::compute_adjoint_gradient(const AdjointGradientRequest &request,
                                             double *local_result, size_t result_count,
                                             BackendState &raw_state) {
  NvidiaBackendState &state = checked_state(raw_state);
  std::string why;
  if (!supports_adjoint_gradient(request, why)) throw std::invalid_argument(why);
  if (!local_result || result_count != request.output_count)
    throw std::invalid_argument("NVIDIA adjoint output-count mismatch");
  if (state.transfer_failed_)
    throw std::logic_error("NVIDIA transfer stream failed; recreate backend state");
  if (request.contributions.empty()) {
    std::fill(local_result, local_result + result_count, 0.0);
    return;
  }

  const size_t forward_base_count =
      checked_add(request.forward->terms.size(), size_t(1),
                  "sizing NVIDIA adjoint forward offsets");
  const size_t adjoint_base_count =
      checked_add(request.adjoint->terms.size(), size_t(1),
                  "sizing NVIDIA adjoint reverse offsets");
  std::vector<size_t> forward_base(forward_base_count, 0);
  std::vector<size_t> adjoint_base(adjoint_base_count, 0);
  std::vector<nvidia::adjoint_complex64> forward_values;
  std::vector<nvidia::adjoint_complex64> adjoint_values;
  for (size_t i = 0; i < request.forward->terms.size(); ++i) {
    forward_base[i] = forward_values.size();
    for (const std::complex<double> value : request.forward->terms[i].values)
      forward_values.push_back(nvidia::adjoint_complex64{value.real(), value.imag()});
  }
  forward_base.back() = forward_values.size();
  for (size_t i = 0; i < request.adjoint->terms.size(); ++i) {
    adjoint_base[i] = adjoint_values.size();
    for (const std::complex<double> value : request.adjoint->terms[i].values)
      adjoint_values.push_back(nvidia::adjoint_complex64{value.real(), value.imag()});
  }
  adjoint_base.back() = adjoint_values.size();

  const size_t offset_count =
      checked_add(result_count, size_t(1), "sizing NVIDIA adjoint output offsets");
  std::vector<uint64_t> offsets(offset_count, 0);
  for (const AdjointGradientContribution &source : request.contributions) {
    if (offsets[source.output_index + 1] == UINT64_MAX)
      throw std::overflow_error("NVIDIA adjoint output contribution count overflows");
    ++offsets[source.output_index + 1];
  }
  for (size_t i = 0; i < result_count; ++i) {
    if (offsets[i + 1] > UINT64_MAX - offsets[i])
      throw std::overflow_error("NVIDIA adjoint contribution prefix overflows");
    offsets[i + 1] += offsets[i];
  }
  std::vector<uint64_t> cursor(offsets.begin(), offsets.end() - 1);
  std::vector<nvidia::adjoint_contribution> contributions(request.contributions.size());
  for (const AdjointGradientContribution &source : request.contributions) {
    nvidia::adjoint_contribution destination = {};
    destination.adjoint_index =
        adjoint_base[source.adjoint_snapshot_term] + source.adjoint_value_index;
    destination.forward_count = source.forward_value_count;
    for (uint32_t i = 0; i < source.forward_value_count; ++i) {
      destination.forward_index[i] =
          source.forward_value_indices[i] == std::numeric_limits<size_t>::max()
              ? UINT64_MAX
              : forward_base[source.forward_snapshot_term] + source.forward_value_indices[i];
      destination.forward_weight[i] = source.forward_weights[i];
    }
    destination.adjoint_coefficient_real = source.adjoint_coefficient.real();
    destination.adjoint_coefficient_imag = source.adjoint_coefficient.imag();
    destination.material_coefficient_real = source.material_coefficient.real();
    destination.material_coefficient_imag = source.material_coefficient.imag();
    destination.accumulation_scale = source.accumulation_scale;
    contributions[size_t(cursor[source.output_index]++)] = destination;
  }

  const size_t forward_bytes = checked_product(forward_values.size(),
                                               sizeof(nvidia::adjoint_complex64),
                                               "sizing NVIDIA adjoint forward snapshot");
  const size_t adjoint_bytes = checked_product(adjoint_values.size(),
                                               sizeof(nvidia::adjoint_complex64),
                                               "sizing NVIDIA adjoint field snapshot");
  const size_t contribution_bytes = checked_product(contributions.size(),
                                                    sizeof(nvidia::adjoint_contribution),
                                                    "sizing NVIDIA adjoint contributions");
  const size_t offset_bytes = checked_product(offsets.size(), sizeof(uint64_t),
                                              "sizing NVIDIA adjoint offsets");
  const size_t result_bytes = checked_product(result_count, sizeof(double),
                                              "sizing NVIDIA adjoint result");
  nvidia::device_buffer forward_device(forward_bytes, device_);
  nvidia::device_buffer adjoint_device(adjoint_bytes, device_);
  nvidia::device_buffer contribution_device(contribution_bytes, device_);
  nvidia::device_buffer offset_device(offset_bytes, device_);
  nvidia::device_buffer result_device(result_bytes, device_);
  state.ensure_staging(result_bytes);

  bool enqueued = false;
  try {
    nvidia::copy_host_to_device_async(forward_device, 0, forward_values.data(), forward_bytes,
                                      *state.transfer_);
    enqueued = true;
    nvidia::copy_host_to_device_async(adjoint_device, 0, adjoint_values.data(), adjoint_bytes,
                                      *state.transfer_);
    nvidia::copy_host_to_device_async(contribution_device, 0, contributions.data(),
                                      contribution_bytes, *state.transfer_);
    nvidia::copy_host_to_device_async(offset_device, 0, offsets.data(), offset_bytes,
                                      *state.transfer_);
    nvidia::adjoint_launch launch = {
        static_cast<const nvidia::adjoint_complex64 *>(forward_device.opaque_handle()),
        static_cast<const nvidia::adjoint_complex64 *>(adjoint_device.opaque_handle()),
        static_cast<const nvidia::adjoint_contribution *>(contribution_device.opaque_handle()),
        static_cast<const uint64_t *>(offset_device.opaque_handle()),
        static_cast<double *>(result_device.opaque_handle()),
        forward_values.size(), adjoint_values.size(), contributions.size(), result_count};
    nvidia::launch_adjoint_gradient(launch, *state.transfer_);
    nvidia::copy_device_to_host_async(state.staging_.data(), result_device, 0, result_bytes,
                                      *state.transfer_);
    state.transfer_->synchronize();
    std::copy(static_cast<const double *>(state.staging_.data()),
              static_cast<const double *>(state.staging_.data()) + result_count, local_result);
  }
  catch (...) {
    if (enqueued) {
      state.transfer_failed_ = true;
      poison();
    }
    throw;
  }
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
  try { state.transfer_->synchronize(); }
  catch (...) { state.transfer_failed_ = true; throw; }
  if (!state.device_authoritative_) return;
  if (!f_.array_catalog || f_.array_catalog->size() > state.plan_.arrays.size())
    throw std::logic_error("cannot migrate NVIDIA state into a changed host catalog");

  std::set<uint32_t> host_authoritative_gamma_inv;
  std::vector<PolarizationDescriptor> live_descriptors;
  build_polarization_descriptors(f_, live_descriptors);
  for (const PolarizationDescriptor &descriptor : live_descriptors)
    if (descriptor.kind == SusceptibilityKind::multilevel) {
      if (!is_valid(descriptor.multilevel_gamma_inv) ||
          descriptor.multilevel_gamma_inv.value >= state.plan_.arrays.size())
        throw std::logic_error("live multilevel GammaInv ArrayId is out of range");
      host_authoritative_gamma_inv.insert(descriptor.multilevel_gamma_inv.value);
    }

  struct MigrationRow { ArraySpec spec; size_t offset; void *destination; };
  std::vector<MigrationRow> rows;
  size_t total_bytes = 0;
  for (size_t i = 0; i < f_.array_catalog->size(); ++i) {
    const ArraySpec &spec = state.plan_.arrays[i];
    if (is_valid(spec.alias_of) || spec.role == array_role::material ||
        host_authoritative_gamma_inv.count(uint32_t(i)))
      continue;
    void *destination = f_.array_catalog->resolve_untyped(spec.id);
    if (!destination) throw std::logic_error("NVIDIA migration found a null host allocation");
    const size_t bytes = storage_bytes(spec);
    if (bytes > std::numeric_limits<size_t>::max() - total_bytes)
      throw std::overflow_error("NVIDIA migration staging size overflow");
    rows.push_back(MigrationRow{spec, total_bytes, destination});
    total_bytes += bytes;
  }

  nvidia::pinned_buffer staged(total_bytes);
  nvidia::device_scope scope(state.device_);
  nvidia::stream migration;
  for (const MigrationRow &row : rows) {
    state.arenas_->copy_to_host_async(static_cast<char *>(staged.data()) + row.offset,
                                      row.spec.id.value, 0, storage_bytes(row.spec), migration);
  }
  if (nvidia::testing::consume_failure_for_testing(
          nvidia::testing::failure_point::state_rebuild_sync))
    throw std::runtime_error("injected NVIDIA rebuild synchronization failure");
  migration.synchronize();

  /* Commit host mirrors only after the entire transfer batch succeeds. */
  for (const MigrationRow &row : rows)
    storage_to_host(row.destination, static_cast<const char *>(staged.data()) + row.offset,
                    row.spec, row.spec.elements);
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

NvidiaCwStatistics NvidiaBackend::cw_statistics_for_testing() const {
  NvidiaBackendState *state = current_state();
  return state ? state->cw_statistics_ : NvidiaCwStatistics();
}

void NvidiaBackend::clear_cw_graphs_for_testing() {
  NvidiaBackendState *state = current_state();
  NvidiaCwExecutable *cw =
      state ? dynamic_cast<NvidiaCwExecutable *>(state->cw_executable) : NULL;
  if (!cw) throw std::logic_error("NVIDIA CW graph cleanup test has no resident executable");
  clear_cw_device_graphs(*cw);
}

NvidiaHostFallbackStatistics NvidiaBackend::host_fallback_statistics_for_testing() const {
  NvidiaBackendState *state = current_state();
  return state ? state->host_fallback_statistics_ : NvidiaHostFallbackStatistics();
}

NvidiaMaterialInitializationStatistics
NvidiaBackend::material_initialization_statistics_for_testing() const {
  NvidiaBackendState *state = current_state();
  return state ? state->material_initialization_statistics_
               : NvidiaMaterialInitializationStatistics();
}

NvidiaGraphStatistics NvidiaBackend::graph_statistics_for_testing() const {
  NvidiaBackendState *state = current_state();
  return state ? state->graph_statistics_ : NvidiaGraphStatistics();
}

NvidiaExecutableCacheStatistics NvidiaBackend::executable_cache_statistics_for_testing() const {
  NvidiaExecutableCacheStatistics result;
  NvidiaBackendState *state = current_state();
  if (!state) return result;
  result.executable_build_count = state->executable_build_count_;
  result.source_value_reuse_count = state->source_value_reuse_count_;
  NvidiaExecutable *ordinary = dynamic_cast<NvidiaExecutable *>(f_.executable);
  if (ordinary && ordinary->owner_ == this && ordinary->state_token_ == state->state_token_)
    result.ordinary_resource_generation = ordinary->resource_generation_;
  NvidiaCwExecutable *cw = dynamic_cast<NvidiaCwExecutable *>(state->cw_executable);
  if (cw && cw->owner_ == this && cw->state_token_ == state->state_token_)
    result.cw_resource_generation = cw->resource_generation_;
  return result;
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
