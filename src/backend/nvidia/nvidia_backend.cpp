/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/nvidia/nvidia_backend.hpp"

#include <stdint.h>
#include <string.h>

#include <complex>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "backend/initialization_plan.hpp"
#include "backend/diagnostics.hpp"
#include "backend/halo_plan.hpp"
#include "backend/nvidia/arena.hpp"
#include "backend/nvidia/runtime.hpp"
#include "backend/nvidia/nvidia_step.hpp"
#include "meep_internals.hpp"

namespace meep {

namespace {

size_t checked_product(size_t left, size_t right, const char *what) {
  if (left && right > std::numeric_limits<size_t>::max() / left)
    throw std::overflow_error(std::string("overflow while ") + what);
  return left * right;
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
  NvidiaBackendState(NvidiaBackend *owner, StoragePlan plan, int device)
      : owner_(owner), plan_(plan), layout_(allocation_requests_for(plan_)), device_(device),
        fingerprint_(storage_fingerprint(plan_)), initialized_(false), transfer_failed_(false),
        device_authoritative_(false) {
    nvidia::device_scope scope(device_);
    transfer_.reset(new nvidia::stream);
    arenas_.reset(new nvidia::device_arenas(layout_, device_));
  }

  ~NvidiaBackendState() override {
    if (owner_) owner_->release_state(this);
  }

  void detach_owner() { owner_ = NULL; }

  void ensure_staging(size_t bytes) {
    if (staging_.size() < bytes) staging_.allocate(bytes);
  }

  NvidiaBackend *owner_;
  StoragePlan plan_;
  nvidia::arena_plan layout_;
  int device_;
  uint64_t fingerprint_;
  bool initialized_;
  bool transfer_failed_;
  bool device_authoritative_;
  std::unique_ptr<nvidia::device_arenas> arenas_;
  std::unique_ptr<nvidia::stream> transfer_;
  nvidia::pinned_buffer staging_;
};

struct NvidiaCompiledOperation {
  OpKind kind;
  size_t first;
  size_t count;
};

class NvidiaExecutable : public Executable {
public:
  NvidiaExecutable(const NvidiaBackend *owner, uint64_t signature, uint64_t storage_fingerprint,
                   const std::vector<NvidiaCompiledOperation> &operations,
                   const std::vector<nvidia::curl_launch> &curl_updates,
                   const std::vector<nvidia::constitutive_launch> &constitutive_updates,
                   const std::vector<nvidia::zero_launch> &zero_updates)
      : owner_(owner), signature_(signature), storage_fingerprint_(storage_fingerprint),
        operations_(operations), curl_updates_(curl_updates),
        constitutive_updates_(constitutive_updates), zero_updates_(zero_updates) {}

  const NvidiaBackend *owner_;
  uint64_t signature_;
  uint64_t storage_fingerprint_;
  std::vector<NvidiaCompiledOperation> operations_;
  std::vector<nvidia::curl_launch> curl_updates_;
  std::vector<nvidia::constitutive_launch> constitutive_updates_;
  std::vector<nvidia::zero_launch> zero_updates_;
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

const void *optional_device_address(NvidiaBackendState &state, ArrayId id,
                                    nvidia::scalar_precision precision, const char *what) {
  if (!is_valid(id)) return NULL;
  require_same_precision(state.plan_, id, precision, what);
  return state.arenas_->resolve(id.value).address;
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

nvidia::curl_launch compile_curl(const CurlUpdate &source, NvidiaBackendState &state) {
  const uint32_t supported_variants = curl_has_second_derivative;
  if (source.region.variant_key & ~supported_variants)
    throw std::invalid_argument("curl descriptor requires PML, conductivity, or BFAST");
  if (is_valid(source.target_u) || is_valid(source.conductivity) || is_valid(source.condinv) ||
      is_valid(source.target_cond) || is_valid(source.pml.sig) || is_valid(source.pml.kap) ||
      is_valid(source.pml.siginv) || is_valid(source.pml_u.sig) || is_valid(source.pml_u.kap) ||
      is_valid(source.pml_u.siginv))
    throw std::invalid_argument("curl descriptor contains unsupported auxiliary arrays");

  nvidia::curl_launch result;
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
  result.dtdx = source.dtdx;

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
  return result;
}

nvidia::constitutive_launch compile_constitutive(const ConstitutiveUpdate &source,
                                                 NvidiaBackendState &state) {
  if (source.region.variant_key)
    throw std::invalid_argument(
        "constitutive descriptor requires anisotropy, PML, nonlinearity, or polarization");
  if (is_valid(source.offdiagonal1) || is_valid(source.offdiagonal2) || is_valid(source.chi2) ||
      is_valid(source.chi3) || is_valid(source.target_w) || is_valid(source.previous_w) ||
      is_valid(source.pml.sig) || is_valid(source.pml.kap) || is_valid(source.pml.siginv))
    throw std::invalid_argument("constitutive descriptor contains unsupported auxiliary arrays");

  nvidia::constitutive_launch result;
  result.region = flat_region_for(source.region);
  result.precision = scalar_precision_for(state.plan_, source.target, "constitutive target");
  result.target = device_address(state, source.target, "constitutive target");
  result.primary =
      optional_device_address(state, source.primary, result.precision, "constitutive primary");
  result.diagonal =
      optional_device_address(state, source.diagonal, result.precision, "constitutive diagonal");
  if (!result.primary) throw std::invalid_argument("constitutive descriptor has no primary field");

  const ptrdiff_t region_max = checked_region_max(result.region);
  validate_index_range(state.plan_, source.target, ptrdiff_t(result.region.base), region_max,
                       "constitutive target");
  validate_index_range(state.plan_, source.primary, ptrdiff_t(result.region.base), region_max,
                       "constitutive primary");
  if (is_valid(source.diagonal))
    validate_index_range(state.plan_, source.diagonal, ptrdiff_t(result.region.base), region_max,
                         "constitutive diagonal");
  return result;
}

nvidia::zero_launch compile_zero(const SlabRef &source, NvidiaBackendState &state) {
  nvidia::zero_launch result;
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

bool has_polarization(const fields &f) {
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    FOR_FIELD_TYPES(ft) if (f.chunks[chunk]->pol[ft]) return true;
  }
  return false;
}

bool has_dfts(const fields &f) {
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    if (f.chunks[chunk]->is_mine() && f.chunks[chunk]->dft_chunks) return true;
  return false;
}

bool has_magnetic_backups(const StoragePlan &plan) {
  for (size_t i = 0; i < plan.keys.size(); ++i) {
    const int kind = plan.keys[i].kind;
    if (kind == int(array_kind::f_backup) || kind == int(array_kind::f_u_backup) ||
        kind == int(array_kind::f_w_backup) || kind == int(array_kind::f_cond_backup) ||
        kind == int(array_kind::f_bfast_backup))
      return true;
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
      active_state_(NULL) {
  if (device_ < 0) throw std::invalid_argument("NVIDIA backend requires a resolved device ID");
  device_memory_bytes_ = nvidia::properties_for_device(device_).total_memory;
}

NvidiaBackend::~NvidiaBackend() {
  if (active_state_) active_state_->detach_owner();
}

BackendState *NvidiaBackend::create_state(const StoragePlan &plan) {
  std::unique_ptr<NvidiaBackendState> state;
  std::string local_error;
  try {
    if (active_state_) throw std::logic_error("NVIDIA backend already owns an active state");
    StoragePlan device_plan = plan;
    apply_precision_policy(device_plan, policy_for(options_.precision));
    std::string why;
    if (!validate_alias_precisions(device_plan, why))
      throw std::invalid_argument(std::string("invalid NVIDIA storage aliases: ") + why);
    state.reset(new NvidiaBackendState(this, device_plan, device_));
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
  active_state_ = state.get();
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
    if (options_.precision == precision_policy_kind::mixed)
      throw std::invalid_argument("NVIDIA PR2 supports precision=native and precision=f32 only");
    if (count_processors() != 1)
      throw std::invalid_argument("NVIDIA PR2 does not yet support MPI timestepping");
    if (f_.gv.dim == Dcyl || f_.m != 0.0 || f_.beta != 0.0)
      throw std::invalid_argument("NVIDIA PR2 supports Cartesian fields with m=beta=0 only");
    if (f_.is_phasing())
      throw std::invalid_argument("NVIDIA PR2 does not support material phasing");
    if (f_.sources)
      throw std::invalid_argument("NVIDIA PR2 source-free slice does not support sources");
    if (f_.fluxes || has_dfts(f_))
      throw std::invalid_argument("NVIDIA PR2 source-free slice does not support monitors");
    if (has_polarization(f_))
      throw std::invalid_argument("NVIDIA PR2 source-free slice does not support dispersion");
    if (has_magnetic_backups(state.plan_))
      throw std::invalid_argument("NVIDIA PR2 does not support synchronized magnetic fields");
    if (finite_check_mode() != FiniteCheckMode::off)
      throw std::invalid_argument(
          "NVIDIA PR2 requires MEEP_FINITE_CHECK=off until device diagnostics land");
    if (!connections_are_current(f_))
      throw std::invalid_argument(
          "NVIDIA PR2 requires Phase 1 to finalize halo topology before backend compilation");
    if (!f_.halos || !f_.array_catalog)
      throw std::logic_error("NVIDIA PR2 requires prepared halo and storage plans");

    std::vector<NvidiaCompiledOperation> operations;
    std::vector<nvidia::curl_launch> curl_updates;
    std::vector<nvidia::constitutive_launch> constitutive_updates;
    std::vector<nvidia::zero_launch> zero_updates;
    operations.reserve(plan.operations.size());

    for (size_t oi = 0; oi < plan.operations.size(); ++oi) {
      const Operation &op = plan.operations[oi];
      NvidiaCompiledOperation compiled{op.kind, 0, 0};
      switch (op.kind) {
        case OpKind::update_db: {
          if (size_t(op.descriptor_index) + op.descriptor_count > plan.db_updates.size()) {
            set_reason(local_error, oi, "curl descriptor span is out of range");
            break;
          }
          compiled.first = curl_updates.size();
          for (size_t i = op.descriptor_index;
               i < size_t(op.descriptor_index) + op.descriptor_count; ++i)
            curl_updates.push_back(compile_curl(plan.db_updates[i], state));
          compiled.count = curl_updates.size() - compiled.first;
          if (!compiled.count) set_reason(local_error, oi, "curl descriptor span is empty");
          break;
        }
        case OpKind::update_eh: {
          if (size_t(op.descriptor_index) + op.descriptor_count > plan.eh_updates.size()) {
            set_reason(local_error, oi, "constitutive descriptor span is out of range");
            break;
          }
          compiled.first = constitutive_updates.size();
          for (size_t i = op.descriptor_index;
               i < size_t(op.descriptor_index) + op.descriptor_count; ++i)
            constitutive_updates.push_back(compile_constitutive(plan.eh_updates[i], state));
          compiled.count = constitutive_updates.size() - compiled.first;
          /* Vacuum H==B and E==D aliases legitimately produce no E/H work. */
          break;
        }
        case OpKind::transfer_halo: {
          for (size_t i = 0; i < f_.halos->plans.size(); ++i)
            if (f_.halos->plans[i].ft == op.ft && f_.halos->plans[i].block_elements) {
              set_reason(local_error, oi,
                         "same-rank and remote halo exchange is deferred beyond this slice");
              break;
            }
          if (!local_error.empty()) break;
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
          break;
        }
        case OpKind::restore_magnetic_fields:
        case OpKind::update_material_coefficients:
        case OpKind::apply_sources:
        case OpKind::update_polarization:
        case OpKind::increment_time:
        case OpKind::synchronize_magnetic_fields:
        case OpKind::finite_value_check: break;

        case OpKind::phase_material:
        case OpKind::evaluate_source_scalars:
        case OpKind::zero_boundary:
        case OpKind::pack_halo:
        case OpKind::exchange_local:
        case OpKind::unpack_halo:
        case OpKind::update_flux_half:
        case OpKind::update_flux:
        case OpKind::update_dft:
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
      executable.reset(new NvidiaExecutable(this, plan.signature, state.fingerprint_, operations,
                                            curl_updates, constitutive_updates, zero_updates));
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

void NvidiaBackend::advance(Executable &raw_executable, BackendState &raw_state, int num_steps) {
  NvidiaExecutable &executable = checked_executable(raw_executable);
  NvidiaBackendState &state = checked_state(raw_state);
  if (!or_to_all(num_steps > 0)) return;
  if (!state.initialized_) throw std::logic_error("cannot advance uninitialized NVIDIA storage");
  if (executable.storage_fingerprint_ != state.fingerprint_)
    throw std::logic_error("NVIDIA executable was compiled for a different storage layout");
  if (state.transfer_failed_)
    throw std::logic_error("NVIDIA execution stream failed; recreate backend state");

  try {
    nvidia::device_scope scope(state.device_);
    for (int step = 0; step < num_steps; ++step) {
      for (size_t oi = 0; oi < executable.operations_.size(); ++oi) {
        const NvidiaCompiledOperation &op = executable.operations_[oi];
        switch (op.kind) {
          case OpKind::update_db:
            for (size_t i = op.first; i < op.first + op.count; ++i)
              nvidia::launch_curl(executable.curl_updates_[i], *state.transfer_);
            break;
          case OpKind::update_eh:
            for (size_t i = op.first; i < op.first + op.count; ++i)
              nvidia::launch_constitutive(executable.constitutive_updates_[i], *state.transfer_);
            break;
          case OpKind::transfer_halo:
            for (size_t i = op.first; i < op.first + op.count; ++i)
              nvidia::launch_zero(executable.zero_updates_[i], *state.transfer_);
            break;
          case OpKind::increment_time: ++f_.t; break;
          default: break; // capability validation proved these operations are no-ops
        }
      }
      state.transfer_->synchronize();
    }
  }
  catch (...) {
    state.transfer_failed_ = true;
    throw;
  }
  state.device_authoritative_ = true;
}

void NvidiaBackend::read(ArrayRef ref, void *host_buffer, size_t bytes) {
  if (!active_state_) throw std::logic_error("NVIDIA backend has no active state");
  NvidiaBackendState &state = *active_state_;
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
  if (!active_state_) throw std::logic_error("NVIDIA backend has no active state");
  NvidiaBackendState &state = *active_state_;
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

    /* PR1 has no device timestep kernels, so explicit writes are the only way
       device values can diverge from the compatibility host catalog. Mirror
       them until PR2 introduces an explicit device-authority/migration epoch;
       this makes initialization refresh and storage rebuild lossless today. */
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

void NvidiaBackend::synchronize() {
  if (!active_state_) return;
  if (active_state_->transfer_failed_)
    throw std::logic_error("NVIDIA transfer stream failed; recreate backend state");
  try {
    active_state_->transfer_->synchronize();
  }
  catch (...) {
    active_state_->transfer_failed_ = true;
    throw;
  }
}

void NvidiaBackend::prepare_state_rebuild(BackendState &raw_state, DirtyMask) {
  NvidiaBackendState &state = checked_state(raw_state);
  synchronize();
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
  if (!state || state != active_state_)
    throw std::invalid_argument("NVIDIA backend received state owned by another backend");
  return *state;
}

NvidiaExecutable &NvidiaBackend::checked_executable(Executable &raw_executable) const {
  NvidiaExecutable *executable = dynamic_cast<NvidiaExecutable *>(&raw_executable);
  if (!executable || executable->owner_ != this)
    throw std::invalid_argument("NVIDIA backend received an executable of the wrong type");
  return *executable;
}

void NvidiaBackend::release_state(NvidiaBackendState *state) {
  if (active_state_ == state) active_state_ = NULL;
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
