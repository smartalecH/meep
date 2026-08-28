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
#include "backend/nvidia/arena.hpp"
#include "backend/nvidia/runtime.hpp"
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
        fingerprint_(storage_fingerprint(plan_)), initialized_(false), transfer_failed_(false) {
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
  std::unique_ptr<nvidia::device_arenas> arenas_;
  std::unique_ptr<nvidia::stream> transfer_;
  nvidia::pinned_buffer staging_;
};

class NvidiaExecutable : public Executable {
public:
  NvidiaExecutable(const NvidiaBackend *owner, uint64_t signature, uint64_t storage_fingerprint,
                   size_t operation_count)
      : owner_(owner), signature_(signature), storage_fingerprint_(storage_fingerprint),
        operation_count_(operation_count) {}

  const NvidiaBackend *owner_;
  uint64_t signature_;
  uint64_t storage_fingerprint_;
  size_t operation_count_;
};

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
  const bool local_unsupported = !plan.operations.empty();
  if (or_to_all(local_unsupported)) {
    if (plan.operations.empty())
      throw std::runtime_error(
          "NVIDIA PR1 cannot compile because another rank has unsupported timestep operations");
    const size_t i = 0;
    std::ostringstream message;
    message << "NVIDIA PR1 has no physics kernels: unsupported timestep operation "
            << op_kind_name(plan.operations[i].kind) << " at index " << i << " (descriptor "
            << plan.operations[i].descriptor_index << ")";
    throw std::runtime_error(message.str());
  }
  return new NvidiaExecutable(this, plan.signature, state.fingerprint_, 0);
}

void NvidiaBackend::advance(Executable &raw_executable, BackendState &raw_state, int num_steps) {
  NvidiaExecutable &executable = checked_executable(raw_executable);
  NvidiaBackendState &state = checked_state(raw_state);
  if (!or_to_all(num_steps > 0)) return;
  if (!state.initialized_) throw std::logic_error("cannot advance uninitialized NVIDIA storage");
  if (executable.storage_fingerprint_ != state.fingerprint_)
    throw std::logic_error("NVIDIA executable was compiled for a different storage layout");
  std::ostringstream message;
  message << "NVIDIA PR1 cannot advance " << num_steps
          << " timestep(s): no physics kernels were compiled for plan signature "
          << executable.signature_;
  throw std::runtime_error(message.str());
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
  /* PR1 launches transfers only: host values remain authoritative after every
     explicit write. Drain the stream before releasing the arena. PR2 will
     tighten this boundary when timestep kernels make device values authoritative. */
  checked_state(raw_state);
  synchronize();
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
