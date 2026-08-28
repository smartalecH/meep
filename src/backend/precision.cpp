/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include <stdlib.h>
#include <string.h>

#include <cerrno>
#include <complex>
#include <limits>
#include <stdexcept>

#include "backend/precision.hpp"
#include "backend/storage_plan.hpp"

namespace meep {

PrecisionPolicy policy_for(precision_policy_kind kind) {
  switch (kind) {
    case precision_policy_kind::native: return precision_native();
    case precision_policy_kind::mixed: return precision_mixed();
    case precision_policy_kind::f32: return precision_f32();
  }
  return precision_native();
}

const char *precision_policy_name(precision_policy_kind kind) {
  switch (kind) {
    case precision_policy_kind::native: return "native";
    case precision_policy_kind::mixed: return "mixed";
    case precision_policy_kind::f32: return "f32";
  }
  return "?";
}

size_t host_element_bytes(ElementType type) {
  switch (type) {
    case ElementType::realnum_value: return sizeof(realnum);
    case ElementType::complex_realnum: return sizeof(std::complex<realnum>);
    case ElementType::float64: return sizeof(double);
    case ElementType::complex_float64: return sizeof(std::complex<double>);
    case ElementType::int32: return sizeof(int32_t);
    case ElementType::index: return sizeof(size_t);
  }
  throw std::invalid_argument("invalid backend element type");
}

size_t storage_element_bytes(ElementType type, Precision storage) {
  switch (type) {
    case ElementType::realnum_value:
      if (storage == Precision::f32) return sizeof(float);
      if (storage == Precision::f64) return sizeof(double);
      break;
    case ElementType::complex_realnum:
      if (storage == Precision::f32) return 2 * sizeof(float);
      if (storage == Precision::f64) return 2 * sizeof(double);
      break;
    case ElementType::float64: return sizeof(double);
    case ElementType::complex_float64: return sizeof(std::complex<double>);
    case ElementType::int32: return sizeof(int32_t);
    case ElementType::index: return sizeof(size_t);
  }
  throw std::invalid_argument("invalid backend element type or storage precision");
}

size_t storage_bytes(const ArraySpec &spec) {
  const size_t element_size = storage_element_bytes(spec.element_type, spec.storage);
  if (spec.elements && element_size > std::numeric_limits<size_t>::max() / spec.elements)
    throw std::overflow_error("backend array storage byte count overflow");
  return spec.elements * element_size;
}

void apply_precision_policy(StoragePlan &plan, const PrecisionPolicy &policy) {
  for (ArraySpec &spec : plan.arrays) {
    switch (spec.element_type) {
      case ElementType::float64:
      case ElementType::complex_float64: spec.storage = Precision::f64; break;
      case ElementType::int32:
      case ElementType::index: break;
      case ElementType::realnum_value:
      case ElementType::complex_realnum:
        switch (spec.role) {
          case array_role::field:
          case array_role::polarization: spec.storage = policy.field; break;
          case array_role::material: spec.storage = policy.material; break;
          case array_role::dft: spec.storage = policy.monitor; break;
          /* Halo payloads follow their field representation. Scratch remains
             native until the operation that owns it provides a narrower typed
             descriptor; leaving either value implicit would make byte sizing
             depend on catalog construction history. */
          case array_role::communication: spec.storage = policy.field; break;
          case array_role::scratch: spec.storage = native_precision; break;
          default: throw std::invalid_argument("invalid backend array role");
        }
        break;
    }
  }

  std::string why;
  if (!validate_alias_precisions(plan, why)) throw std::invalid_argument(why);
}

bool validate_alias_precisions(const CpuArrayCatalog &cat, std::string &why) {
  why.clear();
  for (size_t i = 0; i < cat.size(); ++i) {
    const ArraySpec &s = cat.spec(ArrayId{uint32_t(i)});
    if (!is_valid(s.alias_of)) continue;
    const ArraySpec &o = cat.spec(s.alias_of);
    if (s.storage != o.storage) {
      why = "aliased arrays have different storage precisions";
      return false;
    }
  }
  return true;
}

bool validate_alias_precisions(const StoragePlan &plan, std::string &why) {
  why.clear();
  for (size_t i = 0; i < plan.arrays.size(); ++i) {
    const ArraySpec &spec = plan.arrays[i];
    if (spec.id.value != i) {
      why = "storage plan ArrayId does not match its array index";
      return false;
    }
    if (!is_valid(spec.alias_of)) continue;
    if (spec.alias_of.value >= plan.arrays.size()) {
      why = "storage plan alias refers to an invalid ArrayId";
      return false;
    }
    const ArraySpec &target = plan.arrays[spec.alias_of.value];
    if (spec.storage != target.storage) {
      why = "aliased arrays have different storage precisions";
      return false;
    }
  }
  return true;
}

const char *backend_kind_name(backend_kind k) {
  switch (k) {
    case backend_kind::cpu: return "cpu";
    case backend_kind::nvidia: return "nvidia";
    case backend_kind::automatic: return "auto";
  }
  return "?";
}

/* Environment overrides. An unrecognized value warns and leaves the field
   alone rather than guessing -- a silently ignored MEEP_BACKEND=nvidia would be
   worse than either honoring it or failing. */
void apply_execution_environment(execution_options &opts) {
  if (const char *e = getenv("MEEP_BACKEND")) {
    if (!strcmp(e, "cpu"))
      opts.backend = backend_kind::cpu;
    else if (!strcmp(e, "nvidia"))
      opts.backend = backend_kind::nvidia;
    else if (!strcmp(e, "auto"))
      opts.backend = backend_kind::automatic;
    else
      master_printf("meep: ignoring unrecognized MEEP_BACKEND=%s (expected cpu|nvidia|auto)\n", e);
  }
  if (const char *e = getenv("MEEP_PRECISION")) {
    if (!strcmp(e, "native"))
      opts.precision = precision_policy_kind::native;
    else if (!strcmp(e, "mixed"))
      opts.precision = precision_policy_kind::mixed;
    else if (!strcmp(e, "f32"))
      opts.precision = precision_policy_kind::f32;
    else
      master_printf("meep: ignoring unrecognized MEEP_PRECISION=%s (expected native|mixed|f32)\n",
                    e);
  }
  if (const char *e = getenv("MEEP_DEVICE_ID")) {
    errno = 0;
    char *end = NULL;
    const long value = strtol(e, &end, 10);
    if (!*e || errno || !end || *end || value < 0 ||
        value > std::numeric_limits<int>::max())
      master_printf("meep: invalid MEEP_DEVICE_ID=%s (expected a nonnegative integer)\n", e);
    else
      opts.device_id = int(value);
  }
  if (const char *e = getenv("MEEP_ACCELERATOR_STRICT"))
    opts.strict = !(!strcmp(e, "0") || !strcmp(e, "false") || !strcmp(e, "no"));
}

} // namespace meep
