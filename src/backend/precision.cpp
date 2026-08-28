/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include <stdlib.h>
#include <string.h>

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

bool validate_alias_precisions(const CpuArrayCatalog &cat, std::string &why) {
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
  if (const char *e = getenv("MEEP_DEVICE_ID")) opts.device_id = atoi(e);
  if (const char *e = getenv("MEEP_ACCELERATOR_STRICT"))
    opts.strict = !(!strcmp(e, "0") || !strcmp(e, "false") || !strcmp(e, "no"));
}

} // namespace meep
