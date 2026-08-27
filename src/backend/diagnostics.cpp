/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include <stdlib.h>
#include <string.h>

#include "meep.hpp"
#include "backend/diagnostics.hpp"

namespace meep {

static bool mode_resolved = false;
static FiniteCheckMode cached_mode = FiniteCheckMode::step;

FiniteCheckMode finite_check_mode() {
  if (mode_resolved) return cached_mode;
  mode_resolved = true;
  cached_mode = FiniteCheckMode::step;
  const char *env = getenv("MEEP_FINITE_CHECK");
  if (env && *env) {
    if (!strcmp(env, "step"))
      cached_mode = FiniteCheckMode::step;
    else if (!strcmp(env, "batch"))
      cached_mode = FiniteCheckMode::batch;
    else if (!strcmp(env, "off"))
      cached_mode = FiniteCheckMode::off;
    else
      master_printf("meep: ignoring unrecognized MEEP_FINITE_CHECK=%s "
                    "(expected step|batch|off)\n",
                    env);
  }
  return cached_mode;
}

void set_finite_check_mode(FiniteCheckMode mode) {
  mode_resolved = true;
  cached_mode = mode;
}

} // namespace meep
