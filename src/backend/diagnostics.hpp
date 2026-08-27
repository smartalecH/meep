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

/* Non-finite value diagnostics.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 *
 * The CPU backend deliberately keeps the cheap center-point read that
 * fields::step has always performed. A per-voxel diagnostic would require
 * touching STEP_* (global rule 6) and would regress CPU performance; the
 * device-native per-voxel version is Phase 2 (decision D).
 */

#ifndef MEEP_BACKEND_DIAGNOSTICS_HPP
#define MEEP_BACKEND_DIAGNOSTICS_HPP

#include <stdint.h>

namespace meep {

struct DiagnosticBlock {
  uint32_t nonfinite_flag; // monotone; set when a non-finite value is observed
  int32_t first_bad_step;
  int32_t first_bad_component;
};

/* MEEP_FINITE_CHECK=step|batch|off.

   step  (default) -- the historical behavior: read the cell center once per
                      timestep and abort immediately.
   batch           -- the *same* center-point read, performed once at each
                      advance() boundary. first_bad_step is reported as the
                      batch end, because that is when the observation was made.
   off             -- no check. */
enum class FiniteCheckMode { step, batch, off };

/* Reads MEEP_FINITE_CHECK once and caches it. An unrecognized value warns and
   falls back to `step`. */
FiniteCheckMode finite_check_mode();

/* Test hook: overrides the environment for the remainder of the process. */
void set_finite_check_mode(FiniteCheckMode mode);

} // namespace meep

#endif // MEEP_BACKEND_DIAGNOSTICS_HPP
