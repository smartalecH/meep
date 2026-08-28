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

/* Storage precision policy.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp. The
 * *selector* (precision_policy_kind) is public and lives in meep.hpp; this is
 * the four-way policy it expands into.
 *
 * ElementType says what a value means; Precision says how it is stored.
 * Keeping them separate is what lets `mixed` exist in Phase 2 without
 * duplicating every descriptor.
 */

#ifndef MEEP_BACKEND_PRECISION_HPP
#define MEEP_BACKEND_PRECISION_HPP

#include "meep.hpp"
#include "backend/array_ref.hpp"

namespace meep {

const Precision native_precision =
    sizeof(realnum) == sizeof(float) ? Precision::f32 : Precision::f64;

struct PrecisionPolicy {
  Precision field;    // f, f_u, f_w, f_cond, f_bfast, f_minus_p, f_w_prev,
                      // backups, polarization internal data
  Precision material; // chi1inv, conductivity, condinv, chi2/chi3, sigma,
                      // PML sig/kap/siginv
  Precision monitor;  // DFT accumulators and monitor scratch
  Precision reduction; // flux/energy/norm accumulators and their MPI reductions

  bool operator==(const PrecisionPolicy &o) const {
    return field == o.field && material == o.material && monitor == o.monitor &&
           reduction == o.reduction;
  }
};

inline PrecisionPolicy precision_native() {
  return PrecisionPolicy{native_precision, native_precision, native_precision, native_precision};
}
inline PrecisionPolicy precision_mixed() {
  return PrecisionPolicy{Precision::f32, Precision::f32, Precision::f64, Precision::f64};
}
inline PrecisionPolicy precision_f32() {
  return PrecisionPolicy{Precision::f32, Precision::f32, Precision::f32, Precision::f64};
}

PrecisionPolicy policy_for(precision_policy_kind kind);
const char *precision_policy_name(precision_policy_kind kind);

/* Arrays joined by an alias (H == B) must have identical storage precision.
   Validated even though the CPU backend only supports `native`, because the
   check is cheap and the failure mode in Phase 2 is silent corruption. */
bool validate_alias_precisions(const class CpuArrayCatalog &cat, std::string &why);

} // namespace meep

#endif // MEEP_BACKEND_PRECISION_HPP
