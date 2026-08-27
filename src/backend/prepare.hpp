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

/* Storage preparation.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 *
 * Eight families of array used to be allocated in the middle of a timestep, by
 * step_db, update_eh and update_pols, the first time the code noticed it needed
 * them. That means the first physical timestep can change storage and
 * invalidate boundary connectivity, and it means no plan built before the first
 * step can be trusted. These hooks realize the same allocations up front,
 * against the same conditions, so the timestep only ever executes.
 *
 * The hooks perform NO numerical update. They inspect existing Meep objects,
 * realize the current CPU allocations through their existing owners, and
 * register the result. The CPU loop bodies are untouched.
 */

#ifndef MEEP_BACKEND_PREPARE_HPP
#define MEEP_BACKEND_PREPARE_HPP

#include "meep.hpp"
#include "backend/array_ref.hpp"
#include "backend/storage_plan.hpp"

namespace meep {

/* Each returns true if it allocated something whose existence is visible to
   the boundary connectivity -- i.e. the caller must reconnect. Mirrors the
   `allocated_u` / `allocated_eh` returns the lazy paths used. */
bool prepare_step_db(fields_chunk &fc, field_type ft, StoragePlan &plan);
bool prepare_update_eh(fields_chunk &fc, field_type ft, bool skip_w_components, StoragePlan &plan);
bool prepare_polarizations(fields_chunk &fc, field_type ft, StoragePlan &plan);
void prepare_dfts(fields &f, StoragePlan &plan);

/* Debug-only: assert the lazy paths have nothing left to do. Compiled out with
   NDEBUG; the assertions-on CI configuration is what gives these value. */
void assert_step_db_prepared(const fields_chunk &fc, field_type ft);
void assert_update_eh_prepared(const fields_chunk &fc, field_type ft, bool skip_w_components);

} // namespace meep

#endif // MEEP_BACKEND_PREPARE_HPP
