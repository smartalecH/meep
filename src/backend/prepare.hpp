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

#include <memory>
#include <vector>

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

/* Resident material phasing must freeze a current-storage superset before the
   backend catalog is built.  The target remains host-only: this helper
   transactionally detaches each owned current structure chunk, realizes the
   current/target chi1inv and conductivity union using the CPU defaults, and
   materializes any diagonal condinv row that can be needed during the phase. */
class PreparedMaterialPhaseStorage {
public:
  PreparedMaterialPhaseStorage(fields &f, const structure &target);
  explicit PreparedMaterialPhaseStorage(fields &f);
  ~PreparedMaterialPhaseStorage();
  void commit();

private:
  PreparedMaterialPhaseStorage(fields &f, const structure *target);
  PreparedMaterialPhaseStorage(const PreparedMaterialPhaseStorage &);
  PreparedMaterialPhaseStorage &operator=(const PreparedMaterialPhaseStorage &);
  fields &owner_;
  std::vector<std::unique_ptr<structure_chunk> > chunks_;
  bool committed_;
};

std::unique_ptr<PreparedMaterialPhaseStorage> prepare_material_phase_storage(
    fields &f, const structure &target);
std::unique_ptr<PreparedMaterialPhaseStorage> prepare_material_phase_storage(fields &f);

/* Deterministic allocation-failure seam for the backend contract tests.  A
   negative rank disables it; otherwise preparation throws after staging the
   requested number of owned replacement chunks on that rank. */
void set_material_phase_prepare_failure_for_testing(int rank, int after_chunks);

/* Debug-only: assert the lazy paths have nothing left to do. Compiled out with
   NDEBUG; the assertions-on CI configuration is what gives these value. */
void assert_step_db_prepared(const fields_chunk &fc, field_type ft);
void assert_update_eh_prepared(const fields_chunk &fc, field_type ft, bool skip_w_components);

} // namespace meep

#endif // MEEP_BACKEND_PREPARE_HPP
