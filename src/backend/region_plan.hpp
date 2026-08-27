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

/* Chunk-loop regions as data.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 *
 * fields::loop_in_chunks plans a great deal of useful geometry -- chunk
 * intersections, periodic images, symmetry transforms, Bloch phases, Yee-grid
 * bounds, interpolation and integration weights, cylindrical dV factors,
 * empty-dimension snapping -- and then immediately throws it away by invoking a
 * host callback with a fields_chunk *. All of that is reusable data.
 */

#ifndef MEEP_BACKEND_REGION_PLAN_HPP
#define MEEP_BACKEND_REGION_PLAN_HPP

#include <complex>
#include <stdint.h>
#include <vector>

#include "meep.hpp"

namespace meep {

struct BoundaryWeights {
  vec s0, s1, e0, e1;
  BoundaryWeights() : s0(D1), s1(D1), e0(D1), e1(D1) {}
  BoundaryWeights(const vec &a, const vec &b, const vec &c, const vec &d)
      : s0(a), s1(b), e0(c), e1(d) {}
};

/* Deliberately contains NO fields_chunk *, native array pointer, callback, or
   symmetry object. The compatibility adapter recovers those from `fields`; a
   future GPU operation receives only the values it uses. Do not add a
   fields_chunk * here "just for the adapter" -- that defeats the point. */
struct ChunkLoopRegion {
  int chunk;
  component transformed_grid_component;
  ivec begin;
  ivec end;
  BoundaryWeights weights;
  double dV0;
  double dV1;
  ivec lattice_shift;
  std::complex<double> phase;
  int symmetry_index;
};

struct ChunkLoopPlan {
  std::vector<ChunkLoopRegion> regions;
  uint64_t topology_generation;
  ChunkLoopPlan() : topology_generation(0) {}
};

/* Enumerate the same regions fields::loop_in_chunks would visit, in the same
   order, and return them as data instead of invoking a callback.

   Inherits the collective sum_to_all volume check, so it must be entered on
   every rank. */
ChunkLoopPlan prepare_loop_in_chunks(fields &f, const volume &where, component cgrid,
                                     bool use_symmetry = true, bool snap_empty_dimensions = false);

} // namespace meep

#endif // MEEP_BACKEND_REGION_PLAN_HPP
