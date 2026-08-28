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

/* Sources, monitors and susceptibilities as backend-neutral data.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 *
 * These are descriptions, not implementations. In Phase 1 the CPU path still
 * runs the same code it always did; what changes is that everything a second
 * backend would need to lower these operations now also exists as data, built
 * from the same objects and checked against them.
 */

#ifndef MEEP_BACKEND_DESCRIPTORS_HPP
#define MEEP_BACKEND_DESCRIPTORS_HPP

#include <complex>
#include <stdint.h>
#include <vector>

#include "meep.hpp"
#include "backend/array_ref.hpp"
#include "backend/region_plan.hpp"

namespace meep {

/* --- Sources -------------------------------------------------------------
   A source separates into usually-static spatial data and a small
   time-dependent scalar produced by a src_time object. */

struct SourceDescriptor {
  ArrayId destination;      // cmp 0 D/B destination for ordinary sources
  ArrayId destination_imag; // cmp 1, invalid for real fields
  /* Integrated sources subtract their dipole from f_minus_p during the
     following E/H constitutive update rather than modifying D/B directly. */
  ArrayId integrated_destination;
  ArrayId integrated_destination_imag;
  int chunk;
  component c;
  std::vector<ptrdiff_t> indices;
  std::vector<std::complex<double> > complex_amplitudes;
  ArrayId condinv; // invalid when the destination has no conductivity
  uint32_t source_time_id;
  bool integrated;
  field_type ft;
};

enum class SourceTimeKind { gaussian, continuous, host_custom };

const char *source_time_kind_name(SourceTimeKind k);

struct SourceTimeDescriptor {
  uint32_t source_time_id;
  SourceTimeKind kind;
  /* Exact closed form for the built-ins:
       gaussian   = [frequency, width, peak_time, cutoff]
       continuous = [frequency.real, frequency.imag, width,
                     start_time, end_time, slowness]
     host_custom has no parameters and remains host-polymorphic. */
  std::vector<double> parameters;
  uint32_t scalar_slot;
  uint32_t host_callback_id; // valid only for host_custom
  bool is_integrated;
};

/* The small host/device block the timestep reads. */
struct SourceStepScalar {
  std::complex<double> current;
  std::complex<double> dipole;
};

struct SourcePlan {
  std::vector<SourceDescriptor> sources;
  std::vector<SourceTimeDescriptor> source_times;
  std::vector<SourceStepScalar> scalars; // indexed by scalar_slot
  void clear() {
    sources.clear();
    source_times.clear();
    scalars.clear();
  }
};

/* --- DFT monitors --------------------------------------------------------- */

struct DftDescriptor {
  ArrayId accumulator;   // complex, monitor precision; layout [voxel][frequency]
  ArrayId phase_scratch; // Nomega entries, recomputed from the step time
  ArrayRef source_field;
  int chunk;
  component c;
  ptrdiff_t avg1, avg2; // Yee-to-centered offsets; 0 when not centered

  /* The persist / is_old distinction is load-bearing and easy to lose.
     Adjoint monitors expand their accumulation region by one pixel in each
     direction, clamped to the chunk. Both dft_chunk::norm2 and the
     design-region gradient loop iterate the *unpadded* extent while indexing
     the *padded* array. A lowering that collapses the two silently produces
     wrong dft_norm() values -- and therefore wrong stopping times -- in every
     adjoint run. */
  ivec is, ie;         // accumulation extent (padded when persist)
  ivec is_old, ie_old; // unpadded extent; equals is/ie unless persist
  bool persist;

  int decimation_factor;
  uint32_t due_scalar_slot; // predicate for the decimation guard
  BoundaryWeights weights;
  double dV0, dV1;
  bool include_dV_and_interp_weights;
  bool sqrt_dV_and_interp_weights;
  size_t N;
  size_t Nomega;
};

/* --- Susceptibilities ----------------------------------------------------- */

enum class SusceptibilityKind { lorentzian, noisy_lorentzian, gyrotropic, multilevel, host_custom };

const char *susceptibility_kind_name(SusceptibilityKind k);

struct PolarizationDescriptor {
  SusceptibilityKind kind;
  int chunk;
  field_type ft;
  int state_index;
  std::vector<double> parameters;
  std::vector<InternalArrayLayout> internal_arrays;
  size_t per_thread_scratch_elements;
  uint64_t required_w;      // bit per component
  uint64_t required_w_prev;
  bool needs_halo;
};

struct DescriptorSet {
  SourcePlan sources;
  std::vector<DftDescriptor> dfts;
  std::vector<PolarizationDescriptor> polarizations;
  std::vector<ChunkLoopRegion> regions;
  void clear() {
    sources.clear();
    dfts.clear();
    polarizations.clear();
    regions.clear();
  }
};

/* Built from the live objects; the CPU path continues to use those objects. */
void build_source_descriptors(fields &f, SourcePlan &out);
/* Evaluate a built-in descriptor without consulting the live src_time object.
   host_custom descriptors are rejected. */
SourceStepScalar evaluate_source_time_descriptor(const SourceTimeDescriptor &d, double time,
                                                 double dt);
/* Copy the already-evaluated live src_time values into the canonical scalar
   block. A stale/unprepared plan is left untouched; lifecycle refreshes it
   before a resident backend compiles or executes. */
void populate_source_scalars(fields &f, SourcePlan &out);
void build_dft_descriptors(fields &f, std::vector<DftDescriptor> &out);
void build_polarization_descriptors(fields &f, std::vector<PolarizationDescriptor> &out);

} // namespace meep

#endif // MEEP_BACKEND_DESCRIPTORS_HPP
