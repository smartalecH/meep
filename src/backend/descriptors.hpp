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
  /* Stable position in fields_chunk::get_sources(ft), before the ordinary /
     integrated grouping used by the timestep operations. */
  uint32_t source_ordinal;
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
  ArrayRef source_field_imag; // invalid for real fields
  std::vector<double> omega;  // angular frequencies, in live dft_chunk order
  std::complex<double> scale; // stored_weight * symmetry phase * dt factor
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

/* --- Legacy instantaneous flux monitors ---------------------------------
   The live flux_vol list owns only host callbacks and scalar results.  These
   recipes flatten the same centered-grid E x H integrations into portable
   chunk-local terms.  No flux_vol pointer or private half-step scalar crosses
   this boundary. */

struct LegacyFluxTermDescriptor {
  uint32_t term_ordinal;
  uint32_t region_ordinal;
  int sign;
  int chunk;
  component e_component;
  component h_component;
  ArrayId e_real;
  ArrayId e_imag;
  ArrayId h_real;
  ArrayId h_imag;
  ivec begin;
  ivec end;
  ivec lattice_shift;
  int symmetry_index;
  size_t base;
  size_t counts[3];
  ptrdiff_t strides[3];
  ptrdiff_t e_offsets[2];
  ptrdiff_t h_offsets[2];
  double phase_real;
  double phase_imag;
  double boundary_weights[3][4];
  double dV0;
  double dV1;
};

struct LegacyFluxDescriptor {
  uint32_t flux_ordinal;
  direction normal;
  volume where;
  uint64_t recipe_signature;
  std::vector<LegacyFluxTermDescriptor> terms;

  LegacyFluxDescriptor(uint32_t ordinal, direction d, const volume &v)
      : flux_ordinal(ordinal), normal(d), where(v), recipe_signature(0) {}
};

/* --- Susceptibilities ----------------------------------------------------- */

enum class SusceptibilityKind { lorentzian, noisy_lorentzian, gyrotropic, multilevel, host_custom };

const char *susceptibility_kind_name(SusceptibilityKind k);

struct LorentzianParameters {
  double omega_0;
  double gamma;
  bool drude;
};

struct LorentzianStateArrays {
  component c;
  int cmp;
  ArrayId p;
  ArrayId p_prev;
  size_t elements;
};

struct GyrotropicParameters {
  double omega_0;
  double gamma;
  double alpha;
  double gyro_tensor[3][3];
  gyrotropy_model model;
};

struct GyrotropicStateArrays {
  component c;
  int cmp;
  ArrayId p[3];
  ArrayId p_prev[3];
  size_t elements;
};

struct PolarizationDescriptor {
  SusceptibilityKind kind;
  int chunk;
  field_type ft;
  int state_index;
  LorentzianParameters lorentzian;
  std::vector<LorentzianStateArrays> lorentzian_states;
  GyrotropicParameters gyrotropic;
  std::vector<GyrotropicStateArrays> gyrotropic_states;
  std::vector<InternalArrayLayout> internal_arrays;
  size_t per_thread_scratch_elements;
  uint64_t required_w;      // bit per (component, cmp)
  uint64_t required_w_prev;
  bool needs_halo;
};

struct DescriptorSet {
  SourcePlan sources;
  std::vector<DftDescriptor> dfts;
  std::vector<LegacyFluxDescriptor> legacy_fluxes;
  uint64_t legacy_flux_generation;
  std::vector<PolarizationDescriptor> polarizations;
  std::vector<ChunkLoopRegion> regions;
  DescriptorSet() : legacy_flux_generation(0) {}
  void clear() {
    sources.clear();
    dfts.clear();
    legacy_fluxes.clear();
    legacy_flux_generation = 0;
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
/* Structural identity of source recipes/spatial tables. Runtime scalar values
   are deliberately excluded so ordinary timesteps do not invalidate code. */
uint64_t source_plan_signature(const SourcePlan &plan);
void build_dft_descriptors(fields &f, std::vector<DftDescriptor> &out);
/* Structural identity of the complete final-monitor recipes. Runtime due
   values are deliberately excluded; decimation_factor and due_scalar_slot
   are descriptor metadata and are included. */
uint64_t dft_plan_signature(const std::vector<DftDescriptor> &plan);
void build_legacy_flux_descriptors(fields &f, std::vector<LegacyFluxDescriptor> &out);
/* Rebuild the owned DescriptorSet recipes and stamp the live list generation.
   PR7/8 use this at their collective transactional refresh boundary. */
void refresh_legacy_flux_descriptors(fields &f);
void build_polarization_descriptors(fields &f, std::vector<PolarizationDescriptor> &out);

} // namespace meep

#endif // MEEP_BACKEND_DESCRIPTORS_HPP
