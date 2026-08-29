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

/* The timestep as data.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 *
 * fields::step walked chunks and variants and invoked many small operations,
 * repeating the same plan decisions every step. This replaces the procedure
 * with a constructed plan that a backend executes.
 *
 * An operation is COARSE: an update_db operation refers to a span of
 * chunk/component regions, not one voxel. There is no backend virtual call and
 * no backend branch per voxel, and the CPU kernels are untouched.
 */

#ifndef MEEP_BACKEND_STEP_PLAN_HPP
#define MEEP_BACKEND_STEP_PLAN_HPP

#include <stdint.h>
#include <string>
#include <vector>

#include "meep.hpp"
#include "backend/array_ref.hpp"
#include "backend/descriptors.hpp"

namespace meep {

enum class OpKind {
  restore_magnetic_fields,
  phase_material,
  update_material_coefficients,
  evaluate_source_scalars,
  update_db,
  update_eh,
  update_polarization,
  apply_sources,
  zero_boundary,
  pack_halo,
  transfer_halo,  // host/MPI scheduler operation
  exchange_local, // same-rank fused gather/scatter; deferred, see section 14
  unpack_halo,
  update_flux_half,
  update_flux,
  increment_time,
  update_dft,
  synchronize_magnetic_fields,
  finite_value_check,
  reduction,
  host_callback,
  pack_state, // solve_cw BiCGSTAB vector packing
  unpack_state,
  num_kinds
};

const char *op_kind_name(OpKind k);

enum class AccessMode { read, write, read_write };

struct BufferAccess {
  ArrayRef array;
  AccessMode mode;
};

/* The guard mechanism is recorded in the plan rather than resolved by the
   executor because it determines graph structure in Phase 2: a device_predicate
   keeps a node in the graph, a graph_variant forces a second compiled program,
   a segment_boundary forces a host round trip. The CPU executor treats them all
   the same, but the choice has to be made now -- retrofitting it means
   re-reviewing the whole plan. */
enum class GuardKind {
  always,
  static_predicate, // resolved during preparation; the op is present or absent
  device_predicate, // kernel reads StepScalars and returns immediately when false
  graph_variant,    // the host scheduler selects a separately compiled program
  segment_boundary  // the host evaluates the guard between device segments
};

struct Guard {
  GuardKind kind;
  uint32_t scalar_slot;
  uint32_t variant_index;
};

inline Guard guard_always() { return Guard{GuardKind::always, 0, 0}; }
inline Guard guard_static(bool) { return Guard{GuardKind::static_predicate, 0, 0}; }
inline Guard guard_device(uint32_t slot) { return Guard{GuardKind::device_predicate, slot, 0}; }
inline Guard guard_variant(uint32_t v) { return Guard{GuardKind::graph_variant, 0, v}; }
inline Guard guard_segment(uint32_t slot) { return Guard{GuardKind::segment_boundary, slot, 0}; }

struct Operation {
  OpKind kind;
  uint32_t descriptor_index;
  uint32_t descriptor_count;
  uint32_t material_refresh_index;
  uint32_t material_refresh_count;
  uint32_t beta_descriptor_index;
  uint32_t beta_descriptor_count;
  uint32_t cylindrical_m_descriptor_index;
  uint32_t cylindrical_m_descriptor_count;
  uint32_t cylindrical_origin_action_index;
  uint32_t cylindrical_origin_action_count;
  uint32_t polarization_subtraction_index;
  uint32_t polarization_subtraction_count;
  uint32_t magnetic_state_index;
  uint32_t magnetic_state_count;
  /* Spatial source descriptors consumed by apply_sources or by update_eh's
     integrated-source preparation. The primary descriptor span remains
     available for the operation's ordinary update descriptors. */
  uint32_t source_descriptor_index;
  uint32_t source_descriptor_count;
  Guard guard;
  /* CPU execution keeps using the coarse field_type entry point. Device
     executors consume the half-open descriptor span. For finite_value_check,
     accesses are full canonical physical-field allocations in stable ArrayId
     order. A device diagnostic attributes the first failure by lowest access
     order, then lowest element index; StoragePlan::keys supplies its
     chunk/component/cmp identity. */
  field_type ft;
  double source_time_offset; // for evaluate_source_scalars: 0, 0.5*dt or dt
  std::vector<BufferAccess> accesses;
};

struct UpdateRegion {
  int chunk;
  component c;
  int cmp;
  ivec begin;
  ivec end;
  size_t base;
  size_t counts[3];
  ptrdiff_t strides[3];
  /* PML, conductivity, anisotropy, precision, coordinate and stride cases.
     Does not encode a backend. */
  uint32_t variant_key;
};

enum CurlVariant : uint32_t {
  curl_has_second_derivative = 1u << 0,
  curl_has_pml = 1u << 1,
  curl_has_pml_aux = 1u << 2,
  curl_has_conductivity = 1u << 3,
  curl_has_bfast = 1u << 4
};

struct PmlProfile {
  ArrayId sig;
  ArrayId kap;
  ArrayId siginv;
  int base;
  int strides[3];
};

struct CurlUpdate {
  UpdateRegion region;
  ArrayId target;
  ArrayId plus_source;
  ArrayId minus_source;
  ptrdiff_t plus_stride;
  ptrdiff_t minus_stride;
  ArrayId target_u;
  ArrayId conductivity;
  ArrayId condinv;
  ArrayId target_cond;
  PmlProfile pml;
  PmlProfile pml_u;
  double dtdx;
  double dt;
  /* UINT32_MAX means that this curl has no paired cylindrical radial
     prefix. The prefix, curl, and optional BFAST postpass execute in that
     order before the next curl row may reuse the chunk scratch. */
  uint32_t radial_prefix_index;
  /* BFAST is an ordered postpass for this exact curl row, not a chunk-level
     tail. UINT32_MAX means that the ordinary curl has no paired postpass. */
  uint32_t bfast_update_index;
};

struct CylindricalRadialPrefix {
  int chunk;
  component target_component;
  component source_component;
  int cmp;
  ArrayId source;
  ArrayId scratch;
  size_t nr;
  size_t nz;
  size_t row_stride;
  size_t source_elements;
  size_t scratch_elements;
  double ir0;
};

enum BfastVariant : uint32_t {
  bfast_has_pml = 1u << 0,
  bfast_has_pml_aux = 1u << 1,
  bfast_has_conductivity = 1u << 2
};

struct BfastUpdate {
  UpdateRegion region;
  ArrayId target;
  ArrayId source1;
  ArrayId source2;
  ptrdiff_t stride1;
  ptrdiff_t stride2;
  ArrayId f_bfast;
  ArrayId target_u;
  ArrayId condinv;
  ArrayId target_cond;
  PmlProfile pml;
  PmlProfile pml_u;
  double k1;
  double k2;
};

enum BetaVariant : uint32_t {
  beta_has_pml = 1u << 0,
  beta_has_pml_aux = 1u << 1,
  beta_has_conductivity = 1u << 2
};

/* The 2-D special-kz correction applied after every ordinary curl span.
   betadt is formed in host realnum arithmetic and widened only for storage, so
   native-single lowering can reproduce the CPU coefficient bit for bit. */
struct BetaUpdate {
  UpdateRegion region;
  ArrayId target;
  ArrayId source;
  ArrayId target_u;
  ArrayId condinv;
  ArrayId target_cond;
  PmlProfile pml;
  PmlProfile pml_u;
  double betadt;
};

enum CylindricalMVariant : uint32_t {
  cylindrical_m_has_pml = 1u << 0,
  cylindrical_m_has_pml_aux = 1u << 1,
  cylindrical_m_has_conductivity = 1u << 2
};

struct CylindricalMOverRUpdate {
  UpdateRegion region;
  ArrayId target;
  ArrayId source;
  ArrayId target_u;
  ArrayId condinv;
  ArrayId target_cond;
  PmlProfile pml;
  PmlProfile pml_u;
  double numerator;
  int raw_radial_start;
};

enum class CylindricalAxisKind : uint32_t { m0_dz = 0, abs_m1 = 1 };

enum CylindricalAxisVariant : uint32_t {
  cylindrical_axis_has_pml = 1u << 0,
  cylindrical_axis_has_pml_aux = 1u << 1,
  cylindrical_axis_has_conductivity = 1u << 2
};

struct CylindricalAxisUpdate {
  CylindricalAxisKind kind;
  UpdateRegion region;
  ArrayId target;
  ArrayId source1;
  ArrayId source2;
  ptrdiff_t source1_neighbor_offset;
  ptrdiff_t source2_offset;
  ArrayId target_u;
  ArrayId conductivity;
  ArrayId condinv;
  ArrayId target_cond;
  PmlProfile pml;
  PmlProfile pml_u;
  double scale;
  double source2_multiplier;
  double dt;
};

enum class CylindricalOriginActionKind : uint32_t { axis_update = 0, zero_slab = 1 };

struct CylindricalOriginAction {
  CylindricalOriginActionKind kind;
  uint32_t index;
};

enum ConstitutiveVariant : uint32_t {
  constitutive_one_offdiagonal = 1u << 0,
  constitutive_two_offdiagonals = 1u << 1,
  constitutive_has_pml = 1u << 2,
  constitutive_has_nonlinearity = 1u << 3,
  constitutive_has_minus_p = 1u << 4,
  constitutive_copy_w_previous = 1u << 5,
  constitutive_axis_override = 1u << 6
};

struct ConstitutiveUpdate {
  UpdateRegion region;
  ArrayId target;
  /* Original D/B inputs, retained even when primary/cross* select f_minus_p.
     Preparing f_minus_p is a copy/subtract phase, so a lowering needs both
     sides rather than only the post-subtraction input. */
  ArrayId base_primary;
  ArrayId base_cross1;
  ArrayId base_cross2;
  ArrayId primary;
  ArrayId cross1;
  ArrayId cross2;
  ArrayId diagonal;
  ArrayId offdiagonal1;
  ArrayId offdiagonal2;
  ptrdiff_t primary_stride;
  ptrdiff_t cross1_stride;
  ptrdiff_t cross2_stride;
  ArrayId chi2;
  ArrayId chi3;
  ArrayId target_w;
  ArrayId previous_w;
  PmlProfile pml;
};

enum PolarizationVariant : uint32_t {
  polarization_one_offdiagonal = 1u << 0,
  polarization_two_offdiagonals = 1u << 1,
  polarization_drude = 1u << 2
};

enum class PolarizationUpdateKind : uint32_t { lorentzian = 0, gyrotropic = 1 };

struct PolarizationUpdate {
  PolarizationUpdateKind kind;
  UpdateRegion region;
  int state_index;
  ArrayId p;
  ArrayId p_prev;
  ArrayId p_cross1;
  ArrayId p_prev_cross1;
  ArrayId p_cross2;
  ArrayId p_prev_cross2;
  ArrayId primary_w;
  ArrayId cross_w1;
  ArrayId cross_w2;
  ArrayId diagonal_sigma;
  ArrayId offdiagonal_sigma1;
  ArrayId offdiagonal_sigma2;
  ptrdiff_t primary_stride;
  ptrdiff_t cross_stride1;
  ptrdiff_t cross_stride2;
  double omega_0;
  double gamma;
  double alpha;
  double gyro_tensor[3][3];
  gyrotropy_model gyro_model;
  double dt;
};

struct PolarizationSubtraction {
  int chunk;
  component c;
  int cmp;
  int state_index;
  ArrayId target;
  ArrayId p;
  size_t elements;
};

/* A resident magnetic snapshot is backend-private storage.  The portable plan
   names only the live array and its semantic identity; it deliberately does
   not expose the legacy lazy host *_backup pointers. */
enum class MagneticStateFamily : uint32_t {
  primary = 0,
  u = 1,
  w = 2,
  conductivity = 3,
  bfast = 4
};

struct MagneticStateArray {
  int chunk;
  component c;
  int cmp;
  MagneticStateFamily family;
  ArrayId live;
  size_t elements;
  bool average;
};

/* Material interpolation remains host-authoritative.  These rows identify the
   stable current arrays that a resident backend refreshes after the host has
   mixed them; target material arrays deliberately never enter the catalog. */
enum class MaterialRefreshFamily : uint32_t {
  chi1inv = 0,
  conductivity = 1,
  condinv = 2
};

struct MaterialRefreshArray {
  int chunk;
  component c;
  direction d;
  MaterialRefreshFamily family;
  ArrayId current;
  size_t elements;
};

/* Exact operation indices for the restricted B/H half-step used by magnetic
   synchronization.  UINT32_MAX denotes an omitted source-evaluation node. */
struct MagneticHalfStep {
  uint32_t evaluate_b_sources;
  uint32_t update_b;
  uint32_t apply_b_sources;
  uint32_t transfer_b;
  uint32_t evaluate_h_sources;
  uint32_t update_h;
  uint32_t transfer_h;

  MagneticHalfStep()
      : evaluate_b_sources(UINT32_MAX), update_b(UINT32_MAX), apply_b_sources(UINT32_MAX),
        transfer_b(UINT32_MAX), evaluate_h_sources(UINT32_MAX), update_h(UINT32_MAX),
        transfer_h(UINT32_MAX) {}
};

/* Which timestep program a plan describes. solve_cw is a genuinely different
   program, not a variant: step_source skips non-integrated sources, update_eh
   runs with skip_w_components, and update_dfts is disabled. If step() begins
   executing an ordinary-case plan without a corresponding solve_cw plan, CW
   solves silently run the wrong program. */
enum class StepProgram { ordinary, solve_cw };

struct StepPlan {
  StepProgram program;
  /* Lifecycle provenance, deliberately excluded from the content signature. */
  uint64_t coordinate_generation;
  double beta;
  double cylindrical_m;
  std::vector<double> bfast_scaled_k;
  std::vector<double> cylindrical_origin_r;
  std::vector<uint8_t> cylindrical_zero_near_origin;
  std::vector<Operation> operations;
  std::vector<CurlUpdate> db_updates;
  std::vector<CylindricalRadialPrefix> cylindrical_radial_prefixes;
  std::vector<BfastUpdate> bfast_updates;
  std::vector<BetaUpdate> beta_updates;
  std::vector<CylindricalMOverRUpdate> cylindrical_m_updates;
  std::vector<CylindricalAxisUpdate> cylindrical_axis_updates;
  std::vector<SlabRef> cylindrical_zero_slabs;
  std::vector<CylindricalOriginAction> cylindrical_origin_actions;
  std::vector<ConstitutiveUpdate> eh_updates;
  std::vector<PolarizationUpdate> polarization_updates;
  std::vector<PolarizationSubtraction> polarization_subtractions;
  std::vector<MagneticStateArray> magnetic_state_arrays;
  std::vector<MaterialRefreshArray> material_refresh_arrays;
  MagneticHalfStep magnetic_half_step;
  std::vector<DftDescriptor> dft_updates;
  /* Structural identity of the host-only phase target.  Values are excluded:
     changing them is the purpose of material phasing. */
  uint64_t material_phase_target_signature;
  uint64_t source_signature;
  uint64_t signature;

  StepPlan()
      : program(StepProgram::ordinary), coordinate_generation(0), beta(0), cylindrical_m(0),
        material_phase_target_signature(0), source_signature(0), signature(0) {}
  void clear() {
    coordinate_generation = 0;
    beta = 0;
    cylindrical_m = 0;
    bfast_scaled_k.clear();
    cylindrical_origin_r.clear();
    cylindrical_zero_near_origin.clear();
    operations.clear();
    db_updates.clear();
    cylindrical_radial_prefixes.clear();
    bfast_updates.clear();
    beta_updates.clear();
    cylindrical_m_updates.clear();
    cylindrical_axis_updates.clear();
    cylindrical_zero_slabs.clear();
    cylindrical_origin_actions.clear();
    eh_updates.clear();
    polarization_updates.clear();
    polarization_subtractions.clear();
    magnetic_state_arrays.clear();
    material_refresh_arrays.clear();
    magnetic_half_step = MagneticHalfStep();
    dft_updates.clear();
    material_phase_target_signature = 0;
    source_signature = 0;
    signature = 0;
  }
};

/* A direct transcription of the order in src/step.cpp. It omits empty work and
   NEVER reorders what remains. It is not a scheduler that infers Maxwell's
   equations from array dependencies: if a reviewer cannot line this up against
   fields::step_once by eye, it is wrong. */
StepPlan build_step_plan(fields &f, StepProgram program);

/* Structural identity used by device executables and tests. */
uint64_t compute_step_plan_signature(const StepPlan &plan);

/* Recomputed by resident preflight so externally-mutated target topology is
   rejected before phase_material_mix changes current coefficients. */
uint64_t compute_material_phase_target_signature(const fields &f);
/* Public fields::beta assignment cannot update the per-chunk coefficient that
   the CPU and descriptor builders consume. Resident execution therefore
   rejects any outer/chunk/prepared mismatch instead of reusing stale code. */
bool beta_coordinate_state_matches(const fields &f, const StepPlan *prepared);

/* Human-readable operation sequence, for the trace test and for debugging. */
void format_step_plan(const StepPlan &p, std::vector<std::string> &out);

} // namespace meep

#endif // MEEP_BACKEND_STEP_PLAN_HPP
