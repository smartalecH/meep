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
#include "backend/halo_plan.hpp"

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

/* A host callback marker does not duplicate the operations that retain the
   exact legacy CPU semantics.  It names a contiguous following span which a
   resident backend must execute on the host as one indivisible segment.  PR 6
   populates the callback identities; PR 2 supplies the host-owned halo plan
   namespace.  Their spans are declared here so the StepPlan schema is frozen
   before either live recipe is attached. */
enum class HostSegmentPhase : uint32_t { constitutive = 0, polarization_and_halo = 1 };

const char *host_segment_phase_name(HostSegmentPhase phase);

struct HostSegment {
  HostSegmentPhase phase;
  field_type ft;
  uint32_t operation_index;
  uint32_t operation_count;
  uint32_t callback_index;
  uint32_t callback_count;
  /* Logical PR2 host-halo plan span, never a generation-local HostHaloId. */
  uint32_t host_halo_plan_index;
  uint32_t host_halo_plan_count;
};

bool operator==(const HostSegment &a, const HostSegment &b);
inline bool operator!=(const HostSegment &a, const HostSegment &b) { return !(a == b); }

/* Stable, pointer-free identity for one live PR2 host-owned PE/PH plan.  The
   current HaloPlan and generation-local HostHaloIds are resolved only when a
   host segment is entered. */
struct HostHaloPlanDescriptor {
  field_type ft;
  connect_phase phase;
  chunk_pair chunks;
  uint32_t sequence_index;
  size_t block_offset;
  size_t block_elements;
  std::vector<HostHaloKey> gather_keys;
  std::vector<HostHaloKey> scatter_keys;
  std::vector<std::complex<realnum> > phase_values;
};

/* Convert the stable integer representation used by host-halo metadata only
   after checking that it names a real connect phase.  In particular, callers
   validating serialized or otherwise untrusted metadata must not first form
   an out-of-domain C++ enum value. */
bool decode_host_halo_phase(uint32_t serialized, connect_phase &phase);

bool operator==(const HostHaloPlanDescriptor &a, const HostHaloPlanDescriptor &b);
inline bool operator!=(const HostHaloPlanDescriptor &a, const HostHaloPlanDescriptor &b) {
  return !(a == b);
}

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
  uint32_t polarization_group_index;
  uint32_t polarization_group_count;
  uint32_t polarization_subtraction_index;
  uint32_t polarization_subtraction_count;
  uint32_t magnetic_state_index;
  uint32_t magnetic_state_count;
  uint32_t legacy_flux_index;
  uint32_t legacy_flux_count;
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

enum class PolarizationUpdateKind : uint32_t { lorentzian = 0, gyrotropic = 1, noisy_add = 2 };

struct PolarizationUpdate {
  PolarizationUpdateKind kind;
  UpdateRegion region;
  field_type ft;
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
  double noise_amplitude;
  uint32_t noise_algorithm_version;
};

bool operator==(const PolarizationUpdate &a, const PolarizationUpdate &b);
inline bool operator!=(const PolarizationUpdate &a, const PolarizationUpdate &b) {
  return !(a == b);
}

struct PolarizationSubtraction {
  int chunk;
  component c;
  int cmp;
  int state_index;
  int transition_index;
  ArrayId target;
  ArrayId p;
  size_t elements;
};

bool operator==(const PolarizationSubtraction &a, const PolarizationSubtraction &b);
inline bool operator!=(const PolarizationSubtraction &a, const PolarizationSubtraction &b) {
  return !(a == b);
}

enum class PolarizationGroupKind : uint32_t { recurrence = 0, multilevel = 1 };

/* One susceptibility in live linked-list order. The CPU still executes one
   update_polarization(ft) operation; device backends walk these groups so a
   multilevel population phase cannot be split from its transition phase or
   cause the CPU entry point to run more than once. */
struct PolarizationUpdateGroup {
  PolarizationGroupKind kind;
  int chunk;
  field_type ft;
  int state_index;
  uint32_t recurrence_index;
  uint32_t recurrence_count;
  uint32_t noise_count;
  uint32_t population_index;
  uint32_t population_count;
  uint32_t transition_index;
  uint32_t transition_count;
};

struct MultilevelPopulationTerm {
  int transition_index;
  component c;
  int cmp;
  ArrayId w;
  ArrayId w_prev;
  ArrayId p;
  ArrayId p_prev;
  ptrdiff_t centered_offsets[2];
};

struct MultilevelPopulationUpdate {
  UpdateRegion region;
  field_type ft;
  int state_index;
  uint32_t levels;
  uint32_t transitions;
  uint32_t active_component_cmps;
  ArrayId gamma_inv;
  ArrayId populations;
  uint32_t gamma_index;
  uint32_t gamma_count;
  uint32_t alpha_index;
  uint32_t alpha_count;
  uint32_t term_index;
  uint32_t term_count;
  size_t scratch_elements_per_point;
  double dt;
};

struct MultilevelTransitionUpdate {
  UpdateRegion region;
  field_type ft;
  int state_index;
  int transition_index;
  ArrayId p;
  ArrayId p_prev;
  ArrayId w;
  ArrayId diagonal_sigma;
  ArrayId populations;
  ptrdiff_t population_offsets[2];
  uint32_t population_stride;
  int positive_level;
  int negative_level;
  double omega;
  double gamma;
  double sigmat[5];
  double dt;
};

/* One legacy time-domain flux monitor owns a private half-step scalar and a
   final public scalar. The portable plan names the ordered term span only;
   backend-private scalar storage must never appear as an ArrayId. */
struct LegacyFluxUpdate {
  uint32_t flux_ordinal;
  uint32_t term_index;
  uint32_t term_count;
  /* PR6 hashes the original normal+volume into this identity. Realized terms
     alone are insufficient because distinct requested volumes can clip to the
     same owned region. */
  uint64_t recipe_signature;
};

inline bool operator==(const LegacyFluxUpdate &a, const LegacyFluxUpdate &b) {
  return a.flux_ordinal == b.flux_ordinal && a.term_index == b.term_index &&
         a.term_count == b.term_count && a.recipe_signature == b.recipe_signature;
}

/* One signed E/H product over one canonical chunk-loop region. Boundary
   weights are stored in loop-axis order as {s0,s1,e0,e1}. The instantaneous
   value is accumulated in f64 after four-point Yee centering of each field. */
struct LegacyFluxTerm {
  uint32_t flux_ordinal;
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

inline bool operator==(const LegacyFluxTerm &a, const LegacyFluxTerm &b) {
  if (a.flux_ordinal != b.flux_ordinal || a.term_ordinal != b.term_ordinal ||
      a.region_ordinal != b.region_ordinal || a.sign != b.sign || a.chunk != b.chunk ||
      a.e_component != b.e_component || a.h_component != b.h_component ||
      a.e_real != b.e_real || a.e_imag != b.e_imag || a.h_real != b.h_real ||
      a.h_imag != b.h_imag || !(a.begin == b.begin) || !(a.end == b.end) ||
      !(a.lattice_shift == b.lattice_shift) || a.symmetry_index != b.symmetry_index ||
      a.base != b.base || a.phase_real != b.phase_real || a.phase_imag != b.phase_imag ||
      a.dV0 != b.dV0 || a.dV1 != b.dV1)
    return false;
  for (int i = 0; i < 3; ++i) {
    if (a.counts[i] != b.counts[i] || a.strides[i] != b.strides[i]) return false;
    for (int j = 0; j < 4; ++j)
      if (a.boundary_weights[i][j] != b.boundary_weights[i][j]) return false;
  }
  for (int i = 0; i < 2; ++i)
    if (a.e_offsets[i] != b.e_offsets[i] || a.h_offsets[i] != b.h_offsets[i]) return false;
  return true;
}

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

/* One complex-valued row in the vector used by solve_cw.  The traversal
   component is always D/B and determines the owned Yee-point sequence.  The
   storage component may instead be its paired E/H component for the two
   constitutive-state rows at the end of each component group. */
enum class CwStateFamily : uint32_t {
  primary = 0,
  pml_u = 1,
  conductivity = 2,
  bfast = 3,
  constitutive_w = 4,
  paired_primary = 5
};

struct CwStateRow {
  int chunk;
  component traversal_component;
  component storage_component;
  CwStateFamily family;
  ArrayId real_array;
  ArrayId imag_array;
  UpdateRegion owned_region;
  size_t complex_offset;
  size_t complex_count;
};

/* array_to_fields performs this reconciliation immediately after scattering a
   vector.  Descriptor references for lowering the three actions are added by
   PR6; PR5 records only the semantics available at the StepPlan layer. */
struct CwUnpackPrelude {
  field_type first_boundary;
  field_type constitutive;
  field_type second_boundary;
  bool skip_w_components;
  bool invalidate_field_values;

  CwUnpackPrelude()
      : first_boundary(NO_FIELD_TYPE), constitutive(NO_FIELD_TYPE),
        second_boundary(NO_FIELD_TYPE), skip_w_components(false),
        invalidate_field_values(false) {}
};

/* Canonical, source-independent layout of the solve_cw state vector.  This is
   catalog metadata only: source and DFT descriptors first exist in PR6 and do
   not belong here.  BufferAccess entries conservatively cover whole backing
   allocations; CwStateRow::owned_region is the exact strided gather/scatter
   domain. */
struct CwStateLayout {
  std::vector<CwStateRow> rows;
  std::vector<ArrayRef> zero_arrays;
  std::vector<BufferAccess> pack_accesses;
  std::vector<BufferAccess> unpack_accesses;
  CwUnpackPrelude unpack_prelude;
  size_t complex_count;
  size_t real_count;
  Precision vector_precision;
  uint64_t storage_fingerprint;
  uint64_t coordinate_fingerprint;
  uint64_t material_fingerprint;
  uint64_t signature;

  CwStateLayout()
      : complex_count(0), real_count(0),
        vector_precision(sizeof(realnum) == sizeof(float) ? Precision::f32 : Precision::f64),
        storage_fingerprint(0), coordinate_fingerprint(0), material_fingerprint(0), signature(0) {}

  void clear() {
    rows.clear();
    zero_arrays.clear();
    pack_accesses.clear();
    unpack_accesses.clear();
    unpack_prelude = CwUnpackPrelude();
    complex_count = 0;
    real_count = 0;
    vector_precision = sizeof(realnum) == sizeof(float) ? Precision::f32 : Precision::f64;
    storage_fingerprint = 0;
    coordinate_fingerprint = 0;
    material_fingerprint = 0;
    signature = 0;
  }
};

bool operator==(const CwStateRow &a, const CwStateRow &b);
inline bool operator!=(const CwStateRow &a, const CwStateRow &b) { return !(a == b); }
bool operator==(const CwUnpackPrelude &a, const CwUnpackPrelude &b);
inline bool operator!=(const CwUnpackPrelude &a, const CwUnpackPrelude &b) { return !(a == b); }
bool operator==(const CwStateLayout &a, const CwStateLayout &b);
inline bool operator!=(const CwStateLayout &a, const CwStateLayout &b) { return !(a == b); }

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
  std::vector<HostSegment> host_segments;
  std::vector<HostHaloPlanDescriptor> host_halo_plans;
  std::vector<HostCallbackDescriptor> host_callbacks;
  std::vector<CurlUpdate> db_updates;
  std::vector<CylindricalRadialPrefix> cylindrical_radial_prefixes;
  std::vector<BfastUpdate> bfast_updates;
  std::vector<BetaUpdate> beta_updates;
  std::vector<CylindricalMOverRUpdate> cylindrical_m_updates;
  std::vector<CylindricalAxisUpdate> cylindrical_axis_updates;
  std::vector<SlabRef> cylindrical_zero_slabs;
  std::vector<CylindricalOriginAction> cylindrical_origin_actions;
  std::vector<ConstitutiveUpdate> eh_updates;
  std::vector<PolarizationUpdateGroup> polarization_groups;
  std::vector<PolarizationUpdate> polarization_updates;
  std::vector<PolarizationSubtraction> polarization_subtractions;
  std::vector<MultilevelPopulationUpdate> multilevel_population_updates;
  std::vector<MultilevelPopulationTerm> multilevel_population_terms;
  std::vector<MultilevelTransitionUpdate> multilevel_transition_updates;
  std::vector<double> multilevel_coefficients;
  std::vector<LegacyFluxUpdate> legacy_flux_updates;
  std::vector<LegacyFluxTerm> legacy_flux_terms;
  std::vector<MagneticStateArray> magnetic_state_arrays;
  std::vector<MaterialRefreshArray> material_refresh_arrays;
  CwStateLayout cw_state_layout;
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
    host_segments.clear();
    host_halo_plans.clear();
    host_callbacks.clear();
    db_updates.clear();
    cylindrical_radial_prefixes.clear();
    bfast_updates.clear();
    beta_updates.clear();
    cylindrical_m_updates.clear();
    cylindrical_axis_updates.clear();
    cylindrical_zero_slabs.clear();
    cylindrical_origin_actions.clear();
    eh_updates.clear();
    polarization_groups.clear();
    polarization_updates.clear();
    polarization_subtractions.clear();
    multilevel_population_updates.clear();
    multilevel_population_terms.clear();
    multilevel_transition_updates.clear();
    multilevel_coefficients.clear();
    legacy_flux_updates.clear();
    legacy_flux_terms.clear();
    magnetic_state_arrays.clear();
    material_refresh_arrays.clear();
    cw_state_layout.clear();
    magnetic_half_step = MagneticHalfStep();
    dft_updates.clear();
    material_phase_target_signature = 0;
    source_signature = 0;
    signature = 0;
  }
};

/* Append one susceptibility's portable polarization actions as a single
   contiguous group. Deterministic recurrence rows always precede every noise
   row. PR6 supplies live descriptor rows; PR5 owns only this ordering and
   access contract. */
void append_polarization_update_group(fields &f, StepPlan &plan, Operation &op,
                                      const std::vector<PolarizationUpdate> &recurrences,
                                      const std::vector<PolarizationUpdate> &noise_additions);

void append_multilevel_update_group(fields &f, StepPlan &plan, Operation &op,
                                    const MultilevelPopulationUpdate &population,
                                    const std::vector<MultilevelPopulationTerm> &terms,
                                    const std::vector<MultilevelTransitionUpdate> &transitions,
                                    const std::vector<double> &gamma_matrix,
                                    const std::vector<double> &alpha);

bool operator==(const PolarizationUpdateGroup &a, const PolarizationUpdateGroup &b);
bool operator==(const MultilevelPopulationTerm &a, const MultilevelPopulationTerm &b);
bool operator==(const MultilevelPopulationUpdate &a, const MultilevelPopulationUpdate &b);
bool operator==(const MultilevelTransitionUpdate &a, const MultilevelTransitionUpdate &b);
inline bool operator!=(const PolarizationUpdateGroup &a, const PolarizationUpdateGroup &b) {
  return !(a == b);
}
inline bool operator!=(const MultilevelPopulationTerm &a, const MultilevelPopulationTerm &b) {
  return !(a == b);
}
inline bool operator!=(const MultilevelPopulationUpdate &a,
                       const MultilevelPopulationUpdate &b) {
  return !(a == b);
}
inline bool operator!=(const MultilevelTransitionUpdate &a,
                       const MultilevelTransitionUpdate &b) {
  return !(a == b);
}

/* solve_cw constructs its RHS by applying every matching source, including
   integrated sources, directly to primary D/B with primary -= current*dt.
   This is intentionally distinct from the ordinary integrated-source path,
   which subtracts dipoles through f_minus_p. */
enum class CwRhsSourceMode : uint32_t {
  primary_subtract_current_dt_including_integrated = 0
};

struct CwRhsSourceDescriptor {
  uint32_t source_descriptor_index;
  uint32_t source_ordinal;
  CwRhsSourceMode mode;
};

/* A reference into the sole canonical solve_cw StepPlan. Descriptor spans are
   repeated only as checked lookup metadata; the operation schedule and the
   descriptors themselves remain owned by StepPlan. */
struct CwStepOperationRef {
  uint32_t operation_index;
  OpKind kind;
  field_type ft;
  uint32_t descriptor_index;
  uint32_t descriptor_count;
  uint32_t polarization_subtraction_index;
  uint32_t polarization_subtraction_count;

  CwStepOperationRef()
      : operation_index(UINT32_MAX), kind(OpKind::num_kinds),
        ft(NO_FIELD_TYPE), descriptor_index(0), descriptor_count(0),
        polarization_subtraction_index(0), polarization_subtraction_count(0) {}
};

struct CwRhsStage {
  field_type ft;
  double source_time_offset;
  uint32_t source_time_index;
  uint32_t source_time_count;
  uint32_t source_index;
  uint32_t source_count;
  CwStepOperationRef boundary;
  CwStepOperationRef constitutive;
  std::vector<BufferAccess> accesses;
};

struct CwUnpackDescriptorRefs {
  CwStepOperationRef first_boundary;
  CwStepOperationRef constitutive;
  CwStepOperationRef second_boundary;
  bool skip_w_components;
  bool invalidate_field_values;

  CwUnpackDescriptorRefs() : skip_w_components(false), invalidate_field_values(false) {}
};

struct CwDftDescriptorRef {
  uint32_t descriptor_index;
  int chunk;
  component c;
  int decimation_factor;
  uint32_t due_scalar_slot;
};

/* CW final-DFT predicates occupy the fixed StepScalars predicate bitset. */
const uint32_t cw_dft_predicate_capacity = 64u * 64u;

/* Source/monitor-dependent extension of CwStateLayout. Source and DFT indices
   refer to the DescriptorSet owned by the same prepared fields object (and,
   later, its compiled ordinary executable); the fingerprints bind that owner.
   The plan does not copy payload tables or duplicate the solve_cw schedule. */
struct CwPlan {
  uint64_t state_layout_signature;
  uint64_t step_plan_signature;
  std::vector<CwRhsStage> rhs_stages;
  std::vector<CwRhsSourceDescriptor> rhs_sources;
  CwUnpackDescriptorRefs unpack;
  std::vector<CwDftDescriptorRef> final_dfts;
  std::vector<BufferAccess> rhs_accesses;
  std::vector<BufferAccess> unpack_accesses;
  std::vector<BufferAccess> final_dft_accesses;
  uint32_t source_time_count;
  uint32_t rhs_source_count;
  uint32_t final_dft_count;
  uint64_t source_fingerprint;
  uint64_t monitor_fingerprint;
  uint64_t signature;

  CwPlan()
      : state_layout_signature(0), step_plan_signature(0), source_time_count(0),
        rhs_source_count(0), final_dft_count(0), source_fingerprint(0), monitor_fingerprint(0),
        signature(0) {}
  void clear();
};

bool operator==(const CwRhsSourceDescriptor &a, const CwRhsSourceDescriptor &b);
inline bool operator!=(const CwRhsSourceDescriptor &a, const CwRhsSourceDescriptor &b) {
  return !(a == b);
}
bool operator==(const CwStepOperationRef &a, const CwStepOperationRef &b);
inline bool operator!=(const CwStepOperationRef &a, const CwStepOperationRef &b) {
  return !(a == b);
}
bool operator==(const CwRhsStage &a, const CwRhsStage &b);
inline bool operator!=(const CwRhsStage &a, const CwRhsStage &b) { return !(a == b); }
bool operator==(const CwUnpackDescriptorRefs &a, const CwUnpackDescriptorRefs &b);
inline bool operator!=(const CwUnpackDescriptorRefs &a, const CwUnpackDescriptorRefs &b) {
  return !(a == b);
}
bool operator==(const CwDftDescriptorRef &a, const CwDftDescriptorRef &b);
inline bool operator!=(const CwDftDescriptorRef &a, const CwDftDescriptorRef &b) {
  return !(a == b);
}
bool operator==(const CwPlan &a, const CwPlan &b);
inline bool operator!=(const CwPlan &a, const CwPlan &b) { return !(a == b); }

CwPlan build_cw_plan(fields &f, const StepPlan &step_plan);
uint64_t compute_cw_plan_signature(const CwPlan &plan);
bool validate_cw_plan(fields &f, const StepPlan &step_plan, const CwPlan &plan,
                      std::string *error = NULL);

/* A direct transcription of the order in src/step.cpp. It omits empty work and
   NEVER reorders what remains. It is not a scheduler that infers Maxwell's
   equations from array dependencies: if a reviewer cannot line this up against
   fields::step_once by eye, it is wrong. */
StepPlan build_step_plan(fields &f, StepProgram program);

/* Builds only the two canonical polarization operations from the current
   descriptor/catalog authority.  Resident noisy preflight uses this narrow
   view so unrelated reduced-catalog curl/material rows cannot invalidate a
   stable ordinary plan. */
StepPlan build_polarization_validation_plan(fields &f);

/* Direct transcription of fields_to_array/array_to_fields.  Construction is
   rank-local and performs no collective operation. */
CwStateLayout build_cw_state_layout(fields &f);

/* Structural identity and exact comparison used by later backend preflight.
   Validation compares against a freshly rebuilt canonical layout rather than
   accepting a matching hash as proof of equality. */
uint64_t compute_cw_state_layout_signature(const CwStateLayout &layout);
bool validate_cw_state_layout(fields &f, const CwStateLayout &layout,
                              std::string *error = NULL);

/* Structural identity used by device executables and tests. */
uint64_t compute_step_plan_signature(const StepPlan &plan);

/* Construct the marker's canonical access union in first-use order. Equal
   ArrayRefs are collapsed and read+write is promoted to read_write. Distinct
   ranges carrying one ArrayId are rejected: host segments transfer complete,
   unambiguous catalog allocations. */
std::vector<BufferAccess>
build_host_segment_access_union(const StepPlan &plan, uint32_t operation_index,
                                uint32_t operation_count,
                                const std::vector<BufferAccess> &additional,
                                const CpuArrayCatalog *catalog = NULL);

/* Structural validation for host_callback markers. This checks only the PR 5
   operation-span contract. PR 6 additionally validates callback identities
   and the bounds of callback spans once that vector exists. */
bool validate_host_segments(const StepPlan &plan, std::string *error = NULL);

/* Resolve every callback against current live state, then compare the
   callback/segment representation with an independently rebuilt plan. */
bool validate_host_callback_plan(fields &f, const StepPlan &plan,
                                 std::string *error = NULL);

/* Resolve a logical descriptor against the current connectivity generation
   and validate its complete owned-side host mirror. */
bool resolve_host_halo_plan(fields &f, const HostHaloPlanDescriptor &descriptor,
                            const HaloPlan *&resolved, std::string *error = NULL);

/* Recomputed by resident preflight so externally-mutated target topology is
   rejected before phase_material_mix changes current coefficients. */
uint64_t compute_material_phase_target_signature(const fields &f);
/* Public beta/BFAST assignment cannot update the per-chunk coefficients that
   the CPU and descriptor builders consume. Resident execution therefore
   rejects any outer/chunk/prepared mismatch instead of reusing stale code. */
bool coordinate_state_matches(const fields &f, const StepPlan *prepared);

/* Human-readable operation sequence, for the trace test and for debugging. */
void format_step_plan(const StepPlan &p, std::vector<std::string> &out);

} // namespace meep

#endif // MEEP_BACKEND_STEP_PLAN_HPP
