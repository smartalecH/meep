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

#include <stdio.h>
#include <string.h>

#include <algorithm>

#include "backend/descriptors.hpp"
#include "backend/step_plan.hpp"
#include "backend/halo_plan.hpp"
#include "backend/lifecycle.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

namespace meep {

const char *op_kind_name(OpKind k) {
  switch (k) {
    case OpKind::restore_magnetic_fields: return "restore_magnetic_fields";
    case OpKind::phase_material: return "phase_material";
    case OpKind::update_material_coefficients: return "update_material_coefficients";
    case OpKind::evaluate_source_scalars: return "evaluate_source_scalars";
    case OpKind::update_db: return "update_db";
    case OpKind::update_eh: return "update_eh";
    case OpKind::update_polarization: return "update_polarization";
    case OpKind::apply_sources: return "apply_sources";
    case OpKind::zero_boundary: return "zero_boundary";
    case OpKind::pack_halo: return "pack_halo";
    case OpKind::transfer_halo: return "transfer_halo";
    case OpKind::exchange_local: return "exchange_local";
    case OpKind::unpack_halo: return "unpack_halo";
    case OpKind::update_flux_half: return "update_flux_half";
    case OpKind::update_flux: return "update_flux";
    case OpKind::increment_time: return "increment_time";
    case OpKind::update_dft: return "update_dft";
    case OpKind::synchronize_magnetic_fields: return "synchronize_magnetic_fields";
    case OpKind::finite_value_check: return "finite_value_check";
    case OpKind::reduction: return "reduction";
    case OpKind::host_callback: return "host_callback";
    case OpKind::pack_state: return "pack_state";
    case OpKind::unpack_state: return "unpack_state";
    case OpKind::num_kinds: break;
  }
  return "?";
}

static const char *ft_name(field_type ft) {
  switch (ft) {
    case E_stuff: return "E";
    case H_stuff: return "H";
    case D_stuff: return "D";
    case B_stuff: return "B";
    case PE_stuff: return "PE";
    case PH_stuff: return "PH";
    case WE_stuff: return "WE";
    case WH_stuff: return "WH";
    default: return "?";
  }
}

namespace {

ArrayId find_array(fields &f, int chunk, array_kind kind, int c, int cmp, int aux) {
  if (!f.array_catalog) return invalid_array();
  return f.array_catalog->find(StorageKey{chunk, int(kind), c, cmp, aux});
}

UpdateRegion make_region(const grid_volume &gv, int chunk, component c, int cmp, const ivec &begin,
                         const ivec &end) {
  UpdateRegion r;
  r.chunk = chunk;
  r.c = c;
  r.cmp = cmp;
  r.begin = begin;
  r.end = end;
  r.base = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const direction d = gv.yucky_direction(axis);
    r.counts[axis] = size_t((end.yucky_val(axis) - begin.yucky_val(axis)) / 2 + 1);
    r.strides[axis] = gv.stride(d);
    r.base += size_t((begin.yucky_val(axis) - gv.little_corner().yucky_val(axis)) / 2) *
              size_t(r.strides[axis]);
  }
  r.variant_key = 0;
  return r;
}

PmlProfile no_pml_profile() {
  PmlProfile p;
  p.sig = p.kap = p.siginv = invalid_array();
  p.base = 0;
  p.strides[0] = p.strides[1] = p.strides[2] = 0;
  return p;
}

PmlProfile make_pml_profile(fields &f, const fields_chunk &fc, int chunk, direction d,
                            const ivec &begin) {
  if (d == NO_DIRECTION) return no_pml_profile();
  PmlProfile p;
  p.sig = find_array(f, chunk, array_kind::pml_sig, -1, -1, int(d));
  p.kap = find_array(f, chunk, array_kind::pml_kap, -1, -1, int(d));
  p.siginv = find_array(f, chunk, array_kind::pml_siginv, -1, -1, int(d));
  p.base = begin.in_direction(d) - fc.gv.little_corner().in_direction(d);
  for (int axis = 0; axis < 3; ++axis)
    p.strides[axis] = fc.gv.yucky_direction(axis) == d ? 2 : 0;
  return p;
}

void add_access(fields &f, Operation &op, ArrayId id, AccessMode mode) {
  if (!is_valid(id) || !f.array_catalog || id.value >= f.array_catalog->size()) return;
  for (size_t i = 0; i < op.accesses.size(); ++i) {
    BufferAccess &existing = op.accesses[i];
    if (existing.array.id != id) continue;
    if (existing.mode != mode) existing.mode = AccessMode::read_write;
    return;
  }
  const ArraySpec &spec = f.array_catalog->spec(id);
  op.accesses.push_back(BufferAccess{ArrayRef{id, 0, spec.elements}, mode});
}

struct CurlSources {
  bool have_plus;
  bool have_minus;
  component plus_component;
  component minus_component;
  direction plus_direction;
  direction minus_direction;

  CurlSources()
      : have_plus(false), have_minus(false), plus_component(NO_COMPONENT),
        minus_component(NO_COMPONENT), plus_direction(NO_DIRECTION), minus_direction(NO_DIRECTION) {
  }
};

bool cross_is_negative(direction a, direction b) {
  if (a >= R) a = direction(a - 3);
  if (b >= R) b = direction(b - 3);
  return ((3 + b - a) % 3) == 2;
}

direction cross_direction(direction a, direction b) {
  if (a == b) meep::abort("bug - cross_direction expects different directions");
  const bool cylindrical = a >= R || b >= R;
  if (a >= R) a = direction(a - 3);
  if (b >= R) b = direction(b - 3);
  direction result = direction((3 + 2 * a - b) % 3);
  if (cylindrical && result < Z) result = direction(result + 3);
  return result;
}

CurlSources curl_sources_for(const fields_chunk &fc, component target) {
  CurlSources result;
  const direction target_direction = component_direction(target);
  FOR_COMPONENTS(source) {
    if (!((is_electric(target) && is_magnetic(source)) || (is_D(target) && is_magnetic(source)) ||
          (is_magnetic(target) && is_electric(source)) || (is_B(target) && is_electric(source))))
      continue;
    const direction source_direction = component_direction(source);
    if (target_direction == source_direction || !fc.gv.has_field(source) ||
        !fc.gv.has_field(target))
      continue;
    const direction derivative = cross_direction(target_direction, source_direction);
    if (!(has_direction(fc.gv.dim, derivative) ||
          (fc.gv.dim == Dcyl && has_field_direction(fc.gv.dim, derivative))))
      continue;
    if (cross_is_negative(source_direction, target_direction)) {
      result.have_minus = true;
      result.minus_component = source;
      result.minus_direction = derivative;
    }
    else {
      result.have_plus = true;
      result.plus_component = source;
      result.plus_direction = derivative;
    }
  }
  return result;
}

class StepPlanBuilder {
public:
  explicit StepPlanBuilder(fields &f, StepProgram program) : f_(f) {
    plan_.program = program;
    plan_.coordinate_generation = generation(f, MutationKind::coordinate_definition);
    plan_.beta = f.beta;
    plan_.cylindrical_m = f.m;
    plan_.bfast_scaled_k = f.bfast_scaled_k;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
      const fields_chunk &fc = *f.chunks[chunk];
      plan_.cylindrical_origin_r.push_back(fc.gv.origin_r());
      plan_.cylindrical_zero_near_origin.push_back(fc.zero_fields_near_cylorigin ? 1 : 0);
    }
    plan_.source_signature = f_.descriptors ? source_plan_signature(f_.descriptors->sources) : 0;
  }

  Operation &add(OpKind k, field_type ft = field_type(NUM_FIELD_TYPES), Guard g = guard_always(),
                 double src_offset = 0.0) {
    Operation op;
    op.kind = k;
    op.descriptor_index = 0;
    op.descriptor_count = 0;
    op.beta_descriptor_index = 0;
    op.beta_descriptor_count = 0;
    op.cylindrical_m_descriptor_index = 0;
    op.cylindrical_m_descriptor_count = 0;
    op.cylindrical_origin_action_index = 0;
    op.cylindrical_origin_action_count = 0;
    op.polarization_subtraction_index = 0;
    op.polarization_subtraction_count = 0;
    op.magnetic_state_index = 0;
    op.magnetic_state_count = 0;
    op.source_descriptor_index = 0;
    op.source_descriptor_count = 0;
    op.guard = g;
    op.ft = ft;
    op.source_time_offset = src_offset;
    plan_.operations.push_back(op);
    return plan_.operations.back();
  }

  void add_db(field_type ft);
  void add_eh(field_type ft, Guard guard = guard_always());
  void add_source_evaluation(Guard guard, double src_offset);
  void add_sources(field_type ft);
  void add_dfts();

  uint32_t operation_count() const { return uint32_t(plan_.operations.size()); }

  Operation &add_magnetic_marker(OpKind kind, AccessMode mode) {
    Operation &op = add(kind, field_type(NUM_FIELD_TYPES), guard_variant(0));
    if (plan_.magnetic_state_arrays.empty()) build_magnetic_state_arrays();
    op.magnetic_state_index = 0;
    op.magnetic_state_count = uint32_t(plan_.magnetic_state_arrays.size());
    for (const MagneticStateArray &entry : plan_.magnetic_state_arrays)
      add_access(f_, op, entry.live,
                 mode == AccessMode::write || entry.average ? mode : AccessMode::read);
    if (kind == OpKind::synchronize_magnetic_fields) {
      const uint32_t schedule[] = {
          plan_.magnetic_half_step.evaluate_b_sources, plan_.magnetic_half_step.update_b,
          plan_.magnetic_half_step.apply_b_sources,    plan_.magnetic_half_step.transfer_b,
          plan_.magnetic_half_step.evaluate_h_sources, plan_.magnetic_half_step.update_h,
          plan_.magnetic_half_step.transfer_h};
      const size_t marker = plan_.operations.size() - 1;
      for (size_t i = 0; i < sizeof(schedule) / sizeof(schedule[0]); ++i) {
        if (schedule[i] == UINT32_MAX) continue;
        if (schedule[i] >= marker) meep::abort("invalid magnetic half-step operation index");
        const Operation &referenced = plan_.operations[schedule[i]];
        for (const BufferAccess &access : referenced.accesses)
          add_access(f_, op, access.array.id, access.mode);
      }
    }
    return op;
  }

  void set_magnetic_half_step(uint32_t evaluate_b_sources, uint32_t update_b,
                              uint32_t apply_b_sources, uint32_t transfer_b,
                              uint32_t evaluate_h_sources, uint32_t update_h, uint32_t transfer_h) {
    plan_.magnetic_half_step.evaluate_b_sources = evaluate_b_sources;
    plan_.magnetic_half_step.update_b = update_b;
    plan_.magnetic_half_step.apply_b_sources = apply_b_sources;
    plan_.magnetic_half_step.transfer_b = transfer_b;
    plan_.magnetic_half_step.evaluate_h_sources = evaluate_h_sources;
    plan_.magnetic_half_step.update_h = update_h;
    plan_.magnetic_half_step.transfer_h = transfer_h;
  }

  void add_finite_value_check() {
    Operation &op = add(OpKind::finite_value_check);
    if (!f_.storage_plan) return;

    /* Device diagnostics scan physical field arrays only. Preserve catalog
       order so the first failing access is deterministic, and skip aliases so
       one allocation cannot be attributed twice. StorageKey supplies the
       chunk/component/cmp identity used by device backends for diagnostics. */
    const StoragePlan &storage = *f_.storage_plan;
    for (size_t i = 0; i < storage.arrays.size(); ++i) {
      const ArraySpec &spec = storage.arrays[i];
      const StorageKey &key = storage.keys[i];
      if (key.kind != int(array_kind::f) || spec.element_type != ElementType::realnum_value ||
          !spec.elements || is_valid(spec.alias_of))
        continue;
      add_access(f_, op, spec.id, AccessMode::read);
    }
  }

  void add_if(bool present, OpKind k, field_type ft = field_type(NUM_FIELD_TYPES),
              double src_offset = 0.0) {
    if (present) add(k, ft, guard_static(true), src_offset);
  }

  /* One semantic boundary step.
   *
   * The plan asks for this to expand into zero-metal, pack, transfer and
   * unpack. On CPU it stays fused into a single transfer_halo, for two
   * concrete reasons in step_boundaries: comms_manager's lifetime spans all
   * three communication phases (its destructor is what completes the
   * outstanding requests), and the receives are posted *before* the metal
   * zeroing, so the plan's nominal order is not the implementation's order
   * either. Splitting it would change the request-completion point and the
   * timing scopes and buys nothing on CPU.
   *
   * The four OpKinds exist in the enum because Phase 2 needs the vocabulary;
   * see the deviation note in ~/meep-phase1-pr5.md. */
  void add_boundaries(field_type ft, Guard g = guard_always()) {
    add(OpKind::transfer_halo, ft, g);
  }

  StepPlan finish() {
    plan_.signature = signature_for(plan_);
    return plan_;
  }

  static uint64_t signature_for(const StepPlan &plan) {
    uint64_t sig = 0xcbf29ce484222325ull;
    mix(sig, uint64_t(plan.program));
    mix_double(sig, plan.beta);
    mix_double(sig, plan.cylindrical_m);
    for (double k : plan.bfast_scaled_k) mix_double(sig, k);
    for (double origin : plan.cylindrical_origin_r) mix_double(sig, origin);
    for (uint8_t zero : plan.cylindrical_zero_near_origin) mix(sig, uint64_t(zero));
    mix(sig, plan.source_signature);
    for (const Operation &op : plan.operations) {
      sig ^= uint64_t(op.kind) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.ft) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.guard.kind) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.guard.scalar_slot) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.guard.variant_index) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.descriptor_index) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.descriptor_count) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.beta_descriptor_index) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      sig ^= uint64_t(op.beta_descriptor_count) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      mix(sig, uint64_t(op.cylindrical_m_descriptor_index));
      mix(sig, uint64_t(op.cylindrical_m_descriptor_count));
      mix(sig, uint64_t(op.cylindrical_origin_action_index));
      mix(sig, uint64_t(op.cylindrical_origin_action_count));
      sig ^= uint64_t(op.polarization_subtraction_index) + 0x9e3779b97f4a7c15ull + (sig << 6) +
             (sig >> 2);
      sig ^= uint64_t(op.polarization_subtraction_count) + 0x9e3779b97f4a7c15ull + (sig << 6) +
             (sig >> 2);
      mix(sig, uint64_t(op.magnetic_state_index));
      mix(sig, uint64_t(op.magnetic_state_count));
      sig ^= uint64_t(op.source_descriptor_index) + 0x9e3779b97f4a7c15ull + (sig << 6) +
             (sig >> 2);
      sig ^= uint64_t(op.source_descriptor_count) + 0x9e3779b97f4a7c15ull + (sig << 6) +
             (sig >> 2);
      uint64_t source_bits = 0;
      static_assert(sizeof(source_bits) == sizeof(op.source_time_offset), "double is not 64-bit");
      memcpy(&source_bits, &op.source_time_offset, sizeof(source_bits));
      sig ^= source_bits + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      for (const BufferAccess &access : op.accesses) {
        sig ^= uint64_t(access.array.id.value) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
        sig ^= uint64_t(access.array.offset) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
        sig ^= uint64_t(access.array.elements) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
        sig ^= uint64_t(access.mode) + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
      }
    }
    for (const CurlUpdate &d : plan.db_updates)
      hash_curl(sig, d);
    for (const CylindricalRadialPrefix &d : plan.cylindrical_radial_prefixes)
      hash_cylindrical_radial_prefix(sig, d);
    for (const BfastUpdate &d : plan.bfast_updates)
      hash_bfast(sig, d);
    for (const BetaUpdate &d : plan.beta_updates)
      hash_beta(sig, d);
    for (const CylindricalMOverRUpdate &d : plan.cylindrical_m_updates)
      hash_cylindrical_m(sig, d);
    for (const CylindricalAxisUpdate &d : plan.cylindrical_axis_updates)
      hash_cylindrical_axis(sig, d);
    for (const SlabRef &d : plan.cylindrical_zero_slabs)
      hash_slab(sig, d);
    for (const CylindricalOriginAction &d : plan.cylindrical_origin_actions) {
      mix(sig, uint64_t(d.kind));
      mix(sig, uint64_t(d.index));
    }
    for (const ConstitutiveUpdate &d : plan.eh_updates)
      hash_constitutive(sig, d);
    for (const PolarizationUpdate &d : plan.polarization_updates)
      hash_polarization(sig, d);
    for (const PolarizationSubtraction &d : plan.polarization_subtractions)
      hash_polarization_subtraction(sig, d);
    for (const MagneticStateArray &d : plan.magnetic_state_arrays)
      hash_magnetic_state(sig, d);
    hash_magnetic_half_step(sig, plan.magnetic_half_step);
    for (const DftDescriptor &d : plan.dft_updates) hash_dft(sig, d);
    return sig;
  }

private:
  static void mix(uint64_t &sig, uint64_t value) {
    sig ^= value + 0x9e3779b97f4a7c15ull + (sig << 6) + (sig >> 2);
  }
  static void mix_double(uint64_t &sig, double value) {
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    mix(sig, bits);
  }
  static void hash_region(uint64_t &sig, const UpdateRegion &r) {
    mix(sig, uint64_t(r.chunk));
    mix(sig, uint64_t(r.c));
    mix(sig, uint64_t(r.cmp));
    for (int i = 0; i < 3; ++i) {
      mix(sig, uint64_t(r.begin.yucky_val(i)));
      mix(sig, uint64_t(r.end.yucky_val(i)));
    }
    mix(sig, uint64_t(r.base));
    for (int i = 0; i < 3; ++i) {
      mix(sig, uint64_t(r.counts[i]));
      mix(sig, uint64_t(r.strides[i]));
    }
    mix(sig, uint64_t(r.variant_key));
  }
  static void hash_id(uint64_t &sig, ArrayId id) { mix(sig, uint64_t(id.value)); }
  static void hash_ref(uint64_t &sig, const ArrayRef &ref) {
    hash_id(sig, ref.id);
    mix(sig, uint64_t(ref.offset));
    mix(sig, uint64_t(ref.elements));
  }
  static void hash_ivec(uint64_t &sig, const ivec &v) {
    mix(sig, uint64_t(v.dim));
    LOOP_OVER_DIRECTIONS(v.dim, d) { mix(sig, uint64_t(v.in_direction(d))); }
  }
  static void hash_vec(uint64_t &sig, const vec &v) {
    mix(sig, uint64_t(v.dim));
    LOOP_OVER_DIRECTIONS(v.dim, d) { mix_double(sig, v.in_direction(d)); }
  }
  static void hash_pml(uint64_t &sig, const PmlProfile &p) {
    hash_id(sig, p.sig);
    hash_id(sig, p.kap);
    hash_id(sig, p.siginv);
    mix(sig, uint64_t(p.base));
    for (int i = 0; i < 3; ++i)
      mix(sig, uint64_t(p.strides[i]));
  }
  static void hash_curl(uint64_t &sig, const CurlUpdate &d) {
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.plus_source);
    hash_id(sig, d.minus_source);
    mix(sig, uint64_t(d.plus_stride));
    mix(sig, uint64_t(d.minus_stride));
    hash_id(sig, d.target_u);
    hash_id(sig, d.conductivity);
    hash_id(sig, d.condinv);
    hash_id(sig, d.target_cond);
    hash_pml(sig, d.pml);
    hash_pml(sig, d.pml_u);
    mix_double(sig, d.dtdx);
    mix_double(sig, d.dt);
    mix(sig, uint64_t(d.radial_prefix_index));
    mix(sig, uint64_t(d.bfast_update_index));
  }
  static void hash_cylindrical_radial_prefix(uint64_t &sig, const CylindricalRadialPrefix &d) {
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.target_component));
    mix(sig, uint64_t(d.source_component));
    mix(sig, uint64_t(d.cmp));
    hash_id(sig, d.source);
    hash_id(sig, d.scratch);
    mix(sig, uint64_t(d.nr));
    mix(sig, uint64_t(d.nz));
    mix(sig, uint64_t(d.row_stride));
    mix(sig, uint64_t(d.source_elements));
    mix(sig, uint64_t(d.scratch_elements));
    mix_double(sig, d.ir0);
  }
  static void hash_bfast(uint64_t &sig, const BfastUpdate &d) {
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.source1);
    hash_id(sig, d.source2);
    mix(sig, uint64_t(d.stride1));
    mix(sig, uint64_t(d.stride2));
    hash_id(sig, d.f_bfast);
    hash_id(sig, d.target_u);
    hash_id(sig, d.condinv);
    hash_id(sig, d.target_cond);
    hash_pml(sig, d.pml);
    hash_pml(sig, d.pml_u);
    mix_double(sig, d.k1);
    mix_double(sig, d.k2);
  }
  static void hash_beta(uint64_t &sig, const BetaUpdate &d) {
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.source);
    hash_id(sig, d.target_u);
    hash_id(sig, d.condinv);
    hash_id(sig, d.target_cond);
    hash_pml(sig, d.pml);
    hash_pml(sig, d.pml_u);
    mix_double(sig, d.betadt);
  }
  static void hash_cylindrical_m(uint64_t &sig, const CylindricalMOverRUpdate &d) {
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.source);
    hash_id(sig, d.target_u);
    hash_id(sig, d.condinv);
    hash_id(sig, d.target_cond);
    hash_pml(sig, d.pml);
    hash_pml(sig, d.pml_u);
    mix_double(sig, d.numerator);
    mix(sig, uint64_t(d.raw_radial_start));
  }
  static void hash_cylindrical_axis(uint64_t &sig, const CylindricalAxisUpdate &d) {
    mix(sig, uint64_t(d.kind));
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.source1);
    hash_id(sig, d.source2);
    mix(sig, uint64_t(d.source1_neighbor_offset));
    mix(sig, uint64_t(d.source2_offset));
    hash_id(sig, d.target_u);
    hash_id(sig, d.conductivity);
    hash_id(sig, d.condinv);
    hash_id(sig, d.target_cond);
    hash_pml(sig, d.pml);
    hash_pml(sig, d.pml_u);
    mix_double(sig, d.scale);
    mix_double(sig, d.source2_multiplier);
    mix_double(sig, d.dt);
  }
  static void hash_slab(uint64_t &sig, const SlabRef &d) {
    hash_id(sig, d.array);
    mix(sig, uint64_t(d.base));
    for (int i = 0; i < 3; ++i) {
      mix(sig, uint64_t(d.counts[i]));
      mix(sig, uint64_t(d.strides[i]));
    }
  }
  static void hash_constitutive(uint64_t &sig, const ConstitutiveUpdate &d) {
    hash_region(sig, d.region);
    hash_id(sig, d.target);
    hash_id(sig, d.base_primary);
    hash_id(sig, d.base_cross1);
    hash_id(sig, d.base_cross2);
    hash_id(sig, d.primary);
    hash_id(sig, d.cross1);
    hash_id(sig, d.cross2);
    hash_id(sig, d.diagonal);
    hash_id(sig, d.offdiagonal1);
    hash_id(sig, d.offdiagonal2);
    mix(sig, uint64_t(d.primary_stride));
    mix(sig, uint64_t(d.cross1_stride));
    mix(sig, uint64_t(d.cross2_stride));
    hash_id(sig, d.chi2);
    hash_id(sig, d.chi3);
    hash_id(sig, d.target_w);
    hash_id(sig, d.previous_w);
    hash_pml(sig, d.pml);
  }
  static void hash_polarization(uint64_t &sig, const PolarizationUpdate &d) {
    mix(sig, uint64_t(d.kind));
    hash_region(sig, d.region);
    mix(sig, uint64_t(d.state_index));
    hash_id(sig, d.p);
    hash_id(sig, d.p_prev);
    hash_id(sig, d.p_cross1);
    hash_id(sig, d.p_prev_cross1);
    hash_id(sig, d.p_cross2);
    hash_id(sig, d.p_prev_cross2);
    hash_id(sig, d.primary_w);
    hash_id(sig, d.cross_w1);
    hash_id(sig, d.cross_w2);
    hash_id(sig, d.diagonal_sigma);
    hash_id(sig, d.offdiagonal_sigma1);
    hash_id(sig, d.offdiagonal_sigma2);
    mix(sig, uint64_t(d.primary_stride));
    mix(sig, uint64_t(d.cross_stride1));
    mix(sig, uint64_t(d.cross_stride2));
    mix_double(sig, d.omega_0);
    mix_double(sig, d.gamma);
    mix_double(sig, d.alpha);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        mix_double(sig, d.gyro_tensor[i][j]);
    mix(sig, uint64_t(d.gyro_model));
    mix_double(sig, d.dt);
  }
  static void hash_polarization_subtraction(uint64_t &sig, const PolarizationSubtraction &d) {
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.c));
    mix(sig, uint64_t(d.cmp));
    mix(sig, uint64_t(d.state_index));
    hash_id(sig, d.target);
    hash_id(sig, d.p);
    mix(sig, uint64_t(d.elements));
  }
  static void hash_dft(uint64_t &sig, const DftDescriptor &d) {
    hash_id(sig, d.accumulator);
    hash_id(sig, d.phase_scratch);
    hash_ref(sig, d.source_field);
    hash_ref(sig, d.source_field_imag);
    mix(sig, uint64_t(d.omega.size()));
    for (size_t i = 0; i < d.omega.size(); ++i) mix_double(sig, d.omega[i]);
    mix_double(sig, d.scale.real());
    mix_double(sig, d.scale.imag());
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.c));
    mix(sig, uint64_t(d.avg1));
    mix(sig, uint64_t(d.avg2));
    hash_ivec(sig, d.is);
    hash_ivec(sig, d.ie);
    hash_ivec(sig, d.is_old);
    hash_ivec(sig, d.ie_old);
    mix(sig, uint64_t(d.persist));
    mix(sig, uint64_t(d.decimation_factor));
    mix(sig, uint64_t(d.due_scalar_slot));
    hash_vec(sig, d.weights.s0);
    hash_vec(sig, d.weights.s1);
    hash_vec(sig, d.weights.e0);
    hash_vec(sig, d.weights.e1);
    mix_double(sig, d.dV0);
    mix_double(sig, d.dV1);
    mix(sig, uint64_t(d.include_dV_and_interp_weights));
    mix(sig, uint64_t(d.sqrt_dV_and_interp_weights));
    mix(sig, uint64_t(d.N));
    mix(sig, uint64_t(d.Nomega));
  }

  static void hash_magnetic_state(uint64_t &sig, const MagneticStateArray &d) {
    mix(sig, uint64_t(d.chunk));
    mix(sig, uint64_t(d.c));
    mix(sig, uint64_t(d.cmp));
    mix(sig, uint64_t(d.family));
    hash_id(sig, d.live);
    mix(sig, uint64_t(d.elements));
    mix(sig, uint64_t(d.average));
  }

  static void hash_magnetic_half_step(uint64_t &sig, const MagneticHalfStep &d) {
    mix(sig, uint64_t(d.evaluate_b_sources));
    mix(sig, uint64_t(d.update_b));
    mix(sig, uint64_t(d.apply_b_sources));
    mix(sig, uint64_t(d.transfer_b));
    mix(sig, uint64_t(d.evaluate_h_sources));
    mix(sig, uint64_t(d.update_h));
    mix(sig, uint64_t(d.transfer_h));
  }

  void build_magnetic_state_arrays() {
    const array_kind kinds[] = {array_kind::f, array_kind::f_u, array_kind::f_w, array_kind::f_cond,
                                array_kind::f_bfast};
    const MagneticStateFamily families[] = {
        MagneticStateFamily::primary, MagneticStateFamily::u, MagneticStateFamily::w,
        MagneticStateFamily::conductivity, MagneticStateFamily::bfast};
    for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
      if (!f_.chunks[chunk]->is_mine()) continue;
      const fields_chunk &fc = *f_.chunks[chunk];
      const int components = fc.is_real ? 1 : 2;
      for (int family_type = 0; family_type < 2; ++family_type) {
        const field_type ft = family_type == 0 ? B_stuff : H_stuff;
        FOR_FT_COMPONENTS(ft, c) for (int cmp = 0; cmp < components; ++cmp) {
          const ArrayId primary = find_array(f_, chunk, array_kind::f, int(c), cmp, 0);
          if (!is_valid(primary)) continue;
          if (ft == H_stuff && is_valid(f_.array_catalog->spec(primary).alias_of)) continue;
          for (size_t family = 0; family < sizeof(kinds) / sizeof(kinds[0]); ++family) {
            const ArrayId live = find_array(f_, chunk, kinds[family], int(c), cmp, 0);
            if (!is_valid(live)) continue;
            const ArraySpec &spec = f_.array_catalog->spec(live);
            plan_.magnetic_state_arrays.push_back(
                MagneticStateArray{chunk, c, cmp, families[family], live, spec.elements,
                                   families[family] == MagneticStateFamily::primary});
          }
        }
      }
    }
  }

  fields &f_;
  StepPlan plan_;

  void attach_source_span(Operation &op, field_type ft, bool integrated) {
    if (!f_.descriptors) return;
    const std::vector<SourceDescriptor> &sources = f_.descriptors->sources.sources;
    bool started = false, finished = false;
    for (size_t i = 0; i < sources.size(); ++i) {
      const SourceDescriptor &d = sources[i];
      const bool matches = d.ft == ft && d.integrated == integrated;
      if (matches) {
        if (finished) meep::abort("source descriptors for one operation are not contiguous");
        if (!started) {
          op.source_descriptor_index = uint32_t(i);
          started = true;
        }
        ++op.source_descriptor_count;
        if (integrated) {
          add_access(f_, op, d.destination, AccessMode::read);
          add_access(f_, op, d.destination_imag, AccessMode::read);
          add_access(f_, op, d.integrated_destination, AccessMode::read_write);
          add_access(f_, op, d.integrated_destination_imag, AccessMode::read_write);
        }
        else {
          add_access(f_, op, d.destination, AccessMode::read_write);
          add_access(f_, op, d.destination_imag, AccessMode::read_write);
          add_access(f_, op, d.condinv, AccessMode::read);
        }
      }
      else if (started)
        finished = true;
    }
  }
};

void StepPlanBuilder::add_source_evaluation(Guard guard, double src_offset) {
  Operation &op = add(OpKind::evaluate_source_scalars, field_type(NUM_FIELD_TYPES), guard,
                      src_offset);
  if (f_.descriptors)
    op.descriptor_count = uint32_t(f_.descriptors->sources.source_times.size());
}

void StepPlanBuilder::add_sources(field_type ft) {
  Operation &op = add(OpKind::apply_sources, ft);
  attach_source_span(op, ft, false);
}

void StepPlanBuilder::add_dfts() {
  Operation &op = add(OpKind::update_dft, field_type(NUM_FIELD_TYPES), guard_device(0));
  op.descriptor_index = uint32_t(plan_.dft_updates.size());
  if (f_.descriptors) {
    for (size_t i = 0; i < f_.descriptors->dfts.size(); ++i) {
      const DftDescriptor &d = f_.descriptors->dfts[i];
      plan_.dft_updates.push_back(d);
      add_access(f_, op, d.accumulator, AccessMode::read_write);
      add_access(f_, op, d.phase_scratch, AccessMode::write);
      add_access(f_, op, d.source_field.id, AccessMode::read);
      add_access(f_, op, d.source_field_imag.id, AccessMode::read);
    }
  }
  op.descriptor_count = uint32_t(plan_.dft_updates.size()) - op.descriptor_index;
}

void StepPlanBuilder::add_db(field_type ft) {
  Operation &op = add(OpKind::update_db, ft);
  op.descriptor_index = uint32_t(plan_.db_updates.size());
  op.beta_descriptor_index = uint32_t(plan_.beta_updates.size());
  op.cylindrical_m_descriptor_index = uint32_t(plan_.cylindrical_m_updates.size());
  op.cylindrical_origin_action_index = uint32_t(plan_.cylindrical_origin_actions.size());

  for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
    if (!f_.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f_.chunks[chunk];
    const int components = fc.is_real ? 1 : 2;
    for (size_t tile = 0; tile < fc.gvs_tiled.size(); ++tile) {
      const grid_volume &sub = fc.gvs_tiled[tile];
      for (int cmp = 0; cmp < components; ++cmp)
        FOR_FT_COMPONENTS(ft, cc) {
          const ArrayId target = find_array(f_, chunk, array_kind::f, int(cc), cmp, 0);
          if (!is_valid(target)) continue;

          const CurlSources sources = curl_sources_for(fc, cc);
          const direction dc = component_direction(cc);
          const direction dsig0 = cycle_direction(fc.gv.dim, dc, 1);
          const direction dsig = fc.s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
          const direction dsigu0 = cycle_direction(fc.gv.dim, dc, 2);
          const direction dsigu = fc.s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;

          CurlUpdate d;
          d.region =
              make_region(fc.gv, chunk, cc, cmp, sub.little_owned_corner0(cc), sub.big_corner());
          d.target = target;
          d.plus_source = sources.have_plus ? find_array(f_, chunk, array_kind::f,
                                                         int(sources.plus_component), cmp, 0)
                                            : invalid_array();
          d.minus_source = sources.have_minus ? find_array(f_, chunk, array_kind::f,
                                                           int(sources.minus_component), cmp, 0)
                                              : invalid_array();
          d.plus_stride = sources.have_plus ? fc.gv.stride(sources.plus_direction) : 0;
          d.minus_stride = sources.have_minus ? fc.gv.stride(sources.minus_direction) : 0;
          if (ft == D_stuff) {
            d.plus_stride = -d.plus_stride;
            d.minus_stride = -d.minus_stride;
          }
          d.target_u = find_array(f_, chunk, array_kind::f_u, int(cc), cmp, 0);
          d.conductivity = find_array(f_, chunk, array_kind::conductivity, int(cc), -1, int(dc));
          d.condinv = find_array(f_, chunk, array_kind::condinv, int(cc), -1, int(dc));
          d.target_cond = find_array(f_, chunk, array_kind::f_cond, int(cc), cmp, 0);
          d.pml = make_pml_profile(f_, fc, chunk, dsig, d.region.begin);
          d.pml_u = make_pml_profile(f_, fc, chunk, dsigu, d.region.begin);
          d.dtdx = fc.Courant;
          d.dt = fc.dt;
          d.radial_prefix_index = UINT32_MAX;
          d.bfast_update_index = UINT32_MAX;

          if (fc.gv.dim == Dcyl) {
            switch (dc) {
              case R: d.plus_source = invalid_array(); break;
              case P: break;
              case Z: {
                CylindricalRadialPrefix prefix;
                prefix.chunk = chunk;
                prefix.target_component = cc;
                prefix.source_component = sources.plus_component;
                prefix.cmp = cmp;
                prefix.source = d.plus_source;
                prefix.scratch = find_array(f_, chunk, array_kind::f_rderiv_int, -1, -1, 0);
                prefix.nr = size_t(fc.gv.nr());
                prefix.nz = size_t(fc.gv.nz());
                prefix.row_stride = prefix.nz + 1;
                prefix.source_elements =
                    is_valid(prefix.source) ? f_.array_catalog->spec(prefix.source).elements : 0;
                prefix.scratch_elements =
                    is_valid(prefix.scratch) ? f_.array_catalog->spec(prefix.scratch).elements : 0;
                const realnum ir0 = fc.gv.origin_r() * fc.gv.a +
                                    0.5 * fc.gv.iyee_shift(sources.plus_component).in_direction(R);
                prefix.ir0 = ir0;
                d.radial_prefix_index = uint32_t(plan_.cylindrical_radial_prefixes.size());
                plan_.cylindrical_radial_prefixes.push_back(prefix);
                d.plus_source = prefix.scratch;
                d.minus_source = invalid_array();
                add_access(f_, op, prefix.source, AccessMode::read);
                add_access(f_, op, prefix.scratch, AccessMode::read_write);
                break;
              }
              default: meep::abort("bug - non-cylindrical field component in Dcyl");
            }
          }

          if (is_valid(d.plus_source) && is_valid(d.minus_source))
            d.region.variant_key |= curl_has_second_derivative;
          if (dsig != NO_DIRECTION) d.region.variant_key |= curl_has_pml;
          if (dsigu != NO_DIRECTION) d.region.variant_key |= curl_has_pml_aux;
          if (is_valid(d.conductivity)) d.region.variant_key |= curl_has_conductivity;
          if (fc.bfast_scaled_k[0] || fc.bfast_scaled_k[1] || fc.bfast_scaled_k[2]) {
            d.region.variant_key |= curl_has_bfast;

            BfastUpdate b;
            b.region = d.region;
            b.region.variant_key = 0;
            b.target = d.target;
            b.source1 = d.plus_source;
            b.source2 = d.minus_source;
            b.stride1 = d.plus_stride;
            b.stride2 = d.minus_stride;
            b.f_bfast = find_array(f_, chunk, array_kind::f_bfast, int(cc), cmp, 0);
            b.target_u = d.target_u;
            b.condinv = d.condinv;
            b.target_cond = d.target_cond;
            b.pml = d.pml;
            b.pml_u = d.pml_u;
            realnum k1 = sources.have_minus
                             ? fc.bfast_scaled_k[component_index(sources.minus_component)]
                             : 0;
            realnum k2 =
                sources.have_plus ? fc.bfast_scaled_k[component_index(sources.plus_component)] : 0;
            if (ft == D_stuff) {
              k1 = -k1;
              k2 = -k2;
            }
            b.k1 = k1;
            b.k2 = k2;
            if (dsig != NO_DIRECTION) b.region.variant_key |= bfast_has_pml;
            if (dsigu != NO_DIRECTION) b.region.variant_key |= bfast_has_pml_aux;
            if (is_valid(b.condinv)) b.region.variant_key |= bfast_has_conductivity;
            d.bfast_update_index = uint32_t(plan_.bfast_updates.size());
            plan_.bfast_updates.push_back(b);

            add_access(f_, op, b.target, AccessMode::read_write);
            add_access(f_, op, b.source1, AccessMode::read);
            add_access(f_, op, b.source2, AccessMode::read);
            add_access(f_, op, b.f_bfast, AccessMode::read_write);
            add_access(f_, op, b.target_u, AccessMode::read_write);
            add_access(f_, op, b.condinv, AccessMode::read);
            add_access(f_, op, b.target_cond, AccessMode::read_write);
            add_access(f_, op, b.pml.siginv, AccessMode::read);
            add_access(f_, op, b.pml_u.siginv, AccessMode::read);
          }

          plan_.db_updates.push_back(d);
          add_access(f_, op, d.target, AccessMode::read_write);
          add_access(f_, op, d.plus_source, AccessMode::read);
          add_access(f_, op, d.minus_source, AccessMode::read);
          add_access(f_, op, d.target_u, AccessMode::read_write);
          add_access(f_, op, d.conductivity, AccessMode::read);
          add_access(f_, op, d.condinv, AccessMode::read);
          add_access(f_, op, d.target_cond, AccessMode::read_write);
          add_access(f_, op, d.pml.sig, AccessMode::read);
          add_access(f_, op, d.pml.kap, AccessMode::read);
          add_access(f_, op, d.pml.siginv, AccessMode::read);
          add_access(f_, op, d.pml_u.sig, AccessMode::read);
          add_access(f_, op, d.pml_u.kap, AccessMode::read);
          add_access(f_, op, d.pml_u.siginv, AccessMode::read);
        }
    }
  }
  op.descriptor_count = uint32_t(plan_.db_updates.size()) - op.descriptor_index;

  /* Match step_db.cpp exactly: special-kz rows run only after every ordinary
     curl tile in this half-step has completed, and cover the full chunk rather
     than an individual tile. */
  for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
    if (!f_.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f_.chunks[chunk];
    if (fc.gv.dim != D2 || fc.beta == 0) continue;
    const int components = fc.is_real ? 1 : 2;
    for (int cmp = 0; cmp < components; ++cmp)
      for (direction dc = X; dc <= Y; dc = direction(dc + 1)) {
        const component target_component = direction_component(first_field_component(ft), dc);
        const component source_component =
            direction_component(ft == D_stuff ? Hx : Ex, dc == X ? Y : X);
        const ArrayId target = find_array(f_, chunk, array_kind::f, int(target_component), cmp, 0);
        const ArrayId opposite_source =
            find_array(f_, chunk, array_kind::f, int(source_component), 1 - cmp, 0);
        const ArrayId same_source =
            find_array(f_, chunk, array_kind::f, int(source_component), cmp, 0);
        const ArrayId source = is_valid(opposite_source) ? opposite_source : same_source;
        /* step_beta is a no-op without either operand. Do not turn such a row
           into executable backend work. */
        if (!is_valid(target) || !is_valid(source)) continue;

        const direction dsig0 = cycle_direction(fc.gv.dim, dc, 1);
        const direction dsig = fc.s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
        const direction dsigu0 = cycle_direction(fc.gv.dim, dc, 2);
        const direction dsigu = fc.s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;

        BetaUpdate d;
        d.region = make_region(fc.gv, chunk, target_component, cmp,
                               fc.gv.little_owned_corner0(target_component), fc.gv.big_corner());
        d.target = target;
        d.source = source;
        d.target_u = find_array(f_, chunk, array_kind::f_u, int(target_component), cmp, 0);
        d.condinv = find_array(f_, chunk, array_kind::condinv, int(target_component), -1, int(dc));
        d.target_cond = find_array(f_, chunk, array_kind::f_cond, int(target_component), cmp, 0);
        d.pml = make_pml_profile(f_, fc, chunk, dsig, d.region.begin);
        d.pml_u = make_pml_profile(f_, fc, chunk, dsigu, d.region.begin);
        const realnum betadt =
            2 * pi * fc.beta * fc.dt * (dc == X ? +1 : -1) *
            (is_valid(opposite_source) ? (ft == D_stuff ? -1 : +1) * (2 * cmp - 1) : 1);
        d.betadt = betadt;
        if (dsig != NO_DIRECTION) d.region.variant_key |= beta_has_pml;
        if (dsigu != NO_DIRECTION) d.region.variant_key |= beta_has_pml_aux;
        if (is_valid(d.condinv)) d.region.variant_key |= beta_has_conductivity;

        plan_.beta_updates.push_back(d);
        add_access(f_, op, d.target, AccessMode::read_write);
        add_access(f_, op, d.source, AccessMode::read);
        add_access(f_, op, d.target_u, AccessMode::read_write);
        add_access(f_, op, d.condinv, AccessMode::read);
        add_access(f_, op, d.target_cond, AccessMode::read_write);
        add_access(f_, op, d.pml.siginv, AccessMode::read);
        add_access(f_, op, d.pml_u.siginv, AccessMode::read);
      }
  }
  op.beta_descriptor_count = uint32_t(plan_.beta_updates.size()) - op.beta_descriptor_index;

  /* Cylindrical m/r terms are chunk-wide tails after every tiled curl and
     paired BFAST postpass. */
  for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
    if (!f_.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f_.chunks[chunk];
    if (fc.gv.dim != Dcyl || fc.m == 0) continue;
    const int components = fc.is_real ? 1 : 2;
    for (int cmp = 0; cmp < components; ++cmp)
      FOR_FT_COMPONENTS(ft, cc) {
        const direction dc = component_direction(cc);
        if (dc != R && dc != Z) continue;
        const ArrayId target = find_array(f_, chunk, array_kind::f, int(cc), cmp, 0);
        const CurlSources tail_sources = curl_sources_for(fc, cc);
        const component source_component =
            dc == R ? tail_sources.plus_component : tail_sources.minus_component;
        if (source_component == NO_COMPONENT) continue;
        const ArrayId source =
            find_array(f_, chunk, array_kind::f, int(source_component), 1 - cmp, 0);
        if (!is_valid(target) || !is_valid(source)) continue;

        const direction dsig0 = cycle_direction(fc.gv.dim, dc, 1);
        const direction dsig = fc.s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
        const direction dsigu0 = cycle_direction(fc.gv.dim, dc, 2);
        const direction dsigu = fc.s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;

        CylindricalMOverRUpdate d;
        d.region =
            make_region(fc.gv, chunk, cc, cmp, fc.gv.little_owned_corner0(cc), fc.gv.big_corner());
        d.target = target;
        d.source = source;
        d.target_u = find_array(f_, chunk, array_kind::f_u, int(cc), cmp, 0);
        d.condinv = find_array(f_, chunk, array_kind::condinv, int(cc), -1, int(dc));
        d.target_cond = dsig != NO_DIRECTION && is_valid(d.condinv)
                            ? find_array(f_, chunk, array_kind::f_cond, int(cc), cmp, 0)
                            : invalid_array();
        d.pml = make_pml_profile(f_, fc, chunk, dsig, d.region.begin);
        d.pml_u = make_pml_profile(f_, fc, chunk, dsigu, d.region.begin);
        const realnum numerator =
            2 * fc.m * (1 - 2 * cmp) * (1 - 2 * (ft == B_stuff)) * (1 - 2 * (dc == R)) * fc.Courant;
        d.numerator = numerator;
        d.raw_radial_start = d.region.begin.in_direction(R);
        if (dsig != NO_DIRECTION) d.region.variant_key |= cylindrical_m_has_pml;
        if (dsigu != NO_DIRECTION) d.region.variant_key |= cylindrical_m_has_pml_aux;
        if (is_valid(d.condinv)) d.region.variant_key |= cylindrical_m_has_conductivity;
        plan_.cylindrical_m_updates.push_back(d);

        add_access(f_, op, d.target, AccessMode::read_write);
        add_access(f_, op, d.source, AccessMode::read);
        add_access(f_, op, d.target_u, AccessMode::read_write);
        add_access(f_, op, d.condinv, AccessMode::read);
        add_access(f_, op, d.target_cond, AccessMode::read_write);
        add_access(f_, op, d.pml.siginv, AccessMode::read);
        add_access(f_, op, d.pml_u.siginv, AccessMode::read);
      }
  }
  op.cylindrical_m_descriptor_count =
      uint32_t(plan_.cylindrical_m_updates.size()) - op.cylindrical_m_descriptor_index;

  auto add_zero_slab = [&](fields_chunk &fc, ArrayId id, int radial_row) {
    if (!is_valid(id)) return;
    SlabRef slab;
    slab.array = id;
    slab.base = ptrdiff_t(radial_row) * ptrdiff_t(fc.gv.nz() + 1);
    slab.counts[0] = fc.gv.nz() + 1;
    slab.counts[1] = slab.counts[2] = 1;
    slab.strides[0] = 1;
    slab.strides[1] = slab.strides[2] = 0;
    plan_.cylindrical_zero_slabs.push_back(slab);
    plan_.cylindrical_origin_actions.push_back(CylindricalOriginAction{
        CylindricalOriginActionKind::zero_slab, uint32_t(plan_.cylindrical_zero_slabs.size() - 1)});
    add_access(f_, op, id, AccessMode::write);
  };

  auto add_component_zero = [&](fields_chunk &fc, int chunk, component c, int cmp, int radial_row) {
    add_zero_slab(fc, find_array(f_, chunk, array_kind::f, int(c), cmp, 0), radial_row);
    add_zero_slab(fc, find_array(f_, chunk, array_kind::f_cond, int(c), cmp, 0), radial_row);
    add_zero_slab(fc, find_array(f_, chunk, array_kind::f_u, int(c), cmp, 0), radial_row);
  };

  auto add_family_zero = [&](fields_chunk &fc, int chunk, field_type family, int cmp,
                             int radial_row) {
    const array_kind kinds[] = {array_kind::f, array_kind::f_cond, array_kind::f_u};
    for (array_kind kind : kinds)
      FOR_FT_COMPONENTS(family, c)
    add_zero_slab(fc, find_array(f_, chunk, kind, int(c), cmp, 0), radial_row);
  };

  /* Origin arithmetic and zero slabs execute after the m/r tail. */
  for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
    if (!f_.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f_.chunks[chunk];
    if (fc.gv.dim != Dcyl || fc.gv.origin_r() != 0.0) continue;
    const int components = fc.is_real ? 1 : 2;
    for (int cmp = 0; cmp < components; ++cmp) {
      component target_component = NO_COMPONENT;
      CylindricalAxisKind kind = CylindricalAxisKind::m0_dz;
      ArrayId source1 = invalid_array(), source2 = invalid_array();
      ptrdiff_t source1_neighbor_offset = 0, source2_offset = 0;
      realnum scale = 0, source2_multiplier = 0;
      component zero_after_axis = NO_COMPONENT;

      if (fc.m == 0 && ft == D_stuff) {
        target_component = Dz;
        source1 = find_array(f_, chunk, array_kind::f, int(Hp), cmp, 0);
        scale = fc.Courant * 4;
        zero_after_axis = Dp;
      }
      else if (fc.m == 0 && ft == B_stuff) {
        if (is_valid(find_array(f_, chunk, array_kind::f, int(Br), cmp, 0)))
          add_component_zero(fc, chunk, Br, cmp, 0);
      }
      else if (fabs(fc.m) == 1) {
        kind = CylindricalAxisKind::abs_m1;
        target_component = ft == D_stuff ? Dp : Br;
        const int sd = ft == D_stuff ? +1 : -1;
        source1 = find_array(f_, chunk, array_kind::f, int(ft == D_stuff ? Hr : Ep), cmp, 0);
        source2 = find_array(f_, chunk, array_kind::f, int(ft == D_stuff ? Hz : Ez),
                             ft == D_stuff ? cmp : 1 - cmp, 0);
        source1_neighbor_offset = -sd;
        source2_offset = ft == D_stuff ? 0 : fc.gv.nz() + 1;
        scale = sd * fc.Courant;
        source2_multiplier = ft == D_stuff ? 2 : (1 - 2 * cmp) * fc.m;
        if (ft == D_stuff) zero_after_axis = Dz;
      }
      else if (fc.m != 0) {
        int radial_rows = 1;
        if (fc.zero_fields_near_cylorigin) {
          radial_rows = 0;
          const double rmax = fabs(fc.m) - int(fc.gv.origin_r() * fc.gv.a + 0.5);
          while (radial_rows <= fc.gv.nr() && radial_rows < rmax)
            ++radial_rows;
        }
        for (int row = 0; row < radial_rows; ++row)
          add_family_zero(fc, chunk, ft, cmp, row);
      }

      const ArrayId target =
          target_component == NO_COMPONENT
              ? invalid_array()
              : find_array(f_, chunk, array_kind::f, int(target_component), cmp, 0);
      if (is_valid(target) && is_valid(source1) &&
          (kind == CylindricalAxisKind::m0_dz || is_valid(source2))) {
        const direction dc = component_direction(target_component);
        const direction dsig0 = cycle_direction(fc.gv.dim, dc, 1);
        const direction dsig = fc.s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
        const direction dsigu0 = cycle_direction(fc.gv.dim, dc, 2);
        const direction dsigu = fc.s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;
        ivec begin = fc.gv.little_owned_corner(target_component);
        ivec end = fc.gv.big_owned_corner(target_component);
        end.set_direction(R, 0);

        CylindricalAxisUpdate d;
        d.kind = kind;
        d.region = make_region(fc.gv, chunk, target_component, cmp, begin, end);
        d.target = target;
        d.source1 = source1;
        d.source2 = source2;
        d.source1_neighbor_offset = source1_neighbor_offset;
        d.source2_offset = source2_offset;
        d.target_u = find_array(f_, chunk, array_kind::f_u, int(target_component), cmp, 0);
        d.conductivity =
            find_array(f_, chunk, array_kind::conductivity, int(target_component), -1, int(dc));
        d.condinv = find_array(f_, chunk, array_kind::condinv, int(target_component), -1, int(dc));
        d.target_cond = find_array(f_, chunk, array_kind::f_cond, int(target_component), cmp, 0);
        if (!is_valid(d.target_cond)) {
          d.conductivity = invalid_array();
          d.condinv = invalid_array();
        }
        d.pml = make_pml_profile(f_, fc, chunk, dsig, d.region.begin);
        d.pml_u = make_pml_profile(f_, fc, chunk, dsigu, d.region.begin);
        d.scale = scale;
        d.source2_multiplier = source2_multiplier;
        d.dt = fc.dt;
        if (dsig != NO_DIRECTION) d.region.variant_key |= cylindrical_axis_has_pml;
        if (dsigu != NO_DIRECTION) d.region.variant_key |= cylindrical_axis_has_pml_aux;
        if (is_valid(d.target_cond)) d.region.variant_key |= cylindrical_axis_has_conductivity;
        plan_.cylindrical_axis_updates.push_back(d);
        plan_.cylindrical_origin_actions.push_back(
            CylindricalOriginAction{CylindricalOriginActionKind::axis_update,
                                    uint32_t(plan_.cylindrical_axis_updates.size() - 1)});

        add_access(f_, op, d.target, AccessMode::read_write);
        add_access(f_, op, d.source1, AccessMode::read);
        add_access(f_, op, d.source2, AccessMode::read);
        add_access(f_, op, d.target_u, AccessMode::read_write);
        add_access(f_, op, d.conductivity, AccessMode::read);
        add_access(f_, op, d.condinv, AccessMode::read);
        add_access(f_, op, d.target_cond, AccessMode::read_write);
        add_access(f_, op, d.pml.sig, AccessMode::read);
        add_access(f_, op, d.pml.kap, AccessMode::read);
        add_access(f_, op, d.pml.siginv, AccessMode::read);
        add_access(f_, op, d.pml_u.sig, AccessMode::read);
        add_access(f_, op, d.pml_u.kap, AccessMode::read);
        add_access(f_, op, d.pml_u.siginv, AccessMode::read);
        if (zero_after_axis != NO_COMPONENT) add_component_zero(fc, chunk, zero_after_axis, cmp, 0);
      }
    }
  }
  op.cylindrical_origin_action_count =
      uint32_t(plan_.cylindrical_origin_actions.size()) - op.cylindrical_origin_action_index;
}

void StepPlanBuilder::add_eh(field_type ft, Guard guard) {
  Operation &op = add(OpKind::update_eh, ft, guard);
  op.descriptor_index = uint32_t(plan_.eh_updates.size());
  const field_type ft2 = ft == E_stuff ? D_stuff : B_stuff;

  for (int chunk = 0; chunk < f_.num_chunks; ++chunk) {
    if (!f_.chunks[chunk]->is_mine()) continue;
    fields_chunk &fc = *f_.chunks[chunk];
    const int components = fc.is_real ? 1 : 2;
    for (size_t tile = 0; tile < fc.gvs_eh[ft].size(); ++tile) {
      const grid_volume &sub = fc.gvs_eh[ft][tile];
      for (int cmp = 0; cmp < components; ++cmp)
        FOR_FT_COMPONENTS(ft, ec) {
          if (!fc.f[ec][cmp]) continue;
          const component dc = field_type_component(ft2, ec);
          if (fc.f[ec][cmp] == fc.f[dc][cmp]) continue;

          const direction dec = component_direction(ec);
          const direction d1 = cycle_direction(fc.gv.dim, dec, 1);
          const direction d2 = cycle_direction(fc.gv.dim, dec, 2);
          const component dc1 = direction_component(dc, d1);
          const component dc2 = direction_component(dc, d2);
          const direction dsigw = fc.s->sigsize[dec] > 1 ? dec : NO_DIRECTION;

          ConstitutiveUpdate d;
          d.region =
              make_region(fc.gv, chunk, ec, cmp, sub.little_owned_corner0(ec), sub.big_corner());
          d.target = find_array(f_, chunk, array_kind::f, int(ec), cmp, 0);
          const ArrayId primary_minus_p =
              find_array(f_, chunk, array_kind::f_minus_p, int(dc), cmp, 0);
          const ArrayId cross1_minus_p =
              find_array(f_, chunk, array_kind::f_minus_p, int(dc1), cmp, 0);
          const ArrayId cross2_minus_p =
              find_array(f_, chunk, array_kind::f_minus_p, int(dc2), cmp, 0);
          d.base_primary = find_array(f_, chunk, array_kind::f, int(dc), cmp, 0);
          d.base_cross1 = find_array(f_, chunk, array_kind::f, int(dc1), cmp, 0);
          d.base_cross2 = find_array(f_, chunk, array_kind::f, int(dc2), cmp, 0);
          d.primary = is_valid(primary_minus_p) ? primary_minus_p : d.base_primary;
          d.cross1 = is_valid(cross1_minus_p) ? cross1_minus_p : d.base_cross1;
          d.cross2 = is_valid(cross2_minus_p) ? cross2_minus_p : d.base_cross2;
          d.diagonal = find_array(f_, chunk, array_kind::chi1inv, int(ec), -1, int(dec));
          d.offdiagonal1 = find_array(f_, chunk, array_kind::chi1inv, int(ec), -1, int(d1));
          d.offdiagonal2 = find_array(f_, chunk, array_kind::chi1inv, int(ec), -1, int(d2));
          d.primary_stride = fc.gv.stride(dec) * (ft == H_stuff ? -1 : 1);
          d.cross1_stride = fc.gv.stride(d1) * (ft == H_stuff ? -1 : 1);
          d.cross2_stride = fc.gv.stride(d2) * (ft == H_stuff ? -1 : 1);

          /* Match step_update_EDHB's normalization: the one surviving
             off-diagonal term is always slot 1. */
          if ((!is_valid(d.cross1) && is_valid(d.cross2)) ||
              (is_valid(d.cross1) && is_valid(d.cross2) && !is_valid(d.offdiagonal1) &&
               is_valid(d.offdiagonal2))) {
            std::swap(d.base_cross1, d.base_cross2);
            std::swap(d.cross1, d.cross2);
            std::swap(d.offdiagonal1, d.offdiagonal2);
            std::swap(d.cross1_stride, d.cross2_stride);
          }

          d.chi2 = find_array(f_, chunk, array_kind::chi2, int(ec), -1, 0);
          d.chi3 = find_array(f_, chunk, array_kind::chi3, int(ec), -1, 0);
          d.target_w = find_array(f_, chunk, array_kind::f_w, int(ec), cmp, 0);
          d.previous_w = find_array(f_, chunk, array_kind::f_w_prev, int(ec), cmp, 0);
          d.pml = make_pml_profile(f_, fc, chunk, dsigw, d.region.begin);
          if (is_valid(d.offdiagonal1)) d.region.variant_key |= constitutive_one_offdiagonal;
          if (is_valid(d.offdiagonal2)) d.region.variant_key |= constitutive_two_offdiagonals;
          if (dsigw != NO_DIRECTION) d.region.variant_key |= constitutive_has_pml;
          if (is_valid(d.chi2) || is_valid(d.chi3))
            d.region.variant_key |= constitutive_has_nonlinearity;
          if (is_valid(primary_minus_p) || is_valid(cross1_minus_p) || is_valid(cross2_minus_p))
            d.region.variant_key |= constitutive_has_minus_p;
          if (tile == 0 && is_valid(d.previous_w))
            d.region.variant_key |= constitutive_copy_w_previous;

          plan_.eh_updates.push_back(d);
          add_access(f_, op, d.target, AccessMode::read_write);
          add_access(f_, op, d.base_primary, AccessMode::read);
          add_access(f_, op, d.base_cross1, AccessMode::read);
          add_access(f_, op, d.base_cross2, AccessMode::read);
          add_access(f_, op, d.primary,
                     d.primary != d.base_primary ? AccessMode::read_write : AccessMode::read);
          add_access(f_, op, d.cross1,
                     d.cross1 != d.base_cross1 ? AccessMode::read_write : AccessMode::read);
          add_access(f_, op, d.cross2,
                     d.cross2 != d.base_cross2 ? AccessMode::read_write : AccessMode::read);
          add_access(f_, op, d.diagonal, AccessMode::read);
          add_access(f_, op, d.offdiagonal1, AccessMode::read);
          add_access(f_, op, d.offdiagonal2, AccessMode::read);
          add_access(f_, op, d.chi2, AccessMode::read);
          add_access(f_, op, d.chi3, AccessMode::read);
          add_access(f_, op, d.target_w, AccessMode::read_write);
          add_access(f_, op, d.previous_w, AccessMode::write);
          add_access(f_, op, d.pml.sig, AccessMode::read);
          add_access(f_, op, d.pml.kap, AccessMode::read);
          add_access(f_, op, d.pml.siginv, AccessMode::read);

          if (fc.gv.dim == Dcyl) {
            ivec axis_begin = sub.little_owned_corner(ec);
            if (axis_begin.in_direction(R) == 0) {
              ConstitutiveUpdate axis = d;
              ivec axis_end = sub.big_corner();
              axis_end.set_direction(R, 0);
              axis.region = make_region(fc.gv, chunk, ec, cmp, axis_begin, axis_end);
              axis.region.variant_key =
                  d.region.variant_key &
                  ~(constitutive_one_offdiagonal | constitutive_two_offdiagonals |
                    constitutive_has_minus_p | constitutive_copy_w_previous);
              if (axis.primary != axis.base_primary)
                axis.region.variant_key |= constitutive_has_minus_p;
              axis.region.variant_key |= constitutive_axis_override;
              axis.base_cross1 = axis.base_cross2 = invalid_array();
              axis.cross1 = axis.cross2 = invalid_array();
              axis.offdiagonal1 = axis.offdiagonal2 = invalid_array();
              axis.cross1_stride = axis.cross2_stride = 0;
              axis.previous_w = invalid_array();
              plan_.eh_updates.push_back(axis);

              add_access(f_, op, axis.target, AccessMode::read_write);
              add_access(f_, op, axis.base_primary, AccessMode::read);
              add_access(f_, op, axis.primary,
                         axis.primary != axis.base_primary ? AccessMode::read_write
                                                           : AccessMode::read);
              add_access(f_, op, axis.diagonal, AccessMode::read);
              add_access(f_, op, axis.chi2, AccessMode::read);
              add_access(f_, op, axis.chi3, AccessMode::read);
              add_access(f_, op, axis.target_w, AccessMode::read_write);
              add_access(f_, op, axis.pml.sig, AccessMode::read);
              add_access(f_, op, axis.pml.kap, AccessMode::read);
              add_access(f_, op, axis.pml.siginv, AccessMode::read);
            }
          }
        }
    }
  }
  op.descriptor_count = uint32_t(plan_.eh_updates.size()) - op.descriptor_index;
  attach_source_span(op, ft2, true);
}

} // namespace

uint64_t compute_step_plan_signature(const StepPlan &plan) {
  return StepPlanBuilder::signature_for(plan);
}

/* Transcribed from fields::step_once. Read the two side by side.
 *
 * step_boundaries() begins with zero_metal for every owned chunk and then does
 * pack/transfer/unpack, which is why add_boundaries emits zero_boundary first.
 */
StepPlan build_step_plan(fields &f, StepProgram program) {
  StepPlanBuilder p(f, program);
  const bool cw = program == StepProgram::solve_cw;

  const bool has_sources = f.sources != NULL;
  const bool has_fluxes = f.fluxes != NULL;
  const bool phasing = f.is_phasing();
  /* update_dfts is disabled under solve_cw (see dft.cpp), so the operation is
     statically absent rather than guarded.

     dft_chunks are per-chunk, so a rank owning no chunk that intersects a
     monitor builds a program of a different shape from its peers. On CPU that
     is harmless -- update_dfts() is a no-op with no chunks -- and it is
     deliberately left alone.

     Reducing it with or_to_all here DEADLOCKS, and it is worth being precise
     about why, because it is the plan's section 6.4 hazard showing up for
     real. build_step_plan runs from step_plan_for(), which rebuilds when
     dirty_executable is set -- and dirty_executable can be set on a *subset*
     of ranks, because the lazy-allocation sites in step_db, update_eh and
     update_pols are rank-local by design (each returns a flag saying "I
     allocated, reconnect"). One rank rebuilds, enters the reduction, and waits
     for peers that never arrive. Observed: tests/flux hung indefinitely at
     np=2 with the reduction in place.

     The real fix is to make the *rebuild decision* collective, the way
     connect_chunks() gates on sync_chunk_connections(). That is follow-up
     work, recorded in ~/meep-phase1-pr5.md -- and it applies equally to the
     collectives PR 4 put inside classify(). */
  bool has_dfts = false;
  for (int i = 0; i < f.num_chunks && !has_dfts; ++i)
    if (f.chunks[i]->is_mine() && f.chunks[i]->dft_chunks) has_dfts = true;

  /* Magnetic re-synchronization is a graph_variant: the whole program differs
     when synchronized fields are active. */
  p.add_magnetic_marker(OpKind::restore_magnetic_fields, AccessMode::write);

  /* phase_material's conditional E/H reconciliation is a segment_boundary: the
     condition is an or_to_all over all chunks, so the host has to evaluate it
     between device segments. */
  if (phasing) {
    p.add(OpKind::phase_material, field_type(NUM_FIELD_TYPES), guard_static(true));
    p.add_source_evaluation(guard_segment(0), 0.5);
    p.add_eh(H_stuff, guard_segment(0));
    p.add_boundaries(H_stuff, guard_segment(0));
    p.add_source_evaluation(guard_segment(0), 1.0);
    p.add_eh(E_stuff, guard_segment(0));
    p.add_boundaries(E_stuff, guard_segment(0));
  }

  p.add(OpKind::update_material_coefficients);

  uint32_t magnetic_evaluate_b = UINT32_MAX;
  if (has_sources) {
    magnetic_evaluate_b = p.operation_count();
    p.add_source_evaluation(guard_static(true), 0.0);
  }
  const uint32_t magnetic_update_b = p.operation_count();
  p.add_db(B_stuff);
  const uint32_t magnetic_apply_b = p.operation_count();
  p.add_sources(B_stuff);
  const uint32_t magnetic_transfer_b = p.operation_count();
  p.add_boundaries(B_stuff);

  uint32_t magnetic_evaluate_h = UINT32_MAX;
  if (has_sources) {
    magnetic_evaluate_h = p.operation_count();
    p.add_source_evaluation(guard_static(true), 0.5);
  }
  const uint32_t magnetic_update_h = p.operation_count();
  p.add_eh(H_stuff);
  p.add_boundaries(WH_stuff);
  p.add(OpKind::update_polarization, H_stuff);
  p.add_boundaries(PH_stuff);
  const uint32_t magnetic_transfer_h = p.operation_count();
  p.add_boundaries(H_stuff);

  p.add_if(has_fluxes, OpKind::update_flux_half);

  if (has_sources) p.add_source_evaluation(guard_static(true), 0.5);
  p.add_db(D_stuff);
  p.add_sources(D_stuff);
  p.add_boundaries(D_stuff);

  if (has_sources) p.add_source_evaluation(guard_static(true), 1.0);
  p.add_eh(E_stuff);
  p.add_boundaries(WE_stuff);
  p.add(OpKind::update_polarization, E_stuff);
  p.add_boundaries(PE_stuff);
  p.add_boundaries(E_stuff);

  p.add_if(has_fluxes, OpKind::update_flux);
  p.add(OpKind::increment_time);
  /* solve_cw flattens all D/B state plus f_u, f_cond, f_bfast, the f_w
     companion and (where f_w exists) the E/H array into one contiguous
     complex<realnum> vector on every BiCGSTAB matrix-vector product. The ops
     are recorded so Phase 2 can lower them; on CPU the packing stays in
     fields_to_array / array_to_fields, whose element ordering these ops must
     match exactly. */
  if (cw) {
    p.add(OpKind::unpack_state);
    p.add(OpKind::pack_state);
  }
  /* The decimation predicate is a device_predicate: the node stays in the
     graph and the kernel returns early when the step is not due. */
  if (has_dfts && !cw) p.add_dfts();
  p.set_magnetic_half_step(magnetic_evaluate_b, magnetic_update_b, magnetic_apply_b,
                           magnetic_transfer_b, magnetic_evaluate_h, magnetic_update_h,
                           magnetic_transfer_h);
  p.add_magnetic_marker(OpKind::synchronize_magnetic_fields, AccessMode::read_write);
  p.add_finite_value_check();

  return p.finish();
}

void format_step_plan(const StepPlan &p, std::vector<std::string> &out) {
  out.clear();
  char buf[128];
  for (const Operation &op : p.operations) {
    if (op.ft == field_type(NUM_FIELD_TYPES))
      snprintf(buf, sizeof buf, "%s", op_kind_name(op.kind));
    else
      snprintf(buf, sizeof buf, "%s(%s)", op_kind_name(op.kind), ft_name(op.ft));
    out.push_back(buf);
  }
}

} // namespace meep
