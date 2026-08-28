/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <vector>

#include <meep.hpp>

#include "backend/descriptors.hpp"
#include "backend/step_plan.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

using namespace meep;

static int failures = 0;

#define CHECK(cond, ...)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      printf("[rank %d] FAIL (%s:%d): ", my_rank(), __FILE__, __LINE__);                          \
      printf(__VA_ARGS__);                                                                         \
      printf("\n");                                                                               \
      fflush(stdout);                                                                              \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

static double eps_slab(const vec &p) { return fabs(p.y()) < 0.4 ? 12.0 : 1.0; }
static double unit_conductivity(const vec &) { return 0.25; }

class one_cross_material : public material_function {
public:
  virtual void eff_chi1inv_row(component c, double row[3], const volume &v,
                               double tol = DEFAULT_SUBPIXEL_TOL,
                               int maxeval = DEFAULT_SUBPIXEL_MAXEVAL) {
    (void)v;
    (void)tol;
    (void)maxeval;
    row[0] = row[1] = row[2] = 0.0;
    const direction d = component_direction(c);
    row[int(d)] = 1.0;
    /* For the Ez row, cycle(Z,1)=X and cycle(Z,2)=Y. Populate only the
       second cross term so normalization must move the Y pair into slot 1. */
    if (d == Z) row[int(Y)] = 0.125;
    if (d == Y) row[int(Z)] = 0.125;
  }
};

static bool has_access(const Operation &op, ArrayId id) {
  if (!is_valid(id)) return true;
  for (size_t i = 0; i < op.accesses.size(); ++i)
    if (op.accesses[i].array.id == id) return true;
  return false;
}

static bool has_access(const Operation &op, ArrayId id, AccessMode mode) {
  if (!is_valid(id)) return true;
  for (size_t i = 0; i < op.accesses.size(); ++i)
    if (op.accesses[i].array.id == id && op.accesses[i].mode == mode) return true;
  return false;
}

static void check_finite_value_accesses(const fields &f, const StepPlan &plan) {
  const Operation *finite = NULL;
  for (size_t i = 0; i < plan.operations.size(); ++i)
    if (plan.operations[i].kind == OpKind::finite_value_check) {
      CHECK(!finite, "plan contains more than one finite-value check");
      finite = &plan.operations[i];
    }
  CHECK(finite, "plan has no finite-value check");
  if (!finite) return;
  CHECK(f.storage_plan && f.storage_plan->arrays.size() == f.storage_plan->keys.size(),
        "storage arrays and keys are not available in parallel");
  if (!f.storage_plan || f.storage_plan->arrays.size() != f.storage_plan->keys.size()) return;

  std::vector<ArrayId> expected;
  for (size_t i = 0; i < f.storage_plan->arrays.size(); ++i) {
    const ArraySpec &spec = f.storage_plan->arrays[i];
    const StorageKey &key = f.storage_plan->keys[i];
    if (key.kind == int(array_kind::f) && spec.element_type == ElementType::realnum_value &&
        spec.elements && !is_valid(spec.alias_of))
      expected.push_back(spec.id);
  }

  CHECK(finite->accesses.size() == expected.size(), "finite check has %zu accesses, expected %zu",
        finite->accesses.size(), expected.size());
  const size_t compared = std::min(finite->accesses.size(), expected.size());
  for (size_t i = 0; i < compared; ++i) {
    const BufferAccess &access = finite->accesses[i];
    CHECK(is_valid(access.array.id) && access.array.id.value < f.storage_plan->arrays.size(),
          "finite-check access %zu has an invalid ArrayId", i);
    if (!is_valid(access.array.id) || access.array.id.value >= f.storage_plan->arrays.size())
      continue;
    const ArraySpec &spec = f.storage_plan->arrays[access.array.id.value];
    const StorageKey &key = f.storage_plan->keys[access.array.id.value];
    CHECK(access.array.id == expected[i], "finite-check access %zu is out of stable ArrayId order",
          i);
    if (i)
      CHECK(finite->accesses[i - 1].array.id.value < access.array.id.value,
            "finite-check accesses are not in ascending ArrayId order");
    CHECK(access.mode == AccessMode::read, "finite-check access %zu is not read-only", i);
    CHECK(access.array.offset == 0 && access.array.elements == spec.elements,
          "finite-check access %zu does not cover its full allocation", i);
    CHECK(!is_valid(spec.alias_of), "finite-check access %zu names an alias", i);
    CHECK(key.kind == int(array_kind::f) && spec.element_type == ElementType::realnum_value,
          "finite-check access %zu names a non-physical-field allocation", i);
    CHECK(key.chunk >= 0 && key.component_ >= 0 && key.component_ < NUM_FIELD_COMPONENTS &&
              (key.cmp == 0 || key.cmp == 1),
          "finite-check access %zu lacks deterministic chunk/component/cmp attribution", i);
  }
  CHECK(or_to_all(!finite->accesses.empty()),
        "prepared finite check has no field accesses on any rank");

  if (!finite->accesses.empty()) {
    StepPlan changed = plan;
    for (size_t i = 0; i < changed.operations.size(); ++i)
      if (changed.operations[i].kind == OpKind::finite_value_check) {
        --changed.operations[i].accesses[0].array.elements;
        break;
      }
    CHECK(compute_step_plan_signature(changed) != plan.signature,
          "signature ignored a finite-check access-span change");
  }
}

static void check_region(const fields &f, const UpdateRegion &region) {
  const grid_volume &gv = f.chunks[region.chunk]->gv;
  std::vector<ptrdiff_t> reference;
  LOOP_OVER_IVECS(gv, region.begin, region.end, index) { reference.push_back(index); }

  std::vector<ptrdiff_t> flattened;
  for (size_t i0 = 0; i0 < region.counts[0]; ++i0)
    for (size_t i1 = 0; i1 < region.counts[1]; ++i1)
      for (size_t i2 = 0; i2 < region.counts[2]; ++i2)
        flattened.push_back(ptrdiff_t(region.base) + ptrdiff_t(i0) * region.strides[0] +
                            ptrdiff_t(i1) * region.strides[1] +
                            ptrdiff_t(i2) * region.strides[2]);

  CHECK(reference == flattened, "chunk %d component %d flattened region differs", region.chunk,
        int(region.c));
}

static void check_prepared_updates() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  lorentzian_susceptibility susceptibility(1.1, 0.05);
  gyrotropic_susceptibility gyro(vec(0.17, -0.23, 0.31), 0.8, 0.03, 0.07,
                                 GYROTROPIC_SATURATED);
  s.add_susceptibility(eps_slab, E_stuff, susceptibility);
  s.add_susceptibility(eps_slab, E_stuff, gyro);
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  src.is_integrated = false;
  f.add_point_source(Ez, src, vec(0.13, 0.11));
  f.add_dft(Ez, volume(vec(-0.7, -0.6), vec(0.7, 0.6)), 0.23, 0.37, 3,
            /*include_dV_and_interp_weights=*/true);
  f.advance(2);

  StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  size_t update_ops = 0;
  size_t source_evaluations = 0, source_applications = 0;
  size_t dft_operations = 0;
  size_t polarization_operations = 0, polarization_rows = 0, gyrotropic_rows = 0,
         subtraction_rows = 0;
  for (size_t oi = 0; oi < plan.operations.size(); ++oi) {
    const Operation &op = plan.operations[oi];
    if (op.kind == OpKind::evaluate_source_scalars) {
      ++source_evaluations;
      CHECK(op.descriptor_index == 0 &&
                op.descriptor_count == f.descriptors->sources.source_times.size(),
            "source evaluation has the wrong source-time span");
    }
    if (op.kind == OpKind::apply_sources && op.source_descriptor_count) {
      ++source_applications;
      CHECK(op.ft == D_stuff, "the electric source was attached to the wrong field type");
      CHECK(size_t(op.source_descriptor_index) + op.source_descriptor_count <=
                f.descriptors->sources.sources.size(),
            "source application span is out of range");
      for (size_t i = op.source_descriptor_index;
           i < size_t(op.source_descriptor_index) + op.source_descriptor_count; ++i) {
        const SourceDescriptor &d = f.descriptors->sources.sources[i];
        CHECK(d.ft == op.ft && !d.integrated,
              "ordinary source span contains the wrong source kind");
        CHECK(has_access(op, d.destination, AccessMode::read_write) &&
                  has_access(op, d.destination_imag, AccessMode::read_write) &&
                  has_access(op, d.condinv, AccessMode::read),
              "ordinary source access set is incomplete");
      }
    }
    if (op.kind == OpKind::update_dft) {
      ++dft_operations;
      CHECK(size_t(op.descriptor_index) + op.descriptor_count <= plan.dft_updates.size(),
            "update_dft descriptor span is out of range");
      CHECK(op.descriptor_count > 0 || f.num_chunks > 1,
            "owned DFT chunks produced an empty descriptor span");
      for (size_t i = op.descriptor_index;
           i < size_t(op.descriptor_index) + op.descriptor_count; ++i) {
        const DftDescriptor &d = plan.dft_updates[i];
        CHECK(d.omega.size() == d.Nomega, "DFT descriptor omega table is incomplete");
        CHECK(is_valid(d.accumulator) && is_valid(d.phase_scratch) &&
                  is_valid(d.source_field.id),
              "DFT descriptor lacks a required array");
        CHECK(has_access(op, d.accumulator, AccessMode::read_write),
              "DFT accumulator is not read-write");
        CHECK(has_access(op, d.phase_scratch, AccessMode::write),
              "DFT phase scratch is not write-only");
        CHECK(has_access(op, d.source_field.id, AccessMode::read),
              "DFT real source is not read-only");
        CHECK(has_access(op, d.source_field_imag.id, AccessMode::read),
              "DFT imaginary source is not read-only");
      }
    }
    if (op.kind == OpKind::update_polarization) {
      ++polarization_operations;
      CHECK(size_t(op.descriptor_index) + op.descriptor_count <=
                plan.polarization_updates.size(),
            "polarization update span is out of range");
      for (size_t i = op.descriptor_index;
           i < size_t(op.descriptor_index) + op.descriptor_count; ++i) {
        const PolarizationUpdate &d = plan.polarization_updates[i];
        ++polarization_rows;
        check_region(f, d.region);
        CHECK(is_valid(d.p) && is_valid(d.p_prev) && is_valid(d.primary_w) &&
                  is_valid(d.diagonal_sigma),
              "polarization update lacks a required operand");
        CHECK(has_access(op, d.p, AccessMode::read_write) &&
                  has_access(op, d.p_prev, AccessMode::read_write) &&
                  has_access(op, d.primary_w, AccessMode::read) &&
                  has_access(op, d.diagonal_sigma, AccessMode::read),
              "polarization update access set is incomplete");
        if (d.kind == PolarizationUpdateKind::gyrotropic) {
          ++gyrotropic_rows;
          CHECK(is_valid(d.p_cross1) && is_valid(d.p_prev_cross1) &&
                    is_valid(d.p_cross2) && is_valid(d.p_prev_cross2),
                "gyrotropic update lacks its cross polarization state");
          CHECK(has_access(op, d.p_cross1, AccessMode::read_write) &&
                    has_access(op, d.p_prev_cross1, AccessMode::read_write) &&
                    has_access(op, d.p_cross2, AccessMode::read_write) &&
                    has_access(op, d.p_prev_cross2, AccessMode::read_write),
                "gyrotropic update access set omits cross polarization state");
          CHECK(d.region.variant_key == 0,
                "gyrotropic update reused Lorentzian variant bits");
          CHECK(d.gyro_model == GYROTROPIC_SATURATED && d.alpha == realnum(0.07),
                "gyrotropic update lost its model parameters");
          if (is_magnetic(d.region.c))
            CHECK(d.primary_stride < 0 && d.cross_stride1 < 0 && d.cross_stride2 < 0,
                  "magnetic gyrotropic update lost signed strides");
        }
        else {
          CHECK(!is_valid(d.p_cross1) && !is_valid(d.p_prev_cross1) &&
                    !is_valid(d.p_cross2) && !is_valid(d.p_prev_cross2),
                "Lorentzian update contains gyrotropic state IDs");
        }
      }
    }
    if (op.kind != OpKind::update_db && op.kind != OpKind::update_eh) continue;
    ++update_ops;
    CHECK(or_to_all(op.descriptor_count > 0), "%s has an empty descriptor span on every rank",
          op_kind_name(op.kind));
    if (op.kind == OpKind::update_db) {
      CHECK(size_t(op.descriptor_index) + op.descriptor_count <= plan.db_updates.size(),
            "update_db descriptor span is out of range");
      for (size_t i = op.descriptor_index; i < size_t(op.descriptor_index) + op.descriptor_count;
           ++i) {
        const CurlUpdate &d = plan.db_updates[i];
        check_region(f, d.region);
        CHECK(is_valid(d.target), "curl descriptor has no target");
        CHECK((d.region.variant_key & ~(curl_has_second_derivative | curl_has_pml |
                                        curl_has_pml_aux | curl_has_conductivity |
                                        curl_has_bfast)) == 0,
              "curl descriptor has an unbounded variant bit");
        CHECK(has_access(op, d.target) && has_access(op, d.plus_source) &&
                  has_access(op, d.minus_source) && has_access(op, d.target_u) &&
                  has_access(op, d.conductivity) && has_access(op, d.condinv) &&
                  has_access(op, d.target_cond) && has_access(op, d.pml.sig) &&
                  has_access(op, d.pml.kap) && has_access(op, d.pml.siginv) &&
                  has_access(op, d.pml_u.sig) && has_access(op, d.pml_u.kap) &&
                  has_access(op, d.pml_u.siginv),
              "curl descriptor access set is incomplete");
      }
    }
    else {
      CHECK(size_t(op.polarization_subtraction_index) + op.polarization_subtraction_count <=
                plan.polarization_subtractions.size(),
            "polarization subtraction span is out of range");
      for (size_t i = op.polarization_subtraction_index;
           i < size_t(op.polarization_subtraction_index) + op.polarization_subtraction_count;
           ++i) {
        const PolarizationSubtraction &d = plan.polarization_subtractions[i];
        ++subtraction_rows;
        CHECK(is_valid(d.target) && is_valid(d.p) && d.elements > 0,
              "polarization subtraction lacks a required operand");
        CHECK(has_access(op, d.target, AccessMode::read_write) &&
                  has_access(op, d.p, AccessMode::read),
              "polarization subtraction access set is incomplete");
      }
      CHECK(size_t(op.descriptor_index) + op.descriptor_count <= plan.eh_updates.size(),
            "update_eh descriptor span is out of range");
      for (size_t i = op.descriptor_index; i < size_t(op.descriptor_index) + op.descriptor_count;
           ++i) {
        const ConstitutiveUpdate &d = plan.eh_updates[i];
        check_region(f, d.region);
        CHECK(is_valid(d.target) && is_valid(d.primary),
              "constitutive descriptor lacks target or primary field");
        CHECK((d.region.variant_key & ~(constitutive_one_offdiagonal |
                                        constitutive_two_offdiagonals | constitutive_has_pml |
                                        constitutive_has_nonlinearity | constitutive_has_minus_p |
                                        constitutive_copy_w_previous)) == 0,
              "constitutive descriptor has an unbounded variant bit");
        CHECK(has_access(op, d.target) && has_access(op, d.base_primary) &&
                  has_access(op, d.base_cross1) && has_access(op, d.base_cross2) &&
                  has_access(op, d.primary) && has_access(op, d.cross1) &&
                  has_access(op, d.cross2) && has_access(op, d.diagonal) &&
                  has_access(op, d.offdiagonal1) && has_access(op, d.offdiagonal2) &&
                  has_access(op, d.chi2) && has_access(op, d.chi3) &&
                  has_access(op, d.target_w) && has_access(op, d.previous_w) &&
                  has_access(op, d.pml.sig) && has_access(op, d.pml.kap) &&
                  has_access(op, d.pml.siginv),
              "constitutive descriptor access set is incomplete");
      }
    }
  }
  CHECK(update_ops == 4, "expected four Maxwell update operations, got %zu", update_ops);
  CHECK(polarization_operations == 2, "expected two polarization operations, got %zu",
        polarization_operations);
  CHECK(or_to_all(polarization_rows > 0), "prepared plan contains no polarization updates");
  CHECK(or_to_all(gyrotropic_rows > 0), "prepared plan contains no gyrotropic updates");
  CHECK(or_to_all(subtraction_rows > 0), "prepared plan contains no P subtractions");
  const size_t global_dft_operations = sum_to_all(dft_operations);
  CHECK(max_to_all(int(dft_operations)) <= 1 && global_dft_operations >= 1,
        "expected at most one local and at least one global DFT update operation, got %zu local "
        "and %zu global",
        dft_operations, global_dft_operations);
  check_finite_value_accesses(f, plan);
  CHECK(source_evaluations == 4, "expected four source evaluations, got %zu",
        source_evaluations);
  const size_t global_source_applications = sum_to_all(source_applications);
  CHECK(global_source_applications == 1,
        "expected one nonempty source application globally, got %zu",
        global_source_applications);

  CHECK(compute_step_plan_signature(plan) == plan.signature,
        "stored signature differs from structural signature");
  if (!plan.db_updates.empty()) {
    const uint64_t original = plan.signature;
    ++plan.db_updates[0].plus_stride;
    CHECK(compute_step_plan_signature(plan) != original,
          "signature ignored a structural curl descriptor change");
  }
  if (!plan.dft_updates.empty()) {
    StepPlan changed = plan;
    changed.dft_updates[0].omega[0] += 1e-6;
    CHECK(compute_step_plan_signature(changed) != plan.signature,
          "signature ignored a DFT frequency change");
    changed = plan;
    changed.dft_updates[0].scale += std::complex<double>(0.0, 1e-6);
    CHECK(compute_step_plan_signature(changed) != plan.signature,
          "signature ignored a DFT complex-scale change");
  }
  if (!plan.polarization_updates.empty()) {
    StepPlan changed = plan;
    changed.polarization_updates[0].gamma += 1e-6;
    CHECK(compute_step_plan_signature(changed) != plan.signature,
          "signature ignored a polarization coefficient change");
    changed = plan;
    ++changed.polarization_updates[0].p.value;
    CHECK(compute_step_plan_signature(changed) != plan.signature,
          "signature ignored a polarization ArrayId change");
  }
  if (!plan.polarization_subtractions.empty()) {
    StepPlan changed = plan;
    ++changed.polarization_subtractions[0].elements;
    CHECK(compute_step_plan_signature(changed) != plan.signature,
          "signature ignored a polarization subtraction extent change");
  }
}

static void check_integrated_source_inputs() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, no_pml());
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  src.is_integrated = true;
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(2);

  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  size_t prepared = 0;
  size_t integrated_spans = 0;
  for (size_t oi = 0; oi < plan.operations.size(); ++oi) {
    const Operation &op = plan.operations[oi];
    if (op.kind != OpKind::update_eh || op.ft != E_stuff) continue;
    if (op.source_descriptor_count) ++integrated_spans;
    for (size_t si = op.source_descriptor_index;
         si < size_t(op.source_descriptor_index) + op.source_descriptor_count; ++si) {
      const SourceDescriptor &source = f.descriptors->sources.sources[si];
      CHECK(source.ft == D_stuff && source.integrated,
            "E update contains the wrong integrated source descriptor");
      CHECK(has_access(op, source.destination, AccessMode::read) &&
                has_access(op, source.integrated_destination, AccessMode::read_write),
            "integrated source access set is incomplete");
    }
    for (size_t i = op.descriptor_index; i < size_t(op.descriptor_index) + op.descriptor_count;
         ++i) {
      const ConstitutiveUpdate &d = plan.eh_updates[i];
      if (!(d.region.variant_key & constitutive_has_minus_p)) continue;
      ++prepared;
      CHECK(is_valid(d.base_primary), "f_minus_p descriptor lost its original D/B input");
      if (d.primary != d.base_primary) {
        CHECK(has_access(op, d.base_primary, AccessMode::read),
              "f_minus_p preparation does not read the original D/B field");
        CHECK(has_access(op, d.primary, AccessMode::read_write),
              "f_minus_p preparation does not publish its write access");
      }
    }
  }
  CHECK(sum_to_all(integrated_spans) > 0,
        "integrated electric source is absent from every E-update source span");
  CHECK(or_to_all(prepared > 0), "integrated source produced no prepared f_minus_p inputs");
}

static void check_one_cross_normalization() {
  grid_volume gv = vol3d(2.0, 2.0, 2.0, 6.0);
  one_cross_material material;
  structure s(gv, material, no_pml());
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  src.is_integrated = true;
  f.add_point_source(Ez, src, vec(0.11, 0.13, 0.17));
  f.advance(1);

  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  size_t normalized = 0;
  for (size_t oi = 0; oi < plan.operations.size(); ++oi) {
    const Operation &op = plan.operations[oi];
    if (op.kind != OpKind::update_eh || op.ft != E_stuff) continue;
    for (size_t i = op.descriptor_index; i < size_t(op.descriptor_index) + op.descriptor_count;
         ++i) {
      const ConstitutiveUpdate &d = plan.eh_updates[i];
      if (d.region.c != Ez || !is_valid(d.offdiagonal1) || is_valid(d.offdiagonal2)) continue;
      const direction selected_direction =
          cycle_direction(f.chunks[d.region.chunk]->gv.dim, component_direction(d.region.c), 2);
      const component selected_component =
          direction_component(first_field_component(D_stuff), selected_direction);
      const ArrayId expected_base = f.array_catalog->find(
          {d.region.chunk, int(array_kind::f), int(selected_component), d.region.cmp, 0});
      const ArrayId expected_minus_p = f.array_catalog->find(
          {d.region.chunk, int(array_kind::f_minus_p), int(selected_component), d.region.cmp, 0});
      const ArrayId expected_selected =
          is_valid(expected_minus_p) ? expected_minus_p : expected_base;
      CHECK(d.base_cross1 == expected_base,
            "normalized cross slot lost its matching base D/B input");
      CHECK(d.cross1 == expected_selected,
            "normalized cross slot does not select its matching effective input");
      CHECK(has_access(op, d.base_cross1, AccessMode::read),
            "normalized cross base is not recorded read-only");
      CHECK(has_access(op, d.cross1, is_valid(expected_minus_p) ? AccessMode::read_write
                                                               : AccessMode::read),
            "normalized cross has the wrong access mode");
      ++normalized;
    }
  }
  CHECK(or_to_all(normalized > 0), "test did not exercise one-cross normalization");
}

static void check_alias_elision() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, no_pml());
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  src.is_integrated = false;
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(1);
  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);

  const Operation *finite = NULL;
  for (size_t i = 0; i < plan.operations.size(); ++i)
    if (plan.operations[i].kind == OpKind::finite_value_check) finite = &plan.operations[i];
  CHECK(finite, "plan has no finite-value check");

  size_t field_aliases = 0;
  for (size_t i = 0; i < f.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = f.array_catalog->spec(id);
    if (!is_valid(spec.alias_of)) continue;
    for (size_t j = 0; j < plan.eh_updates.size(); ++j)
      CHECK(plan.eh_updates[j].target != spec.id,
            "H/B alias was emitted as a constitutive write descriptor");
    const StorageKey &key = f.array_catalog->key(id);
    if (key.kind == int(array_kind::f) && spec.element_type == ElementType::realnum_value) {
      ++field_aliases;
      if (finite)
        CHECK(!has_access(*finite, spec.id), "field alias was emitted as a finite-check access");
    }
  }
  CHECK(field_aliases > 0, "alias-elision test prepared no physical-field aliases");
}

static void check_beta_plan(bool real_fields, double beta) {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5));
  s.set_conductivity(Bx, unit_conductivity);
  s.set_conductivity(By, unit_conductivity);
  s.set_conductivity(Dx, unit_conductivity);
  s.set_conductivity(Dy, unit_conductivity);
  fields f(&s, 0, beta);
  if (real_fields) f.use_real_fields();
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(1);

  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  CHECK(plan.beta == beta, "plan did not retain its prepared beta value");
  size_t rows = 0;
  bool saw_main_pml = false, saw_aux_pml = false, saw_conductivity = false;
  bool saw_b = false, saw_d = false, saw_x = false, saw_y = false;
  bool saw_cmp0 = false, saw_cmp1 = false;
  for (size_t oi = 0; oi < plan.operations.size(); ++oi) {
    const Operation &op = plan.operations[oi];
    if (op.kind != OpKind::update_db) {
      CHECK(op.beta_descriptor_count == 0, "non-update_db operation owns beta rows");
      continue;
    }
    CHECK(size_t(op.beta_descriptor_index) + op.beta_descriptor_count <=
              plan.beta_updates.size(),
          "beta descriptor span is out of range");
    for (size_t i = op.beta_descriptor_index;
         i < size_t(op.beta_descriptor_index) + op.beta_descriptor_count; ++i) {
      const BetaUpdate &d = plan.beta_updates[i];
      ++rows;
      check_region(f, d.region);
      CHECK(d.region.c == Bx || d.region.c == By || d.region.c == Dx || d.region.c == Dy,
            "beta row targets a non-transverse component");
      CHECK((d.region.variant_key &
             ~(beta_has_pml | beta_has_pml_aux | beta_has_conductivity)) == 0,
            "beta descriptor has an unbounded variant bit");
      CHECK(is_valid(d.target) && is_valid(d.source), "beta row lacks a required operand");
      CHECK(has_access(op, d.target, AccessMode::read_write) &&
                has_access(op, d.source, AccessMode::read) &&
                has_access(op, d.target_u, AccessMode::read_write) &&
                has_access(op, d.condinv, AccessMode::read) &&
                has_access(op, d.target_cond, AccessMode::read_write) &&
                has_access(op, d.pml.siginv, AccessMode::read) &&
                has_access(op, d.pml_u.siginv, AccessMode::read),
            "beta descriptor access set is incomplete");

      const fields_chunk &fc = *f.chunks[d.region.chunk];
      const direction dc = component_direction(d.region.c);
      const field_type ft = op.ft;
      saw_main_pml |= (d.region.variant_key & beta_has_pml) != 0;
      saw_aux_pml |= (d.region.variant_key & beta_has_pml_aux) != 0;
      saw_conductivity |= (d.region.variant_key & beta_has_conductivity) != 0;
      saw_b |= ft == B_stuff;
      saw_d |= ft == D_stuff;
      saw_x |= dc == X;
      saw_y |= dc == Y;
      saw_cmp0 |= d.region.cmp == 0;
      saw_cmp1 |= d.region.cmp == 1;
      const component source_component =
          direction_component(ft == D_stuff ? Hx : Ex, dc == X ? Y : X);
      const ArrayId opposite = f.array_catalog->find(
          {d.region.chunk, int(array_kind::f), int(source_component), 1 - d.region.cmp, 0});
      const ArrayId same = f.array_catalog->find(
          {d.region.chunk, int(array_kind::f), int(source_component), d.region.cmp, 0});
      CHECK(d.source == (is_valid(opposite) ? opposite : same),
            "beta source does not follow real/complex cmp routing");
      const realnum expected =
          2 * pi * fc.beta * fc.dt * (dc == X ? +1 : -1) *
          (is_valid(opposite) ? (ft == D_stuff ? -1 : +1) * (2 * d.region.cmp - 1) : 1);
      CHECK(d.betadt == double(expected), "beta coefficient differs from host realnum order");
    }
  }
  CHECK(or_to_all(rows > 0), "%s beta=%g produced no beta rows",
        real_fields ? "real" : "complex", beta);
  CHECK(or_to_all(saw_main_pml) && or_to_all(saw_aux_pml) && or_to_all(saw_conductivity),
        "beta plan did not cover primary PML, auxiliary PML, and conductivity");
  CHECK(or_to_all(saw_b) && or_to_all(saw_d) && or_to_all(saw_x) && or_to_all(saw_y),
        "beta plan did not cover B/D and X/Y routing");
  CHECK(or_to_all(saw_cmp0), "beta plan did not cover cmp 0");
  CHECK(real_fields ? !or_to_all(saw_cmp1) : or_to_all(saw_cmp1),
        "beta plan has the wrong real/complex cmp coverage");
}

static void check_zero_beta_plan() {
  grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure s(gv, eps_slab, pml(0.5));
  fields f(&s, 0, 0.0);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(1);
  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  CHECK(plan.beta_updates.empty(), "zero beta emitted beta descriptors");
  for (const Operation &op : plan.operations)
    CHECK(op.beta_descriptor_count == 0, "zero beta emitted a nonempty beta span");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;
  check_prepared_updates();
  check_integrated_source_inputs();
  check_one_cross_normalization();
  check_alias_elision();
  check_beta_plan(false, +0.17);
  check_beta_plan(false, -0.17);
  check_beta_plan(true, +0.17);
  check_beta_plan(true, -0.17);
  check_zero_beta_plan();
  if (failures) {
    master_printf("prepared_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("prepared_plan: all checks passed\n");
  return 0;
}
