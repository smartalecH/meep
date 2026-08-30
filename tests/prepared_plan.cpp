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
      printf("[rank %d] FAIL (%s:%d): ", my_rank(), __FILE__, __LINE__);                           \
      printf(__VA_ARGS__);                                                                         \
      printf("\n");                                                                                \
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

class prepared_custom_lorentzian : public lorentzian_susceptibility {
public:
  prepared_custom_lorentzian(realnum omega_0, realnum gamma)
      : lorentzian_susceptibility(omega_0, gamma) {}
  susceptibility *clone() const override { return new prepared_custom_lorentzian(*this); }
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

static bool covers_access(const Operation &op, ArrayId id, AccessMode mode) {
  if (!is_valid(id)) return true;
  for (size_t i = 0; i < op.accesses.size(); ++i)
    if (op.accesses[i].array.id == id &&
        (op.accesses[i].mode == mode || op.accesses[i].mode == AccessMode::read_write))
      return true;
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
                            ptrdiff_t(i1) * region.strides[1] + ptrdiff_t(i2) * region.strides[2]);

  CHECK(reference == flattened, "chunk %d component %d flattened region differs", region.chunk,
        int(region.c));
}

static void check_prepared_updates() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  lorentzian_susceptibility susceptibility(1.1, 0.05);
  prepared_custom_lorentzian custom(0.87, 0.06);
  gyrotropic_susceptibility gyro(vec(0.17, -0.23, 0.31), 0.8, 0.03, 0.07,
                                 GYROTROPIC_SATURATED);
  const realnum ml_gamma[] = {realnum(0.02), 0, 0, realnum(0.03)};
  const realnum ml_n0[] = {realnum(0.8), realnum(0.2)};
  const realnum ml_alpha[] = {realnum(-0.4), realnum(0.5)};
  const realnum ml_omega[] = {realnum(0.63)};
  const realnum ml_damping[] = {realnum(0.04)};
  const realnum ml_sigmat[] = {1, 1, 1, 1, 1};
  multilevel_susceptibility multilevel(2, 1, ml_gamma, ml_n0, ml_alpha, ml_omega,
                                      ml_damping, ml_sigmat);
  s.add_susceptibility(eps_slab, E_stuff, susceptibility);
  s.add_susceptibility(eps_slab, E_stuff, custom);
  s.add_susceptibility(eps_slab, E_stuff, gyro);
  s.add_susceptibility(eps_slab, E_stuff, multilevel);
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  src.is_integrated = false;
  f.add_point_source(Ez, src, vec(0.13, 0.11));
  f.add_dft(Ez, volume(vec(-0.7, -0.6), vec(0.7, 0.6)), 0.23, 0.37, 3,
            /*include_dV_and_interp_weights=*/true);
  f.advance(2);

  StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  std::string host_error;
  CHECK(validate_host_callback_plan(f, plan, &host_error),
        "prepared custom callback plan rejected: %s", host_error.c_str());
  size_t live_dft_rows = 0;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    if (f.chunks[chunk]->is_mine())
      for (dft_chunk *d = f.chunks[chunk]->dft_chunks; d; d = d->next_in_chunk)
        ++live_dft_rows;
  size_t update_ops = 0;
  size_t source_evaluations = 0, source_applications = 0;
  size_t dft_operations = 0;
  size_t polarization_operations = 0, polarization_rows = 0, gyrotropic_rows = 0,
         subtraction_rows = 0, multilevel_groups = 0, multilevel_rows = 0,
         host_markers = 0, host_callbacks = 0, host_halos = 0;
  for (size_t oi = 0; oi < plan.operations.size(); ++oi) {
    const Operation &op = plan.operations[oi];
    if (op.kind == OpKind::host_callback) {
      ++host_markers;
      CHECK(op.descriptor_count == 1 && op.descriptor_index < plan.host_segments.size(),
            "prepared host marker has an invalid segment span");
      if (op.descriptor_count == 1 && op.descriptor_index < plan.host_segments.size()) {
        const HostSegment &segment = plan.host_segments[op.descriptor_index];
        CHECK(uint64_t(segment.callback_index) + segment.callback_count <=
                  plan.host_callbacks.size(),
              "prepared host segment has an invalid callback span");
        host_callbacks += segment.callback_count;
        CHECK(uint64_t(segment.host_halo_plan_index) + segment.host_halo_plan_count <=
                  plan.host_halo_plans.size(),
              "prepared host segment has an invalid logical halo span");
        for (uint32_t hi = 0; hi < segment.host_halo_plan_count; ++hi) {
          const HostHaloPlanDescriptor &halo =
              plan.host_halo_plans[size_t(segment.host_halo_plan_index) + hi];
          const HaloPlan *resolved = NULL;
          CHECK(resolve_host_halo_plan(f, halo, resolved, &host_error),
                "prepared logical host halo failed resolution: %s", host_error.c_str());
          CHECK(resolved && resolved->storage == HaloStorageDisposition::host_owned,
                "prepared logical host halo did not resolve host-owned storage");
          ++host_halos;
        }
      }
      for (const BufferAccess &access : op.accesses) {
        CHECK(is_valid(access.array.id) && access.array.id.value < f.array_catalog->size(),
              "prepared host marker contains an invalid ArrayId");
        if (!is_valid(access.array.id) || access.array.id.value >= f.array_catalog->size())
          continue;
        const ArraySpec &spec = f.array_catalog->spec(access.array.id);
        CHECK(!is_valid(spec.alias_of), "prepared host marker retained an alias ArrayId");
        CHECK(access.array.offset == 0 && access.array.elements == spec.elements,
              "prepared host marker does not cover a full canonical allocation");
      }
    }
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
      const size_t group_end =
          size_t(op.polarization_group_index) + size_t(op.polarization_group_count);
      CHECK(group_end <= plan.polarization_groups.size(),
            "polarization group span is out of range");
      for (size_t gi = op.polarization_group_index;
           gi < group_end && gi < plan.polarization_groups.size(); ++gi) {
        const PolarizationUpdateGroup &group = plan.polarization_groups[gi];
        if (group.kind != PolarizationGroupKind::multilevel) continue;
        ++multilevel_groups;
        CHECK(group.population_count == 1 &&
                  size_t(group.population_index) < plan.multilevel_population_updates.size(),
              "prepared multilevel group lacks its population action");
        if (size_t(group.population_index) >= plan.multilevel_population_updates.size()) continue;
        const MultilevelPopulationUpdate &population =
            plan.multilevel_population_updates[group.population_index];
        CHECK(has_access(op, population.gamma_inv, AccessMode::read) &&
                  has_access(op, population.populations, AccessMode::read_write),
              "prepared multilevel population access modes are incomplete");
        CHECK(size_t(population.term_index) + population.term_count <=
                  plan.multilevel_population_terms.size() &&
                  size_t(group.transition_index) + group.transition_count <=
                  plan.multilevel_transition_updates.size(),
              "prepared multilevel term/transition spans are out of range");
        for (size_t j = 0; j < population.term_count; ++j) {
          const MultilevelPopulationTerm &term =
              plan.multilevel_population_terms[population.term_index + j];
          const MultilevelTransitionUpdate &transition =
              plan.multilevel_transition_updates[group.transition_index + j];
          CHECK(has_access(op, term.w, AccessMode::read) &&
                    has_access(op, term.w_prev, AccessMode::read) &&
                    has_access(op, term.p, AccessMode::read_write) &&
                    has_access(op, term.p_prev, AccessMode::read_write) &&
                    has_access(op, transition.diagonal_sigma, AccessMode::read) &&
                    has_access(op, transition.populations, AccessMode::read_write),
                "prepared multilevel transition access union is incomplete");
          ++multilevel_rows;
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
        CHECK(
            (d.region.variant_key & ~(curl_has_second_derivative | curl_has_pml | curl_has_pml_aux |
                                      curl_has_conductivity | curl_has_bfast)) == 0,
            "curl descriptor has an unbounded variant bit");
        CHECK(has_access(op, d.target) && has_access(op, d.plus_source) &&
                  has_access(op, d.minus_source) && has_access(op, d.target_u) &&
                  has_access(op, d.conductivity) && has_access(op, d.condinv) &&
                  has_access(op, d.target_cond) && has_access(op, d.pml.sig) &&
                  has_access(op, d.pml.kap) && has_access(op, d.pml.siginv) &&
                  has_access(op, d.pml_u.sig) && has_access(op, d.pml_u.kap) &&
                  has_access(op, d.pml_u.siginv),
              "curl descriptor access set is incomplete");
        if (d.region.variant_key & curl_has_bfast) {
          CHECK(d.bfast_update_index < plan.bfast_updates.size(),
                "BFAST curl has no paired postpass");
        }
        else {
          CHECK(d.bfast_update_index == UINT32_MAX,
                "ordinary curl unexpectedly references a BFAST postpass");
        }
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
        CHECK((d.region.variant_key &
               ~(constitutive_one_offdiagonal | constitutive_two_offdiagonals |
                 constitutive_has_pml | constitutive_has_nonlinearity | constitutive_has_minus_p |
                 constitutive_copy_w_previous)) == 0,
              "constitutive descriptor has an unbounded variant bit");
        CHECK(has_access(op, d.target) && has_access(op, d.base_primary) &&
                  has_access(op, d.base_cross1) && has_access(op, d.base_cross2) &&
                  has_access(op, d.primary) && has_access(op, d.cross1) &&
                  has_access(op, d.cross2) && has_access(op, d.diagonal) &&
                  has_access(op, d.offdiagonal1) && has_access(op, d.offdiagonal2) &&
                  has_access(op, d.chi2) && has_access(op, d.chi3) && has_access(op, d.target_w) &&
                  has_access(op, d.previous_w) && has_access(op, d.pml.sig) &&
                  has_access(op, d.pml.kap) && has_access(op, d.pml.siginv),
              "constitutive descriptor access set is incomplete");
      }
    }
  }
  CHECK(update_ops == 4, "expected four Maxwell update operations, got %zu", update_ops);
  CHECK(plan.dft_updates.size() == live_dft_rows,
        "prepared plan has %zu DFT rows for %zu owned live rows", plan.dft_updates.size(),
        live_dft_rows);
  CHECK(dft_operations == (live_dft_rows ? 1 : 0),
        "local DFT operation count %zu does not match %zu owned live rows", dft_operations,
        live_dft_rows);
  CHECK(or_to_all(live_dft_rows > 0), "fixture contains no owned DFT row on any rank");
  CHECK(polarization_operations == 2, "expected two polarization operations, got %zu",
        polarization_operations);
  CHECK(or_to_all(polarization_rows > 0), "prepared plan contains no polarization updates");
  CHECK(or_to_all(gyrotropic_rows > 0), "prepared plan contains no gyrotropic updates");
  CHECK(or_to_all(multilevel_groups > 0 && multilevel_rows > 0),
        "prepared plan contains no multilevel groups or transition rows");
  CHECK(or_to_all(subtraction_rows > 0), "prepared plan contains no P subtractions");
  CHECK(or_to_all(host_markers > 0 && host_callbacks > 0),
        "prepared plan contains no custom host callback delta");
  CHECK(or_to_all(host_halos > 0),
        "prepared custom polarization segment contains no logical host halos");
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

static void check_bfast_plan() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5));
  s.set_conductivity(Bx, unit_conductivity);
  s.set_conductivity(By, unit_conductivity);
  s.set_conductivity(Bz, unit_conductivity);
  s.set_conductivity(Dx, unit_conductivity);
  s.set_conductivity(Dy, unit_conductivity);
  s.set_conductivity(Dz, unit_conductivity);
  const std::vector<double> scaled_k{0.17, -0.11, 0.07};
  fields f(&s, 0, 0, true, 0, 0, scaled_k);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(1);

  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  size_t rows = 0;
  bool owns_chunk = false;
  for (int i = 0; i < f.num_chunks; ++i)
    owns_chunk = owns_chunk || f.chunks[i]->is_mine();
  bool saw_one_source = false, saw_two_sources = false;
  bool saw_main_pml = false, saw_aux_pml = false, saw_conductivity = false;
  for (const Operation &op : plan.operations) {
    if (op.kind != OpKind::update_db) continue;
    for (size_t i = op.descriptor_index; i < size_t(op.descriptor_index) + op.descriptor_count;
         ++i) {
      const CurlUpdate &curl = plan.db_updates[i];
      CHECK((curl.region.variant_key & curl_has_bfast) != 0,
            "nonzero scaled-k curl lacks the BFAST bit");
      CHECK(curl.bfast_update_index < plan.bfast_updates.size(),
            "nonzero scaled-k curl lacks a paired BFAST row");
      if (curl.bfast_update_index >= plan.bfast_updates.size()) continue;
      const BfastUpdate &d = plan.bfast_updates[curl.bfast_update_index];
      ++rows;
      check_region(f, d.region);
      CHECK(d.region.chunk == curl.region.chunk && d.region.c == curl.region.c &&
                d.region.cmp == curl.region.cmp && d.region.begin == curl.region.begin &&
                d.region.end == curl.region.end,
            "BFAST postpass is not paired to its curl region");
      CHECK(d.target == curl.target && d.source1 == curl.plus_source &&
                d.source2 == curl.minus_source && d.stride1 == curl.plus_stride &&
                d.stride2 == curl.minus_stride,
            "BFAST postpass does not preserve curl sources and strides");
      CHECK(is_valid(d.f_bfast), "BFAST postpass lacks persistent state");
      CHECK(has_access(op, d.target, AccessMode::read_write) &&
                has_access(op, d.source1, AccessMode::read) &&
                has_access(op, d.source2, AccessMode::read) &&
                has_access(op, d.f_bfast, AccessMode::read_write) &&
                has_access(op, d.target_u, AccessMode::read_write) &&
                has_access(op, d.condinv, AccessMode::read) &&
                has_access(op, d.target_cond, AccessMode::read_write) &&
                has_access(op, d.pml.siginv, AccessMode::read) &&
                has_access(op, d.pml_u.siginv, AccessMode::read),
            "BFAST descriptor access set is incomplete");
      CHECK((d.region.variant_key &
             ~(bfast_has_pml | bfast_has_pml_aux | bfast_has_conductivity)) == 0,
            "BFAST descriptor has an unbounded variant bit");

      const realnum expected_k1 =
          is_valid(d.source2)
              ? scaled_k[component_index(component(f.array_catalog->key(d.source2).component_))]
              : 0;
      const realnum expected_k2 =
          is_valid(d.source1)
              ? scaled_k[component_index(component(f.array_catalog->key(d.source1).component_))]
              : 0;
      const double sign = op.ft == D_stuff ? -1.0 : 1.0;
      CHECK(d.k1 == sign * double(expected_k1) && d.k2 == sign * double(expected_k2),
            "BFAST k coefficients differ from host realnum routing");
      saw_one_source |= is_valid(d.source1) != is_valid(d.source2);
      saw_two_sources |= is_valid(d.source1) && is_valid(d.source2);
      saw_main_pml |= (d.region.variant_key & bfast_has_pml) != 0;
      saw_aux_pml |= (d.region.variant_key & bfast_has_pml_aux) != 0;
      saw_conductivity |= (d.region.variant_key & bfast_has_conductivity) != 0;
    }
  }
  CHECK(and_to_all(!owns_chunk || rows > 0),
        "an owning rank produced no BFAST rows for nonzero scaled k");
  CHECK(or_to_all(saw_one_source) && or_to_all(saw_two_sources),
        "BFAST plan did not cover one- and two-source rows");
  CHECK(or_to_all(saw_main_pml) && or_to_all(saw_aux_pml) && or_to_all(saw_conductivity),
        "BFAST plan did not cover PML, auxiliary PML, and conductivity");

  fields zero(&s, 0, 0, true, 0, 0, std::vector<double>{0, 0, 0});
  zero.add_point_source(Ez, src, vec(0.11, 0.13));
  zero.advance(1);
  const StepPlan zero_plan = build_step_plan(zero, StepProgram::ordinary);
  CHECK(zero_plan.bfast_updates.empty(), "zero scaled k emitted BFAST descriptors");
  for (const CurlUpdate &curl : zero_plan.db_updates) {
    CHECK((curl.region.variant_key & curl_has_bfast) == 0,
          "zero scaled k retained the BFAST curl bit");
    CHECK(curl.bfast_update_index == UINT32_MAX, "zero scaled k retained a paired BFAST index");
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
      CHECK(has_access(op, d.cross1,
                       is_valid(expected_minus_p) ? AccessMode::read_write : AccessMode::read),
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
    CHECK(size_t(op.beta_descriptor_index) + op.beta_descriptor_count <= plan.beta_updates.size(),
          "beta descriptor span is out of range");
    for (size_t i = op.beta_descriptor_index;
         i < size_t(op.beta_descriptor_index) + op.beta_descriptor_count; ++i) {
      const BetaUpdate &d = plan.beta_updates[i];
      ++rows;
      check_region(f, d.region);
      CHECK(d.region.c == Bx || d.region.c == By || d.region.c == Dx || d.region.c == Dy,
            "beta row targets a non-transverse component");
      CHECK((d.region.variant_key & ~(beta_has_pml | beta_has_pml_aux | beta_has_conductivity)) ==
                0,
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
  CHECK(or_to_all(rows > 0), "%s beta=%g produced no beta rows", real_fields ? "real" : "complex",
        beta);
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

static void require_cylindrical_components(fields &f) {
  f.require_component(Er);
  f.require_component(Ep);
  f.require_component(Ez);
  f.require_component(Hr);
  f.require_component(Hp);
  f.require_component(Hz);
}

struct ExpectedCylindricalOriginAction {
  CylindricalOriginActionKind kind;
  ArrayId array;
  int radial_row;
};

static ArrayId cylindrical_array(const fields &f, int chunk, array_kind kind, component c,
                                 int cmp) {
  return f.array_catalog->find({chunk, int(kind), int(c), cmp, 0});
}

static component cylindrical_tail_source(field_type ft, direction dc) {
  if (ft == D_stuff) return dc == R ? Hz : Hr;
  return dc == R ? Ez : Er;
}

static void check_cylindrical_plan(double m, bool zero_near_origin, bool use_bfast,
                                   bool force_complex = false, bool annular = false,
                                   bool pure_conductivity = false) {
  grid_volume gv = volcyl(3.0, 4.0, 8.0);
  if (annular) gv.shift_origin(R, 12);
  structure s(gv, eps_slab, pure_conductivity ? no_pml() : pml(0.5));
  const component components[] = {Er, Ep, Ez, Hr, Hp, Hz};
  for (component c : components)
    s.set_conductivity(c, unit_conductivity);
  const std::vector<double> scaled_k =
      use_bfast ? std::vector<double>{0.17, -0.11, 0.07} : std::vector<double>{0, 0, 0};
  fields f(&s, m, 0, zero_near_origin, 64, 64, scaled_k);
  if (m == 0 && !force_complex) f.use_real_fields();
  require_cylindrical_components(f);
  f.advance(1);

  const StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  CHECK(plan.cylindrical_m == m, "plan did not retain cylindrical m=%g", m);
  CHECK(plan.cylindrical_origin_r.size() == size_t(f.num_chunks) &&
            plan.cylindrical_zero_near_origin.size() == size_t(f.num_chunks),
        "plan did not retain one cylindrical fingerprint per chunk");
  for (int i = 0; i < f.num_chunks; ++i) {
    CHECK(plan.cylindrical_origin_r[i] == f.chunks[i]->gv.origin_r(),
          "plan cylindrical origin fingerprint differs on chunk %d", i);
    CHECK(bool(plan.cylindrical_zero_near_origin[i]) == f.chunks[i]->zero_fields_near_cylorigin,
          "plan cylindrical origin policy differs on chunk %d", i);
  }

  size_t z_curls = 0, prefixes = 0, r_suppressed = 0, m_rows = 0;
  size_t axis_actions = 0, zero_actions = 0, axis_replays = 0;
  bool saw_prefix_before_bfast = false, saw_axis_then_zero = false;
  bool saw_main_pml = false, saw_aux_pml = false, saw_conductivity = false;
  bool owns_chunk = false, owns_origin_chunk = false;
  for (int i = 0; i < f.num_chunks; ++i) {
    owns_chunk = owns_chunk || f.chunks[i]->is_mine();
    owns_origin_chunk =
        owns_origin_chunk || (f.chunks[i]->is_mine() && f.chunks[i]->gv.origin_r() == 0.0);
  }
  for (const Operation &op : plan.operations) {
    if (op.kind == OpKind::update_db) {
      CHECK(size_t(op.cylindrical_m_descriptor_index) + op.cylindrical_m_descriptor_count <=
                plan.cylindrical_m_updates.size(),
            "cylindrical m/r span is out of range");
      CHECK(size_t(op.cylindrical_origin_action_index) + op.cylindrical_origin_action_count <=
                plan.cylindrical_origin_actions.size(),
            "cylindrical origin-action span is out of range");
      size_t expected_curls = 0, expected_prefixes = 0;
      std::vector<std::pair<ArrayId, ArrayId> > expected_m_rows;
      std::vector<ExpectedCylindricalOriginAction> expected_origin_actions;
      auto append_zero = [&](int chunk, component c, int cmp, int row) {
        const array_kind kinds[] = {array_kind::f, array_kind::f_cond, array_kind::f_u};
        for (array_kind kind : kinds) {
          const ArrayId id = cylindrical_array(f, chunk, kind, c, cmp);
          if (is_valid(id))
            expected_origin_actions.push_back({CylindricalOriginActionKind::zero_slab, id, row});
        }
      };
      for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
        const fields_chunk &fc = *f.chunks[chunk];
        if (!fc.is_mine()) continue;
        const int cmps = fc.is_real ? 1 : 2;
        for (size_t tile = 0; tile < fc.gvs_tiled.size(); ++tile)
          for (int cmp = 0; cmp < cmps; ++cmp)
            FOR_FT_COMPONENTS(op.ft, cc) {
              if (!is_valid(cylindrical_array(f, chunk, array_kind::f, cc, cmp))) continue;
              ++expected_curls;
              if (component_direction(cc) == Z) ++expected_prefixes;
            }
        if (m != 0)
          for (int cmp = 0; cmp < cmps; ++cmp)
            FOR_FT_COMPONENTS(op.ft, cc) {
              const direction dc = component_direction(cc);
              if (dc != R && dc != Z) continue;
              const ArrayId target = cylindrical_array(f, chunk, array_kind::f, cc, cmp);
              const ArrayId source = cylindrical_array(f, chunk, array_kind::f,
                                                       cylindrical_tail_source(op.ft, dc), 1 - cmp);
              if (is_valid(target) && is_valid(source)) expected_m_rows.push_back({target, source});
            }
        if (fc.gv.origin_r() != 0.0) continue;
        for (int cmp = 0; cmp < cmps; ++cmp) {
          if (m == 0 && op.ft == D_stuff) {
            const ArrayId target = cylindrical_array(f, chunk, array_kind::f, Dz, cmp);
            const ArrayId source = cylindrical_array(f, chunk, array_kind::f, Hp, cmp);
            if (is_valid(target) && is_valid(source)) {
              expected_origin_actions.push_back(
                  {CylindricalOriginActionKind::axis_update, target, 0});
              append_zero(chunk, Dp, cmp, 0);
            }
          }
          else if (m == 0 && op.ft == B_stuff) {
            if (is_valid(cylindrical_array(f, chunk, array_kind::f, Br, cmp)))
              append_zero(chunk, Br, cmp, 0);
          }
          else if (fabs(m) == 1) {
            const component target_component = op.ft == D_stuff ? Dp : Br;
            const component source1_component = op.ft == D_stuff ? Hr : Ep;
            const component source2_component = op.ft == D_stuff ? Hz : Ez;
            const int source2_cmp = op.ft == D_stuff ? cmp : 1 - cmp;
            const ArrayId target =
                cylindrical_array(f, chunk, array_kind::f, target_component, cmp);
            const ArrayId source1 =
                cylindrical_array(f, chunk, array_kind::f, source1_component, cmp);
            const ArrayId source2 =
                cylindrical_array(f, chunk, array_kind::f, source2_component, source2_cmp);
            if (is_valid(target) && is_valid(source1) && is_valid(source2)) {
              expected_origin_actions.push_back(
                  {CylindricalOriginActionKind::axis_update, target, 0});
              if (op.ft == D_stuff) append_zero(chunk, Dz, cmp, 0);
            }
          }
          else if (m != 0) {
            int radial_rows = 1;
            if (zero_near_origin) {
              radial_rows = 0;
              const double rmax = fabs(m) - int(fc.gv.origin_r() * fc.gv.a + 0.5);
              while (radial_rows <= fc.gv.nr() && radial_rows < rmax)
                ++radial_rows;
            }
            const array_kind kinds[] = {array_kind::f, array_kind::f_cond, array_kind::f_u};
            for (int row = 0; row < radial_rows; ++row)
              for (array_kind kind : kinds)
                FOR_FT_COMPONENTS(op.ft, cc) {
                  const ArrayId id = cylindrical_array(f, chunk, kind, cc, cmp);
                  if (is_valid(id))
                    expected_origin_actions.push_back(
                        {CylindricalOriginActionKind::zero_slab, id, row});
                }
          }
        }
      }
      CHECK(op.descriptor_count == expected_curls,
            "cylindrical curl span has %u rows, expected %zu", op.descriptor_count, expected_curls);
      CHECK(op.cylindrical_m_descriptor_count == expected_m_rows.size(),
            "cylindrical m/r span has %u rows, expected %zu", op.cylindrical_m_descriptor_count,
            expected_m_rows.size());
      CHECK(op.cylindrical_origin_action_count == expected_origin_actions.size(),
            "cylindrical origin span has %u actions, expected %zu",
            op.cylindrical_origin_action_count, expected_origin_actions.size());
      size_t op_prefixes = 0;
      for (size_t i = op.descriptor_index; i < size_t(op.descriptor_index) + op.descriptor_count;
           ++i) {
        const CurlUpdate &curl = plan.db_updates[i];
        const direction dc = component_direction(curl.region.c);
        CHECK(has_access(op, curl.target) && has_access(op, curl.plus_source) &&
                  has_access(op, curl.minus_source) && has_access(op, curl.target_u) &&
                  has_access(op, curl.conductivity) && has_access(op, curl.condinv) &&
                  has_access(op, curl.target_cond) && has_access(op, curl.pml.sig) &&
                  has_access(op, curl.pml.kap) && has_access(op, curl.pml.siginv) &&
                  has_access(op, curl.pml_u.sig) && has_access(op, curl.pml_u.kap) &&
                  has_access(op, curl.pml_u.siginv),
              "cylindrical curl access set is incomplete");
        saw_main_pml |= (curl.region.variant_key & curl_has_pml) != 0;
        saw_aux_pml |= (curl.region.variant_key & curl_has_pml_aux) != 0;
        saw_conductivity |= (curl.region.variant_key & curl_has_conductivity) != 0;
        if (dc == Z) {
          ++z_curls;
          CHECK(curl.radial_prefix_index < plan.cylindrical_radial_prefixes.size(),
                "cylindrical Z curl lacks its paired radial prefix");
          if (curl.radial_prefix_index < plan.cylindrical_radial_prefixes.size()) {
            const CylindricalRadialPrefix &prefix =
                plan.cylindrical_radial_prefixes[curl.radial_prefix_index];
            ++prefixes;
            ++op_prefixes;
            const ArrayId expected_scratch = f.array_catalog->find(
                {curl.region.chunk, int(array_kind::f_rderiv_int), -1, -1, 0});
            CHECK(prefix.scratch == expected_scratch && curl.plus_source == expected_scratch,
                  "cylindrical Z curl does not consume its chunk radial scratch");
            CHECK(!is_valid(curl.minus_source),
                  "cylindrical Z curl did not suppress its minus pointer");
            CHECK(prefix.nr == size_t(f.chunks[curl.region.chunk]->gv.nr()) &&
                      prefix.nz == size_t(f.chunks[curl.region.chunk]->gv.nz()) &&
                      prefix.row_stride == prefix.nz + 1,
                  "cylindrical radial-prefix shape is wrong");
            const fields_chunk &fc = *f.chunks[curl.region.chunk];
            const realnum expected_ir0 =
                fc.gv.origin_r() * fc.gv.a +
                0.5 * fc.gv.iyee_shift(prefix.source_component).in_direction(R);
            CHECK(prefix.ir0 == double(expected_ir0),
                  "cylindrical radial-prefix ir0 differs from host realnum order");
            CHECK(prefix.source_elements == f.array_catalog->spec(prefix.source).elements &&
                      prefix.scratch_elements == f.array_catalog->spec(prefix.scratch).elements,
                  "cylindrical radial-prefix extent does not match storage");
            const StorageKey &source_key = f.array_catalog->key(prefix.source);
            CHECK(source_key.component_ == int(prefix.source_component) &&
                      source_key.cmp == prefix.cmp,
                  "cylindrical radial-prefix source identity is wrong");
            CHECK(has_access(op, prefix.source, AccessMode::read) &&
                      has_access(op, prefix.scratch, AccessMode::read_write),
                  "cylindrical radial-prefix accesses are incomplete");
            if (use_bfast) {
              CHECK(curl.bfast_update_index < plan.bfast_updates.size(),
                    "cylindrical Z curl lacks paired BFAST work");
              if (curl.bfast_update_index < plan.bfast_updates.size()) {
                const BfastUpdate &bfast = plan.bfast_updates[curl.bfast_update_index];
                CHECK(bfast.source1 == curl.plus_source && bfast.source2 == curl.minus_source,
                      "cylindrical BFAST did not consume transformed curl sources");
                CHECK(has_access(op, bfast.target) && has_access(op, bfast.source1) &&
                          has_access(op, bfast.source2) && has_access(op, bfast.f_bfast) &&
                          has_access(op, bfast.target_u) && has_access(op, bfast.condinv) &&
                          has_access(op, bfast.target_cond) && has_access(op, bfast.pml.siginv) &&
                          has_access(op, bfast.pml_u.siginv),
                      "cylindrical BFAST access set is incomplete");
                saw_prefix_before_bfast = true;
              }
            }
          }
        }
        else {
          CHECK(curl.radial_prefix_index == UINT32_MAX,
                "non-Z cylindrical curl has a radial prefix");
          if (dc == R) {
            ++r_suppressed;
            CHECK(!is_valid(curl.plus_source),
                  "cylindrical R curl did not suppress its plus pointer");
          }
        }
        if (use_bfast) {
          CHECK(curl.bfast_update_index < plan.bfast_updates.size(),
                "cylindrical curl lacks its paired BFAST row");
          if (curl.bfast_update_index < plan.bfast_updates.size()) {
            const BfastUpdate &bfast = plan.bfast_updates[curl.bfast_update_index];
            component plus_component = NO_COMPONENT, minus_component = NO_COMPONENT;
            const component source_base = op.ft == D_stuff ? Hx : Ex;
            if (dc == R) {
              plus_component = direction_component(source_base, Z);
              minus_component = direction_component(source_base, P);
            }
            else if (dc == P) {
              plus_component = direction_component(source_base, R);
              minus_component = direction_component(source_base, Z);
            }
            else {
              plus_component = direction_component(source_base, P);
              minus_component = direction_component(source_base, R);
            }
            const fields_chunk &fc = *f.chunks[curl.region.chunk];
            const ArrayId expected_minus = cylindrical_array(f, curl.region.chunk, array_kind::f,
                                                             minus_component, curl.region.cmp);
            const ArrayId expected_plus =
                dc == Z ? f.array_catalog->find(
                              {curl.region.chunk, int(array_kind::f_rderiv_int), -1, -1, 0})
                : dc == R ? invalid_array()
                          : cylindrical_array(f, curl.region.chunk, array_kind::f, plus_component,
                                              curl.region.cmp);
            const ArrayId transformed_minus = dc == Z ? invalid_array() : expected_minus;
            CHECK(bfast.source1 == expected_plus && bfast.source2 == transformed_minus,
                  "cylindrical BFAST source transform is wrong for component %d",
                  int(curl.region.c));
            const realnum expected_k1 = fc.bfast_scaled_k[component_index(minus_component)];
            const realnum expected_k2 = fc.bfast_scaled_k[component_index(plus_component)];
            const double sign = op.ft == D_stuff ? -1.0 : 1.0;
            CHECK(bfast.k1 == sign * double(expected_k1) && bfast.k2 == sign * double(expected_k2),
                  "cylindrical BFAST k routing differs from original curl geometry");
            CHECK(bfast.stride1 == curl.plus_stride && bfast.stride2 == curl.minus_stride,
                  "cylindrical BFAST did not preserve transformed curl strides");
          }
        }
        else {
          CHECK(curl.bfast_update_index == UINT32_MAX &&
                    !(curl.region.variant_key & curl_has_bfast),
                "zero-k cylindrical curl retained BFAST work");
        }
      }
      CHECK(op_prefixes == expected_prefixes,
            "cylindrical radial-prefix span has %zu rows, expected %zu", op_prefixes,
            expected_prefixes);
      size_t expected_m_index = 0;
      for (size_t i = op.cylindrical_m_descriptor_index;
           i < size_t(op.cylindrical_m_descriptor_index) + op.cylindrical_m_descriptor_count; ++i) {
        const CylindricalMOverRUpdate &d = plan.cylindrical_m_updates[i];
        ++m_rows;
        CHECK(expected_m_index < expected_m_rows.size(),
              "cylindrical m/r row exceeds the exact expected sequence");
        if (expected_m_index < expected_m_rows.size())
          CHECK(d.target == expected_m_rows[expected_m_index].first &&
                    d.source == expected_m_rows[expected_m_index].second,
                "cylindrical m/r row %zu has the wrong target/source identity", expected_m_index);
        ++expected_m_index;
        CHECK(component_direction(d.region.c) == R || component_direction(d.region.c) == Z,
              "cylindrical m/r row targets phi");
        CHECK(is_valid(d.target) && is_valid(d.source), "cylindrical m/r row lacks an operand");
        CHECK(covers_access(op, d.target, AccessMode::read_write) &&
                  covers_access(op, d.source, AccessMode::read) &&
                  covers_access(op, d.target_u, AccessMode::read_write) &&
                  covers_access(op, d.condinv, AccessMode::read) &&
                  covers_access(op, d.target_cond, AccessMode::read_write) &&
                  covers_access(op, d.pml.siginv, AccessMode::read) &&
                  covers_access(op, d.pml_u.siginv, AccessMode::read),
              "cylindrical m/r accesses are incomplete");
        const fields_chunk &fc = *f.chunks[d.region.chunk];
        const direction dc = component_direction(d.region.c);
        const realnum expected_numerator = 2 * fc.m * (1 - 2 * d.region.cmp) *
                                           (1 - 2 * (op.ft == B_stuff)) * (1 - 2 * (dc == R)) *
                                           fc.Courant;
        CHECK(d.numerator == double(expected_numerator),
              "cylindrical m/r coefficient differs from host realnum order");
        CHECK(d.raw_radial_start == d.region.begin.in_direction(R),
              "cylindrical m/r row lost its raw radial coordinate");
        CHECK(is_valid(d.target_cond) ==
                  bool((d.region.variant_key & cylindrical_m_has_pml) &&
                       (d.region.variant_key & cylindrical_m_has_conductivity)),
              "cylindrical m/r conductivity target does not match the CPU PML branch");
        const StorageKey &source_key = f.array_catalog->key(d.source);
        CHECK(source_key.cmp == 1 - d.region.cmp,
              "cylindrical m/r source is not the opposite complex component");
      }
      CHECK(expected_m_index == expected_m_rows.size(),
            "cylindrical m/r sequence ended after %zu of %zu expected rows", expected_m_index,
            expected_m_rows.size());
      std::vector<int> previous_zero_order;
      for (size_t i = op.cylindrical_origin_action_index;
           i < size_t(op.cylindrical_origin_action_index) + op.cylindrical_origin_action_count;
           ++i) {
        const CylindricalOriginAction &action = plan.cylindrical_origin_actions[i];
        const size_t expected_index = i - op.cylindrical_origin_action_index;
        CHECK(expected_index < expected_origin_actions.size(),
              "cylindrical origin action exceeds the exact expected sequence");
        const ExpectedCylindricalOriginAction *expected =
            expected_index < expected_origin_actions.size()
                ? &expected_origin_actions[expected_index]
                : NULL;
        if (expected)
          CHECK(action.kind == expected->kind, "cylindrical origin action %zu has the wrong kind",
                expected_index);
        if (action.kind == CylindricalOriginActionKind::axis_update) {
          ++axis_actions;
          CHECK(action.index < plan.cylindrical_axis_updates.size(),
                "origin action references an invalid axis row");
          if (action.index < plan.cylindrical_axis_updates.size()) {
            const CylindricalAxisUpdate &d = plan.cylindrical_axis_updates[action.index];
            if (expected)
              CHECK(d.target == expected->array, "cylindrical axis action %zu has the wrong target",
                    expected_index);
            CHECK(d.region.begin.in_direction(R) == 0 && d.region.end.in_direction(R) == 0,
                  "cylindrical axis arithmetic is not restricted to r=0");
            CHECK(is_valid(d.target) && is_valid(d.source1),
                  "cylindrical axis arithmetic lacks a required operand");
            CHECK(has_access(op, d.target) && has_access(op, d.source1),
                  "cylindrical axis accesses are incomplete");
            CHECK(has_access(op, d.source2) && has_access(op, d.target_u) &&
                      has_access(op, d.conductivity) && has_access(op, d.condinv) &&
                      has_access(op, d.target_cond) && has_access(op, d.pml.sig) &&
                      has_access(op, d.pml.kap) && has_access(op, d.pml.siginv) &&
                      has_access(op, d.pml_u.sig) && has_access(op, d.pml_u.kap) &&
                      has_access(op, d.pml_u.siginv),
                  "cylindrical axis auxiliary/profile accesses are incomplete");
            CHECK(is_valid(d.target_cond) ==
                      bool(d.region.variant_key & cylindrical_axis_has_conductivity),
                  "cylindrical axis conductivity bit does not follow f_cond");
            CHECK(is_valid(d.target_cond) || (!is_valid(d.conductivity) && !is_valid(d.condinv)),
                  "nonconductive cylindrical axis row retained conductivity operands");
            if (d.kind == CylindricalAxisKind::m0_dz) {
              const fields_chunk &fc = *f.chunks[d.region.chunk];
              CHECK(d.region.c == Dz &&
                        d.source1 ==
                            cylindrical_array(f, d.region.chunk, array_kind::f, Hp, d.region.cmp) &&
                        !is_valid(d.source2) && d.source1_neighbor_offset == 0 &&
                        d.source2_offset == 0 && d.scale == double(realnum(fc.Courant * 4)) &&
                        d.source2_multiplier == 0,
                    "m=0 cylindrical axis row has the wrong identity/offset/coefficient");
            }
            else {
              const fields_chunk &fc = *f.chunks[d.region.chunk];
              const bool electric = op.ft == D_stuff;
              const component target_component = electric ? Dp : Br;
              const component source1_component = electric ? Hr : Ep;
              const component source2_component = electric ? Hz : Ez;
              const int source2_cmp = electric ? d.region.cmp : 1 - d.region.cmp;
              CHECK(fabs(m) == 1 && d.region.c == target_component &&
                        d.source1 == cylindrical_array(f, d.region.chunk, array_kind::f,
                                                       source1_component, d.region.cmp) &&
                        d.source2 == cylindrical_array(f, d.region.chunk, array_kind::f,
                                                       source2_component, source2_cmp) &&
                        d.source1_neighbor_offset == (electric ? -1 : +1) &&
                        d.source2_offset == (electric ? 0 : fc.gv.nz() + 1) &&
                        d.scale == double(realnum((electric ? +1 : -1) * fc.Courant)) &&
                        d.source2_multiplier ==
                            double(realnum(electric ? 2 : (1 - 2 * d.region.cmp) * m)),
                    "|m|=1 cylindrical axis identity/offset/coefficient is wrong");
            }
            if (pure_conductivity) {
              CHECK((d.region.variant_key & cylindrical_axis_has_conductivity) == 0 &&
                        !(d.region.variant_key & cylindrical_axis_has_pml) &&
                        !(d.region.variant_key & cylindrical_axis_has_pml_aux) &&
                        !is_valid(d.target_cond) && !is_valid(d.conductivity) &&
                        !is_valid(d.condinv) && !is_valid(d.pml.sig) && !is_valid(d.pml_u.sig),
                    "no-PML conductivity case retained inactive axis operands");
            }
          }
          if (op.ft == D_stuff && i + 1 < size_t(op.cylindrical_origin_action_index) +
                                              op.cylindrical_origin_action_count)
            saw_axis_then_zero |= plan.cylindrical_origin_actions[i + 1].kind ==
                                  CylindricalOriginActionKind::zero_slab;
        }
        else {
          ++zero_actions;
          CHECK(action.index < plan.cylindrical_zero_slabs.size(),
                "origin action references an invalid zero slab");
          if (action.index < plan.cylindrical_zero_slabs.size()) {
            const SlabRef &slab = plan.cylindrical_zero_slabs[action.index];
            if (expected)
              CHECK(slab.array == expected->array,
                    "cylindrical zero action %zu has the wrong array identity", expected_index);
            const StorageKey &key = f.array_catalog->key(slab.array);
            const size_t row_stride = size_t(f.chunks[key.chunk]->gv.nz() + 1);
            CHECK(slab.base >= 0 && size_t(slab.base) % row_stride == 0 &&
                      slab.counts[0] == int(row_stride) && slab.strides[0] == 1,
                  "origin zero action is not one complete raw z row");
            if (expected)
              CHECK(slab.base == ptrdiff_t(expected->radial_row) * ptrdiff_t(row_stride),
                    "cylindrical zero action %zu has the wrong radial row", expected_index);
            if (fabs(m) > 1) {
              const int row = int(size_t(slab.base) / row_stride);
              const int expected_rows = zero_near_origin ? int(ceil(fabs(m))) : 1;
              CHECK(row >= 0 && row < expected_rows,
                    "high-m origin zero slab has the wrong radial row");
              int family = -1;
              if (key.kind == int(array_kind::f)) family = 0;
              if (key.kind == int(array_kind::f_cond)) family = 1;
              if (key.kind == int(array_kind::f_u)) family = 2;
              const std::vector<int> order = {key.chunk, key.cmp, row, family,
                                              component_index(component(key.component_))};
              CHECK(family >= 0 && (previous_zero_order.empty() || previous_zero_order <= order),
                    "high-m origin zero slabs do not follow row/family/component order");
              previous_zero_order = order;
            }
          }
        }
      }
    }
    else if (op.kind == OpKind::update_eh) {
      std::vector<ArrayId> expected_axis_replays;
      for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
        const fields_chunk &fc = *f.chunks[chunk];
        if (!fc.is_mine()) continue;
        const int cmps = fc.is_real ? 1 : 2;
        for (size_t tile = 0; tile < fc.gvs_eh[op.ft].size(); ++tile) {
          const grid_volume &sub = fc.gvs_eh[op.ft][tile];
          for (int cmp = 0; cmp < cmps; ++cmp)
            FOR_FT_COMPONENTS(op.ft, ec) {
              const component dc = field_type_component(op.ft == E_stuff ? D_stuff : B_stuff, ec);
              const ArrayId target = cylindrical_array(f, chunk, array_kind::f, ec, cmp);
              if (!is_valid(target) || !fc.f[ec][cmp] || fc.f[ec][cmp] == fc.f[dc][cmp]) continue;
              if (fc.gv.origin_r() == 0.0 && sub.little_owned_corner(ec).in_direction(R) == 0)
                expected_axis_replays.push_back(target);
            }
        }
      }
      size_t replay_index = 0;
      for (size_t i = op.descriptor_index; i < size_t(op.descriptor_index) + op.descriptor_count;
           ++i) {
        const ConstitutiveUpdate &d = plan.eh_updates[i];
        if (!(d.region.variant_key & constitutive_axis_override)) continue;
        ++axis_replays;
        CHECK(replay_index < expected_axis_replays.size(),
              "constitutive axis replay exceeds the exact expected sequence");
        if (replay_index < expected_axis_replays.size())
          CHECK(d.target == expected_axis_replays[replay_index],
                "constitutive axis replay %zu has the wrong target", replay_index);
        ++replay_index;
        CHECK(i > op.descriptor_index, "axis constitutive replay has no ordinary predecessor");
        if (i > op.descriptor_index) {
          const ConstitutiveUpdate &ordinary = plan.eh_updates[i - 1];
          CHECK(ordinary.target == d.target && ordinary.region.chunk == d.region.chunk &&
                    ordinary.region.cmp == d.region.cmp,
                "axis constitutive replay is not adjacent to its ordinary row");
          uint32_t expected_variant =
              ordinary.region.variant_key &
              ~(constitutive_one_offdiagonal | constitutive_two_offdiagonals |
                constitutive_has_minus_p | constitutive_copy_w_previous);
          if (ordinary.primary != ordinary.base_primary)
            expected_variant |= constitutive_has_minus_p;
          CHECK(d.base_primary == ordinary.base_primary && d.primary == ordinary.primary &&
                    d.diagonal == ordinary.diagonal && d.chi2 == ordinary.chi2 &&
                    d.chi3 == ordinary.chi3 && d.target_w == ordinary.target_w &&
                    d.primary_stride == ordinary.primary_stride && d.pml.sig == ordinary.pml.sig &&
                    d.pml.kap == ordinary.pml.kap && d.pml.siginv == ordinary.pml.siginv &&
                    d.pml.base == ordinary.pml.base &&
                    d.pml.strides[0] == ordinary.pml.strides[0] &&
                    d.pml.strides[1] == ordinary.pml.strides[1] &&
                    d.pml.strides[2] == ordinary.pml.strides[2] &&
                    d.region.variant_key == (expected_variant | constitutive_axis_override),
                "axis constitutive replay did not inherit its diagonal operands exactly");
        }
        CHECK(!is_valid(d.cross1) && !is_valid(d.cross2) && !is_valid(d.offdiagonal1) &&
                  !is_valid(d.offdiagonal2) && !is_valid(d.previous_w) && d.cross1_stride == 0 &&
                  d.cross2_stride == 0 && !(d.region.variant_key & constitutive_copy_w_previous),
              "axis constitutive replay retained cross/copy-W operands");
        CHECK(covers_access(op, d.target, AccessMode::read_write) &&
                  covers_access(op, d.base_primary, AccessMode::read) &&
                  covers_access(op, d.primary,
                                d.primary != d.base_primary ? AccessMode::read_write
                                                            : AccessMode::read) &&
                  covers_access(op, d.diagonal, AccessMode::read) &&
                  covers_access(op, d.chi2, AccessMode::read) &&
                  covers_access(op, d.chi3, AccessMode::read) &&
                  covers_access(op, d.target_w, AccessMode::read_write) &&
                  covers_access(op, d.pml.sig, AccessMode::read) &&
                  covers_access(op, d.pml.kap, AccessMode::read) &&
                  covers_access(op, d.pml.siginv, AccessMode::read),
              "axis constitutive replay accesses are incomplete");
        CHECK(d.region.begin.in_direction(R) == 0 && d.region.end.in_direction(R) == 0,
              "axis constitutive replay is not restricted to r=0");
      }
      CHECK(replay_index == expected_axis_replays.size(),
            "constitutive axis replay sequence ended after %zu of %zu expected rows", replay_index,
            expected_axis_replays.size());
    }
  }

  CHECK(and_to_all(!owns_chunk || (z_curls > 0 && prefixes == z_curls && r_suppressed > 0)),
        "an owning rank lacks complete cylindrical curl routing");
  CHECK(m == 0 ? and_to_all(m_rows == 0) : and_to_all(!owns_chunk || m_rows > 0),
        "cylindrical m/r row presence does not match m=%g", m);
  CHECK(and_to_all(!owns_origin_chunk || axis_actions + zero_actions > 0),
        "cylindrical origin emitted no actions");
  CHECK((m != 0 && fabs(m) != 1) || and_to_all(!owns_origin_chunk || saw_axis_then_zero),
        "m=0/|m|=1 D origin actions do not preserve arithmetic-before-zero order");
  CHECK(and_to_all(!owns_origin_chunk || axis_replays > 0),
        "cylindrical plan emitted no constitutive axis replay");
  CHECK(!use_bfast || and_to_all(!owns_chunk || saw_prefix_before_bfast),
        "cylindrical prefix/curl/BFAST pairing was not exercised");
  if (annular) {
    CHECK(and_to_all(axis_actions == 0 && zero_actions == 0 && axis_replays == 0),
          "annular cylindrical grid emitted origin work");
  }
  if (pure_conductivity) {
    CHECK(and_to_all(!owns_chunk || saw_conductivity),
          "pure-conductivity cylindrical plan lost its conductivity variants");
    CHECK(and_to_all(!saw_main_pml && !saw_aux_pml),
          "pure-conductivity cylindrical plan unexpectedly retained PML variants");
  }
  else {
    CHECK(or_to_all(saw_main_pml) && or_to_all(saw_aux_pml),
          "cylindrical plan did not cover primary and auxiliary PML variants");
  }
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
  check_bfast_plan();
  check_cylindrical_plan(0.0, true, false);
  check_cylindrical_plan(0.0, true, false, true);
  check_cylindrical_plan(+1.0, true, true);
  check_cylindrical_plan(-1.0, true, false);
  check_cylindrical_plan(+0.5, true, false);
  check_cylindrical_plan(+0.5, true, false, false, true);
  check_cylindrical_plan(+1.0, true, false, false, false, true);
  check_cylindrical_plan(+3.0, true, false);
  check_cylindrical_plan(-3.0, false, false);
  if (failures) {
    master_printf("prepared_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("prepared_plan: all checks passed\n");
  return 0;
}
