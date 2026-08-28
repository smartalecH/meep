/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include <math.h>
#include <stdio.h>

#include <vector>

#include <meep.hpp>

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
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  src.is_integrated = false;
  f.add_point_source(Ez, src, vec(0.13, 0.11));
  f.advance(2);

  StepPlan plan = build_step_plan(f, StepProgram::ordinary);
  size_t update_ops = 0;
  for (size_t oi = 0; oi < plan.operations.size(); ++oi) {
    const Operation &op = plan.operations[oi];
    if (op.kind != OpKind::update_db && op.kind != OpKind::update_eh) continue;
    ++update_ops;
    CHECK(op.descriptor_count > 0, "%s has an empty descriptor span", op_kind_name(op.kind));
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

  CHECK(compute_step_plan_signature(plan) == plan.signature,
        "stored signature differs from structural signature");
  if (!plan.db_updates.empty()) {
    const uint64_t original = plan.signature;
    ++plan.db_updates[0].plus_stride;
    CHECK(compute_step_plan_signature(plan) != original,
          "signature ignored a structural curl descriptor change");
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
  for (size_t oi = 0; oi < plan.operations.size(); ++oi) {
    const Operation &op = plan.operations[oi];
    if (op.kind != OpKind::update_eh || op.ft != E_stuff) continue;
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

  for (size_t i = 0; i < f.array_catalog->size(); ++i) {
    const ArraySpec &spec = f.array_catalog->spec(ArrayId{uint32_t(i)});
    if (!is_valid(spec.alias_of)) continue;
    for (size_t j = 0; j < plan.eh_updates.size(); ++j)
      CHECK(plan.eh_updates[j].target != spec.id,
            "H/B alias was emitted as a constitutive write descriptor");
  }
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;
  check_prepared_updates();
  check_integrated_source_inputs();
  check_one_cross_normalization();
  check_alias_elision();
  if (failures) {
    master_printf("prepared_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("prepared_plan: all checks passed\n");
  return 0;
}
