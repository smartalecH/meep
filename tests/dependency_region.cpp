/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#include "backend/dependency_region.hpp"

using namespace meep;

namespace {
int failures = 0;
#define CHECK(condition, message)                                                               \
  do {                                                                                          \
    if (!(condition)) {                                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, message);                  \
      ++failures;                                                                               \
    }                                                                                           \
  } while (0)

ArraySpec array_spec(uint32_t id) {
  ArraySpec result;
  result.id = ArrayId{id};
  result.role = array_role::field;
  result.element_type = ElementType::realnum_value;
  result.storage = Precision::f64;
  result.elements = 64;
  result.alignment = alignof(double);
  result.alias_of = invalid_array();
  result.classification_provisional = false;
  result.classification_elided = false;
  return result;
}

void make_fixture(StepPlan &step, StoragePlan &storage, LoweredRemoteHaloProgram &remote,
                  ptrdiff_t received_index = 3) {
  storage.arrays.push_back(array_spec(0));
  storage.arrays.push_back(array_spec(1));
  storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Dx), 0, 0});
  storage.keys.push_back(StorageKey{0, int(array_kind::f), int(Hy), 0, 0});

  Operation halo = {};
  halo.kind = OpKind::transfer_halo;
  halo.ft = B_stuff;
  halo.guard = guard_always();
  step.operations.push_back(halo);
  Operation update = {};
  update.kind = OpKind::update_db;
  update.ft = B_stuff;
  update.descriptor_count = 1;
  update.guard = guard_always();
  step.operations.push_back(update);
  CurlUpdate curl = {};
  curl.region.chunk = 0;
  curl.region.c = Dx;
  curl.region.cmp = 0;
  curl.region.begin = ivec(0, 0, 0);
  curl.region.end = ivec(2, 3, 0);
  curl.region.base = 0;
  curl.region.counts[0] = 3;
  curl.region.counts[1] = 4;
  curl.region.counts[2] = 1;
  curl.region.strides[0] = 1;
  curl.region.strides[1] = 4;
  curl.region.strides[2] = 16;
  curl.target = ArrayId{0};
  curl.plus_source = ArrayId{1};
  curl.minus_source = invalid_array();
  curl.plus_stride = 1;
  curl.radial_prefix_index = UINT32_MAX;
  curl.bfast_update_index = UINT32_MAX;
  step.db_updates.push_back(curl);
  step.signature = compute_step_plan_signature(step);

  remote.version = LoweredRemoteHaloProgram::schema_version;
  remote.program_signature = 41;
  remote.storage_signature = compute_remote_storage_authority_signature(storage);
  remote.authority_signature = compute_remote_lowered_authority_signature(
      remote.program_signature, remote.storage_signature);
  LoweredRemoteHaloStage stage = {};
  stage.ft = B_stuff;
  LoweredRemoteHaloMessage message = {};
  message.direction = RemoteHaloDirection::incoming;
  RemoteHaloScatterDescriptor scatter = {};
  scatter.target_real = invalid_remote_halo_scalar_ref();
  scatter.target_real.root = ArrayId{1};
  scatter.target_real.index = received_index;
  scatter.target_imag = invalid_remote_halo_scalar_ref();
  message.scatters.push_back(scatter);
  stage.receives.push_back(message);
  remote.stages.push_back(stage);
}

void test_exact_split_and_staleness() {
  StepPlan step;
  StoragePlan storage;
  LoweredRemoteHaloProgram remote = {};
  make_fixture(step, storage, remote);
  DependencyRegionPlan plan;
  std::string why;
  CHECK(build_dependency_region_plan(step, 0, remote, storage, plan, why), why.c_str());
  CHECK(plan.rows.size() == 1, "dependency proof lost curl row");
  if (plan.rows.size() == 1) {
    const DependencyRegionRow &row = plan.rows[0];
    CHECK(row.interior.counts[0] == 2 && row.interior.counts[1] == 4,
          "dependency proof did not erode the exact receive-dependent face");
    CHECK(row.boundary.size() == 1 && row.boundary[0].base == 2 &&
              row.boundary[0].counts[0] == 1,
          "dependency proof did not produce the exact boundary remainder");
  }
  CHECK(validate_dependency_region_plan(plan, step, remote, storage, why), why.c_str());
  if (!plan.rows.empty()) {
    ++plan.rows[0].interior.base;
    CHECK(!validate_dependency_region_plan(plan, step, remote, storage, why),
          "mutated dependency proof was accepted");
  }
}

void test_overlap_policy() {
  CHECK(parse_dependency_overlap_policy(NULL).requested ==
            DependencyOverlapPolicy::automatic,
        "unset overlap policy is not auto");
  CHECK(parse_dependency_overlap_policy("off").requested == DependencyOverlapPolicy::off,
        "off overlap policy was not parsed");
  CHECK(parse_dependency_overlap_policy("required").requested ==
            DependencyOverlapPolicy::required,
        "required overlap policy was not parsed");
  CHECK(!parse_dependency_overlap_policy("sometimes").valid,
        "malformed overlap policy was accepted");
  CHECK(!std::strcmp(resolved_dependency_overlap_name(0), "off"),
        "off overlap with no admitted regions resolved incorrectly");
  CHECK(parse_dependency_overlap_policy("auto").requested ==
            DependencyOverlapPolicy::automatic &&
            !std::strcmp(resolved_dependency_overlap_name(0), "off") &&
            !std::strcmp(resolved_dependency_overlap_name(1), "overlap"),
        "automatic overlap did not reflect admitted dependency regions");
  CHECK(parse_dependency_overlap_policy("required").requested ==
            DependencyOverlapPolicy::required &&
            !std::strcmp(resolved_dependency_overlap_name(1), "overlap"),
        "required overlap with an admitted region resolved incorrectly");
}

void test_interior_and_thin_rejection() {
  StepPlan step;
  StoragePlan storage;
  LoweredRemoteHaloProgram remote = {};
  make_fixture(step, storage, remote, 6);
  DependencyRegionPlan plan;
  std::string why;
  CHECK(!build_dependency_region_plan(step, 0, remote, storage, plan, why),
        "interior receive dependency was accepted");

  step = StepPlan();
  storage = StoragePlan();
  remote = LoweredRemoteHaloProgram();
  make_fixture(step, storage, remote, 1);
  step.db_updates[0].region.counts[0] = 1;
  step.db_updates[0].region.counts[1] = 1;
  step.db_updates[0].region.end = ivec(0, 0, 0);
  step.signature = compute_step_plan_signature(step);
  CHECK(!build_dependency_region_plan(step, 0, remote, storage, plan, why),
        "empty dependency interior was accepted");
}

void test_negative_offset_and_authority_rejection() {
  StepPlan step;
  StoragePlan storage;
  LoweredRemoteHaloProgram remote = {};
  make_fixture(step, storage, remote, 1);
  step.db_updates[0].region.base = 2;
  step.db_updates[0].plus_stride = -1;
  step.signature = compute_step_plan_signature(step);
  DependencyRegionPlan plan;
  std::string why;
  CHECK(build_dependency_region_plan(step, 0, remote, storage, plan, why), why.c_str());
  if (!plan.rows.empty()) {
    CHECK(plan.rows[0].interior.base == 3 && plan.rows[0].interior.counts[0] == 2,
          "negative stencil offset did not erode the low face");
    size_t points = plan.rows[0].interior.counts[0] * plan.rows[0].interior.counts[1] *
                    plan.rows[0].interior.counts[2];
    for (const DependencyBox &box : plan.rows[0].boundary)
      points += box.counts[0] * box.counts[1] * box.counts[2];
    CHECK(points == 12, "dependency split does not exactly cover the full region");
  }
  remote.authority_signature ^= 1;
  CHECK(!build_dependency_region_plan(step, 0, remote, storage, plan, why),
        "stale remote authority was accepted");

  step = StepPlan();
  storage = StoragePlan();
  remote = LoweredRemoteHaloProgram();
  make_fixture(step, storage, remote, 3);
  Operation gap = {};
  gap.kind = OpKind::update_flux;
  gap.ft = NO_FIELD_TYPE;
  gap.guard = guard_always();
  step.operations.insert(step.operations.begin() + 1, gap);
  step.signature = compute_step_plan_signature(step);
  CHECK(!build_dependency_region_plan(step, 0, remote, storage, plan, why),
        "non-adjacent dependency update was accepted");

  step = StepPlan();
  storage = StoragePlan();
  remote = LoweredRemoteHaloProgram();
  make_fixture(step, storage, remote, 3);
  step.db_updates[0].plus_stride = std::numeric_limits<ptrdiff_t>::min();
  step.signature = compute_step_plan_signature(step);
  CHECK(!build_dependency_region_plan(step, 0, remote, storage, plan, why),
        "overflowing dependency offset was accepted");
}
} // namespace

int main() {
  test_overlap_policy();
  test_exact_split_and_staleness();
  test_interior_and_thin_rejection();
  test_negative_offset_and_authority_rejection();
  if (failures) std::fprintf(stderr, "%d dependency-region checks failed\n", failures);
  return failures ? 1 : 0;
}
