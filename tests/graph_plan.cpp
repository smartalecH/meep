/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <meep.hpp>

#include "backend/graph_plan.hpp"

using namespace meep;

static int failures = 0;

#define CHECK(condition, ...)                                                                      \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      std::printf("FAIL (%s:%d): ", __FILE__, __LINE__);                                           \
      std::printf(__VA_ARGS__);                                                                    \
      std::printf("\n");                                                                           \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

static Operation operation(OpKind kind, field_type ft = NO_FIELD_TYPE,
                           Guard guard = guard_always()) {
  Operation result = {};
  result.kind = kind;
  result.ft = ft;
  result.guard = guard;
  if (kind == OpKind::evaluate_source_scalars) result.descriptor_count = 1;
  return result;
}

static double source_fixture_epsilon(const vec &) { return 1.0; }

static StepPlan signed_plan(const std::vector<Operation> &operations,
                            StepProgram program = StepProgram::ordinary) {
  StepPlan plan;
  plan.program = program;
  plan.operations = operations;
  plan.signature = compute_step_plan_signature(plan);
  return plan;
}

static CwStepOperationRef cw_ref(const StepPlan &plan, uint32_t operation_index) {
  const Operation &operation = plan.operations[operation_index];
  CwStepOperationRef result;
  result.operation_index = operation_index;
  result.kind = operation.kind;
  result.ft = operation.ft;
  result.descriptor_index = operation.descriptor_index;
  result.descriptor_count = operation.descriptor_count;
  result.polarization_subtraction_index = operation.polarization_subtraction_index;
  result.polarization_subtraction_count = operation.polarization_subtraction_count;
  return result;
}

static HaloPlan halo(field_type ft, bool local, int peer, ArrayId source) {
  HaloPlan result = {};
  result.ft = ft;
  result.chunks = chunk_pair(0, 1);
  result.phase = CONNECT_COPY;
  result.peer_rank = peer;
  result.tag = 17;
  result.same_rank = local;
  result.storage = HaloStorageDisposition::canonical;
  result.sequence_index = uint32_t(CONNECT_COPY);
  result.block_offset = 0;
  result.block_elements = 4;
  for (int i = 0; i < 4; ++i)
    result.gather.push_back(ElementRef{source, i});
  for (int i = 0; i < 4; ++i)
    result.scatter.push_back(ElementRef{source, i + 4});
  result.gather_order.push_back(HaloSegment{0, 0, 0, 4});
  result.scatter_order.push_back(HaloSegment{0, 0, 0, 4});
  return result;
}

struct HaloFixture {
  realnum values[16];
  realnum opaque[4];
  realnum opaque_host[4];
  CpuArrayCatalog catalog;
  halo_plan_set halos;
  ArrayId source;
  ArrayId canonical;

  explicit HaloFixture(bool reverse_ids = false) {
    std::memset(values, 0, sizeof(values));
    std::memset(opaque, 0, sizeof(opaque));
    std::memset(opaque_host, 0, sizeof(opaque_host));
    const HaloArrayKey main_halo{0, int(array_role::field), int(Bx), 0, 0};
    const HaloArrayKey aux_halo{0, int(array_role::field), int(By), 0, 0};
    const StorageKey main_storage{0, int(array_kind::f), int(Bx), 0, 0};
    const StorageKey aux_storage{0, int(array_kind::f), int(By), 0, 0};
    if (reverse_ids) {
      (void)halos.arrays.intern(aux_halo, opaque, 4, array_role::field);
      (void)catalog.register_array(aux_storage, opaque, 4, array_role::field,
                                   ElementType::realnum_value);
    }
    source = halos.arrays.intern(main_halo, values, 16, array_role::field);
    canonical = catalog.register_array(main_storage, values, 16, array_role::field,
                                       ElementType::realnum_value);
    if (!reverse_ids) {
      (void)halos.arrays.intern(aux_halo, opaque, 4, array_role::field);
      (void)catalog.register_array(aux_storage, opaque, 4, array_role::field,
                                   ElementType::realnum_value);
    }
  }

  GraphLoweringAuthorities authority(const StepPlan &plan, const CwPlan *cw = NULL,
                                     int interleave = 1) const {
    return build_graph_lowering_authorities(plan, &halos, &catalog, interleave, cw);
  }
};

static void add_opaque_halo(HaloFixture &fixture, field_type ft) {
  HaloPlan result = {};
  result.ft = ft;
  result.chunks = chunk_pair(0, 1);
  result.phase = CONNECT_COPY;
  result.peer_rank = 0;
  result.tag = 23;
  result.same_rank = true;
  result.storage = HaloStorageDisposition::host_owned;
  result.sequence_index = uint32_t(CONNECT_COPY);
  result.block_elements = 4;
  for (int i = 0; i < 4; ++i) {
    const int state_ft = ft == PE_stuff ? E_stuff : H_stuff;
    const HaloArrayKey array_key{0, int(array_role::polarization), int(Ex), 0, 0, 7, i, i, false};
    const ArrayId source = fixture.halos.arrays.intern(array_key, fixture.opaque_host + i, 1,
                                                       array_role::polarization);
    result.gather.push_back(ElementRef{source, 0});
    const HostHaloKey host_key{0, state_ft, 0, 7, int(Ex), 0, i, i, false};
    result.host_gather.push_back(
        HostElementRef{fixture.halos.host_arrays.intern(host_key, fixture.opaque_host + i)});
  }
  result.gather_order.push_back(HaloSegment{0, 0, 0, 4});
  fixture.halos.plans.push_back(result);
}

static void add_remappable_multilevel_halo(HaloFixture &fixture) {
  const StorageKey storage_key{0, int(array_kind::polarization_internal), int(Ex), 0, 29};
  (void)fixture.catalog.register_array(storage_key, fixture.opaque_host, 4,
                                       array_role::polarization, ElementType::realnum_value);
  HaloPlan result = {};
  result.ft = PE_stuff;
  result.chunks = chunk_pair(0, 1);
  result.phase = CONNECT_COPY;
  result.peer_rank = 0;
  result.tag = 29;
  result.same_rank = true;
  result.storage = HaloStorageDisposition::host_owned;
  result.sequence_index = uint32_t(CONNECT_COPY);
  result.block_elements = 4;
  for (int i = 0; i < 4; ++i) {
    const HaloArrayKey array_key{0, int(array_role::polarization), int(Ex), 0, 2, 7, i, i, false};
    const ArrayId source = fixture.halos.arrays.intern(
        array_key, fixture.opaque_host + i, 1, array_role::polarization);
    result.gather.push_back(ElementRef{source, 0});
    const HostHaloKey host_key{0, E_stuff, 2, 7, int(Ex), 0, i, i, false};
    result.host_gather.push_back(
        HostElementRef{fixture.halos.host_arrays.intern(host_key, fixture.opaque_host + i)});
  }
  result.gather_order.push_back(HaloSegment{0, 0, 0, 4});
  fixture.halos.plans.push_back(result);
}

static void test_scalar_layout() {
  StepPlan plan;
  Operation dft = operation(OpKind::update_dft, NO_FIELD_TYPE, guard_device(7));
  dft.descriptor_index = 0;
  dft.descriptor_count = 3;
  plan.operations.push_back(dft);
  DftDescriptor first = {};
  first.decimation_factor = 2;
  DftDescriptor second = {};
  second.decimation_factor = 5;
  DftDescriptor third = {};
  third.decimation_factor = 2;
  plan.dft_updates.push_back(first);
  plan.dft_updates.push_back(second);
  plan.dft_updates.push_back(third);
  plan.signature = compute_step_plan_signature(plan);

  const StepScalarLayout layout = build_step_scalar_layout(plan);
  CHECK(layout.abi_version == step_scalars_abi_version, "wrong scalar ABI version");
  CHECK(layout.total_bytes == sizeof(StepScalars), "wrong scalar ABI size");
  size_t guard_slots = 0, dft_slots = 0;
  for (const StepScalarSlot &slot : layout.slots) {
    if (slot.semantic == StepScalarSemantic::guard_predicate) ++guard_slots;
    if (slot.semantic == StepScalarSemantic::dft_due_predicate) ++dft_slots;
  }
  CHECK(guard_slots == 1, "expected one unique guard slot, got %zu", guard_slots);
  CHECK(dft_slots == 2, "expected two unique DFT cadence slots, got %zu", dft_slots);

  StepPlan malformed = plan;
  malformed.dft_updates[0].decimation_factor = 0;
  malformed.signature = compute_step_plan_signature(malformed);
  bool rejected = false;
  try {
    (void)build_step_scalar_layout(malformed);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected, "invalid DFT decimation was accepted");
}

static void test_halo_authority_and_boundaries() {
  std::vector<Operation> operations;
  operations.push_back(operation(OpKind::update_db, B_stuff));
  operations.push_back(operation(OpKind::transfer_halo, B_stuff));
  operations.push_back(operation(OpKind::evaluate_source_scalars));
  Operation dft = operation(OpKind::update_dft, NO_FIELD_TYPE, guard_device(0));
  operations.push_back(dft);
  operations.push_back(operation(OpKind::update_flux));
  operations.push_back(operation(OpKind::finite_value_check));
  operations.push_back(operation(OpKind::increment_time));
  StepPlan plan = signed_plan(operations);

  HaloFixture fixture;
  fixture.halos.plans.push_back(halo(B_stuff, true, 0, fixture.source));
  fixture.halos.zeros[B_stuff].resize(1);
  fixture.halos.zeros[B_stuff][0].residue.push_back(ElementRef{fixture.source, 12});
  GraphLoweringAuthorities authority = fixture.authority(plan);
  CHECK(authority.halo_dispositions.size() == 1, "missing halo disposition");
  CHECK(authority.halo_dispositions[0].entirely_local_canonical,
        "local canonical halo was not graphable");
  std::string error;
  CHECK(validate_graph_lowering_authorities(plan, authority, &fixture.halos, &fixture.catalog, 1,
                                            NULL, &error),
        "valid halo authority rejected: %s", error.c_str());

  GraphProgram graph = build_graph_program(plan, authority, GraphVariantKind::ordinary);
  CHECK(validate_graph_program(plan, authority, graph, &error), "valid graph program rejected: %s",
        error.c_str());
  CHECK(graph.segments.size() == 3, "expected 3 maximal device segments, got %zu",
        graph.segments.size());
  CHECK(graph.boundaries.size() == 4, "expected 4 host/completion boundaries, got %zu",
        graph.boundaries.size());
  CHECK(graph.segments[0].first_operation == 0 && graph.segments[0].operation_count == 2,
        "local halo did not remain in the first maximal device segment");
  CHECK(graph.boundaries[1].kind == GraphBoundaryKind::legacy_flux_publish &&
            graph.boundaries[1].completion_only,
        "legacy flux completion boundary is missing");
  CHECK(graph.boundaries[2].kind == GraphBoundaryKind::finite_diagnostic &&
            graph.boundaries[2].completion_only,
        "finite diagnostic completion boundary is missing");

  HaloFixture remote;
  remote.halos.plans.push_back(halo(B_stuff, false, 1, remote.source));
  const GraphLoweringAuthorities remote_authority = remote.authority(plan);
  const GraphProgram remote_graph =
      build_graph_program(plan, remote_authority, GraphVariantKind::ordinary);
  bool found_remote = false;
  for (const GraphBoundary &boundary : remote_graph.boundaries)
    found_remote = found_remote || boundary.kind == GraphBoundaryKind::remote_halo;
  CHECK(found_remote, "remote halo was inferred as graphable");

  StepPlan multilevel_plan =
      signed_plan(std::vector<Operation>(1, operation(OpKind::transfer_halo, PE_stuff)));
  HaloFixture remappable;
  add_remappable_multilevel_halo(remappable);
  CHECK(remappable.halos.plans[0].storage == HaloStorageDisposition::host_owned,
        "multilevel remap fixture did not start host-owned");
  const GraphLoweringAuthorities remappable_authority = remappable.authority(multilevel_plan);
  CHECK(remappable_authority.halo_rows.size() == 1 &&
            remappable_authority.halo_rows[0].route == GraphHaloRoute::local_device &&
            remappable_authority.halo_dispositions[0].entirely_local_canonical,
        "catalog-backed multilevel host row did not remap to canonical local-device storage");

  StepPlan opaque_plan = plan;
  opaque_plan.operations[1].ft = PE_stuff;
  opaque_plan.signature = compute_step_plan_signature(opaque_plan);
  HaloFixture host_owned;
  add_opaque_halo(host_owned, PE_stuff);
  const GraphLoweringAuthorities host_authority = host_owned.authority(opaque_plan);
  CHECK(!host_authority.halo_dispositions[0].entirely_local_canonical,
        "host-owned halo was inferred as graphable");

  GraphLoweringAuthorities stale = authority;
  stale.halo_rows[0].peer_rank = 9;
  stale.signature = compute_graph_lowering_authorities_signature(stale);
  CHECK(!validate_graph_lowering_authorities(plan, stale, &fixture.halos, &fixture.catalog, 1, NULL,
                                             &error),
        "re-signed stale halo authority matched the live HaloPlan");
}

static void test_host_covered_interval_once() {
  StepPlan plan;
  Operation marker = operation(OpKind::host_callback, E_stuff);
  marker.descriptor_index = 0;
  marker.descriptor_count = 1;
  plan.operations.push_back(marker);
  plan.operations.push_back(operation(OpKind::update_eh, E_stuff));
  plan.operations.push_back(operation(OpKind::update_db, D_stuff));
  plan.host_segments.push_back(
      HostSegment{HostSegmentPhase::constitutive, E_stuff, 1, 1, 0, 0, 0, 0});
  plan.signature = compute_step_plan_signature(plan);

  HaloFixture fixture;
  const GraphLoweringAuthorities authority = fixture.authority(plan);
  const GraphProgram graph = build_graph_program(plan, authority, GraphVariantKind::ordinary);
  CHECK(graph.boundaries.size() == 1, "host interval produced duplicate boundaries");
  CHECK(graph.boundaries[0].first_operation == 0 && graph.boundaries[0].operation_count == 2,
        "host interval did not consume marker and covered operation exactly once");
  CHECK(graph.segments.size() == 1 && graph.segments[0].first_operation == 2,
        "operation after host interval was not emitted exactly once");
}

static void test_variants_and_canonical_validation() {
  std::vector<Operation> operations;
  operations.push_back(operation(OpKind::restore_magnetic_fields));
  operations.push_back(operation(OpKind::evaluate_source_scalars));
  operations.push_back(operation(OpKind::update_db, B_stuff));
  operations.push_back(operation(OpKind::apply_sources, B_stuff));
  operations.push_back(operation(OpKind::transfer_halo, B_stuff));
  operations.push_back(operation(OpKind::evaluate_source_scalars));
  operations.push_back(operation(OpKind::update_eh, H_stuff));
  operations.push_back(operation(OpKind::transfer_halo, H_stuff));
  StepPlan plan = signed_plan(operations);
  plan.magnetic_half_step.evaluate_b_sources = 1;
  plan.magnetic_half_step.update_b = 2;
  plan.magnetic_half_step.apply_b_sources = 3;
  plan.magnetic_half_step.transfer_b = 4;
  plan.magnetic_half_step.evaluate_h_sources = 5;
  plan.magnetic_half_step.update_h = 6;
  plan.magnetic_half_step.transfer_h = 7;
  plan.signature = compute_step_plan_signature(plan);
  HaloFixture fixture;
  fixture.halos.plans.push_back(halo(B_stuff, true, 0, fixture.source));
  HaloPlan h_halo = halo(H_stuff, true, 0, fixture.source);
  fixture.halos.plans.push_back(h_halo);
  const GraphLoweringAuthorities authority = fixture.authority(plan);
  const GraphProgram half =
      build_graph_program(plan, authority, GraphVariantKind::magnetic_half_step);
  CHECK(half.boundaries.size() == 2 &&
            half.boundaries[0].kind == GraphBoundaryKind::source_evaluation,
        "magnetic half-step lost its source boundary");
  CHECK(half.segments.size() == 2 && half.segments[0].operation_count == 3 &&
            half.segments[1].operation_count == 2,
        "magnetic half-step device operations were not maximally segmented");
  const GraphProgram restore =
      build_graph_program(plan, authority, GraphVariantKind::magnetic_restore);
  CHECK(restore.segments.size() == 1 && restore.boundaries.empty(),
        "magnetic restore did not lower as a distinct device variant");

  GraphProgram changed = half;
  changed.segments[0].first_operation = 0;
  changed.segments[0].signature = 1;
  changed.signature = compute_graph_program_signature(changed);
  std::string error;
  CHECK(!validate_graph_program(plan, authority, changed, &error),
        "re-signed noncanonical graph program was accepted");

  Operation evaluate_b = operation(OpKind::evaluate_source_scalars);
  Operation update_b = operation(OpKind::update_db, B_stuff);
  update_b.accesses.push_back(
      BufferAccess{ArrayRef{fixture.canonical, 0, 16}, AccessMode::read_write});
  Operation evaluate_h = operation(OpKind::evaluate_source_scalars);
  evaluate_h.source_time_offset = 0.5;
  Operation evaluate_d = evaluate_h;
  Operation evaluate_e = operation(OpKind::evaluate_source_scalars);
  evaluate_e.source_time_offset = 1.0;
  StepPlan cw = signed_plan(
      std::vector<Operation>{evaluate_b,
                             update_b,
                             operation(OpKind::transfer_halo, B_stuff),
                             evaluate_h,
                             operation(OpKind::update_eh, H_stuff),
                             evaluate_d,
                             operation(OpKind::update_db, D_stuff),
                             operation(OpKind::transfer_halo, D_stuff),
                             evaluate_e,
                             operation(OpKind::update_eh, E_stuff),
                             operation(OpKind::transfer_halo, E_stuff)},
      StepProgram::solve_cw);
  fixture.halos.plans.push_back(halo(D_stuff, true, 0, fixture.source));
  fixture.halos.plans.push_back(halo(E_stuff, true, 0, fixture.source));
  CwPlan cw_plan;
  cw_plan.step_plan_signature = cw.signature;
  cw_plan.state_layout_signature = cw.cw_state_layout.signature;
  cw_plan.rhs_sources.push_back(CwRhsSourceDescriptor{
      0, 0, CwRhsSourceMode::primary_subtract_current_dt_including_integrated});
  cw_plan.rhs_sources.push_back(CwRhsSourceDescriptor{
      1, 0, CwRhsSourceMode::primary_subtract_current_dt_including_integrated});
  CwRhsStage b_stage = {};
  b_stage.ft = B_stuff;
  b_stage.source_time_offset = 0.0;
  b_stage.source_time_count = 1;
  b_stage.source_count = 1;
  b_stage.boundary = cw_ref(cw, 2);
  b_stage.constitutive = cw_ref(cw, 4);
  cw_plan.rhs_stages.push_back(b_stage);
  CwRhsStage d_stage = {};
  d_stage.ft = D_stuff;
  d_stage.source_time_offset = 0.5;
  d_stage.source_time_count = 1;
  d_stage.source_index = 1;
  d_stage.source_count = 1;
  d_stage.boundary = cw_ref(cw, 7);
  d_stage.constitutive = cw_ref(cw, 9);
  cw_plan.rhs_stages.push_back(d_stage);
  cw_plan.unpack.first_boundary = cw_ref(cw, 7);
  cw_plan.unpack.constitutive = cw_ref(cw, 9);
  cw_plan.unpack.second_boundary = cw_ref(cw, 10);
  cw_plan.final_dfts.push_back(CwDftDescriptorRef{0, 0, Ez, 1, 0});
  cw_plan.rhs_source_count = 2;
  cw_plan.source_time_count = 1;
  cw_plan.final_dft_count = 1;
  cw_plan.rhs_accesses = update_b.accesses;
  cw_plan.signature = compute_cw_plan_signature(cw_plan);
  const GraphLoweringAuthorities cw_authority =
      build_graph_lowering_authorities(cw, &fixture.halos, &fixture.catalog, 1, &cw_plan);
  const GraphProgram cw_graph =
      build_graph_program(cw, cw_authority, GraphVariantKind::cw_operator);
  CHECK(cw.dft_updates.empty() && cw_graph.cw_plan_signature == cw_plan.signature &&
            !cw_graph.segments.empty(),
        "CW graph did not bind both StepPlan and CwPlan identities");

  auto rejected_cw = [&](CwPlan malformed, const char *message) {
    malformed.signature = compute_cw_plan_signature(malformed);
    bool rejected = false;
    try {
      (void)build_graph_lowering_authorities(cw, &fixture.halos, &fixture.catalog, 1, &malformed);
    }
    catch (const std::invalid_argument &) {
      rejected = true;
    }
    CHECK(rejected, "%s", message);
  };
  CwPlan malformed_cw = cw_plan;
  malformed_cw.final_dfts[0].descriptor_index = 1;
  rejected_cw(malformed_cw, "out-of-order CwPlan-owned DFT descriptor was accepted");
  malformed_cw = cw_plan;
  malformed_cw.rhs_stages[1].source_index = 0;
  rejected_cw(malformed_cw, "overlapping CW RHS source coverage was accepted");
  malformed_cw = cw_plan;
  ++malformed_cw.rhs_stages[0].boundary.descriptor_count;
  rejected_cw(malformed_cw, "stale CW RHS operation reference was accepted");
  malformed_cw = cw_plan;
  malformed_cw.unpack.second_boundary.operation_index = 7;
  rejected_cw(malformed_cw, "stale CW unpack operation reference was accepted");

  HaloFixture rebound(true);
  rebound.halos.plans.push_back(halo(B_stuff, true, 0, rebound.source));
  rebound.halos.plans.push_back(halo(D_stuff, true, 0, rebound.source));
  rebound.halos.plans.push_back(halo(E_stuff, true, 0, rebound.source));
  StepPlan rebound_cw = cw;
  rebound_cw.operations[1].accesses[0].array.id = rebound.canonical;
  rebound_cw.signature = compute_step_plan_signature(rebound_cw);
  CwPlan rebound_plan = cw_plan;
  rebound_plan.step_plan_signature = rebound_cw.signature;
  rebound_plan.rhs_accesses[0].array.id = rebound.canonical;
  rebound_plan.signature = compute_cw_plan_signature(rebound_plan);
  const GraphLoweringAuthorities rebound_authority = build_graph_lowering_authorities(
      rebound_cw, &rebound.halos, &rebound.catalog, 1, &rebound_plan);
  const GraphProgram rebound_graph =
      build_graph_program(rebound_cw, rebound_authority, GraphVariantKind::cw_operator);
  CHECK(rebound_authority.signature == cw_authority.signature &&
            rebound_graph.signature == cw_graph.signature,
        "CW graph identity depends on generation-local ArrayIds");
}

static void test_all_operation_classes() {
  struct Expected {
    OpKind kind;
    bool boundary;
    bool completion;
  };
  const Expected expected[] = {
      {OpKind::restore_magnetic_fields, true, false},
      {OpKind::phase_material, true, false},
      {OpKind::update_material_coefficients, true, false},
      {OpKind::evaluate_source_scalars, true, false},
      {OpKind::update_db, false, false},
      {OpKind::update_eh, false, false},
      {OpKind::update_polarization, false, false},
      {OpKind::apply_sources, false, false},
      {OpKind::zero_boundary, false, false},
      {OpKind::pack_halo, false, false},
      {OpKind::transfer_halo, false, false},
      {OpKind::exchange_local, false, false},
      {OpKind::unpack_halo, false, false},
      {OpKind::update_flux_half, false, false},
      {OpKind::update_flux, false, true},
      {OpKind::increment_time, true, false},
      {OpKind::update_dft, false, false},
      {OpKind::synchronize_magnetic_fields, true, false},
      {OpKind::finite_value_check, false, true},
      {OpKind::reduction, false, false},
      {OpKind::pack_state, false, false},
      {OpKind::unpack_state, false, false},
  };
  for (const Expected &item : expected) {
    StepPlan plan = signed_plan(std::vector<Operation>(1, operation(item.kind, B_stuff)));
    HaloFixture fixture;
    if (item.kind == OpKind::transfer_halo)
      fixture.halos.plans.push_back(halo(B_stuff, true, 0, fixture.source));
    const GraphLoweringAuthorities authority = fixture.authority(plan);
    const GraphProgram graph = build_graph_program(plan, authority, GraphVariantKind::ordinary);
    if (item.boundary) {
      CHECK(graph.segments.empty() && graph.boundaries.size() == 1,
            "%s was not classified as a host boundary", op_kind_name(item.kind));
    }
    else {
      CHECK(graph.segments.size() == 1, "%s was not classified as a device operation",
            op_kind_name(item.kind));
      CHECK((graph.boundaries.size() == 1) == item.completion,
            "%s completion-boundary classification is wrong", op_kind_name(item.kind));
    }
  }

  StepPlan invalid = signed_plan(
      std::vector<Operation>(1, operation(static_cast<OpKind>(uint32_t(OpKind::num_kinds)))));
  bool rejected = false;
  try {
    HaloFixture fixture;
    (void)fixture.authority(invalid);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected, "invalid operation kind was accepted");
}

static void test_formatting_and_stale_signatures() {
  StepPlan plan = signed_plan(std::vector<Operation>(1, operation(OpKind::update_db, B_stuff)));
  HaloFixture fixture;
  GraphLoweringAuthorities authority = fixture.authority(plan);
  GraphProgram graph = build_graph_program(plan, authority, GraphVariantKind::ordinary);
  std::vector<std::string> lines;
  format_graph_program(graph, lines);
  CHECK(lines.size() == 1 && lines[0] == "segment[0+1]", "unexpected graph format");
  CHECK(graph_required_compatible(graph), "device segment rejected required mode");
  CHECK(parse_graph_execution_mode(NULL) == GraphExecutionMode::automatic &&
            parse_graph_execution_mode("auto") == GraphExecutionMode::automatic &&
            parse_graph_execution_mode("eager") == GraphExecutionMode::eager &&
            parse_graph_execution_mode("required") == GraphExecutionMode::required,
        "graph execution mode parser returned the wrong policy");
  bool bad_mode = false;
  try {
    (void)parse_graph_execution_mode("segment-fallback");
  }
  catch (const std::invalid_argument &) {
    bad_mode = true;
  }
  CHECK(bad_mode, "per-segment fallback policy was accepted");

  graph.step_plan_signature ^= 1;
  graph.signature = compute_graph_program_signature(graph);
  std::string error;
  CHECK(!validate_graph_program(plan, authority, graph, &error),
        "re-signed graph with stale StepPlan identity was accepted");

  plan.signature ^= 1;
  bool rejected = false;
  try {
    (void)fixture.authority(plan);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected, "stale StepPlan signature was accepted");
}

static void test_guard_zero_and_stable_identity() {
  HaloFixture first(false), second(true);
  first.halos.plans.push_back(halo(B_stuff, true, 0, first.source));
  second.halos.plans.push_back(halo(B_stuff, true, 0, second.source));
  first.halos.zeros[B_stuff].resize(1);
  second.halos.zeros[B_stuff].resize(1);
  first.halos.zeros[B_stuff][0].residue.push_back(ElementRef{first.source, 10});
  second.halos.zeros[B_stuff][0].residue.push_back(ElementRef{second.source, 10});

  Operation update = operation(OpKind::update_db, B_stuff);
  update.accesses.push_back(BufferAccess{ArrayRef{first.canonical, 0, 16}, AccessMode::read_write});
  Operation guarded = operation(OpKind::transfer_halo, B_stuff, guard_segment(3));
  StepPlan plan1 = signed_plan(std::vector<Operation>{update, guarded});
  StepPlan plan2 = plan1;
  plan2.operations[0].accesses[0].array.id = second.canonical;
  plan2.signature = compute_step_plan_signature(plan2);

  const GraphLoweringAuthorities authority1 = first.authority(plan1);
  const GraphLoweringAuthorities authority2 = second.authority(plan2);
  CHECK(authority1.zero_rows.size() == 1 && authority1.zero_rows[0].elements == 1,
        "fused transfer_halo did not include its metal ZeroPlan in graph authority");
  CHECK(authority1.zero_dispositions.size() == 1 &&
            authority1.zero_dispositions[0].operation_index == 1 &&
            authority1.zero_dispositions[0].ft == B_stuff,
        "metal ZeroPlan was not bound to the canonical transfer_halo operation");
  CHECK(authority1.signature == authority2.signature,
        "authority identity depends on generation-local ArrayId ordering");
  const GraphProgram graph1 = build_graph_program(plan1, authority1, GraphVariantKind::ordinary);
  const GraphProgram graph2 = build_graph_program(plan2, authority2, GraphVariantKind::ordinary);
  CHECK(graph1.signature == graph2.signature,
        "graph identity depends on generation-local ArrayId ordering");
  bool guarded_boundary = false;
  for (const GraphBoundary &boundary : graph1.boundaries)
    guarded_boundary = guarded_boundary || boundary.kind == GraphBoundaryKind::segment_guard;
  CHECK(guarded_boundary, "segment guard did not dominate local halo locality");

  HaloFixture changed;
  changed.halos.plans.push_back(halo(B_stuff, true, 0, changed.source));
  changed.halos.zeros[B_stuff].resize(1);
  changed.halos.zeros[B_stuff][0].residue.push_back(ElementRef{changed.source, 11});
  StepPlan changed_plan = plan1;
  changed_plan.operations[0].accesses[0].array.id = changed.canonical;
  changed_plan.signature = compute_step_plan_signature(changed_plan);
  CHECK(changed.authority(changed_plan).signature != authority1.signature,
        "fused transfer_halo ZeroPlan mutation did not change graph authority identity");
}

static void test_source_time_descriptor_semantics() {
  grid_volume gv = vol2d(2.0, 2.0, 10.0);
  structure s(gv, source_fixture_epsilon, pml(0.25), identity(), 2);
  fields f(&s);
  gaussian_src_time gaussian(0.3, 0.1);
  continuous_src_time continuous(0.2);
  f.add_point_source(Ez, gaussian, vec(0.13, 0.11));
  f.add_point_source(Ez, continuous, vec(-0.17, 0.19));
  f.advance(2);

  const StepPlan live = build_step_plan(f, StepProgram::ordinary);
  const Operation *source_evaluation = NULL;
  for (const Operation &operation : live.operations)
    if (operation.kind == OpKind::evaluate_source_scalars) {
      source_evaluation = &operation;
      break;
    }
  CHECK(source_evaluation, "source-bearing StepPlan has no source evaluation operation");
  if (!source_evaluation) return;
  CHECK(source_evaluation->descriptor_index == 0 && f.descriptors &&
            source_evaluation->descriptor_count == f.descriptors->sources.source_times.size() &&
            source_evaluation->descriptor_count == 2,
        "source evaluation does not use the SourcePlan::source_times descriptor count");

  StepPlan compact;
  compact.source_signature = live.source_signature;
  compact.operations.push_back(*source_evaluation);
  compact.signature = compute_step_plan_signature(compact);
  HaloFixture fixture;
  (void)fixture.authority(compact);

  auto rejected = [&](StepPlan malformed, const char *message) {
    malformed.signature = compute_step_plan_signature(malformed);
    bool did_reject = false;
    try {
      (void)fixture.authority(malformed);
    }
    catch (const std::invalid_argument &) {
      did_reject = true;
    }
    CHECK(did_reject, "%s", message);
  };
  StepPlan malformed = compact;
  malformed.operations[0].descriptor_index = 1;
  rejected(malformed, "nonzero source-time descriptor index was accepted");
  malformed = compact;
  malformed.operations[0].descriptor_count = 0;
  rejected(malformed, "empty source-time descriptor span was accepted");
}

static void test_collective_mode_resolution() {
  std::vector<GraphRankModeSupport> ranks(2);
  ranks[0] = ranks[1] = GraphRankModeSupport{GraphExecutionMode::automatic, true, true};
  CHECK(resolve_collective_graph_execution_mode(ranks).use_graph,
        "collective auto mode did not select graph on full support");
  ranks[1].program_graphable = false;
  CHECK(!resolve_collective_graph_execution_mode(ranks).use_graph,
        "collective auto mode did not fall back all-or-nothing");
  ranks[0].requested = ranks[1].requested = GraphExecutionMode::eager;
  CHECK(!resolve_collective_graph_execution_mode(ranks).use_graph,
        "collective eager mode selected graphs");
  ranks[0].requested = ranks[1].requested = GraphExecutionMode::required;
  bool rejected = false;
  try {
    (void)resolve_collective_graph_execution_mode(ranks);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(rejected, "collective required mode accepted an ungraphable rank");
  ranks[1].requested = GraphExecutionMode::automatic;
  rejected = false;
  try {
    (void)resolve_collective_graph_execution_mode(ranks);
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected, "collective graph mode mismatch was accepted");
}

static void test_malformed_graph_inputs() {
  HaloFixture fixture;
  Operation update = operation(OpKind::update_db, B_stuff);
  update.accesses.push_back(BufferAccess{ArrayRef{fixture.canonical, 0, 16}, AccessMode::read});
  StepPlan base = signed_plan(std::vector<Operation>(1, update));
  auto rejected_plan = [&](StepPlan bad, const char *message) {
    bad.signature = compute_step_plan_signature(bad);
    bool rejected = false;
    try {
      (void)fixture.authority(bad);
    }
    catch (const std::exception &) {
      rejected = true;
    }
    CHECK(rejected, "%s", message);
  };
  StepPlan bad = base;
  bad.operations[0].descriptor_index = UINT32_MAX;
  bad.operations[0].descriptor_count = 1;
  rejected_plan(bad, "overflowing primary descriptor span was accepted");
#define BAD_SPAN(index_field, count_field, message)                                                \
  do {                                                                                             \
    StepPlan changed = base;                                                                       \
    changed.operations[0].index_field = UINT32_MAX;                                                \
    changed.operations[0].count_field = 1;                                                         \
    rejected_plan(changed, message);                                                               \
  } while (0)
  BAD_SPAN(material_refresh_index, material_refresh_count,
           "overflowing material-refresh span was accepted");
  BAD_SPAN(beta_descriptor_index, beta_descriptor_count, "overflowing beta span was accepted");
  BAD_SPAN(cylindrical_m_descriptor_index, cylindrical_m_descriptor_count,
           "overflowing cylindrical-m span was accepted");
  BAD_SPAN(cylindrical_origin_action_index, cylindrical_origin_action_count,
           "overflowing cylindrical-origin span was accepted");
  BAD_SPAN(polarization_group_index, polarization_group_count,
           "overflowing polarization-group span was accepted");
  BAD_SPAN(polarization_subtraction_index, polarization_subtraction_count,
           "overflowing polarization-subtraction span was accepted");
  BAD_SPAN(magnetic_state_index, magnetic_state_count,
           "overflowing magnetic-state span was accepted");
  BAD_SPAN(legacy_flux_index, legacy_flux_count, "overflowing legacy-flux span was accepted");
#undef BAD_SPAN
  bad = base;
  bad.operations[0].source_descriptor_index = UINT32_MAX;
  bad.operations[0].source_descriptor_count = 1;
  rejected_plan(bad, "overflowing source span was accepted");
  bad = base;
  bad.operations[0].accesses[0].array.offset = std::numeric_limits<size_t>::max();
  rejected_plan(bad, "overflowing access range was accepted");
  bad = base;
  bad.operations[0].accesses[0].mode = static_cast<AccessMode>(99);
  rejected_plan(bad, "invalid access mode was accepted");
  bad = base;
  bad.operations[0].guard.kind = static_cast<GuardKind>(99);
  rejected_plan(bad, "invalid guard kind was accepted");
  bad = base;
  bad.operations[0].source_time_offset = std::numeric_limits<double>::infinity();
  rejected_plan(bad, "invalid source time was accepted");

  const GraphLoweringAuthorities authority = fixture.authority(base);
  bool rejected = false;
  try {
    (void)build_graph_program(base, authority, static_cast<GraphVariantKind>(99));
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected, "invalid graph variant was accepted as magnetic restore");

  GraphProgram graph = build_graph_program(base, authority, GraphVariantKind::ordinary);
  graph.schedule[0].kind = static_cast<GraphScheduleKind>(99);
  graph.signature = compute_graph_program_signature(graph);
  std::string error;
  CHECK(!validate_graph_program(base, authority, graph, &error),
        "invalid graph schedule enum was accepted");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;
  test_scalar_layout();
  test_halo_authority_and_boundaries();
  test_host_covered_interval_once();
  test_variants_and_canonical_validation();
  test_all_operation_classes();
  test_formatting_and_stale_signatures();
  test_guard_zero_and_stable_identity();
  test_source_time_descriptor_semantics();
  test_collective_mode_resolution();
  test_malformed_graph_inputs();
  if (failures) {
    master_printf("graph_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("graph_plan: all checks passed\n");
  return 0;
}
