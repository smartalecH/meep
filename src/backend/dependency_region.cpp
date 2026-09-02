/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/dependency_region.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

namespace meep {
namespace {

uint64_t mix(uint64_t h, uint64_t value) { return remote_authority_hash_u64(h, value); }

bool checked_add(ptrdiff_t a, ptrdiff_t b, ptrdiff_t &result) {
  if ((b > 0 && a > std::numeric_limits<ptrdiff_t>::max() - b) ||
      (b < 0 && a < std::numeric_limits<ptrdiff_t>::min() - b))
    return false;
  result = a + b;
  return true;
}

bool checked_sub(ptrdiff_t a, ptrdiff_t b, ptrdiff_t &result) {
  return !__builtin_sub_overflow(a, b, &result);
}

ArrayId root_of(const StoragePlan &storage, ArrayId id) {
  std::set<uint32_t> seen;
  while (is_valid(id)) {
    if (id.value >= storage.arrays.size() || !seen.insert(id.value).second)
      return invalid_array();
    const ArrayId next = storage.arrays[id.value].alias_of;
    if (!is_valid(next)) return id;
    id = next;
  }
  return invalid_array();
}

DependencyBox box_for(const UpdateRegion &region) {
  DependencyBox box = {};
  box.base = region.base;
  for (int axis = 0; axis < 3; ++axis) {
    box.origin[axis] = 0;
    box.counts[axis] = region.counts[axis];
    box.strides[axis] = region.strides[axis];
  }
  return box;
}

bool locate(const DependencyBox &box, ptrdiff_t index, size_t coordinate[3]) {
  ptrdiff_t minimum = ptrdiff_t(box.base);
  for (int axis = 0; axis < 3; ++axis) {
    if (!box.counts[axis]) return false;
    if (box.strides[axis] == std::numeric_limits<ptrdiff_t>::min()) return false;
    if (box.strides[axis] < 0) {
      ptrdiff_t delta = 0;
      if (box.counts[axis] - 1 > size_t(std::numeric_limits<ptrdiff_t>::max()) ||
          __builtin_mul_overflow(ptrdiff_t(box.counts[axis] - 1), box.strides[axis], &delta) ||
          !checked_add(minimum, delta, minimum))
        return false;
    }
  }
  ptrdiff_t remaining = 0;
  if (!checked_sub(index, minimum, remaining)) return false;
  int order[3] = {0, 1, 2};
  std::sort(order, order + 3, [&](int a, int b) {
    const uint64_t sa = box.strides[a] < 0 ? uint64_t(-(box.strides[a] + 1)) + 1
                                           : uint64_t(box.strides[a]);
    const uint64_t sb = box.strides[b] < 0 ? uint64_t(-(box.strides[b] + 1)) + 1
                                           : uint64_t(box.strides[b]);
    return sa > sb;
  });
  size_t normalized[3] = {};
  for (int oi = 0; oi < 3; ++oi) {
    const int axis = order[oi];
    const ptrdiff_t stride = box.strides[axis] < 0 ? -box.strides[axis] : box.strides[axis];
    if (!stride) {
      if (box.counts[axis] != 1) return false;
      normalized[axis] = 0;
      continue;
    }
    if (remaining < 0) return false;
    normalized[axis] = size_t(remaining / stride);
    if (normalized[axis] >= box.counts[axis]) return false;
    remaining %= stride;
  }
  if (remaining != 0) return false;
  ptrdiff_t reconstructed = ptrdiff_t(box.base);
  for (int axis = 0; axis < 3; ++axis) {
    coordinate[axis] = box.strides[axis] < 0 ? box.counts[axis] - 1 - normalized[axis]
                                             : normalized[axis];
    ptrdiff_t delta = 0;
    if (__builtin_mul_overflow(ptrdiff_t(coordinate[axis]), box.strides[axis], &delta) ||
        !checked_add(reconstructed, delta, reconstructed))
      return false;
  }
  return reconstructed == index;
}

bool contains(const DependencyBox &box, ptrdiff_t index) {
  size_t coordinate[3];
  return locate(box, index, coordinate);
}

bool shift_base(DependencyBox &box, int axis, size_t amount) {
  ptrdiff_t delta = 0, shifted = 0;
  if (amount > size_t(std::numeric_limits<ptrdiff_t>::max()) ||
      __builtin_mul_overflow(ptrdiff_t(amount), box.strides[axis], &delta) ||
      !checked_add(ptrdiff_t(box.base), delta, shifted) || shifted < 0)
    return false;
  box.base = size_t(shifted);
  if (amount > std::numeric_limits<size_t>::max() - box.origin[axis]) return false;
  box.origin[axis] += amount;
  return true;
}

void append_read(std::vector<DependencyReadFootprint> &reads, const StoragePlan &storage,
                 ArrayId id, std::initializer_list<ptrdiff_t> offsets, std::string &why) {
  if (!is_valid(id)) return;
  const ArrayId root = root_of(storage, id);
  if (!is_valid(root)) {
    why = "dependency region read has an invalid or cyclic storage alias";
    return;
  }
  for (DependencyReadFootprint &read : reads)
    if (read.root == root) {
      read.offsets.insert(read.offsets.end(), offsets.begin(), offsets.end());
      std::sort(read.offsets.begin(), read.offsets.end());
      read.offsets.erase(std::unique(read.offsets.begin(), read.offsets.end()),
                         read.offsets.end());
      return;
    }
  DependencyReadFootprint read;
  read.root = root;
  read.offsets.assign(offsets.begin(), offsets.end());
  std::sort(read.offsets.begin(), read.offsets.end());
  read.offsets.erase(std::unique(read.offsets.begin(), read.offsets.end()), read.offsets.end());
  reads.push_back(read);
}

bool split_row(DependencyRegionRow &row, const std::vector<RemoteHaloScalarRef> &received,
               std::string &why) {
  bool low[3] = {}, high[3] = {};
  struct unsafe_point { size_t coordinate[3]; };
  std::vector<unsafe_point> unsafe;
  std::vector<DependencyReadFootprint> dependencies = row.reads;
  dependencies.insert(dependencies.end(), row.writes.begin(), row.writes.end());
  for (const DependencyReadFootprint &read : dependencies)
    for (const RemoteHaloScalarRef &remote : received) {
      if (remote.root != read.root) continue;
      for (ptrdiff_t offset : read.offsets) {
        ptrdiff_t target = 0;
        if (!checked_sub(remote.index, offset, target)) {
          why = "dependency region read offset overflows";
          return false;
        }
        unsafe_point point;
        if (!locate(row.full, target, point.coordinate)) continue;
        int selected_axis = -1;
        for (int axis = 0; axis < 3; ++axis)
          if (row.full.counts[axis] > 1 && offset != 0 &&
              (offset == row.full.strides[axis] || offset == -row.full.strides[axis]) &&
              (point.coordinate[axis] == 0 ||
               point.coordinate[axis] + 1 == row.full.counts[axis])) {
            selected_axis = axis;
            break;
          }
        for (int axis = 0; selected_axis < 0 && axis < 3; ++axis)
          if (row.full.counts[axis] > 1 &&
              (point.coordinate[axis] == 0 ||
               point.coordinate[axis] + 1 == row.full.counts[axis]))
            selected_axis = axis;
        if (selected_axis < 0) {
          why = "receive-owned dependency lies in the update interior";
          return false;
        }
        if (point.coordinate[selected_axis] == 0)
          low[selected_axis] = true;
        else
          high[selected_axis] = true;
        unsafe.push_back(point);
      }
    }

  row.interior = row.full;
  size_t lower[3] = {}, upper[3] = {};
  for (int axis = 0; axis < 3; ++axis) {
    lower[axis] = low[axis] ? 1 : 0;
    upper[axis] = high[axis] ? 1 : 0;
    if (lower[axis] + upper[axis] >= row.full.counts[axis]) {
      why = "dependency region has no nonempty safe interior";
      return false;
    }
    if (lower[axis] && !shift_base(row.interior, axis, 1)) {
      why = "dependency region interior base overflows";
      return false;
    }
    row.interior.counts[axis] -= lower[axis] + upper[axis];
  }
  for (const DependencyReadFootprint &read : dependencies)
    for (const RemoteHaloScalarRef &remote : received) {
      if (remote.root != read.root) continue;
      for (ptrdiff_t offset : read.offsets) {
        ptrdiff_t target = 0;
        if (!checked_sub(remote.index, offset, target) || contains(row.interior, target)) {
          why = "dependency region interior still reads receive-owned storage";
          return false;
        }
      }
    }

  DependencyBox remainder = row.full;
  for (int axis = 0; axis < 3; ++axis) {
    if (lower[axis]) {
      DependencyBox slab = remainder;
      slab.counts[axis] = 1;
      row.boundary.push_back(slab);
      if (!shift_base(remainder, axis, 1)) {
        why = "dependency region lower slab overflows";
        return false;
      }
      --remainder.counts[axis];
    }
    if (upper[axis]) {
      DependencyBox slab = remainder;
      if (!shift_base(slab, axis, remainder.counts[axis] - 1)) {
        why = "dependency region upper slab overflows";
        return false;
      }
      slab.counts[axis] = 1;
      row.boundary.push_back(slab);
      --remainder.counts[axis];
    }
  }
  return true;
}

} // namespace

DependencyOverlapPolicyParse parse_dependency_overlap_policy(const char *value) {
  if (!value || !*value || !std::strcmp(value, "auto"))
    return DependencyOverlapPolicyParse{true, DependencyOverlapPolicy::automatic, std::string()};
  if (!std::strcmp(value, "off") || !std::strcmp(value, "no") || !std::strcmp(value, "0"))
    return DependencyOverlapPolicyParse{true, DependencyOverlapPolicy::off, std::string()};
  if (!std::strcmp(value, "required") || !std::strcmp(value, "yes") ||
      !std::strcmp(value, "1"))
    return DependencyOverlapPolicyParse{true, DependencyOverlapPolicy::required, std::string()};
  return DependencyOverlapPolicyParse{false, DependencyOverlapPolicy::automatic,
                                      std::string("invalid MEEP_NVIDIA_MPI_OVERLAP value: ") +
                                          value};
}

const char *dependency_overlap_policy_name(DependencyOverlapPolicy policy) {
  switch (policy) {
    case DependencyOverlapPolicy::off: return "off";
    case DependencyOverlapPolicy::automatic: return "auto";
    case DependencyOverlapPolicy::required: return "required";
  }
  return "invalid";
}

uint64_t compute_dependency_region_signature(const DependencyRegionPlan &plan) {
  uint64_t h = UINT64_C(1469598103934665603);
  h = mix(h, plan.halo_operation_index);
  h = mix(h, plan.update_operation_index);
  h = mix(h, uint64_t(int(plan.ft)));
  h = mix(h, plan.step_signature);
  h = mix(h, plan.storage_signature);
  h = mix(h, plan.remote_authority_signature);
  h = mix(h, plan.rows.size());
  for (const DependencyRegionRow &row : plan.rows) {
    h = mix(h, uint64_t(row.kind));
    h = mix(h, row.descriptor_index);
    h = mix(h, row.target_root.value);
    const DependencyBox *boxes[2] = {&row.full, &row.interior};
    for (const DependencyBox *box : boxes) {
      h = mix(h, box->base);
      for (int axis = 0; axis < 3; ++axis) {
        h = mix(h, box->origin[axis]);
        h = mix(h, box->counts[axis]);
        h = mix(h, uint64_t(int64_t(box->strides[axis])));
      }
    }
    h = mix(h, row.boundary.size());
    for (const DependencyBox &box : row.boundary) {
      h = mix(h, box.base);
      for (int axis = 0; axis < 3; ++axis) {
        h = mix(h, box.origin[axis]);
        h = mix(h, box.counts[axis]);
        h = mix(h, uint64_t(int64_t(box.strides[axis])));
      }
    }
    h = mix(h, row.reads.size());
    for (const DependencyReadFootprint &read : row.reads) {
      h = mix(h, read.root.value);
      h = mix(h, read.offsets.size());
      for (ptrdiff_t offset : read.offsets) h = mix(h, uint64_t(int64_t(offset)));
    }
    h = mix(h, row.writes.size());
    for (const DependencyReadFootprint &write : row.writes) {
      h = mix(h, write.root.value);
      h = mix(h, write.offsets.size());
      for (ptrdiff_t offset : write.offsets) h = mix(h, uint64_t(int64_t(offset)));
    }
  }
  return h;
}

bool build_dependency_region_plan(const StepPlan &step, uint32_t halo_operation_index,
                                  const LoweredRemoteHaloProgram &remote,
                                  const StoragePlan &storage, DependencyRegionPlan &result,
                                  std::string &why) {
  why.clear();
  DependencyRegionPlan staged = {};
  if (step.signature != compute_step_plan_signature(step)) {
    why = "dependency region StepPlan signature is stale";
    return false;
  }
  if (remote.storage_signature != compute_remote_storage_authority_signature(storage) ||
      remote.authority_signature != compute_remote_lowered_authority_signature(
                                        remote.program_signature, remote.storage_signature)) {
    why = "dependency region remote/storage authority is stale";
    return false;
  }
  if (halo_operation_index >= step.operations.size() ||
      step.operations[halo_operation_index].kind != OpKind::transfer_halo ||
      halo_operation_index + 1 >= step.operations.size()) {
    why = "dependency region lacks a canonical halo/update pair";
    return false;
  }
  const Operation &halo = step.operations[halo_operation_index];
  const Operation &update = step.operations[halo_operation_index + 1];
  const LoweredRemoteHaloStage *stage = NULL;
  for (const LoweredRemoteHaloStage &candidate : remote.stages)
    if (candidate.ft == halo.ft) stage = &candidate;
  if (!stage || (stage->receives.empty() && stage->sends.empty())) {
    why = "dependency region halo has no remote message authority";
    return false;
  }
  if (update.kind != OpKind::update_db && update.kind != OpKind::update_eh) {
    why = "dependency region successor is not curl or constitutive";
    return false;
  }
  if (update.kind == OpKind::update_db &&
      (update.beta_descriptor_count || update.cylindrical_m_descriptor_count ||
       update.cylindrical_origin_action_count)) {
    why = "dependency region curl has inseparable auxiliary updates";
    return false;
  }
  if (update.kind == OpKind::update_eh &&
      (update.source_descriptor_count || update.polarization_subtraction_count)) {
    why = "dependency region constitutive update has source or polarization work";
    return false;
  }
  std::vector<RemoteHaloScalarRef> received;
  for (const LoweredRemoteHaloMessage &message : stage->receives)
    for (const RemoteHaloScatterDescriptor &scatter : message.scatters) {
      received.push_back(scatter.target_real);
      if (is_valid(scatter.target_imag.root)) received.push_back(scatter.target_imag);
    }
  staged.halo_operation_index = halo_operation_index;
  staged.update_operation_index = halo_operation_index + 1;
  staged.ft = halo.ft;
  staged.step_signature = step.signature;
  staged.storage_signature = remote.storage_signature;
  staged.remote_authority_signature = remote.authority_signature;

  if (update.kind == OpKind::update_db) {
    if (size_t(update.descriptor_index) + update.descriptor_count > step.db_updates.size()) {
      why = "dependency region curl span is out of range";
      return false;
    }
    for (size_t i = update.descriptor_index;
         i < size_t(update.descriptor_index) + update.descriptor_count; ++i) {
      const CurlUpdate &source = step.db_updates[i];
      if (source.radial_prefix_index != UINT32_MAX || source.bfast_update_index != UINT32_MAX) {
        why = "dependency region curl has an inseparable prefix or BFAST update";
        return false;
      }
      DependencyRegionRow row = {};
      row.kind = DependencyOperationKind::curl;
      row.descriptor_index = uint32_t(i);
      row.target_root = root_of(storage, source.target);
      row.full = box_for(source.region);
      append_read(row.writes, storage, source.target, {0}, why);
      append_read(row.reads, storage, source.target, {0}, why);
      append_read(row.reads, storage, source.plus_source, {0, source.plus_stride}, why);
      append_read(row.reads, storage, source.minus_source, {0, source.minus_stride}, why);
      append_read(row.writes, storage, source.target_u, {0}, why);
      append_read(row.reads, storage, source.target_u, {0}, why);
      append_read(row.writes, storage, source.target_cond, {0}, why);
      append_read(row.reads, storage, source.target_cond, {0}, why);
      if (!why.empty() || !is_valid(row.target_root) || !split_row(row, received, why)) {
        if (why.empty()) why = "dependency region curl target is invalid";
        return false;
      }
      staged.rows.push_back(row);
    }
  }
  else {
    if (size_t(update.descriptor_index) + update.descriptor_count > step.eh_updates.size()) {
      why = "dependency region constitutive span is out of range";
      return false;
    }
    for (size_t i = update.descriptor_index;
         i < size_t(update.descriptor_index) + update.descriptor_count; ++i) {
      const ConstitutiveUpdate &source = step.eh_updates[i];
      if (source.region.variant_key & (constitutive_axis_override | constitutive_copy_w_previous)) {
        why = "dependency region constitutive row has an inseparable replay or copy";
        return false;
      }
      DependencyRegionRow row = {};
      row.kind = DependencyOperationKind::constitutive;
      row.descriptor_index = uint32_t(i);
      row.target_root = root_of(storage, source.target);
      row.full = box_for(source.region);
      append_read(row.writes, storage, source.target, {0}, why);
      if (source.region.variant_key & constitutive_has_pml)
        append_read(row.reads, storage, source.target, {0}, why);
      append_read(row.reads, storage, source.primary, {0}, why);
      if (is_valid(source.cross1))
        append_read(row.reads, storage, source.cross1,
                    {0, source.primary_stride, -source.cross1_stride,
                     source.primary_stride - source.cross1_stride}, why);
      if (is_valid(source.cross2))
        append_read(row.reads, storage, source.cross2,
                    {0, source.primary_stride, -source.cross2_stride,
                     source.primary_stride - source.cross2_stride}, why);
      append_read(row.writes, storage, source.target_w, {0}, why);
      if (!why.empty() || !is_valid(row.target_root) || !split_row(row, received, why)) {
        if (why.empty()) why = "dependency region constitutive target is invalid";
        return false;
      }
      staged.rows.push_back(row);
    }
  }
  if (staged.rows.empty()) {
    why = "dependency region update has no descriptors";
    return false;
  }
  staged.signature = compute_dependency_region_signature(staged);
  result = staged;
  return true;
}

bool validate_dependency_region_plan(const DependencyRegionPlan &plan, const StepPlan &step,
                                     const LoweredRemoteHaloProgram &remote,
                                     const StoragePlan &storage, std::string &why) {
  DependencyRegionPlan canonical;
  if (!build_dependency_region_plan(step, plan.halo_operation_index, remote, storage, canonical,
                                    why))
    return false;
  if (plan.signature != compute_dependency_region_signature(plan) ||
      plan.signature != canonical.signature) {
    why = "dependency region plan is stale or non-canonical";
    return false;
  }
  return true;
}

} // namespace meep
