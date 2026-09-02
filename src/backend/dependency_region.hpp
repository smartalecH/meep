/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#ifndef MEEP_BACKEND_DEPENDENCY_REGION_HPP
#define MEEP_BACKEND_DEPENDENCY_REGION_HPP

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "backend/step_plan.hpp"
#include "backend/storage_plan.hpp"
#include "backend/transport_plan.hpp"

namespace meep {

enum class DependencyOperationKind : uint32_t { curl = 0, constitutive = 1 };
enum class DependencyOverlapPolicy : uint32_t { off = 0, automatic = 1, required = 2 };

struct DependencyOverlapPolicyParse {
  bool valid;
  DependencyOverlapPolicy requested;
  std::string error;
};

DependencyOverlapPolicyParse parse_dependency_overlap_policy(const char *value);
const char *dependency_overlap_policy_name(DependencyOverlapPolicy policy);

struct DependencyBox {
  size_t base;
  size_t origin[3];
  size_t counts[3];
  ptrdiff_t strides[3];
};

struct DependencyReadFootprint {
  ArrayId root;
  std::vector<ptrdiff_t> offsets;
};

struct DependencyRegionRow {
  DependencyOperationKind kind;
  uint32_t descriptor_index;
  ArrayId target_root;
  DependencyBox full;
  DependencyBox interior;
  std::vector<DependencyBox> boundary;
  std::vector<DependencyReadFootprint> reads;
  std::vector<DependencyReadFootprint> writes;
};

/* Immutable proof joining one remote halo boundary to the immediately
   following device update. It contains only canonical ArrayIds and signed
   scalar offsets; backend pointers are deliberately excluded. */
struct DependencyRegionPlan {
  uint32_t halo_operation_index;
  uint32_t update_operation_index;
  field_type ft;
  uint64_t step_signature;
  uint64_t storage_signature;
  uint64_t remote_authority_signature;
  std::vector<DependencyRegionRow> rows;
  uint64_t signature;
};

bool build_dependency_region_plan(const StepPlan &step, uint32_t halo_operation_index,
                                  const LoweredRemoteHaloProgram &remote,
                                  const StoragePlan &storage, DependencyRegionPlan &result,
                                  std::string &why);
bool validate_dependency_region_plan(const DependencyRegionPlan &plan, const StepPlan &step,
                                     const LoweredRemoteHaloProgram &remote,
                                     const StoragePlan &storage, std::string &why);
uint64_t compute_dependency_region_signature(const DependencyRegionPlan &plan);

} // namespace meep

#endif
