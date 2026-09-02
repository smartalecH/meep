/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#ifndef MEEP_BACKEND_ADJOINT_PLAN_HPP
#define MEEP_BACKEND_ADJOINT_PLAN_HPP

#include <complex>
#include <memory>
#include <stdint.h>
#include <vector>

#include "backend/array_ref.hpp"
#include "backend/precision.hpp"
#include "meep.hpp"

namespace meep {

enum class AdjointSupportDisposition : uint32_t {
  device_native,
  host_fallback,
  unsupported
};

enum class AdjointExecutionMode : uint32_t { automatic, required, host };

enum AdjointSupportReason : uint64_t {
  adjoint_support_none = 0,
  adjoint_support_missing_snapshot = UINT64_C(1) << 0,
  adjoint_support_stale_snapshot = UINT64_C(1) << 1,
  adjoint_support_remote_term = UINT64_C(1) << 2,
  adjoint_support_overlap_mode = UINT64_C(1) << 3,
  adjoint_support_dispersion = UINT64_C(1) << 4,
  adjoint_support_offdiagonal = UINT64_C(1) << 5,
  adjoint_support_cylindrical = UINT64_C(1) << 6,
  adjoint_support_symmetry = UINT64_C(1) << 7,
  adjoint_support_subpixel = UINT64_C(1) << 8,
  adjoint_support_precision = UINT64_C(1) << 9,
  adjoint_support_layout = UINT64_C(1) << 10
};

struct AdjointDftIdentity {
  int chunk;
  int component;
  int symmetry_index;
  int dimensions;
  int lattice_shift[5];
  int begin[5];
  int end[5];
  int unpadded_begin[5];
  int unpadded_end[5];

  AdjointDftIdentity();
  bool operator==(const AdjointDftIdentity &other) const;
};

struct AdjointDftSnapshotTerm {
  AdjointDftIdentity identity;
  ArrayRef resident;
  Precision storage_precision;
  size_t spatial_points;
  std::vector<double> frequencies;
  std::vector<std::complex<double> > values;
};

class AdjointSnapshotPayload {
public:
  virtual ~AdjointSnapshotPayload() {}
  virtual int device_id() const { return automatic_device; }
};

struct AdjointDftSnapshot {
  static const uint32_t schema_version = 2;

  uint32_t version;
  uint64_t material_signature;
  uint64_t material_layout_signature;
  uint64_t mutation_generation[fields::num_mutation_kinds];
  uint64_t checkpoint_publication_generation;
  uint64_t communicator_generation;
  int communicator_rank;
  int communicator_size;
  std::complex<double> bloch_k[5];
  int boundary_conditions[2][5];
  double cylindrical_m;
  precision_policy_kind precision_policy;
  int device_id;
  uint64_t structural_signature;
  uint64_t value_signature;
  std::vector<AdjointDftSnapshotTerm> terms;
  std::shared_ptr<const AdjointSnapshotPayload> backend_payload;

  AdjointDftSnapshot();
};

struct AdjointDftTerm {
  AdjointDftIdentity forward_identity;
  AdjointDftIdentity adjoint_identity;
  uint32_t forward_snapshot_term;
  uint32_t adjoint_snapshot_term;
  ArrayRef adjoint;
  size_t forward_spatial_points;
  size_t adjoint_spatial_points;
  size_t frequencies;
};

bool adjoint_same_spatial_image(const AdjointDftIdentity &a, const AdjointDftIdentity &b);

struct AdjointGradientContribution {
  size_t output_index;
  uint32_t adjoint_snapshot_term;
  size_t adjoint_value_index;
  uint32_t forward_snapshot_term;
  size_t forward_value_indices[2];
  double forward_weights[2];
  uint32_t forward_value_count;
  std::complex<double> adjoint_coefficient;
  std::complex<double> material_coefficient;
  double accumulation_scale;

  AdjointGradientContribution();
};

struct AdjointGradientRequest {
  static const uint32_t schema_version = 1;

  uint32_t version;
  std::shared_ptr<const AdjointDftSnapshot> forward;
  std::shared_ptr<const AdjointDftSnapshot> adjoint;
  std::shared_ptr<const void> material_recipe;
  uint64_t material_signature;
  uint64_t material_layout_signature;
  uint64_t mutation_generation[fields::num_mutation_kinds];
  size_t grid_shape[3];
  size_t grid_strides[3];
  double transform[12];
  int overlap_mode;
  double beta;
  double eta;
  double damping;
  size_t output_count;
  std::vector<double> frequencies;
  double scale;
  double finite_difference_step;
  bool cylindrical_measure;
  Precision output_precision;
  AdjointSupportDisposition disposition;
  uint64_t support_reasons;
  uint64_t structural_signature;
  std::vector<AdjointDftTerm> terms;
  std::vector<AdjointGradientContribution> contributions;

  AdjointGradientRequest();
};

uint64_t adjoint_snapshot_structural_signature(const AdjointDftSnapshot &snapshot);
uint64_t adjoint_snapshot_value_signature(const AdjointDftSnapshot &snapshot);
uint64_t adjoint_request_signature(const AdjointGradientRequest &request);
void validate_adjoint_snapshot(const AdjointDftSnapshot &snapshot);
void validate_adjoint_request(const AdjointGradientRequest &request);

std::shared_ptr<const AdjointDftSnapshot>
capture_adjoint_dft_snapshot(const std::vector<dft_fields *> &monitors);
std::shared_ptr<const AdjointDftSnapshot>
make_adjoint_dft_snapshot(const std::vector<dft_fields *> &monitors);
const AdjointDftSnapshot *adjoint_snapshot_from(const dft_fields &monitor);
AdjointDftIdentity adjoint_dft_identity(const fields &owner, const dft_chunk &chunk);
void validate_adjoint_snapshot_freshness(const AdjointDftSnapshot &snapshot,
                                         const fields &owner);
void compute_adjoint_gradient_oracle(const AdjointGradientRequest &request, double *local_result,
                                     size_t result_count);
void set_adjoint_failure_after_for_testing(int checkpoints);
void set_nvidia_adjoint_failure_for_testing(const char *point);
void adjoint_failure_checkpoint();

} // namespace meep

#endif
