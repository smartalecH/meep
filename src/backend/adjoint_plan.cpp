/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/adjoint_plan.hpp"

#include "config.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "backend/backend.hpp"
#include "backend/material_ir.hpp"
#ifdef HAVE_NVIDIA_BACKEND
#include "backend/nvidia/runtime.hpp"
#endif

namespace meep {
namespace {

int adjoint_failure_after = -1;

void mix(uint64_t &hash, uint64_t value) {
  hash ^= value;
  hash *= UINT64_C(1099511628211);
}

void mix_double(uint64_t &hash, double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double hashing requires binary64");
  std::memcpy(&bits, &value, sizeof(bits));
  mix(hash, bits);
}

void mix_identity(uint64_t &hash, const AdjointDftIdentity &identity) {
  mix(hash, uint64_t(uint32_t(identity.chunk)));
  mix(hash, uint64_t(uint32_t(identity.component)));
  mix(hash, uint64_t(uint32_t(identity.symmetry_index)));
  mix(hash, uint64_t(uint32_t(identity.dimensions)));
  for (int d = 0; d < 5; ++d) {
    mix(hash, uint64_t(uint32_t(identity.lattice_shift[d])));
    mix(hash, uint64_t(uint32_t(identity.begin[d])));
    mix(hash, uint64_t(uint32_t(identity.end[d])));
    mix(hash, uint64_t(uint32_t(identity.unpadded_begin[d])));
    mix(hash, uint64_t(uint32_t(identity.unpadded_end[d])));
  }
}

void mix_ref(uint64_t &hash, const ArrayRef &ref) {
  mix(hash, ref.id.value);
  mix(hash, ref.offset);
  mix(hash, ref.elements);
}

void copy_ivec(int destination[5], const ivec &source) {
  for (int d = 0; d < 5; ++d) {
    const direction axis = direction(d);
    destination[d] = has_direction(source.dim, axis) ? source.in_direction(axis) : 0;
  }
}

int chunk_index(const fields &owner, const fields_chunk *chunk) {
  for (int i = 0; i < owner.num_chunks; ++i)
    if (owner.chunks[i] == chunk) return i;
  return -1;
}

AdjointDftIdentity identity_for(const fields &owner, const dft_chunk &chunk) {
  AdjointDftIdentity result;
  result.chunk = chunk_index(owner, chunk.fc);
  if (result.chunk < 0)
    throw std::invalid_argument("adjoint DFT chunk is not owned by the current fields layout");
  result.component = int(chunk.c);
  result.symmetry_index = chunk.sn;
  result.dimensions = int(chunk.is.dim);
  copy_ivec(result.lattice_shift, chunk.shift);
  copy_ivec(result.begin, chunk.is);
  copy_ivec(result.end, chunk.ie);
  copy_ivec(result.unpadded_begin, chunk.persist ? chunk.is_old : chunk.is);
  copy_ivec(result.unpadded_end, chunk.persist ? chunk.ie_old : chunk.ie);
  return result;
}

bool identity_less(const AdjointDftIdentity &a, const AdjointDftIdentity &b) {
  if (a.chunk != b.chunk) return a.chunk < b.chunk;
  if (a.component != b.component) return a.component < b.component;
  if (a.symmetry_index != b.symmetry_index) return a.symmetry_index < b.symmetry_index;
  if (a.dimensions != b.dimensions) return a.dimensions < b.dimensions;
  for (int d = 0; d < 5; ++d) {
    if (a.lattice_shift[d] != b.lattice_shift[d])
      return a.lattice_shift[d] < b.lattice_shift[d];
    if (a.begin[d] != b.begin[d]) return a.begin[d] < b.begin[d];
    if (a.end[d] != b.end[d]) return a.end[d] < b.end[d];
    if (a.unpadded_begin[d] != b.unpadded_begin[d])
      return a.unpadded_begin[d] < b.unpadded_begin[d];
    if (a.unpadded_end[d] != b.unpadded_end[d])
      return a.unpadded_end[d] < b.unpadded_end[d];
  }
  return false;
}

void validate_identity(const AdjointDftIdentity &identity) {
  if (identity.chunk < 0 || identity.component < 0 || identity.component >= NUM_FIELD_COMPONENTS ||
      identity.symmetry_index < 0 || identity.dimensions < int(D1) ||
      identity.dimensions > int(Dcyl))
    throw std::invalid_argument("adjoint DFT identity is invalid");
  const ndim dim = ndim(identity.dimensions);
  LOOP_OVER_DIRECTIONS(dim, d) {
    const int axis = int(d);
    if (identity.begin[axis] > identity.end[axis] ||
        identity.unpadded_begin[axis] > identity.unpadded_end[axis] ||
        identity.unpadded_begin[axis] < identity.begin[axis] ||
        identity.unpadded_end[axis] > identity.end[axis])
      throw std::invalid_argument("adjoint DFT identity has invalid bounds");
  }
}

size_t checked_product(size_t left, size_t right, const char *what) {
  if (left && right > std::numeric_limits<size_t>::max() / left)
    throw std::overflow_error(std::string(what) + " overflows");
  return left * right;
}

size_t checked_add(size_t left, size_t right, const char *what) {
  if (right > std::numeric_limits<size_t>::max() - left)
    throw std::overflow_error(std::string(what) + " overflows");
  return left + right;
}

template <typename T>
void validate_vector_extent(size_t count, const char *what) {
  if (count > std::vector<T>().max_size())
    throw std::length_error(std::string(what) + " exceeds vector capacity");
  checked_product(count, sizeof(T), what);
}

bool same_request_frequencies(const AdjointDftSnapshotTerm &term,
                              const std::vector<double> &frequencies) {
  if (term.frequencies.size() != frequencies.size()) return false;
  for (size_t i = 0; i < frequencies.size(); ++i)
    if (term.frequencies[i] != 2.0 * pi * frequencies[i]) return false;
  return true;
}

bool same_boundary_identity(const AdjointDftSnapshot &snapshot, const fields &owner) {
  for (int d = 0; d < 5; ++d) {
    if (snapshot.bloch_k[d] != owner.k[d]) return false;
    for (int side = 0; side < 2; ++side)
      if (snapshot.boundary_conditions[side][d] != int(owner.boundaries[side][d])) return false;
  }
  return true;
}

} // namespace

AdjointDftIdentity::AdjointDftIdentity()
    : chunk(-1), component(-1), symmetry_index(-1), dimensions(-1) {
  std::fill(lattice_shift, lattice_shift + 5, 0);
  std::fill(begin, begin + 5, 0);
  std::fill(end, end + 5, 0);
  std::fill(unpadded_begin, unpadded_begin + 5, 0);
  std::fill(unpadded_end, unpadded_end + 5, 0);
}

bool AdjointDftIdentity::operator==(const AdjointDftIdentity &other) const {
  return chunk == other.chunk && component == other.component &&
         symmetry_index == other.symmetry_index && dimensions == other.dimensions &&
         std::equal(lattice_shift, lattice_shift + 5, other.lattice_shift) &&
         std::equal(begin, begin + 5, other.begin) && std::equal(end, end + 5, other.end) &&
         std::equal(unpadded_begin, unpadded_begin + 5, other.unpadded_begin) &&
         std::equal(unpadded_end, unpadded_end + 5, other.unpadded_end);
}

bool adjoint_same_spatial_image(const AdjointDftIdentity &a, const AdjointDftIdentity &b) {
  return a.chunk == b.chunk && a.symmetry_index == b.symmetry_index &&
         std::equal(a.lattice_shift, a.lattice_shift + 5, b.lattice_shift);
}

AdjointDftSnapshot::AdjointDftSnapshot()
    : version(schema_version), material_signature(0), material_layout_signature(0),
      checkpoint_publication_generation(0), cylindrical_m(0.0),
      precision_policy(precision_policy_kind::native), device_id(automatic_device),
      structural_signature(0), value_signature(0) {
  std::fill(mutation_generation, mutation_generation + fields::num_mutation_kinds, 0);
  std::fill(bloch_k, bloch_k + 5, std::complex<double>(0.0, 0.0));
  for (int side = 0; side < 2; ++side)
    std::fill(boundary_conditions[side], boundary_conditions[side] + 5, int(None));
}

AdjointGradientContribution::AdjointGradientContribution()
    : output_index(0), adjoint_snapshot_term(0), adjoint_value_index(0),
      forward_snapshot_term(0), forward_value_count(0), adjoint_coefficient(1.0, 0.0),
      material_coefficient(0.0, 0.0) {
  forward_value_indices[0] = forward_value_indices[1] = 0;
  forward_weights[0] = forward_weights[1] = 0.0;
  accumulation_scale = 0.0;
}

AdjointGradientRequest::AdjointGradientRequest()
    : version(schema_version), material_signature(0), material_layout_signature(0), overlap_mode(-1),
      beta(0.0), eta(0.0), damping(0.0), output_count(0), scale(1.0),
      finite_difference_step(0.0), cylindrical_measure(false), output_precision(Precision::f64),
      disposition(AdjointSupportDisposition::unsupported), support_reasons(adjoint_support_none),
      structural_signature(0) {
  std::fill(mutation_generation, mutation_generation + fields::num_mutation_kinds, 0);
  std::fill(grid_shape, grid_shape + 3, 0);
  std::fill(grid_strides, grid_strides + 3, 0);
  std::fill(transform, transform + 12, 0.0);
}

uint64_t adjoint_snapshot_structural_signature(const AdjointDftSnapshot &snapshot) {
  uint64_t hash = UINT64_C(1469598103934665603);
  mix(hash, snapshot.version);
  mix(hash, snapshot.material_signature);
  mix(hash, snapshot.material_layout_signature);
  mix(hash, uint64_t(snapshot.precision_policy));
  mix(hash, uint64_t(uint32_t(snapshot.device_id)));
  mix(hash, snapshot.checkpoint_publication_generation);
  for (int d = 0; d < 5; ++d) {
    mix_double(hash, snapshot.bloch_k[d].real());
    mix_double(hash, snapshot.bloch_k[d].imag());
    for (int side = 0; side < 2; ++side)
      mix(hash, uint64_t(uint32_t(snapshot.boundary_conditions[side][d])));
  }
  mix_double(hash, snapshot.cylindrical_m);
  mix(hash, snapshot.terms.size());
  for (const AdjointDftSnapshotTerm &term : snapshot.terms) {
    mix_identity(hash, term.identity);
    mix_ref(hash, term.resident);
    mix(hash, uint64_t(term.storage_precision));
    mix(hash, term.spatial_points);
    mix(hash, term.frequencies.size());
    for (double frequency : term.frequencies) mix_double(hash, frequency);
  }
  return hash;
}

uint64_t adjoint_snapshot_value_signature(const AdjointDftSnapshot &snapshot) {
  uint64_t hash = adjoint_snapshot_structural_signature(snapshot);
  for (const AdjointDftSnapshotTerm &term : snapshot.terms)
    for (const std::complex<double> value : term.values) {
      mix_double(hash, value.real());
      mix_double(hash, value.imag());
    }
  return hash;
}

uint64_t adjoint_request_signature(const AdjointGradientRequest &request) {
  uint64_t hash = UINT64_C(1469598103934665603);
  mix(hash, request.version);
  mix(hash, request.forward ? request.forward->structural_signature : 0);
  mix(hash, request.adjoint ? request.adjoint->structural_signature : 0);
  mix(hash, request.material_signature);
  mix(hash, request.material_layout_signature);
  for (int i = 0; i < fields::num_mutation_kinds; ++i) mix(hash, request.mutation_generation[i]);
  for (int axis = 0; axis < 3; ++axis) {
    mix(hash, request.grid_shape[axis]);
    mix(hash, request.grid_strides[axis]);
  }
  for (double value : request.transform) mix_double(hash, value);
  mix(hash, uint64_t(uint32_t(request.overlap_mode)));
  mix_double(hash, request.beta);
  mix_double(hash, request.eta);
  mix_double(hash, request.damping);
  mix(hash, request.output_count);
  mix(hash, request.frequencies.size());
  for (double frequency : request.frequencies) mix_double(hash, frequency);
  mix_double(hash, request.scale);
  mix_double(hash, request.finite_difference_step);
  mix(hash, request.cylindrical_measure ? 1 : 0);
  mix(hash, uint64_t(request.output_precision));
  mix(hash, uint64_t(request.disposition));
  mix(hash, request.support_reasons);
  mix(hash, request.terms.size());
  for (const AdjointDftTerm &term : request.terms) {
    mix_identity(hash, term.forward_identity);
    mix_identity(hash, term.adjoint_identity);
    mix(hash, term.forward_snapshot_term);
    mix(hash, term.adjoint_snapshot_term);
    mix_ref(hash, term.adjoint);
    mix(hash, term.forward_spatial_points);
    mix(hash, term.adjoint_spatial_points);
    mix(hash, term.frequencies);
  }
  mix(hash, request.contributions.size());
  for (const AdjointGradientContribution &contribution : request.contributions) {
    mix(hash, contribution.output_index);
    mix(hash, contribution.adjoint_snapshot_term);
    mix(hash, contribution.adjoint_value_index);
    mix(hash, contribution.forward_snapshot_term);
    mix(hash, contribution.forward_value_count);
    for (int i = 0; i < 2; ++i) {
      mix(hash, contribution.forward_value_indices[i]);
      mix_double(hash, contribution.forward_weights[i]);
    }
    mix_double(hash, contribution.adjoint_coefficient.real());
    mix_double(hash, contribution.adjoint_coefficient.imag());
    mix_double(hash, contribution.material_coefficient.real());
    mix_double(hash, contribution.material_coefficient.imag());
    mix_double(hash, contribution.accumulation_scale);
  }
  return hash;
}

void validate_adjoint_snapshot(const AdjointDftSnapshot &snapshot) {
  if (snapshot.version != AdjointDftSnapshot::schema_version)
    throw std::invalid_argument("adjoint DFT snapshot version is unsupported");
  if (!snapshot.material_signature || !snapshot.material_layout_signature)
    throw std::invalid_argument("adjoint DFT snapshot has no material identity");
  if (!std::isfinite(snapshot.cylindrical_m))
    throw std::invalid_argument("adjoint DFT snapshot cylindrical coordinate is not finite");
  for (int d = 0; d < 5; ++d) {
    if (!std::isfinite(snapshot.bloch_k[d].real()) ||
        !std::isfinite(snapshot.bloch_k[d].imag()))
      throw std::invalid_argument("adjoint DFT snapshot Bloch coordinate is not finite");
    for (int side = 0; side < 2; ++side)
      if (snapshot.boundary_conditions[side][d] < int(Periodic) ||
          snapshot.boundary_conditions[side][d] > int(None))
        throw std::invalid_argument("adjoint DFT snapshot boundary condition is invalid");
  }
  const Precision expected_monitor = policy_for(snapshot.precision_policy).monitor;
  if (snapshot.terms.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("adjoint DFT snapshot term index overflows");
  for (size_t i = 0; i < snapshot.terms.size(); ++i) {
    const AdjointDftSnapshotTerm &term = snapshot.terms[i];
    validate_identity(term.identity);
    if (!term.spatial_points || term.frequencies.empty())
      throw std::invalid_argument("adjoint DFT snapshot term is empty");
    if (term.storage_precision != expected_monitor)
      throw std::invalid_argument("adjoint DFT snapshot monitor precision violates policy");
    const size_t values = checked_product(term.spatial_points, term.frequencies.size(),
                                          "adjoint DFT snapshot value count");
    validate_vector_extent<double>(term.frequencies.size(),
                                   "adjoint DFT snapshot frequency storage");
    validate_vector_extent<std::complex<double> >(values,
                                                  "adjoint DFT snapshot value storage");
    if (term.values.size() != values)
      throw std::invalid_argument("adjoint DFT snapshot term value count is invalid");
    if (is_valid(term.resident.id) && term.resident.elements != values)
      throw std::invalid_argument("adjoint DFT snapshot resident range is invalid");
    if (is_valid(term.resident.id))
      checked_add(term.resident.offset, term.resident.elements,
                  "adjoint DFT snapshot resident range");
    for (double frequency : term.frequencies)
      if (!std::isfinite(frequency))
        throw std::invalid_argument("adjoint DFT snapshot frequency is not finite");
    for (size_t j = 0; j < i; ++j)
      if (snapshot.terms[j].identity == term.identity)
        throw std::invalid_argument("adjoint DFT snapshot identity is duplicated");
  }
  if (snapshot.structural_signature != adjoint_snapshot_structural_signature(snapshot) ||
      snapshot.value_signature != adjoint_snapshot_value_signature(snapshot))
    throw std::invalid_argument("adjoint DFT snapshot signature mismatch");
}

void validate_adjoint_request(const AdjointGradientRequest &request) {
  if (request.version != AdjointGradientRequest::schema_version)
    throw std::invalid_argument("adjoint gradient request version is unsupported");
  if (!request.forward)
    throw std::invalid_argument("adjoint gradient request has no forward snapshot");
  validate_adjoint_snapshot(*request.forward);
  if (!request.adjoint)
    throw std::invalid_argument("adjoint gradient request has no adjoint snapshot");
  validate_adjoint_snapshot(*request.adjoint);
  if (request.forward->precision_policy != request.adjoint->precision_policy)
    throw std::invalid_argument("adjoint snapshots use different precision policies");
  if (!request.material_recipe || !request.material_signature || !request.material_layout_signature)
    throw std::invalid_argument("adjoint gradient request has no material identity");
  if (request.material_signature != request.forward->material_signature ||
      request.material_signature != request.adjoint->material_signature ||
      request.material_layout_signature != request.forward->material_layout_signature ||
      request.material_layout_signature != request.adjoint->material_layout_signature)
    throw std::invalid_argument("adjoint gradient request material identity is inconsistent");
  if (!request.output_count || request.frequencies.empty() ||
      request.output_count % request.frequencies.size())
    throw std::invalid_argument("adjoint gradient request output shape is invalid");
  const size_t offset_count = checked_add(request.output_count, size_t(1),
                                          "adjoint gradient offset count");
  validate_vector_extent<double>(request.output_count, "adjoint gradient result storage");
  validate_vector_extent<uint64_t>(offset_count, "adjoint gradient offset storage");
  validate_vector_extent<AdjointGradientContribution>(request.contributions.size(),
                                                      "adjoint gradient contribution storage");
  if (request.terms.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("adjoint gradient DFT pair index overflows");
  if (!(request.finite_difference_step > 0.0) ||
      !std::isfinite(request.finite_difference_step) || !std::isfinite(request.scale))
    throw std::invalid_argument("adjoint gradient request scalar is invalid");
  if (request.output_precision != Precision::f64)
    throw std::invalid_argument("adjoint gradient output must be f64");
  size_t grid_count = 1;
  for (int axis = 0; axis < 3; ++axis) {
    if (!request.grid_shape[axis])
      throw std::invalid_argument("adjoint gradient grid extent is empty");
    grid_count = checked_product(grid_count, request.grid_shape[axis],
                                 "adjoint gradient grid size");
  }
  if (request.output_count != checked_product(grid_count, request.frequencies.size(),
                                              "adjoint gradient output count"))
    throw std::invalid_argument("adjoint gradient output count does not match its grid");
  const size_t expected_stride_2 = 1;
  const size_t expected_stride_1 = request.grid_shape[2];
  const size_t expected_stride_0 = checked_product(request.grid_shape[1], request.grid_shape[2],
                                                   "adjoint gradient grid stride");
  if (request.grid_strides[0] != expected_stride_0 ||
      request.grid_strides[1] != expected_stride_1 ||
      request.grid_strides[2] != expected_stride_2)
    throw std::invalid_argument("adjoint gradient grid strides are invalid");
  for (double frequency : request.frequencies)
    if (!std::isfinite(frequency))
      throw std::invalid_argument("adjoint gradient frequency is not finite");
  size_t forward_values = 0;
  for (const AdjointDftSnapshotTerm &term : request.forward->terms)
    forward_values = checked_add(forward_values, term.values.size(),
                                 "adjoint forward flattened value count");
  size_t adjoint_values = 0;
  for (const AdjointDftSnapshotTerm &term : request.adjoint->terms)
    adjoint_values = checked_add(adjoint_values, term.values.size(),
                                 "adjoint reverse flattened value count");
  validate_vector_extent<std::complex<double> >(forward_values,
                                                "adjoint forward flattened storage");
  validate_vector_extent<std::complex<double> >(adjoint_values,
                                                "adjoint reverse flattened storage");
  for (size_t term_index = 0; term_index < request.terms.size(); ++term_index) {
    const AdjointDftTerm &term = request.terms[term_index];
    validate_identity(term.forward_identity);
    validate_identity(term.adjoint_identity);
    if (term.forward_snapshot_term >= request.forward->terms.size() ||
        term.adjoint_snapshot_term >= request.adjoint->terms.size() ||
        (request.disposition == AdjointSupportDisposition::device_native &&
         !is_valid(term.adjoint.id)) ||
        !term.forward_spatial_points || !term.adjoint_spatial_points ||
        term.frequencies != request.frequencies.size())
      throw std::invalid_argument("adjoint gradient DFT term is invalid");
    const AdjointDftSnapshotTerm &forward = request.forward->terms[term.forward_snapshot_term];
    const AdjointDftSnapshotTerm &adjoint = request.adjoint->terms[term.adjoint_snapshot_term];
    if (!(forward.identity == term.forward_identity) ||
        !(adjoint.identity == term.adjoint_identity) ||
        !adjoint_same_spatial_image(term.forward_identity, term.adjoint_identity) ||
        term.adjoint.id.value != adjoint.resident.id.value ||
        term.adjoint.offset != adjoint.resident.offset ||
        term.adjoint.elements != adjoint.resident.elements ||
        forward.spatial_points != term.forward_spatial_points ||
        adjoint.spatial_points != term.adjoint_spatial_points ||
        forward.frequencies.size() != term.frequencies ||
        adjoint.frequencies.size() != term.frequencies ||
        !same_request_frequencies(forward, request.frequencies) ||
        !same_request_frequencies(adjoint, request.frequencies))
      throw std::invalid_argument("adjoint gradient DFT pairing is inconsistent");
    for (size_t prior = 0; prior < term_index; ++prior)
      if (request.terms[prior].forward_snapshot_term == term.forward_snapshot_term &&
          request.terms[prior].adjoint_snapshot_term == term.adjoint_snapshot_term)
        throw std::invalid_argument("adjoint gradient DFT pair is duplicated");
  }
  for (const AdjointGradientContribution &contribution : request.contributions) {
    if (contribution.output_index >= request.output_count ||
        contribution.adjoint_snapshot_term >= request.adjoint->terms.size() ||
        contribution.forward_snapshot_term >= request.forward->terms.size() ||
        contribution.forward_value_count > 2 || !contribution.forward_value_count)
      throw std::invalid_argument("adjoint gradient contribution is invalid");
    const AdjointDftSnapshotTerm &adjoint =
        request.adjoint->terms[contribution.adjoint_snapshot_term];
    const AdjointDftSnapshotTerm &forward =
        request.forward->terms[contribution.forward_snapshot_term];
    size_t pair_count = 0;
    for (const AdjointDftTerm &term : request.terms)
      if (term.forward_snapshot_term == contribution.forward_snapshot_term &&
          term.adjoint_snapshot_term == contribution.adjoint_snapshot_term)
        ++pair_count;
    if (pair_count != 1)
      throw std::invalid_argument("adjoint gradient contribution has no unique DFT pair");
    if (contribution.adjoint_value_index >= adjoint.values.size())
      throw std::invalid_argument("adjoint gradient contribution exceeds adjoint snapshot");
    const size_t output_frequency = contribution.output_index / grid_count;
    if (contribution.adjoint_value_index % request.frequencies.size() != output_frequency)
      throw std::invalid_argument("adjoint gradient contribution mixes adjoint frequencies");
    for (uint32_t i = 0; i < contribution.forward_value_count; ++i)
      if ((contribution.forward_value_indices[i] != std::numeric_limits<size_t>::max() &&
           contribution.forward_value_indices[i] >= forward.values.size()) ||
          !std::isfinite(contribution.forward_weights[i]))
        throw std::invalid_argument("adjoint gradient contribution exceeds forward snapshot");
      else if (contribution.forward_value_indices[i] != std::numeric_limits<size_t>::max() &&
               contribution.forward_value_indices[i] % request.frequencies.size() !=
                   output_frequency)
        throw std::invalid_argument("adjoint gradient contribution mixes forward frequencies");
    if (!std::isfinite(contribution.adjoint_coefficient.real()) ||
        !std::isfinite(contribution.adjoint_coefficient.imag()) ||
        !std::isfinite(contribution.material_coefficient.real()) ||
        !std::isfinite(contribution.material_coefficient.imag()) ||
        !std::isfinite(contribution.accumulation_scale))
      throw std::invalid_argument("adjoint gradient contribution coefficient is not finite");
  }
  if (request.disposition == AdjointSupportDisposition::device_native && request.support_reasons)
    throw std::invalid_argument("native adjoint request has unsupported reasons");
  if (request.disposition == AdjointSupportDisposition::device_native) {
    const size_t unsigned_grid_limit =
        std::numeric_limits<unsigned>::max() > std::numeric_limits<size_t>::max() / size_t(256)
            ? std::numeric_limits<size_t>::max()
            : size_t(std::numeric_limits<unsigned>::max()) * size_t(256);
    if (request.output_count > unsigned_grid_limit)
      throw std::overflow_error("NVIDIA adjoint result grid overflows");
  }
  if (request.structural_signature != adjoint_request_signature(request))
    throw std::invalid_argument("adjoint gradient request signature mismatch");
}

void set_adjoint_failure_after_for_testing(int checkpoints) {
  adjoint_failure_after = checkpoints;
}

void set_nvidia_adjoint_failure_for_testing(const char *point) {
#ifdef HAVE_NVIDIA_BACKEND
  if (!point || !*point || std::strcmp(point, "clear") == 0) {
    nvidia::testing::clear_failure();
    return;
  }
  nvidia::testing::failure_point failure = nvidia::testing::failure_point::none;
  if (std::strcmp(point, "allocation") == 0)
    failure = nvidia::testing::failure_point::device_allocate;
  else if (std::strcmp(point, "upload") == 0)
    failure = nvidia::testing::failure_point::host_to_device_copy;
  else if (std::strcmp(point, "launch") == 0)
    failure = nvidia::testing::failure_point::adjoint_launch;
  else if (std::strcmp(point, "download") == 0)
    failure = nvidia::testing::failure_point::device_to_host_copy;
  else
    throw std::invalid_argument("unknown NVIDIA adjoint failure point");
  nvidia::testing::fail_next(failure);
#else
  (void)point;
  throw std::runtime_error("NVIDIA adjoint failure injection requires the NVIDIA backend");
#endif
}

void adjoint_failure_checkpoint() {
  if (adjoint_failure_after < 0) return;
  if (adjoint_failure_after-- == 0)
    throw std::runtime_error("injected adjoint transaction failure");
}

std::shared_ptr<const AdjointDftSnapshot>
make_adjoint_dft_snapshot(const std::vector<dft_fields *> &monitors) {
  if (monitors.empty()) throw std::invalid_argument("adjoint DFT snapshot has no monitors");
  fields *owner = NULL;
  std::vector<dft_chunk *> heads;
  heads.reserve(monitors.size());
  for (dft_fields *monitor : monitors) {
    if (!monitor) throw std::invalid_argument("adjoint DFT snapshot monitor is null");
    fields *candidate = monitor->monitor_lifetime ? monitor->monitor_lifetime->owner : NULL;
    if (!candidate) throw std::invalid_argument("adjoint DFT snapshot monitor has no live owner");
    if (owner && owner != candidate)
      throw std::invalid_argument("adjoint DFT snapshot monitors have different owners");
    owner = candidate;
    heads.push_back(monitor->chunks);
  }
  if (!owner || !owner->material_ir)
    throw std::invalid_argument("adjoint DFT snapshot owner has no material recipe");
  backend_refresh_dft_chains(*owner, int(heads.size()), heads.data(),
                             "capture adjoint forward DFT snapshot");

  std::shared_ptr<AdjointDftSnapshot> snapshot(new AdjointDftSnapshot);
  const MaterialIR *ir = material_ir_for(*owner);
  snapshot->material_signature = ir ? ir->signature : 0;
  snapshot->material_layout_signature = ir ? ir->layout_signature : 0;
  for (int i = 0; i < fields::num_mutation_kinds; ++i)
    snapshot->mutation_generation[i] = owner->mutation_generation[i];
  snapshot->checkpoint_publication_generation = owner->checkpoint_publication_generation;
  for (int d = 0; d < 5; ++d) {
    snapshot->bloch_k[d] = owner->k[d];
    for (int side = 0; side < 2; ++side)
      snapshot->boundary_conditions[side][d] = int(owner->boundaries[side][d]);
  }
  snapshot->cylindrical_m = owner->m;
  snapshot->precision_policy = owner->options.precision;
  snapshot->device_id = owner->options.device_id;

  for (dft_fields *monitor : monitors)
    for (dft_chunk *chunk = monitor->chunks; chunk; chunk = chunk->next_in_dft) {
      if (!chunk->persist)
        throw std::invalid_argument("adjoint forward DFT snapshot requires persistent monitors");
      AdjointDftSnapshotTerm term;
      term.identity = identity_for(*owner, *chunk);
      term.resident = ArrayRef{invalid_array(), 0, 0};
      term.storage_precision = policy_for(owner->options.precision).monitor;
      term.spatial_points = chunk->N;
      term.frequencies = chunk->omega;
      const size_t count = checked_product(chunk->N, chunk->omega.size(),
                                           "adjoint DFT snapshot allocation");
      ArrayId id = invalid_array();
      ptrdiff_t offset = 0;
      if (owner->array_catalog && owner->array_catalog->locate(chunk->dft, id, offset) &&
          offset >= 0)
        term.resident = ArrayRef{id, size_t(offset), count};
      term.values.reserve(count);
      for (size_t i = 0; i < count; ++i)
        term.values.push_back(std::complex<double>(chunk->dft[i].real(), chunk->dft[i].imag()));
      snapshot->terms.push_back(term);
    }
  std::sort(snapshot->terms.begin(), snapshot->terms.end(),
            [](const AdjointDftSnapshotTerm &a, const AdjointDftSnapshotTerm &b) {
              return identity_less(a.identity, b.identity);
            });
  snapshot->structural_signature = adjoint_snapshot_structural_signature(*snapshot);
  snapshot->value_signature = adjoint_snapshot_value_signature(*snapshot);
  validate_adjoint_snapshot(*snapshot);
  return snapshot;
}

std::shared_ptr<const AdjointDftSnapshot>
capture_adjoint_dft_snapshot(const std::vector<dft_fields *> &monitors) {
  const std::shared_ptr<const AdjointDftSnapshot> snapshot = make_adjoint_dft_snapshot(monitors);
  for (dft_fields *monitor : monitors) monitor->adjoint_snapshot = snapshot;
  return snapshot;
}

const AdjointDftSnapshot *adjoint_snapshot_from(const dft_fields &monitor) {
  return static_cast<const AdjointDftSnapshot *>(monitor.adjoint_snapshot.get());
}

AdjointDftIdentity adjoint_dft_identity(const fields &owner, const dft_chunk &chunk) {
  return identity_for(owner, chunk);
}

void validate_adjoint_snapshot_freshness(const AdjointDftSnapshot &snapshot,
                                         const fields &owner) {
  const MaterialIR *ir = material_ir_for(owner);
  if (!ir || snapshot.material_signature != ir->signature ||
      snapshot.material_layout_signature != ir->layout_signature)
    throw std::invalid_argument("adjoint DFT snapshot material recipe is stale");
  const MutationKind required[] = {
      MutationKind::material_values, MutationKind::material_region,
      MutationKind::material_phase, MutationKind::material_definition,
      MutationKind::chunk_topology,
      MutationKind::precision_policy};
  for (const MutationKind kind : required) {
    const int index = int(kind);
    if (snapshot.mutation_generation[index] != owner.mutation_generation[index])
      throw std::invalid_argument(std::string("adjoint DFT snapshot is stale after ") +
                                  mutation_kind_name(kind));
  }
  /* The normal adjoint workflow reverses Bloch k or cylindrical m and restores
     it before consuming the forward snapshot. Those reversible transitions
     advance monotonic lifecycle generations, but the immutable DFT values are
     valid again once the exact boundary/coordinate semantics are restored. */
  if (!same_boundary_identity(snapshot, owner))
    throw std::invalid_argument("adjoint DFT snapshot boundary identity is stale");
  if (snapshot.cylindrical_m != owner.m)
    throw std::invalid_argument("adjoint DFT snapshot cylindrical coordinate is stale");
  if (snapshot.precision_policy != owner.options.precision ||
      (snapshot.device_id != automatic_device && owner.options.device_id != automatic_device &&
       snapshot.device_id != owner.options.device_id))
    throw std::invalid_argument("adjoint DFT snapshot backend identity is stale");
  if (snapshot.checkpoint_publication_generation != owner.checkpoint_publication_generation)
    throw std::invalid_argument("adjoint DFT snapshot is stale after checkpoint publication");
}

void compute_adjoint_gradient_oracle(const AdjointGradientRequest &request, double *local_result,
                                     size_t result_count) {
  validate_adjoint_request(request);
  if (!local_result && result_count)
    throw std::invalid_argument("adjoint gradient oracle has no output buffer");
  if (result_count != request.output_count)
    throw std::invalid_argument("adjoint gradient oracle output-count mismatch");
  std::fill(local_result, local_result + result_count, 0.0);
  for (const AdjointGradientContribution &contribution : request.contributions) {
    const AdjointDftSnapshotTerm &adjoint =
        request.adjoint->terms[contribution.adjoint_snapshot_term];
    const AdjointDftSnapshotTerm &forward =
        request.forward->terms[contribution.forward_snapshot_term];
    std::complex<double> fwd(0.0, 0.0);
    for (uint32_t i = 0; i < contribution.forward_value_count; ++i)
      if (contribution.forward_value_indices[i] != std::numeric_limits<size_t>::max())
        fwd += forward.values[contribution.forward_value_indices[i]] *
               contribution.forward_weights[i];
    const std::complex<double> adjusted_adjoint =
        adjoint.values[contribution.adjoint_value_index] * contribution.adjoint_coefficient;
    const std::complex<double> material = fwd * contribution.material_coefficient;
    const std::complex<double> product = adjusted_adjoint * material;
    local_result[contribution.output_index] += product.real() * contribution.accumulation_scale;
  }
}

} // namespace meep
