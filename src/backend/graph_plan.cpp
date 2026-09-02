/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/graph_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace meep {
namespace {

void mix(uint64_t &signature, uint64_t value) {
  signature ^= value + 0x9e3779b97f4a7c15ull + (signature << 6) + (signature >> 2);
}

void mix_double(uint64_t &signature, double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double is not 64-bit");
  std::memcpy(&bits, &value, sizeof(bits));
  mix(signature, bits);
}

void mix_storage_key(uint64_t &signature, const StorageKey &key) {
  mix(signature, uint64_t(key.chunk));
  mix(signature, uint64_t(key.kind));
  mix(signature, uint64_t(key.component_));
  mix(signature, uint64_t(key.cmp));
  mix(signature, key.aux);
}

void mix_halo_array_key(uint64_t &signature, const HaloArrayKey &key) {
  mix(signature, uint64_t(key.chunk));
  mix(signature, uint64_t(key.role));
  mix(signature, uint64_t(key.component_));
  mix(signature, uint64_t(key.cmp));
  mix(signature, uint64_t(key.aux));
  mix(signature, uint64_t(key.susceptibility_id));
  mix(signature, uint64_t(key.internal_index));
  mix(signature, uint64_t(key.point_index));
  mix(signature, uint64_t(key.complex_internal));
}

void mix_array_spec(uint64_t &signature, const ArraySpec &spec) {
  mix(signature, uint64_t(spec.role));
  mix(signature, uint64_t(spec.element_type));
  mix(signature, uint64_t(spec.storage));
  mix(signature, spec.elements);
  mix(signature, spec.alignment);
  mix(signature, uint64_t(spec.classification_provisional));
  mix(signature, uint64_t(spec.classification_elided));
}

uint64_t stable_catalog_array_signature(const CpuArrayCatalog &catalog, ArrayId id) {
  if (!is_valid(id) || id.value >= catalog.size())
    throw std::invalid_argument("graph array identity is out of range");
  const ArraySpec &spec = catalog.spec(id);
  uint64_t signature = 0xcbf29ce484222325ull;
  mix_storage_key(signature, catalog.key(id));
  mix_array_spec(signature, spec);
  if (is_valid(spec.alias_of)) {
    if (spec.alias_of.value >= catalog.size())
      throw std::invalid_argument("graph array alias identity is out of range");
    mix_storage_key(signature, catalog.key(spec.alias_of));
  }
  else
    mix(signature, std::numeric_limits<uint64_t>::max());
  return signature;
}

const GraphArrayIdentity &array_identity(const GraphLoweringAuthorities &authority, ArrayId id) {
  for (const GraphArrayIdentity &identity : authority.array_identities)
    if (identity.source_id == id) return identity;
  throw std::invalid_argument("graph operation refers to an array absent from its authority");
}

void mix_array_ref(uint64_t &signature, const ArrayRef &ref,
                   const GraphLoweringAuthorities &authority) {
  mix(signature, array_identity(authority, ref.id).signature);
  mix(signature, ref.offset);
  mix(signature, ref.elements);
}

void mix_access(uint64_t &signature, const BufferAccess &access,
                const GraphLoweringAuthorities &authority) {
  mix_array_ref(signature, access.array, authority);
  mix(signature, uint64_t(access.mode));
}

void mix_slab(uint64_t &signature, const SlabRef &slab, const CpuArrayCatalog &catalog) {
  mix(signature, stable_catalog_array_signature(catalog, slab.array));
  mix(signature, uint64_t(slab.base));
  for (int axis = 0; axis < 3; ++axis) {
    mix(signature, uint64_t(slab.counts[axis]));
    mix(signature, uint64_t(slab.strides[axis]));
  }
}

void mix_element(uint64_t &signature, const ElementRef &element, const CpuArrayCatalog &catalog) {
  mix(signature, stable_catalog_array_signature(catalog, element.array));
  mix(signature, uint64_t(element.index));
}

void mix_halo_segment(uint64_t &signature, const HaloSegment &segment) {
  mix(signature, segment.first_slab);
  mix(signature, segment.nslabs);
  mix(signature, segment.count);
  mix(signature, segment.residue);
}

void mix_host_halo_key(uint64_t &signature, const HostHaloKey &key) {
  mix(signature, uint64_t(key.chunk));
  mix(signature, uint64_t(key.ft));
  mix(signature, uint64_t(key.state_index));
  mix(signature, uint64_t(key.susceptibility_id));
  mix(signature, uint64_t(key.component_));
  mix(signature, uint64_t(key.cmp));
  mix(signature, uint64_t(key.internal_index));
  mix(signature, uint64_t(key.point_index));
  mix(signature, uint64_t(key.complex_internal));
}

size_t checked_slab_elements(const SlabRef &slab, size_t extent, const char *what) {
  size_t elements = 1;
  __int128 minimum = slab.base, maximum = slab.base;
  for (int axis = 0; axis < 3; ++axis) {
    if (slab.counts[axis] <= 0)
      throw std::invalid_argument(std::string(what) + " has a nonpositive slab count");
    if (elements > std::numeric_limits<size_t>::max() / size_t(slab.counts[axis]))
      throw std::overflow_error(std::string(what) + " slab element count overflow");
    elements *= size_t(slab.counts[axis]);
    const __int128 delta = __int128(slab.counts[axis] - 1) * slab.strides[axis];
    if (delta < 0)
      minimum += delta;
    else
      maximum += delta;
  }
  if (minimum < 0 || maximum < minimum || maximum >= __int128(extent))
    throw std::invalid_argument(std::string(what) + " slab range is out of bounds");
  return elements;
}

void validate_catalog_slab(const SlabRef &slab, const CpuArrayCatalog &catalog, const char *what) {
  if (!is_valid(slab.array) || slab.array.value >= catalog.size())
    throw std::invalid_argument(std::string(what) + " slab array is out of range");
  (void)checked_slab_elements(slab, catalog.spec(slab.array).elements, what);
}

void validate_catalog_element(const ElementRef &element, const CpuArrayCatalog &catalog,
                              const char *what) {
  if (!is_valid(element.array) || element.array.value >= catalog.size() || element.index < 0 ||
      size_t(element.index) >= catalog.spec(element.array).elements)
    throw std::invalid_argument(std::string(what) + " element is out of range");
}

void validate_halo_order(const std::vector<HaloSegment> &order, size_t slabs, size_t residue,
                         size_t expected, const char *what) {
  size_t observed = 0, residue_used = 0;
  for (const HaloSegment &segment : order) {
    if ((!segment.nslabs && !segment.residue) || (segment.nslabs && segment.residue) ||
        (segment.nslabs && !segment.count) || uint64_t(segment.first_slab) + segment.nslabs > slabs)
      throw std::invalid_argument(std::string(what) + " order segment is malformed");
    const uint64_t slab_elements = uint64_t(segment.nslabs) * segment.count;
    if (slab_elements > std::numeric_limits<size_t>::max() ||
        observed > std::numeric_limits<size_t>::max() - size_t(slab_elements) ||
        observed + size_t(slab_elements) > std::numeric_limits<size_t>::max() - segment.residue)
      throw std::overflow_error(std::string(what) + " order element count overflow");
    observed += size_t(slab_elements) + segment.residue;
    if (residue_used > residue || segment.residue > residue - residue_used)
      throw std::invalid_argument(std::string(what) + " order residue is out of range");
    residue_used += segment.residue;
  }
  if (residue_used != residue || observed != expected)
    throw std::invalid_argument(std::string(what) + " order does not cover its block");
}

void validate_source_halo(const HaloPlan &source, const HaloArrayTable &arrays,
                          const HostHaloArrayTable &host_arrays) {
  connect_phase phase = CONNECT_COPY;
  if (!decode_host_halo_phase(uint32_t(source.phase), phase) || source.ft < E_stuff ||
      source.ft >= NUM_FIELD_TYPES || source.chunks.first < 0 || source.chunks.second < 0 ||
      !source.block_elements ||
      source.block_offset > std::numeric_limits<size_t>::max() - source.block_elements)
    throw std::invalid_argument("source halo logical metadata is invalid");
  if (source.sequence_index != uint32_t(phase))
    throw std::invalid_argument("source halo sequence index is invalid");
  if (source.storage != HaloStorageDisposition::canonical &&
      source.storage != HaloStorageDisposition::host_owned)
    throw std::invalid_argument("source halo storage disposition is invalid");
  if (source.storage == HaloStorageDisposition::canonical) {
    for (const SlabRef &slab : source.gather_slabs) {
      if (!arrays.contains(slab.array))
        throw std::invalid_argument("source halo gather slab array is out of range");
      (void)checked_slab_elements(slab, arrays.spec(slab.array).elements, "source halo gather");
    }
    for (const SlabRef &slab : source.scatter_slabs) {
      if (!arrays.contains(slab.array))
        throw std::invalid_argument("source halo scatter slab array is out of range");
      (void)checked_slab_elements(slab, arrays.spec(slab.array).elements, "source halo scatter");
    }
    for (const ElementRef &element : source.gather)
      if (!arrays.contains(element.array) || element.index < 0 ||
          size_t(element.index) >= arrays.spec(element.array).elements)
        throw std::invalid_argument("source halo gather element is out of range");
    for (const ElementRef &element : source.scatter)
      if (!arrays.contains(element.array) || element.index < 0 ||
          size_t(element.index) >= arrays.spec(element.array).elements)
        throw std::invalid_argument("source halo scatter element is out of range");
    const bool has_gather = !source.gather_slabs.empty() || !source.gather.empty();
    const bool has_scatter = !source.scatter_slabs.empty() || !source.scatter.empty();
    if (!has_gather && !has_scatter && source.host_gather.empty() && source.host_scatter.empty())
      throw std::invalid_argument("source halo has no referenced elements");
    if (has_gather)
      validate_halo_order(source.gather_order, source.gather_slabs.size(), source.gather.size(),
                          source.block_elements, "source halo gather");
    if (has_scatter)
      validate_halo_order(source.scatter_order, source.scatter_slabs.size(), source.scatter.size(),
                          source.block_elements, "source halo scatter");
  }
  for (const HostElementRef &element : source.host_gather)
    if (!host_arrays.contains(element.id))
      throw std::invalid_argument("source host halo gather identity is stale");
  for (const HostElementRef &element : source.host_scatter)
    if (!host_arrays.contains(element.id))
      throw std::invalid_argument("source host halo scatter identity is stale");
  if (source.storage == HaloStorageDisposition::host_owned &&
      ((!source.host_gather.empty() && source.host_gather.size() != source.block_elements) ||
       (!source.host_scatter.empty() && source.host_scatter.size() != source.block_elements)))
    throw std::invalid_argument("source host halo mirror has the wrong element count");
  if ((phase != CONNECT_PHASE && !source.phase_values.empty()) ||
      (phase == CONNECT_PHASE && !source.phase_values.empty() &&
       (source.block_elements % 2 || source.phase_values.size() != source.block_elements / 2)))
    throw std::invalid_argument("source halo phase values are malformed");
}

void mix_source_slab(uint64_t &signature, const SlabRef &slab, const HaloArrayTable &arrays) {
  if (!arrays.contains(slab.array))
    throw std::invalid_argument("source halo slab array is out of range");
  (void)checked_slab_elements(slab, arrays.spec(slab.array).elements, "source halo");
  mix_halo_array_key(signature, arrays.key(slab.array));
  mix_array_spec(signature, arrays.spec(slab.array));
  mix(signature, uint64_t(slab.base));
  for (int axis = 0; axis < 3; ++axis) {
    mix(signature, uint64_t(slab.counts[axis]));
    mix(signature, uint64_t(slab.strides[axis]));
  }
}

void mix_source_element(uint64_t &signature, const ElementRef &element,
                        const HaloArrayTable &arrays) {
  if (!arrays.contains(element.array) || element.index < 0 ||
      size_t(element.index) >= arrays.spec(element.array).elements)
    throw std::invalid_argument("source halo element is out of range");
  mix_halo_array_key(signature, arrays.key(element.array));
  mix_array_spec(signature, arrays.spec(element.array));
  mix(signature, uint64_t(element.index));
}

uint64_t compute_halo_row_signature(const HaloPlan &source, const HaloPlan &canonical,
                                    const HaloArrayTable &source_arrays,
                                    const HostHaloArrayTable &host_arrays,
                                    const CpuArrayCatalog &catalog) {
  uint64_t signature = 0xcbf29ce484222325ull;
  mix(signature, uint64_t(canonical.ft));
  mix(signature, uint64_t(canonical.chunks.first));
  mix(signature, uint64_t(canonical.chunks.second));
  mix(signature, uint64_t(canonical.phase));
  mix(signature, uint64_t(canonical.peer_rank));
  mix(signature, uint64_t(canonical.tag));
  mix(signature, uint64_t(canonical.same_rank));
  mix(signature, uint64_t(canonical.storage));
  mix(signature, canonical.sequence_index);
  mix(signature, canonical.block_offset);
  mix(signature, canonical.block_elements);

  /* Source-private IDs are represented only by their stable HaloArrayKey and
     spec; canonical IDs are represented only by StorageKey/spec. */
  mix(signature, source.gather_slabs.size());
  for (const SlabRef &slab : source.gather_slabs)
    mix_source_slab(signature, slab, source_arrays);
  mix(signature, source.scatter_slabs.size());
  for (const SlabRef &slab : source.scatter_slabs)
    mix_source_slab(signature, slab, source_arrays);
  mix(signature, source.gather.size());
  for (const ElementRef &element : source.gather)
    mix_source_element(signature, element, source_arrays);
  mix(signature, source.scatter.size());
  for (const ElementRef &element : source.scatter)
    mix_source_element(signature, element, source_arrays);
  mix(signature, source.host_gather.size());
  for (const HostElementRef &element : source.host_gather) {
    if (!host_arrays.contains(element.id))
      throw std::invalid_argument("source host halo gather identity is stale");
    mix_host_halo_key(signature, host_arrays.key(element.id));
  }
  mix(signature, source.host_scatter.size());
  for (const HostElementRef &element : source.host_scatter) {
    if (!host_arrays.contains(element.id))
      throw std::invalid_argument("source host halo scatter identity is stale");
    mix_host_halo_key(signature, host_arrays.key(element.id));
  }
  mix(signature, source.gather_order.size());
  for (const HaloSegment &segment : source.gather_order)
    mix_halo_segment(signature, segment);
  mix(signature, source.scatter_order.size());
  for (const HaloSegment &segment : source.scatter_order)
    mix_halo_segment(signature, segment);

  mix(signature, canonical.gather_slabs.size());
  for (const SlabRef &slab : canonical.gather_slabs) {
    validate_catalog_slab(slab, catalog, "canonical halo gather");
    mix_slab(signature, slab, catalog);
  }
  mix(signature, canonical.scatter_slabs.size());
  for (const SlabRef &slab : canonical.scatter_slabs) {
    validate_catalog_slab(slab, catalog, "canonical halo scatter");
    mix_slab(signature, slab, catalog);
  }
  mix(signature, canonical.gather.size());
  for (const ElementRef &element : canonical.gather) {
    validate_catalog_element(element, catalog, "canonical halo gather");
    mix_element(signature, element, catalog);
  }
  mix(signature, canonical.scatter.size());
  for (const ElementRef &element : canonical.scatter) {
    validate_catalog_element(element, catalog, "canonical halo scatter");
    mix_element(signature, element, catalog);
  }
  mix(signature, canonical.phase_values.size());
  for (const std::complex<realnum> &phase : canonical.phase_values) {
    mix_double(signature, double(phase.real()));
    mix_double(signature, double(phase.imag()));
  }
  return signature;
}

GraphHaloRow make_halo_row(const HaloPlan &source, const HaloPlan &halo,
                           const HaloArrayTable &source_arrays,
                           const HostHaloArrayTable &host_arrays, const CpuArrayCatalog &catalog) {
  GraphHaloRow row;
  row.ft = halo.ft;
  row.chunks = halo.chunks;
  row.phase = halo.phase;
  row.peer_rank = halo.peer_rank;
  row.tag = halo.tag;
  row.route = halo.storage == HaloStorageDisposition::host_owned ? GraphHaloRoute::host_owned
              : halo.same_rank                                   ? GraphHaloRoute::local_device
                                                                 : GraphHaloRoute::remote_host;
  row.sequence_index = halo.sequence_index;
  row.block_offset = halo.block_offset;
  row.block_elements = halo.block_elements;
  row.signature = compute_halo_row_signature(source, halo, source_arrays, host_arrays, catalog);
  return row;
}

GraphArrayIdentity make_array_identity(const CpuArrayCatalog &catalog, ArrayId id) {
  const ArraySpec &spec = catalog.spec(id);
  return GraphArrayIdentity{id,
                            catalog.key(id),
                            spec.role,
                            spec.element_type,
                            spec.storage,
                            spec.elements,
                            spec.alignment,
                            spec.classification_provisional,
                            spec.classification_elided,
                            stable_catalog_array_signature(catalog, id)};
}

GraphZeroRow make_zero_row(field_type ft, uint32_t chunk, const ZeroPlan &source,
                           const ZeroPlan &canonical, const HaloArrayTable &source_arrays,
                           const CpuArrayCatalog &catalog) {
  uint64_t signature = 0xcbf29ce484222325ull;
  mix(signature, uint64_t(ft));
  mix(signature, chunk);
  size_t elements = 0;
  mix(signature, source.slabs.size());
  for (const SlabRef &slab : source.slabs)
    mix_source_slab(signature, slab, source_arrays);
  mix(signature, source.residue.size());
  for (const ElementRef &element : source.residue)
    mix_source_element(signature, element, source_arrays);
  mix(signature, canonical.slabs.size());
  for (const SlabRef &slab : canonical.slabs) {
    validate_catalog_slab(slab, catalog, "canonical zero");
    const size_t count =
        checked_slab_elements(slab, catalog.spec(slab.array).elements, "canonical zero");
    if (elements > std::numeric_limits<size_t>::max() - count)
      throw std::overflow_error("canonical zero element count overflow");
    elements += count;
    mix_slab(signature, slab, catalog);
  }
  mix(signature, canonical.residue.size());
  for (const ElementRef &element : canonical.residue) {
    validate_catalog_element(element, catalog, "canonical zero");
    if (elements == std::numeric_limits<size_t>::max())
      throw std::overflow_error("canonical zero element count overflow");
    ++elements;
    mix_element(signature, element, catalog);
  }
  return GraphZeroRow{ft, chunk, elements, signature};
}

void validate_source_zero(const ZeroPlan &source, const HaloArrayTable &arrays) {
  for (const SlabRef &slab : source.slabs) {
    if (!arrays.contains(slab.array))
      throw std::invalid_argument("source zero slab array is out of range");
    (void)checked_slab_elements(slab, arrays.spec(slab.array).elements, "source zero");
  }
  for (const ElementRef &element : source.residue)
    if (!arrays.contains(element.array) || element.index < 0 ||
        size_t(element.index) >= arrays.spec(element.array).elements)
      throw std::invalid_argument("source zero element is out of range");
}

bool same_array_ref(const ArrayRef &a, const ArrayRef &b) {
  return a.id == b.id && a.offset == b.offset && a.elements == b.elements;
}

bool same_access(const BufferAccess &a, const BufferAccess &b) {
  return same_array_ref(a.array, b.array) && a.mode == b.mode;
}

bool same_operation(const Operation &a, const Operation &b) {
  if (a.kind != b.kind || a.descriptor_index != b.descriptor_index ||
      a.descriptor_count != b.descriptor_count ||
      a.material_refresh_index != b.material_refresh_index ||
      a.material_refresh_count != b.material_refresh_count ||
      a.beta_descriptor_index != b.beta_descriptor_index ||
      a.beta_descriptor_count != b.beta_descriptor_count ||
      a.cylindrical_m_descriptor_index != b.cylindrical_m_descriptor_index ||
      a.cylindrical_m_descriptor_count != b.cylindrical_m_descriptor_count ||
      a.cylindrical_origin_action_index != b.cylindrical_origin_action_index ||
      a.cylindrical_origin_action_count != b.cylindrical_origin_action_count ||
      a.polarization_group_index != b.polarization_group_index ||
      a.polarization_group_count != b.polarization_group_count ||
      a.polarization_subtraction_index != b.polarization_subtraction_index ||
      a.polarization_subtraction_count != b.polarization_subtraction_count ||
      a.magnetic_state_index != b.magnetic_state_index ||
      a.magnetic_state_count != b.magnetic_state_count ||
      a.legacy_flux_index != b.legacy_flux_index || a.legacy_flux_count != b.legacy_flux_count ||
      a.source_descriptor_index != b.source_descriptor_index ||
      a.source_descriptor_count != b.source_descriptor_count || a.guard.kind != b.guard.kind ||
      a.guard.scalar_slot != b.guard.scalar_slot ||
      a.guard.variant_index != b.guard.variant_index || a.ft != b.ft ||
      a.source_time_offset != b.source_time_offset || a.accesses.size() != b.accesses.size())
    return false;
  for (size_t i = 0; i < a.accesses.size(); ++i)
    if (!same_access(a.accesses[i], b.accesses[i])) return false;
  return true;
}

void hash_operation(uint64_t &signature, const Operation &op,
                    const GraphLoweringAuthorities &authority) {
  mix(signature, uint64_t(op.kind));
  mix(signature, op.descriptor_index);
  mix(signature, op.descriptor_count);
  mix(signature, op.material_refresh_index);
  mix(signature, op.material_refresh_count);
  mix(signature, op.beta_descriptor_index);
  mix(signature, op.beta_descriptor_count);
  mix(signature, op.cylindrical_m_descriptor_index);
  mix(signature, op.cylindrical_m_descriptor_count);
  mix(signature, op.cylindrical_origin_action_index);
  mix(signature, op.cylindrical_origin_action_count);
  mix(signature, op.polarization_group_index);
  mix(signature, op.polarization_group_count);
  mix(signature, op.polarization_subtraction_index);
  mix(signature, op.polarization_subtraction_count);
  mix(signature, op.magnetic_state_index);
  mix(signature, op.magnetic_state_count);
  mix(signature, op.legacy_flux_index);
  mix(signature, op.legacy_flux_count);
  mix(signature, op.source_descriptor_index);
  mix(signature, op.source_descriptor_count);
  mix(signature, uint64_t(op.guard.kind));
  mix(signature, op.guard.scalar_slot);
  mix(signature, op.guard.variant_index);
  mix(signature, uint64_t(op.ft));
  mix_double(signature, op.source_time_offset);
  mix(signature, op.accesses.size());
  for (const BufferAccess &access : op.accesses)
    mix_access(signature, access, authority);
}

uint64_t hash_graph_operation(const GraphOperationRef &operation,
                              const GraphLoweringAuthorities &authority) {
  uint64_t signature = 0xcbf29ce484222325ull;
  mix(signature, operation.operation_index);
  mix(signature, uint64_t(operation.node_class));
  hash_operation(signature, operation.operation, authority);
  mix(signature, operation.scalar_slots.size());
  for (uint32_t slot : operation.scalar_slots)
    mix(signature, slot);
  return signature;
}

void mix_accesses(uint64_t &signature, const std::vector<BufferAccess> &accesses,
                  const GraphLoweringAuthorities &authority) {
  mix(signature, accesses.size());
  for (const BufferAccess &access : accesses)
    mix_access(signature, access, authority);
}

void mix_cw_operation_ref(uint64_t &signature, const CwStepOperationRef &ref) {
  mix(signature, ref.operation_index);
  mix(signature, uint64_t(ref.kind));
  mix(signature, uint64_t(ref.ft));
  mix(signature, ref.descriptor_index);
  mix(signature, ref.descriptor_count);
  mix(signature, ref.polarization_subtraction_index);
  mix(signature, ref.polarization_subtraction_count);
}

uint64_t stable_cw_plan_signature(const CwPlan &plan, const GraphLoweringAuthorities &authority) {
  uint64_t signature = 0xcbf29ce484222325ull;
  mix(signature, plan.rhs_stages.size());
  for (const CwRhsStage &stage : plan.rhs_stages) {
    mix(signature, uint64_t(stage.ft));
    mix_double(signature, stage.source_time_offset);
    mix(signature, stage.source_time_index);
    mix(signature, stage.source_time_count);
    mix(signature, stage.source_index);
    mix(signature, stage.source_count);
    mix_cw_operation_ref(signature, stage.boundary);
    mix_cw_operation_ref(signature, stage.constitutive);
    mix_accesses(signature, stage.accesses, authority);
  }
  mix(signature, plan.rhs_sources.size());
  for (const CwRhsSourceDescriptor &source : plan.rhs_sources) {
    mix(signature, source.source_descriptor_index);
    mix(signature, source.source_ordinal);
    mix(signature, uint64_t(source.mode));
  }
  mix_cw_operation_ref(signature, plan.unpack.first_boundary);
  mix_cw_operation_ref(signature, plan.unpack.constitutive);
  mix_cw_operation_ref(signature, plan.unpack.second_boundary);
  mix(signature, uint64_t(plan.unpack.skip_w_components));
  mix(signature, uint64_t(plan.unpack.invalidate_field_values));
  mix(signature, plan.final_dfts.size());
  for (const CwDftDescriptorRef &dft : plan.final_dfts) {
    mix(signature, dft.descriptor_index);
    mix(signature, uint64_t(dft.chunk));
    mix(signature, uint64_t(dft.c));
    mix(signature, uint64_t(dft.decimation_factor));
    mix(signature, dft.due_scalar_slot);
  }
  mix_accesses(signature, plan.rhs_accesses, authority);
  mix_accesses(signature, plan.unpack_accesses, authority);
  mix_accesses(signature, plan.final_dft_accesses, authority);
  mix(signature, plan.source_time_count);
  mix(signature, plan.rhs_source_count);
  mix(signature, plan.final_dft_count);
  return signature;
}

void validate_cw_operation_ref(const StepPlan &plan, const CwStepOperationRef &ref,
                               OpKind expected_kind, field_type expected_ft,
                               const char *what) {
  if (ref.operation_index >= plan.operations.size())
    throw std::invalid_argument(std::string(what) + " operation index is out of range");
  const Operation &operation = plan.operations[ref.operation_index];
  if (ref.kind != expected_kind || ref.ft != expected_ft || operation.kind != ref.kind ||
      operation.ft != ref.ft || operation.descriptor_index != ref.descriptor_index ||
      operation.descriptor_count != ref.descriptor_count ||
      operation.polarization_subtraction_index != ref.polarization_subtraction_index ||
      operation.polarization_subtraction_count != ref.polarization_subtraction_count)
    throw std::invalid_argument(std::string(what) + " operation reference is stale");
}

const GraphHaloDisposition *find_halo_disposition(const GraphLoweringAuthorities &authority,
                                                  uint32_t operation_index) {
  for (const GraphHaloDisposition &disposition : authority.halo_dispositions)
    if (disposition.operation_index == operation_index) return &disposition;
  return NULL;
}

const GraphHostInterval *find_host_interval(const GraphLoweringAuthorities &authority,
                                            uint32_t marker_operation) {
  for (const GraphHostInterval &interval : authority.host_intervals)
    if (interval.marker_operation == marker_operation) return &interval;
  return NULL;
}

const GraphRemoteOverlap *find_remote_overlap(const GraphLoweringAuthorities &authority,
                                              uint32_t halo_operation) {
  for (const GraphRemoteOverlap &overlap : authority.remote_overlaps)
    if (overlap.halo_operation == halo_operation) return &overlap;
  return NULL;
}

bool operation_is_covered(const GraphLoweringAuthorities &authority, uint32_t operation_index) {
  for (const GraphHostInterval &interval : authority.host_intervals)
    if (operation_index >= interval.first_covered_operation &&
        uint64_t(operation_index) <
            uint64_t(interval.first_covered_operation) + interval.covered_operation_count)
      return true;
  return false;
}

GraphBoundaryKind host_boundary_kind(const Operation &operation, const GraphHaloDisposition *halo) {
  /* A segment guard is a host decision regardless of whether the guarded halo
     happens to be same-rank.  In particular, material-phasing halo transfers
     must not disappear into an unconditional local-device segment. */
  if (operation.guard.kind == GuardKind::segment_boundary)
    return operation.kind == OpKind::phase_material ? GraphBoundaryKind::material_phase
                                                    : GraphBoundaryKind::segment_guard;
  switch (operation.kind) {
    case OpKind::evaluate_source_scalars: return GraphBoundaryKind::source_evaluation;
    case OpKind::host_callback: return GraphBoundaryKind::host_callback;
    case OpKind::phase_material: return GraphBoundaryKind::material_phase;
    case OpKind::update_material_coefficients: return GraphBoundaryKind::material_refresh;
    case OpKind::transfer_halo:
      return halo && halo->entirely_local_canonical ? GraphBoundaryKind::none
                                                    : GraphBoundaryKind::remote_halo;
    case OpKind::increment_time: return GraphBoundaryKind::time_update;
    case OpKind::restore_magnetic_fields:
    case OpKind::synchronize_magnetic_fields: return GraphBoundaryKind::magnetic_transition;
    default: return GraphBoundaryKind::none;
  }
}

bool valid_operation_kind(OpKind kind) { return uint32_t(kind) < uint32_t(OpKind::num_kinds); }

bool valid_guard_kind(GuardKind kind) {
  switch (kind) {
    case GuardKind::always:
    case GuardKind::static_predicate:
    case GuardKind::device_predicate:
    case GuardKind::graph_variant:
    case GuardKind::segment_boundary: return true;
  }
  return false;
}

bool valid_graph_variant(GraphVariantKind variant) {
  switch (variant) {
    case GraphVariantKind::ordinary:
    case GraphVariantKind::magnetic_half_step:
    case GraphVariantKind::magnetic_restore:
    case GraphVariantKind::cw_operator: return true;
  }
  return false;
}

bool valid_graph_execution_mode(GraphExecutionMode mode) {
  switch (mode) {
    case GraphExecutionMode::automatic:
    case GraphExecutionMode::eager:
    case GraphExecutionMode::required: return true;
  }
  return false;
}

bool valid_graph_node_class(GraphNodeClass value) {
  return value == GraphNodeClass::device || value == GraphNodeClass::device_predicate ||
         value == GraphNodeClass::host_boundary;
}

bool valid_graph_boundary_kind(GraphBoundaryKind value) {
  switch (value) {
    case GraphBoundaryKind::none:
    case GraphBoundaryKind::source_evaluation:
    case GraphBoundaryKind::host_callback:
    case GraphBoundaryKind::material_phase:
    case GraphBoundaryKind::material_refresh:
    case GraphBoundaryKind::remote_halo:
    case GraphBoundaryKind::legacy_flux_publish:
    case GraphBoundaryKind::finite_diagnostic:
    case GraphBoundaryKind::time_update:
    case GraphBoundaryKind::magnetic_transition:
    case GraphBoundaryKind::segment_guard: return true;
  }
  return false;
}

bool valid_scalar_semantic(StepScalarSemantic value) {
  return uint32_t(value) <= uint32_t(StepScalarSemantic::dft_due_predicate);
}

bool valid_scalar_type(StepScalarType value) {
  return uint32_t(value) <= uint32_t(StepScalarType::predicate_bit);
}

bool valid_step_program(StepProgram program) {
  return program == StepProgram::ordinary || program == StepProgram::solve_cw;
}

bool checked_span(uint32_t index, uint32_t count, size_t limit) {
  return uint64_t(index) + uint64_t(count) <= uint64_t(limit);
}

void require_span(uint32_t index, uint32_t count, size_t limit, const char *what) {
  if (!checked_span(index, count, limit))
    throw std::invalid_argument(std::string("graph operation ") + what + " span is out of range");
}

void validate_access(const BufferAccess &access, const CpuArrayCatalog *catalog, const char *what) {
  if (access.mode != AccessMode::read && access.mode != AccessMode::write &&
      access.mode != AccessMode::read_write)
    throw std::invalid_argument(std::string(what) + " access mode is invalid");
  if (!is_valid(access.array.id) || !access.array.elements ||
      access.array.offset > std::numeric_limits<size_t>::max() - access.array.elements)
    throw std::invalid_argument(std::string(what) + " access range is invalid");
  if (!catalog) throw std::invalid_argument(std::string(what) + " access has no storage authority");
  if (access.array.id.value >= catalog->size() ||
      access.array.offset + access.array.elements > catalog->spec(access.array.id).elements)
    throw std::invalid_argument(std::string(what) + " access is outside its storage allocation");
}

void validate_operation_accesses(const Operation &operation, const CpuArrayCatalog *catalog) {
  for (const BufferAccess &access : operation.accesses)
    validate_access(access, catalog, "graph operation");
}

void validate_operation_spans(const StepPlan &plan, const Operation &operation) {
  require_span(operation.material_refresh_index, operation.material_refresh_count,
               plan.material_refresh_arrays.size(), "material-refresh");
  require_span(operation.beta_descriptor_index, operation.beta_descriptor_count,
               plan.beta_updates.size(), "beta descriptor");
  require_span(operation.cylindrical_m_descriptor_index, operation.cylindrical_m_descriptor_count,
               plan.cylindrical_m_updates.size(), "cylindrical-m descriptor");
  require_span(operation.cylindrical_origin_action_index, operation.cylindrical_origin_action_count,
               plan.cylindrical_origin_actions.size(), "cylindrical-origin action");
  require_span(operation.polarization_group_index, operation.polarization_group_count,
               plan.polarization_groups.size(), "polarization-group");
  require_span(operation.polarization_subtraction_index, operation.polarization_subtraction_count,
               plan.polarization_subtractions.size(), "polarization-subtraction");
  require_span(operation.magnetic_state_index, operation.magnetic_state_count,
               plan.magnetic_state_arrays.size(), "magnetic-state");
  require_span(operation.legacy_flux_index, operation.legacy_flux_count,
               plan.legacy_flux_updates.size(), "legacy-flux");
  if (uint64_t(operation.source_descriptor_index) + operation.source_descriptor_count >
      std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("graph operation source descriptor span overflows uint32_t");
  if (!operation.source_descriptor_count && operation.source_descriptor_index)
    throw std::invalid_argument("empty graph source descriptor span has a nonzero index");

  const bool material = operation.kind == OpKind::phase_material ||
                        operation.kind == OpKind::update_material_coefficients;
  const bool db = operation.kind == OpKind::update_db;
  const bool polarization = operation.kind == OpKind::update_polarization;
  const bool polarization_subtraction = operation.kind == OpKind::update_eh;
  const bool magnetic = operation.kind == OpKind::restore_magnetic_fields ||
                        operation.kind == OpKind::synchronize_magnetic_fields;
  const bool flux =
      operation.kind == OpKind::update_flux_half || operation.kind == OpKind::update_flux;
  const bool source =
      operation.kind == OpKind::update_eh || operation.kind == OpKind::apply_sources;
  if (!material && (operation.material_refresh_index || operation.material_refresh_count))
    throw std::invalid_argument("graph operation has unrelated material-refresh metadata");
  if (!db &&
      (operation.beta_descriptor_index || operation.beta_descriptor_count ||
       operation.cylindrical_m_descriptor_index || operation.cylindrical_m_descriptor_count ||
       operation.cylindrical_origin_action_index || operation.cylindrical_origin_action_count))
    throw std::invalid_argument("graph operation has unrelated curl action metadata");
  if ((!polarization &&
       (operation.polarization_group_index || operation.polarization_group_count)) ||
      (!polarization_subtraction &&
       (operation.polarization_subtraction_index || operation.polarization_subtraction_count)))
    throw std::invalid_argument("graph operation has unrelated polarization metadata");
  if (!magnetic && (operation.magnetic_state_index || operation.magnetic_state_count))
    throw std::invalid_argument("graph operation has unrelated magnetic-state metadata");
  if (!flux && (operation.legacy_flux_index || operation.legacy_flux_count))
    throw std::invalid_argument("graph operation has unrelated legacy-flux metadata");
  if (!source && (operation.source_descriptor_index || operation.source_descriptor_count))
    throw std::invalid_argument("graph operation has unrelated source metadata");

  size_t descriptor_limit = 0;
  switch (operation.kind) {
    case OpKind::evaluate_source_scalars:
      /* StepPlanBuilder stores the complete SourcePlan::source_times extent
         directly in descriptor_count; those descriptors do not live in a
         StepPlan-owned vector.  A source-free canonical plan retains empty
         evaluation markers. */
      if (operation.descriptor_index != 0 ||
          (operation.descriptor_count == 0 &&
           plan.source_signature != source_plan_signature(SourcePlan())))
        throw std::invalid_argument("graph source-time descriptor span is malformed");
      return;
    case OpKind::update_db: descriptor_limit = plan.db_updates.size(); break;
    case OpKind::update_eh: descriptor_limit = plan.eh_updates.size(); break;
    case OpKind::update_polarization: descriptor_limit = plan.polarization_updates.size(); break;
    case OpKind::update_flux_half:
    case OpKind::update_flux: descriptor_limit = plan.legacy_flux_updates.size(); break;
    case OpKind::update_dft: descriptor_limit = plan.dft_updates.size(); break;
    case OpKind::host_callback: descriptor_limit = plan.host_segments.size(); break;
    case OpKind::pack_state:
    case OpKind::unpack_state: descriptor_limit = plan.cw_state_layout.rows.size(); break;
    default:
      if (operation.descriptor_index || operation.descriptor_count)
        throw std::invalid_argument("graph operation has an unrelated primary descriptor span");
      return;
  }
  require_span(operation.descriptor_index, operation.descriptor_count, descriptor_limit,
               "primary descriptor");
}

void validate_subordinate_spans(const StepPlan &plan) {
  for (const CylindricalOriginAction &action : plan.cylindrical_origin_actions) {
    if (action.kind == CylindricalOriginActionKind::axis_update) {
      if (action.index >= plan.cylindrical_axis_updates.size())
        throw std::invalid_argument("graph cylindrical-axis action is out of range");
    }
    else if (action.kind == CylindricalOriginActionKind::zero_slab) {
      if (action.index >= plan.cylindrical_zero_slabs.size())
        throw std::invalid_argument("graph cylindrical-zero action is out of range");
    }
    else
      throw std::invalid_argument("graph cylindrical-origin action kind is invalid");
  }
  for (const PolarizationUpdate &update : plan.polarization_updates)
    if (update.kind != PolarizationUpdateKind::lorentzian &&
        update.kind != PolarizationUpdateKind::gyrotropic &&
        update.kind != PolarizationUpdateKind::noisy_add)
      throw std::invalid_argument("graph polarization update kind is invalid");
  for (const PolarizationUpdateGroup &group : plan.polarization_groups) {
    if (group.kind != PolarizationGroupKind::recurrence &&
        group.kind != PolarizationGroupKind::multilevel)
      throw std::invalid_argument("graph polarization group kind is invalid");
    require_span(group.recurrence_index, group.recurrence_count, plan.polarization_updates.size(),
                 "group recurrence");
    if (uint64_t(group.recurrence_count) + group.noise_count >
        plan.polarization_updates.size() - group.recurrence_index)
      throw std::invalid_argument("graph polarization group noise span is out of range");
    require_span(group.population_index, group.population_count,
                 plan.multilevel_population_updates.size(), "group population");
    require_span(group.transition_index, group.transition_count,
                 plan.multilevel_transition_updates.size(), "group transition");
  }
  for (const MultilevelPopulationUpdate &update : plan.multilevel_population_updates) {
    require_span(update.gamma_index, update.gamma_count, plan.multilevel_coefficients.size(),
                 "multilevel gamma");
    require_span(update.alpha_index, update.alpha_count, plan.multilevel_coefficients.size(),
                 "multilevel alpha");
    require_span(update.term_index, update.term_count, plan.multilevel_population_terms.size(),
                 "multilevel term");
  }
  for (const LegacyFluxUpdate &update : plan.legacy_flux_updates)
    require_span(update.term_index, update.term_count, plan.legacy_flux_terms.size(),
                 "legacy-flux term");
  for (const MaterialRefreshArray &row : plan.material_refresh_arrays) {
    if (row.family != MaterialRefreshFamily::chi1inv &&
        row.family != MaterialRefreshFamily::conductivity &&
        row.family != MaterialRefreshFamily::condinv)
      throw std::invalid_argument("graph material-refresh family is invalid");
    if (row.d < X || row.d >= NO_DIRECTION)
      throw std::invalid_argument("graph material-refresh direction is invalid");
  }
  for (const MagneticStateArray &row : plan.magnetic_state_arrays)
    if (row.family != MagneticStateFamily::primary && row.family != MagneticStateFamily::u &&
        row.family != MagneticStateFamily::w && row.family != MagneticStateFamily::conductivity &&
        row.family != MagneticStateFamily::bfast)
      throw std::invalid_argument("graph magnetic-state family is invalid");
}

void validate_graph_source_plan(const StepPlan &plan, const CpuArrayCatalog *catalog) {
  if (!valid_step_program(plan.program))
    throw std::invalid_argument("graph StepPlan program is invalid");
  validate_subordinate_spans(plan);
  std::set<uint32_t> scalar_slots;
  size_t source_time_extent = std::numeric_limits<size_t>::max();
  for (const Operation &operation : plan.operations) {
    if (!valid_operation_kind(operation.kind))
      throw std::invalid_argument("graph operation kind is invalid");
    if (!valid_guard_kind(operation.guard.kind))
      throw std::invalid_argument("graph operation guard kind is invalid");
    if (operation.ft < E_stuff || operation.ft > NO_FIELD_TYPE)
      throw std::invalid_argument("graph operation field type is invalid");
    if (!std::isfinite(operation.source_time_offset) || operation.source_time_offset < 0.0 ||
        operation.source_time_offset > 1.0)
      throw std::invalid_argument("graph operation source-time offset is invalid");
    if (operation.kind == OpKind::evaluate_source_scalars && operation.source_time_offset != 0.0 &&
        operation.source_time_offset != 0.5 && operation.source_time_offset != 1.0)
      throw std::invalid_argument("graph source-time offset is not a supported cadence point");
    if (operation.kind != OpKind::evaluate_source_scalars && operation.source_time_offset != 0.0)
      throw std::invalid_argument("non-source graph operation carries a source-time offset");
    if (operation.guard.kind == GuardKind::device_predicate ||
        operation.guard.kind == GuardKind::segment_boundary)
      scalar_slots.insert(operation.guard.scalar_slot);
    else if (operation.guard.scalar_slot)
      throw std::invalid_argument("graph operation guard has an unrelated scalar slot");
    if (operation.guard.kind == GuardKind::graph_variant) {
      if (operation.guard.variant_index > uint32_t(GraphVariantKind::cw_operator))
        throw std::invalid_argument("graph operation variant guard is out of range");
    }
    else if (operation.guard.variant_index)
      throw std::invalid_argument("graph operation guard has an unrelated variant index");
    validate_operation_spans(plan, operation);
    if (operation.kind == OpKind::evaluate_source_scalars) {
      if (source_time_extent == std::numeric_limits<size_t>::max())
        source_time_extent = operation.descriptor_count;
      else if (source_time_extent != operation.descriptor_count)
        throw std::invalid_argument("graph source-time descriptor extents are inconsistent");
    }
    validate_operation_accesses(operation, catalog);
  }
  if (scalar_slots.size() > step_scalar_predicate_word_count * size_t(64))
    throw std::overflow_error("graph scalar predicate slot capacity exceeded");
  for (const DftDescriptor &dft : plan.dft_updates)
    if (dft.decimation_factor < 1)
      throw std::invalid_argument("graph DFT descriptor has invalid decimation");
}

GraphBoundaryKind completion_boundary_kind(const Operation &operation) {
  if (operation.kind == OpKind::update_flux) return GraphBoundaryKind::legacy_flux_publish;
  if (operation.kind == OpKind::finite_value_check) return GraphBoundaryKind::finite_diagnostic;
  return GraphBoundaryKind::none;
}

GraphNodeClass device_class(const Operation &operation) {
  return operation.guard.kind == GuardKind::device_predicate ||
                 operation.kind == OpKind::update_dft ||
                 operation.kind == OpKind::finite_value_check
             ? GraphNodeClass::device_predicate
             : GraphNodeClass::device;
}

void add_core_slot(StepScalarLayout &layout, StepScalarSemantic semantic, StepScalarType type,
                   size_t offset, size_t bytes) {
  if (offset > std::numeric_limits<uint32_t>::max() || bytes > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("StepScalars field offset overflow");
  layout.slots.push_back(StepScalarSlot{semantic, type, 0, uint32_t(offset), 0, uint32_t(bytes)});
}

uint64_t compute_scalar_layout_signature(const StepScalarLayout &layout) {
  uint64_t signature = 0xcbf29ce484222325ull;
  mix(signature, layout.abi_version);
  mix(signature, layout.total_bytes);
  mix(signature, layout.slots.size());
  for (const StepScalarSlot &slot : layout.slots) {
    mix(signature, uint64_t(slot.semantic));
    mix(signature, uint64_t(slot.type));
    mix(signature, slot.semantic_index);
    mix(signature, slot.byte_offset);
    mix(signature, slot.bit_offset);
    mix(signature, slot.byte_size);
  }
  return signature;
}

uint32_t scalar_slot_index(const StepScalarLayout &layout, StepScalarSemantic semantic,
                           uint32_t semantic_index) {
  for (size_t i = 0; i < layout.slots.size(); ++i)
    if (layout.slots[i].semantic == semantic && layout.slots[i].semantic_index == semantic_index) {
      if (i > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("graph scalar slot index overflow");
      return uint32_t(i);
    }
  throw std::logic_error("graph operation refers to a missing scalar slot");
}

GraphOperationRef make_operation_ref(const StepPlan &plan, uint32_t operation_index,
                                     GraphNodeClass node_class,
                                     const StepScalarLayout &scalar_layout,
                                     const GraphLoweringAuthorities &authority) {
  if (operation_index >= plan.operations.size())
    throw std::out_of_range("graph operation index is out of range");
  GraphOperationRef result;
  result.operation_index = operation_index;
  result.node_class = node_class;
  result.operation = plan.operations[operation_index];
  if (!valid_operation_kind(result.operation.kind))
    throw std::invalid_argument("graph operation kind is invalid");
  if (!valid_guard_kind(result.operation.guard.kind))
    throw std::invalid_argument("graph operation guard kind is invalid");
  if (result.operation.guard.kind == GuardKind::device_predicate)
    result.scalar_slots.push_back(scalar_slot_index(
        scalar_layout, StepScalarSemantic::guard_predicate, result.operation.guard.scalar_slot));
  if (result.operation.guard.kind == GuardKind::graph_variant)
    result.scalar_slots.push_back(
        scalar_slot_index(scalar_layout, StepScalarSemantic::graph_variant, 0));
  if (result.operation.kind == OpKind::update_dft) {
    result.scalar_slots.push_back(
        scalar_slot_index(scalar_layout, StepScalarSemantic::dft_timestep, 0));
    std::set<uint32_t> factors;
    const uint64_t end =
        uint64_t(result.operation.descriptor_index) + result.operation.descriptor_count;
    if (end > plan.dft_updates.size())
      throw std::invalid_argument("graph DFT descriptor span is out of range");
    for (size_t i = result.operation.descriptor_index; i < size_t(end); ++i) {
      if (plan.dft_updates[i].decimation_factor < 1)
        throw std::invalid_argument("graph DFT descriptor has invalid decimation");
      factors.insert(uint32_t(plan.dft_updates[i].decimation_factor));
    }
    for (uint32_t factor : factors)
      result.scalar_slots.push_back(
          scalar_slot_index(scalar_layout, StepScalarSemantic::dft_due_predicate, factor));
  }
  if (result.operation.kind == OpKind::update_polarization) {
    result.scalar_slots.push_back(
        scalar_slot_index(scalar_layout, StepScalarSemantic::noisy_counter_time, 0));
    result.scalar_slots.push_back(
        scalar_slot_index(scalar_layout, StepScalarSemantic::noisy_seed_generation, 0));
    result.scalar_slots.push_back(
        scalar_slot_index(scalar_layout, StepScalarSemantic::active_noisy_seed_slot, 0));
  }
  if (result.operation.kind == OpKind::finite_value_check) {
    result.scalar_slots.push_back(
        scalar_slot_index(scalar_layout, StepScalarSemantic::finite_check_mode, 0));
    result.scalar_slots.push_back(
        scalar_slot_index(scalar_layout, StepScalarSemantic::finite_check_due, 0));
  }
  result.signature = hash_graph_operation(result, authority);
  return result;
}

void set_segment_signature(GraphSegment &segment) {
  uint64_t signature = 0xcbf29ce484222325ull;
  mix(signature, uint64_t(segment.variant));
  mix(signature, segment.first_operation);
  mix(signature, segment.operation_count);
  mix(signature, uint64_t(segment.exit_boundary));
  mix(signature, segment.operations.size());
  for (const GraphOperationRef &operation : segment.operations)
    mix(signature, operation.signature);
  segment.signature = signature;
}

void set_boundary_signature(GraphBoundary &boundary) {
  uint64_t signature = 0xcbf29ce484222325ull;
  mix(signature, uint64_t(boundary.kind));
  mix(signature, boundary.first_operation);
  mix(signature, boundary.operation_count);
  mix(signature, uint64_t(boundary.completion_only));
  mix(signature, boundary.operation.signature);
  boundary.signature = signature;
}

void append_boundary(GraphProgram &program, GraphBoundaryKind kind, uint32_t first, uint32_t count,
                     bool completion_only, const GraphOperationRef &operation) {
  if (kind == GraphBoundaryKind::none) return;
  if (program.boundaries.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("graph boundary count overflow");
  GraphBoundary boundary{kind, first, count, completion_only, operation, 0};
  set_boundary_signature(boundary);
  const uint32_t index = uint32_t(program.boundaries.size());
  program.boundaries.push_back(boundary);
  program.schedule.push_back(GraphScheduleEntry{GraphScheduleKind::boundary, index});
}

void flush_segment(GraphProgram &program, GraphSegment &segment) {
  if (segment.operations.empty()) return;
  if (program.segments.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("graph segment count overflow");
  segment.operation_count = uint32_t(segment.operations.size());
  set_segment_signature(segment);
  const uint32_t index = uint32_t(program.segments.size());
  program.segments.push_back(segment);
  program.schedule.push_back(GraphScheduleEntry{GraphScheduleKind::segment, index});
  segment.operations.clear();
  segment.operation_count = 0;
  segment.exit_boundary = GraphBoundaryKind::none;
}

std::vector<uint32_t> selected_operations(const StepPlan &plan, GraphVariantKind variant) {
  std::vector<uint32_t> result;
  if (!valid_graph_variant(variant)) throw std::invalid_argument("graph variant is invalid");
  if (variant == GraphVariantKind::ordinary || variant == GraphVariantKind::cw_operator) {
    if (plan.operations.size() > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("graph source operation count overflow");
    for (uint32_t i = 0; i < plan.operations.size(); ++i)
      result.push_back(i);
    return result;
  }
  if (variant == GraphVariantKind::magnetic_half_step) {
    const uint32_t schedule[] = {
        plan.magnetic_half_step.evaluate_b_sources, plan.magnetic_half_step.update_b,
        plan.magnetic_half_step.apply_b_sources,    plan.magnetic_half_step.transfer_b,
        plan.magnetic_half_step.evaluate_h_sources, plan.magnetic_half_step.update_h,
        plan.magnetic_half_step.transfer_h};
    const OpKind expected_kind[] = {OpKind::evaluate_source_scalars,
                                    OpKind::update_db,
                                    OpKind::apply_sources,
                                    OpKind::transfer_halo,
                                    OpKind::evaluate_source_scalars,
                                    OpKind::update_eh,
                                    OpKind::transfer_halo};
    const field_type expected_ft[] = {NO_FIELD_TYPE, B_stuff, B_stuff, B_stuff,
                                      NO_FIELD_TYPE, H_stuff, H_stuff};
    for (size_t i = 0; i < sizeof(schedule) / sizeof(schedule[0]); ++i) {
      const uint32_t index = schedule[i];
      const bool optional = i == 0 || i == 2 || i == 4;
      if (index == UINT32_MAX) {
        if (!optional)
          throw std::invalid_argument("magnetic graph schedule omits a required operation");
        continue;
      }
      if (index >= plan.operations.size() || plan.operations[index].kind != expected_kind[i] ||
          plan.operations[index].ft != expected_ft[i])
        throw std::invalid_argument("magnetic graph schedule has the wrong operation semantics");
      result.push_back(index);
    }
    return result;
  }
  for (uint32_t i = 0; i < plan.operations.size(); ++i)
    if (plan.operations[i].kind == OpKind::restore_magnetic_fields) result.push_back(i);
  if (result.size() != 1)
    throw std::invalid_argument("magnetic-restore graph variant requires one restore operation");
  return result;
}

const char *boundary_name(GraphBoundaryKind kind) {
  switch (kind) {
    case GraphBoundaryKind::none: return "none";
    case GraphBoundaryKind::source_evaluation: return "source_evaluation";
    case GraphBoundaryKind::host_callback: return "host_callback";
    case GraphBoundaryKind::material_phase: return "material_phase";
    case GraphBoundaryKind::material_refresh: return "material_refresh";
    case GraphBoundaryKind::remote_halo: return "remote_halo";
    case GraphBoundaryKind::legacy_flux_publish: return "legacy_flux_publish";
    case GraphBoundaryKind::finite_diagnostic: return "finite_diagnostic";
    case GraphBoundaryKind::time_update: return "time_update";
    case GraphBoundaryKind::magnetic_transition: return "magnetic_transition";
    case GraphBoundaryKind::segment_guard: return "segment_guard";
  }
  return "invalid";
}

} // namespace

const char *graph_execution_mode_name(GraphExecutionMode mode) {
  switch (mode) {
    case GraphExecutionMode::automatic: return "auto";
    case GraphExecutionMode::eager: return "eager";
    case GraphExecutionMode::required: return "required";
  }
  return "invalid";
}

GraphExecutionMode parse_graph_execution_mode(const char *value) {
  if (!value || !*value || std::strcmp(value, "auto") == 0) return GraphExecutionMode::automatic;
  if (std::strcmp(value, "eager") == 0) return GraphExecutionMode::eager;
  if (std::strcmp(value, "required") == 0) return GraphExecutionMode::required;
  throw std::invalid_argument("invalid NVIDIA graph mode (expected auto, eager, or required)");
}

GraphModeResolution
resolve_collective_graph_execution_mode(const std::vector<GraphRankModeSupport> &ranks) {
  if (ranks.empty()) throw std::invalid_argument("graph mode resolution has no ranks");
  const GraphExecutionMode requested = ranks.front().requested;
  if (!valid_graph_execution_mode(requested))
    throw std::invalid_argument("graph mode resolution received an invalid mode");
  bool all_supported = true;
  for (const GraphRankModeSupport &rank : ranks) {
    if (!valid_graph_execution_mode(rank.requested) || rank.requested != requested)
      throw std::invalid_argument("graph execution mode differs across ranks");
    all_supported = all_supported && rank.runtime_capture_supported && rank.program_graphable;
  }
  if (requested == GraphExecutionMode::eager) return GraphModeResolution{requested, false};
  if (requested == GraphExecutionMode::required && !all_supported)
    throw std::runtime_error("required CUDA graph mode is unsupported on at least one rank");
  return GraphModeResolution{requested, all_supported};
}

bool operator==(const GraphArrayIdentity &a, const GraphArrayIdentity &b) {
  return a.source_id == b.source_id && a.key == b.key && a.role == b.role &&
         a.element_type == b.element_type && a.storage == b.storage && a.elements == b.elements &&
         a.alignment == b.alignment &&
         a.classification_provisional == b.classification_provisional &&
         a.classification_elided == b.classification_elided && a.signature == b.signature;
}

bool operator==(const GraphHaloRow &a, const GraphHaloRow &b) {
  return a.ft == b.ft && a.chunks == b.chunks && a.phase == b.phase && a.peer_rank == b.peer_rank &&
         a.tag == b.tag && a.route == b.route && a.sequence_index == b.sequence_index &&
         a.block_offset == b.block_offset && a.block_elements == b.block_elements &&
         a.signature == b.signature;
}

bool operator==(const GraphHaloDisposition &a, const GraphHaloDisposition &b) {
  return a.operation_index == b.operation_index && a.ft == b.ft && a.row_index == b.row_index &&
         a.row_count == b.row_count && a.entirely_local_canonical == b.entirely_local_canonical &&
         a.signature == b.signature;
}

bool operator==(const GraphZeroRow &a, const GraphZeroRow &b) {
  return a.ft == b.ft && a.chunk == b.chunk && a.elements == b.elements &&
         a.signature == b.signature;
}

bool operator==(const GraphZeroDisposition &a, const GraphZeroDisposition &b) {
  return a.operation_index == b.operation_index && a.ft == b.ft && a.row_index == b.row_index &&
         a.row_count == b.row_count && a.signature == b.signature;
}

bool operator==(const GraphHostInterval &a, const GraphHostInterval &b) {
  return a.marker_operation == b.marker_operation &&
         a.first_covered_operation == b.first_covered_operation &&
         a.covered_operation_count == b.covered_operation_count &&
         a.host_segment_index == b.host_segment_index && a.signature == b.signature;
}

bool operator==(const GraphRemoteOverlap &a, const GraphRemoteOverlap &b) {
  return a.halo_operation == b.halo_operation && a.update_operation == b.update_operation &&
         a.dependency_signature == b.dependency_signature && a.signature == b.signature;
}

uint64_t compute_graph_remote_overlap_signature(const GraphRemoteOverlap &overlap) {
  uint64_t signature = 0xcbf29ce484222325ull;
  mix(signature, overlap.halo_operation);
  mix(signature, overlap.update_operation);
  mix(signature, overlap.dependency_signature);
  return signature;
}

uint64_t compute_graph_lowering_authorities_signature(const GraphLoweringAuthorities &authority) {
  uint64_t signature = 0xcbf29ce484222325ull;
  mix(signature, authority.halo_signature);
  mix(signature, authority.cw_stable_signature);
  mix(signature, authority.array_identities.size());
  std::vector<uint64_t> array_signatures;
  for (const GraphArrayIdentity &identity : authority.array_identities)
    array_signatures.push_back(identity.signature);
  std::sort(array_signatures.begin(), array_signatures.end());
  for (uint64_t identity : array_signatures)
    mix(signature, identity);
  mix(signature, authority.halo_rows.size());
  for (const GraphHaloRow &row : authority.halo_rows) {
    mix(signature, uint64_t(row.ft));
    mix(signature, uint64_t(row.chunks.first));
    mix(signature, uint64_t(row.chunks.second));
    mix(signature, uint64_t(row.phase));
    mix(signature, uint64_t(row.peer_rank));
    mix(signature, uint64_t(row.tag));
    mix(signature, uint64_t(row.route));
    mix(signature, row.sequence_index);
    mix(signature, row.block_offset);
    mix(signature, row.block_elements);
    mix(signature, row.signature);
  }
  mix(signature, authority.halo_dispositions.size());
  for (const GraphHaloDisposition &disposition : authority.halo_dispositions) {
    mix(signature, disposition.operation_index);
    mix(signature, uint64_t(disposition.ft));
    mix(signature, disposition.row_index);
    mix(signature, disposition.row_count);
    mix(signature, uint64_t(disposition.entirely_local_canonical));
    mix(signature, disposition.signature);
  }
  mix(signature, authority.zero_rows.size());
  for (const GraphZeroRow &row : authority.zero_rows) {
    mix(signature, uint64_t(row.ft));
    mix(signature, row.chunk);
    mix(signature, row.elements);
    mix(signature, row.signature);
  }
  mix(signature, authority.zero_dispositions.size());
  for (const GraphZeroDisposition &disposition : authority.zero_dispositions) {
    mix(signature, disposition.operation_index);
    mix(signature, uint64_t(disposition.ft));
    mix(signature, disposition.row_index);
    mix(signature, disposition.row_count);
    mix(signature, disposition.signature);
  }
  mix(signature, authority.host_intervals.size());
  for (const GraphHostInterval &interval : authority.host_intervals) {
    mix(signature, interval.marker_operation);
    mix(signature, interval.first_covered_operation);
    mix(signature, interval.covered_operation_count);
    mix(signature, interval.host_segment_index);
    mix(signature, interval.signature);
  }
  mix(signature, authority.remote_overlaps.size());
  for (const GraphRemoteOverlap &overlap : authority.remote_overlaps) {
    mix(signature, overlap.halo_operation);
    mix(signature, overlap.update_operation);
    mix(signature, overlap.dependency_signature);
    mix(signature, overlap.signature);
  }
  return signature;
}

GraphLoweringAuthorities build_graph_lowering_authorities(const StepPlan &plan,
                                                          const halo_plan_set *halos,
                                                          const CpuArrayCatalog *catalog,
                                                          int field_interleave,
                                                          const CwPlan *cw_plan) {
  if (plan.signature != compute_step_plan_signature(plan))
    throw std::invalid_argument("graph lowering received a stale StepPlan signature");
  std::string host_error;
  if (!validate_host_segments(plan, &host_error))
    throw std::invalid_argument(host_error.empty() ? "invalid host-covered operation intervals"
                                                   : host_error);
  if (plan.program == StepProgram::solve_cw) {
    if (!cw_plan) throw std::invalid_argument("CW graph lowering requires a CwPlan");
    if (cw_plan->step_plan_signature != plan.signature ||
        cw_plan->signature != compute_cw_plan_signature(*cw_plan))
      throw std::invalid_argument("CW graph lowering received a stale or mismatched CwPlan");
  }
  else if (cw_plan) {
    throw std::invalid_argument("ordinary graph lowering must not carry a CwPlan");
  }
  if (field_interleave != 1 && field_interleave != 2)
    throw std::invalid_argument("graph lowering field interleave must be one or two");
  if (halos && !catalog)
    throw std::invalid_argument("graph lowering live halo authority requires a storage catalog");

  GraphLoweringAuthorities result;
  result.step_plan_signature = plan.signature;
  result.cw_plan_signature = cw_plan ? cw_plan->signature : 0;
  if (catalog) {
    if (catalog->size() > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("graph array authority count overflow");
    for (uint32_t i = 0; i < catalog->size(); ++i)
      result.array_identities.push_back(make_array_identity(*catalog, ArrayId{i}));
  }
  validate_graph_source_plan(plan, catalog);
  if (cw_plan) {
    if (cw_plan->rhs_source_count != cw_plan->rhs_sources.size() ||
        cw_plan->final_dft_count != cw_plan->final_dfts.size() ||
        cw_plan->rhs_stages.size() != 2)
      throw std::invalid_argument("CW graph authority has inconsistent aggregate counts");
    bool saw_source_evaluation = false;
    for (const Operation &operation : plan.operations) {
      if (operation.kind != OpKind::evaluate_source_scalars) continue;
      if (operation.descriptor_index != 0 ||
          operation.descriptor_count != cw_plan->source_time_count)
        throw std::invalid_argument("CW graph authority has a mismatched source-time extent");
      saw_source_evaluation = true;
    }
    if (cw_plan->source_time_count && !saw_source_evaluation)
      throw std::invalid_argument("CW graph authority omits source-time evaluation");
    std::vector<uint8_t> rhs_source_seen(cw_plan->rhs_sources.size(), 0);
    std::set<uint32_t> source_descriptors;
    uint32_t expected_source_index = 0;
    const field_type expected_stage_ft[] = {B_stuff, D_stuff};
    const double expected_stage_offset[] = {0.0, 0.5};
    for (size_t stage_index = 0; stage_index < cw_plan->rhs_stages.size(); ++stage_index) {
      const CwRhsStage &stage = cw_plan->rhs_stages[stage_index];
      const field_type constitutive_ft = stage.ft == B_stuff ? H_stuff : E_stuff;
      if (stage.ft != expected_stage_ft[stage_index] ||
          stage.source_time_offset != expected_stage_offset[stage_index] ||
          stage.source_time_index != 0 || stage.source_time_count != cw_plan->source_time_count ||
          stage.source_index != expected_source_index ||
          uint64_t(stage.source_index) + stage.source_count > cw_plan->rhs_sources.size())
        throw std::invalid_argument("CW graph authority has a malformed RHS stage");
      validate_cw_operation_ref(plan, stage.boundary, OpKind::transfer_halo, stage.ft,
                                "CW RHS boundary");
      validate_cw_operation_ref(plan, stage.constitutive, OpKind::update_eh, constitutive_ft,
                                "CW RHS constitutive");
      for (uint32_t i = 0; i < stage.source_count; ++i) {
        const size_t source_index = size_t(stage.source_index) + i;
        if (rhs_source_seen[source_index]++)
          throw std::invalid_argument("CW graph authority has overlapping RHS source spans");
        const CwRhsSourceDescriptor &source = cw_plan->rhs_sources[source_index];
        if (source.mode != CwRhsSourceMode::primary_subtract_current_dt_including_integrated ||
            !source_descriptors.insert(source.source_descriptor_index).second)
          throw std::invalid_argument("CW graph authority has an invalid RHS source row");
      }
      expected_source_index += stage.source_count;
      for (const BufferAccess &access : stage.accesses)
        validate_access(access, catalog, "CW RHS stage");
    }
    if (cw_plan->rhs_stages[0].boundary.operation_index >=
            cw_plan->rhs_stages[0].constitutive.operation_index ||
        cw_plan->rhs_stages[0].constitutive.operation_index >=
            cw_plan->rhs_stages[1].boundary.operation_index ||
        cw_plan->rhs_stages[1].boundary.operation_index >=
            cw_plan->rhs_stages[1].constitutive.operation_index)
      throw std::invalid_argument("CW graph authority has out-of-order RHS operation refs");
    if (expected_source_index != cw_plan->rhs_sources.size())
      throw std::invalid_argument("CW graph authority does not cover every RHS source in order");
    for (uint8_t seen : rhs_source_seen)
      if (!seen)
        throw std::invalid_argument("CW graph authority contains an uncovered RHS source");
    validate_cw_operation_ref(plan, cw_plan->unpack.first_boundary, OpKind::transfer_halo,
                              D_stuff, "CW unpack first boundary");
    validate_cw_operation_ref(plan, cw_plan->unpack.constitutive, OpKind::update_eh, E_stuff,
                              "CW unpack constitutive");
    validate_cw_operation_ref(plan, cw_plan->unpack.second_boundary, OpKind::transfer_halo,
                              E_stuff, "CW unpack second boundary");
    if (cw_plan->unpack.first_boundary != cw_plan->rhs_stages[1].boundary ||
        cw_plan->unpack.constitutive != cw_plan->rhs_stages[1].constitutive ||
        cw_plan->unpack.constitutive.operation_index >=
            cw_plan->unpack.second_boundary.operation_index)
      throw std::invalid_argument("CW graph authority has out-of-order unpack operation refs");
    for (size_t i = 0; i < cw_plan->final_dfts.size(); ++i) {
      const CwDftDescriptorRef &dft = cw_plan->final_dfts[i];
      /* solve_cw deliberately has no update_dft operation.  descriptor_index
         addresses the separately fingerprinted DescriptorSet DFT table. */
      if (dft.descriptor_index != i || dft.chunk < 0 || int(dft.c) < 0 ||
          int(dft.c) >= NUM_FIELD_COMPONENTS || dft.decimation_factor < 1 ||
          dft.due_scalar_slot >= step_scalar_predicate_word_count * uint32_t(64))
        throw std::invalid_argument("CW graph authority has a malformed final DFT reference");
    }
    for (const BufferAccess &access : cw_plan->rhs_accesses)
      validate_access(access, catalog, "CW RHS");
    for (const BufferAccess &access : cw_plan->unpack_accesses)
      validate_access(access, catalog, "CW unpack");
    for (const BufferAccess &access : cw_plan->final_dft_accesses)
      validate_access(access, catalog, "CW final DFT");
    result.cw_stable_signature = stable_cw_plan_signature(*cw_plan, result);
  }

  for (size_t i = 0; i < plan.operations.size(); ++i) {
    const Operation &operation = plan.operations[i];
    if (operation.kind == OpKind::host_callback) {
      if (operation.descriptor_count != 1 ||
          operation.descriptor_index >= plan.host_segments.size())
        throw std::invalid_argument("graph lowering host marker has an invalid segment");
      const HostSegment &segment = plan.host_segments[operation.descriptor_index];
      uint64_t signature = 0xcbf29ce484222325ull;
      mix(signature, i);
      mix(signature, segment.operation_index);
      mix(signature, segment.operation_count);
      mix(signature, operation.descriptor_index);
      result.host_intervals.push_back(GraphHostInterval{uint32_t(i), segment.operation_index,
                                                        segment.operation_count,
                                                        operation.descriptor_index, signature});
    }
  }

  uint64_t halo_signature = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < plan.operations.size(); ++i) {
    if (plan.operations[i].kind != OpKind::transfer_halo) continue;
    if (i > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("graph halo operation index overflow");
    if (result.halo_rows.size() > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("graph halo row index overflow");
    GraphHaloDisposition disposition;
    disposition.operation_index = uint32_t(i);
    disposition.ft = plan.operations[i].ft;
    disposition.row_index = uint32_t(result.halo_rows.size());
    disposition.row_count = 0;
    disposition.entirely_local_canonical = true;
    uint64_t disposition_signature = 0xcbf29ce484222325ull;
    mix(disposition_signature, disposition.operation_index);
    mix(disposition_signature, uint64_t(disposition.ft));
    if (halos) {
      for (const HaloPlan &source : halos->plans) {
        if (source.ft != disposition.ft || !source.block_elements) continue;
        if (disposition.row_count == std::numeric_limits<uint32_t>::max())
          throw std::overflow_error("graph halo row count overflow");
        HaloPlan canonical;
        std::string why;
        validate_source_halo(source, halos->arrays, halos->host_arrays);
        if (!remap_halo_plan(source, halos->arrays, halos->host_arrays, *catalog, field_interleave,
                             canonical, why))
          throw std::invalid_argument(std::string("cannot remap graph halo plan: ") + why);
        const GraphHaloRow row =
            make_halo_row(source, canonical, halos->arrays, halos->host_arrays, *catalog);
        result.halo_rows.push_back(row);
        ++disposition.row_count;
        disposition.entirely_local_canonical =
            disposition.entirely_local_canonical && row.route == GraphHaloRoute::local_device;
        mix(disposition_signature, row.signature);
      }
    }
    else {
      /* No halo authority is not proof of locality.  A zero-row authoritative
         set is represented by a non-null empty halo_plan_set. */
      disposition.entirely_local_canonical = false;
    }
    mix(disposition_signature, disposition.row_count);
    mix(disposition_signature, uint64_t(disposition.entirely_local_canonical));
    disposition.signature = disposition_signature;
    result.halo_dispositions.push_back(disposition);
    mix(halo_signature, disposition.signature);
  }

  for (size_t operation_index = 0; operation_index < plan.operations.size(); ++operation_index) {
    const Operation &operation = plan.operations[operation_index];
    /* The canonical CPU StepPlan keeps metal zeroing fused into transfer_halo,
       and the NVIDIA lowering expands that operation back into zero/pack/
       transfer/unpack.  Bind ZeroPlan topology to that real operation.  Keep
       accepting zero_boundary as Phase-2 vocabulary for explicit plans. */
    if (operation.kind != OpKind::transfer_halo && operation.kind != OpKind::zero_boundary)
      continue;
    if (operation_index > std::numeric_limits<uint32_t>::max() ||
        result.zero_rows.size() > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("graph zero authority index overflow");
    GraphZeroDisposition disposition;
    disposition.operation_index = uint32_t(operation_index);
    disposition.ft = operation.ft;
    disposition.row_index = uint32_t(result.zero_rows.size());
    disposition.row_count = 0;
    uint64_t disposition_signature = 0xcbf29ce484222325ull;
    mix(disposition_signature, disposition.operation_index);
    mix(disposition_signature, uint64_t(disposition.ft));
    if (halos) {
      if (operation.ft < 0 || operation.ft >= NUM_FIELD_TYPES)
        throw std::invalid_argument("graph zero operation has invalid field type");
      const std::vector<ZeroPlan> &source_rows = halos->zeros[operation.ft];
      for (size_t chunk = 0; chunk < source_rows.size(); ++chunk) {
        const ZeroPlan &source = source_rows[chunk];
        if (source.slabs.empty() && source.residue.empty()) continue;
        if (chunk > std::numeric_limits<uint32_t>::max() ||
            disposition.row_count == std::numeric_limits<uint32_t>::max())
          throw std::overflow_error("graph zero row count overflow");
        ZeroPlan canonical;
        std::string why;
        validate_source_zero(source, halos->arrays);
        if (!remap_zero_plan(source, halos->arrays, *catalog, canonical, why))
          throw std::invalid_argument(std::string("cannot remap graph zero plan: ") + why);
        const GraphZeroRow row = make_zero_row(operation.ft, uint32_t(chunk), source, canonical,
                                               halos->arrays, *catalog);
        result.zero_rows.push_back(row);
        ++disposition.row_count;
        mix(disposition_signature, row.signature);
      }
    }
    mix(disposition_signature, disposition.row_count);
    disposition.signature = disposition_signature;
    result.zero_dispositions.push_back(disposition);
    mix(halo_signature, disposition.signature);
  }
  result.halo_signature = halo_signature;
  result.signature = compute_graph_lowering_authorities_signature(result);
  return result;
}

bool validate_graph_lowering_authorities(const StepPlan &plan,
                                         const GraphLoweringAuthorities &authority,
                                         const halo_plan_set *halos, const CpuArrayCatalog *catalog,
                                         int field_interleave, const CwPlan *cw_plan,
                                         std::string *error) {
  if (error) error->clear();
  try {
    if (plan.signature != compute_step_plan_signature(plan))
      throw std::invalid_argument("graph lowering StepPlan signature is stale");
    if (authority.step_plan_signature != plan.signature)
      throw std::invalid_argument("graph lowering authority belongs to another StepPlan");
    if (authority.signature != compute_graph_lowering_authorities_signature(authority))
      throw std::invalid_argument("graph lowering authority signature is stale");
    std::string host_error;
    if (!validate_host_segments(plan, &host_error)) throw std::invalid_argument(host_error);
    if ((plan.program == StepProgram::solve_cw) != (authority.cw_plan_signature != 0))
      throw std::invalid_argument("graph lowering CW authority presence is invalid");
    if (cw_plan && (cw_plan->signature != compute_cw_plan_signature(*cw_plan) ||
                    cw_plan->signature != authority.cw_plan_signature ||
                    cw_plan->step_plan_signature != plan.signature))
      throw std::invalid_argument("graph lowering CwPlan authority is stale");
    if (halos || catalog || cw_plan) {
      const GraphLoweringAuthorities expected =
          build_graph_lowering_authorities(plan, halos, catalog, field_interleave, cw_plan);
      if (authority.array_identities != expected.array_identities ||
          authority.halo_rows != expected.halo_rows ||
          authority.halo_dispositions != expected.halo_dispositions ||
          authority.zero_rows != expected.zero_rows ||
          authority.zero_dispositions != expected.zero_dispositions ||
          authority.host_intervals != expected.host_intervals ||
          authority.halo_signature != expected.halo_signature ||
          authority.cw_plan_signature != expected.cw_plan_signature ||
          authority.cw_stable_signature != expected.cw_stable_signature)
        throw std::invalid_argument("graph lowering authorities differ from live inputs");
    }
    std::vector<uint8_t> overlap_seen(plan.operations.size(), 0);
    for (const GraphRemoteOverlap &overlap : authority.remote_overlaps) {
      if (overlap.halo_operation >= plan.operations.size() ||
          overlap.update_operation != overlap.halo_operation + 1 ||
          overlap.update_operation >= plan.operations.size() ||
          plan.operations[overlap.halo_operation].kind != OpKind::transfer_halo ||
          (plan.operations[overlap.update_operation].kind != OpKind::update_db &&
           plan.operations[overlap.update_operation].kind != OpKind::update_eh) ||
          overlap_seen[overlap.halo_operation] || overlap_seen[overlap.update_operation])
        throw std::invalid_argument("graph remote-overlap authority is malformed");
      if (!overlap.dependency_signature ||
          overlap.signature != compute_graph_remote_overlap_signature(overlap))
        throw std::invalid_argument("graph remote-overlap authority signature is stale");
      overlap_seen[overlap.halo_operation] = 1;
      overlap_seen[overlap.update_operation] = 1;
    }
    std::vector<uint8_t> covered(plan.operations.size(), 0);
    std::vector<GraphHostInterval> expected_intervals;
    for (size_t i = 0; i < plan.operations.size(); ++i) {
      const Operation &operation = plan.operations[i];
      if (operation.kind != OpKind::host_callback) continue;
      if (operation.descriptor_count != 1 ||
          operation.descriptor_index >= plan.host_segments.size())
        throw std::invalid_argument("graph host marker has an invalid segment span");
      const HostSegment &segment = plan.host_segments[operation.descriptor_index];
      uint64_t signature = 0xcbf29ce484222325ull;
      mix(signature, i);
      mix(signature, segment.operation_index);
      mix(signature, segment.operation_count);
      mix(signature, operation.descriptor_index);
      expected_intervals.push_back(GraphHostInterval{uint32_t(i), segment.operation_index,
                                                     segment.operation_count,
                                                     operation.descriptor_index, signature});
    }
    if (authority.host_intervals != expected_intervals)
      throw std::invalid_argument("graph host-covered intervals differ from StepPlan");
    for (const GraphHostInterval &interval : authority.host_intervals) {
      if (interval.marker_operation >= plan.operations.size() ||
          plan.operations[interval.marker_operation].kind != OpKind::host_callback ||
          interval.first_covered_operation != interval.marker_operation + 1 ||
          uint64_t(interval.first_covered_operation) + interval.covered_operation_count >
              plan.operations.size())
        throw std::invalid_argument("graph host-covered interval is malformed");
      for (uint32_t i = 0; i < interval.covered_operation_count; ++i)
        if (covered[size_t(interval.first_covered_operation) + i]++)
          throw std::invalid_argument("graph host-covered intervals overlap");
    }
    std::vector<uint8_t> halo_seen(plan.operations.size(), 0);
    std::vector<uint8_t> halo_row_seen(authority.halo_rows.size(), 0);
    for (const GraphHaloDisposition &disposition : authority.halo_dispositions) {
      if (disposition.operation_index >= plan.operations.size() ||
          plan.operations[disposition.operation_index].kind != OpKind::transfer_halo ||
          plan.operations[disposition.operation_index].ft != disposition.ft ||
          uint64_t(disposition.row_index) + disposition.row_count > authority.halo_rows.size() ||
          halo_seen[disposition.operation_index]++)
        throw std::invalid_argument("graph halo disposition is malformed");
      bool local = true;
      uint64_t signature = 0xcbf29ce484222325ull;
      mix(signature, disposition.operation_index);
      mix(signature, uint64_t(disposition.ft));
      for (uint32_t i = 0; i < disposition.row_count; ++i) {
        const size_t row_index = size_t(disposition.row_index) + i;
        if (halo_row_seen[row_index]++)
          throw std::invalid_argument("graph halo authority rows overlap");
        const GraphHaloRow &row = authority.halo_rows[row_index];
        if (row.ft != disposition.ft)
          throw std::invalid_argument("graph halo row has the wrong field type");
        if (row.route != GraphHaloRoute::local_device && row.route != GraphHaloRoute::remote_host &&
            row.route != GraphHaloRoute::host_owned)
          throw std::invalid_argument("graph halo row has an invalid route");
        local = local && row.route == GraphHaloRoute::local_device;
        mix(signature, row.signature);
      }
      mix(signature, disposition.row_count);
      mix(signature, uint64_t(local));
      if (local != disposition.entirely_local_canonical || signature != disposition.signature)
        throw std::invalid_argument("graph halo disposition signature is stale");
    }
    for (size_t i = 0; i < plan.operations.size(); ++i)
      if (plan.operations[i].kind == OpKind::transfer_halo && !halo_seen[i])
        throw std::invalid_argument("graph lowering omitted a transfer-halo disposition");
    for (uint8_t seen : halo_row_seen)
      if (!seen) throw std::invalid_argument("graph lowering contains an orphan halo row");
    std::vector<uint8_t> zero_seen(plan.operations.size(), 0);
    std::vector<uint8_t> zero_row_seen(authority.zero_rows.size(), 0);
    for (const GraphZeroDisposition &disposition : authority.zero_dispositions) {
      if (disposition.operation_index >= plan.operations.size() ||
          (plan.operations[disposition.operation_index].kind != OpKind::transfer_halo &&
           plan.operations[disposition.operation_index].kind != OpKind::zero_boundary) ||
          plan.operations[disposition.operation_index].ft != disposition.ft ||
          uint64_t(disposition.row_index) + disposition.row_count > authority.zero_rows.size() ||
          zero_seen[disposition.operation_index]++)
        throw std::invalid_argument("graph zero disposition is malformed");
      uint64_t signature = 0xcbf29ce484222325ull;
      mix(signature, disposition.operation_index);
      mix(signature, uint64_t(disposition.ft));
      for (uint32_t i = 0; i < disposition.row_count; ++i) {
        const size_t row_index = size_t(disposition.row_index) + i;
        if (zero_row_seen[row_index]++)
          throw std::invalid_argument("graph zero authority rows overlap");
        const GraphZeroRow &row = authority.zero_rows[row_index];
        if (row.ft != disposition.ft || !row.elements)
          throw std::invalid_argument("graph zero row is malformed");
        mix(signature, row.signature);
      }
      mix(signature, disposition.row_count);
      if (signature != disposition.signature)
        throw std::invalid_argument("graph zero disposition signature is stale");
    }
    for (size_t i = 0; i < plan.operations.size(); ++i)
      if ((plan.operations[i].kind == OpKind::transfer_halo ||
           plan.operations[i].kind == OpKind::zero_boundary) &&
          !zero_seen[i])
        throw std::invalid_argument("graph lowering omitted a fused-zero disposition");
    for (uint8_t seen : zero_row_seen)
      if (!seen) throw std::invalid_argument("graph lowering contains an orphan zero row");
    return true;
  }
  catch (const std::exception &exception) {
    if (error) *error = exception.what();
    return false;
  }
}

StepScalarLayout build_step_scalar_layout(const StepPlan &plan) {
  static_assert(std::is_standard_layout<StepScalars>::value, "StepScalars must be standard layout");
  static_assert(std::is_trivially_copyable<StepScalars>::value,
                "StepScalars must be trivially copyable");
  if (sizeof(StepScalars) > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("StepScalars byte size overflow");
  StepScalarLayout layout;
  layout.total_bytes = uint32_t(sizeof(StepScalars));
#define ADD_SLOT(field, semantic, type)                                                            \
  add_core_slot(layout, StepScalarSemantic::semantic, StepScalarType::type,                        \
                offsetof(StepScalars, field), sizeof(((StepScalars *)0)->field))
  ADD_SLOT(abi_version, abi_version, u32);
  ADD_SLOT(byte_size, byte_size, u32);
  ADD_SLOT(entry_timestep, entry_timestep, i64);
  ADD_SLOT(post_increment_timestep, post_increment_timestep, i64);
  ADD_SLOT(noisy_counter_time, noisy_counter_time, u64);
  add_core_slot(layout, StepScalarSemantic::source_time_0, StepScalarType::f64,
                offsetof(StepScalars, source_times), sizeof(double));
  add_core_slot(layout, StepScalarSemantic::source_time_half, StepScalarType::f64,
                offsetof(StepScalars, source_times) + sizeof(double), sizeof(double));
  add_core_slot(layout, StepScalarSemantic::source_time_1, StepScalarType::f64,
                offsetof(StepScalars, source_times) + 2 * sizeof(double), sizeof(double));
  ADD_SLOT(batch_ordinal, batch_ordinal, u64);
  ADD_SLOT(batch_count, batch_count, u64);
  ADD_SLOT(dft_timestep, dft_timestep, i64);
  ADD_SLOT(noisy_seed_generation, noisy_seed_generation, u64);
  ADD_SLOT(active_noisy_seed_slot, active_noisy_seed_slot, i32);
  ADD_SLOT(finite_check_mode, finite_check_mode, u32);
  ADD_SLOT(finite_check_due, finite_check_due, u32);
  ADD_SLOT(graph_variant, graph_variant, u32);
  ADD_SLOT(material_phase_result, material_phase_result, u32);
#undef ADD_SLOT

  std::set<uint32_t> guard_slots;
  std::set<uint32_t> dft_factors;
  for (const Operation &operation : plan.operations) {
    if (operation.guard.kind == GuardKind::device_predicate)
      guard_slots.insert(operation.guard.scalar_slot);
    if (operation.kind != OpKind::update_dft) continue;
    const uint64_t end = uint64_t(operation.descriptor_index) + operation.descriptor_count;
    if (end > plan.dft_updates.size())
      throw std::invalid_argument("StepScalarLayout DFT span is out of range");
    for (size_t i = operation.descriptor_index; i < size_t(end); ++i) {
      if (plan.dft_updates[i].decimation_factor < 1)
        throw std::invalid_argument("StepScalarLayout DFT decimation is invalid");
      dft_factors.insert(uint32_t(plan.dft_updates[i].decimation_factor));
    }
  }
  const uint64_t predicate_count = uint64_t(guard_slots.size()) + dft_factors.size();
  if (predicate_count > step_scalar_predicate_word_count * uint64_t(64))
    throw std::overflow_error("StepScalarLayout predicate capacity exceeded");
  uint32_t bit = 0;
  auto add_predicate = [&](StepScalarSemantic semantic, uint32_t identity) {
    const uint32_t word = bit / 64;
    const uint32_t in_word = bit % 64;
    const size_t offset = offsetof(StepScalars, predicate_words) + size_t(word) * sizeof(uint64_t);
    layout.slots.push_back(StepScalarSlot{semantic, StepScalarType::predicate_bit, identity,
                                          uint32_t(offset), in_word, sizeof(uint64_t)});
    ++bit;
  };
  for (uint32_t slot : guard_slots)
    add_predicate(StepScalarSemantic::guard_predicate, slot);
  for (uint32_t factor : dft_factors)
    add_predicate(StepScalarSemantic::dft_due_predicate, factor);
  layout.signature = compute_scalar_layout_signature(layout);
  return layout;
}

bool operator==(const StepScalarSlot &a, const StepScalarSlot &b) {
  return a.semantic == b.semantic && a.type == b.type && a.semantic_index == b.semantic_index &&
         a.byte_offset == b.byte_offset && a.bit_offset == b.bit_offset &&
         a.byte_size == b.byte_size;
}

bool operator==(const StepScalarLayout &a, const StepScalarLayout &b) {
  return a.abi_version == b.abi_version && a.total_bytes == b.total_bytes && a.slots == b.slots &&
         a.signature == b.signature;
}

bool operator==(const GraphOperationRef &a, const GraphOperationRef &b) {
  return a.operation_index == b.operation_index && a.node_class == b.node_class &&
         same_operation(a.operation, b.operation) && a.scalar_slots == b.scalar_slots &&
         a.signature == b.signature;
}

bool operator==(const GraphSegment &a, const GraphSegment &b) {
  return a.variant == b.variant && a.first_operation == b.first_operation &&
         a.operation_count == b.operation_count && a.operations == b.operations &&
         a.exit_boundary == b.exit_boundary && a.signature == b.signature;
}

bool operator==(const GraphBoundary &a, const GraphBoundary &b) {
  return a.kind == b.kind && a.first_operation == b.first_operation &&
         a.operation_count == b.operation_count && a.completion_only == b.completion_only &&
         a.operation == b.operation && a.signature == b.signature;
}

bool operator==(const GraphScheduleEntry &a, const GraphScheduleEntry &b) {
  return a.kind == b.kind && a.index == b.index;
}

bool operator==(const GraphProgram &a, const GraphProgram &b) {
  return a.program == b.program && a.variant == b.variant &&
         a.fallback_policy == b.fallback_policy && a.step_plan_signature == b.step_plan_signature &&
         a.authority_signature == b.authority_signature &&
         a.cw_plan_signature == b.cw_plan_signature && a.scalar_layout == b.scalar_layout &&
         a.segments == b.segments && a.boundaries == b.boundaries && a.schedule == b.schedule &&
         a.signature == b.signature;
}

bool graph_required_compatible(const GraphProgram &program) {
  for (const GraphSegment &segment : program.segments) {
    if (segment.operations.empty()) return false;
    for (const GraphOperationRef &operation : segment.operations)
      if (operation.node_class != GraphNodeClass::device &&
          operation.node_class != GraphNodeClass::device_predicate)
        return false;
  }
  return true;
}

GraphProgram build_graph_program(const StepPlan &plan, const GraphLoweringAuthorities &authority,
                                 GraphVariantKind variant) {
  std::string authority_error;
  if (!validate_graph_lowering_authorities(plan, authority, NULL, NULL, 1, NULL, &authority_error))
    throw std::invalid_argument(authority_error);
  if (!valid_graph_variant(variant)) throw std::invalid_argument("graph variant is invalid");
  if (variant == GraphVariantKind::cw_operator && plan.program != StepProgram::solve_cw)
    throw std::invalid_argument("CW graph variant requires a solve_cw StepPlan");
  if (variant != GraphVariantKind::cw_operator && plan.program == StepProgram::solve_cw)
    throw std::invalid_argument("solve_cw StepPlan requires the CW graph variant");

  GraphProgram result;
  result.program = plan.program;
  result.variant = variant;
  result.step_plan_signature = plan.signature;
  result.authority_signature = authority.signature;
  result.cw_plan_signature = authority.cw_plan_signature;
  result.scalar_layout = build_step_scalar_layout(plan);

  GraphSegment segment;
  segment.variant = variant;
  segment.first_operation = 0;
  segment.operation_count = 0;
  segment.exit_boundary = GraphBoundaryKind::none;
  segment.signature = 0;

  const std::vector<uint32_t> selected = selected_operations(plan, variant);
  for (size_t i = 1; i < selected.size(); ++i)
    if (selected[i] <= selected[i - 1])
      throw std::invalid_argument("graph variant operation schedule is not strictly ordered");
  for (size_t selected_index = 0; selected_index < selected.size(); ++selected_index) {
    const uint32_t operation_index = selected[selected_index];
    if (operation_index >= plan.operations.size())
      throw std::invalid_argument("graph variant operation is out of range");
    if (operation_is_covered(authority, operation_index))
      throw std::invalid_argument("graph variant enters a host-covered operation interval");
    const Operation &operation = plan.operations[operation_index];
    const GraphHostInterval *host_interval = find_host_interval(authority, operation_index);
    if (host_interval) {
      flush_segment(result, segment);
      const GraphOperationRef ref = make_operation_ref(
          plan, operation_index, GraphNodeClass::host_boundary, result.scalar_layout, authority);
      append_boundary(result, GraphBoundaryKind::host_callback, operation_index,
                      1 + host_interval->covered_operation_count, false, ref);
      selected_index += host_interval->covered_operation_count;
      continue;
    }

    const GraphHaloDisposition *halo = find_halo_disposition(authority, operation_index);
    GraphBoundaryKind boundary = host_boundary_kind(operation, halo);
    if (variant == GraphVariantKind::magnetic_restore &&
        operation.kind == OpKind::restore_magnetic_fields)
      boundary = GraphBoundaryKind::none;
    if (boundary != GraphBoundaryKind::none) {
      flush_segment(result, segment);
      const GraphOperationRef ref = make_operation_ref(
          plan, operation_index, GraphNodeClass::host_boundary, result.scalar_layout, authority);
      const GraphRemoteOverlap *overlap =
          boundary == GraphBoundaryKind::remote_halo
              ? find_remote_overlap(authority, operation_index)
              : NULL;
      const uint32_t covered_count = overlap ? 2 : 1;
      append_boundary(result, boundary, operation_index, covered_count, false, ref);
      if (overlap) {
        if (selected_index + 1 >= selected.size() ||
            selected[selected_index + 1] != overlap->update_operation)
          throw std::invalid_argument(
              "graph remote-overlap successor is not adjacent in the selected schedule");
        ++selected_index;
      }
      continue;
    }

    if (!segment.operations.empty() &&
        operation_index != segment.operations.back().operation_index + 1)
      flush_segment(result, segment);
    if (segment.operations.empty()) segment.first_operation = operation_index;
    const GraphOperationRef ref = make_operation_ref(plan, operation_index, device_class(operation),
                                                     result.scalar_layout, authority);
    segment.operations.push_back(ref);
    const GraphBoundaryKind completion = completion_boundary_kind(operation);
    if (completion != GraphBoundaryKind::none) {
      segment.exit_boundary = completion;
      flush_segment(result, segment);
      append_boundary(result, completion, operation_index, 0, true, ref);
    }
  }
  flush_segment(result, segment);
  result.signature = compute_graph_program_signature(result);
  return result;
}

uint64_t compute_graph_program_signature(const GraphProgram &program) {
  uint64_t signature = 0xcbf29ce484222325ull;
  mix(signature, uint64_t(program.program));
  mix(signature, uint64_t(program.variant));
  mix(signature, uint64_t(program.fallback_policy));
  mix(signature, program.authority_signature);
  mix(signature, program.scalar_layout.signature);
  mix(signature, program.segments.size());
  for (const GraphSegment &segment : program.segments)
    mix(signature, segment.signature);
  mix(signature, program.boundaries.size());
  for (const GraphBoundary &boundary : program.boundaries)
    mix(signature, boundary.signature);
  mix(signature, program.schedule.size());
  for (const GraphScheduleEntry &entry : program.schedule) {
    mix(signature, uint64_t(entry.kind));
    mix(signature, entry.index);
  }
  return signature;
}

bool validate_graph_program(const StepPlan &plan, const GraphLoweringAuthorities &authority,
                            const GraphProgram &program, std::string *error) {
  if (error) error->clear();
  try {
    std::string authority_error;
    if (!validate_graph_lowering_authorities(plan, authority, NULL, NULL, 1, NULL,
                                             &authority_error))
      throw std::invalid_argument(authority_error);
    if (!valid_step_program(program.program) || !valid_graph_variant(program.variant) ||
        program.fallback_policy != GraphFallbackPolicy::whole_program_eager)
      throw std::invalid_argument("graph program contains an invalid enum value");
    if (program.scalar_layout.abi_version != step_scalars_abi_version ||
        program.scalar_layout.total_bytes != sizeof(StepScalars))
      throw std::invalid_argument("graph scalar layout ABI is invalid");
    std::set<std::tuple<uint32_t, uint32_t, uint32_t> > scalar_locations;
    for (const StepScalarSlot &slot : program.scalar_layout.slots) {
      if (!valid_scalar_semantic(slot.semantic) || !valid_scalar_type(slot.type) ||
          !slot.byte_size || slot.byte_offset > program.scalar_layout.total_bytes ||
          slot.byte_size > program.scalar_layout.total_bytes - slot.byte_offset ||
          (slot.type == StepScalarType::predicate_bit && slot.bit_offset >= 64) ||
          (slot.type != StepScalarType::predicate_bit && slot.bit_offset))
        throw std::invalid_argument("graph scalar slot is malformed");
      const std::tuple<uint32_t, uint32_t, uint32_t> location(slot.byte_offset, slot.bit_offset,
                                                              slot.byte_size);
      if (!scalar_locations.insert(location).second)
        throw std::invalid_argument("graph scalar slots overlap exactly");
    }
    if (program.signature != compute_graph_program_signature(program))
      throw std::invalid_argument("graph program signature is stale");
    const GraphProgram expected = build_graph_program(plan, authority, program.variant);
    if (program != expected)
      throw std::invalid_argument("graph program differs from canonical lowering");

    std::vector<uint8_t> covered(plan.operations.size(), 0);
    for (const GraphScheduleEntry &entry : program.schedule) {
      if (entry.kind != GraphScheduleKind::segment && entry.kind != GraphScheduleKind::boundary)
        throw std::invalid_argument("graph schedule kind is invalid");
      if (entry.kind == GraphScheduleKind::segment) {
        if (entry.index >= program.segments.size())
          throw std::invalid_argument("graph schedule segment index is out of range");
        const GraphSegment &segment = program.segments[entry.index];
        if (!valid_graph_variant(segment.variant) ||
            !valid_graph_boundary_kind(segment.exit_boundary) || segment.operations.empty() ||
            segment.operation_count != segment.operations.size() ||
            uint64_t(segment.first_operation) + segment.operation_count > plan.operations.size())
          throw std::invalid_argument("graph schedule contains an empty or malformed segment");
        for (size_t i = 0; i < segment.operations.size(); ++i) {
          const GraphOperationRef &operation = segment.operations[i];
          if (operation.operation_index >= plan.operations.size() ||
              operation.operation_index != segment.first_operation + i ||
              !valid_graph_node_class(operation.node_class) ||
              operation.node_class == GraphNodeClass::host_boundary ||
              covered[operation.operation_index]++)
            throw std::invalid_argument("graph segment coverage is invalid");
          std::set<uint32_t> operation_slots;
          for (uint32_t slot : operation.scalar_slots)
            if (slot >= program.scalar_layout.slots.size() || !operation_slots.insert(slot).second)
              throw std::invalid_argument("graph operation scalar slot is invalid");
        }
      }
      else {
        if (entry.index >= program.boundaries.size())
          throw std::invalid_argument("graph schedule boundary index is out of range");
        const GraphBoundary &boundary = program.boundaries[entry.index];
        if (!valid_graph_boundary_kind(boundary.kind) || boundary.kind == GraphBoundaryKind::none ||
            !valid_graph_node_class(boundary.operation.node_class) ||
            (!boundary.completion_only &&
             boundary.operation.node_class != GraphNodeClass::host_boundary) ||
            uint64_t(boundary.first_operation) + boundary.operation_count > plan.operations.size())
          throw std::invalid_argument("graph schedule boundary is malformed");
        std::set<uint32_t> boundary_slots;
        for (uint32_t slot : boundary.operation.scalar_slots)
          if (slot >= program.scalar_layout.slots.size() || !boundary_slots.insert(slot).second)
            throw std::invalid_argument("graph boundary scalar slot is invalid");
        if (!boundary.completion_only)
          for (uint32_t i = 0; i < boundary.operation_count; ++i) {
            const size_t operation_index = size_t(boundary.first_operation) + i;
            if (operation_index >= covered.size() || covered[operation_index]++)
              throw std::invalid_argument("graph boundary coverage is invalid");
          }
      }
    }
    const std::vector<uint32_t> selected = selected_operations(plan, program.variant);
    for (uint32_t operation_index : selected)
      if (!covered[operation_index])
        throw std::invalid_argument("graph program omits a selected operation");
    return true;
  }
  catch (const std::exception &exception) {
    if (error) *error = exception.what();
    return false;
  }
}

void format_graph_program(const GraphProgram &program, std::vector<std::string> &out) {
  out.clear();
  for (const GraphScheduleEntry &entry : program.schedule) {
    std::ostringstream line;
    if (entry.kind == GraphScheduleKind::segment) {
      const GraphSegment &segment = program.segments.at(entry.index);
      line << "segment[" << segment.first_operation << "+" << segment.operation_count << "]";
      if (segment.exit_boundary != GraphBoundaryKind::none)
        line << " -> " << boundary_name(segment.exit_boundary);
    }
    else {
      const GraphBoundary &boundary = program.boundaries.at(entry.index);
      line << "boundary(" << boundary_name(boundary.kind) << ")[" << boundary.first_operation;
      if (boundary.operation_count) line << "+" << boundary.operation_count;
      line << "]";
    }
    out.push_back(line.str());
  }
}

} // namespace meep
