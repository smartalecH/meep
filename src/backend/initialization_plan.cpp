/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include "backend/initialization_plan.hpp"
#include "backend/lifecycle.hpp"
#include "backend/material_ir.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

namespace meep {

const char *init_kind_name(InitKind k) {
  switch (k) {
    case InitKind::zero: return "zero";
    case InitKind::constant: return "constant";
    case InitKind::material_geometry: return "material_geometry";
    case InitKind::pml_profile: return "pml_profile";
    case InitKind::host_callback: return "host_callback";
    case InitKind::host_array: return "host_array";
    case InitKind::checkpoint: return "checkpoint";
  }
  return "?";
}

const char *regional_support_reason_name(RegionalSupportReason reason) {
  switch (reason) {
    case RegionalSupportReason::supported: return "supported";
    case RegionalSupportReason::empty: return "empty";
    case RegionalSupportReason::opaque_coordinates: return "opaque-coordinates";
    case RegionalSupportReason::incomplete_group: return "incomplete-group";
    case RegionalSupportReason::whole_row_kernel: return "whole-row-kernel";
    case RegionalSupportReason::remote_dependency: return "remote-dependency";
  }
  return "?";
}

void InitRegion::validate() const {
  if (chunk < -1) throw std::invalid_argument("initialization region has an invalid chunk");
  if (whole) return;
  if (begin.dim != end.dim)
    throw std::invalid_argument("initialization region has inconsistent dimensions");
  LOOP_OVER_DIRECTIONS(begin.dim, d)
    if (end.in_direction(d) < begin.in_direction(d))
      throw std::invalid_argument("initialization region has a reversed extent");
}

bool InitRegion::empty() const {
  validate();
  if (whole) return false;
  LOOP_OVER_DIRECTIONS(begin.dim, d)
    if (end.in_direction(d) == begin.in_direction(d)) return true;
  return false;
}

bool InitRegion::contains(const InitRegion &other) const {
  validate();
  other.validate();
  if (whole) return chunk < 0 || chunk == other.chunk;
  if (other.whole) return false;
  if (begin.dim != other.begin.dim) return false;
  if (chunk >= 0 && chunk != other.chunk) return false;
  LOOP_OVER_DIRECTIONS(begin.dim, d) {
    if (other.begin.in_direction(d) < begin.in_direction(d)) return false;
    if (other.end.in_direction(d) > end.in_direction(d)) return false;
  }
  return true;
}

bool InitRegion::overlaps(const InitRegion &other) const {
  validate();
  other.validate();
  if (chunk >= 0 && other.chunk >= 0 && chunk != other.chunk) return false;
  if (whole || other.whole) return true;
  if (begin.dim != other.begin.dim) return false;
  if (empty() || other.empty()) return false;
  LOOP_OVER_DIRECTIONS(begin.dim, d)
    if (end.in_direction(d) <= other.begin.in_direction(d) ||
        other.end.in_direction(d) <= begin.in_direction(d))
      return false;
  return true;
}

InitRegion InitRegion::intersection(const InitRegion &other) const {
  if (!overlaps(other)) {
    const ndim dim = whole ? other.begin.dim : begin.dim;
    const int selected_chunk = chunk >= 0 ? chunk : other.chunk;
    const ivec zero(dim);
    return InitRegion(selected_chunk, zero, zero);
  }
  const int selected_chunk = chunk >= 0 ? chunk : other.chunk;
  if (whole && other.whole) {
    InitRegion result;
    result.chunk = selected_chunk;
    return result;
  }
  if (whole) {
    InitRegion result = other;
    result.chunk = selected_chunk;
    return result;
  }
  if (other.whole) {
    InitRegion result = *this;
    result.chunk = selected_chunk;
    return result;
  }
  ivec clipped_begin(begin.dim), clipped_end(begin.dim);
  LOOP_OVER_DIRECTIONS(begin.dim, d) {
    clipped_begin.set_direction(
        d, std::max(begin.in_direction(d), other.begin.in_direction(d)));
    clipped_end.set_direction(d, std::min(end.in_direction(d), other.end.in_direction(d)));
  }
  return InitRegion(selected_chunk, clipped_begin, clipped_end);
}

namespace {

std::mutex pending_regions_mutex;
std::map<const fields *, InitRegion> pending_regions;

bool point_in_spans(uint64_t point, const std::vector<InitPointSpan> &spans) {
  for (const InitPointSpan &span : spans)
    if (point >= span.first && point - span.first < span.count) return true;
  return false;
}

void append_point(std::vector<InitPointSpan> &spans, uint64_t point) {
  if (!spans.empty() && spans.back().first + spans.back().count == point) {
    if (spans.back().count == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("regional initialization point-span overflow");
    ++spans.back().count;
  }
  else spans.push_back(InitPointSpan{point, 1});
}

InitRegion material_destination_region(const MaterialIR &ir,
                                       const MaterialIRDestination &destination) {
  if (destination.chunk_index >= ir.chunks.size())
    throw std::invalid_argument("material destination has an invalid chunk index");
  const MaterialIRChunk &chunk = ir.chunks[destination.chunk_index];
  const component c = component(destination.component);
  ivec begin(ndim(chunk.dimensions)), end(ndim(chunk.dimensions));
  for (int axis = 0; axis < 3; ++axis) {
    const direction d = chunk.dimensions == int(Dcyl)
                            ? (axis == 0 ? P : axis == 1 ? R : Z)
                            : chunk.dimensions == int(D2)
                                  ? (axis == 0 ? Z : axis == 1 ? X : Y)
                                  : direction(axis);
    if (!has_direction(ndim(chunk.dimensions), d)) continue;
    int low = chunk.loop_begin[c][axis];
    int high = chunk.loop_end[c][axis];
    if (destination.offdiagonal) {
      const direction shifted = component_direction(c);
      if (d == shifted) {
        const int delta = type(c) == E_stuff ? -1 : 1;
        if ((delta < 0 && low == std::numeric_limits<int>::min()) ||
            (delta > 0 && high == std::numeric_limits<int>::max()))
          throw std::overflow_error("regional off-diagonal destination extent overflow");
        low += delta;
        high += delta;
      }
    }
    if (low == std::numeric_limits<int>::min() ||
        high > std::numeric_limits<int>::max() - 2)
      throw std::overflow_error("regional destination dependency extent overflow");
    begin.set_direction(d, low - 1);
    end.set_direction(d, high + 2);
  }
  return InitRegion(chunk.chunk, begin, end);
}

std::vector<InitPointSpan> material_destination_spans(
    const MaterialIR &ir, const MaterialIRDestination &destination,
    const InitRegion &requested) {
  std::vector<InitPointSpan> result;
  if (destination.chunk_index >= ir.chunks.size()) return result;
  const MaterialIRChunk &chunk = ir.chunks[destination.chunk_index];
  const component c = component(destination.component);
  uint64_t counts[3];
  for (int axis = 0; axis < 3; ++axis) {
    const int64_t delta = int64_t(chunk.loop_end[c][axis]) - chunk.loop_begin[c][axis];
    if (delta < 0 || delta % 2) throw std::invalid_argument("material destination loop is invalid");
    counts[axis] = uint64_t(delta / 2 + 1);
  }
  if (!counts[0] || !counts[1] || !counts[2] ||
      counts[1] > std::numeric_limits<uint64_t>::max() / counts[2] ||
      counts[0] > std::numeric_limits<uint64_t>::max() / (counts[1] * counts[2]) ||
      counts[0] * counts[1] * counts[2] != destination.point_count)
    throw std::invalid_argument("material destination point extent is invalid");
  const uint64_t plane = counts[1] * counts[2];
  for (uint64_t point = 0; point < destination.point_count; ++point) {
    const uint64_t coordinate[3] = {point / plane, (point % plane) / counts[2],
                                    point % counts[2]};
    ivec begin(ndim(chunk.dimensions)), end(ndim(chunk.dimensions));
    for (int axis = 0; axis < 3; ++axis) {
      const direction d = chunk.dimensions == int(Dcyl)
                              ? (axis == 0 ? P : axis == 1 ? R : Z)
                              : chunk.dimensions == int(D2)
                                    ? (axis == 0 ? Z : axis == 1 ? X : Y)
                                    : direction(axis);
      if (!has_direction(ndim(chunk.dimensions), d)) continue;
      int64_t here = int64_t(chunk.loop_begin[c][axis]) + int64_t(2 * coordinate[axis]);
      if (destination.offdiagonal && d == component_direction(c))
        here += type(c) == E_stuff ? -1 : 1;
      if (here <= std::numeric_limits<int>::min() ||
          here >= std::numeric_limits<int>::max())
        throw std::overflow_error("regional material point extent overflow");
      begin.set_direction(d, int(here - 1));
      end.set_direction(d, int(here + 1));
    }
    if (InitRegion(chunk.chunk, begin, end).overlaps(requested)) append_point(result, point);
  }
  return result;
}

} // namespace

void invalidate_material_region(fields &f, const InitRegion &region, const char *reason) {
  region.validate();
  if (region.whole)
    throw std::invalid_argument("regional material invalidation requires a bounded region");
  if (region.empty()) return;
  {
    std::lock_guard<std::mutex> lock(pending_regions_mutex);
    const auto found = pending_regions.find(&f);
    if (found == pending_regions.end()) pending_regions.insert(std::make_pair(&f, region));
    else {
      InitRegion &current = found->second;
      if (current.chunk != region.chunk || current.begin.dim != region.begin.dim) {
        current = InitRegion();
      }
      else {
        LOOP_OVER_DIRECTIONS(current.begin.dim, d) {
          current.begin.set_direction(
              d, std::min(current.begin.in_direction(d), region.begin.in_direction(d)));
          current.end.set_direction(
              d, std::max(current.end.in_direction(d), region.end.in_direction(d)));
        }
      }
    }
  }
  invalidate(f, MutationKind::material_region, reason);
}

bool pending_material_region(const fields &f, InitRegion *region) {
  std::lock_guard<std::mutex> lock(pending_regions_mutex);
  const auto found = pending_regions.find(&f);
  if (found == pending_regions.end()) return false;
  if (region) *region = found->second;
  return true;
}

void clear_pending_material_region(const fields &f) {
  std::lock_guard<std::mutex> lock(pending_regions_mutex);
  pending_regions.erase(&f);
}

bool regional_replacement_preserves_unselected(const InitializationPlan &installed,
                                               const InitializationPlan &candidate,
                                               const InitializationPlan &restricted,
                                               std::string *reason) {
  const auto reject = [&](const char *message) {
    if (reason) *reason = message;
    return false;
  };
  if (!restricted.regional || !restricted.regional_supported)
    return reject("regional replacement is unsupported");
  if (installed.materials.size() != 1 || candidate.materials.size() != 1 ||
      restricted.materials.size() != 1)
    return reject("regional replacement requires one material recipe");
  const MaterialRecipe &old_recipe = installed.materials[0];
  const MaterialRecipe &new_recipe = candidate.materials[0];
  const MaterialIR *ir = new_recipe.ir().get();
  if (!ir) return reject("regional replacement has no coordinate authority");
  const std::vector<MaterialRecipeRow> &old_rows = old_recipe.rows();
  const std::vector<MaterialRecipeRow> &new_rows = new_recipe.rows();
  if (old_rows.size() != new_rows.size())
    return reject("regional replacement changed material row count");
  std::unordered_map<StorageKey, const MaterialRecipeRow *, StorageKeyHash> old_by_key;
  for (const MaterialRecipeRow &row : old_rows) old_by_key[row.key] = &row;
  std::unordered_map<StorageKey, const InitOperation *, StorageKeyHash> selected;
  for (const InitOperation &op : restricted.operations) {
    if (op.kind != InitKind::material_geometry && op.kind != InitKind::pml_profile) continue;
    if (op.kind == InitKind::material_geometry) {
      if (op.descriptor_index >= ir->destinations.size())
        return reject("regional replacement has an invalid destination identity");
      selected[ir->destinations[op.descriptor_index].key] = &op;
    }
    else {
      for (const InitOperation &full : candidate.operations)
        if (full.destination.id == op.destination.id) {
          for (const MaterialRecipeRow &row : new_rows)
            if (row.key.chunk == op.region.chunk &&
                (row.key.kind == int(array_kind::pml_sig) ||
                 row.key.kind == int(array_kind::pml_kap) ||
                 row.key.kind == int(array_kind::pml_siginv)))
              selected[row.key] = &op;
          break;
        }
    }
  }
  for (const MaterialRecipeRow &row : new_rows) {
    const auto prior = old_by_key.find(row.key);
    if (prior == old_by_key.end() || prior->second->element_type != row.element_type ||
        prior->second->elements != row.elements ||
        prior->second->values.size() != row.values.size())
      return reject("regional replacement changed material row identity");
    if (row.values.empty()) return reject("regional replacement lacks dense row values");
    const auto op = selected.find(row.key);
    if (op == selected.end()) {
      if (row.values != prior->second->values)
        return reject("regional replacement changed an unselected material row");
      continue;
    }
    /* A regional PML operation has no point-span authority: the current CUDA
       lowering copies the complete profile row.  Such a copy is safe only
       when it is byte-for-byte unchanged.  Any PML mutation therefore takes
       the transactional full-replacement path instead of overwriting values
       outside the requested material region. */
    if (op->second->kind == InitKind::pml_profile) {
      if (row.values != prior->second->values)
        return reject("regional replacement changed a whole-row PML value");
      continue;
    }
    if (op->second->descriptor_index >= ir->destinations.size())
      return reject("regional replacement destination is out of range");
    const MaterialIRDestination &destination = ir->destinations[op->second->descriptor_index];
    const size_t element_bytes = sizeof(realnum);
    std::vector<uint8_t> selected_element(row.elements, 0);
    for (const InitPointSpan &span : op->second->point_spans)
      for (uint64_t logical = span.first; logical - span.first < span.count; ++logical) {
        const size_t physical = material_ir_destination_storage_index(*ir, destination, logical);
        if (physical >= selected_element.size())
          return reject("regional replacement point exceeds material storage");
        selected_element[physical] = 1;
      }
    for (size_t i = 0; i < row.elements; ++i)
      if (!selected_element[i] &&
          memcmp(row.values.data() + i * element_bytes,
                 prior->second->values.data() + i * element_bytes, element_bytes))
        return reject("regional replacement changed a value outside the requested region");
  }
  if (reason) reason->clear();
  return true;
}

InitializationPlan InitializationPlan::restrict_to(const InitRegion &region) const {
  region.validate();
  InitializationPlan out;
  out.material_values_generation = material_values_generation;
  out.material_region_generation = material_region_generation;
  out.materials = materials;
  out.requires_remote_transport = requires_remote_transport;
  out.regional = !region.whole;
  out.requested_region = region;
  if (region.whole) {
    out.operations = operations;
    out.pml = pml;
    out.host_callbacks = host_callbacks;
    out.material_destinations = material_destinations;
    out.material_bulk_spans = material_bulk_spans;
    out.material_analytic_interfaces = material_analytic_interfaces;
    out.material_hybrid_patches = material_hybrid_patches;
    out.material_callback_tiles = material_callback_tiles;
    return out;
  }

  const MaterialIR *ir = materials.size() == 1 && materials[0].ir()
                             ? materials[0].ir().get()
                             : NULL;
  std::set<uint32_t> selected_destinations;
  std::set<int> selected_chunks;
  if (ir) {
    for (uint32_t destination = 0; destination < ir->destinations.size(); ++destination) {
      const MaterialIRDestination &candidate = ir->destinations[destination];
      if (material_destination_region(*ir, candidate).overlaps(region))
        selected_chunks.insert(ir->chunks[candidate.chunk_index].chunk);
    }
    /* Material tensor/subpixel final producers share row classification. Keep
       the complete destination group for every intersecting chunk, while the
       point spans below remain spatially clipped. */
    for (uint32_t destination = 0; destination < ir->destinations.size(); ++destination)
      if (selected_chunks.count(ir->chunks[ir->destinations[destination].chunk_index].chunk))
        selected_destinations.insert(destination);
  }
  for (const InitOperation &op : operations) {
    if (op.kind != InitKind::material_geometry && op.kind != InitKind::pml_profile) continue;
    if (!op.region.overlaps(region)) continue;
    if (op.kind == InitKind::material_geometry && !ir) {
      out.regional_supported = false;
      out.regional_reason = RegionalSupportReason::opaque_coordinates;
      out.regional_unsupported_reason =
          "regional material initialization requires immutable coordinate metadata";
      continue;
    }
    InitOperation clipped = op;
    clipped.region = op.region.intersection(region);
    if (op.kind == InitKind::material_geometry && ir) {
      if (op.descriptor_index >= ir->destinations.size() ||
          !selected_destinations.count(op.descriptor_index))
        continue;
      clipped.point_spans =
          material_destination_spans(*ir, ir->destinations[op.descriptor_index], region);
      if (clipped.point_spans.empty()) continue;
      out.material_destinations.push_back(op.descriptor_index);
    }
    out.operations.push_back(clipped);
  }

  std::sort(out.material_destinations.begin(), out.material_destinations.end());
  out.material_destinations.erase(
      std::unique(out.material_destinations.begin(), out.material_destinations.end()),
      out.material_destinations.end());
  if (ir) {
    std::map<uint32_t, const std::vector<InitPointSpan> *> spans;
    for (const InitOperation &op : out.operations)
      if (op.kind == InitKind::material_geometry) spans[op.descriptor_index] = &op.point_spans;
    for (const MaterialIRBulkSpan &bulk : ir->bulk_spans) {
      const auto selected = spans.find(bulk.destination);
      if (selected == spans.end()) continue;
      const uint64_t bulk_end = bulk.first_point + bulk.count;
      for (const InitPointSpan &span : *selected->second) {
        const uint64_t begin = std::max(bulk.first_point, span.first);
        const uint64_t end = std::min(bulk_end, span.first + span.count);
        if (begin < end)
          out.material_bulk_spans.push_back(
              MaterialIRBulkSpan{bulk.destination, begin, end - begin});
      }
    }
    for (uint32_t i = 0; i < ir->analytic_interfaces.size(); ++i) {
      const MaterialIRAnalyticInterface &job = ir->analytic_interfaces[i];
      const auto selected = spans.find(job.destination);
      if (selected != spans.end() && point_in_spans(job.point, *selected->second))
        out.material_analytic_interfaces.push_back(i);
    }
    for (uint32_t i = 0; i < ir->hybrid_patches.size(); ++i) {
      const MaterialIRHybridPatch &patch = ir->hybrid_patches[i];
      const auto selected = spans.find(patch.destination);
      if (selected != spans.end() && point_in_spans(patch.point, *selected->second))
        out.material_hybrid_patches.push_back(i);
    }
    for (const MaterialCallbackTile &tile : materials[0].callback_tiles()) {
      const auto selected = spans.find(tile.destination);
      if (selected == spans.end()) continue;
      const uint64_t tile_end = tile.first_point + tile.count;
      for (const InitPointSpan &span : *selected->second) {
        const uint64_t begin = std::max(tile.first_point, span.first);
        const uint64_t end = std::min(tile_end, span.first + span.count);
        if (begin < end)
          out.material_callback_tiles.push_back(
              MaterialCallbackTile{tile.destination, tile.material, begin, end - begin});
      }
    }
  }
  for (const PmlRecipe &recipe : pml)
    if (region.chunk < 0 || region.chunk == recipe.chunk)
      out.pml.push_back(recipe);
  std::set<uint32_t> callback_ids;
  for (const InitOperation &op : out.operations)
    if (op.kind == InitKind::host_callback) callback_ids.insert(op.descriptor_index);
  for (const HostCallbackRecipe &callback : host_callbacks)
    if (callback_ids.count(callback.id)) out.host_callbacks.push_back(callback);
  if (out.requires_remote_transport) {
    out.regional_supported = false;
    out.regional_reason = RegionalSupportReason::remote_dependency;
    out.regional_unsupported_reason =
        "regional resident publication requires deferred PR7 transport";
  }
  if (out.operations.empty() && out.regional_supported)
    out.regional_reason = RegionalSupportReason::empty;
  return out;
}

/* The CPU material implementation is unchanged (§12.4): geom_epsilon,
   structure::set_materials, structure_chunk::set_chi1inv, eff_chi1inv_row and
   libctl adaptive integration still populate the coefficient arrays eagerly,
   with their geometry queries, staggered volumes, averaging formulas,
   tolerances and fallback behavior untouched.

   So this plan *describes* what produced the current values rather than being
   replayed to produce them. That is exactly what makes every pre-run material
   query keep working -- sim.get_epsilon(), get_array(Dielectric), plot2D,
   structure_dump and Simulation.geps all read arrays that already exist. The
   plan's own acceptance criteria call that out as the primary risk of this PR,
   and the way to not have the risk is to not defer the construction. */
InitializationPlan build_initialization_plan(fields &f) {
  InitializationPlan plan;
  plan.requires_remote_transport = count_processors() != 1;

  plan.material_values_generation = generation(f, MutationKind::material_values);
  plan.material_region_generation = generation(f, MutationKind::material_region);
  plan.materials.push_back(build_host_reference_material_recipe(f));

  const MaterialIR *ir = material_ir_for(f);
  if (ir) {
    validate_material_ir(*ir);
    for (const MaterialIRPmlAxis &source : ir->pml_axes) {
      PmlRecipe p;
      p.chunk = source.chunk;
      p.direction_ = source.direction;
      p.sigma = source.sigma;
      p.kappa = source.kappa;
      p.sigma_inv = source.sigma_inv;
      plan.pml.push_back(p);
    }
  }
  else {
    for (int i = 0; i < f.num_chunks; ++i) {
      if (!f.chunks[i]->is_mine()) continue;
      const structure_chunk &sc = *f.chunks[i]->s;
      for (int d = 0; d < 6; ++d) {
        if (!sc.sig[d]) continue;
        PmlRecipe p;
        p.chunk = i;
        p.direction_ = d;
        if (sc.sigsize[d] < 0)
          throw std::invalid_argument("PML initialization recipe has a negative extent");
        const size_t n = size_t(sc.sigsize[d]);
        if (!n || !sc.kap[d] || !sc.siginv[d])
          throw std::invalid_argument("PML initialization recipe has incomplete storage");
        p.sigma.assign(sc.sig[d], sc.sig[d] + n);
        p.kappa.assign(sc.kap[d], sc.kap[d] + n);
        p.sigma_inv.assign(sc.siginv[d], sc.siginv[d] + n);
        plan.pml.push_back(p);
      }
    }
  }

  /* One operation per catalogued array, describing how it got its value. Field
     arrays start at zero; material and PML arrays come from the geometry. */
  for (size_t k = 0; k < f.storage_plan->arrays.size(); ++k) {
    const ArraySpec &spec = f.storage_plan->arrays[k];
    const StorageKey &key = f.storage_plan->keys[k];
    /* Resolved material tombstones preserve canonical ArrayIds but own no
       storage and therefore have no initialization producer. */
    if (spec.role == array_role::material && spec.classification_elided) continue;
    InitOperation op;
    op.destination = ArrayRef{spec.id, 0, spec.elements};
    op.descriptor_index = 0;
    op.region = InitRegion();
    op.region.chunk = key.chunk;
    switch (spec.role) {
      case array_role::field: op.kind = InitKind::zero; break;
      case array_role::material:
        op.kind = (key.kind == int(array_kind::pml_sig) || key.kind == int(array_kind::pml_kap) ||
                   key.kind == int(array_kind::pml_siginv))
                      ? InitKind::pml_profile
                      : InitKind::material_geometry;
        if (op.kind == InitKind::material_geometry && ir) {
          bool found = false;
          for (uint32_t destination = 0; destination < ir->destinations.size(); ++destination)
            if (ir->destinations[destination].key == key) {
              op.descriptor_index = destination;
              op.region = material_destination_region(*ir, ir->destinations[destination]);
              found = true;
              break;
            }
          if (!found)
            throw std::logic_error(
                "material initialization destination is absent from the immutable IR");
        }
        break;
      case array_role::dft: op.kind = InitKind::zero; break;
      default: op.kind = InitKind::zero; break;
    }
    plan.operations.push_back(op);
  }
  if (ir) {
    plan.material_destinations.reserve(ir->destinations.size());
    for (uint32_t i = 0; i < ir->destinations.size(); ++i)
      plan.material_destinations.push_back(i);
    plan.material_bulk_spans = ir->bulk_spans;
    plan.material_analytic_interfaces.reserve(ir->analytic_interfaces.size());
    for (uint32_t i = 0; i < ir->analytic_interfaces.size(); ++i)
      plan.material_analytic_interfaces.push_back(i);
    plan.material_hybrid_patches.reserve(ir->hybrid_patches.size());
    for (uint32_t i = 0; i < ir->hybrid_patches.size(); ++i)
      plan.material_hybrid_patches.push_back(i);
    plan.material_callback_tiles = plan.materials[0].callback_tiles();
  }
  return plan;
}

} // namespace meep
