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

bool InitRegion::contains(const InitRegion &other) const {
  if (whole) return chunk < 0 || chunk == other.chunk;
  if (other.whole) return false;
  if (chunk >= 0 && chunk != other.chunk) return false;
  LOOP_OVER_DIRECTIONS(begin.dim, d) {
    if (other.begin.in_direction(d) < begin.in_direction(d)) return false;
    if (other.end.in_direction(d) > end.in_direction(d)) return false;
  }
  return true;
}

InitializationPlan InitializationPlan::restrict_to(const InitRegion &region) const {
  InitializationPlan out;
  out.material_values_generation = material_values_generation;
  out.material_region_generation = material_region_generation;
  out.materials = materials;
  out.pml = pml;
  out.host_callbacks = host_callbacks;
  for (const InitOperation &op : operations) {
    /* Keep an operation when the requested region overlaps what it writes.
       A whole-array operation always overlaps; otherwise the requested region
       has to intersect the operation's own. */
    if (op.region.whole || region.whole || op.region.contains(region) ||
        region.contains(op.region))
      out.operations.push_back(op);
  }
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
        break;
      case array_role::dft: op.kind = InitKind::zero; break;
      default: op.kind = InitKind::zero; break;
    }
    plan.operations.push_back(op);
  }
  return plan;
}

} // namespace meep
