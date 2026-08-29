/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include "backend/storage_plan.hpp"
#include "backend/halo_plan.hpp"
#include "meep_internals.hpp"

#include <limits>
#include <stdexcept>
#include <stdint.h>

namespace meep {

const char *array_kind_name(array_kind k) {
  switch (k) {
    case array_kind::f: return "f";
    case array_kind::f_u: return "f_u";
    case array_kind::f_w: return "f_w";
    case array_kind::f_w_prev: return "f_w_prev";
    case array_kind::f_cond: return "f_cond";
    case array_kind::f_bfast: return "f_bfast";
    case array_kind::f_minus_p: return "f_minus_p";
    case array_kind::f_rderiv_int: return "f_rderiv_int";
    case array_kind::f_backup: return "f_backup";
    case array_kind::f_u_backup: return "f_u_backup";
    case array_kind::f_w_backup: return "f_w_backup";
    case array_kind::f_cond_backup: return "f_cond_backup";
    case array_kind::f_bfast_backup: return "f_bfast_backup";
    case array_kind::chi1inv: return "chi1inv";
    case array_kind::conductivity: return "conductivity";
    case array_kind::condinv: return "condinv";
    case array_kind::chi2: return "chi2";
    case array_kind::chi3: return "chi3";
    case array_kind::sigma: return "sigma";
    case array_kind::pml_sig: return "pml_sig";
    case array_kind::pml_kap: return "pml_kap";
    case array_kind::pml_siginv: return "pml_siginv";
    case array_kind::dft: return "dft";
    case array_kind::dft_phase: return "dft_phase";
    case array_kind::polarization_internal: return "polarization_internal";
    case array_kind::num_kinds: break;
  }
  return "?";
}

int polarization_storage_aux(int state_index, size_t layout_ordinal) {
  const size_t stride = 1024;
  if (state_index < 0 || layout_ordinal >= stride ||
      size_t(state_index) > (size_t(std::numeric_limits<int>::max()) - layout_ordinal) / stride)
    throw std::overflow_error("polarization storage key overflow");
  return state_index * int(stride) + int(layout_ordinal);
}

static size_t element_bytes(ElementType t) {
  switch (t) {
    case ElementType::realnum_value: return sizeof(realnum);
    case ElementType::complex_realnum: return 2 * sizeof(realnum);
    case ElementType::float64: return sizeof(double);
    case ElementType::complex_float64: return 2 * sizeof(double);
    case ElementType::int32: return 4;
    case ElementType::index: return sizeof(size_t);
  }
  return 0;
}

static void add_array_bytes(size_t &total, const ArraySpec &spec) {
  const size_t element_size = element_bytes(spec.element_type);
  if (element_size && spec.elements > std::numeric_limits<size_t>::max() / element_size)
    throw std::overflow_error("backend storage plan array byte count overflow");
  const size_t bytes = spec.elements * element_size;
  if (bytes > std::numeric_limits<size_t>::max() - total)
    throw std::overflow_error("backend storage plan total byte count overflow");
  total += bytes;
}

size_t StoragePlan::provisional_peak_bytes() const {
  size_t n = 0;
  for (const ArraySpec &s : arrays)
    if (!is_valid(s.alias_of)) add_array_bytes(n, s);
  return n;
}

size_t StoragePlan::steady_state_bytes() const {
  /* Until PR 4's classify() can elide provisional arrays, nothing is dropped
     between the two, so these are equal. Both are reported anyway: Phase 2's
     memory preflight budgets against the peak, and quietly conflating them
     would understate it later. */
  size_t n = 0;
  for (const ArraySpec &s : arrays)
    if (!is_valid(s.alias_of) && !s.classification_provisional) add_array_bytes(n, s);
  return n;
}

ArrayId CpuArrayCatalog::register_array(const StorageKey &key, void *address, size_t elements,
                                        array_role role, ElementType type) {
  auto it = index_.find(key);
  if (it != index_.end()) {
    /* Re-registration under the same key with a new address happens when an
       owner reallocates; keep the identity, refresh the binding. */
    bases_[it->second] = address;
    specs_[it->second].elements = elements;
    return ArrayId{it->second};
  }
  const uint32_t id = uint32_t(specs_.size());
  ArraySpec spec;
  spec.id = ArrayId{id};
  spec.role = role;
  spec.element_type = type;
  spec.storage = sizeof(realnum) == sizeof(float) ? Precision::f32 : Precision::f64;
  spec.elements = elements;
  spec.alignment = alignof(realnum);
  spec.alias_of = invalid_array();
  spec.classification_provisional = false;
  specs_.push_back(spec);
  bases_.push_back(address);
  keys_.push_back(key);
  index_[key] = id;
  by_address_[address] = id;
  return ArrayId{id};
}

bool CpuArrayCatalog::locate(const void *p, ArrayId &id, ptrdiff_t &element_offset) const {
  id = invalid_array();
  element_offset = 0;
  if (!p) return false;
  const uintptr_t address = reinterpret_cast<uintptr_t>(p);
  for (size_t i = 0; i < specs_.size(); ++i) {
    const ArraySpec &spec = specs_[i];
    if (!bases_[i]) continue;
    const size_t bytes_per_element = element_bytes(spec.element_type);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(bases_[i]);
    const size_t bytes = spec.elements * bytes_per_element;
    if (address < begin || address - begin >= bytes) continue;
    const uintptr_t delta = address - begin;
    if (delta % bytes_per_element) continue;
    id = is_valid(spec.alias_of) ? spec.alias_of : spec.id;
    element_offset = ptrdiff_t(delta / bytes_per_element);
    return true;
  }
  return false;
}

size_t CpuArrayCatalog::total_bytes() const {
  size_t n = 0;
  for (const ArraySpec &s : specs_)
    if (!is_valid(s.alias_of)) add_array_bytes(n, s);
  return n;
}

/* ---------------------------------------------------------------------- */

namespace {

struct Registrar {
  CpuArrayCatalog &cat;
  StoragePlan &plan;
  size_t count = 0;

  void add(int chunk, array_kind kind, int c, int cmp, int aux, void *p, size_t n,
           array_role role, ElementType type) {
    if (!p) return;
    const StorageKey key{chunk, int(kind), c, cmp, aux};
    const ArrayId id = cat.register_array(key, p, n, role, type);
    if (id.value == plan.arrays.size()) {
      plan.arrays.push_back(cat.spec(id));
      plan.keys.push_back(key);
    }
    ++count;
  }
};

} // namespace

size_t build_storage_catalog(fields &f, CpuArrayCatalog &cat, StoragePlan &plan) {
  cat.clear();
  plan.clear();
  Registrar r{cat, plan};

  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    fields_chunk &fc = *f.chunks[i];
    const size_t ntot = size_t(fc.gv.ntot());

    DOCMP2 FOR_COMPONENTS(c) {
      r.add(i, array_kind::f, int(c), cmp, 0, fc.f[c][cmp], ntot, array_role::field,
            ElementType::realnum_value);
      r.add(i, array_kind::f_u, int(c), cmp, 0, fc.f_u[c][cmp], ntot, array_role::field,
            ElementType::realnum_value);
      r.add(i, array_kind::f_w, int(c), cmp, 0, fc.f_w[c][cmp], ntot, array_role::field,
            ElementType::realnum_value);
      r.add(i, array_kind::f_w_prev, int(c), cmp, 0, fc.f_w_prev[c][cmp], ntot, array_role::field,
            ElementType::realnum_value);
      r.add(i, array_kind::f_cond, int(c), cmp, 0, fc.f_cond[c][cmp], ntot, array_role::field,
            ElementType::realnum_value);
      r.add(i, array_kind::f_bfast, int(c), cmp, 0, fc.f_bfast[c][cmp], ntot, array_role::field,
            ElementType::realnum_value);
      r.add(i, array_kind::f_minus_p, int(c), cmp, 0, fc.f_minus_p[c][cmp], ntot,
            array_role::field, ElementType::realnum_value);
      r.add(i, array_kind::f_backup, int(c), cmp, 0, fc.f_backup[c][cmp], ntot, array_role::field,
            ElementType::realnum_value);
      r.add(i, array_kind::f_u_backup, int(c), cmp, 0, fc.f_u_backup[c][cmp], ntot,
            array_role::field, ElementType::realnum_value);
      r.add(i, array_kind::f_w_backup, int(c), cmp, 0, fc.f_w_backup[c][cmp], ntot,
            array_role::field, ElementType::realnum_value);
      r.add(i, array_kind::f_cond_backup, int(c), cmp, 0, fc.f_cond_backup[c][cmp], ntot,
            array_role::field, ElementType::realnum_value);
      r.add(i, array_kind::f_bfast_backup, int(c), cmp, 0, fc.f_bfast_backup[c][cmp], ntot,
            array_role::field, ElementType::realnum_value);
    }
    r.add(i, array_kind::f_rderiv_int, -1, -1, 0, fc.f_rderiv_int, ntot, array_role::field,
          ElementType::realnum_value);

    /* H == B until update_eh splits them. Recorded explicitly so the alias
       survives without comparing addresses. */
    DOCMP2 FOR_H_AND_B(hc, bc) {
      if (fc.f[hc][cmp] && fc.f[hc][cmp] == fc.f[bc][cmp]) {
        const ArrayId h = cat.find({i, int(array_kind::f), int(hc), cmp, 0});
        const ArrayId b = cat.find({i, int(array_kind::f), int(bc), cmp, 0});
        if (is_valid(h) && is_valid(b)) cat.set_alias(h, b);
      }
    }

    // ---- structure_chunk ------------------------------------------------
    structure_chunk &sc = *fc.s;
    const size_t sntot = size_t(sc.gv.ntot());
    FOR_COMPONENTS(c) {
      r.add(i, array_kind::chi2, int(c), -1, 0, sc.chi2[c], sntot, array_role::material,
            ElementType::realnum_value);
      r.add(i, array_kind::chi3, int(c), -1, 0, sc.chi3[c], sntot, array_role::material,
            ElementType::realnum_value);
      for (int d = 0; d < 5; ++d) {
        r.add(i, array_kind::chi1inv, int(c), -1, d, sc.chi1inv[c][d], sntot, array_role::material,
              ElementType::realnum_value);
        r.add(i, array_kind::conductivity, int(c), -1, d, sc.conductivity[c][d], sntot,
              array_role::material, ElementType::realnum_value);
        r.add(i, array_kind::condinv, int(c), -1, d, sc.condinv[c][d], sntot, array_role::material,
              ElementType::realnum_value);
      }
    }
    for (int d = 0; d < 6; ++d) {
      r.add(i, array_kind::pml_sig, -1, -1, d, sc.sig[d], sc.sigsize[d], array_role::material,
            ElementType::realnum_value);
      r.add(i, array_kind::pml_kap, -1, -1, d, sc.kap[d], sc.sigsize[d], array_role::material,
            ElementType::realnum_value);
      r.add(i, array_kind::pml_siginv, -1, -1, d, sc.siginv[d], sc.sigsize[d],
            array_role::material, ElementType::realnum_value);
    }

    // susceptibility sigma
    FOR_FIELD_TYPES(ft) {
      int si = 0;
      for (susceptibility *sus = sc.chiP[ft]; sus; sus = sus->next, ++si)
        FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
          r.add(i, array_kind::sigma, int(c), int(d), si * NUM_FIELD_TYPES + int(ft),
                sus->sigma[c][d], sntot, array_role::material, ElementType::realnum_value);
        }
    }

    /* Register each published internal array separately. The blob header
       contains host pointers and is not device data; only the typed arrays are
       portable. The layout ordinal distinguishes P/P_prev entries belonging
       to the same susceptibility/component/cmp tuple. */
    FOR_FIELD_TYPES(ft) {
      int si = 0;
      for (polarization_state *p = fc.pol[ft]; p; p = p->next, ++si) {
        if (!p->data) continue;
        std::vector<InternalArrayLayout> layout;
        if (!p->s->internal_layout(layout, fc.gv, p->data)) continue;
        realnum *base = static_cast<realnum *>(p->data);
        for (size_t li = 0; li < layout.size(); ++li) {
          const InternalArrayLayout &entry = layout[li];
          const ElementType type = entry.element_type == InternalArrayLayout::complex_realnum
                                       ? ElementType::complex_realnum
                                       : ElementType::realnum_value;
          const int aux = polarization_storage_aux(si, li);
          r.add(i, array_kind::polarization_internal, int(entry.c), entry.cmp, aux,
                base + entry.offset_elements, entry.elements, array_role::polarization, type);
        }
      }
    }

    // ---- dft_chunk ------------------------------------------------------
    int di = 0;
    for (dft_chunk *cur = fc.dft_chunks; cur; cur = cur->next_in_chunk, ++di) {
      r.add(i, array_kind::dft, int(cur->c), -1, di, cur->dft, cur->N * cur->omega.size(),
            array_role::dft, ElementType::complex_realnum);
      r.add(i, array_kind::dft_phase, int(cur->c), -1, di, cur->dft_phase, cur->omega.size(),
            array_role::dft, ElementType::complex_realnum);
    }
  }
  /* H/B and D/E aliases are discovered after field specs enter the plan.
     Refresh the frozen descriptors so every consumer sees the catalog's
     canonical alias graph. */
  for (size_t i = 0; i < plan.arrays.size(); ++i)
    plan.arrays[i] = cat.spec(ArrayId{uint32_t(i)});
  return r.count;
}

size_t audit_storage_catalog(fields &f, const CpuArrayCatalog &cat, bool report) {
  size_t problems = 0;
  auto check = [&](const void *p, const char *what, int chunk) {
    if (!p) return;
    if (!cat.contains_address(p)) {
      ++problems;
      if (report) master_printf("  unregistered %s in chunk %d\n", what, chunk);
    }
  };

  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    fields_chunk &fc = *f.chunks[i];
    DOCMP2 FOR_COMPONENTS(c) {
      check(fc.f[c][cmp], "f", i);
      check(fc.f_u[c][cmp], "f_u", i);
      check(fc.f_w[c][cmp], "f_w", i);
      check(fc.f_w_prev[c][cmp], "f_w_prev", i);
      check(fc.f_cond[c][cmp], "f_cond", i);
      check(fc.f_bfast[c][cmp], "f_bfast", i);
      check(fc.f_minus_p[c][cmp], "f_minus_p", i);
      check(fc.f_backup[c][cmp], "f_backup", i);
      check(fc.f_u_backup[c][cmp], "f_u_backup", i);
      check(fc.f_w_backup[c][cmp], "f_w_backup", i);
      check(fc.f_cond_backup[c][cmp], "f_cond_backup", i);
      check(fc.f_bfast_backup[c][cmp], "f_bfast_backup", i);
    }
    check(fc.f_rderiv_int, "f_rderiv_int", i);

    structure_chunk &sc = *fc.s;
    FOR_COMPONENTS(c) {
      check(sc.chi2[c], "chi2", i);
      check(sc.chi3[c], "chi3", i);
      for (int d = 0; d < 5; ++d) {
        check(sc.chi1inv[c][d], "chi1inv", i);
        check(sc.conductivity[c][d], "conductivity", i);
        check(sc.condinv[c][d], "condinv", i);
      }
    }
    for (int d = 0; d < 6; ++d) {
      check(sc.sig[d], "pml sig", i);
      check(sc.kap[d], "pml kap", i);
      check(sc.siginv[d], "pml siginv", i);
    }
    FOR_FIELD_TYPES(ft) {
      for (susceptibility *sus = sc.chiP[ft]; sus; sus = sus->next)
        FOR_COMPONENTS(c) FOR_DIRECTIONS(d) { check(sus->sigma[c][d], "sigma", i); }
    }
    for (dft_chunk *cur = fc.dft_chunks; cur; cur = cur->next_in_chunk) {
      check(cur->dft, "dft", i);
      check(cur->dft_phase, "dft_phase", i);
    }

    FOR_FIELD_TYPES(ft) {
      for (polarization_state *p = fc.pol[ft]; p; p = p->next) {
        if (!p->data) continue;
        std::vector<InternalArrayLayout> layout;
        if (!p->s->internal_layout(layout, fc.gv, p->data)) continue;
        realnum *base = static_cast<realnum *>(p->data);
        for (size_t li = 0; li < layout.size(); ++li)
          check(base + layout[li].offset_elements, "polarization internal", i);
      }
    }
  }
  return problems;
}

} // namespace meep
