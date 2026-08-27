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

/* The finalized set of arrays, and the CPU binding for them.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 */

#ifndef MEEP_BACKEND_STORAGE_PLAN_HPP
#define MEEP_BACKEND_STORAGE_PLAN_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include "meep.hpp"
#include "backend/array_ref.hpp"

namespace meep {

/* Which storage an array is. Extends the halo table's key to everything a
   second backend needs to know about, not just the arrays the halos touch. */
enum class array_kind {
  f,            // fields_chunk::f
  f_u,
  f_w,
  f_w_prev,
  f_cond,
  f_bfast,
  f_minus_p,
  f_rderiv_int,
  f_backup,
  f_u_backup,
  f_w_backup,
  f_cond_backup,
  f_bfast_backup,
  chi1inv,
  conductivity,
  condinv,
  chi2,
  chi3,
  sigma,
  pml_sig,
  pml_kap,
  pml_siginv,
  dft,
  polarization_internal,
  num_kinds
};

const char *array_kind_name(array_kind k);

struct StorageKey {
  int chunk;
  int kind;       // array_kind
  int component_; // meep::component, or -1
  int cmp;        // 0/1, or -1
  int aux;        // direction, susceptibility index, dft index, ...

  bool operator==(const StorageKey &o) const {
    return chunk == o.chunk && kind == o.kind && component_ == o.component_ && cmp == o.cmp &&
           aux == o.aux;
  }
};

struct StorageKeyHash {
  size_t operator()(const StorageKey &k) const {
    size_t h = size_t(k.chunk + 1) * 1000003u;
    h = h * 31 + size_t(k.kind);
    h = h * 31 + size_t(k.component_ + 2);
    h = h * 31 + size_t(k.cmp + 2);
    h = h * 31 + size_t(k.aux + 2);
    return h;
  }
};

struct StoragePlan {
  std::vector<ArraySpec> arrays;
  std::vector<StorageKey> keys; // parallel to `arrays`

  void clear() {
    arrays.clear();
    keys.clear();
  }

  /* Superset allocated in pass 1, before classification can elide anything.
     Distinct from steady_state_bytes because Phase 2's memory preflight has to
     budget for the peak, not the final plan. Until PR 4 introduces
     classification the two are equal, and both are reported. */
  size_t provisional_peak_bytes() const;
  size_t steady_state_bytes() const;
};

/* Non-owning views of storage that fields_chunk, structure_chunk, dft_chunk and
   the susceptibility objects continue to own. This is a catalog, not a
   re-homing: moving ordinary CPU field/material/DFT ownership is explicitly not
   a Phase 1 requirement (decision E). */
class CpuArrayCatalog {
public:
  void clear() {
    specs_.clear();
    bases_.clear();
    keys_.clear();
    index_.clear();
    by_address_.clear();
  }

  ArrayId register_array(const StorageKey &key, void *address, size_t elements, array_role role,
                         ElementType type);

  void *resolve_untyped(ArrayId id) const { return bases_[id.value]; }
  template <typename T> T *resolve(ArrayId id) const {
    return static_cast<T *>(bases_[id.value]);
  }
  const ArraySpec &spec(ArrayId id) const { return specs_[id.value]; }
  const StorageKey &key(ArrayId id) const { return keys_[id.value]; }
  size_t size() const { return specs_.size(); }

  bool contains(const StorageKey &key) const { return index_.count(key) != 0; }
  bool contains_address(const void *p) const { return by_address_.count(p) != 0; }
  ArrayId find(const StorageKey &key) const {
    auto it = index_.find(key);
    return it == index_.end() ? invalid_array() : ArrayId{it->second};
  }

  /* Alias record, e.g. H == B before update_eh splits them. Preserves the
     optimization without comparing pointers. */
  void set_alias(ArrayId a, ArrayId of) { specs_[a.value].alias_of = of; }

  size_t total_bytes() const;

private:
  std::vector<ArraySpec> specs_;
  std::vector<void *> bases_;
  std::vector<StorageKey> keys_;
  std::unordered_map<StorageKey, uint32_t, StorageKeyHash> index_;
  std::unordered_map<const void *, uint32_t> by_address_;
};

/* Walks fields_chunk, structure_chunk and dft_chunk and registers every
   non-null array. Returns the number registered. */
size_t build_storage_catalog(fields &f, CpuArrayCatalog &cat, StoragePlan &plan);

/* Total-coverage check: every non-null array pointer reachable from the chunks
   is registered exactly once, with a correct element count. Returns the number
   of problems found and, when `report` is true, prints them. */
size_t audit_storage_catalog(fields &f, const CpuArrayCatalog &cat, bool report);

} // namespace meep

#endif // MEEP_BACKEND_STORAGE_PLAN_HPP
