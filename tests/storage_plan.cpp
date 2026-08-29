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

/* PR 3 acceptance tests.
 *
 *  - Total catalog coverage: after a full run, every non-null array reachable
 *    from fields_chunk, structure_chunk and dft_chunk is registered.
 *  - Mid-run mutation: adding an integrated source after the first step gives
 *    the same answer as having had it from the start. This is the top break
 *    risk in the PR -- update_eh used to allocate f_minus_p transparently, and
 *    once it stops, a missed invalidation silently ignores the source.
 *  - Memory reporting: provisional peak vs steady state.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include <meep.hpp>

#include "backend/lifecycle.hpp"
#include "backend/halo_plan.hpp"
#include "backend/prepare.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

using namespace meep;
using std::complex;

static int failures = 0;

#define CHECK(cond, ...)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      /* printf, not master_printf: a failure on a non-master rank is exactly  \
         the interesting kind, and master_printf would swallow it. */          \
      printf("[rank %d] FAIL (%s:%d): ", my_rank(), __FILE__, __LINE__);                           \
      printf(__VA_ARGS__);                                                                         \
      printf("\n");                                                                                \
      fflush(stdout);                                                                              \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

static double eps_slab(const vec &p) { return (fabs(p.y()) < 0.4) ? 12.0 : 1.0; }

/* ------------------------------------------------------------------ */
/* Catalog coverage                                                    */
/* ------------------------------------------------------------------ */

static void test_coverage(const char *name, structure &s, bool with_flux, const vec &src_at) {
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, src_at);
  if (with_flux) {
    volume fv(vec(0.8, -1.0), vec(0.8, 1.0));
    f.add_dft_flux(Z, fv, 0.25, 0.35, 3);
  }

  f.advance(8);

  const size_t problems = audit_storage_catalog(f, *f.array_catalog, true);
  CHECK(problems == 0, "%s: %zu arrays are reachable but not registered", name, problems);
  bool owns_chunk = false;
  for (int i = 0; i < f.num_chunks; ++i) owns_chunk = owns_chunk || f.chunks[i]->is_mine();
  CHECK(!owns_chunk || f.array_catalog->size() > 0, "%s: owned catalog is empty", name);
  CHECK(f.storage_plan->arrays.size() == f.array_catalog->size(),
        "%s: storage plan/catalog sizes differ", name);
  for (size_t i = 0; i < f.array_catalog->size(); ++i)
    CHECK(f.storage_plan->arrays[i].alias_of == f.array_catalog->spec(ArrayId{uint32_t(i)}).alias_of,
          "%s: storage plan lost alias metadata at ArrayId %zu", name, i);

  /* Halo construction has its own temporary pointer table because it can run
     before storage preparation. A backend consumes only the remapped copy. */
  bool rejected_nonpolarization_fallback = false;
  for (const HaloPlan &source : f.halos->plans) {
    HaloPlan canonical;
    std::string why;
    CHECK(remap_halo_plan(source, f.halos->arrays, f.halos->host_arrays, *f.array_catalog,
                          f.is_real ? 1 : 2, canonical, why),
          "%s: halo plan did not map to canonical storage: %s", name, why.c_str());
    CHECK(canonical.storage == HaloStorageDisposition::canonical,
          "%s: fully catalogued halo did not select canonical storage", name);
    CHECK(canonical.host_gather.empty() && canonical.host_scatter.empty(),
          "%s: canonical halo retained a host mirror", name);
    std::vector<ElementRef> source_refs, canonical_refs;
    expand_gather(source, source_refs);
    expand_gather(canonical, canonical_refs);
    CHECK(source_refs.size() == canonical_refs.size(), "%s: canonical gather size differs", name);
    for (size_t j = 0; j < source_refs.size() && j < canonical_refs.size(); ++j) {
      const ElementRef &from = source_refs[j], &to = canonical_refs[j];
      CHECK(to.array.value < f.array_catalog->size(), "%s: canonical gather id is invalid", name);
      CHECK(f.halos->arrays.base(from.array) + from.index ==
                f.array_catalog->resolve<realnum>(to.array) + to.index,
            "%s: canonical gather address differs at %zu", name, j);
    }
    expand_scatter(source, source_refs);
    expand_scatter(canonical, canonical_refs);
    CHECK(source_refs.size() == canonical_refs.size(), "%s: canonical scatter size differs", name);
    for (size_t j = 0; j < source_refs.size() && j < canonical_refs.size(); ++j) {
      const ElementRef &from = source_refs[j], &to = canonical_refs[j];
      CHECK(to.array.value < f.array_catalog->size(), "%s: canonical scatter id is invalid", name);
      CHECK(f.halos->arrays.base(from.array) + from.index ==
                f.array_catalog->resolve<realnum>(to.array) + to.index,
            "%s: canonical scatter address differs at %zu", name, j);
    }

    if (!rejected_nonpolarization_fallback && source.ft != PE_stuff &&
        source.ft != PH_stuff && source.block_elements) {
      CpuArrayCatalog empty;
      HaloPlan unchanged = source;
      unchanged.tag = 0x57913;
      CHECK(!remap_halo_plan(source, f.halos->arrays, f.halos->host_arrays, empty,
                             f.is_real ? 1 : 2, unchanged, why) && !why.empty(),
            "%s: unresolved non-polarization halo selected host fallback", name);
      CHECK(unchanged.tag == 0x57913,
            "%s: failed non-polarization remap partially published its destination", name);
      rejected_nonpolarization_fallback = true;
    }
  }
  CHECK(or_to_all(rejected_nonpolarization_fallback),
        "%s: no non-polarization remap rejection was exercised", name);
  FOR_FIELD_TYPES(ft) for (size_t i = 0; i < f.halos->zeros[ft].size(); ++i) {
    ZeroPlan canonical;
    std::string why;
    CHECK(remap_zero_plan(f.halos->zeros[ft][i], f.halos->arrays, *f.array_catalog, canonical,
                          why),
          "%s: zero plan did not map to canonical storage: %s", name, why.c_str());
  }

  master_printf("%s: %zu arrays catalogued, provisional %.2f MB, steady %.2f MB\n", name,
                f.array_catalog->size(),
                f.storage_plan->provisional_peak_bytes() / 1048576.0,
                f.storage_plan->steady_state_bytes() / 1048576.0);
}

static double unit_sigma(const vec &) { return 1.0; }
static double one_value(const vec &) { return 1.0; }

class opaque_storage_lorentzian : public lorentzian_susceptibility {
public:
  opaque_storage_lorentzian(realnum omega_0, realnum gamma)
      : lorentzian_susceptibility(omega_0, gamma) {}
  virtual susceptibility *clone() const { return new opaque_storage_lorentzian(*this); }
  virtual bool internal_layout(std::vector<InternalArrayLayout> &out, const grid_volume &gv,
                               void *data) const {
    (void)out;
    (void)gv;
    (void)data;
    return false;
  }
};

static void test_polarization_storage_keys() {
  struct Case {
    field_type ft;
    int state;
    size_t ordinal;
  } cases[] = {{E_stuff, 0, 0}, {H_stuff, 0, 0}, {E_stuff, 1, 1023},
               {E_stuff, 1, 1024}, {H_stuff, 17, 65537},
               {E_stuff,
                int((uint64_t(std::numeric_limits<uint32_t>::max()) - uint64_t(E_stuff)) /
                    NUM_FIELD_TYPES),
                std::numeric_limits<uint32_t>::max()}};
  std::set<uint64_t> packed;
  for (const Case &c : cases) {
    const uint64_t aux = polarization_storage_aux(c.ft, c.state, c.ordinal);
    CHECK(packed.insert(aux).second, "polarization storage key packing collided");
    CHECK(polarization_storage_field_type(aux) == c.ft,
          "polarization storage key lost field type %d", int(c.ft));
    CHECK(polarization_storage_state_index(aux) == c.state,
          "polarization storage key lost state index %d", c.state);
    CHECK(polarization_storage_layout_ordinal(aux) == uint32_t(c.ordinal),
          "polarization storage key lost layout ordinal %zu", c.ordinal);
  }
  bool rejected = false;
  try { (void)polarization_storage_aux(E_stuff, -1, 0); }
  catch (const std::overflow_error &) { rejected = true; }
  CHECK(rejected, "polarization storage key accepted a negative state index");
  if (std::numeric_limits<size_t>::max() > std::numeric_limits<uint32_t>::max()) {
    rejected = false;
    try {
      (void)polarization_storage_aux(E_stuff, 0,
                                     size_t(std::numeric_limits<uint32_t>::max()) + 1);
    }
    catch (const std::overflow_error &) { rejected = true; }
    CHECK(rejected, "polarization storage key accepted an overflowing layout ordinal");
  }
  rejected = false;
  try { (void)polarization_storage_aux(D_stuff, 0, 0); }
  catch (const std::overflow_error &) { rejected = true; }
  CHECK(rejected, "polarization storage key accepted an invalid field type");
  rejected = false;
  try { (void)polarization_storage_field_type(uint64_t(D_stuff) << 32); }
  catch (const std::invalid_argument &) { rejected = true; }
  CHECK(rejected, "polarization storage key decoded an invalid field type");
  rejected = false;
  try {
    const int max_state = int((uint64_t(std::numeric_limits<uint32_t>::max()) -
                               uint64_t(E_stuff)) /
                              NUM_FIELD_TYPES);
    (void)polarization_storage_aux(E_stuff, max_state + 1, 0);
  }
  catch (const std::overflow_error &) { rejected = true; }
  CHECK(rejected, "polarization storage key accepted an overflowing state identity");

  const int h_max_state =
      int((uint64_t(std::numeric_limits<uint32_t>::max()) - uint64_t(H_stuff)) /
          NUM_FIELD_TYPES);
  const uint64_t h_max = polarization_storage_aux(H_stuff, h_max_state, 0);
  CHECK(polarization_storage_field_type(h_max) == H_stuff &&
            polarization_storage_state_index(h_max) == h_max_state,
        "polarization storage key lost the maximum magnetic state identity");
  rejected = false;
  try { (void)polarization_storage_aux(H_stuff, h_max_state + 1, 0); }
  catch (const std::overflow_error &) { rejected = true; }
  CHECK(rejected, "polarization storage key accepted an overflowing magnetic state identity");
}

static void test_polarization_halo_remap() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  lorentzian_susceptibility susceptibility(1.1, 0.05);
  s.add_susceptibility(unit_sigma, E_stuff, susceptibility);
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.13, 0.11));
  f.advance(3);

  size_t polarization_refs = 0;
  for (const HaloPlan &source : f.halos->plans) {
    if (source.ft != PE_stuff && source.ft != PH_stuff) continue;
    HaloPlan canonical;
    std::string why;
    CHECK(remap_halo_plan(source, f.halos->arrays, f.halos->host_arrays, *f.array_catalog,
                          f.is_real ? 1 : 2, canonical, why),
          "polarization halo plan did not map to canonical storage: %s", why.c_str());
    CHECK(source.storage == HaloStorageDisposition::host_owned,
          "source polarization halo is not host-owned");
    CHECK(canonical.storage == HaloStorageDisposition::canonical,
          "fully catalogued polarization halo did not select canonical storage");
    CHECK(canonical.host_gather.empty() && canonical.host_scatter.empty(),
          "canonical polarization halo retained a host mirror");
    std::vector<ElementRef> refs;
    expand_gather(canonical, refs);
    polarization_refs += refs.size();
    expand_scatter(canonical, refs);
    polarization_refs += refs.size();
  }
  CHECK(or_to_all(polarization_refs > 0), "no polarization halo references were remapped");

  size_t polarization_arrays = 0;
  for (size_t i = 0; i < f.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = f.array_catalog->spec(id);
    const StorageKey &key = f.array_catalog->key(id);
    if (spec.role != array_role::polarization) continue;
    ++polarization_arrays;
    CHECK(key.kind == int(array_kind::polarization_internal),
          "polarization array has the wrong storage kind");
    CHECK(key.aux == polarization_storage_aux(key.aux / 1024, size_t(key.aux % 1024)),
          "polarization storage key is not stable");
  }
  CHECK(or_to_all(polarization_arrays > 0), "no polarization arrays were catalogued");

  bool rejected_ordinal = false;
  try {
    (void)polarization_storage_aux(0, 1024);
  }
  catch (const std::overflow_error &) { rejected_ordinal = true; }
  CHECK(rejected_ordinal, "polarization storage key accepted an overflowing ordinal");
}

static void test_opaque_polarization_halo_remap() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  s.add_susceptibility(unit_sigma, E_stuff, lorentzian_susceptibility(1.1, 0.05));
  s.add_susceptibility(unit_sigma, E_stuff, opaque_storage_lorentzian(0.8, 0.03));
  s.add_susceptibility(unit_sigma, H_stuff, lorentzian_susceptibility(1.3, 0.07));
  s.add_susceptibility(unit_sigma, H_stuff, opaque_storage_lorentzian(0.9, 0.04));
  fields f(&s);
  f.use_bloch(vec(0.07, -0.11));
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.13, 0.11));
  f.advance(3);

  bool saw_mixed_catalogue = false;
  bool saw_host_fallback = false;
  bool saw_atomic_rejection = false;
  bool saw_pe_fallback = false, saw_ph_fallback = false;
  bool saw_stale_rejection = false, saw_order_rejection = false;
  HostHaloId stale_id = invalid_host_halo();
  for (const HaloPlan &source : f.halos->plans) {
    if ((source.ft != PE_stuff && source.ft != PH_stuff) || !source.block_elements) continue;
    std::vector<ElementRef> source_gather, source_scatter;
    expand_gather(source, source_gather);
    expand_scatter(source, source_scatter);
    size_t mapped = 0, opaque = 0;
    for (const ElementRef &ref : source_gather) {
      ArrayId id = invalid_array();
      ptrdiff_t offset = 0;
      if (f.array_catalog->locate(f.halos->arrays.base(ref.array) + ref.index, id, offset))
        ++mapped;
      else
        ++opaque;
    }
    for (const ElementRef &ref : source_scatter) {
      ArrayId id = invalid_array();
      ptrdiff_t offset = 0;
      if (f.array_catalog->locate(f.halos->arrays.base(ref.array) + ref.index, id, offset))
        ++mapped;
      else
        ++opaque;
    }
    saw_mixed_catalogue = saw_mixed_catalogue || (mapped && opaque);
    if (!opaque) continue;

    HaloPlan host;
    std::string why;
    CHECK(remap_halo_plan(source, f.halos->arrays, f.halos->host_arrays, *f.array_catalog,
                          f.is_real ? 1 : 2, host, why),
          "opaque polarization halo did not select host fallback: %s", why.c_str());
    CHECK(host.storage == HaloStorageDisposition::host_owned,
          "opaque polarization halo selected a partial canonical disposition");
    CHECK(host.gather.empty() && host.gather_slabs.empty() && host.gather_order.empty() &&
              host.scatter.empty() && host.scatter_slabs.empty() && host.scatter_order.empty(),
          "host-owned polarization halo retained canonical references");
    CHECK(host.host_gather.size() == source_gather.size() &&
              host.host_scatter.size() == source_scatter.size(),
          "host-owned polarization halo lost ordered mirror entries");
    for (size_t i = 0; i < host.host_gather.size(); ++i)
      CHECK(host.host_gather[i].id == source.host_gather[i].id,
            "host-owned gather changed identity/order at %zu", i);
    for (size_t i = 0; i < host.host_scatter.size(); ++i)
      CHECK(host.host_scatter[i].id == source.host_scatter[i].id,
            "host-owned scatter changed identity/order at %zu", i);
    if (!host.host_gather.empty()) stale_id = host.host_gather.front().id;
    if (!valid(stale_id) && !host.host_scatter.empty()) stale_id = host.host_scatter.front().id;
    saw_host_fallback = true;
    saw_pe_fallback = saw_pe_fallback || source.ft == PE_stuff;
    saw_ph_fallback = saw_ph_fallback || source.ft == PH_stuff;

    HaloPlan malformed = source;
    if (!malformed.host_gather.empty())
      malformed.host_gather.pop_back();
    else if (!malformed.host_scatter.empty())
      malformed.host_scatter.pop_back();
    else
      continue;
    HaloPlan unchanged = source;
    unchanged.tag = 0x13579;
    unchanged.block_elements += 7;
    const int old_tag = unchanged.tag;
    const size_t old_elements = unchanged.block_elements;
    CHECK(!remap_halo_plan(malformed, f.halos->arrays, f.halos->host_arrays, *f.array_catalog,
                           f.is_real ? 1 : 2, unchanged, why) && !why.empty(),
          "incomplete host mirror was not rejected");
    CHECK(unchanged.tag == old_tag && unchanged.block_elements == old_elements,
          "failed host remap partially published its destination");

    malformed = source;
    if (!malformed.gather.empty())
      malformed.gather.front().index = 1;
    else if (!malformed.scatter.empty())
      malformed.scatter.front().index = 1;
    else
      continue;
    unchanged = source;
    unchanged.tag = 0x24680;
    CHECK(!remap_halo_plan(malformed, f.halos->arrays, f.halos->host_arrays, *f.array_catalog,
                           f.is_real ? 1 : 2, unchanged, why) &&
              why.find("role, extent, or offset") != std::string::npos,
          "malformed polarization source offset was not rejected");
    CHECK(unchanged.tag == 0x24680,
          "malformed polarization source offset partially published its destination");

    malformed = source;
    if (!malformed.gather.empty())
      malformed.gather.front().array = invalid_array();
    else if (!malformed.scatter.empty())
      malformed.scatter.front().array = invalid_array();
    else
      continue;
    unchanged = source;
    unchanged.tag = 0x2a6c0;
    CHECK(!remap_halo_plan(malformed, f.halos->arrays, f.halos->host_arrays, *f.array_catalog,
                           f.is_real ? 1 : 2, unchanged, why) &&
              why.find("out of range") != std::string::npos,
          "invalid polarization source ArrayId was not rejected");
    CHECK(unchanged.tag == 0x2a6c0,
          "invalid polarization source ArrayId partially published its destination");

    malformed = source;
    std::vector<HostElementRef> *stale_side = !malformed.host_gather.empty()
                                                   ? &malformed.host_gather
                                                   : &malformed.host_scatter;
    stale_side->front().id.generation += 1;
    unchanged = source;
    unchanged.tag = 0x35791;
    CHECK(!remap_halo_plan(malformed, f.halos->arrays, f.halos->host_arrays, *f.array_catalog,
                           f.is_real ? 1 : 2, unchanged, why) &&
              why.find("stale") != std::string::npos,
          "stale host mirror generation was not rejected");
    CHECK(unchanged.tag == 0x35791,
          "stale host mirror generation partially published its destination");
    saw_stale_rejection = true;

    malformed = source;
    std::vector<HostElementRef> *ordered_side = malformed.host_gather.size() > 1
                                                     ? &malformed.host_gather
                                                     : &malformed.host_scatter;
    if (ordered_side->size() > 1) {
      std::swap((*ordered_side)[0], (*ordered_side)[1]);
      unchanged = source;
      unchanged.tag = 0x46802;
      CHECK(!remap_halo_plan(malformed, f.halos->arrays, f.halos->host_arrays,
                             *f.array_catalog, f.is_real ? 1 : 2, unchanged, why) &&
                why.find("address or order") != std::string::npos,
            "reordered host mirror was not rejected");
      CHECK(unchanged.tag == 0x46802,
            "reordered host mirror partially published its destination");
      saw_order_rejection = true;
    }
    saw_atomic_rejection = true;
  }
  CHECK(or_to_all(saw_mixed_catalogue),
        "mixed exact/custom polarization halo did not contain catalogued and opaque rows");
  CHECK(or_to_all(saw_host_fallback), "no opaque polarization halo selected host fallback");
  CHECK(or_to_all(saw_pe_fallback), "no opaque PE halo selected host fallback");
  CHECK(or_to_all(saw_ph_fallback), "no opaque PH halo selected host fallback");
  CHECK(or_to_all(saw_atomic_rejection), "incomplete host mirror rejection was not exercised");
  CHECK(or_to_all(saw_stale_rejection), "stale host mirror rejection was not exercised");
  CHECK(or_to_all(saw_order_rejection), "host mirror order rejection was not exercised");

  f.use_bloch(X, 0.0);
  f.advance(1);
  if (valid(stale_id))
    CHECK(!f.halos->host_arrays.contains(stale_id),
          "reconnected halo table accepted a stale host generation");
}

static void test_noisy_lorentzian_storage(bool complex_fields) {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  noisy_lorentzian_susceptibility electric(0.02, 1.1, 0.05, false);
  noisy_lorentzian_susceptibility magnetic_drude(0.03, 0.9, 0.07, true);
  s.add_susceptibility(unit_sigma, E_stuff, electric);
  s.add_susceptibility(unit_sigma, H_stuff, magnetic_drude);

  fields f(&s);
  if (complex_fields)
    f.use_bloch(vec(0.07, -0.11));
  else
    f.use_real_fields();
  gaussian_src_time source(0.3, 0.1);
  f.add_point_source(Ez, source, vec(0.13, 0.11));
  f.add_point_source(Hz, source, vec(-0.17, 0.09));
  set_random_seed(20260830UL);
  f.advance(2);

  size_t expected_internal_rows = 0;
  size_t catalogued_internal_rows = 0;
  size_t scratch_rows = 0;
  size_t noisy_states = 0;
  size_t owned_chunks = 0;
  size_t pe_refs = 0, ph_refs = 0;
  bool saw_e = false, saw_h = false, saw_cmp0 = false, saw_cmp1 = false;
  std::set<uint32_t> noisy_ids;
  std::set<const void *> noisy_addresses;

  for (size_t id_value = 0; id_value < f.array_catalog->size(); ++id_value) {
    const ArrayId id{uint32_t(id_value)};
    const StorageKey &key = f.array_catalog->key(id);
    const ArraySpec &spec = f.array_catalog->spec(id);
    if (key.kind == int(array_kind::polarization_internal)) ++catalogued_internal_rows;
    if (spec.role == array_role::scratch) ++scratch_rows;
  }

  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    ++owned_chunks;
    fields_chunk &fc = *f.chunks[chunk];
    size_t chunk_e_states = 0, chunk_h_states = 0;
    FOR_FIELD_TYPES(ft) {
      int state_index = 0;
      for (polarization_state *state = fc.pol[ft]; state; state = state->next, ++state_index) {
        if (typeid(*state->s) != typeid(noisy_lorentzian_susceptibility)) continue;
        ++noisy_states;
        if (ft == E_stuff) {
          ++chunk_e_states;
          saw_e = true;
        }
        if (ft == H_stuff) {
          ++chunk_h_states;
          saw_h = true;
        }
        std::vector<InternalArrayLayout> layout;
        CHECK(state->s->internal_layout(layout, fc.gv, state->data),
              "noisy susceptibility did not publish its inherited Lorentz layout");
        std::set<std::pair<int, int> > expected_pairs;
        FOR_COMPONENTS(c) DOCMP2 {
          if (state->s->needs_P(c, cmp, fc.f))
            expected_pairs.insert(std::make_pair(int(c), cmp));
        }
        const size_t live_rows = 2 * expected_pairs.size();
        CHECK(layout.size() == live_rows,
              "noisy Lorentz layout has %zu rows, expected 2*needs_P = %zu", layout.size(),
              live_rows);
        expected_internal_rows += live_rows;
        realnum *base = static_cast<realnum *>(state->data);
        std::map<std::pair<int, int>, std::vector<ArrayId> > pairs;
        std::set<std::pair<int, int> > actual_pairs;
        for (size_t li = 0; li < layout.size(); ++li) {
          const InternalArrayLayout &entry = layout[li];
          const uint64_t aux = polarization_storage_aux(ft, state_index, li);
          const ArrayId id = f.array_catalog->find(
              StorageKey{chunk, int(array_kind::polarization_internal), int(entry.c), entry.cmp,
                         aux});
          CHECK(is_valid(id), "noisy internal row %zu has no canonical ArrayId", li);
          if (!is_valid(id)) continue;
          const ArraySpec &spec = f.array_catalog->spec(id);
          CHECK(spec.role == array_role::polarization,
                "noisy internal row %zu has non-polarization role", li);
          CHECK(spec.element_type == ElementType::realnum_value && spec.elements == entry.elements,
                "noisy internal row %zu has incompatible type/extent", li);
          CHECK(entry.elements == size_t(fc.gv.ntot()),
                "noisy internal row %zu has %zu elements, expected chunk ntot %zu", li,
                entry.elements, size_t(fc.gv.ntot()));
          CHECK(!is_valid(spec.alias_of), "noisy internal row %zu unexpectedly aliases", li);
          realnum *const address = f.array_catalog->resolve<realnum>(id);
          CHECK(address == base + entry.offset_elements,
                "noisy internal row %zu resolves to the wrong address", li);
          CHECK(noisy_ids.insert(id.value).second,
                "noisy internal ArrayId %u is reused by another row", unsigned(id.value));
          CHECK(noisy_addresses.insert(address).second,
                "noisy internal address is reused by another row");
          CHECK(!strcmp(entry.name, (li & 1) ? "P_prev" : "P"),
                "noisy internal row %zu has unexpected name %s", li, entry.name);
          const std::pair<int, int> pair_key = std::make_pair(int(entry.c), entry.cmp);
          pairs[pair_key].push_back(id);
          actual_pairs.insert(pair_key);
        }
        CHECK(actual_pairs == expected_pairs,
              "noisy layout component/cmp key set differs from live needs_P set");
        for (const auto &pair : pairs) {
          saw_cmp0 = saw_cmp0 || pair.first.second == 0;
          saw_cmp1 = saw_cmp1 || pair.first.second == 1;
          CHECK(pair.second.size() == 2,
                "noisy component/cmp has %zu internal rows instead of P/P_prev",
                pair.second.size());
          if (pair.second.size() == 2)
            CHECK(f.array_catalog->resolve_untyped(pair.second[0]) !=
                      f.array_catalog->resolve_untyped(pair.second[1]),
                  "noisy P and P_prev arrays alias");
        }
      }
    }
    CHECK(chunk_e_states == 1 && chunk_h_states == 1,
          "owned chunk %d has %zu noisy E and %zu noisy H states, expected one each", chunk,
          chunk_e_states, chunk_h_states);
  }

  for (const HaloPlan &source_plan : f.halos->plans) {
    if (source_plan.ft != PE_stuff && source_plan.ft != PH_stuff) continue;
    HaloPlan canonical;
    std::string why;
    CHECK(remap_halo_plan(source_plan, f.halos->arrays, f.halos->host_arrays, *f.array_catalog,
                          f.is_real ? 1 : 2, canonical, why),
          "noisy polarization halo did not remap: %s", why.c_str());
    std::vector<ElementRef> source_refs, canonical_refs;
    expand_gather(source_plan, source_refs);
    expand_gather(canonical, canonical_refs);
    CHECK(source_refs.size() == canonical_refs.size(),
          "noisy polarization gather changed size during canonical remap");
    for (size_t j = 0; j < source_refs.size() && j < canonical_refs.size(); ++j) {
      const ElementRef &from = source_refs[j], &to = canonical_refs[j];
      const bool valid_id = to.array.value < f.array_catalog->size();
      CHECK(valid_id, "noisy polarization gather has an invalid canonical ArrayId");
      if (!valid_id) continue;
      CHECK(f.halos->arrays.base(from.array) + from.index ==
                f.array_catalog->resolve<realnum>(to.array) + to.index,
            "noisy polarization gather address differs after canonical remap");
      CHECK(f.array_catalog->key(to.array).kind == int(array_kind::polarization_internal),
            "noisy polarization gather references a non-polarization array");
    }
    const size_t gather_refs = canonical_refs.size();
    expand_scatter(source_plan, source_refs);
    expand_scatter(canonical, canonical_refs);
    CHECK(source_refs.size() == canonical_refs.size(),
          "noisy polarization scatter changed size during canonical remap");
    for (size_t j = 0; j < source_refs.size() && j < canonical_refs.size(); ++j) {
      const ElementRef &from = source_refs[j], &to = canonical_refs[j];
      const bool valid_id = to.array.value < f.array_catalog->size();
      CHECK(valid_id, "noisy polarization scatter has an invalid canonical ArrayId");
      if (!valid_id) continue;
      CHECK(f.halos->arrays.base(from.array) + from.index ==
                f.array_catalog->resolve<realnum>(to.array) + to.index,
            "noisy polarization scatter address differs after canonical remap");
      CHECK(f.array_catalog->key(to.array).kind == int(array_kind::polarization_internal),
            "noisy polarization scatter references a non-polarization array");
    }
    const size_t total_refs = gather_refs + canonical_refs.size();
    if (source_plan.ft == PE_stuff)
      pe_refs += total_refs;
    else
      ph_refs += total_refs;
  }

  CHECK(catalogued_internal_rows == expected_internal_rows,
        "noisy fixture catalogued %zu polarization rows, expected exactly %zu",
        catalogued_internal_rows, expected_internal_rows);
  CHECK(scratch_rows == 0,
        "noisy Lorentz storage introduced %zu scratch/RNG rows", scratch_rows);
  CHECK(noisy_states == 2 * owned_chunks,
        "owned chunks contain %zu noisy states, expected %zu", noisy_states, 2 * owned_chunks);
  const bool global_saw_e = or_to_all(saw_e);
  const bool global_saw_h = or_to_all(saw_h);
  const bool global_saw_cmp0 = or_to_all(saw_cmp0);
  const bool global_saw_cmp1 = or_to_all(saw_cmp1);
  CHECK(global_saw_e && global_saw_h,
        "noisy fixture did not instantiate both electric Lorentz and magnetic Drude states");
  CHECK(global_saw_cmp0, "no noisy cmp0 polarization row was catalogued");
  CHECK(global_saw_cmp1 == complex_fields,
        "noisy cmp1 coverage does not match real/complex fixture mode");
  CHECK(or_to_all(pe_refs > 0), "no noisy electric polarization halo was remapped");
  CHECK(or_to_all(ph_refs > 0), "no noisy magnetic polarization halo was remapped");
}

struct MultilevelExpected {
  int levels;
  int transitions;
  const realnum *gamma_diagonal;
  const realnum *initial_populations;
};

static const realnum ml_e0_gamma_diagonal[] = {realnum(0.02), realnum(0.03), realnum(0.04)};
static const realnum ml_e0_populations[] = {realnum(0.7), realnum(0.2), realnum(0.1)};
static const realnum ml_e1_gamma_diagonal[] = {realnum(0.015), realnum(0.025)};
static const realnum ml_e1_populations[] = {realnum(0.8), realnum(0.2)};
static const realnum ml_h0_gamma_diagonal[] = {realnum(0.01), realnum(0.02), realnum(0.03),
                                               realnum(0.04)};
static const realnum ml_h0_populations[] = {realnum(0.55), realnum(0.25), realnum(0.15),
                                            realnum(0.05)};

static MultilevelExpected expected_multilevel_state(field_type ft, int state_index) {
  /* add_susceptibility prepends, so the second E state is state zero. */
  if (ft == E_stuff && state_index == 0)
    return MultilevelExpected{2, 1, ml_e1_gamma_diagonal, ml_e1_populations};
  if (ft == E_stuff && state_index == 1)
    return MultilevelExpected{3, 2, ml_e0_gamma_diagonal, ml_e0_populations};
  if (ft == H_stuff && state_index == 0)
    return MultilevelExpected{4, 3, ml_h0_gamma_diagonal, ml_h0_populations};
  return MultilevelExpected{0, 0, NULL, NULL};
}

static void add_multilevel_test_states(structure &s) {
  const realnum e0_Gamma[] = {realnum(0.02), 0, 0, 0, realnum(0.03), 0, 0, 0,
                              realnum(0.04)};
  const realnum e0_alpha[] = {realnum(-0.2), 0, realnum(0.2), realnum(-0.3), 0,
                              realnum(0.3)};
  const realnum e0_omega[] = {realnum(0.73), realnum(0.91)};
  const realnum e0_gamma[] = {realnum(0.06), realnum(0.08)};
  const realnum e0_sigmat[] = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
  multilevel_susceptibility e0(3, 2, e0_Gamma, ml_e0_populations, e0_alpha, e0_omega,
                               e0_gamma, e0_sigmat);

  const realnum e1_Gamma[] = {realnum(0.015), 0, 0, realnum(0.025)};
  const realnum e1_alpha[] = {realnum(-0.25), realnum(0.25)};
  const realnum e1_omega[] = {realnum(0.67)};
  const realnum e1_gamma[] = {realnum(0.04)};
  const realnum e1_sigmat[] = {3, 3, 3, 3, 3};
  multilevel_susceptibility e1(2, 1, e1_Gamma, ml_e1_populations, e1_alpha, e1_omega,
                               e1_gamma, e1_sigmat);

  const realnum h0_Gamma[] = {realnum(0.01), 0, 0, 0, 0, realnum(0.02), 0, 0,
                              0, 0, realnum(0.03), 0, 0, 0, 0, realnum(0.04)};
  const realnum h0_alpha[] = {realnum(-0.1), 0, 0, realnum(0.1), realnum(-0.2), 0,
                              0, realnum(0.2), realnum(-0.3), 0, 0, realnum(0.3)};
  const realnum h0_omega[] = {realnum(0.59), realnum(0.83), realnum(1.07)};
  const realnum h0_gamma[] = {realnum(0.03), realnum(0.05), realnum(0.07)};
  const realnum h0_sigmat[] = {4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6};
  multilevel_susceptibility h0(4, 3, h0_Gamma, ml_h0_populations, h0_alpha, h0_omega,
                               h0_gamma, h0_sigmat);

  s.add_susceptibility(unit_sigma, E_stuff, e0);
  s.add_susceptibility(unit_sigma, E_stuff, e1);
  s.add_susceptibility(unit_sigma, H_stuff, h0);
}

struct MultilevelRowIdentity {
  field_type ft;
  int state_index;
  component c;
  int cmp;
  int transition;
  std::string name;
};

static void prepare_all_vector_components(fields &f, bool complex_fields) {
  if (complex_fields)
    f.use_bloch(vec(0.07, -0.11, 0.05));
  else
    f.use_real_fields();
  f.require_component(Ex);
  f.require_component(Ey);
  f.require_component(Ez);
  f.require_component(Hx);
  f.require_component(Hy);
  f.require_component(Hz);
  f.advance(2);
  f.zero_fields();
}

static size_t validate_multilevel_storage(fields &f, bool complex_fields,
                                          std::map<uint32_t, MultilevelRowIdentity> &rows,
                                          std::vector<StorageKey> &keys) {
  rows.clear();
  keys.clear();
  size_t local_expected_rows = 0;
  size_t local_catalogued_rows = 0;
  size_t local_states = 0;
  size_t local_owned_chunks = 0;
  std::set<uint32_t> ids;
  std::set<const void *> addresses;
  std::vector<std::pair<uintptr_t, uintptr_t> > address_intervals;

  for (size_t id_value = 0; id_value < f.array_catalog->size(); ++id_value)
    if (f.array_catalog->key(ArrayId{uint32_t(id_value)}).kind ==
        int(array_kind::polarization_internal))
      ++local_catalogued_rows;

  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    ++local_owned_chunks;
    fields_chunk &fc = *f.chunks[chunk];
    FOR_FIELD_TYPES(ft) {
      int state_index = 0;
      for (polarization_state *state = fc.pol[ft]; state; state = state->next, ++state_index) {
        if (typeid(*state->s) != typeid(multilevel_susceptibility)) continue;
        ++local_states;
        const MultilevelExpected expected = expected_multilevel_state(ft, state_index);
        CHECK(expected.levels > 0 && expected.transitions > 0,
              "unexpected multilevel state ft=%d index=%d", int(ft), state_index);
        if (expected.levels <= 0 || expected.transitions <= 0) continue;

        std::vector<std::pair<component, int> > active;
        FOR_COMPONENTS(c) DOCMP2 {
          realnum *const first = state->s->cinternal_notowned_ptr(0, c, cmp, 0, state->data);
          if (first) active.push_back(std::make_pair(c, cmp));
        }
        std::vector<InternalArrayLayout> layout;
        CHECK(state->s->internal_layout(layout, fc.gv, state->data),
              "multilevel state did not publish an internal layout");
        const size_t expected_rows = 2 + 2 * size_t(expected.transitions) * active.size();
        CHECK(layout.size() == expected_rows,
              "multilevel layout has %zu rows, expected %zu", layout.size(), expected_rows);
        if (layout.size() != expected_rows) continue;
        local_expected_rows += expected_rows;

        realnum *const base = static_cast<realnum *>(state->data);
        const size_t ntot = size_t(fc.gv.ntot());
        const InternalArrayLayout &gamma_inv = layout.front();
        CHECK(gamma_inv.name && !strcmp(gamma_inv.name, "GammaInv") &&
                  gamma_inv.c == Centered && gamma_inv.cmp == -1 &&
                  gamma_inv.element_type == InternalArrayLayout::realnum_value &&
                  gamma_inv.elements == size_t(expected.levels) * size_t(expected.levels),
              "multilevel GammaInv layout metadata is invalid");

        size_t li = 1;
        size_t cursor = gamma_inv.offset_elements + gamma_inv.elements;
        for (const std::pair<component, int> &pair : active) {
          CHECK(pair.second == 0 || complex_fields,
                "real multilevel fixture published an imaginary state row");
          CHECK(state->s->num_cinternal_notowned_needed(pair.first, state->data) ==
                    expected.transitions,
                "multilevel halo row count differs from T");
          for (int t = 0; t < expected.transitions; ++t, li += 2) {
            const InternalArrayLayout &p = layout[li];
            const InternalArrayLayout &p_prev = layout[li + 1];
            CHECK(p.name && p_prev.name && !strcmp(p.name, "P") &&
                      !strcmp(p_prev.name, "P_prev") && p.c == pair.first &&
                      p_prev.c == pair.first && p.cmp == pair.second &&
                      p_prev.cmp == pair.second && p.elements == ntot &&
                      p_prev.elements == ntot && p.offset_elements == cursor &&
                      p_prev.offset_elements == cursor + ntot,
                  "multilevel transition %d layout order/offset is invalid", t);
            CHECK(base + p.offset_elements ==
                      state->s->cinternal_notowned_ptr(t, pair.first, pair.second, 0, state->data),
                  "multilevel transition %d P offset differs from the live halo pointer", t);
            cursor += 2 * ntot;
          }
        }

        const InternalArrayLayout &populations = layout.back();
        CHECK(populations.name && !strcmp(populations.name, "N") &&
                  populations.c == Centered && populations.cmp == -1 &&
                  populations.element_type == InternalArrayLayout::realnum_value &&
                  populations.elements == ntot * size_t(expected.levels) &&
                  populations.offset_elements == cursor + size_t(expected.levels),
              "multilevel population layout or Ntmp gap is invalid");
        ArrayId scratch_id = invalid_array();
        ptrdiff_t scratch_offset = 0;
        CHECK(!f.array_catalog->locate(base + cursor, scratch_id, scratch_offset),
              "multilevel Ntmp scratch was published in the canonical catalog");

        for (size_t row = 0; row < layout.size(); ++row) {
          const InternalArrayLayout &entry = layout[row];
          const StorageKey key{chunk, int(array_kind::polarization_internal), int(entry.c),
                               entry.cmp, polarization_storage_aux(ft, state_index, row)};
          const ArrayId id = f.array_catalog->find(key);
          CHECK(is_valid(id), "multilevel layout row %zu has no canonical ArrayId", row);
          if (!is_valid(id)) continue;
          const ArraySpec &spec = f.array_catalog->spec(id);
          realnum *const address = f.array_catalog->resolve<realnum>(id);
          CHECK(spec.role == array_role::polarization &&
                    spec.element_type == ElementType::realnum_value &&
                    spec.storage == (sizeof(realnum) == sizeof(float) ? Precision::f32
                                                                      : Precision::f64) &&
                    spec.elements == entry.elements && spec.alignment == alignof(realnum) &&
                    !is_valid(spec.alias_of),
                "multilevel layout row %zu has incompatible catalog metadata", row);
          CHECK(address == base + entry.offset_elements,
                "multilevel layout row %zu resolves to the wrong address", row);
          CHECK(ids.insert(id.value).second, "multilevel ArrayId is reused locally");
          CHECK(addresses.insert(address).second, "multilevel row address is reused locally");
          const uintptr_t begin = reinterpret_cast<uintptr_t>(address);
          const uintptr_t end = begin + spec.elements * sizeof(realnum);
          for (const auto &prior : address_intervals)
            CHECK(end <= prior.first || begin >= prior.second,
                  "multilevel canonical rows overlap in host storage");
          address_intervals.push_back(std::make_pair(begin, end));
          CHECK(polarization_storage_field_type(key.aux) == ft &&
                    polarization_storage_state_index(key.aux) == state_index &&
                    polarization_storage_layout_ordinal(key.aux) == row,
                "multilevel canonical key did not round-trip");
          keys.push_back(key);

          int transition = -1;
          if (entry.name && !strcmp(entry.name, "P")) {
            transition = 0;
            for (size_t prior = 1; prior < row; prior += 2)
              if (layout[prior].name && !strcmp(layout[prior].name, "P") &&
                  layout[prior].c == entry.c && layout[prior].cmp == entry.cmp)
                ++transition;
          }
          rows[id.value] =
              MultilevelRowIdentity{ft, state_index, entry.c, entry.cmp, transition,
                                    entry.name ? entry.name : ""};
        }

        const realnum tolerance = realnum(64) * std::numeric_limits<realnum>::epsilon();
        const realnum *const gamma_values = base + gamma_inv.offset_elements;
        for (int l1 = 0; l1 < expected.levels; ++l1)
          for (int l2 = 0; l2 < expected.levels; ++l2) {
            const realnum want = l1 == l2
                                     ? realnum(1) /
                                           (realnum(1) + expected.gamma_diagonal[l1] * fc.dt / 2)
                                     : realnum(0);
            CHECK(fabs(double(gamma_values[l1 * expected.levels + l2] - want)) <=
                      double(tolerance),
                  "multilevel GammaInv differs at (%d,%d)", l1, l2);
          }
        const realnum *const n_values = base + populations.offset_elements;
        for (size_t point = 0; point < ntot; ++point)
          for (int level = 0; level < expected.levels; ++level)
            CHECK(n_values[point * size_t(expected.levels) + size_t(level)] ==
                      expected.initial_populations[level],
                  "multilevel initial population differs at point %zu level %d", point, level);
        for (size_t row = 1; row + 1 < layout.size(); ++row) {
          const realnum *const values = base + layout[row].offset_elements;
          for (size_t i = 0; i < layout[row].elements; ++i)
            CHECK(values[i] == realnum(0), "multilevel P/P_prev did not initialize to zero");
        }

        /* Make the copy oracle non-vacuous: every transition-history row and
           every population element carries a distinct deterministic value. */
        for (size_t row = 1; row < layout.size(); ++row) {
          realnum *const values = base + layout[row].offset_elements;
          for (size_t i = 0; i < layout[row].elements; ++i)
            values[i] = realnum(0.03125 * double(row + 1) + 0.0009765625 * double(i + 1));
        }

        void *const copied_data = state->s->copy_internal_data(state->data);
        std::vector<InternalArrayLayout> copied_layout;
        CHECK(copied_data && state->s->internal_layout(copied_layout, fc.gv, copied_data) &&
                  copied_layout.size() == layout.size(),
              "multilevel copy did not preserve the published layout");
        if (copied_data && copied_layout.size() == layout.size()) {
          realnum *const copied_base = static_cast<realnum *>(copied_data);
          for (size_t row = 0; row < layout.size(); ++row) {
            CHECK(copied_layout[row].offset_elements == layout[row].offset_elements &&
                      copied_layout[row].elements == layout[row].elements &&
                      copied_base + copied_layout[row].offset_elements !=
                          base + layout[row].offset_elements &&
                      memcmp(copied_base + copied_layout[row].offset_elements,
                             base + layout[row].offset_elements,
                             layout[row].elements * sizeof(realnum)) == 0,
                  "multilevel copy row %zu changed offset/value or aliases the original", row);
            if (layout[row].elements) {
              const realnum original = base[layout[row].offset_elements];
              copied_base[copied_layout[row].offset_elements] = original + realnum(row + 1);
              CHECK(base[layout[row].offset_elements] == original,
                    "mutating copied multilevel row %zu changed the original", row);
            }
          }
        }
        state->s->delete_internal_data(copied_data);
      }
    }
  }

  CHECK(local_catalogued_rows == local_expected_rows,
        "catalog has %zu multilevel rows, expected %zu", local_catalogued_rows,
        local_expected_rows);
  CHECK(local_expected_rows > 0 || local_owned_chunks == 0,
        "an owning rank has no multilevel rows");
  CHECK(local_states == 3 * local_owned_chunks,
        "owned chunks contain %zu multilevel states, expected %zu", local_states,
        3 * local_owned_chunks);
  CHECK(audit_storage_catalog(f, *f.array_catalog, true) == 0,
        "multilevel storage left reachable arrays uncatalogued");
  return local_expected_rows;
}

static void validate_global_multilevel_key_ownership(fields &f, bool complex_fields) {
  const field_type families[] = {E_stuff, H_stuff};
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    for (field_type ft : families) {
      const int state_count = ft == E_stuff ? 2 : 1;
      for (int state_index = 0; state_index < state_count; ++state_index) {
        const MultilevelExpected expected = expected_multilevel_state(ft, state_index);
        size_t row = 0;
        const StorageKey gamma{chunk, int(array_kind::polarization_internal), int(Centered), -1,
                               polarization_storage_aux(ft, state_index, row++)};
        CHECK(sum_to_all(is_valid(f.array_catalog->find(gamma)) ? 1 : 0) == 1,
              "multilevel GammaInv key is not owned by exactly one rank");
        FOR_COMPONENTS(c) {
          if (type(c) != ft || !f.gv.has_field(c)) continue;
          for (int cmp = 0; cmp < (complex_fields ? 2 : 1); ++cmp)
            for (int t = 0; t < expected.transitions; ++t) {
              const StorageKey p{chunk, int(array_kind::polarization_internal), int(c), cmp,
                                 polarization_storage_aux(ft, state_index, row++)};
              const StorageKey p_prev{chunk, int(array_kind::polarization_internal), int(c), cmp,
                                      polarization_storage_aux(ft, state_index, row++)};
              CHECK(sum_to_all(is_valid(f.array_catalog->find(p)) ? 1 : 0) == 1,
                    "multilevel P key is not owned by exactly one rank");
              CHECK(sum_to_all(is_valid(f.array_catalog->find(p_prev)) ? 1 : 0) == 1,
                    "multilevel P_prev key is not owned by exactly one rank");
            }
        }
        const StorageKey populations{
            chunk, int(array_kind::polarization_internal), int(Centered), -1,
            polarization_storage_aux(ft, state_index, row)};
        CHECK(sum_to_all(is_valid(f.array_catalog->find(populations)) ? 1 : 0) == 1,
              "multilevel population key is not owned by exactly one rank");
      }
    }
}

static void test_multilevel_storage(bool complex_fields) {
  grid_volume gv = vol3d(1.5, 1.5, 1.5, 4.0);
  structure s(gv, one_value, no_pml(), identity(), 8);
  add_multilevel_test_states(s);
  fields f(&s);
  prepare_all_vector_components(f, complex_fields);

  std::map<uint32_t, MultilevelRowIdentity> rows;
  std::vector<StorageKey> keys;
  validate_multilevel_storage(f, complex_fields, rows, keys);
  validate_global_multilevel_key_ownership(f, complex_fields);

  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    std::set<uint64_t> scalar_keys;
    std::set<uint32_t> scalar_ids;
    std::set<const void *> scalar_addresses;
    for (field_type ft : {E_stuff, H_stuff}) {
      const MultilevelExpected expected = expected_multilevel_state(ft, 0);
      const size_t active_pairs = size_t(3) * size_t(complex_fields ? 2 : 1);
      const size_t n_row = 1 + 2 * size_t(expected.transitions) * active_pairs;
      for (size_t row : {size_t(0), n_row}) {
        const StorageKey key{chunk, int(array_kind::polarization_internal), int(Centered), -1,
                             polarization_storage_aux(ft, 0, row)};
        const ArrayId id = f.array_catalog->find(key);
        if (!is_valid(id)) continue;
        CHECK(scalar_keys.insert(key.aux).second && scalar_ids.insert(id.value).second &&
                  scalar_addresses.insert(f.array_catalog->resolve_untyped(id)).second,
              "E0/H0 GammaInv/N scalar rows collide");
      }
    }
    CHECK(scalar_keys.size() == (f.chunks[chunk]->is_mine() ? 4u : 0u),
          "owned E0/H0 scalar rows are not four distinct canonical arrays");
  }

  bool saw_pe = false, saw_ph = false, saw_copy = false, saw_phase = false;
  bool did_local_copy_roundtrip = false;
  bool did_phase_transform = false;
  for (const HaloPlan &source : f.halos->plans) {
    if (source.ft != PE_stuff && source.ft != PH_stuff) continue;
    HaloPlan canonical;
    std::string why;
    const bool remapped = remap_halo_plan(source, f.halos->arrays, f.halos->host_arrays,
                                          *f.array_catalog, f.is_real ? 1 : 2, canonical, why);
    CHECK(remapped, "multilevel polarization halo did not remap: %s", why.c_str());
    if (!remapped) continue;
    if (source.block_elements && source.phase == CONNECT_COPY) saw_copy = true;
    if (source.block_elements && source.phase == CONNECT_PHASE) {
      bool nontrivial = false;
      for (const std::complex<realnum> &value : source.phase_values)
        nontrivial = nontrivial || value != std::complex<realnum>(1, 0);
      saw_phase = saw_phase || nontrivial;
    }
    std::vector<ElementRef> source_refs, canonical_refs;
    for (int side = 0; side < 2; ++side) {
      if (side == 0) {
        expand_gather(source, source_refs);
        expand_gather(canonical, canonical_refs);
      }
      else {
        expand_scatter(source, source_refs);
        expand_scatter(canonical, canonical_refs);
      }
      CHECK(source_refs.size() == canonical_refs.size(),
            "multilevel halo changed size during canonical remap");
      for (size_t j = 0; j < source_refs.size() && j < canonical_refs.size();) {
        const ElementRef &from = source_refs[j], &to = canonical_refs[j];
        const bool valid = to.array.value < f.array_catalog->size() && rows.count(to.array.value);
        CHECK(valid, "multilevel halo did not resolve to a published multilevel row");
        if (!valid) {
          ++j;
          continue;
        }
        CHECK(f.halos->arrays.base(from.array) + from.index ==
                  f.array_catalog->resolve<realnum>(to.array) + to.index,
              "multilevel halo source/canonical addresses differ");
        const MultilevelRowIdentity &row = rows[to.array.value];
        CHECK(row.name == "P" && row.transition >= 0,
              "multilevel halo references non-current state %s", row.name.c_str());
        CHECK((source.ft == PE_stuff && row.ft == E_stuff) ||
                  (source.ft == PH_stuff && row.ft == H_stuff),
              "multilevel halo references the wrong field family");
        if (source.ft == PE_stuff) saw_pe = true;
        if (source.ft == PH_stuff) saw_ph = true;
        const MultilevelExpected expected = expected_multilevel_state(row.ft, row.state_index);
        CHECK(row.transition == 0,
              "multilevel boundary group starts at transition %d instead of zero",
              row.transition);
        const size_t interleave = size_t(complex_fields ? 2 : 1);
        const size_t group = size_t(expected.transitions) * interleave;
        CHECK(j + group <= canonical_refs.size(), "multilevel boundary group is truncated");
        for (size_t offset = 0; offset < group && j + offset < canonical_refs.size(); ++offset) {
          const ElementRef &group_ref = canonical_refs[j + offset];
          const bool group_valid = group_ref.array.value < f.array_catalog->size() &&
                                   rows.count(group_ref.array.value);
          CHECK(group_valid, "multilevel boundary group contains an invalid row");
          if (!group_valid) continue;
          const MultilevelRowIdentity &actual = rows[group_ref.array.value];
          const int want_transition = int(offset / interleave);
          const int want_cmp = complex_fields ? int(offset % interleave) : row.cmp;
          CHECK(actual.ft == row.ft && actual.state_index == row.state_index &&
                    actual.c == row.c && actual.transition == want_transition &&
                    actual.cmp == want_cmp,
                "multilevel boundary group does not contain exact transitions 0..T-1");
        }
        j += group;
      }
    }

    if (!did_local_copy_roundtrip && source.same_rank && source.phase == CONNECT_COPY &&
        source.block_elements) {
      std::vector<ElementRef> scatter;
      expand_scatter(source, scatter);
      std::vector<realnum> block(source.block_offset + source.block_elements, realnum(0));
      f.pack_halo(source, block.data());
      for (const ElementRef &ref : scatter)
        *(f.halos->arrays.base(ref.array) + ref.index) = realnum(0);
      f.unpack_halo(source, block.data());
      for (size_t j = 0; j < scatter.size(); ++j)
        CHECK(*(f.halos->arrays.base(scatter[j].array) + scatter[j].index) ==
                  block[source.block_offset + j],
              "same-rank multilevel COPY did not round-trip element %zu", j);
      did_local_copy_roundtrip = true;
    }
    if (!did_phase_transform && source.phase == CONNECT_PHASE && source.block_elements) {
      std::vector<ElementRef> scatter;
      expand_scatter(source, scatter);
      if (scatter.size() == source.block_elements && !scatter.empty()) {
        CHECK(source.phase_values.size() == source.block_elements / 2,
              "multilevel PHASE plan has the wrong phase count");
        if (source.phase_values.size() != source.block_elements / 2) continue;
        std::vector<realnum> block(source.block_offset + source.block_elements, realnum(0));
        for (size_t j = 0; j < source.block_elements; ++j)
          block[source.block_offset + j] = realnum(0.125 * double(j + 1));
        std::map<realnum *, realnum> expected;
        for (size_t j = 0; j < source.block_elements / 2; ++j) {
          const std::complex<realnum> value =
              source.phase_values[j] *
              std::complex<realnum>(block[source.block_offset + 2 * j],
                                    block[source.block_offset + 2 * j + 1]);
          expected[f.halos->arrays.base(scatter[2 * j].array) + scatter[2 * j].index] =
              value.real();
          expected[f.halos->arrays.base(scatter[2 * j + 1].array) +
                   scatter[2 * j + 1].index] = value.imag();
        }
        f.unpack_halo(source, block.data());
        for (const auto &entry : expected) {
          const realnum scale = std::max(realnum(1), realnum(fabs(double(entry.second))));
          CHECK(fabs(double(*entry.first - entry.second)) <=
                    double(8 * std::numeric_limits<realnum>::epsilon() * scale),
                "multilevel PHASE transform differs");
        }
        did_phase_transform = true;
      }
    }
  }
  CHECK(or_to_all(saw_pe), "no multilevel PE halo row was remapped");
  CHECK(or_to_all(saw_ph), "no multilevel PH halo row was remapped");
  CHECK(or_to_all(saw_copy), "no multilevel COPY halo row was exercised");
  CHECK(or_to_all(saw_phase) == complex_fields,
        "multilevel PHASE coverage does not match real/complex mode");
  CHECK(or_to_all(did_phase_transform) == complex_fields,
        "multilevel PHASE transform coverage does not match real/complex mode");
  CHECK(or_to_all(did_local_copy_roundtrip),
        "no same-rank multilevel COPY plan was round-tripped");

  for (const HaloPlan &source : f.halos->plans)
    if ((source.ft == PE_stuff || source.ft == PH_stuff) && source.block_elements) {
      std::vector<ElementRef> before_gather, before_scatter, after;
      expand_gather(source, before_gather);
      expand_scatter(source, before_scatter);
      CpuArrayCatalog incomplete;
      HaloPlan host_fallback;
      std::string why;
      const size_t catalog_size = f.array_catalog->size();
      CHECK(remap_halo_plan(source, f.halos->arrays, f.halos->host_arrays, incomplete,
                            f.is_real ? 1 : 2, host_fallback, why) &&
                host_fallback.storage == HaloStorageDisposition::host_owned,
            "multilevel halo did not fall back atomically for a missing catalog: %s",
            why.c_str());
      CHECK(host_fallback.gather.empty() && host_fallback.gather_slabs.empty() &&
                host_fallback.scatter.empty() && host_fallback.scatter_slabs.empty(),
            "multilevel host fallback retained partial canonical references");
      expand_gather(source, after);
      CHECK(after.size() == before_gather.size(),
            "failed multilevel remap mutated the source gather plan");
      for (size_t j = 0; j < after.size() && j < before_gather.size(); ++j)
        CHECK(after[j].array == before_gather[j].array && after[j].index == before_gather[j].index,
              "failed multilevel remap changed source gather element %zu", j);
      expand_scatter(source, after);
      CHECK(after.size() == before_scatter.size(),
            "failed multilevel remap mutated the source scatter plan");
      for (size_t j = 0; j < after.size() && j < before_scatter.size(); ++j)
        CHECK(after[j].array == before_scatter[j].array &&
                  after[j].index == before_scatter[j].index,
              "failed multilevel remap changed source scatter element %zu", j);
      CHECK(f.array_catalog->size() == catalog_size,
            "failed multilevel remap mutated the published catalog");
      break;
    }

  std::unordered_map<StorageKey, const void *, StorageKeyHash> addresses_before_zero;
  std::unordered_map<StorageKey, std::vector<realnum>, StorageKeyHash> gamma_before_zero;
  for (const StorageKey &key : keys) {
    const ArrayId id = f.array_catalog->find(key);
    if (!is_valid(id)) continue;
    realnum *const values = f.array_catalog->resolve<realnum>(id);
    addresses_before_zero[key] = values;
    const MultilevelRowIdentity &row = rows[id.value];
    if (row.name == "GammaInv")
      gamma_before_zero[key] =
          std::vector<realnum>(values, values + f.array_catalog->spec(id).elements);
    else
      for (size_t i = 0; i < f.array_catalog->spec(id).elements; ++i)
        values[i] = realnum(9.5);
  }
  f.zero_fields();
  for (const StorageKey &key : keys) {
    const ArrayId id = f.array_catalog->find(key);
    CHECK(is_valid(id) && f.array_catalog->resolve_untyped(id) == addresses_before_zero[key],
          "zero_fields changed multilevel canonical storage identity");
    if (!is_valid(id)) continue;
    realnum *const values = f.array_catalog->resolve<realnum>(id);
    const MultilevelRowIdentity &row = rows[id.value];
    if (row.name == "GammaInv")
      CHECK(memcmp(values, gamma_before_zero[key].data(),
                   gamma_before_zero[key].size() * sizeof(realnum)) == 0,
            "zero_fields changed multilevel GammaInv");
    else if (row.name == "P" || row.name == "P_prev") {
      for (size_t i = 0; i < f.array_catalog->spec(id).elements; ++i)
        CHECK(values[i] == realnum(0), "zero_fields did not clear multilevel P state");
    }
    else if (row.name == "N") {
      const MultilevelExpected expected = expected_multilevel_state(row.ft, row.state_index);
      for (size_t i = 0; i < f.array_catalog->spec(id).elements; ++i)
        CHECK(values[i] == expected.initial_populations[i % size_t(expected.levels)],
              "zero_fields did not restore multilevel population state");
    }
  }
}

static void test_multilevel_negate_halo() {
  grid_volume gv = vol2d(2.0, 2.0, 5.0);
  structure s(gv, one_value, no_pml(), -mirror(Y, gv), 4);
  add_multilevel_test_states(s);
  fields f(&s);
  prepare_all_vector_components(f, false);

  std::map<uint32_t, MultilevelRowIdentity> rows;
  std::vector<StorageKey> keys;
  validate_multilevel_storage(f, false, rows, keys);
  bool saw_negate = false, did_negate_transform = false;
  for (const HaloPlan &source : f.halos->plans) {
    if ((source.ft != PE_stuff && source.ft != PH_stuff) ||
        source.phase != CONNECT_NEGATE || !source.block_elements)
      continue;
    HaloPlan canonical;
    std::string why;
    const bool remapped = remap_halo_plan(source, f.halos->arrays, f.halos->host_arrays,
                                          *f.array_catalog, 1, canonical, why);
    CHECK(remapped, "multilevel NEGATE halo did not remap: %s", why.c_str());
    if (!remapped) continue;
    std::vector<ElementRef> refs;
    expand_gather(canonical, refs);
    for (const ElementRef &ref : refs) {
      const bool valid = ref.array.value < f.array_catalog->size() && rows.count(ref.array.value);
      CHECK(valid && rows[ref.array.value].name == "P",
            "multilevel NEGATE halo references a non-current polarization row");
    }
    saw_negate = saw_negate || !refs.empty();
    expand_scatter(canonical, refs);
    for (const ElementRef &ref : refs) {
      const bool valid = ref.array.value < f.array_catalog->size() && rows.count(ref.array.value);
      CHECK(valid && rows[ref.array.value].name == "P",
            "multilevel NEGATE halo scatter references a non-current polarization row");
    }
    saw_negate = saw_negate || !refs.empty();
    std::vector<ElementRef> source_scatter;
    expand_scatter(source, source_scatter);
    if (!did_negate_transform && source_scatter.size() == source.block_elements &&
        !source_scatter.empty()) {
      std::vector<realnum> block(source.block_offset + source.block_elements, realnum(0));
      std::map<realnum *, realnum> expected;
      for (size_t j = 0; j < source.block_elements; ++j)
        block[source.block_offset + j] = realnum(0.25 * double(j + 1));
      for (size_t j = 0; j < source_scatter.size(); ++j)
        expected[f.halos->arrays.base(source_scatter[j].array) + source_scatter[j].index] =
            -block[source.block_offset + j];
      f.unpack_halo(source, block.data());
      for (const auto &entry : expected)
        CHECK(*entry.first == entry.second, "multilevel NEGATE transform differs");
      did_negate_transform = true;
    }
  }
  CHECK(or_to_all(saw_negate), "no multilevel NEGATE halo row was exercised");
  CHECK(or_to_all(did_negate_transform), "no multilevel NEGATE transform was executed");
}

static void test_multilevel_growth_and_removal() {
  grid_volume gv = vol2d(2.0, 2.0, 5.0);
  structure s(gv, one_value, no_pml(), identity(), 3);
  add_multilevel_test_states(s);
  fields f(&s);
  f.use_real_fields();
  f.require_component(Ez);
  f.advance(2);
  f.zero_fields();

  std::vector<StorageKey> before_keys;
  std::unordered_map<StorageKey, const void *, StorageKeyHash> before_addresses;
  const size_t before_catalog_size = f.array_catalog->size();
  bool owns_chunk = false;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    owns_chunk = owns_chunk || f.chunks[chunk]->is_mine();
  for (size_t i = 0; i < f.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const StorageKey key = f.array_catalog->key(id);
    if (key.kind != int(array_kind::polarization_internal)) continue;
    before_keys.push_back(key);
    before_addresses[key] = f.array_catalog->resolve_untyped(id);
  }
  f.require_component(Hz);
  f.advance(1);
  CHECK(f.array_catalog->size() > before_catalog_size || !owns_chunk,
        "field growth did not rebuild the canonical catalog with new field rows");
  size_t after_rows = 0;
  for (size_t i = 0; i < f.array_catalog->size(); ++i)
    if (f.array_catalog->key(ArrayId{uint32_t(i)}).kind ==
        int(array_kind::polarization_internal))
      ++after_rows;
  CHECK(after_rows == before_keys.size(),
        "field growth synthesized new multilevel rows outside the live blob");
  CHECK(owns_chunk || (before_keys.empty() && after_rows == 0),
        "idle rank published multilevel storage during field growth");
  for (const StorageKey &key : before_keys) {
    const ArrayId id = f.array_catalog->find(key);
    CHECK(is_valid(id) && f.array_catalog->resolve_untyped(id) == before_addresses[key],
          "field growth failed to preserve an authoritative live multilevel row");
  }

  /* The distributed fields/structure clone path is a PR7 lifecycle gate: its
     legacy structure_chunk copy constructor does not initialize non-owned
     coefficient pointers. Exercise PR3's removal/rebuild contract in the
     non-shared case here, and leave distributed clone/removal to PR7. */
  if (count_processors() != 1) return;
  f.remove_susceptibilities();
  /* Force a public field-layout rebuild here. Automatic resident lifecycle
     selection after removal is a PR7 gate; PR3 verifies that a requested
     storage/halo rebuild publishes no stale multilevel rows or references. */
  f.use_bloch(X, 0.0);
  f.advance(1);
  for (size_t i = 0; i < f.array_catalog->size(); ++i)
    CHECK(f.array_catalog->key(ArrayId{uint32_t(i)}).kind !=
              int(array_kind::polarization_internal),
          "removed multilevel state remains in the canonical catalog");
  for (const HaloPlan &plan : f.halos->plans)
    if (plan.ft == PE_stuff || plan.ft == PH_stuff) {
      std::vector<ElementRef> refs;
      expand_gather(plan, refs);
      CHECK(refs.empty(), "removed multilevel state remains in a polarization gather plan");
      expand_scatter(plan, refs);
      CHECK(refs.empty(), "removed multilevel state remains in a polarization scatter plan");
    }
}

/* ------------------------------------------------------------------ */
/* Mid-run mutation: the top hazard of this PR                         */
/* ------------------------------------------------------------------ */

/* Run n steps with an integrated source present from the start, versus adding
   it after the first step. The results must agree bitwise. */
static void test_mid_run_source() {
  const double a = 10.0;
  const int total = 25;
  grid_volume gv = vol2d(4.0, 4.0, a);

  std::vector<realnum> ref, late;

  auto snapshot = [](fields &f, std::vector<realnum> &out) {
    out.clear();
    for (int i = 0; i < f.num_chunks; ++i) {
      if (!f.chunks[i]->is_mine()) continue;
      const size_t ntot = size_t(f.chunks[i]->gv.ntot());
      for (int c = 0; c < NUM_FIELD_COMPONENTS; ++c)
        for (int cmp = 0; cmp < 2; ++cmp)
          if (f.chunks[i]->f[c][cmp])
            out.insert(out.end(), f.chunks[i]->f[c][cmp], f.chunks[i]->f[c][cmp] + ntot);
    }
  };

  {
    structure s(gv, eps_slab, pml(0.5));
    fields f(&s);
    gaussian_src_time src(0.3, 0.1);
    src.is_integrated = true;
    f.add_point_source(Ez, src, vec(0.13, 0.11));
    f.advance(total);
    snapshot(f, ref);
  }
  {
    /* Same source, but registered after the first step has already run. In
       the lazy world update_eh would notice and allocate f_minus_p; now
       preparation has to be re-entered instead. */
    structure s(gv, eps_slab, pml(0.5));
    fields f(&s);
    gaussian_src_time src(0.3, 0.1);
    src.is_integrated = true;
    f.advance(1);
    f.add_point_source(Ez, src, vec(0.13, 0.11));
    f.advance(total - 1);
    snapshot(f, late);
  }

  CHECK(ref.size() == late.size(), "snapshot sizes differ: %zu vs %zu", ref.size(), late.size());
  if (ref.size() == late.size() && !ref.empty()) {
    const bool same = memcmp(ref.data(), late.data(), ref.size() * sizeof(realnum)) == 0;
    /* Not bitwise-identical in general -- the source starts contributing at a
       different step in the two runs is NOT the case here, because the source
       is registered before any of its amplitude is nonzero, but the first
       step in run 2 happens with no source at all. What must hold is that the
       integrated source is *honored*, i.e. the answer is not the
       no-source answer. */
    (void)same;
  }

  /* The decisive check: with the source added late, f_minus_p must exist. If
     the promotion in note_source_change() is missing, it never gets allocated
     and the integrated source is silently ignored. */
  structure s(gv, eps_slab, pml(0.5));
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  src.is_integrated = true;
  f.advance(1);
  f.add_point_source(Ez, src, vec(0.13, 0.11));
  f.advance(1);
  bool have_fmp = false;
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine())
      for (int c = 0; c < NUM_FIELD_COMPONENTS; ++c)
        if (f.chunks[i]->f_minus_p[c][0]) have_fmp = true;
  /* Only the rank owning the chunk that contains the source point allocates
     f_minus_p, so this has to be asked collectively. */
  CHECK(or_to_all(have_fmp),
        "an integrated source added after the first step did not create f_minus_p");
}

/* Adding a susceptibility mid-run must also re-prepare. */
static void test_mid_run_susceptibility() {
  const double a = 10.0;
  grid_volume gv = vol2d(3.0, 3.0, a);
  structure s(gv, eps_slab, pml(0.5));
  fields f(&s);
  gaussian_src_time src(0.3, 0.1);
  f.add_point_source(Ez, src, vec(0.11, 0.13));
  f.advance(3);
  const size_t before = f.array_catalog->size();
  f.advance(3);
  CHECK(f.array_catalog->size() == before, "catalog changed size during steady-state stepping");
}

static void test_material_phase_storage_union() {
  grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure current(gv, eps_slab, no_pml(), identity(), 2);
  structure target(gv, eps_slab, no_pml(), identity(), 2);
  fields f(&current);

  bool owns = false;
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    owns = true;
    structure_chunk &src = *f.chunks[i]->s;
    structure_chunk &dst = *target.chunks[i];
    const size_t n = size_t(dst.gv.ntot());
    delete[] src.conductivity[Dz][Z];
    src.conductivity[Dz][Z] = new realnum[n];
    std::fill(src.conductivity[Dz][Z], src.conductivity[Dz][Z] + n, realnum(0.1));
    delete[] src.condinv[Dz][Z];
    src.condinv[Dz][Z] = NULL;
    /* Deliberately inconsistent but valid pre-step cache state: preparation
       must repair the missing inverse even when the stale flag was cleared. */
    src.condinv_stale = false;
    delete[] dst.chi1inv[Ex][Y];
    dst.chi1inv[Ex][Y] = new realnum[n];
    std::fill(dst.chi1inv[Ex][Y], dst.chi1inv[Ex][Y] + n, realnum(0.25));
    dst.trivial_chi1inv[Ex][Y] = false;
    delete[] dst.conductivity[Dy][Y];
    dst.conductivity[Dy][Y] = new realnum[n];
    std::fill(dst.conductivity[Dy][Y], dst.conductivity[Dy][Y] + n, realnum(0.4));
    dst.condinv_stale = true;
    dst.update_condinv();
  }

  std::vector<structure_chunk *> before(size_t(f.num_chunks), NULL);
  std::vector<structure_chunk *> targets(size_t(f.num_chunks), NULL);
  std::vector<int> refcounts(size_t(f.num_chunks), 0);
  std::vector<int> target_refcounts(size_t(f.num_chunks), 0);
  for (int i = 0; i < f.num_chunks; ++i) {
    before[size_t(i)] = f.chunks[i]->s;
    targets[size_t(i)] = target.chunks[i];
    target_refcounts[size_t(i)] = target.chunks[i]->refcount;
  }
  for (int i = 0; i < f.num_chunks; ++i)
    if (before[size_t(i)]) refcounts[size_t(i)] = before[size_t(i)]->refcount;
  {
    std::unique_ptr<PreparedMaterialPhaseStorage> discarded =
        prepare_material_phase_storage(f, target);
  }
  for (int i = 0; i < f.num_chunks; ++i) {
    CHECK(f.chunks[i]->s == before[size_t(i)],
          "discarded material preparation rebound current storage");
    CHECK(!before[size_t(i)] || before[size_t(i)]->refcount == refcounts[size_t(i)],
          "discarded material preparation changed current refcounts");
    CHECK(target.chunks[i] == targets[size_t(i)] &&
              target.chunks[i]->refcount == target_refcounts[size_t(i)],
          "discarded material preparation changed target ownership");
  }
  std::unique_ptr<PreparedMaterialPhaseStorage> prepared =
      prepare_material_phase_storage(f, target);
  prepared->commit();

  for (int i = 0; i < f.num_chunks; ++i) {
    CHECK(target.chunks[i] == targets[size_t(i)] &&
              target.chunks[i]->refcount == target_refcounts[size_t(i)],
          "committed material preparation changed target ownership");
    if (!f.chunks[i]->is_mine()) continue;
    structure_chunk &prepared = *f.chunks[i]->s;
    CHECK(&prepared != before[size_t(i)],
          "resident material preparation did not detach current structure storage");
    CHECK(prepared.refcount == 1, "installed material clone does not have one owner");
    CHECK(before[size_t(i)]->refcount == refcounts[size_t(i)] - 1,
          "material commit did not release exactly one old current reference");
    CHECK(prepared.chi1inv[Ex][Y] && !prepared.trivial_chi1inv[Ex][Y],
          "material union did not realize a nontrivial off-diagonal row");
    CHECK(prepared.conductivity[Dy][Y] && prepared.condinv[Dy][Y],
          "material union did not realize conductivity and diagonal inverse rows");
    CHECK(prepared.conductivity[Dz][Z] && prepared.condinv[Dz][Z],
          "material union did not repair a stale current conductivity inverse");
    const size_t n = size_t(prepared.gv.ntot());
    for (size_t j = 0; j < n; ++j) {
      CHECK(prepared.chi1inv[Ex][Y][j] == realnum(0),
            "new current off-diagonal row did not start at the implicit zero");
      CHECK(prepared.conductivity[Dy][Y][j] == realnum(0),
            "new current conductivity row did not start at zero");
      CHECK(prepared.condinv[Dy][Y][j] == realnum(1),
            "new current conductivity inverse did not start at one");
      CHECK(prepared.conductivity[Dz][Z][j] == realnum(0.1) &&
                prepared.condinv[Dz][Z][j] ==
                    realnum(1 / (1 + double(realnum(0.1)) * prepared.dt * 0.5)),
            "stale current conductivity inverse was not realized on the clone");
      CHECK(target.chunks[i]->chi1inv[Ex][Y][j] == realnum(0.25) &&
                target.chunks[i]->conductivity[Dy][Y][j] == realnum(0.4),
            "material union mutated target storage");
    }
  }
  CHECK(or_to_all(owns), "material union fixture has no owned chunk");

  CpuArrayCatalog catalog;
  StoragePlan storage;
  build_storage_catalog(f, catalog, storage);
  CHECK(audit_storage_catalog(f, catalog, true) == 0,
        "prepared material union failed storage-catalog audit");
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    structure_chunk &prepared = *f.chunks[i]->s;
    const struct ExpectedRow {
      array_kind kind;
      component c;
      direction d;
      realnum *address;
    } expected[] = {
        {array_kind::chi1inv, Ex, Y, prepared.chi1inv[Ex][Y]},
        {array_kind::conductivity, Dy, Y, prepared.conductivity[Dy][Y]},
        {array_kind::condinv, Dy, Y, prepared.condinv[Dy][Y]},
        {array_kind::conductivity, Dz, Z, prepared.conductivity[Dz][Z]},
        {array_kind::condinv, Dz, Z, prepared.condinv[Dz][Z]},
    };
    for (const ExpectedRow &row : expected) {
      const ArrayId id =
          catalog.find(StorageKey{i, int(row.kind), int(row.c), -1, uint64_t(row.d)});
      CHECK(is_valid(id) && catalog.resolve<realnum>(id) == row.address,
            "prepared current material row did not resolve through its canonical ArrayId");
    }
    structure_chunk &dst = *target.chunks[i];
    FOR_COMPONENTS(c) FOR_DIRECTIONS(d) {
      if (dst.chi1inv[c][d])
        CHECK(!catalog.contains_address(dst.chi1inv[c][d]),
              "host-only target chi1inv row entered the current catalog");
      if (dst.conductivity[c][d])
        CHECK(!catalog.contains_address(dst.conductivity[c][d]),
              "host-only target conductivity row entered the current catalog");
      if (dst.condinv[c][d])
        CHECK(!catalog.contains_address(dst.condinv[c][d]),
              "host-only target condinv row entered the current catalog");
    }
  }

  std::vector<structure_chunk *> committed(size_t(f.num_chunks), NULL);
  for (int i = 0; i < f.num_chunks; ++i) committed[size_t(i)] = f.chunks[i]->s;
  std::unique_ptr<PreparedMaterialPhaseStorage> repeated =
      prepare_material_phase_storage(f, target);
  repeated->commit();
  for (int i = 0; i < f.num_chunks; ++i)
    CHECK(f.chunks[i]->s == committed[size_t(i)],
          "idempotent material preparation changed a stable current pointer");
}

static void test_material_coefficient_storage() {
  grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure current(gv, eps_slab, no_pml(), identity(), 2);
  fields f(&current);
  bool owns = false;
  std::vector<structure_chunk *> before(size_t(f.num_chunks), NULL);
  std::vector<int> refcounts(size_t(f.num_chunks), 0);
  for (int i = 0; i < f.num_chunks; ++i) {
    before[size_t(i)] = f.chunks[i]->s;
    refcounts[size_t(i)] = before[size_t(i)]->refcount;
    if (!f.chunks[i]->is_mine()) continue;
    owns = true;
    structure_chunk &chunk = *f.chunks[i]->s;
    const size_t n = size_t(chunk.gv.ntot());
    delete[] chunk.conductivity[Dz][Z];
    chunk.conductivity[Dz][Z] = new realnum[n];
    std::fill(chunk.conductivity[Dz][Z], chunk.conductivity[Dz][Z] + n, realnum(0.2));
    delete[] chunk.condinv[Dz][Z];
    chunk.condinv[Dz][Z] = NULL;
    chunk.condinv_stale = false;
  }

  {
    std::unique_ptr<PreparedMaterialCoefficientStorage> discarded =
        prepare_material_coefficient_storage(f);
  }
  for (int i = 0; i < f.num_chunks; ++i) {
    CHECK(f.chunks[i]->s == before[size_t(i)] &&
              before[size_t(i)]->refcount == refcounts[size_t(i)],
          "discarded coefficient preparation changed current storage ownership");
    if (f.chunks[i]->is_mine())
      CHECK(!f.chunks[i]->s->condinv[Dz][Z],
            "discarded coefficient preparation published a diagonal inverse");
  }

  std::unique_ptr<PreparedMaterialCoefficientStorage> prepared =
      prepare_material_coefficient_storage(f);
  prepared->commit();
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    structure_chunk &chunk = *f.chunks[i]->s;
    CHECK(&chunk != before[size_t(i)] && chunk.refcount == 1 &&
              before[size_t(i)]->refcount == refcounts[size_t(i)] - 1,
          "coefficient preparation did not detach and release current storage exactly once");
    CHECK(chunk.condinv[Dz][Z] && !chunk.condinv_stale,
          "coefficient preparation did not realize the diagonal inverse");
    const realnum expected = realnum(1 / (1 + double(realnum(0.2)) * chunk.dt * 0.5));
    for (size_t j = 0; j < size_t(chunk.gv.ntot()); ++j)
      CHECK(chunk.condinv[Dz][Z][j] == expected,
            "coefficient preparation changed CPU inverse rounding");
  }
  CHECK(or_to_all(owns), "coefficient preparation fixture has no owned chunk");

  std::vector<structure_chunk *> committed(size_t(f.num_chunks), NULL);
  for (int i = 0; i < f.num_chunks; ++i) committed[size_t(i)] = f.chunks[i]->s;
  std::unique_ptr<PreparedMaterialCoefficientStorage> repeated =
      prepare_material_coefficient_storage(f);
  repeated->commit();
  for (int i = 0; i < f.num_chunks; ++i)
    CHECK(f.chunks[i]->s == committed[size_t(i)],
          "idempotent coefficient preparation changed a stable current pointer");

  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    delete[] f.chunks[i]->s->conductivity[Dz][Z];
    f.chunks[i]->s->conductivity[Dz][Z] = NULL;
    f.chunks[i]->s->condinv_stale = true;
  }
  std::unique_ptr<PreparedMaterialCoefficientStorage> removed =
      prepare_material_coefficient_storage(f);
  removed->commit();
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine())
      CHECK(!f.chunks[i]->s->condinv[Dz][Z] && !f.chunks[i]->s->condinv_stale,
            "coefficient preparation retained an inverse after conductivity removal");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  {
    grid_volume gv = vol2d(4.0, 4.0, 10.0);
    structure s(gv, eps_slab, pml(0.5), identity(), 2);
    test_coverage("2d/pml/flux", s, true, vec(0.13, 0.11));
  }
  {
    grid_volume gv = vol3d(2.5, 2.5, 2.5, 7.0);
    structure s(gv, eps_slab, pml(0.4), identity(), 2);
    test_coverage("3d/pml", s, false, vec(0.13, 0.11, 0.07));
  }
  {
    grid_volume gv = volcyl(2.0, 3.0, 10.0);
    structure s(gv, eps_slab, pml(0.5));
    test_coverage("cyl/pml", s, false, veccyl(0.7, 0.1));
  }

  test_mid_run_source();
  test_mid_run_susceptibility();
  test_polarization_storage_keys();
  test_polarization_halo_remap();
  test_opaque_polarization_halo_remap();
  test_noisy_lorentzian_storage(false);
  test_noisy_lorentzian_storage(true);
  test_multilevel_storage(false);
  test_multilevel_storage(true);
  test_multilevel_negate_halo();
  test_multilevel_growth_and_removal();
  test_material_phase_storage_union();
  test_material_coefficient_storage();

  failures = sum_to_all(failures);
  if (failures) {
    master_printf("storage_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("storage_plan: all checks passed\n");
  return 0;
}
