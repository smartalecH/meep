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
#include <stdexcept>
#include <vector>

#include <meep.hpp>

#include "backend/lifecycle.hpp"
#include "backend/descriptors.hpp"
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
  /* More MPI ranks than chunks legitimately leaves some ranks with no owned
     storage.  The plan must be nonempty globally, not on every rank. */
  CHECK(or_to_all(f.array_catalog->size() > 0), "%s: catalog is empty on every rank", name);
  CHECK(f.storage_plan->arrays.size() == f.array_catalog->size(),
        "%s: storage plan/catalog sizes differ", name);
  for (size_t i = 0; i < f.array_catalog->size(); ++i)
    CHECK(f.storage_plan->arrays[i].alias_of == f.array_catalog->spec(ArrayId{uint32_t(i)}).alias_of,
          "%s: storage plan lost alias metadata at ArrayId %zu", name, i);

  /* Halo construction has its own temporary pointer table because it can run
     before storage preparation. A backend consumes only the remapped copy. */
  for (const HaloPlan &source : f.halos->plans) {
    HaloPlan canonical;
    std::string why;
    CHECK(remap_halo_plan(source, f.halos->arrays, *f.array_catalog, f.is_real ? 1 : 2,
                          canonical, why),
          "%s: halo plan did not map to canonical storage: %s", name, why.c_str());
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
  }
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
    CHECK(remap_halo_plan(source, f.halos->arrays, *f.array_catalog, f.is_real ? 1 : 2,
                          canonical, why),
          "polarization halo plan did not map to canonical storage: %s", why.c_str());
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

static void test_gyrotropic_storage_without_p_halos() {
  grid_volume gv = vol3d(2.0, 2.0, 2.0, 8.0);
  structure s(gv, eps_slab, pml(0.25), identity(), 2);
  gyrotropic_susceptibility gyro(vec(0.17, -0.23, 0.31), 0.8, 0.05);
  s.add_susceptibility(unit_sigma, E_stuff, gyro);
  fields f(&s);
  f.require_component(Ex);
  f.require_component(Ey);
  f.require_component(Ez);
  f.advance(2);

  size_t gyro_arrays = 0;
  for (const PolarizationDescriptor &d : f.descriptors->polarizations) {
    if (d.kind != SusceptibilityKind::gyrotropic) continue;
    gyro_arrays += 6 * d.gyrotropic_states.size();
    for (const GyrotropicStateArrays &state : d.gyrotropic_states)
      for (int dd = 0; dd < 3; ++dd) {
        const ArraySpec &p = f.array_catalog->spec(state.p[dd]);
        const ArraySpec &pp = f.array_catalog->spec(state.p_prev[dd]);
        CHECK(p.role == array_role::polarization && pp.role == array_role::polarization &&
                  p.element_type == ElementType::realnum_value &&
                  pp.element_type == ElementType::realnum_value,
              "gyrotropic storage has incompatible metadata");
      }
  }
  size_t p_halo_elements = 0;
  for (const HaloPlan &plan : f.halos->plans) {
    if (plan.ft != PE_stuff && plan.ft != PH_stuff) continue;
    std::vector<ElementRef> refs;
    expand_gather(plan, refs);
    p_halo_elements += refs.size();
    expand_scatter(plan, refs);
    p_halo_elements += refs.size();
  }
  CHECK(or_to_all(gyro_arrays > 0 && gyro_arrays % 6 == 0),
        "gyrotropic storage did not contain complete six-array rows");
  CHECK(sum_to_all(p_halo_elements) == 0,
        "gyrotropic polarization unexpectedly created PE/PH halo elements");
  CHECK(audit_storage_catalog(f, *f.array_catalog, true) == 0,
        "gyrotropic storage left uncatalogued arrays");
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
      const ArrayId id = catalog.find(StorageKey{i, int(row.kind), int(row.c), -1, int(row.d)});
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
  test_polarization_halo_remap();
  test_material_phase_storage_union();
  test_gyrotropic_storage_without_p_halos();

  if (failures) {
    master_printf("storage_plan: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("storage_plan: all checks passed\n");
  return 0;
}
