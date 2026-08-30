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

/* PR 6 acceptance tests: source, DFT and susceptibility descriptors. */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include <meep.hpp>

#include "backend/descriptors.hpp"
#include "backend/step_plan.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

using namespace meep;
using std::complex;

static int failures = 0;

#define CHECK(cond, ...)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      printf("[rank %d] FAIL (%s:%d): ", my_rank(), __FILE__, __LINE__);                           \
      printf(__VA_ARGS__);                                                                         \
      printf("\n");                                                                                \
      fflush(stdout);                                                                              \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

static double one(const vec &) { return 1.0; }
static double eps_slab(const vec &p) { return (fabs(p.y()) < 0.4) ? 12.0 : 1.0; }
static bool close(complex<double> a, complex<double> b) {
  return abs(a - b) <= 1e-13 * std::max(1.0, std::max(abs(a), abs(b)));
}

/* A susceptibility that publishes no layout: it must classify as host_custom
   and still run. This stands in for an unknown third-party subclass. */
class opaque_susceptibility : public lorentzian_susceptibility {
public:
  opaque_susceptibility(realnum w, realnum g) : lorentzian_susceptibility(w, g) {}
  virtual susceptibility *clone() const { return new opaque_susceptibility(*this); }
  virtual bool internal_layout(std::vector<InternalArrayLayout> &, const grid_volume &,
                               void *) const {
    return false;
  }
};

class inherited_lorentzian_susceptibility : public lorentzian_susceptibility {
public:
  inherited_lorentzian_susceptibility(realnum w, realnum g)
      : lorentzian_susceptibility(w, g) {}
  virtual susceptibility *clone() const {
    return new inherited_lorentzian_susceptibility(*this);
  }
};

class inherited_gyrotropic_susceptibility : public gyrotropic_susceptibility {
public:
  inherited_gyrotropic_susceptibility()
      : gyrotropic_susceptibility(vec(0.17, -0.23, 0.31), 0.8, 0.05, 0.07,
                                  GYROTROPIC_LORENTZIAN) {}
  virtual susceptibility *clone() const {
    return new inherited_gyrotropic_susceptibility(*this);
  }
};

class derived_gaussian_source : public gaussian_src_time {
public:
  derived_gaussian_source() : gaussian_src_time(0.31, 0.08) {}
  virtual src_time *clone() const { return new derived_gaussian_source(*this); }
  virtual complex<double> dipole(double time) const {
    return gaussian_src_time::dipole(time) + complex<double>(0.125, -0.25);
  }
};

class derived_continuous_source : public continuous_src_time {
public:
  derived_continuous_source() : continuous_src_time(0.27) {}
  virtual src_time *clone() const { return new derived_continuous_source(*this); }
  virtual complex<double> current(double time, double dt) const {
    (void)time;
    (void)dt;
    return complex<double>(0.375, 0.5);
  }
};

class ordered_custom_source : public continuous_src_time {
public:
  ordered_custom_source(double frequency) : continuous_src_time(frequency) {}
  virtual src_time *clone() const { return new ordered_custom_source(*this); }
  virtual complex<double> current(double time, double dt) const {
    return continuous_src_time::current(time, dt) + complex<double>(0.03125, -0.0625);
  }
};

static void test_source_ordinals() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, one, no_pml(), identity(), 2);
  fields f(&s);
  continuous_src_time m0(0.21), mi(0.22), m2(0.23), e0(0.31), ei(0.32);
  ordered_custom_source e2(0.33);
  m0.is_integrated = false;
  mi.is_integrated = true;
  m2.is_integrated = false;
  e0.is_integrated = false;
  ei.is_integrated = true;
  e2.is_integrated = false;
  const vec p(0.17, 0.19);
  f.add_point_source(Hz, m0, p);
  f.add_point_source(Hz, mi, p);
  f.add_point_source(Hz, m2, p);
  f.add_point_source(Ez, e0, p);
  f.add_point_source(Ez, ei, p);
  f.add_point_source(Ez, e2, p);
  f.advance(1);

  SourcePlan plan;
  build_source_descriptors(f, plan);
  bool saw_interleaved = false, saw_custom = false;
  struct ExpectedSource {
    field_type ft;
    int chunk;
    uint32_t ordinal;
    bool integrated;
  };
  std::vector<ExpectedSource> expected;
  FOR_FIELD_TYPES(ft) for (int integrated = 0; integrated < 2; ++integrated)
    for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
      if (!f.chunks[chunk]->is_mine()) continue;
      const std::vector<src_vol> &live = f.chunks[chunk]->get_sources(ft);
      for (size_t ordinal = 0; ordinal < live.size(); ++ordinal)
        if (int(live[ordinal].t()->is_integrated) == integrated)
          expected.push_back(ExpectedSource{ft, chunk, uint32_t(ordinal), bool(integrated)});
    }
  CHECK(plan.sources.size() == expected.size(),
        "source descriptor grouping has %zu rows, expected %zu", plan.sources.size(),
        expected.size());
  for (size_t i = 0; i < plan.sources.size() && i < expected.size(); ++i)
    CHECK(plan.sources[i].ft == expected[i].ft && plan.sources[i].chunk == expected[i].chunk &&
              plan.sources[i].source_ordinal == expected[i].ordinal &&
              plan.sources[i].integrated == expected[i].integrated,
          "source descriptor %zu changed grouped order or original ordinal", i);
  FOR_FIELD_TYPES(ft) for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
      if (!f.chunks[chunk]->is_mine()) continue;
      const std::vector<src_vol> &live = f.chunks[chunk]->get_sources(ft);
      if (live.size() >= 3 && !live[0].t()->is_integrated && live[1].t()->is_integrated &&
          !live[2].t()->is_integrated)
        saw_interleaved = true;
      std::vector<uint32_t> ordinary, integrated;
      for (const SourceDescriptor &d : plan.sources) {
        if (d.chunk != chunk || d.ft != ft) continue;
        (d.integrated ? integrated : ordinary).push_back(d.source_ordinal);
        CHECK(d.source_ordinal < live.size(), "source ordinal %u is out of range",
              d.source_ordinal);
        if (d.source_ordinal < live.size()) {
          const src_vol &sv = live[d.source_ordinal];
          CHECK(d.integrated == sv.t()->is_integrated,
                "source ordinal does not reconstruct integration kind");
          CHECK(d.indices.size() == sv.num_points(),
                "source ordinal reconstructs the wrong spatial row");
        }
      }
      CHECK(std::is_sorted(ordinary.begin(), ordinary.end()),
            "ordinary source subgroup changed original ordinal order");
      CHECK(std::is_sorted(integrated.begin(), integrated.end()),
            "integrated source subgroup changed original ordinal order");
    }
  for (const SourceTimeDescriptor &d : plan.source_times)
    saw_custom = saw_custom || d.kind == SourceTimeKind::host_custom;
  CHECK(or_to_all(saw_interleaved), "fixture produced no interleaved local source vector");
  CHECK(saw_custom, "ordered-source fixture did not retain its custom source time");

  if (!plan.sources.empty()) {
    const uint64_t signature = source_plan_signature(plan);
    SourcePlan changed = plan;
    ++changed.sources[0].source_ordinal;
    CHECK(source_plan_signature(changed) != signature,
          "source signature ignored the original source ordinal");
    changed = plan;
    if (changed.sources.size() > 1) {
      std::swap(changed.sources[0].source_ordinal, changed.sources[1].source_ordinal);
      CHECK(source_plan_signature(changed) != signature,
            "source signature ignored exchanged source ordinals");
    }
  }
}

static void test_sources() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  fields f(&s);
  gaussian_src_time g(0.3, 0.1);
  continuous_src_time c(complex<double>(0.25, 0.02), 0.4, 0.2, 2.5, 3.0);
  f.add_point_source(Ez, g, vec(0.13, 0.11));
  f.add_point_source(Ez, c, vec(-0.3, 0.2));
  f.advance(2);

  SourcePlan plan;
  build_source_descriptors(f, plan);

  CHECK(plan.source_times.size() == 2, "expected 2 source-time descriptors, got %zu",
        plan.source_times.size());
  CHECK(plan.scalars.size() == plan.source_times.size(), "scalar block is the wrong size");

  bool saw_gaussian = false, saw_continuous = false;
  for (const SourceTimeDescriptor &d : plan.source_times) {
    if (d.kind == SourceTimeKind::gaussian) saw_gaussian = true;
    if (d.kind == SourceTimeKind::continuous) saw_continuous = true;
    CHECK(d.scalar_slot < plan.scalars.size(), "scalar_slot %u out of range", d.scalar_slot);
    if (d.kind == SourceTimeKind::gaussian)
      CHECK(d.parameters.size() == 4, "gaussian descriptor has %zu parameters, expected 4",
            d.parameters.size());
    if (d.kind == SourceTimeKind::continuous)
      CHECK(d.parameters.size() == 6, "continuous descriptor has %zu parameters, expected 6",
            d.parameters.size());
  }
  CHECK(saw_gaussian && saw_continuous, "built-in source kinds were not both classified");

  const uint64_t source_signature = source_plan_signature(plan);
  SourcePlan changed = plan;
  changed.source_times[0].parameters[0] += 1e-6;
  CHECK(source_plan_signature(changed) != source_signature,
        "source signature ignored a built-in parameter change");
  changed = plan;
  if (!changed.sources.empty() && !changed.sources[0].complex_amplitudes.empty()) {
    changed.sources[0].complex_amplitudes[0] += complex<double>(0.0, 1e-6);
    CHECK(source_plan_signature(changed) != source_signature,
          "source signature ignored a spatial amplitude change");
  }

  /* Closed descriptors reproduce the live built-ins, including cutoff and
     finite-difference current semantics. */
  const src_time *live = f.sources;
  for (size_t i = 0; i < plan.source_times.size(); ++i, live = live->next) {
    const SourceTimeDescriptor &d = plan.source_times[i];
    const double sample_times[] = {0.0, 0.5 * f.dt, f.dt, live->last_time()};
    for (size_t ti = 0; ti < sizeof(sample_times) / sizeof(sample_times[0]); ++ti) {
      const double time = sample_times[ti];
      const SourceStepScalar got = evaluate_source_time_descriptor(d, time, f.dt);
      CHECK(close(got.dipole, live->dipole(time)), "%s dipole differs at %.17g",
            source_time_kind_name(d.kind), time);
      CHECK(close(got.current, live->current(time, f.dt)), "%s current differs at %.17g",
            source_time_kind_name(d.kind), time);
    }
  }

  /* The production scalar block follows the same linked-list/slot order. */
  populate_source_scalars(f, plan);
  live = f.sources;
  for (size_t i = 0; i < plan.source_times.size(); ++i, live = live->next) {
    const SourceStepScalar &got = plan.scalars[plan.source_times[i].scalar_slot];
    CHECK(got.current == live->current(), "source scalar current slot %zu is stale", i);
    CHECK(got.dipole == live->dipole(), "source scalar dipole slot %zu is stale", i);
  }

  /* Every spatial table must match the src_vol it came from, point for point,
     and must resolve to a registered destination array. */
  size_t bad = 0, checked = 0;
  size_t src_vols = 0;
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine()) FOR_FIELD_TYPES(ft) { src_vols += f.chunks[i]->get_sources(ft).size(); }
  CHECK(plan.sources.size() == src_vols, "%zu source descriptors for %zu src_vols",
        plan.sources.size(), src_vols);
  for (const SourceDescriptor &d : plan.sources) {
    CHECK(is_valid(d.destination), "source descriptor has no destination array");
    if (!f.chunks[d.chunk]->is_real)
      CHECK(is_valid(d.destination_imag), "complex source descriptor has no imaginary target");
    if (d.integrated)
      CHECK(is_valid(d.integrated_destination),
            "integrated source descriptor has no f_minus_p destination");
    CHECK(d.source_time_id < plan.source_times.size(), "source has no src_time descriptor");
    const std::vector<src_vol> &sv = f.chunks[d.chunk]->get_sources(d.ft);
    bool matched = false;
    for (const src_vol &v : sv) {
      if (v.num_points() != d.indices.size()) continue;
      bool same = true;
      for (size_t j = 0; j < d.indices.size() && same; ++j)
        same = v.index_at(j) == d.indices[j] && v.amplitude_at(j) == d.complex_amplitudes[j];
      if (same) matched = true;
    }
    if (!matched) ++bad;
    ++checked;
  }
  CHECK(bad == 0, "%zu of %zu source spatial tables do not match their src_vol", bad, checked);
  master_printf("sources: %zu descriptors, %zu source times\n", plan.sources.size(),
                plan.source_times.size());
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine()) FOR_FIELD_TYPES(ft) {
        if (!f.chunks[i]->get_sources(ft).empty())
          master_printf("  chunk %d ft %d: %zu src_vol\n", i, int(ft),
                        f.chunks[i]->get_sources(ft).size());
      }
}

static void test_derived_source_times_remain_host_custom() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, one, no_pml());
  fields f(&s);
  derived_gaussian_source gaussian;
  derived_continuous_source continuous;
  gaussian.is_integrated = false;
  continuous.is_integrated = false;
  f.add_point_source(Ez, gaussian, vec(0.11, 0.13));
  f.add_point_source(Hz, continuous, vec(-0.17, 0.19));
  f.advance(1);

  SourcePlan plan;
  build_source_descriptors(f, plan);
  CHECK(plan.source_times.size() == 2, "derived sources produced %zu source times",
        plan.source_times.size());
  for (const SourceTimeDescriptor &d : plan.source_times) {
    CHECK(d.kind == SourceTimeKind::host_custom,
          "derived built-in source was classified as %s", source_time_kind_name(d.kind));
    CHECK(d.parameters.empty(), "host-custom source unexpectedly exported closed parameters");
    CHECK(d.host_callback_id != 0xffffffffu, "host-custom source has no callback id");
  }
}

static void test_dfts() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  fields f(&s);
  gaussian_src_time g(0.3, 0.1);
  f.add_point_source(Ez, g, vec(0.13, 0.11));
  volume fv(vec(0.8, -1.0), vec(0.8, 1.0));
  f.add_dft_flux(Z, fv, 0.25, 0.35, 3);
  /* A persistent (adjoint-style) monitor: its extent is padded by one pixel per
     direction, and the unpadded extent has to survive into the descriptor. */
  volume dv(vec(-0.6, -0.6), vec(0.6, 0.6));
  f.add_dft(Ez, dv, 0.25, 0.35, 3, /*include_dV*/ true, /*stored_weight*/ 1.0,
            /*chunk_next*/ 0, /*sqrt_dV*/ false, /*extra_weight*/ 1.0,
            /*use_centered_grid*/ true, /*vc*/ 0, /*decimation_factor*/ 0, /*persist*/ true);
  f.advance(4);

  std::vector<DftDescriptor> dfts;
  build_dft_descriptors(f, dfts);
  CHECK(!dfts.empty() || f.num_chunks > 1, "no DFT descriptors were built");

  size_t persistent = 0, padded = 0;
  size_t live_count = 0;
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine())
      for (dft_chunk *cur = f.chunks[i]->dft_chunks; cur; cur = cur->next_in_chunk) ++live_count;
  CHECK(dfts.size() == live_count, "%zu DFT descriptors for %zu live chunks", dfts.size(),
        live_count);

  if (!dfts.empty()) {
    const uint64_t signature = dft_plan_signature(dfts);
    std::vector<DftDescriptor> changed = dfts;
    ++changed[0].decimation_factor;
    CHECK(dft_plan_signature(changed) != signature,
          "DFT signature ignored the decimation factor");
    changed = dfts;
    ++changed[0].due_scalar_slot;
    CHECK(dft_plan_signature(changed) != signature, "DFT signature ignored the due slot");
    changed = dfts;
    changed[0].omega[0] += 1e-6;
    CHECK(dft_plan_signature(changed) != signature, "DFT signature ignored the frequency table");
    changed = dfts;
    ++changed[0].source_field.id.value;
    CHECK(dft_plan_signature(changed) != signature, "DFT signature ignored a source reference");
  }

  size_t descriptor_index = 0;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    int di = 0;
    for (dft_chunk *cur = f.chunks[chunk]->dft_chunks; cur;
         cur = cur->next_in_chunk, ++di, ++descriptor_index) {
      const DftDescriptor &d = dfts[descriptor_index];
      const ArrayId phase =
          f.array_catalog->find({chunk, int(array_kind::dft_phase), int(cur->c), -1, di});
      CHECK(d.phase_scratch == phase && is_valid(phase),
            "DFT descriptor has no cataloged phase scratch");
      CHECK(d.omega == cur->omega, "DFT descriptor frequencies differ from the live chunk");
      CHECK(d.scale == cur->scale, "DFT descriptor scale differs from the live chunk");
      CHECK(d.source_field.id ==
                f.array_catalog->find({chunk, int(array_kind::f), int(cur->c), 0, 0}),
            "DFT descriptor has the wrong real source array");
      CHECK(d.source_field.elements == size_t(f.chunks[chunk]->gv.ntot()),
            "DFT real source span has the wrong size");
      const ArrayId imag =
          f.array_catalog->find({chunk, int(array_kind::f), int(cur->c), 1, 0});
      CHECK(d.source_field_imag.id == imag, "DFT descriptor has the wrong imaginary source array");
      CHECK(d.source_field_imag.elements ==
                (is_valid(imag) ? size_t(f.chunks[chunk]->gv.ntot()) : 0),
            "DFT imaginary source span has the wrong size");
    }
  }

  for (const DftDescriptor &d : dfts) {
    CHECK(is_valid(d.accumulator), "DFT descriptor has no accumulator array");
    CHECK(is_valid(d.phase_scratch), "DFT descriptor has no phase scratch array");
    CHECK(d.Nomega == 3, "expected 3 frequencies, got %zu", d.Nomega);
    CHECK(d.omega.size() == d.Nomega, "DFT omega table length differs from Nomega");
    CHECK(d.decimation_factor >= 1, "decimation factor %d is not positive", d.decimation_factor);
    if (d.persist) {
      ++persistent;
      /* The load-bearing distinction: dft_chunk::norm2 and the design-region
         gradient iterate is_old/ie_old while indexing the is/ie array. If a
         lowering collapses the two, dft_norm() is wrong in every adjoint run. */
      bool differs = false;
      LOOP_OVER_DIRECTIONS(d.is.dim, dd) {
        if (d.is.in_direction(dd) != d.is_old.in_direction(dd) ||
            d.ie.in_direction(dd) != d.ie_old.in_direction(dd))
          differs = true;
      }
      if (differs) ++padded;
    }
    else {
      LOOP_OVER_DIRECTIONS(d.is.dim, dd) {
        CHECK(d.is.in_direction(dd) == d.is_old.in_direction(dd) &&
                  d.ie.in_direction(dd) == d.ie_old.in_direction(dd),
              "a non-persistent monitor has a padded extent");
      }
    }
  }
  CHECK(or_to_all(persistent > 0), "the persistent monitor produced no descriptor");
  CHECK(or_to_all(padded > 0), "no persistent descriptor recorded a padded extent");
  master_printf("dfts: %zu descriptors, %zu persistent\n", dfts.size(), persistent);
}

static void flux_components(ndim dim, direction normal, component e[2], component h[2]) {
  switch (normal) {
    case X: e[0] = Ey, e[1] = Ez, h[0] = Hz, h[1] = Hy; return;
    case Y: e[0] = Ez, e[1] = Ex, h[0] = Hx, h[1] = Hz; return;
    case R: e[0] = Ep, e[1] = Ez, h[0] = Hz, h[1] = Hp; return;
    case P: e[0] = Ez, e[1] = Er, h[0] = Hr, h[1] = Hz; return;
    case Z:
      if (dim == Dcyl)
        e[0] = Er, e[1] = Ep, h[0] = Hp, h[1] = Hr;
      else
        e[0] = Ex, e[1] = Ey, h[0] = Hy, h[1] = Hx;
      return;
    case NO_DIRECTION: break;
  }
  CHECK(false, "invalid test flux normal");
}

static void test_legacy_flux_descriptors() {
  grid_volume gv = vol2d(4.0, 4.0, 8.0);
  structure s(gv, one, no_pml(), identity(), 2);
  fields f(&s);
  f.require_component(Ex);
  f.require_component(Ey);
  f.require_component(Ez);
  f.require_component(Hx);
  f.require_component(Hy);
  f.require_component(Hz);
  const volume xwhere(vec(0.25, -0.8), vec(0.25, 0.8));
  const volume zwhere(vec(-0.7, -0.6), vec(0.7, 0.6));
  f.add_flux_vol(X, xwhere);
  f.add_flux_vol(Z, zwhere); // linked-list ordinal zero (newest first)
  f.advance(1);

  refresh_legacy_flux_descriptors(f);
  const std::vector<LegacyFluxDescriptor> &fluxes = f.descriptors->legacy_fluxes;
  CHECK(fluxes.size() == 2, "legacy flux descriptor list has %zu rows, expected 2",
        fluxes.size());
  if (fluxes.size() == 2)
    CHECK(fluxes[0].recipe_signature != fluxes[1].recipe_signature,
          "legacy flux recipe identity ignored normal/volume differences");
  const direction normals[2] = {Z, X};
  const volume wheres[2] = {zwhere, xwhere};
  size_t local_terms = 0;
  bool saw_same_grid_offsets = false;
  for (size_t fi = 0; fi < fluxes.size() && fi < 2; ++fi) {
    const LegacyFluxDescriptor &descriptor = fluxes[fi];
    CHECK(descriptor.flux_ordinal == fi && descriptor.normal == normals[fi],
          "legacy flux descriptor %zu changed linked-list order", fi);
    component e[2], h[2];
    flux_components(f.gv.dim, normals[fi], e, h);
    size_t ti = 0;
    for (uint32_t pair = 0; pair < 2; ++pair) {
      const component cgrid = f.gv.iyee_shift(e[pair]) == f.gv.iyee_shift(h[pair])
                                  ? e[pair]
                                  : Centered;
      const ChunkLoopPlan regions = prepare_loop_in_chunks(f, wheres[fi], cgrid);
      for (size_t ri = 0; ri < regions.regions.size(); ++ri, ++ti) {
        CHECK(ti < descriptor.terms.size(),
              "legacy flux %zu omitted pair %u region %zu", fi, pair, ri);
        if (ti >= descriptor.terms.size()) continue;
        const LegacyFluxTermDescriptor &term = descriptor.terms[ti];
        const ChunkLoopRegion &region = regions.regions[ri];
        const fields_chunk &fc = *f.chunks[region.chunk];
        const component ec = f.S.transform(e[pair], -region.symmetry_index);
        const component hc = f.S.transform(h[pair], -region.symmetry_index);
        CHECK(term.term_ordinal == pair && term.region_ordinal == ri &&
                  term.sign == (pair ? -1 : 1),
              "legacy flux %zu term %zu changed pair/region/sign order", fi, ti);
        CHECK(term.chunk == region.chunk && term.e_component == ec && term.h_component == hc,
              "legacy flux %zu term %zu has the wrong chunk or transformed components", fi, ti);
        CHECK(term.e_real ==
                      f.array_catalog->find({region.chunk, int(array_kind::f), int(ec), 0, 0}) &&
                  term.e_imag ==
                      f.array_catalog->find({region.chunk, int(array_kind::f), int(ec), 1, 0}) &&
                  term.h_real ==
                      f.array_catalog->find({region.chunk, int(array_kind::f), int(hc), 0, 0}) &&
                  term.h_imag ==
                      f.array_catalog->find({region.chunk, int(array_kind::f), int(hc), 1, 0}),
              "legacy flux %zu term %zu has stale field identities", fi, ti);
        ptrdiff_t eo0 = 0, eo1 = 0, ho0 = 0, ho1 = 0;
        if (cgrid == Centered) {
          fc.gv.yee2cent_offsets(ec, eo0, eo1);
          fc.gv.yee2cent_offsets(hc, ho0, ho1);
        }
        else
          saw_same_grid_offsets = true;
        CHECK(term.e_offsets[0] == eo0 && term.e_offsets[1] == eo1 &&
                  term.h_offsets[0] == ho0 && term.h_offsets[1] == ho1,
              "legacy flux %zu term %zu changed cgrid interpolation offsets", fi, ti);
        const complex<double> ep = region.phase * f.S.phase_shift(ec, region.symmetry_index);
        const complex<double> hp = region.phase * f.S.phase_shift(hc, region.symmetry_index);
        CHECK(complex<double>(term.phase_real, term.phase_imag) == conj(ep) * hp,
              "legacy flux %zu term %zu changed the E*/H phase product", fi, ti);
        size_t expected_base = 0;
        for (int axis = 0; axis < 3; ++axis) {
          const direction d = fc.gv.yucky_direction(axis);
          const size_t count = size_t((region.end.yucky_val(axis) -
                                       region.begin.yucky_val(axis)) /
                                          2 +
                                      1);
          const ptrdiff_t stride = fc.gv.stride(d);
          expected_base += size_t((region.begin.yucky_val(axis) -
                                   fc.gv.little_corner().yucky_val(axis)) /
                                  2) *
                           size_t(stride);
          CHECK(term.counts[axis] == count && term.strides[axis] == stride,
                "legacy flux %zu term %zu changed loop shape at axis %d", fi, ti, axis);
          CHECK(term.boundary_weights[axis][0] == region.weights.s0.in_direction(d) &&
                    term.boundary_weights[axis][1] == region.weights.s1.in_direction(d) &&
                    term.boundary_weights[axis][2] == region.weights.e0.in_direction(d) &&
                    term.boundary_weights[axis][3] == region.weights.e1.in_direction(d),
                "legacy flux %zu term %zu changed boundary weights at axis %d", fi, ti, axis);
        }
        CHECK(term.base == expected_base && term.begin == region.begin && term.end == region.end &&
                  term.lattice_shift == region.lattice_shift &&
                  term.symmetry_index == region.symmetry_index && term.dV0 == region.dV0 &&
                  term.dV1 == region.dV1,
              "legacy flux %zu term %zu changed its portable region", fi, ti);
        ++local_terms;
      }
    }
    CHECK(ti == descriptor.terms.size(),
          "legacy flux %zu has %zu terms but exact per-pair plans consumed %zu", fi,
          descriptor.terms.size(), ti);
  }
  CHECK(or_to_all(local_terms > 0), "legacy flux descriptor fixture produced no terms");
  CHECK(or_to_all(saw_same_grid_offsets),
        "Cartesian Z legacy flux did not exercise zero-offset shared-grid recipes");

  f.descriptors->clear();
  CHECK(f.descriptors->legacy_fluxes.empty() && f.descriptors->legacy_flux_generation == 0,
        "DescriptorSet::clear retained legacy flux recipes");
}

static void require_all_cartesian_fields(fields &f) {
  f.require_component(Ex);
  f.require_component(Ey);
  f.require_component(Ez);
  f.require_component(Hx);
  f.require_component(Hy);
  f.require_component(Hz);
}

static void check_flux_pair_mapping(const fields &f, const LegacyFluxDescriptor &descriptor,
                                    const char *name) {
  component e[2], h[2];
  flux_components(f.gv.dim, descriptor.normal, e, h);
  for (const LegacyFluxTermDescriptor &term : descriptor.terms) {
    CHECK(term.term_ordinal < 2, "%s has invalid signed-product ordinal %u", name,
          term.term_ordinal);
    if (term.term_ordinal >= 2) continue;
    CHECK(term.e_component == f.S.transform(e[term.term_ordinal], -term.symmetry_index) &&
              term.h_component == f.S.transform(h[term.term_ordinal], -term.symmetry_index) &&
              term.sign == (term.term_ordinal ? -1 : 1),
          "%s has the wrong signed E/H mapping", name);
  }
}

static void test_legacy_flux_direction_and_geometry_cases() {
  { // All Cartesian normals, real fields, uneven ownership and clipped identity.
    grid_volume gv = vol3d(2.0, 2.5, 3.0, 4.0);
    structure s(gv, one, no_pml(), identity(), 3);
    fields f(&s);
    f.use_real_fields();
    require_all_cartesian_fields(f);
    f.add_flux_vol(X, volume(vec(0.2, -0.8, -0.7), vec(0.2, 0.9, 0.8)));
    f.add_flux_vol(Y, volume(vec(-0.7, -0.1, -0.6), vec(0.8, -0.1, 0.7)));
    f.add_flux_vol(Z, volume(vec(-0.6, -0.7, 0.15), vec(0.7, 0.8, 0.15)));
    f.advance(1);
    refresh_legacy_flux_descriptors(f);
    const direction expected[] = {Z, Y, X};
    CHECK(f.descriptors->legacy_fluxes.size() == 3,
          "Cartesian direction fixture produced %zu monitors",
          f.descriptors->legacy_fluxes.size());
    size_t local_terms = 0;
    for (size_t i = 0; i < f.descriptors->legacy_fluxes.size() && i < 3; ++i) {
      const LegacyFluxDescriptor &descriptor = f.descriptors->legacy_fluxes[i];
      CHECK(descriptor.normal == expected[i], "Cartesian monitor %zu changed list order", i);
      check_flux_pair_mapping(f, descriptor, "Cartesian legacy flux");
      for (const LegacyFluxTermDescriptor &term : descriptor.terms) {
        CHECK(!is_valid(term.e_imag) && !is_valid(term.h_imag),
              "real Cartesian flux recipe invented imaginary field arrays");
        ++local_terms;
      }
    }
    CHECK(or_to_all(local_terms > 0), "Cartesian direction fixture produced no local terms");

    fields fa(&s), fb(&s);
    require_all_cartesian_fields(fa);
    require_all_cartesian_fields(fb);
    fa.add_flux_vol(Z, volume(vec(-3.0, -3.0, 0.0), vec(3.0, 3.0, 0.0)));
    fb.add_flux_vol(Z, volume(vec(-4.0, -4.0, 0.0), vec(4.0, 4.0, 0.0)));
    fa.advance(1);
    fb.advance(1);
    refresh_legacy_flux_descriptors(fa);
    refresh_legacy_flux_descriptors(fb);
    const std::vector<LegacyFluxDescriptor> &clipped_a = fa.descriptors->legacy_fluxes;
    const std::vector<LegacyFluxDescriptor> &clipped_b = fb.descriptors->legacy_fluxes;
    CHECK(clipped_a.size() == 1 && clipped_b.size() == 1,
          "clipped legacy flux fixture did not build both recipes");
    if (clipped_a.size() == 1 && clipped_b.size() == 1) {
      CHECK(clipped_a[0].recipe_signature != clipped_b[0].recipe_signature,
            "clipped legacy flux volumes collide in plan identity");
      const StepPlan plan_a = build_step_plan(fa, StepProgram::ordinary);
      const StepPlan plan_b = build_step_plan(fb, StepProgram::ordinary);
      CHECK(plan_a.legacy_flux_terms == plan_b.legacy_flux_terms,
            "clipped-volume fixture did not realize the same local flux terms");
      CHECK(compute_step_plan_signature(plan_a) != compute_step_plan_signature(plan_b),
            "StepPlan signature ignored different requested volumes with identical clipping");
    }

    fields fnx(&s), fny(&s);
    require_all_cartesian_fields(fnx);
    require_all_cartesian_fields(fny);
    const volume same_where(vec(-0.6, -0.5, -0.4), vec(0.6, 0.5, 0.4));
    fnx.add_flux_vol(X, same_where);
    fny.add_flux_vol(Y, same_where);
    fnx.advance(1);
    fny.advance(1);
    refresh_legacy_flux_descriptors(fnx);
    refresh_legacy_flux_descriptors(fny);
    CHECK(fnx.descriptors->legacy_fluxes[0].recipe_signature !=
              fny.descriptors->legacy_fluxes[0].recipe_signature,
          "legacy flux recipe identity ignored the requested normal");
    CHECK(compute_step_plan_signature(build_step_plan(fnx, StepProgram::ordinary)) !=
              compute_step_plan_signature(build_step_plan(fny, StepProgram::ordinary)),
          "StepPlan signature ignored the requested legacy flux normal");
  }

  { // Symmetry plus periodic images/Bloch phase.
    grid_volume gv = vol2d(3.0, 3.0, 6.0);
    structure s(gv, one, no_pml(), mirror(Y, gv), 2);
    fields f(&s);
    f.use_bloch(vec(0.13, 0.0));
    f.require_component(Ex);
    f.require_component(Ey);
    f.require_component(Hx);
    f.require_component(Hy);
    f.add_flux_vol(Z, volume(vec(-2.2, -1.2), vec(2.2, 1.2)));
    f.advance(1);
    refresh_legacy_flux_descriptors(f);
    bool saw_symmetry = false, saw_nontrivial_phase = false, saw_imag = false;
    if (!f.descriptors->legacy_fluxes.empty()) {
      const LegacyFluxDescriptor &descriptor = f.descriptors->legacy_fluxes[0];
      check_flux_pair_mapping(f, descriptor, "symmetry/Bloch legacy flux");
      for (const LegacyFluxTermDescriptor &term : descriptor.terms) {
        saw_symmetry = saw_symmetry || term.symmetry_index != 0;
        saw_nontrivial_phase = saw_nontrivial_phase || term.phase_real != 1.0 ||
                                                     term.phase_imag != 0.0;
        saw_imag = saw_imag || is_valid(term.e_imag) || is_valid(term.h_imag);
      }
    }
    CHECK(or_to_all(saw_symmetry), "legacy flux symmetry fixture used no transformed region");
    CHECK(or_to_all(saw_nontrivial_phase), "legacy flux Bloch fixture used only unit phases");
    CHECK(or_to_all(saw_imag), "legacy flux Bloch fixture exported no imaginary field arrays");
  }

  { // Cylindrical R/P/Z mappings plus radial integration weights and PML storage.
    grid_volume gv = volcyl(2.0, 2.5, 6.0);
    structure s(gv, one, pml(0.4));
    fields f(&s, 1);
    f.require_component(Er);
    f.require_component(Ep);
    f.require_component(Ez);
    f.require_component(Hr);
    f.require_component(Hp);
    f.require_component(Hz);
    f.add_flux_vol(R, volume(veccyl(0.7, -0.6), veccyl(0.7, 0.6)));
    f.add_flux_vol(P, volume(veccyl(0.2, -0.5), veccyl(1.4, 0.5)));
    f.add_flux_vol(Z, volume(veccyl(0.2, 0.15), veccyl(1.5, 0.15)));
    f.advance(1);
    refresh_legacy_flux_descriptors(f);
    const direction expected[] = {Z, P, R};
    bool saw_radial_weight = false;
    CHECK(f.descriptors->legacy_fluxes.size() == 3,
          "cylindrical direction fixture produced %zu monitors",
          f.descriptors->legacy_fluxes.size());
    for (size_t i = 0; i < f.descriptors->legacy_fluxes.size() && i < 3; ++i) {
      const LegacyFluxDescriptor &descriptor = f.descriptors->legacy_fluxes[i];
      CHECK(descriptor.normal == expected[i], "cylindrical monitor %zu changed list order", i);
      check_flux_pair_mapping(f, descriptor, "cylindrical legacy flux");
      for (const LegacyFluxTermDescriptor &term : descriptor.terms) {
        saw_radial_weight = saw_radial_weight || term.dV1 != 0.0;
        if (descriptor.normal == P)
          CHECK(term.e_offsets[0] == 0 && term.e_offsets[1] == 0 &&
                    term.h_offsets[0] == 0 && term.h_offsets[1] == 0,
                "cylindrical P flux did not retain its shared component grid");
      }
    }
    CHECK(or_to_all(saw_radial_weight), "cylindrical legacy flux lost its radial dV term");
  }
}

static void test_polarizations(const char *name, susceptibility *sus,
                               SusceptibilityKind expect_kind, bool expect_layout) {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5));
  s.add_susceptibility(one, E_stuff, *sus);
  fields f(&s);
  gaussian_src_time g(0.3, 0.1);
  f.add_point_source(Ez, g, vec(0.11, 0.13));
  f.advance(4);

  std::vector<PolarizationDescriptor> pols;
  build_polarization_descriptors(f, pols);
  CHECK(or_to_all(!pols.empty()), "%s: no polarization descriptors", name);

  for (const PolarizationDescriptor &d : pols) {
    CHECK(d.kind == expect_kind, "%s: classified as %s, expected %s", name,
          susceptibility_kind_name(d.kind), susceptibility_kind_name(expect_kind));
    if (expect_layout) {
      CHECK(!d.internal_arrays.empty(), "%s: published an empty layout", name);
      /* The published offsets must address the same memory the object's own
         interior pointers do. If they ever disagree, the object is right. */
      for (const InternalArrayLayout &l : d.internal_arrays)
        CHECK(l.elements > 0, "%s: layout entry '%s' has zero elements", name, l.name);
    }
    else {
      CHECK(d.internal_arrays.empty(), "%s: host_custom must publish no layout", name);
    }
  }
  if (!pols.empty())
    master_printf("%s: %zu descriptors, %zu layout entries\n", name, pols.size(),
                  pols[0].internal_arrays.size());
  delete sus;
}

/* The published offsets must resolve to exactly the pointers
   cinternal_notowned_ptr hands out. */
static void test_layout_matches_pointers() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5));
  lorentzian_susceptibility lor(1.1, 1e-5);
  s.add_susceptibility(one, E_stuff, lor);
  fields f(&s);
  gaussian_src_time g(0.3, 0.1);
  f.add_point_source(Ez, g, vec(0.11, 0.13));
  f.advance(4);

  size_t compared = 0, bad = 0;
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    for (polarization_state *p = f.chunks[i]->pol[E_stuff]; p; p = p->next) {
      if (!p->data) continue;
      std::vector<InternalArrayLayout> layout;
      if (!p->s->internal_layout(layout, f.chunks[i]->gv, p->data)) continue;
      const realnum *base = (const realnum *)p->data;
      for (const InternalArrayLayout &l : layout) {
        if (strcmp(l.name, "P") != 0) continue;
        const realnum *want = p->s->cinternal_notowned_ptr(0, l.c, l.cmp, 0, p->data);
        if (!want) continue;
        ++compared;
        if (base + l.offset_elements != want) ++bad;
      }
    }
  }
  CHECK(bad == 0, "%zu of %zu published P offsets disagree with cinternal_notowned_ptr", bad,
        compared);
  CHECK(or_to_all(compared > 0), "no published offsets were compared");
  master_printf("layout: %zu offsets checked against the object's own pointers\n", compared);
}

static void test_gyrotropic_layout_matches_pointers() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5));
  gyrotropic_susceptibility gyro(vec(0.17, -0.23, 0.31), 1.1, 1e-5);
  s.add_susceptibility(one, E_stuff, gyro);
  fields f(&s);
  f.require_component(Ez);
  f.advance(2);

  size_t rows = 0;
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    for (polarization_state *p = f.chunks[i]->pol[E_stuff]; p; p = p->next) {
      if (!p->data || typeid(*p->s) != typeid(gyrotropic_susceptibility)) continue;
      std::vector<InternalArrayLayout> layout;
      CHECK(p->s->internal_layout(layout, f.chunks[i]->gv, p->data),
            "gyrotropic susceptibility did not publish a layout");
      const realnum *base = (const realnum *)p->data;
      CHECK(layout.size() % 6 == 0, "gyrotropic layout is not grouped in sixes");
      for (size_t li = 0; li + 5 < layout.size(); li += 6) {
        ++rows;
        for (int dd = 0; dd < 3; ++dd) {
          const InternalArrayLayout &pl = layout[li + 2 * dd];
          const InternalArrayLayout &ppl = layout[li + 2 * dd + 1];
          const realnum *want = p->s->cinternal_notowned_ptr(dd, pl.c, pl.cmp, 0, p->data);
          CHECK(want && base + pl.offset_elements == want,
                "gyrotropic P_%c offset disagrees with the object's pointer", 'x' + dd);
          CHECK(ppl.offset_elements == pl.offset_elements + pl.elements,
                "gyrotropic P_prev_%c is not adjacent to P_%c", 'x' + dd, 'x' + dd);
        }
      }
    }
  }
  CHECK(or_to_all(rows > 0), "no gyrotropic layout rows were checked against live pointers");
}

static void test_lorentzian_contract() {
  grid_volume gv = vol2d(3.0, 3.0, 10.0);
  structure s(gv, eps_slab, pml(0.5), identity(), 2);
  lorentzian_susceptibility lorentz(1.25, 0.075);
  lorentzian_susceptibility drude(0.8, 0.125, true);
  s.add_susceptibility(one, E_stuff, lorentz);
  s.add_susceptibility(one, H_stuff, drude);
  fields f(&s);
  f.use_bloch(vec(0.07, 0.11));
  gaussian_src_time g(0.3, 0.1);
  f.add_point_source(Ez, g, vec(0.11, 0.13));
  f.advance(2);

  std::vector<PolarizationDescriptor> pols;
  build_polarization_descriptors(f, pols);
  size_t lorentz_count = 0, drude_count = 0, state_count = 0;
  for (const PolarizationDescriptor &d : pols) {
    if (d.kind != SusceptibilityKind::lorentzian) continue;
    if (d.ft == E_stuff) {
      ++lorentz_count;
      CHECK(d.lorentzian.omega_0 == 1.25 && d.lorentzian.gamma == 0.075 &&
                !d.lorentzian.drude,
            "electric Lorentzian parameters are not exact");
    }
    if (d.ft == H_stuff) {
      ++drude_count;
      CHECK(d.lorentzian.omega_0 == 0.8 && d.lorentzian.gamma == 0.125 &&
                d.lorentzian.drude,
            "magnetic Drude parameters are not exact");
    }
    bool seen[NUM_FIELD_COMPONENTS][2] = {};
    uint64_t state_mask = 0;
    for (const LorentzianStateArrays &state : d.lorentzian_states) {
      ++state_count;
      CHECK(!seen[int(state.c)][state.cmp], "duplicate Lorentzian component/cmp row");
      seen[int(state.c)][state.cmp] = true;
      CHECK(state.elements == size_t(f.chunks[d.chunk]->gv.ntot()),
            "Lorentzian state has the wrong extent");
      CHECK(is_valid(state.p) && is_valid(state.p_prev) && state.p != state.p_prev,
            "Lorentzian state lacks distinct P/P_prev ArrayIds");
      const ArraySpec &p = f.array_catalog->spec(state.p);
      const ArraySpec &p_prev = f.array_catalog->spec(state.p_prev);
      CHECK(p.role == array_role::polarization && p_prev.role == array_role::polarization &&
                p.element_type == ElementType::realnum_value &&
                p_prev.element_type == ElementType::realnum_value,
            "Lorentzian state storage metadata is wrong");
      const uint64_t bit = uint64_t(1) << (2 * int(state.c) + state.cmp);
      state_mask |= bit;
      CHECK(d.required_w & bit, "Lorentzian required-W mask collapsed a component/cmp row");
    }
    CHECK(d.required_w == state_mask,
          "Lorentzian required-W mask advertises fields without state arrays");
  }
  CHECK(or_to_all(lorentz_count > 0 && drude_count > 0),
        "Lorentzian/Drude descriptors were not both emitted");
  CHECK(or_to_all(state_count > 0), "Lorentzian descriptors contained no bound state arrays");
}

/* Polarization state is a snapshot of the field layout at allocation time.
   Growing the live fields later must not make its descriptor advertise state
   arrays that do not exist.  Two chunks leave idle ranks at np=4, while the
   one-sided PML keeps ownership and auxiliary allocation asymmetric. */
static void test_lorentzian_layout_after_field_growth() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(1.0, X, High), identity(), 2);
  lorentzian_susceptibility lorentz(1.1, 1e-5);
  s.add_susceptibility(one, E_stuff, lorentz);
  fields f(&s);

  gaussian_src_time first(0.3, 0.1);
  f.add_point_source(Ez, first, vec(0.13, 0.11));
  f.advance(1);
  f.require_component(Hz);
  gaussian_src_time second(0.25, 0.1, 0.0, 6.0);
  second.is_integrated = true;
  f.add_point_source(Ez, second, vec(-0.4, 0.3));
  f.advance(1);

  size_t descriptors = 0, live_supersets = 0;
  for (const PolarizationDescriptor &d : f.descriptors->polarizations) {
    if (d.kind != SusceptibilityKind::lorentzian) continue;
    ++descriptors;
    CHECK(d.chunk >= 0 && d.chunk < f.num_chunks && f.chunks[d.chunk]->is_mine(),
          "polarization descriptor was emitted for a non-owned chunk");

    uint64_t state_mask = 0;
    for (const LorentzianStateArrays &state : d.lorentzian_states)
      state_mask |= uint64_t(1) << (2 * int(state.c) + state.cmp);
    CHECK(d.required_w == state_mask,
          "grown field layout leaked nonexistent Lorentzian state into required-W mask");

    polarization_state *p = f.chunks[d.chunk]->pol[d.ft];
    for (int i = 0; p && i < d.state_index; ++i)
      p = p->next;
    CHECK(p != NULL, "descriptor state index does not resolve to a live susceptibility");
    if (!p) continue;

    uint64_t live_mask = 0;
    FOR_COMPONENTS(c) DOCMP2 {
      if (p->s->needs_P(c, cmp, f.chunks[d.chunk]->f))
        live_mask |= uint64_t(1) << (2 * int(c) + cmp);
    }
    if (live_mask & ~state_mask) ++live_supersets;
  }

  CHECK(sum_to_all(descriptors) > 0, "field-growth case produced no Lorentzian descriptors");
  CHECK(or_to_all(live_supersets > 0),
        "field-growth case did not exercise a needs_P superset of allocated state");
}

static void test_gyrotropic_contract() {
  grid_volume gv = vol3d(2.0, 2.0, 2.0, 8.0);
  structure s(gv, eps_slab, pml(0.25), identity(), 2);
  const vec bias(0.17, -0.23, 0.31);
  gyrotropic_susceptibility lorentz(bias, 1.25, 0.075, 0.0,
                                    GYROTROPIC_LORENTZIAN);
  gyrotropic_susceptibility drude(bias, 0.8, 0.125, 0.0, GYROTROPIC_DRUDE);
  gyrotropic_susceptibility saturated(bias, 0.55, 0.035, 0.19,
                                      GYROTROPIC_SATURATED);
  s.add_susceptibility(one, E_stuff, lorentz);
  s.add_susceptibility(one, E_stuff, drude);
  s.add_susceptibility(one, H_stuff, saturated);
  fields f(&s);
  f.require_component(Ex);
  f.require_component(Ey);
  f.require_component(Ez);
  f.require_component(Hx);
  f.require_component(Hy);
  f.require_component(Hz);
  f.advance(2);

  std::vector<PolarizationDescriptor> pols;
  build_polarization_descriptors(f, pols);
  size_t models[3] = {};
  size_t rows = 0;
  for (const PolarizationDescriptor &d : pols) {
    if (d.kind != SusceptibilityKind::gyrotropic) continue;
    CHECK(int(d.gyrotropic.model) >= int(GYROTROPIC_LORENTZIAN) &&
              int(d.gyrotropic.model) <= int(GYROTROPIC_SATURATED),
          "gyrotropic descriptor has an invalid model");
    if (int(d.gyrotropic.model) >= 0 && int(d.gyrotropic.model) < 3)
      ++models[int(d.gyrotropic.model)];
    if (d.gyrotropic.model == GYROTROPIC_LORENTZIAN)
      CHECK(d.gyrotropic.omega_0 == realnum(1.25) &&
                d.gyrotropic.gamma == realnum(0.075),
            "gyrotropic Lorentzian parameters are not exact realnum values");
    if (d.gyrotropic.model == GYROTROPIC_DRUDE)
      CHECK(d.gyrotropic.omega_0 == realnum(0.8) &&
                d.gyrotropic.gamma == realnum(0.125),
            "gyrotropic Drude parameters are not exact realnum values");
    if (d.gyrotropic.model == GYROTROPIC_SATURATED) {
      const vec normalized = bias / abs(bias);
      CHECK(d.gyrotropic.alpha == realnum(0.19) &&
                d.gyrotropic.gyro_tensor[Y][Z] == realnum(normalized.x()) &&
                d.gyrotropic.gyro_tensor[Z][Y] == realnum(-normalized.x()) &&
                d.gyrotropic.gyro_tensor[Z][X] == realnum(normalized.y()) &&
                d.gyrotropic.gyro_tensor[X][Z] == realnum(-normalized.y()) &&
                d.gyrotropic.gyro_tensor[X][Y] == realnum(normalized.z()) &&
                d.gyrotropic.gyro_tensor[Y][X] == realnum(-normalized.z()),
            "saturated gyrotropic tensor was not copied after host normalization");
    }

    uint64_t expected_w = 0;
    bool seen[NUM_FIELD_COMPONENTS][2] = {};
    for (const GyrotropicStateArrays &state : d.gyrotropic_states) {
      ++rows;
      CHECK(!seen[int(state.c)][state.cmp], "duplicate gyrotropic component/cmp row");
      seen[int(state.c)][state.cmp] = true;
      CHECK(state.elements == size_t(f.chunks[d.chunk]->gv.ntot()),
            "gyrotropic state has the wrong extent");
      ArrayId ids[6];
      for (int dd = 0; dd < 3; ++dd) {
        ids[2 * dd] = state.p[dd];
        ids[2 * dd + 1] = state.p_prev[dd];
        CHECK(is_valid(ids[2 * dd]) && is_valid(ids[2 * dd + 1]),
              "gyrotropic state lacks a P/P_prev ArrayId");
      }
      for (int a = 0; a < 6; ++a) {
        const ArraySpec &spec = f.array_catalog->spec(ids[a]);
        CHECK(spec.role == array_role::polarization &&
                  spec.element_type == ElementType::realnum_value &&
                  spec.elements == state.elements,
              "gyrotropic state storage metadata is wrong");
        for (int b = a + 1; b < 6; ++b)
          CHECK(ids[a] != ids[b], "gyrotropic state ArrayIds are not distinct");
      }
      const direction d0 = component_direction(state.c);
      expected_w |= uint64_t(1) << (2 * int(state.c) + state.cmp);
      for (int turn = 1; turn <= 2; ++turn) {
        const component cross = direction_component(
            state.c, cycle_direction(f.chunks[d.chunk]->gv.dim, d0, turn));
        if (f.chunks[d.chunk]->f[cross][state.cmp])
          expected_w |= uint64_t(1) << (2 * int(cross) + state.cmp);
      }
    }
    CHECK(d.required_w == expected_w,
          "gyrotropic required-W mask omitted or invented a cross driving field");
    CHECK(d.internal_arrays.size() == 6 * d.gyrotropic_states.size(),
          "gyrotropic descriptor did not publish six arrays per row");
  }
  CHECK(or_to_all(models[0] && models[1] && models[2]),
        "all three gyrotropic models were not described");
  CHECK(or_to_all(rows > 0), "gyrotropic descriptors contained no state rows");
}

static void test_gyrotropic_layout_after_field_growth() {
  grid_volume gv = vol2d(4.0, 4.0, 10.0);
  structure s(gv, eps_slab, pml(1.0, X, High), identity(), 2);
  gyrotropic_susceptibility gyro(vec(0.17, -0.23, 0.31), 1.1, 1e-5);
  s.add_susceptibility(one, E_stuff, gyro);
  fields f(&s);
  f.require_component(Ez);
  f.advance(1);
  f.require_component(Ex);
  f.advance(1);

  size_t descriptors = 0, live_supersets = 0;
  for (const PolarizationDescriptor &d : f.descriptors->polarizations) {
    if (d.kind != SusceptibilityKind::gyrotropic) continue;
    ++descriptors;
    uint64_t row_mask = 0;
    for (const GyrotropicStateArrays &state : d.gyrotropic_states)
      row_mask |= uint64_t(1) << (2 * int(state.c) + state.cmp);
    polarization_state *p = f.chunks[d.chunk]->pol[d.ft];
    for (int i = 0; p && i < d.state_index; ++i)
      p = p->next;
    uint64_t live_mask = 0;
    if (p)
      FOR_COMPONENTS(c) DOCMP2 if (p->s->needs_P(c, cmp, f.chunks[d.chunk]->f)) {
        live_mask |= uint64_t(1) << (2 * int(c) + cmp);
      }
    if (live_mask & ~row_mask) ++live_supersets;
    for (const GyrotropicStateArrays &state : d.gyrotropic_states)
      CHECK(state.elements == size_t(f.chunks[d.chunk]->gv.ntot()),
            "field growth changed a gyrotropic state extent");
  }
  CHECK(sum_to_all(descriptors) > 0, "field-growth case produced no gyrotropic descriptors");
  CHECK(or_to_all(live_supersets > 0),
        "field-growth case did not exercise a gyrotropic needs_P superset");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  test_sources();
  test_source_ordinals();
  test_derived_source_times_remain_host_custom();
  test_dfts();
  test_legacy_flux_descriptors();
  test_legacy_flux_direction_and_geometry_cases();
  test_polarizations("lorentzian", new lorentzian_susceptibility(1.1, 1e-5),
                     SusceptibilityKind::lorentzian, true);
  test_polarizations("noisy", new noisy_lorentzian_susceptibility(0.01, 1.1, 0.05),
                     SusceptibilityKind::noisy_lorentzian, true);
  test_polarizations("third-party opaque", new opaque_susceptibility(1.1, 1e-5),
                     SusceptibilityKind::host_custom, false);
  test_polarizations("derived Lorentzian", new inherited_lorentzian_susceptibility(1.1, 1e-5),
                     SusceptibilityKind::host_custom, true);
  test_polarizations("gyrotropic",
                     new gyrotropic_susceptibility(vec(0.17, -0.23, 0.31), 0.8, 0.05),
                     SusceptibilityKind::gyrotropic, true);
  test_polarizations("derived gyrotropic", new inherited_gyrotropic_susceptibility(),
                     SusceptibilityKind::host_custom, true);
  test_layout_matches_pointers();
  test_gyrotropic_layout_matches_pointers();
  test_lorentzian_contract();
  test_lorentzian_layout_after_field_growth();
  test_gyrotropic_contract();
  test_gyrotropic_layout_after_field_growth();

  if (failures) {
    master_printf("descriptors: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("descriptors: all checks passed\n");
  return 0;
}
