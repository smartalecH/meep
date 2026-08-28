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

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  test_sources();
  test_derived_source_times_remain_host_custom();
  test_dfts();
  test_polarizations("lorentzian", new lorentzian_susceptibility(1.1, 1e-5),
                     SusceptibilityKind::lorentzian, true);
  test_polarizations("noisy", new noisy_lorentzian_susceptibility(0.01, 1.1, 0.05),
                     SusceptibilityKind::noisy_lorentzian, true);
  test_polarizations("third-party opaque", new opaque_susceptibility(1.1, 1e-5),
                     SusceptibilityKind::host_custom, false);
  test_layout_matches_pointers();

  if (failures) {
    master_printf("descriptors: %d FAILURE(S)\n", failures);
    return 1;
  }
  master_printf("descriptors: all checks passed\n");
  return 0;
}
