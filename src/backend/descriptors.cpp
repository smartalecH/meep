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

#include "backend/descriptors.hpp"
#include "backend/lifecycle.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <typeinfo>

namespace meep {

class source_descriptor_builder {
public:
  static void parameters(const gaussian_src_time &st, std::vector<double> &out) {
    out.push_back(st.freq);
    out.push_back(st.width);
    out.push_back(st.peak_time);
    out.push_back(st.cutoff);
  }

  static void parameters(const continuous_src_time &st, std::vector<double> &out) {
    out.push_back(st.freq.real());
    out.push_back(st.freq.imag());
    out.push_back(st.width);
    out.push_back(st.start_time);
    out.push_back(st.end_time);
    out.push_back(st.slowness);
  }
};

const char *source_time_kind_name(SourceTimeKind k) {
  switch (k) {
    case SourceTimeKind::gaussian: return "gaussian";
    case SourceTimeKind::continuous: return "continuous";
    case SourceTimeKind::host_custom: return "host_custom";
  }
  return "?";
}

const char *susceptibility_kind_name(SusceptibilityKind k) {
  switch (k) {
    case SusceptibilityKind::lorentzian: return "lorentzian";
    case SusceptibilityKind::noisy_lorentzian: return "noisy_lorentzian";
    case SusceptibilityKind::gyrotropic: return "gyrotropic";
    case SusceptibilityKind::multilevel: return "multilevel";
    case SusceptibilityKind::host_custom: return "host_custom";
  }
  return "?";
}

/* --- Sources --------------------------------------------------------------
 *
 * A source separates into (1) usually-static spatial data -- destination
 * component, point indices, complex amplitudes, the conductivity array it
 * scales by -- and (2) a small time-dependent scalar from a src_time object.
 *
 * In Phase 1 the src_time objects stay host-polymorphic; only the encoding
 * changes. Built-in gaussian and continuous sources get closed parameter
 * descriptors; custom_src_time and Python's CustomSource remain host
 * operations, keep their callable, reacquire the GIL, and are evaluated at the
 * same times and in the same order as before.
 */

namespace {

SourceTimeKind classify_src_time(const src_time *st) {
  /* Derived classes may override dipole/current while retaining the built-in
     base state. Only the exact Meep types opt into the closed-form encoding;
     every subclass remains host-polymorphic. */
  if (typeid(*st) == typeid(gaussian_src_time)) return SourceTimeKind::gaussian;
  if (typeid(*st) == typeid(continuous_src_time)) return SourceTimeKind::continuous;
  return SourceTimeKind::host_custom;
}

} // namespace

void build_source_descriptors(fields &f, SourcePlan &out) {
  out.clear();

  /* One descriptor per distinct src_time, in the order fields::sources holds
     them -- which is the order calc_sources walks, so scalar_slot indices are
     stable and meaningful. */
  std::vector<const src_time *> times;
  for (const src_time *st = f.sources; st; st = st->next) {
    SourceTimeDescriptor d;
    d.source_time_id = uint32_t(times.size());
    d.kind = classify_src_time(st);
    d.scalar_slot = uint32_t(times.size());
    d.host_callback_id =
        d.kind == SourceTimeKind::host_custom ? uint32_t(times.size()) : 0xffffffffu;
    d.is_integrated = st->is_integrated;
    if (d.kind == SourceTimeKind::gaussian)
      source_descriptor_builder::parameters(static_cast<const gaussian_src_time &>(*st),
                                            d.parameters);
    else if (d.kind == SourceTimeKind::continuous)
      source_descriptor_builder::parameters(static_cast<const continuous_src_time &>(*st),
                                            d.parameters);
    out.source_times.push_back(d);
    times.push_back(st);
  }
  out.scalars.assign(times.size(), SourceStepScalar{0.0, 0.0});

  /* Group by operation, then preserve the legacy chunk/src_vol order within
     each group. This gives apply_sources and integrated E/H preparation exact
     half-open spans without changing either operation's arithmetic order. */
  FOR_FIELD_TYPES(ft) for (int integrated = 0; integrated < 2; ++integrated) {
      for (int i = 0; i < f.num_chunks; ++i) {
        if (!f.chunks[i]->is_mine()) continue;
        fields_chunk &fc = *f.chunks[i];
        for (const src_vol &sv : fc.get_sources(ft)) {
          if (int(sv.t()->is_integrated) != integrated) continue;
          SourceDescriptor d;
          d.chunk = i;
          d.ft = ft;
          /* The destination is the D/B-family component of the same direction,
             exactly as fields_chunk::step_source computes it. */
          d.c = direction_component(first_field_component(ft), component_direction(sv.c));
          d.integrated = sv.t()->is_integrated;
          d.destination = f.array_catalog->find({i, int(array_kind::f), int(d.c), 0, 0});
          d.destination_imag =
              f.array_catalog->find({i, int(array_kind::f), int(d.c), 1, 0});
          d.integrated_destination = d.integrated
                                         ? f.array_catalog->find(
                                               {i, int(array_kind::f_minus_p), int(d.c), 0, 0})
                                         : invalid_array();
          d.integrated_destination_imag =
              d.integrated
                  ? f.array_catalog->find({i, int(array_kind::f_minus_p), int(d.c), 1, 0})
                  : invalid_array();
          d.condinv = f.array_catalog->find(
              {i, int(array_kind::condinv), int(d.c), -1, int(component_direction(sv.c))});
          d.source_time_id = 0xffffffffu;
          for (size_t k = 0; k < times.size(); ++k)
            if (times[k] == sv.t()) d.source_time_id = uint32_t(k);
          d.indices.reserve(sv.num_points());
          d.complex_amplitudes.reserve(sv.num_points());
          for (size_t j = 0; j < sv.num_points(); ++j) {
            d.indices.push_back(sv.index_at(j));
            d.complex_amplitudes.push_back(sv.amplitude_at(j));
          }
          out.sources.push_back(d);
        }
      }
    }
}

namespace {

std::complex<double> descriptor_dipole(const SourceTimeDescriptor &d, double time) {
  if (d.kind == SourceTimeKind::gaussian) {
    if (d.parameters.size() != 4)
      throw std::invalid_argument("gaussian source descriptor has the wrong parameter count");
    const double freq = d.parameters[0];
    const double width = d.parameters[1];
    const double peak_time = d.parameters[2];
    const double cutoff = d.parameters[3];
    const double tt = time - peak_time;
    if (float(std::fabs(tt)) > cutoff) return 0.0;
    const std::complex<double> amp = 1.0 / std::complex<double>(0, -2 * pi * freq);
    return std::exp(-tt * tt / (2 * width * width)) *
           std::polar(1.0, -2 * pi * freq * tt) * amp;
  }
  if (d.kind == SourceTimeKind::continuous) {
    if (d.parameters.size() != 6)
      throw std::invalid_argument("continuous source descriptor has the wrong parameter count");
    const std::complex<double> freq(d.parameters[0], d.parameters[1]);
    const double width = d.parameters[2];
    const double start_time = d.parameters[3];
    const double end_time = d.parameters[4];
    const double slowness = d.parameters[5];
    const float rtime = float(time);
    if (rtime < start_time || rtime > end_time) return 0.0;
    const std::complex<double> amp =
        1.0 / (std::complex<double>(0, -1.0) * (2 * pi) * freq);
    const std::complex<double> carrier =
        std::exp(std::complex<double>(0, -1.0) * (2 * pi) * freq * time) * amp;
    if (width == 0.0) return carrier;
    const double ts = (time - start_time) / width - slowness;
    const double te = (end_time - time) / width - slowness;
    return carrier * (1.0 + std::tanh(ts)) * (1.0 + std::tanh(te)) * 0.25;
  }
  throw std::invalid_argument("host-custom source descriptors require host evaluation");
}

} // namespace

SourceStepScalar evaluate_source_time_descriptor(const SourceTimeDescriptor &d, double time,
                                                 double dt) {
  SourceStepScalar result;
  result.dipole = descriptor_dipole(d, time);
  result.current = (descriptor_dipole(d, time + dt) - result.dipole) / dt;
  return result;
}

void populate_source_scalars(fields &f, SourcePlan &out) {
  size_t count = 0;
  for (const src_time *st = f.sources; st; st = st->next)
    ++count;
  if (out.source_times.size() != count || out.scalars.size() != count) return;

  size_t i = 0;
  for (const src_time *st = f.sources; st; st = st->next, ++i) {
    const SourceTimeDescriptor &d = out.source_times[i];
    if (d.source_time_id != i || d.scalar_slot >= out.scalars.size())
      throw std::logic_error("source descriptor order does not match the live source list");
    SourceStepScalar &scalar = out.scalars[d.scalar_slot];
    scalar.current = st->current();
    scalar.dipole = st->dipole();
  }
}

namespace {

void source_hash_mix(uint64_t &hash, uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
}

void source_hash_double(uint64_t &hash, double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double is not 64-bit");
  std::memcpy(&bits, &value, sizeof(bits));
  source_hash_mix(hash, bits);
}

} // namespace

uint64_t source_plan_signature(const SourcePlan &plan) {
  uint64_t hash = 0xcbf29ce484222325ull;
  source_hash_mix(hash, plan.source_times.size());
  for (const SourceTimeDescriptor &d : plan.source_times) {
    source_hash_mix(hash, d.source_time_id);
    source_hash_mix(hash, uint64_t(d.kind));
    source_hash_mix(hash, d.scalar_slot);
    source_hash_mix(hash, d.host_callback_id);
    source_hash_mix(hash, d.is_integrated);
    source_hash_mix(hash, d.parameters.size());
    for (double value : d.parameters)
      source_hash_double(hash, value);
  }
  source_hash_mix(hash, plan.sources.size());
  for (const SourceDescriptor &d : plan.sources) {
    source_hash_mix(hash, d.destination.value);
    source_hash_mix(hash, d.destination_imag.value);
    source_hash_mix(hash, d.integrated_destination.value);
    source_hash_mix(hash, d.integrated_destination_imag.value);
    source_hash_mix(hash, uint64_t(d.chunk));
    source_hash_mix(hash, uint64_t(d.c));
    source_hash_mix(hash, d.condinv.value);
    source_hash_mix(hash, d.source_time_id);
    source_hash_mix(hash, d.integrated);
    source_hash_mix(hash, uint64_t(d.ft));
    source_hash_mix(hash, d.indices.size());
    for (ptrdiff_t index : d.indices)
      source_hash_mix(hash, uint64_t(index));
    source_hash_mix(hash, d.complex_amplitudes.size());
    for (std::complex<double> amplitude : d.complex_amplitudes) {
      source_hash_double(hash, amplitude.real());
      source_hash_double(hash, amplitude.imag());
    }
  }
  return hash;
}

/* --- DFT monitors --------------------------------------------------------- */

void build_dft_descriptors(fields &f, std::vector<DftDescriptor> &out) {
  out.clear();
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    int di = 0;
    for (dft_chunk *cur = f.chunks[i]->dft_chunks; cur; cur = cur->next_in_chunk, ++di) {
      DftDescriptor d;
      d.chunk = i;
      d.c = cur->c;
      d.accumulator = f.array_catalog->find({i, int(array_kind::dft), int(cur->c), -1, di});
      d.phase_scratch =
          f.array_catalog->find({i, int(array_kind::dft_phase), int(cur->c), -1, di});
      d.source_field.id = f.array_catalog->find({i, int(array_kind::f), int(cur->c), 0, 0});
      d.source_field.offset = 0;
      d.source_field.elements = is_valid(d.source_field.id)
                                    ? f.array_catalog->spec(d.source_field.id).elements
                                    : 0;
      d.source_field_imag.id =
          f.array_catalog->find({i, int(array_kind::f), int(cur->c), 1, 0});
      d.source_field_imag.offset = 0;
      d.source_field_imag.elements = is_valid(d.source_field_imag.id)
                                         ? f.array_catalog->spec(d.source_field_imag.id).elements
                                         : 0;
      d.omega = cur->omega;
      d.scale = cur->scale;
      d.avg1 = cur->avg1;
      d.avg2 = cur->avg2;
      d.is = cur->is;
      d.ie = cur->ie;
      /* dft_chunk only assigns is_old/ie_old when persist is set -- for an
         ordinary monitor they are default-constructed and never read. The
         descriptor gives them a defined value (the unpadded extent equals the
         extent), because a lowering that reads them unconditionally is exactly
         the mistake the persist/is_old distinction invites. */
      d.persist = cur->persist;
      d.is_old = cur->persist ? cur->is_old : cur->is;
      d.ie_old = cur->persist ? cur->ie_old : cur->ie;
      d.decimation_factor = cur->get_decimation_factor();
      d.due_scalar_slot = 0;
      d.weights = BoundaryWeights(cur->s0, cur->s1, cur->e0, cur->e1);
      d.dV0 = cur->dV0;
      d.dV1 = cur->dV1;
      d.include_dV_and_interp_weights = cur->include_dV_and_interp_weights;
      d.sqrt_dV_and_interp_weights = cur->sqrt_dV_and_interp_weights;
      d.N = cur->N;
      d.Nomega = cur->omega.size();
      out.push_back(d);
    }
  }
}

/* --- Susceptibilities ----------------------------------------------------- */

namespace {

SusceptibilityKind classify_susceptibility(const susceptibility *s) {
  /* Order matters: noisy and gyrotropic derive from lorentzian. */
  if (dynamic_cast<const noisy_lorentzian_susceptibility *>(s))
    return SusceptibilityKind::noisy_lorentzian;
  if (dynamic_cast<const gyrotropic_susceptibility *>(s)) return SusceptibilityKind::gyrotropic;
  if (dynamic_cast<const multilevel_susceptibility *>(s)) return SusceptibilityKind::multilevel;
  if (dynamic_cast<const lorentzian_susceptibility *>(s)) return SusceptibilityKind::lorentzian;
  return SusceptibilityKind::host_custom;
}

} // namespace

void build_polarization_descriptors(fields &f, std::vector<PolarizationDescriptor> &out) {
  out.clear();
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    fields_chunk &fc = *f.chunks[i];
    FOR_FIELD_TYPES(ft) {
      int si = 0;
      for (polarization_state *p = fc.pol[ft]; p; p = p->next, ++si) {
        PolarizationDescriptor d;
        d.chunk = i;
        d.ft = ft;
        d.state_index = si;
        d.kind = classify_susceptibility(p->s);
        d.per_thread_scratch_elements = 0;
        d.required_w = 0;
        d.required_w_prev = 0;
        d.needs_halo = false;

        /* A susceptibility that does not publish a layout classifies as
           host_custom regardless of what it is -- that is the escape hatch
           that keeps unknown third-party subclasses working. */
        if (!p->s->internal_layout(d.internal_arrays, fc.gv, p->data))
          d.kind = SusceptibilityKind::host_custom;

        FOR_COMPONENTS(c) {
          DOCMP2 {
            if (p->s->needs_P(c, cmp, fc.f)) d.required_w |= (uint64_t(1) << int(c));
          }
          if (fc.needs_W_notowned(c)) d.needs_halo = true;
        }
        if (p->s->needs_W_prev()) d.required_w_prev = d.required_w;

        out.push_back(d);
      }
    }
  }
}

void refresh_operation_descriptors(fields &f, bool rebuild_all) {
  if (rebuild_all || is_dirty(f, dirty_source_plan))
    build_source_descriptors(f, f.descriptors->sources);
  if (rebuild_all || is_dirty(f, dirty_monitor_plan))
    build_dft_descriptors(f, f.descriptors->dfts);
  if (rebuild_all) build_polarization_descriptors(f, f.descriptors->polarizations);

  /* Region plans are produced for a particular public query/monitor volume by
     prepare_loop_in_chunks rather than from one global definition. The shared
     DescriptorSet therefore acts only as their cache: invalidation discards
     it, and the consumer rebuilds the requested regions on demand. */
  if (rebuild_all || is_dirty(f, dirty_regions)) f.descriptors->regions.clear();

  clear_dirty(f, dirty_source_plan | dirty_monitor_plan | dirty_regions);
}

} // namespace meep
