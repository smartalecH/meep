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
#include "backend/backend.hpp"
#include "backend/lifecycle.hpp"
#include "backend/random_state.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <typeinfo>

namespace meep {

static int legacy_flux_descriptor_failure_rank_for_testing = -1;
static int legacy_flux_descriptor_failure_ordinal_for_testing = -1;

void backend_set_legacy_flux_descriptor_failure_for_testing(int rank, int flux_ordinal) {
  legacy_flux_descriptor_failure_rank_for_testing = rank;
  legacy_flux_descriptor_failure_ordinal_for_testing = flux_ordinal;
}

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

class legacy_flux_descriptor_builder {
public:
  static direction normal(const flux_vol &flux) { return flux.d; }
  static const volume &region(const flux_vol &flux) { return flux.where; }
};

class susceptibility_descriptor_builder {
public:
  static LorentzianParameters lorentzian_parameters(const lorentzian_susceptibility &s) {
    LorentzianParameters p;
    p.omega_0 = s.omega_0;
    p.gamma = s.gamma;
    p.drude = s.no_omega_0_denominator;
    return p;
  }

  static double noise_amplitude(const noisy_lorentzian_susceptibility &s) {
    return s.noise_amp;
  }

  static GyrotropicParameters gyrotropic_parameters(const gyrotropic_susceptibility &s) {
    GyrotropicParameters p = {};
    p.omega_0 = s.omega_0;
    p.gamma = s.gamma;
    p.alpha = s.alpha;
    p.model = s.model;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) p.gyro_tensor[i][j] = s.gyro_tensor[i][j];
    return p;
  }

  static MultilevelParameters multilevel_parameters(const multilevel_susceptibility &s) {
    if (s.L <= 0 || s.T <= 0 || !s.Gamma || !s.N0 || !s.alpha || !s.omega || !s.gamma ||
        !s.sigmat)
      throw std::runtime_error("invalid multilevel susceptibility parameters");
    const size_t levels = size_t(s.L);
    const size_t transitions = size_t(s.T);
    if (levels > std::numeric_limits<size_t>::max() / levels ||
        levels > std::numeric_limits<size_t>::max() / transitions ||
        transitions > std::numeric_limits<size_t>::max() / 5)
      throw std::overflow_error("multilevel susceptibility parameter extent overflow");
    MultilevelParameters p = {};
    p.levels = uint32_t(levels);
    p.transitions = uint32_t(transitions);
    p.gamma_matrix.assign(s.Gamma, s.Gamma + levels * levels);
    p.initial_populations.assign(s.N0, s.N0 + levels);
    p.alpha.assign(s.alpha, s.alpha + levels * transitions);
    p.omega.assign(s.omega, s.omega + transitions);
    p.transition_gamma.assign(s.gamma, s.gamma + transitions);
    p.sigmat.assign(s.sigmat, s.sigmat + 5 * transitions);
    return p;
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
    if (times.size() > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("source-time descriptor index overflow");
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
        size_t source_ordinal = 0;
        for (const src_vol &sv : fc.get_sources(ft)) {
          if (source_ordinal > std::numeric_limits<uint32_t>::max())
            throw std::overflow_error("source descriptor ordinal overflow");
          const uint32_t ordinal = uint32_t(source_ordinal++);
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
          d.source_ordinal = ordinal;
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
    source_hash_mix(hash, d.source_ordinal);
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

namespace {

void dft_hash_ivec(uint64_t &hash, const ivec &v) {
  source_hash_mix(hash, uint64_t(v.dim));
  LOOP_OVER_DIRECTIONS(v.dim, d) { source_hash_mix(hash, uint64_t(v.in_direction(d))); }
}

void dft_hash_vec(uint64_t &hash, const vec &v) {
  source_hash_mix(hash, uint64_t(v.dim));
  LOOP_OVER_DIRECTIONS(v.dim, d) { source_hash_double(hash, v.in_direction(d)); }
}

void dft_hash_ref(uint64_t &hash, const ArrayRef &ref) {
  source_hash_mix(hash, ref.id.value);
  source_hash_mix(hash, ref.offset);
  source_hash_mix(hash, ref.elements);
}

} // namespace

uint64_t dft_plan_signature(const std::vector<DftDescriptor> &plan) {
  uint64_t hash = 0xcbf29ce484222325ull;
  source_hash_mix(hash, plan.size());
  for (const DftDescriptor &d : plan) {
    source_hash_mix(hash, d.accumulator.value);
    source_hash_mix(hash, d.phase_scratch.value);
    dft_hash_ref(hash, d.source_field);
    dft_hash_ref(hash, d.source_field_imag);
    source_hash_mix(hash, d.omega.size());
    for (double omega : d.omega)
      source_hash_double(hash, omega);
    source_hash_double(hash, d.scale.real());
    source_hash_double(hash, d.scale.imag());
    source_hash_mix(hash, uint64_t(d.chunk));
    source_hash_mix(hash, uint64_t(d.c));
    source_hash_mix(hash, uint64_t(d.avg1));
    source_hash_mix(hash, uint64_t(d.avg2));
    dft_hash_ivec(hash, d.is);
    dft_hash_ivec(hash, d.ie);
    dft_hash_ivec(hash, d.is_old);
    dft_hash_ivec(hash, d.ie_old);
    source_hash_mix(hash, d.persist);
    source_hash_mix(hash, uint64_t(d.decimation_factor));
    source_hash_mix(hash, d.due_scalar_slot);
    dft_hash_vec(hash, d.weights.s0);
    dft_hash_vec(hash, d.weights.s1);
    dft_hash_vec(hash, d.weights.e0);
    dft_hash_vec(hash, d.weights.e1);
    source_hash_double(hash, d.dV0);
    source_hash_double(hash, d.dV1);
    source_hash_mix(hash, d.include_dV_and_interp_weights);
    source_hash_mix(hash, d.sqrt_dV_and_interp_weights);
    source_hash_mix(hash, d.N);
    source_hash_mix(hash, d.Nomega);
  }
  return hash;
}

/* --- Legacy instantaneous flux monitors --------------------------------- */

namespace {

uint64_t legacy_flux_recipe_signature(direction normal, const volume &where) {
  uint64_t hash = 0xcbf29ce484222325ull;
  source_hash_mix(hash, uint64_t(normal));
  source_hash_mix(hash, uint64_t(where.dim));
  const vec lo = where.get_min_corner();
  const vec hi = where.get_max_corner();
  LOOP_OVER_DIRECTIONS(where.dim, d) {
    source_hash_double(hash, lo.in_direction(d));
    source_hash_double(hash, hi.in_direction(d));
  }
  return hash;
}

void legacy_flux_components(ndim dim, direction normal, component e[2], component h[2]) {
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
  throw std::invalid_argument("legacy flux descriptor has no normal direction");
}

ArrayId legacy_flux_field(fields &f, int chunk, component c, int cmp) {
  const ArrayId id = f.array_catalog->find({chunk, int(array_kind::f), int(c), cmp, 0});
  if (!is_valid(id)) return id;
  const ArraySpec &spec = f.array_catalog->spec(id);
  if (spec.role != array_role::field || spec.element_type != ElementType::realnum_value ||
      spec.elements != size_t(f.chunks[chunk]->gv.ntot()))
    throw std::runtime_error("legacy flux field has incompatible storage metadata");
  return id;
}

size_t legacy_flux_base(const grid_volume &gv, const ivec &begin,
                        size_t counts[3], ptrdiff_t strides[3]) {
  ptrdiff_t base = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const ptrdiff_t delta = begin.yucky_val(axis) - gv.little_corner().yucky_val(axis);
    /* Match LOOP_OVER_IVECS exactly: Centered regions can differ in parity
       from gv.little_corner(), and the legacy macro deliberately uses
       truncating integer division here. */
    if (delta < 0)
      throw std::runtime_error("legacy flux region begins before its chunk grid");
    const ptrdiff_t stride = gv.stride(gv.yucky_direction(axis));
    const ptrdiff_t coordinate = delta / 2;
    if (stride < 0 || (coordinate && stride > std::numeric_limits<ptrdiff_t>::max() / coordinate) ||
        base > std::numeric_limits<ptrdiff_t>::max() - coordinate * stride)
      throw std::overflow_error("legacy flux region base overflow");
    base += coordinate * stride;
    strides[axis] = stride;
  }
  if (base < 0) throw std::runtime_error("legacy flux region has a negative base index");
  return size_t(base);
}

} // namespace

uint64_t legacy_flux_definition_signature(const fields &f) {
  uint64_t hash = 0xcbf29ce484222325ull;
  uint64_t count = 0;
  for (const flux_vol *flux = f.fluxes; flux; flux = flux->next, ++count) {
    source_hash_mix(hash, count);
    source_hash_mix(hash, legacy_flux_recipe_signature(
                              legacy_flux_descriptor_builder::normal(*flux),
                              legacy_flux_descriptor_builder::region(*flux)));
  }
  source_hash_mix(hash, count);
  return hash;
}

void build_legacy_flux_descriptors(fields &f, std::vector<LegacyFluxDescriptor> &out) {
  std::string local_error;
  if (!f.array_catalog)
    local_error = "legacy flux descriptors require a prepared array catalog";
  backend_reconcile_host_access(local_error, "legacy flux descriptor catalog preflight");

  size_t local_definition = size_t(legacy_flux_definition_signature(f));
  size_t reference_definition = local_definition;
  broadcast(0, &reference_definition, 1);
  if (local_definition != reference_definition)
    local_error = "legacy flux definitions differ across MPI ranks";
  backend_reconcile_host_access(local_error, "legacy flux descriptor definition preflight");

  std::vector<LegacyFluxDescriptor> replacement;
  uint64_t flux_ordinal = 0;
  for (const flux_vol *flux = f.fluxes; flux; flux = flux->next, ++flux_ordinal) {
    local_error.clear();
    if (flux_ordinal > std::numeric_limits<uint32_t>::max())
      local_error = "legacy flux descriptor index overflow";
    backend_reconcile_host_access(local_error, "legacy flux descriptor monitor preflight");

    const direction normal = legacy_flux_descriptor_builder::normal(*flux);
    const volume &where = legacy_flux_descriptor_builder::region(*flux);
    component e[2], h[2];
    legacy_flux_components(f.gv.dim, normal, e, h);
    LegacyFluxDescriptor descriptor(uint32_t(flux_ordinal), normal, where);
    descriptor.recipe_signature = legacy_flux_recipe_signature(normal, where);
    for (uint32_t term_ordinal = 0; term_ordinal < 2; ++term_ordinal) {
      local_error.clear();
      try {
        /* Match fields::integrate exactly: when both operands share a Yee
           grid, integrate directly on that grid and leave interpolation
           offsets zero. Otherwise interpolate both operands to Centered. */
        const component cgrid = f.gv.iyee_shift(e[term_ordinal]) ==
                                        f.gv.iyee_shift(h[term_ordinal])
                                    ? e[term_ordinal]
                                    : Centered;
        const ChunkLoopPlan regions = prepare_loop_in_chunks(f, where, cgrid);
        for (size_t region_ordinal = 0; region_ordinal < regions.regions.size(); ++region_ordinal) {
          if (region_ordinal > std::numeric_limits<uint32_t>::max())
            throw std::overflow_error("legacy flux region index overflow");
          const ChunkLoopRegion &region = regions.regions[region_ordinal];
          if (region.chunk < 0 || region.chunk >= f.num_chunks || !f.chunks[region.chunk] ||
              !f.chunks[region.chunk]->is_mine())
            throw std::runtime_error("legacy flux region names a non-owned chunk");
          fields_chunk &fc = *f.chunks[region.chunk];

          LegacyFluxTermDescriptor term = {};
          term.term_ordinal = term_ordinal;
          term.region_ordinal = uint32_t(region_ordinal);
          term.sign = term_ordinal ? -1 : 1;
          term.chunk = region.chunk;
          term.e_component = f.S.transform(e[term_ordinal], -region.symmetry_index);
          term.h_component = f.S.transform(h[term_ordinal], -region.symmetry_index);
          term.e_real = legacy_flux_field(f, region.chunk, term.e_component, 0);
          term.e_imag = legacy_flux_field(f, region.chunk, term.e_component, 1);
          term.h_real = legacy_flux_field(f, region.chunk, term.h_component, 0);
          term.h_imag = legacy_flux_field(f, region.chunk, term.h_component, 1);
          if ((!is_valid(term.e_real) && is_valid(term.e_imag)) ||
              (!is_valid(term.h_real) && is_valid(term.h_imag)))
            throw std::runtime_error("legacy flux imaginary field has no real counterpart");
          term.begin = region.begin;
          term.end = region.end;
          term.lattice_shift = region.lattice_shift;
          term.symmetry_index = region.symmetry_index;
          for (int axis = 0; axis < 3; ++axis) {
            const ptrdiff_t span = region.end.yucky_val(axis) - region.begin.yucky_val(axis);
            if (span < 0 || span % 2)
              throw std::runtime_error("legacy flux region has an invalid extent");
            term.counts[axis] = size_t(span / 2 + 1);
          }
          term.base = legacy_flux_base(fc.gv, region.begin, term.counts, term.strides);
          if (cgrid == Centered) {
            fc.gv.yee2cent_offsets(term.e_component, term.e_offsets[0], term.e_offsets[1]);
            fc.gv.yee2cent_offsets(term.h_component, term.h_offsets[0], term.h_offsets[1]);
          }
          const std::complex<double> e_phase =
              region.phase * f.S.phase_shift(term.e_component, region.symmetry_index);
          const std::complex<double> h_phase =
              region.phase * f.S.phase_shift(term.h_component, region.symmetry_index);
          const std::complex<double> product_phase = std::conj(e_phase) * h_phase;
          term.phase_real = product_phase.real();
          term.phase_imag = product_phase.imag();
          for (int axis = 0; axis < 3; ++axis) {
            const direction d = fc.gv.yucky_direction(axis);
            term.boundary_weights[axis][0] = region.weights.s0.in_direction(d);
            term.boundary_weights[axis][1] = region.weights.s1.in_direction(d);
            term.boundary_weights[axis][2] = region.weights.e0.in_direction(d);
            term.boundary_weights[axis][3] = region.weights.e1.in_direction(d);
          }
          term.dV0 = region.dV0;
          term.dV1 = region.dV1;
          descriptor.terms.push_back(term);
        }
        if (legacy_flux_descriptor_failure_rank_for_testing == my_rank() &&
            legacy_flux_descriptor_failure_ordinal_for_testing == int(flux_ordinal) &&
            term_ordinal == 0)
          throw std::runtime_error("injected per-monitor legacy flux descriptor failure");
      }
      catch (const std::exception &e) {
        local_error = e.what();
      }
      catch (...) {
        local_error = "unknown legacy flux descriptor pair preparation failure";
      }
      backend_reconcile_host_access(local_error, "legacy flux descriptor pair preparation");
    }
    local_error.clear();
    try {
      replacement.push_back(descriptor);
    }
    catch (const std::exception &e) {
      local_error = e.what();
    }
    catch (...) {
      local_error = "unknown legacy flux descriptor monitor publication failure";
    }
    backend_reconcile_host_access(local_error, "legacy flux descriptor monitor publication");
  }
  out.swap(replacement);
}

void refresh_legacy_flux_descriptors(fields &f) {
  if (!f.descriptors) throw std::runtime_error("fields has no descriptor set");
  build_legacy_flux_descriptors(f, f.descriptors->legacy_fluxes);
  f.descriptors->legacy_flux_generation = generation(f, MutationKind::legacy_flux_definition);
}

/* --- Susceptibilities ----------------------------------------------------- */

SusceptibilityKind classify_susceptibility(const susceptibility *s) {
  if (typeid(*s) == typeid(noisy_lorentzian_susceptibility))
    return SusceptibilityKind::noisy_lorentzian;
  if (typeid(*s) == typeid(gyrotropic_susceptibility)) return SusceptibilityKind::gyrotropic;
  if (typeid(*s) == typeid(multilevel_susceptibility)) return SusceptibilityKind::multilevel;
  if (typeid(*s) == typeid(lorentzian_susceptibility)) return SusceptibilityKind::lorentzian;
  return SusceptibilityKind::host_custom;
}

namespace {

uint64_t polarization_component_bit(component c, int cmp) {
  static_assert(2 * NUM_FIELD_COMPONENTS <= 64,
                "polarization component/cmp mask does not fit in uint64_t");
  return uint64_t(1) << (2 * int(c) + cmp);
}

void *checked_internal_row_address(void *base, size_t offset_elements, size_t elements) {
  if (!base) throw std::runtime_error("published polarization layout has no backing state");
  if (elements > std::numeric_limits<size_t>::max() - offset_elements ||
      offset_elements + elements > std::numeric_limits<size_t>::max() / sizeof(realnum))
    throw std::overflow_error("published polarization layout byte offset overflow");
  const uintptr_t begin = reinterpret_cast<uintptr_t>(base);
  const size_t bytes = offset_elements * sizeof(realnum);
  if (bytes > std::numeric_limits<uintptr_t>::max() - begin)
    throw std::overflow_error("published polarization layout address overflow");
  return reinterpret_cast<void *>(begin + bytes);
}

void validate_internal_layout_overlap(const std::vector<InternalArrayLayout> &layout) {
  for (size_t i = 0; i < layout.size(); ++i) {
    const size_t i_scale = layout[i].element_type == InternalArrayLayout::complex_realnum ? 2 : 1;
    if (layout[i].elements > std::numeric_limits<size_t>::max() / i_scale ||
        layout[i].offset_elements >
            std::numeric_limits<size_t>::max() - layout[i].elements * i_scale)
      throw std::overflow_error("published polarization layout extent overflow");
    const size_t i_end = layout[i].offset_elements + layout[i].elements * i_scale;
    for (size_t j = 0; j < i; ++j) {
      const size_t j_scale =
          layout[j].element_type == InternalArrayLayout::complex_realnum ? 2 : 1;
      const size_t j_end = layout[j].offset_elements + layout[j].elements * j_scale;
      if (layout[i].offset_elements < j_end && layout[j].offset_elements < i_end)
        throw std::runtime_error("published polarization layout rows overlap");
    }
  }
}

void build_lorentzian_state_arrays(fields &f, fields_chunk &fc, polarization_state *state,
                                   PolarizationDescriptor &d) {
  const size_t ntot = size_t(fc.gv.ntot());
  if (!state->data) return;
  if (!f.array_catalog)
    throw std::runtime_error("Lorentzian descriptor requires a prepared array catalog");
  if (d.internal_arrays.size() % 2)
    throw std::runtime_error("Lorentzian internal layout has an unpaired state array");

  bool seen[NUM_FIELD_COMPONENTS][2] = {};
  realnum *base = static_cast<realnum *>(state->data);
  for (size_t i = 0; i < d.internal_arrays.size(); i += 2) {
    const InternalArrayLayout &p = d.internal_arrays[i];
    const InternalArrayLayout &p_prev = d.internal_arrays[i + 1];
    if (!p.name || !p_prev.name || strcmp(p.name, "P") || strcmp(p_prev.name, "P_prev") ||
        p.c != p_prev.c || int(p.c) < 0 || int(p.c) >= NUM_FIELD_COMPONENTS ||
        p.cmp != p_prev.cmp || p.cmp < 0 || p.cmp > 1 || p.elements != ntot ||
        p_prev.elements != ntot || p.element_type != InternalArrayLayout::realnum_value ||
        p_prev.element_type != InternalArrayLayout::realnum_value ||
        p_prev.offset_elements != p.offset_elements + ntot || seen[int(p.c)][p.cmp])
      throw std::runtime_error("invalid Lorentzian P/P_prev internal layout");

    ArrayId p_id = invalid_array(), p_prev_id = invalid_array();
    ptrdiff_t p_offset = 0, p_prev_offset = 0;
    if (!f.array_catalog->locate(base + p.offset_elements, p_id, p_offset) ||
        !f.array_catalog->locate(base + p_prev.offset_elements, p_prev_id, p_prev_offset) ||
        p_offset != 0 || p_prev_offset != 0 || !is_valid(p_id) || !is_valid(p_prev_id) ||
        p_id == p_prev_id)
      throw std::runtime_error("Lorentzian state arrays do not resolve to stable ArrayIds");

    const ArraySpec &p_spec = f.array_catalog->spec(p_id);
    const ArraySpec &p_prev_spec = f.array_catalog->spec(p_prev_id);
    if (p_spec.role != array_role::polarization || p_prev_spec.role != array_role::polarization ||
        p_spec.element_type != ElementType::realnum_value ||
        p_prev_spec.element_type != ElementType::realnum_value || p_spec.elements != ntot ||
        p_prev_spec.elements != ntot)
      throw std::runtime_error("Lorentzian state ArrayIds have incompatible storage metadata");

    LorentzianStateArrays arrays;
    arrays.c = p.c;
    arrays.cmp = p.cmp;
    arrays.p = p_id;
    arrays.p_prev = p_prev_id;
    arrays.elements = ntot;
    d.lorentzian_states.push_back(arrays);
    seen[int(p.c)][p.cmp] = true;
    d.required_w |= polarization_component_bit(p.c, p.cmp);
  }

  /* Polarization state is allocated from the field layout that existed at
     creation time and is not expanded when a later require_component() grows
     fc.f.  The published internal layout is therefore authoritative: a fresh
     needs_P() query may be a strict superset, but it must never reject a state
     row that actually exists. */
  FOR_COMPONENTS(c) DOCMP2 {
    if (seen[int(c)][cmp] && !state->s->needs_P(c, cmp, fc.f))
      throw std::runtime_error("Lorentzian state layout contains an unexpected component");
  }
}

void build_gyrotropic_state_arrays(fields &f, fields_chunk &fc, polarization_state *state,
                                   PolarizationDescriptor &d) {
  const size_t ntot = size_t(fc.gv.ntot());
  if (!state->data) return;
  if (!f.array_catalog)
    throw std::runtime_error("gyrotropic descriptor requires a prepared array catalog");
  if (d.internal_arrays.size() % 6)
    throw std::runtime_error("gyrotropic internal layout has an incomplete state row");

  static const char *p_names[3] = {"P_x", "P_y", "P_z"};
  static const char *p_prev_names[3] = {"P_prev_x", "P_prev_y", "P_prev_z"};
  bool seen[NUM_FIELD_COMPONENTS][2] = {};
  realnum *base = static_cast<realnum *>(state->data);
  for (size_t i = 0; i < d.internal_arrays.size(); i += 6) {
    const InternalArrayLayout &first = d.internal_arrays[i];
    if (int(first.c) < 0 || int(first.c) >= NUM_FIELD_COMPONENTS || first.cmp < 0 ||
        first.cmp > 1 || seen[int(first.c)][first.cmp])
      throw std::runtime_error("invalid gyrotropic state row");

    GyrotropicStateArrays arrays = {};
    arrays.c = first.c;
    arrays.cmp = first.cmp;
    arrays.elements = ntot;
    ArrayId ids[6];
    for (int dd = 0; dd < 3; ++dd) {
      const InternalArrayLayout &p = d.internal_arrays[i + 2 * dd];
      const InternalArrayLayout &p_prev = d.internal_arrays[i + 2 * dd + 1];
      if (!p.name || !p_prev.name || strcmp(p.name, p_names[dd]) ||
          strcmp(p_prev.name, p_prev_names[dd]) || p.c != first.c || p_prev.c != first.c ||
          p.cmp != first.cmp || p_prev.cmp != first.cmp || p.elements != ntot ||
          p_prev.elements != ntot || p.element_type != InternalArrayLayout::realnum_value ||
          p_prev.element_type != InternalArrayLayout::realnum_value ||
          p.offset_elements != first.offset_elements + size_t(2 * dd) * ntot ||
          p_prev.offset_elements != p.offset_elements + ntot)
        throw std::runtime_error("invalid gyrotropic P/P_prev internal layout");

      ptrdiff_t p_offset = 0, p_prev_offset = 0;
      const bool found_p =
          f.array_catalog->locate(base + p.offset_elements, ids[2 * dd], p_offset);
      const bool found_p_prev = f.array_catalog->locate(
          base + p_prev.offset_elements, ids[2 * dd + 1], p_prev_offset);
      if (!found_p || !found_p_prev || p_offset != 0 || p_prev_offset != 0 ||
          !is_valid(ids[2 * dd]) || !is_valid(ids[2 * dd + 1]))
        throw std::runtime_error("gyrotropic state arrays do not resolve to stable ArrayIds at " +
                                 std::to_string(dd));
      arrays.p[dd] = ids[2 * dd];
      arrays.p_prev[dd] = ids[2 * dd + 1];
    }
    for (int a = 0; a < 6; ++a)
      for (int b = a + 1; b < 6; ++b)
        if (ids[a] == ids[b])
          throw std::runtime_error("gyrotropic state arrays alias one another");
    for (int a = 0; a < 6; ++a) {
      const ArraySpec &spec = f.array_catalog->spec(ids[a]);
      if (spec.role != array_role::polarization ||
          spec.element_type != ElementType::realnum_value || spec.elements != ntot)
        throw std::runtime_error("gyrotropic state ArrayIds have incompatible storage metadata");
    }

    d.gyrotropic_states.push_back(arrays);
    seen[int(first.c)][first.cmp] = true;
    d.required_w |= polarization_component_bit(first.c, first.cmp);
    const direction d0 = component_direction(first.c);
    const component c1 = direction_component(first.c, cycle_direction(fc.gv.dim, d0, 1));
    const component c2 = direction_component(first.c, cycle_direction(fc.gv.dim, d0, 2));
    if (fc.f[c1][first.cmp]) d.required_w |= polarization_component_bit(c1, first.cmp);
    if (fc.f[c2][first.cmp]) d.required_w |= polarization_component_bit(c2, first.cmp);
  }

  FOR_COMPONENTS(c) DOCMP2 {
    if (seen[int(c)][cmp] && !state->s->needs_P(c, cmp, fc.f))
      throw std::runtime_error("gyrotropic state layout contains an unexpected component");
  }
}

void build_multilevel_state_arrays(fields &f, fields_chunk &fc, polarization_state *state,
                                   PolarizationDescriptor &d) {
  if (!state->data) return;
  if (!f.array_catalog)
    throw std::runtime_error("multilevel descriptor requires a prepared catalog");
  const size_t levels = d.multilevel.levels;
  const size_t transitions = d.multilevel.transitions;
  const size_t ntot = size_t(fc.gv.ntot());
  if (levels > std::numeric_limits<size_t>::max() / levels ||
      ntot > std::numeric_limits<size_t>::max() / levels)
    throw std::overflow_error("multilevel descriptor extent overflow");
  if (d.internal_arrays.size() < 2 ||
      (d.internal_arrays.size() - 2) % (2 * transitions) != 0)
    throw std::runtime_error("multilevel internal layout has an incomplete transition row");

  realnum *const base = static_cast<realnum *>(state->data);
  std::set<uint32_t> ids;
  auto resolve = [&](const InternalArrayLayout &entry, size_t ordinal, size_t elements,
                     const char *name) {
    if (!entry.name || strcmp(entry.name, name) ||
        entry.element_type != InternalArrayLayout::realnum_value || entry.elements != elements)
      throw std::runtime_error("multilevel internal layout row has incompatible metadata");
    ArrayId id = invalid_array();
    ptrdiff_t offset = 0;
    if (!f.array_catalog->locate(base + entry.offset_elements, id, offset) || offset != 0 ||
        !is_valid(id) || id.value >= f.array_catalog->size() ||
        !ids.insert(id.value).second)
      throw std::runtime_error("multilevel internal row lacks a distinct canonical ArrayId");
    const ArraySpec &spec = f.array_catalog->spec(id);
    const StorageKey &key = f.array_catalog->key(id);
    if (is_valid(spec.alias_of) || spec.role != array_role::polarization ||
        spec.element_type != ElementType::realnum_value || spec.elements != elements ||
        key.chunk != d.chunk || key.kind != int(array_kind::polarization_internal) ||
        key.component_ != int(entry.c) || key.cmp != entry.cmp ||
        key.aux != polarization_storage_aux(d.ft, d.state_index, ordinal))
      throw std::runtime_error("multilevel state ArrayId has incompatible storage metadata");
    return id;
  };

  const InternalArrayLayout &gamma = d.internal_arrays.front();
  if (gamma.c != Centered || gamma.cmp != -1)
    throw std::runtime_error("multilevel GammaInv layout identity is invalid");
  d.multilevel_gamma_inv = resolve(gamma, 0, levels * levels, "GammaInv");
  const realnum *gamma_inv = f.array_catalog->resolve<realnum>(d.multilevel_gamma_inv);
  for (size_t i = 0; i < levels * levels; ++i)
    if (!std::isfinite(double(gamma_inv[i])))
      throw std::runtime_error("multilevel GammaInv contains a nonfinite value");

  bool seen[NUM_FIELD_COMPONENTS][2] = {};
  int previous_component = -1;
  int previous_cmp = -1;
  size_t expected_offset = gamma.offset_elements + levels * levels;
  size_t row = 1;
  while (row + 1 < d.internal_arrays.size()) {
    const component c = d.internal_arrays[row].c;
    const int cmp = d.internal_arrays[row].cmp;
    const int component_index = int(c);
    if (component_index < 0 || component_index >= NUM_FIELD_COMPONENTS || c == Centered ||
        type(c) != d.ft || cmp < 0 || cmp > 1 || seen[component_index][cmp] ||
        component_index < previous_component ||
        (component_index == previous_component && cmp <= previous_cmp) ||
        (cmp == 1 && !seen[int(c)][0]))
      throw std::runtime_error("multilevel transition layout is not component/cmp-major");
    for (size_t transition = 0; transition < transitions; ++transition) {
      const InternalArrayLayout &p = d.internal_arrays[row++];
      const InternalArrayLayout &p_prev = d.internal_arrays[row++];
      if (p.c != c || p.cmp != cmp || p_prev.c != c || p_prev.cmp != cmp ||
          p.offset_elements != expected_offset ||
          p_prev.offset_elements != p.offset_elements + ntot)
        throw std::runtime_error("multilevel P/P_prev layout order is invalid");
      MultilevelStateArrays arrays;
      arrays.transition_index = int(transition);
      arrays.c = c;
      arrays.cmp = cmp;
      arrays.p = resolve(p, row - 2, ntot, "P");
      arrays.p_prev = resolve(p_prev, row - 1, ntot, "P_prev");
      arrays.elements = ntot;
      d.multilevel_states.push_back(arrays);
      expected_offset += 2 * ntot;
    }
    seen[component_index][cmp] = true;
    previous_component = component_index;
    previous_cmp = cmp;
    d.required_w |= polarization_component_bit(c, cmp);
  }

  const InternalArrayLayout &populations = d.internal_arrays.back();
  if (populations.c != Centered || populations.cmp != -1 ||
      populations.offset_elements != expected_offset + levels)
    throw std::runtime_error("multilevel N layout does not exclude exactly its Ntmp gap");
  d.multilevel_populations =
      resolve(populations, d.internal_arrays.size() - 1, ntot * levels, "N");
  d.multilevel_population_points = ntot;

  FOR_COMPONENTS(c) DOCMP2 {
    const int rows = state->s->num_cinternal_notowned_needed(c, state->data);
    const bool expected = rows > 0 &&
                          state->s->cinternal_notowned_ptr(0, c, cmp, 0, state->data) != NULL;
    if (seen[int(c)][cmp] != expected)
      throw std::runtime_error("multilevel state layout omits or invents a live component");
  }
  d.per_thread_scratch_elements = levels;
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
        d.susceptibility_id = p->s->get_id();
        d.has_internal_state = p->data != NULL;
        d.layout_published = false;
        d.kind = classify_susceptibility(p->s);
        d.lorentzian = LorentzianParameters{0.0, 0.0, false};
        d.noise_amplitude = 0.0;
        d.noise_algorithm_version = 0;
        d.gyrotropic = GyrotropicParameters{};
        d.multilevel = MultilevelParameters{};
        d.multilevel_gamma_inv = invalid_array();
        d.multilevel_populations = invalid_array();
        d.multilevel_population_points = 0;
        d.per_thread_scratch_elements = 0;
        d.required_w = 0;
        d.required_w_prev = 0;
        d.needs_halo = false;

        /* A susceptibility that does not publish a layout classifies as
           host_custom regardless of what it is -- that is the escape hatch
           that keeps unknown third-party subclasses working. */
        d.layout_published = p->s->internal_layout(d.internal_arrays, fc.gv, p->data);
        if (!d.layout_published) {
          d.kind = SusceptibilityKind::host_custom;
          d.internal_arrays.clear();
        }

        if (d.layout_published) {
          if (!f.array_catalog)
            throw std::runtime_error("published polarization layout requires a prepared catalog");
          if (!p->data && !d.internal_arrays.empty())
            throw std::runtime_error("stateless polarization published nonempty internal layout");
          validate_internal_layout_overlap(d.internal_arrays);
          std::set<uint32_t> ids;
          for (size_t li = 0; li < d.internal_arrays.size(); ++li) {
            const InternalArrayLayout &entry = d.internal_arrays[li];
            if (!entry.name || !entry.elements)
              throw std::runtime_error("published polarization layout has an invalid row");
            void *const address =
                checked_internal_row_address(
                    p->data, entry.offset_elements,
                    entry.elements *
                        (entry.element_type == InternalArrayLayout::complex_realnum ? 2 : 1));
            ArrayId id = invalid_array();
            ptrdiff_t offset = 0;
            if (!f.array_catalog->locate(address, id, offset) || offset != 0 ||
                !is_valid(id) || id.value >= f.array_catalog->size() ||
                !ids.insert(id.value).second)
              throw std::runtime_error("published polarization row lacks a distinct ArrayId");
            const ArraySpec &spec = f.array_catalog->spec(id);
            const StorageKey &key = f.array_catalog->key(id);
            const Precision native_precision =
                sizeof(realnum) == sizeof(float) ? Precision::f32 : Precision::f64;
            const ElementType type = entry.element_type == InternalArrayLayout::complex_realnum
                                         ? ElementType::complex_realnum
                                         : ElementType::realnum_value;
            if (key.chunk != i || key.kind != int(array_kind::polarization_internal) ||
                key.component_ != int(entry.c) || key.cmp != entry.cmp ||
                key.aux != polarization_storage_aux(ft, si, li) ||
                spec.role != array_role::polarization || spec.element_type != type ||
                spec.storage != native_precision || spec.alignment != alignof(realnum) ||
                spec.elements != entry.elements || is_valid(spec.alias_of) ||
                f.array_catalog->resolve_untyped(id) != address)
              throw std::runtime_error("published polarization row has incompatible storage");
            d.internal_array_refs.push_back(ArrayRef{id, 0, entry.elements});
          }
        }

        if (d.kind == SusceptibilityKind::lorentzian ||
            d.kind == SusceptibilityKind::noisy_lorentzian) {
          const lorentzian_susceptibility &lorentz =
              static_cast<const lorentzian_susceptibility &>(*p->s);
          d.lorentzian = susceptibility_descriptor_builder::lorentzian_parameters(lorentz);
          if (d.kind == SusceptibilityKind::noisy_lorentzian) {
            const noisy_lorentzian_susceptibility &noisy =
                static_cast<const noisy_lorentzian_susceptibility &>(*p->s);
            d.noise_amplitude = susceptibility_descriptor_builder::noise_amplitude(noisy);
            d.noise_algorithm_version = counter_random_algorithm_version;
          }
          build_lorentzian_state_arrays(f, fc, p, d);
        }
        else if (d.kind == SusceptibilityKind::gyrotropic) {
          const gyrotropic_susceptibility &gyro =
              static_cast<const gyrotropic_susceptibility &>(*p->s);
          d.gyrotropic = susceptibility_descriptor_builder::gyrotropic_parameters(gyro);
          build_gyrotropic_state_arrays(f, fc, p, d);
        }
        else if (d.kind == SusceptibilityKind::multilevel) {
          const multilevel_susceptibility &multilevel =
              static_cast<const multilevel_susceptibility &>(*p->s);
          d.multilevel = susceptibility_descriptor_builder::multilevel_parameters(multilevel);
          const std::vector<double> *vectors[] = {
              &d.multilevel.gamma_matrix,       &d.multilevel.initial_populations,
              &d.multilevel.alpha,              &d.multilevel.omega,
              &d.multilevel.transition_gamma,   &d.multilevel.sigmat};
          for (const std::vector<double> *values : vectors)
            for (double value : *values)
              if (!std::isfinite(value))
                throw std::runtime_error("multilevel descriptor contains a nonfinite parameter");
          for (size_t transition = 0; transition < d.multilevel.transitions; ++transition) {
            bool positive = false, negative = false;
            for (size_t level = 0; level < d.multilevel.levels; ++level) {
              const double value =
                  d.multilevel.alpha[level * d.multilevel.transitions + transition];
              positive = positive || value > 0.0;
              negative = negative || value < 0.0;
            }
            if (!positive || !negative)
              throw std::runtime_error("multilevel alpha lacks a positive or negative level");
          }
          build_multilevel_state_arrays(f, fc, p, d);
        }

        FOR_COMPONENTS(c) {
          /* Exact built-in Lorentzian descriptors bind the state arrays that
             exist, rather than a later grow-only field-layout snapshot. */
          if (d.kind != SusceptibilityKind::lorentzian &&
              d.kind != SusceptibilityKind::noisy_lorentzian &&
              d.kind != SusceptibilityKind::gyrotropic &&
              d.kind != SusceptibilityKind::multilevel) {
            DOCMP2 {
              if (p->s->needs_P(c, cmp, fc.f))
                d.required_w |= polarization_component_bit(c, cmp);
            }
          }
          if (p->s->needs_W_notowned(c, fc.f)) d.needs_halo = true;
        }
        if (p->s->needs_W_prev()) d.required_w_prev = d.required_w;

        out.push_back(d);
      }
    }
  }
}

bool operator==(const PublishedInternalLayout &a, const PublishedInternalLayout &b) {
  return a.name == b.name && a.element_type == b.element_type &&
         a.offset_elements == b.offset_elements && a.elements == b.elements && a.c == b.c &&
         a.cmp == b.cmp;
}

bool operator==(const HostCallbackDescriptor &a, const HostCallbackDescriptor &b) {
  if (a.chunk != b.chunk || a.ft != b.ft || a.state_index != b.state_index ||
      a.susceptibility_id != b.susceptibility_id ||
      a.has_internal_state != b.has_internal_state || a.layout_published != b.layout_published ||
      a.required_w != b.required_w || a.required_w_prev != b.required_w_prev ||
      a.needs_halo != b.needs_halo || a.published_layout != b.published_layout ||
      a.published_internal_refs.size() != b.published_internal_refs.size())
    return false;
  for (size_t i = 0; i < a.published_internal_refs.size(); ++i) {
    const ArrayRef &x = a.published_internal_refs[i];
    const ArrayRef &y = b.published_internal_refs[i];
    if (x.id != y.id || x.offset != y.offset || x.elements != y.elements) return false;
  }
  return true;
}

HostCallbackDescriptor make_host_callback_descriptor(const PolarizationDescriptor &d) {
  if (d.kind != SusceptibilityKind::host_custom)
    throw std::invalid_argument("host callback descriptor requires host_custom polarization");
  HostCallbackDescriptor result;
  result.chunk = d.chunk;
  result.ft = d.ft;
  result.state_index = d.state_index;
  result.susceptibility_id = d.susceptibility_id;
  result.has_internal_state = d.has_internal_state;
  result.layout_published = d.layout_published;
  result.required_w = d.required_w;
  result.required_w_prev = d.required_w_prev;
  result.needs_halo = d.needs_halo;
  result.published_internal_refs = d.internal_array_refs;
  for (const InternalArrayLayout &entry : d.internal_arrays) {
    PublishedInternalLayout row;
    row.name = entry.name ? entry.name : "";
    row.element_type = entry.element_type;
    row.offset_elements = entry.offset_elements;
    row.elements = entry.elements;
    row.c = entry.c;
    row.cmp = entry.cmp;
    result.published_layout.push_back(row);
  }
  return result;
}

bool resolve_host_callback(fields &f, const HostCallbackDescriptor &descriptor,
                           ResolvedHostCallback &resolved, std::string *error) {
  resolved = ResolvedHostCallback();
  if (error) error->clear();
  try {
    if (descriptor.chunk < 0 || descriptor.chunk >= f.num_chunks ||
        (descriptor.ft != E_stuff && descriptor.ft != H_stuff))
      throw std::invalid_argument("host callback has invalid chunk or field type");
    fields_chunk *fc = f.chunks[descriptor.chunk];
    if (!fc->is_mine()) throw std::invalid_argument("host callback refers to an unowned chunk");
    if (descriptor.state_index < 0)
      throw std::invalid_argument("host callback has negative state index");
    polarization_state *state = fc->pol[descriptor.ft];
    for (int i = 0; state && i < descriptor.state_index; ++i) state = state->next;
    if (!state) throw std::invalid_argument("host callback state index is absent");
    if (state->s->get_id() != descriptor.susceptibility_id ||
        classify_susceptibility(state->s) != SusceptibilityKind::host_custom ||
        (state->data != NULL) != descriptor.has_internal_state)
      throw std::invalid_argument("host callback live identity changed");

    std::vector<InternalArrayLayout> layout;
    const bool published = state->s->internal_layout(layout, fc->gv, state->data);
    if (published != descriptor.layout_published || layout.size() != descriptor.published_layout.size())
      throw std::invalid_argument("host callback published layout changed");
    if (descriptor.published_internal_refs.size() != layout.size())
      throw std::invalid_argument("host callback layout/ref count differs");
    if (published && !f.array_catalog)
      throw std::invalid_argument("host callback published layout has no catalog");
    validate_internal_layout_overlap(layout);
    std::set<uint32_t> ids;
    for (size_t li = 0; li < layout.size(); ++li) {
      const InternalArrayLayout &live = layout[li];
      PublishedInternalLayout value;
      value.name = live.name ? live.name : "";
      value.element_type = live.element_type;
      value.offset_elements = live.offset_elements;
      value.elements = live.elements;
      value.c = live.c;
      value.cmp = live.cmp;
      if (value != descriptor.published_layout[li])
        throw std::invalid_argument("host callback published layout row changed");
      void *const address =
          checked_internal_row_address(
              state->data, live.offset_elements,
              live.elements *
                  (live.element_type == InternalArrayLayout::complex_realnum ? 2 : 1));
      const ArrayRef &ref = descriptor.published_internal_refs[li];
      if (!is_valid(ref.id) || ref.id.value >= f.array_catalog->size() || ref.offset != 0 ||
          ref.elements != live.elements || !ids.insert(ref.id.value).second)
        throw std::invalid_argument("host callback internal reference is invalid");
      const StorageKey &key = f.array_catalog->key(ref.id);
      const ArraySpec &spec = f.array_catalog->spec(ref.id);
      const Precision native_precision =
          sizeof(realnum) == sizeof(float) ? Precision::f32 : Precision::f64;
      const ElementType type = live.element_type == InternalArrayLayout::complex_realnum
                                   ? ElementType::complex_realnum
                                   : ElementType::realnum_value;
      if (key.chunk != descriptor.chunk || key.kind != int(array_kind::polarization_internal) ||
          key.component_ != int(live.c) || key.cmp != live.cmp ||
          key.aux != polarization_storage_aux(descriptor.ft, descriptor.state_index, li) ||
          spec.role != array_role::polarization || spec.element_type != type ||
          spec.storage != native_precision || spec.alignment != alignof(realnum) ||
          spec.elements != live.elements || is_valid(spec.alias_of) ||
          f.array_catalog->resolve_untyped(ref.id) != address)
        throw std::invalid_argument("host callback internal reference is stale");
    }

    uint64_t required_w = 0;
    bool needs_halo = false;
    FOR_COMPONENTS(c) {
      DOCMP2 if (state->s->needs_P(c, cmp, fc->f))
        required_w |= polarization_component_bit(c, cmp);
      if (state->s->needs_W_notowned(c, fc->f)) needs_halo = true;
    }
    const uint64_t required_w_prev = state->s->needs_W_prev() ? required_w : 0;
    if (required_w != descriptor.required_w || required_w_prev != descriptor.required_w_prev ||
        needs_halo != descriptor.needs_halo)
      throw std::invalid_argument("host callback field requirements changed");
    resolved.chunk = fc;
    resolved.state = state;
    return true;
  }
  catch (const std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

void refresh_operation_descriptors(fields &f, bool rebuild_all) {
  if (rebuild_all || is_dirty(f, dirty_source_plan))
    build_source_descriptors(f, f.descriptors->sources);
  if (rebuild_all || is_dirty(f, dirty_monitor_plan))
    build_dft_descriptors(f, f.descriptors->dfts);
  if (rebuild_all) build_polarization_descriptors(f, f.descriptors->polarizations);

  const bool cpu_flux_refresh =
      !rebuild_all && is_dirty(f, dirty_flux_plan) && !backend_host_refresh_required(f);
  const bool resident_flux_refresh = is_dirty(f, dirty_flux_plan) && backend_host_refresh_required(f);
  if (cpu_flux_refresh) refresh_legacy_flux_descriptors(f);

  /* Region plans are produced for a particular public query/monitor volume by
     prepare_loop_in_chunks rather than from one global definition. The shared
     DescriptorSet therefore acts only as their cache: invalidation discards
     it, and the consumer rebuilds the requested regions on demand. */
  if ((rebuild_all || is_dirty(f, dirty_regions)) && !resident_flux_refresh)
    f.descriptors->regions.clear();

  DirtyMask completed = DirtyMask(dirty_source_plan | dirty_monitor_plan);
  if (cpu_flux_refresh) completed |= dirty_flux_plan;
  /* A legacy-flux refresh owns its collective region planning and clears the
     shared region bit only when the staged descriptor/executable transaction
     commits. Preserve it here so a compile failure leaves the full closure
     retryable. */
  if (cpu_flux_refresh || !is_dirty(f, dirty_flux_plan)) completed |= dirty_regions;
  clear_dirty(f, completed);
}

} // namespace meep
