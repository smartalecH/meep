/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/checkpoint.hpp"

#include "backend/backend.hpp"
#include "backend/descriptors.hpp"
#include "backend/lifecycle.hpp"
#include "backend/initialization_plan.hpp"
#include "backend/material_ir.hpp"
#include "meep_internals.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <unordered_map>

namespace meep {

namespace {

CheckpointFailurePoint failure_point = CheckpointFailurePoint::none;
int failure_rank = -1;
CheckpointFailurePoint secondary_failure_point = CheckpointFailurePoint::none;
int secondary_failure_rank = -1;

void mix(uint64_t &hash, uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= uint8_t(value >> (8 * byte));
    hash *= UINT64_C(0x00000100000001b3);
  }
}

uint64_t encode_signed(int value) {
  return uint64_t(int64_t(value) - int64_t(std::numeric_limits<int>::min()));
}

int decode_signed(uint64_t value) {
  const uint64_t encoded_max = uint64_t(std::numeric_limits<unsigned int>::max());
  if (value > encoded_max)
    throw std::invalid_argument("checkpoint signed integer is outside the encoded int domain");
  return int(int64_t(value) + int64_t(std::numeric_limits<int>::min()));
}

uint64_t disk_hash(uint64_t value) { return value & ((UINT64_C(1) << 52) - 1); }

void mix_double(uint64_t &hash, double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "checkpoint double must be 64-bit");
  memcpy(&bits, &value, sizeof(bits));
  mix(hash, bits);
}

void mix_portable_real(uint64_t &hash, double value, const char *what) {
  if (std::isnan(value)) {
    mix(hash, UINT32_C(0x7fc00000));
    return;
  }
  float canonical = float(value);
  if (std::isfinite(value) && !std::isfinite(double(canonical)))
    throw std::invalid_argument(std::string(what) + " is outside binary32 range");
  if (canonical == 0.0f) canonical = 0.0f;
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(canonical), "binary32 must be 32 bits");
  memcpy(&bits, &canonical, sizeof(bits));
  mix(hash, bits);
}

void mix_native_real(uint64_t &hash, double value, const char *what) {
  if (sizeof(realnum) == sizeof(float)) {
    mix_portable_real(hash, value, what);
    return;
  }
  if (std::isnan(value)) {
    mix(hash, UINT64_C(0x7ff8000000000000));
    return;
  }
  if (value == 0.0) value = 0.0;
  mix_double(hash, value);
}

struct RecipeFingerprints {
  uint64_t portable;
  uint64_t native;
};

size_t checkpoint_precision_alignment(Precision precision) {
  switch (precision) {
    case Precision::f32: return alignof(float);
    case Precision::f64: return alignof(double);
  }
  throw std::invalid_argument("checkpoint row has an unknown precision");
}

realnum checked_checkpoint_scalar(double value, Precision source_precision) {
  /* Persistent host rows historically include IEEE sentinels in inactive
     lanes.  Preserve infinities and canonicalize every NaN to the target
     quiet-NaN; payload/sign bits are not continuation semantics. */
  if (std::isnan(value)) return std::numeric_limits<realnum>::quiet_NaN();
  if (std::isinf(value))
    return value < 0 ? -std::numeric_limits<realnum>::infinity()
                     : std::numeric_limits<realnum>::infinity();
  if (source_precision == Precision::f32) {
    const float source = float(value);
    if (!std::isfinite(double(source)) || double(source) != value)
      throw std::invalid_argument(
          "checkpoint f32 payload is not canonically representable as binary32");
  }
  else if (source_precision != Precision::f64)
    throw std::invalid_argument("checkpoint payload has an unknown source precision");

  if (sizeof(realnum) == sizeof(float)) {
    if (value > double(std::numeric_limits<float>::max()) ||
        value < -double(std::numeric_limits<float>::max()))
      throw std::invalid_argument("checkpoint f64 payload is outside binary32 range");
    const float converted = float(value);
    if (!std::isfinite(double(converted)))
      throw std::invalid_argument("checkpoint precision conversion produced a nonfinite scalar");
    return realnum(converted);
  }
  return realnum(value);
}

uint64_t checkpoint_configuration_signature(const fields &owner) {
  uint64_t hash = UINT64_C(0xcbf29ce484222325);
  mix(hash, uint64_t(owner.gv.dim));
  mix_double(hash, owner.a);
  mix_double(hash, owner.dt);
  mix_double(hash, owner.m);
  mix_double(hash, owner.beta);
  const direction axes[3] = {owner.gv.dim == Dcyl ? R : X, Y, Z};
  const ivec lo = owner.gv.little_corner(), hi = owner.gv.big_corner();
  const ivec user_lo = owner.user_volume.little_corner();
  const ivec user_hi = owner.user_volume.big_corner();
  for (int axis = 0; axis < 3; ++axis) {
    const direction d = axes[axis];
    mix(hash, has_direction(owner.gv.dim, d) ? uint64_t(lo.in_direction(d)) : 0);
    mix(hash, has_direction(owner.gv.dim, d) ? uint64_t(hi.in_direction(d)) : 0);
    mix(hash, has_direction(owner.gv.dim, d) ? uint64_t(user_lo.in_direction(d)) : 0);
    mix(hash, has_direction(owner.gv.dim, d) ? uint64_t(user_hi.in_direction(d)) : 0);
  }
  for (double value : owner.bfast_scaled_k) mix_double(hash, value);
  for (int d = 0; d < 5; ++d) {
    mix_double(hash, owner.k[d].real());
    mix_double(hash, owner.k[d].imag());
    mix(hash, uint64_t(owner.boundaries[Low][d]));
    mix(hash, uint64_t(owner.boundaries[High][d]));
  }
  mix(hash, uint64_t(owner.S.multiplicity()));
  for (int n = 0; n < owner.S.multiplicity(); ++n) {
    const ivec transformed = owner.S.transform(zero_ivec(owner.gv.dim), n);
    for (int axis = 0; axis < 3; ++axis) {
      const direction d = axes[axis];
      mix(hash, has_direction(owner.gv.dim, d) ? uint64_t(transformed.in_direction(d)) : 0);
    }
    for (int c = 0; c < NUM_FIELD_COMPONENTS; ++c) {
      mix(hash, uint64_t(owner.S.transform(component(c), n)));
      const std::complex<double> phase = owner.S.phase_shift(component(c), n);
      mix_double(hash, phase.real());
      mix_double(hash, phase.imag());
    }
  }
  return disk_hash(hash);
}

uint64_t checkpoint_source_definition_signature(fields &owner) {
  SourcePlan plan;
  build_source_descriptors(owner, plan);
  /* Source-time definitions are replicated; include them once. Spatial source
     samples are reduced as an unordered set keyed by global Yee coordinate,
     so processor ownership and chunk partition do not affect the result. */
  std::unordered_map<uint32_t, uint64_t> source_time_signatures;
  std::vector<uint64_t> definition_signatures;
  for (const SourceTimeDescriptor &definition : plan.source_times) {
    uint64_t token = UINT64_C(0xcbf29ce484222325);
    mix(token, uint64_t(definition.kind));
    mix(token, definition.is_integrated ? 1 : 0);
    mix(token, definition.parameters.size());
    for (double parameter : definition.parameters) mix_double(token, parameter);
    token = disk_hash(token);
    if (!source_time_signatures.emplace(definition.source_time_id, token).second)
      throw std::invalid_argument("checkpoint source-time id is not unique");
    definition_signatures.push_back(token);
  }
  std::sort(definition_signatures.begin(), definition_signatures.end());
  uint64_t signature = 0;
  if (am_master()) {
    uint64_t token = UINT64_C(0xcbf29ce484222325);
    mix(token, definition_signatures.size());
    for (uint64_t definition : definition_signatures) mix(token, definition);
    signature += disk_hash(token);
  }
  for (int chunk = 0; chunk < owner.num_chunks; ++chunk) {
    const fields_chunk &fc = *owner.chunks[chunk];
    if (!fc.is_mine()) continue;
    FOR_FIELD_TYPES(ft) for (const src_vol &source : fc.sources[ft]) {
      for (size_t point_index = 0; point_index < source.num_points(); ++point_index) {
        uint64_t row = UINT64_C(0xcbf29ce484222325);
        mix(row, uint64_t(ft));
        mix(row, uint64_t(source.c));
        const auto definition = source_time_signatures.find(uint32_t(source.t()->id));
        if (definition == source_time_signatures.end())
          throw std::invalid_argument("checkpoint source references an unknown source-time id");
        mix(row, definition->second);
        mix(row, source.t()->is_integrated ? 1 : 0);
        const ivec point = fc.gv.iloc(source.c, source.index_at(point_index));
        mix(row, uint64_t(point.dim));
        LOOP_OVER_DIRECTIONS(point.dim, d) { mix(row, encode_signed(point.in_direction(d))); }
        const std::complex<double> amplitude = source.amplitude_at(point_index);
        mix_double(row, amplitude.real());
        mix_double(row, amplitude.imag());
        signature += row;
      }
    }
  }
  return disk_hash(signature);
}

RecipeFingerprints checkpoint_dft_recipe_signatures(fields &owner) {
  std::vector<DftDescriptor> descriptors;
  build_dft_descriptors(owner, descriptors);
  RecipeFingerprints signature = {0, 0};
  int previous_chunk = std::numeric_limits<int>::min();
  uint64_t chunk_ordinal = 0;
  for (const DftDescriptor &descriptor : descriptors) {
    if (descriptor.chunk != previous_chunk) {
      previous_chunk = descriptor.chunk;
      chunk_ordinal = 0;
    }
    uint64_t portable = UINT64_C(0xcbf29ce484222325);
    uint64_t native = UINT64_C(0xcbf29ce484222325);
    const auto mix_word = [&](uint64_t value) {
      mix(portable, value);
      mix(native, value);
    };
    const auto mix_real = [&](double value, const char *what) {
      mix_portable_real(portable, value, what);
      mix_native_real(native, value, what);
    };
    /* Persistent DFT rows use a per-chunk list ordinal in StorageKey::aux.
       Bind that same stable identity into the recipe so swapping equal-shape
       monitors cannot silently attach saved accumulators to the other recipe. */
    mix_word(chunk_ordinal);
    mix_word(is_valid(descriptor.accumulator) ? 1 : 0);
    mix_word(is_valid(descriptor.phase_scratch) ? 1 : 0);
    const ArrayRef refs[] = {descriptor.source_field, descriptor.source_field_imag};
    for (const ArrayRef &ref : refs) {
      mix_word(is_valid(ref.id) ? 1 : 0);
      mix_word(ref.offset);
      mix_word(ref.elements);
    }
    mix_word(descriptor.omega.size());
    for (double omega : descriptor.omega) mix_real(omega, "checkpoint DFT frequency");
    mix_real(descriptor.scale.real(), "checkpoint DFT scale");
    mix_real(descriptor.scale.imag(), "checkpoint DFT scale");
    mix_word(encode_signed(descriptor.chunk));
    mix_word(uint64_t(descriptor.c));
    mix_word(uint64_t(descriptor.avg1));
    mix_word(uint64_t(descriptor.avg2));
    const ivec integer_vectors[] = {descriptor.is, descriptor.ie, descriptor.is_old,
                                    descriptor.ie_old};
    for (const ivec &v : integer_vectors) {
      mix_word(uint64_t(v.dim));
      LOOP_OVER_DIRECTIONS(v.dim, d) { mix_word(encode_signed(v.in_direction(d))); }
    }
    mix_word(descriptor.persist ? 1 : 0);
    mix_word(encode_signed(descriptor.decimation_factor));
    mix_word(descriptor.due_scalar_slot);
    const vec real_vectors[] = {descriptor.weights.s0, descriptor.weights.s1,
                                descriptor.weights.e0, descriptor.weights.e1};
    for (const vec &v : real_vectors) {
      mix_word(uint64_t(v.dim));
      LOOP_OVER_DIRECTIONS(v.dim, d) {
        mix_real(v.in_direction(d), "checkpoint DFT boundary weight");
      }
    }
    mix_real(descriptor.dV0, "checkpoint DFT volume weight");
    mix_real(descriptor.dV1, "checkpoint DFT volume weight");
    mix_word(descriptor.include_dV_and_interp_weights ? 1 : 0);
    mix_word(descriptor.sqrt_dV_and_interp_weights ? 1 : 0);
    mix_word(descriptor.N);
    mix_word(descriptor.Nomega);
    signature.portable += disk_hash(portable);
    signature.native += disk_hash(native);
    ++chunk_ordinal;
  }
  signature.portable = disk_hash(signature.portable);
  signature.native = disk_hash(signature.native);
  return signature;
}

const size_t checkpoint_row_columns = 26;
const char checkpoint_header_name[] = "backend_checkpoint_header";
const char checkpoint_rows_name[] = "backend_checkpoint_rows";
const char checkpoint_values_name[] = "backend_checkpoint_values";
const char checkpoint_scalars_name[] = "backend_checkpoint_scalars";
const char checkpoint_generations_name[] = "backend_checkpoint_generations";
const char checkpoint_seed_name[] = "backend_checkpoint_seed";
const char checkpoint_flux_keys_name[] = "backend_checkpoint_flux_keys";
const char checkpoint_flux_values_name[] = "backend_checkpoint_flux_values";

bool exact_builtin_source_time(const src_time *source) {
  return typeid(*source) == typeid(gaussian_src_time) ||
         typeid(*source) == typeid(continuous_src_time);
}

bool reconstructable_checkpoint_kind(array_kind kind) {
  /* Magnetic synchronization is rejected while active, so these retained
     rollback arrays contain no live continuation state. Their allocation is a
     backend/eager implementation detail and may legitimately differ after a
     backend reselection. */
  return kind == array_kind::f_backup || kind == array_kind::f_u_backup ||
         kind == array_kind::f_w_backup || kind == array_kind::f_cond_backup ||
         kind == array_kind::f_bfast_backup;
}

bool persistent_checkpoint_row(const ArraySpec &spec, const StorageKey &key) {
  return !spec.classification_elided && spec.role != array_role::scratch &&
         spec.role != array_role::communication &&
         !reconstructable_checkpoint_kind(array_kind(key.kind));
}

bool same_row_semantics(const StorageKey &a, const StorageKey &b) {
  return a.kind == b.kind && a.component_ == b.component_ && a.cmp == b.cmp && a.aux == b.aux;
}

bool valid_component(int c) { return c >= 0 && c < NUM_FIELD_COMPONENTS; }

bool valid_checkpoint_key(const StorageKey &key) {
  if (key.chunk < 0 || key.kind < 0 || key.kind >= int(array_kind::num_kinds)) return false;
  const array_kind kind = array_kind(key.kind);
  switch (kind) {
    case array_kind::f:
    case array_kind::f_u:
    case array_kind::f_w:
    case array_kind::f_w_prev:
    case array_kind::f_cond:
    case array_kind::f_bfast:
    case array_kind::f_minus_p:
    case array_kind::f_backup:
    case array_kind::f_u_backup:
    case array_kind::f_w_backup:
    case array_kind::f_cond_backup:
    case array_kind::f_bfast_backup:
      return valid_component(key.component_) && (key.cmp == 0 || key.cmp == 1) && key.aux == 0;
    case array_kind::f_rderiv_int:
      return key.component_ == -1 && key.cmp == -1 && key.aux == 0;
    case array_kind::chi1inv:
    case array_kind::conductivity:
    case array_kind::condinv:
      return valid_component(key.component_) && key.cmp == -1 && key.aux < 5;
    case array_kind::chi2:
    case array_kind::chi3:
      return valid_component(key.component_) && key.cmp == -1 && key.aux == 0;
    case array_kind::sigma: {
      if (!valid_component(key.component_) || key.cmp < int(X) || key.cmp >= int(NO_DIRECTION))
        return false;
      const uint64_t state = key.aux / uint64_t(NUM_FIELD_TYPES);
      const field_type ft = field_type(key.aux % uint64_t(NUM_FIELD_TYPES));
      return state <= uint64_t(std::numeric_limits<int>::max()) &&
             (ft == E_stuff || ft == H_stuff);
    }
    case array_kind::pml_sig:
    case array_kind::pml_kap:
    case array_kind::pml_siginv:
      return key.component_ == -1 && key.cmp == -1 && key.aux < 6;
    case array_kind::dft:
    case array_kind::dft_phase:
      return valid_component(key.component_) && key.cmp == -1 &&
             key.aux <= uint64_t(std::numeric_limits<int>::max());
    case array_kind::polarization_internal:
      if ((key.component_ < 0 || key.component_ > int(Centered)) ||
          (key.cmp < -1 || key.cmp > 1))
        return false;
      try {
        (void)polarization_storage_field_type(key.aux);
        (void)polarization_storage_state_index(key.aux);
      }
      catch (...) { return false; }
      return true;
    case array_kind::num_kinds: return false;
  }
  return false;
}

RecipeFingerprints checkpoint_material_definition_signatures(
    fields &owner, const StoragePlan *prepared_plan = NULL,
    const CpuArrayCatalog *prepared_catalog = NULL) {
  const MaterialIR *ir = material_ir_for(owner);

  /* Direct C++ construction predates MaterialIR capture. Build a commutative
     global identity from owned Yee points, excluding chunk numbers.  This is
     also the cross-precision identity for MaterialIR-backed definitions: the
     resolved coefficient recipe is the portable continuation contract, while
     the native MaterialIR signature below retains every exact definition bit.
     Every decomposition contributes each physical material sample once. */
  CpuArrayCatalog local_catalog;
  StoragePlan local_plan;
  if (!prepared_plan || !prepared_catalog) {
    build_storage_catalog(owner, local_catalog, local_plan);
    prepared_plan = &local_plan;
    prepared_catalog = &local_catalog;
  }
  if (prepared_plan->arrays.size() != prepared_plan->keys.size())
    throw std::invalid_argument("checkpoint material storage plan has mismatched arrays and keys");
  RecipeFingerprints signature = {am_master() ? UINT64_C(1) : UINT64_C(0),
                                  am_master() ? UINT64_C(1) : UINT64_C(0)};
  if (ir && am_master()) {
    signature.portable = material_ir_portable_signature(*ir);
    signature.native = ir->signature ? ir->signature : 1;
  }
  for (size_t row_index = 0; row_index < prepared_plan->arrays.size(); ++row_index) {
    const ArraySpec &spec = prepared_plan->arrays[row_index];
    const StorageKey &key = prepared_plan->keys[row_index];
    if (spec.role != array_role::material || is_valid(spec.alias_of)) continue;
    if (key.chunk < 0 || key.chunk >= owner.num_chunks ||
        !owner.chunks[key.chunk]->is_mine())
      continue;
    if (spec.element_type != ElementType::realnum_value &&
        spec.element_type != ElementType::complex_realnum)
      throw std::invalid_argument("checkpoint material storage has a non-real element type");
    const Precision native_precision =
        sizeof(realnum) == sizeof(float) ? Precision::f32 : Precision::f64;
    if (spec.storage != native_precision || spec.alignment != alignof(realnum))
      throw std::invalid_argument("checkpoint material storage has a non-native layout");
    const realnum *values = prepared_catalog->resolve<realnum>(ArrayId{uint32_t(row_index)});
    if (!values) throw std::invalid_argument("checkpoint material storage has a null row");
    const grid_volume &gv = owner.chunks[key.chunk]->gv;
    if (spec.element_type == ElementType::realnum_value &&
        spec.elements == size_t(gv.ntot())) {
      const component c = key.component_ >= 0 ? component(key.component_) : Centered;
      for (size_t i = 0; i < spec.elements; ++i) {
        const ivec point = gv.iloc(c, ptrdiff_t(i));
        if (!gv.owns(point)) continue;
        uint64_t portable = UINT64_C(0xcbf29ce484222325);
        uint64_t native = UINT64_C(0xcbf29ce484222325);
        const auto mix_word = [&](uint64_t value) {
          mix(portable, value);
          mix(native, value);
        };
        mix_word(uint64_t(key.kind));
        mix_word(encode_signed(key.component_));
        mix_word(encode_signed(key.cmp));
        mix_word(key.aux);
        const direction axes[3] = {gv.dim == Dcyl ? R : X, Y, Z};
        for (int axis = 0; axis < 3; ++axis)
          mix_word(has_direction(gv.dim, axes[axis])
                       ? encode_signed(point.in_direction(axes[axis]))
                       : 0);
        mix_portable_real(portable, double(values[i]), "checkpoint material sample");
        mix_native_real(native, double(values[i]), "checkpoint material sample");
        signature.portable += disk_hash(portable);
        if (!ir) signature.native += disk_hash(native);
      }
    }
    else {
      /* Repartition of non-Yee material rows is unsupported. Keep an exact
         rank-layout identity so these rows are validated rather than skipped. */
      uint64_t portable = UINT64_C(0xcbf29ce484222325);
      uint64_t native = UINT64_C(0xcbf29ce484222325);
      const auto mix_word = [&](uint64_t value) {
        mix(portable, value);
        mix(native, value);
      };
      mix_word(encode_signed(key.chunk));
      mix_word(uint64_t(key.kind));
      mix_word(encode_signed(key.component_));
      mix_word(encode_signed(key.cmp));
      mix_word(key.aux);
      mix_word(spec.elements);
      const size_t scalar_multiplier =
          spec.element_type == ElementType::complex_realnum ? 2 : 1;
      if (spec.elements > std::numeric_limits<size_t>::max() / scalar_multiplier)
        throw std::invalid_argument("checkpoint material row has an invalid scalar extent");
      for (size_t i = 0; i < spec.elements * scalar_multiplier; ++i) {
        mix_portable_real(portable, double(values[i]), "checkpoint material row");
        mix_native_real(native, double(values[i]), "checkpoint material row");
      }
      signature.portable += disk_hash(portable);
      if (!ir) signature.native += disk_hash(native);
    }
  }
  /* Susceptibility objects are cloned into every structure chunk, so include
     one ordered E/H chain identity rather than multiplying it by the current
     decomposition.  Sigma/material arrays above retain their spatial
     identity; this token closes the previously missing dynamics identity. */
  if (am_master() && owner.num_chunks > 0 && owner.chunks[0] && owner.chunks[0]->s) {
    FOR_FIELD_TYPES(ft) {
      uint64_t portable = UINT64_C(0xcbf29ce484222325);
      uint64_t native = UINT64_C(0xcbf29ce484222325);
      mix(portable, uint64_t(ft));
      mix(native, uint64_t(ft));
      mix(portable, susceptibility_chain_signature(owner.chunks[0]->s->chiP[ft]));
      mix(native, susceptibility_chain_native_signature(owner.chunks[0]->s->chiP[ft]));
      signature.portable += disk_hash(portable);
      if (!ir) signature.native += disk_hash(native);
    }
  }
  signature.portable = disk_hash(signature.portable);
  signature.native = disk_hash(signature.native);
  return signature;
}

bool same_random_seed(const RandomSeedSnapshot &a, const RandomSeedSnapshot &b) {
  return a.semantic_seed == b.semantic_seed &&
         a.saved_semantic_seed == b.saved_semantic_seed && a.generation == b.generation &&
         a.algorithm_version == b.algorithm_version && a.initialized == b.initialized &&
         a.semantic_seed_valid == b.semantic_seed_valid &&
         a.saved_semantic_seed_valid == b.saved_semantic_seed_valid &&
         a.explicit_seed == b.explicit_seed &&
         a.saved_explicit_seed == b.saved_explicit_seed;
}

const RandomSeedSnapshot *checkpoint_seed_for_rank(const CheckpointImage &image) {
  if (image.random_seed_ranks.size() != image.random_seeds.size() ||
      image.random_seeds.empty())
    return NULL;
  if (image.saved_rank_count == uint64_t(count_processors())) {
    const uint32_t rank = uint32_t(my_global_rank());
    const RandomSeedSnapshot *match = NULL;
    for (size_t i = 0; i < image.random_seeds.size(); ++i)
      if (image.random_seed_ranks[i] == rank) {
        if (match) return NULL;
        match = &image.random_seeds[i];
      }
    return match;
  }
  /* Rank-count remap is defined only for a shared manifest containing every
     saved rank and only when the semantic RNG state was globally identical.
     Noisy rank-keyed streams therefore reject ambiguous remaps. */
  if (image.random_seeds.size() != image.saved_rank_count) return NULL;
  for (size_t i = 1; i < image.random_seeds.size(); ++i)
    if (!same_random_seed(image.random_seeds.front(), image.random_seeds[i])) return NULL;
  return &image.random_seeds.front();
}

size_t row_region_points(const CheckpointRow &row, ndim dim) {
  const direction axes[3] = {dim == Dcyl ? R : X, Y, Z};
  size_t points = 1;
  for (int axis = 0; axis < 3; ++axis) {
    if (!has_direction(dim, axes[axis])) continue;
    const int64_t extent = int64_t(row.big_corner[axis]) - int64_t(row.little_corner[axis]);
    if (extent < 0 || (extent & 1)) return 0;
    const size_t count = size_t(uint64_t(extent / 2)) + 1;
    if (points > std::numeric_limits<size_t>::max() / count) return 0;
    points *= count;
  }
  return points;
}

bool row_is_full_spatial(const CheckpointRow &row, ndim dim) {
  const array_kind kind = array_kind(row.key.kind);
  if (kind == array_kind::dft || kind == array_kind::dft_phase ||
      kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
      kind == array_kind::pml_siginv || kind == array_kind::polarization_internal)
    return false;
  const size_t points = row_region_points(row, dim);
  return points && row.spec.element_type == ElementType::realnum_value &&
         row.spec.elements == points;
}

bool saved_spatial_index(const CheckpointRow &row, const grid_volume &target_gv,
                         const ivec &point, size_t &index) {
  const ndim dim = target_gv.dim;
  const direction axes[3] = {dim == Dcyl ? R : X, Y, Z};
  const component c = row.key.component_ >= 0 ? component(row.key.component_) : Centered;
  const ivec shift = target_gv.iyee_shift(c);
  size_t coordinates[3] = {0, 0, 0};
  size_t counts[3] = {1, 1, 1};
  for (int axis = 0; axis < 3; ++axis) {
    const direction d = axes[axis];
    if (!has_direction(dim, d)) continue;
    const int64_t extent = int64_t(row.big_corner[axis]) - int64_t(row.little_corner[axis]);
    if (extent < 0 || (extent & 1)) return false;
    counts[axis] = size_t(extent / 2) + 1;
    const int64_t delta = int64_t(point.in_direction(d)) - int64_t(row.little_corner[axis]) -
                          int64_t(shift.in_direction(d));
    if (delta < 0 || (delta & 1) || size_t(delta / 2) >= counts[axis]) return false;
    coordinates[axis] = size_t(delta / 2);
  }
  if (dim == D1)
    index = coordinates[2];
  else if (dim == D2)
    index = coordinates[0] * counts[1] + coordinates[1];
  else if (dim == Dcyl)
    index = coordinates[0] * counts[2] + coordinates[2];
  else
    index = (coordinates[0] * counts[1] + coordinates[1]) * counts[2] + coordinates[2];
  return index < row.spec.elements;
}

void copy_remapped_spatial_row(void *address, const CheckpointRow &target,
                               const grid_volume &target_gv,
                               const std::vector<const CheckpointRow *> &sources,
                               std::vector<bool> *used_sources = NULL) {
  if (!row_is_full_spatial(target, target_gv.dim) ||
      target.spec.elements != size_t(target_gv.ntot()))
    throw std::invalid_argument("checkpoint repartition target row is not a full Yee-grid row");
  realnum *destination = static_cast<realnum *>(address);
  for (size_t i = 0; i < target.spec.elements; ++i) {
    const component c = target.key.component_ >= 0 ? component(target.key.component_) : Centered;
    const ivec point = target_gv.iloc(c, ptrdiff_t(i));
    bool found = false;
    double value = 0.0;
    for (size_t source_ordinal = 0; source_ordinal < sources.size(); ++source_ordinal) {
      const CheckpointRow *source = sources[source_ordinal];
      size_t source_index = 0;
      if (!saved_spatial_index(*source, target_gv, point, source_index)) continue;
      if (found && source->values[source_index] != value)
        throw std::invalid_argument("checkpoint repartition contains conflicting overlap values");
      value = source->values[source_index];
      found = true;
      if (used_sources) (*used_sources)[source_ordinal] = true;
    }
    if (!found)
      throw std::invalid_argument("checkpoint repartition omits a target Yee-grid point");
    destination[i] = checked_checkpoint_scalar(value, sources.front()->spec.storage);
  }
}

void set_row_corners(CheckpointRow &row, const fields &owner) {
  if (row.key.chunk < 0 || row.key.chunk >= owner.num_chunks) return;
  const grid_volume &gv = owner.chunks[row.key.chunk]->gv;
  const ivec lo = gv.little_corner();
  const ivec hi = gv.big_corner();
  const direction directions[3] = {gv.dim == Dcyl ? R : X, Y, Z};
  for (int i = 0; i < 3; ++i) {
    row.little_corner[i] = has_direction(gv.dim, directions[i]) ? lo.in_direction(directions[i]) : 0;
    row.big_corner[i] = has_direction(gv.dim, directions[i]) ? hi.in_direction(directions[i]) : 0;
  }
}

void copy_host_row(const void *address, CheckpointRow &row) {
  const size_t count = checkpoint_scalar_count(row.spec);
  row.values.resize(count);
  const realnum *source = static_cast<const realnum *>(address);
  for (size_t i = 0; i < count; ++i) row.values[i] = double(source[i]);
  row.checksum = checkpoint_row_checksum(row);
}

void copy_to_host_row(void *address, const CheckpointRow &row) {
  const size_t count = checkpoint_scalar_count(row.spec);
  if (row.values.size() != count)
    throw std::invalid_argument("checkpoint row has the wrong scalar count");
  realnum *target = static_cast<realnum *>(address);
  for (size_t i = 0; i < count; ++i)
    target[i] = checked_checkpoint_scalar(row.values[i], row.spec.storage);
}

realnum **field_slot(fields_chunk &chunk, array_kind kind, component c, int cmp) {
  switch (kind) {
    case array_kind::f: return &chunk.f[c][cmp];
    case array_kind::f_u: return &chunk.f_u[c][cmp];
    case array_kind::f_w: return &chunk.f_w[c][cmp];
    case array_kind::f_w_prev: return &chunk.f_w_prev[c][cmp];
    case array_kind::f_cond: return &chunk.f_cond[c][cmp];
    case array_kind::f_bfast: return &chunk.f_bfast[c][cmp];
    case array_kind::f_minus_p: return &chunk.f_minus_p[c][cmp];
    case array_kind::f_backup: return &chunk.f_backup[c][cmp];
    case array_kind::f_u_backup: return &chunk.f_u_backup[c][cmp];
    case array_kind::f_w_backup: return &chunk.f_w_backup[c][cmp];
    case array_kind::f_cond_backup: return &chunk.f_cond_backup[c][cmp];
    case array_kind::f_bfast_backup: return &chunk.f_bfast_backup[c][cmp];
    default: return NULL;
  }
}

susceptibility *susceptibility_at(structure_chunk &chunk, field_type ft, int index) {
  susceptibility *current = chunk.chiP[ft];
  while (current && index-- > 0) current = current->next;
  return current;
}

polarization_state *polarization_at(fields_chunk &chunk, field_type ft, int index) {
  polarization_state *current = chunk.pol[ft];
  while (current && index-- > 0) current = current->next;
  return current;
}

void *resolve_chunk_row(fields_chunk &chunk, const CheckpointRow &row, bool allocate) {
  const array_kind kind = array_kind(row.key.kind);
  if (realnum **slot = field_slot(chunk, kind, component(row.key.component_), row.key.cmp)) {
    if (!*slot && allocate) *slot = new realnum[row.spec.elements];
    return *slot;
  }
  if (kind == array_kind::f_rderiv_int) {
    if (!chunk.f_rderiv_int && allocate) chunk.f_rderiv_int = new realnum[row.spec.elements];
    return chunk.f_rderiv_int;
  }

  structure_chunk &structure = *chunk.s;
  const component c = component(row.key.component_);
  switch (kind) {
    case array_kind::chi1inv: return structure.chi1inv[c][int(row.key.aux)];
    case array_kind::conductivity: return structure.conductivity[c][int(row.key.aux)];
    case array_kind::condinv: return structure.condinv[c][int(row.key.aux)];
    case array_kind::chi2: return structure.chi2[c];
    case array_kind::chi3: return structure.chi3[c];
    case array_kind::pml_sig: return structure.sig[int(row.key.aux)];
    case array_kind::pml_kap: return structure.kap[int(row.key.aux)];
    case array_kind::pml_siginv: return structure.siginv[int(row.key.aux)];
    case array_kind::sigma: {
      const field_type ft = field_type(row.key.aux % NUM_FIELD_TYPES);
      const int state = int(row.key.aux / NUM_FIELD_TYPES);
      susceptibility *sus = susceptibility_at(structure, ft, state);
      return sus ? sus->sigma[c][row.key.cmp] : NULL;
    }
    case array_kind::polarization_internal: {
      const field_type ft = polarization_storage_field_type(row.key.aux);
      const int state_index = polarization_storage_state_index(row.key.aux);
      const uint32_t ordinal = polarization_storage_layout_ordinal(row.key.aux);
      polarization_state *state = polarization_at(chunk, ft, state_index);
      if (!state) return NULL;
      if (!state->data && allocate) {
        state->data = state->s->new_internal_data(chunk.f, chunk.gv);
        state->s->init_internal_data(chunk.f, chunk.dt, chunk.gv, state->data);
      }
      if (!state->data) return NULL;
      std::vector<InternalArrayLayout> layout;
      if (!state->s->internal_layout(layout, chunk.gv, state->data) || ordinal >= layout.size())
        return NULL;
      const InternalArrayLayout &entry = layout[ordinal];
      if (entry.c != c || entry.cmp != row.key.cmp || entry.elements != row.spec.elements)
        return NULL;
      return static_cast<realnum *>(state->data) + entry.offset_elements;
    }
    default: return NULL;
  }
}

std::complex<realnum> **dft_slot(fields &owner, const StorageKey &key) {
  if (key.chunk < 0 || key.chunk >= owner.num_chunks) return NULL;
  int ordinal = 0;
  for (dft_chunk *dft = owner.chunks[key.chunk]->dft_chunks; dft;
       dft = dft->next_in_chunk, ++ordinal) {
    if (ordinal != int(key.aux) || int(dft->c) != key.component_) continue;
    if (key.kind == int(array_kind::dft)) return &dft->dft;
    if (key.kind == int(array_kind::dft_phase)) return &dft->dft_phase;
  }
  return NULL;
}

struct DftReplacement {
  std::complex<realnum> **slot;
  std::complex<realnum> *replacement;
  DftReplacement() : slot(NULL), replacement(NULL) {}
  ~DftReplacement() { delete[] replacement; }
  void publish() noexcept {
    std::swap(*slot, replacement);
  }
};

bool checkpoint_local_exact_layout(fields &owner, const CheckpointImage &image) {
  CpuArrayCatalog catalog;
  StoragePlan plan;
  build_storage_catalog(owner, catalog, plan);
  for (size_t target_index = 0; target_index < plan.arrays.size(); ++target_index) {
    const ArraySpec &spec = plan.arrays[target_index];
    if (!persistent_checkpoint_row(spec, plan.keys[target_index])) continue;
    const StorageKey &key = plan.keys[target_index];
    const CheckpointRow *match = NULL;
    for (const CheckpointRow &source : image.rows)
      if (source.key == key) {
        match = &source;
        break;
      }
    CheckpointRow target;
    target.key = key;
    target.spec = spec;
    set_row_corners(target, owner);
    const bool target_alias = is_valid(spec.alias_of);
    if (!match || match->spec.element_type != spec.element_type ||
        match->spec.elements != spec.elements || match->has_alias != target_alias ||
        !std::equal(target.little_corner, target.little_corner + 3, match->little_corner) ||
        !std::equal(target.big_corner, target.big_corner + 3, match->big_corner))
      return false;
    if (target_alias && !(match->alias_key == catalog.key(spec.alias_of))) return false;
  }
  return true;
}

} // namespace

class PreparedCheckpointCommit {
public:
  PreparedCheckpointCommit(fields &owner, const CheckpointImage &image, bool exact_layout)
      : owner_(owner), covered_sources_(image.rows.size(), 0) {
    CpuArrayCatalog target_catalog;
    StoragePlan target_plan;
    build_storage_catalog(owner, target_catalog, target_plan);
    chunks_.resize(size_t(owner.num_chunks));
    for (int chunk = 0; chunk < owner.num_chunks; ++chunk) {
      if (!owner.chunks[chunk]->is_mine()) continue;
      fields_chunk &live = *owner.chunks[chunk];
      chunks_[size_t(chunk)].reset(new fields_chunk(
          live.s, live.outdir, live.m, live.beta, live.zero_fields_near_cylorigin,
          live.chunk_idx, 0, live.bfast_scaled_k));
      chunks_[size_t(chunk)]->is_real = live.is_real;
      chunks_[size_t(chunk)]->changing_structure();
      /* changing_structure() may clone the structure after the fields_chunk
         constructor built polarization nodes against live.s. Rebind every
         staged node to the corresponding susceptibility in the cloned chain
         before allocating or restoring internal state. */
      FOR_FIELD_TYPES(ft) {
        polarization_state *state = chunks_[size_t(chunk)]->pol[ft];
        susceptibility *sus = chunks_[size_t(chunk)]->s->chiP[ft];
        while (state && sus) {
          state->s = sus;
          state = state->next;
          sus = sus->next;
        }
        if (state || sus)
          throw std::invalid_argument("checkpoint staged polarization chain mismatch");
      }
    }
    checkpoint_fail_if_requested(CheckpointFailurePoint::allocation);

    if (!exact_layout) {
      /* Scoped PR6.5 rechunking: map allocation-stable, full Yee-grid rows by
         semantic identity and global lattice coordinate. Saved chunk numbers
         are provenance only. DFT regions, polarization blobs, and 1-D PML
         profiles require their richer recipe-specific mapping and therefore
         reject transactionally instead of guessing. */
      for (size_t target_index = 0; target_index < target_plan.arrays.size(); ++target_index) {
        const ArraySpec &spec = target_plan.arrays[target_index];
        if (!persistent_checkpoint_row(spec, target_plan.keys[target_index]) ||
            is_valid(spec.alias_of))
          continue;
        CheckpointRow target;
        target.key = target_plan.keys[target_index];
        target.spec = spec;
        target.spec.id = invalid_array();
        target.spec.alias_of = invalid_array();
        set_row_corners(target, owner);
        if (!row_is_full_spatial(target, owner.gv.dim))
          throw std::invalid_argument(
              "checkpoint repartition contains a row without a supported global Yee mapping");
        std::vector<const CheckpointRow *> sources;
        std::vector<size_t> source_indices;
        for (size_t source_index = 0; source_index < image.rows.size(); ++source_index) {
          const CheckpointRow &source = image.rows[source_index];
          if (!source.has_alias && same_row_semantics(source.key, target.key)) {
            if (!row_is_full_spatial(source, owner.gv.dim))
              throw std::invalid_argument(
                  "checkpoint repartition source row is not a full Yee-grid row");
            sources.push_back(&source);
            source_indices.push_back(source_index);
          }
        }
        if (sources.empty())
          throw std::invalid_argument(
              std::string("checkpoint repartition omits a semantic storage row: ") +
              array_kind_name(array_kind(target.key.kind)) + "/" +
              std::to_string(target.key.component_) + "/" + std::to_string(target.key.cmp) +
              "/" + std::to_string(target.key.aux));
        void *destination = resolve_chunk_row(*chunks_[size_t(target.key.chunk)], target, true);
        if (!destination)
          throw std::invalid_argument("checkpoint repartition row has no target allocation");
        std::vector<bool> used(sources.size(), false);
        copy_remapped_spatial_row(destination, target, owner.chunks[target.key.chunk]->gv, sources,
                                  &used);
        for (size_t i = 0; i < source_indices.size(); ++i)
          if (used[i]) covered_sources_[source_indices[i]] = 1;
      }

      /* A later-time checkpoint can contain lazily materialized plain field
         families that a freshly stepped target decomposition has not yet
         allocated. Recreate those semantic rows on each owned target chunk;
         richer state families remain explicitly unsupported below. */
      std::unordered_map<StorageKey, std::vector<size_t>, StorageKeyHash> saved_fields;
      for (size_t source_index = 0; source_index < image.rows.size(); ++source_index) {
        const CheckpointRow &source = image.rows[source_index];
        const array_kind kind = array_kind(source.key.kind);
        if (source.has_alias || kind < array_kind::f || kind > array_kind::f_rderiv_int ||
            !row_is_full_spatial(source, owner.gv.dim))
          continue;
        StorageKey semantic = source.key;
        semantic.chunk = -1;
        saved_fields[semantic].push_back(source_index);
      }
      for (const auto &family : saved_fields) {
        std::vector<const CheckpointRow *> sources;
        for (size_t source_index : family.second) sources.push_back(&image.rows[source_index]);
        for (int chunk = 0; chunk < owner.num_chunks; ++chunk) {
          if (!chunks_[size_t(chunk)]) continue;
          bool already_present = false;
          for (const StorageKey &key : target_plan.keys)
            if (key.chunk == chunk && same_row_semantics(key, family.first)) {
              already_present = true;
              break;
            }
          if (already_present) continue;
          CheckpointRow target;
          target.key = family.first;
          target.key.chunk = chunk;
          target.spec = image.rows[family.second.front()].spec;
          target.spec.id = invalid_array();
          target.spec.alias_of = invalid_array();
          target.spec.elements = size_t(owner.chunks[chunk]->gv.ntot());
          set_row_corners(target, owner);
          void *destination = resolve_chunk_row(*chunks_[size_t(chunk)], target, true);
          if (!destination)
            throw std::invalid_argument(
                "checkpoint repartition cannot allocate a saved field family");
          std::vector<bool> used(sources.size(), false);
          copy_remapped_spatial_row(destination, target, owner.chunks[chunk]->gv, sources, &used);
          for (size_t i = 0; i < family.second.size(); ++i)
            if (used[i]) covered_sources_[family.second[i]] = 1;
        }
      }

      for (size_t target_index = 0; target_index < target_plan.arrays.size(); ++target_index) {
        const ArraySpec &spec = target_plan.arrays[target_index];
        if (!persistent_checkpoint_row(spec, target_plan.keys[target_index]) ||
            !is_valid(spec.alias_of))
          continue;
        const StorageKey &key = target_plan.keys[target_index];
        const StorageKey &canonical_key = target_catalog.key(spec.alias_of);
        bool found_alias = false;
        for (size_t source_index = 0; source_index < image.rows.size(); ++source_index) {
          const CheckpointRow &source = image.rows[source_index];
          if (source.has_alias && same_row_semantics(source.key, key)) {
            if (!same_row_semantics(source.alias_key, canonical_key))
              throw std::invalid_argument("checkpoint repartition has a conflicting alias row");
            found_alias = true;
            covered_sources_[source_index] = 1;
          }
        }
        if (!found_alias)
          throw std::invalid_argument("checkpoint repartition omits a canonical alias row");
        fields_chunk &chunk = *chunks_[size_t(key.chunk)];
        realnum **alias = field_slot(chunk, array_kind(key.kind), component(key.component_), key.cmp);
        realnum **canonical = field_slot(chunk, array_kind(canonical_key.kind),
                                         component(canonical_key.component_), canonical_key.cmp);
        if (!alias || !canonical || !*canonical)
          throw std::invalid_argument("checkpoint repartition alias has no canonical target");
        delete[] *alias;
        *alias = *canonical;
      }

      checkpoint_fail_if_requested(CheckpointFailurePoint::validation);
      return;
    }

    /* First allocate/copy canonical field rows so polarization constructors
       see their complete W-array dependencies. */
    for (const CheckpointRow &row : image.rows) {
      if (row.has_alias || row.key.chunk < 0 || row.key.chunk >= owner.num_chunks ||
          !chunks_[size_t(row.key.chunk)])
        continue;
      const array_kind kind = array_kind(row.key.kind);
      if (kind == array_kind::dft || kind == array_kind::dft_phase ||
          kind == array_kind::polarization_internal ||
          (kind >= array_kind::chi1inv && kind <= array_kind::pml_siginv) ||
          kind == array_kind::sigma)
        continue;
      void *target = resolve_chunk_row(*chunks_[size_t(row.key.chunk)], row, true);
      if (!target) throw std::invalid_argument("checkpoint field row has no target allocation");
      copy_to_host_row(target, row);
      covered_sources_[size_t(&row - image.rows.data())] = 1;
    }

    for (const CheckpointRow &row : image.rows) {
      if (row.has_alias || row.key.chunk < 0 || row.key.chunk >= owner.num_chunks) continue;
      const array_kind kind = array_kind(row.key.kind);
      if (kind == array_kind::dft || kind == array_kind::dft_phase) {
        if (!owner.chunks[row.key.chunk]->is_mine()) continue;
        std::complex<realnum> **slot = dft_slot(owner, row.key);
        if (!slot) throw std::invalid_argument("checkpoint DFT row has no live target");
        if (row.spec.element_type != ElementType::complex_realnum ||
            row.values.size() != checkpoint_scalar_count(row.spec))
          throw std::invalid_argument("checkpoint DFT row has the wrong storage type or size");
        std::unique_ptr<DftReplacement> replacement(new DftReplacement);
        replacement->slot = slot;
        replacement->replacement = new std::complex<realnum>[row.spec.elements];
        for (size_t i = 0; i < row.spec.elements; ++i)
          replacement->replacement[i] =
              std::complex<realnum>(checked_checkpoint_scalar(row.values[2 * i], row.spec.storage),
                                    checked_checkpoint_scalar(row.values[2 * i + 1],
                                                              row.spec.storage));
        dfts_.push_back(std::move(replacement));
        covered_sources_[size_t(&row - image.rows.data())] = 1;
        continue;
      }
      if (!chunks_[size_t(row.key.chunk)]) continue;
      if (kind < array_kind::chi1inv && kind != array_kind::polarization_internal) continue;
      void *target = resolve_chunk_row(*chunks_[size_t(row.key.chunk)], row, true);
      if (!target) throw std::invalid_argument("checkpoint material/polarization row has no target");
      copy_to_host_row(target, row);
      covered_sources_[size_t(&row - image.rows.data())] = 1;
    }

    for (const CheckpointRow &row : image.rows) {
      if (!row.has_alias || row.key.chunk < 0 || row.key.chunk >= owner.num_chunks ||
          !chunks_[size_t(row.key.chunk)])
        continue;
      fields_chunk &chunk = *chunks_[size_t(row.key.chunk)];
      realnum **alias = field_slot(chunk, array_kind(row.key.kind), component(row.key.component_),
                                   row.key.cmp);
      realnum **canonical =
          field_slot(chunk, array_kind(row.alias_key.kind), component(row.alias_key.component_),
                     row.alias_key.cmp);
      if (!alias || !canonical || !*canonical)
        throw std::invalid_argument("checkpoint alias has no canonical field target");
      delete[] *alias;
      *alias = *canonical;
      covered_sources_[size_t(&row - image.rows.data())] = 1;
    }
    checkpoint_fail_if_requested(CheckpointFailurePoint::validation);
  }

  void publish() noexcept {
    for (int chunk = 0; chunk < owner_.num_chunks; ++chunk)
      if (chunks_[size_t(chunk)]) owner_.chunks[chunk]->swap_checkpoint_state(*chunks_[size_t(chunk)]);
    for (std::unique_ptr<DftReplacement> &dft : dfts_) dft->publish();
  }

  const std::vector<size_t> &covered_sources() const noexcept { return covered_sources_; }

private:
  fields &owner_;
  std::vector<std::unique_ptr<fields_chunk> > chunks_;
  std::vector<std::unique_ptr<DftReplacement> > dfts_;
  std::vector<size_t> covered_sources_;
};

uint64_t checkpoint_encode_signed_for_testing(int value) { return encode_signed(value); }

int checkpoint_decode_signed_for_testing(uint64_t value) { return decode_signed(value); }

CheckpointRow::CheckpointRow()
    : key{-1, -1, -1, -1, 0}, spec{invalid_array(), array_role::scratch,
                                   ElementType::realnum_value, Precision::f64, 0, 0,
                                   invalid_array(), false, false},
      alias_key{-1, -1, -1, -1, 0}, has_alias(false), little_corner{0, 0, 0},
      big_corner{0, 0, 0}, checksum(0) {}

CheckpointImage::CheckpointImage()
    : schema_magic(checkpoint_schema_magic), schema_version(checkpoint_schema_version),
      endian_marker(UINT64_C(0x01020304)), host_realnum_bytes(sizeof(realnum)),
      dimension(0), configuration_signature(0), storage_signature(0),
      material_recipe_signature(0), material_native_signature(0), classification_hash(0),
      source_definition_signature(0), dft_recipe_signature(0), dft_native_signature(0),
      saved_rank_count(1), shared_manifest(false), timestep(0), dt(0.0),
      source_times{0.0, 0.0, 0.0}, random_seed{} {
  for (int i = 0; i < fields::num_mutation_kinds; ++i) mutation_generation[i] = 0;
}

size_t checkpoint_scalar_count(const ArraySpec &spec) {
  switch (spec.element_type) {
    case ElementType::realnum_value: return spec.elements;
    case ElementType::complex_realnum:
      if (spec.elements > std::numeric_limits<size_t>::max() / 2)
        throw std::overflow_error("checkpoint complex row size overflow");
      return spec.elements * 2;
    default: throw std::invalid_argument("checkpoint row has a non-real storage type");
  }
}

uint64_t checkpoint_row_checksum(const CheckpointRow &row) {
  uint64_t hash = UINT64_C(0xcbf29ce484222325);
  mix(hash, encode_signed(row.key.chunk));
  mix(hash, encode_signed(row.key.kind));
  mix(hash, encode_signed(row.key.component_));
  mix(hash, encode_signed(row.key.cmp));
  mix(hash, row.key.aux);
  mix(hash, uint64_t(row.spec.role));
  mix(hash, uint64_t(row.spec.element_type));
  mix(hash, uint64_t(row.spec.storage));
  mix(hash, row.spec.elements);
  mix(hash, row.spec.alignment);
  mix(hash, row.spec.classification_provisional ? 1 : 0);
  mix(hash, row.spec.classification_elided ? 1 : 0);
  mix(hash, row.has_alias ? 1 : 0);
  if (row.has_alias) {
    mix(hash, encode_signed(row.alias_key.chunk));
    mix(hash, encode_signed(row.alias_key.kind));
    mix(hash, encode_signed(row.alias_key.component_));
    mix(hash, encode_signed(row.alias_key.cmp));
    mix(hash, row.alias_key.aux);
  }
  for (int d = 0; d < 3; ++d) {
    mix(hash, encode_signed(row.little_corner[d]));
    mix(hash, encode_signed(row.big_corner[d]));
  }
  for (double value : row.values) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "checkpoint double must be 64-bit");
    memcpy(&bits, &value, sizeof(bits));
    mix(hash, bits);
  }
  return disk_hash(hash);
}

uint64_t checkpoint_storage_signature(const StoragePlan &plan, const CpuArrayCatalog &catalog) {
  if (plan.arrays.size() != plan.keys.size())
    throw std::invalid_argument("checkpoint storage plan has mismatched arrays and keys");
  uint64_t hash = UINT64_C(0xcbf29ce484222325);
  size_t persistent_rows = 0;
  for (size_t i = 0; i < plan.arrays.size(); ++i)
    persistent_rows += persistent_checkpoint_row(plan.arrays[i], plan.keys[i]);
  mix(hash, persistent_rows);
  for (size_t i = 0; i < plan.arrays.size(); ++i) {
    const ArraySpec &spec = plan.arrays[i];
    const StorageKey &key = plan.keys[i];
    if (!persistent_checkpoint_row(spec, key)) continue;
    mix(hash, encode_signed(key.chunk));
    mix(hash, encode_signed(key.kind));
    mix(hash, encode_signed(key.component_));
    mix(hash, encode_signed(key.cmp));
    mix(hash, key.aux);
    mix(hash, uint64_t(spec.role));
    mix(hash, uint64_t(spec.element_type));
    mix(hash, uint64_t(spec.storage));
    mix(hash, spec.elements);
    mix(hash, spec.alignment);
    mix(hash, is_valid(spec.alias_of) ? 1 : 0);
    if (is_valid(spec.alias_of)) {
      if (spec.alias_of.value >= catalog.size())
        throw std::invalid_argument("checkpoint storage alias is out of range");
      const StorageKey &alias = catalog.key(spec.alias_of);
      mix(hash, encode_signed(alias.chunk));
      mix(hash, encode_signed(alias.kind));
      mix(hash, encode_signed(alias.component_));
      mix(hash, encode_signed(alias.cmp));
      mix(hash, alias.aux);
    }
  }
  return disk_hash(hash);
}

void checkpoint_set_failure_for_testing(CheckpointFailurePoint point, int rank) {
  failure_point = point;
  failure_rank = rank;
}

void checkpoint_set_secondary_failure_for_testing(CheckpointFailurePoint point, int rank) {
  secondary_failure_point = point;
  secondary_failure_rank = rank;
}

void checkpoint_clear_failure_for_testing() {
  failure_point = CheckpointFailurePoint::none;
  failure_rank = -1;
  secondary_failure_point = CheckpointFailurePoint::none;
  secondary_failure_rank = -1;
}

void checkpoint_fail_if_requested(CheckpointFailurePoint point) {
  if (failure_point == point) {
    const bool selected = failure_rank < 0 || failure_rank == my_rank();
    failure_point = CheckpointFailurePoint::none;
    if (selected) throw std::runtime_error("injected checkpoint transaction failure");
  }
  if (secondary_failure_point == point) {
    const bool selected = secondary_failure_rank < 0 || secondary_failure_rank == my_rank();
    secondary_failure_point = CheckpointFailurePoint::none;
    if (selected) throw std::runtime_error("injected secondary checkpoint transaction failure");
  }
}

void CheckpointTransaction::validate_eligible(fields &owner, const char *operation) {
  std::string error;
  if (owner.backend && owner.backend->is_poisoned())
    error = "checkpoint is unavailable after backend poison";
  else if (owner.synchronized_magnetic_fields)
    error = "checkpoint is unavailable while magnetic synchronization is active";
  else if (owner.phasein_time > 0)
    error = "checkpoint is unavailable during material phasing";
  for (int chunk = 0; error.empty() && chunk < owner.num_chunks; ++chunk)
    if (owner.chunks[chunk]->is_solving_cw())
      error = "checkpoint is unavailable while solve_cw is active";
  for (const src_time *source = owner.sources; error.empty() && source; source = source->next)
    if (!exact_builtin_source_time(source))
      error = "checkpoint requires a declared layout for custom source-time state";
  for (int chunk = 0; error.empty() && chunk < owner.num_chunks; ++chunk)
    FOR_FIELD_TYPES(ft)
      for (polarization_state *state = owner.chunks[chunk]->pol[ft]; state;
           state = state->next)
        if (state->s && classify_susceptibility(state->s) == SusceptibilityKind::host_custom) {
          error = "checkpoint requires a declared layout for custom polarization state";
          break;
        }
  backend_reconcile_host_access(error, operation);
}

CheckpointImage CheckpointTransaction::snapshot(fields &owner) {
  validate_eligible(owner, "fields::dump checkpoint eligibility");
  if (owner.backend_state)
    backend_preflight_field_layout_change(owner, DirtyMask(dirty_storage),
                                          "fields::dump checkpoint snapshot");
  CpuArrayCatalog catalog;
  StoragePlan plan;
  build_storage_catalog(owner, catalog, plan);
  CheckpointImage image;
  image.dimension = uint64_t(owner.gv.dim);
  image.configuration_signature = checkpoint_configuration_signature(owner);
  image.storage_signature = checkpoint_storage_signature(plan, catalog);
  /* MaterialIR::signature deliberately excludes rank/chunk topology. This is
     the definition identity required for safe repartition; legacy construction
     paths fall back to an exact-layout host recipe and therefore reject a
     layout-changing load instead of silently skipping material validation. */
  const RecipeFingerprints material =
      checkpoint_material_definition_signatures(owner, &plan, &catalog);
  image.material_recipe_signature = material.portable;
  image.material_native_signature = material.native;
  image.classification_hash = disk_hash(owner.prepared_classification_hash);
  image.source_definition_signature = checkpoint_source_definition_signature(owner);
  const RecipeFingerprints dft = checkpoint_dft_recipe_signatures(owner);
  image.dft_recipe_signature = dft.portable;
  image.dft_native_signature = dft.native;
  image.saved_rank_count = uint64_t(count_processors());
  image.timestep = owner.t;
  image.dt = owner.dt;
  for (int i = 0; i < 3; ++i) image.source_times[i] = owner.step_source_times[i];
  for (int i = 0; i < fields::num_mutation_kinds; ++i)
    image.mutation_generation[i] = owner.mutation_generation[i];
  image.random_seed = random_seed_snapshot();
  image.random_seed_ranks.push_back(uint32_t(my_global_rank()));
  image.random_seeds.push_back(image.random_seed);

  image.rows.reserve(plan.arrays.size());
  for (size_t i = 0; i < plan.arrays.size(); ++i) {
    const ArraySpec &spec = plan.arrays[i];
    if (!persistent_checkpoint_row(spec, plan.keys[i])) continue;
    CheckpointRow row;
    row.key = plan.keys[i];
    row.spec = spec;
    row.spec.id = invalid_array();
    row.spec.alias_of = invalid_array();
    row.has_alias = is_valid(spec.alias_of);
    if (row.has_alias) row.alias_key = catalog.key(spec.alias_of);
    set_row_corners(row, owner);
    if (!row.has_alias) {
      const void *address = catalog.resolve_untyped(ArrayId{uint32_t(i)});
      if (!address) throw std::runtime_error("checkpoint found a null persistent host row");
      copy_host_row(address, row);
    }
    else
      row.checksum = checkpoint_row_checksum(row);
    image.rows.push_back(std::move(row));
  }

  std::vector<LegacyFluxDescriptor> flux_descriptors;
  build_legacy_flux_descriptors(owner, flux_descriptors);
  flux_vol *flux = owner.fluxes;
  for (size_t i = 0; i < flux_descriptors.size(); ++i) {
    if (!flux) throw std::logic_error("checkpoint legacy flux descriptor count mismatch");
    image.legacy_flux_signatures.push_back(disk_hash(flux_descriptors[i].recipe_signature));
    image.legacy_flux_values.push_back(flux->cur_flux);
    image.legacy_flux_values.push_back(flux->cur_flux_half);
    flux = flux->next;
  }
  if (flux) throw std::logic_error("checkpoint omitted a legacy flux accumulator");
  /* Failure injection belongs at the end of rank-local staging. Placing it
     before catalog/descriptor construction lets one rank jump to the outer
     reconciliation while peers are still entering their normal collectives. */
  checkpoint_fail_if_requested(CheckpointFailurePoint::snapshot);
  return image;
}

void CheckpointTransaction::write_manifest(h5file &file, const CheckpointImage &image,
                                           bool single_parallel_file) {
  const size_t local_rows = image.rows.size();
  size_t local_values = 0;
  for (const CheckpointRow &row : image.rows) local_values += row.values.size();
  const size_t row_start = single_parallel_file ? partial_sum_to_all(local_rows) - local_rows : 0;
  const size_t value_start =
      single_parallel_file ? partial_sum_to_all(local_values) - local_values : 0;
  const size_t total_rows = single_parallel_file ? sum_to_all(local_rows) : local_rows;
  const size_t total_values = single_parallel_file ? sum_to_all(local_values) : local_values;

  size_t header[19] = {size_t(image.schema_magic), size_t(image.schema_version),
                       size_t(image.endian_marker), size_t(image.host_realnum_bytes),
                       size_t(image.dimension), size_t(image.configuration_signature),
                       size_t(image.storage_signature), size_t(image.material_recipe_signature),
                       size_t(image.classification_hash), size_t(image.timestep), total_rows,
                       total_values, image.legacy_flux_signatures.size(),
                       fields::num_mutation_kinds, size_t(image.source_definition_signature),
                       size_t(image.dft_recipe_signature), size_t(image.saved_rank_count),
                       size_t(image.material_native_signature),
                       size_t(image.dft_native_signature)};
  size_t one = 19, zero = 0;
  file.create_data(checkpoint_header_name, 1, &one, false, false);
  if (am_master() || !single_parallel_file) file.write_chunk(1, &zero, &one, header);

  std::vector<size_t> metadata(local_rows * checkpoint_row_columns, 0);
  std::vector<double> values;
  values.reserve(local_values);
  size_t value_offset = value_start;
  for (size_t row_index = 0; row_index < local_rows; ++row_index) {
    const CheckpointRow &row = image.rows[row_index];
    size_t *m = &metadata[row_index * checkpoint_row_columns];
    m[0] = size_t(encode_signed(row.key.chunk));
    m[1] = size_t(encode_signed(row.key.kind));
    m[2] = size_t(encode_signed(row.key.component_));
    m[3] = size_t(encode_signed(row.key.cmp));
    m[4] = size_t(row.key.aux);
    m[5] = size_t(row.spec.role);
    m[6] = size_t(row.spec.element_type);
    m[7] = size_t(row.spec.storage);
    m[8] = row.spec.elements;
    m[9] = row.spec.alignment;
    m[10] = row.has_alias;
    m[11] = size_t(encode_signed(row.alias_key.chunk));
    m[12] = size_t(encode_signed(row.alias_key.kind));
    m[13] = size_t(encode_signed(row.alias_key.component_));
    m[14] = size_t(encode_signed(row.alias_key.cmp));
    m[15] = size_t(row.alias_key.aux);
    m[16] = value_offset;
    m[17] = row.values.size();
    m[18] = size_t(row.checksum);
    for (int d = 0; d < 3; ++d) {
      m[19 + d] = size_t(encode_signed(row.little_corner[d]));
      m[22 + d] = size_t(encode_signed(row.big_corner[d]));
    }
    m[25] = (row.spec.classification_provisional ? 1u : 0u) |
            (row.spec.classification_elided ? 2u : 0u);
    values.insert(values.end(), row.values.begin(), row.values.end());
    value_offset += row.values.size();
  }
  size_t row_dims[2] = {total_rows, checkpoint_row_columns};
  size_t row_chunk_start[2] = {row_start, 0};
  size_t row_chunk_dims[2] = {local_rows, checkpoint_row_columns};
  file.create_data(checkpoint_rows_name, 2, row_dims, false, false);
  if (local_rows) file.write_chunk(2, row_chunk_start, row_chunk_dims, metadata.data());
  file.done_writing_chunks();

  file.create_data(checkpoint_values_name, 1, &total_values, false, false);
  if (local_values) file.write_chunk(1, &value_start, &local_values, values.data());
  file.done_writing_chunks();

  double scalars[4] = {image.dt, image.source_times[0], image.source_times[1],
                       image.source_times[2]};
  size_t scalar_count = 4;
  file.create_data(checkpoint_scalars_name, 1, &scalar_count, false, false);
  if (am_master() || !single_parallel_file)
    file.write_chunk(1, &zero, &scalar_count, scalars);

  std::vector<size_t> generations(2 * fields::num_mutation_kinds);
  for (int i = 0; i < fields::num_mutation_kinds; ++i) {
    generations[2 * size_t(i)] = size_t(uint32_t(image.mutation_generation[i]));
    generations[2 * size_t(i) + 1] =
        size_t(uint32_t(image.mutation_generation[i] >> 32));
  }
  size_t generation_count = generations.size();
  file.create_data(checkpoint_generations_name, 1, &generation_count, false, false);
  if (am_master() || !single_parallel_file)
    file.write_chunk(1, &zero, &generation_count, generations.data());

  const size_t local_seed_rows = image.random_seeds.size();
  if (local_seed_rows != image.random_seed_ranks.size())
    throw std::invalid_argument("checkpoint random seed rank table mismatch");
  const size_t seed_row_start =
      single_parallel_file ? partial_sum_to_all(local_seed_rows) - local_seed_rows : 0;
  const size_t total_seed_rows =
      single_parallel_file ? sum_to_all(local_seed_rows) : local_seed_rows;
  std::vector<size_t> seed(local_seed_rows * 11);
  for (size_t i = 0; i < local_seed_rows; ++i) {
    const RandomSeedSnapshot &state = image.random_seeds[i];
    size_t *row = &seed[i * 11];
    row[0] = image.random_seed_ranks[i];
    row[1] = state.semantic_seed;
    row[2] = state.saved_semantic_seed;
    row[3] = size_t(uint32_t(state.generation));
    row[4] = size_t(uint32_t(state.generation >> 32));
    row[5] = state.algorithm_version;
    row[6] = state.initialized;
    row[7] = state.semantic_seed_valid;
    row[8] = state.saved_semantic_seed_valid;
    row[9] = state.explicit_seed;
    row[10] = state.saved_explicit_seed;
  }
  size_t seed_dims[2] = {total_seed_rows, 11};
  size_t seed_start[2] = {seed_row_start, 0};
  size_t seed_chunk[2] = {local_seed_rows, 11};
  file.create_data(checkpoint_seed_name, 2, seed_dims, false, false);
  if (local_seed_rows) file.write_chunk(2, seed_start, seed_chunk, seed.data());
  file.done_writing_chunks();

  size_t flux_count = image.legacy_flux_signatures.size();
  std::vector<size_t> flux_keys(flux_count);
  for (size_t i = 0; i < flux_count; ++i) flux_keys[i] = size_t(image.legacy_flux_signatures[i]);
  file.create_data(checkpoint_flux_keys_name, 1, &flux_count, false, false);
  if (flux_count && (am_master() || !single_parallel_file))
    file.write_chunk(1, &zero, &flux_count, flux_keys.data());
  size_t flux_value_count = image.legacy_flux_values.size();
  file.create_data(checkpoint_flux_values_name, 1, &flux_value_count, false, false);
  if (flux_value_count && (am_master() || !single_parallel_file))
    file.write_chunk(1, &zero, &flux_value_count,
                     const_cast<double *>(image.legacy_flux_values.data()));
}

bool CheckpointTransaction::has_manifest(h5file &file) {
  return file.dataset_exists(checkpoint_header_name);
}

CheckpointImage CheckpointTransaction::read_manifest(h5file &file, bool single_parallel_file) {
  CheckpointImage image;
  image.shared_manifest = single_parallel_file;
  int rank = 0;
  size_t header_count = 0;
  file.read_size(checkpoint_header_name, &rank, &header_count, 1);
  if (rank != 1 || header_count != 19)
    throw std::invalid_argument("checkpoint header has the wrong extent");
  size_t zero = 0;
  size_t header[19];
  file.read_chunk(1, &zero, &header_count, header);
  image.schema_magic = header[0];
  image.schema_version = header[1];
  image.endian_marker = header[2];
  image.host_realnum_bytes = header[3];
  image.dimension = header[4];
  image.configuration_signature = header[5];
  image.storage_signature = header[6];
  image.material_recipe_signature = header[7];
  image.classification_hash = header[8];
  if (header[9] > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("checkpoint timestep is outside the int domain");
  image.timestep = int(header[9]);
  const size_t row_count = header[10];
  const size_t value_count = header[11];
  const size_t flux_count = header[12];
  if (header[13] != fields::num_mutation_kinds)
    throw std::invalid_argument("checkpoint mutation-generation extent mismatch");
  image.source_definition_signature = header[14];
  image.dft_recipe_signature = header[15];
  image.saved_rank_count = header[16];
  image.material_native_signature = header[17];
  image.dft_native_signature = header[18];

  size_t row_dims[2] = {0, 0};
  file.read_size(checkpoint_rows_name, &rank, row_dims, 2);
  if (rank != 2 || row_dims[0] != row_count || row_dims[1] != checkpoint_row_columns)
    throw std::invalid_argument("checkpoint row table has the wrong extent");
  if (row_count > std::numeric_limits<size_t>::max() / checkpoint_row_columns)
    throw std::invalid_argument("checkpoint row table extent overflows host size");
  std::vector<size_t> metadata(row_count * checkpoint_row_columns);
  size_t row_start[2] = {0, 0};
  if (row_count) file.read_chunk(2, row_start, row_dims, metadata.data());

  size_t stored_values = 0;
  file.read_size(checkpoint_values_name, &rank, &stored_values, 1);
  if (rank != 1 || stored_values != value_count)
    throw std::invalid_argument("checkpoint value table has the wrong extent");
  std::vector<double> values(value_count);
  if (value_count) file.read_chunk(1, &zero, &value_count, values.data());

  image.rows.resize(row_count);
  for (size_t row_index = 0; row_index < row_count; ++row_index) {
    const size_t *m = &metadata[row_index * checkpoint_row_columns];
    CheckpointRow &row = image.rows[row_index];
    row.key = {decode_signed(m[0]), decode_signed(m[1]), decode_signed(m[2]),
               decode_signed(m[3]), uint64_t(m[4])};
    if (!valid_checkpoint_key(row.key))
      throw std::invalid_argument("checkpoint row has an invalid storage-key domain");
    if (m[5] > size_t(array_role::scratch) || m[6] > size_t(ElementType::index) ||
        m[7] > size_t(Precision::f64) || m[10] > 1 || (m[25] & ~size_t(3)) != 0)
      throw std::invalid_argument("checkpoint row has an invalid enum or flag domain");
    row.spec = {invalid_array(), array_role(m[5]), ElementType(m[6]), Precision(m[7]), m[8], m[9],
                invalid_array(), bool(m[25] & 1u), bool(m[25] & 2u)};
    row.has_alias = m[10] != 0;
    row.alias_key = {decode_signed(m[11]), decode_signed(m[12]), decode_signed(m[13]),
                     decode_signed(m[14]), uint64_t(m[15])};
    if (row.has_alias && !valid_checkpoint_key(row.alias_key))
      throw std::invalid_argument("checkpoint alias has an invalid storage-key domain");
    const size_t offset = m[16], count = m[17];
    if (offset > values.size() || count > values.size() - offset)
      throw std::invalid_argument("checkpoint row value range is out of bounds");
    row.values.assign(values.begin() + offset, values.begin() + offset + count);
    row.checksum = uint64_t(m[18]);
    for (int d = 0; d < 3; ++d) {
      row.little_corner[d] = decode_signed(m[19 + d]);
      row.big_corner[d] = decode_signed(m[22 + d]);
    }
    const uint64_t observed_checksum = checkpoint_row_checksum(row);
    if (observed_checksum != row.checksum)
      throw std::invalid_argument("checkpoint row checksum mismatch at row " +
                                  std::to_string(row_index) + " (stored " +
                                  std::to_string(row.checksum) + ", observed " +
                                  std::to_string(observed_checksum) + ")");
  }

  size_t scalar_count = 0;
  file.read_size(checkpoint_scalars_name, &rank, &scalar_count, 1);
  if (rank != 1 || scalar_count != 4)
    throw std::invalid_argument("checkpoint scalar table has the wrong extent");
  double scalars[4];
  file.read_chunk(1, &zero, &scalar_count, scalars);
  image.dt = scalars[0];
  for (int i = 0; i < 3; ++i) image.source_times[i] = scalars[i + 1];

  size_t generation_count = 0;
  file.read_size(checkpoint_generations_name, &rank, &generation_count, 1);
  if (rank != 1 || generation_count != 2 * fields::num_mutation_kinds)
    throw std::invalid_argument("checkpoint generation table has the wrong extent");
  std::vector<size_t> generations(generation_count);
  file.read_chunk(1, &zero, &generation_count, generations.data());
  for (int i = 0; i < fields::num_mutation_kinds; ++i) {
    if (generations[2 * size_t(i)] > size_t(std::numeric_limits<uint32_t>::max()) ||
        generations[2 * size_t(i) + 1] > size_t(std::numeric_limits<uint32_t>::max()))
      throw std::invalid_argument("checkpoint mutation generation word is out of range");
    image.mutation_generation[i] = uint64_t(generations[2 * size_t(i)]) |
                                   (uint64_t(generations[2 * size_t(i) + 1]) << 32);
  }

  size_t seed_dims[2] = {0, 0};
  file.read_size(checkpoint_seed_name, &rank, seed_dims, 2);
  if (rank != 2 || seed_dims[1] != 11 || seed_dims[0] == 0)
    throw std::invalid_argument("checkpoint random seed table has the wrong extent");
  if (seed_dims[0] > std::numeric_limits<size_t>::max() / seed_dims[1])
    throw std::invalid_argument("checkpoint random seed table extent overflows host size");
  std::vector<size_t> seed(seed_dims[0] * seed_dims[1]);
  size_t seed_start[2] = {0, 0};
  file.read_chunk(2, seed_start, seed_dims, seed.data());
  image.random_seed_ranks.resize(seed_dims[0]);
  image.random_seeds.resize(seed_dims[0]);
  for (size_t i = 0; i < seed_dims[0]; ++i) {
    const size_t *row = &seed[i * 11];
    if (row[0] > size_t(std::numeric_limits<uint32_t>::max()) ||
        row[1] > size_t(std::numeric_limits<uint32_t>::max()) ||
        row[2] > size_t(std::numeric_limits<uint32_t>::max()) ||
        row[3] > size_t(std::numeric_limits<uint32_t>::max()) ||
        row[4] > size_t(std::numeric_limits<uint32_t>::max()) ||
        row[5] > size_t(std::numeric_limits<uint32_t>::max()) || row[6] > 1 || row[7] > 1 ||
        row[8] > 1 || row[9] > 1 || row[10] > 1)
      throw std::invalid_argument("checkpoint random seed row has an out-of-range field");
    image.random_seed_ranks[i] = uint32_t(row[0]);
    RandomSeedSnapshot &state = image.random_seeds[i];
    state.semantic_seed = uint32_t(row[1]);
    state.saved_semantic_seed = uint32_t(row[2]);
    state.generation = uint64_t(row[3]) | (uint64_t(row[4]) << 32);
    state.algorithm_version = uint32_t(row[5]);
    state.initialized = row[6] != 0;
    state.semantic_seed_valid = row[7] != 0;
    state.saved_semantic_seed_valid = row[8] != 0;
    state.explicit_seed = row[9] != 0;
    state.saved_explicit_seed = row[10] != 0;
  }
  image.random_seed = image.random_seeds.front();

  size_t stored_flux_count = 0;
  file.read_size(checkpoint_flux_keys_name, &rank, &stored_flux_count, 1);
  if (rank != 1 || stored_flux_count != flux_count)
    throw std::invalid_argument("checkpoint flux key table has the wrong extent");
  std::vector<size_t> flux_keys(flux_count);
  if (flux_count) file.read_chunk(1, &zero, &flux_count, flux_keys.data());
  image.legacy_flux_signatures.assign(flux_keys.begin(), flux_keys.end());
  size_t flux_value_count = 0;
  file.read_size(checkpoint_flux_values_name, &rank, &flux_value_count, 1);
  if (rank != 1 || flux_value_count != 2 * flux_count)
    throw std::invalid_argument("checkpoint flux value table has the wrong extent");
  image.legacy_flux_values.resize(flux_value_count);
  if (flux_value_count)
    file.read_chunk(1, &zero, &flux_value_count, image.legacy_flux_values.data());
  (void)single_parallel_file;
  return image;
}

void CheckpointTransaction::validate_target(fields &owner, const CheckpointImage &image) {
  validate_eligible(owner, "fields::load checkpoint eligibility");
  std::string error;
  if (image.schema_magic != checkpoint_schema_magic ||
      image.schema_version != checkpoint_schema_version)
    error = "unsupported checkpoint schema";
  else if (image.endian_marker != UINT64_C(0x01020304))
    error = "checkpoint endian marker mismatch";
  else if (image.dimension != uint64_t(owner.gv.dim))
    error = "checkpoint dimension mismatch";
  else if (image.host_realnum_bytes != 4 && image.host_realnum_bytes != 8)
    error = "checkpoint host real precision is unsupported";
  else if (image.configuration_signature != checkpoint_configuration_signature(owner))
    error = "checkpoint global grid/boundary/symmetry configuration mismatch";
  else if (!std::isfinite(image.dt) || image.dt != owner.dt || image.timestep < 0)
    error = "checkpoint timestep metadata mismatch";
  std::unordered_map<StorageKey, const CheckpointRow *, StorageKeyHash> unique;
  const Precision saved_precision =
      image.host_realnum_bytes == 4 ? Precision::f32 : Precision::f64;
  try { for (const CheckpointRow &row : image.rows) {
    if (!unique.emplace(row.key, &row).second) {
      error = "checkpoint contains duplicate storage keys";
      break;
    }
    if (!valid_checkpoint_key(row.key) || (row.has_alias && !valid_checkpoint_key(row.alias_key))) {
      error = "checkpoint storage key is invalid";
      break;
    }
    if (uint64_t(row.spec.role) > uint64_t(array_role::scratch) ||
        (row.spec.element_type != ElementType::realnum_value &&
         row.spec.element_type != ElementType::complex_realnum) ||
        (row.spec.storage != Precision::f32 && row.spec.storage != Precision::f64)) {
      error = "checkpoint storage row has an unsupported type or precision";
      break;
    }
    if (!row.spec.elements || row.spec.storage != saved_precision ||
        row.spec.alignment != checkpoint_precision_alignment(row.spec.storage)) {
      error = "checkpoint storage row has an invalid extent or alignment";
      break;
    }
    if (row.has_alias && unique.count(row.alias_key) == 0) {
      /* Forward aliases are accepted here and checked after the full scan. */
    }
    if (row.has_alias && !row.values.empty()) {
      error = "checkpoint alias row unexpectedly owns values";
      break;
    }
    if (!row.has_alias && row.values.size() != checkpoint_scalar_count(row.spec)) {
      error = "checkpoint storage row has the wrong scalar count";
      break;
    }
    if (!row.has_alias)
      for (double value : row.values) (void)checked_checkpoint_scalar(value, row.spec.storage);
    if (checkpoint_row_checksum(row) != row.checksum) {
      error = "checkpoint storage row checksum mismatch";
      break;
    }
  } }
  catch (const std::exception &e) { error = e.what(); }
  catch (...) { error = "unknown checkpoint metadata validation failure"; }
  for (const CheckpointRow &row : image.rows)
    if (error.empty() && row.has_alias && unique.count(row.alias_key) == 0)
      error = "checkpoint alias target is missing";
  for (const CheckpointRow &row : image.rows) {
    if (!error.empty() || !row.has_alias) continue;
    const CheckpointRow *current = &row;
    size_t depth = 0;
    while (current->has_alias) {
      if (++depth > image.rows.size()) {
        error = "checkpoint alias graph contains a cycle";
        break;
      }
      const auto next = unique.find(current->alias_key);
      if (next == unique.end()) break;
      current = next->second;
    }
  }
  backend_reconcile_host_access(error, "fields::load checkpoint metadata validation");

  std::vector<LegacyFluxDescriptor> flux_descriptors;
  error.clear();
  try { build_legacy_flux_descriptors(owner, flux_descriptors); }
    catch (const std::exception &e) { error = e.what(); }
  if (error.empty() && flux_descriptors.size() != image.legacy_flux_signatures.size())
    error = "checkpoint legacy flux count mismatch";
  for (size_t i = 0; error.empty() && i < flux_descriptors.size(); ++i)
    if (disk_hash(flux_descriptors[i].recipe_signature) != image.legacy_flux_signatures[i])
      error = "checkpoint legacy flux recipe mismatch";
  if (error.empty() && image.legacy_flux_values.size() != 2 * flux_descriptors.size())
    error = "checkpoint legacy flux payload mismatch";
  if (error.empty() && image.random_seed.algorithm_version != counter_random_algorithm_version)
    error = "checkpoint random algorithm version mismatch";
  const RandomSeedSnapshot *selected_seed = NULL;
  if (error.empty()) selected_seed = checkpoint_seed_for_rank(image);
  if (error.empty() && !selected_seed)
    error = "checkpoint random seed rank mapping is missing or ambiguous";
  if (error.empty() && selected_seed->algorithm_version != counter_random_algorithm_version)
    error = "checkpoint random algorithm version mismatch";
  backend_reconcile_host_access(error, "fields::load checkpoint flux validation");

  error.clear();
  uint64_t local_source_signature = 0;
  RecipeFingerprints local_dft_signature = {0, 0};
  try {
    local_source_signature = checkpoint_source_definition_signature(owner);
    local_dft_signature = checkpoint_dft_recipe_signatures(owner);
  }
  catch (const std::exception &e) { error = e.what(); }
  catch (...) { error = "unknown checkpoint recipe fingerprint failure"; }
  backend_reconcile_host_access(error, "fields::load checkpoint recipe staging");
  const uint64_t source_signature =
      disk_hash(uint64_t(sum_to_all(size_t(local_source_signature))));
  const uint64_t dft_portable_signature =
      disk_hash(uint64_t(sum_to_all(size_t(local_dft_signature.portable))));
  const uint64_t dft_native_signature =
      disk_hash(uint64_t(sum_to_all(size_t(local_dft_signature.native))));
  const bool same_precision = image.host_realnum_bytes == sizeof(realnum);
  if (source_signature != image.source_definition_signature)
    error = "checkpoint source definition recipe mismatch";
  else if ((same_precision && dft_native_signature != image.dft_native_signature) ||
           (!same_precision && dft_portable_signature != image.dft_recipe_signature))
    error = "checkpoint DFT monitor recipe mismatch";
  backend_reconcile_host_access(error, "fields::load checkpoint recipe validation");

  error.clear();
  const int layout_index = static_cast<int>(MutationKind::field_layout);
  if (error.empty() &&
      std::max(owner.mutation_generation[layout_index], image.mutation_generation[layout_index]) ==
          std::numeric_limits<uint64_t>::max())
    error = "checkpoint field-layout generation overflow";
  if (error.empty() &&
      (owner.connections_generation == std::numeric_limits<uint64_t>::max() ||
       owner.local_invalidation_generation == std::numeric_limits<uint64_t>::max()))
    error = "checkpoint topology generation overflow";
  RecipeFingerprints local_material_signature = {0, 0};
  try { local_material_signature = checkpoint_material_definition_signatures(owner); }
  catch (const std::exception &e) { error = e.what(); }
  catch (...) { error = "unknown checkpoint material recipe validation failure"; }
  backend_reconcile_host_access(error, "fields::load checkpoint material recipe staging");
  const uint64_t material_portable_signature =
      disk_hash(uint64_t(sum_to_all(size_t(local_material_signature.portable))));
  const uint64_t material_native_signature =
      disk_hash(uint64_t(sum_to_all(size_t(local_material_signature.native))));
  const uint64_t saved_material_signature =
      same_precision ? image.material_native_signature : image.material_recipe_signature;
  const uint64_t current_material_signature =
      same_precision ? material_native_signature : material_portable_signature;
  if (!saved_material_signature || current_material_signature != saved_material_signature)
    error = "checkpoint material recipe signature mismatch saved=" +
            std::to_string(saved_material_signature) + " current=" +
            std::to_string(current_material_signature);
  backend_reconcile_host_access(error, "fields::load checkpoint topology validation");

  error.clear();
  if (error.empty()) try { checkpoint_fail_if_requested(CheckpointFailurePoint::validation); }
    catch (const std::exception &e) { error = e.what(); }
    catch (...) { error = "unknown checkpoint validation failure"; }
  backend_reconcile_host_access(error, "fields::load checkpoint validation");
}

void CheckpointTransaction::commit(fields &owner, const CheckpointImage &image) {
  std::unique_ptr<PreparedCheckpointCommit> prepared;
  std::string error;
  bool local_exact_layout = false;
  try { local_exact_layout = checkpoint_local_exact_layout(owner, image); }
  catch (const std::exception &e) { error = e.what(); }
  catch (...) { error = "unknown checkpoint layout classification failure"; }
  backend_reconcile_host_access(error, "fields::load checkpoint layout classification");
  const bool exact_layout = image.saved_rank_count == uint64_t(count_processors()) &&
                            and_to_all(local_exact_layout);
  error.clear();
  try {
    prepared.reset(new PreparedCheckpointCommit(owner, image, exact_layout));
    const int layout_index = static_cast<int>(MutationKind::field_layout);
    if (std::max(owner.mutation_generation[layout_index],
                 image.mutation_generation[layout_index]) ==
            std::numeric_limits<uint64_t>::max() ||
        owner.checkpoint_publication_generation == std::numeric_limits<uint64_t>::max() ||
        owner.connections_generation == std::numeric_limits<uint64_t>::max() ||
        owner.local_invalidation_generation == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("checkpoint publication generation overflow");
    checkpoint_fail_if_requested(CheckpointFailurePoint::precommit);
  }
  catch (const std::exception &e) { error = e.what(); }
  catch (...) { error = "unknown checkpoint commit preparation failure"; }
  backend_reconcile_host_access(error, "fields::load checkpoint precommit");

  std::vector<size_t> global_coverage(prepared->covered_sources());
  /* Shared-file readers all see the same global row table and reconcile which
     rank consumed each row. Sharded readers see different local row counts,
     so their complete local coverage is checked before one scalar error gate. */
  if (image.shared_manifest && !image.rows.empty())
    sum_to_all(prepared->covered_sources().data(), global_coverage.data(), int(image.rows.size()));
  error.clear();
  for (size_t i = 0; i < global_coverage.size(); ++i)
    if (global_coverage[i] == 0) {
      error = std::string("checkpoint saved persistent row is not representable by the target: ") +
              array_kind_name(array_kind(image.rows[i].key.kind)) + "/" +
              std::to_string(image.rows[i].key.component_) + "/" +
              std::to_string(image.rows[i].key.cmp) + "/" +
              std::to_string(image.rows[i].key.aux) + " chunk=" +
              std::to_string(image.rows[i].key.chunk);
      break;
    }
  backend_reconcile_host_access(error, "fields::load checkpoint saved-row coverage");

  /* This is the final fallible boundary. It materializes any last resident
     authority and retires graph/device owners only after every staged row and
     allocation has validated. Publication below is pointer swap/memcpy only. */
  backend_prepare_checkpoint_load(owner);
  prepared->publish();
  /* The staged rows replace every host allocation identity. Publish the
     corresponding topology invalidation as part of that same no-fail epoch
     transition, before any observer can rebuild a pointer-bearing halo or
     metal-zero plan from the replacement image. Capacity for both checked
     increments was admitted by the final collective gate above. */
  note_connections_invalidated(owner);
  mark_local_invalidation(owner);
  clear_pending_material_region(owner);
  owner.chunk_connections_valid = false;
  owner.changed_materials = true;
  owner.t = image.timestep;
  owner.step_source_times[0] = image.source_times[0];
  owner.step_source_times[1] = image.source_times[1];
  owner.step_source_times[2] = image.source_times[2];
  for (int i = 0; i < fields::num_mutation_kinds; ++i)
    owner.mutation_generation[i] =
        std::max(owner.mutation_generation[i], image.mutation_generation[i]);
  ++owner.checkpoint_publication_generation;
  owner.calc_sources(owner.time());
  restore_random_seed_snapshot(*checkpoint_seed_for_rank(image));
  flux_vol *flux = owner.fluxes;
  for (size_t i = 0; i < image.legacy_flux_signatures.size(); ++i) {
    flux->cur_flux = image.legacy_flux_values[2 * i];
    flux->cur_flux_half = image.legacy_flux_values[2 * i + 1];
    flux = flux->next;
  }
}

} // namespace meep
