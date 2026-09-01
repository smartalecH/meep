/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#include <stdio.h>
#include <climits>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "backend/classification.hpp"
#include "backend/backend.hpp"
#include "backend/initialization_plan.hpp"
#include "backend/lifecycle.hpp"
#include "backend/material_recipe.hpp"
#include "backend/step_plan.hpp"
#include "meep_internals.hpp"

namespace meep {

static int material_classification_failure_rank_for_testing = -1;
static int material_classification_failure_mode_for_testing = 0;

void backend_set_material_classification_failure_for_testing(int rank, int mode) {
  material_classification_failure_rank_for_testing = rank;
  material_classification_failure_mode_for_testing = mode;
}

static uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + UINT64_C(0x9e3779b97f4a7c15) + (h << 6) + (h >> 2);
  return h;
}

namespace {

const size_t material_variant_words = 25;

struct StorageKeyLess {
  bool operator()(const StorageKey &a, const StorageKey &b) const {
    if (a.chunk != b.chunk) return a.chunk < b.chunk;
    if (a.kind != b.kind) return a.kind < b.kind;
    if (a.component_ != b.component_) return a.component_ < b.component_;
    if (a.cmp != b.cmp) return a.cmp < b.cmp;
    return a.aux < b.aux;
  }
};

bool retained(const std::map<StorageKey, uint8_t, StorageKeyLess> &rows,
              const StorageKey &key) {
  const auto found = rows.find(key);
  return found != rows.end() && found->second == MaterialClassification::retained;
}

uint32_t expected_variant(const fields &f, int chunk, field_type ft,
                          const std::map<StorageKey, uint8_t, StorageKeyLess> &rows) {
  uint32_t result = 0;
  bool anisotropic = false;
  FOR_FT_COMPONENTS(ft, c) {
    const direction dc = component_direction(c);
    const direction d1 = cycle_direction(f.chunks[chunk]->gv.dim, dc, 1);
    const direction d2 = cycle_direction(f.chunks[chunk]->gv.dim, dc, 2);
    anisotropic = anisotropic ||
                  (retained(rows, {chunk, int(array_kind::chi1inv), int(c), -1, uint64_t(d1)}) &&
                   retained(rows, {chunk, int(array_kind::chi1inv), int(c), -1, uint64_t(d2)}));
    if (retained(rows, {chunk, int(array_kind::chi2), int(c), -1, 0}))
      result |= material_variant_chi2;
    if (retained(rows, {chunk, int(array_kind::chi3), int(c), -1, 0}))
      result |= material_variant_chi3;
    for (int d = 0; d < 5; ++d) {
      if (retained(rows, {chunk, int(array_kind::conductivity), int(c), -1, uint64_t(d)}))
        result |= material_variant_conductivity;
      for (const auto &entry : rows) {
        const StorageKey &key = entry.first;
        if (entry.second == MaterialClassification::retained && key.chunk == chunk &&
            key.kind == int(array_kind::sigma) && key.component_ == int(c) && key.cmp == d)
          result |= material_variant_sigma;
      }
    }
  }
  for (int d = 0; d < 6; ++d)
    if (retained(rows, {chunk, int(array_kind::pml_sig), -1, -1, uint64_t(d)}))
      result |= material_variant_pml;
  if (anisotropic) result |= material_variant_anisotropic;
  return result;
}

MaterialVariantClassificationFact variant_fact(uint32_t operation, uint32_t region,
                                                field_type ft,
                                                const UpdateRegion &update) {
  MaterialVariantClassificationFact fact = {};
  fact.operation = operation;
  fact.region = region;
  fact.chunk = update.chunk;
  fact.field_type_ = int(ft);
  fact.component_ = int(update.c);
  fact.cmp = update.cmp;
  fact.dimension = int(update.begin.dim);
  for (int axis = 0; axis < 5; ++axis) {
    fact.begin[axis] = update.begin.in_direction(direction(axis));
    fact.end[axis] = update.end.in_direction(direction(axis));
  }
  fact.base = update.base;
  for (int axis = 0; axis < 3; ++axis) {
    fact.counts[axis] = update.counts[axis];
    fact.strides[axis] = update.strides[axis];
  }
  fact.variant_key = update.variant_key;
  return fact;
}

bool same_variant_fact(const MaterialVariantClassificationFact &a,
                       const MaterialVariantClassificationFact &b) {
  if (a.operation != b.operation || a.region != b.region || a.chunk != b.chunk ||
      a.field_type_ != b.field_type_ || a.component_ != b.component_ || a.cmp != b.cmp ||
      a.dimension != b.dimension || a.base != b.base || a.variant_key != b.variant_key)
    return false;
  for (int axis = 0; axis < 5; ++axis)
    if (a.begin[axis] != b.begin[axis] || a.end[axis] != b.end[axis]) return false;
  for (int axis = 0; axis < 3; ++axis)
    if (a.counts[axis] != b.counts[axis] || a.strides[axis] != b.strides[axis]) return false;
  return true;
}

struct MaterialVariantFactLess {
  bool operator()(const MaterialVariantClassificationFact &a,
                  const MaterialVariantClassificationFact &b) const {
    if (a.operation != b.operation) return a.operation < b.operation;
    if (a.chunk != b.chunk) return a.chunk < b.chunk;
    return a.region < b.region;
  }
};

std::vector<MaterialVariantClassificationFact>
variant_facts_from_step_plan(const StepPlan &steps) {
  std::vector<MaterialVariantClassificationFact> facts;
  uint32_t operation = 0;
  for (const Operation &op : steps.operations) {
    if (op.kind != OpKind::update_eh) continue;
    if (size_t(op.descriptor_index) + size_t(op.descriptor_count) > steps.eh_updates.size())
      throw std::invalid_argument("material classification encountered an invalid update_eh span");
    std::map<int, uint32_t> next_region;
    for (uint32_t i = 0; i < op.descriptor_count; ++i) {
      const ConstitutiveUpdate &update = steps.eh_updates[size_t(op.descriptor_index) + i];
      facts.push_back(variant_fact(operation, next_region[update.region.chunk]++, op.ft,
                                   update.region));
    }
    ++operation;
  }
  return facts;
}

MaterialClassification classification_from_rows(
    const fields &f, const StoragePlan &plan,
    const std::map<StorageKey, uint8_t, StorageKeyLess> &rows);
std::vector<MaterialVariantClassificationFact> expected_variant_facts(
    fields &f, const StoragePlan &plan, const MaterialRecipe &recipe,
    const MaterialClassification &classification);

std::vector<size_t> serialize_facts(const MaterialClassificationFacts &facts) {
  if (facts.rows.size() > (std::numeric_limits<size_t>::max() - 8) / 6 ||
      facts.variants.size() >
          (std::numeric_limits<size_t>::max() - 8 - 6 * facts.rows.size()) /
              material_variant_words)
    throw std::overflow_error("material classification fact serialization overflow");
  std::vector<size_t> words;
  words.reserve(8 + 6 * facts.rows.size() + material_variant_words * facts.variants.size());
  words.push_back(facts.version);
  words.push_back(facts.rows.size());
  words.push_back(facts.variants.size());
  words.push_back(size_t(facts.required_components));
  words.push_back(facts.has_nonlinearities ? 1 : 0);
  words.push_back(facts.aniso2d ? 1 : 0);
  words.push_back(size_t(facts.min_decimation_factor));
  words.push_back(0);
  for (const MaterialRowClassificationFact &row : facts.rows) {
    words.push_back(size_t(int64_t(row.key.chunk)));
    words.push_back(size_t(int64_t(row.key.kind)));
    words.push_back(size_t(int64_t(row.key.component_)));
    words.push_back(size_t(int64_t(row.key.cmp)));
    words.push_back(size_t(row.key.aux));
    words.push_back(size_t(row.state));
  }
  for (const MaterialVariantClassificationFact &variant : facts.variants) {
    words.push_back(size_t(variant.operation));
    words.push_back(size_t(variant.region));
    words.push_back(size_t(int64_t(variant.chunk)));
    words.push_back(size_t(int64_t(variant.field_type_)));
    words.push_back(size_t(int64_t(variant.component_)));
    words.push_back(size_t(int64_t(variant.cmp)));
    words.push_back(size_t(int64_t(variant.dimension)));
    for (int axis = 0; axis < 5; ++axis)
      words.push_back(size_t(int64_t(variant.begin[axis])));
    for (int axis = 0; axis < 5; ++axis)
      words.push_back(size_t(int64_t(variant.end[axis])));
    words.push_back(variant.base);
    for (int axis = 0; axis < 3; ++axis) words.push_back(variant.counts[axis]);
    for (int axis = 0; axis < 3; ++axis)
      words.push_back(size_t(int64_t(variant.strides[axis])));
    words.push_back(size_t(variant.variant_key));
  }
  return words;
}

MaterialClassificationFacts deserialize_facts(const std::vector<size_t> &words) {
  if (words.size() < 8)
    throw std::invalid_argument("material classification fact packet is truncated");
  const size_t rows = words[1], variants = words[2];
  if (rows > (std::numeric_limits<size_t>::max() - 8) / 6 ||
      variants > (std::numeric_limits<size_t>::max() - 8 - rows * 6) /
                     material_variant_words ||
      words.size() != 8 + rows * 6 + variants * material_variant_words || words[7] != 0)
    throw std::invalid_argument("material classification fact packet has an invalid extent");
  MaterialClassificationFacts facts;
  facts.version = uint32_t(words[0]);
  facts.required_components = component_mask(words[3]);
  if (words[4] > 1 || words[5] > 1 || words[6] > size_t(INT_MAX))
    throw std::invalid_argument("material classification scalar fact is invalid");
  facts.has_nonlinearities = words[4] != 0;
  facts.aniso2d = words[5] != 0;
  facts.min_decimation_factor = int(words[6]);
  size_t at = 8;
  for (size_t i = 0; i < rows; ++i, at += 6)
    facts.rows.push_back(MaterialRowClassificationFact{
        {int(int64_t(words[at])), int(int64_t(words[at + 1])),
         int(int64_t(words[at + 2])), int(int64_t(words[at + 3])), uint64_t(words[at + 4])},
        uint8_t(words[at + 5])});
  for (size_t i = 0; i < variants; ++i, at += material_variant_words) {
    MaterialVariantClassificationFact variant = {};
    variant.operation = uint32_t(words[at]);
    variant.region = uint32_t(words[at + 1]);
    variant.chunk = int(int64_t(words[at + 2]));
    variant.field_type_ = int(int64_t(words[at + 3]));
    variant.component_ = int(int64_t(words[at + 4]));
    variant.cmp = int(int64_t(words[at + 5]));
    variant.dimension = int(int64_t(words[at + 6]));
    for (int axis = 0; axis < 5; ++axis)
      variant.begin[axis] = int(int64_t(words[at + 7 + axis]));
    for (int axis = 0; axis < 5; ++axis)
      variant.end[axis] = int(int64_t(words[at + 12 + axis]));
    variant.base = words[at + 17];
    for (int axis = 0; axis < 3; ++axis) variant.counts[axis] = words[at + 18 + axis];
    for (int axis = 0; axis < 3; ++axis)
      variant.strides[axis] = ptrdiff_t(int64_t(words[at + 21 + axis]));
    variant.variant_key = uint32_t(words[at + 24]);
    facts.variants.push_back(variant);
  }
  return facts;
}

void validate_local_facts(fields &f, const StoragePlan &plan, const MaterialRecipe &recipe,
                          const MaterialClassificationFacts &facts) {
  if (facts.version != material_classification_facts_version)
    throw std::invalid_argument("material classification fact version is unsupported");
  if (facts.min_decimation_factor < 0)
    throw std::invalid_argument("material classification decimation fact is invalid");
  const component_mask allowed =
      NUM_FIELD_COMPONENTS >= int(8 * sizeof(component_mask))
          ? ~component_mask(0)
          : (component_mask(1) << NUM_FIELD_COMPONENTS) - 1;
  if (facts.required_components & ~allowed)
    throw std::invalid_argument("material classification component fact is invalid");
  std::map<StorageKey, uint8_t, StorageKeyLess> expected;
  for (size_t i = 0; i < plan.arrays.size(); ++i)
    if (plan.arrays[i].role == array_role::material)
      expected.insert(std::make_pair(plan.keys[i], uint8_t(0)));
  for (const MaterialRowClassificationFact &row : facts.rows) {
    const auto found = expected.find(row.key);
    if (found == expected.end() || found->second ||
        (row.state != MaterialClassification::retained &&
         row.state != MaterialClassification::elided_row))
      throw std::invalid_argument("material classification row facts are not total and unique");
    found->second = row.state;
  }
  for (const auto &entry : expected)
    if (!entry.second)
      throw std::invalid_argument("material classification omitted a local material row fact");
  std::set<std::tuple<uint32_t, int, uint32_t> > variants;
  const uint32_t allowed_variants = constitutive_one_offdiagonal |
                                    constitutive_two_offdiagonals |
                                    constitutive_has_pml |
                                    constitutive_has_nonlinearity |
                                    constitutive_has_minus_p |
                                    constitutive_copy_w_previous |
                                    constitutive_axis_override;
  MaterialVariantClassificationFact previous = {};
  bool have_previous = false;
  for (const MaterialVariantClassificationFact &variant : facts.variants) {
    if (variant.chunk < 0 || variant.chunk >= f.num_chunks ||
        (variant.field_type_ != E_stuff && variant.field_type_ != H_stuff) ||
        variant.component_ < 0 || variant.component_ >= NUM_FIELD_COMPONENTS ||
        variant.cmp < 0 || variant.cmp >= (f.is_real ? 1 : 2) ||
        variant.dimension < int(D1) || variant.dimension > int(Dcyl) ||
        (is_electric(component(variant.component_)) ? E_stuff : H_stuff) !=
            variant.field_type_ ||
        (variant.variant_key & ~allowed_variants) ||
        !f.chunks[variant.chunk]->is_mine() ||
        !variants.insert(std::make_tuple(variant.operation, variant.chunk, variant.region)).second)
      throw std::invalid_argument("material classification variant fact is malformed");
    if (have_previous && !MaterialVariantFactLess()(previous, variant))
      throw std::invalid_argument("material classification variant facts are out of order");
    for (int axis = 0; axis < 3; ++axis)
      if (!variant.counts[axis] || variant.strides[axis] < 0)
        throw std::invalid_argument("material classification region extent is malformed");
    previous = variant;
    have_previous = true;
  }
  const MaterialClassification local = classification_from_rows(f, plan, expected);
  const std::vector<MaterialVariantClassificationFact> canonical =
      expected_variant_facts(f, plan, recipe, local);
  if (facts.variants.size() != canonical.size())
    throw std::invalid_argument("material classification variant coverage is incomplete");
  for (size_t i = 0; i < canonical.size(); ++i)
    if (!same_variant_fact(facts.variants[i], canonical[i]))
      throw std::invalid_argument("material classification variant differs from canonical plan order");
}

std::vector<MaterialClassificationFacts> gather_facts(
    const MaterialClassificationFacts &local) {
  std::string local_error;
  std::vector<size_t> encoded;
  try {
    if (material_classification_failure_rank_for_testing == my_rank() &&
        material_classification_failure_mode_for_testing == 1)
      throw std::runtime_error("injected material classification serialization failure");
    encoded = serialize_facts(local);
    if (encoded.size() > size_t(INT_MAX))
      throw std::overflow_error("material classification fact packet exceeds MPI INT_MAX");
  }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown material classification serialization failure"; }
  backend_reconcile_host_access(local_error, "material classification serialization");

  std::vector<MaterialClassificationFacts> gathered;
  try { gathered.reserve(size_t(count_processors())); }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown material classification result allocation failure"; }
  backend_reconcile_host_access(local_error, "material classification result allocation");
  for (int rank = 0; rank < count_processors(); ++rank) {
    size_t count = my_rank() == rank ? encoded.size() : 0;
    broadcast(rank, &count, 1);
    try {
      if (count > size_t(INT_MAX))
        throw std::overflow_error("material classification gathered packet exceeds MPI INT_MAX");
    }
    catch (const std::exception &e) { local_error = e.what(); }
    catch (...) { local_error = "unknown material classification packet extent failure"; }
    backend_reconcile_host_access(local_error, "material classification packet extent");

    std::vector<size_t> packet;
    try {
      if (material_classification_failure_rank_for_testing == my_rank() &&
          material_classification_failure_mode_for_testing == 2)
        throw std::bad_alloc();
      packet.resize(count);
      if (my_rank() == rank) packet = encoded;
    }
    catch (const std::exception &e) { local_error = e.what(); }
    catch (...) { local_error = "unknown material classification packet allocation failure"; }
    backend_reconcile_host_access(local_error, "material classification packet allocation");

    if (count) broadcast(rank, packet.data(), int(count));
    try {
      if (material_classification_failure_rank_for_testing == my_rank() &&
          material_classification_failure_mode_for_testing == 3)
        throw std::runtime_error("injected material classification packet failure");
    }
    catch (const std::exception &e) { local_error = e.what(); }
    catch (...) { local_error = "unknown material classification packet failure"; }
    backend_reconcile_host_access(local_error, "material classification packet transfer");

    try {
      if (material_classification_failure_rank_for_testing == my_rank() &&
          material_classification_failure_mode_for_testing == 4)
        throw std::runtime_error("injected material classification deserialization failure");
      gathered.push_back(deserialize_facts(packet));
    }
    catch (const std::exception &e) { local_error = e.what(); }
    catch (...) { local_error = "unknown material classification deserialization failure"; }
    backend_reconcile_host_access(local_error, "material classification deserialization");
  }
  return gathered;
}

void validate_group_topology(const fields &f,
                             const std::map<StorageKey, uint8_t, StorageKeyLess> &rows) {
  typedef std::pair<int, int> ChunkComponent;
  std::set<ChunkComponent> tensor_groups;
  std::set<std::pair<int, uint64_t> > pml_groups;
  struct SigmaGroup {
    int chunk;
    int component;
    uint64_t identity;
    bool operator<(const SigmaGroup &other) const {
      if (chunk != other.chunk) return chunk < other.chunk;
      if (component != other.component) return component < other.component;
      return identity < other.identity;
    }
  };
  std::set<SigmaGroup> sigma_groups;
  for (const auto &entry : rows) {
    const StorageKey &key = entry.first;
    const array_kind kind = static_cast<array_kind>(key.kind);
    if (kind == array_kind::chi1inv || kind == array_kind::chi2 ||
        kind == array_kind::chi3 || kind == array_kind::conductivity ||
        kind == array_kind::condinv)
      tensor_groups.insert(ChunkComponent(key.chunk, key.component_));
    if (kind == array_kind::pml_sig || kind == array_kind::pml_kap ||
        kind == array_kind::pml_siginv)
      pml_groups.insert(std::make_pair(key.chunk, key.aux));
    if (kind == array_kind::sigma)
      sigma_groups.insert(SigmaGroup{key.chunk, key.component_, key.aux});
  }
  for (const ChunkComponent &group : tensor_groups) {
    const int chunk = group.first, component_ = group.second;
    if (component_ < 0 || component_ >= NUM_FIELD_COMPONENTS || chunk < 0 ||
        chunk >= f.num_chunks)
      throw std::invalid_argument("material classification tensor group identity is invalid");
    const direction diagonal = component_direction(component(component_));
    const bool diagonal_chi = retained(
        rows, {chunk, int(array_kind::chi1inv), component_, -1, uint64_t(diagonal)});
    bool offdiagonal_chi = false;
    for (int d = 0; d < 5; ++d)
      if (d != int(diagonal))
        offdiagonal_chi = offdiagonal_chi || retained(
            rows, {chunk, int(array_kind::chi1inv), component_, -1, uint64_t(d)});
    if (offdiagonal_chi && !diagonal_chi)
      throw std::invalid_argument("material classification retained an orphan chi1 offdiagonal");
    const bool chi2 = retained(rows, {chunk, int(array_kind::chi2), component_, -1, 0});
    const bool chi3 = retained(rows, {chunk, int(array_kind::chi3), component_, -1, 0});
    if (chi2 != chi3 || ((chi2 || chi3) && !diagonal_chi))
      throw std::invalid_argument("material classification broke chi2/chi3 pairing");
    for (int d = 0; d < 5; ++d) {
      const StorageKey conductivity_key{
          chunk, int(array_kind::conductivity), component_, -1, uint64_t(d)};
      const StorageKey condinv_key{
          chunk, int(array_kind::condinv), component_, -1, uint64_t(d)};
      const auto conductivity_row = rows.find(conductivity_key);
      const auto condinv_row = rows.find(condinv_key);
      if (conductivity_row != rows.end() && condinv_row != rows.end() &&
          (conductivity_row->second == MaterialClassification::retained) !=
              (condinv_row->second == MaterialClassification::retained)) {
        char message[192];
        snprintf(message, sizeof(message),
                 "material classification broke conductivity/condinv pairing "
                 "(chunk=%d component=%d direction=%d conductivity=%d condinv=%d)",
                 chunk, component_, d,
                 int(conductivity_row->second == MaterialClassification::retained),
                 int(condinv_row->second == MaterialClassification::retained));
        throw std::invalid_argument(message);
      }
    }
  }
  for (const auto &group : pml_groups) {
    const bool sig = retained(rows, {group.first, int(array_kind::pml_sig), -1, -1,
                                     group.second});
    const bool kap = retained(rows, {group.first, int(array_kind::pml_kap), -1, -1,
                                     group.second});
    const bool inv = retained(rows, {group.first, int(array_kind::pml_siginv), -1, -1,
                                     group.second});
    if (sig != kap || sig != inv)
      throw std::invalid_argument("material classification broke a PML row group");
  }
  for (const SigmaGroup &group : sigma_groups) {
    const direction diagonal = component_direction(component(group.component));
    bool any = false;
    for (int d = 0; d < 5; ++d)
      any = any || retained(rows, {group.chunk, int(array_kind::sigma), group.component, d,
                                   group.identity});
    if (any && !retained(rows, {group.chunk, int(array_kind::sigma), group.component,
                                int(diagonal), group.identity}))
      throw std::invalid_argument("material classification retained an orphan sigma offdiagonal");
  }
}

MaterialClassification classification_from_rows(
    const fields &f, const StoragePlan &plan,
    const std::map<StorageKey, uint8_t, StorageKeyLess> &rows) {
  MaterialClassification result;
  result.provisional_row_state.assign(plan.arrays.size(),
                                      MaterialClassification::not_provisional);
  for (size_t i = 0; i < plan.arrays.size(); ++i) {
    if (plan.arrays[i].role != array_role::material) continue;
    const auto found = rows.find(plan.keys[i]);
    if (found == rows.end())
      throw std::invalid_argument("material classification omitted a local row status");
    result.provisional_row_state[i] = found->second;
    if (found->second == MaterialClassification::elided_row)
      result.elided.push_back(ArrayId{uint32_t(i)});
  }
  result.anisotropic_eh.assign(size_t(f.num_chunks) * NUM_FIELD_TYPES, 0);
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    for (field_type ft : {E_stuff, H_stuff})
      result.anisotropic_eh[size_t(chunk) * NUM_FIELD_TYPES + ft] =
          (expected_variant(f, chunk, ft, rows) & material_variant_anisotropic) ? 1 : 0;
  }
  return result;
}

class ScopedVariantPlanView {
public:
  ScopedVariantPlanView(fields &owner, CpuArrayCatalog &catalog, StoragePlan &storage,
                        std::vector<std::vector<grid_volume> > &tiles)
      : owner_(owner), old_catalog_(owner.array_catalog), old_storage_(owner.storage_plan),
        tiles_(tiles) {
    owner_.array_catalog = &catalog;
    owner_.storage_plan = &storage;
    for (int chunk = 0; chunk < owner_.num_chunks; ++chunk)
      for (field_type ft : {E_stuff, H_stuff})
        owner_.chunks[chunk]->gvs_eh[ft].swap(
            tiles_[size_t(chunk) * NUM_FIELD_TYPES + ft]);
  }
  ~ScopedVariantPlanView() {
    for (int chunk = 0; chunk < owner_.num_chunks; ++chunk)
      for (field_type ft : {E_stuff, H_stuff})
        owner_.chunks[chunk]->gvs_eh[ft].swap(
            tiles_[size_t(chunk) * NUM_FIELD_TYPES + ft]);
    owner_.array_catalog = old_catalog_;
    owner_.storage_plan = old_storage_;
  }

private:
  fields &owner_;
  CpuArrayCatalog *old_catalog_;
  StoragePlan *old_storage_;
  std::vector<std::vector<grid_volume> > &tiles_;
};

std::vector<MaterialVariantClassificationFact> expected_variant_facts(
    fields &f, const StoragePlan &plan, const MaterialRecipe &recipe,
    const MaterialClassification &classification) {
  if (!f.array_catalog)
    throw std::logic_error("material variant construction requires a live catalog");
  if (plan.arrays.size() != plan.keys.size() ||
      classification.provisional_row_state.size() != plan.arrays.size())
    throw std::invalid_argument("material variant construction has inconsistent extents");

  StoragePlan authoritative;
  const size_t host_prefix = f.array_catalog->host_backed_size();
  if (host_prefix > plan.arrays.size())
    throw std::invalid_argument("material variant plan is shorter than the host catalog");
  authoritative.arrays.assign(plan.arrays.begin(), plan.arrays.begin() + host_prefix);
  authoritative.keys.assign(plan.keys.begin(), plan.keys.begin() + host_prefix);
  for (ArraySpec &spec : authoritative.arrays) {
    spec.classification_provisional = false;
    spec.classification_elided = false;
  }
  StoragePlan resolved = plan;
  bool has_provisional = false;
  for (const ArraySpec &spec : plan.arrays)
    has_provisional = has_provisional || spec.classification_provisional;
  if (has_provisional)
    resolve_material_storage(recipe, classification, authoritative, resolved);
  CpuArrayCatalog catalog = *f.array_catalog;
  catalog.publish_resolved_plan(resolved);

  std::vector<std::vector<grid_volume> > tiles(size_t(f.num_chunks) * NUM_FIELD_TYPES);
  if (classification.anisotropic_eh.size() != tiles.size())
    throw std::invalid_argument("material variant classification has invalid tile extent");
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    for (field_type ft : {E_stuff, H_stuff}) {
      std::vector<grid_volume> &row = tiles[size_t(chunk) * NUM_FIELD_TYPES + ft];
      if (f.loop_tile_base_eh > 0 &&
          classification.anisotropic_eh[size_t(chunk) * NUM_FIELD_TYPES + ft]) {
        split_into_tiles(f.chunks[chunk]->gv, &row, f.loop_tile_base_eh);
        check_tiles(f.chunks[chunk]->gv, row);
      }
      else
        row.push_back(f.chunks[chunk]->gv);
    }
  }

  ScopedVariantPlanView view(f, catalog, resolved, tiles);
  return variant_facts_from_step_plan(build_step_plan(f, StepProgram::ordinary));
}

MaterialClassificationFacts cpu_facts(fields &f, const StoragePlan &plan,
                                      const MaterialRecipe &recipe) {
  if (!f.array_catalog)
    throw std::logic_error("material classification requires a live host catalog");
  MaterialClassificationFacts facts;
  std::map<StorageKey, uint8_t, StorageKeyLess> local_rows;
  for (size_t i = 0; i < plan.arrays.size(); ++i) {
    if (plan.arrays[i].role != array_role::material) continue;
    const uint8_t state = is_valid(f.array_catalog->find(plan.keys[i]))
                              ? uint8_t(MaterialClassification::retained)
                              : uint8_t(MaterialClassification::elided_row);
    facts.rows.push_back(MaterialRowClassificationFact{plan.keys[i], state});
    local_rows[plan.keys[i]] = state;
  }
  facts.has_nonlinearities = f.has_nonlinearities(false);
  if (f.gv.dim == D2) {
    for (int i = 0; i < f.num_chunks && !facts.aniso2d; ++i)
      facts.aniso2d = f.chunks[i]->s->has_chi(Ex, Z) || f.chunks[i]->s->has_chi(Ey, Z) ||
                      f.chunks[i]->s->has_chi(Ez, X) || f.chunks[i]->s->has_chi(Ez, Y) ||
                      f.chunks[i]->s->has_chi(Hx, Z) || f.chunks[i]->s->has_chi(Hy, Z) ||
                      f.chunks[i]->s->has_chi(Hz, X) || f.chunks[i]->s->has_chi(Hz, Y);
  }
  else if (f.beta != 0)
    throw std::invalid_argument("nonzero beta requires a two-dimensional material classification");
  FOR_COMPONENTS(c)
    if (f.have_component(c)) facts.required_components |= component_mask(1) << int(c);
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    for (dft_chunk *dft = f.chunks[chunk]->dft_chunks; dft; dft = dft->next_in_chunk) {
      const int factor = dft->get_decimation_factor();
      if (factor > 0 && (!facts.min_decimation_factor || factor < facts.min_decimation_factor))
        facts.min_decimation_factor = factor;
    }
  }
  const MaterialClassification local = classification_from_rows(f, plan, local_rows);
  facts.variants = expected_variant_facts(f, plan, recipe, local);
  return facts;
}

MaterialClassificationFacts facts_from_classification(
    fields &f, const StoragePlan &plan, const MaterialRecipe &recipe,
    const MaterialClassification &cls) {
  if (cls.provisional_row_state.size() != plan.arrays.size() ||
      cls.anisotropic_eh.size() != size_t(f.num_chunks) * NUM_FIELD_TYPES)
    throw std::invalid_argument("material classification result has an invalid extent");
  std::set<uint32_t> listed;
  uint32_t previous = 0;
  bool have_previous = false;
  for (ArrayId id : cls.elided) {
    if (!is_valid(id) || id.value >= plan.arrays.size() ||
        (have_previous && id.value <= previous) || !listed.insert(id.value).second)
      throw std::invalid_argument("material classification elision IDs are not canonical");
    previous = id.value;
    have_previous = true;
  }
  MaterialClassificationFacts facts;
  for (size_t i = 0; i < plan.arrays.size(); ++i) {
    if (plan.arrays[i].role == array_role::material) {
      if ((cls.provisional_row_state[i] == MaterialClassification::elided_row) !=
          (listed.count(uint32_t(i)) != 0))
        throw std::invalid_argument("material classification status/elision mismatch");
      facts.rows.push_back(
          MaterialRowClassificationFact{plan.keys[i], cls.provisional_row_state[i]});
    }
    else if (cls.provisional_row_state[i] != MaterialClassification::not_provisional)
      throw std::invalid_argument("material classification changed a non-material row");
  }
  for (const MaterialVariantClassificationFact &variant : cls.variant_facts)
    if (variant.chunk >= 0 && variant.chunk < f.num_chunks &&
        f.chunks[variant.chunk]->is_mine())
      facts.variants.push_back(variant);
  const std::vector<MaterialVariantClassificationFact> expected =
      expected_variant_facts(f, plan, recipe, cls);
  if (facts.variants.size() != expected.size())
    throw std::invalid_argument("material classification omitted or added an update region");
  for (size_t i = 0; i < expected.size(); ++i)
    if (!same_variant_fact(facts.variants[i], expected[i]))
      throw std::invalid_argument("material classification update region disagrees with the canonical plan");
  facts.required_components = cls.required_components;
  facts.has_nonlinearities = cls.has_nonlinearities;
  facts.aniso2d = cls.aniso2d && f.beta == 0;
  facts.min_decimation_factor = cls.min_decimation_factor;
  return facts;
}

void require_live_array(const StoragePlan &plan, ArrayId id, const char *what,
                        bool optional = true) {
  if (!is_valid(id)) {
    if (optional) return;
    throw std::invalid_argument(std::string(what) + " has no ArrayId");
  }
  if (id.value >= plan.arrays.size())
    throw std::invalid_argument(std::string(what) + " has an out-of-range ArrayId");
  if (plan.arrays[id.value].classification_elided)
    throw std::invalid_argument(std::string(what) + " references material tombstone ArrayId " +
                                std::to_string(id.value));
}

void require_pml(const StoragePlan &plan, const PmlProfile &pml, const char *what) {
  require_live_array(plan, pml.sig, what);
  require_live_array(plan, pml.kap, what);
  require_live_array(plan, pml.siginv, what);
}

} // namespace

void refresh_material_classification_variants(fields &f, const StoragePlan &plan,
                                              MaterialClassification &classification) {
  if (plan.arrays.size() != plan.keys.size() ||
      classification.provisional_row_state.size() != plan.arrays.size())
    throw std::invalid_argument("cannot refresh variants from incomplete row status");
  std::map<StorageKey, uint8_t, StorageKeyLess> rows;
  for (size_t i = 0; i < plan.arrays.size(); ++i)
    if (plan.arrays[i].role == array_role::material)
      rows.insert(std::make_pair(plan.keys[i], classification.provisional_row_state[i]));
  classification.anisotropic_eh.assign(size_t(f.num_chunks) * NUM_FIELD_TYPES, 0);
  for (int chunk = 0; chunk < f.num_chunks; ++chunk) {
    if (!f.chunks[chunk]->is_mine()) continue;
    for (field_type ft : {E_stuff, H_stuff}) {
      const size_t index = size_t(chunk) * NUM_FIELD_TYPES + ft;
      classification.anisotropic_eh[index] =
          (expected_variant(f, chunk, ft, rows) & material_variant_anisotropic) ? 1 : 0;
    }
  }
  const MaterialRecipe *recipe = NULL;
  std::unique_ptr<MaterialRecipe> owned;
  if (f.initialization_plan && f.initialization_plan->materials.size() == 1)
    recipe = &f.initialization_plan->materials[0];
  else {
    owned.reset(new MaterialRecipe(build_host_reference_material_recipe(f)));
    recipe = owned.get();
  }
  classification.variant_facts = expected_variant_facts(f, plan, *recipe, classification);
}

MaterialClassification assemble_material_classification(
    fields &f, const StoragePlan &plan, const MaterialRecipe &recipe,
    const MaterialClassificationFacts &local_facts) {
  std::string local_error;
  try {
    if (plan.arrays.size() != plan.keys.size())
      throw std::invalid_argument("material classification plan has inconsistent extents");
    validate_material_recipe(recipe);
    validate_local_facts(f, plan, recipe, local_facts);
  }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown local material classification fact failure"; }
  backend_reconcile_host_access(local_error, "material classification fact validation");

  std::vector<MaterialClassificationFacts> gathered;
  try { gathered = gather_facts(local_facts); }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown material classification fact gather failure"; }
  backend_reconcile_host_access(local_error, "material classification fact gather");

  MaterialClassification result;
  try {
    std::map<StorageKey, uint8_t, StorageKeyLess> rows;
    typedef std::tuple<uint32_t, int, uint32_t> VariantIdentity;
    std::map<VariantIdentity, MaterialVariantClassificationFact> variants;
    bool material_aniso2d = false;
    int minimum = 0;
    for (const MaterialClassificationFacts &facts : gathered) {
      if (facts.version != material_classification_facts_version)
        throw std::invalid_argument("gathered material classification version changed");
      result.required_components |= facts.required_components;
      result.has_nonlinearities = result.has_nonlinearities || facts.has_nonlinearities;
      material_aniso2d = material_aniso2d || facts.aniso2d;
      if (facts.min_decimation_factor > 0 &&
          (!minimum || facts.min_decimation_factor < minimum))
        minimum = facts.min_decimation_factor;
      for (const MaterialRowClassificationFact &row : facts.rows)
        if (!rows.insert(std::make_pair(row.key, row.state)).second)
          throw std::invalid_argument("material classification has duplicate global row facts");
      for (const MaterialVariantClassificationFact &variant : facts.variants)
        if (!variants.insert(std::make_pair(
                VariantIdentity(variant.operation, variant.chunk, variant.region), variant)).second)
          throw std::invalid_argument("material classification has duplicate global variants");
    }
    validate_group_topology(f, rows);
    result.anisotropic_eh.assign(size_t(f.num_chunks) * NUM_FIELD_TYPES, 0);
    for (int chunk = 0; chunk < f.num_chunks; ++chunk)
      for (field_type ft : {E_stuff, H_stuff}) {
        const uint32_t expected = expected_variant(f, chunk, ft, rows);
        const size_t index = size_t(chunk) * NUM_FIELD_TYPES + ft;
        result.anisotropic_eh[index] =
            (expected & material_variant_anisotropic) ? 1 : 0;
      }
    for (const auto &entry : variants) result.variant_facts.push_back(entry.second);
    result.min_decimation_factor = minimum ? minimum : 1;
    if (material_aniso2d && f.beta != 0 && f.is_real)
      throw std::invalid_argument("Nonzero beta need complex fields when mu/epsilon couple TE and TM");
    result.aniso2d = material_aniso2d || f.beta != 0;
    result.provisional_row_state.assign(plan.arrays.size(),
                                        MaterialClassification::not_provisional);
    for (size_t i = 0; i < plan.arrays.size(); ++i) {
      const ArraySpec &spec = plan.arrays[i];
      if (spec.role != array_role::material) {
        if (is_valid(spec.alias_of) &&
            (spec.alias_of.value >= plan.arrays.size() ||
             plan.arrays[spec.alias_of.value].classification_elided))
          throw std::invalid_argument("material classification retained an invalid alias");
        continue;
      }
      if (is_valid(spec.alias_of) || spec.classification_elided)
        throw std::invalid_argument("material classification input contains an alias/tombstone");
      const auto found = rows.find(plan.keys[i]);
      if (found == rows.end())
        throw std::invalid_argument("material classification omitted a local row status");
      result.provisional_row_state[i] = found->second;
      if (found->second == MaterialClassification::elided_row)
        result.elided.push_back(ArrayId{uint32_t(i)});
    }

    uint64_t hash = UINT64_C(1469598103934665603);
    hash = mix(hash, material_classification_facts_version);
    hash = mix(hash, recipe.ir() ? recipe.ir()->signature : 0);
    hash = mix(hash, rows.size());
    for (const auto &entry : rows) {
      hash = mix(hash, uint64_t(int64_t(entry.first.chunk)));
      hash = mix(hash, uint64_t(int64_t(entry.first.kind)));
      hash = mix(hash, uint64_t(int64_t(entry.first.component_)));
      hash = mix(hash, uint64_t(int64_t(entry.first.cmp)));
      hash = mix(hash, entry.first.aux);
      hash = mix(hash, entry.second);
    }
    hash = mix(hash, variants.size());
    for (const auto &entry : variants) {
      const MaterialVariantClassificationFact &variant = entry.second;
      hash = mix(hash, variant.operation);
      hash = mix(hash, variant.region);
      hash = mix(hash, uint64_t(int64_t(variant.chunk)));
      hash = mix(hash, uint64_t(int64_t(variant.field_type_)));
      hash = mix(hash, uint64_t(int64_t(variant.component_)));
      hash = mix(hash, uint64_t(int64_t(variant.cmp)));
      hash = mix(hash, uint64_t(int64_t(variant.dimension)));
      for (int axis = 0; axis < 5; ++axis) hash = mix(hash, uint64_t(int64_t(variant.begin[axis])));
      for (int axis = 0; axis < 5; ++axis) hash = mix(hash, uint64_t(int64_t(variant.end[axis])));
      hash = mix(hash, variant.base);
      for (int axis = 0; axis < 3; ++axis) hash = mix(hash, variant.counts[axis]);
      for (int axis = 0; axis < 3; ++axis)
        hash = mix(hash, uint64_t(int64_t(variant.strides[axis])));
      hash = mix(hash, variant.variant_key);
    }
    hash = mix(hash, result.required_components);
    hash = mix(hash, result.has_nonlinearities ? 1 : 0);
    hash = mix(hash, result.aniso2d ? 1 : 0);
    hash = mix(hash, uint64_t(result.min_decimation_factor));
    result.hash = hash;
  }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown material classification assembly failure"; }
  backend_reconcile_host_access(local_error, "material classification assembly");
  return result;
}

MaterialClassification classify(fields &f, const StoragePlan &plan) {
  MaterialClassificationFacts facts;
  const MaterialRecipe *recipe = NULL;
  std::unique_ptr<MaterialRecipe> owned;
  std::string local_error;
  try {
    if (f.initialization_plan && f.initialization_plan->materials.size() == 1)
      recipe = &f.initialization_plan->materials[0];
    else {
      owned.reset(new MaterialRecipe(build_host_reference_material_recipe(f)));
      recipe = owned.get();
    }
    facts = cpu_facts(f, plan, *recipe);
  }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown local material classification preparation failure"; }
  backend_reconcile_host_access(local_error, "material classification preparation");
  return assemble_material_classification(f, plan, *recipe, facts);
}

component_mask global_field_component_presence(fields &f) {
  int local[NUM_FIELD_COMPONENTS], global[NUM_FIELD_COMPONENTS];
  FOR_COMPONENTS(c) local[c] = f.have_component(c) ? 1 : 0;
  or_to_all(local, global, NUM_FIELD_COMPONENTS);
  component_mask result = 0;
  FOR_COMPONENTS(c)
    if (global[c]) result |= component_mask(1) << int(c);
  return result;
}

void validate_material_classification(fields &f, const StoragePlan &plan,
                                      const MaterialRecipe &recipe,
                                      MaterialClassification &classification) {
  MaterialClassificationFacts facts;
  std::string local_error;
  try { facts = facts_from_classification(f, plan, recipe, classification); }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown material classification result validation failure"; }
  backend_reconcile_host_access(local_error, "material classification result validation");
  classification = assemble_material_classification(f, plan, recipe, facts);
}

void validate_material_classification_consumers(const StoragePlan &plan,
                                                const InitializationPlan &initialization,
                                                const StepPlan &steps,
                                                const MaterialClassification &classification) {
  if (plan.arrays.size() != plan.keys.size())
    throw std::invalid_argument("resolved material plan has inconsistent extents");
  const std::vector<MaterialVariantClassificationFact> rebuilt =
      variant_facts_from_step_plan(steps);
  std::map<std::tuple<uint32_t, int, uint32_t>, MaterialVariantClassificationFact>
      classified;
  for (const MaterialVariantClassificationFact &fact : classification.variant_facts)
    if (!classified.insert(std::make_pair(
            std::make_tuple(fact.operation, fact.chunk, fact.region), fact)).second)
      throw std::invalid_argument("material classification contains duplicate update regions");
  size_t local_classified = 0;
  for (const MaterialVariantClassificationFact &fact : rebuilt) {
    const auto found = classified.find(
        std::make_tuple(fact.operation, fact.chunk, fact.region));
    if (found == classified.end() || !same_variant_fact(found->second, fact))
      throw std::invalid_argument(
          "rebuilt StepPlan update region disagrees with material classification");
    ++local_classified;
  }
  size_t expected_local = 0;
  std::set<int> local_chunks;
  for (size_t i = 0; i < plan.arrays.size(); ++i)
    if (plan.keys[i].chunk >= 0 && !plan.arrays[i].classification_elided)
      local_chunks.insert(plan.keys[i].chunk);
  for (const auto &entry : classified)
    if (local_chunks.count(entry.second.chunk)) ++expected_local;
  if (local_classified != expected_local)
    throw std::invalid_argument("material classification contains an extra local update region");
  for (size_t i = 0; i < plan.arrays.size(); ++i) {
    const ArraySpec &spec = plan.arrays[i];
    if (spec.id.value != i)
      throw std::invalid_argument("resolved material plan has noncanonical ArrayIds");
    if (is_valid(spec.alias_of)) {
      require_live_array(plan, spec.alias_of, "storage alias", false);
      if (spec.classification_elided)
        throw std::invalid_argument("material tombstone retained an alias");
    }
  }
  for (const InitOperation &operation : initialization.operations)
    require_live_array(plan, operation.destination.id, "initialization action", false);
  for (const Operation &operation : steps.operations)
    for (const BufferAccess &access : operation.accesses)
      require_live_array(plan, access.array.id, "step access", false);
  for (const CurlUpdate &u : steps.db_updates) {
    for (ArrayId id : {u.target, u.plus_source, u.minus_source, u.target_u, u.conductivity,
                       u.condinv, u.target_cond})
      require_live_array(plan, id, "curl action");
    require_pml(plan, u.pml, "curl PML action");
    require_pml(plan, u.pml_u, "curl PML action");
  }
  for (const BfastUpdate &u : steps.bfast_updates) {
    for (ArrayId id : {u.target, u.source1, u.source2, u.f_bfast, u.target_u, u.condinv,
                       u.target_cond})
      require_live_array(plan, id, "BFAST action");
    require_pml(plan, u.pml, "BFAST PML action");
    require_pml(plan, u.pml_u, "BFAST PML action");
  }
  for (const BetaUpdate &u : steps.beta_updates) {
    for (ArrayId id : {u.target, u.source, u.target_u, u.condinv, u.target_cond})
      require_live_array(plan, id, "beta action");
    require_pml(plan, u.pml, "beta PML action");
    require_pml(plan, u.pml_u, "beta PML action");
  }
  for (const CylindricalMOverRUpdate &u : steps.cylindrical_m_updates) {
    for (ArrayId id : {u.target, u.source, u.target_u, u.condinv, u.target_cond})
      require_live_array(plan, id, "cylindrical m action");
    require_pml(plan, u.pml, "cylindrical m PML action");
    require_pml(plan, u.pml_u, "cylindrical m PML action");
  }
  for (const CylindricalAxisUpdate &u : steps.cylindrical_axis_updates) {
    for (ArrayId id : {u.target, u.source1, u.source2, u.target_u, u.conductivity, u.condinv,
                       u.target_cond})
      require_live_array(plan, id, "cylindrical axis action");
    require_pml(plan, u.pml, "cylindrical axis PML action");
    require_pml(plan, u.pml_u, "cylindrical axis PML action");
  }
  for (const SlabRef &slab : steps.cylindrical_zero_slabs)
    require_live_array(plan, slab.array, "cylindrical zero action", false);
  for (const ConstitutiveUpdate &u : steps.eh_updates) {
    for (ArrayId id : {u.target, u.base_primary, u.base_cross1, u.base_cross2, u.primary,
                       u.cross1, u.cross2, u.diagonal, u.offdiagonal1, u.offdiagonal2,
                       u.chi2, u.chi3, u.target_w, u.previous_w})
      require_live_array(plan, id, "constitutive action");
    require_pml(plan, u.pml, "constitutive PML action");
  }
  for (const PolarizationUpdate &u : steps.polarization_updates)
    for (ArrayId id : {u.p, u.p_prev, u.p_cross1, u.p_prev_cross1, u.p_cross2,
                       u.p_prev_cross2, u.primary_w, u.cross_w1, u.cross_w2,
                       u.diagonal_sigma, u.offdiagonal_sigma1, u.offdiagonal_sigma2})
      require_live_array(plan, id, "polarization action");
  for (const MultilevelTransitionUpdate &u : steps.multilevel_transition_updates)
    for (ArrayId id : {u.p, u.p_prev, u.w, u.diagonal_sigma, u.populations})
      require_live_array(plan, id, "multilevel transition action");
  for (const MaterialRefreshArray &u : steps.material_refresh_arrays)
    require_live_array(plan, u.current, "material refresh action", false);
}

bool apply_classification(fields &f, const MaterialClassification &cls) {
  component_mask requested = cls.required_components;
  if (cls.aniso2d)
    FOR_COMPONENTS(c)
      if (f.gv.has_field(c)) requested |= component_mask(1) << int(c);

  FOR_COMPONENTS(c)
    if ((requested & (component_mask(1) << int(c))) && !f.gv.has_field(c))
      throw std::invalid_argument("material classification requires an unavailable component");

  const component_mask missing = requested & ~global_field_component_presence(f);
  const bool promoted = missing != 0;
  if (cls.aniso2d && missing) {
    /* One anisotropic requirement realizes every component supported by this
       grid.  All ranks use the same global mask and therefore enter the
       collectives in _require_component together, including idle ranks. */
    FOR_COMPONENTS(c) if (missing & (component_mask(1) << int(c))) {
      f._require_component(c, true);
      break;
    }
  }
  else {
    FOR_COMPONENTS(c)
      if (missing & (component_mask(1) << int(c))) f._require_component(c, false);
  }
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    FOR_FIELD_TYPES(ft) {
      if (ft != E_stuff && ft != H_stuff) continue;
      const bool is_aniso = cls.anisotropic_eh[size_t(i) * NUM_FIELD_TYPES + ft] != 0;
      std::vector<grid_volume> &tiles = f.chunks[i]->gvs_eh[ft];
      tiles.clear();
      if (f.loop_tile_base_eh > 0 && is_aniso) {
        split_into_tiles(f.chunks[i]->gv, &tiles, f.loop_tile_base_eh);
        check_tiles(f.chunks[i]->gv, tiles);
      }
      else
        tiles.push_back(f.chunks[i]->gv);
    }
  }
  return promoted;
}

const char *classification_summary(const MaterialClassification &cls) {
  static char buf[256];
  snprintf(buf, sizeof buf,
           "hash=%016llx nonlinear=%d aniso2d=%d decimation>=%d components=0x%llx",
           (unsigned long long)cls.hash, int(cls.has_nonlinearities), int(cls.aniso2d),
           cls.min_decimation_factor, (unsigned long long)cls.required_components);
  return buf;
}

} // namespace meep
