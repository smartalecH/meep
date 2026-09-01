/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
 *
 * Backend-private checkpoint staging.  The public fields::dump/load surface
 * and its legacy datasets remain unchanged; this manifest is an additional
 * validation/continuation image owned entirely by fields_dump.cpp.
 */

#ifndef MEEP_BACKEND_CHECKPOINT_HPP
#define MEEP_BACKEND_CHECKPOINT_HPP

#include <stdint.h>
#include <string>
#include <vector>

#include "backend/storage_plan.hpp"
#include "backend/random_state.hpp"

namespace meep {

class h5file;

const uint64_t checkpoint_schema_magic = UINT64_C(0x4d4545504331); // "MEEPC1"
const uint64_t checkpoint_schema_version = 3;

enum class CheckpointFailurePoint {
  none,
  snapshot,
  write,
  read,
  allocation,
  validation,
  precommit,
  rename_backup,
  rename_publish,
  rename_restore
};

struct CheckpointRow {
  StorageKey key;
  ArraySpec spec;
  StorageKey alias_key;
  bool has_alias;
  int little_corner[3];
  int big_corner[3];
  std::vector<double> values;
  uint64_t checksum;

  CheckpointRow();
};

struct CheckpointImage {
  uint64_t schema_magic;
  uint64_t schema_version;
  uint64_t endian_marker;
  uint64_t host_realnum_bytes;
  uint64_t dimension;
  uint64_t configuration_signature;
  uint64_t storage_signature;
  /* Portable identities normalize real-valued recipe inputs to binary32 and
     are consulted only for cross-precision loads.  Native identities preserve
     the exact source representation and guard same-precision continuation. */
  uint64_t material_recipe_signature;
  uint64_t material_native_signature;
  uint64_t classification_hash;
  uint64_t source_definition_signature;
  uint64_t dft_recipe_signature;
  uint64_t dft_native_signature;
  uint64_t saved_rank_count;
  bool shared_manifest;
  int timestep;
  double dt;
  double source_times[3];
  uint64_t mutation_generation[fields::num_mutation_kinds];
  RandomSeedSnapshot random_seed;
  std::vector<uint32_t> random_seed_ranks;
  std::vector<RandomSeedSnapshot> random_seeds;
  std::vector<CheckpointRow> rows;
  std::vector<double> legacy_flux_values;
  std::vector<uint64_t> legacy_flux_signatures;

  CheckpointImage();
};

size_t checkpoint_scalar_count(const ArraySpec &spec);
uint64_t checkpoint_row_checksum(const CheckpointRow &row);
uint64_t checkpoint_storage_signature(const StoragePlan &plan, const CpuArrayCatalog &catalog);
uint64_t checkpoint_encode_signed_for_testing(int value);
int checkpoint_decode_signed_for_testing(uint64_t value);

void checkpoint_set_failure_for_testing(CheckpointFailurePoint point, int rank = -1);
void checkpoint_set_secondary_failure_for_testing(CheckpointFailurePoint point, int rank = -1);
void checkpoint_clear_failure_for_testing();
void checkpoint_fail_if_requested(CheckpointFailurePoint point);

class CheckpointTransaction {
public:
  static void validate_eligible(fields &owner, const char *operation);
  static CheckpointImage snapshot(fields &owner);
  static void write_manifest(h5file &file, const CheckpointImage &image,
                             bool single_parallel_file);
  static bool has_manifest(h5file &file);
  static CheckpointImage read_manifest(h5file &file, bool single_parallel_file);
  static void validate_target(fields &owner, const CheckpointImage &image);
  static void commit(fields &owner, const CheckpointImage &image);
};

} // namespace meep

#endif
