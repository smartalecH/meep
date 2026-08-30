/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 */

#ifndef MEEP_BACKEND_RANDOM_STATE_HPP
#define MEEP_BACKEND_RANDOM_STATE_HPP

#include <stddef.h>
#include <stdint.h>

namespace meep {

/* Backend-private semantic metadata for the versioned counter RNG.  This does
   not replace or expose the legacy MT19937 state. */
const uint32_t counter_random_algorithm_version = 1;
const uint32_t counter_random_domain_word = 0x4d4e4f31u;

/* Version-1 static stream identity.  Callers validate the semantic domains
   before converting signed Meep ordinals to these fixed-width words. */
inline uint64_t counter_random_stream_tag(uint32_t version, uint32_t global_rank,
                                          uint32_t stable_chunk, uint32_t field_type,
                                          uint32_t state_index, uint32_t component,
                                          uint32_t cmp) {
  const uint32_t words[] = {counter_random_domain_word, version, global_rank, stable_chunk,
                            field_type, state_index, component, cmp};
  uint64_t hash = UINT64_C(0xcbf29ce484222325);
  for (size_t word = 0; word < sizeof(words) / sizeof(words[0]); ++word)
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= uint8_t(words[word] >> (8 * byte));
      hash *= UINT64_C(0x00000100000001b3);
    }
  return hash;
}

struct RandomSeedSnapshot {
  uint32_t semantic_seed;
  uint32_t saved_semantic_seed;
  uint64_t generation;
  uint32_t algorithm_version;
  bool initialized;
  bool semantic_seed_valid;
  bool saved_semantic_seed_valid;
  bool explicit_seed;
  bool saved_explicit_seed;
};

/* Returns one internally consistent snapshot.  Seed changes and restores are
   published atomically with respect to this accessor. */
RandomSeedSnapshot random_seed_snapshot();

/* Performs the same lazy default initialization as the legacy random draw
   entry points, then returns the resulting snapshot. */
RandomSeedSnapshot ensure_random_seed_snapshot();

} // namespace meep

#endif // MEEP_BACKEND_RANDOM_STATE_HPP
