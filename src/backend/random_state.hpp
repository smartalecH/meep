/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 */

#ifndef MEEP_BACKEND_RANDOM_STATE_HPP
#define MEEP_BACKEND_RANDOM_STATE_HPP

#include <stdint.h>

namespace meep {

/* Backend-private semantic metadata for the versioned counter RNG.  This does
   not replace or expose the legacy MT19937 state. */
const uint32_t counter_random_algorithm_version = 1;

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
