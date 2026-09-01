/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "meep.hpp"
#include "config.h"

#include "backend/random_state.hpp"
#include "support/meep_mt.h"
#include <atomic>
#include <limits>
#include <mutex>
#include <time.h>

using namespace std;

namespace meep {

namespace {

std::mutex random_state_mutex;
std::atomic<bool> rand_inited(false);
RandomSeedSnapshot seed_snapshot = {0, 0, 0, counter_random_algorithm_version,
                                    false, false, false, false, false};

uint32_t effective_seed(unsigned long seed) { return static_cast<uint32_t>(seed & 0xffffffffUL); }

void advance_seed_generation() {
  if (seed_snapshot.generation == std::numeric_limits<uint64_t>::max())
    meep::abort("random seed refresh generation overflow");
  ++seed_snapshot.generation;
}

void set_seed_locked(unsigned long seed, bool explicit_seed) {
  const bool prior_valid = seed_snapshot.semantic_seed_valid;
  const uint32_t prior_seed = seed_snapshot.semantic_seed;
  const bool prior_explicit = seed_snapshot.explicit_seed;
  const uint32_t normalized = effective_seed(seed);

  /* Keep this call before metadata publication.  Besides seeding MT19937 it
     preserves the legacy one-deep mt[] save used by restore_random_seed(). */
  meep_mt_init_genrand(normalized);

  seed_snapshot.saved_semantic_seed = prior_seed;
  seed_snapshot.saved_semantic_seed_valid = prior_valid;
  seed_snapshot.saved_explicit_seed = prior_explicit;
  seed_snapshot.semantic_seed = normalized;
  seed_snapshot.semantic_seed_valid = true;
  seed_snapshot.explicit_seed = explicit_seed;
  seed_snapshot.initialized = true;
  advance_seed_generation();
}

void init_rand_locked() {
  if (rand_inited.load(std::memory_order_relaxed)) return;
  set_seed_locked(time(NULL) * (1 + my_global_rank()), false);
}

void publish_initialized() { rand_inited.store(true, std::memory_order_release); }

} // namespace

static void init_rand(void) {
  if (rand_inited.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(random_state_mutex);
  if (rand_inited.load(std::memory_order_relaxed)) return;
  init_rand_locked();
  publish_initialized();
}

void set_random_seed(unsigned long seed) {
  std::lock_guard<std::mutex> lock(random_state_mutex);
  const bool was_initialized = rand_inited.load(std::memory_order_relaxed);
  init_rand_locked();
  set_seed_locked(seed, true);
  if (!was_initialized) publish_initialized();
}

void restore_random_seed() {
  std::lock_guard<std::mutex> lock(random_state_mutex);
  const bool was_initialized = rand_inited.load(std::memory_order_relaxed);
  init_rand_locked();

  const bool restored_valid = seed_snapshot.saved_semantic_seed_valid;
  const uint32_t restored_seed = seed_snapshot.saved_semantic_seed;
  const bool restored_explicit = seed_snapshot.saved_explicit_seed;

  /* Deliberately retain the historical mt[]-without-mti restore behavior. */
  meep_mt_restore_genrand();

  seed_snapshot.semantic_seed = restored_seed;
  seed_snapshot.semantic_seed_valid = restored_valid;
  seed_snapshot.explicit_seed = restored_explicit;
  advance_seed_generation();
  if (!was_initialized) publish_initialized();
}

RandomSeedSnapshot random_seed_snapshot() {
  std::lock_guard<std::mutex> lock(random_state_mutex);
  return seed_snapshot;
}

RandomSeedSnapshot ensure_random_seed_snapshot() {
  if (rand_inited.load(std::memory_order_acquire)) return random_seed_snapshot();
  std::lock_guard<std::mutex> lock(random_state_mutex);
  init_rand_locked();
  publish_initialized();
  return seed_snapshot;
}

void restore_random_seed_snapshot(const RandomSeedSnapshot &snapshot) {
  std::lock_guard<std::mutex> lock(random_state_mutex);
  if (snapshot.algorithm_version != counter_random_algorithm_version)
    meep::abort("checkpoint random algorithm version mismatch");
  if (snapshot.initialized && snapshot.semantic_seed_valid)
    meep_mt_init_genrand(snapshot.semantic_seed);
  seed_snapshot = snapshot;
  rand_inited.store(snapshot.initialized, std::memory_order_release);
}

int random_int(int a, int b) {
  init_rand();
  return a + meep_mt_genrand_int32() % (b - a + 1);
}

double uniform_random(double a, double b) {
  init_rand();
  return a + meep_mt_genrand_res53() * (b - a);
}

double gaussian_random(double mean, double stddev) {
  init_rand();
  // Box-Muller algorithm to generate Gaussian from uniform
  // see Knuth vol II algorithm P, sec. 3.4.1
  double v1, v2, s;
  do {
    v1 = 2 * meep_mt_genrand_res53() - 1;
    v2 = 2 * meep_mt_genrand_res53() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s >= 1.0);
  if (s == 0) { return mean; }
  else { return mean + v1 * sqrt(-2 * log(s) / s) * stddev; }
}

} // namespace meep
