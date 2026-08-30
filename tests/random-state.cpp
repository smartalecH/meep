/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 */

#include <atomic>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <thread>

#include <meep.hpp>

#include "backend/random_state.hpp"

using namespace meep;

#ifndef MEEP_RANDOM_STATE_SCENARIO
#define MEEP_RANDOM_STATE_SCENARIO 0
#endif

static int failures = 0;

#define CHECK(cond, ...)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      master_printf("FAIL (%s:%d): ", __FILE__, __LINE__);                                       \
      master_printf(__VA_ARGS__);                                                                  \
      master_printf("\n");                                                                       \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (0)

static uint64_t bits(double value) {
  uint64_t result = 0;
  memcpy(&result, &value, sizeof(result));
  return result;
}

static void check_initial(const RandomSeedSnapshot &s) {
  CHECK(s.algorithm_version == counter_random_algorithm_version,
        "initial algorithm version is %u, expected %u", unsigned(s.algorithm_version),
        unsigned(counter_random_algorithm_version));
  CHECK(!s.initialized, "initial seed metadata is already initialized");
  CHECK(!s.semantic_seed_valid && !s.saved_semantic_seed_valid,
        "initial seed metadata unexpectedly has a valid seed");
  CHECK(s.generation == 0, "initial seed generation is %llu, expected zero",
        (unsigned long long)s.generation);
}

static void check_public_random_kat() {
  set_random_seed(5);
  CHECK(random_int(0, 1000000) == 452458, "random_int changed for seed 5");
  CHECK(bits(uniform_random(0.0, 1.0)) == UINT64_C(0x3fac4091b7ba13f0),
        "uniform_random changed for seed 5");
  CHECK(bits(gaussian_random(0.0, 1.0)) == UINT64_C(0x3ff116654bc6428f),
        "gaussian_random changed for seed 5");
}

static void test_initial_restore() {
  check_initial(random_seed_snapshot());

  restore_random_seed();
  RandomSeedSnapshot s = random_seed_snapshot();
  CHECK(s.initialized, "first restore did not perform lazy initialization");
  CHECK(!s.semantic_seed_valid && !s.saved_semantic_seed_valid,
        "first restore invented a semantic seed");
  CHECK(s.generation == 2, "first restore generation is %llu, expected 2",
        (unsigned long long)s.generation);
  CHECK(random_int(0, 1000000) == 0,
        "first invalid restore changed the legacy zero-array/no-cursor output");

  restore_random_seed();
  s = random_seed_snapshot();
  CHECK(!s.semantic_seed_valid && !s.saved_semantic_seed_valid,
        "repeated invalid restore invented a semantic seed");
  CHECK(s.generation == 3, "repeated invalid restore generation is %llu, expected 3",
        (unsigned long long)s.generation);
  CHECK(random_int(0, 1000000) == 0,
        "repeated invalid restore changed the legacy zero-array output");

  set_random_seed(7);
  s = random_seed_snapshot();
  CHECK(s.semantic_seed_valid && s.semantic_seed == 7 && s.explicit_seed,
        "set after invalid restore did not publish explicit seed 7");
  CHECK(!s.saved_semantic_seed_valid,
        "set after invalid restore incorrectly made the saved seed valid");
  CHECK(s.generation == 4, "set-after-restore generation is %llu, expected 4",
        (unsigned long long)s.generation);
}

static void test_initial_set() {
  check_initial(random_seed_snapshot());

  const unsigned long high_seed =
      sizeof(unsigned long) > sizeof(uint32_t) ? ((1UL << 32) | 5UL) : 5UL;
  set_random_seed(high_seed);
  RandomSeedSnapshot s = random_seed_snapshot();
  CHECK(s.initialized && s.semantic_seed_valid && s.explicit_seed,
        "first explicit set did not publish valid explicit metadata");
  CHECK(s.semantic_seed == 5, "seed was not normalized to its low 32 bits");
  CHECK(s.saved_semantic_seed_valid && !s.saved_explicit_seed,
        "first explicit set did not retain the hidden default seed");
  CHECK(s.generation == 2, "first explicit set generation is %llu, expected 2",
        (unsigned long long)s.generation);

  check_public_random_kat();
  const uint64_t generation_after_kat = random_seed_snapshot().generation;
  set_random_seed(high_seed);
  const int high_i = random_int(0, 1000000);
  const uint64_t high_u = bits(uniform_random(0.0, 1.0));
  const uint64_t high_g = bits(gaussian_random(0.0, 1.0));
  set_random_seed(5);
  CHECK(random_int(0, 1000000) == high_i, "low-32-bit-equivalent integer streams differ");
  CHECK(bits(uniform_random(0.0, 1.0)) == high_u,
        "low-32-bit-equivalent uniform streams differ");
  CHECK(bits(gaussian_random(0.0, 1.0)) == high_g,
        "low-32-bit-equivalent Gaussian streams differ");
  CHECK(random_seed_snapshot().generation == generation_after_kat + 2,
        "low-32-bit-equivalent sets did not each advance generation");
}

static void test_lazy_default() {
  check_initial(random_seed_snapshot());
  const RandomSeedSnapshot s = ensure_random_seed_snapshot();
  CHECK(s.initialized && s.semantic_seed_valid && !s.explicit_seed,
        "lazy default did not publish a valid default seed");
  CHECK(!s.saved_semantic_seed_valid,
        "lazy default unexpectedly published a saved semantic seed");
  CHECK(s.generation == 1, "lazy default generation is %llu, expected 1",
        (unsigned long long)s.generation);
  const RandomSeedSnapshot repeated = ensure_random_seed_snapshot();
  CHECK(repeated.generation == s.generation && repeated.semantic_seed == s.semantic_seed,
        "repeated lazy-default snapshot changed metadata");
}

static void test_transitions_and_snapshot_threads() {
  check_initial(random_seed_snapshot());

  set_random_seed(12345);
  CHECK(random_int(0, 1000000) == 666698, "first legacy restore fixture draw changed");
  CHECK(random_int(0, 1000000) == 181558, "second legacy restore fixture draw changed");
  set_random_seed(9);
  RandomSeedSnapshot s = random_seed_snapshot();
  CHECK(s.semantic_seed == 9 && s.saved_semantic_seed == 12345 &&
            s.saved_semantic_seed_valid,
        "nested set did not retain the prior semantic seed");

  restore_random_seed();
  s = random_seed_snapshot();
  CHECK(s.semantic_seed_valid && s.semantic_seed == 12345 && s.explicit_seed,
        "restore did not publish the saved explicit semantic seed");
  CHECK(s.saved_semantic_seed_valid && s.saved_semantic_seed == 12345,
        "restore unexpectedly changed the one-deep saved seed");
  CHECK(random_int(0, 1000000) == 344418,
        "legacy mt[]-without-cursor restore behavior changed");
  const uint64_t before_repeat = s.generation;
  restore_random_seed();
  s = random_seed_snapshot();
  CHECK(s.semantic_seed == 12345 && s.saved_semantic_seed == 12345,
        "repeated restore created another semantic save level");
  CHECK(s.generation == before_repeat + 1,
        "repeated restore did not advance refresh generation");
  CHECK(random_int(0, 1000000) == 181558,
        "repeated valid restore changed the legacy mt[]-without-cursor output");

  set_random_seed(11);
  set_random_seed(12);
  set_random_seed(11);
  std::atomic<bool> ready(false);
  std::atomic<bool> start(false);
  std::atomic<bool> done(false);
  std::atomic<unsigned long> observations(0);
  std::atomic<int> thread_failures(0);
  std::thread reader([&]() {
    ready.store(true, std::memory_order_release);
    while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
    uint64_t previous = 0;
    while (!done.load(std::memory_order_acquire)) {
      const RandomSeedSnapshot observed = random_seed_snapshot();
      const bool coherent_pair =
          (observed.generation & 1) ? (observed.semantic_seed == 12 &&
                                       observed.saved_semantic_seed == 11)
                                    : (observed.semantic_seed == 11 &&
                                       observed.saved_semantic_seed == 12);
      if (!observed.initialized || !observed.semantic_seed_valid ||
          observed.algorithm_version != counter_random_algorithm_version ||
          !observed.saved_semantic_seed_valid || !coherent_pair ||
          !observed.explicit_seed || !observed.saved_explicit_seed ||
          observed.generation < previous)
        thread_failures.fetch_add(1, std::memory_order_relaxed);
      previous = observed.generation;
      observations.fetch_add(1, std::memory_order_relaxed);
    }
  });
  while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
  start.store(true, std::memory_order_release);
  while (!observations.load(std::memory_order_acquire)) std::this_thread::yield();
  for (int i = 0; i < 2000; ++i)
    set_random_seed((i & 1) ? 11 : 12);
  done.store(true, std::memory_order_release);
  reader.join();
  CHECK(observations.load() > 0, "seed snapshot reader did not observe any publication");
  CHECK(thread_failures.load() == 0, "seed snapshot reader observed inconsistent metadata");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

#if MEEP_RANDOM_STATE_SCENARIO == 1
  test_initial_restore();
#elif MEEP_RANDOM_STATE_SCENARIO == 2
  test_initial_set();
#elif MEEP_RANDOM_STATE_SCENARIO == 3
  test_transitions_and_snapshot_threads();
#elif MEEP_RANDOM_STATE_SCENARIO == 4
  test_lazy_default();
#else
#error "MEEP_RANDOM_STATE_SCENARIO must select one fresh-process test"
#endif

  if (failures) {
    master_printf("random-state scenario %d: %d FAILURE(S)\n", MEEP_RANDOM_STATE_SCENARIO,
                  failures);
    return 1;
  }
  master_printf("random-state scenario %d: all checks passed\n", MEEP_RANDOM_STATE_SCENARIO);
  return 0;
}
