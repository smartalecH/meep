/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(HAVE_DIRENT_H)
#include <dirent.h>
#endif
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#if defined(HAVE_SYS_STAT_H)
#include <sys/stat.h>
#endif
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif

#include <meep.hpp>

#include "backend/device_uuid_reservation.hpp"
#include "backend/nvidia/nvidia_backend.hpp"
#include "backend/storage_plan.hpp"

using namespace meep;

#if defined(HAVE_DIRENT_H) && defined(HAVE_SYS_STAT_H) && defined(HAVE_UNISTD_H) && \
    defined(HAVE_MKDIR) && defined(HAVE_MKDTEMP)
namespace {

int failures = 0;
precision_policy_kind test_precision = precision_policy_kind::native;

#define CHECK(condition, message)                                                                 \
  do {                                                                                            \
    if (!(condition)) {                                                                           \
      std::fprintf(stderr, "[rank %d] FAIL %s:%d: %s\n", my_global_rank(), __FILE__, __LINE__, \
                   message);                                                                      \
      ++failures;                                                                                 \
    }                                                                                             \
  } while (0)

double epsilon(const vec &) { return 1.0; }

std::string synthetic_uuid(unsigned value) {
  char text[64] = {};
  std::snprintf(text, sizeof(text), "000000000000000000000000%08x", value);
  return text;
}

void remove_test_root(const std::string &root) {
  DIR *directory = opendir(root.c_str());
  if (!directory) return;
  while (dirent *entry = readdir(directory)) {
    if (!std::strcmp(entry->d_name, ".") || !std::strcmp(entry->d_name, "..")) continue;
    const std::string path = root + "/" + entry->d_name;
    unlink(path.c_str());
  }
  closedir(directory);
  rmdir(root.c_str());
}

void test_state_lifetime(const std::string &root) {
  const std::string uuid = synthetic_uuid(unsigned(my_global_rank() + 1));
  device_uuid_testing::set_overrides(uuid.c_str(), root.c_str());
  grid_volume gv = vol2d(1.0, 1.0, 2.0);
  structure s(gv, epsilon);
  fields f(&s);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  options.precision = test_precision;

  std::unique_ptr<NvidiaBackend> owner(new NvidiaBackend(f, options, 0));
  CHECK(owner->device_reservation_valid_for_testing(),
        "backend admission did not acquire a UUID reservation");
  const void *identity = owner->device_reservation_identity_for_testing();
  CHECK(owner->normalized_device_uuid_for_testing() == uuid,
        "backend did not normalize the provider UUID used for admission");

  bool unrelated_duplicate_rejected = false;
  try {
    grid_volume other_gv = vol2d(1.0, 1.0, 2.0);
    structure other_structure(other_gv, epsilon);
    fields other(&other_structure);
    std::unique_ptr<NvidiaBackend> duplicate(new NvidiaBackend(other, options, 0));
  }
  catch (const std::exception &) { unrelated_duplicate_rejected = true; }
  CHECK(unrelated_duplicate_rejected,
        "unrelated backend acquired an already owned synthetic device UUID");
  f.backend = owner.get();

  std::unique_ptr<NvidiaBackend> same(new NvidiaBackend(f, options, 0));
  CHECK(same->device_reservation_identity_for_testing() == identity,
        "same-device candidate did not borrow the live reservation");

  StoragePlan empty;
  std::unique_ptr<BackendState> live(owner->create_state(empty));
  CHECK(owner->device_reservation_identity_for_testing() == identity,
        "state publication replaced the backend reservation");

  nvidia::testing::set_initialization_memory_budget_for_testing(1);
  bool failed = false;
  try {
    std::unique_ptr<BackendState> rejected(owner->create_state(empty));
  }
  catch (const std::exception &) { failed = true; }
  nvidia::testing::set_initialization_memory_budget_for_testing(
      std::numeric_limits<size_t>::max());
  CHECK(failed, "injected state-admission failure was not observed");
  CHECK(owner->device_reservation_identity_for_testing() == identity &&
            owner->device_reservation_valid_for_testing(),
        "failed state rebuild disturbed the live reservation");

  std::unique_ptr<BackendState> replacement(same->create_state(empty));
  CHECK(same->device_reservation_identity_for_testing() == identity,
        "successful same-device state rebuild reacquired the reservation");
  replacement.reset();
  live.reset();
  f.backend = NULL;
  same.reset();
  owner.reset();

  std::unique_ptr<NvidiaBackend> reacquired(new NvidiaBackend(f, options, 0));
  CHECK(reacquired->device_reservation_valid_for_testing(),
        "final CUDA-state retirement did not release the reservation");
}

void test_split_subgroup_progress(const std::string &root) {
  const int world_size = count_processors();
  if (world_size < 2) return;
  divide_parallel_processes(world_size);
  CHECK(count_processors() == 1, "backend lock split did not create size-one subgroups");

  const std::string first_uuid = synthetic_uuid(0x11111111u);
  const std::string second_uuid = synthetic_uuid(0x22222222u);
  grid_volume gv = vol2d(1.0, 1.0, 2.0);
  structure s(gv, epsilon);
  fields f(&s);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = test_precision;

  std::unique_ptr<NvidiaBackend> owner;
  if (my_global_rank() == 0) {
    device_uuid_testing::set_overrides(first_uuid.c_str(), root.c_str());
    owner.reset(new NvidiaBackend(f, options, 0));
  }
  begin_global_communications();
  all_wait();
  end_global_communications();

  bool contention_rejected = false;
  bool independent_progress = false;
  std::unique_ptr<NvidiaBackend> independent;
  if (my_global_rank() != 0) {
    device_uuid_testing::set_overrides(first_uuid.c_str(), root.c_str());
    try { independent.reset(new NvidiaBackend(f, options, 0)); }
    catch (const std::exception &) { contention_rejected = true; }
  }
  if (my_global_rank() == 1 && contention_rejected) {
    device_uuid_testing::set_overrides(second_uuid.c_str(), root.c_str());
    independent.reset(new NvidiaBackend(f, options, 0));
    independent_progress = independent->device_reservation_valid_for_testing();
  }

  begin_global_communications();
  all_wait();
  CHECK(sum_to_all(owner ? 1 : 0) == 1,
        "split backend test did not retain exactly one contested owner");
  CHECK(sum_to_all(contention_rejected ? 1 : 0) == world_size - 1,
        "split backend contention was not rejected on every nonowner subgroup");
  CHECK(sum_to_all(independent_progress ? 1 : 0) == 1,
        "a contending subgroup could not independently admit another UUID");
  all_wait();
  end_global_communications();

  independent.reset();
  owner.reset();
  device_uuid_testing::set_overrides(NULL, NULL);
  end_divide_parallel();
}

} // namespace

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  if (const char *precision = std::getenv("MEEP_TEST_PRECISION")) {
    if (!std::strcmp(precision, "f32")) test_precision = precision_policy_kind::f32;
    else if (std::strcmp(precision, "native")) return 6;
  }
  std::string support_why;
  if (!DeviceUuidReservation::platform_supported(support_why)) {
    if (my_rank() == 0)
      std::printf("nvidia_device_reservation: SKIP (%s)\n", support_why.c_str());
    return 0;
  }

  char root_text[256] = {};
  if (my_global_rank() == 0) {
    std::strcpy(root_text, "/tmp/meep-pr7-nvidia-lock-XXXXXX");
    if (!mkdtemp(root_text)) root_text[0] = '\0';
  }
  broadcast(0, root_text, int(sizeof(root_text)));
  const std::string root(root_text);
  CHECK(!root.empty(), "could not create shared backend reservation root");

  test_state_lifetime(root);
  all_wait();
  test_split_subgroup_progress(root);
  all_wait();
  if (my_rank() == 0) remove_test_root(root);
  const int total = sum_to_all(failures);
  if (my_rank() == 0)
    std::printf("nvidia_device_reservation: %s (%d failures)\n",
                total ? "FAIL" : "PASS", total);
  return total ? 1 : 0;
}
#else
int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  if (my_rank() == 0)
    std::printf("nvidia_device_reservation: SKIP (POSIX test facilities unavailable)\n");
  return 0;
}
#endif
