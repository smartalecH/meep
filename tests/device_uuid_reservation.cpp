/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "config.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(HAVE_DIRENT_H)
#include <dirent.h>
#endif
#if defined(HAVE_FCNTL_H)
#include <fcntl.h>
#endif
#include <limits.h>
#include <string>
#if defined(HAVE_SYS_STAT_H)
#include <sys/stat.h>
#endif
#if defined(HAVE_SYS_TYPES_H)
#include <sys/types.h>
#endif
#if defined(HAVE_SYS_WAIT_H)
#include <sys/wait.h>
#endif
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif

#include <meep.hpp>

#include "backend/device_uuid_reservation.hpp"

using namespace meep;

#if defined(HAVE_DIRENT_H) && defined(HAVE_FCNTL_H) && defined(HAVE_SYS_STAT_H) && \
    defined(HAVE_SYS_TYPES_H) && defined(HAVE_SYS_WAIT_H) && defined(HAVE_UNISTD_H) && \
    defined(HAVE_CLOSE) && defined(HAVE_FLOCK) && defined(HAVE_FORK) && \
    defined(HAVE_FSTAT) && defined(HAVE_FTRUNCATE) && defined(HAVE_GETEUID) && \
    defined(HAVE_LSEEK) && defined(HAVE_LSTAT) && defined(HAVE_MKDIR) && \
    defined(HAVE_MKDTEMP) && defined(HAVE_OPENAT) && defined(HAVE_PIPE) && \
    defined(HAVE_WAITPID) && defined(HAVE_WRITE)
#define MEEP_DEVICE_RESERVATION_TEST_POSIX 1
#endif

#ifdef MEEP_DEVICE_RESERVATION_TEST_POSIX
namespace {
int failures = 0;
const char *uuid_a = "GPU-00112233-4455-6677-8899-AABBCCDDEEFF";
const char *uuid_a_alias = "00112233445566778899aabbccddeeff";
const char *uuid_b = "ffeeddccbbaa99887766554433221100";

#define CHECK(condition, message)                                                                 \
  do {                                                                                            \
    if (!(condition)) {                                                                           \
      std::fprintf(stderr, "[rank %d] FAIL %s:%d: %s\n", my_rank(), __FILE__, __LINE__,         \
                   message);                                                                      \
      ++failures;                                                                                 \
    }                                                                                             \
  } while (0)

std::string shared_root() {
  char path[PATH_MAX] = {};
  if (my_rank() == 0) {
    std::strcpy(path, "/tmp/meep-pr7-device-lock-XXXXXX");
    if (!mkdtemp(path)) path[0] = '\0';
  }
  broadcast(0, path, PATH_MAX);
  return path;
}

void test_normalization() {
  std::string normalized, why;
  CHECK(DeviceUuidReservation::normalize_uuid(uuid_a, normalized, why), why.c_str());
  CHECK(normalized == uuid_a_alias, "UUID normalization is not canonical");
  CHECK(!DeviceUuidReservation::normalize_uuid("not-a-uuid", normalized, why),
        "malformed UUID was accepted");
}

void test_local_security(const std::string &root) {
  if (my_rank() != 0) return;
  std::string why;
  DeviceUuidReservation first;
  CHECK(first.acquire(uuid_a, why, root), why.c_str());
  const std::string path = first.lock_path();
  CHECK(path.find("5947d7c33d783f94b3b4c1a96ebc8991ed28f1b069b71e03376cba8caa98a720.lock") !=
            std::string::npos,
        "lock filename is not the SHA-256 of the normalized UUID");
  DeviceUuidReservation duplicate;
  CHECK(!duplicate.acquire(uuid_a_alias, why, root),
        "same-process normalized UUID duplicate was accepted");
  first.release();
  CHECK(duplicate.acquire(uuid_a_alias, why, root), why.c_str());
  duplicate.release();

  const std::string hard = path + ".hard";
  CHECK(link(path.c_str(), hard.c_str()) == 0, "could not create hard-link test fixture");
  DeviceUuidReservation hardlink;
  CHECK(!hardlink.acquire(uuid_a, why, root), "multiply-linked lock file was accepted");
  unlink(hard.c_str());

  CHECK(unlink(path.c_str()) == 0, "could not remove lock file for symlink fixture");
  CHECK(symlink("/dev/null", path.c_str()) == 0, "could not create symlink fixture");
  DeviceUuidReservation symlinked;
  CHECK(!symlinked.acquire(uuid_a, why, root), "symlink lock file was accepted");
  unlink(path.c_str());

  CHECK(mkdir(path.c_str(), 0700) == 0, "could not create non-regular lock fixture");
  DeviceUuidReservation nonregular;
  CHECK(!nonregular.acquire(uuid_a, why, root), "directory lock target was accepted");
  rmdir(path.c_str());

  const int insecure_fd = open(path.c_str(), O_CREAT | O_WRONLY, 0644);
  CHECK(insecure_fd >= 0, "could not create insecure-mode lock fixture");
  if (insecure_fd >= 0) close(insecure_fd);
  CHECK(chmod(path.c_str(), 0644) == 0, "could not set insecure lock-file mode");
  DeviceUuidReservation insecure_mode;
  CHECK(!insecure_mode.acquire(uuid_a, why, root), "group/world-readable lock file was accepted");
  unlink(path.c_str());

  const std::string directory_target = root + ".target";
  const std::string directory_link = root + ".link";
  CHECK(mkdir(directory_target.c_str(), 0700) == 0,
        "could not create lock-root symlink target");
  CHECK(symlink(directory_target.c_str(), directory_link.c_str()) == 0,
        "could not create lock-root symlink fixture");
  DeviceUuidReservation linked_root;
  CHECK(!linked_root.acquire(uuid_a, why, directory_link), "symlink lock root was accepted");
  unlink(directory_link.c_str());
  rmdir(directory_target.c_str());

  setenv("MEEP_TEST_DEVICE_LOCK_UNSUPPORTED", "1", 1);
  DeviceUuidReservation unsupported;
  CHECK(!unsupported.acquire(uuid_a, why, root), "forced unsupported platform was accepted");
  unsetenv("MEEP_TEST_DEVICE_LOCK_UNSUPPORTED");

  int ready[2] = {-1, -1};
  int release_child[2] = {-1, -1};
  CHECK(pipe(ready) == 0 && pipe(release_child) == 0,
        "could not create live-contention pipes");
  const pid_t contender = fork();
  CHECK(contender >= 0, "fork failed for live-contention test");
  if (contender == 0) {
    close(ready[0]);
    close(release_child[1]);
    DeviceUuidReservation held;
    std::string child_why;
    const char state = held.acquire(uuid_b, child_why, root) ? '1' : '0';
    if (write(ready[1], &state, 1) != 1) _exit(4);
    char done = 0;
    if (state != '1' || read(release_child[0], &done, 1) != 1) _exit(5);
    _exit(0);
  }
  if (contender > 0) {
    close(ready[1]);
    close(release_child[0]);
    char state = 0;
    CHECK(read(ready[0], &state, 1) == 1 && state == '1',
          "child could not hold live contention reservation");
    DeviceUuidReservation blocked;
    CHECK(!blocked.acquire(uuid_b, why, root), "live cross-process UUID contention was accepted");
    const char done = 'x';
    CHECK(write(release_child[1], &done, 1) == 1,
          "could not release live-contention child");
    int status = 0;
    waitpid(contender, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "live-contention child failed");
    close(ready[0]);
    close(release_child[1]);
  }

  const pid_t child = fork();
  CHECK(child >= 0, "fork failed for crash-release test");
  if (child == 0) {
    DeviceUuidReservation held;
    std::string child_why;
    _exit(held.acquire(uuid_b, child_why, root) ? 0 : 3);
  }
  if (child > 0) {
    int status = 0;
    waitpid(child, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "child could not acquire crash-release fixture");
    DeviceUuidReservation after_crash;
    CHECK(after_crash.acquire(uuid_b, why, root),
          "kernel did not release UUID lock after child exit");
  }
}

void test_split_contention(const std::string &root) {
  const int world_size = count_processors();
  if (world_size < 2) return;
  divide_parallel_processes(world_size); // every active communicator has size one
  CHECK(count_processors() == 1, "split fixture did not create size-one active communicator");
  begin_global_communications();
  all_wait();
  end_global_communications();
  DeviceUuidReservation reservation;
  std::string why;
  const bool acquired = reservation.acquire(uuid_a, why, root);
  begin_global_communications();
  all_wait(); // every contender has completed acquisition while the winner holds the fd
  const int winners = sum_to_all(acquired ? 1 : 0);
  CHECK(winners == 1, "same-UUID split groups did not produce exactly one lock owner");
  all_wait();
  end_global_communications();
  reservation.release();

  begin_global_communications();
  all_wait();
  end_global_communications();
  DeviceUuidReservation distinct;
  const std::string per_rank = my_global_rank() & 1 ? uuid_a : uuid_b;
  const bool distinct_acquired = distinct.acquire(per_rank, why, root);
  begin_global_communications();
  all_wait();
  const int distinct_winners = sum_to_all(distinct_acquired ? 1 : 0);
  CHECK(distinct_winners == std::min(world_size, 2),
        "different UUID split groups did not acquire independently");
  all_wait();
  end_global_communications();
  distinct.release();

  /* One size-one subgroup may fail a contested reservation and immediately
     make progress on an independent UUID while the winner continues holding
     the original lock. There is deliberately no active-communicator barrier
     between the failed and successful local acquisitions. */
  begin_global_communications();
  all_wait();
  end_global_communications();
  DeviceUuidReservation held_or_contended;
  bool original = false;
  if (my_global_rank() == 0) original = held_or_contended.acquire(uuid_a, why, root);
  begin_global_communications();
  all_wait();
  end_global_communications();
  bool rejected = false;
  if (my_global_rank() != 0) rejected = !held_or_contended.acquire(uuid_a, why, root);
  DeviceUuidReservation independent;
  bool progressed = false;
  if (my_global_rank() == 1 && rejected)
    progressed = independent.acquire(uuid_b, why, root);
  begin_global_communications();
  all_wait();
  CHECK(sum_to_all(original ? 1 : 0) == 1,
        "asynchronous split fixture did not retain one original owner");
  CHECK(sum_to_all(rejected ? 1 : 0) == world_size - 1,
        "contending split subgroups did not observe the held reservation");
  CHECK(sum_to_all(progressed ? 1 : 0) == 1,
        "contending subgroup could not progress on an independent device lock");
  all_wait();
  end_global_communications();
  independent.release();
  held_or_contended.release();
  end_divide_parallel();
}

void cleanup_root(const std::string &root) {
  all_wait();
  if (my_rank() != 0) return;
  /* Lock files are persistent by production contract. Tests remove only their
     private directory after all reservations and ranks are finished. */
  DIR *directory = opendir(root.c_str());
  if (!directory) {
    ++failures;
    return;
  }
  while (dirent *entry = readdir(directory)) {
    if (!std::strcmp(entry->d_name, ".") || !std::strcmp(entry->d_name, "..")) continue;
    const std::string path = root + "/" + entry->d_name;
    if (unlink(path.c_str()) != 0) ++failures;
  }
  closedir(directory);
  if (rmdir(root.c_str()) != 0) ++failures;
}

} // namespace

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  std::string why;
  if (!DeviceUuidReservation::platform_supported(why)) {
    if (my_rank() == 0) std::printf("device_uuid_reservation: SKIP (%s)\n", why.c_str());
    return 0;
  }
  const std::string root = shared_root();
  CHECK(!root.empty(), "could not create shared lock test directory");
  test_normalization();
  test_local_security(root);
  all_wait();
  test_split_contention(root);
  cleanup_root(root);
  const int total = sum_to_all(failures);
  if (my_rank() == 0)
    std::printf("device_uuid_reservation: %s (%d failures)\n", total ? "FAIL" : "PASS", total);
  return total ? 1 : 0;
}
#else
int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  std::string why;
  if (DeviceUuidReservation::platform_supported(why)) return 2;
  if (my_rank() == 0)
    std::printf("device_uuid_reservation: PASS (secure POSIX locking unavailable)\n");
  return 0;
}
#endif
