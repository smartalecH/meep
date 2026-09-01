/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/device_uuid_reservation.hpp"

#include "config.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <stdint.h>
#include <utility>
#include <vector>

#if defined(HAVE_FCNTL_H)
#include <fcntl.h>
#endif
#if defined(HAVE_SYS_FILE_H)
#include <sys/file.h>
#endif
#if defined(HAVE_SYS_STAT_H)
#include <sys/stat.h>
#endif
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif

namespace meep {
namespace {

#if defined(HAVE_FCNTL_H) && defined(HAVE_SYS_FILE_H) && defined(HAVE_SYS_STAT_H) && \
    defined(HAVE_UNISTD_H) && defined(HAVE_CLOSE) && defined(HAVE_FLOCK) &&          \
    defined(HAVE_FSTAT) && defined(HAVE_FTRUNCATE) && defined(HAVE_GETEUID) &&       \
    defined(HAVE_LSEEK) && defined(HAVE_LSTAT) && defined(HAVE_MKDIR) &&             \
    defined(HAVE_OPENAT) && defined(HAVE_WRITE) && defined(O_NOFOLLOW) &&            \
    defined(O_CLOEXEC) && defined(O_DIRECTORY) && defined(LOCK_EX) && defined(LOCK_NB)
const bool supported = true;
#else
const bool supported = false;
#endif

std::mutex registry_mutex;
std::set<std::string> process_registry;
std::mutex override_mutex;
std::string testing_uuid_override;
std::string testing_lock_root_override;

uint32_t rotate_right(uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32 - bits));
}

std::string sha256_hex(const std::string &input) {
  static const uint32_t constants[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
      0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
      0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
      0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
      0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  std::vector<unsigned char> bytes(input.begin(), input.end());
  const uint64_t bit_length = uint64_t(bytes.size()) * 8;
  bytes.push_back(0x80);
  while (bytes.size() % 64 != 56) bytes.push_back(0);
  for (int i = 7; i >= 0; --i) bytes.push_back((bit_length >> (8 * i)) & 0xffu);
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  for (size_t offset = 0; offset < bytes.size(); offset += 64) {
    uint32_t words[64] = {};
    for (unsigned i = 0; i < 16; ++i)
      words[i] = uint32_t(bytes[offset + 4 * i]) << 24 |
                 uint32_t(bytes[offset + 4 * i + 1]) << 16 |
                 uint32_t(bytes[offset + 4 * i + 2]) << 8 |
                 uint32_t(bytes[offset + 4 * i + 3]);
    for (unsigned i = 16; i < 64; ++i) {
      const uint32_t s0 = rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18) ^
                          (words[i - 15] >> 3);
      const uint32_t s1 = rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^
                          (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (unsigned i = 0; i < 64; ++i) {
      const uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      const uint32_t choose = (e & f) ^ (~e & g);
      const uint32_t t1 = h + s1 + choose + constants[i] + words[i];
      const uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + majority;
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (uint32_t word : state) out << std::setw(8) << word;
  return out.str();
}

std::string error_text(const char *what) {
  return std::string(what) + ": " + std::strerror(errno);
}

} // namespace

DeviceUuidReservation::DeviceUuidReservation() : fd_(-1), directory_fd_(-1) {}
DeviceUuidReservation::~DeviceUuidReservation() { release(); }

DeviceUuidReservation::DeviceUuidReservation(DeviceUuidReservation &&other) noexcept
    : fd_(other.fd_), directory_fd_(other.directory_fd_),
      normalized_uuid_(std::move(other.normalized_uuid_)), lock_path_(std::move(other.lock_path_)) {
  other.fd_ = -1;
  other.directory_fd_ = -1;
  other.normalized_uuid_.clear();
  other.lock_path_.clear();
}

DeviceUuidReservation &DeviceUuidReservation::operator=(DeviceUuidReservation &&other) noexcept {
  if (this == &other) return *this;
  release();
  fd_ = other.fd_;
  directory_fd_ = other.directory_fd_;
  normalized_uuid_ = std::move(other.normalized_uuid_);
  lock_path_ = std::move(other.lock_path_);
  other.fd_ = -1;
  other.directory_fd_ = -1;
  other.normalized_uuid_.clear();
  other.lock_path_.clear();
  return *this;
}

bool DeviceUuidReservation::platform_supported(std::string &why) {
  why.clear();
  if (!supported) why = "secure device UUID advisory locks are unsupported on this platform";
  if (std::getenv("MEEP_TEST_DEVICE_LOCK_UNSUPPORTED")) {
    why = "secure device UUID advisory locks were disabled for testing";
    return false;
  }
  return supported;
}

bool DeviceUuidReservation::normalize_uuid(const std::string &uuid, std::string &normalized,
                                           std::string &why) {
  why.clear();
  normalized.clear();
  size_t begin = 0;
  if (uuid.size() >= 4 && (uuid.substr(0, 4) == "GPU-" || uuid.substr(0, 4) == "gpu-")) begin = 4;
  for (size_t i = begin; i < uuid.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(uuid[i]);
    if (std::isxdigit(c)) normalized.push_back(char(std::tolower(c)));
    else if (c == '-' || c == ':') continue;
    else {
      why = "device UUID contains a non-hexadecimal character";
      normalized.clear();
      return false;
    }
  }
  if (normalized.size() != 32) {
    why = "device UUID must contain exactly 16 bytes";
    normalized.clear();
    return false;
  }
  return true;
}

bool DeviceUuidReservation::acquire(const std::string &uuid, std::string &why,
                                    const std::string &test_lock_root) {
  why.clear();
  if (valid()) {
    why = "device UUID reservation is already held";
    return false;
  }
  if (!platform_supported(why)) return false;
  std::string normalized;
  if (!normalize_uuid(uuid, normalized, why)) return false;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (!process_registry.insert(normalized).second) {
      why = "this process already owns the device UUID reservation";
      return false;
    }
  }
  normalized_uuid_ = normalized;

#if defined(HAVE_FCNTL_H) && defined(HAVE_SYS_FILE_H) && defined(HAVE_SYS_STAT_H) && \
    defined(HAVE_UNISTD_H) && defined(HAVE_CLOSE) && defined(HAVE_FLOCK) &&          \
    defined(HAVE_FSTAT) && defined(HAVE_FTRUNCATE) && defined(HAVE_GETEUID) &&       \
    defined(HAVE_LSEEK) && defined(HAVE_LSTAT) && defined(HAVE_MKDIR) &&             \
    defined(HAVE_OPENAT) && defined(HAVE_WRITE) && defined(O_NOFOLLOW) &&            \
    defined(O_CLOEXEC) && defined(O_DIRECTORY) && defined(LOCK_EX) && defined(LOCK_NB)
  std::string root = test_lock_root;
  if (root.empty()) {
    const char *runtime = std::getenv("XDG_RUNTIME_DIR");
    if (runtime && *runtime) root = std::string(runtime) + "/meep-gpu-locks";
    else {
      std::ostringstream fallback;
      fallback << "/tmp/meep-gpu-locks-" << static_cast<unsigned long>(geteuid());
      root = fallback.str();
    }
  }
  if (mkdir(root.c_str(), 0700) != 0 && errno != EEXIST) {
    why = error_text("cannot create device lock directory");
    release();
    return false;
  }
  struct stat directory_status;
  if (lstat(root.c_str(), &directory_status) != 0 || !S_ISDIR(directory_status.st_mode) ||
      directory_status.st_uid != geteuid() || (directory_status.st_mode & 077) != 0) {
    why = "device lock directory is not a secure UID-owned directory";
    release();
    return false;
  }
  directory_fd_ = open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (directory_fd_ < 0) {
    why = error_text("cannot open device lock directory");
    release();
    return false;
  }
  struct stat opened_directory_status;
  if (fstat(directory_fd_, &opened_directory_status) != 0 ||
      !S_ISDIR(opened_directory_status.st_mode) ||
      opened_directory_status.st_uid != geteuid() ||
      (opened_directory_status.st_mode & 077) != 0 ||
      opened_directory_status.st_dev != directory_status.st_dev ||
      opened_directory_status.st_ino != directory_status.st_ino) {
    why = "opened device lock directory does not match the validated secure directory";
    release();
    return false;
  }
  const std::string filename = sha256_hex(normalized) + ".lock";
  fd_ = openat(directory_fd_, filename.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd_ < 0) {
    why = error_text("cannot open device UUID lock file");
    release();
    return false;
  }
  struct stat status;
  if (fstat(fd_, &status) != 0 || !S_ISREG(status.st_mode) || status.st_uid != geteuid() ||
      status.st_nlink != 1 || (status.st_mode & 077) != 0) {
    why = "device UUID lock file failed type/owner/link/mode validation";
    release();
    return false;
  }
  if (flock(fd_, LOCK_EX | LOCK_NB) != 0) {
    why = errno == EWOULDBLOCK ? "device UUID is already reserved"
                               : error_text("device UUID flock failed");
    release();
    return false;
  }
  std::ostringstream metadata;
  metadata << "pid=" << long(getpid()) << " uuid=" << normalized << "\n";
  const std::string text = metadata.str();
  if (ftruncate(fd_, 0) != 0 || lseek(fd_, 0, SEEK_SET) < 0 ||
      write(fd_, text.data(), text.size()) != ssize_t(text.size())) {
    why = error_text("cannot write device UUID lock metadata");
    release();
    return false;
  }
  lock_path_ = root + "/" + filename;
  return true;
#else
  (void)test_lock_root;
  why = "secure device UUID advisory locks are unsupported on this platform";
  release();
  return false;
#endif
}

void DeviceUuidReservation::release() noexcept {
#if defined(HAVE_UNISTD_H)
  if (fd_ >= 0) close(fd_);
  if (directory_fd_ >= 0) close(directory_fd_);
#endif
  fd_ = -1;
  directory_fd_ = -1;
  if (!normalized_uuid_.empty()) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    process_registry.erase(normalized_uuid_);
  }
  normalized_uuid_.clear();
  lock_path_.clear();
}

bool DeviceUuidReservation::valid() const { return fd_ >= 0; }
const std::string &DeviceUuidReservation::normalized_uuid() const { return normalized_uuid_; }
const std::string &DeviceUuidReservation::lock_path() const { return lock_path_; }

namespace device_uuid_testing {

void set_overrides(const char *uuid, const char *lock_root) {
  std::lock_guard<std::mutex> lock(override_mutex);
  testing_uuid_override = uuid ? uuid : "";
  testing_lock_root_override = lock_root ? lock_root : "";
}

std::string uuid_override() {
  std::lock_guard<std::mutex> lock(override_mutex);
  return testing_uuid_override;
}

std::string lock_root_override() {
  std::lock_guard<std::mutex> lock(override_mutex);
  return testing_lock_root_override;
}

} // namespace device_uuid_testing

} // namespace meep
