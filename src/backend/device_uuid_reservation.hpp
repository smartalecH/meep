/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#ifndef MEEP_BACKEND_DEVICE_UUID_RESERVATION_HPP
#define MEEP_BACKEND_DEVICE_UUID_RESERVATION_HPP

#include <string>

namespace meep {

class DeviceUuidReservation {
public:
  DeviceUuidReservation();
  ~DeviceUuidReservation();
  DeviceUuidReservation(DeviceUuidReservation &&other) noexcept;
  DeviceUuidReservation &operator=(DeviceUuidReservation &&other) noexcept;

  bool acquire(const std::string &uuid, std::string &why,
               const std::string &test_lock_root = std::string());
  void release() noexcept;
  bool valid() const;
  const std::string &normalized_uuid() const;
  const std::string &lock_path() const;

  static bool platform_supported(std::string &why);
  static bool normalize_uuid(const std::string &uuid, std::string &normalized,
                             std::string &why);

private:
  DeviceUuidReservation(const DeviceUuidReservation &);
  DeviceUuidReservation &operator=(const DeviceUuidReservation &);
  int fd_;
  int directory_fd_;
  std::string normalized_uuid_;
  std::string lock_path_;
};

namespace device_uuid_testing {
/* Private deterministic seam for backend admission tests. Empty values restore
   the production CUDA UUID and lock-root selection. */
void set_overrides(const char *uuid, const char *lock_root);
std::string uuid_override();
std::string lock_root_override();
} // namespace device_uuid_testing

} // namespace meep

#endif
