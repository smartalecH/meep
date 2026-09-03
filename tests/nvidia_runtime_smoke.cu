/* Standalone accelerator runtime smoke test. It intentionally has no dependency on
   libmeep so it can be compiled before the Phase-1 backend API is finalized. */

#include "backend/nvidia/runtime.hpp"

#include <stdint.h>
#include <stdlib.h>

#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using meep::nvidia::current_memory_accounting;
using meep::nvidia::device_buffer;
using meep::nvidia::device_properties;
using meep::nvidia::device_scope;
using meep::nvidia::device_selection;
using meep::nvidia::enumerate_devices;
using meep::nvidia::event;
using meep::nvidia::fill_byte_async;
using meep::nvidia::memory_accounting;
using meep::nvidia::pinned_buffer;
using meep::nvidia::select_device;
using meep::nvidia::selection_source;
using meep::nvidia::stream;
using meep::nvidia::testing::clear_failure;
using meep::nvidia::testing::fail_next;
using meep::nvidia::testing::failure_point;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

class saved_environment {
public:
  explicit saved_environment(const char *name) : name_(name), existed_(false) {
    const char *value = std::getenv(name);
    if (value) {
      existed_ = true;
      value_ = value;
    }
  }
  ~saved_environment() {
    if (existed_)
      setenv(name_.c_str(), value_.c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

private:
  std::string name_;
  std::string value_;
  bool existed_;
};

void test_selection(int count) {
  saved_environment saved("MEEP_DEVICE_ID");
  unsetenv("MEEP_DEVICE_ID");
  const int collision_test_size = count > 1 ? count : 2;
  const bool collision_test_sharing = count == 1;

  const device_selection rank_selected = select_device(-1, count - 1, count, false);
  require(rank_selected.device == count - 1, "node-local rank selection failed");
  require(rank_selected.source == selection_source::node_local_rank,
          "node-local rank source not reported");
  require(!rank_selected.collective_collision_check_required,
          "automatic rank mapping should not require collision validation");

  setenv("MEEP_DEVICE_ID", "0", 1);
  const device_selection environment_selected =
      select_device(-1, collision_test_size - 1, collision_test_size, collision_test_sharing);
  require(environment_selected.device == 0, "MEEP_DEVICE_ID selection failed");
  require(environment_selected.source == selection_source::environment,
          "environment source not reported");
  require(environment_selected.collective_collision_check_required,
          "environment assignment did not request collective collision validation");

  const device_selection duplicate_environment =
      select_device(-1, 0, collision_test_size, collision_test_sharing);
  require(duplicate_environment.device == environment_selected.device,
          "duplicate environment assignment was not reproduced locally");
  require(duplicate_environment.collective_collision_check_required,
          "duplicate environment assignment was incorrectly declared collision-free");

  const device_selection explicit_selected =
      select_device(count - 1, 0, collision_test_size, collision_test_sharing);
  require(explicit_selected.device == count - 1, "explicit selection did not take precedence");
  require(explicit_selected.source == selection_source::explicit_option,
          "explicit source not reported");
  require(explicit_selected.collective_collision_check_required,
          "explicit assignment did not request collective collision validation");

  bool rejected = false;
  try {
    (void)select_device(-1, 0, count + 1, false);
  }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "oversubscribed node-local ranks were not rejected");

  const device_selection shared = select_device(-1, count, count + 1, true);
  require(shared.sharing, "explicitly allowed sharing was not reported");

  bool invalid_explicit = false;
  try {
    (void)select_device(count, 0, count, false);
  }
  catch (const std::out_of_range &) { invalid_explicit = true; }
  require(invalid_explicit, "out-of-range explicit device was not rejected");

  bool invalid_negative_explicit = false;
  try {
    (void)select_device(-2, 0, count, false);
  }
  catch (const std::invalid_argument &) { invalid_negative_explicit = true; }
  require(invalid_negative_explicit, "explicit device IDs below -1 were not rejected");

  setenv("MEEP_DEVICE_ID", "", 1);
  bool empty_environment = false;
  try {
    (void)select_device(-1, 0, count, false);
  }
  catch (const std::invalid_argument &) { empty_environment = true; }
  require(empty_environment, "empty MEEP_DEVICE_ID was not rejected");

  setenv("MEEP_DEVICE_ID", "not-a-device", 1);
  bool invalid_environment = false;
  try {
    (void)select_device(-1, 0, count, false);
  }
  catch (const std::invalid_argument &) { invalid_environment = true; }
  require(invalid_environment, "invalid MEEP_DEVICE_ID was not rejected");
}

void require_active_device(int expected, const std::string &message) {
  stream probe;
  require(probe.device() == expected, message);
}

void test_cross_device_ownership() {
  const size_t bytes = 4096;
  device_buffer storage;
  stream owner_stream;
  event owner_event;
  {
    device_scope owner(0);
    storage.allocate(bytes, 0);
    owner_stream = stream();
    owner_event = event();
    owner_event.record(owner_stream);
  }

  {
    device_scope other(1);
    owner_stream.synchronize();
    require_active_device(1, "stream synchronize did not restore the caller's device");
    owner_event.synchronize();
    require_active_device(1, "event synchronize did not restore the caller's device");
    storage.reset();
    require_active_device(1, "buffer reset did not restore the caller's device");
  }

  device_buffer *destroyed_on_other_device = NULL;
  {
    device_scope owner(0);
    destroyed_on_other_device = new device_buffer(bytes, 0);
  }
  {
    device_scope other(1);
    delete destroyed_on_other_device;
    require_active_device(1, "buffer destructor did not restore the caller's device");
  }

  stream *destroyed_stream = NULL;
  event *destroyed_event = NULL;
  {
    device_scope owner(0);
    destroyed_stream = new stream();
    destroyed_event = new event();
  }
  {
    device_scope other(1);
    delete destroyed_event;
    require_active_device(1, "event destructor did not restore the caller's device");
    delete destroyed_stream;
    require_active_device(1, "stream destructor did not restore the caller's device");
  }

  device_buffer move_destination;
  device_buffer move_source;
  stream *stream_destination = NULL;
  stream *stream_source = NULL;
  event *event_destination = NULL;
  event *event_source = NULL;
  {
    device_scope owner(0);
    move_destination.allocate(bytes, 0);
    move_source.allocate(2 * bytes, 0);
    stream_destination = new stream();
    stream_source = new stream();
    event_destination = new event();
    event_source = new event();
  }
  void *move_source_address = move_source.opaque_handle();
  {
    device_scope other(1);
    move_destination = std::move(move_source);
    require(move_destination.opaque_handle() == move_source_address &&
                move_source.opaque_handle() == NULL,
            "cross-current-device buffer move assignment lost ownership");
    require_active_device(1, "buffer move assignment did not restore the caller's device");

    *stream_destination = std::move(*stream_source);
    require(stream_destination->device() == 0 && stream_source->device() == -1,
            "cross-current-device stream move assignment lost ownership");
    require_active_device(1, "stream move assignment did not restore the caller's device");

    *event_destination = std::move(*event_source);
    require(event_destination->device() == 0 && event_source->device() == -1,
            "cross-current-device event move assignment lost ownership");
    require_active_device(1, "event move assignment did not restore the caller's device");

    delete event_source;
    delete event_destination;
    delete stream_source;
    delete stream_destination;
    move_destination.reset();
    require_active_device(1, "moved resource teardown did not restore the caller's device");
  }
}

void test_move_operations() {
  const size_t bytes = 4096;
  const memory_accounting before = current_memory_accounting();
  device_scope selected(0);

  stream source_stream;
  stream moved_stream(std::move(source_stream));
  require(source_stream.device() == -1, "moved-from stream still owns a handle");
  bool moved_stream_rejected = false;
  try {
    source_stream.synchronize();
  }
  catch (const std::logic_error &) { moved_stream_rejected = true; }
  require(moved_stream_rejected, "moved-from stream operation was not rejected");
  source_stream = std::move(moved_stream);
  require(moved_stream.device() == -1, "move-assigned stream still owns a handle");
  source_stream.synchronize();

  event source_event;
  event moved_event(std::move(source_event));
  require(source_event.device() == -1, "moved-from event still owns a handle");
  bool moved_event_rejected = false;
  try {
    source_event.synchronize();
  }
  catch (const std::logic_error &) { moved_event_rejected = true; }
  require(moved_event_rejected, "moved-from event operation was not rejected");
  source_event = std::move(moved_event);
  require(moved_event.device() == -1, "move-assigned event still owns a handle");
  source_event.record(source_stream);
  source_event.synchronize();

  device_buffer source_buffer(bytes, 0);
  void *source_address = source_buffer.opaque_handle();
  device_buffer moved_buffer(std::move(source_buffer));
  require(source_buffer.opaque_handle() == NULL, "moved-from device buffer still owns storage");
  require(moved_buffer.opaque_handle() == source_address, "device-buffer move changed its address");

  device_buffer assigned_buffer(bytes, 0);
  assigned_buffer = std::move(moved_buffer);
  require(moved_buffer.opaque_handle() == NULL,
          "move-assigned device buffer still owns storage");
  require(assigned_buffer.opaque_handle() == source_address,
          "device-buffer move assignment changed its address");

  pinned_buffer source_pinned(bytes);
  void *source_pinned_address = source_pinned.data();
  pinned_buffer moved_pinned(std::move(source_pinned));
  require(source_pinned.data() == NULL, "moved-from pinned buffer still owns storage");
  require(moved_pinned.data() == source_pinned_address, "pinned-buffer move changed its address");

  pinned_buffer assigned_pinned(bytes);
  assigned_pinned = std::move(moved_pinned);
  require(moved_pinned.data() == NULL, "move-assigned pinned buffer still owns storage");
  require(assigned_pinned.data() == source_pinned_address,
          "pinned-buffer move assignment changed its address");

  assigned_buffer.reset();
  assigned_pinned.reset();
  const memory_accounting after = current_memory_accounting();
  require(after.device_bytes_current == before.device_bytes_current,
          "device accounting changed across move operations");
  require(after.pinned_bytes_current == before.pinned_bytes_current,
          "pinned accounting changed across move operations");
}

void test_checked_release_failures(int other_device) {
  const size_t bytes = 4096;
  const memory_accounting before = current_memory_accounting();
  device_scope selected(0);

  device_buffer device_storage(bytes, 0);
  void *device_address = device_storage.opaque_handle();
  bool device_reset_failed = false;
  {
    device_scope other(other_device);
    fail_next(failure_point::device_free);
    try {
      device_storage.reset();
    }
    catch (const meep::nvidia::runtime_error &) { device_reset_failed = true; }
    require_active_device(other_device,
                          "failed cross-current-device reset did not restore the caller");
  }
  require(device_reset_failed, "injected device-free failure was not reported");
  require(device_storage.opaque_handle() == device_address && device_storage.size() == bytes,
          "failed device reset discarded the live allocation");
  require(current_memory_accounting().device_bytes_current == before.device_bytes_current + bytes,
          "failed device reset decremented accounting");

  fail_next(failure_point::device_free);
  bool device_allocate_failed = false;
  try {
    device_storage.allocate(2 * bytes, 0);
  }
  catch (const meep::nvidia::runtime_error &) { device_allocate_failed = true; }
  require(device_allocate_failed, "allocate did not propagate reset failure");
  require(device_storage.opaque_handle() == device_address && device_storage.size() == bytes,
          "allocate reset failure replaced the existing allocation");

  fail_next(failure_point::device_allocate);
  bool device_reallocate_failed = false;
  try {
    device_storage.allocate(2 * bytes, 0);
  }
  catch (const meep::nvidia::runtime_error &) { device_reallocate_failed = true; }
  require(device_reallocate_failed, "injected device-allocation failure was not reported");
  require(device_storage.opaque_handle() == NULL && device_storage.size() == 0,
          "failed replacement allocation did not leave an empty buffer");
  require(current_memory_accounting().device_bytes_current == before.device_bytes_current,
          "failed replacement allocation left stale device accounting");

  device_storage.allocate(bytes, 0);
  device_address = device_storage.opaque_handle();

  device_buffer move_source(2 * bytes, 0);
  void *move_source_address = move_source.opaque_handle();
  {
    device_scope other(other_device);
    fail_next(failure_point::device_free);
    device_storage = std::move(move_source);
    require_active_device(other_device,
                          "failed cross-current-device move did not restore the caller's device");
  }
  require(device_storage.opaque_handle() == device_address,
          "failed move-assignment cleanup discarded its destination allocation");
  require(move_source.opaque_handle() == move_source_address,
          "failed move-assignment cleanup consumed its source allocation");
  device_storage.reset();
  move_source.reset();

  if (other_device != 0) {
    device_buffer restore_failure(bytes, 0);
    fail_next(failure_point::device_restore);
    bool restore_failed = false;
    {
      device_scope other(other_device);
      try {
        restore_failure.reset();
      }
      catch (const meep::nvidia::runtime_error &) { restore_failed = true; }
    }
    require(restore_failed, "injected device restore failure was not reported");
    require(restore_failure.opaque_handle() == NULL && restore_failure.size() == 0,
            "successful free followed by restore failure retained a stale allocation");
  }

  pinned_buffer pinned(bytes);
  void *pinned_address = pinned.data();
  fail_next(failure_point::pinned_free);
  bool pinned_reset_failed = false;
  try {
    pinned.reset();
  }
  catch (const meep::nvidia::runtime_error &) { pinned_reset_failed = true; }
  require(pinned_reset_failed, "injected pinned-free failure was not reported");
  require(pinned.data() == pinned_address && pinned.size() == bytes,
          "failed pinned reset discarded the live allocation");

  fail_next(failure_point::pinned_allocate);
  bool pinned_reallocate_failed = false;
  try {
    pinned.allocate(2 * bytes);
  }
  catch (const meep::nvidia::runtime_error &) { pinned_reallocate_failed = true; }
  require(pinned_reallocate_failed, "injected pinned-allocation failure was not reported");
  require(pinned.data() == NULL && pinned.size() == 0,
          "failed replacement pinned allocation did not leave an empty buffer");
  require(current_memory_accounting().pinned_bytes_current == before.pinned_bytes_current,
          "failed replacement pinned allocation left stale accounting");

  pinned.allocate(bytes);
  pinned_address = pinned.data();

  pinned_buffer pinned_source(2 * bytes);
  void *pinned_source_address = pinned_source.data();
  fail_next(failure_point::pinned_free);
  pinned = std::move(pinned_source);
  require(pinned.data() == pinned_address,
          "failed pinned move-assignment cleanup discarded its destination allocation");
  require(pinned_source.data() == pinned_source_address,
          "failed pinned move-assignment cleanup consumed its source allocation");
  pinned.reset();
  pinned_source.reset();

  clear_failure();
  const memory_accounting after = current_memory_accounting();
  require(after.device_bytes_current == before.device_bytes_current,
          "device accounting did not recover after checked failures");
  require(after.pinned_bytes_current == before.pinned_bytes_current,
          "pinned accounting did not recover after checked failures");
}

void test_error_paths(int visible_device_count) {
  const size_t bytes = 4096;
  device_scope device_zero(0);
  device_buffer storage(bytes, 0);
  stream stream_zero;

  bool range_failed = false;
  try {
    fill_byte_async(storage, bytes - 1, 0, 2, stream_zero);
  }
  catch (const std::out_of_range &) { range_failed = true; }
  require(range_failed, "out-of-range fill was not rejected");

  if (visible_device_count >= 2) {
    stream *stream_one = NULL;
    {
      device_scope device_one(1);
      stream_one = new stream();
    }
    bool wrong_device_failed = false;
    try {
      fill_byte_async(storage, 0, 0, bytes, *stream_one);
    }
    catch (const std::invalid_argument &) { wrong_device_failed = true; }
    require(wrong_device_failed, "wrong-device stream was not rejected");
    delete stream_one;
  }

  std::vector<unsigned char> source(bytes, 0x5a);
  fail_next(failure_point::host_to_device_copy);
  bool copy_failed = false;
  try {
    copy_host_to_device_async(storage, 0, source.data(), bytes, stream_zero);
  }
  catch (const meep::nvidia::runtime_error &) { copy_failed = true; }
  require(copy_failed, "injected host-to-device copy failure was not reported");
  copy_host_to_device_async(storage, 0, source.data(), bytes, stream_zero);
  stream_zero.synchronize();
}

void test_portable_pinned_memory(int visible_device_count) {
  const size_t bytes = 4096;
  pinned_buffer source(bytes);
  pinned_buffer destination(bytes);
  std::memset(source.data(), 0x5a, bytes);

  const int devices_to_test = visible_device_count < 2 ? visible_device_count : 2;
  for (int device = 0; device < devices_to_test; ++device) {
    device_scope selected(device);
    stream transfer;
    device_buffer storage(bytes, device);
    meep::nvidia::copy_host_to_device_async(storage, 0, source.data(), bytes, transfer);
    meep::nvidia::copy_device_to_host_async(destination.data(), storage, 0, bytes, transfer);
    transfer.synchronize();
    require(std::memcmp(source.data(), destination.data(), bytes) == 0,
            "portable pinned memory failed on a visible accelerator device");
  }
}

void round_trip(int device) {
  const size_t elements = 1u << 20;
  const size_t bytes = elements * sizeof(uint32_t);
  const memory_accounting before = current_memory_accounting();

  {
    device_scope selected(device);
    stream transfer;
    stream compute;
    event copied;
    device_buffer storage(bytes, device);
    device_buffer mirror(bytes, device);
    pinned_buffer source(bytes);
    pinned_buffer destination(bytes);

    uint32_t *source_values = static_cast<uint32_t *>(source.data());
    uint32_t *destination_values = static_cast<uint32_t *>(destination.data());
    for (size_t i = 0; i < elements; ++i)
      source_values[i] = static_cast<uint32_t>((i * 2654435761u) ^ (device * 0x9e3779b9u));
    std::memset(destination_values, 0, bytes);

    meep::nvidia::copy_host_to_device_async(storage, 0, source_values, bytes, transfer);
    copied.record(transfer);
    copied.wait(compute);
    meep::nvidia::copy_device_to_device_async(mirror, 0, storage, 0, bytes, compute);
    meep::nvidia::copy_device_to_host_async(destination_values, mirror, 0, bytes, compute);
    compute.synchronize();
    require(copied.ready(), "recorded accelerator event did not become ready");
    require(std::memcmp(source_values, destination_values, bytes) == 0,
            "host/device round trip changed data");

    fill_byte_async(storage, 0, 0, bytes, transfer);
    meep::nvidia::copy_device_to_host_async(destination_values, storage, 0, bytes, transfer);
    transfer.synchronize();
    for (size_t i = 0; i < elements; ++i)
      require(destination_values[i] == 0, "asynchronous device fill failed");

    const memory_accounting during = current_memory_accounting();
    require(during.device_bytes_current >= before.device_bytes_current + 2 * bytes,
            "device allocation accounting did not increase");
    require(during.pinned_bytes_current >= before.pinned_bytes_current + 2 * bytes,
            "pinned allocation accounting did not increase");
  }

  const memory_accounting after = current_memory_accounting();
  require(after.device_bytes_current == before.device_bytes_current,
          "device allocation accounting did not return to baseline");
  require(after.pinned_bytes_current == before.pinned_bytes_current,
          "pinned allocation accounting did not return to baseline");
}

} // namespace

int main() {
  try {
    const std::vector<device_properties> devices = enumerate_devices();
#if defined(MEEP_HIP_PORTABILITY)
    require(!devices.empty(), "HIP smoke test requires one visible AMD device");
#else
    require(devices.size() >= 2, "smoke test requires at least two visible NVIDIA devices");
#endif

    std::cout << "accelerator driver/runtime: " << meep::nvidia::driver_version() << "/"
              << meep::nvidia::runtime_version() << "\n";
    for (size_t i = 0; i < devices.size(); ++i) {
      const device_properties &d = devices[i];
#if defined(MEEP_HIP_PORTABILITY)
      require(!d.uuid.empty(), "visible accelerator has an empty UUID");
      for (size_t j = 0; j < i; ++j)
        require(d.uuid != devices[j].uuid, "visible accelerators have duplicate UUIDs");
#endif
      std::cout << "device " << d.id << ": " << d.name << " cc " << d.compute_major << "."
                << d.compute_minor << ", bytes=" << d.total_memory << ", uuid=" << d.uuid
                << ", peer01="
                << (devices.size() > 1
                        ? meep::nvidia::devices_can_access_peer(d.id, int((i + 1) % devices.size()))
                        : false)
                << "\n";
    }

    test_selection(static_cast<int>(devices.size()));
    if (devices.size() >= 2) test_cross_device_ownership();
    test_move_operations();
    const int other_device = devices.size() >= 2 ? 1 : 0;
    test_checked_release_failures(other_device);
    test_error_paths(static_cast<int>(devices.size()));
    test_portable_pinned_memory(static_cast<int>(devices.size()));
    for (size_t i = 0; i < devices.size(); ++i) {
      round_trip(static_cast<int>(i));
      std::cout << "round-trip device " << i << ": PASS\n";
    }

    const memory_accounting accounting = current_memory_accounting();
    require(accounting.device_bytes_current == 0, "device bytes remain allocated at process end");
    require(accounting.pinned_bytes_current == 0, "pinned bytes remain allocated at process end");
    std::cout << "peak device bytes=" << accounting.device_bytes_peak
              << ", peak pinned bytes=" << accounting.pinned_bytes_peak << "\n";
    std::cout << "PASS\n";
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
