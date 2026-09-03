/* Standalone role-arena smoke test. It has no dependency on libmeep or the
   Phase-1 backend API and performs all malformed-plan checks before device use. */

#include "backend/nvidia/arena.hpp"

#include <stdint.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using meep::nvidia::allocation_id;
using meep::nvidia::allocation_layout;
using meep::nvidia::allocation_request;
using meep::nvidia::arena_accounting;
using meep::nvidia::arena_element_type;
using meep::nvidia::arena_plan;
using meep::nvidia::arena_role;
using meep::nvidia::arena_storage_precision;
using meep::nvidia::device_allocation;
using meep::nvidia::device_arenas;
using meep::nvidia::device_buffer;
using meep::nvidia::device_properties;
using meep::nvidia::device_scope;
using meep::nvidia::enumerate_devices;
using meep::nvidia::memory_accounting;
using meep::nvidia::no_allocation;
using meep::nvidia::pinned_buffer;
using meep::nvidia::stream;
using meep::nvidia::testing::clear_failure;
using meep::nvidia::testing::fail_next;
using meep::nvidia::testing::failure_point;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void require_throws(Function function, const std::string &message) {
  bool threw = false;
  try {
    function();
  }
  catch (const Exception &) {
    threw = true;
  }
  require(threw, message);
}

std::vector<allocation_request> valid_requests() {
  std::vector<allocation_request> requests;
  requests.push_back(allocation_request(30, arena_role::field, 7, 4));
  requests.push_back(allocation_request(10, arena_role::field, 17, 16, no_allocation,
                                        arena_element_type::realnum_value,
                                        arena_storage_precision::f32));
  requests.push_back(allocation_request(40, arena_role::field, 17, 64, 10,
                                        arena_element_type::realnum_value,
                                        arena_storage_precision::f32));
  requests.push_back(allocation_request(20, arena_role::material, 9, 8));
  return requests;
}

void test_cpu_planning() {
  const arena_plan plan(valid_requests());
  require(plan.allocations().size() == 4, "planner lost an allocation");
  require(plan.canonical_allocation_count() == 3, "canonical allocation count is wrong");
  require(plan.alias_count() == 1, "alias count is wrong");
  require(plan.roles().size() == 2, "nonempty role count is wrong");

  const allocation_layout &owner = plan.layout(10);
  const allocation_layout &alias = plan.layout(40);
  const allocation_layout &following = plan.layout(30);
  require(owner.offset == 0 && owner.effective_alignment == 64,
          "alias alignment was not promoted onto canonical storage");
  require(alias.offset == owner.offset && alias.canonical_id == owner.id,
          "alias did not reuse its canonical layout");
  require(following.offset == 20, "field layout did not follow sorted-ID order");
  require(plan.contains(20) && !plan.contains(999), "plan membership query failed");
  require_throws<std::out_of_range>([&]() { (void)plan.layout(999); },
                                    "unknown allocation ID was accepted");

  std::vector<allocation_request> reversed = valid_requests();
  std::reverse(reversed.begin(), reversed.end());
  const arena_plan reordered(reversed);
  require(reordered.allocations().size() == plan.allocations().size(),
          "reordered plan changed allocation count");
  for (size_t i = 0; i < plan.allocations().size(); ++i) {
    const allocation_layout &left = plan.allocations()[i];
    const allocation_layout &right = reordered.allocations()[i];
    require(left.id == right.id && left.role == right.role && left.offset == right.offset &&
                left.bytes == right.bytes && left.alignment == right.alignment &&
                left.effective_alignment == right.effective_alignment &&
                left.canonical_id == right.canonical_id &&
                left.element_type == right.element_type &&
                left.storage_precision == right.storage_precision,
            "plan depends on request order");
  }
  require(reordered.total_high_water_bytes() == plan.total_high_water_bytes() &&
              reordered.total_reserved_bytes() == plan.total_reserved_bytes() &&
              reordered.roles().size() == plan.roles().size(),
          "reordered plan changed aggregate layout");
  for (size_t i = 0; i < plan.roles().size(); ++i) {
    const meep::nvidia::role_layout &left = plan.roles()[i];
    const meep::nvidia::role_layout &right = reordered.roles()[i];
    require(left.role == right.role && left.high_water_bytes == right.high_water_bytes &&
                left.reserved_bytes == right.reserved_bytes &&
                left.max_alignment == right.max_alignment &&
                left.canonical_allocations == right.canonical_allocations &&
                left.aliases == right.aliases,
            "reordered plan changed a role layout");
  }

  {
    std::vector<allocation_request> requests;
    requests.push_back(allocation_request(1, arena_role::field, 16, 8, no_allocation,
                                          arena_element_type::realnum_value,
                                          arena_storage_precision::f32));
    requests.push_back(allocation_request(2, arena_role::field, 16, 32, 1,
                                          arena_element_type::realnum_value,
                                          arena_storage_precision::f32));
    requests.push_back(allocation_request(3, arena_role::field, 16, 128, 2,
                                          arena_element_type::realnum_value,
                                          arena_storage_precision::f32));
    const arena_plan chain(requests);
    require(chain.layout(2).canonical_id == 1 && chain.layout(3).canonical_id == 1,
            "indirect alias chain did not resolve to its root");
    require(chain.layout(1).effective_alignment == 128 &&
                chain.layout(2).effective_alignment == 128 &&
                chain.layout(3).effective_alignment == 128,
            "indirect alias alignment was not promoted to the root");
  }

  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 4, 4));
        requests.push_back(allocation_request(1, arena_role::field, 4, 4));
        (void)arena_plan(requests);
      },
      "duplicate IDs were accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(no_allocation, arena_role::field, 4, 4));
        (void)arena_plan(requests);
      },
      "reserved ID was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 0, 4));
        (void)arena_plan(requests);
      },
      "zero-byte allocation was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 4, 3));
        (void)arena_plan(requests);
      },
      "non-power-of-two alignment was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, static_cast<arena_role>(99), 4, 4));
        (void)arena_plan(requests);
      },
      "invalid arena role was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 4, 4, no_allocation,
                                              static_cast<arena_element_type>(99),
                                              arena_storage_precision::f32));
        (void)arena_plan(requests);
      },
      "invalid arena element type was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 4, 4, no_allocation,
                                              arena_element_type::realnum_value,
                                              static_cast<arena_storage_precision>(99)));
        (void)arena_plan(requests);
      },
      "invalid arena storage precision was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 4, 4, 2));
        (void)arena_plan(requests);
      },
      "missing alias target was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 4, 4, 2));
        requests.push_back(allocation_request(2, arena_role::field, 4, 4, 1));
        (void)arena_plan(requests);
      },
      "alias cycle was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 4, 4, 1));
        (void)arena_plan(requests);
      },
      "self alias cycle was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 4, 4, 2));
        requests.push_back(allocation_request(2, arena_role::field, 4, 4, 3));
        requests.push_back(allocation_request(3, arena_role::field, 4, 4, 1));
        (void)arena_plan(requests);
      },
      "indirect alias cycle was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 4, 4));
        requests.push_back(allocation_request(2, arena_role::field, 8, 4, 1));
        (void)arena_plan(requests);
      },
      "alias byte mismatch was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 4, 4));
        requests.push_back(allocation_request(2, arena_role::material, 4, 4, 1));
        (void)arena_plan(requests);
      },
      "alias role mismatch was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 8, 8, no_allocation,
                                              arena_element_type::float64,
                                              arena_storage_precision::f64));
        requests.push_back(allocation_request(2, arena_role::field, 8, 8, 1,
                                              arena_element_type::int32,
                                              arena_storage_precision::f64));
        (void)arena_plan(requests);
      },
      "alias element-type mismatch was accepted");
  require_throws<std::invalid_argument>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, 8, 8, no_allocation,
                                              arena_element_type::realnum_value,
                                              arena_storage_precision::f32));
        requests.push_back(allocation_request(2, arena_role::field, 8, 8, 1,
                                              arena_element_type::realnum_value,
                                              arena_storage_precision::f64));
        (void)arena_plan(requests);
      },
      "alias storage-precision mismatch was accepted");
  require_throws<std::overflow_error>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(
            allocation_request(1, arena_role::field, std::numeric_limits<size_t>::max(), 1));
        requests.push_back(allocation_request(2, arena_role::field, 1, 1));
        (void)arena_plan(requests);
      },
      "arena high-water overflow was accepted");
  require_throws<std::overflow_error>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(
            allocation_request(1, arena_role::field, std::numeric_limits<size_t>::max() - 6, 1));
        requests.push_back(allocation_request(2, arena_role::field, 1, 8));
        (void)arena_plan(requests);
      },
      "arena alignment-rounding overflow was accepted");
  require_throws<std::overflow_error>(
      []() {
        std::vector<allocation_request> requests;
        requests.push_back(
            allocation_request(1, arena_role::field, std::numeric_limits<size_t>::max(), 2));
        (void)arena_plan(requests);
      },
      "per-role reservation overflow was accepted");
  require_throws<std::overflow_error>(
      []() {
        const size_t half_plus_one = std::numeric_limits<size_t>::max() / 2 + 1;
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, half_plus_one, 1));
        requests.push_back(allocation_request(2, arena_role::material, half_plus_one, 1));
        (void)arena_plan(requests);
      },
      "global high-water overflow was accepted");
  require_throws<std::overflow_error>(
      []() {
        const size_t half = std::numeric_limits<size_t>::max() / 2;
        std::vector<allocation_request> requests;
        requests.push_back(allocation_request(1, arena_role::field, half, 2));
        requests.push_back(allocation_request(2, arena_role::material, half, 2));
        (void)arena_plan(requests);
      },
      "global reservation overflow was accepted");
}

std::vector<allocation_request> gpu_requests() {
  std::vector<allocation_request> requests;
  requests.push_back(allocation_request(1, arena_role::field, 4096, 256, no_allocation,
                                        arena_element_type::realnum_value,
                                        arena_storage_precision::f32));
  requests.push_back(allocation_request(2, arena_role::field, 1024, 64, no_allocation,
                                        arena_element_type::realnum_value,
                                        arena_storage_precision::f32));
  requests.push_back(allocation_request(3, arena_role::field, 4096, 512, 1,
                                        arena_element_type::realnum_value,
                                        arena_storage_precision::f32));
  requests.push_back(allocation_request(4, arena_role::material, 2048, 128, no_allocation,
                                        arena_element_type::float64, arena_storage_precision::f64));
  return requests;
}

void test_preflight_failure(int device) {
  std::vector<allocation_request> requests;
  requests.push_back(
      allocation_request(99, arena_role::scratch, std::numeric_limits<size_t>::max(), 1));
  const arena_plan impossible(requests);
  const memory_accounting before = meep::nvidia::current_memory_accounting();
  bool preflight_failed = false;
  try {
    (void)device_arenas(impossible, device);
  }
  catch (const std::runtime_error &error) {
    preflight_failed = std::string(error.what()).find("preflight") != std::string::npos;
  }
  require(preflight_failed, "impossible allocation did not fail during memory preflight");
  const memory_accounting after = meep::nvidia::current_memory_accounting();
  require(after.device_bytes_current == before.device_bytes_current,
          "preflight failure allocated device memory");
}

void test_device(int device, int other_device) {
  const arena_plan plan(gpu_requests());
  const memory_accounting before = meep::nvidia::current_memory_accounting();
  {
    device_scope selected(device);
    stream transfer;
    device_arenas initial(plan, device, 1024 * 1024);
    device_arenas arenas(std::move(initial));
    require(initial.device() == -1, "moved-from arenas retained ownership");
    const arena_accounting &accounting = arenas.accounting();
    require(accounting.device == device, "arena accounting reports wrong device");
    require(accounting.allocation_count == 4 && accounting.canonical_allocation_count == 3 &&
                accounting.alias_count == 1 && accounting.roles.size() == 2,
            "arena accounting counts are wrong");
    require(accounting.high_water_bytes == plan.total_high_water_bytes() &&
                accounting.allocated_bytes == plan.total_reserved_bytes(),
            "arena byte accounting is wrong");
    require(meep::nvidia::current_memory_accounting().device_bytes_current ==
                before.device_bytes_current + accounting.allocated_bytes,
            "runtime accounting did not include role arenas");

    const device_allocation owner = arenas.resolve(1);
    const device_allocation alias = arenas.resolve(3);
    const device_allocation neighbor = arenas.resolve(2);
    const device_allocation material = arenas.resolve(4);
    require(owner.address == alias.address && alias.canonical_id == owner.id,
            "device alias identity was not preserved");
    require(reinterpret_cast<uintptr_t>(owner.address) % 512 == 0,
            "canonical address does not meet promoted alignment");
    require(reinterpret_cast<uintptr_t>(neighbor.address) % 64 == 0,
            "neighbor address does not meet alignment");
    require(reinterpret_cast<uintptr_t>(material.address) % 128 == 0,
            "material address does not meet alignment");

    pinned_buffer output(owner.bytes);
    arenas.fill_async(1, 0, 0x5a, owner.bytes, transfer);
    arenas.copy_to_host_async(output.data(), 3, 0, alias.bytes, transfer);
    transfer.synchronize();
    const unsigned char *filled = static_cast<const unsigned char *>(output.data());
    for (size_t i = 0; i < output.size(); ++i)
      require(filled[i] == 0x5a, "fill/alias round trip mismatch");

    pinned_buffer input(material.bytes);
    pinned_buffer copied(material.bytes);
    unsigned char *input_bytes = static_cast<unsigned char *>(input.data());
    for (size_t i = 0; i < input.size(); ++i)
      input_bytes[i] = static_cast<unsigned char>((i * 17 + device) & 0xff);
    arenas.copy_from_host_async(4, 0, input.data(), input.size(), transfer);
    arenas.copy_to_host_async(copied.data(), 4, 0, copied.size(), transfer);
    transfer.synchronize();
    require(std::memcmp(input.data(), copied.data(), copied.size()) == 0,
            "host/device arena round trip mismatch");

    std::vector<allocation_request> replacement_requests;
    replacement_requests.push_back(
        allocation_request(104, arena_role::material, material.bytes, 128, no_allocation,
                           arena_element_type::float64, arena_storage_precision::f64));
    replacement_requests.push_back(allocation_request(105, arena_role::material, material.bytes,
                                                      128, no_allocation, arena_element_type::int32,
                                                      arena_storage_precision::f32));
    replacement_requests.push_back(allocation_request(200, arena_role::scratch, 64, 64));
    device_arenas replacement(arena_plan(replacement_requests), device);
    replacement.copy_from_device_async(104, 0, arenas, 4, 0, material.bytes, transfer);
    std::memset(copied.data(), 0, copied.size());
    replacement.copy_to_host_async(copied.data(), 104, 0, copied.size(), transfer);
    transfer.synchronize();
    require(std::memcmp(input.data(), copied.data(), copied.size()) == 0,
            "compatible reprepare device-to-device copy mismatch");
    require_throws<std::out_of_range>(
        [&]() {
          replacement.copy_from_device_async(104, material.bytes, arenas, 4, 0, 1, transfer);
        },
        "device-to-device destination overrun was accepted");
    require_throws<std::out_of_range>(
        [&]() {
          replacement.copy_from_device_async(104, 0, arenas, 4, material.bytes, 1, transfer);
        },
        "device-to-device source overrun was accepted");
    require_throws<std::invalid_argument>(
        [&]() {
          replacement.copy_from_device_async(105, 0, arenas, 4, 0, material.bytes, transfer);
        },
        "incompatible device-to-device storage types were accepted");

    require_throws<std::out_of_range>([&]() { arenas.fill_async(2, 1024, 0, 1, transfer); },
                                      "logical allocation overrun was accepted");
    require_throws<std::out_of_range>([&]() { (void)arenas.resolve(999); },
                                      "unknown device allocation ID was accepted");
  }
  const memory_accounting after = meep::nvidia::current_memory_accounting();
  require(after.device_bytes_current == before.device_bytes_current,
          "role arenas leaked device memory");
  require(after.pinned_bytes_current == before.pinned_bytes_current,
          "arena test leaked pinned memory");

  test_preflight_failure(device);

  {
    const memory_accounting allocation_before = meep::nvidia::current_memory_accounting();
    fail_next(failure_point::device_allocate);
    require_throws<meep::nvidia::runtime_error>([&]() { (void)device_arenas(plan, device); },
                                                "arena allocation failure was not reported");
    clear_failure();
    require(meep::nvidia::current_memory_accounting().device_bytes_current ==
                allocation_before.device_bytes_current,
            "failed arena allocation changed runtime accounting");
  }

  {
    const size_t accounting_before = meep::nvidia::current_memory_accounting().device_bytes_current;
    {
      device_scope other(other_device);
      device_arenas destination(plan, device);
      device_arenas source(plan, device);
      const void *destination_address = destination.resolve(1).address;
      const void *source_address = source.resolve(1).address;
      const size_t move_accounting = meep::nvidia::current_memory_accounting().device_bytes_current;
      fail_next(failure_point::device_free);
      destination = std::move(source);
      stream current_device_probe;
      require(current_device_probe.device() == other_device,
              "arena move assignment changed the current device");
      require(destination.resolve(1).address == source_address,
              "arena move assignment did not transfer source ownership");
      require(source.resolve(1).address == destination_address,
              "arena move assignment discarded destination ownership");
      require(meep::nvidia::current_memory_accounting().device_bytes_current == move_accounting,
              "arena move assignment changed device accounting");

      /* A swap-based move performs no device teardown. Prove the injected free is
         still pending, then retry successfully so the test itself does not leak. */
      device_buffer failure_probe(64, device);
      require_throws<meep::nvidia::runtime_error>([&]() { failure_probe.reset(); },
                                                  "arena move consumed the pending free failure");
      require(failure_probe.opaque_handle() != NULL,
              "injected post-move free failure lost probe ownership");
      failure_probe.reset();
      clear_failure();
    }
    require(meep::nvidia::current_memory_accounting().device_bytes_current == accounting_before,
            "move-assigned arenas leaked device memory");
  }
  std::cout << "arena device " << device << ": PASS\n";
}

} // namespace

int main() {
  try {
    test_cpu_planning();
    const std::vector<device_properties> devices = enumerate_devices();
#if defined(MEEP_HIP_PORTABILITY)
    require(!devices.empty(), "arena smoke test requires one visible AMD device");
    for (size_t i = 0; i < devices.size(); ++i) {
      std::cout << "device " << devices[i].id << ": " << devices[i].name << "\n";
      const int other_device =
          devices.size() > 1 ? devices[(i + 1) % devices.size()].id : devices[i].id;
      test_device(devices[i].id, other_device);
    }
#else
    require(devices.size() >= 2, "arena smoke test requires both GB200 devices");
    size_t gb200_count = 0;
    for (size_t i = 0; i < devices.size(); ++i) {
      std::cout << "device " << devices[i].id << ": " << devices[i].name << "\n";
      if (devices[i].name.find("GB200") != std::string::npos) {
        const int other_device = devices[i].id == devices[0].id ? devices[1].id : devices[0].id;
        test_device(devices[i].id, other_device);
        ++gb200_count;
      }
    }
    require(gb200_count >= 2, "fewer than two visible GB200 devices were tested");
#endif
    std::cout << "PASS\n";
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
