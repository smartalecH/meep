/* Standalone two-rank GPU/MPI transport smoke test. It intentionally has no
   dependency on libmeep and does not select a production MPI transport policy. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "backend/nvidia/runtime.hpp"

#include <mpi.h>

#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#define MEEP_TEST_HAVE_MPI_EXT 1
#endif
#endif
#ifndef MEEP_TEST_HAVE_MPI_EXT
#define MEEP_TEST_HAVE_MPI_EXT 0
#endif

#include <stdint.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using meep::nvidia::copy_device_to_host_async;
using meep::nvidia::copy_host_to_device_async;
using meep::nvidia::current_memory_accounting;
using meep::nvidia::device_buffer;
using meep::nvidia::device_properties;
using meep::nvidia::device_scope;
using meep::nvidia::device_selection;
using meep::nvidia::event;
using meep::nvidia::memory_accounting;
using meep::nvidia::pinned_buffer;
using meep::nvidia::properties_for_device;
using meep::nvidia::select_device;
using meep::nvidia::stream;
using meep::nvidia::testing::fail_next;
using meep::nvidia::testing::failure_point;

namespace {

const size_t payload_bytes = 4u * 1024u * 1024u;
const int staged_tag = 4101;
const int direct_tag = 4102;
const size_t error_message_bytes = 512;

enum class transport_mode { staged, automatic, direct };

struct options {
  transport_mode mode;
  int inject_preflight_failure_rank;
  bool forbid_device_mpi;
  bool require_device_mpi;
};

void mpi_check(int result, const char *operation) {
  if (result == MPI_SUCCESS) return;
  char detail[MPI_MAX_ERROR_STRING];
  int length = 0;
  MPI_Error_string(result, detail, &length);
  throw std::runtime_error(std::string(operation) +
                           " failed: " + std::string(detail, static_cast<size_t>(length)));
}

int parse_nonnegative_int(const std::string &text, const char *option) {
  if (text.empty()) throw std::invalid_argument(std::string(option) + " is empty");
  errno = 0;
  char *end = NULL;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (errno || end == text.c_str() || *end || value < 0 || value > std::numeric_limits<int>::max())
    throw std::invalid_argument(std::string("invalid ") + option + ": " + text);
  return static_cast<int>(value);
}

options parse_options(int argc, char **argv) {
  options result = {transport_mode::automatic, -1, false, false};
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    const std::string transport_prefix("--transport=");
    const std::string failure_prefix("--inject-preflight-failure-rank=");
    if (argument.compare(0, transport_prefix.size(), transport_prefix) == 0) {
      const std::string value = argument.substr(transport_prefix.size());
      if (value == "staged")
        result.mode = transport_mode::staged;
      else if (value == "auto")
        result.mode = transport_mode::automatic;
      else if (value == "direct")
        result.mode = transport_mode::direct;
      else
        throw std::invalid_argument("unknown transport mode: " + value);
    }
    else if (argument.compare(0, failure_prefix.size(), failure_prefix) == 0) {
      result.inject_preflight_failure_rank = parse_nonnegative_int(
          argument.substr(failure_prefix.size()), "--inject-preflight-failure-rank");
    }
    else if (argument == "--forbid-device-mpi") { result.forbid_device_mpi = true; }
    else if (argument == "--require-device-mpi") { result.require_device_mpi = true; }
    else {
      throw std::invalid_argument(
          "usage: nvidia_mpi_runtime_smoke [--transport=staged|auto|direct] "
          "[--inject-preflight-failure-rank=N] "
          "[--forbid-device-mpi|--require-device-mpi]");
    }
  }
  if (result.forbid_device_mpi && result.require_device_mpi)
    throw std::invalid_argument("device-pointer MPI cannot be both required and forbidden");
  return result;
}

const char *mode_name(transport_mode mode) {
  switch (mode) {
    case transport_mode::staged: return "staged";
    case transport_mode::automatic: return "auto";
    case transport_mode::direct: return "direct";
  }
  return "unknown";
}

uint8_t payload_value(int source_rank, size_t index) {
  return static_cast<uint8_t>(
      (index * 131u + (index >> 8) * 17u + static_cast<size_t>(source_rank) * 67u + 29u) & 0xffu);
}

void initialize_payload(void *address, size_t bytes, int source_rank) {
  uint8_t *values = static_cast<uint8_t *>(address);
  for (size_t i = 0; i < bytes; ++i)
    values[i] = payload_value(source_rank, i);
}

void validate_payload(const void *address, size_t bytes, int source_rank, const char *transport) {
  const uint8_t *values = static_cast<const uint8_t *>(address);
  for (size_t i = 0; i < bytes; ++i) {
    const uint8_t expected = payload_value(source_rank, i);
    if (values[i] != expected) {
      std::ostringstream message;
      message << transport << " payload mismatch at byte " << i << ": expected "
              << static_cast<unsigned int>(expected) << ", received "
              << static_cast<unsigned int>(values[i]);
      throw std::runtime_error(message.str());
    }
  }
}

uint64_t payload_digest(const void *address, size_t bytes) {
  const uint8_t *values = static_cast<const uint8_t *>(address);
  uint64_t digest = UINT64_C(1469598103934665603);
  for (size_t i = 0; i < bytes; ++i) {
    digest ^= values[i];
    digest *= UINT64_C(1099511628211);
  }
  return digest;
}

bool collectively_report(bool local_success, const std::string &local_error, MPI_Comm communicator,
                         int rank, const char *phase) {
  int local = local_success ? 1 : 0;
  int global = 0;
  mpi_check(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, communicator),
            "MPI_Allreduce(test status)");
  if (global) return true;

  std::vector<char> local_message(error_message_bytes, 0);
  if (!local_success) {
    const size_t copy_bytes = std::min(local_error.size(), error_message_bytes - 1);
    std::memcpy(local_message.data(), local_error.data(), copy_bytes);
  }
  int size = 0;
  mpi_check(MPI_Comm_size(communicator, &size), "MPI_Comm_size(error report)");
  std::vector<char> messages(static_cast<size_t>(size) * error_message_bytes, 0);
  mpi_check(MPI_Allgather(local_message.data(), static_cast<int>(error_message_bytes), MPI_CHAR,
                          messages.data(), static_cast<int>(error_message_bytes), MPI_CHAR,
                          communicator),
            "MPI_Allgather(error report)");
  if (rank == 0) {
    std::cerr << "FAIL " << phase << "\n";
    for (int source = 0; source < size; ++source) {
      const char *message = messages.data() + static_cast<size_t>(source) * error_message_bytes;
      if (*message) std::cerr << "  rank " << source << ": " << message << "\n";
    }
  }
  return false;
}

const char *gpu_aware_mpi_query_name() {
#if defined(HAVE_MPIX_QUERY_ROCM_SUPPORT) && HAVE_MPIX_QUERY_ROCM_SUPPORT
  return "MPIX_Query_rocm_support";
#elif defined(HAVE_MPIX_QUERY_CUDA_SUPPORT) && HAVE_MPIX_QUERY_CUDA_SUPPORT
  return "MPIX_Query_cuda_support";
#else
  return "a GPU-aware MPI provider query";
#endif
}

bool gpu_aware_mpi_enabled() {
#if defined(HAVE_MPIX_QUERY_ROCM_SUPPORT) && HAVE_MPIX_QUERY_ROCM_SUPPORT
#if !MEEP_TEST_HAVE_MPI_EXT
  extern int MPIX_Query_rocm_support(void);
#endif
  return MPIX_Query_rocm_support() > 0;
#elif defined(HAVE_MPIX_QUERY_CUDA_SUPPORT) && HAVE_MPIX_QUERY_CUDA_SUPPORT
#if !MEEP_TEST_HAVE_MPI_EXT
  extern int MPIX_Query_cuda_support(void);
#endif
  return MPIX_Query_cuda_support() > 0;
#else
  return false;
#endif
}

[[noreturn]] void abort_transport(MPI_Comm communicator, int error, const char *operation) {
  char detail[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  if (MPI_Error_string(error, detail, &length) != MPI_SUCCESS) { length = 0; }
  std::cerr << "fatal MPI transport error in " << operation << ": code " << error;
  if (length) std::cerr << " (" << std::string(detail, static_cast<size_t>(length)) << ")";
  std::cerr << "\n";
  MPI_Abort(communicator, error == MPI_SUCCESS ? 1 : error);
  std::abort();
}

void transport_check(int result, MPI_Comm communicator, const char *operation) {
  if (result != MPI_SUCCESS) abort_transport(communicator, result, operation);
}

void wait_for_exchange(MPI_Request requests[2], MPI_Comm communicator, const char *transport) {
  MPI_Status statuses[2];
  const int result = MPI_Waitall(2, requests, statuses);
  if (result == MPI_SUCCESS) return;

  /* MPI_ERR_IN_STATUS may leave MPI_ERR_PENDING requests live. Do not unwind
     their backing buffers. A transport failure is fatal for this standalone
     two-rank test, so abort the communicator while every buffer is still alive. */
  std::ostringstream operation;
  operation << "MPI_Waitall(" << transport << ")";
  abort_transport(communicator, result, operation.str().c_str());
}

bool accounting_matches(const memory_accounting &before, const memory_accounting &after,
                        std::string *error) {
  if (before.device_bytes_current == after.device_bytes_current &&
      before.pinned_bytes_current == after.pinned_bytes_current)
    return true;
  std::ostringstream message;
  message << "allocation accounting did not return to baseline: device "
          << before.device_bytes_current << " -> " << after.device_bytes_current << ", pinned "
          << before.pinned_bytes_current << " -> " << after.pinned_bytes_current;
  *error = message.str();
  return false;
}

struct staged_exchange_state {
  explicit staged_exchange_state(int device)
      : selected(device), initial(payload_bytes), send_stage(payload_bytes),
        receive_stage(payload_bytes), verification(payload_bytes),
        send_device(payload_bytes, device), receive_device(payload_bytes, device), transfer(),
        send_ready(), receive_ready() {}

  device_scope selected;
  pinned_buffer initial;
  pinned_buffer send_stage;
  pinned_buffer receive_stage;
  pinned_buffer verification;
  device_buffer send_device;
  device_buffer receive_device;
  stream transfer;
  event send_ready;
  event receive_ready;
};

bool exchange_staged(int device, int rank, int peer, MPI_Comm communicator,
                     bool inject_preflight_failure, uint64_t *received_digest) {
  const memory_accounting before = current_memory_accounting();
  std::unique_ptr<staged_exchange_state> state;
  bool preflight_ok = true;
  std::string preflight_error;
  try {
    if (inject_preflight_failure) fail_next(failure_point::device_allocate);
    state.reset(new staged_exchange_state(device));

    initialize_payload(state->initial.data(), payload_bytes, rank);
    std::memset(state->send_stage.data(), 0, payload_bytes);
    std::memset(state->receive_stage.data(), 0, payload_bytes);
    std::memset(state->verification.data(), 0, payload_bytes);

    /* Complete all fallible GPU producer work before either rank posts an MPI
       request. The collective gate below keeps peers in the same state. */
    copy_host_to_device_async(state->send_device, 0, state->initial.data(), payload_bytes,
                              state->transfer);
    copy_device_to_host_async(state->send_stage.data(), state->send_device, 0, payload_bytes,
                              state->transfer);
    state->send_ready.record(state->transfer);
    state->send_ready.synchronize();
  }
  catch (const std::exception &error) {
    preflight_ok = false;
    preflight_error = error.what();
  }

  const bool all_preflight_ok = collectively_report(preflight_ok, preflight_error, communicator,
                                                    rank, "staged GPU preflight");
  if (!all_preflight_ok) {
    state.reset();
    std::string accounting_error;
    const bool accounting_ok =
        accounting_matches(before, current_memory_accounting(), &accounting_error);
    collectively_report(accounting_ok, accounting_error, communicator, rank,
                        "staged preflight cleanup accounting");
    return false;
  }

  MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  transport_check(MPI_Irecv(state->receive_stage.data(), static_cast<int>(payload_bytes), MPI_BYTE,
                            peer, staged_tag, communicator, &requests[0]),
                  communicator, "MPI_Irecv(staged)");
  transport_check(MPI_Isend(state->send_stage.data(), static_cast<int>(payload_bytes), MPI_BYTE,
                            peer, staged_tag, communicator, &requests[1]),
                  communicator, "MPI_Isend(staged)");
  wait_for_exchange(requests, communicator, "staged");

  /* Complete the mandatory D2H -> MPI -> H2D path, then copy back solely for
     byte-exact validation. */
  bool validation_ok = true;
  std::string validation_error;
  try {
    copy_host_to_device_async(state->receive_device, 0, state->receive_stage.data(), payload_bytes,
                              state->transfer);
    copy_device_to_host_async(state->verification.data(), state->receive_device, 0, payload_bytes,
                              state->transfer);
    state->receive_ready.record(state->transfer);
    state->receive_ready.synchronize();
    validate_payload(state->verification.data(), payload_bytes, peer, "staged");
    if (received_digest)
      *received_digest = payload_digest(state->verification.data(), payload_bytes);
  }
  catch (const std::exception &error) {
    validation_ok = false;
    validation_error = error.what();
  }
  const bool all_validation_ok =
      collectively_report(validation_ok, validation_error, communicator, rank, "staged validation");
  state.reset();
  std::string accounting_error;
  const bool accounting_ok =
      accounting_matches(before, current_memory_accounting(), &accounting_error);
  const bool all_accounting_ok = collectively_report(accounting_ok, accounting_error, communicator,
                                                     rank, "staged allocation accounting");
  return all_validation_ok && all_accounting_ok;
}

struct direct_exchange_state {
  explicit direct_exchange_state(int device)
      : selected(device), initial(payload_bytes), verification(payload_bytes),
        send_device(payload_bytes, device), receive_device(payload_bytes, device), transfer(),
        send_ready(), receive_ready() {}

  device_scope selected;
  pinned_buffer initial;
  pinned_buffer verification;
  device_buffer send_device;
  device_buffer receive_device;
  stream transfer;
  event send_ready;
  event receive_ready;
};

bool exchange_direct(int device, int rank, int peer, MPI_Comm communicator,
                     uint64_t *received_digest) {
  const memory_accounting before = current_memory_accounting();
  std::unique_ptr<direct_exchange_state> state;
  bool preflight_ok = true;
  std::string preflight_error;
  try {
    state.reset(new direct_exchange_state(device));
    initialize_payload(state->initial.data(), payload_bytes, rank);
    std::memset(state->verification.data(), 0, payload_bytes);

    /* Device-pointer MPI may read send_device as soon as MPI_Isend returns, so
       collectively complete its GPU producer before posting either request. */
    copy_host_to_device_async(state->send_device, 0, state->initial.data(), payload_bytes,
                              state->transfer);
    state->send_ready.record(state->transfer);
    state->send_ready.synchronize();
  }
  catch (const std::exception &error) {
    preflight_ok = false;
    preflight_error = error.what();
  }

  const bool all_preflight_ok = collectively_report(preflight_ok, preflight_error, communicator,
                                                    rank, "direct GPU preflight");
  if (!all_preflight_ok) {
    state.reset();
    std::string accounting_error;
    const bool accounting_ok =
        accounting_matches(before, current_memory_accounting(), &accounting_error);
    collectively_report(accounting_ok, accounting_error, communicator, rank,
                        "direct preflight cleanup accounting");
    return false;
  }

  MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  transport_check(MPI_Irecv(state->receive_device.opaque_handle(), static_cast<int>(payload_bytes),
                            MPI_BYTE, peer, direct_tag, communicator, &requests[0]),
                  communicator, "MPI_Irecv(device pointer)");
  transport_check(MPI_Isend(state->send_device.opaque_handle(), static_cast<int>(payload_bytes),
                            MPI_BYTE, peer, direct_tag, communicator, &requests[1]),
                  communicator, "MPI_Isend(device pointer)");
  wait_for_exchange(requests, communicator, "direct");

  bool validation_ok = true;
  std::string validation_error;
  try {
    copy_device_to_host_async(state->verification.data(), state->receive_device, 0, payload_bytes,
                              state->transfer);
    state->receive_ready.record(state->transfer);
    state->receive_ready.synchronize();
    validate_payload(state->verification.data(), payload_bytes, peer, "direct");
    if (received_digest)
      *received_digest = payload_digest(state->verification.data(), payload_bytes);
  }
  catch (const std::exception &error) {
    validation_ok = false;
    validation_error = error.what();
  }
  const bool all_validation_ok =
      collectively_report(validation_ok, validation_error, communicator, rank, "direct validation");
  state.reset();
  std::string accounting_error;
  const bool accounting_ok =
      accounting_matches(before, current_memory_accounting(), &accounting_error);
  const bool all_accounting_ok = collectively_report(accounting_ok, accounting_error, communicator,
                                                     rank, "direct allocation accounting");
  return all_validation_ok && all_accounting_ok;
}

void validate_device_assignment(const std::string &uuid, int local_rank, int local_size,
                                MPI_Comm local_comm) {
  const size_t uuid_bytes = 64;
  std::vector<char> local_uuid(uuid_bytes, 0);
  std::memcpy(local_uuid.data(), uuid.data(), uuid.size());
  std::vector<char> gathered(static_cast<size_t>(local_size) * uuid_bytes, 0);
  mpi_check(MPI_Allgather(local_uuid.data(), static_cast<int>(uuid_bytes), MPI_CHAR,
                          gathered.data(), static_cast<int>(uuid_bytes), MPI_CHAR, local_comm),
            "MPI_Allgather(device UUIDs)");
  for (int other = 0; other < local_size; ++other) {
    if (other == local_rank) continue;
    const char *other_uuid = gathered.data() + static_cast<size_t>(other) * uuid_bytes;
    if (uuid == other_uuid)
      throw std::runtime_error("node-local ranks selected the same GPU device UUID");
  }
}

} // namespace

int main(int argc, char **argv) {
  int provided = MPI_THREAD_SINGLE;
  const int init_result = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (init_result != MPI_SUCCESS) {
    std::cerr << "FAIL MPI_Init_thread returned " << init_result << "\n";
    return 1;
  }

  int exit_code = 0;
  MPI_Comm local_comm = MPI_COMM_NULL;
  int rank = -1;
  int size = 0;
  try {
    mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
              "MPI_Comm_set_errhandler");
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");
    mpi_check(
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm),
        "MPI_Comm_split_type");
    mpi_check(MPI_Comm_set_errhandler(local_comm, MPI_ERRORS_RETURN),
              "MPI_Comm_set_errhandler(local)");

    std::string setup_error;
    transport_mode mode = transport_mode::automatic;
    int inject_preflight_failure_rank = -1;
    bool forbid_device_mpi = false;
    bool require_device_mpi = false;
    int local_rank = -1;
    int local_size = 0;
    device_selection selection = {};
    device_properties properties = {};
    std::string device_description;
    bool setup_ok = true;
    try {
      const options parsed = parse_options(argc, argv);
      mode = parsed.mode;
      inject_preflight_failure_rank = parsed.inject_preflight_failure_rank;
      forbid_device_mpi = parsed.forbid_device_mpi;
      require_device_mpi = parsed.require_device_mpi;
      mpi_check(MPI_Comm_rank(local_comm, &local_rank), "MPI_Comm_rank(local)");
      mpi_check(MPI_Comm_size(local_comm, &local_size), "MPI_Comm_size(local)");
      if (provided < MPI_THREAD_FUNNELED)
        throw std::runtime_error("MPI did not provide MPI_THREAD_FUNNELED");
      if (size != 2 || local_size != 2)
        throw std::runtime_error("test requires exactly two ranks on one shared-memory node");
      if (inject_preflight_failure_rank >= size)
        throw std::invalid_argument("injected failure rank is outside MPI_COMM_WORLD");
      if (payload_bytes > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("payload exceeds MPI int count");
      selection = select_device(local_rank, local_rank, local_size, false);
      if (!selection.collective_collision_check_required)
        throw std::runtime_error("explicit GPU selection did not require collective validation");
      properties = properties_for_device(selection.device);
      if (properties.uuid.size() >= 64)
        throw std::runtime_error("GPU UUID exceeds assignment-validation field");
      device_description = properties.name + " uuid=" + properties.uuid;
    }
    catch (const std::exception &error) {
      setup_ok = false;
      setup_error = error.what();
    }

    if (!collectively_report(setup_ok, setup_error, MPI_COMM_WORLD, rank, "local setup")) {
      exit_code = 1;
    }
    else {
      bool assignment_ok = true;
      std::string assignment_error;
      try {
        validate_device_assignment(properties.uuid, local_rank, local_size, local_comm);
      }
      catch (const std::exception &error) {
        assignment_ok = false;
        assignment_error = error.what();
      }
      if (!collectively_report(assignment_ok, assignment_error, MPI_COMM_WORLD, rank,
                               "collective device assignment")) {
        exit_code = 1;
      }
    }

    if (!exit_code) {
      if (rank == 0)
        std::cout << "MPI thread level=" << provided << ", transport=" << mode_name(mode)
                  << ", bytes=" << payload_bytes << "\n";
      std::cout << "rank " << rank << " local-rank " << local_rank << " -> device "
                << selection.device << " " << device_description << "\n";

      const int peer = 1 - rank;
      uint64_t staged_digest = 0;
      if (!exchange_staged(selection.device, rank, peer, MPI_COMM_WORLD,
                           rank == inject_preflight_failure_rank, &staged_digest)) {
        exit_code = 1;
      }
      else if (rank == 0) { std::cout << "staged D2H->MPI->H2D bidirectional exchange: PASS\n"; }

      int all_gpu_aware = 0;
      if (!exit_code && mode != transport_mode::staged) {
        const int local_gpu_aware = gpu_aware_mpi_enabled() ? 1 : 0;
        mpi_check(
            MPI_Allreduce(&local_gpu_aware, &all_gpu_aware, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
            "MPI_Allreduce(GPU-aware support)");
      }

      if (!exit_code && mode != transport_mode::staged && !all_gpu_aware && require_device_mpi) {
        const std::string unavailable = std::string("required device-pointer MPI needs positive ") +
                                        gpu_aware_mpi_query_name() + " on every rank";
        collectively_report(false, unavailable, MPI_COMM_WORLD, rank, "direct capability");
        exit_code = 1;
      }
      else if (!exit_code && mode != transport_mode::staged && all_gpu_aware && forbid_device_mpi) {
        const std::string forbidden =
            "device-pointer MPI is forbidden by this staged-only validation";
        collectively_report(false, forbidden, MPI_COMM_WORLD, rank, "staged-only capability");
        exit_code = 1;
      }
      else if (!exit_code && mode == transport_mode::direct && !all_gpu_aware) {
        const std::string unsupported = std::string("forced direct transport requires positive ") +
                                        gpu_aware_mpi_query_name() + " on every rank";
        collectively_report(false, unsupported, MPI_COMM_WORLD, rank, "direct capability");
        exit_code = 1;
      }
      else if (!exit_code && mode != transport_mode::staged && all_gpu_aware) {
        uint64_t direct_digest = 0;
        if (!exchange_direct(selection.device, rank, peer, MPI_COMM_WORLD, &direct_digest)) {
          exit_code = 1;
        }
        else {
          const bool same_output = direct_digest == staged_digest;
          if (!collectively_report(
                  same_output, same_output ? std::string() : "staged/direct receive digests differ",
                  MPI_COMM_WORLD, rank, "staged/direct equivalence"))
            exit_code = 1;
          else if (rank == 0)
            std::cout << "direct device-pointer bidirectional exchange and staged/direct "
                         "digest equivalence: PASS\n";
        }
      }
      else if (!exit_code && mode == transport_mode::automatic && rank == 0) {
        std::cout << "direct device-pointer exchange: SKIP (" << gpu_aware_mpi_query_name()
                  << "=0)\n";
      }

      int local_exit = exit_code;
      int global_exit = 0;
      mpi_check(MPI_Allreduce(&local_exit, &global_exit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(exit code)");
      exit_code = global_exit;
      if (!exit_code && rank == 0) std::cout << "PASS\n";
    }
  }
  catch (const std::exception &error) {
    std::cerr << "rank " << rank << " fatal error: " << error.what() << "\n";
    exit_code = 1;
  }

  if (local_comm != MPI_COMM_NULL) {
    const int free_result = MPI_Comm_free(&local_comm);
    if (free_result != MPI_SUCCESS) {
      std::cerr << "rank " << rank << " MPI_Comm_free failed with code " << free_result << "\n";
      exit_code = 1;
    }
  }
  const int finalize_result = MPI_Finalize();
  if (finalize_result != MPI_SUCCESS) {
    std::cerr << "rank " << rank << " MPI_Finalize failed with code " << finalize_result << "\n";
    exit_code = 1;
  }
  return exit_code;
}
