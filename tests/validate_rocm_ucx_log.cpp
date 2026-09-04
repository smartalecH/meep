/* Validate that each attested HIP device independently selected ROCm IPC in a
   UCX protocol trace.  This is deliberately a dependency-free native helper
   so the MPI acceptance gate does not depend on the optional Python build. */

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct topology_identity {
  int rank;
  int device;
  std::string uuid;
  std::string bdf;
};

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

std::string normalize_bdf(std::string value) {
  value = lowercase(value);
  if (value.size() == 7) value = "0000:" + value;
  return value;
}

std::vector<std::string> split_expected_bdfs(const std::string &configured) {
  std::vector<std::string> result;
  std::stringstream values(configured);
  std::string value;
  while (std::getline(values, value, ','))
    if (!value.empty()) result.push_back(normalize_bdf(value));
  if (result.empty()) throw std::runtime_error("expected BDF list is empty");
  return result;
}

bool validate_trace(const std::string &text, const std::vector<std::string> &expected_bdfs,
                    std::string &why) {
  const std::regex topology(
      "HIP_TOPOLOGY rank=([0-9]+) role=owner logical_device=([0-9]+) uuid=([^ ]+) "
      "bdf=([^ ]+)");
  const std::regex source("from rocm/GPU([0-9]+)");
  const std::regex ipc("zero-copy.*rocm_ipc/rocm_ipc");
  std::map<int, topology_identity> identities;
  std::map<std::string, int> current_device_by_ucx_context;
  std::set<int> ipc_devices;
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    std::smatch match;
    if (std::regex_search(line, match, topology)) {
      topology_identity identity = {std::stoi(match[1].str()), std::stoi(match[2].str()),
                                    match[3].str(), normalize_bdf(match[4].str())};
      if (!identities.insert(std::make_pair(identity.device, identity)).second) {
        why = "duplicate HIP topology stanza for logical device " + std::to_string(identity.device);
        return false;
      }
    }

    const size_t context_marker = line.find("] [");
    if (context_marker == std::string::npos) continue;
    const size_t context_begin = context_marker + 3;
    const size_t context_end = line.find(']', context_begin);
    if (context_end == std::string::npos) continue;
    const std::string context = line.substr(context_begin, context_end - context_begin);
    std::smatch source_match;
    if (std::regex_search(line, source_match, source))
      current_device_by_ucx_context[context] = std::stoi(source_match[1].str());
    else if (std::regex_search(line, ipc)) {
      const std::map<std::string, int>::const_iterator current =
          current_device_by_ucx_context.find(context);
      if (current != current_device_by_ucx_context.end()) ipc_devices.insert(current->second);
    }
  }

  if (identities.size() != expected_bdfs.size()) {
    why = "UCX trace does not contain one HIP topology stanza per expected BDF";
    return false;
  }
  std::set<std::string> uuids, bdfs;
  for (size_t device = 0; device < expected_bdfs.size(); ++device) {
    const std::map<int, topology_identity>::const_iterator found =
        identities.find(static_cast<int>(device));
    if (found == identities.end() || found->second.rank != static_cast<int>(device)) {
      why = "HIP topology does not map each rank to its matching logical device";
      return false;
    }
    if (found->second.uuid.empty() || found->second.uuid == "-") {
      why = "HIP topology is missing a runtime UUID";
      return false;
    }
    if (found->second.bdf != expected_bdfs[device]) {
      why =
          "HIP topology does not match the expected physical BDF for GPU" + std::to_string(device);
      return false;
    }
    uuids.insert(found->second.uuid);
    bdfs.insert(found->second.bdf);
    if (!ipc_devices.count(static_cast<int>(device))) {
      why = "UCX did not select zero-copy rocm_ipc/rocm_ipc in the stanza for GPU" +
            std::to_string(device) + " at " + found->second.bdf;
      return false;
    }
  }
  if (uuids.size() != expected_bdfs.size() || bdfs.size() != expected_bdfs.size()) {
    why = "HIP topology contains duplicate runtime UUID or physical BDF identities";
    return false;
  }
  return true;
}

std::string valid_fixture() {
  return "HIP_TOPOLOGY rank=0 role=owner logical_device=0 uuid=uuid0 bdf=0000:08:00.0 "
         "gpu_numa=0 cpu_numa=0 cpu_affinity=0\n"
         "HIP_TOPOLOGY rank=1 role=owner logical_device=1 uuid=uuid1 bdf=0000:88:00.0 "
         "gpu_numa=1 cpu_numa=1 cpu_affinity=1\n"
         "[1.0] [host:10:0] from rocm/GPU0\n"
         "[1.1] [host:11:0] from rocm/GPU1\n"
         "[1.2] [host:10:0] zero-copy read from remote | rocm_ipc/rocm_ipc |\n"
         "[1.3] [host:11:0] zero-copy read from remote | rocm_ipc/rocm_ipc |\n";
}

bool run_self_test() {
  const std::vector<std::string> expected = {"0000:08:00.0", "0000:88:00.0"};
  std::string why;
  if (!validate_trace(valid_fixture(), expected, why)) {
    std::cerr << "valid fixture rejected: " << why << "\n";
    return false;
  }

  std::string one_device_ipc = valid_fixture();
  const std::string gpu1_ipc =
      "[1.3] [host:11:0] zero-copy read from remote | rocm_ipc/rocm_ipc |\n";
  one_device_ipc.erase(one_device_ipc.find(gpu1_ipc), gpu1_ipc.size());
  if (validate_trace(one_device_ipc, expected, why) || why.find("GPU1") == std::string::npos) {
    std::cerr << "one-device IPC fixture was not rejected per device\n";
    return false;
  }

  std::string wrong_bdf = valid_fixture();
  wrong_bdf.replace(wrong_bdf.find("0000:88:00.0"), 12, "0000:f9:00.0");
  if (validate_trace(wrong_bdf, expected, why) || why.find("physical BDF") == std::string::npos) {
    std::cerr << "wrong-BDF fixture was not rejected\n";
    return false;
  }

  std::string duplicate_uuid = valid_fixture();
  duplicate_uuid.replace(duplicate_uuid.find("uuid1"), 5, "uuid0");
  if (validate_trace(duplicate_uuid, expected, why) || why.find("duplicate") == std::string::npos) {
    std::cerr << "duplicate-UUID fixture was not rejected\n";
    return false;
  }
  std::cout << "validate_rocm_ucx_log self-test: PASS\n";
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-test") return run_self_test() ? 0 : 1;
  if (argc != 3) {
    std::cerr << "usage: validate_rocm_ucx_log LOG EXPECTED_BDF0,EXPECTED_BDF1,...\n";
    return 2;
  }
  try {
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error(std::string("cannot read UCX trace: ") + argv[1]);
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::vector<std::string> expected = split_expected_bdfs(argv[2]);
    std::string why;
    if (!validate_trace(contents.str(), expected, why)) {
      std::cerr << why << "\n";
      return 1;
    }
    std::cout << "UCX ROCm IPC topology correlation: PASS";
    for (size_t device = 0; device < expected.size(); ++device)
      std::cout << " GPU" << device << "=" << expected[device];
    std::cout << "\n";
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}
