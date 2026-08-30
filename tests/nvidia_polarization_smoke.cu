/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include "backend/nvidia/nvidia_polarization.hpp"
#include "backend/random_state.hpp"
#include "meep/meep-config.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using namespace meep::nvidia;
using meep::counter_random_stream_tag;

#if MEEP_SINGLE
typedef float test_realnum;
#else
typedef double test_realnum;
#endif

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

static uint64_t double_bits(double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct random_result {
  std::vector<counter_random_words> words;
  std::vector<double> uniforms;
  std::vector<double> normals;
};

static random_result sample_device(int device, const std::vector<counter_random_input> &inputs,
                                   unsigned int threads, size_t padding = 0) {
  device_scope selected(device);
  stream execution;
  const size_t input_bytes = inputs.size() * sizeof(inputs[0]);
  device_buffer d_inputs((inputs.size() + padding) * sizeof(counter_random_input), device);
  device_buffer d_words((inputs.size() + padding) * sizeof(counter_random_words), device);
  device_buffer d_uniforms(2 * (inputs.size() + padding) * sizeof(double), device);
  device_buffer d_normals((inputs.size() + padding) * sizeof(double), device);
  copy_host_to_device_async(d_inputs, padding * sizeof(counter_random_input), inputs.data(),
                            input_bytes, execution);
  launch_counter_random_samples_for_testing(
      static_cast<const counter_random_input *>(d_inputs.opaque_handle()) + padding,
      static_cast<counter_random_words *>(d_words.opaque_handle()) + padding,
      static_cast<double *>(d_uniforms.opaque_handle()) + 2 * padding,
      static_cast<double *>(d_normals.opaque_handle()) + padding, inputs.size(), threads,
      execution);
  random_result result;
  result.words.resize(inputs.size());
  result.uniforms.resize(2 * inputs.size());
  result.normals.resize(inputs.size());
  copy_device_to_host_async(result.words.data(), d_words,
                            padding * sizeof(counter_random_words),
                            result.words.size() * sizeof(result.words[0]), execution);
  copy_device_to_host_async(result.uniforms.data(), d_uniforms,
                            2 * padding * sizeof(double),
                            result.uniforms.size() * sizeof(result.uniforms[0]), execution);
  copy_device_to_host_async(result.normals.data(), d_normals, padding * sizeof(double),
                            result.normals.size() * sizeof(result.normals[0]), execution);
  execution.synchronize();
  return result;
}

struct random_kat {
  uint32_t seed;
  uint64_t tag;
  uint64_t point;
  uint64_t timestep;
  counter_random_words expected;
};

struct random_tuple {
  uint32_t rank;
  uint32_t chunk;
  uint32_t ft;
  uint32_t state;
  uint32_t component;
  uint32_t cmp;
};

static void check_noisy_kats(int device) {
  const random_kat primitive[] = {
      {0, UINT64_C(0), UINT64_C(0), UINT64_C(0),
       {{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}}},
      {0, UINT64_C(0xffffffffffffffff), UINT64_C(0xffffffffffffffff),
       UINT64_C(0xffffffffffffffff),
       {{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}}},
      {0, UINT64_C(0x0000000500000004), UINT64_C(0x0000000100000000),
       UINT64_C(0x0000000300000002),
       {{0xc427af5du, 0xe75eea3au, 0x47c2b122u, 0x5ffb03c7u}}},
      {0, UINT64_C(0xffffffff00000000), UINT64_C(0x00000000ffffffff),
       UINT64_C(0x00000000ffffffff),
       {{0x6889d68cu, 0xff761653u, 0x6f7b9cfeu, 0x29c99a82u}}}};
  const random_kat tuple_edges[] = {
      {0, UINT64_C(0x4306c0b2a97404c9), 0, 0,
       {{0x54019ce3u, 0xa19d8507u, 0x0e577e3bu, 0xaf9972dbu}}},
      {0, UINT64_C(0xbef1dcd6d5312d4d), 0, 0,
       {{0x1f388419u, 0x5cddba38u, 0x5afc648fu, 0x70a4ca6du}}},
      {1, UINT64_C(0xa1bc4254d2fe4e9d), 1, 1,
       {{0x44be18b1u, 0x18e24fb5u, 0x90f008deu, 0xc1487f12u}}},
      {0xffffffffu, UINT64_C(0x98d9c6f8a9849664), UINT64_MAX, UINT64_MAX,
       {{0xe4854376u, 0x5c198bafu, 0x867e1f33u, 0x4cde7ac9u}}},
      {0x89abcdefu, UINT64_C(0xe2ed7de94152e5e9), UINT64_C(0x00000000ffffffff),
       UINT64_C(0x00000000ffffffff),
       {{0x636b2267u, 0xd8c2af87u, 0x787639a5u, 0x801e0d63u}}},
      {0x89abcdefu, UINT64_C(0xe2ed7de94152e5e9), UINT64_C(0x0000000100000000),
       UINT64_C(0x0000000100000000),
       {{0x04206337u, 0xd940287fu, 0xdebe2404u, 0x185721a2u}}}};
  const random_tuple tuple_edge_fields[] = {
      {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 4, 0}, {1, 1, 1, 1, 5, 1},
      {0xffffffffu, 0xffffffffu, 1, 0xffffffffu, 9, 1},
      {2, 3, 0, 4, 4, 1}, {2, 3, 0, 4, 4, 1}};
  const uint64_t tuple_uniform_bits[][2] = {
      {UINT64_C(0x3fd5006738e00000), UINT64_C(0x3fe433b0a0f00000)},
      {UINT64_C(0x3fbf388419800000), UINT64_C(0x3fd7376e8e200000)},
      {UINT64_C(0x3fd12f862c600000), UINT64_C(0x3fb8e24fb5800000)},
      {UINT64_C(0x3fec90a86ed00000), UINT64_C(0x3fd70662ebe00000)},
      {UINT64_C(0x3fd8dac899e00000), UINT64_C(0x3feb1855f0f00000)},
      {UINT64_C(0x3f90818cde000000), UINT64_C(0x3feb28050ff00000)}};
  /* CUDA 12.8, sm_100, nvcc -O3 -lineinfo --fmad=false on both GB200s. */
  const uint64_t tuple_normal_bits[] = {
      UINT64_C(0xbff034f4ebc3d891), UINT64_C(0xbff55b5628f4b510),
      UINT64_C(0x3ff5415ad6685aeb), UINT64_C(0xbfd3680229c50f68),
      UINT64_C(0x3fe921728857f2d5), UINT64_C(0x3ffab39d90473d5f)};
  /* Version-1 domain/FNV words and loop counters are zero-based.  These
     literal rows are the frozen j=0..31 corpus from the validation contract. */
  const random_kat corpus[] = {
      {0x12345678u, UINT64_C(0x79b4cee9ad560d46), UINT64_C(0x0102030405060708), UINT64_C(0xfedcba9876543210), {{0xc15d956cu, 0x4ae31514u, 0x80b918bbu, 0x2d815fa1u}}},
      {0xb06bd031u, UINT64_C(0xac0f5aaa0bde8fc2), UINT64_C(0x0102030505060709), UINT64_C(0xfedcba977654320f), {{0xa4046b52u, 0xf26794c9u, 0x93a41d63u, 0x0af65971u}}},
      {0x4ea349eau, UINT64_C(0x2105cdf522f9ed13), UINT64_C(0x010203060506070a), UINT64_C(0xfedcba967654320e), {{0xa7b30f1du, 0x710f1756u, 0x346c8051u, 0x28bde24au}}},
      {0xecdac3a3u, UINT64_C(0xafc25b72d6eba997), UINT64_C(0x010203070506070b), UINT64_C(0xfedcba957654320d), {{0x1c5070c6u, 0xdaab160du, 0xc8ba201au, 0xcca396cau}}},
      {0x8b123d5cu, UINT64_C(0xdc99aef59915dece), UINT64_C(0x010203080506070c), UINT64_C(0xfedcba947654320c), {{0x349bfb21u, 0x6dbbf4feu, 0x26d1d71au, 0x471f7ba6u}}},
      {0x2949b715u, UINT64_C(0x6be48fd1186c3fad), UINT64_C(0x010203090506070d), UINT64_C(0xfedcba937654320b), {{0x1368df8au, 0x7e13830eu, 0x5f8d90b0u, 0x2d6a4c6bu}}},
      {0xc78130ceu, UINT64_C(0xfd47acc8d8855de4), UINT64_C(0x0102030a0506070e), UINT64_C(0xfedcba927654320a), {{0x3e1afe11u, 0xcc81c822u, 0xcede5643u, 0x91e5d400u}}},
      {0x65b8aa87u, UINT64_C(0xab8d54ad62cb08e4), UINT64_C(0x0102030b0506070f), UINT64_C(0xfedcba9176543209), {{0x85928f77u, 0xe704ae86u, 0x1af35bccu, 0xa081ef2du}}},
      {0x03f02440u, UINT64_C(0x3536f41c7c28873d), UINT64_C(0x0102030c05060710), UINT64_C(0xfedcba9076543208), {{0xb54f3b83u, 0x54d54b23u, 0xcae184b3u, 0x4af71f47u}}},
      {0xa2279df9u, UINT64_C(0xeba663b8aef3e135), UINT64_C(0x0102030d05060711), UINT64_C(0xfedcba8f76543207), {{0x66ddd2fcu, 0xf0560170u, 0x865b255du, 0x5de86495u}}},
      {0x405f17b2u, UINT64_C(0xbd8d2c1ee6dd1ce9), UINT64_C(0x0102030e05060712), UINT64_C(0xfedcba8e76543206), {{0xb319b909u, 0x46ce5731u, 0x67275463u, 0x82b3c551u}}},
      {0xde96916bu, UINT64_C(0x7753edaa12fdd5b1), UINT64_C(0x0102030f05060713), UINT64_C(0xfedcba8d76543205), {{0xc5c6ef68u, 0xac98a341u, 0x7b871a95u, 0xbdbeb990u}}},
      {0x7cce0b24u, UINT64_C(0xac5508e47dadb9e0), UINT64_C(0x0102031005060714), UINT64_C(0xfedcba8c76543204), {{0x214cf1d7u, 0xcf8c1b99u, 0x3b4ed448u, 0x41b0505au}}},
      {0x1b0584ddu, UINT64_C(0xa351d12812fbe228), UINT64_C(0x0102031105060715), UINT64_C(0xfedcba8b76543203), {{0x514a640bu, 0x9960c471u, 0x7c09ae4fu, 0x607fd093u}}},
      {0xb93cfe96u, UINT64_C(0x7cd40de828062879), UINT64_C(0x0102031205060716), UINT64_C(0xfedcba8a76543202), {{0x9718a705u, 0x49f604b4u, 0x12c25424u, 0xa289434bu}}},
      {0x5774784fu, UINT64_C(0x731ee6e49edd34fe), UINT64_C(0x0102031305060717), UINT64_C(0xfedcba8976543201), {{0x322b3532u, 0xc030d234u, 0xca1e54cau, 0x2e63e9bbu}}},
      {0xf5abf208u, UINT64_C(0x229e95e219062bb7), UINT64_C(0x0102031405060718), UINT64_C(0xfedcba8876543200), {{0x8708bc12u, 0xad0b9b80u, 0x884e7fcdu, 0x4f1c69f3u}}},
      {0x93e36bc1u, UINT64_C(0x0f61a55d709f9413), UINT64_C(0x0102031505060719), UINT64_C(0xfedcba87765431ff), {{0x18893ae4u, 0x563ec6afu, 0x8da4d39cu, 0x35a4bf03u}}},
      {0x321ae57au, UINT64_C(0x391954c62eb06622), UINT64_C(0x010203160506071a), UINT64_C(0xfedcba86765431fe), {{0x95f68e17u, 0xbc851e03u, 0xeed44d18u, 0x4142b9bcu}}},
      {0xd0525f33u, UINT64_C(0x823e65fedbb30886), UINT64_C(0x010203170506071b), UINT64_C(0xfedcba85765431fd), {{0x8c20e66cu, 0x3be4d8b4u, 0x2c438a8au, 0x847f3afdu}}},
      {0x6e89d8ecu, UINT64_C(0xea9d92c0ce198a9a), UINT64_C(0x010203180506071c), UINT64_C(0xfedcba84765431fc), {{0x31b28413u, 0x589fc158u, 0x6b37332eu, 0x97997a2eu}}},
      {0x0cc152a5u, UINT64_C(0xcf36da847d2d43fe), UINT64_C(0x010203190506071d), UINT64_C(0xfedcba83765431fb), {{0x5d3dade6u, 0x3ae9c94au, 0x0ae1fee4u, 0x77f51fcfu}}},
      {0xaaf8cc5eu, UINT64_C(0xe626cbd1f0a0f577), UINT64_C(0x0102031a0506071e), UINT64_C(0xfedcba82765431fa), {{0x34328bc7u, 0xc59f6949u, 0xd3b93f5fu, 0xcff221e5u}}},
      {0x49304617u, UINT64_C(0xcac013959fb4aedb), UINT64_C(0x0102031b0506071f), UINT64_C(0xfedcba81765431f9), {{0x5d2b52b5u, 0x0d122937u, 0x45cee6b0u, 0x422da255u}}},
      {0xe767bfd0u, UINT64_C(0xc073cec354ff24aa), UINT64_C(0x0102031c05060720), UINT64_C(0xfedcba80765431f8), {{0xfaa56744u, 0x5d1c32abu, 0xe29da8feu, 0xf605974au}}},
      {0x859f3989u, UINT64_C(0x0a273359cd666b69), UINT64_C(0x0102031d05060721), UINT64_C(0xfedcba7f765431f7), {{0x0567bb4eu, 0x5fa21ad6u, 0xc7b81549u, 0xbe3fc7c0u}}},
      {0x23d6b342u, UINT64_C(0x84bfcad93f0569b8), UINT64_C(0x0102031e05060722), UINT64_C(0xfedcba7e765431f6), {{0x0057a75cu, 0x6a6f32a9u, 0x44de56cdu, 0x1b887da9u}}},
      {0xc20e2cfbu, UINT64_C(0x49cff83617c534a0), UINT64_C(0x0102031f05060723), UINT64_C(0xfedcba7d765431f5), {{0x4a13cabbu, 0x12e6f347u, 0xc9abe7dbu, 0x1a0e347fu}}},
      {0x6045a6b4u, UINT64_C(0xa61fb7f39cec0491), UINT64_C(0x0102032005060724), UINT64_C(0xfedcba7c765431f4), {{0xb05ff03fu, 0xdc6be07eu, 0xd93379bdu, 0x0eeb3731u}}},
      {0xfe7d206du, UINT64_C(0x06a41bdb77bce679), UINT64_C(0x0102032105060725), UINT64_C(0xfedcba7b765431f3), {{0xa5241b50u, 0x95feac78u, 0xce5884cdu, 0xc6db7d45u}}},
      {0x9cb49a26u, UINT64_C(0x82ae29fbb484254d), UINT64_C(0x0102032205060726), UINT64_C(0xfedcba7a765431f2), {{0xd9ad5dbcu, 0x88790061u, 0xf3088aabu, 0x2102c11bu}}},
      {0x3aec13dfu, UINT64_C(0xeb5c559b37dab62d), UINT64_C(0x0102032305060727), UINT64_C(0xfedcba79765431f1), {{0x3176d543u, 0x4a7772e4u, 0x6a5d16a8u, 0x2e356d53u}}}};

  std::vector<counter_random_input> inputs;
  for (const random_kat &kat : primitive)
    inputs.push_back(counter_random_input{kat.seed, kat.tag, kat.point, kat.timestep});
  for (size_t row = 0; row < sizeof(tuple_edges) / sizeof(tuple_edges[0]); ++row) {
    const random_tuple &tuple = tuple_edge_fields[row];
    const uint64_t tag = counter_random_stream_tag(
        1, tuple.rank, tuple.chunk, tuple.ft, tuple.state, tuple.component, tuple.cmp);
    require(tag == tuple_edges[row].tag, "tuple edge FNV stream tag differs");
    inputs.push_back(counter_random_input{tuple_edges[row].seed, tag, tuple_edges[row].point,
                                          tuple_edges[row].timestep});
  }
  for (size_t row = 0; row < sizeof(corpus) / sizeof(corpus[0]); ++row) {
    const uint32_t ft = uint32_t(row & 1);
    const random_tuple tuple = {uint32_t(row), uint32_t(31 - row), ft, uint32_t(3 * row),
                                uint32_t((ft ? 5 : 0) + row % 5),
                                uint32_t((row >> 1) & 1)};
    const uint64_t tag = counter_random_stream_tag(
        1, tuple.rank, tuple.chunk, tuple.ft, tuple.state, tuple.component, tuple.cmp);
    require(tag == corpus[row].tag, "corpus FNV stream tag differs");
    inputs.push_back(counter_random_input{corpus[row].seed, tag, corpus[row].point,
                                          corpus[row].timestep});
  }
  const random_result observed = sample_device(device, inputs, 64);
  size_t cursor = 0;
  for (const random_kat &kat : primitive) {
    for (int lane = 0; lane < 4; ++lane)
      require(observed.words[cursor].lane[lane] == kat.expected.lane[lane],
              "device Philox primitive KAT differs");
    ++cursor;
  }
  for (size_t row = 0; row < sizeof(tuple_edges) / sizeof(tuple_edges[0]); ++row, ++cursor) {
    for (int lane = 0; lane < 4; ++lane)
      require(observed.words[cursor].lane[lane] == tuple_edges[row].expected.lane[lane],
              "device tuple Philox KAT differs");
    require(double_bits(observed.uniforms[2 * cursor]) == tuple_uniform_bits[row][0] &&
                double_bits(observed.uniforms[2 * cursor + 1]) == tuple_uniform_bits[row][1],
            "device uniform conversion KAT differs");
    require(double_bits(observed.normals[cursor]) == tuple_normal_bits[row],
            "device Box-Muller build-tuple KAT differs");
  }
  for (const random_kat &kat : corpus) {
    for (int lane = 0; lane < 4; ++lane)
      require(observed.words[cursor].lane[lane] == kat.expected.lane[lane],
              "device Philox corpus KAT differs");
    ++cursor;
  }
  std::cout << "device " << device << ": noisy normal tuple bits";
  for (size_t row = 0; row < sizeof(tuple_edges) / sizeof(tuple_edges[0]); ++row)
    std::cout << " " << std::hex << std::setw(16) << std::setfill('0')
              << double_bits(observed.normals[sizeof(primitive) / sizeof(primitive[0]) + row]);
  std::cout << std::dec << " PASS\n";
}

static bool same_random_result(const random_result &a, const random_result &b) {
  if (a.words.size() != b.words.size() || a.uniforms.size() != b.uniforms.size() ||
      a.normals.size() != b.normals.size())
    return false;
  for (size_t i = 0; i < a.words.size(); ++i)
    for (int lane = 0; lane < 4; ++lane)
      if (a.words[i].lane[lane] != b.words[i].lane[lane]) return false;
  for (size_t i = 0; i < a.uniforms.size(); ++i)
    if (double_bits(a.uniforms[i]) != double_bits(b.uniforms[i])) return false;
  for (size_t i = 0; i < a.normals.size(); ++i)
    if (double_bits(a.normals[i]) != double_bits(b.normals[i])) return false;
  return true;
}

static std::vector<counter_random_input> replay_inputs(size_t count) {
  std::vector<counter_random_input> inputs(count);
  for (size_t i = 0; i < count; ++i) {
    const uint32_t ft = uint32_t(i & 1);
    inputs[i].semantic_seed = uint32_t(0x6a09e667u + uint32_t(i) * 0x9e3779b9u);
    inputs[i].stream_tag = counter_random_stream_tag(
        1, uint32_t(i % 17), uint32_t(i % 23), ft, uint32_t(5 * i),
        uint32_t((ft ? 5 : 0) + i % 5), uint32_t((i >> 1) & 1));
    inputs[i].point_ordinal = UINT64_C(0x1020304050607080) + UINT64_C(0x100000001) * i;
    inputs[i].timestep = UINT64_C(0xf0e0d0c0b0a09080) - UINT64_C(0x100000001) * i;
  }
  return inputs;
}

static void check_noisy_replay(int device) {
  const std::vector<counter_random_input> inputs = replay_inputs(4097);
  const random_result baseline = sample_device(device, inputs, 256);
  const unsigned int block_sizes[] = {32, 64, 128, 256, 512};
  for (unsigned int threads : block_sizes)
    require(same_random_result(baseline, sample_device(device, inputs, threads, 17)),
            "counter RNG changed with CUDA block size or allocation address");

  const size_t split = 1379;
  const std::vector<counter_random_input> first(inputs.begin(), inputs.begin() + split);
  const std::vector<counter_random_input> second(inputs.begin() + split, inputs.end());
  const random_result first_result = sample_device(device, first, 64);
  const random_result second_result = sample_device(device, second, 512);
  for (size_t i = 0; i < split; ++i) {
    for (int lane = 0; lane < 4; ++lane)
      require(first_result.words[i].lane[lane] == baseline.words[i].lane[lane],
              "counter RNG changed after first split");
    require(double_bits(first_result.normals[i]) == double_bits(baseline.normals[i]),
            "normal RNG changed after first split");
  }
  for (size_t i = split; i < inputs.size(); ++i) {
    const size_t j = i - split;
    for (int lane = 0; lane < 4; ++lane)
      require(second_result.words[j].lane[lane] == baseline.words[i].lane[lane],
              "counter RNG changed after second split");
    require(double_bits(second_result.normals[j]) == double_bits(baseline.normals[i]),
            "normal RNG changed after second split");
  }

  std::vector<counter_random_input> reversed = inputs;
  std::reverse(reversed.begin(), reversed.end());
  const random_result reverse_result = sample_device(device, reversed, 128);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const size_t j = inputs.size() - 1 - i;
    for (int lane = 0; lane < 4; ++lane)
      require(reverse_result.words[j].lane[lane] == baseline.words[i].lane[lane],
              "counter RNG changed after reverse scheduling");
    require(double_bits(reverse_result.normals[j]) == double_bits(baseline.normals[i]),
            "normal RNG changed after reverse scheduling");
  }
  std::cout << "device " << device << ": noisy counter replay PASS\n";
}

static void check_noisy_key_sensitivity(int device) {
  struct semantic_key {
    uint32_t rank, chunk, ft, state, component, cmp;
  };
  const semantic_key base = {17, 23, 0, 29, 4, 0};
  const uint32_t base_seed = 0x12345678u;
  const uint64_t base_point = UINT64_C(0x1020304050607080);
  const uint64_t base_time = UINT64_C(0x1122334455667788);
  const auto input = [&](semantic_key key, uint32_t seed, uint64_t point, uint64_t timestep) {
    return counter_random_input{
        seed,
        counter_random_stream_tag(1, key.rank, key.chunk, key.ft, key.state, key.component,
                                  key.cmp),
        point, timestep};
  };
  std::vector<counter_random_input> inputs;
  inputs.push_back(input(base, base_seed, base_point, base_time));
  inputs.push_back(input(base, base_seed + 1, base_point, base_time));
  inputs.push_back(input(base, base_seed, base_point, base_time + 1));
  inputs.push_back(input(base, base_seed, base_point + 1, base_time));
  semantic_key changed = base;
#define ADD_CHANGED(member)                                                                         \
  do {                                                                                              \
    changed = base;                                                                                 \
    ++changed.member;                                                                               \
    inputs.push_back(input(changed, base_seed, base_point, base_time));                             \
  } while (0)
  ADD_CHANGED(rank);
  ADD_CHANGED(chunk);
  ADD_CHANGED(ft);
  ADD_CHANGED(state);
  ADD_CHANGED(component);
  ADD_CHANGED(cmp);
#undef ADD_CHANGED
  const random_result observed = sample_device(device, inputs, 64);
  for (size_t i = 1; i < inputs.size(); ++i) {
    bool words_differ = false;
    for (int lane = 0; lane < 4; ++lane)
      words_differ |= observed.words[i].lane[lane] != observed.words[0].lane[lane];
    require(words_differ && double_bits(observed.normals[i]) != double_bits(observed.normals[0]),
            "counter RNG did not respond to one semantic-key field");
  }

  std::unordered_set<uint64_t> tags;
  const size_t count = 1u << 16;
  tags.reserve(count * 2);
  for (size_t i = 0; i < count; ++i) {
    semantic_key key = {uint32_t(i), uint32_t((i >> 8) & 15), uint32_t((i >> 7) & 1),
                        uint32_t((i >> 3) & 15), uint32_t(i % 6), uint32_t((i >> 6) & 1)};
    require(tags.insert(counter_random_stream_tag(1, key.rank, key.chunk, key.ft, key.state,
                                                  key.component, key.cmp))
                .second,
            "counter RNG stream-tag collision in deterministic large-set scan");
  }
  std::cout << "device " << device << ": noisy key sensitivity PASS\n";
}

static void check_noisy_statistics(int device) {
  const size_t count = size_t(1) << 20;
  std::vector<counter_random_input> inputs(count);
  for (size_t i = 0; i < count; ++i) {
    inputs[i].semantic_seed = 0x243f6a88u;
    inputs[i].stream_tag = counter_random_stream_tag(
        1, uint32_t(i >> 20), uint32_t(i), uint32_t(i & 1), uint32_t(i >> 1),
        uint32_t((i & 1) ? 5 : 0), uint32_t((i >> 2) & 1));
    inputs[i].point_ordinal = UINT64_C(0x9e3779b97f4a7c15) * i;
    inputs[i].timestep = UINT64_C(0x123456789abcdef0);
  }
  const random_result observed = sample_device(device, inputs, 256);
  long double sum = 0, sum2 = 0;
  for (double x : observed.normals) {
    sum += x;
    sum2 += x * x;
  }
  const long double mean = sum / count;
  const long double second_moment = sum2 / count;
  long double even_sum = 0, odd_sum = 0;
  for (size_t i = 0; i < count; i += 2) {
    even_sum += observed.normals[i];
    odd_sum += observed.normals[i + 1];
  }
  const long double pair_count = count / 2;
  const long double even_mean = even_sum / pair_count;
  const long double odd_mean = odd_sum / pair_count;
  long double covariance = 0, even_var = 0, odd_var = 0;
  for (size_t i = 0; i < count; i += 2) {
    const long double even = observed.normals[i] - even_mean;
    const long double odd = observed.normals[i + 1] - odd_mean;
    covariance += even * odd;
    even_var += even * even;
    odd_var += odd * odd;
  }
  const long double correlation = covariance / std::sqrt(even_var * odd_var);
  require(std::fabs(double(mean)) <= 0.005859375, "counter normal mean exceeds threshold");
  require(std::fabs(double(second_moment - 1)) <= 0.010286407592029855,
          "counter normal second moment exceeds threshold");
  require(std::fabs(double(correlation)) <= 0.007859375,
          "counter normal adjacent-stream correlation exceeds threshold");
  std::cout << std::setprecision(17) << "device " << device << ": noisy stats mean="
            << double(mean) << " second_moment=" << double(second_moment)
            << " correlation=" << double(correlation) << " PASS\n";
}

static double noisy_test_amplitude(double omega, double gamma, double noise, double dt) {
  const test_realnum g2pi = test_realnum(gamma) * 2 * 3.14159265358979323846;
  const test_realnum w2pi = test_realnum(omega) * 2 * 3.14159265358979323846;
  const test_realnum step = test_realnum(dt);
  return double(w2pi * test_realnum(noise) * std::sqrt(g2pi) * step * step /
                (1 + g2pi * step / 2));
}

template <typename T>
static double noisy_variance_cell(int device, double omega, double gamma, double noise,
                                  double dt, double sigma_value) {
  const size_t count = size_t(1) << 20;
  device_scope selected(device);
  stream execution;
  const size_t bytes = count * sizeof(T);
  std::vector<T> zero(count, T(0)), sigma(count, T(sigma_value)), observed(count);
  device_buffer d_p(bytes, device), d_sigma(bytes, device), d_seed(sizeof(noisy_seed_block), device);
  const noisy_seed_block seed = {0x243f6a88u, 1};
  copy_host_to_device_async(d_p, 0, zero.data(), bytes, execution);
  copy_host_to_device_async(d_sigma, 0, sigma.data(), bytes, execution);
  copy_host_to_device_async(d_seed, 0, &seed, sizeof(seed), execution);
  compiled_polarization_update update = {};
  update.kind = compiled_polarization_update::kind_type::noisy_add;
  update.noisy.region.counts[0] = update.noisy.region.counts[1] = 1;
  update.noisy.region.counts[2] = count;
  update.noisy.region.strides[2] = 1;
  update.noisy.p = d_p.opaque_handle();
  update.noisy.diagonal_sigma = d_sigma.opaque_handle();
  update.noisy.amplitude = noisy_test_amplitude(omega, gamma, noise, dt);
  update.noisy.stream_tag = UINT64_C(0xbef1dcd6d5312d4d);
  update.noisy.point_ordinal_base = 0;
  update.noisy.precision = sizeof(T) == sizeof(float) ? scalar_precision::f32
                                                      : scalar_precision::f64;
  launch_polarization_update(
      update, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), 37, execution);
  copy_device_to_host_async(observed.data(), d_p, 0, bytes, execution);
  execution.synchronize();
  long double sum = 0, sum2 = 0;
  for (T value : observed) {
    sum += value;
    sum2 += double(value) * double(value);
  }
  const long double mean = sum / count;
  return double(sum2 / count - mean * mean);
}

template <typename T> static void check_noisy_variance_scaling(int device) {
  struct cell {
    const char *name;
    double omega, gamma, noise, dt, sigma;
  };
  const cell cells[] = {{"base", 0.73, 0.06, 0.03125, 0.017, 0.83},
                        {"amplitude", 0.73, 0.06, 0.0625, 0.017, 0.83},
                        {"sigma", 0.73, 0.06, 0.03125, 0.017, 1.37},
                        {"omega", 0.91, 0.06, 0.03125, 0.017, 0.83},
                        {"gamma", 0.73, 0.11, 0.03125, 0.017, 0.83},
                        {"dt", 0.73, 0.06, 0.03125, 0.023, 0.83},
                        {"negative", 0.73, 0.06, -0.03125, 0.017, 0.83}};
  double observed[sizeof(cells) / sizeof(cells[0])];
  double expected[sizeof(cells) / sizeof(cells[0])];
  for (size_t i = 0; i < sizeof(cells) / sizeof(cells[0]); ++i) {
    observed[i] = noisy_variance_cell<T>(device, cells[i].omega, cells[i].gamma,
                                         cells[i].noise, cells[i].dt, cells[i].sigma);
    const double amplitude = noisy_test_amplitude(cells[i].omega, cells[i].gamma,
                                                  cells[i].noise, cells[i].dt);
    const test_realnum stored_sigma = test_realnum(T(cells[i].sigma));
    const test_realnum standard_deviation =
        test_realnum(amplitude) * std::sqrt(stored_sigma);
    expected[i] = double(standard_deviation) * double(standard_deviation);
  }
  for (size_t i = 1; i < sizeof(cells) / sizeof(cells[0]); ++i) {
    const double observed_ratio = observed[i] / observed[0];
    const double expected_ratio = expected[i] / expected[0];
    const double relative = std::fabs(observed_ratio / expected_ratio - 1.0);
    require(relative <= 0.02, "noisy-add variance scaling exceeds two percent");
    std::cout << std::setprecision(17) << "device " << device << ": "
              << (sizeof(T) == sizeof(float) ? "f32" : "f64") << " noisy variance "
              << cells[i].name << " observed=" << observed_ratio
              << " expected=" << expected_ratio << " relative=" << relative << "\n";
  }
}

template <typename T> static void check_noisy_add_device(int device) {
  device_scope selected(device);
  stream execution;
  const size_t elements = 24;
  const size_t bytes = elements * sizeof(T);
  std::vector<T> p(elements), sigma(elements, T(1)), observed(elements);
  for (size_t i = 0; i < elements; ++i) p[i] = T(0.125 * double(int(i) - 7));
  const T test_sigma[] = {T(0.29), T(0.83), T(1.37), T(2.11), T(3.07), T(4.43)};
  for (size_t i = 0; i < 6; ++i) sigma[5 + i] = test_sigma[i];
  device_buffer d_p(bytes, device), d_sigma(bytes, device), d_seed(sizeof(noisy_seed_block), device);
  const noisy_seed_block seed = {0x89abcdefu, 1};
  copy_host_to_device_async(d_p, 0, p.data(), bytes, execution);
  copy_host_to_device_async(d_sigma, 0, sigma.data(), bytes, execution);
  copy_host_to_device_async(d_seed, 0, &seed, sizeof(seed), execution);

  compiled_polarization_update update = {};
  update.kind = compiled_polarization_update::kind_type::noisy_add;
  update.noisy.region.base = 5;
  update.noisy.region.counts[0] = update.noisy.region.counts[1] = 1;
  update.noisy.region.counts[2] = 6;
  update.noisy.region.strides[2] = 1;
  update.noisy.p = d_p.opaque_handle();
  update.noisy.diagonal_sigma = d_sigma.opaque_handle();
  const test_realnum gamma2pi = test_realnum(0.06) * 2 * 3.14159265358979323846;
  const test_realnum omega2pi = test_realnum(0.73) * 2 * 3.14159265358979323846;
  const test_realnum test_dt = test_realnum(0.017);
  update.noisy.amplitude = double(omega2pi * test_realnum(0.03125) * std::sqrt(gamma2pi) *
                                  test_dt * test_dt / (1 + gamma2pi * test_dt / 2));
  const double nonzero_amplitude = update.noisy.amplitude;
  update.noisy.stream_tag = UINT64_C(0xe2ed7de94152e5e9);
  update.noisy.point_ordinal_base = UINT64_C(0x00000000fffffffd);
  update.noisy.precision = sizeof(T) == sizeof(float) ? scalar_precision::f32
                                                      : scalar_precision::f64;
  const uint64_t timestep = UINT64_C(0x00000001fffffffd);
  launch_polarization_update(
      update, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), timestep,
      execution);
  copy_device_to_host_async(observed.data(), d_p, 0, bytes, execution);
  execution.synchronize();

  std::vector<counter_random_input> inputs(6);
  for (size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = counter_random_input{seed.semantic_seed, update.noisy.stream_tag,
                                     update.noisy.point_ordinal_base + i, timestep};
  const random_result samples = sample_device(device, inputs, 32);
  for (size_t linear = 0; linear < 6; ++linear) {
    const size_t i = update.noisy.region.base + linear;
    const test_realnum standard_deviation =
        test_realnum(update.noisy.amplitude) * std::sqrt(test_realnum(sigma[i]));
    const volatile double increment = double(standard_deviation) * samples.normals[linear];
    const T expected = T(double(p[i]) + increment);
    require(memcmp(&observed[i], &expected, sizeof(T)) == 0,
            "noisy-add final storage bit pattern differs");
  }
#if MEEP_SINGLE
  const uint32_t expected_f32_bits[] = {0xbe80048fu, 0xbe0001c2u, 0xb6a6fb09u,
                                        0x3dfffc4fu, 0x3e7ff0b3u, 0x3ebffd47u};
  const uint64_t expected_f64_bits[] = {
      UINT64_C(0xbfd00091ddd2ece4), UINT64_C(0xbfc0003833d53b65),
      UINT64_C(0xbed4df6111f749b1), UINT64_C(0x3fbfff89e90733b9),
      UINT64_C(0x3fcffe1653c50f7e), UINT64_C(0x3fd7ffa8eb67e213)};
#else
  const uint32_t expected_f32_bits[] = {0xbe80048fu, 0xbe0001c2u, 0xb6a6fb08u,
                                        0x3dfffc4fu, 0x3e7ff0b3u, 0x3ebffd47u};
  const uint64_t expected_f64_bits[] = {
      UINT64_C(0xbfd00091ddd2f6e7), UINT64_C(0xbfc0003833d52ea3),
      UINT64_C(0xbed4df60f4baf060), UINT64_C(0x3fbfff89e9073c72),
      UINT64_C(0x3fcffe1653c42a80), UINT64_C(0x3fd7ffa8eb684de9)};
#endif
  for (size_t linear = 0; linear < 6; ++linear) {
    const size_t i = update.noisy.region.base + linear;
    if (sizeof(T) == sizeof(float)) {
      uint32_t bits = 0;
      memcpy(&bits, &observed[i], sizeof(bits));
      require(bits == expected_f32_bits[linear], "f32 noisy-add literal KAT differs");
    }
    else {
      uint64_t bits = 0;
      memcpy(&bits, &observed[i], sizeof(bits));
      require(bits == expected_f64_bits[linear], "f64 noisy-add literal KAT differs");
    }
  }
  for (size_t i = 0; i < elements; ++i)
    if (i < 5 || i >= 11)
      require(memcmp(&observed[i], &p[i], sizeof(T)) == 0,
              "noisy-add sentinel changed");

  const std::vector<T> whole = observed;
  copy_host_to_device_async(d_p, 0, p.data(), bytes, execution);
  compiled_polarization_update first = update, second = update;
  first.noisy.region.counts[2] = 2;
  second.noisy.region.base += 2;
  second.noisy.region.counts[2] = 4;
  second.noisy.point_ordinal_base += 2;
  launch_polarization_update(
      second, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), timestep,
      execution);
  launch_polarization_update(
      first, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), timestep,
      execution);
  copy_device_to_host_async(observed.data(), d_p, 0, bytes, execution);
  execution.synchronize();
  require(memcmp(observed.data(), whole.data(), bytes) == 0,
          "noisy-add whole and reversed split launches differ");

  std::vector<T> zero(elements, T(0)), positive(elements), negative(elements);
  copy_host_to_device_async(d_p, 0, zero.data(), bytes, execution);
  update.noisy.amplitude = std::fabs(update.noisy.amplitude);
  launch_polarization_update(
      update, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), timestep,
      execution);
  copy_device_to_host_async(positive.data(), d_p, 0, bytes, execution);
  execution.synchronize();
  copy_host_to_device_async(d_p, 0, zero.data(), bytes, execution);
  update.noisy.amplitude = -std::fabs(update.noisy.amplitude);
  launch_polarization_update(
      update, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), timestep,
      execution);
  copy_device_to_host_async(negative.data(), d_p, 0, bytes, execution);
  execution.synchronize();
  for (size_t linear = 0; linear < 6; ++linear) {
    const size_t i = update.noisy.region.base + linear;
    require(positive[i] == -negative[i],
            "negative noisy amplitude did not produce an exact paired sign flip");
  }

  std::vector<T> invalid_sigma = sigma;
  invalid_sigma[5] = T(-1);
  invalid_sigma[6] = std::numeric_limits<T>::quiet_NaN();
  invalid_sigma[7] = std::numeric_limits<T>::infinity();
  copy_host_to_device_async(d_p, 0, p.data(), bytes, execution);
  copy_host_to_device_async(d_sigma, 0, invalid_sigma.data(), bytes, execution);
  update.noisy.region.counts[2] = 3;
  update.noisy.amplitude = std::fabs(update.noisy.amplitude);
  launch_polarization_update(
      update, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), timestep,
      execution);
  copy_device_to_host_async(observed.data(), d_p, 0, bytes, execution);
  execution.synchronize();
  require(std::isnan(double(observed[5])) && std::isnan(double(observed[6])) &&
              !std::isfinite(double(observed[7])),
          "dynamic invalid sigma was clamped instead of propagating");

  copy_host_to_device_async(d_p, 0, p.data(), bytes, execution);
  update.noisy.amplitude = -0.0;
  launch_polarization_update(
      update, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), timestep,
      execution);
  copy_device_to_host_async(observed.data(), d_p, 0, bytes, execution);
  execution.synchronize();
  require(std::isnan(double(observed[5])) && std::isnan(double(observed[6])) &&
              std::isnan(double(observed[7])),
          "zero noisy amplitude incorrectly bypassed invalid dynamic sigma");

  std::vector<T> zero_sigma(elements, T(0));
  copy_host_to_device_async(d_p, 0, p.data(), bytes, execution);
  copy_host_to_device_async(d_sigma, 0, zero_sigma.data(), bytes, execution);
  update.noisy.amplitude = std::fabs(nonzero_amplitude);
  launch_polarization_update(
      update, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), timestep,
      execution);
  copy_device_to_host_async(observed.data(), d_p, 0, bytes, execution);
  execution.synchronize();
  require(memcmp(observed.data(), p.data(), bytes) == 0,
          "zero diagonal sigma was not an exact noisy no-op");

  copy_host_to_device_async(d_p, 0, p.data(), bytes, execution);
  copy_host_to_device_async(d_sigma, 0, sigma.data(), bytes, execution);
  update.noisy.region.counts[2] = 6;
  update.noisy.amplitude = -0.0;
  launch_polarization_update(
      update, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), timestep,
      execution);
  copy_device_to_host_async(observed.data(), d_p, 0, bytes, execution);
  execution.synchronize();
  require(memcmp(observed.data(), p.data(), bytes) == 0,
          "finite negative-zero noise was not an exact no-op");

  bool rejected = false;
  compiled_polarization_update malformed = update;
  malformed.noisy.amplitude = std::fabs(malformed.noisy.amplitude);
  malformed.noisy.point_ordinal_base = UINT64_MAX - 3;
  try {
    launch_polarization_update(
        malformed, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), timestep,
        execution);
  }
  catch (const std::overflow_error &) { rejected = true; }
  require(rejected, "noisy-add launch accepted point ordinal overflow");
  std::cout << "device " << device << ": "
            << (sizeof(T) == sizeof(float) ? "f32" : "f64")
            << " noisy-add arithmetic PASS\n";
}

template <typename T> static void check_noisy_recurrence_order(int device) {
  device_scope selected(device);
  stream execution;
  const size_t elements = 8, bytes = elements * sizeof(T), index = 3;
  std::vector<T> p(elements, T(0)), p_prev(elements, T(0)), w(elements, T(0)),
      sigma(elements, T(0.83)), observed(elements);
  p[index] = T(0.37);
  p_prev[index] = T(-0.19);
  w[index] = T(0.23);
  device_buffer d_p(bytes, device), d_prev(bytes, device), d_w(bytes, device),
      d_sigma(bytes, device), d_seed(sizeof(noisy_seed_block), device);
  const noisy_seed_block seed = {0x13579bdfu, 1};
  copy_host_to_device_async(d_p, 0, p.data(), bytes, execution);
  copy_host_to_device_async(d_prev, 0, p_prev.data(), bytes, execution);
  copy_host_to_device_async(d_w, 0, w.data(), bytes, execution);
  copy_host_to_device_async(d_sigma, 0, sigma.data(), bytes, execution);
  copy_host_to_device_async(d_seed, 0, &seed, sizeof(seed), execution);
  compiled_polarization_update recurrence = {}, noise = {};
  recurrence.kind = compiled_polarization_update::kind_type::lorentzian;
  recurrence.lorentzian.region.base = index;
  recurrence.lorentzian.region.counts[0] = recurrence.lorentzian.region.counts[1] =
      recurrence.lorentzian.region.counts[2] = 1;
  recurrence.lorentzian.p = d_p.opaque_handle();
  recurrence.lorentzian.p_prev = d_prev.opaque_handle();
  recurrence.lorentzian.primary_w = d_w.opaque_handle();
  recurrence.lorentzian.diagonal_sigma = d_sigma.opaque_handle();
  recurrence.lorentzian.precision = sizeof(T) == sizeof(float) ? scalar_precision::f32
                                                               : scalar_precision::f64;
  const T omega2pi = T(0.73 * 2 * 3.14159265358979323846);
  const T gamma2pi = T(0.06 * 2 * 3.14159265358979323846);
  const T dt = T(0.017);
  recurrence.lorentzian.omega0dtsqr = double(omega2pi * omega2pi * dt * dt);
  recurrence.lorentzian.gamma1inv = double(T(1) / (T(1) + gamma2pi * dt / T(2)));
  recurrence.lorentzian.gamma1 = double(T(1) - gamma2pi * dt / T(2));
  recurrence.lorentzian.omega0dtsqr_denom = recurrence.lorentzian.omega0dtsqr;
  noise.kind = compiled_polarization_update::kind_type::noisy_add;
  noise.noisy.region = recurrence.lorentzian.region;
  noise.noisy.p = d_p.opaque_handle();
  noise.noisy.diagonal_sigma = d_sigma.opaque_handle();
  noise.noisy.amplitude = noisy_test_amplitude(0.73, 0.06, 0.03125, 0.017);
  noise.noisy.stream_tag = UINT64_C(0xbef1dcd6d5312d4d);
  noise.noisy.point_ordinal_base = 19;
  noise.noisy.precision = recurrence.lorentzian.precision;
  const uint64_t timestep = 23;
  launch_polarization_update(recurrence, NULL, timestep, execution);
  launch_polarization_update(
      noise, static_cast<const noisy_seed_block *>(d_seed.opaque_handle()), timestep, execution);
  copy_device_to_host_async(observed.data(), d_p, 0, bytes, execution);
  execution.synchronize();

  const T forcing = sigma[index] * w[index];
  const T deterministic = T(recurrence.lorentzian.gamma1inv) *
                          (p[index] * (T(2) - T(recurrence.lorentzian.omega0dtsqr_denom)) -
                           T(recurrence.lorentzian.gamma1) * p_prev[index] +
                           T(recurrence.lorentzian.omega0dtsqr) * forcing);
  const counter_random_input sample_input = {seed.semantic_seed, noise.noisy.stream_tag,
                                             noise.noisy.point_ordinal_base, timestep};
  const double sample = sample_device(device, std::vector<counter_random_input>(1, sample_input),
                                      32)
                            .normals[0];
  const test_realnum standard_deviation =
      test_realnum(noise.noisy.amplitude) * std::sqrt(test_realnum(sigma[index]));
  const volatile double increment = double(standard_deviation) * sample;
  const T expected = T(double(deterministic) + increment);
  const T pre_noise = T(recurrence.lorentzian.gamma1inv) *
                      (T(double(p[index]) + increment) *
                           (T(2) - T(recurrence.lorentzian.omega0dtsqr_denom)) -
                       T(recurrence.lorentzian.gamma1) * p_prev[index] +
                       T(recurrence.lorentzian.omega0dtsqr) * forcing);
  require(memcmp(&observed[index], &expected, sizeof(T)) == 0,
          "noisy recurrence-then-add result differs from independent scalar oracle");
  require(memcmp(&observed[index], &pre_noise, sizeof(T)) != 0,
          "noisy ordering oracle does not distinguish pre-recurrence addition");
  require(noisy_test_amplitude(0.0, 0.06, 0.03125, 0.017) == 0.0 &&
              noisy_test_amplitude(0.73, 0.0, 0.03125, 0.017) == 0.0,
          "zero omega/gamma did not derive exact zero noise");
  std::cout << "device " << device << ": "
            << (sizeof(T) == sizeof(float) ? "f32" : "f64")
            << " noisy recurrence order PASS\n";
}

template <typename T>
static T offdiag(const std::vector<T> &coefficient, const std::vector<T> &field, ptrdiff_t i,
                 ptrdiff_t primary_stride, ptrdiff_t cross_stride) {
  return T(0.25) *
         ((field[i] + field[i - cross_stride]) * coefficient[i] +
          (field[i + primary_stride] + field[i + primary_stride - cross_stride]) *
              coefficient[i + primary_stride]);
}

template <typename T>
static double gyrotropic_centered_host(const std::vector<T> &field, ptrdiff_t i,
                                       ptrdiff_t primary_stride, ptrdiff_t cross_stride) {
  return 0.25 * (field[i] + field[i - cross_stride] + field[i + primary_stride] +
                 field[i + primary_stride - cross_stride]);
}

template <typename T>
static void fill_gyro_coefficients(gyrotropic_update_launch &launch, int primary,
                                   gyrotropic_kernel_model model) {
  T bias[3] = {T(0.17), T(-0.23), T(0.31)};
  if (model == gyrotropic_kernel_model::saturated) {
    const T norm = std::sqrt(bias[0] * bias[0] + bias[1] * bias[1] + bias[2] * bias[2]);
    for (int i = 0; i < 3; ++i) bias[i] /= norm;
  }
  T global[3][3] = {};
  global[0][1] = bias[2];
  global[1][0] = -bias[2];
  global[1][2] = bias[0];
  global[2][1] = -bias[0];
  global[2][0] = bias[1];
  global[0][2] = -bias[1];
  const int order[3] = {primary, (primary + 1) % 3, (primary + 2) % 3};
  const T dt = T(0.017), omega0 = T(0.73), gamma0 = T(0.06), alpha = T(0.19);
  const T omega = 2 * 3.14159265358979323846 * omega0 * dt;
  const T gamma = 2 * 3.14159265358979323846 * gamma0 * dt;
  T gd, gx, gy, gz;
  launch.omega = double(omega);
  launch.gamma = double(gamma);
  launch.alpha = double(alpha);
  launch.dt2pi = double(T(2 * 3.14159265358979323846 * dt));
  if (model == gyrotropic_kernel_model::saturated) {
    gd = 0.5;
    gx = -0.5 * alpha * global[1][2];
    gy = -0.5 * alpha * global[2][0];
    gz = -0.5 * alpha * global[0][1];
  }
  else {
    const T a = omega * omega;
    launch.omega0dtsqr = double(a);
    launch.gamma1 = double(T(1) - gamma / T(2));
    launch.diagonal = double(T(2) -
                             (model == gyrotropic_kernel_model::drude ? T(0) : a));
    launch.pt = double(T(3.14159265358979323846 * dt));
    gd = T(1) + gamma / T(2);
    gx = T(launch.pt) * global[1][2];
    gy = T(launch.pt) * global[2][0];
    gz = T(launch.pt) * global[0][1];
  }
  const T invdet = 1.0 / gd / (gd * gd + gx * gx + gy * gy + gz * gz);
  const T inverse[3][3] = {
      {invdet * (gd * gd + gx * gx), invdet * (gx * gy + gd * gz),
       invdet * (gx * gz - gd * gy)},
      {invdet * (gy * gx - gd * gz), invdet * (gd * gd + gy * gy),
       invdet * (gy * gz + gd * gx)},
      {invdet * (gz * gx + gd * gy), invdet * (gz * gy - gd * gx),
       invdet * (gd * gd + gz * gz)}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      launch.gyro[i][j] = double(global[order[i]][order[j]]);
      launch.inverse[i][j] = double(inverse[order[i]][order[j]]);
    }
  launch.model = model;
}

template <typename T> static void check_gyrotropic_device(int device) {
  device_scope selected(device);
  stream execution;
  const size_t elements = 320, bytes = elements * sizeof(T);
  std::vector<T> p[3], p_prev[3], expected[3], expected_prev[3];
  std::vector<T> w[3], sigma(elements);
  for (int d = 0; d < 3; ++d) {
    p[d].resize(elements);
    p_prev[d].resize(elements);
    w[d].resize(elements);
    for (size_t i = 0; i < elements; ++i) {
      p[d][i] = T(0.11 * (d + 1) + 0.0007 * double(i));
      p_prev[d][i] = T(-0.07 * (d + 1) + 0.0003 * double(i));
      w[d][i] = T(0.23 * (d + 1) - 0.0005 * double(i));
    }
  }
  for (size_t i = 0; i < elements; ++i) sigma[i] = T(0.8 + 0.0002 * double(i));
  device_buffer d_p[3], d_pp[3], d_w[3];
  for (int d = 0; d < 3; ++d) {
    d_p[d].allocate(bytes, device);
    d_pp[d].allocate(bytes, device);
    d_w[d].allocate(bytes, device);
    copy_host_to_device_async(d_w[d], 0, w[d].data(), bytes, execution);
  }
  device_buffer d_sigma(bytes, device);
  copy_host_to_device_async(d_sigma, 0, sigma.data(), bytes, execution);

  flat_region region = {};
  region.base = 30;
  region.counts[0] = region.counts[1] = 1;
  region.counts[2] = 257;
  region.strides[2] = 1;
  const scalar_precision precision =
      sizeof(T) == sizeof(float) ? scalar_precision::f32 : scalar_precision::f64;
  size_t cases = 0;
  for (int primary = 0; primary < 3; ++primary)
    for (int model_index = 0; model_index < 3; ++model_index) {
      const gyrotropic_kernel_model model = static_cast<gyrotropic_kernel_model>(model_index);
      const bool magnetic = ((primary + model_index) & 1) != 0;
      const bool have_w1 = model_index != 1;
      const bool have_w2 = model_index != 0;
      for (int d = 0; d < 3; ++d) {
        expected[d] = p[d];
        expected_prev[d] = p_prev[d];
        copy_host_to_device_async(d_p[d], 0, p[d].data(), bytes, execution);
        copy_host_to_device_async(d_pp[d], 0, p_prev[d].data(), bytes, execution);
      }
      gyrotropic_update_launch launch = {};
      launch.region = region;
      launch.precision = precision;
      launch.primary_stride = magnetic ? -2 : 2;
      launch.cross_stride1 = magnetic ? -5 : 5;
      launch.cross_stride2 = magnetic ? -7 : 7;
      for (int d = 0; d < 3; ++d) {
        launch.p[d] = d_p[d].opaque_handle();
        launch.p_prev[d] = d_pp[d].opaque_handle();
      }
      launch.w[0] = d_w[0].opaque_handle();
      launch.w[1] = have_w1 ? d_w[1].opaque_handle() : NULL;
      launch.w[2] = have_w2 ? d_w[2].opaque_handle() : NULL;
      launch.sigma = d_sigma.opaque_handle();
      fill_gyro_coefficients<T>(launch, primary, model);

      for (size_t linear = 0; linear < region.counts[2]; ++linear) {
        const ptrdiff_t i = ptrdiff_t(region.base + linear);
        const T old0 = expected[0][i], old1 = expected[1][i], old2 = expected[2][i];
        const T prev0 = expected_prev[0][i], prev1 = expected_prev[1][i],
                prev2 = expected_prev[2][i];
        const double c1 = have_w1 ? gyrotropic_centered_host(
                                          w[1], i, launch.primary_stride, launch.cross_stride1)
                                  : 0.0;
        const double c2 = have_w2 ? gyrotropic_centered_host(
                                          w[2], i, launch.primary_stride, launch.cross_stride2)
                                  : 0.0;
        T r0, r1, r2;
        if (model == gyrotropic_kernel_model::saturated) {
          const T q0 = -T(launch.omega) * old0 + 0.5 * T(launch.alpha) * prev0 +
                       T(launch.dt2pi) * sigma[i] * w[0][i];
          const T q1 = -T(launch.omega) * old1 + 0.5 * T(launch.alpha) * prev1 +
                       T(launch.dt2pi) * sigma[i] * c1;
          const T q2 = -T(launch.omega) * old2 + 0.5 * T(launch.alpha) * prev2 +
                       T(launch.dt2pi) * sigma[i] * c2;
          r0 = 0.5 * prev0 - T(launch.gamma) * old0 + T(launch.gyro[0][1]) * q1 +
               T(launch.gyro[0][2]) * q2;
          r1 = 0.5 * prev1 - T(launch.gamma) * old1 + T(launch.gyro[1][2]) * q2 +
               T(launch.gyro[1][0]) * q0;
          r2 = 0.5 * prev2 - T(launch.gamma) * old2 + T(launch.gyro[2][0]) * q0 +
               T(launch.gyro[2][1]) * q1;
        }
        else {
          r0 = T(launch.diagonal) * old0 - T(launch.gamma1) * prev0 +
               T(launch.omega0dtsqr) * sigma[i] * w[0][i] -
               T(launch.pt) * T(launch.gyro[0][1]) * prev1 -
               T(launch.pt) * T(launch.gyro[0][2]) * prev2;
          r1 = T(launch.diagonal) * old1 - T(launch.gamma1) * prev1 +
               T(launch.omega0dtsqr) * sigma[i] * c1 -
               T(launch.pt) * T(launch.gyro[1][0]) * prev0 -
               T(launch.pt) * T(launch.gyro[1][2]) * prev2;
          r2 = T(launch.diagonal) * old2 - T(launch.gamma1) * prev2 +
               T(launch.omega0dtsqr) * sigma[i] * c2 -
               T(launch.pt) * T(launch.gyro[2][1]) * prev1 -
               T(launch.pt) * T(launch.gyro[2][0]) * prev0;
        }
        expected_prev[0][i] = old0;
        expected_prev[1][i] = old1;
        expected_prev[2][i] = old2;
        for (int d = 0; d < 3; ++d)
          expected[d][i] = T(launch.inverse[d][0]) * r0 + T(launch.inverse[d][1]) * r1 +
                           T(launch.inverse[d][2]) * r2;
      }

      launch_gyrotropic_update(launch, execution);
      const double tolerance = 16 * std::numeric_limits<T>::epsilon();
      for (int d = 0; d < 3; ++d) {
        std::vector<T> observed(elements), observed_prev(elements);
        copy_device_to_host_async(observed.data(), d_p[d], 0, bytes, execution);
        copy_device_to_host_async(observed_prev.data(), d_pp[d], 0, bytes, execution);
        execution.synchronize();
        for (size_t i = 0; i < elements; ++i) {
          require((sizeof(T) == sizeof(float) && observed[i] == expected[d][i]) ||
                      (sizeof(T) != sizeof(float) &&
                       std::fabs(double(observed[i] - expected[d][i])) <=
                           tolerance * (1 + std::fabs(double(expected[d][i])))),
                  "gyrotropic recurrence or sentinel differs");
          require(std::fabs(double(observed_prev[i] - expected_prev[d][i])) <=
                      tolerance * (1 + std::fabs(double(expected_prev[d][i]))),
                  "gyrotropic P_prev rotation or sentinel differs");
        }
      }
      ++cases;
    }

  gyrotropic_update_launch malformed = {};
  malformed.region = region;
  malformed.precision = precision;
  malformed.primary_stride = 2;
  malformed.cross_stride1 = 5;
  malformed.cross_stride2 = 7;
  for (int d = 0; d < 3; ++d) {
    malformed.p[d] = d_p[d].opaque_handle();
    malformed.p_prev[d] = d_pp[d].opaque_handle();
    malformed.w[d] = d_w[d].opaque_handle();
  }
  malformed.sigma = d_sigma.opaque_handle();
  fill_gyro_coefficients<T>(malformed, 0, gyrotropic_kernel_model::lorentzian);
  const auto rejected = [&](const gyrotropic_update_launch &candidate) {
    try {
      launch_gyrotropic_update(candidate, execution);
      return false;
    }
    catch (const std::invalid_argument &) { return true; }
    catch (const std::overflow_error &) { return true; }
  };
  gyrotropic_update_launch bad = malformed;
  bad.p_prev[2] = NULL;
  require(rejected(bad), "gyrotropic launch accepted missing state");
  bad = malformed;
  bad.p_prev[2] = bad.p[0];
  require(rejected(bad), "gyrotropic launch accepted aliased state");
  bad = malformed;
  bad.model = static_cast<gyrotropic_kernel_model>(99);
  require(rejected(bad), "gyrotropic launch accepted invalid model");
  bad = malformed;
  bad.precision = static_cast<scalar_precision>(99);
  require(rejected(bad), "gyrotropic launch accepted invalid precision");
  bad = malformed;
  bad.region.counts[2] = 0;
  require(rejected(bad), "gyrotropic launch accepted an empty region");
  bad = malformed;
  bad.region.counts[0] = std::numeric_limits<size_t>::max();
  bad.region.counts[1] = 2;
  require(rejected(bad), "gyrotropic launch accepted a region-size overflow");
  std::cout << "device " << device << ": " << cases << " "
            << (sizeof(T) == sizeof(float) ? "f32" : "f64") << " gyrotropic cases PASS\n";
}

template <typename T> static void check_device(int device) {
  device_scope selected(device);
  stream execution;
  const size_t elements = 320;
  const size_t bytes = elements * sizeof(T);
  std::vector<T> p(elements), p_prev(elements), w(elements), w1(elements), w2(elements),
      sigma(elements), sigma1(elements), sigma2(elements), observed(elements),
      observed_prev(elements);
  for (size_t i = 0; i < elements; ++i) {
    p[i] = T(0.2 + 0.001 * double(i));
    p_prev[i] = T(-0.1 + 0.0007 * double(i));
    w[i] = T(0.4 - 0.0009 * double(i));
    w1[i] = T(-0.3 + 0.0011 * double(i));
    w2[i] = T(0.1 + 0.0005 * double(i));
    sigma[i] = T(0.8 + 0.0003 * double(i));
    sigma1[i] = T(0.04 - 0.0001 * double(i));
    sigma2[i] = T(-0.03 + 0.0002 * double(i));
  }

  device_buffer d_p(bytes, device), d_p_prev(bytes, device), d_w(bytes, device),
      d_w1(bytes, device), d_w2(bytes, device), d_sigma(bytes, device),
      d_sigma1(bytes, device), d_sigma2(bytes, device);
  copy_host_to_device_async(d_w, 0, w.data(), bytes, execution);
  copy_host_to_device_async(d_w1, 0, w1.data(), bytes, execution);
  copy_host_to_device_async(d_w2, 0, w2.data(), bytes, execution);
  copy_host_to_device_async(d_sigma1, 0, sigma1.data(), bytes, execution);
  copy_host_to_device_async(d_sigma2, 0, sigma2.data(), bytes, execution);

  flat_region region = {};
  region.base = 30;
  region.counts[0] = 1;
  region.counts[1] = 1;
  region.counts[2] = 257;
  region.strides[2] = 1;
  const scalar_precision precision =
      sizeof(T) == sizeof(float) ? scalar_precision::f32 : scalar_precision::f64;
  size_t cases = 0;
  for (unsigned int offdiagonals = 0; offdiagonals <= 2; ++offdiagonals)
    for (int drude = 0; drude <= 1; ++drude) {
      std::vector<T> expected = p, expected_prev = p_prev;
      std::vector<T> diagonal = sigma;
      if (offdiagonals) diagonal[44] = T(0);
      copy_host_to_device_async(d_p, 0, p.data(), bytes, execution);
      copy_host_to_device_async(d_p_prev, 0, p_prev.data(), bytes, execution);
      copy_host_to_device_async(d_sigma, 0, diagonal.data(), bytes, execution);

      polarization_update_launch launch = {};
      launch.region = region;
      launch.p = d_p.opaque_handle();
      launch.p_prev = d_p_prev.opaque_handle();
      launch.primary_w = d_w.opaque_handle();
      launch.diagonal_sigma = d_sigma.opaque_handle();
      launch.offdiagonals = offdiagonals;
      launch.drude = drude != 0;
      launch.precision = precision;
      launch.primary_stride = -2;
      launch.cross_stride1 = -5;
      launch.cross_stride2 = -7;
      if (offdiagonals >= 1) {
        launch.cross_w1 = d_w1.opaque_handle();
        launch.offdiagonal_sigma1 = d_sigma1.opaque_handle();
      }
      if (offdiagonals >= 2) {
        launch.cross_w2 = d_w2.opaque_handle();
        launch.offdiagonal_sigma2 = d_sigma2.opaque_handle();
      }
      const double omega = 0.73 + 0.09 * drude;
      const double gamma = drude ? 0.0 : 0.06;
      const double dt = 0.017;
      const double omega2pi = 2 * 3.14159265358979323846 * omega;
      const double gamma2pi = 2 * 3.14159265358979323846 * gamma;
      launch.omega0dtsqr = omega2pi * omega2pi * dt * dt;
      launch.gamma1inv = 1 / (1 + gamma2pi * dt / 2);
      launch.gamma1 = 1 - gamma2pi * dt / 2;
      launch.omega0dtsqr_denom = drude ? 0 : launch.omega0dtsqr;

      for (size_t linear = 0; linear < region.counts[2]; ++linear) {
        const ptrdiff_t i = ptrdiff_t(region.base + linear);
        if (offdiagonals && diagonal[i] == T(0)) continue;
        T forcing = diagonal[i] * w[i];
        if (offdiagonals >= 1)
          forcing += offdiag(sigma1, w1, i, launch.primary_stride, launch.cross_stride1);
        if (offdiagonals >= 2)
          forcing += offdiag(sigma2, w2, i, launch.primary_stride, launch.cross_stride2);
        const T current = expected[i];
        expected[i] = T(launch.gamma1inv) *
                      (current * (T(2) - T(launch.omega0dtsqr_denom)) -
                       T(launch.gamma1) * expected_prev[i] + T(launch.omega0dtsqr) * forcing);
        expected_prev[i] = current;
      }

      launch_polarization_update(launch, execution);
      copy_device_to_host_async(observed.data(), d_p, 0, bytes, execution);
      copy_device_to_host_async(observed_prev.data(), d_p_prev, 0, bytes, execution);
      execution.synchronize();
      const double tolerance = 16 * std::numeric_limits<T>::epsilon();
      for (size_t i = 0; i < elements; ++i) {
        require(std::fabs(double(observed[i] - expected[i])) <=
                    tolerance * (1 + std::fabs(double(expected[i]))),
                "polarization recurrence or sentinel differs");
        require(std::fabs(double(observed_prev[i] - expected_prev[i])) <=
                    tolerance * (1 + std::fabs(double(expected_prev[i]))),
                "polarization P_prev or sentinel differs");
      }
      ++cases;
    }

  std::vector<T> target(elements), p2(elements);
  for (size_t i = 0; i < elements; ++i) {
    target[i] = T(2 + 0.01 * double(i));
    p2[i] = T(0.05 - 0.0002 * double(i));
  }
  copy_host_to_device_async(d_w, 0, target.data(), bytes, execution);
  copy_host_to_device_async(d_p, 0, p.data(), bytes, execution);
  copy_host_to_device_async(d_p_prev, 0, p2.data(), bytes, execution);
  polarization_subtract_launch subtraction = {};
  subtraction.target = d_w.opaque_handle();
  subtraction.p = d_p.opaque_handle();
  subtraction.elements = elements;
  subtraction.precision = precision;
  launch_polarization_subtract(subtraction, execution);
  copy_device_to_host_async(observed.data(), d_w, 0, bytes, execution);
  execution.synchronize();
  for (size_t i = 0; i < elements; ++i)
    require(observed[i] == T(target[i] - p[i]),
            "single full-array polarization subtraction differs");
  subtraction.p = d_p_prev.opaque_handle();
  launch_polarization_subtract(subtraction, execution);
  copy_device_to_host_async(observed.data(), d_w, 0, bytes, execution);
  execution.synchronize();
  for (size_t i = 0; i < elements; ++i)
    require(observed[i] == T(target[i] - p[i] - p2[i]),
            "ordered full-array polarization subtraction differs");

  bool rejected = false;
  polarization_update_launch malformed = {};
  malformed.region = region;
  malformed.p = d_p.opaque_handle();
  malformed.p_prev = d_p_prev.opaque_handle();
  malformed.primary_w = d_w.opaque_handle();
  malformed.diagonal_sigma = d_sigma.opaque_handle();
  malformed.offdiagonals = 2;
  malformed.precision = precision;
  try { launch_polarization_update(malformed, execution); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "malformed second-offdiagonal launch was accepted");
  malformed = {};
  malformed.region = region;
  malformed.p = d_p.opaque_handle();
  malformed.p_prev = d_p_prev.opaque_handle();
  malformed.primary_w = d_w.opaque_handle();
  malformed.diagonal_sigma = d_sigma.opaque_handle();
  malformed.precision = static_cast<scalar_precision>(99);
  rejected = false;
  try { launch_polarization_update(malformed, execution); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "invalid polarization precision was accepted");
  std::cout << "device " << device << ": " << cases << " "
            << (sizeof(T) == sizeof(float) ? "f32" : "f64") << " polarization cases PASS\n";
}

int main(int argc, char **argv) {
  try {
    const bool gyro_only = argc == 2 && std::string(argv[1]) == "--gyro-only";
    const bool noisy_kat_only = argc == 2 && std::string(argv[1]) == "--noisy-kat-only";
    const bool noisy_replay_only = argc == 2 && std::string(argv[1]) == "--noisy-replay-only";
    const bool noisy_stats_only = argc == 2 && std::string(argv[1]) == "--noisy-stats-only";
    if (argc > 2 ||
        (argc == 2 && !gyro_only && !noisy_kat_only && !noisy_replay_only &&
         !noisy_stats_only))
      throw std::invalid_argument("unknown NVIDIA polarization smoke selector");
    const std::vector<device_properties> devices = enumerate_devices();
    require(!devices.empty(), "no CUDA devices found");
    for (size_t i = 0; i < devices.size(); ++i) {
      if (noisy_kat_only) {
        check_noisy_kats(devices[i].id);
        check_noisy_add_device<float>(devices[i].id);
        check_noisy_add_device<double>(devices[i].id);
        check_noisy_recurrence_order<float>(devices[i].id);
        check_noisy_recurrence_order<double>(devices[i].id);
        continue;
      }
      if (noisy_replay_only) {
        check_noisy_replay(devices[i].id);
        check_noisy_key_sensitivity(devices[i].id);
        continue;
      }
      if (noisy_stats_only) {
        check_noisy_statistics(devices[i].id);
        check_noisy_variance_scaling<float>(devices[i].id);
        check_noisy_variance_scaling<double>(devices[i].id);
        continue;
      }
      if (!gyro_only) {
        check_device<float>(devices[i].id);
        check_device<double>(devices[i].id);
      }
      check_gyrotropic_device<float>(devices[i].id);
      check_gyrotropic_device<double>(devices[i].id);
      std::cout << "device " << devices[i].id << " (" << devices[i].uuid
                << "): NVIDIA polarization kernels PASS\n";
    }
  }
  catch (const std::exception &error) {
    std::cerr << "nvidia_polarization_smoke: FAIL: " << error.what() << "\n";
    return 1;
  }
  std::cout << "nvidia_polarization_smoke: PASS\n";
  return 0;
}
