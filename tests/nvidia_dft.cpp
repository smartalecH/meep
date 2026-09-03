/* End-to-end CPU/NVIDIA parity for device-resident DFT monitor families and
   their existing host-side spectral query paths. */

#include "config.h"

#include <meep.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
#include <unistd.h>

#include "backend/backend.hpp"
#include "backend/nvidia/runtime.hpp"
#include "backend/storage_plan.hpp"
#include "meep_internals.hpp"

using namespace meep;

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    meep::abort("nvidia_dft failed");
  }
}

static double vacuum(const vec &) { return 1.0; }

static void compare_scalar(double expected, double observed, double tolerance, const char *what) {
  const double error = std::fabs(observed - expected);
  if (error > tolerance * (1.0 + std::fabs(expected))) {
    std::fprintf(stderr, "%s differs: cpu=%.17g nvidia=%.17g error=%.3g tolerance=%.3g\n", what,
                 expected, observed, error, tolerance * (1.0 + std::fabs(expected)));
    meep::abort("NVIDIA DFT query differs from CPU");
  }
}

static void compare_complex_value(std::complex<double> expected, std::complex<double> observed,
                                  double tolerance, const char *what) {
  const double error = std::abs(observed - expected);
  if (error > tolerance * (1.0 + std::abs(expected))) {
    std::fprintf(stderr,
                 "%s differs: cpu=(%.17g,%.17g) nvidia=(%.17g,%.17g) error=%.3g "
                 "tolerance=%.3g\n",
                 what, expected.real(), expected.imag(), observed.real(), observed.imag(), error,
                 tolerance * (1.0 + std::abs(expected)));
    meep::abort("NVIDIA DFT query differs from CPU");
  }
}

static std::complex<double> access_integrand(const std::complex<realnum> *values, const vec &,
                                             void *) {
  return std::norm(values[0]) + 0.25 * values[1];
}

static void poison_host_fields(fields &gpu) {
  require(gpu.array_catalog != NULL, "NVIDIA access test has no storage catalog");
  const realnum poison = std::numeric_limits<realnum>::quiet_NaN();
  for (size_t i = 0; i < gpu.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = gpu.array_catalog->spec(id);
    if (spec.role != array_role::field || spec.element_type != ElementType::realnum_value) continue;
    realnum *host = gpu.array_catalog->resolve<realnum>(id);
    for (size_t j = 0; j < spec.elements; ++j)
      host[j] = poison;
  }
}

static void poison_host_dft(fields &gpu) {
  require(gpu.array_catalog != NULL, "NVIDIA DFT poison test has no storage catalog");
  const realnum poison = std::numeric_limits<realnum>::quiet_NaN();
  for (size_t i = 0; i < gpu.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = gpu.array_catalog->spec(id);
    if (spec.role != array_role::dft || spec.element_type != ElementType::complex_realnum) continue;
    std::complex<realnum> *host = gpu.array_catalog->resolve<std::complex<realnum> >(id);
    for (size_t j = 0; j < spec.elements; ++j)
      host[j] = std::complex<realnum>(poison, poison);
  }
}

static void require_host_dft_poisoned(fields &gpu) {
  bool saw_poison = false;
  for (size_t i = 0; i < gpu.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = gpu.array_catalog->spec(id);
    if (spec.role != array_role::dft || spec.element_type != ElementType::complex_realnum) continue;
    const std::complex<realnum> *host =
        gpu.array_catalog->resolve<std::complex<realnum> >(id);
    for (size_t j = 0; j < spec.elements; ++j) {
      saw_poison = saw_poison || (std::isnan(host[j].real()) && std::isnan(host[j].imag()));
      if (!std::isnan(host[j].real()) || !std::isnan(host[j].imag()))
        meep::abort("compact DFT query refreshed a poisoned host accumulator");
    }
  }
  require(saw_poison, "DFT poison check found no monitor storage");
}

static void require_compact_transfer(size_t result_count, size_t expected_calls = 1) {
  const nvidia::testing::transfer_accounting transfers =
      nvidia::testing::current_transfer_accounting();
  require(transfers.host_to_device_calls == 0 && transfers.host_to_device_bytes == 0,
          "compact DFT query performed a host-to-device transfer");
  require(transfers.device_to_host_calls == expected_calls,
          "compact DFT query used the wrong D2H call count");
  require(transfers.device_to_host_bytes == result_count * sizeof(std::complex<double>),
          "compact DFT query transferred the wrong byte count");
}

#ifdef HAVE_HDF5
static void compare_hdf5_field_access(fields &cpu, fields &gpu, const volume &where,
                                      double tolerance) {
  static unsigned int invocation = 0;
  char cpu_filename[128], gpu_filename[128];
  snprintf(cpu_filename, sizeof(cpu_filename), "/tmp/meep-nvidia-access-%ld-%u-cpu.h5",
           static_cast<long>(getpid()), invocation);
  snprintf(gpu_filename, sizeof(gpu_filename), "/tmp/meep-nvidia-access-%ld-%u-gpu.h5",
           static_cast<long>(getpid()), invocation);
  ++invocation;

  h5file *cpu_file = new h5file(cpu_filename, h5file::WRITE, true);
  h5file *gpu_file = new h5file(gpu_filename, h5file::WRITE, true);
  cpu.output_hdf5(Ez, where, cpu_file);
  poison_host_fields(gpu);
  gpu.output_hdf5(Ez, where, gpu_file);
  delete cpu_file;
  delete gpu_file;

  cpu_file = new h5file(cpu_filename, h5file::READONLY, true);
  gpu_file = new h5file(gpu_filename, h5file::READONLY, true);
  int cpu_rank = 0, gpu_rank = 0;
  size_t cpu_dims[3] = {0, 0, 0}, gpu_dims[3] = {0, 0, 0};
  std::unique_ptr<double[]> expected(
      static_cast<double *>(cpu_file->read("ez", &cpu_rank, cpu_dims, 3, false)));
  std::unique_ptr<double[]> observed(
      static_cast<double *>(gpu_file->read("ez", &gpu_rank, gpu_dims, 3, false)));
  require(expected.get() && observed.get(), "ordinary HDF5 field readback failed");
  require(cpu_rank == gpu_rank, "ordinary HDF5 field ranks differ");
  size_t elements = 1;
  for (int axis = 0; axis < cpu_rank; ++axis) {
    require(cpu_dims[axis] == gpu_dims[axis], "ordinary HDF5 field dimensions differ");
    elements *= cpu_dims[axis];
  }
  for (size_t i = 0; i < elements; ++i)
    compare_scalar(expected[i], observed[i], tolerance, "ordinary HDF5 field");
  cpu_file->remove();
  gpu_file->remove();
  delete cpu_file;
  delete gpu_file;
}

static void compare_hdf5_dataset(const char *cpu_filename, const char *gpu_filename,
                                 const char *dataset, double tolerance) {
  h5file cpu_file(cpu_filename, h5file::READONLY, false);
  h5file gpu_file(gpu_filename, h5file::READONLY, false);
  int cpu_rank = 0, gpu_rank = 0;
  size_t cpu_dims[3] = {0, 0, 0}, gpu_dims[3] = {0, 0, 0};
  std::unique_ptr<double[]> expected(
      static_cast<double *>(cpu_file.read(dataset, &cpu_rank, cpu_dims, 3, false)));
  std::unique_ptr<double[]> observed(
      static_cast<double *>(gpu_file.read(dataset, &gpu_rank, gpu_dims, 3, false)));
  require(expected.get() && observed.get(), "DFT HDF5 dataset readback failed");
  require(cpu_rank == gpu_rank, "CPU and NVIDIA DFT HDF5 ranks differ");
  size_t elements = 1;
  for (int axis = 0; axis < cpu_rank; ++axis) {
    require(cpu_dims[axis] == gpu_dims[axis], "CPU and NVIDIA DFT HDF5 dimensions differ");
    elements *= cpu_dims[axis];
  }
  for (size_t i = 0; i < elements; ++i)
    compare_scalar(expected[i], observed[i], tolerance, "DFT HDF5 dataset");
}

static std::string temporary_stem(const char *kind, precision_policy_kind policy) {
  char name[192];
  snprintf(name, sizeof(name), "meep-nvidia-dft-%ld-%s-%s", static_cast<long>(getpid()),
           precision_policy_name(policy), kind);
  return std::string(name);
}

static void remove_h5(const std::string &stem) { std::remove((stem + ".h5").c_str()); }
#endif

static void compare_monitor_storage(fields &cpu, fields &gpu, double tolerance) {
  require(cpu.array_catalog && gpu.array_catalog, "DFT comparison has no storage catalog");
  require(cpu.array_catalog->size() == gpu.array_catalog->size(),
          "CPU and NVIDIA DFT catalogs differ in size");
  for (size_t i = 0; i < cpu.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = cpu.array_catalog->spec(id);
    const StorageKey &key = cpu.array_catalog->key(id);
    if (key.kind != int(array_kind::dft) && key.kind != int(array_kind::dft_phase)) continue;
    require(spec.element_type == ElementType::complex_realnum,
            "DFT storage has the wrong host element type");
    require(key == gpu.array_catalog->key(id), "CPU and NVIDIA DFT storage keys differ");
    const std::complex<realnum> *expected = cpu.array_catalog->resolve<std::complex<realnum> >(id);
    std::vector<std::complex<realnum> > observed(spec.elements);
    gpu.backend->read(ArrayRef{id, 0, spec.elements}, observed.data(),
                      observed.size() * sizeof(observed[0]));
    for (size_t j = 0; j < observed.size(); ++j)
      compare_complex_value(expected[j], observed[j], tolerance, "DFT storage");
  }
}

static void compare_dft_array(fields &cpu, fields &gpu, dft_fields cpu_monitor,
                              dft_fields gpu_monitor, component c, int frequency,
                              double tolerance) {
  int cpu_rank = 0, gpu_rank = 0;
  size_t cpu_dims[3] = {0, 0, 0}, gpu_dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> expected(
      cpu.get_dft_array(cpu_monitor, c, frequency, &cpu_rank, cpu_dims));
  std::unique_ptr<std::complex<realnum>[]> observed(
      gpu.get_dft_array(gpu_monitor, c, frequency, &gpu_rank, gpu_dims));
  require(cpu_rank == gpu_rank, "CPU and NVIDIA DFT array ranks differ");
  size_t elements = 1;
  for (int axis = 0; axis < cpu_rank; ++axis) {
    require(cpu_dims[axis] == gpu_dims[axis], "CPU and NVIDIA DFT array dimensions differ");
    elements *= cpu_dims[axis];
  }
  for (size_t i = 0; i < elements; ++i)
    compare_complex_value(expected[i], observed[i], tolerance, "DFT field array");
}

static void expect_dft_compile_rejected(fields &gpu, StepPlan plan,
                                        const char *expected_reason) {
  plan.signature = compute_step_plan_signature(plan);
  Executable *unexpected = NULL;
  bool rejected = false;
  try { unexpected = gpu.backend->compile(plan, *gpu.backend_state); }
  catch (const std::runtime_error &error) {
    rejected = std::strstr(error.what(), expected_reason) != NULL;
    if (!rejected)
      std::fprintf(stderr, "unexpected DFT rejection: %s (wanted: %s)\n", error.what(),
                   expected_reason);
  }
  delete unexpected;
  require(rejected, "malformed DFT descriptor was not rejected as expected");
}

static void check_malformed_dft_rejections(fields &gpu) {
  const StepPlan baseline = build_step_plan(gpu, StepProgram::ordinary);
  require(!baseline.dft_updates.empty(), "DFT rejection test has no descriptor");

  StepPlan malformed = baseline;
  malformed.dft_updates[0].decimation_factor = 0;
  expect_dft_compile_rejected(gpu, malformed, "nonpositive decimation factor");

  malformed = baseline;
  ++malformed.dft_updates[0].N;
  expect_dft_compile_rejected(gpu, malformed, "region size does not match N");

  malformed = baseline;
  size_t wrong_component = 1;
  while (wrong_component < malformed.dft_updates.size() &&
         malformed.dft_updates[wrong_component].c == malformed.dft_updates[0].c)
    ++wrong_component;
  require(wrong_component < malformed.dft_updates.size(),
          "DFT rejection test has no differently identified source field");
  malformed.dft_updates[0].source_field = malformed.dft_updates[wrong_component].source_field;
  expect_dft_compile_rejected(gpu, malformed, "real source has the wrong storage identity");

  malformed = baseline;
  malformed.dft_updates[0].avg1 = ptrdiff_t(malformed.dft_updates[0].source_field.elements);
  malformed.dft_updates[0].avg2 = 0;
  expect_dft_compile_rejected(gpu, malformed, "DFT real source index range exceeds");
}

static void expect_reduction_rejected(fields &gpu, const DftReductionRequest &request,
                                      size_t result_count, const char *message) {
  std::vector<std::complex<double> > result(result_count ? result_count : 1);
  bool rejected = false;
  try { gpu.backend->reduce_dft(request, result.data(), result_count); }
  catch (const std::exception &) { rejected = true; }
  require(rejected, message);
}

static void check_malformed_dft_reduction_rejections(fields &gpu, size_t frequencies) {
  ArrayId dft_id = invalid_array(), field_id = invalid_array();
  for (size_t i = 0; i < gpu.array_catalog->size(); ++i) {
    const ArrayId id{uint32_t(i)};
    const ArraySpec &spec = gpu.array_catalog->spec(id);
    if (!is_valid(dft_id) && spec.role == array_role::dft &&
        spec.element_type == ElementType::complex_realnum &&
        gpu.array_catalog->key(id).kind == int(array_kind::dft))
      dft_id = id;
    if (!is_valid(field_id) && spec.role == array_role::field) field_id = id;
  }
  require(is_valid(dft_id) && is_valid(field_id), "reduction rejection test lacks arrays");
  const ArraySpec &spec = gpu.array_catalog->spec(dft_id);
  require(spec.elements % frequencies == 0, "DFT test array has an unexpected shape");
  DftReductionTerm term;
  term.left = dft_id;
  term.right = dft_id;
  term.storage_points = spec.elements / frequencies;
  term.frequencies = frequencies;
  term.region = DftReductionRegion{0, {term.storage_points, 1, 1}, {1, 0, 0}};
  term.weight = 1.0;
  DftReductionRequest request;
  request.kind = DftReductionKind::real_weighted_product;
  request.accumulation_precision = policy_for(gpu.options.precision).reduction;
  request.result_count = frequencies;
  request.terms.push_back(term);

  DftReductionRequest empty;
  empty.kind = DftReductionKind::norm2;
  empty.accumulation_precision = policy_for(gpu.options.precision).reduction;
  empty.result_count = 1;
  std::complex<double> empty_result(1.0, 1.0);
  gpu.backend->reduce_dft(empty, &empty_result, 1);
  require(empty_result == std::complex<double>(0.0, 0.0),
          "empty compact reduction did not return zero");

  DftReductionRequest malformed = request;
  malformed.terms[0].left = invalid_array();
  expect_reduction_rejected(gpu, malformed, frequencies, "invalid reduction ArrayId was accepted");
  malformed = request;
  malformed.terms[0].left = field_id;
  expect_reduction_rejected(gpu, malformed, frequencies, "non-DFT reduction array was accepted");
  malformed = request;
  ++malformed.terms[0].storage_points;
  expect_reduction_rejected(gpu, malformed, frequencies, "bad reduction array shape was accepted");
  malformed = request;
  ++malformed.terms[0].frequencies;
  expect_reduction_rejected(gpu, malformed, frequencies, "bad reduction frequency was accepted");
  malformed = request;
  ++malformed.result_count;
  expect_reduction_rejected(gpu, malformed, frequencies, "bad reduction result count was accepted");
}

static void run_ldos_boundary(precision_policy_kind policy) {
  const grid_volume gv = vol2d(2.0, 2.0, 8.0);
  structure cpu_structure(gv, vacuum, no_pml(), identity(), 1);
  structure gpu_structure(gv, vacuum, no_pml(), identity(), 1);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();
  gaussian_src_time cpu_source(0.31, 0.20), gpu_source(0.31, 0.20);
  cpu.add_point_source(Ez, cpu_source, vec(0.37, 0.41), 0.9);
  gpu.add_point_source(Ez, gpu_source, vec(0.37, 0.41), 0.9);
  size_t source_points_before_init = 0;
  for (int ic = 0; ic < gpu.num_chunks; ++ic)
    for (const src_vol &sv : gpu.chunks[ic]->get_sources(D_stuff))
      source_points_before_init += sv.num_points();
  require(source_points_before_init > 0, "LDOS fixture source did not intersect the grid");
  cpu.init_backend();
  gpu.init_backend();
  cpu.advance(4);
  gpu.advance(4);

  const double frequency = 0.31;
  dft_ldos cpu_ldos(&frequency, 1);
  dft_ldos gpu_ldos(&frequency, 1);
  cpu_ldos.update(cpu);
  poison_host_fields(gpu);
  nvidia::testing::reset_transfer_accounting();
  gpu_ldos.update(gpu);
  const nvidia::testing::transfer_accounting transfers =
      nvidia::testing::current_transfer_accounting();
  require(transfers.device_to_host_calls > 0 && transfers.device_to_host_bytes > 0,
          "LDOS update did not read resident source-point fields");
  size_t largest_field_bytes = 0;
  for (size_t i = 0; i < gpu.array_catalog->size(); ++i) {
    const ArraySpec &spec = gpu.array_catalog->spec(ArrayId{uint32_t(i)});
    if (spec.role == array_role::field)
      largest_field_bytes = std::max(largest_field_bytes,
                                     spec.elements * host_element_bytes(spec.element_type));
  }
  require(transfers.device_to_host_bytes < largest_field_bytes,
          "LDOS update copied a complete field allocation");
  std::unique_ptr<std::complex<double>[]> expected_f(cpu_ldos.F());
  std::unique_ptr<std::complex<double>[]> observed_f(gpu_ldos.F());
  std::unique_ptr<std::complex<double>[]> expected_j(cpu_ldos.J());
  std::unique_ptr<std::complex<double>[]> observed_j(gpu_ldos.J());
  const double tolerance = policy == precision_policy_kind::native ? 5e-12 : 3e-4;
  compare_complex_value(expected_f[0], observed_f[0], 3 * tolerance, "LDOS F");
  compare_complex_value(expected_j[0], observed_j[0], 3 * tolerance, "LDOS J");
}

static void run_real_monitor_families(precision_policy_kind policy) {
  const grid_volume gv = vol2d(3.0, 3.0, 7.0);
  structure cpu_structure(gv, vacuum, no_pml(), identity(), 1);
  structure gpu_structure(gv, vacuum, no_pml(), identity(), 1);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&gpu_structure, options);
  cpu.use_real_fields();
  gpu.use_real_fields();

  gaussian_src_time cpu_source(0.37, 0.22), gpu_source(0.37, 0.22);
  cpu.add_point_source(Ez, cpu_source, vec(0.13, -0.21), 0.8);
  gpu.add_point_source(Ez, gpu_source, vec(0.13, -0.21), 0.8);
  const std::vector<double> frequencies = {0.23, 0.37, 0.51};
  component components[] = {Ez, Hx, Hy};
  const volume field_region(vec(-0.9, -0.7), vec(0.8, 0.9));
  dft_fields cpu_fields =
      cpu.add_dft_fields(components, 3, field_region, frequencies, true, 2, true);
  dft_fields gpu_fields =
      gpu.add_dft_fields(components, 3, field_region, frequencies, true, 2, true);
  dft_fields cpu_subtract =
      cpu.add_dft_fields(components, 3, field_region, frequencies, true, 2, true);
  dft_fields gpu_subtract =
      gpu.add_dft_fields(components, 3, field_region, frequencies, true, 2, true);

  volume_list cpu_flux_surface(volume(vec(0.55, -0.8), vec(0.55, 0.8)), Sx);
  volume_list gpu_flux_surface(volume(vec(0.55, -0.8), vec(0.55, 0.8)), Sx);
  dft_flux cpu_flux = cpu.add_dft_flux(&cpu_flux_surface, frequencies, true, true, 3);
  dft_flux gpu_flux = gpu.add_dft_flux(&gpu_flux_surface, frequencies, true, true, 3);

  volume_list cpu_energy_region(volume(vec(-0.75, -0.65), vec(0.7, 0.75)), Ex);
  volume_list gpu_energy_region(volume(vec(-0.75, -0.65), vec(0.7, 0.75)), Ex);
  dft_energy cpu_energy = cpu.add_dft_energy(&cpu_energy_region, frequencies, 2);
  dft_energy gpu_energy = gpu.add_dft_energy(&gpu_energy_region, frequencies, 2);

  volume_list cpu_force_surface(volume(vec(0.45, -0.75), vec(0.45, 0.75)), Sx);
  volume_list gpu_force_surface(volume(vec(0.45, -0.75), vec(0.45, 0.75)), Sx);
  dft_force cpu_force = cpu.add_dft_force(&cpu_force_surface, frequencies, 3);
  dft_force gpu_force = gpu.add_dft_force(&gpu_force_surface, frequencies, 3);

  volume_list cpu_near(
      volume(vec(-0.8, 0.8), vec(0.8, 0.8)), Sy, 1.0,
      new volume_list(
          volume(vec(0.8, 0.8), vec(0.8, -0.8)), Sx, 1.0,
          new volume_list(volume(vec(-0.8, -0.8), vec(0.8, -0.8)), Sy, -1.0,
                          new volume_list(volume(vec(-0.8, -0.8), vec(-0.8, 0.8)), Sx, -1.0))));
  volume_list gpu_near(
      volume(vec(-0.8, 0.8), vec(0.8, 0.8)), Sy, 1.0,
      new volume_list(
          volume(vec(0.8, 0.8), vec(0.8, -0.8)), Sx, 1.0,
          new volume_list(volume(vec(-0.8, -0.8), vec(0.8, -0.8)), Sy, -1.0,
                          new volume_list(volume(vec(-0.8, -0.8), vec(-0.8, 0.8)), Sx, -1.0))));
  dft_near2far cpu_n2f = cpu.add_dft_near2far(&cpu_near, frequencies, 2);
  dft_near2far gpu_n2f = gpu.add_dft_near2far(&gpu_near, frequencies, 2);

  cpu.init_backend();
  gpu.init_backend();
  if (policy == precision_policy_kind::native) {
    check_malformed_dft_rejections(gpu);
    check_malformed_dft_reduction_rejections(gpu, frequencies.size());
  }
  const double tolerance = policy == precision_policy_kind::native ? 5e-12 : 3e-4;
  const int checkpoints[] = {1, 2, 6, 12};
  int previous = 0;
  for (size_t checkpoint = 0; checkpoint < 4; ++checkpoint) {
    const int delta = checkpoints[checkpoint] - previous;
    cpu.advance(delta);
    gpu.advance(delta);
    compare_monitor_storage(cpu, gpu, tolerance);
    if (checkpoints[checkpoint] == 6) {
      const std::complex<double> factor(0.73, -0.19);
      cpu_fields.scale_dfts(factor);
      gpu_fields.scale_dfts(factor);
      cpu_subtract.scale_dfts(factor);
      gpu_subtract.scale_dfts(factor);
      compare_monitor_storage(cpu, gpu, tolerance);
    }
    previous = checkpoints[checkpoint];
  }

  compare_dft_array(cpu, gpu, cpu_fields, gpu_fields, Ez, 0, tolerance);
  compare_dft_array(cpu, gpu, cpu_fields, gpu_fields, Hx, 2, tolerance);
  std::unique_ptr<double[]> expected_flux(cpu_flux.flux());
  const std::vector<std::complex<double> > expected_complex_flux = cpu_flux.complexflux();
  std::unique_ptr<double[]> expected_energy(cpu_energy.total());
  std::unique_ptr<double[]> expected_force(cpu_force.force());

  poison_host_dft(gpu);
  nvidia::testing::reset_transfer_accounting();
  std::unique_ptr<double[]> observed_flux(gpu_flux.flux());
  require_compact_transfer(frequencies.size());
  require_host_dft_poisoned(gpu);
  const nvidia::memory_accounting memory_before_repeat = nvidia::current_memory_accounting();
  nvidia::testing::reset_transfer_accounting();
  std::unique_ptr<double[]> repeated_flux(gpu_flux.flux());
  require_compact_transfer(frequencies.size());
  require_host_dft_poisoned(gpu);
  const nvidia::memory_accounting memory_after_repeat = nvidia::current_memory_accounting();
  require(memory_after_repeat.device_bytes_current == memory_before_repeat.device_bytes_current &&
              memory_after_repeat.pinned_bytes_current == memory_before_repeat.pinned_bytes_current,
          "repeated compact DFT query allocated new scratch storage");

  nvidia::testing::reset_transfer_accounting();
  const std::vector<std::complex<double> > observed_complex_flux = gpu_flux.complexflux();
  require_compact_transfer(frequencies.size());
  require_host_dft_poisoned(gpu);

  nvidia::testing::reset_transfer_accounting();
  std::unique_ptr<double[]> observed_energy(gpu_energy.total());
  require_compact_transfer(2 * frequencies.size(), 2);
  require_host_dft_poisoned(gpu);

  nvidia::testing::reset_transfer_accounting();
  std::unique_ptr<double[]> observed_force(gpu_force.force());
  require_compact_transfer(frequencies.size());
  require_host_dft_poisoned(gpu);

  for (size_t i = 0; i < frequencies.size(); ++i) {
    compare_scalar(expected_flux[i], observed_flux[i], 3 * tolerance, "DFT flux");
    compare_scalar(observed_flux[i], repeated_flux[i], 5e-13, "repeated DFT flux");
    compare_complex_value(expected_complex_flux[i], observed_complex_flux[i], 3 * tolerance,
                          "complex DFT flux");
    compare_scalar(expected_energy[i], observed_energy[i], 3 * tolerance, "DFT energy");
    compare_scalar(expected_force[i], observed_force[i], 3 * tolerance, "DFT force");
  }
  const double expected_norm = cpu.dft_norm();
  nvidia::testing::reset_transfer_accounting();
  const double observed_norm = gpu.dft_norm();
  require_compact_transfer(1);
  require_host_dft_poisoned(gpu);
  compare_scalar(expected_norm, observed_norm, 3 * tolerance, "DFT norm");

  const std::complex<double> query_scale(0.61, 0.17);
  cpu_flux.scale_dfts(query_scale);
  gpu_flux.scale_dfts(query_scale);
  std::unique_ptr<double[]> expected_scaled_flux(cpu_flux.flux());
  poison_host_dft(gpu);
  nvidia::testing::reset_transfer_accounting();
  std::unique_ptr<double[]> observed_scaled_flux(gpu_flux.flux());
  require_compact_transfer(frequencies.size());
  require_host_dft_poisoned(gpu);
  for (size_t i = 0; i < frequencies.size(); ++i) {
    compare_scalar(expected_scaled_flux[i], observed_scaled_flux[i], 3 * tolerance,
                   "post-scale DFT flux");
    compare_scalar(expected_flux[i] * std::norm(query_scale), expected_scaled_flux[i],
                   3 * tolerance, "DFT flux scale relation");
  }
  const size_t farfield_values = 6 * frequencies.size();
  std::vector<std::complex<double> > expected_far(farfield_values),
      observed_far(farfield_values);
  cpu_n2f.farfield_lowlevel(expected_far.data(), vec(2.1, 1.7));
  gpu_n2f.farfield_lowlevel(observed_far.data(), vec(2.1, 1.7));
  for (size_t i = 0; i < farfield_values; ++i)
    compare_complex_value(expected_far[i], observed_far[i], 5 * tolerance, "near2far field");

  const volume access_region(vec(-0.7, -0.6), vec(0.8, 0.7));
  component access_components[2] = {Ez, Hx};
  poison_host_fields(gpu);
  compare_scalar(cpu.max_abs(Ez, access_region), gpu.max_abs(Ez, access_region), tolerance,
                 "max_abs resident access");
  poison_host_fields(gpu);
  compare_complex_value(cpu.integrate(2, access_components, access_integrand, NULL, access_region),
                        gpu.integrate(2, access_components, access_integrand, NULL, access_region),
                        3 * tolerance, "integrate resident access");
#ifdef HAVE_HDF5
  compare_hdf5_field_access(cpu, gpu, access_region, tolerance);

  const std::string flux_stem = temporary_stem("flux", policy);
  const std::string energy_stem = temporary_stem("energy", policy);
  const std::string force_stem = temporary_stem("force", policy);
  const std::string n2f_stem = temporary_stem("n2f", policy);
  poison_host_dft(gpu);
  gpu_flux.save_hdf5(gpu, flux_stem.c_str());
  poison_host_dft(gpu);
  gpu_energy.save_hdf5(gpu, energy_stem.c_str());
  poison_host_dft(gpu);
  gpu_force.save_hdf5(gpu, force_stem.c_str());
  poison_host_dft(gpu);
  gpu_n2f.save_hdf5(gpu, n2f_stem.c_str());

  gpu_flux.scale_dfts(0.0);
  gpu_energy.scale_dfts(0.0);
  gpu_force.scale_dfts(0.0);
  gpu_n2f.scale_dfts(0.0);
  gpu_flux.load_hdf5(gpu, flux_stem.c_str());
  gpu_energy.load_hdf5(gpu, energy_stem.c_str());
  gpu_force.load_hdf5(gpu, force_stem.c_str());
  gpu_n2f.load_hdf5(gpu, n2f_stem.c_str());
  std::unique_ptr<double[]> restored_flux(gpu_flux.flux());
  std::unique_ptr<double[]> restored_energy(gpu_energy.total());
  std::unique_ptr<double[]> restored_force(gpu_force.force());
  for (size_t i = 0; i < frequencies.size(); ++i) {
    compare_scalar(expected_scaled_flux[i], restored_flux[i], 3 * tolerance, "restored flux");
    compare_scalar(expected_energy[i], restored_energy[i], 3 * tolerance, "restored energy");
    compare_scalar(expected_force[i], restored_force[i], 3 * tolerance, "restored force");
  }
  gpu_n2f.farfield_lowlevel(observed_far.data(), vec(2.1, 1.7));
  for (size_t i = 0; i < farfield_values; ++i)
    compare_complex_value(expected_far[i], observed_far[i], 5 * tolerance,
                          "restored near2far field");

  const std::string cpu_output_stem = temporary_stem("output-cpu", policy);
  const std::string gpu_output_stem = temporary_stem("output-gpu", policy);
  cpu.output_dft(cpu_fields, cpu_output_stem.c_str());
  poison_host_dft(gpu);
  gpu.output_dft(gpu_fields, gpu_output_stem.c_str());
  compare_hdf5_dataset((cpu_output_stem + ".h5").c_str(), (gpu_output_stem + ".h5").c_str(),
                       "ez_0.r", 3 * tolerance);
  compare_hdf5_dataset((cpu_output_stem + ".h5").c_str(), (gpu_output_stem + ".h5").c_str(),
                       "ez_0.i", 3 * tolerance);

  remove_h5(flux_stem);
  remove_h5(energy_stem);
  remove_h5(force_stem);
  remove_h5(n2f_stem);
  remove_h5(cpu_output_stem);
  remove_h5(gpu_output_stem);
#endif

  *cpu_fields.chunks -= *cpu_subtract.chunks;
  *gpu_fields.chunks -= *gpu_subtract.chunks;
  compare_dft_array(cpu, gpu, cpu_fields, gpu_fields, Ez, 0, tolerance);
  cpu.advance(2);
  gpu.advance(2);
  compare_monitor_storage(cpu, gpu, 3 * tolerance);

  const int time_before_transition = gpu.t;
  cpu.synchronize_magnetic_fields();
  gpu.synchronize_magnetic_fields();
  require(cpu.t == time_before_transition && gpu.t == time_before_transition,
          "magnetic synchronization changed the timestep");
  compare_monitor_storage(cpu, gpu, 3 * tolerance);
  cpu.restore_magnetic_fields();
  gpu.restore_magnetic_fields();
  require(cpu.t == time_before_transition && gpu.t == time_before_transition,
          "magnetic restoration changed the timestep");
  compare_monitor_storage(cpu, gpu, 3 * tolerance);
  cpu.advance(1);
  gpu.advance(1);
  require(cpu.t == time_before_transition + 1 && gpu.t == time_before_transition + 1,
          "NVIDIA execution did not recover after magnetic synchronization and restore");
  compare_monitor_storage(cpu, gpu, 3 * tolerance);
  compare_dft_array(cpu, gpu, cpu_fields, gpu_fields, Ez, 0, tolerance);

  master_printf("nvidia_dft: monitor-families/%s PASS\n", precision_policy_name(policy));
}

static void run_complex_dft_fields(precision_policy_kind policy) {
  const grid_volume gv = vol2d(2.0, 2.0, 7.0);
  structure cpu_structure(gv, vacuum, no_pml(), identity(), 1);
  structure gpu_structure(gv, vacuum, no_pml(), identity(), 1);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
  options.strict = false;
  options.fallback = fallback_policy::warn;
  fields gpu(&gpu_structure, options);
  gaussian_src_time cpu_source(0.29, 0.18), gpu_source(0.29, 0.18);
  cpu.add_point_source(Ez, cpu_source, vec(0.17, -0.11), std::complex<double>(0.7, -0.31));
  gpu.add_point_source(Ez, gpu_source, vec(0.17, -0.11), std::complex<double>(0.7, -0.31));
  component component_list[] = {Ez};
  const std::vector<double> frequencies = {0.21, 0.29};
  dft_fields cpu_fields = cpu.add_dft_fields(
      component_list, 1, volume(vec(-0.7, -0.6), vec(0.8, 0.7)), frequencies, true, 1);
  dft_fields gpu_fields = gpu.add_dft_fields(
      component_list, 1, volume(vec(-0.7, -0.6), vec(0.8, 0.7)), frequencies, true, 1);
  cpu.init_backend();
  gpu.init_backend();
  cpu.advance(9);
  gpu.advance(9);
  const double tolerance = policy == precision_policy_kind::native ? 5e-12 : 3e-4;
  compare_monitor_storage(cpu, gpu, tolerance);
  compare_dft_array(cpu, gpu, cpu_fields, gpu_fields, Ez, 1, tolerance);
  const volume access_region(vec(-0.7, -0.6), vec(0.8, 0.7));
  poison_host_fields(gpu);
  compare_scalar(cpu.max_abs(Ez, access_region), gpu.max_abs(Ez, access_region), tolerance,
                 "complex max_abs resident access");
  master_printf("nvidia_dft: complex-fields/%s PASS\n", precision_policy_name(policy));
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  require(count_processors() == 1, "nvidia_dft is a single-rank test");
  const precision_policy_kind policies[] = {
      precision_policy_kind::native, precision_policy_kind::mixed, precision_policy_kind::f32};
  for (size_t i = 0; i < 3; ++i) {
    run_real_monitor_families(policies[i]);
    run_complex_dft_fields(policies[i]);
    run_ldos_boundary(policies[i]);
  }
  master_printf("nvidia_dft: PASS\n");
  return 0;
}
