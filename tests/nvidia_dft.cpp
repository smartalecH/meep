/* End-to-end CPU/NVIDIA parity for device-resident DFT monitor families and
   their existing host-side spectral query paths. */

#include <meep.hpp>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <unistd.h>

#include "backend/backend.hpp"
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

static void run_real_monitor_families(precision_policy_kind policy) {
  const grid_volume gv = vol2d(3.0, 3.0, 7.0);
  structure cpu_structure(gv, vacuum, no_pml(), identity(), 1);
  structure gpu_structure(gv, vacuum, no_pml(), identity(), 1);
  fields cpu(&cpu_structure);
  execution_options options;
  options.backend = backend_kind::nvidia;
  options.precision = policy;
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
  if (policy == precision_policy_kind::native) check_malformed_dft_rejections(gpu);
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
      compare_monitor_storage(cpu, gpu, tolerance);
    }
    previous = checkpoints[checkpoint];
  }

  compare_dft_array(cpu, gpu, cpu_fields, gpu_fields, Ez, 0, tolerance);
  compare_dft_array(cpu, gpu, cpu_fields, gpu_fields, Hx, 2, tolerance);
  std::unique_ptr<double[]> expected_flux(cpu_flux.flux()), observed_flux(gpu_flux.flux());
  std::unique_ptr<double[]> expected_energy(cpu_energy.total()),
      observed_energy(gpu_energy.total());
  std::unique_ptr<double[]> expected_force(cpu_force.force()), observed_force(gpu_force.force());
  for (size_t i = 0; i < frequencies.size(); ++i) {
    compare_scalar(expected_flux[i], observed_flux[i], 3 * tolerance, "DFT flux");
    compare_scalar(expected_energy[i], observed_energy[i], 3 * tolerance, "DFT energy");
    compare_scalar(expected_force[i], observed_force[i], 3 * tolerance, "DFT force");
  }
  compare_scalar(cpu.dft_norm(), gpu.dft_norm(), 3 * tolerance, "DFT norm");
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
  compare_hdf5_field_access(cpu, gpu, access_region, tolerance);

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
  }
  master_printf("nvidia_dft: PASS\n");
  return 0;
}
