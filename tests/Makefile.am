SRC = aniso_disp.cpp bench.cpp bragg_transmission.cpp			\
convergence_cyl_waveguide.cpp cylindrical.cpp dump_load.cpp flux.cpp    \
harmonics.cpp integrate.cpp known_results.cpp near2far.cpp              \
one_dimensional.cpp physical.cpp stress_tensor.cpp symmetry.cpp 	\
three_d.cpp two_dimensional.cpp 2D_convergence.cpp h5test.cpp pml.cpp \
lifecycle.cpp random-state.cpp halo_plan.cpp storage_plan.cpp classification.cpp step_plan.cpp graph_plan.cpp region_plan.cpp transport_plan.cpp dependency_region.cpp mpi_context.cpp mpi_context_finalized.cpp device_uuid_reservation.cpp descriptors.cpp backend_api.cpp

# The bitwise-neutrality harness (plan section 4) is Python, and is driven by
# CI rather than by `make check`: it needs two builds to compare.
EXTRA_DIST = $(SRC) bitwise frontend-batching.ctl nvidia_runtime_smoke.cu nvidia_graph_smoke.cu nvidia_arena_smoke.cu \
nvidia_dft_smoke.cu nvidia_dft_reduction_smoke.cu nvidia_flux_smoke.cu nvidia_cw_smoke.cu nvidia_mpi_runtime_smoke.cu nvidia_step_smoke.cu nvidia_polarization_smoke.cu nvidia_backend_state.cpp \
nvidia_device_reservation.cpp nvidia_multilevel_smoke.cu nvidia_adjoint_smoke.cu nvidia_timestep.cpp nvidia_dft.cpp nvidia_cw.cpp nvidia_staged_transport.cpp

OBJ = $(SRC:.cpp=.o)
DAC = $(SRC:.cpp=.dac)
DONE = $(SRC:.cpp=.done)

LIBMEEP = $(top_builddir)/src/libmeep.la
MEEPLIBS = $(LIBMEEP)
AM_CPPFLAGS = -I$(top_srcdir)/src -I$(top_builddir)/src -DDATADIR=\"$(srcdir)/\"

.SUFFIXES = .dac .done

check_PROGRAMS = lifecycle random-state-lazy-default random-state-initial-restore random-state-initial-set random-state-transitions halo_plan storage_plan classification step_plan graph_plan prepared_plan region_plan transport_plan dependency_region mpi_context mpi_context_finalized device_uuid_reservation descriptors backend_api aniso_disp bench bragg_transmission convergence_cyl_waveguide cylindrical dump_load flux harmonics integrate known_results near2far one_dimensional physical stress_tensor symmetry three_d two_dimensional 2D_convergence h5test pml pw-source-ll ring-ll cyl-ellipsoid-ll absorber-1d-ll array-slice-ll user-defined-material dft-fields bend-flux-ll array-metadata

array_metadata_SOURCES = array-metadata.cpp
array_metadata_LDADD   = $(MEEPLIBS)

lifecycle_SOURCES = lifecycle.cpp
lifecycle_LDADD = $(MEEPLIBS)

random_state_lazy_default_SOURCES = random-state.cpp
random_state_lazy_default_CPPFLAGS = $(AM_CPPFLAGS) -DMEEP_RANDOM_STATE_SCENARIO=4
random_state_lazy_default_LDADD = $(MEEPLIBS)

random_state_initial_restore_SOURCES = random-state.cpp
random_state_initial_restore_CPPFLAGS = $(AM_CPPFLAGS) -DMEEP_RANDOM_STATE_SCENARIO=1
random_state_initial_restore_LDADD = $(MEEPLIBS)

random_state_initial_set_SOURCES = random-state.cpp
random_state_initial_set_CPPFLAGS = $(AM_CPPFLAGS) -DMEEP_RANDOM_STATE_SCENARIO=2
random_state_initial_set_LDADD = $(MEEPLIBS)

random_state_transitions_SOURCES = random-state.cpp
random_state_transitions_CPPFLAGS = $(AM_CPPFLAGS) -DMEEP_RANDOM_STATE_SCENARIO=3
random_state_transitions_LDADD = $(MEEPLIBS)

halo_plan_SOURCES = halo_plan.cpp
halo_plan_LDADD = $(MEEPLIBS)

storage_plan_SOURCES = storage_plan.cpp
storage_plan_LDADD = $(MEEPLIBS)

classification_SOURCES = classification.cpp
classification_LDADD = $(MEEPLIBS)

step_plan_SOURCES = step_plan.cpp
step_plan_LDADD = $(MEEPLIBS)
graph_plan_SOURCES = graph_plan.cpp
graph_plan_LDADD = $(MEEPLIBS)
prepared_plan_SOURCES = prepared_plan.cpp
prepared_plan_LDADD = $(MEEPLIBS)

region_plan_SOURCES = region_plan.cpp
region_plan_LDADD = $(MEEPLIBS)

transport_plan_SOURCES = transport_plan.cpp
transport_plan_LDADD = $(MEEPLIBS)

dependency_region_SOURCES = dependency_region.cpp
dependency_region_LDADD = $(MEEPLIBS)

mpi_context_SOURCES = mpi_context.cpp
mpi_context_LDADD = $(MEEPLIBS)

mpi_context_finalized_SOURCES = mpi_context_finalized.cpp
mpi_context_finalized_LDADD = $(MEEPLIBS)

device_uuid_reservation_SOURCES = device_uuid_reservation.cpp
device_uuid_reservation_LDADD = $(MEEPLIBS)

descriptors_SOURCES = descriptors.cpp
descriptors_LDADD = $(MEEPLIBS)

backend_api_SOURCES = backend_api.cpp
backend_api_LDADD = $(MEEPLIBS)

aniso_disp_SOURCES = aniso_disp.cpp
aniso_disp_LDADD = $(MEEPLIBS)

bench_SOURCES = bench.cpp
bench_LDADD = $(MEEPLIBS)

bragg_transmission_SOURCES = bragg_transmission.cpp
bragg_transmission_LDADD = $(MEEPLIBS)

convergence_cyl_waveguide_SOURCES = convergence_cyl_waveguide.cpp
convergence_cyl_waveguide_LDADD = $(MEEPLIBS)

cylindrical_SOURCES = cylindrical.cpp
cylindrical_LDADD = $(MEEPLIBS)

dump_load_SOURCES = dump_load.cpp
dump_load_LDADD = $(MEEPLIBS)

flux_SOURCES = flux.cpp
flux_LDADD = $(MEEPLIBS)

harmonics_SOURCES = harmonics.cpp
harmonics_LDADD = $(MEEPLIBS)

integrate_SOURCES = integrate.cpp
integrate_LDADD = $(MEEPLIBS)

known_results_SOURCES = known_results.cpp
known_results_LDADD = $(MEEPLIBS)

near2far_SOURCES = near2far.cpp
near2far_LDADD = $(MEEPLIBS)

one_dimensional_SOURCES = one_dimensional.cpp
one_dimensional_LDADD = $(MEEPLIBS)

physical_SOURCES = physical.cpp
physical_LDADD = $(MEEPLIBS)

stress_tensor_SOURCES = stress_tensor.cpp
stress_tensor_LDADD = $(MEEPLIBS)

symmetry_SOURCES = symmetry.cpp
symmetry_LDADD = $(MEEPLIBS)

three_d_SOURCES = three_d.cpp
three_d_LDADD = $(MEEPLIBS)

two_dimensional_SOURCES = two_dimensional.cpp
two_dimensional_LDADD = $(MEEPLIBS)

2D_convergence_SOURCES = 2D_convergence.cpp
2D_convergence_LDADD = $(MEEPLIBS)

h5test_SOURCES = h5test.cpp
h5test_LDADD = $(MEEPLIBS)

pml_SOURCES = pml.cpp
pml_LDADD = $(MEEPLIBS)

absorber_1d_ll_SOURCES = absorber-1d-ll.cpp
absorber_1d_ll_LDADD   = $(MEEPLIBS)

cyl_ellipsoid_ll_SOURCES = cyl-ellipsoid-ll.cpp
cyl_ellipsoid_ll_LDADD   = $(MEEPLIBS)

pw_source_ll_SOURCES = pw-source-ll.cpp
pw_source_ll_LDADD   = $(MEEPLIBS)

ring_ll_SOURCES = ring-ll.cpp
ring_ll_LDADD   = $(MEEPLIBS)

array_slice_ll_SOURCES = array-slice-ll.cpp
array_slice_ll_LDADD   = $(MEEPLIBS)

dft_fields_SOURCES = dft-fields.cpp
dft_fields_LDADD   = $(MEEPLIBS)

bend_flux_ll_SOURCES = bend-flux-ll.cpp
bend_flux_ll_LDADD   = $(MEEPLIBS)

user_defined_material_SOURCES = user-defined-material.cpp
user_defined_material_LDADD   = $(MEEPLIBS)

dist_noinst_DATA = cyl-ellipsoid-eps-ref.h5 array-slice-ll-ref.h5

TESTS = lifecycle random-state-lazy-default random-state-initial-restore random-state-initial-set random-state-transitions halo_plan storage_plan classification step_plan prepared_plan region_plan transport_plan dependency_region mpi_context mpi_context_finalized device_uuid_reservation descriptors backend_api aniso_disp bench bragg_transmission convergence_cyl_waveguide cylindrical dump_load flux harmonics integrate known_results near2far one_dimensional physical stress_tensor symmetry three_d two_dimensional 2D_convergence h5test pml
check_SCRIPTS =
CLEANFILES =

if WITH_MPI
  LOG_COMPILER = $(RUNCODE)
else
  TESTS += cyl-ellipsoid-ll array-slice-ll ring-ll
endif

if HAVE_NVIDIA_BACKEND
NVIDIA_EAGER_RUNTIME_TESTS = nvidia_runtime_smoke nvidia_arena_smoke nvidia_step_smoke nvidia_polarization_smoke nvidia_multilevel_smoke nvidia_dft_smoke nvidia_dft_reduction_smoke nvidia_flux_smoke nvidia_cw_smoke nvidia_initialization_smoke nvidia_adjoint_smoke
NVIDIA_RUNTIME_TESTS = $(NVIDIA_EAGER_RUNTIME_TESTS)
NVIDIA_INTEGRATION_TESTS = nvidia_backend_state nvidia_device_reservation nvidia_timestep nvidia_dft nvidia_cw
NVIDIA_PHONY_TARGETS = nvidia-eager-runtime-check nvidia-runtime-check amd-eager-check

if USE_CUDA_DEVICE_COMPILER
NVIDIA_DEVICE_LIBRARY = $(top_builddir)/src/libmeep_nvidia_runtime.la
NVIDIA_DEVICE_ARCHIVE = $(top_builddir)/src/.libs/libmeep_nvidia_runtime.a
NVIDIA_DEVICE_BUILD_TARGET = libmeep_nvidia_runtime.la
NVIDIA_DEVICE_COMPILER = $(NVCC)
NVIDIA_DEVICE_HOST_FLAGS = $(CUDA_HOST_FLAGS)
NVIDIA_DEVICE_FLAGS = $(NVCCFLAGS)
NVIDIA_DEVICE_ARCH_FLAGS = $(CUDA_ARCH_FLAGS)
NVIDIA_DEVICE_CPPFLAGS = $(CUDA_CPPFLAGS)
NVIDIA_DEVICE_LINKER_FLAGS = $(CUDA_NVCC_LDFLAGS)
NVIDIA_HOST_LINKER_FLAGS = $(CUDA_LDFLAGS)
NVIDIA_DEVICE_LIBS = $(CUDA_LIBS)
NVIDIA_DEVICE_PRECISE_FLAGS = --fmad=false
NVIDIA_RUNTIME_TESTS += nvidia_graph_smoke
TESTS += $(NVIDIA_INTEGRATION_TESTS)
endif

if USE_HIP_DEVICE_COMPILER
NVIDIA_DEVICE_LIBRARY = $(top_builddir)/src/libmeep_amd_runtime.la
NVIDIA_DEVICE_ARCHIVE = $(top_builddir)/src/.libs/libmeep_amd_runtime.a
NVIDIA_DEVICE_BUILD_TARGET = amd-eager-build-check
NVIDIA_DEVICE_COMPILER = $(HIPCC)
NVIDIA_DEVICE_FLAGS = $(HIPCCFLAGS)
NVIDIA_DEVICE_ARCH_FLAGS = $(HIP_ARCH_FLAGS)
NVIDIA_DEVICE_CPPFLAGS = $(HIP_CPPFLAGS)
NVIDIA_DEVICE_LINKER_FLAGS = $(HIP_LDFLAGS)
NVIDIA_HOST_LINKER_FLAGS = $(HIP_LDFLAGS)
NVIDIA_DEVICE_LIBS = $(HIP_LIBS)
NVIDIA_DEVICE_SOURCE_FLAGS = -x hip
NVIDIA_DEVICE_LINK_FLAGS = -x none
NVIDIA_DEVICE_PRECISE_FLAGS = $(HIP_PRECISE_FLAGS)
if HAVE_HIP_GRAPHS
NVIDIA_RUNTIME_TESTS += nvidia_graph_smoke
NVIDIA_PHONY_TARGETS += amd-graph-check
else
NVIDIA_PHONY_TARGETS += amd-required-graph-rejection-check
endif
endif

if WITH_MPI
NVIDIA_RUNTIME_TESTS += nvidia_mpi_runtime_smoke
check_PROGRAMS += nvidia_staged_transport
nvidia_staged_transport_SOURCES = nvidia_staged_transport.cpp
nvidia_staged_transport_CPPFLAGS = $(AM_CPPFLAGS) $(NVIDIA_DEVICE_CPPFLAGS)
nvidia_staged_transport_LDADD = $(MEEPLIBS)
if USE_CUDA_DEVICE_COMPILER
TESTS += nvidia_staged_transport
endif
if USE_HIP_DEVICE_COMPILER
NVIDIA_PHONY_TARGETS += amd-staged-mpi-check
NVIDIA_PHONY_TARGETS += amd-direct-mpi-check
NVIDIA_PHONY_TARGETS += amd-no-provider-mpi-check
NVIDIA_PHONY_TARGETS += amd-staged-transport-acceptance
NVIDIA_PHONY_TARGETS += amd-direct-transport-acceptance
nvidia_staged_transport_LDFLAGS = $(HIP_LDFLAGS)
check_PROGRAMS += validate_rocm_ucx_log
validate_rocm_ucx_log_SOURCES = validate_rocm_ucx_log.cpp
endif
endif
check_SCRIPTS += $(NVIDIA_RUNTIME_TESTS)
check_PROGRAMS += $(NVIDIA_INTEGRATION_TESTS)
CLEANFILES += $(NVIDIA_RUNTIME_TESTS) nvidia-required-graph-rejection.log \
  nvidia-mpi-direct-rejection.log nvidia-mpi-oversubscription.log \
  nvidia-mpi-topology-rejection.log nvidia-mpi-rocm-provider.log \
  nvidia-mpi-direct-topology-rejection.log nvidia-mpi-direct-oversubscription.log
nvidia_backend_state_SOURCES = nvidia_backend_state.cpp
nvidia_backend_state_LDADD = $(MEEPLIBS)
nvidia_device_reservation_SOURCES = nvidia_device_reservation.cpp
nvidia_device_reservation_LDADD = $(MEEPLIBS)
nvidia_timestep_SOURCES = nvidia_timestep.cpp
nvidia_timestep_LDADD = $(MEEPLIBS)
nvidia_dft_SOURCES = nvidia_dft.cpp
nvidia_dft_LDADD = $(MEEPLIBS)
nvidia_cw_SOURCES = nvidia_cw.cpp
nvidia_cw_LDADD = $(MEEPLIBS)

if USE_HIP_DEVICE_COMPILER
nvidia_backend_state_LDFLAGS = $(HIP_LDFLAGS)
nvidia_device_reservation_LDFLAGS = $(HIP_LDFLAGS)
nvidia_timestep_LDFLAGS = $(HIP_LDFLAGS)
nvidia_dft_LDFLAGS = $(HIP_LDFLAGS)
nvidia_cw_LDFLAGS = $(HIP_LDFLAGS)
endif

check-local: check-TESTS

if USE_CUDA_DEVICE_COMPILER
	$(MAKE) nvidia-runtime-check
endif

if USE_HIP_DEVICE_COMPILER
	$(MAKE) amd-eager-check
if HAVE_HIP_GRAPHS
	$(MAKE) amd-graph-check
else
	$(MAKE) amd-required-graph-rejection-check
endif
if WITH_MPI
	$(MAKE) amd-staged-mpi-check
endif
endif

.PHONY: $(NVIDIA_PHONY_TARGETS)
nvidia-eager-runtime-check: $(NVIDIA_EAGER_RUNTIME_TESTS)
	./nvidia_runtime_smoke
	./nvidia_arena_smoke
	./nvidia_step_smoke
	./nvidia_polarization_smoke
	./nvidia_multilevel_smoke
	./nvidia_dft_smoke
	./nvidia_dft_reduction_smoke
	./nvidia_flux_smoke
	./nvidia_cw_smoke
	./nvidia_initialization_smoke
	./nvidia_adjoint_smoke

nvidia-runtime-check: nvidia-eager-runtime-check $(NVIDIA_RUNTIME_TESTS)
if USE_CUDA_DEVICE_COMPILER
	./nvidia_graph_smoke
if WITH_MPI
	$(RUNCODE) ./nvidia_mpi_runtime_smoke --transport=staged
	$(RUNCODE) ./nvidia_mpi_runtime_smoke --transport=auto
endif
endif

if WITH_MPI
if USE_HIP_DEVICE_COMPILER
AMD_MPIEXEC = $(MPIRUN)
## ROCr and rocm-smi card ordinals differ on the validation host. These
## defaults resolve through HIP to physical BDFs 08:00.0 and 88:00.0;
## override both variables together on another host.
AMD_MPI_VISIBLE_DEVICES = 1,5
AMD_MPI_EXPECTED_OWNER_BDFS = 0000:08:00.0,0000:88:00.0
AMD_MPI_INVALID_OWNER_BDFS = 0000:78:00.0,0000:f9:00.0
AMD_MPI_OVERSUBSCRIBED_VISIBLE_DEVICES = 1
AMD_MPI_NP2_FLAGS = -np 2 --map-by ppr:1:package:PE=2 --bind-to core
AMD_MPI_NP4_FLAGS = -np 4 --map-by ppr:2:package:PE=2 --bind-to core
AMD_MPI_NP4_OWNER_VISIBLE_DEVICES = 1,3,5,7
AMD_MPI_NP4_OWNER_BDFS = 0000:08:00.0,0000:18:00.0,0000:88:00.0,0000:98:00.0
AMD_MPI_NP8_OWNER_VISIBLE_DEVICES = 0,1,2,3,4,5,6,7
AMD_MPI_NP8_OWNER_BDFS = 0000:78:00.0,0000:08:00.0,0000:68:00.0,0000:18:00.0,0000:f9:00.0,0000:88:00.0,0000:e8:00.0,0000:98:00.0
AMD_MPI_NP4_OWNER_FLAGS = -np 4 --map-by ppr:2:package:PE=2 --bind-to core
AMD_MPI_NP8_OWNER_FLAGS = -np 8 --map-by ppr:4:package:PE=2 --bind-to core
AMD_MPI_STAGED_ENV = MEEP_GPU_AWARE_MPI=no \
  ROCR_VISIBLE_DEVICES=$(AMD_MPI_VISIBLE_DEVICES) \
  MEEP_AMD_EXPECTED_OWNER_BDFS=$(AMD_MPI_EXPECTED_OWNER_BDFS)
AMD_MPI_DIRECT_ENV = MEEP_GPU_AWARE_MPI=yes \
  ROCR_VISIBLE_DEVICES=$(AMD_MPI_VISIBLE_DEVICES) \
  MEEP_AMD_EXPECTED_OWNER_BDFS=$(AMD_MPI_EXPECTED_OWNER_BDFS) \
  OMPI_MCA_pml=ucx UCX_TLS=self,sm,rocm_copy,rocm_ipc
AMD_ACCEPTANCE_PREFIX =
AMD_ACCEPTANCE_PYTHON_CMD =
AMD_ACCEPTANCE_PYTHONPATH =
AMD_ACCEPTANCE_LIBRARY_PATH = $(top_builddir)/src/.libs
AMD_ACCEPTANCE_ROCM_SMI = $(dir $(HIPCC))rocm-smi
AMD_ACCEPTANCE_OUTPUT_DIR = $(abs_top_builddir)/amd-staged-transport-acceptance
AMD_DIRECT_ACCEPTANCE_OUTPUT_DIR = $(abs_top_builddir)/amd-direct-transport-acceptance

amd-staged-mpi-check: nvidia_mpi_runtime_smoke nvidia_staged_transport
	$(AMD_MPI_STAGED_ENV) $(AMD_MPIEXEC) $(AMD_MPI_NP2_FLAGS) \
	  ./nvidia_mpi_runtime_smoke --transport=staged
	$(AMD_MPI_STAGED_ENV) $(AMD_MPIEXEC) $(AMD_MPI_NP2_FLAGS) \
	  ./nvidia_staged_transport --staged-only
	$(AMD_MPI_STAGED_ENV) $(AMD_MPIEXEC) $(AMD_MPI_NP4_FLAGS) \
	  ./nvidia_staged_transport --staged-only --idle-np4-only
	@set -e; if MEEP_GPU_AWARE_MPI=no \
	  ROCR_VISIBLE_DEVICES=$(AMD_MPI_VISIBLE_DEVICES) \
	  MEEP_AMD_EXPECTED_OWNER_BDFS=$(AMD_MPI_INVALID_OWNER_BDFS) \
	  $(AMD_MPIEXEC) $(AMD_MPI_NP4_FLAGS) \
	  ./nvidia_staged_transport --staged-only --idle-np4-only \
	  >nvidia-mpi-topology-rejection.log 2>&1; then \
	  cat nvidia-mpi-topology-rejection.log; \
	  echo "expected mismatched physical GPU topology to fail" >&2; exit 1; \
	fi
	@grep -q "expected physical PCI BDF" nvidia-mpi-topology-rejection.log || { \
	  cat nvidia-mpi-topology-rejection.log; exit 1; }
	@set -e; if MEEP_GPU_AWARE_MPI=no \
	  ROCR_VISIBLE_DEVICES=$(AMD_MPI_OVERSUBSCRIBED_VISIBLE_DEVICES) \
	  $(AMD_MPIEXEC) $(AMD_MPI_NP2_FLAGS) \
	  ./nvidia_mpi_runtime_smoke --transport=staged \
	  >nvidia-mpi-oversubscription.log 2>&1; then \
	  cat nvidia-mpi-oversubscription.log; \
	  echo "expected duplicate/oversubscribed GPU assignment to fail" >&2; exit 1; \
	fi
	@grep -q "more node-local ranks than visible NVIDIA devices" \
	  nvidia-mpi-oversubscription.log || { cat nvidia-mpi-oversubscription.log; exit 1; }

amd-no-provider-mpi-check: nvidia_mpi_runtime_smoke
	ROCR_VISIBLE_DEVICES=$(AMD_MPI_VISIBLE_DEVICES) \
	  $(AMD_MPIEXEC) $(AMD_MPI_NP2_FLAGS) \
	  ./nvidia_mpi_runtime_smoke --transport=auto --forbid-device-mpi
	@set -e; if ROCR_VISIBLE_DEVICES=$(AMD_MPI_VISIBLE_DEVICES) \
	  $(AMD_MPIEXEC) $(AMD_MPI_NP2_FLAGS) \
	  ./nvidia_mpi_runtime_smoke --transport=direct --forbid-device-mpi \
	  >nvidia-mpi-direct-rejection.log 2>&1; then \
	  cat nvidia-mpi-direct-rejection.log; \
	  echo "expected forced direct transport to fail closed" >&2; exit 1; \
	fi
	@grep -Eq "forced direct transport requires positive .* on every rank" \
	  nvidia-mpi-direct-rejection.log || { cat nvidia-mpi-direct-rejection.log; exit 1; }

amd-direct-mpi-check: nvidia_mpi_runtime_smoke nvidia_staged_transport validate_rocm_ucx_log
	./validate_rocm_ucx_log --self-test
	@set -e; $(AMD_MPI_DIRECT_ENV) \
	  $(AMD_MPIEXEC) $(AMD_MPI_NP2_FLAGS) \
	  ./nvidia_staged_transport --direct-all-owners-only \
	  >nvidia-mpi-rocm-provider.log 2>&1; \
	  $(AMD_MPI_DIRECT_ENV) UCX_PROTO_INFO=y \
	  $(AMD_MPIEXEC) $(AMD_MPI_NP2_FLAGS) \
	  ./nvidia_mpi_runtime_smoke --transport=direct --require-device-mpi \
	  >>nvidia-mpi-rocm-provider.log 2>&1; \
	  cat nvidia-mpi-rocm-provider.log
	@./validate_rocm_ucx_log nvidia-mpi-rocm-provider.log \
	  $(AMD_MPI_EXPECTED_OWNER_BDFS) || { \
	  cat nvidia-mpi-rocm-provider.log; \
	  echo "UCX trace did not prove two-device ROCm IPC zero-copy" >&2; exit 1; }
	$(AMD_MPI_DIRECT_ENV) $(AMD_MPIEXEC) $(AMD_MPI_NP2_FLAGS) \
	  ./nvidia_mpi_runtime_smoke --transport=auto --require-device-mpi
	$(AMD_MPI_DIRECT_ENV) $(AMD_MPIEXEC) $(AMD_MPI_NP2_FLAGS) \
	  ./nvidia_staged_transport
	MEEP_GPU_AWARE_MPI=yes \
	  ROCR_VISIBLE_DEVICES=$(AMD_MPI_NP4_OWNER_VISIBLE_DEVICES) \
	  MEEP_AMD_EXPECTED_OWNER_BDFS=$(AMD_MPI_NP4_OWNER_BDFS) \
	  OMPI_MCA_pml=ucx UCX_TLS=self,sm,rocm_copy,rocm_ipc \
	  $(AMD_MPIEXEC) $(AMD_MPI_NP4_OWNER_FLAGS) \
	  ./nvidia_staged_transport --direct-all-owners-only
	MEEP_GPU_AWARE_MPI=yes \
	  ROCR_VISIBLE_DEVICES=$(AMD_MPI_NP8_OWNER_VISIBLE_DEVICES) \
	  MEEP_AMD_EXPECTED_OWNER_BDFS=$(AMD_MPI_NP8_OWNER_BDFS) \
	  OMPI_MCA_pml=ucx UCX_TLS=self,sm,rocm_copy,rocm_ipc \
	  $(AMD_MPIEXEC) $(AMD_MPI_NP8_OWNER_FLAGS) \
	  ./nvidia_staged_transport --direct-all-owners-only
	@set -e; if MEEP_GPU_AWARE_MPI=yes \
	  ROCR_VISIBLE_DEVICES=$(AMD_MPI_VISIBLE_DEVICES) \
	  MEEP_AMD_EXPECTED_OWNER_BDFS=$(AMD_MPI_INVALID_OWNER_BDFS) \
	  OMPI_MCA_pml=ucx UCX_TLS=self,sm,rocm_copy,rocm_ipc \
	  $(AMD_MPIEXEC) $(AMD_MPI_NP2_FLAGS) \
	  ./nvidia_staged_transport --direct-all-owners-only \
	  >nvidia-mpi-direct-topology-rejection.log 2>&1; then \
	  cat nvidia-mpi-direct-topology-rejection.log; \
	  echo "expected direct topology mismatch to fail closed" >&2; exit 1; \
	fi
	@grep -q "all-owner rank did not resolve to its expected physical PCI BDF" \
	  nvidia-mpi-direct-topology-rejection.log && \
	  grep -q "DIRECT_RING_SKIPPED topology admission failed" \
	  nvidia-mpi-direct-topology-rejection.log && \
	  ! grep -q "DIRECT_RING_BEGIN" nvidia-mpi-direct-topology-rejection.log || { \
	  cat nvidia-mpi-direct-topology-rejection.log; exit 1; }
	@set -e; if MEEP_GPU_AWARE_MPI=yes \
	  ROCR_VISIBLE_DEVICES=$(AMD_MPI_OVERSUBSCRIBED_VISIBLE_DEVICES) \
	  MEEP_AMD_EXPECTED_OWNER_BDFS=0000:08:00.0 \
	  OMPI_MCA_pml=ucx UCX_TLS=self,sm,rocm_copy,rocm_ipc \
	  $(AMD_MPIEXEC) $(AMD_MPI_NP2_FLAGS) \
	  ./nvidia_staged_transport --direct-all-owners-only \
	  >nvidia-mpi-direct-oversubscription.log 2>&1; then \
	  cat nvidia-mpi-direct-oversubscription.log; \
	  echo "expected direct oversubscription to fail closed" >&2; exit 1; \
	fi
	@grep -q "requires one visible device per rank" \
	  nvidia-mpi-direct-oversubscription.log && \
	  grep -q "DIRECT_RING_SKIPPED topology admission failed" \
	  nvidia-mpi-direct-oversubscription.log && \
	  ! grep -q "DIRECT_RING_BEGIN" nvidia-mpi-direct-oversubscription.log || { \
	  cat nvidia-mpi-direct-oversubscription.log; exit 1; }

# This end-to-end Python gate is explicit because it requires a branch-matched
# Python module build. No direct route is part of the PR4 invocation.
amd-staged-transport-acceptance:
	@test -n "$(AMD_ACCEPTANCE_PREFIX)" || { \
	  echo "set AMD_ACCEPTANCE_PREFIX to the private MPI installation prefix" >&2; exit 2; }
	@test -n "$(AMD_ACCEPTANCE_PYTHON_CMD)" || { \
	  echo "set AMD_ACCEPTANCE_PYTHON_CMD to the branch-matched Python executable" >&2; exit 2; }
	@test -n "$(AMD_ACCEPTANCE_PYTHONPATH)" || { \
	  echo "set AMD_ACCEPTANCE_PYTHONPATH to the branch-matched meep package" >&2; exit 2; }
	$(AMD_ACCEPTANCE_PYTHON_CMD) \
	  $(top_srcdir)/benchmarks/nvidia/paper_2506_16665/run_transport_acceptance.py \
	  --output-dir $(AMD_ACCEPTANCE_OUTPUT_DIR) --routes cpu,staged \
	  --prefix $(AMD_ACCEPTANCE_PREFIX) \
	  --python $(AMD_ACCEPTANCE_PYTHON_CMD) --mpiexec $(AMD_MPIEXEC) \
	  --visible-devices $(AMD_MPI_VISIBLE_DEVICES) \
	  --rocm-smi $(AMD_ACCEPTANCE_ROCM_SMI) \
	  --pythonpath $(AMD_ACCEPTANCE_PYTHONPATH) \
	  --library-path $(AMD_ACCEPTANCE_LIBRARY_PATH)

amd-direct-transport-acceptance: amd-direct-mpi-check
	@test -n "$(AMD_ACCEPTANCE_PREFIX)" || { \
	  echo "set AMD_ACCEPTANCE_PREFIX to the private MPI installation prefix" >&2; exit 2; }
	@test -n "$(AMD_ACCEPTANCE_PYTHON_CMD)" || { \
	  echo "set AMD_ACCEPTANCE_PYTHON_CMD to the branch-matched Python executable" >&2; exit 2; }
	@test -n "$(AMD_ACCEPTANCE_PYTHONPATH)" || { \
	  echo "set AMD_ACCEPTANCE_PYTHONPATH to the branch-matched meep package" >&2; exit 2; }
	$(AMD_ACCEPTANCE_PYTHON_CMD) \
	  $(top_srcdir)/benchmarks/nvidia/paper_2506_16665/run_transport_acceptance.py \
	  --output-dir $(AMD_DIRECT_ACCEPTANCE_OUTPUT_DIR) --routes cpu,staged,direct \
	  --prefix $(AMD_ACCEPTANCE_PREFIX) \
	  --python $(AMD_ACCEPTANCE_PYTHON_CMD) --mpiexec $(AMD_MPIEXEC) \
	  --visible-devices $(AMD_MPI_VISIBLE_DEVICES) \
	  --rocm-smi $(AMD_ACCEPTANCE_ROCM_SMI) \
	  --provider-zero-copy-log $(abs_builddir)/nvidia-mpi-rocm-provider.log \
	  --pythonpath $(AMD_ACCEPTANCE_PYTHONPATH) \
	  --library-path $(AMD_ACCEPTANCE_LIBRARY_PATH)
endif
endif

if USE_HIP_DEVICE_COMPILER
# PR2 validates eager, single-rank execution. The field-layout switch narrowly
# defers an existing backend-neutral candidate-preparation lifecycle defect.
amd-eager-check: nvidia-eager-runtime-check $(NVIDIA_INTEGRATION_TESTS)
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_SKIP_FIELD_LAYOUT_REBUILD=1 \
	  ./nvidia_backend_state
	MEEP_NVIDIA_GRAPH_MODE=eager ./nvidia_device_reservation
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_TIMESTEP_BETA_ONLY=1 ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_TIMESTEP_GYRO_ONLY=1 ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_TIMESTEP_BFAST_ONLY=1 ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_TIMESTEP_CYLINDRICAL_ONLY=1 \
	  MEEP_NVIDIA_CYLINDRICAL_CASE=m0-real-pml-conductive ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_TIMESTEP_MAGNETIC_ONLY=1 ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_TIMESTEP_MATERIAL_ONLY=1 ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_TIMESTEP_FLUX_ONLY=1 ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_TIMESTEP_NOISY_ONLY=1 ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_TIMESTEP_MULTILEVEL_ONLY=1 \
	  MEEP_NVIDIA_EAGER_REPRESENTATIVE_ONLY=1 ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_SOURCE_REUSE_ONLY=1 \
	  MEEP_NVIDIA_EAGER_REPRESENTATIVE_ONLY=1 ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=eager ./nvidia_dft
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_HIP_EAGER_TOLERANCE=1 \
	  ./nvidia_cw --precision=native
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_HIP_EAGER_TOLERANCE=1 \
	  ./nvidia_cw --precision=mixed
	MEEP_NVIDIA_GRAPH_MODE=eager MEEP_NVIDIA_HIP_EAGER_TOLERANCE=1 \
	  ./nvidia_cw --precision=f32

if HAVE_HIP_GRAPHS
# PR3 requires native HIP graphs when the selected compiler headers advertise
# HIP 7 or newer. These are the existing CUDA graph lifecycle, integration,
# value-reuse, recapture, rollback, and numerical-parity gates.
amd-graph-check: nvidia_graph_smoke nvidia_timestep nvidia_cw
	./nvidia_graph_smoke
	MEEP_NVIDIA_GRAPH_ONLY=1 ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=required MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_GRAPH_EXPECT_ENABLED=1 MEEP_NVIDIA_GRAPH_MAGNETIC_ONLY=1 \
	  ./nvidia_timestep
	MEEP_NVIDIA_GRAPH_MODE=required MEEP_NVIDIA_GRAPH_ASSERT=1 \
	  MEEP_NVIDIA_GRAPH_EXPECT_ENABLED=1 MEEP_NVIDIA_SOURCE_REUSE_ONLY=1 \
	  ./nvidia_timestep
	MEEP_NVIDIA_CW_GRAPH_ONLY=1 ./nvidia_cw --precision=native
	MEEP_NVIDIA_CW_GRAPH_ONLY=1 ./nvidia_cw --precision=mixed
	MEEP_NVIDIA_CW_GRAPH_ONLY=1 ./nvidia_cw --precision=f32
else
# HIP before version 7 remains an eager-capable production build, but required
# graph mode must continue to fail closed with the specific capability error.
amd-required-graph-rejection-check: nvidia_backend_state
	@set -e; log=nvidia-required-graph-rejection.log; \
	  if MEEP_NVIDIA_GRAPH_MODE=required ./nvidia_backend_state >$$log 2>&1; then \
	    cat $$log; \
	    echo "required graph mode unexpectedly succeeded" >&2; \
	    exit 1; \
	  fi; \
	  cat $$log; \
	  grep -F "required CUDA graph mode is unsupported" $$log >/dev/null; \
	  rm -f $$log
endif
endif

$(NVIDIA_DEVICE_LIBRARY):
	$(MAKE) -C $(top_builddir)/src $(NVIDIA_DEVICE_BUILD_TARGET)

$(NVIDIA_DEVICE_ARCHIVE): $(NVIDIA_DEVICE_LIBRARY)
	$(AM_V_at)test -f $@

nvidia_runtime_smoke: $(srcdir)/nvidia_runtime_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) $(NVIDIA_DEVICE_CPPFLAGS) \
	  $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< $(NVIDIA_DEVICE_LINK_FLAGS) \
	  $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) $(NVIDIA_DEVICE_LIBS) -o $@

nvidia_graph_smoke: $(srcdir)/nvidia_graph_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) $(NVIDIA_DEVICE_CPPFLAGS) \
	  $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< $(NVIDIA_DEVICE_LINK_FLAGS) \
	  $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) $(NVIDIA_DEVICE_LIBS) -o $@

nvidia_arena_smoke: $(srcdir)/nvidia_arena_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) $(NVIDIA_DEVICE_CPPFLAGS) \
	  $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< $(NVIDIA_DEVICE_LINK_FLAGS) \
	  $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) $(NVIDIA_DEVICE_LIBS) -o $@

nvidia_step_smoke: $(srcdir)/nvidia_step_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) $(NVIDIA_DEVICE_CPPFLAGS) \
	  $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< $(NVIDIA_DEVICE_LINK_FLAGS) \
	  $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) $(NVIDIA_DEVICE_LIBS) -o $@

nvidia_polarization_smoke: $(srcdir)/nvidia_polarization_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) $(NVIDIA_DEVICE_CPPFLAGS) \
	  $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< $(NVIDIA_DEVICE_LINK_FLAGS) \
	  $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) $(NVIDIA_DEVICE_LIBS) -o $@

nvidia_multilevel_smoke: $(srcdir)/nvidia_multilevel_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_PRECISE_FLAGS) $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) \
	  $(NVIDIA_DEVICE_CPPFLAGS) $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< \
	  $(NVIDIA_DEVICE_LINK_FLAGS) $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) \
	  $(NVIDIA_DEVICE_LIBS) -o $@

nvidia_dft_smoke: $(srcdir)/nvidia_dft_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) $(NVIDIA_DEVICE_CPPFLAGS) \
	  $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< $(NVIDIA_DEVICE_LINK_FLAGS) \
	  $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) $(NVIDIA_DEVICE_LIBS) -o $@

nvidia_dft_reduction_smoke: $(srcdir)/nvidia_dft_reduction_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) $(NVIDIA_DEVICE_CPPFLAGS) \
	  $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< $(NVIDIA_DEVICE_LINK_FLAGS) \
	  $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) $(NVIDIA_DEVICE_LIBS) -o $@

nvidia_flux_smoke: $(srcdir)/nvidia_flux_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) $(NVIDIA_DEVICE_CPPFLAGS) \
	  $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< $(NVIDIA_DEVICE_LINK_FLAGS) \
	  $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) $(NVIDIA_DEVICE_LIBS) -o $@

nvidia_cw_smoke: $(srcdir)/nvidia_cw_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_PRECISE_FLAGS) $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) \
	  $(NVIDIA_DEVICE_CPPFLAGS) $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< \
	  $(NVIDIA_DEVICE_LINK_FLAGS) $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) \
	  $(NVIDIA_DEVICE_LIBS) -o $@

nvidia_initialization_smoke: $(srcdir)/nvidia_initialization_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_PRECISE_FLAGS) $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) \
	  $(NVIDIA_DEVICE_CPPFLAGS) $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< \
	  $(NVIDIA_DEVICE_LINK_FLAGS) $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) \
	  $(NVIDIA_DEVICE_LIBS) -o $@

nvidia_adjoint_smoke: $(srcdir)/nvidia_adjoint_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(NVIDIA_DEVICE_COMPILER) $(NVIDIA_DEVICE_HOST_FLAGS) -std=c++11 \
	  $(NVIDIA_DEVICE_PRECISE_FLAGS) $(NVIDIA_DEVICE_FLAGS) $(NVIDIA_DEVICE_ARCH_FLAGS) \
	  $(NVIDIA_DEVICE_CPPFLAGS) $(AM_CPPFLAGS) $(NVIDIA_DEVICE_SOURCE_FLAGS) $< \
	  $(NVIDIA_DEVICE_LINK_FLAGS) $(NVIDIA_DEVICE_ARCHIVE) $(NVIDIA_DEVICE_LINKER_FLAGS) \
	  $(NVIDIA_DEVICE_LIBS) -o $@

if WITH_MPI
nvidia_mpi_runtime_smoke: $(srcdir)/nvidia_mpi_runtime_smoke.cu $(NVIDIA_DEVICE_ARCHIVE)
	$(AM_V_GEN)$(CXX) $(DEFS) $(DEFAULT_INCLUDES) $(CPPFLAGS) $(AM_CPPFLAGS) \
	  $(CXXFLAGS) -x c++ $< -x none \
	  $(NVIDIA_DEVICE_ARCHIVE) $(LDFLAGS) $(NVIDIA_HOST_LINKER_FLAGS) \
	  $(NVIDIA_DEVICE_LIBS) $(LIBS) -o $@
endif
endif

if BUILD_HIP_PORTABILITY_CHECK
HIP_PORTABILITY_ARCHIVE = $(top_builddir)/src/.libs/libmeep_hip_portability.a
HIP_PORTABILITY_RUNTIME_TESTS = hip_portability_runtime_smoke hip_portability_arena_smoke
CLEANFILES += $(HIP_PORTABILITY_RUNTIME_TESTS)

.PHONY: hip-portability-runtime-check hip-portability-source-build-check
hip-portability-runtime-check: $(HIP_PORTABILITY_RUNTIME_TESTS)
	./hip_portability_runtime_smoke
	./hip_portability_arena_smoke

# The source-side target also compiles the other device units as syntax gates.
# Keeping this prerequisite phony makes the explicit runtime target revalidate
# the complete private portability surface before it executes either smoke.
hip-portability-source-build-check:
	$(MAKE) -C $(top_builddir)/src hip-portability-build-check
	$(AM_V_at)test -f $(HIP_PORTABILITY_ARCHIVE)

$(HIP_PORTABILITY_ARCHIVE): hip-portability-source-build-check
	$(AM_V_at)test -f $@

hip_portability_runtime_smoke: $(srcdir)/nvidia_runtime_smoke.cu $(HIP_PORTABILITY_ARCHIVE)
	$(AM_V_GEN)$(HIPCC) -std=c++11 $(HIPCCFLAGS) $(HIP_ARCH_FLAGS) \
	  $(HIP_PIC_FLAGS) $(HIP_CPPFLAGS) $(AM_CPPFLAGS) -x hip $< -x none \
	  $(HIP_PORTABILITY_ARCHIVE) $(HIP_LDFLAGS) $(HIP_LIBS) -o $@

hip_portability_arena_smoke: $(srcdir)/nvidia_arena_smoke.cu $(HIP_PORTABILITY_ARCHIVE)
	$(AM_V_GEN)$(HIPCC) -std=c++11 $(HIPCCFLAGS) $(HIP_ARCH_FLAGS) \
	  $(HIP_PIC_FLAGS) $(HIP_CPPFLAGS) $(AM_CPPFLAGS) -x hip $< -x none \
	  $(HIP_PORTABILITY_ARCHIVE) $(HIP_LDFLAGS) $(HIP_LIBS) -o $@
endif

# Note: this requires GNU make

.o.dac:
	$(CXXLINK) $< $(LIBMEEP) $(LIBS)

$(DAC): $(LIBMEEP)

.dac.done:
	$(RUNCODE) ./$<
	touch $@

benchmark: bench
	$(RUNCODE) ./bench

dac: $(DAC)

clean-local::
	rm -f *.o *.dac debug_out_* *.done

distclean-local:
	rm -f $(shell ls *.h5 | sed '/.*ref.*/d')
