SRC = aniso_disp.cpp bench.cpp bragg_transmission.cpp			\
convergence_cyl_waveguide.cpp cylindrical.cpp dump_load.cpp flux.cpp    \
harmonics.cpp integrate.cpp known_results.cpp near2far.cpp              \
one_dimensional.cpp physical.cpp stress_tensor.cpp symmetry.cpp 	\
three_d.cpp two_dimensional.cpp 2D_convergence.cpp h5test.cpp pml.cpp \
lifecycle.cpp halo_plan.cpp storage_plan.cpp classification.cpp step_plan.cpp region_plan.cpp descriptors.cpp backend_api.cpp

# The bitwise-neutrality harness (plan section 4) is Python, and is driven by
# CI rather than by `make check`: it needs two builds to compare.
EXTRA_DIST = $(SRC) bitwise nvidia_runtime_smoke.cu nvidia_arena_smoke.cu \
nvidia_dft_smoke.cu nvidia_dft_reduction_smoke.cu nvidia_flux_smoke.cu nvidia_cw_smoke.cu nvidia_mpi_runtime_smoke.cu nvidia_step_smoke.cu nvidia_polarization_smoke.cu nvidia_backend_state.cpp \
nvidia_timestep.cpp nvidia_dft.cpp nvidia_cw.cpp

OBJ = $(SRC:.cpp=.o)
DAC = $(SRC:.cpp=.dac)
DONE = $(SRC:.cpp=.done)

LIBMEEP = $(top_builddir)/src/libmeep.la
MEEPLIBS = $(LIBMEEP)
AM_CPPFLAGS = -I$(top_srcdir)/src -DDATADIR=\"$(srcdir)/\"

.SUFFIXES = .dac .done

check_PROGRAMS = lifecycle halo_plan storage_plan classification step_plan prepared_plan region_plan descriptors backend_api aniso_disp bench bragg_transmission convergence_cyl_waveguide cylindrical dump_load flux harmonics integrate known_results near2far one_dimensional physical stress_tensor symmetry three_d two_dimensional 2D_convergence h5test pml pw-source-ll ring-ll cyl-ellipsoid-ll absorber-1d-ll array-slice-ll user-defined-material dft-fields bend-flux-ll array-metadata

array_metadata_SOURCES = array-metadata.cpp
array_metadata_LDADD   = $(MEEPLIBS)

lifecycle_SOURCES = lifecycle.cpp
lifecycle_LDADD = $(MEEPLIBS)

halo_plan_SOURCES = halo_plan.cpp
halo_plan_LDADD = $(MEEPLIBS)

storage_plan_SOURCES = storage_plan.cpp
storage_plan_LDADD = $(MEEPLIBS)

classification_SOURCES = classification.cpp
classification_LDADD = $(MEEPLIBS)

step_plan_SOURCES = step_plan.cpp
step_plan_LDADD = $(MEEPLIBS)
prepared_plan_SOURCES = prepared_plan.cpp
prepared_plan_LDADD = $(MEEPLIBS)

region_plan_SOURCES = region_plan.cpp
region_plan_LDADD = $(MEEPLIBS)

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

TESTS = lifecycle halo_plan storage_plan classification step_plan prepared_plan region_plan descriptors backend_api aniso_disp bench bragg_transmission convergence_cyl_waveguide cylindrical dump_load flux harmonics integrate known_results near2far one_dimensional physical stress_tensor symmetry three_d two_dimensional 2D_convergence h5test pml
check_SCRIPTS =
CLEANFILES =

if WITH_MPI
  LOG_COMPILER = $(RUNCODE)
else
  TESTS += cyl-ellipsoid-ll array-slice-ll ring-ll
endif

if HAVE_NVIDIA_BACKEND
NVIDIA_CUDA_LIBRARY = $(top_builddir)/src/libmeep_nvidia_runtime.la
NVIDIA_CUDA_ARCHIVE = $(top_builddir)/src/.libs/libmeep_nvidia_runtime.a
NVIDIA_RUNTIME_TESTS = nvidia_runtime_smoke nvidia_arena_smoke nvidia_step_smoke nvidia_polarization_smoke nvidia_dft_smoke nvidia_dft_reduction_smoke nvidia_flux_smoke nvidia_cw_smoke
if WITH_MPI
NVIDIA_RUNTIME_TESTS += nvidia_mpi_runtime_smoke
endif
check_SCRIPTS += $(NVIDIA_RUNTIME_TESTS)
check_PROGRAMS += nvidia_backend_state nvidia_timestep nvidia_dft nvidia_cw
TESTS += nvidia_backend_state nvidia_timestep nvidia_dft nvidia_cw
CLEANFILES += $(NVIDIA_RUNTIME_TESTS)
nvidia_backend_state_SOURCES = nvidia_backend_state.cpp
nvidia_backend_state_LDADD = $(MEEPLIBS)
nvidia_timestep_SOURCES = nvidia_timestep.cpp
nvidia_timestep_LDADD = $(MEEPLIBS)
nvidia_dft_SOURCES = nvidia_dft.cpp
nvidia_dft_LDADD = $(MEEPLIBS)
nvidia_cw_SOURCES = nvidia_cw.cpp
nvidia_cw_LDADD = $(MEEPLIBS)

check-local: check-TESTS
	$(MAKE) nvidia-runtime-check

.PHONY: nvidia-runtime-check
nvidia-runtime-check: $(NVIDIA_RUNTIME_TESTS)
	./nvidia_runtime_smoke
	./nvidia_arena_smoke
	./nvidia_step_smoke
	./nvidia_polarization_smoke
	./nvidia_dft_smoke
	./nvidia_dft_reduction_smoke
	./nvidia_flux_smoke
	./nvidia_cw_smoke
if WITH_MPI
	$(RUNCODE) ./nvidia_mpi_runtime_smoke --transport=staged
	$(RUNCODE) ./nvidia_mpi_runtime_smoke --transport=auto
endif

$(NVIDIA_CUDA_LIBRARY):
	$(MAKE) -C $(top_builddir)/src libmeep_nvidia_runtime.la

$(NVIDIA_CUDA_ARCHIVE): $(NVIDIA_CUDA_LIBRARY)
	$(AM_V_at)test -f $@

nvidia_runtime_smoke: $(srcdir)/nvidia_runtime_smoke.cu $(NVIDIA_CUDA_ARCHIVE)
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) $< $(NVIDIA_CUDA_ARCHIVE) \
	  $(CUDA_NVCC_LDFLAGS) $(CUDA_LIBS) -o $@

nvidia_arena_smoke: $(srcdir)/nvidia_arena_smoke.cu $(NVIDIA_CUDA_ARCHIVE)
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) $< $(NVIDIA_CUDA_ARCHIVE) \
	  $(CUDA_NVCC_LDFLAGS) $(CUDA_LIBS) -o $@

nvidia_step_smoke: $(srcdir)/nvidia_step_smoke.cu $(NVIDIA_CUDA_ARCHIVE)
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) $< $(NVIDIA_CUDA_ARCHIVE) \
	  $(CUDA_NVCC_LDFLAGS) $(CUDA_LIBS) -o $@

nvidia_polarization_smoke: $(srcdir)/nvidia_polarization_smoke.cu $(NVIDIA_CUDA_ARCHIVE)
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) $< $(NVIDIA_CUDA_ARCHIVE) \
	  $(CUDA_NVCC_LDFLAGS) $(CUDA_LIBS) -o $@

nvidia_dft_smoke: $(srcdir)/nvidia_dft_smoke.cu $(NVIDIA_CUDA_ARCHIVE)
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) $< $(NVIDIA_CUDA_ARCHIVE) \
	  $(CUDA_NVCC_LDFLAGS) $(CUDA_LIBS) -o $@

nvidia_dft_reduction_smoke: $(srcdir)/nvidia_dft_reduction_smoke.cu $(NVIDIA_CUDA_ARCHIVE)
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) $< $(NVIDIA_CUDA_ARCHIVE) \
	  $(CUDA_NVCC_LDFLAGS) $(CUDA_LIBS) -o $@

nvidia_flux_smoke: $(srcdir)/nvidia_flux_smoke.cu $(NVIDIA_CUDA_ARCHIVE)
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) $< $(NVIDIA_CUDA_ARCHIVE) \
	  $(CUDA_NVCC_LDFLAGS) $(CUDA_LIBS) -o $@

nvidia_cw_smoke: $(srcdir)/nvidia_cw_smoke.cu $(NVIDIA_CUDA_ARCHIVE)
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 --fmad=false $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) $< $(NVIDIA_CUDA_ARCHIVE) \
	  $(CUDA_NVCC_LDFLAGS) $(CUDA_LIBS) -o $@

if WITH_MPI
nvidia_mpi_runtime_smoke: $(srcdir)/nvidia_mpi_runtime_smoke.cu $(NVIDIA_CUDA_ARCHIVE)
	$(AM_V_GEN)$(CXX) $(CPPFLAGS) $(AM_CPPFLAGS) $(CXXFLAGS) -x c++ $< -x none \
	  $(NVIDIA_CUDA_ARCHIVE) $(CUDA_LDFLAGS) $(CUDA_LIBS) $(LIBS) -o $@
endif
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
