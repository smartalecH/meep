lib_LTLIBRARIES = libmeep.la
include_HEADERS = meep.hpp
pkginclude_HEADERS = meep/mympi.hpp meep/vec.hpp meep/meep-config.h meepgeom.hpp material_data.hpp adjust_verbosity.hpp

AM_CPPFLAGS = -I$(top_srcdir)/src -I$(top_builddir)/src

BUILT_SOURCES = sphere-quad.h step_generic_stride1.cpp meep/meep-config.h

HDRS = meep.hpp meep_internals.hpp meep/mympi.hpp meep/vec.hpp	\
bicgstab.hpp meepgeom.hpp material_data.hpp adjust_verbosity.hpp	\
$(BACKEND_HDRS)

# Backend-private headers: NOT installed, and never included from meep.hpp
# (global rule 5 -- meep.hpp is the SWIG surface).
BACKEND_HDRS = backend/lifecycle.hpp backend/diagnostics.hpp \
backend/random_state.hpp backend/array_ref.hpp backend/halo_plan.hpp \
backend/prepare.hpp backend/storage_plan.hpp backend/classification.hpp \
backend/material_callback.hpp backend/material_ir.hpp backend/material_recipe.hpp \
backend/step_plan.hpp backend/region_plan.hpp \
backend/descriptors.hpp backend/backend.hpp \
backend/precision.hpp backend/initialization_plan.hpp backend/cpu/cpu_backend.hpp

EXTRA_DIST = backend/nvidia/runtime.hpp backend/nvidia/runtime.cu \
backend/nvidia/arena.hpp backend/nvidia/arena.cu \
backend/nvidia/nvidia_step.hpp backend/nvidia/nvidia_step.cu \
backend/nvidia/nvidia_coordinates.hpp backend/nvidia/nvidia_coordinates.cu \
backend/nvidia/nvidia_dft.hpp backend/nvidia/nvidia_dft.cu \
backend/nvidia/nvidia_flux.hpp backend/nvidia/nvidia_flux.cu \
backend/nvidia/nvidia_multilevel.hpp backend/nvidia/nvidia_multilevel.cu \
backend/nvidia/nvidia_sources.hpp backend/nvidia/nvidia_sources.cu \
backend/nvidia/nvidia_polarization.hpp backend/nvidia/nvidia_polarization.cu \
backend/nvidia/nvidia_magnetic.hpp backend/nvidia/nvidia_magnetic.cu \
backend/nvidia/nvidia_materials.hpp backend/nvidia/nvidia_materials.cu \
backend/nvidia/nvidia_cw.hpp backend/nvidia/nvidia_cw.cu \
backend/nvidia/nvidia_backend.hpp backend/nvidia/nvidia_backend.cpp

libmeep_la_SOURCES = array_slice.cpp anisotropic_averaging.cpp 		\
bands.cpp boundaries.cpp bicgstab.cpp casimir.cpp 	\
cw_fields.cpp dft.cpp dft_ldos.cpp energy_and_flux.cpp 	\
fields.cpp fields_dump.cpp fix_boundary_sources.cpp loop_in_chunks.cpp h5fields.cpp h5file.cpp 	\
initialize.cpp integrate.cpp integrate2.cpp material_data.cpp monitor.cpp mympi.cpp 	\
multilevel-atom.cpp near2far.cpp output_directory.cpp random.cpp 	\
sources.cpp step.cpp step_db.cpp stress.cpp structure.cpp structure_dump.cpp		\
susceptibility.cpp time.cpp update_eh.cpp mpb.cpp update_pols.cpp 	\
vec.cpp step_generic.cpp meepgeom.cpp GDSIIgeom.cpp 	\
backend/lifecycle.cpp backend/diagnostics.cpp backend/halo_plan.cpp 	\
backend/prepare.cpp backend/storage_plan.cpp backend/classification.cpp 	\
backend/material_ir.cpp backend/material_recipe.cpp 	\
backend/step_plan.cpp backend/cpu/cpu_executor.cpp 	\
backend/descriptors.cpp backend/precision.cpp 	\
backend/initialization_plan.cpp backend/cpu/cpu_backend.cpp backend/cpu/backend_access.cpp 	\
$(HDRS) $(BUILT_SOURCES)

SUBDIRS = support
libmeep_la_LIBADD = support/libsupport.la
libmeep_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@
noinst_LTLIBRARIES =

if HAVE_NVIDIA_BACKEND
NVIDIA_CUDA_OBJECTS = backend/nvidia/runtime.cuda.lo backend/nvidia/arena.cuda.lo \
backend/nvidia/nvidia_step.cuda.lo backend/nvidia/nvidia_coordinates.cuda.lo \
backend/nvidia/nvidia_dft.cuda.lo \
backend/nvidia/nvidia_flux.cuda.lo \
backend/nvidia/nvidia_multilevel.cuda.lo \
backend/nvidia/nvidia_magnetic.cuda.lo \
backend/nvidia/nvidia_initialization.cuda.lo \
backend/nvidia/nvidia_materials.cuda.lo \
backend/nvidia/nvidia_sources.cuda.lo backend/nvidia/nvidia_polarization.cuda.lo
NVIDIA_CUDA_OBJECTS += backend/nvidia/nvidia_cw.cuda.lo
NVIDIA_CUDA_PIC_OBJECTS = backend/nvidia/.libs/runtime.cuda.o backend/nvidia/.libs/arena.cuda.o \
backend/nvidia/.libs/nvidia_step.cuda.o backend/nvidia/.libs/nvidia_coordinates.cuda.o \
backend/nvidia/.libs/nvidia_dft.cuda.o \
backend/nvidia/.libs/nvidia_flux.cuda.o \
backend/nvidia/.libs/nvidia_multilevel.cuda.o \
backend/nvidia/.libs/nvidia_magnetic.cuda.o \
backend/nvidia/.libs/nvidia_initialization.cuda.o \
backend/nvidia/.libs/nvidia_materials.cuda.o \
backend/nvidia/.libs/nvidia_sources.cuda.o backend/nvidia/.libs/nvidia_polarization.cuda.o \
backend/nvidia/.libs/nvidia_cw.cuda.o
NVIDIA_CUDA_ARCHIVE = libmeep_nvidia_runtime.la
noinst_LTLIBRARIES += $(NVIDIA_CUDA_ARCHIVE)
libmeep_la_SOURCES += backend/nvidia/nvidia_backend.cpp
libmeep_nvidia_runtime_la_SOURCES =
libmeep_nvidia_runtime_la_DEPENDENCIES = $(NVIDIA_CUDA_OBJECTS)
libmeep_nvidia_runtime_la_LIBADD = $(NVIDIA_CUDA_OBJECTS)
libmeep_la_LIBADD += $(NVIDIA_CUDA_ARCHIVE) $(CUDA_LIBS)
libmeep_la_LDFLAGS += $(CUDA_LDFLAGS)

backend/nvidia/runtime.cuda.lo: $(srcdir)/backend/nvidia/runtime.cu \
$(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/arena.cuda.lo: $(srcdir)/backend/nvidia/arena.cu \
$(srcdir)/backend/nvidia/arena.hpp $(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "# PIC object generated by nvcc; no non-PIC variant is built."; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/nvidia_initialization.cuda.lo: \
$(srcdir)/backend/nvidia/nvidia_initialization.cu \
$(srcdir)/backend/nvidia/nvidia_initialization.hpp \
$(srcdir)/backend/nvidia/nvidia_step.hpp $(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 --fmad=false $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/nvidia_materials.cuda.lo: \
$(srcdir)/backend/nvidia/nvidia_materials.cu \
$(srcdir)/backend/nvidia/nvidia_materials.hpp \
$(srcdir)/backend/material_geometry_numeric.hpp \
$(srcdir)/backend/nvidia/nvidia_step.hpp $(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 --fmad=false $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/nvidia_step.cuda.lo: $(srcdir)/backend/nvidia/nvidia_step.cu \
$(srcdir)/backend/nvidia/nvidia_step.hpp $(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "# PIC object generated by nvcc; no non-PIC variant is built."; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/nvidia_coordinates.cuda.lo: $(srcdir)/backend/nvidia/nvidia_coordinates.cu \
$(srcdir)/backend/nvidia/nvidia_coordinates.hpp $(srcdir)/backend/nvidia/nvidia_step.hpp \
$(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 --fmad=false $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "# PIC object generated by nvcc; no non-PIC variant is built."; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/nvidia_magnetic.cuda.lo: $(srcdir)/backend/nvidia/nvidia_magnetic.cu \
$(srcdir)/backend/nvidia/nvidia_magnetic.hpp $(srcdir)/backend/nvidia/nvidia_step.hpp \
$(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 --fmad=false $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/nvidia_dft.cuda.lo: $(srcdir)/backend/nvidia/nvidia_dft.cu \
$(srcdir)/backend/nvidia/nvidia_dft.hpp $(srcdir)/backend/nvidia/nvidia_step.hpp \
$(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "# PIC object generated by nvcc; no non-PIC variant is built."; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/nvidia_flux.cuda.lo: $(srcdir)/backend/nvidia/nvidia_flux.cu \
$(srcdir)/backend/nvidia/nvidia_flux.hpp $(srcdir)/backend/nvidia/nvidia_step.hpp \
$(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "# PIC object generated by nvcc; no non-PIC variant is built."; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/nvidia_multilevel.cuda.lo: $(srcdir)/backend/nvidia/nvidia_multilevel.cu \
$(srcdir)/backend/nvidia/nvidia_multilevel.hpp $(srcdir)/backend/nvidia/nvidia_step.hpp \
$(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 --fmad=false $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "# PIC object generated by nvcc; no non-PIC variant is built."; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/nvidia_sources.cuda.lo: $(srcdir)/backend/nvidia/nvidia_sources.cu \
$(srcdir)/backend/nvidia/nvidia_sources.hpp $(srcdir)/backend/nvidia/nvidia_step.hpp \
$(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "# PIC object generated by nvcc; no non-PIC variant is built."; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/nvidia_polarization.cuda.lo: $(srcdir)/backend/nvidia/nvidia_polarization.cu \
$(srcdir)/backend/nvidia/nvidia_polarization.hpp $(srcdir)/backend/nvidia/nvidia_step.hpp \
$(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 --fmad=false $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "# PIC object generated by nvcc; no non-PIC variant is built."; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

backend/nvidia/nvidia_cw.cuda.lo: $(srcdir)/backend/nvidia/nvidia_cw.cu \
$(srcdir)/backend/nvidia/nvidia_cw.hpp $(srcdir)/backend/nvidia/nvidia_step.hpp \
$(srcdir)/backend/nvidia/runtime.hpp
	$(AM_V_at)$(MKDIR_P) $(dir $@).libs
	$(AM_V_GEN)$(NVCC) $(CUDA_HOST_FLAGS) -std=c++11 --fmad=false $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) \
	  $(CUDA_PIC_FLAGS) $(CUDA_CPPFLAGS) $(AM_CPPFLAGS) -c $< \
	  -o $(dir $@).libs/$(notdir $(@:.lo=.o))
	$(AM_V_at){ \
	  echo "# $@ - a libtool object file"; \
	  echo "# Generated by libtool"; \
	  echo "#"; \
	  echo "pic_object='.libs/$(notdir $(@:.lo=.o))'"; \
	  echo "non_pic_object='none'"; \
	} > $@

CLEANFILES = $(NVIDIA_CUDA_OBJECTS) $(NVIDIA_CUDA_PIC_OBJECTS)
endif

PRELUDE = "/* This file was automatically generated --- DO NOT EDIT */"

noinst_PROGRAMS = sphere_quad
sphere_quad_SOURCES = sphere-quad.cpp
SPHERE_QUAD = $(top_builddir)/src/sphere_quad
sphere-quad.h:
	make sphere_quad
	(echo $(PRELUDE); echo; $(SPHERE_QUAD)) > $@

step_generic_stride1.cpp: step_generic.cpp
	(echo $(PRELUDE); echo; \
	 sed -e 's/LOOP_OVER/S1LOOP_OVER/g' \
	     -e 's/step_curl/step_curl_stride1/g' \
	     -e 's/step_update_EDHB/step_update_EDHB_stride1/g' \
	     -e 's/step_beta/step_beta_stride1/g' \
	     -e 's/step_bfast/step_bfast_stride1/g' \
	     $(top_srcdir)/src/step_generic.cpp) > $@

MAINTAINERCLEANFILES = $(BUILT_SOURCES)
