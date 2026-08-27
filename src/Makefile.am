lib_LTLIBRARIES = libmeep.la
include_HEADERS = meep.hpp
pkginclude_HEADERS = meep/mympi.hpp meep/vec.hpp meep/meep-config.h meepgeom.hpp material_data.hpp adjust_verbosity.hpp

AM_CPPFLAGS = -I$(top_srcdir)/src

BUILT_SOURCES = sphere-quad.h step_generic_stride1.cpp meep/meep-config.h

HDRS = meep.hpp meep_internals.hpp meep/mympi.hpp meep/vec.hpp	\
bicgstab.hpp meepgeom.hpp material_data.hpp adjust_verbosity.hpp	\
$(BACKEND_HDRS)

# Backend-private headers: NOT installed, and never included from meep.hpp
# (global rule 5 -- meep.hpp is the SWIG surface).
BACKEND_HDRS = backend/lifecycle.hpp backend/diagnostics.hpp \
backend/array_ref.hpp backend/halo_plan.hpp \
backend/prepare.hpp backend/storage_plan.hpp backend/classification.hpp \
backend/step_plan.hpp backend/region_plan.hpp \
backend/descriptors.hpp

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
backend/step_plan.cpp backend/cpu/cpu_executor.cpp 	\
backend/descriptors.cpp 	\
$(HDRS) $(BUILT_SOURCES)

SUBDIRS = support
libmeep_la_LIBADD = support/libsupport.la

libmeep_la_LDFLAGS = -version-info @SHARED_VERSION_INFO@

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
