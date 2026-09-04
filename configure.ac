# Process this file with autoconf to produce a configure script.

AC_INIT([meep],[m4_esyscmd(./version.sh 1.35.0-beta)])
AC_CONFIG_SRCDIR(src/step.cpp)
AC_CONFIG_AUX_DIR([build-aux])

# Shared-library version number; indicates api compatibility, and is
# not the same as the "public" version number.  (Don't worry about this
# except for public releases.) Note that any change to a C++ class
# definition (in the .hpp file) generally breaks binary compatibility.
SHARED_VERSION_INFO="39:0:0" # CURRENT:REVISION:AGE

AM_INIT_AUTOMAKE([foreign color-tests parallel-tests silent-rules 1.11])
AM_SILENT_RULES(yes)
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])
AM_MAINTAINER_MODE
AC_SUBST(SHARED_VERSION_INFO)
# need shared libraries for python interfaces
AC_ENABLE_SHARED

##############################################################################
# build in single precision

AC_ARG_ENABLE(single,
              [AS_HELP_STRING([--enable-single],[compile in single precision])],
	            enable_single=$enableval, enable_single=no)
if test "x$enable_single" = "xyes"; then
    MEEP_SINGLE=1
else
    MEEP_SINGLE=0
fi
AC_SUBST(MEEP_SINGLE)

##############################################################################
# build with SWIG threads for Python

AC_ARG_ENABLE(swig-python-threads,
              [AS_HELP_STRING([--enable-swig-python-threads],[enable SWIG threads for Python bindings])],
	            enable_swig_python_threads=$enableval, enable_swig_python_threads=no)
if test "x$enable_swig_python_threads" = "xyes"; then
    MEEP_SWIG_PYTHON_THREADS=1
else
    MEEP_SWIG_PYTHON_THREADS=0
fi
AC_SUBST(MEEP_SWIG_PYTHON_THREADS)
AM_CONDITIONAL(MEEP_SWIG_PYTHON_THREADS, test "x$enable_swig_python_threads" = "xyes")

##############################################################################
# build with SWIG debugging enabled for Python

AC_ARG_ENABLE(swig-python-debug,
              [AS_HELP_STRING([--enable-swig-python-debug],[enable SWIG debug for Python bindings])],
	            enable_swig_python_debug=$enableval, enable_swig_python_debug=no)
if test "x$enable_swig_python_debug" = "xyes"; then
    MEEP_SWIG_PYTHON_DEBUG=1
else
    MEEP_SWIG_PYTHON_DEBUG=0
fi
AC_SUBST(MEEP_SWIG_PYTHON_DEBUG)

##############################################################################

# Check for mpiCC immediately after getting C++ compiler...
AC_PROG_CC
AC_PROG_CXX
AC_LANG([C++])

# Parse accelerator selection before MPI provider checks so AMD builds can keep
# the existing CUDA-awareness query disabled until ROCm-aware MPI is added.
AC_ARG_WITH([accelerator],
            [AS_HELP_STRING([--with-accelerator=no|nvidia|amd],
                            [build an accelerator backend (default: no)])],
            [with_accelerator=$withval], [with_accelerator=no])

# Check for MPI library

AC_ARG_WITH(mpi, [AS_HELP_STRING([--with-mpi],[enable MPI parallelization])],
                  with_mpi=$withval, with_mpi=no)
AC_ARG_VAR([MPIRUN], [MPI launcher used by the test suite])

if test "x$with_mpi" = "xyes"; then
  AX_MPI([],[AC_MSG_ERROR([could not find mpi library for --with-mpi])])
  CXX=$MPICXX
  if test "x$with_accelerator" != "xamd"; then
    AC_CHECK_HEADERS([mpi-ext.h], [
      AC_LINK_IFELSE(
        [AC_LANG_PROGRAM([[#include <mpi.h>
#include <mpi-ext.h>]], [[return MPIX_Query_cuda_support() < 0;]])],
        [AC_DEFINE([HAVE_MPIX_QUERY_CUDA_SUPPORT], [1],
                   [Define when MPIX_Query_cuda_support is available])])
      AC_COMPILE_IFELSE(
        [AC_LANG_PROGRAM([[#include <mpi.h>
#include <mpi-ext.h>
#if !defined(MPIX_CUDA_AWARE_SUPPORT) && !defined(OMPI_HAVE_MPI_EXT_CUDA)
#error no Open MPI CUDA-awareness compile-time indicator
#endif]], [[]])],
        [AC_DEFINE([HAVE_OPEN_MPI_CUDA_AWARE_SUPPORT_MACRO], [1],
                   [Define when Open MPI publishes a CUDA-awareness support macro])])
    ], [], [[#include <mpi.h>]])
  fi
fi

# Require C++11 support
AX_CXX_COMPILE_STDCXX([11], [], [mandatory])

# Optional ccache support (speeds up rebuilds of SWIG-generated C++)
AC_ARG_ENABLE(ccache,
              [AS_HELP_STRING([--enable-ccache],[use ccache to speed up compilation])],
              enable_ccache=$enableval, enable_ccache=no)
if test "x$enable_ccache" != "xno"; then
    AC_CHECK_PROG(CCACHE, ccache, ccache)
    if test "x$CCACHE" != "x"; then
        CC="ccache $CC"
        CXX="ccache $CXX"
        AC_MSG_NOTICE([using ccache for compilation])
    elif test "x$enable_ccache" = "xyes"; then
        AC_MSG_ERROR([ccache requested but not found])
    fi
fi

##############################################################################
# optional accelerator backend

AC_ARG_WITH([cuda],
            [AS_HELP_STRING([--with-cuda=DIR],
                            [CUDA toolkit prefix for the NVIDIA backend])],
            [cuda_prefix=$withval], [cuda_prefix=])
AC_ARG_WITH([cuda-architectures],
            [AS_HELP_STRING([--with-cuda-architectures=LIST],
                            [comma-separated CUDA targets, e.g. sm_90,sm_100])],
            [cuda_architectures=$withval], [cuda_architectures=])
AC_ARG_WITH([cuda-host-compiler],
            [AS_HELP_STRING([--with-cuda-host-compiler=PATH],
                            [host compiler used only by nvcc])],
            [cuda_host_compiler=$withval], [cuda_host_compiler=])
AC_ARG_VAR([NVCC], [NVIDIA CUDA compiler])
AC_ARG_VAR([NVCCFLAGS], [additional NVIDIA CUDA compiler flags])

# This switch remains a private build/portability check.  Production AMD
# selection is controlled independently by --with-accelerator=amd.
AC_ARG_ENABLE([hip-portability-check],
              [AS_HELP_STRING([--enable-hip-portability-check],
                              [build the private HIP portability checks (default: no)])],
              [enable_hip_portability_check=$enableval],
              [enable_hip_portability_check=no])
AC_ARG_WITH([rocm],
            [AS_HELP_STRING([--with-rocm=DIR],
                            [ROCm toolkit prefix for AMD/HIP builds])],
            [rocm_prefix=$withval], [rocm_prefix=])
AC_ARG_WITH([hip-architectures],
            [AS_HELP_STRING([--with-hip-architectures=LIST],
                            [comma-separated HIP targets, e.g. gfx950,gfx942])],
            [hip_architectures=$withval], [hip_architectures=])
AC_ARG_VAR([HIPCC],
           [AMD HIP compiler or wrapper (overrides DIR/bin/hipcc with --with-rocm=DIR)])
AC_ARG_VAR([HIPCONFIG], [HIP installation query tool])
AC_ARG_VAR([HIPCCFLAGS], [additional AMD HIP compiler flags])
AC_ARG_VAR([HIP_PRECISE_FLAGS],
           [HIP flags for sources that disable fused multiply-add])

case "$with_accelerator" in
  no)
    have_nvidia_backend=no
    use_cuda_device_compiler=no
    use_hip_device_compiler=no
    ;;
  nvidia)
    have_nvidia_backend=yes
    use_cuda_device_compiler=yes
    use_hip_device_compiler=no
    ;;
  amd)
    have_nvidia_backend=yes
    use_cuda_device_compiler=no
    use_hip_device_compiler=yes
    ;;
  *) AC_MSG_ERROR([--with-accelerator must be no, nvidia, or amd]) ;;
esac

if test "x$use_hip_device_compiler" = "xyes" -a "x$with_mpi" = "xyes"; then
  AC_MSG_NOTICE([AMD/HIP MPI transport is staged-only; provider-direct support is disabled])
fi

if test "x$use_hip_device_compiler" = "xyes"; then
  if test -n "$cuda_prefix" -o -n "$cuda_architectures" -o \
          -n "$cuda_host_compiler"; then
    AC_MSG_ERROR([CUDA toolkit options cannot be used with --with-accelerator=amd])
  fi
fi
if test "x$use_cuda_device_compiler" = "xyes" -a \
        "x$enable_hip_portability_check" != "xyes"; then
  if test -n "$rocm_prefix" -o -n "$hip_architectures"; then
    AC_MSG_ERROR([ROCm toolkit options cannot be used with --with-accelerator=nvidia])
  fi
fi

CUDA_CPPFLAGS=
CUDA_LDFLAGS=
CUDA_NVCC_LDFLAGS=
CUDA_LIBS=
CUDA_ARCH_FLAGS=
CUDA_PIC_FLAGS=
CUDA_HOST_FLAGS=

if test "x$use_cuda_device_compiler" = "xyes"; then
  if test "x$cuda_prefix" = "xyes" -o "x$cuda_prefix" = "xno"; then
    AC_MSG_ERROR([--with-cuda requires an explicit toolkit directory])
  fi

  if test -n "$cuda_prefix"; then
    cuda_prefix=`cd "$cuda_prefix" 2>/dev/null && pwd -P`
    if test -z "$cuda_prefix"; then
      AC_MSG_ERROR([CUDA toolkit directory does not exist])
    fi
    if test -z "$NVCC"; then
      AC_PATH_PROG([NVCC], [nvcc], [], [$cuda_prefix/bin:$PATH])
    fi
  else
    AC_PATH_PROG([NVCC], [nvcc])
    if test -n "$NVCC"; then
      cuda_prefix=`cd "`dirname "$NVCC"`/.." 2>/dev/null && pwd -P`
    fi
  fi

  if test -z "$NVCC"; then
    AC_MSG_ERROR([nvcc is required for --with-accelerator=nvidia])
  fi
  case "$NVCC" in
    */*) ;;
    *) NVCC=`command -v "$NVCC" 2>/dev/null` ;;
  esac
  if test ! -x "$NVCC"; then
    AC_MSG_ERROR([NVCC is not executable: $NVCC])
  fi
  if test -z "$cuda_architectures"; then
    AC_MSG_ERROR([--with-cuda-architectures is required for the NVIDIA backend])
  fi
  if test -n "$cuda_host_compiler"; then
    case "$cuda_host_compiler" in
      */*) ;;
      *) cuda_host_compiler=`command -v "$cuda_host_compiler" 2>/dev/null` ;;
    esac
    if test ! -x "$cuda_host_compiler"; then
      AC_MSG_ERROR([CUDA host compiler is not executable: $cuda_host_compiler])
    fi
    CUDA_HOST_FLAGS="-ccbin=$cuda_host_compiler"
  fi

  cuda_arch_list=`echo "$cuda_architectures" | sed 's/,/ /g'`
  CUDA_ARCHITECTURES=
  for cuda_arch in $cuda_arch_list; do
    case "$cuda_arch" in
      sm_*) cuda_digits=`echo "$cuda_arch" | sed 's/^sm_//'` ;;
      *) cuda_digits=$cuda_arch ; cuda_arch=sm_$cuda_arch ;;
    esac
    case "$cuda_digits" in
      ''|*@<:@!0123456789@:>@*)
        AC_MSG_ERROR([invalid CUDA architecture in --with-cuda-architectures: $cuda_arch])
        ;;
    esac
    cuda_compute=compute_$cuda_digits
    CUDA_ARCH_FLAGS="$CUDA_ARCH_FLAGS -gencode=arch=$cuda_compute,code=$cuda_arch"
    if test -z "$CUDA_ARCHITECTURES"; then
      CUDA_ARCHITECTURES=$cuda_arch
    else
      CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES,$cuda_arch"
    fi
  done

  if test ! -f "$cuda_prefix/include/cuda_runtime_api.h"; then
    AC_MSG_ERROR([CUDA runtime headers not found below $cuda_prefix])
  fi
  CUDA_CPPFLAGS="-I$cuda_prefix/include"
  if test -d "$cuda_prefix/lib64"; then
    cuda_libdir=$cuda_prefix/lib64
  elif test -d "$cuda_prefix/lib"; then
    cuda_libdir=$cuda_prefix/lib
  else
    AC_MSG_ERROR([CUDA runtime library directory not found below $cuda_prefix])
  fi
  CUDA_LDFLAGS="-L$cuda_libdir -Wl,-rpath,$cuda_libdir"
  CUDA_NVCC_LDFLAGS="-L$cuda_libdir -Xlinker=-rpath -Xlinker=$cuda_libdir"
  CUDA_LIBS="-lcudart"
  CUDA_PIC_FLAGS="-Xcompiler=-fPIC"

  AC_MSG_CHECKING([whether nvcc can compile and link the CUDA runtime])
  cat > conftest.cu <<'_ACEOF'
#include <cuda_runtime_api.h>
int main() { return cudaFree(0) == cudaSuccess ? 0 : 1; }
_ACEOF
  if $NVCC $CUDA_HOST_FLAGS -std=c++11 $NVCCFLAGS $CUDA_ARCH_FLAGS \
       $CUDA_CPPFLAGS conftest.cu \
       $CUDA_NVCC_LDFLAGS $CUDA_LIBS -o conftest-cuda \
       >&AS_MESSAGE_LOG_FD; then
    AC_MSG_RESULT([yes])
  else
    rm -f conftest.cu conftest-cuda
    AC_MSG_ERROR([nvcc could not compile and link a CUDA runtime program])
  fi
  rm -f conftest.cu conftest-cuda

fi

AC_SUBST([CUDA_ARCHITECTURES])
AC_SUBST([CUDA_ARCH_FLAGS])
AC_SUBST([CUDA_CPPFLAGS])
AC_SUBST([CUDA_LDFLAGS])
AC_SUBST([CUDA_NVCC_LDFLAGS])
AC_SUBST([CUDA_LIBS])
AC_SUBST([CUDA_PIC_FLAGS])
AC_SUBST([CUDA_HOST_FLAGS])
AM_CONDITIONAL([HAVE_NVIDIA_BACKEND],
               [test "x$have_nvidia_backend" = "xyes"])
AM_CONDITIONAL([USE_CUDA_DEVICE_COMPILER],
               [test "x$use_cuda_device_compiler" = "xyes"])
AM_CONDITIONAL([USE_HIP_DEVICE_COMPILER],
               [test "x$use_hip_device_compiler" = "xyes"])

if test "x$have_nvidia_backend" = "xyes"; then
  AC_DEFINE([HAVE_NVIDIA_BACKEND], [1],
            [Define if the NVIDIA-compatible accelerator backend is enabled])
fi

##############################################################################
# AMD HIP device compiler (production and private portability checks)

case "$enable_hip_portability_check" in
  yes|no) ;;
  *) AC_MSG_ERROR([--enable-hip-portability-check accepts only yes or no]) ;;
esac

HIP_ARCHITECTURES=
HIP_ARCH_FLAGS=
HIP_CPPFLAGS=
HIP_LDFLAGS=
HIP_LIBS=
HIP_PIC_FLAGS=
if test -z "$HIP_PRECISE_FLAGS"; then
  HIP_PRECISE_FLAGS=-ffp-contract=off
fi

build_hip_device_objects=no
have_hip_graphs=no
if test "x$enable_hip_portability_check" = "xyes" -o \
        "x$use_hip_device_compiler" = "xyes"; then
  build_hip_device_objects=yes
  rocm_prefix_was_explicit=no
  hipcc_was_explicit=no
  if test -n "$HIPCC"; then
    hipcc_was_explicit=yes
  fi
  if test "x$rocm_prefix" = "xyes" -o "x$rocm_prefix" = "xno"; then
    AC_MSG_ERROR([--with-rocm requires an explicit toolkit directory])
  fi

  if test -n "$rocm_prefix"; then
    rocm_prefix_was_explicit=yes
    rocm_prefix=`cd "$rocm_prefix" 2>/dev/null && pwd -P`
    if test -z "$rocm_prefix"; then
      AC_MSG_ERROR([ROCm toolkit directory does not exist])
    fi
    if test -z "$HIPCC"; then
      HIPCC=$rocm_prefix/bin/hipcc
      if test ! -x "$HIPCC"; then
        AC_MSG_ERROR([hipcc not found below the requested ROCm toolkit: $rocm_prefix (set HIPCC to use an intentional compiler wrapper override)])
      fi
    fi
    if test -z "$HIPCONFIG"; then
      AC_PATH_PROG([HIPCONFIG], [hipconfig], [], [$rocm_prefix/bin:$PATH])
    fi
  else
    AC_PATH_PROG([HIPCC], [hipcc])
    case "$HIPCC" in
      */*) hip_bindir=`dirname "$HIPCC"` ;;
      *) hip_bindir= ;;
    esac
    if test -z "$HIPCONFIG" -a -n "$hip_bindir" -a -x "$hip_bindir/hipconfig"; then
      HIPCONFIG=$hip_bindir/hipconfig
    fi
    if test -z "$HIPCONFIG"; then
      AC_PATH_PROG([HIPCONFIG], [hipconfig])
    fi
    if test -n "$HIPCONFIG"; then
      rocm_prefix=`$HIPCONFIG --path 2>/dev/null`
    fi
  fi

  case "$HIPCC" in
    */*) ;;
    *) HIPCC=`command -v "$HIPCC" 2>/dev/null` ;;
  esac
  case "$HIPCONFIG" in
    */*) ;;
    *) HIPCONFIG=`command -v "$HIPCONFIG" 2>/dev/null` ;;
  esac
  if test ! -x "$HIPCC"; then
    AC_MSG_ERROR([hipcc is required for AMD/HIP builds])
  fi
  if test "x$rocm_prefix_was_explicit" = "xyes" -a \
          "x$hipcc_was_explicit" = "xyes"; then
    AC_MSG_NOTICE([using user-supplied HIPCC override with requested ROCm toolkit: $HIPCC])
  fi
  if test ! -x "$HIPCONFIG"; then
    AC_MSG_ERROR([hipconfig is required for AMD/HIP builds])
  fi
  hipconfig_prefix=`$HIPCONFIG --path 2>/dev/null`
  hipconfig_prefix=`cd "$hipconfig_prefix" 2>/dev/null && pwd -P`
  if test -z "$hipconfig_prefix"; then
    AC_MSG_ERROR([hipconfig did not report a valid ROCm toolkit prefix])
  fi
  if test "x$rocm_prefix_was_explicit" = "xyes" -a \
          "x$hipconfig_prefix" != "x$rocm_prefix"; then
    AC_MSG_ERROR([HIPCONFIG does not belong to the requested ROCm toolkit: $rocm_prefix])
  fi
  rocm_prefix=$hipconfig_prefix
  hip_platform=`$HIPCONFIG --platform 2>/dev/null`
  if test "x$hip_platform" != "xamd"; then
    AC_MSG_ERROR([AMD/HIP builds require the AMD HIP platform])
  fi
  if test -z "$rocm_prefix"; then
    AC_MSG_ERROR([hipconfig did not report a ROCm toolkit prefix])
  fi
  rocm_prefix=`cd "$rocm_prefix" 2>/dev/null && pwd -P`
  if test -z "$rocm_prefix"; then
    AC_MSG_ERROR([ROCm toolkit directory does not exist])
  fi
  if test -z "$hip_architectures"; then
    AC_MSG_ERROR([--with-hip-architectures is required for AMD/HIP builds])
  fi
  case "$hip_architectures" in
    ,*|*,|*,,*)
      AC_MSG_ERROR([--with-hip-architectures contains an empty architecture])
      ;;
  esac

  hip_arch_list=`echo "$hip_architectures" | sed 's/,/ /g'`
  for hip_arch in $hip_arch_list; do
    case "$hip_arch" in
      gfx*) ;;
      *) AC_MSG_ERROR([invalid HIP architecture in --with-hip-architectures: $hip_arch]) ;;
    esac
    case "$hip_arch" in
      ''|*@<:@!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:+-@:>@*)
        AC_MSG_ERROR([invalid HIP architecture in --with-hip-architectures: $hip_arch])
        ;;
    esac
    HIP_ARCH_FLAGS="$HIP_ARCH_FLAGS --offload-arch=$hip_arch"
    if test -z "$HIP_ARCHITECTURES"; then
      HIP_ARCHITECTURES=$hip_arch
    else
      HIP_ARCHITECTURES="$HIP_ARCHITECTURES,$hip_arch"
    fi
  done

  if test ! -f "$rocm_prefix/include/hip/hip_runtime_api.h"; then
    AC_MSG_ERROR([HIP runtime headers not found below $rocm_prefix])
  fi
  if test -f "$rocm_prefix/lib64/libamdhip64.so"; then
    hip_libdir=$rocm_prefix/lib64
  elif test -f "$rocm_prefix/lib/libamdhip64.so"; then
    hip_libdir=$rocm_prefix/lib
  else
    AC_MSG_ERROR([HIP runtime library not found below $rocm_prefix])
  fi
  HIP_CPPFLAGS="-DMEEP_HIP_PORTABILITY=1 -D__HIP_PLATFORM_AMD__=1 -I$rocm_prefix/include"
  HIP_LDFLAGS="-L$hip_libdir -Wl,-rpath,$hip_libdir"
  HIP_LIBS=-lamdhip64
  HIP_PIC_FLAGS=-fPIC

  if test "x$use_hip_device_compiler" = "xyes"; then
    AC_MSG_CHECKING([whether the selected HIP compiler headers support graphs])
    cat > conftest-hip-graphs.cpp <<'_ACEOF'
#include <hip/hip_version.h>
#if defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7
meep_hip_graph_headers_available
#endif
_ACEOF
    if $HIPCC -x hip -E $HIPCCFLAGS $HIP_ARCH_FLAGS $HIP_CPPFLAGS \
         conftest-hip-graphs.cpp > conftest-hip-graphs.i 2>&AS_MESSAGE_LOG_FD && \
         grep meep_hip_graph_headers_available conftest-hip-graphs.i >/dev/null 2>&1; then
      have_hip_graphs=yes
      AC_MSG_RESULT([yes])
    else
      AC_MSG_RESULT([no])
    fi
    rm -f conftest-hip-graphs.cpp conftest-hip-graphs.i
  fi

  AC_MSG_CHECKING([whether hipcc can compile and link a HIP kernel])
  cat > conftest.hip.cpp <<'_ACEOF'
#include <hip/hip_runtime.h>
__global__ void meep_hip_conftest_kernel(int *value) { *value = 1; }
int main() {
  int *value = 0;
  if (hipMalloc(&value, sizeof(*value)) != hipSuccess) return 1;
  hipLaunchKernelGGL(meep_hip_conftest_kernel, dim3(1), dim3(1), 0, 0, value);
  return hipFree(value) == hipSuccess ? 0 : 1;
}
_ACEOF
  if $HIPCC -std=c++11 $HIPCCFLAGS $HIP_ARCH_FLAGS $HIP_PIC_FLAGS \
       $HIP_CPPFLAGS conftest.hip.cpp $HIP_LDFLAGS $HIP_LIBS \
       -o conftest-hip >&AS_MESSAGE_LOG_FD; then
    AC_MSG_RESULT([yes])
  else
    rm -f conftest.hip.cpp conftest-hip
    AC_MSG_ERROR([hipcc could not compile and link a HIP kernel program])
  fi
  rm -f conftest.hip.cpp conftest-hip
fi

AC_SUBST([HIP_ARCHITECTURES])
AC_SUBST([HIP_ARCH_FLAGS])
AC_SUBST([HIP_CPPFLAGS])
AC_SUBST([HIP_LDFLAGS])
AC_SUBST([HIP_LIBS])
AC_SUBST([HIP_PIC_FLAGS])
AC_SUBST([HIP_PRECISE_FLAGS])
AM_CONDITIONAL([BUILD_HIP_PORTABILITY_CHECK],
               [test "x$enable_hip_portability_check" = "xyes"])
AM_CONDITIONAL([BUILD_HIP_DEVICE_OBJECTS],
               [test "x$build_hip_device_objects" = "xyes"])
AM_CONDITIONAL([HAVE_HIP_GRAPHS],
               [test "x$have_hip_graphs" = "xyes"])

##############################################################################
# Compiler flags

# Pick "good" compiler flags(?)
AX_CXX_MAXOPT

AC_ARG_ENABLE(debug,
              [AS_HELP_STRING([--enable-debug],[compile for debugging])],
	      enable_debug=$enableval, enable_debug=no)
if test "$enable_debug" = "yes"; then
	CFLAGS="-g"
	CXXFLAGS="-g"
	FFLAGS="-g"
	AC_DEFINE(DEBUG,1,[define to enable debugging code])
fi

# Add lots of compiler warnings in maintainer mode if we are using gcc:
# (The variable $GXX is set to "yes" by AC_PROG_CXX if we are using g++.)
if test "$GXX" = "yes" && test "$USE_MAINTAINER_MODE" = yes; then
        CXXFLAGS="$CXXFLAGS -Wall -W"
fi

# For some annoying reason, g++ requires you to compile
# all code with -march if you compile any code with -march,
# otherwise segfaults can occur (observed in g++ 3.3.5).
# We also need to be careful about -malign-double.
ARCHFLAG=""
for flg in $CXXFLAGS; do
    case $flg in
      -march=*|-mcpu=*|-malign*) ARCHFLAG="$ARCHFLAG $flg" ;;
    esac
done
AC_SUBST(ARCHFLAG)

AC_ARG_WITH(openmp, [AS_HELP_STRING([--with-openmp],[use OpenMP directives for parallelism])],
            [], [with_openmp=no])
if test x"$with_openmp" = "xyes"; then
   AC_DEFINE(HAVE_OPENMP,1,[Define to enable OpenMP])
   AX_OPENMP([CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"], [AC_MSG_ERROR([don't know how to enable OpenMP])])
fi

##############################################################################
# More checks

LT_INIT

AC_CHECK_HEADERS([dirent.h fcntl.h sys/file.h sys/stat.h sys/types.h sys/wait.h unistd.h])
AC_CHECK_FUNCS([close flock fork fstat ftruncate geteuid lseek lstat mkdir mkdtemp openat pipe waitpid write])

AC_CHECK_LIB(m, sin)

# FFTW3 is preferred; FFTW2 (dfftw/fftw) is also supported by Meep
# (see src/casimir.cpp for FFTW2 code paths).
AC_CHECK_LIB(fftw3, fftw_plan_dft_1d, [],
  [AC_CHECK_LIB(dfftw, fftw_create_plan, [],
    [AC_CHECK_LIB(fftw, fftw_create_plan, [],
       [AC_MSG_WARN([FFTW needed for MPB])])])])

###########################################################################

AC_PROG_F77
AC_F77_WRAPPERS
# $with_blas and $with_lapack are set by AX_BLAS/AX_LAPACK (--with-blas,
# --with-lapack).  When unset (the default), this block runs to detect them.
if test "x$with_lapack" != xno -o "x$with_blas" != xno; then
   AX_BLAS
   AX_LAPACK
   LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"
fi

###########################################################################
# Harminv library

# First, try pkg-config, if it is installed.  Note that, annoyingly,
# pkg-config doesn't look in /usr/local by default, so we have to
# add this to the path for that common case.  It also doesn't give
# us a way to print its default path, grr, so we have to assume that this
# is /usr/lib/pkgconfig.
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/lib/pkgconfig:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
PKG_CHECK_MODULES(HARMINV, harminv >= 1.1, [CFLAGS="$CFLAGS $HARMINV_CFLAGS"; LIBS="$HARMINV_LIBS $LIBS"; have_harminv=yes], [have_harminv=no])

# As a fallback, check manually for BLAS/LAPACK and harminv libraries:
if test $have_harminv = no; then
  if test "x$ax_lapack_ok" = x; then
    AX_BLAS
    AX_LAPACK([], [AC_MSG_WARN([BLAS/LAPACK needed for harminv])])
    LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"
  elif test "x$ax_lapack_ok" = xno; then
    AC_MSG_WARN([BLAS/LAPACK needed for harminv])
  fi

  save_CC=$CC
  CC=$CXX # harminv test must use $CXX since harminv may be a C++ lib

	# just use lapack/blas in default libs since we always need them
	AC_CHECK_LIB(harminv, harminv_get_freq_error, [have_harminv=yes; LIBS="-lharminv $LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"], [AC_MSG_WARN([harminv support is disabled])], [$LAPACK_LIBS $BLAS_LIBS $FLIBS])

	CC=$save_CC
fi

if test $have_harminv = yes; then
	AC_DEFINE([HAVE_HARMINV],[1],[Define if you have libharminv])
fi

###########################################################################
# MPB library

AC_CHECK_HEADER(mpb.h, [have_mpb=maybe], [have_mpb=no])

if test $have_mpb = maybe; then
  if test "x$ax_lapack_ok" = x -a "x$have_harminv" = xno; then
    AX_BLAS
    AX_LAPACK([], [AC_MSG_WARN([BLAS/LAPACK needed for MPB])])
          LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"
  elif test "x$ax_lapack_ok" = xno; then
    AC_MSG_WARN([BLAS/LAPACK needed for MPB])
  fi
  AC_CHECK_LIB(mpb, maxwell_target_operator1, [have_mpb=yes; LIBS="-lmpb $LIBS"],
               [AC_CHECK_LIB(mpb, maxwell_target_operator1,
                   [have_mpb=yes; LIBS="-lmpb $LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"], [AC_MSG_WARN([MPB support is disabled])], [$LAPACK_LIBS $BLAS_LIBS $FLIBS])])
  if test $have_mpb = yes; then
    AC_DEFINE([HAVE_MPB], [1], [Define if you have libmpb])
  fi
fi

AM_CONDITIONAL(WITH_MPB, test "x$have_mpb" = "xyes")
if test "x$have_mpb" = "xyes"; then
  # check for mpb_printf_callback
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <mpb.h>]], [mpb_printf_callback = 0;])],
    AC_DEFINE([HAVE_MPB_PRINTF_CALLBACK], [1], [If we have the mpb_printf_callback variable]))
fi

##############################################################################
# GNU Scientific Library

AC_CHECK_LIB([gslcblas],[cblas_dgemm])
AC_CHECK_LIB([gsl],[gsl_sf_bessel_Jn], [],
	     [AC_MSG_WARN([Missing GNU GSL library...Bessel-function field initialization will not be supported.])])

##############################################################################
# Check for HDF5 library
AC_ARG_WITH(hdf5, [AS_HELP_STRING([--without-hdf5],[do not allow HDF5 output])], ok=$withval, ok=yes)
if test "$ok" = "yes"; then
    AC_CHECK_LIB(z,deflate, [],[AC_MSG_WARN([zlib is required for HDF5!])])

    save_CPPFLAGS="$CPPFLAGS"
    save_LIBS="$LIBS"

    # Check whether linking with "-lhdf5" just works (installation in default
    # location or due to user-provided flags). If not, try to locate HDF5 via
    # pkg-config.  Optimistically update LIBS and CPPFLAGS. Revert if any
    # AC_CHECKs fail.
    AC_CHECK_LIB(hdf5, H5Pcreate, [], [
        PKG_CHECK_MODULES(HDF5, hdf5, [
            CPPFLAGS="$CPPFLAGS $HDF5_CFLAGS"
            LIBS="$HDF5_LIBS $LIBS"
        ], [])
    ])

    AC_CHECK_FUNC(H5Pcreate, [
        AC_CHECK_HEADER(hdf5.h, [
            AC_DEFINE(HAVE_HDF5,1,[Define if we have & link HDF5])
            AC_CHECK_FUNCS(H5Pset_mpi H5Pset_fapl_mpio)
        ])
    ])

    if test "x$ac_cv_header_hdf5_h" != "xyes"; then
        AC_MSG_WARN([Couldn't find the HDF5 header or library!!  Switching to --without-hdf5.])
        CPPFLAGS="$save_CPPFLAGS"
        LIBS="$save_LIBS"
    fi
fi

##############################################################################

RUNCODE=""
if test "x$with_mpi" = "xyes"; then
   AC_CHECK_PROG(MPIRUN, mpirun, mpirun)
   # always use 2 processes for 'make check'
   RUNCODE="$MPIRUN -np 2"
fi
if test x"$with_openmp" = "xyes"; then
   # use 2 openmp threads
   RUNCODE="env OMP_NUM_THREADS=2 $RUNCODE"
fi
AC_SUBST(RUNCODE)
AM_CONDITIONAL(WITH_MPI, test "x$with_mpi" = "xyes")

##############################################################################
# Libraries and flags (other than -lmeep) required to link Meep:

MEEPLIBS="$LDFLAGS $LIBS"
AC_SUBST(MEEPLIBS)

##############################################################################

AC_ARG_WITH(libctl, [AS_HELP_STRING([--with-libctl=<dir>],[specify libctl directory])], with_libctl=$withval, with_libctl=yes)
if test "x$with_libctl" = "xno"; then
        :
elif test "x$with_libctl" != "xyes"; then
	LIBCTL_DIR="$with_libctl"
fi

AM_CONDITIONAL(WITH_LIBCTL, test "x$with_libctl" != xno)

AC_ARG_WITH(scheme, [AS_HELP_STRING([--without-scheme],[compile without Scheme interface])],
            with_scheme=$withval, with_scheme=yes)
AM_CONDITIONAL(WITH_SCHEME, test x"$with_scheme" = "xyes")

##############################################################################
# Check for Guile library and its behavior (for libctl front end):

if test "x$with_libctl" != xno -a "x$with_scheme" = "xyes"; then

save_LIBS=$LIBS
save_CPPFLAGS=$CPPFLAGS

AC_CHECK_PROG(GUILE_CONFIG, guile-config, guile-config, unknown)
if test "x$GUILE_CONFIG" = "xunknown"; then
  AC_CHECK_LIB(readline, readline)
  AC_CHECK_LIB(dl, dlopen)
  AC_CHECK_LIB(guile-ltdl, scm_lt_dlinit, [], [AC_CHECK_LIB(ltdl, dlopen)])
  AC_CHECK_LIB(guile, gh_eval_str, [],
    [AC_CHECK_LIB(guile, scm_eval_string, [],
    [AC_MSG_ERROR([linking to guile failed])])])
  GUILE_LIBS="-lguile -lguile-ltdl -lreadline"
  GUILE_CPPFLAGS=""
else
  GUILE_LIBS=`$GUILE_CONFIG link`
  GUILE_CPPFLAGS=`$GUILE_CONFIG compile`
  CPPFLAGS="$CPPFLAGS $GUILE_CPPFLAGS"
  LIBS="$GUILE_LIBS $LIBS"
  AC_MSG_CHECKING([if linking to guile works])
  AC_TRY_LINK_FUNC(gh_enter, AC_MSG_RESULT(yes),
    [AC_TRY_LINK_FUNC(scm_boot_guile, AC_MSG_RESULT(yes),
    [AC_MSG_RESULT(no)
     AC_MSG_ERROR([guile-config is broken])])])
fi
AC_CHECK_HEADERS([libguile.h guile/gh.h])

# Check how smob types work in this Guile version:

AC_CHECK_FUNCS(scm_make_smob_type)
AC_MSG_CHECKING([for SCM_SMOB_PREDICATE])
AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <guile/gh.h>
#include <libguile/smob.h>]], [[int x; SCM_SMOB_PREDICATE(1,x);]])],[ok=yes],[ok=no])
AC_MSG_RESULT($ok)
if test $ok = yes; then
	AC_DEFINE(HAVE_SCM_SMOB_PREDICATE, 1, [define if we have SCM_SMOB_PREDICATE])
fi
AC_MSG_CHECKING([for SCM_SMOB_DATA])
AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <guile/gh.h>
#include <libguile/smob.h>]], [[int x; SCM_SMOB_DATA(x);]])],[ok=yes],[ok=no])
AC_MSG_RESULT($ok)
if test $ok = yes; then
	AC_DEFINE(HAVE_SCM_SMOB_DATA, 1, [define if we have SCM_SMOB_DATA])
fi
AC_MSG_CHECKING([for SCM_NEWSMOB])
AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <guile/gh.h>
#include <libguile/smob.h>]], [[int x; SCM_NEWSMOB(x,1,0);]])],[ok=yes],[ok=no])
AC_MSG_RESULT($ok)
if test $ok = yes; then
	AC_DEFINE(HAVE_SCM_NEWSMOB, 1, [define if we have SCM_NEWSMOB])
fi

# Check how to activate Guile readline support:

ACTIVATE_READLINE=""
if test "x$cross_compiling" = xno; then
AC_MSG_CHECKING(how to activate readline in Guile)
ractivate="(use-modules (readline-activator)) (activate-readline)"
if guile -c "$ractivate" > /dev/null 2>&1; then
	AC_MSG_RESULT(readline-activator)
	ACTIVATE_READLINE="$ractivate"
else
	ractivate="(use-modules (ice-9 readline)) (activate-readline)"
	if guile -c "$ractivate" >& /dev/null; then
	        AC_MSG_RESULT(ice-9 readline)
	        ACTIVATE_READLINE="$ractivate"
	else
		AC_MSG_RESULT(cannot)
		ACTIVATE_READLINE=""
	fi
fi
fi
AC_SUBST(ACTIVATE_READLINE)

AC_SUBST(GUILE_LIBS)
AC_SUBST(GUILE_CPPFLAGS)
LIBS=$save_LIBS
CPPFLAGS=$save_CPPFLAGS

##############################################################################
# Check for libctl library and files

save_LIBS=$LIBS
save_CPPFLAGS=$CPPFLAGS
LIBS="$GUILE_LIBS $LIBS"
CPPFLAGS="$CPPFLAGS $GUILE_CPPFLAGS"

AC_MSG_CHECKING([for libctl dir])
if test x != x"$LIBCTL_DIR" -a -r "$LIBCTL_DIR/share/libctl/base/ctl.scm"; then
	LIBCTL_DIR="$LIBCTL_DIR/share/libctl"
fi
if test x != x"$LIBCTL_DIR" -a ! -r "$LIBCTL_DIR/base/ctl.scm"; then
	LIBCTL_DIR=""
fi
if test x = x"$LIBCTL_DIR" -a -r /usr/local/share/libctl/base/ctl.scm; then
	LIBCTL_DIR="/usr/local/share/libctl"
fi
if test x = x"$LIBCTL_DIR" -a -r /usr/share/libctl3/base/ctl.scm; then
        LIBCTL_DIR="/usr/share/libctl3"
fi
if test x = x"$LIBCTL_DIR" -a -r /usr/share/libctl/base/ctl.scm; then
        LIBCTL_DIR="/usr/share/libctl"
fi
if test x = x"$LIBCTL_DIR"; then
	AC_MSG_ERROR([could not find libctl files; use --with-libctl=<dir>])
fi
AC_MSG_RESULT($LIBCTL_DIR)
AC_SUBST(LIBCTL_DIR)

AC_CHECK_PROGS(GEN_CTL_IO, gen-ctl-io, echo)
if test x"$GEN_CTL_IO" = xecho; then
	AC_MSG_ERROR([could not find gen-ctl-io program; check your PATH])
fi

# check for -lctl:
AC_CHECK_LIB(ctl, ctl_get_vector3, [], [AC_MSG_ERROR([Couldn't find the required libctl library.])])

# check for ctl.h (Fedora puts it in /usr/include/ctl, grrr)
CTL_H_CPPFLAG=""
AC_CHECK_HEADER(ctl.h, [CTL_H_CPPFLAG=" "], [
  save_CPPFLAGS=$CPPFLAGS
  AC_MSG_CHECKING([for alternate ctl.h location])
  for ctl_h_path in /usr/include/ctl /usr/local/include/ctl; do
     CPPFLAGS="$save_CPPFLAGS -I$ctl_h_path"
     AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <ctl.h>]],[])], [CTL_H_CPPFLAG=-I$ctl_h_path; break])
  done
  AC_MSG_RESULT([$CTL_H_CPPFLAG])
  CPPFLAGS=$save_CPPFLAGS])
if test "x$CTL_H_CPPFLAG" = "x"; then
  AC_MSG_ERROR([Couldn't find the ctl.h header file for libctl.])
fi
AC_SUBST(CTL_H_CPPFLAG)

save_CPPFLAGS=$CPPFLAGS
CPPFLAGS="$CPPFLAGS $CTL_H_CPPFLAG"
# Check libctl version >= LIBCTL_MAJOR.LIBCTL_MINOR.LIBCTL_BUGFIX
LIBCTL_MAJOR=4; LIBCTL_MINOR=2; LIBCTL_BUGFIX=0
AC_MSG_CHECKING([whether libctl version is at least ${LIBCTL_MAJOR}.${LIBCTL_MINOR}.${LIBCTL_BUGFIX}])
AC_EGREP_CPP(yes, [[
#include <ctl.h>
#if LIBCTL_MAJOR_VERSION > $LIBCTL_MAJOR || (LIBCTL_MAJOR_VERSION == $LIBCTL_MAJOR && (LIBCTL_MINOR_VERSION > $LIBCTL_MINOR || (LIBCTL_MINOR_VERSION == $LIBCTL_MINOR && LIBCTL_BUGFIX_VERSION >= $LIBCTL_BUGFIX)))
  yes
#endif
]], [AC_MSG_RESULT(ok)], [AC_MSG_ERROR([libctl version ${LIBCTL_MAJOR}.${LIBCTL_MINOR}.${LIBCTL_BUGFIX} or later is required])])
CPPFLAGS=$save_CPPFLAGS

# Check for libctl_quiet feature (in libctl 3.1.1 or later):
AC_MSG_CHECKING([for libctl_quiet feature])
if grep libctl_quiet "$LIBCTL_DIR/base/main.c" 2>&1 > /dev/null; then
   AC_MSG_RESULT(yes)
   AC_DEFINE([HAVE_LIBCTL_QUIET], [1], [If we have the libctl_quiet variable])
else
   AC_MSG_RESULT(no)
fi

LIBS=$save_LIBS
CPPFLAGS=$save_CPPFLAGS

LIBCTL_LIBS="-lctl $GUILE_LIBS"
LIBCTL_CPPFLAGS="$GUILE_CPPFLAGS"

fi # if with_libctl && with_scheme

##############################################################################

AC_SUBST(LIBCTL_LIBS)
AC_SUBST(LIBCTL_CPPFLAGS)

##############################################################################

# check for -lctlgeom:
AC_CHECK_LIB(ctlgeom, vector3_dot, [have_libctlgeom=yes; LIBS="-lctlgeom $LIBS"], [have_libctlgeom=no])
AM_CONDITIONAL(WITH_LIBCTLGEOM, test x"$have_libctlgeom" = "xyes")
# check for ctl_printf_callback
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <ctlgeom.h>]], [ctl_printf_callback = 0;])],
  AC_DEFINE([HAVE_CTL_PRINTF_CALLBACK], [1], [If we have the ctl_printf_callback variable]))
# check for make_mesh_with_center (mesh geometry support, libctl >= 4.7.0)
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <ctlgeom.h>]], [void *p = (void *) make_mesh_with_center; (void) p;])],
  AC_DEFINE([HAVE_MAKE_MESH_WITH_CENTER], [1], [If we have make_mesh_with_center in libctl (mesh geometry support)]))

##############################################################################
# check for libGDSII
AC_CHECK_HEADER(libGDSII.h, [have_gdsii=maybe], [have_gdsii=no])
if test "x$have_gdsii" = xmaybe; then
  AC_CHECK_LIB(GDSII, libGDSIIExists)
  if test "x$ac_cv_lib_GDSII_libGDSIIExists" = xyes; then
    AC_MSG_CHECKING([for libGDSII::GetLayers])
    have_gdsii_getlayers=no
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <libGDSII.h>]], [libGDSII::GetLayers("foo")])],
      [have_gdsii_getlayers=yes
       AC_DEFINE([HAVE_GDSII_GETLAYERS], [1], [If we have libGDSII::GetLayers])])
    AC_MSG_RESULT($have_gdsii_getlayers)
  fi
fi
AM_CONDITIONAL(WITH_LIBGDSII, test "x$ac_cv_lib_GDSII_libGDSIIExists" = "xyes")

##############################################################################
# The following function is used only for debugging.  Note that
# we must test for it *after* setting the compiler flags (which
# affect whether it is declared, as it is a GNU extension).
# We need to #include <stdio.h> because that somehow affects whether
# the function is declared with older gcc versions.  We need
# to use AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]], [[]])],[],[]) because the test in AC_HAVE_DECL seems
# to be optimized out.

AC_CHECK_FUNCS(feenableexcept) dnl GNU libc fp exception control function
AC_MSG_CHECKING([whether feenableexcept declaration is usable])
feenableexcept_decl_ok=yes
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <stdio.h>
#define _GNU_SOURCE 1
#include <fenv.h>]], [[feenableexcept(0);]])],[AC_DEFINE([HAVE_DECL_FEENABLEEXCEPT],[1],[Define if fenv.h declares this.])],[feenableexcept_decl_ok=no])
AC_MSG_RESULT($feenableexcept_decl_ok)

##############################################################################
# See if we need to catch SIGFPE to avoid crashing on underflow exceptions
# etcetera, since some stupid operating systems (e.g. Alpha/Tru64) crash on
# SIGFPE by default.

AC_MSG_CHECKING([whether to catch and ignore SIGFPE signals])
AC_RUN_IFELSE([AC_LANG_PROGRAM([], [volatile double x=1,y=0; return x/y;])],
              [ignore_sigfpe=no], [ignore_sigfpe=yes], [ignore_sigfpe=no])
AC_MSG_RESULT($ignore_sigfpe)
if test $ignore_sigfpe = yes; then
    AC_DEFINE(IGNORE_SIGFPE, [1], [Define to catch and ignore SIGFPE signals])
fi

##############################################################################
# Miscellaneous function and header checks
AC_CHECK_HEADERS([sys/time.h immintrin.h])
AC_CHECK_FUNCS([BSDgettimeofday gettimeofday cblas_ddot cblas_daxpy jn])

##############################################################################
# check for restrict keyword in C++
AC_C_RESTRICT

# check for C99 _Pragma support, so that we can emit pragmas in macros
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]], [[_Pragma("ivdep")]])],[],[AC_DEFINE([_Pragma],[],[define to nothing if C99 _Pragma is not supported])])

##############################################################################

# checks for python
##############################################################################

AC_ARG_WITH(python,
	[AS_HELP_STRING([--without-python],[compile without Python interface])],
        with_python=$withval,with_python=yes)

AC_CHECK_PROG(SWIG, swig, swig, echo)
if test "$USE_MAINTAINER_MODE" = yes && (test "x$with_python" = "xyes" || test "x$with_scheme" = "xyes") && test "x$SWIG" = xecho; then
  AC_MSG_ERROR([SWIG not found; configure --without-python --without-scheme or use a release tarball])
fi

AC_ARG_WITH(coverage, [AS_HELP_STRING([--with-coverage],[enable Python coverage tests])],
            with_coverage=$withval, with_coverage=no)

if test "x$with_python" = xno; then
  have_python=no
else
  if test "x$have_libctlgeom" = xno; then
    AC_MSG_ERROR([libctlgeom was not found, is required for Python interface])
  fi

  if test "$enable_shared" = no; then
     AC_MSG_WARN([Python interface requires --enable-shared; disabling])
     have_python=no
  else

    AM_PATH_PYTHON([],[have_python=yes],[have_python=no])
    if test $have_python = yes; then

      AC_MSG_CHECKING([for Python include directory])
      pinc=`echo "import sysconfig; print(sysconfig.get_path('include'))" | $PYTHON - 2>/dev/null`
      AC_MSG_RESULT([${pinc:-unknown}])
      test -n "$pinc" && PYTHON_INCLUDES="-I$pinc"
      save_CPPFLAGS=$CPPFLAGS
      CPPFLAGS="$CPPFLAGS $PYTHON_INCLUDES"
      AC_CHECK_HEADER([Python.h], [], [AC_MSG_WARN([disabling Python wrappers])
                                       have_python=no])

      if test $have_python = yes; then
        AC_MSG_CHECKING([for Numpy include directory])
        pinc=`echo "import numpy; print (numpy.get_include())" | $PYTHON - 2>/dev/null`
        AC_MSG_RESULT([${pinc:-unknown}])
        test -n "$pinc" && PYTHON_INCLUDES="$PYTHON_INCLUDES -I$pinc"
        CPPFLAGS="$save_CPPFLAGS $PYTHON_INCLUDES"
        AC_CHECK_HEADER([numpy/arrayobject.h],[],[
          AC_MSG_WARN([disabling Python wrappers])
          have_python=no],[#include <Python.h>])

        if test x"$with_coverage" = xyes; then
          AC_MSG_CHECKING([for coverage module])
          $PYTHON -c 'import coverage' 2>/dev/null
          if test $? = 0; then
            AC_MSG_RESULT([yes])
          else
            AC_MSG_RESULT([no])
            with_coverage=no
          fi
        fi
      fi

      CPPFLAGS=$save_CPPFLAGS
    fi # have_python

  fi # enabled_shared
fi # with_python

AC_SUBST(PYTHON_INCLUDES)
AM_CONDITIONAL(WITH_PYTHON, test x"$have_python" = "xyes")
AM_CONDITIONAL(WITH_COVERAGE, test x"$with_coverage" = "xyes")

if test "x$with_scheme" = xyes; then
   # Copy/symlink casimir.scm and materials.scm to builddir for out-of-tree builds
   AC_CONFIG_LINKS(scheme/casimir.scm:scheme/casimir.scm)
   AC_CONFIG_LINKS(scheme/materials.scm:scheme/materials.scm)
   AC_CONFIG_FILES([scheme/Makefile scheme/meep.scm])
fi

if test "x$with_python" = xyes; then
   AC_CONFIG_FILES([libpympb/Makefile python/Makefile])
fi

AC_CONFIG_FILES([
	Makefile
	meep-pkgconfig
	src/Makefile
	src/support/Makefile
	tests/Makefile
  src/meep/meep-config.h
])

AC_OUTPUT
