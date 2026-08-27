#!/bin/bash
# Build the libctl and MPB releases the bitwise CI job needs.
#
# libctl uses the release tarball and --without-guile: the git repo needs guile
# to bootstrap, and we only want libctlgeom for the Python interface.
# MPB is built WITHOUT --with-mpi on purpose -- that flag renames its outputs to
# libmpb_mpi/mpb_mpi.h, and Meep's configure only ever looks for mpb.h/-lmpb, so
# MPB support would be silently disabled. CC=mpicc is what links it against the
# MPI-enabled HDF5.
set -euo pipefail

PREFIX="${DEPS_PREFIX:-/usr/local}"
LIBCTL_VERSION="${LIBCTL_VERSION:-4.7.1}"
MPB_VERSION="${MPB_VERSION:-1.12.0}"
WORK="$(mktemp -d)"
cd "$WORK"

curl -sSLO "https://github.com/NanoComp/libctl/releases/download/v${LIBCTL_VERSION}/libctl-${LIBCTL_VERSION}.tar.gz"
tar xzf "libctl-${LIBCTL_VERSION}.tar.gz"
cd "libctl-${LIBCTL_VERSION}"
./configure --prefix="$PREFIX" --enable-shared --without-guile
make -j"$(nproc)" && sudo make install
cd "$WORK"

curl -sSLO "https://github.com/NanoComp/mpb/releases/download/v${MPB_VERSION}/mpb-${MPB_VERSION}.tar.gz"
tar xzf "mpb-${MPB_VERSION}.tar.gz"
cd "mpb-${MPB_VERSION}"
CC=mpicc ./configure --prefix="$PREFIX" --enable-shared --without-libctl --with-hermitian-eps
make -j"$(nproc)" && sudo make install
sudo ldconfig || true
