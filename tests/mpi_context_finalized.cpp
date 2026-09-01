/* Copyright (C) 2005-2026 Massachusetts Institute of Technology */

#include <cstdio>
#include <string>

#include "config.h"
#include "backend/mpi_context.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

using namespace meep;

int main(int argc, char **argv) {
#ifdef HAVE_MPI
  int provided = MPI_THREAD_SINGLE;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS ||
      provided < MPI_THREAD_FUNNELED)
    return 2;
  {
    initialize externally_owned(argc, argv);
    std::string why;
    BackendCommunicatorLease lease;
    if (!create_backend_communicator_lease(lease, why)) return 3;
    int world_size = 1;
    if (MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS) return 4;
    /* Leave a real split communicator installed. The initialize destructor
       runs after externally-owned finalization and must not log, time, free,
       or otherwise enter MPI. */
    divide_parallel_processes(world_size);
    if (MPI_Finalize() != MPI_SUCCESS) return 5;
  }
  return 0;
#else
  (void)argc;
  (void)argv;
  std::printf("mpi_context_finalized: PASS (non-MPI)\n");
  return 0;
#endif
}
