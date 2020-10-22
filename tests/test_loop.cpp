/*
Steps:
    [x] Set up initial benchmark framework
    [ ] Set up initial iterator class
    [ ] Fill in iterator class
    [ ] Debug iterator class
    [ ] Parallelize iterator class
    [ ] Debug parallel version
    [ ] Integrate into main repo
    [ ] Finalize test
    [ ] Try different vectorizations and do basic timing
    [ ] Set up 3 actual test scripts
    [ ] Benchmark added benefit with new scripts
*/
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

#include <meep.hpp>
using namespace meep;
using std::complex;

double one(const vec &) { return 1.0; }

int test_normal_loop() {
  double a = 5.0;

  grid_volume gv = vol3d(3.0, 3.0, 3.0, a);
  structure s1(gv, one, pml(1.0), meep::identity(), 5);
  fields f(&s1);
  f.add_point_source(Ez, 0.8, 1.6, 0.0, 4.0, vec(1.299, 1.401, 0.0), 1.0);
  f.step();
  for (int i = 0; i < f.num_chunks; i++) {
      master_printf("\n---------------------------------\n");
      master_printf("%d",LOOPS_ARE_STRIDE1(f.chunks[i]->gv));
      master_printf("\n---------------------------------\n");
      LOOP_OVER_VOL(f.chunks[i]->gv, Hx, idx) master_printf("%td ",idx);
  }
  return 1;
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;
  master_printf("Testing 2D...\n");
  
  if (!test_normal_loop()) abort("error in test_periodic_tm vacuum\n");

  return 0;
}