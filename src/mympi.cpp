/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include <cstdlib>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <new>
#include <thread>
#include <utility>
#include <stdarg.h>
#include <string.h>

#include "meep.hpp"
#include "config.h"
#include "backend/mpi_context.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#ifdef HAVE_MPI_EXT_H
#include <mpi-ext.h>
#endif
#endif

#ifdef _OPENMP
#include "omp.h"
#else
#define omp_get_num_threads() (1)
#endif

#ifdef IGNORE_SIGFPE
#include <signal.h>
#endif

#if defined(DEBUG) && defined(HAVE_FEENABLEEXCEPT)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <fenv.h>
#if !HAVE_DECL_FEENABLEEXCEPT
extern "C" int feenableexcept(int EXCEPTS);
#endif
#endif

#if HAVE_SYS_TIME_H
#include <sys/time.h>
#include <time.h>
#else
#include <time.h>
#endif
#ifdef HAVE_BSDGETTIMEOFDAY
#ifndef HAVE_GETTIMEOFDAY
#define gettimeofday BSDgettimeofday
#define HAVE_GETTIMEOFDAY 1
#endif
#endif

#if HAVE_IMMINTRIN_H
#include <immintrin.h>
#endif

#define UNUSED(x) (void)x // silence compiler warnings

#define MPI_REALNUM (sizeof(realnum) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT)

using namespace std;

namespace meep {

namespace {

#ifdef HAVE_MPI
MPI_Comm mycomm = MPI_COMM_WORLD;
MPI_Comm mycomm_save = MPI_COMM_WORLD;
uint64_t mycomm_generation = 1;
bool mpi_initialized_by_meep = false;
int mpi_thread_level = MPI_THREAD_SINGLE;
std::thread::id mpi_main_thread;
std::string backend_communicator_failure_point;

[[noreturn]] void fatal_backend_communicator(MPI_Comm communicator, const char *message) {
  std::fprintf(stderr, "fatal MPI communicator mutation: %s\n", message);
  std::fflush(stderr);
  if (communicator != MPI_COMM_NULL) MPI_Abort(communicator, EXIT_FAILURE);
  std::_Exit(EXIT_FAILURE);
}
#endif

bool next_generation(uint64_t current, uint64_t &next) {
  if (current == std::numeric_limits<uint64_t>::max()) return false;
  next = current + 1;
  return true;
}

#ifdef HAVE_MPI
void retire_divided_communicators_without_generation() {
  MPI_Comm active = mycomm;
  if (active != MPI_COMM_WORLD && MPI_Comm_free(&mycomm) != MPI_SUCCESS)
    fatal_backend_communicator(active, "active split communicator retirement failed");
  MPI_Comm saved = mycomm_save;
  if (saved != MPI_COMM_WORLD && MPI_Comm_free(&mycomm_save) != MPI_SUCCESS)
    fatal_backend_communicator(saved, "saved split communicator retirement failed");
  mycomm = mycomm_save = MPI_COMM_WORLD;
}

void require_backend_communicator_mutation_thread(const char *operation) {
  std::string why;
  if (!backend_mpi_thread_ready(why))
    throw std::runtime_error(std::string(operation) + ": " + why);
}
#endif

// comms_manager implementation that uses MPI.
class mpi_comms_manager : public comms_manager {
public:
  mpi_comms_manager() {}
  ~mpi_comms_manager() override {
#ifdef HAVE_MPI
    int num_pending_requests = reqs.size();
    std::vector<int> completed_indices(num_pending_requests);
    while (num_pending_requests) {
      int num_completed_requests = 0;
      MPI_Waitsome(reqs.size(), reqs.data(), &num_completed_requests, completed_indices.data(),
                   MPI_STATUSES_IGNORE);
      for (int i = 0; i < num_completed_requests; ++i) {
        int request_idx = completed_indices[i];
        callbacks[request_idx]();
        reqs[request_idx] = MPI_REQUEST_NULL;
        --num_pending_requests;
      }
    }
#endif
  }

  void send_real_async(const void *buf, size_t count, int dest, int tag) override {
#ifdef HAVE_MPI
    reqs.emplace_back();
    callbacks.push_back(/*no-op*/ []{});
    MPI_Isend(buf, static_cast<int>(count), MPI_REALNUM, dest, tag, mycomm, &reqs.back());
#else
    (void)buf;
    (void)count;
    (void)dest;
    (void)tag;
#endif
  }

  void receive_real_async(void *buf, size_t count, int source, int tag,
                          const receive_callback &cb) override {
#ifdef HAVE_MPI
    reqs.emplace_back();
    callbacks.push_back(cb);
    MPI_Irecv(buf, static_cast<int>(count), MPI_REALNUM, source, tag, mycomm, &reqs.back());
#else
    (void)buf;
    (void)count;
    (void)source;
    (void)tag;
    (void)cb;
#endif
  }

#ifdef HAVE_MPI
  size_t max_transfer_size() const override { return std::numeric_limits<int>::max(); }
#endif

private:
#ifdef HAVE_MPI
  std::vector<MPI_Request> reqs;
#endif
  std::vector<receive_callback> callbacks;
};

} // namespace

std::unique_ptr<comms_manager> create_comms_manager() {
  return std::unique_ptr<comms_manager>(new mpi_comms_manager());
}

void node_local_process_info(int *rank, int *size) {
  if (!rank || !size) abort("node_local_process_info requires non-null outputs");
#ifdef HAVE_MPI
#if MPI_VERSION >= 3
  MPI_Comm local_comm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(mycomm, MPI_COMM_TYPE_SHARED, my_rank(), MPI_INFO_NULL, &local_comm) !=
      MPI_SUCCESS)
    abort("MPI_Comm_split_type failed while selecting an accelerator");
  if (MPI_Comm_rank(local_comm, rank) != MPI_SUCCESS ||
      MPI_Comm_size(local_comm, size) != MPI_SUCCESS) {
    MPI_Comm_free(&local_comm);
    abort("MPI could not query the node-local accelerator rank");
  }
  MPI_Comm_free(&local_comm);
#else
  *rank = my_rank();
  *size = count_processors();
#endif
#else
  *rank = 0;
  *size = 1;
#endif
}

int verbosity = 1; // defined in meep.h

/* Set CPU to flush subnormal values to zero (if iszero == true).  This slightly
   reduces the range of floating-point numbers, but can greatly increase the speed
   in cases where subnormal values might arise (e.g. deep in the tails of
   exponentially decaying sources).

   See also meep#1708.

   code based on github.com/JuliaLang/julia/blob/master/src/processor_x86.cpp#L1087-L1104,
   which is free software under the GPL-compatible "MIT license" */
static void _set_zero_subnormals(bool iszero) {
#if HAVE_IMMINTRIN_H
  unsigned int flags =
      0x00008040; // assume a non-ancient processor with SSE2, supporting both FTZ and DAZ flags
  unsigned int state = _mm_getcsr();
  if (iszero)
    state |= flags;
  else
    state &= ~flags;
  _mm_setcsr(state);
#else
  (void)iszero; // unused
#endif
}
void set_zero_subnormals(bool iszero) {
  int n = omp_get_num_threads();
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
  for (int i = 0; i < n; ++i)
    _set_zero_subnormals(iszero); // This has to be done in every thread for OpenMP.
}

void setup() {
  set_zero_subnormals(true);
#ifdef _OPENMP
  if (getenv("OMP_NUM_THREADS") == NULL) omp_set_num_threads(1);
#endif
}

initialize::initialize(int &argc, char **&argv) {
#ifdef HAVE_MPI
  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS) abort("MPI_Initialized failed");
  int finalized = 0;
  if (MPI_Finalized(&finalized) != MPI_SUCCESS) abort("MPI_Finalized failed");
  if (finalized) abort("MPI was finalized before Meep initialization");
  if (!initialized) {
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_level) != MPI_SUCCESS)
      abort("MPI_Init_thread failed");
    mpi_initialized_by_meep = true;
  }
  else if (MPI_Query_thread(&mpi_thread_level) != MPI_SUCCESS)
    abort("MPI_Query_thread failed");
  if (mpi_thread_level < MPI_THREAD_FUNNELED)
    abort("MPI does not provide MPI_THREAD_FUNNELED");
  int is_main = 0;
  if (MPI_Is_thread_main(&is_main) != MPI_SUCCESS) abort("MPI_Is_thread_main failed");
  if (!is_main) abort("Meep MPI initialization must run on the MPI main thread");
  mpi_main_thread = std::this_thread::get_id();
  int major, minor;
  MPI_Get_version(&major, &minor);
  if (verbosity > 0)
    master_printf("Using MPI version %d.%d, %d processes\n", major, minor, count_processors());
#else
  UNUSED(argc);
  UNUSED(argv);
#endif
#if defined(DEBUG_FP) && defined(HAVE_FEENABLEEXCEPT)
  feenableexcept(FE_INVALID | FE_OVERFLOW); // crash if NaN created, or overflow
#endif
#ifdef IGNORE_SIGFPE
  signal(SIGFPE, SIG_IGN);
#endif
  t_start = wall_time();
  setup();
}

initialize::~initialize() {
#ifdef HAVE_MPI
  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS) std::_Exit(EXIT_FAILURE);
  if (!initialized) return;
  int finalized = 0;
  if (MPI_Finalized(&finalized) != MPI_SUCCESS) std::_Exit(EXIT_FAILURE);
  if (finalized) {
    /* MPI owns all communicator cleanup after externally initiated finalize.
       Do not call logging/timing helpers: both may enter MPI. */
    mycomm = mycomm_save = MPI_COMM_WORLD;
    return;
  }
  end_divide_parallel();
  if (verbosity > 0) master_printf("\nElapsed run time = %g s\n", elapsed_time());
  if (mpi_initialized_by_meep && !finalized && MPI_Finalize() != MPI_SUCCESS)
    std::_Exit(EXIT_FAILURE);
#else
  if (verbosity > 0) master_printf("\nElapsed run time = %g s\n", elapsed_time());
#endif
}

double wall_time(void) {
#ifdef HAVE_MPI
  return MPI_Wtime();
#elif defined(_OPENMP)
  return omp_get_wtime();
#elif HAVE_GETTIMEOFDAY
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (tv.tv_sec + tv.tv_usec * 1e-6);
#else
  return (clock() * 1.0 / CLOCKS_PER_SECOND);
#endif
}

[[noreturn]] void abort(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char *s;
  vasprintf(&s, fmt, ap);
  va_end(ap);
  // Make a std::string to support older compilers (std::runtime_error(char *) was added in C++11)
  std::string error_msg(s);
  free(s);
#ifdef HAVE_MPI
  if (count_processors() == 1) { throw runtime_error("meep: " + error_msg); }
  fprintf(stderr, "meep: %s", error_msg.c_str());
  if (fmt[strlen(fmt) - 1] != '\n') fputc('\n', stderr); // force newline
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::abort(); // Unreachable but MPI_Abort does not have the noreturn attribute.
#else
  throw runtime_error("meep: " + error_msg);
#endif
}

void send(int from, int to, double *data, int size) {
#ifdef HAVE_MPI
  if (from == to) return;
  if (size == 0) return;
  const int me = my_rank();
  if (from == me) MPI_Send(data, size, MPI_DOUBLE, to, 1, mycomm);
  MPI_Status stat;
  if (to == me) MPI_Recv(data, size, MPI_DOUBLE, from, 1, mycomm, &stat);
#else
  UNUSED(from);
  UNUSED(to);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, float *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, size, MPI_FLOAT, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, double *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, size, MPI_DOUBLE, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, char *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, size, MPI_CHAR, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, complex<double> *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, 2 * size, MPI_DOUBLE, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, int *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, size, MPI_INT, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

void broadcast(int from, size_t *data, int size) {
#ifdef HAVE_MPI
  if (size == 0) return;
  MPI_Bcast(data, size, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG, from, mycomm);
#else
  UNUSED(from);
  UNUSED(data);
  UNUSED(size);
#endif
}

complex<double> broadcast(int from, complex<double> data) {
#ifdef HAVE_MPI
  MPI_Bcast(&data, 2, MPI_DOUBLE, from, mycomm);
#else
  UNUSED(from);
#endif
  return data;
}

double broadcast(int from, double data) {
#ifdef HAVE_MPI
  MPI_Bcast(&data, 1, MPI_DOUBLE, from, mycomm);
#else
  UNUSED(from);
#endif
  return data;
}

int broadcast(int from, int data) {
#ifdef HAVE_MPI
  MPI_Bcast(&data, 1, MPI_INT, from, mycomm);
#else
  UNUSED(from);
#endif
  return data;
}

bool broadcast(int from, bool b) { return broadcast(from, (int)b); }

double max_to_master(double in) {
  double out = in;
#ifdef HAVE_MPI
  MPI_Reduce(&in, &out, 1, MPI_DOUBLE, MPI_MAX, 0, mycomm);
#endif
  return out;
}

double max_to_all(double in) {
  double out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, MPI_MAX, mycomm);
#endif
  return out;
}

int max_to_all(int in) {
  int out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MAX, mycomm);
#endif
  return out;
}

int min_to_all(int in) {
  int out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, mycomm);
#endif
  return out;
}

ivec max_to_all(const ivec &pt) {
  int in[5], out[5];
  for (int i = 0; i < 5; ++i)
    in[i] = out[i] = pt.in_direction(direction(i));
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 5, MPI_INT, MPI_MAX, mycomm);
#endif
  ivec ptout(pt.dim);
  for (int i = 0; i < 5; ++i)
    ptout.set_direction(direction(i), out[i]);
  return ptout;
}

float sum_to_master(float in) {
  float out = in;
#ifdef HAVE_MPI
  MPI_Reduce(&in, &out, 1, MPI_FLOAT, MPI_SUM, 0, mycomm);
#endif
  return out;
}

double sum_to_master(double in) {
  double out = in;
#ifdef HAVE_MPI
  MPI_Reduce(&in, &out, 1, MPI_DOUBLE, MPI_SUM, 0, mycomm);
#endif
  return out;
}

double sum_to_all(double in) {
  double out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, MPI_SUM, mycomm);
#endif
  return out;
}

void sum_to_all(const float *in, float *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, MPI_FLOAT, MPI_SUM, mycomm);
#else
  memcpy(out, in, sizeof(float) * size);
#endif
}

void sum_to_all(const double *in, double *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, MPI_DOUBLE, MPI_SUM, mycomm);
#else
  memcpy(out, in, sizeof(double) * size);
#endif
}

void sum_to_master(const float *in, float *out, int size) {
#ifdef HAVE_MPI
  MPI_Reduce((void *)in, out, size, MPI_FLOAT, MPI_SUM, 0, mycomm);
#else
  memcpy(out, in, sizeof(float) * size);
#endif
}

void sum_to_master(const double *in, double *out, int size) {
#ifdef HAVE_MPI
  MPI_Reduce((void *)in, out, size, MPI_DOUBLE, MPI_SUM, 0, mycomm);
#else
  memcpy(out, in, sizeof(double) * size);
#endif
}

void sum_to_all(const float *in, double *out, int size) {
  double *in2 = new double[size];
  for (int i = 0; i < size; ++i)
    in2[i] = in[i];
  sum_to_all(in2, out, size);
  delete[] in2;
}

void sum_to_all(const complex<double> *in, complex<double> *out, int size) {
  sum_to_all((const double *)in, (double *)out, 2 * size);
}

void sum_to_all(const complex<float> *in, complex<double> *out, int size) {
  sum_to_all((const float *)in, (double *)out, 2 * size);
}

void sum_to_all(const complex<float> *in, complex<float> *out, int size) {
  sum_to_all((const float *)in, (float *)out, 2 * size);
}

void sum_to_master(const complex<float> *in, complex<float> *out, int size) {
  sum_to_master((const float *)in, (float *)out, 2 * size);
}

void sum_to_master(const complex<double> *in, complex<double> *out, int size) {
  sum_to_master((const double *)in, (double *)out, 2 * size);
}

long double sum_to_all(long double in) {
  long double out = in;
#ifdef HAVE_MPI
  if (MPI_LONG_DOUBLE == MPI_DATATYPE_NULL)
    out = sum_to_all(double(in));
  else
    MPI_Allreduce(&in, &out, 1, MPI_LONG_DOUBLE, MPI_SUM, mycomm);
#endif
  return out;
}

int sum_to_all(int in) {
  int out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_SUM, mycomm);
#endif
  return out;
}

int partial_sum_to_all(int in) {
  int out = in;
#ifdef HAVE_MPI
  MPI_Scan(&in, &out, 1, MPI_INT, MPI_SUM, mycomm);
#endif
  return out;
}

size_t sum_to_all(size_t in) {
  size_t out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 1, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                mycomm);
#endif
  return out;
}

void sum_to_all(const size_t *in, size_t *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG,
                MPI_SUM, mycomm);
#else
  memcpy(out, in, sizeof(size_t) * size);
#endif
}

void sum_to_master(const size_t *in, size_t *out, int size) {
#ifdef HAVE_MPI
  MPI_Reduce((void *)in, out, size, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG,
             MPI_SUM, 0, mycomm);
#else
  memcpy(out, in, sizeof(size_t) * size);
#endif
}

size_t partial_sum_to_all(size_t in) {
  size_t out = in;
#ifdef HAVE_MPI
  MPI_Scan(&in, &out, 1, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG, MPI_SUM,
           mycomm);
#endif
  return out;
}

complex<double> sum_to_all(complex<double> in) {
  complex<double> out = in;
#ifdef HAVE_MPI
  MPI_Allreduce(&in, &out, 2, MPI_DOUBLE, MPI_SUM, mycomm);
#endif
  return out;
}

complex<long double> sum_to_all(complex<long double> in) {
  complex<long double> out = in;
#ifdef HAVE_MPI
  if (MPI_LONG_DOUBLE == MPI_DATATYPE_NULL) {
    complex<double> dout;
    dout = sum_to_all(complex<double>(double(in.real()), double(in.imag())));
    out = complex<long double>(dout.real(), dout.imag());
  }
  else
    MPI_Allreduce(&in, &out, 2, MPI_LONG_DOUBLE, MPI_SUM, mycomm);
#endif
  return out;
}

bool or_to_all(bool in) {
  int in2 = in, out;
#ifdef HAVE_MPI
  MPI_Allreduce(&in2, &out, 1, MPI_INT, MPI_LOR, mycomm);
#else
  out = in2;
#endif
  return (bool)out;
}

void or_to_all(const int *in, int *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, MPI_INT, MPI_LOR, mycomm);
#else
  memcpy(out, in, sizeof(int) * size);
#endif
}

void bw_or_to_all(const size_t *in, size_t *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, sizeof(size_t) == 4 ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG,
                MPI_BOR, mycomm);
#else
  memcpy(out, in, sizeof(size_t) * size);
#endif
}

bool and_to_all(bool in) {
  int in2 = in, out;
#ifdef HAVE_MPI
  MPI_Allreduce(&in2, &out, 1, MPI_INT, MPI_LAND, mycomm);
#else
  out = in2;
#endif
  return (bool)out;
}

void and_to_all(const int *in, int *out, int size) {
#ifdef HAVE_MPI
  MPI_Allreduce((void *)in, out, size, MPI_INT, MPI_LAND, mycomm);
#else
  memcpy(out, in, sizeof(int) * size);
#endif
}

void all_wait() {
#ifdef HAVE_MPI
  MPI_Barrier(mycomm);
#endif
}

int my_rank() {
#ifdef HAVE_MPI
  int rank;
  MPI_Comm_rank(mycomm, &rank);
  return rank;
#else
  return 0;
#endif
}

int count_processors() {
#ifdef HAVE_MPI
  int n;
  MPI_Comm_size(mycomm, &n);
  return n;
#else
  return 1;
#endif
}

bool with_mpi() {
#ifdef HAVE_MPI
  return true;
#else
  return false;
#endif
}

// IO Routines...

bool am_really_master() { return (my_global_rank() == 0); }

static meep_printf_callback_func master_printf_callback = NULL;
static meep_printf_callback_func master_printf_stderr_callback = NULL;

meep_printf_callback_func set_meep_printf_callback(meep_printf_callback_func func) {
  meep_printf_callback_func old_func = master_printf_callback;
  master_printf_callback = func;
  return old_func;
}

meep_printf_callback_func set_meep_printf_stderr_callback(meep_printf_callback_func func) {
  meep_printf_callback_func old_func = master_printf_stderr_callback;
  master_printf_stderr_callback = func;
  return old_func;
}

static void _do_master_printf(FILE *output, meep_printf_callback_func callback, const char *fmt,
                              va_list ap) {
  if (am_really_master()) {
    if (callback) {
      char *s;
      vasprintf(&s, fmt, ap);
      callback(s);
      free(s);
    }
    else {
      vfprintf(output, fmt, ap);
      fflush(output);
    }
  }
  va_end(ap);
}

void master_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  _do_master_printf(stdout, master_printf_callback, fmt, ap);
  va_end(ap);
}

void master_printf_stderr(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  _do_master_printf(stderr, master_printf_stderr_callback, fmt, ap);
  va_end(ap);
}

static FILE *debf = NULL;

void debug_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  if (debf == NULL) {
    char temp[50];
    snprintf(temp, 50, "debug_out_%d", my_rank());
    debf = fopen(temp, "w");
    if (!debf) meep::abort("Unable to open debug output %s\n", temp);
  }
  vfprintf(debf, fmt, ap);
  fflush(debf);
  va_end(ap);
}

void master_fprintf(FILE *f, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  if (am_master()) {
    vfprintf(f, fmt, ap);
    fflush(f);
  }
  va_end(ap);
}
FILE *master_fopen(const char *name, const char *mode) {
  FILE *f = am_master() ? fopen(name, mode) : 0;

  /* other processes need to know if fopen returned zero, in order
     to abort if fopen failed.  If fopen was successfully, just return
     a random non-zero pointer (which is never used except to compare to zero)
     on non-master processes */
  if (broadcast(0, bool(f != 0)) && !am_master()) f = (FILE *)name;
  return f;
}
void master_fclose(FILE *f) {
  if (am_master()) fclose(f);
}

/* The following functions bracket a "critical section," a region
   of code that should be executed by only one process at a time.

   They work by having each process wait for a message from the
   previous process before starting.

   Each critical section is passed an integer "tag"...ideally, this
   should be a unique identifier for each critical section so that
   messages from different critical sections don't get mixed up
   somehow. */

void begin_critical_section(int tag) {
#ifdef HAVE_MPI
  int process_rank;
  MPI_Comm_rank(mycomm, &process_rank);
  if (process_rank > 0) { /* wait for a message before continuing */
    MPI_Status status;
    int recv_tag = tag - 1; /* initialize to wrong value */
    MPI_Recv(&recv_tag, 1, MPI_INT, process_rank - 1, tag, mycomm, &status);
    if (recv_tag != tag) meep::abort("invalid tag received in begin_critical_section");
  }
#else
  UNUSED(tag);
#endif
}

void end_critical_section(int tag) {
#ifdef HAVE_MPI
  int process_rank, num_procs;
  MPI_Comm_rank(mycomm, &process_rank);
  MPI_Comm_size(mycomm, &num_procs);
  if (process_rank != num_procs - 1) { /* send a message to next process */
    MPI_Send(&tag, 1, MPI_INT, process_rank + 1, tag, mycomm);
  }
#else
  UNUSED(tag);
#endif
}

/* Simple, somewhat hackish API to allow user to run multiple simulations
   in parallel in the same MPI job.  The user calls

   mygroup = divide_parallel_processes(numgroups);

   to divide all of the MPI processes into numgroups equal groups,
   and to return the index (from 0 to numgroups-1) of the current group.
   From this point on, all fields etc. that you create and all
   calls from mympi.cpp will only communicate within your group of
   processes.

   However, there are two calls that you can use to switch back to
   globally communication among all processes:

   begin_global_communications();
   ....do stuff....
   end_global_communications();

   It is important not to mix the two types; e.g. you cannot timestep
   a field created in the local group in global mode, or vice versa.
*/

int divide_parallel_processes(int numgroups) {
#ifdef HAVE_MPI
  require_backend_communicator_mutation_thread("divide_parallel_processes");
  int world_rank = 0, world_size = 1;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) != MPI_SUCCESS)
    fatal_backend_communicator(MPI_COMM_WORLD, "world communicator rank query failed");
  if (MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS)
    fatal_backend_communicator(MPI_COMM_WORLD, "world communicator size query failed");
  if (numgroups > world_size) meep::abort("numgroups > count_processors");
  uint64_t next = 0;
  if (!next_generation(mycomm_generation, next))
    meep::abort("backend communicator generation overflow");
  int mygroup = (world_rank * numgroups) / world_size;
  MPI_Comm candidate = MPI_COMM_NULL;
  if (MPI_Comm_split(MPI_COMM_WORLD, mygroup, world_rank, &candidate) != MPI_SUCCESS)
    fatal_backend_communicator(MPI_COMM_WORLD, "MPI_Comm_split failed");
  retire_divided_communicators_without_generation();
  mycomm = candidate;
  mycomm_generation = next;
  return mygroup;
#else
  if (numgroups != 1) meep::abort("cannot divide processes in non-MPI mode");
  return 0;
#endif
}

void begin_global_communications(void) {
#ifdef HAVE_MPI
  require_backend_communicator_mutation_thread("begin_global_communications");
  if (mycomm_save != MPI_COMM_WORLD)
    meep::abort("nested begin_global_communications is unsupported");
  if (mycomm == MPI_COMM_WORLD) return;
  uint64_t next = 0;
  if (!next_generation(mycomm_generation, next))
    meep::abort("backend communicator generation overflow");
  mycomm_save = mycomm;
  mycomm = MPI_COMM_WORLD;
  mycomm_generation = next;
#endif
}

void end_global_communications(void) {
#ifdef HAVE_MPI
  require_backend_communicator_mutation_thread("end_global_communications");
  if (mycomm_save == MPI_COMM_WORLD) return;
  uint64_t next = 0;
  if (!next_generation(mycomm_generation, next))
    meep::abort("backend communicator generation overflow");
  mycomm = mycomm_save;
  mycomm_save = MPI_COMM_WORLD;
  mycomm_generation = next;
#endif
}

void end_divide_parallel(void) {
#ifdef HAVE_MPI
  require_backend_communicator_mutation_thread("end_divide_parallel");
  if (mycomm == MPI_COMM_WORLD && mycomm_save == MPI_COMM_WORLD) return;
  uint64_t next = 0;
  if (!next_generation(mycomm_generation, next))
    meep::abort("backend communicator generation overflow");
  retire_divided_communicators_without_generation();
  mycomm_generation = next;
#endif
}

int my_global_rank() {
#ifdef HAVE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
#else
  return 0;
#endif
}

struct BackendCommunicatorLease::Impl {
#ifdef HAVE_MPI
  MPI_Comm comm;
  Impl() : comm(MPI_COMM_NULL) {}
#else
  Impl() {}
#endif
};

BackendCommunicatorLease::BackendCommunicatorLease() : impl_(NULL), info_() {}

BackendCommunicatorLease::~BackendCommunicatorLease() {
  /* MPI_Comm_free is collective and is therefore never attempted here. The
     explicit retire transaction must run before destruction. After MPI has
     finalized, the handle is inert and local deletion is sufficient. */
  if (valid()) {
#ifdef HAVE_MPI
    int initialized = 0, finalized = 0;
    if (MPI_Initialized(&initialized) == MPI_SUCCESS && initialized &&
        MPI_Finalized(&finalized) == MPI_SUCCESS && finalized) {
      impl_->comm = MPI_COMM_NULL;
      delete impl_;
      impl_ = NULL;
      return;
    }
#endif
    std::fprintf(stderr,
                 "BackendCommunicatorLease destroyed before explicit collective retirement\n");
    std::abort();
  }
  delete impl_;
}

BackendCommunicatorLease::BackendCommunicatorLease(BackendCommunicatorLease &&other) noexcept
    : impl_(other.impl_), info_(std::move(other.info_)) {
  other.impl_ = NULL;
  other.info_ = BackendCommunicatorInfo();
}

BackendCommunicatorLease &
BackendCommunicatorLease::operator=(BackendCommunicatorLease &&other) noexcept {
  if (this == &other) return *this;
  if (valid()) {
#ifdef HAVE_MPI
    int initialized = 0, finalized = 0;
    if (MPI_Initialized(&initialized) == MPI_SUCCESS && initialized &&
        MPI_Finalized(&finalized) == MPI_SUCCESS && finalized)
      impl_->comm = MPI_COMM_NULL;
    else
#endif
    {
      std::fprintf(stderr,
                   "BackendCommunicatorLease overwritten before explicit collective retirement\n");
      std::abort();
    }
  }
  delete impl_;
  impl_ = other.impl_;
  info_ = std::move(other.info_);
  other.impl_ = NULL;
  other.info_ = BackendCommunicatorInfo();
  return *this;
}

bool BackendCommunicatorLease::valid() const {
#ifdef HAVE_MPI
  return impl_ && impl_->comm != MPI_COMM_NULL;
#else
  return impl_ != NULL;
#endif
}

const BackendCommunicatorInfo &BackendCommunicatorLease::info() const { return info_; }

uint64_t current_backend_communicator_generation() {
#ifdef HAVE_MPI
  return mycomm_generation;
#else
  return 1;
#endif
}

#ifdef HAVE_MPI
MPI_Comm current_backend_communicator() { return mycomm; }

MPI_Comm backend_communicator(const BackendCommunicatorLease &lease) {
  return lease.impl_ ? lease.impl_->comm : MPI_COMM_NULL;
}
#endif

bool next_backend_communicator_generation(uint64_t current, uint64_t &next, std::string &why) {
  why.clear();
  if (!next_generation(current, next)) {
    why = "backend communicator generation overflow";
    return false;
  }
  return true;
}

bool backend_mpi_thread_ready(std::string &why) {
  why.clear();
#ifdef HAVE_MPI
  if (mpi_main_thread != std::thread::id() && std::this_thread::get_id() != mpi_main_thread) {
    why = "MPI transport must run on the MPI main thread";
    return false;
  }
  int initialized = 0, finalized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS) {
    why = "MPI initialization-state query failed";
    return false;
  }
  if (!initialized) {
    why = "MPI is not initialized";
    return false;
  }
  if (MPI_Finalized(&finalized) != MPI_SUCCESS) {
    why = "MPI finalized-state query failed";
    return false;
  }
  if (finalized) {
    why = "MPI is finalized";
    return false;
  }
  int level = MPI_THREAD_SINGLE;
  if (MPI_Query_thread(&level) != MPI_SUCCESS) {
    why = "MPI thread-level query failed";
    return false;
  }
  if (level < MPI_THREAD_FUNNELED) {
    why = "MPI thread level is below MPI_THREAD_FUNNELED";
    return false;
  }
  int is_main = 0;
  if (MPI_Is_thread_main(&is_main) != MPI_SUCCESS) {
    why = "MPI main-thread query failed";
    return false;
  }
  if (!is_main) {
    why = "MPI transport must run on the MPI main thread";
    return false;
  }
#endif
  return true;
}

bool query_gpu_aware_mpi_provider(bool &query_available, bool &supports_direct,
                                  std::string &provider, std::string &why) {
  query_available = false;
  supports_direct = false;
  provider.clear();
  why.clear();
#ifdef HAVE_MPI
  if (!backend_mpi_thread_ready(why)) return false;
  char text[MPI_MAX_LIBRARY_VERSION_STRING] = {};
  int length = 0;
  if (MPI_Get_library_version(text, &length) != MPI_SUCCESS) {
    why = "MPI_Get_library_version failed";
    return false;
  }
  provider.assign(text, size_t(std::max(0, length)));
#ifdef HAVE_MPIX_QUERY_CUDA_SUPPORT
  query_available = true;
  supports_direct = MPIX_Query_cuda_support() > 0;
#endif
#else
  provider = "none";
#endif
  return true;
}

bool collective_resolve_gpu_mpi_policy(bool local_parse_valid, GpuMpiPolicy local_policy,
                                       bool local_query_available, bool local_direct_support,
                                       GpuMpiPolicy &agreed_policy, GpuMpiRoute &route,
                                       std::string &why) {
  why.clear();
#ifdef HAVE_MPI
  if (!backend_mpi_thread_ready(why)) return false;
  int values[3] = {local_parse_valid ? int(local_policy) : -1,
                   local_query_available ? 1 : 0, local_direct_support ? 1 : 0};
  int minima[3] = {}, maxima[3] = {};
  if (MPI_Allreduce(values, minima, 3, MPI_INT, MPI_MIN, mycomm) != MPI_SUCCESS)
    fatal_backend_communicator(mycomm, "MPI policy minimum reconciliation failed");
  if (MPI_Allreduce(values, maxima, 3, MPI_INT, MPI_MAX, mycomm) != MPI_SUCCESS)
    fatal_backend_communicator(mycomm, "MPI policy maximum reconciliation failed");
  if (minima[0] < 0 || minima[0] != maxima[0]) {
    why = minima[0] < 0 ? "GPU MPI policy parse failed on at least one rank"
                        : "GPU MPI policy differs across ranks";
    return false;
  }
  agreed_policy = GpuMpiPolicy(minima[0]);
  const bool all_query = minima[1] != 0;
  const bool all_direct = minima[2] != 0;
  return resolve_gpu_mpi_route(agreed_policy, all_query, all_direct, route, why);
#else
  if (!local_parse_valid) {
    why = "GPU MPI policy parse failed";
    return false;
  }
  agreed_policy = local_policy;
  return resolve_gpu_mpi_route(local_policy, local_query_available, local_direct_support, route,
                               why);
#endif
}

bool create_backend_communicator_lease(BackendCommunicatorLease &lease, std::string &why) {
  why.clear();
  if (lease.valid()) {
    why = "backend communicator lease is already valid";
    return false;
  }
#ifdef HAVE_MPI
  if (!backend_mpi_thread_ready(why)) return false;
  BackendCommunicatorInfo info{};
  info.generation = mycomm_generation;
  int local_failed = 0;
  if (MPI_Comm_rank(mycomm, &info.rank) != MPI_SUCCESS ||
      MPI_Comm_size(mycomm, &info.size) != MPI_SUCCESS) {
    why = "MPI communicator identity query failed";
    local_failed = 1;
  }
  int *tag_ub = NULL, present = 0;
  if (!local_failed &&
      MPI_Comm_get_attr(mycomm, MPI_TAG_UB, &tag_ub, &present) != MPI_SUCCESS) {
    why = "MPI_TAG_UB query failed";
    local_failed = 1;
  }
  info.tag_ub = present && tag_ub ? *tag_ub : 32767;
  if (!local_failed && MPI_Query_thread(&info.thread_level) != MPI_SUCCESS) {
    why = "MPI thread-level query failed";
    local_failed = 1;
  }
  int is_main = 0;
  if (!local_failed && MPI_Is_thread_main(&is_main) != MPI_SUCCESS) {
    why = "MPI main-thread query failed";
    local_failed = 1;
  }
  info.thread_main = is_main != 0;
  if (!local_failed) {
    try {
      if (!query_gpu_aware_mpi_provider(info.provider_query_available,
                                        info.provider_supports_direct, info.provider, why))
        local_failed = 1;
    }
    catch (...) {
      why = "MPI provider metadata allocation failed";
      local_failed = 1;
    }
  }
  if (backend_communicator_failure_point == "before_dup") {
    why = "injected communicator creation preflight failure";
    local_failed = 1;
  }
  const unsigned long long local_generation =
      static_cast<unsigned long long>(mycomm_generation);
  unsigned long long minimum_generation = 0, maximum_generation = 0;
  if (MPI_Allreduce(&local_generation, &minimum_generation, 1, MPI_UNSIGNED_LONG_LONG,
                    MPI_MIN, mycomm) != MPI_SUCCESS)
    fatal_backend_communicator(mycomm,
                               "communicator generation minimum reconciliation failed");
  if (MPI_Allreduce(&local_generation, &maximum_generation, 1, MPI_UNSIGNED_LONG_LONG,
                    MPI_MAX, mycomm) != MPI_SUCCESS)
    fatal_backend_communicator(mycomm,
                               "communicator generation maximum reconciliation failed");
  if (minimum_generation != maximum_generation) {
    why = "active communicator generation differs across ranks";
    local_failed = 1;
  }
  int any_failed = 0;
  if (MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_LOR, mycomm) != MPI_SUCCESS)
    fatal_backend_communicator(mycomm, "communicator creation preflight reconciliation failed");
  if (any_failed) {
    if (!local_failed) why = "MPI communicator lease preflight failed on another rank";
    return false;
  }

  BackendCommunicatorLease::Impl *candidate =
      new (std::nothrow) BackendCommunicatorLease::Impl();
  local_failed = candidate ? 0 : 1;
  if (MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_LOR, mycomm) != MPI_SUCCESS)
    fatal_backend_communicator(mycomm, "communicator allocation reconciliation failed");
  if (any_failed) {
    delete candidate;
    why = local_failed ? "backend communicator lease allocation failed"
                       : "backend communicator lease allocation failed on another rank";
    return false;
  }
  if (MPI_Comm_dup(mycomm, &candidate->comm) != MPI_SUCCESS)
    fatal_backend_communicator(mycomm, "MPI_Comm_dup failed");
  if (MPI_Comm_set_errhandler(candidate->comm, MPI_ERRORS_RETURN) != MPI_SUCCESS)
    fatal_backend_communicator(candidate->comm,
                               "MPI communicator error-handler installation failed");
  local_failed = 0;
  if (backend_communicator_failure_point == "after_dup") local_failed = 1;
  any_failed = 0;
  if (MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_LOR, candidate->comm) !=
      MPI_SUCCESS)
    fatal_backend_communicator(candidate->comm,
                               "communicator creation reconciliation failed");
  if (any_failed) {
    if (MPI_Comm_free(&candidate->comm) != MPI_SUCCESS)
      fatal_backend_communicator(mycomm, "communicator creation rollback failed");
    delete candidate;
    why = "MPI communicator lease creation failed on at least one rank";
    return false;
  }
  lease.impl_ = candidate;
  lease.info_ = std::move(info);
  return true;
#else
  BackendCommunicatorLease::Impl *candidate = new BackendCommunicatorLease::Impl();
  BackendCommunicatorInfo info{};
  info.generation = 1;
  info.rank = 0;
  info.size = 1;
  info.tag_ub = std::numeric_limits<int>::max();
  info.thread_level = 0;
  info.thread_main = true;
  info.provider_query_available = false;
  info.provider_supports_direct = false;
  info.provider = "none";
  lease.impl_ = candidate;
  lease.info_ = info;
  return true;
#endif
}

bool retire_backend_communicator_lease(BackendCommunicatorLease &lease, std::string &why) {
  why.clear();
  if (!lease.valid()) return true;
#ifdef HAVE_MPI
  int finalized = 0;
  if (MPI_Finalized(&finalized) != MPI_SUCCESS)
    fatal_backend_communicator(lease.impl_->comm, "MPI_Finalized query failed");
  if (finalized) {
    delete lease.impl_;
    lease.impl_ = NULL;
    lease.info_ = BackendCommunicatorInfo();
    why = "MPI is finalized; communicator lease was discarded locally without MPI cleanup";
    return false;
  }
  if (!backend_mpi_thread_ready(why)) return false;
  int local_failed = backend_communicator_failure_point == "before_retire" ? 1 : 0;
  int any_failed = 0;
  if (MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_LOR, lease.impl_->comm) !=
      MPI_SUCCESS)
    fatal_backend_communicator(lease.impl_->comm,
                               "communicator retirement reconciliation failed");
  if (any_failed) {
    why = "MPI communicator lease retirement preflight failed on at least one rank";
    return false;
  }
  MPI_Comm retiring = lease.impl_->comm;
  if (MPI_Comm_free(&lease.impl_->comm) != MPI_SUCCESS)
    fatal_backend_communicator(retiring, "MPI_Comm_free failed");
#endif
  delete lease.impl_;
  lease.impl_ = NULL;
  lease.info_ = BackendCommunicatorInfo();
  return true;
}

bool collective_validate_remote_halo_agreement(const BackendCommunicatorLease &lease,
                                               const RemoteHaloProgram &program,
                                               std::string &why) {
  why.clear();
  if (!lease.valid()) {
    why = "remote halo agreement requires a communicator lease";
    return false;
  }
#ifdef HAVE_MPI
  if (!backend_mpi_thread_ready(why)) return false;
  std::string local_why;
  int local_invalid =
      lease.info().generation != current_backend_communicator_generation() ||
              program.communicator_generation != current_backend_communicator_generation() ||
              program.communicator_generation != lease.info().generation ||
              program.communicator_rank != lease.info().rank ||
              program.communicator_size != lease.info().size
          ? 1
          : 0;
  if (local_invalid) local_why = "remote halo program communicator identity is stale";
  if (!local_invalid && !validate_remote_halo_program(program, lease.info().tag_ub, local_why))
    local_invalid = 1;
  int any_invalid = 0;
  if (MPI_Allreduce(&local_invalid, &any_invalid, 1, MPI_INT, MPI_LOR, lease.impl_->comm) !=
      MPI_SUCCESS)
    fatal_backend_communicator(lease.impl_->comm,
                               "remote halo preflight reconciliation failed");
  if (any_invalid) {
    why = local_invalid ? local_why : "remote halo program is invalid on another rank";
    return false;
  }
  struct Packed {
    int32_t source_rank, destination_rank, ft, source_chunk, destination_chunk, tag, direction;
    uint64_t canonical_ordinal, wire_digest, wire_bytes;
  };
  std::vector<RemoteHaloAgreementRecord> local_records;
  std::vector<Packed> local;
  std::vector<int> counts;
  std::vector<int> displacements;
  int local_bytes = 0;
  int local_prepare_failed = 0;
  try {
    remote_halo_agreement_records(program, local_records);
    if (local_records.size() > size_t(INT_MAX) / sizeof(Packed)) {
      local_why = "remote halo agreement metadata exceeds MPI int count";
      local_prepare_failed = 1;
    }
    if (!local_prepare_failed) {
      local.reserve(local_records.size());
      for (const RemoteHaloAgreementRecord &record : local_records)
        local.push_back(Packed{record.key.source_rank, record.key.destination_rank,
                               int32_t(record.key.ft), record.key.source_chunk,
                               record.key.destination_chunk, record.key.tag,
                               int32_t(record.direction), record.key.canonical_ordinal,
                               record.wire_digest, record.wire_bytes});
      local_bytes = int(local.size() * sizeof(Packed));
      counts.resize(lease.info().size);
      displacements.resize(lease.info().size);
    }
  }
  catch (...) {
    local_why = "remote halo agreement metadata allocation failed";
    local_prepare_failed = 1;
  }
  int any_prepare_failed = 0;
  if (MPI_Allreduce(&local_prepare_failed, &any_prepare_failed, 1, MPI_INT, MPI_LOR,
                    lease.impl_->comm) != MPI_SUCCESS)
    fatal_backend_communicator(lease.impl_->comm,
                               "remote halo metadata preparation reconciliation failed");
  if (any_prepare_failed) {
    why = local_prepare_failed ? local_why
                               : "remote halo metadata preparation failed on another rank";
    return false;
  }
  if (MPI_Allgather(&local_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, lease.impl_->comm) !=
      MPI_SUCCESS)
    fatal_backend_communicator(lease.impl_->comm,
                               "remote halo agreement count exchange failed");
  int total = 0;
  for (int i = 0; i < lease.info().size; ++i) {
    if (counts[i] < 0 || counts[i] % int(sizeof(Packed)) || counts[i] > INT_MAX - total) {
      why = "remote halo agreement metadata counts are invalid";
      return false;
    }
    displacements[i] = total;
    total += counts[i];
  }
  std::vector<unsigned char> gathered;
  local_prepare_failed = 0;
  try { gathered.resize(static_cast<size_t>(total), 0); }
  catch (...) {
    local_why = "remote halo agreement receive allocation failed";
    local_prepare_failed = 1;
  }
  any_prepare_failed = 0;
  if (MPI_Allreduce(&local_prepare_failed, &any_prepare_failed, 1, MPI_INT, MPI_LOR,
                    lease.impl_->comm) != MPI_SUCCESS)
    fatal_backend_communicator(lease.impl_->comm,
                               "remote halo receive allocation reconciliation failed");
  if (any_prepare_failed) {
    why = local_prepare_failed ? local_why
                               : "remote halo receive allocation failed on another rank";
    return false;
  }
  if (MPI_Allgatherv(local.empty() ? NULL : static_cast<void *>(local.data()), local_bytes,
                     MPI_BYTE, gathered.empty() ? NULL : gathered.data(), counts.data(),
                     displacements.data(), MPI_BYTE, lease.impl_->comm) != MPI_SUCCESS)
    fatal_backend_communicator(lease.impl_->comm,
                               "remote halo agreement record exchange failed");
  std::vector<RemoteHaloAgreementRecord> records;
  int local_decode_failed = 0;
  try {
    records.reserve(size_t(total) / sizeof(Packed));
    for (int offset = 0; offset < total; offset += int(sizeof(Packed))) {
      Packed packed;
      memcpy(&packed, gathered.data() + offset, sizeof(Packed));
      if (packed.direction < 0 || packed.direction > 1) {
        local_why = "remote halo agreement contains an invalid direction";
        local_decode_failed = 1;
        break;
      }
      RemoteHaloAgreementRecord record;
      record.key = RemoteHaloWireKey{packed.source_rank, packed.destination_rank,
                                     field_type(packed.ft), packed.source_chunk,
                                     packed.destination_chunk, packed.tag,
                                     packed.canonical_ordinal};
      record.direction = RemoteHaloDirection(packed.direction);
      record.wire_digest = packed.wire_digest;
      record.wire_bytes = packed.wire_bytes;
      records.push_back(record);
    }
    if (!local_decode_failed && !validate_remote_halo_agreement(records, local_why))
      local_decode_failed = 1;
  }
  catch (...) {
    local_why = "remote halo agreement decode allocation failed";
    local_decode_failed = 1;
  }
  int any_decode_failed = 0;
  if (MPI_Allreduce(&local_decode_failed, &any_decode_failed, 1, MPI_INT, MPI_LOR,
                    lease.impl_->comm) != MPI_SUCCESS)
    fatal_backend_communicator(lease.impl_->comm,
                               "remote halo agreement result reconciliation failed");
  if (any_decode_failed) {
    why = local_decode_failed ? local_why : "remote halo agreement failed on another rank";
    return false;
  }
  return true;
#else
  if (program.communicator_generation != lease.info().generation ||
      program.communicator_rank != lease.info().rank ||
      program.communicator_size != lease.info().size) {
    why = "remote halo program communicator identity is stale";
    return false;
  }
  if (!validate_remote_halo_program(program, lease.info().tag_ub, why)) return false;
  std::vector<RemoteHaloAgreementRecord> records;
  remote_halo_agreement_records(program, records);
  return validate_remote_halo_agreement(records, why);
#endif
}

void set_backend_communicator_failure_for_testing(const char *point) {
#ifdef HAVE_MPI
  backend_communicator_failure_point = point ? point : "";
#else
  (void)point;
#endif
}

} // namespace meep
