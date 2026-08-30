/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

/* this file implements multilevel atomic materials for Meep */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits>
#include <stdexcept>
#include "meep.hpp"
#include "meep_internals.hpp"
#include "backend/descriptors.hpp"
#include "config.h"

namespace meep {

multilevel_susceptibility::multilevel_susceptibility(int theL, int theT, const realnum *theGamma,
                                                     const realnum *theN0, const realnum *thealpha,
                                                     const realnum *theomega,
                                                     const realnum *thegamma,
                                                     const realnum *thesigmat) {
  L = theL;
  T = theT;
  Gamma = new realnum[L * L];
  memcpy(Gamma, theGamma, sizeof(realnum) * L * L);
  N0 = new realnum[L];
  memcpy(N0, theN0, sizeof(realnum) * L);
  alpha = new realnum[L * T];
  memcpy(alpha, thealpha, sizeof(realnum) * L * T);
  omega = new realnum[T];
  memcpy(omega, theomega, sizeof(realnum) * T);
  gamma = new realnum[T];
  memcpy(gamma, thegamma, sizeof(realnum) * T);
  sigmat = new realnum[T * 5];
  memcpy(sigmat, thesigmat, sizeof(realnum) * T * 5);
}

multilevel_susceptibility::multilevel_susceptibility(const multilevel_susceptibility &from)
    : susceptibility(from) {
  L = from.L;
  T = from.T;
  Gamma = new realnum[L * L];
  memcpy(Gamma, from.Gamma, sizeof(realnum) * L * L);
  N0 = new realnum[L];
  memcpy(N0, from.N0, sizeof(realnum) * L);
  alpha = new realnum[L * T];
  memcpy(alpha, from.alpha, sizeof(realnum) * L * T);
  omega = new realnum[T];
  memcpy(omega, from.omega, sizeof(realnum) * T);
  gamma = new realnum[T];
  memcpy(gamma, from.gamma, sizeof(realnum) * T);
  sigmat = new realnum[T * 5];
  memcpy(sigmat, from.sigmat, sizeof(realnum) * T * 5);
}

multilevel_susceptibility::~multilevel_susceptibility() {
  delete[] Gamma;
  delete[] N0;
  delete[] alpha;
  delete[] omega;
  delete[] gamma;
  delete[] sigmat;
}

#if MEEP_SINGLE
#define DGETRF F77_FUNC(sgetrf, SGETRF)
#define DGETRI F77_FUNC(sgetri, SGETRI)
#else
#define DGETRF F77_FUNC(dgetrf, DGETRF)
#define DGETRI F77_FUNC(dgetri, DGETRI)
#endif
extern "C" void DGETRF(const int *m, const int *n, realnum *A, const int *lda, int *ipiv,
                       int *info);
extern "C" void DGETRI(const int *n, realnum *A, const int *lda, int *ipiv, realnum *work,
                       int *lwork, int *info);

/* S -> inv(S), where S is a p x p matrix in row-major order */
static bool invert(realnum *S, int p) {
#ifdef HAVE_LAPACK
  int info = 0;
  int *ipiv = new int[p];
  DGETRF(&p, &p, S, &p, ipiv, &info);
  if (info < 0) meep::abort("invalid argument %d in DGETRF", -info);
  if (info > 0) {
    delete[] ipiv;
    return false;
  } // singular

  int lwork = -1;
  realnum work1 = 0.0;
  DGETRI(&p, S, &p, ipiv, &work1, &lwork, &info);
  if (info != 0) meep::abort("error %d in DGETRI workspace query", info);
  lwork = int(work1);
  realnum *work = new realnum[lwork]();
  DGETRI(&p, S, &p, ipiv, work, &lwork, &info);
  if (info < 0) meep::abort("invalid argument %d in DGETRI", -info);

  delete[] work;
  delete[] ipiv;
  return info == 0;
#else /* !HAVE_LAPACK */
  meep::abort("LAPACK is needed for multilevel-atom support");
  return false;
#endif
}

typedef realnum *realnumP;
typedef struct {
  size_t sz_data;
  size_t ntot;
  realnum *GammaInv;                    // inv(1 + Gamma * dt / 2)
  realnumP *P[NUM_FIELD_COMPONENTS][2]; // P[c][cmp][transition][i]
  realnumP *P_prev[NUM_FIELD_COMPONENTS][2];
  realnum *N;    // ntot x L array of centered grid populations N[i*L + level]
  realnum *Ntmp; // temporary length L array of levels, used in updating
  realnum data[1];
} multilevel_data;

bool preflight_multilevel_internal_data(const std::vector<double> &gamma_matrix,
                                        size_t levels, size_t transitions, size_t ntot,
                                        size_t active_rows, realnum dt, std::string &error) {
  error.clear();
  if (!levels || !transitions || levels > size_t(std::numeric_limits<int>::max()) ||
      levels > std::numeric_limits<size_t>::max() / levels ||
      gamma_matrix.size() != levels * levels || !std::isfinite(double(dt)) || dt <= 0) {
    error = "invalid multilevel dimensions or timestep";
    return false;
  }
  const size_t max = std::numeric_limits<size_t>::max();
  if (ntot > max / levels || active_rows > max / transitions ||
      active_rows * transitions > max / 2 ||
      (ntot && 2 * active_rows * transitions > max / ntot)) {
    error = "multilevel storage extent overflow";
    return false;
  }
  const size_t polarization_elements = 2 * active_rows * transitions * ntot;
  const size_t level_square = levels * levels;
  const size_t population_elements = ntot * levels;
  if (level_square > max - levels || level_square + levels > max - population_elements ||
      level_square + levels + population_elements > max - polarization_elements) {
    error = "multilevel storage extent overflow";
    return false;
  }
  const size_t payload_elements =
      level_square + levels + population_elements + polarization_elements;
  if (!payload_elements ||
      payload_elements - 1 > (max - sizeof(multilevel_data)) / sizeof(realnum)) {
    error = "multilevel allocation byte size overflow";
    return false;
  }
  if (active_rows > max / 2 || 2 * active_rows > max / transitions ||
      2 * active_rows * transitions > max / sizeof(realnumP)) {
    error = "multilevel pointer-table size overflow";
    return false;
  }
#ifndef HAVE_LAPACK
  error = "LAPACK is needed for resident multilevel support";
  return false;
#else
  std::vector<realnum> matrix(level_square);
  for (size_t i = 0; i < level_square; ++i) {
    matrix[i] = realnum(gamma_matrix[i]);
    if (!std::isfinite(double(matrix[i]))) {
      error = "multilevel Gamma is not representable in realnum";
      return false;
    }
  }
  for (size_t i = 0; i < levels; ++i) {
    matrix[i * levels + i] = realnum(1) + matrix[i * levels + i] * dt / 2;
    if (!std::isfinite(double(matrix[i * levels + i]))) {
      error = "multilevel Gamma timestep matrix is nonfinite";
      return false;
    }
  }
  if (!invert(matrix.data(), int(levels))) {
    error = "multilevel Gamma timestep matrix is singular";
    return false;
  }
  for (realnum value : matrix)
    if (!std::isfinite(double(value))) {
      error = "multilevel Gamma inverse is nonfinite";
      return false;
    }
  return true;
#endif
}

void *multilevel_susceptibility::new_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2],
                                                   const grid_volume &gv) const {
  size_t num = 0; // number of P components
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W)) num += 2 * gv.ntot();
  }
  size_t sz = sizeof(multilevel_data) + sizeof(realnum) * (L * L + L + gv.ntot() * L + num * T - 1);
  multilevel_data *d = (multilevel_data *)malloc(sz);
  if (d == NULL) meep::abort("%s:%i:out of memory(%lu)", __FILE__, __LINE__, sz);
  memset(d, 0, sz);
  d->sz_data = sz;
  return (void *)d;
}

void multilevel_susceptibility::init_internal_data(realnum *W[NUM_FIELD_COMPONENTS][2], realnum dt,
                                                   const grid_volume &gv, void *data) const {
  multilevel_data *d = (multilevel_data *)data;
  size_t sz_data = d->sz_data;
  /* Reinitialization is also used by fields::zero_fields/reset.  Release the
     convenience pointer tables before clearing the blob; otherwise memset
     loses their addresses and every reset leaks two tables per active row. */
  FOR_COMPONENTS(c) DOCMP2 {
    delete[] d->P[c][cmp];
    delete[] d->P_prev[c][cmp];
  }
  memset(d, 0, sz_data);
  d->sz_data = sz_data;
  size_t ntot = d->ntot = gv.ntot();

  /* d->data points to a big block of data that holds GammaInv, P,
     P_prev, Ntmp, and N.  We also initialize a bunch of convenience
     pointer in d to point to the corresponding data in d->data, so
     that we don't have to remember in other functions how d->data is
     laid out. */

  d->GammaInv = d->data;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      d->GammaInv[i * L + j] = (i == j) + Gamma[i * L + j] * dt / 2;
  if (!invert(d->GammaInv, L))
    meep::abort("multilevel_susceptibility: I + Gamma*dt/2 matrix singular");

  realnum *P = d->data + L * L;
  realnum *P_prev = P + ntot;
  FOR_COMPONENTS(c) DOCMP2 {
    if (needs_P(c, cmp, W)) {
      d->P[c][cmp] = new realnumP[T];
      d->P_prev[c][cmp] = new realnumP[T];
      for (int t = 0; t < T; ++t) {
        d->P[c][cmp][t] = P;
        d->P_prev[c][cmp][t] = P_prev;
        P += 2 * ntot;
        P_prev += 2 * ntot;
      }
    }
  }

  d->Ntmp = P;
  d->N = P + L; // the last L*ntot block of the data

  // initial populations
  for (size_t i = 0; i < ntot; ++i)
    for (int l = 0; l < L; ++l)
      d->N[i * L + l] = N0[l];
}

void multilevel_susceptibility::delete_internal_data(void *data) const {
  if (data) {
    multilevel_data *d = (multilevel_data *)data;
    FOR_COMPONENTS(c) DOCMP2 {
      delete[] d->P[c][cmp];
      delete[] d->P_prev[c][cmp];
    }
    free(data);
  }
}

void *multilevel_susceptibility::copy_internal_data(void *data) const {
  multilevel_data *d = (multilevel_data *)data;
  if (!d) return 0;
  multilevel_data *dnew = (multilevel_data *)malloc(d->sz_data);
  memcpy(dnew, d, d->sz_data);
  size_t ntot = d->ntot;
  dnew->GammaInv = dnew->data;
  realnum *P = dnew->data + L * L;
  realnum *P_prev = P + ntot;
  FOR_COMPONENTS(c) DOCMP2 {
    if (d->P[c][cmp]) {
      dnew->P[c][cmp] = new realnumP[T];
      dnew->P_prev[c][cmp] = new realnumP[T];
      for (int t = 0; t < T; ++t) {
        dnew->P[c][cmp][t] = P;
        dnew->P_prev[c][cmp][t] = P_prev;
        P += 2 * ntot;
        P_prev += 2 * ntot;
      }
    }
  }
  dnew->Ntmp = P;
  dnew->N = P + L;
  return (void *)dnew;
}

bool multilevel_susceptibility::internal_layout(std::vector<InternalArrayLayout> &out,
                                                const grid_volume &gv,
                                                void *P_internal_data) const {
  out.clear();
  if (!P_internal_data) return true;
  if (L <= 0 || T <= 0) throw std::invalid_argument("invalid multilevel internal dimensions");
  multilevel_data *d = static_cast<multilevel_data *>(P_internal_data);
  const size_t ntot = size_t(gv.ntot());
  if (d->ntot != ntot || !d->GammaInv || !d->N || !d->Ntmp)
    throw std::invalid_argument("invalid multilevel internal storage");
  const size_t levels = size_t(L);
  const size_t transitions = size_t(T);
  if (levels > std::numeric_limits<size_t>::max() / levels ||
      ntot > std::numeric_limits<size_t>::max() / levels)
    throw std::overflow_error("multilevel internal layout extent overflow");
  const size_t gamma_elements = levels * levels;
  const size_t population_elements = ntot * levels;
  const uintptr_t blob_begin = reinterpret_cast<uintptr_t>(d);
  if (d->sz_data > std::numeric_limits<uintptr_t>::max() - blob_begin)
    throw std::overflow_error("multilevel internal storage extent overflow");
  const uintptr_t blob_end = blob_begin + d->sz_data;
  const auto checked_offset = [&](const realnum *p, size_t elements) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(p);
    if (!p || address < blob_begin || address > blob_end ||
        size_t(blob_end - address) / sizeof(realnum) < elements ||
        size_t(address - blob_begin) % sizeof(realnum))
      throw std::invalid_argument("multilevel internal row lies outside its storage blob");
    return size_t(address - blob_begin) / sizeof(realnum);
  };
  const size_t gamma_offset = checked_offset(d->GammaInv, gamma_elements);
  const size_t scratch_offset = checked_offset(d->Ntmp, levels);
  const size_t population_offset = checked_offset(d->N, population_elements);
  if (population_offset != scratch_offset + levels)
    throw std::invalid_argument("multilevel population scratch layout is invalid");
  size_t live_rows = 0;
  FOR_COMPONENTS(c) DOCMP2 {
    if (bool(d->P[c][cmp]) != bool(d->P_prev[c][cmp]))
      throw std::invalid_argument("incomplete multilevel P/P_prev storage");
    if (d->P[c][cmp]) {
      if (live_rows > std::numeric_limits<size_t>::max() - 2 * transitions)
        throw std::overflow_error("multilevel internal layout row count overflow");
      live_rows += 2 * transitions;
    }
  }
  if (live_rows > std::numeric_limits<size_t>::max() - 2)
    throw std::overflow_error("multilevel internal layout row count overflow");
  out.reserve(live_rows + 2);

  InternalArrayLayout gamma_inv;
  gamma_inv.name = "GammaInv";
  gamma_inv.element_type = InternalArrayLayout::realnum_value;
  gamma_inv.offset_elements = gamma_offset;
  gamma_inv.elements = gamma_elements;
  gamma_inv.c = Centered;
  gamma_inv.cmp = -1;
  out.push_back(gamma_inv);

  FOR_COMPONENTS(c) DOCMP2 {
    if (!d->P[c][cmp]) continue;
    for (int t = 0; t < T; ++t) {
      if (!d->P[c][cmp][t] || !d->P_prev[c][cmp][t])
        throw std::invalid_argument("incomplete multilevel transition storage");
      InternalArrayLayout p;
      p.name = "P";
      p.element_type = InternalArrayLayout::realnum_value;
      p.offset_elements = checked_offset(d->P[c][cmp][t], ntot);
      p.elements = ntot;
      p.c = c;
      p.cmp = cmp;
      out.push_back(p);
      InternalArrayLayout p_prev = p;
      p_prev.name = "P_prev";
      p_prev.offset_elements = checked_offset(d->P_prev[c][cmp][t], ntot);
      out.push_back(p_prev);
    }
  }

  InternalArrayLayout populations;
  populations.name = "N";
  populations.element_type = InternalArrayLayout::realnum_value;
  populations.offset_elements = population_offset;
  populations.elements = population_elements;
  populations.c = Centered;
  populations.cmp = -1;
  out.push_back(populations);
  return true;
}

int multilevel_susceptibility::num_cinternal_notowned_needed(component c,
                                                             void *P_internal_data) const {
  multilevel_data *d = (multilevel_data *)P_internal_data;
  if (!d) return 0;
  return d->P[c][0] ? T : 0;
}

realnum *multilevel_susceptibility::cinternal_notowned_ptr(int inotowned, component c, int cmp,
                                                           int n, void *P_internal_data) const {
  multilevel_data *d = (multilevel_data *)P_internal_data;
  if (!d || !d->P[c][cmp] || inotowned < 0 || inotowned >= T) // never true
    return NULL;
  return d->P[c][cmp][inotowned] + n;
}

void multilevel_susceptibility::update_P(realnum *W[NUM_FIELD_COMPONENTS][2],
                                         realnum *W_prev[NUM_FIELD_COMPONENTS][2], realnum dt,
                                         const grid_volume &gv, void *P_internal_data) const {
  multilevel_data *d = (multilevel_data *)P_internal_data;
  realnum dt2 = 0.5 * dt;

  // field directions and offsets for E * dP dot product.
  component cdot[3] = {Dielectric, Dielectric, Dielectric};
  ptrdiff_t o1[3], o2[3];
  int idot = 0;
  FOR_COMPONENTS(c) {
    if (d->P[c][0]) {
      if (idot == 3) meep::abort("bug in meep: too many polarization components");
      gv.yee2cent_offsets(c, o1[idot], o2[idot]);
      cdot[idot++] = c;
    }
  }

  // update N from W and P
  realnum *GammaInv = d->GammaInv;
  realnum *Ntmp = d->Ntmp;
  LOOP_OVER_VOL_OWNED(gv, Centered, i) {
    realnum *N = d->N + i * L; // N at current point, to update

    // Ntmp = (I - Gamma * dt/2) * N
    for (int l1 = 0; l1 < L; ++l1) {
      Ntmp[l1] = 0;
      for (int l2 = 0; l2 < L; ++l2) {
        Ntmp[l1] += ((l1 == l2) - Gamma[l1 * L + l2] * dt2) * N[l2];
      }
    }

    // compute E*8 at point i
    realnum E8[3][2] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    for (idot = 0; idot < 3 && cdot[idot] != Dielectric; ++idot) {
      realnum *w = W[cdot[idot]][0], *wp = W_prev[cdot[idot]][0];
      E8[idot][0] = w[i] + w[i + o1[idot]] + w[i + o2[idot]] + w[i + o1[idot] + o2[idot]] + wp[i] +
                    wp[i + o1[idot]] + wp[i + o2[idot]] + wp[i + o1[idot] + o2[idot]];
      if (W[cdot[idot]][1]) {
        w = W[cdot[idot]][1];
        wp = W_prev[cdot[idot]][1];
        E8[idot][1] = w[i] + w[i + o1[idot]] + w[i + o2[idot]] + w[i + o1[idot] + o2[idot]] +
                      wp[i] + wp[i + o1[idot]] + wp[i + o2[idot]] + wp[i + o1[idot] + o2[idot]];
      }
      else
        E8[idot][1] = 0;
    }

    // Ntmp = Ntmp + alpha * E * dP
    for (int t = 0; t < T; ++t) {
      // compute 32 * E * dP and 64 * E * P at point i
      realnum EdP32 = 0;
      realnum EPave64 = 0;
      realnum gperpdt = gamma[t] * pi * dt;
      for (idot = 0; idot < 3 && cdot[idot] != Dielectric; ++idot) {
        realnum *p = d->P[cdot[idot]][0][t], *pp = d->P_prev[cdot[idot]][0][t];
        realnum dP = p[i] + p[i + o1[idot]] + p[i + o2[idot]] + p[i + o1[idot] + o2[idot]] -
                     (pp[i] + pp[i + o1[idot]] + pp[i + o2[idot]] + pp[i + o1[idot] + o2[idot]]);
        realnum Pave2 = p[i] + p[i + o1[idot]] + p[i + o2[idot]] + p[i + o1[idot] + o2[idot]] +
                        (pp[i] + pp[i + o1[idot]] + pp[i + o2[idot]] + pp[i + o1[idot] + o2[idot]]);
        EdP32 += dP * E8[idot][0];
        EPave64 += Pave2 * E8[idot][0];
        if (d->P[cdot[idot]][1]) {
          p = d->P[cdot[idot]][1][t];
          pp = d->P_prev[cdot[idot]][1][t];
          dP = p[i] + p[i + o1[idot]] + p[i + o2[idot]] + p[i + o1[idot] + o2[idot]] -
               (pp[i] + pp[i + o1[idot]] + pp[i + o2[idot]] + pp[i + o1[idot] + o2[idot]]);
          Pave2 = p[i] + p[i + o1[idot]] + p[i + o2[idot]] + p[i + o1[idot] + o2[idot]] +
                  (pp[i] + pp[i + o1[idot]] + pp[i + o2[idot]] + pp[i + o1[idot] + o2[idot]]);
          EdP32 += dP * E8[idot][1];
          EPave64 += Pave2 * E8[idot][1];
        }
      }
      EdP32 *= 0.03125;    /* divide by 32 */
      EPave64 *= 0.015625; /* divide by 64 (extra factor of 1/2 is from P_current + P_previous) */
      for (int l = 0; l < L; ++l)
        Ntmp[l] += alpha[l * T + t] * EdP32 + alpha[l * T + t] * gperpdt * EPave64;
    }

    // N = GammaInv * Ntmp
    for (int l1 = 0; l1 < L; ++l1) {
      N[l1] = 0;
      for (int l2 = 0; l2 < L; ++l2)
        N[l1] += GammaInv[l1 * L + l2] * Ntmp[l2];
    }
  }

  // each P is updated as a damped harmonic oscillator
  for (int t = 0; t < T; ++t) {
    const realnum omega2pi = 2 * pi * omega[t], g2pi = gamma[t] * 2 * pi, gperp = gamma[t] * pi;
    const realnum omega0dtsqrCorrected = omega2pi * omega2pi * dt * dt + gperp * gperp * dt * dt;
    const realnum gamma1inv = 1 / (1 + g2pi * dt2), gamma1 = (1 - g2pi * dt2);
    const realnum dtsqr = dt * dt;
    // note that gamma[t]*2*pi = 2*gamma_perp as one would usually write it in SALT. -- AWC

    // figure out which levels this transition couples
    int lp = -1, lm = -1;
    for (int l = 0; l < L; ++l) {
      if (alpha[l * T + t] > 0) lp = l;
      if (alpha[l * T + t] < 0) lm = l;
    }
    if (lp < 0 || lm < 0) meep::abort("invalid alpha array for transition %d", t);

    FOR_COMPONENTS(c) DOCMP2 {
      if (d->P[c][cmp]) {
        const realnum *w = W[c][cmp], *s = sigma[c][component_direction(c)];
        const realnum st = sigmat[5 * t + component_direction(c)];
        if (w && s) {
          realnum *p = d->P[c][cmp][t], *pp = d->P_prev[c][cmp][t];

          ptrdiff_t o1, o2;
          gv.cent2yee_offsets(c, o1, o2);
          o1 *= L;
          o2 *= L;
          const realnum *N = d->N;

          // directions/strides for offdiagonal terms, similar to update_eh
          const direction d = component_direction(c);
          direction d1 = cycle_direction(gv.dim, d, 1);
          component c1 = direction_component(c, d1);
          const realnum *w1 = W[c1][cmp];
          const realnum *s1 = w1 ? sigma[c][d1] : NULL;
          direction d2 = cycle_direction(gv.dim, d, 2);
          component c2 = direction_component(c, d2);
          const realnum *w2 = W[c2][cmp];
          const realnum *s2 = w2 ? sigma[c][d2] : NULL;

          if (s1 || s2) { meep::abort("nondiagonal saturable gain is not yet supported"); }
          else { // isotropic
            LOOP_OVER_VOL_OWNED(gv, c, i) {
              realnum pcur = p[i];
              const realnum *Ni = N + i * L;
              // dNi is population inversion for this transition
              realnum dNi = 0.25 * (Ni[lp] + Ni[lp + o1] + Ni[lp + o2] + Ni[lp + o1 + o2] - Ni[lm] -
                                    Ni[lm + o1] - Ni[lm + o2] - Ni[lm + o1 + o2]);
              p[i] = gamma1inv * (pcur * (2 - omega0dtsqrCorrected) - gamma1 * pp[i] -
                                  dtsqr * (st * s[i] * w[i]) * dNi);
              pp[i] = pcur;
            }
          }
        }
      }
    }
  }
}

void multilevel_susceptibility::subtract_P(field_type ft,
                                           realnum *f_minus_p[NUM_FIELD_COMPONENTS][2],
                                           void *P_internal_data) const {
  multilevel_data *d = (multilevel_data *)P_internal_data;
  field_type ft2 = ft == E_stuff ? D_stuff : B_stuff; // for sources etc.
  size_t ntot = d->ntot;
  for (int t = 0; t < T; ++t) {
    FOR_FT_COMPONENTS(ft, ec) DOCMP2 {
      if (d->P[ec][cmp]) {
        component dc = field_type_component(ft2, ec);
        if (f_minus_p[dc][cmp]) {
          realnum *p = d->P[ec][cmp][t];
          realnum *fmp = f_minus_p[dc][cmp];
          for (size_t i = 0; i < ntot; ++i)
            fmp[i] -= p[i];
        }
      }
    }
  }
}

} // namespace meep
