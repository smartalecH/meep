/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

/* NVIDIA backend ownership and the host/device coherence boundary.
 *
 * BACKEND-PRIVATE and CUDA-free. Device selection and its collective MPI
 * checks happen before construction; this class receives the resolved CUDA
 * ordinal and never reparses environment variables.
 */

#ifndef MEEP_BACKEND_NVIDIA_NVIDIA_BACKEND_HPP
#define MEEP_BACKEND_NVIDIA_NVIDIA_BACKEND_HPP

#include "backend/backend.hpp"

namespace meep {

class NvidiaBackendState;
class NvidiaExecutable;

namespace nvidia {

struct polarization_coefficients {
  double omega0dtsqr;
  double gamma1inv;
  double gamma1;
  double omega0dtsqr_denom;
};

struct gyrotropic_coefficients {
  double omega0dtsqr;
  double gamma1;
  double diagonal;
  double pt;
  double omega;
  double gamma;
  double alpha;
  double dt2pi;
  double gyro[3][3];
  double inverse[3][3];
};

polarization_coefficients derive_polarization_coefficients(double omega_0, double gamma,
                                                           double dt, bool drude);
gyrotropic_coefficients derive_gyrotropic_coefficients(double omega_0, double gamma,
                                                       double alpha,
                                                       const double gyro_tensor[3][3], double dt,
                                                       gyrotropy_model model,
                                                       const direction order[3]);

} // namespace nvidia

class NvidiaBackend : public ExecutionBackend {
public:
  NvidiaBackend(fields &f, const execution_options &options, int selected_device);
  ~NvidiaBackend() override;

  BackendState *create_state(const StoragePlan &plan) override;
  void initialize(const InitializationPlan &plan, BackendState &state) override;
  MaterialClassification classify_state(const StoragePlan &plan, BackendState &state) override;
  void finalize_storage(const StoragePlan &plan, BackendState &state) override;

  Executable *compile(const StepPlan &plan, BackendState &state) override;
  void advance(Executable &executable, BackendState &state, int num_steps) override;
  bool supports_magnetic_synchronization() const override { return true; }
  void preflight_magnetic_transition(Executable &executable, BackendState &state,
                                     bool synchronize) override;
  void synchronize_magnetic_fields(Executable &executable, BackendState &state) override;
  void restore_magnetic_fields(Executable &executable, BackendState &state) override;

  void read(ArrayRef ref, void *host_buffer, size_t bytes) override;
  void write(ArrayRef ref, const void *host_buffer, size_t bytes) override;
  bool supports_compact_dft_reductions() const override { return true; }
  void reduce_dft(const DftReductionRequest &request, std::complex<double> *local_result,
                  size_t result_count) override;
  void synchronize() override;
  backend_capabilities capabilities() const override;
  bool requires_full_storage_preparation() const override { return true; }
  void prepare_state_rebuild(BackendState &state, DirtyMask reasons) override;
  bool accepts(const execution_options &options, std::string &why) const override;

private:
  friend class NvidiaBackendState;

  NvidiaBackendState &checked_state(BackendState &state) const;
  NvidiaBackendState *current_state() const;
  NvidiaExecutable &checked_executable(Executable &executable,
                                       const NvidiaBackendState &state) const;
  void execute_magnetic_half_step(NvidiaExecutable &executable, NvidiaBackendState &state);

  fields &f_;
  execution_options options_;
  int device_;
  size_t device_memory_bytes_;
  uint64_t next_state_token_;
};

/* Device selection must already be resolved and collectively validated. This
   helper converts local construction failures to the factory's reason string;
   the shared factory is responsible for reducing that result across ranks. */
ExecutionBackend *make_nvidia_backend(fields &f, const execution_options &options,
                                      int selected_device, std::string &why);

} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_NVIDIA_BACKEND_HPP
