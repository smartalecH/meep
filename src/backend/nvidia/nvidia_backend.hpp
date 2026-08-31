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

struct NvidiaHostFallbackStatistics {
  size_t segment_executions;
  size_t callback_resolutions;
  size_t device_to_host_calls;
  size_t device_to_host_bytes;
  size_t host_to_device_calls;
  size_t host_to_device_bytes;
  size_t synchronizations;
  size_t steady_capacity_growths;

  NvidiaHostFallbackStatistics()
      : segment_executions(0), callback_resolutions(0), device_to_host_calls(0),
        device_to_host_bytes(0), host_to_device_calls(0), host_to_device_bytes(0),
        synchronizations(0), steady_capacity_growths(0) {}
};

struct NvidiaCwStatistics {
  CwSolveResult result;
  size_t reduction_count;
  size_t scalar_device_to_host_calls;
  size_t scalar_device_to_host_bytes;
  size_t source_scalar_host_to_device_calls;
  size_t source_scalar_host_to_device_bytes;
  size_t vector_host_to_device_bytes;
  size_t vector_device_to_host_bytes;
  size_t setup_scalar_device_to_host_calls;
  size_t setup_scalar_device_to_host_bytes;
  size_t setup_source_scalar_host_to_device_calls;
  size_t setup_source_scalar_host_to_device_bytes;
  size_t iteration_scalar_device_to_host_calls;
  size_t iteration_scalar_device_to_host_bytes;
  size_t iteration_source_scalar_host_to_device_calls;
  size_t iteration_source_scalar_host_to_device_bytes;
  size_t final_scalar_device_to_host_calls;
  size_t final_scalar_device_to_host_bytes;
  size_t final_source_scalar_host_to_device_calls;
  size_t final_source_scalar_host_to_device_bytes;
  size_t diagnostic_device_to_host_calls;
  size_t diagnostic_device_to_host_bytes;
  size_t pack_kernel_launches;
  size_t unpack_kernel_launches;
  size_t zero_kernel_launches;
  size_t rhs_source_kernel_launches;
  size_t reconciliation_kernel_launches;
  size_t vector_kernel_launches;
  size_t operator_kernel_launches;
  size_t reduction_kernel_launches;
  size_t timestep_kernel_launches;
  size_t finite_check_kernel_launches;
  size_t final_dft_kernel_launches;
  size_t kernel_launches;
  size_t setup_kernel_launches;
  size_t iteration_kernel_launches;
  size_t final_kernel_launches;
  size_t iteration_operator_applications;
  size_t iteration_reduction_count;
  size_t iteration_pack_kernel_launches;
  size_t iteration_unpack_kernel_launches;
  size_t iteration_reconciliation_kernel_launches;
  size_t iteration_vector_kernel_launches;
  size_t iteration_operator_kernel_launches;
  size_t iteration_reduction_kernel_launches;
  size_t iteration_timestep_kernel_launches;
  size_t timestep_kernel_launches_per_operator;
  size_t reconciliation_kernel_launches_per_operator;
  size_t workspace_capacity_bytes;
  size_t workspace_allocations;
  bool valid;

  NvidiaCwStatistics()
      : reduction_count(0), scalar_device_to_host_calls(0), scalar_device_to_host_bytes(0),
        source_scalar_host_to_device_calls(0), source_scalar_host_to_device_bytes(0),
        vector_host_to_device_bytes(0), vector_device_to_host_bytes(0),
        setup_scalar_device_to_host_calls(0), setup_scalar_device_to_host_bytes(0),
        setup_source_scalar_host_to_device_calls(0), setup_source_scalar_host_to_device_bytes(0),
        iteration_scalar_device_to_host_calls(0), iteration_scalar_device_to_host_bytes(0),
        iteration_source_scalar_host_to_device_calls(0),
        iteration_source_scalar_host_to_device_bytes(0), final_scalar_device_to_host_calls(0),
        final_scalar_device_to_host_bytes(0), final_source_scalar_host_to_device_calls(0),
        final_source_scalar_host_to_device_bytes(0), diagnostic_device_to_host_calls(0),
        diagnostic_device_to_host_bytes(0), pack_kernel_launches(0),
        unpack_kernel_launches(0), zero_kernel_launches(0), rhs_source_kernel_launches(0),
        reconciliation_kernel_launches(0), vector_kernel_launches(0),
        operator_kernel_launches(0), reduction_kernel_launches(0), timestep_kernel_launches(0),
        finite_check_kernel_launches(0),
        final_dft_kernel_launches(0), kernel_launches(0), setup_kernel_launches(0),
        iteration_kernel_launches(0), final_kernel_launches(0),
        iteration_operator_applications(0), iteration_reduction_count(0),
        iteration_pack_kernel_launches(0), iteration_unpack_kernel_launches(0),
        iteration_reconciliation_kernel_launches(0), iteration_vector_kernel_launches(0),
        iteration_operator_kernel_launches(0), iteration_reduction_kernel_launches(0),
        iteration_timestep_kernel_launches(0), timestep_kernel_launches_per_operator(0),
        reconciliation_kernel_launches_per_operator(0), workspace_capacity_bytes(0),
        workspace_allocations(0), valid(false) {}
};

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
double derive_noisy_amplitude(double omega_0, double gamma, double noise_amplitude, double dt);
gyrotropic_coefficients derive_gyrotropic_coefficients(double omega_0, double gamma,
                                                       double alpha,
                                                       const double gyro_tensor[3][3], double dt,
                                                       gyrotropy_model model,
                                                       const direction order[3]);

/* Collective pre-step validation kept CUDA-free so rank-asymmetric target,
   countdown, and copy-on-write mutations can be rejected before any rank
   enters host material mixing. */
void validate_material_phase_state(const fields &f, uint64_t expected_target_signature);

} // namespace nvidia

class NvidiaBackend : public ExecutionBackend {
public:
  NvidiaBackend(fields &f, const execution_options &options, int selected_device);
  ~NvidiaBackend() override;

  BackendState *create_state(const StoragePlan &plan) override;
  void initialize(const InitializationPlan &plan, BackendState &state) override;
  MaterialClassification classify_state(const StoragePlan &plan, BackendState &state) override;
  void finalize_storage(const StoragePlan &plan, const MaterialClassification &classification,
                        BackendState &state) override;

  Executable *compile(const StepPlan &plan, BackendState &state) override;
  void advance(Executable &executable, BackendState &state, int num_steps) override;
  void refresh_noisy_seed(const RandomSeedSnapshot &candidate, BackendState &state) override;
  void commit_noisy_seed(BackendState &state) noexcept override;
  void discard_noisy_seed(BackendState &state) noexcept override;
  bool supports_magnetic_synchronization() const override { return true; }
  void preflight_magnetic_transition(Executable &executable, BackendState &state,
                                     bool synchronize) override;
  void synchronize_magnetic_fields(Executable &executable, BackendState &state) override;
  void restore_magnetic_fields(Executable &executable, BackendState &state) override;

  bool supports_host_custom_fallback() const override { return true; }
  void preflight_host_custom_fallback(Executable &executable, BackendState &state) override;
  void validate_host_custom_rebuild() override;
  void validate_host_custom_plan(const StepPlan &plan, BackendState &state) override;

  bool supports_cw(const CwSolveRequest &request, std::string &why) const override;
  Executable *preflight_cw(const CwSolveRequest &request, const StepPlan &step_plan,
                           const CwPlan &cw_plan, Executable *cached,
                           BackendState &state) override;
  CwSolveResult solve_cw(const CwSolveRequest &request, const StepPlan &step_plan,
                         const CwPlan &cw_plan, Executable &ordinary, Executable &cw,
                         BackendState &state, CwSolveSession &session) override;

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
  NvidiaCwStatistics cw_statistics_for_testing() const;
  NvidiaHostFallbackStatistics host_fallback_statistics_for_testing() const;

private:
  friend class NvidiaBackendState;

  NvidiaBackendState &checked_state(BackendState &state) const;
  NvidiaBackendState *current_state() const;
  NvidiaExecutable &checked_executable(Executable &executable,
                                       const NvidiaBackendState &state) const;
  void execute_host_segment(NvidiaExecutable &executable, NvidiaBackendState &state,
                            size_t operation_index, size_t segment_index);
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
