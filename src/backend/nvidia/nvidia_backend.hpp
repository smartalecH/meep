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

#include <utility>
#include <vector>

#include "backend/backend.hpp"
#include "backend/graph_plan.hpp"

namespace meep {

class NvidiaBackendState;
class NvidiaExecutable;
struct NvidiaCompiledOperation;

struct NvidiaMaterialInitializationStatistics {
  size_t compact_input_host_to_device_calls;
  size_t compact_input_host_to_device_bytes;
  size_t owned_ir_bytes;
  size_t dense_oracle_bytes;
  size_t dense_output_host_to_device_calls;
  size_t dense_output_host_to_device_bytes;
  size_t tiled_output_host_to_device_calls;
  size_t tiled_output_host_to_device_bytes;
  size_t logical_output_bytes;
  size_t callback_scratch_bytes;
  size_t upload_descriptor_bytes;
  size_t classification_status_bytes;
  size_t classification_result_bytes;
  size_t decoded_parameter_bytes;
  size_t absorber_profile_bytes;
  size_t pml_profile_bytes;
  size_t file_sample_bytes;
  size_t grid_weight_bytes;
  size_t geometry_object_bytes;
  size_t geometry_image_bytes;
  size_t geometry_value_bytes;
  size_t geometry_analytic_bytes;
  size_t geometry_patch_bytes;
  size_t constant_fill_kernel_launches;
  size_t conductivity_kernel_launches;
  size_t file_table_kernel_launches;
  size_t grid_table_kernel_launches;
  size_t geometry_bulk_kernel_launches;
  size_t geometry_analytic_kernel_launches;
  size_t geometry_patch_kernel_launches;
  size_t pointwise_kernel_launches;
  size_t pml_kernel_launches;
  size_t absorber_points_evaluated;
  size_t file_points_evaluated;
  size_t grid_points_evaluated;
  size_t geometry_bulk_points;
  size_t geometry_analytic_points;
  size_t geometry_patch_points;
  size_t synchronizations;
  bool device_native;
  bool valid;

  NvidiaMaterialInitializationStatistics()
      : compact_input_host_to_device_calls(0), compact_input_host_to_device_bytes(0),
        owned_ir_bytes(0), dense_oracle_bytes(0), dense_output_host_to_device_calls(0),
        dense_output_host_to_device_bytes(0), tiled_output_host_to_device_calls(0),
        tiled_output_host_to_device_bytes(0), logical_output_bytes(0),
        callback_scratch_bytes(0), upload_descriptor_bytes(0),
        classification_status_bytes(0), classification_result_bytes(0),
        decoded_parameter_bytes(0), absorber_profile_bytes(0), pml_profile_bytes(0),
        file_sample_bytes(0), grid_weight_bytes(0), geometry_object_bytes(0),
        geometry_image_bytes(0), geometry_value_bytes(0), geometry_analytic_bytes(0),
        geometry_patch_bytes(0), constant_fill_kernel_launches(0),
        conductivity_kernel_launches(0), file_table_kernel_launches(0),
        grid_table_kernel_launches(0), geometry_bulk_kernel_launches(0),
        geometry_analytic_kernel_launches(0), geometry_patch_kernel_launches(0),
        pointwise_kernel_launches(0), pml_kernel_launches(0),
        absorber_points_evaluated(0), file_points_evaluated(0), grid_points_evaluated(0),
        geometry_bulk_points(0), geometry_analytic_points(0), geometry_patch_points(0),
        synchronizations(0),
        device_native(false), valid(false) {}
};

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

struct NvidiaGraphStatistics {
  GraphExecutionMode requested;
  bool enabled;
  bool valid;
  size_t segment_count;
  size_t capture_count;
  size_t instantiate_count;
  size_t scalar_write_count;
  size_t launch_count;
  size_t boundary_count;
  size_t magnetic_segment_count;
  size_t magnetic_capture_count;
  size_t magnetic_instantiate_count;
  size_t magnetic_scalar_write_count;
  size_t magnetic_launch_count;
  size_t magnetic_boundary_count;

  NvidiaGraphStatistics()
      : requested(GraphExecutionMode::automatic), enabled(false), valid(false),
        segment_count(0), capture_count(0), instantiate_count(0), scalar_write_count(0),
        launch_count(0), boundary_count(0), magnetic_segment_count(0),
        magnetic_capture_count(0), magnetic_instantiate_count(0),
        magnetic_scalar_write_count(0), magnetic_launch_count(0),
        magnetic_boundary_count(0) {}
};

struct NvidiaExecutableCacheStatistics {
  uint64_t ordinary_resource_generation;
  uint64_t cw_resource_generation;
  size_t executable_build_count;
  size_t source_value_reuse_count;

  NvidiaExecutableCacheStatistics()
      : ordinary_resource_generation(0), cw_resource_generation(0),
        executable_build_count(0), source_value_reuse_count(0) {}
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
  bool graph_enabled;
  size_t graph_count;
  size_t graph_capture_count;
  size_t graph_instantiate_count;
  size_t graph_scalar_write_count;
  size_t graph_launch_count;
  size_t graph_rhs_launch_count;
  size_t graph_unpack_launch_count;
  size_t graph_pack_launch_count;
  size_t graph_vector_launch_count;
  size_t graph_reduction_launch_count;
  size_t graph_operator_launch_count;
  size_t graph_final_dft_launch_count;
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
        workspace_allocations(0), graph_enabled(false), graph_count(0),
        graph_capture_count(0), graph_instantiate_count(0), graph_scalar_write_count(0),
        graph_launch_count(0), graph_rhs_launch_count(0), graph_unpack_launch_count(0),
        graph_pack_launch_count(0), graph_vector_launch_count(0),
        graph_reduction_launch_count(0), graph_operator_launch_count(0),
        graph_final_dft_launch_count(0), valid(false) {}
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

namespace testing {
/* A process-local ceiling used only by the CUDA integration tests to prove
   that the combined resident-plus-initialization peak is rejected before the
   first candidate allocation. SIZE_MAX restores the physical-device budget. */
void set_initialization_memory_budget_for_testing(size_t bytes);
size_t initialization_memory_budget_for_testing();

/* Exercises the same rank-reconciliation seams used by graph configuration
   without requiring MPI timestepping to be enabled.  Each validity flag is
   deliberately local so MPI tests can inject one-rank failures and prove
   that every rank makes the same publication decision. */
struct graph_collective_probe {
  const char *mode;
  bool lowering_valid;
  bool validation_valid;
  bool runtime_capture_supported;
  bool program_graphable;
  bool allocation_valid;
  bool capture_valid;
  bool instantiate_valid;
  bool graph_exec_destroy_valid;
  bool graph_destroy_valid;
  bool graph_device_restore_valid;
};
GraphModeResolution reconcile_graph_execution_for_testing(const graph_collective_probe &probe);
} // namespace testing

} // namespace nvidia

class NvidiaBackend : public ExecutionBackend {
public:
  NvidiaBackend(fields &f, const execution_options &options, int selected_device);
  ~NvidiaBackend() override;

  void preflight_initialization(const InitializationPlan &plan) const override;
  BackendState *create_state(const StoragePlan &plan) override;
  void prepare_initialization(const InitializationPlan &plan, BackendState &state) override;
  void initialize(const InitializationPlan &plan, BackendState &state) override;
  bool enforces_material_fallback_policy() const override { return true; }
  bool supports_stable_material_refresh() const override { return true; }
  MaterialClassification classify_state(const StoragePlan &plan, BackendState &state) override;
  void finalize_storage(const StoragePlan &plan, const MaterialClassification &classification,
                        BackendState &state) override;

  Executable *compile(const StepPlan &plan, BackendState &state) override;
  bool refresh_source_values(const StepPlan &plan, Executable &executable,
                             BackendState &state) override;
  bool supports_atomic_cw_source_refresh() const override { return true; }
  void preflight_advance(Executable &executable, BackendState &state,
                         int num_steps) override;
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
  Executable *preflight_cw(const CwSolveRequest &request,
                           const StepPlan &ordinary_plan,
                           const StepPlan &step_plan, const CwPlan &cw_plan,
                           Executable &ordinary, Executable *cached,
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
  NvidiaMaterialInitializationStatistics material_initialization_statistics_for_testing() const;
  NvidiaGraphStatistics graph_statistics_for_testing() const;
  NvidiaExecutableCacheStatistics executable_cache_statistics_for_testing() const;
  void set_next_executable_generation_for_testing(uint64_t generation) {
    next_executable_generation_ = generation;
  }
  int device_ordinal_for_testing() const { return device_; }
  void clear_cw_graphs_for_testing();

private:
  friend class NvidiaBackendState;

  NvidiaBackendState &checked_state(BackendState &state) const;
  NvidiaBackendState *current_state() const;
  NvidiaExecutable &checked_executable(Executable &executable,
                                       const NvidiaBackendState &state) const;
  void execute_host_segment(NvidiaExecutable &executable, NvidiaBackendState &state,
                            size_t operation_index, size_t segment_index);
  bool launch_device_operation(NvidiaExecutable &executable, NvidiaBackendState &state,
                               const NvidiaCompiledOperation &operation,
                               const StepScalars *graph_scalars,
                               bool account_cw);
  void configure_graph_execution(const StepPlan &plan, NvidiaExecutable &executable,
                                 NvidiaBackendState &state);
  void configure_cw_graph_execution(const StepPlan &plan, const CwPlan &cw_plan,
                                    NvidiaExecutable &executable, NvidiaBackendState &state);
  void execute_magnetic_half_step(NvidiaExecutable &executable, NvidiaBackendState &state);
  uint64_t claim_executable_generation();
  fields &f_;
  execution_options options_;
  int device_;
  GraphExecutionMode graph_mode_;
  bool graph_mode_parse_valid_;
  size_t device_memory_bytes_;
  uint64_t next_state_token_;
  uint64_t next_executable_generation_;
  mutable size_t pending_initialization_reserve_bytes_;
  mutable size_t pending_initialization_compact_bytes_;
  mutable size_t pending_initialization_scratch_bytes_;
  mutable std::vector<std::pair<StorageKey, size_t> >
      pending_initialization_classification_rows_;
  mutable MaterialRecipeDisposition pending_initialization_route_;
  mutable bool pending_initialization_reserve_valid_;
};

/* Device selection must already be resolved and collectively validated. This
   helper converts local construction failures to the factory's reason string;
   the shared factory is responsible for reducing that result across ranks. */
ExecutionBackend *make_nvidia_backend(fields &f, const execution_options &options,
                                      int selected_device, std::string &why);

} // namespace meep

#endif // MEEP_BACKEND_NVIDIA_NVIDIA_BACKEND_HPP
