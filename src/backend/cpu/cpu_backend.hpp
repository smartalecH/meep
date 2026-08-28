/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
*/

#ifndef MEEP_BACKEND_CPU_CPU_BACKEND_HPP
#define MEEP_BACKEND_CPU_CPU_BACKEND_HPP

#include "backend/backend.hpp"

namespace meep {

/* The CPU backend does not own storage. fields_chunk, structure_chunk,
   dft_chunk and the susceptibility objects keep their allocations
   (decision E); BackendState wraps the finalized catalog, and every operation
   dispatches to the code that was already there. */
class CpuBackend : public ExecutionBackend {
public:
  explicit CpuBackend(fields &f) : f_(f) {}

  BackendState *create_state(const StoragePlan &) override;
  void initialize(const InitializationPlan &, BackendState &) override;
  MaterialClassification classify_state(const StoragePlan &, BackendState &) override;
  void finalize_storage(const StoragePlan &, BackendState &) override;
  Executable *compile(const StepPlan &, BackendState &) override;
  void advance(Executable &, BackendState &, int num_steps) override;
  void read(ArrayRef, void *host_buffer, size_t bytes) override;
  void write(ArrayRef, const void *host_buffer, size_t bytes) override;
  void synchronize() override {}
  backend_capabilities capabilities() const override;
  bool requires_full_storage_preparation() const override { return false; }
  bool accepts(const execution_options &opts, std::string &why) const override;

private:
  fields &f_;
};

} // namespace meep

#endif // MEEP_BACKEND_CPU_CPU_BACKEND_HPP
