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

// Dump/load raw fields data to/from an HDF5 file.  Only
// works if the number of processors/chunks is the same.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <cerrno>
#include <string>
#include <unistd.h>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "backend/backend.hpp"
#include "backend/checkpoint.hpp"
#include "backend/initialization_plan.hpp"
#include "backend/lifecycle.hpp"

namespace meep {

void fields::dump_fields_chunk_field(h5file *h5f, bool single_parallel_file,
                                     const std::string &field_name,
                                     FieldPtrGetter field_ptr_getter) {
  /*
   * make/save a num_chunks x NUM_FIELD_COMPONENTS x 2 array counting
   * the number of entries in the 'field_name' array for each chunk.
   *
   * When 'single_parallel_file' is true, we are creating a single block of data
   * for ALL chunks (that are merged using MPI). Otherwise, we are just
   * making a copy of just the chunks that are ours.
   */
  int my_num_chunks = 0;
  for (int i = 0; i < num_chunks; i++) {
    my_num_chunks += (single_parallel_file || chunks[i]->is_mine());
  }
  size_t num_f_size = my_num_chunks * NUM_FIELD_COMPONENTS * 2;
  std::vector<size_t> num_f_(num_f_size);
  size_t my_ntot = 0;
  std::string local_error;
  for (int i = 0, chunk_i = 0; i < num_chunks; i++) {
    if (chunks[i]->is_mine()) {
      size_t ntot = chunks[i]->gv.ntot();
      for (int c = 0; c < NUM_FIELD_COMPONENTS; ++c) {
        for (int d = 0; d < 2; ++d) {
          realnum **f = field_ptr_getter(chunks[i], c, d);
          if (*f) {
            my_ntot += (num_f_[(chunk_i * NUM_FIELD_COMPONENTS + c) * 2 + d] = ntot);
            backend_read_host_range(*this, *f, ntot, local_error);
          }
        }
      }
    }
    chunk_i += (chunks[i]->is_mine() || single_parallel_file);
  }
  if (backend_host_refresh_required(*this))
    backend_reconcile_host_access(local_error, "fields::dump field storage");

  std::vector<size_t> num_f;
  if (single_parallel_file) {
    num_f.resize(num_f_size);
    sum_to_master(num_f_.data(), num_f.data(), num_f_size);
  }
  else { num_f = std::move(num_f_); }

  /* determine total dataset size and offset of this process's data */
  size_t my_start = 0;
  size_t ntotal = my_ntot;
  if (single_parallel_file) {
    my_start = partial_sum_to_all(my_ntot) - my_ntot;
    ntotal = sum_to_all(my_ntot);
  }

  size_t dims[3] = {(size_t)my_num_chunks, NUM_FIELD_COMPONENTS, 2};
  size_t start[3] = {0, 0, 0};
  std::string num_f_name = std::string("num_") + field_name;
  h5f->create_data(num_f_name.c_str(), 3, dims);
  if (am_master() || !single_parallel_file) { h5f->write_chunk(3, start, dims, num_f.data()); }

  /* write the data */
  h5f->create_data(field_name.c_str(), 1, &ntotal, false /* append_data */,
                   false /* single_precision */);
  for (int i = 0; i < num_chunks; i++) {
    if (chunks[i]->is_mine()) {
      size_t ntot = chunks[i]->gv.ntot();
      for (int c = 0; c < NUM_FIELD_COMPONENTS; ++c) {
        for (int d = 0; d < 2; ++d) {
          realnum **f = field_ptr_getter(chunks[i], c, d);
          if (*f) {
            h5f->write_chunk(1, &my_start, &ntot, *f);
            my_start += ntot;
          }
        }
      }
    }
  }
}

void fields::dump(const char *filename, bool single_parallel_file) {
  if (verbosity > 0) {
    printf("creating fields output file \"%s\" (%d)...\n", filename, single_parallel_file);
  }

  CheckpointImage image;
  std::string local_error;
  try { image = CheckpointTransaction::snapshot(*this); }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown checkpoint snapshot failure"; }
  backend_reconcile_host_access(local_error, "fields::dump checkpoint staging");
  /* Recipe identities are global, but their canonical descriptors are
     rank-local. Combine them only after every rank has completed fallible
     staging, at this fixed collective boundary. */
  const uint64_t disk_mask = (UINT64_C(1) << 52) - 1;
  image.source_definition_signature =
      uint64_t(sum_to_all(size_t(image.source_definition_signature))) & disk_mask;
  image.dft_recipe_signature =
      uint64_t(sum_to_all(size_t(image.dft_recipe_signature))) & disk_mask;
  image.dft_native_signature =
      uint64_t(sum_to_all(size_t(image.dft_native_signature))) & disk_mask;
  image.material_recipe_signature =
      uint64_t(sum_to_all(size_t(image.material_recipe_signature))) & disk_mask;
  image.material_native_signature =
      uint64_t(sum_to_all(size_t(image.material_native_signature))) & disk_mask;

  const std::string temporary = std::string(filename) + ".tmp";
  local_error.clear();
  try { checkpoint_fail_if_requested(CheckpointFailurePoint::write); }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown checkpoint write preflight failure"; }
  backend_reconcile_host_access(local_error, "fields::dump checkpoint write preflight");

  try {
    {
      h5file file(temporary.c_str(), h5file::WRITE, single_parallel_file, !single_parallel_file);

  // Write out the current time 't'
  size_t dims[1] = {1};
  size_t start[1] = {0};
  size_t _t[1] = {(size_t)t};
  file.create_data("t", 1, dims);
  if (am_master() || !single_parallel_file) file.write_chunk(1, start, dims, _t);

  dump_fields_chunk_field(&file, single_parallel_file, "f",
                          [](fields_chunk *chunk, int c, int d) { return &(chunk->f[c][d]); });
  dump_fields_chunk_field(&file, single_parallel_file, "f_u",
                          [](fields_chunk *chunk, int c, int d) { return &(chunk->f_u[c][d]); });
  dump_fields_chunk_field(&file, single_parallel_file, "f_w",
                          [](fields_chunk *chunk, int c, int d) { return &(chunk->f_w[c][d]); });
  dump_fields_chunk_field(&file, single_parallel_file, "f_cond",
                          [](fields_chunk *chunk, int c, int d) { return &(chunk->f_cond[c][d]); });
  dump_fields_chunk_field(
      &file, single_parallel_file, "f_bfast",
      [](fields_chunk *chunk, int c, int d) { return &(chunk->f_bfast[c][d]); });
  dump_fields_chunk_field(
      &file, single_parallel_file, "f_w_prev",
      [](fields_chunk *chunk, int c, int d) { return &(chunk->f_w_prev[c][d]); });

  /* Sharded checkpoint traversal is rank-local: different ranks may own
     different numbers of chunks. Refresh every local DFT chain first, then
     reconcile exactly once while all ranks are still at the same boundary. */
  std::string dft_dump_error;
  if (!single_parallel_file) {
    for (int i = 0; i < num_chunks; ++i)
      if (chunks[i]->is_mine()) backend_read_dft_chain(chunks[i]->dft_chunks, dft_dump_error);
    backend_reconcile_host_access(dft_dump_error, "fields::dump DFT storage");
  }

  // Dump DFT chunks.
  for (int i = 0; i < num_chunks; i++) {
    if (single_parallel_file || chunks[i]->is_mine()) {
      char dataname[1024];
      snprintf(dataname, 1024, "chunk%02d", i);
      save_dft_hdf5(chunks[i]->dft_chunks, dataname, &file, 0, single_parallel_file,
                    !single_parallel_file);
    }
  }
      CheckpointTransaction::write_manifest(file, image, single_parallel_file);
    }
  }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown checkpoint write failure"; }
  if (!local_error.empty() && ((single_parallel_file && am_master()) || !single_parallel_file))
    std::remove(temporary.c_str());
  backend_reconcile_host_access(local_error, "fields::dump checkpoint write");

  all_wait();
  const bool publish_owner = !single_parallel_file || am_master();
  const std::string backup = std::string(filename) + ".bak";
  bool had_previous = false;
  bool published = false;

  /* A sharded checkpoint is one collective transaction even though each rank
     owns a different pathname. Preserve every old shard before publishing any
     replacement; an asymmetric rename failure rolls back all successful
     peers before the collective error escapes. */
  local_error.clear();
  try {
    checkpoint_fail_if_requested(CheckpointFailurePoint::rename_backup);
    if (publish_owner) {
      /* A failed prior rollback deliberately leaves the old checkpoint under
         .bak and no target, never a newly-published partial target. Recover
         that authoritative image before beginning another transaction. */
      errno = 0;
      const bool target_exists = access(filename, F_OK) == 0;
      const int target_errno = errno;
      errno = 0;
      const bool backup_exists = access(backup.c_str(), F_OK) == 0;
      if (!target_exists && target_errno != ENOENT)
        throw std::runtime_error(std::string("checkpoint target probe failed: ") +
                                 strerror(target_errno));
      if (!target_exists && backup_exists && std::rename(backup.c_str(), filename) != 0)
        throw std::runtime_error(std::string("checkpoint prior-backup recovery failed: ") +
                                 strerror(errno));
      if (target_exists && backup_exists && std::remove(backup.c_str()) != 0)
        throw std::runtime_error(std::string("checkpoint stale-backup removal failed: ") +
                                 strerror(errno));
      errno = 0;
      if (std::rename(filename, backup.c_str()) == 0)
        had_previous = true;
      else if (errno != ENOENT)
        throw std::runtime_error(std::string("checkpoint backup rename failed: ") +
                                 strerror(errno));
    }
  }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown checkpoint backup failure"; }
  const bool backup_failed = or_to_all(!local_error.empty());
  if (backup_failed) {
    std::string restore_error;
    if (publish_owner && had_previous) try {
        checkpoint_fail_if_requested(CheckpointFailurePoint::rename_restore);
        if (std::rename(backup.c_str(), filename) != 0)
          throw std::runtime_error(std::string("checkpoint backup restore failed: ") +
                                   strerror(errno));
      }
      catch (const std::exception &e) { restore_error = e.what(); }
      catch (...) { restore_error = "unknown checkpoint backup restore failure"; }
    if (publish_owner) std::remove(temporary.c_str());
    const bool restore_failed = or_to_all(!restore_error.empty());
    if (restore_failed)
      backend_reconcile_host_access(restore_error, "fields::dump checkpoint backup restore");
    backend_reconcile_host_access(local_error, "fields::dump checkpoint backup");
  }

  local_error.clear();
  try {
    checkpoint_fail_if_requested(CheckpointFailurePoint::rename_publish);
    if (publish_owner) {
      if (std::rename(temporary.c_str(), filename) != 0)
        throw std::runtime_error(std::string("checkpoint rename failed: ") + strerror(errno));
      published = true;
    }
  }
  catch (const std::exception &e) { local_error = e.what(); }
  catch (...) { local_error = "unknown checkpoint rename failure"; }
  const bool publish_failed = or_to_all(!local_error.empty());
  if (publish_failed) {
    std::string rollback_error;
    if (publish_owner) {
      if (published && std::remove(filename) != 0 && errno != ENOENT)
        rollback_error = std::string("checkpoint rollback remove failed: ") + strerror(errno);
      if (had_previous) try {
          checkpoint_fail_if_requested(CheckpointFailurePoint::rename_restore);
          if (std::rename(backup.c_str(), filename) != 0)
            throw std::runtime_error(std::string("checkpoint rollback restore failed: ") +
                                     strerror(errno));
          rollback_error.clear();
        }
        catch (const std::exception &e) { rollback_error = e.what(); }
        catch (...) { rollback_error = "unknown checkpoint rollback restore failure"; }
      std::remove(temporary.c_str());
    }
    const bool rollback_failed = or_to_all(!rollback_error.empty());
    if (rollback_failed)
      backend_reconcile_host_access(rollback_error, "fields::dump checkpoint rollback");
    backend_reconcile_host_access(local_error, "fields::dump checkpoint publish");
  }
  if (publish_owner && had_previous) std::remove(backup.c_str());
  all_wait();
}

void fields::load_fields_chunk_field(h5file *h5f, bool single_parallel_file,
                                     const std::string &field_name,
                                     FieldPtrGetter field_ptr_getter) {
  int my_num_chunks = 0;
  for (int i = 0; i < num_chunks; i++) {
    my_num_chunks += (single_parallel_file || chunks[i]->is_mine());
  }
  size_t num_f_size = my_num_chunks * NUM_FIELD_COMPONENTS * 2;
  std::vector<size_t> num_f(num_f_size);

  int rank;
  size_t dims[3], _dims[3] = {(size_t)my_num_chunks, NUM_FIELD_COMPONENTS, 2};
  size_t start[3] = {0, 0, 0};

  std::string num_f_name = std::string("num_") + field_name;
  h5f->read_size(num_f_name.c_str(), &rank, dims, 3);
  if (rank != 3 || _dims[0] != dims[0] || _dims[1] != dims[1] || _dims[2] != dims[2])
    meep::abort("chunk mismatch in fields::load");
  if (am_master() || !single_parallel_file) h5f->read_chunk(3, start, dims, num_f.data());

  if (single_parallel_file) {
    h5f->prevent_deadlock();
    broadcast(0, num_f.data(), dims[0] * dims[1] * dims[2]);
  }

  /* allocate data as needed and check sizes */
  size_t my_ntot = 0;
  for (int i = 0, chunk_i = 0; i < num_chunks; i++) {
    if (chunks[i]->is_mine()) {
      size_t ntot = chunks[i]->gv.ntot();
      for (int c = 0; c < NUM_FIELD_COMPONENTS; ++c) {
        for (int d = 0; d < 2; ++d) {
          size_t n = num_f[(chunk_i * NUM_FIELD_COMPONENTS + c) * 2 + d];
          realnum **f = field_ptr_getter(chunks[i], c, d);
          if (n == 0) {
            delete[] *f;
            *f = NULL;
          }
          else {
            if (n != ntot) meep::abort("grid size mismatch %zd vs %zd in fields::load", n, ntot);
            // here we need to allocate the fields array for H in the PML region
            // because of H = B in fields_chunk::alloc_f whereby H is lazily
            // allocated in fields_chunk::update_eh during the first timestep
            const direction d_c = component_direction(c);
            if (!(*f) || (*f && is_magnetic(component(c)) && chunks[i]->s->sigsize[d_c] > 1))
              *f = new realnum[ntot];
            my_ntot += ntot;
          }
        }
      }
    }
    chunk_i += (chunks[i]->is_mine() || single_parallel_file);
  }

  /* determine total dataset size and offset of this process's data */
  size_t my_start = 0;
  size_t ntotal = my_ntot;
  if (single_parallel_file) {
    my_start = partial_sum_to_all(my_ntot) - my_ntot;
    ntotal = sum_to_all(my_ntot);
  }

  /* read the data */
  h5f->read_size(field_name.c_str(), &rank, dims, 1);
  if (rank != 1 || dims[0] != ntotal) {
    meep::abort("inconsistent data size for '%s' in fields::load (rank, dims[0]): "
                "(%d, %zu) != (1, %zu)",
                field_name.c_str(), rank, dims[0], ntotal);
  }
  for (int i = 0; i < num_chunks; i++) {
    if (chunks[i]->is_mine()) {
      size_t ntot = chunks[i]->gv.ntot();
      for (int c = 0; c < NUM_FIELD_COMPONENTS; ++c) {
        for (int d = 0; d < 2; ++d) {
          realnum **f = field_ptr_getter(chunks[i], c, d);
          if (*f) {
            h5f->read_chunk(1, &my_start, &ntot, *f);
            my_start += ntot;
          }
        }
      }
    }
  }
}

void fields::load(const char *filename, bool single_parallel_file) {
  if (verbosity > 0)
    printf("reading fields from file \"%s\" (%d)...\n", filename, single_parallel_file);

  std::string read_preflight_error;
  try { checkpoint_fail_if_requested(CheckpointFailurePoint::read); }
  catch (const std::exception &e) { read_preflight_error = e.what(); }
  catch (...) { read_preflight_error = "unknown checkpoint read preflight failure"; }
  backend_reconcile_host_access(read_preflight_error, "fields::load checkpoint read preflight");

  h5file file(filename, h5file::READONLY, single_parallel_file, !single_parallel_file);

  if (CheckpointTransaction::has_manifest(file)) {
    CheckpointImage image;
    std::string local_error;
    try { image = CheckpointTransaction::read_manifest(file, single_parallel_file); }
    catch (const std::exception &e) { local_error = e.what(); }
    catch (...) { local_error = "unknown checkpoint read failure"; }
    backend_reconcile_host_access(local_error, "fields::load checkpoint staging");
    CheckpointTransaction::validate_target(*this, image);
    CheckpointTransaction::commit(*this, image);
    return;
  }

  /* Legacy files retain their historical same-layout behavior. They are
     detected explicitly and never interpreted as a versioned checkpoint. */
  backend_prepare_checkpoint_load(*this);

  // Read in the current time 't'
  int rank;
  size_t dims[1] = {1};
  size_t start[1] = {0};
  size_t _t[1];
  file.read_size("t", &rank, dims, 1);
  if (rank != 1 || dims[0] != 1) meep::abort("time size mismatch in fields::load");
  if (am_master() || !single_parallel_file) file.read_chunk(1, start, dims, _t);

  if (single_parallel_file) {
    file.prevent_deadlock();
    broadcast(0, _t, dims[0]);
  }

  t = static_cast<int>(_t[0]);
  calc_sources(time());

  load_fields_chunk_field(&file, single_parallel_file, "f",
                          [](fields_chunk *chunk, int c, int d) { return &(chunk->f[c][d]); });
  load_fields_chunk_field(&file, single_parallel_file, "f_u",
                          [](fields_chunk *chunk, int c, int d) { return &(chunk->f_u[c][d]); });
  load_fields_chunk_field(&file, single_parallel_file, "f_w",
                          [](fields_chunk *chunk, int c, int d) { return &(chunk->f_w[c][d]); });
  load_fields_chunk_field(&file, single_parallel_file, "f_cond",
                          [](fields_chunk *chunk, int c, int d) { return &(chunk->f_cond[c][d]); });
  load_fields_chunk_field(
      &file, single_parallel_file, "f_bfast",
      [](fields_chunk *chunk, int c, int d) { return &(chunk->f_bfast[c][d]); });
  load_fields_chunk_field(
      &file, single_parallel_file, "f_w_prev",
      [](fields_chunk *chunk, int c, int d) { return &(chunk->f_w_prev[c][d]); });

  /* Defer sharded publication failures until every rank has completed its
     rank-local loop, then reconcile once at the common outer boundary. */
  std::string dft_load_error;

  // Load DFT chunks.
  for (int i = 0; i < num_chunks; i++) {
    if (single_parallel_file || chunks[i]->is_mine()) {
      char dataname[1024];
      snprintf(dataname, 1024, "chunk%02d", i);
      load_dft_hdf5(chunks[i]->dft_chunks, dataname, &file, 0, single_parallel_file,
                    single_parallel_file ? NULL : &dft_load_error);
    }
  }
  if (!single_parallel_file)
    backend_reconcile_host_access(dft_load_error, "fields::load DFT storage");
  clear_pending_material_region(*this);
}

} // namespace meep
