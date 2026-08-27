# Bitwise-neutrality harness

Phase 1 of the backend-neutral refactor changes no arithmetic. Every PR in the
stack must therefore produce **bitwise-identical** raw state to its merge base,
across the whole configuration matrix, in both storage precisions, serially and
under MPI. Meep's normal suite asserts tolerances, not bits, so it cannot see
the class of bug this refactor is most likely to introduce.

## Layout

| File | Purpose |
|---|---|
| `configs.py` | The covering configuration matrix and the `Simulation` builder |
| `dump_state.py` | Runs one configuration and emits a JSON manifest of SHA-256 hashes |
| `run_matrix.py` | Drives every (config × ranks × threads) cell, one subprocess each |
| `compare_manifests.py` | Diffs two manifests; nonzero exit on any difference |

Golden hashes are deliberately **not** checked in: they are not stable across
compilers, libc, or HDF5 versions. Both sides are always built and run in the
same environment and compared against each other.

## Running it locally

```sh
$HOME/meep-phase1/bitwise.sh phase1/base phase1/pr1-lifecycle-foundations
```

That script builds both refs out-of-tree (cached by commit sha), runs the
matrix under each, and diffs. Extra arguments are forwarded to `run_matrix.py`,
so `... --ranks 1 2 4 --threads 1 4` widens the sweep and `--only NAME` narrows
it.

To run the matrix by hand against the current build tree:

```sh
PYTHONPATH=$PWD/python LD_LIBRARY_PATH=$PWD/src/.libs \
  python3 tests/bitwise/run_matrix.py --out /tmp/m.json --ranks 1 2
```

### The rpath trap

`libmeep.so` resolves through `RPATH`, and if the configure-time `LDFLAGS` put
the install prefix first, a test binary silently loads the **installed** library
instead of the one you just built. An early PR 1 run looked green this way
before the mistake was caught. `bitwise.sh` configures each comparison build
with `-Wl,-rpath,<builddir>/src/.libs` ahead of the prefix; if you build by
hand, either `make install` first or check with
`ldd tests/.libs/lt-<test> | grep libmeep`.

## What gets hashed

* **`fields::dump`** — the real `fields_chunk::f[c][cmp]` arrays plus `f_u`,
  `f_w`, `f_w_prev`, `f_cond`, and every `dft_chunk::dft[]` accumulator. This is
  the strong probe: raw storage, no interpolation.
* **`structure::dump`** — `chi1inv` and the susceptibility `sigma` arrays.
* **`get_array` over every component** present in the `grid_volume` — available
  in every configuration, including the ones the checkpoint writer refuses.
* **Reductions** — flux and DFT-norm values, hashed as float64 bit patterns.

## Determinism pins

The harness is worthless without these, and every one of them is a real source
of run-to-run variation:

* `OMP_NUM_THREADS` pinned by the driver (the matrix runs at 1 and at 4).
* `split_chunks_evenly=True` with an explicit `num_chunks`, so
  `choose_chunkdivision` does not vary with the host core count.
* `loop_tile_base_db` / `loop_tile_base_eh` fixed, so tiling does not vary.
* `mp.set_random_seed` before every run, for noisy susceptibilities.
* Fixed rank-to-chunk mapping via the even split.

Verified: two independent runs of the full matrix (27 configs × {1,2,4} ranks ×
{1,4} threads = 162 cells, 15,886 arrays) are bit-identical.

## Coverage gaps

Recorded honestly rather than papered over. Keep this list in sync with
`configs.py`.

1. **Raw dumps are unavailable for the dispersion axis.** `fields::dump` aborts
   with `non-null polarization_state in fields::dump (unsupported)`, and under
   MPI that abort is an `MPI_Abort`, not a catchable exception. Dispersive
   configurations (`_lorentzian`, `_drude`, `_noisy`, `_gyrotropic`,
   `_multilevel`) therefore get the `get_array` probe only. That probe sees the
   fields after `array_slice` has interpolated the Yee component onto the
   centered grid, so it is weaker — a bit difference has to survive an
   averaging step to be caught. Closing this needs either polarization support
   in the checkpoint writer or a harness-only raw dumper.
2. **`zero_fields_near_cylorigin` is not exercised.** It is a `fields`
   constructor argument that the Python `Simulation` does not expose. The
   `|m| > 1` origin rules are covered via `dcyl_m3`.
3. **`--enable-single` is CI-only.** The workflow builds both precisions; the
   local driver builds whatever `bitwise.sh` is configured for.
4. **`add_srcdata` and eigenmode-generated source tables** are not in the
   matrix yet; they matter from PR 6 on and should be added there.
5. **Absorber, near2far and `dft_ldos`** have one configuration each, below the
   "every axis value at least twice" bar.

## A pre-existing bug worth knowing about

`get_array_slice_dimensions` calls `am_now_working_on` before it can abort, and
when `meep::abort` unwinds as a C++ exception (the Python build) the matching
`finished_working()` never runs. About 30 caught failures then trip
`assert(was_working_on.size() <= MeepTimingStackSize)` in `time.cpp`. This is
why `dump_state.py` enumerates components with the non-throwing
`grid_volume::has_field` instead of probing with try/except.
