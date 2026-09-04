# Accelerator benchmark runners and arXiv 2506.16665 corpus adapter

## Normative multi-rank runner (PR7.5)

`run_benchmark.py` remains the legacy PR3 single-rank diagnostic. It is not a
multi-GPU evidence producer. `run_mpi_benchmark.py` is the version-2 normative
MPI result producer and validates `mpi_benchmark_result.schema.json` before
communicator rank 0 atomically publishes the final JSON file.

The launcher accepts independent `--python`, `--mpiexec`, `--pythonpath`, and
repeatable `--runtime-library-path` arguments. This is required when Python,
Meep, Open MPI, UCX, and ROCm come from different private prefixes. For
same-node provider-positive HIP runs it sets these variables before `MPI_Init`:

```text
OMPI_MCA_pml=ucx
UCX_TLS=self,sm,rocm_copy,rocm_ipc
ROCR_VISIBLE_DEVICES=<one explicit physical selector per rank>
```

CUDA runs retain `OMPI_MCA_opal_cuda_support=true`,
`UCX_TLS=self,sm,cuda_copy,cuda_ipc`, and `CUDA_VISIBLE_DEVICES`.

Route, overlap, graph mode, and precision are mandatory arguments and must
match the immutable manifest. The result records the exact launcher argv,
environment, timeout, MPI/UCX versions, manifest hash, communicator identity,
host/local rank, GPU identity, provider query, requested/resolved policies,
per-rank repetitions, and rank-local transport counters. Each repetition uses
the maximum rank duration as its critical path and records rank median and
imbalance. Every aggregate counter carries an explicit `sum` or `max` rule.
The runner accepts exactly 1, 2, 4, or 8 owning ranks and requires an explicit,
distinct device selector for every rank. The built-in acceptance path below
uses a collective point-field decay decision. Paper end-to-end manifests
currently request `total_field_energy`; the runner rejects that mode explicitly
because the resident multi-rank host-energy query remains a known crash,
instead of publishing a misleading fixed-step substitute.

Example two-rank provider-positive HIP run:

```sh
/home/alechammond/rocm-mpi/python/bin/python run_mpi_benchmark.py \
  --manifest /absolute/path/run-manifest.json \
  --physics-reference /absolute/path/cpu-native-baseline.json \
  --output /absolute/path/raw-result.json \
  --build-directory /absolute/path/meep-build \
  --accelerator hip --visible-devices 1,5 \
  --python /home/alechammond/rocm-mpi/python/bin/python \
  --mpiexec /home/alechammond/rocm-mpi/openmpi/bin/mpirun \
  --pythonpath /absolute/path/meep-build/python \
  --library-path /absolute/path/meep-build/src/.libs \
  --runtime-library-path /home/alechammond/rocm-mpi/openmpi/lib \
  --runtime-library-path /home/alechammond/rocm-mpi/ucx/lib \
  --runtime-library-path /absolute/path/rocm/lib \
  --toolkit-compiler /absolute/path/rocm/bin/hipcc \
  --rocm-smi /absolute/path/rocm/bin/rocm-smi \
  --ucx-info /home/alechammond/rocm-mpi/ucx/bin/ucx_info \
  --route direct --overlap required --graph required --precision native \
  --timeout 1800
```

`--prefix` remains the CUDA-compatible default. Explicit tool and path options
override it without resolving a virtual-environment Python symlink to the
system interpreter. The launcher records CUDA or ROCr visibility, all runtime
library paths, map/rank/bind policy, HIPCC or NVCC version, `rocm-smi` inventory
for HIP, build flags, executable/module hashes, CPU affinity/NUMA, GPU UUID,
physical selector, PCI BDF, and device NUMA node.

The CPU reference must be a hash-authenticated `cpu_native_baseline` bound to
the same paper corpus, case definitions, case ID, and physics-configuration
hash. Every repetition derives the declared observables from raw monitor
arrays. Non-ring mode-power ratios use linear interpolation at the exact
center wavelength; excess loss is `-10 log10(ratio)`. Ring extraction uses
SciPy's not-a-knot cubic spline on 1000 inclusive wavelength samples and the
declared half-depth crossings. No result is published unless every observable
passes the manifest tolerance.

After matching staged and direct runs, create the route comparison artifact:

```sh
/home/alechammond/meep-env/bin/python compare_mpi_benchmarks.py \
  --staged staged.json --direct direct.json --output route-comparison.json
```

The comparison requires identical manifests, references, builds, rank/GPU
mapping, simulation shape/steps, raw monitor arrays, derived observables, and
wire totals, then enforces the route-specific staging/direct byte contract.
Replay schema, hash, baseline, and raw-observable validation with
`validate_mpi_benchmark.py --result raw-result.json`.

For a checkout-independent provider gate, `run_transport_acceptance.py` runs a
small deterministic 1D case against selected transport routes. The AMD staged
gate selects only `cpu,staged`, resolves each process-visible ROCr selector to
a HIP PCI BDF before matching it to `rocm-smi`, checks NUMA-local process
affinity, and never launches a direct device-pointer MPI route. ROCr and
`rocm-smi` ordinal order differs on the validation host; ROCr selectors `1,5`
resolve to physical SMI cards 0/4 (`08:00.0`/`88:00.0`):

```sh
/home/alechammond/meep-env/bin/python run_transport_acceptance.py \
  --output-dir /tmp/meep-transport-acceptance \
  --routes cpu,staged \
  --prefix /home/alechammond/rocm-mpi/openmpi \
  --python /home/alechammond/meep-env/bin/python \
  --mpiexec /home/alechammond/rocm-mpi/openmpi/bin/mpirun \
  --visible-devices 1,5 \
  --rocm-smi /usr/local/fbcode/platform010/lib/rocm-latest/bin/rocm-smi \
  --pythonpath /tmp/meep-pr75-python-stage \
  --library-path /tmp/meep-rocm-python-build/src/.libs
```

The runner places the branch build library directory before the private MPI
prefix in each child `LD_LIBRARY_PATH`; the branch `libmeep` RUNPATH must place
that same private MPI prefix before its numerical dependency prefix. This keeps
the Python workers on the same MPI implementation as the launcher.

The legacy default still includes `direct` for CUDA/provider validation. Both
modes use collective point-field decay and compare every GPU sample to the CPU
tolerance. This is an acceptance-scale transport proof, not paper-performance
evidence.

Comparison schema v2 keeps device-buffer MPI and provider zero-copy as
different claims. A direct result, staged/direct bitwise result, and positive
device-buffer result must be present together. Provider zero-copy is a separate
optional object that requires a hash-addressed provider log plus its memory type
and selected IPC transport; the staged-only AMD gate emits neither claim.

For PR5 ROCm-aware direct validation, first run the native target. It preserves
the PR4 staged-only target separately, exercises both forced-direct and
automatic selection on two owners, and uses larger all-owner rings to validate
device identity and NUMA locality:

```sh
env -u LD_LIBRARY_PATH \
  OMPI_MCA_pml=ucx UCX_TLS=self,sm,rocm_copy,rocm_ipc \
  make -C /tmp/meep-rocm-build/tests -j1 amd-direct-mpi-check \
    AMD_MPIEXEC=/home/alechammond/rocm-mpi/openmpi/bin/mpirun
```

The target retains `nvidia-mpi-rocm-provider.log`, requires protocol lines for
both ROCm devices and selected `rocm_ipc/rocm_ipc` zero-copy, and rejects a
positive device-buffer result without that provider evidence. Its np4 mapping
uses ROCr selectors `1,3,5,7` for physical BDFs `08,18,88,98`; its np8 mapping
uses selectors `0..7` and asserts the runtime's validated BDF order. These are
host-specific defaults and must be overridden together on another system.

An AMD+MPI build also exposes `make -C tests amd-staged-transport-acceptance`.
Set `AMD_ACCEPTANCE_PREFIX` to the MPI installation used to build Meep and set
`AMD_ACCEPTANCE_PYTHON_CMD` and `AMD_ACCEPTANCE_PYTHONPATH` to the executable
and package from that same build; the MPI launcher, visible devices, output,
and library path are independently overridable make variables.

The corresponding PR5 `amd-direct-transport-acceptance` target first runs the
native direct/provider gates and then invokes the runner with
`--routes cpu,staged,direct` and the hash-addressed UCX provider log. It requires
the same explicit prefix, Python, and Python-package variables. A unit-tested
schema or dry-run is not a substitute for this Python end-to-end result.

Neither native target is a performance benchmark. Do not report an AMD
speedup from it. The pinned external paper corpus remains a separate,
hash-authenticated prerequisite for paper comparisons.

`run_self_contained_benchmark.py` supplies a corpus-independent performance
experiment for bring-up. It uses one fixed 3D vacuum case, initializes once,
warms up exactly 100 steps, and retains five sequential raw 100-step windows.
It publishes the slowest rank for every window. The validator fails closed on
steady allocation, graph recapture, full-field copy, host fallback, wrong
transport byte accounting, duplicate GPU identities, nonlocal CPU/GPU NUMA
placement, CPU field tolerance, or non-bitwise staged/direct fields. Direct
results require the matching staged artifact, so route comparisons cannot use
different devices or work.

Use a pinned one-thread CPU artifact and a disclosed CPU thread sweep before
launching GPU cases. The following illustrates the private-stack arguments;
repeat `--runtime-library-path` in exact loader order and use `ppr:1:package`,
`ppr:2:package`, or `ppr:4:package` for 2, 4, or 8 GPUs across this two-socket
host:

```sh
python=/home/alechammond/rocm-mpi/python/bin/python
$python run_self_contained_benchmark.py \
  --output /tmp/hip-staged-np2.json \
  --backend nvidia --accelerator hip --route staged --ranks 2 \
  --visible-devices 1,5 --omp-threads 1 \
  --python "$python" \
  --mpiexec /home/alechammond/rocm-mpi/openmpi/bin/mpirun \
  --build-directory /absolute/path/meep-build \
  --library-path /absolute/path/meep-build/python \
  --runtime-library-path /absolute/path/meep-build/src/.libs \
  --runtime-library-path /home/alechammond/rocm-mpi/openmpi/lib \
  --runtime-library-path /home/alechammond/rocm-mpi/ucx/lib \
  --runtime-library-path /absolute/path/rocm/lib \
  --runtime-library-path /absolute/path/numerical-dependencies/lib \
  --toolkit-compiler /absolute/path/rocm/bin/hipcc \
  --rocm-smi /absolute/path/rocm/bin/rocm-smi \
  --ucx-info /home/alechammond/rocm-mpi/ucx/bin/ucx_info \
  --map-by ppr:1:package --rank-by fill --bind-to core \
  --reference /tmp/cpu-best.json
```

Run the same command with `--route direct`, a distinct output, and
`--peer-route-result /tmp/hip-staged-np2.json`. Single-rank HIP uses one
explicit selector; 4/8-rank launches must similarly enumerate all devices.
The tiny built-in case is a portability and route-overhead measurement, not a
representative accelerator throughput claim; report slowdowns as readily as
speedups.

`run_amd_fixed_step_matrix.sh OUTPUT_DIRECTORY` reproduces the disclosed local
CPU sweep and HIP 1/2/4/8 matrix. Its private-stack paths are explicit defaults
and can be overridden with `PYTHON`, `MPIEXEC`, `UCX_PREFIX`, `ROCM_PREFIX`,
`MEEP_DEPS`, `MEEP_BUILD`, and `MEEP_PYTHONPATH`.

Workers never parse stdout. They call
`meep.active_communicator_allgather_json(payload)`, which must collectively
return the same communicator-rank-ordered list of JSON strings on every rank.
This helper uses Meep's active communicator, including split communicators;
`mpi4py.COMM_WORLD` is not an acceptable substitute. All workers verify the
manifest hash, communicator generation/size, rank coverage, requested policy,
resolved policy, and repetition count. Only active-communicator rank 0 writes.

Run the schema/reconciliation/atomic-publication tests with `make check`.

This directory contains an authenticated manifest and smoke/throughput runner for the six-device
benchmark corpus described in [arXiv:2506.16665](https://arxiv.org/html/2506.16665).
`benchmark_manifest.py` remains standard-library-only. `run_benchmark.py`
imports a branch-matched Meep build and `gdstk`, constructs the selected GDS
cell, runs a single rank on CPU or NVIDIA, and writes an authenticated
diagnostic result with raw finite DFT flux and mode-power arrays.

The executable scope is deliberately narrower than the paper: it uses fixed,
nondispersive performance-adaptation media and does not claim paper-equivalent
physics, two-GPU MPI scaling, speedup, or publication-ready results. Every PR3
runner output, including fixed-step output, is explicitly diagnostic-only.
The PSR case is rejected because the paper-derived inputs do not define the
Si3N4 top-cladding footprint and z bounds precisely enough to reproduce it.

## Provenance boundary

The paper text and tables are authoritative for solver settings and results.
The linked `JPPhotonics/fdtd-pipeline` repository is used only for GDS/YAML
geometry provenance. Its pinned commit post-dates parts of the paper, and its
current solver configuration is not treated as the exact experimental snapshot.

The GDS files are not vendored here. The upstream repository has a
peer-review/reproducibility statement rather than a conventional open-source
license, so redistribution requires separate confirmation.

Prepare the exact external input:

```sh
git clone https://github.com/JPPhotonics/fdtd-pipeline.git
git -C fdtd-pipeline checkout 622e0a9b7429eaf2335b1000b39e283544a198c4
python3 benchmark_manifest.py validate --fdtd-pipeline ./fdtd-pipeline
```

Validation fails if the checkout commit, any of the six GDS/YAML hashes, or the
layer-stack hash does not match. The validation report also records whether the
external checkout is dirty and preserves its porcelain status entries. A
single-case `manifest` invocation validates only that case's GDS/YAML plus the
shared layer stack; `validate` audits the entire six-case corpus.

`runner_cases.json` is the machine-readable adapter from those external assets
to the Meep runner. It fixes the GDS transform, layer/datatype and z-range
mappings, ports, source and monitor definitions, padding, all six PML faces,
the exact nondispersive material constants, and the field-energy decay region.
The 0.25 um transverse padding is total added span (0.125 um per side); the
earlier 4 um value could not fit several source planes inside the paper cells'
non-PML regions. The ring case separately identifies its
mode monitor and resonance observable. Every transmission/conversion monitor
is normalized by the forward source mode at the case's explicit
`input_incident` monitor. The emitted manifest also records the exact
reciprocal-endpoint conversion from the centered wavelength interval to Meep's
Gaussian center frequency and `fwidth`. These values are versioned benchmark
inputs rather than hidden runner defaults. Each manifest also contains the
expanded wavelength/frequency arrays and the resolved DFT decimation factor.

`benchmark_manifest.schema.json` defines the complete version-4 run document,
including the requested overlap and graph policies.
Result validation rechecks that schema and the bundled manifest-schema,
paper-reference, case-definition, result-schema, pinned-commit, GDS, YAML, and
layer-stack identities; a result cannot authenticate a handwritten partial
manifest merely by hashing it.

## Resolution conversion

The paper specifies cells per wavelength **in the highest-index material**.
Meep specifies pixels per micrometer. They are converted as:

```text
resolution_px_per_um = cells_per_material_wavelength
                       * n_max(lambda_min)
                       / lambda_min_um
```

The CLI requires `--n-max` so the value cannot be hidden or silently inferred.
By default it rounds upward to an integer resolution so the requested sampling
is not undershot. The manifest records the exact value, rounding policy,
selected resolution, and `dx`.

## Emit manifests

Smoke construction/short-run manifest:

```sh
python3 benchmark_manifest.py manifest \
  --fdtd-pipeline ./fdtd-pipeline \
  --device coupler \
  --mode smoke \
  --cells-per-material-wavelength 6 \
  --n-max 3.48 \
  --precision native \
  --output coupler-smoke.json
```

Fixed-step throughput manifest:

```sh
python3 benchmark_manifest.py manifest \
  --fdtd-pipeline ./fdtd-pipeline \
  --device ring \
  --mode fixed-step \
  --steps 10000 \
  --cells-per-material-wavelength 25 \
  --n-max 3.48 \
  --backend nvidia \
  --precision f32 \
  --ranks 1 \
  --mpi-transport none \
  --output ring-fixed.json
```

End-to-end decay-stop manifest, with a mandatory safety cap:

```sh
python3 benchmark_manifest.py manifest \
  --fdtd-pipeline ./fdtd-pipeline \
  --device mode_converter \
  --mode end-to-end \
  --max-steps 200000 \
  --cells-per-material-wavelength 20 \
  --n-max 3.48 \
  --precision mixed \
  --ranks 2 \
  --mpi-transport auto \
  --overlap auto \
  --graph auto \
  --output mode-converter-e2e.json
```

Available device IDs are `coupler`, `crossing`, `mmi2x2`, `mode_converter`,
`psr`, and `ring`.

The execution vocabulary is shared with the planned backend API:
`native|f32|mixed`. CPU runs accept only `native`; GPU storage/reduction modes
are not silently relabeled as CPU precision. Rank counts are positive integers,
while transport is checked against backend and rank count.

The manifest vocabulary includes future multi-rank configurations, but the PR3
runner rejects them. Two GPUs may be exercised as independent rank-1 jobs; do
not use `mpirun -np 2` or describe two concurrent jobs as multi-GPU scaling.

`performance-adaptation` is the default material mode. `--material-mode paper`
is rejected unless `--material-validation PATH` names a passing, hashed JSON
artifact. That artifact must state the pole equation, Fourier time convention,
frequency and residue units, wavelength range, sample count, error norm,
tolerance, measured error, and per-material error. This prevents published
bare pole/residue numbers from being treated as an unproven Meep lowering. Its
reference hash and pole convention must match the bundled source, its tested
wavelength range must cover the requested source/resolution band, and each
material must independently satisfy the declared tolerance. The
reference also records that Lumerical's silicon MCM coefficients were not
published, so an exact three-solver material match cannot be reconstructed from
the paper alone.

## Benchmark modes

- `smoke`: fixed 10 steps by default, no warmup or repetition requirement.
- `fixed-step`: requires `--steps`; records 100 warmup steps and five measured
  repetitions for steady-state throughput.
- `end-to-end`: field-energy decay threshold `1e-5`, explicit check interval,
  and mandatory `--max-steps`. This is a documented Meep adaptation because the
  commercial solvers' auto-shutoff algorithms are not fully specified.

`run_benchmark.py` currently accepts `smoke` and `fixed-step`. A fixed-step run
constructs one simulation, performs the manifest's 100 untimed warmup steps
once, and then records five sequential windows of exactly the requested step
count. Smoke mode runs ten steps once with no warmup. Ten steps occur well
before the 20 nm Gaussian pulse peak and prove only construction, stepping, and
finite monitor access.

## Build and execute

Use a fresh Python-enabled build. Do not preload a new library beneath an old
installed SWIG extension. On aarch64, building `gdstk` may need the environment's
Qhull CMake package:

```sh
/path/to/meep-python -m venv --system-site-packages /tmp/meep-benchmark-venv
env CMAKE_PREFIX_PATH=/path/to/meep-env \
  /tmp/meep-benchmark-venv/bin/python -m pip install gdstk==1.0.1

mkdir -p /tmp/meep-benchmark-build /tmp/meep-benchmark-prefix
cd /tmp/meep-benchmark-build
/path/to/meep/configure \
  --prefix=/tmp/meep-benchmark-prefix --enable-maintainer-mode \
  --enable-shared --with-mpi --without-scheme \
  --with-libctl=/path/to/meep-env/share/libctl \
  --with-accelerator=nvidia --with-cuda=/usr/local/cuda \
  --with-cuda-host-compiler=/usr/bin/g++ --with-cuda-architectures=sm_100 \
  CC=/path/to/meep-env/bin/mpicc CXX=/path/to/meep-env/bin/mpicxx \
  F77=/path/to/meep-env/bin/aarch64-conda-linux-gnu-gfortran \
  PYTHON=/path/to/meep-env/bin/python3 \
  CFLAGS='-O3 -DNDEBUG -fPIC' CXXFLAGS='-O3 -DNDEBUG -fPIC' \
  NVCCFLAGS='-O3 -lineinfo' LDFLAGS='-L/path/to/meep-env/lib' \
  CPPFLAGS='-I/path/to/meep-env/include'
make -j16 && make install
```

Generate and run the smallest smoke case:

```sh
python3 benchmark_manifest.py manifest \
  --fdtd-pipeline ./fdtd-pipeline --device crossing --mode smoke \
  --cells-per-material-wavelength 6 --n-max 3.48 --bandwidth-nm 20 \
  --material-mode performance-adaptation --backend nvidia \
  --precision native --ranks 1 --mpi-transport none \
  --output /tmp/crossing-gpu-smoke.json

env CUDA_VISIBLE_DEVICES=0 MEEP_FINITE_CHECK=step \
  MEEP_SOURCE_TREE=/path/to/meep MEEP_BUILD_DIR=/tmp/meep-benchmark-build \
  PYTHONPATH=/tmp/meep-benchmark-prefix/lib/python3.11/site-packages \
  LD_LIBRARY_PATH=/tmp/meep-benchmark-prefix/lib:/path/to/meep-env/lib \
  /tmp/meep-benchmark-venv/bin/python run_benchmark.py \
  --manifest /tmp/crossing-gpu-smoke.json --device-id 0 \
  --output /tmp/crossing-gpu-smoke.result.json

/tmp/meep-benchmark-venv/bin/python run_benchmark.py \
  --manifest /tmp/crossing-gpu-smoke.json \
  --output /tmp/crossing-gpu-smoke.result.json --validate-only
```

For the private HIP build, keep the public compatibility backend name
`nvidia`, select the runtime explicitly, and mask by the ROCr selector rather
than by a `rocm-smi` card ordinal:

```sh
env MEEP_FINITE_CHECK=step \
  MEEP_SOURCE_TREE=/path/to/meep MEEP_BUILD_DIR=/tmp/meep-amd-build \
  PYTHONPATH=/tmp/meep-amd-prefix/lib/python3.11/site-packages \
  LD_LIBRARY_PATH=/tmp/meep-amd-prefix/lib:/path/to/rocm/lib:/path/to/meep-env/lib \
  /tmp/meep-benchmark-venv/bin/python run_benchmark.py \
  --manifest /tmp/crossing-gpu-smoke.json --accelerator hip \
  --visible-device 1 --toolkit-compiler /path/to/rocm/bin/hipcc \
  --rocm-smi /path/to/rocm/bin/rocm-smi \
  --output /tmp/crossing-hip-smoke.result.json
```

Use `MEEP_FINITE_CHECK=step` for smoke/sanitizer validation and
`MEEP_FINITE_CHECK=off` only for measured throughput. The diagnostic result
records the actual `gv.nx()/ny()/nz()`, timestep, raw timings, explicit sampling,
geometry transform, exact media, module/build provenance, finite DFT/mode
arrays, runtime counter endpoints/deltas, process-lifetime memory gauges, and
host peak memory. This sequential-window diagnostic shape is schema version 2;
version-1 results are rejected. CUDA remains the default accelerator. HIP results additionally
bind the ROCr selector to the runtime UUID and PCI BDF, join that BDF to the
`rocm-smi` inventory, and hash the selected `hipcc` and `rocm-smi` executables.
The runtime report must resolve the requested backend and precision without
fallback, and measured windows reject allocation, full-field-copy, recapture,
host-fallback, or material-warning deltas. It remains separate from the PR7
result document below because a ten-step smoke cannot honestly satisfy the
CPU-reference physics policy.

The runner reads and hashes the manifest bytes once before constructing the
simulation; execution, result generation, and the in-process validation all use
that same immutable snapshot. `--validate-only` rejects a result if the current
manifest bytes differ. Validation requires the complete result shape and binds
status, requested backend/precision/ranks, device cardinality, profile/fixed-step
semantics, repetition and warmup counts, grid and timestep, manifest-derived
geometry/material/sampling inputs, monitor port/band identity, observable policy,
and all claim flags. Missing, duplicate, unknown, or inconsistent records are
rejected rather than treated as optional metadata.

No PR3 diagnostic permits a speedup or publication claim. Those flags can only
be introduced through the authenticated CPU-native-baseline path below after it
verifies compatible physics observables within the manifest's tolerances.

## Profiler window

`--profile-steps N` performs one untimed step to compile and initialize the
resident execution plan, then brackets exactly `fields.advance(N)` with
`cudaProfilerStart/Stop` or `hipProfilerStart/Stop`, according to
`--accelerator`. MPB setup and mode decomposition stay outside the capture.

```sh
env CUDA_VISIBLE_DEVICES=0 MEEP_FINITE_CHECK=off \
  nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output=/tmp/meep-crossing \
  /tmp/meep-benchmark-venv/bin/python run_benchmark.py \
  --manifest /tmp/crossing-gpu-smoke.json --device-id 0 \
  --profile-steps 20 --output /tmp/meep-crossing-profile.json

env CUDA_VISIBLE_DEVICES=0 MEEP_FINITE_CHECK=off \
  ncu --force-overwrite --export /tmp/meep-crossing-kernels \
  --profile-from-start off --target-processes all --clock-control=base \
  --cache-control=none --kernel-name-base=demangled \
  --kernel-name 'regex:.*(curl_kernel|constitutive_kernel|dft_accumulate_kernel).*' \
  --launch-count 3 --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section LaunchStats --section Occupancy \
  /tmp/meep-benchmark-venv/bin/python run_benchmark.py \
  --manifest /tmp/crossing-gpu-smoke.json --device-id 0 \
  --profile-steps 1 --output /tmp/meep-crossing-ncu.json
```

Nsight captures are diagnostics, not unprofiled wall-time or speedup evidence.

## Result and provenance document

Create and validate a PR7-compatible result document with:

```sh
python3 benchmark_manifest.py result-template \
  --manifest ring-fixed.json \
  --output ring-fixed.result.json
python3 benchmark_manifest.py validate-result \
  --result ring-fixed.result.json
```

`benchmark_result.schema.json` types raw timing repetitions and summary
statistics, exact grid/timestep data, memory and transfer counters, kernel and
MPI timing, normalized grid-timesteps/s and component-updates/s, requested and
resolved execution, rank-to-GPU identity, CUDA/MPI
versions, Meep build/checkout state, invocation environment, and physics
observables with monitor definitions and tolerances. The command also performs
cross-field and finite-number validation that JSON Schema alone cannot express,
authenticates the exact referenced run manifest, and rejects successful results
without the configured timing samples and passing physics evidence.
Paper wall times remain stretch goals, never correctness or merge gates.

Each manifest also contains its immutable `validation_policy`. A successful
result must reference a hashed schema-version-1 `cpu_native_baseline` artifact
bound to the same paper-reference hash, case-definition hash, case ID, and
canonical physics-configuration hash. That final hash covers the full case
(including geometry, boundaries, sources, and monitors), excitation,
discretization, material/proof selection, and stopping policy, so a CPU reference
cannot be reused across a different resolution, bandwidth, material, boundary,
or run configuration. The artifact supplies the reference values; the manifest
supplies the exact required observable names, full monitor definitions, units,
and precision-specific tolerances. Results must match both one-to-one. Coupler,
crossing, and MMI require target-mode transmission and excess loss; the mode
converter and PSR require converted-mode efficiency plus residual-mode
crosstalk; the ring requires resonance wavelength, FWHM, and Q.

Templates deliberately contain obvious placeholders. A result marked
successful rejects an all-zero Meep commit, empty command/cwd, missing or empty
CUDA versions for NVIDIA, and a missing or unauthenticated physics-reference
artifact.

## Tests

The tests use only the Python standard library and do not require Meep, MPI, or
a GPU:

```sh
python3 -m unittest discover -s . -p 'test_*.py' -v
```
