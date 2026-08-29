# arXiv 2506.16665 benchmark runner

This directory contains an authenticated manifest and smoke/throughput runner for the six-device
benchmark corpus described in [arXiv:2506.16665](https://arxiv.org/html/2506.16665).
`benchmark_manifest.py` remains standard-library-only. `run_benchmark.py`
imports a branch-matched Meep build and `gdstk`, constructs the selected GDS
cell, runs a single rank on CPU or NVIDIA, and writes an authenticated
diagnostic result with raw finite DFT flux and mode-power arrays.

The executable scope is deliberately narrower than the paper: it uses fixed,
nondispersive performance-adaptation media and does not claim paper-equivalent
physics, two-GPU MPI scaling, or a speedup from a ten-step/profiled smoke run.
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

`benchmark_manifest.schema.json` defines the complete version-3 run document.
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

`run_benchmark.py` currently accepts `smoke` and `fixed-step`. A fixed-step
repetition constructs a fresh simulation, performs the manifest's 100 untimed
warmup steps on that simulation, times exactly the requested step batch, and
queries monitors after the timing window. Smoke mode runs ten steps once with
no warmup. Ten steps occur well before the 20 nm Gaussian pulse peak and prove
only construction, stepping, and finite monitor access.

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

Use `MEEP_FINITE_CHECK=step` for smoke/sanitizer validation and
`MEEP_FINITE_CHECK=off` only for measured throughput. The diagnostic result
records the actual `gv.nx()/ny()/nz()`, timestep, raw timings, explicit sampling,
geometry transform, exact media, module/build provenance, and finite DFT/mode
arrays. It is separate from the PR7 result document below because a ten-step
smoke cannot honestly satisfy the CPU-reference physics policy or the runtime
counters that PR3 does not expose to Python.

## Profiler window

`--profile-steps N` performs one untimed step to compile and initialize the
resident execution plan, then brackets exactly `fields.advance(N)` with
`cudaProfilerStart/Stop`. MPB setup and mode decomposition stay outside the
capture.

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
