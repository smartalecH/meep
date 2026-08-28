# arXiv 2506.16665 benchmark manifest prototype

This directory contains a standard-library-only prototype for the six-device
benchmark corpus described in [arXiv:2506.16665](https://arxiv.org/html/2506.16665).
It does not import Meep, initialize CUDA, or run a simulation. It validates the
external geometry inputs and emits a versioned JSON run specification plus a
typed result/provenance template for a later runner.

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
to a future Meep runner. It fixes the GDS transform, layer/datatype and z-range
mappings, ports, source and monitor definitions, padding, all six PML faces,
and the field-energy decay region. The ring case separately identifies its
mode monitor and resonance observable. Every transmission/conversion monitor
is normalized by the forward source mode at the case's explicit
`input_incident` monitor. The emitted manifest also records the exact
reciprocal-endpoint conversion from the centered wavelength interval to Meep's
Gaussian center frequency and `fwidth`. These values are versioned benchmark
inputs rather than hidden runner defaults.

`benchmark_manifest.schema.json` defines the complete version-2 run document.
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
  --precision f32 \
  --ranks 2 \
  --mpi-transport staged \
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
