# meepbench: validation and speed harness

One driver for every validation problem. Each case is a short declarative file in
`cases/`. A case names a layout source (gdsfactory component, GDS file or polygons), a layer
stack, ports, a mode source, observables and references. The runner builds the
`mp.Simulation` for any resolution, backend, precision, material model and rank count, then
writes a JSON result.

```sh
cd benchmarks/validation
python -m meepbench list
python -m meepbench plot ring                         # eps slice + source/monitor planes
python -m meepbench run crossing --cpl 10             # NVIDIA backend (default)
python -m meepbench run crossing --res 10 --backend cpu
mpirun -np 2 --mca pml ob1 --mca btl self,sm,smcuda python -m meepbench run psr --cpl 15
python -m meepbench run mydevice.gds --cell top --source o1:TE0 --observe o2:TE0 \
    --layer 1/0:0:0.22:Si --band 1.5 1.6 0.01      # ad-hoc GDS case
python -m meepbench report results/*.json
```

Requires gdsfactory >= 9 for gdsfactory/GDS cases (`pip install gdsfactory`).

## Cases
| case | tier | layout | observable | reference |
|---|---|---|---|---|
| `mmi1x2_tidy3d` | 1 | polygons (Tidy3D notebook) | top-arm flux ratio, 50 freqs | CPU Meep res 30 + Tidy3D spectra; Tidy3D ~2 min |
| `crossing` | 2 | gf `crossing` | TE0 through | 95.7%; paper runtimes |
| `mmi2x2` | 2 | gf `mmi2x2_with_sbend` | TE0 cross | ~48%; paper runtimes |
| `coupler` | 2 | gf `coupler` (gap 0.236, L 20) | TE0 cross | ~44% (41-47%); paper runtimes |
| `ring` | 2 | gf `ring_single` (R 10, gap 0.2) | through dip: lambda0, FWHM, Q | 1545 nm, 0.89 nm, ~1780 |
| `psr` | 2 | gf `polarization_splitter_rotator` + SiN top | TM0 -> TE0 | ~95%; paper runtimes |
| `kernel` | 3 | none (uniform SiO2, no PML) | Gcell-steps/s | OPN 2024: 20 (A100) / 33 (H100) Gcell/s |

Tier-2 runtimes, 25-cell workloads, and domains come from
`../nvidia/paper_2506_16665/paper_reference.json` (arXiv 2506.16665). The geometry is regenerated
from the gdsfactory generic PDK, not taken from the paper repository's unlicensed GDS.
The `mmi1x2_tidy3d` reference spectra are AGPL notebook assets. They are read at run time from
`$MEEPBENCH_REFDATA`, `~/Downloads/tidy3d_mmi_ref` or `/workspace` and never committed.

## Conventions
- `--cpl N`: cells per wavelength in the highest-index material at the shortest wavelength,
  `res = ceil(N * n_max / lambda_min)` (the paper/Tidy3D convention). `--res` sets px/um directly.
- Cell = layout bbox + `margin_xy` (or the case's `domain_xy`), `z_span` in z, PML outside.
  Layout ports from gdsfactory are extended 10 um, so the waveguides run through the PML.
- Sources and monitors sit on the straight port extensions. The source is 0.6 um outside the
  port. The incident monitor is 0.3 um downstream of the source. Output monitors are 0.3 um
  outside their port. Mode transmission is |b_out|^2 / |a_in|^2 (band 1 = TE0, 2 = TM0, or
  `band:N`). `flux_ratio` divides net flux by the flux at the incident plane.
- Materials: `const` uses Si 3.48, SiO2 sqrt(2.09) and SiN 2.0, the paper's Meep adaptation.
  `dispersive` fits one Lorentzian pole per material to `meep.materials` over the case band.
- Reference values and tolerances apply at converged resolution (paper: 15-20 cells/lambda;
  MMI: res 30). Low-resolution smoke runs report FAIL there by design. For smoke runs,
  compare GPU against CPU at the same resolution.
- Stop rule: `stop_when_dft_decayed(1e-5)` unless the case fixes `until`.

## Result JSON
`resolution`, `grid`, `cells`, `steps`, `ranks`, `time_s.{init_sim,run,total}`,
`gcell_steps_per_s`, `values`, `checks` (pass/fail vs reference values), `spectrum_checks`
(max/rms vs reference spectra), `spectra`, `wavelengths`, `reference.runtimes_s` (published
times for the same cells/lambda), plus host info.

## Writing a case
```python
from meepbench import Case, Geometry, Layer, ModeSource, Observable, Wavelengths
CASE = Case(
    name="my_mmi", description="...",
    geometry=Geometry(gdsfactory="mmi1x2", params=dict(length_mmi=5.5)),  # or gds=, cell=
    stack={(1, 0): Layer(0, 0.22, "Si")}, background="SiO2",
    wavelengths=Wavelengths(1.5, 1.6, 0.01, center=1.55),
    source=ModeSource("o1", "TE0"),
    observables=[Observable("top", "mode_transmission", "o2", "TE0")],
)
```
`custom=fn(case, ctx)` covers anything the declarative fields cannot express. With
`ctx["stage"] == "simulation"`, the returned dict adds or overrides `mp.Simulation` kwargs.
With `"observables"`, the returned dict adds values computed from `ctx["built"].sim`.
