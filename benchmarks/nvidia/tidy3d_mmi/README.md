# Tidy3D MMI accuracy benchmark on the NVIDIA backend

3D SOI 1x2 MMI splitter from Flexcompute's
[FDTD accuracy benchmark](https://github.com/flexcompute/tidy3d-notebooks/blob/develop/MMIMeepBenchmark.ipynb),
which compares Tidy3D against a CPU Meep run of the same device at 30 px/um
(reference spectrum: `misc/MeepMMI.txt` in that repository; reported CPU Meep
runtime ~690 min on a 10-core laptop).

```sh
python mmi_gpu.py --res 30 --until 900          # staircased geometry (strict mode)
python mmi_gpu.py --res 30 --until 900 --avg    # subpixel averaging (accelerator_strict=False)
```

Writes `mmi_nvidia_res<res>[_avg].txt`: frequency [Hz], top-arm transmittance.

## Results (resolution 30, 57.0M cells, y-mirror symmetry, 54,000 steps)

| Run | GPU | Wall time | max abs(dT) vs CPU Meep | rms dT |
|---|---|---|---|---|
| staircased (`results/mmi_nvidia_res30.txt`) | RTX A6000 48 GB | 29 min | 0.0201 | 0.0084 |
| subpixel averaging (`results/mmi_nvidia_res30_avg.txt`) | RTX 6000 Ada 48 GB | 34 min | **0.0020** | **0.0010** |

With averaging the GPU spectrum matches the published CPU Meep spectrum to
0.4% over 1.50-1.60 um; Tidy3D's curve in the notebook lies within ~0.003 of
the same reference. The staircased error comes from the 0.22 um silicon
layer spanning 6.6 grid cells.

Steady-state step rate (staircased, A6000): 24.1 ms/step, i.e. 2.37
Gcell-steps/s over the full domain (1.18 over the stored half-domain).

## Notes

- `mp.divide_parallel_processes(1)` is called before building the simulation:
  from Python, nothing else installs the backend communicator context.
- Geometry uses vertex-defined `Prism`s; `Block` currently trips an exact
  floating-point bounds comparison in material IR capture.
- Averaged runs need roughly 35 GB of host RAM at this size during setup.
