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

## GPU sweep (staircased, resolution 30, branch head before the finite-check rewrite)

Same case on eight vast.ai GPUs (one `vk jobs` batch, built from this branch).
"Triad" is measured fp64 STREAM-triad bandwidth on that box; "% roofline"
assumes ~490 DRAM bytes per stored cell per step (Nsight Compute on sm_86).
Every spectrum matches the RTX A6000 result to <= 2e-15.

| GPU | $/h | triad GB/s | ms/step | Gcell/s (full domain) | % roofline | setup s | wall min | $/sim |
|---|---|---|---|---|---|---|---|---|
| RTX A6000 | 0.40 | 702 | 20.67 | 2.76 | 96% | 546 | 28.0 | 0.19 |
| RTX 6000 Ada | 0.54 | 815 | 17.87 | 3.19 | 96% | 315 | 21.3 | 0.19 |
| RTX 4090 | 0.35 | 908 | 15.67 | 3.63 | 98% | 465 | 21.8 | 0.13 |
| A100 SXM4 40 GB | 0.43 | 1385 | 14.32 | 3.98 | 70% | 530 | 22.1 | 0.16 |
| RTX PRO 6000 Blackwell | 1.27 | 1466 | 10.59 | 5.38 | 90% | 365 | 15.4 | 0.33 |
| RTX 5090 | 0.51 | 1554 | 10.36 | 5.50 | 87% | 481 | 16.9 | 0.14 |
| H100 SXM | 2.41 | 2939 | 7.48 | 7.61 | 63% | 373 | 13.0 | 0.52 |
| B200 | 6.25 | 6615 | 7.12 | 8.00 | 30% | 412 | 13.3 | 1.39 |

H100 follow-up (Nsight Systems: GPU busy 98% of each step; no host gaps):

| configuration | ms/step | Gcell/s |
|---|---|---|
| sweep build (finite check every step) | 7.48 | 7.61 |
| finite-check rewrite | 6.80 | 8.37 |
| `MEEP_FINITE_CHECK=off` | 4.72 | 12.07 |
| `precision="mixed"` + `MEEP_FINITE_CHECK=off` | 3.20 | 17.82 |

`precision="mixed"` changes the transmittance by at most 1.3e-7 relative to
native precision (the staircase discretization error is 2e-2).

## Hopper/Blackwell after the loads-first kernels (branch head 95c1df4)

Steady-state ms/step at resolution 30 (Gcell-steps/s over the full 57.0M-cell
domain in parentheses; the stored half-domain rate is half that). "check" is
`MEEP_FINITE_CHECK`.

| GPU | triad | native, check=step | native, check=off | mixed, check=off | mixed, check=step |
|---|---|---|---|---|---|
| H100 SXM | 2.93 TB/s | 5.26 (10.8) | 4.42 (12.9) | 2.86 (19.9) | 3.44 (16.6) |
| B200 | 6.64 TB/s | 3.16 (18.0) | 2.58 (22.0) | 2.23 (25.6) | 2.77 (20.6) |

Before this series (eight-GPU sweep build): H100 7.48, B200 7.12 ms/step
(native, check=step). The B200 gain is mostly the constitutive kernel: Meep's
kernel ran at ~1 TB/s on B200 because control flow between its loads
serialized DRAM round trips (reproduced in isolation: 259 us vs 41 us for the
same launch with all loads first).

Bandwidth roofline at the current ~405 DRAM bytes per stored cell per step
(native, check off): H100 3.93 ms (measured 4.42, ~89%), B200 1.74 ms
(measured 2.58, ~68%). Remaining B200 losses: ~237 per-chunk launches per step
(~27 chunks, split at PML boundaries), latency-bound halo/zero kernels, and
mixed precision not yet saturating bandwidth. A minimal Yee update moves
~96 B (fp64) / ~48 B (fp32) per cell, so higher throughput needs fewer bytes
per cell (fused curl+constitutive, fp32 storage, compact materials, one chunk
per GPU), not only kernel tuning.
