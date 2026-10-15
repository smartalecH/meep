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

## Experiments that did not help (H100 NVL, resolution 30, untraced)

Both were bitwise neutral and were not merged.

| change | native ms/step | mixed ms/step |
|---|---|---|
| baseline (this branch) | 3.94 | 2.80 |
| batch same-variant regions into one launch (233 -> 49 launches/step) | 3.92 | 3.07 |
| 2 or 4 elements per thread in curl/constitutive (res 20) | +3% / +15% | +3% / +14% |

Under `nsys --cuda-graph-trace=node` the batched build looked ~20% faster:
node tracing adds per-kernel overhead, so it overstates launch costs. Judge
launch-count changes on untraced timings (`bench/steady.py`). With CUDA graphs
the real per-launch overhead is small; the precision-independent part of the
step is kernel efficiency (an fp32 curl kernel reaches ~75% of STREAM triad on
H100 in isolation, fp64 ~100%), and the remaining lever is bytes per cell.

## Multi-GPU strong scaling (8x RTX PRO 6000 Blackwell, PCIe Gen5, no NVLink)

`mpirun -np N`, one rank per GPU (automatic node-local device selection),
staged transport (`MEEP_GPU_AWARE_MPI=no`), `MEEP_FINITE_CHECK=off`. Fields
after 300 steps are bitwise identical to the single-GPU run on 2, 4 and 8
ranks, for both transports.

| case | GPUs | native ms/step | speedup | efficiency | mixed ms/step | first step (setup) |
|---|---|---|---|---|---|---|
| res 30 (57.0M cells) | 1 | 8.54 | 1.00 | 100% | 4.68 | 221 s |
| | 2 | 4.82 | 1.77 | 89% | | 104 s |
| | 4 | 3.27 | 2.61 | 65% | | 51 s |
| | 8 | 2.45 | 3.49 | 44% | 2.17 | 30 s |
| res 40 (135.0M cells) | 1 | 19.27 | 1.00 | 100% | 10.91 | 562 s |
| | 2 | 10.38 | 1.86 | 93% | | 266 s |
| | 4 | 6.05 | 3.19 | 80% | | 127 s |
| | 8 | 3.72 | 5.18 | 65% | 2.82 | 66 s |

8 GPUs, res 40: 36.3 (native) / 47.8 (mixed) Gcell-steps/s over the full domain.

- Per-rank kernel work at 8 GPUs is balanced and within 4% of 1/8 of the
  single-GPU step; the efficiency loss is ~1.2 ms/step of non-overlapped halo
  exchange (four exchanges per step, each D2H copy -> MPI -> H2D copy, ~4.6 MB
  per rank per step: latency, not bandwidth).
- The first-step backend setup also splits across ranks (221 s -> 30 s).
- Direct (CUDA-aware) transport needs `UCX_MEMTYPE_CACHE=n` (otherwise UCX
  segfaults sending device buffers) and on this node is far slower than
  staged (8 GPUs, res 30: 254 vs 2.45 ms/step) because most GPU pairs lack
  peer access. Communication/compute overlap is only implemented for the
  direct route, so it could not be used here; on NVLink/P2P nodes it is the
  expected path to near-linear scaling.

## Multi-GPU on an NVLink node (4x B200, NV18 between all pairs)

Resolution 40, native unless noted, `MEEP_FINITE_CHECK=off`. Fields on 4 ranks
are bitwise identical to 1 rank (staged and direct transports).

| GPUs | staged ms/step | direct ms/step (CUDA-aware MPI over NVLink) |
|---|---|---|
| 1 | 5.30 (25.4 Gcell/s) | |
| 2 | 4.45 | 12.0 |
| 4 | 3.97 (33.9 Gcell/s); mixed 3.62 (37.3) | 45.1 |

- A B200 computes this case so fast (1.3 ms/step per rank at 4 GPUs) that the
  staged path's per-step halo latency (~2.6 ms at 4 ranks) dominates; at
  resolution 20, 2 staged ranks are slower than 1 GPU (2.16 vs 1.21 ms).
- The direct route is 8-25x slower than staged in every UCX configuration
  tried, and crashes unless `UCX_MEMTYPE_CACHE=n` (UCX misidentifies device
  buffers). It does not benefit from NVLink as implemented.
- `MEEP_NVIDIA_MPI_OVERLAP=required` is rejected for this step program
  ("dependency region successor is not curl or constitutive": the E halo is
  followed by DFT/time-increment work); `auto` crashed with the direct route.
- Implication: on fast GPUs, multi-GPU speedup needs a low-latency in-node
  transport (e.g. CUDA IPC peer copies instead of MPI per halo) and
  communication/compute overlap that works for real step programs. Until
  then, larger problems per GPU and PCIe nodes with slower GPUs scale better
  (8x RTX PRO 6000: 5.2x at resolution 40).

## CUDA-aware MPI transport: UCX vs Open MPI smcuda (2x H100 SXM, NV18)

The conda-forge Open MPI 5 is CUDA-aware, but its default UCX PML is the slow
part on these container hosts. OSU micro-benchmarks, device-to-device:

| path | 8 B latency | 1 MB latency | 4 MB bandwidth |
|---|---|---|---|
| UCX (`UCX_TLS=self,sm,cuda_copy,cuda_ipc UCX_MEMTYPE_CACHE=n`; every other UCX config fails) | 208 us | 114 us | 41 GB/s |
| `--mca pml ob1 --mca btl self,sm,smcuda --mca opal_cuda_support 1` | 12 us | 13 us | 260 GB/s |

MMI resolution 40, native, `MEEP_FINITE_CHECK=off`. Fields on 2 ranks are
bitwise identical to 1 rank with smcuda (staged and direct):

| GPUs | MPI | transport | ms/step | speedup |
|---|---|---|---|---|
| 1 | | | 11.59 | 1.00 |
| 2 | UCX | staged | 7.15 | 1.62 |
| 2 | smcuda | staged | 7.21 | 1.61 |
| 2 | UCX | direct | 14.39 | 0.81 |
| 2 | smcuda | direct | 7.76 | 1.49 |

- Use `--mca pml ob1 --mca btl self,sm,smcuda` for direct transport on a
  single node: it removes the 2x direct-route penalty with no code change.
- Even so, direct is no faster than staged, and 2 ranks lose ~1.4-2 ms/step
  versus ideal (5.8 ms). The halo payload moves in microseconds at these
  rates, so the loss is the backend's per-exchange host synchronization
  (each stage does two `MPI_Allreduce` reconciliations, a host spin on the
  gather event, `MPI_Waitall` and a `cudaStreamSynchronize`), not MPI.
- First-step setup on this host: 865 s on 1 rank, 427 s on 2 ranks.

## Node-local peer transport and graph branches (branch custom/phase2-pr9-peer)

On one node the halo exchange now bypasses MPI: each rank writes its halo
straight into the neighbour's receive buffer through CUDA IPC and
publishes a per-message generation flag; the neighbour waits on the flag
on the device. No host synchronization or MPI call is on the per-step
path. Per-chunk curl/constitutive kernels are also captured as parallel
CUDA graph branches (a ~0.7 ms/step serial-kernel floor on H100 was the
real strong-scaling limit). Fields after 300 steps are bitwise identical
to 1 rank on 2 and 4 ranks (native and mixed); checkpoint round trips are
bitwise on staged, direct and peer routes.

Resolution 40 (135M cells), native unless noted, `MEEP_FINITE_CHECK=off`:

| node | GPUs | route | ms/step | Gcell/s | speedup |
|---|---|---|---|---|---|
| 2x H100 SXM (NV6) | 1 | | 9.06 | 14.9 | 1.00 |
| | 2 | staged MPI (before) | 7.81 | 17.3 | 1.16 |
| | 2 | peer | 4.81 | 28.1 | 1.88 |
| 4x H200 NVL (NV6 all pairs) | 1 | | 7.29 | 18.5 | 1.00 |
| | 2 | peer | 3.86 | 35.0 | 1.89 |
| | 4 | staged MPI | 3.88 | 34.8 | 1.88 |
| | 4 | peer | 2.13 | 63.4 | 3.43 (86%) |
| | 4 | peer, mixed | 1.61 | 83.6 | 4.52 |

Knobs: `MEEP_NVIDIA_PEER_TRANSPORT=off|auto|required` (default auto: used
when every rank is on one node and can map its neighbours' GPUs, else the
MPI route), `MEEP_NVIDIA_PEER_TIMEOUT` (seconds, default 1800),
`MEEP_NVIDIA_PEER_VERBOSE=1`, `MEEP_NVIDIA_GRAPH_BRANCHES` (default 8).

Remaining 4-GPU loss (~0.3 ms/step): middle ranks wait ~0.1 ms for
neighbours (they carry twice the halo work), eager peer kernels ~0.08 ms,
and each remote halo still splits the step graph. When the MPI route is used on one node,
prefer `--mca pml ob1 --mca btl self,sm,smcuda` over UCX in vast
containers (UCX device-buffer latency was 10x worse).

## End to end on 2x H100 SXM (branch custom/phase2-pr10-setup)

Full `mmi_gpu.py --res 30 --until 900` (54,000 steps) with default settings
(`mpirun -np 2`, peer transport auto, default finite check). Wall time covers
the whole run: Python start, `init_sim` (incl. MPB mode solve), first-step
setup, stepping and flux readout.

| geometry | wall time | init_sim | run (setup + steps) | max/rms dT vs CPU Meep | max/rms dT vs Tidy3D |
|---|---|---|---|---|---|
| staircased | **183 s (3.1 min)** | 15 s | 165 s | 0.020 / 0.0088 | 0.028 / 0.012 |
| subpixel averaging (`--avg`) | **568 s (9.5 min)** | 359 s | 205 s | **0.0020 / 0.0010** | 0.0092 / 0.0049 |

Tidy3D's own spectrum (`misc/mmi_simulation_data.h5` in tidy3d-notebooks, a
10 steps/wavelength run) differs from the CPU Meep reference by 0.0079 max /
0.0044 rms, so the averaged GPU result agrees with Tidy3D as closely as CPU
Meep does, and with CPU Meep to 0.2% (unchanged from the single-GPU result
above). For scale: the notebook reports ~690 min for CPU Meep on a 10-core
laptop; the single-A6000 staircased run above took 29 min. With averaging,
`init_sim` (CPU subpixel averaging) is now the dominant cost.
Plot: `results/compare_2xh100_res30.png`.

### After faster averaged setup (branch custom/phase2-pr11-gpu-avg)

Same runs, same 2x H100 box type, with parallel material-IR capture, mu-aware
averaging, OpenMP CPU averaging (`--with-openmp`, `OMP_NUM_THREADS` = cores per
rank) and batched device material initialization:

| geometry | wall time | init_sim | run (setup + steps) | max/rms dT vs CPU Meep |
|---|---|---|---|---|
| staircased | **166 s (2.8 min)** | 2.3 s | 161 s | 0.020 / 0.0088 |
| subpixel averaging | **176 s (2.9 min)** | 5.8 s (was 359 s) | 167 s | **0.0020 / 0.0010** |

The accurate (averaged) configuration now costs only ~10 s more than the
staircased one; stepping (~150 s for 54,000 steps) dominates both.

### Mixed precision and the finite check (2x H100, averaged, res 30)

| precision | finite check | wall time | max/rms dT vs CPU Meep |
|---|---|---|---|
| native | default (every step) | 176 s | 0.0020 / 0.0010 |
| native | off | 152 s | 0.0020 / 0.0010 |
| mixed (fp32 fields, fp64 DFT) | default | 128 s | 0.0020 / 0.0010 |
| mixed | off | **110 s (1.8 min)** | 0.0020 / 0.0010 |

Mixed precision moves the spectrum by at most 1e-7. The per-step non-finite
scan costs 18-24 s here (an extra full read of every field each step).
