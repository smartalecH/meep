"""3D SOI 1x2 MMI splitter on the GPU backend of smartalecH/meep (custom/phase2-pr7.5).

Port of Tidy3D's "FDTD accuracy benchmark" (flexcompute/tidy3d-notebooks,
MMIMeepBenchmark.ipynb), which compares Tidy3D against a CPU Meep run of the
same device at 30 px/um (reported: ~690 min on a 10-core laptop). Geometry,
source/monitor placement and run time follow that notebook; lengths in um.

  python mmi_gpu.py [--backend nvidia|cpu] [--res 30] [--until 900] [--precision native|mixed]

Writes mmi_<backend>_res<res>.txt (freq [Hz], transmittance of the top arm),
same layout as the notebook's misc/MeepMMI.txt reference.
"""
import argparse
import time

import meep as mp
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--backend", default="nvidia", choices=["nvidia", "cpu"])
p.add_argument("--res", type=float, default=30)
p.add_argument("--until", type=float, default=900, help="Tidy3D run_time 3 ps = 900 um/c")
p.add_argument("--precision", default="native", choices=["native", "mixed"])
p.add_argument("--no-symmetry", action="store_true")
p.add_argument("--avg", action="store_true",
               help="subpixel averaging (needs accelerator_strict=False: averaged cells use the backend fallback)")
# profiling switches
p.add_argument("--nfreq", type=int, default=50, help="DFT frequencies per monitor (reference: 50)")
p.add_argument("--point-source", action="store_true", help="plain Ey line source instead of EigenModeSource")
a = p.parse_args()

if a.backend == "nvidia":
    # The NVIDIA backend needs a backend communicator context, which this branch
    # only installs from C++ meep::initialize() (never run from Python) or when
    # the communicator changes. A one-group split installs it and is otherwise a no-op.
    mp.divide_parallel_processes(1)

# --- Tidy3D notebook parameters ------------------------------------------------
si = mp.Medium(epsilon=12.1104)  # n = 3.48
sio2 = mp.Medium(epsilon=2.1316)  # n = 1.46
wg_w, wg_t, wg_len = 0.5, 0.22, 2.0
mmi_w, mmi_len = 3.6, 11.5
out_sep, taper_w, taper_len = 0.6, 1.2, 20.05
sim_size = mp.Vector3(55.6, 6.925, 2.1)  # Tidy3D domain, PML outside it
dpml = 1.0

C0 = 299792458e6  # um/s; Meep frequency = f[Hz] / C0
f0_hz, fw_hz = 193.414e12, 6.24568e12
freqs_hz = np.linspace(f0_hz - fw_hz, f0_hz + fw_hz, a.nfreq)  # reference: 50 points
fcen = f0_hz / C0
df_mon = 2 * fw_hz / C0

x_src = -mmi_len / 2 - taper_len - wg_len / 2  # -26.8
x_out = -x_src
y_out = (out_sep + taper_w) / 2  # 0.9
port = mp.Vector3(0, 3 * wg_w, 4 * wg_t)


def prism(vertices):
    return mp.Prism([mp.Vector3(x, y) for x, y in vertices], height=wg_t, material=si)


def box(x0, x1, y0, y1):
    return prism([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


# Every object is a vertex-defined Prism. mp.Block trips an exact float compare in
# this branch's material IR capture ("object bounds are stale ... stored -42,-25.8;
# rebuilt -42,-25.8"), because center +/- size/2 rounds differently on rebuild.
geometry = [
    box(-mmi_len / 2, mmi_len / 2, -mmi_w / 2, mmi_w / 2),
    prism([(-mmi_len / 2 - taper_len, wg_w / 2), (-mmi_len / 2, taper_w / 2),
           (-mmi_len / 2, -taper_w / 2), (-mmi_len / 2 - taper_len, -wg_w / 2)]),
    prism([(mmi_len / 2, out_sep / 2), (mmi_len / 2, out_sep / 2 + taper_w),
           (mmi_len / 2 + taper_len, (out_sep + wg_w + taper_w) / 2),
           (mmi_len / 2 + taper_len, (out_sep - wg_w + taper_w) / 2)]),
    prism([(mmi_len / 2, -out_sep / 2), (mmi_len / 2, -out_sep / 2 - taper_w),
           (mmi_len / 2 + taper_len, -(out_sep + wg_w + taper_w) / 2),
           (mmi_len / 2 + taper_len, -(out_sep - wg_w + taper_w) / 2)]),
    # straight waveguides (Tidy3D: x in +-[25.8, 40]) extended through the PML
    box(-40 - 2 * dpml, -25.8, -wg_w / 2, wg_w / 2),
    box(25.8, 40 + 2 * dpml, 0.9 - wg_w / 2, 0.9 + wg_w / 2),
    box(25.8, 40 + 2 * dpml, -0.9 - wg_w / 2, -0.9 + wg_w / 2),
]


sources = [
    mp.Source(mp.GaussianSource(fcen, fwidth=3 * df_mon), mp.Ey, center=mp.Vector3(x_src),
              size=mp.Vector3(0, wg_w, wg_t))
] if a.point_source else [
    mp.EigenModeSource(
        mp.GaussianSource(fcen, fwidth=3 * df_mon),  # broader than the monitor band
        center=mp.Vector3(x_src), size=port,
        direction=mp.X, eig_band=1, eig_parity=mp.NO_PARITY, eig_match_freq=True,  # band 1 = TE0
    )
]

sim = mp.Simulation(
    cell_size=sim_size + mp.Vector3(2 * dpml, 2 * dpml, 2 * dpml),
    resolution=a.res,
    boundary_layers=[mp.PML(dpml)],
    default_material=sio2,
    geometry=geometry,
    sources=sources,
    # Tidy3D symmetry=[0,-1,0]: Ey-dominant TE0 is odd under y -> -y
    symmetries=[] if a.no_symmetry else [mp.Mirror(mp.Y, phase=-1)],
    # Strict mode refuses subpixel-averaged (anisotropic) materials, so averaging
    # is opt-in and relaxes strictness; default is staircased geometry.
    eps_averaging=a.avg,
    accelerator_strict=not a.avg,
    backend=a.backend,
    precision=a.precision,
)

meep_freqs = freqs_hz / C0
out_mon = None if a.nfreq == 0 else sim.add_flux(meep_freqs, mp.FluxRegion(center=mp.Vector3(x_out, y_out), size=port))
ref_mon = None if a.nfreq == 0 else sim.add_flux(meep_freqs, mp.FluxRegion(center=mp.Vector3(x_src + 0.5), size=port))

t0 = time.time()
sim.init_sim()
t_init = time.time() - t0
t0 = time.time()
sim.run(until=a.until)
t_run = time.time() - t0

T = (np.array(mp.get_fluxes(out_mon)) / np.array(mp.get_fluxes(ref_mon))
     if a.nfreq else np.full(1, np.nan))
nx, ny, nz = (int(round(s * a.res)) for s in (sim.cell_size.x, sim.cell_size.y, sim.cell_size.z))
steps = int(round(a.until / sim.fields.dt))
if mp.am_master():
    out = f"mmi_{a.backend}_res{a.res:g}{'_avg' if a.avg else ''}.txt"
    if a.nfreq:
        np.savetxt(out, np.column_stack([freqs_hz, T]), delimiter="\t")
    i0 = np.argmin(abs(freqs_hz - f0_hz)) if a.nfreq else 0
    print(f"RESULT backend={a.backend} precision={a.precision} res={a.res:g} "
          f"grid={nx}x{ny}x{nz} ({nx * ny * nz / 1e6:.1f}M cells, symmetry halves the stored grid) "
          f"steps={steps} init={t_init:.1f}s run={t_run:.1f}s "
          f"rate={nx * ny * nz * steps / t_run / 1e9:.2f} Gcell-steps/s  T(1.55um)={T[i0]:.4f}  -> {out}")
