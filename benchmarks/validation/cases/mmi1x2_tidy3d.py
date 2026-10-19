"""3D SOI 1x2 MMI from Tidy3D's FDTD accuracy benchmark (MMIMeepBenchmark).

Same device, domain, source/monitor planes and 3 ps run as
../../nvidia/tidy3d_mmi/mmi_gpu.py. Reference spectra (the notebook's CPU
Meep run at 30 px/um and Tidy3D's flux data) are AGPL notebook assets, so
they are read at run time from $MEEPBENCH_REFDATA, ~/Downloads/tidy3d_mmi_ref
or /workspace (MeepMMI.txt, mmi_simulation_data.h5) and never committed.
"""
import os

import meep as mp
import numpy as np

from meepbench import Case, Geometry, Layer, ModeSource, Observable, Port, Reference, Run, \
    Wavelengths
from meepbench import materials

materials.register("SiO2_1.46", 1.46)   # notebook's cladding (eps 2.1316)

C0 = 299792458.0
F0, FW = 193.414e12, 6.24568e12
FREQS = np.linspace(F0 - FW, F0 + FW, 50)           # notebook monitor frequencies (Hz)

WG_W, MMI_W, MMI_L, SEP, TAPER_W, TAPER_L = 0.5, 3.6, 11.5, 0.6, 1.2, 20.05


def layout():
    def box(x0, x1, y0, y1):
        return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    h = MMI_L / 2
    polys = [
        box(-h, h, -MMI_W / 2, MMI_W / 2),
        np.array([(-h - TAPER_L, WG_W / 2), (-h, TAPER_W / 2), (-h, -TAPER_W / 2),
                  (-h - TAPER_L, -WG_W / 2)]),
        np.array([(h, SEP / 2), (h, SEP / 2 + TAPER_W), (h + TAPER_L, (SEP + WG_W + TAPER_W) / 2),
                  (h + TAPER_L, (SEP - WG_W + TAPER_W) / 2)]),
        np.array([(h, -SEP / 2), (h, -SEP / 2 - TAPER_W),
                  (h + TAPER_L, -(SEP + WG_W + TAPER_W) / 2),
                  (h + TAPER_L, -(SEP - WG_W + TAPER_W) / 2)]),
        box(-42, -25.8, -WG_W / 2, WG_W / 2),            # straights through the PML
        box(25.8, 42, 0.9 - WG_W / 2, 0.9 + WG_W / 2),
        box(25.8, 42, -0.9 - WG_W / 2, -0.9 + WG_W / 2),
    ]
    ports = {"in": Port((-25.8, 0.0), 180.0, WG_W), "top": Port((25.8, 0.9), 0.0, WG_W),
             "bottom": Port((25.8, -0.9), 0.0, WG_W)}
    return {(1, 0): polys}, ports


def _refdir():
    for d in (os.environ.get("MEEPBENCH_REFDATA"), os.path.expanduser("~/Downloads/tidy3d_mmi_ref"),
              "/workspace"):
        if d and os.path.isdir(d):
            yield d


def meep_cpu_spectrum():
    for d in _refdir():
        p = os.path.join(d, "MeepMMI.txt")
        if os.path.exists(p):
            f, t = np.loadtxt(p).T
            return C0 / f * 1e6, t
    return None


def tidy3d_spectrum():
    for d in _refdir():
        p = os.path.join(d, "mmi_simulation_data.h5")
        if os.path.exists(p):
            import h5py
            with h5py.File(p, "r") as h:
                f = h["data/0/flux/f"][:]
                t = h["data/0/flux/__xarray_dataarray_variable__"][:].astype(float)
            return C0 / f * 1e6, t
    return None


CASE = Case(
    name="mmi1x2_tidy3d",
    description="Tidy3D FDTD accuracy benchmark: SOI 1x2 MMI, top-arm flux ratio, 3 ps",
    geometry=Geometry(polygons=layout),
    stack={(1, 0): Layer(0.0, 0.22, "Si")}, background="SiO2_1.46",
    wavelengths=Wavelengths(start=0, stop=0, step=1, center=1.55,
                            points=(C0 / FREQS * 1e6).tolist()),
    domain_xy=(55.6, 6.925), z_span=(-1.05, 1.05),        # Tidy3D domain; PML outside
    symmetries=[mp.Mirror(mp.Y, phase=-1)],                # TE0 (Ey) odd in y
    source=ModeSource("in", "TE0", distance=1.0, monitor_gap=0.5, fwidth_factor=3.0),
    observables=[Observable("T_top", "flux_ratio", "top", distance=1.0)],
    mode_plane=(1.0, 0.88),
    run=Run(until=900.0, after_sources=False),             # run_time 3 ps
    eps_averaging=True,                                    # as the notebook's CPU Meep run
    reference=Reference(
        source="flexcompute/tidy3d-notebooks MMIMeepBenchmark (AGPL; data read at run time)",
        values={"T_top": (0.4913, 0.005)},                 # CPU Meep, res 30, at 1.55 um
        spectra={"T_top": {"meep_cpu_res30": meep_cpu_spectrum, "tidy3d": tidy3d_spectrum}},
        runtimes_s={"tidy3d (text, own mesh)": {"res30": 120.0},
                    "meep_cpu 10-core (text)": {"res30": 690 * 60.0}},
        notes="Tidy3D used its own nonuniform 10 cells/lambda mesh; Meep reference res 30."),
    tags=("tier1",),
)
