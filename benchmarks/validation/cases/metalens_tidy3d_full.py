"""Tidy3D gallery metalens, full domain (the notebook's first run: lens plus
1.7 focal lengths of air, 541x541x589 = 1.74e8 grid points, 31,633 steps,
23.2 s solver / 13.4 s time stepping). Same lens and source as
metalens_tidy3d; the focus is measured directly from DFT fields on the axis
and on a lateral line through the focus instead of a near-to-far projection."""
import dataclasses

import meep as mp
import numpy as np

from meepbench import Reference
from meepbench.cases import load

base = load("metalens_tidy3d")
lens = __import__("sys").modules[base.custom.__module__]
F, Z_TOP, WL = lens.F, lens.Z_TOP, lens.WL
Z_MAX = Z_TOP + 1.7 * F                         # notebook: space -> substrate -> fins -> 1.7 f


def hooks(case, ctx):
    built = ctx["built"]
    if ctx["stage"] == "simulation":
        return lens.hooks(case, ctx)
    if ctx["stage"] == "monitors":
        sim = ctx["sim"]
        built.monitors["axis"] = sim.add_dft_fields(
            [mp.Ex, mp.Ey, mp.Ez], [1 / WL], center=built._local(0, 0, Z_TOP + F),
            size=mp.Vector3(0, 0, 8.0))
        built.monitors["xline"] = sim.add_dft_fields(
            [mp.Ex, mp.Ey, mp.Ez], [1 / WL], center=built._local(0, 0, Z_TOP + F),
            size=mp.Vector3(4.0, 0, 1.0))
    if ctx["stage"] == "observables":
        sim = built.sim

        def intensity(key):
            return sum(np.abs(np.squeeze(sim.get_dft_array(built.monitors[key], c, 0))) ** 2
                       for c in (mp.Ex, mp.Ey, mp.Ez))
        axial = intensity("axis")
        zs = Z_TOP + F - 4.0 + np.arange(axial.size) * 8.0 / max(axial.size - 1, 1)
        k = int(np.argmax(axial))
        plane = intensity("xline")              # (x, z) slab around the nominal focus
        zp = Z_TOP + F - 0.5 + np.arange(plane.shape[1]) * 1.0 / max(plane.shape[1] - 1, 1)
        lateral = plane[:, int(np.argmin(np.abs(zp - zs[k])))]
        xs = -2.0 + np.arange(lateral.size) * 4.0 / max(lateral.size - 1, 1)
        above = np.where(lateral >= lateral.max() / 2)[0]
        return {"focus_z": float(zs[k] - Z_TOP),
                "spot_fwhm": float(xs[above[-1]] - xs[above[0]]) if len(above) else float("nan")}


CASE = dataclasses.replace(
    base, name="metalens_tidy3d_full",
    description="Tidy3D gallery metalens, full domain to 1.7 f (direct DFT focus)",
    z_span=(-0.6, Z_MAX), custom=hooks,
    reference=dataclasses.replace(
        base.reference,
        runtimes_s={"tidy3d solver (log)": {"res46.5": 23.2},
                    "tidy3d time stepping (log)": {"res46.5": 13.4}},
        cells={"tidy3d": {"res46.5": 1.7426e8}}, steps={"tidy3d": {"res46.5": 31633}},
        notes="Notebook full run: dx = dy = 21.5 nm, auto z, absorber x/y, PML z."),
    tags=("tier3", "speed"))
