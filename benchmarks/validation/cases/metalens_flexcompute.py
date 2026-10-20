"""flexcompute/metalens (GPL-3.0; own script from the published parameters):
TiO2 PB-phase fins (85 x 410 x 600 nm, 430 nm pitch) on SiO2, 660 nm, NA 0.8,
side 100 wavelengths = 153 x 153 = 23,409 fins, 18 grid points per wavelength
(dl = 36.7 nm), z = lambda + fins + 1.1 f + lambda. Tidy3D solver log: 290.3 s,
2.73e9 cells, ~6,235 steps (decay shutoff); 40/fwidth run time.

MEEPBENCH_METALENS_N sets fins per side (default 153, the full lens) for
scaling studies; the focal length follows the lens size. Linear x
polarization (the original uses circular); checks the on-axis focus and the
focal spot (diffraction limit 0.514 lambda/NA = 0.42 um)."""
import math
import os

import meep as mp
import numpy as np

from meepbench import Box, Case, Geometry, Layer, Reference, Run, Wavelengths
from meepbench import materials

materials.register("SiO2_1.46", 1.46)

WL, NA = 0.660, 0.8
W, L, H, S = 0.085, 0.410, 0.600, 0.430
N = int(os.environ.get("MEEPBENCH_METALENS_N", 153))
SIDE = N * S
F = SIDE / 2 / NA * math.sqrt(1 - NA ** 2)
SPACE = WL
Z_TOP = H                                       # fins on the substrate top at z = 0
Z_MAX = Z_TOP + 1.1 * F + SPACE


def layout():
    c = S * np.arange(N) - SIDE / 2 + S / 2
    xs, ys = (a.ravel() for a in np.meshgrid(c, c, indexing="ij"))
    theta = math.pi / WL * (F - np.sqrt(xs ** 2 + ys ** 2 + F ** 2))
    base = np.array([[L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2], [L / 2, -W / 2]])
    polys = [base @ np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]]) + [x, y]
             for x, y, t in zip(xs, ys, theta)]
    return {(1, 0): polys}, {}, (-SIDE / 2, -SIDE / 2, SIDE / 2, SIDE / 2)


def hooks(case, ctx):
    built = ctx["built"]
    if ctx["stage"] == "simulation":
        cell = ctx["kwargs"]["cell_size"]
        fw = 1 / WL / 10
        src = mp.Source(mp.GaussianSource(1 / WL, fwidth=2 * math.pi * fw, is_integrated=True),
                        mp.Ex, center=built._local(0, 0, -SPACE / 2), size=mp.Vector3(cell.x, cell.y, 0))
        return {"sources": [src], "boundary_layers": [mp.PML(case.pml, direction=mp.Z)],
                "k_point": mp.Vector3()}           # periodic x/y, as the original
    if ctx["stage"] == "monitors":
        sim = ctx["sim"]
        built.monitors["axis"] = sim.add_dft_fields(
            [mp.Ex, mp.Ey, mp.Ez], [1 / WL], center=built._local(0, 0, Z_TOP + F),
            size=mp.Vector3(0, 0, min(8.0, 0.6 * F)))
        built.monitors["xline"] = sim.add_dft_fields(
            [mp.Ex, mp.Ey, mp.Ez], [1 / WL], center=built._local(0, 0, Z_TOP + F),
            size=mp.Vector3(4.0, 0, 1.0))
    if ctx["stage"] == "observables":
        sim = built.sim

        def intensity(key):
            return sum(np.abs(np.squeeze(sim.get_dft_array(built.monitors[key], c, 0))) ** 2
                       for c in (mp.Ex, mp.Ey, mp.Ez))
        span = min(8.0, 0.6 * F)
        axial = intensity("axis")
        zs = Z_TOP + F - span / 2 + np.arange(axial.size) * span / max(axial.size - 1, 1)
        k = int(np.argmax(axial))
        plane = intensity("xline")
        zp = Z_TOP + F - 0.5 + np.arange(plane.shape[1]) * 1.0 / max(plane.shape[1] - 1, 1)
        lateral = plane[:, int(np.argmin(np.abs(zp - zs[k])))]
        xs = -2.0 + np.arange(lateral.size) * 4.0 / max(lateral.size - 1, 1)
        above = np.where(lateral >= lateral.max() / 2)[0]
        return {"focus_z": float(zs[k] - Z_TOP),
                "spot_fwhm": float(xs[above[-1]] - xs[above[0]]) if len(above) else float("nan")}


CASE = Case(
    name="metalens_flexcompute" + ("" if N == 153 else f"_N{N}"),
    description=f"flexcompute/metalens: {N}x{N} TiO2 fins, 660 nm, NA 0.8 (side {SIDE:.1f} um)",
    geometry=Geometry(polygons=layout, max_edge=None),
    stack={(1, 0): Layer(0.0, H, "TiO2")}, background="air",
    boxes=[Box("SiO2_1.46", zmax=0.0)],
    wavelengths=Wavelengths(start=WL, stop=WL, step=1, center=WL),
    domain_xy=(SIDE, SIDE), z_span=(-SPACE, Z_MAX), pml=15 * WL / 18,
    n_max=2.40,
    run=Run(until=40 / (1 / WL / 10), after_sources=True),
    custom=hooks,
    reference=Reference(
        source="flexcompute/metalens (parameters); diffraction limit",
        values={"focus_z": (F, 0.08 * F), "spot_fwhm": (0.514 * WL / NA, 0.12)},
        runtimes_s={"tidy3d solver (log)": {"res27.3": 290.3}},
        cells={"tidy3d": {"res27.3": 2.73e9}}, steps={"tidy3d": {"res27.3": 6235}},
        notes="18 grid points per wavelength = res 27.27; periodic x/y, PML z."),
    tags=("tier3", "speed", "scale"))
