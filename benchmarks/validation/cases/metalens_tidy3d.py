"""10 um TiO2 metalens, NA 0.5, 600 nm (Tidy3D gallery "Metalens"), near-field
variant: lens + 1.2 um of air simulated, focal plane reached by near-to-far
projection (the notebook's second run; Tidy3D: 541x541x144 = 4.3e7 cells,
31,633 steps, 10.8 s solver / 8.8 s time stepping).

Own script from the notebook's published parameters: 23x23 PB-phase fins
(85 x 410 x 600 nm, period 430 nm) on a 100 nm SiO2 slab. Linear x
polarization (the notebook uses circular): half the power sees the focusing
phase, which is enough to locate the focus and measure the spot. Checks:
on-axis focus at f = 8.565 um above the fins, focal spot FWHM near the
diffraction limit 0.514 lambda/NA = 0.617 um.
"""
import math

import meep as mp
import numpy as np

from meepbench import Box, Case, Geometry, Layer, Reference, Run, Wavelengths
from meepbench import materials

materials.register("SiO2_1.46", 1.46)

WL, NA = 0.6, 0.5
W, L, H, S = 0.085, 0.410, 0.600, 0.430
N = int(10 / S)                                   # 23 fins per side
SIDE = N * S                                      # 9.89 um
F = SIDE / 2 / NA * math.sqrt(1 - NA ** 2)        # 8.565 um
SUB = 0.1
Z_SRC = -0.6 + 0.06                               # plane wave, in air below the slab
Z_TOP = SUB + H                                   # fin top
Z_N2F = Z_TOP + 0.63                              # notebook's projection plane


def layout():
    c = S * np.arange(N) - SIDE / 2 + S / 2
    xs, ys = (a.ravel() for a in np.meshgrid(c, c, indexing="ij"))
    theta = math.pi / WL * (F - np.sqrt(xs ** 2 + ys ** 2 + F ** 2))
    base = np.array([[L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2], [L / 2, -W / 2]])
    polys = []
    for x, y, t in zip(xs, ys, theta):
        rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        polys.append(base @ rot.T + [x, y])
    return {(1, 0): polys}, {}, (-SIDE / 2, -SIDE / 2, SIDE / 2, SIDE / 2)


def hooks(case, ctx):
    built = ctx["built"]
    if ctx["stage"] == "simulation":
        cell = ctx["kwargs"]["cell_size"]
        # is_integrated: the plane-wave source spans the PML too
        src = mp.Source(mp.GaussianSource(1 / WL, fwidth=1 / WL / 10, is_integrated=True),
                        mp.Ex, center=built._local(0, 0, Z_SRC),
                        size=mp.Vector3(cell.x, cell.y, 0))
        return {"sources": [src]}
    if ctx["stage"] == "monitors":
        sx, sy, _ = built.domain
        built.monitors["n2f"] = ctx["sim"].add_near2far(
            [1 / WL], mp.Near2FarRegion(center=built._local(0, 0, Z_N2F),
                                        size=mp.Vector3(sx, sy, 0)))
    if ctx["stage"] == "observables":
        sim, n2f = built.sim, built.monitors["n2f"]

        def intensity(x, y, z):
            e = np.array(sim.get_farfield(n2f, built._local(x, y, Z_TOP + z)))[:3]
            return float(np.sum(np.abs(e) ** 2))
        zs = np.linspace(F - 4, F + 4, 81)
        axial = np.array([intensity(0, 0, z) for z in zs])
        z_peak = float(zs[np.argmax(axial)])
        xs = np.linspace(-2, 2, 161)
        lateral = np.array([intensity(x, 0, z_peak) for x in xs])
        half = lateral.max() / 2
        above = np.where(lateral >= half)[0]
        fwhm = float(xs[above[-1]] - xs[above[0]]) if len(above) else float("nan")
        return {"focus_z": z_peak, "spot_fwhm": fwhm}


CASE = Case(
    name="metalens_tidy3d",
    description="Tidy3D gallery metalens (10 um, NA 0.5, 600 nm), near field + projection",
    geometry=Geometry(polygons=layout),
    stack={(1, 0): Layer(SUB, SUB + H, "TiO2")}, background="air",
    boxes=[Box("SiO2_1.46", zmin=0.0, zmax=SUB)],
    wavelengths=Wavelengths(start=WL, stop=WL, step=1, center=WL),
    domain_xy=(SIDE, SIDE), z_span=(-0.6, Z_TOP + 1.2), pml=0.6,
    n_max=2.40,
    run=Run(until=50 / (1 / WL / 10), after_sources=False),   # run_time = 50 / fwidth
    custom=hooks,
    reference=Reference(
        source="flexcompute/tidy3d-notebooks Metalens (parameters); diffraction limit",
        values={"focus_z": (F, 0.5), "spot_fwhm": (0.514 * WL / NA, 0.12)},
        runtimes_s={"tidy3d solver (log)": {"res46.5": 10.8},
                    "tidy3d time stepping (log)": {"res46.5": 8.84}},
        cells={"tidy3d": {"res46.5": 4.3048e7}}, steps={"tidy3d": {"res46.5": 31633}},
        notes="Notebook grid: dx = dy = S/20 = 21.5 nm (res 46.5), auto z; absorber x/y."),
    tags=("tier3", "speed"),
)
