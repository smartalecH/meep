"""Bus waveguide coupled to a 5 um ring through a 50 nm gap (Tidy3D notebook
"WaveguideToRingCoupling"); the domain cuts through the ring, which runs into
the PML. Speed reference: Tidy3D log 31.8 s solver (PML variant), 9.2e6 grid
points x 62,950 steps (auto grid, 25 steps/wavelength, 2 ps).

Own script from the notebook's published parameters. Differences: constant
Si/SiO2 indices (the notebook uses cSi Li1993 and SiO2 Horiba fits), uniform
grid at 25 cells/wavelength in Si (res 58), and only the through port (the
notebook also has an angled bend-mode drop monitor). No accuracy reference:
the check is that the through spectrum shows ring resonances.
"""
import math

import meep as mp
import numpy as np

from meepbench import Case, Geometry, Layer, ModeSource, Observable, Port, Reference, Run, \
    Wavelengths

LDA0, W, H, GAP, R = 1.55, 0.5, 0.22, 0.05, 5.0
LX, LY, LZ = 2 * R + 2 * LDA0, R / 2 + GAP + 2 * W + LDA0, 9 * H
CY = LY / 4                                         # notebook: center (0, Ly/4, 0)


def layout():
    ring_y = W + GAP + R
    t = np.linspace(0, math.pi, 400)
    half = []
    for sign in (1, -1):                            # two half annuli (a prism has no holes)
        outer = np.c_[(R + W / 2) * np.cos(t), ring_y + sign * (R + W / 2) * np.sin(t)]
        inner = np.c_[(R - W / 2) * np.cos(t[::-1]), ring_y + sign * (R - W / 2) * np.sin(t[::-1])]
        half.append(np.vstack([outer, inner]))
    bus = np.array([(-20, -W / 2), (20, -W / 2), (20, W / 2), (-20, W / 2)])
    ports = {"in": Port((-R, 0.0), 180.0, W), "through": Port((R, 0.0), 0.0, W)}
    bbox = (-LX / 2, CY - LY / 2, LX / 2, CY + LY / 2)
    return {(1, 0): [bus] + half}, ports, bbox


CASE = Case(
    name="waveguide_ring_tidy3d",
    description="Tidy3D WaveguideToRingCoupling: 5 um ring, 50 nm gap, ring cut by the domain",
    geometry=Geometry(polygons=layout),
    stack={(1, 0): Layer(-H / 2, H / 2, "Si")}, background="SiO2",
    wavelengths=Wavelengths(start=1.5, stop=1.6, step=0.001, center=LDA0),
    domain_xy=(LX, LY), z_span=(-LZ / 2, LZ / 2), margin_xy=0.0,
    symmetries=[mp.Mirror(mp.Z, phase=+1)],          # notebook symmetry (0, 0, 1), TE
    source=ModeSource("in", "TE0", distance=LDA0 / 4, monitor_gap=0.1),
    observables=[Observable("through", "ring_resonance", "through", "TE0", near=LDA0,
                            distance=LDA0 / 4, measure="flux")],
    mode_plane=(5 * W, 6 * H),
    run=Run(until=600.0, after_sources=False),      # run_time 2 ps
    eps_averaging=True,
    reference=Reference(
        source="flexcompute/tidy3d-notebooks WaveguideToRingCoupling (parameters, solver log)",
        runtimes_s={"tidy3d solver (log, PML)": {25.0: 31.8},
                    "tidy3d solver (log, absorber)": {25.0: 20.8}},
        cells={"tidy3d": {25.0: 9.2e6}}, steps={"tidy3d": {25.0: 62950}},
        notes="Tidy3D auto grid (25 steps/wavelength in each material) vs our uniform grid."),
    tags=("tier3", "speed"),
)
