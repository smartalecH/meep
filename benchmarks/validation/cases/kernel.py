"""Kernel microbenchmark: uniform medium, no PML, fixed step count.

Compare Gcell-steps/s with OPN Sept 2024 (Tidy3D): 20 Gcell/s on A100,
33 Gcell/s on H100 (fp32). Size with --res (default grid 512^3 at res 20).
"""
import meep as mp

from meepbench import Case, Geometry, Reference, Run, Wavelengths


def point_source(case, ctx):
    if ctx["stage"] == "simulation":
        return {"sources": [mp.Source(mp.GaussianSource(1 / 1.55, fwidth=0.2), mp.Ez,
                                      center=mp.Vector3())],
                "boundary_layers": []}


CASE = Case(
    name="kernel",
    description="Uniform-medium update-rate microbenchmark (no PML, no monitors)",
    geometry=Geometry(polygons=lambda: ({}, {})),
    stack={}, background="SiO2",
    wavelengths=Wavelengths(start=1.55, stop=1.55, step=1, center=1.55),
    domain_xy=(25.6, 25.6), z_span=(-12.8, 12.8), pml=0.0,
    run=Run(until=10.0, after_sources=False),
    custom=point_source,
    reference=Reference(source="Optica OPN 35(9), Sept 2024 (Tidy3D)",
                        notes="Tidy3D: 20 Gcell/s A100, 33 Gcell/s H100, fp32."),
    tags=("tier3", "speed"),
)
