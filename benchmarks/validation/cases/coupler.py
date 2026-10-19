"""Directional coupler (gdsfactory `coupler`, gap 236 nm, 20 um), TE0 cross."""
from meepbench import Case, Geometry, ModeSource, Observable
from meepbench.presets import PAPER_BAND, SOI, SOI_Z, paper_domain, paper_reference

CASE = Case(
    name="coupler",
    description="SOI directional coupler, TE0 o1 -> cross o3 (arXiv 2506.16665)",
    geometry=Geometry(gdsfactory="coupler", params=dict(gap=0.236, length=20, dx=10, dy=4)),
    stack=SOI, background="SiO2", z_span=SOI_Z, wavelengths=PAPER_BAND,
    domain_xy=paper_domain("coupler"),
    source=ModeSource("o1", "TE0"),
    observables=[Observable("cross", "mode_transmission", "o3", "TE0"),
                 Observable("bar", "mode_transmission", "o4", "TE0")],
    # 41-47% across solvers/resolutions; the gap makes it subpixel-sensitive.
    reference=paper_reference("coupler", {"cross": (0.44, 0.04)}),
    tags=("tier2", "paper"),
)
