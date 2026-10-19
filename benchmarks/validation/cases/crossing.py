"""Waveguide crossing (gdsfactory generic PDK `crossing`), TE0 through."""
from meepbench import Case, Geometry, ModeSource, Observable
from meepbench.presets import PAPER_BAND, SOI, SOI_Z, paper_domain, paper_reference

CASE = Case(
    name="crossing",
    description="SOI waveguide crossing, TE0 o1 -> o3 (arXiv 2506.16665)",
    geometry=Geometry(gdsfactory="crossing"),
    stack=SOI, background="SiO2", z_span=SOI_Z, wavelengths=PAPER_BAND,
    domain_xy=paper_domain("crossing"),
    source=ModeSource("o1", "TE0"),
    observables=[Observable("through", "mode_transmission", "o3", "TE0")],
    reference=paper_reference("crossing", {"through": (0.957, 0.015)}),
    tags=("tier2", "paper"),
)
