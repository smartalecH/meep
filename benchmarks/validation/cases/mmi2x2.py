"""2x2 MMI with S-bends (gdsfactory `mmi2x2_with_sbend`), TE0 cross port."""
from meepbench import Case, Geometry, ModeSource, Observable
from meepbench.presets import PAPER_BAND, SOI, SOI_Z, paper_domain, paper_reference

CASE = Case(
    name="mmi2x2",
    description="SOI 2x2 MMI, TE0 o1 -> cross o3 (arXiv 2506.16665)",
    geometry=Geometry(gdsfactory="mmi2x2_with_sbend"),
    stack=SOI, background="SiO2", z_span=SOI_Z, wavelengths=PAPER_BAND,
    domain_xy=paper_domain("mmi2x2"),
    source=ModeSource("o1", "TE0"),
    observables=[Observable("cross", "mode_transmission", "o3", "TE0"),
                 Observable("bar", "mode_transmission", "o4", "TE0")],
    reference=paper_reference("mmi2x2", {"cross": (0.48, 0.03)}),
    tags=("tier2", "paper"),
)
