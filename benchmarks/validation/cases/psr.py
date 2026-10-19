"""Polarization splitter-rotator (gdsfactory `polarization_splitter_rotator`)
under SiN top cladding: TM0 in -> TE0 at the upper output."""
from meepbench import Box, Case, Geometry, ModeSource, Observable
from meepbench.presets import PAPER_BAND, SOI, SOI_Z, paper_domain, paper_reference

CASE = Case(
    name="psr",
    description="SOI PSR with Si3N4 top cladding, TM0 o1 -> TE0 o2 (arXiv 2506.16665)",
    geometry=Geometry(gdsfactory="polarization_splitter_rotator"),
    stack=SOI, background="SiO2", z_span=SOI_Z, wavelengths=PAPER_BAND,
    boxes=[Box("SiN", zmin=0.0)],
    domain_xy=paper_domain("psr"),
    source=ModeSource("o1", "TM0"),
    observables=[Observable("tm0_to_te0", "mode_transmission", "o2", "TE0"),
                 Observable("tm0_through", "mode_transmission", "o3", "TM0")],
    reference=paper_reference("psr", {"tm0_to_te0": (0.95, 0.03)}),
    tags=("tier2", "paper"),
)
