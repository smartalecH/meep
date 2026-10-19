"""All-pass ring, R = 10 um (gdsfactory `ring_single`), through-port resonance."""
from meepbench import Case, Geometry, ModeSource, Observable, Run, Wavelengths
from meepbench.presets import SOI, SOI_Z, paper_domain, paper_reference

CASE = Case(
    name="ring",
    description="SOI all-pass ring R=10 um, gap 0.2 um: lambda0, FWHM, Q (arXiv 2506.16665)",
    geometry=Geometry(gdsfactory="ring_single", params=dict(gap=0.2, radius=10, length_x=4)),
    stack=SOI, background="SiO2", z_span=SOI_Z,
    # 0.2 nm sampling resolves the ~0.9 nm FWHM; 30 nm spans ~3 FSRs.
    wavelengths=Wavelengths(start=1.535, stop=1.565, step=0.0002, center=1.545),
    domain_xy=paper_domain("ring"),
    source=ModeSource("o1", "TE0"),
    observables=[Observable("through", "ring_resonance", "o2", "TE0", near=1.545)],
    # DFT-decay stopping quits before the high-Q ring rings down (seen at res 8:
    # stopped at t~420); run the paper's Tidy3D physical time, 6.4 ps.
    run=Run(until=1920.0),
    reference=paper_reference("ring", {"through.lambda0": (1.545, 0.002),
                                       "through.fwhm": (0.00089, 0.00015),
                                       "through.Q": (1780.0, 200.0)}),
    tags=("tier2", "paper"),
)
