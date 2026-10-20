"""PSR diagnostic: the psr case over 1.45-1.65 um, to see where TM0->TE0
conversion peaks (a resonance shifted by under-resolution converges with
resolution; a low plateau points at a modeling difference)."""
import dataclasses

from meepbench import Wavelengths
from meepbench.cases import load

base = load("psr")
CASE = dataclasses.replace(
    base, name="psr_broadband",
    description="PSR TM0->TE0 over 1.45-1.65 um (diagnostic, no reference)",
    wavelengths=Wavelengths(start=1.45, stop=1.65, step=0.005, center=1.55),
    reference=dataclasses.replace(base.reference, values={}),
    tags=("diagnostic",))
