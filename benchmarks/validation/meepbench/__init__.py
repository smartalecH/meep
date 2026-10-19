"""meepbench: one declarative harness for Meep validation and speed cases.

A case (meepbench.spec.Case) names a layout (gdsfactory component, GDS file
or polygons), a layer stack, ports, sources, observables and references; the
runner builds the mp.Simulation at any resolution, backend, precision and
rank count and writes a JSON result. See ../README.md.
"""
from .spec import (Box, Case, Geometry, Layer, ModeSource, Observable, Port, Reference, Run,
                   Wavelengths)

__all__ = ["Box", "Case", "Geometry", "Layer", "ModeSource", "Observable", "Port", "Reference",
           "Run", "Wavelengths"]
