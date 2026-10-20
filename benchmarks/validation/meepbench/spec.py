"""Case specification: a declarative description of one validation problem.

A Case says *what* to simulate (geometry source, layer stack, materials,
sources, monitors, observables, run control) and *what to compare against*
(reference values and published solver runtimes). meepbench.runner turns it
into an mp.Simulation for any resolution, backend, precision and rank count.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

LayerKey = Tuple[int, int]


@dataclasses.dataclass
class Layer:
    """GDS layer extruded between zmin and zmax (um) in `material`."""
    zmin: float
    zmax: float
    material: str
    sidewall_angle: float = 0.0  # radians, as mp.Prism


@dataclasses.dataclass
class Box:
    """Extra axis-aligned block (e.g. a cladding region). None = whole cell."""
    material: str
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None
    zmin: Optional[float] = None
    zmax: Optional[float] = None


@dataclasses.dataclass
class Geometry:
    """Where the 2D layout comes from. Exactly one source is used:

    gdsfactory: component factory name in gf.components (or a callable)
                plus keyword parameters;
    gds:        path to a GDS file and the cell name;
    polygons:   callable returning {layer: [Nx2 vertex arrays]} and
                {port_name: Port} (for hand-written layouts).

    Ports from gdsfactory/GDS are extended by `port_extension` um so
    waveguides run through the PML.
    """
    gdsfactory: Union[None, str, Callable[..., Any]] = None
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    gds: Optional[str] = None
    cell: Optional[str] = None
    polygons: Optional[Callable[[], Tuple[Dict[LayerKey, list], Dict[str, "Port"]]]] = None
    port_extension: float = 10.0
    # Split polygon edges longer than this (um) before building prisms. libctl's
    # (and the GPU material IR's) point-on-edge test uses an angular tolerance,
    # so points up to ~0.002 x edge length off a long edge count as inside:
    # 44 um taper edges grew by ~0.05 um. Short edges bound that to ~1 nm.
    # None disables.
    max_edge: Optional[float] = 0.2


@dataclasses.dataclass
class Port:
    """Waveguide port: center (x, y) in layout coordinates, outward
    orientation in degrees (gdsfactory convention: 0 = +x faces outward),
    and waveguide width."""
    center: Tuple[float, float]
    orientation: float
    width: float


@dataclasses.dataclass
class ModeSource:
    """Eigenmode source launched into the device from `port`."""
    port: str
    mode: str = "TE0"           # "TE0", "TM0", "TE1", ... or "band:N"
    distance: float = 0.6       # source plane, um outside the port (on the extension)
    monitor_gap: float = 0.3    # incident-power monitor, um downstream of the source
    # Meep GaussianSource fwidth / monitored frequency span. Meep's fwidth is
    # 2*pi times Tidy3D's GaussianPulse fwidth for the same pulse, so 2*pi
    # reproduces Tidy3D's fwidth = span (arXiv 2506.16665 pipeline); smaller
    # factors give longer pulses and more steps.
    fwidth_factor: float = 6.283185307179586


@dataclasses.dataclass
class Observable:
    """Scalar or spectral quantity extracted after the run.

    kind:
      "mode_transmission": |b_out(port, mode)|^2 / |a_in(source)|^2
      "flux_ratio":        flux through a port plane / incident flux
      "ring_resonance":    through-port dip fit -> lambda0, FWHM, Q, FSR
    at: wavelength (um) at which the scalar is reported (spectra are kept).
    """
    name: str
    kind: str
    port: str
    mode: str = "TE0"
    at: Optional[float] = None
    near: Optional[float] = None      # ring: resonance closest to this wavelength
    distance: float = 0.3             # monitor plane, um outside the port
    # ring_resonance only: "mode" (|b|^2 of `mode`) or "flux" (net power; no
    # per-frequency mode solves, right for single-mode buses with many points)
    measure: str = "mode"


@dataclasses.dataclass
class Wavelengths:
    start: float
    stop: float
    step: float
    center: float
    points: Optional[Sequence[float]] = None   # explicit list, overrides the range

    def values(self) -> List[float]:
        if self.points is not None:
            return [float(w) for w in self.points]
        n = int(round((self.stop - self.start) / self.step)) + 1
        return [self.start + i * self.step for i in range(n)]


@dataclasses.dataclass
class Run:
    """Run control. stop="dft": stop when DFT fields change by less than
    `decay` (relative) between checks (stop_when_dft_decayed); stop="energy":
    when total field energy falls by `decay` from its peak (Tidy3D-style
    shutoff, stop_when_energy_decayed every `check_dt`). until: fixed time
    after sources (um/c). max_time caps decay runs."""
    stop: str = "dft"
    decay: Optional[float] = 1e-5
    check_dt: float = 50.0
    until: Optional[float] = None
    min_time: float = 0.0         # 0: one device transit (see runner)
    max_time: Optional[float] = None
    after_sources: bool = True    # until counts from source turn-off (else from t=0)


@dataclasses.dataclass
class Reference:
    """What a run is compared against.

    values: observable name -> (expected, absolute tolerance)
    runtimes_s: solver label -> {cells_per_wavelength: seconds}
    cells / steps: solver label -> {cells_per_wavelength: count}
    """
    source: str = ""
    values: Dict[str, Tuple[float, float]] = dataclasses.field(default_factory=dict)
    runtimes_s: Dict[str, Dict[float, float]] = dataclasses.field(default_factory=dict)
    cells: Dict[str, Dict[float, float]] = dataclasses.field(default_factory=dict)
    steps: Dict[str, Dict[float, float]] = dataclasses.field(default_factory=dict)
    # observable name -> {label: callable() -> (wavelengths_um, values) or None
    # when the data is unavailable}; reported as max/rms differences.
    spectra: Dict[str, Dict[str, Callable[[], Any]]] = dataclasses.field(default_factory=dict)
    notes: str = ""


@dataclasses.dataclass
class Case:
    name: str
    description: str
    geometry: Geometry
    stack: Dict[LayerKey, Layer]
    background: str
    wavelengths: Wavelengths
    source: Optional[ModeSource] = None
    observables: List[Observable] = dataclasses.field(default_factory=list)
    boxes: List[Box] = dataclasses.field(default_factory=list)
    margin_xy: float = 1.0                       # um beyond the layout bbox
    domain_xy: Optional[Tuple[float, float]] = None  # explicit non-PML x/y size (centered on bbox)
    z_span: Tuple[float, float] = (-2.0, 2.0)    # um, non-PML region
    pml: float = 1.0
    symmetries: Sequence[Any] = ()
    mode_plane: Tuple[float, float] = (4.0, 2.0)  # extra width beyond port width, height (um)
    run: Run = dataclasses.field(default_factory=Run)
    reference: Reference = dataclasses.field(default_factory=Reference)
    eps_averaging: bool = False
    # Resolution convention: cells per wavelength in the highest-index
    # material at the shortest wavelength (arXiv 2506.16665 / Tidy3D).
    n_max: float = 3.48
    # Escape hatch for cases the declarative fields cannot express: called as
    # custom(case, context) with context["stage"] in {"simulation" (return
    # extra mp.Simulation kwargs), "monitors" (add monitors to
    # context["sim"]), "observables" (return extra values)}; see runner.
    custom: Optional[Callable[..., Any]] = None
    tags: Sequence[str] = ()
