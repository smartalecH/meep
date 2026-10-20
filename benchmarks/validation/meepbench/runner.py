"""Case -> mp.Simulation -> run -> observables, timing and reference checks."""
from __future__ import annotations

import dataclasses
import inspect
import json
import math
import platform
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import meep as mp

from . import geometry as layout
from . import materials
from .spec import Box, Case, Observable, Port


@dataclasses.dataclass
class Options:
    cells_per_wavelength: Optional[float] = None   # paper/Tidy3D convention
    resolution: Optional[float] = None             # px/um, overrides the above
    backend: str = "nvidia"                         # nvidia | cpu
    precision: str = "native"                       # native | mixed | f32
    materials: str = "const"                        # const | dispersive
    eps_averaging: Optional[bool] = None            # None: the case default
    until: Optional[float] = None                   # override run control (um/c)
    stop: Optional[str] = None                      # override Run.stop: dft | energy
    verbosity: int = 0


def _band(mode: str) -> int:
    """TE0 -> 1, TM0 -> 2, TE1 -> 3 ... (fully vectorial 3D bands ordered by
    effective index for a single-mode-ish Si wire); "band:N" is explicit."""
    if mode.startswith("band:"):
        return int(mode.split(":", 1)[1])
    table = {"TE0": 1, "TM0": 2, "TE1": 3, "TM1": 4, "TE2": 5}
    if mode not in table:
        raise ValueError(f"unknown mode {mode!r}; use TE0/TM0/TE1/... or band:N")
    return table[mode]


def _supports_backend_kwargs() -> bool:
    return "backend" in inspect.signature(mp.Simulation.__init__).parameters


class Built:
    """Simulation plus the bookkeeping needed to extract observables."""

    def __init__(self, case: Case, options: Options):
        self.case, self.options = case, options
        self.polygons, self.ports, bbox = layout.load(case.geometry)
        x0, y0, x1, y1 = bbox
        self.center = ((x0 + x1) / 2, (y0 + y1) / 2, sum(case.z_span) / 2)
        sx, sy = case.domain_xy or (x1 - x0 + 2 * case.margin_xy, y1 - y0 + 2 * case.margin_xy)
        self.domain = (sx, sy, case.z_span[1] - case.z_span[0])
        names = {l.material for l in case.stack.values()} | {case.background} | \
                {b.material for b in case.boxes}
        lo, hi = min(case.wavelengths.values()), max(case.wavelengths.values())
        self.band = (lo - 0.05, hi + 0.05)
        self.media = {n: materials.medium(n, options.materials, self.band) for n in names}
        if options.resolution:
            self.resolution = float(options.resolution)
        else:
            cpl = options.cells_per_wavelength or 10
            n_max = materials.max_index(names, options.materials, lo) if options.materials \
                == "dispersive" else max(case.n_max, materials.max_index(names, "const", lo))
            self.resolution = float(math.ceil(cpl * n_max / lo))
        self.freqs = [1.0 / w for w in case.wavelengths.values()]
        self.monitors: Dict[str, Any] = {}
        self.sim = self._simulation()

    # -- geometry -----------------------------------------------------------
    def _local(self, x, y, z=None):
        cx, cy, cz = self.center
        return mp.Vector3(x - cx, y - cy, (z if z is not None else cz) - cz)

    def _objects(self) -> List[Any]:
        objects = []
        for box in self.case.boxes:   # cladding first: later objects take precedence
            objects.append(self._box(box))
        for key, layer in self.case.stack.items():
            for poly in self.polygons.get(key, []):
                if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
                    poly = poly[:-1]
                objects.append(mp.Prism(
                    vertices=[self._local(x, y, layer.zmin) for x, y in poly],
                    height=layer.zmax - layer.zmin, axis=mp.Vector3(0, 0, 1),
                    sidewall_angle=layer.sidewall_angle, material=self.media[layer.material]))
        return objects

    def _box(self, box: Box):
        cx, cy, cz = self.center
        sx, sy, sz = self.domain
        big = 2 * self.case.pml
        x0 = box.xmin if box.xmin is not None else cx - sx / 2 - big
        x1 = box.xmax if box.xmax is not None else cx + sx / 2 + big
        y0 = box.ymin if box.ymin is not None else cy - sy / 2 - big
        y1 = box.ymax if box.ymax is not None else cy + sy / 2 + big
        z0 = box.zmin if box.zmin is not None else cz - sz / 2 - big
        z1 = box.zmax if box.zmax is not None else cz + sz / 2 + big
        # A vertex-defined Prism rather than mp.Block (see tidy3d_mmi/README.md).
        # Open sides are snapped outward to multiples of 0.5 um in simulation
        # coordinates: the GPU material IR re-derives object bounds and
        # compares them exactly, which fails by an ulp for arbitrary values.
        lo, hi = self._local(x0, y0, z0), self._local(x1, y1, z1)
        snap_lo = lambda v, given: v if given is not None else math.floor(v * 2) / 2
        snap_hi = lambda v, given: v if given is not None else math.ceil(v * 2) / 2
        X0, Y0, Z0 = snap_lo(lo.x, box.xmin), snap_lo(lo.y, box.ymin), snap_lo(lo.z, box.zmin)
        X1, Y1, Z1 = snap_hi(hi.x, box.xmax), snap_hi(hi.y, box.ymax), snap_hi(hi.z, box.zmax)
        return mp.Prism(vertices=[mp.Vector3(X0, Y0, Z0), mp.Vector3(X1, Y0, Z0),
                                  mp.Vector3(X1, Y1, Z0), mp.Vector3(X0, Y1, Z0)],
                        height=Z1 - Z0, axis=mp.Vector3(0, 0, 1),
                        material=self.media[box.material])

    # -- ports --------------------------------------------------------------
    def _port_plane(self, port: Port, outward: float):
        """Center (sim coords), size and normal axis of a plane `outward` um
        outside a port; transverse size clipped to the non-PML domain."""
        theta = math.radians(port.orientation)
        ux, uy = round(math.cos(theta)), round(math.sin(theta))
        if abs(ux) + abs(uy) != 1:
            raise ValueError("meepbench ports must be axis-aligned")
        x, y = port.center[0] + ux * outward, port.center[1] + uy * outward
        width = port.width + self.case.mode_plane[0]
        height = min(self.case.mode_plane[1], self.domain[2])
        zc = self._port_z()
        center = self._local(x, y, zc)
        if ux:
            limit = self.domain[1] - 0.1
            width = min(width, limit, 2 * (self.domain[1] / 2 - abs(center.y)) - 0.05)
            size, axis = mp.Vector3(0, width, height), mp.X
        else:
            limit = self.domain[0] - 0.1
            width = min(width, limit, 2 * (self.domain[0] / 2 - abs(center.x)) - 0.05)
            size, axis = mp.Vector3(width, 0, height), mp.Y
        return center, size, axis, (ux, uy)

    def _port_z(self) -> float:
        """Port planes are centered on the extruded stack."""
        layers = self.case.stack.values()
        if not layers:
            return self.center[2]
        return (max(l.zmax for l in layers) + min(l.zmin for l in layers)) / 2

    # -- simulation -----------------------------------------------------------
    def _simulation(self):
        case, opts = self.case, self.options
        cell = mp.Vector3(*(d + 2 * case.pml for d in self.domain))
        fmin, fmax = min(self.freqs), max(self.freqs)
        sources = []
        if case.source is not None:
            fcen = (fmin + fmax) / 2
            df = case.source.fwidth_factor * (fmax - fmin)
            port = self.ports[case.source.port]
            center, size, axis, (ux, uy) = self._port_plane(port, case.source.distance)
            sources.append(mp.EigenModeSource(
                mp.GaussianSource(fcen, fwidth=df), center=center, size=size,
                direction=mp.NO_DIRECTION, eig_kpoint=mp.Vector3(-ux, -uy),
                eig_band=_band(case.source.mode), eig_parity=mp.NO_PARITY,
                eig_match_freq=True))
        averaging = case.eps_averaging if opts.eps_averaging is None else opts.eps_averaging
        kwargs: Dict[str, Any] = dict(
            cell_size=cell, resolution=self.resolution,
            boundary_layers=[mp.PML(case.pml)], geometry=self._objects(), sources=sources,
            default_material=self.media[case.background], symmetries=list(case.symmetries),
            eps_averaging=averaging)
        if _supports_backend_kwargs():
            kwargs.update(backend=opts.backend, precision=opts.precision,
                          accelerator_strict=not averaging)
        if case.custom is not None:
            extra = case.custom(case, {"stage": "simulation", "built": self, "kwargs": kwargs})
            if isinstance(extra, dict):
                kwargs.update(extra)
        sim = mp.Simulation(**kwargs)
        # Monitors: incident normalization just downstream of the source and
        # one monitor per observed (port, kind).
        if case.source is not None:
            src = case.source
            center, size, _, _ = self._port_plane(self.ports[src.port],
                                                  src.distance - src.monitor_gap)
            measures = {_measure(o) for o in case.observables}
            if "mode" in measures:
                self.monitors["__input__", "mode"] = sim.add_mode_monitor(
                    self.freqs, mp.ModeRegion(center=center, size=size))
            if "flux" in measures:
                self.monitors["__input__", "flux"] = sim.add_flux(
                    self.freqs, mp.FluxRegion(center=center, size=size))
        for obs in case.observables:
            kind = _measure(obs)
            if (obs.name, kind) in self.monitors:
                raise ValueError(f"duplicate observable name {obs.name!r}")
            center, size, _, _ = self._port_plane(self.ports[obs.port], obs.distance)
            self.monitors[obs.name, kind] = (
                sim.add_flux(self.freqs, mp.FluxRegion(center=center, size=size))
                if kind == "flux" else
                sim.add_mode_monitor(self.freqs, mp.ModeRegion(center=center, size=size)))
        if case.custom is not None:   # extra monitors (near2far, field planes, ...)
            case.custom(case, {"stage": "monitors", "built": self, "sim": sim})
        return sim

    def grid(self) -> Tuple[int, int, int]:
        return tuple(int(round((d + 2 * self.case.pml) * self.resolution)) for d in self.domain)


def _measure(obs: Observable) -> str:
    if obs.kind == "flux_ratio" or (obs.kind == "ring_resonance" and obs.measure == "flux"):
        return "flux"
    return "mode"


def _outward_sign(port: Port) -> Tuple[int, int]:
    """(axis, +1/-1): the port faces +axis or -axis."""
    theta = math.radians(port.orientation)
    ux, uy = round(math.cos(theta)), round(math.sin(theta))
    return (mp.X, ux) if ux else (mp.Y, uy)


def _flux(built: Built, key, port: Port, outgoing: bool) -> np.ndarray:
    """Net Poynting flux out of (or into) the device through a port plane."""
    _, sign = _outward_sign(port)
    flux = np.array(mp.get_fluxes(built.monitors[key]))
    return flux * (sign if outgoing else -sign)


def _mode_power(built: Built, key, port: Port, mode: str, outgoing: bool) -> np.ndarray:
    """|alpha|^2 of `mode` travelling out of (or into) the device at `port`."""
    axis, sign = _outward_sign(port)
    res = built.sim.get_eigenmode_coefficients(built.monitors[key], [_band(mode)],
                                               eig_parity=mp.NO_PARITY, direction=axis)
    forward_is_out = sign > 0
    index = 0 if forward_is_out == outgoing else 1
    return np.abs(res.alpha[0, :, index]) ** 2


def ring_fit(wavelengths: np.ndarray, transmission: np.ndarray, near: float) -> Dict[str, float]:
    """Through-port dip nearest `near`: center by parabola through the minimum,
    FWHM at half depth (linear crossings), Q = lambda0/FWHM, FSR to the
    neighbouring dip when present."""
    order = np.argsort(wavelengths)
    wl, t = wavelengths[order], transmission[order]
    minima = [i for i in range(1, len(t) - 1) if t[i] < t[i - 1] and t[i] <= t[i + 1]]
    if not minima:
        return {}
    i = min(minima, key=lambda k: abs(wl[k] - near))
    a, b, c = np.polyfit(wl[i - 1:i + 2], t[i - 1:i + 2], 2)
    lam0 = -b / (2 * a)
    tmin = float(np.polyval([a, b, c], lam0))
    lo_max = t[max(0, i - 50):i].max() if i > 0 else t[i]
    hi_max = t[i + 1:i + 51].max() if i + 1 < len(t) else t[i]
    half = (min(lo_max, hi_max) + tmin) / 2
    left = next((k for k in range(i, 0, -1) if t[k - 1] >= half), None)
    right = next((k for k in range(i, len(t) - 1) if t[k + 1] >= half), None)
    out = {"lambda0": float(lam0), "extinction_db": float(10 * np.log10(min(lo_max, hi_max) / max(tmin, 1e-12)))}
    if left is not None and right is not None:
        wl_l = np.interp(half, [t[left], t[left - 1]], [wl[left], wl[left - 1]])
        wl_r = np.interp(half, [t[right], t[right + 1]], [wl[right], wl[right + 1]])
        out["fwhm"] = float(wl_r - wl_l)
        out["Q"] = float(lam0 / out["fwhm"]) if out["fwhm"] > 0 else float("nan")
    others = [k for k in minima if k != i]
    if others:
        out["fsr"] = float(min(abs(wl[k] - wl[i]) for k in others))
    return out


def run(case: Case, options: Options) -> Dict[str, Any]:
    if options.backend == "nvidia":
        # From Python the NVIDIA backend needs a communicator context, which a
        # one-group split installs (see tidy3d_mmi/README.md).
        mp.divide_parallel_processes(1)
    mp.verbosity(options.verbosity)
    t_wall = time.time()
    built = Built(case, options)
    t_build = time.time() - t_wall
    sim = built.sim
    t0 = time.time()
    sim.init_sim()
    t_init = time.time() - t0
    # One step timed apart from the run: on the NVIDIA backend the first step
    # compiles material recipes and captures graphs, which is setup, not
    # stepping. It goes through sim.run so pymeep creates its DFT monitors
    # before the backend freezes its storage catalog (a bare fields.step()
    # leaves them out and dft_norm later fails).
    t0 = time.time()
    sim.run(until=sim.fields.dt)
    t_first = time.time() - t0
    run_ctl = case.run
    t0 = time.time()
    until = options.until if options.until is not None else run_ctl.until
    if until is not None:
        if run_ctl.after_sources:
            sim.run(until_after_sources=until)
        else:
            sim.run(until=until)
    elif (options.stop or run_ctl.stop) == "energy":
        sim.run(until_after_sources=mp.stop_when_energy_decayed(dt=run_ctl.check_dt,
                                                                decay_by=run_ctl.decay))
    else:
        # DFT decay must not be judged before light crosses the device: with a
        # short pulse the input monitor settles first and the check would stop
        # with nothing at the outputs. Default floor: one transit at group
        # index ~1.2 n_max along the in-plane Manhattan path.
        # minimum_run_time is absolute simulation time, so add the source
        # duration (last_source_time) to the transit.
        min_time = run_ctl.min_time or (sim.fields.last_source_time() + 1.2 * case.n_max *
                                        (built.domain[0] + built.domain[1]))
        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=run_ctl.decay, minimum_run_time=min_time,
            maximum_run_time=run_ctl.max_time))
    t_run = time.time() - t0
    steps = int(sim.fields.t)

    t_obs0 = time.time()
    wl = np.array(case.wavelengths.values())
    values: Dict[str, Any] = {}
    spectra: Dict[str, List[float]] = {}
    order = np.argsort(wl)
    incident = {}
    if case.source is not None:
        port = built.ports[case.source.port]
        if ("__input__", "mode") in built.monitors:
            incident["mode"] = _mode_power(built, ("__input__", "mode"), port,
                                           case.source.mode, outgoing=False)
        if ("__input__", "flux") in built.monitors:
            incident["flux"] = _flux(built, ("__input__", "flux"), port, outgoing=False)
    for obs in case.observables:
        port = built.ports[obs.port]
        if obs.kind not in ("flux_ratio", "mode_transmission", "ring_resonance"):
            raise ValueError(f"unknown observable kind {obs.kind!r}")
        if _measure(obs) == "flux":
            spectrum = _flux(built, (obs.name, "flux"), port, outgoing=True) / incident["flux"]
        else:
            spectrum = _mode_power(built, (obs.name, "mode"), port, obs.mode,
                                   outgoing=True) / incident["mode"]
        spectra[obs.name] = spectrum.tolist()
        if obs.kind == "ring_resonance":
            for key, val in ring_fit(wl, spectrum, obs.near or case.wavelengths.center).items():
                values[f"{obs.name}.{key}"] = val
        else:
            at = obs.at if obs.at is not None else case.wavelengths.center
            values[obs.name] = float(np.interp(at, wl[order], spectrum[order]))
    if case.custom is not None:
        extra = case.custom(case, {"stage": "observables", "built": built, "values": values,
                                   "spectra": spectra, "wavelengths": wl})
        if isinstance(extra, dict):
            values.update(extra)

    t_observables = time.time() - t_obs0
    checks = {}
    for name, (expected, tol) in case.reference.values.items():
        if name in values:
            checks[name] = {"value": values[name], "expected": expected, "tolerance": tol,
                            "pass": bool(abs(values[name] - expected) <= tol)}
    spectrum_checks = {}
    for name, refs in case.reference.spectra.items():
        if name not in spectra:
            continue
        mine = np.array(spectra[name])
        for label, loader in refs.items():
            data = loader()
            if data is None:
                spectrum_checks[f"{name} vs {label}"] = {"available": False}
                continue
            rw, rv = (np.asarray(a, dtype=float) for a in data)
            o = np.argsort(rw)
            diff = mine - np.interp(wl, rw[o], rv[o])
            spectrum_checks[f"{name} vs {label}"] = {
                "available": True, "max_abs": float(np.max(np.abs(diff))),
                "rms": float(np.sqrt(np.mean(diff ** 2)))}
    nx, ny, nz = built.grid()
    cells = nx * ny * nz
    cpl = options.cells_per_wavelength
    # Reference tables are keyed by cells/wavelength (float) or "res<px/um>".
    key = float(cpl) if cpl else f"res{built.resolution:g}"
    result = {
        "case": case.name,
        "options": dataclasses.asdict(options),
        "resolution": built.resolution,
        "grid": [nx, ny, nz], "cells": cells, "steps": steps,
        "ranks": mp.count_processors(),
        "time_s": {"build": t_build, "init_sim": t_init, "first_step": t_first, "run": t_run,
                   "observables": t_observables, "total": time.time() - t_wall},
        "gcell_steps_per_s": cells * (steps - 1) / t_run / 1e9 if t_run > 0 else None,
        "values": values, "checks": checks, "spectrum_checks": spectrum_checks,
        "wavelengths": wl.tolist(), "spectra": spectra,
        "reference": {
            "source": case.reference.source,
            "key": key,
            "runtimes_s": {k: v.get(key) for k, v in case.reference.runtimes_s.items()},
            "cells": {k: v.get(key) for k, v in case.reference.cells.items()},
            "steps": {k: v.get(key) for k, v in case.reference.steps.items()},
            "notes": case.reference.notes,
        },
        "host": {"hostname": socket.gethostname(), "python": platform.python_version(),
                 "meep": getattr(mp, "__version__", "?")},
    }
    return result


def write(result: Dict[str, Any], path: str) -> None:
    if mp.am_master():
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
