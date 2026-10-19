"""Material registry with two fidelity modes.

const:      fixed refractive indices (fastest; what speed comparisons use
            unless a case needs dispersion).
dispersive: one Lorentzian pole per material, fitted over the case's band to
            meep.materials models. One pole per material keeps the
            polarization cost bounded (meep.materials.Si alone has three).
"""
from __future__ import annotations

import functools
from typing import Callable, Dict, Tuple

import numpy as np

import meep as mp

# name -> (constant index, meep.materials reference model or None)
_TABLE: Dict[str, Tuple[float, str]] = {
    # Constant indices follow arXiv 2506.16665's Meep adaptation (Si 3.48,
    # SiO2 eps 2.09) so tier-2 runs match its performance-mode materials.
    "Si": (3.48, "Si"),
    "SiO2": (2.09 ** 0.5, "SiO2"),
    "SiN": (2.0, "Si3N4_NIR"),
    "TiO2": (2.40, ""),
    "air": (1.0, ""),
}


def register(name: str, index: float, model: str = "") -> None:
    """Add a material usable from case stacks (model: meep.materials name)."""
    _TABLE[name] = (index, model)


def _single_pole_fit(model: str, wl_lo: float, wl_hi: float) -> mp.Medium:
    """eps(f) = eps_inf + s f0^2 / (f0^2 - f^2), least squares over the band."""
    import meep.materials as library
    medium = getattr(library, model)
    wl = np.linspace(wl_lo, wl_hi, 41)
    f = 1.0 / wl
    eps = np.array([np.real(medium.epsilon(fi)[0][0]) for fi in f])
    best = None
    for f0 in np.linspace(1.5 * f.max(), 30.0 * f.max(), 200):
        basis = np.column_stack([np.ones_like(f), f0 ** 2 / (f0 ** 2 - f ** 2)])
        coef, *_ = np.linalg.lstsq(basis, eps, rcond=None)
        err = np.linalg.norm(basis @ coef - eps)
        if coef[1] > 0 and (best is None or err < best[0]):
            best = (err, f0, coef)
    _, f0, (eps_inf, sigma) = best
    return mp.Medium(epsilon=eps_inf,
                     E_susceptibilities=[mp.LorentzianSusceptibility(frequency=f0, gamma=0,
                                                                     sigma=sigma)])


@functools.lru_cache(maxsize=None)
def medium(name: str, mode: str = "const", band: Tuple[float, float] = (1.45, 1.65)) -> mp.Medium:
    if name not in _TABLE:
        raise KeyError(f"unknown material {name!r}; known: {sorted(_TABLE)}")
    index, model = _TABLE[name]
    if mode == "const" or not model:
        return mp.Medium(index=index)
    if mode == "dispersive":
        return _single_pole_fit(model, *band)
    raise ValueError(f"material mode must be const or dispersive, not {mode!r}")


def max_index(names, mode: str, wavelength: float) -> float:
    n = 1.0
    for name in names:
        m = medium(name, mode)
        n = max(n, float(np.sqrt(np.real(m.epsilon(1.0 / wavelength)[0][0]))))
    return n
