"""Shared stacks, bands and references for the arXiv 2506.16665 device suite."""
from __future__ import annotations

import json
import os
from typing import Dict

from .spec import Layer, Reference, Wavelengths

PAPER_JSON = os.path.join(os.path.dirname(__file__), "..", "..", "nvidia", "paper_2506_16665",
                          "paper_reference.json")

# gdsfactory generic PDK: WG (1,0) full etch 220 nm, SLAB150 (2,0) 150 nm.
SOI = {(1, 0): Layer(0.0, 0.22, "Si"), (2, 0): Layer(0.0, 0.15, "Si")}
SOI_Z = (-2.0, 2.0)

# Paper: 1.55 um center, 20 nm runtime bandwidth; 5 nm monitor sampling.
PAPER_BAND = Wavelengths(start=1.54, stop=1.56, step=0.005, center=1.55)


def paper_reference(device: str, values: Dict[str, tuple], notes: str = "") -> Reference:
    """Paper runtimes (s) and 25-cell workloads for `device`; values are the
    paper's converged observables with our tolerances."""
    with open(PAPER_JSON) as f:
        data = json.load(f)["devices"][device]
    runtimes: Dict[str, Dict[float, float]] = {"tidy3d": {}, "lumerical_cpu": {},
                                                "lumerical_gpu": {}}
    for cpl, row in data["timings"].items():
        runtimes["tidy3d"][float(cpl)] = row.get("tidy3d_s")
        runtimes["lumerical_cpu"][float(cpl)] = row.get("lumerical_local_s")
        runtimes["lumerical_gpu"][float(cpl)] = row.get("lumerical_cloud_s")
    work = data.get("workload_25", {})
    cells = {k: {25.0: v["grid_points"]} for k, v in work.items()}
    steps = {k: {25.0: v["iterations"]} for k, v in work.items()}
    return Reference(source="arXiv 2506.16665 (Liu & Poon), " + data["display_name"],
                     values=values, runtimes_s=runtimes, cells=cells, steps=steps,
                     notes=notes or "Tidy3D times are cloud wall times (non-monotonic with load).")


def paper_domain(device: str):
    with open(PAPER_JSON) as f:
        sx, sy, _ = json.load(f)["devices"][device]["domain_um"]
    return (sx, sy)
