"""meepbench CLI.

  python -m meepbench list
  python -m meepbench run CASE [--cpl 10 | --res 30] [--backend nvidia|cpu]
                              [--precision native|mixed|f32] [--materials const|dispersive]
                              [--avg | --no-avg] [--until T] [--out results/]
  python -m meepbench run layout.gds --cell top --source o1:TE0 --observe o3:TE0 ...
  python -m meepbench report results/*.json

CASE is a name in cases/, a .py file defining CASE, or a GDS file (ad-hoc case
from --cell/--source/--observe/--layer). Run under mpirun for multiple ranks.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from . import cases as registry
from . import report


def _adhoc_gds(a):
    from .spec import Case, Geometry, Layer, ModeSource, Observable, Wavelengths
    stack = {}
    for spec in a.layer or ["1/0:0:0.22:Si"]:
        key, zmin, zmax, material = spec.split(":")
        layer, datatype = key.split("/")
        stack[(int(layer), int(datatype))] = Layer(float(zmin), float(zmax), material)
    port, _, mode = (a.source or "o1:TE0").partition(":")
    observables = []
    for spec in a.observe or []:
        p, _, m = spec.partition(":")
        observables.append(Observable(f"{p}_{m or 'TE0'}", "mode_transmission", p, m or "TE0"))
    lo, hi, step = a.band
    return Case(name=os.path.splitext(os.path.basename(a.case))[0],
                description=f"ad-hoc GDS case {a.case}:{a.cell}",
                geometry=Geometry(gds=a.case, cell=a.cell), stack=stack,
                background=a.background, wavelengths=Wavelengths(lo, hi, step, (lo + hi) / 2),
                source=ModeSource(port, mode or "TE0"), observables=observables)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="meepbench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list built-in cases")
    r = sub.add_parser("run", help="run one case")
    r.add_argument("case")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--cpl", type=float, help="cells per wavelength in n_max at lambda_min")
    g.add_argument("--res", type=float, help="resolution, px/um")
    r.add_argument("--backend", default="nvidia", choices=["nvidia", "cpu"])
    r.add_argument("--precision", default="native", choices=["native", "mixed", "f32"])
    r.add_argument("--materials", default="const", choices=["const", "dispersive"])
    r.add_argument("--avg", dest="avg", action="store_true", default=None,
                   help="subpixel averaging on (default: the case's setting)")
    r.add_argument("--no-avg", dest="avg", action="store_false")
    r.add_argument("--until", type=float, help="fixed run time (overrides the case)")
    r.add_argument("--stop", choices=["dft", "energy"], help="override the case's stop rule")
    r.add_argument("--max-edge", type=float, default=None,
                   help="polygon edge split length in um (0 disables; default: the case's)")
    r.add_argument("--verbosity", type=int, default=0)
    r.add_argument("--out", default="results")
    r.add_argument("--tag", default="", help="suffix for the result file name")
    # ad-hoc GDS cases
    r.add_argument("--cell")
    r.add_argument("--source", help="port[:mode], e.g. o1:TE0")
    r.add_argument("--observe", action="append", help="port[:mode]; repeatable")
    r.add_argument("--layer", action="append", help="L/D:zmin:zmax:material; repeatable")
    r.add_argument("--background", default="SiO2")
    r.add_argument("--band", type=float, nargs=3, default=(1.5, 1.6, 0.01),
                   metavar=("LO", "HI", "STEP"))
    pl = sub.add_parser("plot", help="PNG of the eps cross-section with sources/monitors")
    pl.add_argument("case")
    pl.add_argument("--res", type=float, default=10)
    pl.add_argument("--z", type=float, help="slice height (layout coords; default: stack middle)")
    pl.add_argument("--out", default="results")
    rp = sub.add_parser("report", help="markdown table from result JSON files")
    rp.add_argument("results", nargs="+")
    a = p.parse_args(argv)

    if a.cmd == "list":
        for name, case in registry.all_cases().items():
            print(f"{name:16s} [{','.join(case.tags)}] {case.description}")
        return 0
    if a.cmd == "report":
        print(report.markdown(a.results))
        return 0

    from . import runner
    import meep as mp
    if a.cmd == "plot":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        case = registry.load(a.case)
        built = runner.Built(case, runner.Options(resolution=a.res, backend="cpu"))
        z = (a.z if a.z is not None else built._port_z()) - built.center[2]
        fig, ax = plt.subplots(figsize=(10, 5))
        built.sim.plot2D(ax=ax, output_plane=mp.Volume(
            center=mp.Vector3(0, 0, z), size=mp.Vector3(built.sim.cell_size.x,
                                                         built.sim.cell_size.y, 0)))
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, f"{case.name}_xy.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(path)
        return 0
    case = _adhoc_gds(a) if a.case.endswith(".gds") else registry.load(a.case)
    if a.max_edge is not None:
        import dataclasses
        case = dataclasses.replace(case, geometry=dataclasses.replace(
            case.geometry, max_edge=a.max_edge or None))
    options = runner.Options(cells_per_wavelength=a.cpl, resolution=a.res, backend=a.backend,
                             precision=a.precision, materials=a.materials, eps_averaging=a.avg,
                             until=a.until, stop=a.stop, verbosity=a.verbosity)
    result = runner.run(case, options)
    if mp.am_master():
        os.makedirs(a.out, exist_ok=True)
        label = f"cpl{a.cpl:g}" if a.cpl else f"res{result['resolution']:g}"
        name = (f"{case.name}_{label}_{a.backend}_{a.precision}_{a.materials}_np{result['ranks']}"
                f"{'_' + a.tag if a.tag else ''}.json")
        path = os.path.join(a.out, name)
        runner.write(result, path)
        print(report.markdown([path]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
