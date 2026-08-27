# Copyright (C) 2005-2026 Massachusetts Institute of Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.

"""Drive the whole bitwise matrix and collect one combined manifest.

Each configuration runs in a subprocess so that a crash or an abort in one does
not take the matrix with it, and so that MPI rank counts can vary per run.

Usage:
    python run_matrix.py --out manifest.json [--ranks 1 2] [--threads 1 4]
                         [--only NAME ...] [--jobs N]
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import configs  # noqa: E402

DEFAULT_RANKS = [1, 2]
DEFAULT_THREADS = [1]


def _run_one(name, ranks, threads, python, mpirun, keep_dir):
    """Run one (config, ranks, threads) cell; return (key, manifest|error)."""
    key = f"{name}|np{ranks}|omp{threads}"
    workdir = keep_dir or tempfile.mkdtemp(prefix="meep-bitwise-")
    if keep_dir:
        workdir = os.path.join(keep_dir, key.replace("|", "_"))
        os.makedirs(workdir, exist_ok=True)
    out = os.path.join(workdir, "manifest.json")

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    # The BLAS thread pool is a separate knob from OMP_NUM_THREADS and is a
    # real source of run-to-run variation: MPB's eigensolver runs BLAS dot
    # products, and a multithreaded OpenBLAS sums them in nondeterministic
    # order. Measured on this tree, four identical runs of
    # test_material_grid.test_symmetry produced two distinct transmittances
    # (29.4686628169574 x3, 29.46866280588633 x1); pinning the BLAS threads
    # made all four identical. Section 4 of the plan does not list this pin.
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"] = str(threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(threads)
    # Keep the check in its historical position so the matrix measures the
    # default configuration; MEEP_FINITE_CHECK is exercised by its own test.
    env.setdefault("MEEP_FINITE_CHECK", "step")

    cmd = []
    if ranks > 1:
        cmd += [mpirun, "-np", str(ranks), "--oversubscribe"]
    cmd += [python, os.path.join(HERE, "dump_state.py"), name, "--workdir", workdir, "--out", out]

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            return key, dict(error=f"exit {proc.returncode}", stderr=proc.stderr[-4000:])
        with open(out) as f:
            return key, json.load(f)["entries"]
    except Exception as exc:  # noqa: BLE001
        return key, dict(error=repr(exc))
    finally:
        if not keep_dir:
            shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    ap.add_argument("--threads", type=int, nargs="+", default=DEFAULT_THREADS)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--keep-dir", default=None, help="keep per-cell workdirs here")
    args = ap.parse_args()

    names = args.only if args.only else [c["name"] for c in configs.CONFIGS]
    unknown = [n for n in names if n not in configs.CONFIG_BY_NAME]
    if unknown:
        raise SystemExit(f"unknown configs: {unknown}")

    cells = [
        (n, r, t) for n in names for r in args.ranks for t in args.threads
    ]
    print(f"running {len(cells)} cells with {args.jobs} workers", flush=True)

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {
            ex.submit(_run_one, n, r, t, args.python, args.mpirun, args.keep_dir): (n, r, t)
            for (n, r, t) in cells
        }
        for fut in concurrent.futures.as_completed(futures):
            key, val = fut.result()
            results[key] = val
            status = "ERROR" if isinstance(val, dict) and "error" in val else f"{len(val)} arrays"
            print(f"  {key}: {status}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)

    errors = [k for k, v in results.items() if isinstance(v, dict) and "error" in v]
    if errors:
        print(f"\n{len(errors)} cell(s) failed to run:", file=sys.stderr)
        for k in errors:
            print(f"  {k}: {results[k]['error']}", file=sys.stderr)
            if "stderr" in results[k]:
                print(results[k]["stderr"][-1500:], file=sys.stderr)
        return 1
    print(f"\nwrote {args.out} ({len(results)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
