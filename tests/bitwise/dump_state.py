# Copyright (C) 2005-2026 Massachusetts Institute of Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.

"""Run one configuration and emit a manifest of raw-state hashes.

What gets dumped, per section 4 of the plan:

  * ``fields::dump``     -- every ``fields_chunk::f[c][cmp]`` array plus
                            ``f_u``, ``f_w``, ``f_w_prev``, ``f_cond``, and
                            every ``dft_chunk::dft[]`` accumulator.
  * ``structure::dump``  -- ``chi1inv`` and the susceptibility ``sigma``
                            arrays.
  * reductions           -- flux / DFT-norm values, as float64 bit patterns.

These are the real chunk-local arrays, not interpolated slices: Meep's
checkpoint writer already walks exactly the storage the plan wants compared,
which is why it is reused here instead of adding a parallel dumper.

Hashes are SHA-256 over raw little-endian bytes.  Golden hashes are
deliberately *not* checked into the repo -- they are not stable across
compilers, libc, or HDF5 versions.  The harness always compares two trees built
in the same environment.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

import meep as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import configs  # noqa: E402


def _hash_bytes(buf):
    return hashlib.sha256(buf).hexdigest()


def _hash_array(arr):
    arr = np.ascontiguousarray(arr)
    if arr.dtype.byteorder not in ("<", "|", "="):
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    elif arr.dtype.byteorder == "=" and sys.byteorder == "big":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    return _hash_bytes(arr.tobytes()), list(arr.shape), str(arr.dtype)


def _hash_h5(path, prefix, out):
    """SHA-256 every dataset in an HDF5 file, keyed by its path."""
    import h5py

    with h5py.File(path, "r") as f:

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                digest, shape, dtype = _hash_array(obj[()])
                out[f"{prefix}/{name}"] = dict(sha256=digest, shape=shape, dtype=dtype)

        f.visititems(visit)


def _advance(sim, n):
    """fields::advance(n) where available, else n step() calls.

    The fallback is what lets this harness run against a pre-PR-1 tree, and it
    is not a compromise: "advance(n) is bitwise-equal to n consecutive step()
    calls" is precisely the property PR 1 has to prove, so comparing a base
    tree stepping against a head tree advancing tests it end-to-end across the
    whole matrix.
    """
    if hasattr(sim.fields, "advance"):
        sim.fields.advance(n)
    else:
        for _ in range(n):
            sim.fields.step()


def run_config(cfg, workdir, verbose=False):
    """Run one configuration; return an ordered dict of hashes."""
    manifest = {}

    # Pin the RNG for noisy susceptibilities. Must happen before init_sim,
    # since noise is drawn during stepping but the generator is global.
    mp.set_random_seed(configs.RANDOM_SEED)

    sim = configs.build(cfg)
    sim.use_output_directory(workdir)
    sim.init_sim()
    # After init_sim: add_dft_fields only materializes its SWIG object once the
    # fields exist, and only when _evaluate_dft_objects runs -- normally that
    # happens inside Simulation.run(), which we bypass.
    reductions = configs.add_monitors(sim, cfg)
    sim._evaluate_dft_objects()

    struct_path = os.path.join(workdir, "structure.h5")
    sim.dump_structure(struct_path, single_parallel_file=True)
    if mp.am_master():
        _hash_h5(struct_path, "structure", manifest)

    # fields::dump refuses any simulation carrying polarization state, and
    # under MPI that refusal is an MPI_Abort rather than a catchable exception,
    # so the decision has to be made from the config up front.
    raw_dump = not cfg.get("susceptibilities")

    if cfg.get("solver") == "cw":
        # solve_cw runs its own BiCGSTAB loop through fields::step with
        # doing_solve_cw set; it is a separate timestep program and has to be
        # exercised or PR 5 can regress it silently.
        sim.solve_cw(1e-6, 200, 2)
        _dump_fields(sim, workdir, "cw", manifest, raw_dump)
    else:
        done = 0
        for target in cfg["steps"]:
            if target > done:
                _advance(sim, target - done)
                done = target
            _dump_fields(sim, workdir, f"t{target}", manifest, raw_dump)
            if cfg.get("synchronize"):
                sim.fields.synchronize_magnetic_fields()
                _dump_fields(sim, workdir, f"t{target}_sync", manifest, raw_dump)
                sim.fields.restore_magnetic_fields()

    for name, fn in sorted(reductions.items()):
        values = np.asarray(fn(), dtype=np.float64)
        digest, shape, dtype = _hash_array(values)
        manifest[f"reduction/{name}"] = dict(sha256=digest, shape=shape, dtype=dtype)

    if verbose and mp.am_master():
        for k in sorted(manifest):
            print(f"  {k}: {manifest[k]['sha256'][:16]}")
    return manifest


# Every field component Meep can allocate.
_CANDIDATE_COMPONENTS = [
    "Ex", "Ey", "Ez", "Er", "Ep",
    "Hx", "Hy", "Hz", "Hr", "Hp",
    "Dx", "Dy", "Dz", "Dr", "Dp",
    "Bx", "By", "Bz", "Br", "Bp",
]


def _present_components(sim):
    """Components this cell actually has, via the non-throwing grid_volume query.

    Do NOT probe by calling get_array() and catching the failure: meep::abort
    unwinds as a C++ exception through get_array_slice_dimensions *after* it has
    pushed onto the timing stack, so every caught failure leaks one
    was_working_on entry and roughly 30 of them trip
    `was_working_on.size() <= MeepTimingStackSize` in time.cpp. (Pre-existing;
    unrelated to this stack, but easy to walk into.)

    grid_volume::has_field depends only on the cell dimensionality, so this
    resolves identically on every rank -- which matters, because get_array is
    collective.
    """
    gv = sim.fields.gv
    out = []
    for name in _CANDIDATE_COMPONENTS:
        comp = getattr(mp, name, None)
        if comp is not None and gv.has_field(comp):
            out.append((name, comp))
    return out


def _dump_fields(sim, workdir, tag, manifest, raw_dump=True):
    """Dump raw chunk state two ways.

    1. fields::dump -- the real f/f_u/f_w/f_w_prev/f_cond arrays and the DFT
       accumulators, exactly the storage section 4 asks for. This is the strong
       probe, but the checkpoint writer refuses any simulation that carries
       polarization state ("non-null polarization_state in fields::dump"), so
       it is unavailable for the whole dispersion axis.

    2. get_array over every component -- available everywhere, including
       dispersive runs. Weaker, because array_slice interpolates the Yee
       component onto the centered grid before we see it, but a bit difference
       in f has to conspire very hard to survive that averaging.

    Both are recorded so a regression in a dispersive configuration is still
    caught; see README.md, "coverage gaps".
    """
    if raw_dump:
        path = os.path.join(workdir, f"fields-{tag}.h5")
        sim.dump_fields(path, single_parallel_file=True)
        if mp.am_master():
            _hash_h5(path, f"fields/{tag}", manifest)
    else:
        manifest[f"fields/{tag}/__raw_dump"] = dict(
            sha256="skipped-polarization-state", shape=[], dtype="n/a"
        )

    for name, comp in _present_components(sim):
        arr = sim.get_array(component=comp)
        if mp.am_master():
            digest, shape, dtype = _hash_array(arr)
            manifest[f"slice/{tag}/{name}"] = dict(sha256=digest, shape=shape, dtype=dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="configuration name (see configs.CONFIGS)")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True, help="manifest JSON path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.config not in configs.CONFIG_BY_NAME:
        raise SystemExit(f"unknown config {args.config!r}")
    cfg = configs.CONFIG_BY_NAME[args.config]

    os.makedirs(args.workdir, exist_ok=True)
    mp.verbosity(0)
    manifest = run_config(cfg, args.workdir, verbose=args.verbose)

    if mp.am_master():
        with open(args.out, "w") as f:
            json.dump(
                dict(config=args.config, entries=manifest),
                f,
                indent=1,
                sort_keys=True,
            )


if __name__ == "__main__":
    main()
