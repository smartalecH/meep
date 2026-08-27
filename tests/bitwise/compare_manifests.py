# Copyright (C) 2005-2026 Massachusetts Institute of Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.

"""Compare two matrix manifests and fail on any difference.

Exit status is 0 only when every cell ran in both trees and every array hash
matches. A missing or extra array is a failure too: Phase 1 must not change
which arrays exist any more than it changes their contents.
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="manifest from the merge base")
    ap.add_argument("head", help="manifest from the PR head")
    ap.add_argument("--max-report", type=int, default=25)
    args = ap.parse_args()

    with open(args.base) as f:
        base = json.load(f)
    with open(args.head) as f:
        head = json.load(f)

    problems = []

    only_base = sorted(set(base) - set(head))
    only_head = sorted(set(head) - set(base))
    for k in only_base:
        problems.append(f"cell missing from head: {k}")
    for k in only_head:
        problems.append(f"cell missing from base: {k}")

    n_arrays = 0
    n_cells = 0
    for cell in sorted(set(base) & set(head)):
        b, h = base[cell], head[cell]
        if "error" in b or "error" in h:
            problems.append(f"{cell}: cell errored (base={'error' in b}, head={'error' in h})")
            continue
        n_cells += 1
        for name in sorted(set(b) - set(h)):
            problems.append(f"{cell}: array missing from head: {name}")
        for name in sorted(set(h) - set(b)):
            problems.append(f"{cell}: array missing from base: {name}")
        for name in sorted(set(b) & set(h)):
            n_arrays += 1
            if b[name]["sha256"] != h[name]["sha256"]:
                problems.append(
                    f"{cell}: {name} differs "
                    f"(shape {b[name]['shape']} vs {h[name]['shape']}, "
                    f"{b[name]['sha256'][:12]} vs {h[name]['sha256'][:12]})"
                )

    if problems:
        print(f"BITWISE NEUTRALITY VIOLATED: {len(problems)} problem(s)", file=sys.stderr)
        for p in problems[: args.max_report]:
            print(f"  {p}", file=sys.stderr)
        if len(problems) > args.max_report:
            print(f"  ... and {len(problems) - args.max_report} more", file=sys.stderr)
        return 1

    print(f"bitwise neutral: {n_cells} cells, {n_arrays} arrays identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
