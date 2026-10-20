"""JSON results -> markdown summary."""
from __future__ import annotations

import json
from typing import Iterable, List


def _fmt(x, spec=".4g"):
    return "-" if x is None else format(x, spec)


def markdown(paths: Iterable[str]) -> str:
    rows: List[str] = [
        "| case | backend/prec | ranks | res (cpl) | cells | steps | init+1st step s | run s | Gcell-steps/s "
        "| ref runtimes s | checks |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    details: List[str] = []
    for path in paths:
        with open(path) as f:
            r = json.load(f)
        o = r["options"]
        checks = r.get("checks", {})
        status = ", ".join(f"{k} {'PASS' if c['pass'] else 'FAIL'} ({c['value']:.4g} vs "
                           f"{c['expected']:.4g}+-{c['tolerance']:.2g})" for k, c in checks.items())
        refs = ", ".join(f"{k} {v:g}" for k, v in r["reference"]["runtimes_s"].items()
                         if v is not None) or "-"
        cpl = o.get("cells_per_wavelength")
        rows.append(
            f"| {r['case']} | {o['backend']}/{o['precision']}/{o['materials']} | {r['ranks']} | "
            f"{r['resolution']:g} ({_fmt(cpl, 'g')}) | {r['cells']:.3g} | {r['steps']} | "
            f"{r['time_s']['init_sim'] + r['time_s'].get('first_step', 0):.1f} | {r['time_s']['run']:.1f} | "
            f"{_fmt(r['gcell_steps_per_s'], '.3g')} | {refs} | {status or '-'} |")
        t = r["time_s"]
        extra = [f"time build {t.get('build', 0):.1f} / init {t['init_sim']:.1f} / first step "
                 f"{t.get('first_step', 0):.1f} / run {t['run']:.1f} / observables "
                 f"{t.get('observables', 0):.1f} / total {t['total']:.1f} s"]
        extra += [f"{k}={v:.5g}" for k, v in r["values"].items() if k not in checks]
        extra += [f"{k}: max {v['max_abs']:.4f} rms {v['rms']:.4f}" if v.get("available")
                  else f"{k}: reference data not found" for k, v in r.get("spectrum_checks", {}).items()]
        if extra:
            details.append(f"- **{r['case']}** ({path}): " + "; ".join(extra))
    return "\n".join(rows + ([""] + details if details else []))
