"""Did two runs of the same evaluation agree?

Rule fixed on 2026-10-14, before any chunked policy had been evaluated: every evaluation that is
reported is run twice, in separate processes, and counts as reproduced only if all episodes agree
on the number of steps and on success, and on start, max and final coverage to within TOLERANCE.
A larger discrepancy is reported with both runs, and a difference between policies smaller than
the discrepancy is to be called unresolved.

Why a tolerance: gym-pusht's coverage occasionally differs in its last bit (about 5.6e-17)
between runs whose simulated states and actions are identical. That is far below TOLERANCE;
genuine divergence is orders of magnitude larger.

A NaN in either run counts as a difference, and it shows up as NaN in the largest-difference
figures instead of being hidden by them.

Run:  python compare_eval_runs.py reports/eval_bc_cnn.json reports/rerun_bc_cnn.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import compare_policies as cp

TOLERANCE = 1e-6
FLOAT_KEYS = ("start_coverage", "max_coverage", "final_coverage")


def compare_runs(a: dict, b: dict, tolerance: float = TOLERANCE) -> dict:
    """Episode-by-episode comparison of two evaluation reports of the same policy."""
    if not tolerance > 0:
        raise ValueError("tolerance must be positive")
    cp.check_comparable(a, b)
    by_a = {e["seed"]: e for e in a["episodes"]}
    by_b = {e["seed"]: e for e in b["episodes"]}
    seeds = sorted(by_a)
    max_diff = {key: 0.0 for key in FLOAT_KEYS}
    steps_mismatches = success_mismatches = missing_mismatches = 0
    beyond = any_exact = 0
    flagged = []
    for seed in seeds:
        ea, eb = by_a[seed], by_b[seed]
        bad = False
        exact = False
        if ea["steps"] != eb["steps"]:
            steps_mismatches += 1
            bad = True
        if bool(ea["success"]) != bool(eb["success"]):
            success_mismatches += 1
            bad = True
        for key in FLOAT_KEYS:
            x, y = ea.get(key), eb.get(key)
            if x is None and y is None:
                continue
            if x is None or y is None:
                missing_mismatches += 1
                bad = True
                continue
            gap = abs(float(x) - float(y))
            if not gap <= tolerance:
                bad = True
            if gap > 0 or math.isnan(gap):
                exact = True
            if math.isnan(gap) or gap > max_diff[key]:
                max_diff[key] = gap  # once NaN, it stays NaN: gap > nan is never true
        beyond += bad
        any_exact += exact
        if bad:
            flagged.append(int(seed))

    def mean_of(report, key):
        values = [e[key] for e in report["episodes"] if e.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "policy": a["policy"],
        "episodes": len(seeds),
        "seed_start": int(seeds[0]),
        "seed_end": int(seeds[-1]),
        "tolerance": tolerance,
        "reproduced": bool(
            steps_mismatches == 0
            and success_mismatches == 0
            and missing_mismatches == 0
            and beyond == 0
        ),
        "steps_mismatches": steps_mismatches,
        "success_mismatches": success_mismatches,
        "missing_value_mismatches": missing_mismatches,
        "episodes_beyond_tolerance": int(beyond),
        "episodes_with_any_exact_difference": int(any_exact),
        "max_abs_difference": max_diff,
        "flagged_seeds": flagged[:10],
        "mean_max_coverage": [mean_of(a, "max_coverage"), mean_of(b, "max_coverage")],
        "mean_final_coverage": [
            mean_of(a, "final_coverage"),
            mean_of(b, "final_coverage"),
        ],
    }


def _pair(values) -> str:
    return " vs ".join("n/a" if v is None else f"{v:.6f}" for v in values)


def format_report(result: dict) -> str:
    d = result["max_abs_difference"]
    verdict = "REPRODUCED" if result["reproduced"] else "NOT REPRODUCED"
    lines = [
        f"reproducibility of {result['policy']}: {result['episodes']} episodes (seeds "
        f"{result['seed_start']} to {result['seed_end']}), tolerance {result['tolerance']:g}",
        f"episodes whose number of steps differs: {result['steps_mismatches']}; whose success "
        f"differs: {result['success_mismatches']}; with a value missing in one run: "
        f"{result['missing_value_mismatches']}",
        f"largest differences: start coverage {d['start_coverage']:.3g}, max coverage "
        f"{d['max_coverage']:.3g}, final coverage {d['final_coverage']:.3g}",
        f"episodes with any difference at all: {result['episodes_with_any_exact_difference']}; "
        f"episodes beyond the tolerance: {result['episodes_beyond_tolerance']}",
        f"mean max coverage, run 1 vs run 2: {_pair(result['mean_max_coverage'])}; mean final "
        f"coverage: {_pair(result['mean_final_coverage'])}",
    ]
    if result["flagged_seeds"]:
        lines.append(f"first seeds beyond the tolerance: {result['flagged_seeds']}")
    lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Did two runs of the same evaluation agree?"
    )
    parser.add_argument("report_a", type=Path)
    parser.add_argument("report_b", type=Path)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        a, b = cp.load_report(args.report_a), cp.load_report(args.report_b)
        result = compare_runs(a, b, args.tolerance)
    except cp.ComparisonError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    out = args.out or Path("reports") / f"reproducibility_{result['policy']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(format_report(result))
    print(f"saved {out}")
    return 0 if result["reproduced"] else 1


if __name__ == "__main__":
    sys.exit(main())
