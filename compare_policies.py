"""Paired comparison of two evaluation reports on the same seeds.

Both reports must come from evaluate.py / evaluate_bc.py with the same protocol and the
same seeds. For each seed the difference B - A is taken, so the seed-to-seed variation in
where the block starts (identical for both policies) cancels out. The interval is a
bootstrap over the per-seed differences.

These intervals describe these particular starting positions and, for learned policies,
one training run. They are not corrected for the number of comparisons made.

Run:  python compare_policies.py reports/eval_hold.json reports/eval_bc_cnn.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import evaluate as ev

TOLERANCE = 1e-6
METRICS = ("max_coverage", "final_coverage")
PROTOCOL_KEYS = ("eval_seed_start", "max_steps")


class ComparisonError(ValueError):
    """The two reports cannot be compared one seed at a time."""


def load_report(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ComparisonError(f"no such report: {path}")
    report = json.loads(path.read_text())
    for key in ("policy", "protocol", "episodes"):
        if key not in report:
            raise ComparisonError(
                f"{path} has no '{key}' entry; is it an evaluation report?"
            )
    return report


def check_comparable(a: dict, b: dict) -> None:
    for key in PROTOCOL_KEYS:
        if a["protocol"].get(key) != b["protocol"].get(key):
            raise ComparisonError(
                f"the reports use different protocols ({key}: "
                f"{a['protocol'].get(key)} vs {b['protocol'].get(key)})"
            )
    seeds_a = [e["seed"] for e in a["episodes"]]
    seeds_b = [e["seed"] for e in b["episodes"]]
    if not seeds_a or not seeds_b:
        raise ComparisonError("a report contains no episodes")
    if len(set(seeds_a)) != len(seeds_a) or len(set(seeds_b)) != len(seeds_b):
        raise ComparisonError("a report contains duplicate seeds")
    if set(seeds_a) != set(seeds_b):
        raise ComparisonError("the reports were not run on the same seeds")


def paired_values(a: dict, b: dict, metric: str):
    """(seeds, values of A, values of B), aligned by seed and sorted by seed."""
    by_a = {e["seed"]: e for e in a["episodes"]}
    by_b = {e["seed"]: e for e in b["episodes"]}
    seeds = sorted(by_a)
    try:
        va = np.array([by_a[s][metric] for s in seeds], dtype=np.float64)
        vb = np.array([by_b[s][metric] for s in seeds], dtype=np.float64)
    except KeyError as err:
        raise ComparisonError(f"episodes have no '{metric}' entry") from err
    return seeds, va, vb


def summarize_differences(seeds, diffs, top: int = 5) -> dict:
    """Summary of per-seed differences (B - A)."""
    if top < 1:
        raise ValueError("top must be at least 1")
    diffs = np.asarray(diffs, dtype=np.float64)
    order = np.argsort(diffs, kind="stable")  # ascending
    gains = [(seeds[i], float(diffs[i])) for i in order[::-1] if diffs[i] > TOLERANCE][
        :top
    ]
    losses = [(seeds[i], float(diffs[i])) for i in order if diffs[i] < -TOLERANCE][:top]
    positive = diffs[diffs > TOLERANCE]
    concentration = (
        None
        if len(positive) == 0
        else float(np.sort(positive)[::-1][:top].sum() / positive.sum())
    )
    lo, hi = ev.bootstrap_ci(diffs)
    return {
        "n": int(len(diffs)),
        "mean_difference": float(diffs.mean()),
        "ci95": [lo, hi],
        "higher": int((diffs > TOLERANCE).sum()),
        "equal": int((np.abs(diffs) <= TOLERANCE).sum()),
        "lower": int((diffs < -TOLERANCE).sum()),
        "largest_gains": gains,
        "largest_losses": losses,
        "top_k": top,
        "gain_concentration": concentration,
    }


def compare_reports(a: dict, b: dict, metrics=METRICS, top: int = 5) -> dict:
    check_comparable(a, b)
    seeds_all = sorted(e["seed"] for e in a["episodes"])
    result = {
        "a": a["policy"],
        "b": b["policy"],
        "n_episodes": len(seeds_all),
        "seed_start": seeds_all[0],
        "seed_end": seeds_all[-1],
        "difference": "B minus A",
        "metrics": {},
    }
    for metric in metrics:
        seeds, va, vb = paired_values(a, b, metric)
        summary = summarize_differences(seeds, vb - va, top)
        summary["mean_a"] = float(va.mean())
        summary["mean_b"] = float(vb.mean())
        result["metrics"][metric] = summary
    return result


def format_comparison(result: dict) -> str:
    lines = [
        f"B minus A on the same {result['n_episodes']} seeds "
        f"({result['seed_start']} to {result['seed_end']})",
        f"A: {result['a']}   B: {result['b']}",
    ]
    for metric, m in result["metrics"].items():
        lo, hi = m["ci95"]
        lines.append(
            f"{metric}: A {m['mean_a']:.4f}  B {m['mean_b']:.4f}  "
            f"mean difference {m['mean_difference']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"B higher/equal/lower on {m['higher']}/{m['equal']}/{m['lower']} seeds"
        )
        if m["largest_gains"]:
            lines.append(
                "  largest gains for B: "
                + ", ".join(f"seed {s} ({d:+.3f})" for s, d in m["largest_gains"])
            )
        if m["largest_losses"]:
            lines.append(
                "  largest losses for B: "
                + ", ".join(f"seed {s} ({d:+.3f})" for s, d in m["largest_losses"])
            )
        if m["gain_concentration"] is not None:
            lines.append(
                f"  the {m['top_k']} largest gains make up "
                f"{100 * m['gain_concentration']:.0f}% of all gains"
            )
    lines.append(
        "note: intervals are not corrected for multiple comparisons and cover these seeds "
        "(and one training run, for learned policies) only"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Paired comparison of two evaluation reports"
    )
    parser.add_argument("report_a", type=Path)
    parser.add_argument("report_b", type=Path)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        result = compare_reports(
            load_report(args.report_a), load_report(args.report_b), top=args.top
        )
    except ComparisonError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    out = args.out or Path("reports") / f"compare_{result['a']}_vs_{result['b']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(format_comparison(result))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
