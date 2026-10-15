"""The pre-registered comparison of the chunked policy with the one-step baseline.

Criterion, fixed on 2026-10-14 before any chunked policy had been evaluated in the simulator:
the chunked policy counts as an improvement only if, for each of REQUIRED_PAIRS training seeds,
its mean final coverage AND its mean max coverage both exceed those of the baseline trained
with the same seed, on the evaluation seeds 10000 to 10049. Anything less is reported as mixed
or no improvement. Start coverage is shared by all policies, so the max-coverage difference
equals the difference in gain over start.

The pairing of training seeds is arbitrary (seed 1 of one network has no special tie to seed 1
of the other), so the per-network means and their spread are printed as well. A "met" verdict
with differences smaller than the baseline's seed-to-seed spread is to be called weak.

Run:  python chunk_vs_baseline.py --baseline reports/eval_bc_cnn.json reports/eval_bc_cnn_s1.json
          reports/eval_bc_cnn_s2.json --chunk reports/eval_chunk_cnn.json
          reports/eval_chunk_cnn_s1.json reports/eval_chunk_cnn_s2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import compare_policies as cp

METRICS = ("max_coverage", "final_coverage")
REQUIRED_PAIRS = 3


def compare_pairs(baselines, chunks) -> list:
    """Paired comparison (chunk minus baseline) for each same-seed pair of reports."""
    if not baselines:
        raise cp.ComparisonError("no reports given")
    if len(baselines) != len(chunks):
        raise cp.ComparisonError("need the same number of baseline and chunk reports")
    return [
        cp.compare_reports(b, c, metrics=METRICS) for b, c in zip(baselines, chunks)
    ]


def pair_improves(result: dict) -> bool:
    return all(result["metrics"][m]["mean_difference"] > cp.TOLERANCE for m in METRICS)


def evaluate_criterion(results) -> dict:
    n = len(results)
    improved = [pair_improves(r) for r in results]
    met = n >= REQUIRED_PAIRS and all(improved)
    if n < REQUIRED_PAIRS:
        verdict = f"not evaluable: the criterion needs {REQUIRED_PAIRS} training-seed pairs, got {n}"
    elif met:
        verdict = "met: both metrics are higher for the chunked policy in every pair"
    else:
        verdict = "not met: mixed or no improvement"
    return {
        "pairs": n,
        "pairs_improved": int(sum(improved)),
        "per_pair": improved,
        "met": met,
        "verdict": verdict,
    }


def group_means(reports) -> dict:
    """Per-network means of both metrics, in the order of the reports."""
    summaries = [r["summary"] for r in reports]
    return {
        "names": [r["policy"] for r in reports],
        "mean_max_coverage": [s["mean_max_coverage"] for s in summaries],
        "mean_final_coverage": [s["mean_final_coverage"] for s in summaries],
    }


def format_comparison(results, criterion, groups) -> str:
    first = results[0]
    lines = [
        f"chunked policy minus one-step baseline, same training seed, on the same "
        f"{first['n_episodes']} evaluation seeds ({first['seed_start']} to {first['seed_end']})"
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"pair {i}: baseline {r['a']}  vs  chunk {r['b']}")
        for metric in METRICS:
            m = r["metrics"][metric]
            lo, hi = m["ci95"]
            lines.append(
                f"  {metric}: baseline {m['mean_a']:.4f}  chunk {m['mean_b']:.4f}  "
                f"difference {m['mean_difference']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                f"chunk higher/equal/lower on {m['higher']}/{m['equal']}/{m['lower']} seeds"
            )
    for kind in ("baseline", "chunk"):
        g = groups[kind]
        for metric in ("mean_max_coverage", "mean_final_coverage"):
            values = g[metric]
            lines.append(
                f"{kind} networks, {metric}: "
                + ", ".join(f"{v:.4f}" for v in values)
                + f"   mean {np.mean(values):.4f}, range {max(values) - min(values):.4f}"
            )
    lines.append(
        f"criterion: {criterion['verdict']} "
        f"({criterion['pairs_improved']} of {criterion['pairs']} pairs improved on both metrics)"
    )
    lines.append(
        "note: one training run per seed, 50 starting positions, intervals not corrected for "
        "multiple comparisons; the pairing of training seeds is arbitrary"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-registered chunked-vs-baseline comparison"
    )
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--chunk", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("reports/chunk_vs_baseline.json")
    )
    args = parser.parse_args(argv)

    try:
        baselines = [cp.load_report(p) for p in args.baseline]
        chunks = [cp.load_report(p) for p in args.chunk]
        results = compare_pairs(baselines, chunks)
    except cp.ComparisonError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    criterion = evaluate_criterion(results)
    groups = {"baseline": group_means(baselines), "chunk": group_means(chunks)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"pairs": results, "criterion": criterion, "groups": groups}, indent=2
        )
        + "\n"
    )
    print(format_comparison(results, criterion, groups))
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
