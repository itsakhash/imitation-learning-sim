"""Pre-registered seed study: does chunking help once training-seed noise is accounted for?

Background. The three-seed comparison (chunk_vs_baseline.py) found its criterion not met, but
training-seed noise was as large as the effect: with three networks per group a method effect
cannot be told from the spread between training runs. This study uses one network per training
seed (seeds 0 to 9, ten per method) and treats the NETWORK as the unit.

Analysis fixed before any of the new networks (seeds 3 to 9) had been trained:
  - every network is evaluated with the standard protocol (50 episodes, seeds 10000 to 10049) and
    must have a passing reproducibility record (two runs, tolerance 1e-6)
  - primary: for each of mean max coverage and mean final coverage, the difference between the
    group means (chunked minus baseline), with a 95% percentile bootstrap interval that resamples
    the networks of each group independently (10,000 resamples, fixed seed)
  - verdict: "supported" if both intervals lie entirely above zero; "contradicted" if either lies
    entirely below zero; otherwise "inconclusive"
  - no network is added, dropped or retrained after its results are seen; a network that fails
    its reproducibility check is reported, and the analysis refuses to run without a passing
    record unless told to (--allow-unverified)
  - secondary and descriptive only, with no verdict rule: for each group, how many networks end
    with a higher mean final coverage than the do-nothing level (the mean start coverage), with a
    Wilson 95% interval, and the group's mean of that difference with a bootstrap interval

Limits: ten networks per group is still a small sample; the intervals describe these training
seeds, these 50 starting positions and this training recipe (30 epochs for both methods, chunk
length 16 with 8 executed), nothing else; both groups were trained by the same code on the same
split, and only the training seed differs. With fewer than PLANNED_PER_GROUP networks in a group
the result is a pilot, not the pre-registered study, and is labelled as such.

Run:  python seed_study.py --baseline reports/eval_bc_cnn.json reports/eval_bc_cnn_s{1..9}.json
          --chunk reports/eval_chunk_cnn.json reports/eval_chunk_cnn_s{1..9}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import compare_policies as cp
import evaluate as ev

PLANNED_PER_GROUP = 10
N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CONFIDENCE = 0.95
PRIMARY = ("mean_max_coverage", "mean_final_coverage")


def network_values(report: dict) -> dict:
    """The per-network numbers the study uses, from an evaluation report's summary."""
    summary = report.get("summary") or {}
    keys = PRIMARY + ("mean_start_coverage",)
    for key in keys:
        if summary.get(key) is None:
            raise cp.ComparisonError(
                f"{report.get('policy')}: the summary has no '{key}'"
            )
    return {"name": report["policy"], **{key: float(summary[key]) for key in keys}}


def check_all_comparable(reports) -> None:
    """All reports must come from the same protocol and seeds, under distinct names."""
    reports = list(reports)
    if not reports:
        raise cp.ComparisonError("no reports given")
    for other in reports[1:]:
        cp.check_comparable(reports[0], other)
    names = [r["policy"] for r in reports]
    if len(set(names)) != len(names):
        raise cp.ComparisonError("two reports have the same network name")


def bootstrap_difference(
    a,
    b,
    confidence: float = CONFIDENCE,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
):
    """(mean of b minus mean of a, low, high). Each group is resampled independently."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or len(a) < 2 or len(b) < 2:
        raise ValueError("each group needs at least two values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_resamples, len(a)))
    ib = rng.integers(0, len(b), size=(n_resamples, len(b)))
    diffs = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(diffs, [alpha, 1.0 - alpha])
    return float(b.mean() - a.mean()), float(low), float(high)


def verdict(intervals: dict) -> str:
    """The pre-registered rule. intervals maps each primary metric to (low, high)."""
    if not intervals:
        raise ValueError("no intervals given")
    if any(high < 0 for low, high in intervals.values()):
        return "contradicted"
    if all(low > 0 for low, high in intervals.values()):
        return "supported"
    return "inconclusive"


def above_donothing(values) -> dict:
    """Secondary, descriptive: how many networks end above the do-nothing level of final
    coverage (their own mean start coverage), and the mean of that difference."""
    diffs = np.array(
        [v["mean_final_coverage"] - v["mean_start_coverage"] for v in values]
    )
    count, n = int((diffs > 0).sum()), len(diffs)
    wilson_low, wilson_high = ev.wilson_ci(count, n)
    boot_low, boot_high = ev.bootstrap_ci(diffs)
    return {
        "networks": n,
        "above": count,
        "above_wilson95": [wilson_low, wilson_high],
        "mean_difference": float(diffs.mean()),
        "mean_difference_ci95": [boot_low, boot_high],
    }


def analyze(baselines, chunks) -> dict:
    baselines, chunks = list(baselines), list(chunks)
    check_all_comparable(baselines + chunks)
    base = [network_values(r) for r in baselines]
    chunk = [network_values(r) for r in chunks]
    first = baselines[0]["summary"]
    primary = {}
    for metric in PRIMARY:
        a, b = [v[metric] for v in base], [v[metric] for v in chunk]
        difference, low, high = bootstrap_difference(a, b)
        primary[metric] = {
            "baseline_values": a,
            "chunk_values": b,
            "baseline_mean": float(np.mean(a)),
            "chunk_mean": float(np.mean(b)),
            "baseline_range": float(max(a) - min(a)),
            "chunk_range": float(max(b) - min(b)),
            "difference": difference,
            "ci95": [low, high],
        }
    warnings = []
    if min(len(base), len(chunk)) < PLANNED_PER_GROUP:
        warnings.append(
            f"fewer than {PLANNED_PER_GROUP} networks in a group: this is a pilot, "
            "not the pre-registered study"
        )
    return {
        "n_baseline": len(base),
        "n_chunk": len(chunk),
        "evaluation": {
            "episodes": first["n_episodes"],
            "seed_start": first["seed_start"],
            "seed_end": first["seed_end"],
        },
        "baseline_networks": [v["name"] for v in base],
        "chunk_networks": [v["name"] for v in chunk],
        "primary": primary,
        "verdict": verdict({m: tuple(primary[m]["ci95"]) for m in PRIMARY}),
        "secondary": {
            "baseline": above_donothing(base),
            "chunk": above_donothing(chunk),
        },
        "warnings": warnings,
    }


def reproducibility_status(report: dict, report_dir) -> bool | None:
    """True or False from the network's reproducibility record, None if there is none."""
    path = Path(report_dir) / f"reproducibility_{report['policy']}.json"
    if not path.exists():
        return None
    return bool(json.loads(path.read_text()).get("reproduced"))


def format_study(result: dict) -> str:
    e = result["evaluation"]
    lines = [
        f"seed study: {result['n_baseline']} baseline and {result['n_chunk']} chunked networks, each "
        f"evaluated on {e['episodes']} episodes (seeds {e['seed_start']} to {e['seed_end']})"
    ]
    for metric in PRIMARY:
        p = result["primary"][metric]
        low, high = p["ci95"]
        lines.append(
            f"{metric}: baseline mean {p['baseline_mean']:.4f} (range {p['baseline_range']:.4f}), "
            f"chunked mean {p['chunk_mean']:.4f} (range {p['chunk_range']:.4f}), chunked minus baseline "
            f"{p['difference']:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}]"
        )
        lines.append(
            "  baseline networks: "
            + ", ".join(f"{v:.4f}" for v in p["baseline_values"])
        )
        lines.append(
            "  chunked networks:  " + ", ".join(f"{v:.4f}" for v in p["chunk_values"])
        )
    lines.append(f"verdict: {result['verdict']}")
    for kind in ("baseline", "chunk"):
        s = result["secondary"][kind]
        wl, wh = s["above_wilson95"]
        ml, mh = s["mean_difference_ci95"]
        lines.append(
            f"{kind}: {s['above']} of {s['networks']} networks end above the do-nothing level of final "
            f"coverage (Wilson 95% [{100 * wl:.0f}%, {100 * wh:.0f}%]); mean difference from that level "
            f"{s['mean_difference']:+.4f} (95% CI [{ml:+.4f}, {mh:+.4f}])"
        )
    for warning in result["warnings"]:
        lines.append(f"WARNING: {warning}")
    lines.append(
        "note: the intervals describe these training seeds, these 50 starting positions and this "
        "recipe; the secondary lines are descriptive and carry no verdict"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-registered chunked-vs-baseline seed study"
    )
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--chunk", type=Path, nargs="+", required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/seed_study.json"))
    args = parser.parse_args(argv)

    try:
        baselines = [cp.load_report(p) for p in args.baseline]
        chunks = [cp.load_report(p) for p in args.chunk]
        unverified = [
            r["policy"]
            for r in baselines + chunks
            if reproducibility_status(r, args.report_dir) is not True
        ]
        if unverified and not args.allow_unverified:
            raise cp.ComparisonError(
                "no passing reproducibility record for: "
                + ", ".join(unverified)
                + " (use --allow-unverified to analyse them anyway)"
            )
        result = analyze(baselines, chunks)
    except (cp.ComparisonError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    result["unverified_networks"] = unverified
    if unverified:
        result["warnings"].append(
            "without a passing reproducibility record: " + ", ".join(unverified)
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(format_study(result))
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
