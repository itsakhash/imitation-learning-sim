"""Analysis of the pre-registered diffusion study (see diffusion_preregistration.md).

One block is ten training seeds: block 1 is seeds 0 to 9, block 2 (the replication) seeds 10 to
19. Blocks are analysed separately and never pooled. The unit is the network.

Verdict (H1 for block 1, H2 for block 2), exactly the seed study's rule and bootstrap: for each
of mean max coverage and mean final coverage, the difference between the group means of arm D
(diffusion) and arm C (the chunked control), D minus C, with a 95% percentile bootstrap interval
that resamples the networks of each group independently (10,000 resamples, seed 0).
"supported" if both intervals lie entirely above zero, "contradicted" if either lies entirely
below zero, otherwise "inconclusive".

Descriptive, no verdict rule (all from the same evaluation reports):
  - D minus B and C minus B, on max coverage, final coverage and coverage given back (max minus
    final), with the same bootstrap
  - for each arm: the number of networks ending above the do-nothing level of final coverage
    (Wilson 95%) and the mean of that difference with a bootstrap interval
  - C against the original 30-epoch chunked networks of the same block, and P against them
    paired by training seed (a paired bootstrap over the networks), where those reports exist
Not computed here: failure categories (failure_categories.py replays episodes; run it
separately on the diffusion reports) and validation curves (reports/training_*.json).

Every network must have a passing reproducibility record; a network without one makes the
analysis refuse to run unless --allow-unverified is given, and it is listed in the output.

    python diffusion_study.py --block 1
    python diffusion_study.py --block 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import arm_train
import compare_policies as cp
import seed_study as ss

BLOCKS = {1: tuple(range(0, 10)), 2: tuple(range(10, 20))}
ARMS = ("D", "C", "B", "P")
REQUIRED = ("D", "C")
METRICS = ("mean_max_coverage", "mean_final_coverage", "mean_giveback")
N_RESAMPLES = ss.N_RESAMPLES
BOOTSTRAP_SEED = ss.BOOTSTRAP_SEED


# ---------------------------------------------------------------- names and loading


def original_name(seed: int) -> str:
    """The original 30-epoch chunked network of a training seed."""
    return "chunk_cnn" if seed == 0 else f"chunk_cnn_s{seed}"


def report_path(report_dir, name: str) -> Path:
    return Path(report_dir) / f"eval_{name}.json"


def load_arm(report_dir, arm: str, seeds) -> list:
    return [cp.load_report(report_path(report_dir, arm_train.default_name(arm, s))) for s in seeds]


def load_originals(report_dir, seeds) -> list:
    """The original chunked reports of these seeds, or [] if any is missing."""
    paths = [report_path(report_dir, original_name(s)) for s in seeds]
    if not all(p.exists() for p in paths):
        return []
    return [cp.load_report(p) for p in paths]


# ---------------------------------------------------------------- statistics


def values(report: dict) -> dict:
    """The numbers of one network: the seed study's, plus coverage given back (max minus final;
    averaged over episodes this is the mean of the per-episode giveback exactly)."""
    v = ss.network_values(report)
    v["mean_giveback"] = v["mean_max_coverage"] - v["mean_final_coverage"]
    return v


def paired_bootstrap(a, b, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED, confidence=0.95):
    """(mean of b minus a, low, high), resampling the pairs (same networks in both)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("need two equally long groups of at least two values")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(diff.mean()), float(low), float(high)


def difference(vals_a: list, vals_b: list, metric: str) -> dict:
    """b minus a on one metric, independent-groups bootstrap."""
    a, b = [v[metric] for v in vals_a], [v[metric] for v in vals_b]
    diff, low, high = ss.bootstrap_difference(a, b, n_resamples=N_RESAMPLES)
    return {"difference": diff, "ci95": [low, high], "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b))}


def arm_summary(vals: list) -> dict:
    out = {"networks": [v["name"] for v in vals]}
    for metric in METRICS:
        xs = [v[metric] for v in vals]
        out[metric] = {"values": xs, "mean": float(np.mean(xs)), "range": float(max(xs) - min(xs))}
    out["do_nothing"] = ss.above_donothing(vals)
    return out


def analyze(reports: dict, originals: list | None = None, seeds=None) -> dict:
    """reports maps an arm letter to its list of evaluation reports (in seed order); D and C are
    required. originals is the list of the original chunked reports of the same seeds."""
    missing = [a for a in REQUIRED if not reports.get(a)]
    if missing:
        raise cp.ComparisonError(f"missing arms: {missing}")
    everything = [r for rs in reports.values() for r in rs] + list(originals or [])
    ss.check_all_comparable(everything)
    sizes = {a: len(rs) for a, rs in reports.items() if rs}
    vals = {a: [values(r) for r in rs] for a, rs in reports.items() if rs}

    h = {}
    for metric in ss.PRIMARY:
        h[metric] = difference(vals["C"], vals["D"], metric)  # D minus C
    result = {
        "seeds": list(seeds) if seeds is not None else None,
        "networks_per_arm": sizes,
        "primary": {"comparison": "D minus C", **{m: h[m] for m in ss.PRIMARY}},
        "verdict": ss.verdict({m: tuple(h[m]["ci95"]) for m in ss.PRIMARY}),
        "arms": {a: arm_summary(v) for a, v in vals.items()},
        "descriptive": {},
        "warnings": [],
    }
    if min(sizes["D"], sizes["C"]) < ss.PLANNED_PER_GROUP:
        result["warnings"].append(
            f"fewer than {ss.PLANNED_PER_GROUP} networks in D or C: this is a pilot, not the pre-registered block"
        )
    if "B" in vals:
        for label, arm in (("D_minus_B", "D"), ("C_minus_B", "C")):
            result["descriptive"][label] = {
                m: difference(vals["B"], vals[arm], m) for m in METRICS
            }
    if originals:
        orig = [values(r) for r in originals]
        result["descriptive"]["C_minus_original_chunked"] = {
            m: difference(orig, vals["C"], m) for m in METRICS
        }
        result["descriptive"]["original_chunked"] = arm_summary(orig)
        if "P" in vals and len(vals["P"]) == len(orig):
            paired = {}
            for m in METRICS:
                d, lo, hi = paired_bootstrap([v[m] for v in orig], [v[m] for v in vals["P"]])
                paired[m] = {"difference": d, "ci95": [lo, hi]}
            result["descriptive"]["P_minus_original_chunked_paired"] = paired
    return result


# ---------------------------------------------------------------- output


def _fmt(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def format_block(result: dict, label: str = "block") -> str:
    lines = [f"diffusion study, {label}: networks per arm {result['networks_per_arm']}"]
    p = result["primary"]
    for m in ss.PRIMARY:
        lines.append(
            f"{m}: C mean {p[m]['a_mean']:.4f}, D mean {p[m]['b_mean']:.4f}, D minus C {_fmt(p[m])}"
        )
    lines.append(f"verdict (D against C): {result['verdict']}")
    lines.append("per arm (mean over networks; range; networks above the do-nothing level of final coverage):")
    for arm, s in result["arms"].items():
        dn = s["do_nothing"]
        lines.append(
            f"  {arm}: max {s['mean_max_coverage']['mean']:.4f}, final {s['mean_final_coverage']['mean']:.4f}, "
            f"given back {s['mean_giveback']['mean']:.4f}; final range {s['mean_final_coverage']['range']:.4f}; "
            f"{dn['above']} of {dn['networks']} above (Wilson 95% [{100 * dn['above_wilson95'][0]:.0f}%, "
            f"{100 * dn['above_wilson95'][1]:.0f}%]); mean difference from that level "
            f"{dn['mean_difference']:+.4f} [{dn['mean_difference_ci95'][0]:+.4f}, {dn['mean_difference_ci95'][1]:+.4f}]"
        )
    names = {
        "D_minus_B": "D minus B (descriptive)",
        "C_minus_B": "C minus B (descriptive)",
        "C_minus_original_chunked": "C minus the original 30-epoch chunked networks (descriptive; confounded by steps and averaging)",
        "P_minus_original_chunked_paired": "P minus the original chunked networks, paired by seed (descriptive)",
    }
    for key, title in names.items():
        block = result["descriptive"].get(key)
        if block:
            lines.append(title + ":")
            for m, d in block.items():
                lines.append(f"  {m}: {_fmt(d)}")
    for w in result["warnings"]:
        lines.append(f"WARNING: {w}")
    lines.append(
        "note: the intervals describe these training seeds, these 50 starting positions and this recipe; "
        "descriptive lines carry no verdict; the blocks are never pooled"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analysis of the pre-registered diffusion study")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--block", type=int, choices=sorted(BLOCKS))
    which.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    seeds = list(BLOCKS[args.block]) if args.block else args.seeds
    label = f"block {args.block}" if args.block else f"seeds {seeds}"
    out = args.out or Path(args.report_dir) / (f"diffusion_study_block{args.block}.json" if args.block else "diffusion_study_custom.json")
    try:
        reports = {arm: [] for arm in ARMS}
        for arm in ARMS:
            paths = [report_path(args.report_dir, arm_train.default_name(arm, s)) for s in seeds]
            if arm in REQUIRED or all(p.exists() for p in paths):
                reports[arm] = load_arm(args.report_dir, arm, seeds)
        originals = load_originals(args.report_dir, seeds)
        everything = [r for rs in reports.values() for r in rs] + originals
        unverified = [r["policy"] for r in everything if ss.reproducibility_status(r, args.report_dir) is not True]
        if unverified and not args.allow_unverified:
            raise cp.ComparisonError(
                "no passing reproducibility record for: " + ", ".join(unverified)
                + " (use --allow-unverified to analyse them anyway)"
            )
        result = analyze(reports, originals, seeds)
    except (cp.ComparisonError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    result["unverified_networks"] = unverified
    if unverified:
        result["warnings"].append("without a passing reproducibility record: " + ", ".join(unverified))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(format_block(result, label))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
