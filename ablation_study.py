"""Pre-registered demonstration-count ablation: does the number of training episodes matter?

Design fixed before any ablation network existed:
  - sizes: 20, 50 and 100 of the 165 training episodes, plus the full 165 (the ten networks per
    method of the seed study, seeds 0 to 9)
  - both methods (one-step baseline and action-chunking network), SMALL_SEEDS = 5 training seeds
    (0 to 4) at each of the three new sizes; training seed s uses subset seed s, so for a given
    seed the smaller subsets are contained in the larger ones, and different seeds see different
    subsets; the validation episodes never change
  - everything else identical to the seed study: 30 epochs (so smaller training sets get fewer
    gradient steps, deliberately; a step-matched variant would be a separate ablation), the same
    50-episode evaluation on seeds 10000 to 10049, and the same two-run reproducibility check
  - primary: for each method and each of mean max coverage and mean final coverage, the difference
    between the full-data cell (165 episodes, 10 networks) and the 20-episode cell (5 networks),
    165 minus 20, with a 95% percentile bootstrap interval that resamples the networks of each
    cell independently (10,000 resamples, fixed seed). Verdict per method and metric: "more data
    helps" if the interval lies entirely above zero, "more data hurts" if entirely below,
    otherwise "no clear effect". That is four verdicts, not corrected for the number of tests.
  - descriptive only: every cell's mean with a network-level bootstrap interval, and how many
    networks end above the do-nothing level of final coverage
  - no network is added, dropped or retrained after its results are seen; the analysis refuses a
    network without a passing reproducibility record unless told to (--allow-unverified)

Limits: five networks per small cell is a small sample; every network was evaluated on the same 50
starting positions; the intervals describe this recipe, these training seeds and these subsets; the
fixed number of epochs is a confound of the size comparison; with fewer networks per cell than
planned the result is a pilot, not the pre-registered ablation.

Run:  python ablation_study.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import compare_policies as cp
import evaluate as ev
import seed_study as ss

FULL_SIZE = 165
SMALLEST = 20
DEFAULT_SIZES = (20, 50, 100, FULL_SIZE)
METHODS = {"baseline": "bc_cnn", "chunk": "chunk_cnn"}
SMALL_SEEDS = 5
FULL_SEEDS = 10
PRIMARY = ss.PRIMARY


def network_name(method: str, size: int, seed: int) -> str:
    """The evaluation name of a network. Full-data networks keep the names of the seed study."""
    base = METHODS[method]
    if size == FULL_SIZE:
        return base if seed == 0 else f"{base}_s{seed}"
    return f"{base}_n{size}_s{seed}"


def default_paths(report_dir=Path("reports"), sizes=DEFAULT_SIZES, small_seeds=SMALL_SEEDS,
                  full_seeds=FULL_SEEDS) -> dict:
    """{method: {size: [evaluation report paths]}} for the planned networks."""
    report_dir = Path(report_dir)
    paths = {}
    for method in METHODS:
        paths[method] = {}
        for size in sizes:
            count = full_seeds if size == FULL_SIZE else small_seeds
            paths[method][size] = [
                report_dir / f"eval_{network_name(method, size, seed)}.json" for seed in range(count)
            ]
    return paths


def effect_verdict(interval) -> str:
    """The pre-registered rule, applied to the interval of (full minus smallest)."""
    low, high = interval
    if low > 0:
        return "more data helps"
    if high < 0:
        return "more data hurts"
    return "no clear effect"


def cell_summary(reports) -> dict:
    """Descriptive numbers for one cell of networks (one method, one training-set size)."""
    reports = list(reports)
    if not reports:
        raise cp.ComparisonError("a cell has no networks")
    values = [ss.network_values(r) for r in reports]
    cell = {"networks": len(values), "names": [v["name"] for v in values]}
    for metric in PRIMARY:
        v = [x[metric] for x in values]
        low, high = ev.bootstrap_ci(v)
        cell[metric] = {"values": v, "mean": float(np.mean(v)), "ci95": [low, high]}
    cell["above_donothing"] = ss.above_donothing(values)
    cell["success_episodes"] = int(sum(r["summary"]["success_count"] for r in reports))
    return cell


def analyze(cells: dict) -> dict:
    """cells maps each method to {training-set size: [evaluation reports]}."""
    if set(cells) != set(METHODS):
        raise cp.ComparisonError(f"need cells for both methods: {sorted(METHODS)}")
    flat = [r for by_size in cells.values() for reports in by_size.values() for r in reports]
    ss.check_all_comparable(flat)
    summaries = {}
    for method in METHODS:
        by_size = cells[method]
        for needed in (SMALLEST, FULL_SIZE):
            if not by_size.get(needed):
                raise cp.ComparisonError(f"{method}: no networks trained on {needed} episodes")
        summaries[method] = {size: cell_summary(reports) for size, reports in sorted(by_size.items())}
    primary, warnings = {}, []
    for method in METHODS:
        primary[method] = {}
        for metric in PRIMARY:
            small = summaries[method][SMALLEST][metric]["values"]
            full = summaries[method][FULL_SIZE][metric]["values"]
            difference, low, high = ss.bootstrap_difference(small, full)
            primary[method][metric] = {
                "small_networks": len(small), "full_networks": len(full),
                "small_mean": float(np.mean(small)), "full_mean": float(np.mean(full)),
                "difference": difference, "ci95": [low, high],
                "verdict": effect_verdict((low, high)),
            }
        for size, cell in summaries[method].items():
            planned = FULL_SEEDS if size == FULL_SIZE else SMALL_SEEDS
            if cell["networks"] < planned:
                warnings.append(
                    f"{method}, {size} episodes: {cell['networks']} networks, fewer than the "
                    f"{planned} planned (a pilot, not the pre-registered ablation)"
                )
    first = flat[0]["summary"]
    return {
        "evaluation": {"episodes": first["n_episodes"], "seed_start": first["seed_start"],
                       "seed_end": first["seed_end"]},
        "cells": summaries,
        "primary": primary,
        "warnings": warnings,
    }


def format_study(result: dict) -> str:
    e = result["evaluation"]
    lines = [
        f"demonstration-count ablation: each network evaluated on {e['episodes']} episodes "
        f"(seeds {e['seed_start']} to {e['seed_end']}); the unit is the network"
    ]
    for method in METHODS:
        lines.append(f"{method} networks:")
        lines.append(
            f"  {'training episodes':>17}{'networks':>10}   {'mean max coverage [95% CI]':<32}"
            f"{'mean final coverage [95% CI]':<32}{'above do-nothing':>18}"
        )
        for size, c in sorted(result["cells"][method].items()):
            mx, fin, above = c["mean_max_coverage"], c["mean_final_coverage"], c["above_donothing"]
            max_text = f"{mx['mean']:.4f} [{mx['ci95'][0]:.4f}, {mx['ci95'][1]:.4f}]"
            fin_text = f"{fin['mean']:.4f} [{fin['ci95'][0]:.4f}, {fin['ci95'][1]:.4f}]"
            lines.append(
                f"  {size:>17}{c['networks']:>10}   {max_text:<32}{fin_text:<32}"
                f"{above['above']:>11} of {c['networks']}"
            )
    for method in METHODS:
        for metric in PRIMARY:
            p = result["primary"][method][metric]
            lo, hi = p["ci95"]
            lines.append(
                f"{method}, {metric}: {FULL_SIZE} episodes {p['full_mean']:.4f} ({p['full_networks']} "
                f"networks), {SMALLEST} episodes {p['small_mean']:.4f} ({p['small_networks']} networks), "
                f"difference {p['difference']:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] -> {p['verdict']}"
            )
    for warning in result["warnings"]:
        lines.append(f"WARNING: {warning}")
    lines.append(
        "note: four uncorrected verdicts; the number of epochs is fixed at 30, so smaller training sets "
        "also get fewer gradient steps; every network was evaluated on the same 50 starting positions"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pre-registered demonstration-count ablation")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--small-seeds", type=int, default=SMALL_SEEDS)
    parser.add_argument("--full-seeds", type=int, default=FULL_SEEDS)
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/ablation_study.json"))
    args = parser.parse_args(argv)

    try:
        if SMALLEST not in args.sizes or FULL_SIZE not in args.sizes:
            raise ValueError(f"--sizes must include {SMALLEST} and {FULL_SIZE}")
        paths = default_paths(args.report_dir, args.sizes, args.small_seeds, args.full_seeds)
        cells = {m: {s: [cp.load_report(p) for p in ps] for s, ps in by_size.items()}
                 for m, by_size in paths.items()}
        unverified = [
            r["policy"] for by_size in cells.values() for reports in by_size.values() for r in reports
            if ss.reproducibility_status(r, args.report_dir) is not True
        ]
        if unverified and not args.allow_unverified:
            raise cp.ComparisonError(
                "no passing reproducibility record for: " + ", ".join(unverified)
                + " (use --allow-unverified to analyse them anyway)"
            )
        result = analyze(cells)
    except (cp.ComparisonError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    result["unverified_networks"] = unverified
    if unverified:
        result["warnings"].append("without a passing reproducibility record: " + ", ".join(unverified))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(format_study(result))
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
