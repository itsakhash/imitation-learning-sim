"""Pre-registered executed-targets study: does executing fewer of the predicted targets reduce the
coverage the chunked networks give back?

Background. The trace analysis (trace_analysis.py) found that the chunked networks give back more
coverage than the one-step baseline: 0.1026 against 0.0314 of mean coverage lost between the peak
and the end, over ten networks each. Every chunked result so far executed 8 of the 16 predicted
targets before predicting again. One untested explanation is open-loop execution: carrying out 8
targets without looking lets the agent push the block away without noticing. This study tests the
part of that explanation that needs no training: the same ten chunked checkpoints are evaluated
again executing 4, 2 and 1 of their 16 predicted targets per prediction (the original 8 are the
existing evaluations).

Design fixed before any of the new evaluations existed:
  - networks: the ten chunked checkpoints of the seed study (training seeds 0 to 9), unchanged
  - conditions: 1, 2, 4 and 8 targets executed per prediction; the 8 are the existing evaluations
  - everything else identical to the seed study: the 50-episode evaluation on seeds 10000 to 10049
    and the two-run reproducibility check; nothing is trained
  - primary: for each of 1, 2 and 4 against 8, the mean over the ten networks of the paired
    per-network difference (fewer executed targets minus 8) in mean coverage given back (max minus
    final) and in mean final coverage, each with a 95% percentile bootstrap interval that resamples
    the networks (10,000 resamples, fixed seed). Verdict per number of executed targets, checked in
    this order:
      "fewer executed targets hurt"  if the giveback interval lies entirely above zero or the
                                     final-coverage interval lies entirely below zero
      "fewer executed targets help"  if the giveback interval lies entirely below zero and the
                                     final-coverage interval lies entirely above zero
      "mixed or no clear effect"     otherwise
    That is three verdicts, not corrected for the number of tests.
  - descriptive only: every condition's mean max coverage, final coverage and giveback, how many
    networks end above the do-nothing level of final coverage, and each condition's difference from
    the ten baseline networks (chunked minus baseline, network-level bootstrap, no verdict rule)
  - no network or condition is added, dropped or re-evaluated after its results are seen; the
    analysis refuses a network without a passing reproducibility record unless told to
    (--allow-unverified)

Reading guide, fixed before any new evaluation existed:
  - "help" for 1 executed target: consistent with open-loop execution accounting for part of the
    extra giveback; not proof, because executing one target is also a different test-time policy
    (the 2 and 4 conditions show how gradual the change is, descriptively)
  - "mixed or no clear effect" for 1 executed target: this study gives the open-loop explanation no
    support
  - "hurt" for 1 executed target: fewer executed targets did worse, the opposite of the explanation

Limits: the checkpoints were trained to be executed 8 targets at a time, so executing fewer is a
test-time change only; every network was evaluated on the same 50 starting positions; the intervals
describe these ten networks and this recipe; three uncorrected verdicts; with fewer networks per
condition than planned the result is a pilot, not the pre-registered study.

Run:  python nexec_study.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import ablation_study as ab
import compare_policies as cp
import evaluate as ev
import seed_study as ss

REFERENCE = 8
N_EXECS = (1, 2, 4, REFERENCE)
NETWORKS = 10
PAIRED_METRICS = ("mean_giveback", "mean_final_coverage", "mean_max_coverage")


def network_name(seed: int, n_exec: int) -> str:
    """The evaluation name of a chunked network executed n_exec targets at a time. The 8-target
    evaluations keep the names of the seed study."""
    base = ab.network_name("chunk", ab.FULL_SIZE, seed)
    return base if n_exec == REFERENCE else f"{base}_x{n_exec}"


def default_paths(report_dir=Path("reports"), n_execs=N_EXECS, networks=NETWORKS) -> dict:
    """{"chunk": {n_exec: [evaluation report paths]}, "baseline": [evaluation report paths]}."""
    report_dir = Path(report_dir)
    chunk = {
        n: [report_dir / f"eval_{network_name(seed, n)}.json" for seed in range(networks)]
        for n in n_execs
    }
    baseline = ab.default_paths(report_dir, sizes=(ab.FULL_SIZE,), full_seeds=networks)["baseline"][ab.FULL_SIZE]
    return {"chunk": chunk, "baseline": baseline}


def check_pairing(cells: dict) -> None:
    """Position i of every condition must be training seed i, so that the differences are
    paired within a network."""
    for n, reports in cells.items():
        for i, report in enumerate(reports):
            expected = network_name(i, n)
            if report["policy"] != expected:
                raise cp.ComparisonError(
                    f"{n} executed targets, position {i}: expected {expected}, got {report['policy']}"
                )


def verdict(giveback_ci, final_ci) -> str:
    """The pre-registered rule, applied to the intervals of (n executed targets minus 8)."""
    if giveback_ci[0] > 0 or final_ci[1] < 0:
        return "fewer executed targets hurt"
    if giveback_ci[1] < 0 and final_ci[0] > 0:
        return "fewer executed targets help"
    return "mixed or no clear effect"


def paired_difference(reference, other) -> dict:
    """The paired per-network differences (other minus reference), their mean, and a 95%
    bootstrap interval for that mean that resamples the networks."""
    d = np.asarray(other, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    low, high = ev.bootstrap_ci(d)
    return {"values": d.tolist(), "difference": float(d.mean()), "ci95": [low, high]}


def cell_summary(reports) -> dict:
    """Descriptive numbers for one condition (one number of executed targets)."""
    reports = list(reports)
    if not reports:
        raise cp.ComparisonError("a condition has no networks")
    values = [ss.network_values(r) for r in reports]
    max_cov = [v["mean_max_coverage"] for v in values]
    final = [v["mean_final_coverage"] for v in values]
    giveback = [m - f for m, f in zip(max_cov, final)]
    return {
        "networks": len(values),
        "names": [v["name"] for v in values],
        "mean_max_coverage": {"values": max_cov, "mean": float(np.mean(max_cov))},
        "mean_final_coverage": {"values": final, "mean": float(np.mean(final))},
        "mean_giveback": {"values": giveback, "mean": float(np.mean(giveback))},
        "above_donothing": ss.above_donothing(values),
    }


def analyze(cells: dict, baseline) -> dict:
    """cells maps each number of executed targets to its evaluation reports (one per training
    seed, in order); baseline holds the evaluation reports of the baseline networks."""
    baseline = list(baseline)
    cells = {int(n): list(reports) for n, reports in cells.items()}
    if REFERENCE not in cells or len(cells) < 2:
        raise cp.ComparisonError(
            f"need the {REFERENCE}-target evaluations and at least one other number of executed targets"
        )
    if len({len(reports) for reports in cells.values()}) != 1:
        raise cp.ComparisonError("every number of executed targets needs the same networks")
    if not baseline:
        raise cp.ComparisonError("no baseline networks given")
    flat = [r for reports in cells.values() for r in reports] + baseline
    ss.check_all_comparable(flat)
    check_pairing(cells)

    summaries = {n: cell_summary(reports) for n, reports in sorted(cells.items())}
    base_values = [ss.network_values(r) for r in baseline]
    base_max = [v["mean_max_coverage"] for v in base_values]
    base_final = [v["mean_final_coverage"] for v in base_values]
    versus = {}
    for n, cell in summaries.items():
        versus[n] = {}
        for metric, base in (("mean_max_coverage", base_max), ("mean_final_coverage", base_final)):
            difference, low, high = ss.bootstrap_difference(base, cell[metric]["values"])
            versus[n][metric] = {"difference": difference, "ci95": [low, high]}

    reference = summaries[REFERENCE]
    primary = {}
    for n, cell in summaries.items():
        if n == REFERENCE:
            continue
        entry = {m: paired_difference(reference[m]["values"], cell[m]["values"]) for m in PAIRED_METRICS}
        entry["verdict"] = verdict(entry["mean_giveback"]["ci95"], entry["mean_final_coverage"]["ci95"])
        primary[n] = entry

    per_cell = len(next(iter(cells.values())))
    warnings = []
    if per_cell < NETWORKS:
        warnings.append(
            f"{per_cell} networks per condition, fewer than the {NETWORKS} planned: a pilot, "
            "not the pre-registered study"
        )
    if len(baseline) < NETWORKS:
        warnings.append(f"{len(baseline)} baseline networks, fewer than the {NETWORKS} planned")
    first = baseline[0]["summary"]
    base_giveback = [m - f for m, f in zip(base_max, base_final)]
    return {
        "networks_per_condition": per_cell,
        "evaluation": {"episodes": first["n_episodes"], "seed_start": first["seed_start"],
                       "seed_end": first["seed_end"]},
        "cells": summaries,
        "baseline": {
            "networks": len(baseline),
            "mean_max_coverage": float(np.mean(base_max)),
            "mean_final_coverage": float(np.mean(base_final)),
            "mean_giveback": float(np.mean(base_giveback)),
        },
        "versus_baseline": versus,
        "primary": primary,
        "warnings": warnings,
    }


def _interval(entry, spec: str = "+.4f") -> str:
    low, high = entry["ci95"]
    return f"{entry['difference']:{spec}} [{low:{spec}}, {high:{spec}}]"


def format_study(result: dict) -> str:
    e = result["evaluation"]
    n = result["networks_per_condition"]
    lines = [
        f"executed-targets study: {n} chunked networks (training seeds 0 to {n - 1}), each evaluated on "
        f"{e['episodes']} episodes (seeds {e['seed_start']} to {e['seed_end']}) executing different numbers of "
        "their 16 predicted targets per prediction; the unit is the network",
        f"  {'executed':>8}{'max cov.':>10}{'final cov.':>12}{'given back':>12}{'above do-nothing':>20}   "
        f"{'max minus baseline [95% CI]':<32}{'final minus baseline [95% CI]':<32}",
    ]
    for k, cell in sorted(result["cells"].items()):
        vb = result["versus_baseline"][k]
        above = f"{cell['above_donothing']['above']} of {cell['networks']}"
        lines.append(
            f"  {k:>8}{cell['mean_max_coverage']['mean']:>10.4f}{cell['mean_final_coverage']['mean']:>12.4f}"
            f"{cell['mean_giveback']['mean']:>12.4f}{above:>20}   "
            f"{_interval(vb['mean_max_coverage']):<32}{_interval(vb['mean_final_coverage']):<32}"
        )
    b = result["baseline"]
    lines.append(
        f"  baseline networks ({b['networks']}): mean max coverage {b['mean_max_coverage']:.4f}, mean final "
        f"coverage {b['mean_final_coverage']:.4f}, mean coverage given back {b['mean_giveback']:.4f}"
    )
    for k, p in sorted(result["primary"].items()):
        lines.append(
            f"{k} executed targets minus {REFERENCE}: coverage given back {_interval(p['mean_giveback'])}, "
            f"final coverage {_interval(p['mean_final_coverage'])}, max coverage {_interval(p['mean_max_coverage'])} "
            f"-> {p['verdict']}"
        )
    for warning in result["warnings"]:
        lines.append(f"WARNING: {warning}")
    lines.append(
        "note: three uncorrected verdicts; the checkpoints were trained to be executed 8 targets at a time, so "
        "executing fewer is a test-time change only; every network was evaluated on the same 50 starting positions"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pre-registered executed-targets study")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--n-execs", type=int, nargs="+", default=list(N_EXECS))
    parser.add_argument("--networks", type=int, default=NETWORKS)
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/nexec_study.json"))
    args = parser.parse_args(argv)

    try:
        if REFERENCE not in args.n_execs or len(set(args.n_execs)) < 2 or len(set(args.n_execs)) != len(args.n_execs):
            raise ValueError(
                f"--n-execs must include {REFERENCE}, at least one other number, and no repeats"
            )
        paths = default_paths(args.report_dir, args.n_execs, args.networks)
        cells = {n: [cp.load_report(p) for p in ps] for n, ps in paths["chunk"].items()}
        baseline = [cp.load_report(p) for p in paths["baseline"]]
        everything = [r for reports in cells.values() for r in reports] + baseline
        unverified = [
            r["policy"] for r in everything if ss.reproducibility_status(r, args.report_dir) is not True
        ]
        if unverified and not args.allow_unverified:
            raise cp.ComparisonError(
                "no passing reproducibility record for: " + ", ".join(unverified)
                + " (use --allow-unverified to analyse them anyway)"
            )
        result = analyze(cells, baseline)
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
