"""Pre-registered: do the failed episodes of the chunked networks change when they execute fewer targets?

Background. failure_categories.py sorted every failed episode of the 40 default-setting networks into
four categories (plus a frozen flag); the chunked networks executed 8 of their 16 predicted targets.
The executed-targets study evaluated the same chunked checkpoints executing 4 and 1 targets. This
replays the failed episodes of those evaluations with the same categories and compares the profiles.

Design fixed before any 4-target or 1-target episode was categorized:
  - networks: the 20 chunked checkpoints (training seeds 0 to 19) evaluated executing 4 and 1 of their
    16 predicted targets (reports/eval_<name>_x4.json and reports/eval_<name>_x1.json). Their failed
    episodes are replayed and checked against those reports as in failure_categories.py; a replay that
    does not match is excluded, counted, and makes the tool exit with status 1.
  - categories, thresholds and the frozen flag: exactly those of failure_categories.py (imported, not
    copied). The 8-target and baseline categorizations are read from reports/failure_categories.json;
    nothing is recomputed for them.
  - analysis, per block of ten consecutive training seeds, for n = 4 and n = 1:
      paired change: for every chunked network its share at n minus its share at 8 (paired by training
        seed), for each of the four categories and the frozen share; the mean over the networks with a
        95% bootstrap interval that resamples the networks' differences
      versus the baseline: the chunked networks' shares executing n minus the baseline's, as in
        failure_categories.py (failure_categories.analyze_block)
      total variation distance between the chunked and the baseline category shares (group means),
        0.5 times the sum over the four categories of the absolute difference, at n = 8, 4 and 1: a
        point estimate, no interval
  - descriptive only: there is no verdict rule, and the 20 paired and 20 unpaired intervals are not
    corrected for multiple tests

Limits: the categories describe the replayed trajectory, not why the policy acted as it did; the 4- and
1-target evaluations are test-time changes of checkpoints trained to execute 8; every network was
evaluated on the same 50 starting positions.

Run:  python nexec_categories.py            (about 20 minutes: it replays about 2,000 episodes)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import ablation_study as ab
import compare_policies as cp
import determinism_check as dc
import evaluate as ev
import failure_categories as fk

DEFAULT_EXECUTED = 8
SETTINGS = (4, 1)
CHUNK_LENGTH = 16


class CategoryError(Exception):
    """A saved report is missing or cannot be used."""


def load_full(path) -> list:
    """The networks of reports/failure_categories.json (both methods, chunked executing 8)."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise CategoryError(f"{path} does not exist (run failure_categories.py first)") from None
    except json.JSONDecodeError as err:
        raise CategoryError(f"{path} is not valid JSON: {err}") from None
    networks = data.get("networks")
    if not networks:
        raise CategoryError(f"{path} has no networks")
    return networks


def networks_of(networks, method: str, label: str) -> list:
    """The networks of one method in one block of training seeds that have categorized episodes."""
    return [n for n in networks
            if n["method"] == method and fk.block_label(n["seed"]) == label and n["categorized"] > 0]


def paired_rows(default_networks, setting_networks) -> dict:
    """Per row: the mean over networks of (share at the setting minus share at 8), paired by training seed."""
    by_default = {n["seed"]: n for n in default_networks}
    by_setting = {n["seed"]: n for n in setting_networks}
    if set(by_default) != set(by_setting):
        raise CategoryError(
            "the two settings cover different training seeds: "
            f"only at 8 {sorted(set(by_default) - set(by_setting))}, "
            f"only at the other setting {sorted(set(by_setting) - set(by_default))}"
        )
    seeds = sorted(by_default)
    if len(seeds) < 2:
        raise CategoryError("need at least two networks per block")
    rows = {}
    for key in fk.KEYS:
        diffs = np.array([fk.shares(by_setting[s])[key] - fk.shares(by_default[s])[key] for s in seeds])
        low, high = ev.bootstrap_ci(diffs)
        rows[key] = {"mean": float(diffs.mean()), "ci95": [low, high]}
    return rows


def tvd(rows: dict) -> float:
    """Total variation distance between the chunked and the baseline category shares (group means)."""
    return 0.5 * sum(abs(rows[c]["chunk"]["mean"] - rows[c]["baseline"]["mean"]) for c in fk.CATEGORIES)


def analyze(full_networks, by_setting: dict) -> dict:
    """Per block: the baseline shares, and for n = 8 and every setting the shares, the distance to the
    baseline, the comparison with the baseline and (for the settings) the paired change from 8."""
    labels = sorted({fk.block_label(n["seed"]) for n in full_networks},
                    key=lambda s: int(s.split()[1].split("-")[0]))
    out = {}
    for label in labels:
        baseline = networks_of(full_networks, "baseline", label)
        default = networks_of(full_networks, "chunk", label)
        at_default = fk.analyze_block(baseline, default)
        if at_default["rows"] is None:
            raise CategoryError(f"{label}: too few networks for intervals")
        entry = {
            "baseline": {
                "networks": len(baseline),
                "failed_episodes": at_default["failed_episodes"]["baseline"],
                "shares": {k: at_default["rows"][k]["baseline"]["mean"] for k in fk.KEYS},
            },
            DEFAULT_EXECUTED: {
                "networks": len(default),
                "failed_episodes": at_default["failed_episodes"]["chunk"],
                "versus_baseline": at_default["rows"],
                "tvd_to_baseline": tvd(at_default["rows"]),
                "paired_vs_8": None,
            },
        }
        for n, networks in by_setting.items():
            chunk_n = networks_of(networks, "chunk", label)
            analysis = fk.analyze_block(baseline, chunk_n)
            if analysis["rows"] is None:
                raise CategoryError(f"{label}: too few networks executing {n} targets for intervals")
            entry[n] = {
                "networks": len(chunk_n),
                "failed_episodes": analysis["failed_episodes"]["chunk"],
                "versus_baseline": analysis["rows"],
                "tvd_to_baseline": tvd(analysis["rows"]),
                "paired_vs_8": paired_rows(default, chunk_n),
            }
        out[label] = entry
    return out


def _ci(mean: float, low: float, high: float) -> str:
    return f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]"


def _name(key: str) -> str:
    return key if key in fk.CATEGORIES else "frozen at the end (flag)"


def format_summary(result: dict) -> str:
    settings = result["settings"]["executed"]
    replayed = sum(n["categorized"] for networks in result["networks"].values() for n in networks)
    lines = [
        f"failure categories by executed targets: {replayed} failed episodes replayed "
        f"({', '.join(f'{n} targets' for n in settings)}), {result['mismatched_episodes']} replays did not "
        "match their evaluation report"
    ]
    for label, entry in result["blocks"].items():
        lines.append(f"{label}: baseline {entry['baseline']['networks']} networks; chunked executing 8 and "
                     + ", ".join(str(n) for n in settings) + " targets")
        columns = ["baseline", "chunked 8"] + [f"chunked {n}" for n in settings]
        lines.append("  shares of failed episodes")
        lines.append(f"  {'category':<26}" + "".join(f"{c:<12}" for c in columns))
        for key in fk.KEYS:
            values = [entry["baseline"]["shares"][key], entry[DEFAULT_EXECUTED]["versus_baseline"][key]["chunk"]["mean"]]
            values += [entry[n]["versus_baseline"][key]["chunk"]["mean"] for n in settings]
            lines.append(f"  {_name(key):<26}" + "".join(f"{v:<12.3f}" for v in values))
        lines.append("  total variation distance to the baseline profile (four categories): "
                     + ", ".join(f"{entry[n]['tvd_to_baseline']:.3f} at {n}" for n in [DEFAULT_EXECUTED] + list(settings)))
        lines.append("  paired change, n executed minus 8, mean over networks [95% CI]")
        lines.append(f"  {'category':<26}" + "".join(f"{f'{n} minus 8':<30}" for n in settings))
        for key in fk.KEYS:
            cells = []
            for n in settings:
                row = entry[n]["paired_vs_8"][key]
                cells.append(_ci(row["mean"], *row["ci95"]))
            lines.append(f"  {_name(key):<26}" + "".join(f"{c:<30}" for c in cells))
        lines.append("  chunked executing n minus baseline [95% CI]")
        order = [DEFAULT_EXECUTED] + list(settings)
        lines.append(f"  {'category':<26}" + "".join(f"{f'n = {n}':<30}" for n in order))
        for key in fk.KEYS:
            cells = []
            for n in order:
                row = entry[n]["versus_baseline"][key]
                cells.append(_ci(row["difference"], *row["ci95"]))
            lines.append(f"  {_name(key):<26}" + "".join(f"{c:<30}" for c in cells))
    lines.append(
        "note: descriptive, with no verdict rule and no correction for the number of comparisons; the "
        "categories describe the replayed trajectory, not why the policy acted as it did"
    )
    return "\n".join(lines)


def main(argv=None, *, env_factory=ev.make_eval_env, policy_builder=dc.build_policy, log=print) -> int:
    parser = argparse.ArgumentParser(description="Pre-registered failure categories by executed targets")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--settings", type=int, nargs="+", default=list(SETTINGS))
    parser.add_argument("--full", type=Path, default=Path("reports/failure_categories.json"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("reports/nexec_categories.json"))
    args = parser.parse_args(argv)

    by_setting = {n: [] for n in args.settings}
    try:
        if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
            raise ValueError("--seeds must be distinct non-negative integers")
        if (not args.settings or len(set(args.settings)) != len(args.settings)
                or any(n < 1 or n > CHUNK_LENGTH or n == DEFAULT_EXECUTED for n in args.settings)):
            raise ValueError(f"--settings must be distinct numbers of executed targets, from 1 to {CHUNK_LENGTH} "
                             f"and not {DEFAULT_EXECUTED}")
        full = load_full(args.full)
        for n in args.settings:
            for seed in args.seeds:
                name = ab.network_name("chunk", ab.FULL_SIZE, seed)
                report = cp.load_report(args.report_dir / f"eval_{name}_x{n}.json")
                policy = policy_builder("chunk_cnn", args.checkpoint_dir / f"{name}.pt", env_factory, n)
                env = env_factory()
                try:
                    network = fk.categorize_network(env, policy, report)
                finally:
                    env.close()
                network.update(method="chunk", seed=seed, n_exec=n)
                by_setting[n].append(network)
                log(f"{name} executing {n}: {network['categorized']} of {network['failed_in_report']} failed "
                    f"episodes categorized, {len(network['mismatched'])} replays did not match")
        blocks = analyze(full, by_setting)
    except (CategoryError, cp.ComparisonError, OSError, ValueError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    mismatched = sum(len(n["mismatched"]) for networks in by_setting.values() for n in networks)
    result = {
        "settings": {"executed": list(args.settings), "default_executed": DEFAULT_EXECUTED,
                     "seeds": list(args.seeds)},
        "mismatched_episodes": mismatched,
        "networks": {str(n): by_setting[n] for n in args.settings},
        "blocks": blocks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(format_summary(result))
    print(f"saved {args.out}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
