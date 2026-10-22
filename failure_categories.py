"""Pre-registered categorization of failed episodes: what do the networks do when they fail?

Background. The trace analysis (trace_analysis.py) showed how much coverage the networks lose, not
what the agent does when it fails. This replays every failed episode of the default-setting networks
and sorts it into one of four categories from the replayed trajectory.

Design fixed before any episode was categorized:
  - episodes: every failed episode (success false) of the 40 default-setting networks of the seed
    study and the replication (training seeds 0 to 19 of both methods; the checkpoints execute their
    own number of targets, 8 for the chunked ones), replayed as evaluate.run_episode does. A replay
    that does not match the saved evaluation report (steps, success, start, max and final coverage and
    the coverage trace, as in make_gif.compare_with_report) is excluded, counted, and makes the tool
    exit with status 1.
  - four mutually exclusive categories, decided in this order:
      never moved               the block's position changes by less than MOVED_PIXELS (1 px) and its angle
                                by less than MOVED_RADIANS (0.02 rad) over the whole episode, so the agent
                                never pushed it
      moved without improving   the block moved, and the peak coverage is less than MIN_GAIN (0.02) above
                                the start coverage
      improved then lost        the peak coverage is at least MIN_GAIN above the start coverage and the final
                                coverage is more than BIG_GIVEBACK (0.1) below the peak
      improved and kept         the peak coverage is at least MIN_GAIN above the start coverage and the final
                                coverage is within BIG_GIVEBACK of the peak, but the episode did not succeed
  - one flag, independent of the category: frozen means the agent's position changes by less than
    FROZEN_PIXELS (2 px) in each of the last FROZEN_STEPS (50) steps (in all steps if there are fewer)
  - analysis: per network, the share of its categorized failed episodes in each category and the frozen
    share; per method and per block of BLOCK_SIZE (10) consecutive training seeds, the mean of the networks'
    shares with a 95% bootstrap interval that resamples the networks (10,000 resamples, fixed seed), and
    the difference chunked minus baseline with the interval of seed_study.bootstrap_difference; the pooled
    counts are reported as well. Descriptive only: there is no verdict rule, and the comparisons (five
    shares per block) are not corrected for multiple tests.
  - the thresholds are not tuned on the results

Limits: the categories describe the replayed trajectory, not why the policy acted as it did; the block
pose and agent position come from the environment; every network was evaluated on the same 50 starting
positions; with fewer than ten networks per group in a block the result is a pilot.

Run:  python failure_categories.py            (about 25 minutes: it replays about 2,000 episodes)
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
import make_gif as mg
import seed_study as ss

CATEGORIES = ("never moved", "moved without improving", "improved then lost", "improved and kept")
KEYS = CATEGORIES + ("frozen",)
MOVED_PIXELS = 1.0
MOVED_RADIANS = 0.02
MIN_GAIN = 0.02
BIG_GIVEBACK = 0.1
FROZEN_STEPS = 50
FROZEN_PIXELS = 2.0
BLOCK_SIZE = 10


def block_label(seed: int) -> str:
    """The block of ten consecutive training seeds a seed belongs to."""
    start = BLOCK_SIZE * (seed // BLOCK_SIZE)
    return f"seeds {start}-{start + BLOCK_SIZE - 1}"


def replay_trajectory(env, policy, seed: int, max_steps: int = ev.MAX_STEPS) -> dict:
    """Replay one seeded episode as evaluate.run_episode does, recording the agent's position and
    the block's pose before the first step and after every step."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    obs, reset_info = env.reset(seed=seed)
    start_coverage = ev.read_start_coverage(env, reset_info)  # read now, before any step
    if hasattr(policy, "reset"):
        policy.reset(seed)
    agents = [np.array(obs["agent_pos"], dtype=np.float64)]
    blocks = [np.array(reset_info["block_pose"], dtype=np.float64)]
    trace = []
    success = False
    terminated = truncated = False
    while len(trace) < max_steps and not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        trace.append(float(info["coverage"]))
        success = success or bool(info["is_success"])
        agents.append(np.array(obs["agent_pos"], dtype=np.float64))
        blocks.append(np.array(info["block_pose"], dtype=np.float64))
    return {
        "seed": int(seed),
        "steps": len(trace),
        "start_coverage": None if start_coverage is None else float(start_coverage),
        "coverage_trace": trace,
        "max_coverage": max(trace),
        "final_coverage": trace[-1],
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "agent_positions": np.array(agents),
        "block_poses": np.array(blocks),
    }


def categorize(result: dict) -> dict:
    """The category and the frozen flag of one replayed episode."""
    start = result["start_coverage"]
    if start is None:
        raise ValueError("the replay has no start coverage")
    trace = result["coverage_trace"]
    peak, final = float(max(trace)), float(trace[-1])
    blocks = np.asarray(result["block_poses"], dtype=np.float64)
    moved_px = float(np.max(np.linalg.norm(blocks[:, :2] - blocks[0, :2], axis=1)))
    moved_rad = float(np.max(np.abs(np.angle(np.exp(1j * (blocks[:, 2] - blocks[0, 2]))))))
    if moved_px < MOVED_PIXELS and moved_rad < MOVED_RADIANS:
        category = CATEGORIES[0]
    elif peak - start < MIN_GAIN:
        category = CATEGORIES[1]
    elif peak - final > BIG_GIVEBACK:
        category = CATEGORIES[2]
    else:
        category = CATEGORIES[3]
    agents = np.asarray(result["agent_positions"], dtype=np.float64)
    window = np.linalg.norm(np.diff(agents, axis=0), axis=1)[-FROZEN_STEPS:]
    frozen = bool(len(window) > 0 and np.all(window < FROZEN_PIXELS))
    return {
        "category": category,
        "frozen": frozen,
        "start_coverage": float(start),
        "peak_coverage": peak,
        "final_coverage": final,
        "block_moved_px": moved_px,
        "block_moved_rad": moved_rad,
    }


def categorize_network(env, policy, report: dict, max_steps: int = ev.MAX_STEPS) -> dict:
    """Replay and categorize every failed episode of a network's evaluation report."""
    failed = [e for e in report["episodes"] if not e["success"]]
    episodes, mismatched = [], []
    for stored in failed:
        result = replay_trajectory(env, policy, int(stored["seed"]), max_steps)
        problems = mg.compare_with_report(result, stored)
        if problems:
            mismatched.append({"seed": int(stored["seed"]), "problems": problems})
            continue
        record = categorize(result)
        record["seed"] = int(stored["seed"])
        episodes.append(record)
    return {
        "policy": report["policy"],
        "failed_in_report": len(failed),
        "categorized": len(episodes),
        "mismatched": mismatched,
        "counts": {c: sum(1 for e in episodes if e["category"] == c) for c in CATEGORIES},
        "frozen": sum(1 for e in episodes if e["frozen"]),
        "episodes": episodes,
    }


def shares(network: dict) -> dict:
    """The share of a network's categorized failed episodes in each category, and the frozen share."""
    n = network["categorized"]
    if n < 1:
        raise ValueError("the network has no categorized episode")
    out = {c: network["counts"][c] / n for c in CATEGORIES}
    out["frozen"] = network["frozen"] / n
    return out


def _mean_ci(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    low, high = ev.bootstrap_ci(v)
    return {"mean": float(v.mean()), "ci95": [low, high]}


def _pooled(networks) -> dict:
    pooled = {c: int(sum(n["counts"][c] for n in networks)) for c in CATEGORIES}
    pooled["frozen"] = int(sum(n["frozen"] for n in networks))
    return pooled


def analyze_block(baseline_networks, chunk_networks) -> dict:
    """The per-method shares and their difference for one block of training seeds."""
    base = [n for n in baseline_networks if n["categorized"] > 0]
    chunk = [n for n in chunk_networks if n["categorized"] > 0]
    out = {
        "networks": {"baseline": len(base), "chunk": len(chunk)},
        "failed_episodes": {"baseline": sum(n["categorized"] for n in base),
                            "chunk": sum(n["categorized"] for n in chunk)},
        "pooled_counts": {"baseline": _pooled(base), "chunk": _pooled(chunk)},
        "rows": None,
        "warnings": [],
    }
    if len(base) < 2 or len(chunk) < 2:
        out["warnings"].append("fewer than two networks with failed episodes in a group: no intervals")
        return out
    if min(len(base), len(chunk)) < BLOCK_SIZE:
        out["warnings"].append(f"fewer than {BLOCK_SIZE} networks in a group: a pilot, not the full analysis")
    rows = {}
    for key in KEYS:
        a = [shares(n)[key] for n in base]
        b = [shares(n)[key] for n in chunk]
        difference, low, high = ss.bootstrap_difference(a, b)
        rows[key] = {"baseline": _mean_ci(a), "chunk": _mean_ci(b), "difference": difference, "ci95": [low, high]}
    out["rows"] = rows
    return out


def _fmt(entry: dict) -> str:
    low, high = entry["ci95"]
    return f"{entry['mean']:.3f} [{low:.3f}, {high:.3f}]"


def format_summary(result: dict) -> str:
    total = sum(n["categorized"] for n in result["networks"])
    lines = [
        f"failure categories: {len(result['networks'])} networks, {total} failed episodes categorized, "
        f"{result['mismatched_episodes']} replays did not match their evaluation report"
    ]
    for label, block in result["blocks"].items():
        n, f = block["networks"], block["failed_episodes"]
        lines.append(
            f"{label}: baseline {n['baseline']} networks ({f['baseline']} failed episodes), "
            f"chunked {n['chunk']} networks ({f['chunk']} failed episodes)"
        )
        if block["rows"] is not None:
            lines.append(f"  {'category':<26}{'baseline share [95% CI]':<30}{'chunked share [95% CI]':<30}"
                         "chunked minus baseline [95% CI]")
            for key in KEYS:
                row = block["rows"][key]
                low, high = row["ci95"]
                name = key if key in CATEGORIES else "frozen at the end (flag)"
                lines.append(f"  {name:<26}{_fmt(row['baseline']):<30}{_fmt(row['chunk']):<30}"
                             f"{row['difference']:+.3f} [{low:+.3f}, {high:+.3f}]")
        for method, label_text in (("baseline", "baseline"), ("chunk", "chunked")):
            pooled = block["pooled_counts"][method]
            lines.append(f"  pooled counts, {label_text}: "
                         + ", ".join(f"{c} {pooled[c]}" for c in CATEGORIES)
                         + f"; frozen {pooled['frozen']} of {f[method]}")
        for warning in block["warnings"]:
            lines.append(f"  WARNING: {warning}")
    lines.append(
        "note: descriptive, with no verdict rule and no correction for the number of comparisons; the "
        "categories describe the replayed trajectory, not why the policy acted as it did"
    )
    return "\n".join(lines)


def main(argv=None, *, env_factory=ev.make_eval_env, policy_builder=dc.build_policy, log=print) -> int:
    parser = argparse.ArgumentParser(description="Pre-registered categorization of failed episodes")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("reports/failure_categories.json"))
    parser.add_argument("--max-steps", type=int, default=ev.MAX_STEPS)
    args = parser.parse_args(argv)

    networks = []
    try:
        if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
            raise ValueError("--seeds must be distinct non-negative integers")
        for seed in args.seeds:
            for kind, method in (("bc_cnn", "baseline"), ("chunk_cnn", "chunk")):
                name = ab.network_name(method, ab.FULL_SIZE, seed)
                report = cp.load_report(args.report_dir / f"eval_{name}.json")
                policy = policy_builder(kind, args.checkpoint_dir / f"{name}.pt", env_factory, None)
                env = env_factory()
                try:
                    network = categorize_network(env, policy, report, args.max_steps)
                finally:
                    env.close()
                network.update(method=method, seed=seed)
                networks.append(network)
                log(f"{name}: {network['categorized']} of {network['failed_in_report']} failed episodes "
                    f"categorized, {len(network['mismatched'])} replays did not match")
    except (cp.ComparisonError, OSError, ValueError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    starts = sorted({BLOCK_SIZE * (n["seed"] // BLOCK_SIZE) for n in networks})
    blocks = {}
    for start in starts:
        label = block_label(start)
        members = [n for n in networks if block_label(n["seed"]) == label]
        blocks[label] = analyze_block([n for n in members if n["method"] == "baseline"],
                                      [n for n in members if n["method"] == "chunk"])
    mismatched = sum(len(n["mismatched"]) for n in networks)
    result = {
        "settings": {"moved_pixels": MOVED_PIXELS, "moved_radians": MOVED_RADIANS, "min_gain": MIN_GAIN,
                     "big_giveback": BIG_GIVEBACK, "frozen_steps": FROZEN_STEPS, "frozen_pixels": FROZEN_PIXELS,
                     "block_size": BLOCK_SIZE},
        "mismatched_episodes": mismatched,
        "networks": networks,
        "blocks": blocks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(format_summary(result))
    print(f"saved {args.out}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
