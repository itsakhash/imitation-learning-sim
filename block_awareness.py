"""Does the policy head for the block, and where does it park?

timeline_survey.py showed that the trained network's commanded target collapses onto the
agent's own position after the first quarter of an episode, that 30 of 50 episodes end
frozen, and that 22 of the 45 classified episodes never touched anything. first_step_check.py
found that at the very first step the network commands a target about as far from the agent
as a human does, and that pictures do not explain the early behaviour. What is still open is
what the network does relative to the BLOCK: does it head for it, and does it park near it or
far from it?

For every step of every evaluation episode (replayed exactly as evaluate.py ran it) this
compares the direction of the commanded target (target minus agent position) with the
direction from the agent to the block's reference position (gym-pusht's block body position,
close to but not exactly the centre of the T). The cosine is only defined on steps where both
vectors are longer than MIN_VECTOR_PX.

Reading guide, fixed before looking at any results:
  heads for the block    the mean cosine in phase 1, and at the very first step, is above 0.5
  no clear sign of it    that mean is below 0.2. This is NOT conclusive: a policy that
                         approaches obliquely to push from the far side would also score low
  approached the block   an episode whose final agent-to-block distance is under half of its
                         starting distance
  did not approach       a final distance of at least 80% of the starting distance (the agent
                         did not get even 20% closer)

Limits: the demonstrations do not record the block, so there is no human reference for any of
these numbers; direction towards the block's reference point is a crude measure of what a
policy is aiming at; this describes one training run on 50 starting positions.

Run:  python block_awareness.py --policy bc_cnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import evaluate as ev
import inspect_episode as ie
import timeline_survey as tl

N_PHASES = 4
MIN_VECTOR_PX = 1.0
APPROACH_FRACTION = 0.5
NOT_CLOSER_FRACTION = 0.8
HEADS_FOR_BLOCK_COSINE = 0.5
NO_SIGN_COSINE = 0.2


def step_vectors(ep: dict):
    """(commanded displacement, direction to the block) at every step of a replayed episode.
    Both use the agent's position and the block's position BEFORE the step."""
    agent = np.asarray(ep["agent"], dtype=np.float64)
    block = np.asarray(ep["block"], dtype=np.float64)
    actions = np.asarray(ep["actions"], dtype=np.float64)
    steps = len(actions)
    if steps == 0:
        raise ValueError("episode has no steps")
    if agent.shape != (steps + 1, 2) or block.shape != (steps + 1, 3):
        raise ValueError("agent and block need one more row than there are actions")
    return actions - agent[:-1], block[:-1, :2] - agent[:-1]


def cosines(commanded, to_block) -> np.ndarray:
    """Cosine of the angle between the two vectors at every step; NaN where either vector
    is not longer than MIN_VECTOR_PX."""
    commanded = np.asarray(commanded, dtype=np.float64)
    to_block = np.asarray(to_block, dtype=np.float64)
    if (
        commanded.shape != to_block.shape
        or commanded.ndim != 2
        or commanded.shape[1] != 2
    ):
        raise ValueError("expected two (N, 2) arrays of the same shape")
    nc = np.linalg.norm(commanded, axis=1)
    nb = np.linalg.norm(to_block, axis=1)
    valid = (nc > MIN_VECTOR_PX) & (nb > MIN_VECTOR_PX)
    out = np.full(len(nc), np.nan)
    out[valid] = (commanded[valid] * to_block[valid]).sum(axis=1) / (
        nc[valid] * nb[valid]
    )
    return out


def episode_block_stats(
    ep: dict, n_phases: int = N_PHASES, tail_min: int = tl.FROZEN_TAIL_MIN
) -> dict:
    """Direction and distance statistics of one replayed episode."""
    if n_phases < 1:
        raise ValueError("n_phases must be at least 1")
    commanded, to_block = step_vectors(ep)
    steps = len(commanded)
    if steps < n_phases:
        raise ValueError("need at least one step per phase")
    cos = cosines(commanded, to_block)
    dist = np.linalg.norm(to_block, axis=1)
    agent = np.asarray(ep["agent"], dtype=np.float64)
    block = np.asarray(ep["block"], dtype=np.float64)
    final_distance = float(np.linalg.norm(block[-1, :2] - agent[-1]))
    timeline = tl.timeline_stats(ep, n_phases=n_phases, tail_min=tail_min)
    parts = np.array_split(np.arange(steps), n_phases)
    return {
        "seed": int(ep["seed"]),
        "steps": int(steps),
        "ends_frozen": bool(timeline["ends_frozen"]),
        "starts_in_contact": bool(timeline["starts_in_contact"]),
        "gain_over_start": timeline["gain_over_start"],
        "start_distance_px": float(dist[0]),
        "final_distance_px": final_distance,
        "min_distance_px": float(min(dist.min(), final_distance)),
        "dist_by_phase": [float(dist[p].mean()) for p in parts],
        "cos_sum_by_phase": [float(np.nansum(cos[p])) for p in parts],
        "cos_count_by_phase": [int((~np.isnan(cos[p])).sum()) for p in parts],
        "first_step_cosine": None if np.isnan(cos[0]) else float(cos[0]),
    }


def _group(stats) -> dict:
    stats = list(stats)
    if not stats:
        return {
            "episodes": 0,
            "median_start_px": None,
            "median_final_px": None,
            "approached": 0,
            "not_closer": 0,
        }
    start = [s["start_distance_px"] for s in stats]
    final = [s["final_distance_px"] for s in stats]
    return {
        "episodes": len(stats),
        "median_start_px": float(np.median(start)),
        "median_final_px": float(np.median(final)),
        "approached": sum(f < APPROACH_FRACTION * st for st, f in zip(start, final)),
        "not_closer": sum(f >= NOT_CLOSER_FRACTION * st for st, f in zip(start, final)),
    }


def aggregate_block_stats(stats_list) -> dict:
    """Overall numbers over many episodes. Cosines are pooled over all usable steps."""
    stats_list = list(stats_list)
    if not stats_list:
        raise ValueError("no episodes to aggregate")
    n_phases = len(stats_list[0]["dist_by_phase"])
    cosine_by_phase, steps_by_phase = [], []
    for i in range(n_phases):
        total = sum(s["cos_sum_by_phase"][i] for s in stats_list)
        count = sum(s["cos_count_by_phase"][i] for s in stats_list)
        cosine_by_phase.append(None if count == 0 else total / count)
        steps_by_phase.append(int(count))
    first = [
        s["first_step_cosine"] for s in stats_list if s["first_step_cosine"] is not None
    ]
    return {
        "episodes": len(stats_list),
        "n_phases": n_phases,
        "cosine_by_phase": cosine_by_phase,
        "cosine_steps_by_phase": steps_by_phase,
        "median_distance_by_phase": [
            float(np.median([s["dist_by_phase"][i] for s in stats_list]))
            for i in range(n_phases)
        ],
        "first_step_cosine_mean": float(np.mean(first)) if first else None,
        "first_step_cosine_median": float(np.median(first)) if first else None,
        "first_step_cosine_episodes": len(first),
        "frozen": _group(s for s in stats_list if s["ends_frozen"]),
        "active": _group(s for s in stats_list if not s["ends_frozen"]),
    }


def survey_block_awareness(env, policy, seeds, report=None) -> list:
    """Replay each seed and return (block stats, match) pairs, where match says whether the
    replay equals the saved evaluation trace (None if there is no report)."""
    results = []
    for seed in seeds:
        ep = ie.replay_episode(env, policy, seed, frame_steps=())
        stats = episode_block_stats(ep)
        match = None if report is None else ie.replay_matches_report(ep, report)
        results.append((stats, match))
    return results


def _num(value, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _group_line(name: str, g: dict) -> str:
    if g["episodes"] == 0:
        return f"{name}: 0 episodes"
    return (
        f"{name}: {g['episodes']} episodes; median agent-to-block distance "
        f"{g['median_start_px']:.0f} px at the start and {g['median_final_px']:.0f} px at the end; "
        f"final distance under half the start in {g['approached']}, "
        f"not even 20% closer in {g['not_closer']}"
    )


def format_report(policy: str, agg: dict, counts: dict | None = None) -> str:
    n, phases = agg["episodes"], agg["n_phases"]
    width = 46

    def row(label, values, spec):
        return f"{label:<{width}s}" + "".join(f"{_num(v, spec):>10s}" for v in values)

    header = " " * width + "".join(
        f"{'phase ' + str(i + 1):>10s}" for i in range(phases)
    )
    lines = [
        f"{policy}: {n} episodes, each cut into {phases} equal phases of its steps",
        header,
        row("mean cosine: commanded direction vs block", agg["cosine_by_phase"], ".2f"),
        row("steps with a usable direction", agg["cosine_steps_by_phase"], "d"),
        row(
            "median distance, agent to block (px)",
            agg["median_distance_by_phase"],
            ".0f",
        ),
        f"first step: mean cosine {_num(agg['first_step_cosine_mean'], '.2f')}, median "
        f"{_num(agg['first_step_cosine_median'], '.2f')}, over {agg['first_step_cosine_episodes']} episodes",
        _group_line("ends frozen", agg["frozen"]),
        _group_line("active to the end", agg["active"]),
    ]
    if counts is not None:
        lines.append(
            f"replays matching the saved evaluation: {counts['yes']} yes, {counts['no']} NO, "
            f"{counts['unchecked']} unchecked"
        )
    lines.append(
        "note: the block position is the reference point of gym-pusht's block body; cosine "
        f"above {HEADS_FOR_BLOCK_COSINE} counts as heading for the block, below {NO_SIGN_COSINE} as no "
        "clear sign (not conclusive: an oblique approach scores low); there is no human reference "
        "because the demonstrations do not record the block; one training run, 50 starts"
    )
    return "\n".join(lines)


def main(argv=None, *, env_factory=ev.make_eval_env) -> int:
    parser = argparse.ArgumentParser(
        description="Does the policy head for the block, and where does it park?"
    )
    parser.add_argument("--policy", choices=ie.POLICY_NAMES, default="bc_cnn")
    parser.add_argument("--episodes", type=int, default=ev.MIN_EPISODES)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/bc_cnn.pt")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report_path = args.report_dir / f"eval_{args.policy}.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    env = env_factory()
    try:
        policy = ie.build_policy(args.policy, env, args.checkpoint)
        results = survey_block_awareness(
            env, policy, ev.eval_seeds(args.episodes), report
        )
    finally:
        env.close()
    agg = aggregate_block_stats([stats for stats, _ in results])
    matches = [match for _, match in results]
    counts = {
        "yes": sum(m is True for m in matches),
        "no": sum(m is False for m in matches),
        "unchecked": sum(m is None for m in matches),
    }
    out = args.out or args.report_dir / f"block_awareness_{args.policy}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "policy": args.policy,
                "summary": agg,
                "replay_matches": counts,
                "per_episode": [stats for stats, _ in results],
            },
            indent=2,
        )
        + "\n"
    )
    print(format_report(args.policy, agg, counts))
    print(f"saved {out}")
    return 1 if counts["no"] else 0


if __name__ == "__main__":
    sys.exit(main())
