"""Timeline survey: when during an episode does the policy move, touch the block and stop?

behavior_survey.py gives whole-episode averages. This looks at time: how the share of
standing still, the gap between the commanded target and the agent, and contact with
something change over the course of an episode, and how many episodes end with the agent
frozen. A freeze that comes before any contact (never got going) and one that comes after
touching the block (reached it, then gave up) are different problems.

Definitions, fixed before looking at any results:
  moving             the agent moved at least ie.STALL_PX (1 px) during the step
  ends frozen        the agent did not move for the last FROZEN_TAIL_MIN (50) steps or more
  contact            info["n_contacts"] > 0 during the step. gym-pusht counts contact points
                     of every shape pair, so contact is not proof of pushing the block
  starts in contact  contact on the very first step. Some start positions have the agent or
                     the block touching something from the start (doing nothing already shows
                     it), so these episodes are reported separately, not counted as engaging
  engaged            contact on at least one step, in an episode that does not start in contact
  phases             each episode is cut into N_PHASES equal parts of its steps

Run:  python timeline_survey.py --policy bc_cnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import evaluate as ev
import inspect_episode as ie

N_PHASES = 4
FROZEN_TAIL_MIN = 50


def phase_means(values, n_phases: int = N_PHASES) -> list:
    """Mean of the values within each of n_phases consecutive, nearly equal parts."""
    values = np.asarray(values, dtype=np.float64)
    if n_phases < 1:
        raise ValueError("n_phases must be at least 1")
    if values.ndim != 1 or len(values) < n_phases:
        raise ValueError("need at least one value per phase")
    return [float(part.mean()) for part in np.array_split(values, n_phases)]


def timeline_stats(
    ep: dict, n_phases: int = N_PHASES, tail_min: int = FROZEN_TAIL_MIN
) -> dict:
    """When the agent moved, touched something and stopped, for one replayed episode."""
    base = ie.behavior_stats(ep)  # also rejects an empty episode
    agent = np.asarray(ep["agent"], dtype=np.float64)
    actions = np.asarray(ep["actions"], dtype=np.float64)
    contacts = np.asarray(ep["contacts"], dtype=np.float64) > 0
    moves = np.linalg.norm(np.diff(agent, axis=0), axis=1)
    steps = len(moves)
    moving = moves >= ie.STALL_PX
    gap = np.linalg.norm(actions - agent[:-1], axis=1)

    first_move = int(np.argmax(moving)) + 1 if moving.any() else None
    last_move = int(steps - np.argmax(moving[::-1])) if moving.any() else None
    tail = steps - (last_move or 0)
    touched = bool(contacts.any())
    starts_in_contact = bool(contacts[0])
    return {
        "seed": int(ep["seed"]),
        "steps": int(steps),
        "tail_min": int(tail_min),
        "first_move_step": first_move,
        "last_move_step": last_move,
        "frozen_tail_steps": int(tail),
        "ends_frozen": bool(tail >= tail_min),
        "starts_in_contact": starts_in_contact,
        "engaged": None if starts_in_contact else touched,
        "first_contact_step": int(np.argmax(contacts)) + 1 if touched else None,
        "last_contact_step": (
            int(steps - np.argmax(contacts[::-1])) if touched else None
        ),
        "contact_steps": int(contacts.sum()),
        "stalled_by_phase": phase_means(~moving, n_phases),
        "gap_by_phase": phase_means(gap, n_phases),
        "contact_by_phase": phase_means(contacts, n_phases),
        "start_coverage": base["start_coverage"],
        "max_coverage": base["max_coverage"],
        "final_coverage": base["final_coverage"],
        "gain_over_start": base["gain_over_start"],
    }


def _mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _median_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def aggregate_timelines(stats_list) -> dict:
    """Overall numbers over many episodes, and the frozen-versus-touched table."""
    stats_list = list(stats_list)
    if not stats_list:
        raise ValueError("no episodes to aggregate")
    classified = [s for s in stats_list if s["engaged"] is not None]
    table = {}
    for frozen in (True, False):
        for engaged in (True, False):
            group = [
                s
                for s in classified
                if s["ends_frozen"] == frozen and s["engaged"] == engaged
            ]
            name = f"{'frozen' if frozen else 'active'}_{'engaged' if engaged else 'not_engaged'}"
            table[name] = {
                "episodes": len(group),
                "mean_gain_over_start": _mean_or_none(
                    [s["gain_over_start"] for s in group]
                ),
            }

    def phase_mean(key):
        return np.mean([s[key] for s in stats_list], axis=0).tolist()

    return {
        "episodes": len(stats_list),
        "n_phases": len(stats_list[0]["stalled_by_phase"]),
        "frozen_tail_min": stats_list[0]["tail_min"],
        "stalled_by_phase_mean": phase_mean("stalled_by_phase"),
        "gap_by_phase_mean": phase_mean("gap_by_phase"),
        "contact_by_phase_mean": phase_mean("contact_by_phase"),
        "ends_frozen": sum(s["ends_frozen"] for s in stats_list),
        "never_moved": sum(s["first_move_step"] is None for s in stats_list),
        "starts_in_contact": sum(s["starts_in_contact"] for s in stats_list),
        "median_frozen_tail_steps": _median_or_none(
            [s["frozen_tail_steps"] for s in stats_list]
        ),
        "median_last_move_step": _median_or_none(
            [s["last_move_step"] for s in stats_list]
        ),
        "median_first_contact_step": _median_or_none(
            [s["first_contact_step"] for s in stats_list if s["engaged"]]
        ),
        "table": table,
    }


def survey_timelines(
    env, policy, seeds, report=None, tail_min: int = FROZEN_TAIL_MIN
) -> list:
    """Replay each seed and return (timeline stats, match) pairs, where match says whether
    the replay equals the saved evaluation trace (None if there is no report)."""
    results = []
    for seed in seeds:
        ep = ie.replay_episode(env, policy, seed, frame_steps=())
        stats = timeline_stats(ep, tail_min=tail_min)
        match = None if report is None else ie.replay_matches_report(ep, report)
        results.append((stats, match))
    return results


def _fmt(value, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _cell(group: dict) -> str:
    if group["episodes"] == 0:
        return "0"
    return f"{group['episodes']} (mean gain {group['mean_gain_over_start']:+.3f})"


def format_timeline(policy: str, agg: dict, counts: dict | None = None) -> str:
    n, phases = agg["episodes"], agg["n_phases"]
    label_width = 34

    def row(label, values, spec):
        return f"{label:{label_width}s}" + "".join(
            f"{format(v, spec):>10s}" for v in values
        )

    header = " " * label_width + "".join(
        f"{'phase ' + str(i + 1):>10s}" for i in range(phases)
    )
    t = agg["table"]
    lines = [
        f"{policy}: {n} episodes, each cut into {phases} equal phases of its steps",
        header,
        row("share of steps standing still", agg["stalled_by_phase_mean"], ".0%"),
        row("mean gap, target to agent (px)", agg["gap_by_phase_mean"], ".1f"),
        row("share of steps in contact", agg["contact_by_phase_mean"], ".0%"),
        f"ends frozen (no move in the last {agg['frozen_tail_min']}+ steps): "
        f"{agg['ends_frozen']} of {n}; median frozen tail "
        f"{_fmt(agg['median_frozen_tail_steps'], '.0f')} steps; "
        f"agent never moved in {agg['never_moved']} episodes",
        f"median step of the last move (episodes that moved): "
        f"{_fmt(agg['median_last_move_step'], '.0f')}; median step of the first contact "
        f"(engaged episodes): {_fmt(agg['median_first_contact_step'], '.0f')}",
        f"{agg['starts_in_contact']} episodes start in contact and are left out of the table",
        f"{'':20s}{'touched something':>30s}{'never touched anything':>30s}",
        f"{'ends frozen':20s}{_cell(t['frozen_engaged']):>30s}{_cell(t['frozen_not_engaged']):>30s}",
        f"{'active to the end':20s}{_cell(t['active_engaged']):>30s}{_cell(t['active_not_engaged']):>30s}",
    ]
    if counts is not None:
        lines.append(
            f"replays matching the saved evaluation: {counts['yes']} yes, {counts['no']} NO, "
            f"{counts['unchecked']} unchecked"
        )
    lines.append(
        "note: contact means any shape pair touching (walls included), the 50-step freeze rule "
        "and the 1 px movement threshold were fixed in advance, and this describes one "
        "training run on these 50 starting positions"
    )
    return "\n".join(lines)


def main(argv=None, *, env_factory=ev.make_eval_env) -> int:
    parser = argparse.ArgumentParser(
        description="When does a policy move, touch and stop?"
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
        results = survey_timelines(env, policy, ev.eval_seeds(args.episodes), report)
    finally:
        env.close()
    agg = aggregate_timelines([stats for stats, _ in results])
    matches = [match for _, match in results]
    counts = {
        "yes": sum(m is True for m in matches),
        "no": sum(m is False for m in matches),
        "unchecked": sum(m is None for m in matches),
    }
    out = args.out or args.report_dir / f"timeline_{args.policy}.json"
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
    print(format_timeline(args.policy, agg, counts))
    print(f"saved {out}")
    return 1 if counts["no"] else 0


if __name__ == "__main__":
    sys.exit(main())
