"""Survey how a policy behaves across ALL of its evaluation episodes, next to the human
demonstrations.

inspect_episode.py looks closely at a few episodes picked for their scores. This looks at
every evaluation episode of a policy and puts the same behaviour numbers next to the ones
computed from the recorded human demonstrations, so that a statement like "the network
stands still a lot" can be checked against how much the humans stood still.

For the trained network it also measures, offline on the held-out demonstration episodes,
how far the network's commanded target is from the agent compared with the human's, and how
well the directions agree. Shrinking the commanded target towards the agent's position is
what a mean-squared-error regressor tends to do when it is unsure where to go.

Caveats: demonstration episodes end when the human finishes while evaluation episodes run
until success or 300 steps; starting positions differ; the demonstrations do not record the
block, so contact and block motion exist only for the policy. Read the two columns as a
rough guide, not a controlled test.

Run:  python behavior_survey.py --policy bc_cnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import evaluate as ev
import inspect_episode as ie
import pusht_data
import train_bc as tb
from pusht_data import COORD_SCALE, PushTDataset

BLOCK_MOVED_PX = (
    1.0  # an episode in which the block travelled less than this never moved it
)
LOST_COVERAGE = (
    0.1  # peak minus final above this counts as "lost coverage it had reached"
)


def demo_motion_stats(states, actions, episode_index) -> dict:
    """Motion statistics of the recorded human demonstrations, computed the same way as
    inspect_episode.behavior_stats does for a policy (moves are only taken between frames
    of the same episode; the action gap is measured from the agent's current position).
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    episode_index = np.asarray(episode_index)
    if states.ndim != 2 or states.shape[1] != 2 or actions.shape != states.shape:
        raise ValueError("states and actions must both have shape (N, 2)")
    if episode_index.shape != (len(states),):
        raise ValueError("episode_index must have one entry per frame")
    same_episode = episode_index[1:] == episode_index[:-1]
    moves = np.linalg.norm(np.diff(states, axis=0), axis=1)[same_episode]
    if len(moves) == 0:
        raise ValueError("need at least one step inside an episode")
    gap = np.linalg.norm(actions - states, axis=1)
    border = (actions < ie.BORDER_PX) | (actions > COORD_SCALE - ie.BORDER_PX)
    return {
        "episodes": int(len(np.unique(episode_index))),
        "frames": int(len(states)),
        "stalled_fraction": float((moves < ie.STALL_PX).mean()),
        "mean_move_px_per_step": float(moves.mean()),
        "mean_action_gap_px": float(gap.mean()),
        "median_action_gap_px": float(np.median(gap)),
        "border_action_fraction": float(border.any(axis=1).mean()),
    }


def survey_policy(env, policy, seeds, report=None) -> list:
    """Replay each seed (without keeping frames) and return (behaviour stats, match)
    pairs, where match says whether the replay equals the saved evaluation trace."""
    results = []
    for seed in seeds:
        ep = ie.replay_episode(env, policy, seed, frame_steps=())
        stats = ie.behavior_stats(ep)
        match = None if report is None else ie.replay_matches_report(ep, report)
        results.append((stats, match))
    return results


def aggregate_behavior(stats_list) -> dict:
    """Overall behaviour numbers. Shares are weighted by episode length, so they are the
    share of all steps."""
    stats_list = list(stats_list)
    if not stats_list:
        raise ValueError("no episodes to aggregate")

    def column(key):
        return np.array([s[key] for s in stats_list], dtype=np.float64)

    steps = column("steps")

    def weighted(key):
        return float(np.average(column(key), weights=steps))

    return {
        "episodes": len(stats_list),
        "stalled_fraction": weighted("stalled_fraction"),
        "stalled_fraction_median_episode": float(np.median(column("stalled_fraction"))),
        "mean_move_px_per_step": float(column("agent_path_px").sum() / steps.sum()),
        "mean_action_gap_px": weighted("mean_action_gap_px"),
        "median_episode_action_gap_px": float(np.median(column("mean_action_gap_px"))),
        "border_action_fraction": weighted("border_action_fraction"),
        "contact_step_fraction": weighted("contact_step_fraction"),
        "episodes_block_never_moved": int(
            (column("block_path_px") < BLOCK_MOVED_PX).sum()
        ),
        "episodes_lost_coverage": int((column("drop_from_peak") > LOST_COVERAGE).sum()),
    }


def offline_gap_stats(model, arrays, episode_ids, device="cpu", batch_size=512) -> dict:
    """On the given demonstration episodes: how far the human's commanded target was from
    the agent, how far the network's is, how much of the human's displacement the network
    reproduces (displacement_gain, 1.0 means no shrinkage) and how well the directions
    agree (mean_cosine over frames where both displacements exceed 1 px)."""
    dataset = PushTDataset(arrays, episode_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    human, network = [], []
    with torch.no_grad():
        for batch in loader:
            pos = batch["agent_pos"].to(device)
            pred = model(batch["image"].to(device), pos)
            network.append(
                ((pred - pos) * COORD_SCALE).cpu().numpy().astype(np.float64)
            )
            human.append(
                ((batch["action"].to(device) - pos) * COORD_SCALE)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    dh = np.concatenate(human)
    dp = np.concatenate(network)
    norm_h = np.linalg.norm(dh, axis=1)
    norm_p = np.linalg.norm(dp, axis=1)
    denominator = float((dh * dh).sum())
    both = (norm_h > 1.0) & (norm_p > 1.0)
    cosine = None
    if both.any():
        cosine = float(
            ((dh[both] * dp[both]).sum(axis=1) / (norm_h[both] * norm_p[both])).mean()
        )
    return {
        "episodes": int(len(set(int(i) for i in episode_ids))),
        "frames": int(len(dh)),
        "human_gap_px": float(norm_h.mean()),
        "network_gap_px": float(norm_p.mean()),
        "displacement_gain": (
            None if denominator == 0 else float((dh * dp).sum() / denominator)
        ),
        "mean_cosine": cosine,
        "cosine_frames": int(both.sum()),
    }


def format_survey(policy, agg, demo, counts, offline=None) -> str:
    rows = [
        ("share of steps with the agent standing still", "stalled_fraction", "{:.1%}"),
        ("mean agent move per step (px)", "mean_move_px_per_step", "{:.1f}"),
        ("mean gap, commanded target to agent (px)", "mean_action_gap_px", "{:.1f}"),
        (
            "share of steps with the target at the world's edge",
            "border_action_fraction",
            "{:.1%}",
        ),
    ]
    lines = [f"{'':52s}{'human demos':>14s}{policy:>14s}"]
    for label, key, fmt in rows:
        lines.append(
            f"{label:52s}{fmt.format(demo[key]):>14s}{fmt.format(agg[key]):>14s}"
        )
    n = agg["episodes"]
    lines += [
        f"policy only: the median episode stands still on "
        f"{100 * agg['stalled_fraction_median_episode']:.0f}% of steps; contact with the block "
        f"on {100 * agg['contact_step_fraction']:.0f}% of steps (the demos do not record the block)",
        f"episodes in which the block never moved: {agg['episodes_block_never_moved']} of {n}",
        f"episodes that lost more than {LOST_COVERAGE} of the coverage they had reached: "
        f"{agg['episodes_lost_coverage']} of {n}",
        f"replays matching the saved evaluation: {counts['yes']} yes, {counts['no']} NO, "
        f"{counts['unchecked']} unchecked",
    ]
    if offline is not None:
        gain = (
            "n/a"
            if offline["displacement_gain"] is None
            else f"{offline['displacement_gain']:.2f}"
        )
        cosine = (
            "n/a" if offline["mean_cosine"] is None else f"{offline['mean_cosine']:.2f}"
        )
        lines.append(
            f"offline, on {offline['episodes']} held-out demonstration episodes "
            f"({offline['frames']} frames): the human's target is {offline['human_gap_px']:.1f} px "
            f"from the agent, the network's {offline['network_gap_px']:.1f} px; the network "
            f"reproduces {gain} of the human displacement (1.00 = no shrinkage); direction "
            f"agreement (cosine) {cosine}"
        )
    lines.append(
        "note: demonstration episodes end when the human finishes, evaluation episodes run "
        "until success or 300 steps, and starting positions differ; read the columns as a "
        "rough guide, not a controlled test"
    )
    return "\n".join(lines)


def main(
    argv=None,
    *,
    env_factory=ev.make_eval_env,
    demo_loader=pusht_data.read_table,
    arrays_loader=pusht_data.load_arrays,
) -> int:
    parser = argparse.ArgumentParser(
        description="Survey a policy's behaviour over all episodes"
    )
    parser.add_argument("--policy", choices=ie.POLICY_NAMES, default="bc_cnn")
    parser.add_argument("--episodes", type=int, default=ev.MIN_EPISODES)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/bc_cnn.pt")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--data-root", type=Path, default=pusht_data.DEFAULT_ROOT)
    parser.add_argument("--no-offline", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report_path = args.report_dir / f"eval_{args.policy}.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    table = demo_loader(args.data_root)
    demo = demo_motion_stats(table["states"], table["actions"], table["episode_index"])

    env = env_factory()
    try:
        policy = ie.build_policy(args.policy, env, args.checkpoint)
        results = survey_policy(env, policy, ev.eval_seeds(args.episodes), report)
    finally:
        env.close()
    agg = aggregate_behavior([stats for stats, _ in results])
    matches = [match for _, match in results]
    counts = {
        "yes": sum(m is True for m in matches),
        "no": sum(m is False for m in matches),
        "unchecked": sum(m is None for m in matches),
    }

    offline = None
    if args.policy == "bc_cnn" and not args.no_offline:
        model, ckpt = tb.load_checkpoint(args.checkpoint, device="cpu")
        offline = offline_gap_stats(
            model, arrays_loader(), ckpt["split"]["val_episode_ids"]
        )

    out = args.out or args.report_dir / f"behavior_{args.policy}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "policy": args.policy,
                "episodes": agg["episodes"],
                "demos": demo,
                "policy_summary": agg,
                "replay_matches": counts,
                "offline": offline,
                "per_episode": [stats for stats, _ in results],
            },
            indent=2,
        )
        + "\n"
    )
    print(format_survey(args.policy, agg, demo, counts, offline))
    print(f"saved {out}")
    return 1 if counts["no"] else 0


if __name__ == "__main__":
    sys.exit(main())
