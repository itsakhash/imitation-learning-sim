"""Replay evaluation episodes and look at what the policy actually did.

For each seed this re-runs the episode exactly as evaluate.py ran it, then writes
  reports/inspect_<policy>_<seed>.png   rendered frames at chosen steps (the red cross is
                                        the position the policy last asked the agent to go
                                        to) and the coverage over time
  reports/inspect_<policy>_<seed>.json  summary numbers about the behaviour
and checks that the replayed coverage trace matches the one saved in the evaluation
report, so that what you look at is the episode that was scored.

Inspecting a handful of episodes is exploration. It suggests what to look for, but it is
not a systematic failure analysis: the episodes worth looking at were picked after seeing
their scores.

Run:  python inspect_episode.py --policy bc_cnn --seed 10022 10034 10000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import evaluate as ev
from bc_policy import BCPolicy
from pusht_data import COORD_SCALE
from train_bc import load_checkpoint

DEFAULT_FRAME_STEPS = (0, 25, 50, 100, 150, 200, 250, 300)
STALL_PX = 1.0  # an agent move shorter than this in one step counts as standing still
BORDER_PX = 5.0  # an action this close to the edge of the world counts as at the border
TRACE_TOLERANCE = 1e-4
POLICY_NAMES = ("bc_cnn", "random", "hold")


def replay_episode(
    env, policy, seed, frame_steps=DEFAULT_FRAME_STEPS, max_steps=ev.MAX_STEPS
):
    """Run one seeded episode like evaluate.run_episode, and also record the agent's and
    block's positions, the actions, the contact counts and rendered frames. Step 0 is the
    state just after reset. The last frame of the episode is always captured."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    wanted = {int(s) for s in frame_steps}
    obs, reset_info = env.reset(seed=seed)
    start_coverage = ev.read_start_coverage(env, reset_info)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    agent = [np.array(reset_info["pos_agent"], dtype=np.float64)]
    block = [np.array(reset_info["block_pose"], dtype=np.float64)]
    coverage, actions, contacts = [], [], []
    frames = {}
    if 0 in wanted:
        frames[0] = np.array(env.render())
    success = terminated = truncated = False
    while len(coverage) < max_steps and not (terminated or truncated):
        action = policy(obs)
        obs, _, terminated, truncated, info = env.step(action)
        actions.append(np.array(action, dtype=np.float64))
        coverage.append(float(info["coverage"]))
        agent.append(np.array(info["pos_agent"], dtype=np.float64))
        block.append(np.array(info["block_pose"], dtype=np.float64))
        contacts.append(float(info["n_contacts"]))
        success = success or bool(info["is_success"])
        if len(coverage) in wanted:
            frames[len(coverage)] = np.array(env.render())
    steps = len(coverage)
    if steps not in frames:
        frames[steps] = np.array(env.render())
    return {
        "seed": int(seed),
        "steps": steps,
        "start_coverage": start_coverage,
        "success": success,
        "coverage": np.array(coverage),
        "agent": np.array(agent),  # (steps + 1, 2), step 0 first
        "block": np.array(block),  # (steps + 1, 3): x, y, angle
        "actions": np.array(actions),  # (steps, 2)
        "contacts": np.array(contacts),
        "frames": frames,
    }


def behavior_stats(ep: dict) -> dict:
    """Numbers that describe what the agent and the block did during an episode."""
    coverage = np.asarray(ep["coverage"], dtype=np.float64)
    if len(coverage) == 0:
        raise ValueError("episode has no steps")
    agent = np.asarray(ep["agent"], dtype=np.float64)
    block = np.asarray(ep["block"], dtype=np.float64)
    actions = np.asarray(ep["actions"], dtype=np.float64)
    contacts = np.asarray(ep["contacts"], dtype=np.float64)
    moves = np.linalg.norm(np.diff(agent, axis=0), axis=1)
    block_moves = np.linalg.norm(np.diff(block[:, :2], axis=0), axis=1)
    start = ep.get("start_coverage")
    peak = float(coverage.max())
    final = float(coverage[-1])
    border = (actions < BORDER_PX) | (actions > COORD_SCALE - BORDER_PX)
    return {
        "seed": int(ep["seed"]),
        "steps": int(len(coverage)),
        "start_coverage": None if start is None else float(start),
        "max_coverage": peak,
        "step_of_max": int(np.argmax(coverage)) + 1,
        "final_coverage": final,
        "drop_from_peak": peak - final,
        "gain_over_start": None if start is None else peak - float(start),
        "agent_path_px": float(moves.sum()),
        "stalled_fraction": float((moves < STALL_PX).mean()),
        "mean_action_gap_px": float(
            np.linalg.norm(actions - agent[:-1], axis=1).mean()
        ),
        "border_action_fraction": float(border.any(axis=1).mean()),
        "block_net_displacement_px": float(
            np.linalg.norm(block[-1, :2] - block[0, :2])
        ),
        "block_path_px": float(block_moves.sum()),
        "contact_step_fraction": float((contacts > 0).mean()),
    }


def replay_matches_report(ep: dict, report: dict):
    """True or False: does the replayed coverage trace equal the saved one? None if the
    seed is not in the report."""
    for saved in report["episodes"]:
        if saved["seed"] == ep["seed"]:
            trace = np.round(np.asarray(ep["coverage"], dtype=np.float64), 4)
            reference = np.asarray(saved["coverage_trace"], dtype=np.float64)
            return bool(
                trace.shape == reference.shape
                and np.allclose(trace, reference, atol=TRACE_TOLERANCE)
            )
    return None


def save_inspection_figure(ep: dict, path, title: str = "") -> None:
    """A contact sheet of the captured frames with the coverage over time underneath."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = dict(sorted(ep["frames"].items()))
    if not frames:
        raise ValueError("no frames to draw")
    coverage = np.asarray(ep["coverage"], dtype=np.float64)
    start = ep.get("start_coverage")
    rows = math.ceil(len(frames) / 4)
    fig = plt.figure(figsize=(12, 3 * rows + 3))
    grid = fig.add_gridspec(rows + 1, 4, height_ratios=[1.0] * rows + [0.8])
    for i, (step, image) in enumerate(frames.items()):
        ax = fig.add_subplot(grid[i // 4, i % 4])
        ax.imshow(image)
        ax.axis("off")
        value = start if step == 0 else coverage[step - 1]
        label = "?" if value is None else f"{float(value):.2f}"
        ax.set_title(f"step {step}   coverage {label}", fontsize=9)
    ax = fig.add_subplot(grid[rows, :])
    ax.plot(
        np.arange(1, len(coverage) + 1), coverage, color="tab:blue", label="coverage"
    )
    ax.axhline(
        0.95, color="tab:green", linestyle="--", label="success threshold (0.95)"
    )
    if start is not None:
        ax.axhline(start, color="tab:orange", linestyle=":", label="start coverage")
    for step in frames:
        if step >= 1:
            ax.axvline(step, color="tab:gray", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("coverage")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def format_stats(policy: str, s: dict, match) -> str:
    verdict = {
        True: "yes",
        False: "NO",
        None: "not checked (seed not in the saved report)",
    }[match]
    start = "?" if s["start_coverage"] is None else f"{s['start_coverage']:.3f}"
    return "\n".join(
        [
            f"policy {policy}, seed {s['seed']}: {s['steps']} steps",
            f"  coverage: start {start}, peak {s['max_coverage']:.3f} at step "
            f"{s['step_of_max']}, final {s['final_coverage']:.3f} "
            f"(drop from peak {s['drop_from_peak']:.3f})",
            f"  agent: path {s['agent_path_px']:.0f} px, standing still on "
            f"{100 * s['stalled_fraction']:.0f}% of steps, mean gap between commanded "
            f"target and agent {s['mean_action_gap_px']:.0f} px, commanded target within "
            f"{BORDER_PX:.0f} px of the edge on {100 * s['border_action_fraction']:.0f}% of steps",
            f"  block: moved {s['block_net_displacement_px']:.0f} px net "
            f"({s['block_path_px']:.0f} px travelled); contact on "
            f"{100 * s['contact_step_fraction']:.0f}% of steps",
            f"  replay matches the saved evaluation trace: {verdict}",
        ]
    )


def build_policy(name: str, env, checkpoint):
    if name == "bc_cnn":
        model, _ = load_checkpoint(checkpoint, device="cpu")
        return BCPolicy(model, device="cpu")
    if name == "random":
        return ev.SeededRandomPolicy(env.action_space)
    if name == "hold":
        return ev.HoldStillPolicy()
    raise ValueError(f"unknown policy {name!r}; choose from {POLICY_NAMES}")


def main(argv=None, *, env_factory=ev.make_eval_env) -> int:
    parser = argparse.ArgumentParser(
        description="Replay and inspect evaluation episodes"
    )
    parser.add_argument("--policy", choices=POLICY_NAMES, default="bc_cnn")
    parser.add_argument("--seed", type=int, nargs="+", required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/bc_cnn.pt")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--frame-steps", type=int, nargs="+", default=list(DEFAULT_FRAME_STEPS)
    )
    args = parser.parse_args(argv)

    report_path = args.report_dir / f"eval_{args.policy}.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    env = env_factory()
    mismatch = False
    try:
        policy = build_policy(args.policy, env, args.checkpoint)
        for seed in args.seed:
            ep = replay_episode(env, policy, seed, args.frame_steps)
            stats = behavior_stats(ep)
            match = None if report is None else replay_matches_report(ep, report)
            mismatch = mismatch or match is False
            args.report_dir.mkdir(parents=True, exist_ok=True)
            stem = args.report_dir / f"inspect_{args.policy}_{seed}"
            save_inspection_figure(
                ep, stem.with_suffix(".png"), f"{args.policy}, seed {seed}"
            )
            stem.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "policy": args.policy,
                        "stats": stats,
                        "replay_matches_saved_trace": match,
                        "frame_steps": sorted(ep["frames"]),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(format_stats(args.policy, stats, match))
            print(f"  saved {stem.with_suffix('.png')} and {stem.with_suffix('.json')}")
    finally:
        env.close()
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
