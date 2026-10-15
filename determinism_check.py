"""Is a policy's evaluation reproducible, and if not, where does it first diverge?

Background. tests/test_chunk_policy.py::test_real_checkpoint_evaluation_is_reproducible
failed once: two evaluations of the same chunked checkpoint on the same seeds, in one process,
gave different episodes, while the equivalent test for the one-step baseline passed (and every
replay of the baseline matched its saved evaluation). Every comparison in this project assumes
an evaluation can be repeated, so the cause had to be found.

What was found: the simulated state (pictures, agent positions, block poses) and the actions are
bit-for-bit identical between runs, but the coverage computed from that state occasionally differs
in its last bit (5.55e-17 in the observed cases, at a single step, with later steps equal). That
cannot reach a policy, which sees only the picture and the agent position. compare_recordings
therefore treats coverage differences up to COVERAGE_TOLERANCE as equal and reports their size and
number, while states, pictures and actions must still match exactly.

Three checks, each separating a possible cause:
  1. output repeatability: one observation is shown to the same policy many times and to
     freshly built policies; any difference means the network's output is not bit-reproducible
  2. two independent runs of the same seed (fresh environment and fresh policy each time),
     compared step by step on the picture, the agent and block positions, the action and the
     coverage. The first difference decides the probable cause: if the action differs while the
     picture and positions before that step are identical, the policy produced different output
     for the same input; if the picture or a position differs first (or at the same step as the
     action), the environment diverged given identical earlier actions
  3. the actions recorded in the first run replayed into a fresh environment: if that
     reproduces the first run, the environment is deterministic given the actions

Notes: for a chunked policy a differing action can come from a prediction made up to
n_exec - 1 steps earlier than the step where the action first differs. Settings that might
matter can be set from the command line (--threads, --deterministic) and are restored
afterwards.

Run:  python determinism_check.py --policy chunk_cnn --seed 10000 10001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import evaluate as ev
import inspect_episode as ie
from chunk_policy import ChunkPolicy
from pusht_data import COORD_SCALE
from train_chunk import load_chunk_checkpoint

POLICY_KINDS = ie.POLICY_NAMES + ("chunk_cnn",)
DEFAULT_CHECKPOINTS = {
    "bc_cnn": Path("checkpoints/bc_cnn.pt"),
    "chunk_cnn": Path("checkpoints/chunk_cnn.pt"),
}
COLUMNS = ("pixel_hash", "agent_pos", "block_pose", "action", "coverage")
STATE_COLUMNS = ("pixel_hash", "agent_pos", "block_pose")
COVERAGE_TOLERANCE = (
    1e-13  # observed last-bit differences are 5.55e-17; real divergences are far larger
)


def pixel_hash(pixels) -> str:
    return hashlib.sha1(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def record_episode(env, policy, seed, max_steps: int = ev.MAX_STEPS) -> dict:
    """Run one seeded episode and record, for every step, the state the policy saw (picture
    hash, agent position, block pose, all BEFORE the step), the action it took and the
    coverage after the step."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    hashes, agents, blocks, actions, coverage = [], [], [], [], []
    terminated = truncated = False
    while len(actions) < max_steps and not (terminated or truncated):
        hashes.append(pixel_hash(obs["pixels"]))
        agents.append(np.array(obs["agent_pos"], dtype=np.float64))
        blocks.append(
            np.array(info["block_pose"], dtype=np.float64)
        )  # copied before the step
        action = np.array(policy(obs), dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        actions.append(action)
        coverage.append(float(info["coverage"]))
    return {
        "seed": int(seed),
        "steps": len(actions),
        "pixel_hash": hashes,
        "agent_pos": np.array(agents),
        "block_pose": np.array(blocks),
        "action": np.array(actions),
        "coverage": np.array(coverage),
    }


def _first_difference(x, y):
    """First index at which two sequences differ exactly, or None. Sequences of different
    length differ at the length of the shorter one."""
    n = min(len(x), len(y))
    for t in range(n):
        if not np.array_equal(x[t], y[t]):
            return t
    return None if len(x) == len(y) else n


def _first_difference_close(x, y, tolerance):
    """Like _first_difference, for numbers that may differ by up to tolerance. A NaN counts
    as a difference."""
    n = min(len(x), len(y))
    for t in range(n):
        if not abs(float(x[t]) - float(y[t])) <= tolerance:
            return t
    return None if len(x) == len(y) else n


def compare_recordings(
    a: dict, b: dict, coverage_tolerance: float = COVERAGE_TOLERANCE
) -> dict:
    """Compare two recorded episodes step by step and name the probable cause of any
    difference: 'policy' if the action differed while the state before that step was still
    identical, otherwise 'environment'. Coverage may differ by up to coverage_tolerance.
    """
    firsts = {}
    for column in COLUMNS:
        if column == "coverage":
            firsts[column] = _first_difference_close(
                a[column], b[column], coverage_tolerance
            )
        else:
            firsts[column] = _first_difference(a[column], b[column])
    state_steps = [firsts[c] for c in STATE_COLUMNS if firsts[c] is not None]
    t_state = min(state_steps) if state_steps else None
    t_action = firsts["action"]
    differs = any(v is not None for v in firsts.values())
    if not differs:
        cause = None
    elif t_action is not None and (t_state is None or t_action < t_state):
        cause = "policy"
    else:
        cause = "environment"
    cov_a = np.asarray(a["coverage"], dtype=np.float64)
    cov_b = np.asarray(b["coverage"], dtype=np.float64)
    n = min(len(cov_a), len(cov_b))
    gap = np.abs(cov_a[:n] - cov_b[:n])
    return {
        "steps_a": int(a["steps"]),
        "steps_b": int(b["steps"]),
        "identical": not differs,
        "first_difference": firsts,
        "first_state_difference": t_state,
        "first_action_difference": t_action,
        "cause": cause,
        "coverage_max_abs_difference": float(gap.max()) if n else 0.0,
        "coverage_steps_differing_at_all": int((gap > 0).sum()),
    }


class ReplayPolicy:
    """Returns a recorded sequence of actions, ignoring the observations."""

    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.index = 0

    def reset(self, seed=None) -> None:
        self.index = 0

    def __call__(self, obs):
        action = self.actions[self.index]
        self.index += 1
        return action


def first_observation(env_factory, seed) -> dict:
    env = env_factory()
    try:
        obs, _ = env.reset(seed=seed)
        return {
            "pixels": np.array(obs["pixels"]),
            "agent_pos": np.array(obs["agent_pos"]),
        }
    finally:
        env.close()


def output_repeatability(make_policy, obs, repeats: int = 20, fresh: int = 3) -> dict:
    """Largest difference (pixels) between the first actions a policy returns for one
    observation: over repeated calls on one policy, and over freshly built policies."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    if fresh < 1:
        raise ValueError("fresh must be at least 1")

    def first_action(policy):
        if hasattr(policy, "reset"):
            policy.reset(0)
        return np.array(policy(obs), dtype=np.float64)

    policy = make_policy()
    outputs = [first_action(policy) for _ in range(repeats)]
    repeated = max(float(np.abs(o - outputs[0]).max()) for o in outputs)
    built = [first_action(make_policy()) for _ in range(fresh)]
    fresh_diff = max(float(np.abs(o - outputs[0]).max()) for o in built)
    return {
        "calls": int(repeats),
        "fresh_loads": int(fresh),
        "repeated_calls_max_abs_diff": repeated,
        "fresh_loads_max_abs_diff": fresh_diff,
    }


def _record_fresh(env_factory, policy, seed, max_steps) -> dict:
    env = env_factory()
    try:
        return record_episode(env, policy, seed, max_steps)
    finally:
        env.close()


def check_seed(env_factory, make_policy, seed, max_steps: int = ev.MAX_STEPS) -> dict:
    """Two independent runs of a seed, and a replay of the first run's actions."""
    a = _record_fresh(env_factory, make_policy(), seed, max_steps)
    b = _record_fresh(env_factory, make_policy(), seed, max_steps)
    c = _record_fresh(env_factory, ReplayPolicy(a["action"]), seed, a["steps"])
    return {
        "seed": int(seed),
        "run_a_vs_run_b": compare_recordings(a, b),
        "replay_of_a_actions_vs_run_a": compare_recordings(a, c),
    }


def build_policy(kind, checkpoint, env_factory=ev.make_eval_env, n_exec=None):
    """A fresh policy of the given kind (a freshly loaded network for the trained ones).
    env_factory is accepted for compatibility and is not used: the random policy builds the
    same action space the environment has, Box(0, 512, (2,), float32)."""
    if kind == "chunk_cnn":
        model, ckpt = load_chunk_checkpoint(checkpoint, device="cpu")
        used = ckpt["config"]["n_exec"] if n_exec is None else n_exec
        return ChunkPolicy(model, used, device="cpu")
    if kind == "random":
        return ev.SeededRandomPolicy(
            gym.spaces.Box(0.0, float(COORD_SCALE), (2,), np.float32)
        )
    if kind not in ie.POLICY_NAMES:
        raise ValueError(f"unknown policy {kind!r}; choose from {POLICY_KINDS}")
    return ie.build_policy(kind, None, checkpoint)


def _describe(cmp: dict) -> str:
    if cmp["identical"]:
        text = f"identical over {cmp['steps_a']} steps"
        if cmp["coverage_steps_differing_at_all"]:
            text += (
                f" (coverage differed in its last bits at {cmp['coverage_steps_differing_at_all']} "
                f"step(s), by at most {cmp['coverage_max_abs_difference']:.2g})"
            )
        return text
    where = ", ".join(
        f"{name} {t}" for name, t in cmp["first_difference"].items() if t is not None
    )
    return (
        f"DIFFERENT (first difference at step: {where}); probable cause: {cmp['cause']}"
    )


def format_report(
    policy: str, settings: dict, repeatability: dict, results: list
) -> str:
    lines = [
        f"determinism check for {policy}: torch threads {settings['torch_threads']}, "
        f"deterministic algorithms {'on' if settings['deterministic_algorithms'] else 'off'}",
        f"output repeatability (first action, reset observation of seed {repeatability['seed']}): "
        f"{repeatability['calls']} calls on one policy differ by at most "
        f"{repeatability['repeated_calls_max_abs_diff']:.3g} px, {repeatability['fresh_loads']} "
        f"freshly built policies by at most {repeatability['fresh_loads_max_abs_diff']:.3g} px",
    ]
    for r in results:
        lines.append(
            f"seed {r['seed']}: two independent runs: {_describe(r['run_a_vs_run_b'])}"
        )
        lines.append(
            f"seed {r['seed']}: run A's recorded actions replayed in a fresh environment: "
            f"{_describe(r['replay_of_a_actions_vs_run_a'])}"
        )
    lines.append(
        "reading guide: cause 'policy' means the action differed while the state before that step "
        "was identical; cause 'environment' means the picture or a position differed first, or at "
        "the same step as the action; coverage differences below "
        f"{COVERAGE_TOLERANCE:g} are called identical and reported separately; if the replay is "
        "identical the environment is deterministic given its actions; for a chunked policy a "
        "differing action may come from a prediction made up to n_exec - 1 steps earlier"
    )
    return "\n".join(lines)


def main(
    argv=None, *, env_factory=ev.make_eval_env, policy_builder=build_policy
) -> int:
    parser = argparse.ArgumentParser(
        description="Is a policy's evaluation reproducible?"
    )
    parser.add_argument("--policy", choices=POLICY_KINDS, default="chunk_cnn")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, nargs="+", default=[ev.EVAL_SEED_START])
    parser.add_argument("--max-steps", type=int, default=ev.MAX_STEPS)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--fresh", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        raise ValueError("threads must be at least 1")

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS.get(args.policy)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        if args.threads is not None:
            torch.set_num_threads(args.threads)
        if args.deterministic:
            torch.use_deterministic_algorithms(True)
        settings = {
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch": str(torch.__version__),
        }

        def make_policy():
            return policy_builder(args.policy, checkpoint, env_factory)

        obs = first_observation(env_factory, args.seed[0])
        repeatability = output_repeatability(make_policy, obs, args.repeats, args.fresh)
        repeatability["seed"] = int(args.seed[0])
        results = [
            check_seed(env_factory, make_policy, s, args.max_steps) for s in args.seed
        ]
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)

    suffix = "" if args.threads is None else f"_threads{args.threads}"
    suffix += "_deterministic" if args.deterministic else ""
    out = args.out or Path("reports") / f"determinism_{args.policy}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "policy": args.policy,
                "checkpoint": None if checkpoint is None else str(checkpoint),
                "settings": settings,
                "coverage_tolerance": COVERAGE_TOLERANCE,
                "repeatability": repeatability,
                "seeds": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(format_report(args.policy, settings, repeatability, results))
    print(f"saved {out}")
    return 0 if all(r["run_a_vs_run_b"]["identical"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
