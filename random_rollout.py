"""Milestone 0: run a random policy on PushT and report what happens.

Run:  python random_rollout.py --episodes 10

Seed convention: evaluation episodes use seeds from --seed-start upward
(default 10000). Demonstration collection will use a lower range, so training
and evaluation never share seeds.
"""

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  (importing registers the environment)
from PIL import Image

ENV_ID = "gym_pusht/PushT-v0"
DEFAULT_EVAL_SEED_START = 10_000


def make_env(render: bool = False):
    """Create the PushT environment with state observations."""
    return gym.make(
        ENV_ID,
        obs_type="state",
        render_mode="rgb_array" if render else None,
    )


class RandomPolicy:
    """Samples uniformly from the action space, reproducibly for a given seed."""

    def __init__(self, action_space, seed: int):
        self.action_space = action_space
        self.action_space.seed(seed)

    def __call__(self, obs):
        return self.action_space.sample()


def episode_success(terminated: bool, info: dict) -> bool:
    """Use the environment's own success flag if it reports one, else 'terminated'."""
    return bool(info.get("is_success", terminated))


def run_episode(env, policy, seed: int, max_steps: int = 1000) -> dict:
    """Run one episode and return a plain dict of what happened."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while steps < max_steps and not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total_reward += float(reward)
        steps += 1
    return {
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": episode_success(terminated, info),
        "info_keys": sorted(info.keys()),
    }


def summarize(results: list) -> dict:
    """Aggregate per-episode results."""
    n = len(results)
    if n == 0:
        raise ValueError("no episodes to summarize")
    return {
        "episodes": n,
        "successes": sum(1 for r in results if r["success"]),
        "mean_steps": sum(r["steps"] for r in results) / n,
        "mean_reward": sum(r["total_reward"] for r in results) / n,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Random policy on PushT")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_EVAL_SEED_START)
    parser.add_argument(
        "--save-frame",
        type=Path,
        default=None,
        help="save the last rendered frame of episode 0 as a PNG",
    )
    args = parser.parse_args(argv)

    env = make_env(render=args.save_frame is not None)
    results = []
    start = time.perf_counter()
    try:
        for i in range(args.episodes):
            seed = args.seed_start + i
            policy = RandomPolicy(env.action_space, seed)
            result = run_episode(env, policy, seed)
            results.append(result)
            print(
                f"episode {i}: seed={seed} steps={result['steps']} "
                f"reward={result['total_reward']:.2f} success={result['success']} "
                f"terminated={result['terminated']} truncated={result['truncated']}"
            )
            if i == 0 and args.save_frame is not None:
                args.save_frame.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(env.render()).save(args.save_frame)
                print(f"saved frame to {args.save_frame}")
    finally:
        env.close()
    elapsed = time.perf_counter() - start

    summary = summarize(results)
    total_steps = sum(r["steps"] for r in results)
    print()
    print(f"info keys on the last step: {results[0]['info_keys']}")
    print(
        f"success: {summary['successes']}/{summary['episodes']} episodes "
        f"(seeds {args.seed_start} to {args.seed_start + args.episodes - 1})"
    )
    print(
        f"mean steps: {summary['mean_steps']:.1f}  mean reward: {summary['mean_reward']:.2f}"
    )
    print(
        f"{total_steps} steps in {elapsed:.1f}s ({total_steps / elapsed:.0f} steps/s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
