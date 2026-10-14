"""Evaluation harness implementing the protocol fixed in PROJECT_CONTEXT.md, section 7.

Per episode: seeded reset, at most 300 steps, coverage recorded after every step.
  primary metric   mean over episodes of the maximum info["coverage"] reached,
                   with a bootstrap 95% confidence interval
  also reported    success rate (info["is_success"], i.e. coverage above 0.95, at any
                   step) with a Wilson 95% interval, and the mean final coverage
Evaluation seeds are 10000 + episode number, the same for every policy. There is no
option to change them, so training and evaluation seeds cannot be mixed up by accident.

Run:  python evaluate.py --policy random --episodes 50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

ENV_ID = "gym_pusht/PushT-v0"
EVAL_SEED_START = 10_000
MAX_STEPS = 300
MIN_EPISODES = 50
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


def eval_seeds(n_episodes: int) -> list:
    """The evaluation seeds: 10000, 10001, ... for every policy."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be at least 1")
    return list(range(EVAL_SEED_START, EVAL_SEED_START + n_episodes))


def make_eval_env(obs_type: str = "pixels_agent_pos"):
    """The PushT environment used for every evaluation."""
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (importing registers the environment)

    return gym.make(ENV_ID, obs_type=obs_type, render_mode="rgb_array")


class SeededRandomPolicy:
    """Uniform random actions, reproducible for a given episode seed."""

    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self, seed: int) -> None:
        self.action_space.seed(seed)

    def __call__(self, obs):
        return self.action_space.sample()


def run_episode(env, policy, seed: int, max_steps: int = MAX_STEPS) -> dict:
    """Run one seeded episode. A policy is a callable obs -> action; if it has a
    reset(seed) method, that is called right after the environment is reset."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    trace = []
    success = False
    terminated = truncated = False
    while len(trace) < max_steps and not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if "coverage" not in info or "is_success" not in info:
            raise KeyError("environment info must contain 'coverage' and 'is_success'")
        trace.append(float(info["coverage"]))  # read now: the env reuses its info dict
        success = success or bool(info["is_success"])
    return {
        "seed": int(seed),
        "steps": len(trace),
        "max_coverage": max(trace),
        "final_coverage": trace[-1],
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "coverage_trace": [round(c, 4) for c in trace],
    }


def evaluate_policy(policy, env, n_episodes: int, max_steps: int = MAX_STEPS) -> list:
    """Run the policy on the standard evaluation seeds."""
    return [
        run_episode(env, policy, seed, max_steps) for seed in eval_seeds(n_episodes)
    ]


def bootstrap_ci(
    values,
    confidence: float = CONFIDENCE,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
):
    """Percentile bootstrap confidence interval for the mean of the values."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or len(v) == 0:
        raise ValueError("values must be a non-empty 1-D sequence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_resamples, len(v)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def wilson_ci(successes: int, n: int, confidence: float = CONFIDENCE):
    """Wilson score interval for a success rate."""
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("need n >= 1 and 0 <= successes <= n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = int(successes) / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize(results: list) -> dict:
    """Headline numbers for a list of episode results."""
    if not results:
        raise ValueError("no episodes to summarize")
    seeds = [r["seed"] for r in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds in results")
    n = len(results)
    max_cov = np.array([r["max_coverage"] for r in results], dtype=np.float64)
    final_cov = np.array([r["final_coverage"] for r in results], dtype=np.float64)
    successes = sum(1 for r in results if r["success"])
    cov_lo, cov_hi = bootstrap_ci(max_cov)
    suc_lo, suc_hi = wilson_ci(successes, n)
    return {
        "n_episodes": n,
        "seed_start": min(seeds),
        "seed_end": max(seeds),
        "meets_minimum_episodes": n >= MIN_EPISODES,
        "mean_max_coverage": float(max_cov.mean()),
        "mean_max_coverage_ci95": [cov_lo, cov_hi],
        "success_count": successes,
        "success_rate": successes / n,
        "success_rate_ci95": [suc_lo, suc_hi],
        "mean_final_coverage": float(final_cov.mean()),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
    }


def build_report(name: str, results: list) -> dict:
    return {
        "policy": name,
        "protocol": {
            "primary_metric": "mean over episodes of the maximum info['coverage'] reached",
            "success": "info['is_success'] (coverage above 0.95) at any step",
            "eval_seed_start": EVAL_SEED_START,
            "max_steps": MAX_STEPS,
            "min_episodes": MIN_EPISODES,
            "confidence": CONFIDENCE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "summary": summarize(results),
        "episodes": results,
    }


def write_report(path: Path, report: dict) -> None:
    """Write the report as JSON, one line per episode so the file stays readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in report.items() if k != "episodes"}
    head_text = json.dumps(head, indent=2)[:-2]  # drop the closing "\n}"
    episodes = ",\n".join("    " + json.dumps(e) for e in report["episodes"])
    path.write_text(f'{head_text},\n  "episodes": [\n{episodes}\n  ]\n}}\n')


def format_summary(name: str, s: dict) -> str:
    cov_lo, cov_hi = s["mean_max_coverage_ci95"]
    suc_lo, suc_hi = s["success_rate_ci95"]
    lines = [
        f"policy: {name}",
        f"episodes: {s['n_episodes']} (seeds {s['seed_start']} to {s['seed_end']})",
        f"mean max coverage: {s['mean_max_coverage']:.4f}  "
        f"95% CI [{cov_lo:.4f}, {cov_hi:.4f}] (bootstrap)",
        f"success: {s['success_count']}/{s['n_episodes']} = {100 * s['success_rate']:.1f}%  "
        f"95% CI [{100 * suc_lo:.1f}%, {100 * suc_hi:.1f}%] (Wilson)",
        f"mean final coverage: {s['mean_final_coverage']:.4f}",
        f"mean steps: {s['mean_steps']:.1f}",
    ]
    if not s["meets_minimum_episodes"]:
        lines.append(
            f"WARNING: fewer than {MIN_EPISODES} episodes. "
            "This is a smoke test, not a reportable result."
        )
    return "\n".join(lines)


POLICIES = {"random": lambda env: SeededRandomPolicy(env.action_space)}


def main(argv=None, *, env_factory=make_eval_env) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a policy on PushT")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="random")
    parser.add_argument("--episodes", type=int, default=MIN_EPISODES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or Path("reports") / f"eval_{args.policy}.json"
    env = env_factory()
    try:
        results = evaluate_policy(POLICIES[args.policy](env), env, args.episodes)
    finally:
        env.close()
    report = build_report(args.policy, results)
    write_report(out, report)
    print(format_summary(args.policy, report["summary"]))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
