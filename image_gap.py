"""Quantify how the dataset's video frames differ from the environment's own images.

The dataset images went through lossy AV1 compression; the environment's images are
rendered directly. A CNN trained on the first and run on the second could face a
distribution shift. These statistics put numbers on the most likely differences:
compression noise (many tiny neighbour-to-neighbour changes in flat areas) and
softness (hard edges spread over several pixels).

They describe how the images were formed, but they also depend a little on scene
content, so read the gap as an indication, not a precise measurement.

Run:  python image_gap.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SMALL_STEP_MAX = 8  # a neighbour change of 1..8 levels counts as small (noise-like)
EDGE_MIN = 20  # a neighbour change of at least this many levels counts as an edge
SHARP_MIN = 100  # an edge change of at least this many levels counts as sharp
NEAR_WHITE = 250
SCALAR_KEYS = (
    "exact_white_fraction",
    "near_white_fraction",
    "small_step_fraction",
    "edge_fraction",
    "sharp_edge_share",
)


def _check_frames(frames: np.ndarray) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(
            f"expected an (N, H, W, 3) uint8 array, got shape {frames.shape} "
            f"and dtype {frames.dtype}"
        )
    if len(frames) == 0:
        raise ValueError("no frames to describe")


def _neighbour_steps(frames: np.ndarray) -> np.ndarray:
    """Largest per-channel change between horizontally or vertically adjacent pixels."""
    f = frames.astype(np.int16)
    dx = np.abs(np.diff(f, axis=2)).max(axis=-1)
    dy = np.abs(np.diff(f, axis=1)).max(axis=-1)
    return np.concatenate([dx.ravel(), dy.ravel()])


def frame_stats(frames: np.ndarray) -> dict:
    """Image-formation statistics for a stack of (N, H, W, 3) uint8 frames."""
    _check_frames(frames)
    steps = _neighbour_steps(frames)
    edges = steps >= EDGE_MIN
    n_edges = int(edges.sum())
    return {
        "frames": int(len(frames)),
        "mean_rgb": frames.reshape(-1, 3).mean(axis=0, dtype=np.float64).tolist(),
        "exact_white_fraction": float((frames == 255).all(axis=-1).mean()),
        "near_white_fraction": float((frames >= NEAR_WHITE).all(axis=-1).mean()),
        "small_step_fraction": float(((steps >= 1) & (steps <= SMALL_STEP_MAX)).mean()),
        "edge_fraction": float(edges.mean()),
        "sharp_edge_share": (
            float((steps >= SHARP_MIN).sum() / n_edges) if n_edges else 0.0
        ),
    }


def compare_stats(dataset: dict, environment: dict) -> dict:
    """Side-by-side values and dataset-minus-environment differences."""
    out = {
        key: {
            "dataset": dataset[key],
            "environment": environment[key],
            "difference": dataset[key] - environment[key],
        }
        for key in SCALAR_KEYS
    }
    out["mean_rgb_difference"] = [
        d - e for d, e in zip(dataset["mean_rgb"], environment["mean_rgb"])
    ]
    return out


def load_dataset_frames(step: int = 25, root: Path | None = None) -> np.ndarray:
    """Every step-th decoded frame of the dataset video."""
    if step < 1:
        raise ValueError("step must be at least 1")
    import pusht_data

    root = Path(root) if root is not None else pusht_data.DEFAULT_ROOT
    return pusht_data.decode_video(root / pusht_data.VIDEO_FILE)[::step]


def collect_env_frames(
    n_episodes: int,
    seed_start: int = 10_000,
    frames_per_episode: int = 3,
    steps_between: int = 10,
) -> np.ndarray:
    """Image observations from the environment under a seeded random policy."""
    if n_episodes < 1 or frames_per_episode < 1 or steps_between < 1:
        raise ValueError(
            "n_episodes, frames_per_episode and steps_between must be >= 1"
        )
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (importing registers the environment)

    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
    )
    frames = []
    try:
        for k in range(n_episodes):
            seed = seed_start + k
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            frames.append(obs["pixels"].copy())
            finished = False
            for _ in range(frames_per_episode - 1):
                for _ in range(steps_between):
                    if finished:
                        break
                    obs, _, terminated, truncated, _ = env.step(
                        env.action_space.sample()
                    )
                    finished = terminated or truncated
                frames.append(obs["pixels"].copy())
    finally:
        env.close()
    return np.stack(frames)


def main(argv=None, *, dataset_loader=None, env_collector=None) -> int:
    parser = argparse.ArgumentParser(description="Dataset frames vs environment frames")
    parser.add_argument("--dataset-step", type=int, default=25)
    parser.add_argument("--env-episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("reports/image_gap.json"))
    args = parser.parse_args(argv)

    dataset_loader = dataset_loader or load_dataset_frames
    env_collector = env_collector or collect_env_frames
    dataset = frame_stats(dataset_loader(args.dataset_step))
    environment = frame_stats(env_collector(args.env_episodes, args.seed_start))
    comparison = compare_stats(dataset, environment)

    report = {
        "settings": {
            "dataset_step": args.dataset_step,
            "env_episodes": args.env_episodes,
            "seed_start": args.seed_start,
            "small_step_max": SMALL_STEP_MAX,
            "edge_min": EDGE_MIN,
            "sharp_min": SHARP_MIN,
            "near_white": NEAR_WHITE,
        },
        "dataset": dataset,
        "environment": environment,
        "comparison": comparison,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"frames: dataset {dataset['frames']}, environment {environment['frames']}")
    for key in SCALAR_KEYS:
        c = comparison[key]
        print(
            f"{key:22s} dataset {c['dataset']:.4f}  "
            f"environment {c['environment']:.4f}  difference {c['difference']:+.4f}"
        )
    print("mean_rgb dataset      ", [round(v, 2) for v in dataset["mean_rgb"]])
    print("mean_rgb environment  ", [round(v, 2) for v in environment["mean_rgb"]])
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
