"""First-step check: does the network answer the environment's pictures the way it answers
the dataset's?

Background. In the simulator the trained network stands still most of the time and its
commanded target collapses onto the agent's own position after the first quarter of an
episode (timeline_survey.py), while on held-out demonstration frames it does no such thing
(offline_progress.py). One explanation is already ruled out: the environment's pictures use
the same pixel coordinates as the dataset's (the agent is where agent_pos says, to 0.06 px).
Two remain:
  2  the environment's crisp pictures differ from the dataset's compressed ones in a way the
     network is sensitive to
  3  the network ends up in states humans never visited and answers those with a near-zero
     displacement

This looks at the very first step of an episode, where state shift cannot have happened yet:
  - the network's commanded target (minus the agent position) on the environment's reset
    pictures, for the evaluation seeds
  - the same network on the first frame of the held-out demonstration episodes
  - the humans' own commanded target on those first frames, for scale
  - how much the commanded target moves when the environment's pictures are degraded towards
    the dataset's compressed look (3x3 blur, +-3 noise, both), with the image statistics of
    each degraded set next to the dataset's

Reading guide, fixed before looking at any results. Pictures are UNLIKELY to explain the early
behaviour if the ratio of the network's mean gap on environment reset pictures to its mean gap
on dataset first frames lies between 0.75 and 1.25 AND each degradation moves the commanded
target by less than 25% of that gap; then the collapse comes later in an episode, which points
to explanation 3. Otherwise pictures are a candidate, and training with augmentation would be
worth trying.

Limits: only the reset state is examined; the evaluation resets and the demonstration first
frames are different samples of starting positions; the degradations are crude stand-ins for
video compression; and this cannot confirm the state-shift explanation, only fail to support
the other one.

Run:  python first_step_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import evaluate as ev
import image_gap
import pusht_data
import train_bc as tb
from offline_progress import NEAR_PX
from pusht_data import COORD_SCALE

NOISE_LEVEL = 3
DEGRADATIONS = ("blur", "noise", "blur+noise")
IMAGE_KEYS = (
    "exact_white_fraction",
    "small_step_fraction",
    "edge_fraction",
    "sharp_edge_share",
)


def _check_frames(frames) -> np.ndarray:
    frames = np.asarray(frames)
    if (
        frames.ndim != 4
        or frames.shape[-1] != 3
        or frames.dtype != np.uint8
        or len(frames) == 0
    ):
        raise ValueError(
            f"expected a non-empty (N, H, W, 3) uint8 array, got shape {frames.shape} "
            f"and dtype {frames.dtype}"
        )
    return frames


def box_blur(frames) -> np.ndarray:
    """3x3 box blur of (N, H, W, 3) uint8 frames, with the edges replicated."""
    frames = _check_frames(frames)
    h, w = frames.shape[1:3]
    padded = np.pad(
        frames.astype(np.float32), ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge"
    )
    blurred = (
        sum(padded[:, i : i + h, j : j + w] for i in range(3) for j in range(3)) / 9.0
    )
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def add_noise(frames, level: int = NOISE_LEVEL, seed: int = 0) -> np.ndarray:
    """Add independent integer noise in [-level, level] to every value, seeded."""
    frames = _check_frames(frames)
    if level < 0:
        raise ValueError("level must not be negative")
    noise = np.random.default_rng(seed).integers(-level, level + 1, size=frames.shape)
    return np.clip(frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def degrade(frames, kind: str, level: int = NOISE_LEVEL, seed: int = 0) -> np.ndarray:
    if kind == "blur":
        return box_blur(frames)
    if kind == "noise":
        return add_noise(frames, level, seed)
    if kind == "blur+noise":
        return add_noise(box_blur(frames), level, seed)
    raise ValueError(f"unknown degradation {kind!r}; choose from {DEGRADATIONS}")


def commanded_displacement(
    model, frames, agent_pos, device="cpu", batch_size: int = 256
):
    """The network's commanded target minus the agent's position, in world pixels, for
    (N, H, W, 3) uint8 frames and (N, 2) agent positions in world pixels. The target is clipped
    to the action range, exactly as the policy used in evaluation does."""
    frames = _check_frames(frames)
    pos = np.asarray(agent_pos, dtype=np.float64)
    if pos.shape != (len(frames), 2):
        raise ValueError(
            f"agent_pos has shape {pos.shape}; expected ({len(frames)}, 2)"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            image = (
                torch.from_numpy(frames[i : i + batch_size]).permute(0, 3, 1, 2).float()
                / 255.0
            )
            scaled = torch.from_numpy(pos[i : i + batch_size] / COORD_SCALE).float()
            pred = (
                model(image.to(device), scaled.to(device))
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            target = np.clip(pred * COORD_SCALE, 0.0, COORD_SCALE)
            out.append(target - pos[i : i + batch_size])
    return np.concatenate(out)


def gap_summary(displacement) -> dict:
    """Distance statistics of (N, 2) displacements from the agent (pixels)."""
    d = np.asarray(displacement, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 2 or len(d) == 0:
        raise ValueError("expected a non-empty (N, 2) array")
    norms = np.linalg.norm(d, axis=1)
    return {
        "n": int(len(d)),
        "mean_gap_px": float(norms.mean()),
        "median_gap_px": float(np.median(norms)),
        "near_copy_fraction": float((norms < NEAR_PX).mean()),
    }


def collect_reset_frames(env, seeds):
    """The environment's image observation and agent position right after reset, per seed."""
    frames, positions = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        frames.append(np.array(obs["pixels"]))
        positions.append(np.array(obs["agent_pos"], dtype=np.float64))
    if not frames:
        raise ValueError("no seeds given")
    return np.stack(frames), np.stack(positions)


def dataset_first_frames(arrays, episode_ids):
    """(images, agent positions, human actions) of frame 0 of each given episode."""
    rows = np.flatnonzero(
        np.isin(arrays.episode_index, list(episode_ids)) & (arrays.frame_index == 0)
    )
    if len(rows) == 0:
        raise ValueError("none of the given episodes has a first frame")
    return (
        arrays.images[rows],
        arrays.states[rows].astype(np.float64),
        arrays.actions[rows].astype(np.float64),
    )


def _image_stats(frames) -> dict:
    stats = image_gap.frame_stats(frames)
    return {key: stats[key] for key in IMAGE_KEYS} | {"frames": stats["frames"]}


def degradation_effect(
    model, frames, agent_pos, kind, level=NOISE_LEVEL, seed=0, device="cpu"
) -> dict:
    """How far degrading the frames moves the commanded target, and what the degraded
    frames' image statistics are."""
    clean = commanded_displacement(model, frames, agent_pos, device)
    degraded_frames = degrade(frames, kind, level, seed)
    degraded = commanded_displacement(model, degraded_frames, agent_pos, device)
    change = np.linalg.norm(degraded - clean, axis=1)
    clean_gap = float(np.linalg.norm(clean, axis=1).mean())
    return {
        "kind": kind,
        "mean_change_px": float(change.mean()),
        "median_change_px": float(np.median(change)),
        "change_share_of_gap": (
            None if clean_gap == 0 else float(change.mean() / clean_gap)
        ),
        "mean_gap_px": float(np.linalg.norm(degraded, axis=1).mean()),
        **_image_stats(degraded_frames),
    }


def analyze(
    model,
    env_frames,
    env_pos,
    ds_frames,
    ds_pos,
    ds_actions,
    reference_images,
    level=NOISE_LEVEL,
    seed=0,
    device="cpu",
) -> dict:
    env_disp = commanded_displacement(model, env_frames, env_pos, device)
    ds_disp = commanded_displacement(model, ds_frames, ds_pos, device)
    human = np.asarray(ds_actions, dtype=np.float64) - np.asarray(
        ds_pos, dtype=np.float64
    )
    env_summary, ds_summary = gap_summary(env_disp), gap_summary(ds_disp)
    return {
        "network_on_environment": env_summary,
        "network_on_dataset": ds_summary,
        "human_on_dataset": gap_summary(human),
        "gap_ratio_environment_over_dataset": (
            None
            if ds_summary["mean_gap_px"] == 0
            else env_summary["mean_gap_px"] / ds_summary["mean_gap_px"]
        ),
        "image_stats": {
            "environment_reset_frames": _image_stats(env_frames),
            "dataset_every_25th_frame": _image_stats(
                np.asarray(reference_images)[::25]
            ),
        },
        "degradations": [
            degradation_effect(model, env_frames, env_pos, kind, level, seed, device)
            for kind in DEGRADATIONS
        ],
        "noise_level": int(level),
    }


def _gap_row(label: str, s: dict) -> str:
    return (
        f"{label:<44s}{s['n']:>7d}{s['mean_gap_px']:>11.1f}{s['median_gap_px']:>12.1f}"
        f"{100 * s['near_copy_fraction']:>13.0f}"
    )


def _image_row(label: str, s: dict, change=None, share=None) -> str:
    tail = f"{'-':>12s}{'-':>12s}"
    if change is not None:
        share_text = "n/a" if share is None else f"{100 * share:.0f}%"
        tail = f"{change:>12.1f}{share_text:>12s}"
    return (
        f"{label:<44s}{s['exact_white_fraction']:>12.4f}{s['small_step_fraction']:>12.4f}"
        f"{s['sharp_edge_share']:>12.4f}{tail}"
    )


def format_report(result: dict) -> str:
    env, ds, human = (
        result[k]
        for k in ("network_on_environment", "network_on_dataset", "human_on_dataset")
    )
    ratio = result["gap_ratio_environment_over_dataset"]
    images = result["image_stats"]
    lines = [
        "first step of an episode: how far from the agent is the commanded target?",
        f"{'':<44s}{'frames':>7s}{'mean px':>11s}{'median px':>12s}{'within 3 px %':>13s}",
        _gap_row("network, environment reset frames", env),
        _gap_row("network, dataset frame 0 (held-out)", ds),
        _gap_row("human, dataset frame 0 (held-out)", human),
        "mean gap on environment frames / mean gap on dataset first frames: "
        + ("n/a" if ratio is None else f"{ratio:.2f}"),
        "",
        "image statistics, and how far degrading the environment's reset frames moves the target",
        f"{'':<44s}{'exact white':>12s}{'small steps':>12s}{'sharp share':>12s}"
        f"{'target moves':>12s}{'% of gap':>12s}",
        _image_row(
            f"environment reset frames ({images['environment_reset_frames']['frames']})",
            images["environment_reset_frames"],
        ),
        _image_row(
            f"dataset, every 25th frame ({images['dataset_every_25th_frame']['frames']})",
            images["dataset_every_25th_frame"],
        ),
    ]
    for d in result["degradations"]:
        lines.append(
            _image_row(
                f"environment frames, {d['kind']}",
                d,
                d["mean_change_px"],
                d["change_share_of_gap"],
            )
        )
    lines += [
        f"noise level +-{result['noise_level']}; 'target moves' is the mean distance (px) between the "
        "commanded target on the clean and on the degraded frame",
        "note: reset states only; the two samples of starting positions differ; the degradations are "
        "crude stand-ins for video compression; this cannot confirm the state-shift explanation",
    ]
    return "\n".join(lines)


def main(
    argv=None,
    *,
    env_factory=ev.make_eval_env,
    arrays_loader=pusht_data.load_arrays,
    checkpoint_loader=tb.load_checkpoint,
) -> int:
    parser = argparse.ArgumentParser(
        description="First-step check of the trained network"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/bc_cnn.pt")
    )
    parser.add_argument("--episodes", type=int, default=ev.MIN_EPISODES)
    parser.add_argument("--noise-level", type=int, default=NOISE_LEVEL)
    parser.add_argument(
        "--out", type=Path, default=Path("reports/first_step_check.json")
    )
    args = parser.parse_args(argv)

    model, ckpt = checkpoint_loader(args.checkpoint, device="cpu")
    arrays = arrays_loader()
    ds_frames, ds_pos, ds_actions = dataset_first_frames(
        arrays, ckpt["split"]["val_episode_ids"]
    )
    env = env_factory()
    try:
        env_frames, env_pos = collect_reset_frames(env, ev.eval_seeds(args.episodes))
    finally:
        env.close()
    result = analyze(
        model,
        env_frames,
        env_pos,
        ds_frames,
        ds_pos,
        ds_actions,
        arrays.images,
        level=args.noise_level,
    )
    result["checkpoint"] = str(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(format_report(result))
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
