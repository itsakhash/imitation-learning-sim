"""Loader for the public lerobot/pusht human demonstrations.

The dataset (pinned revision below) keeps all 206 episodes in one parquet file
(agent position, action, reward and flags) and all images in one AV1 video,
with video frame i belonging to parquet row i. It does NOT store the block's pose.

Commands:
    python pusht_data.py download   # fetch the pinned files into data/raw/lerobot_pusht
    python pusht_data.py report     # print dataset statistics and save them as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

DATASET_REPO = "lerobot/pusht"
DATASET_REVISION = "7628202a2180972f291ba1bc6723834921e72c19"
DEFAULT_ROOT = Path(__file__).resolve().parent / "data" / "raw" / "lerobot_pusht"
DATA_FILE = "data/chunk-000/file-000.parquet"
EPISODES_FILE = "meta/episodes/chunk-000/file-000.parquet"
VIDEO_FILE = "videos/observation.image/chunk-000/file-000.mp4"
IMAGE_SIZE = 96
COORD_SCALE = 512.0  # positions and actions live in a 512 x 512 pixel world


class DatasetError(ValueError):
    """The data on disk does not look like the dataset this code expects."""


@dataclass(frozen=True)
class PushTArrays:
    images: np.ndarray  # (N, 96, 96, 3) uint8
    states: np.ndarray  # (N, 2) float32, agent position in world pixels
    actions: np.ndarray  # (N, 2) float32, target position in world pixels
    episode_index: np.ndarray  # (N,) int64
    frame_index: np.ndarray  # (N,) int64
    reward: np.ndarray  # (N,) float32, definition not documented by the dataset
    done: np.ndarray  # (N,) bool
    success: np.ndarray  # (N,) bool
    episode_lengths: np.ndarray  # (E,) int64, from the episodes metadata table

    @property
    def n_frames(self) -> int:
        return len(self.states)

    @property
    def n_episodes(self) -> int:
        return len(self.episode_lengths)


def read_table(root: Path) -> dict:
    """Read the per-frame parquet file into plain numpy arrays."""
    t = pq.read_table(Path(root) / DATA_FILE).to_pydict()
    return {
        "states": np.asarray(t["observation.state"], dtype=np.float32),
        "actions": np.asarray(t["action"], dtype=np.float32),
        "episode_index": np.asarray(t["episode_index"], dtype=np.int64),
        "frame_index": np.asarray(t["frame_index"], dtype=np.int64),
        "reward": np.asarray(t["next.reward"], dtype=np.float32),
        "done": np.asarray(t["next.done"], dtype=bool),
        "success": np.asarray(t["next.success"], dtype=bool),
    }


def read_episode_lengths(root: Path) -> np.ndarray:
    """Episode lengths from the metadata table, ordered by episode index."""
    t = pq.read_table(
        Path(root) / EPISODES_FILE, columns=["episode_index", "length"]
    ).to_pydict()
    ids = np.asarray(t["episode_index"], dtype=np.int64)
    lengths = np.asarray(t["length"], dtype=np.int64)
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(len(ids))):
        raise DatasetError("episode metadata must list episodes 0..E-1 exactly once")
    return lengths[order]


def decode_video(path: Path) -> np.ndarray:
    """Decode every frame of the video into an (N, H, W, 3) uint8 array."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"missing {path}; run: python pusht_data.py download")
    import av

    with av.open(str(path)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    return np.stack(frames)


def validate_arrays(a: PushTArrays) -> None:
    """Raise DatasetError unless the arrays are internally consistent."""
    n = len(a.states)
    if n == 0:
        raise DatasetError("dataset is empty")
    if a.states.ndim != 2 or a.states.shape[1] != 2:
        raise DatasetError(f"states have shape {a.states.shape}; expected (N, 2)")
    if a.images.shape != (n, IMAGE_SIZE, IMAGE_SIZE, 3) or a.images.dtype != np.uint8:
        raise DatasetError(
            f"images have shape {a.images.shape} and dtype {a.images.dtype}; "
            f"expected ({n}, {IMAGE_SIZE}, {IMAGE_SIZE}, 3) uint8"
        )
    for name in (
        "actions",
        "episode_index",
        "frame_index",
        "reward",
        "done",
        "success",
    ):
        rows = len(getattr(a, name))
        if rows != n:
            raise DatasetError(f"{name} has {rows} rows but states has {n}")
    if a.actions.shape != (n, 2):
        raise DatasetError(f"actions have shape {a.actions.shape}; expected ({n}, 2)")
    if not (np.isfinite(a.states).all() and np.isfinite(a.actions).all()):
        raise DatasetError("states or actions contain non-finite values")
    if a.actions.min() < 0 or a.actions.max() > COORD_SCALE:
        raise DatasetError(f"actions fall outside [0, {COORD_SCALE}]")
    steps = np.diff(a.episode_index)
    if a.episode_index[0] != 0 or not np.isin(steps, (0, 1)).all():
        raise DatasetError(
            "episode_index must start at 0 and grow by at most 1 per row"
        )
    counts = np.bincount(a.episode_index)
    if not np.array_equal(counts, a.episode_lengths):
        raise DatasetError(
            "episode lengths in the metadata disagree with the frame table"
        )
    expected = np.concatenate([np.arange(c) for c in counts])
    if not np.array_equal(a.frame_index, expected):
        raise DatasetError("frame_index must count 0..length-1 within each episode")


def load_arrays(
    root: Path = DEFAULT_ROOT,
    decoder: Callable[[Path], np.ndarray] = decode_video,
) -> PushTArrays:
    """Load and validate the dataset. The decoder is injectable for testing."""
    root = Path(root)
    for rel in (DATA_FILE, EPISODES_FILE):
        if not (root / rel).exists():
            raise DatasetError(
                f"missing {root / rel}; run: python pusht_data.py download"
            )
    arrays = PushTArrays(
        images=decoder(root / VIDEO_FILE),
        episode_lengths=read_episode_lengths(root),
        **read_table(root),
    )
    validate_arrays(arrays)
    return arrays


def download_dataset(
    root: Path = DEFAULT_ROOT, *, snapshot=None, single_file=None
) -> Path:
    """Fetch the pinned dataset files (no login needed; it is a public dataset)."""
    if snapshot is None or single_file is None:
        from huggingface_hub import hf_hub_download, snapshot_download

        snapshot = snapshot or snapshot_download
        single_file = single_file or hf_hub_download
    root = Path(root)
    snapshot(
        DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=["meta/*", "data/*"],
        local_dir=str(root),
    )
    single_file(
        DATASET_REPO,
        VIDEO_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=str(root),
    )
    return root


def split_episode_ids(n_episodes: int, val_fraction: float = 0.2, seed: int = 0):
    """Split whole episodes into (train_ids, val_ids). Never splits inside an episode."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    n_val = int(round(n_episodes * val_fraction))
    if n_episodes < 2 or n_val < 1 or n_val >= n_episodes:
        raise ValueError("need at least one training and one validation episode")
    perm = np.random.default_rng(seed).permutation(n_episodes)
    train = sorted(int(i) for i in perm[n_val:])
    val = sorted(int(i) for i in perm[:n_val])
    return train, val


class PushTDataset(Dataset):
    """One (image, agent position) -> action example per frame.

    Images are float32 in [0, 1], channels first. Positions and actions are divided
    by 512 so they lie in [0, 1]. Pass episode_ids to use only some episodes.
    """

    def __init__(
        self, arrays: PushTArrays, episode_ids: Optional[Sequence[int]] = None
    ):
        self.arrays = arrays
        if episode_ids is None:
            self.indices = np.arange(arrays.n_frames)
            return
        ids = np.unique(np.asarray(list(episode_ids), dtype=np.int64))
        if len(ids) == 0:
            raise ValueError("no episodes selected")
        unknown = ids[(ids < 0) | (ids >= arrays.n_episodes)]
        if len(unknown):
            raise ValueError(f"unknown episode ids: {unknown.tolist()}")
        self.indices = np.flatnonzero(np.isin(arrays.episode_index, ids))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        j = int(self.indices[i])
        a = self.arrays
        return {
            "image": torch.from_numpy(a.images[j]).permute(2, 0, 1).float() / 255.0,
            "agent_pos": torch.from_numpy(a.states[j]) / COORD_SCALE,
            "action": torch.from_numpy(a.actions[j]) / COORD_SCALE,
            "episode_index": int(a.episode_index[j]),
            "frame_index": int(a.frame_index[j]),
        }


def dataset_stats(a: PushTArrays) -> dict:
    """Summary statistics of the dataset itself, JSON-serializable."""
    lengths = a.episode_lengths
    starts = np.cumsum(lengths) - lengths
    ep_max_reward = np.maximum.reduceat(a.reward, starts)
    white = (a.images >= 250).all(axis=-1).mean()
    return {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "episodes": int(a.n_episodes),
        "frames": int(a.n_frames),
        "episode_length_min": int(lengths.min()),
        "episode_length_median": float(np.median(lengths)),
        "episode_length_max": int(lengths.max()),
        "action_min": a.actions.min(axis=0).tolist(),
        "action_max": a.actions.max(axis=0).tolist(),
        "state_min": a.states.min(axis=0).tolist(),
        "state_max": a.states.max(axis=0).tolist(),
        "mean_abs_action_minus_state": float(np.abs(a.actions - a.states).mean()),
        "reward_max_per_episode_min": float(ep_max_reward.min()),
        "reward_max_per_episode_median": float(np.median(ep_max_reward)),
        "reward_max_per_episode_max": float(ep_max_reward.max()),
        "success_flags": int(a.success.sum()),
        "done_flags": int(a.done.sum()),
        "image_mean_rgb": a.images.reshape(-1, 3)
        .mean(axis=0, dtype=np.float64)
        .tolist(),
        "image_white_pixel_fraction": float(white),
    }


def main(argv=None, *, loader=load_arrays, downloader=download_dataset) -> int:
    parser = argparse.ArgumentParser(description="lerobot/pusht data tools")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="fetch the pinned dataset files")
    dl.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    rp = sub.add_parser("report", help="print and save dataset statistics")
    rp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    rp.add_argument(
        "--out", type=Path, default=Path("reports/pusht_dataset_stats.json")
    )
    args = parser.parse_args(argv)

    if args.command == "download":
        path = downloader(args.root)
        print(f"downloaded revision {DATASET_REVISION} to {path}")
        return 0

    stats = dataset_stats(loader(args.root))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2) + "\n")
    for key, value in stats.items():
        print(f"{key:34s} {value}")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
