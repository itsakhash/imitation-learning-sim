"""GPU-resident training data for the chunk-style networks: the same examples, much faster batches.

ChunkDataset builds every example in Python on one CPU core (about 11 ms per batch of 256),
which is most of a training step. Here every example is built once, kept on the GPU, and a
batch is made by indexing. The tensors are produced with the same operations as
ChunkDataset.__getitem__ (in particular, the image is converted to float and divided by 255 on
the CPU, as there), so a batch equals what DataLoader + ChunkDataset would deliver for the same
indices; `mismatching_keys` checks that, and `epoch_order` reproduces the order in which
DataLoader(shuffle=True, generator=...) visits the examples.

Memory: about 2.3 GB for the 165 training episodes (float32 images) plus about 0.6 GB for the
41 validation episodes.

Check on the real data (about 30 seconds, no training):
    python resident_data.py --check
"""

from __future__ import annotations

import argparse
import sys
import time

KEYS = ("image", "agent_pos", "actions", "mask")


# ---------------------------------------------------------------- pure helper (tested)


def batch_bounds(n: int, batch_size: int) -> list:
    """(start, stop) pairs covering range(n) in order; the last batch may be shorter."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]


# ---------------------------------------------------------------- sampling order


def epoch_order(n: int, generator):
    """Visiting order of one epoch, as DataLoader(shuffle=True, generator=generator) draws it.

    DataLoader consumes the generator in two places per epoch: its iterator draws a base seed
    when it is created, and torch's RandomSampler then draws the permutation (and, in torch
    2.14.1, one further permutation after the last index, which is discarded but still moves
    the generator on). The base seed is drawn here; the sampler's own code is used for the rest,
    so any such detail is followed instead of copied. A test compares several epochs with a
    real DataLoader."""
    import torch
    from torch.utils.data import RandomSampler

    if n < 1:
        raise ValueError("n must be at least 1")
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    return torch.tensor(
        list(RandomSampler(range(n), generator=generator)), dtype=torch.int64
    )


def epoch_batches(n: int, batch_size: int, generator) -> list:
    """The index tensors of one shuffled epoch, in DataLoader's order."""
    order = epoch_order(n, generator)
    return [order[start:stop] for start, stop in batch_bounds(n, batch_size)]


# ---------------------------------------------------------------- the data


class ResidentData:
    """All examples of one ChunkDataset as tensors on one device."""

    def __init__(self, tensors: dict):
        missing = [k for k in KEYS if k not in tensors]
        if missing:
            raise ValueError(f"missing tensors: {missing}")
        sizes = {int(tensors[k].shape[0]) for k in KEYS}
        if len(sizes) != 1:
            raise ValueError(
                f"tensors disagree on the number of examples: {sorted(sizes)}"
            )
        self.tensors = {k: tensors[k] for k in KEYS}
        self.n = sizes.pop()
        self.device = self.tensors["image"].device

    def __len__(self) -> int:
        return self.n

    def batch(self, ids) -> dict:
        """The batch for the given example indices (a tensor on the same device)."""
        return {k: self.tensors[k][ids] for k in KEYS}

    def megabytes(self) -> float:
        return sum(t.numel() * t.element_size() for t in self.tensors.values()) / 1e6


def build_resident(arrays, ds, device, coord_scale=None) -> ResidentData:
    """Build every example of `ds` (a ChunkDataset) once and move it to `device`."""
    import numpy as np
    import torch

    if coord_scale is None:
        from pusht_data import COORD_SCALE as coord_scale
    idx = np.asarray(ds.indices)
    # the same CPU operations as ChunkDataset.__getitem__, applied to all frames at once
    image = (
        torch.from_numpy(arrays.images[idx]).permute(0, 3, 1, 2).contiguous().float()
        / 255.0
    )
    agent_pos = torch.from_numpy(arrays.states[idx]) / coord_scale
    actions, masks = [], []
    for i in range(len(ds)):
        item = ds[i]
        actions.append(item["actions"])
        masks.append(item["mask"])
    tensors = {
        "image": image,
        "agent_pos": agent_pos,
        "actions": torch.stack(actions),
        "mask": torch.stack(masks),
    }
    tensors = {k: v.to(device) for k, v in tensors.items()}
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return ResidentData(tensors)


def mismatching_keys(ds, resident: ResidentData, positions) -> list:
    """Keys whose values differ between ChunkDataset batches and the resident ones for the
    examples at `positions`. An empty list means the batches are identical."""
    import torch
    from torch.utils.data import default_collate

    positions = list(positions)
    reference = default_collate([ds[i] for i in positions])
    got = resident.batch(torch.as_tensor(positions, device=resident.device))
    return [k for k in KEYS if not torch.equal(got[k].cpu(), reference[k])]


# ---------------------------------------------------------------- check on the real data


def _load_data():
    """The same arrays, split and datasets as train_chunk.main with its defaults."""
    import pusht_data
    from train_chunk import ChunkDataset

    arrays = pusht_data.load_arrays()
    train_ids, val_ids = pusht_data.split_episode_ids(arrays.n_episodes, 0.2, 0)
    return arrays, ChunkDataset(arrays, train_ids), ChunkDataset(arrays, val_ids)


def check_order(n: int, batch_size: int, seed: int, epochs: int) -> bool:
    """True if epoch_batches equals DataLoader's order for the given number of epochs."""
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        list(range(n)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    mine = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        expected = [b for b in loader]
        got = epoch_batches(n, batch_size, mine)
        if len(expected) != len(got) or not all(
            torch.equal(e, g) for e, g in zip(expected, got)
        ):
            return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GPU-resident chunk data")
    parser.add_argument(
        "--check", action="store_true", help="check against the real data"
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return 0

    import torch

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    t0 = time.perf_counter()
    arrays, train_ds, val_ds = _load_data()
    print(f"loaded the data in {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    train_res = build_resident(arrays, train_ds, device)
    val_res = build_resident(arrays, val_ds, device)
    print(
        f"built resident data on {device} in {time.perf_counter() - t0:.1f} s: "
        f"{len(train_res):,} training and {len(val_res):,} validation examples, "
        f"{train_res.megabytes() + val_res.megabytes():.0f} MB"
    )

    ok = True
    for name, ds, res in (
        ("training", train_ds, train_res),
        ("validation", val_ds, val_res),
    ):
        n = len(ds)
        positions = sorted({p for p in (0, 1, 2, 100, 5000, n // 2, n - 1) if p < n})
        bad = mismatching_keys(ds, res, positions)
        print(
            f"{name}: batches equal ChunkDataset's for {len(positions)} checked examples: "
            + ("yes" if not bad else "NO, differs in " + ", ".join(bad))
        )
        ok = ok and not bad

    same_order = check_order(len(train_ds), 256, seed=0, epochs=3)
    print(
        f"epoch order equals DataLoader's for 3 epochs (seed 0): {'yes' if same_order else 'NO'}"
    )
    ok = ok and same_order

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
