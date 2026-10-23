"""Tests for resident_data.py.

The pure helper runs anywhere. The rest need torch (skipped without it) and use a small fake
dataset that applies the same operations as ChunkDataset.__getitem__, so they do not need the
real dataset. Equality with the real ChunkDataset on the real data is checked by
`python resident_data.py --check`.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import resident_data as rd


# ---------------------------------------------------------------- pure


def test_batch_bounds_cover_everything_in_order():
    assert rd.batch_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert rd.batch_bounds(8, 4) == [(0, 4), (4, 8)]
    assert rd.batch_bounds(3, 256) == [(0, 3)]


@pytest.mark.parametrize("n,b", [(0, 4), (5, 0)])
def test_batch_bounds_reject_bad_sizes(n, b):
    with pytest.raises(ValueError):
        rd.batch_bounds(n, b)


# ---------------------------------------------------------------- with torch


def _torch():
    return pytest.importorskip("torch")


def test_epoch_batches_match_dataloader_for_several_epochs():
    torch = _torch()
    from torch.utils.data import DataLoader

    n, batch, seed = 1000, 256, 7
    loader = DataLoader(
        list(range(n)),
        batch_size=batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    mine = torch.Generator().manual_seed(seed)
    for _ in range(3):
        expected = list(loader)
        got = rd.epoch_batches(n, batch, mine)
        assert len(expected) == len(got) == 4
        assert all(torch.equal(e, g) for e, g in zip(expected, got))
        assert len(got[-1]) == n - 3 * batch  # the last batch is shorter


def test_epoch_batches_visit_every_example_once():
    torch = _torch()
    batches = rd.epoch_batches(50, 8, torch.Generator().manual_seed(0))
    seen = torch.cat(batches)
    assert sorted(seen.tolist()) == list(range(50))


class FakeChunkDataset:
    """Applies the same operations as train_chunk.ChunkDataset.__getitem__ to small arrays."""

    def __init__(self, arrays, chunk_len=4, coord_scale=512.0):
        self.arrays = arrays
        self.chunk_len = chunk_len
        self.coord_scale = coord_scale
        self.indices = arrays.indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        import numpy as np
        import torch

        j = int(self.indices[i])
        a = self.arrays
        frame = int(a.frame_index[j])
        length = int(a.episode_lengths[int(a.episode_index[j])])
        valid = min(self.chunk_len, length - frame)
        actions = np.empty((self.chunk_len, 2), dtype=np.float32)
        actions[:valid] = a.actions[j : j + valid]
        actions[valid:] = a.actions[j + valid - 1]
        mask = np.zeros(self.chunk_len, dtype=np.float32)
        mask[:valid] = 1.0
        return {
            "image": torch.from_numpy(a.images[j]).permute(2, 0, 1).float() / 255.0,
            "agent_pos": torch.from_numpy(a.states[j]) / self.coord_scale,
            "actions": torch.from_numpy(actions) / self.coord_scale,
            "mask": torch.from_numpy(mask),
        }


def make_fake(n_frames=30):
    import numpy as np

    rng = np.random.default_rng(0)
    lengths = np.array([12, 10, 8])  # three episodes, 30 frames in total
    episode_index = np.concatenate([np.full(l, e) for e, l in enumerate(lengths)])
    frame_index = np.concatenate([np.arange(l) for l in lengths])
    arrays = SimpleNamespace(
        images=rng.integers(0, 256, size=(n_frames, 8, 8, 3), dtype=np.uint8),
        states=rng.uniform(0, 512, size=(n_frames, 2)).astype(np.float32),
        actions=rng.uniform(0, 512, size=(n_frames, 2)).astype(np.float32),
        episode_index=episode_index,
        frame_index=frame_index,
        episode_lengths=lengths,
        indices=np.arange(n_frames)[::-1].copy(),  # not the identity, to catch index mix-ups
    )
    return arrays, FakeChunkDataset(arrays)


def test_resident_batches_equal_dataset_batches():
    torch = _torch()
    arrays, ds = make_fake()
    res = rd.build_resident(arrays, ds, torch.device("cpu"), coord_scale=512.0)
    assert len(res) == len(ds) == 30
    assert rd.mismatching_keys(ds, res, [0, 1, 7, 12, 29]) == []


def test_resident_images_are_standard_contiguous_float():
    torch = _torch()
    arrays, ds = make_fake()
    res = rd.build_resident(arrays, ds, torch.device("cpu"), coord_scale=512.0)
    images = res.batch(torch.tensor([3, 4, 5]))["image"]
    assert images.dtype == torch.float32
    assert images.shape == (3, 3, 8, 8)
    assert images.is_contiguous()


def test_mismatch_check_catches_a_broken_tensor():
    """Deliberate-break check: corrupting one value must be reported."""
    torch = _torch()
    arrays, ds = make_fake()
    res = rd.build_resident(arrays, ds, torch.device("cpu"), coord_scale=512.0)
    res.tensors["image"][5, 0, 0, 0] += 1.0
    assert rd.mismatching_keys(ds, res, [5]) == ["image"]
    res.tensors["mask"][2, 0] = 1.0 - res.tensors["mask"][2, 0]
    assert set(rd.mismatching_keys(ds, res, [2, 5])) == {"image", "mask"}


def test_resident_data_rejects_inconsistent_tensors():
    torch = _torch()
    good = {
        "image": torch.zeros(4, 3, 2, 2),
        "agent_pos": torch.zeros(4, 2),
        "actions": torch.zeros(4, 3, 2),
        "mask": torch.zeros(4, 3),
    }
    rd.ResidentData(good)
    with pytest.raises(ValueError):
        rd.ResidentData({**good, "mask": torch.zeros(5, 3)})
    with pytest.raises(ValueError):
        rd.ResidentData({k: v for k, v in good.items() if k != "mask"})
