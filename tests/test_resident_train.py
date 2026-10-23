"""Tests for resident_train.py.

The history helpers run anywhere. The training tests need torch (skipped without it) and use a
tiny network on a small fake dataset on the CPU: train_resident must give the same per-epoch
losses as a plain DataLoader loop, including a short last batch and the shuffling order.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import resident_train as rt


# ---------------------------------------------------------------- pure


def test_find_history_nested():
    history = {"train_loss": [1.0], "val_loss": [2.0]}
    assert rt.find_history({"a": {"b": [history]}}) is history
    assert rt.find_history({"a": 1, "b": {"c": "x"}}) is None
    assert rt.find_history({"train_loss": [1.0]}) is None  # needs val_loss too


def test_max_relative_difference():
    assert rt.max_relative_difference([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert rt.max_relative_difference([1.1, 2.0], [1.0, 2.0]) == pytest.approx(0.1)
    # measured against the second argument
    assert rt.max_relative_difference([2.0], [4.0]) == pytest.approx(0.5)


@pytest.mark.parametrize("a,b", [([], []), ([1.0], [1.0, 2.0])])
def test_max_relative_difference_rejects_bad_lengths(a, b):
    with pytest.raises(ValueError):
        rt.max_relative_difference(a, b)


def test_compare_histories():
    a = {"train_loss": [1.0, 0.5], "val_loss": [1.0, 0.6], "val_mae_px": [10.0, 8.0]}
    b = {"train_loss": [1.0, 0.5], "val_loss": [1.0, 0.5], "val_mae_px": [10.0, 7.5]}
    c = rt.compare_histories(a, b)
    assert c["epochs"] == 2
    assert c["train_loss_max_rel"] == 0.0
    assert c["val_loss_max_rel"] == pytest.approx(0.2)
    assert c["final_val_mae_px_abs"] == pytest.approx(0.5)


def test_compare_histories_without_mae():
    a = {"train_loss": [1.0], "val_loss": [1.0]}
    assert "final_val_mae_px_abs" not in rt.compare_histories(a, a)


def test_equivalence_verdict_rule():
    # threshold = 3 * max(rerun, 1e-3)
    assert rt.equivalence_verdict(0.01, 0.029) == "equivalent"
    assert rt.equivalence_verdict(0.01, 0.031) == "not equivalent"
    # a rerun that matches exactly still leaves the floor: 3 * 1e-3
    assert rt.equivalence_verdict(0.0, 0.0029) == "equivalent"
    assert rt.equivalence_verdict(0.0, 0.0031) == "not equivalent"


# ---------------------------------------------------------------- with torch


def _torch():
    return pytest.importorskip("torch")


def _tiny_net_class():
    torch = _torch()

    class TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 8 * 8 + 2, 4 * 2)

        def forward(self, image, agent_pos):
            return self.fc(torch.cat([image.flatten(1), agent_pos], dim=1)).view(-1, 4, 2)

    return TinyNet


def masked_mse(pred, target, mask):
    weight = mask.unsqueeze(-1)
    return (((pred - target) ** 2) * weight).sum() / (weight.sum() * pred.shape[-1]).clamp_min(1.0)


def _seed(seed):
    import torch

    torch.manual_seed(seed)


def _setup():
    torch = _torch()
    from test_resident_data import make_fake
    import resident_data as rd

    arrays, ds = make_fake()  # 30 frames; batch 8 leaves a short last batch of 6
    res = rd.build_resident(arrays, ds, torch.device("cpu"), coord_scale=512.0)
    return torch, ds, res


def test_train_resident_matches_a_dataloader_loop():
    torch, ds, res = _setup()
    TinyNet = _tiny_net_class()
    epochs, batch, lr, seed = 4, 8, 1e-2, 3

    _seed(seed)
    model_a = TinyNet()
    history = rt.train_resident(
        model_a, res, res, epochs=epochs, batch_size=batch, lr=lr, seed=seed,
        log=lambda *_: None, loss_fn=masked_mse, seed_fn=_seed, coord_scale=512.0,
    )

    from torch.utils.data import DataLoader

    _seed(seed)
    model_b = TinyNet()
    _seed(seed)
    loader = DataLoader(
        ds, batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    optimizer = torch.optim.Adam(model_b.parameters(), lr=lr)
    expected = []
    for _ in range(epochs):
        running = seen = 0.0
        for b in loader:
            loss = masked_mse(model_b(b["image"], b["agent_pos"]), b["actions"], b["mask"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps = float(b["mask"].sum())
            running += float(loss.detach()) * steps
            seen += steps
        expected.append(running / seen)

    assert history["train_loss"] == pytest.approx(expected, rel=1e-6)
    assert history["epoch"] == [1, 2, 3, 4]
    assert history["train_loss"][-1] < history["train_loss"][0]  # it is learning


def test_evaluate_resident_matches_a_manual_computation():
    torch, ds, res = _setup()
    TinyNet = _tiny_net_class()
    _seed(0)
    model = TinyNet()
    got = rt.evaluate_resident(model, res, batch_size=8, coord_scale=512.0)

    model.eval()
    squared = chunk_abs = first_abs = valid = 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            error = model(item["image"][None], item["agent_pos"][None])[0] - item["actions"]
            weight = item["mask"].unsqueeze(-1)
            squared += float(((error**2) * weight).sum())
            chunk_abs += float((error.abs() * weight).sum())
            first_abs += float(error[0].abs().sum())
            valid += float(item["mask"].sum())
    n = len(ds)
    assert got["loss"] == pytest.approx(squared / (valid * 2), rel=1e-5)
    assert got["first_mae_px"] == pytest.approx(first_abs / (n * 2) * 512.0, rel=1e-5)
    assert got["chunk_mae_px"] == pytest.approx(chunk_abs / (valid * 2) * 512.0, rel=1e-5)


def test_train_resident_rejects_bad_arguments():
    torch, ds, res = _setup()
    TinyNet = _tiny_net_class()
    kwargs = dict(log=lambda *_: None, loss_fn=masked_mse, seed_fn=_seed, coord_scale=512.0)
    with pytest.raises(ValueError):
        rt.train_resident(TinyNet(), res, res, epochs=0, **kwargs)
    with pytest.raises(ValueError):
        rt.train_resident(TinyNet(), res, res, epochs=1, batch_size=0, **kwargs)
    with pytest.raises(ValueError):
        rt.train_resident(TinyNet(), res, res, epochs=1, lr=0.0, **kwargs)
