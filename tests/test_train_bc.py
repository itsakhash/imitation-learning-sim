import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import pusht_data
import train_bc as tb
from pusht_data import PushTDataset


def structured_arrays(n_episodes=24, length=16, seed=0, shuffle_labels=False):
    """Fake data where the action is a simple function of the agent position:
    action = 0.5 * position + 128 pixels. Images are blank, so position is the only
    useful input. With shuffle_labels the actions are mismatched with the positions."""
    rng = np.random.default_rng(seed)
    n = n_episodes * length
    states = rng.uniform(0, 512, size=(n, 2)).astype(np.float32)
    actions = (0.5 * states + 128.0).astype(np.float32)
    if shuffle_labels:
        actions = actions[rng.permutation(n)]
    return pusht_data.PushTArrays(
        images=np.zeros((n, 96, 96, 3), dtype=np.uint8),
        states=states,
        actions=actions,
        episode_index=np.repeat(np.arange(n_episodes), length).astype(np.int64),
        frame_index=np.tile(np.arange(length), n_episodes).astype(np.int64),
        reward=np.zeros(n, dtype=np.float32),
        done=np.zeros(n, dtype=bool),
        success=np.zeros(n, dtype=bool),
        episode_lengths=np.full(n_episodes, length, dtype=np.int64),
    )


class ConstantHalf(nn.Module):
    """Always predicts the middle of the world, whatever it is shown."""

    def forward(self, image, agent_pos):
        return torch.full((len(agent_pos), 2), 0.5)


def fit_and_score(arrays, epochs=40, seed=0):
    """Train on CPU with a fixed split; return (validation MAE in px, reference MAEs)."""
    train_ids, val_ids = pusht_data.split_episode_ids(arrays.n_episodes, 0.25, seed=0)
    train_ds = PushTDataset(arrays, train_ids)
    val_ds = PushTDataset(arrays, val_ids)
    torch.manual_seed(seed)
    history = tb.train(
        tb.BCPolicyNet(),
        train_ds,
        val_ds,
        epochs=epochs,
        batch_size=32,
        lr=2e-3,
        seed=seed,
        log=lambda *_: None,
    )
    return history["val_mae_px"][-1], tb.reference_maes(arrays, train_ids, val_ids)


# ------------------------------------------------------------- network, loss


def test_net_output_shape_and_size():
    net = tb.BCPolicyNet()
    out = net(torch.rand(4, 3, 96, 96), torch.rand(4, 2))
    assert out.shape == (4, 2) and torch.isfinite(out).all()
    n_params = sum(p.numel() for p in net.parameters())
    assert 700_000 < n_params < 740_000


def test_net_is_deterministic_in_eval_mode():
    torch.manual_seed(0)
    net = tb.BCPolicyNet().eval()
    image, pos = torch.rand(3, 3, 96, 96), torch.rand(3, 2)
    with torch.no_grad():
        assert torch.equal(net(image, pos), net(image, pos))


def test_bc_loss_values():
    zeros, ones = torch.zeros(4, 2), torch.ones(4, 2)
    assert float(tb.bc_loss(ones, ones)) == 0.0
    assert float(tb.bc_loss(zeros, ones)) == pytest.approx(1.0)
    assert float(tb.bc_loss(torch.full((4, 2), 0.25), ones)) == pytest.approx(0.5625)


def test_pick_device_cpu():
    assert tb.pick_device("cpu").type == "cpu"


def test_pick_device_auto_matches_cuda_availability():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert tb.pick_device("auto").type == expected


# ---------------------------------------------------------------- evaluation


def test_evaluate_loader_known_values():
    arrays = structured_arrays(n_episodes=4, length=5)
    loader = DataLoader(PushTDataset(arrays), batch_size=7)
    result = tb.evaluate_loader(ConstantHalf(), loader, torch.device("cpu"))
    target = arrays.actions.astype(np.float64) / 512.0
    assert result["loss"] == pytest.approx(
        float(((0.5 - target) ** 2).mean()), rel=1e-4
    )
    assert result["mae_px"] == pytest.approx(
        float(np.abs(0.5 - target).mean() * 512.0), rel=1e-4
    )


def test_evaluate_loader_rejects_an_empty_loader():
    with pytest.raises(ValueError):
        tb.evaluate_loader(
            ConstantHalf(), DataLoader([], batch_size=2), torch.device("cpu")
        )


# ------------------------------------------------------------------ training


def test_train_records_one_history_entry_per_epoch():
    arrays = structured_arrays(n_episodes=8, length=8)
    train_ids, val_ids = pusht_data.split_episode_ids(8, 0.25, seed=0)
    lines = []
    history = tb.train(
        tb.BCPolicyNet(),
        PushTDataset(arrays, train_ids),
        PushTDataset(arrays, val_ids),
        epochs=2,
        batch_size=16,
        log=lines.append,
    )
    assert history["epoch"] == [1, 2] and len(lines) == 2
    for key in ("train_loss", "val_loss", "val_mae_px", "seconds"):
        assert len(history[key]) == 2 and np.isfinite(history[key]).all()


@pytest.mark.parametrize(
    "kwargs",
    [dict(epochs=0), dict(epochs=1, batch_size=0), dict(epochs=1, lr=0.0)],
)
def test_train_rejects_bad_arguments(kwargs):
    ds = PushTDataset(structured_arrays(n_episodes=4, length=4))
    with pytest.raises(ValueError):
        tb.train(tb.BCPolicyNet(), ds, ds, log=lambda *_: None, **kwargs)


def test_training_reduces_the_loss():
    arrays = structured_arrays()
    train_ids, val_ids = pusht_data.split_episode_ids(arrays.n_episodes, 0.25, seed=0)
    torch.manual_seed(0)
    history = tb.train(
        tb.BCPolicyNet(),
        PushTDataset(arrays, train_ids),
        PushTDataset(arrays, val_ids),
        epochs=6,
        batch_size=32,
        lr=2e-3,
        seed=0,
        log=lambda *_: None,
    )
    assert history["train_loss"][-1] < history["train_loss"][0]


# -------------------- the checks that fail if the loss or the data code is broken


def test_learns_a_simple_mapping_from_agent_position():
    val_mae, refs = fit_and_score(structured_arrays())
    assert val_mae < 0.5 * refs["mean_action_mae_px"], (val_mae, refs)


def test_shuffled_labels_are_caught():
    # Data code that pairs frames with the wrong actions must NOT look like it learned.
    val_mae, refs = fit_and_score(structured_arrays(shuffle_labels=True))
    assert val_mae >= 0.8 * refs["mean_action_mae_px"], (val_mae, refs)


BROKEN_LOSSES = {
    "zero gradient": lambda pred, target: (pred * 0).sum() + 1.0,
    "inverted targets": lambda pred, target: torch.nn.functional.mse_loss(
        pred, 1.0 - target
    ),
}


@pytest.mark.parametrize("name", list(BROKEN_LOSSES))
def test_broken_loss_is_caught(monkeypatch, name):
    # With a broken loss the same check used above must fail to show learning.
    monkeypatch.setattr(tb, "bc_loss", BROKEN_LOSSES[name])
    val_mae, refs = fit_and_score(structured_arrays())
    assert val_mae >= 0.5 * refs["mean_action_mae_px"], (val_mae, refs)


# ------------------------------------------------------- references, subsets


def test_reference_maes_known_values():
    arrays = structured_arrays(n_episodes=6, length=5)
    train_ids, val_ids = [0, 1, 2, 3], [4, 5]
    refs = tb.reference_maes(arrays, train_ids, val_ids)
    val = np.isin(arrays.episode_index, val_ids)
    train = np.isin(arrays.episode_index, train_ids)
    a, s = arrays.actions.astype(np.float64), arrays.states.astype(np.float64)
    assert refs["copy_agent_position_mae_px"] == pytest.approx(
        np.abs(a[val] - s[val]).mean()
    )
    assert refs["mean_action_mae_px"] == pytest.approx(
        np.abs(a[val] - a[train].mean(axis=0)).mean()
    )


def test_subsample_episodes_is_a_deterministic_sorted_subset():
    ids = list(range(0, 100, 2))
    a = tb.subsample_episodes(ids, 10, seed=1)
    assert len(a) == 10 and set(a) <= set(ids) and a == sorted(a)
    assert a == tb.subsample_episodes(ids, 10, seed=1)
    assert a != tb.subsample_episodes(ids, 10, seed=2)


def test_subsample_of_everything_returns_everything():
    assert tb.subsample_episodes([3, 1, 2], 3) == [1, 2, 3]


@pytest.mark.parametrize("n", [0, 4, -1])
def test_subsample_rejects_bad_counts(n):
    with pytest.raises(ValueError):
        tb.subsample_episodes([1, 2, 3], n)


# ------------------------------------------------------------- saved outputs

SMALL_HISTORY = {
    "epoch": [1, 2, 3],
    "train_loss": [0.3, 0.1, 0.05],
    "val_loss": [0.3, 0.12, 0.08],
    "val_mae_px": [90.0, 40.0, 30.0],
    "seconds": [1.0, 2.0, 3.0],
}


def test_save_loss_curve_writes_a_png(tmp_path):
    path = tmp_path / "sub" / "curve.png"
    tb.save_loss_curve(
        SMALL_HISTORY,
        path,
        references={"copy_agent_position_mae_px": 14.7, "mean_action_mae_px": 100.0},
    )
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 1000


def test_save_loss_curve_tolerates_an_unknown_reference_name(tmp_path):
    path = tmp_path / "curve.png"
    tb.save_loss_curve(SMALL_HISTORY, path, references={"something_new_px": 20.0})
    assert path.stat().st_size > 1000


def test_reference_lines_are_visually_distinct():
    styles = list(tb.REFERENCE_STYLES.values())
    assert len({(s["color"], s["linestyle"]) for s in styles}) == len(styles)
    assert len({s["label"] for s in styles}) == len(styles)
    assert all(s["color"] != "tab:blue" for s in styles)  # the network line's colour


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = tb.BCPolicyNet().eval()
    path = tmp_path / "ckpt" / "m.pt"
    split = {"train_episode_ids": [0, 1], "val_episode_ids": [2]}
    tb.save_checkpoint(
        path,
        model,
        config={"epochs": 2},
        split=split,
        history={"epoch": [1], "val_mae_px": [1.0]},
        references={"copy_agent_position_mae_px": 1.0},
    )
    loaded, ckpt = tb.load_checkpoint(path)
    image, pos = torch.rand(3, 3, 96, 96), torch.rand(3, 2)
    with torch.no_grad():
        assert torch.equal(model(image, pos), loaded(image, pos))
    assert ckpt["config"] == {"epochs": 2} and ckpt["split"] == split
    assert ckpt["dataset_revision"] == pusht_data.DATASET_REVISION


def test_load_checkpoint_accepts_a_stored_torch_version_object(tmp_path):
    # The first real training run stored torch.__version__ (a TorchVersion, not a plain
    # str) in the checkpoint; torch's safe loader rejected it. Those files must still load.
    torch.manual_seed(0)
    model = tb.BCPolicyNet().eval()
    path = tmp_path / "old_style.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"torch": torch.__version__},
            "split": {},
            "history": {},
            "references": {},
            "dataset_revision": pusht_data.DATASET_REVISION,
        },
        path,
    )
    loaded, ckpt = tb.load_checkpoint(path)
    assert isinstance(loaded, tb.BCPolicyNet)
    assert str(ckpt["config"]["torch"]) == str(torch.__version__)


# ---------------------------------------------------------------- command line


def run_main(tmp_path, name, *extra):
    argv = [
        "--epochs",
        "2",
        "--batch-size",
        "32",
        "--device",
        "cpu",
        "--val-fraction",
        "0.25",
        "--checkpoint",
        str(tmp_path / f"{name}.pt"),
        "--report",
        str(tmp_path / f"{name}.json"),
        "--curve",
        str(tmp_path / f"{name}.png"),
        *extra,
    ]
    return tb.main(argv, arrays_loader=lambda: structured_arrays())


def test_main_runs_end_to_end_on_small_data(tmp_path, capsys):
    assert run_main(tmp_path, "run") == 0
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["config"]["epochs"] == 2 and len(report["history"]["epoch"]) == 2
    assert set(report["split"]["train_episode_ids"]).isdisjoint(
        report["split"]["val_episode_ids"]
    )
    assert report["dataset_revision"] == pusht_data.DATASET_REVISION
    assert (tmp_path / "run.pt").exists() and (tmp_path / "run.png").exists()
    assert "validation MAE" in capsys.readouterr().out


def test_checkpoint_written_by_main_loads_with_plain_weights_only(tmp_path):
    # The check that would have caught the bug: reload what main() actually saved, with
    # torch's strict loader and no special allowances.
    run_main(tmp_path, "run")
    ckpt = torch.load(tmp_path / "run.pt", map_location="cpu", weights_only=True)
    assert type(ckpt["config"]["torch"]) is str
    loaded, _ = tb.load_checkpoint(tmp_path / "run.pt")
    assert isinstance(loaded, tb.BCPolicyNet)


def test_main_train_episodes_keeps_the_validation_set_fixed(tmp_path):
    run_main(tmp_path, "all")
    run_main(tmp_path, "few", "--train-episodes", "6")
    all_report = json.loads((tmp_path / "all.json").read_text())
    few_report = json.loads((tmp_path / "few.json").read_text())
    assert few_report["config"]["train_episodes"] == 6
    assert len(few_report["split"]["train_episode_ids"]) == 6
    assert set(few_report["split"]["train_episode_ids"]) <= set(
        all_report["split"]["train_episode_ids"]
    )
    assert (
        few_report["split"]["val_episode_ids"] == all_report["split"]["val_episode_ids"]
    )


def test_main_rejects_more_training_episodes_than_available(tmp_path):
    with pytest.raises(ValueError):
        run_main(tmp_path, "x", "--train-episodes", "999")


# ------------------------------------------------------ the real demonstrations

needs_data = pytest.mark.skipif(
    not (pusht_data.DEFAULT_ROOT / pusht_data.VIDEO_FILE).exists(),
    reason="dataset not downloaded; run: python pusht_data.py download",
)


@needs_data
def test_a_real_batch_flows_through_the_network_and_references_look_right():
    arrays = pusht_data.load_arrays()
    train_ids, val_ids = pusht_data.split_episode_ids(arrays.n_episodes, 0.2, seed=0)
    batch = next(iter(DataLoader(PushTDataset(arrays, val_ids), batch_size=32)))
    torch.manual_seed(0)
    pred = tb.BCPolicyNet()(batch["image"], batch["agent_pos"])
    assert pred.shape == (32, 2)
    assert torch.isfinite(tb.bc_loss(pred, batch["action"]))
    refs = tb.reference_maes(arrays, train_ids, val_ids)
    assert 10.0 < refs["copy_agent_position_mae_px"] < 20.0
    assert refs["mean_action_mae_px"] > 2 * refs["copy_agent_position_mae_px"]
