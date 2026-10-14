import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import pusht_data
import train_bc as tb
import train_chunk as tc

ORIGINAL_LOSS = tc.chunk_loss


def make_arrays(lengths, states=None, actions=None, seed=0):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=np.int64)
    n = int(lengths.sum())
    if states is None:
        states = rng.uniform(0, 512, size=(n, 2))
    states = np.asarray(states, dtype=np.float32)
    if actions is None:
        actions = 0.5 * states + 128.0
    return pusht_data.PushTArrays(
        images=np.zeros((n, 96, 96, 3), dtype=np.uint8),
        states=states,
        actions=np.asarray(actions, dtype=np.float32),
        episode_index=np.repeat(np.arange(len(lengths)), lengths).astype(np.int64),
        frame_index=np.concatenate([np.arange(c) for c in lengths]).astype(np.int64),
        reward=np.zeros(n, dtype=np.float32),
        done=np.zeros(n, dtype=bool),
        success=np.zeros(n, dtype=bool),
        episode_lengths=lengths,
    )


def ramp_arrays():
    """Two episodes of 5 frames. The action at step t of episode e is (100*e + 10*t, 7)."""
    actions = [[100.0 * e + 10.0 * t, 7.0] for e in range(2) for t in range(5)]
    return make_arrays([5, 5], actions=actions)


def constant_action_arrays(n_episodes=2, length=6):
    """Every action is (256, 128), i.e. (0.5, 0.25) once scaled."""
    n = n_episodes * length
    return make_arrays([length] * n_episodes, actions=np.tile([256.0, 128.0], (n, 1)))


def drifting_arrays(n_episodes=32, length=16, seed=0, shuffle_labels=False):
    """The agent drifts 2 px right per step, and the action is 0.5 * position + 128, so the
    action k steps ahead is the current one plus k pixels in x. Images are blank, so the agent
    position is the only useful input and every chunk position is learnable from it."""
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, 400, size=(n_episodes, 2))
    drift = np.arange(length)[None, :, None] * np.array([2.0, 0.0])
    states = (starts[:, None, :] + drift).reshape(-1, 2).astype(np.float32)
    actions = (0.5 * states + 128.0).astype(np.float32)
    if shuffle_labels:
        actions = actions[rng.permutation(len(actions))]
    return make_arrays([length] * n_episodes, states=states, actions=actions)


class StepStub(nn.Module):
    """Predicts 0.5 for the first chunk position and 0.9 for all the others."""

    def __init__(self, chunk_len):
        super().__init__()
        self.chunk_len = chunk_len

    def forward(self, image, agent_pos):
        out = torch.full((len(agent_pos), self.chunk_len, 2), 0.9)
        out[:, 0] = 0.5
        return out


# --------------------------------------------------------------------- network


def test_net_output_shape_is_a_chunk_of_positions():
    out = tc.ChunkPolicyNet(16)(torch.rand(4, 3, 96, 96), torch.rand(4, 2))
    assert out.shape == (4, 16, 2) and torch.isfinite(out).all()


def test_parameter_counts_differ_from_the_baseline_only_in_the_last_layer():
    baseline = sum(p.numel() for p in tb.BCPolicyNet().parameters())
    assert baseline == 718_178
    assert sum(p.numel() for p in tc.ChunkPolicyNet(1).parameters()) == baseline
    assert (
        sum(p.numel() for p in tc.ChunkPolicyNet(16).parameters())
        == baseline + 514 * 15
    )


def test_the_trunk_has_the_same_layers_as_the_baseline():
    ours = [p.shape for p in tc.make_trunk().parameters()]
    theirs = [p.shape for p in tb.BCPolicyNet().conv.parameters()]
    assert ours == theirs and len(ours) == 8


def test_net_is_deterministic_in_eval_mode():
    torch.manual_seed(0)
    net = tc.ChunkPolicyNet(4).eval()
    image, pos = torch.rand(3, 3, 96, 96), torch.rand(3, 2)
    with torch.no_grad():
        assert torch.equal(net(image, pos), net(image, pos))


def test_net_rejects_a_chunk_length_below_one():
    with pytest.raises(ValueError):
        tc.ChunkPolicyNet(0)


# ------------------------------------------------------------------------ data


def test_a_chunk_from_the_start_of_an_episode_is_full():
    ds = tc.ChunkDataset(ramp_arrays(), chunk_len=4)
    item = ds[5]  # episode 1, frame 0
    assert item["episode_index"] == 1 and item["frame_index"] == 0
    assert (item["actions"] * 512).round().tolist() == [
        [100, 7],
        [110, 7],
        [120, 7],
        [130, 7],
    ]
    assert item["mask"].tolist() == [1, 1, 1, 1]


def test_a_chunk_near_the_end_is_padded_with_the_last_action_and_masked():
    item = tc.ChunkDataset(ramp_arrays(), chunk_len=4)[3]  # episode 0, frame 3 of 5
    assert (item["actions"] * 512).round().tolist() == [
        [30, 7],
        [40, 7],
        [40, 7],
        [40, 7],
    ]
    assert item["mask"].tolist() == [1, 1, 0, 0]


def test_the_last_frame_has_one_valid_position_and_never_reads_the_next_episode():
    item = tc.ChunkDataset(ramp_arrays(), chunk_len=4)[4]  # last frame of episode 0
    assert (item["actions"] * 512).round().tolist() == [[40, 7]] * 4
    assert item["mask"].tolist() == [1, 0, 0, 0]
    last = tc.ChunkDataset(ramp_arrays(), chunk_len=4)[9]  # last frame of episode 1
    assert (last["actions"] * 512).round().tolist() == [[140, 7]] * 4


def test_item_shapes_and_ranges():
    item = tc.ChunkDataset(structured := make_arrays([6, 6]), chunk_len=5)[0]
    assert item["image"].shape == (3, 96, 96) and item["image"].dtype == torch.float32
    assert item["agent_pos"].shape == (2,)
    assert item["actions"].shape == (5, 2) and item["mask"].shape == (5,)
    assert 0.0 <= float(item["actions"].min()) and float(item["actions"].max()) <= 1.0
    assert len(structured.states) == 12


def test_chunk_length_one_matches_the_single_action_dataset():
    arrays = make_arrays([5, 5])
    chunked = tc.ChunkDataset(arrays, chunk_len=1)
    single = pusht_data.PushTDataset(arrays)
    assert len(chunked) == len(single)
    for i in (0, 4, 7):
        a, b = chunked[i], single[i]
        assert torch.equal(a["actions"][0], b["action"]) and a["mask"].tolist() == [1.0]
        assert torch.equal(a["image"], b["image"]) and torch.equal(
            a["agent_pos"], b["agent_pos"]
        )


def test_subsets_contain_only_the_requested_episodes():
    ds = tc.ChunkDataset(ramp_arrays(), episode_ids=[1], chunk_len=4)
    assert len(ds) == 5 and {ds[i]["episode_index"] for i in range(5)} == {1}


@pytest.mark.parametrize("ids", [[], [2], [-1]])
def test_unknown_episode_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        tc.ChunkDataset(ramp_arrays(), episode_ids=ids, chunk_len=4)


def test_dataset_rejects_a_chunk_length_below_one():
    with pytest.raises(ValueError):
        tc.ChunkDataset(ramp_arrays(), chunk_len=0)


# ------------------------------------------------------------------------ loss


def test_chunk_loss_known_values():
    target = torch.ones(1, 4, 2)
    assert float(tc.chunk_loss(target.clone(), target, torch.ones(1, 4))) == 0.0
    assert float(
        tc.chunk_loss(torch.zeros(1, 4, 2), target, torch.ones(1, 4))
    ) == pytest.approx(1.0)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert float(tc.chunk_loss(torch.zeros(1, 4, 2), target, mask)) == pytest.approx(
        1.0
    )


def test_padded_positions_do_not_affect_the_loss():
    target = torch.rand(3, 4, 2)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32)
    pred = torch.rand(3, 4, 2)
    other = pred.clone()
    other[mask == 0] += 5.0
    assert float(tc.chunk_loss(pred, target, mask)) == pytest.approx(
        float(tc.chunk_loss(other, target, mask))
    )
    assert float(tc.chunk_loss(pred, target, torch.ones(3, 4))) != pytest.approx(
        float(tc.chunk_loss(other, target, torch.ones(3, 4)))
    )


def test_chunk_loss_weights_every_valid_position_equally():
    target = torch.zeros(2, 4, 2)
    pred = torch.zeros(2, 4, 2)
    pred[1, 0] = 1.0  # only the single valid position of the second sample is wrong
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32)
    assert float(tc.chunk_loss(pred, target, mask)) == pytest.approx(2.0 / 10.0)


# ------------------------------------------------------------------ evaluation


def test_evaluate_loader_known_values():
    arrays = constant_action_arrays(n_episodes=2, length=6)
    loader = DataLoader(tc.ChunkDataset(arrays, chunk_len=4), batch_size=5)
    result = tc.evaluate_loader(StepStub(4), loader, torch.device("cpu"))
    # 12 frames; valid positions per episode 4 + 4 + 4 + 3 + 2 + 1 = 18, so 36 in all
    assert result["loss"] == pytest.approx(14.73 / 72, rel=1e-4)
    assert result["first_mae_px"] == pytest.approx(64.0, rel=1e-4)
    assert result["chunk_mae_px"] == pytest.approx(28.2 / 72 * 512, rel=1e-4)


def test_evaluate_loader_rejects_an_empty_loader():
    with pytest.raises(ValueError):
        tc.evaluate_loader(
            StepStub(4), DataLoader([], batch_size=2), torch.device("cpu")
        )


# -------------------------------------------------------------------- training


def split_datasets(arrays, chunk_len=4):
    train_ids, val_ids = pusht_data.split_episode_ids(arrays.n_episodes, 0.25, seed=0)
    return (
        tc.ChunkDataset(arrays, train_ids, chunk_len),
        tc.ChunkDataset(arrays, val_ids, chunk_len),
        train_ids,
        val_ids,
    )


def fit_and_score(arrays, epochs=40, seed=0, chunk_len=4):
    train_ds, val_ds, train_ids, val_ids = split_datasets(arrays, chunk_len)
    torch.manual_seed(seed)
    history = tc.train(
        tc.ChunkPolicyNet(chunk_len),
        train_ds,
        val_ds,
        epochs=epochs,
        batch_size=32,
        lr=2e-3,
        seed=seed,
        log=lambda *_: None,
    )
    return history, tb.reference_maes(arrays, train_ids, val_ids)


def test_train_records_one_history_entry_per_epoch():
    arrays = drifting_arrays(n_episodes=8, length=8)
    train_ds, val_ds, _, _ = split_datasets(arrays)
    lines = []
    history = tc.train(
        tc.ChunkPolicyNet(4),
        train_ds,
        val_ds,
        epochs=2,
        batch_size=16,
        log=lines.append,
    )
    assert history["epoch"] == [1, 2] and len(lines) == 2
    for key in ("train_loss", "val_loss", "val_mae_px", "val_chunk_mae_px", "seconds"):
        assert len(history[key]) == 2 and np.isfinite(history[key]).all()


@pytest.mark.parametrize(
    "kwargs", [dict(epochs=0), dict(epochs=1, batch_size=0), dict(epochs=1, lr=0.0)]
)
def test_train_rejects_bad_arguments(kwargs):
    ds = tc.ChunkDataset(drifting_arrays(n_episodes=4, length=4), chunk_len=2)
    with pytest.raises(ValueError):
        tc.train(tc.ChunkPolicyNet(2), ds, ds, log=lambda *_: None, **kwargs)


def test_training_reduces_the_loss():
    history, _ = fit_and_score(drifting_arrays(), epochs=6)
    assert history["train_loss"][-1] < history["train_loss"][0]


# -------------------- the checks that fail if the loss or the data code is broken


def test_learns_every_position_of_a_predictable_chunk():
    history, refs = fit_and_score(drifting_arrays())
    assert history["val_mae_px"][-1] < 0.5 * refs["mean_action_mae_px"], (
        history["val_mae_px"][-1],
        refs,
    )
    assert history["val_chunk_mae_px"][-1] < 0.5 * refs["mean_action_mae_px"], (
        history["val_chunk_mae_px"][-1],
        refs,
    )


def test_shuffled_labels_are_caught():
    history, refs = fit_and_score(drifting_arrays(shuffle_labels=True))
    assert history["val_mae_px"][-1] >= 0.8 * refs["mean_action_mae_px"], (
        history["val_mae_px"][-1],
        refs,
    )


BROKEN_LOSSES = {
    "zero gradient": lambda pred, target, mask: (pred * 0).sum() + 1.0,
    "inverted targets": lambda pred, target, mask: ORIGINAL_LOSS(
        pred, 1.0 - target, mask
    ),
}


@pytest.mark.parametrize("name", list(BROKEN_LOSSES))
def test_broken_loss_is_caught(monkeypatch, name):
    monkeypatch.setattr(tc, "chunk_loss", BROKEN_LOSSES[name])
    history, refs = fit_and_score(drifting_arrays())
    assert history["val_mae_px"][-1] >= 0.5 * refs["mean_action_mae_px"], (
        history["val_mae_px"][-1],
        refs,
    )


# ------------------------------------------------------------------ checkpoints


def test_checkpoint_round_trip_keeps_the_chunk_length_and_weights(tmp_path):
    torch.manual_seed(0)
    model = tc.ChunkPolicyNet(4).eval()
    path = tmp_path / "ckpt" / "m.pt"
    tb.save_checkpoint(
        path,
        model,
        config={"chunk_len": 4, "n_exec": 2},
        split={"train_episode_ids": [0], "val_episode_ids": [1]},
        history={"epoch": [1], "val_mae_px": [1.0]},
        references={},
    )
    loaded, ckpt = tc.load_chunk_checkpoint(path)
    image, pos = torch.rand(3, 3, 96, 96), torch.rand(3, 2)
    with torch.no_grad():
        assert torch.equal(model(image, pos), loaded(image, pos))
    assert loaded.chunk_len == 4 and ckpt["config"]["n_exec"] == 2


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
        "--chunk-len",
        "4",
        "--n-exec",
        "2",
        "--checkpoint",
        str(tmp_path / f"{name}.pt"),
        "--report",
        str(tmp_path / f"{name}.json"),
        "--curve",
        str(tmp_path / f"{name}.png"),
        *extra,
    ]
    return tc.main(argv, arrays_loader=drifting_arrays)


def test_main_runs_end_to_end_and_the_checkpoint_loads_in_safe_mode(tmp_path, capsys):
    assert run_main(tmp_path, "run") == 0
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["config"]["chunk_len"] == 4 and report["config"]["n_exec"] == 2
    assert report["config"]["architecture"] == "ChunkPolicyNet"
    assert set(report["split"]["train_episode_ids"]).isdisjoint(
        report["split"]["val_episode_ids"]
    )
    assert len(report["history"]["val_chunk_mae_px"]) == 2
    assert report["dataset_revision"] == pusht_data.DATASET_REVISION
    assert (tmp_path / "run.png").exists()
    ckpt = torch.load(tmp_path / "run.pt", map_location="cpu", weights_only=True)
    assert type(ckpt["config"]["torch"]) is str
    model, _ = tc.load_chunk_checkpoint(tmp_path / "run.pt")
    assert model.chunk_len == 4
    assert "validation MAE of the first action" in capsys.readouterr().out


def test_main_train_episodes_keeps_the_validation_set_fixed(tmp_path):
    run_main(tmp_path, "all")
    run_main(tmp_path, "few", "--train-episodes", "6")
    all_report = json.loads((tmp_path / "all.json").read_text())
    few_report = json.loads((tmp_path / "few.json").read_text())
    assert len(few_report["split"]["train_episode_ids"]) == 6
    assert set(few_report["split"]["train_episode_ids"]) <= set(
        all_report["split"]["train_episode_ids"]
    )
    assert (
        few_report["split"]["val_episode_ids"] == all_report["split"]["val_episode_ids"]
    )


@pytest.mark.parametrize("chunk_len, n_exec", [(4, 5), (4, 0), (0, 1)])
def test_main_rejects_inconsistent_chunk_settings(tmp_path, chunk_len, n_exec):
    argv = [
        "--chunk-len",
        str(chunk_len),
        "--n-exec",
        str(n_exec),
        "--device",
        "cpu",
        "--checkpoint",
        str(tmp_path / "x.pt"),
        "--report",
        str(tmp_path / "x.json"),
        "--curve",
        str(tmp_path / "x.png"),
    ]
    with pytest.raises(ValueError):
        tc.main(argv, arrays_loader=drifting_arrays)


# ------------------------------------------------------ the real demonstrations

needs_data = pytest.mark.skipif(
    not (pusht_data.DEFAULT_ROOT / pusht_data.VIDEO_FILE).exists(),
    reason="dataset not downloaded; run: python pusht_data.py download",
)


@pytest.fixture(scope="module")
def real_arrays():
    return pusht_data.load_arrays()


@needs_data
def test_real_chunks_start_at_the_known_first_action_and_stay_inside_the_episode(
    real_arrays,
):
    ds = tc.ChunkDataset(real_arrays, chunk_len=16)
    assert len(ds) == 25650
    first = ds[0]
    assert np.allclose(first["actions"][0].numpy() * 512, [233, 71], atol=1e-2)
    assert first["mask"].tolist() == [1.0] * 16
    last = ds[160]  # the last frame of episode 0, which has 161 frames
    assert last["episode_index"] == 0 and last["frame_index"] == 160
    assert last["mask"].sum() == 1
    assert torch.equal(last["actions"], last["actions"][0].expand(16, 2))


@needs_data
def test_real_batch_flows_through_the_chunk_network(real_arrays):
    val_ids = pusht_data.split_episode_ids(real_arrays.n_episodes, 0.2, seed=0)[1]
    batch = next(
        iter(DataLoader(tc.ChunkDataset(real_arrays, val_ids, 16), batch_size=32))
    )
    torch.manual_seed(0)
    pred = tc.ChunkPolicyNet(16)(batch["image"], batch["agent_pos"])
    assert pred.shape == (32, 16, 2)
    assert torch.isfinite(tc.chunk_loss(pred, batch["actions"], batch["mask"]))
