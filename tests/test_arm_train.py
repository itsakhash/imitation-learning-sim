"""Tests for arm_train.py.

The naming and seed helpers and the settings run anywhere. The rest need torch (skipped without
it) and run on the CPU with tiny networks on a fake dataset, except the command-line tests,
which train the real networks for two epochs on a fake 96x96 dataset of 57 frames.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

import arm_train as at


# ---------------------------------------------------------------- pure


def test_default_names_follow_the_project_convention():
    assert at.default_name("D", 0) == "diff_cnn"
    assert at.default_name("D", 7) == "diff_cnn_s7"
    assert at.default_name("C", 3) == "chunk_long_s3"
    assert at.default_name("B", 0) == "bc_long"
    assert at.default_name("P", 12) == "chunk_res30_s12"
    with pytest.raises(ValueError):
        at.default_name("X", 0)


def test_parse_seeds():
    assert at.parse_seeds("3") == [3]
    assert at.parse_seeds("0-3") == [0, 1, 2, 3]
    assert at.parse_seeds("0, 2,5-6") == [0, 2, 5, 6]


@pytest.mark.parametrize("text", ["", "3-1", "1,1", "0-2,2", "a", "-1", "1,,2"])
def test_parse_seeds_rejects_bad_input(text):
    with pytest.raises(ValueError):
        at.parse_seeds(text)


def test_settings_match_the_preregistration():
    s = at.ARM_SETTINGS
    for code in "DCB":
        assert s[code]["epochs"] == 370 and s[code]["ema"] == 0.999
    assert s["P"]["epochs"] == 30 and s["P"]["ema"] is None and s["P"]["val_every"] == 1
    assert s["D"]["arm"].chunk_len == 16 and s["C"]["arm"].chunk_len == 16
    assert s["B"]["arm"].chunk_len == 1
    assert at.EMA_DECAY == 0.999 and at.EPOCHS == 370  # 370 epochs x 81 steps = 29,970 steps


# ---------------------------------------------------------------- helpers with torch


def _torch():
    return pytest.importorskip("torch")


def _tiny_diff_net_class():
    torch = _torch()

    class TinyDiffNet(torch.nn.Module):
        """Has the encode / denoise interface the diffusion code needs, on 8x8 images."""

        def __init__(self, chunk_len=4):
            super().__init__()
            self.chunk_len = chunk_len
            self.enc = torch.nn.Linear(3 * 8 * 8 + 2, 8)
            self.den = torch.nn.Linear(chunk_len * 2 + 8 + 1, chunk_len * 2)

        def encode(self, image, agent_pos):
            return torch.tanh(self.enc(torch.cat([image.flatten(1), agent_pos], dim=1)))

        def denoise(self, x_t, t, cond):
            b = x_t.shape[0]
            inp = torch.cat([x_t.reshape(b, -1), cond, (t.float() / 100.0)[:, None]], dim=1)
            return self.den(inp).view(b, self.chunk_len, 2)

    return TinyDiffNet


def _setup():
    torch = _torch()
    from test_resident_train import _setup as setup

    return setup()  # torch, a fake ChunkDataset (chunk length 4), its resident copy (30 frames)


def _quiet(*_):
    return None


def _weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------- EMA


def test_ema_update_arithmetic():
    torch = _torch()
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.bias.copy_(torch.tensor([3.0]))
    ema = at.EMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[11.0, 12.0]]))
        model.bias.copy_(torch.tensor([13.0]))
    ema.update(model)
    assert torch.allclose(ema.shadow["weight"], torch.tensor([[2.0, 3.0]]))  # 0.9*old + 0.1*new
    assert torch.allclose(ema.shadow["bias"], torch.tensor([4.0]))
    ema.update(model)  # the model is unchanged: the shadow moves 10% of the way again
    assert torch.allclose(ema.shadow["weight"], torch.tensor([[0.9 * 2 + 1.1, 0.9 * 3 + 1.2]]))
    assert ema.updates == 2


def test_ema_starts_at_the_initial_weights_and_is_independent_of_the_model():
    torch = _torch()
    model = torch.nn.Linear(2, 1)
    initial = _weights(model)
    ema = at.EMA(model, 0.999)
    with torch.no_grad():
        model.weight.add_(5.0)
    for k, v in initial.items():
        assert torch.equal(ema.shadow[k], v)  # changing the model must not move the shadow


def test_ema_decay_zero_follows_the_model_and_decay_validation():
    torch = _torch()
    model = torch.nn.Linear(2, 1)
    ema = at.EMA(model, 0.0)
    with torch.no_grad():
        model.weight.add_(5.0)
    ema.update(model)
    assert torch.equal(ema.shadow["weight"], model.weight)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            at.EMA(model, bad)


def test_ema_copies_non_float_state_and_copy_to_loads_the_shadow():
    torch = _torch()

    class WithCounter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 1)
            self.register_buffer("count", torch.tensor(0))

    model = WithCounter()
    ema = at.EMA(model, 0.9)
    model.count += 7
    with torch.no_grad():
        model.fc.weight.add_(10.0)
    ema.update(model)
    assert int(ema.shadow["count"]) == 7  # copied, not averaged
    other = WithCounter()
    ema.copy_to(other)
    assert torch.equal(other.fc.weight, ema.shadow["fc.weight"]) and int(other.count) == 7


# ---------------------------------------------------------------- the loop


def test_without_ema_and_with_validation_every_epoch_the_loop_is_train_resident():
    torch, ds, res = _setup()
    import resident_train as rt
    from test_resident_train import _tiny_net_class, _seed

    TinyNet = _tiny_net_class()
    _seed(2)
    model_a = TinyNet()
    _seed(2)
    model_b = TinyNet()
    kwargs = dict(epochs=4, batch_size=8, lr=1e-2, seed=2)
    old = rt.train_resident(model_a, res, res, log=_quiet, coord_scale=512.0, **kwargs)
    new, averaged = at.train_arm(
        at.ChunkArm(), model_b, res, res, ema_decay=None, val_every=1, log=_quiet, coord_scale=512.0, **kwargs
    )
    for key in ("train_loss", "val_loss", "val_mae_px", "val_chunk_mae_px"):
        assert new[key] == pytest.approx(old[key], rel=1e-6), key
    assert new["val_epoch"] == [1, 2, 3, 4]
    for k, v in model_a.state_dict().items():
        assert torch.allclose(averaged.state_dict()[k], v, atol=1e-6)


def test_the_averaged_model_is_the_manual_ema_of_the_weights_after_each_step():
    """With the whole data set as one batch, an epoch is one optimiser step, so the weights after
    1, 2 and 3 epochs are those of runs of 1, 2 and 3 epochs without EMA (same seed, same shuffles)."""
    torch, ds, res = _setup()
    from test_resident_train import _tiny_net_class, _seed

    TinyNet = _tiny_net_class()

    def run(epochs, decay):
        _seed(5)
        model = TinyNet()
        initial = _weights(model)
        _, averaged = at.train_arm(
            at.ChunkArm(), model, res, res, epochs=epochs, batch_size=30, lr=1e-2, seed=5,
            ema_decay=decay, val_every=1, log=_quiet, coord_scale=512.0,
        )
        return initial, _weights(model), _weights(averaged)

    initial, w1, _ = run(1, None)
    _, w2, _ = run(2, None)
    _, w3, avg3 = run(3, 0.9)
    expected = {}
    for k in initial:
        s = initial[k]
        for w in (w1[k], w2[k], w3[k]):
            s = 0.9 * s + 0.1 * w
        expected[k] = s
    for k in expected:
        assert torch.allclose(avg3[k], expected[k], atol=1e-6), k
        assert not torch.allclose(avg3[k], w3[k]), k  # and it is not simply the last weights


def test_ema_stays_closer_to_the_start_than_the_trained_weights_do():
    torch, ds, res = _setup()
    from test_resident_train import _tiny_net_class, _seed

    TinyNet = _tiny_net_class()
    _seed(1)
    model = TinyNet()
    initial = _weights(model)
    _, averaged = at.train_arm(
        at.ChunkArm(), model, res, res, epochs=6, batch_size=8, lr=1e-2, seed=1,
        ema_decay=0.999, val_every=2, log=_quiet, coord_scale=512.0,
    )
    far_trained = sum(float((model.state_dict()[k] - initial[k]).norm()) for k in initial)
    far_average = sum(float((averaged.state_dict()[k] - initial[k]).norm()) for k in initial)
    assert 0.0 < far_average < 0.2 * far_trained  # decay 0.999 over 24 steps barely moves it


def test_validation_schedule_and_history_lengths():
    torch, ds, res = _setup()
    from test_resident_train import _tiny_net_class, _seed

    _seed(0)
    model = _tiny_net_class()()
    history, _ = at.train_arm(
        at.ChunkArm(), model, res, res, epochs=7, batch_size=8, lr=1e-2, seed=0,
        val_every=3, log=_quiet, coord_scale=512.0,
    )
    assert history["epoch"] == list(range(1, 8)) and len(history["train_loss"]) == 7
    assert history["val_epoch"] == [3, 6, 7]  # every 3rd epoch and the last
    for key in ("val_loss", "val_mae_px", "val_chunk_mae_px"):
        assert len(history[key]) == 3


def test_training_is_repeatable_for_a_seed_and_depends_on_it():
    torch, ds, res = _setup()
    from test_resident_train import _tiny_net_class, _seed

    TinyDiffNet = _tiny_diff_net_class()

    def run(seed):
        _seed(0)
        h, a = at.train_arm(
            at.DiffusionArm(), TinyDiffNet(), res, res, epochs=3, batch_size=8, lr=1e-2, seed=seed,
            log=_quiet, coord_scale=512.0, val_every=1,
        )
        return h, a

    h1, a1 = run(4)
    h2, a2 = run(4)
    h3, _ = run(5)
    assert h1["train_loss"] == h2["train_loss"] and h1["val_loss"] == h2["val_loss"]
    assert torch.equal(a1.den.weight, a2.den.weight)
    assert h1["train_loss"] != h3["train_loss"]


def test_the_diffusion_loss_falls_when_the_loop_trains():
    torch, ds, res = _setup()
    from test_resident_train import _seed

    _seed(0)
    model = _tiny_diff_net_class()()
    history, _ = at.train_arm(
        at.DiffusionArm(), model, res, res, epochs=80, batch_size=8, lr=1e-2, seed=0,
        log=_quiet, coord_scale=512.0, val_every=40,
    )
    first = sum(history["train_loss"][:5]) / 5
    last = sum(history["train_loss"][-10:]) / 10
    assert last < 0.9 * first


def test_non_finite_loss_stops_training():
    torch, ds, res = _setup()
    from test_resident_train import _tiny_net_class, _seed

    class NanArm(at.ChunkArm):
        def loss(self, model, batch, rng):
            return model(batch["image"], batch["agent_pos"]).sum() * float("nan")

    _seed(0)
    with pytest.raises(FloatingPointError):
        at.train_arm(NanArm(), _tiny_net_class()(), res, res, epochs=2, batch_size=8, lr=1e-2, log=_quiet,
                     coord_scale=512.0)


def test_train_arm_rejects_bad_arguments():
    torch, ds, res = _setup()
    from test_resident_train import _tiny_net_class

    kwargs = dict(log=_quiet, coord_scale=512.0)
    arm = at.ChunkArm()
    with pytest.raises(ValueError):
        at.train_arm(arm, _tiny_net_class()(), res, res, epochs=0, lr=1e-2, **kwargs)
    with pytest.raises(ValueError):
        at.train_arm(arm, _tiny_net_class()(), res, res, epochs=1, lr=0.0, **kwargs)
    with pytest.raises(ValueError):
        at.train_arm(arm, _tiny_net_class()(), res, res, epochs=1, lr=1e-2, batch_size=0, **kwargs)
    with pytest.raises(ValueError):
        at.train_arm(arm, _tiny_net_class()(), res, res, epochs=1, lr=1e-2, val_every=0, **kwargs)


# ---------------------------------------------------------------- the arms' losses and validation


def test_diffusion_loss_uses_the_noise_generator_and_backpropagates():
    torch, ds, res = _setup()
    from test_resident_train import _seed

    arm = at.DiffusionArm()
    _seed(0)
    model = _tiny_diff_net_class()()
    batch = res.batch(torch.arange(8))
    a = arm.loss(model, batch, torch.Generator().manual_seed(1))
    b = arm.loss(model, batch, torch.Generator().manual_seed(1))
    c = arm.loss(model, batch, torch.Generator().manual_seed(2))
    assert float(a) == float(b) and float(a) != float(c)
    a.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_diffusion_loss_matches_the_core_function_on_the_diffusion_scale():
    torch, ds, res = _setup()
    import diffusion_core as dc
    from test_resident_train import _seed

    arm = at.DiffusionArm()
    _seed(0)
    model = _tiny_diff_net_class()()
    batch = res.batch(torch.arange(8))
    got = arm.loss(model, batch, torch.Generator().manual_seed(3))
    cond = model.encode(batch["image"], batch["agent_pos"])
    ab = torch.as_tensor(dc.cosine_alpha_bars(), dtype=torch.float32)
    expected = dc.diffusion_loss(
        lambda x, t: model.denoise(x, t, cond),
        batch["actions"] * 2.0 - 1.0,  # written out, not via the helper under test
        batch["mask"],
        ab,
        generator=torch.Generator().manual_seed(3),
    )
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_diffusion_validation_is_repeatable_and_sensitive_to_the_weights():
    torch, ds, res = _setup()
    from test_resident_train import _seed

    arm = at.DiffusionArm()
    _seed(0)
    model = _tiny_diff_net_class()()
    a = arm.validate(model, res, 8, 512.0)
    b = arm.validate(model, res, 8, 512.0)
    assert a == b  # the same seeded noise every time
    assert all(np.isfinite(v) for v in a.values()) and a["loss"] > 0
    assert 0.0 < a["val_mae_px"] < 512.0
    with torch.no_grad():
        model.den.weight.add_(0.5)
    assert arm.validate(model, res, 8, 512.0)["val_mae_px"] != a["val_mae_px"]


def test_validation_of_chunk_and_baseline_arms_does_not_depend_on_the_batch_size():
    torch, ds, res = _setup()
    from test_resident_train import _tiny_net_class, _seed

    _seed(0)
    chunk = _tiny_net_class()()
    a, b = at.ChunkArm().validate(chunk, res, 8, 512.0), at.ChunkArm().validate(chunk, res, 30, 512.0)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-5)

    class TinyBaseline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 8 * 8 + 2, 2)

        def forward(self, image, agent_pos):
            return self.fc(torch.cat([image.flatten(1), agent_pos], dim=1))

    # the baseline arm needs chunk length 1 data: use the first action only, as an oracle check
    class OneStep:
        def __init__(self, res):
            self.res, self.device = res, res.device
            self.tensors = {k: v for k, v in res.tensors.items()}
            self.tensors["actions"] = res.tensors["actions"][:, :1]
            self.tensors["mask"] = torch.ones(len(res), 1)

        def __len__(self):
            return len(self.res)

        def batch(self, ids):
            return {k: v[ids] for k, v in self.tensors.items()}

    one = OneStep(res)
    base = TinyBaseline()
    c, d = at.BaselineArm().validate(base, one, 8, 512.0), at.BaselineArm().validate(base, one, 30, 512.0)
    for k in c:
        assert c[k] == pytest.approx(d[k], rel=1e-5)
    # by hand, one frame at a time
    base.eval()
    sq = ab = 0.0
    with torch.no_grad():
        for i in range(len(res)):
            item = res.batch(torch.tensor([i]))
            err = base(item["image"], item["agent_pos"]) - item["actions"][:, 0]
            sq += float((err**2).sum())
            ab += float(err.abs().sum())
    assert c["loss"] == pytest.approx(sq / (len(res) * 2), rel=1e-5)
    assert c["val_mae_px"] == pytest.approx(ab / (len(res) * 2) * 512.0, rel=1e-5)


# ---------------------------------------------------------------- the baseline's data and targets


def make_pusht_arrays():
    """A fake PushTArrays with real 96x96 images: six episodes, 57 frames."""
    from pusht_data import PushTArrays

    rng = np.random.default_rng(0)
    lengths = np.array([10, 9, 8, 12, 7, 11], dtype=np.int64)
    n = int(lengths.sum())
    return PushTArrays(
        images=rng.integers(0, 256, size=(n, 96, 96, 3), dtype=np.uint8),
        states=rng.uniform(0, 512, size=(n, 2)).astype(np.float32),
        actions=rng.uniform(0, 512, size=(n, 2)).astype(np.float32),
        episode_index=np.concatenate([np.full(l, e) for e, l in enumerate(lengths)]).astype(np.int64),
        frame_index=np.concatenate([np.arange(l) for l in lengths]).astype(np.int64),
        reward=np.zeros(n, dtype=np.float32),
        done=np.zeros(n, dtype=bool),
        success=np.zeros(n, dtype=bool),
        episode_lengths=lengths,
    )


def test_baseline_targets_equal_the_baseline_datasets():
    """The pre-registered check for arm B: chunk length 1 resident data gives the same images,
    positions and single action as the dataset train_bc.py trains on, frame for frame."""
    torch = _torch()
    from torch.utils.data import default_collate

    import resident_data as rd
    from pusht_data import PushTDataset
    from train_chunk import ChunkDataset

    arrays = make_pusht_arrays()
    ids = [0, 2, 3, 5]
    bc = PushTDataset(arrays, ids)
    chunk1 = ChunkDataset(arrays, ids, chunk_len=1)
    res = rd.build_resident(arrays, chunk1, torch.device("cpu"))
    assert len(bc) == len(chunk1) == len(res)
    reference = default_collate([bc[i] for i in range(len(bc))])
    got = res.batch(torch.arange(len(res)))
    assert torch.equal(got["image"], reference["image"])
    assert torch.equal(got["agent_pos"], reference["agent_pos"])
    assert torch.equal(got["actions"][:, 0], reference["action"])
    assert torch.equal(got["mask"], torch.ones(len(res), 1))  # the single position is always valid


def test_the_baseline_target_check_catches_a_shifted_target():
    """Deliberate break: pair each frame with the NEXT frame's action. The check above must fail."""
    torch = _torch()
    from dataclasses import replace

    import resident_data as rd
    from pusht_data import PushTDataset
    from train_chunk import ChunkDataset

    arrays = make_pusht_arrays()
    shifted = replace(arrays, actions=np.roll(arrays.actions, -1, axis=0))
    ids = [0, 2, 3, 5]
    res = rd.build_resident(shifted, ChunkDataset(shifted, ids, chunk_len=1), torch.device("cpu"))
    bc = PushTDataset(arrays, ids)
    reference = torch.stack([bc[i]["action"] for i in range(len(bc))])
    assert not torch.equal(res.batch(torch.arange(len(res)))["actions"][:, 0], reference)


def test_baseline_arm_loss_equals_bc_loss_on_the_baseline_datasets_batch():
    torch = _torch()
    from torch.utils.data import default_collate

    import resident_data as rd
    import train_bc as tb
    from pusht_data import PushTDataset
    from train_chunk import ChunkDataset

    arrays = make_pusht_arrays()
    ids = [1, 4]
    bc = PushTDataset(arrays, ids)
    res = rd.build_resident(arrays, ChunkDataset(arrays, ids, chunk_len=1), torch.device("cpu"))
    tb.seed_everything(0)
    model = tb.BCPolicyNet()
    reference = default_collate([bc[i] for i in range(len(bc))])
    expected = tb.bc_loss(model(reference["image"], reference["agent_pos"]), reference["action"])
    got = at.BaselineArm().loss(model, res.batch(torch.arange(len(res))), None)
    assert float(got) == float(expected)


# ---------------------------------------------------------------- the command line (real networks)


def _run_main(tmp_path, *extra, arm="C", seeds="0"):
    argv = [
        "--arm", arm, "--seeds", seeds, "--epochs", "2", "--batch-size", "16", "--device", "cpu",
        "--checkpoint-dir", str(tmp_path / "ckpt"), "--report-dir", str(tmp_path / "rep"), *extra,
    ]
    return at.main(argv, arrays_loader=make_pusht_arrays)


def _load(path):
    torch = _torch()
    import train_bc as tb

    with torch.serialization.safe_globals(tb._version_type_allowlist()):
        return torch.load(path, map_location="cpu", weights_only=True)


def test_cli_chunk_arm_writes_a_checkpoint_the_existing_loader_reads(tmp_path):
    _torch()
    import json

    import train_chunk as tc

    assert _run_main(tmp_path, "--lr", "1e-3", arm="C", seeds="0,1") == 0
    for name in ("chunk_long", "chunk_long_s1"):
        model, ckpt = tc.load_chunk_checkpoint(tmp_path / "ckpt" / f"{name}.pt")
        cfg = ckpt["config"]
        assert (cfg["chunk_len"], cfg["n_exec"], cfg["ema_decay"], cfg["epochs"]) == (16, 8, 0.999, 2)
        assert cfg["n_parameters"] == 725_888 and cfg["lr"] == 1e-3
        assert ckpt["history"]["val_mae_px"][-1] > 0  # what evaluate_chunk.py reads
        report = json.loads((tmp_path / "rep" / f"training_{name}.json").read_text())
        assert report["final"]["val_mae_px"] == ckpt["history"]["val_mae_px"][-1]
    assert _load(tmp_path / "ckpt" / "chunk_long_s1.pt")["config"]["seed"] == 1


def test_cli_baseline_arm_checkpoint_loads_with_train_bc(tmp_path):
    _torch()
    import train_bc as tb

    assert _run_main(tmp_path, "--lr", "1e-3", arm="B") == 0
    model, ckpt = tb.load_checkpoint(tmp_path / "ckpt" / "bc_long.pt")
    assert ckpt["config"]["architecture"] == "BCPolicyNet" and ckpt["config"]["n_parameters"] == 718_178
    assert ckpt["config"]["chunk_len"] == 1 and ckpt["history"]["val_mae_px"][-1] > 0


def test_cli_diffusion_arm_checkpoint_loads_with_the_safe_loader(tmp_path):
    torch = _torch()
    import diffusion_model as dm

    assert _run_main(tmp_path, "--lr", "3e-4", arm="D") == 0
    ckpt = _load(tmp_path / "ckpt" / "diff_cnn.pt")
    net = dm.DiffusionPolicyNet(ckpt["config"]["chunk_len"])
    net.load_state_dict(ckpt["model_state"])
    cfg = ckpt["config"]
    assert cfg["architecture"] == "DiffusionPolicyNet" and cfg["n_parameters"] == 1_856_960
    assert cfg["diffusion"]["t_steps"] == 100 and cfg["diffusion"]["sampling_steps"] == 10
    assert (cfg["chunk_len"], cfg["n_exec"], cfg["lr"]) == (16, 8, 3e-4)
    assert np.isfinite(ckpt["history"]["val_mae_px"][-1])
    assert all(torch.isfinite(v).all() for v in ckpt["model_state"].values())


def test_cli_path_check_arm_has_no_ema_and_validates_every_epoch(tmp_path):
    _torch()
    assert _run_main(tmp_path, arm="P") == 0  # lr defaults to 1e-3
    ckpt = _load(tmp_path / "ckpt" / "chunk_res30.pt")
    assert ckpt["config"]["ema_decay"] is None and ckpt["config"]["lr"] == 1e-3
    assert ckpt["history"]["val_epoch"] == [1, 2]


def test_cli_saves_the_averaged_weights_not_the_raw_ones(tmp_path):
    """Same seed with and without EMA: the saved weights must differ (arm P and C share the net)."""
    torch = _torch()
    _run_main(tmp_path, "--lr", "1e-3", "--name", "with_ema", arm="C")
    _run_main(tmp_path, "--name", "no_ema", arm="P")
    a = _load(tmp_path / "ckpt" / "with_ema.pt")["model_state"]
    b = _load(tmp_path / "ckpt" / "no_ema.pt")["model_state"]
    assert any(not torch.equal(a[k], b[k]) for k in a)


def test_cli_refuses_to_overwrite_and_validates_arguments(tmp_path):
    _torch()
    assert _run_main(tmp_path, "--lr", "1e-3", arm="C") == 0
    with pytest.raises(FileExistsError):
        _run_main(tmp_path, "--lr", "1e-3", arm="C")
    assert _run_main(tmp_path, "--lr", "1e-3", "--overwrite", arm="C") == 0
    with pytest.raises(SystemExit):
        _run_main(tmp_path, arm="D")  # no --lr
    with pytest.raises(SystemExit):
        _run_main(tmp_path, "--lr", "1e-3", "--name", "x", arm="C", seeds="0,1")
    with pytest.raises(SystemExit):
        _run_main(tmp_path, "--lr", "3e-4", arm="P")  # P is lr 1e-3 only
