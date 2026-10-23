"""Tests for diffusion_model.py. All of them need torch (the file is skipped without it)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import diffusion_core as dc
import diffusion_model as dm
from train_chunk import make_trunk

AB = dc.cosine_alpha_bars()


def make_inputs(batch=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.rand(batch, 16, 2, generator=g) * 2 - 1,
        "t": torch.randint(0, 100, (batch,), generator=g),
        "image": torch.rand(batch, 3, 96, 96, generator=g),
        "pos": torch.rand(batch, 2, generator=g),
    }


def make_net(seed=0, **kwargs):
    torch.manual_seed(seed)
    return dm.DiffusionPolicyNet(**kwargs).eval()


# ---------------------------------------------------------------- timestep embedding


def test_embedding_shape_and_values_at_zero():
    e = dm.sinusoidal_embedding(torch.tensor([0, 5, 99]), 64)
    assert e.shape == (3, 64)
    assert torch.allclose(e[0, :32], torch.zeros(32))  # sin(0)
    assert torch.allclose(e[0, 32:], torch.ones(32))  # cos(0)


def test_embedding_distinguishes_all_steps_and_is_bounded():
    e = dm.sinusoidal_embedding(torch.arange(100), 64)
    assert float(e.abs().max()) <= 1.0 + 1e-6
    dists = torch.cdist(e, e) + torch.eye(100) * 10.0
    assert float(dists.min()) > 1e-3  # no two steps share an embedding


@pytest.mark.parametrize("dim", [0, 3, 63])
def test_embedding_rejects_bad_dim(dim):
    with pytest.raises(ValueError):
        dm.sinusoidal_embedding(torch.tensor([1]), dim)


def test_embedding_rejects_wrong_rank():
    with pytest.raises(ValueError):
        dm.sinusoidal_embedding(torch.zeros(2, 2), 64)


# ---------------------------------------------------------------- structure


def test_trunk_is_the_chunk_networks_trunk_layer_for_layer():
    net = make_net()
    reference = make_trunk()
    mine = {k: tuple(v.shape) for k, v in net.trunk.state_dict().items()}
    theirs = {k: tuple(v.shape) for k, v in reference.state_dict().items()}
    assert mine == theirs and len(mine) == 8  # four convolutions, weight and bias each
    assert [type(m) for m in net.trunk] == [type(m) for m in reference]


def test_parameter_count_is_in_the_expected_range():
    n = sum(p.numel() for p in make_net().parameters())
    # by my count about 1.86 million (trunk 61,280); the exact figure goes in the README
    assert 1_700_000 < n < 2_000_000


def test_constructor_validates():
    for bad in ({"chunk_len": 0}, {"width": 0}, {"n_blocks": 0}, {"cond_dim": 0}, {"time_dim": 3}):
        with pytest.raises(ValueError):
            dm.DiffusionPolicyNet(**bad)


# ---------------------------------------------------------------- forward pass


def test_output_shape_and_finiteness():
    net = make_net()
    d = make_inputs()
    out = net(d["x"], d["t"], d["image"], d["pos"])
    assert out.shape == (3, 16, 2) and out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_other_chunk_lengths_work():
    net = make_net(chunk_len=4)
    d = make_inputs()
    out = net(d["x"][:, :4], d["t"], d["image"], d["pos"])
    assert out.shape == (3, 4, 2)


def test_forward_equals_encode_then_denoise():
    net = make_net()
    d = make_inputs()
    a = net(d["x"], d["t"], d["image"], d["pos"])
    b = net.denoise(d["x"], d["t"], net.encode(d["image"], d["pos"]))
    assert torch.equal(a, b)


def test_output_is_repeatable():
    net = make_net()
    d = make_inputs()
    assert torch.equal(
        net(d["x"], d["t"], d["image"], d["pos"]), net(d["x"], d["t"], d["image"], d["pos"])
    )


def test_every_input_changes_the_output():
    """The noisy chunk, the step, the image and the agent position must all matter."""
    net = make_net()
    d = make_inputs()
    base = net(d["x"], d["t"], d["image"], d["pos"])
    changed = {
        "x": net(d["x"] + 0.5, d["t"], d["image"], d["pos"]),
        "t": net(d["x"], (d["t"] + 7) % 100, d["image"], d["pos"]),
        "image": net(d["x"], d["t"], 1.0 - d["image"], d["pos"]),
        "pos": net(d["x"], d["t"], d["image"], 1.0 - d["pos"]),
    }
    for name, out in changed.items():
        # any difference at all proves a path from that input to the output; how strong the
        # path is at random initialisation is not what this test is about
        assert not torch.equal(out, base), f"{name} has no effect"


def test_examples_in_a_batch_do_not_affect_each_other():
    net = make_net()
    d = make_inputs()
    batched = net(d["x"], d["t"], d["image"], d["pos"])
    for i in range(3):
        single = net(d["x"][i : i + 1], d["t"][i : i + 1], d["image"][i : i + 1], d["pos"][i : i + 1])
        assert torch.allclose(single, batched[i : i + 1], atol=1e-4)


def test_denoise_validates_shapes():
    net = make_net()
    d = make_inputs()
    cond = net.encode(d["image"], d["pos"])
    with pytest.raises(ValueError):
        net.denoise(d["x"][:, :8], d["t"], cond)  # wrong chunk length
    with pytest.raises(ValueError):
        net.denoise(d["x"], d["t"][:2], cond)  # wrong number of step indices
    with pytest.raises(ValueError):
        net.denoise(d["x"].reshape(3, 32), d["t"], cond)  # not (B, K, 2)


# ---------------------------------------------------------------- training


def test_every_parameter_receives_a_gradient():
    net = make_net().train()
    d = make_inputs()
    x0 = d["x"]
    mask = torch.ones(3, 16)
    ab = torch.as_tensor(AB, dtype=torch.float32)
    cond = net.encode(d["image"], d["pos"])
    loss = dc.diffusion_loss(
        lambda xt, t: net.denoise(xt, t, cond), x0, mask, ab, generator=torch.Generator().manual_seed(0)
    )
    loss.backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert missing == []


def test_network_can_memorise_a_fixed_batch():
    """Deliberate-break check for the whole training chain (loss, backward, Adam): with the step
    indices and the noise held fixed, the same batch must be fitted, so the loss must fall to a
    fraction of its start."""
    net = make_net(seed=1).train()
    d = make_inputs(batch=4, seed=1)
    mask = torch.ones(4, 16)
    ab = torch.as_tensor(AB, dtype=torch.float32)
    t = torch.tensor([10, 40, 70, 95])
    noise = torch.randn(4, 16, 2, generator=torch.Generator().manual_seed(7))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(100):
        cond = net.encode(d["image"], d["pos"])
        loss = dc.diffusion_loss(
            lambda xt, tt: net.denoise(xt, tt, cond), d["x"], mask, ab, t=t, noise=noise
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]


# ---------------------------------------------------------------- sampling


def test_sample_chunk_shape_range_and_determinism():
    net = make_net()
    d = make_inputs()
    noise = dc.seeded_noise((3, 16, 2), seed=4)
    a = dm.sample_chunk(net, d["image"], d["pos"], noise, AB)
    b = dm.sample_chunk(net, d["image"], d["pos"], noise, AB)
    c = dm.sample_chunk(net, d["image"], d["pos"], dc.seeded_noise((3, 16, 2), seed=5), AB)
    assert a.shape == (3, 16, 2)
    assert float(a.abs().max()) <= 1.0 + 1e-6
    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


def test_sample_chunk_encodes_once_and_denoises_once_per_step():
    net = make_net()
    d = make_inputs()
    calls = {"encode": 0, "denoise": 0}
    encode, denoise = net.encode, net.denoise

    def counting_encode(*args, **kwargs):
        calls["encode"] += 1
        return encode(*args, **kwargs)

    def counting_denoise(*args, **kwargs):
        calls["denoise"] += 1
        return denoise(*args, **kwargs)

    net.encode, net.denoise = counting_encode, counting_denoise
    dm.sample_chunk(net, d["image"], d["pos"], dc.seeded_noise((3, 16, 2), seed=0), AB)
    assert calls == {"encode": 1, "denoise": 10}


def test_sample_chunk_leaves_no_gradients_and_sets_eval_mode():
    net = make_net().train()
    d = make_inputs()
    out = dm.sample_chunk(net, d["image"], d["pos"], dc.seeded_noise((3, 16, 2), seed=0), AB)
    assert not out.requires_grad
    assert not net.training


def test_sample_chunk_depends_on_the_image():
    net = make_net()
    d = make_inputs()
    noise = dc.seeded_noise((3, 16, 2), seed=0)
    a = dm.sample_chunk(net, d["image"], d["pos"], noise, AB)
    b = dm.sample_chunk(net, 1.0 - d["image"], d["pos"], noise, AB)
    assert not torch.equal(a, b)
