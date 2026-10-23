"""Tests for diffusion_core.py.

Everything up to the "with torch" marker needs only numpy. The sampling tests use an oracle
denoiser: given the true clean chunk it returns exactly the noise that produced x_t, so a correct
DDIM sampler must return that clean chunk. The torch tests (skipped without torch) check the
masked loss, the seeded noise and agreement between the numpy and torch code paths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import diffusion_core as dc

AB = dc.cosine_alpha_bars()


# ---------------------------------------------------------------- scaling


def test_scaling_maps_ends_and_round_trips():
    a = np.array([0.0, 0.25, 0.5, 1.0])
    x = dc.to_diffusion_scale(a)
    assert x.tolist() == [-1.0, -0.5, 0.0, 1.0]
    assert np.allclose(dc.from_diffusion_scale(x), a)


# ---------------------------------------------------------------- schedule


def test_schedule_length_range_and_monotone():
    assert len(AB) == dc.T_STEPS == 100
    assert np.all(AB > 0.0) and np.all(AB < 1.0)
    assert np.all(np.diff(AB) < 0.0)


def test_schedule_end_values():
    assert AB[0] > 0.999  # almost no noise after the first step
    assert 0.45 < AB[len(AB) // 2 - 1] < 0.55  # about half the signal at the middle
    assert AB[-1] < 1e-5  # essentially pure noise at the last step


def test_schedule_betas_are_valid_and_clipped():
    betas = 1.0 - AB[1:] / AB[:-1]
    assert np.all(betas > 0.0) and np.all(betas <= dc.MAX_BETA + 1e-12)
    assert 1.0 - AB[0] == pytest.approx(0.0006312815983416931)  # beta_0, from the formula


def test_schedule_is_deterministic_and_validates():
    assert np.array_equal(dc.cosine_alpha_bars(), dc.cosine_alpha_bars())
    with pytest.raises(ValueError):
        dc.cosine_alpha_bars(1)
    with pytest.raises(ValueError):
        dc.cosine_alpha_bars(100, max_beta=1.0)
    with pytest.raises(ValueError):
        dc.cosine_alpha_bars(100, offset=-0.1)


# ---------------------------------------------------------------- timesteps


def test_ddim_timesteps_default():
    ts = dc.ddim_timesteps()
    assert ts == [99, 88, 77, 66, 55, 44, 33, 22, 11, 0]


def test_ddim_timesteps_properties():
    for n in (2, 5, 10, 50, 100):
        ts = dc.ddim_timesteps(100, n)
        assert ts[0] == 99 and ts[-1] == 0
        assert len(ts) == n
        assert all(a > b for a, b in zip(ts, ts[1:]))


@pytest.mark.parametrize("n", [1, 101])
def test_ddim_timesteps_validates(n):
    with pytest.raises(ValueError):
        dc.ddim_timesteps(100, n)


# ---------------------------------------------------------------- forward process


def test_q_sample_values():
    x0 = np.full((2, 3, 2), 0.5)
    noise = np.full((2, 3, 2), -1.0)
    out = dc.q_sample(x0, noise, 0.64)
    assert np.allclose(out, 0.8 * 0.5 + 0.6 * -1.0)


def test_q_sample_identity_limits():
    rng = np.random.default_rng(0)
    x0, noise = rng.normal(size=(4, 16, 2)), rng.normal(size=(4, 16, 2))
    assert np.allclose(dc.q_sample(x0, noise, 1.0), x0)  # no noise
    assert np.allclose(dc.q_sample(x0, noise, 0.0), noise)  # pure noise


def test_q_sample_keeps_unit_variance():
    rng = np.random.default_rng(1)
    x0 = rng.choice([-1.0, 1.0], size=(200000,))  # unit variance
    noise = rng.normal(size=x0.shape)
    for ab in (0.9, 0.5, 0.1):
        assert np.var(dc.q_sample(x0, noise, ab)) == pytest.approx(1.0, abs=0.02)


def test_predict_x0_inverts_q_sample():
    rng = np.random.default_rng(2)
    x0, noise = rng.normal(size=(3, 16, 2)), rng.normal(size=(3, 16, 2))
    for t in (0, 30, 70, 98):
        x_t = dc.q_sample(x0, noise, AB[t])
        assert np.allclose(dc.predict_x0(x_t, noise, AB[t]), x0)


# ---------------------------------------------------------------- DDIM sampling


def oracle(x0):
    """A denoiser that knows the clean chunk: returns the noise that makes x_t from x0."""

    def model_fn(x, t):
        return (x - AB[t] ** 0.5 * x0) / (1.0 - AB[t]) ** 0.5

    return model_fn


def test_ddim_follows_the_exact_noising_path_with_the_true_noise():
    """With the TRUE noise as the prediction, every DDIM step must land exactly on the noised
    chunk of the next timestep: x_prev = q_sample(x0, noise, alpha_bar_prev). Unlike the oracle
    above (which recomputes the noise from whatever it is given and so forgives a wrong sign),
    this fixes the direction and the size of every step."""
    rng = np.random.default_rng(11)
    x0 = rng.uniform(-1, 1, size=(3, 16, 2))
    noise = rng.normal(size=x0.shape)
    ts = dc.ddim_timesteps()
    x = dc.q_sample(x0, noise, AB[ts[0]])
    for i, t in enumerate(ts):
        ab_prev = AB[ts[i + 1]] if i + 1 < len(ts) else 1.0
        x = dc.ddim_step(x, noise, AB[t], ab_prev, clip_x0=False)
        assert np.allclose(x, dc.q_sample(x0, noise, ab_prev), atol=1e-9)
    assert np.allclose(x, x0, atol=1e-9)
    full = dc.ddim_sample(
        lambda xx, t: noise, dc.q_sample(x0, noise, AB[ts[0]]), AB, clip_x0=False
    )
    assert np.allclose(full, x0, atol=1e-9)


def test_ddim_step_to_one_returns_x0():
    rng = np.random.default_rng(3)
    x0 = rng.uniform(-1, 1, size=(2, 16, 2))
    noise = rng.normal(size=x0.shape)
    x_t = dc.q_sample(x0, noise, AB[60])
    assert np.allclose(dc.ddim_step(x_t, noise, AB[60], 1.0), x0)


def test_oracle_ddim_recovers_the_clean_chunk():
    rng = np.random.default_rng(4)
    x0 = rng.uniform(-1, 1, size=(5, 16, 2))
    start = rng.normal(size=x0.shape)
    for n_steps in (2, 10, 50):
        out = dc.ddim_sample(
            oracle(x0), start, AB, dc.ddim_timesteps(100, n_steps), clip_x0=False
        )
        assert np.allclose(out, x0, atol=1e-6)


def test_oracle_ddim_with_clipping_matches_when_x0_is_in_range():
    rng = np.random.default_rng(5)
    x0 = rng.uniform(-1, 1, size=(3, 16, 2))
    start = rng.normal(size=x0.shape)
    a = dc.ddim_sample(oracle(x0), start, AB, clip_x0=True)
    b = dc.ddim_sample(oracle(x0), start, AB, clip_x0=False)
    assert np.allclose(a, x0, atol=1e-6) and np.allclose(a, b, atol=1e-9)


def test_clipping_bounds_the_output():
    """A denoiser that predicts an out-of-range clean chunk is pulled back into [-1, 1]."""
    rng = np.random.default_rng(6)
    wild = rng.uniform(-3, 3, size=(4, 16, 2))
    start = rng.normal(size=wild.shape)
    clipped = dc.ddim_sample(oracle(wild), start, AB, clip_x0=True)
    unclipped = dc.ddim_sample(oracle(wild), start, AB, clip_x0=False)
    assert np.all(np.abs(clipped) <= 1.0 + 1e-9)
    assert np.any(np.abs(unclipped) > 1.0)


def test_ddim_is_deterministic_and_depends_on_the_noise():
    x0 = np.random.default_rng(7).uniform(-1, 1, size=(2, 16, 2))
    flawed = lambda x, t: 0.3 * x  # not an oracle, so the start matters
    s1 = np.random.default_rng(8).normal(size=x0.shape)
    s2 = np.random.default_rng(9).normal(size=x0.shape)
    a1 = dc.ddim_sample(flawed, s1, AB)
    a2 = dc.ddim_sample(flawed, s1, AB)
    b = dc.ddim_sample(flawed, s2, AB)
    assert np.array_equal(a1, a2)
    assert not np.allclose(a1, b)


def test_ddim_calls_the_model_at_each_timestep_in_order():
    seen = []

    def model_fn(x, t):
        seen.append(t)
        return np.zeros_like(x)

    dc.ddim_sample(model_fn, np.zeros((1, 4, 2)), AB)
    assert seen == dc.ddim_timesteps()


@pytest.mark.parametrize("bad", [[], [5, 5], [3, 7], [100, 5], [5, -1]])
def test_ddim_sample_rejects_bad_timesteps(bad):
    with pytest.raises(ValueError):
        dc.ddim_sample(lambda x, t: x, np.zeros((1, 4, 2)), AB, bad)


# ---------------------------------------------------------------- with torch


def _torch():
    return pytest.importorskip("torch")


def test_seeded_noise_is_repeatable_and_seed_dependent():
    torch = _torch()
    a = dc.seeded_noise((2, 16, 2), seed=5)
    b = dc.seeded_noise((2, 16, 2), seed=5)
    c = dc.seeded_noise((2, 16, 2), seed=6)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == (2, 16, 2) and a.dtype == torch.float32


def test_torch_ddim_matches_numpy_ddim():
    torch = _torch()
    rng = np.random.default_rng(10)
    x0 = rng.uniform(-1, 1, size=(2, 16, 2))
    start = rng.normal(size=x0.shape)
    ab_t = torch.as_tensor(AB, dtype=torch.float64)

    def torch_oracle(x, t):
        return (x - float(ab_t[t]) ** 0.5 * torch.as_tensor(x0)) / (1.0 - float(ab_t[t])) ** 0.5

    out_t = dc.ddim_sample(torch_oracle, torch.as_tensor(start), AB)
    out_n = dc.ddim_sample(oracle(x0), start, AB)
    assert np.allclose(out_t.numpy(), out_n, atol=1e-9)


def _loss_inputs(torch, batch=6, k=16):
    g = torch.Generator().manual_seed(0)
    x0 = torch.rand(batch, k, 2, generator=g) * 2 - 1
    mask = torch.ones(batch, k)
    mask[:3, 10:] = 0.0  # three chunks padded past the episode's end
    return x0, mask, torch.as_tensor(AB, dtype=torch.float32)


def test_loss_is_zero_for_a_perfect_denoiser():
    torch = _torch()
    x0, mask, ab = _loss_inputs(torch)
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1))
    t = torch.randint(0, 100, (6,), generator=torch.Generator().manual_seed(2))
    loss = dc.diffusion_loss(lambda x, tt: noise, x0, mask, ab, t=t, noise=noise)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_loss_ignores_padded_positions():
    """The denoiser is perfect on valid positions and wildly wrong on padded ones: loss stays 0."""
    torch = _torch()
    x0, mask, ab = _loss_inputs(torch)
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1))
    t = torch.randint(0, 100, (6,), generator=torch.Generator().manual_seed(2))
    wrong = noise.clone()
    wrong[mask == 0] += 100.0
    loss = dc.diffusion_loss(lambda x, tt: wrong, x0, mask, ab, t=t, noise=noise)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_loss_counts_valid_positions_only_in_the_average():
    """A constant error of 2 on every valid position gives a loss of 4, whatever the padding."""
    torch = _torch()
    x0, mask, ab = _loss_inputs(torch)
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1))
    t = torch.randint(0, 100, (6,), generator=torch.Generator().manual_seed(2))
    loss = dc.diffusion_loss(lambda x, tt: noise + 2.0, x0, mask, ab, t=t, noise=noise)
    assert float(loss) == pytest.approx(4.0, rel=1e-6)


def test_loss_breaks_when_the_target_is_misaligned():
    """Deliberate-break check: predicting the noise of a different sample must give a large loss."""
    torch = _torch()
    x0, mask, ab = _loss_inputs(torch)
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1))
    t = torch.randint(0, 100, (6,), generator=torch.Generator().manual_seed(2))
    shifted = noise.roll(1, dims=0)  # the right noise for the wrong sample
    loss = dc.diffusion_loss(lambda x, tt: shifted, x0, mask, ab, t=t, noise=noise)
    assert float(loss) > 0.5


def test_loss_passes_the_noisy_chunk_and_step_indices_to_the_model():
    torch = _torch()
    x0, mask, ab = _loss_inputs(torch)
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1))
    t = torch.tensor([0, 10, 20, 30, 40, 99])
    got = {}

    def model_fn(x, tt):
        got["x"], got["t"] = x.clone(), tt.clone()
        return noise

    dc.diffusion_loss(model_fn, x0, mask, ab, t=t, noise=noise)
    expected = dc.q_sample(x0, noise, ab[t].view(6, 1, 1))
    assert torch.allclose(got["x"], expected)
    assert torch.equal(got["t"], t) and got["t"].dtype == torch.int64


def test_loss_draws_are_repeatable_with_a_seeded_generator():
    torch = _torch()
    x0, mask, ab = _loss_inputs(torch)
    model_fn = lambda x, tt: x * 0.5
    a = dc.diffusion_loss(model_fn, x0, mask, ab, generator=torch.Generator().manual_seed(3))
    b = dc.diffusion_loss(model_fn, x0, mask, ab, generator=torch.Generator().manual_seed(3))
    c = dc.diffusion_loss(model_fn, x0, mask, ab, generator=torch.Generator().manual_seed(4))
    assert float(a) == float(b)
    assert float(a) != float(c)


def test_loss_gradient_reaches_the_model():
    torch = _torch()
    x0, mask, ab = _loss_inputs(torch)
    weight = torch.nn.Parameter(torch.tensor(0.5))
    loss = dc.diffusion_loss(
        lambda x, tt: x * weight, x0, mask, ab, generator=torch.Generator().manual_seed(0)
    )
    loss.backward()
    assert weight.grad is not None and float(weight.grad.abs()) > 0.0
