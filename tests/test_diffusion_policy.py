"""Tests for diffusion_policy.py (the policy wrapper and the checkpoint loader).

All of them need torch (skipped without it). The logic is tested with stub networks that have the
encode / denoise interface: an "oracle" that always denoises to a known chunk (so DDIM must return
exactly that chunk), and a "zero" network whose output depends on the initial noise. A few tests
use the real network and the real environment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

import arm_train as at
import diffusion_core as dc
import diffusion_model as dm
import diffusion_policy as dp
import evaluate as ev
import train_bc as tb
import train_chunk as tc

AB = dc.cosine_alpha_bars()


def make_obs(pos=(256.0, 128.0), fill=255):
    return {
        "pixels": np.full((96, 96, 3), fill, dtype=np.uint8),
        "agent_pos": np.array(pos, dtype=np.float64),
    }


class OracleNet(nn.Module):
    """Denoises every noisy chunk to the chunk with position k = agent_pos + step * k (dataset
    scale), so deterministic DDIM must return exactly that chunk (clipped to the action range).
    Records what it was shown and how often it was called."""

    def __init__(self, chunk_len=4, step=0.02):
        super().__init__()
        self.chunk_len, self.step = chunk_len, step
        self.encodes = 0
        self.denoises = 0
        self.seen = {}

    def encode(self, image, agent_pos):
        self.encodes += 1
        self.seen["image"], self.seen["pos"] = image, agent_pos
        return agent_pos

    def denoise(self, x_t, t, cond):
        self.denoises += 1
        k = torch.arange(self.chunk_len, dtype=x_t.dtype)[None, :, None]
        x0 = (cond[:, None, :] + self.step * k) * 2.0 - 1.0  # may leave [-1, 1]; DDIM clips it
        ab = float(AB[int(t[0])])
        return (x_t - ab**0.5 * x0) / (1.0 - ab) ** 0.5


class ZeroNet(nn.Module):
    """Predicts no noise at all, so the sampled chunk is a function of the initial noise only."""

    def __init__(self, chunk_len=4):
        super().__init__()
        self.chunk_len = chunk_len

    def encode(self, image, agent_pos):
        return agent_pos

    def denoise(self, x_t, t, cond):
        return torch.zeros_like(x_t)


def run_actions(policy, obs, n):
    return np.stack([policy(obs) for _ in range(n)])


# ---------------------------------------------------------------- conversion and execution


def test_oracle_denoiser_gives_the_planned_targets_in_world_pixels():
    policy = dp.DiffusionPolicy(OracleNet(4), n_exec=4)
    policy.reset(10000)
    got = run_actions(policy, make_obs(pos=(256.0, 128.0)), 4)
    expected = np.array([[256.0 + 10.24 * k, 128.0 + 10.24 * k] for k in range(4)])
    assert np.allclose(got, expected, atol=0.05)
    assert got.dtype == np.float32 and got.shape == (4, 2)


def test_targets_outside_the_action_range_are_clipped():
    high = dp.DiffusionPolicy(OracleNet(4, step=0.02), 4)
    high.reset(0)
    assert run_actions(high, make_obs(pos=(505.0, 300.0)), 4)[:, 0].max() == 512.0
    low = dp.DiffusionPolicy(OracleNet(4, step=-0.02), 4)
    low.reset(0)
    assert run_actions(low, make_obs(pos=(300.0, 5.0)), 4)[:, 1].min() == 0.0


def test_n_exec_targets_are_executed_before_the_next_prediction():
    net = OracleNet(4)
    policy = dp.DiffusionPolicy(net, n_exec=2)
    policy.reset(1)
    obs = make_obs(pos=(100.0, 100.0))
    a0, a1 = policy(obs), policy(obs)
    assert policy.predictions == 1 and net.encodes == 1
    assert np.allclose(a0, [100.0, 100.0], atol=0.05) and np.allclose(a1, [110.24, 110.24], atol=0.05)
    moved = make_obs(pos=(300.0, 200.0))  # a new prediction is made from the then-current observation
    a2 = policy(moved)
    assert policy.predictions == 2 and np.allclose(a2, [300.0, 200.0], atol=0.05)


def test_only_the_first_n_exec_targets_are_used_and_each_prediction_denoises_in_ten_steps():
    net = OracleNet(4)
    policy = dp.DiffusionPolicy(net, n_exec=1)
    policy.reset(1)
    obs = make_obs(pos=(100.0, 100.0))
    for _ in range(3):
        assert np.allclose(policy(obs), [100.0, 100.0], atol=0.05)  # always the first target
    assert policy.predictions == 3
    assert net.denoises == 3 * 10 and net.encodes == 3


def test_a_custom_schedule_sets_the_number_of_denoising_steps():
    net = OracleNet(4)
    policy = dp.DiffusionPolicy(net, 2, timesteps=[99, 50, 0])
    policy.reset(1)
    policy(make_obs())
    assert net.denoises == 3


def test_reset_forgets_the_planned_targets():
    net = OracleNet(4)
    policy = dp.DiffusionPolicy(net, n_exec=4)
    policy.reset(1)
    policy(make_obs(pos=(100.0, 100.0)))
    policy.reset(1)
    assert np.allclose(policy(make_obs(pos=(300.0, 300.0))), [300.0, 300.0], atol=0.05)


@pytest.mark.parametrize("n_exec", [0, 5, -1])
def test_n_exec_must_lie_between_one_and_the_chunk_length(n_exec):
    with pytest.raises(ValueError):
        dp.DiffusionPolicy(ZeroNet(4), n_exec=n_exec)


def test_the_network_receives_a_scaled_channels_first_image_and_position():
    net = OracleNet(4)
    obs = make_obs(pos=(256.0, 128.0))
    obs["pixels"][5, 7] = (10, 20, 30)
    policy = dp.DiffusionPolicy(net, 2)
    policy.reset(0)
    policy(obs)
    assert net.seen["image"].shape == (1, 3, 96, 96) and net.seen["image"].dtype == torch.float32
    assert torch.allclose(net.seen["image"][0, :, 5, 7] * 255, torch.tensor([10.0, 20.0, 30.0]), atol=1e-3)
    assert torch.allclose(net.seen["pos"], torch.tensor([[0.5, 0.25]]))


def test_bad_observations_are_rejected_without_asking_the_network():
    net = OracleNet(4)
    policy = dp.DiffusionPolicy(net, 2)
    policy.reset(0)
    bad = make_obs()
    bad["pixels"] = np.zeros((48, 48, 3), np.uint8)
    with pytest.raises(ValueError):
        policy(bad)
    assert net.encodes == 0


# ---------------------------------------------------------------- the seeded noise


def test_reset_needs_a_seed_and_the_policy_needs_a_reset():
    policy = dp.DiffusionPolicy(ZeroNet(4), 2)
    with pytest.raises(RuntimeError):
        policy(make_obs())  # never reset: there is no seeded noise
    with pytest.raises(ValueError):
        policy.reset()
    with pytest.raises(ValueError):
        policy.reset(None)


def test_the_first_prediction_uses_the_seeded_noise_of_the_episode_seed():
    """The noise is exactly diffusion_core.seeded_noise(shape, seed): checked against a direct
    sample of the same network from that noise."""
    net = ZeroNet(4)
    obs = make_obs(pos=(200.0, 300.0))
    policy = dp.DiffusionPolicy(net, 4)
    policy.reset(10007)
    got = run_actions(policy, obs, 4)
    image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float()[None] / 255.0
    pos = torch.tensor([[200.0, 300.0]]) / 512.0
    sample = dm.sample_chunk(net, image, pos, dc.seeded_noise((1, 4, 2), 10007), AB)
    expected = np.clip(dc.from_diffusion_scale(sample)[0].numpy() * 512.0, 0.0, 512.0)
    assert np.allclose(got, expected, atol=1e-3)


def test_an_episode_depends_only_on_its_seed():
    obs = make_obs()
    a = dp.DiffusionPolicy(ZeroNet(4), 2)
    b = dp.DiffusionPolicy(ZeroNet(4), 2)
    a.reset(5)
    first = run_actions(a, obs, 6)
    a.reset(6)  # another episode in between ...
    run_actions(a, obs, 3)
    b.reset(5)  # ... must not change what seed 5 gives, even in a fresh policy
    assert np.array_equal(run_actions(b, obs, 6), first)
    a.reset(5)
    assert np.array_equal(run_actions(a, obs, 6), first)


def test_different_seeds_give_different_episodes():
    obs = make_obs()
    a, b = dp.DiffusionPolicy(ZeroNet(4), 4), dp.DiffusionPolicy(ZeroNet(4), 4)
    a.reset(3)
    b.reset(4)
    assert not np.allclose(run_actions(a, obs, 4), run_actions(b, obs, 4))


def test_successive_predictions_in_an_episode_take_fresh_noise():
    obs = make_obs()
    policy = dp.DiffusionPolicy(ZeroNet(4), 1)  # predicts at every step
    policy.reset(3)
    first, second = policy(obs), policy(obs)  # the same observation, so only the noise differs
    assert policy.predictions == 2 and not np.allclose(first, second)


# ---------------------------------------------------------------- schedule from the checkpoint

GOOD_BLOCK = {
    "t_steps": 100,
    "sampling_steps": 10,
    "schedule": "squared cosine",
    "cosine_offset": 0.008,
    "max_beta": 0.999,
    "ddim_eta": 0.0,
}


def test_schedule_from_config_matches_the_core_defaults():
    alpha_bars, timesteps = dp.schedule_from_config({"diffusion": GOOD_BLOCK})
    assert np.array_equal(alpha_bars, dc.cosine_alpha_bars())
    assert timesteps == [99, 88, 77, 66, 55, 44, 33, 22, 11, 0]


def test_schedule_from_config_follows_the_stored_numbers_and_refuses_the_unknown():
    _, timesteps = dp.schedule_from_config({"diffusion": {**GOOD_BLOCK, "sampling_steps": 5}})
    assert len(timesteps) == 5 and timesteps[0] == 99 and timesteps[-1] == 0
    with pytest.raises(ValueError):
        dp.schedule_from_config({"diffusion": {**GOOD_BLOCK, "schedule": "linear"}})
    with pytest.raises(ValueError):
        dp.schedule_from_config({"diffusion": {**GOOD_BLOCK, "ddim_eta": 0.5}})
    with pytest.raises(KeyError):
        dp.schedule_from_config({})


# ---------------------------------------------------------------- checkpoints


def save_diffusion_checkpoint(path, seed=0):
    torch.manual_seed(seed)
    net = dm.DiffusionPolicyNet()
    config = at.build_config(
        "D", {"epochs": 1, "batch_size": 8, "lr": 1e-3, "seed": seed}, 1_856_960, [0, 1], 2, torch.device("cpu")
    )
    tb.save_checkpoint(
        path, net, config, {"train_episode_ids": [0], "val_episode_ids": [1]},
        {"epoch": [1], "val_mae_px": [7.5]}, {},
    )
    return net


def test_a_checkpoint_written_by_arm_train_round_trips(tmp_path):
    net = save_diffusion_checkpoint(tmp_path / "d.pt")
    model, ckpt = dp.load_diffusion_checkpoint(tmp_path / "d.pt")
    assert not model.training
    for k, v in net.state_dict().items():
        assert torch.equal(model.state_dict()[k], v)
    assert ckpt["config"]["architecture"] == "DiffusionPolicyNet"


def test_from_checkpoint_takes_n_exec_and_schedule_from_the_config(tmp_path):
    save_diffusion_checkpoint(tmp_path / "d.pt")
    policy, ckpt = dp.DiffusionPolicy.from_checkpoint(tmp_path / "d.pt")
    assert policy.n_exec == 8 and policy.chunk_len == 16
    assert policy.timesteps == [99, 88, 77, 66, 55, 44, 33, 22, 11, 0]
    assert np.array_equal(policy.alpha_bars, AB)
    assert dp.DiffusionPolicy.from_checkpoint(tmp_path / "d.pt", n_exec=3)[0].n_exec == 3


def test_a_checkpoint_of_another_network_is_refused(tmp_path):
    torch.manual_seed(0)
    config = at.build_config(
        "C", {"epochs": 1, "batch_size": 8, "lr": 1e-3, "seed": 0}, 725_888, [0, 1], 2, torch.device("cpu")
    )
    tb.save_checkpoint(
        tmp_path / "c.pt", tc.ChunkPolicyNet(16), config,
        {"train_episode_ids": [0], "val_episode_ids": [1]}, {"val_mae_px": [1.0]}, {},
    )
    with pytest.raises(ValueError):
        dp.load_diffusion_checkpoint(tmp_path / "c.pt")


def test_a_missing_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        dp.load_diffusion_checkpoint(tmp_path / "nope.pt")


# ---------------------------------------------------------------- the real network and environment


def test_the_real_network_gives_valid_repeatable_actions():
    torch.manual_seed(0)
    model = dm.DiffusionPolicyNet().eval()
    rng = np.random.default_rng(0)
    obs = {
        "pixels": rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8),
        "agent_pos": rng.uniform(0, 512, size=2),
    }
    first = dp.DiffusionPolicy(model, 8)
    second = dp.DiffusionPolicy(model, 8)
    first.reset(10000)
    second.reset(10000)
    a, b = run_actions(first, obs, 8), run_actions(second, obs, 8)
    assert a.shape == (8, 2) and a.dtype == np.float32
    assert (a >= 0).all() and (a <= 512).all() and np.isfinite(a).all()
    assert np.array_equal(a, b)
    other = dp.DiffusionPolicy(model, 8)
    other.reset(10001)
    assert not np.array_equal(run_actions(other, obs, 8), a)


def test_the_policy_runs_in_the_real_environment():
    pytest.importorskip("gym_pusht")
    torch.manual_seed(0)
    env = ev.make_eval_env()
    try:
        result = ev.run_episode(env, dp.DiffusionPolicy(dm.DiffusionPolicyNet(), 8), 10000, max_steps=20)
    finally:
        env.close()
    assert result["steps"] == 20 and 0.0 <= result["max_coverage"] <= 1.0
