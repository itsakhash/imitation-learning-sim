from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import chunk_policy as cpol
import evaluate as ev
import train_chunk as tc


def make_obs(pos=(256.0, 128.0), fill=255):
    return {
        "pixels": np.full((96, 96, 3), fill, dtype=np.uint8),
        "agent_pos": np.array(pos, dtype=np.float64),
    }


class CountingStub(nn.Module):
    """Predicts a chunk that depends on the agent's x position: chunk position k has
    x = agent_x + 0.1 * k (scaled units) and y = 0.5. Counts how often it is asked."""

    def __init__(self, chunk_len=4):
        super().__init__()
        self.chunk_len = chunk_len
        self.calls = 0

    def forward(self, image, agent_pos):
        self.calls += 1
        k = torch.arange(self.chunk_len).float()[None, :]
        x = agent_pos[:, 0:1] + 0.1 * k
        return torch.stack([x, torch.full_like(x, 0.5)], dim=-1)


class ConstantStub(nn.Module):
    def __init__(self, x, y, chunk_len=4):
        super().__init__()
        self.chunk_len, self.x, self.y = chunk_len, x, y

    def forward(self, image, agent_pos):
        out = torch.empty(len(agent_pos), self.chunk_len, 2)
        out[..., 0], out[..., 1] = self.x, self.y
        return out


# ------------------------------------------------------------------ execution


def test_n_exec_targets_are_executed_before_the_next_prediction():
    stub = CountingStub(4)
    policy = cpol.ChunkPolicy(stub, n_exec=2)
    a1 = policy(make_obs(pos=(256.0, 128.0)))  # asks the network: x = 256
    a2 = policy(
        make_obs(pos=(0.0, 100.0))
    )  # from the queue, not from the new observation
    assert a1[0] == pytest.approx(256.0, abs=1e-3) and a2[0] == pytest.approx(
        307.2, abs=1e-3
    )
    assert stub.calls == 1
    a3 = policy(
        make_obs(pos=(0.0, 100.0))
    )  # queue empty: asks again, from the new observation
    assert stub.calls == 2 and a3[0] == pytest.approx(0.0, abs=1e-3)
    assert policy(make_obs())[0] == pytest.approx(51.2, abs=1e-3)
    assert policy.predictions == 2


def test_only_the_first_n_exec_targets_of_a_chunk_are_ever_used():
    policy = cpol.ChunkPolicy(CountingStub(4), n_exec=2)
    x = [policy(make_obs())[0] for _ in range(2)]
    assert x == pytest.approx(
        [256.0, 307.2], abs=1e-3
    )  # never 358.4 or 409.6 from this chunk


def test_n_exec_of_one_predicts_at_every_step():
    stub = CountingStub(4)
    policy = cpol.ChunkPolicy(stub, n_exec=1)
    for _ in range(5):
        policy(make_obs())
    assert stub.calls == 5 and policy.predictions == 5


def test_n_exec_equal_to_the_chunk_length_runs_the_whole_chunk():
    stub = CountingStub(4)
    policy = cpol.ChunkPolicy(stub, n_exec=4)
    x = [policy(make_obs())[0] for _ in range(4)]
    assert (
        x == pytest.approx([256.0, 307.2, 358.4, 409.6], abs=1e-3) and stub.calls == 1
    )
    policy(make_obs())
    assert stub.calls == 2


def test_reset_forgets_the_planned_targets():
    stub = CountingStub(4)
    policy = cpol.ChunkPolicy(stub, n_exec=2)
    policy(make_obs(pos=(256.0, 128.0)))
    policy.reset(seed=7)
    action = policy(make_obs(pos=(0.0, 100.0)))
    assert stub.calls == 2 and action[0] == pytest.approx(0.0, abs=1e-3)
    policy.reset()  # a seed is optional


def test_scaled_output_is_converted_back_to_world_pixels_and_clipped():
    assert np.allclose(
        cpol.ChunkPolicy(ConstantStub(0.5, 0.25), 2)(make_obs()), [256.0, 128.0]
    )
    assert np.allclose(
        cpol.ChunkPolicy(ConstantStub(5.0, -3.0), 2)(make_obs()), [512.0, 0.0]
    )


def test_action_is_a_float32_pair():
    action = cpol.ChunkPolicy(CountingStub(4), 2)(make_obs())
    assert action.shape == (2,) and action.dtype == np.float32


@pytest.mark.parametrize("n_exec", [0, 5, -1])
def test_n_exec_must_lie_between_one_and_the_chunk_length(n_exec):
    with pytest.raises(ValueError):
        cpol.ChunkPolicy(CountingStub(4), n_exec=n_exec)


def test_bad_observations_are_rejected_without_asking_the_network():
    stub = CountingStub(4)
    policy = cpol.ChunkPolicy(stub, 2)
    bad = make_obs()
    bad["pixels"] = np.zeros((48, 48, 3), np.uint8)
    with pytest.raises(ValueError):
        policy(bad)
    assert stub.calls == 0


def test_the_network_receives_a_scaled_channels_first_image_and_position():
    seen = {}

    class Recorder(CountingStub):
        def forward(self, image, agent_pos):
            seen["image"], seen["pos"] = image, agent_pos
            return super().forward(image, agent_pos)

    obs = make_obs(pos=(256.0, 128.0))
    obs["pixels"][5, 7] = (10, 20, 30)
    cpol.ChunkPolicy(Recorder(4), 2)(obs)
    assert (
        seen["image"].shape == (1, 3, 96, 96) and seen["image"].dtype == torch.float32
    )
    assert torch.allclose(
        seen["image"][0, :, 5, 7] * 255, torch.tensor([10.0, 20.0, 30.0]), atol=1e-3
    )
    assert torch.allclose(seen["pos"], torch.tensor([[0.5, 0.25]]))


def test_n_exec_of_one_equals_the_first_action_of_the_network_at_every_step():
    torch.manual_seed(0)
    model = tc.ChunkPolicyNet(16).eval()
    policy = cpol.ChunkPolicy(model, n_exec=1)
    rng = np.random.default_rng(0)
    for _ in range(3):
        obs = {
            "pixels": rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8),
            "agent_pos": rng.uniform(0, 512, size=2),
        }
        image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float()[None] / 255.0
        pos = torch.from_numpy(obs["agent_pos"].astype(np.float32))[None] / 512.0
        with torch.no_grad():
            expected = np.clip(model(image, pos)[0, 0].numpy() * 512.0, 0.0, 512.0)
        assert np.allclose(policy(obs), expected, atol=1e-3)


# ------------------------------------------------ the real environment and checkpoint

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "chunk_cnn.pt"
needs_chunk_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="train the chunked network first: python train_chunk.py",
)


def test_the_policy_runs_in_the_real_environment():
    torch.manual_seed(0)
    env = ev.make_eval_env()
    try:
        result = ev.run_episode(
            env, cpol.ChunkPolicy(tc.ChunkPolicyNet(16), 8), 10000, max_steps=20
        )
    finally:
        env.close()
    assert result["steps"] == 20 and 0.0 <= result["max_coverage"] <= 1.0


@needs_chunk_checkpoint
def test_real_checkpoint_evaluation_is_reproducible():
    def run():
        model, ckpt = tc.load_chunk_checkpoint(CHECKPOINT)
        policy = cpol.ChunkPolicy(model, ckpt["config"]["n_exec"])
        env = ev.make_eval_env()
        try:
            return ev.evaluate_policy(policy, env, 2)
        finally:
            env.close()

    assert run() == run()
