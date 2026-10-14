import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import bc_policy as bp
import evaluate as ev
import evaluate_bc as eb
import train_bc as tb


def make_obs(pos=(256.0, 128.0)):
    return {
        "pixels": np.full((96, 96, 3), 255, dtype=np.uint8),
        "agent_pos": np.array(pos, dtype=np.float64),
    }


class Constant(nn.Module):
    """Answers the same scaled action whatever it is shown."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, image, agent_pos):
        return torch.full((len(agent_pos), 2), float(self.value))


class Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = None
        self.pos = None

    def forward(self, image, agent_pos):
        self.image, self.pos = image, agent_pos
        return torch.full((len(agent_pos), 2), 0.5)


class FakeObsEnv:
    """Minimal stand-in for the environment, with the observation layout the policy needs."""

    def __init__(self, limit=3):
        self.limit, self.t, self.closed = limit, 0, False

    def _obs(self):
        return make_obs()

    def reset(self, seed=None):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        info = {"coverage": 0.1, "is_success": False}
        return self._obs(), 0.0, False, self.t >= self.limit, info

    def close(self):
        self.closed = True


# -------------------------------------------------------------------- policy


def test_action_is_a_float32_pair_inside_the_action_range():
    torch.manual_seed(0)
    action = bp.BCPolicy(tb.BCPolicyNet())(make_obs())
    assert action.shape == (2,) and action.dtype == np.float32
    assert np.isfinite(action).all() and action.min() >= 0.0 and action.max() <= 512.0


def test_the_model_is_put_in_eval_mode():
    net = tb.BCPolicyNet().train()
    policy = bp.BCPolicy(net)
    assert policy.model.training is False


def test_the_policy_is_deterministic():
    torch.manual_seed(0)
    policy = bp.BCPolicy(tb.BCPolicyNet())
    assert np.array_equal(policy(make_obs()), policy(make_obs()))


def test_scaled_output_is_converted_back_to_world_pixels():
    action = bp.BCPolicy(Constant(0.5))(make_obs())
    assert np.allclose(action, [256.0, 256.0])


@pytest.mark.parametrize("value, expected", [(5.0, 512.0), (-3.0, 0.0)])
def test_out_of_range_output_is_clipped(value, expected):
    assert np.allclose(bp.BCPolicy(Constant(value))(make_obs()), [expected, expected])


def test_inputs_are_shaped_scaled_and_in_the_right_channel_order():
    obs = make_obs(pos=(256.0, 128.0))
    obs["pixels"][5, 7] = (10, 20, 30)  # row 5, column 7
    recorder = Recorder()
    bp.BCPolicy(recorder)(obs)
    assert (
        recorder.image.shape == (1, 3, 96, 96) and recorder.image.dtype == torch.float32
    )
    assert 0.0 <= float(recorder.image.min()) and float(recorder.image.max()) <= 1.0
    assert torch.allclose(
        recorder.image[0, :, 5, 7] * 255, torch.tensor([10.0, 20.0, 30.0]), atol=1e-3
    )
    assert torch.allclose(recorder.pos, torch.tensor([[0.5, 0.25]]))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o.update(pixels=np.zeros((48, 48, 3), np.uint8)),
        lambda o: o.update(pixels=np.zeros((96, 96, 3), np.float32)),
        lambda o: o.update(agent_pos=np.zeros(3)),
        lambda o: o.pop("pixels"),
    ],
)
def test_bad_observations_are_rejected(mutate):
    obs = make_obs()
    mutate(obs)
    with pytest.raises(ValueError):
        bp.BCPolicy(Constant(0.5))(obs)


def test_the_policy_runs_in_the_real_environment():
    torch.manual_seed(0)
    env = ev.make_eval_env()
    try:
        result = ev.run_episode(
            env, bp.BCPolicy(tb.BCPolicyNet()), seed=10000, max_steps=5
        )
    finally:
        env.close()
    assert result["steps"] == 5 and 0.0 <= result["max_coverage"] <= 1.0


# ------------------------------------------------------------ command line


def save_tiny_checkpoint(path):
    torch.manual_seed(0)
    tb.save_checkpoint(
        path,
        tb.BCPolicyNet(),
        config={"epochs": 2},
        split={"train_episode_ids": [0], "val_episode_ids": [1]},
        history={"epoch": [1], "val_mae_px": [7.5]},
        references={"copy_agent_position_mae_px": 1.0},
    )


def test_main_writes_a_report_that_names_the_checkpoint(tmp_path, capsys):
    ckpt, out, env = tmp_path / "m.pt", tmp_path / "eval.json", FakeObsEnv()
    save_tiny_checkpoint(ckpt)
    code = eb.main(
        ["--checkpoint", str(ckpt), "--episodes", "2", "--out", str(out)],
        env_factory=lambda: env,
    )
    assert code == 0 and env.closed is True
    saved = json.loads(out.read_text())
    assert saved["policy"] == "bc_cnn"
    assert saved["checkpoint"]["config"] == {"epochs": 2}
    assert saved["checkpoint"]["offline_val_mae_px"] == 7.5
    assert [e["seed"] for e in saved["episodes"]] == [10000, 10001]
    assert "offline validation MAE" in capsys.readouterr().out


def test_main_fails_clearly_when_the_checkpoint_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        eb.main(
            ["--checkpoint", str(tmp_path / "nope.pt"), "--episodes", "1"],
            env_factory=FakeObsEnv,
        )


# ------------------------------------------------- the real trained checkpoint

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "bc_cnn.pt"
needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="no trained checkpoint; run: python train_bc.py"
)


@needs_checkpoint
def test_real_checkpoint_loads_and_acts_on_a_real_observation():
    model, ckpt = tb.load_checkpoint(CHECKPOINT)
    env = ev.make_eval_env()
    try:
        obs, _ = env.reset(seed=10000)
        action = bp.BCPolicy(model)(obs)
    finally:
        env.close()
    assert action.shape == (2,) and 0.0 <= action.min() and action.max() <= 512.0
    assert ckpt["config"]["architecture"] == "BCPolicyNet"


@needs_checkpoint
def test_real_checkpoint_evaluation_is_reproducible():
    def run():
        model, _ = tb.load_checkpoint(CHECKPOINT)
        env = ev.make_eval_env()
        try:
            return ev.evaluate_policy(bp.BCPolicy(model), env, 2)
        finally:
            env.close()

    assert run() == run()
