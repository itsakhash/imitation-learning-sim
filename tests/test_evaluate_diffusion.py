"""Tests for evaluate_diffusion.py, with a fake environment (the pattern of test_evaluate_chunk.py)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import diffusion_policy as dp
import evaluate as ev
import evaluate_diffusion as ed
from test_diffusion_policy import save_diffusion_checkpoint


class FakeObsEnv:
    """Constant observations, so that a policy's actions depend only on its noise. Records the
    actions it receives, by the seed of the episode."""

    def __init__(self, limit=3):
        self.limit, self.t, self.closed = limit, 0, False
        self.seed = None
        self.actions = {}

    def _obs(self):
        return {
            "pixels": np.full((96, 96, 3), 255, dtype=np.uint8),
            "agent_pos": np.array([256.0, 128.0]),
        }

    def reset(self, seed=None):
        self.t, self.seed = 0, seed
        self.actions[seed] = []
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        self.actions[self.seed].append(np.array(action))
        info = {"coverage": 0.1, "is_success": False}
        return self._obs(), 0.0, False, self.t >= self.limit, info

    def close(self):
        self.closed = True


def test_main_writes_a_report_that_names_the_checkpoint_and_the_sampling(tmp_path, capsys):
    ckpt, out, env = tmp_path / "m.pt", tmp_path / "eval.json", FakeObsEnv()
    save_diffusion_checkpoint(ckpt)
    code = ed.main(["--checkpoint", str(ckpt), "--episodes", "2", "--out", str(out)], env_factory=lambda: env)
    assert code == 0 and env.closed is True
    saved = json.loads(out.read_text())
    assert saved["policy"] == "diff_cnn"
    c = saved["checkpoint"]
    assert c["n_exec"] == 8 and c["offline_val_mae_px"] == 7.5
    assert c["config"]["architecture"] == "DiffusionPolicyNet"
    assert c["sampling"]["steps"] == 10 and c["sampling"]["timesteps"][0] == 99
    assert c["sampling"]["clip_x0"] is True
    assert [e["seed"] for e in saved["episodes"]] == [10000, 10001]
    assert "executing 8 of 16" in capsys.readouterr().out


def test_an_n_exec_override_is_used_and_recorded(tmp_path):
    ckpt, out = tmp_path / "m.pt", tmp_path / "eval.json"
    save_diffusion_checkpoint(ckpt)
    ed.main(
        ["--checkpoint", str(ckpt), "--episodes", "1", "--n-exec", "3", "--out", str(out)],
        env_factory=FakeObsEnv,
    )
    assert json.loads(out.read_text())["checkpoint"]["n_exec"] == 3


@pytest.mark.parametrize("n_exec", ["0", "17"])
def test_an_invalid_n_exec_override_is_rejected_before_anything_runs(tmp_path, n_exec):
    ckpt = tmp_path / "m.pt"
    save_diffusion_checkpoint(ckpt)
    made = []
    with pytest.raises(ValueError):
        ed.main(
            ["--checkpoint", str(ckpt), "--n-exec", n_exec, "--out", str(tmp_path / "x.json")],
            env_factory=lambda: made.append(1) or FakeObsEnv(),
        )
    assert made == []  # no environment was even created


def test_a_missing_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        ed.main(["--checkpoint", str(tmp_path / "nope.pt"), "--episodes", "1"], env_factory=FakeObsEnv)


def test_the_default_output_path_is_named_after_the_policy(tmp_path, monkeypatch):
    ckpt = tmp_path / "m.pt"
    save_diffusion_checkpoint(ckpt)
    monkeypatch.chdir(tmp_path)
    assert (
        ed.main(["--checkpoint", str(ckpt), "--episodes", "1", "--name", "diff_cnn_s9"], env_factory=FakeObsEnv)
        == 0
    )
    assert (tmp_path / "reports" / "eval_diff_cnn_s9.json").exists()


def test_an_episode_repeats_exactly_and_does_not_depend_on_what_ran_before_it(tmp_path):
    """The reproducibility the study relies on, at the level of the actions sent to the
    environment: the same seed gives the same actions in a fresh process-like setting, and
    evaluating seed 10001 alone gives what it gave after seed 10000."""
    save_diffusion_checkpoint(tmp_path / "m.pt")

    def run(seeds):
        policy, _ = dp.DiffusionPolicy.from_checkpoint(tmp_path / "m.pt")
        env = FakeObsEnv(limit=20)
        for seed in seeds:
            ev.run_episode(env, policy, seed, max_steps=20)
        return env.actions

    both = run([10000, 10001])
    again = run([10000, 10001])
    alone = run([10001])
    for seed in (10000, 10001):
        assert np.array_equal(np.stack(both[seed]), np.stack(again[seed]))
    assert np.array_equal(np.stack(both[10001]), np.stack(alone[10001]))
    assert not np.array_equal(np.stack(both[10000]), np.stack(both[10001]))
