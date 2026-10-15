import json

import numpy as np
import pytest
import torch

import evaluate_chunk as ec
import train_bc as tb
import train_chunk as tc


class FakeObsEnv:
    """Minimal stand-in for the environment, with the observation layout the policy needs."""

    def __init__(self, limit=3):
        self.limit, self.t, self.closed = limit, 0, False

    def _obs(self):
        return {
            "pixels": np.full((96, 96, 3), 255, dtype=np.uint8),
            "agent_pos": np.array([256.0, 128.0]),
        }

    def reset(self, seed=None):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        info = {"coverage": 0.1, "is_success": False}
        return self._obs(), 0.0, False, self.t >= self.limit, info

    def close(self):
        self.closed = True


def save_tiny_checkpoint(path, chunk_len=4, n_exec=2):
    torch.manual_seed(0)
    tb.save_checkpoint(
        path,
        tc.ChunkPolicyNet(chunk_len),
        config={"chunk_len": chunk_len, "n_exec": n_exec},
        split={"train_episode_ids": [0], "val_episode_ids": [1]},
        history={"epoch": [1], "val_mae_px": [7.5]},
        references={},
    )


def test_main_writes_a_report_that_names_the_checkpoint_and_n_exec(tmp_path, capsys):
    ckpt, out, env = tmp_path / "m.pt", tmp_path / "eval.json", FakeObsEnv()
    save_tiny_checkpoint(ckpt)
    code = ec.main(
        ["--checkpoint", str(ckpt), "--episodes", "2", "--out", str(out)],
        env_factory=lambda: env,
    )
    assert code == 0 and env.closed is True
    saved = json.loads(out.read_text())
    assert saved["policy"] == "chunk_cnn"
    assert saved["checkpoint"]["config"] == {"chunk_len": 4, "n_exec": 2}
    assert (
        saved["checkpoint"]["n_exec"] == 2
        and saved["checkpoint"]["offline_val_mae_px"] == 7.5
    )
    assert [e["seed"] for e in saved["episodes"]] == [10000, 10001]
    assert "executing 2 of 4" in capsys.readouterr().out


def test_an_n_exec_override_is_used_and_recorded(tmp_path):
    ckpt, out = tmp_path / "m.pt", tmp_path / "eval.json"
    save_tiny_checkpoint(ckpt)
    ec.main(
        [
            "--checkpoint",
            str(ckpt),
            "--episodes",
            "1",
            "--n-exec",
            "3",
            "--out",
            str(out),
        ],
        env_factory=FakeObsEnv,
    )
    assert json.loads(out.read_text())["checkpoint"]["n_exec"] == 3


@pytest.mark.parametrize("n_exec", ["0", "5"])
def test_an_invalid_n_exec_override_is_rejected_before_anything_runs(tmp_path, n_exec):
    ckpt = tmp_path / "m.pt"
    save_tiny_checkpoint(ckpt)
    with pytest.raises(ValueError):
        ec.main(
            [
                "--checkpoint",
                str(ckpt),
                "--n-exec",
                n_exec,
                "--out",
                str(tmp_path / "x.json"),
            ],
            env_factory=FakeObsEnv,
        )


def test_a_missing_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        ec.main(
            ["--checkpoint", str(tmp_path / "nope.pt"), "--episodes", "1"],
            env_factory=FakeObsEnv,
        )


def test_the_default_output_path_is_named_after_the_policy(tmp_path, monkeypatch):
    ckpt = tmp_path / "m.pt"
    save_tiny_checkpoint(ckpt)
    monkeypatch.chdir(tmp_path)
    assert (
        ec.main(
            ["--checkpoint", str(ckpt), "--episodes", "1", "--name", "chunk_cnn_s9"],
            env_factory=FakeObsEnv,
        )
        == 0
    )
    assert (tmp_path / "reports" / "eval_chunk_cnn_s9.json").exists()
