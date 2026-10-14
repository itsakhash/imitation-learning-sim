import json
from pathlib import Path

import numpy as np
import pytest
import torch

import bc_policy as bp
import evaluate as ev
import inspect_episode as ie
import train_bc as tb


class FakeInspectEnv:
    """Scripted environment. The agent moves 5 px right per step, the block 1 px down,
    contact happens on odd steps, and coverage follows the given list. Like gym-pusht,
    the reset info carries no coverage; it is available through _get_coverage()."""

    def __init__(self, coverage=(0.1, 0.5, 0.3, 0.2), start=0.05):
        self.coverage = list(coverage)
        self.start = start
        self.t = 0

    def _info(self):
        return {
            "pos_agent": np.array([100.0 + 5 * self.t, 100.0]),
            "block_pose": np.array([200.0, 200.0 + self.t, 0.0]),
            "n_contacts": float(self.t % 2),
        }

    def _obs(self):
        return {"agent_pos": np.array([100.0 + 5 * self.t, 100.0]), "pixels": None}

    def reset(self, seed=None):
        self.t = 0
        return self._obs(), self._info()

    def step(self, action):
        self.t += 1
        info = self._info()
        cov = self.coverage[min(self.t, len(self.coverage)) - 1]
        info["coverage"] = cov
        info["is_success"] = cov > 0.95
        done = self.t >= len(self.coverage)
        return self._obs(), 0.0, bool(info["is_success"]), done, info

    def render(self):
        return np.full((20, 20, 3), (10 * self.t) % 256, dtype=np.uint8)

    def _get_coverage(self):
        return self.start

    def close(self):
        pass


def follow(obs):
    """Asks for a point 3 px right and 4 px below the agent, i.e. 5 px away."""
    return obs["agent_pos"] + np.array([3.0, 4.0])


def make_ep(agent, actions, coverage, start=0.0, seed=1):
    n = len(coverage)
    return {
        "seed": seed,
        "steps": n,
        "start_coverage": start,
        "coverage": np.asarray(coverage, dtype=np.float64),
        "agent": np.asarray(agent, dtype=np.float64),
        "block": np.zeros((n + 1, 3)),
        "actions": np.asarray(actions, dtype=np.float64),
        "contacts": np.zeros(n),
        "frames": {},
    }


# -------------------------------------------------------------------- replay


def test_replay_records_positions_actions_and_contacts():
    ep = ie.replay_episode(FakeInspectEnv(), follow, seed=7, frame_steps=(0, 2, 4))
    assert ep["seed"] == 7 and ep["steps"] == 4 and ep["success"] is False
    assert ep["start_coverage"] == pytest.approx(0.05)
    assert np.allclose(ep["coverage"], [0.1, 0.5, 0.3, 0.2])
    assert ep["agent"].shape == (5, 2)
    assert np.allclose(ep["agent"][:, 0], [100, 105, 110, 115, 120])
    assert ep["block"].shape == (5, 3)
    assert ep["actions"].shape == (4, 2) and np.allclose(
        ep["actions"][0], [103.0, 104.0]
    )
    assert np.allclose(ep["contacts"], [1, 0, 1, 0])
    assert set(ep["frames"]) == {0, 2, 4}
    assert int(ep["frames"][2][0, 0, 0]) == 20


def test_replay_stops_at_success_and_only_captures_reached_frames():
    ep = ie.replay_episode(
        FakeInspectEnv(coverage=(0.2, 0.96, 0.4)), follow, seed=1, frame_steps=(0, 2, 5)
    )
    assert ep["steps"] == 2 and ep["success"] is True
    assert set(ep["frames"]) == {0, 2}


def test_the_final_frame_is_always_captured():
    ep = ie.replay_episode(FakeInspectEnv(), follow, seed=1, frame_steps=())
    assert set(ep["frames"]) == {4}


def test_replay_respects_max_steps():
    ep = ie.replay_episode(
        FakeInspectEnv(), follow, seed=1, frame_steps=(), max_steps=2
    )
    assert ep["steps"] == 2 and set(ep["frames"]) == {2}


def test_replay_rejects_bad_max_steps():
    with pytest.raises(ValueError):
        ie.replay_episode(FakeInspectEnv(), follow, seed=1, max_steps=0)


def test_replay_gives_the_same_coverage_as_the_evaluation_loop():
    evaluated = ev.run_episode(FakeInspectEnv(), follow, seed=7)
    replayed = ie.replay_episode(FakeInspectEnv(), follow, seed=7)
    assert np.round(replayed["coverage"], 4).tolist() == evaluated["coverage_trace"]
    assert replayed["start_coverage"] == evaluated["start_coverage"]


def test_replay_resets_the_policy_with_the_seed():
    class Recording:
        def __init__(self):
            self.seeds = []

        def reset(self, seed):
            self.seeds.append(seed)

        def __call__(self, obs):
            return follow(obs)

    policy = Recording()
    ie.replay_episode(FakeInspectEnv(), policy, seed=42, frame_steps=())
    assert policy.seeds == [42]


# ----------------------------------------------------------------- statistics


def test_behavior_stats_known_values():
    ep = ie.replay_episode(FakeInspectEnv(), follow, seed=7, frame_steps=())
    s = ie.behavior_stats(ep)
    assert s["seed"] == 7 and s["steps"] == 4
    assert s["max_coverage"] == pytest.approx(0.5) and s["step_of_max"] == 2
    assert s["final_coverage"] == pytest.approx(0.2)
    assert s["drop_from_peak"] == pytest.approx(0.3)
    assert s["gain_over_start"] == pytest.approx(0.45)
    assert s["agent_path_px"] == pytest.approx(20.0)
    assert s["stalled_fraction"] == 0.0
    assert s["mean_action_gap_px"] == pytest.approx(5.0)
    assert s["border_action_fraction"] == 0.0
    assert s["block_net_displacement_px"] == pytest.approx(4.0)
    assert s["block_path_px"] == pytest.approx(4.0)
    assert s["contact_step_fraction"] == pytest.approx(0.5)


def test_stalled_fraction_counts_short_moves():
    ep = make_ep(
        agent=[[0, 0], [0.5, 0], [0.5, 0], [10, 0]],
        actions=[[100, 100]] * 3,
        coverage=[0.1, 0.1, 0.1],
    )
    assert ie.behavior_stats(ep)["stalled_fraction"] == pytest.approx(2 / 3)


def test_border_fraction_counts_actions_near_the_edge():
    ep = make_ep(
        agent=[[0, 0]] * 5,
        actions=[[2, 100], [100, 509], [100, 100], [300, 300]],
        coverage=[0.1] * 4,
    )
    assert ie.behavior_stats(ep)["border_action_fraction"] == pytest.approx(0.5)


def test_gain_over_start_is_none_without_a_start_coverage():
    ep = make_ep(
        agent=[[0, 0]] * 3, actions=[[1, 1]] * 2, coverage=[0.1, 0.2], start=None
    )
    s = ie.behavior_stats(ep)
    assert s["start_coverage"] is None and s["gain_over_start"] is None


def test_behavior_stats_rejects_an_empty_episode():
    ep = make_ep(agent=[[0, 0]], actions=np.zeros((0, 2)), coverage=[])
    with pytest.raises(ValueError):
        ie.behavior_stats(ep)


# ------------------------------------------------ checking against saved reports


def saved_report(n=2):
    results = ev.evaluate_policy(follow, FakeInspectEnv(), n)
    return ev.build_report("x", results)


def test_replay_matches_a_saved_report():
    ep = ie.replay_episode(FakeInspectEnv(), follow, seed=10001, frame_steps=())
    assert ie.replay_matches_report(ep, saved_report()) is True


def test_replay_detects_a_different_trace():
    report = saved_report()
    report["episodes"][1]["coverage_trace"][0] = 0.9
    ep = ie.replay_episode(FakeInspectEnv(), follow, seed=10001, frame_steps=())
    assert ie.replay_matches_report(ep, report) is False


def test_replay_detects_a_different_length():
    report = saved_report()
    report["episodes"][1]["coverage_trace"].pop()
    ep = ie.replay_episode(FakeInspectEnv(), follow, seed=10001, frame_steps=())
    assert ie.replay_matches_report(ep, report) is False


def test_replay_of_a_seed_missing_from_the_report_is_unchecked():
    ep = ie.replay_episode(FakeInspectEnv(), follow, seed=12345, frame_steps=())
    assert ie.replay_matches_report(ep, saved_report()) is None


# --------------------------------------------------------------------- figure


def png_ok(path):
    return path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 1000


def test_figure_is_written_as_a_png(tmp_path):
    ep = ie.replay_episode(FakeInspectEnv(), follow, seed=7, frame_steps=(0, 2, 4))
    path = tmp_path / "sub" / "fig.png"
    ie.save_inspection_figure(ep, path, title="x")
    assert png_ok(path)


def test_figure_handles_more_than_one_row_of_frames(tmp_path):
    ep = ie.replay_episode(
        FakeInspectEnv(), follow, seed=7, frame_steps=(0, 1, 2, 3, 4)
    )
    path = tmp_path / "fig.png"
    ie.save_inspection_figure(ep, path)
    assert png_ok(path)


def test_figure_without_a_start_coverage(tmp_path):
    ep = ie.replay_episode(FakeInspectEnv(), follow, seed=7, frame_steps=(0, 4))
    ep["start_coverage"] = None
    path = tmp_path / "fig.png"
    ie.save_inspection_figure(ep, path)
    assert png_ok(path)


def test_figure_needs_frames(tmp_path):
    ep = ie.replay_episode(FakeInspectEnv(), follow, seed=7, frame_steps=())
    ep["frames"] = {}
    with pytest.raises(ValueError):
        ie.save_inspection_figure(ep, tmp_path / "fig.png")


def test_format_stats_mentions_the_key_facts():
    ep = ie.replay_episode(FakeInspectEnv(), follow, seed=7, frame_steps=())
    text = ie.format_stats("hold", ie.behavior_stats(ep), True)
    assert "seed 7" in text and "peak 0.500 at step 2" in text and ": yes" in text


# -------------------------------------------------------------- building policies


def test_build_policy_known_names():
    assert isinstance(
        ie.build_policy("hold", FakeInspectEnv(), None), ev.HoldStillPolicy
    )
    env = ev.make_eval_env()
    try:
        assert isinstance(ie.build_policy("random", env, None), ev.SeededRandomPolicy)
    finally:
        env.close()


def test_build_policy_rejects_unknown_names():
    with pytest.raises(ValueError):
        ie.build_policy("nope", FakeInspectEnv(), None)


def test_build_policy_loads_a_checkpoint(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "m.pt"
    tb.save_checkpoint(
        path,
        tb.BCPolicyNet(),
        config={"epochs": 1},
        split={"train_episode_ids": [0], "val_episode_ids": [1]},
        history={"epoch": [1], "val_mae_px": [1.0]},
        references={},
    )
    assert isinstance(ie.build_policy("bc_cnn", None, path), bp.BCPolicy)


# ---------------------------------------------------------------- command line


def write_hold_report(directory, tamper=False):
    results = ev.evaluate_policy(ev.HoldStillPolicy(), FakeInspectEnv(), 2)
    if tamper:
        results[0]["coverage_trace"][0] = 0.99
    ev.write_report(directory / "eval_hold.json", ev.build_report("hold", results))


def run_main(tmp_path, seeds=("10000", "10001")):
    return ie.main(
        [
            "--policy",
            "hold",
            "--seed",
            *seeds,
            "--report-dir",
            str(tmp_path),
            "--frame-steps",
            "0",
            "2",
            "4",
        ],
        env_factory=FakeInspectEnv,
    )


def test_main_writes_pictures_and_summaries(tmp_path, capsys):
    write_hold_report(tmp_path)
    assert run_main(tmp_path) == 0
    for seed in (10000, 10001):
        assert png_ok(tmp_path / f"inspect_hold_{seed}.png")
        saved = json.loads((tmp_path / f"inspect_hold_{seed}.json").read_text())
        assert saved["replay_matches_saved_trace"] is True
        assert saved["frame_steps"] == [0, 2, 4]
    assert "matches the saved evaluation trace: yes" in capsys.readouterr().out


def test_main_fails_when_the_replay_does_not_match_the_saved_trace(tmp_path):
    write_hold_report(tmp_path, tamper=True)
    assert run_main(tmp_path) == 1
    assert (tmp_path / "inspect_hold_10000.png").exists()


def test_main_without_a_saved_report_still_works(tmp_path):
    assert run_main(tmp_path, seeds=("10000",)) == 0
    saved = json.loads((tmp_path / "inspect_hold_10000.json").read_text())
    assert saved["replay_matches_saved_trace"] is None


# ------------------------------------------- the real environment and reports

ROOT = Path(__file__).resolve().parents[1]
HOLD_REPORT = ROOT / "reports" / "eval_hold.json"
BC_REPORT = ROOT / "reports" / "eval_bc_cnn.json"
CHECKPOINT = ROOT / "checkpoints" / "bc_cnn.pt"


@pytest.mark.skipif(
    not HOLD_REPORT.exists(), reason="run evaluate.py --policy hold first"
)
def test_real_hold_replay_matches_the_saved_evaluation():
    env = ev.make_eval_env()
    try:
        ep = ie.replay_episode(
            env, ev.HoldStillPolicy(), 10000, frame_steps=(0, 50, 100)
        )
    finally:
        env.close()
    assert ie.replay_matches_report(ep, json.loads(HOLD_REPORT.read_text())) is True
    assert ep["start_coverage"] == pytest.approx(0.6339, abs=1e-3)
    s = ie.behavior_stats(ep)
    assert abs(s["gain_over_start"]) < 1e-3 and abs(s["drop_from_peak"]) < 1e-3


@pytest.mark.skipif(
    not (BC_REPORT.exists() and CHECKPOINT.exists()),
    reason="train and evaluate the BC network first",
)
def test_real_bc_replay_matches_the_saved_evaluation_and_the_comparison_numbers():
    model, _ = tb.load_checkpoint(CHECKPOINT)
    env = ev.make_eval_env()
    try:
        ep = ie.replay_episode(env, bp.BCPolicy(model), 10022)  # renders along the way
    finally:
        env.close()
    assert ie.replay_matches_report(ep, json.loads(BC_REPORT.read_text())) is True
    s = ie.behavior_stats(ep)
    # the paired comparison against doing nothing printed +0.848 (max) and +0.329 (final)
    assert s["gain_over_start"] == pytest.approx(0.848, abs=2e-3)
    assert s["final_coverage"] - s["start_coverage"] == pytest.approx(0.329, abs=2e-3)
