import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import behavior_survey as bs
import bc_policy as bp
import evaluate as ev
import pusht_data
import train_bc as tb


class FakeSurveyEnv:
    """Scripted environment: the agent moves 5 px right per step, the block 1 px down,
    contact happens on odd steps, coverage follows a fixed list. Like gym-pusht, the reset
    info has no coverage; it is available through _get_coverage()."""

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
        return (
            self._obs(),
            0.0,
            bool(info["is_success"]),
            self.t >= len(self.coverage),
            info,
        )

    def render(self):
        return np.zeros((20, 20, 3), dtype=np.uint8)

    def _get_coverage(self):
        return self.start

    def close(self):
        pass


def follow(obs):
    return obs["agent_pos"] + np.array([3.0, 4.0])


def stat(
    steps=100,
    stalled=0.5,
    gap=5.0,
    border=0.0,
    contact=0.1,
    block_path=10.0,
    drop=0.0,
    path=500.0,
):
    return {
        "steps": steps,
        "stalled_fraction": stalled,
        "mean_action_gap_px": gap,
        "border_action_fraction": border,
        "contact_step_fraction": contact,
        "block_path_px": block_path,
        "drop_from_peak": drop,
        "agent_path_px": path,
    }


def structured_arrays(n_episodes=12, length=16, seed=0):
    """Fake data where the action is 0.5 * position + 128 pixels."""
    rng = np.random.default_rng(seed)
    n = n_episodes * length
    states = rng.uniform(0, 512, size=(n, 2)).astype(np.float32)
    return pusht_data.PushTArrays(
        images=np.zeros((n, 96, 96, 3), dtype=np.uint8),
        states=states,
        actions=(0.5 * states + 128.0).astype(np.float32),
        episode_index=np.repeat(np.arange(n_episodes), length).astype(np.int64),
        frame_index=np.tile(np.arange(length), n_episodes).astype(np.int64),
        reward=np.zeros(n, dtype=np.float32),
        done=np.zeros(n, dtype=bool),
        success=np.zeros(n, dtype=bool),
        episode_lengths=np.full(n_episodes, length, dtype=np.int64),
    )


class AffineStub(nn.Module):
    """Answers scale * agent_pos + offset, ignoring the image (units scaled to [0, 1])."""

    def __init__(self, scale, offset):
        super().__init__()
        self.scale, self.offset = scale, offset

    def forward(self, image, agent_pos):
        return self.scale * agent_pos + self.offset


# ------------------------------------------------------------- demo statistics


def test_demo_motion_stats_known_values():
    states = [[0, 0], [3, 4], [3, 4], [10, 10], [10, 10]]
    offsets = [[3, 4], [0, 0], [6, 8], [0, 0], [0, 0]]
    actions = (np.array(states) + np.array(offsets)).tolist()
    s = bs.demo_motion_stats(states, actions, [0, 0, 0, 1, 1])
    assert s["episodes"] == 2 and s["frames"] == 5
    assert s["stalled_fraction"] == pytest.approx(2 / 3)  # moves are 5, 0, 0
    assert s["mean_move_px_per_step"] == pytest.approx(5 / 3)
    assert s["mean_action_gap_px"] == pytest.approx(3.0)  # gaps 5, 0, 10, 0, 0
    assert s["median_action_gap_px"] == pytest.approx(0.0)
    assert s["border_action_fraction"] == pytest.approx(
        0.4
    )  # the first two targets, (3, 4)


def test_demo_motion_stats_ignores_moves_across_episode_boundaries():
    s = bs.demo_motion_stats(
        [[0, 0], [0, 0], [100, 100], [100, 100]], [[50, 50]] * 4, [0, 0, 1, 1]
    )
    assert s["stalled_fraction"] == 1.0 and s["mean_move_px_per_step"] == 0.0


@pytest.mark.parametrize(
    "states, actions, episodes",
    [
        (np.zeros((3, 3)), np.zeros((3, 3)), [0, 0, 0]),
        (np.zeros((3, 2)), np.zeros((4, 2)), [0, 0, 0]),
        (np.zeros((3, 2)), np.zeros((3, 2)), [0, 0]),
        (np.zeros((2, 2)), np.zeros((2, 2)), [0, 1]),  # no step inside any episode
    ],
)
def test_demo_motion_stats_rejects_bad_input(states, actions, episodes):
    with pytest.raises(ValueError):
        bs.demo_motion_stats(states, actions, episodes)


# ----------------------------------------------------------------- aggregation


def test_aggregate_known_values():
    a = stat(steps=100, stalled=1.0, gap=10.0, border=0.2, contact=0.5, path=100.0)
    b = stat(steps=300, stalled=0.0, gap=2.0, border=0.0, contact=0.1, path=600.0)
    agg = bs.aggregate_behavior([a, b])
    assert agg["episodes"] == 2
    assert agg["stalled_fraction"] == pytest.approx(0.25)
    assert agg["stalled_fraction_median_episode"] == pytest.approx(0.5)
    assert agg["mean_move_px_per_step"] == pytest.approx(1.75)
    assert agg["mean_action_gap_px"] == pytest.approx(4.0)
    assert agg["median_episode_action_gap_px"] == pytest.approx(6.0)
    assert agg["border_action_fraction"] == pytest.approx(0.05)
    assert agg["contact_step_fraction"] == pytest.approx(0.2)


def test_aggregate_counts_use_strict_thresholds():
    stats = [
        stat(block_path=0.5, drop=0.1),
        stat(block_path=1.0, drop=0.11),
        stat(block_path=5.0, drop=0.5),
    ]
    agg = bs.aggregate_behavior(stats)
    assert agg["episodes_block_never_moved"] == 1
    assert agg["episodes_lost_coverage"] == 2


def test_aggregate_rejects_no_episodes():
    with pytest.raises(ValueError):
        bs.aggregate_behavior([])


# --------------------------------------------------------------------- survey


def test_survey_replays_each_seed_and_checks_the_saved_report():
    report = ev.build_report("x", ev.evaluate_policy(follow, FakeSurveyEnv(), 2))
    results = bs.survey_policy(FakeSurveyEnv(), follow, ev.eval_seeds(2), report)
    assert [s["seed"] for s, _ in results] == [10000, 10001]
    assert [m for _, m in results] == [True, True]
    assert all(s["steps"] == 4 for s, _ in results)


def test_survey_detects_a_tampered_report():
    report = ev.build_report("x", ev.evaluate_policy(follow, FakeSurveyEnv(), 2))
    report["episodes"][0]["coverage_trace"][0] = 0.9
    results = bs.survey_policy(FakeSurveyEnv(), follow, ev.eval_seeds(2), report)
    assert [m for _, m in results] == [False, True]


def test_survey_without_a_report_leaves_matches_unchecked():
    results = bs.survey_policy(FakeSurveyEnv(), follow, [10000], None)
    assert results[0][1] is None


# --------------------------------------------------------------- offline gaps


def test_offline_gap_stats_of_a_perfect_network():
    arrays = structured_arrays()
    s = bs.offline_gap_stats(AffineStub(0.5, 0.25), arrays, [0, 1, 2])
    assert s["episodes"] == 3 and s["frames"] == 48
    assert s["displacement_gain"] == pytest.approx(1.0, rel=1e-4)
    assert s["network_gap_px"] == pytest.approx(s["human_gap_px"], rel=1e-4)
    assert s["mean_cosine"] == pytest.approx(1.0, abs=1e-4)


def test_offline_gap_stats_of_a_network_that_shrinks_by_half():
    s = bs.offline_gap_stats(AffineStub(0.75, 0.125), structured_arrays(), [0, 1, 2])
    assert s["displacement_gain"] == pytest.approx(0.5, rel=1e-4)
    assert s["network_gap_px"] == pytest.approx(0.5 * s["human_gap_px"], rel=1e-4)
    assert s["mean_cosine"] == pytest.approx(1.0, abs=1e-4)


def test_offline_gap_stats_of_a_network_that_copies_the_agent():
    s = bs.offline_gap_stats(AffineStub(1.0, 0.0), structured_arrays(), [0, 1, 2])
    assert s["network_gap_px"] == pytest.approx(0.0, abs=1e-6)
    assert s["displacement_gain"] == pytest.approx(0.0, abs=1e-6)
    assert s["mean_cosine"] is None and s["cosine_frames"] == 0


def test_offline_gap_stats_rejects_unknown_episodes():
    with pytest.raises(ValueError):
        bs.offline_gap_stats(AffineStub(1.0, 0.0), structured_arrays(), [999])


# --------------------------------------------------------------------- output


def small_demo():
    return bs.demo_motion_stats([[0, 0], [3, 4], [3, 4]], [[3, 4]] * 3, [0, 0, 0])


def test_format_survey_shows_both_columns_and_the_key_lines():
    agg = bs.aggregate_behavior([stat(), stat(block_path=0.0)])
    text = bs.format_survey(
        "bc_cnn", agg, small_demo(), {"yes": 2, "no": 0, "unchecked": 0}
    )
    assert "human demos" in text and "bc_cnn" in text
    assert "standing still" in text and "never moved: 1 of 2" in text
    assert "2 yes, 0 NO, 0 unchecked" in text and "offline" not in text


def test_format_survey_includes_the_offline_line_when_given():
    offline = bs.offline_gap_stats(AffineStub(0.75, 0.125), structured_arrays(), [0, 1])
    text = bs.format_survey(
        "bc_cnn",
        bs.aggregate_behavior([stat()]),
        small_demo(),
        {"yes": 1, "no": 0, "unchecked": 0},
        offline,
    )
    assert "offline, on 2 held-out demonstration episodes" in text and "0.50" in text


# ---------------------------------------------------------------- command line


def write_hold_report(directory, tamper=False):
    results = ev.evaluate_policy(ev.HoldStillPolicy(), FakeSurveyEnv(), 2)
    if tamper:
        results[0]["coverage_trace"][0] = 0.99
    ev.write_report(directory / "eval_hold.json", ev.build_report("hold", results))


def fake_demo_table(root):
    return {
        "states": np.array([[0, 0], [3, 4], [3, 4], [10, 10]], dtype=np.float32),
        "actions": np.array([[3, 4], [3, 4], [9, 12], [10, 10]], dtype=np.float32),
        "episode_index": np.array([0, 0, 0, 1]),
    }


def run_main(tmp_path):
    return bs.main(
        ["--policy", "hold", "--episodes", "2", "--report-dir", str(tmp_path)],
        env_factory=FakeSurveyEnv,
        demo_loader=fake_demo_table,
    )


def test_main_writes_the_survey_and_prints_it(tmp_path, capsys):
    write_hold_report(tmp_path)
    assert run_main(tmp_path) == 0
    saved = json.loads((tmp_path / "behavior_hold.json").read_text())
    assert saved["policy"] == "hold" and saved["episodes"] == 2
    assert saved["replay_matches"] == {"yes": 2, "no": 0, "unchecked": 0}
    assert saved["offline"] is None and len(saved["per_episode"]) == 2
    assert "human demos" in capsys.readouterr().out


def test_main_fails_when_a_replay_does_not_match_the_saved_report(tmp_path):
    write_hold_report(tmp_path, tamper=True)
    assert run_main(tmp_path) == 1


# ------------------------------------------- the real demonstrations and network

ROOT = Path(__file__).resolve().parents[1]
BC_REPORT = ROOT / "reports" / "eval_bc_cnn.json"
CHECKPOINT = ROOT / "checkpoints" / "bc_cnn.pt"
needs_dataset = pytest.mark.skipif(
    not (pusht_data.DEFAULT_ROOT / pusht_data.VIDEO_FILE).exists(),
    reason="dataset not downloaded; run: python pusht_data.py download",
)
needs_network = pytest.mark.skipif(
    not (BC_REPORT.exists() and CHECKPOINT.exists()),
    reason="train and evaluate the BC network first",
)


@needs_dataset
def test_real_demo_statistics_agree_with_the_known_dataset_numbers():
    table = pusht_data.read_table(pusht_data.DEFAULT_ROOT)
    s = bs.demo_motion_stats(table["states"], table["actions"], table["episode_index"])
    assert s["episodes"] == 206 and s["frames"] == 25650
    # the mean per-coordinate |action - state| is 14.7476 px, which bounds the mean
    # Euclidean distance between 14.7476 * sqrt(2) = 20.86 and 2 * 14.7476 = 29.50
    assert 20.8 < s["mean_action_gap_px"] < 29.6
    assert 0.0 <= s["stalled_fraction"] <= 1.0


@needs_network
def test_real_survey_reproduces_the_three_inspected_episodes():
    model, _ = tb.load_checkpoint(CHECKPOINT)
    env = ev.make_eval_env()
    try:
        results = bs.survey_policy(
            env,
            bp.BCPolicy(model),
            [10022, 10034, 10000],
            json.loads(BC_REPORT.read_text()),
        )
    finally:
        env.close()
    expected = {
        10022: (0.848, 0.329, 0.24),
        10034: (0.670, 0.670, 0.74),
        10000: (0.634, 0.067, 0.55),
    }
    for stats, match in results:
        peak, final, still = expected[stats["seed"]]
        assert match is True
        assert stats["max_coverage"] == pytest.approx(peak, abs=1e-3)
        assert stats["final_coverage"] == pytest.approx(final, abs=1e-3)
        assert stats["stalled_fraction"] == pytest.approx(still, abs=6e-3)


@needs_network
@needs_dataset
def test_real_offline_stats_cover_the_validation_episodes():
    model, ckpt = tb.load_checkpoint(CHECKPOINT)
    s = bs.offline_gap_stats(
        model, pusht_data.load_arrays(), ckpt["split"]["val_episode_ids"]
    )
    assert s["episodes"] == 41 and s["frames"] == 4966
    assert s["human_gap_px"] > 0.0 and s["network_gap_px"] >= 0.0
