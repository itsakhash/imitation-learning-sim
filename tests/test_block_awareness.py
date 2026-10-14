import json
from pathlib import Path

import numpy as np
import pytest

import bc_policy as bp
import block_awareness as ba
import evaluate as ev
import inspect_episode as ie
import train_bc as tb


class FakeEnv:
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


def make_ep(agent, block, actions, seed=1):
    agent = np.asarray(agent, dtype=np.float64)
    block = np.asarray(block, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    n = len(actions)
    return {
        "seed": seed,
        "steps": n,
        "start_coverage": 0.0,
        "coverage": np.full(n, 0.1),
        "agent": agent,
        "block": block,
        "actions": actions,
        "contacts": np.zeros(n),
        "frames": {},
    }


def walker_ep(seed=1, step_px=10.0, n=8, block_x=300.0):
    """The agent walks along x towards a fixed block and always asks for a point ahead."""
    agent = np.stack([step_px * np.arange(n + 1), np.zeros(n + 1)], axis=1)
    block = np.tile([block_x, 0.0, 0.0], (n + 1, 1))
    return make_ep(agent, block, agent[:-1] + [20.0, 0.0], seed)


def parked_ep(seed=2, n=8):
    """The agent never moves and always asks for its own position."""
    agent = np.zeros((n + 1, 2))
    block = np.tile([300.0, 0.0, 0.0], (n + 1, 1))
    return make_ep(agent, block, agent[:-1], seed)


def approach_ep(seed=3):
    """The agent walks 25 px per step towards the block: 0 to 200 of the 300 px."""
    return walker_ep(seed=seed, step_px=25.0)


# -------------------------------------------------------------------- vectors


def test_step_vectors_use_the_state_before_each_step():
    ep = walker_ep()
    commanded, to_block = ba.step_vectors(ep)
    assert commanded.shape == (8, 2) and np.allclose(commanded, [20.0, 0.0])
    assert np.allclose(
        to_block[:, 0], 300.0 - 10.0 * np.arange(8)
    )  # agent x before step t


def test_step_vectors_reject_misshaped_episodes():
    ep = walker_ep()
    ep["agent"] = ep["agent"][:-1]
    with pytest.raises(ValueError):
        ba.step_vectors(ep)
    with pytest.raises(ValueError):
        ba.step_vectors(make_ep(np.zeros((1, 2)), np.zeros((1, 3)), np.zeros((0, 2))))


# changed: the earlier version used vectors exactly 1 px long, which are not "longer than 1 px"
def test_cosines_known_values():
    commanded = [[2, 0], [0, 2], [-2, 0], [2, 2], [0.5, 0], [2, 0]]
    to_block = [[3, 0], [3, 0], [3, 0], [3, 0], [3, 0], [0.5, 0]]
    c = ba.cosines(commanded, to_block)
    assert c[:4] == pytest.approx([1.0, 0.0, -1.0, 2**-0.5])
    assert np.isnan(c[4]) and np.isnan(
        c[5]
    )  # a vector of 1 px or less has no direction


# new
def test_a_vector_of_exactly_the_minimum_length_has_no_direction():
    c = ba.cosines([[1.0, 0.0], [1.01, 0.0]], [[3.0, 0.0], [3.0, 0.0]])
    assert np.isnan(c[0]) and c[1] == pytest.approx(1.0)


# new
def test_cosines_are_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    first = rng.normal(0, 10, size=(200, 2))
    second = rng.normal(0, 10, size=(200, 2))
    forward, backward = ba.cosines(first, second), ba.cosines(second, first)
    assert np.array_equal(np.isnan(forward), np.isnan(backward))
    valid = ~np.isnan(forward)
    assert np.allclose(forward[valid], backward[valid])
    assert (np.abs(forward[valid]) <= 1.0 + 1e-12).all()


@pytest.mark.parametrize(
    "a, b", [(np.zeros((3, 2)), np.zeros((4, 2))), (np.zeros((3, 3)), np.zeros((3, 3)))]
)
def test_cosines_reject_bad_shapes(a, b):
    with pytest.raises(ValueError):
        ba.cosines(a, b)


# ------------------------------------------------------------ episode statistics


def test_walker_heads_straight_for_the_block():
    s = ba.episode_block_stats(walker_ep(), tail_min=3)
    assert s["cos_count_by_phase"] == [2, 2, 2, 2] and s[
        "cos_sum_by_phase"
    ] == pytest.approx([2.0] * 4)
    assert s["first_step_cosine"] == pytest.approx(1.0)
    assert s["dist_by_phase"] == pytest.approx([295.0, 275.0, 255.0, 235.0])
    assert s["start_distance_px"] == pytest.approx(300.0)
    assert s["final_distance_px"] == pytest.approx(220.0)
    assert s["min_distance_px"] == pytest.approx(220.0)
    assert s["ends_frozen"] is False and s["steps"] == 8


def test_parked_agent_has_no_direction_and_never_gets_closer():
    s = ba.episode_block_stats(parked_ep(), tail_min=3)
    assert s["cos_count_by_phase"] == [0, 0, 0, 0] and s["first_step_cosine"] is None
    assert s["start_distance_px"] == pytest.approx(s["final_distance_px"])
    assert s["ends_frozen"] is True


def test_episode_stats_reject_bad_settings():
    with pytest.raises(ValueError):
        ba.episode_block_stats(walker_ep(), n_phases=0)
    with pytest.raises(ValueError):
        ba.episode_block_stats(walker_ep(n=3), n_phases=4)


# ------------------------------------------------------------------ aggregation


def three_stats():
    return [
        ba.episode_block_stats(walker_ep(1), tail_min=3),
        ba.episode_block_stats(parked_ep(2), tail_min=3),
        ba.episode_block_stats(approach_ep(3), tail_min=3),
    ]


def test_aggregate_pools_the_cosines_over_usable_steps():
    agg = ba.aggregate_block_stats(three_stats())
    assert agg["episodes"] == 3 and agg["n_phases"] == 4
    assert agg["cosine_by_phase"] == pytest.approx([1.0] * 4)
    assert agg["cosine_steps_by_phase"] == [
        4,
        4,
        4,
        4,
    ]  # the parked episode contributes none
    assert agg["first_step_cosine_mean"] == pytest.approx(1.0)
    assert agg["first_step_cosine_episodes"] == 2


def test_aggregate_median_distance_by_phase():
    agg = ba.aggregate_block_stats(three_stats())
    # phase 1 means: walker 295, parked 300, approach 287.5
    assert agg["median_distance_by_phase"][0] == pytest.approx(295.0)


def test_aggregate_groups_and_thresholds():
    agg = ba.aggregate_block_stats(three_stats())
    frozen, active = agg["frozen"], agg["active"]
    assert (
        frozen["episodes"] == 1
        and frozen["not_closer"] == 1
        and frozen["approached"] == 0
    )
    assert (
        frozen["median_start_px"]
        == pytest.approx(300.0)
        == pytest.approx(frozen["median_final_px"])
    )
    assert active["episodes"] == 2 and active["median_start_px"] == pytest.approx(300.0)
    assert active["median_final_px"] == pytest.approx(160.0)  # 220 and 100
    assert (
        active["approached"] == 1 and active["not_closer"] == 0
    )  # 100 < 150; 220 < 240


def test_aggregate_without_a_frozen_episode():
    agg = ba.aggregate_block_stats([ba.episode_block_stats(walker_ep(), tail_min=3)])
    assert agg["frozen"]["episodes"] == 0 and agg["frozen"]["median_final_px"] is None


def test_aggregate_with_no_usable_direction_at_all():
    agg = ba.aggregate_block_stats([ba.episode_block_stats(parked_ep(), tail_min=3)])
    assert (
        agg["cosine_by_phase"] == [None] * 4 and agg["first_step_cosine_mean"] is None
    )


def test_aggregate_rejects_no_episodes():
    with pytest.raises(ValueError):
        ba.aggregate_block_stats([])


def test_the_aggregate_is_json_serializable():
    json.dumps(ba.aggregate_block_stats(three_stats()))


def test_format_report_has_the_rows_and_the_note():
    text = ba.format_report(
        "bc_cnn",
        ba.aggregate_block_stats(three_stats()),
        {"yes": 3, "no": 0, "unchecked": 0},
    )
    for expected in (
        "phase 1",
        "mean cosine",
        "usable direction",
        "median distance",
        "first step",
        "ends frozen",
        "active to the end",
        "3 yes, 0 NO",
        "note:",
    ):
        assert expected in text


def test_format_report_handles_missing_numbers():
    text = ba.format_report(
        "hold",
        ba.aggregate_block_stats([ba.episode_block_stats(parked_ep(), tail_min=3)]),
    )
    assert "n/a" in text and "active to the end: 0 episodes" in text


# ----------------------------------------------------------------- the survey


def follow(obs):
    return obs["agent_pos"] + np.array([3.0, 4.0])


def test_survey_replays_each_seed_and_checks_the_saved_report():
    report = ev.build_report("x", ev.evaluate_policy(follow, FakeEnv(), 2))
    results = ba.survey_block_awareness(FakeEnv(), follow, ev.eval_seeds(2), report)
    assert [s["seed"] for s, _ in results] == [10000, 10001]
    assert [m for _, m in results] == [True, True]


def test_survey_detects_a_tampered_report():
    report = ev.build_report("x", ev.evaluate_policy(follow, FakeEnv(), 2))
    report["episodes"][0]["coverage_trace"][0] = 0.9
    results = ba.survey_block_awareness(FakeEnv(), follow, ev.eval_seeds(2), report)
    assert [m for _, m in results] == [False, True]


def test_survey_without_a_report_leaves_matches_unchecked():
    assert ba.survey_block_awareness(FakeEnv(), follow, [10000])[0][1] is None


def run_main(tmp_path, tamper=False):
    results = ev.evaluate_policy(ev.HoldStillPolicy(), FakeEnv(), 2)
    if tamper:
        results[0]["coverage_trace"][0] = 0.99
    ev.write_report(tmp_path / "eval_hold.json", ev.build_report("hold", results))
    return ba.main(
        ["--policy", "hold", "--episodes", "2", "--report-dir", str(tmp_path)],
        env_factory=FakeEnv,
    )


def test_main_writes_the_report_and_prints_it(tmp_path, capsys):
    assert run_main(tmp_path) == 0
    saved = json.loads((tmp_path / "block_awareness_hold.json").read_text())
    assert saved["policy"] == "hold" and saved["summary"]["episodes"] == 2
    assert saved["replay_matches"] == {"yes": 2, "no": 0, "unchecked": 0}
    assert len(saved["per_episode"]) == 2
    assert "phase 1" in capsys.readouterr().out


def test_main_fails_when_a_replay_does_not_match_the_saved_report(tmp_path):
    assert run_main(tmp_path, tamper=True) == 1


# ------------------------------------------- the real environment and network

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "bc_cnn.pt"
needs_network = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="train the BC network first: python train_bc.py"
)


def real_episode(policy, seed):
    env = ev.make_eval_env()
    try:
        return ie.replay_episode(env, policy, seed, frame_steps=())
    finally:
        env.close()


def test_real_do_nothing_policy_never_changes_the_distance_to_the_block():
    s = ba.episode_block_stats(real_episode(ev.HoldStillPolicy(), 10000))
    assert s["start_distance_px"] == pytest.approx(s["final_distance_px"], abs=1e-6)
    assert s["cos_count_by_phase"] == [0, 0, 0, 0] and s["first_step_cosine"] is None
    assert s["ends_frozen"] is True and s["steps"] == 300


@needs_network
def test_real_network_statistics_agree_with_the_replayed_episode():
    model, _ = tb.load_checkpoint(CHECKPOINT)
    ep = real_episode(bp.BCPolicy(model), 10022)
    s = ba.episode_block_stats(ep)
    agent, block = ep["agent"], ep["block"]
    assert s["start_distance_px"] == pytest.approx(
        np.linalg.norm(block[0, :2] - agent[0])
    )
    assert s["final_distance_px"] == pytest.approx(
        np.linalg.norm(block[-1, :2] - agent[-1])
    )
    assert s["steps"] == 300
    assert s["first_step_cosine"] is None or -1.0 <= s["first_step_cosine"] <= 1.0
    for total, count in zip(s["cos_sum_by_phase"], s["cos_count_by_phase"]):
        assert abs(total) <= count + 1e-9
