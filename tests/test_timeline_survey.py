import json
from pathlib import Path

import numpy as np
import pytest

import bc_policy as bp
import evaluate as ev
import inspect_episode as ie
import timeline_survey as tl
import train_bc as tb


class FakeTimelineEnv:
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


def make_ep(moves, contacts, gaps=None, coverage=None, start=0.0, seed=1):
    """An episode in which the agent moves along x by the given amount at each step."""
    moves = np.asarray(moves, dtype=np.float64)
    n = len(moves)
    x = np.concatenate([[0.0], np.cumsum(moves)])
    agent = np.stack([x, np.zeros(n + 1)], axis=1)
    gaps = np.full(n, 5.0) if gaps is None else np.asarray(gaps, dtype=np.float64)
    actions = agent[:-1] + np.stack([gaps, np.zeros(n)], axis=1)
    coverage = (
        np.full(n, 0.1) if coverage is None else np.asarray(coverage, dtype=np.float64)
    )
    return {
        "seed": seed,
        "steps": n,
        "start_coverage": start,
        "coverage": coverage,
        "agent": agent,
        "block": np.zeros((n + 1, 3)),
        "actions": actions,
        "contacts": np.asarray(contacts, dtype=np.float64),
        "frames": {},
    }


def stats_for(moves, contacts, coverage=None, start=0.0, tail_min=3, seed=1):
    return tl.timeline_stats(
        make_ep(moves, contacts, coverage=coverage, start=start, seed=seed),
        tail_min=tail_min,
    )


# ---------------------------------------------------------------- phase means


def test_phase_means_known_values():
    assert tl.phase_means([1, 1, 0, 0], 2) == [1.0, 0.0]


def test_phase_means_uneven_split_puts_the_extra_value_in_the_early_phase():
    assert tl.phase_means([1, 2, 3, 4, 5], 2) == [2.0, 4.5]


@pytest.mark.parametrize(
    "values, n", [([1, 2, 3], 0), ([1, 2, 3], 4), ([[1, 2], [3, 4]], 2)]
)
def test_phase_means_rejects_bad_arguments(values, n):
    with pytest.raises(ValueError):
        tl.phase_means(values, n)


# ----------------------------------------------------------- one episode


def test_timeline_known_values():
    s = stats_for([5, 5, 0, 0, 5, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0])
    assert s["steps"] == 8 and s["first_move_step"] == 1 and s["last_move_step"] == 5
    assert s["frozen_tail_steps"] == 3 and s["ends_frozen"] is True
    assert s["starts_in_contact"] is False and s["engaged"] is True
    assert (s["first_contact_step"], s["last_contact_step"], s["contact_steps"]) == (
        3,
        4,
        2,
    )
    assert s["stalled_by_phase"] == pytest.approx([0.0, 1.0, 0.5, 1.0])
    assert s["contact_by_phase"] == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert s["gap_by_phase"] == pytest.approx([5.0] * 4)


def test_the_frozen_threshold_follows_tail_min():
    moves, contacts = [5, 5, 0, 0, 5, 0, 0, 0], [0] * 8
    assert stats_for(moves, contacts, tail_min=3)["ends_frozen"] is True
    assert stats_for(moves, contacts, tail_min=4)["ends_frozen"] is False


def test_an_agent_that_never_moves():
    s = stats_for([0] * 8, [0] * 8, tail_min=8)
    assert s["first_move_step"] is None and s["last_move_step"] is None
    assert s["frozen_tail_steps"] == 8 and s["ends_frozen"] is True
    assert s["engaged"] is False and s["first_contact_step"] is None
    assert s["last_contact_step"] is None and s["contact_steps"] == 0


def test_an_episode_that_starts_in_contact_is_not_classified():
    s = stats_for([5] * 8, [1, 1, 0, 0, 0, 0, 0, 0])
    assert s["starts_in_contact"] is True and s["engaged"] is None
    assert (s["first_contact_step"], s["last_contact_step"]) == (1, 2)


def test_moves_under_one_pixel_do_not_count_as_moving():
    s = stats_for([0.9, 1.5, 0.9, 0.9], [0] * 4, tail_min=2)
    assert s["first_move_step"] == 2 and s["last_move_step"] == 2
    assert s["frozen_tail_steps"] == 2 and s["ends_frozen"] is True


def test_an_episode_shorter_than_the_number_of_phases_is_rejected():
    with pytest.raises(ValueError):
        tl.timeline_stats(make_ep([5, 5, 5], [0, 0, 0]))


def test_phase_shares_average_to_the_whole_episode_share():
    ep = make_ep([5, 0, 0, 5, 0, 5, 5, 0], [0, 1, 0, 0, 1, 1, 0, 0])
    s = tl.timeline_stats(ep)
    assert np.mean(s["stalled_by_phase"]) == pytest.approx(
        ie.behavior_stats(ep)["stalled_fraction"]
    )
    assert np.mean(s["contact_by_phase"]) == pytest.approx(
        ie.behavior_stats(ep)["contact_step_fraction"]
    )


# ------------------------------------------------------------------ aggregation


def four_episodes():
    return [
        stats_for(
            [5, 5, 5, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            coverage=[0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
            seed=1,
        ),  # frozen, engaged
        stats_for(
            [5] * 8, [0] * 8, coverage=[0.2] * 8, seed=2
        ),  # active, never touched
        stats_for(
            [0] * 8, [0] * 8, coverage=[0.0] * 8, seed=3
        ),  # frozen, never touched
        stats_for([5] * 8, [1, 1, 0, 0, 0, 0, 0, 0], seed=4),  # starts in contact
    ]


def test_aggregate_known_values():
    stats = four_episodes()
    agg = tl.aggregate_timelines(stats)
    assert agg["episodes"] == 4 and agg["n_phases"] == 4 and agg["frozen_tail_min"] == 3
    assert (
        agg["ends_frozen"] == 2
        and agg["never_moved"] == 1
        and agg["starts_in_contact"] == 1
    )
    assert agg["median_frozen_tail_steps"] == pytest.approx(2.5)  # tails 5, 0, 8, 0
    assert agg["median_last_move_step"] == pytest.approx(
        8.0
    )  # 3, 8, 8; the idle one is skipped
    assert agg["median_first_contact_step"] == pytest.approx(
        2.0
    )  # only the first episode engaged
    t = agg["table"]
    assert t["frozen_engaged"] == {
        "episodes": 1,
        "mean_gain_over_start": pytest.approx(0.4),
    }
    assert t["frozen_not_engaged"] == {
        "episodes": 1,
        "mean_gain_over_start": pytest.approx(0.0),
    }
    assert t["active_engaged"] == {"episodes": 0, "mean_gain_over_start": None}
    assert t["active_not_engaged"] == {
        "episodes": 1,
        "mean_gain_over_start": pytest.approx(0.2),
    }
    expected = np.mean([s["stalled_by_phase"] for s in stats], axis=0)
    assert agg["stalled_by_phase_mean"] == pytest.approx(expected.tolist())


def test_aggregate_with_only_unclassified_episodes():
    stats = [stats_for([5] * 8, [1, 0, 0, 0, 0, 0, 0, 0], seed=s) for s in (1, 2)]
    agg = tl.aggregate_timelines(stats)
    assert agg["starts_in_contact"] == 2 and agg["median_first_contact_step"] is None
    assert all(
        g["episodes"] == 0 and g["mean_gain_over_start"] is None
        for g in agg["table"].values()
    )


def test_aggregate_rejects_no_episodes():
    with pytest.raises(ValueError):
        tl.aggregate_timelines([])


# ------------------------------------------------------------------- the survey


def hold_report(n=2):
    return ev.build_report(
        "hold", ev.evaluate_policy(ev.HoldStillPolicy(), FakeTimelineEnv(), n)
    )


def test_survey_replays_each_seed_and_reads_the_timeline():
    results = tl.survey_timelines(
        FakeTimelineEnv(), follow, ev.eval_seeds(2), hold_report_for_follow()
    )
    assert [s["seed"] for s, _ in results] == [10000, 10001]
    stats, match = results[0]
    assert match is True
    assert stats["first_move_step"] == 1 and stats["last_move_step"] == 4
    assert stats["ends_frozen"] is False and stats["starts_in_contact"] is True


def hold_report_for_follow(n=2):
    return ev.build_report("follow", ev.evaluate_policy(follow, FakeTimelineEnv(), n))


def test_survey_detects_a_tampered_report():
    report = hold_report_for_follow()
    report["episodes"][0]["coverage_trace"][0] = 0.9
    results = tl.survey_timelines(FakeTimelineEnv(), follow, ev.eval_seeds(2), report)
    assert [m for _, m in results] == [False, True]


def test_survey_without_a_report_leaves_matches_unchecked():
    assert tl.survey_timelines(FakeTimelineEnv(), follow, [10000])[0][1] is None


# ----------------------------------------------------------------------- output


def test_format_timeline_shows_the_phases_and_the_table():
    agg = tl.aggregate_timelines(four_episodes())
    text = tl.format_timeline("bc_cnn", agg, {"yes": 4, "no": 0, "unchecked": 0})
    assert "phase 1" in text and "phase 4" in text
    assert (
        "ends frozen" in text
        and "touched something" in text
        and "never touched anything" in text
    )
    assert "1 episodes start in contact" in text
    assert "4 yes, 0 NO, 0 unchecked" in text and "note:" in text


def run_main(tmp_path, tamper=False):
    report = hold_report()
    if tamper:
        report["episodes"][0]["coverage_trace"][0] = 0.99
    ev.write_report(tmp_path / "eval_hold.json", report)
    return tl.main(
        ["--policy", "hold", "--episodes", "2", "--report-dir", str(tmp_path)],
        env_factory=FakeTimelineEnv,
    )


def test_main_writes_the_timeline_and_prints_it(tmp_path, capsys):
    assert run_main(tmp_path) == 0
    saved = json.loads((tmp_path / "timeline_hold.json").read_text())
    assert saved["policy"] == "hold" and saved["summary"]["episodes"] == 2
    assert saved["replay_matches"] == {"yes": 2, "no": 0, "unchecked": 0}
    assert len(saved["per_episode"]) == 2
    assert "phase 1" in capsys.readouterr().out


def test_main_fails_when_a_replay_does_not_match_the_saved_report(tmp_path):
    assert run_main(tmp_path, tamper=True) == 1


# ---------------------------------------------- the real environment and network

ROOT = Path(__file__).resolve().parents[1]
BC_REPORT = ROOT / "reports" / "eval_bc_cnn.json"
CHECKPOINT = ROOT / "checkpoints" / "bc_cnn.pt"
needs_network = pytest.mark.skipif(
    not (BC_REPORT.exists() and CHECKPOINT.exists()),
    reason="train and evaluate the BC network first",
)


def real_episode(policy_factory, seed):
    env = ev.make_eval_env()
    try:
        return ie.replay_episode(env, policy_factory(), seed, frame_steps=())
    finally:
        env.close()


def test_real_do_nothing_policy_stays_frozen_from_the_start():
    s = tl.timeline_stats(real_episode(ev.HoldStillPolicy, 10000))
    assert s["steps"] == 300 and s["ends_frozen"] is True
    assert np.mean(s["stalled_by_phase"]) > 0.99


def test_real_do_nothing_policy_starts_in_contact_on_seed_10015():
    # Seen in the survey: doing nothing already has contact on every step of this start.
    s = tl.timeline_stats(real_episode(ev.HoldStillPolicy, 10015))
    assert s["starts_in_contact"] is True and s["engaged"] is None
    assert s["contact_steps"] == 300


@needs_network
@pytest.mark.parametrize(
    "seed, still_share", [(10022, 0.24), (10034, 0.74), (10000, 0.55)]
)
def test_real_network_timelines_agree_with_the_survey_numbers(seed, still_share):
    model, _ = tb.load_checkpoint(CHECKPOINT)
    ep = real_episode(lambda: bp.BCPolicy(model), seed)
    s = tl.timeline_stats(ep)
    assert np.mean(s["stalled_by_phase"]) == pytest.approx(still_share, abs=6e-3)
    assert np.mean(s["stalled_by_phase"]) == pytest.approx(
        ie.behavior_stats(ep)["stalled_fraction"]
    )
