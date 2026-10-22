import json
from pathlib import Path

import numpy as np
import pytest

import ablation_study as ab
import compare_policies as cp
import evaluate as ev
import failure_categories as fk


def hold(obs):
    return np.zeros(2, dtype=np.float32)


def script(trace, start=0.1, agent="moving", block="still", success_at=None):
    """A scripted episode: coverage per step, agent path and block path (length steps + 1)."""
    n = len(trace)
    paths = {
        "moving": [(100.0 + 5 * i, 100.0) for i in range(n + 1)],
        "still": [(100.0, 100.0)] * (n + 1),
    }
    blocks = {
        "still": [(200.0, 200.0, 0.0)] * (n + 1),
        "pushed": [(200.0, 200.0, 0.0)] + [(230.0, 200.0, 0.5)] * n,
    }
    return {
        "trace": list(trace),
        "start": start,
        "success_at": success_at,
        "agent": paths[agent] if isinstance(agent, str) else agent,
        "block": blocks[block] if isinstance(block, str) else block,
    }


class SeededFakeEnv:
    """Plays the script registered for the seed given to reset."""

    def __init__(self, scripts, default):
        self.scripts, self.default = scripts, default
        self.script, self.t = default, 0

    @property
    def unwrapped(self):
        return self

    def _get_coverage(self):
        return self.script["start"]

    def _obs(self):
        return {"agent_pos": np.array(self.script["agent"][self.t], dtype=np.float64)}

    def reset(self, seed=None):
        self.script, self.t = self.scripts.get(seed, self.default), 0
        return self._obs(), {
            "block_pose": np.array(self.script["block"][0], dtype=np.float64)
        }

    def step(self, action):
        s = self.script
        coverage = s["trace"][self.t]
        self.t += 1
        success = s["success_at"] is not None and self.t >= s["success_at"]
        info = {
            "coverage": coverage,
            "is_success": success,
            "block_pose": np.array(s["block"][self.t], dtype=np.float64),
        }
        return self._obs(), 0.0, success, self.t >= len(s["trace"]), info

    def close(self):
        pass


def replay(**kw):
    return fk.replay_trajectory(SeededFakeEnv({}, script(**kw)), hold, 1)


# climbs from 0.1 to 0.39, then goes 0.4, 0.39, ... down to 0.11: the peak is 0.4 and the final value 0.11
RISING_THEN_LOST = [0.1 + 0.01 * i for i in range(30)] + [
    0.4 - 0.01 * i for i in range(30)
]


# ----------------------------------------------------------------------------- replay


def test_replay_records_the_agent_and_the_block_before_and_after_every_step():
    r = replay(trace=[0.1, 0.2, 0.3, 0.4, 0.5], block="pushed")
    assert r["steps"] == 5 and r["coverage_trace"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert r["agent_positions"].shape == (6, 2) and r["block_poses"].shape == (6, 3)
    assert r["agent_positions"][3][0] == 115.0
    assert list(r["block_poses"][0]) == [200.0, 200.0, 0.0] and list(
        r["block_poses"][1]
    ) == [230.0, 200.0, 0.5]
    assert (
        r["start_coverage"] == 0.1 and r["success"] is False and r["truncated"] is True
    )


def test_replay_stops_when_the_episode_is_solved():
    r = replay(trace=[0.1] * 10, start=0.0, success_at=3)
    assert r["steps"] == 3 and r["success"] is True and len(r["agent_positions"]) == 4


def test_replay_stops_at_the_step_limit_and_rejects_a_bad_limit():
    r = fk.replay_trajectory(
        SeededFakeEnv({}, script(trace=[0.1] * 10)), hold, 1, max_steps=4
    )
    assert r["steps"] == 4 and len(r["block_poses"]) == 5
    with pytest.raises(ValueError):
        fk.replay_trajectory(
            SeededFakeEnv({}, script(trace=[0.1] * 10)), hold, 1, max_steps=0
        )


# -------------------------------------------------------------------------- categories


def test_a_block_that_never_moves_is_never_moved():
    c = fk.categorize(replay(trace=[0.1] * 60, block="still"))
    assert (
        c["category"] == "never moved"
        and c["block_moved_px"] == 0.0
        and c["block_moved_rad"] == 0.0
    )


def test_a_moved_block_without_a_gain_is_moved_without_improving():
    assert (
        fk.categorize(replay(trace=[0.1] * 60, block="pushed"))["category"]
        == "moved without improving"
    )


@pytest.mark.parametrize(
    "peak, expected", [(0.119, "moved without improving"), (0.121, "improved and kept")]
)
def test_the_gain_threshold_is_two_hundredths_above_the_start(peak, expected):
    trace = [0.1] * 59 + [peak]
    assert (
        fk.categorize(replay(trace=trace, start=0.1, block="pushed"))["category"]
        == expected
    )


def test_a_gain_that_is_given_back_is_improved_then_lost():
    c = fk.categorize(replay(trace=RISING_THEN_LOST, block="pushed"))
    assert c["category"] == "improved then lost"
    assert c["peak_coverage"] == pytest.approx(0.4) and c[
        "final_coverage"
    ] == pytest.approx(0.11)


def test_a_gain_that_is_kept_is_improved_and_kept():
    trace = [0.1 + 0.01 * i for i in range(31)] + [0.35] * 29
    assert (
        fk.categorize(replay(trace=trace, block="pushed"))["category"]
        == "improved and kept"
    )


@pytest.mark.parametrize(
    "final, expected", [(0.31, "improved and kept"), (0.29, "improved then lost")]
)
def test_the_giveback_threshold_is_one_tenth_below_the_peak(final, expected):
    trace = [0.1 + 0.01 * i for i in range(31)] + [final] * 29  # the peak is 0.4
    assert fk.categorize(replay(trace=trace, block="pushed"))["category"] == expected


def test_a_block_that_only_rotates_has_moved():
    block = [(200.0, 200.0, 0.0)] + [(200.0, 200.0, 0.5)] * 60
    assert (
        fk.categorize(replay(trace=[0.1] * 60, block=block))["category"]
        == "moved without improving"
    )


def test_an_angle_that_wraps_around_is_not_a_movement():
    block = [(200.0, 200.0, 3.14)] + [
        (200.0, 200.0, -3.14)
    ] * 60  # a difference of 0.003 rad
    assert (
        fk.categorize(replay(trace=[0.1] * 60, block=block))["category"]
        == "never moved"
    )


def test_frozen_means_the_agent_barely_moves_in_the_last_fifty_steps():
    assert fk.categorize(replay(trace=[0.1] * 60, agent="still"))["frozen"] is True
    assert fk.categorize(replay(trace=[0.1] * 60, agent="moving"))["frozen"] is False
    early = [(100.0 + 5 * i, 100.0) for i in range(11)] + [
        (150.0, 100.0)
    ] * 50  # moves, then stops
    assert fk.categorize(replay(trace=[0.1] * 60, agent=early))["frozen"] is True
    late = [(100.0, 100.0)] * 41 + [
        (100.0 + 5 * i, 100.0) for i in range(1, 21)
    ]  # stops, then moves
    assert fk.categorize(replay(trace=[0.1] * 60, agent=late))["frozen"] is False


def test_a_short_episode_is_frozen_if_the_agent_never_moved():
    assert fk.categorize(replay(trace=[0.1] * 20, agent="still"))["frozen"] is True


def test_a_replay_without_a_start_coverage_cannot_be_categorized():
    r = replay(trace=[0.1] * 5)
    r["start_coverage"] = None
    with pytest.raises(ValueError):
        fk.categorize(r)


# ------------------------------------------------------------------------- networks

FLAT = script(trace=[0.1] * 60, start=0.1, block="still")
WON = script(trace=[0.1] * 10, start=0.0, success_at=3)
LOST = script(trace=RISING_THEN_LOST, start=0.1, block="pushed")
SCRIPTS = {10000: FLAT, 10001: WON, 10002: LOST}


def synthetic_report(name="fake", tamper=False):
    episodes = [
        ev.run_episode(SeededFakeEnv(SCRIPTS, FLAT), hold, s)
        for s in (10000, 10001, 10002)
    ]
    if tamper:
        episodes[0]["steps"] += 1
    return ev.build_report(name, episodes)


def test_only_failed_episodes_are_replayed_and_categorized():
    net = fk.categorize_network(SeededFakeEnv(SCRIPTS, FLAT), hold, synthetic_report())
    assert (net["failed_in_report"], net["categorized"]) == (2, 2) and net[
        "mismatched"
    ] == []
    assert net["counts"] == {
        "never moved": 1,
        "moved without improving": 0,
        "improved then lost": 1,
        "improved and kept": 0,
    }
    assert net["frozen"] == 0 and [e["seed"] for e in net["episodes"]] == [10000, 10002]
    json.dumps(net)


def test_a_replay_that_does_not_match_the_report_is_excluded_and_counted():
    net = fk.categorize_network(
        SeededFakeEnv(SCRIPTS, FLAT), hold, synthetic_report(tamper=True)
    )
    assert net["categorized"] == 1 and len(net["mismatched"]) == 1
    assert net["mismatched"][0]["seed"] == 10000 and any(
        "steps" in p for p in net["mismatched"][0]["problems"]
    )


def test_a_network_without_failures_has_nothing_categorized():
    report = ev.build_report(
        "fake", [ev.run_episode(SeededFakeEnv(SCRIPTS, FLAT), hold, 10001)]
    )
    net = fk.categorize_network(SeededFakeEnv(SCRIPTS, FLAT), hold, report)
    assert (net["failed_in_report"], net["categorized"]) == (0, 0)
    with pytest.raises(ValueError):
        fk.shares(net)


# --------------------------------------------------------------------------- analysis


def network(counts, frozen=0):
    return {
        "categorized": sum(counts),
        "counts": dict(zip(fk.CATEGORIES, counts)),
        "frozen": frozen,
    }


def test_shares_are_fractions_of_the_categorized_episodes():
    s = fk.shares(network((10, 5, 3, 2), frozen=4))
    assert s["never moved"] == 0.5 and s["frozen"] == 0.2
    assert sum(s[c] for c in fk.CATEGORIES) == pytest.approx(1.0)


def test_a_block_is_compared_network_by_network():
    base = [network((10, 5, 3, 2), frozen=4) for _ in range(10)]
    chunk = [network((4, 6, 8, 2), frozen=10) for _ in range(10)]
    block = fk.analyze_block(base, chunk)
    rows = block["rows"]
    assert rows["never moved"]["baseline"]["mean"] == pytest.approx(0.5)
    assert rows["never moved"]["chunk"]["mean"] == pytest.approx(0.2)
    assert rows["never moved"]["difference"] == pytest.approx(-0.3)
    assert rows["never moved"]["ci95"] == pytest.approx([-0.3, -0.3])
    assert rows["frozen"]["difference"] == pytest.approx(0.3)
    assert (
        block["pooled_counts"]["baseline"]["never moved"] == 100
        and block["pooled_counts"]["chunk"]["frozen"] == 100
    )
    assert (
        block["networks"] == {"baseline": 10, "chunk": 10} and block["warnings"] == []
    )
    assert block["failed_episodes"] == {"baseline": 200, "chunk": 200}


def test_networks_without_categorized_episodes_are_left_out():
    base = [network((10, 5, 3, 2)) for _ in range(10)] + [network((0, 0, 0, 0))]
    block = fk.analyze_block(base, [network((4, 6, 8, 2)) for _ in range(10)])
    assert block["networks"]["baseline"] == 10


def test_fewer_than_ten_networks_is_labelled_a_pilot_and_fewer_than_two_gives_no_intervals():
    pilot = fk.analyze_block([network((10, 5, 3, 2))] * 3, [network((4, 6, 8, 2))] * 3)
    assert any("pilot" in w for w in pilot["warnings"]) and pilot["rows"] is not None
    thin = fk.analyze_block([network((10, 5, 3, 2))], [network((4, 6, 8, 2))] * 3)
    assert thin["rows"] is None and any("two networks" in w for w in thin["warnings"])


@pytest.mark.parametrize(
    "seed, label",
    [(0, "seeds 0-9"), (9, "seeds 0-9"), (10, "seeds 10-19"), (23, "seeds 20-29")],
)
def test_block_labels(seed, label):
    assert fk.block_label(seed) == label


# ------------------------------------------------------------------------ command line


def write_reports(directory, tamper=False):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("bc_cnn", "bc_cnn_s1", "chunk_cnn", "chunk_cnn_s1"):
        report = synthetic_report(name, tamper=tamper and name == "chunk_cnn_s1")
        ev.write_report(directory / f"eval_{name}.json", report)


def run_main(directory, extra=(), calls=None):
    def builder(kind, checkpoint, env_factory, n_exec):
        if calls is not None:
            calls.append((kind, checkpoint, n_exec))
        return hold

    return fk.main(
        [
            "--seeds",
            "0",
            "1",
            "--report-dir",
            str(directory),
            "--checkpoint-dir",
            str(directory),
            "--out",
            str(directory / "out.json"),
            *extra,
        ],
        env_factory=lambda: SeededFakeEnv(SCRIPTS, FLAT),
        policy_builder=builder,
        log=lambda *_: None,
    )


def test_main_categorizes_every_network_writes_the_report_and_prints_the_summary(
    tmp_path, capsys
):
    write_reports(tmp_path)
    calls = []
    assert run_main(tmp_path, calls=calls) == 0
    saved = json.loads((tmp_path / "out.json").read_text())
    assert len(saved["networks"]) == 4 and saved["mismatched_episodes"] == 0
    assert [n["categorized"] for n in saved["networks"]] == [2, 2, 2, 2]
    assert (
        list(saved["blocks"]) == ["seeds 0-9"]
        and saved["blocks"]["seeds 0-9"]["rows"] is not None
    )
    assert (
        calls[0] == ("bc_cnn", tmp_path / "bc_cnn.pt", None)
        and calls[1][0] == "chunk_cnn"
    )
    text = capsys.readouterr().out
    for expected in (
        "failure categories: 4 networks, 8 failed episodes categorized",
        "never moved",
        "frozen at the end (flag)",
        "pooled counts, baseline",
        "WARNING",
        "note: descriptive",
    ):
        assert expected in text


def test_main_exits_with_one_but_still_writes_the_report_when_a_replay_does_not_match(
    tmp_path,
):
    write_reports(tmp_path, tamper=True)
    assert run_main(tmp_path) == 1
    assert json.loads((tmp_path / "out.json").read_text())["mismatched_episodes"] == 1


def test_main_reports_a_missing_evaluation_report_as_an_error(tmp_path, capsys):
    assert run_main(tmp_path, extra=["--seeds", "5"]) == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("seeds", [["1", "1"], ["-1"]])
def test_main_rejects_bad_seeds(tmp_path, capsys, seeds):
    write_reports(tmp_path)
    assert run_main(tmp_path, extra=["--seeds", *seeds]) == 1
    assert "--seeds" in capsys.readouterr().err


# ------------------------------------------------------------ the real saved reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"
REAL_NAMES = [
    ab.network_name(m, ab.FULL_SIZE, s)
    for s in range(20)
    for m in ("baseline", "chunk")
]
needs_reports = pytest.mark.skipif(
    not all((REPORTS / f"eval_{n}.json").exists() for n in REAL_NAMES),
    reason="run the studies first",
)


@needs_reports
def test_the_real_evaluation_reports_hold_1998_failed_episodes():
    failed = sum(
        1
        for n in REAL_NAMES
        for e in cp.load_report(REPORTS / f"eval_{n}.json")["episodes"]
        if not e["success"]
    )
    assert failed == 1998  # 40 networks x 50 episodes, minus the two solved ones
