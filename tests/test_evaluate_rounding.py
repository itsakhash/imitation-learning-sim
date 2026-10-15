import numpy as np
import pytest

import evaluate as ev


class ScriptedEnv:
    """Reports the given coverage after each step. The success flag follows the environment's
    own rule: coverage above 0.95."""

    def __init__(self, trace, start=None):
        self.trace, self.start, self.t = list(trace), start, 0

    def reset(self, seed=None):
        self.t = 0
        return {"agent_pos": np.zeros(2)}, (
            {} if self.start is None else {"coverage": self.start}
        )

    def step(self, action):
        coverage = self.trace[self.t]
        self.t += 1
        success = coverage > 0.95
        info = {"coverage": coverage, "is_success": success}
        return {"agent_pos": np.zeros(2)}, 0.0, success, self.t >= len(self.trace), info


def hold(obs):
    return np.zeros(2, dtype=np.float32)


def test_coverage_values_are_rounded_to_nine_decimals():
    r = ev.run_episode(
        ScriptedEnv([0.3, 0.1234567891234], start=0.9876543214999), hold, seed=1
    )
    assert r["max_coverage"] == 0.3
    assert r["final_coverage"] == 0.123456789
    assert r["start_coverage"] == 0.987654321


def test_start_coverage_stays_none_when_the_environment_does_not_report_it():
    assert (
        ev.run_episode(ScriptedEnv([0.1, 0.2]), hold, seed=1)["start_coverage"] is None
    )


def test_a_last_bit_difference_does_not_change_the_result():
    assert 0.4 + 5.55e-17 != 0.4 and 0.2 + 2.8e-17 != 0.2
    a = ev.run_episode(ScriptedEnv([0.3, 0.4], start=0.2), hold, seed=1)
    b = ev.run_episode(
        ScriptedEnv([0.3, 0.4 + 5.55e-17], start=0.2 + 2.8e-17), hold, seed=1
    )
    assert a == b


def test_success_comes_from_the_environment_flag_not_from_the_rounded_value():
    r = ev.run_episode(ScriptedEnv([0.2, 0.9500000001]), hold, seed=1)
    assert r["success"] is True and r["terminated"] is True
    assert r["max_coverage"] == 0.95  # rounded to nine decimals, yet still a success


def test_rounding_moves_a_value_by_at_most_half_a_unit_of_the_ninth_decimal():
    for v in np.random.default_rng(0).uniform(0, 1, 200):
        assert abs(ev.round_coverage(float(v)) - v) <= 5e-10 + 1e-15


def test_round_coverage_keeps_none_and_accepts_numpy_floats():
    assert ev.round_coverage(None) is None
    assert ev.round_coverage(np.float64(0.12345678912)) == 0.123456789


def test_summaries_are_unaffected_at_the_reported_precision():
    r = ev.run_episode(ScriptedEnv([0.1234567891234]), hold, seed=1)
    assert round(ev.summarize([r])["mean_max_coverage"], 4) == 0.1235
