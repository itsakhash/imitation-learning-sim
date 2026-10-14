import json

import gymnasium as gym
import numpy as np
import pytest

import evaluate as ev


class FakeEnv:
    """Scripted environment: coverage after step t comes from trace_for(seed)."""

    def __init__(self, trace_for, limit=300, info_keys=("coverage", "is_success")):
        self.trace_for = trace_for
        self.limit = limit
        self.info_keys = info_keys
        self.action_space = gym.spaces.Box(0, 512, (2,), np.float32)
        self.seed = None
        self.t = 0
        self.closed = False

    def reset(self, seed=None):
        self.seed, self.t = seed, 0
        return {"obs": seed}, {}

    def step(self, action):
        trace = self.trace_for(self.seed)
        coverage = trace[min(self.t, len(trace) - 1)]
        self.t += 1
        success = coverage > 0.95
        info = {"coverage": coverage, "is_success": success}
        info = {k: v for k, v in info.items() if k in self.info_keys}
        return {"obs": self.t}, 0.0, success, self.t >= self.limit, info

    def close(self):
        self.closed = True


def zero_policy(obs):
    return np.zeros(2, dtype=np.float32)


def ep(seed, max_cov, final_cov, success, steps=300):
    return {
        "seed": seed,
        "steps": steps,
        "max_coverage": max_cov,
        "final_coverage": final_cov,
        "success": success,
        "terminated": success,
        "truncated": not success,
        "coverage_trace": [max_cov],
    }


# --------------------------------------------------------------------- seeds


def test_eval_seeds_start_at_10000_and_are_consecutive():
    assert ev.eval_seeds(3) == [10000, 10001, 10002]


@pytest.mark.parametrize("n", [0, -1])
def test_eval_seeds_rejects_non_positive_counts(n):
    with pytest.raises(ValueError):
        ev.eval_seeds(n)


# ------------------------------------------------------------------- episodes


def test_run_episode_tracks_max_and_final_coverage():
    env = FakeEnv(lambda s: [0.1, 0.5, 0.3], limit=3)
    r = ev.run_episode(env, zero_policy, seed=7)
    assert r["steps"] == 3
    assert r["max_coverage"] == pytest.approx(0.5)
    assert r["final_coverage"] == pytest.approx(0.3)
    assert r["success"] is False and r["truncated"] is True and r["terminated"] is False


def test_run_episode_detects_success_and_stops_there():
    env = FakeEnv(lambda s: [0.2, 0.96, 0.4])
    r = ev.run_episode(env, zero_policy, seed=1)
    assert r["steps"] == 2
    assert r["success"] is True and r["terminated"] is True
    assert r["max_coverage"] == pytest.approx(0.96)
    assert r["final_coverage"] == pytest.approx(0.96)


def test_run_episode_near_miss_is_not_a_success():
    env = FakeEnv(lambda s: [0.9499], limit=4)
    r = ev.run_episode(env, zero_policy, seed=1)
    assert r["success"] is False
    assert r["max_coverage"] == pytest.approx(0.9499)


def test_run_episode_respects_max_steps():
    env = FakeEnv(lambda s: [0.1], limit=1000)
    r = ev.run_episode(env, zero_policy, seed=1, max_steps=5)
    assert r["steps"] == 5 and r["truncated"] is False and r["terminated"] is False


def test_run_episode_passes_observations_to_the_policy():
    seen = []
    env = FakeEnv(lambda s: [0.1, 0.2], limit=2)
    ev.run_episode(env, lambda obs: seen.append(obs) or zero_policy(obs), seed=3)
    assert seen == [{"obs": 3}, {"obs": 1}]


def test_policy_reset_receives_the_episode_seed():
    class Recording:
        def __init__(self):
            self.seeds = []

        def reset(self, seed):
            self.seeds.append(seed)

        def __call__(self, obs):
            return zero_policy(obs)

    policy = Recording()
    env = FakeEnv(lambda s: [0.1], limit=2)
    ev.run_episode(env, policy, seed=42)
    assert policy.seeds == [42]


def test_policy_without_a_reset_method_is_fine():
    env = FakeEnv(lambda s: [0.1], limit=2)
    assert ev.run_episode(env, zero_policy, seed=1)["steps"] == 2


@pytest.mark.parametrize("keys", [("coverage",), ("is_success",)])
def test_run_episode_requires_both_info_keys(keys):
    env = FakeEnv(lambda s: [0.1], limit=2, info_keys=keys)
    with pytest.raises(KeyError):
        ev.run_episode(env, zero_policy, seed=1)


@pytest.mark.parametrize("max_steps", [0, -1])
def test_run_episode_rejects_bad_max_steps(max_steps):
    with pytest.raises(ValueError):
        ev.run_episode(
            FakeEnv(lambda s: [0.1]), zero_policy, seed=1, max_steps=max_steps
        )


def test_coverage_trace_is_recorded_rounded_to_four_places():
    env = FakeEnv(lambda s: [0.123456, 0.5], limit=2)
    r = ev.run_episode(env, zero_policy, seed=1)
    assert r["coverage_trace"] == [0.1235, 0.5]


# -------------------------------------------------------------------- Wilson


def test_wilson_zero_successes_of_fifty():
    lo, hi = ev.wilson_ci(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.0713, abs=1e-3)


def test_wilson_five_of_ten():
    lo, hi = ev.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_all_successes():
    lo, hi = ev.wilson_ci(50, 50)
    assert lo == pytest.approx(0.9287, abs=1e-3)
    assert hi == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("k, n", [(1, 10), (3, 20), (7, 50)])
def test_wilson_is_symmetric(k, n):
    lo, hi = ev.wilson_ci(k, n)
    lo2, hi2 = ev.wilson_ci(n - k, n)
    assert lo == pytest.approx(1 - hi2, abs=1e-12)
    assert hi == pytest.approx(1 - lo2, abs=1e-12)


def test_wilson_narrows_with_more_data():
    lo_a, hi_a = ev.wilson_ci(5, 10)
    lo_b, hi_b = ev.wilson_ci(500, 1000)
    assert (hi_b - lo_b) < (hi_a - lo_a)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(successes=1, n=0),
        dict(successes=5, n=4),
        dict(successes=-1, n=4),
        dict(successes=1, n=4, confidence=1.0),
        dict(successes=1, n=4, confidence=0.0),
    ],
)
def test_wilson_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        ev.wilson_ci(**kwargs)


# ------------------------------------------------------------------ bootstrap


def test_bootstrap_of_constant_values_is_that_value():
    lo, hi = ev.bootstrap_ci([0.3] * 20)
    assert lo == pytest.approx(0.3) and hi == pytest.approx(0.3)


def test_bootstrap_is_deterministic_for_a_seed():
    values = np.random.default_rng(1).uniform(0, 1, 30)
    assert ev.bootstrap_ci(values, seed=5) == ev.bootstrap_ci(values, seed=5)
    assert ev.bootstrap_ci(values, seed=5) != ev.bootstrap_ci(values, seed=6)


def test_bootstrap_interval_contains_the_sample_mean():
    values = np.linspace(0, 1, 101)  # mean is exactly 0.5
    lo, hi = ev.bootstrap_ci(values)
    assert lo < 0.5 < hi


def test_bootstrap_narrows_with_more_samples():
    rng = np.random.default_rng(0)
    lo_a, hi_a = ev.bootstrap_ci(rng.normal(0.5, 0.1, 10))
    lo_b, hi_b = ev.bootstrap_ci(rng.normal(0.5, 0.1, 1000))
    assert (hi_b - lo_b) < (hi_a - lo_a)


def test_bootstrap_of_a_single_value():
    lo, hi = ev.bootstrap_ci([0.7])
    assert lo == pytest.approx(0.7) and hi == pytest.approx(0.7)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(values=[]),
        dict(values=[1, 2], confidence=1.0),
        dict(values=[1, 2], confidence=0.0),
        dict(values=[1, 2], n_resamples=10),
        dict(values=[[1, 2], [3, 4]]),
    ],
)
def test_bootstrap_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        ev.bootstrap_ci(**kwargs)


# ------------------------------------------------------------------ summarize


def test_summarize_known_values():
    results = [
        ep(10000, 0.2, 0.1, False, 300),
        ep(10001, 0.4, 0.2, False, 300),
        ep(10002, 0.6, 0.3, True, 100),
    ]
    s = ev.summarize(results)
    assert s["n_episodes"] == 3
    assert (s["seed_start"], s["seed_end"]) == (10000, 10002)
    assert s["mean_max_coverage"] == pytest.approx(0.4)
    assert s["mean_final_coverage"] == pytest.approx(0.2)
    assert s["success_count"] == 1 and s["success_rate"] == pytest.approx(1 / 3)
    assert s["mean_steps"] == pytest.approx(700 / 3)
    lo, hi = s["mean_max_coverage_ci95"]
    assert lo <= s["mean_max_coverage"] <= hi
    lo, hi = s["success_rate_ci95"]
    assert lo <= s["success_rate"] <= hi


def test_summarize_flags_too_few_episodes():
    s = ev.summarize([ep(10000 + i, 0.1, 0.1, False) for i in range(3)])
    assert s["meets_minimum_episodes"] is False


def test_summarize_accepts_the_protocol_minimum():
    s = ev.summarize([ep(10000 + i, 0.1, 0.1, False) for i in range(50)])
    assert s["meets_minimum_episodes"] is True


def test_summarize_rejects_duplicate_seeds():
    with pytest.raises(ValueError):
        ev.summarize([ep(10000, 0.1, 0.1, False), ep(10000, 0.2, 0.2, False)])


def test_summarize_rejects_empty_results():
    with pytest.raises(ValueError):
        ev.summarize([])


# ----------------------------------------------------------- reports and main


def test_format_warns_below_the_minimum():
    s = ev.summarize([ep(10000 + i, 0.1, 0.1, False) for i in range(3)])
    assert "WARNING" in ev.format_summary("x", s)


def test_format_has_no_warning_at_the_minimum():
    s = ev.summarize([ep(10000 + i, 0.1, 0.1, False) for i in range(50)])
    assert "WARNING" not in ev.format_summary("x", s)


def test_write_report_round_trips(tmp_path):
    report = ev.build_report(
        "x", [ep(10000, 0.2, 0.1, False), ep(10001, 0.4, 0.2, True)]
    )
    path = tmp_path / "nested" / "report.json"
    ev.write_report(path, report)
    assert json.loads(path.read_text()) == json.loads(json.dumps(report))


def test_main_writes_a_traceable_report(tmp_path, capsys):
    fake = FakeEnv(lambda s: [0.1, 0.2], limit=3)
    out = tmp_path / "eval.json"
    code = ev.main(
        ["--policy", "random", "--episodes", "3", "--out", str(out)],
        env_factory=lambda: fake,
    )
    assert code == 0 and fake.closed is True
    saved = json.loads(out.read_text())
    assert saved["policy"] == "random"
    assert saved["protocol"]["max_steps"] == 300
    assert saved["protocol"]["eval_seed_start"] == 10000
    assert saved["summary"]["n_episodes"] == 3
    assert [e["seed"] for e in saved["episodes"]] == [10000, 10001, 10002]
    assert len(saved["episodes"][0]["coverage_trace"]) == 3
    assert "WARNING" in capsys.readouterr().out


def test_main_default_output_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeEnv(lambda s: [0.1], limit=2)
    assert ev.main(["--episodes", "2"], env_factory=lambda: fake) == 0
    assert (tmp_path / "reports" / "eval_random.json").exists()


def test_seeded_random_policy_is_reproducible_per_seed():
    policy = ev.SeededRandomPolicy(gym.spaces.Box(0, 512, (2,), np.float32))
    policy.reset(1)
    a = policy(None)
    policy.reset(1)
    b = policy(None)
    policy.reset(2)
    c = policy(None)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# --------------------------------------------------- the real environment


def _real_random_results(n):
    env = ev.make_eval_env()
    try:
        return ev.evaluate_policy(ev.SeededRandomPolicy(env.action_space), env, n)
    finally:
        env.close()


def test_random_policy_on_the_real_environment():
    results = _real_random_results(2)
    assert [r["seed"] for r in results] == [10000, 10001]
    for r in results:
        assert 0.0 <= r["final_coverage"] <= r["max_coverage"] <= 1.0
        assert 1 <= r["steps"] <= 300
        assert isinstance(r["success"], bool)
        assert len(r["coverage_trace"]) == r["steps"]


def test_real_evaluation_is_reproducible():
    assert _real_random_results(2) == _real_random_results(2)
