import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import bc_policy as bp
import chunk_policy as cpol
import determinism_check as dc
import evaluate as ev
import train_bc as tb
import train_chunk as tc


class FakeEnv:
    """Scripted environment: the agent moves by 0.1 times the action, the picture encodes
    the agent's x position, and an optional noise() callable moves the block every step (a
    stand-in for a nondeterministic environment). With share_info the same info dict is reused,
    as gym-pusht does."""

    def __init__(self, noise=None, limit=6, share_info=False):
        self.noise, self.limit, self.share_info = noise, limit, share_info
        self._shared = {}
        self.t = 0
        self.closed = False

    def _obs(self):
        return {
            "pixels": np.full((4, 4, 3), int(self.pos[0]) % 256, dtype=np.uint8),
            "agent_pos": self.pos.copy(),
        }

    def reset(self, seed=None):
        self.t = 0
        self.pos = np.array([100.0, 100.0])
        self.block = np.array([200.0, 200.0, 0.0])
        info = self._shared if self.share_info else {}
        info.clear()
        info["block_pose"] = self.block.copy()
        return self._obs(), info

    def step(self, action):
        self.t += 1
        self.pos = self.pos + 0.1 * np.asarray(action, dtype=np.float64)
        if self.noise is not None:
            self.block = self.block + self.noise()
        info = self._shared if self.share_info else {}
        info.clear()
        info.update(
            {
                "block_pose": self.block.copy(),
                "coverage": 0.001 * float(self.pos[0]) + 0.0001 * float(self.block[0]),
                "is_success": False,
            }
        )
        return self._obs(), 0.0, False, self.t >= self.limit, info

    def close(self):
        self.closed = True


class Const:
    def __call__(self, obs):
        return np.array([10.0, 0.0], dtype=np.float32)


class Jittery:
    """A different action on every call (a stand-in for non-reproducible inference)."""

    def __init__(self, counter):
        self.counter = counter

    def __call__(self, obs):
        return np.array([10.0, 0.0], dtype=np.float32) + np.float32(
            next(self.counter) * 1e-3
        )


def noisy_env_factory():
    counter = itertools.count(1)
    return lambda: FakeEnv(noise=lambda: np.array([next(counter) * 1e-9, 0.0, 0.0]))


def rec(n=5):
    return {
        "seed": 1,
        "steps": n,
        "pixel_hash": [f"h{t}" for t in range(n)],
        "agent_pos": np.arange(2 * n, dtype=float).reshape(n, 2),
        "block_pose": np.arange(3 * n, dtype=float).reshape(n, 3),
        "action": np.ones((n, 2), dtype=np.float32),
        "coverage": np.linspace(0.0, 0.4, n),
    }


def copy_rec(r):
    out = dict(r)
    for key in ("agent_pos", "block_pose", "action", "coverage"):
        out[key] = r[key].copy()
    out["pixel_hash"] = list(r["pixel_hash"])
    return out


# ------------------------------------------------------------------ recording


def test_record_episode_records_the_state_before_each_step():
    r = dc.record_episode(FakeEnv(), Const(), seed=3)
    assert r["seed"] == 3 and r["steps"] == 6
    assert (
        len(r["pixel_hash"]) == 6 and len(set(r["pixel_hash"])) == 6
    )  # the picture changes
    assert r["agent_pos"].shape == (6, 2) and np.allclose(
        r["agent_pos"][:, 0], 100.0 + np.arange(6)
    )
    assert r["block_pose"].shape == (6, 3) and np.allclose(
        r["block_pose"][0], [200.0, 200.0, 0.0]
    )
    assert r["action"].shape == (6, 2) and r["action"].dtype == np.float32
    assert r["coverage"].shape == (6,)
    assert r["coverage"][0] == pytest.approx(
        0.001 * 101.0 + 0.0001 * 200.0
    )  # after the first step


def test_record_episode_reads_the_info_before_the_environment_reuses_it():
    env = FakeEnv(noise=lambda: np.array([0.0, 1.0, 0.0]), share_info=True)
    r = dc.record_episode(env, Const(), seed=1)
    assert np.allclose(r["block_pose"][:, 1], 200.0 + np.arange(6))


def test_record_episode_respects_max_steps_and_rejects_bad_values():
    assert (
        dc.record_episode(FakeEnv(limit=50), Const(), seed=1, max_steps=4)["steps"] == 4
    )
    with pytest.raises(ValueError):
        dc.record_episode(FakeEnv(), Const(), seed=1, max_steps=0)


def test_record_episode_resets_the_policy_with_the_seed_and_works_without_reset():
    class Recording(Const):
        def __init__(self):
            self.seeds = []

        def reset(self, seed=None):
            self.seeds.append(seed)

    policy = Recording()
    dc.record_episode(FakeEnv(), policy, seed=42)
    assert policy.seeds == [42]
    assert dc.record_episode(FakeEnv(), Const(), seed=1)["steps"] == 6


# ------------------------------------------------------------------ comparison


def test_first_difference():
    assert dc._first_difference([1, 2, 3], [1, 2, 3]) is None
    assert dc._first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert dc._first_difference([1, 2, 3], [1, 2]) == 2
    assert dc._first_difference(["a", "b"], ["a", "c"]) == 1
    assert dc._first_difference(np.zeros((3, 2)), np.zeros((3, 2))) is None


def test_identical_recordings():
    c = dc.compare_recordings(rec(), copy_rec(rec()))
    assert c["identical"] is True and c["cause"] is None
    assert all(v is None for v in c["first_difference"].values())


def test_an_action_that_differs_while_the_state_is_identical_is_blamed_on_the_policy():
    b = copy_rec(rec())
    b["action"][2] += 1.0
    c = dc.compare_recordings(rec(), b)
    assert c["identical"] is False and c["cause"] == "policy"
    assert c["first_action_difference"] == 2 and c["first_state_difference"] is None


def test_a_state_that_differs_first_is_blamed_on_the_environment():
    b = copy_rec(rec())
    b["block_pose"][3, 0] += 1.0
    c = dc.compare_recordings(rec(), b)
    assert c["cause"] == "environment" and c["first_state_difference"] == 3


def test_the_earlier_of_state_and_action_decides():
    state_first = copy_rec(rec())
    state_first["agent_pos"][2] += 1.0
    state_first["action"][3] += 1.0
    assert dc.compare_recordings(rec(), state_first)["cause"] == "environment"
    action_first = copy_rec(rec())
    action_first["action"][1] += 1.0
    action_first["agent_pos"][3] += 1.0
    assert dc.compare_recordings(rec(), action_first)["cause"] == "policy"


def test_a_state_and_an_action_that_differ_at_the_same_step_are_blamed_on_the_environment():
    b = copy_rec(rec())
    b["pixel_hash"][2] = "other"
    b["action"][2] += 1.0
    assert dc.compare_recordings(rec(), b)["cause"] == "environment"


def test_a_coverage_that_differs_with_identical_state_and_actions_is_an_environment_difference():
    b = copy_rec(rec())
    b["coverage"][4] += 1e-12
    c = dc.compare_recordings(rec(), b)
    assert c["identical"] is False and c["cause"] == "environment"
    assert c["first_difference"]["coverage"] == 4


def test_recordings_of_different_lengths_differ():
    short = copy_rec(rec(3))
    short["steps"] = 3
    c = dc.compare_recordings(rec(5), short)
    assert c["identical"] is False and c["steps_a"] == 5 and c["steps_b"] == 3


# ---------------------------------------------------------------------- replay


def test_replay_policy_returns_the_actions_in_order_and_resets():
    actions = np.arange(8, dtype=np.float32).reshape(4, 2)
    policy = dc.ReplayPolicy(actions)
    assert [policy(None)[0] for _ in range(3)] == [0.0, 2.0, 4.0]
    policy.reset(seed=1)
    assert policy(None)[0] == 0.0


def test_replay_policy_does_not_run_past_its_actions():
    policy = dc.ReplayPolicy(np.zeros((1, 2)))
    policy(None)
    with pytest.raises(IndexError):
        policy(None)


# ------------------------------------------------------------ output repeatability


def test_a_reproducible_policy_has_no_output_differences():
    r = dc.output_repeatability(
        Const, {"pixels": None, "agent_pos": None}, repeats=5, fresh=2
    )
    assert r["calls"] == 5 and r["fresh_loads"] == 2
    assert (
        r["repeated_calls_max_abs_diff"] == 0.0 and r["fresh_loads_max_abs_diff"] == 0.0
    )


def test_a_policy_whose_output_changes_between_calls_is_detected():
    counter = itertools.count()
    r = dc.output_repeatability(
        lambda: Jittery(counter), {"pixels": None}, repeats=4, fresh=1
    )
    assert r["repeated_calls_max_abs_diff"] > 0.0


def test_networks_that_differ_only_between_loads_are_detected():
    offsets = itertools.count()

    class PerLoad:
        def __init__(self):
            self.offset = float(next(offsets))

        def __call__(self, obs):
            return np.array([self.offset, 0.0], dtype=np.float32)

    r = dc.output_repeatability(PerLoad, {"pixels": None}, repeats=3, fresh=2)
    assert (
        r["repeated_calls_max_abs_diff"] == 0.0 and r["fresh_loads_max_abs_diff"] > 0.0
    )


@pytest.mark.parametrize("repeats, fresh", [(1, 1), (5, 0)])
def test_output_repeatability_rejects_bad_arguments(repeats, fresh):
    with pytest.raises(ValueError):
        dc.output_repeatability(Const, {}, repeats=repeats, fresh=fresh)


def test_first_observation_has_the_picture_and_the_position_and_closes_the_environment():
    env = FakeEnv()
    obs = dc.first_observation(lambda: env, seed=1)
    assert set(obs) == {"pixels", "agent_pos"} and env.closed is True


# -------------------------------------------------------------------- check_seed


def test_a_deterministic_environment_and_policy_give_identical_runs_and_replay():
    r = dc.check_seed(FakeEnv, Const, seed=10000)
    assert r["seed"] == 10000
    assert r["run_a_vs_run_b"]["identical"] is True
    assert r["replay_of_a_actions_vs_run_a"]["identical"] is True


def test_a_nondeterministic_environment_is_blamed_on_the_environment_and_the_replay_differs_too():
    r = dc.check_seed(noisy_env_factory(), Const, seed=1)
    assert r["run_a_vs_run_b"]["cause"] == "environment"
    assert r["replay_of_a_actions_vs_run_a"]["cause"] == "environment"


def test_a_nondeterministic_policy_is_blamed_on_the_policy_and_the_replay_is_identical():
    counter = itertools.count()
    r = dc.check_seed(FakeEnv, lambda: Jittery(counter), seed=1)
    assert r["run_a_vs_run_b"]["cause"] == "policy"
    assert r["replay_of_a_actions_vs_run_a"]["identical"] is True


def test_check_seed_closes_every_environment_it_opens():
    made = []

    def factory():
        made.append(FakeEnv())
        return made[-1]

    dc.check_seed(factory, Const, seed=1)
    assert len(made) == 3 and all(e.closed for e in made)


# ------------------------------------------------------------------ build_policy


def save_bc_checkpoint(path):
    torch.manual_seed(0)
    tb.save_checkpoint(
        path,
        tb.BCPolicyNet(),
        config={"epochs": 1},
        split={"train_episode_ids": [0], "val_episode_ids": [1]},
        history={"epoch": [1], "val_mae_px": [1.0]},
        references={},
    )


def save_chunk_checkpoint(path, n_exec=2):
    torch.manual_seed(0)
    tb.save_checkpoint(
        path,
        tc.ChunkPolicyNet(4),
        config={"chunk_len": 4, "n_exec": n_exec},
        split={"train_episode_ids": [0], "val_episode_ids": [1]},
        history={"epoch": [1], "val_mae_px": [1.0]},
        references={},
    )


def test_build_policy_known_kinds(tmp_path):
    assert isinstance(dc.build_policy("hold", None), ev.HoldStillPolicy)
    assert isinstance(
        dc.build_policy("random", None, env_factory=FakeEnv), ev.SeededRandomPolicy
    )
    save_bc_checkpoint(tmp_path / "bc.pt")
    assert isinstance(dc.build_policy("bc_cnn", tmp_path / "bc.pt"), bp.BCPolicy)
    save_chunk_checkpoint(tmp_path / "ch.pt")
    chunk = dc.build_policy("chunk_cnn", tmp_path / "ch.pt")
    assert isinstance(chunk, cpol.ChunkPolicy) and chunk.n_exec == 2
    assert dc.build_policy("chunk_cnn", tmp_path / "ch.pt", n_exec=3).n_exec == 3


def test_build_policy_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        dc.build_policy("nope", None)


# ----------------------------------------------------------------- command line


def run_main(tmp_path, *extra, env_factory=FakeEnv, builder=None):
    builder = builder or (lambda kind, checkpoint, env_factory: Const())
    out = tmp_path / "out.json"
    code = dc.main(
        [
            "--policy",
            "hold",
            "--seed",
            "1",
            "2",
            "--max-steps",
            "4",
            "--repeats",
            "3",
            "--fresh",
            "2",
            "--out",
            str(out),
            *extra,
        ],
        env_factory=env_factory,
        policy_builder=builder,
    )
    return code, out


def test_main_reports_a_reproducible_policy_and_writes_json(tmp_path, capsys):
    code, out = run_main(tmp_path)
    assert code == 0
    saved = json.loads(out.read_text())
    assert saved["policy"] == "hold" and len(saved["seeds"]) == 2
    assert saved["seeds"][0]["run_a_vs_run_b"]["identical"] is True
    assert saved["repeatability"]["seed"] == 1 and "torch_threads" in saved["settings"]
    printed = capsys.readouterr().out
    assert (
        "determinism check for hold" in printed and "identical over 4 steps" in printed
    )


def test_main_exits_with_one_when_the_runs_differ(tmp_path, capsys):
    code, out = run_main(tmp_path, env_factory=noisy_env_factory())
    assert code == 1
    assert (
        json.loads(out.read_text())["seeds"][0]["run_a_vs_run_b"]["identical"] is False
    )
    assert "DIFFERENT" in capsys.readouterr().out


def test_main_applies_and_restores_the_thread_setting(tmp_path):
    before = torch.get_num_threads()
    seen = []

    def builder(kind, checkpoint, env_factory):
        seen.append(torch.get_num_threads())
        return Const()

    run_main(tmp_path, "--threads", "1", builder=builder)
    assert seen and set(seen) == {1} and torch.get_num_threads() == before


def test_main_applies_and_restores_deterministic_algorithms(tmp_path):
    seen = []

    def builder(kind, checkpoint, env_factory):
        seen.append(torch.are_deterministic_algorithms_enabled())
        return Const()

    run_main(tmp_path, "--deterministic", builder=builder)
    assert seen and all(seen) and torch.are_deterministic_algorithms_enabled() is False


def test_main_rejects_a_thread_count_below_one(tmp_path):
    with pytest.raises(ValueError):
        run_main(tmp_path, "--threads", "0")


def test_the_default_report_name_records_the_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dc.main(
        [
            "--policy",
            "hold",
            "--seed",
            "1",
            "--max-steps",
            "3",
            "--repeats",
            "2",
            "--fresh",
            "1",
            "--threads",
            "1",
            "--deterministic",
        ],
        env_factory=FakeEnv,
        policy_builder=lambda kind, checkpoint, env_factory: Const(),
    )
    assert (
        tmp_path / "reports" / "determinism_hold_threads1_deterministic.json"
    ).exists()


# ----------------------------------------------- the real environment and networks

ROOT = Path(__file__).resolve().parents[1]
BC_CHECKPOINT = ROOT / "checkpoints" / "bc_cnn.pt"
needs_bc = pytest.mark.skipif(
    not BC_CHECKPOINT.exists(), reason="train the baseline first"
)


def real_random_policy_factory():
    env = ev.make_eval_env()
    space = env.action_space
    env.close()
    return lambda: ev.SeededRandomPolicy(space)


def test_the_real_environment_is_deterministic_given_the_same_actions():
    r = dc.check_seed(
        ev.make_eval_env, real_random_policy_factory(), 10000, max_steps=40
    )
    assert r["run_a_vs_run_b"]["identical"] is True, r["run_a_vs_run_b"]
    assert r["replay_of_a_actions_vs_run_a"]["identical"] is True, r[
        "replay_of_a_actions_vs_run_a"
    ]


@needs_bc
def test_the_baseline_network_is_reproducible_over_a_short_run():
    r = dc.check_seed(
        ev.make_eval_env,
        lambda: dc.build_policy("bc_cnn", BC_CHECKPOINT),
        10000,
        max_steps=60,
    )
    assert r["run_a_vs_run_b"]["identical"] is True, r["run_a_vs_run_b"]
