import gymnasium as gym
import numpy as np
import pytest

import random_rollout as rr


@pytest.fixture
def env():
    e = rr.make_env()
    yield e
    e.close()


@pytest.fixture
def render_env():
    e = rr.make_env(render=True)
    yield e
    e.close()


def test_spaces_have_expected_shapes(env):
    assert env.observation_space.shape == (5,)
    assert env.action_space.shape == (2,)


def test_random_policy_actions_are_valid(env):
    policy = rr.RandomPolicy(env.action_space, seed=0)
    obs, _ = env.reset(seed=0)
    action = policy(obs)
    assert action.shape == (2,)
    assert env.action_space.contains(action)


def test_different_seeds_give_different_initial_states(env):
    obs0, _ = env.reset(seed=0)
    obs1, _ = env.reset(seed=1)
    assert not np.allclose(obs0, obs1)


def test_same_seed_gives_identical_episode(env):
    a = rr.run_episode(env, rr.RandomPolicy(env.action_space, 7), seed=7, max_steps=50)
    b = rr.run_episode(env, rr.RandomPolicy(env.action_space, 7), seed=7, max_steps=50)
    assert a == b


def test_max_steps_caps_the_episode(env):
    result = rr.run_episode(
        env, rr.RandomPolicy(env.action_space, 0), seed=0, max_steps=5
    )
    assert result["steps"] == 5


def test_episode_ends_on_its_own_before_the_safety_cap(env):
    result = rr.run_episode(env, rr.RandomPolicy(env.action_space, 0), seed=0)
    assert result["terminated"] or result["truncated"]
    assert result["steps"] < 1000


def test_render_returns_a_real_image(render_env):
    render_env.reset(seed=0)
    frame = render_env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert frame.std() > 0


@pytest.mark.parametrize(
    "terminated, info, expected",
    [
        (True, {"is_success": True}, True),
        (False, {"is_success": False}, False),
        (True, {}, True),
        (False, {}, False),
    ],
)
def test_episode_success_logic(terminated, info, expected):
    assert rr.episode_success(terminated, info) is expected


def test_summarize_computes_counts_and_means():
    results = [
        {"success": True, "steps": 10, "total_reward": 1.0},
        {"success": False, "steps": 20, "total_reward": 3.0},
    ]
    summary = rr.summarize(results)
    assert summary == {
        "episodes": 2,
        "successes": 1,
        "mean_steps": 15.0,
        "mean_reward": 2.0,
    }


def test_summarize_rejects_empty_input():
    with pytest.raises(ValueError):
        rr.summarize([])


def test_main_runs_and_reports(capsys):
    assert rr.main(["--episodes", "2"]) == 0
    out = capsys.readouterr().out
    assert "success: " in out
    assert "episode 1:" in out


def test_main_saves_a_frame(tmp_path):
    target = tmp_path / "frame.png"
    assert rr.main(["--episodes", "1", "--save-frame", str(target)]) == 0
    assert target.exists() and target.stat().st_size > 0
