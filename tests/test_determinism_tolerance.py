import numpy as np
import pytest

import determinism_check as dc
import evaluate as ev


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


def test_a_last_bit_coverage_difference_is_identical_but_reported():
    b = copy_rec(rec())
    b["coverage"][2] += 5.55e-17
    assert b["coverage"][2] != rec()["coverage"][2]
    c = dc.compare_recordings(rec(), b)
    assert c["identical"] is True and c["cause"] is None
    assert c["coverage_steps_differing_at_all"] == 1
    assert 0.0 < c["coverage_max_abs_difference"] < 1e-15
    assert all(v is None for v in c["first_difference"].values())


def test_a_coverage_difference_above_the_tolerance_is_still_a_difference():
    b = copy_rec(rec())
    b["coverage"][2] += 1e-9
    c = dc.compare_recordings(rec(), b)
    assert c["identical"] is False and c["cause"] == "environment"
    assert c["first_difference"]["coverage"] == 2


def test_the_tolerance_can_be_set_to_zero():
    b = copy_rec(rec())
    b["coverage"][2] += 5.55e-17
    assert dc.compare_recordings(rec(), b, coverage_tolerance=0.0)["identical"] is False


def test_a_nan_coverage_counts_as_a_difference():
    b = copy_rec(rec())
    b["coverage"][1] = float("nan")
    assert dc.compare_recordings(rec(), b)["identical"] is False


def test_a_state_difference_is_never_excused_by_the_coverage_tolerance():
    b = copy_rec(rec())
    b["block_pose"][3, 0] += 1e-15
    c = dc.compare_recordings(rec(), b, coverage_tolerance=1.0)
    assert c["identical"] is False and c["cause"] == "environment"


def test_the_description_mentions_last_bit_differences():
    b = copy_rec(rec())
    b["coverage"][2] += 5.55e-17
    text = dc._describe(dc.compare_recordings(rec(), b))
    assert text.startswith("identical over 5 steps") and "last bits" in text


def test_the_description_of_a_clean_run_has_no_remark():
    assert (
        dc._describe(dc.compare_recordings(rec(), copy_rec(rec())))
        == "identical over 5 steps"
    )


def test_the_random_policy_needs_no_environment():
    p1, p2 = dc.build_policy("random", None), dc.build_policy("random", None)
    assert isinstance(p1, ev.SeededRandomPolicy)
    p1.reset(1)
    p2.reset(1)
    assert np.array_equal(p1(None), p2(None))  # fresh, independent action spaces
    assert p1.action_space.shape == (2,) and p1.action_space.dtype == np.float32


def test_the_random_action_space_matches_the_real_environments():
    env = ev.make_eval_env()
    try:
        real = env.action_space
    finally:
        env.close()
    ours = dc.build_policy("random", None).action_space
    assert ours.shape == real.shape and ours.dtype == real.dtype
    assert np.array_equal(ours.low, real.low) and np.array_equal(ours.high, real.high)
