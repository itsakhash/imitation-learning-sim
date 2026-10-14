import json

import numpy as np
import pytest

import image_gap as ig
import pusht_data


def clean_frame():
    f = np.full((96, 96, 3), 255, dtype=np.uint8)
    f[30:60, 30:60] = (60, 90, 160)  # a solid 30 x 30 object
    return f


def stack(frame, n=3):
    return np.stack([frame] * n)


def blur(frame):
    p = np.pad(frame.astype(np.float32), ((1, 1), (1, 1), (0, 0)), mode="edge")
    out = sum(p[i : i + 96, j : j + 96] for i in range(3) for j in range(3)) / 9
    return np.round(out).astype(np.uint8)


def noisy(frame, seed=0):
    rng = np.random.default_rng(seed)
    f = frame.astype(np.int16) + rng.integers(-3, 4, size=frame.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def test_clean_frame_statistics_have_known_values():
    s = ig.frame_stats(stack(clean_frame()))
    assert s["frames"] == 3
    assert s["small_step_fraction"] == 0.0
    assert s["exact_white_fraction"] == pytest.approx(1 - 900 / 9216)
    assert s["edge_fraction"] == pytest.approx(
        120 / 18240
    )  # 4 x 30 boundary pairs per axis
    assert s["sharp_edge_share"] == 1.0


def test_mean_rgb_of_a_flat_colour():
    flat = np.full((4, 96, 96, 3), (10, 20, 30), dtype=np.uint8)
    s = ig.frame_stats(flat)
    assert s["mean_rgb"] == pytest.approx([10.0, 20.0, 30.0])
    assert s["edge_fraction"] == 0.0 and s["sharp_edge_share"] == 0.0


def test_noise_raises_the_small_step_fraction():
    clean = ig.frame_stats(stack(clean_frame()))
    noisy_stats = ig.frame_stats(stack(noisy(clean_frame())))
    assert noisy_stats["small_step_fraction"] > clean["small_step_fraction"] + 0.3
    assert noisy_stats["exact_white_fraction"] < clean["exact_white_fraction"]


def test_blur_lowers_the_sharp_edge_share_and_spreads_edges():
    clean = ig.frame_stats(stack(clean_frame()))
    soft = ig.frame_stats(stack(blur(clean_frame())))
    assert soft["sharp_edge_share"] < clean["sharp_edge_share"]
    assert soft["edge_fraction"] > clean["edge_fraction"]


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((96, 96, 3), np.uint8),
        np.zeros((2, 96, 96, 3), np.float32),
        np.zeros((2, 96, 96, 4), np.uint8),
        np.zeros((0, 96, 96, 3), np.uint8),
    ],
)
def test_frame_stats_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        ig.frame_stats(bad)


def test_compare_stats_reports_differences():
    a = ig.frame_stats(stack(clean_frame()))
    b = ig.frame_stats(stack(noisy(clean_frame())))
    cmp = ig.compare_stats(b, a)
    assert cmp["small_step_fraction"]["difference"] == pytest.approx(
        b["small_step_fraction"] - a["small_step_fraction"]
    )
    assert cmp["small_step_fraction"]["dataset"] == b["small_step_fraction"]
    assert len(cmp["mean_rgb_difference"]) == 3


def test_collect_env_frames_shape_and_determinism():
    kwargs = dict(seed_start=5, frames_per_episode=2, steps_between=2)
    a = ig.collect_env_frames(2, **kwargs)
    b = ig.collect_env_frames(2, **kwargs)
    assert a.shape == (4, 96, 96, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a[0], a[2])  # different seeds give different scenes


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_episodes=0),
        dict(n_episodes=1, frames_per_episode=0),
        dict(n_episodes=1, steps_between=0),
    ],
)
def test_collect_env_frames_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        ig.collect_env_frames(**kwargs)


def test_load_dataset_frames_rejects_a_bad_step():
    with pytest.raises(ValueError):
        ig.load_dataset_frames(0)


def test_main_writes_a_report(tmp_path, capsys):
    frames = stack(clean_frame(), 4)
    out = tmp_path / "gap.json"
    code = ig.main(
        ["--out", str(out), "--env-episodes", "7"],
        dataset_loader=lambda step: frames,
        env_collector=lambda n, seed_start: frames,
    )
    assert code == 0
    saved = json.loads(out.read_text())
    assert {"settings", "dataset", "environment", "comparison"} <= set(saved)
    assert saved["settings"]["env_episodes"] == 7
    assert saved["dataset"]["frames"] == 4
    assert saved["comparison"]["small_step_fraction"]["difference"] == 0.0
    assert "small_step_fraction" in capsys.readouterr().out


@pytest.mark.skipif(
    not (pusht_data.DEFAULT_ROOT / pusht_data.VIDEO_FILE).exists(),
    reason="dataset not downloaded; run: python pusht_data.py download",
)
def test_real_dataset_frames_have_the_expected_shape():
    frames = ig.load_dataset_frames(25)
    assert frames.shape == (1026, 96, 96, 3) and frames.dtype == np.uint8
