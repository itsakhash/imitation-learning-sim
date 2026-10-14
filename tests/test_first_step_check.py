import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import bc_policy as bp
import evaluate as ev
import first_step_check as fs
import pusht_data
import train_bc as tb


def structured_arrays(n_episodes=12, length=16, seed=0):
    rng = np.random.default_rng(seed)
    n = n_episodes * length
    states = rng.uniform(0, 512, size=(n, 2)).astype(np.float32)
    return pusht_data.PushTArrays(
        images=np.full((n, 96, 96, 3), 255, dtype=np.uint8),
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
    """Answers scale * agent_pos + offset, ignoring the picture (units scaled to [0, 1])."""

    def __init__(self, scale, offset):
        super().__init__()
        self.scale, self.offset = scale, offset

    def forward(self, image, agent_pos):
        return self.scale * agent_pos + self.offset


class CornerStub(nn.Module):
    """Reacts to the picture: adds 0.3 times the brightness of the top-left pixel."""

    def forward(self, image, agent_pos):
        return 0.5 * agent_pos + 0.3 * image[:, 0, 0, 0].unsqueeze(1)


def white_frames(n=20):
    return np.full((n, 96, 96, 3), 255, dtype=np.uint8)


def positions(n=20, seed=0):
    return np.random.default_rng(seed).uniform(50, 450, size=(n, 2))


class FakeResetEnv:
    def __init__(self):
        self.closed = False

    def reset(self, seed=None):
        frame = np.full((96, 96, 3), seed % 256, dtype=np.uint8)
        return {"pixels": frame, "agent_pos": np.array([seed % 500, 100.0])}, {}

    def close(self):
        self.closed = True


# ---------------------------------------------------------------- degradations


def test_box_blur_keeps_a_flat_frame_unchanged():
    flat = np.full((2, 20, 20, 3), 77, dtype=np.uint8)
    assert np.array_equal(fs.box_blur(flat), flat)


def test_box_blur_spreads_a_single_bright_pixel_over_nine_pixels():
    frames = np.zeros((1, 20, 20, 3), dtype=np.uint8)
    frames[0, 10, 10] = 90
    out = fs.box_blur(frames)
    assert out.dtype == np.uint8 and out.shape == frames.shape
    assert (out[0, 9:12, 9:12] == 10).all()
    assert out[0, 8, 8].sum() == 0 and out[0, 12, 12].sum() == 0


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((20, 20, 3), np.uint8),
        np.zeros((1, 20, 20, 3), np.float32),
        np.zeros((1, 20, 20, 4), np.uint8),
        np.zeros((0, 20, 20, 3), np.uint8),
    ],
)
def test_degradations_reject_bad_frames(bad):
    with pytest.raises(ValueError):
        fs.box_blur(bad)
    with pytest.raises(ValueError):
        fs.add_noise(bad)


def test_add_noise_stays_within_level_and_range():
    frames = np.full((3, 30, 30, 3), 128, dtype=np.uint8)
    out = fs.add_noise(frames, level=3)
    delta = out.astype(int) - 128
    assert delta.min() >= -3 and delta.max() <= 3 and np.abs(delta).max() > 0


def test_add_noise_is_deterministic_per_seed():
    frames = white_frames(2)
    assert np.array_equal(fs.add_noise(frames, seed=1), fs.add_noise(frames, seed=1))
    assert not np.array_equal(
        fs.add_noise(frames, seed=1), fs.add_noise(frames, seed=2)
    )


def test_add_noise_with_level_zero_changes_nothing():
    frames = white_frames(2)
    assert np.array_equal(fs.add_noise(frames, level=0), frames)


def test_add_noise_clips_at_both_ends():
    assert fs.add_noise(np.full((1, 20, 20, 3), 255, np.uint8), level=5).max() == 255
    assert fs.add_noise(np.zeros((1, 20, 20, 3), np.uint8), level=5).min() == 0


def test_add_noise_rejects_a_negative_level():
    with pytest.raises(ValueError):
        fs.add_noise(white_frames(1), level=-1)


def test_degrade_known_kinds_and_composition():
    frames = np.random.default_rng(0).integers(
        0, 256, size=(2, 24, 24, 3), dtype=np.uint8
    )
    assert np.array_equal(fs.degrade(frames, "blur"), fs.box_blur(frames))
    assert np.array_equal(fs.degrade(frames, "noise", 3, 5), fs.add_noise(frames, 3, 5))
    assert np.array_equal(
        fs.degrade(frames, "blur+noise", 3, 5), fs.add_noise(fs.box_blur(frames), 3, 5)
    )


def test_degrade_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        fs.degrade(white_frames(1), "sharpen")


# -------------------------------------------------------- commanded displacement


def test_commanded_displacement_known_values():
    pos = positions()
    d = fs.commanded_displacement(AffineStub(0.5, 0.25), white_frames(), pos)
    assert d.shape == (20, 2)
    assert np.allclose(d, (0.5 * pos / 512 + 0.25) * 512 - pos, atol=0.01)


def test_commanded_displacement_clips_the_target_to_the_action_range():
    pos = positions()
    d = fs.commanded_displacement(AffineStub(0.0, 2.0), white_frames(), pos)
    assert np.allclose(d, 512.0 - pos, atol=1e-6)
    d = fs.commanded_displacement(AffineStub(0.0, -2.0), white_frames(), pos)
    assert np.allclose(d, -pos, atol=1e-6)


def test_commanded_displacement_does_not_depend_on_the_batch_size():
    frames = np.random.default_rng(0).integers(
        0, 256, size=(20, 96, 96, 3), dtype=np.uint8
    )
    torch.manual_seed(0)
    model = tb.BCPolicyNet()
    a = fs.commanded_displacement(model, frames, positions(), batch_size=1)
    b = fs.commanded_displacement(model, frames, positions(), batch_size=7)
    assert np.allclose(a, b, atol=1e-3)


@pytest.mark.parametrize("pos_shape, batch", [((19, 2), 4), ((20, 3), 4), ((20, 2), 0)])
def test_commanded_displacement_rejects_bad_arguments(pos_shape, batch):
    with pytest.raises(ValueError):
        fs.commanded_displacement(
            AffineStub(0.5, 0.25), white_frames(), np.zeros(pos_shape), batch_size=batch
        )


def test_gap_summary_known_values():
    s = fs.gap_summary([[3, 4], [0, 0], [6, 8], [1, 1]])
    assert s["n"] == 4
    assert s["mean_gap_px"] == pytest.approx((5 + 0 + 10 + 2**0.5) / 4)
    assert s["median_gap_px"] == pytest.approx((2**0.5 + 5) / 2)
    assert s["near_copy_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [[], [[1, 2, 3]], [1, 2]])
def test_gap_summary_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        fs.gap_summary(bad)


# ----------------------------------------------------------- collecting inputs


def test_collect_reset_frames_with_a_fake_environment():
    frames, pos = fs.collect_reset_frames(FakeResetEnv(), [10, 20, 30])
    assert frames.shape == (3, 96, 96, 3) and frames.dtype == np.uint8
    assert frames[1, 0, 0, 0] == 20 and np.allclose(pos[:, 0], [10, 20, 30])


def test_collect_reset_frames_needs_seeds():
    with pytest.raises(ValueError):
        fs.collect_reset_frames(FakeResetEnv(), [])


def test_dataset_first_frames_picks_frame_zero_of_the_requested_episodes():
    arrays = structured_arrays()
    images, pos, actions = fs.dataset_first_frames(arrays, [2, 5])
    assert len(images) == len(pos) == len(actions) == 2
    rows = np.flatnonzero(
        np.isin(arrays.episode_index, [2, 5]) & (arrays.frame_index == 0)
    )
    assert np.allclose(pos, arrays.states[rows])


def test_dataset_first_frames_rejects_unknown_episodes():
    with pytest.raises(ValueError):
        fs.dataset_first_frames(structured_arrays(), [999])


# ------------------------------------------------------------- degradation effect


def test_a_network_that_ignores_the_picture_is_not_moved_by_degradation():
    for kind in fs.DEGRADATIONS:
        r = fs.degradation_effect(
            AffineStub(0.5, 0.25), white_frames(), positions(), kind
        )
        assert r["mean_change_px"] == pytest.approx(0.0, abs=1e-9)


def test_a_network_that_reads_the_picture_is_moved_by_noise():
    r = fs.degradation_effect(
        CornerStub(), white_frames(), positions(), "noise", level=3
    )
    assert 0.0 < r["mean_change_px"] < 3.0
    assert r["change_share_of_gap"] is not None and r["change_share_of_gap"] > 0.0


def test_noise_raises_the_small_step_fraction_of_a_flat_frame():
    r = fs.degradation_effect(
        AffineStub(0.5, 0.25), white_frames(), positions(), "noise", level=3
    )
    assert r["small_step_fraction"] > 0.4
    assert r["exact_white_fraction"] < 0.9


def test_degradation_effect_reports_all_image_statistics():
    r = fs.degradation_effect(
        AffineStub(0.5, 0.25), white_frames(), positions(), "blur"
    )
    for key in fs.IMAGE_KEYS + (
        "mean_change_px",
        "median_change_px",
        "mean_gap_px",
        "frames",
    ):
        assert key in r


# ------------------------------------------------------------ analysis and output


def run_analyze(model=None):
    arrays = structured_arrays()
    ds_frames, ds_pos, ds_actions = fs.dataset_first_frames(arrays, [0, 1, 2])
    return fs.analyze(
        model or AffineStub(0.5, 0.25),
        white_frames(5),
        positions(5),
        ds_frames,
        ds_pos,
        ds_actions,
        arrays.images,
    )


def test_analyze_counts_and_ratio():
    r = run_analyze()
    assert r["network_on_environment"]["n"] == 5 and r["network_on_dataset"]["n"] == 3
    assert r["human_on_dataset"]["n"] == 3
    assert [d["kind"] for d in r["degradations"]] == list(fs.DEGRADATIONS)
    assert r["gap_ratio_environment_over_dataset"] == pytest.approx(
        r["network_on_environment"]["mean_gap_px"]
        / r["network_on_dataset"]["mean_gap_px"]
    )
    json.dumps(r)  # must be JSON serializable


def test_a_perfect_network_has_the_same_gap_as_the_human_on_the_dataset_frames():
    r = run_analyze(AffineStub(0.5, 0.25))
    assert r["network_on_dataset"]["mean_gap_px"] == pytest.approx(
        r["human_on_dataset"]["mean_gap_px"], abs=0.02
    )


def test_format_report_has_both_tables_and_the_note():
    text = fs.format_report(run_analyze())
    for expected in (
        "network, environment reset frames",
        "human, dataset frame 0",
        "environment frames, blur+noise",
        "exact white",
        "note:",
    ):
        assert expected in text


def test_main_end_to_end_with_a_fake_environment(tmp_path, capsys):
    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    tb.save_checkpoint(
        ckpt,
        tb.BCPolicyNet(),
        config={"epochs": 1},
        split={"train_episode_ids": list(range(3, 12)), "val_episode_ids": [0, 1, 2]},
        history={"epoch": [1], "val_mae_px": [1.0]},
        references={},
    )
    env = FakeResetEnv()
    out = tmp_path / "o.json"
    code = fs.main(
        ["--checkpoint", str(ckpt), "--episodes", "4", "--out", str(out)],
        env_factory=lambda: env,
        arrays_loader=structured_arrays,
    )
    assert code == 0 and env.closed is True
    saved = json.loads(out.read_text())
    assert (
        saved["network_on_environment"]["n"] == 4
        and saved["network_on_dataset"]["n"] == 3
    )
    assert saved["checkpoint"] == str(ckpt)
    assert "first step of an episode" in capsys.readouterr().out


# ----------------------------------------------- the real environment and network

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "bc_cnn.pt"
needs_network = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="train the BC network first: python train_bc.py"
)


def real_reset(seeds):
    env = ev.make_eval_env()
    try:
        return fs.collect_reset_frames(env, seeds)
    finally:
        env.close()


def blue_centroid(img):
    r, g, b = (img[..., i].astype(int) for i in range(3))
    mask = (b > 180) & (r < 130) & (g < 160) & (b - r > 80)
    if mask.sum() < 3:
        return None
    rows, cols = np.nonzero(mask)
    return np.array([cols.mean() + 0.5, rows.mean() + 0.5])


def test_real_reset_frames_have_the_known_agent_position_for_seed_10000():
    frames, pos = real_reset([10000, 10001])
    assert frames.shape == (2, 96, 96, 3) and frames.dtype == np.uint8
    assert np.allclose(pos[0], [116.0, 256.0])


def test_real_environment_agent_pixels_line_up_with_agent_pos():
    # The coordinate check that found 0.06 px for "as is" and 28 to 43 px for every other
    # convention, kept as a permanent test.
    frames, pos = real_reset(ev.eval_seeds(20))
    expected = pos / pusht_data.COORD_SCALE * pusht_data.IMAGE_SIZE
    fits = {
        "as is": lambda q: q,
        "y flipped": lambda q: np.array([q[0], 96 - q[1]]),
        "x flipped": lambda q: np.array([96 - q[0], q[1]]),
        "x and y swapped": lambda q: q[::-1],
    }
    centroids = [blue_centroid(f) for f in frames]
    used = [(c, p) for c, p in zip(centroids, expected) if c is not None]
    assert len(used) >= 15
    errors = {
        k: float(np.median([np.linalg.norm(c - f(p)) for c, p in used]))
        for k, f in fits.items()
    }
    assert errors["as is"] < 0.5, errors
    assert all(v > 5.0 for k, v in errors.items() if k != "as is"), errors


@needs_network
def test_real_first_step_displacement_matches_the_policy_used_in_evaluation():
    model, _ = tb.load_checkpoint(CHECKPOINT)
    env = ev.make_eval_env()
    try:
        obs, _ = env.reset(seed=10000)
    finally:
        env.close()
    expected = bp.BCPolicy(model)(obs) - np.asarray(obs["agent_pos"])
    got = fs.commanded_displacement(
        model, np.asarray(obs["pixels"])[None], np.asarray(obs["agent_pos"])[None]
    )[0]
    assert np.allclose(got, expected, atol=0.05)


@needs_network
def test_real_first_step_results_are_reproducible():
    model, _ = tb.load_checkpoint(CHECKPOINT)
    frames, pos = real_reset(ev.eval_seeds(5))
    a = fs.commanded_displacement(model, frames, pos)
    b = fs.commanded_displacement(model, frames, pos)
    assert a.shape == (5, 2) and np.isfinite(a).all() and np.array_equal(a, b)
