import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import offline_progress as op
import pusht_data
import train_bc as tb


def structured_arrays(n_episodes=12, length=16, seed=0):
    """Fake data where the action is 0.5 * position + 128 pixels."""
    rng = np.random.default_rng(seed)
    n = n_episodes * length
    states = rng.uniform(0, 512, size=(n, 2)).astype(np.float32)
    return pusht_data.PushTArrays(
        images=np.zeros((n, 96, 96, 3), dtype=np.uint8),
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
    """Answers scale * agent_pos + offset, ignoring the image (units scaled to [0, 1]).
    AffineStub(0.5, 0.25) reproduces the actions of structured_arrays exactly."""

    def __init__(self, scale, offset):
        super().__init__()
        self.scale, self.offset = scale, offset

    def forward(self, image, agent_pos):
        return self.scale * agent_pos + self.offset


# ------------------------------------------------------------------- binning


def test_progress_bins_known_values():
    assert op.progress_bins([0, 5, 9], [10, 10, 10], 10).tolist() == [0, 5, 9]
    assert op.progress_bins([0, 1, 2], [4, 4, 4], 2).tolist() == [0, 0, 1]


# changed: the earlier version used an episode of 7 frames, which cannot reach the last bin
def test_progress_bins_puts_the_last_frame_in_the_last_bin():
    lengths = np.array([49, 122, 246, 10])
    assert op.progress_bins(lengths - 1, lengths, 10).tolist() == [9, 9, 9, 9]


# new
def test_the_last_frame_is_in_the_last_bin_for_every_length_of_at_least_n_bins():
    for length in range(10, 301):
        assert op.progress_bins([length - 1], [length], 10).tolist() == [9], length


# new
def test_an_episode_shorter_than_the_number_of_bins_cannot_reach_the_last_bin():
    # Integer binning: 6 * 10 // 7 = 8. Real demonstrations are at least 49 frames long.
    assert op.progress_bins([6], [7], 10).tolist() == [8]


# new
def test_the_first_frame_is_always_in_the_first_bin():
    assert op.progress_bins([0, 0, 0], [7, 49, 246], 10).tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    "frames, lengths, n_bins",
    [([0], [10], 0), ([10], [10], 10), ([-1], [10], 10), ([0], [0], 10)],
)
def test_progress_bins_rejects_bad_arguments(frames, lengths, n_bins):
    with pytest.raises(ValueError):
        op.progress_bins(frames, lengths, n_bins)


def test_step_bins_known_values():
    frames = [0, 74, 75, 149, 150, 224, 225, 299, 300, 1000, -1]
    assert op.step_bins(frames).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, -1, -1, -1]


@pytest.mark.parametrize("edges", [(0,), (0, 75, 75), (0, 100, 50)])
def test_step_bins_rejects_bad_edges(edges):
    with pytest.raises(ValueError):
        op.step_bins([0, 1], edges)


# -------------------------------------------------------------- binned stats


def test_binned_stats_known_values():
    dh = [[3, 4], [0, 0], [10, 0], [0, 10]]
    dp = [[3, 4], [0, 0], [5, 0], [0, -10]]
    move = [np.nan, 5.0, 0.5, np.nan]
    a, b = op.binned_stats([0, 0, 1, 1], 2, dh, dp, move)
    assert a["frames"] == 2
    assert a["human_gap_px"] == pytest.approx(2.5) and a[
        "network_gap_px"
    ] == pytest.approx(2.5)
    assert a["displacement_gain"] == pytest.approx(1.0)
    assert a["mean_cosine"] == pytest.approx(1.0) and a["cosine_frames"] == 1
    assert a["human_near_copy_fraction"] == pytest.approx(0.5)
    assert a["network_near_copy_fraction"] == pytest.approx(0.5)
    assert a["human_move_px"] == pytest.approx(5.0) and a["human_still_fraction"] == 0.0
    assert b["human_gap_px"] == pytest.approx(10.0) and b[
        "network_gap_px"
    ] == pytest.approx(7.5)
    assert b["displacement_gain"] == pytest.approx(-0.25)
    assert b["mean_cosine"] == pytest.approx(0.0) and b["cosine_frames"] == 2
    assert (
        b["human_near_copy_fraction"] == 0.0 and b["network_near_copy_fraction"] == 0.0
    )
    assert b["human_move_px"] == pytest.approx(0.5) and b["human_still_fraction"] == 1.0


def test_binned_stats_empty_bin_has_no_numbers():
    rows = op.binned_stats([0, 0], 3, [[3, 4], [1, 1]], [[3, 4], [1, 1]], [1.0, 1.0])
    assert [r["frames"] for r in rows] == [2, 0, 0]
    for key in (
        "human_gap_px",
        "network_gap_px",
        "displacement_gain",
        "mean_cosine",
        "human_move_px",
        "human_near_copy_fraction",
    ):
        assert rows[1][key] is None and rows[2][key] is None


def test_binned_stats_skips_frames_outside_every_bin():
    rows = op.binned_stats(
        [-1, 0, 0],
        1,
        [[100, 0], [3, 4], [3, 4]],
        [[0, 0], [3, 4], [3, 4]],
        [1.0, 1.0, 1.0],
    )
    assert rows[0]["frames"] == 2 and rows[0]["human_gap_px"] == pytest.approx(5.0)


def test_binned_stats_of_a_halved_copy():
    dh = np.random.default_rng(0).normal(0, 20, size=(30, 2))
    row = op.binned_stats(np.zeros(30, int), 1, dh, 0.5 * dh, np.ones(30))[0]
    assert row["displacement_gain"] == pytest.approx(0.5)
    assert row["mean_cosine"] == pytest.approx(1.0)
    assert row["network_gap_px"] == pytest.approx(0.5 * row["human_gap_px"])


def test_binned_stats_of_a_network_that_copies_the_agent():
    dh = np.random.default_rng(1).normal(0, 20, size=(30, 2))
    row = op.binned_stats(np.zeros(30, int), 1, dh, np.zeros((30, 2)), np.ones(30))[0]
    assert row["displacement_gain"] == pytest.approx(0.0)
    assert row["mean_cosine"] is None and row["cosine_frames"] == 0
    assert row["network_near_copy_fraction"] == 1.0


@pytest.mark.parametrize(
    "args",
    [
        ([0, 0], np.zeros((3, 2)), np.zeros((2, 2)), np.zeros(2)),
        ([0, 0], np.zeros((2, 2)), np.zeros((2, 3)), np.zeros(2)),
        ([0, 0], np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(3)),
    ],
)
def test_binned_stats_rejects_mismatched_shapes(args):
    with pytest.raises(ValueError):
        op.binned_stats(args[0], 1, *args[1:])


# --------------------------------------------------------------- per-frame data


def test_per_frame_data_of_a_perfect_network():
    arrays = structured_arrays()
    d = op.per_frame_data(AffineStub(0.5, 0.25), arrays, [0, 2])
    assert len(d["rows"]) == 32 and (np.diff(d["rows"]) > 0).all()
    assert set(arrays.episode_index[d["rows"]].tolist()) == {0, 2}
    assert np.allclose(d["dp"], d["dh"], atol=5e-3)
    assert (d["length"] == 16).all()
    assert np.array_equal(d["frame"], arrays.frame_index[d["rows"]])


def test_per_frame_data_move_is_nan_only_on_the_last_frame_of_each_episode():
    arrays = structured_arrays()
    d = op.per_frame_data(AffineStub(0.5, 0.25), arrays, [0, 2])
    last = d["frame"] == 15
    assert np.isnan(d["move"][last]).all() and not np.isnan(d["move"][~last]).any()
    states = arrays.states.astype(np.float64)
    rows = d["rows"][~last]
    assert np.allclose(
        d["move"][~last], np.linalg.norm(states[rows + 1] - states[rows], axis=1)
    )


# ---------------------------------------------------------------------- analysis


def test_analyze_accounts_for_every_frame():
    r = op.analyze(AffineStub(0.5, 0.25), structured_arrays(), [0, 1, 2])
    assert r["episodes"] == 3 and r["frames"] == 48 and len(r["by_progress"]) == 10
    assert r["overall"]["frames"] == 48
    assert sum(b["frames"] for b in r["by_progress"]) == 48
    assert [b["frames"] for b in r["by_phase"]] == [48, 0, 0, 0]


def test_analyze_of_a_network_that_shrinks_every_displacement_by_half():
    r = op.analyze(AffineStub(0.75, 0.125), structured_arrays(), [0, 1, 2])
    for b in r["by_progress"]:
        assert b["displacement_gain"] == pytest.approx(0.5, abs=1e-3)
        assert b["network_gap_px"] == pytest.approx(0.5 * b["human_gap_px"], rel=1e-3)
    assert r["overall"]["displacement_gain"] == pytest.approx(0.5, abs=1e-3)


def test_format_tables_has_both_tables_and_the_legend():
    text = op.format_tables(
        op.analyze(AffineStub(0.5, 0.25), structured_arrays(), [0, 1, 2])
    )
    for expected in (
        "all frames",
        "0-10%",
        "90-100%",
        "steps 0-74",
        "steps 225-299",
        "h = human",
        "note:",
    ):
        assert expected in text


def test_figure_is_a_png(tmp_path):
    r = op.analyze(AffineStub(0.5, 0.25), structured_arrays(), [0, 1, 2])
    path = tmp_path / "sub" / "fig.png"
    op.save_progress_figure(r, path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 1000


def test_figure_needs_frames(tmp_path):
    r = op.analyze(AffineStub(0.5, 0.25), structured_arrays(), [0, 1, 2])
    for b in r["by_progress"]:
        b["frames"] = 0
    with pytest.raises(ValueError):
        op.save_progress_figure(r, tmp_path / "fig.png")


def test_main_end_to_end_on_small_data(tmp_path, capsys):
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
    out, fig = tmp_path / "o.json", tmp_path / "f.png"
    code = op.main(
        ["--checkpoint", str(ckpt), "--out", str(out), "--figure", str(fig)],
        arrays_loader=structured_arrays,
    )
    assert code == 0 and fig.exists()
    saved = json.loads(out.read_text())
    assert (
        saved["episodes"] == 3
        and saved["frames"] == 48
        and saved["checkpoint"] == str(ckpt)
    )
    assert len(saved["by_progress"]) == 10 and len(saved["by_phase"]) == 4
    assert "saved" in capsys.readouterr().out


# ---------------------------------------- the real network on the real demonstrations

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "bc_cnn.pt"
needs_both = pytest.mark.skipif(
    not (
        CHECKPOINT.exists()
        and (pusht_data.DEFAULT_ROOT / pusht_data.VIDEO_FILE).exists()
    ),
    reason="needs the trained checkpoint and the downloaded dataset",
)


@pytest.fixture(scope="module")
def real_result():
    model, ckpt = tb.load_checkpoint(CHECKPOINT)
    return op.analyze(model, pusht_data.load_arrays(), ckpt["split"]["val_episode_ids"])


@needs_both
def test_real_overall_numbers_match_the_behavior_survey(real_result):
    o = real_result["overall"]
    assert real_result["episodes"] == 41 and o["frames"] == 4966
    assert o["human_gap_px"] == pytest.approx(24.2, abs=0.06)
    assert o["network_gap_px"] == pytest.approx(17.7, abs=0.06)
    assert o["displacement_gain"] == pytest.approx(0.38, abs=0.006)
    assert o["mean_cosine"] == pytest.approx(0.52, abs=0.006)


@needs_both
def test_real_progress_bins_are_all_populated_and_add_up(real_result):
    assert all(b["frames"] > 100 for b in real_result["by_progress"])
    assert sum(b["frames"] for b in real_result["by_progress"]) == 4966
    assert (
        sum(b["frames"] for b in real_result["by_phase"]) == 4966
    )  # every demo frame is before step 300
