import dataclasses
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import pusht_data as pdata


def make_fake_arrays(lengths=(5, 3, 4), seed=0):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=np.int64)
    n = int(lengths.sum())
    done = np.zeros(n, dtype=bool)
    done[np.cumsum(lengths) - 1] = True
    return pdata.PushTArrays(
        images=rng.integers(0, 256, size=(n, 96, 96, 3), dtype=np.uint8),
        states=rng.uniform(0, 512, size=(n, 2)).astype(np.float32),
        actions=rng.uniform(0, 512, size=(n, 2)).astype(np.float32),
        episode_index=np.repeat(np.arange(len(lengths)), lengths).astype(np.int64),
        frame_index=np.concatenate([np.arange(c) for c in lengths]).astype(np.int64),
        reward=rng.uniform(0, 1, size=n).astype(np.float32),
        done=done,
        success=np.zeros(n, dtype=bool),
        episode_lengths=lengths,
    )


@pytest.fixture
def fake():
    return make_fake_arrays()


# ---------------------------------------------------------------- validation

BROKEN = {
    "wrong image size": lambda a: dict(images=a.images[:, :48]),
    "wrong image dtype": lambda a: dict(images=a.images.astype(np.float32)),
    "fewer images than rows": lambda a: dict(images=a.images[:-1]),
    "episode index not sorted": lambda a: dict(
        episode_index=a.episode_index[::-1].copy()
    ),
    "episode index skips a value": lambda a: dict(episode_index=a.episode_index * 2),
    "frame index never restarts": lambda a: dict(
        frame_index=np.arange(len(a.frame_index), dtype=np.int64)
    ),
    "metadata lengths disagree": lambda a: dict(episode_lengths=a.episode_lengths + 1),
    "action above range": lambda a: dict(actions=a.actions + 600),
    "action below range": lambda a: dict(actions=a.actions - 600),
    "nan in states": lambda a: dict(
        states=np.where(
            np.arange(len(a.states))[:, None] == 0, np.nan, a.states
        ).astype(np.float32)
    ),
    "fewer actions than rows": lambda a: dict(actions=a.actions[:-1]),
}


def test_valid_fake_arrays_pass(fake):
    pdata.validate_arrays(fake)


@pytest.mark.parametrize("name", list(BROKEN))
def test_validation_rejects_broken_data(fake, name):
    broken = dataclasses.replace(fake, **BROKEN[name](fake))
    with pytest.raises(pdata.DatasetError):
        pdata.validate_arrays(broken)


# ------------------------------------------------------------------- dataset


def test_item_shapes_and_ranges(fake):
    item = pdata.PushTDataset(fake)[0]
    assert item["image"].shape == (3, 96, 96)
    assert item["image"].dtype == torch.float32
    assert 0.0 <= float(item["image"].min()) and float(item["image"].max()) <= 1.0
    for key in ("agent_pos", "action"):
        assert item[key].shape == (2,)
        assert 0.0 <= float(item[key].min()) and float(item[key].max()) <= 1.0


def test_image_is_channels_first_with_correct_channel_order(fake):
    images = fake.images.copy()
    images[0, 5, 7] = (10, 20, 30)  # row 5, column 7
    img = pdata.PushTDataset(dataclasses.replace(fake, images=images))[0]["image"]
    assert torch.allclose(
        img[:, 5, 7] * 255, torch.tensor([10.0, 20.0, 30.0]), atol=1e-3
    )


def test_positions_and_actions_are_scaled_by_512(fake):
    states = fake.states.copy()
    actions = fake.actions.copy()
    states[0] = (256, 128)
    actions[0] = (512, 0)
    item = pdata.PushTDataset(
        dataclasses.replace(fake, states=states, actions=actions)
    )[0]
    assert torch.allclose(item["agent_pos"], torch.tensor([0.5, 0.25]))
    assert torch.allclose(item["action"], torch.tensor([1.0, 0.0]))


def test_length_without_subset(fake):
    assert len(pdata.PushTDataset(fake)) == 12


def test_subset_contains_only_requested_episodes(fake):
    ds = pdata.PushTDataset(fake, episode_ids=[0, 2])
    assert len(ds) == 5 + 4
    assert {ds[i]["episode_index"] for i in range(len(ds))} == {0, 2}


@pytest.mark.parametrize("ids", [[], [3], [-1], [0, 99]])
def test_bad_episode_ids_raise(fake, ids):
    with pytest.raises(ValueError):
        pdata.PushTDataset(fake, episode_ids=ids)


# --------------------------------------------------------------------- split


def test_split_is_disjoint_and_complete():
    train, val = pdata.split_episode_ids(206, 0.2, seed=0)
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(206))
    assert len(val) == 41


def test_split_is_deterministic_and_depends_on_seed():
    a = pdata.split_episode_ids(50, 0.2, seed=1)
    assert a == pdata.split_episode_ids(50, 0.2, seed=1)
    assert a != pdata.split_episode_ids(50, 0.2, seed=2)


@pytest.mark.parametrize(
    "n, frac", [(1, 0.5), (10, 0.0), (10, 1.0), (10, 0.01), (3, 0.99)]
)
def test_split_rejects_bad_arguments(n, frac):
    with pytest.raises(ValueError):
        pdata.split_episode_ids(n, frac)


def test_train_and_validation_datasets_share_no_frames(fake):
    train, val = pdata.split_episode_ids(3, 0.34, seed=0)
    a = pdata.PushTDataset(fake, train)
    b = pdata.PushTDataset(fake, val)
    assert not set(a.indices.tolist()) & set(b.indices.tolist())
    assert len(a) + len(b) == fake.n_frames


# --------------------------------------------------------------------- stats


def test_dataset_stats_known_values(fake):
    stats = pdata.dataset_stats(fake)
    assert stats["episodes"] == 3 and stats["frames"] == 12
    assert stats["episode_length_min"] == 3 and stats["episode_length_max"] == 5
    assert stats["success_flags"] == 0 and stats["done_flags"] == 3
    assert 0.0 <= stats["image_white_pixel_fraction"] <= 1.0
    json.dumps(stats)  # must be JSON serializable


def test_dataset_stats_reward_maximum_per_episode(fake):
    reward = (fake.frame_index / 10.0).astype(
        np.float32
    )  # per-episode max: 0.4, 0.2, 0.3
    stats = pdata.dataset_stats(dataclasses.replace(fake, reward=reward))
    assert stats["reward_max_per_episode_min"] == pytest.approx(0.2)
    assert stats["reward_max_per_episode_median"] == pytest.approx(0.3)
    assert stats["reward_max_per_episode_max"] == pytest.approx(0.4)


# ---------------------------------------------------------- disk round trips


def _fixed_list(values):
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, 2)


def write_fake_dataset(root, a):
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "observation.state": _fixed_list(a.states),
                "action": _fixed_list(a.actions),
                "episode_index": pa.array(a.episode_index, pa.int64()),
                "frame_index": pa.array(a.frame_index, pa.int64()),
                "next.reward": pa.array(a.reward, pa.float32()),
                "next.done": pa.array(a.done, pa.bool_()),
                "next.success": pa.array(a.success, pa.bool_()),
            }
        ),
        root / pdata.DATA_FILE,
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array(
                    np.arange(len(a.episode_lengths)), pa.int64()
                ),
                "length": pa.array(a.episode_lengths, pa.int64()),
            }
        ),
        root / pdata.EPISODES_FILE,
    )


def test_load_arrays_round_trip_through_parquet(tmp_path, fake):
    write_fake_dataset(tmp_path, fake)
    loaded = pdata.load_arrays(tmp_path, decoder=lambda path: fake.images)
    for name in (
        "states",
        "actions",
        "episode_index",
        "frame_index",
        "reward",
        "done",
        "success",
        "episode_lengths",
        "images",
    ):
        assert np.array_equal(getattr(loaded, name), getattr(fake, name)), name


def test_load_arrays_gives_the_decoder_the_video_path(tmp_path, fake):
    write_fake_dataset(tmp_path, fake)
    seen = []
    pdata.load_arrays(tmp_path, decoder=lambda p: seen.append(p) or fake.images)
    assert seen == [tmp_path / pdata.VIDEO_FILE]


def test_load_arrays_reports_missing_files(tmp_path):
    with pytest.raises(pdata.DatasetError, match="download"):
        pdata.load_arrays(tmp_path, decoder=lambda p: None)


def test_decode_video_reports_a_missing_file(tmp_path):
    with pytest.raises(pdata.DatasetError, match="download"):
        pdata.decode_video(tmp_path / "nope.mp4")


def test_download_pins_the_revision_and_fetches_the_video(tmp_path):
    calls = []

    def snapshot(repo, **kwargs):
        calls.append(("snapshot", repo, kwargs))

    def single_file(repo, filename, **kwargs):
        calls.append(("file", repo, filename, kwargs))

    assert (
        pdata.download_dataset(tmp_path, snapshot=snapshot, single_file=single_file)
        == tmp_path
    )
    assert calls[0][1] == "lerobot/pusht"
    assert calls[0][2]["repo_type"] == "dataset"
    assert calls[0][2]["revision"] == pdata.DATASET_REVISION
    assert calls[1][2] == pdata.VIDEO_FILE
    assert calls[1][3]["revision"] == pdata.DATASET_REVISION


def test_main_report_writes_json(tmp_path, fake, capsys):
    out = tmp_path / "stats.json"
    code = pdata.main(["report", "--out", str(out)], loader=lambda root: fake)
    assert code == 0
    saved = json.loads(out.read_text())
    assert saved["episodes"] == 3 and saved["revision"] == pdata.DATASET_REVISION
    assert "episodes" in capsys.readouterr().out


def test_main_download_calls_the_downloader(tmp_path):
    seen = []
    code = pdata.main(
        ["download", "--root", str(tmp_path)],
        downloader=lambda root: seen.append(root) or root,
    )
    assert code == 0 and seen == [tmp_path]


# ------------------------------------------------- tests on the real dataset

needs_data = pytest.mark.skipif(
    not (pdata.DEFAULT_ROOT / pdata.VIDEO_FILE).exists(),
    reason="dataset not downloaded; run: python pusht_data.py download",
)


@pytest.fixture(scope="module")
def real():
    return pdata.load_arrays()


@needs_data
def test_real_counts_and_episode_lengths(real):
    assert real.n_episodes == 206 and real.n_frames == 25650
    assert real.images.shape == (25650, 96, 96, 3)
    lengths = real.episode_lengths
    assert (int(lengths.min()), int(np.median(lengths)), int(lengths.max())) == (
        49,
        122,
        246,
    )


@needs_data
def test_real_dataset_has_no_success_flags(real):
    # Observed on the pinned revision. If this fails, the data is not what we studied.
    assert not real.success.any()


@needs_data
def test_real_first_rows_are_read_in_the_right_order(real):
    assert np.allclose(real.states[0], [222, 97], atol=1e-3)
    assert np.allclose(real.actions[0], [233, 71], atol=1e-3)


@needs_data
def test_real_images_are_mostly_white_like_the_environment(real):
    assert 247.0 < float(real.images[::50].mean()) < 251.0


def _blue_centroid(img):
    r, g, b = (img[..., i].astype(int) for i in range(3))
    mask = (b > 180) & (r < 130) & (g < 160) & (b - r > 80)
    if mask.sum() < 3:
        return None
    rows, cols = np.nonzero(mask)
    return np.array([cols.mean() + 0.5, rows.mean() + 0.5])  # (x, y) in image pixels


def _median_agent_error(real, offset, step=50):
    errors = []
    for i in range(1, real.n_frames - 1, step):
        centroid = _blue_centroid(real.images[i])
        if centroid is None:
            continue
        expected = real.states[i + offset] / pdata.COORD_SCALE * pdata.IMAGE_SIZE
        errors.append(float(np.linalg.norm(centroid - expected)))
    return float(np.median(errors)), len(errors)


@needs_data
def test_real_video_frames_line_up_with_agent_positions(real):
    err0, n0 = _median_agent_error(real, 0)
    err_prev, _ = _median_agent_error(real, -1)
    err_next, _ = _median_agent_error(real, +1)
    print(
        f"median agent error (px): offset -1: {err_prev:.2f}, 0: {err0:.2f}, +1: {err_next:.2f}; frames used: {n0}"
    )
    assert n0 >= 400
    assert err0 < 2.0, f"agent not where the state says (offset 0: {err0:.2f}px)"
    assert err0 <= min(
        err_prev, err_next
    ), "a shifted frame matches better than the aligned one"
