import json

import numpy as np
import pytest

import pusht_data
import train_bc as tb
import train_chunk as tc


def arrays():
    """24 episodes of 16 frames; blank images; the action is 0.5 * position + 128."""
    rng = np.random.default_rng(0)
    n_episodes, length = 24, 16
    n = n_episodes * length
    states = rng.uniform(0, 400, size=(n, 2)).astype(np.float32)
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


def run(module, tmp_path, name, *extra):
    argv = [
        "--epochs",
        "1",
        "--batch-size",
        "32",
        "--device",
        "cpu",
        "--val-fraction",
        "0.25",
        "--checkpoint",
        str(tmp_path / f"{name}.pt"),
        "--report",
        str(tmp_path / f"{name}.json"),
        "--curve",
        str(tmp_path / f"{name}.png"),
        *extra,
    ]
    if module is tc:
        argv += ["--chunk-len", "4", "--n-exec", "2"]
    assert module.main(argv, arrays_loader=arrays) == 0
    return json.loads((tmp_path / f"{name}.json").read_text())


MODULES = pytest.mark.parametrize("module", [tb, tc], ids=["baseline", "chunked"])


@MODULES
def test_different_subset_seeds_keep_different_episodes_and_the_same_validation_set(
    module, tmp_path
):
    pool = run(module, tmp_path, "all")["split"]["train_episode_ids"]
    a = run(module, tmp_path, "a", "--train-episodes", "6", "--subset-seed", "1")[
        "split"
    ]
    b = run(module, tmp_path, "b", "--train-episodes", "6", "--subset-seed", "2")[
        "split"
    ]
    assert a["train_episode_ids"] != b["train_episode_ids"]
    assert a["val_episode_ids"] == b["val_episode_ids"]
    assert len(a["train_episode_ids"]) == len(b["train_episode_ids"]) == 6
    assert set(a["train_episode_ids"]) <= set(pool) and set(
        b["train_episode_ids"]
    ) <= set(pool)
    assert set(a["train_episode_ids"]).isdisjoint(a["val_episode_ids"])


@MODULES
def test_the_same_subset_seed_keeps_the_same_episodes(module, tmp_path):
    a = run(module, tmp_path, "a", "--train-episodes", "6", "--subset-seed", "3")[
        "split"
    ]
    b = run(module, tmp_path, "b", "--train-episodes", "6", "--subset-seed", "3")[
        "split"
    ]
    assert a == b


@MODULES
def test_the_subset_does_not_depend_on_the_training_seed(module, tmp_path):
    a = run(
        module,
        tmp_path,
        "a",
        "--train-episodes",
        "6",
        "--subset-seed",
        "3",
        "--seed",
        "0",
    )["split"]
    b = run(
        module,
        tmp_path,
        "b",
        "--train-episodes",
        "6",
        "--subset-seed",
        "3",
        "--seed",
        "1",
    )["split"]
    assert a == b


@MODULES
def test_subsets_are_nested_for_a_fixed_subset_seed(module, tmp_path):
    small = run(module, tmp_path, "s", "--train-episodes", "6", "--subset-seed", "4")[
        "split"
    ]["train_episode_ids"]
    large = run(module, tmp_path, "l", "--train-episodes", "12", "--subset-seed", "4")[
        "split"
    ]["train_episode_ids"]
    assert set(small) <= set(large) and len(small) == 6 and len(large) == 12


@MODULES
def test_the_default_subset_seed_is_the_split_seed(module, tmp_path):
    default = run(module, tmp_path, "d", "--train-episodes", "6")["split"]
    explicit = run(
        module, tmp_path, "e", "--train-episodes", "6", "--subset-seed", "0"
    )["split"]
    assert default == explicit
    other_default = run(
        module, tmp_path, "f", "--train-episodes", "6", "--split-seed", "3"
    )["split"]
    other_explicit = run(
        module,
        tmp_path,
        "g",
        "--train-episodes",
        "6",
        "--split-seed",
        "3",
        "--subset-seed",
        "3",
    )["split"]
    assert other_default == other_explicit


@MODULES
def test_the_config_records_the_subset_seed_only_when_a_subset_is_used(
    module, tmp_path
):
    with_subset = run(
        module, tmp_path, "a", "--train-episodes", "6", "--subset-seed", "2"
    )["config"]
    assert with_subset["subset_seed"] == 2 and with_subset["train_episodes"] == 6
    default = run(module, tmp_path, "b", "--train-episodes", "6")["config"]
    assert default["subset_seed"] == 0
    full = run(module, tmp_path, "c")["config"]
    assert (
        full["subset_seed"] is None
        and full["train_episodes"] == full["available_train_episodes"]
    )
