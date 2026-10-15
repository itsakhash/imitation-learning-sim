from pathlib import Path
from types import SimpleNamespace

import pytest

import ablation_run as ar
import ablation_study as ab
import train_bc as tb

ROOT = Path(__file__).resolve().parents[1]


class FakeRunner:
    def __init__(self, codes=None):
        self.codes = codes or {}
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        for fragment, code in self.codes.items():
            if any(fragment in str(part) for part in command):
                return SimpleNamespace(returncode=code)
        return SimpleNamespace(returncode=0)


def ticking_clock():
    t = {"now": 0.0}

    def clock():
        t["now"] += 1.0
        return t["now"]

    return clock


def args_of(step):
    return [str(a) for a in step.command]


def flag(tokens, name):
    return tokens[tokens.index(name) + 1]


# ----------------------------------------------------------------------- planning

def test_a_baseline_network_is_planned_as_train_evaluate_again_compare():
    train, evaluate, again, compare = ar.plan_network("bc", 20, 3)
    assert [s.label for s in (train, evaluate, again, compare)] == [
        "train bc_cnn_n20_s3", "evaluate bc_cnn_n20_s3", "evaluate bc_cnn_n20_s3 again",
        "compare the two runs of bc_cnn_n20_s3",
    ]
    t = args_of(train)
    assert "train_bc.py" in t
    expected = [("--epochs", "30"), ("--seed", "3"), ("--train-episodes", "20"), ("--subset-seed", "3"),
                ("--checkpoint", "checkpoints/bc_cnn_n20_s3.pt"),
                ("--report", "reports/bc_training_n20_s3.json"),
                ("--curve", "reports/bc_loss_curve_n20_s3.png")]
    for name, value in expected:
        assert flag(t, name) == value
    assert [str(p) for p in train.outputs] == [
        "checkpoints/bc_cnn_n20_s3.pt", "reports/bc_training_n20_s3.json", "reports/bc_loss_curve_n20_s3.png"]
    for step, out in ((evaluate, "reports/eval_bc_cnn_n20_s3.json"), (again, "reports/rerun_bc_cnn_n20_s3.json")):
        tokens = args_of(step)
        assert "evaluate_bc.py" in tokens and flag(tokens, "--name") == "bc_cnn_n20_s3"
        assert flag(tokens, "--episodes") == "50" and flag(tokens, "--out") == out
        assert flag(tokens, "--checkpoint") == "checkpoints/bc_cnn_n20_s3.pt"
    assert compare.check is True and not (train.check or evaluate.check or again.check)
    assert args_of(compare)[-2:] == ["reports/eval_bc_cnn_n20_s3.json", "reports/rerun_bc_cnn_n20_s3.json"]
    assert [str(p) for p in compare.outputs] == ["reports/reproducibility_bc_cnn_n20_s3.json"]


def test_a_chunk_network_uses_the_chunk_scripts_and_the_checkpoints_own_settings():
    steps = ar.plan_network("chunk", 100, 2)
    t = args_of(steps[0])
    assert "train_chunk.py" in t and flag(t, "--train-episodes") == "100" and flag(t, "--subset-seed") == "2"
    assert flag(t, "--report") == "reports/chunk_training_n100_s2.json"
    assert flag(t, "--curve") == "reports/chunk_loss_curve_n100_s2.png"
    assert "evaluate_chunk.py" in args_of(steps[1])
    assert not any("--n-exec" in args_of(s) or "--chunk-len" in args_of(s) for s in steps)


def test_the_default_plan_has_thirty_trainings_sixty_evaluations_and_thirty_checks():
    steps = ar.plan(ar.DEFAULT_SIZES, ar.DEFAULT_SEEDS)
    assert len(steps) == 120
    assert sum(s.label.startswith("train ") for s in steps) == 30
    assert sum(s.label.startswith("evaluate ") for s in steps) == 60
    assert sum(s.check for s in steps) == 30


def test_the_subset_seed_is_the_training_seed_in_every_training_command():
    for step in ar.plan(ar.DEFAULT_SIZES, ar.DEFAULT_SEEDS):
        if step.label.startswith("train "):
            t = args_of(step)
            assert flag(t, "--subset-seed") == flag(t, "--seed")


def test_the_recipe_is_fixed_in_every_command():
    for step in ar.plan([20, 50], [0, 1]):
        tokens = args_of(step)
        if "--epochs" in tokens:
            assert flag(tokens, "--epochs") == "30"
        if "--episodes" in tokens:
            assert flag(tokens, "--episodes") == "50"
        assert not any(t in tokens for t in ("--lr", "--batch-size", "--val-fraction", "--split-seed"))


def test_the_plan_goes_seed_by_seed_then_size_then_method():
    labels = [s.label for s in ar.plan(ar.DEFAULT_SIZES, ar.DEFAULT_SEEDS)]
    trainings = [label for label in labels if label.startswith("train ")]
    assert trainings[:6] == ["train bc_cnn_n20_s0", "train chunk_cnn_n20_s0", "train bc_cnn_n50_s0",
                             "train chunk_cnn_n50_s0", "train bc_cnn_n100_s0", "train chunk_cnn_n100_s0"]
    assert trainings[6] == "train bc_cnn_n20_s1"


def test_every_output_path_in_the_plan_is_distinct():
    outputs = [str(p) for s in ar.plan(ar.DEFAULT_SIZES, ar.DEFAULT_SEEDS) for p in s.outputs]
    assert len(outputs) == len(set(outputs))


def test_the_planned_evaluation_files_match_the_names_the_analysis_expects():
    expected = {
        str(path)
        for method_paths in ab.default_paths(Path("reports"), sizes=(20, 50, 100)).values()
        for paths in method_paths.values() for path in paths
    }
    planned = {str(p) for s in ar.plan(ar.DEFAULT_SIZES, ar.DEFAULT_SEEDS)
               if s.label.startswith("evaluate ") and not s.label.endswith("again") for p in s.outputs}
    assert planned == expected


def test_every_script_the_plan_runs_exists():
    scripts = {t for s in ar.plan([20], [0]) for t in args_of(s) if t.endswith(".py")}
    assert scripts == {"train_bc.py", "evaluate_bc.py", "train_chunk.py", "evaluate_chunk.py", "compare_eval_runs.py"}
    assert all((ROOT / s).exists() for s in scripts)


def test_the_subsets_a_seed_uses_are_nested_across_sizes():
    pool = list(range(165))
    for seed in range(5):
        small, medium, large = (tb.subsample_episodes(pool, n, seed) for n in (20, 50, 100))
        assert set(small) <= set(medium) <= set(large)
    assert tb.subsample_episodes(pool, 20, 0) != tb.subsample_episodes(pool, 20, 1)


@pytest.mark.parametrize("sizes", [[], [0], [165], [200], [20, 20], [2.5]])
def test_bad_sizes_are_rejected(sizes):
    with pytest.raises(ValueError):
        ar.plan(sizes, [0])


@pytest.mark.parametrize("seeds", [[], [0, 0], [-1]])
def test_bad_seeds_are_rejected(seeds):
    with pytest.raises(ValueError):
        ar.plan([20], seeds)


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        ar.plan_network("diffusion", 20, 0)


# ------------------------------------------------------------------------- running

def test_every_step_runs_when_no_output_exists():
    runner = FakeRunner()
    results = ar.run_plan(ar.plan([20], [0]), runner=runner, exists=lambda p: False,
                          clock=ticking_clock(), log=lambda *_: None)
    assert len(runner.commands) == 8 and [s for _, s, _ in results] == ["ran"] * 8


def test_a_finished_network_is_skipped_and_the_rest_runs():
    runner = FakeRunner()
    steps = ar.plan([20], [0])
    done = {str(p) for s in steps[:4] for p in s.outputs}
    results = ar.run_plan(steps, runner=runner, exists=lambda p: str(p) in done, log=lambda *_: None)
    assert [s for _, s, _ in results] == ["skipped"] * 4 + ["ran"] * 4
    assert all("chunk" in " ".join(c) for c in runner.commands)


def test_a_failed_training_stops_the_run():
    runner = FakeRunner({"train_bc.py": 1})
    results = ar.run_plan(ar.plan([20], [0]), runner=runner, exists=lambda p: False, log=lambda *_: None)
    assert [s for _, s, _ in results] == ["failed"] and len(runner.commands) == 1


def test_a_failed_reproducibility_check_does_not_stop_the_run():
    runner = FakeRunner({"compare_eval_runs.py": 1})
    results = ar.run_plan(ar.plan([20], [0]), runner=runner, exists=lambda p: False, log=lambda *_: None)
    statuses = [s for _, s, _ in results]
    assert len(results) == 8 and statuses.count("check_failed") == 2 and "failed" not in statuses


# --------------------------------------------------------------------------- main

def test_a_dry_run_prints_every_command_and_runs_nothing():
    runner, lines = FakeRunner(), []
    assert ar.main(["--dry-run"], runner=runner, log=lines.append) == 0
    assert runner.commands == [] and len(lines) == 121 and "nothing was run" in lines[-1]
    assert any("train_chunk.py" in line and "--train-episodes 100" in line and "--seed 4" in line for line in lines)


def test_main_returns_zero_when_everything_ran_or_was_skipped():
    runner = FakeRunner()
    assert ar.main(["--sizes", "20", "--seeds", "0"], runner=runner, exists=lambda p: False,
                   log=lambda *_: None, clock=ticking_clock()) == 0
    assert len(runner.commands) == 8


def test_main_returns_one_when_a_step_failed_or_a_check_did_not_pass():
    assert ar.main(["--sizes", "20", "--seeds", "0"], runner=FakeRunner({"train_bc.py": 1}),
                   exists=lambda p: False, log=lambda *_: None) == 1
    assert ar.main(["--sizes", "20", "--seeds", "0"], runner=FakeRunner({"compare_eval_runs.py": 1}),
                   exists=lambda p: False, log=lambda *_: None) == 1


def test_main_rejects_bad_arguments_before_running_anything():
    runner = FakeRunner()
    with pytest.raises(ValueError):
        ar.main(["--sizes", "165"], runner=runner, log=lambda *_: None)
    with pytest.raises(ValueError):
        ar.main(["--seeds", "1", "1"], runner=runner, log=lambda *_: None)
    assert runner.commands == []
