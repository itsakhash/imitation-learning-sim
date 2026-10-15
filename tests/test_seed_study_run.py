from pathlib import Path
from types import SimpleNamespace

import pytest

import seed_study_run as sr

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


# ------------------------------------------------------------------------- naming


def test_network_names():
    assert (
        sr.network_name("bc", 0) == "bc_cnn" and sr.network_name("bc", 3) == "bc_cnn_s3"
    )
    assert (
        sr.network_name("chunk", 0) == "chunk_cnn"
        and sr.network_name("chunk", 9) == "chunk_cnn_s9"
    )


def test_the_default_seeds_are_three_to_nine():
    assert sr.DEFAULT_SEEDS == (3, 4, 5, 6, 7, 8, 9)


# ----------------------------------------------------------------------- planning


def test_a_baseline_seed_is_planned_as_train_evaluate_again_compare():
    train, evaluate, again, compare = sr.plan_network("bc", 3)
    assert [s.label for s in (train, evaluate, again, compare)] == [
        "train bc_cnn_s3",
        "evaluate bc_cnn_s3",
        "evaluate bc_cnn_s3 again",
        "compare the two runs of bc_cnn_s3",
    ]
    t = args_of(train)
    assert "train_bc.py" in t
    for flag, value in [
        ("--epochs", "30"),
        ("--seed", "3"),
        ("--checkpoint", "checkpoints/bc_cnn_s3.pt"),
        ("--report", "reports/bc_training_s3.json"),
        ("--curve", "reports/bc_loss_curve_s3.png"),
    ]:
        assert t[t.index(flag) + 1] == value
    assert [str(p) for p in train.outputs] == [
        "checkpoints/bc_cnn_s3.pt",
        "reports/bc_training_s3.json",
        "reports/bc_loss_curve_s3.png",
    ]
    e, r = args_of(evaluate), args_of(again)
    assert "evaluate_bc.py" in e and "evaluate_bc.py" in r
    for tokens, out in (
        (e, "reports/eval_bc_cnn_s3.json"),
        (r, "reports/rerun_bc_cnn_s3.json"),
    ):
        assert tokens[tokens.index("--name") + 1] == "bc_cnn_s3"
        assert tokens[tokens.index("--episodes") + 1] == "50"
        assert tokens[tokens.index("--checkpoint") + 1] == "checkpoints/bc_cnn_s3.pt"
        assert tokens[tokens.index("--out") + 1] == out
    assert compare.check is True and not (train.check or evaluate.check or again.check)
    assert args_of(compare)[-2:] == [
        "reports/eval_bc_cnn_s3.json",
        "reports/rerun_bc_cnn_s3.json",
    ]
    assert [str(p) for p in compare.outputs] == [
        "reports/reproducibility_bc_cnn_s3.json"
    ]


def test_a_chunk_seed_uses_the_chunk_scripts_and_the_checkpoints_own_n_exec():
    steps = sr.plan_network("chunk", 5)
    t = args_of(steps[0])
    assert "train_chunk.py" in t and t[t.index("--seed") + 1] == "5"
    assert t[t.index("--checkpoint") + 1] == "checkpoints/chunk_cnn_s5.pt"
    assert t[t.index("--report") + 1] == "reports/chunk_training_s5.json"
    assert t[t.index("--curve") + 1] == "reports/chunk_loss_curve_s5.png"
    assert "evaluate_chunk.py" in args_of(steps[1])
    assert not any(
        "--n-exec" in args_of(s) or "--chunk-len" in args_of(s) for s in steps
    )


def test_seed_zero_uses_the_unsuffixed_names_of_the_existing_networks():
    steps = sr.plan_network("bc", 0)
    assert [str(p) for p in steps[0].outputs][0] == "checkpoints/bc_cnn.pt"
    assert [str(p) for p in steps[1].outputs] == ["reports/eval_bc_cnn.json"]


def test_the_default_plan_has_fourteen_trainings_twenty_eight_evaluations_and_fourteen_checks():
    steps = sr.plan(sr.DEFAULT_SEEDS)
    assert len(steps) == 56
    assert sum(s.label.startswith("train ") for s in steps) == 14
    assert sum(s.label.startswith("evaluate ") for s in steps) == 28
    assert sum(s.check for s in steps) == 14


def test_the_recipe_is_fixed_in_every_command():
    for step in sr.plan([3, 4]):
        tokens = args_of(step)
        if "--epochs" in tokens:
            assert tokens[tokens.index("--epochs") + 1] == "30"
        if "--episodes" in tokens:
            assert tokens[tokens.index("--episodes") + 1] == "50"
        assert not any(
            t in tokens
            for t in ("--lr", "--batch-size", "--val-fraction", "--split-seed")
        )


def test_every_script_the_plan_runs_exists():
    scripts = {t for s in sr.plan([3]) for t in args_of(s) if t.endswith(".py")}
    assert scripts == {
        "train_bc.py",
        "evaluate_bc.py",
        "train_chunk.py",
        "evaluate_chunk.py",
        "compare_eval_runs.py",
    }
    assert all((ROOT / s).exists() for s in scripts)


def test_seeds_must_be_distinct_non_negative_integers():
    for bad in ([], [3, 3], [-1], [1.5]):
        with pytest.raises(ValueError):
            sr.plan(bad)
    with pytest.raises(ValueError):
        sr.plan_network("diffusion", 3)


# ------------------------------------------------------------------------ running


def test_every_step_runs_when_no_output_exists():
    runner = FakeRunner()
    steps = sr.plan([3])
    results = sr.run_plan(
        steps,
        runner=runner,
        exists=lambda p: False,
        clock=ticking_clock(),
        log=lambda *_: None,
    )
    assert len(runner.commands) == 8 and [s for _, s, _ in results] == ["ran"] * 8
    assert all(seconds == 1.0 for _, _, seconds in results)


def test_steps_whose_outputs_all_exist_are_skipped():
    runner = FakeRunner()
    steps = sr.plan([3])
    done = {str(p) for s in steps[:4] for p in s.outputs}  # the baseline is finished
    results = sr.run_plan(
        steps, runner=runner, exists=lambda p: str(p) in done, log=lambda *_: None
    )
    assert [s for _, s, _ in results] == ["skipped"] * 4 + ["ran"] * 4
    assert len(runner.commands) == 4 and all(
        "chunk" in " ".join(c) for c in runner.commands
    )


def test_a_step_with_only_some_outputs_is_run_again():
    runner = FakeRunner()
    steps = sr.plan_network("bc", 3)
    only_checkpoint = {str(steps[0].outputs[0])}
    sr.run_plan(
        steps[:1],
        runner=runner,
        exists=lambda p: str(p) in only_checkpoint,
        log=lambda *_: None,
    )
    assert len(runner.commands) == 1


def test_a_failed_training_stops_the_run():
    runner = FakeRunner({"train_bc.py": 1})
    results = sr.run_plan(
        sr.plan([3]), runner=runner, exists=lambda p: False, log=lambda *_: None
    )
    assert [s for _, s, _ in results] == ["failed"] and len(runner.commands) == 1


def test_a_failed_evaluation_stops_the_run():
    runner = FakeRunner({"evaluate_bc.py": 2})
    results = sr.run_plan(
        sr.plan([3]), runner=runner, exists=lambda p: False, log=lambda *_: None
    )
    assert [s for _, s, _ in results] == ["ran", "failed"]


def test_a_failed_reproducibility_check_does_not_stop_the_run():
    runner = FakeRunner({"compare_eval_runs.py": 1})
    results = sr.run_plan(
        sr.plan([3]), runner=runner, exists=lambda p: False, log=lambda *_: None
    )
    statuses = [s for _, s, _ in results]
    assert (
        len(results) == 8
        and statuses.count("check_failed") == 2
        and "failed" not in statuses
    )


def test_the_summary_lists_failures_and_unreproduced_networks():
    text = sr.format_run_summary(
        [
            ("train a", "ran", 60.0),
            ("compare the two runs of a", "check_failed", 1.0),
            ("train b", "skipped", 0.0),
        ]
    )
    assert "1 check_failed" in text and "1 ran" in text and "1 skipped" in text
    assert (
        "time spent in the steps that ran: 61 s" in text
        and "not reproduced: compare the two runs of a" in text
    )
    assert "FAILED" in sr.format_run_summary([("train a", "failed", 3.0)])


# ---------------------------------------------------------------------- main


def test_a_dry_run_prints_every_command_and_runs_nothing():
    runner, lines = FakeRunner(), []
    assert sr.main(["--dry-run"], runner=runner, log=lines.append) == 0
    assert runner.commands == [] and len(lines) == 57 and "nothing was run" in lines[-1]
    assert any("train_chunk.py" in line and "--seed 9" in line for line in lines)


def test_main_returns_zero_when_everything_ran_or_was_skipped():
    runner = FakeRunner()
    assert (
        sr.main(
            ["--seeds", "3"],
            runner=runner,
            exists=lambda p: False,
            log=lambda *_: None,
            clock=ticking_clock(),
        )
        == 0
    )
    assert len(runner.commands) == 8


def test_main_returns_one_when_a_step_failed_or_a_check_did_not_pass():
    assert (
        sr.main(
            ["--seeds", "3"],
            runner=FakeRunner({"train_bc.py": 1}),
            exists=lambda p: False,
            log=lambda *_: None,
        )
        == 1
    )
    assert (
        sr.main(
            ["--seeds", "3"],
            runner=FakeRunner({"compare_eval_runs.py": 1}),
            exists=lambda p: False,
            log=lambda *_: None,
        )
        == 1
    )


def test_main_rejects_bad_seeds_before_running_anything():
    runner = FakeRunner()
    with pytest.raises(ValueError):
        sr.main(["--seeds", "3", "3"], runner=runner, log=lambda *_: None)
    assert runner.commands == []
