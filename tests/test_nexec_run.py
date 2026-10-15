from pathlib import Path
from types import SimpleNamespace

import pytest

import nexec_run as nr
import nexec_study as nx

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


def checkpoints_only(path):
    return str(path).endswith(".pt")


# ----------------------------------------------------------------------- planning

def test_a_network_is_planned_as_evaluate_again_compare():
    evaluate, again, compare = nr.plan_network(0, 4)
    assert [s.label for s in (evaluate, again, compare)] == [
        "evaluate chunk_cnn_x4", "evaluate chunk_cnn_x4 again", "compare the two runs of chunk_cnn_x4"]
    for step, out in ((evaluate, "reports/eval_chunk_cnn_x4.json"), (again, "reports/rerun_chunk_cnn_x4.json")):
        tokens = args_of(step)
        assert "evaluate_chunk.py" in tokens
        assert flag(tokens, "--checkpoint") == "checkpoints/chunk_cnn.pt"
        assert flag(tokens, "--name") == "chunk_cnn_x4" and flag(tokens, "--episodes") == "50"
        assert flag(tokens, "--n-exec") == "4" and flag(tokens, "--out") == out
        assert [str(p) for p in step.outputs] == [out]
    assert compare.check is True and not (evaluate.check or again.check)
    assert args_of(compare)[-2:] == ["reports/eval_chunk_cnn_x4.json", "reports/rerun_chunk_cnn_x4.json"]
    assert [str(p) for p in compare.outputs] == ["reports/reproducibility_chunk_cnn_x4.json"]


def test_a_later_training_seed_uses_its_own_checkpoint_and_name():
    evaluate = nr.plan_network(3, 1)[0]
    tokens = args_of(evaluate)
    assert flag(tokens, "--checkpoint") == "checkpoints/chunk_cnn_s3.pt"
    assert flag(tokens, "--name") == "chunk_cnn_s3_x1" and flag(tokens, "--n-exec") == "1"


def test_the_default_plan_has_sixty_evaluations_thirty_checks_and_no_training():
    steps = nr.plan(nr.DEFAULT_N_EXECS, nr.DEFAULT_SEEDS)
    assert len(steps) == 90
    assert sum(s.label.startswith("evaluate ") for s in steps) == 60
    assert sum(s.check for s in steps) == 30
    assert not any(s.label.startswith("train") for s in steps)


def test_only_the_evaluation_and_comparison_scripts_are_run_and_they_exist():
    scripts = {t for s in nr.plan([1, 2, 4], [0]) for t in args_of(s) if t.endswith(".py")}
    assert scripts == {"evaluate_chunk.py", "compare_eval_runs.py"}
    assert all((ROOT / s).exists() for s in scripts)


def test_the_plan_goes_network_by_network_then_by_executed_targets():
    steps = nr.plan(nr.DEFAULT_N_EXECS, nr.DEFAULT_SEEDS)
    firsts = [s.label for s in steps if s.label.startswith("evaluate ") and not s.label.endswith("again")]
    assert firsts[:4] == ["evaluate chunk_cnn_x1", "evaluate chunk_cnn_x2", "evaluate chunk_cnn_x4",
                          "evaluate chunk_cnn_s1_x1"]


def test_every_output_path_in_the_plan_is_distinct():
    outputs = [str(p) for s in nr.plan(nr.DEFAULT_N_EXECS, nr.DEFAULT_SEEDS) for p in s.outputs]
    assert len(outputs) == len(set(outputs))


def test_the_planned_evaluation_files_match_the_names_the_analysis_expects():
    expected = {str(p) for ps in nx.default_paths(Path("reports"), n_execs=(1, 2, 4))["chunk"].values() for p in ps}
    planned = {str(p) for s in nr.plan(nr.DEFAULT_N_EXECS, nr.DEFAULT_SEEDS)
               if s.label.startswith("evaluate ") and not s.label.endswith("again") for p in s.outputs}
    assert planned == expected


def test_the_checkpoints_are_the_ones_the_seed_study_trained():
    used = {flag(args_of(s), "--checkpoint") for s in nr.plan(nr.DEFAULT_N_EXECS, nr.DEFAULT_SEEDS)
            if "--checkpoint" in args_of(s)}
    assert used == {"checkpoints/chunk_cnn.pt"} | {f"checkpoints/chunk_cnn_s{k}.pt" for k in range(1, 10)}


@pytest.mark.parametrize("n_execs", [[], [0], [8], [17], [2, 2], [2.5]])
def test_bad_numbers_of_executed_targets_are_rejected(n_execs):
    with pytest.raises(ValueError):
        nr.plan(n_execs, [0])


@pytest.mark.parametrize("seeds", [[], [0, 0], [-1]])
def test_bad_seeds_are_rejected(seeds):
    with pytest.raises(ValueError):
        nr.plan([1], seeds)


# ------------------------------------------------------------------------- running

def test_every_step_runs_when_no_output_exists():
    runner = FakeRunner()
    results = nr.run_plan(nr.plan([1, 2], [0]), runner=runner, exists=lambda p: False,
                          clock=ticking_clock(), log=lambda *_: None)
    assert len(runner.commands) == 6 and [s for _, s, _ in results] == ["ran"] * 6


def test_a_finished_condition_is_skipped_and_the_rest_runs():
    runner = FakeRunner()
    steps = nr.plan([1, 2], [0])
    done = {str(p) for s in steps[:3] for p in s.outputs}
    results = nr.run_plan(steps, runner=runner, exists=lambda p: str(p) in done, log=lambda *_: None)
    assert [s for _, s, _ in results] == ["skipped"] * 3 + ["ran"] * 3
    assert all("x2" in " ".join(c) for c in runner.commands)


def test_a_failed_evaluation_stops_the_run():
    runner = FakeRunner({"evaluate_chunk.py": 1})
    results = nr.run_plan(nr.plan([1], [0]), runner=runner, exists=lambda p: False, log=lambda *_: None)
    assert [s for _, s, _ in results] == ["failed"] and len(runner.commands) == 1


def test_a_failed_reproducibility_check_does_not_stop_the_run():
    runner = FakeRunner({"compare_eval_runs.py": 1})
    results = nr.run_plan(nr.plan([1, 2], [0]), runner=runner, exists=lambda p: False, log=lambda *_: None)
    statuses = [s for _, s, _ in results]
    assert len(results) == 6 and statuses.count("check_failed") == 2 and "failed" not in statuses


# --------------------------------------------------------------------------- main

def test_a_dry_run_prints_every_command_and_runs_nothing():
    runner, lines = FakeRunner(), []
    assert nr.main(["--dry-run"], runner=runner, log=lines.append) == 0
    assert runner.commands == [] and len(lines) == 91 and "nothing was run" in lines[-1]
    assert any("--n-exec 1 " in line and "chunk_cnn_s9_x1" in line for line in lines)


def test_main_returns_zero_when_everything_ran_or_was_skipped():
    runner = FakeRunner()
    assert nr.main(["--n-execs", "1", "--seeds", "0"], runner=runner, exists=checkpoints_only,
                   log=lambda *_: None, clock=ticking_clock()) == 0
    assert len(runner.commands) == 3


def test_main_returns_one_when_a_step_failed_or_a_check_did_not_pass():
    assert nr.main(["--n-execs", "1", "--seeds", "0"], runner=FakeRunner({"evaluate_chunk.py": 1}),
                   exists=checkpoints_only, log=lambda *_: None) == 1
    assert nr.main(["--n-execs", "1", "--seeds", "0"], runner=FakeRunner({"compare_eval_runs.py": 1}),
                   exists=checkpoints_only, log=lambda *_: None) == 1


def test_main_refuses_to_run_when_a_checkpoint_is_missing():
    runner, lines = FakeRunner(), []
    code = nr.main(["--n-execs", "1", "--seeds", "0"], runner=runner, exists=lambda p: False, log=lines.append)
    assert code == 1 and runner.commands == []
    assert any("checkpoints/chunk_cnn.pt" in line for line in lines)


def test_main_rejects_bad_arguments_before_running_anything():
    runner = FakeRunner()
    with pytest.raises(ValueError):
        nr.main(["--n-execs", "8"], runner=runner, log=lambda *_: None)
    with pytest.raises(ValueError):
        nr.main(["--seeds", "1", "1"], runner=runner, log=lambda *_: None)
    assert runner.commands == []
