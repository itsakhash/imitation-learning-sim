"""Tests for diffusion_study_run.py. They need no torch and run no real training or evaluation:
the commands are checked, and a fake runner stands in for subprocess.run."""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import diffusion_pilot as dpil
import diffusion_study_run as sr

LRS = {"D": 3e-4, "C": 1e-3, "B": 1e-4}


class FakeRunner:
    """Records every command (and the env it got), can fail chosen commands, and measures how
    many ran at the same time."""

    def __init__(self, fail=(), sleep=0.0):
        self.calls, self.fail, self.sleep = [], tuple(fail), sleep
        self.lock = threading.Lock()
        self.running = self.max_running = 0

    def __call__(self, command, **kwargs):
        with self.lock:
            self.calls.append((list(command), kwargs.get("env")))
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        if self.sleep:
            time.sleep(self.sleep)
        with self.lock:
            self.running -= 1
        text = " ".join(str(c) for c in command)
        return SimpleNamespace(returncode=1 if any(f in text for f in self.fail) else 0)

    def commands(self):
        return [" ".join(str(c) for c in cmd) for cmd, _ in self.calls]


def nothing_exists(_):
    return False


def quiet(*_):
    return None


def clock():
    return 0.0


# ---------------------------------------------------------------- validation


def test_blocks_are_the_preregistered_seed_ranges():
    assert sr.BLOCKS[1] == tuple(range(10)) and sr.BLOCKS[2] == tuple(range(10, 20))


@pytest.mark.parametrize("seeds", [[], [99], [20], [-1], [1, 1], [1.5]])
def test_bad_seeds_are_refused(seeds):
    with pytest.raises(ValueError):
        sr.validate_seeds(seeds)


def test_good_seeds_pass_and_bad_arms_are_refused():
    assert sr.validate_seeds([0, 19, 7]) == [0, 19, 7]
    assert sr.validate_arms(["D", "P"]) == ["D", "P"]
    for arms in ([], ["X"], ["D", "D"]):
        with pytest.raises(ValueError):
            sr.validate_arms(arms)


# ---------------------------------------------------------------- planning


def test_train_steps_train_the_missing_seeds_of_an_arm_in_one_process():
    [step] = sr.train_steps("D", [0, 1, 2], LRS, nothing_exists)
    cmd = " ".join(step.command)
    assert "arm_train.py --arm D --seeds 0,1,2 --lr 0.0003" in cmd
    assert sr.checkpoint_path("D", 0) in step.outputs and sr.training_report_path("D", 2) in step.outputs
    assert Path("checkpoints/diff_cnn_s1.pt") in step.outputs and len(step.outputs) == 6


def test_only_missing_seeds_are_trained_and_nothing_if_all_exist():
    have = {sr.checkpoint_path("C", 0), sr.training_report_path("C", 0)}
    [step] = sr.train_steps("C", [0, 1], LRS, lambda p: p in have)
    assert "--seeds 1 " in " ".join(step.command) + " " and "--lr 0.001" in " ".join(step.command)
    both = {p for s in (0, 1) for p in (sr.checkpoint_path("C", s), sr.training_report_path("C", s))}
    assert sr.train_steps("C", [0, 1], LRS, lambda p: p in both) == []


def test_a_checkpoint_without_its_report_is_trained_again_so_the_problem_shows():
    have = {sr.checkpoint_path("B", 0)}
    [step] = sr.train_steps("B", [0], LRS, lambda p: p in have)
    assert "--seeds 0" in " ".join(step.command)


def test_the_path_check_arm_gets_no_learning_rate_option():
    [step] = sr.train_steps("P", [3], LRS, nothing_exists)
    assert "--arm P" in " ".join(step.command) and "--lr" not in step.command


def test_the_learning_rates_come_from_the_pilot_per_arm():
    for arm, lr in LRS.items():
        [step] = sr.train_steps(arm, [0], LRS, nothing_exists)
        assert f"--lr {lr:g}" in " ".join(step.command)


def test_eval_chain_is_evaluate_again_compare_with_the_right_scripts():
    for arm, script, name in (
        ("D", "evaluate_diffusion.py", "diff_cnn_s4"),
        ("C", "evaluate_chunk.py", "chunk_long_s4"),
        ("B", "evaluate_bc.py", "bc_long_s4"),
        ("P", "evaluate_chunk.py", "chunk_res30_s4"),
    ):
        first, again, compare = sr.eval_chain(arm, 4)
        assert script in first.command and script in again.command
        assert f"--checkpoint {Path('checkpoints') / (name + '.pt')}" in " ".join(first.command)
        assert "--episodes 50" in " ".join(first.command) and f"--name {name}" in " ".join(first.command)
        assert first.outputs == (Path("reports") / f"eval_{name}.json",)
        assert again.outputs == (Path("reports") / f"rerun_{name}.json",)
        assert first.command[-1] != again.command[-1]  # different output files
        assert "compare_eval_runs.py" in compare.command and compare.check is True
        assert compare.outputs == (Path("reports") / f"reproducibility_{name}.json",)


def test_plan_stages():
    both = sr.plan(["D", "C"], [0, 1], LRS, nothing_exists, "all")
    assert len(both["train"]) == 2 and len(both["chains"]) == 4
    train_only = sr.plan(["D", "C"], [0, 1], LRS, nothing_exists, "train")
    assert len(train_only["train"]) == 2 and train_only["chains"] == []
    eval_only = sr.plan(["D", "C"], [0, 1], LRS, nothing_exists, "evaluate")
    assert eval_only["train"] == [] and len(eval_only["chains"]) == 4
    with pytest.raises(ValueError):
        sr.plan(["D"], [0], LRS, nothing_exists, "later")


def test_full_block_sizes():
    plan = sr.plan(sr.ARMS, sr.BLOCKS[1], LRS, nothing_exists)
    assert len(plan["train"]) == 4 and len(plan["chains"]) == 40  # 4 arms x 10 networks


def test_child_env_fixes_the_thread_count():
    env = sr.child_env(3, base={"PATH": "x"})
    assert env["OMP_NUM_THREADS"] == "3" and env["MKL_NUM_THREADS"] == "3" and env["PATH"] == "x"
    with pytest.raises(ValueError):
        sr.child_env(0)


# ---------------------------------------------------------------- running


def test_run_chain_runs_skips_and_stops_on_failure():
    chain = sr.eval_chain("D", 0)
    skip_first = {chain[0].outputs[0]}
    runner = FakeRunner()
    results = sr.run_chain(chain, runner, lambda p: p in skip_first, clock, quiet)
    assert [s for _, s, _ in results] == ["skipped", "ran", "ran"] and len(runner.calls) == 2
    failing = FakeRunner(fail=["--out reports/rerun"])
    results = sr.run_chain(chain, failing, nothing_exists, clock, quiet)
    assert [s for _, s, _ in results] == ["ran", "failed"]  # the comparison is never run
    assert len(failing.calls) == 2


def test_a_failed_comparison_is_recorded_but_is_not_a_failure_of_the_chain():
    runner = FakeRunner(fail=["compare_eval_runs"])
    results = sr.run_chain(sr.eval_chain("C", 1), runner, nothing_exists, clock, quiet)
    assert [s for _, s, _ in results] == ["ran", "ran", "check_failed"]


def test_run_chain_passes_the_env_only_when_given():
    runner = FakeRunner()
    sr.run_chain(sr.eval_chain("D", 0)[:1], runner, nothing_exists, clock, quiet, env={"A": "1"})
    sr.run_chain(sr.eval_chain("D", 1)[:1], runner, nothing_exists, clock, quiet)
    assert runner.calls[0][1] == {"A": "1"} and runner.calls[1][1] is None


def test_chains_run_in_parallel_only_when_jobs_allow_and_results_keep_chain_order():
    chains = [sr.eval_chain("D", s)[:1] for s in range(6)]
    one = FakeRunner(sleep=0.02)
    results = sr.run_chains(chains, 1, one, nothing_exists, clock, quiet)
    assert one.max_running == 1
    many = FakeRunner(sleep=0.05)
    results_many = sr.run_chains(chains, 4, many, nothing_exists, clock, quiet)
    assert many.max_running >= 2
    assert [r[0] for r in results] == [r[0] for r in results_many] == [c[0].label for c in chains]
    with pytest.raises(ValueError):
        sr.run_chains(chains, 0, one, nothing_exists, clock, quiet)


def test_run_study_trains_before_it_evaluates_and_fixes_eval_threads():
    runner = FakeRunner()
    results = sr.run_study(["D", "P"], [0, 1], LRS, jobs=2, eval_threads=2, runner=runner,
                           exists=nothing_exists, clock=clock, log=quiet)
    cmds = runner.commands()
    kinds = ["train" if "arm_train.py" in c else "eval" for c in cmds]
    assert kinds == ["train", "train"] + ["eval"] * 12  # 2 trainings, then 4 networks x 3 steps
    for (cmd, env), kind in zip(runner.calls, kinds):
        if kind == "train":
            assert env is None
        else:
            assert env["OMP_NUM_THREADS"] == "2" and env["MKL_NUM_THREADS"] == "2"
    assert all(status == "ran" for _, status, _ in results)


def test_a_failed_training_stops_the_run_before_any_evaluation():
    runner = FakeRunner(fail=["--arm C"])
    results = sr.run_study(["D", "C", "B"], [0], LRS, runner=runner, exists=nothing_exists, clock=clock, log=quiet)
    assert [s for _, s, _ in results] == ["ran", "failed", "ran"][: len(results)]
    assert not any("evaluate" in c for c in runner.commands())


def test_a_failed_evaluation_ends_only_that_networks_chain():
    runner = FakeRunner(fail=["--name diff_cnn --episodes"])  # seed 0 only; seed 1 is diff_cnn_s1
    results = sr.run_study(["D"], [0, 1], LRS, stage="evaluate", runner=runner, exists=nothing_exists, clock=clock, log=quiet)
    statuses = {label: s for label, s, _ in results}
    assert statuses["evaluate diff_cnn"] == "failed"
    assert "evaluate diff_cnn again" not in statuses
    assert statuses["compare the two runs of diff_cnn_s1"] == "ran"


def test_run_study_with_everything_done_runs_nothing():
    runner = FakeRunner()
    results = sr.run_study(["C"], [0], LRS, runner=runner, exists=lambda p: True, clock=clock, log=quiet)
    assert runner.calls == [] and all(s == "skipped" for _, s, _ in results)


def test_summary_names_failures_and_missing_reproducibility():
    text = sr.format_run_summary([("a", "ran", 2.0), ("b", "failed", 1.0), ("c", "check_failed", 1.0), ("d", "skipped", 0.0)])
    assert "FAILED, chain stopped: b" in text and "not reproduced: c" in text and "1 skipped" in text


# ---------------------------------------------------------------- the command line


def write_pilot_result(path):
    collected = {a: {lr: 10.0 + i for i, lr in enumerate(dpil.GRID)} for a in dpil.ARMS}
    dpil.write_result(path, dpil.build_result(collected, 370))


def test_main_needs_the_pilot_result_unless_only_the_path_check_runs(tmp_path):
    missing = str(tmp_path / "none.json")
    with pytest.raises(FileNotFoundError):
        sr.main(["--block", "1", "--dry-run", "--pilot-result", missing], log=quiet)
    assert sr.main(["--block", "1", "--arms", "P", "--dry-run", "--pilot-result", missing], log=quiet) == 0


def test_main_dry_run_prints_commands_and_runs_nothing(tmp_path):
    result = tmp_path / "pilot_result.json"
    write_pilot_result(result)
    lines, runner = [], FakeRunner()
    code = sr.main(["--block", "2", "--arms", "D", "--dry-run", "--pilot-result", str(result)],
                   runner=runner, exists=nothing_exists, log=lines.append)
    assert code == 0 and runner.calls == []
    text = "\n".join(lines)
    assert "--seeds 10,11,12,13,14,15,16,17,18,19 --lr 0.001" in text  # the pilot made lr 1e-3 first in the grid
    assert "nothing was run" in text and "1 training steps and 10 evaluation chains" in text


def test_main_block_and_seeds_are_exclusive_and_one_is_required(tmp_path):
    with pytest.raises(SystemExit):
        sr.main(["--block", "1", "--seeds", "3", "--dry-run"], log=quiet)
    with pytest.raises(SystemExit):
        sr.main(["--dry-run"], log=quiet)
    with pytest.raises(ValueError):
        sr.main(["--seeds", "99", "--arms", "P", "--dry-run"], log=quiet)


def test_main_returns_one_after_a_failure_and_zero_otherwise(tmp_path):
    result = tmp_path / "pilot_result.json"
    write_pilot_result(result)
    argv = ["--seeds", "0", "--arms", "B", "--stage", "evaluate", "--pilot-result", str(result)]
    assert sr.main(argv, runner=FakeRunner(), exists=nothing_exists, log=quiet) == 0
    assert sr.main(argv, runner=FakeRunner(fail=["evaluate_bc.py"]), exists=nothing_exists, log=quiet) == 1
    assert sr.main(argv, runner=FakeRunner(fail=["compare_eval_runs"]), exists=nothing_exists, log=quiet) == 1
