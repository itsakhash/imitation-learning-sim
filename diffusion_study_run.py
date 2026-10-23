"""Run the pre-registered diffusion study: train and evaluate the networks of arms D, C, B and P.

Block 1 is training seeds 0 to 9, block 2 (the replication) is seeds 10 to 19. For each arm and
seed this trains the network (arm_train.py), evaluates it on the standard 50 episodes (seeds
10000 to 10049), evaluates it a second time in a separate process, and compares the two runs with
compare_eval_runs.py, exactly as seed_study_run.py does.

The recipe is fixed. There is no option for the number of epochs, the EMA, the chunk settings or
the number of evaluation episodes. The learning rates of D, C and B are the ones the pilot chose
(reports/pilot_result.json, written by diffusion_pilot.py); the run refuses to start without it.
Arm P always uses 1e-3. Seed 99 (the pilot's) and seeds above 19 are refused.

Two stages, which can be run separately with --stage:
  train     one process per arm builds the data once and trains the arm's missing seeds in turn
            (--train-jobs N runs N arms side by side; each holds about 3 GB of GPU memory)
  evaluate  per network: evaluate, evaluate again, compare. --jobs N runs N networks at once on
            the CPU. Every evaluation process is given --eval-threads (default 1) torch/OpenMP
            threads, whatever --jobs is, so the arithmetic, and therefore the result, does not
            depend on how many jobs ran together.
A step is skipped if all of its output files already exist, so an interrupted run can simply be
started again. A failed training or evaluation stops that network's chain (and a failed training
stops the run before the evaluation stage); a failed reproducibility check does not (the network
stays in the study and its record says it did not reproduce).

Run:  python diffusion_study_run.py --block 1 --dry-run     (print every command, run nothing)
      python diffusion_study_run.py --block 1 --stage train
      python diffusion_study_run.py --block 1 --stage evaluate --jobs 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import arm_train
import diffusion_pilot as pilot

ARMS = ("D", "C", "B", "P")
BLOCKS = {1: tuple(range(0, 10)), 2: tuple(range(10, 20))}
STUDY_SEEDS = tuple(range(0, 20))
EPISODES = 50
EVAL_SCRIPTS = {
    "D": "evaluate_diffusion.py",
    "C": "evaluate_chunk.py",
    "B": "evaluate_bc.py",
    "P": "evaluate_chunk.py",
}
REPORTS = Path("reports")
CHECKPOINTS = Path("checkpoints")


@dataclass(frozen=True)
class Step:
    label: str
    command: tuple
    outputs: tuple
    check: bool = False  # a failing check is recorded but does not stop the run


# ---------------------------------------------------------------- planning (pure, tested)


def validate_seeds(seeds) -> list:
    seeds = list(seeds)
    if not seeds:
        raise ValueError("no training seeds given")
    if any(not isinstance(s, int) for s in seeds):
        raise ValueError("training seeds must be integers")
    outside = [s for s in seeds if s not in STUDY_SEEDS]
    if outside:
        raise ValueError(f"study seeds are 0 to 19 (99 belongs to the pilot); got {outside}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be distinct")
    return seeds


def validate_arms(arms) -> list:
    arms = list(arms)
    unknown = [a for a in arms if a not in ARMS]
    if unknown or not arms or len(set(arms)) != len(arms):
        raise ValueError(f"arms must be distinct and from {list(ARMS)}; got {arms}")
    return arms


def python_command(script: str, *args) -> tuple:
    return (sys.executable, "-W", "ignore::UserWarning", script, *[str(a) for a in args])


def checkpoint_path(arm: str, seed: int) -> Path:
    return CHECKPOINTS / f"{arm_train.default_name(arm, seed)}.pt"


def training_report_path(arm: str, seed: int) -> Path:
    return REPORTS / f"training_{arm_train.default_name(arm, seed)}.json"


def train_steps(arm: str, seeds, lrs: dict, exists) -> list:
    """One step training the seeds of `arm` whose checkpoint and report are not both there."""
    missing = [s for s in seeds if not (exists(checkpoint_path(arm, s)) and exists(training_report_path(arm, s)))]
    if not missing:
        return []
    args = ["--arm", arm, "--seeds", ",".join(str(s) for s in missing)]
    if arm != "P":
        args += ["--lr", f"{lrs[arm]:g}"]
    outputs = tuple(p for s in missing for p in (checkpoint_path(arm, s), training_report_path(arm, s)))
    return [Step(f"train {arm} seeds {missing}", python_command("arm_train.py", *args), outputs)]


def eval_chain(arm: str, seed: int) -> list:
    """evaluate, evaluate again, compare: the three steps of one network."""
    name = arm_train.default_name(arm, seed)
    checkpoint = checkpoint_path(arm, seed)
    evaluation = REPORTS / f"eval_{name}.json"
    rerun = REPORTS / f"rerun_{name}.json"
    reproducibility = REPORTS / f"reproducibility_{name}.json"
    script = EVAL_SCRIPTS[arm]

    def evaluate(out):
        return python_command(script, "--checkpoint", checkpoint, "--name", name, "--episodes", EPISODES, "--out", out)

    return [
        Step(f"evaluate {name}", evaluate(evaluation), (evaluation,)),
        Step(f"evaluate {name} again", evaluate(rerun), (rerun,)),
        Step(
            f"compare the two runs of {name}",
            (sys.executable, "compare_eval_runs.py", str(evaluation), str(rerun)),
            (reproducibility,),
            check=True,
        ),
    ]


def child_env(threads: int, base=None) -> dict:
    if threads < 1:
        raise ValueError("eval threads must be at least 1")
    env = dict(os.environ if base is None else base)
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(threads)
    return env


def needs_pilot(arms) -> bool:
    return any(a in ("D", "C", "B") for a in arms)


def plan(arms, seeds, lrs: dict, exists, stage="all") -> dict:
    """{'train': [steps], 'chains': [[steps of one network], ...]} for the stage asked for."""
    arms, seeds = validate_arms(arms), validate_seeds(seeds)
    if stage not in ("train", "evaluate", "all"):
        raise ValueError("stage must be train, evaluate or all")
    out = {"train": [], "chains": []}
    if stage in ("train", "all"):
        for arm in arms:
            out["train"].extend(train_steps(arm, seeds, lrs, exists))
    if stage in ("evaluate", "all"):
        out["chains"] = [eval_chain(arm, seed) for arm in arms for seed in seeds]
    return out


# ---------------------------------------------------------------- running


def run_chain(steps, runner, exists, clock, log, env=None, lock=None) -> list:
    """Run the steps of one chain in order. Returns (label, status, seconds) for each step
    reached; status is 'ran', 'skipped', 'check_failed' or 'failed' (which ends the chain)."""
    lock = lock or threading.Lock()

    def say(text):
        with lock:
            log(text)

    results = []
    for step in steps:
        if all(exists(p) for p in step.outputs):
            say(f"skip {step.label}: its outputs already exist")
            results.append((step.label, "skipped", 0.0))
            continue
        say(f"run  {step.label}")
        started = clock()
        kwargs = {} if env is None else {"env": env}
        code = runner(list(step.command), **kwargs).returncode
        seconds = clock() - started
        status = "ran" if code == 0 else ("check_failed" if step.check else "failed")
        say(f"     {step.label}: {status} after {seconds:.0f} s")
        results.append((step.label, status, seconds))
        if status == "failed":
            break
    return results


def run_chains(chains, jobs, runner, exists, clock, log, env=None) -> list:
    """Run chains, `jobs` at a time. Returns the flat list of results in chain order."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_chain, chain, runner, exists, clock, log, env, lock) for chain in chains]
        return [r for f in futures for r in f.result()]


def run_study(
    arms,
    seeds,
    lrs,
    *,
    stage="all",
    jobs=1,
    train_jobs=1,
    eval_threads=1,
    runner=subprocess.run,
    exists=lambda p: Path(p).exists(),
    clock=time.perf_counter,
    log=print,
) -> list:
    steps = plan(arms, seeds, lrs, exists, stage)
    results = []
    if steps["train"]:
        results += run_chains([[s] for s in steps["train"]], train_jobs, runner, exists, clock, log)
        if any(status == "failed" for _, status, _ in results):
            return results  # no point evaluating after a failed training
    if steps["chains"]:
        results += run_chains(steps["chains"], jobs, runner, exists, clock, log, env=child_env(eval_threads))
    return results


def format_run_summary(results) -> str:
    counts = {}
    for _, status, _ in results:
        counts[status] = counts.get(status, 0) + 1
    ran = sum(seconds for _, status, seconds in results if status != "skipped")
    lines = [
        "steps: " + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())),
        f"time spent in the steps that ran, added up over parallel jobs: {ran:.0f} s",
    ]
    for label, status, _ in results:
        if status == "failed":
            lines.append(f"FAILED, chain stopped: {label}")
        if status == "check_failed":
            lines.append(f"not reproduced: {label} (the network stays in the study; its record says so)")
    return "\n".join(lines)


def main(argv=None, *, runner=subprocess.run, exists=lambda p: Path(p).exists(), log=print, clock=time.perf_counter) -> int:
    parser = argparse.ArgumentParser(description="Run the pre-registered diffusion study")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--block", type=int, choices=sorted(BLOCKS), help="1: seeds 0-9, 2: seeds 10-19")
    which.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=1, help="evaluation processes at once")
    parser.add_argument("--train-jobs", type=int, default=1, help="arms trained side by side")
    parser.add_argument("--eval-threads", type=int, default=1, help="threads per evaluation process")
    parser.add_argument("--pilot-result", type=Path, default=pilot.RESULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    seeds = list(BLOCKS[args.block]) if args.block else args.seeds
    lrs = pilot.load_chosen_lrs(args.pilot_result) if needs_pilot(args.arms) else {}
    if args.dry_run:
        steps = plan(args.arms, seeds, lrs, exists, args.stage)
        for step in steps["train"] + [s for chain in steps["chains"] for s in chain]:
            log(" ".join(step.command) + "   -> " + ", ".join(str(p) for p in step.outputs))
        log(f"{len(steps['train'])} training steps and {len(steps['chains'])} evaluation chains "
            f"for seeds {seeds}; nothing was run")
        return 0
    results = run_study(
        args.arms, seeds, lrs, stage=args.stage, jobs=args.jobs, train_jobs=args.train_jobs,
        eval_threads=args.eval_threads, runner=runner, exists=exists, clock=clock, log=log,
    )
    log(format_run_summary(results))
    return 0 if all(status in ("ran", "skipped") for _, status, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
