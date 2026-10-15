"""Run the pre-registered seed study: train and evaluate more training seeds of both networks.

For each requested training seed and each method (one-step baseline and action-chunking network),
this trains the network with the standard recipe (30 epochs; the same code, split and settings as
for seeds 0 to 2), evaluates it on the standard 50 episodes (seeds 10000 to 10049), evaluates it a
second time in a separate process, and compares the two runs with compare_eval_runs.py.

The recipe is fixed: there is no option to change the number of epochs, the chunk settings or the
number of evaluation episodes, so that nothing can be tuned along the way. A step is skipped if
all of its output files already exist, so an interrupted run can simply be started again. A failed
training or evaluation stops the run; a failed reproducibility check does not (the network stays in
the study, its record says it did not reproduce, and seed_study.py will refuse to analyse it unless
told to).

Run:  python seed_study_run.py --seeds 3            (one seed: measures how long a seed takes)
      python seed_study_run.py --seeds 4 5 6 7 8 9
      python seed_study_run.py --dry-run             (print every command without running it)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

EPOCHS = 30
EPISODES = 50
DEFAULT_SEEDS = (3, 4, 5, 6, 7, 8, 9)
REPORTS = Path("reports")
CHECKPOINTS = Path("checkpoints")

KINDS = {
    "bc": {
        "name": "bc_cnn",
        "train": "train_bc.py",
        "evaluate": "evaluate_bc.py",
        "training_report": "bc_training",
        "curve": "bc_loss_curve",
    },
    "chunk": {
        "name": "chunk_cnn",
        "train": "train_chunk.py",
        "evaluate": "evaluate_chunk.py",
        "training_report": "chunk_training",
        "curve": "chunk_loss_curve",
    },
}


@dataclass(frozen=True)
class Step:
    label: str
    command: tuple
    outputs: tuple
    check: bool = False  # a failing check is recorded but does not stop the run


def network_name(kind: str, seed: int) -> str:
    base = KINDS[kind]["name"]
    return base if seed == 0 else f"{base}_s{seed}"


def python_command(script: str, *args) -> tuple:
    return (
        sys.executable,
        "-W",
        "ignore::UserWarning",
        script,
        *[str(a) for a in args],
    )


def validate_seeds(seeds) -> list:
    seeds = list(seeds)
    if not seeds:
        raise ValueError("no training seeds given")
    if any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("training seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be distinct")
    return seeds


def plan_network(kind: str, seed: int) -> list:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(KINDS)}")
    validate_seeds([seed])
    spec = KINDS[kind]
    name = network_name(kind, seed)
    suffix = "" if seed == 0 else f"_s{seed}"
    checkpoint = CHECKPOINTS / f"{spec['name']}{suffix}.pt"
    training = REPORTS / f"{spec['training_report']}{suffix}.json"
    curve = REPORTS / f"{spec['curve']}{suffix}.png"
    evaluation = REPORTS / f"eval_{name}.json"
    rerun = REPORTS / f"rerun_{name}.json"
    reproducibility = REPORTS / f"reproducibility_{name}.json"
    return [
        Step(
            f"train {name}",
            python_command(
                spec["train"],
                "--epochs",
                EPOCHS,
                "--seed",
                seed,
                "--checkpoint",
                checkpoint,
                "--report",
                training,
                "--curve",
                curve,
            ),
            (checkpoint, training, curve),
        ),
        Step(
            f"evaluate {name}",
            python_command(
                spec["evaluate"],
                "--checkpoint",
                checkpoint,
                "--name",
                name,
                "--episodes",
                EPISODES,
                "--out",
                evaluation,
            ),
            (evaluation,),
        ),
        Step(
            f"evaluate {name} again",
            python_command(
                spec["evaluate"],
                "--checkpoint",
                checkpoint,
                "--name",
                name,
                "--episodes",
                EPISODES,
                "--out",
                rerun,
            ),
            (rerun,),
        ),
        Step(
            f"compare the two runs of {name}",
            (sys.executable, "compare_eval_runs.py", str(evaluation), str(rerun)),
            (reproducibility,),
            check=True,
        ),
    ]


def plan(seeds) -> list:
    steps = []
    for seed in validate_seeds(seeds):
        for kind in KINDS:
            steps.extend(plan_network(kind, seed))
    return steps


def run_plan(
    steps,
    runner=subprocess.run,
    exists=lambda p: Path(p).exists(),
    clock=time.perf_counter,
    log=print,
) -> list:
    """Run the steps in order. Returns (label, status, seconds) for every step reached, where
    status is 'ran', 'skipped', 'check_failed' or 'failed'. A 'failed' step stops the run.
    """
    results = []
    total = len(steps)
    for i, step in enumerate(steps, 1):
        if all(exists(p) for p in step.outputs):
            log(f"[{i}/{total}] skip {step.label}: its outputs already exist")
            results.append((step.label, "skipped", 0.0))
            continue
        log(f"[{i}/{total}] run  {step.label}")
        started = clock()
        code = runner(list(step.command)).returncode
        seconds = clock() - started
        status = "ran" if code == 0 else ("check_failed" if step.check else "failed")
        log(f"      {status} after {seconds:.0f} s")
        results.append((step.label, status, seconds))
        if status == "failed":
            break
    return results


def format_run_summary(results) -> str:
    counts = {}
    for _, status, _ in results:
        counts[status] = counts.get(status, 0) + 1
    ran = sum(seconds for _, status, seconds in results if status != "skipped")
    lines = [
        "steps: " + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())),
        f"time spent in the steps that ran: {ran:.0f} s",
    ]
    for label, status, _ in results:
        if status == "failed":
            lines.append(f"FAILED, run stopped: {label}")
        if status == "check_failed":
            lines.append(
                f"not reproduced: {label} (it stays in the study; seed_study.py will refuse "
                "to analyse it without a passing record unless told to)"
            )
    return "\n".join(lines)


def main(
    argv=None,
    *,
    runner=subprocess.run,
    exists=lambda p: Path(p).exists(),
    log=print,
    clock=time.perf_counter,
) -> int:
    parser = argparse.ArgumentParser(description="Run the pre-registered seed study")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    steps = plan(args.seeds)
    if args.dry_run:
        for step in steps:
            log(
                " ".join(step.command)
                + "   -> "
                + ", ".join(str(p) for p in step.outputs)
            )
        log(f"{len(steps)} steps for training seeds {args.seeds}; nothing was run")
        return 0
    results = run_plan(steps, runner=runner, exists=exists, clock=clock, log=log)
    log(format_run_summary(results))
    return 0 if all(status in ("ran", "skipped") for _, status, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
