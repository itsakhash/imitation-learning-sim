"""Run the pre-registered demonstration-count ablation: train and evaluate the networks.

For each requested training-set size and training seed, and for both methods (one-step baseline and
action-chunking network), this trains the network on a random subset of that many of the 165
training episodes, evaluates it on the standard 50 episodes (seeds 10000 to 10049), evaluates it a
second time in a separate process, and compares the two runs with compare_eval_runs.py.

The training seed also picks the subset (--subset-seed equals --seed), so for a given seed the
smaller subsets are contained in the larger ones and different seeds see different subsets. The
recipe is fixed (30 epochs, 50 evaluation episodes) and there is no option to change it, so nothing
can be tuned along the way. A step is skipped if all its output files already exist, so an
interrupted run can simply be started again. A failed training or evaluation stops the run; a failed
reproducibility check does not (the network stays in the study, its record says it did not
reproduce, and ablation_study.py refuses to analyse it unless told to).

Run:  python ablation_run.py --sizes 20 --seeds 0       (one pair of networks: measures the time)
      python ablation_run.py                             (sizes 20 50 100, seeds 0 to 4)
      python ablation_run.py --dry-run                   (print every command without running it)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import ablation_study as ab
import seed_study_run as sr
from seed_study_run import (
    CHECKPOINTS, EPISODES, EPOCHS, KINDS, REPORTS, Step, format_run_summary, python_command, run_plan,
)

DEFAULT_SIZES = (20, 50, 100)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
METHOD_OF_KIND = {"bc": "baseline", "chunk": "chunk"}


def validate_sizes(sizes) -> list:
    sizes = list(sizes)
    if not sizes:
        raise ValueError("no training-set sizes given")
    if any(not isinstance(s, int) or s < 1 or s >= ab.FULL_SIZE for s in sizes):
        raise ValueError(
            f"sizes must be integers from 1 to {ab.FULL_SIZE - 1}: the full {ab.FULL_SIZE} episodes "
            "are the networks of the seed study"
        )
    if len(set(sizes)) != len(sizes):
        raise ValueError("sizes must be distinct")
    return sizes


def plan_network(kind: str, size: int, seed: int) -> list:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(KINDS)}")
    validate_sizes([size])
    sr.validate_seeds([seed])
    spec = KINDS[kind]
    name = ab.network_name(METHOD_OF_KIND[kind], size, seed)
    suffix = f"_n{size}_s{seed}"
    checkpoint = CHECKPOINTS / f"{name}.pt"
    training = REPORTS / f"{spec['training_report']}{suffix}.json"
    curve = REPORTS / f"{spec['curve']}{suffix}.png"
    evaluation = REPORTS / f"eval_{name}.json"
    rerun = REPORTS / f"rerun_{name}.json"
    reproducibility = REPORTS / f"reproducibility_{name}.json"
    return [
        Step(
            f"train {name}",
            python_command(spec["train"], "--epochs", EPOCHS, "--seed", seed, "--train-episodes", size,
                           "--subset-seed", seed, "--checkpoint", checkpoint, "--report", training,
                           "--curve", curve),
            (checkpoint, training, curve),
        ),
        Step(
            f"evaluate {name}",
            python_command(spec["evaluate"], "--checkpoint", checkpoint, "--name", name,
                           "--episodes", EPISODES, "--out", evaluation),
            (evaluation,),
        ),
        Step(
            f"evaluate {name} again",
            python_command(spec["evaluate"], "--checkpoint", checkpoint, "--name", name,
                           "--episodes", EPISODES, "--out", rerun),
            (rerun,),
        ),
        Step(
            f"compare the two runs of {name}",
            (sys.executable, "compare_eval_runs.py", str(evaluation), str(rerun)),
            (reproducibility,),
            check=True,
        ),
    ]


def plan(sizes, seeds) -> list:
    """Every step, seed by seed, then size by size, then method."""
    sizes, seeds = validate_sizes(sizes), sr.validate_seeds(seeds)
    steps = []
    for seed in seeds:
        for size in sizes:
            for kind in KINDS:
                steps.extend(plan_network(kind, size, seed))
    return steps


def main(argv=None, *, runner=subprocess.run, exists=lambda p: Path(p).exists(),
         log=print, clock=time.perf_counter) -> int:
    parser = argparse.ArgumentParser(description="Run the pre-registered demonstration-count ablation")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    steps = plan(args.sizes, args.seeds)
    if args.dry_run:
        for step in steps:
            log(" ".join(step.command) + "   -> " + ", ".join(str(p) for p in step.outputs))
        log(f"{len(steps)} steps for training sizes {args.sizes} and training seeds {args.seeds}; "
            "nothing was run")
        return 0
    results = run_plan(steps, runner=runner, exists=exists, clock=clock, log=log)
    log(format_run_summary(results))
    return 0 if all(status in ("ran", "skipped") for _, status, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
