"""Run the pre-registered executed-targets study: evaluate the chunked checkpoints with fewer
predicted targets executed per prediction.

Nothing is trained here. For each requested number of executed targets and each of the ten chunked
checkpoints of the seed study (training seeds 0 to 9), this evaluates the checkpoint on the standard
50 episodes (seeds 10000 to 10049) executing that many of its 16 predicted targets before predicting
again (evaluate_chunk.py --n-exec), evaluates it a second time in a separate process, and compares
the two runs with compare_eval_runs.py. The 8-target evaluations already exist (they are the seed
study), so 8 is not accepted here.

The recipe is fixed (50 evaluation episodes) and there is no option to change it. A step is skipped
if all its output files already exist, so an interrupted run can simply be started again. A failed
evaluation stops the run; a failed reproducibility check does not (the network stays in the study,
its record says it did not reproduce, and nexec_study.py refuses to analyse it unless told to).

Run:  python nexec_run.py --n-execs 1 --seeds 0      (one network, one condition: measures the time)
      python nexec_run.py                            (1, 2 and 4 executed targets, seeds 0 to 9)
      python nexec_run.py --dry-run                  (print every command without running it)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import ablation_study as ab
import nexec_study as nx
import seed_study_run as sr
from seed_study_run import (
    CHECKPOINTS, EPISODES, REPORTS, Step, format_run_summary, python_command, run_plan,
)

CHUNK_LEN = 16
DEFAULT_N_EXECS = (1, 2, 4)
DEFAULT_SEEDS = tuple(range(nx.NETWORKS))


def checkpoint_path(seed: int) -> Path:
    """The seed study's chunked checkpoint for a training seed."""
    return CHECKPOINTS / f"{ab.network_name('chunk', ab.FULL_SIZE, seed)}.pt"


def validate_n_execs(n_execs) -> list:
    n_execs = list(n_execs)
    if not n_execs:
        raise ValueError("no numbers of executed targets given")
    for n in n_execs:
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= CHUNK_LEN or n == nx.REFERENCE:
            raise ValueError(
                f"executed targets must be integers from 1 to {CHUNK_LEN} other than {nx.REFERENCE} "
                f"(the {nx.REFERENCE}-target evaluations are the seed study), got {n!r}"
            )
    if len(set(n_execs)) != len(n_execs):
        raise ValueError("numbers of executed targets must be distinct")
    return n_execs


def plan_network(seed: int, n_exec: int) -> list:
    validate_n_execs([n_exec])
    sr.validate_seeds([seed])
    name = nx.network_name(seed, n_exec)
    checkpoint = checkpoint_path(seed)
    evaluation = REPORTS / f"eval_{name}.json"
    rerun = REPORTS / f"rerun_{name}.json"
    reproducibility = REPORTS / f"reproducibility_{name}.json"
    return [
        Step(
            f"evaluate {name}",
            python_command("evaluate_chunk.py", "--checkpoint", checkpoint, "--name", name,
                           "--episodes", EPISODES, "--n-exec", n_exec, "--out", evaluation),
            (evaluation,),
        ),
        Step(
            f"evaluate {name} again",
            python_command("evaluate_chunk.py", "--checkpoint", checkpoint, "--name", name,
                           "--episodes", EPISODES, "--n-exec", n_exec, "--out", rerun),
            (rerun,),
        ),
        Step(
            f"compare the two runs of {name}",
            (sys.executable, "compare_eval_runs.py", str(evaluation), str(rerun)),
            (reproducibility,),
            check=True,
        ),
    ]


def plan(n_execs, seeds) -> list:
    """Every step, network by network (training seed), then number of executed targets."""
    n_execs, seeds = validate_n_execs(n_execs), sr.validate_seeds(seeds)
    steps = []
    for seed in seeds:
        for n_exec in n_execs:
            steps.extend(plan_network(seed, n_exec))
    return steps


def main(argv=None, *, runner=subprocess.run, exists=lambda p: Path(p).exists(),
         log=print, clock=time.perf_counter) -> int:
    parser = argparse.ArgumentParser(description="Run the pre-registered executed-targets study")
    parser.add_argument("--n-execs", type=int, nargs="+", default=list(DEFAULT_N_EXECS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    steps = plan(args.n_execs, args.seeds)
    if args.dry_run:
        for step in steps:
            log(" ".join(step.command) + "   -> " + ", ".join(str(p) for p in step.outputs))
        log(f"{len(steps)} steps for executed targets {args.n_execs} and training seeds {args.seeds}; "
            "nothing was run")
        return 0
    missing = [str(checkpoint_path(s)) for s in args.seeds if not exists(checkpoint_path(s))]
    if missing:
        log("error: missing checkpoints: " + ", ".join(missing))
        return 1
    results = run_plan(steps, runner=runner, exists=exists, clock=clock, log=log)
    log(format_run_summary(results))
    return 0 if all(status in ("ran", "skipped") for _, status, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
