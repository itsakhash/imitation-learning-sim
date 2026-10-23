"""The learning-rate pilot of the diffusion study (see diffusion_preregistration.md).

For each of D (diffusion), C (chunked control) and B (baseline control), three learning rates
(1e-3, 3e-4, 1e-4) are tried once on a pilot network with training seed 99 (outside the study
seeds 0 to 19), at the full 370 epochs, with the same recipe as the study (arm_train.py: EMA 0.999,
batch 256, Adam, no selection of checkpoints). The rate with the lowest first-action error in
pixels on the 41 validation episodes at the last epoch (EMA weights; for D one seeded DDIM sample
per frame) is the rate for all 20 networks of that arm. Ties go to the rate listed first. Nothing
else is tuned, the grid is fixed, and no simulator episode is run: the pilot only trains and
validates offline.

Nine trainings, 2 data builds (chunk length 16 for D and C, length 1 for B). Resumable: a pilot
network whose report exists is not trained again, and the result file is written once, when all
nine reports exist. It is never overwritten (delete it by hand only if the pilot is to be redone,
and say so in the README).

    python diffusion_pilot.py --dry-run      (lists the nine trainings, runs nothing)
    python diffusion_pilot.py                (several minutes to an hour of GPU time; ask first)

Writes checkpoints/pilot/pilot_<arm>_lr<rate>.pt, reports/pilot/training_pilot_<arm>_lr<rate>.json
and reports/pilot_result.json, which diffusion_study_run.py reads.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

GRID = (1e-3, 3e-4, 1e-4)
ARMS = ("D", "C", "B")
PILOT_SEED = 99
BATCH_SIZE = 256
REPORT_DIR = Path("reports/pilot")
CHECKPOINT_DIR = Path("checkpoints/pilot")
RESULT = Path("reports/pilot_result.json")


# ---------------------------------------------------------------- pure helpers (tested)


def lr_key(lr: float) -> str:
    return f"{lr:g}"


def pilot_name(arm: str, lr: float) -> str:
    return f"pilot_{arm}_lr{lr_key(lr)}"


def report_path(report_dir, arm: str, lr: float) -> Path:
    return Path(report_dir) / f"training_{pilot_name(arm, lr)}.json"


def plan(arms=ARMS, grid=GRID) -> list:
    """(arm, lr) in training order: arm by arm, rates in the listed order."""
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {list(ARMS)}")
    if len(set(arms)) != len(list(arms)):
        raise ValueError("an arm is listed twice")
    return [(arm, lr) for arm in arms for lr in grid]


def choose_lr(maes: dict, grid=GRID) -> float:
    """The rate with the lowest error. maes maps lr -> first-action error (px). Non-finite
    errors are not eligible; ties go to the rate that comes first in the grid."""
    eligible = [(maes[lr], i, lr) for i, lr in enumerate(grid) if lr in maes and math.isfinite(maes[lr])]
    if not eligible:
        raise ValueError("no finite validation error to choose from")
    return min(eligible)[2]


def collect(report_dir, arms=ARMS, grid=GRID) -> dict:
    """arm -> {lr: final first-action error} for every pilot report that exists."""
    out = {}
    for arm in arms:
        for lr in grid:
            path = report_path(report_dir, arm, lr)
            if path.exists():
                out.setdefault(arm, {})[lr] = float(json.loads(path.read_text())["final"]["val_mae_px"])
    return out


def build_result(collected: dict, epochs, arms=ARMS, grid=GRID) -> dict:
    """The result record; raises if any of the arms x rates is missing."""
    missing = [(a, lr) for a in arms for lr in grid if lr not in collected.get(a, {})]
    if missing:
        raise ValueError(f"pilot incomplete, missing {missing}")
    result = {
        "pilot_seed": PILOT_SEED,
        "epochs": epochs,
        "grid": list(grid),
        "rule": "lowest first-action validation error (px) at the last epoch, EMA weights; ties to the first rate in the grid",
        "arms": {},
    }
    for arm in arms:
        maes = collected[arm]
        chosen = choose_lr(maes, grid)
        result["arms"][arm] = {
            "val_mae_px": {lr_key(lr): maes[lr] for lr in grid},
            "chosen_lr": chosen,
            "at_edge_of_grid": chosen in (grid[0], grid[-1]),
        }
    return result


def write_result(path, result: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; the pilot result is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def load_chosen_lrs(path=RESULT) -> dict:
    """arm -> chosen learning rate, from the pilot result file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run diffusion_pilot.py first")
    arms = json.loads(path.read_text())["arms"]
    missing = [a for a in ARMS if a not in arms]
    if missing:
        raise ValueError(f"{path} has no result for arms {missing}")
    return {arm: float(arms[arm]["chosen_lr"]) for arm in ARMS}


def format_result(result: dict) -> str:
    lines = [f"pilot (seed {result['pilot_seed']}, {result['epochs']} epochs): first-action validation error in px"]
    for arm, entry in result["arms"].items():
        cells = "  ".join(f"lr {k}: {v:.2f}" for k, v in entry["val_mae_px"].items())
        edge = "  (best rate is at the edge of the grid; the grid stays as pre-registered)" if entry["at_edge_of_grid"] else ""
        lines.append(f"  {arm}: {cells}  ->  chosen {entry['chosen_lr']:g}{edge}")
    return "\n".join(lines)


# ---------------------------------------------------------------- running


def run_pilot(arms, train_fn, *, report_dir=REPORT_DIR, result_path=RESULT, epochs=None, log=print):
    """Train what has not been trained (train_fn(arm, lr, name) writes the report), then, if all
    nine reports exist, write and return the result record (else None)."""
    todo = plan(arms)
    for i, (arm, lr) in enumerate(todo, 1):
        name = pilot_name(arm, lr)
        if report_path(report_dir, arm, lr).exists():
            log(f"[{i}/{len(todo)}] skip {name}: its report exists")
            continue
        log(f"[{i}/{len(todo)}] train {name}")
        train_fn(arm, lr, name)
    collected = collect(report_dir)
    complete = all(lr in collected.get(a, {}) for a in ARMS for lr in GRID)
    if not complete:
        log("not all nine pilot networks exist yet: no result file written")
        return None
    if Path(result_path).exists():
        log(f"{result_path} already exists: left as it is")
        return json.loads(Path(result_path).read_text())
    result = build_result(collected, epochs if epochs is not None else _full_epochs())
    write_result(result_path, result)
    log(format_result(result))
    log(f"saved {result_path}")
    return result


def _full_epochs() -> int:
    import arm_train

    return arm_train.EPOCHS


class RealTrainer:
    """Trains one pilot network with arm_train.run_one on resident data, keeping the data of the
    last chunk length in memory (D and C share it)."""

    def __init__(self, device_name, arrays_loader, report_dir, checkpoint_dir, epochs=None):
        self.device_name, self.arrays_loader = device_name, arrays_loader
        self.report_dir, self.checkpoint_dir, self.epochs = report_dir, checkpoint_dir, epochs
        self._arrays = None
        self._cache = None  # (chunk_len, train_res, val_res)

    def _data(self, chunk_len):
        import arm_train as at
        import pusht_data
        import resident_data as rd
        import train_bc as tb
        from train_chunk import ChunkDataset

        if self._arrays is None:
            self.device = tb.pick_device(self.device_name)
            self._arrays = (self.arrays_loader or pusht_data.load_arrays)()
            self.train_ids, self.val_ids = pusht_data.split_episode_ids(
                self._arrays.n_episodes, at.VAL_FRACTION, at.SPLIT_SEED
            )
        if self._cache is None or self._cache[0] != chunk_len:
            self._cache = None  # free the old data before building the new
            train_res = rd.build_resident(self._arrays, ChunkDataset(self._arrays, self.train_ids, chunk_len), self.device)
            val_res = rd.build_resident(self._arrays, ChunkDataset(self._arrays, self.val_ids, chunk_len), self.device)
            self._cache = (chunk_len, train_res, val_res)
        return self._cache[1], self._cache[2]

    def __call__(self, arm, lr, name):
        import arm_train as at

        train_res, val_res = self._data(at.ARM_SETTINGS[arm]["arm"].chunk_len)
        args = SimpleNamespace(
            name=name,
            checkpoint_dir=self.checkpoint_dir,
            report_dir=self.report_dir,
            overwrite=False,
            epochs=self.epochs,
            batch_size=BATCH_SIZE,
        )
        at.run_one(arm, PILOT_SEED, lr, args, self._arrays, self.train_ids, self.val_ids, train_res, val_res, self.device)


def main(argv=None, *, arrays_loader=None, epochs=None) -> int:
    parser = argparse.ArgumentParser(description="Learning-rate pilot of the diffusion study")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args(argv)

    todo = plan(args.arms)
    if args.dry_run:
        for arm, lr in todo:
            done = "exists" if report_path(args.report_dir, arm, lr).exists() else "to train"
            print(f"{pilot_name(arm, lr)}  (seed {PILOT_SEED}, lr {lr_key(lr)})  {done}")
        print(f"{len(todo)} pilot networks; nothing was run")
        return 0
    trainer = RealTrainer(args.device, arrays_loader, args.report_dir, args.checkpoint_dir, epochs)
    run_pilot(args.arms, trainer, report_dir=args.report_dir, result_path=args.result, epochs=epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
