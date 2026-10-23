"""Training on GPU-resident data, mirroring train_chunk.train, and the anchor run.

train_resident is train_chunk.train with the DataLoader replaced by resident_data: the same
examples, the same shuffling order, the same Adam steps, the same running-loss and validation
bookkeeping. evaluate_resident mirrors train_chunk.evaluate_loader.

The anchor run asks whether this really reproduces the project's existing recipe. It trains
the chunked network with training seed 0 for 30 epochs three ways and compares the histories:
  - the saved history of the original run (reports/chunk_training.json), if it is there;
  - a rerun of the original training loop (train_chunk.train), which shows how much two runs
    of the SAME recipe differ on this GPU;
  - the resident loop.
Criterion fixed before the anchor run: the resident loop counts as equivalent if the largest
relative difference of its validation loss from the saved history, over the 30 epochs, is at
most 3 times the rerun's own difference from the saved history (the rerun's difference is
taken as at least 1e-3, so a rerun that happens to match exactly does not make the bar
unreachable). Without a saved history only the resident-versus-rerun numbers are printed and
no verdict is given.

Run (about 2 minutes, uses the GPU; nothing of the study is changed):
    python resident_train.py --anchor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import resident_data as rd

REPORT_DEFAULT = Path("reports/anchor_resident.json")
REFERENCE_DEFAULT = Path("reports/chunk_training.json")
FACTOR = 3.0
FLOOR = 1e-3
EPS = 1e-12


# ---------------------------------------------------------------- pure helpers (tested)


def find_history(obj):
    """The first dict with list values for 'train_loss' and 'val_loss', searched recursively."""
    if isinstance(obj, dict):
        if isinstance(obj.get("train_loss"), list) and isinstance(obj.get("val_loss"), list):
            return obj
        for value in obj.values():
            found = find_history(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_history(value)
            if found is not None:
                return found
    return None


def max_relative_difference(a, b) -> float:
    """Largest |a_i - b_i| / max(|b_i|, tiny) over two equally long sequences."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("sequences must have the same, non-zero length")
    return max(abs(x - y) / max(abs(y), EPS) for x, y in zip(a, b))


def compare_histories(a: dict, b: dict) -> dict:
    """Largest relative differences of the per-epoch losses (a measured against b)."""
    out = {
        "epochs": len(a["train_loss"]),
        "train_loss_max_rel": max_relative_difference(a["train_loss"], b["train_loss"]),
        "val_loss_max_rel": max_relative_difference(a["val_loss"], b["val_loss"]),
    }
    if a.get("val_mae_px") and b.get("val_mae_px"):
        out["final_val_mae_px_abs"] = abs(a["val_mae_px"][-1] - b["val_mae_px"][-1])
    return out


def equivalence_verdict(
    rerun_rel: float, resident_rel: float, factor: float = FACTOR, floor: float = FLOOR
) -> str:
    """The pre-stated rule: resident <= factor * max(rerun, floor)."""
    return "equivalent" if resident_rel <= factor * max(rerun_rel, floor) else "not equivalent"


# ---------------------------------------------------------------- training on resident data


def _defaults(loss_fn, seed_fn, coord_scale):
    if loss_fn is None:
        from train_chunk import chunk_loss as loss_fn
    if seed_fn is None:
        import train_chunk

        seed_fn = train_chunk.tb.seed_everything
    if coord_scale is None:
        from pusht_data import COORD_SCALE as coord_scale
    return loss_fn, seed_fn, coord_scale


def evaluate_resident(model, res, batch_size: int = 256, coord_scale=None) -> dict:
    """Mean loss over valid chunk positions, first-action error and whole-chunk error in pixels,
    like train_chunk.evaluate_loader."""
    import torch

    if coord_scale is None:
        from pusht_data import COORD_SCALE as coord_scale
    model.eval()
    squared = chunk_abs = first_abs = valid = frames = 0.0
    with torch.no_grad():
        for start, stop in rd.batch_bounds(len(res), batch_size):
            batch = res.batch(torch.arange(start, stop, device=res.device))
            error = model(batch["image"], batch["agent_pos"]) - batch["actions"]
            weight = batch["mask"].unsqueeze(-1)
            squared += float(((error**2) * weight).sum())
            chunk_abs += float((error.abs() * weight).sum())
            first_abs += float(error[:, 0].abs().sum())
            valid += float(batch["mask"].sum())
            frames += len(batch["actions"])
    return {
        "loss": squared / (valid * 2),
        "first_mae_px": first_abs / (frames * 2) * coord_scale,
        "chunk_mae_px": chunk_abs / (valid * 2) * coord_scale,
    }


def train_resident(
    model,
    train_res,
    val_res,
    *,
    epochs: int,
    batch_size: int = 256,
    lr: float = 1e-3,
    device=None,
    seed: int = 0,
    log=print,
    loss_fn=None,
    seed_fn=None,
    coord_scale=None,
) -> dict:
    """train_chunk.train on resident data; returns the same per-epoch history."""
    import torch

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if lr <= 0:
        raise ValueError("lr must be positive")
    loss_fn, seed_fn, coord_scale = _defaults(loss_fn, seed_fn, coord_scale)
    device = device or torch.device("cpu")
    seed_fn(seed)
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae_px": [],
        "val_chunk_mae_px": [],
        "seconds": [],
    }
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0.0
        for ids in rd.epoch_batches(len(train_res), batch_size, generator):
            batch = train_res.batch(ids.to(train_res.device))
            loss = loss_fn(
                model(batch["image"], batch["agent_pos"]), batch["actions"], batch["mask"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps = float(batch["mask"].sum())
            running += float(loss.detach()) * steps
            seen += steps
        val = evaluate_resident(model, val_res, batch_size, coord_scale)
        history["epoch"].append(epoch)
        history["train_loss"].append(running / seen)
        history["val_loss"].append(val["loss"])
        history["val_mae_px"].append(val["first_mae_px"])
        history["val_chunk_mae_px"].append(val["chunk_mae_px"])
        history["seconds"].append(round(time.perf_counter() - start, 2))
        log(
            f"epoch {epoch:3d}/{epochs}  train loss {running / seen:.5f}  "
            f"val loss {val['loss']:.5f}  val MAE first action {val['first_mae_px']:.2f} px  "
            f"whole chunk {val['chunk_mae_px']:.2f} px  ({history['seconds'][-1]:.0f}s)"
        )
    return history


# ---------------------------------------------------------------- the anchor run


def _show(label: str, c: dict) -> None:
    extra = (
        f", final first-action error differs by {c['final_val_mae_px_abs']:.3f} px"
        if "final_val_mae_px_abs" in c
        else ""
    )
    print(
        f"{label}: largest relative difference over {c['epochs']} epochs: "
        f"train loss {c['train_loss_max_rel']:.2e}, val loss {c['val_loss_max_rel']:.2e}{extra}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Resident training and its anchor run")
    parser.add_argument("--anchor", action="store_true")
    parser.add_argument("--reference", type=Path, default=REFERENCE_DEFAULT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args(argv)
    if not args.anchor:
        parser.print_help()
        return 0

    import torch

    if not torch.cuda.is_available():
        print("no CUDA device available: the anchor run needs the GPU")
        return 1
    import train_chunk

    device = torch.device("cuda")
    arrays, train_ds, val_ds = rd._load_data()
    train_res = rd.build_resident(arrays, train_ds, device)
    val_res = rd.build_resident(arrays, val_ds, device)

    def fresh_model():
        train_chunk.tb.seed_everything(0)
        return train_chunk.ChunkPolicyNet()

    quiet = lambda *_: None  # noqa: E731
    print(f"rerunning the original training loop ({args.epochs} epochs, training seed 0) ...")
    old = train_chunk.train(
        fresh_model(), train_ds, val_ds, epochs=args.epochs, device=device, seed=0, log=quiet
    )
    print("running the resident loop ...")
    new = train_resident(
        fresh_model(), train_res, val_res, epochs=args.epochs, device=device, seed=0, log=quiet
    )
    print(
        f"\ntime: original loop {old['seconds'][-1]:.1f} s, resident loop {new['seconds'][-1]:.1f} s "
        f"({old['seconds'][-1] / new['seconds'][-1]:.1f}x faster, validation included)"
    )

    result: dict = {"epochs": args.epochs, "rerun": old, "resident": new}
    comparison = compare_histories(new, old)
    result["resident_vs_rerun"] = comparison
    print()
    _show("resident vs rerun of the original loop", comparison)

    saved = None
    if args.reference.exists():
        saved = find_history(json.loads(args.reference.read_text()))
    if saved is None or len(saved["train_loss"]) != args.epochs:
        print(f"no usable saved history of {args.epochs} epochs in {args.reference}: no verdict")
    else:
        rerun_vs_saved = compare_histories(old, saved)
        resident_vs_saved = compare_histories(new, saved)
        _show("rerun of the original loop vs the saved history", rerun_vs_saved)
        _show("resident loop vs the saved history", resident_vs_saved)
        verdict = equivalence_verdict(
            rerun_vs_saved["val_loss_max_rel"], resident_vs_saved["val_loss_max_rel"]
        )
        result.update(
            rerun_vs_saved=rerun_vs_saved, resident_vs_saved=resident_vs_saved, verdict=verdict
        )
        print(
            f"verdict by the rule fixed in advance (resident <= {FACTOR:g} x max(rerun, {FLOOR:g})): "
            f"{verdict}"
        )
        epochs = [e for e in (1, 2, 5, 10, 20, args.epochs) if e <= args.epochs]
        print("\nvalidation loss at selected epochs:")
        print("epoch    saved     rerun  resident")
        for e in sorted(set(epochs)):
            print(
                f"{e:>5}  {saved['val_loss'][e - 1]:.5f}  {old['val_loss'][e - 1]:.5f}  "
                f"{new['val_loss'][e - 1]:.5f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
