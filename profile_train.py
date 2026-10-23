"""Time the REAL training loop: where does a training step spend its time?

Uses the project's own ChunkDataset, ChunkPolicyNet, chunk_loss and train() on the real data
(165 training episodes, batch 256). Measures:
  1. batch building alone (DataLoader, no GPU work) with 0, 4 and 8 loader workers;
  2. the model step alone (forward, backward, Adam) on one fixed batch already on the GPU;
  3. what train() really costs per epoch (81 steps plus one validation pass);
  4. an alternative: all training examples held on the GPU, batches made by indexing, with a
     check that its batches equal what ChunkDataset produces.

Nothing is trained for real and no checkpoint or report of the study is touched; the
output goes to reports/profile_train.json. Loading the dataset takes a little while.
Close games and other heavy programs first. Takes a couple of minutes. Run:
    python profile_train.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from profile_speed import project_minutes, summarize_ms

REPORT_DEFAULT = Path("reports/profile_train.json")
BATCH = 256
BUDGET_STEPS = 30000


# ---------------------------------------------------------------- pure helpers (tested)


def time_iterator(make_iterator, warmup: int, measure: int, clock=time.perf_counter) -> list:
    """Seconds taken by each of `measure` next() calls after `warmup` untimed ones."""
    if warmup < 0 or measure < 1:
        raise ValueError("warmup must be >= 0 and measure >= 1")
    iterator = make_iterator()
    times = []
    for i in range(warmup + measure):
        t0 = clock()
        next(iterator)
        elapsed = clock() - t0
        if i >= warmup:
            times.append(elapsed)
    return times


def epoch_seconds(cumulative) -> list:
    """Per-epoch durations from the cumulative seconds stored in a training history."""
    if len(cumulative) == 0:
        raise ValueError("empty history")
    out = [cumulative[0]]
    for before, after in zip(cumulative, cumulative[1:]):
        out.append(after - before)
    return out


def steps_per_epoch(n_examples: int, batch_size: int) -> int:
    if n_examples < 1 or batch_size < 1:
        raise ValueError("n_examples and batch_size must be positive")
    return -(-n_examples // batch_size)


def ratio(slower_ms: float, faster_ms: float) -> float:
    if faster_ms <= 0:
        raise ValueError("faster_ms must be positive")
    return slower_ms / faster_ms


# ---------------------------------------------------------------- real data and code


def load_data():
    """The same arrays, split and datasets as train_chunk.main with its defaults."""
    import pusht_data
    from train_chunk import ChunkDataset

    arrays = pusht_data.load_arrays()
    train_ids, val_ids = pusht_data.split_episode_ids(arrays.n_episodes, 0.2, 0)
    return arrays, ChunkDataset(arrays, train_ids), ChunkDataset(arrays, val_ids)


def loader_times(train_ds, num_workers: int, measure: int, warmup: int = 10) -> list:
    import torch
    from torch.utils.data import DataLoader

    def make():
        generator = torch.Generator().manual_seed(0)
        return iter(
            DataLoader(
                train_ds,
                batch_size=BATCH,
                shuffle=True,
                generator=generator,
                num_workers=num_workers,
            )
        )

    return time_iterator(make, warmup, measure)


def _fresh_model_and_optimizer(device):
    import torch
    from train_chunk import ChunkPolicyNet

    torch.manual_seed(0)
    model = ChunkPolicyNet().to(device)
    return model, torch.optim.Adam(model.parameters(), lr=1e-3)


def model_step_times(train_ds, steps: int, warmup: int) -> list:
    import torch
    from torch.utils.data import default_collate
    from train_chunk import chunk_loss

    device = torch.device("cuda")
    model, optimizer = _fresh_model_and_optimizer(device)
    batch = default_collate([train_ds[i] for i in range(BATCH)])
    image, pos, target, mask = (
        batch[k].to(device) for k in ("image", "agent_pos", "actions", "mask")
    )
    times = []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = chunk_loss(model(image, pos), target, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return times


def real_epoch_seconds(train_ds, val_ds, epochs: int) -> list:
    import torch
    from train_chunk import ChunkPolicyNet, train

    torch.manual_seed(0)
    model = ChunkPolicyNet()
    history = train(
        model,
        train_ds,
        val_ds,
        epochs=epochs,
        device=torch.device("cuda"),
        seed=0,
        log=lambda *_: None,
    )
    return epoch_seconds(history["seconds"])


def build_resident(arrays, ds, device):
    """All examples of `ds` on the GPU: uint8 images, positions, action chunks, masks."""
    import numpy as np
    import torch
    from pusht_data import COORD_SCALE

    t0 = time.perf_counter()
    idx = np.asarray(ds.indices)
    images = torch.from_numpy(arrays.images[idx]).permute(0, 3, 1, 2).contiguous()
    agent_pos = torch.from_numpy(arrays.states[idx]) / COORD_SCALE
    actions, masks = [], []
    for i in range(len(ds)):
        item = ds[i]
        actions.append(item["actions"])
        masks.append(item["mask"])
    resident = {
        "images": images.to(device),
        "agent_pos": agent_pos.to(device),
        "actions": torch.stack(actions).to(device),
        "mask": torch.stack(masks).to(device),
    }
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    megabytes = sum(t.numel() * t.element_size() for t in resident.values()) / 1e6
    return resident, seconds, megabytes


def resident_mismatches(ds, resident, positions) -> list:
    """Keys whose values differ between ChunkDataset batches and the GPU-resident ones."""
    import torch
    from torch.utils.data import default_collate

    batch = default_collate([ds[i] for i in positions])
    ids = torch.as_tensor(list(positions), device=resident["images"].device)
    got = {
        "image": (resident["images"][ids].float() / 255.0).cpu(),
        "agent_pos": resident["agent_pos"][ids].cpu(),
        "actions": resident["actions"][ids].cpu(),
        "mask": resident["mask"][ids].cpu(),
    }
    return [k for k, v in got.items() if not torch.equal(v, batch[k])]


def resident_step_times(resident, steps: int, warmup: int) -> list:
    import torch
    from train_chunk import chunk_loss

    device = resident["images"].device
    model, optimizer = _fresh_model_and_optimizer(device)
    n = resident["images"].shape[0]
    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=generator).to(device)
    times = []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start = (i * BATCH) % (n - BATCH)
        ids = perm[start : start + BATCH]
        image = resident["images"][ids].float() / 255.0
        loss = chunk_loss(
            model(image, resident["agent_pos"][ids]),
            resident["actions"][ids],
            resident["mask"][ids],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return times


# ---------------------------------------------------------------- report


def line(label: str, s: dict) -> str:
    return (
        f"{label}: median {s['median_ms']:.2f} ms, mean {s['mean_ms']:.2f} ms, "
        f"p95 {s['p95_ms']:.2f} ms (n={s['n']})"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Time the real training loop")
    parser.add_argument("--batches", type=int, default=40, help="batches timed per loader setting")
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--model-steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2, help="epochs of the real train() to time")
    parser.add_argument("--out", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    arrays, train_ds, val_ds = load_data()
    n_train = len(train_ds)
    steps = steps_per_epoch(n_train, BATCH)
    print(
        f"loaded the data in {time.perf_counter() - t0:.1f} s: {n_train:,} training frames, "
        f"{len(val_ds):,} validation frames, {steps} steps per epoch at batch {BATCH}"
    )
    report: dict = {"train_frames": n_train, "steps_per_epoch": steps}

    # loaders first, before CUDA is touched, so forked workers never see a CUDA context
    print("\n== batch building alone (no GPU work), per batch of 256 ==")
    report["loader"] = {}
    for workers in [0] + list(args.worker_counts):
        s = summarize_ms(loader_times(train_ds, workers, args.batches))
        report["loader"][str(workers)] = s
        print(line(f"DataLoader num_workers={workers}", s))

    import torch

    if not torch.cuda.is_available():
        print("\nno CUDA device available: GPU parts skipped")
    else:
        device = torch.device("cuda")
        print(f"\n== GPU: {torch.cuda.get_device_name(0)} ==")
        model_s = summarize_ms(model_step_times(train_ds, args.model_steps, args.warmup))
        report["model_step"] = model_s
        print(line("model step alone, batch already on the GPU", model_s))

        secs = real_epoch_seconds(train_ds, val_ds, args.epochs)
        report["train_epoch_seconds"] = secs
        steady = secs[-1]
        per_step_ms = steady / steps * 1000.0
        full_minutes = steady * (BUDGET_STEPS / steps) / 60.0
        print(
            "real train(), seconds per epoch (the first includes CUDA start-up): "
            + ", ".join(f"{x:.2f}" for x in secs)
        )
        print(
            f"last epoch = {per_step_ms:.1f} ms per step including validation; "
            f"{BUDGET_STEPS:,} steps would take about {full_minutes:.1f} min at that rate"
        )
        report["real_ms_per_step_with_validation"] = per_step_ms
        report["projected_minutes_30000_steps_real_loop"] = full_minutes

        resident, build_s, megabytes = build_resident(arrays, train_ds, device)
        positions = [p for p in (0, 1, 2, 100, 5000, n_train - 1) if p < n_train]
        bad = resident_mismatches(train_ds, resident, positions)
        res_s = summarize_ms(resident_step_times(resident, args.model_steps, args.warmup))
        report["resident"] = {
            "build_seconds": build_s,
            "megabytes": megabytes,
            "mismatching_keys": bad,
            "step": res_s,
        }
        print(f"\n== GPU-resident alternative: built in {build_s:.1f} s, {megabytes:.0f} MB on the GPU ==")
        print(
            "batches equal ChunkDataset's for the "
            f"{len(positions)} checked examples: {'yes' if not bad else 'NO, differs in ' + ', '.join(bad)}"
        )
        print(line("batch assembly + model step", res_s))
        print(
            f"{BUDGET_STEPS:,} steps at that rate (no validation): "
            f"{project_minutes(res_s['median_ms'], BUDGET_STEPS):.1f} min"
        )
        print(
            f"real loop step / resident step = "
            f"{ratio(per_step_ms, res_s['median_ms']):.1f}x (includes validation in the real loop)"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
