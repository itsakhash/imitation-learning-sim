"""Action-chunking behavior cloning: predict the next CHUNK_LEN commanded targets at once.

The baseline (train_bc.py) predicts one commanded target per frame. Here the same convolutional
trunk predicts the targets for steps t .. t+CHUNK_LEN-1 from the frame at step t. At run time
the policy (chunk_policy.py) will execute the first N_EXEC of them before predicting again.

Design fixed before any result (see PROJECT_CONTEXT.md and the message that introduced this):
  - same trunk, same split, same optimizer as the baseline; only the last layer differs
  - near the end of a demonstration the chunk is padded by repeating the last action, and the
    loss ignores the padded positions (they carry no information about what the human did)
  - CHUNK_LEN = 16 and N_EXEC = 8, taken from memory of the Diffusion Policy PushT setup and
    not checked; they are not to be tuned on the evaluation seeds

The number comparable with the baseline is the validation mean absolute error of the FIRST
action of the chunk (history key "val_mae_px"). The error averaged over the whole chunk is
reported separately. Neither is a simulator result: a trained policy is evaluated in the
simulator separately, with evaluate.py.

Run:  python train_chunk.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import pusht_data
import train_bc as tb
from pusht_data import COORD_SCALE

CHUNK_LEN = 16
N_EXEC = 8


def make_trunk() -> nn.Sequential:
    """The baseline's convolutional trunk, layer for layer."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
        nn.ReLU(),  # 96 -> 48
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),  # 48 -> 24
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),  # 24 -> 12
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),  # 12 -> 6
        nn.Flatten(),
    )


class ChunkPolicyNet(nn.Module):
    """Image (3, 96, 96) in [0, 1] and agent position (2,) in [0, 1] -> chunk (K, 2)."""

    def __init__(self, chunk_len: int = CHUNK_LEN):
        super().__init__()
        if chunk_len < 1:
            raise ValueError("chunk_len must be at least 1")
        self.chunk_len = chunk_len
        self.conv = make_trunk()
        self.head = nn.Sequential(
            nn.Linear(64 * 6 * 6 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * chunk_len),
        )

    def forward(self, image, agent_pos):
        features = self.conv(image)
        out = self.head(torch.cat([features, agent_pos], dim=1))
        return out.view(-1, self.chunk_len, 2)


class ChunkDataset(Dataset):
    """One example per frame: (image, agent position) -> the next chunk_len actions.

    A chunk never reaches into the next episode. Where an episode ends before the chunk is
    full, the missing actions repeat the episode's last action and mask is 0 there.
    Positions and actions are divided by 512, as in PushTDataset.
    """

    def __init__(self, arrays, episode_ids=None, chunk_len: int = CHUNK_LEN):
        if chunk_len < 1:
            raise ValueError("chunk_len must be at least 1")
        self.arrays = arrays
        self.chunk_len = chunk_len
        self.indices = pusht_data.PushTDataset(arrays, episode_ids).indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        j = int(self.indices[i])
        a = self.arrays
        frame = int(a.frame_index[j])
        length = int(a.episode_lengths[int(a.episode_index[j])])
        valid = min(self.chunk_len, length - frame)  # always at least 1
        actions = np.empty((self.chunk_len, 2), dtype=np.float32)
        actions[:valid] = a.actions[j : j + valid]
        actions[valid:] = a.actions[j + valid - 1]
        mask = np.zeros(self.chunk_len, dtype=np.float32)
        mask[:valid] = 1.0
        return {
            "image": torch.from_numpy(a.images[j]).permute(2, 0, 1).float() / 255.0,
            "agent_pos": torch.from_numpy(a.states[j]) / COORD_SCALE,
            "actions": torch.from_numpy(actions) / COORD_SCALE,
            "mask": torch.from_numpy(mask),
            "episode_index": int(a.episode_index[j]),
            "frame_index": frame,
        }


def chunk_loss(pred, target, mask):
    """Mean squared error over the valid (unpadded) positions of the chunks only.
    pred and target have shape (B, K, 2); mask has shape (B, K)."""
    weight = mask.unsqueeze(-1)
    denominator = (weight.sum() * pred.shape[-1]).clamp_min(1.0)
    return (((pred - target) ** 2) * weight).sum() / denominator


def _unpack(batch, device):
    return (
        batch["image"].to(device),
        batch["agent_pos"].to(device),
        batch["actions"].to(device),
        batch["mask"].to(device),
    )


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    """Mean loss over valid chunk positions, the error of the first action in pixels, and the
    error averaged over all valid chunk positions in pixels."""
    model.eval()
    squared = chunk_abs = first_abs = valid = frames = 0.0
    for batch in loader:
        image, pos, target, mask = _unpack(batch, device)
        error = model(image, pos) - target
        weight = mask.unsqueeze(-1)
        squared += float(((error**2) * weight).sum())
        chunk_abs += float((error.abs() * weight).sum())
        first_abs += float(error[:, 0].abs().sum())
        valid += float(mask.sum())
        frames += len(target)
    if frames == 0:
        raise ValueError("the loader produced no examples")
    return {
        "loss": squared / (valid * 2),
        "first_mae_px": first_abs / (frames * 2) * COORD_SCALE,
        "chunk_mae_px": chunk_abs / (valid * 2) * COORD_SCALE,
    }


def train(
    model,
    train_ds,
    val_ds,
    *,
    epochs: int,
    batch_size: int = 256,
    lr: float = 1e-3,
    device=None,
    seed: int = 0,
    log=print,
) -> dict:
    """Train with Adam and return the per-epoch history."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if lr <= 0:
        raise ValueError("lr must be positive")
    device = device or torch.device("cpu")
    tb.seed_everything(seed)
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
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
        for batch in train_loader:
            image, pos, target, mask = _unpack(batch, device)
            loss = chunk_loss(model(image, pos), target, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps = float(mask.sum())
            running += float(loss.detach()) * steps
            seen += steps
        val = evaluate_loader(model, val_loader, device)
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


def load_chunk_checkpoint(path, device="cpu"):
    """Return (chunk model in eval mode, the full checkpoint dict). Stays in torch's safe
    weights-only mode, like train_bc.load_checkpoint."""
    with torch.serialization.safe_globals(tb._version_type_allowlist()):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ChunkPolicyNet(ckpt["config"]["chunk_len"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def main(argv=None, *, arrays_loader=pusht_data.load_arrays) -> int:
    parser = argparse.ArgumentParser(description="Train the action-chunking baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--chunk-len", type=int, default=CHUNK_LEN)
    parser.add_argument("--n-exec", type=int, default=N_EXEC)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/chunk_cnn.pt")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/chunk_training.json")
    )
    parser.add_argument(
        "--curve", type=Path, default=Path("reports/chunk_loss_curve.png")
    )
    args = parser.parse_args(argv)
    if args.chunk_len < 1:
        raise ValueError("chunk_len must be at least 1")
    if not 1 <= args.n_exec <= args.chunk_len:
        raise ValueError("n_exec must be between 1 and chunk_len")

    device = tb.pick_device(args.device)
    arrays = arrays_loader()
    train_ids, val_ids = pusht_data.split_episode_ids(
        arrays.n_episodes, args.val_fraction, args.split_seed
    )
    available = len(train_ids)
    if args.train_episodes is not None:
        train_ids = tb.subsample_episodes(
            train_ids, args.train_episodes, args.split_seed
        )
    train_ds = ChunkDataset(arrays, train_ids, args.chunk_len)
    val_ds = ChunkDataset(arrays, val_ids, args.chunk_len)
    references = tb.reference_maes(arrays, train_ids, val_ids)

    tb.seed_everything(args.seed)
    model = ChunkPolicyNet(args.chunk_len)
    n_params = sum(p.numel() for p in model.parameters())
    where = (
        f"cuda ({torch.cuda.get_device_name(0)})"
        if device.type == "cuda"
        else device.type
    )
    print(
        f"device: {where}   parameters: {n_params:,}   chunk length: {args.chunk_len}"
    )
    print(
        f"training on {len(train_ds):,} frames from {len(train_ids)} episodes "
        f"(of {available} available); validating on {len(val_ds):,} frames "
        f"from {len(val_ids)} episodes"
    )
    print(
        "reference predictors for the first action on the validation episodes: "
        f"copy agent position {references['copy_agent_position_mae_px']:.2f} px, "
        f"predict mean action {references['mean_action_mae_px']:.2f} px"
    )

    history = train(
        model,
        train_ds,
        val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )
    final = {
        "val_loss": history["val_loss"][-1],
        "val_mae_px": history["val_mae_px"][-1],
        "val_chunk_mae_px": history["val_chunk_mae_px"][-1],
    }
    config = {
        "architecture": "ChunkPolicyNet",
        "loss": "mse over the valid chunk positions",
        "optimizer": "adam",
        "n_parameters": n_params,
        "chunk_len": args.chunk_len,
        "n_exec": args.n_exec,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "train_episodes": len(train_ids),
        "available_train_episodes": available,
        "device": str(device),
        "torch": str(torch.__version__),  # a plain str, so the safe loader accepts it
    }
    split = {"train_episode_ids": train_ids, "val_episode_ids": val_ids}
    tb.save_checkpoint(args.checkpoint, model, config, split, history, references)
    report = {
        "config": config,
        "dataset_revision": pusht_data.DATASET_REVISION,
        "split": split,
        "references": references,
        "history": history,
        "final": final,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    tb.save_loss_curve(history, args.curve, references)

    print()
    print(
        f"final validation MAE of the first action: {final['val_mae_px']:.2f} px  "
        f"(copy agent position: {references['copy_agent_position_mae_px']:.2f} px, "
        f"predict mean action: {references['mean_action_mae_px']:.2f} px); "
        f"averaged over the whole chunk: {final['val_chunk_mae_px']:.2f} px"
    )
    print(
        "these are offline numbers on held-out demonstration episodes, not simulator results"
    )
    print(f"saved {args.checkpoint}, {args.report}, {args.curve}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
