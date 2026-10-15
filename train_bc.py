"""Behavior-cloning baseline: a small CNN from (image, agent position) to the next action.

Each training example is one frame of a human demonstration: the 96x96 image and the
agent's position go in, the action the human chose at that frame comes out. In PushT an
action is a target position for the agent (a PD controller steers the agent towards it),
so the network predicts a position. Positions are scaled to [0, 1] by dividing by 512.

The loss is the mean squared error on those scaled positions. The number to read is the
validation mean absolute error in pixels on episodes the network never trained on, next
to two trivial reference predictors. Offline error is NOT a simulator result: a trained
policy is evaluated in the simulator separately, with evaluate.py.

--train-episodes N keeps N of the training episodes. Which N is chosen by --subset-seed
(default: the split seed), independently of the training seed (--seed), so the choice of
episodes and the randomness of training can be varied separately. For a fixed subset seed the
subsets are nested: the 20 episodes chosen are among the 50 chosen.

Run:  python train_bc.py --epochs 30
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
from torch.utils.data import DataLoader

import pusht_data
from pusht_data import COORD_SCALE, PushTDataset

# How the reference predictors are drawn on the right-hand loss-curve panel. Each one
# must look different from the others and from the network's own line (tab:blue).
REFERENCE_STYLES = {
    "copy_agent_position_mae_px": {
        "color": "tab:orange",
        "linestyle": "--",
        "label": "copy agent position",
    },
    "mean_action_mae_px": {
        "color": "tab:green",
        "linestyle": ":",
        "label": "predict mean action",
    },
}


class BCPolicyNet(nn.Module):
    """Image (3, 96, 96) in [0, 1] and agent position (2,) in [0, 1] -> action (2,)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
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
        self.head = nn.Sequential(
            nn.Linear(64 * 6 * 6 + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, image, agent_pos):
        features = self.conv(image)
        return self.head(torch.cat([features, agent_pos], dim=1))


def bc_loss(pred, target):
    """Mean squared error between predicted and demonstrated actions (both in [0, 1])."""
    return torch.nn.functional.mse_loss(pred, target)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _unpack(batch, device):
    return (
        batch["image"].to(device),
        batch["agent_pos"].to(device),
        batch["action"].to(device),
    )


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    """Mean loss and mean absolute error (in pixels) over everything in the loader."""
    model.eval()
    total_loss, total_abs, n = 0.0, 0.0, 0
    for batch in loader:
        image, pos, target = _unpack(batch, device)
        pred = model(image, pos)
        total_loss += float(bc_loss(pred, target)) * len(target)
        total_abs += float((pred - target).abs().sum())
        n += len(target)
    if n == 0:
        raise ValueError("the loader produced no examples")
    return {"loss": total_loss / n, "mae_px": total_abs / (n * 2) * COORD_SCALE}


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
    seed_everything(seed)
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
        "seconds": [],
    }
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            image, pos, target = _unpack(batch, device)
            loss = bc_loss(model(image, pos), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(target)
            seen += len(target)
        val = evaluate_loader(model, val_loader, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(running / seen)
        history["val_loss"].append(val["loss"])
        history["val_mae_px"].append(val["mae_px"])
        history["seconds"].append(round(time.perf_counter() - start, 2))
        log(
            f"epoch {epoch:3d}/{epochs}  train loss {running / seen:.5f}  "
            f"val loss {val['loss']:.5f}  val MAE {val['mae_px']:.2f} px  "
            f"({history['seconds'][-1]:.0f}s)"
        )
    return history


def reference_maes(arrays, train_ids, val_ids) -> dict:
    """Validation MAE (pixels) of two trivial predictors: copy the agent's position, and
    always predict the mean training action. A learned policy has to beat both."""
    train_idx = np.flatnonzero(np.isin(arrays.episode_index, list(train_ids)))
    val_idx = np.flatnonzero(np.isin(arrays.episode_index, list(val_ids)))
    actions = arrays.actions.astype(np.float64)
    states = arrays.states.astype(np.float64)
    mean_action = actions[train_idx].mean(axis=0)
    return {
        "copy_agent_position_mae_px": float(
            np.abs(actions[val_idx] - states[val_idx]).mean()
        ),
        "mean_action_mae_px": float(np.abs(actions[val_idx] - mean_action).mean()),
    }


def subsample_episodes(ids, n: int, seed: int = 0) -> list:
    """A seeded random subset of n episode ids (for the number-of-demonstrations study)."""
    ids = list(ids)
    if n < 1 or n > len(ids):
        raise ValueError(f"n must be between 1 and {len(ids)}, got {n}")
    order = np.random.default_rng(seed).permutation(len(ids))
    return sorted(ids[int(i)] for i in order[:n])


def save_loss_curve(history: dict, path, references: dict | None = None) -> None:
    """Training and validation curves; the right panel shows the reference predictors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(epochs, history["train_loss"], label="training")
    left.plot(epochs, history["val_loss"], label="validation")
    left.set_xlabel("epoch")
    left.set_ylabel("mean squared error (positions scaled to [0, 1])")
    if min(history["train_loss"] + history["val_loss"]) > 0:
        left.set_yscale("log")
    left.legend()

    right.plot(
        epochs, history["val_mae_px"], label="network (validation)", color="tab:blue"
    )
    for key, value in (references or {}).items():
        style = REFERENCE_STYLES.get(
            key, {"color": "tab:gray", "linestyle": "-.", "label": key}
        )
        right.axhline(value, **style)
    right.set_xlabel("epoch")
    right.set_ylabel("mean absolute error (pixels)")
    if min(history["val_mae_px"]) > 0:
        right.set_yscale("log")
    right.legend()

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_checkpoint(path, model, config, split, history, references) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "split": split,
            "history": history,
            "references": references,
            "dataset_revision": pusht_data.DATASET_REVISION,
        },
        path,
    )


def _version_type_allowlist() -> list:
    """The class of torch.__version__ (TorchVersion). Checkpoints written before the
    version was stored as a plain string contain one, and torch's safe loader refuses
    unknown classes unless they are listed. This lists that one class and nothing else.
    """
    version_type = type(torch.__version__)
    return [] if version_type is str else [version_type]


def load_checkpoint(path, device="cpu"):
    """Return (model in eval mode, the full checkpoint dict). Stays in torch's safe
    weights-only mode: only tensors, plain containers and the version class are accepted.
    """
    with torch.serialization.safe_globals(_version_type_allowlist()):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    model = BCPolicyNet()
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def main(argv=None, *, arrays_loader=pusht_data.load_arrays) -> int:
    parser = argparse.ArgumentParser(description="Train the behavior-cloning baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        help="seed for choosing which training episodes --train-episodes keeps "
        "(default: the split seed)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/bc_cnn.pt")
    )
    parser.add_argument("--report", type=Path, default=Path("reports/bc_training.json"))
    parser.add_argument("--curve", type=Path, default=Path("reports/bc_loss_curve.png"))
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    arrays = arrays_loader()
    train_ids, val_ids = pusht_data.split_episode_ids(
        arrays.n_episodes, args.val_fraction, args.split_seed
    )
    available = len(train_ids)
    subset_seed = args.split_seed if args.subset_seed is None else args.subset_seed
    if args.train_episodes is not None:
        train_ids = subsample_episodes(train_ids, args.train_episodes, subset_seed)
    train_ds = PushTDataset(arrays, train_ids)
    val_ds = PushTDataset(arrays, val_ids)
    references = reference_maes(arrays, train_ids, val_ids)

    seed_everything(args.seed)
    model = BCPolicyNet()
    n_params = sum(p.numel() for p in model.parameters())
    where = (
        f"cuda ({torch.cuda.get_device_name(0)})"
        if device.type == "cuda"
        else device.type
    )
    print(f"device: {where}   parameters: {n_params:,}")
    print(
        f"training on {len(train_ds):,} frames from {len(train_ids)} episodes "
        f"(of {available} available); validating on {len(val_ds):,} frames "
        f"from {len(val_ids)} episodes"
    )
    print(
        "reference predictors on the validation episodes: "
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
    }
    config = {
        "architecture": "BCPolicyNet",
        "loss": "mse",
        "optimizer": "adam",
        "n_parameters": n_params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "train_episodes": len(train_ids),
        "available_train_episodes": available,
        "subset_seed": subset_seed if args.train_episodes is not None else None,
        "device": str(device),
        "torch": str(torch.__version__),  # a plain str, so the safe loader accepts it
    }
    split = {"train_episode_ids": train_ids, "val_episode_ids": val_ids}
    save_checkpoint(args.checkpoint, model, config, split, history, references)
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
    save_loss_curve(history, args.curve, references)

    print()
    print(
        f"final validation MAE: {final['val_mae_px']:.2f} px  "
        f"(copy agent position: {references['copy_agent_position_mae_px']:.2f} px, "
        f"predict mean action: {references['mean_action_mae_px']:.2f} px)"
    )
    print(
        "this is an offline number on held-out demonstration episodes, not a simulator result"
    )
    print(f"saved {args.checkpoint}, {args.report}, {args.curve}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
