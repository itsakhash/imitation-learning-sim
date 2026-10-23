"""Training of the arms of the diffusion study on GPU-resident data, with an exponential moving
average (EMA) of the weights. See diffusion_preregistration.md.

Arms (the letters of the pre-registration):
  D  diffusion policy (diffusion_model.DiffusionPolicyNet, epsilon-prediction loss), 370 epochs, EMA
  C  chunked control (train_chunk.ChunkPolicyNet, chunk_loss), 370 epochs, EMA
  B  baseline control (train_bc.BCPolicyNet, bc_loss on chunk length 1 data), 370 epochs, EMA
  P  path check: the original chunked recipe on the resident path, 30 epochs, no EMA, lr 1e-3

The loop is resident_train.train_resident with three changes: the loss and the validation come
from an "arm" object (so one loop serves all arms), an EMA of the weights is updated after every
optimiser step, and validation is done with the EMA weights every `val_every` epochs and at the
last epoch. Shuffling uses resident_data.epoch_batches with a generator seeded by the training
seed, exactly as resident_train does. For D the step indices and noise of the loss are drawn on
the CPU from a second generator (seed + TRAIN_NOISE_OFFSET), so a run is repeatable up to GPU
arithmetic. With ema_decay=None and val_every=1 the loop is train_resident (tested).

EMA: shadow = decay * shadow + (1 - decay) * weights after every step, decay 0.999 (the
pre-registered, untuned value), no warm-up; the shadow starts as the initial weights. Floating
point tensors are averaged; any other state (none in these networks) is copied.

Validation (done with the EMA weights):
  loss        C, B: the masked MSE of the training loss; D: the denoising loss with step indices
              and noise from a generator seeded with VAL_NOISE_SEED (the same every time)
  val_mae_px  mean absolute error of the first action against the demonstration, in pixels
              (D: from one seeded DDIM sample per frame, so it includes the sampling spread)
  val_chunk_mae_px  the same over the valid positions of the whole chunk
history lists: epoch and train_loss have one entry per epoch; val_epoch, val_loss, val_mae_px and
val_chunk_mae_px have one entry per validation.

Checkpoints are written with train_bc.save_checkpoint in the existing format (model_state holds
the EMA weights) so evaluate_chunk.py and evaluate_bc.py read C, B and P unchanged; D is read by
evaluate_diffusion.py. A checkpoint is never overwritten unless --overwrite is given.

Run (nothing here is started automatically; one network of D takes minutes on a GPU):
    python arm_train.py --arm D --seeds 0 --lr 3e-4
    python arm_train.py --arm C --seeds 0-9 --lr 1e-3          # one process, data built once
    python arm_train.py --arm P --seeds 0-9
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import diffusion_core as dc
import resident_data as rd

EMA_DECAY = 0.999
EPOCHS = 370
P_EPOCHS = 30
VAL_EVERY = 10
VAL_NOISE_SEED = 777
TRAIN_NOISE_OFFSET = 1_000_003
VAL_FRACTION = 0.2
SPLIT_SEED = 0


# ---------------------------------------------------------------- EMA


class EMA:
    """Exponential moving average of a model's state."""

    def __init__(self, model, decay: float = EMA_DECAY):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model) -> None:
        import torch

        with torch.no_grad():
            for key, value in model.state_dict().items():
                shadow = self.shadow[key]
                if shadow.dtype.is_floating_point:
                    shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
                else:
                    shadow.copy_(value)
        self.updates += 1

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------- the arms


class ChunkArm:
    """Arm C (and the path check P): ChunkPolicyNet with the masked MSE."""

    code = "C"
    chunk_len = 16

    def make_model(self):
        from train_chunk import ChunkPolicyNet

        return ChunkPolicyNet(self.chunk_len)

    def loss(self, model, batch, rng):
        from train_chunk import chunk_loss

        return chunk_loss(model(batch["image"], batch["agent_pos"]), batch["actions"], batch["mask"])

    def validate(self, model, res, batch_size, coord_scale):
        from resident_train import evaluate_resident

        out = evaluate_resident(model, res, batch_size, coord_scale)
        return {
            "loss": out["loss"],
            "val_mae_px": out["first_mae_px"],
            "val_chunk_mae_px": out["chunk_mae_px"],
        }


class BaselineArm:
    """Arm B: BCPolicyNet on chunk length 1 data. The target is the single action
    batch["actions"][:, 0], which equals PushTDataset's action (tested)."""

    code = "B"
    chunk_len = 1

    def make_model(self):
        from train_bc import BCPolicyNet

        return BCPolicyNet()

    def loss(self, model, batch, rng):
        from train_bc import bc_loss

        return bc_loss(model(batch["image"], batch["agent_pos"]), batch["actions"][:, 0])

    def validate(self, model, res, batch_size, coord_scale):
        import torch

        model.eval()
        squared = absolute = 0.0
        with torch.no_grad():
            for start, stop in rd.batch_bounds(len(res), batch_size):
                batch = res.batch(torch.arange(start, stop, device=res.device))
                error = model(batch["image"], batch["agent_pos"]) - batch["actions"][:, 0]
                squared += float((error**2).sum())
                absolute += float(error.abs().sum())
        values = len(res) * 2
        mae = absolute / values * coord_scale
        return {"loss": squared / values, "val_mae_px": mae, "val_chunk_mae_px": mae}


class DiffusionArm:
    """Arm D: DiffusionPolicyNet with the epsilon-prediction loss."""

    code = "D"
    chunk_len = 16

    def __init__(self):
        self._alpha_bars = {}

    def make_model(self):
        from diffusion_model import DiffusionPolicyNet

        return DiffusionPolicyNet(self.chunk_len)

    def alpha_bars(self, device):
        import torch

        key = str(device)
        if key not in self._alpha_bars:
            self._alpha_bars[key] = torch.as_tensor(dc.cosine_alpha_bars(), dtype=torch.float32, device=device)
        return self._alpha_bars[key]

    def loss(self, model, batch, rng):
        x0 = dc.to_diffusion_scale(batch["actions"])
        cond = model.encode(batch["image"], batch["agent_pos"])
        return dc.diffusion_loss(
            lambda x_t, t: model.denoise(x_t, t, cond),
            x0,
            batch["mask"],
            self.alpha_bars(x0.device),
            generator=rng,
        )

    def validate(self, model, res, batch_size, coord_scale):
        import torch

        import diffusion_model as dm

        model.eval()
        ab = self.alpha_bars(res.device)
        generator = torch.Generator().manual_seed(VAL_NOISE_SEED)
        squared = chunk_abs = first_abs = valid = frames = 0.0
        with torch.no_grad():
            for start, stop in rd.batch_bounds(len(res), batch_size):
                batch = res.batch(torch.arange(start, stop, device=res.device))
                x0 = dc.to_diffusion_scale(batch["actions"])
                cond = model.encode(batch["image"], batch["agent_pos"])
                loss = dc.diffusion_loss(
                    lambda x_t, t: model.denoise(x_t, t, cond), x0, batch["mask"], ab, generator=generator
                )
                n_valid = float(batch["mask"].sum())
                squared += float(loss) * n_valid * x0.shape[-1]  # undo the loss's own normalisation
                noise = torch.randn(tuple(x0.shape), generator=generator).to(res.device)
                sample = dm.sample_chunk(model, batch["image"], batch["agent_pos"], noise, ab)
                error = dc.from_diffusion_scale(sample) - batch["actions"]
                weight = batch["mask"].unsqueeze(-1)
                chunk_abs += float((error.abs() * weight).sum())
                first_abs += float(error[:, 0].abs().sum())
                valid += n_valid
                frames += len(batch["actions"])
        return {
            "loss": squared / (valid * x0.shape[-1]),
            "val_mae_px": first_abs / (frames * 2) * coord_scale,
            "val_chunk_mae_px": chunk_abs / (valid * 2) * coord_scale,
        }


ARM_SETTINGS = {
    "D": {"arm": DiffusionArm, "epochs": EPOCHS, "ema": EMA_DECAY, "val_every": VAL_EVERY, "prefix": "diff_cnn"},
    "C": {"arm": ChunkArm, "epochs": EPOCHS, "ema": EMA_DECAY, "val_every": VAL_EVERY, "prefix": "chunk_long"},
    "B": {"arm": BaselineArm, "epochs": EPOCHS, "ema": EMA_DECAY, "val_every": VAL_EVERY, "prefix": "bc_long"},
    "P": {"arm": ChunkArm, "epochs": P_EPOCHS, "ema": None, "val_every": 1, "prefix": "chunk_res30"},
}


# ---------------------------------------------------------------- pure helpers (tested)


def default_name(arm: str, seed: int) -> str:
    """diff_cnn, chunk_long, bc_long, chunk_res30 for seed 0; with _s{seed} appended otherwise
    (the project's convention)."""
    if arm not in ARM_SETTINGS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARM_SETTINGS)}")
    prefix = ARM_SETTINGS[arm]["prefix"]
    return prefix if seed == 0 else f"{prefix}_s{seed}"


def parse_seeds(text: str) -> list:
    """'0-9' -> [0..9]; '3' -> [3]; '0,2,5-6' -> [0, 2, 5, 6]. Order kept, no repeats allowed."""
    seeds = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            raise ValueError("empty seed entry")
        if "-" in part:
            low, _, high = part.partition("-")
            low, high = int(low), int(high)
            if high < low:
                raise ValueError(f"bad seed range {part!r}")
            seeds.extend(range(low, high + 1))
        else:
            seeds.append(int(part))
    if len(set(seeds)) != len(seeds):
        raise ValueError("a seed is listed twice")
    if any(s < 0 for s in seeds):
        raise ValueError("seeds must be non-negative")
    return seeds


# ---------------------------------------------------------------- the loop


def train_arm(
    arm,
    model,
    train_res,
    val_res,
    *,
    epochs: int,
    lr: float,
    batch_size: int = 256,
    device=None,
    seed: int = 0,
    ema_decay=EMA_DECAY,
    val_every: int = VAL_EVERY,
    log=print,
    seed_fn=None,
    coord_scale=None,
):
    """Train `model` and return (history, averaged_model): averaged_model holds the EMA weights
    (a copy of the final weights if ema_decay is None) and is what gets evaluated and saved."""
    import torch

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if val_every < 1:
        raise ValueError("val_every must be at least 1")
    if seed_fn is None:
        import train_bc

        seed_fn = train_bc.seed_everything
    if coord_scale is None:
        from pusht_data import COORD_SCALE as coord_scale
    device = device or torch.device("cpu")
    seed_fn(seed)
    model.to(device)
    ema = EMA(model, ema_decay) if ema_decay is not None else None
    averaged = copy.deepcopy(model)
    shuffle = torch.Generator().manual_seed(seed)
    noise = torch.Generator().manual_seed(seed + TRAIN_NOISE_OFFSET)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_epoch": [],
        "val_loss": [],
        "val_mae_px": [],
        "val_chunk_mae_px": [],
        "seconds": [],
    }
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0.0
        for ids in rd.epoch_batches(len(train_res), batch_size, shuffle):
            batch = train_res.batch(ids.to(train_res.device))
            loss = arm.loss(model, batch, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            steps = float(batch["mask"].sum())
            running += float(loss.detach()) * steps
            seen += steps
        if not math.isfinite(running):
            raise FloatingPointError(f"training loss is not finite in epoch {epoch}")
        history["epoch"].append(epoch)
        history["train_loss"].append(running / seen)
        history["seconds"].append(round(time.perf_counter() - start, 2))
        if epoch % val_every == 0 or epoch == epochs:
            if ema is not None:
                ema.copy_to(averaged)
            else:
                averaged.load_state_dict(model.state_dict())
            val = arm.validate(averaged, val_res, batch_size, coord_scale)
            history["val_epoch"].append(epoch)
            history["val_loss"].append(val["loss"])
            history["val_mae_px"].append(val["val_mae_px"])
            history["val_chunk_mae_px"].append(val["val_chunk_mae_px"])
            log(
                f"epoch {epoch:3d}/{epochs}  train loss {running / seen:.5f}  "
                f"val loss {val['loss']:.5f}  val MAE first action {val['val_mae_px']:.2f} px  "
                f"whole chunk {val['val_chunk_mae_px']:.2f} px  ({history['seconds'][-1]:.0f}s)"
            )
    averaged.eval()
    return history, averaged


# ---------------------------------------------------------------- data, config, command line


def build_config(code, args, n_params, train_ids, available, device) -> dict:
    import torch

    settings = ARM_SETTINGS[code]
    config = {
        "arm": code,
        "optimizer": "adam",
        "n_parameters": n_params,
        "epochs": args["epochs"],
        "batch_size": args["batch_size"],
        "lr": args["lr"],
        "val_fraction": VAL_FRACTION,
        "split_seed": SPLIT_SEED,
        "seed": args["seed"],
        "train_episodes": len(train_ids),
        "available_train_episodes": available,
        "subset_seed": None,
        "ema_decay": settings["ema"],
        "val_every": settings["val_every"],
        "train_noise_seed": args["seed"] + TRAIN_NOISE_OFFSET,
        "val_noise_seed": VAL_NOISE_SEED,
        "device": str(device),
        "torch": str(torch.__version__),  # a plain str, so the safe loader accepts it
    }
    if code in ("C", "P"):
        config.update(
            architecture="ChunkPolicyNet",
            loss="mse over the valid chunk positions",
            chunk_len=ChunkArm.chunk_len,
            n_exec=8,
        )
    elif code == "B":
        config.update(architecture="BCPolicyNet", loss="mse", chunk_len=1, n_exec=1)
    else:
        config.update(
            architecture="DiffusionPolicyNet",
            loss="epsilon-prediction mse over the valid chunk positions",
            chunk_len=DiffusionArm.chunk_len,
            n_exec=8,
            width=256,
            n_blocks=4,
            time_dim=64,
            cond_dim=256,
            diffusion={
                "t_steps": dc.T_STEPS,
                "sampling_steps": dc.SAMPLING_STEPS,
                "schedule": "squared cosine",
                "cosine_offset": dc.COSINE_OFFSET,
                "max_beta": dc.MAX_BETA,
                "ddim_eta": 0.0,
                "clip_x0": True,
                "action_scale": "x = 2a - 1, a in [0, 1]",
            },
        )
    return config


def run_one(code, seed, lr, args, arrays, train_ids, val_ids, train_res, val_res, device, log=print):
    """Train one network and write its checkpoint and report. Returns the report dict."""
    import torch

    import pusht_data
    import train_bc as tb

    settings = ARM_SETTINGS[code]
    arm = settings["arm"]()
    name = args.name or default_name(code, seed)
    checkpoint = Path(args.checkpoint_dir) / f"{name}.pt"
    report_path = Path(args.report_dir) / f"training_{name}.json"
    if (checkpoint.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{checkpoint} or {report_path} exists; pass --overwrite to replace it")

    epochs = args.epochs if args.epochs is not None else settings["epochs"]
    tb.seed_everything(seed)
    model = arm.make_model()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[{name}] arm {code}  seed {seed}  lr {lr:g}  epochs {epochs}  parameters {n_params:,}")
    history, averaged = train_arm(
        arm,
        model,
        train_res,
        val_res,
        epochs=epochs,
        lr=lr,
        batch_size=args.batch_size,
        device=device,
        seed=seed,
        ema_decay=settings["ema"],
        val_every=settings["val_every"],
        log=log,
    )
    config = build_config(
        code,
        {"epochs": epochs, "batch_size": args.batch_size, "lr": lr, "seed": seed},
        n_params,
        train_ids,
        len(train_ids),
        device,
    )
    references = tb.reference_maes(arrays, train_ids, val_ids)
    split = {"train_episode_ids": list(train_ids), "val_episode_ids": list(val_ids)}
    tb.save_checkpoint(checkpoint, averaged.cpu(), config, split, history, references)
    final = {k: history[k][-1] for k in ("val_loss", "val_mae_px", "val_chunk_mae_px")}
    report = {
        "name": name,
        "config": config,
        "dataset_revision": pusht_data.DATASET_REVISION,
        "split": split,
        "references": references,
        "history": history,
        "final": final,
        "train_seconds": history["seconds"][-1],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    log(
        f"[{name}] final first-action MAE {final['val_mae_px']:.2f} px "
        f"(copy agent position {references['copy_agent_position_mae_px']:.2f}, "
        f"predict mean action {references['mean_action_mae_px']:.2f}); "
        f"saved {checkpoint}, {report_path}"
    )
    return report


def main(argv=None, *, arrays_loader=None) -> int:
    parser = argparse.ArgumentParser(description="Train study arms on resident data with EMA")
    parser.add_argument("--arm", required=True, choices=sorted(ARM_SETTINGS))
    parser.add_argument("--seeds", default="0", help="e.g. 0, 0-9 or 0,3,5-6; one process builds the data once")
    parser.add_argument("--lr", type=float, default=None, help="required for D, C, B; P uses 1e-3")
    parser.add_argument("--epochs", type=int, default=None, help="default: 370 (30 for P)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default=None, help="checkpoint name; only with a single seed")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    if args.name is not None and len(seeds) != 1:
        parser.error("--name needs exactly one seed")
    lr = args.lr
    if lr is None:
        if args.arm != "P":
            parser.error("--lr is required for arms D, C and B (it comes from the pilot)")
        lr = 1e-3
    if args.arm == "P" and lr != 1e-3:
        parser.error("arm P is the original recipe: lr 1e-3")

    import torch

    import pusht_data
    import train_bc as tb
    from train_chunk import ChunkDataset

    device = tb.pick_device(args.device)
    arrays = (arrays_loader or pusht_data.load_arrays)()
    train_ids, val_ids = pusht_data.split_episode_ids(arrays.n_episodes, VAL_FRACTION, SPLIT_SEED)
    chunk_len = ARM_SETTINGS[args.arm]["arm"].chunk_len
    print(f"building resident data on {device} (chunk length {chunk_len}) ...")
    train_res = rd.build_resident(arrays, ChunkDataset(arrays, train_ids, chunk_len), device)
    val_res = rd.build_resident(arrays, ChunkDataset(arrays, val_ids, chunk_len), device)
    print(f"resident data: {train_res.megabytes() + val_res.megabytes():.0f} MB")

    for seed in seeds:
        run_one(args.arm, seed, lr, args, arrays, train_ids, val_ids, train_res, val_res, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
