"""Run a trained DiffusionPolicyNet as a policy for the evaluation harness.

Same protocol as chunk_policy.ChunkPolicy: at the start of an episode, and again every n_exec
steps, the network is shown the current observation and predicts a chunk of 16 targets; the
first n_exec are executed one per step without looking at the observations in between.

What differs is how a chunk is predicted: it is sampled with 10 deterministic DDIM steps
(diffusion_model.sample_chunk) from initial noise. The noise comes from a CPU generator that
reset(seed) seeds with the episode's evaluation seed; every prediction in the episode takes the
next draw from it. So an episode is repeated exactly by repeating the seed (on the same
hardware and software), whatever else ran in between. reset() needs a seed, and calling the
policy before reset(seed) is an error, so no episode can silently run on unseeded or reused
noise.

The observation is converted exactly as in ChunkPolicy (image float in [0, 1], channels first;
position divided by 512), and the sampled chunk, on the diffusion scale [-1, 1], is converted
back with a = (x + 1) / 2, multiplied by 512 and clipped to the action range [0, 512].
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

import diffusion_core as dc
import diffusion_model as dm
from bc_policy import check_observation
from pusht_data import COORD_SCALE


def schedule_from_config(config: dict):
    """(alpha_bars, DDIM timesteps) described by a checkpoint's config['diffusion'] block, so a
    checkpoint is sampled with the schedule it was trained with. Raises KeyError / ValueError if
    the block is missing or describes something this code does not implement."""
    block = config["diffusion"]
    if block.get("schedule") != "squared cosine":
        raise ValueError(f"unsupported schedule {block.get('schedule')!r}")
    if float(block.get("ddim_eta", 0.0)) != 0.0:
        raise ValueError("only deterministic DDIM (eta 0) is implemented")
    alpha_bars = dc.cosine_alpha_bars(
        int(block["t_steps"]), float(block["cosine_offset"]), float(block["max_beta"])
    )
    timesteps = dc.ddim_timesteps(int(block["t_steps"]), int(block["sampling_steps"]))
    return alpha_bars, timesteps


class DiffusionPolicy:
    def __init__(self, model, n_exec: int, device="cpu", alpha_bars=None, timesteps=None, clip_x0=True):
        chunk_len = int(model.chunk_len)
        if not 1 <= n_exec <= chunk_len:
            raise ValueError(f"n_exec must be between 1 and {chunk_len}, got {n_exec}")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.n_exec = int(n_exec)
        self.chunk_len = chunk_len
        self.alpha_bars = dc.cosine_alpha_bars() if alpha_bars is None else alpha_bars
        self.timesteps = dc.ddim_timesteps(len(self.alpha_bars)) if timesteps is None else list(timesteps)
        self.clip_x0 = bool(clip_x0)
        self.queue = deque()
        self.predictions = 0  # how many times the network has been asked (this episode and before)
        self._generator = None

    @classmethod
    def from_checkpoint(cls, path, n_exec=None, device="cpu"):
        """(policy, checkpoint dict) from a diffusion checkpoint, with the schedule and the
        number of executed targets taken from its config unless n_exec is given."""
        model, ckpt = load_diffusion_checkpoint(path, device=device)
        used = ckpt["config"]["n_exec"] if n_exec is None else n_exec
        alpha_bars, timesteps = schedule_from_config(ckpt["config"])
        policy = cls(model, used, device=device, alpha_bars=alpha_bars, timesteps=timesteps)
        return policy, ckpt

    def reset(self, seed=None) -> None:
        """Forget the planned targets and seed the noise for a new episode."""
        if seed is None:
            raise ValueError("DiffusionPolicy.reset needs the episode's seed (it seeds the noise)")
        self.queue.clear()
        self._generator = torch.Generator().manual_seed(int(seed))

    def _predict(self, obs) -> np.ndarray:
        check_observation(obs)
        if self._generator is None:
            raise RuntimeError("call reset(seed) before the first prediction")
        pixels = np.ascontiguousarray(obs["pixels"])
        image = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
        pos = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)) / COORD_SCALE
        noise = torch.randn((1, self.chunk_len, 2), generator=self._generator).to(self.device)
        sample = dm.sample_chunk(
            self.model,
            image.unsqueeze(0).to(self.device),
            pos.unsqueeze(0).to(self.device),
            noise,
            self.alpha_bars,
            self.timesteps,
            clip_x0=self.clip_x0,
        )
        chunk = dc.from_diffusion_scale(sample[0, : self.n_exec]).cpu().numpy() * COORD_SCALE
        return np.clip(chunk, 0.0, COORD_SCALE).astype(np.float32)

    def __call__(self, obs):
        if not self.queue:
            self.queue.extend(self._predict(obs))
            self.predictions += 1
        return self.queue.popleft()


def load_diffusion_checkpoint(path, device="cpu"):
    """Return (diffusion network in eval mode, the full checkpoint dict). Stays in torch's safe
    weights-only mode, like train_bc.load_checkpoint. Refuses a checkpoint of another network."""
    import train_bc as tb

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with torch.serialization.safe_globals(tb._version_type_allowlist()):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    config = ckpt["config"]
    if config.get("architecture") != "DiffusionPolicyNet":
        raise ValueError(f"{path} holds a {config.get('architecture')!r}, not a DiffusionPolicyNet")
    model = dm.DiffusionPolicyNet(
        chunk_len=config["chunk_len"],
        width=config["width"],
        n_blocks=config["n_blocks"],
        time_dim=config["time_dim"],
        cond_dim=config["cond_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
