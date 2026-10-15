"""Run a trained ChunkPolicyNet as a policy for the evaluation harness.

At the start of an episode, and again every n_exec steps, the network is shown the current
observation and predicts chunk_len commanded targets. The first n_exec of them are then
executed one per step WITHOUT looking at the observations that arrive in between ("open
loop"), after which the policy predicts again from the then-current observation. The
settings (16 predicted, 8 executed) are taken from memory of the Diffusion Policy PushT
setup and are not checked here.

Like BCPolicy, it converts the environment's observation to the network's input (image as
float in [0, 1], channels first; position divided by 512) and converts the answer back to
world pixels, clipped to the environment's action range [0, 512].

Unlike BCPolicy this policy has memory: the queue of planned targets. reset() empties it,
and the evaluation harness calls it at the start of every episode.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from bc_policy import check_observation
from pusht_data import COORD_SCALE


class ChunkPolicy:
    def __init__(self, model, n_exec: int, device="cpu"):
        chunk_len = int(model.chunk_len)
        if not 1 <= n_exec <= chunk_len:
            raise ValueError(f"n_exec must be between 1 and {chunk_len}, got {n_exec}")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.n_exec = int(n_exec)
        self.queue = deque()
        self.predictions = 0  # how many times the network has been asked

    def reset(self, seed=None) -> None:
        """Forget the planned targets (called at the start of every episode)."""
        self.queue.clear()

    def _predict(self, obs) -> np.ndarray:
        check_observation(obs)
        pixels = np.ascontiguousarray(obs["pixels"])
        image = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
        pos = (
            torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
            / COORD_SCALE
        )
        with torch.no_grad():
            out = self.model(
                image.unsqueeze(0).to(self.device), pos.unsqueeze(0).to(self.device)
            )
        chunk = out[0, : self.n_exec].cpu().numpy() * COORD_SCALE
        return np.clip(chunk, 0.0, COORD_SCALE).astype(np.float32)

    def __call__(self, obs):
        if not self.queue:
            self.queue.extend(self._predict(obs))
            self.predictions += 1
        return self.queue.popleft()
