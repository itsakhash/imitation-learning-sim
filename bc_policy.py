"""Wrap a trained BCPolicyNet as a policy the evaluation harness can run.

The environment gives a dict with 'pixels' (96x96x3 uint8) and 'agent_pos' (2 numbers in
world pixels). The network wants the image as float in [0, 1], channels first, and the
position divided by 512, and it answers in the same scaled units. The reply is converted
back to world pixels and clipped to the environment's action range [0, 512].

The policy is stateless: it looks only at the current observation.
"""

from __future__ import annotations

import numpy as np
import torch

from pusht_data import COORD_SCALE, IMAGE_SIZE


def check_observation(obs) -> None:
    """Raise ValueError unless obs has the 'pixels' and 'agent_pos' the policy expects."""
    if "pixels" not in obs or "agent_pos" not in obs:
        raise ValueError("observation needs 'pixels' and 'agent_pos'")
    pixels = np.asarray(obs["pixels"])
    if pixels.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or pixels.dtype != np.uint8:
        raise ValueError(
            f"pixels have shape {pixels.shape} and dtype {pixels.dtype}; "
            f"expected ({IMAGE_SIZE}, {IMAGE_SIZE}, 3) uint8"
        )
    if np.asarray(obs["agent_pos"]).shape != (2,):
        raise ValueError(
            f"agent_pos has shape {np.asarray(obs['agent_pos']).shape}; expected (2,)"
        )


class BCPolicy:
    def __init__(self, model, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def __call__(self, obs):
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
        action = out[0].cpu().numpy() * COORD_SCALE
        return np.clip(action, 0.0, COORD_SCALE).astype(np.float32)
