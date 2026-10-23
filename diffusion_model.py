"""The diffusion policy network (arm D of the pre-registered diffusion study).

DiffusionPolicyNet predicts the noise in a noisy action chunk, given the camera image and the
agent's position:
  - the convolutional trunk is train_chunk.make_trunk(), layer for layer the one of the baseline
    and of the chunked network (96x96 image -> 2,304 features);
  - encode(image, agent_pos) turns the image features and the agent position (both as in the
    other networks) into one conditioning vector. It is computed once per prediction, because
    the 10 DDIM steps share it;
  - denoise(x_t, t, cond) runs an MLP of residual blocks over the flattened noisy chunk. Each
    block is modulated by FiLM (a scale and a shift computed from the conditioning vector and the
    timestep embedding), so the image and the noise level enter only through FiLM;
  - the timestep goes through a 64-dimensional sinusoidal embedding and one linear layer.

Sizes (width 256, 4 blocks, 64-dimensional timestep embedding, 256-dimensional conditioning) are
from my memory of published setups and are not tuned. Nothing is zero-initialised.

Shapes: x_t (B, K, 2) on the diffusion scale [-1, 1]; t (B,) integer step indices 0 .. T-1;
image (B, 3, 96, 96) in [0, 1]; agent_pos (B, 2) in [0, 1]. The output (B, K, 2) is the
predicted noise.
"""

from __future__ import annotations

import math

import torch
from torch import nn

import diffusion_core as dc
from train_chunk import CHUNK_LEN, make_trunk

TRUNK_FEATURES = 64 * 6 * 6


def sinusoidal_embedding(t, dim: int = 64):
    """Sinusoidal embedding of integer step indices t (shape (B,)) -> (B, dim): the first half
    sines, the second half cosines, with frequencies from 1 down to 1/10000."""
    if dim < 2 or dim % 2 != 0:
        raise ValueError("dim must be an even number of at least 2")
    if t.dim() != 1:
        raise ValueError("t must have shape (B,)")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.to(torch.float32)[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMResidualBlock(nn.Module):
    """h + MLP(FiLM(LayerNorm(h), c)): the scale and shift come from the conditioning c."""

    def __init__(self, width: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.film = nn.Linear(cond_dim, 2 * width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.SiLU()

    def forward(self, h, c):
        scale, shift = self.film(c).chunk(2, dim=-1)
        y = self.norm(h) * (1.0 + scale) + shift
        y = self.fc1(self.act(y))
        y = self.fc2(self.act(y))
        return h + y


class DiffusionPolicyNet(nn.Module):
    def __init__(
        self,
        chunk_len: int = CHUNK_LEN,
        width: int = 256,
        n_blocks: int = 4,
        time_dim: int = 64,
        cond_dim: int = 256,
    ):
        super().__init__()
        if chunk_len < 1 or width < 1 or n_blocks < 1 or cond_dim < 1:
            raise ValueError("chunk_len, width, n_blocks and cond_dim must be at least 1")
        if time_dim < 2 or time_dim % 2 != 0:
            raise ValueError("time_dim must be an even number of at least 2")
        self.chunk_len = chunk_len
        self.time_dim = time_dim
        self.trunk = make_trunk()
        self.cond_net = nn.Sequential(nn.Linear(TRUNK_FEATURES + 2, cond_dim), nn.SiLU())
        self.time_net = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU())
        self.in_proj = nn.Linear(2 * chunk_len, width)
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(width, cond_dim + time_dim) for _ in range(n_blocks)]
        )
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, 2 * chunk_len)

    def encode(self, image, agent_pos):
        """The conditioning vector (B, cond_dim) from the image and the agent position."""
        features = self.trunk(image)
        return self.cond_net(torch.cat([features, agent_pos], dim=1))

    def denoise(self, x_t, t, cond):
        """Predicted noise (B, K, 2) for the noisy chunk x_t at step indices t."""
        if x_t.dim() != 3 or tuple(x_t.shape[1:]) != (self.chunk_len, 2):
            raise ValueError(f"x_t must have shape (B, {self.chunk_len}, 2), got {tuple(x_t.shape)}")
        if t.dim() != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError("t must have shape (B,), one step index per example")
        batch = x_t.shape[0]
        c = torch.cat([cond, self.time_net(sinusoidal_embedding(t, self.time_dim))], dim=1)
        h = self.in_proj(x_t.reshape(batch, -1))
        for block in self.blocks:
            h = block(h, c)
        return self.out_proj(self.out_norm(h)).view(batch, self.chunk_len, 2)

    def forward(self, x_t, t, image, agent_pos):
        return self.denoise(x_t, t, self.encode(image, agent_pos))


def sample_chunk(net, image, agent_pos, noise, alpha_bars, timesteps=None, clip_x0: bool = True):
    """Sample an action chunk with deterministic DDIM, starting from the given noise.

    The image is encoded once; the network is put in eval mode and no gradients are kept. The
    result is on the diffusion scale [-1, 1] (use diffusion_core.from_diffusion_scale for the
    dataset's [0, 1]). alpha_bars is the schedule (a numpy array or a tensor)."""
    net.eval()
    with torch.no_grad():
        cond = net.encode(image, agent_pos)
        batch = noise.shape[0]

        def model_fn(x, t):
            steps = torch.full((batch,), int(t), dtype=torch.long, device=x.device)
            return net.denoise(x, steps, cond)

        return dc.ddim_sample(model_fn, noise, alpha_bars, timesteps, clip_x0=clip_x0)
