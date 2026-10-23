"""Diffusion core for the pre-registered diffusion study (see diffusion_preregistration.md).

Pure mathematics of DDPM training and DDIM sampling for an action chunk of shape (B, K, 2),
kept apart from any network so it can be tested on its own:
  - scaling of the action chunk between the dataset's [0, 1] and the diffusion's [-1, 1];
  - the squared-cosine noise schedule (T = 100 training steps);
  - the forward (noising) process and the noise-prediction (epsilon) training loss, averaged
    over the valid chunk positions only, exactly as train_chunk.chunk_loss is;
  - deterministic DDIM sampling (eta = 0) with a subset of the training steps (10 in the study).

The schedule, T, the number of sampling steps and clipping the predicted clean sample to
[-1, 1] are from my memory of published setups and are not tuned. The clipping is not stated in
the pre-registration; it is an implementation detail, fixed here (clip_x0=True) before any
network is trained, and is recorded in the README once the study is run.

The schedule and the sampling loop use only arithmetic that works the same on numpy arrays and
torch tensors, so the algorithm is tested here with numpy (an oracle denoiser must make DDIM
return the clean chunk exactly). The training loss and the seeded noise need torch.

Conventions: timestep index t runs 0 .. T-1; alpha_bars[t] is the product of (1 - beta_i) for
i <= t, i.e. the signal fraction after t + 1 noising steps; the noisy chunk is
x_t = sqrt(alpha_bars[t]) * x0 + sqrt(1 - alpha_bars[t]) * noise.
"""

from __future__ import annotations

import numpy as np

T_STEPS = 100
SAMPLING_STEPS = 10
COSINE_OFFSET = 0.008
MAX_BETA = 0.999


# ---------------------------------------------------------------- scaling (works on both)


def to_diffusion_scale(a):
    """Dataset scale [0, 1] (pixels divided by 512) -> diffusion scale [-1, 1]: x = 2a - 1.
    Fixed constants; nothing is taken from the data."""
    return a * 2.0 - 1.0


def from_diffusion_scale(x):
    """Diffusion scale [-1, 1] -> dataset scale [0, 1]: a = (x + 1) / 2."""
    return (x + 1.0) / 2.0


# ---------------------------------------------------------------- schedule (numpy)


def cosine_alpha_bars(n_steps: int = T_STEPS, offset: float = COSINE_OFFSET, max_beta: float = MAX_BETA):
    """Squared-cosine schedule (Nichol and Dhariwal). Returns float64 alpha_bars of length
    n_steps: alpha_bars[t] = prod over i <= t of (1 - beta_i), with every beta_i clipped to
    max_beta so the last step stays finite."""
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if not 0.0 < max_beta < 1.0:
        raise ValueError("max_beta must be strictly between 0 and 1")
    if offset < 0.0:
        raise ValueError("offset must be non-negative")
    steps = np.arange(n_steps + 1, dtype=np.float64)
    f = np.cos(((steps / n_steps) + offset) / (1.0 + offset) * np.pi / 2.0) ** 2
    continuous = f / f[0]
    betas = np.minimum(1.0 - continuous[1:] / continuous[:-1], max_beta)
    return np.cumprod(1.0 - betas)


def ddim_timesteps(n_train_steps: int = T_STEPS, n_sample_steps: int = SAMPLING_STEPS):
    """The training-step indices a DDIM sampler visits, from the noisiest to the cleanest:
    evenly spaced over 0 .. n_train_steps - 1, always including both ends."""
    if n_sample_steps < 2 or n_sample_steps > n_train_steps:
        raise ValueError("n_sample_steps must be between 2 and n_train_steps")
    steps = np.round(np.linspace(0, n_train_steps - 1, n_sample_steps)).astype(int)
    steps = np.unique(steps)  # sorted ascending, no repeats
    return [int(t) for t in steps[::-1]]


# ---------------------------------------------------------------- forward process and DDIM


def q_sample(x0, noise, alpha_bar_t):
    """The noisy chunk x_t = sqrt(ab) * x0 + sqrt(1 - ab) * noise. alpha_bar_t is a float or an
    array that broadcasts against x0 (for example shape (B, 1, 1))."""
    return alpha_bar_t**0.5 * x0 + (1.0 - alpha_bar_t) ** 0.5 * noise


def predict_x0(x_t, eps, alpha_bar_t):
    """The clean chunk implied by x_t and a noise prediction."""
    return (x_t - (1.0 - alpha_bar_t) ** 0.5 * eps) / alpha_bar_t**0.5


def _clip(x, low: float, high: float):
    if hasattr(x, "clamp"):  # torch tensor
        return x.clamp(low, high)
    return np.clip(x, low, high)


def ddim_step(x_t, eps, alpha_bar_t: float, alpha_bar_prev: float, clip_x0: bool = True):
    """One deterministic DDIM step (eta = 0) from alpha_bar_t to alpha_bar_prev. With
    alpha_bar_prev = 1 the result is the predicted clean chunk. If clip_x0, the predicted clean
    chunk is clipped to [-1, 1] and the noise is recomputed from it, so x_prev stays consistent."""
    x0 = predict_x0(x_t, eps, alpha_bar_t)
    if clip_x0:
        x0 = _clip(x0, -1.0, 1.0)
        eps = (x_t - alpha_bar_t**0.5 * x0) / (1.0 - alpha_bar_t) ** 0.5
    return alpha_bar_prev**0.5 * x0 + (1.0 - alpha_bar_prev) ** 0.5 * eps


def ddim_sample(model_fn, x_init, alpha_bars, timesteps=None, clip_x0: bool = True):
    """Run DDIM from the initial noise x_init to a clean chunk.

    model_fn(x, t) returns the predicted noise for the noisy chunk x at training-step index t
    (a python int; the caller turns it into whatever its network needs). timesteps defaults to
    ddim_timesteps(len(alpha_bars)), noisiest first. The last step ends at alpha_bar = 1."""
    if timesteps is None:
        timesteps = ddim_timesteps(len(alpha_bars))
    timesteps = list(timesteps)
    if not timesteps or any(b >= a for a, b in zip(timesteps, timesteps[1:])):
        raise ValueError("timesteps must be a non-empty, strictly decreasing sequence")
    if timesteps[0] > len(alpha_bars) - 1 or timesteps[-1] < 0:
        raise ValueError("timesteps must lie between 0 and len(alpha_bars) - 1")
    x = x_init
    for i, t in enumerate(timesteps):
        ab_t = float(alpha_bars[t])
        ab_prev = float(alpha_bars[timesteps[i + 1]]) if i + 1 < len(timesteps) else 1.0
        x = ddim_step(x, model_fn(x, t), ab_t, ab_prev, clip_x0=clip_x0)
    return x


# ---------------------------------------------------------------- torch: noise and loss


def seeded_noise(shape, seed: int, device="cpu", dtype=None):
    """Standard normal noise drawn on the CPU from a generator seeded with `seed`, then moved to
    `device`, so the same seed gives the same noise on any device."""
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(tuple(shape), generator=generator)
    return noise.to(device=device, dtype=dtype or torch.float32)


def diffusion_loss(model_fn, x0, mask, alpha_bars, generator=None, t=None, noise=None):
    """Noise-prediction (epsilon) loss, averaged over the valid chunk positions only.

    x0: (B, K, 2) clean chunk on the diffusion scale; mask: (B, K), 1 for valid positions and 0
    for positions padded past an episode's end; alpha_bars: float tensor of length T on the same
    device. model_fn(x_t, t) gets the noisy chunk and a (B,) long tensor of step indices and
    returns the predicted noise, shape (B, K, 2). t and noise are drawn on the CPU from
    `generator` (so a seeded generator makes a run repeatable) unless they are given. The
    normalisation is the one of train_chunk.chunk_loss."""
    import torch

    batch = x0.shape[0]
    n_steps = len(alpha_bars)
    if t is None:
        t = torch.randint(0, n_steps, (batch,), generator=generator).to(x0.device)
    if noise is None:
        noise = torch.randn(tuple(x0.shape), generator=generator).to(
            device=x0.device, dtype=x0.dtype
        )
    ab = alpha_bars[t].view(batch, 1, 1)
    x_t = q_sample(x0, noise, ab)
    pred = model_fn(x_t, t)
    weight = mask.unsqueeze(-1)
    denominator = (weight.sum() * pred.shape[-1]).clamp_min(1.0)
    return (((pred - noise) ** 2) * weight).sum() / denominator
