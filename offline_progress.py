"""Offline look at when, within a demonstration, the trained network's commanded target
falls short of the human's.

The timeline survey (timeline_survey.py) found that in the simulator the network's commanded
target collapses onto the agent's own position after the first quarter of an episode, and the
agent parks. Two explanations are on the table, and they call for different fixes:

  A  it learned how humans END an episode. Demonstrations stop when the human is done: the
     block is near the goal and the agent slows down. A network that cannot tell a finished-
     looking picture from a really finished one parks whenever the picture looks finished.
  B  it shrinks its answer whenever it is unsure. Several different things a human might do
     from the same picture average out to a small displacement, even mid-episode.

This measures, on held-out demonstration episodes only, how the human's displacement (commanded
target minus agent position) and the network's compare across the course of an episode.

Reading guide, fixed before looking at any results. A is supported if the humans move clearly
less in the last progress bins than in the middle ones, and the network's near-copy share rises
together with theirs, while mid-episode the network reproduces most of the human displacement.
B is supported if mid-episode, where humans move most, the network's gap and displacement gain
are clearly below the humans' and its near-copy share is high. Both can be partly true.

Limits: the demonstrations show the states humans visited, not the states the network reaches
once it is in control, so this cannot confirm either explanation. Episode progress is the frame
number divided by the episode's length, which uses the end of the episode: the network never
has access to it.

Run:  python offline_progress.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import inspect_episode as ie
import pusht_data
import train_bc as tb
from pusht_data import COORD_SCALE, PushTDataset

N_BINS = 10
PHASE_EDGES = (
    0,
    75,
    150,
    225,
    300,
)  # the same 75-step phases as the closed-loop timeline
NEAR_PX = 3.0  # a target this close to the agent counts as a copy of it
MIN_DISPLACEMENT_PX = 1.0  # both displacements must exceed this for a direction


def progress_bins(frame_index, lengths, n_bins: int = N_BINS):
    """Bin number (0 .. n_bins-1) of each frame by how far through its episode it is.
    Integer arithmetic, so the boundaries are exact."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    frames = np.asarray(frame_index, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if (lengths < 1).any():
        raise ValueError("episode lengths must be at least 1")
    if (frames < 0).any() or (frames >= lengths).any():
        raise ValueError("every frame index must lie inside its episode")
    return np.minimum(frames * n_bins // lengths, n_bins - 1)


def step_bins(frame_index, edges=PHASE_EDGES):
    """Bin number of each frame by absolute step, with bin i covering [edges[i], edges[i+1]).
    Frames outside every bin get -1."""
    edges = np.asarray(edges, dtype=np.int64)
    if len(edges) < 2 or (np.diff(edges) <= 0).any():
        raise ValueError(
            "edges must be strictly increasing and contain at least two values"
        )
    frames = np.asarray(frame_index, dtype=np.int64)
    ids = np.searchsorted(edges, frames, side="right") - 1
    ids[(ids < 0) | (ids >= len(edges) - 1)] = -1
    return ids


def binned_stats(bin_ids, n_bins: int, dh, dp, move) -> list:
    """Per-bin comparison of the human displacement dh and the network's dp (pixels, shape
    (N, 2)). move is the human agent's displacement per step (NaN where there is no next
    frame in the episode). Frames whose bin id is outside 0 .. n_bins-1 are ignored."""
    bin_ids = np.asarray(bin_ids)
    dh = np.asarray(dh, dtype=np.float64)
    dp = np.asarray(dp, dtype=np.float64)
    move = np.asarray(move, dtype=np.float64)
    n = len(bin_ids)
    if (
        bin_ids.shape != (n,)
        or dh.shape != (n, 2)
        or dp.shape != (n, 2)
        or move.shape != (n,)
    ):
        raise ValueError("bin_ids, dh, dp and move must describe the same N frames")
    norm_h = np.linalg.norm(dh, axis=1)
    norm_p = np.linalg.norm(dp, axis=1)
    rows = []
    for b in range(n_bins):
        sel = bin_ids == b
        count = int(sel.sum())
        row = {
            "bin": b,
            "frames": count,
            "human_gap_px": None,
            "network_gap_px": None,
            "displacement_gain": None,
            "mean_cosine": None,
            "cosine_frames": 0,
            "human_near_copy_fraction": None,
            "network_near_copy_fraction": None,
            "human_move_px": None,
            "human_still_fraction": None,
        }
        if count:
            h, p = dh[sel], dp[sel]
            denominator = float((h * h).sum())
            both = (norm_h[sel] > MIN_DISPLACEMENT_PX) & (
                norm_p[sel] > MIN_DISPLACEMENT_PX
            )
            row["human_gap_px"] = float(norm_h[sel].mean())
            row["network_gap_px"] = float(norm_p[sel].mean())
            row["displacement_gain"] = (
                None if denominator == 0 else float((h * p).sum() / denominator)
            )
            row["cosine_frames"] = int(both.sum())
            if both.any():
                row["mean_cosine"] = float(
                    (
                        (h[both] * p[both]).sum(axis=1)
                        / (norm_h[sel][both] * norm_p[sel][both])
                    ).mean()
                )
            row["human_near_copy_fraction"] = float((norm_h[sel] < NEAR_PX).mean())
            row["network_near_copy_fraction"] = float((norm_p[sel] < NEAR_PX).mean())
            moves = move[sel]
            moves = moves[~np.isnan(moves)]
            if len(moves):
                row["human_move_px"] = float(moves.mean())
                row["human_still_fraction"] = float((moves < ie.STALL_PX).mean())
        rows.append(row)
    return rows


def per_frame_data(model, arrays, episode_ids, device="cpu", batch_size=512) -> dict:
    """For every frame of the given episodes (in dataset order): the human's displacement
    (commanded target minus agent position, in pixels), the network's, the frame number, the
    episode length and the agent's move to the next frame (NaN on an episode's last frame).
    """
    dataset = PushTDataset(arrays, episode_ids)
    rows = dataset.indices
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    human, network = [], []
    with torch.no_grad():
        for batch in loader:
            pos = batch["agent_pos"].to(device)
            pred = model(batch["image"].to(device), pos)
            network.append(
                ((pred - pos) * COORD_SCALE).cpu().numpy().astype(np.float64)
            )
            human.append(
                ((batch["action"].to(device) - pos) * COORD_SCALE)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    episode = arrays.episode_index[rows]
    states = arrays.states.astype(np.float64)
    following = rows + 1
    has_next = following < arrays.n_frames
    same_episode = has_next & (
        arrays.episode_index[np.minimum(following, arrays.n_frames - 1)] == episode
    )
    move = np.full(len(rows), np.nan)
    move[same_episode] = np.linalg.norm(
        states[following[same_episode]] - states[rows[same_episode]], axis=1
    )
    return {
        "rows": rows,
        "dh": np.concatenate(human),
        "dp": np.concatenate(network),
        "frame": arrays.frame_index[rows],
        "length": arrays.episode_lengths[episode],
        "move": move,
    }


def analyze(
    model,
    arrays,
    episode_ids,
    device="cpu",
    batch_size=512,
    n_bins: int = N_BINS,
    phase_edges=PHASE_EDGES,
) -> dict:
    d = per_frame_data(model, arrays, episode_ids, device, batch_size)
    n = len(d["rows"])
    data = (d["dh"], d["dp"], d["move"])
    return {
        "episodes": int(len(np.unique(arrays.episode_index[d["rows"]]))),
        "frames": int(n),
        "n_bins": int(n_bins),
        "phase_edges": [int(e) for e in phase_edges],
        "overall": binned_stats(np.zeros(n, dtype=np.int64), 1, *data)[0],
        "by_progress": binned_stats(
            progress_bins(d["frame"], d["length"], n_bins), n_bins, *data
        ),
        "by_phase": binned_stats(
            step_bins(d["frame"], phase_edges), len(phase_edges) - 1, *data
        ),
    }


def _fmt(value, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _pct(value):
    return None if value is None else 100.0 * value


def _row(label: str, b: dict) -> str:
    return (
        f"{label:<16s}{b['frames']:>8d}"
        f"{_fmt(b['human_gap_px'], '.1f'):>10s}{_fmt(b['network_gap_px'], '.1f'):>10s}"
        f"{_fmt(b['displacement_gain'], '.2f'):>10s}{_fmt(b['mean_cosine'], '.2f'):>10s}"
        f"{_fmt(b['human_move_px'], '.1f'):>10s}{_fmt(_pct(b['human_still_fraction']), '.0f'):>10s}"
        f"{_fmt(_pct(b['human_near_copy_fraction']), '.0f'):>10s}"
        f"{_fmt(_pct(b['network_near_copy_fraction']), '.0f'):>10s}"
    )


def format_tables(result: dict) -> str:
    n_bins, edges = result["n_bins"], result["phase_edges"]
    header = f"{'':<16s}{'frames':>8s}" + "".join(
        f"{c:>10s}"
        for c in (
            "h. gap",
            "n. gap",
            "gain",
            "cosine",
            "h. move",
            "h. still%",
            "h. near%",
            "n. near%",
        )
    )
    lines = [
        f"offline, {result['episodes']} held-out demonstration episodes, {result['frames']} frames",
        header,
        _row("all frames", result["overall"]),
        "-- by progress through the episode --",
    ]
    for b in result["by_progress"]:
        lines.append(
            _row(
                f"{100 * b['bin'] / n_bins:.0f}-{100 * (b['bin'] + 1) / n_bins:.0f}%", b
            )
        )
    lines.append(
        "-- by absolute step, the same 75-step phases as the closed-loop timeline --"
    )
    for b in result["by_phase"]:
        lines.append(_row(f"steps {edges[b['bin']]}-{edges[b['bin'] + 1] - 1}", b))
    lines += [
        "h = human, n = network; gap = distance from the agent to the commanded target (px); "
        "gain = share of the human displacement the network reproduces (1.00 = none lost); "
        "cosine = direction agreement; move = the human agent's displacement per step (px); "
        f"still = share of steps with a move under {ie.STALL_PX:.0f} px; "
        f"near = share of frames whose target is within {NEAR_PX:.0f} px of the agent",
        "note: these are demonstration states, not the states the network reaches once it "
        "is in control, and progress uses each episode's length, which the network never sees",
    ]
    return "\n".join(lines)


def save_progress_figure(result: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [b for b in result["by_progress"] if b["frames"] > 0]
    if not bins:
        raise ValueError("no frames to draw")
    x = [100.0 * (b["bin"] + 0.5) / result["n_bins"] for b in bins]

    def series(key):
        return [np.nan if b[key] is None else b[key] for b in bins]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        x,
        series("human_gap_px"),
        marker="o",
        color="tab:orange",
        label="human: commanded target to agent",
    )
    ax.plot(
        x,
        series("network_gap_px"),
        marker="o",
        color="tab:blue",
        label="network: commanded target to agent",
    )
    ax.plot(
        x,
        series("human_move_px"),
        marker="s",
        color="tab:green",
        linestyle="--",
        label="human: agent move per step",
    )
    ax.set_xlabel("progress through the episode (%)")
    ax.set_ylabel("pixels")
    ax.set_title("held-out demonstration episodes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(
    argv=None,
    *,
    arrays_loader=pusht_data.load_arrays,
    checkpoint_loader=tb.load_checkpoint,
) -> int:
    parser = argparse.ArgumentParser(
        description="Human vs network displacement over an episode"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/bc_cnn.pt")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("reports/offline_progress_bc_cnn.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("reports/offline_progress_bc_cnn.png")
    )
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    args = parser.parse_args(argv)

    model, ckpt = checkpoint_loader(args.checkpoint, device="cpu")
    val_ids = ckpt["split"]["val_episode_ids"]
    result = analyze(model, arrays_loader(), val_ids, n_bins=args.n_bins)
    result["checkpoint"] = str(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    save_progress_figure(result, args.figure)
    print(format_tables(result))
    print(f"saved {args.out} and {args.figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
