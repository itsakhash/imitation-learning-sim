"""Plots of the saved results. No simulation and no training: this only reads the JSON files in
reports/ and writes PNG files.

Figures:
  demonstrations.png    mean max and mean final coverage against the number of training episodes, for
                        both methods (reports/ablation_study.json)
  methods.png           the networks of the seed study and of the replication, baseline against
                        chunked, with the pre-registered difference printed on each block
                        (reports/seed_study.json, reports/replication_seed_study.json)
  executed_targets.png  peak, final and given-back coverage against the number of targets executed
                        per prediction, for both blocks of chunked networks, with each block's
                        baseline as a dotted line (reports/nexec_study.json,
                        reports/nexec_study_s10_s19.json); the two blocks are shifted slightly
                        sideways so that their bars do not overlap

Every dot is one network (its mean over the 50 evaluation episodes). Error bars are the 95%
bootstrap interval of the mean over the networks, the same bootstrap as the studies (10,000
resamples, fixed seed). Gray lines are the do-nothing and random policies (reports/eval_hold.json,
reports/eval_random.json). All numbers come from the saved files; the only thing computed here is
those intervals.

Run:  python plot_results.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import evaluate as ev  # noqa: E402

FIGURES = ("demonstrations", "methods", "executed_targets")
FILES = {
    "ablation": "ablation_study.json",
    "seed_study": "seed_study.json",
    "replication": "replication_seed_study.json",
    "nexec": "nexec_study.json",
    "nexec_replication": "nexec_study_s10_s19.json",
}
EXECUTED = (1, 2, 4, 8)
BASELINE_COLOR = "#1f77b4"
CHUNK_COLOR = "#ff7f0e"
ORIGINAL_COLOR = "#2ca02c"
REPLICATION_COLOR = "#9467bd"
BLOCK_OFFSET = 0.07
DPI = 150


class PlotError(Exception):
    """A saved report is missing or cannot be used."""


# ------------------------------------------------------------------------------ data


def load_json(path) -> dict:
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise PlotError(
            f"{path} does not exist (run the study that writes it first)"
        ) from None
    except json.JSONDecodeError as err:
        raise PlotError(f"{path} is not valid JSON: {err}") from None


def group_ci(values):
    """(mean, low, high): the mean of the values and the 95% bootstrap interval that resamples them."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or len(v) == 0:
        raise ValueError("values must be a non-empty 1-D sequence")
    low, high = ev.bootstrap_ci(v)
    return float(v.mean()), low, high


def reference_levels(report_dir) -> dict:
    """{'hold': {'max', 'final'}, 'random': {'max', 'final'}} from the saved evaluation reports."""
    levels = {}
    for name in ("hold", "random"):
        summary = load_json(Path(report_dir) / f"eval_{name}.json")["summary"]
        levels[name] = {
            "max": float(summary["mean_max_coverage"]),
            "final": float(summary["mean_final_coverage"]),
        }
    return levels


def ablation_points(ablation: dict) -> dict:
    """{method: {training episodes: {'max': [per-network values], 'final': [...]}}}, sizes sorted."""
    points = {}
    for method, by_size in ablation["cells"].items():
        cells = {
            int(size): {
                "max": [float(x) for x in cell["mean_max_coverage"]["values"]],
                "final": [float(x) for x in cell["mean_final_coverage"]["values"]],
            }
            for size, cell in by_size.items()
        }
        points[method] = dict(sorted(cells.items()))
    return points


def seed_block_points(study: dict) -> dict:
    """Per-network values of both methods and the pre-registered difference of one block of ten networks."""
    primary = study["primary"]
    out = {"baseline": {}, "chunk": {}, "difference": {}}
    for key, metric in (("max", "mean_max_coverage"), ("final", "mean_final_coverage")):
        entry = primary[metric]
        out["baseline"][key] = [float(x) for x in entry["baseline_values"]]
        out["chunk"][key] = [float(x) for x in entry["chunk_values"]]
        low, high = entry["ci95"]
        out["difference"][key] = (float(entry["difference"]), float(low), float(high))
    return out


def nexec_points(study: dict) -> dict:
    """{'cells': {executed targets: {'max', 'final', 'giveback': per-network values}}, 'baseline': {...means}}."""
    cells = {
        int(n): {
            "max": [float(x) for x in cell["mean_max_coverage"]["values"]],
            "final": [float(x) for x in cell["mean_final_coverage"]["values"]],
            "giveback": [float(x) for x in cell["mean_giveback"]["values"]],
        }
        for n, cell in study["cells"].items()
    }
    base = study["baseline"]
    return {
        "cells": dict(sorted(cells.items())),
        "baseline": {
            "max": float(base["mean_max_coverage"]),
            "final": float(base["mean_final_coverage"]),
            "giveback": float(base["mean_giveback"]),
        },
    }


# --------------------------------------------------------------------------- drawing


def _spread(n: int, half_width: float):
    """Deterministic horizontal offsets that keep the dots of one group from overlapping."""
    return np.zeros(1) if n == 1 else np.linspace(-half_width, half_width, n)


def _errors(stats):
    """Means and the (2, n) array of distances to the lower and upper end of each interval."""
    means = np.array([s[0] for s in stats])
    lows = np.array([s[1] for s in stats])
    highs = np.array([s[2] for s in stats])
    return means, np.vstack(
        [np.clip(means - lows, 0, None), np.clip(highs - means, 0, None)]
    )


def _draw_references(ax, references: dict, key: str) -> None:
    ax.axhline(
        references["hold"][key],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="do nothing",
    )
    ax.axhline(
        references["random"][key],
        color="gray",
        linestyle=":",
        linewidth=1,
        label="random",
    )


def figure_demonstrations(ablation: dict, references: dict):
    points = ablation_points(ablation)
    sizes = sorted({size for by_size in points.values() for size in by_size})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = (
        ("max", "Peak coverage (mean max coverage)"),
        ("final", "Final coverage (mean final coverage)"),
    )
    for ax, (key, title) in zip(axes, panels):
        for method, label, color, offset in (
            ("baseline", "baseline", BASELINE_COLOR, -4.0),
            ("chunk", "chunked", CHUNK_COLOR, 4.0),
        ):
            xs, stats = [], []
            for size, cell in points[method].items():
                values = cell[key]
                xs.append(size + offset)
                stats.append(group_ci(values))
                ax.scatter(
                    size + offset + _spread(len(values), 2.5),
                    values,
                    s=12,
                    alpha=0.35,
                    color=color,
                    zorder=2,
                )
            means, errors = _errors(stats)
            ax.errorbar(
                xs,
                means,
                yerr=errors,
                fmt="-o",
                color=color,
                capsize=3,
                markersize=5,
                label=label,
                zorder=3,
            )
        _draw_references(ax, references, key)
        ax.set_xticks(sizes)
        ax.set_xlim(min(sizes) - 14, max(sizes) + 14)
        ax.set_xlabel("training episodes")
        ax.set_ylabel("coverage")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Every size was trained for 30 epochs, so smaller training sets also take fewer gradient steps. "
        "Dots: networks. Bars: 95% bootstrap interval of the mean over networks.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def figure_methods(seed_study: dict, replication: dict, references: dict):
    blocks = (
        ("seeds 0-9", seed_block_points(seed_study)),
        ("seeds 10-19", seed_block_points(replication)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = (
        ("max", "Peak coverage (mean max coverage)"),
        ("final", "Final coverage (mean final coverage)"),
    )
    for ax, (key, title) in zip(axes, panels):
        top = max(
            max(data[group][key])
            for _, data in blocks
            for group in ("baseline", "chunk")
        )
        for b, (name, data) in enumerate(blocks):
            for j, (group, color) in enumerate(
                (("baseline", BASELINE_COLOR), ("chunk", CHUNK_COLOR))
            ):
                x = 3 * b + j
                values = data[group][key]
                ax.scatter(
                    x + _spread(len(values), 0.22),
                    values,
                    s=14,
                    alpha=0.4,
                    color=color,
                    zorder=2,
                )
                mean, low, high = group_ci(values)
                ax.errorbar(
                    [x],
                    [mean],
                    yerr=[[max(0.0, mean - low)], [max(0.0, high - mean)]],
                    fmt="o",
                    color=color,
                    capsize=5,
                    markersize=7,
                    markeredgecolor="black",
                    zorder=3,
                )
            diff, low, high = data["difference"][key]
            ax.text(
                3 * b + 0.5,
                top * 1.06,
                f"chunked minus baseline\n{diff:+.4f} [{low:+.4f}, {high:+.4f}]",
                ha="center",
                va="bottom",
                fontsize=7,
            )
            ax.text(
                3 * b + 0.5,
                -0.17,
                f"training {name}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
            )
        _draw_references(ax, references, key)
        ax.set_xticks([0, 1, 3, 4])
        ax.set_xticklabels(["baseline", "chunked", "baseline", "chunked"])
        ax.set_xlim(-0.7, 4.7)
        ax.set_ylim(0, top * 1.3)
        ax.set_ylabel("coverage")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="center", bbox_to_anchor=(0.5, 0.55))
    fig.text(
        0.5,
        0.01,
        "Dots: networks. Bars: 95% bootstrap interval of the group mean. "
        "The printed difference and interval are the pre-registered ones.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def _position(n: int) -> int:
    if n not in EXECUTED:
        raise PlotError(f"unexpected number of executed targets: {n}")
    return EXECUTED.index(n)


def figure_executed_targets(original: dict, replication: dict, references: dict):
    blocks = (
        ("seeds 0-9", nexec_points(original), ORIGINAL_COLOR),
        ("seeds 10-19", nexec_points(replication), REPLICATION_COLOR),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = (
        ("max", "Peak coverage (mean max coverage)"),
        ("final", "Final coverage (mean final coverage)"),
        ("giveback", "Coverage given back (max minus final)"),
    )
    for ax, (key, title) in zip(axes, panels):
        for b, (name, data, color) in enumerate(blocks):
            shift = BLOCK_OFFSET * (1 if b else -1)
            xs, stats = [], []
            for n, cell in data["cells"].items():
                xs.append(_position(n) + shift)
                stats.append(group_ci(cell[key]))
            means, errors = _errors(stats)
            ax.errorbar(
                xs,
                means,
                yerr=errors,
                fmt="-o",
                color=color,
                capsize=3,
                markersize=5,
                label=f"chunked, {name}",
            )
            ax.axhline(
                data["baseline"][key],
                color=color,
                linestyle=":",
                linewidth=1.2,
                label=f"baseline, {name}",
            )
        if key != "giveback":
            ax.axhline(
                references["hold"][key],
                color="gray",
                linestyle="--",
                linewidth=1,
                label="do nothing",
            )
        ax.set_xticks(range(len(EXECUTED)))
        ax.set_xticklabels([str(n) for n in EXECUTED])
        ax.set_xlabel("targets executed per prediction (of 16)")
        ax.set_ylabel("coverage")
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.text(
        0.5,
        0.01,
        "The same checkpoints, evaluated executing fewer of their predicted targets (a test-time change only). "
        "Bars: 95% bootstrap interval of the mean over networks; the two blocks are shifted slightly sideways.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def save_figure(fig, path, dpi: int = DPI) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- command line


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Plot the saved results")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/plots"))
    parser.add_argument("--which", choices=FIGURES + ("all",), default="all")
    args = parser.parse_args(argv)

    wanted = FIGURES if args.which == "all" else (args.which,)
    written = []
    try:
        references = reference_levels(args.report_dir)
        for name in wanted:
            if name == "demonstrations":
                fig = figure_demonstrations(
                    load_json(args.report_dir / FILES["ablation"]), references
                )
            elif name == "methods":
                fig = figure_methods(
                    load_json(args.report_dir / FILES["seed_study"]),
                    load_json(args.report_dir / FILES["replication"]),
                    references,
                )
            else:
                fig = figure_executed_targets(
                    load_json(args.report_dir / FILES["nexec"]),
                    load_json(args.report_dir / FILES["nexec_replication"]),
                    references,
                )
            path = args.out_dir / f"{name}.png"
            save_figure(fig, path)
            written.append(path)
    except PlotError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyError as err:
        print(f"error: a saved report is missing the field {err}", file=sys.stderr)
        return 1
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    for path in written:
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
