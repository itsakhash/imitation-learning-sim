"""Trace analysis: how do the chunked and the one-step networks lose coverage?

Background. The seed study (seed_study.py) found that the chunked networks reach a higher peak
coverage (+0.0495 on the mean maximum) and end lower (-0.0218 on the mean final coverage). For
every episode, final = max - giveback, where giveback is the coverage lost between the peak and
the end; averaged over episodes this is an exact identity, so the chunked networks must give back
about 0.071 more than the baselines. This asks where that extra giveback comes from, using only
the coverage traces already saved in the evaluation reports (no simulation is run).

Per-episode quantities. Thresholds fixed before looking at any output; they are not tuned:
  giveback          max_coverage - final_coverage
  big giveback      giveback > BIG_GIVEBACK (0.1)
  window drop       the largest fall, at any moment, from the coverage then to the lowest
                    coverage within the next WINDOW (8) steps; 8 is one executed chunk
  abrupt            a big-giveback episode whose window drop is at least ABRUPT_FRACTION (50%)
                    of its giveback; any other big-giveback episode is gradual
  ended empty       final_coverage < EMPTY_COVERAGE (0.001)
  peak step         the first step at which the maximum coverage is reached
A network's abrupt share is defined only if it has at least MIN_BIG_EPISODES (3) big-giveback
episodes; networks without are left out of that comparison and counted.

Reading guide, fixed before looking at any output. The unit is the network (the same bootstrap as
the seed study: 10,000 resamples of the networks of each group, fixed seed).
  step 1  if the interval of the difference in mean giveback (chunked minus baseline) lies
          entirely above zero, the chunked networks give back more coverage; entirely below zero,
          the baseline networks do; otherwise there is no clear difference and the abruptness
          question is not interpreted
  step 2  only if step 1 found that the chunked networks give back more: if the interval of the
          difference in the abrupt share of big-giveback episodes lies entirely above zero, the
          extra loss is more often abrupt (consistent with sudden shoves, not proof); entirely
          below zero, more often gradual; otherwise there is no clear difference in how coverage
          is lost
Every other measure is descriptive.

Limits: coverage traces are stored rounded to 4 decimals; every network was evaluated on the same
50 starting positions; this is one training recipe; an abrupt drop in a coverage trace can have
many causes, so this cannot show why the chunked networks lose coverage, only how it is lost.

Run:  python trace_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import compare_policies as cp
import seed_study as ss

BIG_GIVEBACK = 0.1
WINDOW = 8
ABRUPT_FRACTION = 0.5
MIN_BIG_EPISODES = 3
EMPTY_COVERAGE = 0.001
TRACE_TOLERANCE = 1e-4
N_TRAINING_SEEDS = 10

MEASURES = (
    "mean_giveback",
    "big_giveback_share",
    "abrupt_share_of_big",
    "median_peak_step_of_big",
    "ended_empty_share",
)
GROUP_MEASURES = ("mean_max_coverage", "mean_final_coverage") + MEASURES
LABELS = {
    "mean_max_coverage": ("mean max coverage", ".4f"),
    "mean_final_coverage": ("mean final coverage", ".4f"),
    "mean_giveback": ("mean coverage given back (max minus final)", ".4f"),
    "big_giveback_share": ("share of episodes giving back more than 0.1", ".3f"),
    "abrupt_share_of_big": ("  of those, share with an abrupt drop", ".3f"),
    "median_peak_step_of_big": ("  median step of the peak in those episodes", ".0f"),
    "ended_empty_share": ("share of episodes ending with coverage below 0.001", ".3f"),
}


def peak_step(trace) -> int:
    """The first step (1-based) at which the maximum coverage is reached."""
    t = np.asarray(trace, dtype=np.float64)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("a coverage trace must be a non-empty 1-D sequence")
    return int(np.argmax(t)) + 1


def largest_window_drop(trace, window: int = WINDOW) -> float:
    """The largest fall, at any moment, from the coverage then to the lowest coverage within
    the next `window` steps (0.0 if coverage never falls)."""
    t = np.asarray(trace, dtype=np.float64)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("a coverage trace must be a non-empty 1-D sequence")
    if window < 1:
        raise ValueError("window must be at least 1")
    best = 0.0
    for i in range(len(t) - 1):
        best = max(best, float(t[i] - t[i + 1 : i + 1 + window].min()))
    return best


def episode_measures(
    episode: dict,
    *,
    big: float = BIG_GIVEBACK,
    window: int = WINDOW,
    abrupt_fraction: float = ABRUPT_FRACTION,
    empty: float = EMPTY_COVERAGE,
) -> dict:
    """The per-episode quantities, from one episode of an evaluation report."""
    for key in ("coverage_trace", "max_coverage", "final_coverage"):
        if episode.get(key) is None:
            raise cp.ComparisonError(f"episode {episode.get('seed')} has no '{key}'")
    trace = np.asarray(episode["coverage_trace"], dtype=np.float64)
    if trace.ndim != 1 or len(trace) == 0:
        raise cp.ComparisonError(
            f"episode {episode.get('seed')} has an empty coverage trace"
        )
    max_cov, final = float(episode["max_coverage"]), float(episode["final_coverage"])
    if (
        abs(float(trace.max()) - max_cov) > TRACE_TOLERANCE
        or abs(float(trace[-1]) - final) > TRACE_TOLERANCE
    ):
        raise cp.ComparisonError(
            f"episode {episode.get('seed')}: the coverage trace does not match the stored "
            "max and final coverage"
        )
    giveback = max_cov - final
    drop = largest_window_drop(trace, window)
    is_big = giveback > big
    return {
        "seed": int(episode["seed"]),
        "peak_step": peak_step(trace),
        "giveback": giveback,
        "big": bool(is_big),
        "window_drop": drop,
        "abrupt": bool(is_big and drop >= abrupt_fraction * giveback),
        "ended_empty": bool(final < empty),
    }


def network_measures(
    report: dict,
    *,
    big: float = BIG_GIVEBACK,
    window: int = WINDOW,
    abrupt_fraction: float = ABRUPT_FRACTION,
    empty: float = EMPTY_COVERAGE,
    min_big: int = MIN_BIG_EPISODES,
) -> dict:
    """One network's numbers, from its evaluation report."""
    episodes = report.get("episodes") or []
    if not episodes:
        raise cp.ComparisonError(f"{report.get('policy')}: the report has no episodes")
    rows = [
        episode_measures(
            e, big=big, window=window, abrupt_fraction=abrupt_fraction, empty=empty
        )
        for e in episodes
    ]
    big_rows = [r for r in rows if r["big"]]
    n, n_big = len(rows), len(big_rows)
    return {
        "name": report["policy"],
        "n_episodes": n,
        "mean_max_coverage": float(np.mean([e["max_coverage"] for e in episodes])),
        "mean_final_coverage": float(np.mean([e["final_coverage"] for e in episodes])),
        "mean_giveback": float(np.mean([r["giveback"] for r in rows])),
        "n_big": n_big,
        "big_giveback_share": n_big / n,
        "abrupt_share_of_big": (
            (sum(r["abrupt"] for r in big_rows) / n_big) if n_big >= min_big else None
        ),
        "median_peak_step_of_big": (
            float(np.median([r["peak_step"] for r in big_rows])) if n_big else None
        ),
        "ended_empty_share": sum(r["ended_empty"] for r in rows) / n,
    }


def compare_measure(baseline_values, chunk_values) -> dict:
    """Group means and the bootstrap interval of chunked minus baseline, skipping networks
    whose value is undefined (None). The interval needs two defined values in each group.
    """
    a = [v for v in baseline_values if v is not None]
    b = [v for v in chunk_values if v is not None]
    out = {
        "baseline_values": list(baseline_values),
        "chunk_values": list(chunk_values),
        "baseline_used": len(a),
        "chunk_used": len(b),
        "baseline_mean": float(np.mean(a)) if a else None,
        "chunk_mean": float(np.mean(b)) if b else None,
        "difference": None,
        "ci95": None,
    }
    if len(a) >= 2 and len(b) >= 2:
        difference, low, high = ss.bootstrap_difference(a, b)
        out["difference"], out["ci95"] = difference, [low, high]
    return out


def compare_groups(baseline_nets, chunk_nets) -> dict:
    return {
        key: compare_measure(
            [n[key] for n in baseline_nets], [n[key] for n in chunk_nets]
        )
        for key in GROUP_MEASURES
    }


def identity_residual(measures: dict):
    """final difference minus (max difference minus giveback difference): zero up to rounding
    for any set of networks, because final = max - giveback in every episode. None if the
    differences are not defined."""
    parts = [
        measures[k]["difference"]
        for k in ("mean_final_coverage", "mean_max_coverage", "mean_giveback")
    ]
    if any(p is None for p in parts):
        return None
    final, maximum, giveback = parts
    return float(final - (maximum - giveback))


def reading_guide(measures: dict) -> dict:
    """The two-step reading guide, applied to the interval of each measure."""
    interval = measures["mean_giveback"]["ci95"]
    if interval is None:
        return {"step1": "not evaluable (too few networks)", "step2": None}
    low, high = interval
    if high < 0:
        return {
            "step1": "baseline networks give back more coverage than chunked networks",
            "step2": None,
        }
    if not low > 0:
        return {"step1": "no clear difference in coverage given back", "step2": None}
    step1 = "chunked networks give back more coverage than baseline networks"
    interval = measures["abrupt_share_of_big"]["ci95"]
    if interval is None:
        step2 = "not evaluable (too few networks with enough big-giveback episodes)"
    elif interval[0] > 0:
        step2 = "the extra loss is more often abrupt (consistent with sudden shoves, not proof)"
    elif interval[1] < 0:
        step2 = "the extra loss is more often gradual"
    else:
        step2 = "no clear difference in how coverage is lost"
    return {"step1": step1, "step2": step2}


def analyze(
    baselines,
    chunks,
    *,
    big=BIG_GIVEBACK,
    window=WINDOW,
    abrupt_fraction=ABRUPT_FRACTION,
    empty=EMPTY_COVERAGE,
    min_big=MIN_BIG_EPISODES,
) -> dict:
    baselines, chunks = list(baselines), list(chunks)
    ss.check_all_comparable(baselines + chunks)
    settings = {
        "big_giveback": big,
        "window": window,
        "abrupt_fraction": abrupt_fraction,
        "empty_coverage": empty,
        "min_big_episodes": min_big,
    }
    base = [
        network_measures(
            r,
            **{
                k: v
                for k, v in zip(
                    ("big", "window", "abrupt_fraction", "empty", "min_big"),
                    (big, window, abrupt_fraction, empty, min_big),
                )
            },
        )
        for r in baselines
    ]
    chunk = [
        network_measures(
            r,
            **{
                k: v
                for k, v in zip(
                    ("big", "window", "abrupt_fraction", "empty", "min_big"),
                    (big, window, abrupt_fraction, empty, min_big),
                )
            },
        )
        for r in chunks
    ]
    measures = compare_groups(base, chunk)
    first = baselines[0]["summary"]
    warnings = []
    if min(len(base), len(chunk)) < ss.PLANNED_PER_GROUP:
        warnings.append(
            f"fewer than {ss.PLANNED_PER_GROUP} networks in a group: not the full analysis"
        )
    return {
        "n_baseline": len(base),
        "n_chunk": len(chunk),
        "evaluation": {
            "episodes": first["n_episodes"],
            "seed_start": first["seed_start"],
            "seed_end": first["seed_end"],
        },
        "settings": settings,
        "baseline_networks": base,
        "chunk_networks": chunk,
        "measures": measures,
        "identity_residual": identity_residual(measures),
        "guide": reading_guide(measures),
        "warnings": warnings,
    }


def _fmt(value, spec: str, sign: bool = False) -> str:
    return "n/a" if value is None else format(value, ("+" if sign else "") + spec)


def format_report(result: dict) -> str:
    e = result["evaluation"]
    lines = [
        f"trace analysis: {result['n_baseline']} baseline and {result['n_chunk']} chunked networks, each on "
        f"{e['episodes']} episodes (seeds {e['seed_start']} to {e['seed_end']}); the unit is the network"
    ]
    for key in GROUP_MEASURES:
        m = result["measures"][key]
        label, spec = LABELS[key]
        ci = m["ci95"]
        ci_text = "n/a" if ci is None else f"[{ci[0]:+{spec}}, {ci[1]:+{spec}}]"
        lines.append(
            f"{label}: baseline {_fmt(m['baseline_mean'], spec)}  chunked {_fmt(m['chunk_mean'], spec)}  "
            f"chunked minus baseline {_fmt(m['difference'], spec, True)}  95% CI {ci_text}  "
            f"(networks used {m['baseline_used']}/{m['chunk_used']})"
        )
    residual = result["identity_residual"]
    lines.append(
        "identity: final difference = max difference minus giveback difference, residual "
        + ("n/a" if residual is None else f"{residual:.1e}")
    )
    for kind in ("baseline", "chunk"):
        values = result["measures"]["mean_giveback"][f"{kind}_values"]
        lines.append(
            f"  {kind} networks, coverage given back: "
            + ", ".join(f"{v:.4f}" for v in values)
        )
    guide = result["guide"]
    lines.append(f"reading guide, step 1: {guide['step1']}")
    if guide["step2"] is not None:
        lines.append(f"reading guide, step 2: {guide['step2']}")
    s = result["settings"]
    lines.append(
        f"thresholds fixed in advance: big giveback above {s['big_giveback']}; abrupt means the largest drop "
        f"within {s['window']} steps is at least {100 * s['abrupt_fraction']:.0f}% of the giveback; ended empty "
        f"means final coverage below {s['empty_coverage']}; a network needs at least {s['min_big_episodes']} "
        "big-giveback episodes for an abrupt share"
    )
    for warning in result["warnings"]:
        lines.append(f"WARNING: {warning}")
    lines.append(
        "note: descriptive; the same 50 starting positions for every network; coverage traces are rounded to "
        "4 decimals; an abrupt drop can have many causes, so this shows how coverage is lost, not why"
    )
    return "\n".join(lines)


def default_paths(report_dir=Path("reports"), seeds=range(N_TRAINING_SEEDS)):
    def name(base, seed):
        return base if seed == 0 else f"{base}_s{seed}"

    report_dir = Path(report_dir)
    return (
        [report_dir / f"eval_{name('bc_cnn', s)}.json" for s in seeds],
        [report_dir / f"eval_{name('chunk_cnn', s)}.json" for s in seeds],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="How do chunked and one-step networks lose coverage?"
    )
    parser.add_argument("--baseline", type=Path, nargs="+", default=None)
    parser.add_argument("--chunk", type=Path, nargs="+", default=None)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/trace_analysis.json"))
    args = parser.parse_args(argv)

    default_baseline, default_chunk = default_paths(args.report_dir)
    try:
        baselines = [cp.load_report(p) for p in (args.baseline or default_baseline)]
        chunks = [cp.load_report(p) for p in (args.chunk or default_chunk)]
        unverified = [
            r["policy"]
            for r in baselines + chunks
            if ss.reproducibility_status(r, args.report_dir) is not True
        ]
        if unverified and not args.allow_unverified:
            raise cp.ComparisonError(
                "no passing reproducibility record for: "
                + ", ".join(unverified)
                + " (use --allow-unverified to analyse them anyway)"
            )
        result = analyze(baselines, chunks)
    except (cp.ComparisonError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    result["unverified_networks"] = unverified
    if unverified:
        result["warnings"].append(
            "without a passing reproducibility record: " + ", ".join(unverified)
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(format_report(result))
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
