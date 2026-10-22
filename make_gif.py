"""Render one episode of a policy as a GIF, and check that it is the episode the evaluation saw.

The evaluation reports (reports/eval_*.json) store, for every episode, its number of steps, its
success, its start, max and final coverage and a coverage trace rounded to 4 decimals. This tool
replays one seeded episode of a policy exactly as evaluate.run_episode does (the same seed, the same
policy reset, the same loop), collects a picture after the reset and after every step, and writes them
as a GIF with the step number and the coverage written above the picture.

If a report is given, the replay is also compared with the stored episode and the tool says whether
they match. A GIF whose episode does not match the report is not the episode the README's numbers
describe, and the tool then exits with status 1 (it still writes the file so it can be looked at).

Which episode: either an explicit --seed, or --pick together with --report. The pick rules are fixed
in advance so that the episodes shown are not chosen by eye:
  first-success   the successful episode with the lowest seed
  median-failure  among the failed episodes, ordered by max coverage (ties by seed), the lower median;
                  neither the best nor the worst failure

Pictures: --source render (default) uses the environment's own rendering, --source pixels uses the
96x96 picture the network was given, scaled up. Both are scaled to --width pixels.

Run:  python make_gif.py --policy chunk_cnn --checkpoint checkpoints/chunk_cnn_s11.pt \\
          --report reports/eval_chunk_cnn_s11.json --pick first-success
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import compare_policies as cp
import determinism_check as dc
import evaluate as ev

PICK_RULES = ("first-success", "median-failure")
SOURCES = ("render", "pixels")
HEADER_HEIGHT = 26
HOLD_MS = 1500
VALUE_TOLERANCE = 1e-6   # the report stores start, max and final coverage rounded to 9 decimals
TRACE_TOLERANCE = 1e-4   # the report stores the coverage trace rounded to 4 decimals


def pick_episode(report: dict, rule: str) -> dict:
    """The episode of an evaluation report chosen by one of the pre-registered rules."""
    episodes = report.get("episodes") or []
    if rule == "first-success":
        wins = sorted((e for e in episodes if e["success"]), key=lambda e: e["seed"])
        if not wins:
            raise ValueError("the report has no successful episode")
        return wins[0]
    if rule == "median-failure":
        fails = sorted((e for e in episodes if not e["success"]), key=lambda e: (e["max_coverage"], e["seed"]))
        if not fails:
            raise ValueError("the report has no failed episode")
        return fails[(len(fails) - 1) // 2]
    raise ValueError(f"unknown rule {rule!r}; choose from {PICK_RULES}")


def _frame(env, obs, source: str) -> np.ndarray:
    image = obs["pixels"] if source == "pixels" else env.render()
    if image is None:
        raise RuntimeError("the environment returned no picture; try --source pixels")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"expected an RGB picture, got an array of shape {array.shape}")
    return np.array(array, dtype=np.uint8)


def replay_episode(env, policy, seed: int, max_steps: int = ev.MAX_STEPS, source: str = "render") -> dict:
    """Replay one seeded episode as evaluate.run_episode does, keeping a picture after the reset
    (frame 0) and after every step (frame t is the picture after step t)."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {SOURCES}")
    obs, reset_info = env.reset(seed=seed)
    start_coverage = ev.read_start_coverage(env, reset_info)  # read now, before any step
    if hasattr(policy, "reset"):
        policy.reset(seed)
    frames = [_frame(env, obs, source)]
    trace, solved = [], []
    success = False
    terminated = truncated = False
    while len(trace) < max_steps and not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        trace.append(float(info["coverage"]))
        success = success or bool(info["is_success"])
        solved.append(success)
        frames.append(_frame(env, obs, source))
    return {
        "seed": int(seed),
        "steps": len(trace),
        "start_coverage": None if start_coverage is None else float(start_coverage),
        "coverage_trace": trace,
        "max_coverage": max(trace),
        "final_coverage": trace[-1],
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "solved": solved,
        "frames": frames,
    }


def compare_with_report(result: dict, episode: dict) -> list:
    """Why the replay differs from the stored episode; an empty list means they match."""
    problems = []
    if result["steps"] != episode["steps"]:
        problems.append(f"steps: {result['steps']} in the replay, {episode['steps']} in the report")
    if bool(result["success"]) != bool(episode["success"]):
        problems.append(f"success: {result['success']} in the replay, {episode['success']} in the report")
    for key in ("start_coverage", "max_coverage", "final_coverage"):
        stored, mine = episode.get(key), result.get(key)
        if stored is None or mine is None:
            continue
        if not abs(float(mine) - float(stored)) <= VALUE_TOLERANCE:
            problems.append(f"{key}: {float(mine):.6f} in the replay, {float(stored):.6f} in the report")
    stored_trace = episode.get("coverage_trace")
    if stored_trace is not None and len(stored_trace) == len(result["coverage_trace"]):
        for i, (mine, stored) in enumerate(zip(result["coverage_trace"], stored_trace)):
            if not abs(mine - stored) <= TRACE_TOLERANCE:
                problems.append(
                    f"coverage trace first differs at step {i + 1}: {mine:.4f} in the replay, {stored:.4f} in the report"
                )
                break
    return problems


def select_frames(n_frames: int, every: int) -> list:
    """Indices of the frames to keep: every `every`-th one, always including the first and the last."""
    if n_frames < 1:
        raise ValueError("n_frames must be at least 1")
    if every < 1:
        raise ValueError("every must be at least 1")
    keep = list(range(0, n_frames, every))
    if keep[-1] != n_frames - 1:
        keep.append(n_frames - 1)
    return keep


def draw_frame(array: np.ndarray, step: int, steps: int, coverage: float, solved: bool, width: int) -> Image.Image:
    """The picture scaled to `width` pixels, below a white header with the step and the coverage."""
    image = Image.fromarray(array)
    if width != image.width:
        height = max(1, round(image.height * width / image.width))
        resample = Image.Resampling.NEAREST if width > image.width else Image.Resampling.LANCZOS
        image = image.resize((width, height), resample)
    canvas = Image.new("RGB", (image.width, image.height + HEADER_HEIGHT), (255, 255, 255))
    canvas.paste(image, (0, HEADER_HEIGHT))
    text = f"step {step}/{steps}   coverage {coverage:.3f}" + ("   SOLVED" if solved else "")
    ImageDraw.Draw(canvas).text((6, 6), text, fill=(0, 0, 0))
    return canvas


def write_gif(images: list, path, ms: int = 60, hold_ms: int = HOLD_MS) -> None:
    """Write the images as a looping GIF, `ms` per frame and `hold_ms` on the last one."""
    if not images:
        raise ValueError("no images to write")
    durations = [ms] * len(images)
    durations[-1] = hold_ms
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)


def main(argv=None, *, env_factory=ev.make_eval_env, policy_builder=dc.build_policy) -> int:
    parser = argparse.ArgumentParser(description="Render one episode of a policy as a GIF")
    parser.add_argument("--policy", choices=dc.POLICY_KINDS, default="chunk_cnn")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--n-exec", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--pick", choices=PICK_RULES, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--ms", type=int, default=60)
    parser.add_argument("--hold-ms", type=int, default=HOLD_MS)
    parser.add_argument("--source", choices=SOURCES, default="render")
    parser.add_argument("--max-steps", type=int, default=ev.MAX_STEPS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        if (args.seed is None) == (args.pick is None):
            raise ValueError("give exactly one of --seed and --pick")
        if args.pick is not None and args.report is None:
            raise ValueError("--pick needs --report")
        if args.every < 1 or args.width < 8 or args.ms < 1 or args.hold_ms < 1:
            raise ValueError("--every, --ms and --hold-ms must be at least 1 and --width at least 8")
        report = cp.load_report(args.report) if args.report is not None else None
        episode = None
        if args.pick is not None:
            episode = pick_episode(report, args.pick)
            seed = int(episode["seed"])
        else:
            seed = args.seed
            if report is not None:
                found = [e for e in report["episodes"] if e["seed"] == seed]
                if not found:
                    raise ValueError(f"seed {seed} is not in the report")
                episode = found[0]
        checkpoint = args.checkpoint or dc.DEFAULT_CHECKPOINTS.get(args.policy)
        policy = policy_builder(args.policy, checkpoint, env_factory, args.n_exec)
        env = env_factory()
        try:
            result = replay_episode(env, policy, seed, args.max_steps, args.source)
        finally:
            env.close()
    except (ValueError, RuntimeError, OSError, cp.ComparisonError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    problems = [] if episode is None else compare_with_report(result, episode)
    start = result["start_coverage"]
    keep = select_frames(len(result["frames"]), args.every)
    images = []
    for i in keep:
        coverage = (0.0 if start is None else start) if i == 0 else result["coverage_trace"][i - 1]
        solved = i > 0 and result["solved"][i - 1]
        images.append(draw_frame(result["frames"][i], i, result["steps"], coverage, solved, args.width))
    stem = Path(checkpoint).stem if checkpoint else args.policy
    suffix = "" if args.n_exec is None else f"_x{args.n_exec}"
    out = args.out or Path("reports/gifs") / f"{stem}_seed{seed}{suffix}.gif"
    write_gif(images, out, args.ms, args.hold_ms)

    lines = [
        f"policy {args.policy}" + (f" ({checkpoint})" if checkpoint else "")
        + ("" if args.n_exec is None else f", executing {args.n_exec} targets per prediction")
        + f", starting position {seed}",
        f"episode: {result['steps']} steps, start coverage "
        + ("n/a" if start is None else f"{start:.3f}")
        + f", max coverage {result['max_coverage']:.3f}, final coverage {result['final_coverage']:.3f}, "
        + f"success {'yes' if result['success'] else 'no'}",
    ]
    if episode is None:
        lines.append("not compared with an evaluation report (none given)")
    elif problems:
        lines.append("does NOT match the evaluation report: " + "; ".join(problems))
    else:
        lines.append("matches the evaluation report: yes (steps, success, start, max and final coverage, "
                     "and the whole coverage trace)")
    lines.append(f"saved {out} ({len(images)} frames of {images[0].width} x {images[0].height} px, "
                 f"every {args.every} step(s))")
    print("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
