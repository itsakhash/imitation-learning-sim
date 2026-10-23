"""Evaluate a trained diffusion checkpoint on PushT with the project's protocol.

Mirrors evaluate_chunk.py: 50 episodes on seeds 10000 to 10049 by default, 300-step cap, the
same report (reports/eval_<name>.json). The noise of each episode is seeded from the episode's
evaluation seed by DiffusionPolicy.reset, so the evaluation is repeatable; it still has to be
run twice in separate processes and compared, like every evaluation of the study.

Inference runs on the CPU by default, like the other evaluators.

    python evaluate_diffusion.py --checkpoint checkpoints/diff_cnn.pt --name diff_cnn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import evaluate as ev
from diffusion_policy import DiffusionPolicy


def main(argv=None, *, env_factory=ev.make_eval_env) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a diffusion checkpoint on PushT")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/diff_cnn.pt"))
    parser.add_argument("--name", default="diff_cnn")
    parser.add_argument("--episodes", type=int, default=ev.MIN_EPISODES)
    parser.add_argument("--n-exec", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    policy, ckpt = DiffusionPolicy.from_checkpoint(args.checkpoint, n_exec=args.n_exec, device=args.device)
    n_exec = policy.n_exec
    out = args.out or Path("reports") / f"eval_{args.name}.json"

    env = env_factory()
    try:
        results = ev.evaluate_policy(policy, env, args.episodes)
    finally:
        env.close()

    report = ev.build_report(args.name, results)
    config = ckpt["config"]
    report["checkpoint"] = {
        "path": str(args.checkpoint),
        "config": config,
        "dataset_revision": ckpt["dataset_revision"],
        "offline_val_mae_px": ckpt["history"]["val_mae_px"][-1],
        "n_exec": int(n_exec),
        "sampling": {
            "method": "DDIM, eta 0",
            "steps": len(policy.timesteps),
            "timesteps": list(policy.timesteps),
            "clip_x0": policy.clip_x0,
            "noise": "CPU generator seeded with the episode's evaluation seed",
        },
    }
    ev.write_report(out, report)
    print(ev.format_summary(args.name, report["summary"]))
    print(
        f"offline validation MAE of the first action of this checkpoint (one sample per frame): "
        f"{report['checkpoint']['offline_val_mae_px']:.2f} px; executing {n_exec} of "
        f"{config['chunk_len']} sampled targets per chunk, {len(policy.timesteps)} DDIM steps"
    )
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
