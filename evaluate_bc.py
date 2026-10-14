"""Evaluate a trained behavior-cloning checkpoint with the standard harness.

Same protocol, same seeds (10000 + episode number) and same report format as
evaluate.py, so the result can be compared with the random and do-nothing floors.
Inference runs on the CPU by default so that the result does not depend on GPU kernels.

Run:  python evaluate_bc.py --episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import evaluate as ev
from bc_policy import BCPolicy
from train_bc import load_checkpoint


def main(argv=None, *, env_factory=ev.make_eval_env) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a BC checkpoint on PushT")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/bc_cnn.pt")
    )
    parser.add_argument("--name", default="bc_cnn")
    parser.add_argument("--episodes", type=int, default=ev.MIN_EPISODES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    model, ckpt = load_checkpoint(args.checkpoint, device=args.device)
    policy = BCPolicy(model, device=args.device)
    out = args.out or Path("reports") / f"eval_{args.name}.json"

    env = env_factory()
    try:
        results = ev.evaluate_policy(policy, env, args.episodes)
    finally:
        env.close()

    report = ev.build_report(args.name, results)
    report["checkpoint"] = {
        "path": str(args.checkpoint),
        "config": ckpt["config"],
        "dataset_revision": ckpt["dataset_revision"],
        "offline_val_mae_px": ckpt["history"]["val_mae_px"][-1],
    }
    ev.write_report(out, report)
    print(ev.format_summary(args.name, report["summary"]))
    print(
        f"offline validation MAE of this checkpoint: "
        f"{report['checkpoint']['offline_val_mae_px']:.2f} px"
    )
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
