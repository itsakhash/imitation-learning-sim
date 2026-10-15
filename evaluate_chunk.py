"""Evaluate a trained action-chunking checkpoint with the standard harness.

Same protocol, same seeds (10000 + episode number) and same report format as evaluate.py
and evaluate_bc.py, so the result can be compared with every other policy. The policy
executes n_exec targets of each predicted chunk: the value stored in the checkpoint unless
--n-exec is given. Choosing n_exec by looking at results on the evaluation seeds would be an
ablation, and the report records the value used.

Run:  python evaluate_chunk.py --checkpoint checkpoints/chunk_cnn.pt --name chunk_cnn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import evaluate as ev
from chunk_policy import ChunkPolicy
from train_chunk import load_chunk_checkpoint


def main(argv=None, *, env_factory=ev.make_eval_env) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a chunked checkpoint on PushT"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/chunk_cnn.pt")
    )
    parser.add_argument("--name", default="chunk_cnn")
    parser.add_argument("--episodes", type=int, default=ev.MIN_EPISODES)
    parser.add_argument("--n-exec", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    model, ckpt = load_chunk_checkpoint(args.checkpoint, device=args.device)
    n_exec = ckpt["config"]["n_exec"] if args.n_exec is None else args.n_exec
    policy = ChunkPolicy(model, n_exec, device=args.device)
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
        "n_exec": int(n_exec),
    }
    ev.write_report(out, report)
    print(ev.format_summary(args.name, report["summary"]))
    print(
        f"offline validation MAE of the first action of this checkpoint: "
        f"{report['checkpoint']['offline_val_mae_px']:.2f} px; executing {n_exec} of "
        f"{ckpt['config']['chunk_len']} predicted targets per chunk"
    )
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
