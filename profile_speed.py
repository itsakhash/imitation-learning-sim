"""Speed profile: where does the time go in training and evaluation?

Measures, on this machine and at this moment:
  1. one simulator step and reset (gym-pusht, random actions, one process);
  2. how simulator throughput scales when several processes run at the same time;
  3. one training step of a stand-in network (batch 256 and 1024, data already on the GPU,
     and with a host-to-GPU copy every step), with GPU utilization sampled from nvidia-smi;
  4. policy inference on a batch of one, on the GPU and on one CPU thread.

The network is a STAND-IN with a similar shape to the baseline (4 convolutions, 3 fully
connected layers, about 0.66 million parameters, a 96x96 image plus the agent position in,
2 numbers out) trained on synthetic random data. It is not the project's network, so its
timings show the kind of bottleneck, not exact per-network costs; the real networks are timed
in the pilot.

Close games and other heavy programs first: the numbers describe whatever else the machine is
doing. Takes about two minutes. Run:  python profile_speed.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPORT_DEFAULT = Path("reports/profile_speed.json")
EPOCH_STEPS = 2430  # about 30 epochs of the baseline recipe (81 steps per epoch)
BUDGET_STEPS = 30000  # the proposed diffusion budget


# ---------------------------------------------------------------- pure helpers (tested)


def summarize_ms(samples_s) -> dict:
    """Median, mean and 95th percentile, in milliseconds, of timings given in seconds."""
    if len(samples_s) == 0:
        raise ValueError("no samples")
    ordered = sorted(samples_s)
    n = len(ordered)
    p95 = ordered[min(n - 1, int(round(0.95 * (n - 1))))]
    return {
        "n": n,
        "median_ms": statistics.median(ordered) * 1000.0,
        "mean_ms": statistics.fmean(ordered) * 1000.0,
        "p95_ms": p95 * 1000.0,
    }


def steps_per_second(steps: int, seconds: float) -> float:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return steps / seconds


def project_minutes(ms_per_step: float, steps: int) -> float:
    """Minutes needed for `steps` steps at `ms_per_step` milliseconds each."""
    return ms_per_step * steps / 60000.0


def scaling_rows(throughput_by_workers: dict) -> list:
    """Speedup and efficiency against the one-worker throughput."""
    if 1 not in throughput_by_workers:
        raise ValueError("a 1-worker measurement is needed as the reference")
    base = throughput_by_workers[1]
    rows = []
    for workers in sorted(throughput_by_workers):
        throughput = throughput_by_workers[workers]
        speedup = throughput / base
        rows.append(
            {
                "workers": workers,
                "steps_per_s": throughput,
                "speedup": speedup,
                "efficiency": speedup / workers,
            }
        )
    return rows


def parse_utilization(text: str):
    """First number of nvidia-smi's csv output (GPU utilization in percent), or None."""
    for line in text.strip().splitlines():
        try:
            return float(line.split(",")[0].strip())
        except ValueError:
            continue
    return None


def summarize_utilization(values) -> dict | None:
    if not values:
        return None
    return {"mean": statistics.fmean(values), "max": max(values), "samples": len(values)}


# ---------------------------------------------------------------- stand-in network


def build_stand_in():
    """4 convolutions + 3 fully connected layers; (image, agent position) -> 2 numbers."""
    import torch
    from torch import nn

    class StandIn(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 2),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.head = nn.Sequential(
                nn.Linear(128 * 5 * 5 + 2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        def forward(self, image, pos):
            return self.head(torch.cat([self.convs(image), pos], dim=1))

    return StandIn()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- simulator


def make_env():
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the environment)

    return gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")


def time_sim_steps(steps: int, seed: int):
    env = make_env()
    env.action_space.seed(seed)
    env.reset(seed=seed)
    step_times, reset_times = [], []
    for _ in range(steps):
        action = env.action_space.sample()
        t0 = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(action)
        step_times.append(time.perf_counter() - t0)
        if terminated or truncated:
            t0 = time.perf_counter()
            env.reset()
            reset_times.append(time.perf_counter() - t0)
    env.close()
    return step_times, reset_times


def _scaling_worker(barrier, queue, steps: int, seed: int) -> None:
    try:
        env = make_env()
        env.action_space.seed(seed)
        env.reset(seed=seed)
        for _ in range(20):  # warm-up, not timed
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()
        barrier.wait(timeout=120)
        t0 = time.perf_counter()
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()
        elapsed = time.perf_counter() - t0
        env.close()
        queue.put((steps, elapsed, None))
    except Exception as error:  # reported to the parent instead of hanging it
        queue.put((0, 0.0, f"{type(error).__name__}: {error}"))


def measure_scaling(workers_list, steps_per_worker: int) -> dict:
    ctx = mp.get_context("spawn")
    result = {}
    for n in workers_list:
        barrier = ctx.Barrier(n)
        queue = ctx.Queue()
        procs = [
            ctx.Process(target=_scaling_worker, args=(barrier, queue, steps_per_worker, 1000 + i))
            for i in range(n)
        ]
        for p in procs:
            p.start()
        got = [queue.get(timeout=300) for _ in procs]
        for p in procs:
            p.join()
        errors = [g[2] for g in got if g[2] is not None]
        if errors:
            raise RuntimeError(f"a worker failed: {errors[0]}")
        total_steps = sum(g[0] for g in got)
        wall = max(g[1] for g in got)
        result[n] = steps_per_second(total_steps, wall)
    return result


# ---------------------------------------------------------------- GPU


class GpuSampler(threading.Thread):
    """Samples GPU utilization from nvidia-smi in the background."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.values: list = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
            except (OSError, subprocess.SubprocessError):
                return
            value = parse_utilization(out)
            if value is not None:
                self.values.append(value)
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
        self.join()


def time_training(batch_size: int, steps: int, warmup: int, data_on_gpu: bool) -> list:
    import torch

    device = torch.device("cuda")
    model = build_stand_in().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(0)
    image = torch.rand(batch_size, 3, 96, 96, generator=gen)
    pos = torch.rand(batch_size, 2, generator=gen)
    target = torch.rand(batch_size, 2, generator=gen)
    if data_on_gpu:
        image, pos, target = image.to(device), pos.to(device), target.to(device)
    times = []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if data_on_gpu:
            img_d, pos_d, tgt_d = image, pos, target
        else:
            img_d, pos_d, tgt_d = image.to(device), pos.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(img_d, pos_d), tgt_d)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return times


def time_inference(device_name: str, samples: int, warmup: int) -> list:
    import torch

    if device_name == "cpu":
        torch.set_num_threads(1)
    device = torch.device(device_name)
    model = build_stand_in().to(device).eval()
    times = []
    with torch.no_grad():
        for i in range(warmup + samples):
            image = torch.rand(1, 3, 96, 96)
            pos = torch.rand(1, 2)
            if device_name == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(image.to(device), pos.to(device)).cpu()
            if device_name == "cuda":
                torch.cuda.synchronize()
            _ = out.numpy()
            if i >= warmup:
                times.append(time.perf_counter() - t0)
    return times


# ---------------------------------------------------------------- report


def machine_info() -> dict:
    from importlib import metadata

    def version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not installed"

    info = {"cpu_count_logical": os.cpu_count(), "python": sys.version.split()[0]}
    for name in ("torch", "gymnasium", "gym-pusht", "numpy", "pymunk"):
        info[name] = version(name)
    return info


def line(label: str, s: dict) -> str:
    return (
        f"{label}: median {s['median_ms']:.2f} ms, mean {s['mean_ms']:.2f} ms, "
        f"p95 {s['p95_ms']:.2f} ms (n={s['n']})"
    )


def profile_gpu(train_steps: int, warmup: int, infer_samples: int) -> dict:
    import torch

    out: dict = {
        "device": torch.cuda.get_device_name(0),
        "stand_in_parameters": count_parameters(build_stand_in()),
    }
    sampler = GpuSampler()
    sampler.start()
    out["train_b256_data_on_gpu"] = summarize_ms(time_training(256, train_steps, warmup, True))
    sampler.stop()
    out["gpu_utilization_during_b256"] = summarize_utilization(sampler.values)
    out["train_b256_copy_each_step"] = summarize_ms(time_training(256, train_steps, warmup, False))
    out["train_b1024_data_on_gpu"] = summarize_ms(time_training(1024, train_steps, warmup, True))
    out["inference_b1_gpu"] = summarize_ms(time_inference("cuda", infer_samples, warmup))
    out["inference_b1_cpu_1thread"] = summarize_ms(time_inference("cpu", infer_samples, warmup))
    return out


def print_gpu(gpu: dict) -> None:
    print(f"\n== GPU: {gpu['device']}, stand-in network with {gpu['stand_in_parameters']:,} parameters ==")
    for key, label in (
        ("train_b256_data_on_gpu", "training step, batch 256, data on GPU  "),
        ("train_b256_copy_each_step", "training step, batch 256, copy each step"),
        ("train_b1024_data_on_gpu", "training step, batch 1024, data on GPU "),
    ):
        print(line(label, gpu[key]))
    m256 = gpu["train_b256_data_on_gpu"]["median_ms"]
    m1024 = gpu["train_b1024_data_on_gpu"]["median_ms"]
    print(f"4x the data per step costs {m1024 / m256:.2f}x the time (4.00x = GPU-bound, near 1.00x = overhead-bound)")
    print(
        f"at the batch-256 median: {EPOCH_STEPS:,} steps = {project_minutes(m256, EPOCH_STEPS):.2f} min, "
        f"{BUDGET_STEPS:,} steps = {project_minutes(m256, BUDGET_STEPS):.1f} min (stand-in, not the real network)"
    )
    util = gpu["gpu_utilization_during_b256"]
    if util is None:
        print("GPU utilization: nvidia-smi gave no readings")
    else:
        print(f"GPU utilization during the batch-256 run: mean {util['mean']:.0f}%, max {util['max']:.0f}% ({util['samples']} samples)")
    print(line("inference, batch of one, GPU            ", gpu["inference_b1_gpu"]))
    print(line("inference, batch of one, CPU (1 thread)  ", gpu["inference_b1_cpu_1thread"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Where does the time go? Timing profile.")
    parser.add_argument("--sim-steps", type=int, default=600)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--infer-samples", type=int, default=200)
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--out", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args(argv)

    report: dict = {"machine": machine_info()}
    print("== machine ==")
    for key, value in report["machine"].items():
        print(f"{key}: {value}")

    if not args.skip_sim:
        step_times, reset_times = time_sim_steps(args.sim_steps, seed=0)
        report["sim_step"] = summarize_ms(step_times)
        print("\n== simulator, one process, random actions ==")
        print(line("step ", report["sim_step"]))
        if reset_times:
            report["sim_reset"] = summarize_ms(reset_times)
            print(line("reset", report["sim_reset"]))
        throughput = measure_scaling(args.workers, args.sim_steps)
        report["sim_scaling"] = scaling_rows(throughput)
        print("\n== simulator throughput with several processes ==")
        print("workers  steps/s  speedup  efficiency")
        for row in report["sim_scaling"]:
            print(f"{row['workers']:>7}  {row['steps_per_s']:>7.0f}  {row['speedup']:>6.2f}x  {row['efficiency']:>9.0%}")

    if not args.skip_gpu:
        import torch

        if torch.cuda.is_available():
            report["gpu"] = profile_gpu(args.train_steps, args.warmup, args.infer_samples)
            print_gpu(report["gpu"])
        else:
            print("\nno CUDA device available: GPU part skipped")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
