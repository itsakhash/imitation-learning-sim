"""Milestone 0 environment check.

Reports the versions in use and confirms PyTorch can run real work on the GPU.
Run:  python check_env.py   (exit code 0 = all good, 1 = something is wrong)
"""

import platform
import sys

import torch


def collect_info() -> dict:
    """Gather version and GPU facts. Does not raise if CUDA is missing."""
    cuda_ok = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "in_venv": sys.prefix != sys.base_prefix,
        "ros_on_path": [p for p in sys.path if "/opt/ros" in p],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_ok,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_ok else None,
        "gpu_capability": torch.cuda.get_device_capability(0) if cuda_ok else None,
        "arch_list": torch.cuda.get_arch_list(),
    }


def arch_supported(info: dict) -> bool:
    """True if this PyTorch build contains kernels for the GPU's architecture."""
    cap = info["gpu_capability"]
    if cap is None:
        return False
    return f"sm_{cap[0]}{cap[1]}" in info["arch_list"]


def gpu_matmul_matches_cpu(size: int = 512, seed: int = 0) -> bool:
    """Run a real matrix multiply on the GPU and compare it with the CPU result."""
    if not torch.cuda.is_available():
        return False
    torch.manual_seed(seed)
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    cpu = a @ b
    gpu = (a.cuda() @ b.cuda()).cpu()
    return bool(torch.allclose(cpu, gpu, atol=1e-3, rtol=1e-3))


def main() -> int:
    info = collect_info()
    for key, value in info.items():
        print(f"{key:18s} {value}")

    checks = {
        "running inside a venv": info["in_venv"],
        "no ROS folders on sys.path": not info["ros_on_path"],
        "CUDA available to PyTorch": info["cuda_available"],
        "build supports this GPU": arch_supported(info),
        "GPU matmul matches CPU": gpu_matmul_matches_cpu(),
    }
    print()
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
