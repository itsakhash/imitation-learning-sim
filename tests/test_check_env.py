import pytest
import torch

import check_env


def test_collect_info_has_expected_keys():
    info = check_env.collect_info()
    for key in (
        "python",
        "torch",
        "torch_cuda_build",
        "cuda_available",
        "in_venv",
        "ros_on_path",
        "arch_list",
    ):
        assert key in info


def test_runs_inside_venv():
    assert check_env.collect_info()["in_venv"] is True


def test_ros_not_on_python_path():
    assert check_env.collect_info()["ros_on_path"] == []


def test_cuda_available():
    assert check_env.collect_info()["cuda_available"] is True


def test_build_supports_this_gpu():
    assert check_env.arch_supported(check_env.collect_info()) is True


def test_gpu_matmul_matches_cpu():
    assert check_env.gpu_matmul_matches_cpu() is True


@pytest.mark.parametrize(
    "capability, arch_list, expected",
    [
        ((12, 0), ["sm_90", "sm_120"], True),
        ((12, 0), ["sm_90"], False),
        (None, ["sm_120"], False),
    ],
)
def test_arch_supported_logic(capability, arch_list, expected):
    info = {"gpu_capability": capability, "arch_list": arch_list}
    assert check_env.arch_supported(info) is expected


def test_matmul_check_is_false_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert check_env.gpu_matmul_matches_cpu() is False
