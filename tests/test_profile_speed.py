"""Tests for profile_speed.py: the pure helpers and the stand-in network (CPU only).

The simulator and GPU parts need gym-pusht and a CUDA device and are exercised by running
`python profile_speed.py` on the training machine, not here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import profile_speed as ps


def test_summarize_ms_basic():
    s = ps.summarize_ms([0.001, 0.002, 0.003, 0.004, 0.005])
    assert s["n"] == 5
    assert s["median_ms"] == pytest.approx(3.0)
    assert s["mean_ms"] == pytest.approx(3.0)
    assert s["p95_ms"] == pytest.approx(5.0)


def test_summarize_ms_ignores_input_order():
    a = ps.summarize_ms([0.005, 0.001, 0.003])
    b = ps.summarize_ms([0.001, 0.003, 0.005])
    assert a == b


def test_summarize_ms_empty_raises():
    with pytest.raises(ValueError):
        ps.summarize_ms([])


def test_steps_per_second():
    assert ps.steps_per_second(600, 2.0) == pytest.approx(300.0)


@pytest.mark.parametrize("seconds", [0.0, -1.0])
def test_steps_per_second_rejects_nonpositive_time(seconds):
    with pytest.raises(ValueError):
        ps.steps_per_second(10, seconds)


def test_project_minutes():
    # 25 ms for 2,430 steps is 60.75 s, which is 1.0125 min
    assert ps.project_minutes(25.0, 2430) == pytest.approx(1.0125)


def test_scaling_rows_speedup_and_efficiency():
    rows = ps.scaling_rows({4: 300.0, 1: 100.0, 2: 190.0})
    assert [r["workers"] for r in rows] == [1, 2, 4]
    assert rows[0]["speedup"] == pytest.approx(1.0)
    assert rows[0]["efficiency"] == pytest.approx(1.0)
    assert rows[1]["speedup"] == pytest.approx(1.9)
    assert rows[1]["efficiency"] == pytest.approx(0.95)
    assert rows[2]["speedup"] == pytest.approx(3.0)
    assert rows[2]["efficiency"] == pytest.approx(0.75)


def test_scaling_rows_needs_one_worker_reference():
    with pytest.raises(ValueError):
        ps.scaling_rows({2: 190.0, 4: 300.0})


def test_parse_utilization():
    assert ps.parse_utilization("37\n") == 37.0
    assert ps.parse_utilization("55\n12\n") == 55.0
    assert ps.parse_utilization("  8 , extra") == 8.0
    assert ps.parse_utilization("") is None
    assert ps.parse_utilization("not a number") is None


def test_summarize_utilization():
    assert ps.summarize_utilization([]) is None
    s = ps.summarize_utilization([10.0, 30.0, 50.0])
    assert s == {"mean": pytest.approx(30.0), "max": 50.0, "samples": 3}


def test_stand_in_shape_and_size():
    torch = pytest.importorskip("torch")
    model = ps.build_stand_in()
    out = model(torch.rand(4, 3, 96, 96), torch.rand(4, 2))
    assert out.shape == (4, 2)
    # similar in size to the baseline's 718,178 parameters, as the module docstring says
    assert 600_000 < ps.count_parameters(model) < 760_000


def test_stand_in_learns_a_fixed_batch():
    """Deliberate-break check: if the model or the loss wiring is broken, the loss does not fall."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = ps.build_stand_in()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    image, pos, target = torch.rand(8, 3, 96, 96), torch.rand(8, 2), torch.rand(8, 2)
    first = loss_fn(model(image, pos), target).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = loss_fn(model(image, pos), target)
        loss.backward()
        optimizer.step()
    assert loss.item() < first
