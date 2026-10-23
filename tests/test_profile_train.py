"""Tests for profile_train.py: the pure helpers (no torch, no GPU, no dataset needed).

The real-loop timings need the dataset and a CUDA device and are exercised by running
`python profile_train.py` on the training machine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import profile_train as pt


class FakeClock:
    """Advances by a scripted amount on every pair of calls, so timings are exact."""

    def __init__(self, durations):
        self.durations = list(durations)
        self.now = 0.0
        self.calls = 0

    def __call__(self):
        # calls alternate: start of a next(), then end of it
        if self.calls % 2 == 1:
            self.now += self.durations[self.calls // 2]
        self.calls += 1
        return self.now


def test_time_iterator_skips_warmup_and_times_each_call():
    clock = FakeClock([9.0, 9.0, 1.0, 2.0, 3.0])
    times = pt.time_iterator(lambda: iter(range(100)), warmup=2, measure=3, clock=clock)
    assert times == pytest.approx([1.0, 2.0, 3.0])


def test_time_iterator_consumes_warmup_plus_measure_items():
    consumed = []

    def make():
        for i in range(100):
            consumed.append(i)
            yield i

    pt.time_iterator(make, warmup=4, measure=6, clock=FakeClock([0.0] * 10))
    assert len(consumed) == 10


@pytest.mark.parametrize("warmup,measure", [(-1, 3), (0, 0)])
def test_time_iterator_rejects_bad_counts(warmup, measure):
    with pytest.raises(ValueError):
        pt.time_iterator(lambda: iter(range(5)), warmup=warmup, measure=measure)


def test_epoch_seconds_differences():
    assert pt.epoch_seconds([4.0, 6.5, 9.0]) == pytest.approx([4.0, 2.5, 2.5])
    assert pt.epoch_seconds([3.0]) == pytest.approx([3.0])


def test_epoch_seconds_empty_raises():
    with pytest.raises(ValueError):
        pt.epoch_seconds([])


def test_steps_per_epoch_rounds_up():
    assert pt.steps_per_epoch(20684, 256) == 81  # the real training split
    assert pt.steps_per_epoch(256, 256) == 1
    assert pt.steps_per_epoch(257, 256) == 2


@pytest.mark.parametrize("n,b", [(0, 256), (10, 0)])
def test_steps_per_epoch_rejects_nonpositive(n, b):
    with pytest.raises(ValueError):
        pt.steps_per_epoch(n, b)


def test_ratio():
    assert pt.ratio(25.0, 5.0) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        pt.ratio(1.0, 0.0)
