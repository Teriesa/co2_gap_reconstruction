"""Unit tests for co2gaps.perf."""

import pytest

from co2gaps import perf


def test_time_call_returns_a_positive_median():
    """time_call runs the function and returns a non-negative time in seconds."""
    calls = []
    elapsed = perf.time_call(lambda: calls.append(1), repeat=3)
    assert len(calls) == 3
    assert elapsed >= 0


def test_time_call_rejects_zero_repeats():
    """At least one run is needed to time anything."""
    with pytest.raises(ValueError):
        perf.time_call(lambda: None, repeat=0)


def test_flop_counts_scale_linearly():
    """FLOPs grow in proportion to the number of slots or steps (O(n) work)."""
    assert perf.linear_flops(10) == 10 * perf.LINEAR_FLOPS_PER_SLOT
    assert perf.rk4_flops(200) == 2 * perf.rk4_flops(100)


def test_arithmetic_intensity():
    """Intensity is FLOPs per byte; linear interpolation is far below 1 FLOP/byte."""
    intensity = perf.arithmetic_intensity(perf.LINEAR_FLOPS_PER_SLOT, perf.LINEAR_BYTES_PER_SLOT)
    assert intensity == pytest.approx(5 / 16)
    with pytest.raises(ValueError):
        perf.arithmetic_intensity(1, 0)
