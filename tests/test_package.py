"""Smoke tests: the package imports and its declared constants are sensible."""

import co2gaps
from co2gaps import io, gaps


def test_package_has_version():
    """The package exposes a version string."""
    assert isinstance(co2gaps.__version__, str)


def test_co2_valid_range_is_ordered():
    """The CO₂ validity range has a lower bound below its upper bound."""
    low, high = io.CO2_VALID_RANGE_PPM
    assert 0 < low < high


def test_sampling_interval_positive():
    """The nominal sensor sampling interval is a positive number of minutes."""
    assert gaps.SAMPLE_INTERVAL_MIN > 0
