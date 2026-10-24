"""Unit tests for co2gaps.baselines, using synthetic series with a known daily pattern."""

import numpy as np
import pandas as pd
import pytest

from co2gaps import baselines, gaps

TZ = "Australia/Melbourne"
START = pd.Timestamp("2024-03-04 00:00", tz=TZ)          # a Monday


def _pattern(days=14, offset=0.0):
    """CO₂ = 450 ppm + 100 ppm on weekdays 09:00–17:00 (+ offset), 10-minute samples."""
    index = pd.date_range(START, periods=days * 144, freq="10min")
    working = (index.dayofweek < 5) & (index.hour >= 9) & (index.hour < 17)
    return pd.Series(450.0 + 100.0 * working + offset, index=index, name="co2_ppm")


def test_typical_day_learns_hour_and_day_type():
    """The fitted table has 24 hours × 2 day types and recovers the pattern exactly."""
    profile = baselines.TypicalDay().fit(_pattern())
    assert len(profile.table_) == 48
    monday_10am = pd.DatetimeIndex([START + pd.Timedelta(hours=10)])
    saturday_10am = pd.DatetimeIndex([START + pd.Timedelta(days=5, hours=10)])
    assert profile.predict(monday_10am)[0] == pytest.approx(550.0)
    assert profile.predict(saturday_10am)[0] == pytest.approx(450.0)


def test_typical_day_uses_fallback_for_unseen_hours():
    """An (hour, day type) never seen in training gets the overall training median."""
    train = _pattern(days=5)                               # weekdays only: no weekend rows
    profile = baselines.TypicalDay().fit(train)
    saturday = pd.DatetimeIndex([START + pd.Timedelta(days=5, hours=3)])
    assert profile.predict(saturday)[0] == pytest.approx(train.median())


def test_mean_estimator_predicts_training_mean():
    """The DummyRegressor baseline returns the training mean for every query."""
    train = _pattern()
    estimate = baselines.mean_estimator(train)(train, train.index[:5])
    np.testing.assert_allclose(estimate, train.mean())


def test_anomaly_estimator_matches_edges_and_follows_pattern():
    """With a constant offset from the typical day, the anomaly method is exact.

    The test week sits 30 ppm above the training pattern. The gap (06:00–20:00) hides the
    whole working-day rise, so linear interpolation of the raw values misses it by
    100 ppm; the anomaly method adds the learned rise back and recovers every value.
    """
    train = _pattern(days=14)
    profile = baselines.TypicalDay().fit(train)
    later = _pattern(days=21, offset=30.0).iloc[14 * 144:]  # the following week, +30 ppm
    start = later.index[0] + pd.Timedelta(days=2, hours=6)  # Wednesday 06:00
    known, truth = gaps.hide(later, start, start + pd.Timedelta(hours=14))  # 06:00-20:00
    estimate = baselines.anomaly_estimator(profile)(known, truth.index)
    np.testing.assert_allclose(estimate, truth.to_numpy(), atol=1e-9)
    raw_linear = gaps.interpolate_gap(known, truth.index, "linear", max_gap_h=np.inf)
    assert np.abs(raw_linear - truth.to_numpy()).max() == pytest.approx(100.0)


def test_make_estimators_returns_three_named_methods():
    """make_estimators gives the three baselines under fixed names."""
    estimators = baselines.make_estimators(_pattern())
    assert set(estimators) == {"mean", "typical_day", "typical_day_anomaly"}
