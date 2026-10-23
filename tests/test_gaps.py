"""Unit tests for co2gaps.gaps and co2gaps.evaluate, using synthetic series."""

import numpy as np
import pandas as pd
import pytest

from co2gaps import evaluate, gaps

TZ = "Australia/Melbourne"
START = pd.Timestamp("2024-03-01 00:00", tz=TZ)


def _series(values, step_min=10, start=START):
    """Regular series with ``step_min`` spacing."""
    index = pd.date_range(start, periods=len(values), freq=f"{step_min}min")
    return pd.Series(np.asarray(values, dtype=float), index=index, name="co2_ppm")


def _daily_wave(days=10):
    """Smooth synthetic CO₂: 450 ppm plus a 100 ppm daily cycle, 10-minute samples."""
    t_h = np.arange(days * 144) / 6.0
    return _series(450 + 100 * np.sin(2 * np.pi * t_h / 24))


# --- split and masking -----------------------------------------------------------------

def test_split_dev_test_is_disjoint_and_ordered():
    """Development readings all come before the test start; none are lost."""
    series = _daily_wave(4)
    dev, test = gaps.split_dev_test(series, "2024-03-03")
    assert len(dev) + len(test) == len(series)
    assert dev.index.max() < pd.Timestamp("2024-03-03", tz=TZ) <= test.index.min()


def test_masked_gaps_are_repeatable_and_non_overlapping():
    """Same seed gives the same windows; windows do not overlap and have the set length."""
    series = _daily_wave(10)
    a = gaps.make_masked_gaps(series, gap_h=6, n_gaps=8, seed=3, context_h=12)
    b = gaps.make_masked_gaps(series, gap_h=6, n_gaps=8, seed=3, context_h=12)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 8
    assert ((a["end"] - a["start"]) == pd.Timedelta(hours=6)).all()
    assert (a["start"].iloc[1:].to_numpy() >= a["end"].iloc[:-1].to_numpy()).all()


def test_masked_gaps_leave_room_for_context():
    """Every window has at least context_h hours of series on both sides."""
    series = _daily_wave(10)
    windows = gaps.make_masked_gaps(series, gap_h=3, n_gaps=10, seed=0, context_h=24)
    assert (windows["start"] - series.index[0] >= pd.Timedelta(hours=24)).all()
    assert (series.index[-1] - windows["end"] >= pd.Timedelta(hours=24)).all()


def test_masked_gaps_avoid_real_gaps():
    """No window or its context may touch a real gap longer than max_real_gap_h."""
    series = _daily_wave(10)
    hole = (series.index >= START + pd.Timedelta(days=5)) & \
           (series.index < START + pd.Timedelta(days=5, hours=2))
    series = series[~hole]                                     # a real 2 h gap on day 5
    windows = gaps.make_masked_gaps(series, gap_h=2, n_gaps=50, seed=0, context_h=6)
    hole_start = START + pd.Timedelta(days=5)
    for start, end in windows.itertuples(index=False):
        near = (start - pd.Timedelta(hours=6) <= hole_start <= end + pd.Timedelta(hours=6))
        assert not near


def test_hide_splits_series_without_changing_it():
    """known + truth together are the original series, and the original is unchanged."""
    series = _daily_wave(2)
    copy = series.copy()
    start, end = START + pd.Timedelta(hours=5), START + pd.Timedelta(hours=7)
    known, truth = gaps.hide(series, start, end)
    assert len(truth) == 12 and len(known) + len(truth) == len(series)
    assert truth.index.min() >= start and truth.index.max() < end
    pd.testing.assert_series_equal(series, copy)


# --- interpolation ----------------------------------------------------------------------

@pytest.mark.parametrize("method", gaps.METHODS)
def test_every_method_reproduces_a_straight_line(method):
    """All four interpolants are exact for linear data (a basic correctness check)."""
    series = _series(400 + 2.0 * np.arange(200))              # +2 ppm per 10 min
    known, truth = gaps.hide(series, series.index[80], series.index[110])
    estimate = gaps.interpolate_gap(known, truth.index, method)
    np.testing.assert_allclose(estimate, truth.to_numpy(), atol=1e-8)


def test_pchip_does_not_overshoot_but_cubic_does():
    """On a step-like rise, PCHIP stays within the data range; a cubic spline overshoots."""
    values = np.r_[np.full(60, 450.0), np.full(60, 600.0)]
    series = _series(values)
    known = series.drop(series.index[[55, 57, 62, 64]])        # sparse points near step
    query = series.index[[55, 57, 62, 64]]
    pchip = gaps.interpolate_gap(known, query, "pchip")
    cubic = gaps.interpolate_gap(known, query, "cubic")
    assert pchip.min() >= 450 - 1e-9 and pchip.max() <= 600 + 1e-9
    assert cubic.min() < 450 or cubic.max() > 600


def test_interpolate_refuses_extrapolation():
    """A gap at the end of the record has no reading after it, so it is not filled."""
    series = _daily_wave(2)
    known = series.iloc[:-10]
    with pytest.raises(ValueError, match="extrapolation"):
        gaps.interpolate_gap(known, series.index[-10:], "linear")


def test_interpolate_refuses_gaps_over_limit():
    """A gap with more than 24 h of missing readings is not filled."""
    series = _daily_wave(6)
    known, truth = gaps.hide(series, series.index[200], series.index[200 + 150])  # 25 h
    with pytest.raises(ValueError, match="limit"):
        gaps.interpolate_gap(known, truth.index, "linear")


def test_interpolate_accepts_exactly_24_hours():
    """Exactly 24 h of missing readings (144 slots) is still within the limit."""
    series = _daily_wave(6)
    known, truth = gaps.hide(series, series.index[200], series.index[200 + 144])
    assert len(gaps.interpolate_gap(known, truth.index, "linear")) == 144


def test_interpolate_rejects_unknown_method():
    """A misspelt method name gives a clear error."""
    series = _daily_wave(2)
    known, truth = gaps.hide(series, series.index[50], series.index[60])
    with pytest.raises(ValueError, match="Unknown method"):
        gaps.interpolate_gap(known, truth.index, "quadratic")


def test_missing_slots_between_two_readings():
    """Readings 70 min apart have six missing 10-minute slots between them."""
    slots = gaps.missing_slots(START, START + pd.Timedelta(minutes=70))
    assert len(slots) == 6
    assert slots[0] == START + pd.Timedelta(minutes=10)
    assert slots[-1] == START + pd.Timedelta(minutes=60)


def test_fill_gaps_fills_short_and_reports_long():
    """A 2 h gap is filled; a 30 h gap is left empty and reported as too long."""
    series = _daily_wave(8)
    short = (series.index >= START + pd.Timedelta(days=2)) & \
            (series.index < START + pd.Timedelta(days=2, hours=2))
    long_ = (series.index >= START + pd.Timedelta(days=5)) & \
            (series.index < START + pd.Timedelta(days=6, hours=6))
    with_gaps = series[~short & ~long_]
    from co2gaps import io
    found = io.find_gaps(with_gaps)
    filled, report = gaps.fill_gaps(with_gaps, "linear", found)
    assert report["status"].tolist() == ["filled", "too long"]
    assert report["n_filled"].tolist() == [12, 0]
    assert filled["filled"].sum() == 12
    assert filled.index.is_monotonic_increasing


# --- metrics and comparison -------------------------------------------------------------

def test_metrics_on_known_values():
    """RMSE, MAE and overshoot match hand calculations."""
    truth = [400.0, 410.0, 420.0]
    estimate = [403.0, 406.0, 420.0]             # errors +3, −4, 0
    assert evaluate.mae(estimate, truth) == pytest.approx(7 / 3)
    assert evaluate.rmse(estimate, truth) == pytest.approx(np.sqrt(25 / 3))
    assert evaluate.overshoot(estimate, truth) == 0.0
    assert evaluate.overshoot([395.0, 430.0], truth) == pytest.approx(10.0)


def test_compare_methods_scores_every_window_and_method():
    """One row per (gap length, window, method); errors on smooth data are small."""
    series = _daily_wave(10)
    results = evaluate.compare_methods(series, [1, 3], n_gaps=5, seed=0)
    assert len(results) == 2 * 5 * len(gaps.METHODS)
    assert set(results["method"]) == set(gaps.METHODS)
    pchip_1h = results.query("method == 'pchip' and gap_h == 1")["rmse"]
    assert (pchip_1h < 1.0).all()                # sine wave: interpolation is near-exact
    tables = evaluate.summarise(results)
    assert list(tables["rmse"].columns) == [1, 3]


def test_compare_methods_accepts_extra_estimators():
    """A custom estimator (e.g. a baseline) is scored alongside the interpolants."""
    series = _daily_wave(10)
    constant = {"constant_450": lambda known, query: np.full(len(query), 450.0)}
    results = evaluate.compare_methods(series, [2], n_gaps=3, seed=0, methods=("linear",),
                                       estimators=constant)
    assert set(results["method"]) == {"linear", "constant_450"}
