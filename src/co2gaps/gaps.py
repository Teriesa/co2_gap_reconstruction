"""Create artificial (masked) gaps and reconstruct missing CO₂ readings.

Validation idea: a real gap has no true values, so accuracy is measured on data where
the truth is known. Readings are hidden from a nearly complete record, reconstructed
from the readings that remain, and compared with the hidden values.

Reconstruction methods (Weeks 3.1 and 3.4):

* ``"linear"``: piecewise-linear interpolation (``make_interp_spline`` with k = 1).
* ``"pchip"``: shape-preserving cubic Hermite interpolation; never overshoots the data.
* ``"akima"``: cubic interpolation built from local slopes.
* ``"cubic"``: cubic spline with SciPy's default "not-a-knot" end conditions; smooth,
  but can overshoot between data points.

Typical use::

    windows = make_masked_gaps(co2, gap_h=6, n_gaps=40, seed=1)
    known, truth = hide(co2, *windows.iloc[0])
    estimate = interpolate_gap(known, truth.index, "pchip")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import (Akima1DInterpolator, CubicSpline, PchipInterpolator,
                               make_interp_spline)

SAMPLE_INTERVAL_MIN = 10
"""Nominal sampling interval of the environmental sensors (minutes)."""

MAX_FILL_GAP_H = 24.0
"""Longest gap the project fills, measured as hours of missing readings (missed slots ×
10 min). No method is validated on longer gaps."""

CONTEXT_H = 48.0
"""Hours of known readings used on each side of a gap to build the interpolant."""

MAX_REAL_GAP_H = 0.5
"""A masked window and its context may contain real gaps up to this length (hours) only,
so that the known readings around a masked gap are themselves reliable."""

METHODS = ("linear", "pchip", "akima", "cubic")
"""Names of the interpolation methods accepted by :func:`interpolate_gap`."""


def _hours(index, origin):
    """Hours from ``origin`` to each timestamp in ``index`` (any datetime resolution)."""
    return ((index - origin) / pd.Timedelta(hours=1)).to_numpy(dtype=float)


def _missing_hours(last_before, first_after, interval_min=SAMPLE_INTERVAL_MIN):
    """Hours of missing readings between two readings: (number of missed slots) × interval.

    Two readings 24 h 10 min apart have 144 missed 10-minute slots, i.e. 24 h missing.
    Rounding to whole slots makes the count insensitive to small timing jitter.
    """
    n_missed = int(np.rint((first_after - last_before) / pd.Timedelta(minutes=interval_min))) - 1
    return max(n_missed, 0) * interval_min / 60.0


def split_dev_test(series, test_start):
    """Split a series in time into a development part and a test part.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed readings.
    test_start : str or pandas.Timestamp
        First time of the test period. Strings are read as local time of the series.

    Returns
    -------
    dev, test : pandas.Series
        Readings before ``test_start`` and from ``test_start`` on. No reading is in both.
    """
    start = pd.Timestamp(test_start)
    if start.tzinfo is None and series.index.tz is not None:
        start = start.tz_localize(series.index.tz)
    return series[series.index < start], series[series.index >= start]


def make_masked_gaps(series, gap_h, n_gaps, seed=0, context_h=CONTEXT_H,
                     max_real_gap_h=MAX_REAL_GAP_H, max_tries=10_000):
    """Choose random, non-overlapping windows of known data to hide.

    A window starts at a randomly chosen reading and covers ``gap_h`` hours. It is
    accepted only if the window and ``context_h`` hours on each side lie inside the
    series and contain no real gap longer than ``max_real_gap_h``, and if it does not
    overlap a window already chosen.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed readings to hide data from (e.g. the development period).
    gap_h : float
        Length of each artificial gap in hours.
    n_gaps : int
        Number of windows wanted.
    seed : int
        Seed of the random generator, so results are repeatable.
    context_h, max_real_gap_h : float
        See the module constants :data:`CONTEXT_H` and :data:`MAX_REAL_GAP_H`.
    max_tries : int
        Number of random starts to try before giving up.

    Returns
    -------
    pandas.DataFrame
        Columns ``start`` and ``end`` (timestamps), sorted by ``start``. Readings with
        ``start <= time < end`` are the ones to hide. May hold fewer than ``n_gaps`` rows
        if not enough valid windows exist.
    """
    rng = np.random.default_rng(seed)
    times = series.index
    t = _hours(times, times[0])
    step_h = np.diff(t)
    # Cumulative count of "too long" intervals lets us test any time range in O(1).
    bad = np.concatenate([[0], np.cumsum(step_h > max_real_gap_h)])
    chosen = []
    for _ in range(max_tries):
        if len(chosen) == n_gaps:
            break
        i0 = int(rng.integers(len(t)))
        lo_t, hi_t = t[i0] - context_h, t[i0] + gap_h + context_h
        if lo_t < t[0] or hi_t > t[-1]:
            continue                                   # window + context must fit
        lo, hi = np.searchsorted(t, lo_t), np.searchsorted(t, hi_t, side="right") - 1
        # Intervals lo-1 ... hi touch [lo_t, hi_t] (the first and last ones cross its
        # ends), so all of them must be short.
        if bad[min(hi + 1, len(t) - 1)] - bad[max(lo - 1, 0)] > 0:
            continue                                   # a real gap is too close
        start_t, end_t = t[i0], t[i0] + gap_h
        if any(start_t < e and s < end_t for s, e, _ in chosen):
            continue                                   # overlaps an earlier window
        chosen.append((start_t, end_t, i0))
    chosen.sort()
    length = pd.Timedelta(hours=gap_h)
    starts = [times[i0] for _, _, i0 in chosen]        # each window starts at a reading
    return pd.DataFrame({"start": starts, "end": [s + length for s in starts]})


def hide(series, start, end):
    """Hide the readings in ``[start, end)`` and return what is left and what was hidden.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed readings.
    start, end : pandas.Timestamp
        Window to hide.

    Returns
    -------
    known : pandas.Series
        The series without the hidden readings (a copy; ``series`` is unchanged).
    truth : pandas.Series
        The hidden readings: the true values the methods must reconstruct.
    """
    inside = (series.index >= start) & (series.index < end)
    return series[~inside].copy(), series[inside].copy()


def interpolate_gap(known, query_times, method, context_h=CONTEXT_H,
                    max_gap_h=MAX_FILL_GAP_H):
    """Estimate CO₂ at ``query_times`` inside one gap by interpolation.

    Only known readings within ``context_h`` hours of the gap are used. The gap is the
    span between the last known reading before the first query time and the first
    known reading after the last query time.

    Parameters
    ----------
    known : pandas.Series
        Time-indexed known readings (the gap's readings removed).
    query_times : pandas.DatetimeIndex
        Times to estimate, all inside the same gap.
    method : str
        One of :data:`METHODS`.
    context_h : float
        Hours of known data used on each side of the gap.
    max_gap_h : float
        Longest gap that may be filled.

    Returns
    -------
    numpy.ndarray
        Estimated CO₂ (ppm) at each query time.

    Raises
    ------
    ValueError
        If ``method`` is unknown, if there is no known reading on one side of the gap
        (filling it would be extrapolation), or if the gap is longer than ``max_gap_h``.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    query_times = pd.DatetimeIndex(query_times)
    before = known[known.index < query_times.min()]
    after = known[known.index > query_times.max()]
    if before.empty or after.empty:
        raise ValueError("Gap has no known reading on one side: that would be "
                         "extrapolation, which is not supported.")
    missing_h = _missing_hours(before.index[-1], after.index[0])
    if missing_h > max_gap_h:
        raise ValueError(f"Gap with {missing_h:.1f} h of missing readings is longer than "
                         f"the {max_gap_h:.0f} h limit and is not filled.")

    context = pd.Timedelta(hours=context_h)
    local = known[(known.index >= before.index[-1] - context)
                  & (known.index <= after.index[0] + context)]
    origin = local.index[0]
    x, y = _hours(local.index, origin), local.to_numpy(dtype=float)
    xq = _hours(query_times, origin)

    if method == "linear":
        interpolant = make_interp_spline(x, y, k=1)
    elif method == "pchip":
        interpolant = PchipInterpolator(x, y)
    elif method == "akima":
        interpolant = Akima1DInterpolator(x, y)
    else:
        interpolant = CubicSpline(x, y)        # SciPy default "not-a-knot" end conditions
    return np.asarray(interpolant(xq), dtype=float)


def missing_slots(gap_start, gap_end, interval_min=SAMPLE_INTERVAL_MIN):
    """Times of the readings missing between two consecutive readings.

    Parameters
    ----------
    gap_start, gap_end : pandas.Timestamp
        Last reading before and first reading after the gap.
    interval_min : float
        Sampling interval in minutes.

    Returns
    -------
    pandas.DatetimeIndex
        ``gap_start + k × interval`` for every slot strictly before ``gap_end``, keeping
        at least half an interval clear of ``gap_end``.
    """
    step = pd.Timedelta(minutes=interval_min)
    n = int(np.rint((gap_end - gap_start) / step)) - 1
    return pd.DatetimeIndex([gap_start + k * step for k in range(1, max(n, 0) + 1)])


def fill_gaps_by_rule(series, gaps, choose, max_gap_h=MAX_FILL_GAP_H):
    """Fill real gaps, choosing the method for each gap from its length.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed readings with real gaps.
    gaps : pandas.DataFrame
        Gaps as returned by ``co2gaps.io.find_gaps`` (columns ``start`` and ``end``).
    choose : callable
        ``choose(missing_h) -> (name, estimator)``, where ``missing_h`` is the hours of
        missing readings and ``estimator(known, query_times)`` returns the estimates.
    max_gap_h : float
        Gaps with more missing hours than this are left empty (see
        :data:`MAX_FILL_GAP_H`).

    Returns
    -------
    filled : pandas.DataFrame
        Columns ``co2_ppm``, ``filled`` (True for reconstructed values) and ``method``
        (empty string for measured values), indexed by time and sorted.
    report : pandas.DataFrame
        ``gaps`` plus ``missing_h``, ``status`` (``"filled"`` or ``"too long"``),
        ``method`` and ``n_filled``.
    """
    new_times, new_values, new_methods = [], [], []
    missing, status, used, n_filled = [], [], [], []
    for gap in gaps.itertuples():
        missing_h = _missing_hours(gap.start, gap.end)
        missing.append(missing_h)
        if missing_h > max_gap_h:
            status.append("too long")
            used.append("")
            n_filled.append(0)
            continue
        slots = missing_slots(gap.start, gap.end)
        name, estimator = choose(missing_h)
        values = estimator(series, slots)
        new_times.extend(slots)
        new_values.extend(values)
        new_methods.extend([name] * len(slots))
        status.append("filled")
        used.append(name)
        n_filled.append(len(slots))

    original = pd.DataFrame({"co2_ppm": series.to_numpy(dtype=float), "filled": False,
                             "method": ""}, index=series.index)
    added = pd.DataFrame({"co2_ppm": new_values, "filled": True, "method": new_methods},
                         index=pd.DatetimeIndex(new_times, tz=series.index.tz))
    filled = pd.concat([original, added]).sort_index()
    report = gaps.copy()
    report["missing_h"] = missing
    report["status"] = status
    report["method"] = used
    report["n_filled"] = n_filled
    return filled, report


def fill_gaps(series, method, gaps, max_gap_h=MAX_FILL_GAP_H, context_h=CONTEXT_H):
    """Fill real gaps up to ``max_gap_h`` with one interpolation method.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed readings with real gaps.
    method : str
        One of :data:`METHODS`.
    gaps : pandas.DataFrame
        Gaps as returned by ``co2gaps.io.find_gaps``.
    max_gap_h, context_h : float
        See :data:`MAX_FILL_GAP_H` and :data:`CONTEXT_H`.

    Returns
    -------
    filled, report : pandas.DataFrame
        As for :func:`fill_gaps_by_rule`.
    """
    def estimator(known, query_times):
        return interpolate_gap(known, query_times, method, context_h, max_gap_h)
    return fill_gaps_by_rule(series, gaps, lambda missing_h: (method, estimator), max_gap_h)
