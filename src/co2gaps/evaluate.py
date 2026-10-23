"""Error metrics and the masked-gap comparison of reconstruction methods (Week 5.5).

All errors are in ppm and are computed only on readings that were really recorded
inside a masked window; slots the sensor itself skipped have no true value and are not
scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from co2gaps import gaps


def rmse(estimate, truth):
    """Root-mean-square error, sqrt(mean((estimate − truth)²)), in the data's units."""
    diff = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(estimate, truth):
    """Mean absolute error, mean(|estimate − truth|), in the data's units."""
    diff = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.mean(np.abs(diff)))


def overshoot(estimate, truth):
    """How far the estimate goes outside the range of the true values (≥ 0).

    Returns ``max(max(estimate) − max(truth), min(truth) − min(estimate), 0)``. A value
    of 0 means the estimate stays within the true minimum and maximum of the gap; large
    values reveal the spurious swings that cubic splines can produce (Week 3.4).
    """
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(max(estimate.max() - truth.max(), truth.min() - estimate.min(), 0.0))


def compare_methods(series, gap_lengths_h, n_gaps, seed=0, methods=gaps.METHODS,
                    estimators=None):
    """Hide windows of known data, reconstruct them with every method, and score them.

    Parameters
    ----------
    series : pandas.Series
        Nearly complete, time-indexed readings (e.g. the development period).
    gap_lengths_h : sequence of float
        Artificial gap lengths to test, in hours.
    n_gaps : int
        Number of random windows per gap length.
    seed : int
        Base random seed; gap length ``i`` uses ``seed + i`` so results are repeatable.
    methods : sequence of str
        Interpolation methods from :data:`co2gaps.gaps.METHODS`.
    estimators : dict, optional
        Extra methods as ``{name: function(known, query_times) -> array}``, e.g. a
        regression baseline.

    Returns
    -------
    pandas.DataFrame
        One row per (gap length, window, method) with columns ``gap_h``, ``window``,
        ``start``, ``method``, ``rmse``, ``mae``, ``overshoot`` and ``n_scored``.
    """
    estimators = dict(estimators or {})
    rows = []
    for i, gap_h in enumerate(gap_lengths_h):
        windows = gaps.make_masked_gaps(series, gap_h, n_gaps, seed=seed + i)
        for w, (start, end) in enumerate(windows.itertuples(index=False)):
            known, truth = gaps.hide(series, start, end)
            if truth.empty:
                continue
            # The 24 h fill limit applies to real gaps; here the gap length is set by
            # the experiment, so no limit is imposed (max_gap_h=inf).
            candidates = {m: (lambda k, q, m=m: gaps.interpolate_gap(k, q, m,
                                                                   max_gap_h=np.inf))
                          for m in methods}
            candidates.update(estimators)
            for name, estimate_fn in candidates.items():
                estimate = estimate_fn(known, truth.index)
                rows.append({
                    "gap_h": gap_h, "window": w, "start": start, "method": name,
                    "rmse": rmse(estimate, truth), "mae": mae(estimate, truth),
                    "overshoot": overshoot(estimate, truth), "n_scored": len(truth),
                })
    return pd.DataFrame(rows)


def summarise(results, statistic="median"):
    """Table of one metric statistic per method (rows) and gap length (columns).

    Parameters
    ----------
    results : pandas.DataFrame
        Output of :func:`compare_methods`.
    statistic : str
        Any pandas aggregation name, e.g. ``"median"``, ``"mean"`` or ``"max"``.

    Returns
    -------
    dict of pandas.DataFrame
        ``{"rmse": table, "mae": table, "overshoot": table}``.
    """
    return {metric: results.pivot_table(index="method", columns="gap_h", values=metric,
                                        aggfunc=statistic)
            for metric in ("rmse", "mae", "overshoot")}
