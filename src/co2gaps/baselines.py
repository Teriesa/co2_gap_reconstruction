"""Regression baselines and the typical-day anomaly method (Week 5).

Interpolation (``co2gaps.gaps``) only uses the readings at the edges of a gap. The
methods here instead use what is *usual* at that time of day:

* ``mean``: the training-period mean for every missing value. This is
  ``sklearn.dummy.DummyRegressor(strategy="mean")``, the simplest possible baseline.
* ``typical_day``: the training-period median CO₂ for the same hour of day and the same
  day type (weekday or weekend). A regression on time-of-day features that ignores the
  readings around the gap.
* ``typical_day_anomaly``: the typical-day value plus a linearly interpolated *anomaly*
  (measured CO₂ minus the typical-day value). It follows the usual daily shape inside the
  gap while still matching the readings at both edges.

Every baseline is fitted on training data only and returned as an estimator function
``f(known, query_times) -> numpy.ndarray`` that ``co2gaps.evaluate.compare_methods`` accepts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from co2gaps import gaps


def _day_type(index):
    """1 for Saturday/Sunday, 0 for Monday–Friday."""
    return (index.dayofweek >= 5).astype(int)


class TypicalDay:
    """Median CO₂ by hour of day and day type, learned from a training series.

    Parameters
    ----------
    statistic : str
        ``"median"`` (default, robust to occasional spikes) or ``"mean"``.

    Attributes
    ----------
    table_ : pandas.Series
        The fitted value (ppm) for each (hour, day type) pair, after :meth:`fit`.
    fallback_ : float
        Overall training value used for any (hour, day type) pair never seen in training.
    """

    def __init__(self, statistic="median"):
        self.statistic = statistic

    def fit(self, series):
        """Learn the typical value for each hour and day type.

        Parameters
        ----------
        series : pandas.Series
            Time-indexed training readings (ppm).

        Returns
        -------
        TypicalDay
            ``self``, so calls can be chained.
        """
        keys = [series.index.hour, _day_type(series.index)]
        self.table_ = series.groupby(keys).agg(self.statistic)
        self.table_.index.names = ["hour", "weekend"]
        self.fallback_ = float(series.agg(self.statistic))
        return self

    def predict(self, times):
        """Typical CO₂ (ppm) at each time in ``times``.

        Parameters
        ----------
        times : pandas.DatetimeIndex
            Times to predict, in the same time zone as the training data.

        Returns
        -------
        numpy.ndarray
            One value per time.
        """
        times = pd.DatetimeIndex(times)
        keys = pd.MultiIndex.from_arrays([times.hour, _day_type(times)])
        return self.table_.reindex(keys).fillna(self.fallback_).to_numpy(dtype=float)


def mean_estimator(train):
    """Estimator that predicts the training mean everywhere (``DummyRegressor``).

    Parameters
    ----------
    train : pandas.Series
        Training readings (ppm).

    Returns
    -------
    callable
        ``f(known, query_times) -> numpy.ndarray``.
    """
    model = DummyRegressor(strategy="mean").fit(np.zeros((len(train), 1)), train.to_numpy())

    def estimate(known, query_times):
        return model.predict(np.zeros((len(query_times), 1)))
    return estimate


def typical_day_estimator(profile):
    """Estimator that predicts the typical-day value and ignores the gap's edges.

    Parameters
    ----------
    profile : TypicalDay
        A fitted profile.

    Returns
    -------
    callable
        ``f(known, query_times) -> numpy.ndarray``.
    """
    def estimate(known, query_times):
        return profile.predict(query_times)
    return estimate


def anomaly_estimator(profile, method="linear", max_gap_h=np.inf):
    """Estimator: typical-day value plus an interpolated anomaly.

    The anomaly is ``known − profile(known times)``. It is interpolated across the gap
    with ``co2gaps.gaps.interpolate_gap`` and added back to the profile, so the estimate
    equals the measured values at the gap's edges and follows the usual daily shape in
    between.

    Parameters
    ----------
    profile : TypicalDay
        A fitted profile.
    method : str
        Interpolation method for the anomaly, from ``co2gaps.gaps.METHODS``.
    max_gap_h : float
        Passed to ``interpolate_gap``. Use ``numpy.inf`` in masked-gap experiments,
        where the gap length is set by the experiment.

    Returns
    -------
    callable
        ``f(known, query_times) -> numpy.ndarray``.
    """
    def estimate(known, query_times):
        query_times = pd.DatetimeIndex(query_times)
        # Only readings near the gap are needed (the interpolant uses CONTEXT_H each side,
        # and a gap may be up to a day long), so restrict the work to them.
        margin = pd.Timedelta(hours=gaps.CONTEXT_H + 48)
        known = known[(known.index >= query_times.min() - margin)
                      & (known.index <= query_times.max() + margin)]
        anomaly = known - profile.predict(known.index)
        filled = gaps.interpolate_gap(anomaly, query_times, method, max_gap_h=max_gap_h)
        return profile.predict(query_times) + filled
    return estimate


def make_estimators(train):
    """All three baselines, fitted on ``train`` only.

    Parameters
    ----------
    train : pandas.Series
        Training readings (ppm). Must not overlap the data the estimators are tested on.

    Returns
    -------
    dict
        ``{"mean": f, "typical_day": f, "typical_day_anomaly": f}``.
    """
    profile = TypicalDay().fit(train)
    return {
        "mean": mean_estimator(train),
        "typical_day": typical_day_estimator(profile),
        "typical_day_anomaly": anomaly_estimator(profile),
    }
