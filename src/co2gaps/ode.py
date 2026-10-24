"""Physics-based alternative: a single-zone CO₂ mass-balance ODE (Week 7).

The room is treated as one well-mixed volume:

    dC/dt = λ (C_out − C) + S(t)

* ``C`` is indoor CO₂ (ppm) and ``t`` is time in hours.
* ``λ`` (1/h) is the air-change rate: ventilation pulls ``C`` towards the outdoor level.
* ``C_out`` (ppm) is the background level that ventilation brings the room back to.
* ``S(t)`` (ppm/h) is the CO₂ added by people, divided by the room volume. It is not a
  headcount, because the room volume and per-person emission rate are unknown.

Calibration and solution:

1. ``λ`` and ``C_out`` are fitted with ``scipy.optimize.least_squares`` (Week 7.2) to
   overnight decays, when the room is assumed empty (``S = 0``) and the exact solution is
   ``C(t) = C_out + (C_0 − C_out) e^(−λt)``.
2. ``S(t)`` is recovered from the data by inverting the ODE,
   ``S = dC/dt + λ (C − C_out)``, with ``np.gradient`` for dC/dt (Week 7.1), and summarised
   as a median for each hour of the day and day type.
3. A gap is filled by integrating the ODE forward from the last reading before the gap
   with Euler's method (Week 7.3) or classical RK4 (Week 7.4). Optionally, the mismatch at
   the first reading after the gap is removed with a linear correction, so the estimate
   meets the measured values at both edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from co2gaps import baselines, gaps, io


def overnight_segments(series, start_hour=20, end_hour=6, weekdays_only=True,
                       min_readings=30):
    """Cut the series into overnight stretches, assumed to have no occupants.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed CO₂ readings (ppm).
    start_hour, end_hour : int
        Each segment runs from ``start_hour`` on one day to ``end_hour`` the next day.
    weekdays_only : bool
        Keep only nights after Monday–Friday, when CO₂ is decaying after the working day.
    min_readings : int
        Shorter segments (e.g. broken by gaps) are dropped.

    Returns
    -------
    list of pandas.Series
    """
    segments = []
    days = pd.date_range(series.index[0].normalize(), series.index[-1].normalize(), freq="D")
    for day in days:
        if weekdays_only and day.dayofweek >= 5:
            continue
        start = day + pd.Timedelta(hours=start_hour)
        end = day + pd.Timedelta(days=1, hours=end_hour)
        segment = series[(series.index >= start) & (series.index < end)]
        if len(segment) >= min_readings:
            segments.append(segment)
    return segments


def calibrate_decay(segments, initial=(0.2, 430.0)):
    """Fit λ and C_out to overnight decays by least squares (Week 7.2).

    Each segment is modelled by the exact solution with ``S = 0`` and its own measured
    starting value: ``C(t) = C_out + (C_0 − C_out) e^(−λ t)``.

    Parameters
    ----------
    segments : list of pandas.Series
        Output of :func:`overnight_segments`.
    initial : tuple of float
        Starting guess ``(λ, C_out)``.

    Returns
    -------
    dict
        ``lam`` (1/h), ``c_out`` (ppm), ``rmse`` (ppm, fit residual) and ``n_segments``.
    """
    hours = [((s.index - s.index[0]) / pd.Timedelta(hours=1)).to_numpy() for s in segments]
    values = [s.to_numpy(dtype=float) for s in segments]

    def residuals(params):
        lam, c_out = params
        return np.concatenate([c_out + (y[0] - c_out) * np.exp(-lam * t) - y
                               for t, y in zip(hours, values)])

    fit = least_squares(residuals, initial, bounds=([1e-4, 300.0], [5.0, 600.0]))
    return {"lam": float(fit.x[0]), "c_out": float(fit.x[1]),
            "rmse": float(np.sqrt(np.mean(fit.fun ** 2))), "n_segments": len(segments)}


def source_profile(series, lam, c_out):
    """Typical occupancy source S (ppm/h) by hour and day type, from the inverted ODE.

    ``S = dC/dt + λ (C − C_out)``, with dC/dt from ``np.gradient`` (central differences,
    Week 7.1). The median by hour and day type limits the noise that differentiation
    amplifies.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed training readings (ppm).
    lam, c_out : float
        Calibrated parameters from :func:`calibrate_decay`.

    Returns
    -------
    co2gaps.baselines.TypicalDay
        Fitted to the series of S values; ``predict(times)`` gives S in ppm/h.
    """
    rate = io.rate_of_change(series)
    source = pd.Series(rate + lam * (series.to_numpy(dtype=float) - c_out), index=series.index)
    return baselines.TypicalDay().fit(source)


def simulate(c0, hours, lam, c_out, source, method="rk4"):
    """Integrate dC/dt = λ(C_out − C) + S from ``C(hours[0]) = c0``.

    ``S`` is held constant within each step at its value at the start of the step.

    Parameters
    ----------
    c0 : float
        Starting CO₂ (ppm).
    hours : numpy.ndarray
        Increasing time points (h); steps may be uneven.
    lam, c_out : float
        Air-change rate (1/h) and background CO₂ (ppm).
    source : numpy.ndarray
        S (ppm/h) at each time point (the last value is not used).
    method : str
        ``"euler"`` (first order, Week 7.3) or ``"rk4"`` (fourth order, Week 7.4).

    Returns
    -------
    numpy.ndarray
        C (ppm) at every time point.
    """
    if method not in ("euler", "rk4"):
        raise ValueError(f"Unknown method {method!r}; choose 'euler' or 'rk4'")
    c = np.empty(len(hours))
    c[0] = c0
    for n, h in enumerate(np.diff(hours)):
        s = source[n]

        def f(value):
            return lam * (c_out - value) + s

        if method == "euler":
            c[n + 1] = c[n] + h * f(c[n])
        else:
            k1 = f(c[n])
            k2 = f(c[n] + 0.5 * h * k1)
            k3 = f(c[n] + 0.5 * h * k2)
            k4 = f(c[n] + h * k3)
            c[n + 1] = c[n] + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return c


def exact_constant_source(c0, hours, lam, c_out, s):
    """Exact solution for constant S: C(t) = C_eq + (C_0 − C_eq) e^(−λt), C_eq = C_out + S/λ."""
    c_eq = c_out + s / lam
    return c_eq + (c0 - c_eq) * np.exp(-lam * (hours - hours[0]))


def ode_estimator(lam, c_out, profile, method="rk4", end_correction=True,
                  step_min=gaps.SAMPLE_INTERVAL_MIN):
    """Estimator that fills a gap by integrating the ODE across it.

    Parameters
    ----------
    lam, c_out : float
        Calibrated parameters.
    profile : co2gaps.baselines.TypicalDay
        Fitted source profile from :func:`source_profile`.
    method : str
        ``"euler"`` or ``"rk4"``.
    end_correction : bool
        If True, the difference between the simulated and the measured value at the first
        reading after the gap is spread linearly across the gap, so the estimate meets the
        measured values at both edges. If False, only the reading before the gap is used
        (a forward forecast).
    step_min : float
        Solver step in minutes.

    Returns
    -------
    callable
        ``f(known, query_times) -> numpy.ndarray``.
    """
    def estimate(known, query_times):
        query_times = pd.DatetimeIndex(query_times)
        before = known[known.index < query_times.min()]
        after = known[known.index > query_times.max()]
        if before.empty or after.empty:
            raise ValueError("Gap has no known reading on one side: that would be "
                             "extrapolation, which is not supported.")
        t0, c0 = before.index[-1], float(before.iloc[-1])
        t1, c1 = after.index[0], float(after.iloc[0])
        grid = pd.date_range(t0, t1, freq=f"{step_min}min")
        if grid[-1] < t1:
            grid = grid.append(pd.DatetimeIndex([t1]))
        hours = ((grid - t0) / pd.Timedelta(hours=1)).to_numpy()
        c = simulate(c0, hours, lam, c_out, profile.predict(grid), method)
        if end_correction:
            c = c + (c1 - c[-1]) * hours / hours[-1]
        query_hours = ((query_times - t0) / pd.Timedelta(hours=1)).to_numpy()
        return np.interp(query_hours, hours, c)
    return estimate


def make_ode_estimators(train, **segment_options):
    """Calibrate on ``train`` and return the ODE estimators and the calibration.

    Parameters
    ----------
    train : pandas.Series
        Training readings (ppm).
    **segment_options
        Passed to :func:`overnight_segments`.

    Returns
    -------
    estimators : dict
        ``{"ode": RK4 with end correction, "ode_forward": RK4 without it}``.
    calibration : dict
        Output of :func:`calibrate_decay`.
    """
    calibration = calibrate_decay(overnight_segments(train, **segment_options))
    profile = source_profile(train, calibration["lam"], calibration["c_out"])
    lam, c_out = calibration["lam"], calibration["c_out"]
    return {
        "ode": ode_estimator(lam, c_out, profile, end_correction=True),
        "ode_forward": ode_estimator(lam, c_out, profile, end_correction=False),
    }, calibration
