"""Load, parse, de-duplicate and audit the raw environmental-sensor extract.

The raw file (``5EnvSensor_MayToDec2024_180kRows.csv``) has one row per sensor upload.
Each row's ``jsondata`` column holds a JSON list of variables (CO₂, temperature,
humidity, ...), each with one or more ``{"time": <unix seconds>, "value": <number>}``
readings. This module turns that into a clean, time-ordered CO₂ series per sensor and
reports the data-quality problems found along the way (Week 5.6 data audit).

Typical use::

    raw = load_raw("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
    long = parse_jsondata(raw)                      # one row per reading
    clean, report = deduplicate(long)               # drop repeated uploads
    co2 = co2_series(clean, "6012002000869")        # pandas Series, ppm
    audit_series(co2)                               # coverage, gaps, validity
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

LOCAL_TIMEZONE = "Australia/Melbourne"
"""Time zone of the building; all timestamps are converted to it."""

CO2_VALID_RANGE_PPM = (350.0, 5000.0)
"""Physically plausible indoor CO₂ range (ppm). Outdoor air is about 420 ppm, so
readings below 350 ppm indicate a sensor fault; 5000 ppm is the sensor's upper limit."""

NOMINAL_INTERVAL_S = 600
"""Nominal sampling interval of the environmental sensors (10 minutes)."""

MAX_RATE_PPM_PER_H = 1000.0
"""Rate-of-change threshold (ppm per hour) above which a reading is flagged for
inspection. It equals a change of about 170 ppm in one 10-minute step."""

REQUIRED_COLUMNS = ("sensorid", "jsondata")
"""Columns the raw CSV must contain."""

CO2_VARIABLE = "Carbon dioxide"
"""Name of the CO₂ variable inside ``jsondata``."""


def load_raw(path, nrows=None):
    """Read the raw sensor CSV.

    Parameters
    ----------
    path : str or path-like
        Location of the raw CSV file.
    nrows : int, optional
        Read only the first ``nrows`` rows (useful for quick checks).

    Returns
    -------
    pandas.DataFrame
        The raw table, with ``sensorid`` read as text so long IDs are not rounded.

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    raw = pd.read_csv(path, nrows=nrows, dtype={"sensorid": str})
    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"Raw file is missing required column(s): {missing}")
    return raw


def parse_jsondata(raw, variables=(CO2_VARIABLE,)):
    """Expand the ``jsondata`` column into one row per reading.

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of :func:`load_raw` (needs ``sensorid`` and ``jsondata``).
    variables : sequence of str or None
        Variable names to keep. ``None`` keeps every variable.

    Returns
    -------
    pandas.DataFrame
        Columns ``sensor`` (str), ``epoch_s`` (int, Unix seconds), ``time``
        (timezone-aware, local time), ``variable``, ``unit`` and ``value`` (float).
        Rows are sorted by sensor and time. ``df.attrs["n_bad_json"]`` holds the number
        of rows whose JSON could not be read; readings without a value become NaN.
    """
    keep = None if variables is None else set(variables)
    rows = []
    n_bad = 0
    for sensor, text in zip(raw["sensorid"].astype(str), raw["jsondata"]):
        try:
            entries = json.loads(text)
        except (TypeError, ValueError):
            n_bad += 1                       # empty or malformed JSON: skip the row
            continue
        for entry in entries:
            var = entry.get("variable", {})
            name = var.get("name")
            if keep is not None and name not in keep:
                continue
            for reading in entry.get("values", []):
                rows.append((sensor, reading.get("time"), name, var.get("unit"),
                             reading.get("value", np.nan)))

    long = pd.DataFrame(rows, columns=["sensor", "epoch_s", "variable", "unit", "value"])
    long = long.dropna(subset=["epoch_s"])
    long["epoch_s"] = long["epoch_s"].astype("int64")
    long["value"] = pd.to_numeric(long["value"], errors="coerce").astype(float)
    long["time"] = (pd.to_datetime(long["epoch_s"], unit="s", utc=True)
                    .dt.tz_convert(LOCAL_TIMEZONE))
    long = long[["sensor", "epoch_s", "time", "variable", "unit", "value"]]
    long = long.sort_values(["sensor", "epoch_s"], kind="stable").reset_index(drop=True)
    long.attrs["n_bad_json"] = n_bad
    return long


def deduplicate(long):
    """Remove repeated readings of the same sensor, variable and timestamp.

    The same reading is often uploaded several times. Exact repeats are dropped. If two
    copies disagree on the value (a *conflict*), the first copy is kept and the conflict
    is counted so it can be reported.

    Parameters
    ----------
    long : pandas.DataFrame
        Output of :func:`parse_jsondata`.

    Returns
    -------
    clean : pandas.DataFrame
        One row per (sensor, variable, epoch_s).
    report : dict
        ``n_in``, ``n_out``, ``n_exact_duplicates`` and ``n_conflicts``.
    """
    key = ["sensor", "variable", "epoch_s"]
    n_in = len(long)
    exact = long.duplicated(key + ["value"])
    n_values_per_key = long.groupby(key)["value"].nunique(dropna=False)
    clean = long.drop_duplicates(key, keep="first").reset_index(drop=True)
    report = {
        "n_in": n_in,
        "n_out": len(clean),
        "n_exact_duplicates": int(exact.sum()),
        "n_conflicts": int((n_values_per_key > 1).sum()),
    }
    return clean, report


def co2_series(clean, sensor):
    """Return one sensor's CO₂ readings as a time-indexed series.

    Parameters
    ----------
    clean : pandas.DataFrame
        De-duplicated readings from :func:`deduplicate`.
    sensor : str
        Sensor ID, e.g. ``"6012002000869"``.

    Returns
    -------
    pandas.Series
        Name ``co2_ppm``, float values in ppm, indexed by local time, sorted and with
        unique timestamps.

    Raises
    ------
    KeyError
        If the sensor has no CO₂ readings.
    """
    rows = clean[(clean["sensor"] == str(sensor)) & (clean["variable"] == CO2_VARIABLE)]
    if rows.empty:
        raise KeyError(f"No CO2 readings for sensor {sensor}")
    series = rows.set_index("time")["value"].astype(float).sort_index()
    series = series[~series.index.duplicated(keep="first")]
    series.name = "co2_ppm"
    return series


def _elapsed_seconds(index):
    """Seconds since the first timestamp (works for any datetime resolution)."""
    return ((index - index[0]) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def find_gaps(series, nominal_s=NOMINAL_INTERVAL_S):
    """List the gaps (missing samples) in a regularly sampled series.

    An interval counts as a gap when it is longer than 1.5 × the nominal interval, so
    normal timing jitter (599–601 s) is ignored.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed readings (e.g. from :func:`co2_series`).
    nominal_s : float
        Nominal sampling interval in seconds.

    Returns
    -------
    pandas.DataFrame
        One row per gap: ``start`` (last reading before), ``end`` (first reading
        after), ``duration_h`` and ``n_missing`` (estimated missing samples).
    """
    times = series.index
    dt_s = np.diff(_elapsed_seconds(times))          # interval lengths in seconds
    is_gap = dt_s > 1.5 * nominal_s
    starts, ends, durations = times[:-1][is_gap], times[1:][is_gap], dt_s[is_gap]
    return pd.DataFrame({
        "start": starts,
        "end": ends,
        "duration_h": durations / 3600.0,
        "n_missing": np.rint(durations / nominal_s).astype(int) - 1,
    })


def rate_of_change(series):
    """Rate of change of the series in units per hour (Week 7.1 finite differences).

    Uses ``numpy.gradient`` with the actual timestamps, which gives second-order central
    differences inside the series and one-sided differences at the ends.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed readings with at least two values.

    Returns
    -------
    numpy.ndarray
        dC/dt at each reading (e.g. ppm per hour).
    """
    hours = _elapsed_seconds(series.index) / 3600.0  # seconds -> hours
    return np.gradient(series.to_numpy(dtype=float), hours)


def flag_readings(series, valid_range=CO2_VALID_RANGE_PPM,
                  max_rate=MAX_RATE_PPM_PER_H):
    """Flag readings that are outside the valid range or change implausibly fast.

    Flags are for inspection and reporting; nothing is deleted.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed CO₂ readings in ppm.
    valid_range : tuple of float
        ``(low, high)`` plausible range in ppm.
    max_rate : float
        Largest plausible ``|dC/dt|`` in ppm per hour.

    Returns
    -------
    pandas.DataFrame
        Columns ``co2_ppm``, ``rate_ppm_per_h``, ``out_of_range`` and ``fast_change``.
    """
    low, high = valid_range
    values = series.to_numpy(dtype=float)
    rate = rate_of_change(series) if len(series) > 1 else np.zeros(len(series))
    return pd.DataFrame({
        "co2_ppm": values,
        "rate_ppm_per_h": rate,
        "out_of_range": (values < low) | (values > high),
        "fast_change": np.abs(rate) > max_rate,
    }, index=series.index)


def audit_series(series, nominal_s=NOMINAL_INTERVAL_S):
    """Summarise coverage, gaps and validity of one sensor's series.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed CO₂ readings in ppm.
    nominal_s : float
        Nominal sampling interval in seconds.

    Returns
    -------
    dict
        ``n_readings``, ``start``, ``end``, ``median_interval_s``, ``coverage``
        (readings received / readings expected), ``n_gaps``, ``n_missing``,
        ``longest_gap_h``, ``min_ppm``, ``max_ppm``, ``n_out_of_range`` and
        ``n_fast_change``.
    """
    gaps = find_gaps(series, nominal_s)
    flags = flag_readings(series)
    span_s = (series.index[-1] - series.index[0]).total_seconds()
    expected = span_s / nominal_s + 1
    return {
        "n_readings": len(series),
        "start": series.index[0],
        "end": series.index[-1],
        "median_interval_s": float(np.median(np.diff(_elapsed_seconds(series.index))))
        if len(series) > 1 else np.nan,
        "coverage": len(series) / expected,
        "n_gaps": len(gaps),
        "n_missing": int(gaps["n_missing"].sum()),
        "longest_gap_h": float(gaps["duration_h"].max()) if len(gaps) else 0.0,
        "min_ppm": float(series.min()),
        "max_ppm": float(series.max()),
        "n_out_of_range": int(flags["out_of_range"].sum()),
        "n_fast_change": int(flags["fast_change"].sum()),
    }
