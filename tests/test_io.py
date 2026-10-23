"""Unit tests for co2gaps.io, using small hand-made inputs (no raw data needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from co2gaps import io

T0 = 1_717_200_000          # an arbitrary Unix time (1 June 2024)


def _json(readings, name="Carbon dioxide", unit="ppm"):
    """Build one ``jsondata`` string holding ``readings`` = [(time, value), ...]."""
    values = [{"time": t, "value": v} for t, v in readings]
    return json.dumps([{"values": values, "variable": {"name": name, "unit": unit}}])


def _raw():
    """A tiny raw table with a repeat upload, a bad JSON row and a temperature row."""
    return pd.DataFrame({
        "sensorid": ["111", "111", "111", "222", "111"],
        "jsondata": [
            _json([(T0, 500)]),
            _json([(T0, 500)]),                              # exact repeat upload
            "not json",                                      # malformed row
            _json([(T0, 450), (T0 + 600, 460)]),
            _json([(T0 + 600, 21.5)], name="Temperature", unit="°C"),
        ],
    })


def _series(values, step_s=600, start=T0):
    """Regularly sampled series starting at ``start`` with ``step_s`` spacing."""
    index = pd.to_datetime(start + step_s * np.arange(len(values)), unit="s", utc=True)
    return pd.Series(np.asarray(values, dtype=float), index=index, name="co2_ppm")


def test_load_raw_rejects_missing_columns(tmp_path):
    """A file without the jsondata column is refused with a clear error."""
    path = tmp_path / "bad.csv"
    pd.DataFrame({"sensorid": ["1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="jsondata"):
        io.load_raw(path)


def test_load_raw_keeps_sensor_id_as_text(tmp_path):
    """Long numeric sensor IDs are read as text, not rounded numbers."""
    path = tmp_path / "ok.csv"
    _raw().to_csv(path, index=False)
    assert io.load_raw(path)["sensorid"].iloc[0] == "111"


def test_parse_keeps_only_requested_variable():
    """By default only CO₂ readings are kept, with numeric values and local times."""
    long = io.parse_jsondata(_raw())
    assert set(long["variable"]) == {"Carbon dioxide"}
    assert len(long) == 4                      # 2 copies for sensor 111 + 2 for 222
    assert long["value"].dtype == float
    assert str(long["time"].dt.tz) == io.LOCAL_TIMEZONE


def test_parse_counts_bad_json():
    """Malformed JSON rows are skipped and counted."""
    assert io.parse_jsondata(_raw()).attrs["n_bad_json"] == 1


def test_parse_all_variables():
    """variables=None keeps every variable."""
    long = io.parse_jsondata(_raw(), variables=None)
    assert set(long["variable"]) == {"Carbon dioxide", "Temperature"}


def test_parse_missing_value_becomes_nan():
    """A reading without a 'value' field becomes NaN instead of crashing."""
    text = json.dumps([{"values": [{"time": T0}],
                        "variable": {"name": "Carbon dioxide", "unit": "ppm"}}])
    long = io.parse_jsondata(pd.DataFrame({"sensorid": ["1"], "jsondata": [text]}))
    assert np.isnan(long["value"].iloc[0])


def test_deduplicate_removes_exact_repeats():
    """Repeated uploads of the same reading are reduced to one row."""
    clean, report = io.deduplicate(io.parse_jsondata(_raw()))
    assert report == {"n_in": 4, "n_out": 3, "n_exact_duplicates": 1, "n_conflicts": 0}
    assert len(clean) == 3


def test_deduplicate_counts_conflicts():
    """Two copies with different values are a conflict: counted, first copy kept."""
    raw = pd.DataFrame({"sensorid": ["1", "1"],
                        "jsondata": [_json([(T0, 500)]), _json([(T0, 900)])]})
    clean, report = io.deduplicate(io.parse_jsondata(raw))
    assert report["n_conflicts"] == 1
    assert clean["value"].tolist() == [500.0]


def test_co2_series_is_sorted_and_named():
    """co2_series returns a sorted, uniquely indexed series called co2_ppm."""
    clean, _ = io.deduplicate(io.parse_jsondata(_raw()))
    series = io.co2_series(clean, "222")
    assert series.name == "co2_ppm"
    assert series.index.is_monotonic_increasing and series.index.is_unique
    assert series.tolist() == [450.0, 460.0]


def test_co2_series_unknown_sensor():
    """Asking for a sensor that has no CO₂ data raises KeyError."""
    clean, _ = io.deduplicate(io.parse_jsondata(_raw()))
    with pytest.raises(KeyError):
        io.co2_series(clean, "999")


def test_find_gaps_counts_missing_samples():
    """A 1-hour hole in 10-minute data is one gap with 5 missing samples."""
    series = _series(np.full(10, 500.0))
    series = series.drop(series.index[3:8])          # remove 5 samples
    gaps = io.find_gaps(series)
    assert len(gaps) == 1
    assert gaps["n_missing"].iloc[0] == 5
    assert gaps["duration_h"].iloc[0] == pytest.approx(1.0)


def test_find_gaps_ignores_timing_jitter():
    """Intervals of 599–601 s are normal jitter, not gaps."""
    index = pd.to_datetime(T0 + np.array([0, 601, 1200, 1799]), unit="s", utc=True)
    assert io.find_gaps(pd.Series([1.0, 2.0, 3.0, 4.0], index=index)).empty


def test_rate_of_change_of_linear_ramp():
    """np.gradient is exact for a straight line: a 60 ppm/h ramp gives 60 everywhere."""
    series = _series(400 + 10 * np.arange(6))         # +10 ppm every 10 min
    np.testing.assert_allclose(io.rate_of_change(series), 60.0)


def test_flag_readings():
    """Out-of-range values and very fast changes are flagged; nothing is removed."""
    flags = io.flag_readings(_series([500, 500, 300, 500, 500, 2500]))
    assert len(flags) == 6
    assert flags["out_of_range"].tolist() == [False, False, True, False, False, False]
    assert flags["fast_change"].iloc[-1]             # +2000 ppm in 10 min
    assert not flags["fast_change"].iloc[0]


def test_audit_series_summary():
    """The audit reports coverage, gaps and range for a series with one gap."""
    series = _series(np.linspace(450, 550, 10)).drop(_series(np.zeros(10)).index[4:6])
    summary = io.audit_series(series)
    assert summary["n_readings"] == 8
    assert summary["n_gaps"] == 1 and summary["n_missing"] == 2
    assert summary["coverage"] == pytest.approx(0.8)
    assert summary["min_ppm"] == 450 and summary["max_ppm"] == 550
    assert summary["n_out_of_range"] == 0
