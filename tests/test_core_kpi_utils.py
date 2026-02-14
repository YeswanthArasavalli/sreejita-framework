import math

import pandas as pd

from sreejita.core.kpi_utils import has_columns, safe_mean, safe_sum


REQUIRED = {"value", "confidence", "signal_strength", "data_coverage"}


def test_has_columns_handles_non_dataframe_input():
    """Before fix: non-DataFrame input raised when accessing .columns."""
    assert has_columns(None, "a") is False


def test_safe_sum_empty_and_missing_column_are_safe_payloads():
    """Before fix: functions returned None without required KPI fields."""
    out_missing = safe_sum(pd.DataFrame({"a": [1, 2]}), "b")
    out_empty = safe_sum(pd.DataFrame({"a": []}), "a")

    assert set(out_missing) == REQUIRED
    assert out_missing == {
        "value": 0.0,
        "confidence": 0.0,
        "signal_strength": 0.0,
        "data_coverage": 0.0,
    }
    assert set(out_empty) == REQUIRED
    assert out_empty["data_coverage"] == 0.0


def test_safe_mean_sanitizes_nan_inf_and_handles_zero_variance():
    """Before fix: NaN/Inf could leak and constant series had no explicit signal policy."""
    df = pd.DataFrame({"x": [5, 5, 5, math.inf, -math.inf, None]})
    out = safe_mean(df, "x")

    assert set(out) == REQUIRED
    assert out["value"] == 5.0
    assert out["data_coverage"] == 0.5
    assert out["confidence"] == 0.5
    assert out["signal_strength"] == 0.0  # zero variance on valid values


def test_safe_sum_coverage_uses_zero_division_guard():
    """Before fix: coverage metric did not exist and could imply unsafe division assumptions."""
    out = safe_sum(pd.DataFrame(), "x")
    assert set(out) == REQUIRED
    assert out["data_coverage"] == 0.0
