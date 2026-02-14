import math

import pandas as pd

from sreejita.core.cleaner import clean_dataframe


REQUIRED = {"value", "confidence", "signal_strength", "data_coverage"}


def test_non_dataframe_input_is_guarded_and_returns_safe_summary():
    """Before fix: None input failed on df.copy()."""
    out = clean_dataframe(None)

    assert out["df"].empty
    assert out["summary"]["rows_original"] == 0
    assert set(out["summary"]["kpi"]["rows_original"]) == REQUIRED
    assert out["summary"]["kpi"]["rows_original"]["data_coverage"] == 0.0


def test_outlier_calc_handles_inf_and_constant_series_safely():
    """Before fix: inf values could produce unsafe stats and undefined denominator behavior."""
    df = pd.DataFrame(
        {
            "a": [5, 5, 5, 5],
            "b": [1.0, 2.0, math.inf, -math.inf],
        }
    )
    out = clean_dataframe(df)
    summary = out["summary"]

    assert summary["outlier_counts_by_column"]["a"] == 0
    assert summary["outlier_counts_by_column"]["b"] == 0

    kpi_a = summary["kpi"]["outlier_counts_by_column"]["a"]
    kpi_b = summary["kpi"]["outlier_counts_by_column"]["b"]
    assert set(kpi_a) == REQUIRED
    assert set(kpi_b) == REQUIRED
    assert kpi_a["signal_strength"] == 0.0  # constant series guard
    assert 0.0 <= kpi_b["data_coverage"] <= 1.0


def test_kpi_payloads_exist_for_core_summary_metrics_and_null_ratio():
    """Before fix: summary metrics were raw scalars/dicts without required KPI fields."""
    df = pd.DataFrame(
        {
            "x": [1, 1, None],
            "y": [" a ", "", "b"],
        }
    )
    out = clean_dataframe(df)
    kpi = out["summary"]["kpi"]

    for key in ("rows_original", "rows_after_cleaning", "columns", "duplicate_rows_removed"):
        assert set(kpi[key]) == REQUIRED

    for col, payload in kpi["null_ratio_by_column"].items():
        assert col in {"x", "y"}
        assert set(payload) == REQUIRED
        assert 0.0 <= payload["value"] <= 1.0
