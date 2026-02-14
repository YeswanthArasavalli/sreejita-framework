import pandas as pd

from sreejita.core.validator import DataQualityValidator


REQUIRED = {"value", "confidence", "signal_strength", "data_coverage"}


def test_validate_guards_non_dataframe_input():
    """Before fix: non-DataFrame input raised on len(df.columns)."""
    passed, results = DataQualityValidator(strict=True).validate(None)

    assert passed is True
    assert results["rows"] == 0
    assert results["columns"] == 0
    assert results["duplicate_rows"] == 0
    assert set(results["kpi"]["rows"]) == REQUIRED
    assert results["kpi"]["rows"]["data_coverage"] == 0.0


def test_missing_ratio_and_duplicate_ratio_are_divide_by_zero_safe():
    """Before fix: ratio-style internals were absent and could be undefined for empty input."""
    passed, results = DataQualityValidator(strict=False).validate(pd.DataFrame(columns=["a", "b"]))

    assert passed is True
    # KPI payloads still exist and are finite.
    assert set(results["kpi"]["duplicate_rows"]) == REQUIRED
    assert results["kpi"]["duplicate_rows"]["data_coverage"] == 0.0


def test_kpi_payloads_exist_for_internal_and_output_metrics():
    """Before fix: results had raw values only, without required KPI fields."""
    df = pd.DataFrame({"a": [1, None, 1], "b": ["x", "x", "x"]})
    passed, results = DataQualityValidator(strict=True).validate(df)

    assert passed is False
    for key in ("rows", "columns", "duplicate_rows", "validation_passed"):
        assert set(results["kpi"][key]) == REQUIRED

    for payload in results["kpi"]["missing_values"].values():
        assert set(payload) == REQUIRED
        assert 0.0 <= payload["confidence"] <= 1.0

    for payload in results["kpi"]["missing_ratio"].values():
        assert set(payload) == REQUIRED
        assert 0.0 <= payload["value"] <= 1.0
