import pandas as pd

from sreejita.core.dataset_shape import DatasetShape, detect_dataset_shape


REQUIRED = {"value", "confidence", "signal_strength", "data_coverage"}


def test_invalid_input_returns_safe_kpi_payload():
    """Before fix: output had no standardized KPI payload fields."""
    out = detect_dataset_shape(None)

    assert out["shape"] == DatasetShape.UNKNOWN
    assert set(out["kpi"]) == REQUIRED
    assert out["kpi"] == {
        "value": 0.0,
        "confidence": 0.0,
        "signal_strength": 0.0,
        "data_coverage": 0.0,
    }


def test_empty_dataframe_has_zero_coverage_and_defined_metrics():
    """Before fix: no explicit KPI/metric payload for empty dataframe case."""
    out = detect_dataset_shape(pd.DataFrame())

    assert out["shape"] == DatasetShape.UNKNOWN
    assert out["kpi"]["data_coverage"] == 0.0
    assert out["signal_metrics"] == {}


def test_valid_dataframe_exposes_safe_signal_metrics_for_all_shapes():
    """Before fix: per-shape internals were raw floats, no required fields."""
    df = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "admission_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "visit_count": [2, 3, 1],
        }
    )
    out = detect_dataset_shape(df)

    assert out["shape"] == DatasetShape.ROW_LEVEL_CLINICAL
    assert out["kpi"]["value"] >= 0.6
    assert out["kpi"]["data_coverage"] == 1.0

    for shape_key in (
        "row_level_clinical",
        "aggregated_operational",
        "financial_summary",
        "quality_metrics",
    ):
        assert set(out["signal_metrics"][shape_key]) == REQUIRED
        assert 0.0 <= out["signal_metrics"][shape_key]["value"] <= 1.0
        assert 0.0 <= out["signal_metrics"][shape_key]["confidence"] <= 1.0
        assert 0.0 <= out["signal_metrics"][shape_key]["signal_strength"] <= 1.0
        assert out["signal_metrics"][shape_key]["data_coverage"] == 1.0
