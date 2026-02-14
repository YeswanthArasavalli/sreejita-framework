import math

import pandas as pd

from sreejita.core.kpis import compute_kpis


REQUIRED_FIELDS = {"value", "confidence", "signal_strength", "data_coverage"}


def test_empty_dataframe_has_safe_structural_kpis():
    """Regression: previously only raw ints were returned, with no safety metadata."""
    kpis = compute_kpis(pd.DataFrame())

    assert set(kpis["row_count"]) == REQUIRED_FIELDS
    assert set(kpis["column_count"]) == REQUIRED_FIELDS
    assert kpis["row_count"]["value"] == 0.0
    assert kpis["row_count"]["data_coverage"] == 0.0
    assert kpis["column_count"]["value"] == 0.0
    assert kpis["column_count"]["data_coverage"] == 0.0


def test_scalar_kpis_are_normalized_with_required_fields():
    """Regression: scalar KPIs previously lacked confidence/signal/coverage fields."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    kpis = compute_kpis(df, base_kpis={"quality_score": 0.82}, domain_kpis={"roas": 2.7})

    for key in ("quality_score", "roas"):
        assert set(kpis[key]) == REQUIRED_FIELDS
        assert kpis[key]["data_coverage"] == 1.0
        assert kpis[key]["confidence"] == 1.0
        assert kpis[key]["signal_strength"] == 1.0


def test_nan_inf_and_out_of_range_scores_are_sanitized():
    """Regression: NaN/Inf scores and values could leak into KPI output."""
    df = pd.DataFrame({"x": [1]})
    kpis = compute_kpis(
        df,
        base_kpis={
            "unstable": {
                "value": math.inf,
                "confidence": math.nan,
                "signal_strength": 1.4,
                "data_coverage": -0.5,
            }
        },
    )

    unstable = kpis["unstable"]
    assert unstable["value"] == 0.0
    assert unstable["confidence"] == 0.0
    assert unstable["signal_strength"] == 1.0
    assert unstable["data_coverage"] == 0.0


def test_non_dataframe_input_is_guarded_and_metadata_preserved():
    """Regression: non-DataFrame input could crash on shape access."""
    kpis = compute_kpis(None, base_kpis={"_confidence": {"a": 0.9}})

    assert kpis["row_count"]["value"] == 0.0
    assert kpis["_confidence"] == {"a": 0.9}
