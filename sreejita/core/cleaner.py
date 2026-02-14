import math
from typing import Any, Dict

import pandas as pd
import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Return finite float; fallback for NaN/Inf/non-numeric values."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clamp01(value: Any) -> float:
    """Clamp score-like values to [0, 1]."""
    return max(0.0, min(_safe_float(value, default=0.0), 1.0))


def _kpi_payload(
    value: Any,
    confidence: Any,
    signal_strength: Any,
    data_coverage: Any,
) -> Dict[str, float]:
    """Standardized KPI payload with guaranteed required fields."""
    return {
        "value": _safe_float(value, default=0.0),
        "confidence": _clamp01(confidence),
        "signal_strength": _clamp01(signal_strength),
        "data_coverage": _clamp01(data_coverage),
    }


def clean_dataframe(df: pd.DataFrame, preserve_date_cols: list = None):
    """
    Clean a dataframe and produce a data integrity summary.

    This function performs light, deterministic cleaning and reports
    data quality metrics required for audit and review readiness.

    Args:
        df: Input dataframe
        preserve_date_cols: List of columns to preserve as-is (e.g., date columns)

    Returns:
        dict with:
            - 'df': cleaned dataframe
            - 'summary': data quality and structural summary
    """
    preserve_date_cols = preserve_date_cols or []

    # Guard clause: callers may pass None/non-DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    df_original = df.copy()
    df = df.copy()

    # -----------------------------
    # Standardize column names
    # -----------------------------
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # -----------------------------
    # Data quality metrics (pre-clean)
    # -----------------------------
    total_rows = int(len(df))
    duplicate_rows = int(df.duplicated().sum())

    null_ratio = (
        {
            col: _clamp01(val)
            for col, val in df.isna().mean().to_dict().items()
        }
        if total_rows > 0
        else {}
    )

    # -----------------------------
    # Drop duplicates
    # -----------------------------
    df = df.drop_duplicates()

    # -----------------------------
    # Replace empty strings with NaN
    # -----------------------------
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # -----------------------------
    # Clean whitespace in object columns
    # -----------------------------
    for c in df.select_dtypes(include="object"):
        df[c] = df[c].astype(str).str.strip()

    # -----------------------------
    # Simple outlier signal (numeric only)
    # -----------------------------
    outlier_flags = {}
    outlier_kpis = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        coverage = _clamp01((len(series) / len(df)) if len(df) else 0.0)

        if series.empty:
            outlier_flags[col] = 0
            outlier_kpis[col] = _kpi_payload(0.0, 0.0, 0.0, coverage)
            continue

        std = _safe_float(series.std(), default=0.0)
        if std == 0.0:
            outlier_flags[col] = 0
            outlier_kpis[col] = _kpi_payload(0.0, coverage, 0.0, coverage)
            continue

        z_scores = (series - series.mean()) / std
        z_scores = z_scores.replace([np.inf, -np.inf], np.nan).dropna()

        outlier_count = int((z_scores.abs() > 3).sum())
        outlier_flags[col] = outlier_count

        signal_strength = _clamp01(
            1.0 - (outlier_count / max(len(series), 1))
        )

        outlier_kpis[col] = _kpi_payload(
            value=outlier_count,
            confidence=coverage,
            signal_strength=signal_strength,
            data_coverage=coverage,
        )

    # -----------------------------
    # Reset index
    # -----------------------------
    df = df.reset_index(drop=True)

    # -----------------------------
    # Summary (audit-friendly)
    # -----------------------------
    rows_after_cleaning = int(len(df))
    dedupe_ratio = _clamp01(
        (duplicate_rows / total_rows) if total_rows else 0.0
    )

    summary = {
        "rows_original": total_rows,
        "rows_after_cleaning": rows_after_cleaning,
        "columns": int(df.shape[1]),
        "duplicate_rows_removed": duplicate_rows,
        "null_ratio_by_column": null_ratio,
        "outlier_counts_by_column": outlier_flags,
        "dtypes": df.dtypes.to_dict(),
        "kpi": {
            "rows_original": _kpi_payload(
                total_rows, 1.0, 1.0, 1.0 if total_rows > 0 else 0.0
            ),
            "rows_after_cleaning": _kpi_payload(
                rows_after_cleaning,
                1.0,
                1.0,
                1.0 if total_rows > 0 else 0.0,
            ),
            "columns": _kpi_payload(
                int(df.shape[1]),
                1.0,
                1.0,
                1.0 if total_rows > 0 else 0.0,
            ),
            "duplicate_rows_removed": _kpi_payload(
                duplicate_rows,
                dedupe_ratio,
                1.0 - dedupe_ratio,
                1.0 if total_rows > 0 else 0.0,
            ),
            "null_ratio_by_column": {
                col: _kpi_payload(
                    ratio,
                    ratio,
                    1.0 - ratio,
                    1.0 if total_rows > 0 else 0.0,
                )
                for col, ratio in null_ratio.items()
            },
            "outlier_counts_by_column": outlier_kpis,
        },
    }

    return {
        "df": df,
        "summary": summary,
    }
