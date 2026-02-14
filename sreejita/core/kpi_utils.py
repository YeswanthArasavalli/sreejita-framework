from typing import Any, Dict
import math
import pandas as pd


_REQUIRED_FIELDS = ("value", "confidence", "signal_strength", "data_coverage")


def _safe_float(value: Any) -> float:
    """Return finite float; map invalid/NaN/Inf values to 0.0 for numeric safety."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _clamp01(value: Any) -> float:
    """Clamp confidence-like scores into [0.0, 1.0]."""
    return max(0.0, min(_safe_float(value), 1.0))


def _coverage(df: pd.DataFrame, col: str) -> float:
    """Coverage = valid rows / total rows, with zero-row guard."""
    total_rows = int(len(df))
    if total_rows == 0:
        return 0.0

    numeric = pd.to_numeric(df[col], errors="coerce").replace(
        [math.inf, -math.inf], pd.NA
    )
    valid_rows = int(numeric.notna().sum())

    return _clamp01(valid_rows / total_rows)


def _empty_payload() -> Dict[str, float]:
    """Return a fully-formed empty KPI payload."""
    return {
        "value": 0.0,
        "confidence": 0.0,
        "signal_strength": 0.0,
        "data_coverage": 0.0,
    }


def has_columns(df: pd.DataFrame, *cols) -> bool:
    """Check presence of required columns with defensive guards."""
    if not isinstance(df, pd.DataFrame):
        return False
    return all(col in df.columns for col in cols)


def safe_sum(df: pd.DataFrame, col: str) -> Dict[str, float]:
    """
    Safely compute SUM KPI payload with explicit reliability metadata.
    """
    if not isinstance(df, pd.DataFrame) or col not in df.columns:
        return _empty_payload()

    numeric = pd.to_numeric(df[col], errors="coerce").replace(
        [math.inf, -math.inf], pd.NA
    )
    coverage = _coverage(df, col)

    # Guard clause: all values missing or non-numeric
    if numeric.dropna().empty:
        payload = _empty_payload()
        payload["data_coverage"] = coverage
        return payload

    value = _safe_float(numeric.sum())

    return {
        "value": value,
        "confidence": coverage,
        "signal_strength": coverage,
        "data_coverage": coverage,
    }


def safe_mean(df: pd.DataFrame, col: str) -> Dict[str, float]:
    """
    Safely compute MEAN KPI payload with explicit reliability metadata.
    """
    if not isinstance(df, pd.DataFrame) or col not in df.columns:
        return _empty_payload()

    numeric = pd.to_numeric(df[col], errors="coerce").replace(
        [math.inf, -math.inf], pd.NA
    )
    coverage = _coverage(df, col)

    # Guard clause: all values missing or non-numeric
    if numeric.dropna().empty:
        payload = _empty_payload()
        payload["data_coverage"] = coverage
        return payload

    value = _safe_float(numeric.mean())

    # Constant series → weak signal, even if coverage is high
    variance = _safe_float(numeric.var(ddof=0))
    signal_strength = coverage if variance > 0.0 else 0.0

    return {
        "value": value,
        "confidence": coverage,
        "signal_strength": signal_strength,
        "data_coverage": coverage,
    }
