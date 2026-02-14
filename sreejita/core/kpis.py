from typing import Dict, Any, Optional
import math
import pandas as pd


_REQUIRED_KPI_FIELDS = ("value", "confidence", "signal_strength", "data_coverage")


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float value or a safe default."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(out):
        return default

    return out


def _normalize_score(value: Any, default: float = 0.0) -> float:
    """Normalize confidence-like scores to the [0, 1] range."""
    out = _finite_float(value, default=default)
    return max(0.0, min(out, 1.0))


def _safe_value(value: Any) -> Any:
    """Prevent NaN or Inf leakage into KPI payloads."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(float(value)) else 0.0
    return value


def _normalize_kpi_payload(raw_value: Any, default_coverage: float) -> Dict[str, Any]:
    """
    Ensure every KPI is returned with complete, numerically safe fields.
    """
    if isinstance(raw_value, dict):
        return {
            "value": _safe_value(raw_value.get("value", 0.0)),
            "confidence": _normalize_score(raw_value.get("confidence", 0.0), default=0.0),
            "signal_strength": _normalize_score(
                raw_value.get("signal_strength", 0.0), default=0.0
            ),
            "data_coverage": _normalize_score(
                raw_value.get("data_coverage", default_coverage),
                default=default_coverage,
            ),
        }

    finite_value = _safe_value(raw_value)
    has_numeric_value = isinstance(finite_value, (int, float))

    return {
        "value": finite_value,
        "confidence": 1.0 if has_numeric_value else 0.0,
        "signal_strength": 1.0 if has_numeric_value else 0.0,
        "data_coverage": default_coverage,
    }


def compute_kpis(
    df: pd.DataFrame,
    base_kpis: Optional[Dict[str, Any]] = None,
    domain_kpis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge structural, base, and domain-specific KPIs into a single dictionary.

    This function is intentionally lightweight and deterministic.
    It does not calculate KPIs itself — it only consolidates them
    for reporting, insights, and narrative layers.

    Args:
        df: Input dataframe
        base_kpis: Framework-level KPIs (e.g., quality, volume)
        domain_kpis: Domain-specific KPIs (e.g., LOS, ROAS, GMROI)

    Returns:
        Dictionary of consolidated KPIs
    """

    # Guard clause: upstream callers may pass None or invalid objects
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    row_count = int(df.shape[0])
    default_coverage = 0.0 if row_count == 0 else 1.0

    # -----------------------------
    # Structural KPIs (always present)
    # -----------------------------
    kpis: Dict[str, Any] = {
        "row_count": _normalize_kpi_payload(row_count, default_coverage),
        "column_count": _normalize_kpi_payload(int(df.shape[1]), default_coverage),
    }

    # -----------------------------
    # Base KPIs (optional)
    # -----------------------------
    if isinstance(base_kpis, dict):
        for key, value in base_kpis.items():
            # Avoid overwriting structural KPIs
            if key in kpis:
                continue

            # Preserve metadata channels untouched
            if isinstance(key, str) and key.startswith("_"):
                kpis[key] = value
            else:
                kpis[key] = _normalize_kpi_payload(value, default_coverage)

    # -----------------------------
    # Domain KPIs (optional)
    # -----------------------------
    if isinstance(domain_kpis, dict):
        for key, value in domain_kpis.items():
            # Preserve metadata channels untouched
            if isinstance(key, str) and key.startswith("_"):
                kpis[key] = value
            else:
                kpis[key] = _normalize_kpi_payload(value, default_coverage)

    return kpis
