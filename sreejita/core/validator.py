from dataclasses import dataclass
from typing import Any, Dict, Tuple
import math

import pandas as pd


# =====================================================
# NUMERICAL SAFETY HELPERS
# =====================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Return finite float to avoid NaN/Inf propagation."""
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
    """
    Standardized KPI payload.

    Semantics:
    - value: raw metric (counts allowed)
    - confidence: trust in metric existence (coverage-based)
    - signal_strength: quality/cleanliness
    - data_coverage: availability of data
    """
    return {
        "value": _safe_float(value, default=0.0),
        "confidence": _clamp01(confidence),
        "signal_strength": _clamp01(signal_strength),
        "data_coverage": _clamp01(data_coverage),
    }


# =====================================================
# DATA QUALITY VALIDATOR
# =====================================================

@dataclass
class DataQualityValidator:
    """
    Deterministic, audit-safe data validator.

    - No inference
    - No hallucination
    - No domain assumptions
    """
    strict: bool = False

    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        results: Dict[str, Any] = {}

        # -----------------------------
        # Guard: invalid input
        # -----------------------------
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()

        row_count = int(len(df))
        col_count = int(len(df.columns))

        data_coverage = 0.0 if row_count == 0 else 1.0

        # -----------------------------
        # Raw metrics (backward compatible)
        # -----------------------------
        results["rows"] = row_count
        results["columns"] = col_count

        missing_counts = {
            k: int(v) for k, v in df.isnull().sum().to_dict().items()
        }
        duplicate_rows = int(df.duplicated().sum())

        results["missing_values"] = missing_counts
        results["duplicate_rows"] = duplicate_rows

        # -----------------------------
        # Derived ratios
        # -----------------------------
        missing_ratio = {
            col: _clamp01((count / row_count) if row_count else 0.0)
            for col, count in missing_counts.items()
        }

        duplicate_ratio = _clamp01(
            (duplicate_rows / row_count) if row_count else 0.0
        )

        # -----------------------------
        # Validation decision
        # -----------------------------
        passed = True
        if self.strict:
            passed = duplicate_rows == 0 and row_count > 0

        # -----------------------------
        # KPI CHANNEL (SEMANTICALLY CORRECT)
        # -----------------------------
        results["kpi"] = {
            "rows": _kpi_payload(
                row_count,
                confidence=data_coverage,
                signal_strength=1.0,
                data_coverage=data_coverage,
            ),
            "columns": _kpi_payload(
                col_count,
                confidence=data_coverage,
                signal_strength=1.0,
                data_coverage=data_coverage,
            ),
            "duplicate_rows": _kpi_payload(
                duplicate_rows,
                confidence=data_coverage,
                signal_strength=1.0 - duplicate_ratio,
                data_coverage=data_coverage,
            ),
            "missing_values": {
                col: _kpi_payload(
                    count,
                    confidence=data_coverage,
                    signal_strength=1.0 - missing_ratio.get(col, 0.0),
                    data_coverage=data_coverage,
                )
                for col, count in missing_counts.items()
            },
            "missing_ratio": {
                col: _kpi_payload(
                    ratio,
                    confidence=data_coverage,
                    signal_strength=1.0 - ratio,
                    data_coverage=data_coverage,
                )
                for col, ratio in missing_ratio.items()
            },
            "validation_passed": _kpi_payload(
                1.0 if passed else 0.0,
                confidence=data_coverage,
                signal_strength=1.0 if passed else 0.0,
                data_coverage=data_coverage,
            ),
        }

        return passed, results
