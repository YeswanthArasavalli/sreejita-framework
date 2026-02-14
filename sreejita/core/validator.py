from dataclasses import dataclass
from typing import Any, Dict, Tuple
import math

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Return finite float to avoid NaN/Inf propagation in KPI fields."""
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
    """Build required KPI payload with guaranteed keys."""
    return {
        "value": _safe_float(value, default=0.0),
        "confidence": _clamp01(confidence),
        "signal_strength": _clamp01(signal_strength),
        "data_coverage": _clamp01(data_coverage),
    }


@dataclass
class DataQualityValidator:
    strict: bool = False

    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        results: Dict[str, Any] = {}

        # Guard clause: invalid input should fail safely
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()

        row_count = int(len(df))
        col_count = int(len(df.columns))
        coverage = 0.0 if row_count == 0 else 1.0

        # Backward-compatible raw metrics
        results["rows"] = row_count
        results["columns"] = col_count
        results["missing_values"] = {
            k: int(v) for k, v in df.isnull().sum().to_dict().items()
        }
        results["duplicate_rows"] = int(df.duplicated().sum())

        missing_ratio = {
            col: _clamp01((count / row_count) if row_count else 0.0)
            for col, count in results["missing_values"].items()
        }

        passed = True
        if self.strict:
            passed = results["duplicate_rows"] == 0

        duplicate_ratio = _clamp01(
            (results["duplicate_rows"] / row_count) if row_count else 0.0
        )

        # KPI channel with guaranteed required fields
        results["kpi"] = {
            "rows": _kpi_payload(row_count, 1.0, 1.0, coverage),
            "columns": _kpi_payload(col_count, 1.0, 1.0, coverage),
            "duplicate_rows": _kpi_payload(
                results["duplicate_rows"],
                confidence=1.0 - duplicate_ratio,
                signal_strength=1.0 - duplicate_ratio,
                data_coverage=coverage,
            ),
            "missing_values": {
                col: _kpi_payload(
                    count,
                    confidence=1.0 - missing_ratio.get(col, 0.0),
                    signal_strength=1.0 - missing_ratio.get(col, 0.0),
                    data_coverage=coverage,
                )
                for col, count in results["missing_values"].items()
            },
            "missing_ratio": {
                col: _kpi_payload(
                    ratio,
                    confidence=1.0 - ratio,
                    signal_strength=1.0 - ratio,
                    data_coverage=coverage,
                )
                for col, ratio in missing_ratio.items()
            },
            "validation_passed": _kpi_payload(
                1.0 if passed else 0.0,
                confidence=1.0,
                signal_strength=1.0,
                data_coverage=coverage,
            ),
        }

        return passed, results
