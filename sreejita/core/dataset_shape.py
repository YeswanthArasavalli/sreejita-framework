# =====================================================
# DATASET SHAPE DETECTOR — UNIVERSAL (LOCKED)
# Sreejita Framework v3.6 STABILIZED
# =====================================================

from enum import Enum
from typing import Dict, Any
import math
import pandas as pd


# =====================================================
# DATASET SHAPE ENUM (STRUCTURE ONLY — NOT DOMAIN)
# =====================================================

class DatasetShape(str, Enum):
    """
    Dataset SHAPE describes STRUCTURE, never business domain.
    """
    ROW_LEVEL_CLINICAL = "row_level_clinical"
    AGGREGATED_OPERATIONAL = "aggregated_operational"
    FINANCIAL_SUMMARY = "financial_summary"
    QUALITY_METRICS = "quality_metrics"
    UNKNOWN = "unknown"


# =====================================================
# NORMALIZATION (STRICT, NON-FUZZY)
# =====================================================

def _norm(col: str) -> str:
    col = str(col).lower().strip()
    col = col.replace(" ", "_").replace("-", "_")
    return col


def _safe_score(value: Any) -> float:
    """Ensure score-like values are finite and clamped to [0, 1]."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, min(out, 1.0))


def _metric_payload(value: Any, data_coverage: float) -> Dict[str, float]:
    """Standardized KPI-like payload for safe downstream consumption."""
    safe_value = _safe_score(value)
    safe_coverage = _safe_score(data_coverage)
    return {
        "value": safe_value,
        "confidence": safe_value,
        "signal_strength": safe_value,
        "data_coverage": safe_coverage,
    }


# =====================================================
# SHAPE DETECTION (CONSERVATIVE & EXPLAINABLE)
# =====================================================

def detect_dataset_shape(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect dataset STRUCTURAL shape.

    GUARANTEES:
    - Never raises
    - Conservative (UNKNOWN > wrong)
    - Healthcare-safe
    - Zero sub-domain inference
    """

    try:
        # -----------------------------
        # BASIC SAFETY
        # -----------------------------
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {
                "shape": DatasetShape.UNKNOWN,
                "confidence": 0.0,
                "signals": {},
                "signal_metrics": {},
                "kpi": _metric_payload(0.0, 0.0),
                "reason": "Empty or invalid dataset",
            }

        cols = {_norm(c) for c in df.columns}
        row_count = int(len(df))
        data_coverage = 0.0 if row_count == 0 else 1.0

        # -----------------------------
        # SCORE BUCKETS (0–1 RANGE)
        # -----------------------------
        score = {
            DatasetShape.ROW_LEVEL_CLINICAL: 0.0,
            DatasetShape.AGGREGATED_OPERATIONAL: 0.0,
            DatasetShape.FINANCIAL_SUMMARY: 0.0,
            DatasetShape.QUALITY_METRICS: 0.0,
        }

        reasons = {k: [] for k in score}

        # =================================================
        # ROW-LEVEL CLINICAL (HARD ANCHORS)
        # =================================================

        if any(
            any(tok in c for tok in ("patient", "mrn", "subject", "person"))
            for c in cols
        ):
            score[DatasetShape.ROW_LEVEL_CLINICAL] += 0.55
            reasons[DatasetShape.ROW_LEVEL_CLINICAL].append(
                "patient-level identifiers present"
            )

        if any(
            any(tok in c for tok in ("admit", "discharge", "diagnosis", "procedure"))
            for c in cols
        ):
            score[DatasetShape.ROW_LEVEL_CLINICAL] += 0.45
            reasons[DatasetShape.ROW_LEVEL_CLINICAL].append(
                "clinical event attributes present"
            )

        # -------------------------------------------------
        # CLINICAL TEMPORAL SIGNAL (WEAK BUT IMPORTANT)
        # -------------------------------------------------
        # Only boost if patient identifiers already exist.
        # Prevents false positives in non-clinical datasets.
        if score[DatasetShape.ROW_LEVEL_CLINICAL] > 0.0:
            if any(("date" in c or "time" in c) for c in cols):
                score[DatasetShape.ROW_LEVEL_CLINICAL] += 0.10
                reasons[DatasetShape.ROW_LEVEL_CLINICAL].append(
                    "clinical temporal attributes present"
                )

        # =================================================
        # AGGREGATED OPERATIONAL
        # =================================================

        if any(
            any(tok in c for tok in ("count", "volume", "total", "rate"))
            for c in cols
        ):
            score[DatasetShape.AGGREGATED_OPERATIONAL] += 0.5
            reasons[DatasetShape.AGGREGATED_OPERATIONAL].append(
                "aggregated operational metrics present"
            )

        if any(
            any(tok in c for tok in ("department", "unit", "service", "location"))
            for c in cols
        ):
            score[DatasetShape.AGGREGATED_OPERATIONAL] += 0.3
            reasons[DatasetShape.AGGREGATED_OPERATIONAL].append(
                "operational grouping attributes present"
            )

        # =================================================
        # QUALITY METRICS
        # =================================================

        if any(
            any(tok in c for tok in ("quality", "compliance", "benchmark", "score"))
            for c in cols
        ):
            score[DatasetShape.QUALITY_METRICS] += 0.6
            reasons[DatasetShape.QUALITY_METRICS].append(
                "quality/compliance indicators present"
            )

        if any(
            any(tok in c for tok in ("numerator", "denominator", "percentage", "pct"))
            for c in cols
        ):
            score[DatasetShape.QUALITY_METRICS] += 0.4
            reasons[DatasetShape.QUALITY_METRICS].append(
                "ratio-based quality metrics present"
            )

        # =================================================
        # FINANCIAL SUMMARY
        # =================================================

        financial_amount = any(
            any(tok in c for tok in ("cost", "charge", "revenue", "expense", "amount"))
            for c in cols
        )

        financial_grouping = any(
            any(tok in c for tok in ("cost_center", "gl_code", "ledger"))
            for c in cols
        )

        # 🔒 COST ALONE IS NOT A FINANCIAL SUMMARY
        if financial_amount and financial_grouping:
            score[DatasetShape.FINANCIAL_SUMMARY] += 1.0
            reasons[DatasetShape.FINANCIAL_SUMMARY].append(
                "financial amounts with accounting groupings"
            )

        # =================================================
        # CONFLICT RESOLUTION (STRUCTURE-FIRST)
        # =================================================

        if score[DatasetShape.ROW_LEVEL_CLINICAL] >= 0.6:
            score[DatasetShape.FINANCIAL_SUMMARY] *= 0.25
            score[DatasetShape.AGGREGATED_OPERATIONAL] *= 0.5

        if score[DatasetShape.QUALITY_METRICS] >= 0.6:
            score[DatasetShape.AGGREGATED_OPERATIONAL] *= 0.5

        # Normalize scores for absolute safety
        score = {shape: _safe_score(val) for shape, val in score.items()}

        signal_metrics = {
            shape.value: _metric_payload(val, data_coverage)
            for shape, val in score.items()
        }

        # =================================================
        # FINAL DECISION (SEMANTICALLY CORRECT)
        # =================================================

        best_shape = max(score, key=score.get)
        best_score = _safe_score(score[best_shape])

        # UNKNOWN only when there is literally no signal
        if best_score <= 0.0:
            return {
                "shape": DatasetShape.UNKNOWN,
                "confidence": 0.0,
                "signals": {k.value: v for k, v in score.items()},
                "signal_metrics": signal_metrics,
                "kpi": _metric_payload(0.0, data_coverage),
                "reason": "No dominant structural pattern",
            }

        return {
            "shape": best_shape,
            "confidence": round(best_score, 2),
            "signals": {k.value: v for k, v in score.items()},
            "signal_metrics": signal_metrics,
            "kpi": _metric_payload(best_score, data_coverage),
            "reason": "; ".join(reasons[best_shape]) or "Heuristic match",
        }

    except Exception:
        # 🔒 ABSOLUTE SAFETY FALLBACK
        return {
            "shape": DatasetShape.UNKNOWN,
            "confidence": 0.0,
            "signals": {},
            "signal_metrics": {},
            "kpi": _metric_payload(0.0, 0.0),
            "reason": "Shape detection failed safely",
        }
