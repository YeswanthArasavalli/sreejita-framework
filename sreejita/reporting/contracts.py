"""
Reporting Contracts (Phase 2 — Confidence Safe)
----------------------------------------------
Authoritative output contracts for reporting layer.

Phase-2 Rules:
- Never inflate confidence
- Never strengthen language by default
- Preserve suppression and status markers
- Domain engines may return partial objects
- Reporting layer normalizes shape, not meaning
"""

from typing import Dict, Any, List
from copy import deepcopy


# =====================================================
# RECOMMENDATION OUTPUT CONTRACT (AUTHORITATIVE)
# =====================================================

RECOMMENDATION_FIELDS: Dict[str, Any] = {
    "action": "",
    "priority": "MEDIUM",              # HIGH | MEDIUM | LOW
    "expected_outcome": "",
    "timeline": "TBD",
    "owner": "Business Team",
    "confidence": None,                # 0–1 (optional)
    "signal_strength": None,           # 0–1 (optional)
    "data_coverage": None,             # 0–1 (optional)
    "evidence": None,                  # KPI / reference
    "goal": "",
    "rationale": "",
    "sub_domain": None,
    "status": None,                    # detected | ambiguous | insufficient_data
    "suppressed": False,
}


def _safe_confidence(value: Any) -> float | None:
    try:
        v = float(value)
        if 0.0 <= v <= 1.0:
            return round(v, 2)
    except Exception:
        pass
    return None


def normalize_recommendation(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single recommendation.

    GUARANTEES:
    - Never raises
    - Shape-complete
    - Never inflates confidence
    - Preserves suppression/status flags
    """
    base = deepcopy(RECOMMENDATION_FIELDS)

    if not isinstance(rec, dict):
        return base

    for k, v in rec.items():
        base[k] = v

    if not base.get("action"):
        base["action"] = "Review operational performance"

    try:
        base["priority"] = str(base.get("priority", "MEDIUM")).upper()
    except Exception:
        base["priority"] = "MEDIUM"

    if base["priority"] not in {"HIGH", "MEDIUM", "LOW"}:
        base["priority"] = "MEDIUM"

    base["confidence"] = _safe_confidence(base.get("confidence"))
    base["signal_strength"] = _safe_confidence(base.get("signal_strength"))
    base["data_coverage"] = _safe_confidence(base.get("data_coverage"))

    return base


def normalize_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not isinstance(recommendations, list):
        return []

    return [normalize_recommendation(r) for r in recommendations]


# =====================================================
# INSIGHT OUTPUT CONTRACT (AUTHORITATIVE)
# =====================================================

INSIGHT_FIELDS: Dict[str, Any] = {
    "level": "INFO",                   # INFO | WARNING | RISK | STRENGTH
    "title": "",
    "so_what": "",
    "confidence": None,
    "signal_strength": None,
    "data_coverage": None,
    "sub_domain": None,
    "source": "",
    "status": None,                    # detected | ambiguous | insufficient_data
    "suppressed": False,
    "executive_summary_flag": False,
}


def normalize_insight(insight: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single insight.

    Phase-2 Rules:
    - Never upgrade level by default
    - Preserve suppression markers
    """
    base = deepcopy(INSIGHT_FIELDS)

    if not isinstance(insight, dict):
        return base

    for k, v in insight.items():
        base[k] = v

    if not base.get("title"):
        base["title"] = "Operational Observation"

    try:
        base["level"] = str(base.get("level", "INFO")).upper()
    except Exception:
        base["level"] = "INFO"

    if base["level"] not in {"INFO", "STRENGTH", "WARNING", "RISK"}:
        base["level"] = "INFO"

    base["confidence"] = _safe_confidence(base.get("confidence"))
    base["signal_strength"] = _safe_confidence(base.get("signal_strength"))
    base["data_coverage"] = _safe_confidence(base.get("data_coverage"))

    return base


def normalize_insights(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(insights, list):
        return []
    return [normalize_insight(i) for i in insights]


# =====================================================
# VISUAL OUTPUT CONTRACT (REFERENCE, PHASE-2 SAFE)
# =====================================================

VISUAL_FIELDS: Dict[str, Any] = {
    "path": "",
    "caption": "",
    "importance": 0.0,
    "confidence": None,
    "signal_strength": None,
    "data_coverage": None,
    "sub_domain": None,
    "inference_type": "direct",  # direct | proxy | fallback
    "status": None,
}


def normalize_visual(vis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single visual descriptor.

    Phase-2 Rules:
    - Visuals never imply confidence by default
    - Confidence remains optional
    """
    base = deepcopy(VISUAL_FIELDS)

    if not isinstance(vis, dict):
        return base

    for k, v in vis.items():
        base[k] = v

    try:
        base["importance"] = float(base.get("importance", 0.0))
    except Exception:
        base["importance"] = 0.0

    base["confidence"] = _safe_confidence(base.get("confidence"))
    base["signal_strength"] = _safe_confidence(base.get("signal_strength"))
    base["data_coverage"] = _safe_confidence(base.get("data_coverage"))

    return base


def normalize_visuals(visuals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(visuals, list):
        return []
    return [normalize_visual(v) for v in visuals]
