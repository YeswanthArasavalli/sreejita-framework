"""
Reporting Contracts
-------------------
Authoritative output contracts for reporting layer.

Rules:
- Domain engines may return partial or noisy objects
- Reporting layer MUST normalize them
- Contracts here are domain-agnostic
- Never raises
"""

from typing import Dict, Any, List
from copy import deepcopy


# =====================================================
# RECOMMENDATION OUTPUT CONTRACT (AUTHORITATIVE)
# =====================================================

RECOMMENDATION_FIELDS: Dict[str, Any] = {
    "action": "",                     # What to do (required)
    "priority": "MEDIUM",              # HIGH | MEDIUM | LOW
    "expected_outcome": "",            # Business / operational outcome
    "timeline": "TBD",                 # e.g. 30 days, Q2
    "owner": "Business Team",           # Accountable owner
    "confidence": None,                # 0–1 (optional)
    "goal": "",                        # Optional strategic intent
    "rationale": "",                   # Why this recommendation exists
    "sub_domain": None,                # Optional scoping
}


def normalize_recommendation(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single recommendation to contract.

    GUARANTEES:
    - Never raises
    - All required fields exist
    - Extra fields preserved (forward compatible)
    """
    base = deepcopy(RECOMMENDATION_FIELDS)

    if not isinstance(rec, dict):
        return base

    # Preserve all domain-provided fields
    for k, v in rec.items():
        base[k] = v

    # -------------------------------
    # HARD SAFETY RULES
    # -------------------------------
    if not base.get("action"):
        base["action"] = "Review operational performance"

    # Priority normalization
    try:
        base["priority"] = str(base.get("priority", "MEDIUM")).upper()
    except Exception:
        base["priority"] = "MEDIUM"

    if base["priority"] not in {"HIGH", "MEDIUM", "LOW"}:
        base["priority"] = "MEDIUM"

    # Confidence normalization
    if base.get("confidence") is not None:
        try:
            base["confidence"] = round(float(base["confidence"]), 2)
        except Exception:
            base["confidence"] = None

    return base


def normalize_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Normalize a list of recommendations.

    GUARANTEES:
    - Always returns a list
    - Each item is contract-compliant
    """
    if not isinstance(recommendations, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for rec in recommendations:
        try:
            normalized.append(normalize_recommendation(rec))
        except Exception:
            continue

    return normalized


# =====================================================
# INSIGHT OUTPUT CONTRACT (AUTHORITATIVE)
# =====================================================

INSIGHT_FIELDS: Dict[str, Any] = {
    "level": "INFO",                   # STRENGTH | WARNING | RISK
    "title": "",
    "so_what": "",
    "confidence": None,                # 0–1 (optional)
    "sub_domain": None,                # Optional
    "source": "",
    "executive_summary_flag": False,
}


def normalize_insight(insight: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single insight.

    GUARANTEES:
    - Never raises
    - Always executive-safe
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

    if base["level"] not in {"STRENGTH", "WARNING", "RISK"}:
        base["level"] = "WARNING"

    if base.get("confidence") is not None:
        try:
            base["confidence"] = round(float(base["confidence"]), 2)
        except Exception:
            base["confidence"] = None

    return base


def normalize_insights(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize a list of insights.
    """
    if not isinstance(insights, list):
        return []

    return [normalize_insight(i) for i in insights]


# =====================================================
# VISUAL OUTPUT CONTRACT (REFERENCE, EXECUTIVE SAFE)
# =====================================================

VISUAL_FIELDS: Dict[str, Any] = {
    "path": "",
    "caption": "",
    "importance": 0.0,     # 0–1 (ranking)
    "confidence": None,    # 0–1 (optional)
    "sub_domain": None,
    "inference_type": "direct",  # direct | proxy | fallback
}


def normalize_visual(vis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single visual descriptor.

    GUARANTEES:
    - Never raises
    - Safe defaults
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

    if base.get("confidence") is not None:
        try:
            base["confidence"] = round(float(base["confidence"]), 2)
        except Exception:
            base["confidence"] = None

    return base


def normalize_visuals(visuals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize a list of visuals.
    """
    if not isinstance(visuals, list):
        return []

    return [normalize_visual(v) for v in visuals]
