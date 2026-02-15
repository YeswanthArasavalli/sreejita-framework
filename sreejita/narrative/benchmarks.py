# sreejita/narrative/benchmarks.py

"""
UNIVERSAL EXECUTIVE BENCHMARKS & GUARDRAILS
==========================================

This module provides NON-BINDING, DOMAIN-AGNOSTIC
contextual language used to support executive narratives.

GUARANTEES:
- No analytics or scoring logic
- No mutation of observed values
- No domain-specific KPIs
- No decision thresholds
- Narrative context only

Benchmarks in this file:
- Do NOT evaluate performance
- Do NOT enforce policy
- Do NOT influence confidence
"""

from typing import Dict, Any, Optional


# =====================================================
# 1. CAPABILITY-LEVEL NARRATIVE CONTEXT
# =====================================================

CAPABILITY_BENCHMARKS: Dict[str, Dict[str, str]] = {
    "VOLUME": {
        "low": "Activity volume appears lower relative to typical operating patterns.",
        "normal": "Activity volume appears within an expected operating range.",
        "high": "Sustained activity volume may warrant capacity review.",
    },
    "TIME_FLOW": {
        "good": "Process cycle times appear within acceptable ranges.",
        "warning": "Observed delays suggest emerging flow constraints.",
        "critical": "Sustained delays may present operational risk.",
    },
    "COST": {
        "efficient": "Costs appear aligned with delivered value.",
        "warning": "Cost growth may be increasing faster than outcomes.",
        "critical": "Cost structure may present material financial risk.",
    },
    "QUALITY": {
        "stable": "Quality indicators appear stable.",
        "warning": "Early signs of quality degradation may be present.",
        "critical": "Quality performance may require immediate attention.",
    },
    "VARIANCE": {
        "low": "Performance variation appears well-controlled.",
        "high": "Significant variation may indicate standardization gaps.",
    },
    "ACCESS": {
        "adequate": "Access levels appear sufficient for current demand.",
        "limited": "Access constraints may affect outcomes or experience.",
    },
}


# =====================================================
# 2. CONFIDENCE CONTEXT LABELS (DESCRIPTIVE ONLY)
# =====================================================

CONFIDENCE_CONTEXT = [
    (0.85, "High confidence context"),
    (0.70, "Moderate confidence context"),
    (0.00, "Limited confidence context"),
]


def describe_confidence(confidence: Optional[float]) -> Dict[str, Any]:
    """
    Returns a descriptive confidence context.

    This function does NOT score, judge, or gate decisions.
    """
    if confidence is None:
        return {
            "label": "Unknown confidence context",
            "value": None,
        }

    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return {
            "label": "Unknown confidence context",
            "value": None,
        }

    for threshold, label in CONFIDENCE_CONTEXT:
        if value >= threshold:
            return {
                "label": label,
                "value": round(value, 2),
            }

    return {
        "label": "Limited confidence context",
        "value": round(value, 2),
    }


# =====================================================
# 3. GOVERNANCE REFERENCES (NON-OPERATIVE)
# =====================================================

GOVERNANCE_REFERENCES: Dict[str, Dict[str, str]] = {
    "COST": {
        "source": "Executive financial governance reference",
    },
    "TIME_FLOW": {
        "source": "Operational resilience reference",
    },
}


def get_governance_reference(capability: str) -> str:
    """
    Returns a human-readable governance reference.

    This is contextual only and does not impose limits.
    """
    return GOVERNANCE_REFERENCES.get(capability, {}).get("source", "")


# =====================================================
# 4. SAFE ACCESS HELPERS (CANONICAL, READ-ONLY)
# =====================================================

def get_capability_context(capability: str) -> Dict[str, str]:
    """
    Returns descriptive narrative context for a capability.

    Never returns evaluative or binding language.
    """
    return CAPABILITY_BENCHMARKS.get(capability, {}).copy()


# =====================================================
# END OF FILE — CONTEXT ONLY, NO ENFORCEMENT
# =====================================================
