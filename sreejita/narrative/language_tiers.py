"""
Centralized narrative language tiers for executive-safe, deterministic messaging.

This module:
- Owns all confidence-based narrative differentiation
- Prevents wording leakage into domain or cognition layers
- Ensures consulting-grade, board-safe language
"""

from typing import Any


# =====================================================
# CONFIDENCE NORMALIZATION (SAFE)
# =====================================================

def _normalize_confidence(confidence: Any) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.0

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0

    return round(value, 2)


# =====================================================
# CONFIDENCE TIERS (CANONICAL)
# =====================================================

def confidence_tier(confidence: Any) -> str:
    """
    Returns the canonical confidence tier.

    Tiers are intentionally coarse to avoid false precision.
    """
    value = _normalize_confidence(confidence)

    if value >= 0.75:
        return "high"
    if value >= 0.50:
        return "moderate"
    return "limited"


# =====================================================
# EXECUTIVE TONE LABELS (NON-NARRATIVE)
# =====================================================

def confidence_tone(confidence: Any) -> str:
    """
    Returns a neutral executive tone descriptor.

    These labels are safe to embed in sentences but do not
    themselves constitute narrative wording.
    """
    tier = confidence_tier(confidence)

    if tier == "high":
        return "clear"
    if tier == "moderate":
        return "measured"
    return "cautious"


# =====================================================
# EXECUTIVE LANGUAGE PRIMITIVES (OPTIONAL, SAFE)
# =====================================================

def executive_assessment_phrase(confidence: Any) -> str:
    """
    Standardized executive assessment framing.

    Use this when introducing summaries or briefs.
    """
    tier = confidence_tier(confidence)

    if tier == "high":
        return "provides a clear executive assessment"
    if tier == "moderate":
        return "provides a measured executive assessment"
    return "provides a limited executive assessment"


def interpretation_strength_phrase(confidence: Any) -> str:
    """
    Standardized interpretation framing.

    Separates interpretation strength from evidence.
    """
    tier = confidence_tier(confidence)

    if tier == "high":
        return "patterns are well-supported by consistent signals"
    if tier == "moderate":
        return "patterns are directionally supported by available signals"
    return "patterns are indicative but constrained by limited signal strength"
