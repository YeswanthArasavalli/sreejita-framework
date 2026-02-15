"""Centralized narrative language tiers for executive-safe messaging."""

from typing import Any


def _normalize_confidence(confidence: Any) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return round(value, 2)


def confidence_tier(confidence: Any) -> str:
    value = _normalize_confidence(confidence)
    if value >= 0.8:
        return "high"
    if value >= 0.65:
        return "moderate"
    return "limited"


def confidence_tone(confidence: Any) -> str:
    tier = confidence_tier(confidence)
    if tier == "high":
        return "clear"
    if tier == "moderate":
        return "measured"
    return "cautious"
