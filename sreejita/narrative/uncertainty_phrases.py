"""Deterministic uncertainty phrasing for executive narratives."""

from typing import Any

from sreejita.narrative.language_tiers import confidence_tier


def uncertainty_phrase(confidence: Any) -> str:
    tier = confidence_tier(confidence)
    if tier == "high":
        return "Uncertainty remains visible: conclusions are supported by strong confidence signals."
    if tier == "moderate":
        return "Uncertainty remains visible: conclusions are directionally reliable and should be monitored."
    return "Uncertainty remains visible: conclusions are preliminary and should be treated cautiously."
