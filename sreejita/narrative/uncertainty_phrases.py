"""
Deterministic, executive-safe uncertainty phrasing.

This module ensures:
- Uncertainty is always explicit
- Language is non-alarming and non-technical
- Scope and data coverage are clearly communicated
- Phrasing is deterministic and auditable
"""

from typing import Any

from sreejita.narrative.language_tiers import confidence_tier


def uncertainty_phrase(confidence: Any) -> str:
    """
    Returns an explicit uncertainty statement suitable for executive consumption.

    Uncertainty is never implied and never technical.
    """
    tier = confidence_tier(confidence)

    if tier == "high":
        return (
            "Based on available data, this view reflects a well-supported "
            "assessment within the current scope of measurement."
        )

    if tier == "moderate":
        return (
            "Based on available data, this view reflects a directional "
            "assessment within the current scope of measurement."
        )

    return (
        "Based on available data, this view reflects a preliminary "
        "assessment within the current scope of measurement."
    )
