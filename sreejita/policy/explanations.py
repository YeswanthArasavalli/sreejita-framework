"""
Policy explanation utilities for executive-facing transparency.

This module:
- Makes policy actions visible and auditable
- Explains why insights or recommendations are gated, suppressed, or downgraded
- Never alters analytical or policy outcomes
- Uses executive-safe, non-technical language
"""

from typing import Any, Dict, Optional


# =====================================================
# CORE POLICY EXPLANATION INTERFACE
# =====================================================

def policy_explanation(
    *,
    action: str,
    reason: str,
    confidence: Optional[float] = None,
    policy_rule: Optional[str] = None,
) -> str:
    """
    Constructs a standardized executive-safe policy explanation.

    Parameters:
    - action: what happened (e.g., "suppressed", "downgraded", "not surfaced")
    - reason: plain-language reason (non-technical)
    - confidence: optional confidence context
    - policy_rule: optional policy identifier (for audit, not interpretation)

    Returns:
    - A single deterministic explanation sentence.
    """

    parts = [f"Policy decision: this item was {action} because {reason}."]

    if confidence is not None:
        try:
            conf = round(float(confidence), 2)
            parts.append(
                f"This decision reflects the current confidence context ({conf:.2f})."
            )
        except (TypeError, ValueError):
            pass

    if policy_rule:
        parts.append(f"Reference: policy rule '{policy_rule}'.")

    return " ".join(parts)


# =====================================================
# COMMON POLICY SCENARIOS (HELPERS)
# =====================================================

def low_confidence_suppression(
    confidence: Any,
    threshold: float,
) -> str:
    """
    Explains suppression due to insufficient confidence.
    """
    return policy_explanation(
        action="not surfaced",
        reason=(
            "available evidence does not meet the minimum confidence "
            f"threshold required for executive guidance ({threshold:.2f})"
        ),
        confidence=confidence,
        policy_rule="confidence_gate",
    )


def recommendation_blocked(
    confidence: Any,
) -> str:
    """
    Explains why a recommendation was blocked.
    """
    return policy_explanation(
        action="not presented",
        reason=(
            "recommendations are restricted in scenarios where confidence "
            "coverage is insufficient for responsible executive action"
        ),
        confidence=confidence,
        policy_rule="recommendation_confidence_policy",
    )


def severity_downgraded(
    original_level: str,
    new_level: str,
    confidence: Any,
) -> str:
    """
    Explains why severity was downgraded.
    """
    return policy_explanation(
        action="downgraded",
        reason=(
            f"the original severity level ('{original_level}') was adjusted "
            f"to '{new_level}' to align with current evidence strength"
        ),
        confidence=confidence,
        policy_rule="severity_alignment_policy",
    )


# =====================================================
# SAFE FALLBACK
# =====================================================

def generic_policy_notice(message: str) -> str:
    """
    Provides a safe fallback for uncommon policy explanations.
    """
    return f"Policy notice: {message.strip()}"
