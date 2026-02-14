from typing import List, Dict, Any

from sreejita.core.insight_semantics import validate_insight_semantics


MIN_INSIGHT_SIGNAL = 0.4
MIN_INSIGHT_CONFIDENCE = 0.4


def normalize_and_validate_insights(
    insights: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Phase-2 Insight Governance

    Rules:
    - No insight without sufficient signal_strength
    - No insight without sufficient confidence
    - Weak insights are suppressed, not reworded
    - Semantic validation is applied only to surviving insights
    """
    validated: List[Dict[str, Any]] = []

    for insight in insights:
        if not isinstance(insight, dict):
            continue

        signal_strength = insight.get("signal_strength")
        confidence = insight.get("confidence")

        try:
            signal_strength = float(signal_strength)
        except Exception:
            signal_strength = None

        try:
            confidence = float(confidence)
        except Exception:
            confidence = None

        # -----------------------------
        # Phase-2 suppression rules
        # -----------------------------
        if signal_strength is None or signal_strength < MIN_INSIGHT_SIGNAL:
            validated.append({
                "status": "insufficient_data",
                "reason": "Insight suppressed due to weak or missing signal_strength",
                "signal_strength": signal_strength,
                "confidence": confidence,
            })
            continue

        if confidence is None or confidence < MIN_INSIGHT_CONFIDENCE:
            validated.append({
                "status": "insufficient_data",
                "reason": "Insight suppressed due to weak or missing confidence",
                "signal_strength": signal_strength,
                "confidence": confidence,
            })
            continue

        # -----------------------------
        # Semantic validation (only if allowed)
        # -----------------------------
        validated.append(validate_insight_semantics(insight))

    return validated
