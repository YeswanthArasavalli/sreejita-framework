from typing import List, Dict, Any


def retail_recommendations(
    insights: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Phase-2 Recommendation Governance

    Rules:
    - No recommendation without explicit evidence
    - No heuristic string matching
    - No priority assignment without confidence
    - Suppress rather than guess
    """
    recommendations: List[Dict[str, Any]] = []

    for ins in insights:
        if not isinstance(ins, dict):
            continue

        # Evidence checks
        signal_strength = ins.get("signal_strength")
        confidence = ins.get("confidence")
        evidence = ins.get("evidence")

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
        if (
            signal_strength is None
            or signal_strength < 0.4
            or confidence is None
            or confidence < 0.4
            or not evidence
        ):
            recommendations.append({
                "status": "insufficient_data",
                "reason": (
                    "Recommendation suppressed due to missing or weak evidence "
                    "(signal_strength, confidence, or KPI linkage)"
                ),
                "signal_strength": signal_strength,
                "confidence": confidence,
                "evidence": evidence,
            })
            continue

        # -----------------------------
        # Explicit, evidence-backed recommendation only
        # -----------------------------
        recommendations.append({
            "action": ins.get("recommended_action"),
            "priority": ins.get("priority", "Medium"),
            "expected_impact": ins.get("expected_impact"),
            "signal_strength": signal_strength,
            "confidence": confidence,
            "evidence": evidence,
        })

    return recommendations
