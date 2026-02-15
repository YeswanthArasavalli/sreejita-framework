from sreejita.core.decision import PolicyDecision
from sreejita.policy.explanations import low_confidence_suppression


class PolicyEngine:
    """
    Stabilization-mode policy engine.

    GUARANTEES:
    - Ambiguity is not blocked
    - Weak confidence produces warnings, not hard stops
    - Explicitly unknown domains are blocked (legacy contract)
    - Deterministic, explainable outcomes
    """

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence

    def evaluate(self, decision):
        reasons = []
        explanations = []
        status = "allowed"

        # -----------------------------------------
        # Rule 0: Explicitly unknown domain → BLOCK
        # (legacy / test contract)
        # -----------------------------------------
        if getattr(decision, "selected_domain", None) == "unknown":
            decision_obj = PolicyDecision(
                status="blocked",
                reasons=["Domain explicitly marked as unknown"],
            )
            decision_obj.explanations = [
                "Policy decision: this item was blocked because the domain "
                "was explicitly marked as unknown, which is restricted under "
                "current governance rules."
            ]
            return decision_obj

        # -----------------------------------------
        # Rule 1: Insufficient data → allow but warn
        # -----------------------------------------
        if getattr(decision, "status", None) == "insufficient_data":
            status = "allowed_with_warning"
            reasons.append("Insufficient data to confidently classify domain")
            explanations.append(
                "Policy decision: this item is allowed with warning because "
                "available data coverage is insufficient for high-confidence "
                "classification."
            )

        # -----------------------------------------
        # Rule 2: Ambiguous domain → allow but warn
        # -----------------------------------------
        if getattr(decision, "status", None) == "ambiguous":
            status = "allowed_with_warning"
            reasons.append("Domain classification is ambiguous")
            explanations.append(
                "Policy decision: this item is allowed with warning because "
                "domain signals are directionally consistent but not definitive."
            )

        # -----------------------------------------
        # Rule 3: Low confidence → allow with warning
        # -----------------------------------------
        if (
            decision.confidence is not None
            and decision.confidence < self.min_confidence
        ):
            status = "allowed_with_warning"
            reasons.append(
                f"Domain confidence below minimum threshold ({self.min_confidence})"
            )
            explanations.append(
                low_confidence_suppression(
                    confidence=decision.confidence,
                    threshold=self.min_confidence,
                )
            )

        decision_obj = PolicyDecision(
            status=status,
            reasons=reasons,
        )
        decision_obj.explanations = explanations
        return decision_obj

