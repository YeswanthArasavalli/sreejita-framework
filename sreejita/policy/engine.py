from sreejita.core.decision import PolicyDecision


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
        status = "allowed"

        # -----------------------------------------
        # Rule 0: Explicitly unknown domain → BLOCK
        # (legacy / test contract)
        # -----------------------------------------
        if getattr(decision, "selected_domain", None) == "unknown":
            return PolicyDecision(
                status="blocked",
                reasons=["Domain explicitly marked as unknown"],
            )

        # -----------------------------------------
        # Rule 1: Insufficient data → allow but warn
        # -----------------------------------------
        if getattr(decision, "status", None) == "insufficient_data":
            status = "allowed_with_warning"
            reasons.append("Insufficient data to confidently classify domain")

        # -----------------------------------------
        # Rule 2: Ambiguous domain → allow but warn
        # -----------------------------------------
        if getattr(decision, "status", None) == "ambiguous":
            status = "allowed_with_warning"
            reasons.append("Domain classification is ambiguous")

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

        return PolicyDecision(
            status=status,
            reasons=reasons,
        )
