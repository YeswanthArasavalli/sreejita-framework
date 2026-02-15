# =====================================================
# DOMAIN INTENT SCORING — STABILIZED & BOUNDARY-SAFE
# Sreejita Framework v3.6 (FINAL)
# =====================================================

"""
PURPOSE:
- Provide SOFT semantic alignment signals
- Assist explainability & UI diagnostics
- NEVER override domain detectors
- NEVER create or force domain selection

INTENT SCORING IS:
✔ Advisory
✔ Conservative
✔ Detector-subordinate

INTENT SCORING IS NOT:
✘ A classifier
✘ A router
✘ A decision-maker
"""

from typing import Iterable, Tuple, Dict, Any, Set

from .domain_intents import DOMAIN_INTENTS


# -------------------------------------------------
# WEIGHTS (CONSERVATIVE BY DESIGN)
# -------------------------------------------------

HIGH_WEIGHT = 3.0          # strong semantic alignment
AMBIGUOUS_WEIGHT = 0.5    # weak contextual hint only

# Absolute hard cap — prevents intent inflation
MAX_INTENT_SCORE = 10.0


# -------------------------------------------------
# DOMAIN-EXCLUSIVE CONFLICT SIGNALS
# -------------------------------------------------
# Presence of these signals REDUCES confidence
# They NEVER fully eliminate intent (graceful degradation)

DOMAIN_EXCLUSIVE_SIGNALS: Dict[str, Set[str]] = {
    "healthcare": {
        "revenue", "profit", "margin", "invoice",
        "sku", "inventory", "sales",
    },
    "finance": {
        "patient", "diagnosis", "treatment",
        "clinical", "mortality",
    },
    "customer": {
        "salary", "ctc", "payroll",
        "attrition", "leave", "attendance",
        "performance",
    },
    "hr": {
        "customer", "order", "purchase",
        "cart", "checkout",
    },
}


# -------------------------------------------------
# INTENT SCORING (ADVISORY ONLY)
# -------------------------------------------------

def score_domain_intent(
    normalized_columns: Iterable[str],
    domain: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Score semantic intent alignment for a SINGLE domain.

    INPUT:
    - normalized_columns: iterable of normalized column tokens
    - domain: candidate domain name

    OUTPUT:
    - score: float (0.0 → MAX_INTENT_SCORE)
    - signals: structured explanation payload

    GUARANTEES:
    - Never raises
    - Never returns negative scores
    - Never introduces domains
    - Never exceeds MAX_INTENT_SCORE
    """

    try:
        if not domain or domain not in DOMAIN_INTENTS:
            return 0.0, {}

        tokens = set(normalized_columns or [])

        intents = DOMAIN_INTENTS.get(domain, {})
        high_tokens = intents.get("high", set())
        amb_tokens = intents.get("ambiguous", set())

        # ---------------------------------------------
        # MATCHES
        # ---------------------------------------------
        high_hits = high_tokens.intersection(tokens)
        amb_hits = amb_tokens.intersection(tokens)

        # ---------------------------------------------
        # BASE SCORE
        # ---------------------------------------------
        score = (
            len(high_hits) * HIGH_WEIGHT +
            len(amb_hits) * AMBIGUOUS_WEIGHT
        )

        # ---------------------------------------------
        # EXCLUSIVE CONFLICT PENALTY (SOFT)
        # ---------------------------------------------
        exclusive = DOMAIN_EXCLUSIVE_SIGNALS.get(domain)
        conflict_hits = set()

        if exclusive:
            conflict_hits = exclusive.intersection(tokens)
            if conflict_hits:
                score *= 0.5  # soft but meaningful penalty

        # ---------------------------------------------
        # HARD SAFETY CAPS
        # ---------------------------------------------
        score = max(0.0, min(float(score), MAX_INTENT_SCORE))

        # ---------------------------------------------
        # EXPLANATION PAYLOAD (CLEAN & UI-SAFE)
        # ---------------------------------------------
        signals = {
            "domain": domain,
            "high_confidence_matches": sorted(high_hits),
            "ambiguous_matches": sorted(amb_hits),
            "conflicting_signals": sorted(conflict_hits),
            "raw_token_count": len(tokens),
        }

        return score, signals

    except Exception:
        # Absolute safety — intent layer must never crash
        return 0.0, {}
