# =====================================================
# DOMAIN INTELLIGENCE — DETECTOR v2.1 (PHASE 4)
# Sreejita Framework v3.6.x
# =====================================================

from .column_normalizer import normalize_columns
from .intent_scoring import score_domain_intent

from sreejita.core.column_roles import (
    infer_column_roles,
    summarize_roles,
    compute_generic_penalty,
)

# -------------------------------------------------
# SAFETY CONSTANTS (SEMANTIC)
# -------------------------------------------------

MIN_CONFIDENCE_FLOOR = 0.30     # minimum evidence sufficiency
MAX_CONFIDENCE_CAP = 1.0

# Rule evidence is dominant
INTENT_WEIGHT = 0.25
RULE_WEIGHT = 0.75

INTENT_SCORE_MAX = 20.0
AMBIGUITY_DELTA = 0.10


# -------------------------------------------------
# DOMAIN SCORING ENGINE (AUTHORITATIVE)
# -------------------------------------------------

def compute_domain_scores(df, rule_based_results):
    """
    Compute evidence sufficiency scores per domain.

    PHASE-4 GUARANTEES:
    - No domain is introduced by intent
    - Rule evidence is primary
    - Scores are explainable, not probabilistic
    """

    if not rule_based_results:
        return {}

    # -----------------------------------------
    # SCHEMA ANALYSIS (GENERICITY PENALTY)
    # -----------------------------------------
    role_map = infer_column_roles(df)
    role_summary = summarize_roles(role_map)
    generic_penalty = compute_generic_penalty(role_summary)

    normalized_cols, _ = normalize_columns(df.columns)
    final_scores = {}

    for domain, rb in rule_based_results.items():
        rule_conf = float(rb.get("confidence", 0.0))
        rule_conf = max(0.0, min(rule_conf, 1.0))

        # 🚫 No rule evidence → no candidacy
        if rule_conf <= 0.0:
            continue

        # -------------------------------
        # INTENT (SUPPORTING SIGNAL ONLY)
        # -------------------------------
        intent_score, intent_signals = score_domain_intent(
            normalized_cols, domain
        )

        intent_conf = min(
            max(intent_score / INTENT_SCORE_MAX, 0.0),
            1.0,
        )

        if intent_conf < 0.15:
            intent_conf = 0.0

        # -------------------------------
        # EVIDENCE SUFFICIENCY SCORE
        # -------------------------------
        combined = (
            RULE_WEIGHT * rule_conf +
            INTENT_WEIGHT * intent_conf
        )

        combined -= generic_penalty

        combined = round(
            min(MAX_CONFIDENCE_CAP, max(combined, 0.0)),
            3,
        )

        final_scores[domain] = {
            "confidence": combined,
            "components": {
                "rule_confidence": round(rule_conf, 3),
                "intent_confidence": round(intent_conf, 3),
                "generic_penalty": round(generic_penalty, 3),
            },
            "signals": {
                "rule_based": rb.get("signals", {}),
                "intent_based": intent_signals or {},
                "schema_roles": role_summary,
            },
        }

    return final_scores


# -------------------------------------------------
# DOMAIN CANDIDATE RESOLUTION (NO ACCEPTANCE)
# -------------------------------------------------

def select_best_domain(domain_scores):
    """
    Identify best domain candidate WITHOUT forcing selection.

    Returns:
    - domain: Optional[str]
    - confidence: float
    - explanation: Dict[str, Any]
    """

    if not domain_scores:
        return None, 0.0, {
            "reason": "no_domain_scores"
        }

    ordered = sorted(
        domain_scores.items(),
        key=lambda x: x[1].get("confidence", 0.0),
        reverse=True,
    )

    top_domain, top_meta = ordered[0]
    top_conf = float(top_meta.get("confidence", 0.0))

    # -------------------------------------------------
    # BELOW SUFFICIENCY FLOOR → UNKNOWN
    # -------------------------------------------------
    if top_conf < MIN_CONFIDENCE_FLOOR:
        return None, top_conf, {
            "reason": "below_confidence_floor",
            "best_candidate": top_domain,
            "best_confidence": round(top_conf, 3),
            "all_domain_scores": {
                d: m.get("confidence", 0.0)
                for d, m in domain_scores.items()
            },
        }

    # -------------------------------------------------
    # AMBIGUITY CHECK
    # -------------------------------------------------
    if len(ordered) > 1:
        second_domain, second_meta = ordered[1]
        second_conf = float(second_meta.get("confidence", 0.0))

        if abs(top_conf - second_conf) <= AMBIGUITY_DELTA:
            return None, top_conf, {
                "reason": "ambiguous_top_domains",
                "top_candidates": [
                    {
                        "domain": top_domain,
                        "confidence": round(top_conf, 3),
                    },
                    {
                        "domain": second_domain,
                        "confidence": round(second_conf, 3),
                    },
                ],
                "all_domain_scores": {
                    d: m.get("confidence", 0.0)
                    for d, m in domain_scores.items()
                },
            }

    # -------------------------------------------------
    # SINGLE STRONG CANDIDATE (NOT ACCEPTED HERE)
    # -------------------------------------------------
    return top_domain, top_conf, {
        "reason": "single_strong_candidate",
        "domain": top_domain,
        "confidence": round(top_conf, 3),
        "components": top_meta.get("components", {}),
        "all_domain_scores": {
            d: m.get("confidence", 0.0)
            for d, m in domain_scores.items()
        },
    }
