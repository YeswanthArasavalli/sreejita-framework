# =====================================================
# DOMAIN INTELLIGENCE — DETECTOR v2.1 (PHASE 4)
# Sreejita Framework v3.6.x
# =====================================================

from typing import Dict, Any, Optional, Tuple

from .column_normalizer import normalize_columns
from .intent_scoring import score_domain_intent

from sreejita.domains.utils.column_semantics import (
    infer_column_roles,
    summarize_roles,
    compute_generic_penalty,
)


# -------------------------------------------------
# SAFETY CONSTANTS (SEMANTIC, NOT TUNABLE)
# -------------------------------------------------

MIN_CONFIDENCE_FLOOR = 0.30     # minimum evidence sufficiency
MAX_CONFIDENCE_CAP = 1.0

# Rule evidence dominates; intent is supporting only
RULE_WEIGHT = 0.75
INTENT_WEIGHT = 0.25

INTENT_SCORE_MAX = 20.0
AMBIGUITY_DELTA = 0.10


# =====================================================
# DOMAIN SCORING ENGINE (AUTHORITATIVE)
# =====================================================

def compute_domain_scores(
    df,
    rule_based_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute evidence sufficiency scores per domain.

    PHASE-4 GUARANTEES:
    - Intent NEVER introduces domains
    - Rule-based evidence is dominant
    - Scores are explainable (component-based)
    - Generic schemas are penalized conservatively
    """

    if not rule_based_results:
        return {}

    # -----------------------------------------
    # DOMAIN-AGNOSTIC SCHEMA ANALYSIS
    # -----------------------------------------
    role_map = infer_column_roles(df)
    role_summary = summarize_roles(role_map)
    generic_penalty = compute_generic_penalty(role_summary)

    normalized_cols, _ = normalize_columns(df.columns)
    final_scores: Dict[str, Dict[str, Any]] = {}

    for domain, rb in rule_based_results.items():
        try:
            rule_conf = float(rb.get("confidence", 0.0))
        except Exception:
            rule_conf = 0.0

        rule_conf = max(0.0, min(rule_conf, 1.0))

        # 🚫 No rule evidence → no candidacy
        if rule_conf <= 0.0:
            continue

        # -------------------------------------
        # INTENT (SUPPORTING SIGNAL ONLY)
        # -------------------------------------
        intent_score, intent_signals = score_domain_intent(
            normalized_cols, domain
        )

        intent_conf = max(
            0.0,
            min(intent_score / INTENT_SCORE_MAX, 1.0),
        )

        # Weak intent does not contribute
        if intent_conf < 0.15:
            intent_conf = 0.0

        # -------------------------------------
        # EVIDENCE SUFFICIENCY SCORE
        # -------------------------------------
        combined = (
            RULE_WEIGHT * rule_conf +
            INTENT_WEIGHT * intent_conf
        )

        combined -= generic_penalty

        combined = round(
            max(0.0, min(combined, MAX_CONFIDENCE_CAP)),
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


# =====================================================
# DOMAIN CANDIDATE RESOLUTION (NON-FORCING)
# =====================================================

def select_best_domain(
    domain_scores: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Identify the best domain candidate WITHOUT forcing acceptance.

    RETURNS:
    - domain: Optional[str]
    - confidence: float
    - explanation: Dict[str, Any]

    PHASE-4 RULE:
    Detection ≠ acceptance. Router decides acceptance.
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
    # BELOW EVIDENCE FLOOR → UNKNOWN
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
