"""
Router v2.2 — Detector-Authoritative, Evidence-Governed
Sreejita Framework v3.6.x

PHASE-4 GUARANTEES:
- Unknown domain is first-class
- No forced domain selection
- Acceptance is evidence-governed
- Confidence is explainable
- Router owns acceptance, detectors do not
"""

from typing import Optional, Dict
import pandas as pd

from sreejita.domains.registry import registry
from sreejita.domains.contracts import DomainDetectionResult


# =====================================================
# GOVERNANCE CONSTANTS (SEMANTIC, LOCKED)
# =====================================================

MIN_CONFIDENCE_ACCEPT = 0.40     # sufficient evidence threshold
HINT_CONFIDENCE = 0.95           # explicit user trust
NO_DOMAIN_CONFIDENCE = 0.0       # explicit unknown
WEAK_SIGNAL_FLOOR = 0.20         # informational only (non-accepting)


# =====================================================
# DOMAIN DETECTION — AUTHORITATIVE
# =====================================================

def detect_domain(
    df: pd.DataFrame,
    *,
    domain_hint: Optional[str] = None,
    strict: bool = False,
) -> DomainDetectionResult:
    """
    Canonical domain detection (Phase 4).

    ROUTER RESPONSIBILITIES:
    - Aggregate detector outputs
    - Decide acceptance vs rejection
    - Preserve explainability
    - Never force a domain
    """

    # -------------------------------------------------
    # SAFETY: INVALID INPUT
    # -------------------------------------------------
    if not isinstance(df, pd.DataFrame) or df.empty:
        return DomainDetectionResult(
            domain=None,
            confidence=NO_DOMAIN_CONFIDENCE,
            signals={"reason": "empty_or_invalid_dataframe"},
        )

    # -------------------------------------------------
    # STEP 1: USER DOMAIN HINT (EXPLICIT, VERIFIED)
    # -------------------------------------------------
    hint_signals: Dict[str, object] = {}

    if isinstance(domain_hint, str) and domain_hint.strip():
        hint = domain_hint.strip().lower()

        if registry.has_domain(hint):
            return DomainDetectionResult(
                domain=hint,
                confidence=HINT_CONFIDENCE,
                signals={
                    "source": "user_hint",
                    "hint": hint,
                    "acceptance": "explicit",
                },
            )
        else:
            hint_signals["invalid_user_hint"] = hint

    # -------------------------------------------------
    # STEP 2: DETECTOR EVALUATION (NO ACCEPTANCE YET)
    # -------------------------------------------------
    best: Optional[DomainDetectionResult] = None
    all_scores: Dict[str, float] = {}

    for domain_name in registry.list_domains():
        detector = registry.get_detector(domain_name)
        if detector is None:
            continue

        try:
            result = detector.detect(df)

            if not isinstance(result, DomainDetectionResult):
                continue

            # Router owns execution lifecycle
            result.engine = None

            score = float(result.confidence or 0.0)
            all_scores[domain_name] = score

            if best is None or score > float(best.confidence or 0.0):
                best = result

        except Exception:
            # Detector failure must never break routing
            continue

    best_conf = float(best.confidence) if best else 0.0
    best_domain = best.domain if best else None

    # -------------------------------------------------
    # STEP 3: ACCEPTANCE GOVERNANCE (SEMANTIC)
    # -------------------------------------------------
    if (
        best
        and isinstance(best_domain, str)
        and best_domain
        and best_conf >= MIN_CONFIDENCE_ACCEPT
    ):
        return DomainDetectionResult(
            domain=best_domain,
            confidence=best_conf,
            signals={
                "source": "detector",
                "accepted": True,
                "best_candidate": best_domain,
                "best_confidence": round(best_conf, 3),
                "all_domain_scores": all_scores,
                **hint_signals,
            },
        )

    # -------------------------------------------------
    # STEP 4: EXPLICIT UNKNOWN DOMAIN
    # -------------------------------------------------
    signals = {
        "source": "detector",
        "accepted": False,
        "best_candidate": best_domain,
        "best_confidence": round(best_conf, 3),
        "all_domain_scores": all_scores,
        **hint_signals,
    }

    if strict:
        signals["reason"] = "strict_mode_reject"
        return DomainDetectionResult(
            domain=None,
            confidence=best_conf,
            signals=signals,
        )

    signals["reason"] = "insufficient_domain_evidence"

    return DomainDetectionResult(
        domain=None,
        confidence=max(best_conf, WEAK_SIGNAL_FLOOR),
        signals=signals,
    )


# =====================================================
# LEGACY HELPER — PREPROCESS ONLY (SAFE)
# =====================================================

def apply_domain(
    df: pd.DataFrame,
    *,
    domain_hint: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply domain preprocessing ONLY if domain is confidently detected.

    GUARANTEES:
    - Never forces a domain
    - Never mutates original dataframe
    - Safe under weak or ambiguous detection
    """

    result = detect_domain(df, domain_hint=domain_hint)

    if (
        not result
        or not result.domain
        or float(result.confidence) < MIN_CONFIDENCE_ACCEPT
    ):
        return df

    domain = registry.get_domain(result.domain)
    if domain is None:
        return df

    try:
        # Domain preprocess must be pure
        return domain.preprocess(df)
    except Exception:
        return df
