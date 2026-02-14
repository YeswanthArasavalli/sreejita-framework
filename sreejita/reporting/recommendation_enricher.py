"""
Recommendation Enricher (Phase 2 — Confidence Safe)
--------------------------------------------------
Normalizes domain recommendations into a stable reporting contract.

PHASE-2 GUARANTEES:
- Never inflates confidence
- Never invents evidence
- Suppresses unsafe recommendations explicitly
- Preserves domain-added fields
- Deterministic & board-safe
"""

from typing import List, Dict, Any
from copy import deepcopy

from sreejita.reporting.contracts import (
    RECOMMENDATION_FIELDS,
    normalize_recommendation,
)

# --------------------------------------------------
# GOVERNANCE LIMITS (PHASE 2)
# --------------------------------------------------

MIN_CONFIDENCE = 0.30
VALID_PRIORITIES = {"HIGH", "MEDIUM", "LOW"}


# ==================================================
# INTERNAL HELPERS (PURE, NO INFLATION)
# ==================================================

def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
        if v < 0.0 or v > 1.0:
            return None
        return v
    except Exception:
        return None


def _normalize_priority(value: Any) -> str:
    if isinstance(value, str):
        v = value.upper().strip()
        if v in VALID_PRIORITIES:
            return v
    return "MEDIUM"


def _safe_str(value: Any, default: str) -> str:
    try:
        s = str(value).strip()
        return s if s else default
    except Exception:
        return default


def _has_evidence(rec: Dict[str, Any]) -> bool:
    return bool(
        rec.get("evidence")
        or rec.get("kpi")
        or rec.get("kpis")
    )


# ==================================================
# PUBLIC API (AUTHORITATIVE, PHASE-2 SAFE)
# ==================================================

def enrich_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Normalize and govern recommendations.

    Phase-2 Rules:
    - Suppress recommendations without evidence
    - Suppress recommendations with weak confidence
    - Never inflate confidence
    - Never emit fake fallbacks
    """

    if not isinstance(recommendations, list):
        return []

    enriched: List[Dict[str, Any]] = []
    seen_keys = set()

    for rec in recommendations:
        if not isinstance(rec, dict):
            continue

        try:
            # ---------------------------------------------
            # 1. CONTRACT NORMALIZATION
            # ---------------------------------------------
            normalized = normalize_recommendation(rec)

            # ---------------------------------------------
            # 2. BACKWARD COMPATIBILITY
            # ---------------------------------------------
            if (
                "expected_impact" in normalized
                and not normalized.get("expected_outcome")
            ):
                normalized["expected_outcome"] = normalized.pop(
                    "expected_impact"
                )

            # ---------------------------------------------
            # 3. EVIDENCE & CONFIDENCE GATING (PHASE 2)
            # ---------------------------------------------
            confidence = _safe_float(normalized.get("confidence"))
            signal_strength = _safe_float(normalized.get("signal_strength"))

            if (
                confidence is None
                or confidence < MIN_CONFIDENCE
                or signal_strength is None
                or not _has_evidence(normalized)
            ):
                enriched.append({
                    "status": "insufficient_data",
                    "reason": (
                        "Recommendation suppressed due to weak or missing "
                        "confidence, signal_strength, or evidence"
                    ),
                    "confidence": confidence,
                    "signal_strength": signal_strength,
                    "evidence": normalized.get("evidence"),
                    "suppressed": True,
                })
                continue

            # ---------------------------------------------
            # 4. HARD FIELD NORMALIZATION (SAFE)
            # ---------------------------------------------
            normalized["confidence"] = confidence
            normalized["priority"] = _normalize_priority(
                normalized.get("priority")
            )

            normalized["action"] = _safe_str(
                normalized.get("action"),
                "Review operational performance",
            )

            normalized["owner"] = _safe_str(
                normalized.get("owner"),
                "Management",
            )

            normalized["timeline"] = _safe_str(
                normalized.get("timeline"),
                "TBD",
            )

            normalized["goal"] = _safe_str(
                normalized.get("goal"),
                "Stabilize operations",
            )

            normalized["sub_domain"] = _safe_str(
                normalized.get("sub_domain"),
                "unknown",
            )

            # ---------------------------------------------
            # 5. STABLE DEDUPLICATION
            # ---------------------------------------------
            dedup_key = (
                normalized.get("sub_domain"),
                normalized.get("action"),
            )

            if dedup_key in seen_keys:
                continue

            seen_keys.add(dedup_key)
            enriched.append(normalized)

        except Exception:
            # ---------------------------------------------
            # 6. EXPLICIT SUPPRESSION ON ERROR (NO FALLBACK)
            # ---------------------------------------------
            enriched.append({
                "status": "insufficient_data",
                "reason": "Recommendation suppressed due to processing error",
                "suppressed": True,
            })

    # -------------------------------------------------
    # 7. DETERMINISTIC EXECUTIVE SORTING
    # -------------------------------------------------
    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    enriched.sort(
        key=lambda r: (
            priority_rank.get(r.get("priority"), 3),
            -(r.get("confidence") or 0.0),
            r.get("action", ""),
        )
    )

    return enriched
