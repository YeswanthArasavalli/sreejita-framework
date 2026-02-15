# =====================================================
# EXECUTIVE COGNITION — UNIVERSAL (FINAL, GOVERNED)
# Sreejita Framework v3.6 (STEP-7 COMPLIANT)
# =====================================================

from typing import Dict, Any, List
from sreejita.core.capabilities import Capability
from sreejita.narrative.language_tiers import confidence_tone, confidence_tier
from sreejita.narrative.uncertainty_phrases import uncertainty_phrase

# =====================================================
# DOMAIN EXECUTIVE PROFILES (POLICY ONLY)
# =====================================================

EXECUTIVE_DOMAIN_PROFILES = {
    "healthcare": {
        "escalate_info": True,
        "warning_penalty": 6,
        "risk_penalty": 12,
        "confidence_floor": 0.75,
        "mixed_domain_penalty": 1.0,
        "readiness_bias": -5,
        "tone": "clinical",
    },
    "retail": {
        "escalate_info": False,
        "warning_penalty": 3,
        "risk_penalty": 6,
        "confidence_floor": 0.65,
        "mixed_domain_penalty": 0.5,
        "readiness_bias": 10,
        "tone": "commercial",
    },
    "finance": {
        "escalate_info": False,
        "warning_penalty": 4,
        "risk_penalty": 8,
        "confidence_floor": 0.70,
        "mixed_domain_penalty": 0.6,
        "readiness_bias": 5,
        "tone": "financial",
    },
    "marketing": {
        "escalate_info": False,
        "warning_penalty": 2,
        "risk_penalty": 4,
        "confidence_floor": 0.60,
        "mixed_domain_penalty": 0.4,
        "readiness_bias": 15,
        "tone": "growth",
    },
    "supply_chain": {
        "escalate_info": True,
        "warning_penalty": 5,
        "risk_penalty": 10,
        "confidence_floor": 0.70,
        "mixed_domain_penalty": 0.8,
        "readiness_bias": 0,
        "tone": "operational",
    },
    "hr": {
        "escalate_info": False,
        "warning_penalty": 4,
        "risk_penalty": 7,
        "confidence_floor": 0.65,
        "mixed_domain_penalty": 0.5,
        "readiness_bias": 5,
        "tone": "people",
    },
    "customer": {
        "escalate_info": False,
        "warning_penalty": 3,
        "risk_penalty": 5,
        "confidence_floor": 0.60,
        "mixed_domain_penalty": 0.4,
        "readiness_bias": 10,
        "tone": "experience",
    },
    "customer_value": {
        "escalate_info": False,
        "warning_penalty": 3,
        "risk_penalty": 6,
        "confidence_floor": 0.65,
        "mixed_domain_penalty": 0.4,
        "readiness_bias": 8,
        "tone": "commercial",
    },
}

DEFAULT_PROFILE = EXECUTIVE_DOMAIN_PROFILES["retail"]


def get_domain_profile(domain: str) -> Dict[str, Any]:
    return EXECUTIVE_DOMAIN_PROFILES.get(domain, DEFAULT_PROFILE)


# =====================================================
# SAFE HELPERS
# =====================================================

EXECUTIVE_RISK_BANDS = [
    (85, "VERY HIGH"),
    (70, "HIGH"),
    (50, "MODERATE"),
    (0, "LOW"),
]


def derive_risk_level(score: int) -> Dict[str, Any]:
    score = int(score or 0)
    for threshold, label in EXECUTIVE_RISK_BANDS:
        if score >= threshold:
            return {"label": label, "score": score}
    return {"label": "LOW", "score": score}


def infer_domain_from_kpis(kpis: Dict[str, Any]) -> str:
    return kpis.get("domain") or kpis.get("domain_name") or "unknown"


def format_kpi_name(key: str) -> str:
    name = key.replace("_", " ").title()
    for abbr in ("Ctr", "Cpc", "Cpm", "Cpa", "Roas"):
        name = name.replace(abbr, abbr.upper())
    return name


# =====================================================
# KPI SELECTION (NO FABRICATION)
# =====================================================

def select_executive_kpis(kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
    cap_map = kpis.get("_kpi_capabilities", {}) or {}
    conf_map = kpis.get("_confidence", {}) or {}

    ranked: List[Dict[str, Any]] = []

    for key, capability in cap_map.items():
        value = kpis.get(key)
        confidence = conf_map.get(key)

        if not isinstance(value, (int, float)):
            continue
        if not isinstance(confidence, (int, float)) or confidence < 0.5:
            continue

        weight = {
            Capability.QUALITY.value: 1.30,
            Capability.TIME_FLOW.value: 1.25,
            Capability.COST.value: 1.10,
            Capability.VOLUME.value: 1.00,
            Capability.VARIANCE.value: 1.15,
            Capability.ACCESS.value: 1.00,
        }.get(capability, 1.0)

        ranked.append({
            "key": key,
            "name": format_kpi_name(key),
            "value": round(value, 2),
            "capability": capability,
            "confidence": round(confidence, 2),
            "rank_score": round(confidence * weight, 3),
        })

    ranked.sort(key=lambda x: x["rank_score"], reverse=True)
    return ranked[:9]


# =====================================================
# INSIGHT STRUCTURING (SAFE, HONEST)
# =====================================================

def structure_insights(
    insights: List[Dict[str, Any]],
    domain: str,
) -> Dict[str, Any]:

    insights = [i for i in (insights or []) if isinstance(i, dict)]

    if not insights:
        return {
            "strengths": [],
            "warnings": [],
            "risks": [],
            "composite": {
                "title": "Insufficient Evidence",
                "summary": (
                    "Available data does not provide sufficient "
                    "high-confidence signals for executive insight."
                ),
                "confidence": 0.0,
            },
        }

    strengths = [i for i in insights if i.get("level") in ("STRENGTH", "OPPORTUNITY")][:3]
    warnings = [i for i in insights if i.get("level") == "WARNING"][:2]
    risks = [i for i in insights if i.get("level") == "RISK"][:1]

    avg_confidence = round(
        sum(float(i.get("confidence", 0.5)) for i in insights) / len(insights),
        2,
    )

    return {
        "strengths": strengths,
        "warnings": warnings,
        "risks": risks,
        "composite": {
            "title": "Executive Assessment",
            "summary": "Insights are governed by confidence and available evidence.",
            "confidence": avg_confidence,
        },
    }


# =====================================================
# BOARD READINESS (CONFIDENCE-GATED)
# =====================================================

def compute_board_readiness_score(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    domain: str,
) -> Dict[str, Any]:

    conf_map = kpis.get("_confidence", {}) or {}
    high_conf = [
        v for v in conf_map.values()
        if isinstance(v, (int, float)) and v >= 0.6
    ]

    if len(high_conf) < 2:
        return {
            "score": None,
            "band": "Insufficient Data",
        }

    score = min(100, int(len(high_conf) * 20))
    risk = derive_risk_level(score)

    return {
        "score": score,
        "band": risk["label"],
    }

# =====================================================
# EXECUTIVE BRIEF (NON-HALLUCINATING)
# =====================================================

def build_executive_brief(
    board_score: Any,
    insight_block: Dict[str, Any],
    sub_domain: str,
    domain: str,
    recommendations: List[Dict[str, Any]] = None,
) -> str:

    confidence = _extract_composite_confidence(insight_block)
    uncertainty = uncertainty_phrase(confidence)
    policy_statement = _policy_visibility_statement(board_score)

    if not isinstance(board_score, int):
        return (
            f"This review summarizes current performance in the {domain} domain. "
            "Evidence: board readiness is not displayed because confidence-gated KPI coverage is insufficient. "
            "Interpretation: executive direction is intentionally constrained until confidence coverage improves. "
            f"{uncertainty} "
            f"{policy_statement} "
            f"{_recommendation_explainability_statement(recommendations or [], confidence)}"
        )

        return (
        f"This review summarizes current performance in the {domain} domain. "
        f"{_compose_evidence_statement(domain, board_score)} "
        f"{_compose_interpretation_statement(confidence)} "
        f"{uncertainty} "
        f"{policy_statement} "
        f"{_recommendation_explainability_statement(recommendations or [], confidence)}"
    )




def _extract_composite_confidence(insight_block: Dict[str, Any]) -> float:
    composite = (insight_block or {}).get("composite", {})
    try:
        value = float(composite.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return round(value, 2)


def _policy_visibility_statement(board_score: Any) -> str:
    if isinstance(board_score, int):
        return (
            "Policy visibility: board readiness is shown because the confidence gate "
            "(at least two KPI confidence values >= 0.60) was satisfied."
        )
    return (
        "Policy visibility: board readiness is suppressed because the confidence gate "
        "(at least two KPI confidence values >= 0.60) was not satisfied."
    )


def _compose_evidence_statement(domain: str, board_score: int) -> str:
    risk = derive_risk_level(board_score)
    return (
        f"Evidence: current {domain} signals yield a Board Readiness Score of "
        f"{risk['score']} out of 100, mapping to a {risk['label'].lower()} readiness band."
    )


def _compose_interpretation_statement(confidence: float) -> str:
    tone = confidence_tone(confidence)
    tier = confidence_tier(confidence)

    if tier == "high":
        return (
            "Interpretation: this is a clear executive view based on consistently high-confidence insights."
        )
    if tier == "moderate":
        return (
            "Interpretation: this is a measured executive view; patterns are credible but should be tracked closely."
        )

    return (
        f"Interpretation: this is a {tone} executive view; decisions should prioritize reversible actions "
        "until confidence strengthens."
    )


def _recommendation_explainability_statement(
    recommendations: List[Dict[str, Any]],
    confidence: float,
) -> str:
    valid = [r for r in (recommendations or []) if isinstance(r, dict)]
    if not valid:
        return (
            "Recommendation policy: no recommendation is presented because no structured, attributable "
            "recommendation input was provided."
        )

    avg_conf = round(
        sum(float(r.get("confidence", 0.6)) for r in valid) / len(valid),
        2,
    )
    combined = round((avg_conf + confidence) / 2, 2)
    return (
        "Recommendation policy: recommendations are retained as provided and are presented as directional "
        f"actions aligned to available evidence (combined confidence context: {combined:.2f})."
    )
    
# =====================================================
# RECOMMENDATION NORMALIZATION (SAFE)
# =====================================================

def normalize_recommendations(
    recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    normalized: List[Dict[str, Any]] = []

    for r in (recommendations or [])[:5]:
        if not isinstance(r, dict):
            continue
        normalized.append({
            "priority": r.get("priority", "MEDIUM"),
            "action": r.get("action"),
            "owner": r.get("owner"),
            "timeline": r.get("timeline"),
            "goal": r.get("goal"),
            "confidence": round(float(r.get("confidence", 0.6)), 2),
        })

    return normalized


# =====================================================
# EXECUTIVE PAYLOAD (GLOBAL)
# =====================================================

def build_executive_payload(
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    domain: str = None,
) -> Dict[str, Any]:

    domain = domain or infer_domain_from_kpis(kpis)

    executive_kpis = select_executive_kpis(kpis)
    insight_block = structure_insights(insights, domain)
    board = compute_board_readiness_score(kpis, insights, domain)

    executive_brief = build_executive_brief(
        board.get("score"),
        insight_block,
        kpis.get("primary_sub_domain", "unknown"),
        domain,
        recommendations,
    )

    return {
        "executive_brief": executive_brief,
        "primary_kpis": executive_kpis,
        "insights": insight_block,
        "recommendations": normalize_recommendations(recommendations),
        "board_readiness": board,
        "sub_domain": kpis.get("primary_sub_domain"),
        "domain": domain,
    }


# =====================================================
# SUB-DOMAIN EXECUTIVE PAYLOADS (NON-THROWING)
# =====================================================

def build_subdomain_executive_payloads(*args, **kwargs) -> Dict[str, Dict[str, Any]]:
    try:
        kpis = kwargs.get("kpis") or next(
            (a for a in args if isinstance(a, dict)), {}
        )
        domain = kwargs.get("domain") or infer_domain_from_kpis(kpis)
        insights = kwargs.get("insights") or []
        recommendations = kwargs.get("recommendations") or []

        sub_domains = kpis.get("sub_domains", {}) or {}
        results: Dict[str, Dict[str, Any]] = {}

        for sub in sub_domains.keys():
            results[sub] = build_executive_payload(
                kpis,
                insights,
                recommendations,
                domain,
            )

        return results

    except Exception:
        return {}
