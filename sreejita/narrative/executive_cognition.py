# =====================================================
# EXECUTIVE COGNITION — UNIVERSAL (FINAL, GOVERNED)
# Sreejita Framework v3.6 (STEP-7 COMPLIANT)
# =====================================================

from typing import Dict, Any, List
from sreejita.core.capabilities import Capability


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
) -> str:

    if not isinstance(board_score, int):
        return (
            "An executive assessment could not be generated due to "
            "insufficient high-confidence data."
        )

    risk = derive_risk_level(board_score)

    return (
        f"This review summarizes current performance in the {domain} domain. "
        f"Overall readiness is assessed as {risk['label'].lower()}, "
        f"with a Board Readiness Score of {risk['score']} out of 100."
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
