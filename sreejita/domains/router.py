# =====================================================
# DOMAIN ROUTER — UNIVERSAL (AUTHORITATIVE, FINAL)
# Sreejita Framework v3.6
# =====================================================

from typing import List, Dict, Any
import logging

import pandas as pd

# -----------------------------------------------------
# DOMAIN IMPORTS
# -----------------------------------------------------

from sreejita.domains.retail import RetailDomain, RetailDomainDetector
from sreejita.domains.customer import CustomerDomain, CustomerDomainDetector
from sreejita.domains.customer_value import CustomerValueDomain, CustomerValueDomainDetector
from sreejita.domains.finance import FinanceDomain, FinanceDomainDetector
from sreejita.domains.ecommerce import EcommerceDomain, EcommerceDomainDetector
from sreejita.domains.healthcare import HealthcareDomain, HealthcareDomainDetector
from sreejita.domains.marketing import MarketingDomain, MarketingDomainDetector
from sreejita.domains.hr import HRDomain, HRDomainDetector
from sreejita.domains.supply_chain import SupplyChainDomain, SupplyChainDomainDetector

# 🚑 GENERIC — ABSOLUTE LAST RESORT
from sreejita.domains.generic import GenericDomain, GenericDomainDetector

# -----------------------------------------------------
# CORE FRAMEWORK
# -----------------------------------------------------

from sreejita.core.decision import DecisionExplanation
from sreejita.observability.hooks import DecisionObserver
from sreejita.core.fingerprint import dataframe_fingerprint

log = logging.getLogger("sreejita.router")

# =====================================================
# CONFIG (LOCKED)
# =====================================================

MIN_DOMAIN_CONFIDENCE = 0.45  # 🚨 hard guardrail

# =====================================================
# DOMAIN DETECTORS (GENERIC EXCLUDED FROM COMPETITION)
# =====================================================

DOMAIN_DETECTORS = [
    RetailDomainDetector(),
    CustomerDomainDetector(),
    CustomerValueDomainDetector(),  # 🆕 ADD HERE
    FinanceDomainDetector(),
    EcommerceDomainDetector(),
    HealthcareDomainDetector(),
    MarketingDomainDetector(),
    HRDomainDetector(),
    SupplyChainDomainDetector(),
]

GENERIC_DETECTOR = GenericDomainDetector()

# =====================================================
# DOMAIN IMPLEMENTATION FACTORY (DETERMINISTIC)
# =====================================================

_DOMAIN_FACTORY = {
    "retail": RetailDomain,
    "customer": CustomerDomain,
    "customer_value": CustomerValueDomain,  # 🆕 ADD
    "finance": FinanceDomain,
    "ecommerce": EcommerceDomain,
    "healthcare": HealthcareDomain,
    "marketing": MarketingDomain,
    "hr": HRDomain,
    "supply_chain": SupplyChainDomain,
    "generic": GenericDomain,
}


def _get_domain_engine(name: str):
    cls = _DOMAIN_FACTORY.get(name)
    try:
        return cls() if cls else GenericDomain()
    except Exception:
        return GenericDomain()

# =====================================================
# OBSERVABILITY
# =====================================================

_OBSERVERS: List[DecisionObserver] = []


def register_observer(observer: DecisionObserver):
    if observer:
        _OBSERVERS.append(observer)

# =====================================================
# DOMAIN DECISION ENGINE (AUTHORITATIVE)
# =====================================================

def decide_domain(df: pd.DataFrame) -> DecisionExplanation:
    """
    Stabilization-mode domain decision.

    GUARANTEES:
    - No domain is forced
    - Ambiguity is explicit
    - Generic is NOT auto-selected
    - Downstream execution can be safely gated
    """

    rule_results: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------
    # PHASE 1 — RULE-BASED DETECTION (READ-ONLY)
    # -------------------------------------------------
    for detector in DOMAIN_DETECTORS:
        try:
            result = detector.detect(df)
            if not result or not getattr(result, "domain", None):
                continue

            prev = rule_results.get(result.domain)
            if prev is None or result.confidence > prev["confidence"]:
                rule_results[result.domain] = {
                    "confidence": float(result.confidence or 0.0),
                    "signals": result.signals or {},
                    "detector": detector.__class__.__name__,
                }
        except Exception:
            continue

    # -------------------------------------------------
    # PHASE 2 — SELECT BEST DOMAIN (NO FORCING)
    # -------------------------------------------------
    selected_domain: Optional[str] = None
    confidence: float = 0.0
    status: str = "insufficient_data"
    meta: Dict[str, Any] = {}

    if rule_results:
        selected_domain, meta = max(
            rule_results.items(),
            key=lambda x: x[1]["confidence"],
        )
        confidence = float(meta.get("confidence", 0.0))

        if confidence >= MIN_DOMAIN_CONFIDENCE:
            status = "detected"
        else:
            selected_domain = None
            status = "ambiguous"

    # -------------------------------------------------
    # EXPLAINABILITY — ALTERNATIVES
    # -------------------------------------------------
    alternatives = [
        {
            "domain": d,
            "confidence": round(info["confidence"], 2),
        }
        for d, info in sorted(
            rule_results.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True,
        )
    ]

    # -------------------------------------------------
    # DECISION OBJECT (CANONICAL, HONEST)
    # -------------------------------------------------
    decision = DecisionExplanation(
        decision_type="domain_detection",
        selected_domain=selected_domain,
        confidence=round(confidence, 2),
        status=status,
        alternatives=alternatives,
        signals=meta.get("signals", {}),
        rules_applied=[
            "rule_based_detection",
            "highest_confidence_wins",
            "ambiguity_allowed",
        ],
        domain_scores={
            d: {"confidence": v["confidence"]}
            for d, v in rule_results.items()
        },
    )

    # -------------------------------------------------
    # ENGINE ATTACHMENT (ONLY IF SAFE)
    # -------------------------------------------------
    if status == "detected" and selected_domain in _DOMAIN_FACTORY:
        decision.attach_engine(_get_domain_engine(selected_domain))

    # -------------------------------------------------
    # TRACEABILITY
    # -------------------------------------------------
    decision.fingerprint = dataframe_fingerprint(df)

    # -------------------------------------------------
    # OBSERVABILITY
    # -------------------------------------------------
    for observer in _OBSERVERS:
        try:
            observer.record(decision)
        except Exception:
            pass

    return decision

# =====================================================
# DOMAIN PREPROCESSING (UTILITY)
# =====================================================

def apply_domain(df: pd.DataFrame, domain_name: str):
    """
    Apply ONLY domain-level preprocessing.
    No KPIs, no insights, no visuals.
    """
    cls = _DOMAIN_FACTORY.get(domain_name)
    if not cls:
        return df

    try:
        return cls().preprocess(df)
    except Exception:
        return df
