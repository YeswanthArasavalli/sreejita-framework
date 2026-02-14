# =====================================================
# DOMAIN SIGNALS — AUTHORITATIVE EVIDENCE LAYER
# Phase 4 — Detection Robustness
# Sreejita Framework v3.6.x
# =====================================================

from typing import Dict, Set, Any
import pandas as pd


# -------------------------------------------------
# DOMAIN SIGNAL DEFINITIONS (STRONG INDICATORS)
# -------------------------------------------------
# RULES:
# - Signals are POSITIVE evidence only
# - Absence is neutral (not negative)
# - Signals NEVER force domain selection
# - Used for explanation & reinforcement only
# -------------------------------------------------

DOMAIN_SIGNALS: Dict[str, Set[str]] = {

    "healthcare": {
        "patient_id",
        "diagnosis",
        "treatment",
        "admission_date",
        "discharge_date",
        "length_of_stay",
        "readmitted",
    },

    "finance": {
        "revenue",
        "expense",
        "profit",
        "loss",
        "balance",
        "cash_flow",
        "ebitda",
    },

    "retail": {
        "store_id",
        "pos_id",
        "basket_size",
        "promotion",
        "discount",
        "sales",
    },

    "supply_chain": {
        "inventory_level",
        "stock_on_hand",
        "reorder_point",
        "supplier",
        "lead_time",
        "sku",
    },

    "marketing": {
        "campaign_id",
        "impressions",
        "clicks",
        "ctr",
        "cpc",
        "roas",
    },

    "hr": {
        "employee_id",
        "department",
        "salary",
        "ctc",
        "attrition",
        "hire_date",
    },

    "customer": {
        "customer_id",
        "segment",
        "lifetime_value",
        "churn",
        "nps",
    },
}


# =====================================================
# SIGNAL EXTRACTION (COLUMN-BASED ONLY)
# =====================================================

def extract_domain_signals(
    normalized_columns: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Extract explicit domain signals from normalized column names.

    RETURNS:
    {
        domain: {
            "matched_signals": List[str],
            "signal_count": int,
            "signal_strength": float  # 0.0–1.0 (coverage-based, not probability)
        }
    }

    GUARANTEES:
    - Never raises
    - Deterministic
    - No domain is forced
    - Missing signals are neutral
    """

    results: Dict[str, Dict[str, Any]] = {}

    if not isinstance(normalized_columns, set) or not normalized_columns:
        return results

    for domain, signals in DOMAIN_SIGNALS.items():
        matches = signals.intersection(normalized_columns)

        if not matches:
            continue

        # Conservative normalization:
        # strength = proportion of known strong signals present
        denom = max(len(signals), 1)
        strength = min(len(matches) / denom, 1.0)

        results[domain] = {
            "matched_signals": sorted(matches),
            "signal_count": len(matches),
            "signal_strength": round(float(strength), 3),
        }

    return results


# =====================================================
# DATAFRAME-AWARE WRAPPER (FUTURE-SAFE)
# =====================================================

def extract_signals_from_df(
    df: pd.DataFrame,
    normalized_columns: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """
    DataFrame-aware signal extraction.

    CURRENT BEHAVIOR:
    - Column-name signals only

    FUTURE EXTENSIONS (EXPLICIT, NOT IMPLIED):
    - Non-null ratios
    - Value ranges
    - Cardinality checks

    GUARANTEES:
    - Never raises
    - Never infers domain
    """

    if not isinstance(df, pd.DataFrame):
        return {}

    return extract_domain_signals(normalized_columns)
