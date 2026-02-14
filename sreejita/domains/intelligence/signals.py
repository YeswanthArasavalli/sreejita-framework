# =====================================================
# DOMAIN SIGNALS — AUTHORITATIVE EVIDENCE LAYER
# Phase 4 — Detection Robustness
# Sreejita Framework v3.6.x
# =====================================================

from typing import Dict, Set, Any
import pandas as pd


# -------------------------------------------------
# DOMAIN SIGNAL DEFINITIONS
# -------------------------------------------------
# These are STRONG, explicit indicators.
# Presence implies real domain relevance.
# Absence implies no signal — not negative evidence.
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


# -------------------------------------------------
# SIGNAL EVALUATION
# -------------------------------------------------

def extract_domain_signals(
    normalized_columns: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Extract explicit domain signals from normalized column names.

    RETURNS:
    {
        domain: {
            "matched_signals": [...],
            "signal_strength": float (0–1),
            "signal_count": int
        }
    }

    GUARANTEES:
    - Never raises
    - No domain is forced
    - Missing signals are neutral (not negative)
    """

    results: Dict[str, Dict[str, Any]] = {}

    for domain, signals in DOMAIN_SIGNALS.items():
        matches = signals.intersection(normalized_columns)

        if not matches:
            continue

        # Conservative normalization
        strength = min(len(matches) / max(len(signals), 1), 1.0)

        results[domain] = {
            "matched_signals": sorted(matches),
            "signal_count": len(matches),
            "signal_strength": round(strength, 3),
        }

    return results


# -------------------------------------------------
# DATAFRAME-AWARE HELPER (OPTIONAL)
# -------------------------------------------------

def extract_signals_from_df(
    df: pd.DataFrame,
    normalized_columns: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience wrapper for detectors.

    Allows future extension to value-based signals
    (e.g., non-null ratios, value ranges).
    """

    signals = extract_domain_signals(normalized_columns)

    # Future-safe hook (currently no value inference)
    return signals
