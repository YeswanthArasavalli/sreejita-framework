"""
Domain-agnostic column role inference utilities.

Phase 4 rules:
- Column roles are CONTEXT ONLY (never domain-forcing)
- Heuristics must remain conservative
- Confidence is local to each column
- Generic schemas are explicitly penalized
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Set

import pandas as pd


# -------------------------------------------------
# ROLE VOCABULARY (DOMAIN-AGNOSTIC)
# -------------------------------------------------

ROLES: Set[str] = {
    "identifier",
    "temporal",
    "monetary",
    "quantity",
    "categorical",
    "clinical",       # informational only (NOT healthcare forcing)
    "financial",      # informational only
    "operational",
    "demographic",
    "free_text",
}

# Generic column names carry very low signal
_GENERIC_NAME_LOW_CONFIDENCE = 0.25


# -------------------------------------------------
# TOKENIZATION HELPERS
# -------------------------------------------------

def _tokenize(name: str) -> Set[str]:
    """
    Tokenize column name into lowercase alphanumeric tokens.
    """
    return {
        token
        for token in re.split(r"[^a-z0-9]+", name.lower())
        if token
    }


def _has_any(tokens: Iterable[str], candidates: Iterable[str]) -> bool:
    return any(token in candidates for token in tokens)


# -------------------------------------------------
# ROLE INFERENCE (COLUMN-LOCAL)
# -------------------------------------------------

def infer_column_roles(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """
    Infer column roles using conservative, domain-agnostic heuristics.

    RETURNS:
    {
        column_name: {
            "roles": set[str],
            "confidence": float (0–1)
        }
    }

    GUARANTEES:
    - Never raises
    - Never infers domain
    - Confidence is per-column only
    """

    role_map: Dict[str, Dict[str, object]] = {}

    for column in df.columns:
        name = str(column)
        tokens = _tokenize(name)

        roles: Set[str] = set()
        confidence = 0.0

        dtype = df[column].dtype

        is_datetime = pd.api.types.is_datetime64_any_dtype(dtype)
        is_numeric = pd.api.types.is_numeric_dtype(dtype)
        is_textual = pd.api.types.is_string_dtype(dtype)
        is_categorical = (
            pd.api.types.is_categorical_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or is_textual
        )

        # -------------------------------
        # IDENTIFIER
        # -------------------------------
        if name.lower().endswith("_id") or _has_any(
            tokens, {"id", "identifier", "uuid", "guid", "key"}
        ):
            roles.add("identifier")
            confidence = max(confidence, 0.7)

        # -------------------------------
        # TEMPORAL
        # -------------------------------
        if is_datetime or _has_any(
            tokens, {"date", "time", "timestamp", "datetime"}
        ):
            roles.add("temporal")
            confidence = max(confidence, 0.6)

        # -------------------------------
        # MONETARY
        # -------------------------------
        if _has_any(tokens, {"amount", "price", "cost", "total", "currency"}):
            roles.add("monetary")
            confidence = max(confidence, 0.6)

        # -------------------------------
        # QUANTITY
        # -------------------------------
        if is_numeric and _has_any(
            tokens, {"qty", "quantity", "count", "number"}
        ):
            roles.add("quantity")
            confidence = max(confidence, 0.5)

        # -------------------------------
        # CATEGORICAL
        # -------------------------------
        if is_categorical and _has_any(
            tokens, {"status", "type", "category", "class"}
        ):
            roles.add("categorical")
            confidence = max(confidence, 0.4)

        # -------------------------------
        # FREE TEXT
        # -------------------------------
        if _has_any(
            tokens, {"description", "desc", "notes", "comment", "text", "message"}
        ):
            roles.add("free_text")
            confidence = max(confidence, 0.4)

        # -------------------------------
        # DEMOGRAPHIC (NON-DOMAIN)
        # -------------------------------
        if _has_any(
            tokens, {"age", "gender", "sex", "dob", "birth"}
        ):
            roles.add("demographic")
            confidence = max(confidence, 0.55)

        # -------------------------------
        # CLINICAL (WEAK, NON-FORCING)
        # -------------------------------
        if _has_any(tokens, {"clinical"}):
            roles.add("clinical")
            confidence = max(confidence, 0.35)

        # -------------------------------
        # FINANCIAL (WEAK, NON-FORCING)
        # -------------------------------
        if _has_any(tokens, {"financial", "finance"}):
            roles.add("financial")
            confidence = max(confidence, 0.35)

        # -------------------------------
        # OPERATIONAL
        # -------------------------------
        if _has_any(tokens, {"operational", "ops"}):
            roles.add("operational")
            confidence = max(confidence, 0.35)

        # -------------------------------
        # GENERIC NAME DOWNGRADE
        # -------------------------------
        if name.lower() in {"date", "amount", "status"}:
            confidence = min(confidence, _GENERIC_NAME_LOW_CONFIDENCE)

        # -------------------------------
        # NO ROLES → NO CONFIDENCE
        # -------------------------------
        if not roles:
            confidence = 0.0

        role_map[name] = {
            "roles": roles,
            "confidence": round(float(confidence), 3),
        }

    return role_map


# -------------------------------------------------
# ROLE SUMMARY (DATASET-LEVEL CONTEXT)
# -------------------------------------------------

def summarize_roles(
    role_map: Dict[str, Dict[str, object]]
) -> Dict[str, int]:
    """
    Summarize inferred role counts across columns.

    RETURNS:
    { role: count }
    """

    summary = {role: 0 for role in ROLES}

    for column_info in role_map.values():
        for role in column_info.get("roles", set()):
            if role in summary:
                summary[role] += 1

    return summary


# -------------------------------------------------
# GENERIC SCHEMA PENALTY
# -------------------------------------------------

def compute_generic_penalty(role_summary: Dict[str, int]) -> float:
    """
    Penalize datasets that contain only generic roles.

    Generic roles:
    - identifier
    - temporal
    - monetary
    - quantity
    - categorical
    - free_text

    RETURNS:
    penalty ∈ [0.0, 0.5]
    """

    generic_roles = {
        "identifier",
        "temporal",
        "monetary",
        "quantity",
        "categorical",
        "free_text",
    }

    total = sum(role_summary.values())
    if total == 0:
        return 0.5

    non_generic = sum(
        count
        for role, count in role_summary.items()
        if role not in generic_roles
    )

    if non_generic == 0:
        return 0.5

    generic = total - non_generic
    penalty = 0.5 * (generic / total)

    return round(max(0.0, min(0.5, penalty)), 3)
