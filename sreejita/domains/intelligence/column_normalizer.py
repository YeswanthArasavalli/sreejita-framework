# =====================================================
# COLUMN NORMALIZER — STABLE & DOMAIN-SAFE
# Sreejita Framework v3.6 (FINAL)
# =====================================================

"""
Purpose:
- Provide deterministic, domain-agnostic column normalization
- Serve as a shared utility for intelligence + detectors
- NEVER infer meaning or business semantics

This module performs ONLY syntactic normalization.
"""

import re
from typing import Iterable, Tuple, Dict, Set


# -------------------------------------------------
# SINGLE COLUMN NORMALIZATION
# -------------------------------------------------

def normalize_column(col: str) -> str:
    """
    Normalize a column name into stable snake_case.

    GUARANTEES:
    - Preserves word boundaries
    - Deterministic output
    - Never raises
    - Never injects semantics
    - Never returns None

    EXAMPLES:
    - "Order Date"        -> "order_date"
    - "Total-Revenue($)" -> "total_revenue"
    - "Patient/ID"       -> "patient_id"
    """

    try:
        if col is None:
            return ""

        # Convert to string, trim, lowercase
        col = str(col).strip().lower()

        if not col:
            return ""

        # Replace common separators with space
        # (preserves word boundaries)
        col = re.sub(r"[\/\-\.\(\)\[\]\{\}%]", " ", col)

        # Remove remaining non-alphanumeric characters
        # (keep letters, numbers, underscores, spaces)
        col = re.sub(r"[^a-z0-9_\s]", "", col)

        # Normalize whitespace to underscores
        col = re.sub(r"\s+", "_", col)

        # Collapse multiple underscores
        col = re.sub(r"_+", "_", col)

        # Strip leading / trailing underscores
        col = col.strip("_")

        # Final safety
        return col or ""

    except Exception:
        # Absolute safety — never raise
        return ""


# -------------------------------------------------
# BULK NORMALIZATION
# -------------------------------------------------

def normalize_columns(
    columns: Iterable[str],
) -> Tuple[Set[str], Dict[str, str]]:
    """
    Normalize a collection of column names.

    Returns:
    - normalized_columns: Set[str]
        Unique normalized column names
    - mapping: Dict[original -> normalized]

    GUARANTEES:
    - One-way mapping only (no reverse guessing)
    - Original names preserved as keys
    - Invalid / empty columns safely ignored
    - Never raises
    """

    normalized_set: Set[str] = set()
    mapping: Dict[str, str] = {}

    try:
        for col in columns:
            normalized = normalize_column(col)
            if normalized:
                mapping[col] = normalized
                normalized_set.add(normalized)

    except Exception:
        # Absolute safety — partial results are acceptable
        pass

    return normalized_set, mapping


# -------------------------------------------------
# OPTIONAL UTILITY (NON-AUTHORITATIVE)
# -------------------------------------------------

def normalize_dataframe_columns(df):
    """
    Convenience helper (OPTIONAL).

    Returns a COPY of df with normalized column names.

    ⚠️ IMPORTANT:
    - This function MUST NOT be used inside domains directly
    - Intended for intelligence / exploration layers only
    - Domains should rely on resolve_column(), not renaming
    """

    try:
        cols = list(df.columns)
        _, mapping = normalize_columns(cols)
        return df.rename(columns=mapping, copy=True)
    except Exception:
        return df
