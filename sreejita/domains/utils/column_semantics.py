"""Domain-agnostic column role inference utilities.

These helpers provide lightweight, conservative heuristics that inspect
column names and dtypes only. They avoid business-domain assumptions and
return low-confidence signals when columns are generic or ambiguous.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Set

import pandas as pd


ROLES: Set[str] = {
    "identifier",
    "temporal",
    "monetary",
    "quantity",
    "categorical",
    "clinical",
    "financial",
    "operational",
    "demographic",
    "free_text",
}

_GENERIC_NAME_LOW_CONFIDENCE = 0.25


def _tokenize(name: str) -> Set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", name.lower()) if token}


def _has_any(tokens: Iterable[str], candidates: Iterable[str]) -> bool:
    return any(token in candidates for token in tokens)


def infer_column_roles(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Infer column roles using conservative, domain-agnostic heuristics.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Mapping of column name to a dict containing:
          - "roles": set of inferred roles.
          - "confidence": float confidence score.
    """

    role_map: Dict[str, Dict[str, object]] = {}

    for column in df.columns:
        name = str(column)
        tokens = _tokenize(name)
        roles: Set[str] = set()
        confidence = 0.0

        is_datetime = pd.api.types.is_datetime64_any_dtype(df[column].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[column].dtype)
        is_textual = pd.api.types.is_string_dtype(df[column].dtype)
        is_categorical = (
            pd.api.types.is_categorical_dtype(df[column].dtype)
            or pd.api.types.is_bool_dtype(df[column].dtype)
            or is_textual
        )

        if name.lower().endswith("_id") or _has_any(
            tokens, {"id", "identifier", "uuid", "guid", "key"}
        ):
            roles.add("identifier")
            confidence = max(confidence, 0.7)

        if is_datetime or _has_any(tokens, {"date", "time", "timestamp", "datetime"}):
            roles.add("temporal")
            confidence = max(confidence, 0.6)

        if _has_any(tokens, {"amount", "price", "cost", "total", "currency"}):
            roles.add("monetary")
            confidence = max(confidence, 0.6)

        if is_numeric and _has_any(tokens, {"qty", "quantity", "count", "number"}):
            roles.add("quantity")
            confidence = max(confidence, 0.5)

        if is_categorical and _has_any(tokens, {"status", "type", "category", "class"}):
            roles.add("categorical")
            confidence = max(confidence, 0.4)

        if _has_any(tokens, {"description", "desc", "notes", "comment", "text", "message"}):
            roles.add("free_text")
            confidence = max(confidence, 0.4)

        if _has_any(tokens, {"demographic", "age", "gender", "sex", "dob", "birth"}):
            roles.add("demographic")
            confidence = max(confidence, 0.55)

        if _has_any(tokens, {"clinical"}):
            roles.add("clinical")
            confidence = max(confidence, 0.4)

        if _has_any(tokens, {"financial", "finance"}):
            roles.add("financial")
            confidence = max(confidence, 0.4)

        if _has_any(tokens, {"operational", "ops"}):
            roles.add("operational")
            confidence = max(confidence, 0.4)

        if name.lower() in {"date", "amount", "status"}:
            confidence = min(confidence, _GENERIC_NAME_LOW_CONFIDENCE)

        if not roles:
            confidence = 0.0

        role_map[name] = {"roles": roles, "confidence": float(confidence)}

    return role_map


def summarize_roles(role_map: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    """Summarize inferred role counts across columns.

    Args:
        role_map: Output of infer_column_roles.

    Returns:
        Mapping of role to count of columns containing that role.
    """

    summary = {role: 0 for role in ROLES}
    for column_info in role_map.values():
        for role in column_info.get("roles", set()):
            if role in summary:
                summary[role] += 1
    return summary


def compute_generic_penalty(role_summary: Dict[str, int]) -> float:
    """Compute a penalty when only generic roles are present.

    Generic roles are identifier, temporal, monetary, quantity, categorical,
    and free_text. A penalty is applied when no non-generic roles are found.

    Args:
        role_summary: Output of summarize_roles.

    Returns:
        Float penalty between 0.0 and 0.5.
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
        count for role, count in role_summary.items() if role not in generic_roles
    )
    if non_generic == 0:
        return 0.5

    generic = total - non_generic
    penalty = 0.5 * (generic / total)
    return max(0.0, min(0.5, penalty))
