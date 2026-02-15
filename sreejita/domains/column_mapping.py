"""
Universal Column Mapping
Sreejita Framework v3.6

PURPOSE:
- Provide domain-agnostic semantic column discovery
- Eliminate retail / finance bias
- Support graceful degradation across all domains
- Never mutate input DataFrames
- Never assume domain ownership
"""

from typing import Optional, Set, Dict
import pandas as pd


class ColumnMapping:
    """
    Universal semantic column mapping helper.

    DESIGN PRINCIPLES:
    - Capability-first (not domain-first)
    - Case-insensitive matching
    - No mutation
    - No hard dependency on any domain
    """

    # -------------------------------------------------
    # NUMERIC / FINANCIAL-LIKE SIGNALS (GENERIC)
    # -------------------------------------------------
    VALUE_COLS: Set[str] = {
        "value",
        "amount",
        "total",
        "metric",
    }

    REVENUE_COLS: Set[str] = {
        "revenue",
        "sales",
        "income",
        "turnover",
        "gmv",
        "total_sales",
        "order_value",
    }

    COST_COLS: Set[str] = {
        "cost",
        "expense",
        "expenses",
        "spend",
        "spending",
        "total_cost",
    }

    PROFIT_COLS: Set[str] = {
        "profit",
        "margin",
        "net_profit",
        "net_income",
        "earnings",
    }

    # -------------------------------------------------
    # CATEGORICAL / STRUCTURAL SIGNALS
    # -------------------------------------------------
    CATEGORY_COLS: Set[str] = {
        "category",
        "department",
        "unit",
        "segment",
        "division",
        "group",
        "type",
        "class",
    }

    ENTITY_COLS: Set[str] = {
        "entity",
        "product",
        "item",
        "sku",
        "service",
        "resource",
    }

    # -------------------------------------------------
    # TEMPORAL SIGNALS (GENERIC)
    # -------------------------------------------------
    DATE_COLS: Set[str] = {
        "date",
        "timestamp",
        "time",
        "period",
        "event_date",
        "record_date",
        "created_at",
        "updated_at",
    }

    # -------------------------------------------------
    # IDENTIFIERS (NON-DOMAIN-SPECIFIC)
    # -------------------------------------------------
    ID_COLS: Set[str] = {
        "id",
        "record_id",
        "entity_id",
        "order_id",
        "transaction_id",
        "user_id",
        "employee_id",
        "patient_id",
        "case_id",
    }

    # -------------------------------------------------
    # CORE MATCHING LOGIC (SAFE)
    # -------------------------------------------------
    @staticmethod
    def find_column(
        df_columns: Set[str],
        semantic_candidates: Set[str],
    ) -> Optional[str]:
        """
        Find the first matching column in df_columns
        based on semantic_candidates.

        GUARANTEES:
        - Case-insensitive
        - Returns original column name
        - Never raises
        """
        try:
            lookup = {
                str(c).lower().strip(): c
                for c in df_columns
            }

            for token in semantic_candidates:
                token_l = token.lower().strip()
                if token_l in lookup:
                    return lookup[token_l]

            return None
        except Exception:
            return None

    # -------------------------------------------------
    # AUTO-DETECTION (CAPABILITY SNAPSHOT)
    # -------------------------------------------------
    @staticmethod
    def auto_detect(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Perform lightweight semantic column detection.

        RETURNS:
        - Mapping of semantic capability → column name
        - Missing signals return None
        - NEVER mutates df
        """

        if not isinstance(df, pd.DataFrame):
            return {}

        cols = set(df.columns)

        return {
            # Value signals
            "value": ColumnMapping.find_column(cols, ColumnMapping.VALUE_COLS),
            "revenue": ColumnMapping.find_column(cols, ColumnMapping.REVENUE_COLS),
            "cost": ColumnMapping.find_column(cols, ColumnMapping.COST_COLS),
            "profit": ColumnMapping.find_column(cols, ColumnMapping.PROFIT_COLS),

            # Structure
            "category": ColumnMapping.find_column(cols, ColumnMapping.CATEGORY_COLS),
            "entity": ColumnMapping.find_column(cols, ColumnMapping.ENTITY_COLS),

            # Time
            "date": ColumnMapping.find_column(cols, ColumnMapping.DATE_COLS),

            # Identity
            "id": ColumnMapping.find_column(cols, ColumnMapping.ID_COLS),
        }
