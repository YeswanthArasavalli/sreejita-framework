"""
Capability-Driven Domain Base
Sreejita Framework v3.6

PURPOSE:
- Enable capability-first domain logic
- Allow declarative KPI / visual / insight rules
- Avoid hard-coded domain assumptions
- Support graceful degradation
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import pandas as pd

from sreejita.core.column_resolver import resolve_column
from sreejita.domains.base import BaseDomain


class CapabilityDrivenDomain(BaseDomain):
    """
    Capability-driven domain engine.

    DESIGN PHILOSOPHY:
    - Domains declare WHAT they can do (capabilities)
    - Rules decide WHEN logic applies
    - Engine orchestrates execution safely
    """

    # -------------------------------------------------
    # DOMAIN DECLARATIONS (OVERRIDE PER DOMAIN)
    # -------------------------------------------------
    CAPABILITIES: Dict[str, str] = {}
    KPI_RULES: List[Dict[str, Any]] = []
    VISUAL_RULES: List[Dict[str, Any]] = []
    INSIGHT_RULES: List[Callable] = []
    RECOMMENDATION_RULES: List[Callable] = []

    # -------------------------------------------------
    # PREPROCESS — CAPABILITY RESOLUTION (SAFE)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve declared capabilities using semantic column resolution.

        GUARANTEES:
        - No mutation
        - Boolean capability map only
        - No domain inference
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("CapabilityDrivenDomain expects a DataFrame")

        self.capabilities: Dict[str, bool] = {}

        for name, semantic in (self.CAPABILITIES or {}).items():
            try:
                self.capabilities[name] = resolve_column(df, semantic) is not None
            except Exception:
                self.capabilities[name] = False

        return df.copy(deep=False)

    # -------------------------------------------------
    # KPI ENGINE (DECLARATIVE, GUARDED)
    # -------------------------------------------------
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute KPI rules based on resolved capabilities.

        RULE CONTRACT:
        {
            "name": str,
            "when": Callable[[Dict[str, bool]], bool],
            "compute": Callable[[pd.DataFrame], Any],
        }
        """

        kpis: Dict[str, Any] = {}

        for rule in self.KPI_RULES or []:
            try:
                if not rule.get("when", lambda _: False)(self.capabilities):
                    continue

                value = rule.get("compute", lambda _: None)(df)
                if value is not None:
                    kpis[rule.get("name")] = value

            except Exception:
                # Absolute safety — skip rule
                continue

        return kpis

    # -------------------------------------------------
    # VISUAL ENGINE (OPTIONAL, SAFE)
    # -------------------------------------------------
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Execute visual rules safely.

        RULE CONTRACT:
        {
            "when": Callable[[Dict[str, bool]], bool],
            "plot": Callable[[pd.DataFrame, Path], Optional[Dict]]
        }
        """

        visuals: List[Dict[str, Any]] = []

        if not output_dir:
            return visuals

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for rule in self.VISUAL_RULES or []:
            try:
                if not rule.get("when", lambda _: False)(self.capabilities):
                    continue

                visual = rule.get("plot", lambda *_: None)(df, output_dir)
                if isinstance(visual, dict):
                    visuals.append(visual)

            except Exception:
                continue

        return visuals

    # -------------------------------------------------
    # INSIGHT ENGINE (CAPABILITY-AWARE)
    # -------------------------------------------------
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Execute insight rules safely.

        RULE CONTRACT:
        Callable(df, kpis, capabilities) -> Optional[Dict]
        """

        insights: List[Dict[str, Any]] = []

        for rule in self.INSIGHT_RULES or []:
            try:
                insight = rule(df, kpis, self.capabilities)
                if isinstance(insight, dict):
                    insights.append(insight)
            except Exception:
                continue

        return insights

    # -------------------------------------------------
    # RECOMMENDATION ENGINE (OPTIONAL)
    # -------------------------------------------------
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Execute recommendation rules safely.

        RULE CONTRACT:
        Callable(df, kpis, insights, capabilities) -> Optional[Dict]
        """

        recs: List[Dict[str, Any]] = []

        for rule in self.RECOMMENDATION_RULES or []:
            try:
                rec = rule(df, kpis, insights, self.capabilities)
                if isinstance(rec, dict):
                    recs.append(rec)
            except Exception:
                continue

        return recs
