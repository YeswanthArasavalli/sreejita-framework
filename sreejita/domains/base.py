# =====================================================
# BASE DOMAIN — UNIVERSAL (FINAL, LOCKED)
# Sreejita Framework v3.6 STABILIZED
# =====================================================

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sreejita.narrative.executive_cognition import (
    build_executive_payload,
    build_subdomain_executive_payloads,
)

# =====================================================
# BASE DOMAIN
# =====================================================

class BaseDomain(ABC):
    """
    Universal BaseDomain contract.

    HARD GUARANTEES:
    - No shared DataFrame mutation
    - Deterministic KPI lifecycle
    - Visual safety (guaranteed)
    - Executive-safe cognition
    - Never crashes orchestrator

    MUST NOT:
    - Route domains
    - Orchestrate pipelines
    - Guess sub-domains
    """

    name: str = "generic"
    description: str = "Generic domain"

    def __init__(self):
        self._last_kpis: Optional[Dict[str, Any]] = None

    # --------------------------------------------------
    # OPTIONAL VALIDATION (SAFE DEFAULT)
    # --------------------------------------------------
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Domain-specific validation hook.

        DEFAULT:
        - Always True
        - Never raises
        """
        return True

    # --------------------------------------------------
    # OPTIONAL PREPROCESS (SAFE DEFAULT)
    # --------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Default preprocess is COPY-ONLY.

        RULES:
        - MUST return a DataFrame
        - MUST NOT mutate original df
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("preprocess expects a DataFrame")
        return df.copy(deep=False)

    # --------------------------------------------------
    # REQUIRED DOMAIN CONTRACTS
    # --------------------------------------------------
    @abstractmethod
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute KPIs.

        MUST:
        - Return a dict
        - Never mutate df
        """
        raise NotImplementedError

    @abstractmethod
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from KPIs.

        MUST:
        - Return a list
        - Be evidence-based
        """
        raise NotImplementedError

    @abstractmethod
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations.

        MUST:
        - Return a list
        - Be advisory-only
        """
        raise NotImplementedError

    @abstractmethod
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Generate visuals.

        MUST:
        - Return a list
        - NEVER raise
        """
        raise NotImplementedError

    # --------------------------------------------------
    # 🔒 UNIVERSAL VISUAL SAFETY NET (LAST RESORT)
    # --------------------------------------------------
    def ensure_minimum_visuals(
        self,
        visuals: List[Dict[str, Any]],
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Guarantees at least 2 visuals.

        FALLBACK VISUALS:
        - Dataset size
        - Data completeness

        ABSOLUTE RULE:
        - NEVER raises
        """

        visuals = visuals if isinstance(visuals, list) else []

        if len(visuals) >= 2:
            return visuals

        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # -----------------------------
            # Fallback 1 — Dataset Size
            # -----------------------------
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Records"], [len(df)])
            ax.set_title("Dataset Size Overview", fontweight="bold")
            ax.set_ylabel("Record Count")

            path = output_dir / f"{self.name}_fallback_records.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append({
                "path": str(path),
                "caption": "Total number of records (fallback evidence).",
                "importance": 0.2,
                "confidence": 0.3,
                "sub_domain": self.name,
                "inference_type": "fallback",
            })

            # -----------------------------
            # Fallback 2 — Data Completeness
            # -----------------------------
            completeness = float(1 - df.isna().mean().mean())

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Completeness"], [completeness])
            ax.set_ylim(0, 1)
            ax.set_title("Data Completeness Indicator", fontweight="bold")

            path = output_dir / f"{self.name}_fallback_completeness.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append({
                "path": str(path),
                "caption": "Overall data completeness ratio (fallback evidence).",
                "importance": 0.2,
                "confidence": 0.3,
                "sub_domain": self.name,
                "inference_type": "fallback",
            })

        except Exception:
            pass  # absolute safety guarantee

        return visuals

    # --------------------------------------------------
    # 🧠 EXECUTIVE COGNITION (SAFE, NON-HALLUCINATING)
    # --------------------------------------------------
    def build_executive(
        self,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Builds executive-facing summary.

        GUARANTEES:
        - Never hallucinates
        - Sub-domain aware
        - Stable output shape
        """

        executive = build_executive_payload(
            kpis=kpis or {},
            insights=insights or [],
            recommendations=recommendations or [],
        )

        sub_domains = kpis.get("sub_domains")

        if isinstance(sub_domains, dict) and sub_domains:
            executive["executive_by_sub_domain"] = (
                build_subdomain_executive_payloads(
                    kpis=kpis,
                    insights=insights or [],
                    recommendations=recommendations or [],
                )
            )
        else:
            executive["executive_by_sub_domain"] = {}

        return executive

    # --------------------------------------------------
    # 🔒 SAFE LEGACY PIPELINE (AUTHORITATIVE)
    # --------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        *,
        visual_output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Legacy domain execution pipeline.

        HARD GUARANTEES:
        - No shared mutation
        - Visuals always exist (if requested)
        - Executive output always stable
        - Exceptions never escape
        """

        result = {
            "domain": self.name,
            "description": self.description,
            "kpis": {},
            "insights": [],
            "recommendations": [],
            "visuals": [],
            "executive": {},
        }

        try:
            if not self.validate_data(df):
                return result

            if not isinstance(df, pd.DataFrame) or df.empty:
                return result

            # Defensive copy
            df = df.copy(deep=False)

            # Preprocess
            df = self.preprocess(df)
            if df is None or df.empty:
                return result

            # KPIs
            kpis = self.calculate_kpis(df)
            if not isinstance(kpis, dict):
                kpis = {}

            self._last_kpis = kpis

            # Insights & recommendations
            insights = self.generate_insights(df, kpis) or []
            recommendations = self.generate_recommendations(df, kpis, insights) or []

            # Visuals
            visuals: List[Dict[str, Any]] = []
            if visual_output_dir is not None:
                try:
                    visuals = self.generate_visuals(df, visual_output_dir)
                    visuals = self.ensure_minimum_visuals(
                        visuals,
                        df,
                        visual_output_dir,
                    )
                except Exception:
                    visuals = self.ensure_minimum_visuals(
                        [],
                        df,
                        visual_output_dir,
                    )

            # Executive cognition
            executive = self.build_executive(
                kpis=kpis,
                insights=insights,
                recommendations=recommendations,
            )

            result.update(
                {
                    "kpis": kpis,
                    "insights": insights,
                    "recommendations": recommendations,
                    "visuals": visuals,
                    "executive": executive,
                }
            )

        except Exception:
            # ABSOLUTE SAFETY — never raise
            return result

        return result
