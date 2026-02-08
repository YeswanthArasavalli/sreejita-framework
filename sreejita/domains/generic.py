# =====================================================
# GENERIC FALLBACK DOMAIN — UNIVERSAL (FINAL, HARDENED)
# Sreejita Framework v3.6
# =====================================================

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# GENERIC DOMAIN (ABSOLUTE FALLBACK)
# =====================================================

class GenericDomain(BaseDomain):
    """
    Universal fallback domain.

    DESIGN PRINCIPLES:
    - Never competes with real domains
    - Never infers business meaning
    - Never blocks reporting
    - Always executive-safe
    """

    name = "generic"
    description = "Generic dataset analysis (fallback mode)"

    # -------------------------------------------------
    # PREPROCESS (PASS-THROUGH)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

     # -------------------------------------------------
    # SAFE RUN WRAPPER (STABLE OUTPUT CONTRACT)
    # -------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        *,
        visual_output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute the generic pipeline with stable, safe outputs.

        Guarantees:
        - Returns a fixed output schema even on weak data
        - No exceptions escape the method
        """

        result: Dict[str, Any] = {
            "domain": self.name,
            "kpis": {},
            "visuals": [],
            "insights": [],
            "recommendations": [],
        }

        try:
            if not self.validate_data(df):
                return result

            df = df.copy(deep=False)
            df = self.preprocess(df)
            if df is None or df.empty:
                return result

            kpis = self.calculate_kpis(df) or {}
            if not isinstance(kpis, dict):
                kpis = {}

            if kpis and "_confidence" not in kpis:
                kpis["_confidence"] = {}

            self._last_kpis = kpis

            insights = self.generate_insights(df, kpis) or []
            recommendations = self.generate_recommendations(
                df, kpis, insights
            ) or []

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

            result.update(
                {
                    "kpis": kpis,
                    "visuals": visuals,
                    "insights": insights,
                    "recommendations": recommendations,
                }
            )
        except Exception:
            return result

        return result
    # -------------------------------------------------
    # KPI ENGINE (MINIMAL & FLAT)
    # -------------------------------------------------
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        volume = int(len(df))

        kpis: Dict[str, Any] = {
            "primary_sub_domain": "generic",
            "sub_domains": {"generic": 1.0},
            "record_count": volume,
            "column_count": int(len(df.columns)),
            "numeric_column_count": int(
                df.select_dtypes(include="number").shape[1]
            ),
            "date_column_count": int(
                df.select_dtypes(include=["datetime", "datetimetz"]).shape[1]
            ),
            "data_completeness": round(
                float(1 - df.isna().mean().mean()), 3
            ),
        }

        if volume < 20:
            kpis["data_warning"] = (
                "Very small dataset — results are indicative only"
            )

        # 🔒 EXECUTIVE-SAFE CONFIDENCE (FLAT & CONSERVATIVE)
        kpis["_confidence"] = {
            k: 0.35 for k in kpis.keys()
        }

        self._last_kpis = kpis
        return kpis

    # -------------------------------------------------
    # VISUALS (GUARANTEED, LOW PRIORITY)
    # -------------------------------------------------
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:

        output_dir.mkdir(parents=True, exist_ok=True)
        visuals: List[Dict[str, Any]] = []

        # -----------------------------
        # Visual 1 — Dataset Size
        # -----------------------------
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Records"], [len(df)])
            ax.set_title("Dataset Size", fontweight="bold")
            ax.set_ylabel("Record Count")

            path = output_dir / "generic_dataset_size.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append({
                "path": str(path),
                "caption": "Dataset record count (fallback evidence).",
                "importance": 0.25,   # 🔒 always lowest
                "confidence": 0.4,
                "sub_domain": "generic",
            })
        except Exception:
            pass

        # -----------------------------
        # Visual 2 — Column Completeness
        # -----------------------------
        try:
            completeness = (
                df.notna().mean()
                .sort_values(ascending=False)
                .head(10)
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            completeness.plot(kind="bar", ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title("Column Completeness (Top Fields)", fontweight="bold")
            ax.set_ylabel("Completeness Ratio")

            path = output_dir / "generic_column_completeness.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append({
                "path": str(path),
                "caption": "Data completeness overview (fallback evidence).",
                "importance": 0.25,
                "confidence": 0.4,
                "sub_domain": "generic",
            })
        except Exception:
            pass

        return visuals

    # -------------------------------------------------
    # INSIGHTS (DISCLAIMER-FIRST)
    # -------------------------------------------------
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        *_,
    ) -> List[Dict[str, Any]]:

        insights: List[Dict[str, Any]] = [
            {
                "sub_domain": "generic",
                "level": "INFO",
                "title": "Dataset Not Mapped to a Supported Domain",
                "so_what": (
                    "The dataset does not strongly match any supported "
                    "business domain. Analysis is limited to structural signals."
                ),
                "confidence": 0.4,
            }
        ]

        if kpis.get("data_completeness", 1.0) < 0.7:
            insights.append({
                "sub_domain": "generic",
                "level": "WARNING",
                "title": "Low Data Completeness",
                "so_what": (
                    "High levels of missing data may limit analytical reliability "
                    "and decision confidence."
                ),
                "confidence": 0.45,
            })

        while len(insights) < 3:
            insights.append({
                "sub_domain": "generic",
                "level": "INFO",
                "title": "Fallback Analysis Mode",
                "so_what": (
                    "Further domain context is required for deeper intelligence."
                ),
                "confidence": 0.35,
            })

        return insights

    # -------------------------------------------------
    # RECOMMENDATIONS (META-LEVEL ONLY)
    # -------------------------------------------------
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        *_,
    ) -> List[Dict[str, Any]]:

        return [
            {
                "sub_domain": "generic",
                "priority": "HIGH",
                "action": "Provide business context and dataset purpose",
                "owner": "Data Owner",
                "timeline": "Immediate",
                "goal": "Enable accurate domain classification",
                "confidence": 0.5,
            },
            {
                "sub_domain": "generic",
                "priority": "MEDIUM",
                "action": "Standardize column naming and add metadata",
                "owner": "Data Engineering",
                "timeline": "30–60 days",
                "goal": "Improve semantic resolvability",
                "confidence": 0.45,
            },
        ]


# =====================================================
# GENERIC DOMAIN DETECTOR (TRUE FALLBACK)
# =====================================================

class GenericDomainDetector(BaseDomainDetector):
    """
    Absolute fallback detector.
    NEVER competes with real domains.
    """

    domain_name = "generic"

    def detect(self, df: pd.DataFrame):
        return DomainDetectionResult(
            domain="generic",
            confidence=0.15,   # 🔒 must stay LOW
            signals={"fallback": True},
        )


# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register(
        "generic",
        GenericDomain,
        GenericDomainDetector,
        overwrite=True,
    )
