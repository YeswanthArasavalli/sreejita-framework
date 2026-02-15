# =====================================================
# GENERIC FALLBACK DOMAIN — UNIVERSAL (FINAL, LOCKED)
# Sreejita Framework v3.6
# =====================================================

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, Optional, List

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# GENERIC DOMAIN (ABSOLUTE FALLBACK)
# =====================================================

class GenericDomain(BaseDomain):
    """
    Universal fallback domain.

    HARD GUARANTEES:
    - Never competes with real domains
    - Never infers business semantics
    - Never blocks reporting
    - Always executive-safe
    """

    name = "generic"
    description = "Generic dataset analysis (fallback mode)"

    # -------------------------------------------------
    # PREPROCESS (PASS-THROUGH, SAFE)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("GenericDomain expects a DataFrame")
        return df.copy(deep=False)

    # -------------------------------------------------
    # KPI ENGINE (STRUCTURAL ONLY)
    # -------------------------------------------------
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        volume = int(len(df))

        numeric_cols = df.select_dtypes(include="number").shape[1]
        date_cols = df.select_dtypes(
            include=["datetime", "datetimetz"]
        ).shape[1]

        completeness = float(1 - df.isna().mean().mean())

        kpis: Dict[str, Any] = {
            "primary_sub_domain": "generic",
            "sub_domains": {"generic": 1.0},
            "record_count": volume,
            "column_count": int(len(df.columns)),
            "numeric_column_count": int(numeric_cols),
            "date_column_count": int(date_cols),
            "data_completeness": round(completeness, 3),
        }

        if volume < 20:
            kpis["data_warning"] = (
                "Very small dataset — insights are indicative only"
            )

        # Conservative, flat confidence (explicit)
        kpis["_confidence"] = {
            k: 0.35 for k in kpis if not k.startswith("_")
        }

        self._last_kpis = kpis
        return kpis

    # -------------------------------------------------
    # VISUAL ENGINE (LOW-PRIORITY, GUARANTEED)
    # -------------------------------------------------
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:

        visuals: List[Dict[str, Any]] = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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
                "caption": "Total number of records (fallback evidence).",
                "importance": 0.25,
                "confidence": 0.4,
                "sub_domain": "generic",
                "inference_type": "fallback",
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
            ax.set_title(
                "Column Completeness (Top Fields)",
                fontweight="bold",
            )
            ax.set_ylabel("Completeness Ratio")

            path = output_dir / "generic_column_completeness.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append({
                "path": str(path),
                "caption": "Top column completeness ratios (fallback evidence).",
                "importance": 0.25,
                "confidence": 0.4,
                "sub_domain": "generic",
                "inference_type": "fallback",
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
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:

        insights: List[Dict[str, Any]] = [
            {
                "sub_domain": "generic",
                "level": "INFO",
                "title": "Dataset Not Mapped to a Supported Domain",
                "so_what": (
                    "The dataset does not strongly match any supported business domain. "
                    "Analysis is limited to structural characteristics."
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
                    "High levels of missing data may reduce analytical reliability "
                    "and confidence in interpretation."
                ),
                "confidence": 0.45,
            })

        while len(insights) < 3:
            insights.append({
                "sub_domain": "generic",
                "level": "INFO",
                "title": "Fallback Analysis Mode",
                "so_what": (
                    "Additional domain context is required for deeper intelligence."
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
        *args,
        **kwargs,
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

    RULE:
    - Always matches
    - Always lowest confidence
    - Never blocks stronger domains
    """

    domain_name = "generic"

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        return DomainDetectionResult(
            domain="generic",
            confidence=0.15,  # intentionally low
            signals={"fallback": True},
        )


# =====================================================
# REGISTRATION (AUTHORITATIVE)
# =====================================================

def register(registry):
    registry.register(
        "generic",
        GenericDomain,
        GenericDomainDetector,
        overwrite=True,
    )
