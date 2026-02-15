# =====================================================
# MANUFACTURING DOMAIN — PART 1
# Imports & Governed Helpers
# =====================================================

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Governance: non-interactive backend
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import (
    BaseDomainDetector,
    DomainDetectionResult,
)

# =====================================================
# GOVERNED NUMERIC HELPERS
# =====================================================

def safe_div(n: Any, d: Any) -> Optional[float]:
    """
    Governance-safe division.

    Guarantees:
    - No ZeroDivision
    - No NaN / Inf leakage
    - Returns None on weak or invalid inputs
    """
    try:
        if n is None or d is None:
            return None
        if pd.isna(n) or pd.isna(d):
            return None
        d = float(d)
        if d == 0.0:
            return None
        val = float(n) / d
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    except Exception:
        return None


def safe_sum(series: pd.Series) -> Optional[float]:
    """
    Safe numeric sum with graceful degradation.
    """
    if series is None or series.empty:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.sum())


def safe_mean(series: pd.Series) -> Optional[float]:
    """
    Safe numeric mean with graceful degradation.
    """
    if series is None or series.empty:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


# =====================================================
# TIME & DATE HELPERS (MANUFACTURING-SAFE)
# =====================================================

MANUFACTURING_TIME_KEYWORDS = {
    "date",
    "time",
    "timestamp",
    "production_date",
    "prod_date",
    "shift_start",
    "shift_time",
    "run_time",
    "start_time",
    "end_time",
}


def _is_datetime_series(series: pd.Series) -> bool:
    """
    Verifies whether a column can reliably be interpreted as datetime.
    """
    try:
        parsed = pd.to_datetime(
            series.dropna().iloc[:10],
            errors="coerce",
        )
        return parsed.notna().sum() >= 3
    except Exception:
        return False


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Boundary-safe manufacturing time column detector.

    Design guarantees:
    - Prefers production / shift lifecycle timestamps
    - Rejects numeric-only columns
    - Returns None if confidence is weak
    """
    if df is None or df.empty:
        return None

    candidates: List[str] = []

    for col in df.columns:
        col_l = str(col).lower()
        if any(k in col_l for k in MANUFACTURING_TIME_KEYWORDS):
            if _is_datetime_series(df[col]):
                candidates.append(col)

    # Prefer production lifecycle over generic timestamps
    priority = [
        "production_date",
        "prod_date",
        "shift_start",
        "start_time",
    ]

    for p in priority:
        for c in candidates:
            if p in c.lower():
                return c

    return candidates[0] if candidates else None


def coerce_datetime(
    df: pd.DataFrame,
    col: Optional[str],
) -> Optional[pd.Series]:
    """
    Safely coerces a column to datetime.
    Returns None if coercion confidence is weak.
    """
    if col is None or col not in df.columns:
        return None
    try:
        s = pd.to_datetime(df[col], errors="coerce")
        if s.notna().sum() < 3:
            return None
        return s
    except Exception:
        return None


# =====================================================
# MANUFACTURING DOMAIN (UNIVERSAL 10/10)
# =====================================================

class ManufacturingDomain(BaseDomain):
    """
    Universal Manufacturing Intelligence Domain

    Scope:
    - Production volume & throughput
    - Quality & yield signals
    - Downtime & efficiency
    - Cost & operational context
    """

    name = "manufacturing"
    description = "Universal Manufacturing Intelligence (OEE, Quality, Downtime, Production)"

    # -------------------------------------------------
    # PREPROCESS (CENTRALIZED, GOVERNED STATE)
    # -------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Governance guarantees:
        - No domain assumptions
        - No row drops
        - No unsafe fills
        - No raw data mutation beyond coercion
        - Graceful degradation
        """

        df = df.copy(deep=False)

        # -------------------------------------------------
        # TIME DETECTION (BOUNDARY-SAFE)
        # -------------------------------------------------
        self.time_col: Optional[str] = detect_time_column(df)
        self._time_series: Optional[pd.Series] = (
            coerce_datetime(df, self.time_col)
            if self.time_col
            else None
        )

        # -------------------------------------------------
        # COLUMN RESOLUTION (AUTHORITATIVE, SINGLE PASS)
        # -------------------------------------------------
        def _res(col: Optional[str]) -> Optional[str]:
            return col if col in df.columns else None

        self.cols: Dict[str, Optional[str]] = {
            # ---------------- PRODUCTION ----------------
            "order_id": _res(
                resolve_column(df, "order_id")
                or resolve_column(df, "production_order")
            ),
            "product": _res(
                resolve_column(df, "product")
                or resolve_column(df, "item")
                or resolve_column(df, "material")
            ),
            "machine": _res(
                resolve_column(df, "machine")
                or resolve_column(df, "equipment")
                or resolve_column(df, "line")
            ),
            "quantity": _res(
                resolve_column(df, "quantity")
                or resolve_column(df, "produced_qty")
                or resolve_column(df, "output")
            ),
            "target": _res(
                resolve_column(df, "target_qty")
                or resolve_column(df, "planned_qty")
            ),

            # ---------------- QUALITY ----------------
            "defects": _res(
                resolve_column(df, "defects")
                or resolve_column(df, "rejected_qty")
                or resolve_column(df, "scrap")
            ),
            "quality_score": _res(resolve_column(df, "quality_score")),

            # ---------------- TIME / EFFICIENCY ----------------
            "cycle_time": _res(
                resolve_column(df, "cycle_time")
                or resolve_column(df, "run_time")
            ),
            "downtime": _res(
                resolve_column(df, "downtime")
                or resolve_column(df, "stop_time")
                or resolve_column(df, "breakdown")
            ),
            "uptime": _res(
                resolve_column(df, "uptime")
                or resolve_column(df, "operating_time")
            ),
            "shift": _res(resolve_column(df, "shift")),

            # ---------------- COST ----------------
            "cost": _res(
                resolve_column(df, "cost")
                or resolve_column(df, "production_cost")
            ),
        }

        # -------------------------------------------------
        # NUMERIC NORMALIZATION (STRICT & NON-DESTRUCTIVE)
        # -------------------------------------------------
        numeric_cols = [
            "quantity",
            "target",
            "defects",
            "cycle_time",
            "downtime",
            "uptime",
            "cost",
        ]

        for key in numeric_cols:
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # -------------------------------------------------
        # DATE NORMALIZATION (NO SORTING SIDE-EFFECTS)
        # -------------------------------------------------
        if self.time_col and self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(
                df[self.time_col],
                errors="coerce",
            )

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
        Execute the manufacturing pipeline with stable, safe outputs.

        Guarantees:
        - Fixed output schema
        - No exceptions escape
        - Graceful degradation on weak data
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

            # ---------------- KPIs ----------------
            kpis = self.calculate_kpis(df) or {}
            if not isinstance(kpis, dict):
                kpis = {}

            if "_confidence" not in kpis:
                kpis["_confidence"] = {}

            self._last_kpis = kpis

            # ---------------- INSIGHTS ----------------
            insights = self.generate_insights(df, kpis) or []

            # ---------------- RECOMMENDATIONS ----------------
            recommendations = (
                self.generate_recommendations(df, kpis, insights) or []
            )

            # ---------------- VISUALS ----------------
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

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Manufacturing KPI Engine (Foundation Layer)
    
        Guarantees:
        - No targets or benchmarks
        - No hard thresholds
        - No fabricated KPIs
        - No dataframe mutation
        - Graceful degradation
        """
    
        kpis: Dict[str, Any] = {}
        c = self.cols
    
        # ==================================================
        # PRODUCTION VOLUME & OUTPUT
        # ==================================================
    
        if c.get("quantity"):
            qty = pd.to_numeric(df[c["quantity"]], errors="coerce").dropna()
    
            if not qty.empty:
                total_production = float(qty.sum())
                kpis["total_production"] = total_production
                kpis["avg_units_per_record"] = float(qty.mean())
                kpis["max_units_single_run"] = float(qty.max())
    
        # ==================================================
        # QUALITY / YIELD SIGNALS (DESCRIPTIVE ONLY)
        # ==================================================
    
        if c.get("quantity") and c.get("defects"):
            qty = pd.to_numeric(df[c["quantity"]], errors="coerce").dropna()
            defects = pd.to_numeric(df[c["defects"]], errors="coerce").dropna()
    
            if not qty.empty and not defects.empty:
                total_qty = float(qty.sum())
                total_defects = float(defects.sum())
    
                defect_rate = safe_div(total_defects, total_qty)
    
                kpis["total_defects"] = total_defects
                kpis["defect_rate"] = defect_rate
                kpis["yield_rate"] = (
                    1.0 - defect_rate if defect_rate is not None else None
                )
    
        # ==================================================
        # AVAILABILITY (TIME UTILIZATION PROXY)
        # ==================================================
    
        if c.get("uptime") and c.get("downtime"):
            uptime = pd.to_numeric(df[c["uptime"]], errors="coerce").dropna()
            downtime = pd.to_numeric(df[c["downtime"]], errors="coerce").dropna()
    
            if not uptime.empty or not downtime.empty:
                total_uptime = float(uptime.sum()) if not uptime.empty else 0.0
                total_downtime = float(downtime.sum()) if not downtime.empty else 0.0
                total_time = total_uptime + total_downtime
    
                kpis["total_uptime"] = total_uptime
                kpis["total_downtime"] = total_downtime
                kpis["availability"] = safe_div(total_uptime, total_time)
    
        # ==================================================
        # PERFORMANCE (OUTPUT VS PLAN – NON-JUDGMENTAL)
        # ==================================================
    
        if c.get("quantity") and c.get("target"):
            qty = pd.to_numeric(df[c["quantity"]], errors="coerce").dropna()
            target = pd.to_numeric(df[c["target"]], errors="coerce").dropna()
    
            if not qty.empty and not target.empty:
                kpis["performance"] = safe_div(
                    float(qty.sum()),
                    float(target.sum())
                )
    
        # ==================================================
        # OEE (ONLY IF ALL COMPONENTS EXIST)
        # ==================================================
    
        availability = kpis.get("availability")
        performance = kpis.get("performance")
        yield_rate = kpis.get("yield_rate")
    
        if (
            isinstance(availability, (int, float))
            and isinstance(performance, (int, float))
            and isinstance(yield_rate, (int, float))
        ):
            kpis["oee"] = availability * performance * yield_rate
    
        # ==================================================
        # EFFICIENCY / FLOW SIGNALS
        # ==================================================
    
        if c.get("cycle_time") and c.get("quantity"):
            cycle = pd.to_numeric(df[c["cycle_time"]], errors="coerce").dropna()
            qty = pd.to_numeric(df[c["quantity"]], errors="coerce").dropna()
    
            if not cycle.empty and not qty.empty:
                kpis["avg_cycle_time_per_unit"] = safe_div(
                    float(cycle.sum()),
                    float(qty.sum())
                )
    
        # ==================================================
        # COST SIGNALS (DESCRIPTIVE ONLY)
        # ==================================================
    
        if c.get("cost"):
            cost = pd.to_numeric(df[c["cost"]], errors="coerce").dropna()
    
            if not cost.empty:
                total_cost = float(cost.sum())
                kpis["total_production_cost"] = total_cost
    
                if kpis.get("total_production"):
                    kpis["cost_per_unit"] = safe_div(
                        total_cost,
                        kpis["total_production"]
                    )
    
        # ==================================================
        # KPI CONFIDENCE (STRUCTURAL, NOT PERFORMANCE)
        # ==================================================
    
        kpis["_confidence"] = {}
    
        record_count = len(df)
    
        for k, v in kpis.items():
            if k.startswith("_"):
                continue
            if not isinstance(v, (int, float)):
                continue
    
            base_conf = 0.65
    
            if record_count < 30:
                base_conf -= 0.15
    
            if "proxy" in k or "approx" in k:
                base_conf -= 0.1
    
            kpis["_confidence"][k] = round(
                max(0.35, min(0.9, base_conf)),
                2
            )
    
        return kpis


    # ---------------- VISUALS (8 CANDIDATES) ----------------

    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Manufacturing Visual Intelligence Engine
    
        Guarantees:
        - No thresholds or targets
        - No outcome-based importance
        - No unsafe ratios
        - No dataframe mutation
        - Executive-safe narratives
        - Many → few pruning
        """
    
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
    
        c = self.cols
        kpis = getattr(self, "_last_kpis", None) or self.calculate_kpis(df)
    
        # -------------------------------------------------
        # SAVE HELPER
        # -------------------------------------------------
        def save(fig, name, caption, importance, category):
            path = output_dir / name
            fig.savefig(path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            visuals.append({
                "path": str(path),
                "caption": caption,
                "importance": float(importance),
                "category": category
            })
    
        # -------------------------------------------------
        # FORMATTERS
        # -------------------------------------------------
        def human_fmt(x, _):
            if x >= 1e6:
                return f"{x / 1e6:.1f}M"
            if x >= 1e3:
                return f"{x / 1e3:.0f}K"
            return str(int(x))
    
        # =================================================
        # 1. PRODUCTION VOLUME TREND
        # =================================================
        if self.time_col and c.get("quantity"):
            s = (
                df[[self.time_col, c["quantity"]]]
                .dropna()
                .set_index(self.time_col)[c["quantity"]]
                .resample("D")
                .sum()
            )
    
            if s.nunique() > 2:
                fig, ax = plt.subplots(figsize=(7, 4))
                s.plot(ax=ax)
                ax.set_title("Production Output Over Time")
                ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(
                    fig,
                    "production_trend.png",
                    "Observed production volume over time",
                    0.95,
                    "production",
                )
    
        # =================================================
        # 2. DEFECT DISTRIBUTION BY MACHINE
        # =================================================
        if c.get("machine") and c.get("defects"):
            grp = (
                df[[c["machine"], c["defects"]]]
                .dropna()
                .groupby(c["machine"])[c["defects"]]
                .sum()
            )
    
            if grp.nunique() > 1:
                fig, ax = plt.subplots(figsize=(7, 4))
                grp.sort_values(ascending=False).head(6).plot(
                    kind="bar", ax=ax
                )
                ax.set_title("Defect Distribution by Machine")
                save(
                    fig,
                    "defects_by_machine.png",
                    "Distribution of recorded defects across machines",
                    0.9,
                    "quality",
                )
    
        # =================================================
        # 3. DOWNTIME CONTRIBUTION
        # =================================================
        downtime_key = (
            resolve_column(df, "downtime_reason")
            or c.get("machine")
        )
    
        if c.get("downtime") and downtime_key:
            grp = (
                df[[downtime_key, c["downtime"]]]
                .dropna()
                .groupby(downtime_key)[c["downtime"]]
                .sum()
            )
    
            if grp.nunique() > 1:
                fig, ax = plt.subplots(figsize=(7, 4))
                grp.sort_values(ascending=False).head(7).plot(
                    kind="bar", ax=ax
                )
                ax.set_title("Downtime Contribution by Source")
                save(
                    fig,
                    "downtime_contributors.png",
                    "Primary contributors to recorded downtime",
                    0.88,
                    "maintenance",
                )
    
        # =================================================
        # 4. PLAN VS ACTUAL OUTPUT (DESCRIPTIVE)
        # =================================================
        if c.get("target") and c.get("quantity"):
            planned = pd.to_numeric(df[c["target"]], errors="coerce").sum()
            actual = pd.to_numeric(df[c["quantity"]], errors="coerce").sum()
    
            if planned > 0 and actual > 0:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(
                    ["Planned", "Actual"],
                    [planned, actual],
                    color=["gray", "steelblue"],
                )
                ax.set_title("Planned vs Actual Production")
                save(
                    fig,
                    "plan_vs_actual.png",
                    "Comparison of planned and actual production volumes",
                    0.85,
                    "planning",
                )
    
        # =================================================
        # 5. OEE COMPONENTS (ONLY IF VALID)
        # =================================================
        if all(
            isinstance(kpis.get(k), (int, float))
            for k in ("availability", "performance", "yield_rate", "oee")
        ):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Availability", "Performance", "Quality"],
                [
                    kpis["availability"],
                    kpis["performance"],
                    kpis["yield_rate"],
                ],
            )
            ax.set_ylim(0, 1.05)
            ax.set_title("OEE Component Breakdown")
            save(
                fig,
                "oee_components.png",
                "Contributing components of overall equipment effectiveness",
                0.92,
                "oee",
            )
    
        # =================================================
        # 6. CYCLE TIME VARIABILITY
        # =================================================
        if c.get("cycle_time"):
            s = pd.to_numeric(df[c["cycle_time"]], errors="coerce").dropna()
    
            if s.nunique() > 4:
                fig, ax = plt.subplots(figsize=(6, 4))
                s.plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Cycle Time Distribution")
                save(
                    fig,
                    "cycle_time_distribution.png",
                    "Observed variability in production cycle times",
                    0.8,
                    "efficiency",
                )
    
        # =================================================
        # FINAL PRUNING (EXECUTIVE-SAFE)
        # =================================================
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]


    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Manufacturing Composite Insight Engine
    
        Guarantees:
        - Composite-first logic
        - No thresholds, targets, or benchmarks
        - No judgment or alert language
        - No prescriptive actions
        - Graceful degradation
        """
    
        insights: List[Dict[str, Any]] = []
    
        defect_rate = kpis.get("defect_rate")
        oee = kpis.get("oee")
        availability = kpis.get("availability")
        performance = kpis.get("performance")
        yield_rate = kpis.get("yield_rate")
        downtime = kpis.get("total_downtime_hours")
        cycle_time = kpis.get("avg_cycle_time")
    
        # =================================================
        # COMPOSITE INSIGHTS (PRIMARY)
        # =================================================
    
        # 1. Speed vs Quality Trade-off
        if isinstance(performance, (int, float)) and isinstance(defect_rate, (int, float)):
            insights.append({
                "type": "composite",
                "title": "Production Speed and Quality Relationship",
                "so_what": (
                    "Observed production performance and defect patterns together describe "
                    "how throughput and quality outcomes interact within the current process design."
                )
            })
    
        # 2. Availability Influence on Overall Effectiveness
        if isinstance(availability, (int, float)) and isinstance(oee, (int, float)):
            insights.append({
                "type": "composite",
                "title": "Equipment Availability Influence",
                "so_what": (
                    "Availability levels and overall equipment effectiveness jointly indicate "
                    "how equipment uptime shapes realized production capability."
                )
            })
    
        # 3. Yield Contribution to Output Effectiveness
        if isinstance(yield_rate, (int, float)) and isinstance(oee, (int, float)):
            insights.append({
                "type": "composite",
                "title": "Yield Contribution to Production Outcomes",
                "so_what": (
                    "Observed yield rates contribute directly to overall production effectiveness, "
                    "highlighting the role of quality losses in total output realization."
                )
            })
    
        # 4. Downtime and Capacity Utilization
        if isinstance(downtime, (int, float)) and isinstance(availability, (int, float)):
            insights.append({
                "type": "composite",
                "title": "Downtime and Capacity Utilization Pattern",
                "so_what": (
                    "Recorded downtime levels and availability measures together describe "
                    "how interruptions affect usable production capacity over time."
                )
            })
    
        # 5. Process Stability Signal
        if isinstance(cycle_time, (int, float)) and isinstance(performance, (int, float)):
            insights.append({
                "type": "composite",
                "title": "Process Stability and Throughput Pattern",
                "so_what": (
                    "Cycle time behavior, when viewed alongside throughput performance, "
                    "provides insight into the consistency of the underlying production process."
                )
            })
    
        # =================================================
        # ATOMIC FALLBACK (ONLY IF NO COMPOSITES)
        # =================================================
        if not insights:
            insights.append({
                "type": "atomic",
                "title": "Manufacturing Process Overview Available",
                "so_what": (
                    "Available data supports a descriptive overview of production, quality, "
                    "and equipment utilization without strong composite signal confidence."
                )
            })
    
        return insights


    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        *_,
    ) -> List[Dict[str, Any]]:
        """
        Manufacturing Advisory Recommendation Engine
    
        Guarantees:
        - Advisory-only (no mandates or urgency)
        - No thresholds, targets, or benchmarks
        - No insight-name coupling
        - No operational prescriptions
        - Graceful degradation
        """
    
        recommendations: List[Dict[str, Any]] = []
    
        defect_rate = kpis.get("defect_rate")
        availability = kpis.get("availability")
        performance = kpis.get("performance")
        yield_rate = kpis.get("yield_rate")
        downtime = kpis.get("total_downtime_hours")
        cycle_time = kpis.get("avg_cycle_time")
        oee = kpis.get("oee")
    
        # =================================================
        # QUALITY & YIELD CONTEXT
        # =================================================
        if defect_rate is not None or yield_rate is not None:
            recommendations.append({
                "theme": "Quality & Yield",
                "advice": (
                    "Observed defect and yield patterns may be useful context when reviewing "
                    "process consistency, material behavior, or inspection coverage across production lines."
                )
            })
    
        # =================================================
        # EQUIPMENT AVAILABILITY & DOWNTIME
        # =================================================
        if availability is not None or downtime is not None:
            recommendations.append({
                "theme": "Equipment Availability",
                "advice": (
                    "Availability and downtime signals can inform discussions around equipment reliability, "
                    "maintenance planning, and capacity buffering strategies."
                )
            })
    
        # =================================================
        # THROUGHPUT & PROCESS STABILITY
        # =================================================
        if performance is not None or cycle_time is not None:
            recommendations.append({
                "theme": "Throughput & Process Stability",
                "advice": (
                    "Throughput performance and cycle time behavior together provide insight into "
                    "process stability and may support future workflow or sequencing evaluations."
                )
            })
    
        # =================================================
        # OVERALL PRODUCTION EFFECTIVENESS
        # =================================================
        if oee is not None:
            recommendations.append({
                "theme": "Production Effectiveness",
                "advice": (
                    "Overall production effectiveness can serve as a high-level reference point "
                    "when aligning quality, availability, and throughput considerations."
                )
            })
    
        # =================================================
        # GRACEFUL FALLBACK
        # =================================================
        if not recommendations:
            recommendations.append({
                "theme": "Manufacturing Overview",
                "advice": (
                    "Current data supports a descriptive overview of manufacturing performance. "
                    "Additional signals may enable deeper operational and reliability considerations."
                )
            })
    
        return recommendations


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class ManufacturingDomainDetector(BaseDomainDetector):
    """
    Boundary-safe Manufacturing domain detector.

    Guarantees:
    - Capability-based (not keyword spam)
    - Numeric signal validation
    - Safe against Retail, Finance, and Supply Chain collision
    """

    domain_name = "manufacturing"

    # -------------------------------------------------
    # SEMANTIC TOKEN GROUPS
    # -------------------------------------------------
    PRODUCTION_TOKENS = {
        "production", "produced", "output", "yield", "assembly"
    }

    EQUIPMENT_TOKENS = {
        "machine", "equipment", "line", "workcenter", "asset"
    }

    TIME_LOSS_TOKENS = {
        "downtime", "uptime", "runtime", "cycle_time", "run_time"
    }

    QUALITY_TOKENS = {
        "defect", "scrap", "reject", "rework", "quality"
    }

    EXCLUDED_TOKENS = {
        "price", "revenue", "order", "customer", "sales"
    }

    # -------------------------------------------------
    # DETECTION LOGIC
    # -------------------------------------------------
    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:

        # ---------------- SAFETY ----------------
        if df is None or df.empty:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals={}
            )

        cols = [str(c).lower() for c in df.columns]

        def tokenize(col: str) -> set:
            return set(col.replace("_", " ").split())

        tokenized = {c: tokenize(c) for c in cols}

        prod_hits = {
            c for c, t in tokenized.items()
            if t & self.PRODUCTION_TOKENS
        }

        equip_hits = {
            c for c, t in tokenized.items()
            if t & self.EQUIPMENT_TOKENS
        }

        time_hits = {
            c for c, t in tokenized.items()
            if t & self.TIME_LOSS_TOKENS
        }

        quality_hits = {
            c for c, t in tokenized.items()
            if t & self.QUALITY_TOKENS
        }

        excluded_hits = {
            c for c, t in tokenized.items()
            if t & self.EXCLUDED_TOKENS
        }

        # ---------------- NUMERIC VALIDATION ----------------
        numeric_cols = {
            c for c in (prod_hits | time_hits | quality_hits)
            if pd.api.types.is_numeric_dtype(df[c])
        }

        # ---------------- SIGNAL GROUPING ----------------
        signal_groups = sum([
            bool(prod_hits),
            bool(equip_hits),
            bool(time_hits),
            bool(quality_hits),
        ])

        # ---------------- CONFIDENCE ----------------
        if not numeric_cols or signal_groups < 2:
            confidence = 0.0
        else:
            confidence = min(0.25 + 0.18 * signal_groups, 0.9)

        # Suppress retail / finance leakage
        if excluded_hits and signal_groups <= 2:
            confidence *= 0.5

        confidence = round(confidence, 2)

        return DomainDetectionResult(
            domain=self.domain_name if confidence > 0 else None,
            confidence=confidence,
            signals={
                "production_columns": sorted(prod_hits),
                "equipment_columns": sorted(equip_hits),
                "time_columns": sorted(time_hits),
                "quality_columns": sorted(quality_hits),
                "numeric_signal_columns": sorted(numeric_cols),
                "excluded_columns": sorted(excluded_hits),
            }
        )

def register(registry):
    registry.register(
        "manufacturing",
        ManufacturingDomain,
        ManufacturingDomainDetector
    )
