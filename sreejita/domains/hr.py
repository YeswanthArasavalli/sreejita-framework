import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # governance: non-interactive backend
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Set, Optional

from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# GOVERNED NUMERIC HELPERS
# =====================================================

def safe_div(n: Any, d: Any) -> Optional[float]:
    """
    Governance-safe division.
    Returns None for zero, NaN, inf, or invalid inputs.
    """
    try:
        if n is None or d is None:
            return None
        if pd.isna(n) or pd.isna(d):
            return None
        if float(d) == 0.0:
            return None
        val = float(n) / float(d)
        if np.isinf(val) or np.isnan(val):
            return None
        return val
    except Exception:
        return None


def safe_mean(series: pd.Series) -> Optional[float]:
    """
    Safe mean with graceful degradation.
    """
    if series is None or series.empty:
        return None
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def safe_rate(numerator: Any, denominator: Any) -> Optional[float]:
    """
    Alias for safe_div — semantic clarity for HR rates.
    """
    return safe_div(numerator, denominator)


# =====================================================
# TIME & DATE HELPERS (HR-SAFE)
# =====================================================

HR_TIME_KEYWORDS = [
    "hire",
    "joining",
    "start",
    "onboard",
    "exit",
    "termination",
    "resign",
    "separation",
    "leave",
    "end",
    "date"
]


def _is_datetime_series(series: pd.Series) -> bool:
    """
    Verifies whether a column can reliably be interpreted as datetime.
    """
    try:
        parsed = pd.to_datetime(series.dropna().iloc[:10], errors="coerce")
        return parsed.notna().sum() >= 3
    except Exception:
        return False


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Boundary-safe HR time column detector.

    Design guarantees:
    - Prefers employment lifecycle dates (hire, exit, etc.)
    - Rejects ambiguous numeric-only columns
    - Returns None if confidence is weak
    """
    if df is None or df.empty:
        return None

    candidates: List[str] = []

    for col in df.columns:
        col_l = str(col).lower()
        if any(k in col_l for k in HR_TIME_KEYWORDS):
            if _is_datetime_series(df[col]):
                candidates.append(col)

    # Prefer hire/join/start over generic date
    priority = ["hire", "joining", "start", "onboard"]
    for p in priority:
        for c in candidates:
            if p in c.lower():
                return c

    return candidates[0] if candidates else None


def coerce_datetime(df: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
    """
    Safely coerces a column to datetime.
    Returns None if coercion fails.
    """
    if col is None or col not in df.columns:
        return None
    try:
        series = pd.to_datetime(df[col], errors="coerce")
        if series.notna().sum() < 3:
            return None
        return series
    except Exception:
        return None

# =====================================================
# HR DOMAIN (UNIVERSAL 10/10)
# =====================================================

class HRDomain(BaseDomain):
    """
    HR / Workforce Intelligence Domain
    Scope: workforce structure, stability, capacity, and people-related signals
    """

    name = "hr"
    description = "Workforce structure, stability, and people-related operational signals"

    # -------------------------------------------------
    # PREPROCESS (CENTRALIZED, GOVERNED STATE)
    # -------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Governance guarantees:
        - No domain assumptions
        - No row reordering
        - No raw data mutation
        - Graceful degradation
        """

        # -------- Time Detection (Boundary-Safe) --------
        self.time_col: Optional[str] = detect_time_column(df)
        self.today: pd.Timestamp = pd.Timestamp.today()

        if self.time_col:
            self._time_series = coerce_datetime(df, self.time_col)
        else:
            self._time_series = None

        # -------- Column Resolution (Lazy & Guarded) --------
        def _res(col: Optional[str]) -> Optional[str]:
            return col if col in df.columns else None

        self.cols: Dict[str, Optional[str]] = {
            "employee_id": _res(resolve_column(df, "employee_id") or resolve_column(df, "employee")),
            "department": _res(resolve_column(df, "department") or resolve_column(df, "team")),
            "salary": _res(resolve_column(df, "salary") or resolve_column(df, "compensation")),
            "gender": _res(resolve_column(df, "gender") or resolve_column(df, "sex")),
            "performance_rating": _res(resolve_column(df, "rating") or resolve_column(df, "performance")),
            "hire_date": _res(resolve_column(df, "hire_date") or resolve_column(df, "joining_date")),
            "exit_date": _res(resolve_column(df, "exit_date") or resolve_column(df, "termination_date")),
            "employment_status": _res(resolve_column(df, "status") or resolve_column(df, "active_status")),
            "absence_days": _res(resolve_column(df, "absence") or resolve_column(df, "leave_days")),
            "tenure": _res(resolve_column(df, "tenure") or resolve_column(df, "years_of_service")),
            "revenue_proxy": _res(resolve_column(df, "revenue"))
        }

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
        Execute the hr pipeline with stable, safe outputs.

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

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        HR / Workforce KPI Engine (Foundation Layer)
    
        Guarantees:
        - No thresholds, targets, or benchmarks
        - No judgment language
        - No sensitive or regulated inference
        - No dataframe mutation
        - Graceful degradation
        """
    
        kpis: Dict[str, Any] = {}
        c = self.cols
    
        # ==================================================
        # WORKFORCE SIZE & STRUCTURE
        # ==================================================
    
        if c.get("employee_id"):
            kpis["headcount"] = df[c["employee_id"]].nunique()
        else:
            kpis["headcount"] = len(df)
    
        # ==================================================
        # ATTRITION SIGNALS (NON-INFERENTIAL)
        # ==================================================
    
        if c.get("exit_date"):
            exits = pd.to_datetime(df[c["exit_date"]], errors="coerce").notna().sum()
    
            base = (
                df[c["employee_id"]].nunique()
                if c.get("employee_id")
                else len(df)
            )
    
            kpis["exit_event_count"] = exits
            kpis["exit_event_rate"] = safe_rate(exits, base)
    
        elif c.get("employment_status"):
            status = df[c["employment_status"]].astype(str).str.lower()
            exit_events = status.str.contains("exit|left|resign|term", na=False)
    
            kpis["exit_event_count"] = int(exit_events.sum())
            kpis["exit_event_rate"] = safe_rate(exit_events.sum(), len(df))
    
        # ==================================================
        # TENURE SIGNALS
        # ==================================================
    
        if c.get("hire_date"):
            hire = pd.to_datetime(df[c["hire_date"]], errors="coerce")
    
            if c.get("exit_date"):
                exit_ = pd.to_datetime(df[c["exit_date"]], errors="coerce")
                tenure_days = (exit_ - hire).dt.days
            else:
                tenure_days = (self.today - hire).dt.days
    
            tenure_days = tenure_days.dropna()
    
            kpis["avg_tenure_days"] = safe_mean(tenure_days)
            kpis["median_tenure_days"] = (
                float(tenure_days.median()) if not tenure_days.empty else None
            )
    
        elif c.get("tenure"):
            tenure = pd.to_numeric(df[c["tenure"]], errors="coerce").dropna()
            kpis["avg_reported_tenure"] = safe_mean(tenure)
    
        # ==================================================
        # COMPENSATION SIGNALS (DESCRIPTIVE ONLY)
        # ==================================================
    
        if c.get("salary"):
            salary = pd.to_numeric(
                df[c["salary"]].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
                errors="coerce"
            ).dropna()
    
            kpis["avg_salary"] = safe_mean(salary)
            kpis["median_salary"] = (
                float(salary.median()) if not salary.empty else None
            )
    
        # ==================================================
        # PRODUCTIVITY PROXIES (NON-JUDGMENTAL)
        # ==================================================
    
        if c.get("revenue_proxy") and c.get("employee_id"):
            total_revenue = pd.to_numeric(
                df[c["revenue_proxy"]], errors="coerce"
            ).sum()
    
            hc = df[c["employee_id"]].nunique()
    
            kpis["revenue_per_employee"] = safe_div(total_revenue, hc)
    
        # ==================================================
        # PERFORMANCE SIGNALS (DESCRIPTIVE ONLY)
        # ==================================================
    
        if c.get("performance_rating"):
            perf = pd.to_numeric(
                df[c["performance_rating"]], errors="coerce"
            ).dropna()
    
            kpis["avg_performance_rating"] = safe_mean(perf)
            kpis["performance_rating_dispersion"] = (
                float(perf.std()) if perf.size > 1 else None
            )
    
        # ==================================================
        # ABSENCE SIGNALS
        # ==================================================
    
        if c.get("absence_days"):
            absence = pd.to_numeric(
                df[c["absence_days"]], errors="coerce"
            ).dropna()
    
            kpis["avg_absence_days"] = safe_mean(absence)
    
        return kpis

    # ---------------- VISUALS (SMART SELECTION) ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        """
        HR Visual Engine
    
        Guarantees:
        - No thresholds or outcome-based importance
        - No sensitive or regulated inference
        - No dataframe mutation
        - Executive-safe language
        - Visual hygiene (many → few)
        """
    
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
    
        def save(fig, name, caption, importance, category):
            path = output_dir / name
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            visuals.append({
                "path": str(path),
                "caption": caption,
                "importance": importance,
                "category": category
            })
    
        # ==================================================
        # WORKFORCE STRUCTURE
        # ==================================================
    
        if c.get("department"):
            counts = df[c["department"]].value_counts()
            if counts.nunique() > 2:
                fig, ax = plt.subplots(figsize=(7, 4))
                counts.head(8).plot(kind="bar", ax=ax)
                ax.set_title("Workforce Distribution by Unit")
                save(fig, "headcount_by_unit.png",
                     "Distribution of workforce across organizational units",
                     0.9, "workforce")
    
        # ==================================================
        # EXIT EVENT DISTRIBUTION
        # ==================================================
    
        if c.get("department") and (c.get("exit_date") or c.get("employment_status")):
            if c.get("exit_date"):
                exit_mask = pd.to_datetime(df[c["exit_date"]], errors="coerce").notna()
            else:
                status = df[c["employment_status"]].astype(str).str.lower()
                exit_mask = status.str.contains("exit|left|resign|term", na=False)
    
            exit_mask = exit_mask.fillna(False)
            exit_df = df.loc[exit_mask]
    
            if not exit_df.empty:
                counts = exit_df[c["department"]].value_counts()
                if counts.nunique() > 1:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    counts.head(6).plot(kind="bar", ax=ax)
                    ax.set_title("Recorded Exit Events by Unit")
                    save(fig, "exit_events_by_unit.png",
                         "Observed distribution of recorded exit events",
                         0.85, "retention")
    
        # ==================================================
        # TENURE DISTRIBUTION
        # ==================================================
    
        if c.get("hire_date"):
            hire = pd.to_datetime(df[c["hire_date"]], errors="coerce")
            tenure_days = (self.today - hire).dt.days.dropna()
    
            if tenure_days.nunique() > 3:
                fig, ax = plt.subplots(figsize=(6, 4))
                tenure_days.hist(ax=ax, bins=10)
                ax.set_title("Tenure Distribution (Days)")
                save(fig, "tenure_distribution.png",
                     "Observed distribution of workforce tenure",
                     0.8, "stability")
    
        # ==================================================
        # COMPENSATION DISTRIBUTION
        # ==================================================
    
        if c.get("salary"):
            salary = pd.to_numeric(
                df[c["salary"]].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
                errors="coerce"
            ).dropna()
    
            if salary.nunique() > 5:
                fig, ax = plt.subplots(figsize=(6, 4))
                salary.hist(ax=ax, bins=15)
                ax.set_title("Compensation Distribution")
                save(fig, "salary_distribution.png",
                     "Observed distribution of reported compensation values",
                     0.75, "compensation")
    
        # ==================================================
        # PERFORMANCE SIGNAL DISTRIBUTION
        # ==================================================
    
        if c.get("performance_rating"):
            perf = pd.to_numeric(df[c["performance_rating"]], errors="coerce").dropna()
            if perf.nunique() > 3:
                fig, ax = plt.subplots(figsize=(6, 4))
                perf.hist(ax=ax, bins=6)
                ax.set_title("Performance Rating Distribution")
                save(fig, "performance_distribution.png",
                     "Distribution of recorded performance ratings",
                     0.7, "performance")
    
        # ==================================================
        # PRODUCTIVITY PROXY (STRUCTURAL)
        # ==================================================
    
        if c.get("revenue_proxy") and c.get("employee_id"):
            revenue = pd.to_numeric(df[c["revenue_proxy"]], errors="coerce").sum()
            hc = df[c["employee_id"]].nunique()
    
            if hc > 0 and revenue > 0:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(["Revenue per Employee"], [revenue / hc])
                ax.set_title("Revenue per Employee")
                save(fig, "revenue_per_employee.png",
                     "Observed revenue scaled by workforce size",
                     0.85, "productivity")
    
        # ==================================================
        # FINAL TRIM: MANY → FEW
        # ==================================================
    
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]

    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        HR / Workforce Composite Insight Engine
    
        Guarantees:
        - Composite-first logic
        - No thresholds, targets, or benchmarks
        - No judgment or alert language
        - No sensitive or regulated inference
        - Graceful degradation
        """
    
        insights: List[Dict[str, Any]] = []
    
        headcount = kpis.get("headcount")
        exit_rate = kpis.get("exit_event_rate")
        avg_tenure = kpis.get("avg_tenure_days")
        median_tenure = kpis.get("median_tenure_days")
        avg_absence = kpis.get("avg_absence_days")
        revenue_per_emp = kpis.get("revenue_per_employee")
        perf_dispersion = kpis.get("performance_rating_dispersion")
    
        # ==================================================
        # COMPOSITE INSIGHTS (PRIMARY)
        # ==================================================
    
        # 1. Workforce Stability Signal
        if exit_rate is not None and avg_tenure is not None:
            insights.append({
                "type": "composite",
                "title": "Workforce Stability Pattern",
                "so_what": (
                    "Observed exit activity and tenure patterns together describe the "
                    "current workforce stability profile, indicating how long employees "
                    "tend to remain relative to recorded exits."
                )
            })
    
        # 2. Experience Depth Signal
        if avg_tenure is not None and median_tenure is not None:
            insights.append({
                "type": "composite",
                "title": "Experience Distribution Shape",
                "so_what": (
                    "Average and median tenure together outline the balance between "
                    "long-tenured and recently joined employees, shaping institutional knowledge depth."
                )
            })
    
        # 3. Capacity Utilization Signal
        if avg_absence is not None and headcount is not None:
            insights.append({
                "type": "composite",
                "title": "Workforce Capacity Utilization",
                "so_what": (
                    "Reported absence patterns, when viewed alongside workforce size, "
                    "provide insight into how available capacity fluctuates over time."
                )
            })
    
        # 4. Productivity Context Signal
        if revenue_per_emp is not None and headcount is not None:
            insights.append({
                "type": "composite",
                "title": "Productivity Context",
                "so_what": (
                    "Revenue scaled by workforce size offers contextual insight into how "
                    "organizational output relates to current staffing levels."
                )
            })
    
        # 5. Performance Consistency Signal
        if perf_dispersion is not None:
            insights.append({
                "type": "composite",
                "title": "Performance Rating Consistency",
                "so_what": (
                    "Variation in recorded performance ratings reflects how evenly performance "
                    "outcomes are distributed across the workforce."
                )
            })
    
        # ==================================================
        # ATOMIC INSIGHTS (FALLBACK ONLY)
        # ==================================================
    
        if not insights:
            insights.append({
                "type": "atomic",
                "title": "Workforce Overview Available",
                "so_what": (
                    "Available data supports a descriptive overview of workforce structure "
                    "without strong composite signal confidence."
                )
            })
    
        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        HR / Workforce Advisory Recommendation Engine
    
        Guarantees:
        - Advisory-only (no mandates or urgency labels)
        - No thresholds, targets, or benchmarks
        - No sensitive or regulated actions
        - Signal-aware but insight-decoupled
        - Graceful degradation
        """
    
        recommendations: List[Dict[str, Any]] = []
    
        exit_rate = kpis.get("exit_event_rate")
        avg_tenure = kpis.get("avg_tenure_days")
        avg_absence = kpis.get("avg_absence_days")
        revenue_per_emp = kpis.get("revenue_per_employee")
        perf_dispersion = kpis.get("performance_rating_dispersion")
    
        # ==================================================
        # WORKFORCE STABILITY CONSIDERATIONS
        # ==================================================
    
        if exit_rate is not None or avg_tenure is not None:
            recommendations.append({
                "theme": "Workforce Stability",
                "advice": (
                    "Consider reviewing workforce stability patterns over time, including "
                    "how tenure distribution and recorded exit events align with current operating needs."
                )
            })
    
        # ==================================================
        # CAPACITY & UTILIZATION CONSIDERATIONS
        # ==================================================
    
        if avg_absence is not None:
            recommendations.append({
                "theme": "Capacity Utilization",
                "advice": (
                    "Observed absence patterns may be useful input when evaluating workload balance, "
                    "resourcing flexibility, or support mechanisms across teams."
                )
            })
    
        # ==================================================
        # PRODUCTIVITY CONTEXT CONSIDERATIONS
        # ==================================================
    
        if revenue_per_emp is not None:
            recommendations.append({
                "theme": "Productivity Context",
                "advice": (
                    "Revenue scaled by workforce size can serve as contextual input when discussing "
                    "staffing models, role design, or operational efficiency initiatives."
                )
            })
    
        # ==================================================
        # PERFORMANCE SIGNAL CONSIDERATIONS
        # ==================================================
    
        if perf_dispersion is not None:
            recommendations.append({
                "theme": "Performance Distribution",
                "advice": (
                    "Variation in recorded performance ratings may warrant reflection on role clarity, "
                    "goal alignment, or feedback consistency across the organization."
                )
            })
    
        # ==================================================
        # GRACEFUL FALLBACK
        # ==================================================
    
        if not recommendations:
            recommendations.append({
                "theme": "Workforce Overview",
                "advice": (
                    "Current data supports a descriptive overview of workforce structure. "
                    "Additional signals may enable deeper workforce planning considerations."
                )
            })
    
        return recommendations

# =====================================================
# DOMAIN DETECTOR (WITH CONFIDENCE FLOOR FIX)
# =====================================================

class HRDomainDetector(BaseDomainDetector):
    """
    Boundary-safe HR / Workforce domain detector
    """

    domain_name = "hr"

    # HR-exclusive semantic anchors
    CORE_TOKENS = {
        "employee",
        "employee_id",
        "headcount",
        "hire",
        "joining",
        "termination",
        "exit",
        "attrition",
        "tenure",
        "compensation",
        "salary",
        "payroll",
        "absence",
        "leave",
        "performance"
    }

    # Supportive but non-exclusive signals
    CONTEXT_TOKENS = {
        "department",
        "team",
        "role",
        "designation",
        "manager",
        "status"
    }

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        if df is None or df.empty:
            return DomainDetectionResult(self.domain_name, 0.0, {"matched_columns": []})

        cols = [str(c).lower() for c in df.columns]

        core_hits = [
            c for c in cols
            if any(t in c for t in self.CORE_TOKENS)
        ]

        context_hits = [
            c for c in cols
            if any(t in c for t in self.CONTEXT_TOKENS)
        ]

        # ---------------------------------------------
        # Boundary-Safe Confidence Logic
        # ---------------------------------------------

        unique_core = set(core_hits)
        unique_context = set(context_hits)

        # Minimum signal requirement
        if len(unique_core) < 2:
            return DomainDetectionResult(
                self.domain_name,
                0.0,
                {"matched_columns": list(unique_core | unique_context)}
            )

        # Confidence emerges from signal richness
        signal_strength = (
            len(unique_core) * 0.6 +
            len(unique_context) * 0.4
        )

        confidence = min(signal_strength / 4.0, 1.0)

        return DomainDetectionResult(
            self.domain_name,
            round(confidence, 2),
            {
                "matched_columns": list(unique_core | unique_context),
                "core_signal_count": len(unique_core),
                "context_signal_count": len(unique_context)
            }
        )


def register(registry):
    registry.register("hr", HRDomain, HRDomainDetector)
